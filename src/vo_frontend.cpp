#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include <boost/format.hpp>


#include "toyslam/vo_frontend.h"

namespace toyslam {

bool VOFront::init(){
  data = DataStereo::Ptr(new DataStereo(dataset_path_));
  if (data->init())
    LOG(INFO) << "Data loaded. ";
  else 
    LOG(ERROR) << "Data failed to load. ";
  
  // Obtain the first frame.
  frame_current_ = data->nextFrame();
  frame_previous_ = frame_current_; 
  assert(frame_current_ != nullptr);
  
  return true;
}

int VOFront::run(){
  LOG(INFO) << " VO frontend now start to process the first frame. ";

  while (frame_current_ != nullptr) {
    // Detect and match left and right image from the same frame.
    frame_current_->features_left = detectAndMatchLR(frame_current_->img_left, 
                                                     frame_current_->img_right);
    // Go to the next loop. 
    if (frame_current_->id == 0) {
      frame_previous_ = frame_current_;
      frame_current_ = data->nextFrame();
      continue;
    }
    std::vector<cv::DMatch> matches_two_frames;
    // Detect and match left images from two consecutive frames.
    matches_two_frames = MatchTwoFrames(frame_current_->img_left, 
                                        frame_previous_->img_left,
                                        frame_current_->features_left, 
                                        frame_previous_->features_left);
    
    // // Estimate the pose of the frame.
    // frame_current_->pose =  estimateTransform(frame_current_, 
    //                                           frame_previous_,
    //                                           matches_two_frames);
    pose = frame_current_->pose;

    frame_previous_ = frame_current_;
    frame_current_ = data->nextFrame();
  }

  return 0;
}

// Sophus::SE3d updatePose();
// {

// }

std::vector<Feature> VOFront::detectAndMatchLR(cv::Mat &img1, cv::Mat &img2) {
  assert(img1.data != nullptr && img2.data != nullptr);
  // Initialize ORB detector.
  std::vector<cv::KeyPoint> kps1, kps2;
  cv::Mat dpts1, dpts2;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  
  // Detect Oriented FAST corner points.
  detector->detect(img1, kps1);
  detector->detect(img2, kps2);
  
  // Compute BRIEF descriptors.
  descriptor->compute(img1, kps1, dpts1);
  descriptor->compute(img2, kps2, dpts2);
  
  // Match the keypoints.
  std::vector<cv::DMatch> matches;
  matcher->match(dpts1, dpts2, matches);

  // Check the matched point by its distance to the min distance detected.
  auto min_max = std::minmax_element(matches.begin(), matches.end(),
                                [](const cv::DMatch &m1, const cv::DMatch &m2) 
                                { return m1.distance < m2.distance; });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;
  VLOG(2) << "-- Max dist : " << max_dist;
  VLOG(2) << "-- Min dist : " << min_dist;

  // Reject those twice the size of the min distance. 
  // Use 30 as an emphrical value in case the min distance is too small.
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < dpts1.rows; i++) {
    if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }
  VLOG(2) << "Obatined " << matches.size() << " matches (left and right).";
  VLOG(2) << "Obatined " << good_matches.size() << " good matches (left and right).";

  // Generate the features.
  std::vector<Feature> v;
  for (cv::DMatch m : good_matches) { 
    int i1 = m.queryIdx;
    int i2 = m.trainIdx;
    Feature f(kps1[i1]);
    f.xyz = VOFront::triangulate(kps1[i1], kps2[i2], data->getCamera(0), data->getCamera(1));
    v.push_back(f);
  }

  // Display the matching result. 
  cv::Mat img_match;
  cv::Mat img_goodmatch;
  cv::drawMatches(img1, kps1, img2, kps2, matches, img_match);
  cv::drawMatches(img1, kps1, img2, kps2, good_matches, img_goodmatch);
  cv::resize(img_match, img_match, cv::Size(), 0.75, 0.75);
  cv::resize(img_goodmatch, img_goodmatch, cv::Size(), 0.75, 0.75);
  cv::imshow("All matches", img_match);
  cv::imshow("Good matches", img_goodmatch);
  cv::waitKey(0);

  return v;
}


Eigen::Matrix<double, 3, 1> VOFront::triangulate(cv::KeyPoint &k1, 
                                                 cv::KeyPoint &k2,
                                                 const Camera::Ptr &c1, 
                                                 const Camera::Ptr &c2){
  Eigen::Matrix<double, 3, 1> XYZ;
  Eigen::Matrix<double, 3, 1> uv1;
  Eigen::Matrix<double, 3, 1> uv2;
  uv1 << k1.pt.x, k1.pt.y, 1;
  uv2 << k2.pt.x, k2.pt.y, 1;

  // In Kitti, R = I, t = [0, 0.54, 0]
  // R21 = 
  Eigen::Matrix<double, 3, 1> t21;
  t21 << 0.54, 0.0, 0.0;  
  
  VLOG(5) << "uv1_mat: \n" << uv1;
  VLOG(5) << "uv2_mat: \n" << uv2;
  VLOG(5) << "c1->K_inv: \n" << c1->K_inv;
  VLOG(5) << "c2->K_inv: \n" << c2->K_inv;

  Eigen::Matrix<double, 3, 2> A;
  A << c1->K_inv * uv1, -1 * c2->K_inv * uv2;
  VLOG(4) << "A = \n" << A;

  // Solve the least square error, QR method from Eigen.
  Eigen::Matrix<double, 2, 1>  s = A.colPivHouseholderQr().solve(t21);
  double relative_error = (A*s - t21).norm() / t21.norm(); // norm() is L2 norm
  // Recover XYZ.
  double s1_scalar = s(0,0);
  double s2_scalar = s(1,0);

  if ( s1_scalar < 0 ) LOG(WARNING) << "Depth in left image estimated as negative.";
  if ( s2_scalar < 0 ) LOG(WARNING) << "Depth in right image estimated as negative.";

  XYZ = s1_scalar * c1->K_inv * uv1;
  VLOG(3) << "\nFeature point obtained w.r.t the left camera frame: " << XYZ;
  VLOG(3) << "The relative error in triangulation: " << relative_error;
  
  return XYZ;
}


// Match the keypoints in two frames.
std::vector<cv::DMatch> VOFront::MatchTwoFrames(cv::Mat &img1, 
                                                cv::Mat &img2,
                                                std::vector<Feature> &features_curr, 
                                                std::vector<Feature> &features_prev){  
  std::vector<cv::KeyPoint> kps1, kps2;
  for (Feature f : features_curr){ kps1.push_back(f.kp); }
  for (Feature f : features_prev){ kps2.push_back(f.kp); }
  
  cv::Mat dpts1, dpts2;
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  
  // Compute BRIEF descriptors.
  descriptor->compute(img1, kps1, dpts1);
  descriptor->compute(img2, kps2, dpts2);
  // Match the keypoints.
  std::vector<cv::DMatch> matches;
  matcher->match(dpts1, dpts2, matches);

  // Check the matched point by its distance to the min distance detected.
  auto min_max = std::minmax_element(matches.begin(), matches.end(),
                                [](const cv::DMatch &m1, const cv::DMatch &m2) 
                                { return m1.distance < m2.distance; });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;
  VLOG(2) << "-- Max dist : " << max_dist;
  VLOG(2) << "-- Min dist : " << min_dist;

  // Reject those twice the size of the min distance. 
  // Use 30 as an emphrical value in case the min distance is too small.
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < dpts1.rows; i++) {
    if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }
  
  VLOG(2) << "Obatined " << matches.size() << " matches (two frames).";
  VLOG(2) << "Obatined " << good_matches.size() << " good matches (two frames).";

  // Display the matching result. 
  cv::Mat img_match;
  cv::Mat img_goodmatch;
  cv::drawMatches(img1, kps1, img2, kps2, matches, img_match);
  cv::drawMatches(img1, kps1, img2, kps2, good_matches, img_goodmatch);
  cv::resize(img_match, img_match, cv::Size(), 0.75, 0.75);
  cv::resize(img_goodmatch, img_goodmatch, cv::Size(), 0.75, 0.75);
  cv::imshow("All matches between frames", img_match);
  cv::imshow("Good matches between frames", img_goodmatch);
  cv::waitKey(0);
  
  return good_matches;
  }

// Estiamte the transform from corresponding features.
Sophus::SE3d estimateTransform(Frame::Ptr &frame_curr, 
                               Frame::Ptr &frame_prev,
                               std::vector<cv::DMatch> matches){
  Sophus::SE3d transfrom_est;

  std::vector<Feature> v;
  for (cv::DMatch m : matches) { 
    int i1 = m.queryIdx;
    int i2 = m.trainIdx;
    
    // Paired feature points' 3D coordinate in prev. frame and 2D in current image.
    frame_curr->features_left[i1].uv, frame_prev->features_left[i2].xyz;
  }

  return transfrom_est*frame_prev->pose; // Need to check.
}

} // namespace toyslam