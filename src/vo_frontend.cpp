#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>

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
    frame_current_->features_left = detectAndMatch(frame_current_->img_left, 
                                                   frame_current_->img_right);
    
    // Detect and match left images from two consective frames.
    // detectAndMatch(frame_current_->img_left, frame_previous_->img_left);
    
    // Estimate the pose of the frame.
    // frame_current_.pose =  estimateTransform(frame_current_, frame_previous_);
    // vo_front.pose = frame_current_.pose;

    frame_previous_ = frame_current_;
    frame_current_ = data->nextFrame();
  }

  return 0;
}

// Sophus::SE3d updatePose();
// {

// }

std::vector<Feature> VOFront::detectAndMatch(cv::Mat &img1, cv::Mat &img2) {
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
  VLOG(2) << "Obatined " << matches.size() << " matches.";
  VLOG(2) << "Obatined " << good_matches.size() << " good matches.";

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
                                                 const Camera::Ptr &c2) {
  Eigen::Matrix<double, 3, 1> XYZ;
  Eigen::Matrix<double, 3, 1> uv1;
  Eigen::Matrix<double, 3, 3> uv1_mat;
  Eigen::Matrix<double, 3, 3> uv2_mat;
  uv1 << k1.pt.x, k1.pt.y, 1;
  // Put into the dot product form.
  uv1_mat << 0.0, -1.0, k1.pt.y,
             1.0, 0.0, -k1.pt.x,
             -k1.pt.y, -k1.pt.x, 0.0;
  uv2_mat << 0.0, -1.0, k2.pt.y,
             1.0, 0.0, -k2.pt.x,
             -k2.pt.y, -k2.pt.x, 0.0;
  // In Kitti, R = I, t = [0, 0.54, 0]
  // R21 = 
  Eigen::Matrix<double, 3, 1> t21;
  t21 << 0.0, 0.54, 0.0; // -0.54??  
  
  VLOG(4) << "uv1_mat: \n" << uv1_mat;
  VLOG(4) << "uv2_mat: \n" << uv2_mat;
  VLOG(4) << "c1->K_inv: \n" << c1->K_inv;
  VLOG(4) << "c2->K_inv: \n" << c2->K_inv;

  Eigen::Matrix<double, 3, 1> A;
  Eigen::Matrix<double, 3, 1> b;
  A = c2->K_inv * uv1_mat * c1->K_inv * uv1;
  b = -1 * c2->K_inv * uv1_mat * t21;

  // Solve the least square error, QR method from Eigen.
  Eigen::Matrix<double, 1, 1>  s1 = A.colPivHouseholderQr().solve(b);
  double relative_error = (A*s1 - b).norm() / b.norm(); // norm() is L2 norm
  // Recover XYZ.
  double s1_scalar = s1(0,0);
  XYZ = s1_scalar * c1->K_inv * uv1;
  VLOG(3) << "Feature point obtained w.r.t the left camera frame: " << XYZ;
  VLOG(3) << "The relative error in triangulation: " << relative_error;
  
  return XYZ;
}


// Frame::Ptr VOFront::estimateTransform(Frame& frame) {

// }

} // namespace toyslam