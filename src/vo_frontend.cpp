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
  LOG(INFO)<< "\n\n\nNow processing the frame no." << frame_current_->id;
    // Detect and match left and right image from the same frame.
    VLOG(1)<< "\n\n\nDetecting and matching left and right image from the same frame." ;
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
    VLOG(1)<< "\n\n\nDetecting and matching left images from two consecutive frames." ;

    matches_two_frames = MatchTwoFrames(frame_current_->img_left, 
                                        frame_previous_->img_left,
                                        frame_current_->features_left, 
                                        frame_previous_->features_left);
    
    // Estimate the pose of the frame.
    VLOG(1)<< "\n\n\nEstimating the transfrom between two frames." ;
    frame_current_->pose =  estimateTransformPnP(frame_current_, 
                                              frame_previous_,
                                              matches_two_frames,
                                              data->getCamera(0));
    pose = frame_current_->pose;
    LOG(INFO) << " VO frontend at pose: \n" << pose.matrix();

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
  VLOG(3) << "-- Max dist : " << max_dist;
  VLOG(3) << "-- Min dist : " << min_dist;

  // Reject those twice the size of the min distance. 
  // Use 30 as an emphrical value in case the min distance is too small.
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < dpts1.rows; i++) {
    if (matches[i].distance <= std::max(3 * min_dist, 30.0)) {
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
    f.uv << kps1[i1].pt.x, kps1[i1].pt.y;
    double relative_error = 1.0;
    f.xyz = VOFront::triangulate(kps1[i1], kps2[i2], 
                                 data->getCamera(0), data->getCamera(1),
                                 relative_error);
    // Display the matched pair for debugging.
    if ( diaplay_single_match_ )
      VOFront::displaySingleMatch(img1, img2, kps1[i1], kps2[i2]);
    if (do_triangulation_rejection_  && (relative_error > reprojection_threshold_))
        continue;
    else 
      v.push_back(f);
  }

  if (show_left_and_right_matches_){
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
  }

  return v;
}


Eigen::Matrix<double, 3, 1> VOFront::triangulate(cv::KeyPoint &k1, 
                                                 cv::KeyPoint &k2,
                                                 const Camera::Ptr &c1, 
                                                 const Camera::Ptr &c2,
                                                 double &relative_error){
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
  relative_error = (A*s - t21).norm() / t21.norm(); // norm() is L2 norm
  // Recover XYZ.
  double s1_scalar = s(0,0);
  double s2_scalar = s(1,0);

  if ( s1_scalar < 0 ) LOG(WARNING) << "Depth in left image estimated as negative.";
  if ( s2_scalar < 0 ) LOG(WARNING) << "Depth in right image estimated as negative.";

  XYZ = s1_scalar * c1->K_inv * uv1;
  VLOG(3) << "\nFeature point obtained w.r.t the left camera frame: " << XYZ.transpose();
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
  VLOG(3) << "-- Max dist : " << max_dist;
  VLOG(3) << "-- Min dist : " << min_dist;

  // Reject those twice the size of the min distance. 
  // Use 30 as an emphrical value in case the min distance is too small.
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < dpts1.rows; i++) {
    if (matches[i].distance <= std::max(3 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }
  VLOG(2) << "Obatined " << matches.size() << " matches (two frames).";
  VLOG(2) << "Obatined " << good_matches.size() << " good matches (two frames).";

  // Display the matching result. 
  if ( show_prev_and_curr_matches_ ){
    cv::Mat img_match;
    cv::Mat img_goodmatch;
    cv::drawMatches(img1, kps1, img2, kps2, matches, img_match);
    cv::drawMatches(img1, kps1, img2, kps2, good_matches, img_goodmatch);
    cv::resize(img_match, img_match, cv::Size(), 0.75, 0.75);
    cv::resize(img_goodmatch, img_goodmatch, cv::Size(), 0.75, 0.75);
    cv::imshow("All matches between frames", img_match);
    cv::imshow("Good matches between frames", img_goodmatch);
    cv::waitKey(0);
  }
  
  return good_matches;
  }

// Estiamte the transform from corresponding features.
Sophus::SE3d VOFront::estimateTransformPnP(Frame::Ptr &frame_curr, 
                                           Frame::Ptr &frame_prev,
                                           std::vector<cv::DMatch> matches,
                                           const Camera::Ptr &c_curr){
  // Prepare the camera intrinsic parameters.
  double fx = c_curr->K(0, 0);
  double fy = c_curr->K(1, 1);
  double cx = c_curr->K(0, 2);
  double cy = c_curr->K(1, 2);
  // Create the matched point pair lists.
  std::vector<Eigen::Matrix<double, 2, 1>> u_list;
  std::vector<Eigen::Matrix<double, 3, 1>> P_list;
  double u1, u2, X, Y, Z_inv;
  Eigen::Matrix<double, 6, 1> epsilon = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> epsilon_tmp = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> epsilon_diff = Eigen::Matrix<double, 6, 1>::Zero();

  // The transfrom from previous to current.
  Sophus::SE3d T_est;

  std::vector<Feature> v;
  int matches_num = 0;
  for (cv::DMatch m : matches) { 
    int i1 = m.queryIdx;
    int i2 = m.trainIdx;
    // Paired feature points' 3D coordinate in prev. frame and 2D in current image.
    u_list.push_back(frame_curr->features_left[i1].uv); 
    P_list.push_back(frame_prev->features_left[i2].xyz);
    VLOG(3) << "uv in current frame: " << frame_curr->features_left[i1].uv.transpose();
    VLOG(3) << "uv in previous frame: " << frame_prev->features_left[i2].uv.transpose();
    VLOG(3) << "xyz in previous frame: " << frame_prev->features_left[i2].xyz.transpose();
    ++matches_num;
  }

  Eigen::Matrix<double, 2, 6> delta_transpose;
  Eigen::Matrix<double, 2, 1> beta;  
  int iter_no = 0;

  do{
    ++iter_no;
    VLOG(2) << "\nGauss-Newton iteration no." << iter_no;
    VLOG(3) << "Now the transfrom matrix estiamted as \n" << T_est.matrix();
    Eigen::Matrix<double, 6, 6> sum_delta_deltaT = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> sum_delta_beta = Eigen::Matrix<double, 6, 1>::Zero();
    double sum_reprojection_error = 0.0;
    for ( int i = 0; i < matches_num; ++i ){
      auto TP = T_est*P_list[i];
      // Prepare u and P.
      X = TP(0, 0);
      Y = TP(1, 0);
      Z_inv = 1.0 / TP(2, 0);
      u1 = u_list[i](0, 0);
      u2 = u_list[i](0, 0);
      // Calculate delta_m and beta_m.
      // delta_transpose: 6*2 matrix
      delta_transpose << -fx*Z_inv, 0.0, fx*X*Z_inv*Z_inv,
                         fx*X*Y*Z_inv*Z_inv, -fx-fx*X*X*Z_inv*Z_inv, fx*Y*Z_inv,
                         0.0, -fy*Z_inv, fy*Y*Z_inv,
                         fy+fy*Y*Y*Z_inv*Z_inv, -fy*X*Y*Z_inv*Z_inv, -fy*X*Z_inv;
      beta << u1-fx*X*Z_inv-cx, u2-fy*Y*Z_inv-cy;
      sum_delta_deltaT += delta_transpose.transpose()*delta_transpose;
      sum_delta_beta += -delta_transpose.transpose()*beta;
      sum_reprojection_error += (beta(0,0)*beta(0,0) + beta(1,0)*beta(1,0));
    }
    VLOG(2) << "Total reprojection error: " << sum_reprojection_error;
    VLOG(2) << "Average reprojection error for each pair: " 
              << sum_reprojection_error/matches_num;

    // Solve the equation in the least square sense.
    epsilon = sum_delta_deltaT.colPivHouseholderQr().solve(sum_delta_beta);
    double relative_error = (sum_delta_deltaT*epsilon - sum_delta_beta).norm() 
                            / sum_delta_beta.norm(); // norm() is L2 norm
    VLOG(2) << "\nEpsilon (rho phi) in current iteration " << epsilon.transpose();
    VLOG(3) << "\nTransfrom in current iteration\n " << Sophus::SE3d::exp(epsilon).matrix();
    VLOG(4) << "Relative error in current reprojection: " << relative_error;
  
    // Backtracking?

    // Restore T from epsilon and iterate.
    T_est = Sophus::SE3d::exp(epsilon)*T_est;
    epsilon_diff = epsilon - epsilon_tmp;
    epsilon_tmp = epsilon;
  } while( epsilon_diff.norm() > epsilon_mag_threshold_);
  // } while ( iter_no < 10 );
  // LOG(INFO)<< "\nTransfrom:\n " << T_est.inverse().matrix();
  // LOG(INFO)<< "\nPose of the camera:\n " << T_est.inverse().matrix();
  return frame_prev->pose*T_est.inverse(); // Need to reassure.
}

// Display the matched pair for debugging.
void VOFront::displaySingleMatch(cv::Mat &img1, 
                                 cv::Mat &img2,
                                 cv::KeyPoint &k1, 
                                 cv::KeyPoint &k2){
  double draw_multiplier = 1.0;
  cv::Mat img_k1;
  cv::Mat img_k2;
  cv::Mat img_match;
  std::vector<cv::KeyPoint> kps1, kps2;
  kps1.push_back(k1);
  kps2.push_back(k2);
  cv::drawKeypoints(img1, kps1, img_k1);
  cv::drawKeypoints(img2, kps2, img_k2);
  cv::vconcat(img_k1, img_k2, img_match);
  cv::resize(img_match, img_match, cv::Size(), draw_multiplier, draw_multiplier);
  cv::Point2f pt1 = k1.pt, pt2 = k2.pt,
              dpt2 = cv::Point2f( pt2.x, pt2.y+img1.rows );

  cv::line( img_match,
            cv::Point(cvRound(pt1.x*draw_multiplier), cvRound(pt1.y*draw_multiplier)),
            cv::Point(cvRound(dpt2.x*draw_multiplier), cvRound(dpt2.y*draw_multiplier)),
            cv::Scalar(255, 128, 0), 3 );
  cv::imshow("Good matches between frames", img_match);
  cv::waitKey(0);
}

} // namespace toyslam