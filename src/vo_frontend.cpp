#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Dense>
#include <boost/format.hpp>
#include <random>
#include <chrono> 

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
    LOG(INFO)<< "Now processing the frame no." << frame_current_->id;
    
    // Detect and match left and right image from the same frame.
    VLOG(2)<< "Detecting and matching left and right image from the same frame." ;
    frame_current_->features_left = detectAndMatchLR(frame_current_->img_left, 
                                                      frame_current_->img_right);
    // Omit the rest for the first frame.  
    if (frame_current_->id == 0) {
      frame_previous_ = frame_current_;
      frame_current_ = data->nextFrame();
      continue;
    }
    
    // Detect and match left images from two consecutive frames.
    std::vector<cv::DMatch> matches_two_frames;
    VLOG(2)<< "Detecting and matching left images from two consecutive frames." ;
    matches_two_frames = MatchTwoFrames(frame_current_->img_left, 
                                        frame_previous_->img_left,
                                        frame_current_->features_left, 
                                        frame_previous_->features_left);
    
    // Estimate the pose of the frame.
    VLOG(2)<< "Estimating the transfrom between two frames." ;
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
  VLOG(4) << "Max dist in ORB matching (left and right image): " << max_dist;
  VLOG(4) << "Min dist in ORB matching (left and right image): " << min_dist;
  // Reject those threefold the size of the min distance. 
  // Use 30 as an emphrical value in case the min distance is too small.
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < dpts1.rows; i++) {
    if (matches[i].distance <= std::max(3 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }
  VLOG(2) << "Obatined " << matches.size() << " matches (left and right).";
  VLOG(1) << "Obatined " << good_matches.size() << " better matches (left and right).";

  // Generate the features. 
  // Right now only creat features for left image.
  std::vector<Feature> v;
  for (cv::DMatch m : good_matches) { 
    int i1 = m.queryIdx;
    int i2 = m.trainIdx;
    Feature f(kps1[i1]);
    f.uv << kps1[i1].pt.x, kps1[i1].pt.y;
    double relative_error = 1.0;
    TrianglationResult res = VOFront::triangulate(kps1[i1], kps2[i2], 
                                 data->getCamera(0), data->getCamera(1));
    f.xyz = res.XYZ;
    // Display the matched pair for debugging.
    if ( diaplay_single_match_ )
      VOFront::displaySingleMatch(img1, img2, kps1[i1], kps2[i2]);
    // Check the reprojection of the trianglated point.
    if (do_triangulation_rejection_ 
        && (res.relative_error > triangulate_error_threshold_)){
        VLOG(2) << "Rejected a feature due to large reprojection error.";
        continue;
    }
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


TrianglationResult VOFront::triangulate(cv::KeyPoint &k1, 
                                        cv::KeyPoint &k2,
                                        const Camera::Ptr &c1, 
                                        const Camera::Ptr &c2){
  Eigen::Matrix<double, 3, 1> XYZ;
  Eigen::Matrix<double, 3, 1> uv1;
  Eigen::Matrix<double, 3, 1> uv2;
  uv1 << k1.pt.x, k1.pt.y, 1;
  uv2 << k2.pt.x, k2.pt.y, 1;
  VLOG(4) << "Pixel location in left image: " << uv1.transpose();
  VLOG(4) << "Pixel location in right image: " << uv2.transpose();

  // Not retrieving K from camera pose since in Kitti the relation is simple.
  // In Kitti, R = I, t = [0.54, 0, 0] (camera2 w.r.t camera1)
  Eigen::Matrix<double, 3, 1> t21;
  t21 << 0.54, 0.0, 0.0;  
  // Formulate the equation.
  Eigen::Matrix<double, 3, 2> A;
  A << c1->K_inv * uv1, -1 * c2->K_inv * uv2;
  VLOG(4) << "A = \n" << A;
  // Solve the least square error, QR method from Eigen.
  Eigen::Matrix<double, 2, 1>  s = A.colPivHouseholderQr().solve(t21);
  double relative_error = (A*s - t21).norm() / t21.norm(); // norm() is L2 norm
  // Recover XYZ.
  XYZ = s(0,0) * c1->K_inv * uv1;
  if ( s(0,0) < 0 || s(1,0) < 0 ){
    VLOG(1) << "Depth estimated as negative.";
    relative_error = 1.0;
  }  
  VLOG(3) << "\nFeature point obtained w.r.t the left camera frame: " << XYZ.transpose();
  VLOG(3) << "The relative error in triangulation: " << relative_error;
  VLOG(3) << "The reprojection error: " <<(uv1 - 1/s(0,0)*c1->K*XYZ).norm();
  
  return TrianglationResult{XYZ, relative_error};
}


// Match the keypoints in two frames.
std::vector<cv::DMatch> VOFront::MatchTwoFrames(cv::Mat &img1, 
                                                cv::Mat &img2,
                                                std::vector<Feature> &features_curr, 
                                                std::vector<Feature> &features_prev){  
  std::vector<cv::KeyPoint> kps1, kps2;
  for (Feature f : features_curr){ kps1.push_back(f.kp); }
  for (Feature f : features_prev){ kps2.push_back(f.kp); }
  // Compute BRIEF descriptors.
  cv::Mat dpts1, dpts2;
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
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
  VLOG(4) << "Max dist in ORB matching (current and previous frame): " << max_dist;
  VLOG(4) << "Min dist in ORB matching (current and previous frame): " << min_dist;
  // Reject those triple the size of the min distance. 
  // Use 30 as an emphrical value in case the min distance is too small.
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < dpts1.rows; i++) {
    if (matches[i].distance <= std::max(3 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }
  VLOG(2) << "Obatined " << matches.size() << " matches (two frames).";
  VLOG(1) << "Obatined " << good_matches.size() << " better matches (two frames).";

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
  Sophus::SE3d T_est;
  // Create the matched point pair lists.
  std::vector<Eigen::Matrix<double, 2, 1>> u_list;
  std::vector<Eigen::Matrix<double, 3, 1>> P_list;

  for (cv::DMatch m : matches) { 
    int i1 = m.queryIdx;
    int i2 = m.trainIdx;
    // Paired feature points' 3D coordinate in prev. frame and 2D in current image.
    u_list.push_back(frame_curr->features_left[i1].uv); 
    P_list.push_back(frame_prev->features_left[i2].xyz);
    // To test the algorithm, we can feed in the XYZ of uv1 itself 
    // and the result should be the identity matrix.
    // P_list.push_back(frame_curr->features_left[i1].xyz);

    VLOG(4) << "uv in current frame: " << frame_curr->features_left[i1].uv.transpose();
    VLOG(4) << "uv in previous frame: " << frame_prev->features_left[i2].uv.transpose();
    VLOG(4) << "xyz in previous frame: " << frame_prev->features_left[i2].xyz.transpose();
  }
  // Without RANSAC
  T_est = Gauss_Newton(u_list, P_list, c_curr);
  VLOG(1) << "(Without RANSAC) Estimated transfrom between last two frames: \n " << T_est.inverse().matrix();
  
  // RANSAC
  if (do_RANSAC_){
    double threshold = sin(reprojection_angle_threshold_);
    int count = 0;
    std::vector<Sophus::SE3d> T_list;
    std::vector<int> count_list;
    for (int i = 0; i <= RANSAC_iteration_times_; ++i ){
      std::vector<Eigen::Matrix<double, 2, 1>> u_list_est;
      std::vector<Eigen::Matrix<double, 3, 1>> P_list_est;
      std::vector<int> k_list;
      // Randomly select s pairs of correspondencs.
      std::random_device rd; // obtain a random number from hardware
      std::mt19937 gen(rd()); // seed the generator
      std::uniform_int_distribution<> distr(0, u_list.size()-1); // define the range
      for (int j = 0; j < amount_pairs_; ++j){
        int k = distr(gen);
        // To avoid drawing the same pair.
        if( std::find(k_list.begin(), k_list.end(), k) == k_list.end() ){
          // VLOG(1) << "Got random number k: " << k;
          k_list.push_back(k);
          u_list_est.push_back(u_list.at(k));
          P_list_est.push_back(P_list.at(k));
        }
        else {
          --j;
          continue;  
        }
      }
      // VLOG(1) << "u_list length to draw estimation: " << u_list_est.size();
      // VLOG(1) << "P_list length to draw estimation: " << P_list_est.size();

      // Estimate the transform based on selected pairs.
      auto T_tmp = Gauss_Newton(u_list_est, P_list_est, c_curr);
      // Iterate throught the other pairs to see how many carries small enough error.
      count = 0;
      for (int j = 0; j < u_list.size(); ++j ) {
        if( std::find(k_list.begin(), k_list.end(), j) == k_list.end() ) {
          double err = computeReprojectionAngleError(u_list.at(j), P_list.at(j), 
                                                     T_tmp.matrix3x4(), c_curr->K_inv);
        VLOG(1) << "Reprojection angle error " << err;
        if ( abs(err) < threshold ) ++count;
      }
      // Store the result of one iteration.
      T_list.push_back(T_tmp);
      count_list.push_back(count);
      }
      VLOG(1) << "Has got " << count << " inliers out of " 
              << u_list.size() - amount_pairs_;  
    }

    // Find the index i that has the most counts.
    int i_max_count = std::distance(count_list.begin(), std::max_element(count_list.begin(),count_list.end()));
    VLOG(1)<< "Estimation no. " << i_max_count << " has the most counts"
            << count_list[i_max_count] << " out of " << u_list.size() - amount_pairs_;
    T_est = T_list[i_max_count]; 
    VLOG(1)<< "(RANSAC) Estimated transfrom between last two frames: \n " << T_est.inverse().matrix();
  }

  LOG(INFO)<< "Estimated transfrom between last two frames: \n " << T_est.inverse().matrix();
  return frame_prev->pose*T_est.inverse(); 
}

Sophus::SE3d VOFront::Gauss_Newton(std::vector<Eigen::Matrix<double, 2, 1>> &u_list,
                                   std::vector<Eigen::Matrix<double, 3, 1>> &P_list,
                                   const Camera::Ptr &c_curr){

  // Terminate criterion:
  // 1 - Converge; 2 - Reached max iter. times; 3 - Cost function incresing (N/A yet).                                              
  int iteration_to_terminate = 0;
  // Prepare the camera intrinsic parameters.
  double fx = c_curr->K(0, 0);
  double fy = c_curr->K(1, 1);
  double cx = c_curr->K(0, 2);
  double cy = c_curr->K(1, 2);
  double u1, u2, X, Y, Z_inv;
  Eigen::Matrix<double, 6, 1> epsilon = Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> epsilon_tmp = Eigen::Matrix<double, 6, 1>::Zero();
  // The transfrom from previous to current: P_curr = T_est * P_prev
  Sophus::SE3d T_est;
  Eigen::Matrix<double, 2, 6> delta_transpose;
  Eigen::Matrix<double, 2, 1> beta;  
  int iter_no = 0;

  do{
    assert( u_list.size() == P_list.size() );
    ++iter_no;
    VLOG(3) << "Gauss-Newton iteration no." << iter_no;
    VLOG(3) << "Initial transfrom matrix input: \n" << T_est.matrix();
    Eigen::Matrix<double, 6, 6> sum_delta_deltaT = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> sum_delta_beta = Eigen::Matrix<double, 6, 1>::Zero();
    double sum_reprojection_error = 0.0;
    for ( int i = 0; i < u_list.size(); ++i ){
      auto TP = T_est*P_list[i];
      // Prepare u and P.
      X = TP(0, 0);
      Y = TP(1, 0);
      Z_inv = 1.0 / TP(2, 0);
      u1 = u_list[i](0, 0);
      u2 = u_list[i](1, 0);
      // Calculate delta_m and beta_m.
      // delta_transpose: 6*2 matrix
      delta_transpose << -fx*Z_inv, 0.0, fx*X*Z_inv*Z_inv,
                         fx*X*Y*Z_inv*Z_inv, -fx-fx*X*X*Z_inv*Z_inv, fx*Y*Z_inv,
                         0.0, -fy*Z_inv, fy*Y*Z_inv,
                         fy+fy*Y*Y*Z_inv*Z_inv, -fy*X*Y*Z_inv*Z_inv, -fy*X*Z_inv;
      // beta: 2*1 matrix
      beta << u1-fx*X*Z_inv-cx, u2-fy*Y*Z_inv-cy;
      sum_delta_deltaT += delta_transpose.transpose()*delta_transpose;
      sum_delta_beta += -delta_transpose.transpose()*beta;
      sum_reprojection_error += beta.norm()*beta.norm();
    }
    VLOG(3) << "Total reprojection error before this iteration: " << sum_reprojection_error;
    VLOG(3) << "Average reprojection error for each pair before this iteration: " 
              << sum_reprojection_error/u_list.size();

    // Solve the equation in the least square sense.
    epsilon = sum_delta_deltaT.colPivHouseholderQr().solve(sum_delta_beta);
    double relative_error = (sum_delta_deltaT*epsilon - sum_delta_beta).norm() 
                            / sum_delta_beta.norm(); // norm() is L2 norm
    T_est = Sophus::SE3d::exp(epsilon)*T_est;
    VLOG(3) << "Epsilon (rho phi) in current iteration " << epsilon.transpose();
    VLOG(3) << "Transfrom in current iteration\n " << T_est.matrix();
    VLOG(4) << "Relative error in QR decompostion: " << relative_error;
    sum_reprojection_error = 0.0;
    for ( int i = 0; i < u_list.size(); ++i ){
        auto TP = T_est*P_list[i];
        // Prepare u and P.
        X = TP(0, 0);
        Y = TP(1, 0);
        Z_inv = 1.0 / TP(2, 0);
        u1 = u_list[i](0, 0);
        u2 = u_list[i](1, 0);
        // Calculate delta_m and beta_m.
        beta << u1-fx*X*Z_inv-cx, u2-fy*Y*Z_inv-cy;
        sum_reprojection_error += beta.norm()*beta.norm();
    }
    VLOG(3) << "Total reprojection error after this iteration: " << sum_reprojection_error;
    VLOG(3) << "Average reprojection error for each pair after this iteration: " 
            << sum_reprojection_error/u_list.size();

    // Backtracking?

    // Restore T from epsilon and iterate.
    if ( (epsilon - epsilon_tmp).norm() < epsilon_mag_threshold_) iteration_to_terminate = 1;
    if ( iter_no > GN_iteration_times_max_) iteration_to_terminate = 2;
  } while( iteration_to_terminate == 0);
  
  switch(iteration_to_terminate) {
    case 1 : VLOG(2)<< "Iteration over due to CONVERGENCE"; break;
    case 2 : VLOG(2)<< "Iteration over due to MAXIMAL ITERATION TIMES REACHED";
  }
  return T_est;
}

double VOFront::computeReprojectionAngleError(Eigen::Matrix<double, 2, 1> u, 
                                              Eigen::Matrix<double, 3, 1> P, 
                                              Eigen::Matrix<double, 3, 4> T, 
                                              Eigen::Matrix<double, 3, 3> K_inv){
  Eigen::Matrix<double, 3, 1> u_homogenerous;
  u_homogenerous << u(0,0), u(1,0), 1.0;
  Eigen::Matrix<double, 4, 1> P_homogenerous; 
  P_homogenerous << P(0,0), P(1,0), P(2,0), 1.0;
  Eigen::Matrix<double, 3, 1> t;
  t << T(0,3), T(1,3), T(2,3);

  Eigen::Matrix<double, 3, 1> vec1 = K_inv * u_homogenerous;
  Eigen::Matrix<double, 3, 1> vec2 = t.cross(T*P_homogenerous);

  return vec1.dot(vec2)/vec1.norm()/vec2.norm(); 
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