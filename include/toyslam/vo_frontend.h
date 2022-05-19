/**
 * @file vo_frontend.h
 * @author Tianze Li (tianze.li.eu@gmail.com)
 * @brief Visual odometry front end pipeline,
 *        that is, without bundle adjustment. 
 * @version 0.1
 * @date 2022-04-06
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef TOYSLAM_VO_FRONTEND_H
#define TOYSLAM_VO_FRONTEND_H

#include "toyslam/common_include.h"
#include "toyslam/frame.h"
#include "toyslam/camera.h"
#include "toyslam/data_stereo.h"
#include "toyslam/feature.h"
#include "toyslam/vo_ba.h"

namespace toyslam{

struct TrianglationResult {Eigen::Matrix<double, 3, 1> XYZ; double relative_error;};


class VOFront {
public:
  VOFront() = default;
  VOFront(DataStereo::Ptr dataset) : data(dataset) {};
  
  // Initialize the frontend.
  bool init();
  // Run the frontend.
  int run();

  // Register the frontend with BA.
  void resigterBA(VOBA::Ptr ba) { bundle_adjustment_ = ba; };

  // Parameter setting interfaces for configuration. 
  void set_do_RANSAC(std::string s) { 
    std::istringstream(s) >> std::boolalpha >> do_RANSAC_ ; 
    }
  bool get_do_RANSAC() { return do_RANSAC_ ; }

  void set_RANSAC_iteration_times(int n) { RANSAC_iteration_times_ =  n; }
  void set_amount_pairs(int n) { RANSAC_amount_pairs_ = n; }

  void set_do_triangulation_rejection(std::string s) 
    { std::istringstream(s) >> std::boolalpha >> do_triangulation_rejection_;}
  void set_triangulate_error_threshold(double t) { triangulate_error_threshold_ = t; }
  void set_epsilon_mag_threshold(double t) { epsilon_mag_threshold_ = t; }
  
  void set_GN_iteration_times_max(int n) { GN_iteration_times_max_ = n; }
  void set_reprojection_angle_threshold(double t) { reprojection_angle_threshold_ = t; }
  
  void set_write_est_to_file(std::string s) 
    { std::istringstream(s) >> std::boolalpha >> write_est_to_file_ ; }
  void set_outfile_path(std::string s) { outfile_path = s; }
  
  void set_show_left_and_right_matches(std::string s) 
    { std::istringstream(s) >> std::boolalpha >> show_left_and_right_matches_ ; }
  void set_show_prev_and_curr_matches(std::string s) 
    { std::istringstream(s) >> std::boolalpha >> show_prev_and_curr_matches_; }
  void set_diaplay_single_match(std::string s) 
    { std::istringstream(s) >> std::boolalpha >> diaplay_single_match_ ; }


  // The estimated pose of the left camera center.
  Sophus::SE3d pose;

private:
  // Configuration file path.
  std::string config_file_path_;
  // State of the VO frontend.
  int state_ = 0;
  // Triangulation. 
  // If true, use triangulation reprojection error to reject mismatches.
  bool do_triangulation_rejection_ = true;
  // Triangulation reprojection error threshold.
  double triangulate_error_threshold_ = 0.15;
  // Gauss-Newton
  // Gauss-Newton optimization iteration threshold. 
  double epsilon_mag_threshold_ = 0.001; 
  // Gauss-Newton optimization maximal iteration times. 
  double GN_iteration_times_max_ = 60; 
  // RANSAC
  // If true, do RANSAC during the estimation.
  bool do_RANSAC_ = true;
  // RANSAC iteration times, equations in VO Tutorial.
  int RANSAC_iteration_times_ = 20;
  // RANSAC amounts of pairs for estimation in each iteration.
  int RANSAC_amount_pairs_ = 7;
  // RANSAC threshold for the angle between epipolar plane and reprojected arrow. 
  double reprojection_angle_threshold_ = 0.001; 
  // Bundle adjustment.
  bool do_bundle_adjustment_ = false;
  // If true, write estimation to an output file.
  bool write_est_to_file_ = true;
  std::string outfile_path = "../bin/est_tmp.txt";

  // Display images settings.
  bool show_left_and_right_matches_ = false;
  bool show_prev_and_curr_matches_ = false;
  bool diaplay_single_match_ = false;

  // Pointer to the dataset.
  DataStereo::Ptr data = nullptr;

  // Pointer to the bundle adjustment.
  VOBA::Ptr bundle_adjustment_;

  // Pointer to the current frame
  Frame::Ptr frame_current_ = nullptr;
  Frame::Ptr frame_previous_ = nullptr;

  // Detect and match left and right image from the same frame.
  std::vector<Feature> detectAndMatchLR(cv::Mat &img1, cv::Mat &img2);

  // Triangulate the matched point from two images and compute the XYZ w.r.t frame 1.
  TrianglationResult triangulate(cv::KeyPoint &k1, 
                                 cv::KeyPoint &k2,
                                 const Camera::Ptr &c1, 
                                 const Camera::Ptr &c2);
  
  // Match the keypoints in two frames.
  std::vector<cv::DMatch> MatchTwoFrames(cv::Mat &img1, 
                                         cv::Mat &img2,
                                         std::vector<Feature> &features_curr, 
                                         std::vector<Feature> &features_prev);
  
  // Compute the transform from the previous frame to the current frame.
  Sophus::SE3d estimateTransformPnP(Frame::Ptr &frame_curr, 
                                    Frame::Ptr &frame_prev,
                                    std::vector<cv::DMatch> match_two_frames,
                                    const Camera::Ptr &c_curr);
  
  // Gauss-Newton iteration on P-n-P.
  Sophus::SE3d GaussNewton(std::vector<Eigen::Matrix<double, 2, 1>> &u_list,
                           std::vector<Eigen::Matrix<double, 3, 1>> &P_list,
                           const Camera::Ptr &c_curr);

  double computeReprojectionAngleError(Eigen::Matrix<double, 2, 1> u, 
                                       Eigen::Matrix<double, 3, 1> P, 
                                       Eigen::Matrix<double, 3, 4> T, 
                                       Eigen::Matrix<double, 3, 3> K_inv);

  // Display the matched pair for debugging.
  void displaySingleMatch(cv::Mat &img1, 
                          cv::Mat &img2,
                          cv::KeyPoint &k1, 
                          cv::KeyPoint &k2);

  // Register a frame with BA.
  void registerFrame2BA(Frame::Ptr frame);
};

} // namespace toyslam

#endif // TOYSLAM_VO_FRONTEND_H