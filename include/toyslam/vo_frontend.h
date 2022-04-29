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

namespace toyslam{

class VOFront {
public:
  // The estimated pose of the left camera center.
  Sophus::SE3d pose;

  VOFront() = default;
  VOFront(std::string path) : dataset_path_(path) {};
  
  // Initialize the frontend.
  bool init();
  // Run the frontend.
  int run();

  Sophus::SE3d updatePose();


private:
  std::string dataset_path_;
  // State of the VO frontend.
  unsigned state_ = 0;

  // If true, use triangulation reprojection error to reject mismatches.
  bool do_triangulation_rejection_ = true;
  // Triangulation reprojection error threshold.
  double reprojection_threshold_ = 0.15;
  // Gauss-Newton optimization iteration threshold. 
  double epsilon_mag_threshold_ = 0.01; 

  // Display images settings.
  bool show_left_and_right_matches_ = true;
  bool show_prev_and_curr_matches_ = true;
  bool diaplay_single_match_ = false;

  DataStereo::Ptr data = nullptr;

  // Pointer to the current frame
  Frame::Ptr frame_current_ = nullptr;
  Frame::Ptr frame_previous_ = nullptr;

  // Detect and match left and right image from the same frame.
  std::vector<Feature> detectAndMatchLR(cv::Mat &img1, cv::Mat &img2);
  // Triangulate the matched point from two images and compute the XYZ w.r.t frame 1.
  Eigen::Matrix<double, 3, 1> triangulate(cv::KeyPoint &k1, 
                                          cv::KeyPoint &k2,
                                          const Camera::Ptr &c1, 
                                          const Camera::Ptr &c2,
                                          double &error);
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

  // Display the matched pair for debugging.
  void displaySingleMatch(cv::Mat &img1, 
                          cv::Mat &img2,
                          cv::KeyPoint &k1, 
                          cv::KeyPoint &k2);
};

} // namespace toyslam
#endif // TOYSLAM_VO_FRONTEND_H