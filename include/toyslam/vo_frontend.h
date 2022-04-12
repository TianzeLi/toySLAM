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

  DataStereo::Ptr data = nullptr;

  // Pointer to the current frame
  Frame::Ptr frame_current_ = nullptr;
  Frame::Ptr frame_previous_ = nullptr;
  
  std::vector<Feature> detectAndMatch(cv::Mat &img1, cv::Mat &img2);
  
  //  trianglation(std::vector<cv::KeyPoint> k1,
  //                                        std::vector<cv::KeyPoint> k2,
  //                                        Camera::Ptr c1, 
  //                                        Camera::Ptr c2);
  // Frame::Ptr estimateTransform(Frame& frame);
};

} // namespace toyslam
#endif // TOYSLAM_VO_FRONTEND_H