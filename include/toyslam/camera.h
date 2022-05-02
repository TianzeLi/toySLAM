/**
 * @file camera.h
 * @author Tianze Li (tianze.li.eu@gmail.com)
 * @brief Class defined for stereo camera pinhole model.
 * @version 0.1
 * @date 2022-03-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef TOYSLAM_CAMERA_H
#define TOYSLAM_CAMERA_H

#include "toyslam/common_include.h"

namespace toyslam {

class Camera {
public:
  typedef std::shared_ptr<Camera> Ptr;

  // Camera id.
  int id = 0;
  // Stereo camera baseline length.
  double baseline = 0;
  // Camera intrinsic matrix.
  Eigen::Matrix<double, 3, 3> K;
  // Frequently used inverse matrix of K 
  Eigen::Matrix<double, 3, 3> K_inv;
  
  // Pose of the camera w.r.t to a fixed frame on the robot system.
  Sophus::SE3d pose_;

  Camera() = default;
  Camera(int i, double fx, double fy, double cx, double cy, double bsl, 
         const Sophus::SE3d &p)
    : id(i), fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline(bsl), pose_(p) {
    K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
    K_inv = K.inverse();
  }
  Camera(Eigen::Matrix<double, 3, 3> K_in, double bsl) : K(K_in), baseline(bsl) {}
private:
  // Pinhole model intrinsic parameters.
  double fx_ = 0.0, fy_ = 0.0;
  double cx_ = 0.0, cy_ = 0.0;

};

} // namespace toyslam
#endif // TOYSLAM_CAMERA_H
