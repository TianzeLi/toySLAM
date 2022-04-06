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

  // Stereo camera baseline length.
  double baseline = 0;
  // Camera intrinsic matrix.
  Eigen::Matrix<double, 3, 3> K;

  Camera() = default;
  Camera(double fx, double fy, double cx, double cy, double bsl)
    : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline(bsl) {
    K << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0;
  }
  Camera(Eigen::Matrix<double, 3, 3> K_in, double bsl) : K(K_in), baseline(bsl) {}
private:
  // Pinhole model intrinsic parameters.
  double fx_ = 0.0, fy_ = 0.0;
  double cx_ = 0.0, cy_ = 0.0;

};

} // namespace toyslam
#endif // TOYSLAM_CAMERA_H
