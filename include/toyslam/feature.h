/**
 * @file feature.h
 * @author Tianze Li (tianze.li.eu@gmail.com)
 * @brief Feature class to store the detected feature in a single image.
 * @version 0.1
 * @date 2022-04-11
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef TOYSLAM_FEATURE_H
#define TOYSLAM_FEATURE_H

#include "toyslam/common_include.h"

namespace toyslam {

class Feature { 
typedef std::shared_ptr<Feature> Ptr;

public:
  // Pixel location in the image.
  Eigen::Matrix<double, 2, 1> uv;
  // Descriptor of the feature.
  cv::KeyPoint kp;
  // If the depth is available.
  bool has_depth;
  // 3D spatial coordinate w.r.t the camera center.
  Eigen::Matrix<double, 3, 1> xyz;

  // Constructors.
  Feature() = default;
  Feature(const cv::KeyPoint &keypoint) : kp(keypoint) {}
};

} // namespace toyslam

#endif // TOYSLAM_FEATURE_H
