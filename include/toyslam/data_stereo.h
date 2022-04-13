/**
 * @file data_stereo.h
 * @author Tianze Li (you@tianze.li.eu@gmail.com)
 * @brief Load the dataset, in particular, Kitti stereo set.
 * @version 0.1
 * @date 2022-03-16
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef TOYSLAM_DATA_STEREO_H
#define TOYSLAM_DATA_STEREO_H

#include "toyslam/common_include.h"
#include "toyslam/frame.h"
#include "toyslam/camera.h"

namespace toyslam {

class DataStereo {
public:
  typedef std::shared_ptr<DataStereo> Ptr;

  bool init();
  DataStereo(const std::string& path);
  // To obtain the pointer to the next frame.
  Frame::Ptr nextFrame();
  // To get the camera by its No. in the camera pointer vector.
  Camera::Ptr getCamera(int id) const { return cameras_.at(id); }

private:
  // The path of the dataset. 
  std::string dataset_path_;
  // Image resize ratio. 
  double resize_ratio_ = 1.0;

  int current_image_index_ = 0;
  std::vector<Camera::Ptr> cameras_;
};

} // namespace toyslam
#endif // TOYSLAM_DATA_STEREO_H