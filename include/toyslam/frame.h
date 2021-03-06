/**
 * @file frame.h
 * @author Tianze Li (tianze.li.eu@gmail.com)
 * @brief Frame class for stereo camera.
 * @version 0.1
 * @date 2022-04-02
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef TOYSLAM_FRAME_H
#define TOYSLAM_FRAME_H

#include "toyslam/common_include.h"
#include "toyslam/feature.h"

namespace toyslam {

class Frame {
public:
  typedef std::shared_ptr<Frame> Ptr;

  unsigned long id;
  double time_stamp;
  // Pose in the homogeneous matrix. P_frame = T_frame * P_absolute. 
  Sophus::SE3d pose;              
  // Left and right images from the stereo camera.
  cv::Mat img_left, img_right;    
  // Features in the left and right image.
  std::vector<Feature> features_left;
  std::vector<Feature> features_right;
  
  // Constructors 
  Frame() = default;
  Frame(unsigned long i, double t, Sophus::SE3d p, 
        cv::Mat& imgl, cv::Mat& imgr): id(i), time_stamp(t){
        pose = p;
        img_left = imgl;
        img_right = imgr;
  }
  // Obtain the next frame.
  static std::shared_ptr<Frame> CreateFrame(){
    static long factory_id = 0;
    Frame::Ptr new_frame(new Frame);
    new_frame->id = factory_id++;
    return new_frame;
  }
};

} //namespace toyslam
#endif //TOYSLAM_FRAME_H