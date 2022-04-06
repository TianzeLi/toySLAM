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

namespace toyslam {

class Frame {
public:
  typedef std::shared_ptr<Frame> Ptr;

  unsigned long id;
  double time_stamp;
  Sophus::SE3d pose;              // pose in the homogeneous matrix
  cv::Mat img_left, img_right;    // stereo images

  Frame() = default;
  Frame(unsigned long i, double t, Sophus::SE3d p, 
        cv::Mat& imgl, cv::Mat& imgr): id(i), time_stamp(t){
        pose = p;
        img_left = imgl;
        img_right = imgr;
  }

  static std::shared_ptr<Frame> CreateFrame();
};

Frame::Ptr Frame::CreateFrame() {
    static long factory_id = 0;
    Frame::Ptr new_frame(new Frame);
    new_frame->id = factory_id++;
    return new_frame;
}

} //namespace toyslam
#endif //TOYSLAM_FRAME_H