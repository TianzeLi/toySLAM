/**
 * @file vo_ba.cpp
 * @author Tianze Li (tianze.li.eu@gmail.com)
 * @brief Bundle adjustment for VO based on g2o.
 * @version 0.1
 * @date 2022-05-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef TOYSLAM_VO_BA_H
#define TOYSLAM_VO_BA_H

#include "toyslam/common_include.h"
#include "toyslam/frame.h"
#include "toyslam/feature.h"

namespace toyslam {

class VOBA {
public:
  typedef std::shared_ptr<VOBA> Ptr;
  
  // Frames to process.
  std::vector<Frame::Ptr> frames_;
  
  // Constructors.
  VOBA() = default;
  VOBA(unsigned n) : bundle_size(n) {};
  void appendFrame(Frame::Ptr f) { frames_.push_back(f); }

private:
  // The amount of frames in each bundle to process.
  unsigned bundle_size = 5;

  
  // Bundle adjustment.
  void bundleAdjust(std::vector<Frame::Ptr>);




};

} // namespace toyslam
#endif // TOYSLAM_VO_BA_H