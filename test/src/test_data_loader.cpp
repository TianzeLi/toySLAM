/**
 * @file test_data_loader.cpp
 * @author Tianze Li (tianze.li.eu@gmail.com)
 * @brief Test the data loading class, Kitti stereo dataset for now.
 * @version 0.1
 * @date 2022-03-29
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#include <opencv2/opencv.hpp>
#include "toyslam/common_include.h"
#include "toyslam/data_stereo.h"
#include "toyslam/frame.h"
#include "toyslam/camera.h"

using namespace std;

// Print the intrinsic matrix and display the left and right images one by one.
int main(int argc, char* argv[]) {
  // google::InitGoogleLogging(argv[0]);
  
  string dataset_path = "/home/tianze/Downloads/KittiBenchmark/data_odometry_gray/dataset/sequences/00";
  toyslam::DataStereo data(dataset_path);
  if (data.init())
    LOG(INFO) << "Data loaded. ";
  else 
    LOG(ERROR) << "Data failed to load. ";

  // Print the intrinsic matrix of camera 0 and 1.
  for (int i = 0; i < 2; i++) {
    toyslam::Camera::Ptr camera = data.getCamera(i);
    LOG(INFO) << "The stereo camera info." << endl 
              << "Baseline length (m): " << camera->baseline << endl
              << "Camera intrinsic matrix: " 
              << endl << camera->K;
  }
  
  // Display the stream according to its timestamp.
  auto frame = data.nextFrame();
  while (frame != nullptr){
    auto img_left = frame->img_left;
    auto img_right = frame->img_right;
    if( !img_left.empty() ){
      cv::imshow( "Left image", img_left);
      cv::imshow( "Right image", img_right);
      // cv::waitKey(0);
    }                      // Check for invalid input
    else
      LOG(ERROR) <<  "Left image empty.";
    frame = data.nextFrame();
  }
  
  LOG(INFO) << "Frame empty.";
  
  return 0;
}