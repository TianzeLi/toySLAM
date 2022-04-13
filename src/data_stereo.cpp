#include "toyslam/data_stereo.h"
#include <boost/format.hpp>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;

namespace toyslam {

DataStereo::DataStereo(const std::string& path) : dataset_path_(path) {}

bool DataStereo::init() {
  ifstream fin(dataset_path_ + "/calib.txt");
  if (!fin) {
    LOG(ERROR) << "cannot find " << dataset_path_ << "calib.txt.";
    return false;
  }
  // Already know the info of 4 cameras are available in Kitti.
  // The camera 0 and 1 supply the gray images used in this project.
  // The orientation of the cameras are the same in Kitti dataset.
  for (int i = 0; i < 4; i++) {
    char camera_name[3];
    for (int k = 0; k < 3; k++) {
      fin >> camera_name[k];
    }
    double camera_parameter[12];
    for (int k = 0; k < 12; ++k) {
      fin >> camera_parameter[k];
    }
    Eigen::Matrix<double, 3, 3> K;
    K << camera_parameter[0], camera_parameter[1], camera_parameter[2],
         camera_parameter[4], camera_parameter[5], camera_parameter[6],
         camera_parameter[8], camera_parameter[9], camera_parameter[10];
    Eigen::Matrix<double, 3, 1> t;
    t << camera_parameter[3], camera_parameter[7], camera_parameter[11];
    t = K.inverse() * t;
    K = K * resize_ratio_;
    // Append the camera to the vector.
    Camera::Ptr new_camera(new Camera(i, K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                      t.norm(), Sophus::SE3d(Sophus::SO3d(), t)));
    cameras_.push_back(new_camera);
    LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
  }

  fin.close();
  current_image_index_ = 0;
  return true;
}

Frame::Ptr DataStereo::nextFrame() {
  boost::format fmt("%s/image_%d/%06d.png");
  cv::Mat image_left, image_right;
  image_left = cv::imread((fmt % dataset_path_ % 0 % current_image_index_).str(),
                          cv::IMREAD_GRAYSCALE);
  image_right = cv::imread((fmt % dataset_path_ % 1 % current_image_index_).str(),
                          cv::IMREAD_GRAYSCALE);
  if ( (!image_left.empty()) && (!image_right.empty()) ) {// Check for invalid input
    auto new_frame = Frame::CreateFrame();
    LOG(INFO) << "A new frame No." << current_image_index_ <<" is created.";
    cv::resize(image_left, image_left, cv::Size(), resize_ratio_, resize_ratio_);
    cv::resize(image_right, image_right, cv::Size(), resize_ratio_, resize_ratio_);
    new_frame->img_left = image_left;
    new_frame->img_right = image_right;
    current_image_index_++;
    return new_frame;
  }
  else {
    LOG(ERROR) << "No imaged loaded."; 
    return nullptr;
  } 
}

} // namespace: toyslam