#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "toyslam/vo_frontend.h"

namespace toyslam {

bool VOFront::init(){
  data = DataStereo::Ptr(new DataStereo(dataset_path_));
  if (data->init())
    LOG(INFO) << "Data loaded. ";
  else 
    LOG(ERROR) << "Data failed to load. ";

  // Obtain the camera intrinsic matrix.
  Camera::Ptr camera = data->getCamera();
  
  // Obtain the first frame.
  frame_current_ = data->nextFrame();
  frame_previous_ = frame_current_; 
  assert(frame_current_ != nullptr);

  return true;
}

int VOFront::run(){
    while (frame_current_ != nullptr) {
      // Detect and match left and right image from the same frame.
      frame_current_->features_left = detectAndMatch(frame_current_->img_left, 
                                                    frame_current_->img_right);
      
      // Detect and match left images from two consective frames.
      // detectAndMatch(frame_current_->img_left, frame_previous_->img_left);
      
      
      frame_previous_ = frame_current_;
      frame_current_ = data->nextFrame();
    }

  return 0;
}

// Sophus::SE3d updatePose();
// {

// }

std::vector<Feature> VOFront::detectAndMatch(cv::Mat &img1, cv::Mat &img2) {
  assert(img1.data != nullptr && img2.data != nullptr);
  // Initialize ORB detector.
  std::vector<cv::KeyPoint> kps1, kps2;
  cv::Mat dpts1, dpts2;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  
  // Detect Oriented FAST corner points.
  detector->detect(img1, kps1);
  detector->detect(img2, kps2);
  
  // Compute BRIEF descriptors.
  descriptor->compute(img1, kps1, dpts1);
  descriptor->compute(img2, kps2, dpts2);
  
  // Match the keypoints.
  std::vector<cv::DMatch> matches;
  matcher->match(dpts1, dpts2, matches);

  // Check the matched point by its distance to the min distance detected.
  auto min_max = std::minmax_element(matches.begin(), matches.end(),
                                [](const cv::DMatch &m1, const cv::DMatch &m2) 
                                { 
                                  return m1.distance < m2.distance; 
                                });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;
  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  // Reject those twice the size of the min distance. 
  // Use 30 as an emphrical value in case the min distance is too small.
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < dpts1.rows; i++) {
    if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  // Display the matching result. 
  cv::Mat img_match;
  cv::Mat img_goodmatch;
  cv::drawMatches(img1, kps1, img2, kps2, matches, img_match);
  cv::drawMatches(img1, kps1, img2, kps2, good_matches, img_goodmatch);
  cv::resize(img_match, img_match, cv::Size(), 0.75, 0.75);
  cv::resize(img_goodmatch, img_goodmatch, cv::Size(), 0.75, 0.75);
  cv::imshow("All matches", img_match);
  cv::imshow("Good matches", img_goodmatch);
  cv::waitKey(0);

  std::vector<Feature> v;
  for (cv::KeyPoint kp : kps1) {
    v.push_back(Feature(kp));
  }

  return v;
}


// std::vector<cv::KeyPoint> VOFront trianglation(std::vector<cv::KeyPoint> k1,
//                                                std::vector<cv::KeyPoint> k2,
//                                                Camera::Ptr c1, 
//                                                Camera::Ptr c2){
     
//                                                }


// Frame::Ptr VOFront::estimateTransform(Frame& frame) {

// }

} // namespace toyslam