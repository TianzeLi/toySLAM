/**
 * @file test_vo_frontend.cpp
 * @author Tianze Li (tianze.li.eu@gmail.com)
 * @brief Test the VO including data loading, frontend and BA.
 * @version 0.1
 * @date 2022-05-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "toyslam/vo_frontend.h"
#include "toyslam/vo_ba.h"
#include "toyslam/config.h"
#include "toyslam/evaluate_trajectory.h"

using namespace toyslam;

int main(int argc, char* argv[]) {
  std::string config_file_path = "../config/default.yaml";
  Config::SetParameterFile(config_file_path);

  std::string dataset_path = Config::Get<std::string>("dataset_dir");
  std::string groundtruth_file = Config::Get<std::string>("groundtruth_file_path");
  std::string estimated_file = Config::Get<std::string>("outfile_path");
  std::string do_evaluation_str = Config::Get<std::string>("do_evaluation");
  bool do_evaluation = false; 
  std::istringstream(do_evaluation_str) >> std::boolalpha >> do_evaluation;
  
  // Prepare the dataset.
  auto data = DataStereo::Ptr(new DataStereo(dataset_path));
  if (data->init())
    LOG(INFO) << "Data loaded. ";
  else 
    LOG(ERROR) << "Data failed to load. "; 
  int bundle_size = 7;
  
  // Initialize the BA.
  auto vo_ba = VOBA::Ptr(new VOBA(bundle_size));

  // Initialize the VO frontend.  
  VOFront vo_front(data);
  vo_front.set_do_RANSAC(Config::Get<std::string>("do_RANSAC"));
  LOG(INFO) << vo_front.get_do_RANSAC();

  vo_front.set_outfile_path(Config::Get<std::string>("outfile_path"));
  vo_front.set_do_triangulation_rejection
    (Config::Get<std::string>("do_triangulation_rejection"));
  vo_front.set_triangulate_error_threshold
    (Config::Get<double>("triangulate_error_threshold"));
  vo_front.set_epsilon_mag_threshold(Config::Get<double>("epsilon_mag_threshold"));
  vo_front.set_GN_iteration_times_max(Config::Get<int>("GN_iteration_times_max"));
  vo_front.set_RANSAC_iteration_times
    (Config::Get<int>("RANSAC_iteration_times"));
  vo_front.set_amount_pairs(Config::Get<int>("amount_pairs"));
  vo_front.set_reprojection_angle_threshold
    (Config::Get<double>("reprojection_angle_threshold"));
  vo_front.set_write_est_to_file(Config::Get<std::string>("write_est_to_file"));
  vo_front.set_show_left_and_right_matches
    (Config::Get<std::string>("show_left_and_right_matches"));  
  vo_front.set_show_prev_and_curr_matches
    (Config::Get<std::string>("show_prev_and_curr_matches"));
  vo_front.set_diaplay_single_match
    (Config::Get<std::string>("diaplay_single_match"));



  vo_front.resigterBA(vo_ba);
  vo_front.init();
  vo_front.run();  
  

  // Evaluate the estimated against the ground truth and plot.
  if ( do_evaluation ) {
    LOG(INFO) << "Starting evaluation. ";
    evaluate(groundtruth_file, estimated_file);
  }
  return 0;
}