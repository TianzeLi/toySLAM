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
  std::string config_file_path = "/home/tianze/toySLAM/config/default.yaml";
  Config::SetParameterFile(config_file_path);

  std::string dataset_path = Config::Get<std::string>("dataset_dir");
  std::string groundtruth_file = Config::Get<std::string>("groundtruth_file_path");
  std::string estimated_file = Config::Get<std::string>("outfile_path");
  
  // Prepare the dataset.
  auto data = DataStereo::Ptr(new DataStereo(dataset_path));
  if (data->init())
    LOG(INFO) << "Data loaded. ";
  else 
    LOG(ERROR) << "Data failed to load. "; 
  unsigned bundle_size = 7;
  
  // Initialize the BA.
  auto vo_ba = VOBA::Ptr(new VOBA(bundle_size));

  // Initialize the VO frontend.  
  VOFront vo_front(data);
  vo_front.set_do_RANSAC(Config::Get<std::string>("do_RANSAC"));
  vo_front.set_outfile_path(Config::Get<std::string>("outfile_path"));

  vo_front.resigterBA(vo_ba);
  vo_front.init();
  vo_front.run();  
  
  // Evaluate the estimated against the ground truth and plot.
  LOG(INFO) << "Starting evaluation. ";
  evaluate(groundtruth_file, estimated_file);

  return 0;
}