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

using namespace toyslam;

int main(int argc, char* argv[]) {
  std::string dataset_path = "/home/tianze/toySLAM/test/data/00";
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
  vo_front.resigterBA(vo_ba);
  vo_front.init();
  vo_front.run();  
  

  return 0;
}