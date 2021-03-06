/**
 * @file test_vo_frontend.cpp
 * @author Tianze Li (tianze.li.eu@gmail.com)
 * @brief Test the VO frontend.
 * @version 0.1
 * @date 2022-04-06
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "toyslam/vo_frontend.h"

using namespace toyslam;

int main(int argc, char* argv[]) {
  std::string dataset_path = "/home/tianze/toySLAM/test/data/00";
  auto data = DataStereo::Ptr(new DataStereo(dataset_path));
  if (data->init())
    LOG(INFO) << "Data loaded. ";
  else 
    LOG(ERROR) << "Data failed to load. "; 
  VOFront vo_front(data);
  vo_front.init();
  vo_front.run();  
  
  return 0;
}