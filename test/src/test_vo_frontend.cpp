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

int main(int argc, char* argv[]) {
  
  std::string dataset_path = "/home/tianze/toySLAM/test/data/00";
  toyslam::VOFront vo_front(dataset_path);
  vo_front.init();
  vo_front.run();  
  
  return 0;
}