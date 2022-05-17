cmake_minimum_required(VERSION 3.10)
project(toySLAM)

# User build settings.
set(USE_G20 True)
# set(USE_ROS False)
# set(CMAKE_BUILD_TYPE Release)

# Compiler settings.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_CXX_FLAGS "-std=c++14 -Wall")
# set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++14 -O3 -fopenmp -pthread")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

if(HAVE_G2O)
  add_definitions(-DUSE_BUNDLE_ADJUSTMENT)
endif()
if(USE_ROS)
  add_definitions(-DSVO_USE_ROS)
endif()

# Dependencies.
include_directories("/usr/include/eigen3")
include_directories("/usr/include/gnuplot-iostream")
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Glog REQUIRED) 
find_package(gflags REQUIRED)
find_package(Pangolin REQUIRED)

# Include dirs
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
)

# Set link libraries
list(APPEND LINK_LIBS 
  ${OpenCV_LIBS}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
  ${fast_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  ${Pangolin_LIBRARIES}
)

# Set sourcefiles.
list(APPEND SOURCEFILES
  src/data_stereo.cpp
  src/vo_frontend.cpp
  src/vo_ba.cpp
  src/config.cpp
)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Add g2o if available
if(HAVE_G2O)
  find_package(G2O REQUIRED)
  include_directories(
    ${G2O_INCLUDE_DIR}
  )
  list(APPEND LINK_LIBS
    ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_SOLVER_CHOLMOD} ${G2O_SOLVER_CSPARSE} ${G2O_SOLVER_DENSE}
    ${G2O_SOLVER_PCG} ${G2O_TYPES_SBA}
  )
#   list(APPEND SOURCEFILES src/bundle_adjustment.cpp)
endif(HAVE_G2O)

# Create toyslam library.
add_library(toyslam SHARED ${SOURCEFILES})
target_link_libraries(toyslam ${LINK_LIBS} fmt)

# Testing modules.
add_executable(check_dep_version test/src/check_dep_version.cpp)
target_link_libraries(check_dep_version ${LINK_LIBS})
add_executable(test_data_loader test/src/test_data_loader.cpp)
target_link_libraries(test_data_loader ${LINK_LIBS} toyslam)
add_executable(test_vo_frontend test/src/test_vo_frontend.cpp)
target_link_libraries(test_vo_frontend ${LINK_LIBS} toyslam)
add_executable(test_vo test/src/test_vo.cpp)
target_link_libraries(test_vo ${LINK_LIBS} toyslam)
add_executable(evaluate_trajectory test/src/evaluate_trajectory.cpp)
target_link_libraries(evaluate_trajectory ${LINK_LIBS} toyslam)