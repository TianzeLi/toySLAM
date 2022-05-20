/**
 * @file compute_estimation_error.cpp
 * @author Tianze Li (tianze.li.eu@gmail.com)
 * @brief Compute the RMSE error against the ground truth.
 *        Code adapted from slambook.
 * @version 0.1
 * @date 2022-04-30
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef EVALUATE_TRACECTORY_H
#define EVALUATE_TRACECTORY_H 

#include "toyslam/common_include.h"
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <boost/tuple/tuple.hpp>
#include "gnuplot//gnuplot-iostream.h"

using namespace std;

typedef vector<Eigen::Vector3d> TrajectoryType;

void plotTrajectory2D(const TrajectoryType &gt, const TrajectoryType &est);
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &est);

TrajectoryType ReadTrajectory(const string &path);

void evaluate(std::string groundtruth_file, std::string estimated_file) {
  TrajectoryType groundtruth = ReadTrajectory(groundtruth_file);
  TrajectoryType estimated = ReadTrajectory(estimated_file);
  assert( !groundtruth.empty() && !estimated.empty() );
  assert( groundtruth.size() >= estimated.size() );

  double rmse = 0;
  for (size_t i = 0; i < estimated.size(); i++) {
      Eigen::Vector3d p1 = estimated[i], p2 = groundtruth[i];
      double error = (p1 - p2).norm();
      rmse += error*error;
  }
  rmse = rmse / double(estimated.size());
  rmse = sqrt(rmse);
  LOG(INFO) << "RMSE = " << rmse << endl;

  plotTrajectory2D(groundtruth, estimated);
  // DrawTrajectory(groundtruth, estimated);
}


TrajectoryType ReadTrajectory(const string &path) {
  ifstream fin(path);
  TrajectoryType trajectory;
  if( !fin ) {
    cerr << "trajectory " << path << "not found. " << endl;
    return trajectory;
  }
  
  while(!fin.eof()) {
    double r11, r12, r13, t1, r21, r22, r23, t2, r31, r32, r33, t3;
    fin >> r11 >> r12 >> r13 >> t1 >> r21 >> r22 >> r23 >> t2 >> r31 >> r32 >> r33 >> t3;
    Eigen::Vector3d V;
    V << t1, t2, t3;
    trajectory.push_back(V);
  }
  return trajectory;
}

void plotTrajectory2D(const TrajectoryType &gt, const TrajectoryType &est) {
  Gnuplot gp;

  std::vector<std::pair<double, double> > xy_pts_gt;
  std::vector<std::pair<double, double> > xy_pts_est;

  for(unsigned i = 0; i < est.size(); ++i) {
    // Note that the transform between the camera frame and word frame.
    xy_pts_gt.push_back(std::make_pair(gt[i](0), gt[i](2)));
    xy_pts_est.push_back(std::make_pair(est[i](0), est[i](2)));
  }

  // gp << "set size ratio -1" << std::endl;
  gp << "set xrange[-2:2]; set yrange[-2:10]; set size ratio -1;" << std::endl;
  // gp << "set grid" << std::endl;
  gp << "plot" << gp.file1d(xy_pts_gt) << "lw 4 with lines title 'Ground truth',"
	   << gp.file1d(xy_pts_est) << "lw 4 with lines title 'Estimated'," << std::endl; 
  // gp << "set style line 1 lw 7" << std::endl;
  // gp << "set style line 2 lw 7" << std::endl;

}

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
      .SetHandler(new pangolin::Handler3D(s_cam));


  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glLineWidth(2);
    for (size_t i = 0; i < esti.size() - 1; i++) {
      glColor3f(0.0f, 1.0f, 0.0f);  // green for ground truth
      glBegin(GL_LINES);
      auto p1 = gt[i], p2 = gt[i + 1];
      glVertex3d(p1(0), p1(1), p1(2));
      glVertex3d(p2(0), p2(1), p2(2));
      glEnd();
    }

    for (size_t i = 0; i < esti.size() - 1; i++) {
      glColor3f(0.0f, 0.0f, 1.0f);  // blue for estimated
      glBegin(GL_LINES);
      auto p1 = esti[i], p2 = esti[i + 1];
      glVertex3d(p1(0), p1(1), p1(2));
      glVertex3d(p2(0), p2(1), p2(2));
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }

}


#endif // EVALUATE_TRACECTORY_H