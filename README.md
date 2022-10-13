# toySLAM 
**_(Currently contains the feature-based stereo VO but the content shall be enriched.)_**
<!-- ![](doc/media/projectTheme2.png) -->

## Contents

- [Overview](#Overview)
- [Dependency and installation](#Installation)
- [Launch](#Launch)
<!-- - [Documentation](#Documentation) -->
<!-- - [License](#License) -->
<!-- - [API documentation](#API-documentation) -->
<!-- - [Read more](##Read-more) -->


## Overview
<a name="Overview"></a>

A visual SLAM implementation for learning's sake. Partly based on ![slambook](https://github.com/gaoxiang12/slambook2). In particular, the feature-based visual odometry minimizes the re-projection error that formulated in the perspective-n-point method. It is solved by Gauss-Newton iteration on transform matrix group and outliers are rejected by RANSAC. The performance is evaluated on the Kitti stereo dataset.

Initial report that covers up till VO without bundle adjustment is available in the ![doc](https://github.com/TianzeLi/toySLAM/tree/main/doc/report-EL2620-TianzeLi.pdf).


## Dependency and installation

cmake files included. 

<a name="Installation"></a>
Module | Dependency
---------------- | -------
Feature detection     | `OpenCV` 
Feature matching      | `OpenCV`
P-n-P estimation      | `Sophus` and `Eigen`
Bundle adjustment     | `g2o`
Visualization         | `gnuplot` and `Pangolin`
Logging               | `GLOG`



## Launch and testing
<a name="Launch"></a>

The following executable are compiled to folder `/bin`. The configuration is up for changes in `/config/default.yaml`. 

Executable files | Functions
---------------- | -------
`test_vo`	            | Launch the visual odometry. 
`test_vo_frontend`      | Test the visual odometry front end, i.e. without bundle adjustment.
`test_data_loader`      | Test the data loading.
`evaluate_trajectory`   | Evaluate and plot the estimated trajectory against the ground truth.