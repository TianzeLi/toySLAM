# toySLAM 
**_(Current contains the feature-based stereo VO but the content will be enriched.)_**
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

A SLAM implementation for learning's sake. Partly based on the ![slambook](https://github.com/gaoxiang12/slambook2). In particular, the feature-based visual odometry minimizes the re-projection error that formulated in the perspective-n-point method. It is solved by Gauss-Newton iteration on transform matrix group and outliers are rejected by RANSAC. The performance is evaluated on the Kitti stereo dataset.



## Dependency and installation
<a name="Installation"></a>
Module | Dependency
---------------- | -------
Feature detection     | `OpenCV` 
Feature matching      | `OpenCV`
P-n-P estimation      | `Sophus`
Bundle adjustment     | `g2o`



## Launch and testing
<a name="Launch"></a>

Once the project is compiled, the following executable will be available in the `/bin` folder. The configuration is up for changes in `/config/default.yaml`. 

Executable files | Functions
---------------- | -------
`test_vo`	            | Launch the visual odometry. 
`test_vo_frontend`      | Test the visual odometry front end, i.e. without bundle adjustment.
`test_data_loader`      | Test the data loading.
`evaluate_trajectory`   | Evaluate and plot the estimated trajectory against the ground truth.