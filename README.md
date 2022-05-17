# toySLAM 
======
<!-- **_(Finishing soon but still under construction.)_** -->
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

A SLAM implementation for learning's sake. Partly based on the ![slambook](https://github.com/gaoxiang12/slambook2). 



<!-- ## Dependency and installation
<a name="Installation"></a> -->


## Launch
<a name="Launch"></a>

In order for testing, we seperate the functions into different launch files, which can be combined in one overall launch if desired.

Launch files   | Functions
-------------- | -------
`roslaunch am_driver_safe automower_hrp.launch`	| Launch the robot
`roslaunch am_sensors sensors.launch`          	| Launch the added sensors
`rosrun am_driver hrp_teleop.py`            	  | Control via keyboard
`roslaunch am_driver_safe ekf_template.launch`  | Launch localization

Using the GUI in `test/scripts/map_measure.py`, we can calculate the positions of the interested points on the map: