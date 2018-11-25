cmake_minimum_required(VERSION 2.8.3)
project(simulator)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

set( packages
    rospy
    simulator_msgs
)

find_package(catkin REQUIRED COMPONENTS
    ${packages}
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES simulator
  CATKIN_DEPENDS ${packages}
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#############
## Install ##
#############

install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY
  resources
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS
  src/__init__.py
  src/run_simulation.py
  src/simulator.py
  src/simulator_helper_functions.py
  src/traffic_generation_functions.py

  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  src/helper_classes
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)