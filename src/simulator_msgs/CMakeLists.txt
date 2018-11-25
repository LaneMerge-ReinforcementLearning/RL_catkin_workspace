cmake_minimum_required(VERSION 2.8.12)
project(simulator_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  geometry_msgs
  std_msgs
)

add_message_files(
  FILES
  Vehicle.msg
  EgoVehicle.msg
  TrafficVehicles.msg
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime geometry_msgs std_msgs
)