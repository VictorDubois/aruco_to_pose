cmake_minimum_required(VERSION 3.0.2)
project(aruco_to_pose)


find_package(catkin REQUIRED)

catkin_package(
)

include_directories(
)

catkin_install_python(PROGRAMS
  src/aruco_to_pose.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

