CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(Disparity)

set(CMAKE_BUILD_TYPE Release)

ADD_EXECUTABLE(disparity disparity.cpp main.cpp image.cpp)
TARGET_LINK_LIBRARIES(disparity ${OpenCV_LIBS})
