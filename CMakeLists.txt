cmake_minimum_required(VERSION 3.5)
project(lsd_demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions(-DCOMPILEDWITHC11)
message(STATUS "Using flag -std=c++11.")

find_package(OpenCV 4.6)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV >= 4.6 not found. LSD depends on it")
endif()

find_package(Eigen3 REQUIRED)

# 包含所需的头文件目录
include_directories(
    ${PROJECT_SOURCE_DIR}/includes
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_library(feature_detection src/feature_detection.cpp)

target_link_libraries(feature_detection ${OpenCV_LIBS})

link_libraries(feature_detection)

add_executable(main src/main.cpp)

add_executable(visualization src/visualization.cpp)