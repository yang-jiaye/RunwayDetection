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
include_directories(${EIGEN3_INCLUDE_DIR})

# 编译lsd.cpp成一个库
add_library(lsd src/lsd.cpp)

# 链接OpenCV库到lsd_lib
target_link_libraries(lsd ${OpenCV_LIBS})

link_libraries(lsd)

# 创建可执行目标main并链接lsd_lib
add_executable(main src/main.cpp)

# 创建可执行目标animation并链接lsd_lib
add_executable(animation src/animation.cpp)

# 包含所需的头文件目录
include_directories(
    ${PROJECT_SOURCE_DIR}/includes
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_custom_command(OUTPUT clean_png COMMAND rm -f ./*.png)
add_custom_target(clean_png DEPENDS clean_png)