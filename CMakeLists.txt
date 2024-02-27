cmake_minimum_required(VERSION 3.5)
project(lsd_demo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions(-DCOMPILEDWITHC11)
message(STATUS "Using flag -std=c++11.")

find_package(OpenCV 4.6)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV >= 4.6 not found. LSD depends on it")
endif()

add_executable(demo main.cpp lsd.cpp)

include_directories(
	${PROJECT_SOURCE_DIR}
	${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(demo
	${OpenCV_LIBS}
)