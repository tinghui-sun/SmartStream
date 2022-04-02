cmake_minimum_required(VERSION 3.10)

project(Algorithm)

set(ThirdPartPath ${PROJECT_SOURCE_DIR}/3rdpart/${CMAKE_SYSTEM_NAME})
set(FFMPEG_PATH ${ThirdPartPath}/ffmpeg)
set(POCO_PATH ${ThirdPartPath}/Poco)
set(OpenCV_DIR ${ThirdPartPath}/opencv)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${ThirdPartPath}/cmake)
list(APPEND CMAKE_MODULE_PATH ${OpenCV_DIR})

message ("CMAKE_MODULE_PATH" ${CMAKE_MODULE_PATH})

set(CMAKE_PREFIX_PATH ${CMAKE_MODULE_PATH})

set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/install CACHE PATH "" FORCE)

add_subdirectory(AlgorithmManager)
add_subdirectory(AlgorithmTest)
add_subdirectory(AlgorithmPlugin)

