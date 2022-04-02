cmake_minimum_required(VERSION 3.10)

project(TrafficCount)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x -Wl,-Bsymbolic")
endif()

#set(CMAKE_BUILD_TYPE "Release" CACHE STRING "set build type to release" FORCE)

# CPP STANDARD
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# SET THIRD PARTY PATH
set(TrafficCountSDK_DIR ${PROJECT_SOURCE_DIR}/TrafficCount)
set(smdnnmodel_DIR ${TrafficCountSDK_DIR}/thirdParty/smdnnmodel)
set(OpenCV_DIR ${TrafficCountSDK_DIR}/thirdParty/OpenCV/cmake)
set(TensorRT_DIR ${TrafficCountSDK_DIR}/thirdParty/TensorRT)
set(ONNXRuntime_DIR ${TrafficCountSDK_DIR}/thirdParty/ONNXRuntime)
set(Eigen3_DIR ${TrafficCountSDK_DIR}/thirdParty/Eigen/include/eigen3/cmake)
set(TrafficCounter_DIR ${TrafficCountSDK_DIR}/TrafficCounterModule)
set(ByteTrack_DIR ${TrafficCountSDK_DIR}/TrackerModule)


list(APPEND CMAKE_MODULE_PATH ${TrafficCountSDK_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${TrafficCountSDK_DIR}/cmake)


find_package(POCO REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(ONNXRuntime REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(smdnnmodel REQUIRED)
find_package(TrafficCounter REQUIRED)
find_package(ByteTrack REQUIRED)

aux_source_directory(src __CODES)
set(CODES ${__CODES})

include_directories(../../include)
include_directories(${POCO_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${TrafficCounter_INCLUDE_DIRS})
include_directories(${ByteTrack_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

message("CUDA_INCLUDE"${CUDA_INCLUDE_DIRS})

link_directories(${POCO_LINK_DIRECTORIES})

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	add_definitions(-DGLGOSDK_EXPORTS)
endif()

add_library(TrafficCount SHARED ${CODES})

target_link_libraries(TrafficCount ${smdnnmodel_LIBRARIES}
                           ${OpenCV_LIBS}
                           ${TensorRT_LIBRARIES}
                           ${ONNXRuntime_LIBRARIES}
                           ${TrafficCounter_LIBRARIES}
                           ${ByteTrack_LIBRARIES}
                           ${CUDA_LIBRARIES})

message("ONNXRuntime_LIBRARIES"${ONNXRuntime_LIBRARIES})


#set_target_properties(TrafficCount PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set_target_properties(TrafficCount PROPERTIES LINK_FLAGS "-Wl,-rpath,./plugins/TrafficCount/lib:./ -Wl,--disable-new-dtags")
#set_target_properties(TrafficCount PROPERTIES LINK_FLAGS "-Wl,-rpath,./plugins/TrafficCount/lib:./:./lib:/vms/code/sunth/SmartStream/Algorithm/build/AlgorithmTest/plugins/TrafficCount/lib:$ORIGIN:$ORIGIN/plugins/TrafficCount/lib")
#set_target_properties(TrafficCount PROPERTIES LINK_FLAGS "-Wl,--enable-new-dtags -Wl,-rpath,./plugins/TrafficCount/lib:./:./lib:/vms/code/sunth/SmartStream/Algorithm/build/AlgorithmTest/plugins/TrafficCount/lib:$ORIGIN:$ORIGIN/plugins/TrafficCount/lib")
#set_target_properties(TrafficCount PROPERTIES LINK_FLAGS "-Wl,--disable-new-dtags")

#install(TARGETS PluginDemo DESTINATION ${LIBRARY_PATH})
# SET OUTPUT PATH
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

