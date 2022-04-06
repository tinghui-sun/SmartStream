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


include_directories(../../include)
include_directories(${POCO_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${TrafficCounter_INCLUDE_DIRS})
include_directories(${ByteTrack_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

message("CUDA_INCLUDE" ${CUDA_INCLUDE_DIRS})

link_directories(${POCO_LINK_DIRECTORIES})

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	add_definitions(-DGLGOSDK_EXPORTS)
endif()

aux_source_directory(src __CODES)
set(CODES ${__CODES})


add_library(TrafficCount SHARED ${CODES})

target_link_libraries(TrafficCount ${smdnnmodel_LIBRARIES}
                           ${OpenCV_LIBS}
                           ${TensorRT_LIBRARIES}
                           ${ONNXRuntime_LIBRARIES}
                           ${TrafficCounter_LIBRARIES}
                           ${ByteTrack_LIBRARIES}
                           ${CUDA_LIBRARIES})


#设置rpath，修改库查找路径
set_target_properties(TrafficCount PROPERTIES LINK_FLAGS "-Wl,-rpath,./plugins/3rdpart/Poco/lib/:./plugins/TrafficCount/lib/ -Wl,--disable-new-dtags")

#安装插件库
install(TARGETS TrafficCount DESTINATION ${CMAKE_PLUGINS_PREFIX}/${PROJECT_NAME}/lib/)

#安装第三方以来库
install(
    DIRECTORY ${smdnnmodel_DIR}/lib/ ${OpenCV_DIR}/../lib/ ${TensorRT_DIR}/lib/ ${ONNXRuntime_DIR}/lib/ ${TrafficCounter_DIR}/lib/ ${ByteTrack_DIR}/lib/
    DESTINATION  ${CMAKE_PLUGINS_PREFIX}/${PROJECT_NAME}/lib
    PATTERN "*.a" EXCLUDE
    PATTERN "*so*"
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    GROUP_EXECUTE GROUP_READ
    )

#安装配置文件
install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/config
    DESTINATION  ${CMAKE_PLUGINS_PREFIX}/${PROJECT_NAME}/
    )

#安装模型文件
install(
    DIRECTORY ${TrafficCountSDK_DIR}/model
    DESTINATION  ${CMAKE_PLUGINS_PREFIX}/${PROJECT_NAME}/
)


#需要移除libTrafficCounter.so的rpath，否则会覆盖掉插件库的rpath，导致无法找到依赖库
#patchelf --remove-rpath /vms/code/sunth/SmartStream/Algorithm/build_linux/install/plugins/TrafficCount/lib/libTrafficCounter.so

