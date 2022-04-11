cmake_minimum_required(VERSION 3.10)

project(LeaveDetection)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x -Wl,-Bsymbolic")
endif()

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0/)
#set(CUDA_TOOLKIT_ROOT_DIR  /usr/local/cuda-11.1/)

#set(CMAKE_BUILD_TYPE "Release" CACHE STRING "set build type to release" FORCE)

# CPP STANDARD
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

# SET THIRD PARTY PATH
set(SDK_DIR ${PROJECT_SOURCE_DIR}/VimoDetection)
set(OpenCV_DIR ${SDK_DIR}/opencv)
message("OPENCV DIR1 "  ${OpenCV_DIR}/../lib/)

message("SDK cmake " ${SDK_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${SDK_DIR}/cmake)

set(CMAKE_PREFIX_PATH ${CMAKE_MODULE_PATH})

find_package(OpenCV REQUIRED)
find_package(VimoDetection REQUIRED)
find_package(CUDA REQUIRED)


# find_package(
#     VimoDetection REQUIRED
#     PATHS ..
# )

find_package(POCO REQUIRED)

include_directories(../../include)
include_directories(${POCO_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

link_directories(${POCO_LINK_DIRECTORIES})

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	add_definitions(-DGLGOSDK_EXPORTS)
endif()

aux_source_directory(src __CODES)
set(CODES ${__CODES})


add_library(LeaveDetection SHARED ${CODES})

target_link_libraries(LeaveDetection VimoDetection ${OpenCV_LIBS} ${CUDA_LIBRARIES})

message("CUDA_LIBRARIES " ${CUDA_LIBRARIES})


#设置rpath，修改库查找路径
set_target_properties(LeaveDetection PROPERTIES LINK_FLAGS "-Wl,-rpath,./plugins/3rdpart/Poco/lib/:./plugins/LeaveDetection/lib/:./:/usr/local/cuda-10.0/targets/x86_64-linux/lib/: -Wl,--disable-new-dtags")

#安装插件库
install(TARGETS LeaveDetection DESTINATION ${CMAKE_PLUGINS_PREFIX}/${PROJECT_NAME}/lib/)

#安装第三方以来库
install(
    DIRECTORY ${SDK_DIR}/lib/ ${SDK_DIR}/opencv/lib/ 
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
    DIRECTORY ${SDK_DIR}/model
    DESTINATION  ${CMAKE_PLUGINS_PREFIX}/${PROJECT_NAME}/
)

#需要移除libVimoDetection.so的rpath，否则会覆盖掉插件库的rpath，导致无法找到依赖库
#readelf -d ./plugins/LeaveDetection/lib/libc_vimo_detection.so
#patchelf --remove-rpath ./plugins/LeaveDetection/lib/libVimoDetection.so;patchelf --remove-rpath ./plugins/LeaveDetection/lib/libopencv_imgproc.so.4.0;patchelf --remove-rpath ./plugins/LeaveDetection/lib/libopencv_core.so.4.0;patchelf --remove-rpath ./plugins/LeaveDetection/lib/libonnxruntime.so.1.1.2;patchelf --remove-rpath ./plugins/LeaveDetection/lib/libsodium.so.23

