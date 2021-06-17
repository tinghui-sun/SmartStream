project(SmartStream)

cmake_minimum_required(VERSION 3.1.3)
#使能c++11
set(CMAKE_CXX_STANDARD 11)
#加载自定义模块
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/Cmake")

#set(CMAKE_BUILD_TYPE "Release")

if(${CMAKE_BUILD_TYPE} MATCHES "Release")
    message(STATUS "Release版本")
    set(BuildType "Release")
else()
    set(BuildType "Debug")
    message(STATUS "Debug版本")
endif()

set(VS_FALGS "/wd4819" )

set(RELEASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/release)
SET(LIBRARY_OUTPUT_PATH ${RELEASE_DIR}/${CMAKE_SYSTEM_NAME})
SET(EXECUTABLE_OUTPUT_PATH ${RELEASE_DIR}/${CMAKE_SYSTEM_NAME})


if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    add_compile_options(-fPIC -Wall -Wno-unused-variable -Wno-unused-value)
endif()

set(LINK_LIB_LIST)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Src)

find_package(FFMPEG QUIET)
if(FFMPEG_FOUND)
	message("find FFMPEG ${FFMPEG_INCLUDE_DIR}")
    include_directories(${FFMPEG_INCLUDE_DIR})
    list(APPEND  LINK_LIB_LIST ${FFMPEG_LIBRARIES})
endif()

find_package(ZLMEDIAKIT QUIET)
if(ZLMEDIAKIT_FOUND)
	message("find ZLMEDIAKIT ${ZLMEDIAKIT_INCLUDE_DIRS}")
    include_directories(${ZLMEDIAKIT_INCLUDE_DIRS})
	link_directories(${ZLMEDIAKIT_LINK_DIRECTORIES})
endif()

find_package(POCO QUIET)
if(POCO_FOUND)
	message("find POCO ${POCO_INCLUDE_DIR}")
    include_directories(${POCO_INCLUDE_DIR})
	link_directories(${POCO_LINK_DIRECTORIES})
	add_definitions(-DPOCO_STATIC)
	#POCO的库在头文件指定链接名字,不能放到LINK_LIB_LIST,只需要指定link_directories,POCO会自动查找
    #list(APPEND  LINK_LIB_LIST ${POCO_LIBRARIES})
endif()

add_subdirectory(Test)

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	#add_folders(cbb/DeviceSDK DaHuaSDK HikVisionSDK Onvif rapidonvif)
endif()


