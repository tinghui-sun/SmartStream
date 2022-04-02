cmake_minimum_required(VERSION 3.10)

project(PluginDemo)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x -Wl,-Bsymbolic")
endif ()

find_package(POCO REQUIRED)

set(SDK_DIR ${PROJECT_SOURCE_DIR}/AlgorithmPlugin/PluginDemo/SDK)

aux_source_directory(src __CODES)
set(CODES ${__CODES})

include_directories(../../include)
include_directories(${POCO_INCLUDE_DIRS})

link_directories(${POCO_LINK_DIRECTORIES})

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	add_definitions(-DGLGOSDK_EXPORTS)
endif()

add_library(PluginDemo SHARED ${CODES})

#install(TARGETS PluginDemo DESTINATION ${LIBRARY_PATH})

