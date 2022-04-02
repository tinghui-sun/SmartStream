cmake_minimum_required(VERSION 3.10)

project(AlgorithmManager)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x -Wl,-Bsymbolic")
endif ()

find_package(POCO REQUIRED)

aux_source_directory(src __CODES)
set(CODES ${__CODES})
#aux_source_directory(./http __CODES)
#set(CODES ${CODES} ${__CODES})

message("POCO_INCLUDE_DIRS" ${POCO_INCLUDE_DIRS})

include_directories(../include)
include_directories(${POCO_INCLUDE_DIRS})

link_directories(${POCO_LINK_DIRECTORIES})

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	add_definitions(-DGLGOSDK_EXPORTS)
endif()

add_library(AlgorithmManager SHARED ${CODES})

install(TARGETS AlgorithmManager DESTINATION ${CMAKE_INSTALL_PREFIX})