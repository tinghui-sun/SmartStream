cmake_minimum_required(VERSION 3.20)

project(AlgorithmManager)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x -Wl,-Bsymbolic")
endif ()

find_package(FFMPEG REQUIRED)
find_package(POCO REQUIRED)


aux_source_directory(src __CODES)
set(CODES ${__CODES})
#aux_source_directory(./http __CODES)
#set(CODES ${CODES} ${__CODES})

include_directories(../include)
include_directories(${POCO_INCLUDE_DIRS})
include_directories(${FFMPEG_INCLUDE_DIRS})

link_directories(${POCO_LINK_DIRECTORIES})

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	add_definitions(-DGLGOSDK_EXPORTS)
endif()

add_library(AlgorithmManager SHARED ${CODES})
#add_library(AlgorithmManager SHARED ${CODES} ${FFMPEG_LIBRARYS} ${POCO_LIBRARYS})

install(TARGETS AlgorithmManager DESTINATION ${LIBRARY_PATH})
#add_subdirectory(test)
