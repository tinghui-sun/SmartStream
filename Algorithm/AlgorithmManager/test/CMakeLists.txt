cmake_minimum_required(VERSION 3.14)
project(AlgorithmTest)
set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x -Wl,-Bsymbolic")
aux_source_directory(./ AlgorithmTest_SRC)

include(../../../cbb.cmake)
include_directories(../../include)
if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	add_definitions(-D GLGOSDK_EXPORTS)
	if (CMAKE_BUILD_TYPE STREQUAL "Debug")
		add_definitions(-D _DEBUG -D _WINDOWS -D _USRDLL)
	endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif()

include_directories(./ /home/jiuling/vms/opensource/poco/CppUnit/include)

find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)
message("${OpenCV_INCLUDE_DIRS}")
message("${OpenCV_LIBS}")

add_executable(AlgorithmTest ${AlgorithmTest_SRC})

#message("${POCO_LIBRARY}")

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	target_link_libraries(AlgorithmTest common sip resiprocate reprolib rutil dum ares opencv_core3410 opencv_ts3410 opencv_videoio3410 
	opencv_highgui3410 opencv_imgproc3410 opencv_imgcodecs3410 opencv_world3410 avresample Ws2_32 cares libcrypto libssl 
	media apiLib baseLib mediaLib)
	if (CMAKE_BUILD_TYPE MATCHES "Debug")
		target_link_libraries(vias libprotobuf-lited gmock libprotocd libprotobufd grpc grpc++ gpr zlibstaticd json_vc71_libmt)
	endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
	target_link_libraries(AlgorithmTest pthread AlgorithmManager ${POCO_LIBRARY} /usr/local/lib/libCppUnit.so ${OPENCV_LIBRARYS}) 
endif()
