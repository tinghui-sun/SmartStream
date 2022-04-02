#	
project(AlgorithmTest)

#set(CMAKE_BUILD_TYPE Release)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x -Wl,-Bsymbolic")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	add_definitions(-D GLGOSDK_EXPORTS)
endif()


find_package(FFMPEG REQUIRED)
find_package(POCO REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(../include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${POCO_INCLUDE_DIRS})
include_directories(${FFMPEG_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

aux_source_directory(./ __CODES)
set(CODES ${__CODES})

link_directories(${POCO_LINK_DIRECTORIES})
link_directories(${PROJECT_BINARY_DIR}/../AlgorithmManager/Debug/)

add_executable(AlgorithmTest ${CODES})

message(${OpenCV_LIBS})
message(${POCO_LIBRARIES})
message(${FFMPEG_LIBRARIES})

#set_target_properties(AlgorithmTest PROPERTIES LINK_FLAGS "-Wl,-rpath,./plugins/TrafficCount/lib:./:./lib:/vms/code/sunth/SmartStream/Algorithm/build/AlgorithmTest/plugins/TrafficCount/lib:$ORIGIN:$ORIGIN/plugins/TrafficCount/lib -Wl,--disable-new-dtags ")
#set_target_properties(AlgorithmTest PROPERTIES LINK_FLAGS "-Wl,-rpath,./plugins/TrafficCount/lib:./:./lib:/vms/code/sunth/SmartStream/Algorithm/build/AlgorithmTest/plugins/TrafficCount/lib:$ORIGIN:$ORIGIN/plugins/TrafficCount/lib")
#set_target_properties(AlgorithmTest PROPERTIES LINK_FLAGS "-Wl,--enable-new-dtags -Wl,-rpath,./plugins/TrafficCount/lib:./:./lib:/vms/code/sunth/SmartStream/Algorithm/build/AlgorithmTest/plugins/TrafficCount/lib:$ORIGIN:$ORIGIN/plugins/TrafficCount/lib")

target_link_libraries(AlgorithmTest AlgorithmManager ${POCO_LIBRARIES} ${OpenCV_LIBS} ${FFMPEG_LIBRARIES} ${CUDA_LIBRARIES})


#if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
#	if (CMAKE_BUILD_TYPE MATCHES "Debug")
#		target_link_libraries(vias libprotobuf-lited gmock libprotocd libprotobufd grpc grpc++ gpr zlibstaticd json_vc71_libmt)
#	endif()
#elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
#	target_link_libraries(AlgorithmTest pthread AlgorithmManager ${POCO_LIBRARY} /usr/local/lib/libCppUnit.so ${OPENCV_LIBRARYS}) 
#endif()
