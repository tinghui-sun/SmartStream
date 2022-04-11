#	
project(AlgorithmTest)

set(CUDA_TOOLKIT_ROOT_DIR  /usr/local/cuda-11.1/)

#set(CMAKE_BUILD_TYPE Release)

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=gnu++0x -Wl,-Bsymbolic")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
	add_definitions(-D GLGOSDK_EXPORTS)
endif()


find_package(POCO REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(../include)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${POCO_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

aux_source_directory(./ __CODES)
set(CODES ${__CODES})

link_directories(${POCO_LINK_DIRECTORIES})

add_executable(AlgorithmTest ${CODES})

message(${OpenCV_LIBS})
message(${POCO_LIBRARIES})

target_link_libraries(AlgorithmTest AlgorithmManager ${POCO_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES})

install(TARGETS AlgorithmTest DESTINATION ${CMAKE_INSTALL_PREFIX})
install(
    DIRECTORY ${OpenCV_DIR}/../../
    DESTINATION ${CMAKE_INSTALL_PREFIX}/plugins/3rdpart
	PATTERN "*.a" EXCLUDE
    PATTERN "*so*"
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    GROUP_EXECUTE GROUP_READ
    )
message("OpenCV_DIR" ${OpenCV_DIR})
message("OpenCV_DIR DESTINATION" ${CMAKE_INSTALL_PREFIX}/plugins/3rdpart)
set_target_properties(AlgorithmTest PROPERTIES LINK_FLAGS "-Wl,-rpath,./plugins/:./plugins/3rdpart/Poco/lib/:./plugins/3rdpart/opencv/lib/ -Wl,--disable-new-dtags")