#set(POCO_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdpart/Poco/${CMAKE_SYSTEM_NAME})
find_path(POCO_INCLUDE_DIRS 
	NAMES 
	Poco/Version.h
	PATHS 
	${POCO_PATH}/include/
	)

find_library(PocoFoundation_LIBRARIES 
	NAMES 		 	
	PocoFoundation
	PATHS 
	${POCO_PATH}/lib/
	)

find_library(PocoUtil_LIBRARIES 
	NAMES 		 	
	PocoUtil
	PATHS 
	${POCO_PATH}/lib/
)

find_library(CppUnit_LIBRARIES 
	NAMES 		 	
	CppUnit
	PATHS 
	${POCO_PATH}/lib/
	)

if(PocoFoundation_LIBRARIES AND CppUnit_LIBRARIES)
	set(POCO_LIBRARIES ${PocoFoundation_LIBRARIES} ${CppUnit_LIBRARIES} ${PocoUtil_LIBRARIES})
endif()


set(POCO_LINK_DIRECTORIES ${POCO_PATH}/lib/)

if(POCO_INCLUDE_DIRS AND POCO_LINK_DIRECTORIES AND POCO_LIBRARIES)
	include(FindPackageHandleStandardArgs)
	find_package_handle_standard_args(POCO DEFAULT_MSG POCO_INCLUDE_DIRS POCO_LINK_DIRECTORIES POCO_LIBRARIES)
else()
	message(FATAL_ERROR, "poco not found! ${POCO_PATH}")
endif()