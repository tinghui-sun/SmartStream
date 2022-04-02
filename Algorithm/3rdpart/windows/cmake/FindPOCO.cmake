message(${POCO_PATH})
find_path(POCO_INCLUDE_DIRS 
	NAMES 
	Poco/Version.h
	PATHS 
	${POCO_PATH}/include/
	)

find_library(CPPUNIT_LIBRARIE
	NAMES 		 	
	CppUnitd
	PATHS 
	${POCO_PATH}/lib/
	)

find_library(POCO_FOUNDATION_LIBRARIE
	NAMES 		 	
	PocoFoundationd
	PATHS 
	${POCO_PATH}/lib/
	)

if(CPPUNIT_LIBRARIE AND POCO_FOUNDATION_LIBRARIE)
	set(POCO_LINK_DIRECTORIES ${POCO_PATH}/lib/)	
else()
	message(FATAL_ERROR, "poco include dir not find! ${POCO_PATH}")
endif()

if(POCO_INCLUDE_DIRS AND POCO_LINK_DIRECTORIES)
	include(FindPackageHandleStandardArgs)
	find_package_handle_standard_args(POCO DEFAULT_MSG POCO_INCLUDE_DIRS POCO_LINK_DIRECTORIES)
else()
	message(FATAL_ERROR, "poco not found! ${POCO_PATH}")
endif()