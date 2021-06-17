project(Test)

#收集代码
file(GLOB TEST_SRC ../Src/*/*.cpp ../Src/*/*.h ../Src/*/*.h ./*.cpp ./*.cpp ./*.h ./*.c)

#添加可执行文件
add_executable(${PROJECT_NAME} ${TEST_SRC})

#添加链接库
if(WIN32)
	set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS ${VS_FALGS} )
endif(WIN32)

target_link_libraries(${PROJECT_NAME} ${LINK_LIB_LIST})

