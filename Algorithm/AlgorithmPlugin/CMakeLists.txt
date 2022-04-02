project(AlgorithmPlugin)

#1. 如果第三方库有RUNPATH 通过下面工具删除
#https://blog.csdn.net/Longyu_wlz/article/details/108550528
#2. 在lib***插件动态库上加入 rpath，三方库查找路径
add_subdirectory(PluginDemo)
add_subdirectory(TrafficCount)
#add_subdirectory(FaceDetection)
#add_subdirectory(FaceRecognition)

