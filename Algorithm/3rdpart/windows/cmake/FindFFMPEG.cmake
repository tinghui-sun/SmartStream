message(${FFMPEG_PATH})

find_path(FFMPEG_INCLUDE_DIRS 
	NAMES 
	libavcodec/avcodec.h
	#libavdevice/avdevice.h
	libavfilter/avfilter.h
	libavformat/avformat.h
	#libavresample/avresample.h
	libavutil/avutil.h
	#libswresample/swresample.h
	libswscale/swscale.h
	PATHS 
	${FFMPEG_PATH}/include/
	)
	
message(${FFMPEG_PATH}/include/)
message(${FFMPEG_CODEC_LIBRARIES})	
message(${FFMPEG_AVFILTER_LIBRARIES})

find_library(FFMPEG_CODEC_LIBRARIES 
	NAMES 		 	
	avcodec
	PATHS 
	${FFMPEG_PATH}/lib/
	)

find_library(FFMPEG_AVFILTER_LIBRARIES 
	NAMES 		 	
	avfilter
	PATHS 
	${FFMPEG_PATH}/lib/
	)

find_library(FFMPEG_AVFORMAT_LIBRARIES 
	NAMES 		 	
	avformat
	PATHS 
	${FFMPEG_PATH}/lib/
	)

find_library(FFMPEG_AVUTIL_LIBRARIES 
	NAMES 		 	
	avutil
	PATHS 
	${FFMPEG_PATH}/lib/
	)

find_library(FFMPEG_SWSCALE_LIBRARIES 
	NAMES 		 	
	swscale
	PATHS 
	${FFMPEG_PATH}/lib/
	)
	
if(FFMPEG_CODEC_LIBRARIES AND FFMPEG_AVFILTER_LIBRARIES AND FFMPEG_AVFORMAT_LIBRARIES AND FFMPEG_AVUTIL_LIBRARIES AND FFMPEG_SWSCALE_LIBRARIES)
	set(FFMPEG_LIBRARIES ${FFMPEG_CODEC_LIBRARIES} ${FFMPEG_AVFILTER_LIBRARIES} ${FFMPEG_AVFORMAT_LIBRARIES} ${FFMPEG_AVUTIL_LIBRARIES} ${FFMPEG_SWSCALE_LIBRARIES})
endif()

if(FFMPEG_INCLUDE_DIRS AND FFMPEG_LIBRARIES)
	message("FFMPEG_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIRS}")
	message("FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES}")
	include(FindPackageHandleStandardArgs)
	find_package_handle_standard_args(FFMPEG DEFAULT_MSG FFMPEG_INCLUDE_DIRS FFMPEG_LIBRARIES)
else()
	message("ffmpeg not found! ${FFMPEG_PATH}")
endif()