/*
 * Copyright (c) 2016 The ZLMediaKit project authors. All Rights Reserved.
 *
 * This file is part of ZLMediaKit(https://github.com/xiongziliang/ZLMediaKit).
 *
 * Use of this source code is governed by MIT license that can be found in the
 * LICENSE file in the root of the source tree. All contributing project authors
 * may be found in the AUTHORS file in the root of the source tree.
 */

#ifndef FFMpegImageLoader_H_
#define FFMpegImageLoader_H_
#include <string>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <string.h>"

#ifdef __cplusplus
extern "C" {
#endif
#include "libavcodec/avcodec.h"
#include "libavutil/opt.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#ifdef __cplusplus
}
#endif

using namespace std;
class FFMpegImageLoader {
	public:
		FFMpegImageLoader()
		{

		}

		virtual ~FFMpegImageLoader(void)
		{

		}

		AVFrame* loadImages(const string& imageFilePath)
		{
			AVFormatContext *pFormatCtx = NULL;
			if (avformat_open_input(&(pFormatCtx), imageFilePath.c_str(), NULL, NULL) != 0)
			{
				printf("Can't open image file '%s'\n", imageFilePath);
				return nullptr;
			}

			m_pFormatContext.reset(pFormatCtx, [](AVFormatContext *pCtx) 
			{
				avformat_close_input(&pCtx);
				avformat_free_context(pCtx);
			});

			if (avformat_find_stream_info(pFormatCtx, NULL) < 0) 
			{
				printf("Can't find stream\n");
				return nullptr;
			}

			av_dump_format(pFormatCtx, 0, imageFilePath.c_str(), false);

			int index = av_find_best_stream(pFormatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);

			if (index == AVERROR_STREAM_NOT_FOUND)
			{
				printf("stream not found!\n");
				return nullptr;
			}

			AVCodec *pCodec = avcodec_find_decoder(pFormatCtx->streams[index]->codecpar->codec_id);
			if (!pCodec)
			{
				printf("Codec not found\n");
				return nullptr;
			}

			m_pCodecContext.reset(avcodec_alloc_context3(pCodec), [](AVCodecContext *pCtx) 
			{
				avcodec_close(pCtx);
				avcodec_free_context(&pCtx);
			});

			//AVCodecContext *pCodecCtx = m_pCodecContext.get();
			avcodec_parameters_to_context(m_pCodecContext.get(), pFormatCtx->streams[index]->codecpar);

			// Open codec
			if (avcodec_open2(m_pCodecContext.get(), pCodec, NULL) < 0)
			{
				printf("Could not open codec\n");
				return nullptr;
			}

			m_frame.reset(av_frame_alloc(), [](AVFrame *pFrame) 
			{
				av_frame_free(&pFrame);
			});

			int frameFinished;

			m_packet.reset(av_packet_alloc(), [](AVPacket *pPackage) {
				av_packet_free(&pPackage);
			});

			//memset(m_packet.get(), 0, sizeof(AVPacket));
			av_init_packet(m_packet.get());

			int framesNumber = 0;
			while (av_read_frame(pFormatCtx, m_packet.get()) >= 0)
			{
				if (m_packet.get()->stream_index != index) 
				{
					av_packet_unref(m_packet.get());
					continue;
				}

				int ret = avcodec_decode_video2(m_pCodecContext.get(), m_frame.get(), &frameFinished, m_packet.get());
				av_packet_unref(m_packet.get());
				if (frameFinished)
				{
					printf("Frame is decoded, size %d\n", ret);
					break;
				}
			}

			if (frameFinished)
			{
				return m_frame.get();
			}
			else
			{
				return nullptr;
			}
		}

	private:
		std::shared_ptr<AVFormatContext> m_pFormatContext;
		std::shared_ptr<AVCodecContext> m_pCodecContext;
		std::shared_ptr<AVFrame>  m_frame;
		std::shared_ptr<AVPacket> m_packet;
		int m_index = 0;
	};
#endif


