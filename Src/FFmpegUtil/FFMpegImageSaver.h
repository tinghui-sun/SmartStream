/*
 * Copyright (c) 2016 The ZLMediaKit project authors. All Rights Reserved.
 *
 * This file is part of ZLMediaKit(https://github.com/xiongziliang/ZLMediaKit).
 *
 * Use of this source code is governed by MIT license that can be found in the
 * LICENSE file in the root of the source tree. All contributing project authors
 * may be found in the AUTHORS file in the root of the source tree.
 */

#ifndef FFMpegImageSaver_H
#define FFMpegImageSaver_H

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
class FFMpegImageSaver {
	public:
		//encoders: libx264 libx264rgb h264_amf h264_mf h264_nvenc h264_qsv nvenc nvenc_h264
		//			libx265 nvenc_hevc hevc_amf hevc_mf hevc_nvenc hevc_qsv
		FFMpegImageSaver()
		{

		}

		virtual ~FFMpegImageSaver(void)
		{

		}

		bool saveImages(const string& imageFilePath, const AVFrame* frameData)
		{
			AVFormatContext* pFormatCtx;// = avformat_alloc_context();
			int ret = avformat_alloc_output_context2(&pFormatCtx, NULL, NULL, imageFilePath.c_str());
			if (ret < 0)
			{
				printf("avformat_alloc_output_context2 failed!.");
				return false;
			}

			m_pFormatContext.reset(pFormatCtx, [](AVFormatContext *pCtx)
			{
				avio_close(pCtx->pb);
				avformat_free_context(pCtx);
			});

			if (avio_open(&pFormatCtx->pb, imageFilePath.c_str(), AVIO_FLAG_READ_WRITE) < 0) 
			{
				printf("Couldn't open output file.");
				return false;
			}

			AVStream* pAVStream = avformat_new_stream(pFormatCtx, 0);
			if (pAVStream == NULL) 
			{
				return false;
			}

			m_pAVStream.reset(pAVStream, [](AVStream *pStream)
			{
				avcodec_close(pStream->codec);
			});
			
#if 0 //如果色彩空间、图片大小不同,需要转换色彩空间
			AVFrame* dst = av_frame_alloc();
			int width = 1280;
			int height = 960;
			enum AVPixelFormat dst_pixfmt = AV_PIX_FMT_YUV420P;
			int numBytes = avpicture_get_size(dst_pixfmt, width, height);
			uint8_t *buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
			avpicture_fill((AVPicture *)dst, buffer, dst_pixfmt, width, height);

			struct SwsContext *convert_ctx = NULL;
			enum AVPixelFormat src_pixfmt = (enum AVPixelFormat)pFrame->format;

			convert_ctx = sws_getContext(pFrame->width, pFrame->height, pCodecCtx->pix_fmt, width, height, dst_pixfmt,
				SWS_POINT, NULL, NULL, NULL);
			sws_scale(convert_ctx, pFrame->data, pFrame->linesize, 0, pFrame->height,
				dst->data, dst->linesize);
			sws_freeContext(convert_ctx);
			dst->format = (int)dst_pixfmt;
			dst->width = width;
			dst->height = height;
			dst->pts = 0;
			dst->pkt_pts = 0;
			dst->pkt_dts = 0;
#endif

			AVCodecContext* pCodecCtx = m_pAVStream->codec;

			pCodecCtx->codec_id = m_pFormatContext->oformat->video_codec;
			pCodecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
			pCodecCtx->pix_fmt = (AVPixelFormat)frameData->format;
			pCodecCtx->width = frameData->width;
			pCodecCtx->height = frameData->height;
			pCodecCtx->time_base.num = 1;
			pCodecCtx->time_base.den = 25;

			// Begin Output some information
			av_dump_format(m_pFormatContext.get(), 0, imageFilePath.c_str(), 1);
			// End Output some information

			AVCodec* pCodec = avcodec_find_encoder(pCodecCtx->codec_id);
			if (!pCodec) 
			{
				printf("Codec not found.");
				return false;
			}

			if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) 
			{
				printf("Could not open codec.");
				return false;
			}

			//Write Header
			avformat_write_header(m_pFormatContext.get(), NULL);

			int y_size = pCodecCtx->width * pCodecCtx->height;

			m_packet.reset(av_packet_alloc(), [](AVPacket *pPackage) 
			{
				av_packet_free(&pPackage);
			});

			if (!m_packet) 
			{
				printf("av_packet_alloc failed.");
				return false;
			}
			av_init_packet(m_packet.get());

			int got_picture = 0;
			ret = avcodec_encode_video2(pCodecCtx, m_packet.get(), frameData, &got_picture);
			if (ret < 0) 
			{
				printf("Encode Error.\n");
				return false;
			}
			if (got_picture == 1) 
			{
				//pkt.stream_index = pAVStream->index;
				//ret = av_write_frame(pFormatCtx, &pkt);
				ret = av_interleaved_write_frame(m_pFormatContext.get(), m_packet.get());
			}
			else
			{
				printf("avcodec_encode_video2 no frame!!\n");
			}

			//Write Trailer
			av_write_trailer(pFormatCtx);
			av_packet_unref(m_packet.get());
		}


	private:
		std::shared_ptr<AVFormatContext> m_pFormatContext;
		std::shared_ptr<AVStream> m_pAVStream;
		std::shared_ptr<AVPacket> m_packet;
		int m_index = 0;
	};
#endif


