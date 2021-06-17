/*
 * Copyright (c) 2016 The ZLMediaKit project authors. All Rights Reserved.
 *
 * This file is part of ZLMediaKit(https://github.com/xiongziliang/ZLMediaKit).
 *
 * Use of this source code is governed by MIT license that can be found in the
 * LICENSE file in the root of the source tree. All contributing project authors
 * may be found in the AUTHORS file in the root of the source tree.
 */

#ifndef H264Decoder_H_
#define H264Decoder_H_
#include <string>
#include <memory>
#include <stdexcept>
#include <stdio.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif
//#include "libavutil/mathematics.h"
#include "libavcodec/avcodec.h"
//#include "libswscale/swscale.h"
#include "libavutil/opt.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#ifdef __cplusplus
}
#endif

using namespace std;
class FFMpegDecoder {
	public:
		FFMpegDecoder(AVCodecID ff_codec_id) 
		{
			avcodec_register_all();
			av_log_set_level(AV_LOG_INFO);
			AVCodec *pCodec = avcodec_find_decoder(ff_codec_id);
			if (!pCodec) {
				throw std::runtime_error("未找到解码器");
			}
			m_pContext.reset(avcodec_alloc_context3(pCodec), [](AVCodecContext *pCtx) {
				avcodec_close(pCtx);
				avcodec_free_context(&pCtx);
			});
			if (!m_pContext) {
				throw std::runtime_error("创建解码器失败");
			}
			m_pContext->thread_count = 2;
			if (pCodec->capabilities & AV_CODEC_CAP_TRUNCATED) {
				/* we do not send complete frames */
				m_pContext->flags |= AV_CODEC_FLAG_TRUNCATED;
			}
			if (avcodec_open2(m_pContext.get(), pCodec, NULL) < 0) {
				throw std::runtime_error("打开编码器失败");
			}
			m_pFrame.reset(av_frame_alloc(), [](AVFrame *pFrame) {
				av_frame_free(&pFrame);
			});
			if (!m_pFrame) {
				throw std::runtime_error("创建帧缓存失败");
			}
		}
		virtual ~FFMpegDecoder(void) {}
		bool inputVideo(unsigned char* data, unsigned int dataSize, uint32_t ui32Stamp, AVFrame **ppFrame) 
		{
			AVPacket pkt;
			av_init_packet(&pkt);
			pkt.data = data;
			pkt.size = dataSize;
			pkt.dts = ui32Stamp;
			int iGotPicture;
			auto iLen = avcodec_decode_video2(m_pContext.get(), m_pFrame.get(), &iGotPicture, &pkt);
			if (!iGotPicture || iLen < 0) {
				if (iLen < 0)
				{
					char errLog[1024] = { 0 };;
					av_strerror(iLen, errLog, 1024);
					printf("FFMpegEncoder::encode failed! [%d]%s \r\n", iLen, errLog);
				}
				return false;
			}
			*ppFrame = m_pFrame.get();
			return true;
		}

		bool inputVideo(unsigned char* data, unsigned int dataSize, uint32_t ui32Stamp, AVFrame *ppFrame) 
		{
			AVPacket pkt;
			av_init_packet(&pkt);
			pkt.data = data;
			pkt.size = dataSize;
			pkt.dts = ui32Stamp;
			int iGotPicture;
			auto iLen = avcodec_decode_video2(m_pContext.get(), ppFrame, &iGotPicture, &pkt);
			if (!iGotPicture || iLen < 0) {
				return false;
			}
			return true;
		}
	private:
		std::shared_ptr<AVCodecContext> m_pContext;
		std::shared_ptr<AVFrame> m_pFrame;
	};

#endif /* H264Decoder_H_ */


