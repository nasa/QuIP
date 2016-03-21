
#ifndef _AVCODEC_HDR_H

#if defined(__cplusplus) || defined(c_plusplus)

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

#else

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>

#endif

/* seeks jump by 250, this size allows up to a million frames - enough? */
#define MAX_AVI_SEEK_TBL_SIZE 4096

typedef struct {
	uint32_t	seek_target;
	uint32_t	seek_result;
} Seek_Info;

//#define MAX_STORED_OFFSETS 1024
#define MAX_STORED_OFFSETS 8192

typedef struct {
	int32_t	frame_index;
	int32_t	pts_offset;
} Frame_Skew;


typedef struct avcodec_hdr {
	/* This stuff is not really the header,
	 * it's per-file state info that we need to keep...
	 */
	AVFormatContext *	avch_format_ctx_p;
	AVCodecContext *	avch_codec_ctx_p;
	int			avch_frame_finished;
	AVPacket		avch_packet;
	AVStream *		avch_video_stream_p;
	AVFrame *		avch_frame_p;
	AVFrame *		avch_rgb_frame_p;
	uint8_t *		avch_buffer;
	int			avch_num_bytes;
	AVCodec *		avch_codec_p;
	struct SwsContext *	avch_img_convert_ctx_p;
	int			avch_video_stream_index;
	int			avch_have_frame_ready;
	uint32_t		avch_fps;
	double			avch_pts;	/* most recently read presentation time stamp */
	double			avch_duration;	/* in seconds */
	int32_t			avch_n_seek_tbl_entries;
	int32_t			avch_n_skew_tbl_entries;
	Seek_Info *		avch_seek_tbl;
	Frame_Skew *		avch_skew_tbl;
} AVCodec_Hdr;

#define _AVCODEC_HDR_H

#endif /* ! _AVCODEC_HDR_H */
