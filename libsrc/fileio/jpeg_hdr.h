
#ifndef _JPEG_HDR_H_
#define _JPEG_HDR_H_

#ifdef INC_VERSION
char VersionId_inc_jpeg_hdr[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#ifdef HAVE_JPEGLIB_H
#include <jpeglib.h>
#endif

typedef struct jpeg_hdr {
	/* info we get from SOF0 marker, and pass to fileio lib to fill in if_dp */
	int				jpeg_comps;
	int				jpeg_width;
	int				jpeg_height;
	int				jpeg_frames;

	u_int				jpeg_size_offset;
	size_t				jpeg_last_offset;

	u_int *				jpeg_seek_table;

	/* LML data */
	long				lml_sec;
	long				lml_usec;
	int32_t				lml_frameSize;
	int32_t				lml_frameSeqNo;
	int32_t				lml_colorEncoding;
	int32_t				lml_videoStream;
	short				lml_timeDecimation;
	int				lml_frameNo;

	int				lml_fieldNo;

	/* JPEG lib stuff */
	union {
		struct jpeg_decompress_struct	d_cinfo;
		struct jpeg_compress_struct	c_cinfo;
	} u;
	struct jpeg_error_mgr		jerr;

#ifdef PROGRESS_REPORT
	struct cdjpeg_progress_mgr progress;
#endif

} Jpeg_Hdr;

#endif /* _JPEG_HDR_H_ */

