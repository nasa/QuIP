#ifndef _JPEG_PRIVATE_H_
#define _JPEG_PRIVATE_H_

#ifdef HAVE_JPEG_SUPPORT

#ifdef HAVE_JPEGLIB_H
#include <jpeglib.h>
#endif

typedef enum {
	JPEG_INFO_FORMAT_UNSPECIFIED,
	JPEG_INFO_FORMAT_BINARY,
	JPEG_INFO_FORMAT_ASCII
} jpeg_info_format;

#define N_JPEG_INFO_FORMATS	2	/* unspecified doesn't count */

extern int default_jpeg_info_format;

#ifdef INC_VERSION
char VersionId_inc_fiojpeg[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#include "img_file/img_file_hdr.h"

struct jpeg_hdr {
	/* info we get from SOF0 marker, and pass to fileio lib to fill in if_dp */
	int				jpeg_comps;
	int				jpeg_width;
	int				jpeg_height;
	int				jpeg_frames;

	u_int				jpeg_size_offset;
	size_t				jpeg_last_offset;

	seek_tbl_type *			jpeg_seek_table;

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

};

/* prototypes */


/* jpeg.c */

FIO_INTERFACE_PROTOTYPES( jpeg , Jpeg_Hdr )
FIO_INTERFACE_PROTOTYPES( lml , void )

/* jp_opts.c */
extern void install_djpeg_params(j_decompress_ptr cinfop);
//extern void jpeg_param_menu(void);
extern COMMAND_FUNC( do_djpeg_param_menu );

/* copts.c */
extern void install_cjpeg_params(j_compress_ptr cinfo);
extern void set_my_sample_factors(int hfactor[],int vfactor[]);

#endif /* HAVE_JPEG_SUPPORT */

#endif /* _JPEG_PRIVATE_H_ */

