
#ifdef HAVE_JPEG_SUPPORT

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

#include "img_file_hdr.h"

/* prototypes */


/* jpeg.c */

FIO_INTERFACE_PROTOTYPES( jpeg , Jpeg_Hdr )
FIO_INTERFACE_PROTOTYPES( lml , void )

/* jp_opts.c */
extern void install_djpeg_params(j_decompress_ptr cinfop);
extern void jpeg_param_menu(void);

/* copts.c */
extern void install_cjpeg_params(j_compress_ptr cinfo);
extern void set_my_sample_factors(int hfactor[],int vfactor[]);

#endif /* HAVE_JPEG_SUPPORT */
