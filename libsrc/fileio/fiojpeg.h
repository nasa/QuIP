#ifndef _FIOJPEG_H_
#define _FIOJPEG_H_

#ifdef HAVE_JPEG_SUPPORT

#include "img_file/img_file_hdr.h"

/* prototypes */


/* jpeg.c */

FIO_INTERFACE_PROTOTYPES( jpeg , Jpeg_Hdr )
FIO_INTERFACE_PROTOTYPES( lml , void )

/* jp_opts.c */
#ifdef FOOBAR
extern void install_djpeg_params(j_decompress_ptr cinfop);
#endif // FOOBAR
//extern void jpeg_param_menu(void);
extern COMMAND_FUNC( do_djpeg_param_menu );

/* copts.c */
#ifdef FOOBAR
extern void install_cjpeg_params(j_compress_ptr cinfo);
#endif // FOOBAR
extern void set_my_sample_factors(int hfactor[],int vfactor[]);

#endif /* HAVE_JPEG_SUPPORT */

#endif /* _FIOJPEG_H_ */

