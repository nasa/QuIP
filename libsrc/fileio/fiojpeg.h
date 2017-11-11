#ifndef _FIOJPEG_H_
#define _FIOJPEG_H_

#ifdef HAVE_JPEG_SUPPORT

#include "img_file/img_file_hdr.h"

/* prototypes */


/* jpeg.c */

FIO_INTERFACE_PROTOTYPES( jpeg , Jpeg_Hdr )
FIO_INTERFACE_PROTOTYPES( lml , void )

#define jpeg_unconv(a,b)	_jpeg_unconv(QSP_ARG  a,b)
#define jpeg_conv(a,b)		_jpeg_conv(QSP_ARG  a,b)
#define jpeg_to_dp(a,b)		_jpeg_to_dp(QSP_ARG  a,b)

/* jp_opts.c */
extern COMMAND_FUNC( do_djpeg_param_menu );

/* copts.c */
extern void set_my_sample_factors(int hfactor[],int vfactor[]);

#endif /* HAVE_JPEG_SUPPORT */

#endif /* _FIOJPEG_H_ */

