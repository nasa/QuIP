

#ifndef _IMG_FILE_HDR_H_
#define _IMG_FILE_HDR_H_

#include "quip_config.h"

// Is there a reason why this is first?  We're having some
// compile difficulties...
// It has to be included before tiff.h, because both use
// COMPRESSION_NONE - it's an enum in matio.h, and a macro
// in tiff.h...
// Now we have another conflict, EXTERN defined in matio.h
// and jmorecfg.h (libjpeg)
// WHAT DO WE DO???

#ifdef HAVE_MATIO
#include <matio.h>
#endif /* HAVE_MATIO */

// This is a kludge!
#ifdef EXTERN
#undef EXTERN
#endif /* EXTERN */

#include "vistahdr.h"
#include "hip1hdr.h"
#include "hip2hdr.h"
#include "wav_hdr.h"
#include "bmp_hdr.h"
#include "bdf_hdr.h"
#include "rv_api.h"

#ifdef HAVE_LIBAVCODEC
#include "avcodec_hdr.h"
#endif /* HAVE_LIBAVCODEC */

#ifdef HAVE_MPLAYER
#include "mplayer_hdr.h"
#endif /* HAVE_XINE */

#ifdef HAVE_XINE
#include "xine_hdr.h"
#endif /* HAVE_XINE */

#include "ppmhdr.h"
#include "lshdr.h"
#include "sunras.h"
#include "sir_disk.h"
//#include "rawvol.h"
//#include "data_obj.h"

#ifdef HAVE_QUICKTIME
#include "qt_hdr.h"
#endif

#ifdef HAVE_JPEG_SUPPORT
#include "jpeg_hdr.h"
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_PNG
#include <png.h>		// library header file
#include "png_hdr.h"
#endif /* HAVE_PNG */

#ifdef HAVE_MPEG
#include "mpeg.h"
#include "mpege.h"
#include "mpege_im.h"
#include "mpeg_hdr.h"
#endif /* HAVE_MPEG */

#include "nports_api.h"

#include "glimage.h"

typedef struct image_file_hdr {
	union {
		Vista_Hdr vista_hd;
		Hips1_Header hips1_hd;
		Hips2_Header hips2_hd;
		Ppm_Header ppm_hd;
		Dis_Header dis_hd;
		struct rasterfile rf_hd;
		//struct xvimage xvi_hd;
#ifdef HAVE_RGB
		/* was this in glimage.h? */
		IMAGE rgb_hd;
#endif
		Sir_Disk_Hdr sir_dsk_hd;
		RV_Inode rv_ino;
		Lumisys_Hdr ls_hd;
		Wav_Header wav_hd;
#ifdef HAVE_JPEG_SUPPORT
#ifndef OMIT_JPEG
		Jpeg_Hdr jpeg_hd;
#endif /* OMIT_JPEG */
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_PNG
		Png_Hdr png_hd;
#endif /* HAVE_PNG */

#ifdef HAVE_LIBAVCODEC
		AVCodec_Hdr avc_hd;
#endif /* HAVE_LIBAVCODEC */

#ifdef HAVE_MPEG
		Mpeg_Hdr mpeg_hd;
#endif /* HAVE_MPEG */

#ifdef HAVE_QUICKTIME
		Qt_Hdr 		qt_hd;
#endif
		BMP_Header	bmp_hd;
		BDF_info	bdf_info;
#ifdef HAVE_MATIO
		matvar_t	matvar_hd;
#endif /* HAVE_MATIO */

#ifdef FOOBAR
		Vista_Hdr *vista_hd_p;
		Hips1_Header *hips1_hd_p;
		Hips2_Header *hips2_hd_p;
		Ppm_Header *ppm_hd_p;
		Dis_Header *dis_hd_p;
		struct rasterfile *rf_p;
		struct xvimage *xvi_p;
#ifdef HAVE_RGB
		/* was this in glimage.h? */
		IMAGE *rgb_ip;
#endif
		Sir_Disk_Hdr *sir_dsk_hd_p;
		RV_Inode *rv_inp;
		Lumisys_Hdr *ls_hd_p;
		Wav_Header *wav_hd_p;
#ifdef HAVE_JPEG_SUPPORT
#ifndef OMIT_JPEG
		Jpeg_Hdr *jpeg_hd_p;
#endif /* OMIT_JPEG */
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_PNG
		Png_Hdr *png_hd_p;
#endif /* HAVE_PNG */

#ifdef HAVE_LIBAVCODEC
		AVCodec_Hdr *avc_hd_p;
#endif /* HAVE_LIBAVCODEC */

#ifdef HAVE_MPEG
		Mpeg_Hdr *mpeg_hd_p;
#endif /* HAVE_MPEG */

#ifdef HAVE_QUICKTIME
		Qt_Hdr *	qt_hd_p;
#endif
		BMP_Header *	bmp_hd_p;
		BDF_info *	bdf_info_p;
#ifdef HAVE_MATIO
		matvar_t *	matvar_p;
#endif /* HAVE_MATIO */

#endif /* FOOBAR */
	} ifh_u;
} Image_File_Hdr;

#define NO_IMAGE_FILE_HDR		((Image_File_Hdr*) NULL )

#endif /* ! _IMG_FILE_HDR_H_ */
