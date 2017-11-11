

#ifndef _IMG_FILE_HDR_H_
#define _IMG_FILE_HDR_H_

#include "quip_config.h"

// Is there a reason why this is first?  We're having some
// compile difficulties...
// It has to be included before tiff.h, because both use
// COMPRESSION_NONE - it's an enum in matio.h, and a macro
// in tiff.h...

#ifdef FOOBAR
#ifdef HAVE_MATIO
#include <matio.h>
#endif /* HAVE_MATIO */
#endif // FOOBAR

#include "vistahdr.h"
#include "hips/hip1hdr.h"
#include "hips/hip2hdr.h"
#include "wav_hdr.h"
#include "bmp_hdr.h"
#include "bdf_hdr.h"
#include "rv_api.h"
#include "matio_hdr.h"

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

#ifdef HAVE_TIFF
#include "tiff_hdr.h"
#endif /* HAVE_TIFF */

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

//#include "glimage.h"

#endif /* ! _IMG_FILE_HDR_H_ */
