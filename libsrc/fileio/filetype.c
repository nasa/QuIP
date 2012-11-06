
#include "quip_config.h"

char VersionId_fio_filetype[] = QUIP_VERSION_STRING;


#include <stdio.h>

#include "fio_prot.h"
#include "debug.h"
#include "data_obj.h"
#include "rgb.h"
#include "hips1.h"
#include "hips2.h"
#include "wav.h"
#include "raw.h"
#include "vl.h"
#include "vista.h"
#include "jbm_ppm.h"
#include "rv.h"
#include "lumisys.h"
#include "fiojpeg.h"
#include "fio_png.h"
#include "fio_mpeg.h"
#include "sunras.h"
#include "bmp.h"
#include "bdf.h"

#if defined(HAVE_LIBAVCODEC)
#include "my_avi.h"
#endif /* HAVE_LIBAVCODEC */

#ifdef HAVE_MATIO
#include "matio_api.h"
#endif /* HAVE_MATIO */

#include "fioasc.h"

#ifdef HAVE_TIFF
#include "fiotiff.h"
#endif /* HAVE_TIFF */

#ifdef HAVE_KHOROS
#include "my_viff.h"
#endif /* HAVE_KHOROS */

#ifdef HAVE_QUICKTIME
#include "fioqt.h"
#endif

static FIO_WT_FUNC( dummy_wt )
{return(0);}

static void null_info(QSP_ARG_DECL  Image_File *ifp)
{}

static int null_conv(Data_Obj *dp,void *vp)
{ return(-1); }

static int null_unconv(void *vp,Data_Obj *dp)
{ return(-1); }

static FIO_RD_FUNC( null_rd ) {}
static FIO_OPEN_FUNC( null_open ) { return(NO_IMAGE_FILE); }
static FIO_CLOSE_FUNC( null_close ) {}

static int null_seek(QSP_ARG_DECL  Image_File *ifp, dimension_t pos)
{
	sprintf(error_string,"Sorry, can't seek on file %s",ifp->if_name);
	WARN(error_string);
	return(-1);
}

#include "filetype.h"

Filetype ft_tbl[N_FILETYPE]={
{
	"network",
	null_open,
	null_rd,

	dummy_wt,
	null_close,
	null_unconv,
	null_conv,
	null_info,
	null_seek,
	0,
	IFT_NETWORK
},
{
	"raw",			raw_open,		raw_rd,
	raw_wt,			generic_imgfile_close,	raw_unconv,
	raw_conv,		null_info,		uio_seek,
	USE_UNIX_IO|CAN_DO_FORMAT,
	IFT_RAW
},
{
	"hips1",		hips1_open,		raw_rd,
	hips1_wt,		hips1_close,		hips1_unconv,
	hips1_conv,		null_info,		uio_seek,
	USE_UNIX_IO|CAN_DO_FORMAT,
	IFT_HIPS1
},
{
	"hips2",		hips2_open,		hips2_rd,
	hips2_wt,		hips2_close,		hips2_unconv,
	hips2_conv,		null_info,		std_seek,
	CAN_DO_FORMAT|USE_STDIO,
	IFT_HIPS2
},
{
	"sunraster",		sunras_open,		sunras_rd,
	dummy_wt,		sunras_close,		sunras_unconv,
	sunras_conv,		null_info,		std_seek,
	CAN_READ_FORMAT|USE_STDIO,
	IFT_SUNRAS
},
{
	"ppm",			ppm_open,		raw_rd,
	ppm_wt,			ppm_close,		ppm_unconv,
	ppm_conv,		null_info,		std_seek,
	CAN_DO_FORMAT|USE_STDIO,
	IFT_PPM
},
{
	"dis",			dis_open,		raw_rd,
	dis_wt,			dis_close,		dis_unconv,
	dis_conv,		null_info,		std_seek,
	CAN_DO_FORMAT|USE_STDIO,
	IFT_DIS
},
{
	"vista",		vista_open,		raw_rd,
	vista_wt,		generic_imgfile_close,	vista_unconv,
	vista_conv,		null_info,		std_seek,
	CAN_DO_FORMAT|USE_STDIO,
	IFT_VISTA
},
{
	"VL",			vl_open,		raw_rd,
	dummy_wt,		vl_close,		vl_unconv,
	vl_conv,		null_info,		std_seek,
	CAN_READ_FORMAT|USE_STDIO,
	IFT_VL
},
#ifdef HAVE_RGB
{
	"rgb",			rgb_open,		rgb_rd,
	rgb_wt,			rgb_close,		rgb_unconv,
	rgb_conv,		null_info,		std_seek,
	CAN_DO_FORMAT|USE_STDIO,
	IFT_RGB
},
#endif /* HAVE_RGB */
{
	"disk",			dsk_open,		dsk_rd,
	dsk_wt,			generic_imgfile_close,	dsk_unconv,
	dsk_conv,		dsk_info,		uio_seek,
	USE_UNIX_IO|CAN_DO_FORMAT,
	IFT_DISK
},
{
	"rv",			rvfio_open,		rvfio_rd,
	rvfio_wt,		rvfio_close,		rvfio_unconv,
	rvfio_conv,		rvfio_info,		rvfio_seek_frame,
	CAN_DO_FORMAT,
	IFT_RV
},
{
	"lumisys",		ls_open,		raw_rd,
	dummy_wt,		ls_close,		ls_unconv,
	ls_conv,		null_info,		uio_seek,
	USE_UNIX_IO|CAN_READ_FORMAT,
	IFT_LUM
},
{
	"wav",			wav_open,		wav_rd,
	wav_wt,			wav_close,		wav_unconv,
	wav_conv,		wav_info,		std_seek,
	USE_STDIO|CAN_DO_FORMAT,
	IFT_WAV
},

{
	"bmp",			bmp_open,		bmp_rd,
	dummy_wt,		bmp_close,		bmp_unconv,
	bmp_conv,		bmp_info,		std_seek,
	CAN_READ_FORMAT|USE_STDIO,
	IFT_BMP
},

{
	"ascii",
	asc_open,
	asc_rd,

	asc_wt,
	generic_imgfile_close,
	asc_unconv,
	asc_conv,
	null_info,
	std_seek,
	USE_STDIO|CAN_DO_FORMAT,
	IFT_ASC
},

{
	"bdf",
	bdf_open,
	bdf_rd,
	bdf_wt,
	generic_imgfile_close,
	bdf_unconv,
	bdf_conv,
	null_info,
	std_seek,
	USE_STDIO|CAN_DO_FORMAT,
	IFT_BDF
},

/* These next two should all be mutually exclusive... */
#if defined(HAVE_LIBAVCODEC)

{
	"avi",
	avi_open,
	avi_rd,
	/*avi_wt*/dummy_wt,
	generic_imgfile_close,
	avi_unconv,
	avi_conv,
	null_info,
	avi_seek_frame,
	CAN_READ_FORMAT,
	IFT_AVI
},
#endif /* HAVE_LIBAVCODEC */

#ifdef HAVE_MATIO
{
	"matlab",
	mat_open,
	mat_rd,
	mat_wt,
	mat_close,
	mat_unconv,
	mat_conv,
	null_info,
	null_seek,
	CAN_DO_FORMAT,
	IFT_MATLAB
},
#endif /* HAVE_MATIO */

#ifdef HAVE_JPEG_SUPPORT

/* The LML type is jpeg, but with the sampling factors and sizes forced to match
 * the lml33 board requirements when we write files.
 * We have a problem using this filetype for i/o in the lml utility
 * because the jpeg library uses stdio, but the lml hardware handlers
 * want to use read/write with fd's...
 */

{
	"lml",			lml_open,		jpeg_rd,
	lml_wt,			jpeg_close,		jpeg_unconv,
	jpeg_conv,		lml_info,		jpeg_seek_frame,
	CAN_DO_FORMAT|USE_STDIO,
	IFT_LML
},
{
	"jpeg",			jpeg_open,		jpeg_rd,
	jpeg_wt,		jpeg_close,		jpeg_unconv,
	jpeg_conv,		null_info,		jpeg_seek_frame,
	CAN_DO_FORMAT|USE_STDIO,
	IFT_JPEG
},
#endif /* HAVE_JPEG_SUPPORT */

#ifdef HAVE_MPEG
{
	"mpeg",			mpeg_open,		mpeg_rd,
	mpeg_wt,		mpeg_close,		mpeg_unconv,
	mpeg_conv,		report_mpeg_info,	null_seek,
	CAN_DO_FORMAT|USE_STDIO,
	IFT_MPEG
},
#endif /* HAVE_MPEG */

#ifdef HAVE_TIFF
{
	"tiff",
	tiff_open,
	tiff_rd,
	tiff_wt,
	tiff_close,
	tiff_unconv,
	tiff_conv,
	tiff_info,
	null_seek,
	CAN_DO_FORMAT,
	IFT_TIFF
},
#endif /* HAVE_TIFF */

#ifdef HAVE_PNG
{
	"png",			pngfio_open,	pngfio_rd,
	pngfio_wt,		pngfio_close,	pngfio_unconv,
	pngfio_conv,		pngfio_info,	std_seek,
	CAN_DO_FORMAT|USE_STDIO,
	IFT_PNG
},
#endif /* HAVE_PNG */

#ifdef HAVE_KHOROS
{
	"viff",			viff_open,		viff_rd,
	viff_wt,		viff_close,		viff_unconv,
	viff_conv,		null_info,		uio_seek,
	USE_UNIX_IO|CAN_DO_FORMAT,
	IFT_VIFF
},
#endif /* HAVE_KHOROS */

#ifdef HAVE_QUICKTIME
{
	"quicktime",		qt_open,		qt_rd,
	qt_wt,			qt_close,		qt_unconv,
	qt_conv,		qt_info,		null_seek,
	CAN_READ_FORMAT,
	IFT_QT
},
#endif /* HAVE_QUICKTIME */

};


#ifdef CAUTIOUS

/*
 * check the file type table to make sure the types match the indices
 */

void ft_tbl_check(SINGLE_QSP_ARG_DECL)
{
	int i;
	static int checked=0;


	if( checked ){
		NWARN("filetype table already checked");
		return;
	}

	for(i=0;i<N_FILETYPE;i++){
		if( ft_tbl[i].ft_type != i ){
			sprintf(ERROR_STRING,
				"File type table entry %d has type code %d (should be equal)",
				i,ft_tbl[i].ft_type);
			NWARN(ERROR_STRING);
		}
	}
	checked++;
}

#endif /* CAUTIOUS */
