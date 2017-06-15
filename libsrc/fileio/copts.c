
#include "quip_config.h"

#ifdef HAVE_JPEG_SUPPORT

#include "fio_prot.h"


#include "quip_prot.h"
#include "fiojpeg.h"
#include "jpeg_private.h"
//#include "cdjpeg.h"

int default_jpeg_info_format = JPEG_INFO_FORMAT_UNSPECIFIED;

typedef struct cjpeg_params {
	int	h_samp_factor[3];
	int	v_samp_factor[3];
	int	colorspace;
	int	quality;
} Cjpeg_Params;

static Cjpeg_Params cparams1={
	{1,1,1},
	{1,1,1},
	JCS_YCbCr,
	75
};

void set_my_sample_factors(int hfactors[3],int vfactors[3])
{
	/* BUG check valid */

	cparams1.h_samp_factor[0] = hfactors[0];
	cparams1.v_samp_factor[0] = vfactors[0];

	cparams1.h_samp_factor[1] = hfactors[1];
	cparams1.v_samp_factor[1] = vfactors[1];

	cparams1.h_samp_factor[2] = hfactors[2];
	cparams1.v_samp_factor[2] = vfactors[2];
}

static COMMAND_FUNC( do_set_sample_factors )
{
	int hfactors[3],vfactors[3];

	hfactors[0] = HOW_MANY("Y horizontal sampling factor");
	vfactors[0] = HOW_MANY("Y vertical sampling factor");
	hfactors[1] = HOW_MANY("Cr horizontal sampling factor");
	vfactors[1] = HOW_MANY("Cr vertical sampling factor");
	hfactors[2] = HOW_MANY("Cb horizontal sampling factor");
	vfactors[2] = HOW_MANY("Cb vertical sampling factor");
	/* BUG check valid */
	set_my_sample_factors(hfactors,vfactors);
}

#define N_COLORSPACES	2

static const char *colorspace_names[N_COLORSPACES]={
	"YCbCr",
	"grayscale"
};

static COMMAND_FUNC( do_set_colorspace )
{
	int n;

	n=WHICH_ONE("target color space",N_COLORSPACES,colorspace_names);
	if( n < 0 ) return;

	switch(n){
		case 0: cparams1.colorspace = JCS_YCbCr; break;
		case 1: cparams1.colorspace = JCS_GRAYSCALE; break;
		default:
			assert( ! "do_set_colorspace:  bad colorspace code");
			break;
	}
}

static COMMAND_FUNC( do_set_quality )
{
	int q;

	q=HOW_MANY("quality factor (1-99)");

	if( q < 1 || q > 99 ){
		sprintf(ERROR_STRING,
	"JPEG quality factor (%d) should be between 1 and 99",q);
		WARN(ERROR_STRING);
		return;
	}

	cparams1.quality = q;
}


void install_cjpeg_params(j_compress_ptr cinfo)
{
	int ci;
	int q_scale_factor;

	/* jpeg_set_colorspace resets the sample factors!? */ 
	/* So we need to call this before we reset them... */

	jpeg_set_colorspace(cinfo,(J_COLOR_SPACE)cparams1.colorspace);

	for(ci=0;ci<3;ci++){
		cinfo->comp_info[ci].h_samp_factor = cparams1.h_samp_factor[ci];
		cinfo->comp_info[ci].v_samp_factor = cparams1.v_samp_factor[ci];
	}

	q_scale_factor = jpeg_quality_scaling(cparams1.quality);
	// check the return value just to suppress a compiler warning...
	// The return value should be an integer representing the percentage...
	if( q_scale_factor < 0 )
		NWARN("install_cjpeg_params:  bad quality index!?");

	/* more stuff done by cjpeg after scanning options */

	jpeg_set_quality(cinfo,cparams1.quality,TRUE);
				/* TRUE means force baseline jpeg (8bit) */
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(cjpeg_menu,s,f,h)

MENU_BEGIN(cjpeg)
ADD_CMD(sample_factors,	do_set_sample_factors,	set sampling factors	)
ADD_CMD(colorspace,	do_set_colorspace,	set target color space	)
ADD_CMD(quality,	do_set_quality,		set Q-factor	)
MENU_END(cjpeg)

static COMMAND_FUNC( do_cjpeg_param_menu )
{
	PUSH_MENU(cjpeg);
}

static const char *fmt_choices[N_JPEG_INFO_FORMATS]={"binary","ascii"};

static COMMAND_FUNC( do_set_info_format )
{
	default_jpeg_info_format =  WHICH_ONE("default jpeg info format",N_JPEG_INFO_FORMATS,fmt_choices);
	if( default_jpeg_info_format < 0 )
		default_jpeg_info_format = JPEG_INFO_FORMAT_UNSPECIFIED;
	else
		default_jpeg_info_format ++;
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(jpeg_menu,s,f,h)

MENU_BEGIN(jpeg)
ADD_CMD(compressor,	do_cjpeg_param_menu,	compressor parameter submenu )
ADD_CMD(decompressor,	do_djpeg_param_menu,	decompressor parameter submenu )
ADD_CMD(info_format,	do_set_info_format,	select default jpeg info format )
MENU_END(jpeg)

COMMAND_FUNC( do_jpeg_menu )
{
	PUSH_MENU(jpeg);
}

#endif /* HAVE_JPEG_SUPPORT */

