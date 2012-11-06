
#include "quip_config.h"

char VersionId_fio_copts[] = QUIP_VERSION_STRING;

#ifdef HAVE_JPEG_SUPPORT

#include "fio_prot.h"


#include "fiojpeg.h"
#include "cdjpeg.h"

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
#ifdef CAUTIOUS
		default:  ERROR1("CAUTIOUS:  do_set_colorspace");
#endif /* CAUTIOUS */
	}
}

static COMMAND_FUNC( do_set_quality )
{
	int q;

	q=HOW_MANY("quality factor (1-99)");

	if( q < 1 || q > 99 ){
		sprintf(error_string,
	"JPEG quality factor (%d) should be between 1 and 99",q);
		WARN(error_string);
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

	/* more stuff done by cjpeg after scanning options */

	jpeg_set_quality(cinfo,cparams1.quality,TRUE);
				/* TRUE means force baseline jpeg (8bit) */
}

static Command cjpeg_ctbl[]={
{ "sample_factors",	do_set_sample_factors,	"set sampling factors"	},
{ "colorspace",		do_set_colorspace,	"set target color space" },
{ "quality",		do_set_quality,		"set Q-factor"		},
{ "quit",		popcmd,			"exit submenu"		},
{ NULL_COMMAND								}
};

COMMAND_FUNC( cjpeg_param_menu )
{
	PUSHCMD(cjpeg_ctbl,"jpeg_compress");
}

extern COMMAND_FUNC( djpeg_param_menu );

static const char *fmt_choices[N_JPEG_INFO_FORMATS]={"binary","ascii"};

static COMMAND_FUNC( do_set_info_format )
{
	default_jpeg_info_format =  WHICH_ONE("default jpeg info format",N_JPEG_INFO_FORMATS,fmt_choices);
	if( default_jpeg_info_format < 0 )
		default_jpeg_info_format = JPEG_INFO_FORMAT_UNSPECIFIED;
	else
		default_jpeg_info_format ++;
}

static Command jpeg_ctbl[]={
{ "compressor",		cjpeg_param_menu,	"compressor parameter submenu"		},
{ "decompressor",	djpeg_param_menu,	"decompressor parameter submenu"	},
{ "info_format",	do_set_info_format,	"select default jpeg info format"	},
{ "quit",		popcmd,			"exit submenu"				},
{ NULL_COMMAND										}
};

COMMAND_FUNC( jpeg_menu )
{
	PUSHCMD(jpeg_ctbl,"jpeg");
}

#endif /* HAVE_JPEG_SUPPORT */

