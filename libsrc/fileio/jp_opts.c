
#include "quip_config.h"

#ifdef HAVE_JPEG_SUPPORT

#include <stdio.h>

#ifdef HAVE_JPEGLIB_H
#include <jpeglib.h>
#endif


//#include "jversion.h"

#include "quip_prot.h"

#include "cdjpeg.h"		/* read_color_map() */

#include "fio_prot.h"
#include "fiojpeg.h"
#include "jpeg_private.h"

typedef struct djpeg_params {
	int	djp_trace_level;
	int	djp_desired_number_of_colors;
	int	djp_quantize_colors;
	int	djp_dct_method;
	int	djp_dither_mode;
	int	djp_two_pass_quantize;
	int	djp_do_fancy_upsampling;
	int	djp_out_color_space;
	long	djp_max_memory_to_use;
	char *	djp_mapfilename;
	int	djp_scale_num;
	int	djp_scale_denom;
} dJPEG_Params;

dJPEG_Params djp_p1;
static int defaults_initialized=0;

static void set_djpeg_defaults(void)
{
	/* Set up default JPEG parameters. */

	djp_p1.djp_trace_level = 0;
	djp_p1.djp_desired_number_of_colors = 0;
	djp_p1.djp_quantize_colors = FALSE;
	djp_p1.djp_dct_method = JDCT_FLOAT;
	djp_p1.djp_dither_mode = JDITHER_NONE;
	djp_p1.djp_two_pass_quantize = TRUE;
	djp_p1.djp_do_fancy_upsampling = TRUE;
	djp_p1.djp_out_color_space = JCS_RGB;
	djp_p1.djp_max_memory_to_use = 4096L * 1024L;	/* 4Mb */
	djp_p1.djp_mapfilename = NULL;
	djp_p1.djp_scale_num = 1;
	djp_p1.djp_scale_denom = 1;
}

static COMMAND_FUNC( set_ncolors )
{
	int val;

	val=HOW_MANY("number of colors on output");
	/* BUG check for valid */

	djp_p1.djp_desired_number_of_colors = val;
	djp_p1.djp_quantize_colors = TRUE;
}

#define N_DCT_METHODS	3

static const char *dct_method_list[N_DCT_METHODS]={
	"int",
	"fast",
	"float"
};

static COMMAND_FUNC( set_dct )
{
	int i;

	i=WHICH_ONE("DCT method",N_DCT_METHODS,dct_method_list);
	if( i < 0 ) return;
	switch(i){
		case 0: djp_p1.djp_dct_method = JDCT_ISLOW; break;
		case 1: djp_p1.djp_dct_method = JDCT_IFAST; break;
		case 2: djp_p1.djp_dct_method = JDCT_FLOAT; break;
	}
}

#define N_DITHER_METHODS	3

static const char *dither_method_list[N_DITHER_METHODS]={
	"fs",
	"none",
	"ordered"
};

static COMMAND_FUNC( set_dither )
{
	int i;

	i=WHICH_ONE("dithering method",N_DITHER_METHODS,dither_method_list);
	if( i < 0 ) return;
	switch(i){
		case 0: djp_p1.djp_dither_mode = JDITHER_FS; break;
		case 1: djp_p1.djp_dither_mode = JDITHER_NONE; break;
		case 2: djp_p1.djp_dither_mode = JDITHER_ORDERED; break;
	}
}

static COMMAND_FUNC( report_version )
{
	sprintf(ERROR_STRING, "JPEG library version %d%c\n",
		JPEG_LIB_VERSION/10, 'a'-1+JPEG_LIB_VERSION%10);
	advise(ERROR_STRING);
}

static COMMAND_FUNC( set_fast )
{
	if( ASKIF("Set processing options for quick-and-dirty output") ){
		djp_p1.djp_two_pass_quantize = FALSE;
		djp_p1.djp_dither_mode = JDITHER_ORDERED;
			if (! djp_p1.djp_quantize_colors) /* don't override an earlier -colors */
		djp_p1.djp_desired_number_of_colors = 216;
		djp_p1.djp_dct_method = JDCT_FASTEST;
		djp_p1.djp_do_fancy_upsampling = FALSE;
	}
}

static COMMAND_FUNC( set_grayscale )
{
	if( ASKIF("force monochrome output") )
		djp_p1.djp_out_color_space = JCS_GRAYSCALE;
}

static COMMAND_FUNC( set_map )
{
#ifdef QUANT_2PASS_SUPPORTED	/* otherwise can't quantize to supplied map */
	char *filename;

	filename = NAMEOF("color map file");

	if( djp_p1.djp_mapfilename != NULL )
		rls_str(djp_p1.djp_mapfilename);
	djp_p1.djp_mapfilename = savestr(filename);
#else
	WARN("Sorry, support for color quantization not included");
#endif
}

static COMMAND_FUNC( set_maxmem )
{
	long lval;

	/* djpeg allows 10M etc */
	lval = HOW_MANY("maximum amount of memory to use (in kilobytes)");
	/* BUG check for valid value */
	djp_p1.djp_max_memory_to_use = lval * 1000L;
}

static COMMAND_FUNC( set_nosmooth )
{
	if( ASKIF("Suppress fancy upsampling") )
		djp_p1.djp_do_fancy_upsampling = FALSE;
	else
		djp_p1.djp_do_fancy_upsampling = TRUE;
}

static COMMAND_FUNC( set_onepass )
{
	if( ASKIF("use fast one-pass quantization") )
		djp_p1.djp_two_pass_quantize = FALSE;
	else
		djp_p1.djp_two_pass_quantize = TRUE;
}

#ifdef IDCT_SCALING_SUPPORTED
static COMMAND_FUNC( set_scale )
{
	int num,denom;

	num=HOW_MANY("numerator");
	denom=HOW_MANY("denominator");
	/* check for valid values here BUG */
	djp_p1.djp_scale_num = num;
	djp_p1.djp_scale_denom = denom;
}
#endif

static COMMAND_FUNC( do_prm_display )
{
	const char *s;

	sprintf(msg_str,"\ttrace_level:\t%d",djp_p1.djp_trace_level);
	prt_msg(msg_str);

	sprintf(msg_str,"\tncolors:\t%d",djp_p1.djp_desired_number_of_colors);
	prt_msg(msg_str);

	sprintf(msg_str,"\tquantize_colors:\t%s",
		(djp_p1.djp_quantize_colors==TRUE)?"TRUE":"FALSE" );
	prt_msg(msg_str);

	prt_msg_frag("\tdct_method:\t");
	switch(djp_p1.djp_dct_method){
		case JDCT_ISLOW: s="slow integer"; break;
		case JDCT_IFAST: s="fast integer"; break;
		case JDCT_FLOAT: s="float"; break;
		default:
			s=NULL;		/* elim compiler warning */
			assert( ! "bad dct_method code" );
			break;
	}
	prt_msg(s);

	prt_msg_frag("\tdither_mode:\t");
	switch(djp_p1.djp_dither_mode){
		case JDITHER_FS: s="Floyd-Steinberg"; break;
		case JDITHER_NONE: s="(none)"; break;
		case JDITHER_ORDERED: s="ordered"; break;
		default:
			assert( ! "bad dither_mode code!?");
			break;
	}
	prt_msg(s);

	sprintf(msg_str,"\ttwo_pass_quantize:\t%s",
		(djp_p1.djp_two_pass_quantize==TRUE)?"TRUE":"FALSE" );
	prt_msg(msg_str);

	sprintf(msg_str,"\tdo_fancy_upsampling:\t%s",
		(djp_p1.djp_do_fancy_upsampling==TRUE)?"TRUE":"FALSE" );
	prt_msg(msg_str);

	prt_msg_frag("\tout_color_space:\t");
	switch(djp_p1.djp_out_color_space){
		case JCS_RGB: s="RGB"; break;
		case JCS_GRAYSCALE: s="grayscale"; break;
		default:
			assert( ! "bad out_color_space code!?");
			break;
	}
	prt_msg(s);

	sprintf(msg_str,"\tmax_memory_to_use:\t%ld (0x%lx)",
		djp_p1.djp_max_memory_to_use,
		djp_p1.djp_max_memory_to_use);
	prt_msg(msg_str);

	sprintf(msg_str,"\tscaling:\t\t%d/%d",djp_p1.djp_scale_num,djp_p1.djp_scale_denom);
	prt_msg(msg_str);
}

static COMMAND_FUNC( do_set_djpeg_defaults ){ set_djpeg_defaults(); }

#define ADD_CMD(s,f,h)	ADD_COMMAND(djpeg_param_menu,s,f,h)

MENU_BEGIN(djpeg_param)
ADD_CMD(	defaults,	do_set_djpeg_defaults,	set default parameters )
#ifdef IDCT_SCALING_SUPPORTED
ADD_CMD(	scale,		set_scale,	scale output image by fraction M/N )
#endif /* IDCT_SCALING_SUPPORTED */
ADD_CMD(	ncolors,	set_ncolors,	set number of output colors )
ADD_CMD(	set_dct,	set_dct,	select DCT algorithm )
ADD_CMD(	set_dither,	set_dither,	select dither algorithm )
ADD_CMD(	fast,		set_fast,	enable fast algorithm )
ADD_CMD(	grayscale,	set_grayscale,	set grayscale output )
ADD_CMD(	set_map,	set_map,	quantize output using colormap )
ADD_CMD(	maxmem,		set_maxmem,	specify max mem usage )
ADD_CMD(	nosmooth,	set_nosmooth,	suppress fancy upsampling )
ADD_CMD(	onepass,	set_onepass,	enable fast one-pass quantization )
ADD_CMD(	display,	do_prm_display,	display current parameter settings )
ADD_CMD(	jpeg_version,	report_version,	report version of JPEG library )
MENU_END(djpeg_param)

COMMAND_FUNC( do_djpeg_param_menu )
{
	if( ! defaults_initialized )
		set_djpeg_defaults();
	PUSH_MENU(djpeg_param);
}

void install_djpeg_params(j_decompress_ptr cinfop)
{
	if( ! defaults_initialized )
		set_djpeg_defaults();

	cinfop->err->trace_level = djp_p1.djp_trace_level;
	cinfop->desired_number_of_colors = djp_p1.djp_desired_number_of_colors;
	cinfop->quantize_colors = djp_p1.djp_quantize_colors;
	cinfop->dct_method = (J_DCT_METHOD) djp_p1.djp_dct_method;
	cinfop->dither_mode = (J_DITHER_MODE) djp_p1.djp_dither_mode;
	cinfop->two_pass_quantize = djp_p1.djp_two_pass_quantize;
	cinfop->do_fancy_upsampling = djp_p1.djp_do_fancy_upsampling;
	cinfop->out_color_space = (J_COLOR_SPACE) djp_p1.djp_out_color_space;
	cinfop->mem->max_memory_to_use = djp_p1.djp_max_memory_to_use;

#ifdef QUANT_2PASS_SUPPORTED	/* otherwise can't quantize to supplied map */
	if( djp_p1.djp_mapfilename != NULL ){
		FILE *fp;
		if ((fp = fopen(djp_p1.djp_mapfilename, "rb")) == NULL) {
			sprintf(ERROR_STRING, "set_map: can't open %s\n", djp_p1.djp_mapfilename);
			WARN(ERROR_STRING);
			cinfop->quantize_colors = FALSE;
			rls_str(djp_p1.djp_mapfilename);
			djp_p1.djp_mapfilename = NULL;
		} else {
			read_color_map(cinfop,fp);
			fclose(fp);
			cinfop->quantize_colors = TRUE;
		}
	}
#endif /* QUANT_2PASS_SUPPORTED */

	cinfop->scale_num = djp_p1.djp_scale_num;
	cinfop->scale_denom = djp_p1.djp_scale_denom;
}

#endif /* HAVE_JPEG_SUPPORT */

