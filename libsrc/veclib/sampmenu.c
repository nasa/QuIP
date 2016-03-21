#include "quip_config.h"

#include <stdio.h>

#include "quip_prot.h"
#include "vec_util.h"
#include "nvf.h"

static COMMAND_FUNC( do_samp_image )
{
	Data_Obj *intens_dp, *image_dp, *coord_dp;

	intens_dp=PICK_OBJ( "target list for sampled intensities" );
	image_dp=PICK_OBJ( "source image for intensities" );
	coord_dp=PICK_OBJ( "coordinate list" );
	if( intens_dp==NO_OBJ || image_dp==NO_OBJ || coord_dp==NO_OBJ )
		return;

	sample_image(QSP_ARG  intens_dp,image_dp,coord_dp);
}

static COMMAND_FUNC( do_render )
{
	Data_Obj *intens_dp, *image_dp, *coord_dp;

	image_dp=PICK_OBJ( "target image" );
	intens_dp=PICK_OBJ( "source list of sampled intensities" );
	coord_dp=PICK_OBJ( "coordinate list" );
	if( intens_dp==NO_OBJ || image_dp==NO_OBJ || coord_dp==NO_OBJ )
		return;

	render_samples(QSP_ARG  image_dp,coord_dp,intens_dp);
}

static COMMAND_FUNC( do_render2 )
{
	Data_Obj *intens_dp, *image_dp, *coord_dp;

	image_dp=PICK_OBJ( "target image" );
	intens_dp=PICK_OBJ( "source list of sampled intensities" );
	coord_dp=PICK_OBJ( "coordinate list" );
	if( intens_dp==NO_OBJ || image_dp==NO_OBJ || coord_dp==NO_OBJ )
		return;

	render_samples2(QSP_ARG  image_dp,coord_dp,intens_dp);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(sample_menu,s,f,h)

MENU_BEGIN(sample)
ADD_CMD( sample,	do_samp_image,	sample intensities from an image )
ADD_CMD( render,	do_render,	render sample intensities into an image )
ADD_CMD( render2,	do_render2,	render samples using bilinear interpolation )
MENU_END(sample)

COMMAND_FUNC( do_samp_menu )
{
	PUSH_MENU(sample);
}



