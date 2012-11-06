#include "quip_config.h"


char VersionId_newvec_sampmenu[] = QUIP_VERSION_STRING;

#include <stdio.h>

#include "menuname.h"
#include "vec_util.h"

/* local prototypes */

static COMMAND_FUNC( do_samp_image );
static COMMAND_FUNC( do_render );
static COMMAND_FUNC( do_render2 );

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

Command samp_ctbl[]={
{ "sample",	do_samp_image,	"sample intensities from an image"	},
{ "render",	do_render,	"render sample intensities into an image"},
{ "render2",	do_render2,	"render samples using bilinear interpolation"},
#ifndef MAC
{ "quit",	popcmd,		"exit warrior command menu"		},
#endif /* ! MAC */
{ NULL_COMMAND								}
};

COMMAND_FUNC( sampmenu )
{
	PUSHCMD(samp_ctbl,SAMPLE_MENU_NAME);
}



