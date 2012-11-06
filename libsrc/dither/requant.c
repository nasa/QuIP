#include "quip_config.h"

char VersionId_dither_requant[] = QUIP_VERSION_STRING;

#include "vec_util.h"
#include "requant.h"
#include "ctone.h"
#include "menuname.h"

debug_flag_t spread_debug=0;

static COMMAND_FUNC( do_scan2_requant )
{
	int n;

	n=HOW_MANY("number of passes");
	scan2_requant(QSP_ARG  n);
}

static COMMAND_FUNC( do_scan2_requant3d )
{
	int n;

	n=HOW_MANY("number of passes");
	scan2_requant3d(n);
}

static COMMAND_FUNC( do_scan_requant )
{
	int n;

	n=HOW_MANY("number of passes");
	scan_requant(n);
}

static COMMAND_FUNC( do_anneal )
{
	double temp;
	int n;

	temp = HOW_MUCH("temperature");
	n=HOW_MANY("number of passes");
	scan_anneal(temp,n);
}

static COMMAND_FUNC( do_scan_requant3d )
{
	int n;

	n=HOW_MANY("number of passes");
	scan_requant3d(n);
}

static COMMAND_FUNC( do_anneal3d )
{
	double temp;
	int n;

	temp = HOW_MUCH("temperature");
	n=HOW_MANY("number of passes");
	scan_anneal3d(temp,n);
}

static COMMAND_FUNC( do_dich_anneal )
{
	double temp1,temp2;
	u_long n;

	n=HOW_MANY("number of passes");
	temp1=HOW_MUCH("initial temperature");
	temp2=HOW_MUCH("final temperature");
	dich_scan_anneal(n,temp1,temp2);
}

static const char *scanlist[]={"raster","scattered","random"};

static COMMAND_FUNC( do_pickscan )
{
	switch( WHICH_ONE("type of scanning pattern",3,scanlist) ){
		case 0: scan_func=get_xy_raster_point; break;
		case 1: scan_func=get_xy_scattered_point; break;
		case 2: scan_func=get_xy_random_point; break;
	}
}

static COMMAND_FUNC( do_pickscan3d )
{
	switch( WHICH_ONE("type of scanning pattern",3,scanlist) ){
		case 0: scan_func3d=get_3d_raster_point; break;
		case 1: scan_func3d=get_3d_scattered_point; break;
		case 2: scan_func3d=get_3d_random_point; break;
	}
}

static COMMAND_FUNC( do_set_input )
{
	Data_Obj *gdp;

	gdp = PICK_OBJ( "source image" );
	if( gdp == NO_OBJ ) return;
	set_grayscale(gdp);
}

static COMMAND_FUNC( do_set_input3d )
{
	Data_Obj *gdp;

	gdp = PICK_OBJ( "source image" );
	if( gdp == NO_OBJ ) return;
	set_grayscale3d(gdp);
}



static COMMAND_FUNC( do_set_output )
{
	Data_Obj *hdp;

	hdp = PICK_OBJ( "output image" );
	if( hdp == NO_OBJ ) return;
	set_halftone(hdp);
}

static COMMAND_FUNC( do_set_filter )
{
	Data_Obj *fdp;

	fdp = PICK_OBJ( "filter image" );
	if( fdp == NO_OBJ ) return;
	set_filter(fdp);
}

static COMMAND_FUNC( do_set_output3d )
{
	Data_Obj *hdp;

	hdp = PICK_OBJ( "output image" );
	if( hdp == NO_OBJ ) return;
	set_halftone3d(hdp);
}

static COMMAND_FUNC( do_set_filter3d )
{
	Data_Obj *fdp;

	fdp = PICK_OBJ( "filter image" );
	if( fdp == NO_OBJ ) return;
	set_filter3d(fdp);
}

static COMMAND_FUNC( do_init_requant )
{
	if( setup_requantize(SINGLE_QSP_ARG) == -1 ) return;
	init_requant();
}

static COMMAND_FUNC( do_init_requant3d )
{
	if( setup_requantize3d(SINGLE_QSP_ARG) == -1 ) return;
	init_requant3d(SINGLE_QSP_ARG);
}

#ifdef FOOBAR
static COMMAND_FUNC( do_qt_dither )
{
	Data_Obj *dpto, *dpfr;

	dpto = PICK_OBJ( "target image" );
	dpfr = PICK_OBJ( "source image" );

	if( dpto == NO_OBJ || dpfr == NO_OBJ ) return;

	qt_dither(dpto,dpfr);
}
#endif /* FOOBAR */

static COMMAND_FUNC( do_tweak )
{
	int x,y;

	x=HOW_MANY("x coord");
	y=HOW_MANY("y coord");

	redo_two_pixels(x,y);
}
static COMMAND_FUNC( do_tweak3d )
{
	dimension_t posn[N_DIMENSIONS];

	posn[0]=posn[4]=0;

	posn[1] = HOW_MANY("x coord");
	posn[2] = HOW_MANY("y coord");
	posn[3] = HOW_MANY("t coord");

	redo_two_pixels3d(posn);
}

static COMMAND_FUNC( do_set_clr_input )
{
	Data_Obj *lumdp, *rgdp, *bydp;

	lumdp = PICK_OBJ( "luminance image" );
	rgdp = PICK_OBJ( "red-green image" );
	bydp = PICK_OBJ( "blue-yellow image" );

	if( lumdp==NO_OBJ || rgdp==NO_OBJ || bydp==NO_OBJ ) return;

	set_rgb_input(lumdp,rgdp,bydp);
}

static COMMAND_FUNC( do_set_dich_input )
{
	Data_Obj *lumdp, *rgdp;

	lumdp = PICK_OBJ( "luminance image" );
	rgdp = PICK_OBJ( "red-green image" );

	if( lumdp==NO_OBJ || rgdp==NO_OBJ ) return;

	set_dich_input(lumdp,rgdp);
}

static COMMAND_FUNC( do_set_clr_output )
{
	Data_Obj *hdp;

	hdp = PICK_OBJ( "byte image for composite halftone" );
	if( hdp==NO_OBJ ) return;
	set_rgb_output(hdp);
}

static COMMAND_FUNC( do_set_dich_output )
{
	Data_Obj *hdp;

	hdp = PICK_OBJ( "byte image for composite halftone" );
	if( hdp==NO_OBJ ) return;
	set_dich_output(hdp);
}

static COMMAND_FUNC( do_set_clr_filter )
{
	Data_Obj *rdp, *gdp, *bdp;

	rdp = PICK_OBJ( "red filter image" );
	gdp = PICK_OBJ( "green filter image" );
	bdp = PICK_OBJ( "blue filter image" );

	if( rdp==NO_OBJ || gdp==NO_OBJ || bdp==NO_OBJ ) return;

	set_rgb_filter(rdp,gdp,bdp);
}

static COMMAND_FUNC( do_set_dich_filter )
{
	Data_Obj *rdp, *gdp;

	rdp = PICK_OBJ( "red filter image" );
	gdp = PICK_OBJ( "green filter image" );

	if( rdp==NO_OBJ || gdp==NO_OBJ ) return;

	set_dich_filter(rdp,gdp);
}

static COMMAND_FUNC( do_clr_migrate_pixel )
{
	int x,y;

	x=HOW_MANY("x coordinate");
	y=HOW_MANY("y coordinate");
	clr_migrate_pixel(x,y);
}


static COMMAND_FUNC( do_clr_redo_pixel )
{
	int x,y;

	x=HOW_MANY("x coordinate");
	y=HOW_MANY("y coordinate");
	clr_redo_pixel(x,y);
}

static COMMAND_FUNC( do_dich_migrate_pixel )
{
	int x,y;

	x=HOW_MANY("x coordinate");
	y=HOW_MANY("y coordinate");
	dich_migrate_pixel(x,y);
}


static COMMAND_FUNC( do_dich_redo_pixel )
{
	int x,y;

	x=HOW_MANY("x coordinate");
	y=HOW_MANY("y coordinate");
	dich_redo_pixel(x,y);
}

static COMMAND_FUNC( do_clr_setxform )
{
	Data_Obj *matrix;

	matrix = PICK_OBJ( "transformation matrix" );
	if( matrix == NO_OBJ ) return;
	set_clr_xform(matrix);
}

static COMMAND_FUNC( do_dich_setxform )
{
	Data_Obj *matrix;

	matrix = PICK_OBJ( "transformation matrix" );
	if( matrix == NO_OBJ ) return;
	set_dich_xform(matrix);
}

static COMMAND_FUNC( do_clr_descend )
{
	clr_scan_requant(1);
}

static COMMAND_FUNC( do_clr_migrate )
{
	clr_scan_migrate(1);
}

static COMMAND_FUNC( do_dich_descend )
{
	dich_scan_requant(1);
}

static COMMAND_FUNC( do_dich_migrate )
{
	dich_scan_migrate(1);
}

Command achrom_req_ctbl[]={
{ "set_input",	do_set_input,		"specify input grayscale image"	},
{ "set_filter",	do_set_filter,		"specify error filter image"	},
{ "set_output",	do_set_output,		"specify output halftone image"	},
{ "anneal",	do_anneal,		"requant image at specified temp"},
{ "descend",	do_scan_requant,	"requantize entire image"	},
{ "migrate",	do_scan2_requant,	"migrate pixels"		},
{ "tweak",	do_tweak,		"tweak at one pixel"		},
#ifdef FOOBAR
{ "quadtree",	do_qt_dither,		"quadtree dither algorithm"	},
#endif /* FOOBAR */
{ "scan",	do_pickscan,		"select scanning pattern"	},
{ "setup_error",do_init_requant,	"initialize error & filtered error"},
#ifndef MAC
{ "quit",	popcmd,			"exit submenu"			},
#endif
{ NULL_COMMAND								}
};


Command threeD_req_ctbl[]={
{ "set_input",	do_set_input3d,		"specify input grayscale image"	},
{ "set_filter",	do_set_filter3d,		"specify error filter image"	},
{ "set_output",	do_set_output3d,		"specify output halftone image"	},
{ "anneal",	do_anneal3d,		"anneal 3D image at specified temp"},
{ "descend",	do_scan_requant3d,	"requantize entire image"	},
{ "migrate",	do_scan2_requant3d,	"migrate pixels"		},
{ "tweak",	do_tweak3d,		"tweak at one pixel"		},
/*
{ "quadtree",	do_qt_dither3d,		"quadtree dither algorithm"	},
*/
{ "scan",	do_pickscan3d,		"select scanning pattern"	},
{ "setup_error",do_init_requant3d,	"initialize error & filtered error"},
#ifndef MAC
{ "quit",	popcmd,			"exit submenu"			},
#endif
{ NULL_COMMAND								}
};


static COMMAND_FUNC( do_achrom )
{
	PUSHCMD(achrom_req_ctbl,"achrom_requant");
}

static COMMAND_FUNC( do_3d )
{
	PUSHCMD(threeD_req_ctbl,"threeD_requant");
}

Command color_req_ctbl[]={
{ "set_input",	do_set_clr_input,		"specify input grayscale image"	},
{ "set_filter",	do_set_clr_filter,		"specify error filter image"	},
{ "set_output",	do_set_clr_output,		"specify output halftone image"	},
{ "redo_pixel",	do_clr_redo_pixel,		"redo a particular pixel"	},
{ "redo_migrate",	do_clr_migrate_pixel,	"redo a particular pixel using migration"	},
{ "descend",	do_clr_descend,		"scan image & reduce error SOS"	},
{ "migrate",	do_clr_migrate,		"scan image & migrate bits"	},
{ "scan",	do_pickscan,		"select scanning pattern"	},
{ "matrix",	do_clr_setxform,		"specify color transformation matrix"},
{ "initialize",	init_clr_requant,	"initialize error images"	},
{ "sos",	tell_sos,		"report SOS's"			},
{ "tell",	cspread_tell,		"info for all internal data images"},
#ifndef MAC
{ "quit",	popcmd,			"exit submenu"			},
#endif
{ NULL_COMMAND								}
};

static COMMAND_FUNC( do_dich_set_weights )
{
	float l,r;

	l=HOW_MUCH("weighting factor for filtered luma errors");
	r=HOW_MUCH("weighting factor for filtered chroma errors");
	set_dich_weights(l,r);
}

Command dich_req_ctbl[]={
{ "set_input",	do_set_dich_input,		"specify input grayscale image"	},
{ "set_filter",	do_set_dich_filter,		"specify error filter image"	},
{ "set_output",	do_set_dich_output,		"specify output halftone image"	},
{ "redo_pixel",	do_dich_redo_pixel,		"redo a particular pixel"	},
{ "redo_migrate",	do_dich_migrate_pixel,	"redo a particular pixel using migration"	},
{ "descend",	do_dich_descend,		"scan image & reduce error SOS"	},
{ "anneal",	do_dich_anneal,			"scan image & reduce error SOS"	},
{ "migrate",	do_dich_migrate,		"scan image & migrate bits"	},
{ "scan",	do_pickscan,		"select scanning pattern"	},
{ "matrix",	do_dich_setxform,		"specify color transformation matrix"},
{ "initialize",	init_dich_requant,	"initialize error images"	},
{ "sos",	dich_tell_sos,		"report SOS's"			},
{ "tell",	dspread_tell,		"info for all internal data images"},
{ "weights",	do_dich_set_weights,	"specify relative weighting for luma and chroma errors"},
#ifndef MAC
{ "quit",	popcmd,			"exit submenu"			},
#endif
{ NULL_COMMAND								}
};

static COMMAND_FUNC( do_dichrom )
{
	PUSHCMD(dich_req_ctbl,"dichrom_requant");
}

static COMMAND_FUNC( do_color )
{
	PUSHCMD(color_req_ctbl,"color_requant");
}

static Command req_ctbl[]={
{ "color",	do_color,		"color requantization submenu"	},
{ "achrom",	do_achrom,		"achromatic requantization submenu"	},
{ "threeD",	do_3d,			"spatiotemporal achromatic requantization submenu"	},
{ "dichrom",	do_dichrom,		"dichromatic requantization submenu"	},
{ "cdiff",	ctone_menu,		"color error diffusion submenu"	},
#ifndef MAC
{ "quit",	popcmd,			"exit submenu"			},
#endif
{ NULL_COMMAND								}
};

COMMAND_FUNC( do_requant )
{
	if( spread_debug == 0 )
		spread_debug=add_debug_module(QSP_ARG  "requantize");

	PUSHCMD(req_ctbl,REQUANT_MENU_NAME);
}


