#include "quip_config.h"

#include "quip_prot.h"
#include "vec_util.h"
#include "requant.h"
#include "ctone.h"

debug_flag_t spread_debug=0;

static COMMAND_FUNC( do_scan2_requant )
{
	int n;

	n=(int)HOW_MANY("number of passes");
	scan2_requant(QSP_ARG  n);
}

static COMMAND_FUNC( do_scan2_requant3d )
{
	int n;

	n=(int)HOW_MANY("number of passes");
	scan2_requant3d(n);
}

static COMMAND_FUNC( do_scan_requant )
{
	int n;

	n=(int)HOW_MANY("number of passes");
	scan_requant(n);
}

static COMMAND_FUNC( do_anneal )
{
	double temp;
	int n;

	temp = HOW_MUCH("temperature");
	n=(int)HOW_MANY("number of passes");
	scan_anneal(temp,n);
}

static COMMAND_FUNC( do_scan_requant3d )
{
	int n;

	n=(int)HOW_MANY("number of passes");
	scan_requant3d(n);
}

static COMMAND_FUNC( do_anneal3d )
{
	double temp;
	int n;

	temp = HOW_MUCH("temperature");
	n=(int)HOW_MANY("number of passes");
	scan_anneal3d(temp,n);
}

static COMMAND_FUNC( do_dich_anneal )
{
	double temp1,temp2;
	index_t n;

	n=(index_t)HOW_MANY("number of passes");
	temp1=HOW_MUCH("initial temperature");
	temp2=HOW_MUCH("final temperature");
	dich_scan_anneal(QSP_ARG  n,temp1,temp2);
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

	x=(int)HOW_MANY("x coord");
	y=(int)HOW_MANY("y coord");

	redo_two_pixels(x,y);
}
static COMMAND_FUNC( do_tweak3d )
{
	incr_t posn[N_DIMENSIONS];

	posn[0]=posn[4]=0;

	posn[1] = (int)HOW_MANY("x coord");
	posn[2] = (int)HOW_MANY("y coord");
	posn[3] = (int)HOW_MANY("t coord");

	redo_two_pixels3d(posn);
}

static COMMAND_FUNC( do_set_clr_input )
{
	Data_Obj *lumdp, *rgdp, *bydp;

	lumdp = PICK_OBJ( "luminance image" );
	rgdp = PICK_OBJ( "red-green image" );
	bydp = PICK_OBJ( "blue-yellow image" );

	if( lumdp==NO_OBJ || rgdp==NO_OBJ || bydp==NO_OBJ ) return;

	set_rgb_input(QSP_ARG  lumdp,rgdp,bydp);
}

static COMMAND_FUNC( do_set_dich_input )
{
	Data_Obj *lumdp, *rgdp;

	lumdp = PICK_OBJ( "luminance image" );
	rgdp = PICK_OBJ( "red-green image" );

	if( lumdp==NO_OBJ || rgdp==NO_OBJ ) return;

	set_dich_input(QSP_ARG  lumdp,rgdp);
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

	x=(int)HOW_MANY("x coordinate");
	y=(int)HOW_MANY("y coordinate");
	clr_migrate_pixel(QSP_ARG  x,y);
}


static COMMAND_FUNC( do_clr_redo_pixel )
{
	int x,y;

	x=(int)HOW_MANY("x coordinate");
	y=(int)HOW_MANY("y coordinate");
	clr_redo_pixel(QSP_ARG  x,y);
}

static COMMAND_FUNC( do_dich_migrate_pixel )
{
	int x,y;

	x=(int)HOW_MANY("x coordinate");
	y=(int)HOW_MANY("y coordinate");
	dich_migrate_pixel(QSP_ARG  x,y);
}


static COMMAND_FUNC( do_dich_redo_pixel )
{
	int x,y;

	x=(int)HOW_MANY("x coordinate");
	y=(int)HOW_MANY("y coordinate");
	dich_redo_pixel(QSP_ARG  x,y);
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
	clr_scan_requant(QSP_ARG  1);
}

static COMMAND_FUNC( do_clr_migrate )
{
	clr_scan_migrate(QSP_ARG  1);
}

static COMMAND_FUNC( do_dich_descend )
{
	dich_scan_requant(QSP_ARG  1);
}

static COMMAND_FUNC( do_dich_migrate )
{
	dich_scan_migrate(QSP_ARG  1);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(achrom_requant_menu,s,f,h)

MENU_BEGIN(achrom_requant)
ADD_CMD( set_input,	do_set_input,		specify input grayscale image )
ADD_CMD( set_filter,	do_set_filter,		specify error filter image )
ADD_CMD( set_output,	do_set_output,		specify output halftone image )
ADD_CMD( anneal,	do_anneal,		requant image at specified temp )
ADD_CMD( descend,	do_scan_requant,	requantize entire image )
ADD_CMD( migrate,	do_scan2_requant,	migrate pixels )
ADD_CMD( tweak,		do_tweak,		tweak at one pixel )
#ifdef FOOBAR
ADD_CMD( quadtree,	do_qt_dither,		quadtree dither algorithm )
#endif /* FOOBAR */
ADD_CMD( scan,		do_pickscan,		select scanning pattern )
ADD_CMD( setup_error,	do_init_requant,	initialize error & filtered error )
MENU_END(achrom_requant)

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(threeD_requant_menu,s,f,h)

MENU_BEGIN(threeD_requant)
ADD_CMD( set_input,	do_set_input3d,		specify input grayscale image )
ADD_CMD( set_filter,	do_set_filter3d,	specify error filter image )
ADD_CMD( set_output,	do_set_output3d,	specify output halftone image )
ADD_CMD( anneal,	do_anneal3d,		anneal 3D image at specified temp )
ADD_CMD( descend,	do_scan_requant3d,	requantize entire image )
ADD_CMD( migrate,	do_scan2_requant3d,	migrate pixels )
ADD_CMD( tweak,		do_tweak3d,		tweak at one pixel )
/*
ADD_CMD( quadtree,	do_qt_dither3d,		quadtree dither algorithm )
*/
ADD_CMD( scan,		do_pickscan3d,		select scanning pattern )
ADD_CMD( setup_error,	do_init_requant3d,	initialize error & filtered error )
MENU_END(threeD_requant)


static COMMAND_FUNC( do_achrom )
{
	PUSH_MENU(achrom_requant);
}

static COMMAND_FUNC( do_3d )
{
	PUSH_MENU(threeD_requant);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(color_requant_menu,s,f,h)

MENU_BEGIN(color_requant)
ADD_CMD( set_input,	do_set_clr_input,	specify input grayscale image )
ADD_CMD( set_filter,	do_set_clr_filter,	specify error filter image )
ADD_CMD( set_output,	do_set_clr_output,	specify output halftone image )
ADD_CMD( redo_pixel,	do_clr_redo_pixel,	redo a particular pixel )
ADD_CMD( redo_migrate,	do_clr_migrate_pixel,	redo a particular pixel using migration )
ADD_CMD( descend,	do_clr_descend,		scan image & reduce error SOS )
ADD_CMD( migrate,	do_clr_migrate,		scan image & migrate bits )
ADD_CMD( scan,		do_pickscan,		select scanning pattern )
ADD_CMD( matrix,	do_clr_setxform,	specify color transformation matrix )
ADD_CMD( initialize,	init_clr_requant,	initialize error images )
ADD_CMD( sos,		tell_sos,		report SOSs )
ADD_CMD( tell,		cspread_tell,		info for all internal data images )
MENU_END(color_requant)

static COMMAND_FUNC( do_dich_set_weights )
{
	float l,r;

	l=(float)HOW_MUCH("weighting factor for filtered luma errors");
	r=(float)HOW_MUCH("weighting factor for filtered chroma errors");
	set_dich_weights(l,r);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(dichrom_requant_menu,s,f,h)

MENU_BEGIN(dichrom_requant)
ADD_CMD( set_input,	do_set_dich_input,	specify input grayscale image )
ADD_CMD( set_filter,	do_set_dich_filter,	specify error filter image )
ADD_CMD( set_output,	do_set_dich_output,	specify output halftone image )
ADD_CMD( redo_pixel,	do_dich_redo_pixel,	redo a particular pixel )
ADD_CMD( redo_migrate,	do_dich_migrate_pixel,	redo a particular pixel using migration )
ADD_CMD( descend,	do_dich_descend,	scan image & reduce error SOS )
ADD_CMD( anneal,	do_dich_anneal,		scan image & reduce error SOS )
ADD_CMD( migrate,	do_dich_migrate,	scan image & migrate bits )
ADD_CMD( scan,		do_pickscan,		select scanning pattern )
ADD_CMD( matrix,	do_dich_setxform,	specify color transformation matrix )
ADD_CMD( initialize,	init_dich_requant,	initialize error images )
ADD_CMD( sos,		dich_tell_sos,		report SOSs )
ADD_CMD( tell,		dspread_tell,		info for all internal data images )
ADD_CMD( weights,	do_dich_set_weights,	specify relative weighting for luma and chroma errors )
MENU_END(dichrom_requant)

static COMMAND_FUNC( do_dichrom )
{
	PUSH_MENU(dichrom_requant);
}

static COMMAND_FUNC( do_color )
{
	PUSH_MENU(color_requant);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(requantize_menu,s,f,h)

MENU_BEGIN(requantize)
ADD_CMD( color,		do_color,	color requantization submenu )
ADD_CMD( achrom,	do_achrom,	achromatic requantization submenu )
ADD_CMD( threeD,	do_3d,		spatiotemporal achromatic requantization submenu )
ADD_CMD( dichrom,	do_dichrom,	dichromatic requantization submenu )
ADD_CMD( cdiff,		do_ctone_menu,	color error diffusion submenu )
MENU_END(requantize)

COMMAND_FUNC( do_requant )
{
	if( spread_debug == 0 )
		spread_debug=add_debug_module(QSP_ARG  "requantize");

	PUSH_MENU(requantize);
}


