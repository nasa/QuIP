#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "quip_prot.h"
#include "data_obj.h"
#include "linear.h"
#include "lut_cmds.h"
#include "cmaps.h"
#include "viewer.h"
#include "gen_win.h"
#include "check_dpy.h"

#ifdef BUILD_FOR_OBJC
//#ifdef BUILD_FOR_IOS
Gen_Win *curr_genwin;		// Can we have a global pointer???
//#endif /* BUILD_FOR_IOS */
#endif /* BUILD_FOR_OBJC */

static COMMAND_FUNC( do_setcolor )
{
	int c,r,g,b;

	c=(int)HOW_MANY("lut index");
	r=(int)HOW_MANY("red value");
	g=(int)HOW_MANY("green value");
	b=(int)HOW_MANY("blue value");

	CHECK_DPYP("do_setcolor")
	setcolor(c,r,g,b);
}

static COMMAND_FUNC( do_grayscale )
{
	int base,n;

	base=(int)HOW_MANY("base index");
	n=(int)HOW_MANY("number of colors");

	CHECK_DPYP("do_grayscale")
	make_grayscale(base,n);
}

static COMMAND_FUNC( do_const_alpha )
{
	int value;

	value=(int)HOW_MANY("value");
	CHECK_DPYP("do_const_alpha")
	const_alpha(value);
}

static COMMAND_FUNC( do_const_cmap )
{
	int r,g,b;
	int base,n;

	base=(int)HOW_MANY("start color");
	n=(int)HOW_MANY("number of colors");
	r=(int)HOW_MANY("red");
	g=(int)HOW_MANY("green");
	b=(int)HOW_MANY("blue");
	CHECK_DPYP("do_const_cmap")
	const_cmap(base,n,r,g,b);
}

static COMMAND_FUNC( do_make_rgb )
{
	int r,g,b,base;

	base=(int)HOW_MANY("base color index");
	r=(int)HOW_MANY("number of red levels");
	g=(int)HOW_MANY("number of green levels");
	b=(int)HOW_MANY("number of blue levels");

	CHECK_DPYP("do_make_rgb")
	make_rgb(base,r,g,b);
}

static COMMAND_FUNC( do_poke )
{
	int c,r,g,b;

	c=(int)HOW_MANY("color index");
	r=(int)HOW_MANY("red value");
	g=(int)HOW_MANY("green value");
	b=(int)HOW_MANY("blue value");

	CHECK_DPYP("do_poke")
	poke_lut(c,r,g,b);
}


/* BUG these used to be initialized with null_pick, presumabely
 * so that the lut module wouldn't have any linker dependencies
 * with the dat_obj module...
 * But as of today, it is not getting initialized anywhere,
 * so we are going to apply the quick fix by initializing to pick_obj()
 */

static void *(*pick_func)(const char *)=(void *(*)(const char *))_pick_obj;

#ifdef NOT_USED
/* call this to link this module with the data module */

void set_obj_pick_func( void *(*func)(const char *) )
{
	pick_func = func;
}
#endif /* NOT_USED */

static COMMAND_FUNC( do_getmap )
{
	Data_Obj *dp;

	dp = (Data_Obj *) (*pick_func)("data object");
	if( dp == NULL ) return;
	getmap( dp );
}

static COMMAND_FUNC( do_setmap )
{
	Data_Obj *dp;

	dp = (Data_Obj *) (*pick_func)("data object");
	if( dp == NULL ) return;
	setmap( dp );
}

static COMMAND_FUNC( do_cm_imm )
{
	if( CM_IS_IMMEDIATE ){
		if( verbose ) advise("old state was immediate");
	} else {
		if( verbose ) advise("old state was deferred");
	}
	cm_immediate( ASKIF("update colors immediately") );
}

static COMMAND_FUNC( do_index_alpha )
{
	int index,hv,lv;

	/* set alpha entries */

	index = (int)HOW_MANY("index to display");
	lv = (int)HOW_MANY("alpha value for zero bit");
	hv = (int)HOW_MANY("alpha value for one bit");
	CHECK_DPYP("do_index_alpha")
	index_alpha(index,lv,hv);
}

static COMMAND_FUNC( do_setalpha )
{
	int index,val;

	index=(int)HOW_MANY("index");
	val=(int)HOW_MANY("alpha value");
	CHECK_DPYP("do_setalpha")
	set_alpha(index,val);
}

static COMMAND_FUNC( do_default_cmap )
{
	CHECK_DPYP("do_default_cmap")
#ifdef HAVE_X11
	default_cmap(current_dpyp);
#endif
}

static COMMAND_FUNC( do_update_all )
{
	CHECK_DPYP("do_update_all")
	update_all();
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(cmaps_menu,s,f,h)

MENU_BEGIN(cmaps)
ADD_CMD( setcolor,	do_setcolor,	set a single LUT entry (linearized) )
ADD_CMD( alpha,		do_setalpha,	set alpha colormap entry )
ADD_CMD( index_alpha,	do_index_alpha,	color map to represent a binary index )
ADD_CMD( grayscale,	do_grayscale,	make a grayscale LUT )
ADD_CMD( constant,	do_const_cmap,	make a constant LUT )
ADD_CMD( const_alpha,	do_const_alpha,	make a constant alpha table )
ADD_CMD( poke,		do_poke,	set to single LUT entry (unlinearized) )
ADD_CMD( setmap,	do_setmap,	load current color map from a vector )
ADD_CMD( getmap,	do_getmap,	load vector from current color map )
ADD_CMD( rgb,		do_make_rgb,	make a LUT for an 8 bit RGB image )
ADD_CMD( default,	do_default_cmap,	make the default LUT )
ADD_CMD( update,	do_update_all,	flush buffered data to HW color map )
MENU_END(cmaps)

COMMAND_FUNC( do_cmaps )
{
	CHECK_AND_PUSH_MENU(cmaps);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(luts_menu,s,f,h)

MENU_BEGIN(luts)
#ifndef BUILD_FOR_OBJC
ADD_CMD( lutbuffers,	do_lutbufs,	LUT buffer item submenu )
#endif /* BUILD_FOR_OBJC */
ADD_CMD( linearize,	do_linearize,	gamma correction submenu )
ADD_CMD( cmaps,		do_cmaps,	color map submenu )
ADD_CMD( bitplanes,	do_bit_menu,	bitplane contrast submenu )
ADD_CMD( immediate,	do_cm_imm,	enable/disable immediate color map updates )
MENU_END(luts)

COMMAND_FUNC( do_lut_menu )
{
	Gen_Win *gwp;
	const char *s;
	static int lut_inited = FALSE;

	if (!lut_inited){
		// auto_version???
		lut_inited=TRUE;
	}

	s=NAMEOF("name of viewer or panel");
	gwp = find_genwin(s);

	if( gwp == NULL ){
		/* find_genwin() has already printed an error msg? */
		sprintf(ERROR_STRING,"No viewer or panel named \"%s\"!?",s);
		WARN(ERROR_STRING);
	} else {
#ifdef HAVE_X11
		Dpyable *dpyp;

		dpyp = genwin_display(gwp);
		if( dpyp != NULL ){
			current_dpyp = dpyp;
		}
#endif /* HAVE_X11 */
#ifdef BUILD_FOR_OBJC
//#ifdef BUILD_FOR_IOS
		// How do we tell if this item is a viewer or a panel?
		curr_genwin = gwp;
//#endif /* BUILD_FOR_IOS */
#endif /* BUILD_FOR_OBJC */
	}

	/* We may not have called insure_x11_server() at this point -
	 * but in that case, we cannot have a viewer,
	 * although we could have a panel...
	 * Let's try it without and hope for the best...
	 * But this may be a BUG!?
	 */
	

	CHECK_AND_PUSH_MENU(luts);
}

