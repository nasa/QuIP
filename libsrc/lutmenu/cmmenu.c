#include "quip_config.h"

char VersionId_lutmenu_cmmenu[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "data_obj.h"
#include "debug.h"
#include "linear.h"
#include "lut_cmds.h"
#include "verlutm.h"
#include "cmaps.h"
#include "menuname.h"
#include "viewer.h"
#include "gui.h"
#include "gen_win.h"
#include "check_dpy.h"

static int lut_inited = FALSE;

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

static void *(*pick_func)(const char *)=(void *(*)(const char *))pick_obj;

/* call this to link this module with the data module */

void set_obj_pick_func( void *(*func)(const char *) )
{
	pick_func = func;
}

static COMMAND_FUNC( do_getmap )
{
	Data_Obj *dp;

	dp = (Data_Obj *) (*pick_func)("data object");
	if( dp == NO_OBJ ) return;
	getmap( dp );
}

static COMMAND_FUNC( do_setmap )
{
	Data_Obj *dp;

	dp = (Data_Obj *) (*pick_func)("data object");
	if( dp == NO_OBJ ) return;
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

Command cmap_ctbl[]={
{ "setcolor",	do_setcolor,	"set a single LUT entry (linearized)"	},
{ "alpha",	do_setalpha,	"set alpha colormap entry"		},
{ "index_alpha",do_index_alpha,	"color map to represent a binary index"	},
{ "grayscale",	do_grayscale,	"make a grayscale LUT"			},
{ "constant",	do_const_cmap,	"make a constant LUT"			},
{ "const_alpha",do_const_alpha,	"make a constant alpha table"		},
{ "poke",	do_poke,	"set to single LUT entry (unlinearized)"},
{ "setmap",	do_setmap,	"load current color map from a vector"	},
{ "getmap",	do_getmap,	"load vector from current color map"	},
{ "rgb",	do_make_rgb,	"make a LUT for an 8 bit RGB image"	},
{ "default",	do_default_cmap,"make the default LUT"			},
{ "update",	do_update_all,	"flush buffered data to HW color map"	},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif /* ! MAC */
{ NULL,		NULL,		NULL					}
};

COMMAND_FUNC( do_cmaps )
{
	PUSHCMD(cmap_ctbl,CMAP_MENU_NAME);
}

#ifndef MAC
Command  lut_ctbl[]={
{ "lutbuffers",	do_lutbufs,	"LUT buffer item submenu"		},
{ "linearize",	do_linearize,	"gamma correction submenu"		},
{ "cmaps",	do_cmaps,	"color map submenu"			},
{ "bitplanes",	bitmenu,	"bitplane contrast submenu"		},
{ "immediate",	do_cm_imm,	"enable/disable immediate color map updates" },
{ "quit",	popcmd,		"exit submenu"				},
{ NULL,		NULL,		NULL					}
};
#endif /* ! MAC */

void lut_init(SINGLE_QSP_ARG_DECL)
{
	if (lut_inited)  return;

	lut_inited = TRUE;

	verlutm(SINGLE_QSP_ARG);
	verluts(SINGLE_QSP_ARG);
}

COMMAND_FUNC( lutmenu )
{
	Item *ip;
	const char *s;

	if (!lut_inited) lut_init(SINGLE_QSP_ARG);

	s=NAMEOF("name of viewer of panel");
	ip = find_genwin(QSP_ARG  s);

	if( ip == NO_GENWIN ){
		/* find_genwin() has already printed an error msg? */
		sprintf(error_string,"No viewer or panel named \"%s\"!?",s);
		WARN(error_string);
	} else {

#ifdef HAVE_X11
		Viewer *vp;

		vp = genwin_viewer(QSP_ARG  ip);
		if( vp != NO_VIEWER ){
			current_dpyp = &vp->vw_top;
		} else {
#endif /* HAVE_X11 */

#ifdef HAVE_MOTIF
			Panel_Obj *po;

			po = genwin_panel(QSP_ARG  ip);
			if( po != NO_PANEL_OBJ ){
				current_dpyp = &po->po_top;
			}
#endif /* HAVE_MOTIF */

#ifdef HAVE_X11
		}
#endif /* HAVE_X11 */

	}

	/* We may not have called insure_x11_server() at this point -
	 * but in that case, we cannot have a viewer,
	 * although we could have a panel...
	 * Let's try it without and hope for the best...
	 * But this may be a BUG!?
	 */
	

#ifndef MAC
	PUSHCMD(lut_ctbl,"luts");
#else
	do_lutbufs();
	do_linearize();
	do_cmaps();
	bitmenu();
#endif /* MAC */
}

