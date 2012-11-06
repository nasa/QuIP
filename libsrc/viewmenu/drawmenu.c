
#include "quip_config.h"

char VersionId_viewmenu_drawmenu[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>	/* floor() */
#endif

#include "xsupp.h"
#include "viewer.h"
#include "view_cmds.h"
#include "view_util.h"
#include "debug.h"	/* verbose */
#include "items.h"

#include "get_viewer.h"

static Viewer *draw_vp=NO_VIEWER;

#define DRAW_CHECK	if( draw_vp == NO_VIEWER ) return
#define XFORMING_COORDS(vp)					\
	( ( vp ) != NO_VIEWER && ( vp )->vw_flags & VIEW_XFORM_COORDS )

/* a menu of stuff to draw into windows */
/* the window-dependent stuff ought to go into xsupp, but for now
 * it's all together here */

#define MAX_FONT_NAMES	512

ITEM_INTERFACE_PROTOTYPES( XFont , xfont )

#define XFONT_OF( s )		xfont_of( QSP_ARG  s)
/* local prototypes */
/* static XFont *pick_xfont(const char *); */
//static XFont *new_xfont(const char *);
//static XFont *xfont_of(const char *);
//static COMMAND_FUNC( do_list_xfonts );
//static void list_xfonts(void);
//static void xfont_init(void);

static void get_cpair(QSP_ARG_DECL  int *px,int *py);
static COMMAND_FUNC( do_set_font );
static COMMAND_FUNC( do_load_font );
static COMMAND_FUNC( do_set_fg );
static COMMAND_FUNC( do_set_bg );
static COMMAND_FUNC( do_draw_string );
static COMMAND_FUNC( do_show_gc );
static COMMAND_FUNC( do_move );
static COMMAND_FUNC( do_cont );
static COMMAND_FUNC( do_clear );
static COMMAND_FUNC( do_scale );

static int curr_x=0;
static int curr_y=0;


ITEM_INTERFACE_DECLARATIONS(XFont,xfont)

static void get_cpair(QSP_ARG_DECL   int *px, int *py)
{
	float fx,fy;

	fx=HOW_MUCH("x");
	fy=HOW_MUCH("y");

	if( XFORMING_COORDS(draw_vp) )
		scale_fxy(draw_vp,&fx,&fy);

	*px=floor(fx+0.5);
	*py=floor(fy+0.5);
}

void load_font(QSP_ARG_DECL  const char *fontname)
{
	Font id;
	XFont *xfp;
	char **flist;
	int nfonts;

	DRAW_CHECK;

	flist = XListFonts(draw_vp->vw_dpy,fontname,MAX_FONT_NAMES,&nfonts);
	if( nfonts > 1 ){
		/* This is not really an error, not sure
		 * why it would be that a font would list twice??
		 */
		if( verbose ){
			int i;

			advise("more than 1 font matches this specification");
			for(i=0;i<nfonts;i++){
				sprintf(error_string,"\t%s",flist[i]);
				advise(error_string);
			}
		}
	} else if( nfonts != 1 ){
		sprintf(error_string,"Font %s is not available",fontname);
		WARN(error_string);
		XFreeFontNames(flist);
		return;
	}
	XFreeFontNames(flist);

	id = XLoadFont(draw_vp->vw_dpy,fontname);
	xfp = new_xfont(QSP_ARG  fontname);
	if( xfp != NO_XFONT )
		xfp->xf_id = id;
}

/* BUG the X library calls for the fonts should be moved to xsupp */

static COMMAND_FUNC( do_set_font )
{
	const char *s;
	XFont *xfp;

	/* xfp=pick_xfont(""); */

	s=NAMEOF("font name");

	xfp = XFONT_OF(s);
	if( xfp == NO_XFONT ){
		load_font(QSP_ARG  s);
		xfp = XFONT_OF(s);
		if( xfp == NO_XFONT ) return;
	}

	DRAW_CHECK;

	set_font(draw_vp,xfp);
}


static COMMAND_FUNC( do_load_font )
{
	const char *s;

	s=NAMEOF("font");
	load_font(QSP_ARG  s);
}


static COMMAND_FUNC( do_set_fg )
{
	u_long val;

	val = HOW_MANY("foreground");

	DRAW_CHECK;

	_xp_select(draw_vp,val);
}

static COMMAND_FUNC( do_set_bg )
{
	u_long val;

	val = HOW_MANY("background");

	DRAW_CHECK;

	_xp_bgselect(draw_vp,val);
}

static COMMAND_FUNC( do_draw_string )
{
	const char *s;
	int x,y;

	s=NAMEOF("string");
	get_cpair(QSP_ARG  &x,&y);

	DRAW_CHECK;

	_xp_text(draw_vp,x,y,s);
}

static COMMAND_FUNC( do_show_gc )
{
	u_long mask;
	XGCValues gcvals;

	DRAW_CHECK;

	mask = GCPlaneMask | GCForeground | GCBackground;

	XGetGCValues(draw_vp->vw_dpy,draw_vp->vw_gc,mask,&gcvals);

	sprintf(msg_str,"Graphics Context for viewer %s:",draw_vp->vw_name);
	prt_msg(msg_str);
	sprintf(msg_str,"planemask\t%ld",gcvals.plane_mask);
	prt_msg(msg_str);
	sprintf(msg_str,"foreground\t%ld",gcvals.foreground);
	prt_msg(msg_str);
	sprintf(msg_str,"background\t%ld",gcvals.background);
	prt_msg(msg_str);
}

static COMMAND_FUNC( do_move )
{
	get_cpair(QSP_ARG  &curr_x,&curr_y);
}

static COMMAND_FUNC( do_cont )
{
	int x,y;

	get_cpair(QSP_ARG  &x,&y);

	DRAW_CHECK;

	_xp_line(draw_vp,curr_x,curr_y,x,y);
	curr_x = x;
	curr_y = y;
}

static COMMAND_FUNC( do_arc )
{
	int xl,yu,w,h,a1,a2;

	xl = HOW_MANY("xl");
	yu = HOW_MANY("yu");
	w = HOW_MANY("w");
	h = HOW_MANY("h");
	a1 = HOW_MANY("a1");
	a2 = HOW_MANY("a2");

	DRAW_CHECK;

	_xp_arc(draw_vp,xl,yu,w,h,a1,a2);
}

static COMMAND_FUNC( do_clear )
{
	DRAW_CHECK;

	_xp_erase(draw_vp);
}

static COMMAND_FUNC( do_scale )
{
	int scal_flag;

	scal_flag = ASKIF("scale coordinates in using plotting space");

	DRAW_CHECK;

	if( scal_flag )
		draw_vp->vw_flags |= VIEW_XFORM_COORDS;
	else
		draw_vp->vw_flags &= ~VIEW_XFORM_COORDS;
}

COMMAND_FUNC( do_remem_gfx )
{
	int flag;

	flag = ASKIF("remember draw ops to allow refresh on expose events");
	set_remember_gfx(flag);
}

static COMMAND_FUNC( do_fill_arc )
{
	int xl,yu,w,h,a1,a2;

	xl = HOW_MANY("xl");
	yu = HOW_MANY("yu");
	w = HOW_MANY("w");
	h = HOW_MANY("h");
	a1 = HOW_MANY("a1");
	a2 = HOW_MANY("a2");

	DRAW_CHECK;

	_xp_fill_arc(draw_vp,xl,yu,w,h,a1,a2);
}

static COMMAND_FUNC( do_fill_poly )
{
	int* x_vals=NULL, *y_vals = NULL;
	unsigned int num_points;
	unsigned int i;

	num_points = HOW_MUCH("number of polygon points");
	x_vals = (int *) getbuf(sizeof(int) * num_points);
	y_vals = (int *) getbuf(sizeof(int) * num_points);
	
	for (i=0; i < num_points; i++) {
		char s[100];
		sprintf(s, "point %d x value", i+1);
		x_vals[i] = HOW_MANY(s);
		sprintf(s, "point %d y value", i+1);
		y_vals[i] = HOW_MANY(s);
	}

	_xp_fill_polygon(draw_vp,num_points, x_vals, y_vals);

	givbuf(x_vals);
	givbuf(y_vals);
}

#define N_TEXT_MODES	2
static const char *text_modes[N_TEXT_MODES]={"centered","left_justified"};

static COMMAND_FUNC( do_set_text_mode )
{
	int i;

	i=WHICH_ONE("text positioning mode",N_TEXT_MODES,text_modes);
	if( i < 0 ) return;

	switch(i){
		case 0:  center_text(); break;
		case 1:  left_justify(); break;
	}
}

static COMMAND_FUNC( do_list_xfonts )
{ list_xfonts(SINGLE_QSP_ARG); }

static COMMAND_FUNC( do_get_string_width )
{
	const char *v, *s;
	int n;

	v=savestr(NAMEOF("variable name for string width"));
	s=NAMEOF("string");

	n = get_string_width(draw_vp,s);
	sprintf(msg_str,"%d",n);
	ASSIGN_VAR(v,msg_str);
	rls_str((char *)v);
}

static Command drawctbl[]={
{ "load",		do_load_font,		"load a font"				},
{ "font",		do_set_font,		"select a loaded font for drawing"	},
{ "text_mode",		do_set_text_mode,	"select text drawing mode"		},
{ "list",		do_list_xfonts,		"list all loaded fonts"			},
{ "string",		do_draw_string,		"draw a string in foreground color"	},
{ "get_string_width",	do_get_string_width,	"return"				},
{ "gc",			do_show_gc,		"show graphics context"			},
{ "foreground",		do_set_fg,		"set foreground color"			},
{ "background",		do_set_bg,		"set background color"			},
{ "move",		do_move,		"move pen position"			},
{ "cont",		do_cont,		"draw to new position"			},
{ "arc",		do_arc,			"draw circular arc"			},
{ "fill_arc",		do_fill_arc,		"draw a filled arc"			},
{ "fill_poly",		do_fill_poly,		"draw a filled polygon"			},
{ "clear",		do_clear,		"clear current window"			},
{ "scale",		do_scale,		"enable coordinate scale transform"	},
{ "remember",		do_remem_gfx,		"remember graphics for expose event refresh"	},
{ "quit",		popcmd,			"exit submenu"				},
{ NULL_COMMAND										}
};


COMMAND_FUNC( drawmenu )
{
	Viewer *vp;

	insure_x11_server(SINGLE_QSP_ARG);
	GET_VIEWER("drawmenu")
	draw_vp = vp;
	select_viewer(QSP_ARG  vp);

	PUSHCMD(drawctbl,"draw");
}

#endif /* HAVE_X11 */

