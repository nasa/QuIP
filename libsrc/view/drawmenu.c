
#include "quip_config.h"

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>	/* floor() */
#endif

#include "quip_prot.h"
#include "xsupp.h"
#include "viewer.h"
#include "view_cmds.h"
#include "view_prot.h"
#include "view_util.h"
#include "debug.h"	/* verbose */
#include "item_type.h"

#include "get_viewer.h"

// BUG for this module to be thread-safe, this static
// variable has to be part of the Query_Stack...
static Viewer *draw_vp=NULL;

#define DRAW_CHECK(s)							\
									\
	if( draw_vp == NULL ){						\
		sprintf(ERROR_STRING,"%s:  no drawing viewer selected!?",#s);	\
		WARN(ERROR_STRING);					\
		return;							\
	}

#define DRAW_CHECK_INT(s,error_ret_val)					\
									\
	if( draw_vp == NULL ){						\
		sprintf(ERROR_STRING,"%s:  no drawing viewer selected!?",#s);	\
		WARN(ERROR_STRING);					\
		return error_ret_val;					\
	}


#define XFORMING_COORDS(vp)					\
	( ( vp ) != NULL && VW_FLAGS(vp) & VIEW_XFORM_COORDS )

/* a menu of stuff to draw into windows */
/* the window-dependent stuff ought to go into xsupp, but for now
 * it's all together here */

#define MAX_FONT_NAMES	512
// on the mac, xlsfonts outputs 10277 lines!

#ifdef HAVE_X11
ITEM_INTERFACE_PROTOTYPES( XFont , xfont )
#define xfont_of( s )		_xfont_of( QSP_ARG  s)
#define new_xfont( s )		_new_xfont( QSP_ARG  s)
#define list_xfonts( fp )		_list_xfonts( QSP_ARG  fp)
#endif /* HAVE_X11 */

static int curr_x=0;
static int curr_y=0;

#ifdef HAVE_X11
ITEM_INTERFACE_DECLARATIONS(XFont,xfont,0)
#endif /* HAVE_X11 */

static void get_cpair(QSP_ARG_DECL  int *px, int *py)
{
	float fx,fy;

	fx=(float)HOW_MUCH("x");
	fy=(float)HOW_MUCH("y");

	if( XFORMING_COORDS(draw_vp) )
		scale_fxy(draw_vp,&fx,&fy);

	*px=(int)floor(fx+0.5);
	*py=(int)floor(fy+0.5);
}

// BUG - move to OS-specific file!

#ifdef BUILD_FOR_IOS

static int ios_check_font(const char *fontname)
{
	// We don't need to do anything to load the fonts,
	// although we could print an error message for a bad font name?

	// compare the requested font against the available font families

	NSString *s;
	NSArray *a = [UIFont familyNames];
	int found=0;
	NSUInteger i;
	for(i=0;i<a.count;i++){
		s=[a objectAtIndex:i];
		if( !strcmp(s.UTF8String,fontname) ){
			//fprintf(stderr,"Found font %s\n",s.UTF8String);
			found=1;
			i=a.count;
		}
	}
	if( ! found ){
		sprintf(ERROR_STRING,"Font %s not found!?",fontname);
		WARN(ERROR_STRING);
		fprintf(stderr,"Font %s not found!?",fontname);
		fprintf(stderr,"Available fonts:\n");
		for(i=0;i<a.count;i++){
			s=[a objectAtIndex:i];
			fprintf(stderr,"%s\n",s.UTF8String);
		}
		return 0;
	}
	return 1;
}
#endif // BUILD_FOR_IOS

// quick_load_font - load a font we are sure exists
// returns 1 if loaded, 0 if already loaded

#define quick_load_font(fontname) _quick_load_font(QSP_ARG  fontname)

static int _quick_load_font(QSP_ARG_DECL  const char *fontname)
{
#ifdef HAVE_X11
	// BUG - this stuff should go in xsupp...
	Font id;
	XFont *xfp;
	//char **flist;
	//int nfonts;

	DRAW_CHECK_INT(quick_load_font,0)

	xfp = xfont_of(fontname);
	if( xfp != NULL ){
		if( verbose ){
			// xlsfonts lists some fonts twice
			// (perhaps because of multiple font directories???)
			// so we suppress this warning.
			sprintf(ERROR_STRING,"load_font:  font %s is already loaded!?",
				fontname);
			advise(ERROR_STRING);
		}
		return 0;
	}

	id = XLoadFont(VW_DPY(draw_vp),fontname);
	xfp = new_xfont(fontname);
	if( xfp != NULL ){
		xfp->xf_id = id;
		xfp->xf_fsp = XQueryFont(VW_DPY(draw_vp),id);
		// If we ever delete this thing, we have to free
		// with XFreeFontInfo
	}
	return 1;

#endif /* HAVE_X11 */

#ifdef BUILD_FOR_IOS
	return ios_check_font(fontname);
#endif /* BUILD_FOR_IOS */

	
}

#define find_font(varname, family, bold_name, font_size ) _find_font(QSP_ARG  varname, family, bold_name, font_size )

static void _find_font(QSP_ARG_DECL  const char *varname, const char *family, const char * bold_name, int font_size )
{
	char **flist;
	int nfonts;
	char pattern[LLEN];

	DRAW_CHECK(find_font)

	//                      | change that r to i for italic
	//                      | |star can be "normal" or "normal-sans" or ???
	sprintf(pattern,"*%s-%s-r-*--%d-*",family,bold_name,font_size);
	flist = XListFonts(VW_DPY(draw_vp),pattern,MAX_FONT_NAMES,&nfonts);
	if( nfonts < 0 ){
		warn("find_font:  XListFonts returned negative!?");
		assign_var(varname,"no_font");
	} else if( nfonts == 0 ){
		assign_var(varname,"no_font");
	} else {
		// return the first font
		assign_var(varname,flist[0]);
	}
	XFreeFontNames(flist);
}

// load_font - make sure the font exists before trying to load

#define load_font(fontname) _load_font(QSP_ARG  fontname)

static void _load_font(QSP_ARG_DECL  const char *fontname)
{
#ifdef HAVE_X11
	// BUG - this stuff should go in xsupp...
	//Font id;
	//XFont *xfp;
	char **flist;
	int nfonts;

	DRAW_CHECK(load_font)

	flist = XListFonts(VW_DPY(draw_vp),fontname,MAX_FONT_NAMES,&nfonts);
	if( nfonts > 1 ){
		/* This is not really an error, not sure
		 * why it would be that a font would list twice??
		 */
		if( verbose ){
			int i;

			advise("more than 1 font matches this specification");
			for(i=0;i<nfonts;i++){
				sprintf(ERROR_STRING,"\t%s",flist[i]);
				advise(ERROR_STRING);
			}
		}
	} else if( nfonts != 1 ){
		sprintf(ERROR_STRING,"Font %s is not available",fontname);
		WARN(ERROR_STRING);
		XFreeFontNames(flist);
		return;
	}
	XFreeFontNames(flist);

#endif /* HAVE_X11 */

	quick_load_font(fontname);
}

#define load_font_set(pattern) _load_font_set(QSP_ARG  pattern)

static void _load_font_set(QSP_ARG_DECL  const char *pattern)
{
#ifdef HAVE_X11
	// BUG - this stuff should go in xsupp...
	char **flist;
	int nfonts, n_loaded;
	int i;

	DRAW_CHECK(load_font_set)

	flist = XListFonts(VW_DPY(draw_vp),pattern,MAX_FONT_NAMES,&nfonts);
	if( nfonts == 0 ){
		sprintf(ERROR_STRING,"No fonts found matching '%s'!?",pattern);
		advise(ERROR_STRING);
		return;
	}
fprintf(stderr,"%d fonts found matching pattern '%s'\n",nfonts,pattern);
	// Now load each font...
	n_loaded=0;
	for(i=0;i<nfonts;i++){
		n_loaded += quick_load_font(flist[i]);
	}
	XFreeFontNames(flist);

fprintf(stderr,"%d unique fonts loaded\n",n_loaded);
#endif /* HAVE_X11 */

#ifdef BUILD_FOR_IOS
	warn("Sorry, load_font_set not implemented yet for iOS!?");
#endif /* BUILD_FOR_IOS */

}

/* BUG the X library calls for the fonts should be moved to xsupp */

static COMMAND_FUNC( do_set_font )
{
	const char *s;
#ifdef HAVE_X11
	XFont *xfp;
#endif /* HAVE_X11 */

	/* xfp=pick_xfont(""); */

	s=NAMEOF("font name");

#ifdef HAVE_X11
	xfp = xfont_of(s);
	if( xfp == NULL ){
		load_font(s);
		xfp = xfont_of(s);
		if( xfp == NULL ){
			WARN("Unable to load font!?");
			return;
		}
	}

	DRAW_CHECK(do_set_font)

	set_font(draw_vp,xfp);
#endif /* HAVE_X11 */

#ifdef BUILD_FOR_IOS

	set_font_by_name(draw_vp,s);

#endif /* BUILD_FOR_IOS */
}


static COMMAND_FUNC( do_load_font )
{
	const char *s;

	s=NAMEOF("font");
	load_font(s);
}

static COMMAND_FUNC( do_load_font_set )
{
	const char *s;

	s=NAMEOF("match pattern");
	load_font_set(s);
}


static COMMAND_FUNC( do_set_fg )
{
	u_long val;

	val = (u_long) HOW_MANY("foreground");

	DRAW_CHECK(do_set_fg)

	xp_select(draw_vp,val);
}

static COMMAND_FUNC( do_set_bg )
{
	u_long val;

	val = (u_long) HOW_MANY("background");

	DRAW_CHECK(do_set_bg)

	xp_bgselect(draw_vp,val);
}

static COMMAND_FUNC( do_draw_string )
{
	const char *s;
	int x,y;

	s=NAMEOF("string");
	get_cpair(QSP_ARG  &x,&y);

	DRAW_CHECK(do_draw_string)

	xp_text(draw_vp,x,y,s);
}

static COMMAND_FUNC( do_show_gc )
{
#ifdef HAVE_X11
	u_long mask;
	XGCValues gcvals;

	DRAW_CHECK(do_show_gc)

	mask = GCPlaneMask | GCForeground | GCBackground;

	XGetGCValues(VW_DPY(draw_vp),VW_GC(draw_vp),mask,&gcvals);

	sprintf(msg_str,"Graphics Context for viewer %s:",VW_NAME(draw_vp));
	prt_msg(msg_str);
	sprintf(msg_str,"planemask\t%ld",gcvals.plane_mask);
	prt_msg(msg_str);
	sprintf(msg_str,"foreground\t%ld",gcvals.foreground);
	prt_msg(msg_str);
	sprintf(msg_str,"background\t%ld",gcvals.background);
	prt_msg(msg_str);
#endif /* HAVE_X11 */
}

static COMMAND_FUNC( do_move )
{
	get_cpair(QSP_ARG  &curr_x,&curr_y);
}

static COMMAND_FUNC( do_cont )
{
	int x,y;

	get_cpair(QSP_ARG  &x,&y);

	DRAW_CHECK(do_cont)

	xp_line(draw_vp,curr_x,curr_y,x,y);
	curr_x = x;
	curr_y = y;
}

static COMMAND_FUNC( do_linewidth )
{
#ifdef BUILD_FOR_IOS	// BUILD_FOR_OBJC ?
	CGFloat w;

	w = HOW_MUCH("line width in points");

	DRAW_CHECK(do_linewidth)

	_xp_linewidth(draw_vp,w);
#else // ! BUILD_FOR_IOS
	int w;

	w = (int) HOW_MANY("line width in pixels");

	DRAW_CHECK(do_linewidth)

	_xp_linewidth(draw_vp,w);
#endif // ! BUILD_FOR_IOS
}


static COMMAND_FUNC( do_arc )
{
	int xl,yu,w,h,a1,a2;

	xl = (int)HOW_MANY("xl");
	yu = (int)HOW_MANY("yu");
	w = (int)HOW_MANY("w");
	h = (int)HOW_MANY("h");
	a1 = (int)HOW_MANY("a1");
	a2 = (int)HOW_MANY("a2");

	DRAW_CHECK(do_arc)

	_xp_arc(draw_vp,xl,yu,w,h,a1,a2);
}

static COMMAND_FUNC( do_clear )
{
	DRAW_CHECK(do_clear)

	xp_erase(draw_vp);
}

static COMMAND_FUNC( do_update )
{
	DRAW_CHECK(do_update)

	_xp_update(draw_vp);
}

static COMMAND_FUNC( do_scale )
{
	int scal_flag;

	scal_flag = ASKIF("scale coordinates in using plotting space");

	DRAW_CHECK(do_scale)

	if( scal_flag )
		SET_VW_FLAG_BITS(draw_vp, VIEW_XFORM_COORDS);
	else
		CLEAR_VW_FLAG_BITS(draw_vp, VIEW_XFORM_COORDS);
}

static COMMAND_FUNC( do_remem_gfx )
{
	int flag;

	flag = ASKIF("remember draw ops to allow refresh on expose events");
	set_remember_gfx(flag);
}

static COMMAND_FUNC( do_fill_arc )
{
	int xl,yu,w,h,a1,a2;

	xl = (int)HOW_MANY("xl");
	yu = (int)HOW_MANY("yu");
	w = (int)HOW_MANY("w");
	h = (int)HOW_MANY("h");
	a1 = (int)HOW_MANY("a1");
	a2 = (int)HOW_MANY("a2");

	DRAW_CHECK(do_fill_arc)

	_xp_fill_arc(draw_vp,xl,yu,w,h,a1,a2);
}

static COMMAND_FUNC( do_fill_poly )
{
	int* x_vals=NULL, *y_vals = NULL;
	unsigned int num_points;
	unsigned int i;

	num_points = (int)HOW_MANY("number of polygon points");
	x_vals = (int *) getbuf(sizeof(int) * num_points);
	y_vals = (int *) getbuf(sizeof(int) * num_points);

	for (i=0; i < num_points; i++) {
		char s[100];
		sprintf(s, "point %d x value", i+1);
		x_vals[i] = (int)HOW_MANY(s);
		sprintf(s, "point %d y value", i+1);
		y_vals[i] = (int)HOW_MANY(s);
	}

	_xp_fill_polygon(draw_vp,num_points, x_vals, y_vals);

	givbuf(x_vals);
	givbuf(y_vals);
}

#define N_TEXT_MODES	3
static const char *text_modes[N_TEXT_MODES]={"centered","left_justified","right_justified"};

static COMMAND_FUNC( do_set_text_mode )
{
	int i;

	i=WHICH_ONE("text positioning mode",N_TEXT_MODES,text_modes);
	if( i < 0 ) return;

	switch(i){
		case 0:  center_text(draw_vp); break;
		case 1:  left_justify(draw_vp); break;
		case 2:  right_justify(draw_vp); break;
#ifdef CAUTIOUS
		default:
error1("CAUTIOUS:  do_set_text_mode:  Unexpected text justification index!?");
			break;
#endif /* CAUTIOUS */
	}
}

static float radians_per_degree=0.0;

static COMMAND_FUNC( do_set_font_size )
{
	int s;

	s=(int)HOW_MANY("font size");
	set_font_size(draw_vp,s);
}

#define N_BOLD_TYPES	2
const char *bold_type_name[N_BOLD_TYPES]={"bold","medium"};

static COMMAND_FUNC( do_find_font )
{
	const char *varname, *family;
	int bold_type_idx, font_size;

	varname = nameof("variable name for font result");
	family = nameof("font family");
	bold_type_idx = which_one("bold",N_BOLD_TYPES,bold_type_name);
	font_size = how_many("font size");

	if( bold_type_idx < 0 ) return;
	if( font_size <= 0 ) {
		warn("font size must be positive");
		return;
	}

	find_font(varname,family,bold_type_name[bold_type_idx],font_size);
}

static COMMAND_FUNC( do_set_text_angle )
{
	float a;

	a=(float)HOW_MUCH("text angle in degrees");
	if( radians_per_degree == 0.0 )
		radians_per_degree = (float)(atan(1.0)/45.0);

	set_text_angle(draw_vp,a*radians_per_degree);
}

static COMMAND_FUNC( do_list_xfonts )
{
#ifdef HAVE_X11
	list_xfonts(tell_msgfile());
#endif /* HAVE_X11 */
}

static COMMAND_FUNC( do_get_string_width )
{
	const char *v, *s;
	int n;

	v=savestr(NAMEOF("variable name for string width"));
	s=NAMEOF("string");

	n = get_string_width(draw_vp,s);
	sprintf(msg_str,"%d",n);
	assign_var(v,msg_str);
	rls_str((char *)v);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(draw_menu,s,f,h)

MENU_BEGIN(draw)
ADD_CMD( load,		do_load_font,		load a font )
ADD_CMD( load_set,	do_load_font_set,	load a group of fonts )
ADD_CMD( font,		do_set_font,		select a loaded font for drawing )
ADD_CMD( font_size,	do_set_font_size,	set font size )
ADD_CMD( find_font,	do_find_font,		find a font matching criteria )
ADD_CMD( text_mode,	do_set_text_mode,	select text drawing mode )
ADD_CMD( text_angle,	do_set_text_angle,	select text drawing angle )
ADD_CMD( list,		do_list_xfonts,		list all loaded fonts )
ADD_CMD( string,	do_draw_string,		draw a string in foreground color )
ADD_CMD( get_string_width,	do_get_string_width,	return )
ADD_CMD( gc,		do_show_gc,		show graphics context )
ADD_CMD( foreground,	do_set_fg,		set foreground color )
ADD_CMD( background,	do_set_bg,		set background color )
ADD_CMD( move,		do_move,		move pen position )
ADD_CMD( cont,		do_cont,		draw to new position )
ADD_CMD( line_width,	do_linewidth,		specify line width )
ADD_CMD( arc,		do_arc,			draw circular arc )
ADD_CMD( fill_arc,	do_fill_arc,		draw a filled arc )
ADD_CMD( fill_poly,	do_fill_poly,		draw a filled polygon )
ADD_CMD( clear,		do_clear,		clear current window )
ADD_CMD( update,	do_update,		force update of current window )
ADD_CMD( scale,		do_scale,		enable coordinate scale transform )
ADD_CMD( remember,	do_remem_gfx,		remember graphics for expose event refresh )

MENU_END(draw)

COMMAND_FUNC( do_draw_menu )
{
	Viewer *vp;

	GET_VIEWER("drawmenu")
	draw_vp = vp;
	select_viewer(vp);

#ifdef BUILD_FOR_OBJC
	if( VW_CANVAS(vp) == NULL )
		init_viewer_canvas(vp);
#endif /* BUILD_FOR_OBJC */

	CHECK_AND_PUSH_MENU(draw);
}

