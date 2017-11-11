#include "quip_config.h"

#include "quip_prot.h"
#include "viewer.h"
#include "xsupp.h"
#include "debug.h"

#ifdef HAVE_X11

/*
 * this file SHOULD contain all of the Xlib function calls for viewers
 *
 * To reimplement under a new window manager, only these functions
 * need to be redone
 *
 * ifdef YES8BIT  try to get an 8 bit visual instead of the default
 */

#define MIN(a,b)	( (a) < (b) ? (a) : (b) )

#define ALLOC_DATA	1
#define DONT_ALLOC_DATA	0

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* abs() */
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* sleep() */
#endif

#ifdef HAVE_STRING_H
#include <string.h>		/* memcpy() */
#endif

#include "quip_prot.h"
#include "data_obj.h"
#include "xsupp_prot.h"
#include "cmaps.h"
#include "viewer.h"

#ifdef HAVE_SYS_IPC_H
#include <sys/ipc.h>
#endif

#ifdef HAVE_SYS_SHM_H
#include <sys/shm.h>		/* shmget() */
#endif

#ifdef HAVE_X11_EXTENSIONS_XSHM_H
#include <X11/extensions/XShm.h>
#endif

#ifdef SGI_GL
#include "glxhelper.h"
#endif /* SGI_GL */

#include "cmaps.h"

const char *def_geom="";
#ifdef QUIP_DEBUG
u_long xdebug=0;
#endif /* QUIP_DEBUG */

/* draw-op stuff, local to this file */

static List *unused_dop_list=NULL;
// BUG not thread-safe...
static XFont *current_xfp=NULL;

static int display_to_mapped=0;		/* flag - if set, then wait for windows to be mapped before displaying */

typedef enum {
	LEFT_JUSTIFY,	// 0
	CENTER_TEXT,	// 1
	RIGHT_JUSTIFY	// 2
} Text_Mode;


typedef struct draw_op_args {
	const char *		doa_str;
	union {
		uint32_t	u_color;
		int		u_int[7];
		XFont *		u_xfp;
		Text_Mode	u_mode;
	} doa_u;
} Draw_Op_Args;

/* move, cont args */
#define doa_color	doa_u.u_color
#define doa_int		doa_u.u_int
#define doa_xfp		doa_u.u_xfp

#define doa_text_mode	doa_u.u_mode

#define doa_x		doa_int[0]
#define doa_y		doa_int[1]
#define doa_lw		doa_int[0]

#define doa_xl		doa_int[0]
#define doa_yu		doa_int[1]
#define doa_w		doa_int[2]
#define doa_h		doa_int[3]
#define doa_a1		doa_int[4]
#define doa_a2		doa_int[5]
#define doa_filled	doa_int[6]

/* drawing operations */
typedef enum {
	DRAW_OP_NONE,		// 0
	DRAW_OP_TEXT,		// 1
	DRAW_OP_TEXT_MODE,	// 2
	DRAW_OP_MOVE,		// 3
	DRAW_OP_CONT,		// 4
	DRAW_OP_FOREGROUND,	// 5
	DRAW_OP_BACKGROUND,	// 6
	DRAW_OP_ARC,		// 7
	DRAW_OP_LINEWIDTH,	// 8
	N_DRAW_OP_TYPES		// 9
} Draw_Op_Code;

typedef struct draw_op {
	Draw_Op_Code	do_op;
	Draw_Op_Args	do_doa;
} Draw_Op;

#define do_x		do_doa.doa_x
#define do_y		do_doa.doa_y
#define do_lw		do_doa.doa_lw
#define do_str		do_doa.doa_str
#define do_xfp		do_doa.doa_xfp
#define do_color	do_doa.doa_color
#define do_xl		do_doa.doa_xl
#define do_yu		do_doa.doa_yu
#define do_w		do_doa.doa_w
#define do_h		do_doa.doa_h
#define do_a1		do_doa.doa_a1
#define do_a2		do_doa.doa_a2
#define do_filled	do_doa.doa_filled
#define do_text_mode	do_doa.doa_text_mode

#define WINDOW_BORDER_WIDTH	2

static Bool WaitForNotify(Display *dpy, XEvent *ep, XPointer arg)
{
	switch( ep->type ){
		case MapNotify:
			/* advise("MapNotify"); */
			return(1);
		default:
			return(0);
	}
return(1);
}


static int quick=0;
/*
 * We remember all drawing ops to be able to redraw after a window expose event.
 * However, we have been sloppy about freeing up the nodes, plust for some reason,
 * this thing uses handles, which use malloc...
 *
 * This didn't cause a problem until we started running big ray tracing sims...
 * Here we do a hack kludge to disable remembering drawing...
 * Actually, drawing should be disabled in the sim, but the memory is leaking
 * out of somewhere!?
 */

// BUG static globals not thread-safe!
static int from_memory=0;		/* set to ture when redrawing after expose */
					/* if true, then don't re-remember draw cmds */
static int remember_gfx=1;		/* by default, remember gfx so can refresh on expose */
					/* We will want to disable when redrawing a lot of stuff
					 * without erasing...
					 */
#define REMEMBER_GFX	( remember_gfx && ! from_memory )

static int currx=0,curry=0;

#define DEFAULT_EVENT_MASK	( ExposureMask | StructureNotifyMask | KeyPressMask | KeyReleaseMask )

#define CreateWindow(name,geom,w,h) _CreateWindow(QSP_ARG  name,geom,w,h)

static Window _CreateWindow(QSP_ARG_DECL  const char *name,const char *geom,u_int  w,u_int  h)
{
	Window			win;
	XSetWindowAttributes	attributes;
	u_long			valuemask;
	XWMHints		xwmh;
	XSizeHints		hints;
	int			i,x,y;
	XGCValues		gcvals;
	Colormap colormap;
	Disp_Obj *dop;

	colormap = (Colormap) NULL;	// quiet compiler
	dop = curr_dop();

	// dop can be null if user does not own display!?
	//assert( dop != NULL );
	if( dop == NULL ){
		warn("CreateWindow:  no current display!?");
		return (Window) 0;
	}

	/* note that only x,y are gotten from geom spec.  w,h are fixed */
	x = y = 50;	// has to default to something!?
	i = XParseGeometry(geom,&x,&y,&w,&h);

	if ((i&XValue || i&YValue)) hints.flags = USPosition;
						else hints.flags = PPosition;

	hints.flags |= USSize;

	if (i&XValue && i&XNegative) x = DO_WIDTH(dop) - w - abs(x);
	if (i&YValue && i&YNegative) y = DO_HEIGHT(dop) - h - abs(y);

//fprintf(stderr,"Hint posn is %d, %d\n",x,y);
	hints.x = x;				hints.y = y;
	hints.width = w;			hints.height = h;
	hints.min_width  = w;		hints.min_height = h;
	hints.max_width  = w;		hints.max_height = h;
	hints.flags |= PMaxSize | PMinSize;

	if( XGetGCValues( DO_DISPLAY(dop),DO_GC(dop),
		GCBackground,&gcvals) == 0 )
		warn("error getting GC value for bg");

	attributes.background_pixel = gcvals.background;
	attributes.border_pixel	= gcvals.background;

	valuemask = CWBackPixel | CWBorderPixel ;

#define SHOW_MASK_BITS(k)				\
fprintf(stderr,"%s = 0x%lx\n",#k,k);

//SHOW_MASK_BITS(CWBackPixel)
//SHOW_MASK_BITS(CWBorderPixel)

	attributes.event_mask = StructureNotifyMask;
	valuemask |= CWEventMask;
//SHOW_MASK_BITS(CWEventMask)


	/* On the mac, X11 sometimes aborts because of a BadMatch error
	 * (invalid parameter attributes)
	 * The random nature of this occurrence suggests that 
	 * there may be some kind of race condition...
	 *
	 * It also seems to happen slightly less frequently when the printf
	 * statement is present???
	 *
	 * Do we really need a colormap when the depth is 24 bpp???
	 * For TrueColor, the colormap is read-only, but for DirectColor
	 * we can write it?
	 *
	 * If these lines are commented out, the XCreateWindow ALWAYS fails!?
	 *
	 * AllocNone is right for TrueColor, but not for DirectColor???
	 */

//#ifdef TRY_WITHOUT_THIS
	if( DO_DEPTH(dop) == 24 ) {
//fprintf(stderr,"CreateWindow calling XCreateColormap...\n");
		colormap = XCreateColormap (DO_DISPLAY(dop), DO_ROOTW(dop),
			DO_VISUAL(dop), AllocNone);
		valuemask |= CWColormap;
//SHOW_MASK_BITS(CWColormap)
		attributes.colormap = colormap;
	}
//#endif // TRY_WITHOUT_THIS

#ifdef QUIP_DEBUG
if( debug & xdebug ){
NADVISE("XCreateWindow");
sprintf(ERROR_STRING,"dpy = %s,  dispDEEP = %d",
DO_NAME(dop),DO_DEPTH(dop));
NADVISE(ERROR_STRING);
sprintf(ERROR_STRING,"calling XCreateWindow, depth = %d",DO_DEPTH(dop));
NADVISE(ERROR_STRING);
sprintf(ERROR_STRING,"\tx = %d, y = %d, w = %d, h = %d, border = %d, vis = %ld (0x%lx)",
x,y,w,h,WINDOW_BORDER_WIDTH,(u_long)DO_VISUAL(dop),(u_long)DO_VISUAL(dop));
NADVISE(ERROR_STRING);
}
#endif
	if( w <=0 || h <= 0 ){
		warn("bad window dimensions will wedge window manager");
		abort();
	}

/*
*/

/*fprintf(stderr,"Calling XCreateWindow\n"
"\tdepth = %d\n"
"\tvaluemask = 0x%lx\n",
DO_DEPTH(dop),valuemask);*/
// what about the visual???

	win = XCreateWindow(DO_DISPLAY(dop), DO_ROOTW(dop), x, y, w, h,
		WINDOW_BORDER_WIDTH, DO_DEPTH(dop), InputOutput,
		DO_VISUAL(dop), valuemask, &attributes);

	if (!win){
		warn("error creating window");
		return(win);   /* leave immediately if couldn't create */
	}

	XMapWindow(DO_DISPLAY(dop),win);
	{
		/* what is this for? */
		XEvent event;
		XIfEvent(DO_DISPLAY(dop),&event,WaitForNotify,(char*)win);
	}
#ifdef QUIP_DEBUG
if( debug & xdebug ){
NADVISE("window created");
}
#endif

	SET_DO_CURRW(dop, win);

	if( DO_DEPTH(dop) == 24 )
		XSetWindowColormap(DO_DISPLAY(dop), DO_CURRW(dop), colormap);

	set_curr_win(win);	/* for lut_xlib */

	XSetStandardProperties(DO_DISPLAY(dop), win, name, name, None, NULL, 0, &hints);

	xwmh.input = True;
	xwmh.flags = InputHint;
	/*
	if (iconPix) { xwmh.icon_pixmap = iconPix;  xwmh.flags |= IconPixmapHint; }
	*/
	XSetWMHints(DO_DISPLAY(dop), win, &xwmh);

	XClearArea(DO_DISPLAY(dop),win,0,0,w,h,True);

	return(win);

}

#ifdef SGI_GL

/* BUG these functions should be integrated better with the normal ones,
 * too much duplicated code!
 */

static Window CreateGLWindow(char *name,char *geom,u_int w,u_int h)
{
	Window			win;
	XSetWindowAttributes	attributes;
	u_long			valuemask=0;
	XWMHints		xwmh;
	XSizeHints		hints;
	int			i,x,y;
	XGCValues		gcvals;
	Disp_Obj *dop;

	if( (dop=curr_dop()) == NULL ) return(NULL);

	/* note that only x,y are gotten from geom spec.  w,h are fixed */
	x = y = 1;
	i = XParseGeometry(geom,&x,&y,&w,&h);

	if ((i&XValue || i&YValue)) hints.flags = USPosition;
						else hints.flags = PPosition;

	hints.flags |= USSize;

	if (i&XValue && i&XNegative) x = DO_WIDTH(dop) - w - abs(x);
	if (i&YValue && i&YNegative) y = DO_HEIGHT(dop) - h - abs(y);

	hints.x = x;				hints.y = y;
	hints.width = w;			hints.height = h;
	hints.min_width  = w;		hints.min_height = h;
	hints.max_width  = w;		hints.max_height = h;
	hints.flags |= PMaxSize | PMinSize;

	if( XGetGCValues(DO_DISPLAY(dop),DO_GC(dop),
		GCBackground,&gcvals) == 0 )
		warn("error getting GC value for bg");

	attributes.background_pixel = gcvals.background;
	attributes.border_pixel	= gcvals.background;

	/*
	valuemask = CWBackPixel | CWBorderPixel ;
	*/

	win = GLXCreateWindow(DO_DISPLAY(dop), DO_ROOTW(dop), x, y, w, h,
		WINDOW_BORDER_WIDTH, valuemask, &attributes, GLXrgbSingleBuffer);

	if (!win){
		warn("error creating window");
		return(win);   /* leave immediately if couldn't create */
	}

	assert( dop != NULL );

	SET_DO_CURRW(dop, win);

	set_curr_win(win);	/* for lut_xlib */

	XSetStandardProperties(DO_DISPLAY(dop), win, name, name, None, NULL, 0, &hints);

	xwmh.input = True;
	xwmh.flags = InputHint;
/*
	if (iconPix) {
		xwmh.icon_pixmap = iconPix;
		xwmh.flags |= IconPixmapHint;
	}
*/

	XSetWMHints(DO_DISPLAY(dop), win, &xwmh);

	return(win);

}

Window creat_gl_window(const char *name,int w,int h,long event_mask)
{
	Window scrW;
	XClassHint classh;
	CARD32 data[2];
	Atom prop;
	char *s;

	/* s=savestr(name); */

	scrW = CreateGLWindow(name,def_geom,w,h);
	if(!scrW) ERROR1("can't create window");

	classh.res_name = (char *)tell_progname();
	classh.res_class = (char *)name;
	XSetClassHint(DO_DISPLAY(dop), scrW, &classh);

	data[0] = (CARD32)XInternAtom(DO_DISPLAY(dop), "WM_DELETE_WINDOW", False);
	data[1] = (CARD32)time((long *)0);
	prop = XInternAtom(DO_DISPLAY(dop), "WM_PROTOCOLS", False);


	XChangeProperty(DO_DISPLAY(dop), scrW, prop, prop,
		32, PropModeReplace, (u_char *) data, 2);

	XSelectInput(DO_DISPLAY(dop), scrW,
		  ExposureMask
		| StructureNotifyMask
		| KeyPressMask		/* receive keystrokes in windows */
		| KeyReleaseMask
		| event_mask
		);

	return(scrW);
} /* end create_gl_window() */
#endif

#define creat_window(name,w,h,event_mask) _creat_window(QSP_ARG  name,w,h,event_mask)

static Window _creat_window(QSP_ARG_DECL  const char *name,int w,int h,long event_mask)
{
	Window scrW;
	XClassHint classh;
	CARD32 data[2];
	Atom prop;
	Disp_Obj *dop;

	/*
	 * Originally, we saved the name string before passing the address
	 * to the X server...  this was because we weren't sure whether or
	 * not the server made its own copy of the name string.  (If it didn't,
	 * and we passed it the address of a string on the stack, or some
	 * other dynamically allocated area, then a later reference might
	 * produce garbage.)
	 *
	 * This didn't seem like a big deal, but an application that created
	 * and destroyed a large number of windows ended up running out of
	 * memory.  Passing the server a pointer to a transient string seems
	 * to be ok, so we will plug the leak and hope for the best...
	 */

	/* s=savestr(name); */

	dop = curr_dop();

	scrW = CreateWindow(name,def_geom,w,h);
	if(!scrW) NERROR1("can't create window");

	classh.res_name = (char *) tell_progname();
	classh.res_class = (char *)name;
	XSetClassHint(DO_DISPLAY(dop), scrW, &classh);

	data[0] = (CARD32)XInternAtom(DO_DISPLAY(dop), "WM_DELETE_WINDOW", False);
	data[1] = (CARD32)time((long *)0);
	prop = XInternAtom(DO_DISPLAY(dop), "WM_PROTOCOLS", False);


	XChangeProperty(DO_DISPLAY(dop), scrW, prop, prop,
		32, PropModeReplace, (u_char *) data, 2);

	XSelectInput(DO_DISPLAY(dop), scrW, event_mask );

	return(scrW);
} /* end creat_window */

void set_viewer_display(QSP_ARG_DECL  Viewer *vp)
{
	Disp_Obj *dop;

	dop=curr_dop();

	// dop can be null if user doesn't own the X display!
	if( dop == NULL ){
		WARN("set_viewer_display:  no current display object");
		return;
	}

	vp->vw_dop = dop;
}

#define make_generic_window(vp,w,h,m)	_make_generic_window(QSP_ARG  vp,w,h,m)

static int _make_generic_window(QSP_ARG_DECL  Viewer *vp, int width, int height, long event_mask)
{
	Window scrW;
	XGCValues values;
	const char *label;

	window_sys_init(SINGLE_QSP_ARG);

	event_mask |= DEFAULT_EVENT_MASK;
	label = VW_LABEL(vp) == NULL ? VW_NAME(vp) : VW_LABEL(vp);
	scrW=creat_window(label,width,height,event_mask);

	vp->vw_xwin = scrW;
	vp->vw_dop = curr_dop();
	vp->vw_gc = XCreateGC(VW_DPY(vp),vp->vw_xwin,0L,&values);
	vp->vw_event_mask = event_mask;

	return(0);
}

void enable_masked_events(Viewer *vp, long event_mask)
{
	vp->vw_event_mask |= event_mask;
	XSelectInput(VW_DPY(vp), vp->vw_xwin, vp->vw_event_mask );
}

#ifdef FOOBAR		// no longer needed???
static void disable_masked_events(Viewer *vp, long event_mask)
{
	vp->vw_event_mask &= ~event_mask;
	XSelectInput(VW_DPY(vp), vp->vw_xwin, vp->vw_event_mask );
}
#endif /* FOOBAR */

int _make_2d_adjuster(QSP_ARG_DECL  Viewer *vp,int width,int height)
{
	return( make_generic_window(vp, width, height, ButtonMotionMask |
				ButtonPressMask |
				ButtonReleaseMask
				/* | PointerMotionHintMask */
				) );
}

int _make_button_arena(QSP_ARG_DECL  Viewer *vp, int width, int height)
{
	/* We didn't used to look at Release events? */
	return( make_generic_window(vp, width, height, ButtonPressMask|ButtonReleaseMask ) );
}

int _make_dragscape(QSP_ARG_DECL  Viewer *vp, int width, int height)
{
	return( make_generic_window(vp, width, height, ButtonMotionMask |
				ButtonPressMask |
				ButtonReleaseMask
				/* | PointerMotionHintMask */
				) );
}

int _make_mousescape(QSP_ARG_DECL  Viewer *vp, int width, int height)
{
	return( make_generic_window(vp, width, height, PointerMotionMask
				| ButtonPressMask
				| ButtonReleaseMask
				/* | PointerMotionHintMask */
				) );
}

int _make_viewer(QSP_ARG_DECL  Viewer *vp, int width, int height)
{
	return( make_generic_window(vp,width, height, 0L) );
}

int _make_gl_window(QSP_ARG_DECL  Viewer *vp, int width, int height)
{
	return( make_generic_window(vp,width, height, 0L) );
}

static int is_mapped(Viewer *vp)
{
	XWindowAttributes attr;

	XGetWindowAttributes(VW_DPY(vp),vp->vw_xwin,&attr);

	if( attr.map_state != IsViewable ) return(0);
	return(1);
}

void _show_viewer(QSP_ARG_DECL  Viewer *vp)
{
	window_sys_init(SINGLE_QSP_ARG);

	//XMapRaised(DO_DISPLAY(dop),vp->vw_xwin);
//sprintf(ERROR_STRING,"show_viewer %s calling XMapRaised...",vp->vw_name);
//advise(ERROR_STRING);
	XMapRaised(VW_DPY(vp),vp->vw_xwin);
	// When we command a movement, the content window ends up lower
	// by the width of the top border (22 pixels)
	// Is this why we sometimes see the window march down the screen?
	XMoveWindow(VW_DPY(vp),vp->vw_xwin,VW_X_REQUESTED(vp),
					VW_Y_REQUESTED(vp));

	//usleep(100);
	XSync(VW_DPY(vp),False);

	/* but window may be mapped but not raised? */

	/* We used to check events here, but that can cause undesireable macro recursion... */
	do {
		usleep(100);
		//i_loop();		/* process events if any */
	} while( ! is_mapped(vp) );
}

void _unshow_viewer(QSP_ARG_DECL  Viewer *vp)
{
	// Does it make sense to have to do this here???
	window_sys_init(SINGLE_QSP_ARG);

	XUnmapWindow(VW_DPY(vp),vp->vw_xwin);
}

/* create a suitable image to be use with XPutImage */

#define x_image_for(vp,dp) _x_image_for(QSP_ARG  vp,dp)

static int _x_image_for(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp)
{
	/* We used to do a complicated depth calculation based on tdim
	 * and prec of dp...   Now we just use the viewer depth.
	 */

	/* If this viewer already has an XImage, reuse it
	 * if the size is appropriate.
	 */

	if( vp->vw_ip != NULL ){		/* has XImage? */
		if (vp->vw_ip->width == (int)OBJ_COLS(dp) &&
			vp->vw_ip->height == (int)OBJ_ROWS(dp) )

			return(0);

		/* The size is different, so we discard the old XImage */

		if( ! OWNS_IMAGE_DATA(vp) )
			vp->vw_ip->data=NULL;
#ifdef QUIP_DEBUG
if( debug & xdebug )
{
sprintf(ERROR_STRING,"Destroying old X image, viewer %s",vp->vw_name);
NADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		XDestroyImage(vp->vw_ip);

	} else {

#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(ERROR_STRING,"x_image_for %s:  viewer %s has no old X image",OBJ_NAME(dp),vp->vw_name);
NADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	}


#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(ERROR_STRING,"x_image_for %s:  calling XCreateImage",OBJ_NAME(dp));
NADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	vp->vw_ip = XCreateImage(VW_DPY(vp),VW_VISUAL(vp),vp->vw_depth,ZPixmap,
		/* offset */ 0, (char *) OBJ_DATA_PTR(dp),
		OBJ_COLS(dp),OBJ_ROWS(dp),8,0);

	if( vp->vw_ip == NULL ){
		warn("XCreateImage failed");
		return(-1);
	}

/*
 * on LINUX this prints LSBFirst, and we usually get BGR, i.e. c{0} goes to blue...
 * if( vp->vw_ip->byte_order == LSBFirst ) NADVISE("LSBFirst");
 * else if( vp->vw_ip->byte_order == MSBFirst ) NADVISE("MSBFirst");
 * else NADVISE("unexpected byte order???");
 */

	/* Xlib has not allocated any image memory yet */
	vp->vw_flags &= ~VIEWER_OWNS_IMAGE_DATA;

	return(0);
}

#define copy_components(n,dst_dp,dstart,dinc,src_dp,sstart,sinc) _copy_components(QSP_ARG  n,dst_dp,dstart,dinc,src_dp,sstart,sinc)

static void _copy_components(QSP_ARG_DECL  int n, Data_Obj *dst_dp, int dstart, int dinc,
				Data_Obj *src_dp, int sstart, int sinc )
{
	int i,j;
	Data_Obj *dpto=NULL,*dpfr=NULL;

	i=dstart;
	j=sstart;
	while(n--){
		if( dinc == 0 )
			dpto=dst_dp;
		else	dpto=c_subscript(dst_dp,i);

		if( sinc == 0 )
			dpfr=src_dp;
		else	dpfr=c_subscript(src_dp,j);

		dp_copy(dpto,dpfr);
		i += dinc;
		j += sinc;
	}
}

void _wait_for_mapped(QSP_ARG_DECL  Viewer *vp, int max_wait_time)
{
	/* BUG max_wait_time not implemented yet */

	//show_viewer(vp);	/* make sure mapped */

	while( ! is_mapped(vp) ){
		usleep(100);
		i_loop(SINGLE_QSP_ARG);		/* process events if any */
	}
}

void _embed_image(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp,int x,int y)
{
	/* u_long fno; */
	u_int display_bpp;
	Data_Obj *disp_dp;
	incr_t fno;

	if( display_to_mapped && ! is_mapped(vp) ){
		// Can we be sure that a mapping has been requested???
		if( verbose ) advise("embed_image:  viewer not mapped, waiting...");
		wait_for_mapped(vp,10);
	}

#ifdef QUIP_DEBUG
//if( debug & xdebug ){
//sprintf(ERROR_STRING,"embed_image %s (depth = %d), viewer %s (depth = %d)",
//OBJ_NAME(dp),8*OBJ_COMPS(dp),vp->vw_name,vp->vw_depth);
//advise(ERROR_STRING);
//}
#endif /* QUIP_DEBUG */

/*error checks exit on failure */

	if( x < 0 || y < 0 ||	x+OBJ_COLS(dp) > vp->vw_width ||
				y+OBJ_ROWS(dp) > vp->vw_height ){
		sprintf(ERROR_STRING,
	"embed_image:  Can't embed image %s (%d x %d) at %d %d in viewer %s (%d x %d)",
			OBJ_NAME(dp),OBJ_ROWS(dp),OBJ_COLS(dp),
			x,y,vp->vw_name,vp->vw_height,vp->vw_width);
		warn(ERROR_STRING);
		return;
	}

	if( vp->vw_depth == 15 || vp->vw_depth == 16 ){
		if( OBJ_PREC(dp) != PREC_IN && OBJ_PREC(dp) != PREC_UIN ){
			sprintf(ERROR_STRING,
				"embed_image:  image \"%s\" is type %s, should be short",
				OBJ_NAME(dp),OBJ_PREC_NAME(dp));
			warn(ERROR_STRING);
			return;
		}
	}
	else{
		if( OBJ_MACH_PREC(dp) != PREC_BY && OBJ_MACH_PREC(dp) != PREC_UBY ){
			if( OBJ_NAME(dp) == NULL ){
				warn("non-byte image (with null name) passed to embed_image!?");
				abort();
			}
			sprintf(ERROR_STRING,
				"embed_image:  image \"%s\" is type %s, should be %s or %s",
				OBJ_NAME(dp),OBJ_PREC_NAME(dp),
				PREC_NAME(PREC_FOR_CODE(PREC_BY)), PREC_NAME(PREC_FOR_CODE(PREC_UBY)));
			warn(ERROR_STRING);
			return;
		}
	}
/* more error checks */
	if( vp->vw_depth < 8 ){
		static int warned=0;
		if( !warned ){
			warn("Sorry, can't embed images for display depth < 8");
			warned++;
		}
		return;
	}

/*  dp should should be valid at this point so let get vp->vw_ip with x_image_for().
 *  If the function returns without error then we can use the viewer image pointer
 *  to determine 32 bpp for 24 bit color or 24bpp for 24 bit color
 */
	if( x_image_for(vp,dp) < 0 ){
		sprintf(ERROR_STRING,
			"embed_image:  can't find x_image for viewer %s, object %s",
			vp->vw_name,OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}

	display_bpp = vp->vw_ip->bytes_per_line / vp->vw_ip->width;

	if( display_bpp != OBJ_COMPS(dp) * PREC_SIZE( OBJ_MACH_PREC_PTR(dp) ) ){
		/* BUG don't deal with short case correctly here... */
		if( PREC_SIZE(OBJ_MACH_PREC_PTR(dp)) != 1 ){
			sprintf(ERROR_STRING,
	"embed_image:  expected byte precision for object %s (%d %s components)!?",
				OBJ_NAME(dp),OBJ_COMPS(dp),OBJ_PREC_NAME(dp));
			warn(ERROR_STRING);
			return;
		}
		disp_dp = comp_replicate(dp,display_bpp,ALLOC_DATA);
		if( OBJ_COMPS(dp) == 1 )
			copy_components(display_bpp,disp_dp,0,1,dp,0,0);
		else {
			int n;

			n = MIN(OBJ_COMPS(dp),OBJ_COMPS(disp_dp));
			copy_components(n,disp_dp,0,1,dp,0,1);
			unlock_children(dp);	// so they delete properly
		}
		unlock_children(disp_dp);	// so they delete properly
	} else	disp_dp = dp;


	/* allow loading of an entire sequence... */
	/* vw_frameno should be set by the caller! */
	fno=vp->vw_frameno % OBJ_FRAMES(disp_dp);

#ifdef QUIP_DEBUG
//if( debug & xdebug ){
//advise("x_image gotten");
//longlist(dp);
//}
#endif /* QUIP_DEBUG */

	vp->vw_ip->data = ((char *)OBJ_DATA_PTR(disp_dp))+fno*OBJ_FRM_INC(disp_dp);
	/* vp->vw_ip->data = (char *)OBJ_DATA_PTR(disp_dp) ; */

#ifdef HAVE_VBL
	// We call vbl_wait here with the idea that if we are
	// showing a movie then we would like for all the frames
	// to display for at least one refresh...  But if we
	// are trying to update many windows, then we really
	// do not want to do this!?  Perhaps there should be
	// a software flag to control it?
	//
	// vbl_wait depends on linux-only calls ioperm and inb!?
	// not available on MacOS...
	vbl_wait();
#endif /* HAVE_VBL */

#ifdef QUIP_DEBUG
//if( debug & xdebug ){
//advise("calling XPutImage");
//}
#endif /* QUIP_DEBUG */

	XPutImage(VW_DPY(vp), vp->vw_xwin, vp->vw_gc, vp->vw_ip,
		0, 0, x, y, OBJ_COLS(dp), OBJ_ROWS(dp));

	if( disp_dp != dp ){
		/* Is it safe to release the image now it the x server is not
		 * running synchronously???
		 */

#ifdef QUIP_DEBUG
//if( debug & xdebug ){
//sprintf(ERROR_STRING,"embed_image:  deleting disp_dp %s",OBJ_NAME(disp_dp));
//advise(ERROR_STRING);
//}
#endif /* QUIP_DEBUG */
		// memory leak on children because locked - same command cycle!?

		delvec(disp_dp);
	}
} // end embed_image

static const char *string_for_text_mode(Text_Mode m)
{
	switch(m){
	case LEFT_JUSTIFY:  return("left_justify"); break;
	case RIGHT_JUSTIFY:  return("right_justify"); break;
	case CENTER_TEXT:  return("center"); break;
	}
	// NOTREACHED
	return("bad text mode!?");
}

static void dop_info( QSP_ARG_DECL  Draw_Op *dop)
{
	switch(dop->do_op){
		case DRAW_OP_FOREGROUND:
			sprintf(msg_str,"\tselect 0x%x",dop->do_color);
			break;
		case DRAW_OP_BACKGROUND:
			sprintf(msg_str,"\tbgselect 0x%x",dop->do_color);
			break;
		case DRAW_OP_MOVE:
			sprintf(msg_str,"\tmove %d %d",dop->do_x,dop->do_y);
			break;
		case DRAW_OP_CONT:
			sprintf(msg_str,"\tcont %d %d",dop->do_x,dop->do_y);
			break;
		case DRAW_OP_TEXT:
			sprintf(msg_str,"\ttext \"%s\"",dop->do_str);
			break;
		case DRAW_OP_TEXT_MODE:
			sprintf(msg_str,"\ttext_mode \"%s\"",
				string_for_text_mode(dop->do_text_mode));
			break;
		case DRAW_OP_ARC:
			sprintf(msg_str,"\tarc %d %d",dop->do_xl,dop->do_yu);
			break;

		default:
			sprintf(ERROR_STRING,
			"dop_info:  unrecognized drawing op %d (0x%x)",
					dop->do_op,dop->do_op);
			warn(ERROR_STRING);
			break;
	}
	prt_msg(msg_str);
}

#define refresh_drawing(vp) _refresh_drawing(QSP_ARG  vp)

static void _refresh_drawing(QSP_ARG_DECL  Viewer *vp)
{
	Node *np;
	int cx=0,cy=0;
	Draw_Op *dop;
	//Handle hdl;

	if( vp->vw_drawlist == NULL ){
		return;
	}

	from_memory =1;

	np=QLIST_HEAD(vp->vw_drawlist);
	while(np!=NULL){
		//hdl = (void **) np->n_data;
		//dop = (Draw_Op *) *hdl;
		dop = (Draw_Op *) np->n_data;

#ifdef QUIP_DEBUG
if( debug & xdebug ){
dop_info(DEFAULT_QSP_ARG  dop);
}
#endif
		switch(dop->do_op){
			case DRAW_OP_FOREGROUND:
				xp_select(vp,dop->do_color);
				break;
			case DRAW_OP_BACKGROUND:
				xp_bgselect(vp,dop->do_color);
				break;
			case DRAW_OP_MOVE:
				cx = dop->do_x;
				cy = dop->do_y;
				break;
			case DRAW_OP_CONT:
				xp_line(vp,cx,cy,dop->do_x,dop->do_y);
				cx = dop->do_x;
				cy = dop->do_y;
				break;
			case DRAW_OP_LINEWIDTH:
				_xp_linewidth(vp,dop->do_lw);
				break;
			case DRAW_OP_TEXT:
				if( dop->do_xfp != NULL ){
					set_font(vp,dop->do_xfp);
				}
				xp_text(vp,cx,cy,dop->do_str);
				break;
			case DRAW_OP_TEXT_MODE:
				switch(dop->do_text_mode){
					case LEFT_JUSTIFY:
						left_justify(vp);
						break;
					case RIGHT_JUSTIFY:
						right_justify(vp);
						break;
					case CENTER_TEXT:
						center_text(vp);
						break;
					default:
		assert( ! "refresh_drawing:  Bad text justification mode!?");
						break;
				}
				break;
			case DRAW_OP_ARC:
				if( dop->do_filled )
					_xp_fill_arc(vp,dop->do_xl,dop->do_yu,dop->do_w,
						dop->do_h,dop->do_a1,dop->do_a2);
				else
					_xp_arc(vp,dop->do_xl,dop->do_yu,dop->do_w,
						dop->do_h,dop->do_a1,dop->do_a2);
				break;

			default:
				sprintf(ERROR_STRING,
			"refresh_drawing:  unrecognized drawing op %d (0x%x)",
					dop->do_op,dop->do_op);
				warn(ERROR_STRING);
				break;
		}
		np=np->n_next;
	}

	from_memory=0;
}

void _unembed_image(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp,int x,int y)
{
	u_long plane_mask;
	u_int display_bpp;

	/* We don't need to be mapped to display (although nothing will
	 * happen at all if we're not mapped).
	 * In this case, however, if the window is not mapped we will bomb out of X
	 * with an error in XGetImage.
	 *
	 * We also bomb out if any part of the window that we are trying to
	 * extract falls off the edge of the screen...  We need to check this,
	 * but how do we find out the width of the window border?
	 */

	wait_for_mapped(vp,1/* this arg is not currently used!? */ );

	/* Now make sure that the target image is appropriate for this depth */
	if( x < 0 || y < 0 ||	x+OBJ_COLS(dp) > vp->vw_width  ||
				y+OBJ_ROWS(dp) > vp->vw_height ){

		sprintf(ERROR_STRING,
	"Can't extract image %s (%d x %d) from %d %d in viewer %s (%d x %d)",
			OBJ_NAME(dp),OBJ_ROWS(dp),OBJ_COLS(dp),
			x,y,vp->vw_name,vp->vw_height,vp->vw_width);
		warn(ERROR_STRING);
		return;
	}

	if( vp->vw_ip2 != NULL ){
		/* this data is always allocated by Xlib
		 * so we don't need to check...
		 */
		/* if( ! OWNS_IMAGE2_DATA(vp) ) vp->vw_ip2->data=NULL; */
		XDestroyImage(vp->vw_ip2);
	}

	/* XGetImage will fail if the viewer is not mapped -
	 * how do we insure this?
	 *
	 * here we test if mapped (open), but really also need to make sure in front!
	 *
	 * BUG How can we figure out if the window is in front?
	 * for now, we just comment out the test...
	 */


	/* if( ! is_mapped(vp) ){ */

		XMapRaised(VW_DPY(vp),vp->vw_xwin);
		usleep(100);
		XSync(VW_DPY(vp),False);

		/*
		 * To be sure, we could explicitly call refresh_drawing here?
		 * But how do we know when the window is mapped?
		 * The client program (us) thinks it is mapped now,
		 * but the expose event is probably not processed!
		 */

		while( ! is_mapped(vp) )	/* wait for window to be mapped */
			;

		usleep(100);
		//sleep(1);			/* give the server a chance to map */
		//redraw_viewer(vp);
	/* } */


	/* should we do something based on depth??? */
	plane_mask = 0xffffffff;

	vp->vw_ip2=XGetImage(VW_DPY(vp), vp->vw_xwin,
		x, y, OBJ_COLS(dp), OBJ_ROWS(dp), plane_mask, ZPixmap );

	if( vp->vw_ip2 == NULL ){
		warn("error getting X image");
		return;
	}

	display_bpp = vp->vw_ip2->bytes_per_line / vp->vw_ip2->width;

	if( (display_bpp != OBJ_COMPS(dp) * PREC_SIZE( OBJ_MACH_PREC_PTR(dp) ) ) ){
		/* in general, the depths should match, but we allow
		 * 3 and 4 to match each other...
		 */
		if( display_bpp == 3 && OBJ_COMPS(dp) == 4 ){
			/* ok */
		} else if ( display_bpp == 4 && OBJ_COMPS(dp) == 3 ){
			/* ok */
		} else {
			sprintf(ERROR_STRING,
	"unembed_image:  display has %d bpp, image %s (%s) has depth %d",
				display_bpp,OBJ_NAME(dp),OBJ_PREC_NAME(dp),
				OBJ_COMPS(dp));
			warn(ERROR_STRING);
			return;
		}
		/* do we need to do something with the 3/4 depth mismatch? */
	}


	/* copy the data from the XImage to the data object area */

	if( display_bpp == OBJ_COMPS(dp)*PREC_SIZE( OBJ_MACH_PREC_PTR(dp) ) )
		memcpy(OBJ_DATA_PTR(dp),vp->vw_ip2->data,
			OBJ_ROWS(dp)*OBJ_COLS(dp)*display_bpp);
	else {
		Data_Obj *disp_dp;
		int n;

		disp_dp = comp_replicate(dp,display_bpp,DONT_ALLOC_DATA);
		SET_OBJ_DATA_PTR(disp_dp, vp->vw_ip2->data );

		n = MIN(OBJ_COMPS(dp),OBJ_COMPS(disp_dp));
		copy_components(n,dp,0,1,disp_dp,0,1);
		unlock_children(disp_dp);	// so they delete properly
		delvec(disp_dp);
	}

	/* set the flag to show the image has some stuff */
	SET_OBJ_FLAG_BITS(dp, DT_ASSIGNED);

	propagate_flag_to_children(dp,DT_ASSIGNED);

} /* end unembed_image() */

#define refresh_image(vp) _refresh_image(QSP_ARG  vp)

static void _refresh_image(QSP_ARG_DECL  Viewer *vp)
{
	if( ! is_mapped(vp) )
		return;

	if( vp->vw_dp != NULL )
		embed_image(vp,vp->vw_dp,0,0);
}

void _redraw_viewer(QSP_ARG_DECL  Viewer *vp)
{
	//time_t now_time;

	if( ! is_mapped(vp) ){
		return;
	}

	/* We used to check the time of the last redraw here, but after
	 * reading the xlib docs a little it appears that it's not necessary!
	 */

#ifdef FOOBAR
	if( (now_time=time((time_t *)NULL)) == (time_t) -1 ){
		warn("redraw_viewer:  error getting time");
		return;
	}
	if( (now_time - vp->vw_time) <= 1 ){

#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(ERROR_STRING,"not redrawing viewer %s, drawn within last 1 sec",
vp->vw_name);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

		return;
	}
#endif /* FOOBAR */

	if( IS_DRAGSCAPE(vp) ) update_image(vp);

	refresh_image(vp);

	x_dump_lut( VW_DPYABLE(vp) );		/* why do we need to do this?? */
	refresh_drawing(vp);

#ifdef FOOBAR
	/* We do this again now, because refreshing the drawing can take a long time */
	if( (now_time=time((time_t *)NULL)) == (time_t) -1 ){
		warn("redraw_viewer:  error getting time");
		return;
	}

	vp->vw_time = now_time;
#endif /* FOOBAR */

}

/* BUG should load default colormap */

void posn_viewer(Viewer *vp,int x,int y)
{
	//vp->vw_x = x;
	//vp->vw_y = y;
	SET_VW_X_REQUESTED(vp,x);
	SET_VW_Y_REQUESTED(vp,y);
	SET_VW_FLAG_BITS(vp, VW_PROG_MOVE_REQ);
//fprintf(stderr,"requested position set to %d, %d\n",x,y);
	// These are now set by StructureNotify events...
	// move window seems to be relative!?
	XMoveWindow(VW_DPY(vp),vp->vw_xwin,x,y);
}

void zap_viewer(Viewer *vp)
{
	XDestroyWindow(VW_DPY(vp),vp->vw_xwin);
}

/* The "old" method, using XSetWMName() caused Xlib errors on the newer
 * linux systems...  this method, using XStoreName seems to work ok,
 * but has not been tested on other platforms.
 */

void relabel_viewer(Viewer *vp,const char *s)
{
#ifdef OLD_LABEL
	XTextProperty xtp;
#endif /* OLD_LABEL */

	if( VW_LABEL(vp) != NULL  ){
		rls_str((char *)VW_LABEL(vp));
	}
	SET_VW_LABEL(vp, savestr(s));

#ifdef OLD_LABEL
	XGetWMName(DO_DISPLAY(dop),vp->vw_xwin,&xtp);

	xtp.value=(u_char *)VW_LABEL(vp);
	xtp.nitems=strlen(s);

	XSetWMName(DO_DISPLAY(dop),vp->vw_xwin,&xtp);
#endif /* OLD_LABEL */

	XStoreName(VW_DPY(vp),vp->vw_xwin,VW_LABEL(vp));
}

void set_font(Viewer *vp,XFont *xfp)
{
	current_xfp = xfp;

	XSetFont(VW_DPY(vp),vp->vw_gc,xfp->xf_id);
}

int _get_string_width(QSP_ARG_DECL  Viewer *vp, const char *s)
{
	int n;

	/* We use current_xfp for now, but really we should query the font from the viewer... */
	if( current_xfp == NULL ){
		warn("get_string_width:  need to specify a font before calling this function...");
		return(-1);
	}
	n = XTextWidth(current_xfp->xf_fsp,s,strlen(s));
	return(n);
}

void _set_font_size(QSP_ARG_DECL  Viewer *vp, int s)
{
	warn("set_font_size:  not implemented");
}

void _set_text_angle(QSP_ARG_DECL  Viewer *vp, float a)
{
	warn("set_text_angle:  not implemented");
}

// BUG text_mode should be a viewer property...
// Or a per-thread viewer property???
//
// Why do we use handles for the draw list??
// They don't have to be relocated???

static Text_Mode text_mode = LEFT_JUSTIFY;

static void remember_drawing(Viewer *vp,Draw_Op_Code op,Draw_Op_Args *doap)
{
	Node *np;
	Draw_Op *dop;
	//Handle hdl;

	assert( REMEMBER_GFX );

	if( vp->vw_drawlist == NULL ){
		vp->vw_drawlist = new_list();
	}

	if( unused_dop_list != NULL &&
		(np=remHead(unused_dop_list)) != NULL ){

		dop = (Draw_Op *) np->n_data;
	} else {
		dop = getbuf(sizeof(*dop));
		np = mk_node(dop);
	}

	//dop = (Draw_Op *) *hdl;
	dop->do_op = op;
	dop->do_doa = *doap;

	addTail(vp->vw_drawlist,np);
}

static void remember_text_mode(Viewer *vp,Text_Mode m)
{
	Draw_Op_Args doa;

	if( vp != NULL && !quick ){
		doa.doa_text_mode = m;;
		remember_drawing(vp,DRAW_OP_TEXT_MODE,&doa);
	}
}


/* stuff to refresh drawings - formerly in libview, xplot.c
 * Put here to make compatible with drawmenu.
 */

static void remember_move(Viewer *vp,int x,int y)
{
	Draw_Op_Args doa;

	if( vp != NULL && !quick ){
		doa.doa_x = x;
		doa.doa_y = y;
		remember_drawing(vp,DRAW_OP_MOVE,&doa);
	}
}

static void remember_cont(Viewer *vp,int x,int y)
{
	Draw_Op_Args doa;

	if( vp != NULL && !quick ){
		doa.doa_x = x;
		doa.doa_y = y;
		remember_drawing(vp,DRAW_OP_CONT,&doa);
	}
}

static void remember_linewidth(Viewer *vp, int w)
{
	Draw_Op_Args doa;

	if( vp != NULL && !quick ){
		doa.doa_lw = w;
		remember_drawing(vp,DRAW_OP_LINEWIDTH,&doa);
	}
}

static void remember_text(Viewer *vp,const char *s)
{
	Draw_Op_Args doa;

	if( vp != NULL && !quick ){
		doa.doa_str = savestr(s);
		doa.doa_xfp = current_xfp;
		remember_drawing(vp,DRAW_OP_TEXT,&doa);
	}
}

static void remember_arc(Viewer *vp,int xl,int yu,int w,int h,int a1,int a2,int filled)
{
	Draw_Op_Args doa;

	if( vp != NULL && !quick ){
		doa.doa_xl = xl;
		doa.doa_yu = yu;
		doa.doa_w = w;
		doa.doa_h = h;
		doa.doa_a1 = a1;
		doa.doa_a2 = a2;
		doa.doa_filled = filled;
		remember_drawing(vp,DRAW_OP_ARC,&doa);
	}
}

static void remember_fg(Viewer *vp,u_long color)
{
	Draw_Op_Args doa;

	if( vp != NULL ){
		if( !quick ){
			doa.doa_color = color;
			remember_drawing(vp,DRAW_OP_FOREGROUND,&doa);
		}
	}
}

static void remember_bg(Viewer *vp,u_long color)
{
	Draw_Op_Args doa;

	if( vp != NULL ){
		if( !quick ){
			doa.doa_color = color;
			remember_drawing(vp,DRAW_OP_BACKGROUND,&doa);
		}
	}
}

static void free_drawlist(Viewer *vp)
{
	Node *np;

	if( vp->vw_drawlist==NULL ) return;

	if( unused_dop_list == NULL )
		unused_dop_list = new_list();

	while( (np=remHead(vp->vw_drawlist)) != NULL ){
		//Handle hdl;
		Draw_Op *dop;

		//hdl = (void **) np->n_data;
		//dop = (Draw_Op *) *hdl;
		dop = (Draw_Op *) np->n_data;

		/* If this is a string op, free the string! */

		if( dop->do_op == DRAW_OP_TEXT ){
			rls_str(dop->do_str);
		}

		addHead(unused_dop_list,np);
	}
}

static void forget_drawing(Viewer *vp)
{
	if( vp != NULL && !quick )
		free_drawlist(vp);
}

void set_remember_gfx(int flag)
{
	remember_gfx=flag;
}

void _dump_drawlist(QSP_ARG_DECL  Viewer *vp)
{
	Node *np;

	if( vp->vw_drawlist == NULL ) return;
	np=QLIST_HEAD(vp->vw_drawlist);
	/*
	 * The plotting space is in terms of the window size
	 */
	sprintf(msg_str,"space 0 0 %d %d",vp->vw_width-1,vp->vw_height-1);
	prt_msg(msg_str);

	while(np!=NULL){
		//Handle hdl;
		Draw_Op *dop;

		//hdl=(void **)np->n_data;
		//dop = (Draw_Op *) *hdl;
		dop = (Draw_Op *) np->n_data;

		switch(dop->do_op){
			case DRAW_OP_FOREGROUND:
				sprintf(msg_str,"select %d",dop->do_color);
				break;
			case DRAW_OP_BACKGROUND:
				sprintf(msg_str,"background %d",dop->do_color);
				break;
			case DRAW_OP_MOVE:
				sprintf(msg_str,"move %d %d",dop->do_x,dop->do_y);
				break;
			case DRAW_OP_CONT:
				sprintf(msg_str,"cont %d %d",dop->do_x,dop->do_y);
				break;
			case DRAW_OP_TEXT:
				sprintf(msg_str,"font \"%s\"",dop->do_xfp->xf_name);
				sprintf(msg_str,"text \"%s\"",dop->do_str);
				break;
			case DRAW_OP_TEXT_MODE:
				sprintf(msg_str,"text_mode \"%s\"",
					string_for_text_mode(dop->do_text_mode) );
				break;
			case DRAW_OP_ARC:
				sprintf(msg_str,"arc %d %d %d %d %d %d",
					dop->do_xl,dop->do_yu,dop->do_w,
					dop->do_h,dop->do_a1,dop->do_a2);
				break;
			default:
				sprintf(ERROR_STRING,
					"bad draw op %d\n",dop->do_op);
				warn(ERROR_STRING);
				msg_str[0]=0;
				break;
		}
		if( msg_str[0] != 0 )
			prt_msg(msg_str);

		np=np->n_next;
	}
}

void _center_text(QSP_ARG_DECL  Viewer *vp)
{
	text_mode = CENTER_TEXT;
	if( REMEMBER_GFX ) remember_text_mode(vp,text_mode);
}

void _right_justify(QSP_ARG_DECL  Viewer *vp)
{
	text_mode = RIGHT_JUSTIFY;
	if( REMEMBER_GFX ) remember_text_mode(vp,text_mode);
}

void _left_justify(QSP_ARG_DECL  Viewer *vp)
{
	text_mode = LEFT_JUSTIFY;
	if( REMEMBER_GFX ) remember_text_mode(vp,text_mode);
}

void _xp_text(QSP_ARG_DECL  Viewer *vp,int x,int y,const char *s)
{
	int dir, ascent, descent;
	XCharStruct overall;
	int h_offset=0;
	int orig_x;

	orig_x = x;
	if( text_mode != LEFT_JUSTIFY ){
		if( current_xfp == NULL ){
			warn("_xp_text:  no font specified, can't center text");
		} else {
			XTextExtents(current_xfp->xf_fsp,s,strlen(s),
				&dir,&ascent,&descent,&overall);

			if( text_mode == CENTER_TEXT )
				h_offset = overall.width/2;
			else if( text_mode == RIGHT_JUSTIFY )
				h_offset = overall.width;
		}
	}

	x -= h_offset;
	if( x < 0 ){
		sprintf(ERROR_STRING,"_xp_text:  negative x offset (%d) requested",
			x);
		warn(ERROR_STRING);
		x=0;
	}

//proceed:

	XDrawString(VW_DPY(vp),vp->vw_xwin,vp->vw_gc,
		x,y,s,strlen(s));

	if( REMEMBER_GFX ){
		remember_move(vp,orig_x,y);
		remember_text(vp,s);
	}
}

void _xp_line(QSP_ARG_DECL  Viewer *vp,int x1,int y1,int x2,int y2)
{
#ifdef QUIP_DEBUG
if( debug & xdebug ){
sprintf(ERROR_STRING,"XDrawLine %s, %d %d %d %d",vp->vw_name,x1,y1,x2,y2);
NADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	XDrawLine(VW_DPY(vp),vp->vw_xwin,vp->vw_gc,x1,y1,x2,y2);

	if( REMEMBER_GFX ){
		remember_move(vp,x1,y1);
		remember_cont(vp,x2,y2);
	}
}

void _xp_linewidth(Viewer *vp,int w)
{
	//sprintf(msg_str,"\tGC\t0x%lx",(u_long)vp->vw_gc);
	//prt_msg(msg_str);
	SET_VW_LINE_WIDTH(vp,w);

	XSetLineAttributes(VW_DPY(vp),VW_GC(vp),	VW_LINE_WIDTH(vp),
							VW_LINE_STYLE(vp),
							VW_CAP_STYLE(vp),
							VW_JOIN_STYLE(vp) );
	if( REMEMBER_GFX ){
		remember_linewidth(vp,w);
	}

	// how do we check for errors???
}

void _xp_cont(Viewer *vp,int x,int y)
{
	XDrawLine(VW_DPY(vp),vp->vw_xwin,vp->vw_gc,currx,curry,x,y);
	if( REMEMBER_GFX ){
		remember_cont(vp,x,y);
	}
}

void _xp_move(Viewer *vp,int x,int y)
{
	currx = x;
	curry = y;
	if( REMEMBER_GFX )
		remember_move(vp,currx,curry);
}

void _xp_arc(Viewer *vp,int xl,int yu,int w,int h,int a1,int a2)
{
	XDrawArc(VW_DPY(vp),vp->vw_xwin,vp->vw_gc,
		xl,yu,w,h,a1,a2);

	if( REMEMBER_GFX ){
		remember_arc(vp,xl,yu,w,h,a1,a2,0);
	}
}

void _xp_fill_arc(Viewer *vp,int xl,int yu,int w,int h,int a1,int a2)
{
	XFillArc(VW_DPY(vp),vp->vw_xwin,vp->vw_gc,
		xl,yu,w,h,a1,a2);

	if( REMEMBER_GFX ){
		remember_arc(vp,xl,yu,w,h,a1,a2,1);
	}
}

void _xp_fill_polygon(Viewer* vp, int num_points, int* px_vals, int* py_vals)
{
	XPoint* pxp = (XPoint *) getbuf(sizeof(XPoint) * num_points);
	int i;

	for (i=0; i < num_points; i++) {
		pxp[i].x = px_vals[i];
		pxp[i].y = py_vals[i];
	}

	XFillPolygon(VW_DPY(vp), vp->vw_xwin, vp->vw_gc, pxp, num_points, Nonconvex, CoordModeOrigin);

	givbuf(pxp);
}

void _xp_erase(QSP_ARG_DECL  Viewer *vp)
{
	XClearWindow(VW_DPY(vp),vp->vw_xwin);
	forget_drawing(vp);
	// The call to forget_drawing was commented out, but I think that
	// was an attempt to find out why the digits in a timer weren't displaying.
}

void _xp_update(Viewer *vp)
{
	/* a no-op */
}

void _xp_select(QSP_ARG_DECL  Viewer *vp,u_long color)
{
	if( REMEMBER_GFX )
		remember_fg(vp,color);

	if( SIMULATING_LUTS(vp) )
		color = simulate_lut_mapping(vp,color);

	XSetForeground(VW_DPY(vp),vp->vw_gc,color);
}

void _xp_bgselect(QSP_ARG_DECL  Viewer *vp,u_long color)
{
	if( REMEMBER_GFX )
		remember_bg(vp,color);

	if( SIMULATING_LUTS(vp) ){
		color = simulate_lut_mapping(vp,color);
	}

	/*
	XSetBackground(VW_DPY(vp),vp->vw_gc,c);
	*/
	XSetWindowBackground(VW_DPY(vp),vp->vw_xwin,color);
}

#ifdef FOOBAR		// no longer needed???
static void get_geom(Viewer* vp, u_int* width, u_int* height, u_int* depth)
{
	Window root;
	int x,y;  /* who cares */
	u_int border_width;
	if( XGetGeometry(VW_DPY(vp),vp->vw_top.c_xwin,&root,&x,&y,width,height,
			 &border_width,depth) == False ){
		warn("can't get geometry");
		return;
	}

}
#endif /* FOOBAR */

void _show_geom(QSP_ARG_DECL  Viewer *vp)
{
	Window root;
	u_int width, height, border_width, depth;
	int x,y;

	if( XGetGeometry(VW_DPY(vp),VW_XWIN(vp),&root,&x,&y,&width,&height,
		&border_width,&depth) == False ){
		warn("can't get geometry");
		return;
	}

	sprintf(msg_str,"window is at %d, %d, size %d by %d",x,y,width,height);
	prt_msg(msg_str);
	sprintf(msg_str,"border width is %d, depth is %d",border_width,depth);
	prt_msg(msg_str);
}

void _extra_viewer_info(QSP_ARG_DECL  Viewer *vp)
{
	sprintf(msg_str,"\tDisplay\t0x%lx",(u_long)VW_DPY(vp));
	prt_msg(msg_str);
	sprintf(msg_str,"\tScreen\t0x%x",VW_SCREEN_NO(vp));
	prt_msg(msg_str);
	sprintf(msg_str,"\tGC\t0x%lx",(u_long)vp->vw_gc);
	prt_msg(msg_str);
	sprintf(msg_str,"\tWindow\t0x%lx",(u_long)vp->vw_xwin);
	prt_msg(msg_str);
}

void insert_image(Data_Obj *dpto,Data_Obj *dpfr,int x,int y,int frameno)
{
	u_char *to, *from;
	incr_t i,j;

	from = (u_char *) OBJ_DATA_PTR(dpfr);
	to = (u_char *) OBJ_DATA_PTR(dpto);

	frameno %= OBJ_FRAMES(dpfr);

	for(j=0;j<OBJ_ROWS(dpfr);j++){
		if( y+j < 0 || y+j >= OBJ_ROWS(dpto) ) continue;
		for(i=0;i<OBJ_COLS(dpfr);i++){
			if( x+i < 0 || x+i >= OBJ_COLS(dpto) ) continue;

			*(to + (x + i)*OBJ_PXL_INC(dpto)
			     + (y + j)*OBJ_ROW_INC(dpto)) =
			*(from + frameno*OBJ_FRM_INC(dpfr)
			       + i*OBJ_PXL_INC(dpfr)
			       + j*OBJ_ROW_INC(dpfr));
		}
	}
}

void embed_draggable(Data_Obj *dp,Draggable *dgp)
{
	int i,j;
	int bit,word,words_per_row;
	WORD_TYPE *words;
	u_char *from, *to;
	incr_t x,y;	// instead of dimension_t, so we can check for underflow

	/* we could do this using warrior bitmask selection copy... */

	words_per_row = (dgp->dg_width+WORDLEN-1)/WORDLEN;
	words = (WORD_TYPE *) OBJ_DATA_PTR(dgp->dg_bitmap);
	from = (u_char *) OBJ_DATA_PTR(dgp->dg_image);
	to = (u_char *) OBJ_DATA_PTR(dp);
	x=dgp->dg_x;
	y=dgp->dg_y;

	for(j=0;j<dgp->dg_height;j++){
		if( y+j < 0 || y+j >= OBJ_ROWS(dp) ) continue;
		for(i=0;i<dgp->dg_width;i++){
			if( x+i < 0 || x+i >= OBJ_COLS(dp) ) continue;

			/* check the bitmask */
			word = i/WORDLEN;
			bit = 1 << (i%WORDLEN);

			if( (words[ j*words_per_row + word ] & bit) == 0 )
				continue;

			/* now copy the pixel */

			*(to + x + i + (y+j)*OBJ_ROW_INC(dp)) =
				*(from + i + j*OBJ_ROW_INC(dgp->dg_image));
		}
	}
}

void update_image(Viewer *vp)
{
	/* update the shadow image */

	Node *np;
	Window_Image *wip;
	Draggable *dgp;

	np=QLIST_HEAD(vp->vw_image_list);
	if( vp->vw_dp == NULL ){
	/*
		sprintf(ERROR_STRING,
	"update_image:  no associated data object for viewer %s",vp->vw_name);
		warn(ERROR_STRING);
	*/
		return;
	}
	while(np!=NULL){
		wip=(Window_Image *)np->n_data;
		insert_image(vp->vw_dp,wip->wi_dp,wip->wi_x,wip->wi_y,
			vp->vw_frameno);
		np=np->n_next;
	}
	np=QLIST_HEAD(vp->vw_draglist);
	while(np!=NULL){
		dgp=(Draggable *)np->n_data;
		embed_draggable(vp->vw_dp,dgp);
		np=np->n_next;
	}
}

#ifdef FOOBAR
unsigned long convert_color_8to24(unsigned int ui8bitcolor)
{
	return (cm_image[0][ui8bitcolor] + 256 * cm_image[1][ui8bitcolor] + 65536 * cm_image[2][ui8bitcolor]);
}
#endif /* FOOBAR */

#ifdef HAVE_X11_EXT

/* shared mem stuff */


/* global X objects */
// BUG - not thread-safe!?
static XImage *shmimage;
static XShmSegmentInfo* shminfo;
static int have_shmimage=0;
static int shm_bpp=0;


#ifdef NOT_USED
void refresh_shm_window(Viewer *vp)
{
	assert( have_shmimage );

	/* Draw screen onto display */
	XShmPutImage(VW_DPY(vp) , vp->vw_xwin, vp->vw_gc, shmimage,
														0, 0, 0, 0, vp->vw_width, vp->vw_height, False);
	XSync(VW_DPY(vp), 0);
}
#endif /* NOT_USED */

int shm_setup(Viewer *vp)
{
	/* SHARED MEMORY PORTION */

	shminfo = (XShmSegmentInfo*) getbuf (sizeof(XShmSegmentInfo));
	//XMapRaised(DO_DISPLAY(dop),vp->vw_xwin);
	XMapRaised(VW_DPY(vp),vp->vw_xwin);
	//usleep(100);
	XSync(VW_DPY(vp),False);

	shmimage = XShmCreateImage(VW_DPY(vp), VW_VISUAL(vp),
					vp->vw_depth,
					ZPixmap,
					NULL,
					shminfo,
					vp->vw_width, vp->vw_height);

	/* We can calculate bits per pixel by dividing the bytes_per_line member
	 * by the image width.  This is necessary because some X servers (sgi, Xaccel)
	 * use 32 bpp even when the depth is 24, but XF86 uses 24 bpp!?
	 */

	shm_bpp = shmimage->bytes_per_line / shmimage->width;

	if ((shminfo->shmid = shmget(IPC_PRIVATE,  /*ftok(DevName, 'v'),*/
				shmimage->bytes_per_line * shmimage->height,
				IPC_CREAT | 0777)) == -1) {
		perror("Shared memory error (shmget)");
		return(-1);
	}
	if( (shminfo->shmaddr = (char *) shmat(shminfo->shmid, (void *)0, 0))
		== (char *)(-1)) {
		perror("Shared memory error (shmat)");
		return(-1);
	}
	shmimage->data = shminfo->shmaddr;

	XShmAttach(VW_DPY(vp), shminfo);	/* check return val?? */
	have_shmimage=1;
	return(0);
}

/* This code was cribbed from mvid.c, to update a window
 * from the meteor frame grabber...  The frame grabber is assumed
 * to be grabbing in 24 bit mode.
 */

void update_shm_viewer(Viewer *vp,char *src,int pinc,int cinc,int dx,int dy,int x0,int y0)
{
	char *dest;
	int x,y;

	assert( have_shmimage );

	/* copy the data into the shared memory object */

	dest = shmimage->data;
	dest += x0 + vp->vw_width*y0; /* offset in bytes? what about 24 bpp ?  BUG? */
	for(y=0;y<dy;y++){
		for(x=0;x<dx;x++){
			int i;
			char *cptr;
			cptr = src;
			for(i=0;i<shm_bpp;i++){
				*dest++ = *cptr;
				cptr += cinc;		/* add component increment */
			}
			src+=pinc;			/* add pixel increment; =4 for rgbx */
		}
		dest += vp->vw_width - dx;
	}

#ifdef HAVE_VBL
	vbl_wait();
#endif /* HAVE_VBL */

	/* Draw screen onto display */
	XShmPutImage(VW_DPY(vp), vp->vw_xwin, vp->vw_gc, shmimage,
		0, 0, 0, 0, vp->vw_width, vp->vw_height, False);
	XSync(VW_DPY(vp), 0);
} // end update_shm_viewer

#endif /* HAVE_X11_EXT */


#else /* ! HAVE_X11 */

void posn_viewer(Viewer *vp,int x,int y)
{ UNIMP_MSG(posn_viewer) }

void show_viewer(QSP_ARG_DECL  Viewer *vp)
{ UNIMP_MSG(show_viewer) }

void unshow_viewer(QSP_ARG_DECL  Viewer *vp)
{ UNIMP_MSG(unshow_viewer) }

void extra_viewer_info(QSP_ARG_DECL  Viewer *vp)
{ UNIMP_MSG(extra_viewer_info) }

void zap_viewer(Viewer *vp)
{ UNIMP_MSG(zap_viewer) }

int make_mousescape(QSP_ARG_DECL  Viewer *vp, int width, int height)
{
	UNIMP_MSG(make_mousescape)
	return -1;
}

int make_viewer(QSP_ARG_DECL  Viewer *vp, int width, int height)
{
	UNIMP_MSG(make_viewer)
	return -1;
}

int make_button_arena(QSP_ARG_DECL  Viewer *vp, int width, int height)
{
	UNIMP_MSG(make_button_arena)
	return -1;
}

int make_2d_adjuster(QSP_ARG_DECL  Viewer *vp,int width,int height)
{
	UNIMP_MSG(make_2d_adjuster)
	return -1;
}

int make_dragscape(QSP_ARG_DECL  Viewer *vp, int width, int height)
{
	UNIMP_MSG(make_dragscape)
	return -1;
}

int get_string_width(Viewer *vp, const char *s)
{
	UNIMP_MSG(get_string_width)
	return -1;
}

void _xp_arc(Viewer *vp,int xl,int yu,int w,int h,int a1,int a2)
{ UNIMP_MSG(_xp_arc) }

void _xp_text(Viewer *vp,int x,int y,const char *s)
{ UNIMP_MSG(_xp_text) }

void _xp_line(Viewer *vp,int x1,int y1,int x2,int y2)
{ UNIMP_MSG(_xp_line) }

void _xp_linewidth(Viewer *vp,int w)
{ UNIMP_MSG(_xp_linewidth) }

void _xp_cont(Viewer *vp,int x,int y)
{ UNIMP_MSG(_xp_cont) }

void _xp_move(Viewer *vp,int x,int y)
{ UNIMP_MSG(_xp_move) }

void _xp_fill_polygon(Viewer* vp, int num_points, int* px_vals, int* py_vals)
{ UNIMP_MSG(_xp_fill_polygon) }

void _xp_fill_arc(Viewer *vp,int xl,int yu,int w,int h,int a1,int a2)
{ UNIMP_MSG(_xp_fill_arc) }

void _xp_erase(Viewer *vp)
{ UNIMP_MSG(_xp_erase) }

void _xp_update(Viewer *vp)
{ UNIMP_MSG(_xp_update) }

void _xp_select(Viewer *vp,u_long color)
{ UNIMP_MSG(_xp_select) }

void _xp_bgselect(Viewer *vp,u_long color)
{ UNIMP_MSG(_xp_bgselect) }

void embed_image(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp,int x,int y)
{ UNIMP_MSG(embed_image) }

void redraw_viewer(QSP_ARG_DECL  Viewer *vp)
{ UNIMP_MSG(redraw_viewer) }

void unembed_image(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp,int x,int y)
{ UNIMP_MSG(unembed_image) }

void dump_drawlist(QSP_ARG_DECL  Viewer *vp)
{ UNIMP_MSG(dump_drawlist) }

void set_font_size(Viewer *vp, int s)
{ UNIMP_MSG(set_font_size) }

void center_text(Viewer *vp)
{ UNIMP_MSG(center_text) }

void right_justify(Viewer *vp)
{ UNIMP_MSG(right_justify) }

void left_justify(Viewer *vp)
{ UNIMP_MSG(left_justify) }

void set_text_angle(Viewer *vp, float a)
{ UNIMP_MSG(set_text_angle) }

void set_remember_gfx(int flag)
{ UNIMP_MSG(set_remember_gfx) }

void embed_draggable(Data_Obj *dp,Draggable *dgp)
{ UNIMP_MSG(embed_draggable) }

void relabel_viewer(Viewer *vp,const char *s)
{ UNIMP_MSG(relabel_viewer) }

#endif /* ! HAVE_X11 */

void _cycle_viewer_images(QSP_ARG_DECL  Viewer *vp, int frame_duration )
{
	// BUG are we displaying the head or the tail???
	Node *np;
	Window_Image *wip;

	assert( VW_IMAGE_LIST(vp) != NULL );
	assert( QLIST_HEAD( VW_IMAGE_LIST(vp) ) != NULL );

	np = remHead( VW_IMAGE_LIST(vp) );
	addTail( VW_IMAGE_LIST(vp), np );
	np = QLIST_HEAD( VW_IMAGE_LIST(vp) );

	wip = (Window_Image *) NODE_DATA(np);
	embed_image(vp,wip->wi_dp,wip->wi_x,wip->wi_y);

	// embed_image does one vbl_wait
#ifdef HAVE_VBL
	// BUG it would be better to return, do other stuff, then wait if needed next time...
	if( frame_duration > 1 ){
		int i=frame_duration-1;
		while(i--){
			vbl_wait();
		}
	}
#endif /* HAVE_VBL */
}


