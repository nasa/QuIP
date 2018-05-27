
#ifndef _VIEWER_H_
#define _VIEWER_H_

#ifdef HAVE_OPENGL
#ifdef HAVE_GL_GLX_H
#include <GL/glx.h>
#endif /* HAVE_GL_GLX_H */
#endif /* HAVE_OPENGL */

#ifdef __cplusplus
extern "C" {
#endif

#include "quip_config.h"

#include "data_obj.h"
#include "cmaps.h"
#include "display.h"
#include "dispobj.h"
#include "gen_win.h"
#include "genwin_flags.h"
//#include "function.h"	// Size_Functions

typedef enum {
	CE_INVALID_CODE,
	CE_TOUCH_DOWN,
	CE_TOUCH_UP,
	CE_TOUCH_MOVE,
	CE_LEFT_BUTTON_DOWN,
	CE_MIDDLE_BUTTON_DOWN,
	CE_RIGHT_BUTTON_DOWN,
	CE_LEFT_BUTTON_UP,
	CE_MIDDLE_BUTTON_UP,
	CE_RIGHT_BUTTON_UP,
	CE_MOUSE_MOVE,
	N_CANVAS_EVENTS
} Canvas_Event_Code;

#ifdef BUILD_FOR_OBJC

#include "ios_item.h"
#include "sizable.h"
#include "gen_win.h"
#include "quipCanvas.h"
#include "quipView.h"

// BUG?  this defn relies on order of codes above...
#define IS_TOUCH_EVENT(code)	((code)>=CE_TOUCH_DOWN && (code)<=CE_TOUCH_MOVE)

#ifdef BUILD_FOR_IOS
#define MAKE_NEEDY(vp)		{ \
/*sprintf(DEFAULT_ERROR_STRING,"MAKE_NEEDY:  canvas = 0x%lx",(long)VW_CANVAS(vp));\
advise(DEFAULT_ERROR_STRING);*/ \
[VW_CANVAS(vp) setNeedsDisplay];}

#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
#include <AppKit/NSOpenGLView.h>
#define MAKE_NEEDY(vp)
#endif // BUILD_FOR_MACOS

#define FLAG_NEEDY(vp)		SET_VW_FLAG_BITS(vp,VW_NEEDS_REFRESH)


@interface Canvas_Event : IOS_Item
@property Canvas_Event_Code		event_code;
+(void) initClass;
@end

#define SET_CE_CODE(cep,c)	cep.event_code = c

IOS_ITEM_INIT_PROT(Canvas_Event,canvas_event)
IOS_ITEM_NEW_PROT(Canvas_Event,canvas_event)
IOS_ITEM_CHECK_PROT(Canvas_Event,canvas_event)
//IOS_ITEM_GET_PROT(Canvas_Event,canvas_event)
IOS_ITEM_PICK_PROT(Canvas_Event,canvas_event)
IOS_ITEM_LIST_PROT(Canvas_Event,canvas_event)

@class quipAppDelegate;
@class quipView;
@class quipImageView;
    

@interface Viewer : IOS_Item

@property (retain) Gen_Win *		gwp;
@property CGContextRef			ctx_ref;
@property int32_t			event_mask;
@property List *			image_lp;
@property (retain) IOS_List *		draw_lp;
@property List *			drag_lp;
@property Data_Obj *			vw_dp;
@property time_t			vw_time;
@property int				frameno;
// width and height are inherited from Gen_Win
@property int				depth;
@property const char *			text;
// These 3 strings are the button actions;
// We will ultimately replace them with more general event handling
@property const char *			text1;
@property const char *			text2;
@property const char *			text3;

@property const char *			label;

// for plotting...
@property int				bgcolor;
@property int				fgcolor;
@property float				xmin;
@property float				xdel;
@property float				ymin;
@property float				ydel;
#ifdef HAVE_OPENGL
@property NSOpenGLView			*nsogl_vp;
#define VW_OGLV(vp)		(vp).nsogl_vp
#define SET_VW_OGLV(vp,v)	(vp).nsogl_vp = v
#define VW_OGL_CTX(vp)		(vp).nsogl_vp.openGLContext
#define SET_VW_OGL_CTX(vp,v)	(vp).nsogl_vp.openGLContext = v
#endif /* HAVE_OPENGL */

@property quipImageView	*	vw_qiv_p;

+(void) initClass;

@end

#define VW_QIV(vp)		(vp).vw_qiv_p
#define VW_NAME(vp)		(vp).name.UTF8String
#define VW_FLAGS(vp)		GW_FLAGS(VW_GW(vp))
#define VW_OBJ(vp)		(vp).vw_dp
#define VW_LABEL(vp)		(vp).label
#define VW_TEXT(vp)		(vp).text
#define VW_TEXT1(vp)		(vp).text1
#define VW_TEXT2(vp)		(vp).text2
#define VW_TEXT3(vp)		(vp).text3
#define VW_X(vp)		GW_X(VW_GW(vp))
#define VW_Y(vp)		GW_Y(VW_GW(vp))
#define VW_X_REQUESTED(vp)	GW_X(VW_GW(vp))
#define VW_Y_REQUESTED(vp)	GW_Y(VW_GW(vp))
#define VW_WIDTH(vp)		GW_WIDTH(VW_GW(vp))
#define VW_HEIGHT(vp)		GW_HEIGHT(VW_GW(vp))
#define VW_XMIN(vp)		(vp).xmin
#define VW_XDEL(vp)		(vp).xdel
#define VW_YMIN(vp)		(vp).ymin
#define VW_YDEL(vp)		(vp).ydel
#define VW_IMAGE_LIST(vp)	(vp).image_lp
#define VW_DRAG_LIST(vp)	(vp).drag_lp
#define VW_DRAW_LIST(vp)	(vp).draw_lp
#define VW_FRAMENO(vp)		(vp).frameno
#define VW_DEPTH(vp)		(vp).depth
#define VW_GW(vp)		(vp).gwp

//#ifdef BUILD_FOR_IOS
#define VW_QVC(vp)		((quipViewController *)GW_VC(VW_GW(vp)))

#ifdef BUILD_FOR_IOS
#define VW_QV(vp)		((quipView *)VW_QVC(vp).view)
#define VW_CANVAS(vp)		QV_CANVAS(VW_QV(vp))
#define VW_IMAGES(vp)		QV_IMAGES(VW_QV(vp))
#endif // BUILD_FOR_IOS

#define VW_BG_IMG(vp)		QV_BG_IMG(VW_QV(vp))

#define SET_VW_QVC(vp,v)	SET_GW_VC(VW_GW(vp),v)
#define SET_VW_CANVAS(vp,v)	SET_QV_CANVAS(VW_QV(vp),v)
#define SET_VW_BG_IMG(vp,iv)	SET_QV_BG_IMG(VW_QV(vp),iv)
//#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
//#define VW_WINDOW(vp)		GW_WINDOW(VW_GW(vp))
#define VW_IMAGES(vp)		QWC_IMAGES(VW_QWC(vp))
#define VW_QWC(vp)		(GW_WC(VW_GW(vp)))
#define VW_CANVAS(vp)		GW_CANVAS(VW_GW(vp))
#endif // BUILD_FOR_MACOS

#define VW_EVENT_TBL(vp)	GW_EVENT_TBL(VW_GW(vp))
#define VW_CMAP(vp)		(vp).cmap
#define VW_GFX_CTX(vp)		(vp).ctx_ref
#define VW_BGCOLOR(vp)		(vp).bgcolor
#define VW_FGCOLOR(vp)		(vp).fgcolor

#define SET_VW_GW(vp,v)		(vp).gwp = v
#define SET_VW_OBJ(vp,dp)	(vp).vw_dp = dp
#define SET_VW_DEPTH(vp,v)	(vp).depth = v
#define SET_VW_FRAMENO(vp,v)	(vp).frameno = v
#define SET_VW_LABEL(vp,s)	(vp).label = s
#define SET_VW_TEXT(vp,s)	(vp).text = s
#define SET_VW_TEXT1(vp,s)	(vp).text1 = s
#define SET_VW_TEXT2(vp,s)	(vp).text2 = s
#define SET_VW_TEXT3(vp,s)	(vp).text3 = s
#define SET_VW_TIME(vp,s)	(vp).vw_time = s
#define SET_VW_X(vp,v)		SET_GW_X(VW_GW(vp),v)
#define SET_VW_Y(vp,v)		SET_GW_Y(VW_GW(vp),v)
#define SET_VW_WIDTH(vp,v)	SET_GW_WIDTH(VW_GW(vp),v)
#define SET_VW_HEIGHT(vp,v)	SET_GW_HEIGHT(VW_GW(vp),v)
#define SET_VW_XMIN(vp,v)	(vp).xmin = v
#define SET_VW_XDEL(vp,v)	(vp).xdel = v
#define SET_VW_YMIN(vp,v)	(vp).ymin = v
#define SET_VW_YDEL(vp,v)	(vp).ydel = v
#define SET_VW_IMAGE_LIST(vp,v)	(vp).image_lp = v
#define SET_VW_DRAG_LIST(vp,v)	(vp).drag_lp = v
#define SET_VW_DRAW_LIST(vp,v)	(vp).draw_lp = v
#define SET_VW_FLAGS(vp,v)	SET_GW_FLAGS(VW_GW(vp),v)
#define SET_VW_EVENT_TBL(vp,v)	SET_GW_EVENT_TBL(VW_GW(vp),v)
#define SET_VW_CMAP(vp,v)	(vp).cmap = v
#define SET_VW_GFX_CTX(vp,v)	(vp).ctx_ref = v

#define SET_VW_BGCOLOR(vp,v)	(vp).bgcolor = v
#define SET_VW_FGCOLOR(vp,v)	(vp).fgcolor = v

#define INSURE_X11_SERVER

IOS_ITEM_INIT_PROT(Viewer,vwr)
IOS_ITEM_NEW_PROT(Viewer,vwr)
IOS_ITEM_CHECK_PROT(Viewer,vwr)
IOS_ITEM_GET_PROT(Viewer,vwr)
IOS_ITEM_PICK_PROT(Viewer,vwr)
IOS_ITEM_DEL_PROT(Viewer,vwr)
IOS_ITEM_LIST_PROT(Viewer,vwr)


extern void init_viewer_canvas(Viewer *vp);
extern void init_viewer_images(Viewer *vp);

#define INSIST_IMAGE_VIEWER(whence)						\
										\
	if( !is_image_viewer(QSP_ARG  vp) ){					\
		sprintf(ERROR_STRING,"%s:  %s is not an image viewer!?",	\
			#whence,VW_NAME(vp));					\
		WARN(ERROR_STRING);						\
		return;								\
	}



#else /* ! BUILD_FOR_OBJC */


#define add_ios_sizable add_sizable

#ifdef HAVE_X11
#define INSURE_X11_SERVER  insure_x11_server(SINGLE_QSP_ARG);
#else
#define INSURE_X11_SERVER
#endif /* ! HAVE_X11 */

/* viewer.h */

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#ifdef HAVE_TIME_H
#include <time.h>
#endif

typedef struct canvas_event {
	Item	ce_item;
#define ce_name	ce_item.item_name
	int	ce_code;
} Canvas_Event;

ITEM_INTERFACE_PROTOTYPES(Canvas_Event,canvas_event)

#define CE_CODE(cep)		cep->ce_code
#define SET_CE_CODE(cep,c)	cep->ce_code = c

/* used for draggable objects... */
#define WORD_TYPE	long
#define WORDLEN		((sizeof(WORD_TYPE))<<3)

// From X11 XSetLineAttributes...

typedef struct line_params {
	unsigned int	lp_width;
	int		lp_line_style;
	int		lp_cap_style;
	int		lp_join_style;
} Line_Params;

struct viewer {
	Gen_Win		vw_genwin;
	Dpyable		vw_dpyable;
#ifdef HAVE_X11
	XImage *	vw_ip;
	GC		vw_gc;
	XImage *	vw_ip2;			/* an extra for image reads */
#endif /* HAVE_X11 */
	long		vw_event_mask;
	List *		vw_image_list;
	int		vw_frameno;		/* which frame of movie is in */
	const char *	vw_text;
	const char *	vw_text1;		/* text1, text2, text3 button texts */
	const char *	vw_text2;		/* text1, text2, text3 button texts */
	const char *	vw_text3;
	List *		vw_drawlist;
	List *		vw_draglist;
	Data_Obj *	vw_dp;			/* a private copy */
/* we don't need this, but let's not make big changes until we don't have a deadline looming! */
	time_t		vw_time;
	float		vw_xmin,vw_xdel,vw_ymin,vw_ydel;	/* for plotting */
	const char *	vw_label;		/* text label on window bar */
	const char **	vw_event_tbl;		/* table of strings */

	Line_Params	vw_line_params;

#ifdef HAVE_OPENGL
	// We had an OpenGL context as part of the viewer, but it really goes with
	// the visual - we use the same visual for all windows, so we
	// keep the context as part of the display object
	/* Should this be ifdef'd? */
//	GLXContext	vw_ctx;
#endif /* HAVE_OPENGL */

} ;

#define VW_DOP(vp)	DPA_DOP(VW_DPA(vp))

#ifdef HAVE_OPENGL
#define VW_OGL_CTX(vp)		DO_OGL_CTX(VW_DOP(vp))
// Never set, this is owned by the display!!!
//#define SET_VW_OGL_CTX(vp,v)	SET_DO_OGL_CTX(VW_DOP(vp),v)
#endif /* HAVE_OPENGL */

#define vw_cmap_dp		vw_dpyable.dpa_cmap_dp
#define vw_lintbl_dp		vw_dpyable.dpa_lintbl_dp
#define vw_cmap			vw_dpyable.dpa_cmap
#define vw_xctbl		vw_dpyable.dpa_xctbl
#define vw_n_protected_colors	vw_dpyable.dpa_n_protected_colors

#define vw_name		vw_genwin.gw_name
#define vw_width	vw_dpyable.dpa_width
#define vw_height	vw_dpyable.dpa_height
#define vw_depth	vw_dpyable.dpa_depth
#define vw_x		vw_dpyable.dpa_x
#define vw_y		vw_dpyable.dpa_y
#define vw_x_requested	vw_dpyable.dpa_x_requested
#define vw_y_requested	vw_dpyable.dpa_y_requested
#define vw_y_offset	vw_dpyable.dpa_y_offset
#define vw_xwin		vw_dpyable.dpa_xwin
#define vw_flags	vw_dpyable.dpa_flags
#define vw_dop		vw_dpyable.dpa_dop
//#define vw_dpy		vw_dpyable.dpa_dpy
#define VW_DPA(vp)	(&((vp)->vw_dpyable))

#define VW_DPY(vp)	DPA_DPY(VW_DPA(vp))
#define VW_ROOTW(vp)	DPA_ROOTW(VW_DPA(vp))
#define VW_VISUAL(vp)	DPA_VISUAL(VW_DPA(vp))
#define VW_SCREEN_NO(vp)	DPA_SCREEN_NO(VW_DPA(vp))
//#define VW_CTX(vp)

//#define vw_screen_no	vw_dpyable.dpa_screen_no
//#define vw_visual	vw_dpyable.dpa_visual

// X11 stuff
#define VW_CMAP_OBJ(vp)		DPA_CMAP_OBJ( VW_DPYABLE(vp) )
#define VW_LINTBL_OBJ(vp)	DPA_LINTBL_OBJ( VW_DPYABLE(vp) )
#define VW_XCTBL(vp)		DPA_XCTBL( VW_DPYABLE(vp) )
#define SET_VW_XCTBL(vp,v)	SET_DPA_XCTBL( VW_DPYABLE(vp) , v )
#define SET_VW_XCTBL_PIXEL(vp,i,v)	SET_DPA_XCTBL_PIXEL(VW_DPYABLE(vp),i,v)
#define SET_VW_XCTBL_RED(vp,i,v)	SET_DPA_XCTBL_RED(VW_DPYABLE(vp),i,v)
#define SET_VW_XCTBL_GREEN(vp,i,v)	SET_DPA_XCTBL_GREEN(VW_DPYABLE(vp),i,v)
#define SET_VW_XCTBL_BLUE(vp,i,v)	SET_DPA_XCTBL_BLUE(VW_DPYABLE(vp),i,v)
#define SET_VW_XCTBL_FLAGS(vp,i,v)	SET_DPA_XCTBL_FLAGS(VW_DPYABLE(vp),i,v)
#define VW_DPYABLE(cp)		(&((vp)->vw_dpyable))
//#define VW_DPY(vp)		(vp)->vw_dpy
#define VW_GC(vp)		(vp)->vw_gc
#define VW_XWIN(vp)		(vp)->vw_xwin

#define VW_GW(vp)		(&((vp)->vw_genwin))
#define VW_OBJ(vp)		(vp)->vw_dp
#define VW_NAME(vp)		(vp)->vw_name
#define VW_FLAGS(vp)		(vp)->vw_flags
#define VW_IMAGE_LIST(vp)	(vp)->vw_image_list
#define VW_DRAG_LIST(vp)	(vp)->vw_draglist
#define VW_DRAW_LIST(vp)	(vp)->vw_drawlist
#define VW_TEXT(vp)		(vp)->vw_text
#define VW_TEXT1(vp)		(vp)->vw_text1
#define VW_TEXT2(vp)		(vp)->vw_text2
#define VW_TEXT3(vp)		(vp)->vw_text3
#define VW_WIDTH(vp)		(vp)->vw_width
#define VW_HEIGHT(vp)		(vp)->vw_height
#define VW_DEPTH(vp)		(vp)->vw_depth
#define VW_XMIN(vp)		(vp)->vw_xmin
#define VW_YMIN(vp)		(vp)->vw_ymin
#define VW_XDEL(vp)		(vp)->vw_xdel
#define VW_YDEL(vp)		(vp)->vw_ydel
#define VW_LABEL(vp)		(vp)->vw_label
#define VW_X(vp)		(vp)->vw_x
#define VW_Y(vp)		(vp)->vw_y
#define VW_X_REQUESTED(vp)	(vp)->vw_x_requested
#define VW_Y_REQUESTED(vp)	(vp)->vw_y_requested
#define VW_Y_OFFSET(vp)		(vp)->vw_y_offset
#define VW_EVENT_TBL(vp)	(vp)->vw_event_tbl
#define VW_EVENT_ACTION(vp,idx)	(vp)->vw_event_tbl[idx]

#define VW_LINE_WIDTH(vp)	(vp)->vw_line_params.lp_width
#define VW_LINE_STYLE(vp)	(vp)->vw_line_params.lp_line_style
#define VW_CAP_STYLE(vp)	(vp)->vw_line_params.lp_cap_style
#define VW_JOIN_STYLE(vp)	(vp)->vw_line_params.lp_join_style
#define SET_VW_LINE_WIDTH(vp,v)	(vp)->vw_line_params.lp_width = v
#define SET_VW_LINE_STYLE(vp,v)	(vp)->vw_line_params.lp_line_style = v
#define SET_VW_CAP_STYLE(vp,v)	(vp)->vw_line_params.lp_cap_style = v
#define SET_VW_JOIN_STYLE(vp,v)	(vp)->vw_line_params.lp_join_style = v

#define SET_VW_OBJ(vp,v)	(vp)->vw_dp = v
#define SET_VW_FRAMENO(vp,v)	(vp)->vw_frameno = v
#define SET_VW_FLAGS(vp,v)	(vp)->vw_flags = v
#define SET_VW_TEXT(vp,v)	(vp)->vw_text = v
#define SET_VW_TEXT1(vp,v)	(vp)->vw_text1 = v
#define SET_VW_TEXT2(vp,v)	(vp)->vw_text2 = v
#define SET_VW_TEXT3(vp,v)	(vp)->vw_text3 = v
#define SET_VW_XMIN(vp,v)	(vp)->vw_xmin = v
#define SET_VW_YMIN(vp,v)	(vp)->vw_ymin = v
#define SET_VW_XDEL(vp,v)	(vp)->vw_xdel = v
#define SET_VW_YDEL(vp,v)	(vp)->vw_ydel = v
#define SET_VW_LABEL(vp,v)	(vp)->vw_label = v
#define SET_VW_X(vp,v)		(vp)->vw_x = v
#define SET_VW_Y(vp,v)		(vp)->vw_y = v
#define SET_VW_X_REQUESTED(vp,v)		(vp)->vw_x_requested = v
#define SET_VW_Y_REQUESTED(vp,v)		(vp)->vw_y_requested = v
#define SET_VW_Y_OFFSET(vp,v)	(vp)->vw_y_offset = v
#define SET_VW_WIDTH(vp,v)	(vp)->vw_width = v
#define SET_VW_HEIGHT(vp,v)	(vp)->vw_height = v
#define SET_VW_DEPTH(vp,v)	(vp)->vw_depth = v
#define SET_VW_TIME(vp,v)	(vp)->vw_time = v
#define SET_VW_IMAGE_LIST(vp,v)	(vp)->vw_image_list = v
#define SET_VW_DRAG_LIST(vp,v)	(vp)->vw_draglist = v
#define SET_VW_DRAW_LIST(vp,v)	(vp)->vw_drawlist = v
#define SET_VW_EVENT_TBL(vp,t)	(vp)->vw_event_tbl = t
#define SET_VW_EVENT_ACTION(vp,idx,a)	(vp)->vw_event_tbl[idx] = a

/* prototypes */

ITEM_INTERFACE_PROTOTYPES(Viewer,vwr);

extern Disp_Obj *the_dop;

#include "map_ios_item.h"


#endif /* ! BUILD_FOR_OBJC */

#define init_canvas_events()	_init_canvas_events(SINGLE_QSP_ARG)
#define pick_canvas_event(p)	_pick_canvas_event(QSP_ARG  p)
#define new_canvas_event(s)	_new_canvas_event(QSP_ARG  s)
#define canvas_event_of(s)	_canvas_event_of(QSP_ARG  s)

#define SET_VW_FLAG_BITS(vp,v)		SET_VW_FLAGS(vp, VW_FLAGS(vp) | v )
#define CLEAR_VW_FLAG_BITS(vp,v)	SET_VW_FLAGS(vp, VW_FLAGS(vp) & ~(v) )

/* Viewer flag macros */

#define VWR_IS_MAPPED( vp )	( VW_FLAGS(vp) & VIEW_MAPPED)

#define IS_PLOTTER( vp )	( VW_FLAGS(vp) & VIEW_PLOTTER )
#define IS_ADJUSTER( vp )	( VW_FLAGS(vp) & VIEW_ADJUSTER )
#define IS_DRAGSCAPE( vp )	( VW_FLAGS(vp) & VIEW_DRAGSCAPE )
#define IS_MOUSESCAPE( vp )	( VW_FLAGS(vp) & VIEW_MOUSESCAPE )
#define IS_TRACKING( vp )	( VW_FLAGS(vp) & VIEW_TRACK )
#define IS_BUTTON_ARENA( vp )	( VW_FLAGS(vp) & VIEW_BUTTON_ARENA )
#define IS_GL_WINDOW( vp )	( VW_FLAGS(vp) & VIEW_GL )

#define SIMULATING_LUTS(vp)	( VW_FLAGS(vp) & VIEW_LUT_SIM )
#define OWNS_IMAGE_DATA( vp )	( VW_FLAGS(vp) & VIEWER_OWNS_IMAGE_DATA )

#define READY_FOR_GLX( vp )	( VW_FLAGS(vp) & VIEW_GLX_RDY )

#define VW_TEXT_LJUST(vp)	((VW_FLAGS(vp) & VW_JUSTIFY_MASK)==VW_LEFT_JUSTIFY)
#define VW_TEXT_RJUST(vp)	((VW_FLAGS(vp) & VW_JUSTIFY_MASK)==VW_RIGHT_JUSTIFY)
#define VW_TEXT_CENTER(vp)	((VW_FLAGS(vp) & VW_JUSTIFY_MASK)==VW_CENTER_TEXT)

#define VW_TXT_MTRX_READY(vp)	(VW_FLAGS(vp) & VW_TXT_MTRX_INITED)

#define VW_MOVE_REQUESTED(vp)	(VW_FLAGS(vp) & VW_PROG_MOVE_REQ)

typedef struct window_image {
	Data_Obj *	wi_dp;
	int		wi_x, wi_y;
} Window_Image;



typedef struct draggable {
		Item		dg_item;
#define dg_name		dg_item.item_name

		int		dg_width;
		int		dg_height;
		int		dg_x;		/* position of corner */
		int		dg_y;
		int		dg_rx;		/* position of where grasped */
		int		dg_ry;
		Data_Obj *	dg_bitmap;
		Data_Obj *	dg_image;
		Node *		dg_np;
} Draggable;

typedef struct view_cursor {
		char *		vc_name;
#ifdef HAVE_X11
		Cursor		vc_cursor;	// is this really an Xlib struct?
#endif /* HAVE_X11 */
		unsigned int	vc_xhot;
		unsigned int	vc_yhot;
} View_Cursor;



/* viewer.c */

extern void zap_image_list(Viewer *vp);
extern void init_genwin_viewer(void);

extern void _init_viewer_genwin(SINGLE_QSP_ARG_DECL);
extern void _select_viewer(QSP_ARG_DECL  Viewer *vp);
extern void _release_image(QSP_ARG_DECL  Data_Obj *dp);
extern void _delete_viewer(QSP_ARG_DECL  Viewer *vp);
extern Viewer *_viewer_init(QSP_ARG_DECL  const char *name,int dx,int dy,int flags);
extern IOS_Node *_first_viewer_node(SINGLE_QSP_ARG_DECL);
extern IOS_List *_viewer_list(SINGLE_QSP_ARG_DECL);
extern void _info_viewer(QSP_ARG_DECL  Viewer *vp);

#define init_viewer_genwin() _init_viewer_genwin(SINGLE_QSP_ARG)
#define select_viewer(vp) _select_viewer(QSP_ARG  vp)
#define release_image(dp) _release_image(QSP_ARG  dp)
#define delete_viewer(vp) _delete_viewer(QSP_ARG  vp)
#define viewer_init(name,dx,dy,flags) _viewer_init(QSP_ARG  name,dx,dy,flags)
#define first_viewer_node() _first_viewer_node(SINGLE_QSP_ARG)
#define viewer_list() _viewer_list(SINGLE_QSP_ARG)
#define info_viewer(vp) _info_viewer(QSP_ARG  vp)

/* canvas.c */

extern int add_image(Viewer *vp,Data_Obj *dp,int x,int y);
extern void insert_image(Data_Obj *dpto,Data_Obj *dpfr,int x,int y,int frameno);
extern void update_image(Viewer *vp);
extern void load_viewer(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp);
extern void _old_load_viewer(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp);
#define old_load_viewer(vp,dp) _old_load_viewer(QSP_ARG  vp,dp)

/* xplot.c */

extern void scale_fxy(Viewer *,float *px,float *py);
extern void scalexy(Viewer *,int *px,int *py);
extern void _xplot_fpoint(QSP_ARG_DECL  float x,float y);
extern void _xplot_fcont(QSP_ARG_DECL  float x,float y);
extern void _tell_plot_space(SINGLE_QSP_ARG_DECL);
extern void _xplot_ffill_arc(QSP_ARG_DECL  float, float, float, float, float, float);
extern void _xplot_fill_polygon(QSP_ARG_DECL  int num_points, float* xarray, float* yarray);
extern void _xplot_space(QSP_ARG_DECL  int x1,int y1,int x2,int y2);
extern void _xplot_fspace(QSP_ARG_DECL  float x1,float y1,float x2,float y2);
extern void _xplot_fmove(QSP_ARG_DECL  float x,float y);
extern void _xplot_move(QSP_ARG_DECL  int x,int y);
extern void _xplot_point(QSP_ARG_DECL  int x,int y);
extern void _xplot_cont(QSP_ARG_DECL  int x,int y);
extern void _xplot_line(QSP_ARG_DECL  int x1,int y1,int x2,int y2);
extern void _xplot_text(QSP_ARG_DECL  const char *);
extern void _xplot_fline(QSP_ARG_DECL  float x1,float y1,float x2,float y2);
extern void _xplot_setup(QSP_ARG_DECL  Viewer *vp);
extern void _xplot_select(QSP_ARG_DECL  u_long color);
extern void _xplot_bgselect(QSP_ARG_DECL  u_long color);
extern void _xplot_arc(QSP_ARG_DECL  int,int,int,int,int,int);
extern void _xplot_fill_arc(QSP_ARG_DECL  int,int,int,int,int,int);
extern void _xplot_farc(QSP_ARG_DECL  float,float, float, float, float, float);
extern void _xplot_circle(QSP_ARG_DECL  float radius);

#define xplot_fcont(x,y) _xplot_fcont(QSP_ARG  x,y)
#define xplot_fpoint(x,y) _xplot_fpoint(QSP_ARG  x,y)
#define tell_plot_space() _tell_plot_space(SINGLE_QSP_ARG)
#define xplot_ffill_arc(a,b,c,d,e,f) _xplot_ffill_arc(QSP_ARG  a,b,c,d,e,f)
#define xplot_fill_polygon(num_points,xarray,yarray) _xplot_fill_polygon(QSP_ARG  num_points,xarray,yarray)
#define xplot_space(x1,y1,x2,y2) _xplot_space(QSP_ARG  x1,y1,x2,y2)
#define xplot_fspace(x1,y1,x2,y2) _xplot_fspace(QSP_ARG  x1,y1,x2,y2)
#define xplot_fmove(x,y) _xplot_fmove(QSP_ARG  x,y)
#define xplot_move(x,y) _xplot_move(QSP_ARG  x,y)
#define xplot_point(x,y) _xplot_point(QSP_ARG  x,y)
#define xplot_cont(x,y) _xplot_cont(QSP_ARG  x,y)
#define xplot_line(x1,y1,x2,y2) _xplot_line(QSP_ARG  x1,y1,x2,y2)
#define xplot_text(s) _xplot_text(QSP_ARG  s)
#define xplot_fline(x1,y1,x2,y2) _xplot_fline(QSP_ARG  x1,y1,x2,y2)
#define xplot_setup(vp) _xplot_setup(QSP_ARG  vp)
#define xplot_select(color) _xplot_select(QSP_ARG  color)
#define xplot_bgselect(color) _xplot_bgselect(QSP_ARG  color)
#define xplot_arc(a,b,c,d,e,f) _xplot_arc(QSP_ARG  a,b,c,d,e,f)
#define xplot_fill_arc(a,b,c,d,e,f) _xplot_fill_arc(QSP_ARG  a,b,c,d,e,f)
#define xplot_farc(a,b,c,d,e,f) _xplot_farc(QSP_ARG  a,b,c,d,e,f)
#define xplot_circle(radius) _xplot_circle(QSP_ARG  radius)

extern void _xplot_erase(SINGLE_QSP_ARG_DECL);
#define xplot_erase() _xplot_erase(SINGLE_QSP_ARG)

#ifdef BUILD_FOR_IOS
extern void _xplot_update(SINGLE_QSP_ARG_DECL);
#define xplot_update()	_xplot_update(SINGLE_QSP_ARG)
extern quipImageView *image_view_for_viewer(Viewer *vp);
#endif /* BUILD_FOR_IOS */

extern void _dump_drawlist(QSP_ARG_DECL  Viewer *vp);
#define dump_drawlist(vp) _dump_drawlist(QSP_ARG  vp)

/* rdplot.c */

extern void getpair(FILE *fp,int *px,int *py);
extern void getone(FILE *fp,int *p);
extern void rdplot(QSP_ARG_DECL  FILE *fp);

/* drag.c */

ITEM_INTERFACE_PROTOTYPES(Draggable,dragg)
#define new_dragg(s)		_new_dragg(QSP_ARG  s)
#define pick_dragg(pmpt)	_pick_dragg(QSP_ARG  pmpt)
#define list_draggs(fp)		_list_draggs(QSP_ARG  fp)

extern void make_dragg(QSP_ARG_DECL  const char *name,Data_Obj *bm,Data_Obj *dp);
extern Draggable *in_draggable(Viewer *vp,int x,int y);
extern void extract_image(Data_Obj *dpto,Data_Obj *dpfr,int x,int y);

/* cursors.c */

ITEM_INTERFACE_PROTOTYPES(View_Cursor,cursor)

#define pick_cursor(pmpt)	_pick_cursor(QSP_ARG  pmpt)
#define new_cursor(s)		_new_cursor(QSP_ARG  s)
#define list_cursors(fp)	_list_cursors(QSP_ARG  fp)

extern void default_cursors(SINGLE_QSP_ARG_DECL);
extern void make_cursor(QSP_ARG_DECL  const char *name,Data_Obj *bitmap_dp,int x,int y);
extern void mk_cursor(QSP_ARG_DECL  const char *name,u_short *data,dimension_t dx,dimension_t dy,dimension_t x,dimension_t y);
extern void root_cursor(View_Cursor *vcp);
extern void assign_cursor(Viewer *vp,View_Cursor *vcp);

/* from the implentation file (e.g. xsupp) */
/* These functions define the api that we present from the windowing system */
extern void posn_viewer(Viewer *vp,int x,int y);
extern void relabel_viewer(Viewer *vp,const char *s);
extern void zap_viewer(Viewer *vp);
extern void _xp_select(QSP_ARG_DECL  Viewer *vp,u_long color);
extern void _xp_bgselect(QSP_ARG_DECL  Viewer *vp,u_long color);
extern void _xp_text(QSP_ARG_DECL  Viewer *vp,int x1,int y1,const char *);
extern void _xp_line(QSP_ARG_DECL  Viewer *vp,int x1,int y1,int x2,int y2);

#define xp_select(vp,color) _xp_select(QSP_ARG  vp,color)
#define xp_bgselect(vp,color) _xp_bgselect(QSP_ARG  vp,color)
#define xp_text(vp,x1,y1,s) _xp_text(QSP_ARG  vp,x1,y1,s)
#define xp_line(vp,x1,y1,x2,y2) _xp_line(QSP_ARG  vp,x1,y1,x2,y2)

extern void _show_viewer(QSP_ARG_DECL  Viewer *vp);
extern void _unshow_viewer(QSP_ARG_DECL  Viewer *vp);
extern void _redraw_viewer(QSP_ARG_DECL  Viewer *vp);
extern void _embed_image(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp,int x,int y);
extern void _unembed_image(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp,int x,int y);
extern int _make_2d_adjuster(QSP_ARG_DECL  Viewer *vp,int,int);
extern int _make_dragscape(QSP_ARG_DECL  Viewer *vp,int,int);
extern int _make_mousescape(QSP_ARG_DECL  Viewer *vp,int,int);
extern int _make_button_arena(QSP_ARG_DECL  Viewer *vp,int,int);
extern int _make_viewer(QSP_ARG_DECL  Viewer *vp,int,int);
/* should this be HAVE_GL only??? */
extern int _make_gl_window(QSP_ARG_DECL  Viewer *vp, int width, int height);
extern int _display_depth(SINGLE_QSP_ARG_DECL);
extern void _extra_viewer_info(QSP_ARG_DECL  Viewer *vp);
#ifdef BUILD_FOR_OBJC
extern void _xp_linewidth(Viewer *vp,CGFloat w);
extern int is_image_viewer(QSP_ARG_DECL  Viewer *vp);
#else // ! BUILD_FOR_OBJC
extern void _xp_linewidth(Viewer *vp,int w);
#endif // ! BUILD_FOR_OBJC
extern void _xp_cont(Viewer *vp,int x,int y);
extern void _xp_arc(Viewer *,int,int,int,int,int,int);
extern void _xp_update(Viewer *vp);
extern void set_remember_gfx(int flag);
extern void _xp_fill_arc(Viewer*, int, int, int, int, int, int);
extern void _xp_fill_polygon(Viewer* vp, int num_points, int* px_vals, int* py_vals);
extern void _xp_move(Viewer *vp,int x1,int y1);


extern void _set_char_spacing(QSP_ARG_DECL  Viewer *vp,int sz);
extern int _exec_drawlist(QSP_ARG_DECL  Viewer *vp);
extern void _set_font_by_name(QSP_ARG_DECL  Viewer *vp,const char *s);
extern void _xp_erase(QSP_ARG_DECL  Viewer *vp);
extern void _set_text_angle(QSP_ARG_DECL  Viewer *vp,float a);
extern void _set_font_size(QSP_ARG_DECL  Viewer *vp,int sz);

#define set_char_spacing(vp,sz) _set_char_spacing(QSP_ARG  vp,sz)
#define exec_drawlist(vp) _exec_drawlist(QSP_ARG  vp)
#define set_font_by_name(vp,s) _set_font_by_name(QSP_ARG  vp,s)
#define xp_erase(vp) _xp_erase(QSP_ARG  vp)
#define set_text_angle(vp,a) _set_text_angle(QSP_ARG  vp,a)
#define set_font_size(vp,sz) _set_font_size(QSP_ARG  vp,sz)

int event_loop(SINGLE_QSP_ARG_DECL);
extern void embed_draggable(Data_Obj *dp,Draggable *dgp);
extern void window_sys_init(SINGLE_QSP_ARG_DECL);

extern void _init_reserved_vars(SINGLE_QSP_ARG_DECL);
#define init_reserved_vars() _init_reserved_vars(SINGLE_QSP_ARG)

extern void _center_text(QSP_ARG_DECL  Viewer *vp);
extern void _left_justify(QSP_ARG_DECL  Viewer *vp);
extern void _right_justify(QSP_ARG_DECL  Viewer *vp);
#define center_text(vp) _center_text(QSP_ARG  vp)
#define left_justify(vp) _left_justify(QSP_ARG  vp)
#define right_justify(vp) _right_justify(QSP_ARG  vp)

extern void	set_viewer_display(QSP_ARG_DECL  Viewer *vp);
extern void	cmap_setup(Viewer *);
extern void set_action_for_event(Viewer *vp,Canvas_Event *cep,const char *s);

extern void declare_canvas_events(SINGLE_QSP_ARG_DECL);
extern int	_get_string_width(QSP_ARG_DECL  Viewer *vp, const char *s);

//#endif /* HAVE_X11 */

#define show_viewer(vp)				_show_viewer(QSP_ARG  vp)
#define unshow_viewer(vp)			_unshow_viewer(QSP_ARG  vp)
#define redraw_viewer(vp)			_redraw_viewer(QSP_ARG  vp)
#define embed_image(vp,dp,x,y)			_embed_image(QSP_ARG  vp,dp,x,y)
#define unembed_image(vp,dp,x,y)		_unembed_image(QSP_ARG  vp,dp,x,y)
#define make_2d_adjuster(vp,w,h)		_make_2d_adjuster(QSP_ARG  vp,w,h)
#define make_dragscape(vp,w,h)			_make_dragscape(QSP_ARG  vp,w,h)
#define make_mousescape(vp,w,h)			_make_mousescape(QSP_ARG  vp,w,h)
#define make_button_arena(vp,w,h)		_make_button_arena(QSP_ARG  vp,w,h)
#define make_viewer(vp,w,h)			_make_viewer(QSP_ARG  vp,w,h)
#define make_gl_window(vp, width, height)	_make_gl_window(QSP_ARG  vp, width, height)
#define display_depth()				_display_depth(SINGLE_QSP_ARG)
#define extra_viewer_info(vp)			_extra_viewer_info(QSP_ARG  vp)
#define get_string_width(vp,s)			_get_string_width(QSP_ARG  vp,s)

// shm_viewer.c
extern Viewer *_init_shm_viewer(QSP_ARG_DECL  const char *name, int width, int height, int depth);
#define init_shm_viewer(name,width,height,depth) _init_shm_viewer(QSP_ARG  name,width,height,depth)
extern void display_to_shm_viewer(Viewer *vp,Data_Obj *dp);

#ifdef __cplusplus
}
#endif


#endif /* ! _VIEWER_H_ */

