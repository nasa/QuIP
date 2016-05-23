
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

#define NO_X_IMAGE	((XImage *)NULL)

// From X11 XSetLineAttributes...

typedef struct line_params {
	unsigned int	lp_width;
	int		lp_line_style;
	int		lp_cap_style;
	int		lp_join_style;
} Line_Params;

typedef struct viewer {
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

	/* Should this be ifdef'd? */
//	GLXContext	vw_ctx;
} Viewer;

#define VW_DOP(vp)	DPA_DOP(VW_DPA(vp))

#ifdef HAVE_OPENGL
#define VW_OGL_CTX(vp)		DO_OGL_CTX(VW_DOP(vp))
#define SET_VW_OGL_CTX(vp,v)	SET_DO_OGL_CTX(VW_DOP(vp),v)
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

#define NO_CANVAS_EVENT	((Canvas_Event *)NULL)
#define NO_VIEWER	((Viewer *) NULL)

#define PICK_CANVAS_EVENT(p)	pick_canvas_event(QSP_ARG  p)

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

#define NO_DRAGG	((Draggable *)NULL)



typedef struct view_cursor {
		char *		vc_name;
#ifdef HAVE_X11
		Cursor		vc_cursor;	// is this really an Xlib struct?
#endif /* HAVE_X11 */
		unsigned int	vc_xhot;
		unsigned int	vc_yhot;
} View_Cursor;

#define NO_CURSOR	((View_Cursor *)NULL)

#define PICK_CURSOR(pmpt)	pick_cursor(QSP_ARG  pmpt)


/* viewer.c */

extern void init_viewer_genwin(SINGLE_QSP_ARG_DECL);

#define VWR_OF( s )	vwr_of(QSP_ARG  s )
#define GET_VWR( s )	get_vwr(QSP_ARG  s )
#define PICK_VWR( s )	pick_vwr(QSP_ARG  s )

extern void zap_image_list(Viewer *vp);
extern void select_viewer(QSP_ARG_DECL  Viewer *vp);
extern void release_image(QSP_ARG_DECL  Data_Obj *dp);
extern void delete_viewer(QSP_ARG_DECL  Viewer *vp);
extern Viewer *viewer_init(QSP_ARG_DECL  const char *name,int dx,int dy,int flags);
extern IOS_Node *first_viewer_node(SINGLE_QSP_ARG_DECL);
extern IOS_List *viewer_list(SINGLE_QSP_ARG_DECL);
extern void info_viewer(QSP_ARG_DECL  Viewer *vp);
//extern double viewer_exists(QSP_ARG_DECL  const char *);
extern void init_genwin_viewer(void);


/* canvas.c */

extern int add_image(Viewer *vp,Data_Obj *dp,int x,int y);
extern void insert_image(Data_Obj *dpto,Data_Obj *dpfr,int x,int y,int frameno);
extern void update_image(Viewer *vp);
extern void load_viewer(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp);
extern void old_load_viewer(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp);

/* xplot.c */

extern void tell_plot_space(SINGLE_QSP_ARG_DECL);
extern void xp_ffill_arc(float, float, float, float, float, float);
extern void xp_fill_polygon(int num_points, float* xarray, float* yarray);
extern void xp_space(int x1,int y1,int x2,int y2);
extern void xp_fspace(float x1,float y1,float x2,float y2);
extern void scale_fxy(Viewer *,float *px,float *py);
extern void scalexy(Viewer *,int *px,int *py);
extern void xp_fmove(float x,float y);
extern void xp_move(int x,int y);
extern void xp_fcont(float x,float y);
extern void xp_point(int x,int y);
extern void xp_fpoint(float x,float y);
extern void xp_cont(int x,int y);
extern void xp_line(int x1,int y1,int x2,int y2);
extern void xp_text(const char *);
extern void xp_fline(float x1,float y1,float x2,float y2);
extern void xp_setup(QSP_ARG_DECL  Viewer *vp);

extern void xp_erase(void );

#ifdef BUILD_FOR_IOS
extern void xp_update(void);
extern quipImageView *image_view_for_viewer(Viewer *vp);
#endif /* BUILD_FOR_IOS */

extern void dump_drawlist(QSP_ARG_DECL  Viewer *vp);
extern void xp_select(u_long color);
extern void xp_bgselect(u_long color);
extern void xp_arc(int,int,int,int,int,int);
extern void xp_fill_arc(int,int,int,int,int,int);
extern void xp_farc(float,float,
	float, float, float, float);
extern void xp_circle(float radius);

/* rdplot.c */

extern void getpair(FILE *fp,int *px,int *py);
extern void getone(FILE *fp,int *p);
extern void rdplot(QSP_ARG_DECL  FILE *fp);

/* drag.c */

ITEM_INTERFACE_PROTOTYPES(Draggable,dragg)
#define PICK_DRAGG(pmpt)	pick_dragg(QSP_ARG  pmpt)

extern void make_dragg(QSP_ARG_DECL  const char *name,Data_Obj *bm,Data_Obj *dp);
extern Draggable *in_draggable(Viewer *vp,int x,int y);
extern void extract_image(Data_Obj *dpto,Data_Obj *dpfr,int x,int y);

/* cursors.c */

ITEM_INTERFACE_PROTOTYPES(View_Cursor,cursor)
extern void default_cursors(SINGLE_QSP_ARG_DECL);
extern void make_cursor(QSP_ARG_DECL  const char *name,Data_Obj *bitmap_dp,int x,int y);
extern void mk_cursor(QSP_ARG_DECL  const char *name,u_short *data,dimension_t dx,dimension_t dy,dimension_t x,dimension_t y);
extern void root_cursor(View_Cursor *vcp);
extern void assign_cursor(Viewer *vp,View_Cursor *vcp);

/* from the implentation file (e.g. xsupp) */
/* These functions define the api that we present from the windowing system */
extern void show_viewer(QSP_ARG_DECL  Viewer *vp);
extern void unshow_viewer(QSP_ARG_DECL  Viewer *vp);
extern void posn_viewer(Viewer *vp,int x,int y);
extern void relabel_viewer(Viewer *vp,const char *s);
extern void redraw_viewer(QSP_ARG_DECL  Viewer *vp);
extern void embed_image(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp,int x,int y);
extern void unembed_image(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp,int x,int y);
extern void zap_viewer(Viewer *vp);
extern int make_2d_adjuster(QSP_ARG_DECL  Viewer *vp,int,int);
extern int make_dragscape(QSP_ARG_DECL  Viewer *vp,int,int);
extern int make_mousescape(QSP_ARG_DECL  Viewer *vp,int,int);
extern int make_button_arena(QSP_ARG_DECL  Viewer *vp,int,int);
extern int make_viewer(QSP_ARG_DECL  Viewer *vp,int,int);
/* should this be HAVE_GL only??? */
extern int make_gl_window(QSP_ARG_DECL  Viewer *vp, int width, int height);
extern int		display_depth(SINGLE_QSP_ARG_DECL);
extern void extra_viewer_info(QSP_ARG_DECL  Viewer *vp);
extern void _xp_select(Viewer *vp,u_long color);
extern void _xp_bgselect(Viewer *vp,u_long color);
extern void _xp_text(Viewer *vp,int x1,int y1,const char *);
extern void _xp_line(Viewer *vp,int x1,int y1,int x2,int y2);
#ifdef BUILD_FOR_OBJC
extern void _xp_linewidth(Viewer *vp,CGFloat w);
extern int is_image_viewer(QSP_ARG_DECL  Viewer *vp);
#else // ! BUILD_FOR_OBJC
extern void _xp_linewidth(Viewer *vp,int w);
#endif // ! BUILD_FOR_OBJC
extern void _xp_cont(Viewer *vp,int x,int y);
extern void _xp_arc(Viewer *,int,int,int,int,int,int);
extern void _xp_erase(Viewer *vp);
extern void _xp_update(Viewer *vp);
extern void set_remember_gfx(int flag);
extern void _xp_fill_arc(Viewer*, int, int, int, int, int, int);
extern void _xp_fill_polygon(Viewer* vp, int num_points, int* px_vals, int* py_vals);
extern void _xp_move(Viewer *vp,int x1,int y1);
extern int exec_drawlist(Viewer *vp);
extern void set_font_size(Viewer *vp,int sz);
extern void set_char_spacing(Viewer *vp,int sz);
extern void set_font_by_name(Viewer *vp,const char *s);
extern void set_text_angle(Viewer *vp,float a);

extern void center_text(Viewer *vp);
extern void left_justify(Viewer *vp);
extern void right_justify(Viewer *vp);
extern int	get_string_width(Viewer *vp, const char *s);
int event_loop(SINGLE_QSP_ARG_DECL);
extern void embed_draggable(Data_Obj *dp,Draggable *dgp);
extern void window_sys_init(SINGLE_QSP_ARG_DECL);
extern void	set_viewer_display(Viewer *vp);
extern void	cmap_setup(Viewer *);
extern void set_action_for_event(Viewer *vp,Canvas_Event *cep,const char *s);

extern void declare_canvas_events(SINGLE_QSP_ARG_DECL);

//#endif /* HAVE_X11 */


#ifdef __cplusplus
}
#endif


#endif /* ! _VIEWER_H_ */

