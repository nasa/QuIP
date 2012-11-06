
#ifndef NO_VIEWER

#ifdef HAVE_OPENGL
#ifdef HAVE_GL_GLX_H
#include <GL/glx.h>
#endif /* HAVE_GL_GLX_H */
#endif /* HAVE_OPENGL */

#ifdef __cplusplus
extern "C" {
#endif

#include "quip_config.h"

#ifdef HAVE_X11

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

#include "node.h"
#include "data_obj.h"
#include "cmaps.h"
#include "display.h"
#include "dispobj.h"

/* used for draggable objects... */
#define WORD_TYPE	long
#define WORDLEN		((sizeof(WORD_TYPE))<<3)

typedef struct window_image {
	Data_Obj *	wi_dp;
	int		wi_x, wi_y;
} Window_Image;

#define NO_X_IMAGE	((XImage *)NULL)

typedef struct viewer {
	Dpyable		vw_top;
	XImage *	vw_ip;
	GC		vw_gc;
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
	XImage *	vw_ip2;			/* an extra for image reads */
/* we don't need this, but let's not make big changes until we don't have a deadline looming! */
//#ifdef FOOBAR
	time_t		vw_time;
//#endif /* FOOBAR */
	float		vw_xmin,vw_xdel,vw_ymin,vw_ydel;	/* for plotting */
	const char *	vw_label;		/* text label on window bar */

	/* Should this be ifdef'd? */
#ifdef HAVE_OPENGL
	GLXContext	vw_ctx;
#endif /* HAVE_OPENGL */
} Viewer;

#define vw_cm_dp		vw_top.c_cm_dp
#define vw_lt_dp		vw_top.c_lt_dp
#define vw_cmap			vw_top.c_cmap
#define vw_xctbl		vw_top.c_xctbl
#define vw_n_protected_colors	vw_top.c_n_protected_colors

#define vw_name		vw_top.c_name
#define vw_width	vw_top.c_width
#define vw_height	vw_top.c_height
#define vw_depth	vw_top.c_depth
#define vw_x		vw_top.c_x
#define vw_y		vw_top.c_y
#define vw_xwin		vw_top.c_xwin
#define vw_flags	vw_top.c_flags
#define vw_dop		vw_top.c_dop
#define vw_dpy		vw_top.c_dpy
#define vw_screen_no	vw_top.c_screen_no
#define vw_visual	vw_top.c_visual

#define NO_VIEWER	((Viewer *) NULL)

/* flag values */
#define CMAP_UPDATE	1		/* needs to reset color map */
#define IMAGE_UPDATE	2		/* needs to redraw image */
#define VIEW_EXPOSED	4
		/*	8	*/	/* what went here? */
#define VIEW_ADJUSTER	16		/* is looking for mouse events */
#define VIEW_MAPPED	32
#define VIEW_TRACK	64		/* track motion events */
#define VIEW_DRAGSCAPE	128

#define VIEW_MOUSESCAPE	256
#define VIEW_BUTTON_ARENA	512	/* 0x100 just watch for button presses */
#define VIEWER_OWNS_IMAGE_DATA	1024	/* 0x200 XImage data allocated by xlib */
#define VIEW_GL		2048		/* 0x400 GL window for sgi */

#define VIEW_XFORM_COORDS	4096	/* 0x800 use plot(5) style coordinate space */
#define VIEW_UNSHOWN	8192		/* 0x1000 unmapped */
#define VIEW_LUT_SIM	0x4000		/* simulate LUT's */
#define VIEW_GLX_RDY	0x8000		/* has been initialized for GLX  */

#define VWR_IS_MAPPED( vp )		(( vp )->vw_flags&VIEW_MAPPED)

#define IS_ADJUSTER( vp )	( ( vp )->vw_flags & VIEW_ADJUSTER )
#define IS_DRAGSCAPE( vp )	( ( vp )->vw_flags & VIEW_DRAGSCAPE )
#define IS_MOUSESCAPE( vp )	( ( vp )->vw_flags & VIEW_MOUSESCAPE )
#define IS_TRACKING( vp )	( ( vp )->vw_flags & VIEW_TRACK )
#define IS_BUTTON_ARENA( vp )	( ( vp )->vw_flags & VIEW_BUTTON_ARENA )
#define IS_GL_WINDOW( vp )	( ( vp )->vw_flags & VIEW_GL )

#define SIMULATING_LUTS(vp)	( ( vp )->vw_flags & VIEW_LUT_SIM )
#define OWNS_IMAGE_DATA( vp ) 	( ( vp )->vw_flags & VIEWER_OWNS_IMAGE_DATA )

#define READY_FOR_GLX( vp )	( ( vp )->vw_flags & VIEW_GLX_RDY )


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
	Cursor		vc_cursor;
	unsigned int	vc_xhot;
	unsigned int	vc_yhot;
} View_Cursor;

#define NO_CURSOR	((View_Cursor *)NULL)

/* prototypes */

extern Disp_Obj *the_dop;


/* viewer.c */

extern void init_viewer_genwin(SINGLE_QSP_ARG_DECL);

ITEM_INTERFACE_PROTOTYPES(Viewer,vwr);
#define VWR_OF( s )	vwr_of(QSP_ARG  s )
#define GET_VWR( s )	get_vwr(QSP_ARG  s )

extern void zap_image_list(Viewer *vp);
extern void select_viewer(QSP_ARG_DECL  Viewer *vp);
extern void release_image(QSP_ARG_DECL  Data_Obj *dp);
extern void delete_viewer(QSP_ARG_DECL  Viewer *vp);
extern Viewer *viewer_init(QSP_ARG_DECL  const char *name,int dx,int dy,int flags);
extern Node *first_viewer_node(SINGLE_QSP_ARG_DECL);
extern List *viewer_list(SINGLE_QSP_ARG_DECL);
extern void info_viewer(Viewer *vp);
extern double viewer_exists(QSP_ARG_DECL  const char *);
extern void init_genwin_viewer(void);


/* canvas.c */

extern void add_image(Viewer *vp,Data_Obj *dp,int x,int y);
extern void insert_image(Data_Obj *dpto,Data_Obj *dpfr,int x,int y,int frameno);
extern void update_image(Viewer *vp);
extern void load_viewer(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp);
extern void old_load_viewer(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp);

/* xplot.c */

extern void tell_plot_space(void);
extern void xp_ffill_arc(float, float, float, float, float, float);
extern void xp_fill_polygon(int num_points, int* xarray, int* yarray);
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
extern void xp_setup(Viewer *vp);

extern void xp_erase(void );
extern void dump_drawlist(Viewer *vp);
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
extern void make_dragg(QSP_ARG_DECL  const char *name,Data_Obj *bm,Data_Obj *dp);
extern void embed_draggable(Data_Obj *dp,Draggable *dgp);
extern Draggable *in_draggable(Viewer *vp,int x,int y);
extern void extract_image(Data_Obj *dpto,Data_Obj *dpfr,int x,int y);

/* cursors.c */

ITEM_INTERFACE_PROTOTYPES(View_Cursor,cursor)
extern void default_cursors(SINGLE_QSP_ARG_DECL);
extern void make_cursor(QSP_ARG_DECL  const char *name,Data_Obj *bitmap_dp,int x,int y);
extern void mk_cursor(QSP_ARG_DECL  const char *name,u_short *data,dimension_t dx,dimension_t dy,dimension_t x,dimension_t y);
extern void root_cursor(View_Cursor *vcp);
extern void assign_cursor(Viewer *vp,View_Cursor *vcp);

#endif /* HAVE_X11 */


#ifdef __cplusplus
}
#endif


#endif /* ! NO_VIEWER */

