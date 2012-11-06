#include "quip_config.h"

#ifdef HAVE_X11

#ifndef _XSUPP_H_
#define _XSUPP_H_

#include "viewer.h"

typedef struct sfont {
	char *	xf_name;
	Font	xf_id;
} XFont;

#define NO_XFONT	((XFont *)NULL)

#ifdef COLORMAP_STUFF_FOOBAR
/* This stuff is needed to deal with colormaps under X */
/* Why are we not using it??? */

typedef struct xllutdata {
	Disp_Obj *	xld_dop;
	Window		xld_xwin;
	XColor *	xld_xctbl;
	Colormap	xld_cmap;		/* X color map */
	short		xld_protected_colors;	/* reserve for system? */
	short		xld_flags;
} XlibData;

extern XlibData *curr_xldp;

#define xld_dpy		xld_dop->do_dpy
#define xld_screen	xld_dop->do_screen
#define xld_visual	xld_dop->do_visual
#define xld_depth	xld_dop->do_depth


#define NO_XLIBDATA	((XlibData *) NO_PTR)

/* flag values */
#define CMAP_UPDATE	1		/* needs to reset color map */
#define KNOW_SYSCOLORS	2		/* have already grabbed sys colors */

#define HAS_SYSCOLS( xldp )	( ( xldp )->xld_flags & KNOW_SYSCOLORS )
#define HAS_COLORMAP( xldp )	( ( xldp )->xld_depth != 24 )
#endif /* COLORMAP_STUFF_FOOBAR */

/* globals */
extern u_long xdebug;
extern int simulating_luts;


unsigned long convert_color_8to24(unsigned int ui8bitcolor);


/* check_display.c */
extern const char *check_display(void);

/* dpy.c */

ITEM_INTERFACE_PROTOTYPES(Disp_Obj,disp_obj)

extern void		set_display(Disp_Obj *dop);
extern Disp_Obj *	curr_dop(void);
extern List *		displays_list(SINGLE_QSP_ARG_DECL);
extern void		info_do(Disp_Obj *dop);
extern void		dop_create(const char *name,Disp_Obj **dopp);
extern Disp_Obj *	open_display(QSP_ARG_DECL  const char *name,int desired_depth);
extern Disp_Obj *	default_x_display(SINGLE_QSP_ARG_DECL);
extern int		display_depth(SINGLE_QSP_ARG_DECL);

/* view_xlib.c */
extern int	get_string_width(Viewer *vp, const char *s);
extern void	set_viewer_display(Viewer *vp);
extern void enable_masked_events(Viewer *,long);
extern void disable_masked_events(Viewer *,long);
extern void wait_for_mapped(QSP_ARG_DECL  Viewer *, int);
extern void center_text(void);
extern void left_justify(void);
extern void right_justify(void);
extern void set_font(Viewer *,XFont *);
extern void set_remember_gfx(int flag);
extern Visual *GetEightBitVisual(Disp_Obj *dop);
extern Visual *Get24BitVisual(Disp_Obj *dop);
extern int make_generic_window(QSP_ARG_DECL  Viewer *vp,long event_mask);
extern int make_2d_adjuster(QSP_ARG_DECL  Viewer *vp);
extern int make_button_arena(QSP_ARG_DECL  Viewer *vp);
extern int make_dragscape(QSP_ARG_DECL  Viewer *vp);
extern int make_mousescape(QSP_ARG_DECL  Viewer *vp);
extern int make_viewer(QSP_ARG_DECL  Viewer *vp);
extern int make_gl_window(QSP_ARG_DECL  Viewer *vp);
extern Viewer *find_viewer(QSP_ARG_DECL  Window win);
extern void show_viewer(QSP_ARG_DECL  Viewer *vp);
extern void unshow_viewer(QSP_ARG_DECL  Viewer *vp);
extern void embed_image(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp,u_int x,u_int y);
extern void unembed_image(QSP_ARG_DECL  Viewer *vp,Data_Obj *dp,u_int x,u_int y);
extern void refresh_image(QSP_ARG_DECL  Viewer *vp);
extern void redraw_viewer(QSP_ARG_DECL  Viewer *vp);
extern void posn_viewer(Viewer *vp,int x,int y);
extern void zap_viewer(Viewer *vp);
extern void relabel_viewer(Viewer *vp,const char *s);
extern void _xp_fill_polygon(Viewer* vp, int num_points, int* px_vals, int* py_vals);
extern void _xp_fill_arc(Viewer*, int, int, int, int, int, int);
extern void _xp_line(Viewer *vp,int x1,int y1,int x2,int y2);
extern void _xp_move(Viewer *vp,int x1,int y1);
extern void _xp_cont(Viewer *vp,int x1,int y1);
extern void _xp_text(Viewer *vp,int x1,int y1,const char *);
extern void _xp_erase(Viewer *vp);
extern void _xp_select(Viewer *vp,u_long color);
extern void _xp_bgselect(Viewer *vp,u_long color);
extern void _xp_arc(Viewer *,int,int,int,int,int,int);
extern void show_geom(Viewer *vp);
extern void extra_viewer_info(Viewer *vp);
extern void window_sys_init(SINGLE_QSP_ARG_DECL);

extern void update_shm_viewer(Viewer *,char *,int pinc, int cinc,int dx,int dy,int x0,int y0);
extern int	shm_setup(Viewer *);

void get_geom(Viewer* vp, u_int* width, u_int* height, u_int* depth);
/* xsync.c */

extern void x_sync_off(void);
extern void x_sync_on(void);

/* event.c */

int check_one_display(QSP_ARG_DECL  Disp_Obj *dop);
int event_loop(SINGLE_QSP_ARG_DECL);
void do_enter_leave(XEvent *event);
void pickup(QSP_ARG_DECL  Draggable *dgp,Viewer *vp);
void drag_to(int x,int y,Viewer *vp);
void put_down(QSP_ARG_DECL  int x,int y,Viewer *vp);
void i_loop(SINGLE_QSP_ARG_DECL);
void discard_events(SINGLE_QSP_ARG_DECL);


/* lut_xlib.c */

extern void	cmap_setup(Viewer *);
extern u_long	simulate_lut_mapping(Viewer *vp, u_long color);
extern Window	curr_win(void);
extern void	set_curr_win(Window win);
/*
//extern void	xld_setup(XlibData *xldp,Disp_Obj *dop,Window win);
//extern void	x_dispose_lb(Lutbuf *lbp);
//extern void	 x_init_lb_data(Lutbuf *lbp);
//extern void	 set_xl_cmap(XlibData *xldp,Data_Obj *cm_dp);
//extern void	 x_assign_lutbuf(Lutbuf *lbp,Data_Obj *cm_dp);
//extern void	 x_read_lutbuf(Data_Obj *cm_dp,Lutbuf *lbp);
//extern void	 x_show_lb_value(Lutbuf *lbp,int index);
//extern void	 x_lb_extra_info(Lutbuf *lbp);
*/
extern void	 x_set_n_protect(int n);
extern void	 fetch_system_colors(Dpyable *dpyp);
extern void	 install_colors(Dpyable *dpyp);
extern void	 x_dump_lut(Dpyable *dpyp);
extern void	 x_lut_init(void);

/* which_display.c */

extern const char *which_display( void );

/* vgt.c */

extern void vbl_wait(void);



#endif /* ! _XSUPP_H_ */

#endif /* HAVE_X11 */

