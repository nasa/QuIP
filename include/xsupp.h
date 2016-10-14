#ifndef _XSUPP_H_
#define _XSUPP_H_


#include "quip_config.h"
#include "item_type.h"
#include "viewer.h"

#ifdef HAVE_X11


typedef struct sfont {
	char *		xf_name;
	Font		xf_id;
	XFontStruct *	xf_fsp;
} XFont;

#define NO_XFONT	((XFont *)NULL)

/* vbl.c */
extern void vbl_wait(void);

extern void set_font(Viewer *,XFont *);
extern Viewer *find_viewer(QSP_ARG_DECL  Window win);

#endif /* HAVE_X11 */

/* dpy.c */
extern void		set_display(Disp_Obj *dop);
extern Disp_Obj *	open_display(QSP_ARG_DECL  const char *name,int desired_depth);
extern void		info_do(Disp_Obj *dop);
extern void		show_visuals(QSP_ARG_DECL  Disp_Obj *dop);

/* xsync.c */

extern void x_sync_off(void);
extern void x_sync_on(void);

/* view_xlib.c */
extern void wait_for_mapped(QSP_ARG_DECL  Viewer *, int);
extern void show_geom(QSP_ARG_DECL  Viewer *vp);
extern void enable_masked_events(Viewer *,long);
extern int	shm_setup(Viewer *);
extern void update_shm_viewer(Viewer *,char *,int pinc, int cinc,int dx,int dy,int x0,int y0);
extern void cycle_viewer_images(QSP_ARG_DECL  Viewer *vp, int frame_duration);


/* lut_xlib.c */
extern void	 x_dump_lut(Dpyable *dpyp);
extern void	 install_colors(Dpyable *dpyp);
#ifdef HAVE_X11
extern void	set_curr_win(Window win);
#endif /* HAVE_X11 */

/* which_display.c */
extern const char *which_display( SINGLE_QSP_ARG_DECL );

/* check_display.c */
extern const char *check_display(SINGLE_QSP_ARG_DECL);


#endif /* ! _XSUPP_H_ */

