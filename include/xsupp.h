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

/* vbl.c */
extern void _vbl_wait(SINGLE_QSP_ARG_DECL);
#define vbl_wait() _vbl_wait(SINGLE_QSP_ARG)

extern void set_font(Viewer *,XFont *);
extern Viewer *_find_viewer(QSP_ARG_DECL  Window win);

#define find_viewer(win) _find_viewer(QSP_ARG  win)

#endif /* HAVE_X11 */

/* dpy.c */
extern void		set_display(Disp_Obj *dop);
extern void		info_do(Disp_Obj *dop);

extern Disp_Obj *	_open_display(QSP_ARG_DECL  const char *name,int desired_depth);
extern void		_show_visuals(QSP_ARG_DECL  Disp_Obj *dop);

#define open_display(name,d)	_open_display(QSP_ARG  name,d)
#define show_visuals(dop)	_show_visuals(QSP_ARG  dop)


/* xsync.c */

extern void x_sync_off(void);
extern void x_sync_on(void);

/* view_xlib.c */
extern void enable_masked_events(Viewer *,long);
extern int	shm_setup(Viewer *);
extern void update_shm_viewer(Viewer *,char *,int pinc, int cinc,int dx,int dy,int x0,int y0);

extern void _wait_for_mapped(QSP_ARG_DECL  Viewer *, int);
extern void _show_geom(QSP_ARG_DECL  Viewer *vp);
extern void _cycle_viewer_images(QSP_ARG_DECL  Viewer *vp, int frame_duration);

#define wait_for_mapped(vp,i)		_wait_for_mapped(QSP_ARG  vp,i)
#define show_geom(vp)			_show_geom(QSP_ARG  vp)
#define cycle_viewer_images(vp,d)	_cycle_viewer_images(QSP_ARG  vp,d)

/* lut_xlib.c */
extern void	 x_dump_lut(Dpyable *dpyp);
extern void	 install_colors(Dpyable *dpyp);
#ifdef HAVE_X11
extern void	set_curr_win(Window win);
#endif /* HAVE_X11 */

/* which_display.c */
extern const char *_which_display( SINGLE_QSP_ARG_DECL );

/* check_display.c */
extern const char *_check_display(SINGLE_QSP_ARG_DECL);

#define which_display() _which_display(SINGLE_QSP_ARG)
#define check_display() _check_display(SINGLE_QSP_ARG)

#endif /* ! _XSUPP_H_ */

