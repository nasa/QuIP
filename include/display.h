#include "quip_config.h"


/* display.h */
#ifndef NO_DISPLAY

#include "node.h"
#include "data_obj.h"

#ifdef HAVE_X11
#include "Xhs.h"

#if HAVE_X11_INTRINSIC_H
#include <X11/Intrinsic.h>
#endif

/* These next two files aren't tested for in configure, but it doesn't seem
 * to make an difference!???
 */

#if HAVE_X11_XATOM_H
#include <X11/Xatom.h>
#endif

#if HAVE_X11_SHELL_H
#include <X11/Shell.h>
#endif

#if HAVE_MOTIF
#include <Xm/Xm.h>
/* use XmAll.h instead? */
#include <Xm/CascadeB.h>
#include <Xm/DialogS.h>
#include <Xm/Form.h>
#include <Xm/List.h>
#include <Xm/PushB.h>
#include <Xm/RowColumn.h>
#include <Xm/Scale.h>
#include <Xm/ScrollBar.h>
#include <Xm/TextF.h>
#include <Xm/CascadeBG.h>
#include <Xm/LabelG.h>
#include <Xm/ToggleBG.h>
#endif /* HAVE_MOTIF */

#endif /* HAVE_X11 */

#include "quip_config.h"

#include "dispobj.h"

typedef struct dpyable {		/* canvas & panel shared */
//	Item		dpa_item;
//#define dpa_name		dpa_item.item_name

	unsigned int	dpa_width;
	unsigned int	dpa_height;
	int		dpa_depth;
	int		dpa_x;
	int		dpa_y;		/* for the viewer */
	// "requested" values were introduced when it appeared
	// that on the Mac implementation of the X server,
	// the y position is offset by the thickness of the
	// window top bar...
	int		dpa_x_requested;
	int		dpa_y_requested;
	int		dpa_y_offset;	// difference between actual and requested

	uint32_t	dpa_flags;
	List *		dpa_children;		/* list of sub objects */
	void *		dpa_parent;

	/* colormap stuff */
	Data_Obj *	dpa_cmap_dp;		/* color map */
	Data_Obj *	dpa_lintbl_dp;		/* linearization table */
	short		dpa_n_protected_colors;	/* reserve for system? */


#ifdef HAVE_X11
	/* things needed by xlib */
	/* these should be global!? */

	Disp_Obj *	dpa_dop;
	Window		dpa_xwin;

#define dpa_dpy		dpa_dop->do_dpy
#define dpa_gc		dpa_dop->do_gc
#define dpa_screen_no	dpa_dop->do_screen
#define dpa_visual	dpa_dop->do_visual

	Colormap	dpa_cmap;			/* X color map */
	XColor *	dpa_xctbl;

	/* BUG?  If this stuff is motif-only, then
	 * this structure is not really canvas/panel shared?
	 *
	 * BUT - we have kind of merged viewers and panels
	 * in the iOS implementation, and might like to
	 * emulate that here as well...
	 */
#ifdef HAVE_MOTIF
	Widget		dpa_frame_obj;
	Widget		dpa_thing_obj;	/* what is this for??? */
	Widget		dpa_pw;
	int		dpa_realized;	/* flag to indicate whether or not
					   realized by the Xt toolkit */
#endif /* HAVE_MOTIF */

#endif /* HAVE_X11 */


} Dpyable ;

#define NO_DISPLAY ((Dpyable *)NULL)

#define DPA_CMAP(dpy)		(dpy)->dpa_cmap
#define DPA_CMAP_OBJ(dpy)	(dpy)->dpa_cmap_dp
#define DPA_LINTBL_OBJ(dpy)	(dpy)->dpa_lintbl_dp
#define DPA_N_PROT_CLRS(dpy)	(dpy)->dpa_n_protected_colors
#define DPA_DEPTH(dpy)		(dpy)->dpa_depth

#define DPA_DISPLAY(dpy)	(dpy)->dpa_dpy
#define DPA_XCTBL(dpy)		(dpy)->dpa_xctbl
#define DPA_XWIN(dpy)		(dpy)->dpa_xwin
#define SET_DPA_XCTBL(dpy,v)	(dpy)->dpa_xctbl = v
#define SET_DPA_XCTBL_PIXEL(dpy,i,v)	(dpy)->dpa_xctbl[i].pixel = v
#define SET_DPA_XCTBL_RED(dpy,i,v)	(dpy)->dpa_xctbl[i].red = v
#define SET_DPA_XCTBL_GREEN(dpy,i,v)	(dpy)->dpa_xctbl[i].green = v
#define SET_DPA_XCTBL_BLUE(dpy,i,v)	(dpy)->dpa_xctbl[i].blue = v
#define SET_DPA_XCTBL_FLAGS(dpy,i,v)	(dpy)->dpa_xctbl[i].flags = v

#define SET_DPA_FLAG_BITS(dpy,v)	SET_DPA_FLAGS(dpy,DPA_FLAGS(dpy)|v)
#define CLEAR_DPA_FLAG_BITS(dpy,v)	SET_DPA_FLAGS(dpy,DPA_FLAGS(dpy)&(~v))

#define DPA_FLAGS(dpy)			(dpy)->dpa_flags
#define SET_DPA_FLAGS(dpy,v)		(dpy)->dpa_flags = v

/* flag values are set in gen_win.h */

#ifndef HAVE_X11

/* dummy functions so program will link w/o X11 */
#define UNIMP_MSG(func_name)					\
								\
	sprintf(DEFAULT_ERROR_STRING,				\
		"%s:  program not configured with X11 support!?",	\
		#func_name);					\
	NWARN(DEFAULT_ERROR_STRING);

#endif /* ! HAVE_X11 */

#endif /* NO_DISPLAY */

