#include "quip_config.h"

#ifdef HAVE_X11

/* display.h */
#ifndef NO_DISPLAY

#include "node.h"
#include "data_obj.h"

#include "Xhs.h"

#include "quip_config.h"

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
#endif

#include "dispobj.h"

typedef struct dpyable {		/* canvas & panel shared */
	Item		c_item;
#define c_name		c_item.item_name

	unsigned int	c_width;
	unsigned int	c_height;
	int		c_depth;
	int		c_x;
	int		c_y;		/* for the viewer */

	Window		c_xwin;
	u_long		c_flags;
	List *		c_children;		/* list of sub objects */
	void *		c_parent;

	/* things needed by xlib */
	/* these should be global!? */

	Disp_Obj *	c_dop;

#define c_dpy		c_dop->do_dpy
#define c_gc		c_dop->do_gc
#define c_screen_no	c_dop->do_screen
#define c_visual	c_dop->do_visual

	/* colormap stuff */
	Data_Obj *	c_cm_dp;		/* color map */
	Data_Obj *	c_lt_dp;		/* linearization table */
	Colormap	c_cmap;			/* X color map */
	XColor *	c_xctbl;
	short		c_n_protected_colors;	/* reserve for system? */

	/* If this stuff is motif-only, then this structure is not
	 * really canvas/panel shared?
	 */
#ifdef HAVE_MOTIF
	Widget		c_frame_obj;
	Widget		c_thing_obj;	/* what is this for??? */
	Widget		c_pw;
	int		c_realized;	/* flag to indicate whether or not
					   realized by the Xt toolkit */
#endif /* HAVE_MOTIF */

} Dpyable ;

#define NO_DISPLAY ((Dpyable *)NULL)

/* flag values */

#define KNOW_SYSCOLORS	1

#define HAS_SYSCOLORS(dpyp)		(((dpyp)->c_flags) & KNOW_SYSCOLORS)
#define HAS_COLORMAP(dpyp)		(((dpyp)->c_depth) == 8)

#endif /* NO_DISPLAY */

#endif /* HAVE_X11 */

