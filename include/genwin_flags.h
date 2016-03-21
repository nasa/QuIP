
#ifndef _GENWIN_FLAGS_H_
#define _GENWIN_FLAGS_H_

// BUG?  should we use an enum for the flag bits?

/* Viewer flag values */
#define CMAP_UPDATE		0x0001		/* needs to reset color map */
#define IMAGE_UPDATE		0x0002		/* needs to redraw image */
#define VIEW_EXPOSED		0x0004
#define VIEW_PLOTTER		0x0008
#define VIEW_ADJUSTER		0x0010		/* is looking for mouse events */
#define VIEW_MAPPED		0x0020
#define VIEW_TRACK		0x0040		/* track motion events */
#define VIEW_DRAGSCAPE		0x0080

#define VIEW_MOUSESCAPE		0x0100
#define VIEW_BUTTON_ARENA	0x0200	/* just watch for button presses */
#define VIEWER_OWNS_IMAGE_DATA	0x0400	/* XImage data allocated by xlib */
#define VIEW_GL			0x0800	/* GL window for sgi */

#define VIEW_XFORM_COORDS	0x1000	/* use plot(5) style coordinate space */
#define VIEW_UNSHOWN		0x2000	/* unmapped */
#define VIEW_LUT_SIM		0x4000	/* simulate LUT's */
#define VIEW_GLX_RDY		0x8000	/* has been initialized for GLX  */
#define VIEW_PIXMAP		0x10000	/* an off-screen drawable */

/* panel flag bits */
#define PANEL_IS_SHOWABLE		0x10000
#define PANEL_KNOWS_SYSTEM_COLORS	0x20000
#define PANEL_SHOWN			0x40000		/* "mapped" */
#define NAVP_SHOWN			PANEL_SHOWN

#define VW_NEEDS_REFRESH		0x80000
#define VW_LEFT_JUSTIFY			0x100000
#define VW_CENTER_TEXT			0x200000
#define VW_RIGHT_JUSTIFY		0x400000
#define VW_JUSTIFY_MASK			(VW_LEFT_JUSTIFY|VW_CENTER_TEXT|VW_RIGHT_JUSTIFY)

/* genwin flags */
#define GENWIN_SHOWN			0x800000

#define VW_TXT_MTRX_INITED		0x1000000

/* display flags */
#define KNOW_SYSCOLORS			0x2000000

#define VW_PROG_MOVE_REQ		0x4000000

#define HAS_SYSCOLORS(dpyp)		(DPA_FLAGS(dpyp) & KNOW_SYSCOLORS)
#define HAS_COLORMAP(dpyp)		(DPA_DEPTH(dpyp) == 8)

/* type flags */
// BUG these are not implemented everywhere...
#define GW_IS_VIEWER			0x8000000
#define GW_IS_PANEL			0x10000000
#define GW_IS_NAV_PANEL			0x20000000

#endif /* ! _GENWIN_FLAGS_H_ */


