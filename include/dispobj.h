
#ifndef NO_DISP_OBJ

#include "Xhs.h"

#ifdef HAVE_OPENGL
#ifdef HAVE_GL_GLX_H
#include <GL/glx.h>
#endif /* HAVE_GL_GLX_H */
#endif /* HAVE_OPENGL */

typedef struct display_object {
	Item		do_item;
#define do_name		do_item.item_name
	Display *	do_dpy;
	Visual *	do_visual;
	int		do_screen;
	GC		do_gc;
	Window		do_rootw;
	Window		do_currw;
	int		do_width, do_height, do_depth;
#ifdef HAVE_OPENGL
	GLXContext	do_ctx;
#endif /* HAVE_OPENGL */

} Disp_Obj;

#define do_fg	do_gc->values.foreground
#define do_bg	do_gc->values.background

#define NO_DISP_OBJ	((Disp_Obj *) NULL)

#endif /* NO_DISP_OBJ */

