
#ifndef _DISPOBJ_H_
#define _DISPOBJ_H_

#ifdef HAVE_X11
#include "Xhs.h"
#endif /* HAVE_X11 */

#ifdef HAVE_OPENGL
#ifdef HAVE_GL_GLX_H
#ifdef BUILD_FOR_OBJC
#include <OpenGL/glx.h>
#else // ! BUILD_FOR_OBJC
#include <GL/glx.h>
#endif // ! BUILD_FOR_OBJC
#endif /* HAVE_GL_GLX_H */
#endif /* HAVE_OPENGL */

typedef struct display_object {
	Item		do_item;
#define do_name		do_item.item_name
	int		do_width, do_height, do_depth;

#ifdef HAVE_X11
	Display *	do_dpy;
	Visual *	do_visual;
	int		do_screen;
	GC		do_gc;
	Window		do_rootw;
	Window		do_currw;
#ifdef HAVE_OPENGL
	GLXContext	do_ctx;
#endif /* HAVE_OPENGL */
#endif /* HAVE_X11 */

} Disp_Obj;

#define do_fg	do_gc->values.foreground
#define do_bg	do_gc->values.background

#define NO_DISP_OBJ	((Disp_Obj *) NULL)

ITEM_INTERFACE_PROTOTYPES(Disp_Obj,disp_obj)

#define PICK_DISP_OBJ(p)	pick_disp_obj(QSP_ARG  p)

extern Disp_Obj *	curr_dop(void);

#endif /* ! _DISPOBJ_H_ */

