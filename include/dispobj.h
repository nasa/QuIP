
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
	Item		dispobj_item;
#define dispobj_name		dispobj_item.item_name
	int		dispobj_width, dispobj_height, dispobj_depth;

#ifdef HAVE_X11
	Display *	dispobj_dpy;
	Visual *	dispobj_visual;
	int		dispobj_screen;
	GC		dispobj_gc;
	Window		dispobj_rootw;
	Window		dispobj_currw;
#ifdef HAVE_OPENGL
	GLXContext	dispobj_ogl_ctx;
#endif /* HAVE_OPENGL */
#endif /* HAVE_X11 */

} Disp_Obj;

#define DO_NAME(dop)		(dop)->dispobj_item.item_name
#define DO_HEIGHT(dop)		(dop)->dispobj_height
#define DO_DEPTH(dop)		(dop)->dispobj_depth
#define DO_WIDTH(dop)		(dop)->dispobj_width
#define DO_DISPLAY(dop)		(dop)->dispobj_dpy
#define DO_VISUAL(dop)		(dop)->dispobj_visual
#define DO_SCREEN(dop)		(dop)->dispobj_screen
#define DO_GC(dop)		(dop)->dispobj_gc
#define DO_ROOTW(dop)		(dop)->dispobj_rootw
#define DO_CURRW(dop)		(dop)->dispobj_currw
#define DO_OGL_CTX(dop)		(dop)->dispobj_ogl_ctx

#define SET_DO_HEIGHT(dop,v)	(dop)->dispobj_height = v
#define SET_DO_DEPTH(dop,v)	(dop)->dispobj_depth = v
#define SET_DO_WIDTH(dop,v)	(dop)->dispobj_width = v
#define SET_DO_DISPLAY(dop,v)	(dop)->dispobj_dpy = v
#define SET_DO_VISUAL(dop,v)	(dop)->dispobj_visual = v
#define SET_DO_SCREEN(dop,v)	(dop)->dispobj_screen = v
#define SET_DO_GC(dop,v)	(dop)->dispobj_gc = v
#define SET_DO_ROOTW(dop,v)	(dop)->dispobj_rootw = v
#define SET_DO_CURRW(dop,v)	(dop)->dispobj_currw = v
#define SET_DO_OGL_CTX(dop,v)	(dop)->dispobj_ogl_ctx = v

#define dispobj_fg	dispobj_gc->values.foreground
#define dispobj_bg	dispobj_gc->values.background

ITEM_INTERFACE_PROTOTYPES(Disp_Obj,disp_obj)

#define PICK_DISP_OBJ(p)	pick_disp_obj(QSP_ARG  p)

extern Disp_Obj *	curr_dop(void);

#endif /* ! _DISPOBJ_H_ */

