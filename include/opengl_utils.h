
#ifndef HAVE_OPENGL_UTILS_H
#define HAVE_OPENGL_UTILS_H

#include "data_obj.h"

#ifdef HAVE_OPENGL
extern int gl_pixel_type(Data_Obj *dp);
extern void glew_check(SINGLE_QSP_ARG_DECL);
#endif // HAVE_OPENGL

#endif // ! HAVE_OPENGL_UTILS_H

