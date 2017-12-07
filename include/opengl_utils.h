
#ifndef HAVE_OPENGL_UTILS_H
#define HAVE_OPENGL_UTILS_H

#include "data_obj.h"

#ifdef HAVE_OPENGL
extern int _gl_pixel_type(QSP_ARG_DECL  Data_Obj *dp);
#define gl_pixel_type(dp) _gl_pixel_type(QSP_ARG  dp)

extern void glew_check(SINGLE_QSP_ARG_DECL);
#endif // HAVE_OPENGL

#endif // ! HAVE_OPENGL_UTILS_H

