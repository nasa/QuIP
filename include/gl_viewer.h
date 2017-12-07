#ifndef _GL_VIEWER_H_
#define _GL_VIEWER_H_

#include "viewer.h"

#ifdef __cplusplus
extern "C" {
#endif


/* glx_supp.c */
extern void _swap_buffers(SINGLE_QSP_ARG_DECL);
#define swap_buffers() _swap_buffers(SINGLE_QSP_ARG)

extern void select_gl_viewer(QSP_ARG_DECL  Viewer *vp);


#ifdef __cplusplus
}
#endif

#endif /* ! _GL_VIEWER_H_ */

