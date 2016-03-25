#ifndef _GL_VIEWER_H_
#define _GL_VIEWER_H_

#include "viewer.h"

#ifdef __cplusplus
extern "C" {
#endif


/* glx_supp.c */
extern void swap_buffers(void);
extern void select_gl_viewer(QSP_ARG_DECL  Viewer *vp);


#ifdef __cplusplus
}
#endif

#endif /* ! _GL_VIEWER_H_ */

