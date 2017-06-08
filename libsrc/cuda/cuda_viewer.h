#ifndef _CUDA_VIEWER_H_
#define _CUDA_VIEWER_H_

#include "quip_config.h"


#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_GL_GLEW_H
#include <GL/glew.h>
#endif


/*
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
*/

#ifdef HAVE_GL_GLUT_H
#include <GL/glut.h>
#endif

#ifdef HAVE_CUDA
#ifdef OLD_CUDA4
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cuda_gl_interop.h>
#endif /* OLD_CUDA4 */
#endif /* HAVE_CUDA */

#include "glx_hack.h"
#include "viewer.h"

typedef struct cuda_viewer {
	char *		cv_name;
	Viewer *	cv_vp;
	GLuint		cv_pbo_buffer;			// Front and back CA buffers
	GLuint		cv_texid;			// Texture for display
} Cuda_Viewer;

#define cv_cols		cv_vp->vw_width
#define cv_rows		cv_vp->vw_height

#define OFFSET(i) ((char *)NULL + (i))

#include "data_obj.h"

extern COMMAND_FUNC( do_new_cuda_vwr );
extern COMMAND_FUNC( do_load_cuda_vwr );
extern COMMAND_FUNC( do_map_cuda_vwr );
extern COMMAND_FUNC( gl_test );
extern COMMAND_FUNC( gl_disp );
extern COMMAND_FUNC( do_new_gl_buffer );

#endif  /* ! _CUDA_VIEWER_H_ */

