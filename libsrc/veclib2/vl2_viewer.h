
#ifndef _OCL_VIEWER_H_
#define _OCL_VIEWER_H_

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

#include "viewer.h"

// There doesn't seem to be anything cuda-specific in this struct,
// So it should be shareable!  PORT

typedef struct vl2_viewer {
	char *		ov_name;
	Viewer *	ov_vp;
	GLuint		ov_pbo_buffer;			// Front and back CA buffers
	GLuint		ov_texid;			// Texture for display
} OCL_Viewer;

#define ov_cols		ov_vp->vw_width
#define ov_rows		ov_vp->vw_height

#define NO_OCL_VWR		((OCL_Viewer *)NULL)

#define OFFSET(i) ((char *)NULL + (i))

#include "query.h"
#include "data_obj.h"

extern COMMAND_FUNC( do_new_vl2_vwr );
extern COMMAND_FUNC( do_load_vl2_vwr );
extern COMMAND_FUNC( gl_test );
extern COMMAND_FUNC( gl_disp );
extern COMMAND_FUNC( do_new_gl_buffer );

#endif  /* ! _OCL_VIEWER_H_ */

