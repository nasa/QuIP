#ifndef _PF_VIEWER_H_
#define _PF_VIEWER_H_

#include "quip_config.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef HAVE_GL_GLEW_H
#include <GL/glew.h>
#endif

#ifdef HAVE_GL_GLUT_H
#include <GL/glut.h>
#endif

#include "viewer.h"

typedef struct platform_viewer {
	char *		pv_name;
#ifndef BUILD_FOR_OBJC
	Viewer *	pv_vp;
#endif // ! BUILD_FOR_OBJC
#ifdef HAVE_OPENGL
	GLuint		pv_pbo_buffer;			// Front and back CA buffers
	GLuint		pv_texid;			// Texture for display
#endif // HAVE_OPENGL
} Platform_Viewer;

#define pv_cols		pv_vp->vw_width
#define pv_rows		pv_vp->vw_height

#define NO_PF_VWR		((Platform_Viewer *)NULL)

#define OFFSET(i) ((char *)NULL + (i))

#include "query.h"
#include "data_obj.h"

extern COMMAND_FUNC( do_new_pf_vwr );
extern COMMAND_FUNC( do_load_pf_vwr );
//extern COMMAND_FUNC( do_new_gl_buffer );
//extern int gl_pixel_type(Data_Obj *dp);

#endif  /* ! _PF_VIEWER_H_ */

