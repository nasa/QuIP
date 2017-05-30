#ifndef _PF_VIEWER_H_
#define _PF_VIEWER_H_

#include "quip_config.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#ifdef HAVE_GL_GLEW_H
//#include <GL/glew.h>
//#endif

#ifdef HAVE_GL_GLUT_H
#ifndef BUILD_FOR_OBJC
#include <GL/glut.h>
#endif // ! BUILD_FOR_OBJC
#endif // HAVE_GL_GLUT_H

#include "viewer.h"

#ifndef BUILD_FOR_OBJC

typedef struct platform_viewer {
	char *		pv_name;
	Viewer *	pv_vp;
#define PFVWR_VIEWER(pvp)	(pvp)->pv_vp
#define SET_PFVWR_VIEWER(pvp,v)	(pvp)->pv_vp = v
#ifdef HAVE_OPENGL
	GLuint		pv_pbo_buffer;			// Front and back CA buffers
	GLuint		pv_texid;			// Texture for display
#define PFVWR_BUFFER(pvp)	(pvp)->pv_pbo_buffer
#define PFVWR_TEXID(pvp)	(pvp)->pv_texid
#define SET_PFVWR_BUFFER(pvp,v)	(pvp)->pv_pbo_buffer = v
#define SET_PFVWR_TEXID(pvp,v)	(pvp)->pv_texid = v
#endif // HAVE_OPENGL
} Platform_Viewer;

#define pv_cols		pv_vp->vw_width
#define pv_rows		pv_vp->vw_height

#else // BUILD_FOR_OBJC

@interface Platform_Viewer : IOS_Item
@property Viewer *	pv_vp;

#ifdef HAVE_OPENGL
@property GLuint	pv_pbo_buffer;			// Front and back CA buffers
@property GLuint	pv_texid;			// Texture for display
#endif // HAVE_OPENGL

+(void) initClass;
@end

#define PFVWR_VIEWER(pvp)	(pvp).pv_vp
#define PFVWR_BUFFER(pvp)	(pvp).pv_pbo_buffer
#define PFVWR_TEXID(pvp)	(pvp).pv_texid

#define SET_PFVWR_VIEWER(pvp,v)	(pvp).pv_vp = v
#define SET_PFVWR_BUFFER(pvp,v)	(pvp).pv_pbo_buffer = v
#define SET_PFVWR_TEXID(pvp,v)	(pvp).pv_texid = v

#endif // BUILD_FOR_OBJC

#define NO_PF_VWR		((Platform_Viewer *)NULL)
#define OFFSET(i) ((char *)NULL + (i))	// what is this used for???


#include "data_obj.h"

extern COMMAND_FUNC( do_new_pf_vwr );
extern COMMAND_FUNC( do_load_pf_vwr );
//extern COMMAND_FUNC( do_new_gl_buffer );
//extern int gl_pixel_type(Data_Obj *dp);

#endif  /* ! _PF_VIEWER_H_ */

