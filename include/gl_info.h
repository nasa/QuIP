#ifndef _GL_INFO_H_
#define _GL_INFO_H_

#ifdef HAVE_OPENGL

#ifndef BUILD_FOR_OPENCL
#ifdef HAVE_GL_GLEW_H
#include <GL/glew.h>
#endif
#endif // ! BUILD_FOR_OPENCL

// This structure may be pointed to by unaligned_data...  a kludge!

typedef struct gl_info {
	GLuint		buf_id;
	GLuint		tex_id;
} GL_Info;

#define GLI_BUF_ID(gli_p)	(gli_p)->buf_id
#define GLI_TEX_ID(gli_p)	(gli_p)->tex_id

#define OBJ_BUF_ID(dp)		GLI_BUF_ID(OBJ_GL_INFO(dp))
#define OBJ_TEX_ID(dp)		GLI_TEX_ID(OBJ_GL_INFO(dp))
#define OBJ_BUF_ID_P(dp)	( & (GLI_BUF_ID(OBJ_GL_INFO(dp))) )
#define OBJ_TEX_ID_P(dp)	( & (GLI_TEX_ID(OBJ_GL_INFO(dp))) )

#endif /* HAVE_OPENGL */

#endif /* ! _GL_INFO_H_ */

