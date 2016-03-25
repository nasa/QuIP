
#include "quip_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef HAVE_GL_GLEW_H
#include <GL/glew.h>
#endif

// used to include GL/glut.h and rendercheck_gl.h...

#ifdef HAVE_GL_GLX_H
#include <GL/glx.h>	// jbm added for glXSwapBuffers()
#endif

#include "GL/glext.h"	// from sample code

#ifdef FOOBAR
void cleanup_cu2_viewer(void);
#endif /* FOOBAR */

#define NO_OGL_MSG		WARN("Sorry, no openGL in this build!?");

#include "quip_prot.h"
#include "my_cu2.h"
#include "veclib/cu2_menu_prot.h"
#include "cu2_viewer.h"
#include "gl_viewer.h"		/* select_gl_viewer() */
#include "gl_info.h"




#ifdef FOOBAR

void cleanup_cu2_viewer(Cuda_Viewer *ovp)
{
	cutilSafeCall(cudaGLUnregisterBufferObject(ovp->ov_pbo_buffer));

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glDeleteBuffers(1, &ovp->ov_pbo_buffer);
	glDeleteTextures(1, &ovp->ov_texid);
	//deleteTexture();
}
#endif /* FOOBAR */

int cu2_register_buf(QSP_ARG_DECL  Data_Obj *dp)
{
	cudaError_t e;

	/* how do we check for an error? */
	e = cudaGLRegisterBufferObject( OBJ_BUF_ID(dp) );
	if( e != cudaSuccess ){
		describe_cuda_driver_error2("cu2_register_buf",
				"cudaGLRegisterBufferObject",e);
		return -1;
	}
	return 0;
}

int cu2_map_buf(QSP_ARG_DECL  Data_Obj *dp)
{
	cudaError_t e;

	e = cudaGLMapBufferObject( &OBJ_DATA_PTR(dp),  OBJ_BUF_ID(dp) );
	if( e != cudaSuccess ){
		describe_cuda_driver_error2("cu2_map_buf",
				"cudaGLMapBufferObject",e);
		return -1;
	}
	return 0;
}




