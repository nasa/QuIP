
#include "quip_config.h"

char VersionId_cuda_cuda_viewer[] = QUIP_VERSION_STRING;

#ifdef HAVE_CUDA
#ifdef HAVE_OPENGL
#ifdef HAVE_GLUT

/* Put the cuda includes first to compile on mac??? */

#include <cuda_runtime.h>
#include <cutil_inline.h>
#ifdef HAVE_GL_GLEW_H
#include <GL/glew.h>
#endif

#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STDIO_H
#include <stdio.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

// used to include GL/glut.h and rendercheck_gl.h...

#ifdef HAVE_GL_GLX_H
#include <GL/glx.h>	// jbm added for glXSwapBuffers()
#endif

void cleanup_cuda_viewer(void);

#include "my_cuda.h"
#include "cuda_supp.h"
#include "cuda_viewer.h"
#include "query.h"
#include "gl_viewer.h"		/* select_gl_viewer() */

ITEM_INTERFACE_DECLARATIONS( Cuda_Viewer, cuda_vwr )

static void propagate_up(Data_Obj *dp,uint32_t flagbit)
{
	if( dp->dt_parent != NO_OBJ ){
		xfer_cuda_flag(dp->dt_parent,dp,flagbit);
		propagate_up(dp->dt_parent,flagbit);
	}
}

static void propagate_down(Data_Obj *dp,uint32_t flagbit)
{
	Node *np;

	if( dp->dt_children != NO_LIST ){
		np=dp->dt_children->l_head;
		while(np!=NO_NODE){
			Data_Obj *child_dp;
			child_dp = (Data_Obj *)np->n_data;
			xfer_cuda_flag(child_dp,dp,flagbit);
			propagate_down(child_dp,flagbit);
			np = np->n_next;
		}
	}
}

static void propagate_flag(Data_Obj *dp,uint32_t flagbit)
{
	propagate_up(dp,flagbit);
	propagate_down(dp,flagbit);
}

static int gl_pixel_type(Data_Obj *dp)
{
	int t;

	switch(dp->dt_comps){
		case 1: t = GL_LUMINANCE; break;
		/* 2 is allowable, but what do we do with it? */
		case 3: t = GL_BGR; break;
		case 4: t = GL_BGRA; break;
		default:
			t=0;	// quiet compiler
			NERROR1("bad pixel depth!?");
			break;
	}
	return(t);
}

void init_cuda_viewer(Cuda_Viewer *cvp)
{
	cvp->cv_pbo_buffer = 0;
	cvp->cv_texid = 0;
}

// This is the normal display path
void update_cuda_viewer(Cuda_Viewer *cvp, Data_Obj *dp) 
{
	int t;
	cudaError_t e;

	// unmap buffer before using w/ GL
	if( BUF_IS_MAPPED(dp) ){
		e = cudaGLUnmapBufferObject( BUF_ID(dp) );   
		if( e != cudaSuccess ){
			describe_cuda_error2("update_cuda_viewer",
				"cudaGLUnmapBufferObject",e);
			NERROR1("failed to unmap buffer object");
		}
		dp->dt_flags &= ~DT_BUF_MAPPED;
		// propagate change to children and parents
		propagate_flag(dp,DT_BUF_MAPPED);

	}


	//
	//bind_texture(dp->dt_data);

	glClear(GL_COLOR_BUFFER_BIT);

/*
sprintf(error_string,"update_cuda_viewer:  tex_id = %d, buf_id = %d",
TEX_ID(dp),BUF_ID(dp));
advise(error_string);
*/
	glBindTexture(GL_TEXTURE_2D, TEX_ID(dp));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, BUF_ID(dp));

#ifdef FOOBAR
	switch(dp->dt_comps){
		/* what used to be here??? */
	}
#endif /* FOOBAR */

	t=gl_pixel_type(dp);
	glTexSubImage2D(GL_TEXTURE_2D, 0,			// target, level
		0, 0,						// x0, y0
		dp->dt_cols, dp->dt_rows, 			// dx, dy
		t,
		GL_UNSIGNED_BYTE,				// type
		OFFSET(0));					// offset into PIXEL_UNPACK_BUFFER

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	glBegin(GL_QUADS);
	glTexCoord2f(0, 1); glVertex2f(-1.0, -1.0);
	glTexCoord2f(0, 0); glVertex2f(-1.0, 1.0);
	glTexCoord2f(1, 0); glVertex2f(1.0, 1.0);
	glTexCoord2f(1, 1); glVertex2f(1.0, -1.0);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);

	//glutSwapBuffers();
	//glutPostRedisplay();

	// Maybe we want to call swap buffers ourselves,
	// if we are trying to synchronize the display
	// and are updating multiple windows?

	//glXSwapBuffers(cvp->cv_vp->vw_dpy,cvp->cv_vp->vw_xwin);

	// re-map so we can use again with CUDA
	// BUG?  Is it safe to do this before the call to swap_buffers???
	cutilSafeCall(cudaGLMapBufferObject( &dp->dt_data,  BUF_ID(dp) ));
	dp->dt_flags |= DT_BUF_MAPPED;
	// propagate change to children and parents
	propagate_flag(dp,DT_BUF_MAPPED);
}

void idle(void)
{
	glutPostRedisplay();
}

void reshape(int x, int y)
{
	glViewport(0, 0, x, y);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, 0, 1); 
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glutPostRedisplay();
}

void cleanup_cuda_viewer(Cuda_Viewer *cvp)
{
	cutilSafeCall(cudaGLUnregisterBufferObject(cvp->cv_pbo_buffer));

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	glDeleteBuffers(1, &cvp->cv_pbo_buffer);
	glDeleteTextures(1, &cvp->cv_texid);
	//deleteTexture();
}


int cuda_viewer_subsystem_inited=0;

void init_cuda_viewer_subsystem(void)
{
	const char *pn;
	int n;

	pn = tell_progname();

	if( cuda_viewer_subsystem_inited ){
		NWARN("Cuda viewer subsystem already initialized!?");
		return;
	}

	// First initialize OpenGL context, so we can properly set
	// the GL for CUDA.
	// This is necessary in order to achieve optimal performance
	// with OpenGL/CUDA interop.

	//glutInit( &argc, argv);		/* BUG?  where should this be done? */
	n=1;
	glutInit( &n, (char **)&pn);		/* BUG?  where should this be done? */

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);

	cuda_viewer_subsystem_inited=1;
}

void glew_check()
{
	static int glew_checked=0;

	if( glew_checked ){
		if( verbose )
			advise("glew_check:  glew already checked.");
		return;
	}

	glewInit();
	if (!glewIsSupported( "GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object" )) {
		/*
		fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
		fprintf(stderr, "This sample requires:\n");
		fprintf(stderr, "  OpenGL version 1.5\n");
		fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
		fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
		*/
		/*
		cudaThreadExit();
		exit(-1);
		*/
NERROR1("glew_check:  Please create a GL window before specifying a cuda viewer.");
	}

	glew_checked=1;
}

Cuda_Viewer * new_cuda_viewer(QSP_ARG_DECL  Viewer *vp)
{
	Cuda_Viewer *cvp;

	if( !cuda_viewer_subsystem_inited ){
		if( verbose )
	advise("new_cuda_viewer:  initializing cuda viewer subsys");
		init_cuda_viewer_subsystem();
	}

	cvp = new_cuda_vwr(QSP_ARG  vp->vw_name);
	if( cvp == NO_CUDA_VWR ) return(cvp);

	cvp->cv_vp = vp;
	init_cuda_viewer(cvp);

	return(cvp);
}

COMMAND_FUNC( do_new_cuda_vwr )
{
	Cuda_Viewer *cvp;
	Viewer *vp;

	glew_check();	/* without this, we get a segmentation violation on glGenBuffers??? */

	vp = PICK_VWR("name of existing viewer to use with CUDA");
	if( vp == NO_VIEWER ) return;

	cvp = new_cuda_viewer(QSP_ARG  vp);
}

COMMAND_FUNC( do_load_cuda_vwr )
{
	Cuda_Viewer *cvp;
	Data_Obj *dp;

	cvp = PICK_CUDA_VWR("CUDA viewer");
	dp = PICK_OBJ("GL buffer object");

	if( cvp == NO_CUDA_VWR || dp == NO_OBJ ) return;

	select_gl_viewer( cvp->cv_vp );

	if( ! IS_GL_BUFFER(dp) ){
		sprintf(ERROR_STRING,"Object %s is not a GL buffer object.",dp->dt_name);
		WARN(ERROR_STRING);
		return;
	}

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		
	//setup_gl_image(cvp,dp);
	update_cuda_viewer(cvp,dp);
}

// Does the GL context have to be set when we do this??

COMMAND_FUNC( do_new_gl_buffer )
{
	const char *s;
	Data_Obj *dp;
	Dimension_Set ds;
	dimension_t d,w,h;
	cudaError_t e;
	int t;

	s = NAMEOF("name for GL buffer object");
	w = HOW_MANY("width");
	h = HOW_MANY("height");
	d = HOW_MANY("depth");

	/* what should the depth be??? default to 1 for now... */

	/* Make sure this name isn't already in use... */
	dp = dobj_of(QSP_ARG  s);
	if( dp != NO_OBJ ){
		sprintf(error_string,"Data object name '%s' is already in use, can't use for GL buffer object.",s);
		NWARN(error_string);
		return;
	}

	// BUG need to be able to set the cuda device.
	// Note, however, that we don't need GL buffers on the Tesla...
	set_data_area(cuda_data_area[0][0]);

	ds.ds_dimension[0]=d;
	ds.ds_dimension[1]=w;
	ds.ds_dimension[2]=h;
	ds.ds_dimension[3]=1;
	ds.ds_dimension[4]=1;
	dp = _make_dp(QSP_ARG  s,&ds,PREC_UBY);
	if( dp == NO_OBJ ){
		sprintf(error_string,
			"Error creating data_obj header for %s",s);
		ERROR1(error_string);
	}

	dp->dt_flags |= DT_NO_DATA;	/* can't free this data */
	dp->dt_flags |= DT_GL_BUF;	/* indicate obj is a GL buffer */

	dp->dt_data = NULL;
	dp->dt_gl_info_p = (GL_Info *) getbuf( sizeof(GL_Info) );

	glew_check();	/* without this, we get a segmentation
			 * violation on glGenBuffers???
			 */

	// We need an extra field in which to store the GL identifier...
	// AND another extra field in which to store the associated texid.

	glGenBuffers(1, BUF_ID_P(dp) );	// first arg is # buffers to generate?

//sprintf(error_string,"glGenBuffers gave us buf_id = %d",BUF_ID(dp));
//advise(error_string);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,  BUF_ID(dp) ); 

	// glBufferData will allocate the memory for the buffer,
	// but won't copy unless the pointer is non-null
	// How do we get the gpu memory space address?
	// That must be with map


	glBufferData(GL_PIXEL_UNPACK_BUFFER,
		dp->dt_comps * dp->dt_cols * dp->dt_rows, NULL, GL_STREAM_DRAW);  

	/* buffer arg set to 0 unbinds any previously bound buffers...
	 * and restores client memory usage.
	 */
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	/* how do we check for an error? */
	e = cudaGLRegisterBufferObject( BUF_ID(dp) );
	if( e != cudaSuccess ){
		describe_cuda_error2("do_new_gl_buffer",
				"cudaGLRegisterBufferObject",e);
	}

	glGenTextures(1, TEX_ID_P(dp) );
//sprintf(error_string,"glGenTextures gave us tex_id = %d",TEX_ID(dp));
//advise(error_string);
	glBindTexture(GL_TEXTURE_2D, TEX_ID(dp) );
	t = gl_pixel_type(dp);
	glTexImage2D(GL_TEXTURE_2D, 0, dp->dt_comps,
			dp->dt_cols, dp->dt_rows,  0, t,
			GL_UNSIGNED_BYTE,
			NULL	// null pointer means
				// - offset into PIXEL_UNPACK_BUFFER??
			);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glBindTexture(GL_TEXTURE_2D, 0);
	
	// Leave the buffer mapped by default
	//cutilSafeCall(cudaGLMapBufferObject( &dp->dt_data,  BUF_ID(dp) ));
sprintf(error_string,"Mapping buffer %s",dp->dt_name);
advise(error_string);
	e = cudaGLMapBufferObject( &dp->dt_data,  BUF_ID(dp) );
	if( e != cudaSuccess ){
		describe_cuda_error2("do_new_gl_buffer",
				"cudaGLMapBufferObject",e);
	}
	dp->dt_flags |= DT_BUF_MAPPED;
	// propagate change to children and parents
	propagate_flag(dp,DT_BUF_MAPPED);


	//cutilSafeCall(cudaGLUnmapBufferObject( BUF_ID(dp) ));   
	// Remember we have to map this object before using it for CUDA, and unmap it before using it for GL!!!
} /* end do_new_gl_buffer */

#endif /* HAVE_GLUT */
#endif /* HAVE_OPENGL */
#endif /* HAVE_CUDA */



