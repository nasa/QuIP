
#include "quip_config.h"

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

#ifdef HAVE_GL_GLEW_H
#include <GL/glew.h>
#endif

// used to include GL/glut.h and rendercheck_gl.h...
//#include <sys/types.h>

#define NO_OGL_MSG	WARN("Sorry, no openGL support in this build!?");

#ifdef HAVE_CUDA
#define BUILD_FOR_CUDA
#include <cuda_runtime.h>
#include <curand.h>


//#ifdef OLD_CUDA4
//#include <cutil_inline.h>
//#include <cutil_gl_inline.h>
//#include <cutil_gl_error.h>
//#else
#include "GL/glext.h"	// from sample code
//#endif

#include <cuda_gl_interop.h>
#include <vector_types.h>
#endif // HAVE_CUDA

#include "quip_prot.h"

#include "glx_hack.h"

#ifdef FOOBAR
// moved to glx_hack.h

#if !defined(__STDC_VERSION__)
// jbm:  a total hack, to make the nvidia glx.h compile
// This didn't used to throw an error, what happened???
#define __STDC_VERSION__ 199901L
#endif

#ifdef HAVE_GL_GLX_H
#include <GL/glx.h>	// jbm added for glXSwapBuffers()
#endif
#endif // FOOBAR

#include "my_cuda.h"
#include "cuda_supp.h"
#include "cuda_viewer.h"
#include "gl_viewer.h"		/* select_gl_viewer() */
#include "gl_info.h"

//ITEM_INTERFACE_DECLARATIONS_STATIC( Cuda_Viewer, cuda_vwr )
static Item_Type *cuda_vwr_itp=NULL;
static ITEM_INIT_FUNC(Cuda_Viewer,cuda_vwr)
static ITEM_NEW_FUNC(Cuda_Viewer,cuda_vwr)
static ITEM_PICK_FUNC(Cuda_Viewer,cuda_vwr)

#define PICK_CUDA_VWR(p)	pick_cuda_vwr(QSP_ARG  p)

#ifdef HAVE_CUDA

#ifdef FOOBAR

// moved to libdata

static void propagate_up(Data_Obj *dp,uint32_t flagbit)
{
	if( dp->dt_parent != NULL ){
		xfer_cuda_flag(dp->dt_parent,dp,flagbit);
		propagate_up(dp->dt_parent,flagbit);
	}
}

static void propagate_down(Data_Obj *dp,uint32_t flagbit)
{
	Node *np;

	if( dp->dt_children != NULL ){
		np=dp->dt_children->l_head;
		while(np!=NULL){
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
#endif // FOOBAR

static int gl_pixel_type(Data_Obj *dp)
{
	int t;

	switch(OBJ_COMPS(dp)){
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
#endif // HAVE_CUDA

static void init_cuda_viewer(Cuda_Viewer *cvp)
{
	cvp->cv_pbo_buffer = 0;
	cvp->cv_texid = 0;
}

#ifdef HAVE_CUDA

static void prepare_image_for_mapping(Data_Obj *dp)
{
#ifdef HAVE_OPENGL
	int t;
	cudaError_t e;

	// unmap buffer before using w/ GL
	if( BUF_IS_MAPPED(dp) ){
		e = cudaGLUnmapBufferObject( OBJ_BUF_ID(dp) );   
		if( e != cudaSuccess ){
			describe_cuda_driver_error2("update_cuda_viewer",
				"cudaGLUnmapBufferObject",e);
			NERROR1("failed to unmap buffer object");
		}
		CLEAR_OBJ_FLAG_BITS(dp, DT_BUF_MAPPED);
		// propagate change to children and parents
		propagate_flag(dp,DT_BUF_MAPPED);

	}


	//
	//bind_texture(OBJ_DATA_PTR(dp));

	glClear(GL_COLOR_BUFFER_BIT);

/*
sprintf(ERROR_STRING,"update_cuda_viewer:  tex_id = %d, buf_id = %d",
OBJ_TEX_ID(dp),OBJ_BUF_ID(dp));
advise(ERROR_STRING);
*/
	glBindTexture(GL_TEXTURE_2D, OBJ_TEX_ID(dp));
#ifdef HAVE_LIBGLEW
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, OBJ_BUF_ID(dp));
#endif // HAVE_LIBGLEW

#ifdef FOOBAR
	switch(OBJ_COMPS(dp)){
		/* what used to be here??? */
	}
#endif /* FOOBAR */

	t=gl_pixel_type(dp);
	glTexSubImage2D(GL_TEXTURE_2D, 0,			// target, level
		0, 0,						// x0, y0
		OBJ_COLS(dp), OBJ_ROWS(dp), 			// dx, dy
		t,
		GL_UNSIGNED_BYTE,				// type
		OFFSET(0));					// offset into PIXEL_UNPACK_BUFFER

#ifdef HAVE_LIBGLEW
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
#endif // HAVE_LIBGLEW
}

static void cuda_display_finish(QSP_ARG_DECL  Data_Obj *dp)
{
	cudaError_t e;

	// re-map so we can use again with CUDA
	// BUG?  Is it safe to do this before the call to swap_buffers???
	//cutilSafeCall(cudaGLMapBufferObject( &OBJ_DATA_PTR(dp),  OBJ_BUF_ID(dp) ));

	e = cudaGLMapBufferObject( &OBJ_DATA_PTR(dp),  OBJ_BUF_ID(dp) );

	if( e != cudaSuccess ){
		WARN("Error mapping buffer object!?");
		// should we return now, with possibly other cleanup???
	}

	SET_OBJ_FLAG_BITS(dp, DT_BUF_MAPPED);
	// propagate change to children and parents
	propagate_flag(dp,DT_BUF_MAPPED);
}

// This is the normal display path
static void update_cuda_viewer(QSP_ARG_DECL  Cuda_Viewer *cvp, Data_Obj *dp) 
{
	prepare_image_for_mapping(dp);

	glBegin(GL_QUADS);
	glTexCoord2f(0, 1); glVertex2f(-1.0, -1.0);
	glTexCoord2f(0, 0); glVertex2f(-1.0, 1.0);
	glTexCoord2f(1, 0); glVertex2f(1.0, 1.0);
	glTexCoord2f(1, 1); glVertex2f(1.0, -1.0);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);

#ifdef FOOBAR
	//glutSwapBuffers();
	//glutPostRedisplay();
	// Maybe we want to call swap buffers ourselves,
	// if we are trying to synchronize the display
	// and are updating multiple windows?
	//glXSwapBuffers(cvp->cv_vp->vw_dpy,cvp->cv_vp->vw_xwin);
#endif // FOOBAR


	cuda_display_finish(QSP_ARG  dp);
#else // ! HAVE_OPENGL
	NO_OGL_MSG
#endif // ! HAVE_OPENGL
}

// This function allows us to do a different mapping of the image...
static void map_cuda_viewer(QSP_ARG_DECL  Cuda_Viewer *cvp,
				Data_Obj *img_dp, Data_Obj *coord_dp) 
{
	if( OBJ_PREC(coord_dp) != PREC_SP ){
		sprintf(ERROR_STRING,
	"map_cuda_viewer:  coord object %s must have %s precision!?",
			OBJ_NAME(coord_dp),PREC_NAME(OBJ_PREC_PTR(coord_dp)));
		WARN(ERROR_STRING);
		return;
	}
	if( ! IS_CONTIGUOUS(coord_dp) ){
		sprintf(ERROR_STRING,
	"map_cuda_viewer:  coord object %s must be contiguous!?",
			OBJ_NAME(coord_dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_COMPS(coord_dp) != 2 ){
		sprintf(ERROR_STRING,
	"map_cuda_viewer:  coord object %s must have 2 components!?",
			OBJ_NAME(coord_dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_COLS(coord_dp) != 2 ){
		sprintf(ERROR_STRING,
	"map_cuda_viewer:  coord object %s must have 2 columns!?",
			OBJ_NAME(coord_dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_ROWS(coord_dp) != 2 ){
		sprintf(ERROR_STRING,
	"map_cuda_viewer:  coord object %s must have 2 rows!?",
			OBJ_NAME(coord_dp));
		WARN(ERROR_STRING);
		return;
	}

#ifdef HAVE_OPENGL
	float *f;

	prepare_image_for_mapping(img_dp);

	f=(float *)OBJ_DATA_PTR(coord_dp);

	glBegin(GL_QUADS);
fprintf(stderr,"first vertex at %f, %f (normally -1, -1)\n",f[0],f[1]);
	glTexCoord2f(0, 1); glVertex2f( f[0], f[1] );	// -1, -1
fprintf(stderr,"second vertex at %f, %f (normally -1,  1)\n",f[4],f[5]);
	glTexCoord2f(0, 0); glVertex2f( f[4], f[5] );	// -1,  1
fprintf(stderr,"third vertex at %f, %f (normally  1,  1)\n",f[6],f[7]);
	glTexCoord2f(1, 0); glVertex2f( f[6], f[7] );	//  1,  1
fprintf(stderr,"fourth vertex at %f, %f (normally  1, -1)\n",f[2],f[3]);
	glTexCoord2f(1, 1); glVertex2f( f[2], f[3] );	//  1, -1
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);

	cuda_display_finish(QSP_ARG  img_dp);
#else // ! HAVE_OPENGL
	NO_OGL_MSG
#endif // ! HAVE_OPENGL
}
#endif // HAVE_CUDA

#ifdef FOOBAR
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
#endif /* FOOBAR */


static int cuda_viewer_subsystem_inited=0;

static void init_cuda_viewer_subsystem(void)
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

static void glew_check()
{
#ifdef HAVE_LIBGLEW
	static int glew_checked=0;

	if( glew_checked ){
		if( verbose )
			NADVISE("glew_check:  glew already checked.");
		return;
	}

	// BUG glewInit will core dump if GL is not already initialized!?
	// We try to fix this by making sure that the cuda viewer is already
	// specified for GL before calling this...

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
#else // ! HAVE_LIBGLEW
NERROR1("glew_check:  libglew not present!?.");
#endif // ! HAVE_LIBGLEW
}

static Cuda_Viewer * new_cuda_viewer(QSP_ARG_DECL  Viewer *vp)
{
	Cuda_Viewer *cvp;

	if( !cuda_viewer_subsystem_inited ){
		if( verbose )
	advise("new_cuda_viewer:  initializing cuda viewer subsys");
		init_cuda_viewer_subsystem();
	}

	cvp = new_cuda_vwr(QSP_ARG  vp->vw_name);
	if( cvp == NULL ) return(cvp);

	cvp->cv_vp = vp;
	init_cuda_viewer(cvp);

	return(cvp);
}

COMMAND_FUNC( do_new_cuda_vwr )
{
	Viewer *vp;


	vp = PICK_VWR("name of existing viewer to use with CUDA");
	if( vp == NULL ) return;

	if( ! READY_FOR_GLX(vp) ) {
		sprintf(ERROR_STRING,"Existing viewer %s must be initialized for GL before using with CUDA!?",VW_NAME(vp) );
		WARN(ERROR_STRING);
		return;
	}

	glew_check();	/* without this, we get a segmentation violation on glGenBuffers??? */

	if( new_cuda_viewer(QSP_ARG  vp) == NULL ){
		sprintf(ERROR_STRING,"Error making %s a cuda viewer!?",VW_NAME(vp));
		WARN(ERROR_STRING);
	}
}

static int image_mapping_checks(QSP_ARG_DECL  Cuda_Viewer *cvp, Data_Obj *dp)
{
	select_gl_viewer( QSP_ARG  cvp->cv_vp );

	if( ! IS_GL_BUFFER(dp) ){
		sprintf(ERROR_STRING,"Object %s is not a GL buffer object.",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return -1;
	}

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

	return 0;
}

COMMAND_FUNC( do_load_cuda_vwr )
{
	Cuda_Viewer *cvp;
	Data_Obj *dp;

	cvp = PICK_CUDA_VWR("CUDA viewer");
	dp = PICK_OBJ("GL buffer object");

	if( cvp == NULL || dp == NULL ) return;

	if( image_mapping_checks(QSP_ARG  cvp, dp ) < 0 ){
		WARN("Image mapping checks failed!?");
		return;
	}

		
#ifdef HAVE_CUDA
	//setup_gl_image(cvp,dp);
	update_cuda_viewer(QSP_ARG  cvp,dp);
#endif // HAVE_CUDA
}

COMMAND_FUNC( do_map_cuda_vwr )
{
	Cuda_Viewer *cvp;
	Data_Obj *img_dp;
	Data_Obj *coord_dp;

	cvp = PICK_CUDA_VWR("CUDA viewer");
	img_dp = PICK_OBJ("GL buffer object");
	coord_dp = PICK_OBJ("corner coordinate object");

	if( cvp == NULL || img_dp == NULL || coord_dp == NULL ){
fprintf(stderr,"do_map_cuda_vwr aborting...\n");
		return;
	}

	if( image_mapping_checks(QSP_ARG  cvp, img_dp ) < 0 ){
fprintf(stderr,"do_map_cuda_vwr: aborting (#2)...\n");
		return;
	}

		
#ifdef HAVE_CUDA
	//setup_gl_image(cvp,dp);
	map_cuda_viewer(QSP_ARG  cvp,img_dp,coord_dp);
#endif // HAVE_CUDA
}

// Does the GL context have to be set when we do this??

COMMAND_FUNC( do_new_gl_buffer )
{
	const char *s;
	Data_Obj *dp;
	Platform_Device *pdp;
	dimension_t d,w,h;
#ifdef HAVE_OPENGL
#ifdef HAVE_CUDA
	Dimension_Set ds;
	cudaError_t e;
	int t;
#endif // HAVE_CUDA
#endif // HAVE_OPENGL

	s = NAMEOF("name for GL buffer object");
	pdp = PICK_PFDEV("device");
	w = HOW_MANY("width");
	h = HOW_MANY("height");
	d = HOW_MANY("depth");

	/* what should the depth be??? default to 1 for now... */

	if( pdp == NULL ) return;

	/* Make sure this name isn't already in use... */
	dp = dobj_of(QSP_ARG  s);
	if( dp != NULL ){
		sprintf(ERROR_STRING,"Data object name '%s' is already in use, can't use for GL buffer object.",s);
		NWARN(ERROR_STRING);
		return;
	}

#ifdef HAVE_OPENGL
#ifdef HAVE_CUDA
	// BUG need to be able to set the cuda device.
	// Note, however, that we don't need GL buffers on the Tesla...
	//set_data_area(cuda_data_area[0][0]);
	set_data_area( PFDEV_AREA(pdp,PFDEV_GLOBAL_AREA_INDEX) );

	ds.ds_dimension[0]=d;
	ds.ds_dimension[1]=w;
	ds.ds_dimension[2]=h;
	ds.ds_dimension[3]=1;
	ds.ds_dimension[4]=1;
	dp = _make_dp(QSP_ARG  s,&ds,PREC_FOR_CODE(PREC_UBY));
	if( dp == NULL ){
		sprintf(ERROR_STRING,
			"Error creating data_obj header for %s",s);
		ERROR1(ERROR_STRING);
	}

	SET_OBJ_FLAG_BITS(dp, DT_NO_DATA);	/* can't free this data */
	SET_OBJ_FLAG_BITS(dp, DT_GL_BUF);	/* indicate obj is a GL buffer */

	SET_OBJ_DATA_PTR(dp, NULL);
	SET_OBJ_GL_INFO(dp, (GL_Info *) getbuf( sizeof(GL_Info) ) );

	glew_check();	/* without this, we get a segmentation
			 * violation on glGenBuffers???
			 */

	// We need an extra field in which to store the GL identifier...
	// AND another extra field in which to store the associated texid.

#ifdef HAVE_LIBGLEW
	glGenBuffers(1, OBJ_BUF_ID_P(dp) );	// first arg is # buffers to generate?

//sprintf(ERROR_STRING,"glGenBuffers gave us buf_id = %d",OBJ_BUF_ID(dp));
//advise(ERROR_STRING);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,  OBJ_BUF_ID(dp) ); 

	// glBufferData will allocate the memory for the buffer,
	// but won't copy unless the pointer is non-null
	// How do we get the gpu memory space address?
	// That must be with map


	glBufferData(GL_PIXEL_UNPACK_BUFFER,
		OBJ_COMPS(dp) * OBJ_COLS(dp) * OBJ_ROWS(dp), NULL, GL_STREAM_DRAW);  

	/* buffer arg set to 0 unbinds any previously bound buffers...
	 * and restores client memory usage.
	 */
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
#endif // HAVE_LIBGLEW

	/* how do we check for an error? */
	e = cudaGLRegisterBufferObject( OBJ_BUF_ID(dp) );
	if( e != cudaSuccess ){
		describe_cuda_driver_error2("do_new_gl_buffer",
				"cudaGLRegisterBufferObject",e);
	}

	glGenTextures(1, OBJ_TEX_ID_P(dp) );
//sprintf(ERROR_STRING,"glGenTextures gave us tex_id = %d",OBJ_TEX_ID(dp));
//advise(ERROR_STRING);
	glBindTexture(GL_TEXTURE_2D, OBJ_TEX_ID(dp) );
	t = gl_pixel_type(dp);
	glTexImage2D(GL_TEXTURE_2D, 0, OBJ_COMPS(dp),
			OBJ_COLS(dp), OBJ_ROWS(dp),  0, t,
			GL_UNSIGNED_BYTE,
			NULL	// null pointer means
				// - offset into PIXEL_UNPACK_BUFFER??
			);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glBindTexture(GL_TEXTURE_2D, 0);
	
	// Leave the buffer mapped by default
	//cutilSafeCall(cudaGLMapBufferObject( &OBJ_DATA_PTR(dp),  OBJ_BUF_ID(dp) ));
sprintf(ERROR_STRING,"Mapping buffer %s",OBJ_NAME(dp));
advise(ERROR_STRING);
	e = cudaGLMapBufferObject( &OBJ_DATA_PTR(dp),  OBJ_BUF_ID(dp) );
	if( e != cudaSuccess ){
		describe_cuda_driver_error2("do_new_gl_buffer",
				"cudaGLMapBufferObject",e);
	}
	SET_OBJ_FLAG_BITS(dp, DT_BUF_MAPPED);
	// propagate change to children and parents
	propagate_flag(dp,DT_BUF_MAPPED);


	//cutilSafeCall(cudaGLUnmapBufferObject( OBJ_BUF_ID(dp) ));   
	// Remember we have to map this object before using it for CUDA, and unmap it before using it for GL!!!
#else // ! HAVE_CUDA
	NO_CUDA_MSG(new_gl_buffer)
#endif // ! HAVE_CUDA
#else // ! HAVE_OPENGL
	NO_OGL_MSG
#endif // ! HAVE_OPENGL
} /* end do_new_gl_buffer */

