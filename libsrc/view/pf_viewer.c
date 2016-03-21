// Based on old cuda_viewer.cpp, this file provides
// a platform-independent linkage between gpu and openGL

// This file doesn't really belong in libdata, but as that
// is where the platform menu is, we leave it here for now

// OpenGL appears not to have access to GLEW...
// So the port from Cuda may not be so simple
// as implementing REGBUF_FN etc...

#include "quip_config.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//#ifdef HAVE_GL_GL_H
//#include <GL/gl.h>
//#endif
//#ifdef HAVE_GL_GLEXT_H
//#define GL_GLEXT_PROTOTYPES	// glBindBuffer
//#include <GL/glext.h>
//#endif

#ifdef HAVE_GL_GLEW_H
#include <GL/glew.h>
#endif

// used to include GL/glut.h and rendercheck_gl.h...

#ifdef HAVE_GL_GLX_H
#include <GL/glx.h>	// jbm added for glXSwapBuffers()
#endif

#define NO_OGL_MSG	WARN("Sorry, no openGL support in this build!?");

#include "quip_prot.h"
#include "gl_viewer.h"		/* select_gl_viewer() */
#include "gl_info.h"

#include "platform.h"
#include "pf_viewer.h"
#include "opengl_utils.h"

static Item_Type *pf_vwr_itp=NULL;
static ITEM_INIT_FUNC(Platform_Viewer,pf_vwr)
static ITEM_NEW_FUNC(Platform_Viewer,pf_vwr)
static ITEM_PICK_FUNC(Platform_Viewer,pf_vwr)

#define PICK_PF_VWR(p)	pick_pf_vwr(QSP_ARG  p)

static void init_pf_viewer(Platform_Viewer *pvp)
{
#ifdef HAVE_OPENGL
	pvp->pv_pbo_buffer = 0;
	pvp->pv_texid = 0;
#endif // HAVE_OPENGL
}

// This is the normal display path
static void update_pf_viewer(QSP_ARG_DECL  Platform_Viewer *pvp, Data_Obj *dp) 
{
#ifdef HAVE_OPENGL
	int t;
	//cudaError_t e;

	// unmap buffer before using w/ GL
	if( BUF_IS_MAPPED(dp) ){
		if( (*PF_UNMAPBUF_FN(PFDEV_PLATFORM(OBJ_PFDEV(dp))))
				(QSP_ARG  dp) < 0 ) {
			WARN("update_pf_viewer:  buffer unmap error!?");
		}
#ifdef FOOBAR
		e = cudaGLUnmapBufferObject( OBJ_BUF_ID(dp) );   
		if( e != cudaSuccess ){
			describe_cuda_driver_error2("update_pf_viewer",
				"cudaGLUnmapBufferObject",e);
			NERROR1("failed to unmap buffer object");
		}
#endif // FOOBAR
		CLEAR_OBJ_FLAG_BITS(dp, DT_BUF_MAPPED);
		// propagate change to children and parents
		propagate_flag(dp,DT_BUF_MAPPED);

	}

	//
	//bind_texture(OBJ_DATA_PTR(dp));

	glClear(GL_COLOR_BUFFER_BIT);

/*
sprintf(ERROR_STRING,"update_pf_viewer:  tex_id = %d, buf_id = %d",
OBJ_TEX_ID(dp),OBJ_BUF_ID(dp));
advise(ERROR_STRING);
*/
	glBindTexture(GL_TEXTURE_2D, OBJ_TEX_ID(dp));
	// is glBindBuffer REALLY part of libGLEW???
//#ifdef HAVE_LIBGLEW
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, OBJ_BUF_ID(dp));
//#endif // HAVE_LIBGLEW

#ifdef FOOBAR
	switch(OBJ_COMPS(dp)){
		/* what used to be here??? */
	}
#endif /* FOOBAR */

	t=gl_pixel_type(dp);
	glTexSubImage2D(GL_TEXTURE_2D, 0,	// target, level
		0, 0,				// x0, y0
		OBJ_COLS(dp), OBJ_ROWS(dp), 	// dx, dy
		t,
		GL_UNSIGNED_BYTE,		// type
		OFFSET(0));			// offset into PIXEL_UNPACK_BUFFER

//#ifdef HAVE_LIBGLEW
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
//#endif // HAVE_LIBGLEW

	glBegin(GL_QUADS);
	glTexCoord2f(0, 1); glVertex2f(-1.0, -1.0);
	glTexCoord2f(0, 0); glVertex2f(-1.0, 1.0);
	glTexCoord2f(1, 0); glVertex2f(1.0, 1.0);
	glTexCoord2f(1, 1); glVertex2f(1.0, -1.0);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);

#ifdef FOOBAR
	e = cudaGLMapBufferObject( &OBJ_DATA_PTR(dp),  OBJ_BUF_ID(dp) );
	if( e != cudaSuccess ){
		WARN("Error mapping buffer object!?");
		// should we return now, with possibly other cleanup???
	}
#endif // FOOBAR
	if( (*PF_MAPBUF_FN(PFDEV_PLATFORM(OBJ_PFDEV(dp))))(QSP_ARG  dp) < 0 ){
		WARN("update_pf_viewer:  Error mapping buffer!?");
	}


	SET_OBJ_FLAG_BITS(dp, DT_BUF_MAPPED);
	// propagate change to children and parents
	propagate_flag(dp,DT_BUF_MAPPED);
#else // ! HAVE_OPENGL
	NO_OGL_MSG
#endif // ! HAVE_OPENGL
}

static int pf_viewer_subsystem_inited=0;

static void init_pf_viewer_subsystem(void)
{
	const char *pn;
	int n;

	pn = tell_progname();

	if( pf_viewer_subsystem_inited ){
		NWARN("Platform viewer subsystem already initialized!?");
		return;
	}

	// First initialize OpenGL context, so we can properly set
	// the GL for CUDA.
	// This is necessary in order to achieve optimal performance
	// with OpenGL/CUDA interop.

	//glutInit( &argc, argv);		/* BUG?  where should this be done? */
	n=1;
#ifdef HAVE_GLUT
	glutInit( &n, (char **)&pn);		/* BUG?  where should this be done? */

	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
#endif // HAVE_GLUT

	pf_viewer_subsystem_inited=1;
}

static Platform_Viewer * new_pf_viewer(QSP_ARG_DECL  Viewer *vp)
{
	Platform_Viewer *pvp;

	if( !pf_viewer_subsystem_inited ){
		if( verbose )
	advise("new_pf_viewer:  initializing cuda viewer subsys");
		init_pf_viewer_subsystem();
	}

	pvp = new_pf_vwr(QSP_ARG  VW_NAME(vp));
	if( pvp == NO_PF_VWR ) return(pvp);

#ifndef BUILD_FOR_OBJC
	pvp->pv_vp = vp;
#endif // BUILD_FOR_OBJC
	
	init_pf_viewer(pvp);

	return(pvp);
}

COMMAND_FUNC( do_new_pf_vwr )
{
	Viewer *vp;


	vp = PICK_VWR("name of existing viewer to use with Cuda/OpenCL");
	if( vp == NO_VIEWER ) return;

	if( ! READY_FOR_GLX(vp) ) {
		sprintf(ERROR_STRING,"do_new_pf_vwr:  Existing viewer %s must be initialized for GL before using!?",VW_NAME(vp) );
		WARN(ERROR_STRING);
		return;
	}

#ifdef HAVE_OPENGL
	glew_check(SINGLE_QSP_ARG);	/* without this, we get a segmentation violation on glGenBuffers??? */
#endif // HAVE_OPENGL

	if( new_pf_viewer(QSP_ARG  vp) == NULL ){
		sprintf(ERROR_STRING,"Error making %s a cuda viewer!?",VW_NAME(vp));
		WARN(ERROR_STRING);
	}
} // do_new_pf_viewer

#ifndef BUILD_FOR_IOS
COMMAND_FUNC( do_load_pf_vwr )
{
	Platform_Viewer *pvp;
	Data_Obj *dp;

	pvp = PICK_PF_VWR("platform viewer");
	dp = PICK_OBJ("GL buffer object");

	if( pvp == NO_PF_VWR || dp == NO_OBJ ) return;

#ifdef HAVE_OPENGL
	select_gl_viewer( QSP_ARG  pvp->pv_vp );

	if( ! IS_GL_BUFFER(dp) ){
		sprintf(ERROR_STRING,"Object %s is not a GL buffer object.",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}

	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
#else // ! HAVE_OPENGL
	WARN("do_load_pf_viewer:  Sorry, no OpenGL support in this build!?");
#endif // ! HAVE_OPENGL
		
	update_pf_viewer(QSP_ARG  pvp,dp);
}
#endif // BUILD_FOR_IOS

#ifdef FOOBAR
// moved to glmenu.c in opengl...


// Does the GL context have to be set when we do this??

COMMAND_FUNC( do_new_gl_buffer )
{
	const char *s;
	Data_Obj *dp;
	Platform_Device *pdp;
	Compute_Platform *cdp;
	dimension_t d,w,h;
#ifdef HAVE_OPENGL
	Dimension_Set ds;
	int t;
#endif // HAVE_OPENGL

	s = NAMEOF("name for GL buffer object");
	cdp = PICK_PLATFORM("platform");
	if( cdp != NO_PLATFORM )
		push_pfdev_context(QSP_ARG  PF_CONTEXT(cdp) );
	pdp = PICK_PFDEV("device");
	if( cdp != NO_PLATFORM )
		pop_pfdev_context(SINGLE_QSP_ARG);

	w = HOW_MANY("width");
	h = HOW_MANY("height");
	d = HOW_MANY("depth");

	/* what should the depth be??? default to 1 for now... */

	if( pdp == NO_PFDEV ) return;

	/* Make sure this name isn't already in use... */
	dp = dobj_of(QSP_ARG  s);
	if( dp != NO_OBJ ){
		sprintf(ERROR_STRING,"Data object name '%s' is already in use, can't use for GL buffer object.",s);
		NWARN(ERROR_STRING);
		return;
	}

#ifdef HAVE_OPENGL
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
	if( dp == NO_OBJ ){
		sprintf(ERROR_STRING,
			"Error creating data_obj header for %s",s);
		ERROR1(ERROR_STRING);
	}

	SET_OBJ_FLAG_BITS(dp, DT_NO_DATA);	/* can't free this data */
	SET_OBJ_FLAG_BITS(dp, DT_GL_BUF);	/* indicate obj is a GL buffer */

	SET_OBJ_DATA_PTR(dp, NULL);
	SET_OBJ_GL_INFO(dp, (GL_Info *) getbuf( sizeof(GL_Info) ) );

	glew_check(SINGLE_QSP_ARG);	/* without this, we get a segmentation
			 * violation on glGenBuffers???
			 */

	// We need an extra field in which to store the GL identifier...
	// AND another extra field in which to store the associated texid.

// Why is this ifdef here?  These don't seem to depend
// on libglew???
// Answer:  We need libglew to bring in openGL extensions like glBindBuffer...
advise("calling glGenBuffers");
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
//#endif // HAVE_LIBGLEW

	glGenTextures(1, OBJ_TEX_ID_P(dp) );		// makes a texture name
	glBindTexture(GL_TEXTURE_2D, OBJ_TEX_ID(dp) );
	t = gl_pixel_type(dp);
	glTexImage2D(	GL_TEXTURE_2D,
			0,			// level-of-detail - is this the same as miplevel???
			OBJ_COMPS(dp),		// internal format, can also be symbolic constant such as
						// GL_RGBA etc
			OBJ_COLS(dp),		// width - must be 2^n+2 (border) for some n???
			OBJ_ROWS(dp),		// height - must be 2^m+2 (border) for some m???
			0,			// border - must be 0 or 1
			t,			// format of pixel data
			GL_UNSIGNED_BYTE,	// type of pixel data
			NULL			// pixel data - null pointer means
						// allocate but do not copy?
						// - offset into PIXEL_UNPACK_BUFFER??
			);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	// Why was this here?  It would seem to un-bind the target???
	glBindTexture(GL_TEXTURE_2D, 0);
	
	//glFinish();	// necessary or not?

advise("calling platform-specific buffer registration function");
	if( (*PF_REGBUF_FN(PFDEV_PLATFORM(pdp)))( QSP_ARG  dp ) < 0 ){
		WARN("do_new_gl_buffer:  Error in platform-specific buffer registration!?");
		// BUG? - should clean up here!
	}

	// Leave the buffer mapped by default
	//cutilSafeCall(cudaGLMapBufferObject( &OBJ_DATA_PTR(dp),  OBJ_BUF_ID(dp) ));
advise("calling platform-specific buffer mapping function");
	if( (*PF_MAPBUF_FN(PFDEV_PLATFORM(pdp)))( QSP_ARG  dp ) < 0 ){
		WARN("do_new_gl_buffer:  Error in platform-specific buffer mapping!?");
		// BUG? - should clean up here!
	}

	SET_OBJ_FLAG_BITS(dp, DT_BUF_MAPPED);
	// propagate change to children and parents
	propagate_flag(dp,DT_BUF_MAPPED);

#else // ! HAVE_OPENGL
	NO_OGL_MSG
#endif // ! HAVE_OPENGL
} /* end do_new_gl_buffer */


#endif // FOOBAR
