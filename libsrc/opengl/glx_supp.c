#include "quip_config.h"

char VersionId_opengl_glx_supp[] = QUIP_VERSION_STRING;

#ifdef HAVE_OPENGL

#define GLX_GLXEXT_PROTOTYPES

#ifdef HAVE_GL_GLX_H
#include <GL/glx.h>
#endif


#include "viewer.h"
#include "gl_viewer.h"
#include "debug.h"

static Viewer *gl_vp=NO_VIEWER;

//static GLXContext the_ctx;
//static GLXContext first_ctx=NULL;

void swap_buffers(void)
{
	if( gl_vp == NO_VIEWER ){
		NWARN("swap_buffers:  no viewer selected");
		return;
	}
	// This should wait for retrace
	// IF __GL_SYNC_TO_VBLANK is set to 1!?

	//glutSwapBuffers();
	glXSwapBuffers(gl_vp->vw_dpy,gl_vp->vw_xwin);
}

void wait_video_sync(int n)
{
#ifdef HAVE_VIDEOSYNCSGI
	unsigned int count;
	int divisor, remainder;

	if( gl_vp == NO_VIEWER ){
		NWARN("wait_video_sync:  no viewer selected");
		return;
	}

	glXGetVideoSyncSGI(&count);
/*
sprintf(ERROR_STRING,"Calling glXWaitVideoSyncSGI(%d (0x%x), %d (0x%x), %d (0x%x))",
divisor,divisor,remainder,remainder,count,count);
advise(ERROR_STRING);
*/
	// We used to use a big number for the divisor.
	// Sometimes the call would hang when we were waiting one frame.
	// What was happening was that when we had some other
	// variable timing events, sometimes the above call to
	// glXGetVideoSyncSGI would happen just before the count ticked over...
	// Then when we asked it to wait, it was already there, and
	// it was going to wait until it goes all the way around again!?
	//
	// The solution we adopt is to use the smallest divisor
	// possible - that way, if we do miss the event, there is a glitch
	// but at least no hang.  For example, if we want to wait one
	// frame, then we use a divisor of 2.  Say the count is 1 when
	// we read, we want to wait until 2 (0 mod 2), it ticks over before
	// we make the call so we wait until 4 instead.  Not perfect,
	// but better than hanging forever!?

	divisor = n+1;
	remainder = (count+n) % divisor;

	glXWaitVideoSyncSGI(divisor,remainder,&count);

#else /* ! HAVE_VIDEOSYNCSGI */
	static int warned=0;

	if( ! warned ){
		NWARN("wait_video_sync:  glXWaitVideoSyncSGI is not present!?");
		warned=1;
	}
#endif /* ! HAVE_VIDEOSYNCSGI */
}

static void init_glx_context(Viewer *vp)
{
	XVisualInfo vinfo_template, *vis_info_p;
	long mask=0;
	int n;
	int list[20];
	int index;

//sprintf(ERROR_STRING,"width: %d height: %d\n", vp->vw_width, vp->vw_height);
//advise(ERROR_STRING);
	vinfo_template.visual = vp->vw_visual;
	vinfo_template.depth = vp->vw_depth;
	vinfo_template.screen = vp->vw_screen_no;
	mask |= VisualScreenMask | VisualDepthMask;

	/* We don't seem to get a depth buffer... */
#define GLX_FLAG(bool)		list[index++]=bool;
#define GLX_PARAM(flag,val)		list[index++]=(flag); list[index++]=val;

	index=0;
	GLX_PARAM(GLX_DEPTH_SIZE,/*24 */ 16 )
	GLX_PARAM(GLX_LEVEL,0)
	GLX_FLAG(GLX_RGBA)
	GLX_FLAG(GLX_DOUBLEBUFFER)
	list[index++]=None;

	vis_info_p = glXChooseVisual(vp->vw_dpy,vp->vw_screen_no,list);

	if( vis_info_p == NULL ){
		sprintf(DEFAULT_ERROR_STRING,"unable to get a visual with DEPTH_SIZE %d",list[1]);
		NWARN(DEFAULT_ERROR_STRING);
		vis_info_p = XGetVisualInfo(vp->vw_dpy,mask,&vinfo_template,&n);
		if( vis_info_p == NULL )
			NERROR1("XGetVisualInfo failed!?");
	} else {
//		advise("FOUND visual with desired DEPTH_SIZE");
//sprintf(DEFAULT_ERROR_STRING,"visual id = %ld (0x%lx)",vis_info_p->visualid,vis_info_p->visualid);
//advise(DEFAULT_ERROR_STRING);
	}

//sprintf(DEFAULT_ERROR_STRING,"init_glx_context:  visual id = %ld (0x%lx)",vis_info_p->visualid,vis_info_p->visualid);
//advise(DEFAULT_ERROR_STRING);


	/* Old comment when all windows used the same context:
	 * This is a hack - we tried having each window create its
	 * own context, but that failed after the first 3 windows???
	 * Perhaps there is an undocumented limit???
	 * For now, we create one context and share it with all
	 * windows (assuming they're all on the same display!? BUG)
	 *
	 * This caused problems when trying to display images (from cuda)
	 * into windows with different sizes - the same size got used
	 * for both.
	 *
	 * Make a share list for all viewers on the same display.
	 */

	if( vp->vw_ctx == NULL ){
		vp->vw_ctx = glXCreateContext(vp->vw_dpy,vis_info_p,
			vp->vw_dop->do_ctx,	/* list of shared contexts? */
			True);
		if( vp->vw_ctx== NULL ){
			sprintf(DEFAULT_ERROR_STRING,
		"init_glx_context( %s ): glXCreateContext failed!?",
				vp->vw_name);
			NWARN(DEFAULT_ERROR_STRING);
		} else {
			if( verbose ){
				sprintf(DEFAULT_ERROR_STRING,
		"init_glx_context( %s ):  created GL context 0x%lx",
			vp->vw_name,(int_for_addr)vp->vw_ctx);
				advise(DEFAULT_ERROR_STRING);
			}

			if( vp->vw_dop->do_ctx == NULL )
				vp->vw_dop->do_ctx = vp->vw_ctx;

#ifdef FOOBAR
			/* now see if there is a z buffer? */
			/* Why? */

			{
			int val;

			if( glXGetConfig(vp->vw_dpy,vis_info_p,
				GLX_DEPTH_SIZE,&val) == 0 ){

				sprintf(DEFAULT_ERROR_STRING,
			"init_glx_context:  DEPTH_SIZE value is %d",val);
				advise(DEFAULT_ERROR_STRING);
			} else {
				NWARN("glXGetConfig error");
			}
			}
#endif /* FOOBAR */

		}
	}
}

COMMAND_FUNC( do_render_to )
{
	Viewer *vp;

	vp = PICK_VWR("");
	if( vp == NO_VIEWER ) return;

	select_gl_viewer(vp);
}

void select_gl_viewer(Viewer *vp)
{
	if( ! READY_FOR_GLX(vp) ) {
		init_glx_context(vp);
		vp->vw_flags |= VIEW_GLX_RDY;
	}

#ifdef CAUTIOUS
	if( vp->vw_ctx == NULL ){
		sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  select_gl_viewer %s:  no GL context!?",vp->vw_name);
		NERROR1(DEFAULT_ERROR_STRING);
	}
#endif /* CAUTIOUS */

/*
if( verbose ){
sprintf(ERROR_STRING,"select_gl_viewer( %s ) setting current context to 0x%lx",
vp->vw_name,(int_for_addr)vp->vw_ctx);
advise(ERROR_STRING);
}
*/
	if( glXMakeCurrent(vp->vw_dpy,vp->vw_xwin,vp->vw_ctx) != True ){
		sprintf(DEFAULT_ERROR_STRING,
		"Unable to set current GLX context to %s!?",vp->vw_name);
		NWARN(DEFAULT_ERROR_STRING);
	}
	gl_vp = vp;
} /* end select_gl_viewer */

#endif /* HAVE_OPENGL */

