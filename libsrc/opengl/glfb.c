
// GL framebuffer objects

#include "quip_config.h"

#ifdef HAVE_OPENGL

#define GL_GLEXT_PROTOTYPES

// this line was commented out to compile on ace...
//#include <GL/glew.h>	// on craik???
// needed on poisson...
#ifndef BUILD_FOR_OBJC
#ifdef HAVE_GL_GLEW_H
#include <GL/glew.h>	// on craik???
#endif // HAVE_GL_GLEW_H
#endif // BUILD_FOR_OBJC

#ifdef HAVE_GL_GLX_H
#include <GL/glx.h>
#endif


#ifdef GL_ARB_framebuffer_object
#undef GL_ARB_framebuffer_object
#endif //

//#include <GL/glext.h>		// included by glx.h

#include "quip_prot.h"	// now has to come after glew.h, because platform.h is included by data_obj.h,
			// and includes opencl stuff

//#include "glx_supp.h"
#include "glfb.h"
#include "item_type.h"

ITEM_INTERFACE_DECLARATIONS(Framebuffer,glfb,0)

Framebuffer *create_framebuffer(QSP_ARG_DECL  const char *name, int width, int height)
{
#ifdef HAVE_LIBGLEW
	Framebuffer *fbp;
	GLenum status;

	fbp = new_glfb(name);
	if( fbp == NULL ) return NULL;

	fbp->fb_width = width;
	fbp->fb_height = height;

	glGenFramebuffersEXT(1,&(fbp->fb_id));
	glBindFramebufferEXT(GL_FRAMEBUFFER,fbp->fb_id);
	glGenRenderbuffersEXT(1,&(fbp->fb_renderbuffer));
	glBindRenderbufferEXT(GL_RENDERBUFFER,fbp->fb_renderbuffer);
	glRenderbufferStorageEXT(GL_RENDERBUFFER,GL_RGBA,width,height);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,
						GL_RENDERBUFFER,fbp->fb_renderbuffer);

	status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER);
	if( status != GL_FRAMEBUFFER_COMPLETE ){
		sprintf(ERROR_STRING,"create_framebuffer:  frame buffer %s is not complete!?",
			name);
		WARN(ERROR_STRING);
		delete_framebuffer(QSP_ARG  fbp);
		return NULL;
	}

	return( fbp );

#else // ! HAVE_LIBGLEW
	error1("Sorry, need libglew to be present for gl frame buffers.");
	return NULL;	// NOTREACHED
#endif  // ! HAVE_LIBGLEW
}

void delete_framebuffer(QSP_ARG_DECL  Framebuffer *fbp)
{
	//GLenum status;

#ifdef HAVE_LIBGLEW
	glBindFramebufferEXT(GL_FRAMEBUFFER,0);
	glDeleteRenderbuffersEXT(1,&(fbp->fb_renderbuffer));
	glDeleteFramebuffersEXT(1,&(fbp->fb_id));
#else // ! HAVE_LIBGLEW
	error1("Sorry, need libglew to be present for gl frame buffers.");
#endif // ! HAVE_LIBGLEW
	del_glfb(fbp);
}

void glfb_info(QSP_ARG_DECL  Framebuffer *fbp)
{
	sprintf(MSG_STR,"Framebuffer %s:  %d x %d",
		fbp->fb_name,fbp->fb_height,fbp->fb_width);
	prt_msg(MSG_STR);
}

#ifdef FOOBAR
glIsFramebuffer(id_tbl[0]);

glFramebufferRenderbuffer( target, attachment, /*renderbuffer_target*/ GL_RENDERBUFFER,
	renderbuffer );

GL_FRAMEBUFFER_DEFAULT_WIDTH
GL_FRAMEBUFFER_DEFAULT_HEIGHT
glFramebufferParameteri(fbp->fb_target,pname,value);

glCheckFramebufferStatus(fbp->fb_target);

glCheckFramebufferStatusEXT(/*targt*/ GL_FRAMEBUFFER_EXT );

glGenRenderBuffersEXT( n, rb_tbl );
glBindRenderbufferEXT( GL_RENDERBUFFER_EXT, rb_tbl[0] );
glRenderbufferStorageEXT( GL_RENDERBUFFER_EXT, RGB/RGBA/DEPTH_COMPONENT, width, height );

glReadPixels(

#endif // FOOBAR

#endif // HAVE_OPENGL
