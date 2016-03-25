#include "quip_config.h"

#ifdef SGI

/*
 * File:glxhelper.c:
 *
 * Description: This file provides a helper function "GLXCreateWindow",
 *  which does all the necessary magic to create an X window 
 *  suit able for GL drawing to take place within. See the
 *  definition of GLXCreateWindow() for a description of how
 *  to call it.
 *
 * Functions:   Uses no SGI Movie Library functions.
 */

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <gl/glws.h>
#include <stdio.h>
#include <stdlib.h>
#include "glxhelper.h"

char *typeToName[] = {
	"color index single buffer",
	"color index double buffer",
	"rgb single buffer",
	"rgb double buffer",
};

/*
 * Little helper function used to build up a GLXconfig array.
 */

static void setEntry ( GLXconfig* ptr, int b, int m, int a )
{
	ptr->buffer = b;
	ptr->mode = m;
	ptr->arg = a;
}

int my_error_handler( Display *dpy, XErrorEvent *ee )
{
	NWARN("X error");
}

/*
 * GLXCreateWindow( dpy, parent, x, y, w, h, borderWidth, xValueMask, 
   winAttribs, type )
 *
 * Return value is the X window id of the newly created window.
 *
 * Arguments are:
 *	dpy		The X "Display*" returned by XOpenDisplay
 *	parent		The parent of the newly created window,
 *			a typical value for this is
 *			RootWindow(dpy, DefaultScreen(dpy))
 *
 *	x,y		The location of the window to be created,
 *			y coordinate is measured from the top down.
 *
 *	w,h		size of the new window
 *
 *	borderWidth	the X border size for this window, should probably
 *			be zero.
 *
 *  xValueMaskSpecifies which window attributes are defined in
 *  the winAttribs argument.  This mask is the bitwise
 *  inclusive OR of the valid attribute mask bits.  If
 *  xValueMask is zero, the attributes are ignored and
 *  are not referenced. (See XCreateWindow(3X11)).
 *
 *  winAttribs  Specifies the structure from which the values (as
 *  specified by xValueMask) are to be taken. xValueMask 
 *  should have the appropriate bits set to indicate
 *  which attributes have been set in the structure.
 *
 *
 *	type		the GLXWindowType (see glxhelper.h) desribing the
 *			typer of GL drawing to be done in this window
 */

Window GLXCreateWindow( Display* dpy, Window parent, int x, int y,
		   int w, int h, int borderWidth, unsigned long xValueMask,
   XSetWindowAttributes * winAttribs, 
   GLXWindowType type )
{
	GLXconfig params[50];
	GLXconfig* nextConfig;
	GLXconfig* retConfig;
	XVisualInfo* vis=((XVisualInfo *)NULL);	/* init eliminates warning */
	XVisualInfo template;
	XSetWindowAttributes childWinAttribs = *winAttribs;
	XWindowAttributes	parentWinAttribs;
	int i, nret;
	Window win;

	/*
	 * This builds an array in "params" that describes for GLXgetconfig(3G)
	 * the type of GL drawing that will be done.
	 */

	nextConfig = params;
	switch ( type ) {
	  case GLXcolorIndexSingleBuffer:
	setEntry( nextConfig++, GLX_NORMAL, GLX_RGB, FALSE );
	setEntry( nextConfig++, GLX_NORMAL, GLX_DOUBLE, FALSE );
	break;
	  case GLXcolorIndexDoubleBuffer:
	setEntry( nextConfig++, GLX_NORMAL, GLX_RGB, FALSE );
	setEntry( nextConfig++, GLX_NORMAL, GLX_DOUBLE, TRUE );
	break;
	  case GLXrgbSingleBuffer:
	setEntry( nextConfig++, GLX_NORMAL, GLX_RGB, TRUE );
	setEntry( nextConfig++, GLX_NORMAL, GLX_DOUBLE, FALSE );
	break;
	  case GLXrgbDoubleBuffer:
	setEntry( nextConfig++, GLX_NORMAL, GLX_RGB, TRUE );
	setEntry( nextConfig++, GLX_NORMAL, GLX_DOUBLE, TRUE );
	break;
	}

	/* 
	 * Input to GLXgetconfig() is null terminated.
	 */

	setEntry( nextConfig, 0, 0, 0 );

	/*
	 * Get configuration data for a window based on above parameters.
	 * First we have to find out which screen the parent window is on,
	 * then we can call GXLgetconfig().
	 */

	XGetWindowAttributes( dpy, parent, &parentWinAttribs );
	retConfig = GLXgetconfig( dpy,
		XScreenNumberOfScreen( parentWinAttribs.screen ), 
		params );
	if ( retConfig == NULL ) {
		printf( "Sorry, can't support %s type of windows\n",
			typeToName[type] );
		exit( EXIT_FAILURE );
	}

	/*
	 * The GL sets its own X error handlers, which aren't as informative
	 * when errors happen.  Calling XSetErrorHandler(0) here will
	 * reset back to the default Xlib version.
	 */

	XSetErrorHandler( my_error_handler );

	/*
	 * Scan through config info, pulling info needed to create a window
	 * that supports the rendering mode.
	 */

	for ( nextConfig = retConfig; nextConfig->buffer; nextConfig++ ) {
		unsigned long buffer = nextConfig->buffer;
		unsigned long mode = nextConfig->mode;
		unsigned long value = nextConfig->arg;
		switch ( mode ) {
	  	case GLX_COLORMAP:
		if ( buffer == GLX_NORMAL ) {
			childWinAttribs.colormap = value;
		}
		break;
	  	case GLX_VISUAL:
		if ( buffer == GLX_NORMAL ) {
			template.visualid = value;
			template.screen = DefaultScreen( dpy );
			vis = XGetVisualInfo( dpy, VisualScreenMask|VisualIDMask,
				 &template, &nret );
		}
		break;
		}
	}

	/* Create the window.  */

	win = XCreateWindow( dpy, parent, x, y, w, h,
		borderWidth, vis->depth, InputOutput, vis->visual,
		xValueMask, &childWinAttribs );

	/*
	 * Rescan configuration info and find window slot that getconfig
	 * provided.  Fill it in with the window we just created.
	 */

	for ( nextConfig = retConfig; nextConfig->buffer; nextConfig++ ) {
		if ( ( nextConfig->buffer == GLX_NORMAL ) && 
		( nextConfig->mode == GLX_WINDOW ) ) {
		nextConfig->arg = win;
		break;
		}
	}

	/*
	 * Now "retConfig" contains all the information the GL needs to
	 * configure the window and its own internal state.
	 */

	i = GLXlink( dpy, retConfig );
	if ( i < 0 ) {
		printf( "GLXlink returned %d\n", i );
		exit( EXIT_FAILURE );
	}

	return win;
}

#endif /* SGI */

