
/* this should work if /usr/include is set up right... */
/*
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xmd.h>
*/

#ifdef HAVE_X11_XLIB_H
#include "X11/Xlib.h"
#endif

#ifdef HAVE_X11_XUTIL_H
#include "X11/Xutil.h"
#endif

#ifdef HAVE_X11_XMD_H
#include "X11/Xmd.h"
#endif


