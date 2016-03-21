
#if !defined(__STDC_VERSION__)
// jbm:  a total hack, to make the nvidia glx.h compile
// This didn't used to throw an error, what happened???
#define __STDC_VERSION__ 199901L
#endif

#ifdef HAVE_GL_GLX_H
#include <GL/glx.h>	// jbm added for glXSwapBuffers()
#endif

