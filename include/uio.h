
/* includes for unix-style i/o */


#if HAVE_UNISTD_H
#include <unistd.h>
#endif

/* BUG - need to figure this out in the configure script!? */
#ifdef PC
#include <io.h>
#endif /* PC */

#ifdef MAC
#include <unix.h>
#endif /* MAC */


