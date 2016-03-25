
/* 
 * Sometimes we might compile -Dgetbuf=malloc -Dgivbuf=free
 */

#ifndef _GETBUF_H_
#define _GETBUF_H_

//#define USE_GETBUF		// for debugging...

#ifdef USE_GETBUF

#include "typedefs.h"

extern void showmaps(void);
extern char *get_env_var(const char *name,u_long *ptr);
void * bigbuf(u_long size);

/* We use getbuf & givbuf instead of malloc & free, but usually we #define
 * getbuf to be malloc and givbuf to be free...  But when we suspect a memory
 * leak, we can use our own versions and dump more info about allocation...
 */


/* Not using malloc... */
#define getbuf(s)		bigbuf( (u_long) ( s ) )
void givbuf(const void *addr);

#else /* ! USE_GETBUF */

// Use this define for extra debugging...
//#define DEBUG_GETBUF

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* malloc */
#endif /* HAVE_STDLIB_H */

/* malloc and free are supposed to be thread-safe if linking w/ -lpthreads? */
// We wrap malloc in getbuf to provide error-checking.
//#define getbuf( s )	malloc( s )
extern void * getbuf(size_t size);

#ifdef DEBUG_GETBUF
extern void givbuf(void *a);
#else // ! DEBUG_GETBUF
#define givbuf( a )	free( a )
#endif // ! DEBUG_GETBUF


#endif /* ! USE_GETBUF */

/* This function exists in either case */
__attribute__ ((__noreturn__)) extern void mem_err(const char *);


#endif /* ! _GETBUF_H_ */

