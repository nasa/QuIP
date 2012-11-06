
/* 
 * Sometimes we might compile -Dgetbuf=malloc -Dgivbuf=free
 */

#ifndef _GETBUF_H_
#define _GETBUF_H_

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

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* malloc */
#endif /* HAVE_STDLIB_H */

/* malloc and free are supposed to be thread-safe if linking w/ -lpthreads? */
#define getbuf( s )	malloc( s )
#define givbuf( a )	free( a )


#endif /* ! USE_GETBUF */

/* This function exists in either case */
extern void mem_err(const char *);


#endif /* ! _GETBUF_H_ */

