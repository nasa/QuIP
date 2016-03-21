
#include "quip_config.h"
#include "quip_prot.h"

/* This is a mess! */

#include <stdio.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_LINUX_UNISTD_H
#include <linux/unistd.h>	/* why do we need this??? */
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#include "llseek.h"

#ifdef HAVE_LSEEK64

off64_t my_lseek64(int fd,off64_t offset,int whence)
{
	off_t r;
	r=lseek64(fd,offset,whence);
	/*
	if( r < 0 )
		return (off64_t) r;	// is off_t unsigned?
	else
		return r;
	*/
	return r;
}

#else /* ! HAVE_LSEEK64 */

#ifdef HAVE_LLSEEK

off64_t my_lseek64(int fd,off64_t offset,int whence)
{
	u_long hi,lo;
	off64_t result;

#ifdef SUN
	hi = 0;
#else
	hi = offset>>32;
#endif
	lo = offset & 0xffffffff;

	/* use llseek or _llseek??? */
	if( llseek(fd,hi,lo,&result,whence) < 0 ){
		perror("llseek");
		return(-1);
	}

	return(result);
}

#else /* ! HAVE_LLSEEK */

#if __WORDSIZE == 64

/* just use lseek */

off64_t my_lseek64(int fd, off64_t offset, int whence )
{
	return( lseek(fd,(off_t)offset,whence) );
}

#else /* __WORDSIZE != 64 */

#error Do not know what to do about lseek64.

#endif /* __WORDSIZE != 64 */
#endif /* ! HAVE_LLSEEK */
#endif /* ! HAVE_LSEEK64 */


