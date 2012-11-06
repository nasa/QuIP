
#include "quip_config.h"

char VersionId_qutil_strbuf[] = QUIP_VERSION_STRING;

/* utilities for manipulating strings of arbitrary length */

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#include "strbuf.h"
#include "getbuf.h"

void enlarge_buffer(String_Buf *sbp,u_int size)
{
	char *newbuf;

	size += 32;	/* give us some headroom */

	newbuf = (char*) getbuf(size);
	if( sbp->sb_size > 0 ){
		/* copy old contents */
		memcpy(newbuf,sbp->sb_buf,sbp->sb_size);
		givbuf(sbp->sb_buf);
	} else {
		/* if this is a new buffer, initialize w/ null string.
		 * This insures that cat_string will work to a null stringbuf.
		 */
		*newbuf=0;
	}

	sbp->sb_buf = newbuf;
	sbp->sb_size = size;
}

void copy_string(String_Buf *sbp,const char* str)
{
	if( strlen(str)+1 > sbp->sb_size )
		enlarge_buffer(sbp,strlen(str)+1);
	strcpy(sbp->sb_buf,str);
}

void copy_strbuf(String_Buf *dst_sbp,String_Buf *src_sbp)
{
	u_int n;

	if( src_sbp->sb_size == 0 ) return;	/* nothing to copy */

	if( dst_sbp->sb_size < (n=strlen(src_sbp->sb_buf)+1) )
		enlarge_buffer(dst_sbp,n);

	strcpy(dst_sbp->sb_buf,src_sbp->sb_buf);
}

void cat_string(String_Buf *sbp,const char *str)
{
	u_int need;

	if( (need=strlen(sbp->sb_buf)+strlen(str)+1) > sbp->sb_size )
		enlarge_buffer(sbp,need);
	strcat(sbp->sb_buf,str);
}


