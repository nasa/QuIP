#include "quip_config.h"
char VersionId_qutil_tkspace[] = QUIP_VERSION_STRING;

#include <stdio.h>

#include "freel.h"
#include "query.h"	// ERROR_STRING


/*
 * Remove a block at a specific address from a free list.
 * This function was written to recreate databases when restarting
 * an application with a know environment.
 */

int takespace(FreeList *list, u_long a, u_long s)
			/* list = list from which to allocate */
			/* a = address */
			/* s = size */
{
	register FreeBlk *frp;
	u_long offset;

	for( frp=list->fl_blockp; frp->size; frp++ ){
		if( a >= frp->blkno && a < (frp->blkno+frp->size) ){
			offset = a - frp->blkno;
			if( s > (frp->size - offset) ){
				sprintf(DEFAULT_ERROR_STRING,
			"takespace:  can't allocate %ld blocks at %ld",s,a);
				NWARN(DEFAULT_ERROR_STRING);
				return(-1);
			}
			if( offset == 0 ){
				frp->blkno += s;
				frp->size -= s;
				if( frp->size == 0 ){
					do {
						frp++;
						(frp-1)->blkno = frp->blkno;
					} while( ((frp-1)->size = frp->size) );
				}
			} else if( (offset+s) == frp->size ){
				frp->size = offset;
			} else {
				FreeBlk *frpsav;
				u_long old_a,new_a,old_s,new_s;

				/* need to split one free block into two */

				frpsav=frp;

				/* push list out */
				new_a=frp->blkno;
				new_s=frp->size;
				do { 
					old_a=new_a;
					old_s=new_s;
					new_s=(frp+1)->size;
					new_a=(frp+1)->blkno;
					(frp+1)->size = old_s;
					(frp+1)->blkno = old_a;
					frp++;
				} while( old_s );
				frp=frpsav;
				frp->size = offset;
				frp++;
				frp->blkno += offset + s;
				frp->size -= offset + s;
			}
		}
	}
	return(0);
}

