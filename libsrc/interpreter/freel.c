#include "quip_config.h"

/* resource allocation patterned after the kernels malloc/mfree
 *
 * The basic approach is to maintain a list of address/size records
 * indicating a block of free space.  Intially there is a single
 * block at the start address, with size equal to the whole area.
 * As blocks of space are allocated and freed, the list grows,
 * reflecting "fragmentation" of the memory area.  One bug in
 * the original version (taken from v6 kernel) is that there are
 * no checks on the number of blocks used...  the list is simply
 * assumed to be longer than the largest possible amount of fragmentation.
 * This may be an erroneous assumption!
 */

#include <stdio.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* abort() */
#endif

#include "quip_prot.h"
#include "freel.h"
#include "debug.h"

#ifdef LONG_64_BIT
#define MAX_SIGNED_LONG	0x7fffffffffffffff
#else
#define MAX_SIGNED_LONG	0x7fffffffL
#endif

#ifdef QUIP_DEBUG
static u_long fldebug=FREEL_DEBUG_MASK;	/* one of the auto modules */
#endif /* QUIP_DEBUG */

/*
 * Initialize a freelist
 *
 * Perhaps we should use uint32_t instead of u_long???
 */

void freeinit(FreeList *list, count_t n_elts, u_long ntotal)
     /* list = pointer to the list to be initialized */
     /* n_elts = number of elements in the list */
     /* ntotal = number of free blocks that the list represents */
{
	count_t i;
	FreeBlk *blkp;

	list->fl_n_blocks = n_elts;

	list->fl_blockp = (FreeBlk *) malloc( n_elts * sizeof(FreeBlk) );
	if( list->fl_blockp == NULL ){
		NERROR1("freeinit:  can't malloc FreeBlk list");
		IOS_RETURN
	}

	blkp = list->fl_blockp;

#ifdef QUIP_DEBUG
if( debug & fldebug ){
sprintf(DEFAULT_ERROR_STRING,"freeinit:  blkp= 0x%lx\tn_elts= %d\nntotal= 0x%lx",
(u_long)blkp,n_elts,ntotal);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif

	/* so getspace() can return -1 on error... */
	if( ntotal > MAX_SIGNED_LONG ){
		sprintf(DEFAULT_ERROR_STRING,"requested arena size %ld too big for freeinit (max = %ld)",ntotal,MAX_SIGNED_LONG);
		NERROR1(DEFAULT_ERROR_STRING);
		IOS_RETURN
	}

	blkp->blkno = 0;
	blkp->size = ntotal;
	blkp++;

	for(i=1;i<n_elts;i++){
		/*
		 * We initialize blkno to ntotal (instead of 0)
		 * to fix a pernicious bug which was exhibited
		 * when the memory at the very tail end of the
		 * space was allocated - when this block was
		 * later freed an error occurred.  Perhaps there
		 * should be better checks in givspace, but
		 * this is an easy fix.
		 */
		blkp->blkno = ntotal;
		blkp->size=0;
		blkp++;
	}
}

/*
 * Get a block of size s from a freelist
 * Returns -1 if space cannot be found.
 * Note that although the block addresses are u_long,
 * we use the sign bit here... so we limit the arena
 * size to the biggest signed long.
 *
 * The end of the list is signified by a zero sized block.
 */

long getspace(FreeList *list, u_long s)
				/* u = freelist to be searched */
				/* s = size of desired free space */
{
	FreeBlk *frp;
	u_long a;

	for( frp=list->fl_blockp; frp->size; frp++ ){
		if( frp->size >= s ){		/* this block is big enuff */
			a=frp->blkno;
			frp->blkno += s;
			if( (frp->size -= s) == 0){	/* entire block used */
				do {
					/* this block is gone, shift the others down in the table */
					frp++;
					(frp-1)->blkno = frp->blkno;
				} while( ((frp-1)->size = frp->size) );
			}
#ifdef QUIP_DEBUG
if( debug & fldebug ){
sprintf(DEFAULT_ERROR_STRING,"getspace returning 0x%lx blocks at 0x%lx",s,a);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif
			return((long)a);
		}
	}
#ifdef QUIP_DEBUG
if( debug & fldebug ){
sprintf(DEFAULT_ERROR_STRING,"getspace couldn't find space for 0x%lx",s);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif
	return(-1);
}


/*
 * Return size blocks at address addr to the given freelist.
 * The freed area can be contiguous with other free space
 * at neither, one, or both ends.  This routine checks that the
 * freed space is not already free.
 *
 * If the freed block is contiguous with at least one other free block,
 * then no additional free blocks are used.  If it is in the middle, however,
 * then the latter half of the list will have to be pushed out...
 * We need to check that this doesn't overflow the number of freeblk structures.
 */

int givspace(FreeList *list, u_long size, u_long addr)
		/* list = free list */
		/* size = size of block to be returned */
		/* addr = address of block to be returned */
{
	FreeBlk *frp;
	u_long a, t;

#ifdef QUIP_DEBUG
if( debug & fldebug ){
sprintf(DEFAULT_ERROR_STRING,"givspace:  0x%lx blocks at 0x%lx",size,addr);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif

	a=addr;

	/* find the first free block past the new space */

	for(frp=list->fl_blockp;frp->blkno<= a && frp->size != 0; frp++)
		;

	/*
	 * two ways to screw up:
	 *
	 * 1) the newly freed block can overlap
	 * the tail end of the preceding one
	 *
	 * 2) the newly freed block can overlap
	 * the beginning of the following one
	 */

//#ifdef CAUTIOUS
//	/* error #1 */
//	if(frp != list->fl_blockp && (frp-1)->blkno + (frp-1)->size > a ){
//showmap(list);
//		NWARN("CAUTIOUS:  givspace:  freeing unallocated memory!? (error #1)");
//		abort();
//	}
	assert(frp == list->fl_blockp || (frp-1)->blkno + (frp-1)->size <= a );

//	/* error #2 */
//	if( a+size > frp->blkno ){
//showmap(list);
//		NWARN("CAUTIOUS:  givspace:  freeing unallocated memory!? (error #2)");
//		abort();
//	}
//#endif
	assert( a+size <= frp->blkno );

	/* if not head of list and new area right after frp-1 */
	if( frp != list->fl_blockp && (frp-1)->blkno + (frp-1)->size == a ){
		/* new area is contiguous with lower area */

#ifdef QUIP_DEBUG
if( debug & fldebug ) NADVISE("coalescing back");
#endif
		/* coalesce new blocks with frp-1 */
		(frp-1)->size += size;

		if( a+size == frp->blkno ){

#ifdef QUIP_DEBUG
if( debug & fldebug ) NADVISE("coalescing forward");
#endif
			/* coalesce with frp */
			(frp-1)->size += frp->size;

			while( frp->size ){
				frp++;
				(frp-1)->size = frp->size;
				(frp-1)->blkno = frp->blkno;
			}
		}
	} else {		/* first free block is head of list */
		/* contiguous with frp? */
		if( a+size == frp->blkno && frp->size ){
			/* contiguous with later area */
#ifdef QUIP_DEBUG
if( debug & fldebug ) NADVISE("coalescing forward #2");
#endif
			/* coalesce */
			frp->blkno -= size;
			frp->size += size;
		} else if( size ) {
#ifdef QUIP_DEBUG
if( debug & fldebug ) NADVISE("restoring blocks");
#endif
			t=size;
			/* push list out */
			do { 
				long block_index;

				block_index = frp - list->fl_blockp;
				if( block_index >= (int)list->fl_n_blocks )
					NERROR1("givspace:  excessive fragmentation!?");

				size=t;
#ifdef QUIP_DEBUG
if( debug & fldebug ) {
sprintf(DEFAULT_ERROR_STRING,"givspace:  frp= 0x%lx, s=%ld a=%ld",(u_long)frp,size,a);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif
				t=frp->blkno;
				frp->blkno=a;
				a=t;
				t=frp->size;
				frp->size = size;
				frp++;
			} while( size );
		}
	}
#ifdef QUIP_DEBUG
if( debug & fldebug ){
NADVISE("givspace DONE");
}
#endif
	return(0);
}

/* This is like givspace(), but we use it to add a non-contiguous area
 * of memory to an existing pool.  This is not really kosher, because it
 * defeats the checks performed by givspace()...  this is ok if we
 * only return chunks we have previously allocated.
 */

int addspace(FreeList* list, u_long size, u_long addr)
     /* list = free list */
     /* size = size of block to be returned */
     /* addr = address of block to be returned */
{
	FreeBlk *frp;
	u_long a, t;

#ifdef QUIP_DEBUG
if( debug & fldebug ){
sprintf(DEFAULT_ERROR_STRING,"addspace:  0x%lx blocks at 0x%lx",size,addr);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif

	a=addr;

	/* find the first free block past the new space */

	for(frp=list->fl_blockp;frp->blkno<= a && frp->size != 0; frp++)
		;

	/*
	 * two ways to screw up:
	 *
	 * 1) the newly freed block can overlap
	 * the tail end of the preceding one
	 *
	 * 2) the newly freed block can overlap
	 * the beginning of the following one
	 */

//#ifdef CAUTIOUS
//	/* error #1 */
//	if(frp != list->fl_blockp && (frp-1)->blkno + (frp-1)->size > a ){
//		NWARN("CAUTIOUS:  addspace:  freeing unallocated memory!? (error #1)");
//		abort();
//	}
//#endif
	assert(frp == list->fl_blockp || (frp-1)->blkno + (frp-1)->size <= a );

	/* if not head of list and new area right after frp-1 */
	if( frp != list->fl_blockp && (frp-1)->blkno + (frp-1)->size == a ){
		/* new area is contiguous with lower area */

#ifdef QUIP_DEBUG
if( debug & fldebug ) NADVISE("coalescing back");
#endif
		/* coalesce new blocks with frp-1 */
		(frp-1)->size += size;

		if( a+size == frp->blkno ){

#ifdef QUIP_DEBUG
if( debug & fldebug ) NADVISE("coalescing forward");
#endif
			/* coalesce with frp */
			(frp-1)->size += frp->size;

			while( frp->size ){
				frp++;
				(frp-1)->size = frp->size;
				(frp-1)->blkno = frp->blkno;
			}
		}
	} else {		/* first free block is head of list */
		/* contiguous with frp? */
		if( a+size == frp->blkno && frp->size ){
			/* contiguous with later area */
#ifdef QUIP_DEBUG
if( debug & fldebug ) NADVISE("coalescing forward #2");
#endif
			/* coalesce */
			frp->blkno -= size;
			frp->size += size;
		} else if( size ) {
#ifdef QUIP_DEBUG
if( debug & fldebug ) NADVISE("restoring blocks");
#endif
			t=size;
			/* push list out */
			do { 
				size=t;
#ifdef QUIP_DEBUG
if( debug & fldebug ) {
sprintf(DEFAULT_ERROR_STRING,"givspace:  frp= 0x%lx, s=%ld a=%ld",(u_long)frp,size,a);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif
				t=frp->blkno;
				frp->blkno=a;
				a=t;
				t=frp->size;
				frp->size = size;
				frp++;
			} while( size );
		}
	}
#ifdef QUIP_DEBUG
if( debug & fldebug ){
NADVISE("givspace DONE");
}
#endif
	return(0);
}

/*
 * Print the allocation map on stderr
 */

void showmap(FreeList *list)
		/* free list to be printed */
{
	u_long i=0;
	FreeBlk *blkp;

	blkp = list->fl_blockp;
	if( blkp->size == 0 ){
		NADVISE("All memory allocated");
		return;
	}
	sprintf(DEFAULT_ERROR_STRING,"\nAvailable memory (list = 0x%lx):",
		(u_long)blkp);
	NADVISE(DEFAULT_ERROR_STRING);

	while( blkp->size != 0 ){
		sprintf(DEFAULT_ERROR_STRING,"%ld\t0x%lx at 0x%lx",i++,
			blkp->size,blkp->blkno);
		NADVISE(DEFAULT_ERROR_STRING);
		blkp++;
	}
	NADVISE("");	/* print a blank line */
}

/* Return the number of fragments in a free list */

int n_map_frags(FreeList *list)
{
	FreeBlk *frp;
	int i=0;

	for( frp=list->fl_blockp; frp->size; frp++ )
		i++;
	return(i);
}


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

