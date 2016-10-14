#include "quip_config.h"
#include <stdio.h>
#include "quip_prot.h"
#include "debug.h"

#ifdef QUIP_DEBUG
static u_long gbdebug=GETBUF_DEBUG_MASK;
#endif

#ifdef USE_GETBUF

/* originally motivated by distrust of malloc,
 * getbuf is a replacement allocator.  It is also useful
 * for tracing memory leaks.  But most of the time,
 * we assume that the system's malloc/free will
 * be more efficient.  Also, using malloc
 * we have access to all of the system's memory
 * without having to set aside a fixed amount.
 */

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* getenv(), abort() */
#endif

#ifdef HAVE_MALLOC_H
#include <malloc.h>	/* malloc() & _halloc() */
#elif HAVE_SYS_MALLOC_H
#include <sys/malloc.h>	/* malloc() & _halloc(), OSX */
#endif

#include "query.h"
#include "getbuf.h"
#include "freel.h"

/*
 * Instead of one static heap, we keep two:
 * a small one for strings & structs, & another one for images.
 * This was made necessary by the following problem:
 * a big image is allocated, then some small stuff is allocated,
 * then the big image is free, more small stuff allocated... repeat
 * the "hole" made by the big image gets used for small stuff, then
 * there's no room for the next big image...
 *
 * jbm 4-25-94
 */

/* This stuff is here in case we compile -Dgetbuf=malloc -Dgivbuf=free */

/*
 * How many heap chunks do we really need?
 * (This depends on chunksize, fragmentation).
 * These are probably way too big...
 * Maybe use a linked list that can be grown???
 *
 * Maybe have a separate heap for strings with small chunksize?
 */

#define MAX_HEAP_CHUNKS 0x1000	/* 4K */

typedef struct heap { 
	FreeList	heap_fl;
	char *		heap_base;
	u_long		heap_total;
	u_long		heap_chunk;
	u_long		heap_mask;
#ifdef QUIP_DEBUG
	u_long		heap_free;
#endif /* QUIP_DEBUG */
} Heap;

/* local prototypes */
static void setup_heap(Heap *hp,u_long total,u_long chunksize);
static void heap_init(void);
static void *get_from_heap(Heap *hp,u_long s);

static Heap small_heap;

static Heap big_heap;

#define BIG_HEAPSIZE	0x2000000L

/* This (0x100000=1Mb) was too small on a big sgi application!? */
/* this also ran out running vt on a big set of expression files...
 * there are probably memory leaks and wasteful stuff, but for now we'll make
 * it a bit bigger.
 */

//#define SMALL_HEAPSIZE	0x200000L	/* 2 Meg */
#define SMALL_HEAPSIZE	0x400000L	/* 4 Meg */


/* Some machines like the SUN's require an array of doubles
 * to be aligned... not true on SGI's and ???
 *
 * After "upgrade" to irix 6.2, sgi indigo2 seems to want doubles aligned too.
 *
 * Each allocated block is preceded by a word containing the number of bytes.
 * This would normally be a 4 byte u_long, but in some cases we need the following
 * data to be aligned...
 *
 * The above comment was written when a long was 32 bits, but now it is 64...
 * 32 bits should still be enough to hold a block size though...
 * But we probably also want things to be aligned...
 */

#define SIZE_OFFSET	((long)(sizeof(double)))

/* smallest unit of memory - must be a power of 2 */
/* #define CHUNK_SIZE	(sizeof(double)) */
#define SMALL_CHUNK_SIZE	((long)sizeof(double))

/* hex digits: 1 16 256 4K 64K */
/* an xlib lutbuffer is about 3k, we would like these to go in the big heap */
//#define BIG_CHUNK_SIZE		0x800L	/* 2K */
#define BIG_CHUNK_SIZE		0x200L	/* 512 */

#define BIG_HEAP_THRESH		BIG_CHUNK_SIZE
static u_long big_heapsize=BIG_HEAPSIZE;

static u_long small_heapsize=SMALL_HEAPSIZE;

static int heap_ready=0;

/* The determination of which heap a block goes into, and is in,
 * is a bit complicated:  The user requests a block of size s.
 * This then has SIZE_OFFSET added to it, which guarantees that
 * we have room for the size word, plus the callers data.
 * (On the SUN it has to be the length of a double so the user
 * area is double-aligned.)
 * This augmented size is then passed to get_from_heap().
 * This routine may further increase the size to make it an
 * integral number of heap chunks (to minimize fragmentation
 * and preserve alignment of blocks).  This size is finally
 * stored at the beginning of the block, and the address
 * incremented by SIZE_OFFSET is returned to the user.
 *
 * The chunk size has to be a multiple of 2*SIZE_OFFSET - see setup_heap().
 *
 *
 */


void mem_err(const char *whence) __attribute__((__noreturn__))
{
	if( heap_ready ){
		sprintf(DEFAULT_ERROR_STRING,"big_heapsize = 0x%lx",big_heapsize);
		NADVISE(DEFAULT_ERROR_STRING);
		NADVISE("Consider changing BIG_HEAPSIZE");
	}
	sprintf(DEFAULT_ERROR_STRING,"%s:  out of memory",whence);
	NERROR1(DEFAULT_ERROR_STRING);
}

/*
 * Print the allocation maps for the large and small heaps
 * Used for debugging & finding memory leaks.
 */

void showmaps(void)
{
	NADVISE("showing allocation map for big heap:");
	showmap(&big_heap.heap_fl);

	NADVISE("showing allocation map for small heap:");
	showmap(&small_heap.heap_fl);
}

static void setup_heap(Heap *hp, u_long total, u_long chunksize)
{
	/* we don't really want to typecase total to short! */
	hp->heap_base = (char*)  malloc((int)total);
	if( hp->heap_base == (char *)NULL ){
		sprintf(DEFAULT_ERROR_STRING,"tried to allocate 0x%lx bytes",total);
		NADVISE(DEFAULT_ERROR_STRING);
		NERROR1("couldn't malloc heap storage");
	}
	hp->heap_total = total;
#ifdef QUIP_DEBUG
	hp->heap_free = total;
#endif /* QUIP_DEBUG */

	/* If chunk_size is not a multiple of 2*SIZE_OFFSET,
	 * we can lose memory:  we ask for two chunklets, the
	 * system adds a third to store the size...  if we free
	 * this space and then try to get one (the system will
	 * ask for two), we have created a hole of one chunklet
	 * that can never be used!?
	 */
	
	/* We may have data alignment requirements that are more stringent...
	 * e.g. for pentium mmx operations, the data must be aligned
	 * on a 128 bit (16 byte) boundary.
	 *
	 * This means that we will have to use 16 bytes to store the size,
	 * wasting 12 bytes, but maybe for the large heap this can be tolerated...
	 */

	if( chunksize < 2*SIZE_OFFSET )
		chunksize = 2*SIZE_OFFSET;

	hp->heap_chunk = chunksize;
	hp->heap_mask = ~(chunksize-1);

#ifdef CAUTIOUS
	/* check that chunksize is a power of 2 */
	chunksize -= 1;
	while( chunksize & 1 ) chunksize >>=1;
	// BUG better to use is_power_of_two() in the assertion
	// so is meaningful...
	assert( chunksize == 0 );
//	if( chunksize != 0 ) NERROR1("CAUTIOUS:  setup_heap:  chunksize is not a power of 2");
#endif /* CAUTIOUS */

	freeinit(&hp->heap_fl,MAX_HEAP_CHUNKS,(u_long)total);
}

/*
 * Search the environment for a variable with name name.
 * if it exists, convert the string to a number (%d) and assign
 * the value to the pointed to variable.  Returns a value
 * to the string, or NULL.
 */

char *get_env_var(const char* name, u_long* ptr)
		/* name = name of environment variable */
		/* ptr = ptr to program variable where to store the value */
{
	char *s;

	s=getenv(name);
	if( s != NULL ){
		*ptr = strtol(s,NULL,0);
	}
	return(s);
}

/* This is only called if someone calls bigbuf().
 * If we want to use malloc/free instead of getbuf/givbuf, then we
 * use macro defines on the command line.
 *
 * note that we print a different memory error if we have called this.
 */

static void heap_init()
{
	if( get_env_var("SMALL_HEAPSIZE",&small_heapsize) == NULL )
		small_heapsize=SMALL_HEAPSIZE;
	else {
		sprintf(DEFAULT_ERROR_STRING,
		"using evironment value 0x%lx for SMALL_HEAPSIZE",
		small_heapsize);
		NADVISE(DEFAULT_ERROR_STRING);
	}
	setup_heap(&small_heap,small_heapsize,SMALL_CHUNK_SIZE);

	if( get_env_var("BIG_HEAPSIZE",&big_heapsize) == NULL ) {
		big_heapsize=BIG_HEAPSIZE;
	} else {
		sprintf(DEFAULT_ERROR_STRING,
		"using evironment value 0x%lx for BIG_HEAPSIZE",
		big_heapsize);
		NADVISE(DEFAULT_ERROR_STRING);
	}
	setup_heap(&big_heap,big_heapsize,BIG_CHUNK_SIZE);

	heap_ready = 1;

}

static void *get_from_heap(Heap *hp, u_long s)
{
	u_long *ip;
	long index;
	char *cp;

	/* round size up to be an integral number of chunks.
	 * The passed size has already had the size offset added.
	 * But we also have to make sure that we can align the data.
	 * Can we assume that the chunks are aligned???
	 */

	s += hp->heap_chunk-1;
	s &= hp->heap_mask;

	index = getspace(&hp->heap_fl,s);
	if( index == -1 )
		return(NULL);

	ip = (u_long *) (hp->heap_base + index);
	*ip = s;
	cp = (char *) ip;
	cp += SIZE_OFFSET;

#ifdef QUIP_DEBUG
	hp->heap_free -= s;
#endif /* QUIP_DEBUG */
	return( cp );
}

/*
 * Return a pointer to a buffer of size bytes.
 * Prints a warning message if unsuccessful.
 * Returns NULL if unable to allocate the space.
 */

void * bigbuf(u_long size)
		/* number of bytes to allocate */
{
	u_long s;
	void *cp;

//#ifdef CAUTIOUS
//	if( size == 0 )
//		NERROR1("CAUTIOUS:  getbuf:  0 bytes requested!?");
//#endif /* CAUTIOUS */
	assert( size != 0 );

	if( !heap_ready ) heap_init();

	/* add a word to store the size of this block */
	s = size+SIZE_OFFSET;
	
	if(
		/* We subtract SMALL_CHUNK_SIZE from BIG_HEAP_THRESH,
		 * because blocks above this modified threshold will
		 * have their sizes rounded up to BIG_HEAP_THRESH.
		 *
		 * This used to be the SMALL_CHUNK_SIZE constant,
		 * but it had to be changed because sometimes (on Sun)
		 * setup_heap will increase the actual chunksize.
		 */

		s > (BIG_HEAP_THRESH-small_heap.heap_chunk) ){

		cp = get_from_heap(&big_heap,s);
		if( cp == NULL ){
			sprintf(DEFAULT_ERROR_STRING,
				"error getting %ld (0x%lx) bytes from big heap",s,s);
			NWARN(DEFAULT_ERROR_STRING);
			showmap(&big_heap.heap_fl);
		} else {
			/* get_from_heap succeeded... */

#ifdef QUIP_DEBUG
if( debug & gbdebug ){
    /*
sprintf(DEFAULT_ERROR_STRING,"GetBuf:  %10ld at 0x%8lx",
	*((u_long *) ((u_long)cp-SIZE_OFFSET)),
	(u_long)((((char *)cp)-SIZE_OFFSET)-big_heap.heap_base) );
NADVISE(DEFAULT_ERROR_STRING);
     */
    fprintf(stderr,"GetBuf:  %10ld at 0x%8lx\n",
            *((u_long *) ((u_long)cp-SIZE_OFFSET)),
            (u_long)((((char *)cp)-SIZE_OFFSET)-big_heap.heap_base) );

}
#endif /* QUIP_DEBUG */
		}

	} else {
		cp = get_from_heap(&small_heap,s);
		if( cp == NULL ){
			sprintf(DEFAULT_ERROR_STRING,
				"error getting %ld (0x%lx) bytes from small heap",s,s);
			NWARN(DEFAULT_ERROR_STRING);
			showmap(&small_heap.heap_fl);
			NADVISE("Consider changing SMALL_HEAPSIZE");
			sprintf(DEFAULT_ERROR_STRING,
			"Current value is %ld (0x%lx)",SMALL_HEAPSIZE,
				SMALL_HEAPSIZE);
			NADVISE(DEFAULT_ERROR_STRING);
			abort();
		}
#ifdef QUIP_DEBUG
if( debug & gbdebug ){
    /*
sprintf(DEFAULT_ERROR_STRING,"getbuf:  %10ld at 0x%8lx (addr = 0x%lx)",
	*((u_long *) ((u_long)cp-SIZE_OFFSET)),
	(u_long)((((char *)cp)-SIZE_OFFSET)-small_heap.heap_base),(u_long)cp );
NADVISE(DEFAULT_ERROR_STRING);
     */
    /* DEAFULT_ERROR_STRING not valid at startup!? */
    fprintf(stderr,"getbuf:  %10ld at 0x%8lx (addr = 0x%lx)\n",
            *((u_long *) ((u_long)cp-SIZE_OFFSET)),
            (u_long)((((char *)cp)-SIZE_OFFSET)-small_heap.heap_base),(u_long)cp );
  
}
#endif /* QUIP_DEBUG */

	}

	if( cp == NULL ){
		sprintf(DEFAULT_ERROR_STRING,"0x%lx bytes requested",s);
		NADVISE(DEFAULT_ERROR_STRING);
		mem_err("getbuf");
	}

	return(cp);
}

/*
 * Deallocate space previously allocated with bigbuf()/getbuf()
 */

void givbuf(const void* addr)
		/* address at which to deallocate space */
{
	u_long *ip,index,this_size;
	char *caddr;

	/* figure out which heap this is in */
#ifdef QUIP_DEBUG
//if( debug & gbdebug ){
//sprintf(DEFAULT_ERROR_STRING,"givbuf:\t\t\t  freeing addr 0x%8lx",(u_long)addr);
//NADVISE(DEFAULT_ERROR_STRING);
//}
#endif /* QUIP_DEBUG */

	caddr = (char*) addr;
	caddr -= SIZE_OFFSET;
	ip = (u_long *) caddr;
	this_size = *ip;
	/* zero the size so we can detect a repeated free'ing */
	*ip = 0;
//#ifdef CAUTIOUS
//	if( this_size == 0 ) {
//		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  givbuf attempting to free a zero-length (unallocated or freed?) segment");
//		NWARN(DEFAULT_ERROR_STRING);
//		abort();
//	}
//#endif /* CAUTIOUS */
	assert( this_size != 0 );


	if( this_size >= BIG_HEAP_THRESH ){
		index = caddr - big_heap.heap_base;

#ifdef QUIP_DEBUG
if( debug & gbdebug ){
sprintf(DEFAULT_ERROR_STRING,"GivBuf:\t\t\t  %10ld at 0x%8lx",this_size,index);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

//#ifdef CAUTIOUS
//		if( index >= big_heap.heap_total ){
//			sprintf(DEFAULT_ERROR_STRING,
//		"Big heap block with size %ld has index %ld (0x%lx) !?",
//				this_size,index,index);
//			NADVISE(DEFAULT_ERROR_STRING);
//			NERROR1("CAUTIOUS:  wrong (big) heap!?");
//		}
//#endif /* CAUTIOUS */
		assert( index < big_heap.heap_total );

		if( givspace(&big_heap.heap_fl,this_size,index) < 0 ){
			sprintf(DEFAULT_ERROR_STRING,
	"givbuf:  error in givspace returning %ld bytes at address 0x%lx to big heap",
				this_size,(u_long)caddr);
			NWARN(DEFAULT_ERROR_STRING);
			return;
		}
#ifdef QUIP_DEBUG
		big_heap.heap_free += this_size;
#endif /* QUIP_DEBUG */
	} else {
		index = caddr - small_heap.heap_base;
//#ifdef CAUTIOUS
//		if( index >= small_heap.heap_total ){
//			sprintf(DEFAULT_ERROR_STRING,
//	"givbuf:  Small heap block at addr 0x%lx with size %ld has index %ld (0x%lx) !?",
//				(u_long)caddr,this_size,index,index);
//			NADVISE(DEFAULT_ERROR_STRING);
//			NERROR1("CAUTIOUS:  wrong (small) heap!?");
//		}
//#endif /* CAUTIOUS */
		assert( index < small_heap.heap_total );

#ifdef QUIP_DEBUG
if( debug & gbdebug ){
sprintf(DEFAULT_ERROR_STRING,"givbuf:\t\t\t  %10ld at 0x%8lx (addr = 0x%lx)",this_size,index,(long)/*caddr*/ addr);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

		if( givspace(&small_heap.heap_fl,this_size,index) < 0 ){
			sprintf(DEFAULT_ERROR_STRING,
	"givbuf:  error in givspace returning %ld bytes at address 0x%lx to small heap",
				this_size,(u_long)caddr);
			NWARN(DEFAULT_ERROR_STRING);
			return;
		}
#ifdef QUIP_DEBUG
		small_heap.heap_free += this_size;
#endif /* QUIP_DEBUG */

	}
}

#ifdef QUIP_DEBUG

COMMAND_FUNC( heap_report )
{
	sprintf(ERROR_STRING,"big_heap, %ld (0x%lx) bytes free, %d fragments",
		big_heap.heap_free,big_heap.heap_free,n_map_frags(&big_heap.heap_fl));
	advise(ERROR_STRING);
	sprintf(ERROR_STRING,"small_heap, %ld (0x%lx) bytes free, %d fragments",
		small_heap.heap_free,small_heap.heap_free,n_map_frags(&small_heap.heap_fl));
	advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

#else /* ! USE_GETBUF */

#ifdef MAX_DEBUG
static void mem_alert(size_t size)
{
static int n_allow=5;
	fprintf(stderr,"getbuf %ld (0x%lx)\n",(long)size,(long)size);
	fflush(stderr);
	if( (n_allow--) <= 0 ) abort();
}
#endif // MAX_DEBUG

void * getbuf(size_t size)
{
	void *p;

//if( size > 200000 ){
//mem_alert(size);
//}

//#ifdef CAUTIOUS
	// What is the difference between calloc and malloc???
	// calloc zeroes the memory!
//	p=calloc(size,1);
//#else // ! CAUTIOUS

	/* Call calloc instead of malloc to zero the memory.
	 * We expect callers to do this themselves if required!
	 *
	 * When we changed this to malloc, it revealed a buf in vectree...
	 * something that was assumed to be NULL if not set is not getting intialized!?
	 *
	 * Fixed one problem with VN_DECL_OBJ, but other problems appear to remain...
	 */
//	p=malloc(size);

	p=calloc(size,1);

//#endif // ! CAUTIOUS

	if( p == NULL ){
		NWARN("Out of memory!?");
		NERROR1("fatal memory error");
	}

#ifdef QUIP_DEBUG
if( debug & gbdebug ){
    /* DEAFULT_ERROR_STRING not valid at startup!? */
    fprintf(stderr,"getbuf:  %10ld at addr = 0x%lx\n", (long)size, (u_long)p );
}
#endif /* QUIP_DEBUG */

	return p;
}

#ifdef DEBUG_GETBUF

void givbuf( void * a )
{

#ifdef QUIP_DEBUG
if( debug & gbdebug ){
sprintf(DEFAULT_ERROR_STRING,"givbuf:\t\t\t  freeing addr 0x%lx)",(long)a);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	free(a);
}
#endif // DEBUG_GETBUF

void mem_err(const char *whence)
{
	sprintf(DEFAULT_ERROR_STRING,"%s:  out of memory.",whence);
	NERROR1(DEFAULT_ERROR_STRING);
    while(1) ;  // to silence compiler with noreturn attribute
}

#endif /* ! USE_GETBUF */

