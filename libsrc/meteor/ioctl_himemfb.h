
/*
 * ioctl_himemfb.h
 *
 * 03/23/99
 *
 */

#define HM_GETADDR	_IOR('x', 1, uint32_t)       /* get physical address */
#define HM_GETSIZE	_IOR('x', 2, uint32_t)       /* get size */

/* jbm:  these two use chunk structures, the first requests a chunk of
 * a certain size, and is returned an address and actual size.
 * If the requested size is 0, then no allocation is done, but
 * the maximum possible chunk size is returned, with a null address.
 */

#define HM_REQCHUNK	_IOWR('x', 3, Mem_Chunk )
#define HM_RLSCHUNK	_IOWR('x', 4, Mem_Chunk )
#define HM_ADDPID	_IOWR('x', 5, Mem_Chunk )
#define HM_SUBPID	_IOWR('x', 6, Mem_Chunk )

