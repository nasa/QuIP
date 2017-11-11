
/* structure for generalized free lists */

#ifndef NO_FREEL
#define NO_FREEL

#include "typedefs.h"

typedef struct freeblk {
	u_long blkno;
	u_long size;
} FreeBlk;

struct freelist {
	u_long		fl_n_blocks;
	FreeBlk *	fl_blockp;
} ;

extern void freeinit(FreeList *list,count_t n_elts,u_long ntotal);
extern long getspace(FreeList *list,u_long s);
extern int givspace(FreeList *list,u_long size,u_long addr);
extern int addspace(FreeList *list,u_long size,u_long addr);
extern void showmap(FreeList *list);
extern int n_map_frags(FreeList *list);

extern int _takespace(QSP_ARG_DECL  FreeList *,u_long,u_long);
#define takespace(flp,l,m) _takespace(QSP_ARG  flp,l,m)

#endif /* NO_FREEL */

