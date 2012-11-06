
#ifndef _NEW_CHAINS_H_
#define _NEW_CHAINS_H_

#include "vecgen.h"

/* globals */

/* BUG should be per-qsp for thread safe */

typedef struct vec_chn_blk {
	void 			(*vcb_func)(Vector_Args *);
	Vector_Args		vcb_args;
	struct vec_chn_blk	*vcb_next;
} Vec_Chn_Blk;

#define NO_VEC_CHN_BLK		((Vec_Chn_Blk *)NULL)

typedef struct chain {
	Item	ch_item;
	List *	ch_lp;
} Chain;

#define ch_name	ch_item.item_name

#define NO_CHAIN	((Chain *)NULL)

#define CH_MAGIC	0x5798
#define MAX_CHAIN_LEN	1024

/* vec_chn.c */

ITEM_INTERFACE_PROTOTYPES(Chain,vec_chain)

extern Chain *load_chain(QSP_ARG_DECL  const char *,const char *);
extern int save_chain(Chain *,FILE *);
extern void dump_chain(Chain *);
extern void terse_dump_chain(Chain *);
extern void chain_info(Chain *);
extern void start_chain(QSP_ARG_DECL  const char *);
extern void end_chain(void);
extern void exec_chain(Chain *);


/* chn_menu.c */
extern COMMAND_FUNC( do_chains );


#endif /* ! _NEW_CHAINS_H_ */

