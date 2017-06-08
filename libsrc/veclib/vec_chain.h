#ifndef _VEC_CHAIN_H_
#define _VEC_CHAIN_H_

#include "veclib/vecgen.h"
#include "veclib/obj_args.h"
#include "item_type.h"
#include "quip_menu.h"

/* globals */

/* BUG should be per-qsp for thread safe */

typedef struct vec_chn_blk {
	void (*			vcb_func)(LINK_FUNC_ARG_DECLS);
	int			vcb_vf_code;
	const Vector_Args *	vcb_args;
	struct vec_chn_blk *	vcb_next;
} Vec_Chn_Blk;


#define CHAIN_FUNC(vcbp)	vcbp->vcb_func
#define CHAIN_ARGS(vcbp)	vcbp->vcb_vf_code, vcbp->vcb_args

/* Chain_Block */
#define NEW_CHAIN_BLOCK(vcp_p,code,func,args)				\
									\
	vcb_p = (Vec_Chn_Blk *)getbuf(sizeof(Vec_Chn_Blk));		\
	vcb_p->vcb_vf_code = code;					\
	vcb_p->vcb_func = func;						\
	vcb_p->vcb_args = args;

#define RELEASE_BLOCK(p)	/* nop */


typedef struct chain {
	Item	ch_item;
	List	*ch_op_lp;
} Chain;


#define CH_MAGIC	0x5798
#define MAX_CHAIN_LEN	1024

#define CHAIN_LIST(cp)		cp->ch_op_lp
#define CHAIN_NAME(cp)		cp->ch_item.item_name
#define SET_CHAIN_NAME(cp,s)	cp->ch_item.item_name = s

/* vec_chn.c */

extern Chain *load_chain(QSP_ARG_DECL  const char *,const char *);
extern int save_chain(Chain *,FILE *);
extern void dump_chain(Chain *);
extern void terse_dump_chain(Chain *);
extern void chain_info(QSP_ARG_DECL  Chain *);
extern void start_chain(QSP_ARG_DECL  const char *);
extern void end_chain(void);
extern void exec_chain(Chain *);

ITEM_CHECK_PROT(Chain, vec_chain)
ITEM_LIST_PROT(Chain, vec_chain)
ITEM_PICK_PROT(Chain, vec_chain)
#define PICK_CHAIN(p)	pick_vec_chain(QSP_ARG   p)



/* chn_menu.c */
extern COMMAND_FUNC( do_chains );


extern Chain *new_vec_chain(const char *name);
extern void del_vec_chain(QSP_ARG_DECL  const char *s);

extern Chain *new_chain( QSP_ARG_DECL  const char *s );
extern int chain_breaks(const char *routine_name);

#endif /* _VEC_CHAIN_H_ */

