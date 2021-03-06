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
#define CHAIN_ARGS(vcbp)	QSP_ARG  vcbp->vcb_vf_code, vcbp->vcb_args

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

#define VEC_CHAIN_NAME(vcp)	(vcp)->ch_item.item_name
#define VEC_CHAIN_LIST(vcp)	(vcp)->ch_op_lp

#define SET_VEC_CHAIN_LIST(vcp,v)	(vcp)->ch_op_lp = v

#define CH_MAGIC	0x5798
#define MAX_CHAIN_LEN	1024

#define CHAIN_LIST(cp)		cp->ch_op_lp
#define CHAIN_NAME(cp)		cp->ch_item.item_name
#define SET_CHAIN_NAME(cp,s)	cp->ch_item.item_name = s

/* vec_chn.c */

extern int save_chain(Chain *,FILE *);
extern void dump_chain(Chain *);
extern void terse_dump_chain(Chain *);

extern Chain * _load_chain(QSP_ARG_DECL  const char *,const char *);
extern void _chain_info(QSP_ARG_DECL  Chain *);
extern void _start_chain(QSP_ARG_DECL  const char *);
extern void _exec_chain(QSP_ARG_DECL  Chain *);

#define load_chain(s1,s2) _load_chain(QSP_ARG  s1,s2)
#define chain_info(cp) _chain_info(QSP_ARG  cp)
#define start_chain(s) _start_chain(QSP_ARG  s)
#define exec_chain(chp) _exec_chain(QSP_ARG  chp)

extern void _end_chain(SINGLE_QSP_ARG_DECL);
#define end_chain() _end_chain(SINGLE_QSP_ARG)

ITEM_CHECK_PROT(Chain, vec_chain)
ITEM_LIST_PROT(Chain, vec_chain)
ITEM_PICK_PROT(Chain, vec_chain)
ITEM_NEW_PROT(Chain, vec_chain)
ITEM_DEL_PROT(Chain, vec_chain)

#define vec_chain_of(s)		_vec_chain_of(QSP_ARG  s)
#define list_vec_chains(fp)	_list_vec_chains(QSP_ARG  fp)
#define pick_vec_chain(p)	_pick_vec_chain(QSP_ARG   p)
#define new_vec_chain(s)	_new_vec_chain(QSP_ARG  s)
#define del_vec_chain(p)	_del_vec_chain(QSP_ARG   p)

/* chn_menu.c */
extern COMMAND_FUNC( do_chains );

extern Chain *new_chain( QSP_ARG_DECL  const char *s );

extern int _chain_breaks(QSP_ARG_DECL  const char *routine_name);
#define chain_breaks(routine_name) _chain_breaks(QSP_ARG  routine_name)

#endif /* _VEC_CHAIN_H_ */

