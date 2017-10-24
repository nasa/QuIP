#ifndef _SUBRT_H_
#define _SUBRT_H_

#include "item_type.h"
#include "vec_expr_node.h"
#include "platform.h"

struct subrt {
	Item		sr_item;
	Vec_Expr_Node *	sr_arg_decls;
	union {
		Vec_Expr_Node *	u_body;
		const char *	u_text;
	} sr_u;
#define sr_body sr_u.u_body
#define sr_text sr_u.u_text
	int		sr_n_args;
	List *		sr_ret_lp;
	List *		sr_call_lp;
	Precision *	sr_prec_p;
	int		sr_flags;

	// stuff for kernel fusion
	Kernel_Info_Ptr	sr_kernel_info_p[N_PLATFORM_TYPES];
};

/* Subrt */

#define NEW_FUNC_PTR		((Subrt *)getbuf(sizeof(Subrt)))

#define SR_ARG_DECLS(srp)		(srp)->sr_arg_decls
#define SET_SR_ARG_DECLS(srp,enp)	(srp)->sr_arg_decls = enp
#define SR_BODY(srp)			(srp)->sr_body
#define SR_TEXT(srp)			(srp)->sr_text
#define SET_SR_BODY(srp,enp)		(srp)->sr_body = enp
#define SET_SR_TEXT(srp,s)		(srp)->sr_text = s
#define SR_ARG_DECLS(srp)		(srp)->sr_arg_decls
#define SR_FLAGS(srp)			(srp)->sr_flags
#define SET_SR_FLAG_BITS(srp,b)		(srp)->sr_flags |= b
#define CLEAR_SR_FLAG_BITS(srp,b)	(srp)->sr_flags &= ~(b)
#define SET_SR_FLAGS(srp,f)		(srp)->sr_flags = f
#define SR_NAME(srp)			(srp)->sr_item.item_name
#define SR_N_ARGS(srp)			(srp)->sr_n_args
#define SET_SR_N_ARGS(srp,n)		(srp)->sr_n_args = n
#define SR_RET_LIST(srp)		(srp)->sr_ret_lp
#define SET_SR_RET_LIST(srp,lp)		(srp)->sr_ret_lp = lp
#define SR_CALL_LIST(srp)		(srp)->sr_call_lp
#define SET_SR_CALL_LIST(srp,lp)	(srp)->sr_call_lp = lp

#define SR_PREC_PTR(srp)		(srp)->sr_prec_p
#define SR_PREC_CODE(srp)		PREC_CODE(SR_PREC_PTR(srp))
#define SET_SR_PREC_PTR(srp,p)		(srp)->sr_prec_p = p

/*#define SET_SR_CALL_ENP(srp,enp)	[srp setCall_enp : enp] */

#define SR_KERNEL_INFO_PTR(srp,idx)		(srp)->sr_kernel_info_p[idx]
#define SR_KERNEL_INFO_PTR_ADDR(srp,idx)	(&((srp)->sr_kernel_info_p[idx]))
#define SET_SR_KERNEL_INFO_PTR(srp,idx,p)	(srp)->sr_kernel_info_p[idx].any_kernel_info_p = p

	

/* flag bits */
#define SR_SCANNING	1
#define SR_SCRIPT	2
#define SR_PROTOTYPE	4
#define SR_REFFUNC	8
#define SR_COMPILED	16
// we had a FUSED flag, but makes no sense because can be fused independently
// for different platorms & devices...

#define IS_SCANNING(srp)	( SR_FLAGS(srp) & SR_SCANNING )
#define IS_SCRIPT(srp)		( SR_FLAGS(srp) & SR_SCRIPT )
#define IS_REFFUNC(srp)		( SR_FLAGS(srp) & SR_REFFUNC )
#define IS_COMPILED(srp)	( SR_FLAGS(srp) & SR_COMPILED )

ITEM_INIT_PROT(Subrt,subrt)
ITEM_CHECK_PROT(Subrt,subrt)
ITEM_NEW_PROT(Subrt,subrt)
ITEM_PICK_PROT(Subrt,subrt)
ITEM_ENUM_PROT(Subrt,subrt)
ITEM_LIST_PROT(Subrt,subrt)

//extern Subrt *subrt_of(QSP_ARG_DECL  const char *name);
//extern Subrt *new_subrt(QSP_ARG_DECL  const char *name);
//extern Subrt *pick_subrt(QSP_ARG_DECL const char *prompt);

#define PICK_SUBRT(p)	pick_subrt(QSP_ARG  p)

extern Item_Type *subrt_itp;

struct subrt_call {
	struct subrt *	sc_srp;
	Vec_Expr_Node *	sc_arg_vals;
	Vec_Expr_Node *	sc_call_enp;
	Shape_Info *	sc_shpp;
	Shape_Info *	sc_dest_shpp;
} ;

#define SC_SUBRT(scp)			(scp)->sc_srp
#define SET_SC_SUBRT(scp,srp)		(scp)->sc_srp = srp
#define SC_ARG_VALS(scp)		(scp)->sc_arg_vals
#define SET_SC_ARG_VALS(scp,lp)		(scp)->sc_arg_vals = lp
#define SC_CALL_VN(scp)			(scp)->sc_call_enp
#define SET_SC_CALL_VN(scp,enp)		(scp)->sc_call_enp = enp
#define SC_SHAPE(scp)			(scp)->sc_shpp
#define SET_SC_SHAPE(scp,shpp)		(scp)->sc_shpp = shpp
#define SC_DEST_SHAPE(scp)		(scp)->sc_dest_shpp
#define SET_SC_DEST_SHAPE(scp,shpp)	(scp)->sc_dest_shpp = shpp

extern void fuse_kernel(QSP_ARG_DECL  Vec_Expr_Node *enp);
extern void fuse_subrt(QSP_ARG_DECL  Subrt *srp);
extern Subrt_Call *make_call_instance(Subrt *srp);
extern void * find_fused_kernel(QSP_ARG_DECL  Subrt *srp, Platform_Device *pdp);
extern void run_fused_kernel(QSP_ARG_DECL  Subrt_Call *scp, void * kp, Platform_Device *pdp);
extern void update_pfdev_from_children(QSP_ARG_DECL  Vec_Expr_Node *enp);

#ifdef MAX_DEBUG
extern void dump_resolvers(Vec_Expr_Node *enp);
#define DUMP_RESOLVERS(n)	dump_resolvers(n)
#else // ! MAX_DEBUG
#define DUMP_RESOLVERS(n)
#endif // ! MAX_DEBUG

#endif // ! _SUBRT_H_

