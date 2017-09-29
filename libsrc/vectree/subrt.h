#ifndef _SUBRT_H_
#define _SUBRT_H_

#include "item_type.h"
#include "vec_expr_node.h"

#ifdef HAVE_ANY_GPU

#ifdef HAVE_OPENCL
//typedef struct opencl_kernel_info {
//	cl_kernel	kernel[MAX_OPENCL_DEVICES];
//} OpenCL_Kernel_Info;

struct opencl_kernel_info;
typedef struct opencl_kernel_info OpenCL_Kernel_Info;

#endif // HAVE_OPENCL

typedef enum {
	GPU_NONE,
	GPU_OPENCL,
	GPU_CUDA
} GPU_Platform_Type;

typedef union {
#ifdef HAVE_OPENCL
	OpenCL_Kernel_Info *ocl_kernel_info_p;
#endif // HAVE_OPENCL
	void *any_kernel_info_p;
} Kernel_Info_Ptr;

typedef struct gpu_kernel_info {
	GPU_Platform_Type type;
	Kernel_Info_Ptr info_ptr;
} GPU_Kernel_Info;

#endif // HAVE_ANY_GPU

struct subrt {
	Item		sr_item;
	Vec_Expr_Node *	sr_arg_decls;
	Vec_Expr_Node *	sr_arg_vals;
	union {
		Vec_Expr_Node *	u_body;
		const char *	u_text;
	} sr_u;
#define sr_body sr_u.u_body
#define sr_text sr_u.u_text
	int		sr_n_args;
	Shape_Info *	sr_shpp;
	Shape_Info *	sr_dest_shpp;
	List *		sr_ret_lp;
	List *		sr_call_lp;
	Precision *	sr_prec_p;
	int		sr_flags;
	Vec_Expr_Node *	sr_call_enp;

#ifdef HAVE_ANY_GPU
	// stuff for kernel fusion
	GPU_Kernel_Info	sr_gpu_info;
#endif // HAVE_ANY_GPU
};

/* Subrt */

#define NEW_FUNC_PTR		((Subrt *)getbuf(sizeof(Subrt)))

#define SR_DEST_SHAPE(srp)		(srp)->sr_dest_shpp
#define SET_SR_DEST_SHAPE(srp,shpp)	(srp)->sr_dest_shpp = shpp
#define SR_ARG_DECLS(srp)		(srp)->sr_arg_decls
#define SET_SR_ARG_DECLS(srp,enp)	(srp)->sr_arg_decls = enp
#define SR_ARG_VALS(srp)		(srp)->sr_arg_vals
#define SET_SR_ARG_VALS(srp,lp)		(srp)->sr_arg_vals = lp
#define SR_SHAPE(srp)			(srp)->sr_shpp
#define SET_SR_SHAPE(srp,shpp)		(srp)->sr_shpp = shpp
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
#define SR_CALL_VN(srp)			(srp)->sr_call_enp
#define SET_SR_CALL_VN(srp,enp)		(srp)->sr_call_enp = enp

#define SR_PREC_PTR(srp)		(srp)->sr_prec_p
#define SR_PREC_CODE(srp)		PREC_CODE(SR_PREC_PTR(srp))
#define SET_SR_PREC_PTR(srp,p)		(srp)->sr_prec_p = p

/*#define SET_SR_CALL_ENP(srp,enp)	[srp setCall_enp : enp] */
#ifdef HAVE_ANY_GPU
#define SR_GPU_TYPE(srp)		(srp)->sr_gpu_info.type
#define SET_SR_GPU_TYPE(srp,t)		(srp)->sr_gpu_info.type = t
#define SR_GPU_INFO_PTR(srp)		(srp)->sr_gpu_info.info_ptr._kernel_info_p
#define SET_SR_GPU_INFO_PTR(srp,p)	(srp)->sr_gpu_info.info_ptr.any_kernel_info_p = p
#endif // HAVE_ANY_GPU

	

/* flag bits */
#define SR_SCANNING	1
#define SR_SCRIPT	2
#define SR_PROTOTYPE	4
#define SR_REFFUNC	8
#define SR_COMPILED	16
#define SR_FUSED	32	// for at least one platform device...

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

extern String_Buf *fuse_kernel(QSP_ARG_DECL  Vec_Expr_Node *enp);
extern String_Buf *fuse_subrt(QSP_ARG_DECL  Subrt *srp);

#endif // ! _SUBRT_H_

