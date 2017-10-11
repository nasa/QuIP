#ifndef _VEC_EXPR_NODE_H_
#define _VEC_EXPR_NODE_H_

#include "treecode.h"
#include "veclib/vecgen.h"
#include "function.h"
#include "string_ref.h"

#define MAX_NODE_CHILDREN	4	// when did we start needing 4???

struct subrt;

typedef struct list_node_data {
	long		n_elts;	/* number of list elements */
} List_Node_Data;

typedef struct dbl_node_data {
	double		dblval;
} Dbl_Node_Data;

typedef struct int_node_data {
	long		intval;
} Int_Node_Data;

typedef struct subrt_node_data {
	struct subrt *	srp;
} Subrt_Node_Data;


typedef struct callf_node_data {
	struct subrt_call *	scp;
	List *		uk_args_lp;
} Callf_Node_Data;

typedef struct string_node_data {
	const char *	string;
} String_Node_Data;

typedef struct cast_node_data {
	Precision *	prec_p;
} Cast_Node_Data;

typedef struct vec_func_node_data {
	// BUG crashes when this order didn't match bitmap_node_data...
	Quip_Function *	vf_func_p;
	int		vf_code;
} Vec_Func_Node_Data;

typedef struct sizechng_node_data {
	Shape_Info *	shpp;
} Sizechng_Node_Data;

typedef struct bitmap_node_data {
	Vec_Func_Code	code;
	Shape_Info *	shpp;		/* was en_bm_child_shape ???  why child? */
} Bitmap_Node_Data;

typedef struct data_obj_node_data {
	Data_Obj *	dp;
} Data_Obj_Node_Data;


/* BUG  A lot of these elements are only used by a few types of nodes...  we should
 * make better use of unions to conserve memory.
 */

typedef struct decl_node_data {
	const char *	decl_name;
	Precision *	decl_prec_p;
	Item_Context *	decl_icp;		/* for declaration nodes */
	List *		decl_ref_list;
	int		decl_flags;
	Data_Obj *	decl_dp;
} Decl_Node_Data;


typedef union {
	List_Node_Data		list_data;
	String_Node_Data	string_data;
	Cast_Node_Data		cast_data;
	Vec_Func_Node_Data	vf_data;
	Sizechng_Node_Data	sizch_data;
	Subrt_Node_Data		subrt_data;
	Int_Node_Data		int_data;
	Dbl_Node_Data		dbl_data;
	Decl_Node_Data		decl_data;
	Callf_Node_Data		call_data;
	Bitmap_Node_Data	bitmap_data;
	Data_Obj_Node_Data	dobj_data;
} Node_Data;

typedef enum {
	ND_UNUSED,		/* 0	can delete when whole table is set */
	ND_NONE,		/* 1  */
	ND_LIST,		/* 2  */
	ND_STRING,		/* 3  */
	ND_CAST,		/* 4  */
	ND_VFUNC,		/* 5  */
	ND_SIZE_CHANGE,		/* 6  */
	ND_SUBRT,		/* 7  */
	ND_INT,			/* 8  */
	ND_DBL,			/* 9  */
	ND_DECL,		/* 10 */
	ND_CALLF,		/* 11 */
	ND_BMAP,		/* 12 */
	ND_FUNC,		/* 13   this doesn't appear in the union!?  BUG??? */
	N_NODE_DATA_TYPES	/* 14 */
} Node_Data_Type;

struct vec_expr_node {
	struct vec_expr_node *		ven_child[MAX_NODE_CHILDREN];
	int				ven_serial;
	int				ven_flags;
	Tree_Code			ven_code;
	struct vec_expr_node *		ven_parent;
	String_Ref *			ven_infile;
	int				ven_lineno;
	List *				ven_resolvers;
	Shape_Info *			ven_shpp;
	int				ven_n_lhs_refs;
	uint32_t			ven_flops;
	uint32_t			ven_nmath;
	Node_Data			ven_data;
#ifdef HAVE_ANY_GPU
	Platform_Device *		ven_pdp;
#endif // HAVE_ANY_GPU
} ;

#define INIT_ENODE_PTR(enp)		enp=((Vec_Expr_Node *)getbuf(sizeof(Vec_Expr_Node)));

/* VecExprNode */
#define VN_PFDEV(enp)		(enp)->ven_pdp
#define SET_VN_PFDEV(enp,pdp)	(enp)->ven_pdp = pdp
#define VN_FLOPS(enp)		(enp)->ven_flops
#define SET_VN_FLOPS(enp,n)	(enp)->ven_flops = n
#define VN_INFILE(enp)		(enp)->ven_infile
#define SET_VN_INFILE(enp,s)	(enp)->ven_infile = s
#define VN_CODE(enp)		(enp)->ven_code
#define SET_VN_CODE(enp,c)	(enp)->ven_code = c
#define VN_N_MATH(enp)		(enp)->ven_nmath
#define SET_VN_N_MATH(enp,n)	(enp)->ven_nmath = n
#define VN_SERIAL(enp)		(enp)->ven_serial
#define SET_VN_SERIAL(enp,n)	(enp)->ven_serial = n
#define VN_LINENO(enp)		(enp)->ven_lineno
#define SET_VN_LINENO(enp,n)	(enp)->ven_lineno = n
#define VN_LHS_REFS(enp)	(enp)->ven_n_lhs_refs
#define SET_VN_LHS_REFS(enp,n)	(enp)->ven_n_lhs_refs = n

#define VN_PARENT(enp)		(enp)->ven_parent
#define VN_RESOLVERS(enp)	(enp)->ven_resolvers
#define VN_CHILD(enp,idx)	(enp)->ven_child[idx]
#define VN_PREC(enp)		SHP_PREC(VN_SHAPE(enp))
#define VN_PREC_PTR(enp)	SHP_PREC_PTR(VN_SHAPE(enp))
#define SET_VN_PREC(enp,p)	SET_SHP_PREC_PTR(VN_SHAPE(enp),PREC_FOR_CODE(p))
#define VN_CHILD_SHAPE(enp,idx)	VN_SHAPE(VN_CHILD(enp,idx))
#define VN_CHILD_PREC(enp,idx)	SHP_PREC(VN_CHILD_SHAPE(enp,idx))
#define VN_CHILD_PREC_PTR(enp,idx)	SHP_PREC_PTR(VN_CHILD_SHAPE(enp,idx))
#define VN_FLAGS(enp)		(enp)->ven_flags
#define VN_SHAPE(enp)		(enp)->ven_shpp
#define SET_VN_SHAPE(enp,shpp)	(enp)->ven_shpp =  shpp
#define COPY_VN_SHAPE(enp,shpp)	COPY_SHAPE(VN_SHAPE(enp),shpp)
#define SET_VN_CHILD_PREC(enp,idx,p)	SET_SHP_PREC_PTR(VN_CHILD_SHAPE(enp,idx),PREC_FOR_CODE(p))

#define SET_VN_CHILD(enp,idx,c)	(enp)->ven_child[idx] = c
#define SET_VN_PARENT(enp,p)	(enp)->ven_parent = p
#define SET_VN_RESOLVERS(enp,r)	(enp)->ven_resolvers = r

#define SET_VN_FLAGS(enp,f)	(enp)->ven_flags = f
#define SET_VN_FLAG_BITS(enp,f)	(enp)->ven_flags |= f
#define CLEAR_VN_FLAG_BITS(enp,f)	(enp)->ven_flags &= ~(f)
#define VN_CAST_PREC_CODE(enp)	PREC_CODE(VN_CAST_PREC_PTR(enp))


#define VN_DATA(enp,member)		(((enp)->ven_data).member)

#define VN_DECL_NAME(enp)	VN_DATA(enp,decl_data.decl_name)
#define VN_DECL_CTX(enp)	VN_DATA(enp,decl_data.decl_icp)
#define VN_DECL_PREC(enp)	VN_DATA(enp,decl_data.decl_prec_p)
#define VN_DECL_REFS(enp)	VN_DATA(enp,decl_data.decl_ref_list)
#define VN_DECL_FLAGS(enp)	VN_DATA(enp,decl_data.decl_flags)
#define VN_DECL_OBJ(enp)	VN_DATA(enp,decl_data.decl_dp)

#define VN_DECL_PREC_CODE(enp)	PREC_CODE(VN_DECL_PREC(enp))

#define VN_OBJ(enp)		((enp)->ven_data).dobj_data.dp
#define VN_SUBRT_CALL(enp)	((enp)->ven_data).call_data.scp
#define VN_SUBRT(enp)		((enp)->ven_data).subrt_data.srp

#define VN_STRING(enp)		((enp)->ven_data).string_data.string
#define VN_N_ELTS(enp)		((enp)->ven_data).list_data.n_elts
#define VN_FUNC_PTR(enp)	((enp)->ven_data).vf_data.vf_func_p
#define VN_UK_ARGS(enp)		((enp)->ven_data).call_data.uk_args_lp
#define VN_SIZCH_SHAPE(enp)	((enp)->ven_data).sizch_data.shpp
#define VN_CAST_PREC_PTR(enp)	((enp)->ven_data).cast_data.prec_p
#define VN_VFUNC_CODE(enp)	((enp)->ven_data).vf_data.vf_code
#define VN_BM_SHAPE(enp)	((enp)->ven_data).bitmap_data.shpp
#define VN_BM_CODE(enp)		((enp)->ven_data).bitmap_data.code
#define VN_INTVAL(enp)		((enp)->ven_data).int_data.intval
#define VN_DBLVAL(enp)		((enp)->ven_data).dbl_data.dblval


#define SET_VN_DATA(enp,member,val)	VN_DATA(enp,member) = val
//#define SET_VN_DATA(enp,member,val)	{ fprintf(stderr,"SET_VN_DATA( 0x%lx, %s, %s )\n",(long)enp,#member,#val); VN_DATA(enp,member) = val; }

#define SET_VN_SIZCH_SHAPE(enp,_shpp)	SET_VN_DATA(enp,sizch_data.shpp,_shpp)
#define SET_VN_CAST_PREC_PTR(enp,p)	SET_VN_DATA(enp,cast_data.prec_p,p)
#define SET_VN_VFUNC_CODE(enp,c)	SET_VN_DATA(enp,vf_data.vf_code,c)
#define SET_VN_BM_SHAPE(enp,_shpp)	SET_VN_DATA(enp,bitmap_data.shpp,_shpp)
#define SET_VN_BM_CODE(enp,c)		SET_VN_DATA(enp,bitmap_data.code,c)
#define SET_VN_INTVAL(enp,v)		SET_VN_DATA(enp,int_data.intval,v)
#define SET_VN_DBLVAL(enp,d)		SET_VN_DATA(enp,dbl_data.dblval,d)
#define SET_VN_OBJ(enp,_dp)		SET_VN_DATA(enp,dobj_data.dp, _dp)
#define SET_VN_SUBRT_CALL(enp,_scp)	SET_VN_DATA(enp,call_data.scp,_scp)
#define SET_VN_SUBRT(enp,_srp)		SET_VN_DATA(enp,subrt_data.srp,_srp)

#define SET_VN_DECL_NAME(enp,s)		SET_VN_DATA(enp,decl_data.decl_name,s)
#define SET_VN_DECL_CTX(enp,icp)	SET_VN_DATA(enp,decl_data.decl_icp,icp)
#define SET_VN_DECL_PREC(enp,p)		SET_VN_DATA(enp,decl_data.decl_prec_p,p)
#define SET_VN_DECL_REFS(enp,lp)	SET_VN_DATA(enp,decl_data.decl_ref_list,lp)
#define SET_VN_DECL_FLAGS(enp,f)	SET_VN_DATA(enp,decl_data.decl_flags,f)
#define SET_VN_DECL_OBJ(enp,dp)		SET_VN_DATA(enp,decl_data.decl_dp,dp)

#define SET_VN_STRING(enp,s)		SET_VN_DATA(enp,string_data.string,s)
#define SET_VN_N_ELTS(enp,v)		SET_VN_DATA(enp,list_data.n_elts,v)
#define SET_VN_FUNC_PTR(enp,v)		SET_VN_DATA(enp,vf_data.vf_func_p,v)
#define SET_VN_UK_ARGS(enp,lp)		SET_VN_DATA(enp,call_data.uk_args_lp,lp)

#define SET_VN_DECL_FLAG_BITS(enp,b)	SET_VN_DATA(enp,decl_data.decl_flags,VN_DECL_FLAGS(enp)|b)

#define VN_DATA_TYPE(enp)		tnt_tbl[VN_CODE(enp)].tnt_data_type


//#ifdef FOOBAR
#define DEBUG_IT_1(enp,msg)	{ DEBUG_IT(enp,msg) last_debugged_enp=enp; }
#define DEBUG_IT_3(enp,msg)	{ DEBUG_IT_1(enp,msg) DEBUG_IT_2(msg) }

#define DEBUG_IT(enp,msg)	{ fprintf(stderr,"%s - node 0x%lx  child[0] = 0x%lx\n",#msg,(long)enp,(long)VN_CHILD(enp,0)); }

extern Vec_Expr_Node *special_enp;
extern Vec_Expr_Node *last_debugged_enp;
#define DEBUG_IT_2(msg)	if( special_enp != NULL ) { DEBUG_IT(special_enp,msg) }
//#else // ! FOOBAR
//#define DEBUG_IT_3(enp,msg)
//#endif // FOOBAR

#endif /* ! _VEC_EXPR_NODE_H_ */

