
#ifdef INC_VERSION
char VersionId_inc_vectree[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */


#ifndef NO_VEXPR_NODE

#include "data_obj.h"
#include "strbuf.h"
#include "query.h"
#include "vecgen.h"

typedef struct keyword {
	const char *	kw_token;
	int		kw_code;
} Keyword;

extern Keyword vt_native_func_tbl[];
extern Keyword ml_native_func_tbl[];
// BUG globals make not thread-safe !!!
extern int parser_lineno;
extern const char *curr_infile;

typedef struct undef_sym {
	const char *	us_name;
} Undef_Sym;


typedef struct context_pair {
	Item_Context *	cp_id_icp;
	Item_Context *	cp_dobj_icp;
} Context_Pair;

#define NO_CONTEXT_PAIR	((Context_Pair *)NULL)



#define EXPECT_PTR_SET	1
#define UNSET_PTR_OK	0

#define NO_UNDEF	((Undef_Sym *)NULL)

#include "treecode.h"


/* fourth child needed for new conditional ops */
#define MAX_NODE_CHILDREN	4

/* BUG  A lot of these elements are only used by a few types of nodes...  we should
 * make better use of unions to conserve memory.
 */

typedef struct decl_node_data {
	const char *	lnd_decl_name;
	prec_t		lnd_decl_prec;		/* why not en_prec ??? */
	Item_Context *	lnd_decl_icp;	/* for declaration nodes */
	List *		lnd_decl_ref_list;
	int		lnd_decl_flags;
} Decl_Node_Data;

/* flag bits */
#define DECL_IS_CONST	1
#define DECL_IS_STATIC	2


typedef struct list_node_data {
	long	list_nd_n_elts;	/* number of list elements */
} List_Node_Data;

typedef struct dbl_node_data {
	double	dnd_dblval;
} Dbl_Node_Data;

typedef struct int_node_data {
	long	ind_intval;
} Int_Node_Data;

typedef struct subrt_node_data {
	struct subrt *	bnd_srp;
} Subrt_Node_Data;

typedef struct callf_node_data {
	struct subrt *	call_nd_srp;
	List *		call_nd_uk_args;
} Callf_Node_Data;

typedef struct string_node_data {
	const char *	snd_string;
} String_Node_Data;

typedef struct cast_node_data {
	prec_t	cnd_cast_prec;
} Cast_Node_Data;

typedef struct vec_func_node_data {
	Vec_Func_Code	vfnd_func_code;
} Vec_Func_Node_Data;

typedef struct sizechng_node_data {
	Shape_Info *	znd_child_shpp;
} Sizechng_Node_Data;

typedef struct bitmap_node_data {
	Vec_Func_Code	bm_nd_bm_code;
	Shape_Info *	bm_nd_child_shpp;
} Bitmap_Node_Data;

typedef struct data_obj_node_data {
	Data_Obj *	dobj_nd_dp;
} Data_Obj_Node_Data;

typedef enum {
	ND_UNUSED,		/* can delete when whole table is set */
	ND_NONE,
	ND_LIST,
	ND_DBL,
	ND_INT,
	ND_SUBRT,
	ND_CALLF,
	ND_STRING,
	ND_CAST,
	ND_FUNC,
	ND_VFUNC,
	ND_DECL,
	ND_SIZE_CHANGE,
	ND_BMAP,
	N_NODE_DATA_TYPES
} Node_Data_Type;

#define VERIFY_DATA_TYPE( enp , type , where )				\
									\
if( tnt_tbl[enp->en_code].tnt_data_type != type ){			\
	sprintf(error_string,						\
"CAUTIOUS:  %s:  %s has data type code %d (%s), expected %d (%s)",	\
	where, node_desc(enp),tnt_tbl[enp->en_code].tnt_data_type,	\
	node_data_type_desc(tnt_tbl[enp->en_code].tnt_data_type),	\
				type,node_data_type_desc(type));	\
	ERROR1(error_string);						\
}

typedef struct tree_node_type {
	Tree_Code	tnt_code;
	const char *		tnt_name;
	short		tnt_nchildren;
	short		tnt_flags;
	Node_Data_Type	tnt_data_type;
} Tree_Node_Type;

/* flag bits */
#define NO_SHP		1
#define PT_SHP		2
#define CP_SHP		4

#define NODE_SHOULD_PT_TO_SHAPE(enp)	(tnt_tbl[enp->en_code].tnt_flags&PT_SHP)
#define NODE_SHOULD_OWN_SHAPE(enp)	(tnt_tbl[enp->en_code].tnt_flags&CP_SHP)

extern Tree_Node_Type tnt_tbl[];

#define NNAME(enp)		tnt_tbl[(enp)->en_code].tnt_name
#define MAX_CHILDREN(enp)	tnt_tbl[(enp)->en_code].tnt_nchildren



typedef struct enode {
	int			en_serial;	/* a number instead of a name */
	Tree_Code		en_code;
	struct enode *		en_parent;
	struct enode *		en_child[MAX_NODE_CHILDREN]; /* put the third child in the union? */
	int			en_flags;
	const char *			en_infile;
	int			en_lineno;	/* line number and file where this node generated */
	List *			en_resolvers;	/* adjacent nodes which can resolve this one */
	Shape_Info *		en_shpp;	/* ptr to shape of this node */
	int			en_lhs_refs;	/* # of refs to lhs target */

	/* this stuff is just for cost computation, not needed by all nodes... */
	uint32_t		en_flops;	/* for cost computation */
	uint32_t		en_nmath;	/* # of math lib calls (for cost computation) */

	union {
		List_Node_Data		u_list_nd;
		String_Node_Data	u_snd;
		Cast_Node_Data		u_cnd;
		Vec_Func_Node_Data	u_vfnd;
		Sizechng_Node_Data	u_znd;
		Subrt_Node_Data		u_bnd;
		Int_Node_Data		u_ind;
		Dbl_Node_Data		u_dnd;
		Decl_Node_Data		u_lnd;
		Callf_Node_Data		u_call_nd;
		Bitmap_Node_Data	u_bm_nd;
		Data_Obj_Node_Data	u_dobj_nd;
	} en_u;

#define en_dp		en_u.u_dobj_nd.dobj_nd_dp
#define en_string	en_u.u_snd.snd_string
#define en_cast_prec	en_u.u_cnd.cnd_cast_prec
#define en_vfunc_code	en_u.u_vfnd.vfnd_func_code
#define en_child_shpp	en_u.u_znd.znd_child_shpp
#define en_srp		en_u.u_bnd.bnd_srp
#define en_intval	en_u.u_ind.ind_intval
#define en_func_index	en_u.u_ind.ind_intval
#define en_dblval	en_u.u_dnd.dnd_dblval
#define en_decl_name	en_u.u_lnd.lnd_decl_name
#define en_decl_prec	en_u.u_lnd.lnd_decl_prec
#define en_decl_icp	en_u.u_lnd.lnd_decl_icp
#define en_decl_ref_list	en_u.u_lnd.lnd_decl_ref_list
#define en_decl_flags	en_u.u_lnd.lnd_decl_flags
#define en_call_srp	en_u.u_call_nd.call_nd_srp
#define en_uk_args	en_u.u_call_nd.call_nd_uk_args
#define en_n_elts	en_u.u_list_nd.list_nd_n_elts
#define en_bm_code	en_u.u_bm_nd.bm_nd_bm_code
#define en_bm_child_shpp	en_u.u_bm_nd.bm_nd_child_shpp

} Vec_Expr_Node;

extern Vec_Expr_Node *last_node;

#define en_prec			en_shpp->si_prec

#define NODE_PREC(enp)		( (enp)->en_shpp == NO_SHAPE ? PREC_VOID : enp->en_shpp->si_prec )

#define COMPLEX_NODE(enp)	( (enp)->en_prec == PREC_CPX || (enp)->en_prec == PREC_DBLCPX )

#define NO_VEXPR_NODE	((Vec_Expr_Node *)NULL)

#define NULL_CHILD( enp , index )		( (enp)->en_child[ index ] == NO_VEXPR_NODE )

#define IS_LITERAL(enp)		(enp->en_code==T_LIT_DBL||enp->en_code==T_LIT_INT)

/* flag bits */
#define NODE_CURDLED			1
#define NODE_IS_SHAPE_OWNER		2
#define NODE_HAS_SHAPE_RESOLVED		4
#define NODE_NEEDS_CALLTIME_RES		8
#define NODE_WAS_WARNED			16
#define NODE_HAS_CONST_VALUE		32

#define HAS_CONSTANT_VALUE(enp)		( (enp)->en_flags & NODE_HAS_CONST_VALUE )

#define CURDLE(enp)		(enp)->en_flags |= NODE_CURDLED;
#define MARK_WARNED(enp)	(enp)->en_flags |= NODE_WAS_WARNED;

/*
#define TRAVERSED	8
#define UK_LEAF		16
#define PRELIM_SHAPE	32
#define NODE_COMPILED	64
#define HAS_UNKNOWN_LEAF(enp)	( ( enp )->en_flags & UK_LEAF )
#define PRELIM_SHAPE_SET(enp)	( ( enp )->en_flags & PRELIM_SHAPE )
#define NODE_IS_COMPILED(enp)	( ( enp )->en_flags & NODE_COMPILED )
#define ALREADY_TRAVERSED(enp)	( ( enp )->en_flags & TRAVERSED )
*/



#define WAS_WARNED(enp)		( ( enp )->en_flags & NODE_WAS_WARNED)
#define IS_CURDLED(enp)		( ( enp )->en_flags & NODE_CURDLED)
#define OWNS_SHAPE(enp)		( ( enp )->en_flags & NODE_IS_SHAPE_OWNER)

/* not the best identifier, since it really means we are either resolved, or must wait??? */
#define IS_RESOLVED(enp)	( ( enp )->en_flags & (NODE_HAS_SHAPE_RESOLVED|NODE_NEEDS_CALLTIME_RES) )

#define RESOLVED_AT_CALLTIME(enp)	( ( enp )->en_flags & NODE_NEEDS_CALLTIME_RES )

#define IS_VECTOR_SHAPE(shpp)	( UNKNOWN_SHAPE(shpp) || (shpp->si_n_type_elts > 1) )
#define IS_VECTOR_NODE(enp)	(enp->en_shpp==NO_SHAPE || IS_VECTOR_SHAPE((enp)->en_shpp))
#define NODE_SHAPE_KNOWN(enp)	( (enp)->en_shpp != NO_SHAPE && (!UNKNOWN_SHAPE(enp->en_shpp)))

#define UNKNOWN_SOMETHING(enp)	( ( (enp)->en_shpp!=NO_SHAPE && UNKNOWN_SHAPE((enp)->en_shpp) ) || HAS_UNKNOWN_LEAF(enp) )

typedef struct subrt {
	Item		sr_item;
#define sr_name		sr_item.item_name

	Vec_Expr_Node *	sr_arg_decls;
	Vec_Expr_Node *	sr_arg_vals;
	Vec_Expr_Node *	sr_body;
	prec_t		sr_prec;	/* really a precision code? ... */
	int		sr_nargs;
	Shape_Info *	sr_shpp;	/* if we know what shape is returned */
	Shape_Info *	sr_dst_shpp;	/* shape we are returning to (dynamic) */
	List *		sr_ret_lp;	/* list of return nodes */
	List *		sr_call_lp;	/* list of callfunc nodes */
	int		sr_flags;
	Vec_Expr_Node *	sr_call_enp;
} Subrt;

/* flag bits */
#define SR_SCANNING	1
#define SR_SCRIPT	2
#define SR_PROTOTYPE	4
#define SR_REFFUNC	8
#define SR_COMPILED	16

#define IS_SCANNING(srp)	( (srp)->sr_flags & SR_SCANNING )
#define IS_SCRIPT(srp)		( (srp)->sr_flags & SR_SCRIPT )
#define IS_REFFUNC(srp)		( (srp)->sr_flags & SR_REFFUNC )
#define IS_COMPILED(srp)	( (srp)->sr_flags & SR_COMPILED )


typedef struct run_info {
	Context_Pair *	ri_prev_cpp;
	int		ri_arg_stat;
	Subrt *		ri_srp;
	Subrt *		ri_old_srp;
} Run_Info;

#define NO_RUN_INFO ((Run_Info *)NULL)

typedef enum {
	OBJ_REFERENCE,
	STR_REFERENCE
} Reference_Type;

typedef struct reference {
	Vec_Expr_Node *		ref_decl_enp;
	struct identifier *	ref_idp;		/* points back to the owning struct */
	Reference_Type		ref_typ;
	union {
		Data_Obj *		u_dp;
		String_Buf *		u_sbp;
	} ref_u;
} Reference;

#define ref_dp	ref_u.u_dp
#define ref_sbp	ref_u.u_sbp

#define NO_REFERENCE	((Reference *)NULL)

#define IS_OBJECT_REF(refp)			((refp)->ref_typ == OBJ_REFERENCE)
#define IS_STRING_REF(refp)			((refp)->ref_typ == STR_REFERENCE)

typedef struct pointer {
	uint32_t	ptr_flags;
	Vec_Expr_Node *	ptr_decl_enp;
	Reference *	ptr_refp;
} Pointer;

#define NO_POINTER	((Pointer *)NULL)

/* We might be cautious and have a flag that says ptr or ref? */

typedef struct funcptr {
	Subrt *		fp_srp;
} Function_Ptr;

#define NO_FUNC_PTR	((Function_Ptr *)NULL)

/* pointer flags */
#define POINTER_SET	1


typedef struct identifier {
	const char *		id_name;
	int		id_type;
	union		{
		/* Data_Obj *	u_dp; */
		Subrt *		u_srp;
		Pointer *	u_ptrp;
		Reference *	u_refp;
		Function_Ptr *	u_fpp;
	} id_u ;
	Shape_Info	id_shape;	/* We keep a copy here, instead of
					 * pointing, so that nodes can point here,
					 * and it is a stable address, even when
					 * the size is changed (matlab)
					 * THis is kind of a waste of memory if
					 * not matlab...
					 */
	Item_Context *	id_dobj_icp;	/* only relevant for ID_REFERENCE */
} Identifier;

#define NO_IDENTIFIER	((Identifier *)NULL)

#define id_fpp		id_u.u_fpp
#define id_ptrp		id_u.u_ptrp
/* #define id_dp		id_u.u_dp */
/*#define id_sbp		id_u.u_sbp */
#define id_refp		id_u.u_refp

/* identifier flags */
typedef enum {
	/* ID_OBJECT, */
	ID_POINTER,
	ID_REFERENCE,
	ID_SUBRT,
	ID_STRING,
	ID_FUNCPTR,
	ID_LABEL
} Id_Type;

#define IS_STRING_ID(idp)	((idp)->id_type == ID_STRING)
#define IS_POINTER(idp)		((idp)->id_type == ID_POINTER)
#define IS_REFERENCE(idp)	((idp)->id_type == ID_REFERENCE)
#define IS_SUBRT(idp)		((idp)->id_type == ID_SUBRT)
/* #define IS_OBJECT(idp)		((idp)->id_type == ID_OBJECT) */
#define IS_FUNCPTR(idp)		((idp)->id_type == ID_FUNCPTR)
#define IS_LABEL(idp)		((idp)->id_type == ID_LABEL)

#define STRING_IS_SET(idp)	((idp)->id_refp->ref_sbp->sb_buf != NULL)
#define POINTER_IS_SET(idp)	((idp)->id_ptrp->ptr_flags & POINTER_SET)


#define NO_SUBRT	((Subrt *)NULL)

/* globals */
extern Subrt *curr_srp;
extern Item_Type *id_itp;
extern int mode_is_matlab;

extern void	(*native_prelim_func)(QSP_ARG_DECL  Vec_Expr_Node *);
extern void	(*native_update_func)(Vec_Expr_Node *);
extern const char *	(*native_string_func)(Vec_Expr_Node *);
extern float	(*native_flt_func)(Vec_Expr_Node *);
extern void	(*native_work_func)(QSP_ARG_DECL  Vec_Expr_Node *);
extern void	(*native_assign_func)(Data_Obj *,Vec_Expr_Node *);

extern int dump_flags;

#define SHOW_SHAPES	1
#define SHOW_COST	2
#define SHOW_LHS_REFS	4
#define SHOW_RES	8
#define SHOW_KEY	16
#define SHOW_ALL	(SHOW_SHAPES|SHOW_COST|SHOW_LHS_REFS|SHOW_RES|SHOW_KEY)

#define SHOWING_SHAPES		(dump_flags & SHOW_SHAPES)
#define SHOWING_COST		(dump_flags & SHOW_COST)
#define SHOWING_LHS_REFS	(dump_flags & SHOW_LHS_REFS)
#define SHOWING_RESOLVERS	(dump_flags & SHOW_RES)







/* globals */
extern int executing;
extern int vecexp_ing;
extern debug_flag_t resolve_debug;
extern debug_flag_t cast_debug;
extern debug_flag_t eval_debug;
extern debug_flag_t scope_debug;
extern debug_flag_t parser_debug;
extern int scanning_args;
extern int dumpit;
extern int dumping;		/* a flag that is set when in dump_tree */


#define eval_ptr_lhs_ref(enp)		eval_ptr_ref(QSP_ARG enp,0)
#define eval_ptr_rhs_ref(enp)		eval_ptr_ref(QSP_ARG enp,1)


#define IS_DECL( code )		( ( code )==T_SCAL_DECL || \
				  ( code )==T_VEC_DECL  || \
				  ( code )==T_IMG_DECL  || \
				  ( code )==T_SEQ_DECL  || \
				  ( code )==T_PTR_DECL	)

/* prototypes */


/* this function may be defined outside the library, but here we
 * define it's form.
 */
extern void init_vt_native_kw_tbl(void);
extern void init_ml_native_kw_tbl(void);


/* subrt.c */

extern void delete_id(TMP_QSP_ARG_DECL  Item *);
extern void pop_subrt_cpair(QSP_ARG_DECL  Context_Pair *cpp,const char *name);
#define POP_SUBRT_CPAIR(cpp,name)	pop_subrt_cpair(QSP_ARG  cpp, name)
extern void dump_subrt(QSP_ARG_DECL  Subrt *);
#define DUMP_SUBRT(srp)		dump_subrt(QSP_ARG  srp)
extern Vec_Expr_Node *find_node_by_number(QSP_ARG_DECL  int);

extern Item_Context *	pop_subrt_ctx(QSP_ARG_DECL  const char *,Item_Type *);
#define POP_SUBRT_CTX(s,itp)	pop_subrt_ctx(QSP_ARG  s, itp)

ITEM_INTERFACE_PROTOTYPES(Identifier,id)
#define ID_OF(s)		id_of(QSP_ARG s)
#define GET_ID(s)		get_id(QSP_ARG s)
ITEM_INTERFACE_PROTOTYPES(Subrt,subrt)
ITEM_INTERFACE_PROTOTYPES(Undef_Sym,undef)
#define UNDEF_OF(s)		undef_of(QSP_ARG  s)

extern Subrt *remember_subrt(QSP_ARG_DECL  prec_t prec,const char *,Vec_Expr_Node *,Vec_Expr_Node *);
extern void update_subrt(QSP_ARG_DECL  Subrt *srp, Vec_Expr_Node *body );
extern COMMAND_FUNC( do_run_subrt );
extern void exec_subrt(QSP_ARG_DECL  Vec_Expr_Node *,Data_Obj *dst_dp);
#define EXEC_SUBRT(enp,dp)		exec_subrt(QSP_ARG enp,dp)
extern void run_subrt(QSP_ARG_DECL  Subrt *srp,Vec_Expr_Node *,Data_Obj *dst_dp);
#define RUN_SUBRT(srp,enp,dst_dp)		run_subrt(QSP_ARG srp,enp,dst_dp)
extern COMMAND_FUNC( do_dump_subrt );
extern COMMAND_FUNC( do_opt_subrt );
extern COMMAND_FUNC( do_subrt_info );

extern void expr_init(void);
extern void expr_file(SINGLE_QSP_ARG_DECL);
extern Subrt *	create_script_subrt(QSP_ARG_DECL  const char *,int,const char *);

extern void list_node(Vec_Expr_Node *);
extern int eval_tree(QSP_ARG_DECL  Vec_Expr_Node *enp,Data_Obj *dst_dp);
#define EVAL_TREE(enp,dst_dp)		eval_tree(QSP_ARG enp,dst_dp)
extern void undeclare_stuff(Vec_Expr_Node *enp);
extern void set_subrt_ctx(QSP_ARG_DECL  const char *name);
extern void delete_subrt_ctx(QSP_ARG_DECL  const char *name);

extern void compare_arg_trees(QSP_ARG_DECL  Vec_Expr_Node *,Vec_Expr_Node *);

/* nodetbl.c */
extern void sort_tree_tbl(void);

/* subrt.c */
extern COMMAND_FUNC( do_tell_cost );

/* vectree.y */

extern Vec_Expr_Node *dup_tree(QSP_ARG_DECL  Vec_Expr_Node *);
#define DUP_TREE(enp)			dup_tree(QSP_ARG  enp)
extern void show_context_stack(QSP_ARG_DECL  Item_Type *);
extern Vec_Expr_Node *node3(QSP_ARG_DECL  Tree_Code,Vec_Expr_Node *,Vec_Expr_Node *,Vec_Expr_Node *);
#define NODE3(code,enp1,enp2,enp3)	node3(QSP_ARG  code,enp1,enp2,enp3)
extern Vec_Expr_Node *node2(QSP_ARG_DECL  Tree_Code,Vec_Expr_Node *,Vec_Expr_Node *);
#define NODE2(code,enp1,enp2)	node2(QSP_ARG  code,enp1,enp2)
extern Vec_Expr_Node *node1(QSP_ARG_DECL  Tree_Code,Vec_Expr_Node *);
#define NODE1(code,enp)	node1(QSP_ARG  code, enp)
extern Vec_Expr_Node *node0(QSP_ARG_DECL  Tree_Code);
#define NODE0(code)	node0(QSP_ARG  code)
extern void rls_tree(Vec_Expr_Node *);
extern void node_error(QSP_ARG_DECL  Vec_Expr_Node *);
#define NODE_ERROR(enp)		node_error(QSP_ARG  enp)
extern void init_expr_node(QSP_ARG_DECL  Vec_Expr_Node *);
extern void set_global_ctx(SINGLE_QSP_ARG_DECL);
extern void unset_global_ctx(SINGLE_QSP_ARG_DECL);

/* dumptree.c */
extern void	set_native_func_tbl(Keyword *);
extern void	set_show_shape(int);
extern void	set_show_lhs_refs(int);
extern void	print_dump_legend(void);
extern void	print_shape_key(void);
extern void	dump_tree(QSP_ARG_DECL  Vec_Expr_Node *);
#define DUMP_TREE(enp)		dump_tree(QSP_ARG  enp)
extern void	dump_node(QSP_ARG_DECL  Vec_Expr_Node *);
#define DUMP_NODE(enp)		dump_node(QSP_ARG  enp)

/* costtree.c */
extern void tell_cost(QSP_ARG_DECL  Subrt *);
extern void cost_tree(QSP_ARG_DECL  Vec_Expr_Node *);

/* comptree.c */

#ifdef CAUTIOUS
extern Shape_Info *product_shape(Shape_Info *,Shape_Info *);
extern void verify_null_shape(QSP_ARG_DECL  Vec_Expr_Node *enp);
#endif /* CAUTIOUS */
extern void shapify(QSP_ARG_DECL   Vec_Expr_Node *enp);
#define SHAPIFY(enp)			shapify(QSP_ARG  enp)
extern Vec_Expr_Node *	nth_arg(QSP_ARG_DECL  Vec_Expr_Node *enp, int n);
#define NTH_ARG(enp,n)	nth_arg(QSP_ARG  enp,n)
extern void		link_uk_nodes(QSP_ARG_DECL  Vec_Expr_Node *,Vec_Expr_Node *);
#define LINK_UK_NODES(enp1,enp2)	link_uk_nodes(QSP_ARG  enp1,enp2)
extern void		update_tree_shape(QSP_ARG_DECL  Vec_Expr_Node *);
#define UPDATE_TREE_SHAPE(enp)		update_tree_shape(QSP_ARG  enp)
extern void		prelim_tree_shape(Vec_Expr_Node *);
extern void		compile_tree(QSP_ARG_DECL  Vec_Expr_Node *);
#define COMPILE_TREE(enp)		compile_tree(QSP_ARG enp)
extern Vec_Expr_Node *	compile_prog(QSP_ARG_DECL  Vec_Expr_Node **);
#define COMPILE_PROG(enp)		compile_prog(QSP_ARG enp)
extern void		compile_subrt(QSP_ARG_DECL Subrt *);
#define COMPILE_SUBRT(srp)		compile_subrt(QSP_ARG srp)
extern Shape_Info *	scalar_shape(prec_t);
extern Shape_Info *	uk_shape(prec_t);
extern int		shapes_match(Shape_Info *,Shape_Info *);
extern void		init_fixed_nodes(SINGLE_QSP_ARG_DECL);
extern void		copy_node_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,Shape_Info *shpp);
#define COPY_NODE_SHAPE( enp, shpp )	copy_node_shape(QSP_ARG enp , shpp )
extern void		discard_node_shape(Vec_Expr_Node *);
extern const char *	get_lhs_name(QSP_ARG_DECL Vec_Expr_Node *enp);
#define GET_LHS_NAME(enp)	get_lhs_name(QSP_ARG enp)
extern Shape_Info *	calc_outer_shape(Vec_Expr_Node *,Vec_Expr_Node *);


/* scantree.c */

extern void		xform_to_bitmap(Shape_Info *);
extern void		xform_from_bitmap(Shape_Info *,prec_t);
extern void		resolve_argval_shapes(QSP_ARG_DECL Vec_Expr_Node *,Vec_Expr_Node *,Subrt *);
#define RESOLVE_ARGVAL_SHAPES(enp1,enp2,srp)	resolve_argval_shapes(QSP_ARG enp1,enp2,srp)
extern void		resolve_pointer(QSP_ARG_DECL  Vec_Expr_Node *,Shape_Info *);
#define RESOLVE_POINTER(enp,shpp)		resolve_pointer(QSP_ARG enp,shpp)
extern void propagate_shape(QSP_ARG_DECL  Vec_Expr_Node *enp, Shape_Info *shpp);
#define PROPAGATE_SHAPE(enp,shpp)		propagate_shape(QSP_ARG enp,shpp)
extern Vec_Expr_Node *	effect_resolution(QSP_ARG_DECL Vec_Expr_Node *,Vec_Expr_Node *);
#define EFFECT_RESOLUTION(enp1,enp2)		effect_resolution(QSP_ARG enp1,enp2)

extern void	resolve_runtime_shapes(Vec_Expr_Node *);
extern void	resolve_from_rhs(Vec_Expr_Node *);
extern void	report_uk_shapes(Subrt *);
extern int	check_uk_shapes(Subrt *srp);
extern int	check_arg_shapes(QSP_ARG_DECL  Vec_Expr_Node *arg,Vec_Expr_Node *valp,Subrt *srp);
#define CHECK_ARG_SHAPES(arg,valp,srp)		check_arg_shapes(QSP_ARG arg,valp,srp)


extern void		check_resolution(Subrt *srp);
extern void		point_node_shape(QSP_ARG_DECL  Vec_Expr_Node *,Shape_Info *);
#define POINT_NODE_SHAPE( enp , sip )	point_node_shape(QSP_ARG  enp , sip )
extern int		decl_count(QSP_ARG_DECL  Vec_Expr_Node *);
extern int		arg_count(Vec_Expr_Node *);
extern void		resolve_subrt(QSP_ARG_DECL  Subrt *,List *uk_list,Vec_Expr_Node *val_enp,
						Shape_Info *ret_shpp);
#define RESOLVE_SUBRT(srp,uk_list,val_enp,ret_shpp)	resolve_subrt(QSP_ARG srp,uk_list,val_enp,ret_shpp)


/* resolve.c */

extern Vec_Expr_Node *	resolve_node(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Shape_Info *shpp);
#define RESOLVE_NODE(uk_enp,shpp)		resolve_node(QSP_ARG uk_enp,shpp)
extern void		forget_resolved_shapes(QSP_ARG_DECL  Subrt *srp);
extern void		forget_resolved_tree(QSP_ARG_DECL  Vec_Expr_Node *enp);
extern void		resolve_tree(QSP_ARG_DECL  Vec_Expr_Node *enp,Vec_Expr_Node *whence);
#define RESOLVE_TREE(enp,whence)		resolve_tree(QSP_ARG enp,whence)
extern void		late_calltime_resolve(QSP_ARG_DECL  Subrt *srp, Data_Obj *dst_dp);
#define LATE_CALLTIME_RESOLVE(srp,dst_dp)	late_calltime_resolve(QSP_ARG srp,dst_dp)
extern void		early_calltime_resolve(QSP_ARG_DECL  Subrt *srp, Data_Obj *dst_dp);
#define EARLY_CALLTIME_RESOLVE(srp,dst_dp)	early_calltime_resolve(QSP_ARG srp,dst_dp)

/* ml_supp.c */
void insure_object_size(QSP_ARG_DECL  Data_Obj *dp,index_t index);

/* evaltree.c */

extern Data_Obj *	make_local_dobj(QSP_ARG_DECL  Dimension_Set *,prec_t);
extern int		zero_dp(QSP_ARG_DECL  Data_Obj *);
extern Data_Obj *	mlab_reshape(QSP_ARG_DECL  Data_Obj *,Shape_Info *,const char *);
extern void		eval_immediate(QSP_ARG_DECL  Vec_Expr_Node *enp);
#define EVAL_IMMEDIATE(enp)		eval_immediate(QSP_ARG enp)
extern void		wrapup_context(QSP_ARG_DECL  Run_Info *rip);
extern Run_Info *	setup_call(QSP_ARG_DECL  Subrt *srp,Data_Obj *dst_dp);
#define SETUP_CALL(srp,dst_dp)	setup_call(QSP_ARG srp,dst_dp)
extern Subrt *		runnable_subrt(QSP_ARG_DECL  Vec_Expr_Node *enp);
extern Identifier *	make_named_reference(QSP_ARG_DECL  const char *name);
extern Identifier *	get_set_ptr(QSP_ARG_DECL Vec_Expr_Node *);
#define GET_SET_PTR(enp)		get_set_ptr(QSP_ARG enp)
extern Data_Obj *	eval_obj_ref(QSP_ARG_DECL  Vec_Expr_Node *);
#define EVAL_OBJ_REF(enp)		eval_obj_ref(QSP_ARG enp)
extern Data_Obj *	eval_obj_exp(QSP_ARG_DECL  Vec_Expr_Node *,Data_Obj *);
#define EVAL_OBJ_EXP(enp,dp)		eval_obj_exp(QSP_ARG enp,dp)
extern Context_Pair *	pop_previous(SINGLE_QSP_ARG_DECL);
#define POP_PREVIOUS()	pop_previous(SINGLE_QSP_ARG)	
extern void		restore_previous(QSP_ARG_DECL  Context_Pair *);
#define RESTORE_PREVIOUS(cpp)	restore_previous(QSP_ARG cpp)
extern Identifier *	eval_ptr_ref(QSP_ARG_DECL  Vec_Expr_Node *enp,int expect_ptr_set);
#define EVAL_PTR_REF(enp,expect_ptr_set)	eval_ptr_ref(QSP_ARG enp,expect_ptr_set)
extern char *		node_desc(Vec_Expr_Node *);
extern void		reeval_decl_stat(QSP_ARG_DECL  prec_t prec,Vec_Expr_Node *,int ro);
extern const char *	eval_string(QSP_ARG_DECL Vec_Expr_Node *);
#define EVAL_STRING(enp)			eval_string(QSP_ARG enp)
extern void		missing_case(QSP_ARG_DECL  Vec_Expr_Node *,const char *);
#define MISSING_CASE(enp,str)			missing_case(QSP_ARG  enp,str)
extern long		eval_int_exp(QSP_ARG_DECL Vec_Expr_Node *);
#define EVAL_INT_EXP(enp)	eval_int_exp(QSP_ARG enp)
extern double		eval_flt_exp(QSP_ARG_DECL Vec_Expr_Node *);
#define EVAL_FLT_EXP(enp)	eval_flt_exp(QSP_ARG enp)
extern void		eval_decl_tree(QSP_ARG_DECL  Vec_Expr_Node *);
#define EVAL_DECL_TREE(enp)		eval_decl_tree(QSP_ARG enp)

/* opt_tree.c */

extern void optimize_subrt(QSP_ARG_DECL  Subrt *);
#define OPTIMIZE_SUBRT(srp)		optimize_subrt(QSP_ARG  srp)

/* vt_menu.c */

extern void vt_init(SINGLE_QSP_ARG_DECL);

/* vecnodes.c */

extern const char *node_data_type_desc(Node_Data_Type t);


#endif /* ! NO_VEXPR_NODE */

