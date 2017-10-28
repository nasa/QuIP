#ifndef _VECTREE_H_
#define _VECTREE_H_

#include "data_obj.h"
//#include "strbuf.h"
#include "veclib/vecgen.h"
#include "vec_expr_node.h"
//#include "subrt.h"
#include "identifier.h"
#include "pointer.h"
#include "ctx_pair.h"

typedef struct keyword {
	const char *	kw_token;
	int		kw_code;
} Keyword;

extern Keyword vt_native_func_tbl[];
extern Keyword ml_native_func_tbl[];



#define EXPECT_PTR_SET	1
#define UNSET_PTR_OK	0

#include "treecode.h"


/* flag bits */
#define DECL_IS_CONST	1
#define DECL_IS_STATIC	2


//#ifdef NOW_PERFORMED_BY_ASSERTION

#define VERIFY_DATA_TYPE( enp , type , where )				\
									\
if( VN_DATA_TYPE(enp) != type ){					\
	sprintf(ERROR_STRING,						\
"CAUTIOUS:  %s:  %s has data type code %d (%s), expected %d (%s)",	\
	where, node_desc(enp),VN_DATA_TYPE(enp),			\
	node_data_type_desc(VN_DATA_TYPE(enp)),	\
				type,node_data_type_desc(type));	\
	error1(ERROR_STRING);						\
}
//#endif // NOW_PERFORMED_BY_ASSERTION

#define ASSERT_NODE_DATA_TYPE( enp, t )	assert( VN_DATA_TYPE(enp) == t );

typedef struct tree_node_type {
	Tree_Code	tnt_code;
	const char *	tnt_name;
	short		tnt_nchildren;
	short		tnt_flags;
	Node_Data_Type	tnt_data_type;
} Tree_Node_Type;

/* flag bits */
#define NO_SHP		1
#define PT_SHP		2
#define CP_SHP		4

#define NODE_SHOULD_PT_TO_SHAPE(enp)	(tnt_tbl[VN_CODE(enp)].tnt_flags&PT_SHP)
#define NODE_SHOULD_OWN_SHAPE(enp)	(tnt_tbl[VN_CODE(enp)].tnt_flags&CP_SHP)

extern Tree_Node_Type tnt_tbl[];

#define NNAME(enp)		tnt_tbl[VN_CODE((enp))].tnt_name
#define MAX_CHILDREN(enp)	tnt_tbl[VN_CODE((enp))].tnt_nchildren


#ifdef FOOBAR
// not thread safe
extern Vec_Expr_Node *last_node;
#endif // FOOBAR


#define NODE_PREC(enp)		( VN_SHAPE(enp) == NULL ? PREC_VOID : VN_PREC(enp) )

#define COMPLEX_NODE(enp)	( VN_PREC(enp) == PREC_CPX || VN_PREC(enp) == PREC_DBLCPX )

#define NULL_CHILD( enp , index )		( VN_CHILD(enp,index) == NULL )

#define IS_LITERAL(enp)		(VN_CODE(enp)==T_LIT_DBL||VN_CODE(enp)==T_LIT_INT)

/* flag bits */
#define NODE_CURDLED			1
#define NODE_IS_SHAPE_OWNER		2
#define NODE_HAS_SHAPE_RESOLVED		4
#define NODE_NEEDS_CALLTIME_RES		8
#define NODE_WAS_WARNED			16
#define NODE_HAS_CONST_VALUE		32
#define NODE_FINISHED			64	// can be released

#define NODE_IS_FINISHED(enp)		(VN_FLAGS(enp)&NODE_FINISHED)
#define HAS_CONSTANT_VALUE(enp)		( VN_FLAGS((enp)) & NODE_HAS_CONST_VALUE )

#define CURDLE(enp)		VN_FLAGS((enp)) |= NODE_CURDLED;
#define MARK_WARNED(enp)	VN_FLAGS((enp)) |= NODE_WAS_WARNED;

/*
#define TRAVERSED	8
#define UK_LEAF		16
#define PRELIM_SHAPE	32
#define NODE_COMPILED	64
#define HAS_UNKNOWN_LEAF(enp)	( VN_FLAGS(( enp )) & UK_LEAF )
#define PRELIM_SHAPE_SET(enp)	( VN_FLAGS(( enp )) & PRELIM_SHAPE )
#define NODE_IS_COMPILED(enp)	( VN_FLAGS(( enp )) & NODE_COMPILED )
#define ALREADY_TRAVERSED(enp)	( VN_FLAGS(( enp )) & TRAVERSED )
*/



#define WAS_WARNED(enp)		( VN_FLAGS(( enp )) & NODE_WAS_WARNED)
#define IS_CURDLED(enp)		( VN_FLAGS(( enp )) & NODE_CURDLED)
#define OWNS_SHAPE(enp)		( VN_FLAGS(( enp )) & NODE_IS_SHAPE_OWNER)

/* not the best identifier, since it really means we are either resolved, or must wait??? */
#define IS_RESOLVED(enp)	( VN_FLAGS(( enp )) & (NODE_HAS_SHAPE_RESOLVED|NODE_NEEDS_CALLTIME_RES) )

#define RESOLVED_AT_CALLTIME(enp)	( VN_FLAGS(( enp )) & NODE_NEEDS_CALLTIME_RES )

#define IS_VECTOR_SHAPE(shpp)	( UNKNOWN_SHAPE(shpp) || (SHP_N_TYPE_ELTS(shpp) > 1) )
#define IS_VECTOR_NODE(enp)	( VN_SHAPE(enp)==NULL || IS_VECTOR_SHAPE(VN_SHAPE(enp)))
#define NODE_SHAPE_KNOWN(enp)	( VN_SHAPE(enp) != NULL && (!UNKNOWN_SHAPE(VN_SHAPE(enp))))

#define UNKNOWN_SOMETHING(enp)	( ( VN_SHAPE(enp) !=NULL && UNKNOWN_SHAPE(VN_SHAPE(enp)) ) || HAS_UNKNOWN_LEAF(enp) )

typedef struct run_info {
	Context_Pair *	ri_prev_cpp;
	int		ri_arg_stat;
	Subrt *		ri_srp;
	Subrt *		ri_old_srp;
} Run_Info;

/* We might be cautious and have a flag that says ptr or ref? */

typedef struct funcptr {
	Subrt *		fp_srp;
} Function_Ptr;


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

extern void delete_id(QSP_ARG_DECL  Item *);
extern void pop_subrt_cpair(QSP_ARG_DECL  Context_Pair *cpp,const char *name);
extern void _dump_subrt(QSP_ARG_DECL  Subrt *);
#define dump_subrt(srp)		_dump_subrt(QSP_ARG  srp)
extern Vec_Expr_Node *find_node_by_number(QSP_ARG_DECL  int);

extern Item_Context *	pop_subrt_ctx(QSP_ARG_DECL  const char *,Item_Type *);
#define POP_SUBRT_ID_CTX(s)		pop_subrt_ctx(QSP_ARG  s, id_itp)
#define POP_SUBRT_DOBJ_CTX(s)		pop_subrt_ctx(QSP_ARG  s, dobj_itp)


#define UNDEF_OF(s)		undef_of(QSP_ARG  s)

extern Subrt *remember_subrt(QSP_ARG_DECL  Precision * prec_p,const char *,Vec_Expr_Node *,Vec_Expr_Node *);
extern void update_subrt(QSP_ARG_DECL  Subrt *srp, Vec_Expr_Node *body );
extern COMMAND_FUNC( do_run_subrt );
extern void _exec_subrt(QSP_ARG_DECL  Vec_Expr_Node *,Data_Obj *dst_dp);
#define exec_subrt(enp,dp)		_exec_subrt(QSP_ARG enp,dp)
extern void _run_subrt(QSP_ARG_DECL  Subrt_Call *scp,Data_Obj *dst_dp);
extern void _run_subrt_immed(QSP_ARG_DECL  Subrt_Call *scp,Data_Obj *dst_dp);
#define run_subrt(scp,dp)	_run_subrt(QSP_ARG  scp,dp)
#define run_subrt_immed(scp,dp)	_run_subrt_immed(QSP_ARG  scp,dp)

extern COMMAND_FUNC( do_dump_subrt );
extern COMMAND_FUNC( do_opt_subrt );
extern COMMAND_FUNC( do_fuse_kernel );
extern COMMAND_FUNC( do_subrt_info );

extern void expr_init(void);
extern void expr_file(SINGLE_QSP_ARG_DECL);
extern Subrt *	create_script_subrt(QSP_ARG_DECL  const char *,int,const char *);

extern void list_node(Vec_Expr_Node *);
extern int _eval_tree(QSP_ARG_DECL  Vec_Expr_Node *enp,Data_Obj *dst_dp);
#define eval_tree(enp,dst_dp)		_eval_tree(QSP_ARG enp,dst_dp)
extern void undeclare_stuff(Vec_Expr_Node *enp);
extern void set_subrt_ctx(QSP_ARG_DECL  const char *name);
extern void delete_subrt_ctx(QSP_ARG_DECL  const char *name);

extern void compare_arg_trees(QSP_ARG_DECL  Vec_Expr_Node *,Vec_Expr_Node *);

/* nodetbl.c */
extern void sort_tree_tbl(void);

/* subrt.c */
extern COMMAND_FUNC( do_tell_cost );

/* vectree.y */

extern Vec_Expr_Node *_dup_tree(QSP_ARG_DECL  Vec_Expr_Node *);
#define dup_tree(enp)			_dup_tree(QSP_ARG  enp)

extern void show_context_stack(QSP_ARG_DECL  Item_Type *);

extern Vec_Expr_Node *_node0(QSP_ARG_DECL  Tree_Code);
extern Vec_Expr_Node *_node1(QSP_ARG_DECL  Tree_Code,Vec_Expr_Node *);
extern Vec_Expr_Node *_node2(QSP_ARG_DECL  Tree_Code,Vec_Expr_Node *,Vec_Expr_Node *);
extern Vec_Expr_Node *_node3(QSP_ARG_DECL  Tree_Code,Vec_Expr_Node *,Vec_Expr_Node *,Vec_Expr_Node *);

extern void rls_vectree(Vec_Expr_Node *);
extern void check_release(Vec_Expr_Node *);

extern void _node_error(QSP_ARG_DECL  Vec_Expr_Node *);

extern void init_expr_node(QSP_ARG_DECL  Vec_Expr_Node *);
extern void set_global_ctx(SINGLE_QSP_ARG_DECL);
extern void unset_global_ctx(SINGLE_QSP_ARG_DECL);

/* dumptree.c */
extern void	set_native_func_tbl(Keyword *);
extern void	set_show_shape(int);
extern void	set_show_lhs_refs(int);
extern void	print_dump_legend(SINGLE_QSP_ARG_DECL);
extern void	print_shape_key(SINGLE_QSP_ARG_DECL);
extern void	_dump_tree_with_key(QSP_ARG_DECL  Vec_Expr_Node *);
extern void	_dump_node_with_shape(QSP_ARG_DECL  Vec_Expr_Node *);

/* costtree.c */
extern void tell_cost(QSP_ARG_DECL  Subrt *);
extern void cost_tree(QSP_ARG_DECL  Vec_Expr_Node *);

/* comptree.c */

extern Shape_Info *alloc_shape(void);
extern Shape_Info *product_shape(Shape_Info *,Shape_Info *);
extern void _update_node_shape(QSP_ARG_DECL   Vec_Expr_Node *enp);
extern Vec_Expr_Node *	_nth_arg(QSP_ARG_DECL  Vec_Expr_Node *enp, int n);
extern void		_link_uk_nodes(QSP_ARG_DECL  Vec_Expr_Node *,Vec_Expr_Node *);
extern void		_update_tree_shape(QSP_ARG_DECL  Vec_Expr_Node *);
extern void		prelim_tree_shape(Vec_Expr_Node *);
extern void		_compile_tree(QSP_ARG_DECL  Vec_Expr_Node *);
extern Vec_Expr_Node *	_compile_prog(QSP_ARG_DECL  Vec_Expr_Node *);
extern void		_compile_subrt(QSP_ARG_DECL Subrt *);
extern Shape_Info *	scalar_shape(prec_t);
extern Shape_Info *	uk_shape(prec_t);
extern int		shapes_match(Shape_Info *,Shape_Info *);
extern void		init_fixed_nodes(SINGLE_QSP_ARG_DECL);
extern void		_copy_node_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,Shape_Info *shpp);
extern void		discard_node_shape(Vec_Expr_Node *);
extern const char *	_get_lhs_name(QSP_ARG_DECL Vec_Expr_Node *enp);
extern Shape_Info *	calc_outer_shape(Vec_Expr_Node *,Vec_Expr_Node *);


#define node0(code)			_node0(QSP_ARG  code)
#define node1(code,enp)			_node1(QSP_ARG  code, enp)
#define node2(code,enp1,enp2)		_node2(QSP_ARG  code,enp1,enp2)
#define node3(code,enp1,enp2,enp3)	_node3(QSP_ARG  code,enp1,enp2,enp3)
#define node_error(enp)		_node_error(QSP_ARG  enp)
#define dump_tree(enp)		_dump_tree_with_key(QSP_ARG  enp)
#define dump_node_with_shape(enp)		_dump_node_with_shape(QSP_ARG  enp)
#define update_node_shape(enp)		_update_node_shape(QSP_ARG  enp)
#define nth_arg(enp,n)	_nth_arg(QSP_ARG  enp,n)
#define link_uk_nodes(enp1,enp2)	_link_uk_nodes(QSP_ARG  enp1,enp2)
#define update_tree_shape(enp)		_update_tree_shape(QSP_ARG  enp)
#define compile_tree(enp)		_compile_tree(QSP_ARG enp)
#define compile_prog(enp)		_compile_prog(QSP_ARG enp)
#define compile_subrt(srp)		_compile_subrt(QSP_ARG srp)
#define copy_node_shape( enp, shpp )	_copy_node_shape(QSP_ARG enp , shpp )
#define get_lhs_name(enp)	_get_lhs_name(QSP_ARG enp)

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
extern int	check_arg_shapes(QSP_ARG_DECL  Vec_Expr_Node *arg,Vec_Expr_Node *valp,Subrt_Call *scp);
#define CHECK_ARG_SHAPES(arg,valp,scp)		check_arg_shapes(QSP_ARG arg,valp,scp)


extern void		check_resolution(Subrt *srp);
extern void		_point_node_shape(QSP_ARG_DECL  Vec_Expr_Node *,Shape_Info *);
#define point_node_shape( enp , sip )	_point_node_shape(QSP_ARG  enp , sip )
extern int		decl_count(QSP_ARG_DECL  Vec_Expr_Node *);
extern int		arg_count(Vec_Expr_Node *);
extern void		resolve_subrt(QSP_ARG_DECL  Subrt_Call *,List *uk_list, Shape_Info *ret_shpp);
#define RESOLVE_SUBRT(srp,uk_list,ret_shpp)	resolve_subrt(QSP_ARG srp,uk_list,ret_shpp)


/* resolve.c */

extern Vec_Expr_Node *	resolve_node(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Shape_Info *shpp);
#define RESOLVE_NODE(uk_enp,shpp)		resolve_node(QSP_ARG uk_enp,shpp)
extern void		forget_resolved_shapes(QSP_ARG_DECL  Subrt *srp);
extern void		forget_resolved_tree(QSP_ARG_DECL  Vec_Expr_Node *enp);
extern void		resolve_tree(QSP_ARG_DECL  Vec_Expr_Node *enp,Vec_Expr_Node *whence);
#define RESOLVE_TREE(enp,whence)		resolve_tree(QSP_ARG enp,whence)
extern void		late_calltime_resolve(QSP_ARG_DECL  Subrt *srp, Data_Obj *dst_dp);
#define LATE_CALLTIME_RESOLVE(srp,dst_dp)	late_calltime_resolve(QSP_ARG srp,dst_dp)
extern void		early_calltime_resolve(QSP_ARG_DECL  Subrt_Call *scp, Data_Obj *dst_dp);
#define EARLY_CALLTIME_RESOLVE(scp,dst_dp)	early_calltime_resolve(QSP_ARG scp,dst_dp)

/* ml_supp.c */
void insure_object_size(QSP_ARG_DECL  Data_Obj *dp,index_t index);

/* evaltree.c */

extern void		note_assignment(Data_Obj *dp);
extern Data_Obj *	_make_local_dobj(QSP_ARG_DECL  Dimension_Set *,Precision *prec_p, Platform_Device *pdp);
extern int		zero_dp(QSP_ARG_DECL  Data_Obj *);
extern Data_Obj *	mlab_reshape(QSP_ARG_DECL  Data_Obj *,Shape_Info *,const char *);
extern void		_eval_immediate(QSP_ARG_DECL  Vec_Expr_Node *enp);
extern void		wrapup_context(QSP_ARG_DECL  Run_Info *rip);
extern Run_Info *	setup_subrt_call(QSP_ARG_DECL  Subrt_Call *scp,Data_Obj *dst_dp);
extern Subrt_Call *		runnable_subrt(QSP_ARG_DECL  Vec_Expr_Node *enp);
extern Identifier *	make_named_reference(QSP_ARG_DECL  const char *name);
extern Identifier *	_get_set_ptr(QSP_ARG_DECL Vec_Expr_Node *);
extern Data_Obj *	_eval_obj_ref(QSP_ARG_DECL  Vec_Expr_Node *);
extern Data_Obj *	_eval_obj_exp(QSP_ARG_DECL  Vec_Expr_Node *,Data_Obj *);
extern Context_Pair *	pop_previous(SINGLE_QSP_ARG_DECL);
extern void		_restore_previous(QSP_ARG_DECL  Context_Pair *);
extern Identifier *	_eval_ptr_ref(QSP_ARG_DECL  Vec_Expr_Node *enp,int expect_ptr_set);
extern char *		node_desc(Vec_Expr_Node *);
extern void		_reeval_decl_stat(QSP_ARG_DECL  Precision *prec_p,Vec_Expr_Node *,int ro);
extern const char *	_eval_string(QSP_ARG_DECL Vec_Expr_Node *);
extern void		_missing_case(QSP_ARG_DECL  Vec_Expr_Node *,const char *);
extern long		_eval_int_exp(QSP_ARG_DECL Vec_Expr_Node *);
extern double		_eval_flt_exp(QSP_ARG_DECL Vec_Expr_Node *);
extern void		_eval_decl_tree(QSP_ARG_DECL  Vec_Expr_Node *);

#define make_local_dobj(dsp,prec_p,pdp)	_make_local_dobj(QSP_ARG  dsp,prec_p,pdp)
#define reeval_decl_stat(prec,enp,decl_flags)		_reeval_decl_stat(QSP_ARG  prec,enp,decl_flags)
#define eval_immediate(enp)		_eval_immediate(QSP_ARG enp)
#define get_set_ptr(enp)		_get_set_ptr(QSP_ARG enp)
#define eval_obj_ref(enp)		_eval_obj_ref(QSP_ARG enp)
#define eval_obj_exp(enp,dp)		_eval_obj_exp(QSP_ARG enp,dp)
#define eval_decl_tree(enp)		_eval_decl_tree(QSP_ARG enp)
#define eval_flt_exp(enp)		_eval_flt_exp(QSP_ARG enp)
#define eval_int_exp(enp)		_eval_int_exp(QSP_ARG enp)
#define eval_string(enp)		_eval_string(QSP_ARG enp)
#define eval_ptr_ref(enp,expect_ptr_set)	_eval_ptr_ref(QSP_ARG enp,expect_ptr_set)
#define missing_case(enp,str)		_missing_case(QSP_ARG  enp,str)
#define restore_previous(cpp)		_restore_previous(QSP_ARG cpp)

/* opt_tree.c */

extern void _optimize_subrt(QSP_ARG_DECL  Subrt *);
#define optimize_subrt(srp)		_optimize_subrt(QSP_ARG  srp)

/* vt_menu.c */

extern void vt_init(SINGLE_QSP_ARG_DECL);

/* vecnodes.c */

extern const char *node_data_type_desc(Node_Data_Type t);


#endif // ! _VECTREE_H_

