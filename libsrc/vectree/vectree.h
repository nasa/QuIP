
#ifdef INC_VERSION
char VersionId_inc_vectree[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */


#ifndef NO_VEXPR_NODE

#include "data_obj.h"
//#include "strbuf.h"
#include "query.h"
#include "veclib/vecgen.h"
#include "vec_expr_node.h"
#include "subrt.h"
#include "identifier.h"
#include "pointer.h"
#include "strbuf.h"
#include "ctx_pair.h"

typedef struct keyword {
	const char *	kw_token;
	int		kw_code;
} Keyword;

extern Keyword vt_native_func_tbl[];
extern Keyword ml_native_func_tbl[];



#define EXPECT_PTR_SET	1
#define UNSET_PTR_OK	0

#define NO_UNDEF	((Undef_Sym *)NULL)

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
	ERROR1(ERROR_STRING);						\
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


#define NODE_PREC(enp)		( VN_SHAPE(enp) == NO_SHAPE ? PREC_VOID : VN_PREC(enp) )

#define COMPLEX_NODE(enp)	( VN_PREC(enp) == PREC_CPX || VN_PREC(enp) == PREC_DBLCPX )

#define NO_VEXPR_NODE	((Vec_Expr_Node *)NULL)

#define NULL_CHILD( enp , index )		( VN_CHILD(enp,index) == NO_VEXPR_NODE )

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
#define IS_VECTOR_NODE(enp)	( VN_SHAPE(enp)==NO_SHAPE || IS_VECTOR_SHAPE(VN_SHAPE(enp)))
#define NODE_SHAPE_KNOWN(enp)	( VN_SHAPE(enp) != NO_SHAPE && (!UNKNOWN_SHAPE(VN_SHAPE(enp))))

#define UNKNOWN_SOMETHING(enp)	( ( VN_SHAPE(enp) !=NO_SHAPE && UNKNOWN_SHAPE(VN_SHAPE(enp)) ) || HAS_UNKNOWN_LEAF(enp) )

typedef struct run_info {
	Context_Pair *	ri_prev_cpp;
	int		ri_arg_stat;
	Subrt *		ri_srp;
	Subrt *		ri_old_srp;
} Run_Info;

#define NO_RUN_INFO ((Run_Info *)NULL)


/* We might be cautious and have a flag that says ptr or ref? */

typedef struct funcptr {
	Subrt *		fp_srp;
} Function_Ptr;

#define NO_FUNC_PTR	((Function_Ptr *)NULL)



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
#define POP_SUBRT_CPAIR(cpp,name)	pop_subrt_cpair(QSP_ARG  cpp, name)
extern void dump_subrt(QSP_ARG_DECL  Subrt *);
#define DUMP_SUBRT(srp)		dump_subrt(QSP_ARG  srp)
extern Vec_Expr_Node *find_node_by_number(QSP_ARG_DECL  int);

extern Item_Context *	pop_subrt_ctx(QSP_ARG_DECL  const char *,Item_Type *);
//#define POP_SUBRT_CTX(s,itp)	pop_subrt_ctx(QSP_ARG  s, itp)


//ITEM_INTERFACE_PROTOTYPES(Identifier,id)
#define ID_OF(s)		id_of(QSP_ARG s)
#define GET_ID(s)		get_id(QSP_ARG s)
//ITEM_INTERFACE_PROTOTYPES(Subrt,subrt)
//ITEM_INTERFACE_PROTOTYPES(Undef_Sym,undef)
#define UNDEF_OF(s)		undef_of(QSP_ARG  s)

extern Subrt *remember_subrt(QSP_ARG_DECL  Precision * prec_p,const char *,Vec_Expr_Node *,Vec_Expr_Node *);
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
//extern void rls_tree(Vec_Expr_Node *);
extern void rls_vectree(Vec_Expr_Node *);
extern void check_release(Vec_Expr_Node *);
#define RLS_VECTREE(enp)	rls_vectree(enp)
extern void node_error(QSP_ARG_DECL  Vec_Expr_Node *);
#define NODE_ERROR(enp)		node_error(QSP_ARG  enp)
extern void init_expr_node(QSP_ARG_DECL  Vec_Expr_Node *);
extern void set_global_ctx(SINGLE_QSP_ARG_DECL);
extern void unset_global_ctx(SINGLE_QSP_ARG_DECL);

/* dumptree.c */
extern void	set_native_func_tbl(Keyword *);
extern void	set_show_shape(int);
extern void	set_show_lhs_refs(int);
extern void	print_dump_legend(SINGLE_QSP_ARG_DECL);
extern void	print_shape_key(SINGLE_QSP_ARG_DECL);
extern void	dump_tree(QSP_ARG_DECL  Vec_Expr_Node *);
#define DUMP_TREE(enp)		dump_tree(QSP_ARG  enp)
extern void	dump_node(QSP_ARG_DECL  Vec_Expr_Node *);
#define DUMP_NODE(enp)		dump_node(QSP_ARG  enp)

/* costtree.c */
extern void tell_cost(QSP_ARG_DECL  Subrt *);
extern void cost_tree(QSP_ARG_DECL  Vec_Expr_Node *);

/* comptree.c */

extern Shape_Info *alloc_shape(void);
extern Shape_Info *product_shape(Shape_Info *,Shape_Info *);
//#ifdef CAUTIOUS
//extern void verify_null_shape(QSP_ARG_DECL  Vec_Expr_Node *enp);
//#endif /* CAUTIOUS */
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
extern Vec_Expr_Node *	compile_prog(QSP_ARG_DECL  Vec_Expr_Node *);
#define COMPILE_PROG(enp)		compile_prog(QSP_ARG enp)
extern void		compile_subrt(QSP_ARG_DECL Subrt *);
#define COMPILE_SUBRT(srp)		compile_subrt(QSP_ARG srp)
extern Shape_Info *	scalar_shape(prec_t);
extern Shape_Info *	uk_shape(prec_t);
extern int		shapes_match(Shape_Info *,Shape_Info *);
extern void		init_fixed_nodes(SINGLE_QSP_ARG_DECL);
extern void		_copy_node_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,Shape_Info *shpp);
#define copy_node_shape( enp, shpp )	_copy_node_shape(QSP_ARG enp , shpp )
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

extern void		note_assignment(Data_Obj *dp);
extern Data_Obj *	make_local_dobj(QSP_ARG_DECL  Dimension_Set *,Precision *prec_p);
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
extern void		reeval_decl_stat(QSP_ARG_DECL  Precision *prec_p,Vec_Expr_Node *,int ro);
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

