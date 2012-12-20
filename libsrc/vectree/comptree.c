/* comptree.c
 *
 * "Compiling" the tree does a number of things:
 * It figures out some node "shapes", that is,
 * the dimensions of the object the node represents
 * by calling prelim_node_shape
 *
 * It also handles changing node opcodes when necessary,
 * eg T_PLUS->T_VV_FUNC,T_VS_FUNC.
 *
 * T_MINVAL->???, etc
 *
 */

#include "quip_config.h"

char VersionId_vectree_comptree[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* abort */
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#define SET_SHAPE_FLAGS(shpp,dp)	set_shape_flags(shpp,dp,AUTO_SHAPE)
#ifdef HAVE_STRING_H
#include <string.h>
#endif


#include "data_obj.h"
#include "debug.h"
#include "getbuf.h"
#include "function.h"
#include "fio_api.h"
#include "vectree.h"
#include "nvf_api.h"

/* global var */
debug_flag_t cast_debug=0;

static Shape_Info *_cpx_scalar_shpp[N_NAMED_PRECS];

/* local prototypes */

static void check_bitmap_arg(QSP_ARG_DECL Vec_Expr_Node *enp,int index);
static void check_xx_v_condass_code(QSP_ARG_DECL Vec_Expr_Node *enp);
static void check_xx_s_condass_code(QSP_ARG_DECL Vec_Expr_Node *enp);
static void check_xx_xx_condass_code(QSP_ARG_DECL Vec_Expr_Node *enp);

#define CHECK_BITMAP_ARG(enp,index)		check_bitmap_arg(QSP_ARG enp,index)
#define CHECK_XX_V_CONDASS_CODE(enp)		check_xx_v_condass_code(QSP_ARG enp)
#define CHECK_XX_S_CONDASS_CODE(enp)		check_xx_s_condass_code(QSP_ARG enp)
#define CHECK_XX_XX_CONDASS_CODE(enp)		check_xx_xx_condass_code(QSP_ARG enp)

#ifdef FOOBAR
/* New */
static void check_vx_vx_condass_code(QSP_ARG_DECL Vec_Expr_Node *enp);
#define CHECK_VX_VX_CONDASS_CODE(enp)		check_vx_vx_condass_code(QSP_ARG enp)
#endif /* FOOBAR */

static void promote_child(QSP_ARG_DECL Vec_Expr_Node *enp, int i1, int i2);
#define PROMOTE_CHILD(enp,i1,i2)		promote_child(QSP_ARG enp,i1,i2)
static void check_typecast(QSP_ARG_DECL  Vec_Expr_Node *enp,int i1, int i2);
#define CHECK_TYPECAST(enp,i1,i2)		check_typecast(QSP_ARG  enp,i1,i2)
static Shape_Info * get_mating_shapes(QSP_ARG_DECL   Vec_Expr_Node *enp,int i1, int i2);
#define GET_MATING_SHAPES(enp,i1,i2)		get_mating_shapes(QSP_ARG   enp,i1,i2)
static Vec_Expr_Node *one_matlab_subscript(QSP_ARG_DECL   Vec_Expr_Node *obj_enp,Vec_Expr_Node *subscr_enp);
#define ONE_MATLAB_SUBSCRIPT(obj_enp,subscr_enp)	one_matlab_subscript(QSP_ARG   obj_enp,subscr_enp)
static Vec_Expr_Node *compile_matlab_subscript(QSP_ARG_DECL   Vec_Expr_Node *obj_enp,Vec_Expr_Node *subscr_enp);
#define COMPILE_MATLAB_SUBSCRIPT(obj_enp,subscr_enp)	compile_matlab_subscript(QSP_ARG  obj_enp,subscr_enp)

static Shape_Info *cpx_scalar_shape(prec_t prec);
static Vec_Expr_Node * balance_list(Vec_Expr_Node *enp, Tree_Code list_code );
static Shape_Info * get_child_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,int child_index);

static void update_node_shape(QSP_ARG_DECL   Vec_Expr_Node *enp);
#define UPDATE_NODE_SHAPE(enp)		update_node_shape(QSP_ARG  enp)
static void update_assign_shape(QSP_ARG_DECL  Vec_Expr_Node *enp);
static void compute_assign_shape(QSP_ARG_DECL  Vec_Expr_Node *enp);
static void prelim_node_shape(QSP_ARG_DECL  Vec_Expr_Node *enp);
#define PRELIM_NODE_SHAPE(enp)		prelim_node_shape(QSP_ARG enp)
static Vec_Expr_Node * compile_node(QSP_ARG_DECL   Vec_Expr_Node **enpp);
#define COMPILE_NODE(enpp)		compile_node(QSP_ARG   enpp);
static void check_minmax_code(QSP_ARG_DECL   Vec_Expr_Node *enp);
#define CHECK_MINMAX_CODE(enp)		check_minmax_code(QSP_ARG  enp)

static void count_lhs_refs(QSP_ARG_DECL  Vec_Expr_Node *enp);
#define COUNT_LHS_REFS(enp)		count_lhs_refs(QSP_ARG enp)
static int remember_node(List **lpp,Vec_Expr_Node *enp);
static void invert_op(QSP_ARG_DECL  Vec_Expr_Node *enp,Vec_Func_Code wc,Tree_Code new_tc);
static void check_arith_code(QSP_ARG_DECL  Vec_Expr_Node *enp);
static void set_bool_vecop_code(QSP_ARG_DECL  Vec_Expr_Node *enp);
static void remember_return_node(Subrt *srp,Vec_Expr_Node *enp);
static int count_name_refs(QSP_ARG_DECL  Vec_Expr_Node *enp,const char *name);
#define COUNT_NAME_REFS( enp, name )	count_name_refs(QSP_ARG  enp , name )
static Shape_Info * shapes_mate(QSP_ARG_DECL  Vec_Expr_Node *enp1,Vec_Expr_Node *enp2,Vec_Expr_Node *enp);
#define SHAPES_MATE(enp1,enp2,enp)	shapes_mate(QSP_ARG  enp1,enp2,enp)
static Shape_Info * compatible_shape(Vec_Expr_Node *enp1,Vec_Expr_Node *enp2,Vec_Expr_Node *enp);
static void remember_callfunc_node(Subrt *srp,Vec_Expr_Node *enp);

static Shape_Info *check_mating_shapes(QSP_ARG_DECL  Vec_Expr_Node *,int,int);
static void typecast_child(QSP_ARG_DECL Vec_Expr_Node *enp,int index,prec_t prec);
#define TYPECAST_CHILD(enp,index,prec)		typecast_child(QSP_ARG enp,index,prec)



int mode_is_matlab=0;

static Vec_Expr_Node minus_one_node;

static Shape_Info *_scalar_shpp[N_NAMED_PRECS];
static Shape_Info *_uk_shpp[N_NAMED_PRECS];
static Shape_Info *_void_shpp[N_NAMED_PRECS];

#include "vt_native.h"

void (*native_prelim_func)(QSP_ARG_DECL  Vec_Expr_Node *)=prelim_vt_native_shape;
void (*native_update_func)(Vec_Expr_Node *)=update_vt_native_shape;

#define MULTIPLY_DIMENSIONS(p,dim_array)		\
							\
	p  = dim_array[0];				\
	p *= dim_array[1];				\
	p *= dim_array[2];				\
	p *= dim_array[3];				\
	p *= dim_array[4];

#define REDUCE_MINDIM(shpp)	REDUCE_DIMENSION(shpp,si_mindim,++)
#define REDUCE_MAXDIM(shpp)	REDUCE_DIMENSION(shpp,si_maxdim,--)

#define REDUCE_DIMENSION(shpp,which_dim,op)				\
									\
	shpp->si_n_mach_elts /= shpp->si_mach_dim[shpp->which_dim];	\
	shpp->si_n_type_elts /= shpp->si_type_dim[shpp->which_dim];	\
	shpp->si_mach_dim[shpp->si_mindim]=1;				\
	shpp->si_type_dim[shpp->si_mindim]=1;				\
	shpp->si_mindim op ;

#define UPDATE_DIM_LEN(shpp,which_dim,new_len,op)			\
									\
	shpp->si_n_type_elts /= shpp->si_type_dim[shpp->which_dim];	\
	shpp->si_n_mach_elts /= shpp->si_mach_dim[shpp->which_dim];	\
	shpp->si_n_type_elts *= new_len;				\
	shpp->si_n_mach_elts *= new_len;				\
	shpp->si_type_dim[shpp->which_dim]=new_len;			\
	shpp->si_mach_dim[shpp->which_dim]=new_len;			\
	shpp->which_dim op ;

/* return the index of this child among its siblings, or -1 if no parent */
static int which_child(Vec_Expr_Node *enp)
{
	int i;

	if( enp->en_parent == NO_VEXPR_NODE ) return(-1);

	for(i=0;i<tnt_tbl[enp->en_parent->en_code].tnt_nchildren;i++){
		if( enp->en_parent->en_child[i] == enp ) return(i);
	}
#ifdef CAUTIOUS
	NWARN("CAUTIOUS:  which_child:  node not found among parent's children!?!?");
#endif /* CAUTIOUS */
	return(-1);
}

/* Set the shape of this node, by copying the data into this
 * node's personal structure.  The node should already own
 * it's own shape info...
 */

void copy_node_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,Shape_Info *shpp)
{

#ifdef CAUTIOUS
	if( ! NODE_SHOULD_OWN_SHAPE(enp) ){
		sprintf(ERROR_STRING,"CAUTIOUS:  copy_node_shape %s:  node shouldn't own shape!?",
			node_desc(enp));
		WARN(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	if( ! OWNS_SHAPE(enp) ){
		enp->en_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
		enp->en_flags |= NODE_IS_SHAPE_OWNER;
	}

#ifdef CAUTIOUS
	if( enp->en_shpp == NO_SHAPE ){
		WARN("CAUTIOUS:  copy_node_shape:  can't copy to null ptr!?");
		DUMP_TREE(enp);
		return;
	}
	if( shpp == NO_SHAPE ){
		WARN("CAUTIOUS:  copy_node_shape:  can't copy from null ptr!?");
		DUMP_TREE(enp);
		return;
	}
#endif /* CAUTIOUS */

	*enp->en_shpp = *shpp;
}

/* Release a nodes shape struct */

void discard_node_shape(Vec_Expr_Node *enp)
{
	if( enp->en_shpp == NO_SHAPE ) return;

	if( OWNS_SHAPE(enp) ){
		givbuf(enp->en_shpp);
		enp->en_flags &= ~NODE_IS_SHAPE_OWNER;
	}
	enp->en_shpp=NO_SHAPE;
}

/* Set the shape of this node, by setting it's pointer to the arg.
 * The node should NOT already own it's own shape info...
 */

void point_node_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,Shape_Info *shpp)
{

/*
if( enp->en_code == T_DYN_OBJ ){
sprintf(error_string,"point_node_shape OBJECT %s:  shape at 0x%lx",node_desc(enp),(u_long)shpp);
advise(error_string);
describe_shape(shpp);
}
*/

#ifdef CAUTIOUS
	if( ! NODE_SHOULD_PT_TO_SHAPE(enp) ){
		sprintf(error_string,"CAUTIOUS:  point_node_shape %s:  node shouldn't pt to shape!?",
			node_desc(enp));
		WARN(error_string);
		DUMP_TREE(enp);
	}
#endif /* CAUTIOUS */
	if( OWNS_SHAPE(enp) ){
		NODE_ERROR(enp);
		sprintf(error_string,
	"point_node_shape:  %s node n%d already owns shape info!?",NNAME(enp),enp->en_serial);
		WARN(error_string);
		/*
		describe_shape(enp->en_shpp);
		*/
		discard_node_shape(enp);
	}
	enp->en_shpp = shpp;
}


#ifdef FOOBAR
/* We call this to find out if an expression can be evaluated before runtime */

static int is_variable(Vec_Expr_Node *enp)
{
	int i;

	if( enp == NO_VEXPR_NODE ) return(0);

	if( enp->en_code == T_DYN_OBJ ) return(1);

	for(i=0;i<MAX_CHILDREN(enp);i++){
		if( enp->en_child[i] != NO_VEXPR_NODE ){
			if( is_variable(enp->en_child[i]) )
				return(1);
		}
	}
	return(0);
}
#endif /* FOOBAR */

/* When do we call typecast_child?
 *
 * When we want to insert a typecast node between the given node and one of its children.
 */

static void typecast_child(QSP_ARG_DECL Vec_Expr_Node *enp,int index,prec_t prec)
{
	Vec_Expr_Node *new_enp;

//sprintf(error_string,"typecast_child:  %s  %d  %s",
//node_desc(enp), index, name_for_prec(prec));
//advise(error_string);

	/* A few vector operators allow mixed mode ops */
	if( enp->en_code == T_TIMES ){
		if( COMPLEX_PRECISION(prec) && ! COMPLEX_PRECISION(enp->en_child[index]->en_prec) ){
/*
advise("mixed mode case");
*/
				if( enp->en_child[index]->en_prec == (prec&MACH_PREC_MASK) ){
/*
sprintf(error_string,"child %s precision  %s matches machine precision %s",
node_desc(enp->en_child[index]),name_for_prec(enp->en_child[index]->en_prec),prec_name[prec&MACH_PREC_MASK]);
advise(error_string);
*/
					return;
				}
/*
else advise("mixed mode machine precs do not match, casting");
*/
		}
	}

	switch(enp->en_child[index]->en_code){	/* typecast_child */

		case T_VV_B_CONDASS:
		case T_SS_B_CONDASS:
#ifdef CAUTIOUS
			if( index == 0 ){
				sprintf(error_string,
		"typecast child %s:  should not typecast bitmap child",
					node_desc(enp));
				WARN(error_string);
				DUMP_TREE(enp);
				return;
			}
#endif /* CAUTIOUS */
			break;

		/* These cases don't actually do anything (safe to ignore type mismatch) */

		case T_LOAD:	/* load does its own casting if necessary */
		case T_RAMP:	/* typecast_child */ /* should be OK, all args are scalars */

		case T_SCALMAX:  case T_SCALMIN:
		ALL_SCALAR_FUNCTION_CASES
		case T_LIT_INT:
		case T_LIT_DBL:
			/* Don't bother to typecast scalars...  whoever needs them will
			 * do the right thing?
			 * BUT if we are casting to bit?
			 */
			if( prec != PREC_BIT )
				return;
			break;

		case T_END:			/* matlab */

			return;

		/* for scroll, only typecast the image
		 * BUG?  do we need to?  is scroll smart enough to use convert?
		 */
		case T_SCROLL:
			if( index > 0 ) return;
			break;

		/* We'd like to be able to treat a list of components
		 * as a complex or quaternion without a typecast...
		 */
		case T_COMP_OBJ:
			/* If the machine precisions match, and the number
			 * of components is correct, then change the precision
			 * of the comp_obj.
			 */
			if( COMPLEX_PRECISION(prec) &&
			enp->en_child[index]->en_shpp->si_mach_dim[0] == 2 ){
				if( enp->en_child[index]->en_prec == PREC_SP ){
					enp->en_child[index]->en_prec = PREC_CPX;
					enp->en_child[index]->en_shpp->si_type_dim[0] = 1;
					return;
				} else if( enp->en_child[index]->en_prec == PREC_DP ){
					enp->en_child[index]->en_prec = PREC_DBLCPX;
					enp->en_child[index]->en_shpp->si_type_dim[0] = 1;
					return;
				}
				break;
			} else if( QUAT_PRECISION(prec) &&
			enp->en_child[index]->en_shpp->si_mach_dim[0] == 4 ){
				if( enp->en_child[index]->en_prec == PREC_SP ){
					enp->en_child[index]->en_prec = PREC_QUAT;
					enp->en_child[index]->en_shpp->si_type_dim[0] = 1;
					return;
				} else if( enp->en_child[index]->en_prec == PREC_DP ){
					enp->en_child[index]->en_prec = PREC_DBLQUAT;
					enp->en_child[index]->en_shpp->si_type_dim[0] = 1;
					return;
				}
				break;
			}
			break;

		/* For the remaining cases, we have to insert an explicit typecast node */

		ALL_MATH_VFN_CASES
		case T_CALLFUNC:		/* typecast_child */
		case T_UMINUS:
		case T_RECIP:
		case T_STRING:
		case T_RANGE2:
		MOST_OBJREF_CASES
		/*
		ALL_VECTOR_SCALAR_CASES
		ALL_VECTOR_VECTOR_CASES
		*/
		case T_VV_FUNC:			/* typecast_child */
		case T_VS_FUNC:			/* typecast_child */
		case T_INNER:
		ALL_SCALAR_BINOP_CASES
		case T_CALL_NATIVE:	/* ?? should we have a special function to check this? */
		case T_SUM:
			break;

		default:
			MISSING_CASE(enp->en_child[index],"typecast_child");
sprintf(error_string,"wanted to typecast to %s precision",name_for_prec(prec));
advise(error_string);
DUMP_TREE(enp);
			break;
	}

#ifdef DEBUG
if( debug & cast_debug ){
sprintf(error_string,"typecast_child %s:  typecasting child %s to %s",node_desc(enp),node_desc(enp->en_child[index]),
name_for_prec(prec));
advise(error_string);
describe_shape(enp->en_child[index]->en_shpp);
}
#endif /* DEBUG */

	new_enp = NODE1(T_TYPECAST,enp->en_child[index]);

	enp->en_child[index] = new_enp;
	new_enp->en_parent = enp;

	new_enp->en_cast_prec = prec;

	PRELIM_NODE_SHAPE(new_enp);
} /* typecast_child */


void link_one_uk_arg(Vec_Expr_Node *call_enp, Vec_Expr_Node *arg_enp)
{
	Node *np;

	if( call_enp->en_uk_args == NO_LIST )
		call_enp->en_uk_args = new_list();

	np = mk_node(arg_enp);
	addTail(call_enp->en_uk_args,np);
	/*
	LINK_UK_NODES(call_enp,arg_enp);
	*/
}

/* We have two nodes which can resolve each other - remember this! */

void link_uk_nodes(QSP_ARG_DECL  Vec_Expr_Node *enp1,Vec_Expr_Node *enp2)
{
	Node *np;

	if( enp1->en_resolvers == NO_LIST )
		enp1->en_resolvers = new_list();
	if( enp2->en_resolvers == NO_LIST )
		enp2->en_resolvers = new_list();

#ifdef CAUTIOUS
	/* When we started linking unknown args with the callfunc nodes,
	 * we lost the parent relationship...
	 * Does it matter???
	 */

	/* We make sure that these nodes aren't already linked before we make
	 * a new link...
	 */

	if( nodeOf(enp1->en_resolvers,enp2) != NO_NODE ){
		NODE_ERROR(enp1);
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  link_uk_nodes:  %s and %s are already linked!?",
			node_desc(enp1),node_desc(enp2));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

	np = mk_node(enp1);
	addTail(enp2->en_resolvers,np);

	np = mk_node(enp2);
	addTail(enp1->en_resolvers,np);
}


/* Add a tree node to a list...
 * We pass a ptr to the list pointer, in case the list needs to be created.
 */

static int remember_node(List **lpp,Vec_Expr_Node *enp)
{
	Node *np;

	/* We do a CAUTIOUS check to make sure that the node
	 * is not on the list already...  But because we deviate from
	 * a strict tree structure, we may visit a node twice
	 * during tree traversal...  Therefore we use a flag (which we
	 * have to clear!!)
	 *
	 * However, we have more than one list on which we remember nodes...
	 */

	if( *lpp == NO_LIST )
		*lpp = new_list();

#ifdef CAUTIOUS
	/* make sure enp not on list already */
	np = (*lpp)->l_head;
	while(np!=NO_NODE){
		if( np->n_data == enp ){
			/* Because nodes can have multiple parents,
			 * this is not an error!?
			 */
			/*
			sprintf(error_string,
		"CAUTIOUS:  %s node n%d already remembered!?",NNAME(enp),enp->en_serial);
			WARN(error_string);
			return(-1);
			*/
			return(0);
		}
		np=np->n_next;
	}
#endif /* CAUTIOUS */

	np = mk_node(enp);
	addHead(*lpp,np);
	return(0);
}

static void remember_return_node(Subrt *srp,Vec_Expr_Node *enp)
{
	if( remember_node(&srp->sr_ret_lp,enp) < 0 )
		NWARN("error adding node to return list");
}

/* what is the purpose of this??? */

static void remember_callfunc_node(Subrt *srp,Vec_Expr_Node *enp)
{
	if( remember_node(&srp->sr_call_lp,enp) < 0 )
		NWARN("error adding node to callfunc list");
}



/* A utility routine for v/s and v-s...
 * We insert a new node between this node and its second child.
 */

static void invert_op(QSP_ARG_DECL  Vec_Expr_Node *enp,Vec_Func_Code wc,Tree_Code new_tc)
{
	Vec_Expr_Node *new_enp;

	VERIFY_DATA_TYPE(enp,ND_FUNC,"invert_op")

	enp->en_vfunc_code = wc;

	/* this just suppresses a warning in node1() */
	enp->en_child[1]->en_parent = NO_VEXPR_NODE;

	new_enp = NODE1(new_tc,enp->en_child[1]);
	POINT_NODE_SHAPE(new_enp,enp->en_child[1]->en_shpp);

	enp->en_child[1] = new_enp;
}

/* given a node, and the shapes of the two children,
 * determine whether a vector-vector or vector-scalar
 * operation is needed.  If so, change the node code,
 * and set the en_vfunc_code field with the appropriate vectbl code.
 *
 * This would also be a good place to check for precision mismatches,
 * and insert a typecast node if necessary...
 */

static void check_arith_code(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int vector_scalar;

	VERIFY_DATA_TYPE(enp,ND_FUNC,"check_arith_code")
	vector_scalar=0;	/* default */
	if( IS_VECTOR_SHAPE(enp->en_child[0]->en_shpp) ){
		if( IS_VECTOR_SHAPE(enp->en_child[1]->en_shpp) ){
			/* vector-vector op */
			if( enp->en_code == T_PLUS ){
				enp->en_vfunc_code = FVADD;
			} else if( enp->en_code == T_MINUS ){
				enp->en_vfunc_code = FVSUB;
			} else if( enp->en_code == T_TIMES ){
				enp->en_vfunc_code = FVMUL;
			} else if( enp->en_code == T_DIVIDE ){
				enp->en_vfunc_code = FVDIV;
			} else if( enp->en_code == T_MODULO ){
				enp->en_vfunc_code = FVMOD;
			} else if( enp->en_code == T_BITRSHIFT ){
				enp->en_vfunc_code = FVSHR;
			} else if( enp->en_code == T_BITLSHIFT ){
				enp->en_vfunc_code = FVSHL;
			} else if( enp->en_code == T_BITAND ){
				enp->en_vfunc_code = FVAND;
			} else if( enp->en_code == T_BITOR ){
				enp->en_vfunc_code = FVOR;
			} else if( enp->en_code == T_BITXOR ){
				enp->en_vfunc_code = FVXOR;
			}
#ifdef CAUTIOUS
			  else {
				sprintf(error_string,
	"CAUTIOUS:  check_arith_code:  unhandled vector-vector operation %s",NNAME(enp));
				WARN(error_string);
			}
#endif /* CAUTIOUS */

			enp->en_code = T_VV_FUNC;
			/* minus node points to shape but vv_func must own */

			{
			Shape_Info *tmp_shpp;
			tmp_shpp = enp->en_shpp;
			discard_node_shape(enp);
			COPY_NODE_SHAPE(enp,tmp_shpp);
			}
		} else {
			/* vector scalar operation,
			 * second operand is the scalar,
			 * so we leave the order unchanged.
			 */
			vector_scalar=2;
		}
	} else if( IS_VECTOR_SHAPE(enp->en_child[1]->en_shpp) ){
		/* vector scalar op, first node is
		 * the scalar, so we interchange the nodes
		 */
		Vec_Expr_Node *tmp_enp;

		tmp_enp = enp->en_child[0];
		enp->en_child[0] = enp->en_child[1];
		enp->en_child[1] = tmp_enp;
		vector_scalar=1;
	}

	/* else scalar-scalar - do nothing */

	if( vector_scalar ){
		/* vector-vector op */
		if( enp->en_code == T_PLUS ){
			enp->en_vfunc_code = FVSADD;
		} else if( enp->en_code == T_MINUS ){
			if( vector_scalar==1 ){	/* scalar is 1st op */
				enp->en_vfunc_code = FVSSUB;
			} else {
				invert_op(QSP_ARG  enp,FVSADD,T_UMINUS);
			}
		} else if( enp->en_code == T_TIMES ){
			enp->en_vfunc_code = FVSMUL;
		} else if( enp->en_code == T_BITOR ){
			enp->en_vfunc_code = FVSOR;
		} else if( enp->en_code == T_BITAND ){
			enp->en_vfunc_code = FVSAND;
		} else if( enp->en_code == T_BITXOR ){
			enp->en_vfunc_code = FVSXOR;
		} else if( enp->en_code == T_DIVIDE ){
			if( vector_scalar==1 ){	/* scalar is 1st op */
				enp->en_vfunc_code = FVSDIV;
			} else {
				/* invert_op(QSP_ARG  enp,FVSMUL,T_RECIP); */
				enp->en_vfunc_code = FVSDIV2;
			}
		} else if ( enp->en_code == T_MODULO ){
			if( vector_scalar==1 ){ /* scalar is 1st op */
				enp->en_vfunc_code = FVSMOD2;
			} else {		/* scalar is 2nd op */
				enp->en_vfunc_code = FVSMOD;
			}
		} else if( enp->en_code == T_BITLSHIFT ){
			if( vector_scalar==1 ){ /* scalar is 1st op */
				enp->en_vfunc_code = FVSSHL2;
			} else {		/* scalar is 2nd op */
				enp->en_vfunc_code = FVSSHL;
			}
		} else if( enp->en_code == T_BITRSHIFT ){
			if( vector_scalar==1 ){ /* scalar is 1st op */
				enp->en_vfunc_code = FVSSHR2;
			} else {		/* scalar is 2nd op */
				enp->en_vfunc_code = FVSSHR;
			}
		}
#ifdef CAUTIOUS
		  else {
			sprintf(error_string,
	"CAUTIOUS:  check_arith_code:  unhandled vector-scalar operation %s",NNAME(enp));
			WARN(error_string);
		}
#endif /* CAUTIOUS */

		enp->en_code = T_VS_FUNC;
	}
} /* check_arith_code */

/* Set the code field for a boolean test node */

static void set_vs_bool_code(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	switch(enp->en_code){
		case T_BOOL_NOT:
		case T_DYN_OBJ:
			return;
			break;
		default:
			break;
	}

	VERIFY_DATA_TYPE(enp,ND_FUNC,"set_vs_bool_code")
	switch(enp->en_code){
		case T_BOOL_LT: enp->en_bm_code = FVSMLT; break;
		case T_BOOL_GT: enp->en_bm_code = FVSMGT; break;
		case T_BOOL_LE: enp->en_bm_code = FVSMLE; break;
		case T_BOOL_GE: enp->en_bm_code = FVSMGE; break;
		case T_BOOL_NE: enp->en_bm_code = FVSMNE; break;
		case T_BOOL_EQ: enp->en_bm_code = FVSMEQ; break;

		/* No-ops */
		case T_BOOL_AND: break;
		case T_BOOL_XOR: break;
		case T_BOOL_OR: break;

#ifdef CAUTIOUS
		default:
			advise(node_desc(enp));
			sprintf(error_string,"CAUTIOUS:  unexpected case in set_vs_bool_code");
			WARN(error_string);
			break;
#endif /* CAUTIOUS */
	}
} /* end set_vs_bool_code */

static void set_vv_bool_code(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	VERIFY_DATA_TYPE(enp,ND_FUNC,"set_vv_bool_code")
	switch(enp->en_code){
		case T_BOOL_LT: enp->en_bm_code = FVVMLT; break;
		case T_BOOL_GT: enp->en_bm_code = FVVMGT; break;
		case T_BOOL_LE: enp->en_bm_code = FVVMLE; break;
		case T_BOOL_GE: enp->en_bm_code = FVVMGE; break;
		case T_BOOL_NE: enp->en_bm_code = FVVMNE; break;
		case T_BOOL_EQ: enp->en_bm_code = FVVMEQ; break;

		/* We added these cases here to shut up the missing case warning
		 * (which didn't always display correctly???), but do we really
		 * want to call this routine when these codes are encountered?
		 * FIXME
		 */

		case T_BOOL_AND:
		case T_BOOL_OR:
		case T_BOOL_XOR:
		case T_BOOL_NOT:
				break;

#ifdef CAUTIOUS
		default:
			WARN(error_string);
			sprintf(error_string,"CAUTIOUS:  unexpected case (%s) in set_vv_bool_code",
				vec_func_tbl[enp->en_vfunc_code].vf_name);
			break;
#endif /* CAUTIOUS */
	}
}

static void commute_bool_test(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	switch(enp->en_code){
		case T_BOOL_GT:  enp->en_code = T_BOOL_LT; break;
		case T_BOOL_LT:  enp->en_code = T_BOOL_GT; break;
		case T_BOOL_GE:  enp->en_code = T_BOOL_LE; break;
		case T_BOOL_LE:  enp->en_code = T_BOOL_GE; break;
		case T_BOOL_EQ:  break;
		case T_BOOL_NE:  break;
		default:
			MISSING_CASE(enp,"commute_bool_test");
			break;
	}
}

/* We call this when we have to invert the order of the arguments in a conditional assignment... */

static void invert_bool_test(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	switch(enp->en_code){
		/* This might be a little confusing if someone were to look at the tree...
		 * because the function codes don't match the tree codes...
		 */
		case T_BOOL_GT:  enp->en_code = T_BOOL_LE; break;
		case T_BOOL_LT:  enp->en_code = T_BOOL_GE; break;
		case T_BOOL_GE:  enp->en_code = T_BOOL_LT; break;
		case T_BOOL_LE:  enp->en_code = T_BOOL_GT; break;
		case T_BOOL_EQ:  enp->en_code = T_BOOL_NE; break;
		case T_BOOL_NE:  enp->en_code = T_BOOL_EQ; break;
		/* Here we need to either insert a not node, or change the code.
		 * Changing the code would be simpler, but we don't have BOOL_NXOR, etc.
		 */
		case T_BOOL_AND:
		case T_BOOL_XOR:
		case T_BOOL_OR:

		case T_DYN_OBJ:
			{
			Vec_Expr_Node *new_enp;
			Vec_Expr_Node *parent;

			/* Need to insert an inversion node */

			parent = enp->en_parent;		/* save this, gets reset in node1() */
			new_enp=NODE1(T_BOOL_NOT,enp);
			/* can we assume this is the first child? */
#ifdef CAUTIOUS
			if( enp->en_parent->en_child[0] != enp ){
				sprintf(error_string,"CAUTIOUS:  invert_bool_test:  %s is not first child of %s!?",
					node_desc(enp),node_desc(enp->en_parent));
				ERROR1(error_string);
			}
#endif /* CAUTIOUS */
			parent->en_child[0] = new_enp;
			new_enp->en_parent = parent;
			PRELIM_NODE_SHAPE(new_enp);
			break;
			}
		default:
			MISSING_CASE(enp,"invert_bool_test");
			break;
	}
}


/* set_bool_vecop_code
 *
 */

static void set_bool_vecop_code(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int vector_scalar;

	vector_scalar=0;	/* default */
	if( IS_VECTOR_SHAPE(enp->en_child[0]->en_shpp) ){
		if( IS_VECTOR_SHAPE(enp->en_child[1]->en_shpp) ){
			/* vector-vector op */
			/* BUT could be an outer op - we should have already checked this... */
			set_vv_bool_code(QSP_ARG  enp);
		} else {
			/* vector scalar operation,
			 * second operand is the scalar,
			 * so we leave the order unchanged.
			 */
			set_vs_bool_code(QSP_ARG  enp);
		}
	} else if( IS_VECTOR_SHAPE(enp->en_child[1]->en_shpp) ){
		/* vector scalar op, first node is
		 * the scalar, so we interchange the nodes.
		 * We also need to then invert the sense of the comparison.
		 */
		Vec_Expr_Node *tmp_enp;

		tmp_enp = enp->en_child[0];
		enp->en_child[0] = enp->en_child[1];
		enp->en_child[1] = tmp_enp;
//DUMP_TREE(enp);
		commute_bool_test(QSP_ARG  enp);

		set_vs_bool_code(QSP_ARG  enp);
	}
#ifdef CAUTIOUS
	else {
		WARN("CAUTIOUS:  set_bool_vecop_code:  no vector shapes");
	}
#endif /* CAUTIOUS */
}

/* The child node is supposed to be a bitmap...
 * But since we can use an object as a truth value (implies a!=0),
 * we have to handle this case.
 * Boolean truth operators, we descend recursively.
 */

static void check_bitmap_arg(QSP_ARG_DECL Vec_Expr_Node *enp,int index)
{
	Vec_Expr_Node *enp2,*enp3;

	switch(enp->en_child[index]->en_code){

		/*
		ALL_BINOP_CASES
		*/
		ALL_OBJREF_CASES
			/* If the object is not a bitmap, then we have to test != 0 */
			/* do we have the shape? */
			/* if( BITMAP_SHAPE(enp->en_child[index]->en_shpp) ) */
			if( BITMAP_PRECISION(enp->en_child[index]->en_shpp->si_prec) )
				break;

			/* else fall through if not bitmap */

		case T_VV_FUNC:			/* check_bitmap_arg */
		case T_VS_FUNC:
			VERIFY_DATA_TYPE(enp,ND_FUNC,"check_bitmap_arg")
			if( FLOATING_PREC(enp->en_child[index]->en_prec) ){
				enp2 = NODE0(T_LIT_INT);
				enp2->en_intval = 0;
			} else {
				enp2 = NODE0(T_LIT_DBL);
				enp2->en_dblval = 0.0;
			}
			PRELIM_NODE_SHAPE(enp2);

			enp3 = NODE2(T_BOOL_NE,enp->en_child[index],enp2);
			enp3->en_bm_code = FVSMNE;
			PRELIM_NODE_SHAPE(enp3);

			enp->en_child[index] = enp3;
			enp3->en_parent = enp;

			break;

		ALL_NUMERIC_COMPARISON_CASES		/* check_bitmap_arg */
			/* do nothing */
			break;

		BINARY_BOOLOP_CASES			/* &&, ||, ^^ */
			/* We are treating these like logical ops -
			 * but are they also bitwise operators on vectors??
			 */
			enp = enp->en_child[index];
			CHECK_BITMAP_ARG(enp,0);
			CHECK_BITMAP_ARG(enp,1);
			break;

		case T_BOOL_NOT: /* check_bitmap_arg */
			enp = enp->en_child[index];
			CHECK_BITMAP_ARG(enp,0);
			break;

		default:
			MISSING_CASE(enp->en_child[index],"check_bitmap_arg");
			break;
	}
} /* check_bitmap_arg */

/* We can use a vector as a logical variable, as in C.
 * When this is encountered, we add a T_BOOL_NE 0 node.
 */

/* check_xx_v_condass code - We know that the condition is a vector (bitmap).
 * We examine the shapes of the two (assignment source) children,
 * determine whether a vector-vector-vector, vector-vector-scalar,
 * or vector-scalar-scalar operation is needed.  If so, change the node code,
 * and set the en_bm_code field with the appropriate vectbl code.
 */

static void check_xx_v_condass_code(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	if( IS_VECTOR_SHAPE(enp->en_child[1]->en_shpp) ){
		if( IS_VECTOR_SHAPE(enp->en_child[2]->en_shpp) ){
			/* vector-vector op */
			enp->en_code = T_VV_B_CONDASS;
		} else {
			/* vector scalar operation,
			 * second operand is the scalar,
			 * so we leave the order unchanged.
			 */
			enp->en_code = T_VS_B_CONDASS;
		}
	} else {
		if( IS_VECTOR_SHAPE(enp->en_child[2]->en_shpp) ){
			/* vector scalar op, first node is
			 * the scalar, so we interchange the nodes.
			 * After we do this, however, we need to invert
			 * the sense of the boolean code!
			 */
			Vec_Expr_Node *tmp_enp;

			tmp_enp = enp->en_child[1];
			enp->en_child[1] = enp->en_child[2];
			enp->en_child[2] = tmp_enp;
			enp->en_code = T_VS_B_CONDASS;
			invert_bool_test(QSP_ARG enp->en_child[0]);
			set_vs_bool_code(QSP_ARG  enp->en_child[0]);
		} else {	/* scalar-scalar */
			enp->en_code = T_SS_B_CONDASS;
		}
	}
	CHECK_BITMAP_ARG(enp,0);

	/* Now we should see whether the bitmap arg can be reduced to
	 * one of the new, fast condass functions.
	 * That will be true if the test is a numeric comparison.
	 */

	switch( enp->en_child[0]->en_code ){
		ALL_NUMERIC_COMPARISON_CASES
			CHECK_XX_XX_CONDASS_CODE(enp);
			break;
		default:
			break;	/* do nothing */
	}
} /* check_xx_v_condass_code */


static void check_xx_s_condass_code(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	if( IS_VECTOR_SHAPE(enp->en_child[1]->en_shpp) ){
		if( IS_VECTOR_SHAPE(enp->en_child[2]->en_shpp) ){
			/* vector-vector op */
			enp->en_code = T_VV_S_CONDASS;
		} else {
			/* vector scalar operation,
			 * second operand is the scalar,
			 * so we leave the order unchanged.
			 */
			enp->en_code = T_VS_S_CONDASS;
		}
	} else {
		if( IS_VECTOR_SHAPE(enp->en_child[2]->en_shpp) ){
			/* vector scalar op, first node is
			 * the scalar, so we interchange the nodes.
			 * After we do this, however, we need to invert
			 * the sense of the boolean code!
			 */
			Vec_Expr_Node *tmp_enp;

			tmp_enp = enp->en_child[1];
			enp->en_child[1] = enp->en_child[2];
			enp->en_child[2] = tmp_enp;
			enp->en_code = T_VS_S_CONDASS;

			invert_bool_test(QSP_ARG enp->en_child[0]);
			set_vs_bool_code(QSP_ARG  enp->en_child[0]);
		} else {	/* scalar-scalar */
			/* don't think we ever get here... */
			enp->en_code = T_SS_S_CONDASS;
		}
	}
} /* check_xx_s_condass_code */

#ifdef FOOBAR
static void check_xx_sx_condass_code(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	if( IS_VECTOR_SHAPE(enp->en_child[1]->en_shpp) ){
		if( IS_VECTOR_SHAPE(enp->en_child[2]->en_shpp) ){
			/* vector-vector op */
			enp->en_code = T_VV_SS_CONDASS;
		} else {
			/* vector scalar operation,
			 * second operand is the scalar,
			 * so we leave the order unchanged.
			 */
			enp->en_code = T_VS_SS_CONDASS;
		}
	} else {
		if( IS_VECTOR_SHAPE(enp->en_child[2]->en_shpp) ){
			/* vector scalar op, first node is
			 * the scalar, so we interchange the nodes.
			 * After we do this, however, we need to invert
			 * the sense of the boolean code!
			 */
			Vec_Expr_Node *tmp_enp;

			tmp_enp = enp->en_child[1];
			enp->en_child[1] = enp->en_child[2];
			enp->en_child[2] = tmp_enp;
			enp->en_code = T_VS_SS_CONDASS;

			invert_bool_test(QSP_ARG enp->en_child[0]);
			set_vs_bool_code(QSP_ARG  enp->en_child[0]);
		} else {	/* scalar-scalar */
			/* don't think we ever get here... */
			enp->en_code = T_SS_SS_CONDASS;
		}
	}
} /* check_xx_sx_condass_code */
#endif /* FOOBAR */


int decl_count(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int i1,i2;

	if( enp==NO_VEXPR_NODE ) return(0);

	switch(enp->en_code){
		case T_DECL_STAT_LIST:
			i1=decl_count(QSP_ARG  enp->en_child[0]);
			i2=decl_count(QSP_ARG  enp->en_child[1]);
			return(i1+i2);

		case T_DECL_STAT:
			i1=decl_count(QSP_ARG  enp->en_child[0]);
			return(i1);

		/* We don't need a case for T_EXTERN_DECL
		 * because this routine is only used for function args...
		 */

		/*
		case T_SEQ_DECL:
		case T_VEC_DECL:
		case T_SCAL_DECL:
		case T_IMG_DECL:
		case T_PTR_DECL:
		case T_FUNCPTR_DECL:
		case T_CSEQ_DECL:
		case T_CIMG_DECL:
		case T_CVEC_DECL:
		case T_CSCAL_DECL:
		*/
		case T_BADNAME:			/* decl_count */
		ALL_DECL_ITEM_CASES
			return(1);

		default:
			MISSING_CASE(enp,"decl_count");
			DUMP_TREE(enp);
			break;
	}
	return(1);
}

/* final_return() returns 1 if the subroutine ends with a return node.
 */

static int final_return(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
/*
sprintf(error_string,"final_return %s BEGIN",node_desc(enp));
advise(error_string);
*/
	while(enp!=NO_VEXPR_NODE){
/*
sprintf(error_string,"final_return %s",node_desc(enp));
advise(error_string);
*/
		switch(enp->en_code){
			case T_RETURN:
				return(1);

			case T_STAT_LIST:
				enp = enp->en_child[1];
				break;

			case T_EXP_PRINT:
			case T_ASSIGN:
			case T_SET_STR:
			case T_SET_PTR:
			case T_CALLFUNC:			/* final_return */
				/* Why call node_error here??? */
				/* NODE_ERROR(enp); */
				return(0);

			/* BUG probably need other loops here ? */
			case T_IFTHEN:
				if( ! final_return(QSP_ARG  enp->en_child[1]) ) return(0);
				if( enp->en_child[2] == NO_VEXPR_NODE ) return(1);
				return( final_return(QSP_ARG  enp->en_child[2]) );

			default:
				MISSING_CASE(enp,"final_return");
				return(0);
		}
	}
	return(0);
} /* final_return */

/* Return the dominant precision (the other will be promoted to it... */

/* There are 8 machine precisions, so we can represent the dominance
 * relations using an 8x8 table.
 * Here, we set the dominant precision to always be one of the two we
 * start with.  But might we want BY,UIN to be dominated by IN or UDI???
 */

static prec_t dominance_tbl[N_MACHINE_PRECS][N_MACHINE_PRECS];
static int dominance_table_inited=0;

static void set_precision_precedence(prec_t this_prec)
{
	prec_t p;

	for(p=0;p<N_MACHINE_PRECS;p++){
		if( dominance_tbl[this_prec][p] == N_MACHINE_PRECS ){	/* not set yet */
			dominance_tbl[this_prec][p] = this_prec;
			dominance_tbl[p][this_prec] = this_prec;
		}
	}
}

static void init_dominance_table(void)
{
	prec_t p1,p2;

	for(p1=0;p1<N_MACHINE_PRECS;p1++)
		for(p2=0;p2<N_MACHINE_PRECS;p2++)
			dominance_tbl[p1][p2] = N_MACHINE_PRECS;	/* flag as unset */

	set_precision_precedence(PREC_DP);
	set_precision_precedence(PREC_SP);
	set_precision_precedence(PREC_DI);
	set_precision_precedence(PREC_UDI);
	set_precision_precedence(PREC_IN);
	set_precision_precedence(PREC_UIN);
	set_precision_precedence(PREC_BY);
	set_precision_precedence(PREC_UBY);

	dominance_table_inited=1;
}

/* promote_child
 *
 * We call this when we have a binary operator that wants its two children to have
 * the same type.  We insert a typecast node using the rules.
 * The precedence order is:  double, float, long, u_long, short, u_short, byte, u_byte.
 *
 * This can also be called for condass nodes - which is why we have to include
 * the child indices as arguments.
 */

static void promote_child(QSP_ARG_DECL   Vec_Expr_Node *enp, int i1, int i2)
{
	int d;	/* dominant index */
	int i;	/* promoted index */
	prec_t p1,p2;

	p1=enp->en_child[i1]->en_prec;
	p2=enp->en_child[i2]->en_prec;

	if( !dominance_table_inited ) init_dominance_table();
	if( dominance_tbl[p1][p2] == p1 ){
		d=i1;
		i=i2;
	} else {
		d=i2;
		i=i1;
	}

	TYPECAST_CHILD(enp,i,enp->en_child[d]->en_prec);
}

/* check_typecast
 *
 * We check the precisions of the specified children.
 * If the precisions differ, we cast one to the other...
 * Right now, we have no good way to decide which to cast,
 * but for efficiency's sake we should always cast the scalar
 * (if only one is a scalar).  Otherwise, we might as well use
 * the C rules (cast up).
 *
 * There are exceptions to this rule:
 * If we have an operator that is inherently integer, e.g. bitwise
 * and shift operators, then we should cast any floating objects to
 * the integer precision.
 *
 * Also if one operand is boolean (bit), then we cast the other to a boolean.
 *
 * When we have real and complex with the same machine precision, we assume
 * that what we are doing is legal; this may need to be fixed, we may need
 * reorder nodes...
 *
 * There are some unhandled cases with the boolean operators...
 *
 * We don't need to enforce matching types between sibling nodes of
 * a condass operator - but the 2nd and 3rd nodes may have to be cast
 * to match the destination object!
 */

static void check_typecast(QSP_ARG_DECL  Vec_Expr_Node *enp,int i1, int i2)
{

#ifdef DEBUG
if( debug & cast_debug ){
sprintf(error_string,"check_typecast: %d %d",i1,i2);
advise(error_string);
DUMP_TREE(enp);
}
#endif /* DEBUG */

	if( enp->en_child[i1]->en_prec == enp->en_child[i2]->en_prec ) return;

	/* Here we know the two children have different types... */
	switch(enp->en_code){
		BINARY_BOOLOP_CASES
			/* logical and etc */
			/* handled by evaluator? */
			return;
		
		case T_VV_FUNC:				/* check_typecast */
		case T_VS_FUNC:
			/* we need to switch here on the function code... */
			switch(enp->en_vfunc_code){
				VV_ALLPREC_FUNC_CASES
				VS_ALLPREC_FUNC_CASES
					goto not_integer_only;
					break;

				VS_INTONLY_FUNC_CASES
					goto integer_only_cases;
					break;

				default:
					sprintf(error_string,
	"check_typecast:  unhandled function code %d (%s)",enp->en_vfunc_code,
						vec_func_tbl[enp->en_vfunc_code].vf_name
						);
					WARN(error_string);
					break;
			}
			break;

		/* ALL_INTEGER_BINOP_CASES */
		ALL_SCALINT_BINOP_CASES		/* check_typecast */
			/* modulo, bitwise operators and shifts... */
integer_only_cases:
			/* these are operators where we want to cast to int */
			if( FLOATING_PREC(enp->en_child[i1]->en_prec) ){
				if( INTEGER_PREC(enp->en_child[i2]->en_prec) ){
					TYPECAST_CHILD(enp,i1,enp->en_child[i2]->en_prec);
				} else {
					TYPECAST_CHILD(enp,i1,PREC_DI);
					TYPECAST_CHILD(enp,i2,PREC_DI);
				}
			} else if( FLOATING_PREC(enp->en_child[i2]->en_prec) ){
				TYPECAST_CHILD(enp,i2,enp->en_child[i1]->en_prec);
			} else {
				/* Both are integer, use promotion */
				PROMOTE_CHILD(enp,i1,i2);
			}
			return;


		ALL_NUMERIC_COMPARISON_CASES
			/* bitmaps are a special case */
			if( BITMAP_SHAPE(enp->en_child[i1]->en_shpp) ){
				if( ! BITMAP_SHAPE(enp->en_child[i2]->en_shpp) ){
					TYPECAST_CHILD(enp,i2,enp->en_child[i1]->en_prec);
					return;
				}
			} else if( BITMAP_SHAPE(enp->en_child[i2]->en_shpp) ){
				TYPECAST_CHILD(enp,i1,enp->en_child[i2]->en_prec);
				return;
			}
			/* If neither is a bitmap we do nothing??? */
			break;
		/*
		OTHER_BINOP_CASES
		*/
		case T_MAXVAL:
		case T_MINVAL:
		case T_ROW_LIST: 		/* check_typecast */
		OTHER_SCALAR_BINOP_CASES
		ALL_MATH2FN_CASES
not_integer_only:
			break;

		ALL_CONDASS_CASES			/* check_typecast */
#ifdef CAUTIOUS
			if( i1 == 0 || i2 == 0 ){
				sprintf(error_string,
		"CAUTIOUS:  check_typecast %s:  shouldn't typecast bitmap node", node_desc(enp));
				ERROR1(error_string);
			}
#endif /* CAUTIOUS */
			/* We'd rather not cast the children, we take care of the typecast
			 * at the super node...  BUT it is hard to know the destination type...
			 * Might be better to let this go and have a cleanup pass?
			 */
			break;

		default:
			MISSING_CASE(enp,"check_typecast");
			break;
	}
	/* Don't typecast if we have real and complex... */

	/* The precisions differ - which should we cast? */
	if( SCALAR_SHAPE(enp->en_child[i1]->en_shpp) ){
		if( SCALAR_SHAPE(enp->en_child[i2]->en_shpp) ){
			/* if both are scalars, use C rules */
			PROMOTE_CHILD(enp,i1,i2);
		} else {
			TYPECAST_CHILD(enp,i1,enp->en_child[i2]->en_prec);
		}
	} else if( SCALAR_SHAPE(enp->en_child[i2]->en_shpp) )
		TYPECAST_CHILD(enp,i2,enp->en_child[i1]->en_prec);

	else if( COMPLEX_SHAPE(enp->en_child[i1]->en_shpp) && !  COMPLEX_SHAPE(enp->en_child[i2]->en_shpp) ){
		TYPECAST_CHILD(enp,i2,enp->en_child[i1]->en_prec);
	} else if( COMPLEX_SHAPE(enp->en_child[i2]->en_shpp) && !  COMPLEX_SHAPE(enp->en_child[i1]->en_shpp) ){
		TYPECAST_CHILD(enp,i1,enp->en_child[i2]->en_prec);
	}
	else {
		/* use the C promotion rules */
		PROMOTE_CHILD(enp,i1,i2);
	}
} /* end check_typecast() */

/* check_mating_shapes
 *
 * This is called to verify that a pair of child nodes has shapes that "mate",
 * i.e. if both are vectors they must have the same shape.
 * One or both of the shapes may be scalar.
 *
 * If everything is ok, then it assigns the appropriate shape to the parent node,
 * generally by pointing to one of the children.  A special case is the BOOL operators,
 * which have bitmap precision if they are vector.
 */

Shape_Info * check_mating_shapes(QSP_ARG_DECL  Vec_Expr_Node *enp,int index1,int index2)
{
	Shape_Info *shpp;

	/* THis is kind of a hack... T_CALLFUNC nodes point to the subrt's
	 * shape pointer, which is NO_SHAPE if the return shape can't
	 * be figured out ahead of time...
	 * (Is this comment still true?  If it can't be figured out,
	 * it probably should point to ukshape.)
	 * OR if it's a void subroutine.  But we hope that assignments using void functions
	 * will be flagged as syntax errors!
	 */
	if( enp->en_child[index1]->en_shpp == NO_SHAPE && enp->en_child[index1]->en_code == T_CALLFUNC )
		enp->en_child[index1]->en_shpp = uk_shape(((Subrt *)enp->en_child[index1]->en_srp)->sr_prec);

	if( enp->en_child[index2]->en_shpp == NO_SHAPE && enp->en_child[index2]->en_code == T_CALLFUNC )
		enp->en_child[index2]->en_shpp = uk_shape(((Subrt *)enp->en_child[index2]->en_srp)->sr_prec);

	/* If one of the nodes has no shape, we get rid of the shape of the other (why?)
	 */

	if( enp->en_child[index1]->en_shpp == NO_SHAPE ){
		enp->en_child[index2]->en_shpp = NO_SHAPE;
		return(NO_SHAPE);
	}

	if( enp->en_child[index2]->en_shpp == NO_SHAPE ){
		enp->en_child[index1]->en_shpp = NO_SHAPE;
		return(NO_SHAPE);
	}


	/* The shape of this node will generally be the larger of the shapes
	 * of its children.  If there was a syntax error with either of
	 * the children, the shape info ptr is set to NULL.  Here we
	 * assume that the node's shape info struct has already been
	 * allocated - maybe it would make more sense to do that here?? BUG?
	 *
	 * For outer ops (add, times, etc), the destination size is always the larger
	 * of the operands sizes in any dimension.  But for "projection" ops (max, sum),
	 * it can be smaller...
	 */

	shpp = SHAPES_MATE(enp->en_child[index1],enp->en_child[index2],enp);

	if( shpp==NO_SHAPE ){
		discard_node_shape(enp);
advise("check_mating_shapes:  no mating shapes");
		return(NO_SHAPE);
	}

	switch(enp->en_code){
		ALL_NUMERIC_COMPARISON_CASES			/* check_mating_shapes */
			COPY_NODE_SHAPE(enp,shpp);
			if( ! BITMAP_SHAPE(shpp) ){
				xform_to_bitmap(enp->en_shpp);
			}
			break;

		BINARY_BOOLOP_CASES				/* check_mating_shapes */
			COPY_NODE_SHAPE(enp,shpp);
			if( ! BITMAP_SHAPE(shpp) ){
				xform_to_bitmap(enp->en_shpp);
			}
			break;

		case T_MAXVAL:
		case T_MINVAL:
			COPY_NODE_SHAPE(enp,shpp);
			break;


		default:
			MISSING_CASE(enp,"check_mating_shapes");
			/* fall-thru */

		/*
		ALL_VECTOR_SCALAR_CASES
		ALL_VECTOR_VECTOR_CASES
		*/
		case T_VS_FUNC:
		case T_MATH2_FN:
		ALL_SCALAR_BINOP_CASES
/*
sprintf(error_string,"check_mating_shapes %s:  pointing to shape:",node_desc(enp));
advise(error_string);
describe_shape(shpp);
*/
			POINT_NODE_SHAPE(enp,shpp);
			break;

		ALL_CONDASS_CASES				/* check_mating_shapes */
		case T_VV_FUNC:					/* check_mating_shapes */
			COPY_NODE_SHAPE(enp,shpp);
			break;
	}

	return(shpp);
} /* check_mating_shapes */

/* Make sure that the bitmap has the proper shape for the target node */

#define CHECK_MATING_BITMAP(enp,bm_enp)  check_mating_bitmap(QSP_ARG  enp, bm_enp)

static void check_mating_bitmap(QSP_ARG_DECL  Vec_Expr_Node *enp,Vec_Expr_Node *bm_enp)
{
#ifdef CAUTIOUS
	if( enp->en_shpp == NO_SHAPE || bm_enp->en_shpp == NO_SHAPE ){
		NODE_ERROR(enp);
		NWARN("CAUTIOUS:  check_mating_bitmap:  missing shape");
		return;
	}
#endif /* CAUTIOUS */

	if( SCALAR_SHAPE(bm_enp->en_shpp) )
		return;	/* not really a bitmap! */

	if( UNKNOWN_SHAPE(bm_enp->en_shpp) ){
		/*
		sprintf(error_string,"check_bitmap_shape:  bitmap %s has unknown shape",node_desc(bm_enp));
		advise(error_string);
		*/
		return;
	}
	if( UNKNOWN_SHAPE(enp->en_shpp) ){
		/*
		sprintf(error_string,"check_bitmap_shape:  %s has unknown shape",node_desc(enp));
		advise(error_string);
		*/
		return;
	}

	/* Now we've set the node shape to something which is compatible with
	 * the two sources... make sure that the bitmap shape is compatible.
	 */

	/* The bitmap may have dimensions which the sources do not */
	{
		int i, enlarged=0;

		for(i=0;i<N_DIMENSIONS;i++){
			if( bm_enp->en_shpp->si_type_dim[i] > 1 ){
				if( enp->en_shpp->si_type_dim[i] == 1 ){
					enp->en_shpp->si_type_dim[i] = bm_enp->en_shpp->si_type_dim[i];
					enlarged = 1;
				} else if( enp->en_shpp->si_type_dim[i] != bm_enp->en_shpp->si_type_dim[i] ){
					NODE_ERROR(enp);
					sprintf(DEFAULT_ERROR_STRING,
	"check_mating_bitmap:  Bitmap %s does not have shape to match %s!?",
						node_desc(bm_enp),node_desc(enp));
					NWARN(DEFAULT_ERROR_STRING);
					CURDLE(enp)
					return;
				}
			}
		}
		if( enlarged )
			SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ);
	}
} /* check_mating_bitmap */

/* Get the shapes of the first two child nodes, and check that they "mate"
 * If either node has no shape, both ptrs are set to NO_SHAPE for the return,
 * Shapes mate when they are identical, or one is a scalar.
 *
 * For generalized outer binops, this is more complicated...
 */


static Shape_Info * get_mating_shapes(QSP_ARG_DECL   Vec_Expr_Node *enp,int i1, int i2)
{
//sprintf(error_string,"get_mating_shapes:  %s %d %d",
//node_desc(enp),i1,i2);
//advise(error_string);

	CHECK_TYPECAST(enp,i1,i2);
	return( check_mating_shapes(QSP_ARG  enp,i1,i2) );
}

Shape_Info *calc_outer_shape(Vec_Expr_Node *enp1, Vec_Expr_Node *enp2)
{
	static Shape_Info shp1;
	int i;

	/* we assume get_mating shapes has already been called,
	 * so we don't need to call check_typecast() again...
	 */

	for(i=0;i<N_DIMENSIONS;i++){
		dimension_t d1,d2;

		d1=enp1->en_shpp->si_type_dim[i];
		d2=enp2->en_shpp->si_type_dim[i];
		if( d1 == 1 )
			shp1.si_type_dim[i] = d2;
		else if( d2 == 1 )
			shp1.si_type_dim[i] = d1;
		else if( d1 == d2 )
			shp1.si_type_dim[i] = d1;
		else
			/* do something else here? */
			return(NO_SHAPE);
	}
	/* Now we know the dimensions, we need to figure out the precision... */
	shp1.si_prec = enp1->en_prec;

	/* BUG?  are mach_dim and type_dim both set? */
	shp1.si_n_mach_elts = 1;
	shp1.si_n_type_elts = 1;
	for(i=0;i<N_DIMENSIONS;i++){
		shp1.si_n_mach_elts *= shp1.si_mach_dim[i];
		shp1.si_n_type_elts *= shp1.si_type_dim[i];
	}

	SET_SHAPE_FLAGS(&shp1,NO_OBJ);

	return(&shp1);
}


/* Scan the tree, checking for object size errors.
 * Set the shape parameters of each node, from the bottom up.
 * Generic operation nodes like T_PLUS, T_TIMES, etc
 * may be replaced w/ T_VV_FUNC or T_VS_FUNC,
 * based on the shapes of the operands.
 *
 * We would like to use this later as the run-time scan too...
 * We really should separate out the one-time things from
 * the repetitive things...
 */

void compile_subrt(QSP_ARG_DECL Subrt *srp)
{
	Subrt *save_srp;

/*
sprintf(error_string,"compile_subrt %s",srp->sr_name);
advise(error_string);
*/

#ifdef CAUTIOUS
	if( IS_COMPILED(srp) ){
		sprintf(error_string,"CAUTIOUS:  compile_subrt:  Subroutine %s has already been compiled",srp->sr_name);
		WARN(error_string);
		abort();
	}
	srp->sr_flags |= SR_COMPILED;
#endif /* CAUTIOUS */

/*
sprintf(error_string,"compile_subrt %s",srp->sr_name);
advise(error_string);
*/
	executing=0;

	save_srp = curr_srp;
	curr_srp = srp;

	srp->sr_nargs = decl_count(QSP_ARG  srp->sr_arg_decls);	/* set # args */

/*
sprintf(error_string,"compile_subrt %s setting SCANNING flag",srp->sr_name);
advise(error_string);
*/
	srp->sr_flags |= SR_SCANNING;

	/* set the context */
	set_subrt_ctx(QSP_ARG  srp->sr_name);

	/* declare the arg variables */
	EVAL_DECL_TREE(srp->sr_arg_decls);

	/* no values to assign, because we haven't been called! */


/*
sprintf(error_string,"compile_subrt %s, before balance_list",srp->sr_name);
advise(error_string);
DUMP_TREE(srp->sr_body);
*/
	srp->sr_body = balance_list(srp->sr_body,T_STAT_LIST);
/*
sprintf(error_string,"compile_subrt %s, after balance_list",srp->sr_name);
advise(error_string);
DUMP_TREE(srp->sr_body);
*/

	COMPILE_TREE(srp->sr_body);

	delete_subrt_ctx(QSP_ARG  srp->sr_name);

	if( srp->sr_prec != PREC_VOID && ! final_return(QSP_ARG  srp->sr_body) ){
		/* what node should we report the error at ? */
		sprintf(error_string,"subroutine %s does not end with a return statement",srp->sr_name);
		WARN(error_string);
	}

/*
sprintf(error_string,"compile_subrt %s clearing SCANNING flag",srp->sr_name);
advise(error_string);
*/
	srp->sr_flags &= ~SR_SCANNING;

	curr_srp = save_srp;
}

/* compile_tree is used to compile the body of a subroutine,
 * it will evaluate declaration statements.
 */

void compile_tree(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	int i;

	if( enp == NO_VEXPR_NODE ) return;

	/* BUG need to set & destroy the context! */

	/* How do we know whether or not to evaluate declarations?
	 * we do need to evaluate declarations within subroutines,
	 * but not global declarations...
	 */

	/* BUG?  we set up uk links in prelim_node_shape, but will
	 * we be ok for auto-init resolutions?
	 */

	if( enp->en_code == T_DECL_STAT || enp->en_code == T_EXTERN_DECL ){
		EVAL_TREE(enp,NO_OBJ);
		return;		/* BUG what if we don't return? */
	}

	for(i=0;i<MAX_CHILDREN(enp);i++){
		if( enp->en_child[i] != NO_VEXPR_NODE )
			COMPILE_TREE(enp->en_child[i]);
	}

	/* now all the child nodes have been scanned, process this one */

	enp=COMPILE_NODE(&enp);
	PRELIM_NODE_SHAPE(enp);
} /* end compile_tree */

/* compile_prog is used to compile  a program with external decls and statements.
 * it will NOT evaluate declaration statements.
 * (why not??)
 */

Vec_Expr_Node * compile_prog(QSP_ARG_DECL   Vec_Expr_Node **enpp)
{
	int i;
	Vec_Expr_Node *enp;

	enp = *enpp;
	executing=0;

	if( enp == NO_VEXPR_NODE ) return(enp);

	/* We didn't used to compile declarations, but after we started allowing
	 * auto-initialization with expressions, this became necessary.
	 */

	/*
	if( enp->en_code == T_DECL_STAT || enp->en_code == T_DECL_STAT_LIST )
		return;
	*/

	/* prototype declarations don't need to be compiled... */
	if( enp->en_code == T_PROTO )
		return(enp);

	for(i=0;i<MAX_CHILDREN(enp);i++){
		if( enp->en_child[i] != NO_VEXPR_NODE )
			COMPILE_PROG(&enp->en_child[i]);
	}

	/* now all the child nodes have been scanned, process this one */

	enp = COMPILE_NODE(&enp);
	PRELIM_NODE_SHAPE(enp);

	if( enp != *enpp )
		*enpp = enp;

	return(enp);
} /* end compile_prog */


/* check_curds - check a tree to see if any nodes are "curdled", such as
 * a syntax error or something else that would preclude execution.
 * Returns 0 for a good tree, -1 otherwise.
 */

static int check_curds(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int i;

	switch(enp->en_code){
		/* Don't curdle stat_list nodes, execute what we can */
		case T_STRING_LIST:
		ALL_DECL_STAT_CASES
		case T_DECL_STAT_LIST:
		case T_DECL_ITEM_LIST:
		case T_STAT_LIST:
		case T_EXP_PRINT:
		case T_SWITCH_LIST:
		case T_CASE_LIST:
			return(0);

		case T_BOOL_PTREQ:
		case T_MAX_INDEX:
		case T_MIN_INDEX:
		case T_ADD_OPT_PARAM:
		case T_PROTO:
		case T_NAME_FUNC:
		case T_FIX_SIZE:
		case T_EXIT:
		case T_MAX_TIMES:
		case T_ADVISE:
		case T_WARN:
		case T_EXPR_LIST:
		case T_ARGLIST:
		case T_ASSIGN:
		case T_CALLFUNC:		/* check_curds */
		case T_CALL_NATIVE:
		case T_RETURN:
		case T_SCRIPT:
		case T_REFERENCE:
		case T_UMINUS:
		case T_RAMP:
		case T_CONJ:
		case T_INNER:
		case T_TRANSPOSE:
		case T_SET_STR:
		case T_SET_PTR:
		/* case T_INFO: */
		ALL_OBJFUNC_CASES
		ALL_CTL_FLOW_CASES
		ALL_FILEIO_CASES
		ALL_UNARY_CASES
		ALL_DFT_CASES
		ALL_INCDEC_CASES
		ALL_STRFUNC_CASES
		/*
		ALL_BINOP_CASES
		*/
		ALL_SCALAR_BINOP_CASES
		case T_VV_FUNC:			/* check_curds */
		case T_VS_FUNC:
		case T_MAXVAL:
		case T_MINVAL:
		case T_SUM:
		case T_TYPECAST:
		case T_REDUCE:
		case T_ENLARGE:
		ALL_BOOLOP_CASES
		ALL_DECL_ITEM_CASES
		ALL_CONDASS_CASES
		ALL_NUMERIC_COMPARISON_CASES
		ALL_SCALAR_FUNCTION_CASES
		ALL_OBJREF_CASES
		case T_EQUIVALENCE:
		SOME_MATLAB_CASES		/* check_curds */

		case T_MIXED_LIST:
		case T_PRINT_LIST:
		case T_ROW_LIST:
		case T_COMP_LIST:
		case T_INDIR_CALL:
			break;
		default:
			MISSING_CASE(enp,"check_curds");
			break;
	}

	for(i=0;i<MAX_CHILDREN(enp);i++)
		if( enp->en_child[i] != NO_VEXPR_NODE && IS_CURDLED(enp->en_child[i]) ){
			CURDLE(enp)
			return(-1);
		}

	return(0);
} /* end check_curds */

/* With any list node, return the number of elements in that list */

int leaf_count(Vec_Expr_Node *enp,Tree_Code list_code)
{
	if( enp == NO_VEXPR_NODE ) return(0);
	if( enp->en_code == list_code ){
		int n1,n2;
		n1=leaf_count(enp->en_child[0],list_code);
		n2=leaf_count(enp->en_child[1],list_code);
		return(n1+n2);
	}
	return(1);
}

static Vec_Expr_Node *one_matlab_subscript(QSP_ARG_DECL   Vec_Expr_Node *obj_enp,Vec_Expr_Node *subscr_enp)
{
	Vec_Expr_Node *minus_one_enp, *enp1, *enp2, *new_enp;

	switch(subscr_enp->en_code){
		case T_ENTIRE_RANGE:
			new_enp = NODE3(T_SUBVEC,obj_enp,NULL,NULL);
			new_enp=COMPILE_NODE(&new_enp); /* needed?? */
			PRELIM_NODE_SHAPE(new_enp);
			/* BUG we should clean up the discarded nodes */
			/* BUG? do we need to compile the plus nodes? */
			return(new_enp);
			
		case T_INDEX_SPEC:		/* one_matlab_subscript */
		case T_RANGE2:			/* one_matlab_subscript */
			/* subtract 1 from the starting index */
			minus_one_enp = NODE0(T_LIT_INT);
			minus_one_enp->en_intval = -1;
			PRELIM_NODE_SHAPE(minus_one_enp);
			enp1=NODE2(T_PLUS,subscr_enp->en_child[0],minus_one_enp);
			enp1=COMPILE_NODE(&enp1); /* needed?? */
			PRELIM_NODE_SHAPE(enp1);

			/* subtract one from the final index */
			minus_one_enp = NODE0(T_LIT_INT);
			minus_one_enp->en_intval = -1;
			PRELIM_NODE_SHAPE(minus_one_enp);
			enp2=NODE2(T_PLUS,subscr_enp->en_child[1],minus_one_enp);
			enp2=COMPILE_NODE(&enp2); /* needed? */
			PRELIM_NODE_SHAPE(enp2);

			new_enp = NODE3(T_SUBVEC,obj_enp,enp1,enp2);
			new_enp=COMPILE_NODE(&new_enp); /* needed?? */
			PRELIM_NODE_SHAPE(new_enp);
			/* BUG we should clean up the discarded nodes */
			/* BUG? do we need to compile the plus nodes? */
			return(new_enp);
		case T_LIT_DBL:
		case T_LIT_INT:
		case T_DYN_OBJ:
		ALL_UNMIXED_SCALAR_BINOP_CASES
			/*
			minus_one_enp = NODE0(T_LIT_INT);
			minus_one_enp->en_intval = -1;
			enp1=NODE2(T_PLUS,subscr_enp,minus_one_enp);
			new_enp = NODE2(T_SQUARE_SUBSCR,obj_enp,enp1);
			*/
			new_enp = NODE2(T_SQUARE_SUBSCR,obj_enp,subscr_enp);
			new_enp = COMPILE_NODE(&new_enp);
			PRELIM_NODE_SHAPE(new_enp);
			/* BUG we should clean up the discarded nodes */
			/* BUG? do we need to compile the plus nodes? */
			return(new_enp);
		default:
			MISSING_CASE(subscr_enp,"one_matlab_subscript");
			break;
	}
	return(NO_VEXPR_NODE);
}

/* matlab has us put multiple subscripts as lists:
 * m(1,2)
 * which has a tree representation as:
 *
 * 	subscript1
 * 	m	index_list
 * 		1       2
 *
 * We'd like to transform this to:
 *
 * 		square_subscr
 * 	square_subscr	1
 * 	m	0
 *
 * 	In the case where we have an index range m(1:3,2:5) :
 *
 *	subscript1
 *	m	index_list
 *		range2	range2
 *		1    3  2    5
 *
 *		subvec
 *	subvec		1  4
 *	m  0  2
 */

static Vec_Expr_Node *compile_matlab_subscript(QSP_ARG_DECL   Vec_Expr_Node *obj_enp,Vec_Expr_Node *subscr_enp)
{
	Vec_Expr_Node *new_enp;

	switch(subscr_enp->en_code){
		case T_INDEX_LIST:
			new_enp = ONE_MATLAB_SUBSCRIPT(obj_enp,subscr_enp->en_child[0]);
			new_enp = COMPILE_MATLAB_SUBSCRIPT(new_enp,subscr_enp->en_child[1]);
			return(new_enp);
		case T_DYN_OBJ:
		case T_LIT_INT:
		case T_LIT_DBL:
		ALL_UNMIXED_SCALAR_BINOP_CASES
			/* BUG all operators should go here? */
			new_enp = ONE_MATLAB_SUBSCRIPT(obj_enp,subscr_enp);
			return(new_enp);
		case T_INDEX_SPEC:
		case T_RANGE2:			/* compile_matlab_subscript */
		case T_ENTIRE_RANGE:
			new_enp = ONE_MATLAB_SUBSCRIPT(obj_enp,subscr_enp);
			return(new_enp);
		default:
			MISSING_CASE(subscr_enp,"compile_matlab_subscript");
			DUMP_TREE(subscr_enp);
			break;
	}
	return(NO_VEXPR_NODE);
}

static int has_vector_subscript(Vec_Expr_Node *enp)
{
	int i;
	int retval=0;

	for(i=0;i<tnt_tbl[enp->en_code].tnt_nchildren;i++)
		retval |= has_vector_subscript(enp->en_child[i]);

	if( retval ) return(1);

	switch(enp->en_code){
		case T_SQUARE_SUBSCR:
		case T_CURLY_SUBSCR:
			if( ! SCALAR_SHAPE(enp->en_child[1]->en_shpp) )
				return 1;
			break;
		default:
			break;
	}
	return(0);
}

/* enp is a T_ASSIGN node that has a vector subscript on the left ( a[coords] = samples )
 *
 * The goal here is to transform the node to one which is render(&a,&coords,&samples)
 */

static Vec_Expr_Node * fix_render_code(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Vec_Expr_Node *new_enp, *a1_enp, *lhs_enp;
	Vec_Expr_Node *l1_enp, *l2_enp;

	lhs_enp = enp->en_child[0];
	if( lhs_enp->en_code != T_SQUARE_SUBSCR ){
		NODE_ERROR(enp);
		NWARN("fix_render_code:  only know how to fix SQUARE_SUBSCR lhs - sorry!");
		return(NO_VEXPR_NODE);
	}

	/* For now, we assume that this assign node is a simple one:  a[coords] = samples.
	 *
	 * It will be trickier to handle more complicated cases, such as:
	 * a[ coord_img[ warp_coords ] ] = samples;
	 *
	 * But actually, that example only has 1 render, the inner vector subscript is a sample operation...
	 */

	/* We have to build an arglist - grows to left!... */
	a1_enp = NODE1(T_REFERENCE,lhs_enp->en_child[0]);
	l1_enp = NODE2(T_ARGLIST,a1_enp,lhs_enp->en_child[1]);
	l2_enp = NODE2(T_ARGLIST,l1_enp,enp->en_child[1]);

	new_enp = NODE1(T_CALL_NATIVE,l2_enp);
	new_enp->en_intval = NATIVE_RENDER;
	return(new_enp);
} /* end fix_render_code */

#ifdef FOOBAR
static void check_vx_vx_condass_code(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	if( IS_VECTOR_SHAPE(enp->en_child[1]->en_shpp) ){
		if( IS_VECTOR_SHAPE(enp->en_child[2]->en_shpp) ){
			/* vector-vector op */
			enp->en_code = T_VV_VV_CONDASS;
		} else {
			/* vector scalar operation,
			 * second operand is the scalar,
			 * so we leave the order unchanged.
			 */
			enp->en_code = T_VS_B_CONDASS;
		}
	} else {
		if( IS_VECTOR_SHAPE(enp->en_child[2]->en_shpp) ){
			/* vector scalar op, first node is
			 * the scalar, so we interchange the nodes.
			 * After we do this, however, we need to invert
			 * the sense of the boolean code!
			 */
			Vec_Expr_Node *tmp_enp;

			tmp_enp = enp->en_child[1];
			enp->en_child[1] = enp->en_child[2];
			enp->en_child[2] = tmp_enp;
			enp->en_code = T_VS_B_CONDASS;
			invert_bool_test(QSP_ARG enp->en_child[0]);
			set_vs_bool_code(QSP_ARG  enp->en_child[0]);
		} else {	/* scalar-scalar */
			enp->en_code = T_SS_B_CONDASS;
		}
	}
	CHECK_BITMAP_ARG(enp,0);
} /* check_vx_vx_condass_code */
#endif /* FOOBAR */

static Vec_Func_Code vv_vv_test_code(Tree_Code bool_code)
{
	switch(bool_code){
		case T_BOOL_GT: return(FVV_VV_GT); break;
		case T_BOOL_LT: return(FVV_VV_LT); break;
		case T_BOOL_GE: return(FVV_VV_GE); break;
		case T_BOOL_LE: return(FVV_VV_LE); break;
		case T_BOOL_EQ: return(FVV_VV_EQ); break;
		case T_BOOL_NE: return(FVV_VV_NE); break;
#ifdef CAUTIOUS
		default:
			NERROR1("CAUTIOUS:  vv_vv_test_code:  unexpected boolean test");
			break;
#endif /* CAUTIOUS */
	}
	return(INVALID_VFC);
}

static Vec_Func_Code vv_vs_test_code(Tree_Code bool_code)
{
	switch(bool_code){
		case T_BOOL_GT: return(FVV_VS_GT); break;
		case T_BOOL_LT: return(FVV_VS_LT); break;
		case T_BOOL_GE: return(FVV_VS_GE); break;
		case T_BOOL_LE: return(FVV_VS_LE); break;
		case T_BOOL_EQ: return(FVV_VS_EQ); break;
		case T_BOOL_NE: return(FVV_VS_NE); break;
#ifdef CAUTIOUS
		default:
			NERROR1("CAUTIOUS:  vv_vs_test_code:  unexpected boolean test");
			break;
#endif /* CAUTIOUS */
	}
	return(INVALID_VFC);
}


static Vec_Func_Code vs_vv_test_code(Tree_Code bool_code)
{
	switch(bool_code){
		case T_BOOL_GT: return(FVS_VV_GT); break;
		case T_BOOL_LT: return(FVS_VV_LT); break;
		case T_BOOL_GE: return(FVS_VV_GE); break;
		case T_BOOL_LE: return(FVS_VV_LE); break;
		case T_BOOL_EQ: return(FVS_VV_EQ); break;
		case T_BOOL_NE: return(FVS_VV_NE); break;
#ifdef CAUTIOUS
		default:
			NERROR1("CAUTIOUS:  vs_vv_test_code:  unexpected boolean test");
			break;
#endif /* CAUTIOUS */
	}
	return(INVALID_VFC);
}

static Vec_Func_Code vs_vs_test_code(Tree_Code bool_code)
{
	switch(bool_code){
		case T_BOOL_GT: return(FVS_VS_GT); break;
		case T_BOOL_LT: return(FVS_VS_LT); break;
		case T_BOOL_GE: return(FVS_VS_GE); break;
		case T_BOOL_LE: return(FVS_VS_LE); break;
		case T_BOOL_EQ: return(FVS_VS_EQ); break;
		case T_BOOL_NE: return(FVS_VS_NE); break;
#ifdef CAUTIOUS
		default:
			NERROR1("CAUTIOUS:  vs_vs_test_code:  unexpected boolean test");
			break;
#endif /* CAUTIOUS */
	}
	return(INVALID_VFC);
}

static void invert_vec4_condass(Vec_Expr_Node *enp)
{
	switch(enp->en_vfunc_code){
		case FVV_VV_GT: enp->en_vfunc_code = FVV_VV_LE; break;
		case FVV_VV_LE: enp->en_vfunc_code = FVV_VV_GT; break;
		case FVV_VV_LT: enp->en_vfunc_code = FVV_VV_GE; break;
		case FVV_VV_GE: enp->en_vfunc_code = FVV_VV_LT; break;

		case FVV_VS_GT: enp->en_vfunc_code = FVV_VS_LE; break;
		case FVV_VS_LE: enp->en_vfunc_code = FVV_VS_GT; break;
		case FVV_VS_LT: enp->en_vfunc_code = FVV_VS_GE; break;
		case FVV_VS_GE: enp->en_vfunc_code = FVV_VS_LT; break;

		case FVS_VV_GT: enp->en_vfunc_code = FVS_VV_LE; break;
		case FVS_VV_LE: enp->en_vfunc_code = FVS_VV_GT; break;
		case FVS_VV_LT: enp->en_vfunc_code = FVS_VV_GE; break;
		case FVS_VV_GE: enp->en_vfunc_code = FVS_VV_LT; break;

		case FVS_VS_GT: enp->en_vfunc_code = FVS_VS_LE; break;
		case FVS_VS_LE: enp->en_vfunc_code = FVS_VS_GT; break;
		case FVS_VS_LT: enp->en_vfunc_code = FVS_VS_GE; break;
		case FVS_VS_GE: enp->en_vfunc_code = FVS_VS_LT; break;

		case FVV_VV_EQ:
		case FVV_VV_NE:
		case FVV_VS_EQ:
		case FVV_VS_NE:
		case FVS_VV_EQ:
		case FVS_VV_NE:
		case FVS_VS_EQ:
		case FVS_VS_NE:
			break;
#ifdef CAUTIOUS
		default:
			NERROR1("CAUTIOUS:  invert_vec4_condass:  unexpected function code!?");
			break;
#endif /* CAUTIOUS */
	}
}

/* For the old condass funcs, the first child is the bitmap and the next two
 * are the sources.  But for the new types, the first two are the sources,
 * and the second two are the tests.
 *
 * If we get here, we know that the condition is a simple numerical test.
 * So we should be able to substitute one of the new fast conditional
 * assignments, eliminating the intermediate bitmap.
 *
 * The source args have already been fixed, so that the scalar is the second child
 * if it exists, with the test inverted if necessary.
 */

#define FIX_CHILDREN					\
							\
	enp->en_child[0] = enp->en_child[1];		\
	enp->en_child[1] = enp->en_child[2];		\
	enp->en_child[2] = test_enp->en_child[0];	\
	enp->en_child[3] = test_enp->en_child[1];

/* swap the order of the test args */

#define FIX_CHILDREN_2					\
							\
	enp->en_child[0] = enp->en_child[1];		\
	enp->en_child[1] = enp->en_child[2];		\
	enp->en_child[3] = test_enp->en_child[0];	\
	enp->en_child[2] = test_enp->en_child[1];

static void check_xx_xx_condass_code(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Vec_Expr_Node *test_enp;
	test_enp=enp->en_child[0];

	if( enp->en_code == T_VV_B_CONDASS ){	/* both sources vectors? */
		if( IS_VECTOR_SHAPE(test_enp->en_child[0]->en_shpp) ){
			if( IS_VECTOR_SHAPE(test_enp->en_child[1]->en_shpp) ){
				enp->en_code = T_VV_VV_CONDASS;
				FIX_CHILDREN
				enp->en_vfunc_code = vv_vv_test_code(test_enp->en_code);
			} else {
				enp->en_code = T_VV_VS_CONDASS;
				FIX_CHILDREN
				enp->en_vfunc_code = vv_vs_test_code(test_enp->en_code);
			}
		} else if( IS_VECTOR_SHAPE(test_enp->en_child[1]->en_shpp) ){
			FIX_CHILDREN_2
			enp->en_vfunc_code = vv_vv_test_code(test_enp->en_code);
			/* invert sense of test */
			invert_vec4_condass(enp);
		}
#ifdef CAUTIOUS
		  else {
		  	ERROR1("CAUTIOUS:  check_xx_xx_condass_code should not have been called!?");
		}
#endif /* CAUTIOUS */
		
	} else if( enp->en_code == T_VS_B_CONDASS ){	/* second source is a scalar */
		if( IS_VECTOR_SHAPE(test_enp->en_child[0]->en_shpp) ){
			if( IS_VECTOR_SHAPE(test_enp->en_child[1]->en_shpp) ){
				enp->en_code = T_VS_VV_CONDASS;
				FIX_CHILDREN
				enp->en_vfunc_code = vs_vv_test_code(test_enp->en_code);
			} else {
				enp->en_code = T_VS_VS_CONDASS;
				FIX_CHILDREN
				enp->en_vfunc_code = vs_vs_test_code(test_enp->en_code);
			}
		} else if( IS_VECTOR_SHAPE(test_enp->en_child[1]->en_shpp) ){
			FIX_CHILDREN_2
			enp->en_vfunc_code = vs_vv_test_code(test_enp->en_code);
			invert_vec4_condass(enp);
		}
#ifdef CAUTIOUS
		  else {
		  	ERROR1("CAUTIOUS:  check_xx_xx_condass_code should not have been called!?");
		}
#endif /* CAUTIOUS */

	}
#ifdef CAUTIOUS
	  else {
	  	ERROR1("CAUTIOUS:  unexpected node code in check_xx_xx_condass_code!?");
	}
#endif /* CAUTIOUS */
}


/* "compile" is really a misnomer...  What this is, is a one-time
 * scan where we possibly modify the tree based on the shapes of
 * the child nodes...  That is, we determine whether each operation
 * is vector-vector, vector-scalar, and change the node code
 * accordingly.  In some cases, we also have to insert or remove
 * a node.
 * for this purpose, all unknown shape nodes are assumed to have vector shape.
 *
 * Basically, in this phase we transform a tree from something that
 * represents how the program was parsed, into something closer to what
 * we want to compute.
 *
 * (Do we still need uko's?)
 * This phase is also where the lists of uko's get created...
 *
 * A more complicated tree reorganization arises when we allow a
 * vector subscript to indicate a rendering operation:
 * a[v] = b    ->  render_samples(a,v,b);
 */

static Vec_Expr_Node * compile_node(QSP_ARG_DECL   Vec_Expr_Node **enpp)
{
	//int vf_code=(-2);
	Vec_Func_Code vf_code=N_VEC_FUNCS;
	Vec_Expr_Node *new_enp;
	Vec_Expr_Node *enp;

	enp = *enpp;

	/* now all the child nodes have been scanned, process this one */

	/* check for CURDLED children */
	if( MAX_CHILDREN(enp)>0 && check_curds(QSP_ARG  enp) < 0 ) return(enp);

#ifdef CAUTIOUS
	if( OWNS_SHAPE(enp) ){
		sprintf(error_string,
	"CAUTIOUS:  compile_node %s:  already owns shape!?",node_desc(enp));
		WARN(error_string);
		DUMP_TREE(enp);
	}
#endif /* CAUTIOUS */

	enp->en_shpp = NO_SHAPE;

	switch(enp->en_code){
		case T_SUBSCRIPT1:		/* matlab   compile_node */
			new_enp=COMPILE_MATLAB_SUBSCRIPT(enp->en_child[0],enp->en_child[1]);
			/* we need to link this node in place of the original */
			new_enp->en_parent = enp->en_parent;
			if( enp->en_parent != NO_VEXPR_NODE ){
				int i;

				for(i=0;i<MAX_CHILDREN(enp->en_parent);i++)
					if( enp->en_parent->en_child[i] == enp )
						enp->en_parent->en_child[i] = new_enp;
			}
			/* COMPILE_TREE(new_enp); */
			enp=new_enp;

			break;

		ALL_NUMERIC_COMPARISON_CASES		/* compile_node */
			/* These are T_BOOL_EQ etc */
			/* We don't know the shape yet */ 
			/* enp->en_bm_child_shpp = getbuf(sizeof(Shape_Info)); */
			/* Call get mating shapes here? */
			break;

		BINARY_BOOLOP_CASES			/* compile_node */
			/* logical AND etc. */
			if( GET_MATING_SHAPES(enp,0,1) == NO_SHAPE )
{
WARN("compile_node (binary boolop):  no mating shapes!?");
DUMP_TREE(enp);
}

			if( ! SCALAR_SHAPE(enp->en_child[0]->en_shpp) )
				CHECK_BITMAP_ARG(enp,0);
			if( ! SCALAR_SHAPE(enp->en_child[1]->en_shpp) )
				CHECK_BITMAP_ARG(enp,1);

			/* enp->en_bm_child_shpp = getbuf(sizeof(Shape_Info)); */

			break;

		case T_BOOL_NOT:				/* compile_node */
			if( ! SCALAR_SHAPE(enp->en_child[0]->en_shpp) )
				CHECK_BITMAP_ARG(enp,0);
			break;

		case T_UMINUS:		/* compile_node */
			if( enp->en_child[0]->en_shpp == NO_SHAPE ){
	WARN("compile_node:  uminus arg has not shape!?");
				break;		/* an error */
			}

			if( ! SCALAR_SHAPE(enp->en_child[0]->en_shpp) ){
				enp->en_code = T_VS_FUNC;
				enp->en_vfunc_code = FVSMUL;
				enp->en_child[1] = &minus_one_node;
			} else {
				/* If the child node is a literal, just change the value */
				switch(enp->en_child[0]->en_code){
					case T_LIT_INT:
						enp->en_child[0]->en_intval *= -1;
						goto excise_uminus;

					case T_LIT_DBL:
						enp->en_child[0]->en_dblval *= -1;
						goto excise_uminus;

excise_uminus:
						/* We just return the new node & fix links later... */
#ifdef FOOBAR
						/* Now replace the UMINUS node with this one...
						 * The hard part is updating the reference in
						 * the parent...
						 */
						for(i=0;i<MAX_NODE_CHILDREN;i++){
							if( enp->en_parent->en_child[i] == enp ){
								enp->en_parent->en_child[i] = enp->en_child[0];
								enp->en_child[0]->en_parent = enp->en_parent;
								/* Now we can release the UMINUS node */
							/* BUG but we don't seem to have a release func? */
								i=MAX_NODE_CHILDREN; /* will be incremented */
							}
						}
#ifdef CAUTIOUS
						/* make sure we replaced the node */
						if( i == MAX_NODE_CHILDREN )
					ERROR1("CAUTIOUS:  compile node:  couldn't remove UMINUS node!?");
#endif /* CAUTIOUS */
#endif /* FOOBAR */
						enp = enp->en_child[0];
						break;

					default:
						enp->en_code = T_TIMES;
						enp->en_child[1] = &minus_one_node;
						enp=COMPILE_NODE(&enp);	/* necessary? */
						break;
				}
			}
			break;


		case T_BITCOMP:		/* compile_node */
			if( enp->en_child[0]->en_shpp == NO_SHAPE ){
	WARN("compile_node:  bitcomp arg has not shape!?");
				break;		/* an error */
			}

			if( ! SCALAR_SHAPE(enp->en_child[0]->en_shpp) ){
				enp->en_code = T_VCOMP;
			VERIFY_DATA_TYPE(enp,ND_FUNC,"compile_node T_BITCOMP")
				enp->en_vfunc_code = FVCOMP;
			}
			break;



		case T_VS_FUNC:
			if( GET_MATING_SHAPES(enp,0,1) == NO_SHAPE ) {
				/* We shouldn't have to check for outer product shapes with vector-scalar */
				CURDLE(enp)
				break;
			}
			break;


		ALL_SCALINT_BINOP_CASES
		OTHER_SCALAR_MATHOP_CASES		/* compile_node */
		/* why not integer math? */

			/* first check for a special case: */
			if( enp->en_code == T_PLUS && enp->en_child[0]->en_prec == PREC_CHAR &&
				enp->en_child[1]->en_prec == PREC_CHAR ){

				enp->en_code = T_STRING_LIST;
				break;
			}

			/* We check the shapes of the children, and use the info
			 * to change the code to a vector-vector or vector-scalar op
			 * if necessary.
			 */

			/* Now get_mating_shapes handles outer shapes too */
			if( GET_MATING_SHAPES(enp,0,1) == NO_SHAPE ) {
				CURDLE(enp)
				break;
			}

			/* for the arithmetic opcodes, determine if we
			 * need to change the node code
			 */

			check_arith_code(QSP_ARG  enp);
/*
sprintf(error_string,"compile_node binop:  %s",node_desc(enp));
advise(error_string);
DUMP_TREE(enp);
*/
			break;


		case T_SS_S_CONDASS:		/* compile_node */
			/* The parser uses T_SS_S_CONDASS code for all
			 * conditional assignments;
			 * Here we look at the shape of the children
			 * and decide what the code should really be.
			 */

			/* The second and third args are the sources,
			 * and have to have a precision that matches
			 * the destination... If we make them match
			 * each other, then we may need to insert
			 * an unnecessary cast.
			 * (e.g.  byt1 = test ? flt : byt2 ;
			 * In the above example, byt2 gets promoted to float, and
			 * then the whole mess gets cast back to byte - better
			 * to cast the float to byte...
			 */

			/* BUT the boolean arg can be larger,
			 * so we have to look at it too!
			 */

			if( GET_MATING_SHAPES(enp,1,2) == NO_SHAPE ){
				return(enp);
			}



			/* If the two possible values are both scalars,
			 * then get_mating shapes will assign a scalar
			 * shape to the T_SS_S_CONDASS node.  If the
			 * test is a vector, then we need to fix this.
			 */

			/* The test node is either a scalar or a bitmap array */

			if( ! SCALAR_SHAPE(enp->en_child[0]->en_shpp) ){	/* control word is vector */
				if( SCALAR_SHAPE(enp->en_shpp) ){
					/* both sources (targets?) are scalars, but the bitmap is a vector */
					COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
					xform_from_bitmap(enp->en_shpp,enp->en_child[1]->en_prec);
					enp->en_code = T_SS_B_CONDASS;
				} else {
					/* The source expression is vector, and so is the bitmap */
					CHECK_MATING_BITMAP(enp,enp->en_child[0]);
					CHECK_XX_V_CONDASS_CODE(enp);
				}
				enp->en_bm_child_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
			} else {					/* control word is a scalar */
				if( ! SCALAR_SHAPE(enp->en_shpp) ){
					/* at least one target is a vector */
					CHECK_XX_S_CONDASS_CODE(enp);
					enp->en_bm_child_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
				}
				/* else everything is a scalar, code stays the same */
			}

			break;

		case T_MAXVAL:		/* compile_node */
		case T_MINVAL:

			/* max and min can take a list of expressions */
			if( enp->en_child[0]->en_code == T_EXPR_LIST ){
				/* 2 or more args,
				 * this next call will definitely change the opcode.
				 *
				 * At least, with the old veclib it did...  What should we
				 * do now that we support outer ops and projection?
				 */
				CHECK_MINMAX_CODE(enp);

#ifdef CAUTIOUS
				if( enp->en_code == T_MAXVAL || enp->en_code == T_MINVAL ){
					NODE_ERROR(enp);
					WARN("CAUTIOUS:  check_minmax_code did not change opcode!?");
					DUMP_TREE(enp);
				}
#endif /* CAUTIOUS */

			}
			break;

		case T_MATH0_FN:		/* compile_node */
			/* This node has no children, we want to set it's shape from
			 * the LHS, here we assume it's a vector...
			 */
			enp->en_code = T_MATH0_VFN;
			VERIFY_DATA_TYPE(enp,ND_FUNC,"compile_node T_MATH0_FN")
			vf_code = (Vec_Func_Code)
		math0_functbl[enp->en_func_index].fn_vv_code;
			if( vf_code == -1 ){
				sprintf(error_string,
		"Sorry, no vector implementation of math function %s yet",
			math1_functbl[enp->en_func_index].fn_name);
				WARN(error_string);
				CURDLE(enp)
				return(enp);
			}
			/* this overwrites en_func_index, because it is a union! */
			enp->en_vfunc_code = vf_code;
			break;

		case T_MATH1_FN:		/* compile_node */
			if( enp->en_child[0]->en_shpp == NO_SHAPE ){
				WARN("compile_node:  child has no shape");
				break;
			}

			/* The math functions take double args, so if the child
			 * node is not a scalar, and is not FLOAT_PREC, we have
			 * to cast it.  We cast to float instead of double to
			 * save on memory.
			 */

			if( (! SCALAR_SHAPE(enp->en_child[0]->en_shpp) ) &&
				! FLOATING_PREC(enp->en_child[0]->en_shpp->si_prec) ){
				TYPECAST_CHILD(enp,0,PREC_SP);
			}

			/* We should never have to change T_MATH1_VFN
			 * to T_MATH1_FN, because the parser only uses
			 * T_MATH1_FN.
			 */

			if( IS_VECTOR_NODE(enp->en_child[0]) ){
				enp->en_code = T_MATH1_VFN;
				VERIFY_DATA_TYPE(enp,ND_FUNC,"compile_node T_MATH1_FN")
				vf_code = (Vec_Func_Code)
			math1_functbl[enp->en_func_index].fn_vv_code;
				if( vf_code == -1 ){
					sprintf(error_string,
		"Sorry, no vector implementation of math function %s yet",
			math1_functbl[enp->en_func_index].fn_name);
					WARN(error_string);
					CURDLE(enp)
					return(enp);
				}
			/* this overwrites en_func_index, because it is a union! */
				enp->en_vfunc_code = vf_code;
			}

			break;


		case T_MATH2_FN:		/* compile_node */
			VERIFY_DATA_TYPE(enp,ND_FUNC,"compile_node T_MATH2_FN")
			if( enp->en_child[0] == NO_VEXPR_NODE ||
				enp->en_child[1] == NO_VEXPR_NODE ){
				/* parse error - should zap this node? */
				CURDLE(enp)
				return(enp);
			}

			/* cast children if necessary (see MATH_VFN above) */
			/* do we really want to cast if unknown??? */

			if( (! SCALAR_SHAPE(enp->en_child[0]->en_shpp) ) &&
				! FLOATING_PREC(enp->en_child[0]->en_shpp->si_prec) ){

				TYPECAST_CHILD(enp,0,PREC_SP);
			}

			if( (! SCALAR_SHAPE(enp->en_child[1]->en_shpp) ) &&
				! FLOATING_PREC(enp->en_child[1]->en_shpp->si_prec) ){

				TYPECAST_CHILD(enp,1,PREC_SP);
			}

			if( GET_MATING_SHAPES(enp,0,1) == NO_SHAPE ){
				CURDLE(enp)
				break;
			}

			/* vector-scalar functions:
			 * If the first arg is the vector, we use the first
			 * version of the function, otherwise the second.
			 */

			if( SCALAR_SHAPE(enp->en_child[0]->en_shpp) ){
				if( ! SCALAR_SHAPE(enp->en_child[1]->en_shpp) ){
					Vec_Expr_Node *tmp_enp;

					enp->en_code = T_MATH2_VSFN;

					/* switch order, scalar is 2nd */
					/* BUG this is only good for functions where
					 * the order of the operands doesn't matter,
					 * like max and min.  pow() and atan2()
					 * on the other hand...
					 * But at the moment, their vs versions
					 * have not been written...
					 */
					tmp_enp = enp->en_child[0];
					enp->en_child[0] = enp->en_child[1];
					enp->en_child[1] = tmp_enp;
					/* We need to make sure to use the second vsfunc code! */
					vf_code = (Vec_Func_Code)
			math2_functbl[enp->en_func_index].fn_vs_code2;
				} else {
					/* both scalars; leave it alone */
					vf_code = (Vec_Func_Code)-2;	/* flag value for later... */
				}
			} else {
				if( SCALAR_SHAPE(enp->en_child[1]->en_shpp) ){
					enp->en_code = T_MATH2_VSFN;
					vf_code = (Vec_Func_Code)
			math2_functbl[enp->en_func_index].fn_vs_code;
				} else {
					enp->en_code = T_MATH2_VFN;

					vf_code = (Vec_Func_Code)
			math2_functbl[enp->en_func_index].fn_vv_code;
					if( vf_code == -1 ){
						sprintf(error_string,
		"Sorry, no vector implementation of math function %s yet",
				math2_functbl[enp->en_func_index].fn_name);
						WARN(error_string);
						CURDLE(enp)
						return(enp);
					}
				}
			}
			/* what does a code of -2 mean??? */
			if( vf_code != (-2) ){
				if( vf_code < 0 ){
					sprintf(error_string,
		"Sorry, no vector-scalar implementation of math function %s yet",
					math2_functbl[enp->en_func_index].fn_name);
					WARN(error_string);
					CURDLE(enp)
					return(enp);
				}
advise("overwriting en_func_index with vf_code");
				enp->en_vfunc_code = vf_code;
			}

			break;

		case T_RETURN:				/* compile_node */
			if( curr_srp == NO_SUBRT ){
				NODE_ERROR(enp);
				advise("return statement occurs outside of subroutine");
				CURDLE(enp)
				break;
			}
			if( curr_srp->sr_prec == PREC_VOID ){
				if( enp->en_child[0] != NO_VEXPR_NODE ){
					NODE_ERROR(enp);
					sprintf(error_string,
						"void subroutine %s can't return an expression",
						curr_srp->sr_name);
					advise(error_string);
					CURDLE(enp)
				}
			} else {
				if( enp->en_child[0] == NO_VEXPR_NODE ){
					NODE_ERROR(enp);
					sprintf(error_string,
						"subroutine %s returns without an expression",
						curr_srp->sr_name);
					advise(error_string);
					CURDLE(enp)
				}
			}
			enp->en_srp = curr_srp;
			break;

		case T_SIZE_FN:
			/* The integer is the index of the function in the table (see support/function.c)
			 * but we want to use it as a dimension index.  This requires having the function
			 * table in the correct order.  This code here handles that fact that we
			 * have two entries ("depth" and "ncomps") that refer to the same function.
			 */
			VERIFY_DATA_TYPE(enp,ND_FUNC,"compile_node T_SIZE_FN")
			if( enp->en_func_index == N_DIMENSIONS )
				enp->en_func_index=0;
#ifdef CAUTIOUS
			if( enp->en_func_index < 0 || enp->en_func_index >= N_DIMENSIONS ){
				NODE_ERROR(enp);
				sprintf(error_string,"CAUTIOUS:  unexpected size function index %ld",enp->en_func_index);
				WARN(error_string);
			}
#endif /* CAUTIOUS */
			break;

		case T_ASSIGN:			/* compile_node */
			/* We should already have the shapes and precisions
			 * of the children, we check here to see if we have to
			 * insert a TYPECAST node.
			 */

			/* Make sure that the left and right hand precisions match.
			 * If not, then insert a typecast node on the right.
			 *
			 * BUG an exception is the special mixed integer modes,
			 * e.g. char + char = short, short + short = long FIXME
			 * Also complex/real functions like rdft.
			 *
			 * Here is also where we do the magic that allows vector
			 * subscripts on the RHS, i.e.  img[coords]=samples;
			 * by transforming to render(img,coords,samples);
			 */

#ifdef CAUTIOUS
			if( enp->en_child[1]->en_shpp == NO_SHAPE ){
				sprintf(error_string,"CAUTIOUS:  compile_node %s:  RHS %s has no shape!?",
					node_desc(enp),node_desc(enp->en_child[1]));
				WARN(error_string);
				CURDLE(enp)
				DUMP_TREE(enp);
				break;
			}
#endif /* CAUTIOUS */

			if( (!SCALAR_SHAPE(enp->en_child[1]->en_shpp)) &&
				/* next line added for matlab */
				enp->en_child[0]->en_shpp != NO_SHAPE &&
				/* the shape may be unknown, but we still know the precision!? */
				/*
				( ! UNKNOWN_SHAPE( enp->en_child[0]->en_shpp ) ) &&
				*/
				enp->en_child[0]->en_prec != enp->en_child[1]->en_prec ){
/*
sprintf(error_string,"compile_node ASSIGN:  casting %s to precision %s of %s",
node_desc(enp->en_child[1]),
prec_name[enp->en_child[0]->en_prec],
node_desc(enp->en_child[0]));
advise(error_string);
*/
#ifdef DEBUG
if( debug & cast_debug ){
DUMP_TREE(enp);
}
#endif /* DEBUG */
				TYPECAST_CHILD(enp,1,enp->en_child[0]->en_prec);
			}

			/* Check for vector subscripts on LSH */
			if( has_vector_subscript(enp->en_child[0]) )
				enp=fix_render_code(QSP_ARG  enp);
			break;

		case T_COMP_LIST:		/* compile_node */
		case T_ROW_LIST:		/* compile_node */
			/* Like EXPR_LIST case handled below, but precisions of elements have to match */
			CHECK_TYPECAST(enp,0,1);
			enp->en_n_elts = leaf_count(enp,enp->en_code);
			break;

		ALL_LIST_NODE_CASES		/* compile_node */
			/*
			balance_list(enp,enp->en_code);
			*/
			enp->en_n_elts = leaf_count(enp,enp->en_code);
			break;


		case T_INNER:			/* compile_node */
			/* If the child nodes are both scalars, change to TIMES? */
			if( (enp->en_child[0]->en_shpp != NO_SHAPE &&
				SCALAR_SHAPE(enp->en_child[0]->en_shpp) ) ||
			    ( enp->en_child[1]->en_shpp != NO_SHAPE &&
				SCALAR_SHAPE(enp->en_child[1]->en_shpp) ) ){

				enp->en_code = T_TIMES;
				enp = COMPILE_NODE(&enp);	/* to get the shape set... */

			}
			break;

		/* all these cases are do-nothings... */

		case T_SUBRT:
		case T_EXIT:
		case T_CALLFUNC:		/* compile_node */
		case T_CALL_NATIVE:
		case T_EXP_PRINT:
		case T_RAMP:
		case T_STATIC_OBJ:		/* compile_node */
		case T_DYN_OBJ:			/* compile_node */
		case T_UNDEF:			/* undefined object */
		case T_SUBVEC:
		case T_CSUBVEC:
		case T_SUBSAMP:
		case T_CSUBSAMP:
		case T_LIT_INT:
		case T_LIT_DBL:
		case T_SET_FUNCPTR:
		case T_CURLY_SUBSCR:	/* compile_node */
		case T_SQUARE_SUBSCR:		/* compile_node */
		case T_SCRIPT:
		case T_STRING:
		case T_CONJ:
		case T_SUM:
		case T_WARN:
		case T_ADVISE:
		case T_END:			/* compile_node (matlab) */
		case T_TRANSPOSE:
		case T_REAL_PART:
		case T_IMAG_PART:
		case T_OBJ_LOOKUP:
		case T_REFERENCE:	/* compile_node */
		case T_SET_PTR:
		case T_SET_STR:
		case T_NAME_FUNC:
		case T_PROTO:
		//ALL_DECL_STAT_CASES
		ALL_DECL_CASES
		//case T_EXTERN_DECL:	/* for immediate execution... */
		case T_CLR_OPT_PARAMS:
		case T_ADD_OPT_PARAM:
		case T_OPTIMIZE:
		case T_POINTER:
		case T_EQUIVALENCE:
		case T_FUNCPTR:
		case T_FUNCREF:
		case T_STR_PTR:
		case T_MAX_INDEX:
		case T_MIN_INDEX:
		case T_DEREFERENCE:
		case T_INDIR_CALL:
		case T_POSTINC:
		case T_PREINC:
		case T_POSTDEC:
		case T_PREDEC:
		case T_BOOL_PTREQ:
		case T_FIX_SIZE:
		case T_BADNAME:		/* compile_node */
		case T_MAX_TIMES:
		case T_TYPECAST:
		case T_ENLARGE:
		case T_REDUCE:
		case T_ENTIRE_RANGE:

		/* we didn't used to compile declarations, but we need to because
		 * of the possibility of auto-initialization w/ expressions.
		 */
		case T_LIST_OBJ:	/* compile_node */
		case T_COMP_OBJ:	/* compile_node */


		SOME_MATLAB_CASES	/* compile_node */
		ALL_FILEIO_CASES
		ALL_OBJFUNC_CASES
		ALL_STRFUNC_CASES
		ALL_DFT_CASES
		ALL_UNARY_CASES
		ALL_CTL_FLOW_CASES
			/* decl s get their shapes when they're evaluated? */

			break;

		default:		/* compile_node */
			MISSING_CASE(enp,"compile_node");
			DUMP_TREE(enp);
			break;
	}

	/* if we changed a node, splice it in... */

	if( enp != *enpp ){	/* did we change it?  If so, fix links... */
		if( (*enpp)->en_parent != NO_VEXPR_NODE ){
			int i;
			i = which_child(*enpp);
			(*enpp)->en_parent->en_child[i] = enp;
		}
		enp->en_parent = (*enpp)->en_parent;
		*enpp = enp;
	}

	return(enp);

} /* end compile_node() */


/* enp is a T_MAXVAL/T_MINVAL node, w/ an T_EXPR_LIST child.
 * We need to determine (from the shapes of the children)
 * what the code should be.
 *
 * This also gets called recursively, if we have more than two args:
 * the EXPR_LIST node can get changed to a MINMAX node...
 */

static void check_minmax_code(QSP_ARG_DECL   Vec_Expr_Node *enp)
{
	int i;

	/* We can ditch the first exprlist node */

	VERIFY_DATA_TYPE(enp,ND_FUNC,"check_minmax_code")

	if( enp->en_child[0]->en_code == T_EXPR_LIST ){
		enp->en_child[0]->en_child[1]->en_parent = enp;
		enp->en_child[0]->en_child[0]->en_parent = enp;
		enp->en_child[1] = enp->en_child[0]->en_child[1];
		enp->en_child[0] = enp->en_child[0]->en_child[0];
		/* BUG ? do we ever deallocate the EXPR_LIST node?? */
		/* Why not leave it in, anyway? */
	}
else if( enp->en_child[1]->en_code == T_EXPR_LIST ){
WARN("check_minmax_code:  EXPR_LIST in child[1]!?");
}

	/* One of the new children could be a list... */
	/* We only should need to check one, but which one?  BUG */
	for(i=0;i<2;i++){
		if( enp->en_child[i]->en_code == T_EXPR_LIST ){
			enp->en_child[i]->en_code = enp->en_code;
			discard_node_shape(enp->en_child[i]);
			CHECK_MINMAX_CODE(enp->en_child[i]);
			/* BUG? or update_node_shape??? */
			PRELIM_NODE_SHAPE(enp);
		}
	}

	/* comparing two vectors,
	 * or a vector and a scalar?
	 */
	if( GET_MATING_SHAPES(enp,0,1) == NO_SHAPE ){
		WARN("check_minmax_code:  bad shapes!?");
		return;
	}

	/* in the new veclib, we just use T_MAXVAL for everything... or do we?
	 * We use T_MAXVAL instead of PROJECT_OP when we have a single operand
	 * but are not projecting all the way down to a scalar.  We ought to allow
	 * this sort of things with multiple operands as well...
	 */
	if( SCALAR_SHAPE(enp->en_child[1]->en_shpp) ){
		if( SCALAR_SHAPE(enp->en_child[0]->en_shpp) ){
			/* scalar-scalar */
			if( enp->en_code == T_MAXVAL )
				enp->en_code = T_SCALMAX;
			else
				enp->en_code = T_SCALMIN;
		} else {
			/* vector-scalar */
			if( enp->en_code == T_MAXVAL ){
				enp->en_vfunc_code = FVSMAX;
			} else {
				enp->en_vfunc_code = FVSMIN;
			}
			enp->en_code = T_VS_FUNC;
		}
	} else {
		if( SCALAR_SHAPE(enp->en_child[0]->en_shpp) ){
			Vec_Expr_Node *tmp_enp;

			/* scalar-vector */
			if( enp->en_code == T_MAXVAL ){
				enp->en_vfunc_code = FVSMAX;
			} else {
				enp->en_vfunc_code = FVSMIN;
			}
			enp->en_code = T_VS_FUNC;

			/* switch order of children */

			tmp_enp = enp->en_child[0];
			enp->en_child[0] = enp->en_child[1];
			enp->en_child[1] = tmp_enp;
		} else {
			/* vector-vector */
			if( enp->en_code == T_MAXVAL ){
				enp->en_vfunc_code = FVMAX;
			} else {
				enp->en_vfunc_code = FVMIN;
			}
			enp->en_code = T_VV_FUNC;
			UPDATE_NODE_SHAPE(enp);
		}
	}
	if( enp->en_code == T_VS_FUNC ){
		/* vs_func nodes point to their shapes - BUT maxval nodes
		 * own it...
		 */
		discard_node_shape(enp);
		POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
	}

} /* end check_minmax_code */

void init_fixed_nodes(SINGLE_QSP_ARG_DECL)
{
	static int fixed_nodes_inited=0;

	int i;

	if( fixed_nodes_inited ) return;

	init_expr_node(QSP_ARG  &minus_one_node);
	minus_one_node.en_code = T_LIT_DBL;
	minus_one_node.en_dblval = (-1.0);
	POINT_NODE_SHAPE(&minus_one_node, scalar_shape(PREC_DP) );

	for(i=0;i<N_NAMED_PRECS;i++){
		_uk_shpp[i]=NO_SHAPE;
		_scalar_shpp[i]=NO_SHAPE;
	}
}

static void scalarize(Shape_Info *shpp)
{
	int i;

	for(i=0;i<N_DIMENSIONS;i++){
		shpp->si_mach_dim[i]=1;
		shpp->si_type_dim[i]=1;
	}
	shpp->si_n_mach_elts = 1;
	shpp->si_n_type_elts = 1;
}

/********** scantree.c read in below this line **************/

/* scantree.c
 *
 * Scanning the tree does a number of things:
 * It figures out the "shape" of each node, that is,
 * the dimensions of the object the node represents.
 *
 * Should nodes "own" their shape structs?
 * For Objects, we can't just point to dp->dt_shape, because
 * the objects can be dynamically created and destroyed.
 * So we point references to the shape at the declaration
 * node for an object...
 *
 * Which nodes need shape info?  Primarly, all the constituents
 * of expressions, to allow error checking.
 * Object declaration nodes - copy from object, overwrite during rescan
 * Object reference nodes - point to decl node shape
 *
 * Subroutine shape is dynamically set...
 *
 * The really tricky part here is resolution of unknown shapes.
 * Assume that u and v are unknown, and k is known...
 * There are several methods:
 *
 *	v=u+k;		u gets shape of k (resolve_node)
 *			v gets shape of u+k (resolve_from_rhs)
 *
 *	v=func(k);	v gets shape of func() (scan_subrt())
 *
 *	k=func(v);	uko's inside func get shape from k (resolve_from_lhs)
 *			v gets shape from func arg decl (resolve_arg_shapes)
 *
 * Unknown shape objects can be:
 *	local objects (declared within a function)
 *	arguments
 *
 * Currently, the functions of scanning, verifying, and resolving are
 * rather messily intertwined...  It would probably be helpful to try
 * and separate them, even if it is a bit wasteful computationally.
 * It would also be good to try and determine HOW uko's will be resolved,
 * to save time when we actually are going to do the resolution...
 *
 * We need to have a good strategy for forgetting run-time resolutions
 * of unknown objects...  There may be nodes (dft, etc) which have
 * their own shape copies, which are derived from uko shapes...
 * (Nodes which have the shape of the uko will point to the uko shape,
 * and so be automatically set when the uko is resolved, but computed
 * copies will not be updated like this...)
 *
 *	scan		point nodes to uko's, make list of uko's
 *
 *	resolve		determine shapes of uko's
 *
 *	verify		check for errors...
 *
 * One problem with this approach is that, if we have multiple calls
 * to a subrt, with different shape contexts, there will be two
 * distinct resolutions in that subrt...  we can't have them both
 * sitting around at the same time to be verified.
 */

Shape_Info *scalar_shape(prec_t prec)
{
	int index;

	index = PREC_INDEX(prec);	/* does the correct thing for pseudo prec's */

	if( _scalar_shpp[index]!=NO_SHAPE ){
		return(_scalar_shpp[index]);
	}

	_scalar_shpp[index] = (Shape_Info *)getbuf(sizeof(Shape_Info));
	scalarize(_scalar_shpp[index]);

	/* should some of these things be put in scalarize? */
	_scalar_shpp[index]->si_flags = 0;
	/* _scalar_shpp[index]->si_rowpad = 0; */

	_scalar_shpp[index]->si_prec = prec;

	SET_SHAPE_FLAGS(_scalar_shpp[index],NO_OBJ);

	return(_scalar_shpp[index]);
}

Shape_Info *uk_shape(prec_t prec)
{
	int i;
	int i_prec;

	if( prec < N_MACHINE_PRECS ){
		i_prec = prec;
	} else if( BITMAP_PRECISION(prec) ){
		i_prec = N_MACHINE_PRECS + PP_BIT;
	} else if( STRING_PRECISION(prec) ){
		/* Not sure if we still use string precision... */
		i_prec = N_MACHINE_PRECS + PP_STRING;
	} else if( CHAR_PRECISION(prec) ){
		i_prec = N_MACHINE_PRECS + PP_CHAR;
	} else if( prec == PREC_VOID ){
		i_prec = N_MACHINE_PRECS + PP_VOID;
	}
#ifdef CAUTIOUS
	  /* Why are the complex cases CAUTIOUS??? */
	  else if( COMPLEX_PRECISION(prec) ){
		if( (prec& MACH_PREC_MASK) == PREC_SP )
			i_prec = N_MACHINE_PRECS + PP_CPX;
		else if( (prec&MACH_PREC_MASK) == PREC_DP )
			i_prec = N_MACHINE_PRECS + /* PP_DBLCPX */ PP_CPX ; /* BUG? double complex prec or type? */
		else {
			sprintf(DEFAULT_ERROR_STRING,"uk_shape:  No complex support for machine precision %s",
				prec_name[prec&MACH_PREC_MASK]);
			NERROR1(DEFAULT_ERROR_STRING);
			i_prec=0;	/* silence compiler warning NOTREACHED */
		}
	} else if( QUAT_PRECISION(prec) ){
		if( (prec& MACH_PREC_MASK) == PREC_SP )
			i_prec = N_MACHINE_PRECS + PP_QUAT;
		else if( (prec&MACH_PREC_MASK) == PREC_DP )
			i_prec = N_MACHINE_PRECS + /* PP_DBLCPX */ PP_QUAT ; /* BUG? double complex prec or type? */
		else {
			sprintf(DEFAULT_ERROR_STRING,"uk_shape:  No quaternion support for machine precision %s",
				prec_name[prec&MACH_PREC_MASK]);
			NERROR1(DEFAULT_ERROR_STRING);
			i_prec=0;	/* silence compiler warning NOTREACHED */
		}
	} else {
		sprintf(DEFAULT_ERROR_STRING,"prec is %s (0x%x)",name_for_prec(prec),prec);
		advise(DEFAULT_ERROR_STRING);
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  uk_shape:  don't know how to handle pseudo prec  %d (0%o 0x%x)",
			prec,prec,prec);
		NERROR1(DEFAULT_ERROR_STRING);
		i_prec=0;	/* silence compiler warning NOTREACHED */
	}
#endif /* CAUTIOUS */

	if( _uk_shpp[i_prec]!=NO_SHAPE ){
		return(_uk_shpp[i_prec]);
	}

	_uk_shpp[i_prec] = (Shape_Info *)getbuf(sizeof(Shape_Info));
	for(i=0;i<N_DIMENSIONS;i++)
		_uk_shpp[i_prec]->si_type_dim[i]=0;

	/* SET_SHAPE_FLAGS(_uk_shpp[i_prec],NO_OBJ); */
	_uk_shpp[i_prec]->si_flags = DT_UNKNOWN_SHAPE;
	_uk_shpp[i_prec]->si_prec = prec;
	/* Set BIT & COMPLEX flags if necessary */
	if( COMPLEX_PRECISION(prec) )
		_uk_shpp[i_prec]->si_flags |= DT_COMPLEX;
	if( BITMAP_PRECISION(prec) )
		_uk_shpp[i_prec]->si_flags |= DT_BIT;

	return(_uk_shpp[i_prec]);
}

Shape_Info *void_shape(prec_t prec)
{
	Shape_Info *shpp;

	if( _void_shpp[prec] != NO_SHAPE )
		return(_void_shpp[prec]);

	shpp=uk_shape(prec);
	_void_shpp[prec]=(Shape_Info *)getbuf(sizeof(Shape_Info));
	*_void_shpp[prec] = *shpp;
	_void_shpp[prec]->si_flags = DT_VOID;
	return(_void_shpp[prec]);
}

int shapes_match(Shape_Info *shpp1,Shape_Info *shpp2)
{
	int i;

	if( shpp1 == NO_SHAPE ){
		if( shpp2 == NO_SHAPE ) return(1);
		else return(0);
	} else if( shpp2 == NO_SHAPE ) return(0);

	for(i=0;i<N_DIMENSIONS;i++)
		if( shpp1->si_type_dim[i] != shpp2->si_type_dim[i] )
			return(0);
	return(1);
}


/* Get the shape info for the designated child node.
 * If the child node has no shape (usually due to an input error)
 * return NO_SHAPE, and deallocate the node's own shape info
 * struct.  If all is well, a new shape_info will be allocated
 * for this node if needed.
 */

static Shape_Info * get_child_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,int child_index)
{
	Shape_Info *shpp;
	Vec_Expr_Node *enp2;

	enp2=enp->en_child[child_index];
	shpp=enp2->en_shpp;

	if( shpp==NO_SHAPE ){

		/* an error deeper in the tree... */

		/* We make sure that this node has no shape,
		 * but is this really necessary?
		 * It seems as if we are calling get_child_shape()
		 * in order to *set* its shape...
		 *
		 * Object nodes *might* have their shape set
		 * if the object has ben declared already,
		 * but probably not if it's an external object.
		 */

		if( enp2->en_code == T_DYN_OBJ ){
			Data_Obj *dp;

			dp=DOBJ_OF(enp2->en_string);
			if( dp == NO_OBJ ){
				sprintf(error_string,
					"Missing obj %s",enp2->en_string);
				WARN(error_string);
			} else {
				/* BUG should we use the decl node?? */
				shpp = &dp->dt_shape;
				return(shpp);
			}
		}

		discard_node_shape(enp);

		return( NO_SHAPE );
	}
	return(shpp);
}

#ifdef CAUTIOUS
int insure_child_shape(Vec_Expr_Node *enp, int child_index)
{
	if( enp->en_child[child_index]->en_shpp == NO_SHAPE ){
		sprintf(DEFAULT_ERROR_STRING,
			"CAUTIOUS:  insure_child_shape:  %s has no shape!?",
			node_desc(enp->en_child[child_index]));
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	return(0);
}
#endif /* CAUTIOUS */



/* This is a special case of check_uk_links */

#define CHECK_UK_CHILD(enp,index)	check_uk_child(QSP_ARG  enp,index)

static void check_uk_child(QSP_ARG_DECL  Vec_Expr_Node *enp,int index)
{
	if( ( UNKNOWN_SHAPE(enp->en_shpp) && ! SCALAR_SHAPE(enp->en_child[index]->en_shpp) )
			|| UNKNOWN_SHAPE(enp->en_child[index]->en_shpp) ){
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(DEFAULT_ERROR_STRING,"check_uk_child calling link_uk_nodes for %s and child %s",
node_desc(enp),node_desc(enp->en_child[index]));
advise(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */
		LINK_UK_NODES(enp,enp->en_child[index]);
	}
}

#define CHECK_BINOP_LINKS(enp)	check_binop_links(QSP_ARG  enp)

static void check_binop_links(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	CHECK_UK_CHILD(enp,0);
	CHECK_UK_CHILD(enp,1);
}

/* compatible_shape doesn't insist that two objects have the same shape, but if the
 * dimensions do not match then one must be equal to 1.
 * The result shape always has the larger dimension (suitable for outer product, etc).
 */

static Shape_Info *compatible_shape(Vec_Expr_Node *enp1,Vec_Expr_Node *enp2,Vec_Expr_Node *enp)
{
	int i;
	Shape_Info si,*shpp;

	for(i=0;i<N_DIMENSIONS;i++){
		if(	enp1->en_shpp->si_type_dim[i] != 1 &&
			enp2->en_shpp->si_type_dim[i] != 1 &&
			enp1->en_shpp->si_type_dim[i] != enp2->en_shpp->si_type_dim[i] ){

			/* We put in a special case here to handle the assigment of a string to
			 * a row vector of bytes...
			 */
			if( i == 1 && enp1->en_shpp->si_prec == PREC_CHAR && enp2->en_shpp->si_prec == PREC_CHAR &&
				enp1->en_shpp->si_type_dim[i] > enp2->en_shpp->si_type_dim[i] ){
				/* allow this */
			} else {
advise("compatible_shape:  enp1");
describe_shape(enp1->en_shpp);
advise("compatible_shape:  enp2");
describe_shape(enp2->en_shpp);
				return(NO_SHAPE);
			}
		}

		if( enp1->en_shpp->si_type_dim[i] > enp2->en_shpp->si_type_dim[i] )
			si.si_type_dim[i] = enp1->en_shpp->si_type_dim[i];
		else
			si.si_type_dim[i] = enp2->en_shpp->si_type_dim[i];
	}
	shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
	*shpp = si;
	shpp->si_flags = 0;
	scalarize(shpp);	/* set all dimensions to 1 */

	if( !dominance_table_inited ) init_dominance_table();
	shpp->si_prec = dominance_tbl[ enp1->en_shpp->si_prec & MACH_PREC_MASK ]
					[ enp2->en_shpp->si_prec & MACH_PREC_MASK ];
	SET_SHAPE_FLAGS(shpp,NO_OBJ);
	return(shpp);
}


/* if the shapes match, or either one is a scalar, return a pointer
 * to the larger shape.
 * Otherwise, return NO_SHAPE.
 *
 * modified this to support generalized outer binops!
 *
 * We don't use the final node except to print the line number if there is an error!
 *
 * Originally, we required that the shapes either matched, or one had to be a scalar.
 * But now, if neither is a scalar we return a shape appropriate for an outer binop.
 */

static Shape_Info * shapes_mate(QSP_ARG_DECL  Vec_Expr_Node *enp1,Vec_Expr_Node *enp2,Vec_Expr_Node *enp)
{
	/* We do this test first, so in the case of scalar assignment,
	 * shapes_mate() returns the shape ptr of the destination...
	 */

	if( SCALAR_SHAPE(enp2->en_shpp) ){
		return(enp1->en_shpp);
	} else if( SCALAR_SHAPE(enp1->en_shpp) ){
		return(enp2->en_shpp);
	} else if( same_shape(enp1->en_shpp,enp2->en_shpp) )
		return(enp1->en_shpp);

	/* At this point, we know that either one of the shapes is unknown,
	 * or they have different shapes...
	 */

	/* If one of the shapes is unknown, return the known shape. 
	 */

	if( UNKNOWN_SHAPE(enp2->en_shpp) ){
		/* return(enp2->en_shpp); */
		return(enp1->en_shpp);
	}
	if( UNKNOWN_SHAPE(enp1->en_shpp) ){
		/* return(enp1->en_shpp); */
		return(enp2->en_shpp);
	}

	/* what if they have the same shape, but one is real, one is complex? */
	/* BUG?  should we check which kind of operation? */
	if( COMPLEX_SHAPE(enp1->en_shpp) && ! COMPLEX_SHAPE(enp2->en_shpp) ) {
		/* make sure the other dimensions match */
		int i;

		for(i=1;i<N_DIMENSIONS;i++){
			if( enp1->en_shpp->si_type_dim[i] != enp2->en_shpp->si_type_dim[i] )
				goto mismatch;
		}
		return(enp1->en_shpp);
	} else if( COMPLEX_SHAPE(enp2->en_shpp) && ! COMPLEX_SHAPE(enp1->en_shpp) ){
		/* make sure the other dimensions match */
		int i;

		for(i=1;i<N_DIMENSIONS;i++){
			if( enp1->en_shpp->si_type_dim[i] != enp2->en_shpp->si_type_dim[i] )
				goto mismatch;
		}
		return(enp2->en_shpp);
	}

/*
describe_shape(enp1->en_shpp);
describe_shape(enp2->en_shpp);
	advise("shapes_mate:  should we fall through??");
	*/

	/* The shapes don't mate, but they may be ok for an outer binop...
	 * If so, we return the outer shape.
	 */
	{
		static Shape_Info shp;
		int i;

		shp.si_n_type_elts=1;
		for(i=0;i<N_DIMENSIONS;i++){
			if( enp1->en_shpp->si_type_dim[i] == 1 ){
				shp.si_type_dim[i] = enp2->en_shpp->si_type_dim[i];
			} else if( enp2->en_shpp->si_type_dim[i] == 1 ){
				shp.si_type_dim[i] = enp1->en_shpp->si_type_dim[i];
			} else if( enp1->en_shpp->si_type_dim[i] == enp2->en_shpp->si_type_dim[i] ){
				shp.si_type_dim[i] = enp1->en_shpp->si_type_dim[i];
			} else {
				/* mismatch */
				goto mismatch;
			}
			shp.si_n_type_elts *= shp.si_type_dim[i];
		}
		/* we assume the precisions match ...  is this correct?  BUG? */
		shp.si_prec = enp1->en_shpp->si_prec;
		SET_SHAPE_FLAGS(&shp,NO_OBJ);
		return(&shp);	/* BUG?  can we get away with a single static shape here??? */
	}

mismatch:
	NODE_ERROR(enp);
	NWARN("shapes_mate:  Operands have incompatible shapes");
	advise(node_desc(enp));
	/*
	dump_shape(enp1->en_shpp);
	dump_shape(enp2->en_shpp);
	DUMP_TREE(enp);
	*/
#ifdef DEBUG
if( debug ){
//DUMP_TREE(enp);

sprintf(DEFAULT_ERROR_STRING,"flgs1 = 0x%x     flgs2 = 0x%x",
enp1->en_shpp->si_flags&SHAPE_MASK,enp2->en_shpp->si_flags&SHAPE_MASK);
advise(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */

	return(NO_SHAPE);
}

Shape_Info * product_shape(Shape_Info *shpp1,Shape_Info *shpp2)
{
	if( SCALAR_SHAPE(shpp2) ){
		return(shpp1);
	} else if( SCALAR_SHAPE(shpp1) ){
		return(shpp2);
	} else if( same_shape(shpp1,shpp2) )
		return(shpp1);

	/* At this point, we know that either one of the shapes is unknown,
	 * or they have different shapes...
	 */

	/* If one of the shapes is unknown, return the known shape. 
	 */

	if( UNKNOWN_SHAPE(shpp2) ){
		/* return(shpp2); */
		return(shpp1);
	}
	if( UNKNOWN_SHAPE(shpp1) ){
		/* return(shpp1); */
		return(shpp2);
	}

#ifdef FOOBAR
	/* These tests are not needed now that we have type_dim */

	/* what if they have the same shape, but one is real, one is complex? */
	/* BUG?  should we check which kind of operation? */
	if( COMPLEX_SHAPE(shpp1) && ! COMPLEX_SHAPE(shpp2) ) {
		/* make sure the other dimensions match */
		int i;

		for(i=1;i<N_DIMENSIONS;i++){
			if( shpp1->si_type_dim[i] != shpp2->si_type_dim[i] )
				goto mismatch;
		}
		return(shpp1);
	} else if( COMPLEX_SHAPE(shpp2) && ! COMPLEX_SHAPE(shpp1) ){
		/* make sure the other dimensions match */
		int i;

		for(i=1;i<N_DIMENSIONS;i++){
			if( shpp1->si_type_dim[i] != shpp2->si_type_dim[i] )
				goto mismatch;
		}
		return(shpp2);
	}
#endif /* FOOBAR */

/*
describe_shape(shpp1);
describe_shape(shpp2);
	advise("product_shape:  should we fall through??");
	*/

	/* The shapes don't mate, but they may be ok for an outer binop...
	 * If so, we return the outer shape.
	 */
	{
		static Shape_Info shp;
		int i;

		shp.si_n_type_elts=1;
		for(i=0;i<N_DIMENSIONS;i++){
			if( shpp1->si_type_dim[i] == 1 ){
				shp.si_type_dim[i] = shpp2->si_type_dim[i];
			} else if( shpp2->si_type_dim[i] == 1 ){
				shp.si_type_dim[i] = shpp1->si_type_dim[i];
			} else if( shpp1->si_type_dim[i] == shpp2->si_type_dim[i] ){
				shp.si_type_dim[i] = shpp1->si_type_dim[i];
			} else {
				/* mismatch */
				goto mismatch;
			}
			shp.si_n_type_elts *= shp.si_type_dim[i];
		}
		/* BUG - should we set si_n_mach_elts etc? */
		/* we assume the precisions match ...  is this correct?  BUG? */
		shp.si_prec = shpp1->si_prec;
		SET_SHAPE_FLAGS(&shp,NO_OBJ);
		return(&shp);	/* BUG?  can we get away with a single static shape here??? */
	}

mismatch:
	return(NO_SHAPE);
} /* end product_shape */

/* call w/ ARGLIST node, returns the number of arguments in the subtree */

int arg_count(Vec_Expr_Node *enp)
{
	return(leaf_count(enp,T_ARGLIST));
}

Vec_Expr_Node *nth_arg(QSP_ARG_DECL  Vec_Expr_Node *enp, int n)
{
	int l;

	if( enp->en_code != T_ARGLIST ){
		if( n == 0 ) return(enp);
		NODE_ERROR(enp);
		sprintf(DEFAULT_ERROR_STRING,"nth_arg:  %s is not an arg_list node, but arg %d requested",
			node_desc(enp),n);
		NWARN(DEFAULT_ERROR_STRING);
		return(NO_VEXPR_NODE);
	}

	if( n >= (l=leaf_count(enp,T_ARGLIST)) ){
		NODE_ERROR(enp);
		sprintf(DEFAULT_ERROR_STRING,"nth_arg:  %s has %d leaves, but arg %d requested",
			node_desc(enp),l,n);
		NWARN(DEFAULT_ERROR_STRING);
		return(NO_VEXPR_NODE);
	}

	if( n < (l=leaf_count(enp->en_child[0],T_ARGLIST)) ){
		return( nth_arg(QSP_ARG  enp->en_child[0],n) );
	} else {
		return( nth_arg(QSP_ARG  enp->en_child[1],n-l) );
	}
	/* NOTREACHED */
	return(NO_VEXPR_NODE);
}

Identifier *get_named_ptr(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Identifier *idp;

	switch(enp->en_code){
		case T_POINTER:
			idp = ID_OF(enp->en_string);

			if( idp == NO_IDENTIFIER ){
				NODE_ERROR(enp);
				sprintf(error_string,
					"CAUTIOUS:  get_named_ptr:  missing identifier %s",enp->en_string);
				WARN(error_string);
				return(NO_IDENTIFIER);
			}
			if( ! IS_POINTER(idp) ){
				NODE_ERROR(enp);
				sprintf(error_string,
					"CAUTIOUS:  get_named_ptr:  id %s is not a pointer!?",idp->id_name);
				WARN(error_string);
				return(NO_IDENTIFIER);
			}
			return(idp);
		default:
			MISSING_CASE(enp,"get_named_ptr");
			break;
	}
	return(NO_IDENTIFIER);
}

/* link_uk_args
 * For a CALLFUNC node, descend the arg value tree, looking for uk objects that need to be resolved.
 * I'm not sure exatly how this is going to work???
 */

static void link_uk_args(QSP_ARG_DECL  Vec_Expr_Node *call_enp,Vec_Expr_Node *arg_enp)
{
	Data_Obj *dp;

	switch(arg_enp->en_code){
		/* scalar args are uninteresting, their shape is known */
		ALL_SCALAR_FUNCTION_CASES
		case T_BOOL_NE:
		case T_LIT_INT:
		case T_LIT_DBL:
			break;

		
		/* these get handled elsewhere - we hope!? */
		case T_POINTER:
		case T_STR_PTR:
		case T_SET_STR:
		case T_REFERENCE:
		case T_STRING:
		case T_FUNCREF:
		case T_DEREFERENCE:
		/* these look like binop nodes (handled below), but we are
		 * pretty sure that these are scalars???
		 * Maybe we should descend them anyway, since a scalar expression
		 * could contain a function with vector args?
		 */
		case T_MINUS:
		case T_PLUS:
			break;

		case T_DYN_OBJ:					/* link_uk_args */
			dp = DOBJ_OF(arg_enp->en_string);
#ifdef CAUTIOUS
			if( dp == NO_OBJ ){
				NODE_ERROR(arg_enp);
				sprintf(error_string,"Obj Arg %s has no associated object %s!?",
					node_desc(arg_enp),arg_enp->en_string);
				WARN(error_string);
				return;
			}
#endif /* CAUTIOUS */
			if( UNKNOWN_SHAPE(&dp->dt_shape) )
				link_one_uk_arg(call_enp,arg_enp);

			break;

		/* for arglist or binop cases, we descend looking for objects... */
		/*
		ALL_BINOP_CASES
		*/
		case T_VV_FUNC:			/* link_uk_args */
		case T_VS_FUNC:
		case T_ARGLIST:
		case T_ROW:
		/* case T_ROWLIST: */
			link_uk_args(QSP_ARG  call_enp,arg_enp->en_child[0]);
			link_uk_args(QSP_ARG  call_enp,arg_enp->en_child[1]);
			break;

		/* only one child to descend */
		case T_TRANSPOSE:
		case T_TYPECAST:
			link_uk_args(QSP_ARG  call_enp,arg_enp->en_child[0]);
			break;

		default:
			MISSING_CASE(arg_enp,"link_uk_args");
			break;
	}
} /* end link_uk_args */

static void note_uk_objref(QSP_ARG_DECL  Vec_Expr_Node *decl_enp, Vec_Expr_Node *enp)
{
	Node *np;

	VERIFY_DATA_TYPE(decl_enp,ND_DECL,"note_uk_objref")

	if( decl_enp->en_decl_ref_list == NO_LIST )
		decl_enp->en_decl_ref_list = new_list();

	np=mk_node(enp);
	addTail(decl_enp->en_decl_ref_list,np);
}

/* prelim_node_shape		get the node shape if we can,
 *				but don't try to do any resolution.
 *
 * When should we check for errors?
 *
 * Most nodes will be pointed to a shape.
 * But there are exceptions; the following nodes will own their shape,
 * because their shape will differ from that of their child:
 *
 *	list_obj, expr_list
 *	typecast
 *	subscript, real_part, imag_part
 *	transpose
 *	subvec
 *	rdft,ridft
 *	outer
 *	load
 *	callfunc
 *
 * This routine should only be called once per node, after that we call update_node_shape.
 */

static void prelim_node_shape(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp;
	Shape_Info tmp_shape;
	Subrt *srp;
	Vec_Expr_Node *decl_enp;
	dimension_t i1;
	const char *s;
	long n1,n2,n3;

	if( IS_CURDLED(enp) ) return;


	/* now all the child nodes have been scanned, process this one */

	switch(enp->en_code){
		/* matlab */
		case T_SSCANF:
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_DP));
			break;

		/* T_ROWIST merged with T_ROW_LIST - ?? */
		case T_ROW:
		/* case T_ROWLIST: */					/* prelim_node_shape */
			UPDATE_NODE_SHAPE(enp);
			break;

		case T_I:
			/* Do we need to set the complex bit??? */
			POINT_NODE_SHAPE(enp,cpx_scalar_shape(PREC_DP));
			break;

		case T_SS_B_CONDASS:				/* prelim_node_shape */
			/* Get's it's shape from child[0] (bitmap), but precision from
			 * the other children (scalars).
			 * BUG we'd like to determine the precision from the destination.
			 */
			COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			xform_from_bitmap(enp->en_shpp,enp->en_child[1]->en_prec);
			CHECK_UK_CHILD(enp,0);
			break;

		case T_VS_B_CONDASS:				/* prelim_node_shape */
			/* the shape is getting set in compile_node??? */
			/* BUT what about if some of the child shapes are unknown? */
			/* Why is this FOOBAR'd ??? */
#ifdef CONDASS_FOOBAR
			/* copy the shape from the first (vector) node */
			COPY_NODE_SHAPE(enp,enp->en_child[1]->en_shpp);
			CHECK_UK_CHILD(enp,0);
			CHECK_UK_CHILD(enp,1);
			/* If the shape is unknown, but we have the bitmap, then copy from it */
			if( UNKNOWN_SHAPE(enp->en_shpp) && ! UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
				COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
				xform_from_bitmap(enp->en_shpp,enp->en_child[1]->en_prec);
			}
			/* might be an outer... */
sprintf(error_string,"prelim_node_shape %s",node_desc(enp));
advise(error_string);
DUMP_TREE(enp);

sprintf(error_string,"prelim_node_shape %s DONE",node_desc(enp));
advise(error_string);
DUMP_TREE(enp);
#endif /* CONDASS_FOOBAR */
			break;

		case T_VV_S_CONDASS:				/* prelim_node_shape */
			/* shape should already be assigned by get_mating_shapes */
			CHECK_UK_CHILD(enp,1);
			CHECK_UK_CHILD(enp,2);
			break;

		case T_VS_S_CONDASS:				/* prelim_node_shape */
			CHECK_UK_CHILD(enp,1);
			break;

		case T_SS_S_CONDASS:				/* prelim_node_shape */
			break;

		case T_VV_B_CONDASS:				/* prelim_node_shape */
			/* shape should already be assigned by get_mating_shapes */
			CHECK_UK_CHILD(enp,0);
			CHECK_UK_CHILD(enp,1);
			CHECK_UK_CHILD(enp,2);

			if( UNKNOWN_SHAPE(enp->en_shpp) && ! UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
				COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
				xform_from_bitmap(enp->en_shpp, enp->en_child[1]->en_prec);
			}

			break;

		case T_VV_VV_CONDASS:				/* prelim_node_shape */
			/* shape should already be assigned by get_mating_shapes */
			CHECK_UK_CHILD(enp,0);
			CHECK_UK_CHILD(enp,1);
			CHECK_UK_CHILD(enp,2);
			CHECK_UK_CHILD(enp,3);

			if( UNKNOWN_SHAPE(enp->en_shpp) ){
				/* BUG we copy the shape from one of the vector condition
				 * args, but we really should find a mating shape for the pair.
				 * E.g., this will be wrong if we have row < col
				 */
				if( ! UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
					COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
				} else if( ! UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){
					COPY_NODE_SHAPE(enp,enp->en_child[1]->en_shpp);
				}
			}

			break;

		case T_VS_VV_CONDASS:				/* prelim_node_shape */
			/* shape should already be assigned by get_mating_shapes */
			CHECK_UK_CHILD(enp,0);
			CHECK_UK_CHILD(enp,1);
			CHECK_UK_CHILD(enp,2);

			if( UNKNOWN_SHAPE(enp->en_shpp) ){
				/* BUG we copy the shape from one of the vector condition
				 * args, but we really should find a mating shape for the pair.
				 * E.g., this will be wrong if we have row < col
				 */
				if( ! UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
					COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
				} else if( ! UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){
					COPY_NODE_SHAPE(enp,enp->en_child[1]->en_shpp);
				}
			}

			break;

		case T_VV_VS_CONDASS:				/* prelim_node_shape */
			/* shape should already be assigned by get_mating_shapes */
			CHECK_UK_CHILD(enp,0);
			CHECK_UK_CHILD(enp,2);
			CHECK_UK_CHILD(enp,3);

			if( UNKNOWN_SHAPE(enp->en_shpp) ){
				if( ! UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
					COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
				}
			}

			break;

		case T_VS_VS_CONDASS:				/* prelim_node_shape */
			/* shape should already be assigned by get_mating_shapes */
			CHECK_UK_CHILD(enp,0);
			CHECK_UK_CHILD(enp,2);

			if( UNKNOWN_SHAPE(enp->en_shpp) ){
				if( ! UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
					COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
				}
			}

			break;

		case T_UNDEF:
			POINT_NODE_SHAPE(enp,uk_shape(PREC_SP));
			break;

		case T_MAX_TIMES:			/* prelim_node_shape */
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_DI));
			break;

		case T_ENLARGE:		/* prelim_node_shape */
			COPY_NODE_SHAPE(enp, enp->en_child[0]->en_shpp );
			enp->en_shpp->si_cols *= 2;
			enp->en_shpp->si_rows *= 2;
			CHECK_UK_CHILD(enp,0);
			break;

		case T_REDUCE:
			COPY_NODE_SHAPE(enp, enp->en_child[0]->en_shpp );
			enp->en_shpp->si_cols /= 2;
			enp->en_shpp->si_rows /= 2;
			CHECK_UK_CHILD(enp,0);
			break;

		case T_COMP_OBJ:		/* prelim_node_shape */
			COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			/* This is a list of expressions enclosed in curly braces.
			 * The main use is to list pixel components.
			 * We make sure the dimension goes into dimension 0 (type dim)
			 */
			/* The expression list thinks of itself as a row vector... */
			/* BUG - won't work if N_DIMENSIONS is changed */
			/* The child node is a row vector (rowlist), we want to make it a scalar... */
			/* The above comments seem to be true for a ROW_LIST (square braces),
			 * but what about a COMP_LIST (curly braces)???
			 */
			/* When would a COMP_LIST node ever have ROW_LIST children???
			 * ROW_LIST might, but not COMP_LIST...
			 * This block of code can probably be eliminated.
			 * But is it legal to have nexted curly brace exprs?
			 */
			if( enp->en_child[0]->en_code == T_ROW_LIST ){
				enp->en_shpp->si_type_dim[0] = enp->en_shpp->si_type_dim[1];
				enp->en_shpp->si_type_dim[1] = enp->en_shpp->si_type_dim[2];
				enp->en_shpp->si_type_dim[2] = enp->en_shpp->si_type_dim[3];
				enp->en_shpp->si_type_dim[3] = enp->en_shpp->si_type_dim[4];
				enp->en_shpp->si_type_dim[4] = 1;
				SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ);
			}
			/* probably a COMP_LIST node?  do nothing? */
			break;

		case T_LIST_OBJ:		/* prelim_node_shape */
			/* all of the elements of the list
			 * should have the same shape...
			 * BUG?  should we verify?
			 */
			COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			/* We used to increment maxdim here, but the EXPR_LIST child
			 * should already have the proper value...
			 */
			/* BUT - if the list has only one element, we still need to do this! */
			if( enp->en_child[0]->en_code != T_ROW_LIST ){
				enp->en_shpp->si_maxdim++;
			}
			break;

		case T_TYPECAST:				/* prelim_node_shape */
			tmp_shape = *enp->en_child[0]->en_shpp;
			tmp_shape.si_prec = enp->en_cast_prec;
			tmp_shape.si_prec &= ~DT_RDONLY;/* might be const? */
			if( COMPLEX_PRECISION(tmp_shape.si_prec) && !COMPLEX_PRECISION(enp->en_child[0]->en_prec) ){
#ifdef CAUTIOUS
				if( tmp_shape.si_type_dim[0] > 1 ){
					DUMP_TREE(enp);
					NODE_ERROR(enp);
					sprintf(error_string,
			"CAUTIOUS:  prelim_node_shape %s (tdim=%d):  casting to complex, but tdim > 1",
						node_desc(enp),tmp_shape.si_type_dim[0]);
					ERROR1(error_string);
				}
#endif /* CAUTIOUS */
				tmp_shape.si_type_dim[0] = 1;
				tmp_shape.si_mach_dim[0] = 2;
				/* BUG?  should we recalculate n_type_elts and n_mach_elts? */
				tmp_shape.si_flags |= DT_COMPLEX;
			}
			COPY_NODE_SHAPE(enp,&tmp_shape);
			if( UNKNOWN_SHAPE(&tmp_shape) )
				LINK_UK_NODES(enp,enp->en_child[0]);
			break;

		case T_EQUIVALENCE:		/* prelim_node_shape */
			COPY_NODE_SHAPE(enp,scalar_shape(enp->en_decl_prec));
			UPDATE_NODE_SHAPE(enp);
			break;


		/* case T_SUBSCRIPT1: */	/* prelim_node_shape (matlab) */
		case T_CURLY_SUBSCR:			/* prelim_node_shape */
		case T_SQUARE_SUBSCR:
#ifdef CAUTIOUS
			if( enp->en_child[0] == NO_VEXPR_NODE ){
		WARN("CAUTIOUS:  prelim_node_shape:  null object subscripted!?");
				return;
			}
			if( enp->en_child[0]->en_shpp == NO_SHAPE ){
		WARN("CAUTIOUS:  prelim_node_shape:  subscript, no child shape!?");
DUMP_NODE(enp);
				discard_node_shape(enp);
				return;
			}
#endif /* CAUTIOUS */
			/* Now compute the shape of the subscripted object */
			COPY_NODE_SHAPE(enp,uk_shape(enp->en_child[0]->en_shpp->si_prec));
			/* If the subscripted object is unknown,
			 * link these...
			 */
			if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
				LINK_UK_NODES(enp,enp->en_child[0]);
			}

			UPDATE_NODE_SHAPE(enp);
			break;

		case T_SUBVEC:				/* prelim_node_shape */
		case T_CSUBVEC:
#ifdef CAUTIOUS
			if( enp->en_child[0] == NO_VEXPR_NODE ){
		WARN("CAUTIOUS:  prelim_node_shape:  null object subscripted!?");
				return;
			}
			if( enp->en_child[0]->en_shpp == NO_SHAPE ){
		WARN("CAUTIOUS:  prelim_node_shape:  subvec, no child shape!?");
DUMP_NODE(enp);
				discard_node_shape(enp);
				return;
			}
#endif /* CAUTIOUS */
			/* Now compute the shape of the subscripted object */
			/* The child node chould have a known shape (scalar), but an
			 * unknown value!?  that would allow us to compute the shape of
			 * a subscripted object but not a subvector...
			 * We must assign the shape to unkown before calling update node shape.
			 */
			COPY_NODE_SHAPE(enp,uk_shape(enp->en_child[0]->en_shpp->si_prec));
			/* For prelim node shape, we don't evaluate range expressions,
			 * because they may have different values at run-time.
			 * But we need to have a run-time resolution!
			 */
			/*
			if( (! is_variable(enp->en_child[1])) &&
					(!is_variable(enp->en_child[2])) ){
					*/
			if( HAS_CONSTANT_VALUE(enp->en_child[1]) && HAS_CONSTANT_VALUE(enp->en_child[2]) ){
				UPDATE_NODE_SHAPE(enp);
			}
			break;


		case T_SUBSAMP:				/* prelim_node_shape */
		case T_CSUBSAMP:
#ifdef CAUTIOUS
			if( enp->en_child[0] == NO_VEXPR_NODE ){
		WARN("CAUTIOUS:  prelim_node_shape:  null object subscripted!?");
				return;
			}
			if( enp->en_child[0]->en_shpp == NO_SHAPE ){
		WARN("CAUTIOUS:  prelim_node_shape:  subsamp, no child shape!?");
DUMP_NODE(enp);
				discard_node_shape(enp);
				return;
			}
#endif /* CAUTIOUS */
			/* Now compute the shape of the subscripted object */
			/* The child node chould have a known shape (scalar), but an
			 * unknown value!?  that would allow us to compute the shape of
			 * a subscripted object but not a subvector...
			 * We must assign the shape to unkown before calling update node shape.
			 */
			COPY_NODE_SHAPE(enp,uk_shape(enp->en_child[0]->en_shpp->si_prec));
			/*
			if( (! is_variable(enp->en_child[1]))  ){
			*/
			if( HAS_CONSTANT_VALUE( enp->en_child[1] ) ){
				UPDATE_NODE_SHAPE(enp);
			}
			break;



		case T_REAL_PART:
		case T_IMAG_PART:
			tmp_shape = *enp->en_child[0]->en_shpp;
			tmp_shape.si_mach_dim[0]=1;
			tmp_shape.si_n_mach_elts /= 2;
			tmp_shape.si_flags &= ~DT_COMPLEX;
			tmp_shape.si_prec &= ~COMPLEX_PREC_BITS;

			COPY_NODE_SHAPE(enp,&tmp_shape);
			/* BUG verify shpp is complex */
			break;

		case T_TRANSPOSE:
			tmp_shape = *enp->en_child[0]->en_shpp;
			i1                    = tmp_shape.si_rows;
			tmp_shape.si_rows = tmp_shape.si_cols;
			tmp_shape.si_cols = i1;
			SET_SHAPE_FLAGS(&tmp_shape,NO_OBJ);

			COPY_NODE_SHAPE(enp,&tmp_shape);
			CHECK_UK_CHILD(enp,0);

			break;

		case T_DFT:			/* prelim_node_shape */
			if( enp->en_child[0]->en_shpp == NO_SHAPE )
				return;		/* an error */

			/* This is the complex DFT */
			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			CHECK_UK_CHILD(enp,0);
			break;

		case T_RDFT:		/* prelim_node_shape */
			if( enp->en_child[0]->en_shpp == NO_SHAPE )
				return;		/* an error */

			/* BUG should verify that arg is real and power of 2 */

			tmp_shape = *enp->en_child[0]->en_shpp;
			tmp_shape.si_prec |= COMPLEX_PREC_BITS;
			tmp_shape.si_flags |= DT_COMPLEX;
			if( !UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
				tmp_shape.si_cols /= 2;
				tmp_shape.si_cols += 1;
				/* BUG?  set mach_dim in addition to type_dim? */
				tmp_shape.si_mach_dim[0] = 2;
				/* BUG should we recalculate number of elements? */
			}
			COPY_NODE_SHAPE(enp,&tmp_shape);
			CHECK_UK_CHILD(enp,0);


			break;

		case T_RIDFT:		/* prelim_node_shape */
			if( enp->en_child[0]->en_shpp == NO_SHAPE )
				return;		/* an error */

			/* BUG should verify that arg is real and power of 2 */

			tmp_shape = *enp->en_child[0]->en_shpp;
			tmp_shape.si_prec &= ~COMPLEX_PREC_BITS;
			tmp_shape.si_flags &= ~DT_COMPLEX;
			if( !UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
				tmp_shape.si_cols -= 1;
				tmp_shape.si_cols *= 2;
				tmp_shape.si_mach_dim[0] = 1;
				/* BUG set more things?  n_elts?  mach_dim? */
				SET_SHAPE_FLAGS(&tmp_shape,NO_OBJ);
			}
			COPY_NODE_SHAPE(enp,&tmp_shape);
			CHECK_UK_CHILD(enp,0);

			break;

		case T_CONJ:
		/* case T_WRAP: case T_SCROLL: case T_DILATE: case T_ERODE: case T_FILL: */
		ALL_UNARY_CASES				/* prelim_node_shape */

#ifdef CAUTIOUS
			if( insure_child_shape(enp,0) < 0 ){
WARN("CAUTIOUS:  prelim_node_shape:  unary op child has no shape");
				return;		/* an error */
			}
#endif /* CAUTIOUS */

			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			CHECK_UK_CHILD(enp,0);

			break;


		case T_MAXVAL:
		case T_MINVAL:
		case T_SUM:
			COPY_NODE_SHAPE(enp, uk_shape(enp->en_child[0]->en_shpp->si_prec) );

			/*
			POINT_NODE_SHAPE(enp,scalar_shape(enp->en_child[0]->en_shpp->si_prec));
			*/
			/* Before we introduced generalized projection, sum was a scalar,
			 * but now it takes its shape from the LHS!?
			 */
			break;

		case T_INNER:				/* prelim_node_shape */
#ifdef CAUTIOUS
			insure_child_shape(enp,0);
			insure_child_shape(enp,1);
#endif /* CAUTIOUS */

			/* inner product */
			/*
			 * This can be a row dotted with a column, or a matrix
			 * times a column, or a pair of scalars...
			 */

			if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) || UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){
				COPY_NODE_SHAPE(enp, uk_shape(enp->en_child[0]->en_shpp->si_prec) );
				CHECK_UK_CHILD(enp,0);
				CHECK_UK_CHILD(enp,1);
				return;
			}

			/* Use copy_node_shape just to do the mem alloc */
			tmp_shape = *enp->en_child[0]->en_shpp;
			tmp_shape.si_cols=enp->en_child[1]->en_shpp->si_cols;
			SET_SHAPE_FLAGS(&tmp_shape,NO_OBJ);
			COPY_NODE_SHAPE(enp,&tmp_shape);
			/* Shape is known so don't bother with this */
			/*
			CHECK_UK_CHILD(enp,0);
			CHECK_UK_CHILD(enp,1);
			*/
			break;

		case T_STR2_FN:	/* strcmp, and ? */
		case T_STR1_FN:	/* strlen, and ? */
		case T_SIZE_FN:		/* prelim_node_shape */
			/* This returns a number,
			 * so we make it have a scalar shape
			 */
			 POINT_NODE_SHAPE(enp,scalar_shape(PREC_DI));
			 break;

		case T_FILETYPE:	/* could check for valid type here... */
		case T_SAVE:
		case T_INFO:
			/* no-op */
			break;

		case T_LOAD:		/* prelim_node_shape */
			/* We must assign the shape to unkown before calling update node shape.  */
			POINT_NODE_SHAPE(enp,uk_shape(PREC_SP));
			UPDATE_NODE_SHAPE(enp);
			break;

		case T_RETURN:		/* prelim_node_shape */

			/* why do we neet to remember the return nodes? */

			remember_return_node(curr_srp,enp);

			/* This is our chance to figure out the size
			 * of a subroutine...
			 * at least if we are returning an object of known
			 * size...
			 */

			/* Hopeuflly we checked in vectree.y that all returns have
			 * the same type and match the subrt decl
			 */

			if( enp->en_child[0] == NO_VEXPR_NODE ){ /* a void subrt */
				/* leave the shape pointer null */
#ifdef CAUTIOUS
				if( enp->en_shpp != NO_SHAPE ){
					sprintf(error_string,
				"CAUTIOUS:  prelim_node_shape:  %s has a shape!?",node_desc(enp));
					WARN(error_string);
					enp->en_shpp = NO_SHAPE;
				}
#endif /* CAUTIOUS */
				return;

			} else if( IS_CURDLED(enp->en_child[0]) ){
				/* an error expr */
				WARN("return expression is curdled!?");
				discard_node_shape(enp);
				return;
			} else {
				POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
#ifdef CAUTIOUS
if( enp->en_shpp == NO_SHAPE ){
sprintf(error_string,"CAUTIOUS:  prelim_node_shape:  return expression has no shape!?");
WARN(error_string);
return;
}
#endif /* CAUTIOUS */
			}

			/* Here we need to see if there are other
			 * returns, and if so we need to make sure
			 * all the returns match in size...
			 * We also need to communicate this info
			 * to the main subroutine body...
			 */

			srp = curr_srp;

			if( enp->en_shpp != NO_SHAPE && ! UNKNOWN_SHAPE(enp->en_shpp) ){
				if( UNKNOWN_SHAPE(srp->sr_shpp) ){
					srp->sr_shpp = enp->en_shpp;
				} else if( !shapes_match(srp->sr_shpp, enp->en_shpp) ){
					/* does the shape of this return match? */
					NODE_ERROR(enp);
					WARN("mismatched return shapes");
				}
			}
			CHECK_UK_CHILD(enp,0);

			break;		/* end T_RETURN case */

		case T_CALL_NATIVE:			/* prelim_node_shape */
			/* The native functions have shapes that depend
			 * on which function it is, and that requires knowledge
			 * of exactly which function it is!
			 */
			/* prelim_native_shape(enp); */
			(*native_prelim_func)(QSP_ARG  enp);
			break;

		case T_CALLFUNC:			/* prelim_node_shape */

			srp=enp->en_srp;
			srp->sr_call_enp = enp;

			/* We probably need a separate list! */
			/* link any unknown shape args to the callfunc node */
			if( ! NULL_CHILD(enp,0) )
				link_uk_args(QSP_ARG  enp,enp->en_child[0]);

			/* void subrt's never need to have a shape */
			if( srp->sr_prec == PREC_VOID ){
				enp->en_shpp = NO_SHAPE;
				return;
			}

			/* we don't need to remember void subrts... */
			if( curr_srp != NO_SUBRT	/* WHY??? */
				&& enp != curr_srp->sr_body &&
				enp->en_parent->en_code != T_STAT_LIST ){

				/* Why are we doing this?? */
				remember_callfunc_node(curr_srp,enp);
			}


			/* make sure the number of aruments is correct */
			if( arg_count(enp->en_child[0]) != srp->sr_nargs ){
				NODE_ERROR(enp);
				sprintf(error_string,
	"Subrt %s expects %d arguments (%d passed)",srp->sr_name, srp->sr_nargs,
					arg_count(enp->en_child[0]));
				WARN(error_string);
				CURDLE(enp)
				return;
			}



			COPY_NODE_SHAPE(enp,srp->sr_shpp);
			break;

		case T_ROW_LIST:				/* prelim_node_shape */
			/* We used to use EXPR_LIST for explicit initialization,
			 * But we need to have distinct handling of matrices (where
			 * precision and shape should be consistent), and arbitrary lists.
			 */
			/* can arise as print args,
			 * or a child node of T_LIST_OBJ (declaration)
			 * The latter case is why we need shape info...
			 * Also used to provide dimension sets in equivalence().
			 */
#ifdef CAUTIOUS
			if( insure_child_shape(enp,0) < 0 ||
				insure_child_shape(enp,1) ) return;
#endif /* CAUTIOUS */

			/* top node of a list has a literal on the right side,
			 * and the lists descending on the left.
			 * BUG will this code work if the tree is balanced?
			 */

			/* if either child node has unknown shape, then we might as well give up... */
			if( UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){
				COPY_NODE_SHAPE(enp,enp->en_child[1]->en_shpp);
				break;
			}
			if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
				COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
				break;
			}

			/* Here we arbitrarily take the shape from the right-child;
			 * But what if the precisions differ?  This can arise if we
			 * omit the decimal point from float values with no decimal
			 * fraction, they are stored as literal long values...
			 */
			/*
			if( enp->en_child[0]->en_prec != enp->en_child[1]->en_prec )
				PROMOTE_CHILD(enp);
			*/

			tmp_shape = *enp->en_child[1]->en_shpp;

			tmp_shape.si_type_dim[tmp_shape.si_maxdim+1] =
				enp->en_child[0]->en_shpp->si_type_dim[tmp_shape.si_maxdim+1]+1;

			SET_SHAPE_FLAGS(&tmp_shape,NO_OBJ);	/* set_shape_flags sets maxdim?? */
			MULTIPLY_DIMENSIONS(tmp_shape.si_n_type_elts,tmp_shape.si_type_dim)
			/* BUG?  n_mach_elts? */
			COPY_NODE_SHAPE(enp,&tmp_shape);
			break;

		case T_COMP_LIST:				/* prelim_node_shape */
			/* arises as a child node of T_LIST_OBJ (declaration)
			 * The latter case is why we need shape info...
			 */
#ifdef CAUTIOUS
			if( insure_child_shape(enp,0) < 0 ||
				insure_child_shape(enp,1) ) return;
#endif /* CAUTIOUS */

			/* top node of a list has a literal on the right side,
			 * and the lists descending on the left.
			 * BUG will this code work if the tree is balanced?
			 */

			/* if either child node has unknown shape, then we might as well give up... */
			if( UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){
				COPY_NODE_SHAPE(enp,enp->en_child[1]->en_shpp);
				break;
			}
			if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
				COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
				break;
			}

			/* Here we arbitrarily take the shape from the right-child;
			 * But what if the precisions differ?  This can arise if we
			 * omit the decimal point from float values with no decimal
			 * fraction, they are stored as literal long values...
			 */
			/*
			if( enp->en_child[0]->en_prec != enp->en_child[1]->en_prec )
				PROMOTE_CHILD(enp);
			*/

			tmp_shape = *enp->en_child[1]->en_shpp;

			if( tmp_shape.si_mindim == (N_DIMENSIONS-1) ){
				tmp_shape.si_mindim =
				tmp_shape.si_maxdim = 0;
			}

			/* BUG?  type_dim or mach_dim? */
			tmp_shape.si_type_dim[tmp_shape.si_mindim] =
				enp->en_child[0]->en_shpp->si_type_dim[tmp_shape.si_mindim]+1;

			SET_SHAPE_FLAGS(&tmp_shape,NO_OBJ);	/* set_shape_flags sets maxdim?? */
			MULTIPLY_DIMENSIONS(tmp_shape.si_n_type_elts,tmp_shape.si_type_dim)
			/* BUG?  n_mach_elts? */
			COPY_NODE_SHAPE(enp,&tmp_shape);
			break;


		/* Dereference nodes can't be assigned a shape until runtime,
		 * but we can still point to the child shape.
		 * We also set up a link for resolution.
		 */
		case T_DEREFERENCE:	/* prelim_node_shape */
#ifdef CAUTIOUS
			if( enp->en_child[0]->en_shpp == NO_SHAPE ){
				NODE_ERROR(enp);
				sprintf(error_string,"dereference child %s node n%d has no shape!?",
					NNAME(enp->en_child[0]),enp->en_child[0]->en_serial);
				WARN(error_string);
			}
#endif /* CAUTIOUS */
			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
#ifdef DEBUG
if( debug & resolve_debug ){
NODE_ERROR(enp);
sprintf(error_string,"prelim_node_shape calling link_uk_nodes for %s and child %s",
node_desc(enp),node_desc(enp->en_child[0]));
advise(error_string);
}
#endif /* DEBUG */
			LINK_UK_NODES(enp,enp->en_child[0]);
			break;

		case T_STRING:			/* prelim_node_shape */
			COPY_NODE_SHAPE(enp,scalar_shape(PREC_CHAR));
			enp->en_shpp->si_type_dim[1] =
			enp->en_shpp->si_n_type_elts = strlen(enp->en_string)+1;
			SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ);
			break;

		case T_SCALMAX:
		case T_SCALMIN:
			break;

		case T_MAX_INDEX:
		case T_MIN_INDEX:
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_DI));
			break;

		case T_RAMP:			/* prelim_node_shape */
			/* A ramp node must assign the shape of whatever it is
			 * assigned to or combined with.
			 */
			/* BUG we'd like to know the precision! */
			POINT_NODE_SHAPE(enp, uk_shape(PREC_SP) );	/* BUG ramp */

			break;

		case T_UMINUS:
			if( HAS_CONSTANT_VALUE(enp->en_child[0]) )
				enp->en_flags |= NODE_HAS_CONST_VALUE;

			if( enp->en_child[0]->en_shpp == NO_SHAPE )
				return;		/* an error */
			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			CHECK_UK_CHILD(enp,0);

			break;

		case T_RECIP:
			if( HAS_CONSTANT_VALUE(enp->en_child[0]) )
				enp->en_flags |= NODE_HAS_CONST_VALUE;

			if( enp->en_child[0]->en_shpp == NO_SHAPE )
				return;		/* an error */

#ifdef CAUTIOUS
			if( ! SCALAR_SHAPE(enp->en_child[0]->en_shpp) ){
		WARN("CAUTIOUS:  T_RECIP arg should have scalar shape!?");
				return;
			}
#endif /* CAUTIOUS */

			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			CHECK_UK_CHILD(enp,0);

			break;

		case T_BOOL_NOT:				/* prelim_node_shape */
			if( HAS_CONSTANT_VALUE(enp->en_child[0]) )
				enp->en_flags |= NODE_HAS_CONST_VALUE;

			COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			CHECK_UK_CHILD(enp,0);
			break;

		case T_BITCOMP:
		case T_VCOMP:
			if( HAS_CONSTANT_VALUE(enp->en_child[0]) )
				enp->en_flags |= NODE_HAS_CONST_VALUE;

			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			if( enp->en_shpp != NO_SHAPE )
				CHECK_UK_CHILD(enp,0);
			break;

		case T_MATH0_VFN:	/* prelim_node_shape */
		case T_MATH0_FN:
			UPDATE_NODE_SHAPE(enp);
			break;

		case T_MATH1_VFN:	/* prelim_node_shape */
		case T_MATH1_FN:
			if( HAS_CONSTANT_VALUE(enp->en_child[0]) )
				enp->en_flags |= NODE_HAS_CONST_VALUE;

			UPDATE_NODE_SHAPE(enp);
			CHECK_UK_CHILD(enp,0);
			break;

		case T_MATH2_FN:			/* prelim_node_shape */
		case T_MATH2_VFN:
		case T_MATH2_VSFN:
			if( HAS_CONSTANT_VALUE(enp->en_child[0]) &&
				HAS_CONSTANT_VALUE(enp->en_child[1]) )
				enp->en_flags |= NODE_HAS_CONST_VALUE;

/*
sprintf(error_string,"prelim_node_shape %s: begin ",node_desc(enp));
advise(error_string);
DUMP_TREE(enp);
*/
			UPDATE_NODE_SHAPE(enp);
/*
sprintf(error_string,"prelim_node_shape %s: after update_node_shape ",node_desc(enp));
advise(error_string);
DUMP_TREE(enp);
*/
			CHECK_BINOP_LINKS(enp);
/*
sprintf(error_string,"prelim_node_shape %s: done ",node_desc(enp));
advise(error_string);
DUMP_TREE(enp);
*/
			break;



		case T_ASSIGN:			/* prelim_node_shape */

			/* The case for T_ASSIGN is much like that for
			 * the arithmetic operands below...
			 */
			if( IS_CURDLED(enp->en_child[0]) ||
				IS_CURDLED(enp->en_child[1]) ){
				CURDLE(enp)
				return;
			}

			/* check the shapes of the children, try to
			 * determine the shape of this node
			 */

			compute_assign_shape(QSP_ARG  enp);

			/* search the rhs for refs to the lhs object,
			 *
			 * We do this so we can know if we need to use a temporary
			 * object to store results...
			 */

			COUNT_LHS_REFS(enp);

			break;

		case T_REFERENCE:	/* prelim_node_shape */
			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			LINK_UK_NODES(enp,enp->en_child[0]);
			break;


		/* Boolean operators.
		 * This can operate on vectors only in the conditional
		 * assignment statements:
		 * a = v < w ? c : d ;   # selection
		 *
		 * But their node precision is bitmap, regardless of the types of the children...
		 * The children must match, however...
		 */

		BINARY_BOOLOP_CASES					/* prelim_node_shape */
		ALL_NUMERIC_COMPARISON_CASES				/* prelim_node_shape */
			if( HAS_CONSTANT_VALUE(enp->en_child[0]) &&
				HAS_CONSTANT_VALUE(enp->en_child[1]) )
				enp->en_flags |= NODE_HAS_CONST_VALUE;

			if( GET_MATING_SHAPES(enp,0,1) == NO_SHAPE )
{
WARN("prelim_node_shape (boolop):  no mating shapes!?");
DUMP_TREE(enp);
				return;
}
			if( ! SCALAR_SHAPE(enp->en_shpp) )
				set_bool_vecop_code(QSP_ARG  enp);
			CHECK_BINOP_LINKS(enp);
			break;

		/*
		ALL_VECTOR_SCALAR_CASES
		ALL_VECTOR_VECTOR_CASES
		*/
		case T_VV_FUNC:						/* prelim_node_shape */
			/* where do we call copy_node_shape?  check_mating_shapes? */
		case T_VS_FUNC:

	/* It seems that T_VV_FUNC owns its shape, but T_VS_FUNC points ...
	 *
	 * I guess that makes sense in that vs ops always end up with the shape of the
	 * consituent vector, which a vv func could be an outer thingy...
	 */
		ALL_SCALINT_BINOP_CASES					/* prelim_node_shape */
		OTHER_SCALAR_MATHOP_CASES
			if( HAS_CONSTANT_VALUE(enp->en_child[0]) &&
				HAS_CONSTANT_VALUE(enp->en_child[1]) )
				enp->en_flags |= NODE_HAS_CONST_VALUE;
#ifdef FOOBAR
			/* when are these nodes supposed to get their shapes set??? */
#ifdef CAUTIOUS
			if( enp->en_shpp == NO_SHAPE ){
				NODE_ERROR(enp);
				sprintf(error_string,"CAUTIOUS:  prelim_node_shape %s:  null shape ptr!?",node_desc(enp));
				ERROR1(error_string);
			}
#endif /* CAUTIOUS */
#endif /* FOOBAR */
			/* shape pointer is generally uninitialized here??? */

			/* used to be a nop for vv_func in update_node_shape... */
			UPDATE_NODE_SHAPE(enp);
			CHECK_BINOP_LINKS(enp);

			break;

		case T_OBJ_LOOKUP:	/* prelim_node_shape */
			if( ! executing ){
				POINT_NODE_SHAPE(enp,uk_shape(PREC_SP));
				break;
			}

			s=EVAL_STRING(enp->en_child[0]);
			if( s == NULL )
				dp=NO_OBJ;
			else
				dp=DOBJ_OF(s);

#ifdef CAUTIOUS
			if( dp==NO_OBJ ){
				if( s != NULL ){
					sprintf(error_string,
		"CAUTIOUS:  prelim_node_shape:  missing lookup object %s",s);
					WARN(error_string);
					DUMP_TREE(enp);
				}
				return;
			}
#endif /* CAUTIOUS */
			goto gen_obj_shape;

		case T_STATIC_OBJ:			/* prelim_node_shape */
			dp=enp->en_dp;
			goto handle_obj;

		case T_DYN_OBJ:				/* prelim_node_shape */
			/* BUG we should use the identifier?? */
			dp=DOBJ_OF(enp->en_string);

handle_obj:
			if( mode_is_matlab ){
				Identifier *idp;

				if( dp == NO_OBJ ){
					POINT_NODE_SHAPE(enp,uk_shape(PREC_DP));
					return;	/* not an error */
				}

				idp = ID_OF(enp->en_string);
				if( idp == NO_IDENTIFIER ){
					POINT_NODE_SHAPE(enp,uk_shape(PREC_DP));
					return;	/* not an error */
				}

				POINT_NODE_SHAPE(enp,&idp->id_shape);
				return;
			}

#ifdef CAUTIOUS
			if( dp==NO_OBJ ){
				NODE_ERROR(enp);
				sprintf(error_string,
		"CAUTIOUS:  prelim_node_shape:  missing object %s",enp->en_string);
				WARN(error_string);
				DUMP_TREE(enp);
				return;
			}
#endif /* CAUTIOUS */

gen_obj_shape:
			/* copy the size of this object to the node */

			/* If the object's shape is unknown, shouldn't
			 * we add it to the list here???
			 */
			decl_enp = (Vec_Expr_Node *)dp->dt_extra;

/*
sprintf(error_string,"prelim_node_shape OBJECT:  decl_enp = 0x%lx",
(u_long)decl_enp);
advise(error_string);
*/
			/* We used to think it was an error if there was no decl_enp,
			 * and it would be if we were operating strictly within
			 * the expression language - but since we are initially
			 * mixing cmds in the old script environment with vector
			 * expressions, we may want to refer to objects created
			 * outside of the expression parser environment...
			 * But these should all be fixed size objects,
			 * so this should be safe???
			 *
			 * And then there's matlab, where there are no declarations,
			 * and the sizes are a bit more malleable...
			 */
			if( decl_enp == NO_VEXPR_NODE ){
				if( (dp->dt_flags&DT_EXPORTED) == 0 && ! mode_is_matlab ){
sprintf(error_string,"prelim_node_shape OBJECT %s (%s):  no decl node, and object was not exported!?",
enp->en_string,node_desc(enp));
WARN(error_string);
				}
				POINT_NODE_SHAPE(enp,&dp->dt_shape);
			} else {
/*
sprintf(error_string,"prelim_node_shape %s:  pointing to %s",node_desc(enp),node_desc(decl_enp));
advise(error_string);
describe_shape(decl_enp->en_shpp);
*/
#ifdef CAUTIOUS
				if( decl_enp->en_shpp == NO_SHAPE ){
					NODE_ERROR(enp);
					sprintf(error_string,
		"prelim_node_shape %s:  declaration node %s has no shape!?",node_desc(enp),
						node_desc(decl_enp));
					ERROR1(error_string);
				}
#endif /* CAUTIOUS */
				POINT_NODE_SHAPE(enp,decl_enp->en_shpp);
			}
			/* we link all references to the declaration node, that way
			 * when we resolve the object we can get to all the other occurrences.
			 */
			if( UNKNOWN_SHAPE(enp->en_shpp) )
				note_uk_objref(QSP_ARG  decl_enp,enp);

			break;

		case T_POINTER:		/* prelim_node_shape */
			{
			Identifier *idp;
#ifdef CAUTIOUS
			idp = get_named_ptr(QSP_ARG  enp);
			if( idp == NO_IDENTIFIER ) break;

			if( idp->id_ptrp->ptr_decl_enp == NO_VEXPR_NODE ){
				sprintf(error_string,
					"CAUTIOUS:  prelim_node_shape:  pointer %s has no decl node!?",idp->id_name);
				WARN(error_string);
			}
			if( idp->id_ptrp->ptr_decl_enp->en_shpp == NO_SHAPE ){
				sprintf(error_string,
					"CAUTIOUS:  prelim_node_shape:  pointer decl node has no shape!?");
				WARN(error_string);
			}
#else
			idp = ID_OF(enp->en_string);
#endif /* CAUTIOUS */
			POINT_NODE_SHAPE(enp,idp->id_ptrp->ptr_decl_enp->en_shpp);
			break;
			}

		case T_END:			/* prelim_node_shape (matlab) */
		case T_LIT_INT:
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_DI));
			enp->en_flags |= NODE_HAS_CONST_VALUE;
			break;
		case T_LIT_DBL:			/* prelim_node_shape */
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_DP));
			enp->en_flags |= NODE_HAS_CONST_VALUE;
			break;
		case T_FILE_EXISTS:
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_IN));
			break;

		case T_FIX_SIZE:		/* prelim_node_shape */
			LINK_UK_NODES(enp,enp->en_child[0]);
			POINT_NODE_SHAPE(enp,uk_shape(enp->en_child[0]->en_prec));
			break;

		/* with badname, we really want to do nothing!? */
		case T_BADNAME:		/* prelim_node_shape */
		case T_INDIR_CALL:	/* can't determine the shape until runtime */
			POINT_NODE_SHAPE(enp,uk_shape(PREC_SP));
			break;

		case T_BOOL_PTREQ:	/* prelim_node_shape */
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_DI));
			break;

		case T_FUNCREF:
			POINT_NODE_SHAPE(enp,enp->en_srp->sr_shpp);
			break;

		case T_SET_FUNCPTR:		/* prelim_node_shape */
		case T_FUNCPTR:			/* prelim_node_shape */
		/* These get a shape when they are assigned */
			/* Do we know the return precision of the funcptr? */
			break;

		ALL_INCDEC_CASES
			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			break;

		/* matlab */
		case T_FIRST_INDEX:
		case T_LAST_INDEX:
		case T_STRUCT:
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_DI));
			break;
		/* end matlab */

		case T_RANGE2:		/* prelim_node_shape */
			if( HAS_CONSTANT_VALUE(enp) ){
				COPY_NODE_SHAPE(enp,scalar_shape(PREC_DI));
				n1=EVAL_INT_EXP(enp->en_child[0]);
				n2=EVAL_INT_EXP(enp->en_child[1]);
				enp->en_shpp->si_cols = floor( n2 - n1 );
				SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ);
			} else {
				COPY_NODE_SHAPE(enp,uk_shape(PREC_DI));
			}
			break;

		case T_RANGE:
			COPY_NODE_SHAPE(enp,scalar_shape(PREC_DI));
			/* we need to figure out how many elements? */
			n1=EVAL_INT_EXP(enp->en_child[0]);
			n2=EVAL_INT_EXP(enp->en_child[1]);
			n3=EVAL_INT_EXP(enp->en_child[2]);
			/* BUG? should we allow float start and stop? */
			/* enp->en_shpp->si_cols = floor( (n3-n1)/n2 ); */
			/* NOT columns, could be any dimension... */
			SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ);
			break;

		case T_STRING_LIST:	/* prelim_node_shape */
			{
			COPY_NODE_SHAPE(enp,scalar_shape(PREC_CHAR));
			enp->en_shpp->si_cols = enp->en_child[0]->en_shpp->si_cols
						+ enp->en_child[1]->en_shpp->si_cols
						- 1;
			SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ);
			break;
			}

		case T_RET_LIST:		/* prelim_node_shape */
			LINK_UK_NODES(enp,enp->en_child[0]);
			LINK_UK_NODES(enp,enp->en_child[1]);
			/* when we have a list of matrices, who knows what we'll do...
			 * but for now, we assume it's a row of scalars!
			 */
			COPY_NODE_SHAPE(enp,uk_shape(PREC_NONE));
			break;

		/* These are a bunch of codes we don't need to scan,
		 * they should probably be a default case once we
		 * know all the cases we have to process!
		 *
		 * These nodes all have no shape.
		 */
		case T_STAT_LIST:
		case T_PRINT_LIST:		/* prelim node_shape */
		case T_STR_PTR:
		case T_SET_STR:			/* prelim_node_shape */
		case T_DECL_STAT_LIST:
		case T_DECL_ITEM_LIST:
		case T_MIXED_LIST:
		case T_PERFORM:
		case T_SCRIPT:
		case T_SUBRT:
		case T_STRCPY:
		case T_STRCAT:
		case T_OUTPUT_FILE:
		case T_SET_PTR:
		case T_DISPLAY:
		case T_NAME_FUNC:
		case T_PROTO:
		case T_EXIT:
		ALL_DECL_STAT_CASES
		//case T_EXTERN_DECL:
		//case T_DECL_STAT:
		case T_CLR_OPT_PARAMS:
		case T_ADD_OPT_PARAM:
		case T_OPTIMIZE:
		case T_SWITCH_LIST:		/* prelim_node_shape */
		case T_CASE_LIST:		/* prelim_node_shape */
		case T_ARGLIST:
		/*
		case T_ADVISE:
		case T_WARN:
		case T_EXP_PRINT:
		*/
		ALL_PRINT_CASES
		/*
		ALL_LIST_NODE_CASES
		*/
		ALL_CTL_FLOW_CASES

		/* matlab */
		case T_MFILE:
		case T_CLEAR:
		case T_DRAWNOW:
		case T_CLF:
		case T_MCMD:
		case T_SUBMTRX:
		case T_INDEX_SPEC:
		case T_MLFUNC:
		case T_INDEX_LIST:
		case T_OBJ_LIST:
		case T_GLOBAL:
		case T_FIND:
		case T_ENTIRE_RANGE:

		case T_EXPR_LIST:

			/* DECL_STAT's shouldn't have a shape, because they can have
			 * multiple object delcarations
			 */

#ifdef CAUTIOUS
			verify_null_shape(QSP_ARG  enp);
#endif /* CAUTIOUS */
			break;

		NON_INIT_DECL_ITEM_CASES
			COPY_NODE_SHAPE(enp,uk_shape(enp->en_decl_prec));
			break;

		case T_DECL_INIT:			/* prelim_node_shape */

			/* Do we need shapes on declaration nodes???
			 * For globals we don't, but for subrt args we do!? (WHY???)
			 *
			 * Also, we need them for unknown objects, because that is where we
			 * store the resolution information...
			 *
			 * And now we learn we have to resolve:  float f[]=a;
			 *
			 * There may be a difference if it is immediate execution...
			 */

			if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
				COPY_NODE_SHAPE(enp,enp->en_child[1]->en_shpp);
				if( ! UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){
					COPY_NODE_SHAPE(enp->en_child[0],enp->en_child[1]->en_shpp);
				} else {
					LINK_UK_NODES(enp,enp->en_child[1]);
					LINK_UK_NODES(enp->en_child[0],enp->en_child[1]);
				}
			}


			break;


		default:
			MISSING_CASE(enp,"prelim_node_shape");
			break;
	}
} /* end prelim_node_shape */

#ifdef CAUTIOUS
void verify_null_shape(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	if( enp->en_shpp != NO_SHAPE ){
		NODE_ERROR(enp);
		DUMP_TREE(enp);
		sprintf(error_string,"CAUTIOUS:  prelim_node_shape:  %s has a non-null shape ptr!?",node_desc(enp));
		ERROR1(error_string);
	}
}
#endif /* CAUTIOUS */

static int get_subvec_indices(QSP_ARG_DECL Vec_Expr_Node *enp, dimension_t *i1p, dimension_t *i2p )
{
	int which_dim=0;	/* auto-init just to silence compiler */

	if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ) return(0);

	/* the subvector node will have the same shape
	 * (an image is still an image) but will have
	 * different dimensions, determined by the value
	 * of the children nodes...
	 */

	/* These may not be determinable until run-time,
	 * if the expressions contain variables!?
	 * How do we know when to evaluate???
	 *
	 * We ought to have a flag on the node that says whether the value is constant or variable...
	 * CONST
	 */

	if( enp->en_child[1] == NO_VEXPR_NODE ){
		*i1p=0;
	} else {
		if( (!HAS_CONSTANT_VALUE(enp->en_child[1])) ){
			if( !executing ){
			/* only evaluate variable expressions at run-time */
				return(0);
			} else {
				enp->en_flags |= NODE_NEEDS_CALLTIME_RES;
			}
		}
		*i1p = EVAL_INT_EXP(enp->en_child[1]);
	}

	if( enp->en_code==T_SUBVEC ){
		which_dim = enp->en_child[0]->en_shpp->si_maxdim ;
	} else if( enp->en_code==T_CSUBVEC ){
		which_dim = enp->en_child[0]->en_shpp->si_mindim ;
	}
#ifdef CAUTIOUS
	  else {
	  	ERROR1("CAUTIOUS:  get_subvec_indices:  unexpected node code");
	}
#endif /* CAUTIOUS */
											;
	if( enp->en_child[2] == NO_VEXPR_NODE ){
		*i2p=enp->en_child[0]->en_shpp->si_type_dim[which_dim] -1;
	} else {
		if( (!HAS_CONSTANT_VALUE(enp->en_child[2])) ){
			if( !executing ){
				/* only evaluate variable expressions at run-time */
				return(0);
			} else {
				enp->en_flags |= NODE_NEEDS_CALLTIME_RES;
			}
		}
		*i2p = EVAL_INT_EXP(enp->en_child[2]);
	}

	if( *i1p > *i2p ){
		NODE_ERROR(enp);
		sprintf(error_string,"first range index (%d) should be less than or equal to second range index (%d)",
		*i1p,*i2p);
		WARN(error_string);
		return(0);
	}

	return(1);
}

/* Set the shape of this node based on its children.
 */

static void update_node_shape(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Shape_Info tmp_shape;
	Subrt *srp;
	dimension_t i1,i2,len ;
	Image_File *ifp;
	const char *str;

	/* now all the child nodes have been scanned, process this one */

	if( IS_CURDLED(enp) ) return;

	switch(enp->en_code){
		case T_RANGE:
			/* do ranges even need to have a shape? */
			break;

		case T_EQUIVALENCE:			/* update_node_shape */
			/* The grammar doesn't specify the length of the list,
			 * but there are 5 dimensions.  We will allow fewer than 5
			 * to be specified (how often does one have multiple sequences?)
			 * those specified are mapped to the lowest dimensions, the
			 * higher ones default to 1.
			 */
			{
			Vec_Expr_Node *lenp;
			int i_dim;
			dimension_t d;

			lenp = enp->en_child[1];	/* T_EXPR_LIST */
			/* expression lists have the final expression on the right child,
			 * and the preceding list on the left child.
			 */
			i_dim = 0;
			do {
				d = EVAL_INT_EXP(lenp->en_child[1]);
				if( d == 0 ){
					/* not an error if it's not run-time yet */
					COPY_NODE_SHAPE(enp,uk_shape(enp->en_decl_prec));
					return;
				}
				enp->en_shpp->si_type_dim[i_dim] = d;
				i_dim++;
				lenp = lenp->en_child[0];
			} while( i_dim < N_DIMENSIONS && lenp->en_code == T_EXPR_LIST );

			if( i_dim >= N_DIMENSIONS ){
				NODE_ERROR(enp);
				WARN("too many dimension specifiers");
				CURDLE(enp)
				return;
			}

			d = EVAL_INT_EXP( lenp );
			if( d == 0 ){
				/* not an error if it's not run-time yet */
				COPY_NODE_SHAPE(enp,uk_shape(enp->en_decl_prec));
				return;
			}
			enp->en_shpp->si_type_dim[i_dim] = d;
			enp->en_shpp->si_flags &= ~DT_UNKNOWN_SHAPE;	/* if we are here, all dimensions are non-zero */
			SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ);
			}

			break;

		case T_SUBSAMP:				/* update_node_shape */
			if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ) break;

			/* the subsample node will have the same shape
			 * (an image is still an image) but will have
			 * different dimensions, determined by the value
			 * of the children nodes...
			 */

			COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);

			{
			Vec_Expr_Node *range_enp;
			incr_t inc,l;

			/* second child should be a range specification, it's shape is long scalar */
			/* We need to evaluate it to know what is going on here */
			range_enp=enp->en_child[1];
			/* do a cautious check here for correct node type */
			i1 = EVAL_INT_EXP(range_enp->en_child[0]);
			i2 = EVAL_INT_EXP(range_enp->en_child[1]);
			inc = EVAL_INT_EXP(range_enp->en_child[2]);
			if( i2 > i1 ) {
				l = 1+(i2-i1+1)/inc;
			} else {
				l = 1+(i1-i2+1)/(-inc);
			}
			if( l < 0 )
				WARN("apparent length of subsample object is negative!?");

			enp->en_shpp->si_n_type_elts /=
				enp->en_shpp->si_type_dim[enp->en_shpp->si_maxdim];
			enp->en_shpp->si_n_mach_elts /=
				enp->en_shpp->si_mach_dim[enp->en_shpp->si_maxdim];
			enp->en_shpp->si_n_type_elts *= l;
			enp->en_shpp->si_n_mach_elts *= l;
			enp->en_shpp->si_type_dim[enp->en_shpp->si_maxdim]=l;
			enp->en_shpp->si_mach_dim[enp->en_shpp->si_maxdim]=l;
			enp->en_shpp->si_maxdim --;
			}
				
			break;

		case T_RANGE2:
			i1 = EVAL_INT_EXP(enp->en_child[0]);
			i2 = EVAL_INT_EXP(enp->en_child[1]);
			/* see T_ROW below... */
			COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			enp->en_shpp->si_cols = 1 + labs(i2 - i1) ;
			SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ);
			break;
		/* matlab! */
		case T_RET_LIST:		/* prelim_node_shape */
			break;

		case T_CALL_NATIVE:
			/* update_native_shape(enp); */
			(*native_update_func)(enp);
			break;

		case T_SUBSCRIPT1:	/* update_node_shape */
			ERROR1("update_node_shape T_SUBSCRIPT1:  should have been compiled to another code!?");
			break;

		case T_ROW:						/* update_node_shape */
			/* this is really a list node... */
			COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			enp->en_shpp->si_cols ++;
			SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ);
			break;

		/* end matlab */

		case T_FIX_SIZE:
			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			break;

		/* Do-nothing cases */
		case T_REFERENCE: /* reference node should always point to child node */	/* update_node_shape */
		case T_DEREFERENCE: /* dereference node should always point to child node */
			break;

		case T_SS_B_CONDASS:					/* update_node_shape */
			/* child 0 is a bitmap...
			 * the other children are scalars
			 */

			if( ! UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
				COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
				xform_from_bitmap(enp->en_shpp,enp->en_child[1]->en_prec);
			}
			break;

		ALL_BOOLOP_CASES			/* update_node_shape */
		ALL_NUMERIC_COMPARISON_CASES		/* update_node_shape */

			/* should be pointed to child node by check_mating_shapes */

			break;

		case T_ENLARGE:		/* update_node_shape */
		case T_REDUCE:		/* update_node_shape */
			break;

		case T_LIST_OBJ:		/* update_node_shape */
			/* all of the elements of the list
			 * should have the same shape...
			 * BUG?  should we verify?
			 */
			COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			break;

		case T_TYPECAST:		/* update_node_shape */
			tmp_shape = *enp->en_child[0]->en_shpp;
			tmp_shape.si_prec = enp->en_cast_prec;
			COPY_NODE_SHAPE(enp,&tmp_shape);
			break;

		case T_SQUARE_SUBSCR:		/* update_node_shape */
			if( !SCALAR_SHAPE(enp->en_child[1]->en_shpp) ){
				/* The subscript is a vector, use its shape.
				 * Note that this is going to have problems with
				 * multiple vector-valued subscripts...
				 */
				COPY_NODE_SHAPE(enp,enp->en_child[1]->en_shpp);
				enp->en_prec = enp->en_child[0]->en_prec;

				/* The subscript can have a non-1 type dimension
				 * if it needs to address multiple indexing dimensions.
				 * But we don't want to raise the type dimension
				 * of the target.
				 */
				enp->en_shpp->si_type_dim[0] = enp->en_child[0]->en_shpp->si_type_dim[0];
				SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ);	/* DO we need this? */
				break;
			}
			if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ) break;

			if( enp->en_child[0]->en_shpp->si_type_dim[enp->en_child[0]->en_shpp->si_maxdim] == 0 ){
				sprintf(error_string,
			"update_node_shape:  %s node n%d has dimension %d zero, can't subscript",
					NNAME(enp->en_child[0]),enp->en_child[0]->en_serial,enp->en_child[0]->en_shpp->si_maxdim);
				WARN(error_string);
				break;
			}
			tmp_shape = *enp->en_child[0]->en_shpp;

			COPY_NODE_SHAPE(enp,&tmp_shape);
			REDUCE_MAXDIM(enp->en_shpp)
			SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ);
			/* BUG do range checking here */
			break;

		case T_CURLY_SUBSCR:		/* update_node_shape */
			/* We normally get the shape from the object
			 * being subscripted...
			 */
			if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) )
				break;

			if( enp->en_child[0]->en_shpp->si_type_dim[enp->en_child[0]->en_shpp->si_mindim] == 0 ){
				sprintf(error_string,
			"update_node_shape:  %s node n%d has dimension %d zero, can't subscript",
					NNAME(enp->en_child[0]),enp->en_child[0]->en_serial,enp->en_child[0]->en_shpp->si_mindim);
				WARN(error_string);
				break;
			}
			tmp_shape = *enp->en_child[0]->en_shpp;
			COPY_NODE_SHAPE(enp,&tmp_shape);
			REDUCE_MINDIM(enp->en_shpp)
			SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ);
			/* BUG do range checking here */
			break;

		case T_REAL_PART:		/* update_node_shape */
		case T_IMAG_PART:		/* update_node_shape */
			tmp_shape = *enp->en_child[0]->en_shpp;
			tmp_shape.si_mach_dim[0]=1;
			tmp_shape.si_n_mach_elts/=2;
			tmp_shape.si_flags &= ~DT_COMPLEX;
			tmp_shape.si_prec &= ~COMPLEX_PREC_BITS;

			COPY_NODE_SHAPE(enp,&tmp_shape);
			/* BUG verify enp->en_child[0]->en_shpp is complex */
			break;

		case T_TRANSPOSE:		/* update_node_shape */
			enp->en_child[0]->en_shpp = get_child_shape(QSP_ARG  enp,0);
			tmp_shape = *enp->en_child[0]->en_shpp;
			i1                    = tmp_shape.si_rows;
			tmp_shape.si_rows = tmp_shape.si_cols;
			tmp_shape.si_cols = i1;
			SET_SHAPE_FLAGS(&tmp_shape,NO_OBJ);

			COPY_NODE_SHAPE(enp,&tmp_shape);

			break;

		case T_SUBVEC:		/* update_node_shape */
			if( !get_subvec_indices(QSP_ARG enp,&i1,&i2) ) break;

			len = i2+1-i1;

			/* Now, we need a good way to determine which dimension
			 * is being indexed!?
			 */

			COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);

			/* BUG there is code missing here !? */
			/* what code? */
			UPDATE_DIM_LEN(enp->en_shpp,si_maxdim,len,--)

			/* does set_shape_flags set mindim/maxdim??? */
			/* set_shape_flags(enp->en_shpp,NO_OBJ); */

			break;

		case T_CSUBVEC:		/* update_node_shape */
			if( !get_subvec_indices(QSP_ARG enp,&i1,&i2) ) break;

			len = i2+1-i1;

			COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);

			/* Now, we need a good way to determine which dimension
			 * is being indexed!?
			 */
			/* BUG there is code missing here !? */
			UPDATE_DIM_LEN(enp->en_shpp,si_mindim,len,++)

			/* SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ); */

			break;

		case T_RDFT:		/* update_node_shape */
			if( enp->en_child[0]->en_shpp == NO_SHAPE )
				return;		/* an error */

			/* BUG should verify that arg is real and power of 2 */

			tmp_shape = *enp->en_child[0]->en_shpp;
			if( !UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
				tmp_shape.si_cols /= 2;
				tmp_shape.si_cols += 1;
				tmp_shape.si_mach_dim[0] = 2;
				/* BUG?  recalc other things? */
				tmp_shape.si_prec |= COMPLEX_PREC_BITS;
				tmp_shape.si_flags |= DT_COMPLEX;
			}
			COPY_NODE_SHAPE(enp,&tmp_shape);


			break;

		case T_RIDFT:		/* update_node_shape */
			if( enp->en_child[0]->en_shpp == NO_SHAPE )
				return;		/* an error */

			/* BUG should verify that arg is real and power of 2 */


			tmp_shape = *enp->en_child[0]->en_shpp;
			if( !UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
				tmp_shape.si_cols -= 1;
				tmp_shape.si_cols *= 2;
				tmp_shape.si_mach_dim[0] = 1;
				/* BUG?  set other things? */
				SET_SHAPE_FLAGS(&tmp_shape,NO_OBJ);
			}
			COPY_NODE_SHAPE(enp,&tmp_shape);

			break;

		case T_WRAP:		/* update_node_shape */
		case T_SCROLL:		/* update_node_shape */
			if( enp->en_child[0]->en_shpp == NO_SHAPE )
				return;		/* an error */

			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);

			break;


		case T_SUM:		/* update_node_shape */
			/* takes shape from LHS */
			break;

		case T_SIZE_FN:		/* update_node_shape */
			 break;

		case T_FILETYPE:	/* could check for valid type here... */
		case T_SAVE:		/* update_node_shape */
		case T_INFO:		/* update_node_shape */
			/* no-op */
			break;

		case T_LOAD:		/* update node shape */
			if( ! executing ) goto no_file;

			str=EVAL_STRING(enp->en_child[0]);
			if( str == NULL ) goto no_file;	/* probably a variable that we can't know until runtime */
			ifp = img_file_of(QSP_ARG  str);

			if( ifp == NO_IMAGE_FILE ){
				ifp = read_image_file(QSP_ARG  str);
				if( ifp==NO_IMAGE_FILE ){
					NODE_ERROR(enp);
					sprintf(error_string,
	"update_node_shape READ/LOAD:  Couldn't open image file %s",str);
					WARN(error_string);
					goto no_file;
				}
			}

			if( ! IS_READABLE(ifp) ){
				sprintf(error_string,
		"File %s is not readable!?",str);
				WARN(error_string);
				goto no_file;
			}
#ifdef CAUTIOUS
			if( ifp->if_dp == NO_OBJ ){
				sprintf(error_string,
	"CAUTIOUS:  file %s has no data object in header!?",str);
				WARN(error_string);
				goto no_file;
			}
#endif /* CAUTIOUS */

			/* BUG - the file shape is not necessarily
			 * the same as what the ultimate destination
			 * object will be - we will want to be able
			 * to read single frames from a file containing
			 * a sequence of frames.
			 */

			COPY_NODE_SHAPE(enp,&ifp->if_dp->dt_shape);
			break;
no_file:
			/* BUG we don't know the prec if we have no file!? */
			POINT_NODE_SHAPE(enp,uk_shape(PREC_SP));


			break;


		case T_RETURN:		/* update_node_shape */

			/* This is our chance to figure out the size
			 * of a subroutine...
			 * at least if we are returning an object of known
			 * size...
			 */

			if( enp->en_child[0] == NO_VEXPR_NODE ){ /* a void subrt */

				/* leave the shape pointer null */
				/* BUG? do we need to do this? */

				POINT_NODE_SHAPE(enp,void_shape(PREC_VOID));
				return;
			} else if( IS_CURDLED(enp->en_child[0]) ){
				/* an error expr */
				WARN("return expression is curdled!?");
				discard_node_shape(enp);
				return;
			} else {
				POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
#ifdef CAUTIOUS
if( enp->en_shpp == NO_SHAPE ){
sprintf(error_string,"CAUTIOUS:  update_node_shape:  return expression has no shape!?");
WARN(error_string);
return;
}
#endif /* CAUTIOUS */
			}

			/* Here we need to see if there are other
			 * returns, and if so we need to make sure
			 * all the returns match in size...
			 * We also need to communicate this info
			 * to the main subroutine body...
			 */

			srp = curr_srp;

			if( UNKNOWN_SHAPE(srp->sr_shpp) &&
				! UNKNOWN_SHAPE(enp->en_shpp) ){

				srp->sr_shpp = enp->en_shpp;

			} else if( ! UNKNOWN_SHAPE(enp->en_shpp) ){
				/* does the shape of this return match? */
				if( !shapes_match(srp->sr_shpp, enp->en_shpp) ){


					NODE_ERROR(enp);
					WARN("mismatched return shapes");
				}
			}

			break;

		case T_CALLFUNC:			/* update_node_shape */

			srp=enp->en_srp;
			srp->sr_call_enp = enp;

			/* Why do we do this here??? */

			/* make sure the number of aruments is correct */
			if( arg_count(enp->en_child[0]) != srp->sr_nargs ){
				NODE_ERROR(enp);
				sprintf(error_string,
	"Subrt %s expects %d arguments (%d passed)",srp->sr_name, srp->sr_nargs,
					arg_count(enp->en_child[0]));
				WARN(error_string);
				CURDLE(enp)
				return;
			}

			if( srp->sr_shpp != NO_SHAPE ){
				COPY_NODE_SHAPE(enp,srp->sr_shpp);
			}
			break;

		case T_ARGLIST:		/* update_node_shape */
			break;


		case T_ROW_LIST:		/* update_node_shape */

			/* can arise as print args,
			 * or a child node of T_LIST_OBJ (declaration)
			 */

			/*
			if( GET_MATING_SHAPES(enp,0,1) == NO_SHAPE )
				return;
			*/

			if(enp->en_child[0]->en_shpp==NO_SHAPE || enp->en_child[1]->en_shpp==NO_SHAPE)
				return;


			tmp_shape = *enp->en_child[1]->en_shpp;

			tmp_shape.si_type_dim[tmp_shape.si_maxdim+1] =
				enp->en_child[0]->en_shpp->si_type_dim[tmp_shape.si_maxdim+1]+1;
			SET_SHAPE_FLAGS(&tmp_shape,NO_OBJ);

			COPY_NODE_SHAPE(enp,&tmp_shape);

			break;

//		/* old matlab case */
//		case T_ROWLIST:						/* update_node_shape */
//			COPY_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
//			enp->en_shpp->si_rows ++;
//			SET_SHAPE_FLAGS(enp->en_shpp,NO_OBJ);
//			break;


		case T_RAMP:		/* update_node_shape */
			break;

		case T_RECIP:		/* update_node_shape */
		case T_UMINUS:		/* update_node_shape */
			if( enp->en_child[0]->en_shpp == NO_SHAPE ){
				CURDLE(enp)
				return;		/* an error */
			}
			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			break;

		case T_MATH0_VFN:				/* update_node_shape */
			/* BUG?  prec SP or DP?? */
			POINT_NODE_SHAPE(enp,uk_shape(PREC_DP));
			break;

		case T_MATH2_VFN:				/* update_node_shape */
		case T_MATH1_VFN:				/* update_node_shape */
			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			break;

		ALL_SCALINT_BINOP_CASES
#ifdef CAUTIOUS
			if( enp->en_child[0]->en_shpp == NO_SHAPE ){
				NODE_ERROR(enp);
		ERROR1("CAUTIOUS: update_node_shape:  scalint binop node left child has no shape!?");
			}
#endif /* CAUTIOUS */
			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
			break;

		case T_MATH1_FN:					/* update_node_shape */
#ifdef CAUTIOUS
			if( enp->en_child[0]->en_shpp == NO_SHAPE ){
				NODE_ERROR(enp);
				ERROR1("CAUTIOUS:  math fn arg node has no shape!?");
			}
#endif /* CAUTIOUS */

			POINT_NODE_SHAPE(enp,scalar_shape(PREC_SP));

			break;

		case T_MATH0_FN:		/* update_node_shape */
		case T_MATH2_FN:		/* update_node_shape */
			/* always has a scalar shape */
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_SP));
			break;

		case T_MATH2_VSFN:		/* update_node_shape */
#ifdef CAUTIOUS
			if( enp->en_child[0] == NO_VEXPR_NODE ||
				enp->en_child[1] == NO_VEXPR_NODE ){
				WARN("CAUTIOUS:  update_node_shape MATH2_VSFN:  missing child");
				/* parse error - should zap this node? */
				return;
			}
#endif /* CAUTIOUS */
			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp); /* BUG DP? math2_fn */
			return;

		/* These are a bunch of codes we don't need to scan,
		 * they should probably be a default case once we
		 * know all the cases we have to process!
		 */
		case T_STAT_LIST:		/* update_node_shape */
			break;

		case T_ASSIGN:			/* update_node_shape */
			update_assign_shape(QSP_ARG  enp);
			break;


		/*
		ALL_BINOP_CASES
		*/
		case T_VV_FUNC:		/* update_node_shape */
		case T_VS_FUNC:
			/* Used to do nothing here, but need to update after changing T_MAXVAL
			 * to T_VV_FUNC...
			 */

#ifdef FOOBAR
#ifdef CAUTIOUS
			if( enp->en_shpp == NO_SHAPE ){
				NODE_ERROR(enp);
				sprintf(error_string,"CAUTIOUS:  update_node_shape %s:  null shape ptr!?",node_desc(enp));
				ERROR1(error_string);
			}
#endif /* CAUTIOUS */
#endif /* FOOBAR */

			/* this is redundant most of the time... */
			if( GET_MATING_SHAPES(enp,0,1) == NO_SHAPE )
{
WARN("update_node_shape (vv/vs func):  no mating shapes!?");
DUMP_TREE(enp);
				return;
}

			break;

		case T_POINTER:
		case T_STR_PTR:
		case T_DYN_OBJ:		/* update_node_shape */
			/* done once, don't need to update */
			/* but - doesn't seem to work right for matlab? */
			if( mode_is_matlab ){
				/* matlab doesn't use ptr's, this must be object */
				Data_Obj *dp;

				/* BUG? what if the object gets moved?
				 * possible dangling ptr BUG
				 */
				dp = DOBJ_OF(enp->en_string);
				if( dp == NO_OBJ ){
/*
sprintf(error_string,"update_node_shape T_DYN_OBJ (matlab):  no object %s!?",enp->en_string);
advise(error_string);
*/
					break;
				}
				if( ! UNKNOWN_SHAPE(&dp->dt_shape) ){
					POINT_NODE_SHAPE(enp,&dp->dt_shape);
				}
else
advise("update_node_shape T_DYN_OBJ (matlab):  object has unknown shape!?");
			} else {
//advise("update_node_shape OBJECT, doing nothing");
			}
			break;

		case T_LIT_INT:		/* update_node_shape */
		case T_LIT_DBL:		/* update_node_shape */
		case T_PLUS:
		case T_MINUS:
		case T_TIMES:
		case T_DIVIDE:
		case T_INNER:		/* should be updated??? */
			/* done once, don't need to update */
			break;

		/* these are do-nothing cases, but we put them here
		 * for completeness...
		 */
		ALL_DECL_STAT_CASES
		//case T_DECL_STAT_LIST:		/* update_node_shape */
		case T_STRING_LIST:		/* update_node_shape */
		case T_STRING:		/* update_node_shape */
		/*
		case T_ADVISE:
		case T_WARN:
		case T_EXP_PRINT:
		*/
		case T_DECL_ITEM_LIST:		/* update_node_shape */
		case T_PERFORM:		/* update_node_shape */
		case T_SCRIPT:		/* update_node_shape */
		ALL_PRINT_CASES

		/* the declaration statements didn't appear here before
		 * we put in the extra runtime checking...
		 * In principle, we might have some useful work to
		 * do here, in the case where we have variables
		 * serving as dimensions, e.g.  int arr[x];
		 * where x is a variable...
		 * But for now, we will do nothing and see what happens.
		 */
		case T_SCAL_DECL:
		case T_IMG_DECL:

			break;

		case T_INDIR_CALL:
		case T_FUNCPTR:
			/* do nothing */
			break;

		default:
			MISSING_CASE(enp,"update_node_shape");
			break;
	}
} /* end update_node_shape */

void update_tree_shape(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int i;

	if( IS_CURDLED(enp) ) return;

	for(i=0;i<MAX_CHILDREN(enp);i++)
		if( enp->en_child[i] != NO_VEXPR_NODE ){
			UPDATE_TREE_SHAPE(enp->en_child[i]);
		}

	UPDATE_NODE_SHAPE(enp);
}


/*
 * These are things we do when we see an assignment.
 * We get the shapes of the lhs, and the rhs.
 *
 * If both are known, then we check for a match.
 */

static void compute_assign_shape(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Shape_Info *shpp;
	prec_t prec;

	/* Here the shape is the shape of the LHS;
	 * If the lhs is a scalar and the rhs is a vector,
	 * that's an error
	 */

	if( mode_is_matlab ){
		/* in matlab, the size can be determined from the RHS */
		/* but we really only want to do this if the LHS is an object??? */
		/* && enp->en_child[0]->en_code == T_DYN_OBJ ){ */

		POINT_NODE_SHAPE(enp,enp->en_child[1]->en_shpp);
		if( UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) )
			LINK_UK_NODES(enp,enp->en_child[1]);
		return;
	}

#ifdef CAUTIOUS
	if( enp->en_child[0]->en_shpp == NO_SHAPE ){
		/* a syntax error?? */
		NODE_ERROR(enp);
		WARN("CAUTIOUS:  compute_assign_shape:  No shape for LHS");
		if( enp->en_child[1]->en_shpp != NO_SHAPE )
			prec=enp->en_child[1]->en_shpp->si_prec;
		else	prec=PREC_SP;			/* BUG cautious compute_assign_shape */
		POINT_NODE_SHAPE(enp->en_child[0],uk_shape(prec));
		enp->en_child[0]->en_shpp = get_child_shape(QSP_ARG  enp,0);
	}

	if( enp->en_child[1]->en_shpp == NO_SHAPE ){
		/* a syntax error?? */
		NODE_ERROR(enp);
		sprintf(error_string,"CAUTIOUS:  compute_assign_shape:  No shape for RHS (%s node n%d)",
			NNAME(enp->en_child[1]),enp->en_child[1]->en_serial);
		WARN(error_string);
		if( enp->en_child[0]->en_shpp != NO_SHAPE )
			prec=enp->en_child[0]->en_shpp->si_prec;
		else	prec=PREC_SP;			/* BUG cautious compute_assign_shape */
		POINT_NODE_SHAPE(enp->en_child[1],uk_shape(prec));
	}
#endif /* CAUTIOUS */


	/* Check for both shapes known */

	if( (! UNKNOWN_SHAPE(enp->en_child[0]->en_shpp)) && ! UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){

		/* make sure the shape of the expression is compatible
		 * with that of the target
		 */

		if( (shpp=SHAPES_MATE(enp->en_child[0],enp->en_child[1],enp)) == NO_SHAPE ){
			if( (shpp=compatible_shape(enp->en_child[0],enp->en_child[1],enp)) == NO_SHAPE ){
				/* Here we might want to allow assignment of a row to an image... */
				NODE_ERROR(enp);
				WARN("compute_assign_shape:  assignment shapes do not mate");
				advise(node_desc(enp->en_child[0]));
				describe_shape(enp->en_child[0]->en_shpp);
				advise(node_desc(enp->en_child[1]));
				describe_shape(enp->en_child[1]->en_shpp);
DUMP_TREE(enp);
				CURDLE(enp)
			} else {
				/* Now we know that we are doing something odd,
				 * like assigning an image from a row or a column...
				 */
				enp->en_code = T_DIM_ASSIGN;
				COPY_NODE_SHAPE(enp,shpp);
				return;
			}
		}

		POINT_NODE_SHAPE(enp,shpp);
		return;
	}

	/* Now we know that at least one of the shapes is unknown */

	/* Check if both are unknown */

	if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) && UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){

		/* Both sides of the assignment have unknown
		 * shape.  We need to scan both sides, and update
		 * everyone's lists...
		 */

		/* We used to point to the shape of one of the unknown nodes, but that created
		 * a problem:  if the pointed to object resolved during execution,
		 * it would update the assign node's shape making everything
		 * look hunky-dory, so runtime resolution would not be applied
		 * to the other branch.
		 * Therefore we call uk_shape
		 */

		POINT_NODE_SHAPE(enp, uk_shape(enp->en_child[0]->en_shpp->si_prec) );

		/* remember_uk_assignment(enp); */
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"compute_assign_shape calling link_uk_nodes for %s and child %s",
node_desc(enp),node_desc(enp->en_child[0]));
advise(error_string);
sprintf(error_string,"compute_assign_shape calling link_uk_nodes for %s and child %s",
node_desc(enp),node_desc(enp->en_child[1]));
advise(error_string);
}
#endif /* DEBUG */
		LINK_UK_NODES(enp,enp->en_child[0]);
		LINK_UK_NODES(enp,enp->en_child[1]);
		return;
	}

	/* Now we know that only one of the two shapes is unknown.
	 *
	 * If the known shape is a vector, we use that shape for the assignemnt;
	 * If the LHS is a scalar, we use that for the assignemnt;
	 * If the RHS is a scalar and the LHS is unknown, then the assignment is unknown.
	 */

	/* Now we know that the RHS has unknown shape, but the LHS is known */
	/* We do the redundant test anyway... */

	if( ! UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){		/* LHS shape known */
		LINK_UK_NODES(enp,enp->en_child[1]);
		/* we could resolve here and now? */

		POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);

		return;
	}

	if( ! UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){		/* RHS shape known */
		if( !SCALAR_SHAPE(enp->en_child[1]->en_shpp) )
			POINT_NODE_SHAPE(enp,enp->en_child[1]->en_shpp);
		else
			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);

		LINK_UK_NODES(enp,enp->en_child[0]);

		return;
	}
} /* end compute_assign_shape */

/* Reexamine an assignment, but don't bother to remember anything.
 *
 * With the new scheme, is this different from compute_assign_shape??
 */

static void update_assign_shape(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	prec_t prec;

	/* Here the shape is the shape of the LHS;
	 * If the lhs is a scalar and the rhs is a vector,
	 * that's an error
	 */

#ifdef CAUTIOUS
	if( enp->en_child[0]->en_shpp == NO_SHAPE ){
		/* a syntax error?? */
		NODE_ERROR(enp);
		WARN("CAUTIOUS:  update_assign_shape:  No shape for LHS");
		if( enp->en_child[1]->en_shpp != NO_SHAPE )
			prec=enp->en_child[1]->en_shpp->si_prec;
		else	prec=PREC_SP;			/* BUG cautious update_assign_shape */
		POINT_NODE_SHAPE(enp->en_child[0],uk_shape(prec));
	}

	if( enp->en_child[1]->en_shpp == NO_SHAPE ){
		/* a syntax error?? */
		NODE_ERROR(enp);
		WARN("CAUTIOUS:  update_assign_shape:  No shape for RHS");
		if( enp->en_child[0]->en_shpp != NO_SHAPE )
			prec=enp->en_child[0]->en_shpp->si_prec;
		else	prec=PREC_SP;			/* BUG cautious update_assign_shape */
		POINT_NODE_SHAPE(enp->en_child[1],uk_shape(prec));
	}

#endif /* CAUTIOUS */

	/* Check for both shapes known */

	if( ! UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) && ! UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){

		/* make sure the shape of the expression is compatible
		 * with that of the target
		 */

sprintf(error_string,"update_assign_shape %s:  shapes are known, calling shapes_mate",node_desc(enp));
advise(error_string);
DUMP_TREE(enp);

		if( ! SHAPES_MATE(enp->en_child[0],enp->en_child[1],enp) ){
			NODE_ERROR(enp);
			WARN("update_assign_shape:  assignment shapes do not mate");
			describe_shape(enp->en_child[0]->en_shpp);
			describe_shape(enp->en_child[1]->en_shpp);
			CURDLE(enp)
		} else {
			POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
		}
		return;
	}

	/* Now we know that at least one of the shapes is unknown */

	/* Check if both are unknown */

	if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) && UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){

		/* Both sides of the assignment have unknown
		 * shape.  We need to scan both sides, and update
		 * everyone's lists...
		 */

		POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
		return;
	}

	/* Now we know that only one of the two shapes is unknown */

	if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
		POINT_NODE_SHAPE(enp,enp->en_child[0]->en_shpp);
	} else {
		POINT_NODE_SHAPE(enp,enp->en_child[1]->en_shpp);
	}
} /* update_assign_shape */

const char *get_lhs_name(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	switch(enp->en_code){
		case T_POINTER:
		case T_STR_PTR:
		case T_DYN_OBJ:
			return(enp->en_string);
		case T_STATIC_OBJ:
			return(enp->en_dp->dt_name);
		case T_SUBVEC:
		case T_CSUBVEC:
		case T_REAL_PART:
		case T_IMAG_PART:
		case T_SQUARE_SUBSCR:
		case T_CURLY_SUBSCR:
		case T_DEREFERENCE:
		case T_REFERENCE:
		case T_SUBSCRIPT1:	/* matlab */

		case T_RET_LIST:	/* BUG there is more than one name!? */
			return( GET_LHS_NAME(enp->en_child[0]) );

		case T_OBJ_LOOKUP:
			return( EVAL_STRING(enp->en_child[0]) );

		case T_UNDEF: return("undefined symbol");

		default:
			MISSING_CASE(enp,"get_lhs_name");
			break;
	}
	return(NULL);
}

static const char *struct_name(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	char buf[LLEN];
	static char sn[LLEN];

	switch(enp->en_code){
		case T_STRUCT:
			sprintf(buf,"%s.%s",enp->en_string,struct_name(QSP_ARG  enp->en_child[0]));
			strcpy(sn,buf);
			return(sn);

		case T_DYN_OBJ:
			return(enp->en_string);

		default:
			MISSING_CASE(enp,"struct_name");
			break;
	}
	return("xxx");
}

/* We call count_name_refs to see if an assignment target appears in the RHS.
 * If it does, then we have to be careful in using it as the assignment target
 * during the computation, because the partially written results may disrupt
 * the computation.
 */

static int count_name_refs(QSP_ARG_DECL  Vec_Expr_Node *enp,const char *lhs_name)
{
	int i1,i2,i3,i4;

	switch(enp->en_code){
		case T_STRUCT:
			/* check the node itself, and its child,
			 * This is probably wrong, but we do it
			 * until we understand...
			 */
			if( strcmp( struct_name(QSP_ARG  enp), lhs_name ) ){
				enp->en_lhs_refs=0;
			} else {
				enp->en_lhs_refs=1;
			}
			break;

		case T_FUNCPTR:
		case T_POINTER:
		case T_STR_PTR:
		case T_DYN_OBJ:
			if( strcmp(enp->en_string,lhs_name) ){
				enp->en_lhs_refs=0;
			} else {
				enp->en_lhs_refs=1;
			}
			break;
		case T_STATIC_OBJ:
			if( strcmp(enp->en_dp->dt_name,lhs_name) ){
				enp->en_lhs_refs=0;
			} else {
				enp->en_lhs_refs=1;
			}
			break;

		/* 4 children to check */
		case T_VV_VV_CONDASS:
		case T_VS_VV_CONDASS:
		case T_VV_VS_CONDASS:
		case T_VS_VS_CONDASS:
			i1 = COUNT_NAME_REFS(enp->en_child[0],lhs_name);
			i2 = COUNT_NAME_REFS(enp->en_child[1],lhs_name);
			i3 = COUNT_NAME_REFS(enp->en_child[2],lhs_name);
			i4 = COUNT_NAME_REFS(enp->en_child[3],lhs_name);
			enp->en_lhs_refs = i1+i2+i3+i4;
			break;

		/* 3 children to check */
		case T_RANGE:			/* matlab */
		case T_RAMP:
		case T_SCROLL:
		case T_CSUBVEC:
		case T_VV_S_CONDASS:		/* count_name_refs() */
		case T_VS_S_CONDASS:		/* count_name_refs() */
		case T_SS_S_CONDASS:		/* count_name_refs() */
		case T_VV_B_CONDASS:		/* count_name_refs() */
		case T_VS_B_CONDASS:		/* count_name_refs() */
		case T_SS_B_CONDASS:		/* count_name_refs() */
			i1 = COUNT_NAME_REFS(enp->en_child[0],lhs_name);
			i2 = COUNT_NAME_REFS(enp->en_child[1],lhs_name);
			i3 = COUNT_NAME_REFS(enp->en_child[2],lhs_name);
			enp->en_lhs_refs = i1+i2+i3;
			break;

		case T_SUBVEC:
			i1 = COUNT_NAME_REFS(enp->en_child[0],lhs_name);
			if( enp->en_child[1] == NO_VEXPR_NODE )
				i2=0;
			else
				i2 = COUNT_NAME_REFS(enp->en_child[1],lhs_name);
			if( enp->en_child[2] == NO_VEXPR_NODE )
				i2=0;
			else
				i2 = COUNT_NAME_REFS(enp->en_child[2],lhs_name);

			enp->en_lhs_refs = i1+i2;
			break;

		/* 2 children to check */
		case T_SUBSAMP:
		case T_CSUBSAMP:
		case T_INDIR_CALL:
		case T_SQUARE_SUBSCR:
		case T_CURLY_SUBSCR:
		case T_INNER:
		case T_ARGLIST:
		case T_EXPR_LIST:
		case T_ROW_LIST:			/* count_name_refs */
		case T_COMP_LIST:
		case T_STRING_LIST:
		case T_PRINT_LIST:
		case T_MIXED_LIST:
		case T_MATH2_FN:
		case T_MATH2_VFN:
		case T_MATH2_VSFN:
		/* ALL_BINOP_CASES */
		case T_VV_FUNC:				/* count_name_refs */
		case T_VS_FUNC:
		ALL_SCALAR_BINOP_CASES
		case T_RANGE2:			/* matlab */

		BINARY_BOOLOP_CASES			/* count_name_refs */
		ALL_NUMERIC_COMPARISON_CASES		/* count_name_refs */

		case T_SCALMAX:
		case T_SCALMIN:
		case T_ASSIGN:
		/* matlab */
		/* case T_ROWLIST: */		/* count_name_refs */
		case T_ROW:
		case T_INDEX_LIST:
		case T_SUBSCRIPT1:	/* matlab */
		case T_INDEX_SPEC:	/* matlab */

			i1 = COUNT_NAME_REFS(enp->en_child[0],lhs_name);
			i2 = COUNT_NAME_REFS(enp->en_child[1],lhs_name);
			enp->en_lhs_refs = i1+i2;
			break;

		/* one child to check */
		case T_BOOL_NOT:
		case T_FIX_SIZE:
		case T_CALLFUNC:			/* count name refs */
		case T_CALL_NATIVE:			/* count name refs */
		case T_ENLARGE:  case T_REDUCE:
		case T_DILATE: case T_ERODE:
		/* ALL_UNARY_CASES */ /* don't include T_SCROLL */
			/* Why do we special case only these codes???
			 * Why -
			 */

			if( enp->en_child[0] == NO_VEXPR_NODE ){
				enp->en_lhs_refs=0;
				break;
			}
			/* else fall through */
		case T_MAXVAL:
		case T_MINVAL:
		case T_OBJ_LOOKUP:
		case T_DEREFERENCE:
		case T_REFERENCE:		/* count_name_refs */
		case T_TYPECAST:
		case T_REAL_PART:  case T_IMAG_PART:
		case T_TRANSPOSE:
		case T_WRAP:
		case T_DFT:			/* count_name_refs() */
		case T_RDFT:			/* count_name_refs() */
		case T_RIDFT:
		case T_UMINUS:
		case T_BITCOMP:
		case T_VCOMP:
		case T_RECIP:
		case T_MATH1_VFN:
		case T_MATH1_FN:
		case T_SUM:
		case T_LIST_OBJ:		/* count_name_refs */
		case T_COMP_OBJ:		/* count_name_refs */
		case T_MAX_INDEX:
		case T_MIN_INDEX:
		case T_CONJ:
			enp->en_lhs_refs = COUNT_NAME_REFS(enp->en_child[0],lhs_name);
			break;

		/* no children to check */
		case T_MATH0_FN:
		case T_MATH0_VFN:
		case T_END:
		case T_UNDEF:
		case T_LIT_DBL:
		case T_LIT_INT:
		case T_STRING:
		case T_SIZE_FN:		/* may refer to lhs, but not to values! */
		case T_STR1_FN:		/* may refer to lhs, but not to values! */
		case T_STR2_FN:		/* may refer to lhs, but not to values! */
		case T_LOAD:		/* count_name_refs() */
		case T_FIRST_INDEX:	/* count_name_refs (matlab) */
		case T_LAST_INDEX:	/* count_name_refs (matlab) */
			enp->en_lhs_refs=0;
			break;

		case T_MAX_TIMES:
			enp->en_lhs_refs = COUNT_NAME_REFS(enp->en_child[2],lhs_name);
			break;

		default:
			MISSING_CASE(enp,"count_name_refs");
			enp->en_lhs_refs=0;
			break;
	}
	return(enp->en_lhs_refs);
}

/* Scan an assignment tree, looking on the right side for references
 * to the lhs target.  If there is more than one, we will need to use
 * a temp obj for results, to avoid overwriting the lhs before we are
 * finished with its old value.
 */

static void count_lhs_refs(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	const char *lhs_name;

	/* find the name of the lhs object */

	lhs_name = GET_LHS_NAME(enp->en_child[0]);

	/* we may get a null name if we have a variable string ( obj_of(ptr) )... */
	if( lhs_name == NULL ){
		enp->en_lhs_refs = 0;
		return;
	}

	enp->en_lhs_refs = COUNT_NAME_REFS(enp->en_child[1],lhs_name);
}

/* Lists get built up in a lopsided way, because of how the parser works:
 *
 *
 *		      c
 *		     / \
 *		    b   4
 *		   / \
 *		  a   3
 *		 / \
 *		1   2
 *
 * Nodes a and b are balanced, but c is not...
 * We can fix things by making c the r-child of b, and leaf 3 the l-child of c:
 *
 *		     b
 *		   /   \
 *		  a     c
 *		 / \   / \
 *		1   2 3   4
 *
 * Let;s look at a larger example:
 *
 * 1.		          e
 *		         / \
 *		        d   6
 *		       / \
 *		      c   5
 *		     / \
 *		    b   4
 *		   / \
 *		  a   3
 *		 / \
 *		1   2
 *
 *			
 * 2.		           e
 *		          / \
 *		         d   6
 *		       /   \
 *		     b      5
 *		   /   \
 *		  a     c
 *		 / \   / \
 *		1   2 3   4
 *
 *
 * 3.		       e
 *		      / \
 *		     b   6
 *		   /   \
 *		  a     d
 *		 / \   / \
 *		1   2 c   5
 *		    /  \
 *		   3    4
 *
 *
 *		     b 
 * 4.		   /   \
 *		  a     e
 *		 / \   / \
 *		1   2 d   6
 *		    /  \
 *		   c    5
 *		  / \
 *		 3   4
 *
 * Now node B is right-heavy!
 *
 *
 * 5a.		         e
 *		       /   \
 *		     b      6
 *		   /   \
 *		  a     d
 *		 / \   / \
 *		1   2 c   5
 *		    /  \
 *		   3    4
 *
 * If we deal with it right away, we would get this, and then dealing with
 * e we would be in an infinite loop, We need to balance node e in the above case 4 first!
 *
 *
 *		     b 
 * 5.		   /   \
 *		  a       d
 *		 / \    /   \
 *		1   2 c       e
 *		     / \     / \
 *		    3   4   5   6
 *
 * Done!
 */

static Vec_Expr_Node * lighten_branch(Vec_Expr_Node *enp, int heavy_side)
{
	Vec_Expr_Node *exch_enp, *parent_enp;
	int i,light_side, index;

	light_side = heavy_side == 0 ? 1 : 0 ;

	exch_enp = enp->en_child[heavy_side];

/*		  a
 *		 / \
 *		1   b
 *		   / \
 *		  2   c
 *		     / \
 *		    3   4
 */
	enp->en_child[heavy_side] = exch_enp->en_child[light_side];
	enp->en_child[heavy_side]->en_parent = enp;


/*		      a
 *		     / \
 *		    1   2
 *
 *
 *		    b
 *		   / \
 *		  2   c
 *		     / \
 *		    3   4
 */

	parent_enp = enp->en_parent;

	/* There may be no parent node, if this is the root of a subrt body */
	if( parent_enp != NO_VEXPR_NODE ){
		index = (-1);
		for(i=0;i<MAX_CHILDREN(parent_enp);i++)
			if( parent_enp->en_child[i] == enp ){
				index=i;
				i=MAX_NODE_CHILDREN+1;
			}
#ifdef CAUTIOUS
		if( index < 0 ){
			sprintf(DEFAULT_ERROR_STRING,"lighten_branch %s: is not a child of %s!?",node_desc(enp),node_desc(parent_enp));
			NERROR1(DEFAULT_ERROR_STRING);
		}
#endif /* CAUTIOUS */

		parent_enp->en_child[index] = exch_enp;
		exch_enp->en_parent = parent_enp;
	} else {
		exch_enp->en_parent = NO_VEXPR_NODE;
	}

	exch_enp->en_child[light_side] = enp;
	enp->en_parent = exch_enp;

	/* BUG there is a more efficient way to compute these node counts! */
	enp->en_n_elts = leaf_count(enp,enp->en_code);
	exch_enp->en_n_elts = leaf_count(exch_enp,enp->en_code);
	/* DONE! */

	return(exch_enp);
}

/* balance a list node.
 * If an action is taken, the root node may be moved, so we return it...
 */

static Vec_Expr_Node * balance_list(Vec_Expr_Node *enp, Tree_Code list_code )
{
	int n1,n2;
	Vec_Expr_Node *root_enp;

#ifdef CAUTIOUS
	if( enp == NO_VEXPR_NODE ){
		NWARN("CAUTIOUS:  balance_list passed null node");
		return(enp);
	}
#endif /* CAUTIOUS */

	if( enp->en_code != list_code ) return(enp);

	/* first balance the subtrees */
	if( balance_list(enp->en_child[0],list_code) == NO_VEXPR_NODE )
		return NO_VEXPR_NODE;
	if( balance_list(enp->en_child[1],list_code) == NO_VEXPR_NODE )
		return NO_VEXPR_NODE;

	n1=leaf_count(enp->en_child[0],list_code);
	n2=leaf_count(enp->en_child[1],list_code);

	if( n1 > (2*n2) ){	/* node is left_heavy */
		root_enp = lighten_branch(enp,0);
		balance_list(enp,list_code);
	} else if( n2 > (2*n1) ){	/* node is right-heavy */
		root_enp = lighten_branch(enp,1);
		balance_list(enp,list_code);
	} else root_enp = enp;
	return(root_enp);
}


static Shape_Info *cpx_scalar_shape(prec_t prec)
{
	int i;

	if( _cpx_scalar_shpp[prec]!=NO_SHAPE )
		return(_cpx_scalar_shpp[prec]);

	_cpx_scalar_shpp[prec] = (Shape_Info *)getbuf(sizeof(Shape_Info));
	for(i=0;i<N_DIMENSIONS;i++){
		_cpx_scalar_shpp[prec]->si_type_dim[i]=1;
		_cpx_scalar_shpp[prec]->si_mach_dim[i]=1;
	}
	_cpx_scalar_shpp[prec]->si_mach_dim[0]=2;

	_cpx_scalar_shpp[prec]->si_flags = 0;

	_cpx_scalar_shpp[prec]->si_prec = (prec&~PSEUDO_PREC_MASK) | COMPLEX_PREC_BITS;

	SET_SHAPE_FLAGS(_cpx_scalar_shpp[prec],NO_OBJ);

	return(_cpx_scalar_shpp[prec]);
}

void shapify(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	UPDATE_NODE_SHAPE(enp);
}

