
#include "quip_config.h"

char VersionId_vectree_resolve[] = QUIP_VERSION_STRING;

/* resolve.c
 *
 * Scanning the tree does a number of things:
 * It figures out the "shape" of each node, that is,
 * the dimensions of the object the node represents.
 * (Actually, that is done by prelim_node_shape and update_node_shape!?)
 *
 * Should nodes "own" their shape structs?
 * For Objects, we can't just point to dp->dt_shape, because
 * the objects can be dynamically created and destroyed.
 * (e.g. automatic objects within subroutines)
 * So we point references to the shape at the declaration
 * node for an object (which DOES own the shape)...
 *
 * Which nodes need shape info?  Primarly, all the constituents
 * of expressions, to allow error checking.
 * (Error checking is also performed within veclib, but because veclib
 * is more-or-less independent of the vectree module, it is difficult
 * to refer the errors back to the source code from veclib.)
 *
 * Object declaration nodes - copy from object, overwrite during rescan
 * Object reference nodes - point to decl node shape
 * Operator nodes (+,-, etc) point to the shape of one of the children.
 * Subrt nodes (CALLFUNC) own their own shapes, since a given subroutine
 * can return a shape which depends upon its arguments.
 *
 * Subroutine shape is dynamically set...
 *
 * The really tricky part here is resolution of unknown shapes.
 *
 * The assignment of a shape to an unknown shape we call "resolution".
 * Resolution can be eternal (for a global object) or transient (for
 * an object whose scope is restricted to a subroutine.
 * (Actually, a transient object can have eternal resolution IF it is
 * resolved by an eternal shape.)
 *
 * We need to have a mechanism for forgetting transient resolutions,
 * and more importantly, knowing WHEN to forget.  Typically, this would
 * be upon exit from a subroutine.
 *
 * In the following examples, u and v are unknown shapes, a, b, and c are known eternally,
 * and p, q, and r are known transiently.
 *
 * If a global object is resolved by a transient shape, it's shape is eternal.
 *
 *
 * We have several opportunities to resolve unknown shapes:
 *
 * At compile time:
 *			a+u		siblings in an expression
 *			a=u;		siblings in assignment
 *			u=a;
 *
 * At run time (or subroutine call time?):
 *
 *			float u[a][b];	here the shape of u depends not on the shape
 *					of a and b, but rather on their runtime values.
 *
 * Most of the complications arise from function calls;  For example:
 *
 * float f2(v[])
 * {
 *	return(2*v);
 * }
 *
 * float f1(u[])
 * {
 *	return(1+f2(u));
 * }
 *
 * w=f1(a);
 *
 * How do we resolve w?  w gets it shape from f1, which is unknown...  f1 is
 * resolved by the return value of f2, which is unknown...  f2 is resolved by
 * the argument v, which is assigned from u, which is unknown; u is resolved
 * by the known argument a...  So the sequence of resolutions is:
 *
 * u <- a
 * v <- u
 * f2 <- v
 * f1 <- f2
 * w <- f1
 *
 * HOWEVER what if we instead had
 *
 * a=f1(w)
 *
 * Now the sequence of resolutions is reversed:
 *
 * f1 <- a
 * f2 <- f1
 * v <- f2
 * u <- v
 * w <- u
 *	
 * One approach we could try would be to keep a list of possible resolutions for each uko, e.g.:
 *
 * f1_ret     : lhs,        f2_ret
 * f2_ret     : f2_arg (v), f1_ret
 * f2_arg (v) : f2_ret,     f1_arg (u)
 * f1_arg (u) : f2_arg (v), f1_argval
 *
 * What we can see here is that the lhs or the f1_argval will resolve all the others:
 *
 * lhs -> f1_ret, f1_arg, f2_ret, f2_arg
 * f1_argval -> f1_ret, f1_arg, f2_ret, f2_arg
 *
 * Obviously, if lhs and f1_argval both have known shapes, then they must match!
 *
 * What about this:
 *
 * int a=2,b=3;
 * float u[],v[];
 *
 * float f1(int x,int y)
 * {
 *	float w[x][y];
 *	w=x*y;
 *	return(w);
 * }
 *
 * u=func(a,b);
 * b=4;
 * v=func(a,b);
 *
 * w,f1_ret <- values(f1_args)
 * u        <- f1_ret
 * v        <- f1_ret
 *
 * These resolutions can only happen at runtime, and should be forgotten when f1 returns.
 *
 * Proposed approach:
 *
 * For each node with an unknown shape, we keep a list of resolvers.  We link back,
 * so that each possible resolver has a list of possible resolvee's.  When we make an eternal
 * resolution, we should (but are not compelled to) clean up the lists...  When we make
 * a transient resolution, we should record the resolved node on a list owned by the
 * subroutine whose responsibility it will be to do the forgetting upon return.
 *
 * Here is one problem with that approach as currently implemented:
 *
 * float a[8];
 * int n;
 * float v[];
 *
 * ...		# fiddle with n
 *
 * v=a[0:n];
 *
 * In this example, v is resolved from the subvector a[0:n], but since this depends on the value
 * of n, we can't resolve until the statement is executed - or, at the earliest, after the last time
 * that the value of n is modified...  It may be hard to figure out when that is!?
 *
 * The current implementation sort of blindly searches the tree - how do people figure this out
 * when they stare at the code?  It would probably be cleaner if we could figure out how to analyze
 * this at compile time...
 *
 * BUG The division into early and late call-time is probably bad.  Too many things get tried too many
 * times, making execution very inefficient.
 *
 * Here is a new proposed set of stages for resolution:
 *
 * compile-time		resolve from global objects of known shape
 * call-time		resolve from subroutine args and return targets
 * run-time		objects whose declarations or subscripts depend on variables...
 *
 * I wonder who in the world has thought deeply about this?
 * Data flow analysis?
 */

#include <stdio.h>
#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* sleep() */
#endif

#include "savestr.h"
#include "data_obj.h"
#include "debug.h"
#include "getbuf.h"
#include "node.h"
#include "function.h"
#include "nexpr.h"
#include "nvf_api.h"
#include "query.h"
#include "chewtext.h"
#include "vec_util.h"		/* dilate, erode */
#include "img_file.h"
#include "filetype.h"
#include "fileck.h"		/* file exists */

#include "vectree.h"

#define MAX_HIDDEN_CONTEXTS	32

/* local prototypes */

static void resolve_uk_nodes(QSP_ARG_DECL  List *lp);
static void resolve_obj_list(QSP_ARG_DECL  Vec_Expr_Node *enp, Shape_Info *shpp);
static Vec_Expr_Node *resolve_parent(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Shape_Info *shpp);
static Vec_Expr_Node * resolve_unknown_parent(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp);
static Vec_Expr_Node * resolve_unknown_child(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp);

#define RESOLVE_UK_NODES(lp)		resolve_uk_nodes(QSP_ARG  lp)
#define RESOLVE_OBJ_LIST(enp,shpp)	resolve_obj_list(QSP_ARG  enp,shpp)
#define RESOLVE_PARENT(uk_enp,shpp)	resolve_parent(QSP_ARG  uk_enp,shpp)
#define RESOLVE_UNKNOWN_PARENT(uk_enp,enp)	resolve_unknown_parent(QSP_ARG  uk_enp,enp)
#define RESOLVE_UNKNOWN_CHILD(uk_enp,enp)	resolve_unknown_child(QSP_ARG  uk_enp,enp)

#define RESOLVE_OBJ_ID(idp,shpp)	resolve_obj_id(QSP_ARG  idp,shpp)
#define RESOLVE_OBJECT(uk_enp,shpp)	resolve_object(QSP_ARG  uk_enp,shpp)

uint32_t resolution_flags=0;		/* why is this global? */
debug_flag_t resolve_debug=0;


#define max( n1 , n2 )		(n1>n2?n1:n2)


/* find_uk_nodes
 *
 * Descend this node, looking for unknown shapes.
 * When we find one, we add it to the list.
 */

static void find_uk_nodes(QSP_ARG_DECL  Vec_Expr_Node *enp,List *lp)
{
	Node *np;
	int i;

	if( enp->en_shpp != NO_SHAPE && UNKNOWN_SHAPE(enp->en_shpp) && enp->en_prec != PREC_VOID ){
		switch(enp->en_code){
			case T_EXPR_LIST:
			ALL_DECL_CASES
			case T_UNDEF:
				/* We don't resolve from declaration nodes */
				break;

			/*
			ALL_VECTOR_SCALAR_CASES
			ALL_VECTOR_VECTOR_CASES
			*/
			case T_VS_FUNC:

			ALL_OBJREF_CASES
			ALL_MATHFN_CASES
			ALL_UNARY_CASES
			ALL_CTL_FLOW_CASES
			ALL_CONDASS_CASES
			ALL_NUMERIC_COMPARISON_CASES
			BINARY_BOOLOP_CASES
			case T_BOOL_NOT:

			case T_SUM:
			case T_CALLFUNC:
			case T_INDIR_CALL:
			case T_RETURN:
			case T_FIX_SIZE:
			case T_POINTER:
			case T_REFERENCE:
			case T_EQUIVALENCE:
			case T_TYPECAST:
			case T_ASSIGN:
			case T_DFT:
			case T_RAMP:
			case T_RIDFT: case T_RDFT:
			case T_ENLARGE: case T_REDUCE:
			case T_LOAD:
			case T_VV_FUNC:		/* find_uk_nodes */
			case T_MAXVAL:
			case T_MINVAL:
			case T_TRANSPOSE:
/*
sprintf(error_string,"find_uk_nodes:  adding %s",node_desc(enp));
advise(error_string);
*/
				np=mk_node(enp);
				addTail(lp,np);
				break;
			default:
				MISSING_CASE(enp,"find_uk_nodes");
				np=mk_node(enp);
				addTail(lp,np);
				break;
		}
	}
	for(i=0;i<MAX_CHILDREN(enp);i++)
		if( enp->en_child[i] != NO_VEXPR_NODE )
			find_uk_nodes(QSP_ARG  enp->en_child[i],lp);
}

static List *get_uk_list(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	List *lp;

	lp=new_list();
	find_uk_nodes(QSP_ARG  enp,lp);
	if( eltcount(lp) == 0 ){
		dellist(lp);
		return(NO_LIST);
	}
	return(lp);
}

/* forget_resolved_tree
 *
 * Descend a tree (arg decls or body), looking for declarations of unknown shape objects.
 */

void forget_resolved_tree(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int i;

	if( RESOLVED_AT_CALLTIME(enp) ){
		//enp->en_flags &= ~ NODE_NEEDS_CALLTIME_RES;
#ifdef CAUTIOUS
		if( enp->en_shpp == NO_SHAPE ){
			sprintf(error_string,"CAUTIOUS:  forget_resolved_tree %s:  resolved node has no shape!?",
				node_desc(enp));
			ERROR1(error_string);
		}
#endif /* CAUTIOUS */

		if( enp->en_code == T_DYN_OBJ ){
			/* OBJECT nodes point to the shape at the decl node,
			 * so we leave it alone.
			 * BUT matlab has no decl nodes, so we'd better do something different?
			 */
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"forget_resolved_tree %s:  leaving OBJECT node alone",node_desc(enp));
advise(error_string);
}
#endif /* DEBUG */

		} else {
			if( OWNS_SHAPE(enp) )
				COPY_NODE_SHAPE(enp,uk_shape(enp->en_prec));
			else
				point_node_shape(QSP_ARG  enp,uk_shape(enp->en_prec));
		}
	}

	for(i=0;i<MAX_CHILDREN(enp);i++)
		if( enp->en_child[i] != NO_VEXPR_NODE )
			forget_resolved_tree(QSP_ARG  enp->en_child[i]);
} /* end forget_resolved_tree */

/* forget_resolved_shapes
 *
 * Shapes of unknown shape objs in this subroutine get stored in the declaration
 * nodes...  We used to keep a list of them, here we try to find them by rescanning
 * the declaration trees...
 */

void forget_resolved_shapes(QSP_ARG_DECL  Subrt *srp)
{
	/* First check the arguments */

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"forget_resolve_shapes %s:",srp->sr_name);
advise(error_string);
DUMP_TREE(srp->sr_body);
}
#endif /* DEBUG */

	if( srp->sr_arg_decls != NO_VEXPR_NODE )
		forget_resolved_tree(QSP_ARG  srp->sr_arg_decls);
	if( srp->sr_body != NO_VEXPR_NODE )
		forget_resolved_tree(QSP_ARG  srp->sr_body);

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"forget_resolve_shapes %s DONE",srp->sr_name);
advise(error_string);
DUMP_TREE(srp->sr_body);
}
#endif /* DEBUG */

}

/* late_calltime_resolve
 *
 * called after subroutine context has been set, and the arg values have been set.
 */

void late_calltime_resolve(QSP_ARG_DECL  Subrt *srp, Data_Obj *dst_dp)
{
	List *lp;
	u_long save_res;

#ifdef CAUTIOUS
	if( curr_srp != srp ){
		sprintf(error_string,"CAUTIOUS:  late_calltime_resolve %s:  current subrt is %s!?",
			srp->sr_name,curr_srp->sr_name);
		WARN(error_string);
	}
#endif /* CAUTIOUS */

/*
sprintf(error_string,"late_calltime_resolve %s setting SCANNING flag",srp->sr_name);
advise(error_string);
*/
	srp->sr_flags |= SR_SCANNING;
	save_res = resolution_flags;
	resolution_flags = NODE_NEEDS_CALLTIME_RES;

	/*
	if( srp->sr_prec != PREC_VOID ){
		if( ret_shpp != NO_SHAPE && ! UNKNOWN_SHAPE(ret_shpp) ){
			srp->sr_shpp = ret_shpp;
		} else {
			srp->sr_shpp = uk_shape(srp->sr_prec);
		}
	}
	*/
#ifdef CAUTIOUS
	  /*
	  else if( srp->sr_shpp != NO_SHAPE ){
		sprintf(error_string,"CAUTIOUS:  late_calltime_resolve %s:  prec is void, but shape is not null!?",
			srp->sr_name);
		WARN(error_string);
	}
	*/
#endif /* CAUTIOUS */

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"late_calltime_resolve %s begin:",srp->sr_name);
advise(error_string);
DUMP_SUBRT(srp);
}
#endif /* DEBUG */

	lp = get_uk_list(QSP_ARG  srp->sr_body);
	if( lp != NO_LIST ){
		RESOLVE_UK_NODES(lp);

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"late_calltime_resolve %s done:",srp->sr_name);
advise(error_string);
DUMP_SUBRT(srp);
}
#endif /* DEBUG */

		dellist(lp);
	}

/*
sprintf(error_string,"late_calltime_resolve %s clearing SCANNING flag",srp->sr_name);
advise(error_string);
*/
	srp->sr_flags &= ~SR_SCANNING;

	resolution_flags = save_res;
}

static void find_known_return(QSP_ARG_DECL  Vec_Expr_Node *enp,Subrt *srp)
{
	int i;

	if( enp == NO_VEXPR_NODE ) enp=srp->sr_body;

	switch(enp->en_code){
		case T_RETURN:

#ifdef CAUTIOUS
			if( enp->en_shpp == NO_SHAPE ){
				NODE_ERROR(enp);
				sprintf(error_string,
	"CAUTIOUS:  find_known_return %s:  return node has no shape!?",node_desc(enp));
				WARN(error_string);
				return;
			}
#endif /* CAUTIOUS */

			if( ! UNKNOWN_SHAPE(enp->en_shpp) ){
/*
sprintf(error_string,"find_known_return: setting subrt %s shape from %s",srp->sr_name,node_desc(enp));
advise(error_string);
describe_shape(enp->en_shpp);
*/
				srp->sr_shpp = enp->en_shpp;
			}
			return;

		/* for all of these cases we descend and keep looking */
		case T_STAT_LIST:
			break;

		case T_IFTHEN:
			/* child 0 is the test,and can't have a return statement */
			if( ! NULL_CHILD(enp,1) ) find_known_return(QSP_ARG  enp->en_child[1],srp);
			if( ! NULL_CHILD(enp,2) ) find_known_return(QSP_ARG  enp->en_child[2],srp);
			return;

		/* for all of these cases, we know there will not be a
		 * child return node, so we return right away
		 */
		case T_DECL_STAT:
		case T_DECL_STAT_LIST:
		case T_INFO:
		case T_DISPLAY:
		case T_EXIT:
		case T_ASSIGN:
		case T_SET_STR:
		case T_CALLFUNC:
		case T_SCRIPT:
		case T_EXP_PRINT:
			return;

		default:
			MISSING_CASE(enp,"find_known_return");
			break;
	}
	for(i=0;i<MAX_CHILDREN(enp);i++)
		if( ! NULL_CHILD(enp,i) )
			find_known_return(QSP_ARG  enp->en_child[i],srp);
} /* find_known_return */


/* resolve_tree
 *
 * descend a tree looking for a node with a resolver with known shape.
 * If we find 
 * Perhaps we should keep separate lists for child resolvers and parent resolvers?
 *
 * When do we call this?   We know we call it at calltime and runtime resolution...
 * Do we try to evaluate ptr refs?
 *
 * The whence arg is needed to prevent infinte recursion...
 */

void resolve_tree(QSP_ARG_DECL  Vec_Expr_Node *enp,Vec_Expr_Node *whence)
{
	Node *np;
	Vec_Expr_Node *enp2,*resolved_enp;
	Identifier *idp;
	Subrt *srp;
	Run_Info *rip;

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"BEGIN resolve_tree %s:",node_desc(enp));
advise(error_string);
DUMP_TREE(enp);
}
#endif /* DEBUG */

	/* We first handle a couple of special cases:
	 *
	 * Pointer nodes:  the pointer may have been set previously, but the
	 * pointer struct does not link to all the tree nodes that reference it.
	 * So when we see a pointer node, we check and see if it is set, and if
	 * it is we dereference it and see if the object has known shape;
	 * if it does, we point this node to that shape.
	 *
	 * Equivalence nodes are kind of like pointer nodes... the args should tell
	 * us the shape, but the args themselves may contain pointer references...
	 *
	 * Actually, *any* expression can contain pointer references???
	 *
	 * Callfunc nodes:  we want to find out if there is a return node with known
	 * shape.  Before we do this, we do both early and late calltime resolution.
	 * BUG?  for efficiency's sake, we probably should check before we bother
	 * to do these resolutions.
	 *
	 * Return nodes:  we figure out what subroutine we are returning from,
	 * and then see if we have a return shape for it.
	 *
	 * vv_func nodes can be outer ops, so we need to look at both children.
	 * 
	 */
	switch(enp->en_code){
		case T_SUBVEC:
		case T_CSUBVEC:
			/* we really want to do this as late as possible... */
			/* If the subvector is defined by indices which are variable, we need to set
			 * the NEEDS_CALLTIME_RES flag...
			 */
			UPDATE_TREE_SHAPE(enp);
			break;

		case T_VV_FUNC:
			if( ( ! UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ) &&
				! UNKNOWN_SHAPE(enp->en_child[1]->en_shpp ) ){
				Shape_Info *shpp;

				shpp = calc_outer_shape(enp->en_child[0],enp->en_child[1]);
				COPY_NODE_SHAPE(enp,shpp);
			}

			break;

		case T_EQUIVALENCE:
			/* We really just want to call update_node_shape here... */
			SHAPIFY(enp);
			enp->en_flags |= NODE_NEEDS_CALLTIME_RES;	/* this forces us to forget */
			break;

		case T_INDIR_CALL:
			/* First see if the function ptr has been set */
			/*
			WARN("resolve_tree:  not sure how to handle INDIR_CALL");
			*/
			break;

		case T_POINTER:
			idp = EVAL_PTR_REF(enp,EXPECT_PTR_SET);
			if( idp == NO_IDENTIFIER ) break;	/* probably not set */
#ifdef CAUTIOUS
			if( ! IS_POINTER(idp) ){
				NODE_ERROR(enp);
				sprintf(error_string,"Identifier %s is not a ptr",idp->id_name);
				WARN(error_string);
				break;
			}
#endif /* CAUTIOUS */
			if( ! POINTER_IS_SET(idp) ) break;

			if( idp->id_ptrp->ptr_refp == NO_REFERENCE ){
				/* This sometimes happens during resolution, but not sure why??? */
				/*
				sprintf(error_string,"CAUTIOUS:  resolve_tree %s:  ptr %s has null ref arg!?",
					node_desc(enp),idp->id_name);
				WARN(error_string);
				*/
				break;
			}
#ifdef CAUTIOUS
			if( idp->id_ptrp->ptr_refp->ref_dp == NO_OBJ ){
				sprintf(error_string,"CAUTIOUS:  resolve_tree %s:  ref dp is NULL!?",node_desc(enp));
				WARN(error_string);
				break;
			}
#endif /* CAUTIOUS */
			if( ! UNKNOWN_SHAPE( &idp->id_ptrp->ptr_refp->ref_dp->dt_shape ) ){
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resolve_tree POINTER %s:  object %s has known shape",node_desc(enp),
idp->id_ptrp->ptr_refp->ref_dp->dt_name);
advise(error_string);
describe_shape(&idp->id_ptrp->ptr_refp->ref_dp->dt_shape);
}
#endif /* DEBUG */
				RESOLVE_POINTER(enp,&idp->id_ptrp->ptr_refp->ref_dp->dt_shape);
				return;
/*
advise("after resolution pointer:");
DUMP_TREE(enp);
*/
			}

			break;

		case T_CALLFUNC:			/* resolve_tree */
							/* should we try a call-time resolution? */
/*
if( UNKNOWN_SHAPE(enp->en_shpp) ){
sprintf(error_string,"resolve_tree %s:  shape is unknown",node_desc(enp));
advise(error_string);
} else {
sprintf(error_string,"resolve_tree %s:  shape is KNOWN",node_desc(enp));
advise(error_string);
}
*/

			srp = runnable_subrt(QSP_ARG  enp);
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resolve_tree %s:  before setup_call %s:",node_desc(enp),srp->sr_name);
advise(error_string);
DUMP_TREE(enp);
}
#endif /* DEBUG */
			/* BUG how can we get dst_dp?? */
			rip = SETUP_CALL(srp,NO_OBJ);
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resolve_tree %s:  after setup_call %s:",node_desc(enp),srp->sr_name);
advise(error_string);
DUMP_TREE(enp);
}
#endif /* DEBUG */
			if( rip == NO_RUN_INFO ){
				return;
			}

			if( rip->ri_arg_stat >= 0 ){ 
				EVAL_DECL_TREE(srp->sr_body);
				LATE_CALLTIME_RESOLVE(srp,NO_OBJ);
#ifdef DEBUG
if( debug & resolve_debug ){
advise("resolve_tree:  after late_calltime_resolve:");
DUMP_TREE(srp->sr_body);
}
#endif /* DEBUG */
			}

			wrapup_context(QSP_ARG  rip);

			find_known_return(QSP_ARG  NO_VEXPR_NODE,srp);

			/* now maybe we have a return shape??? */
			if( srp->sr_shpp != NO_SHAPE && ! UNKNOWN_SHAPE(srp->sr_shpp) ){
/*
sprintf(error_string,"resolve_tree %s:  setting shape from %s return shape",
node_desc(enp),srp->sr_name);
advise(error_string);
*/
				COPY_NODE_SHAPE(enp,srp->sr_shpp);
			}

			break;

		case T_RETURN:
			/* For a return node, we look up the subroutine and
			 * see if the return shape is known.
			 */
			if( enp->en_srp->sr_dst_shpp != NO_SHAPE && enp->en_srp->sr_dst_shpp->si_prec != PREC_VOID ){
				if( ! UNKNOWN_SHAPE(enp->en_srp->sr_dst_shpp) ){
					/* We know the return destination shape! */
					point_node_shape(QSP_ARG  enp,enp->en_srp->sr_dst_shpp);
				}
/*
else {
sprintf(error_string,"resolve_tree %s:  subrt %s return shape unknown",
node_desc(enp),enp->en_srp->sr_name);
advise(error_string);
}
*/
			}
/*
else {
sprintf(error_string,"resolve_tree %s:  subrt %s return shape nonexistent",
node_desc(enp),enp->en_srp->sr_name);
advise(error_string);
}
*/

			break;

		/* Now we list all the other cases, for which we do nothing special here */

		case T_CALL_NATIVE:			/* matlab */
		case T_IMG_DECL:
		case T_VS_FUNC:
		ALL_MATH_VFN_CASES			/* resolve_tree */
		ALL_CONDASS_CASES
		//ALL_OBJREF_CASES
		NONSUBVEC_OBJREF_CASES
		/* case T_ROWLIST: */				/* matlab */
		ALL_NUMERIC_COMPARISON_CASES
		BINARY_BOOLOP_CASES
		case T_BOOL_NOT:
		ALL_DFT_CASES
		case T_TRANSPOSE:
		case T_REFERENCE:
		case T_ASSIGN:
		case T_RET_LIST:
		case T_TYPECAST:
		case T_RAMP:
		case T_LOAD:
		case T_FIX_SIZE:
		case T_RANGE2:
		ALL_UNARY_CASES
		case T_REDUCE:  case T_ENLARGE:
		case T_DECL_INIT:
		/* projection operator cases */
		case T_MAXVAL:
		case T_MINVAL:
		case T_SUM:

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resove_tree:  %s is not a special case",node_desc(enp));
advise(error_string);
}
#endif /* DEBUG */
			break;

		default:
			MISSING_CASE(enp,"resolve_tree (special cases)");
			break;
	}

	/* Now we are finished with the special cases.
	 * We traverse this node's list of resolvers, looking at all nodes
	 * except the one we got here from.  If the resolving node is unknown,
	 * we do a recursive call to resolve_tree on it.  After that, we check
	 * it again, and if it has a shape we then use it to resolve this node.
	 * BUG?  we should probably return right away after the resolution happens!?
	 *
	 * The code as originally written assumes that most nodes can be resolved
	 * from a single node - but now vv_func nodes (like the original outer_op
	 * nodes) need to have both their children examined...
	 * Does vv_func need to be another special case?
	 */

	if( enp->en_resolvers == NO_LIST ){
		/* We don't set the shapes of T_SUBVEC nodes if the range
		 * expressions contain variables (which will be set at runtime).
		 */
		UPDATE_TREE_SHAPE(enp);


		if( UNKNOWN_SHAPE(enp->en_shpp) && ( enp->en_code != T_SUBVEC
							&& enp->en_code != T_CSUBVEC
							&& enp->en_code != T_SUBSAMP
							&& enp->en_code != T_CSUBSAMP )
			){
			/* This message gets printed sometimes (see tests/outer_test.scr),
			 * but the computation seems to be done correctly, so we suppress
			 * the msg for now...
			 */
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resolve_tree %s:  no resolvers",node_desc(enp));
advise(error_string);
DUMP_TREE(enp);
}
#endif /* DEBUG */
		}
		return;
	}

	np=enp->en_resolvers->l_head;
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resolve_tree:  %s has %d resolver nodes",
node_desc(enp),eltcount(enp->en_resolvers));
advise(error_string);
}
#endif /* DEBUG */
	while(np!=NO_NODE){
		enp2 = (Vec_Expr_Node *)np->n_data;
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resolve_tree:  %s is a resolver node for %s",
node_desc(enp2),node_desc(enp));
advise(error_string);
}
#endif /* DEBUG */
#ifdef CAUTIOUS
		if( enp2 == NO_VEXPR_NODE ){
			sprintf(error_string,"CAUTIOUS:  resolve_tree:  %s has null node on resolver list!?",
				node_desc(enp));
			WARN(error_string);
		} else
#endif /* CAUTIOUS */
		if( enp2 != whence ){
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resolve_tree %s:  trying %s",node_desc(enp),node_desc(enp2));
advise(error_string);
}
#endif /* DEBUG */
			if( enp2->en_shpp == NO_SHAPE ||	/* could be a void callfunc node */
				UNKNOWN_SHAPE(enp2->en_shpp) )

				RESOLVE_TREE(enp2,enp);
			
			/* The above call to resolve_tree could have resolved enp2 */
			if( enp2->en_shpp!=NO_SHAPE && ! UNKNOWN_SHAPE(enp2->en_shpp) ){
				resolved_enp=EFFECT_RESOLUTION(enp,enp2);
				if(enp->en_resolvers != NO_LIST ){
					/* careful - we dont want a recursive call, we may
					 * have gotten here from resolve_uk_nodes!
					 */
					RESOLVE_UK_NODES(enp->en_resolvers);
				}
				return;
			}
		}
		np=np->n_next;
	}
} /* end resolve_tree() */



/* We call this with a list of nodes which may need resolution.
 * This can be called with a list we construct from a tree, or
 * with a node's list (propagation), so we therefore cannot assume
 * that all the nodes on the list actually need resolution.
 */

static void resolve_uk_nodes(QSP_ARG_DECL  List *lp)
{
	Node *np;

	np=lp->l_head;
	while(np!=NO_NODE){
		Vec_Expr_Node *enp;

		enp=(Vec_Expr_Node *)np->n_data;
		if( UNKNOWN_SHAPE(enp->en_shpp) ){
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resolve_uk_nodes trying to fix %s",node_desc(enp));
advise(error_string);
}
#endif /* DEBUG */
			/*
			resolve_one_uk_node(enp);
			*/
			RESOLVE_TREE(enp,NO_VEXPR_NODE);
		}

		/* If we remove nodes from the list as they are resolved,
		 * We can't do this next business...
		 */
		np=np->n_next;
	}
}

/* We call resolve_subrt from early_calltime_resolve
 * Currently, we are calling this BEFORE arg val assignment...
 * What is the context assumed to be?
 */

void resolve_subrt(QSP_ARG_DECL  Subrt *srp,List *uk_list, Vec_Expr_Node *argval_tree,Shape_Info *ret_shpp)
{
	Subrt *save_srp;
	int stat;
	Context_Pair *prev_cpp;

/*
if( ret_shpp != NO_SHAPE ){
sprintf(error_string,"resolve_subrt %s:  return shape known:",srp->sr_name);
advise(error_string);
describe_shape(ret_shpp);
}
*/

/*
sprintf(error_string,"resolve_subrt %s:  calling pop_previous",srp->sr_name);
advise(error_string);
*/
	prev_cpp = POP_PREVIOUS();

	save_srp = curr_srp;
	curr_srp = srp;

/*
sprintf(error_string,"resolve_subrt %s setting SCANNING flag",srp->sr_name);
advise(error_string);
*/
	srp->sr_flags |= SR_SCANNING;

	if( argval_tree != NO_VEXPR_NODE ){
		stat = CHECK_ARG_SHAPES(srp->sr_arg_decls,argval_tree,srp);
		if( stat < 0 ) {
sprintf(error_string,"resolve_subrt %s:  argument error",srp->sr_name);
WARN(error_string);
			goto givup;
		}
	}

	/* set the context */
/*
sprintf(error_string,"resolve_subrt %s:  done w/ check_arg_shapes",srp->sr_name);
advise(error_string);
DUMP_SUBRT(srp);
*/

	set_subrt_ctx(QSP_ARG  srp->sr_name);

	/* declare the arg variables */
	EVAL_DECL_TREE(srp->sr_arg_decls);	/* resolve_subrt() */
	EVAL_DECL_TREE(srp->sr_body);		/* resolve_subrt() */
	if( srp->sr_prec != PREC_VOID ){
		if( ret_shpp != NO_SHAPE && ! UNKNOWN_SHAPE(ret_shpp) ){
			srp->sr_shpp = ret_shpp;
		} else {
			srp->sr_shpp = uk_shape(srp->sr_prec);
		}
	}
#ifdef CAUTIOUS
	/*
	  else if( srp->sr_shpp != NO_SHAPE ){
		sprintf(error_string,"CAUTIOUS:  resolve_subrt %s:  prec is void, but shape is not null!?",
			srp->sr_name);
		WARN(error_string);
	}
	*/
#endif /* CAUTIOUS */

	/* we need to assign the arg vals for any ptr arguments! */

	srp->sr_dst_shpp = ret_shpp;

	RESOLVE_UK_NODES(uk_list);

	/* now try for the args */
	/* resolve_uk_args( */

	srp->sr_dst_shpp = NO_SHAPE;

/*
sprintf(error_string,"resolve_subrt %s:  deleting subrt context",srp->sr_name);
advise(error_string);
*/
	delete_subrt_ctx(QSP_ARG  srp->sr_name);

givup:

/*
sprintf(error_string,"resolve_subrt %s clearing SCANNING flag",srp->sr_name);
advise(error_string);
*/
	srp->sr_flags &= ~SR_SCANNING;

	curr_srp = save_srp;

	if( prev_cpp != NO_CONTEXT_PAIR ){
/*
sprintf(error_string,"resolve_subrt %s:  restoring context",srp->sr_name);
advise(error_string);
*/
		RESTORE_PREVIOUS(prev_cpp);
	}

	/* we may have been passed uko arg vals,
	 * and our caller may expect us to resolve these...
	 * see if the corresponding arg decls have their shapes set...
	 */

	if( argval_tree != NO_VEXPR_NODE )
		RESOLVE_ARGVAL_SHAPES(argval_tree,srp->sr_arg_decls,srp);

} /* end resolve_subrt() */

/* We call calltime_resolve before we call a subroutine,
 * before the arg vals are set.
 * We really need a way to call this (or something like it) AFTER the arg
 * vals have been set...
 *
 * Well, now we have the later version - why do we need the early one???
 */

void early_calltime_resolve(QSP_ARG_DECL  Subrt *srp, Data_Obj *dst_dp)
{
	List *lp;
	int save_exec;
	uint32_t save_res;

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"Begin early_calltime_resolve %s",srp->sr_name);
advise(error_string);
DUMP_SUBRT(srp);
}
#endif /* DEBUG */

	/* We are about to call a subroutine, but we are not actually executing.
	 * We have a destination object (which may be unknown or NULL for void).
	 * BUT we do not yet have the arg values!?
	 * Maybe what we want to do here is resolve the arg shapes?
	 *
	 * We begin by descending the tree and making a list of nodes
	 * that need to be resolved...  Then we attempt to resolve each one.
	 */

	lp = get_uk_list(QSP_ARG  srp->sr_body);
	if( lp == NO_LIST ){
#ifdef DEBUG
/*
if( debug & resolve_debug ){
sprintf(error_string,"early_calltime_resolve %s:  no UK nodes, returning",srp->sr_name);
advise(error_string);
DUMP_TREE(srp->sr_body);
}
*/
#endif /* DEBUG */
		return;
	}

	/* set up the subroutine context */


	save_exec = executing;
	executing=0;
	save_res = resolution_flags;
	resolution_flags = NODE_NEEDS_CALLTIME_RES;

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"early_calltime_resolve calling resolve_subrt %s, flags = 0x%x, saved flags = 0x%x",
srp->sr_name,resolution_flags, save_res);
advise(error_string);
}
#endif /* DEBUG */

	if( dst_dp == NO_OBJ )
		RESOLVE_SUBRT(srp,lp,srp->sr_arg_vals,NO_SHAPE);
	else
		RESOLVE_SUBRT(srp,lp,srp->sr_arg_vals,&dst_dp->dt_shape);

	executing = save_exec;

	/* Before we introduced pointers, it was an error to have any unknown shapes at
	 * this point...  but because ptrs can be assigned to objects of various shapes
	 * during execution, we really can't resolve things involving them until runtime
	 * of the lines in question after the assignments have been made.  This is
	 * going to be a problem if subroutine return values are needed to resolve
	 * upstream nodes, but we'll worry about *that* when we run into it...
	 */

	dellist(lp);

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"early_calltime_resolve %s DONE",srp->sr_name);
advise(error_string);
DUMP_SUBRT(srp);
}
#endif /* DEBUG */

	resolution_flags = save_res;
}

void xform_to_bitmap(Shape_Info *shpp)
{
#ifdef CAUTIOUS
	if( BITMAP_SHAPE(shpp) ){
		NWARN("CAUTIOUS:  xform_to_bitmap:  shape is already a bitmap!?");
		return;
	}
#endif /* CAUTIOUS */
	/* shpp->si_rowpad = N_PAD_BITS(shpp->si_cols); */
	/* shpp->si_cols = N_WORDS_FOR_BITS(shpp->si_cols); */
	shpp->si_prec = PREC_BIT;
	shpp->si_flags |= DT_BIT;
}

void xform_from_bitmap(Shape_Info *shpp,prec_t prec)
{
#ifdef CAUTIOUS
	if( ! BITMAP_SHAPE(shpp) ){
		NWARN("CAUTIOUS:  xform_from_bitmap:  shape is not a bitmap!?");
		return;
	}
#endif /* CAUTIOUS */
	/*shpp->si_cols = N_BITMAP_COLS(shpp); */
	/*shpp->si_rowpad=0; */
	shpp->si_prec = prec;
	shpp->si_flags &= ~ DT_BIT;
}

/* setup_spcdom_shape - setup space domain shape (from freq. domain transform shape) */

void setup_spcdom_shape(Shape_Info *spcdom_shpp, Shape_Info *xf_shpp)
{
	*spcdom_shpp = *xf_shpp;
	spcdom_shpp->si_cols -= 1;
	spcdom_shpp->si_cols *= 2;
	spcdom_shpp->si_comps = 1;
	/* BUG?  should we make sure that the known shape is complex? */
	/* The space domain shape is real */
	spcdom_shpp->si_flags &= ~DT_COMPLEX;
	spcdom_shpp->si_prec &= ~COMPLEX_PREC_BITS;
}

void setup_xform_shape(Shape_Info *xf_shpp,Shape_Info *im_shpp)
{
	*xf_shpp = *im_shpp;
	xf_shpp->si_cols /= 2;
	xf_shpp->si_cols += 1;
	xf_shpp->si_mach_dim[0] = 2;
	/* BUG?  should we make sure that the known shape is real? */
	xf_shpp->si_flags |= DT_COMPLEX;
	xf_shpp->si_prec |= COMPLEX_PREC_BITS;
}


/* Reshape this object.  for now, we don't worry about preserving any data... */

Data_Obj * reshape_obj(QSP_ARG_DECL  Data_Obj *dp,Shape_Info *shpp)
{
	Data_Obj *tmp_dp;
	char s[LLEN];

	/* the simplest thing is just to blow away the object and make a new one */
	/* Are we sure that we want to use a local (volatile) object here???  I don't think so... */
	/* we use localname to get a unique name in the short term, but we do NOT want
	 * to use make_local_dobj, because then it would be cleaned up at the end of the command...
	 */

	tmp_dp = make_dobj(QSP_ARG  localname(),&shpp->si_type_dimset,shpp->si_prec);
	strcpy(s,dp->dt_name);
	delvec(QSP_ARG  dp);
	obj_rename(QSP_ARG  tmp_dp,s);
	return(tmp_dp);
}


/* resolve_obj_id
 * 
 * resolve an object given its identifier
 */

static void resolve_obj_id(QSP_ARG_DECL  Identifier *idp, Shape_Info *shpp)
{
	Vec_Expr_Node *decl_enp;
	Shape_Info tmp_shape;
	Data_Obj *dp;
/* char remember_name[LLEN]; */

	/* decl_enp = dp->dt_extra; */

	/* matlab has no declarations!? */
	if( mode_is_matlab ){
/*
sprintf(error_string,"resolve_obj_id %s:  mode is matlab",idp->id_name);
advise(error_string);
*/
		dp = GET_OBJ(idp->id_name);

#ifdef CAUTIOUS
		if( dp == NO_OBJ ){
			sprintf(error_string,"resolve_obj_id %s:  missing object",idp->id_name);
			ERROR1(error_string);
		}
#endif /* CAUTIOUS */

		idp->id_refp->ref_dp = reshape_obj(QSP_ARG  dp,shpp);

		return;
	}

#ifdef CAUTIOUS
	if( idp->id_type != ID_REFERENCE ){
		sprintf(error_string,"CAUTIOUS:  resolve_obj_id:  identifier %s is not a reference!?",idp->id_name);
		ERROR1(error_string);
	}
#endif /* CAUTIOUS */

	decl_enp = idp->id_refp->ref_decl_enp;
	dp = idp->id_refp->ref_dp;

#ifdef CAUTIOUS
	if( dp->dt_extra != decl_enp ){
		sprintf(error_string,"CAUTIOUS:  resolve_obj_id:  object %s decl_enp (0x%lx) does not match 0x%lx!?",
			dp->dt_name,(u_long)dp->dt_extra,(u_long)decl_enp);
		ERROR1(error_string);
	}
#endif /* CAUTIOUS */

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resolve_obj_id:  setting decl node shape for %s", decl_enp->en_string);
advise(error_string);
if( shpp == NO_SHAPE ) WARN("shape ptr is null!?");
describe_shape(shpp);
}
#endif /* DEBUG */
	tmp_shape = *shpp;
	tmp_shape.si_prec=decl_enp->en_prec;

	/* used to propagate complex bit here */
	if( COMPLEX_PRECISION(tmp_shape.si_prec) )
		tmp_shape.si_flags |= DT_COMPLEX;

	COPY_NODE_SHAPE(decl_enp,&tmp_shape);
	/* make sure to do this, needed for runtime! */
	/* need to copy NEEDS_CALLTIME_RES flag too!? */
	decl_enp->en_flags |= resolution_flags;

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resolve_obj_id:  resolution_flags = 0x%x",resolution_flags);
advise(error_string);

sprintf(error_string,"resolve_obj_id:  calling reeval_decl_stat %s", decl_enp->en_string);
advise(error_string);
if( shpp == NO_SHAPE ) WARN("shape ptr is null!?");
describe_shape(shpp);
}
#endif /* DEBUG */

	/* the pointer points to the object, but the object's name may be in a different
	 * context.  We need to reinstate the context before reevaluating the declaration.
	 * We do this by saving the identifier context in the declaration node.
	 */
/*
strcpy(remember_name,dp->dt_name);
sprintf(error_string,"resolve_obj_id %s, old dp = 0x%lx",dp->dt_name,(u_long)dp);
advise(error_string);
longlist(dp);
*/
	reeval_decl_stat(QSP_ARG  dp->dt_prec,decl_enp,decl_enp->en_decl_flags);
/*
dp = idp->id_refp->ref_dp;
dp=dobj_of(remember_name);
if( dp == NO_OBJ ) {
sprintf(error_string,"resolve_obj_id:  missing object %s!?",remember_name);
ERROR1(error_string);
}
sprintf(error_string,"resolve_obj_id %s, new dp = 0x%lx",dp->dt_name,(u_long)dp);
advise(error_string);
longlist(dp);
*/

}

#ifdef FOOBAR

/* rc_match is not needed now that we have separate dimensions sets for type and machine elemets */

static int rc_match(Shape_Info *shpp1,Shape_Info *shpp2)
{
	int i;

	for(i=1;i<N_DIMENSIONS;i++)
		if( shpp1->si_dimension[i] != shpp2->si_dimension[i] ) return(0);

	if( COMPLEX_PRECISION(shpp1->si_prec) ){
		if( COMPLEX_PRECISION(shpp2->si_prec) ){
			if( shpp1->si_tdim != 2 || shpp2->si_tdim != 2 ) return(0);
		} else {
			if( shpp1->si_tdim != 2 || shpp2->si_tdim != 1 ) return(0);
		}
	} else {
		if( COMPLEX_PRECISION(shpp2->si_prec) ){
			if( shpp1->si_tdim != 1 || shpp2->si_tdim != 2 ) return(0);
		} else {
			if( shpp1->si_tdim != 1 || shpp2->si_tdim != 1 ) return(0);
		}
	}
	return(1);
}
#endif /* FOOBAR */

/* there are two reasons to call resolve_pointer:
 * First, if it points to an unknown target, we want to resolve the target.
 * Secondly, if it points to a known target, but the node itself (and hence
 * siblings etc) is unknown.
 */

void resolve_pointer(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Shape_Info *shpp)
{
	Identifier *idp;
	Data_Obj *dp;

	/* For a pointer, we have to make sure that the pointer has been
	 * set...  if it has, and if it points to an unknown size object,
	 * then we propagate the shape to the object.  If the pointer is not
	 * set, or if it points to an object that is already set to a
	 * different size, it is an error.
	 *
	 * We DO install a shape on the pointer node itself.
	 * 
	 * If we resolve an object here, we would like to propagate this,
	 * but it is not obvious how the node gets linked?  FIXME
	 *
	 * We also have a problem propagating from arg values, because
	 * typically the arg values have not been set at calltime...
	 */

	/* idp = EVAL_PTR_REF(uk_enp,1); */
	idp = GET_SET_PTR(uk_enp);

	if( idp == NO_IDENTIFIER ){
/*
NODE_ERROR(uk_enp);
sprintf(error_string,"resolve_pointer:  no set identifier for %s",node_desc(uk_enp));
advise(error_string);
*/
		/* this probably means that the ptr is not set, needs to be resolved
		 * at runtime.
		 */
		return;
	}
	/* At one time we did a CAUTIOUS check to see if the object was in the database,
	 * but that could fail because the pointed to object might be out of scope
	 * (e.g.  ptr to local object passed as subrt arg)
	 */

	dp = idp->id_ptrp->ptr_refp->ref_dp;

	/* BUG check ptr set first */

#ifdef CAUTIOUS
	if( dp == NO_OBJ ){
		advise("CAUTIOUS:  resolve_pointer:  ptr has no object!?");
		return;
	}
#endif /* CAUTIOUS */

	if( UNKNOWN_SHAPE(&dp->dt_shape) ){
		/* Now we know the object in question has unknown shape -
		 * Resolve it!
		 */
		/* now we try to resolve the pointed-to object */
sprintf(error_string,"resolve_pointer %s:  resolving pointer target object",node_desc(uk_enp));
advise(error_string);
		RESOLVE_OBJ_ID(idp->id_ptrp->ptr_refp->ref_idp,shpp);
	} else {
		/* make sure that the shapes are the same */
		if( ! same_shape(&dp->dt_shape,shpp) ){
			NODE_ERROR(uk_enp);
			sprintf(error_string,
		"resolve_pointer:  Shape mismatch with object %s",dp->dt_name);
			WARN(error_string);
describe_shape(uk_enp->en_shpp);
describe_shape(shpp);
		}
	}

/*
sprintf(error_string,"resolve_pointer fixing %s",node_desc(uk_enp));
advise(error_string);
*/

	/* also fix the node where we found this */
	point_node_shape(QSP_ARG  uk_enp,shpp);
}



/* We call resolve_object with a T_DYN_OBJ node.
 * The shape pointer should point to a valid shape...
 */

void resolve_object(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Shape_Info *shpp)
{
	Identifier *idp;
	Data_Obj *dp;

	/* If this is an automatic object, and we haven't evaluated the decls yet... */
	idp = GET_ID(uk_enp->en_string);
#ifdef CAUTIOUS
	if( idp == NO_IDENTIFIER ){
		sprintf(error_string,
	"CAUTIOUS:  resolve_object:  missing identifier %s!?",uk_enp->en_string);
		WARN(error_string);
		return;
	}
	if( ! IS_REFERENCE(idp) ){
		sprintf(error_string,
	"CAUTIOUS:  resolve_object:  identifier %s is not an object reference!?",idp->id_name);
		WARN(error_string);
		return;
	}
#endif /* CAUTIOUS */

	dp = idp->id_refp->ref_dp;

#ifdef CAUTIOUS
	if( dp==NO_OBJ ){
		sprintf(error_string,
	"CAUTIOUS:  resolve_object:  missing object %s!?",uk_enp->en_string);
		WARN(error_string);
list_dobjs(SINGLE_QSP_ARG);
		return;
	}
#endif /* CAUTIOUS */

	uk_enp->en_flags |= resolution_flags;		/* resolve_object */

	if( ! UNKNOWN_SHAPE(&dp->dt_shape) ){
		/* This can occur as a matter of course if we have multiple T_DYN_OBJ
		 * nodes, after the first is resolved the shape is propagated to the
		 * declaration node.  The other instances don't get fixed right away...
		 * (when do they get fixed???)
		 * BUT wait:  shouldn't the nodes all point to the declaration shape???
		 */

		return;
	}

	RESOLVE_OBJ_ID(idp,shpp);

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resolve_object %s:",node_desc(uk_enp));
advise(error_string);
describe_shape(uk_enp->en_shpp);
}
#endif /* DEBUG */

} /* end resolve_object */


/* here we pass a shape with which to resolve the children (scalar) nodes */

static void resolve_obj_list(QSP_ARG_DECL  Vec_Expr_Node *enp, Shape_Info *shpp)
{
	switch(enp->en_code){
		case T_RET_LIST:
			RESOLVE_OBJ_LIST(enp->en_child[0],shpp);
			RESOLVE_OBJ_LIST(enp->en_child[1],shpp);
			break;

		case T_DYN_OBJ:
			RESOLVE_NODE(enp,shpp);
			break;

		default:
			MISSING_CASE(enp,"resolve_obj_list");
			break;
	}
}



/* resolve_node
 *
 * This is where we put most of the cases where it does not really depend whether the
 * node to be resolved is parent or child.  Special cases are put in resolve_parent()
 * This stuff mostly came from resolve_unkown_child, with stuff from resolve_uknown_parent
 * grafted in...
 *
 * If we can continue propagating, we return the unknown node pointer, otherwise we
 * return NULL.
 */

Vec_Expr_Node *resolve_node(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Shape_Info *shpp)
{
	uk_enp->en_flags |= resolution_flags;		/* resolve_node */

	switch(uk_enp->en_code){
		/* projection operator cases */

		/* For most cases, we just install the shape */
		case T_DEREFERENCE:
		/*
		ALL_BINOP_CASES
		*/
		case T_VS_FUNC:
		ALL_UNARY_CASES
		ALL_MATHFN_CASES
		/*
		ALL_MATH_VFN_CASES
		*/
		case T_FIX_SIZE:
		case T_DFT:
		case T_LOAD:
		case T_RAMP:
		/*
		case T_FIX_SIZE:
		*/
		case T_ASSIGN:
		case T_RETURN:			/* resolve_node */
			point_node_shape(QSP_ARG  uk_enp,shpp);
			return(uk_enp);


		/* These nodes own their own shape */

		/* These nodes have a different shape than their children...
		 * We assume this has been accounted for by the caller
		 */

		case T_CALLFUNC:
		case T_SUBSAMP:
		case T_CSUBSAMP:
		case T_SUBVEC:			/* resolve_node */
		case T_CSUBVEC:
		case T_IMG_DECL:
		case T_RIDFT:
		case T_RDFT:
		case T_ENLARGE:
		case T_REDUCE:
		case T_MAXVAL:
		case T_MINVAL:
		case T_SUM:
		case T_TRANSPOSE:
		case T_CURLY_SUBSCR:
		case T_SQUARE_SUBSCR:
		case T_REAL_PART:
		case T_IMAG_PART:
		case T_VV_FUNC:		/* resolve_node */
			COPY_NODE_SHAPE(uk_enp,shpp);
			return(uk_enp);

		BINARY_BOOLOP_CASES			/* T_BOOL_AND etc, bitmap shapes */
		case T_BOOL_NOT:			/* does this belong here??? */
		ALL_NUMERIC_COMPARISON_CASES		/* T_BOOL_GT etc, bitmap shapes */
			/* for numeric comparisons, we know that the known child node can't be a bitmap */
			COPY_NODE_SHAPE(uk_enp,shpp);
			if( ! BITMAP_SHAPE(shpp) ){
				xform_to_bitmap(uk_enp->en_shpp);
			}
			break;


		case T_TYPECAST:
			COPY_NODE_SHAPE(uk_enp,shpp);
			uk_enp->en_shpp->si_prec = uk_enp->en_intval;
			return(uk_enp);

		case T_POINTER:			/* resolve_node */
			RESOLVE_POINTER(uk_enp,shpp);

			/* We can't propagate from the pointer node... */
			return(NO_VEXPR_NODE);

		/*
		case T_STR_PTR:
		case T_POINTER:
		*/
		case T_DYN_OBJ:			/* resolve_node */
			RESOLVE_OBJECT(uk_enp,shpp);
			return(uk_enp);

		case T_RET_LIST:		/* a list of objects */
			if( ROWVEC_SHAPE(shpp) ){	/* a list of scalars? */
				RESOLVE_OBJ_LIST(uk_enp,scalar_shape(shpp->si_prec));
			} else {
sprintf(error_string,"resolve_node %s:  oops, only know how to resolve from rowvec!?",node_desc(uk_enp));
WARN(error_string);
			}
			return(uk_enp);

		case T_SS_B_CONDASS:
			/* The known child is the bitmap */
			COPY_NODE_SHAPE(uk_enp,shpp);
			return(uk_enp);

		case T_VS_B_CONDASS:
			/* The known child may be the bitmap, but hopefully the
			 * correct shape was determined above.
			 */
			COPY_NODE_SHAPE(uk_enp,shpp);
			return(uk_enp);


		default:
			MISSING_CASE(uk_enp,"resolve_node");
			break;
	}
	return(NO_VEXPR_NODE);
} /* end resolve_node */

/* The parent is the unknown node? */

static Vec_Expr_Node *resolve_parent(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Shape_Info *shpp)
{
	prec_t prec;

	switch(uk_enp->en_code){

		case T_CURLY_SUBSCR:
			*uk_enp->en_shpp = *shpp;	/* copy shape */
			/* set a dimension to one */
			uk_enp->en_shpp->si_mach_dim[shpp->si_mindim] = 1;
			uk_enp->en_shpp->si_type_dim[shpp->si_mindim] = 1;
			uk_enp->en_shpp->si_mindim++;
			return(uk_enp);

			
		case T_TRANSPOSE:
			*uk_enp->en_shpp = *shpp;	/* copy shape */
			uk_enp->en_shpp->si_cols = shpp->si_rows;
			uk_enp->en_shpp->si_rows = shpp->si_cols;
			/* BUG?  reset shape flags? */
			return(uk_enp);

		case T_RIDFT:		/* resolve_parent, xform shape is known, get spc. dom. size */
			setup_spcdom_shape(uk_enp->en_shpp,shpp);
			return(uk_enp);

		case T_RDFT:		/* resolve_parent, input shape is known, get xform size */
			setup_xform_shape(uk_enp->en_shpp,shpp);
			return(uk_enp);

		ALL_NUMERIC_COMPARISON_CASES
			/* for numeric comparisons, we know that the known child node can't be a bitmap */
			COPY_NODE_SHAPE(uk_enp,shpp);
			xform_to_bitmap(uk_enp->en_shpp);
			return(uk_enp);

		case T_SS_B_CONDASS:	/* resolve_parent, branch shape is known */
		case T_VS_B_CONDASS:	/* resolve_parent, branch shape is known */
		case T_VV_B_CONDASS:	/* resolve_parent, branch shape is known */
			prec = uk_enp->en_prec;
			COPY_NODE_SHAPE(uk_enp,shpp);
			if( BITMAP_SHAPE(shpp) )
				xform_from_bitmap(uk_enp->en_shpp,prec);
			return(uk_enp);

		case T_VS_FUNC:
		ALL_UNARY_CASES
		ALL_MATHFN_CASES
		case T_FIX_SIZE:
		case T_DEREFERENCE:
		case T_ASSIGN:
		case T_TYPECAST:
		case T_RETURN:
			return( RESOLVE_NODE(uk_enp,shpp) );

		case T_VV_FUNC:
		/* Now with outer ops being handled by vv_func, we need to know
		 * the shape of both children before we can resolve.
		 */
		if( (!UNKNOWN_SHAPE(uk_enp->en_child[0]->en_shpp)) &&
			! UNKNOWN_SHAPE(uk_enp->en_child[1]->en_shpp) ){
			Shape_Info *shpp;

			shpp = calc_outer_shape(uk_enp->en_child[0],uk_enp->en_child[1]);
			COPY_NODE_SHAPE(uk_enp,shpp);
			return(uk_enp);
		}
		break;

		case T_BOOL_NOT:
			return( RESOLVE_NODE(uk_enp,shpp) );
			break;

		default:
			MISSING_CASE(uk_enp,"resolve_parent");
			return( RESOLVE_NODE(uk_enp,shpp) );
	}
	return(NO_VEXPR_NODE);
} /* resolve_parent */

/* What is the difference between resolve_unknown_parent and resolve_parent???
 */

static Vec_Expr_Node * resolve_unknown_parent(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp)
{
	Shape_Info *shpp;

	/* First get the target shape - this is easy!? */
	switch(enp->en_code){
		case T_FIX_SIZE:
		case T_POINTER:
		case T_TYPECAST:
		case T_RAMP:
		case T_RDFT:
		case T_RIDFT:
		case T_CALLFUNC:
		ALL_OBJREF_CASES
		ALL_UNARY_CASES
		/*
		ALL_BINOP_CASES
		*/
		case T_VV_FUNC:
		case T_VS_FUNC:
		ALL_MATHFN_CASES
		ALL_CONDASS_CASES
		case T_TRANSPOSE:
			shpp = enp->en_shpp;
			break;

		ALL_NUMERIC_COMPARISON_CASES
			/* This is a bitmap shape! */
			shpp = enp->en_shpp;
			break;

		default:
			MISSING_CASE(enp,"resolve_unknown_parent (known node)");
			shpp = enp->en_shpp;
			break;
	}

#ifdef DEBUG
if( debug & resolve_debug ){
	/* We usually thing of resolution as applying to images etc, and not scalars,
	 * but when we have multiple scalar arguments to return from a function, we may
	 * want to do it with ptr args whose targets are assigned...  When these pointers
	 * are resolved to their scalar targets, this msg will be printed...
	 */
	if( SCALAR_SHAPE(shpp) ){
		sprintf(error_string,"ADVISORY:  resolve_unknown_parent %s from %s:  known shape is scalar!?",
			node_desc(uk_enp),node_desc(enp));
		prt_msg(error_string);
		describe_shape(shpp);
	}
}
#endif /* DEBUG */

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resolve_unknown_parent %s from %s",node_desc(uk_enp),node_desc(enp));
advise(error_string);
}
#endif /* DEBUG */

	/* BUG we'd like to merge this with resolve_node... */

	return( RESOLVE_PARENT(uk_enp,shpp));
} /* end resolve_unknown_parent */


/* shape_for_child - what does this do???
 */

static Shape_Info *shape_for_child(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp)
{
	Shape_Info *shpp;

	switch(enp->en_code){

		BINARY_BOOLOP_CASES			/* shape_for_child (known node) */
			if( BITMAP_SHAPE(uk_enp->en_shpp) ) return(enp->en_shpp);
			else {
				*enp->en_bm_child_shpp = *enp->en_shpp;
				xform_from_bitmap(enp->en_bm_child_shpp,uk_enp->en_prec);
				return(enp->en_bm_child_shpp);
			}

		ALL_NUMERIC_COMPARISON_CASES		/* T_BOOL_GT etc */
			*enp->en_bm_child_shpp = *enp->en_shpp;
			xform_from_bitmap(enp->en_bm_child_shpp,uk_enp->en_prec);
			return(enp->en_bm_child_shpp);

		ALL_CONDASS_CASES
			if( uk_enp == enp->en_child[0] ){
				*enp->en_bm_child_shpp = *enp->en_shpp;
				xform_to_bitmap(enp->en_bm_child_shpp);
				return(enp->en_bm_child_shpp);
			}
			/* fall-thru */
		/*
		ALL_BINOP_CASES
		*/
		case T_VV_FUNC:
		case T_VS_FUNC:
		ALL_MATHFN_CASES
		ALL_UNARY_CASES
		case T_ASSIGN:
		case T_DEREFERENCE:
		case T_DFT:
		case T_RETURN:			/* shape_for_child */

			return( enp->en_shpp );

		case T_RIDFT:	/* resolve_unkown_child, RIDFT output shape is known, get xform shape */
			/* We know the shape of the real image, but we want
			 * to propagate a reduced shape to the child xform.
			 */

			/* We need to allocate the shape because the downstream nodes
			 * are going to want to point to it...  But for repeated subroutine
			 * calls this is a big memory leak...  We need to have some sort
			 * of garbage collection! FIXME
			 *
			 * A possible approach would be to have these nodes own
			 * an extra shape structure...
			 */

			setup_xform_shape(enp->en_child_shpp,enp->en_shpp);
			return(enp->en_child_shpp);

		case T_RDFT:		/* shape_for_child, RDFT xform shape is known, get input shape */
			setup_spcdom_shape(enp->en_child_shpp,enp->en_shpp);
			return(enp->en_child_shpp);

		case T_ENLARGE:
			shpp = enp->en_child_shpp;
			*shpp = *enp->en_shpp;
			shpp->si_cols /= 2;
			shpp->si_rows /= 2;
			return(shpp);

		case T_REDUCE:
			shpp = enp->en_child_shpp;
			*shpp = *enp->en_shpp;
			shpp->si_cols *= 2;
			shpp->si_rows *= 2;
			return(shpp);

		case T_TRANSPOSE:
			shpp = enp->en_child_shpp;
			*shpp = *enp->en_shpp;
			shpp->si_cols = enp->en_shpp->si_rows;
			shpp->si_rows = enp->en_shpp->si_cols;
			return(shpp);

		case T_TYPECAST:
			/* for a typecast node, the child needs to keep the original precision.
			 * We create a new shape, potentially a garbage collection problem?
			 */
			shpp = (Shape_Info *)getbuf(sizeof(*shpp));
			*shpp = *enp->en_shpp;
			shpp->si_prec = uk_enp->en_shpp->si_prec;
			return(shpp);

		default:
			MISSING_CASE(enp,"shape_for_child");
			return( enp->en_shpp );
	}
} /* end shape_for_child */

static Vec_Expr_Node * resolve_unknown_child(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp)
{
	Shape_Info *shpp;

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"resolve_unknown_child %s from %s",node_desc(uk_enp),node_desc(enp));
advise(error_string);
}
#endif /* DEBUG */

	shpp = shape_for_child(QSP_ARG  uk_enp,enp);

	if( mode_is_matlab && uk_enp->en_code == T_RET_LIST ){
		Data_Obj *dp;

		dp = EVAL_OBJ_REF(uk_enp);	/* force object creation */
	}

	/* Now shpp point to the shape we will use for resolution */

#ifdef DEBUG
if( debug & resolve_debug ){
	/* We usually thing of resolution as applying to images etc, and not scalars,
	 * but when we have multiple scalar arguments to return from a function, we may
	 * want to do it with ptr args whose targets are assigned...  When these pointers
	 * are resolved to their scalar targets, this msg will be printed...
	 */
	if( SCALAR_SHAPE(shpp) ){
		sprintf(error_string,"ADVISORY:  resolve_unknown_child %s from %s:  known shape is scalar!?",
			node_desc(uk_enp),node_desc(enp));
		prt_msg(error_string);
		describe_shape(shpp);
	}
}
#endif /* DEBUG */

	return(RESOLVE_NODE(uk_enp,shpp));
} /* resolve_unknown_child */

/* resolve_return
 *
 * We call this when we resolve a return node from the calling CALLFUNC node - these
 * are not linked, although the program would be simpler if the could be !? (BUG?)
 */

void resolve_return(QSP_ARG_DECL  Vec_Expr_Node *enp,Shape_Info *shpp)
{
	point_node_shape(QSP_ARG  enp,shpp);
}

/* effect_resolution
 *
 * The main entry point to this module.
 * We have a pair of nodes which can resolve each other, and one of
 * them is now known.
 * Returns a ptr to the resolved node, or NO_VEXPR_NODE
 */

Vec_Expr_Node * effect_resolution(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp)
{
	Vec_Expr_Node *ret_enp=NULL;

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"effect_resolution of %s from %s",node_desc(uk_enp),node_desc(enp));
advise(error_string);
advise("known shape:");
describe_shape(enp->en_shpp);
}
#endif /* DEBUG */

	/* BUG?  do we need to save the old value of the flags? */
	if( enp->en_flags & NODE_NEEDS_CALLTIME_RES )
		resolution_flags |= NODE_NEEDS_CALLTIME_RES;

	/* We know that one of these nodes is the parent of the
	 * other - which one is which ?
	 */
	
	if( enp->en_parent == uk_enp ) ret_enp=RESOLVE_UNKNOWN_PARENT(uk_enp,enp);
	else if( uk_enp->en_parent == enp ) ret_enp=RESOLVE_UNKNOWN_CHILD(uk_enp,enp);
	else {
		RESOLVE_NODE(uk_enp,enp->en_shpp);
	}


#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"After effect_resolution of %s from %s:",node_desc(uk_enp),node_desc(enp));
advise(error_string);
advise("known shape:");
describe_shape(enp->en_shpp);
advise("resolved shape:");
describe_shape(uk_enp->en_shpp);
}
#endif /* DEBUG */

	return(ret_enp);
} /* effect_resolution */

#ifdef FOOBAR

/* Return true if all dimensions (except tdim) match.
 * This says that they are the same shape, but may be different types...
 *
 * Not needed now that we have type_dim.
 */

static int same_outline(Shape_Info *shpp1,Shape_Info *shpp2)
{
	int i;

	for(i=1;i<N_DIMENSIONS;i++)
		if( shpp1->si_dimension[i] != shpp2->si_dimension[i] ) return(0);
	return(1);
}

#endif /* FOOBAR */


/* Scan the arg values...  if any have unknown shapes,
 * see if the corresponding subrt decl now has known shape.
 * If so, propagate it back.
 *
 * What is the context here???
 */

void resolve_argval_shapes(QSP_ARG_DECL  Vec_Expr_Node *val_enp,Vec_Expr_Node *decl_enp,Subrt *srp)
{
	if( decl_enp == NO_VEXPR_NODE ) return;

	switch(decl_enp->en_code){
		case T_DECL_STAT_LIST:
			RESOLVE_ARGVAL_SHAPES(val_enp->en_child[0],
				decl_enp->en_child[0],srp);
			RESOLVE_ARGVAL_SHAPES(val_enp->en_child[1],
				decl_enp->en_child[1],srp);
			break;
		case T_DECL_STAT:
			RESOLVE_ARGVAL_SHAPES(val_enp,
				decl_enp->en_child[0],srp);
			break;

		case T_PTR_DECL:
		case T_SEQ_DECL:
		case T_IMG_DECL:
		case T_VEC_DECL:
		case T_SCAL_DECL:
		case T_CSEQ_DECL:
		case T_CIMG_DECL:
		case T_CVEC_DECL:
		case T_CSCAL_DECL:
		case T_FUNCPTR_DECL:
#ifdef CAUTIOUS
			if( val_enp->en_shpp == NO_SHAPE ){
				sprintf(error_string,"CAUTIOUS:  resolve_argval_shapes %s:  node has no shape!?",
					node_desc(val_enp));
				WARN(error_string);
				return;
			}
#endif /* CAUTIOUS */
			/* See if the value has unknown shape */
			if( ! UNKNOWN_SHAPE(val_enp->en_shpp) )
				return;

			/* value has unknown shape, see if decl is known */
			if( UNKNOWN_SHAPE( decl_enp->en_shpp ) )
				return;

			/* value is unknown, decl is known:  propagate! */

#ifdef DEBUG
if( debug & resolve_debug ){
advise("resolve_argval_shapes:  propagating a shape!!");
describe_shape(decl_enp->en_shpp);
}
#endif /* DEBUG */
			PROPAGATE_SHAPE(val_enp,decl_enp->en_shpp);
#ifdef DEBUG
if( debug & resolve_debug ){
advise("resolve_argval_shapes:  DONE");
describe_shape(val_enp->en_shpp);
}
#endif /* DEBUG */
			break;
			
			
		default:
			MISSING_CASE(decl_enp,"resolve_argval_shapes");
			break;
	}
}

/* propagate_shape
 *
 * Descend a tree, assigning shapes when possible.
 */

void propagate_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,Shape_Info *shpp)
{
	Shape_Info tmp_shape;

#ifdef CAUTIOUS
	if( ! UNKNOWN_SHAPE(enp->en_shpp) ){
		if( ! same_shape(enp->en_shpp,shpp) ){
			NODE_ERROR(enp);
	WARN("CAUTIOUS:  propagate_shape:  shapes don't match!?");
			describe_shape(enp->en_shpp);
			describe_shape(shpp);
DUMP_TREE(enp);
		}
		sprintf(error_string,"CAUTIOUS:  propagate_shape %s already has a shape!?",node_desc(enp));
		WARN(error_string);
		return;
	}
#endif /* CAUTIOUS */

	switch(enp->en_code){
		case T_POINTER:		/* propagate_shape */
			RESOLVE_POINTER(enp,shpp);
			break;

		case T_STR_PTR:
		case T_DYN_OBJ:
			RESOLVE_OBJECT(enp,shpp);
			break;

		case T_CURLY_SUBSCR:
			tmp_shape = *shpp;
			tmp_shape.si_type_dim[tmp_shape.si_mindim]=1;
			tmp_shape.si_mach_dim[tmp_shape.si_mindim]=1;
			tmp_shape.si_mindim++;
			PROPAGATE_SHAPE(enp->en_child[0],&tmp_shape);
			break;

		case T_SQUARE_SUBSCR:
			tmp_shape = *shpp;
			tmp_shape.si_type_dim[tmp_shape.si_maxdim]=1;
			tmp_shape.si_mach_dim[tmp_shape.si_maxdim]=1;
			tmp_shape.si_maxdim--;
			PROPAGATE_SHAPE(enp->en_child[0],&tmp_shape);
			break;

		case T_VV_FUNC:
			PROPAGATE_SHAPE(enp->en_child[0],shpp);
			PROPAGATE_SHAPE(enp->en_child[1],shpp);
			break;

		default:
			MISSING_CASE(enp,"propagate_shape");
			break;
	}
}

/* If we're sure we really don't need this code, remove it! */

#ifdef UNUSED

/* Call with a subrt argument list...
 * return 1 if all have known sizes, 0 if any UNKNOWN
 */

static int arg_sizes_known(Vec_Expr_Node *enp)
{
	int stat;

	if( enp==NO_VEXPR_NODE ) return(1);

	switch(enp->en_code){
		case T_LIT_INT:
		case T_LIT_DBL:
			return(1);

		case T_ARGLIST:
			stat=arg_sizes_known(enp->en_child[0]);
			if( ! stat ) return(stat);
			stat=arg_sizes_known(enp->en_child[1]);
			return(stat);

		case T_DYN_OBJ:
			if( UNKNOWN_SHAPE(enp->en_shpp) )
				return(0);
			else
				return(1);
		default:
			MISSING_CASE(enp,"arg_sizes_known");
			break;
	}
	return(0);
}
#endif /* UNUSED */




/* For an object w/ unknown shape, scan it's list of dependencies.
 * If an object is encountered that now has known shape,
 * copy the shape over.
 *
 * This may be a bit tricky, because not all of the references
 * may be to currently exisiting objects:  they may be other local
 * variables that have not yet been created...  On the other hand,
 * the arguments probaly will already exist at runtime.  So
 * we need to check both the existing objects, and a separate
 * database of now-known objects.  There are likely to be problems
 * with scope conflicts, i.e. what should happen if there is a global
 * variable that creates a name conflict with the unknown list?
 * Probably the now-known list should be checked first - or
 * the unknown list?  This could be a real mess to make it work correctly.
 * Maybe the solution is to create ALL the unknown objects...
 * and then resize them.
 *
 * In an ideal world, we have things like:
 *
 * a=func(b);
 * return(a);
 *
 * The first line generates a dependency for a with element b...
 * The second line puts a on the retobj list...
 * So first, the shape of a will be set when the return dest is known.
 * Then we should set the shape of b from a...
 *
 * The situation is somewhat different when the shapes are determined
 * from arguments, but we will figure that out later...
 */


/*
 * Compare the value list with the declaration list...
 * If all shapes are known, make sure they match.
 * If the decl shapes are unknown, then install the shape
 * from the argument.
 *
 * We need to remember to forget the shape of the arg decls when
 * we exit this call...
 *
 * We could also use this to resolve the shape of an unknown arg val...
 * (Not sure WHY we would need to do this, because to pass by value,
 * we would want to have some data in there already!?)
 *
 * We assume that the counts have already been insured to match.
 *
 * We also install shapes for pointer nodes.
 *
 * What is the context when we call check_arg_shapes???
 *
 * We call it (from resolve_subrt and setup_call) before setting the context of the new
 * subroutine.
 */

int check_arg_shapes(QSP_ARG_DECL  Vec_Expr_Node *arg_enp,Vec_Expr_Node *val_enp,Subrt *srp)
{
	int stat;

	if( arg_enp == NO_VEXPR_NODE ) return(0);
	else if( val_enp == NO_VEXPR_NODE ){
		/* BUG we want to report this error at the line number of the callfunc node */
		sprintf(error_string,"Subroutine %s requires arguments",srp->sr_name);
		WARN(error_string);
		return(-1);
	}

	switch(arg_enp->en_code){
		case  T_DECL_STAT:
			/* en_intval is the type (float,short,etc) */
			stat=CHECK_ARG_SHAPES(arg_enp->en_child[0],
						val_enp,srp);
			return(stat);

		case T_DECL_STAT_LIST:
			/* val_enp should be T_ARGLIST */
#ifdef CAUTIOUS
			if( val_enp->en_code != T_ARGLIST ){
				sprintf(error_string,"CAUTIOUS:  expected T_ARGLIST in args for subrt %s, code is %s",
					srp->sr_name,NNAME(val_enp));
				WARN(error_string);
					
			}
#endif /* CAUTIOUS */

			stat=CHECK_ARG_SHAPES(arg_enp->en_child[0],
				val_enp->en_child[0],srp);
			if( stat < 0 ) return(stat);
			stat=CHECK_ARG_SHAPES(arg_enp->en_child[1],
				val_enp->en_child[1],srp);
			return(stat);

		case T_PTR_DECL:		/* check_arg_shapes */
			/* we really don't care if
			 * the shapes match for a ptr arg, it just assumes the
			 * shape of the value...
			 */
			if( arg_enp->en_shpp != NO_SHAPE && UNKNOWN_SHAPE(arg_enp->en_shpp) ){
				if( ! UNKNOWN_SHAPE(val_enp->en_shpp) ){
/*
sprintf(error_string,"check_arg_shapes %s, %s:  copying shape for ptr",node_desc(arg_enp),node_desc(val_enp));
advise(error_string);
*/
					/*copy_decl_shape*/ COPY_NODE_SHAPE(arg_enp,
						val_enp->en_shpp);

				}
				/* it is ok if both are unknown... */
			}
			return(0);
			
		case T_FUNCPTR_DECL:
			return(0);

		case T_SCAL_DECL:
		case T_VEC_DECL:
		case T_IMG_DECL:
		case T_SEQ_DECL:
		case T_CSCAL_DECL:
		case T_CVEC_DECL:
		case T_CIMG_DECL:
		case T_CSEQ_DECL:
#ifdef CAUTIOUS
if( arg_enp->en_shpp == NO_SHAPE ){
sprintf(error_string,"CAUTIOUS:  check_arg_shapes:  subrt %s, arg %s has no shape!?",
srp->sr_name,arg_enp->en_string);
WARN(error_string);
return(-1);
}
if( val_enp->en_shpp == NO_SHAPE ){
sprintf(error_string,"CAUTIOUS:  check_arg_shapes:  subrt %s, arg val %s has no shape!?",
srp->sr_name,val_enp->en_string);
WARN(error_string);
return(-1);
}
#endif /* CAUTIOUS */
			if( UNKNOWN_SHAPE(arg_enp->en_shpp) ){
				if( ! UNKNOWN_SHAPE(val_enp->en_shpp) ){
					COPY_NODE_SHAPE(arg_enp, val_enp->en_shpp);
					/* Now we would like to propagate into the body of the subroutine!? */
					arg_enp->en_flags |= resolution_flags;
				}

				/* it's not necessarily an error for both
				 * to be unknown at this time...
				 */

			} else if( UNKNOWN_SHAPE(val_enp->en_shpp) ){
				/* here the arg decl has known shape, but the arg val is unknown;
				 * We'd like to propagate upwards...
				 */
				PROPAGATE_SHAPE(val_enp,arg_enp->en_shpp);
			} else {	/* everything is known */
				if( !shapes_match(arg_enp->en_shpp,
							val_enp->en_shpp) ){
					NODE_ERROR(srp->sr_call_enp);
					sprintf(error_string,
	"subrt %s:  argument shape mismatch",srp->sr_name);
					WARN(error_string);
advise("argument prototype shape:");
describe_shape(arg_enp->en_shpp);
advise("argument value shape:");
describe_shape(val_enp->en_shpp);
DUMP_TREE(arg_enp);

#ifdef DEBUG
if( debug & resolve_debug ){
advise(arg_enp->en_string);
describe_shape(arg_enp->en_shpp);
describe_shape(val_enp->en_shpp);
}
#endif /* DEBUG */
					return(-1);
				}
			}
			return(0);

		default:
			MISSING_CASE(arg_enp,"check_arg_shapes");
			return(-1);
	}
}

