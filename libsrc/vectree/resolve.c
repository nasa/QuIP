
#include "quip_config.h"

/* resolve.c
 *
 * Scanning the tree does a number of things:
 * It figures out the "shape" of each node, that is,
 * the dimensions of the object the node represents.
 * (Actually, that is done by prelim_node_shape and update_node_shape!?)
 *
 * Should nodes "own" their shape structs?
 * For Objects, we can't just point to OBJ_SHAPE(dp), because
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
 *
 * 5/17/13 - a bug discovered:
 *
 * void func(float u1[])
 * {
 * 	float u2[],s{2};
 *	s=1;
 *	u2=u1+s;
 * }
 *
 * float v[4]{2};
 * v=1;
 * func(v);
 *
 * u2 is resolved to match s, not v...  In this situation, we really need to use
 * the outer product shape...
 * 
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

#include "quip_prot.h"
#include "query_bits.h"	// LLEN - BUG
#include "data_obj.h"
#include "nexpr.h"
#include "veclib_api.h"
#include "vec_util.h"		/* dilate, erode */
//#include "img_file.h"
//#include "filetype.h"

#include "vectree.h"
#include "subrt.h"

#define MAX_HIDDEN_CONTEXTS	32

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

	if( enp == NULL ) return;

	if( VN_SHAPE(enp) != NULL && UNKNOWN_SHAPE(VN_SHAPE(enp)) && VN_PREC(enp) != PREC_VOID ){
		switch(VN_CODE(enp)){
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
//sprintf(ERROR_STRING,"find_uk_nodes:  adding %s",node_desc(enp));
//ADVISE(ERROR_STRING);
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
		if( VN_CHILD(enp,i) != NULL )
			find_uk_nodes(QSP_ARG  VN_CHILD(enp,i),lp);
}

static List *get_uk_list(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	List *lp;

	lp=new_list();
	find_uk_nodes(QSP_ARG  enp,lp);
	if( eltcount(lp) == 0 ){
		dellist(lp);
		return(NULL);
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
		//CLEAR_VN_FLAG_BITS(enp, NODE_NEEDS_CALLTIME_RES);
		assert( VN_SHAPE(enp) != NULL );

		if( VN_CODE(enp) == T_DYN_OBJ ){
			/* OBJECT nodes point to the shape at the decl node,
			 * so we leave it alone.
			 * BUT matlab has no decl nodes, so we'd better do something different?
			 */
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"forget_resolved_tree %s:  leaving OBJECT node alone",node_desc(enp));
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

		} else {
			if( OWNS_SHAPE(enp) )
				copy_node_shape(enp,uk_shape(VN_PREC(enp)));
			else
				point_node_shape(QSP_ARG  enp,uk_shape(VN_PREC(enp)));
		}
	}

	for(i=0;i<MAX_CHILDREN(enp);i++)
		if( VN_CHILD(enp,i) != NULL )
			forget_resolved_tree(QSP_ARG  VN_CHILD(enp,i));
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

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"forget_resolve_shapes %s:",SR_NAME(srp));
ADVISE(ERROR_STRING);
DUMP_TREE(SR_BODY(srp));
}
#endif /* QUIP_DEBUG */

	if( SR_ARG_DECLS(srp) != NULL )
		forget_resolved_tree(QSP_ARG  SR_ARG_DECLS(srp));
	if( SR_BODY(srp) != NULL )
		forget_resolved_tree(QSP_ARG  SR_BODY(srp));

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"forget_resolve_shapes %s DONE",SR_NAME(srp));
ADVISE(ERROR_STRING);
DUMP_TREE(SR_BODY(srp));
}
#endif /* QUIP_DEBUG */

}


/* We call this with a list of nodes which may need resolution.
 * This can be called with a list we construct from a tree, or
 * with a node's list (propagation), so we therefore cannot assume
 * that all the nodes on the list actually need resolution.
 */

static void resolve_uk_nodes(QSP_ARG_DECL  List *lp)
{
	Node *np;

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		Vec_Expr_Node *enp;

		enp=(Vec_Expr_Node *)NODE_DATA(np);
		if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_uk_nodes trying to fix %s",node_desc(enp));
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			/*
			resolve_one_uk_node(enp);
			*/
			RESOLVE_TREE(enp,NULL);
		}

		/* If we remove nodes from the list as they are resolved,
		 * We can't do this next business...
		 */
		np=NODE_NEXT(np);
	}
}

/* late_calltime_resolve
 *
 * called after subroutine context has been set, and the arg values have been set.
 */

void late_calltime_resolve(QSP_ARG_DECL  Subrt *srp, Data_Obj *dst_dp)
{
	List *lp;
	uint32_t save_res;

	assert( curr_srp == srp );

/*
sprintf(ERROR_STRING,"late_calltime_resolve %s setting SCANNING flag",SR_NAME(srp));
ADVISE(ERROR_STRING);
*/
	SET_SR_FLAG_BITS(srp, SR_SCANNING);
	save_res = resolution_flags;
	resolution_flags = NODE_NEEDS_CALLTIME_RES;

	/*
	if( SR_PREC(srp) != PREC_VOID ){
		if( ret_shpp != NULL && ! UNKNOWN_SHAPE(ret_shpp) ){
			SET_SR_SHAPE(srp, ret_shpp);
		} else {
			SET_SR_SHAPE(srp, uk_shape(SR_PREC(srp)));
		}
	}
	*/
	/* assert( SR_SHAPE(srp) == NULL ); */

#ifdef QUIP_DEBUG
//if( debug & resolve_debug ){
sprintf(ERROR_STRING,"late_calltime_resolve %s begin:",SR_NAME(srp));
ADVISE(ERROR_STRING);
DUMP_SUBRT(srp);
//}
#endif /* QUIP_DEBUG */

	lp = get_uk_list(QSP_ARG  SR_BODY(srp));
	if( lp != NULL ){
		RESOLVE_UK_NODES(lp);

#ifdef QUIP_DEBUG
//if( debug & resolve_debug ){
sprintf(ERROR_STRING,"late_calltime_resolve %s done:",SR_NAME(srp));
ADVISE(ERROR_STRING);
DUMP_SUBRT(srp);
//}
#endif /* QUIP_DEBUG */

		dellist(lp);
	}

/*
sprintf(ERROR_STRING,"late_calltime_resolve %s clearing SCANNING flag",SR_NAME(srp));
ADVISE(ERROR_STRING);
*/
	CLEAR_SR_FLAG_BITS(srp, SR_SCANNING);

	resolution_flags = save_res;
}

static void find_known_return(QSP_ARG_DECL  Vec_Expr_Node *enp,Subrt *srp)
{
	int i;

	if( enp == NULL ) enp=SR_BODY(srp);

	switch(VN_CODE(enp)){
		case T_RETURN:
			assert( VN_SHAPE(enp) != NULL );

			if( ! UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
/*
sprintf(ERROR_STRING,"find_known_return: setting subrt %s shape from %s",SR_NAME(srp),node_desc(enp));
ADVISE(ERROR_STRING);
DESCRIBE_SHAPE(VN_SHAPE(enp));
*/
				SET_SR_SHAPE(srp, VN_SHAPE(enp));
			}
			return;

		/* for all of these cases we descend and keep looking */
		case T_STAT_LIST:
			break;

		case T_IFTHEN:
			/* child 0 is the test,and can't have a return statement */
			if( ! NULL_CHILD(enp,1) ) find_known_return(QSP_ARG  VN_CHILD(enp,1),srp);
			if( ! NULL_CHILD(enp,2) ) find_known_return(QSP_ARG  VN_CHILD(enp,2),srp);
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
			find_known_return(QSP_ARG  VN_CHILD(enp,i),srp);
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
	Vec_Expr_Node *enp2;
    //Vec_Expr_Node *resolved_enp;
	Identifier *idp;
	Subrt *srp;
	Run_Info *rip;

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"BEGIN resolve_tree %s:",node_desc(enp));
ADVISE(ERROR_STRING);
DUMP_TREE(enp);
}
#endif /* QUIP_DEBUG */

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
//advise("ready to check special cases");
//dump_tree(enp);
	switch(VN_CODE(enp)){
		case T_SUBVEC:
		case T_CSUBVEC:
			/* we really want to do this as late as possible... */
			/* If the subvector is defined by indices which are variable, we need to set
			 * the NEEDS_CALLTIME_RES flag...
			 */
			UPDATE_TREE_SHAPE(enp);
			break;

		case T_VV_FUNC:
//advise("checking special case T_VV_FUNC");
//dump_tree(enp);
			if( ( ! UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) ) &&
				! UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,1)) ) ){
				Shape_Info *shpp;

//ADVISE("resolve_tree:  BOTH child shapes are known, getting outer shape...");
				shpp = calc_outer_shape(VN_CHILD(enp,0),VN_CHILD(enp,1));
				copy_node_shape(enp,shpp);
			}

			break;

		case T_EQUIVALENCE:
			/* We really just want to call update_node_shape here... */
			SHAPIFY(enp);
			SET_VN_FLAG_BITS(enp, NODE_NEEDS_CALLTIME_RES);	/* this forces us to forget */
			break;

		case T_INDIR_CALL:
			/* First see if the function ptr has been set */
			/*
			WARN("resolve_tree:  not sure how to handle INDIR_CALL");
			*/
			break;

		case T_POINTER:
			idp = EVAL_PTR_REF(enp,EXPECT_PTR_SET);
			if( idp == NULL ) break;	/* probably not set */
			assert( IS_POINTER(idp) );

			if( ! POINTER_IS_SET(idp) ) break;

			if( PTR_REF(ID_PTR(idp)) == NULL ){
				/* This sometimes happens during resolution, but not sure why??? */
				/*
	//			sprintf(ERROR_STRING,"CAUTIOUS:  resolve_tree %s:  ptr %s has null ref arg!?",
					node_desc(enp),ID_NAME(idp));
				WARN(ERROR_STRING);
				*/
				break;
			}
			assert( REF_OBJ(PTR_REF(ID_PTR(idp))) != NULL );

			if( ! UNKNOWN_SHAPE( OBJ_SHAPE(REF_OBJ(PTR_REF(ID_PTR(idp)))) ) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree POINTER %s:  object %s has known shape",node_desc(enp),
OBJ_NAME(REF_OBJ(PTR_REF(ID_PTR(idp)))));
ADVISE(ERROR_STRING);
DESCRIBE_SHAPE(OBJ_SHAPE(REF_OBJ(PTR_REF(ID_PTR(idp)))));
}
#endif /* QUIP_DEBUG */
				RESOLVE_POINTER(enp,OBJ_SHAPE(REF_OBJ(PTR_REF(ID_PTR(idp)))));
				return;
/*
ADVISE("after resolution pointer:");
DUMP_TREE(enp);
*/
			}

			break;

		case T_CALLFUNC:			/* resolve_tree */
							/* should we try a call-time resolution? */
/*
if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
sprintf(ERROR_STRING,"resolve_tree %s:  shape is unknown",node_desc(enp));
ADVISE(ERROR_STRING);
} else {
sprintf(ERROR_STRING,"resolve_tree %s:  shape is KNOWN",node_desc(enp));
ADVISE(ERROR_STRING);
}
*/

			srp = runnable_subrt(QSP_ARG  enp);
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree %s:  before setup_call %s:",node_desc(enp),SR_NAME(srp));
ADVISE(ERROR_STRING);
DUMP_TREE(enp);
}
#endif /* QUIP_DEBUG */
			/* BUG how can we get dst_dp?? */
			rip = SETUP_CALL(srp,NULL);
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree %s:  after setup_call %s:",node_desc(enp),SR_NAME(srp));
ADVISE(ERROR_STRING);
DUMP_TREE(enp);
}
#endif /* QUIP_DEBUG */
			if( rip == NULL ){
				return;
			}

			if( rip->ri_arg_stat >= 0 ){ 
				EVAL_DECL_TREE(SR_BODY(srp));
				LATE_CALLTIME_RESOLVE(srp,NULL);
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
ADVISE("resolve_tree:  after late_calltime_resolve:");
DUMP_TREE(SR_BODY(srp));
}
#endif /* QUIP_DEBUG */
			}

			wrapup_context(QSP_ARG  rip);

			find_known_return(QSP_ARG  NULL,srp);

			/* now maybe we have a return shape??? */
			if( SR_SHAPE(srp) != NULL && ! UNKNOWN_SHAPE(SR_SHAPE(srp)) ){
/*
sprintf(ERROR_STRING,"resolve_tree %s:  setting shape from %s return shape",
node_desc(enp),SR_NAME(srp));
ADVISE(ERROR_STRING);
*/
				copy_node_shape(enp,SR_SHAPE(srp));
			}

			break;

		case T_RETURN:
			/* For a return node, we look up the subroutine and
			 * see if the return shape is known.
			 */
			if( SR_DEST_SHAPE(VN_SUBRT(enp)) != NULL && SHP_PREC(SR_DEST_SHAPE(VN_SUBRT(enp))) != PREC_VOID ){
				if( ! UNKNOWN_SHAPE(SR_DEST_SHAPE(VN_SUBRT(enp))) ){
					/* We know the return destination shape! */
					point_node_shape(QSP_ARG  enp,SR_DEST_SHAPE(VN_SUBRT(enp)));
				}
/*
else {
sprintf(ERROR_STRING,"resolve_tree %s:  subrt %s return shape unknown",
node_desc(enp),SR_NAME(VN_SUBRT(enp)));
ADVISE(ERROR_STRING);
}
*/
			}
/*
else {
sprintf(ERROR_STRING,"resolve_tree %s:  subrt %s return shape nonexistent",
node_desc(enp),SR_NAME(VN_SUBRT(enp)));
ADVISE(ERROR_STRING);
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

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resove_tree:  %s is not a special case",node_desc(enp));
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
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
//advise("resolve_tree:  ready to resolve!");
	if( VN_RESOLVERS(enp) == NULL ){
		/* We don't set the shapes of T_SUBVEC nodes if the range
		 * expressions contain variables (which will be set at runtime).
		 */
		UPDATE_TREE_SHAPE(enp);


		if( UNKNOWN_SHAPE(VN_SHAPE(enp)) && ( VN_CODE(enp) != T_SUBVEC
							&& VN_CODE(enp) != T_CSUBVEC
							&& VN_CODE(enp) != T_SUBSAMP
							&& VN_CODE(enp) != T_CSUBSAMP )
			){
			/* This message gets printed sometimes (see tests/outer_test.scr),
			 * but the computation seems to be done correctly, so we suppress
			 * the msg for now...
			 */
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree %s:  no resolvers",node_desc(enp));
ADVISE(ERROR_STRING);
DUMP_TREE(enp);
}
#endif /* QUIP_DEBUG */
		}
		return;
	}

	np=QLIST_HEAD(VN_RESOLVERS(enp));
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree:  %s has %d resolver nodes",
node_desc(enp),eltcount(VN_RESOLVERS(enp)));
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	while(np!=NULL){
		enp2 = (Vec_Expr_Node *)NODE_DATA(np);
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree:  %s is a resolver node for %s",
node_desc(enp2),node_desc(enp));
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

		assert( enp2 != NULL );

		if( enp2 != whence ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree %s:  trying %s",node_desc(enp),node_desc(enp2));
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			if( VN_SHAPE(enp2) == NULL ||	/* could be a void callfunc node */
				UNKNOWN_SHAPE(VN_SHAPE(enp2)) )

				RESOLVE_TREE(enp2,enp);
			
			/* The above call to resolve_tree could have resolved enp2 */
			if( VN_SHAPE(enp2)!=NULL && ! UNKNOWN_SHAPE(VN_SHAPE(enp2)) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree %s:  effecting resolution using %s",node_desc(enp),node_desc(enp2));
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
				/*resolved_enp=*/EFFECT_RESOLUTION(enp,enp2);
				if(VN_RESOLVERS(enp) != NULL ){
					/* careful - we dont want a recursive call, we may
					 * have gotten here from resolve_uk_nodes!
					 */
					RESOLVE_UK_NODES(VN_RESOLVERS(enp));
				}
				return;
			}
		}
		np=NODE_NEXT(np);
	}
} /* end resolve_tree() */


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
if( ret_shpp != NULL ){
sprintf(ERROR_STRING,"resolve_subrt %s:  return shape known:",SR_NAME(srp));
ADVISE(ERROR_STRING);
DESCRIBE_SHAPE(ret_shpp);
}
*/

/*
sprintf(ERROR_STRING,"resolve_subrt %s:  calling pop_previous",SR_NAME(srp));
ADVISE(ERROR_STRING);
*/
	prev_cpp = POP_PREVIOUS();

	save_srp = curr_srp;
	curr_srp = srp;

/*
sprintf(ERROR_STRING,"resolve_subrt %s setting SCANNING flag",SR_NAME(srp));
ADVISE(ERROR_STRING);
*/
	SET_SR_FLAG_BITS(srp, SR_SCANNING);

	if( argval_tree != NULL ){
		stat = CHECK_ARG_SHAPES(SR_ARG_DECLS(srp),argval_tree,srp);
		if( stat < 0 ) {
sprintf(ERROR_STRING,"resolve_subrt %s:  argument error",SR_NAME(srp));
WARN(ERROR_STRING);
			goto givup;
		}
	}

	/* set the context */
/*
sprintf(ERROR_STRING,"resolve_subrt %s:  done w/ check_arg_shapes",SR_NAME(srp));
ADVISE(ERROR_STRING);
DUMP_SUBRT(srp);
*/

	set_subrt_ctx(QSP_ARG  SR_NAME(srp));

	/* declare the arg variables */
	EVAL_DECL_TREE(SR_ARG_DECLS(srp));	/* resolve_subrt() */
	EVAL_DECL_TREE(SR_BODY(srp));		/* resolve_subrt() */
	if( SR_PREC_CODE(srp) != PREC_VOID ){
		if( ret_shpp != NULL && ! UNKNOWN_SHAPE(ret_shpp) ){
			SET_SR_SHAPE(srp, ret_shpp);
		} else {
			SET_SR_SHAPE(srp, uk_shape(SR_PREC_CODE(srp)));
		}
	}
	/* assert( SR_SHAPE(srp) == NULL ); */

	/* we need to assign the arg vals for any ptr arguments! */

	SET_SR_DEST_SHAPE(srp, ret_shpp);

	RESOLVE_UK_NODES(uk_list);

	/* now try for the args */
	/* resolve_uk_args( */

	SET_SR_DEST_SHAPE(srp, NULL);

/*
sprintf(ERROR_STRING,"resolve_subrt %s:  deleting subrt context",SR_NAME(srp));
ADVISE(ERROR_STRING);
*/
	delete_subrt_ctx(QSP_ARG  SR_NAME(srp));

givup:

/*
sprintf(ERROR_STRING,"resolve_subrt %s clearing SCANNING flag",SR_NAME(srp));
ADVISE(ERROR_STRING);
*/
	CLEAR_SR_FLAG_BITS(srp, SR_SCANNING);

	curr_srp = save_srp;

	if( prev_cpp != NULL ){
/*
sprintf(ERROR_STRING,"resolve_subrt %s:  restoring context",SR_NAME(srp));
ADVISE(ERROR_STRING);
*/
		RESTORE_PREVIOUS(prev_cpp);
	}

	/* we may have been passed uko arg vals,
	 * and our caller may expect us to resolve these...
	 * see if the corresponding arg decls have their shapes set...
	 */

	if( argval_tree != NULL )
		RESOLVE_ARGVAL_SHAPES(argval_tree,SR_ARG_DECLS(srp),srp);

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

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"Begin early_calltime_resolve %s",SR_NAME(srp));
ADVISE(ERROR_STRING);
DUMP_SUBRT(srp);
}
#endif /* QUIP_DEBUG */

	/* We are about to call a subroutine, but we are not actually executing.
	 * We have a destination object (which may be unknown or NULL for void).
	 * BUT we do not yet have the arg values!?
	 * Maybe what we want to do here is resolve the arg shapes?
	 *
	 * We begin by descending the tree and making a list of nodes
	 * that need to be resolved...  Then we attempt to resolve each one.
	 */

	lp = get_uk_list(QSP_ARG  SR_BODY(srp));
	if( lp == NULL ){
#ifdef QUIP_DEBUG
/*
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"early_calltime_resolve %s:  no UK nodes, returning",SR_NAME(srp));
ADVISE(ERROR_STRING);
DUMP_TREE(SR_BODY(srp));
}
*/
#endif /* QUIP_DEBUG */
		return;
	}

	/* set up the subroutine context */


	save_exec = executing;
	executing=0;
	save_res = resolution_flags;
	resolution_flags = NODE_NEEDS_CALLTIME_RES;

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"early_calltime_resolve calling resolve_subrt %s, flags = 0x%x, saved flags = 0x%x",
SR_NAME(srp),resolution_flags, save_res);
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( dst_dp == NULL )
		RESOLVE_SUBRT(srp,lp,SR_ARG_VALS(srp),NULL);
	else
		RESOLVE_SUBRT(srp,lp,SR_ARG_VALS(srp),OBJ_SHAPE(dst_dp));

	executing = save_exec;

	/* Before we introduced pointers, it was an error to have any unknown shapes at
	 * this point...  but because ptrs can be assigned to objects of various shapes
	 * during execution, we really can't resolve things involving them until runtime
	 * of the lines in question after the assignments have been made.  This is
	 * going to be a problem if subroutine return values are needed to resolve
	 * upstream nodes, but we'll worry about *that* when we run into it...
	 */

	dellist(lp);

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"early_calltime_resolve %s DONE",SR_NAME(srp));
ADVISE(ERROR_STRING);
DUMP_SUBRT(srp);
}
#endif /* QUIP_DEBUG */

	resolution_flags = save_res;
}

void xform_to_bitmap(Shape_Info *shpp)
{
	assert( ! BITMAP_SHAPE(shpp) );

	/* shpp->si_rowpad = N_PAD_BITS(SHP_COLS(shpp)); */
	/* SET_SHP_COLS(shpp, N_WORDS_FOR_BITS(SHP_COLS(shpp))); */
	SET_SHP_PREC_PTR(shpp, PREC_FOR_CODE(PREC_BIT) );
	SET_SHP_FLAG_BITS(shpp, DT_BIT);
}

void xform_from_bitmap(Shape_Info *shpp,prec_t prec)
{
	assert( BITMAP_SHAPE(shpp) );

	/*SET_SHP_COLS(shpp, N_BITMAP_COLS(shpp)); */
	/*shpp->si_rowpad=0; */
	SET_SHP_PREC_PTR(shpp, PREC_FOR_CODE(prec) );
	CLEAR_SHP_FLAG_BITS(shpp, DT_BIT);
fprintf(stderr,"xform_from_bitmap:\n");
describe_shape(DEFAULT_QSP_ARG  shpp);
}

/* setup_spcdom_shape - setup space domain shape (from freq. domain transform shape) */

static void setup_spcdom_shape(Shape_Info *spcdom_shpp, Shape_Info *xf_shpp)
{
	COPY_SHAPE(spcdom_shpp, xf_shpp);
	SET_SHP_COLS(spcdom_shpp,
		SHP_COLS(spcdom_shpp)-1);
	SET_SHP_COLS(spcdom_shpp,
		SHP_COLS(spcdom_shpp)*2);
	SET_SHP_COMPS(spcdom_shpp, 1);
	/* BUG?  should we make sure that the known shape is complex? */
	/* The space domain shape is real */
	CLEAR_SHP_FLAG_BITS(spcdom_shpp,DT_COMPLEX);
	SET_SHP_PREC_PTR(spcdom_shpp,
		PREC_MACH_PREC_PTR(SHP_PREC_PTR(spcdom_shpp)) );
}

static void setup_xform_shape(Shape_Info *xf_shpp,Shape_Info *im_shpp)
{
	*xf_shpp = *im_shpp;
	SET_SHP_COLS(xf_shpp,
		SHP_COLS(xf_shpp) / 2);
	SET_SHP_COLS(xf_shpp,
		SHP_COLS(xf_shpp) + 1 );
	SET_SHP_MACH_DIM(xf_shpp,0, 2);
	/* BUG?  should we make sure that the known shape is real? */
	SET_SHP_FLAG_BITS(xf_shpp, DT_COMPLEX);
	SET_SHP_PREC_PTR(xf_shpp,
		complex_precision(SHP_PREC_PTR(xf_shpp)) );
}


/* Reshape this object.  for now, we don't worry about preserving any data... */

static Data_Obj * reshape_obj(QSP_ARG_DECL  Data_Obj *dp,Shape_Info *shpp)
{
	Data_Obj *tmp_dp;
	char s[LLEN];

	/* the simplest thing is just to blow away the object and make a new one */
	/* Are we sure that we want to use a local (volatile) object here???  I don't think so... */
	/* we use localname to get a unique name in the short term, but we do NOT want
	 * to use make_local_dobj, because then it would be cleaned up at the end of the command...
	 */

	tmp_dp = make_dobj(QSP_ARG  localname(),SHP_TYPE_DIMS(shpp),SHP_PREC_PTR(shpp));
	assert( dp != NULL );
	assert( tmp_dp != NULL );

	strcpy(s,OBJ_NAME(dp));
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
	Shape_Info *tmp_shpp;
	Data_Obj *dp;
/* char remember_name[LLEN]; */

fprintf(stderr,"resolve_obj_id:  BEGIN, idp = 0x%lx, shpp = 0x%lx\n",(long)idp,(long)shpp);
fflush(stderr);
fprintf(stderr,"resolve_obj_id:  have id %s\n",ID_NAME(idp));
	assert( shpp != NULL );
    
	INIT_SHAPE_PTR(tmp_shpp)

	/* decl_enp = OBJ_EXTRA(dp); */

	/* matlab has no declarations!? */
	if( mode_is_matlab ){
/*
sprintf(ERROR_STRING,"resolve_obj_id %s:  mode is matlab",ID_NAME(idp));
ADVISE(ERROR_STRING);
*/
		dp = GET_OBJ(ID_NAME(idp));

		assert( dp != NULL );

		REF_OBJ(ID_REF(idp)) = reshape_obj(QSP_ARG  dp,shpp);

		return;
	}

	assert( ID_TYPE(idp) == ID_REFERENCE );

	decl_enp = REF_DECL_VN(ID_REF(idp));
	dp = REF_OBJ(ID_REF(idp));

	assert( OBJ_EXTRA(dp) == decl_enp );

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_obj_id:  setting decl node shape for %s", VN_STRING(decl_enp));
ADVISE(ERROR_STRING);
DESCRIBE_SHAPE(shpp);
}
#endif /* QUIP_DEBUG */
	COPY_SHAPE(tmp_shpp, shpp);
	SET_SHP_PREC_PTR(tmp_shpp,VN_PREC_PTR(decl_enp));

	/* used to propagate complex bit here */
	if( COMPLEX_PRECISION(SHP_PREC(tmp_shpp)) )
		SET_SHP_FLAG_BITS(tmp_shpp, DT_COMPLEX);

	copy_node_shape(decl_enp,tmp_shpp);
	/* make sure to do this, needed for runtime! */
	/* need to copy NEEDS_CALLTIME_RES flag too!? */
	SET_VN_FLAG_BITS(decl_enp, resolution_flags);

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_obj_id:  resolution_flags = 0x%x",resolution_flags);
ADVISE(ERROR_STRING);

sprintf(ERROR_STRING,"resolve_obj_id:  calling reeval_decl_stat %s", VN_STRING(decl_enp));
ADVISE(ERROR_STRING);
if( shpp == NULL ) WARN("shape ptr is null!?");
DESCRIBE_SHAPE(shpp);
}
#endif /* QUIP_DEBUG */

	/* the pointer points to the object, but the object's name may be in a different
	 * context.  We need to reinstate the context before reevaluating the declaration.
	 * We do this by saving the identifier context in the declaration node.
	 */
/*
strcpy(remember_name,OBJ_NAME(dp));
sprintf(ERROR_STRING,"resolve_obj_id %s, old dp = 0x%lx",OBJ_NAME(dp),(u_long)dp);
ADVISE(ERROR_STRING);
longlist(dp);
*/
	reeval_decl_stat(QSP_ARG  OBJ_PREC_PTR(dp),decl_enp,VN_DECL_FLAGS(decl_enp));
/*
dp = REF_OBJ(ID_REF(idp));
dp=dobj_of(remember_name);
if( dp == NULL ) {
sprintf(ERROR_STRING,"resolve_obj_id:  missing object %s!?",remember_name);
ERROR1(ERROR_STRING);
}
sprintf(ERROR_STRING,"resolve_obj_id %s, new dp = 0x%lx",OBJ_NAME(dp),(u_long)dp);
ADVISE(ERROR_STRING);
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

	if( COMPLEX_PRECISION(SHP_PREC(shpp1)) ){
		if( COMPLEX_PRECISION(SHP_PREC(shpp2)) ){
			if( shpp1->si_tdim != 2 || shpp2->si_tdim != 2 ) return(0);
		} else {
			if( shpp1->si_tdim != 2 || shpp2->si_tdim != 1 ) return(0);
		}
	} else {
		if( COMPLEX_PRECISION(SHP_PREC(shpp2)) ){
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

	if( idp == NULL ){
/*
NODE_ERROR(uk_enp);
sprintf(ERROR_STRING,"resolve_pointer:  no set identifier for %s",node_desc(uk_enp));
ADVISE(ERROR_STRING);
*/
		/* this probably means that the ptr is not set, needs to be resolved
		 * at runtime.
		 */
		return;
	}
	/* At one time we did a CAUTIOUS check to see if the object was in the database, //
	 * but that could fail because the pointed to object might be out of scope
	 * (e.g.  ptr to local object passed as subrt arg)
	 */

	dp = REF_OBJ(PTR_REF(ID_PTR(idp)));

	/* BUG check ptr set first */

	assert( dp != NULL );

	if( UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
		/* Now we know the object in question has unknown shape -
		 * Resolve it!
		 */
		/* now we try to resolve the pointed-to object */
sprintf(ERROR_STRING,"resolve_pointer %s:  resolving pointer target object",node_desc(uk_enp));
ADVISE(ERROR_STRING);
		RESOLVE_OBJ_ID(REF_ID(PTR_REF(ID_PTR(idp))),shpp);
	} else {
		/* make sure that the shapes are the same */
		if( ! same_shape(OBJ_SHAPE(dp),shpp) ){
			NODE_ERROR(uk_enp);
			sprintf(ERROR_STRING,
		"resolve_pointer:  Shape mismatch with object %s",OBJ_NAME(dp));
			WARN(ERROR_STRING);
DESCRIBE_SHAPE(VN_SHAPE(uk_enp));
DESCRIBE_SHAPE(shpp);
		}
	}

/*
sprintf(ERROR_STRING,"resolve_pointer fixing %s",node_desc(uk_enp));
ADVISE(ERROR_STRING);
*/

	/* also fix the node where we found this */
	point_node_shape(QSP_ARG  uk_enp,shpp);
}



/* We call resolve_object with a T_DYN_OBJ node.
 * The shape pointer should point to a valid shape...
 */

static void resolve_object(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Shape_Info *shpp)
{
	Identifier *idp;
	Data_Obj *dp;

	/* If this is an automatic object, and we haven't evaluated the decls yet... */
	// Why do we think the object name is in VN_STRING?
dump_tree(QSP_ARG  uk_enp);
	
	switch(VN_CODE(uk_enp)){
		case T_DYN_OBJ:
fprintf(stderr,"resolve_object passed unknown dynamic object %s\n",VN_STRING(uk_enp));
			idp = GET_ID(VN_STRING(uk_enp));
			assert( idp != NULL );
			assert( IS_REFERENCE(idp) );

			dp = REF_OBJ(ID_REF(idp));
			assert( dp != NULL );
			break;

		case T_STATIC_OBJ:
			dp = VN_OBJ(uk_enp);
			idp = GET_ID(OBJ_NAME(dp));
			break;

		default:
			MISSING_CASE(uk_enp,"resolve_object");
			dp = NULL;
			idp = NULL;	// silence compiler
			break;
	}

	SET_VN_FLAG_BITS(uk_enp, resolution_flags);		/* resolve_object */

	if( ! UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
		/* This can occur as a matter of course if we have multiple T_DYN_OBJ
		 * nodes, after the first is resolved the shape is propagated to the
		 * declaration node.  The other instances don't get fixed right away...
		 * (when do they get fixed???)
		 * BUT wait:  shouldn't the nodes all point to the declaration shape???
		 */

		return;
	}

fprintf(stderr,"resolve_object calling resolve_obj_id, idp = 0x%lx\n",(long)idp);
	RESOLVE_OBJ_ID(idp,shpp);

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_object %s:",node_desc(uk_enp));
ADVISE(ERROR_STRING);
DESCRIBE_SHAPE(VN_SHAPE(uk_enp));
}
#endif /* QUIP_DEBUG */

} /* end resolve_object */


/* here we pass a shape with which to resolve the children (scalar) nodes */

static void resolve_obj_list(QSP_ARG_DECL  Vec_Expr_Node *enp, Shape_Info *shpp)
{
	switch(VN_CODE(enp)){
		case T_RET_LIST:
			RESOLVE_OBJ_LIST(VN_CHILD(enp,0),shpp);
			RESOLVE_OBJ_LIST(VN_CHILD(enp,1),shpp);
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
	SET_VN_FLAG_BITS(uk_enp, resolution_flags);		/* resolve_node */

	switch(VN_CODE(uk_enp)){
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
			copy_node_shape(uk_enp,shpp);
			return(uk_enp);

		BINARY_BOOLOP_CASES			/* T_BOOL_AND etc, bitmap shapes */
		case T_BOOL_NOT:			/* does this belong here??? */
		ALL_NUMERIC_COMPARISON_CASES		/* T_BOOL_GT etc, bitmap shapes */
			/* for numeric comparisons, we know that the known child node can't be a bitmap */
			copy_node_shape(uk_enp,shpp);
			if( ! BITMAP_SHAPE(shpp) ){
				xform_to_bitmap(VN_SHAPE(uk_enp));
			}
			break;


		case T_TYPECAST:
			copy_node_shape(uk_enp,shpp);
			//SET_SHP_PREC_PTR(VN_SHAPE(uk_enp), PREC_FOR_CODE(VN_INTVAL(uk_enp)) );
			SET_SHP_PREC_PTR(VN_SHAPE(uk_enp), VN_CAST_PREC_PTR(uk_enp) );
			return(uk_enp);

		case T_POINTER:			/* resolve_node */
			RESOLVE_POINTER(uk_enp,shpp);

			/* We can't propagate from the pointer node... */
			return(NULL);

		/*
		case T_STR_PTR:
		case T_POINTER:
		*/
		case T_STATIC_OBJ:
		case T_DYN_OBJ:			/* resolve_node */
			RESOLVE_OBJECT(uk_enp,shpp);
			return(uk_enp);

		case T_RET_LIST:		/* a list of objects */
			if( ROWVEC_SHAPE(shpp) ){	/* a list of scalars? */
				RESOLVE_OBJ_LIST(uk_enp,scalar_shape(SHP_PREC(shpp)));
			} else {
sprintf(ERROR_STRING,"resolve_node %s:  oops, only know how to resolve from rowvec!?",node_desc(uk_enp));
WARN(ERROR_STRING);
			}
			return(uk_enp);

		case T_SS_B_CONDASS:
			/* The known child is the bitmap */
			copy_node_shape(uk_enp,shpp);
			return(uk_enp);

		case T_VS_B_CONDASS:
			/* The known child may be the bitmap, but hopefully the
			 * correct shape was determined above.
			 */
			copy_node_shape(uk_enp,shpp);
			return(uk_enp);


		default:
			MISSING_CASE(uk_enp,"resolve_node");
			break;
	}
	return(NULL);
} /* end resolve_node */

/* The parent is the unknown node? */

static Vec_Expr_Node *resolve_parent(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Shape_Info *shpp)
{
	prec_t prec;

	switch(VN_CODE(uk_enp)){

		case T_CURLY_SUBSCR:
			COPY_SHAPE(VN_SHAPE(uk_enp), shpp);	/* copy shape */
			/* set a dimension to one */
			SET_SHP_MACH_DIM(VN_SHAPE(uk_enp),SHP_MINDIM(shpp), 1);
			SET_SHP_TYPE_DIM(VN_SHAPE(uk_enp),SHP_MINDIM(shpp), 1);
			SET_SHP_MINDIM(VN_SHAPE(uk_enp),
				SHP_MINDIM(VN_SHAPE(uk_enp)) + 1 );
			return(uk_enp);

			
		case T_TRANSPOSE:
			COPY_SHAPE(VN_SHAPE(uk_enp),shpp);	/* copy shape */
			SET_SHP_COLS(VN_SHAPE(uk_enp), SHP_ROWS(shpp));
			SET_SHP_ROWS(VN_SHAPE(uk_enp), SHP_COLS(shpp));
			/* BUG?  reset shape flags? */
			return(uk_enp);

		case T_RIDFT:		/* resolve_parent, xform shape is known, get spc. dom. size */
			setup_spcdom_shape(VN_SHAPE(uk_enp),shpp);
			return(uk_enp);

		case T_RDFT:		/* resolve_parent, input shape is known, get xform size */
			setup_xform_shape(VN_SHAPE(uk_enp),shpp);
			return(uk_enp);

		ALL_NUMERIC_COMPARISON_CASES
			/* for numeric comparisons, we know that the known child node can't be a bitmap */
			copy_node_shape(uk_enp,shpp);
			xform_to_bitmap(VN_SHAPE(uk_enp));
			return(uk_enp);

		case T_SS_B_CONDASS:	/* resolve_parent, branch shape is known */
		case T_VS_B_CONDASS:	/* resolve_parent, branch shape is known */
		case T_VV_B_CONDASS:	/* resolve_parent, branch shape is known */
			prec = VN_PREC(uk_enp);
			copy_node_shape(uk_enp,shpp);
			if( BITMAP_SHAPE(shpp) )
				xform_from_bitmap(VN_SHAPE(uk_enp),prec);
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
		if( (!UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(uk_enp,0)))) &&
			! UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(uk_enp,1))) ){
			Shape_Info *o_shpp;

			o_shpp = calc_outer_shape(VN_CHILD(uk_enp,0),VN_CHILD(uk_enp,1));
			copy_node_shape(uk_enp,o_shpp);
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
	return(NULL);
} /* resolve_parent */

/* What is the difference between resolve_unknown_parent and resolve_parent???
 */

static Vec_Expr_Node * resolve_unknown_parent(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp)
{
	Shape_Info *shpp;

	/* First get the target shape - this is easy!? */
	switch(VN_CODE(enp)){
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
			shpp = VN_SHAPE(enp);
			break;

		ALL_NUMERIC_COMPARISON_CASES
			/* This is a bitmap shape! */
			shpp = VN_SHAPE(enp);
			break;

		default:
			MISSING_CASE(enp,"resolve_unknown_parent (known node)");
			shpp = VN_SHAPE(enp);
			break;
	}

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
	/* We usually thing of resolution as applying to images etc, and not scalars,
	 * but when we have multiple scalar arguments to return from a function, we may
	 * want to do it with ptr args whose targets are assigned...  When these pointers
	 * are resolved to their scalar targets, this msg will be printed...
	 */
	if( SCALAR_SHAPE(shpp) ){
		sprintf(ERROR_STRING,"ADVISORY:  resolve_unknown_parent %s from %s:  known shape is scalar!?",
			node_desc(uk_enp),node_desc(enp));
		prt_msg(ERROR_STRING);
		DESCRIBE_SHAPE(shpp);
	}
}
#endif /* QUIP_DEBUG */

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_unknown_parent %s from %s",node_desc(uk_enp),node_desc(enp));
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	/* BUG we'd like to merge this with resolve_node... */

	return( RESOLVE_PARENT(uk_enp,shpp));
} /* end resolve_unknown_parent */


/* shape_for_child - what does this do???
 */

static Shape_Info *shape_for_child(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp)
{
	Shape_Info *shpp;

	switch(VN_CODE(enp)){

		BINARY_BOOLOP_CASES			/* shape_for_child (known node) */
			if( BITMAP_SHAPE(VN_SHAPE(uk_enp)) ) return(VN_SHAPE(enp));
			else {
				COPY_SHAPE( VN_BM_SHAPE(enp) , VN_SHAPE(enp));
				xform_from_bitmap(VN_BM_SHAPE(enp),VN_PREC(uk_enp));
				return(VN_BM_SHAPE(enp));
			}

		ALL_NUMERIC_COMPARISON_CASES		/* T_BOOL_GT etc */
			COPY_SHAPE(VN_BM_SHAPE(enp) , VN_SHAPE(enp) );
			xform_from_bitmap(VN_BM_SHAPE(enp),VN_PREC(uk_enp));
			return(VN_BM_SHAPE(enp));

		ALL_CONDASS_CASES
			if( uk_enp == VN_CHILD(enp,0) ){
				COPY_SHAPE(VN_BM_SHAPE(enp) , VN_SHAPE(enp) );
				xform_to_bitmap(VN_BM_SHAPE(enp));
				return(VN_BM_SHAPE(enp));
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

			return( VN_SHAPE(enp) );

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

			setup_xform_shape(VN_SIZCH_SHAPE(enp),VN_SHAPE(enp));
			return(VN_SIZCH_SHAPE(enp));

		case T_RDFT:		/* shape_for_child, RDFT xform shape is known, get input shape */
			setup_spcdom_shape(VN_SIZCH_SHAPE(enp),VN_SHAPE(enp));
			return(VN_SIZCH_SHAPE(enp));

		case T_ENLARGE:
			shpp = VN_SIZCH_SHAPE(enp);
			COPY_SHAPE(shpp , VN_SHAPE(enp));
			SET_SHP_COLS(shpp,
				SHP_COLS(shpp) / 2 );
			SET_SHP_ROWS(shpp,
				SHP_ROWS(shpp) / 2);
			return(shpp);

		case T_REDUCE:
			shpp = VN_SIZCH_SHAPE(enp);
			COPY_SHAPE(shpp , VN_SHAPE(enp));
			SET_SHP_COLS(shpp,
				SHP_COLS(shpp) * 2 );
			SET_SHP_ROWS(shpp,
				SHP_ROWS(shpp) * 2);
			return(shpp);

		case T_TRANSPOSE:
			shpp = VN_SIZCH_SHAPE(enp);
			COPY_SHAPE(shpp , VN_SHAPE(enp));
			SET_SHP_COLS(shpp, SHP_ROWS(VN_SHAPE(enp)) );
			SET_SHP_ROWS(shpp, SHP_COLS(VN_SHAPE(enp)) );
			return(shpp);

		case T_TYPECAST:
			/* for a typecast node, the child needs to keep the original precision.
			 * We create a new shape, potentially a garbage collection problem?
			 */
			shpp = (Shape_Info *)getbuf(sizeof(*shpp));
			*shpp = *VN_SHAPE(enp);
			SET_SHP_PREC_PTR(shpp, SHP_PREC_PTR(VN_SHAPE(uk_enp)));
			return(shpp);

		default:
			MISSING_CASE(enp,"shape_for_child");
			return( VN_SHAPE(enp) );
	}
} /* end shape_for_child */

static Vec_Expr_Node * resolve_unknown_child(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp)
{
	Shape_Info *shpp;

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_unknown_child %s from %s",node_desc(uk_enp),node_desc(enp));
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	shpp = shape_for_child(QSP_ARG  uk_enp,enp);

	if( mode_is_matlab && VN_CODE(uk_enp) == T_RET_LIST ){
		//Data_Obj *dp;

		/*dp =*/ EVAL_OBJ_REF(uk_enp);	/* force object creation */
	}

	/* Now shpp point to the shape we will use for resolution */

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
	/* We usually thing of resolution as applying to images etc, and not scalars,
	 * but when we have multiple scalar arguments to return from a function, we may
	 * want to do it with ptr args whose targets are assigned...  When these pointers
	 * are resolved to their scalar targets, this msg will be printed...
	 */
	if( SCALAR_SHAPE(shpp) ){
		sprintf(ERROR_STRING,"ADVISORY:  resolve_unknown_child %s from %s:  known shape is scalar!?",
			node_desc(uk_enp),node_desc(enp));
		prt_msg(ERROR_STRING);
		DESCRIBE_SHAPE(shpp);
	}
}
#endif /* QUIP_DEBUG */

	return(RESOLVE_NODE(uk_enp,shpp));
} /* resolve_unknown_child */

#ifdef NOT_USED

/* resolve_return
 *
 * We call this when we resolve a return node from the calling CALLFUNC node - these
 * are not linked, although the program would be simpler if the could be !? (BUG?)
 */

void resolve_return(QSP_ARG_DECL  Vec_Expr_Node *enp,Shape_Info *shpp)
{
	point_node_shape(QSP_ARG  enp,shpp);
}

#endif /* NOT_USED */

/* effect_resolution
 *
 * The main entry point to this module.
 * We have a pair of nodes which can resolve each other, and one of
 * them is now known.
 * Returns a ptr to the resolved node, or NULL
 */

Vec_Expr_Node * effect_resolution(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp)
{
	Vec_Expr_Node *ret_enp=NULL;

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"effect_resolution of %s from %s",node_desc(uk_enp),node_desc(enp));
ADVISE(ERROR_STRING);
ADVISE("known shape:");
DESCRIBE_SHAPE(VN_SHAPE(enp));
}
#endif /* QUIP_DEBUG */

	/* BUG?  do we need to save the old value of the flags? */
	if( VN_FLAGS(enp) & NODE_NEEDS_CALLTIME_RES )
		resolution_flags |= NODE_NEEDS_CALLTIME_RES;

	/* We know that one of these nodes is the parent of the
	 * other - which one is which ?
	 */
	
	if( VN_PARENT(enp) == uk_enp ) ret_enp=RESOLVE_UNKNOWN_PARENT(uk_enp,enp);
	else if( VN_PARENT(uk_enp) == enp ) ret_enp=RESOLVE_UNKNOWN_CHILD(uk_enp,enp);
	else {
		RESOLVE_NODE(uk_enp,VN_SHAPE(enp));
	}


#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"After effect_resolution of %s from %s:",node_desc(uk_enp),node_desc(enp));
ADVISE(ERROR_STRING);
ADVISE("known shape:");
DESCRIBE_SHAPE(VN_SHAPE(enp));
ADVISE("resolved shape:");
DESCRIBE_SHAPE(VN_SHAPE(uk_enp));
}
#endif /* QUIP_DEBUG */

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
	if( decl_enp == NULL ) return;

	switch(VN_CODE(decl_enp)){
		case T_DECL_STAT_LIST:
			RESOLVE_ARGVAL_SHAPES(VN_CHILD(val_enp,0),
				VN_CHILD(decl_enp,0),srp);
			RESOLVE_ARGVAL_SHAPES(VN_CHILD(val_enp,1),
				VN_CHILD(decl_enp,1),srp);
			break;
		case T_DECL_STAT:
			RESOLVE_ARGVAL_SHAPES(val_enp,
				VN_CHILD(decl_enp,0),srp);
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
			assert( VN_SHAPE(val_enp) != NULL );

			/* See if the value has unknown shape */
			if( ! UNKNOWN_SHAPE(VN_SHAPE(val_enp)) )
				return;

			/* value has unknown shape, see if decl is known */
			if( UNKNOWN_SHAPE( VN_SHAPE(decl_enp) ) )
				return;

			/* value is unknown, decl is known:  propagate! */

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
ADVISE("resolve_argval_shapes:  propagating a shape!!");
DESCRIBE_SHAPE(VN_SHAPE(decl_enp));
}
#endif /* QUIP_DEBUG */
			PROPAGATE_SHAPE(val_enp,VN_SHAPE(decl_enp));
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
ADVISE("resolve_argval_shapes:  DONE");
DESCRIBE_SHAPE(VN_SHAPE(val_enp));
}
#endif /* QUIP_DEBUG */
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
	Shape_Info *tmp_shpp;

	INIT_SHAPE_PTR(tmp_shpp)

	assert( UNKNOWN_SHAPE(VN_SHAPE(enp)) );

	switch(VN_CODE(enp)){
		case T_POINTER:		/* propagate_shape */
			RESOLVE_POINTER(enp,shpp);
			break;

		case T_STR_PTR:
		case T_DYN_OBJ:
			RESOLVE_OBJECT(enp,shpp);
			break;

		case T_CURLY_SUBSCR:
			COPY_SHAPE(tmp_shpp , shpp);
			SET_SHP_TYPE_DIM(tmp_shpp,SHP_MINDIM(tmp_shpp),1);
			SET_SHP_MACH_DIM(tmp_shpp,SHP_MINDIM(tmp_shpp),1);
			SET_SHP_MINDIM(tmp_shpp,
				SHP_MINDIM(tmp_shpp) + 1 );
			PROPAGATE_SHAPE(VN_CHILD(enp,0),tmp_shpp);
			break;

		case T_SQUARE_SUBSCR:
			COPY_SHAPE(tmp_shpp , shpp);
			SET_SHP_TYPE_DIM(tmp_shpp,SHP_MAXDIM(tmp_shpp),1);
			SET_SHP_MACH_DIM(tmp_shpp,SHP_MAXDIM(tmp_shpp),1);
			SET_SHP_MAXDIM(tmp_shpp,
				SHP_MAXDIM(tmp_shpp) - 1 );
			PROPAGATE_SHAPE(VN_CHILD(enp,0),tmp_shpp);
			break;

		case T_VV_FUNC:
			PROPAGATE_SHAPE(VN_CHILD(enp,0),shpp);
			PROPAGATE_SHAPE(VN_CHILD(enp,1),shpp);
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

	if( enp==NULL ) return(1);

	switch(VN_CODE(enp)){
		case T_LIT_INT:
		case T_LIT_DBL:
			return(1);

		case T_ARGLIST:
			stat=arg_sizes_known(VN_CHILD(enp,0));
			if( ! stat ) return(stat);
			stat=arg_sizes_known(VN_CHILD(enp,1));
			return(stat);

		case T_DYN_OBJ:
			if( UNKNOWN_SHAPE(VN_SHAPE(enp)) )
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

	if( arg_enp == NULL ) return(0);
	else if( val_enp == NULL ){
		/* BUG we want to report this error at the line number of the callfunc node */
		sprintf(ERROR_STRING,"Subroutine %s requires arguments",SR_NAME(srp));
		WARN(ERROR_STRING);
		return(-1);
	}

	switch(VN_CODE(arg_enp)){
		case  T_DECL_STAT:
			/* en_intval is the type (float,short,etc) */
			stat=CHECK_ARG_SHAPES(VN_CHILD(arg_enp,0),
						val_enp,srp);
			return(stat);

		case T_DECL_STAT_LIST:
			/* val_enp should be T_ARGLIST */
			assert( VN_CODE(val_enp) == T_ARGLIST );

			stat=CHECK_ARG_SHAPES(VN_CHILD(arg_enp,0),
				VN_CHILD(val_enp,0),srp);
			if( stat < 0 ) return(stat);
			stat=CHECK_ARG_SHAPES(VN_CHILD(arg_enp,1),
				VN_CHILD(val_enp,1),srp);
			return(stat);

		case T_PTR_DECL:		/* check_arg_shapes */
			/* we really don't care if
			 * the shapes match for a ptr arg, it just assumes the
			 * shape of the value...
			 */
			if( VN_SHAPE(arg_enp) != NULL && UNKNOWN_SHAPE(VN_SHAPE(arg_enp)) ){
				if( ! UNKNOWN_SHAPE(VN_SHAPE(val_enp)) ){
/*
sprintf(ERROR_STRING,"check_arg_shapes %s, %s:  copying shape for ptr",node_desc(arg_enp),node_desc(val_enp));
ADVISE(ERROR_STRING);
*/
					/*copy_decl_shape*/ copy_node_shape(arg_enp,
						VN_SHAPE(val_enp));

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
			assert( VN_SHAPE(arg_enp) != NULL );
			assert( VN_SHAPE(val_enp) != NULL );

			if( UNKNOWN_SHAPE(VN_SHAPE(arg_enp)) ){
				if( ! UNKNOWN_SHAPE(VN_SHAPE(val_enp)) ){
					copy_node_shape(arg_enp, VN_SHAPE(val_enp));
					/* Now we would like to propagate into the body of the subroutine!? */
					SET_VN_FLAG_BITS(arg_enp, resolution_flags);
				}

				/* it's not necessarily an error for both
				 * to be unknown at this time...
				 */

			} else if( UNKNOWN_SHAPE(VN_SHAPE(val_enp)) ){
				/* here the arg decl has known shape, but the arg val is unknown;
				 * We'd like to propagate upwards...
				 */
				PROPAGATE_SHAPE(val_enp,VN_SHAPE(arg_enp));
			} else {	/* everything is known */
				if( !shapes_match(VN_SHAPE(arg_enp),
							VN_SHAPE(val_enp)) ){
					NODE_ERROR(SR_CALL_VN(srp));
					sprintf(ERROR_STRING,
	"subrt %s:  argument shape mismatch",SR_NAME(srp));
					WARN(ERROR_STRING);
ADVISE("argument prototype shape:");
DESCRIBE_SHAPE(VN_SHAPE(arg_enp));
ADVISE("argument value shape:");
DESCRIBE_SHAPE(VN_SHAPE(val_enp));
DUMP_TREE(arg_enp);

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
ADVISE(VN_STRING(arg_enp));
DESCRIBE_SHAPE(VN_SHAPE(arg_enp));
DESCRIBE_SHAPE(VN_SHAPE(val_enp));
}
#endif /* QUIP_DEBUG */
					return(-1);
				}
			}
			return(0);

		default:
			MISSING_CASE(arg_enp,"check_arg_shapes");
			return(-1);
	}
}

