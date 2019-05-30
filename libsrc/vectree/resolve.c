
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

uint32_t resolution_flags=0;		/* why is this global? */
debug_flag_t resolve_debug=0;


#define max( n1 , n2 )		(n1>n2?n1:n2)


/* find_uk_nodes
 *
 * Descend this node, looking for unknown shapes.
 * When we find one, we add it to the list.
 */

#define find_uk_nodes(enp,lp) _find_uk_nodes(QSP_ARG  enp,lp)

static void _find_uk_nodes(QSP_ARG_DECL  Vec_Expr_Node *enp,List *lp)
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
			ALL_BOOLOP_CASES

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
//advise(ERROR_STRING);
				np=mk_node(enp);
				addTail(lp,np);
				break;
			default:
				missing_case(enp,"find_uk_nodes");
				np=mk_node(enp);
				addTail(lp,np);
				break;
		}
	}
	for(i=0;i<MAX_CHILDREN(enp);i++)
		if( VN_CHILD(enp,i) != NULL )
			find_uk_nodes(VN_CHILD(enp,i),lp);
}

#define get_uk_list(enp) _get_uk_list(QSP_ARG  enp)

static List *_get_uk_list(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	List *lp;

	lp=new_list();
	find_uk_nodes(enp,lp);
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

#define forget_resolved_tree(enp) _forget_resolved_tree(QSP_ARG  enp)

static void _forget_resolved_tree(QSP_ARG_DECL  Vec_Expr_Node *enp)
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
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

		} else {
			if( OWNS_SHAPE(enp) )
				copy_node_shape(enp,uk_shape(VN_PREC(enp)));
			else
				point_node_shape(enp,uk_shape(VN_PREC(enp)));
		}
	}

	for(i=0;i<MAX_CHILDREN(enp);i++)
		if( VN_CHILD(enp,i) != NULL )
			forget_resolved_tree(VN_CHILD(enp,i));
} /* end forget_resolved_tree */

/* forget_resolved_shapes
 *
 * Shapes of unknown shape objs in this subroutine get stored in the declaration
 * nodes...  We used to keep a list of them, here we try to find them by rescanning
 * the declaration trees...
 */

void _forget_resolved_shapes(QSP_ARG_DECL  Subrt *srp)
{
	/* First check the arguments */

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"forget_resolve_shapes %s:",SR_NAME(srp));
advise(ERROR_STRING);
dump_tree(SR_BODY(srp));
}
#endif /* QUIP_DEBUG */

	if( SR_ARG_DECLS(srp) != NULL )
		forget_resolved_tree(SR_ARG_DECLS(srp));
	if( SR_BODY(srp) != NULL )
		forget_resolved_tree(SR_BODY(srp));

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"forget_resolve_shapes %s DONE",SR_NAME(srp));
advise(ERROR_STRING);
dump_tree(SR_BODY(srp));
}
#endif /* QUIP_DEBUG */

}


/* We call this with a list of nodes which may need resolution.
 * This can be called with a list we construct from a tree, or
 * with a node's list (propagation), so we therefore cannot assume
 * that all the nodes on the list actually need resolution.
 */

#define resolve_uk_nodes(lp) _resolve_uk_nodes(QSP_ARG  lp)

static void _resolve_uk_nodes(QSP_ARG_DECL  List *lp)
{
	Node *np;

	assert(lp!=NULL);
	np=QLIST_HEAD(lp);
	while(np!=NULL){
		Vec_Expr_Node *enp;

		enp=(Vec_Expr_Node *)NODE_DATA(np);
		if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_uk_nodes trying to fix %s",node_desc(enp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			resolve_tree(enp,NULL);
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

void _late_calltime_resolve(QSP_ARG_DECL  Subrt *srp, Data_Obj *dst_dp)
{
	List *lp;
	uint32_t save_res;

	assert( curr_srp == srp );

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
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"late_calltime_resolve %s begin:",SR_NAME(srp));
advise(ERROR_STRING);
dump_subrt(srp);
}
#endif /* QUIP_DEBUG */

	lp = get_uk_list(SR_BODY(srp));
	if( lp != NULL ){
		resolve_uk_nodes(lp);

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"late_calltime_resolve %s done:",SR_NAME(srp));
advise(ERROR_STRING);
dump_subrt(srp);
}
#endif /* QUIP_DEBUG */

		dellist(lp);
	}

	CLEAR_SR_FLAG_BITS(srp, SR_SCANNING);

	resolution_flags = save_res;
}

// find_known_return - recursively descend the tree, looking for
// a return node with a known shape

#define find_known_return(enp) _find_known_return(QSP_ARG  enp)

static Vec_Expr_Node * _find_known_return(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int i;
	Vec_Expr_Node *ret_enp;

	assert(enp!=NULL);

	switch(VN_CODE(enp)){
		case T_FOR:
			// probably don't need to descend the 3
			// children, but no harm in doing so to be safe.
			break;
		case T_RETURN:
			assert( VN_SHAPE(enp) != NULL );

			/*
			if( ! UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
				copy_node_shape(call_enp,VN_SHAPE(enp));
			}
			*/

			return enp;

		/* for all of these cases we descend and keep looking */
		case T_STAT_LIST:
			break;

		case T_IFTHEN:
			/* child 0 is the test,and can't have a return statement */
			if( ! NULL_CHILD(enp,1) ){
				ret_enp = find_known_return(VN_CHILD(enp,1));
				if( ret_enp != NULL )
					return ret_enp;
			}
			if( ! NULL_CHILD(enp,2) ){
				ret_enp = find_known_return(VN_CHILD(enp,2));
				return ret_enp;
			}

		/* for all of these cases, we know there will not be a
		 * child return node, so we return right away
		 */
		ALL_NUMERIC_COMPARISON_CASES
		ALL_BOOLOP_CASES
		ALL_INCDEC_CASES

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
			return NULL;

		default:
			missing_case(enp,"find_known_return");
			break;
	}
	for(i=0;i<MAX_CHILDREN(enp);i++)
		if( ! NULL_CHILD(enp,i) ){
			ret_enp = find_known_return(VN_CHILD(enp,i));
			if( ret_enp != NULL )
				return ret_enp;
		}

	return NULL;
} /* find_known_return */

#ifdef MAX_DEBUG

static void show_resolver_list(const char *prefix, Vec_Expr_Node *enp)
{
	List *lp;
	Node *np;

	lp = VN_RESOLVERS(enp);
	if( lp == NULL || (np=QLIST_HEAD(lp)) == NULL ){
		fprintf(stderr,"%s%s has no resolvers.\n",prefix,node_desc(enp));
		return;
	}

	fprintf(stderr,"%s%s (0x%lx) has %d resolvers:\n",prefix,node_desc(enp),(long)enp,eltcount(lp));
	while(np!=NULL){
		enp = NODE_DATA(np);
		fprintf(stderr,"%s\t%s (np = 0x%lx, enp = 0x%lx)\n",prefix,node_desc(enp),(long)np,(long)enp);
		if( strlen(prefix) < 3 ){
			char p[12];
			sprintf(p,"\t%s",prefix);
			show_resolver_list(p,enp);
		}
		np = NODE_NEXT(np);
	}
}

void dump_resolvers(Vec_Expr_Node *enp)
{
	show_resolver_list("",enp);
}
#endif // MAX_DEBUG

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

void _resolve_tree(QSP_ARG_DECL  Vec_Expr_Node *enp,Vec_Expr_Node *whence)
{
	Node *np;
	Vec_Expr_Node *resolver_enp;
	Vec_Expr_Node *ret_enp;
	Identifier *idp;
	Subrt *srp;
	Run_Info *rip;

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"BEGIN resolve_tree %s:",node_desc(enp));
advise(ERROR_STRING);
dump_tree(enp);
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
			update_tree_shape(enp);
			break;

		case T_VV_FUNC:
//advise("checking special case T_VV_FUNC");
//dump_tree(enp);
			if( ( ! UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) ) &&
				! UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,1)) ) ){
				Shape_Info *shpp;

//advise("resolve_tree:  BOTH child shapes are known, getting outer shape...");
				shpp = calc_outer_shape(VN_CHILD(enp,0),VN_CHILD(enp,1));
				copy_node_shape(enp,shpp);
			}

			break;

		case T_EQUIVALENCE:
			update_node_shape(enp);
			SET_VN_FLAG_BITS(enp, NODE_NEEDS_CALLTIME_RES);	/* this forces us to forget */
			break;

		case T_INDIR_CALL:
			/* First see if the function ptr has been set */
			/*
			warn("resolve_tree:  not sure how to handle INDIR_CALL");
			*/
			break;

		case T_POINTER:
			idp = eval_ptr_expr(enp,EXPECT_PTR_SET);
			if( idp == NULL ){
				break;	/* probably not set */
			}
			assert( IS_POINTER(idp) );

			if( ! POINTER_IS_SET(idp) ){
				break;
			}

			if( PTR_REF(ID_PTR(idp)) == NULL ){
				/* This sometimes happens during resolution, but not sure why??? */
				/*
	//			sprintf(ERROR_STRING,"CAUTIOUS:  resolve_tree %s:  ptr %s has null ref arg!?",
					node_desc(enp),ID_NAME(idp));
				warn(ERROR_STRING);
				*/
				break;
			}
			assert( REF_OBJ(PTR_REF(ID_PTR(idp))) != NULL );

			if( ! UNKNOWN_SHAPE( OBJ_SHAPE(REF_OBJ(PTR_REF(ID_PTR(idp)))) ) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree POINTER %s:  object %s has known shape",node_desc(enp),
OBJ_NAME(REF_OBJ(PTR_REF(ID_PTR(idp)))));
advise(ERROR_STRING);
describe_shape(OBJ_SHAPE(REF_OBJ(PTR_REF(ID_PTR(idp)))));
}
#endif /* QUIP_DEBUG */
				resolve_pointer(enp,OBJ_SHAPE(REF_OBJ(PTR_REF(ID_PTR(idp)))));
				return;
/*
advise("after resolution pointer:");
dump_tree(enp);
*/
			}

			break;

		case T_CALLFUNC:			/* resolve_tree */
							/* should we try a call-time resolution? */
/*
if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
sprintf(ERROR_STRING,"resolve_tree %s:  shape is unknown",node_desc(enp));
advise(ERROR_STRING);
} else {
sprintf(ERROR_STRING,"resolve_tree %s:  shape is KNOWN",node_desc(enp));
advise(ERROR_STRING);
}
*/

			srp = runnable_subrt(enp);
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree %s:  before setup_call %s:",node_desc(enp),SR_NAME(srp));
advise(ERROR_STRING);
dump_tree(enp);
}
#endif /* QUIP_DEBUG */
			/* BUG how can we get dst_dp?? */
			rip = setup_subrt_call(srp,enp,NULL);
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree %s:  after setup_call %s:",node_desc(enp),SR_NAME(srp));
advise(ERROR_STRING);
dump_tree(enp);
}
#endif /* QUIP_DEBUG */
			if( rip == NULL ){
				return;
			}

			if( rip->ri_arg_stat >= 0 ){ 
				eval_decl_tree(SR_BODY(srp));
				late_calltime_resolve(srp,NULL);
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
advise("resolve_tree:  after late_calltime_resolve:");
dump_tree(SR_BODY(srp));
}
#endif /* QUIP_DEBUG */
			}

			wrapup_context(rip);

			ret_enp = find_known_return(SR_BODY(VN_SUBRT(enp)));

			/* now maybe we have a return shape??? */
			if( ret_enp != NULL && ! UNKNOWN_SHAPE(VN_SHAPE(ret_enp)) ){
				copy_node_shape(enp,VN_SHAPE(ret_enp));
			}

			break;

		case T_RETURN:			/* resolve_tree */
			/* For a return node, we look up the subroutine and
			 * see if the return shape is known.
			 *
			 * Now that we have introduced the subrt_call struct, the return
			 * node only makes sense in relation to a call???
			 */
			/*
//			if( SR_DEST_SHAPE(VN_SUBRT(enp)) != NULL && SHP_PREC(SR_DEST_SHAPE(VN_SUBRT(enp))) != PREC_VOID ){
//				if( ! UNKNOWN_SHAPE(SR_DEST_SHAPE(VN_SUBRT(enp))) ){
//					// We know the return destination shape!
//					point_node_shape(enp,SR_DEST_SHAPE(VN_SUBRT(enp)));
//				}
//			}
//			*/
			// BUG not testing return shape!?

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
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			break;

		default:
			missing_case(enp,"resolve_tree (special cases)");
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
		update_tree_shape(enp);


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
advise(ERROR_STRING);
dump_tree(enp);
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
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	while(np!=NULL){
		resolver_enp = (Vec_Expr_Node *)NODE_DATA(np);
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree:  %s is a resolver node for %s",
node_desc(resolver_enp),node_desc(enp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

		assert( resolver_enp != NULL );

		if( resolver_enp != whence ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree %s:  trying %s",node_desc(enp),node_desc(resolver_enp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			if( VN_SHAPE(resolver_enp) == NULL ||	/* could be a void callfunc node */
				UNKNOWN_SHAPE(VN_SHAPE(resolver_enp)) ){

				resolve_tree(resolver_enp,enp);	// should this be enp or whence?
			}
			
			/* The above call to resolve_tree could have resolved resolver_enp */
			if( VN_SHAPE(resolver_enp)!=NULL && ! UNKNOWN_SHAPE(VN_SHAPE(resolver_enp)) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_tree %s:  effecting resolution using %s",node_desc(enp),node_desc(resolver_enp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
				/*resolved_enp=*/effect_resolution(enp,resolver_enp);
				if(VN_RESOLVERS(enp) != NULL ){
					/* careful - we dont want a recursive call, we may
					 * have gotten here from resolve_uk_nodes!
					 */
					resolve_uk_nodes(VN_RESOLVERS(enp));
				}
				return;
			}
		}
		np=NODE_NEXT(np);
	}
} /* end resolve_tree() */


/* We call resolve_subrt_call from early_calltime_resolve
 * Currently, we are calling this BEFORE arg val assignment...
 * What is the context assumed to be?
 */

void _resolve_subrt_call(QSP_ARG_DECL  Vec_Expr_Node *call_enp,List *uk_list, Shape_Info *ret_shpp)
{
	Subrt *srp;
	Subrt *save_srp;
	Vec_Expr_Node *argval_tree;
	int stat;
	Context_Pair *prev_cpp;

	srp = VN_SUBRT(call_enp);
	assert(srp!=NULL);
	argval_tree = VN_CHILD(call_enp,0);

	prev_cpp = pop_previous(SINGLE_QSP_ARG);

	save_srp = curr_srp;
	curr_srp = srp;

	SET_SR_FLAG_BITS(srp, SR_SCANNING);

	if( argval_tree != NULL ){
		stat = check_arg_shapes(SR_ARG_DECLS(srp),argval_tree,call_enp);
		if( stat < 0 ) {
sprintf(ERROR_STRING,"resolve_subrt_call %s:  argument error",SR_NAME(srp));
warn(ERROR_STRING);
			goto givup;
		}
	}

	/* set the context */

	set_subrt_ctx(SR_NAME(srp));

	/* declare the arg variables */
	eval_decl_tree(SR_ARG_DECLS(srp));	/* resolve_subrt_call() */
	eval_decl_tree(SR_BODY(srp));		/* resolve_subrt_call() */
	if( SR_PREC_CODE(srp) != PREC_VOID ){
		if( ret_shpp != NULL && ! UNKNOWN_SHAPE(ret_shpp) ){
			//SET_SC_SHAPE(scp, ret_shpp);
			copy_node_shape(call_enp,ret_shpp);
		} else {
			//SET_SC_SHAPE(scp, uk_shape(SR_PREC_CODE(srp)));
			copy_node_shape(call_enp, uk_shape(SR_PREC_CODE(srp)));
		}
	}

	/* we need to assign the arg vals for any ptr arguments! */

//	// I Don't understand why we set this and the set it back to NULL???
//	SET_SC_DEST_SHAPE(scp, ret_shpp);

	resolve_uk_nodes(uk_list);

	/* now try for the args */
	/* resolve_uk_args( */

	// scp now holds shape struct, not a ptr
	//SET_SC_DEST_SHAPE(scp, NULL);

	delete_subrt_ctx(SR_NAME(srp));

givup:

	CLEAR_SR_FLAG_BITS(srp, SR_SCANNING);

	curr_srp = save_srp;

	if( prev_cpp != NULL ){
/*
sprintf(ERROR_STRING,"resolve_subrt_call %s:  restoring context",SR_NAME(srp));
advise(ERROR_STRING);
*/
		restore_previous(prev_cpp);
	}

	/* we may have been passed uko arg vals,
	 * and our caller may expect us to resolve these...
	 * see if the corresponding arg decls have their shapes set...
	 */

	if( argval_tree != NULL ){
		resolve_argval_shapes(argval_tree,SR_ARG_DECLS(srp),srp);
	}

} /* end resolve_subrt_call() */

/* We call calltime_resolve before we call a subroutine,
 * before the arg vals are set.
 * We really need a way to call this (or something like it) AFTER the arg
 * vals have been set...
 *
 * Well, now we have the later version - why do we need the early one???
 */

void _early_calltime_resolve(QSP_ARG_DECL  Subrt *srp, Vec_Expr_Node *call_enp, Data_Obj *dst_dp)
{
	List *lp;
	int save_exec;
	uint32_t save_res;
    // If the args are not set, there's no reason to have a ptr to them...
	//Vec_Expr_Node *args_enp;

	//args_enp = VN_CHILD(call_enp,0);

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"Begin early_calltime_resolve %s",SR_NAME(srp));
advise(ERROR_STRING);
dump_subrt(srp);
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

	lp = get_uk_list(SR_BODY(srp));
	if( lp == NULL ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"early_calltime_resolve %s:  no UK nodes, returning",SR_NAME(srp));
advise(ERROR_STRING);
dump_tree(SR_BODY(srp));
}
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
sprintf(ERROR_STRING,"early_calltime_resolve calling resolve_subrt_call %s, flags = 0x%x, saved flags = 0x%x",
SR_NAME(srp),resolution_flags, save_res);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( dst_dp == NULL ){
		resolve_subrt_call(call_enp,lp,NULL);
	} else {
		resolve_subrt_call(call_enp,lp,OBJ_SHAPE(dst_dp));
	}

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
advise(ERROR_STRING);
dump_subrt(srp);
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
//fprintf(stderr,"xform_from_bitmap:\n");
//describe_shape(DEFAULT_QSP_ARG  shpp);
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

#define reshape_obj(dp,shpp) _reshape_obj(QSP_ARG  dp,shpp)

static Data_Obj * _reshape_obj(QSP_ARG_DECL  Data_Obj *dp,Shape_Info *shpp)
{
	Data_Obj *tmp_dp;
	char s[LLEN];

	/* the simplest thing is just to blow away the object and make a new one */
	/* Are we sure that we want to use a local (volatile) object here???  I don't think so... */
	/* we use localname to get a unique name in the short term, but we do NOT want
	 * to use make_local_dobj, because then it would be cleaned up at the end of the command...
	 */

	tmp_dp = make_dobj(localname(),SHP_TYPE_DIMS(shpp),SHP_PREC_PTR(shpp));
	assert( dp != NULL );
	assert( tmp_dp != NULL );

	strcpy(s,OBJ_NAME(dp));
	delvec(dp);
	obj_rename(tmp_dp,s);
	return(tmp_dp);
}


/* resolve_obj_id
 * 
 * resolve an object given its identifier
 */

#define resolve_obj_id(idp, shpp) _resolve_obj_id(QSP_ARG  idp, shpp)

static void _resolve_obj_id(QSP_ARG_DECL  Identifier *idp, Shape_Info *shpp)
{
	Vec_Expr_Node *decl_enp;
	Shape_Info *tmp_shpp;
	Data_Obj *dp;
/* char remember_name[LLEN]; */

	assert( shpp != NULL );
    
	INIT_SHAPE_PTR(tmp_shpp)

	/* decl_enp = OBJ_EXTRA(dp); */

	/* matlab has no declarations!? */
	if( mode_is_matlab ){
/*
sprintf(ERROR_STRING,"resolve_obj_id %s:  mode is matlab",ID_NAME(idp));
advise(ERROR_STRING);
*/
		dp = get_obj(ID_NAME(idp));

		assert( dp != NULL );

		REF_OBJ(ID_REF(idp)) = reshape_obj(dp,shpp);

		return;
	}

	assert( ID_TYPE(idp) == ID_OBJ_REF );

	decl_enp = REF_DECL_VN(ID_REF(idp));
	dp = REF_OBJ(ID_REF(idp));

	assert( OBJ_EXTRA(dp) == decl_enp );

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_obj_id:  setting decl node shape for %s", VN_STRING(decl_enp));
advise(ERROR_STRING);
describe_shape(shpp);
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
advise(ERROR_STRING);

sprintf(ERROR_STRING,"resolve_obj_id:  calling reeval_decl_stat %s", VN_STRING(decl_enp));
advise(ERROR_STRING);
if( shpp == NULL ) warn("shape ptr is null!?");
describe_shape(shpp);
}
#endif /* QUIP_DEBUG */

	/* the pointer points to the object, but the object's name may be in a different
	 * context.  We need to reinstate the context before reevaluating the declaration.
	 * We do this by saving the identifier context in the declaration node.
	 */
	reeval_decl_stat(OBJ_PREC_PTR(dp),decl_enp,VN_DECL_FLAGS(decl_enp));

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

void _resolve_pointer(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Shape_Info *shpp)
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

	/* idp = eval_ptr_expr(uk_enp,1); */
	idp = get_set_ptr(uk_enp);

	if( idp == NULL ){
/*
node_error(uk_enp);
sprintf(ERROR_STRING,"resolve_pointer:  no set identifier for %s",node_desc(uk_enp));
advise(ERROR_STRING);
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
advise(ERROR_STRING);
		resolve_obj_id(REF_ID(PTR_REF(ID_PTR(idp))),shpp);
	} else {
		/* make sure that the shapes are the same */
		if( ! same_shape(OBJ_SHAPE(dp),shpp) ){
			node_error(uk_enp);
			sprintf(ERROR_STRING,
		"resolve_pointer:  Shape mismatch with object %s",OBJ_NAME(dp));
			warn(ERROR_STRING);
describe_shape(VN_SHAPE(uk_enp));
describe_shape(shpp);
		}
	}

/*
sprintf(ERROR_STRING,"resolve_pointer fixing %s",node_desc(uk_enp));
advise(ERROR_STRING);
*/

	/* also fix the node where we found this */
	point_node_shape(uk_enp,shpp);
}



/* We call resolve_object with a T_DYN_OBJ node.
 * The shape pointer should point to a valid shape...
 */

#define resolve_object(uk_enp,shpp) _resolve_object(QSP_ARG  uk_enp,shpp)

static void _resolve_object(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Shape_Info *shpp)
{
	Identifier *idp;
	Data_Obj *dp;

	/* If this is an automatic object, and we haven't evaluated the decls yet... */
	// Why do we think the object name is in VN_STRING?
dump_tree(uk_enp);
	
	switch(VN_CODE(uk_enp)){
		case T_DYN_OBJ:
fprintf(stderr,"resolve_object passed unknown dynamic object %s\n",VN_STRING(uk_enp));
			idp = get_id(VN_STRING(uk_enp));
			assert( idp != NULL );
			assert( IS_OBJ_REF(idp) );

			dp = REF_OBJ(ID_REF(idp));
			assert( dp != NULL );
			break;

		case T_STATIC_OBJ:
			dp = VN_OBJ(uk_enp);
			idp = get_id(OBJ_NAME(dp));
			break;

		default:
			missing_case(uk_enp,"resolve_object");
			return;
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
	resolve_obj_id(idp,shpp);

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_object %s:",node_desc(uk_enp));
advise(ERROR_STRING);
describe_shape(VN_SHAPE(uk_enp));
}
#endif /* QUIP_DEBUG */

} /* end resolve_object */


/* here we pass a shape with which to resolve the children (scalar) nodes */

#define resolve_obj_list(enp, shpp) _resolve_obj_list(QSP_ARG  enp, shpp)

static void _resolve_obj_list(QSP_ARG_DECL  Vec_Expr_Node *enp, Shape_Info *shpp)
{
	switch(VN_CODE(enp)){
		case T_RET_LIST:
			resolve_obj_list(VN_CHILD(enp,0),shpp);
			resolve_obj_list(VN_CHILD(enp,1),shpp);
			break;

		case T_DYN_OBJ:
			resolve_node(enp,shpp);
			break;

		default:
			missing_case(enp,"resolve_obj_list");
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

Vec_Expr_Node *_resolve_node(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Shape_Info *shpp)
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
			point_node_shape(uk_enp,shpp);
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
			resolve_pointer(uk_enp,shpp);

			/* We can't propagate from the pointer node... */
			return(NULL);

		/*
		case T_STR_PTR:
		case T_POINTER:
		*/
		case T_STATIC_OBJ:
		case T_DYN_OBJ:			/* resolve_node */
			resolve_object(uk_enp,shpp);
			return(uk_enp);

		case T_RET_LIST:		/* a list of objects */
			if( ROWVEC_SHAPE(shpp) ){	/* a list of scalars? */
				resolve_obj_list(uk_enp,scalar_shape(SHP_PREC(shpp)));
			} else {
sprintf(ERROR_STRING,"resolve_node %s:  oops, only know how to resolve from rowvec!?",node_desc(uk_enp));
warn(ERROR_STRING);
			}
			return(uk_enp);

		case T_VV_VV_CONDASS:
		case T_VS_VV_CONDASS:
		case T_SS_VV_CONDASS:
		case T_VV_VS_CONDASS:
		case T_VS_VS_CONDASS:
		case T_SS_VS_CONDASS:
		case T_VV_SS_CONDASS:
		case T_VS_SS_CONDASS:

		case T_SS_B_CONDASS:
			/* The known child is the bitmap */
		case T_VS_B_CONDASS:
		case T_VV_B_CONDASS:
			/* The known child may be the bitmap, but hopefully the
			 * correct shape was determined above.
			 */
			copy_node_shape(uk_enp,shpp);
			return(uk_enp);


		default:
			missing_case(uk_enp,"resolve_node");
			break;
	}
	return(NULL);
} /* end resolve_node */

/* The parent is the unknown node? */

#define resolve_parent(uk_enp,shpp) _resolve_parent(QSP_ARG  uk_enp,shpp)

static Vec_Expr_Node *_resolve_parent(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Shape_Info *shpp)
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

		case T_VV_VV_CONDASS:
		case T_VS_VV_CONDASS:
		case T_SS_VV_CONDASS:
		case T_VV_VS_CONDASS:
		case T_VS_VS_CONDASS:
		case T_SS_VS_CONDASS:
		case T_VV_SS_CONDASS:
		case T_VS_SS_CONDASS:
			copy_node_shape(uk_enp,shpp);
			return(uk_enp);
			break;

		case T_SS_B_CONDASS:	/* resolve_parent, branch shape is known */
		case T_VS_B_CONDASS:	/* resolve_parent, branch shape is known */
		case T_VV_B_CONDASS:	/* resolve_parent, branch shape is known */
			prec = VN_PREC(uk_enp);
			copy_node_shape(uk_enp,shpp);
			if( BITMAP_SHAPE(shpp) )
				xform_from_bitmap(VN_SHAPE(uk_enp),prec);
			return(uk_enp);
			break;

		case T_VS_FUNC:
		ALL_UNARY_CASES
		ALL_MATHFN_CASES
		case T_FIX_SIZE:
		case T_DEREFERENCE:
		case T_ASSIGN:
		case T_TYPECAST:
		case T_RETURN:
			return( resolve_node(uk_enp,shpp) );
			break;

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
			return( resolve_node(uk_enp,shpp) );
			break;

		default:
			missing_case(uk_enp,"resolve_parent");
			return( resolve_node(uk_enp,shpp) );
			break;
	}
	return(NULL);
} /* resolve_parent */

/* What is the difference between resolve_unknown_parent and resolve_parent???
 */

#define resolve_unknown_parent(uk_enp,enp) _resolve_unknown_parent(QSP_ARG  uk_enp,enp)

static Vec_Expr_Node * _resolve_unknown_parent(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp)
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
			missing_case(enp,"resolve_unknown_parent (known node)");
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
		describe_shape(shpp);
	}
}
#endif /* QUIP_DEBUG */

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_unknown_parent %s from %s",node_desc(uk_enp),node_desc(enp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	/* BUG we'd like to merge this with resolve_node... */

	return( resolve_parent(uk_enp,shpp));
} /* end resolve_unknown_parent */


/* shape_for_child - what does this do???
 */

#define shape_for_child(uk_enp,enp) _shape_for_child(QSP_ARG  uk_enp,enp)

static Shape_Info *_shape_for_child(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp)
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
			missing_case(enp,"shape_for_child");
			return( VN_SHAPE(enp) );
	}
} /* end shape_for_child */

#define resolve_unknown_child(uk_enp,enp) _resolve_unknown_child(QSP_ARG  uk_enp,enp)

static Vec_Expr_Node * _resolve_unknown_child(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp)
{
	Shape_Info *shpp;

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"resolve_unknown_child %s from %s",node_desc(uk_enp),node_desc(enp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	shpp = shape_for_child(uk_enp,enp);

	if( mode_is_matlab && VN_CODE(uk_enp) == T_RET_LIST ){
		//Data_Obj *dp;

		/*dp =*/ eval_obj_ref(uk_enp);	/* force object creation */
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
		describe_shape(shpp);
	}
}
#endif /* QUIP_DEBUG */

	return(resolve_node(uk_enp,shpp));
} /* resolve_unknown_child */

#ifdef NOT_USED

/* resolve_return
 *
 * We call this when we resolve a return node from the calling CALLFUNC node - these
 * are not linked, although the program would be simpler if the could be !? (BUG?)
 */

void resolve_return(QSP_ARG_DECL  Vec_Expr_Node *enp,Shape_Info *shpp)
{
	point_node_shape(enp,shpp);
}

#endif /* NOT_USED */

/* effect_resolution
 *
 * The main entry point to this module.
 * We have a pair of nodes which can resolve each other, and one of
 * them is now known.
 * Returns a ptr to the resolved node, or NULL
 */

Vec_Expr_Node * _effect_resolution(QSP_ARG_DECL  Vec_Expr_Node *uk_enp,Vec_Expr_Node *enp)
{
	Vec_Expr_Node *ret_enp=NULL;

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"effect_resolution of %s from %s",node_desc(uk_enp),node_desc(enp));
advise(ERROR_STRING);
advise("known shape:");
describe_shape(VN_SHAPE(enp));
}
#endif /* QUIP_DEBUG */

	/* BUG?  do we need to save the old value of the flags? */
	if( VN_FLAGS(enp) & NODE_NEEDS_CALLTIME_RES )
		resolution_flags |= NODE_NEEDS_CALLTIME_RES;

	/* We know that one of these nodes is the parent of the
	 * other - which one is which ?
	 */
	
	if( VN_PARENT(enp) == uk_enp ) ret_enp=resolve_unknown_parent(uk_enp,enp);
	else if( VN_PARENT(uk_enp) == enp ) ret_enp=resolve_unknown_child(uk_enp,enp);
	else {
		resolve_node(uk_enp,VN_SHAPE(enp));
	}


#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"After effect_resolution of %s from %s:",node_desc(uk_enp),node_desc(enp));
advise(ERROR_STRING);
advise("known shape:");
describe_shape(VN_SHAPE(enp));
advise("resolved shape:");
describe_shape(VN_SHAPE(uk_enp));
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

void _resolve_argval_shapes(QSP_ARG_DECL  Vec_Expr_Node *val_enp,Vec_Expr_Node *decl_enp,Subrt *srp)
{
	if( decl_enp == NULL ) return;

	switch(VN_CODE(decl_enp)){
		case T_DECL_STAT_LIST:
			resolve_argval_shapes(VN_CHILD(val_enp,0),
				VN_CHILD(decl_enp,0),srp);
			resolve_argval_shapes(VN_CHILD(val_enp,1),
				VN_CHILD(decl_enp,1),srp);
			break;
		case T_DECL_STAT:
			resolve_argval_shapes(val_enp,
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
advise("resolve_argval_shapes:  propagating a shape!!");
describe_shape(VN_SHAPE(decl_enp));
}
#endif /* QUIP_DEBUG */
			propagate_shape(val_enp,VN_SHAPE(decl_enp));
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
advise("resolve_argval_shapes:  DONE");
describe_shape(VN_SHAPE(val_enp));
}
#endif /* QUIP_DEBUG */
			break;
			
			
		default:
			missing_case(decl_enp,"resolve_argval_shapes");
			break;
	}
}

/* propagate_shape
 *
 * Descend a tree, assigning shapes when possible.
 */

void _propagate_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,Shape_Info *shpp)
{
	Shape_Info *tmp_shpp;

	INIT_SHAPE_PTR(tmp_shpp)

	assert( UNKNOWN_SHAPE(VN_SHAPE(enp)) );

	switch(VN_CODE(enp)){
		case T_POINTER:		/* propagate_shape */
			resolve_pointer(enp,shpp);
			break;

		case T_STR_PTR:
		case T_DYN_OBJ:
			resolve_object(enp,shpp);
			break;

		case T_CURLY_SUBSCR:
			COPY_SHAPE(tmp_shpp , shpp);
			SET_SHP_TYPE_DIM(tmp_shpp,SHP_MINDIM(tmp_shpp),1);
			SET_SHP_MACH_DIM(tmp_shpp,SHP_MINDIM(tmp_shpp),1);
			SET_SHP_MINDIM(tmp_shpp,
				SHP_MINDIM(tmp_shpp) + 1 );
			propagate_shape(VN_CHILD(enp,0),tmp_shpp);
			break;

		case T_SQUARE_SUBSCR:
			COPY_SHAPE(tmp_shpp , shpp);
			SET_SHP_TYPE_DIM(tmp_shpp,SHP_MAXDIM(tmp_shpp),1);
			SET_SHP_MACH_DIM(tmp_shpp,SHP_MAXDIM(tmp_shpp),1);
			SET_SHP_MAXDIM(tmp_shpp,
				SHP_MAXDIM(tmp_shpp) - 1 );
			propagate_shape(VN_CHILD(enp,0),tmp_shpp);
			break;

		case T_VV_FUNC:
			propagate_shape(VN_CHILD(enp,0),shpp);
			propagate_shape(VN_CHILD(enp,1),shpp);
			break;

		default:
			missing_case(enp,"propagate_shape");
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
			missing_case(enp,"arg_sizes_known");
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
 * We call it (from resolve_subrt_call and setup_call) before setting the context of the new
 * subroutine.
 */

int _check_arg_shapes(QSP_ARG_DECL  Vec_Expr_Node *arg_enp,Vec_Expr_Node *val_enp,Vec_Expr_Node *call_enp)
{
	int stat;
	Subrt *srp;

	srp = VN_SUBRT(call_enp);	// works for reffunc???

	if( arg_enp == NULL ) return(0);
	else if( val_enp == NULL ){
		/* BUG we want to report this error at the line number of the callfunc node */
		sprintf(ERROR_STRING,"check_arg_shapes:  Subroutine %s requires arguments",SR_NAME(srp));
		warn(ERROR_STRING);
		return(-1);
	}

	switch(VN_CODE(arg_enp)){
		case  T_DECL_STAT:
			/* en_intval is the type (float,short,etc) */
			stat=check_arg_shapes(VN_CHILD(arg_enp,0), val_enp,call_enp);
			return(stat);

		case T_DECL_STAT_LIST:
			/* val_enp should be T_ARGLIST */
			assert( VN_CODE(val_enp) == T_ARGLIST );

			stat=check_arg_shapes(VN_CHILD(arg_enp,0), VN_CHILD(val_enp,0),call_enp);
			if( stat < 0 ) return(stat);
			stat=check_arg_shapes(VN_CHILD(arg_enp,1), VN_CHILD(val_enp,1),call_enp);
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
advise(ERROR_STRING);
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
				propagate_shape(val_enp,VN_SHAPE(arg_enp));
			} else {	/* everything is known */
				if( !shapes_match(VN_SHAPE(arg_enp),
							VN_SHAPE(val_enp)) ){
					node_error(call_enp);
					sprintf(ERROR_STRING,
	"subrt %s:  argument shape mismatch",SR_NAME(srp));
					warn(ERROR_STRING);
advise("argument prototype shape:");
describe_shape(VN_SHAPE(arg_enp));
advise("argument value shape:");
describe_shape(VN_SHAPE(val_enp));
dump_tree(arg_enp);

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
advise(VN_STRING(arg_enp));
describe_shape(VN_SHAPE(arg_enp));
describe_shape(VN_SHAPE(val_enp));
}
#endif /* QUIP_DEBUG */
					return(-1);
				}
			}
			return(0);

		default:
			missing_case(arg_enp,"check_arg_shapes");
			return(-1);
	}
}

