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

#ifdef HAVE_STRING_H
#include <string.h>
#endif


#include "quip_prot.h"
#include "data_obj.h"
//#include "fio_api.h"
#include "vectree.h"
#include "veclib_api.h"

/* global var */
debug_flag_t cast_debug=0;

static Shape_Info *_cpx_scalar_shpp[N_NAMED_PRECS];

/* There are 8 machine precisions, so we can represent the dominance
 * relations using an 8x8 table.
 * Here, we set the dominant precision to always be one of the two we
 * start with.  But might we want BY,UIN to be dominated by IN or UDI???
 */

static prec_t dominance_tbl[N_MACHINE_PRECS][N_MACHINE_PRECS];
static int dominance_table_inited=0;

/* local prototypes - needed because of circularities, recursion */

// we tried to eliminate these, but apparently there are some circularities?
// The cycle seems to be related to matlab?

static void _prelim_node_shape(QSP_ARG_DECL Vec_Expr_Node *enp);
#ifdef SUPPORT_MATLAB_MODE
static Vec_Expr_Node * _compile_node(QSP_ARG_DECL   Vec_Expr_Node *enp);
#endif /* SUPPORT_MATLAB_MODE */

#define check_bitmap_arg(enp,index)		_check_bitmap_arg(QSP_ARG enp,index)
#define check_xx_v_condass_code(enp)		_check_xx_v_condass_code(QSP_ARG enp)
#define check_xx_s_condass_code(enp)		_check_xx_s_condass_code(QSP_ARG enp)
#define check_xx_xx_condass_code(enp)		_check_xx_xx_condass_code(QSP_ARG enp)
#define promote_child(enp,i1,i2)		_promote_child(QSP_ARG enp,i1,i2)
#define check_typecast(enp,i1,i2)		_check_typecast(QSP_ARG  enp,i1,i2)
#define get_mating_shapes(enp,i1,i2)		_get_mating_shapes(QSP_ARG   enp,i1,i2)
#define check_binop_links(enp)			_check_binop_links(QSP_ARG  enp)
#define check_mating_bitmap(enp,bm_enp)  _check_mating_bitmap(QSP_ARG  enp, bm_enp)
#define one_matlab_subscript(obj_enp,subscr_enp)	_one_matlab_subscript(QSP_ARG   obj_enp,subscr_enp)
#define compile_matlab_subscript(obj_enp,subscr_enp)	_compile_matlab_subscript(QSP_ARG  obj_enp,subscr_enp)

#define update_node_shape(enp)		_update_node_shape(QSP_ARG  enp)
#define prelim_node_shape(enp)		_prelim_node_shape(QSP_ARG enp)
#define compile_node(enpp)		_compile_node(QSP_ARG   enpp);
#define check_minmax_code(enp)		_check_minmax_code(QSP_ARG  enp)
#define count_lhs_refs(enp)		_count_lhs_refs(QSP_ARG enp)
#define count_name_refs( enp, name )	_count_name_refs(QSP_ARG  enp , name )
#define shapes_mate(enp1,enp2,enp)	_shapes_mate(QSP_ARG  enp1,enp2,enp)
#define typecast_child(enp,index,prec)		_typecast_child(QSP_ARG enp,index,prec)

int mode_is_matlab=0;
static Vec_Expr_Node *minus_one_enp=NULL;
static Shape_Info *_scalar_shpp[N_NAMED_PRECS];
static Shape_Info *_uk_shpp[N_NAMED_PRECS];
static Shape_Info *_void_shpp;

#include "vt_native.h"

void (*native_prelim_func)(QSP_ARG_DECL  Vec_Expr_Node *)=prelim_vt_native_shape;
void (*native_update_func)(Vec_Expr_Node *)=update_vt_native_shape;

#define MULTIPLY_DIMENSIONS(p,dsp)			\
							\
	p  = DIMENSION(dsp,0);				\
	p *= DIMENSION(dsp,1);				\
	p *= DIMENSION(dsp,2);				\
	p *= DIMENSION(dsp,3);				\
	p *= DIMENSION(dsp,4);

//#define REDUCE_MINDIM(shpp)	REDUCE_DIMENSION(shpp,si_mindim,++)

#define REDUCE_MINDIM(shpp)					\
								\
	SHP_N_MACH_ELTS(shpp) = SHP_N_MACH_ELTS(shpp) / SHP_MACH_DIM(shpp,SHP_MINDIM(shpp)); \
	SHP_N_TYPE_ELTS(shpp) = SHP_N_TYPE_ELTS(shpp) / SHP_TYPE_DIM(shpp,SHP_MINDIM(shpp)); \
	SET_SHP_MACH_DIM(shpp,SHP_MINDIM(shpp),1);				\
	SET_SHP_TYPE_DIM(shpp,SHP_MINDIM(shpp),1);				\
	SET_SHP_MINDIM(shpp,SHP_MINDIM(shpp)+1);

//#define REDUCE_MAXDIM(shpp)	REDUCE_DIMENSION(shpp,si_maxdim,--)
#define REDUCE_MAXDIM(shpp)					\
								\
	SHP_N_MACH_ELTS(shpp) = SHP_N_MACH_ELTS(shpp) / SHP_MACH_DIM(shpp,SHP_MAXDIM(shpp)); \
	SHP_N_TYPE_ELTS(shpp) = SHP_N_TYPE_ELTS(shpp) / SHP_TYPE_DIM(shpp,SHP_MAXDIM(shpp)); \
	SET_SHP_MACH_DIM(shpp,SHP_MAXDIM(shpp),1);				\
	SET_SHP_TYPE_DIM(shpp,SHP_MAXDIM(shpp),1);				\
	SET_SHP_MAXDIM(shpp,SHP_MAXDIM(shpp)-1);



//			UPDATE_DIM_LEN(VN_SHAPE(enp),si_mindim,len,++)
//			UPDATE_DIM_LEN(VN_SHAPE(enp),si_maxdim,len,--)

#ifdef FOOBAR	// not needed?
#define UPDATE_MAXDIM_LEN(shpp,new_len)			\
									\
	SET_SHP_N_TYPE_ELTS(shpp, new_len * (SHP_N_TYPE_ELTS(shpp) / SHP_TYPE_DIM(shpp,SHP_MAXDIM(shpp))) );	\
	SET_SHP_N_MACH_ELTS(shpp, new_len * (SHP_N_MACH_ELTS(shpp) / SHP_MACH_DIM(shpp,SHP_MAXDIM(shpp))) );	\
	SET_SHP_TYPE_DIM(shpp,SHP_MAXDIM(shpp),new_len);		\
	SET_SHP_MACH_DIM(shpp,SHP_MAXDIM(shpp),new_len);		\
	SET_SHP_MAXDIM(shpp,SHP_MAXDIM(shpp)-1);

#define UPDATE_MINDIM_LEN(shpp,new_len)			\
									\
	SET_SHP_N_TYPE_ELTS(shpp, new_len * (SHP_N_TYPE_ELTS(shpp) / SHP_TYPE_DIM(shpp,SHP_MINDIM(shpp))) );	\
	SET_SHP_N_MACH_ELTS(shpp, new_len * (SHP_N_MACH_ELTS(shpp) / SHP_MACH_DIM(shpp,SHP_MINDIM(shpp))) );	\
	SET_SHP_TYPE_DIM(shpp,SHP_MINDIM(shpp),new_len);		\
	SET_SHP_MACH_DIM(shpp,SHP_MINDIM(shpp),new_len);		\
	SET_SHP_MINDIM(shpp,SHP_MINDIM(shpp)+1);
#endif // FOOBAR


#define UPDATE_RANGE_MAXDIM_LEN(shpp,new_len)			\
									\
	SET_SHP_N_TYPE_ELTS(shpp, new_len * (SHP_N_TYPE_ELTS(shpp) / SHP_TYPE_DIM(shpp,SHP_RANGE_MAXDIM(shpp))) );	\
	SET_SHP_N_MACH_ELTS(shpp, new_len * (SHP_N_MACH_ELTS(shpp) / SHP_MACH_DIM(shpp,SHP_RANGE_MAXDIM(shpp))) );	\
	SET_SHP_TYPE_DIM(shpp,SHP_RANGE_MAXDIM(shpp),new_len);		\
	SET_SHP_MACH_DIM(shpp,SHP_RANGE_MAXDIM(shpp),new_len);		\
	SET_SHP_RANGE_MAXDIM(shpp,SHP_RANGE_MAXDIM(shpp)-1);

#define UPDATE_RANGE_MINDIM_LEN(shpp,new_len)			\
									\
	SET_SHP_N_TYPE_ELTS(shpp, new_len * (SHP_N_TYPE_ELTS(shpp) / SHP_TYPE_DIM(shpp,SHP_RANGE_MINDIM(shpp))) );	\
	SET_SHP_N_MACH_ELTS(shpp, new_len * (SHP_N_MACH_ELTS(shpp) / SHP_MACH_DIM(shpp,SHP_RANGE_MINDIM(shpp))) );	\
	SET_SHP_TYPE_DIM(shpp,SHP_RANGE_MINDIM(shpp),new_len);		\
	SET_SHP_MACH_DIM(shpp,SHP_RANGE_MINDIM(shpp),new_len);		\
	SET_SHP_RANGE_MINDIM(shpp,SHP_RANGE_MINDIM(shpp)+1);


static Precision *parent_decl_prec(Vec_Expr_Node *enp)
{
	enp = VN_PARENT(enp);
	if( enp == NULL ) return NULL;
	if( VN_DECL_PREC(enp) == NULL )
		return parent_decl_prec(enp);
	else
		return VN_DECL_PREC(enp);
}

/* return the index of this child among its siblings, or -1 if no parent */
static int which_child(Vec_Expr_Node *enp)
{
	int i;

	if( VN_PARENT(enp) == NO_VEXPR_NODE ) return(-1);

	for(i=0;i<tnt_tbl[ VN_CODE(VN_PARENT(enp))].tnt_nchildren;i++){
		if( VN_CHILD(VN_PARENT(enp),i) == enp ) return(i);
	}
//#ifdef CAUTIOUS
//	NWARN("CAUTIOUS:  which_child:  node not found among parent's children!?!?");
//#endif /* CAUTIOUS */
	assert( AERROR("not not found among parent's children!?") );
	return(-1);
}

/* Set the shape of this node, by copying the data into this
 * node's personal structure.  The node should already own
 * it's own shape info...
 *
 * The underscore version has the qsp arg...
 */

void _copy_node_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,Shape_Info *shpp)
{

#ifdef CAUTIOUS
	if( ! NODE_SHOULD_OWN_SHAPE(enp) ){
		sprintf(ERROR_STRING,"CAUTIOUS:  copy_node_shape %s:  node shouldn't own shape!?",
			node_desc(enp));
		WARN(ERROR_STRING);
	}
#endif /* CAUTIOUS */
	assert( NODE_SHOULD_OWN_SHAPE(enp) );

	if( ! OWNS_SHAPE(enp) ){
		SET_VN_SHAPE(enp, ALLOC_SHAPE );
		SET_VN_FLAG_BITS(enp, NODE_IS_SHAPE_OWNER);
	}

//#ifdef CAUTIOUS
//	if( VN_SHAPE(enp) == NO_SHAPE ){
//		WARN("CAUTIOUS:  copy_node_shape:  can't copy to null ptr!?");
//		DUMP_TREE(enp);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( VN_SHAPE(enp) != NO_SHAPE );

#ifdef CAUTIOUS
	if( shpp == NO_SHAPE ){
		WARN("CAUTIOUS:  copy_node_shape:  can't copy from null ptr!?");
		DUMP_TREE(enp);
		//return;
	}
#endif /* CAUTIOUS */

	assert( shpp != NO_SHAPE );

	COPY_VN_SHAPE(enp,shpp);
}

/* Release a nodes shape struct */

void discard_node_shape(Vec_Expr_Node *enp)
{
	if( VN_SHAPE(enp) == NO_SHAPE ) return;

	if( OWNS_SHAPE(enp) ){
		rls_shape(VN_SHAPE(enp));
		CLEAR_VN_FLAG_BITS(enp, NODE_IS_SHAPE_OWNER);
	}
	SET_VN_SHAPE(enp,NO_SHAPE);
}

/* Set the shape of this node, by setting it's pointer to the arg.
 * The node should NOT already own it's own shape info...
 */

void point_node_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,Shape_Info *shpp)
{

/*
if( e VN_CODE(np) == T_DYN_OBJ ){
sprintf(ERROR_STRING,"point_node_shape OBJECT %s:  shape at 0x%lx",node_desc(enp),(u_long)shpp);
advise(ERROR_STRING);
DESCRIBE_SHAPE(shpp);
}
*/

#ifdef CAUTIOUS
	if( ! NODE_SHOULD_PT_TO_SHAPE(enp) ){
		sprintf(ERROR_STRING,"CAUTIOUS:  point_node_shape %s:  node shouldn't pt to shape!?",
			node_desc(enp));
		WARN(ERROR_STRING);
		DUMP_TREE(enp);
	}
#endif /* CAUTIOUS */
	assert( NODE_SHOULD_PT_TO_SHAPE(enp) );

	if( OWNS_SHAPE(enp) ){
		NODE_ERROR(enp);
		sprintf(ERROR_STRING,
	"point_node_shape:  %s node n%d already owns shape info!?",NNAME(enp),VN_SERIAL(enp));
		WARN(ERROR_STRING);
		/*
		DESCRIBE_SHAPE(VN_SHAPE(enp));
		*/
		discard_node_shape(enp);
	}
//sprintf(ERROR_STRING,"point_node_shape %s 0x%lx",node_desc(enp),(long)shpp);
//advise(ERROR_STRING);
//dump_tree(enp);
	SET_VN_SHAPE(enp, shpp);
}


#ifdef FOOBAR
/* We call this to find out if an expression can be evaluated before runtime */

static int is_variable(Vec_Expr_Node *enp)
{
	int i;

	if( enp == NO_VEXPR_NODE ) return(0);

	if( VN_CODE(enp) == T_DYN_OBJ ) return(1);

	for(i=0;i<MAX_CHILDREN(enp);i++){
		if( VN_CHILD(enp,i) != NO_VEXPR_NODE ){
			if( is_variable(VN_CHILD(enp,i)) )
				return(1);
		}
	}
	return(0);
}
#endif /* FOOBAR */


/* This is a special case of check_uk_links */

#define CHECK_UK_CHILD(enp,index)	check_uk_child(QSP_ARG  enp,index)

static void check_uk_child(QSP_ARG_DECL  Vec_Expr_Node *enp,int index)
{
	if( ( UNKNOWN_SHAPE(VN_SHAPE(enp)) && ! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,index)) )
			|| UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,index)) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(DEFAULT_ERROR_STRING,"check_uk_child calling link_uk_nodes for %s and child %s",
node_desc(enp),node_desc(VN_CHILD(enp,index)));
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		LINK_UK_NODES(enp,VN_CHILD(enp,index));
	}
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

	enp2=VN_CHILD(enp,child_index);
	shpp=VN_SHAPE(enp2);

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

		if( VN_CODE(enp2) == T_DYN_OBJ ){	// get_child_shape
			Data_Obj *dp;

			dp=DOBJ_OF(VN_STRING(enp2));
			if( dp == NO_OBJ ){
				sprintf(ERROR_STRING,
					"Missing obj %s",VN_STRING(enp2));
				WARN(ERROR_STRING);
			} else {
				/* BUG should we use the decl node?? */
				shpp = OBJ_SHAPE(dp);
				return(shpp);
			}
		}

		discard_node_shape(enp);

		return( NO_SHAPE );
	}
	return(shpp);
}

static int get_subvec_indices(QSP_ARG_DECL Vec_Expr_Node *enp, dimension_t *i1p, dimension_t *i2p )
{
	int which_dim=0;	/* auto-init just to silence compiler */

	if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ) return(0);

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

	if( VN_CHILD(enp,1) == NO_VEXPR_NODE ){
		*i1p=0;
	} else {
		if( (!HAS_CONSTANT_VALUE(VN_CHILD(enp,1))) ){
			if( !executing ){
			/* only evaluate variable expressions at run-time */
				return(0);
			} else {
				SET_VN_FLAG_BITS(enp, NODE_NEEDS_CALLTIME_RES);
			}
		}
		*i1p = (dimension_t) EVAL_INT_EXP(VN_CHILD(enp,1));
	}

	if( VN_CODE(enp)==T_SUBVEC ){	// get_subvec_indices
		//which_dim = SHP_MAXDIM(VN_CHILD_SHAPE(enp,0)) ;
		which_dim = SHP_RANGE_MAXDIM(VN_CHILD_SHAPE(enp,0)) ;
	} else if( VN_CODE(enp)==T_CSUBVEC ){
		which_dim = SHP_RANGE_MINDIM(VN_CHILD_SHAPE(enp,0)) ;
	}
//#ifdef CAUTIOUS
	  else {
//	  	ERROR1("CAUTIOUS:  get_subvec_indices:  unexpected node code");
		assert( AERROR("get_subvec_indices:  unexpected node code") );
	}
//#endif /* CAUTIOUS */

	if( VN_CHILD(enp,2) == NO_VEXPR_NODE ){
		*i2p=SHP_TYPE_DIM(VN_CHILD_SHAPE(enp,0),which_dim) -1;
	} else {
		if( (!HAS_CONSTANT_VALUE(VN_CHILD(enp,2))) ){
			if( !executing ){
				/* only evaluate variable expressions at run-time */
				return(0);
			} else {
				SET_VN_FLAG_BITS(enp, NODE_NEEDS_CALLTIME_RES);
			}
		}
		*i2p = (dimension_t) EVAL_INT_EXP(VN_CHILD(enp,2));
	}

	if( *i1p > *i2p ){
		NODE_ERROR(enp);
		sprintf(ERROR_STRING,"first range index (%d) should be less than or equal to second range index (%d)",
		*i1p,*i2p);
		WARN(ERROR_STRING);
		return(0);
	}

	return(1);
}

static Shape_Info *void_shape(void)
{
	Shape_Info *shpp;

	if( _void_shpp != NO_SHAPE )
		return(_void_shpp);

	shpp=uk_shape(PREC_VOID);
	//_void_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
	_void_shpp = alloc_shape();
	*_void_shpp = *shpp;
	SET_SHP_FLAGS(_void_shpp, DT_VOID);
	return(_void_shpp);
}

// We will also use this to size temp objects for GPU operations,
// perhaps should be in a different library?

Shape_Info *make_outer_shape(QSP_ARG_DECL  Shape_Info *shpp1, Shape_Info *shpp2)
{
	int i;
	static Shape_Info ret_shp;
	Shape_Info *shpp=(&ret_shp);

	//INIT_SHAPE_PTR(shpp)	// BUG?  memory leak?

	SET_SHP_N_TYPE_ELTS(shpp,1);
	for(i=0;i<N_DIMENSIONS;i++){
		if( SHP_TYPE_DIM(shpp1,i) == 1 ){
			SET_SHP_TYPE_DIM(shpp,i, SHP_TYPE_DIM(shpp2,i) );
		} else if( SHP_TYPE_DIM(shpp2,i) == 1 ){
			SET_SHP_TYPE_DIM(shpp,i, SHP_TYPE_DIM(shpp1,i) );
		} else if( SHP_TYPE_DIM(shpp1,i) == SHP_TYPE_DIM(shpp2,i) ){
			SET_SHP_TYPE_DIM(shpp,i, SHP_TYPE_DIM(shpp1,i) );
		} else {
			/* mismatch */
			return NULL;
		}
		SET_SHP_N_TYPE_ELTS(shpp,
			SHP_N_TYPE_ELTS(shpp) * SHP_TYPE_DIM(shpp,i) );
	}
	/* we assume the precisions match ...  is this correct?  BUG? */
	// If the precisions are different, then it is up to the caller to change the precision...
	SET_SHP_PREC_PTR(shpp, SHP_PREC_PTR(shpp1) );
	auto_shape_flags(shpp,NO_OBJ);

	return shpp;	/* BUG?  are we sure we can get away with a single static shape here??? */
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

static Shape_Info * _shapes_mate(QSP_ARG_DECL  Vec_Expr_Node *enp1,Vec_Expr_Node *enp2,Vec_Expr_Node *enp)
{
	/* We do this test first, so in the case of scalar assignment,
	 * shapes_mate() returns the shape ptr of the destination...
	 */

	if( SCALAR_SHAPE(VN_SHAPE(enp2)) ){
		return(VN_SHAPE(enp1));
	} else if( SCALAR_SHAPE(VN_SHAPE(enp1)) ){
		return(VN_SHAPE(enp2));
	} else if( same_shape(VN_SHAPE(enp1),VN_SHAPE(enp2)) )
		return(VN_SHAPE(enp1));

	/* At this point, we know that either one of the shapes is unknown,
	 * or they have different shapes...
	 */

	/* OLD:  If one of the shapes is unknown, return the known shape. 
	 * NO, NO, NO!
	 * If one of the shapes is unknown, we must return the unknown shape!
	 * But it looks like the code used to do this - why was it changed???
	 * Possible BUG...
	 */

	if( UNKNOWN_SHAPE(VN_SHAPE(enp2)) ){
		return(VN_SHAPE(enp2));
		/* return(VN_SHAPE(enp1)); */
	}
	if( UNKNOWN_SHAPE(VN_SHAPE(enp1)) ){
		return(VN_SHAPE(enp1));
		/* return(VN_SHAPE(enp2)); */
	}

	/* what if they have the same shape, but one is real, one is complex? */
	/* BUG?  should we check which kind of operation? */
	if( COMPLEX_SHAPE(VN_SHAPE(enp1)) && ! COMPLEX_SHAPE(VN_SHAPE(enp2)) ) {
		/* make sure the other dimensions match */
		int i;

		for(i=1;i<N_DIMENSIONS;i++){
			if( SHP_TYPE_DIM(VN_SHAPE(enp1),i) != SHP_TYPE_DIM(VN_SHAPE(enp2),i) )
				goto mismatch;
		}
		return(VN_SHAPE(enp1));
	} else if( COMPLEX_SHAPE(VN_SHAPE(enp2)) && ! COMPLEX_SHAPE(VN_SHAPE(enp1)) ){
		/* make sure the other dimensions match */
		int i;

		for(i=1;i<N_DIMENSIONS;i++){
			if( SHP_TYPE_DIM(VN_SHAPE(enp1),i) != SHP_TYPE_DIM(VN_SHAPE(enp2),i) )
				goto mismatch;
		}
		return(VN_SHAPE(enp2));
	}

/*
DESCRIBE_SHAPE(VN_SHAPE(enp1));
DESCRIBE_SHAPE(VN_SHAPE(enp2));
	advise("shapes_mate:  should we fall through??");
	*/

	/* The shapes don't mate, but they may be ok for an outer binop...
	 * If so, we return the outer shape.
	 */
	{
		Shape_Info *shpp;
#ifdef FOOBAR	// we will encapsulate this into a library routine for more general use!
		int i;

		INIT_SHAPE_PTR(shpp)	// BUG?  memory leak?

		SET_SHP_N_TYPE_ELTS(shpp,1);
		for(i=0;i<N_DIMENSIONS;i++){
			if( SHP_TYPE_DIM(VN_SHAPE(enp1),i) == 1 ){
				SET_SHP_TYPE_DIM(shpp,i, SHP_TYPE_DIM(VN_SHAPE(enp2),i) );
			} else if( SHP_TYPE_DIM(VN_SHAPE(enp2),i) == 1 ){
				SET_SHP_TYPE_DIM(shpp,i, SHP_TYPE_DIM(VN_SHAPE(enp1),i) );
			} else if( SHP_TYPE_DIM(VN_SHAPE(enp1),i) == SHP_TYPE_DIM(VN_SHAPE(enp2),i) ){
				SET_SHP_TYPE_DIM(shpp,i, SHP_TYPE_DIM(VN_SHAPE(enp1),i) );
			} else {
				/* mismatch */
				goto mismatch;
			}
			SET_SHP_N_TYPE_ELTS(shpp,
				SHP_N_TYPE_ELTS(shpp) * SHP_TYPE_DIM(shpp,i) );
		}
		/* we assume the precisions match ...  is this correct?  BUG? */
		SET_SHP_PREC_PTR(shpp, SHP_PREC_PTR(VN_SHAPE(enp1)) );
		auto_shape_flags(shpp,NO_OBJ);
		return(shpp);	/* BUG?  can we get away with a single static shape here??? */
#endif // FOOBAR
		shpp = make_outer_shape(QSP_ARG  VN_SHAPE(enp1), VN_SHAPE(enp2));
		if( shpp != NULL ) return shpp;
	}

mismatch:
	NODE_ERROR(enp);
	NWARN("shapes_mate:  Operands have incompatible shapes");
	advise(node_desc(enp));
	/*
	dump_shape(VN_SHAPE(enp1));
	dump_shape(VN_SHAPE(enp2));
	DUMP_TREE(enp);
	*/
#ifdef QUIP_DEBUG
if( debug ){
//DUMP_TREE(enp);

// flag word is now a 64 bit long long...
sprintf(DEFAULT_ERROR_STRING,"flgs1 = 0x%llx     flgs2 = 0x%llx",
(long long unsigned int)SHP_FLAGS(VN_SHAPE(enp1))&SHAPE_MASK,
(long long unsigned int)SHP_FLAGS(VN_SHAPE(enp2))&SHAPE_MASK);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	return(NO_SHAPE);
}

/* Reexamine an assignment, but don't bother to remember anything.
 *
 * With the new scheme, is this different from compute_assign_shape??
 */

static void update_assign_shape(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
//	prec_t prec;

	/* Here the shape is the shape of the LHS;
	 * If the lhs is a scalar and the rhs is a vector,
	 * that's an error
	 */

//#ifdef CAUTIOUS
//	if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
//		/* a syntax error?? */
//		NODE_ERROR(enp);
//		WARN("CAUTIOUS:  update_assign_shape:  No shape for LHS");
//		if( VN_CHILD_SHAPE(enp,1) != NO_SHAPE )
//			prec=SHP_PREC(VN_CHILD_SHAPE(enp,1));
//		else	prec=PREC_SP;			/* BUG cautious update_assign_shape */
//		POINT_NODE_SHAPE(VN_CHILD(enp,0),uk_shape(prec));
//	}
	assert( VN_CHILD_SHAPE(enp,0) != NO_SHAPE );

//	if( VN_CHILD_SHAPE(enp,1) == NO_SHAPE ){
//		/* a syntax error?? */
//		NODE_ERROR(enp);
//		WARN("CAUTIOUS:  update_assign_shape:  No shape for RHS");
//	   // left child was given shape above, so we don't need to test here...
//		/*
//		if( VN_CHILD_SHAPE(enp,0) != NO_SHAPE )
//			prec=SHP_PREC(VN_CHILD_SHAPE(enp,0));
//		else	prec=PREC_SP; */			/* BUG cautious update_assign_shape */
//		
//		prec=SHP_PREC(VN_CHILD_SHAPE(enp,0));
//	   
//		POINT_NODE_SHAPE(VN_CHILD(enp,1),uk_shape(prec));
//	}
	assert( VN_CHILD_SHAPE(enp,1) != NO_SHAPE );

//#endif /* CAUTIOUS */

	/* Check for both shapes known */
	// BUG if child 0 has a null shape, we'll dereference it below!?!?

	if( ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) && ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,1)) ){

		/* make sure the shape of the expression is compatible
		 * with that of the target
		 */

		if( ! shapes_mate(VN_CHILD(enp,0),VN_CHILD(enp,1),enp) ){
			NODE_ERROR(enp);
			WARN("update_assign_shape:  assignment shapes do not mate");
			DESCRIBE_SHAPE(VN_CHILD_SHAPE(enp,0));
			DESCRIBE_SHAPE(VN_CHILD_SHAPE(enp,1));
			CURDLE(enp)
		} else {
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
		}
		return;
	}

	/* Now we know that at least one of the shapes is unknown */

	/* Check if both are unknown */

	if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) && UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,1)) ){

		/* Both sides of the assignment have unknown
		 * shape.  We need to scan both sides, and update
		 * everyone's lists...
		 */

		POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
		return;
	}

	/* Now we know that only one of the two shapes is unknown */

	if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
		POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
	} else {
		POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,1));
	}
} /* update_assign_shape */

/* When do we call typecast_child?
 *
 * When we want to insert a typecast node between the given node and one of its children.
 */

static void _typecast_child(QSP_ARG_DECL Vec_Expr_Node *enp,int index,Precision * prec_p)
{
	Vec_Expr_Node *new_enp;

//sprintf(ERROR_STRING,"typecast_child:  %s  %d  %s",
//node_desc(enp), index, PREC_NAME(prec_p));
//advise(ERROR_STRING);

	/* A few vector operators allow mixed mode ops */
	if( VN_CODE(enp) == T_TIMES ){
		if( COMPLEX_PRECISION(PREC_CODE(prec_p)) && ! COMPLEX_PRECISION(VN_CHILD_PREC(enp,index)) ){
/*
advise("mixed mode case");
*/
				if( VN_CHILD_PREC(enp,index)== PREC_MACH_CODE(prec_p) ){
/*
sprintf(ERROR_STRING,"child %s precision  %s matches machine precision %s",
node_desc(VN_CHILD(enp,index)),PREC_NAME(VN_CHILD_PREC_PTR(enp,index)),PREC_MACH_NAME(prec_p));
advise(ERROR_STRING);
*/
					return;
				}
/*
else advise("mixed mode machine precs do not match, casting");
*/
		}
	}

	switch(VN_CODE(VN_CHILD(enp,index))){	/* typecast_child */

		case T_VV_B_CONDASS:
		case T_SS_B_CONDASS:
//#ifdef CAUTIOUS
//			if( index == 0 ){
//				sprintf(ERROR_STRING,
//		"typecast child %s:  should not typecast bitmap child",
//					node_desc(enp));
//				WARN(ERROR_STRING);
//				DUMP_TREE(enp);
//				return;
//			}
//#endif /* CAUTIOUS */
			assert( index != 0 );
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
			if( PREC_CODE(prec_p) != PREC_BIT )
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
			if( COMPLEX_PRECISION(PREC_CODE(prec_p)) &&
			SHP_MACH_DIM(VN_SHAPE(VN_CHILD(enp,index)),0) == 2 ){
				if( VN_CHILD_PREC(enp,index)== PREC_SP ){
					VN_CHILD_PREC(enp,index)= PREC_CPX;
					SET_SHP_TYPE_DIM(VN_SHAPE(VN_CHILD(enp,index)),0, 1);
					return;
				} else if( VN_CHILD_PREC(enp,index)== PREC_DP ){
					SET_VN_CHILD_PREC(enp,index, PREC_DBLCPX);
					SET_SHP_TYPE_DIM(VN_SHAPE(VN_CHILD(enp,index)),0, 1);
					return;
				}
				break;
			} else if( QUAT_PRECISION(PREC_CODE(prec_p)) &&
			SHP_MACH_DIM(VN_SHAPE(VN_CHILD(enp,index)),0) == 4 ){
				if( VN_CHILD_PREC(enp,index)== PREC_SP ){
					SET_VN_CHILD_PREC(enp,index,PREC_QUAT);
					SET_SHP_TYPE_DIM(VN_SHAPE(VN_CHILD(enp,index)),0,1);
					return;
				} else if( VN_CHILD_PREC(enp,index) == PREC_DP ){
					SET_VN_CHILD_PREC(enp,index,  PREC_DBLQUAT);
					SET_SHP_TYPE_DIM(VN_SHAPE(VN_CHILD(enp,index)),0, 1);
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
		NON_BITMAP_CONDASS_CASES			/* typecast_child */
			break;

		default:
			MISSING_CASE(VN_CHILD(enp,index),"typecast_child");
			break;
	}

#ifdef QUIP_DEBUG
if( debug & cast_debug ){
sprintf(ERROR_STRING,"typecast_child %s:  typecasting child %s to %s",node_desc(enp),node_desc(VN_CHILD(enp,index)),
PREC_NAME(prec_p));
advise(ERROR_STRING);
DESCRIBE_SHAPE(VN_SHAPE(VN_CHILD(enp,index)));
}
#endif /* QUIP_DEBUG */

	new_enp = NODE1(T_TYPECAST,VN_CHILD(enp,index));

	SET_VN_CHILD(enp,index, new_enp);
	SET_VN_PARENT(new_enp, enp);

	SET_VN_CAST_PREC_PTR(new_enp, prec_p);

	prelim_node_shape(new_enp);
} /* typecast_child */

/* Return the dominant precision (the other will be promoted to it... */

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

static void _promote_child(QSP_ARG_DECL   Vec_Expr_Node *enp, int i1, int i2)
{
	int d;	/* dominant index */
	int i;	/* promoted index */
	prec_t p1,p2;

	p1=VN_CHILD_PREC(enp,i1);
	p2=VN_CHILD_PREC(enp,i2);

	if( !dominance_table_inited ) init_dominance_table();
	if( dominance_tbl[p1][p2] == p1 ){
		d=i1;
		i=i2;
	} else {
		d=i2;
		i=i1;
	}

	typecast_child(enp,i,VN_CHILD_PREC_PTR(enp,d));
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

static void _check_typecast(QSP_ARG_DECL  Vec_Expr_Node *enp,int i1, int i2)
{

#ifdef QUIP_DEBUG
if( debug & cast_debug ){
sprintf(ERROR_STRING,"check_typecast: %d %d",i1,i2);
advise(ERROR_STRING);
DUMP_TREE(enp);
}
#endif /* QUIP_DEBUG */

	if( VN_CHILD_PREC(enp,i1) == VN_CHILD_PREC(enp,i2) ) return;

	/* Here we know the two children have different types... */
	switch(VN_CODE(enp)){
		BINARY_BOOLOP_CASES
			/* logical and etc */
			/* handled by evaluator? */
			return;
		
		case T_VV_FUNC:				/* check_typecast */
		case T_VS_FUNC:
			/* we need to switch here on the function code... */
			switch(VN_VFUNC_CODE(enp)){
				VV_ALLPREC_FUNC_CASES
				VS_ALLPREC_FUNC_CASES
					goto not_integer_only;
					break;

				VS_INTONLY_FUNC_CASES
					goto integer_only_cases;
					break;

				default:
					sprintf(ERROR_STRING,
	"check_typecast:  unhandled function code %d (%s)",VN_VFUNC_CODE(enp),
						VF_NAME( FIND_VEC_FUNC( VN_VFUNC_CODE(enp) ) )
						);
					WARN(ERROR_STRING);
					break;
			}
			break;

		/* ALL_INTEGER_BINOP_CASES */
		ALL_SCALINT_BINOP_CASES		/* check_typecast */
			/* modulo, bitwise operators and shifts... */
integer_only_cases:
			/* these are operators where we want to cast to int */
			if( FLOATING_PREC(VN_CHILD_PREC(enp,i1)) ){
				if( INTEGER_PREC(VN_CHILD_PREC(enp,i2)) ){
					typecast_child(enp,i1,VN_CHILD_PREC_PTR(enp,i2));
				} else {
					typecast_child(enp,i1,PREC_FOR_CODE(PREC_DI));
					typecast_child(enp,i2,PREC_FOR_CODE(PREC_DI));
				}
			} else if( FLOATING_PREC(VN_CHILD_PREC(enp,i2)) ){
				typecast_child(enp,i2,VN_CHILD_PREC_PTR(enp,i1));
			} else {
				/* Both are integer, use promotion */
				promote_child(enp,i1,i2);
			}
			return;


		ALL_NUMERIC_COMPARISON_CASES
			/* bitmaps are a special case */
			if( BITMAP_SHAPE(VN_CHILD_SHAPE(enp,i1)) ){
				if( ! BITMAP_SHAPE(VN_CHILD_SHAPE(enp,i2)) ){
					typecast_child(enp,i2,VN_CHILD_PREC_PTR(enp,i1));
					return;
				}
			} else if( BITMAP_SHAPE(VN_CHILD_SHAPE(enp,i2)) ){
				typecast_child(enp,i1,VN_CHILD_PREC_PTR(enp,i2));
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

		//ALL_CONDASS_CASES			/* check_typecast */
		TRINARY_CONDASS_CASES			/* check_typecast */
//#ifdef CAUTIOUS
//			// This check is just for trinary condass nodes...
//			if( i1 == 0 || i2 == 0 ){
//				sprintf(ERROR_STRING,
//		"CAUTIOUS:  check_typecast %s:  shouldn't typecast bitmap node", node_desc(enp));
//				ERROR1(ERROR_STRING);
//			}
//#endif /* CAUTIOUS */
			assert( i1 != 0 && i2 != 0 );

			/* We'd rather not cast the children, we take care of the typecast
			 * at the super node...  BUT it is hard to know the destination type...
			 * Might be better to let this go and have a cleanup pass?
			 */
			break;
		ALL_NEW_CONDASS_CASES			/* check_typecast */
			// new 4-arg ops
			break;

		default:
			MISSING_CASE(enp,"check_typecast");
			break;
	}
	/* Don't typecast if we have real and complex... */

	/* The precisions differ - which should we cast? */
	if( SCALAR_SHAPE(VN_CHILD_SHAPE(enp,i1)) ){
		if( SCALAR_SHAPE(VN_CHILD_SHAPE(enp,i2)) ){
			/* if both are scalars, use C rules */
			promote_child(enp,i1,i2);
		} else {
			typecast_child(enp,i1,VN_CHILD_PREC_PTR(enp,i2));
		}
	} else if( SCALAR_SHAPE(VN_CHILD_SHAPE(enp,i2)) )
		typecast_child(enp,i2,VN_CHILD_PREC_PTR(enp,i1));

	else if( COMPLEX_SHAPE(VN_CHILD_SHAPE(enp,i1)) && !  COMPLEX_SHAPE(VN_CHILD_SHAPE(enp,i2)) ){
		typecast_child(enp,i2,VN_CHILD_PREC_PTR(enp,i1));
	} else if( COMPLEX_SHAPE(VN_CHILD_SHAPE(enp,i2)) && !  COMPLEX_SHAPE(VN_CHILD_SHAPE(enp,i1)) ){
		typecast_child(enp,i1,VN_CHILD_PREC_PTR(enp,i2));
	}
	else {
		/* use the C promotion rules */
		promote_child(enp,i1,i2);
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

static Shape_Info * check_mating_shapes(QSP_ARG_DECL  Vec_Expr_Node *enp,int index1,int index2)
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
	if( VN_CHILD_SHAPE(enp,index1)== NO_SHAPE && VN_CODE(VN_CHILD(enp,index1)) == T_CALLFUNC )
		VN_CHILD_SHAPE(enp,index1)= uk_shape( SR_PREC_CODE(VN_SUBRT(VN_CHILD(enp,index1))));

	if( VN_CHILD_SHAPE(enp,index2)== NO_SHAPE && VN_CODE(VN_CHILD(enp,index2)) == T_CALLFUNC )
		VN_CHILD_SHAPE(enp,index2)= uk_shape(SR_PREC_CODE(VN_SUBRT(VN_CHILD(enp,index2))));

	/* If one of the nodes has no shape, we get rid of the shape of the other (why?)
	 */

	if( VN_CHILD_SHAPE(enp,index1)== NO_SHAPE ){
		VN_CHILD_SHAPE(enp,index2)= NO_SHAPE;
		return(NO_SHAPE);
	}

	if( VN_CHILD_SHAPE(enp,index2)== NO_SHAPE ){
		VN_CHILD_SHAPE(enp,index1)= NO_SHAPE;
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

	shpp = shapes_mate(VN_CHILD(enp,index1),VN_CHILD(enp,index2),enp);
//sprintf(ERROR_STRING,"check_mating_shapes:  shapes_mate 0x%lx",
//(long)shpp);
//advise(ERROR_STRING);

	if( shpp==NO_SHAPE ){
		discard_node_shape(enp);
advise("check_mating_shapes:  no mating shapes");
		return(NO_SHAPE);
	}

	switch(VN_CODE(enp)){
		ALL_NUMERIC_COMPARISON_CASES			/* check_mating_shapes */
			copy_node_shape(enp,shpp);
			if( ! BITMAP_SHAPE(shpp) ){
				xform_to_bitmap(VN_SHAPE(enp));
			}
			break;

		BINARY_BOOLOP_CASES				/* check_mating_shapes */
			copy_node_shape(enp,shpp);
			if( ! BITMAP_SHAPE(shpp) ){
				xform_to_bitmap(VN_SHAPE(enp));
			}
			break;

		case T_MAXVAL:
		case T_MINVAL:
//#ifdef CAUTIOUS
//			if( VN_SHAPE(enp) != NULL ){
//				sprintf(ERROR_STRING,
//	"CAUTIOUS:  check_mating_shapes:  prior shape for %s not null!?\n",node_desc(enp));
//				WARN(ERROR_STRING);
//			}
//#endif // CAUTIOUS
			assert( VN_SHAPE(enp) == NULL );

			copy_node_shape(enp,shpp);

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
//sprintf(ERROR_STRING,"check_mating_shapes %s:  pointing to shape:",node_desc(enp));
//advise(ERROR_STRING);
//DESCRIBE_SHAPE(shpp);
			POINT_NODE_SHAPE(enp,shpp);
			break;

		ALL_CONDASS_CASES				/* check_mating_shapes */
		case T_VV_FUNC:					/* check_mating_shapes */
			copy_node_shape(enp,shpp);
			break;
	}

	return(shpp);
} /* check_mating_shapes */

/* Get the shapes of the first two child nodes, and check that they "mate"
 * If either node has no shape, both ptrs are set to NO_SHAPE for the return,
 * Shapes mate when they are identical, or one is a scalar.
 *
 * For generalized outer binops, this is more complicated...
 */


static Shape_Info * _get_mating_shapes(QSP_ARG_DECL   Vec_Expr_Node *enp,int i1, int i2)
{
//sprintf(ERROR_STRING,"get_mating_shapes:  %s %d %d",
//node_desc(enp),i1,i2);
//advise(ERROR_STRING);

	check_typecast(enp,i1,i2);
	return( check_mating_shapes(QSP_ARG  enp,i1,i2) );
}

/* Set the shape of this node based on its children.
 */

static void _update_node_shape(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Shape_Info *tmp_shpp;
	Subrt *srp;
	dimension_t i1,i2,len ;
#ifdef NOT_YET
	Image_File *ifp;
	const char *str;
#endif /* NOT_YET */

	/* now all the child nodes have been scanned, process this one */

	if( IS_CURDLED(enp) ) return;

	INIT_SHAPE_PTR(tmp_shpp)

	switch(VN_CODE(enp)){
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

			lenp = VN_CHILD(enp,1);	/* T_EXPR_LIST */
			/* expression lists have the final expression on the right child,
			 * and the preceding list on the left child.
			 */
			i_dim = 0;
			do {
				d = (dimension_t) EVAL_INT_EXP(VN_CHILD(lenp,1));
				if( d == 0 ){
					/* not an error if it's not run-time yet */
					copy_node_shape(enp,uk_shape(PREC_CODE(VN_DECL_PREC(enp))));
					break;
				}
				SET_SHP_TYPE_DIM(VN_SHAPE(enp),i_dim, d);
				i_dim++;
				lenp = VN_CHILD(lenp,0);
			} while( i_dim < N_DIMENSIONS && VN_CODE(lenp) == T_EXPR_LIST );

			if( i_dim >= N_DIMENSIONS ){
				NODE_ERROR(enp);
				WARN("too many dimension specifiers");
				CURDLE(enp)
				break;
			}

			d = (dimension_t) EVAL_INT_EXP( lenp );
			if( d == 0 ){
				/* not an error if it's not run-time yet */
				copy_node_shape(enp,uk_shape(PREC_CODE(VN_DECL_PREC(enp))));
				break;
			}
			SET_SHP_TYPE_DIM(VN_SHAPE(enp),i_dim, d);
			CLEAR_SHP_FLAG_BITS(VN_SHAPE(enp),DT_UNKNOWN_SHAPE);	/* if we are here, all dimensions are non-zero */
			auto_shape_flags(VN_SHAPE(enp),NO_OBJ);
			}

			break;

		case T_SUBSAMP:				/* update_node_shape */
			if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ) break;

			/* the subsample node will have the same shape
			 * (an image is still an image) but will have
			 * different dimensions, determined by the value
			 * of the children nodes...
			 */

			copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));

			{
			Vec_Expr_Node *range_enp;
			incr_t inc,l;

			/* second child should be a range specification, it's shape is long scalar */
			/* We need to evaluate it to know what is going on here */
			range_enp=VN_CHILD(enp,1);
			/* do a cautious check here for correct node type */
			i1 = (dimension_t) EVAL_INT_EXP(VN_CHILD(range_enp,0));
			i2 = (dimension_t) EVAL_INT_EXP(VN_CHILD(range_enp,1));
			inc = (incr_t) EVAL_INT_EXP(VN_CHILD(range_enp,2));
			if( i2 > i1 ) {
				l = 1+(i2-i1+1)/inc;
			} else {
				l = 1+(i1-i2+1)/(-inc);
			}
			if( l < 0 )
				WARN("apparent length of subsample object is negative!?");

			SET_SHP_N_TYPE_ELTS(VN_SHAPE(enp),
				SHP_N_TYPE_ELTS(VN_SHAPE(enp)) /
				SHP_TYPE_DIM(VN_SHAPE(enp),/*SHP_MAXDIM*/SHP_RANGE_MAXDIM(VN_SHAPE(enp))) );
			SET_SHP_N_MACH_ELTS(VN_SHAPE(enp),
				SHP_N_MACH_ELTS(VN_SHAPE(enp)) /
				SHP_MACH_DIM(VN_SHAPE(enp),/*SHP_MAXDIM*/SHP_RANGE_MAXDIM(VN_SHAPE(enp))) );
			SET_SHP_N_TYPE_ELTS(VN_SHAPE(enp),
				SHP_N_TYPE_ELTS(VN_SHAPE(enp)) * l );
			SET_SHP_N_MACH_ELTS(VN_SHAPE(enp),
				SHP_N_MACH_ELTS(VN_SHAPE(enp)) * l );
			SET_SHP_TYPE_DIM(VN_SHAPE(enp),/*SHP_MAXDIM*/SHP_RANGE_MAXDIM(VN_SHAPE(enp)),l);
			SET_SHP_MACH_DIM(VN_SHAPE(enp),/*SHP_MAXDIM*/SHP_RANGE_MAXDIM(VN_SHAPE(enp)),l);
			/*SET_SHP_MAXDIM*/SET_SHP_RANGE_MAXDIM(VN_SHAPE(enp),
				/*SHP_MAXDIM*/SHP_RANGE_MAXDIM(VN_SHAPE(enp))-1);
			}
				
			break;

		case T_RANGE2:
			i1 = (dimension_t) EVAL_INT_EXP(VN_CHILD(enp,0));
			i2 = (dimension_t) EVAL_INT_EXP(VN_CHILD(enp,1));
			/* see T_ROW below... */
			copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
			if( i2 < i1 ){
				sprintf(ERROR_STRING,
"First element of range specification (%d) should be smaller than the second (%d)!?",
					i1,i2);
				WARN(ERROR_STRING);
				advise("Reversing.");
				SET_SHP_COLS(VN_SHAPE(enp), 1 + (i1 - i2) );
			} else {
				SET_SHP_COLS(VN_SHAPE(enp), 1 + (i2 - i1) );
			}
			auto_shape_flags(VN_SHAPE(enp),NO_OBJ);
			break;
		/* matlab! */
		case T_RET_LIST:		/* update_node_shape */
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
			copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
			SET_SHP_COLS(VN_SHAPE(enp),
				SHP_COLS(VN_SHAPE(enp)) +1 );
			auto_shape_flags(VN_SHAPE(enp),NO_OBJ);
			break;

		/* end matlab */

		case T_FIX_SIZE:
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
			break;

		/* Do-nothing cases */
		case T_REFERENCE: /* reference node should always point to child node */	/* update_node_shape */
		case T_DEREFERENCE: /* dereference node should always point to child node */
			break;

		case T_SS_B_CONDASS:					/* update_node_shape */
			/* child 0 is a bitmap...
			 * the other children are scalars
			 */

			if( ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
//fprintf(stderr,"calling xform_from_bitmap #5\n");
				xform_from_bitmap(VN_SHAPE(enp),VN_CHILD_PREC(enp,1));
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
			copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
			break;

		case T_TYPECAST:		/* update_node_shape */
			COPY_SHAPE(tmp_shpp,VN_CHILD_SHAPE(enp,0));
			SET_SHP_PREC_PTR(tmp_shpp, VN_CAST_PREC_PTR(enp));
			copy_node_shape(enp,tmp_shpp);
			break;

		case T_SQUARE_SUBSCR:		/* update_node_shape */
			if( !SCALAR_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
				/* The subscript is a vector, use its shape.
				 * Note that this is going to have problems with
				 * multiple vector-valued subscripts...
				 */
				copy_node_shape(enp,VN_CHILD_SHAPE(enp,1));
				SET_VN_PREC(enp, VN_CHILD_PREC(enp,0) );

				/* The subscript can have a non-1 type dimension
				 * if it needs to address multiple indexing dimensions.
				 * But we don't want to raise the type dimension
				 * of the target.
				 */
				SET_SHP_TYPE_DIM(VN_SHAPE(enp),0, SHP_TYPE_DIM(VN_CHILD_SHAPE(enp,0),0) );
				auto_shape_flags(VN_SHAPE(enp),NO_OBJ);	/* DO we need this? */
				break;
			}
			if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ) break;

			if( SHP_TYPE_DIM(VN_CHILD_SHAPE(enp,0),SHP_MAXDIM(VN_CHILD_SHAPE(enp,0))) == 0 ){
				sprintf(ERROR_STRING,
			"update_node_shape:  %s node n%d has dimension %d zero, can't subscript",
					NNAME(VN_CHILD(enp,0)),VN_SERIAL(VN_CHILD(enp,0)),SHP_MAXDIM(VN_CHILD_SHAPE(enp,0)));
				WARN(ERROR_STRING);
				break;
			}

			//*tmp_shpp = *VN_CHILD_SHAPE(enp,0);
			//copy_node_shape(enp,tmp_shpp);

			copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));

			REDUCE_MAXDIM(VN_SHAPE(enp))
			auto_shape_flags(VN_SHAPE(enp),NO_OBJ);
			/* BUG do range checking here */
			break;

		case T_CURLY_SUBSCR:		/* update_node_shape */
			/* We normally get the shape from the object
			 * being subscripted...
			 */
			if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) )
				break;

			if( SHP_TYPE_DIM(VN_CHILD_SHAPE(enp,0),SHP_MINDIM(VN_CHILD_SHAPE(enp,0))) == 0 ){
				sprintf(ERROR_STRING,
			"update_node_shape:  %s node n%d has dimension %d zero, can't subscript",
					NNAME(VN_CHILD(enp,0)),VN_SERIAL(VN_CHILD(enp,0)),SHP_MINDIM(VN_CHILD_SHAPE(enp,0)));
				WARN(ERROR_STRING);
				break;
			}

			//*tmp_shpp = *VN_CHILD_SHAPE(enp,0);
			//copy_node_shape(enp,tmp_shpp);

			copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));

			REDUCE_MINDIM(VN_SHAPE(enp))
			auto_shape_flags(VN_SHAPE(enp),NO_OBJ);
			/* BUG do range checking here */
			break;

		case T_REAL_PART:		/* update_node_shape */
		case T_IMAG_PART:		/* update_node_shape */
			COPY_SHAPE(tmp_shpp,VN_CHILD_SHAPE(enp,0));
			SET_SHP_MACH_DIM(tmp_shpp,0,1);
			SET_SHP_N_MACH_ELTS(tmp_shpp,
				SHP_N_MACH_ELTS(tmp_shpp) / 2 );
			CLEAR_SHP_FLAG_BITS(tmp_shpp,DT_COMPLEX);
			SET_SHP_PREC_PTR(tmp_shpp, PREC_MACH_PREC_PTR(SHP_PREC_PTR(tmp_shpp)) );

			copy_node_shape(enp,tmp_shpp);
			/* BUG verify VN_CHILD_SHAPE(enp,0) is complex */
			break;

		case T_TRANSPOSE:		/* update_node_shape */
			VN_CHILD_SHAPE(enp,0) = get_child_shape(QSP_ARG  enp,0);
			COPY_SHAPE(tmp_shpp,VN_CHILD_SHAPE(enp,0));
			i1                    = SHP_ROWS(tmp_shpp);
			SET_SHP_ROWS(tmp_shpp, SHP_COLS(tmp_shpp) );
			SET_SHP_COLS(tmp_shpp, i1);
			auto_shape_flags(tmp_shpp,NO_OBJ);

			copy_node_shape(enp,tmp_shpp);

			break;

		case T_SUBVEC:		/* update_node_shape */
			if( !get_subvec_indices(QSP_ARG enp,&i1,&i2) ) break;

			len = i2+1-i1;

			/* Now, we need a good way to determine which dimension
			 * is being indexed!?
			 */

			copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));

			/* BUG there is code missing here !? */
			/* what code? */
			/*UPDATE_MAXDIM_LEN*/UPDATE_RANGE_MAXDIM_LEN(VN_SHAPE(enp),len)

			/* does auto_shape_flags set mindim/maxdim??? */
			/* auto_shape_flags(VN_SHAPE(enp),NO_OBJ); */

			break;

		case T_CSUBVEC:		/* update_node_shape */
			if( !get_subvec_indices(QSP_ARG enp,&i1,&i2) ) break;

			len = i2+1-i1;

			copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));

			/* Now, we need a good way to determine which dimension
			 * is being indexed!?
			 */
			/* BUG there is code missing here !? */
			/*UPDATE_MINDIM_LEN*/UPDATE_RANGE_MINDIM_LEN(VN_SHAPE(enp),len)

			/* auto_shape_flags(VN_SHAPE(enp),NO_OBJ); */

			break;

		case T_RDFT:		/* update_node_shape */
			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE )
				break;		/* an error */

			/* BUG should verify that arg is real and power of 2 */

			COPY_SHAPE(tmp_shpp,VN_CHILD_SHAPE(enp,0));
			if( !UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				SET_SHP_COLS(tmp_shpp,
					SHP_COLS(tmp_shpp) / 2 );
				SET_SHP_COLS(tmp_shpp,
					SHP_COLS(tmp_shpp) + 1 );
				SET_SHP_MACH_DIM(tmp_shpp,0, 2);
				/* BUG?  recalc other things? */
				SET_SHP_PREC_PTR(tmp_shpp, complex_precision(SHP_PREC_PTR(tmp_shpp)) );
				SET_SHP_FLAG_BITS(tmp_shpp, DT_COMPLEX);
			}
			copy_node_shape(enp,tmp_shpp);


			break;

		case T_RIDFT:		/* update_node_shape */
			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE )
				break;		/* an error */

			/* BUG should verify that arg is real and power of 2 */

			COPY_SHAPE(tmp_shpp,VN_CHILD_SHAPE(enp,0));
			if( !UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				SET_SHP_COLS(tmp_shpp,
					SHP_COLS(tmp_shpp) - 1 );
				SET_SHP_COLS(tmp_shpp,
					SHP_COLS(tmp_shpp) * 2 );
				SET_SHP_MACH_DIM(tmp_shpp,0, 1);
				/* BUG?  set other things? */
				auto_shape_flags(tmp_shpp,NO_OBJ);
			}
			copy_node_shape(enp,tmp_shpp);

			break;

		case T_WRAP:		/* update_node_shape */
		case T_SCROLL:		/* update_node_shape */
			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE )
				break;		/* an error */

			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));

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

#ifdef NOT_YET
		case T_LOAD:		/* update node shape */
			if( ! executing ) goto no_file;

			str=EVAL_STRING(VN_CHILD(enp,0));
			if( str == NULL ) goto no_file;	/* probably a variable that we can't know until runtime */
			ifp = img_file_of(QSP_ARG  str);

			if( ifp == NO_IMAGE_FILE ){
				ifp = read_image_file(QSP_ARG  str);
				if( ifp==NO_IMAGE_FILE ){
					NODE_ERROR(enp);
					sprintf(ERROR_STRING,
	"update_node_shape READ/LOAD:  Couldn't open image file %s",str);
					WARN(ERROR_STRING);
					goto no_file;
				}
			}

			if( ! IS_READABLE(ifp) ){
				sprintf(ERROR_STRING,
		"File %s is not readable!?",str);
				WARN(ERROR_STRING);
				goto no_file;
			}
//#ifdef CAUTIOUS
//			if( ifp->if_dp == NO_OBJ ){
//				sprintf(ERROR_STRING,
//	"CAUTIOUS:  file %s has no data object in header!?",str);
//				WARN(ERROR_STRING);
//				goto no_file;
//			}
//#endif /* CAUTIOUS */
			assert( ifp->if_dp != NO_OBJ );

			/* BUG - the file shape is not necessarily
			 * the same as what the ultimate destination
			 * object will be - we will want to be able
			 * to read single frames from a file containing
			 * a sequence of frames.
			 */

			copy_node_shape(enp,OBJ_SHAPE(ifp->if_dp));
			break;
no_file:
			/* BUG we don't know the prec if we have no file!? */
			POINT_NODE_SHAPE(enp,uk_shape(PREC_SP));


			break;
#endif /* NOT_YET */


		case T_RETURN:		/* update_node_shape */

			/* This is our chance to figure out the size
			 * of a subroutine...
			 * at least if we are returning an object of known
			 * size...
			 */

			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){ /* a void subrt */

				/* leave the shape pointer null */
				/* BUG? do we need to do this? */

				POINT_NODE_SHAPE(enp,void_shape());
				break;
			} else if( IS_CURDLED(VN_CHILD(enp,0)) ){
				/* an error expr */
				WARN("return expression is curdled!?");
				discard_node_shape(enp);
				break;
			} else {
				POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
//#ifdef CAUTIOUS
//if( VN_SHAPE(enp) == NO_SHAPE ){
//sprintf(ERROR_STRING,"CAUTIOUS:  update_node_shape:  return expression has no shape!?");
//WARN(ERROR_STRING);
//break;
//}
//#endif /* CAUTIOUS */
				assert( VN_SHAPE(enp) != NO_SHAPE );
			}

			/* Here we need to see if there are other
			 * returns, and if so we need to make sure
			 * all the returns match in size...
			 * We also need to communicate this info
			 * to the main subroutine body...
			 */

			srp = curr_srp;

			if( UNKNOWN_SHAPE(SR_SHAPE(srp)) &&
				! UNKNOWN_SHAPE(VN_SHAPE(enp)) ){

				SET_SR_SHAPE(srp, VN_SHAPE(enp) );

			} else if( ! UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
				/* does the shape of this return match? */
				if( !shapes_match(SR_SHAPE(srp), VN_SHAPE(enp)) ){


					NODE_ERROR(enp);
					WARN("mismatched return shapes");
				}
			}

			break;

		case T_CALLFUNC:			/* update_node_shape */

			srp=VN_SUBRT(enp);
			SET_SR_CALL_VN(srp, enp);

			/* Why do we do this here??? */

			/* make sure the number of aruments is correct */
			if( arg_count(VN_CHILD(enp,0)) != SR_N_ARGS(srp) ){
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,
	"Subrt %s expects %d arguments (%d passed)",SR_NAME(srp), SR_N_ARGS(srp),
					arg_count(VN_CHILD(enp,0)));
				WARN(ERROR_STRING);
				CURDLE(enp)
				break;
			}

			if( SR_SHAPE(srp) != NO_SHAPE ){
				copy_node_shape(enp,SR_SHAPE(srp));
			}
			break;

		case T_ARGLIST:		/* update_node_shape */
			break;


		case T_ROW_LIST:		/* update_node_shape */

			/* can arise as print args,
			 * or a child node of T_LIST_OBJ (declaration)
			 *
			 * Actually, now arg lists are T_EXPR_LIST?
			 */

			/*
			if( get_mating_shapes(enp,0,1) == NO_SHAPE )
				break;
			*/

			if(VN_CHILD_SHAPE(enp,0)==NO_SHAPE || VN_CHILD_SHAPE(enp,1)==NO_SHAPE)
				break;


			COPY_SHAPE(tmp_shpp,VN_CHILD_SHAPE(enp,1));

			// There is a problem when the children are LIST_OBJ's !?

			SET_SHP_TYPE_DIM(tmp_shpp,SHP_MAXDIM(tmp_shpp)+1,
				SHP_TYPE_DIM(VN_CHILD_SHAPE(enp,0),SHP_MAXDIM(tmp_shpp)+1)+1 );
			// Do we need to set/use RANGE_MAXDIM???

			auto_shape_flags(tmp_shpp,NO_OBJ);

			copy_node_shape(enp,tmp_shpp);

			break;

//		/* old matlab case */
//		case T_ROWLIST:						/* update_node_shape */
//			copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
//			SET_SHP_ROWS(VN_SHAPE(enp),
//				SHP_ROWS(VN_SHAPE(enp))+1);
//			auto_shape_flags(VN_SHAPE(enp),NO_OBJ);
//			break;


		case T_RAMP:		/* update_node_shape */
			break;

		case T_RECIP:		/* update_node_shape */
		case T_UMINUS:		/* update_node_shape */
			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
				CURDLE(enp)
				break;		/* an error */
			}
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
			break;

		case T_MATH0_VFN:				/* update_node_shape */
			/* BUG?  prec SP or DP?? */
			POINT_NODE_SHAPE(enp,uk_shape(PREC_DP));
			break;

		case T_MATH2_VFN:				/* update_node_shape */
		case T_MATH1_VFN:				/* update_node_shape */
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
			break;

		case T_INT1_VFN:				/* update_node_shape */
			// source is floating pt, but result is integer...
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
			break;

		ALL_SCALINT_BINOP_CASES
//#ifdef CAUTIOUS
//			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
//				NODE_ERROR(enp);
//		ERROR1("CAUTIOUS: update_node_shape:  scalint binop node left child has no shape!?");
//			}
//#endif /* CAUTIOUS */
			assert( VN_CHILD_SHAPE(enp,0) != NO_SHAPE );
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
			break;

		case T_INT1_FN:				/* update_node_shape */
//#ifdef CAUTIOUS
//			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
//				NODE_ERROR(enp);
//				ERROR1("CAUTIOUS:  math fn arg node has no shape!?");
//			}
//#endif /* CAUTIOUS */
			assert( VN_CHILD_SHAPE(enp,0) != NO_SHAPE );
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_IN));
			break;

		case T_MATH1_FN:					/* update_node_shape */
//#ifdef CAUTIOUS
//			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
//				NODE_ERROR(enp);
//				ERROR1("CAUTIOUS:  math fn arg node has no shape!?");
//			}
//#endif /* CAUTIOUS */
			assert( VN_CHILD_SHAPE(enp,0) != NO_SHAPE );

			POINT_NODE_SHAPE(enp,scalar_shape(PREC_SP));
			break;

		case T_MATH0_FN:		/* update_node_shape */
		case T_MATH2_FN:		/* update_node_shape */
			/* always has a scalar shape */
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_SP));
			break;

		case T_MATH2_VSFN:		/* update_node_shape */
//#ifdef CAUTIOUS
//			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ||
//				VN_CHILD(enp,1) == NO_VEXPR_NODE ){
//				WARN("CAUTIOUS:  update_node_shape MATH2_VSFN:  missing child");
//				/* parse error - should zap this node? */
//				break;
//			}
//#endif /* CAUTIOUS */
			assert( VN_CHILD(enp,0) != NO_VEXPR_NODE );
			assert( VN_CHILD(enp,1) != NO_VEXPR_NODE );
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0)); /* BUG DP? math2_fn */
			break;

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
//#ifdef CAUTIOUS
//			if( VN_SHAPE(enp) == NO_SHAPE ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,"CAUTIOUS:  update_node_shape %s:  null shape ptr!?",node_desc(enp));
//				ERROR1(ERROR_STRING);
//			}
//#endif /* CAUTIOUS */
			assert( VN_SHAPE(enp) != NO_SHAPE );
#endif /* FOOBAR */

			/* this is redundant most of the time... */
			if( get_mating_shapes(enp,0,1) == NO_SHAPE )
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
				dp = DOBJ_OF(VN_STRING(enp));
				if( dp == NO_OBJ ){
/*
sprintf(ERROR_STRING,"update_node_shape T_DYN_OBJ (matlab):  no object %s!?",VN_STRING(enp));
advise(ERROR_STRING);
*/
					break;
				}
				if( ! UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
fprintf(stderr,"pointing node shape to object shape at 0x%lx\n",(long)OBJ_SHAPE(dp));
					POINT_NODE_SHAPE(enp,OBJ_SHAPE(dp));
				}
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

	RELEASE_SHAPE_PTR(tmp_shpp);

} /* end update_node_shape */

static Shape_Info *cpx_scalar_shape(prec_t prec)
{
	int i;

	if( _cpx_scalar_shpp[prec]!=NO_SHAPE )
		return(_cpx_scalar_shpp[prec]);

	//_cpx_scalar_shpp[prec] = (Shape_Info *)getbuf(sizeof(Shape_Info));
	_cpx_scalar_shpp[prec] = alloc_shape();
	for(i=0;i<N_DIMENSIONS;i++){
		SET_SHP_TYPE_DIM(_cpx_scalar_shpp[prec],i, 1);
		SET_SHP_MACH_DIM(_cpx_scalar_shpp[prec],i, 1);
	}
	SET_SHP_MACH_DIM(_cpx_scalar_shpp[prec],0,2);

	SET_SHP_FLAGS(_cpx_scalar_shpp[prec], 0);

	SET_SHP_PREC_PTR(_cpx_scalar_shpp[prec], complex_precision(PREC_FOR_CODE(prec)) );

	auto_shape_flags(_cpx_scalar_shpp[prec],NO_OBJ);

	return(_cpx_scalar_shpp[prec]);
}

#ifdef FOOBAR
//#ifdef CAUTIOUS
//static int insure_child_shape(Vec_Expr_Node *enp, int child_index)
//{
//	if( VN_CHILD_SHAPE(enp,child_index)== NO_SHAPE ){
//		sprintf(DEFAULT_ERROR_STRING,
//			"CAUTIOUS:  insure_child_shape:  %s has no shape!?",
//			node_desc(VN_CHILD(enp,child_index)));
//		NWARN(DEFAULT_ERROR_STRING);
//		return(-1);
//	}
//	return(0);
//}
//#endif /* CAUTIOUS */
#endif // FOOBAR


/* Add a tree node to a list...
 * We pass a ptr to the list pointer, in case the list needs to be created.
 */

static void remember_node(List **lpp,Vec_Expr_Node *enp)
{
	Node *np;
	List *lp;

	/* We used to do a CAUTIOUS check to make sure that the node
	 * is not on the list already...  But because we deviate from
	 * a strict tree structure, we may visit a node twice
	 * during tree traversal...  Therefore we use a flag (which we
	 * have to clear!!)
	 *
	 * However, we have more than one list on which we remember nodes...
	 */

	if( *lpp == NO_LIST )
		*lpp = new_list();

	assert( *lpp != NO_LIST );

	lp = *lpp;

	/* make sure enp not on list already */
	np = QLIST_HEAD((lp));
	while(np!=NO_NODE){
		if( NODE_DATA(np) == enp ){
			/* Because nodes can have multiple parents,
			 * this is not an error!?
			 */
			return;
		}
		np=NODE_NEXT(np);
	}

	np = mk_node(enp);
	addHead(lp,np);
}


static void link_one_uk_arg(Vec_Expr_Node *call_enp, Vec_Expr_Node *arg_enp)
{
	Node *np;

	if( VN_UK_ARGS(call_enp) == NO_LIST )
		SET_VN_UK_ARGS(call_enp, NEW_LIST );

	np = mk_node(arg_enp);
	addTail(VN_UK_ARGS(call_enp),np);
	/*
	LINK_UK_NODES(call_enp,arg_enp);
	*/
}


/* link_uk_args
 * For a CALLFUNC node, descend the arg value tree,
 * looking for uk objects that need to be resolved.
 * I'm not sure exatly how this is going to work???
 */

static void link_uk_args(QSP_ARG_DECL  Vec_Expr_Node *call_enp,Vec_Expr_Node *arg_enp)
{
	Data_Obj *dp;

	switch(VN_CODE(arg_enp)){
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
		case T_TIMES:
		case T_DIVIDE:
			break;

		case T_DYN_OBJ:				/* link_uk_args */
			dp = DOBJ_OF(VN_STRING(arg_enp));
//#ifdef CAUTIOUS
//			if( dp == NO_OBJ ){
//				NODE_ERROR(arg_enp);
//				sprintf(ERROR_STRING,"Obj Arg %s has no associated object %s!?",
//					node_desc(arg_enp),VN_STRING(arg_enp));
//				WARN(ERROR_STRING);
//				return;
//			}
//#endif /* CAUTIOUS */
			assert( dp != NO_OBJ );

			if( UNKNOWN_SHAPE(OBJ_SHAPE(dp)) )
				link_one_uk_arg(call_enp,arg_enp);

			break;

		case T_STATIC_OBJ:
			dp = VN_OBJ(arg_enp);
			assert( dp != NO_OBJ );
			if( UNKNOWN_SHAPE(OBJ_SHAPE(dp)) )
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
			link_uk_args(QSP_ARG  call_enp,VN_CHILD(arg_enp,0));
			link_uk_args(QSP_ARG  call_enp,VN_CHILD(arg_enp,1));
			break;

		/* only one child to descend
		 * The subscripts are tricky because the value inside could be
		 * an expression?
		 */
		case T_TRANSPOSE:
		case T_TYPECAST:
		case T_CURLY_SUBSCR:
			link_uk_args(QSP_ARG  call_enp,VN_CHILD(arg_enp,0));
			break;

		default:
			MISSING_CASE(arg_enp,"link_uk_args");
			break;
	}
} /* end link_uk_args */


/* what is the purpose of this??? */

static void remember_callfunc_node(Subrt *srp,Vec_Expr_Node *enp)
{
	remember_node( & SR_CALL_LIST(srp),enp);
}



static void _check_binop_links(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	CHECK_UK_CHILD(enp,0);
	CHECK_UK_CHILD(enp,1);
}

static void scalarize(Shape_Info *shpp)
{
	int i;

	for(i=0;i<N_DIMENSIONS;i++){
		SET_SHP_MACH_DIM(shpp,i,1);
		SET_SHP_TYPE_DIM(shpp,i,1);
	}
	SET_SHP_N_MACH_ELTS(shpp, 1);
	SET_SHP_N_TYPE_ELTS(shpp, 1);
}


/* compatible_shape doesn't insist that two objects have the same shape, but if the
 * dimensions do not match then one must be equal to 1.
 * The result shape always has the larger dimension (suitable for outer product, etc).
 */

static Shape_Info *compatible_shape(Vec_Expr_Node *enp1,Vec_Expr_Node *enp2,Vec_Expr_Node *enp)
{
	int i;
	Shape_Info *_si_p,*shpp;

	INIT_SHAPE_PTR(_si_p)

	for(i=0;i<N_DIMENSIONS;i++){
		if(	SHP_TYPE_DIM(VN_SHAPE(enp1),i) != 1 &&
			SHP_TYPE_DIM(VN_SHAPE(enp2),i) != 1 &&
			SHP_TYPE_DIM(VN_SHAPE(enp1),i) != SHP_TYPE_DIM(VN_SHAPE(enp2),i) ){

			/* We put in a special case here to handle the assigment of a string to
			 * a row vector of bytes...
			 */
			if( i == 1 && SHP_PREC(VN_SHAPE(enp1)) == PREC_CHAR && SHP_PREC(VN_SHAPE(enp2)) == PREC_CHAR &&
				SHP_TYPE_DIM(VN_SHAPE(enp1),i) > SHP_TYPE_DIM(VN_SHAPE(enp2),i) ){
				/* allow this */
			} else {
NADVISE("compatible_shape:  enp1");
describe_shape(DEFAULT_QSP_ARG  VN_SHAPE(enp1));
NADVISE("compatible_shape:  enp2");
describe_shape(DEFAULT_QSP_ARG  VN_SHAPE(enp2));
				return(NO_SHAPE);
			}
		}

		if( SHP_TYPE_DIM(VN_SHAPE(enp1),i) > SHP_TYPE_DIM(VN_SHAPE(enp2),i) )
			SET_SHP_TYPE_DIM(_si_p,i, SHP_TYPE_DIM(VN_SHAPE(enp1),i) );
		else
			SET_SHP_TYPE_DIM(_si_p,i, SHP_TYPE_DIM(VN_SHAPE(enp2),i) );
	}
	//shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
	shpp = alloc_shape();
	*shpp = *_si_p;
	SET_SHP_FLAGS(shpp, 0);
	scalarize(shpp);	/* set all dimensions to 1 */

	if( !dominance_table_inited ) init_dominance_table();
	SET_SHP_PREC_PTR(shpp, PREC_FOR_CODE( dominance_tbl[ SHP_PREC(VN_SHAPE(enp1)) & MACH_PREC_MASK ]
					[ SHP_PREC(VN_SHAPE(enp2)) & MACH_PREC_MASK ] ) );
	auto_shape_flags(shpp,NO_OBJ);
	return(shpp);
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
//	prec_t prec;

	/* Here the shape is the shape of the LHS;
	 * If the lhs is a scalar and the rhs is a vector,
	 * that's an error
	 */

#ifdef SUPPORT_MATLAB_MODE
	if( mode_is_matlab ){
		/* in matlab, the size can be determined from the RHS */
		/* but we really only want to do this if the LHS is an object??? */
		/* && VN_CODE(VN_CHILD(enp,0)) == T_DYN_OBJ ) */

		POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,1));
		if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,1)) )
			LINK_UK_NODES(enp,VN_CHILD(enp,1));
		return;
	}
#endif /* SUPPORT_MATLAB_MODE */

//#ifdef CAUTIOUS
//	if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
//		/* a syntax error?? */
//		NODE_ERROR(enp);
//		WARN("CAUTIOUS:  compute_assign_shape:  No shape for LHS");
//		if( VN_CHILD_SHAPE(enp,1) != NO_SHAPE )
//			prec=SHP_PREC(VN_CHILD_SHAPE(enp,1));
//		else	prec=PREC_SP;			/* BUG cautious compute_assign_shape */
//		POINT_NODE_SHAPE(VN_CHILD(enp,0),uk_shape(prec));
//		VN_CHILD_SHAPE(enp,0) = get_child_shape(QSP_ARG  enp,0);
//	}
	assert( VN_CHILD_SHAPE(enp,0) != NO_SHAPE );

//	if( VN_CHILD_SHAPE(enp,1) == NO_SHAPE ){
//		/* a syntax error?? */
//		NODE_ERROR(enp);
//		sprintf(ERROR_STRING,"CAUTIOUS:  compute_assign_shape:  No shape for RHS (%s node n%d)",
//			NNAME(VN_CHILD(enp,1)),VN_SERIAL(VN_CHILD(enp,1)));
//		WARN(ERROR_STRING);
//		// left child shape can't be null because of block above...
//		prec=SHP_PREC(VN_CHILD_SHAPE(enp,0));
//		POINT_NODE_SHAPE(VN_CHILD(enp,1),uk_shape(prec));
//	}
//#endif /* CAUTIOUS */
	assert( VN_CHILD_SHAPE(enp,1) != NO_SHAPE );


	/* Check for both shapes known */

	if( (! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0))) && ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,1)) ){

		/* make sure the shape of the expression is compatible
		 * with that of the target
		 */

//sprintf(ERROR_STRING,"compute_assign_shape %s:  both child shapes are known.",
//node_desc(enp));
//advise(ERROR_STRING);
		if( (shpp=shapes_mate(VN_CHILD(enp,0),VN_CHILD(enp,1),enp)) == NO_SHAPE ){
			if( (shpp=compatible_shape(VN_CHILD(enp,0),VN_CHILD(enp,1),enp)) == NO_SHAPE ){
				/* Here we might want to allow assignment of a row to an image... */
				NODE_ERROR(enp);
				WARN("compute_assign_shape:  assignment shapes do not mate");
				advise(node_desc(VN_CHILD(enp,0)));
				DESCRIBE_SHAPE(VN_CHILD_SHAPE(enp,0));
				advise(node_desc(VN_CHILD(enp,1)));
				DESCRIBE_SHAPE(VN_CHILD_SHAPE(enp,1));
DUMP_TREE(enp);
				CURDLE(enp)
			} else {
				/* Now we know that we are doing something odd,
				 * like assigning an image from a row or a column...
				 */
				SET_VN_CODE(enp, T_DIM_ASSIGN);
				copy_node_shape(enp,shpp);
				return;
			}
		}

		POINT_NODE_SHAPE(enp,shpp);
		return;
	}

	/* Now we know that at least one of the shapes is unknown */

	/* Check if both are unknown */

	if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) && UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,1)) ){

		/* Both sides of the assignment have unknown
		 * shape.  We need to scan both sides, and update
		 * everyone's lists...
		 */
//sprintf(ERROR_STRING,"compute_assign_shape %s:  both child shapes are UNknown.",
//node_desc(enp));
//advise(ERROR_STRING);

		/* We used to point to the shape of one of the unknown nodes, but that created
		 * a problem:  if the pointed to object resolved during execution,
		 * it would update the assign node's shape making everything
		 * look hunky-dory, so runtime resolution would not be applied
		 * to the other branch.
		 * Therefore we call uk_shape
		 */

		POINT_NODE_SHAPE(enp, uk_shape(SHP_PREC(VN_CHILD_SHAPE(enp,0))) );

		/* remember_uk_assignment(enp); */
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"compute_assign_shape calling link_uk_nodes for %s and child %s",
node_desc(enp),node_desc(VN_CHILD(enp,0)));
advise(ERROR_STRING);
sprintf(ERROR_STRING,"compute_assign_shape calling link_uk_nodes for %s and child %s",
node_desc(enp),node_desc(VN_CHILD(enp,1)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		LINK_UK_NODES(enp,VN_CHILD(enp,0));
		LINK_UK_NODES(enp,VN_CHILD(enp,1));
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

	if( ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){		/* LHS shape known */
		LINK_UK_NODES(enp,VN_CHILD(enp,1));
		/* we could resolve here and now? */

		POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));

		return;
	}

	if( ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,1)) ){		/* RHS shape known */
//sprintf(ERROR_STRING,"compute_assign_shape %s:  RHS shape is known.",
//node_desc(enp));
//advise(ERROR_STRING);
		if( !SCALAR_SHAPE(VN_CHILD_SHAPE(enp,1)) )
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,1));
		else
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));

		LINK_UK_NODES(enp,VN_CHILD(enp,0));

		return;
	}
} /* end compute_assign_shape */


static const char *struct_name(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	char buf[LLEN];
	static char sn[LLEN];

	switch(VN_CODE(enp)){
		case T_STRUCT:
			sprintf(buf,"%s.%s",VN_STRING(enp),struct_name(QSP_ARG  VN_CHILD(enp,0)));
			strcpy(sn,buf);
			return(sn);

		case T_DYN_OBJ:
			return(VN_STRING(enp));

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

static int _count_name_refs(QSP_ARG_DECL  Vec_Expr_Node *enp,const char *lhs_name)
{
	int i1,i2,i3,i4;

	switch(VN_CODE(enp)){
		case T_STRUCT:
			/* check the node itself, and its child,
			 * This is probably wrong, but we do it
			 * until we understand...
			 */
			if( strcmp( struct_name(QSP_ARG  enp), lhs_name ) ){
				SET_VN_LHS_REFS(enp,0);
			} else {
				SET_VN_LHS_REFS(enp,1);
			}
			break;

		case T_FUNCPTR:
		case T_POINTER:
		case T_STR_PTR:
		case T_DYN_OBJ:		// _count_name_refs
			if( strcmp(VN_STRING(enp),lhs_name) ){
				SET_VN_LHS_REFS(enp,0);
			} else {
				SET_VN_LHS_REFS(enp,1);
			}
			break;
		case T_STATIC_OBJ:
			if( strcmp(OBJ_NAME(VN_OBJ(enp)),lhs_name) ){
				SET_VN_LHS_REFS(enp,0);
			} else {
				SET_VN_LHS_REFS(enp,1);
			}
			break;

		/* 4 children to check */
		case T_VV_VV_CONDASS:
		case T_VS_VV_CONDASS:
		case T_VV_VS_CONDASS:
		case T_VS_VS_CONDASS:
			i1 = count_name_refs(VN_CHILD(enp,0),lhs_name);
			i2 = count_name_refs(VN_CHILD(enp,1),lhs_name);
			i3 = count_name_refs(VN_CHILD(enp,2),lhs_name);
			i4 = count_name_refs(VN_CHILD(enp,3),lhs_name);
			SET_VN_LHS_REFS(enp, i1+i2+i3+i4);
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
			i1 = count_name_refs(VN_CHILD(enp,0),lhs_name);
			i2 = count_name_refs(VN_CHILD(enp,1),lhs_name);
			i3 = count_name_refs(VN_CHILD(enp,2),lhs_name);
			SET_VN_LHS_REFS(enp, i1+i2+i3);
			break;

		case T_SUBVEC:
			i1 = count_name_refs(VN_CHILD(enp,0),lhs_name);
            
			if( VN_CHILD(enp,1) == NO_VEXPR_NODE )
				i2=0;
			else
				i2 = count_name_refs(VN_CHILD(enp,1),lhs_name);
            
			if( VN_CHILD(enp,2) == NO_VEXPR_NODE )
				i3=0;
			else
				i3 = count_name_refs(VN_CHILD(enp,2),lhs_name);

			SET_VN_LHS_REFS(enp, i1+i2+i3);
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

			i1 = count_name_refs(VN_CHILD(enp,0),lhs_name);
			i2 = count_name_refs(VN_CHILD(enp,1),lhs_name);
			SET_VN_LHS_REFS(enp, i1+i2);
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

			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
				SET_VN_LHS_REFS(enp,0);
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
		case T_INT1_VFN:
		case T_INT1_FN:
		case T_MATH1_VFN:
		case T_MATH1_FN:
		case T_STRV_FN:
		case T_CHAR_FN:
		case T_CHAR_VFN:
		case T_SUM:
		case T_LIST_OBJ:		/* count_name_refs */
		case T_COMP_OBJ:		/* count_name_refs */
		case T_MAX_INDEX:
		case T_MIN_INDEX:
		case T_CONJ:
			SET_VN_LHS_REFS(enp, count_name_refs(VN_CHILD(enp,0),lhs_name));
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
			SET_VN_LHS_REFS(enp,0);
			break;

		case T_MAX_TIMES:
			SET_VN_LHS_REFS(enp, count_name_refs(VN_CHILD(enp,2),lhs_name));
			break;

		default:
			MISSING_CASE(enp,"count_name_refs");
			SET_VN_LHS_REFS(enp,0);
			break;
	}
	return(VN_LHS_REFS(enp));
}

/* Scan an assignment tree, looking on the right side for references
 * to the lhs target.  If there is more than one, we will need to use
 * a temp obj for results, to avoid overwriting the lhs before we are
 * finished with its old value.
 */

static void _count_lhs_refs(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	const char *lhs_name;

	/* find the name of the lhs object */

	lhs_name = GET_LHS_NAME(VN_CHILD(enp,0));

	/* we may get a null name if we have a variable string ( obj_of(ptr) )... */
	if( lhs_name == NULL ){
		SET_VN_LHS_REFS(enp, 0);
		return;
	}

	SET_VN_LHS_REFS(enp, count_name_refs(VN_CHILD(enp,1),lhs_name));
}

static void set_vv_bool_code(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	//VERIFY_DATA_TYPE(enp,ND_FUNC,"set_vv_bool_code")
	ASSERT_NODE_DATA_TYPE( enp, ND_FUNC )
	switch(VN_CODE(enp)){
		case T_BOOL_LT: SET_VN_BM_CODE(enp, FVVMLT); break;
		case T_BOOL_GT: SET_VN_BM_CODE(enp, FVVMGT); break;
		case T_BOOL_LE: SET_VN_BM_CODE(enp, FVVMLE); break;
		case T_BOOL_GE: SET_VN_BM_CODE(enp, FVVMGE); break;
		case T_BOOL_NE: SET_VN_BM_CODE(enp, FVVMNE); break;
		case T_BOOL_EQ: SET_VN_BM_CODE(enp, FVVMEQ); break;

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

//#ifdef CAUTIOUS
		default:
//			WARN(ERROR_STRING);
//			sprintf(ERROR_STRING,"CAUTIOUS:  unexpected case (%s) in set_vv_bool_code",
//				VF_NAME( FIND_VEC_FUNC( VN_VFUNC_CODE(enp) ) ) );
			assert( AERROR("unexpected case in set_vv_bool_code!?") );
			break;
//#endif /* CAUTIOUS */
	}
}


/* Set the code field for a boolean test node */

static void set_vs_bool_code(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	switch(VN_CODE(enp)){
		case T_BOOL_NOT:
		case T_DYN_OBJ:
			return;
			break;
		default:
			break;
	}

	//VERIFY_DATA_TYPE(enp,ND_FUNC,"set_vs_bool_code")
	ASSERT_NODE_DATA_TYPE( enp, ND_FUNC )
	switch(VN_CODE(enp)){
		case T_BOOL_LT: SET_VN_BM_CODE(enp, FVSMLT); break;
		case T_BOOL_GT: SET_VN_BM_CODE(enp, FVSMGT); break;
		case T_BOOL_LE: SET_VN_BM_CODE(enp, FVSMLE); break;
		case T_BOOL_GE: SET_VN_BM_CODE(enp, FVSMGE); break;
		case T_BOOL_NE: SET_VN_BM_CODE(enp, FVSMNE); break;
		case T_BOOL_EQ: SET_VN_BM_CODE(enp, FVSMEQ); break;

		/* No-ops */
		case T_BOOL_AND: break;
		case T_BOOL_XOR: break;
		case T_BOOL_OR: break;

//#ifdef CAUTIOUS
		default:
//			advise(node_desc(enp));
//			sprintf(ERROR_STRING,"CAUTIOUS:  unexpected case in set_vs_bool_code");
//			WARN(ERROR_STRING);
			assert( AERROR("unexecpected case in set_vs_bool_code!?") );
			break;
//#endif /* CAUTIOUS */
	}
} /* end set_vs_bool_code */

static void commute_bool_test(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	switch(VN_CODE(enp)){
		case T_BOOL_GT:  SET_VN_CODE(enp, T_BOOL_LT); break;
		case T_BOOL_LT:  SET_VN_CODE(enp, T_BOOL_GT); break;
		case T_BOOL_GE:  SET_VN_CODE(enp, T_BOOL_LE); break;
		case T_BOOL_LE:  SET_VN_CODE(enp, T_BOOL_GE); break;
		case T_BOOL_EQ:  break;
		case T_BOOL_NE:  break;
		default:
			MISSING_CASE(enp,"commute_bool_test");
			break;
	}
}

/* set_bool_vecop_code
 *
 */

static void set_bool_vecop_code(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	//int vector_scalar;

	//vector_scalar=0;	/* default */
	if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
		if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
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
	} else if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
		/* vector scalar op, first node is
		 * the scalar, so we interchange the nodes.
		 * We also need to then invert the sense of the comparison.
		 */
		Vec_Expr_Node *tmp_enp;

		tmp_enp = VN_CHILD(enp,0);
		SET_VN_CHILD(enp,0, VN_CHILD(enp,1));
		SET_VN_CHILD(enp,1, tmp_enp);
//DUMP_TREE(enp);
		commute_bool_test(QSP_ARG  enp);

		set_vs_bool_code(QSP_ARG  enp);
	}
//#ifdef CAUTIOUS
	else {
//		WARN("CAUTIOUS:  set_bool_vecop_code:  no vector shapes");
		assert( AERROR("set_bool_vecop_code:  no vector shapes!?") );
	}
//#endif /* CAUTIOUS */
}

static void note_uk_objref(QSP_ARG_DECL  Vec_Expr_Node *decl_enp, Vec_Expr_Node *enp)
{
	Node *np;

	VERIFY_DATA_TYPE(decl_enp,ND_DECL,"note_uk_objref")

	ASSERT_NODE_DATA_TYPE( decl_enp, ND_DECL )

	if( VN_DECL_REFS(decl_enp) == NO_LIST )
		SET_VN_DECL_REFS(decl_enp, NEW_LIST );

	np=mk_node(enp);
	addTail(VN_DECL_REFS(decl_enp),np);
}

#ifdef FOOBAR
static Identifier *get_named_ptr(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Identifier *idp;

	switch(VN_CODE(enp)){
		case T_POINTER:
			idp = ID_OF(VN_STRING(enp));

//			if( idp == NO_IDENTIFIER ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,
//					"CAUTIOUS:  get_named_ptr:  missing identifier %s",VN_STRING(enp));
//				WARN(ERROR_STRING);
//				return(NO_IDENTIFIER);
//			}
			assert( idp != NO_IDENTIFIER );

//			if( ! IS_POINTER(idp) ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,
//					"CAUTIOUS:  get_named_ptr:  id %s is not a pointer!?",ID_NAME(idp));
//				WARN(ERROR_STRING);
//				return(NO_IDENTIFIER);
//			}
			assert( IS_POINTER(idp) );

			return(idp);
		default:
			MISSING_CASE(enp,"get_named_ptr");
			break;
	}
	return(NO_IDENTIFIER);
}
#endif // FOOBAR

/* check_curds - check a tree to see if any nodes are "curdled", such as
 * a syntax error or something else that would preclude execution.
 * Returns 0 for a good tree, -1 otherwise.
 */

static int check_curds(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int i;

	switch(VN_CODE(enp)){
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
		if( VN_CHILD(enp,i) != NO_VEXPR_NODE && IS_CURDLED(VN_CHILD(enp,i)) ){
			CURDLE(enp)
			return(-1);
		}

	return(0);
} /* end check_curds */

#ifdef SUPPORT_MATLAB_MODE

static Vec_Expr_Node *_one_matlab_subscript(QSP_ARG_DECL   Vec_Expr_Node *obj_enp,Vec_Expr_Node *subscr_enp)
{
	Vec_Expr_Node *minus_one_enp, *enp1, *enp2, *new_enp;

	switch(VN_CODE(subscr_enp)){
		case T_ENTIRE_RANGE:
			new_enp = NODE3(T_SUBVEC,obj_enp,NULL,NULL);
			new_enp=compile_node(new_enp); /* needed?? */
			prelim_node_shape(new_enp);
			/* BUG we should clean up the discarded nodes */
			/* BUG? do we need to compile the plus nodes? */
			return(new_enp);
			
		case T_INDEX_SPEC:		/* one_matlab_subscript */
		case T_RANGE2:			/* one_matlab_subscript */
			/* subtract 1 from the starting index */
			minus_one_enp = NODE0(T_LIT_INT);
			SET_VN_INTVAL(minus_one_enp, -1);
			prelim_node_shape(minus_one_enp);
			enp1=NODE2(T_PLUS,VN_CHILD(subscr_enp,0),minus_one_enp);
			enp1=compile_node(enp1); /* needed?? */
			prelim_node_shape(enp1);

			/* subtract one from the final index */
			minus_one_enp = NODE0(T_LIT_INT);
			SET_VN_INTVAL(minus_one_enp, -1);
			prelim_node_shape(minus_one_enp);
			enp2=NODE2(T_PLUS,VN_CHILD(subscr_enp,1),minus_one_enp);
			enp2=compile_node(enp2); /* needed? */
			prelim_node_shape(enp2);

			new_enp = NODE3(T_SUBVEC,obj_enp,enp1,enp2);
			new_enp=compile_node(new_enp); /* needed?? */
			prelim_node_shape(new_enp);
			/* BUG we should clean up the discarded nodes */
			/* BUG? do we need to compile the plus nodes? */
			return(new_enp);
		case T_LIT_DBL:
		case T_LIT_INT:
		case T_DYN_OBJ:	// _one_matlab_subscript
		ALL_UNMIXED_SCALAR_BINOP_CASES
			/*
			minus_one_enp = NODE0(T_LIT_INT);
			SET_VN_INTVAL(minus_one_enp, -1);
			enp1=NODE2(T_PLUS,subscr_enp,minus_one_enp);
			new_enp = NODE2(T_SQUARE_SUBSCR,obj_enp,enp1);
			*/
			new_enp = NODE2(T_SQUARE_SUBSCR,obj_enp,subscr_enp);
			new_enp = compile_node(new_enp);
			prelim_node_shape(new_enp);
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

static Vec_Expr_Node *_compile_matlab_subscript(QSP_ARG_DECL   Vec_Expr_Node *obj_enp,Vec_Expr_Node *subscr_enp)
{
	Vec_Expr_Node *new_enp;

	switch(VN_CODE(subscr_enp)){
		case T_INDEX_LIST:
			new_enp = one_matlab_subscript(obj_enp,VN_CHILD(subscr_enp,0));
			new_enp = compile_matlab_subscript(new_enp,VN_CHILD(subscr_enp,1));
			return(new_enp);
		case T_DYN_OBJ:
		case T_LIT_INT:
		case T_LIT_DBL:
		ALL_UNMIXED_SCALAR_BINOP_CASES
			/* BUG all operators should go here? */
			new_enp = one_matlab_subscript(obj_enp,subscr_enp);
			return(new_enp);
		case T_INDEX_SPEC:
		case T_RANGE2:			/* compile_matlab_subscript */
		case T_ENTIRE_RANGE:
			new_enp = one_matlab_subscript(obj_enp,subscr_enp);
			return(new_enp);
		default:
			MISSING_CASE(subscr_enp,"compile_matlab_subscript");
			DUMP_TREE(subscr_enp);
			break;
	}
	return(NO_VEXPR_NODE);
}

#endif /* SUPPORT_MATLAB_MODE */

/* The child node is supposed to be a bitmap...
 * But since we can use an object as a truth value (implies a!=0),
 * we have to handle this case.
 * Boolean truth operators, we descend recursively.
 */

static void _check_bitmap_arg(QSP_ARG_DECL Vec_Expr_Node *enp,int index)
{
	Vec_Expr_Node *enp2,*enp3;

	switch(VN_CODE(VN_CHILD(enp,index))){

		/*
		ALL_BINOP_CASES
		*/
		ALL_OBJREF_CASES
			/* If the object is not a bitmap, then we have to test != 0 */
			/* do we have the shape? */
			/* if( BITMAP_SHAPE(VN_SHAPE(VN_CHILD(enp,index))) ) */
			if( BITMAP_PRECISION(SHP_PREC(VN_SHAPE(VN_CHILD(enp,index)))) )
				break;

			/* else fall through if not bitmap */

		// added to support isnan etc.
		case T_MATH1_FN:
		case T_MATH1_VFN:
		case T_INT1_FN:
		case T_INT1_VFN:
		// BUG other functions should be here too!

		case T_VV_FUNC:			/* check_bitmap_arg */
		case T_VS_FUNC:
			//VERIFY_DATA_TYPE(enp,ND_FUNC,"check_bitmap_arg")
			ASSERT_NODE_DATA_TYPE( enp, ND_FUNC )
			if( FLOATING_PREC(VN_CHILD_PREC(enp,index)) ){
				enp2 = NODE0(T_LIT_INT);
				SET_VN_INTVAL(enp2, 0);
			} else {
				enp2 = NODE0(T_LIT_DBL);
				SET_VN_DBLVAL(enp2, 0.0);
			}
			prelim_node_shape(enp2);

			enp3 = NODE2(T_BOOL_NE,VN_CHILD(enp,index),enp2);
			SET_VN_BM_CODE(enp3, FVSMNE);
			prelim_node_shape(enp3);

			SET_VN_CHILD(enp,index, enp3);
			SET_VN_PARENT(enp3, enp);

			break;

		ALL_NUMERIC_COMPARISON_CASES		/* check_bitmap_arg */
			/* do nothing */
			break;

		BINARY_BOOLOP_CASES			/* &&, ||, ^^ */
			/* We are treating these like logical ops -
			 * but are they also bitwise operators on vectors??
			 */
			enp = VN_CHILD(enp,index);
			check_bitmap_arg(enp,0);
			check_bitmap_arg(enp,1);
			break;

		case T_BOOL_NOT: /* check_bitmap_arg */
			enp = VN_CHILD(enp,index);
			check_bitmap_arg(enp,0);
			break;

		default:
			MISSING_CASE(VN_CHILD(enp,index),"check_bitmap_arg");
			break;
	}
} /* check_bitmap_arg */

/* A utility routine for v/s and v-s...
 * We insert a new node between this node and its second child.
 */

static void invert_op(QSP_ARG_DECL  Vec_Expr_Node *enp,Vec_Func_Code wc,Tree_Code new_tc)
{
	Vec_Expr_Node *new_enp;

	//VERIFY_DATA_TYPE(enp,ND_FUNC,"invert_op")
	ASSERT_NODE_DATA_TYPE( enp, ND_FUNC )

	SET_VN_VFUNC_CODE(enp, wc);

	/* this just suppresses a warning in node1() */
	SET_VN_PARENT(VN_CHILD(enp,1), NO_VEXPR_NODE);

	new_enp = NODE1(new_tc,VN_CHILD(enp,1));
	POINT_NODE_SHAPE(new_enp,VN_CHILD_SHAPE(enp,1));

	SET_VN_CHILD(enp,1, new_enp);
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

	//VERIFY_DATA_TYPE(enp,ND_FUNC,"check_arith_code")
	ASSERT_NODE_DATA_TYPE( enp, ND_FUNC )
	vector_scalar=0;	/* default */
	if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
		if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
			/* vector-vector op */
			if( VN_CODE(enp) == T_PLUS ){
				SET_VN_VFUNC_CODE(enp, FVADD);
			} else if( VN_CODE(enp) == T_MINUS ){
				SET_VN_VFUNC_CODE(enp, FVSUB);
			} else if( VN_CODE(enp) == T_TIMES ){
				SET_VN_VFUNC_CODE(enp, FVMUL);
			} else if( VN_CODE(enp) == T_DIVIDE ){
				SET_VN_VFUNC_CODE(enp, FVDIV);
			} else if( VN_CODE(enp) == T_MODULO ){
				SET_VN_VFUNC_CODE(enp, FVMOD);
			} else if( VN_CODE(enp) == T_BITRSHIFT ){
				SET_VN_VFUNC_CODE(enp, FVSHR);
			} else if( VN_CODE(enp) == T_BITLSHIFT ){
				SET_VN_VFUNC_CODE(enp, FVSHL);
			} else if( VN_CODE(enp) == T_BITAND ){
				SET_VN_VFUNC_CODE(enp, FVAND);
			} else if( VN_CODE(enp) == T_BITOR ){
				SET_VN_VFUNC_CODE(enp, FVOR);
			} else if( VN_CODE(enp) == T_BITXOR ){
				SET_VN_VFUNC_CODE(enp, FVXOR);
			}
//#ifdef CAUTIOUS
			  else {
//				sprintf(ERROR_STRING,
//	"CAUTIOUS:  check_arith_code:  unhandled vector-vector operation %s",NNAME(enp));
//				WARN(ERROR_STRING);
				assert( AERROR("check_arith_code:  unhandled vector-vector op!?") );
			}
//#endif /* CAUTIOUS */

			SET_VN_CODE(enp, T_VV_FUNC);
			/* minus node points to shape but vv_func must own */

			{
			Shape_Info *tmp_shpp;
			tmp_shpp = VN_SHAPE(enp);
			discard_node_shape(enp);
			copy_node_shape(enp,tmp_shpp);
			}
		} else {
			/* vector scalar operation,
			 * second operand is the scalar,
			 * so we leave the order unchanged.
			 */
			vector_scalar=2;
		}
	} else if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
		/* vector scalar op, first node is
		 * the scalar, so we interchange the nodes
		 */
		Vec_Expr_Node *tmp_enp;

		tmp_enp = VN_CHILD(enp,0);
		SET_VN_CHILD(enp,0, VN_CHILD(enp,1));
		SET_VN_CHILD(enp,1, tmp_enp);
		vector_scalar=1;
	}

	/* else scalar-scalar - do nothing */

	if( vector_scalar ){
		/* vector-vector op */
		if( VN_CODE(enp) == T_PLUS ){
			SET_VN_VFUNC_CODE(enp, FVSADD);
		} else if( VN_CODE(enp) == T_MINUS ){
			if( vector_scalar==1 ){	/* scalar is 1st op */
				SET_VN_VFUNC_CODE(enp, FVSSUB);
			} else {
				invert_op(QSP_ARG  enp,FVSADD,T_UMINUS);
			}
		} else if( VN_CODE(enp) == T_TIMES ){
			SET_VN_VFUNC_CODE(enp, FVSMUL);
		} else if( VN_CODE(enp) == T_BITOR ){
			SET_VN_VFUNC_CODE(enp, FVSOR);
		} else if( VN_CODE(enp) == T_BITAND ){
			SET_VN_VFUNC_CODE(enp, FVSAND);
		} else if( VN_CODE(enp) == T_BITXOR ){
			SET_VN_VFUNC_CODE(enp, FVSXOR);
		} else if( VN_CODE(enp) == T_DIVIDE ){
			if( vector_scalar==1 ){	/* scalar is 1st op */
				SET_VN_VFUNC_CODE(enp, FVSDIV);
			} else {
				/* invert_op(QSP_ARG  enp,FVSMUL,T_RECIP); */
				SET_VN_VFUNC_CODE(enp, FVSDIV2);
			}
		} else if ( VN_CODE(enp) == T_MODULO ){
			if( vector_scalar==1 ){ /* scalar is 1st op */
				SET_VN_VFUNC_CODE(enp, FVSMOD2);
			} else {		/* scalar is 2nd op */
				SET_VN_VFUNC_CODE(enp, FVSMOD);
			}
		} else if( VN_CODE(enp) == T_BITLSHIFT ){
			if( vector_scalar==1 ){ /* scalar is 1st op */
				SET_VN_VFUNC_CODE(enp, FVSSHL2);
			} else {		/* scalar is 2nd op */
				SET_VN_VFUNC_CODE(enp, FVSSHL);
			}
		} else if( VN_CODE(enp) == T_BITRSHIFT ){
			if( vector_scalar==1 ){ /* scalar is 1st op */
				SET_VN_VFUNC_CODE(enp, FVSSHR2);
			} else {		/* scalar is 2nd op */
				SET_VN_VFUNC_CODE(enp, FVSSHR);
			}
		}
//#ifdef CAUTIOUS
		  else {
//			sprintf(ERROR_STRING,
//	"CAUTIOUS:  check_arith_code:  unhandled vector-scalar operation %s",NNAME(enp));
//			WARN(ERROR_STRING);
			assert( AERROR("check_arith_code:  unhandled vector-scalar op!?") );
		}
//#endif /* CAUTIOUS */

		SET_VN_CODE(enp, T_VS_FUNC);
	}
} /* check_arith_code */

/* Make sure that the bitmap has the proper shape for the target node */

static void _check_mating_bitmap(QSP_ARG_DECL  Vec_Expr_Node *enp,Vec_Expr_Node *bm_enp)
{
//#ifdef CAUTIOUS
//	if( VN_SHAPE(enp) == NO_SHAPE || VN_SHAPE(bm_enp) == NO_SHAPE ){
//		NODE_ERROR(enp);
//		NWARN("CAUTIOUS:  check_mating_bitmap:  missing shape");
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( VN_SHAPE(enp) != NO_SHAPE );
	assert( VN_SHAPE(bm_enp) != NO_SHAPE );

	if( SCALAR_SHAPE(VN_SHAPE(bm_enp)) )
		return;	/* not really a bitmap! */

	if( UNKNOWN_SHAPE(VN_SHAPE(bm_enp)) ){
		/*
		sprintf(ERROR_STRING,"check_bitmap_shape:  bitmap %s has unknown shape",node_desc(bm_enp));
		advise(ERROR_STRING);
		*/
		return;
	}
	if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
		/*
		sprintf(ERROR_STRING,"check_bitmap_shape:  %s has unknown shape",node_desc(enp));
		advise(ERROR_STRING);
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
			if( SHP_TYPE_DIM(VN_SHAPE(bm_enp),i) > 1 ){
				if( SHP_TYPE_DIM(VN_SHAPE(enp),i) == 1 ){
					SET_SHP_TYPE_DIM(VN_SHAPE(enp),i, SHP_TYPE_DIM(VN_SHAPE(bm_enp),i) );
					enlarged = 1;
				} else if( SHP_TYPE_DIM(VN_SHAPE(enp),i) != SHP_TYPE_DIM(VN_SHAPE(bm_enp),i) ){
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
			auto_shape_flags(VN_SHAPE(enp),NO_OBJ);
	}
} /* check_mating_bitmap */

/* We call this when we have to invert the order of the arguments in a conditional assignment... */

static void invert_bool_test(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	switch(VN_CODE(enp)){
		/* This might be a little confusing if someone were to look at the tree...
		 * because the function codes don't match the tree codes...
		 */
		case T_BOOL_GT:  SET_VN_CODE(enp, T_BOOL_LE); break;
		case T_BOOL_LT:  SET_VN_CODE(enp, T_BOOL_GE); break;
		case T_BOOL_GE:  SET_VN_CODE(enp, T_BOOL_LT); break;
		case T_BOOL_LE:  SET_VN_CODE(enp, T_BOOL_GT); break;
		case T_BOOL_EQ:  SET_VN_CODE(enp, T_BOOL_NE); break;
		case T_BOOL_NE:  SET_VN_CODE(enp, T_BOOL_EQ); break;
		/* Here we need to either insert a not node, or change the code.
		 * Changing the code would be simpler, but we don't have BOOL_NXOR, etc.
		 */
		case T_BOOL_AND:
		case T_BOOL_XOR:
		case T_BOOL_OR:
		// these cases were added after we added isnan, isinf, etc
		case T_MATH1_FN:
		case T_MATH1_VFN:
		case T_INT1_FN:
		case T_INT1_VFN:

		case T_DYN_OBJ:	// invert_bool_test
			{
			Vec_Expr_Node *new_enp;
			Vec_Expr_Node *parent;

			/* Need to insert an inversion node */

			parent = VN_PARENT(enp);		/* save this, gets reset in node1() */
			new_enp=NODE1(T_BOOL_NOT,enp);
			/* can we assume this is the first child? */
//#ifdef CAUTIOUS
//			if( VN_CHILD(VN_PARENT(enp),0) != enp ){
//				sprintf(ERROR_STRING,"CAUTIOUS:  invert_bool_test:  %s is not first child of %s!?",
//					node_desc(enp),node_desc(VN_PARENT(enp)));
//				ERROR1(ERROR_STRING);
//			}
//#endif /* CAUTIOUS */
			assert( VN_CHILD(VN_PARENT(enp),0) == enp );

			SET_VN_CHILD(parent,0, new_enp);
			SET_VN_PARENT(new_enp, parent);
			prelim_node_shape(new_enp);
			break;
			}

		default:
			MISSING_CASE(enp,"invert_bool_test");
			break;
	}
}

static Vec_Func_Code vv_vv_test_code(Tree_Code bool_code)
{
	switch(bool_code){
		case T_BOOL_GT: return(FVV_VV_GT); break;
		case T_BOOL_LT: return(FVV_VV_LT); break;
		case T_BOOL_GE: return(FVV_VV_GE); break;
		case T_BOOL_LE: return(FVV_VV_LE); break;
		case T_BOOL_EQ: return(FVV_VV_EQ); break;
		case T_BOOL_NE: return(FVV_VV_NE); break;
//#ifdef CAUTIOUS
		default:
//			NERROR1("CAUTIOUS:  vv_vv_test_code:  unexpected boolean test");
			assert( AERROR("vv_vv_test_code:  unexpected boolean test") );
			break;
//#endif /* CAUTIOUS */
	}
	return(INVALID_VFC);	// should never happen?
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
//#ifdef CAUTIOUS
		default:
//			NERROR1("CAUTIOUS:  vv_vs_test_code:  unexpected boolean test");
			assert( AERROR("vv_vs_test_code:  unexpected boolean test") );
			break;
//#endif /* CAUTIOUS */
	}
	return(INVALID_VFC);	// should never happen?
}


static void invert_vec4_condass(Vec_Expr_Node *enp)
{
	switch(VN_BM_CODE(enp)){
		case FVV_VV_GT: SET_VN_BM_CODE(enp, FVV_VV_LE); break;
		case FVV_VV_LE: SET_VN_BM_CODE(enp, FVV_VV_GT); break;
		case FVV_VV_LT: SET_VN_BM_CODE(enp, FVV_VV_GE); break;
		case FVV_VV_GE: SET_VN_BM_CODE(enp, FVV_VV_LT); break;

		case FVV_VS_GT: SET_VN_BM_CODE(enp, FVV_VS_LE); break;
		case FVV_VS_LE: SET_VN_BM_CODE(enp, FVV_VS_GT); break;
		case FVV_VS_LT: SET_VN_BM_CODE(enp, FVV_VS_GE); break;
		case FVV_VS_GE: SET_VN_BM_CODE(enp, FVV_VS_LT); break;

		case FVS_VV_GT: SET_VN_BM_CODE(enp, FVS_VV_LE); break;
		case FVS_VV_LE: SET_VN_BM_CODE(enp, FVS_VV_GT); break;
		case FVS_VV_LT: SET_VN_BM_CODE(enp, FVS_VV_GE); break;
		case FVS_VV_GE: SET_VN_BM_CODE(enp, FVS_VV_LT); break;

		case FVS_VS_GT: SET_VN_BM_CODE(enp, FVS_VS_LE); break;
		case FVS_VS_LE: SET_VN_BM_CODE(enp, FVS_VS_GT); break;
		case FVS_VS_LT: SET_VN_BM_CODE(enp, FVS_VS_GE); break;
		case FVS_VS_GE: SET_VN_BM_CODE(enp, FVS_VS_LT); break;

		case FVV_VV_EQ:
		case FVV_VV_NE:
		case FVV_VS_EQ:
		case FVV_VS_NE:
		case FVS_VV_EQ:
		case FVS_VV_NE:
		case FVS_VS_EQ:
		case FVS_VS_NE:
			break;
//#ifdef CAUTIOUS
		default:
//			NERROR1("CAUTIOUS:  invert_vec4_condass:  unexpected function code!?");
			assert( AERROR("invert_vec4_condass:  unexpected function code!?") );
			break;
//#endif /* CAUTIOUS */
	}
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
//#ifdef CAUTIOUS
		default:
//			NERROR1("CAUTIOUS:  vs_vv_test_code:  unexpected boolean test");
			assert( AERROR("vs_vv_test_code:  unexpected boolean test") );
			break;
//#endif /* CAUTIOUS */
	}
	return(INVALID_VFC);	// should never happen?
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
//#ifdef CAUTIOUS
		default:
//			NERROR1("CAUTIOUS:  vs_vs_test_code:  unexpected boolean test");
			assert( AERROR("vs_vs_test_code:  unexpected boolean test") );
			break;
//#endif /* CAUTIOUS */
	}
	return(INVALID_VFC);	// should never happen?
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
	SET_VN_CHILD(enp,0, VN_CHILD(enp,1));		\
	SET_VN_CHILD(enp,1, VN_CHILD(enp,2));		\
	SET_VN_CHILD(enp,2, VN_CHILD(test_enp,0));	\
	SET_VN_PARENT(VN_CHILD(enp,2),enp);		\
	SET_VN_CHILD(enp,3, VN_CHILD(test_enp,1));	\
	SET_VN_PARENT(VN_CHILD(enp,3),enp);

/* swap the order of the test args */

#define FIX_CHILDREN_2					\
							\
	SET_VN_CHILD(enp,0, VN_CHILD(enp,1));		\
	SET_VN_CHILD(enp,1, VN_CHILD(enp,2));		\
	SET_VN_CHILD(enp,3, VN_CHILD(test_enp,0));	\
	SET_VN_PARENT(VN_CHILD(enp,3),enp);		\
	SET_VN_CHILD(enp,2, VN_CHILD(test_enp,1));	\
	SET_VN_PARENT(VN_CHILD(enp,2),enp);

#define CHECK_SHAPES(checkpoint_number,comparison_index)			\
										\
	if( get_mating_shapes(enp,0,comparison_index) == NO_SHAPE ){		\
		sprintf(ERROR_STRING,						\
	"check_xx_xx_condass_code:  bad shapes #%d - shouldn't happen?",	\
			checkpoint_number);					\
		WARN(ERROR_STRING);						\
	}									\
	if( get_mating_shapes(enp,1,comparison_index) == NO_SHAPE ){		\
		sprintf(ERROR_STRING,						\
	"check_xx_xx_condass_code:  bad shapes #%d - shouldn't happen?",	\
			checkpoint_number+1);					\
		WARN(ERROR_STRING);						\
	}

#define RELEASE_BOOL(enp)							\
	SET_VN_CHILD(enp,0,NULL);						\
	SET_VN_CHILD(enp,1,NULL);						\
	rls_vectree(enp);

// We may need to insert a typecast node here...
// This would have been done in compile_node, but because the test
// was a bitmap the child nodes weren't checked...

static void _check_xx_xx_condass_code(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Vec_Expr_Node *test_enp;
	test_enp=VN_CHILD(enp,0);

	if( VN_CODE(enp) == T_VV_B_CONDASS ){	/* both sources vectors? */
		if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(test_enp,0)) ){
			if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(test_enp,1)) ){
				// both test sources vectors
				SET_VN_CODE(enp, T_VV_VV_CONDASS);
				// This "fix" overwrites the reference to child 0
				// Does it exist, and should we release it?
				FIX_CHILDREN
				SET_VN_BM_CODE(enp, vv_vv_test_code(VN_CODE(test_enp)));
				// check to insure typecase if needed
				CHECK_SHAPES(1,2)
				RELEASE_BOOL(test_enp)
			} else {
				// only first test source vector
				SET_VN_CODE(enp, T_VV_VS_CONDASS);
				FIX_CHILDREN
				SET_VN_BM_CODE(enp, vv_vs_test_code(VN_CODE(test_enp)));
				CHECK_SHAPES(3,2)
				RELEASE_BOOL(test_enp)
			}
		} else if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(test_enp,1)) ){
			// only second test source is a vector
			FIX_CHILDREN_2
			SET_VN_BM_CODE(enp, vv_vv_test_code(VN_CODE(test_enp)));
			/* invert sense of test */
			invert_vec4_condass(enp);
			CHECK_SHAPES(5,2)
			RELEASE_BOOL(test_enp)
		}
//#ifdef CAUTIOUS
		  else {
//		  	ERROR1("CAUTIOUS:  check_xx_xx_condass_code should not have been called!?");
		  	assert( AERROR("check_xx_xx_condass_code should not have been called!?") );
		}
//#endif /* CAUTIOUS */
		
	} else if( VN_CODE(enp) == T_VS_B_CONDASS ){	/* second source is a scalar */
		if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(test_enp,0)) ){
			if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(test_enp,1)) ){
				SET_VN_CODE(enp, T_VS_VV_CONDASS);
				FIX_CHILDREN
				SET_VN_BM_CODE(enp, vs_vv_test_code(VN_CODE(test_enp)));
				CHECK_SHAPES(7,2)
				RELEASE_BOOL(test_enp)
			} else {
				SET_VN_CODE(enp, T_VS_VS_CONDASS);
				FIX_CHILDREN
				SET_VN_BM_CODE(enp, vs_vs_test_code(VN_CODE(test_enp)));
				CHECK_SHAPES(9,2)
				RELEASE_BOOL(test_enp)
			}
		} else if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(test_enp,1)) ){
			FIX_CHILDREN_2
			SET_VN_BM_CODE(enp, vs_vv_test_code(VN_CODE(test_enp)));
			invert_vec4_condass(enp);
			CHECK_SHAPES(11,2)
			RELEASE_BOOL(test_enp)
		}
//#ifdef CAUTIOUS
		  else {
//		  	ERROR1("CAUTIOUS:  check_xx_xx_condass_code should not have been called!?");
		  	assert( AERROR("check_xx_xx_condass_code should not have been called!?") );
		}
//#endif /* CAUTIOUS */

	}
//#ifdef CAUTIOUS
	  else {
//	  	ERROR1("CAUTIOUS:  unexpected node code in check_xx_xx_condass_code!?");
	  	assert( AERROR("unexpected node code in check_xx_xx_condass_code!?") );
	}
//#endif /* CAUTIOUS */
}

/* We can use a vector as a logical variable, as in C.
 * When this is encountered, we add a T_BOOL_NE 0 node.
 */

/* check_xx_v_condass code - We know that the condition is a vector (bitmap).
 * We examine the shapes of the two (assignment source) children,
 * determine whether a vector-vector-vector, vector-vector-scalar,
 * or vector-scalar-scalar operation is needed.  If so, change the node code,
 * and set the en_bm_code field with the appropriate vectbl code.
 */

static void _check_xx_v_condass_code(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
		if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,2)) ){
			/* vector-vector op */
			SET_VN_CODE(enp, T_VV_B_CONDASS);
		} else {
			/* vector scalar operation,
			 * second operand is the scalar,
			 * so we leave the order unchanged.
			 */
			SET_VN_CODE(enp, T_VS_B_CONDASS);
		}
	} else {
		if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,2)) ){
			/* vector scalar op, first node is
			 * the scalar, so we interchange the nodes.
			 * After we do this, however, we need to invert
			 * the sense of the boolean code!
			 */
			Vec_Expr_Node *tmp_enp;

			tmp_enp = VN_CHILD(enp,1);
			SET_VN_CHILD(enp,1, VN_CHILD(enp,2));
			SET_VN_CHILD(enp,2, tmp_enp);
			SET_VN_CODE(enp, T_VS_B_CONDASS);
			invert_bool_test(QSP_ARG VN_CHILD(enp,0));
			set_vs_bool_code(QSP_ARG  VN_CHILD(enp,0));
		} else {	/* scalar-scalar */
			SET_VN_CODE(enp, T_SS_B_CONDASS);
		}
	}
	check_bitmap_arg(enp,0);

	/* Now we should see whether the bitmap arg can be reduced to
	 * one of the new, fast condass functions.
	 * That will be true if the test is a numeric comparison.
	 */

	switch( VN_CODE(VN_CHILD(enp,0)) ){
		ALL_NUMERIC_COMPARISON_CASES
			check_xx_xx_condass_code(enp);
			break;
		default:
			break;	/* do nothing */
	}
} /* check_xx_v_condass_code */

static void _check_xx_s_condass_code(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
		if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,2)) ){
			/* vector-vector op */
			SET_VN_CODE(enp, T_VV_S_CONDASS);
		} else {
			/* vector scalar operation,
			 * second operand is the scalar,
			 * so we leave the order unchanged.
			 */
			SET_VN_CODE(enp, T_VS_S_CONDASS);
		}
	} else {
		if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,2)) ){
			/* vector scalar op, first node is
			 * the scalar, so we interchange the nodes.
			 * After we do this, however, we need to invert
			 * the sense of the boolean code!
			 */
			Vec_Expr_Node *tmp_enp;

			tmp_enp = VN_CHILD(enp,1);
			SET_VN_CHILD(enp,1, VN_CHILD(enp,2));
			SET_VN_CHILD(enp,2, tmp_enp);
			SET_VN_CODE(enp, T_VS_S_CONDASS);

			invert_bool_test(QSP_ARG VN_CHILD(enp,0));
			set_vs_bool_code(QSP_ARG  VN_CHILD(enp,0));
		} else {	/* scalar-scalar */
			/* don't think we ever get here... */
			SET_VN_CODE(enp, T_SS_S_CONDASS);
		}
	}
} /* check_xx_s_condass_code */

/* enp is a T_MAXVAL/T_MINVAL node, w/ an T_EXPR_LIST child.
 * We need to determine (from the shapes of the children)
 * what the code should be.
 *
 * This also gets called recursively, if we have more than two args:
 * the EXPR_LIST node can get changed to a MINMAX node...
 */

static void _check_minmax_code(QSP_ARG_DECL   Vec_Expr_Node *enp)
{
	int i;

	/* We can ditch the first exprlist node */

	//VERIFY_DATA_TYPE(enp,ND_FUNC,"check_minmax_code")
	ASSERT_NODE_DATA_TYPE( enp, ND_FUNC )

	// T_MINVAL, T_MAXVAL nodes just have one child after parsing...
	if( VN_CODE(VN_CHILD(enp,0)) == T_EXPR_LIST && VN_CHILD(enp,1)==NULL ){
		Vec_Expr_Node *list_enp;

		list_enp = VN_CHILD(enp,0);

		SET_VN_CHILD(enp,0, VN_CHILD(list_enp,0) );
		SET_VN_PARENT(VN_CHILD(list_enp,0), enp);

		SET_VN_CHILD(enp,1, VN_CHILD(list_enp,1) );
		SET_VN_PARENT(VN_CHILD(list_enp,1), enp);

		/* deallocate the EXPR_LIST node */
		/* null out the children to prevent recursion */
		SET_VN_CHILD(list_enp,0,NULL);
		SET_VN_CHILD(list_enp,1,NULL);
		rls_vectree(list_enp);
	}
else {
sprintf(ERROR_STRING,"check_minmax_code:  expected T_EXPR_LIST in child[0], found %s",
node_desc(VN_CHILD(enp,0)));
WARN(ERROR_STRING);
}

//#ifdef CAUTIOUS
//if( VN_CODE(VN_CHILD(enp,1)) == T_EXPR_LIST ){
//WARN("CAUTIOUS:  check_minmax_code:  EXPR_LIST in child[1]!?");
//}
//#endif // CAUTIOUS
	assert( VN_CODE(VN_CHILD(enp,1)) != T_EXPR_LIST );

	/* One of the new children could be a list... */
	/* We only should need to check one, but which one?  BUG */
	for(i=0;i<2;i++){
		if( VN_CODE(VN_CHILD(enp,i)) == T_EXPR_LIST ){
			SET_VN_CODE(VN_CHILD(enp,i), VN_CODE(enp));
			discard_node_shape(VN_CHILD(enp,i));
			check_minmax_code(VN_CHILD(enp,i));
			/* BUG? or update_node_shape??? */
			prelim_node_shape(enp);
		}
	}

	/* comparing two vectors,
	 * or a vector and a scalar?
	 */
	if( get_mating_shapes(enp,0,1) == NO_SHAPE ){
		WARN("check_minmax_code:  bad shapes!?");
		return;
	}

	/* in the new veclib, we just use T_MAXVAL for everything... or do we?
	 * We use T_MAXVAL instead of PROJECT_OP when we have a single operand
	 * but are not projecting all the way down to a scalar.  We ought to allow
	 * this sort of things with multiple operands as well...
	 */
	if( SCALAR_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
		if( SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
			/* scalar-scalar */
			if( VN_CODE(enp) == T_MAXVAL )
				SET_VN_CODE(enp, T_SCALMAX);
			else
				SET_VN_CODE(enp, T_SCALMIN);
		} else {
			/* vector-scalar */
			if( VN_CODE(enp) == T_MAXVAL ){
				SET_VN_VFUNC_CODE(enp, FVSMAX);
			} else {
				SET_VN_VFUNC_CODE(enp, FVSMIN);
			}
			SET_VN_CODE(enp, T_VS_FUNC);
		}
	} else {
		if( SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
			Vec_Expr_Node *tmp_enp;

			/* scalar-vector */
			if( VN_CODE(enp) == T_MAXVAL ){
				SET_VN_VFUNC_CODE(enp, FVSMAX);
			} else {
				SET_VN_VFUNC_CODE(enp, FVSMIN);
			}
			SET_VN_CODE(enp, T_VS_FUNC);

			/* switch order of children */

			tmp_enp = VN_CHILD(enp,0);
			SET_VN_CHILD(enp,0, VN_CHILD(enp,1));
			SET_VN_CHILD(enp,1, tmp_enp);
		} else {
			/* vector-vector */
			if( VN_CODE(enp) == T_MAXVAL ){
				SET_VN_VFUNC_CODE(enp, FVMAX);
			} else {
				SET_VN_VFUNC_CODE(enp, FVMIN);
			}
			SET_VN_CODE(enp, T_VV_FUNC);
			update_node_shape(enp);
		}
	}
	if( VN_CODE(enp) == T_VS_FUNC ){
		/* vs_func nodes point to their shapes - BUT maxval nodes
		 * own it...
		 */
		discard_node_shape(enp);
		POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
	}

} /* end check_minmax_code */

static int has_vector_subscript(Vec_Expr_Node *enp)
{
	int i;
	int retval=0;

	for(i=0;i<tnt_tbl[VN_CODE(enp)].tnt_nchildren;i++)
		retval |= has_vector_subscript(VN_CHILD(enp,i));

	if( retval ) return(1);

	switch(VN_CODE(enp)){
		case T_SQUARE_SUBSCR:
		case T_CURLY_SUBSCR:
			if( ! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,1)) )
				return 1;
			break;
		default:
			break;
	}
	return(0);
}

/* With any list node, return the number of elements in that list */

static int leaf_count(Vec_Expr_Node *enp,Tree_Code list_code)
{
	if( enp == NO_VEXPR_NODE ) return(0);
	if( VN_CODE(enp) == list_code ){
		int n1,n2;
		n1=leaf_count(VN_CHILD(enp,0),list_code);
		n2=leaf_count(VN_CHILD(enp,1),list_code);
		return(n1+n2);
	}
	return(1);
}


static Vec_Expr_Node * _compile_node(QSP_ARG_DECL   Vec_Expr_Node *enp)
{
	//int vf_code=(-2);
	Vec_Func_Code vf_code=N_VEC_FUNCS;
	Vec_Expr_Node *orig_enp;
#ifdef SUPPORT_MATLAB_MODE
	Vec_Expr_Node *new_enp;
#endif /* SUPPORT_MATLAB_MODE */

	orig_enp = enp;

	/* now all the child nodes have been scanned, process this one */

	/* check for CURDLED children */
	if( MAX_CHILDREN(enp)>0 && check_curds(QSP_ARG  enp) < 0 ) return(enp);

//#ifdef CAUTIOUS
//	if( OWNS_SHAPE(enp) ){
//		sprintf(ERROR_STRING,
//	"CAUTIOUS:  compile_node %s:  already owns shape!?",node_desc(enp));
//		WARN(ERROR_STRING);
//		DUMP_TREE(enp);
//	}
//#endif /* CAUTIOUS */
	assert( ! OWNS_SHAPE(enp) );

	SET_VN_SHAPE(enp, NO_SHAPE);

	switch(VN_CODE(enp)){
#ifdef SUPPORT_MATLAB_MODE
		case T_SUBSCRIPT1:		/* matlab   compile_node */
			new_enp=compile_matlab_subscript(VN_CHILD(enp,0),VN_CHILD(enp,1));
			/* we need to link this node in place of the original */
			SET_VN_PARENT(new_enp, VN_PARENT(enp));
			if( VN_PARENT(enp) != NO_VEXPR_NODE ){
				int i;

				for(i=0;i<MAX_CHILDREN(VN_PARENT(enp));i++)
					if( VN_CHILD(VN_PARENT(enp),i) == enp )
						SET_VN_CHILD(VN_PARENT(enp),i, new_enp);
			}
			/* COMPILE_TREE(new_enp); */
			enp=new_enp;

			break;
#endif /* SUPPORT_MATLAB_MODE */

		ALL_NUMERIC_COMPARISON_CASES		/* compile_node */
			/* These are T_BOOL_EQ etc */
			/* We don't know the shape yet */ 
			/* SET_VN_BM_SHAPE(enp, ALLOC_SHAPE ); */
			/* Call get mating shapes here? */
			break;

		BINARY_BOOLOP_CASES			/* compile_node */
			/* logical AND etc. */
			if( get_mating_shapes(enp,0,1) == NO_SHAPE )
{
WARN("compile_node (binary boolop):  no mating shapes!?");
DUMP_TREE(enp);
}

			if( ! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) )
				check_bitmap_arg(enp,0);
			if( ! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,1)) )
				check_bitmap_arg(enp,1);

			/* SET_VN_BM_SHAPE(enp, ALLOC_SHAPE ); */

			break;

		case T_BOOL_NOT:				/* compile_node */
			if( ! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) )
				check_bitmap_arg(enp,0);
			break;

		case T_UMINUS:		/* compile_node */
			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
	WARN("compile_node:  uminus arg has not shape!?");
				break;		/* an error */
			}

			if( ! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				SET_VN_CODE(enp, T_VS_FUNC);
				SET_VN_VFUNC_CODE(enp, FVSMUL);
				SET_VN_CHILD(enp,1, minus_one_enp);
			} else {
				/* If the child node is a literal, just change the value */
				switch(VN_CODE(VN_CHILD(enp,0))){
					case T_LIT_INT:
						SET_VN_INTVAL( VN_CHILD(enp,0),
							-1 * VN_INTVAL(VN_CHILD(enp,0)) );
						goto excise_uminus;

					case T_LIT_DBL:
						SET_VN_DBLVAL(VN_CHILD(enp,0),
							VN_DBLVAL(VN_CHILD(enp,0)) * -1 );
						goto excise_uminus;

excise_uminus:
						/* We just return the new node & fix links later... */
#ifdef FOOBAR
						/* Now replace the UMINUS node with this one...
						 * The hard part is updating the reference in
						 * the parent...
						 */
						for(i=0;i<MAX_NODE_CHILDREN;i++){
							if( VN_CHILD(VN_PARENT(enp),i) == enp ){
								VN_CHILD(VN_PARENT(enp),i) = VN_CHILD(enp,0);
								SET_VN_PARENT(VN_CHILD(enp,0), VN_PARENT(enp));
								/* Now we can release the UMINUS node */
							/* BUG but we don't seem to have a release func? */
								i=MAX_NODE_CHILDREN; /* will be incremented */
							}
						}
//#ifdef CAUTIOUS
//						/* make sure we replaced the node */
//						if( i == MAX_NODE_CHILDREN )
//					ERROR1("CAUTIOUS:  compile node:  couldn't remove UMINUS node!?");
//#endif /* CAUTIOUS */
						assert( i != MAX_NODE_CHILDREN );
#endif /* FOOBAR */
						enp = VN_CHILD(enp,0);
						break;

					default:
						SET_VN_CODE(enp, T_TIMES);
						SET_VN_CHILD(enp,1, minus_one_enp);
						enp=compile_node(enp);	/* necessary? */
						break;
				}
			}
			break;


		case T_BITCOMP:		/* compile_node */
			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
	WARN("compile_node:  bitcomp arg has not shape!?");
				break;		/* an error */
			}

			if( ! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				SET_VN_CODE(enp, T_VCOMP);
			//VERIFY_DATA_TYPE(enp,ND_FUNC,"compile_node T_BITCOMP")
				ASSERT_NODE_DATA_TYPE( enp, ND_FUNC )
				SET_VN_VFUNC_CODE(enp, FVCOMP);
			}
			break;



		case T_VS_FUNC:
			if( get_mating_shapes(enp,0,1) == NO_SHAPE ) {
				/* We shouldn't have to check for outer product shapes with vector-scalar */
				CURDLE(enp)
				break;
			}
			break;


		ALL_SCALINT_BINOP_CASES
		OTHER_SCALAR_MATHOP_CASES		/* compile_node */
		/* why not integer math? */

			/* first check for a special case: */
			if( VN_CODE(enp) == T_PLUS && VN_CHILD_PREC(enp,0) == PREC_CHAR &&
				VN_CHILD_PREC(enp,1) == PREC_CHAR ){

				SET_VN_CODE(enp, T_STRING_LIST);
				break;
			}

			/* We check the shapes of the children, and use the info
			 * to change the code to a vector-vector or vector-scalar op
			 * if necessary.
			 */

			/* Now get_mating_shapes handles outer shapes too */
			if( get_mating_shapes(enp,0,1) == NO_SHAPE ) {
				CURDLE(enp)
				break;
			}

			/* for the arithmetic opcodes, determine if we
			 * need to change the node code
			 */

			check_arith_code(QSP_ARG  enp);
/*
sprintf(ERROR_STRING,"compile_node binop:  %s",node_desc(enp));
advise(ERROR_STRING);
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

			if( get_mating_shapes(enp,1,2) == NO_SHAPE ){
				return(enp);
			}



			/* If the two possible values are both scalars,
			 * then get_mating shapes will assign a scalar
			 * shape to the T_SS_S_CONDASS node.  If the
			 * test is a vector, then we need to fix this.
			 */

			/* The test node is either a scalar or a bitmap array */

			if( ! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) ){	/* control word is vector */
				if( SCALAR_SHAPE(VN_SHAPE(enp)) ){
					/* both sources (targets?) are scalars, but the bitmap is a vector */
					copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
//fprintf(stderr,"calling xform_from_bitmap #1\n");
//dump_shape(QSP_ARG  VN_SHAPE(enp));
					xform_from_bitmap(VN_SHAPE(enp),VN_CHILD_PREC(enp,1));
					SET_VN_CODE(enp, T_SS_B_CONDASS);
				} else {
					/* The source expression is vector, and so is the bitmap */
					check_mating_bitmap(enp,VN_CHILD(enp,0));
					check_xx_v_condass_code(enp);
				}
				// Not all node types need a bitmap shape??
				if( VN_DATA_TYPE(enp) == ND_BMAP ){
					SET_VN_BM_SHAPE(enp, ALLOC_SHAPE );
				}
			} else {					/* control word is a scalar */
				if( ! SCALAR_SHAPE(VN_SHAPE(enp)) ){
					/* at least one target is a vector */
					check_xx_s_condass_code(enp);
					SET_VN_BM_SHAPE(enp, ALLOC_SHAPE );
				}
				/* else everything is a scalar, code stays the same */
			}

			break;

		case T_MAXVAL:		/* compile_node */
		case T_MINVAL:

			/* max and min can take a list of expressions */
			if( VN_CODE(VN_CHILD(enp,0)) == T_EXPR_LIST ){
				/* 2 or more args,
				 * this next call will definitely change the opcode.
				 *
				 * At least, with the old veclib it did...  What should we
				 * do now that we support outer ops and projection?
				 */
				check_minmax_code(enp);

//#ifdef CAUTIOUS
//				if( VN_CODE(enp) == T_MAXVAL || VN_CODE(enp) == T_MINVAL ){
//					NODE_ERROR(enp);
//					WARN("CAUTIOUS:  check_minmax_code did not change opcode!?");
//					DUMP_TREE(enp);
//				}
//#endif /* CAUTIOUS */
				assert( VN_CODE(enp) != T_MAXVAL && VN_CODE(enp) != T_MINVAL );

			}
			break;

		case T_MATH0_FN:		/* compile_node */
			/* This node has no children, we want to set it's shape from
			 * the LHS, here we assume it's a vector...
			 */
			SET_VN_CODE(enp, T_MATH0_VFN);
			//VERIFY_DATA_TYPE(enp,ND_FUNC,"compile_node T_MATH0_FN")
			ASSERT_NODE_DATA_TYPE( enp, ND_FUNC )
			vf_code = FUNC_VV_CODE( VN_FUNC_PTR(enp) );
//#ifdef CAUTIOUS
//			if( vf_code == INVALID_VFC ){	// should never happen?
//				sprintf(ERROR_STRING,
//		"CAUTIOUS:  compile_node:  Sorry, no vector implementation of math function %s yet",
//			FUNC_NAME(VN_FUNC_PTR(enp)) );
//				WARN(ERROR_STRING);
//				CURDLE(enp)
//				return(enp);
//			}
//#endif // CAUTIOUS
			assert( vf_code != INVALID_VFC );

			/* this overwrites en_func_index, because it is a union! */
			SET_VN_VFUNC_CODE(enp, vf_code);
			break;

		case T_INT1_FN:			/* compile_node */
			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
				WARN("compile_node:  child has no shape");
				break;
			}

			if( (! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) ) &&
				! FLOATING_PREC(SHP_PREC(VN_CHILD_SHAPE(enp,0))) ){
				typecast_child(enp,0,PREC_FOR_CODE(PREC_SP));
			}

			if( IS_VECTOR_NODE(VN_CHILD(enp,0)) ){
				SET_VN_CODE(enp, T_INT1_VFN);
				//VERIFY_DATA_TYPE(enp,ND_FUNC,"compile_node T_INT1_FN")
				ASSERT_NODE_DATA_TYPE( enp, ND_FUNC )
				vf_code = FUNC_VV_CODE( VN_FUNC_PTR(enp) );
//#ifdef CAUTIOUS
//				if( vf_code == INVALID_VFC ){	// should never happen?
//					sprintf(ERROR_STRING,
//		"CAUTIOUS:  compile_node:  Sorry, no vector implementation of math function %s yet",
//			FUNC_NAME(VN_FUNC_PTR(enp)) );
//					WARN(ERROR_STRING);
//					CURDLE(enp)
//					return(enp);
//				}
//#endif // CAUTIOUS
				assert( vf_code != INVALID_VFC );
			/* this overwrites en_func_index, because it is a union! */
				SET_VN_VFUNC_CODE(enp, vf_code);
			}

			break;




		case T_MATH1_FN:		/* compile_node */
			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
				WARN("compile_node:  child has no shape");
				break;
			}

			/* The math functions take double args, so if the child
			 * node is not a scalar, and is not FLOAT_PREC, we have
			 * to cast it.  We cast to float instead of double to
			 * save on memory.
			 *
			 * BUG - there are float versions of the math lib functions now!?
			 */

			if( (! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) ) &&
				! FLOATING_PREC(SHP_PREC(VN_CHILD_SHAPE(enp,0))) ){
				typecast_child(enp,0,PREC_FOR_CODE(PREC_SP));
			}

			/* We should never have to change T_MATH1_VFN
			 * to T_MATH1_FN, because the parser only uses
			 * T_MATH1_FN.
			 */

			if( IS_VECTOR_NODE(VN_CHILD(enp,0)) ){
				SET_VN_CODE(enp, T_MATH1_VFN);
				//VERIFY_DATA_TYPE(enp,ND_FUNC,"compile_node T_MATH1_FN")
				ASSERT_NODE_DATA_TYPE( enp, ND_FUNC )
			//	vf_code = (Vec_Func_Code)
			//math1_functbl[VN_FUNC_PTR(enp)].fn_vv_code;
				vf_code = FUNC_VV_CODE( VN_FUNC_PTR(enp) );
//#ifdef CAUTIOUS
//				if( vf_code == INVALID_VFC ){	// should never happen?
//					sprintf(ERROR_STRING,
//		"CAUTIOUS:  compile_node:  Sorry, no vector implementation of math function %s yet",
//			FUNC_NAME(VN_FUNC_PTR(enp)) );
//					WARN(ERROR_STRING);
//					CURDLE(enp)
//					return(enp);
//				}
//#endif // CAUTIOUS
				assert( vf_code != INVALID_VFC );

			/* this overwrites en_func_index, because it is a union! */
				SET_VN_VFUNC_CODE(enp, vf_code);
			}

			break;

		case T_STRV_FN:			/* compile_node */
		case T_CHAR_FN:			/* compile_node */

			// This code mostly copied from above - good candidate for a macro or
			// subroutine?  BUG?

			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
				WARN("compile_node:  child has no shape");
				break;
			}

			/* The char functions take char or string args, so if the child
			 * node is not a scalar, and is not PREC_BY or PREC_UBY, we have
			 * to cast it.
			 *
			 */

			if( (! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) ) &&
				! CHAR_PREC(SHP_PREC(VN_CHILD_SHAPE(enp,0))) ){
				typecast_child(enp,0,PREC_FOR_CODE(PREC_BY));
			}

			/* We should never have to change T_CHAR_VFN
			 * to T_CHAR_FN, because the parser only uses
			 * T_CHAR_FN.
			 */

			if( IS_VECTOR_NODE(VN_CHILD(enp,0)) ){
				SET_VN_CODE(enp, T_CHAR_VFN);
				//VERIFY_DATA_TYPE(enp,ND_FUNC,"compile_node T_CHAR_FN")
				ASSERT_NODE_DATA_TYPE( enp, ND_FUNC )
				vf_code = FUNC_VV_CODE( VN_FUNC_PTR(enp) );
//#ifdef CAUTIOUS
//				if( vf_code == INVALID_VFC ){	// should never happen?
//					sprintf(ERROR_STRING,
//		"CAUTIOUS:  compile_node:  Sorry, no vector implementation of char function %s yet",
//			FUNC_NAME(VN_FUNC_PTR(enp)) );
//					WARN(ERROR_STRING);
//					CURDLE(enp)
//					return(enp);
//				}
//#endif // CAUTIOUS
				assert( vf_code != INVALID_VFC );
			/* this overwrites en_func_index, because it is a union! */
				SET_VN_VFUNC_CODE(enp, vf_code);
			}

			break;



		case T_MATH2_FN:		/* compile_node */
			//VERIFY_DATA_TYPE(enp,ND_FUNC,"compile_node T_MATH2_FN")
			ASSERT_NODE_DATA_TYPE( enp, ND_FUNC )
			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ||
				VN_CHILD(enp,1) == NO_VEXPR_NODE ){
				/* parse error - should zap this node? */
				CURDLE(enp)
				return(enp);
			}

			/* cast children if necessary (see MATH_VFN above) */
			/* do we really want to cast if unknown??? */

			if( (! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) ) &&
				! FLOATING_PREC(SHP_PREC(VN_CHILD_SHAPE(enp,0))) ){

				typecast_child(enp,0,PREC_FOR_CODE(PREC_SP));
			}

			if( (! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,1)) ) &&
				! FLOATING_PREC(SHP_PREC(VN_CHILD_SHAPE(enp,1))) ){

				typecast_child(enp,1,PREC_FOR_CODE(PREC_SP));
			}

			if( get_mating_shapes(enp,0,1) == NO_SHAPE ){
				CURDLE(enp)
				break;
			}

			/* vector-scalar functions:
			 * If the first arg is the vector, we use the first
			 * version of the function, otherwise the second.
			 */

			if( SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				if( ! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
					Vec_Expr_Node *tmp_enp;

					SET_VN_CODE(enp, T_MATH2_VSFN);

					/* switch order, scalar is 2nd */
					/* BUG this is only good for functions where
					 * the order of the operands doesn't matter,
					 * like max and min.  pow() and atan2()
					 * on the other hand...
					 * But at the moment, their vs versions
					 * have not been written...
					 */
					tmp_enp = VN_CHILD(enp,0);
					SET_VN_CHILD(enp,0, VN_CHILD(enp,1));
					SET_VN_CHILD(enp,1, tmp_enp);
					/* We need to make sure to use the second vsfunc code! */
					vf_code = FUNC_VS_CODE2( VN_FUNC_PTR(enp));
				} else {
					/* both scalars; leave it alone */
					vf_code = (Vec_Func_Code)-2;	/* flag value for later... */
				}
			} else {
				if( SCALAR_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
					SET_VN_CODE(enp, T_MATH2_VSFN);
					vf_code = FUNC_VS_CODE( VN_FUNC_PTR(enp) );
				} else {
					SET_VN_CODE(enp, T_MATH2_VFN);

					vf_code = FUNC_VV_CODE( VN_FUNC_PTR(enp) );
//#ifdef CAUTIOUS
//					if( vf_code == INVALID_VFC ){
//						sprintf(ERROR_STRING,
//		"CAUTIOUS:  compile_node:  Sorry, no vector implementation of math function %s yet",
//				FUNC_NAME( VN_FUNC_PTR(enp)) );
//						WARN(ERROR_STRING);
//						CURDLE(enp)
//						return(enp);
//					}
//#endif // CAUTIOUS
					assert( vf_code != INVALID_VFC );
				}
			}
			/* N_VEC_FUNCS is the initial value... */
			if( vf_code != N_VEC_FUNCS ){
//#ifdef CAUTIOUS
//				if( vf_code == INVALID_VFC ){
//					sprintf(ERROR_STRING,
//		"CAUTIOUS:  compile_node:  Sorry, no vector-scalar implementation of math function %s yet",
//					FUNC_NAME( VN_FUNC_PTR(enp)) );
//					WARN(ERROR_STRING);
//					CURDLE(enp)
//					return(enp);
//				}
//#endif // CAUTIOUS
				assert( vf_code != INVALID_VFC );
				SET_VN_VFUNC_CODE(enp, vf_code);
			}

			break;

		case T_RETURN:				/* compile_node */
			if( curr_srp == NO_SUBRT ){
				NODE_ERROR(enp);
				advise("return statement occurs outside of subroutine");
				CURDLE(enp)
				break;
			}
			if( SR_PREC_CODE(curr_srp) == PREC_VOID ){
				if( VN_CHILD(enp,0) != NO_VEXPR_NODE ){
					NODE_ERROR(enp);
					sprintf(ERROR_STRING,
						"void subroutine %s can't return an expression",
						SR_NAME(curr_srp));
					advise(ERROR_STRING);
					CURDLE(enp)
				}
			} else {
				if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
					NODE_ERROR(enp);
					sprintf(ERROR_STRING,
						"subroutine %s returns without an expression",
						SR_NAME(curr_srp));
					advise(ERROR_STRING);
					CURDLE(enp)
				}
			}
			SET_VN_SUBRT(enp, curr_srp);
			break;

		case T_SIZE_FN:
			/* OLD:
			 *  The integer is the index of the function in the table (see support/function.c)
			 * but we want to use it as a dimension index.  This requires having the function
			 * table in the correct order.  This code here handles that fact that we
			 * have two entries ("depth" and "ncomps") that refer to the same function.
			 *
			 * NEW:
			 * The node has the pointer to the function...
			 */
			//VERIFY_DATA_TYPE(enp,ND_FUNC,"compile_node T_SIZE_FN")
			ASSERT_NODE_DATA_TYPE( enp, ND_FUNC )

#ifdef FOOBAR
			/* old hack to handle depth and ncomps... */
			if( VN_FUNC_PTR(enp) == N_DIMENSIONS )
				SET_VN_FUNC_PTR(enp,0);

//#ifdef CAUTIOUS
//			if( VN_FUNC_IDX(enp) < 0 || VN_FUNC_IDX(enp) >= N_DIMENSIONS ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,
//					"CAUTIOUS:  unexpected size function index %d",
//						FUNC_NAME(VN_FUNC_IDX(enp)));
//				WARN(ERROR_STRING);
//			}
//#endif /* CAUTIOUS */
			assert( VN_FUNC_IDX(enp) >= 0 );
			assert( VN_FUNC_IDX(enp) < N_DIMENSIONS );
#endif /* FOOBAR */
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

//#ifdef CAUTIOUS
//			if( VN_CHILD_SHAPE(enp,1) == NO_SHAPE ){
//				sprintf(ERROR_STRING,"CAUTIOUS:  compile_node %s:  RHS %s has no shape!?",
//					node_desc(enp),node_desc(VN_CHILD(enp,1)));
//				WARN(ERROR_STRING);
//				CURDLE(enp)
//				DUMP_TREE(enp);
//				break;
//			}
//#endif /* CAUTIOUS */
			assert( VN_CHILD_SHAPE(enp,1) != NO_SHAPE );

//fprintf(stderr,"compile_node T_ASSIGN\n");
//DUMP_TREE(enp);
			if( (!SCALAR_SHAPE(VN_CHILD_SHAPE(enp,1))) &&
				/* next line added for matlab */
				VN_CHILD_SHAPE(enp,0) != NO_SHAPE &&
				/* the shape may be unknown, but we still know the precision!? */
				/*
				( ! UNKNOWN_SHAPE( VN_CHILD_SHAPE(enp,0) ) ) &&
				*/
				VN_CHILD_PREC(enp,0) != VN_CHILD_PREC(enp,1) ){
/*
sprintf(ERROR_STRING,"compile_node ASSIGN:  casting %s to precision %s of %s",
node_desc(VN_CHILD(enp,1)),
NAME_FOR_PREC_CODE(VN_CHILD_PREC(enp,0)),
node_desc(VN_CHILD(enp,0)));
advise(ERROR_STRING);
*/
#ifdef QUIP_DEBUG
if( debug & cast_debug ){
DUMP_TREE(enp);
}
#endif /* QUIP_DEBUG */
				typecast_child(enp,1,VN_CHILD_PREC_PTR(enp,0));
			}

			/* Check for vector subscripts on LSH */
			if( has_vector_subscript(VN_CHILD(enp,0)) ){
#ifdef NOT_YET
				enp=fix_render_code(QSP_ARG  enp);
#else /* ! NOT_YET */
				WARN("Sorry, vector subscripts not implemented for objC yet...");
#endif /* NOT_YET */
			}
			break;

		case T_COMP_LIST:		/* compile_node */
		case T_ROW_LIST:		/* compile_node */
			/* Like EXPR_LIST case handled below, but precisions of elements have to match */
			check_typecast(enp,0,1);
			SET_VN_N_ELTS(enp, leaf_count(enp,VN_CODE(enp)) );
			break;

		ALL_LIST_NODE_CASES		/* compile_node */
			/*
			balance_list(enp,VN_CODE(enp));
			*/
			SET_VN_N_ELTS(enp, leaf_count(enp,VN_CODE(enp)) );
			break;


		case T_INNER:			/* compile_node */
			/* If the child nodes are both scalars, change to TIMES? */
			if( (VN_CHILD_SHAPE(enp,0) != NO_SHAPE &&
				SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) ) ||
			    ( VN_CHILD_SHAPE(enp,1) != NO_SHAPE &&
				SCALAR_SHAPE(VN_CHILD_SHAPE(enp,1)) ) ){

				SET_VN_CODE(enp, T_TIMES);
				enp = compile_node(enp);	/* to get the shape set... */

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
		//ALL_STRFUNC_CASES
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

	if( enp != orig_enp ){	/* did we change it?  If so, fix links... */
		if( VN_PARENT((orig_enp)) != NO_VEXPR_NODE ){
			int i;
			i = which_child(orig_enp);
			SET_VN_CHILD(VN_PARENT(orig_enp),i, enp);
		}
		SET_VN_PARENT(enp, VN_PARENT(orig_enp) );

		//*enpp = enp;
	}

	return(enp);

} /* end compile_node() */

// anchor

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

static void _prelim_node_shape(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp;
	Shape_Info *tmp_shpp;
	Subrt *srp;
	Vec_Expr_Node *decl_enp;
	dimension_t i1;
	const char *s;
	long n1,n2,n3;

	if( IS_CURDLED(enp) ) return;

//sprintf(ERROR_STRING,"prelim_node_shape %s BEGIN",node_desc(enp));
//advise(ERROR_STRING);
//dump_tree(enp);

	/* now all the child nodes have been scanned, process this one */

	switch(VN_CODE(enp)){
		/* matlab */
		case T_SSCANF:
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_DP));
			break;

		/* T_ROWIST merged with T_ROW_LIST - ?? */
		case T_ROW:
		/* case T_ROWLIST: */					/* prelim_node_shape */
			update_node_shape(enp);
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
			copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
//fprintf(stderr,"calling xform_from_bitmap #2\n");
			xform_from_bitmap(VN_SHAPE(enp),VN_CHILD_PREC(enp,1));
			CHECK_UK_CHILD(enp,0);
			break;

		case T_VS_B_CONDASS:				/* prelim_node_shape */
			/* the shape is getting set in compile_node??? */
			/* BUT what about if some of the child shapes are unknown? */
			/* Why is this FOOBAR'd ??? */
#ifdef CONDASS_FOOBAR
			/* copy the shape from the first (vector) node */
			copy_node_shape(enp,VN_CHILD_SHAPE(enp,1));
			CHECK_UK_CHILD(enp,0);
			CHECK_UK_CHILD(enp,1);
			/* If the shape is unknown, but we have the bitmap, then copy from it */
			if( UNKNOWN_SHAPE(VN_SHAPE(enp)) && ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
//fprintf(stderr,"calling xform_from_bitmap #3\n");
				xform_from_bitmap(VN_SHAPE(enp),VN_CHILD_PREC(enp,1));
			}
			/* might be an outer... */
//sprintf(ERROR_STRING,"prelim_node_shape %s",node_desc(enp));
//advise(ERROR_STRING);
//DUMP_TREE(enp);

//sprintf(ERROR_STRING,"prelim_node_shape %s DONE",node_desc(enp));
//advise(ERROR_STRING);
//DUMP_TREE(enp);

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

			if( UNKNOWN_SHAPE(VN_SHAPE(enp)) && ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
//fprintf(stderr,"calling xform_from_bitmap #4\n");
				xform_from_bitmap(VN_SHAPE(enp), VN_CHILD_PREC(enp,1));
			}

			break;

		case T_VV_VV_CONDASS:				/* prelim_node_shape */
			/* shape should already be assigned by get_mating_shapes */
			CHECK_UK_CHILD(enp,0);
			CHECK_UK_CHILD(enp,1);
			CHECK_UK_CHILD(enp,2);
			CHECK_UK_CHILD(enp,3);

			if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
				/* BUG we copy the shape from one of the vector condition
				 * args, but we really should find a mating shape for the pair.
				 * E.g., this will be wrong if we have row < col
				 */
				if( ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
					copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
				} else if( ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
					copy_node_shape(enp,VN_CHILD_SHAPE(enp,1));
				}
			}

			break;

		case T_VS_VV_CONDASS:				/* prelim_node_shape */
			/* shape should already be assigned by get_mating_shapes */
			CHECK_UK_CHILD(enp,0);
			CHECK_UK_CHILD(enp,1);
			CHECK_UK_CHILD(enp,2);

			if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
				/* BUG we copy the shape from one of the vector condition
				 * args, but we really should find a mating shape for the pair.
				 * E.g., this will be wrong if we have row < col
				 */
				if( ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
					copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
				} else if( ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
					copy_node_shape(enp,VN_CHILD_SHAPE(enp,1));
				}
			}

			break;

		case T_VV_VS_CONDASS:				/* prelim_node_shape */
			/* shape should already be assigned by get_mating_shapes */
			CHECK_UK_CHILD(enp,0);
			CHECK_UK_CHILD(enp,2);
			CHECK_UK_CHILD(enp,3);

			if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
				if( ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
					copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
				}
			}

			break;

		case T_VS_VS_CONDASS:				/* prelim_node_shape */
			/* shape should already be assigned by get_mating_shapes */
			CHECK_UK_CHILD(enp,0);
			CHECK_UK_CHILD(enp,2);

			if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
				if( ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
					copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
				}
			}

			break;

		case T_UNDEF:
			POINT_NODE_SHAPE(enp,uk_shape(PREC_SP));
			break;

		case T_MAX_TIMES:			/* prelim_node_shape */
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_DI));
			break;

		case T_REDUCE:
		case T_ENLARGE:		/* prelim_node_shape */
#ifdef FOOBAR
			// This assumes an enlargement factor of 2!?
			// NOT correct!?
			copy_node_shape(enp, VN_CHILD_SHAPE(enp,0) );
			SET_SHP_COLS(VN_SHAPE(enp),
				SHP_COLS(VN_SHAPE(enp)) * 2);
			SET_SHP_ROWS(VN_SHAPE(enp),
				SHP_ROWS(VN_SHAPE(enp)) *2 );
			CHECK_UK_CHILD(enp,0);
#endif // FOOBAR
			// We need to determine the enlargement factor
			// during resolution...
			/*POINT_NODE_SHAPE*/ copy_node_shape(enp,
				uk_shape(PREC_CODE(VN_PREC_PTR(VN_CHILD(enp,0)))));
			break;

#ifdef FOOBAR
		case T_REDUCE:
			copy_node_shape(enp, VN_CHILD_SHAPE(enp,0) );
			SET_SHP_COLS(VN_SHAPE(enp),
				SHP_COLS(VN_SHAPE(enp)) / 2 );
			SET_SHP_ROWS(VN_SHAPE(enp),
				SHP_ROWS(VN_SHAPE(enp)) / 2 );
			CHECK_UK_CHILD(enp,0);
			break;
#endif //FOOBAR

		case T_COMP_OBJ:		/* prelim_node_shape */
			copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
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
			if( VN_CODE(VN_CHILD(enp,0)) == T_ROW_LIST ){
				SET_SHP_TYPE_DIM(VN_SHAPE(enp),0, SHP_TYPE_DIM(VN_SHAPE(enp),1) );
				SET_SHP_TYPE_DIM(VN_SHAPE(enp),1, SHP_TYPE_DIM(VN_SHAPE(enp),2) );
				SET_SHP_TYPE_DIM(VN_SHAPE(enp),2, SHP_TYPE_DIM(VN_SHAPE(enp),3) );
				SET_SHP_TYPE_DIM(VN_SHAPE(enp),3, SHP_TYPE_DIM(VN_SHAPE(enp),4) );
				SET_SHP_TYPE_DIM(VN_SHAPE(enp),4, 1);
				auto_shape_flags(VN_SHAPE(enp),NO_OBJ);
			}
			/* probably a COMP_LIST node?  do nothing? */
			break;

		case T_LIST_OBJ:		/* prelim_node_shape */
			/* all of the elements of the list
			 * should have the same shape...
			 * BUG?  should we verify?
			 */
			copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
			/* We used to increment maxdim here, but the EXPR_LIST child
			 * should already have the proper value...
			 */
			/* BUT - if the list has only one element, we still need to do this! */
			if( VN_CODE(VN_CHILD(enp,0)) != T_ROW_LIST ){
				SET_SHP_MAXDIM(VN_SHAPE(enp),
					SHP_MAXDIM(VN_SHAPE(enp))+1);
				SET_SHP_RANGE_MAXDIM(VN_SHAPE(enp),
					SHP_MAXDIM(VN_SHAPE(enp)));
			}
			break;

		case T_TYPECAST:				/* prelim_node_shape */
			INIT_SHAPE_PTR(tmp_shpp)
			COPY_SHAPE(tmp_shpp, VN_CHILD_SHAPE(enp,0) );
			SET_SHP_PREC_PTR(tmp_shpp, VN_CAST_PREC_PTR(enp));

			// Not sure why this was here...
			// BUG constant or not???
			//SET_SHP_PREC_PTR(tmp_shpp, const_precision(SHP_PREC_PTR(tmp_shpp)) );/* might be const? */
			if( COMPLEX_PRECISION(SHP_PREC(tmp_shpp)) && !COMPLEX_PRECISION(VN_CHILD_PREC(enp,0)) ){
//#ifdef CAUTIOUS
//				if( SHP_TYPE_DIM(tmp_shpp,0) > 1 ){
//					DUMP_TREE(enp);
//					NODE_ERROR(enp);
//					sprintf(ERROR_STRING,
//			"CAUTIOUS:  prelim_node_shape %s (tdim=%d):  casting to complex, but tdim > 1",
//						node_desc(enp),SHP_TYPE_DIM(tmp_shpp,0));
//					ERROR1(ERROR_STRING);
//				}
//#endif /* CAUTIOUS */
				// perhaps should be straight equality test?
				assert( SHP_TYPE_DIM(tmp_shpp,0) <= 1 );

				SET_SHP_TYPE_DIM(tmp_shpp,0, 1);
				SET_SHP_MACH_DIM(tmp_shpp,0, 2);
				/* BUG?  should we recalculate n_type_elts and n_mach_elts? */
				SET_SHP_FLAG_BITS(tmp_shpp, DT_COMPLEX);
			}
			copy_node_shape(enp,tmp_shpp);
			if( UNKNOWN_SHAPE(tmp_shpp) )
				LINK_UK_NODES(enp,VN_CHILD(enp,0));

			RELEASE_SHAPE_PTR(tmp_shpp)
			break;

		case T_EQUIVALENCE:		/* prelim_node_shape */
			copy_node_shape(enp,scalar_shape(VN_DECL_PREC_CODE(enp)));
			update_node_shape(enp);
			break;


		/* case T_SUBSCRIPT1: */	/* prelim_node_shape (matlab) */
		case T_CURLY_SUBSCR:			/* prelim_node_shape */
		case T_SQUARE_SUBSCR:
//#ifdef CAUTIOUS
//			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
//		WARN("CAUTIOUS:  prelim_node_shape:  null object subscripted!?");
//				return;
//			}
			assert( VN_CHILD(enp,0) != NO_VEXPR_NODE );

//			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
//		WARN("CAUTIOUS:  prelim_node_shape:  subscript, no child shape!?");
//DUMP_NODE(enp);
//				discard_node_shape(enp);
//				return;
//			}
//#endif /* CAUTIOUS */
			assert( VN_CHILD_SHAPE(enp,0) != NO_SHAPE );

			/* Now compute the shape of the subscripted object */
			copy_node_shape(enp,uk_shape(SHP_PREC(VN_CHILD_SHAPE(enp,0))));
			/* If the subscripted object is unknown,
			 * link these...
			 */
			if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				LINK_UK_NODES(enp,VN_CHILD(enp,0));
			}

			update_node_shape(enp);
			break;

		case T_SUBVEC:				/* prelim_node_shape */
		case T_CSUBVEC:
//#ifdef CAUTIOUS
//			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
//		WARN("CAUTIOUS:  prelim_node_shape:  null object subscripted!?");
//				return;
//			}
			assert( VN_CHILD(enp,0) != NO_VEXPR_NODE );

//			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
//		WARN("CAUTIOUS:  prelim_node_shape:  subvec, no child shape!?");
//DUMP_NODE(enp);
//				discard_node_shape(enp);
//				return;
//			}
//#endif /* CAUTIOUS */
			assert( VN_CHILD_SHAPE(enp,0) != NO_SHAPE );

			/* Now compute the shape of the subscripted object */
			/* The child node chould have a known shape (scalar), but an
			 * unknown value!?  that would allow us to compute the shape of
			 * a subscripted object but not a subvector...
			 * We must assign the shape to unkown before calling update node shape.
			 */
			copy_node_shape(enp,uk_shape(SHP_PREC(VN_CHILD_SHAPE(enp,0))));
			/* For prelim node shape, we don't evaluate range expressions,
			 * because they may have different values at run-time.
			 * But we need to have a run-time resolution!
			 */
#ifdef FOOBAR
					// deleted a brace in this comment
			if( (! is_variable(VN_CHILD(enp,1))) &&
					(!is_variable(VN_CHILD(enp,2))) )
#endif /* FOOBAR */
			if( HAS_CONSTANT_VALUE(VN_CHILD(enp,1)) && HAS_CONSTANT_VALUE(VN_CHILD(enp,2)) ){
				update_node_shape(enp);
			}
			break;


		case T_SUBSAMP:				/* prelim_node_shape */
		case T_CSUBSAMP:
//#ifdef CAUTIOUS
//			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
//		WARN("CAUTIOUS:  prelim_node_shape:  null object subscripted!?");
//				return;
//			}
			assert( VN_CHILD(enp,0) != NO_VEXPR_NODE );

//			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
//		WARN("CAUTIOUS:  prelim_node_shape:  subsamp, no child shape!?");
//DUMP_NODE(enp);
//				discard_node_shape(enp);
//				return;
//			}
//#endif /* CAUTIOUS */
			assert( VN_CHILD_SHAPE(enp,0) != NO_SHAPE );

			/* Now compute the shape of the subscripted object */
			/* The child node chould have a known shape (scalar), but an
			 * unknown value!?  that would allow us to compute the shape of
			 * a subscripted object but not a subvector...
			 * We must assign the shape to unkown before calling update node shape.
			 */
			copy_node_shape(enp,uk_shape(SHP_PREC(VN_CHILD_SHAPE(enp,0))));
#ifdef FOOBAR
			// deleted brace in comment
			if( (! is_variable(VN_CHILD(enp,1)))  )
#endif /* FOOBAR */
			if( HAS_CONSTANT_VALUE( VN_CHILD(enp,1) ) ){
				update_node_shape(enp);
			}
			break;



		case T_REAL_PART:
		case T_IMAG_PART:
			INIT_SHAPE_PTR(tmp_shpp)
			// COPY_SHAPE copies everything - is that what we want?
			COPY_SHAPE(tmp_shpp, VN_CHILD_SHAPE(enp,0) );
			SET_SHP_MACH_DIM(tmp_shpp,0,1);
			SET_SHP_N_MACH_ELTS(tmp_shpp,
				SHP_N_MACH_ELTS(tmp_shpp) / 2 );
			CLEAR_SHP_FLAG_BITS(tmp_shpp,DT_COMPLEX);
			SET_SHP_PREC_PTR(tmp_shpp, PREC_MACH_PREC_PTR(SHP_PREC_PTR(tmp_shpp)) );

			copy_node_shape(enp,tmp_shpp);
			RELEASE_SHAPE_PTR(tmp_shpp)
			/* BUG verify shpp is complex */
			break;

		case T_TRANSPOSE:
			INIT_SHAPE_PTR(tmp_shpp)
			// COPY_SHAPE copies everything - is that what we want?
			COPY_SHAPE(tmp_shpp, VN_CHILD_SHAPE(enp,0) );
			i1                    = SHP_ROWS(tmp_shpp);
			SET_SHP_ROWS(tmp_shpp, SHP_COLS(tmp_shpp) );
			SET_SHP_COLS(tmp_shpp, i1);
			auto_shape_flags(tmp_shpp,NO_OBJ);

			copy_node_shape(enp,tmp_shpp);
			CHECK_UK_CHILD(enp,0);
			RELEASE_SHAPE_PTR(tmp_shpp)

			break;

		case T_DFT:			/* prelim_node_shape */
			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE )
				return;		/* an error */

			/* This is the complex DFT */
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
			CHECK_UK_CHILD(enp,0);
			break;

		case T_RDFT:		/* prelim_node_shape */
			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE )
				return;		/* an error */

			/* BUG should verify that arg is real and power of 2 */
//advise("prelim_node_shape T_RDFT");

			INIT_SHAPE_PTR(tmp_shpp)
			// COPY_SHAPE copies everything - is that what we want?
			COPY_SHAPE(tmp_shpp, VN_CHILD_SHAPE(enp,0) );
			SET_SHP_PREC_PTR(tmp_shpp, complex_precision(SHP_PREC_PTR(tmp_shpp)) );
			SET_SHP_FLAG_BITS(tmp_shpp, DT_COMPLEX);
			if( !UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				SET_SHP_COLS(tmp_shpp,
					SHP_COLS(tmp_shpp) / 2);
				SET_SHP_COLS(tmp_shpp,
					SHP_COLS(tmp_shpp) + 1);
				/* BUG?  set mach_dim in addition to type_dim? */
				SET_SHP_MACH_DIM(tmp_shpp,0, 2);
				/* BUG should we recalculate number of elements? */
			}
			copy_node_shape(enp,tmp_shpp);
			RELEASE_SHAPE_PTR(tmp_shpp)
			CHECK_UK_CHILD(enp,0);


			break;

		case T_RIDFT:		/* prelim_node_shape */
			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE )
				return;		/* an error */

			/* BUG should verify that arg is real and power of 2 */
//advise("prelim_node_shape T_RIDFT");


			INIT_SHAPE_PTR(tmp_shpp)
			// COPY_SHAPE copies everything - is that what we want?
			COPY_SHAPE(tmp_shpp, VN_CHILD_SHAPE(enp,0) );
			SET_SHP_PREC_PTR(tmp_shpp, PREC_MACH_PREC_PTR(SHP_PREC_PTR(tmp_shpp)) );
			CLEAR_SHP_FLAG_BITS(tmp_shpp,DT_COMPLEX);
			if( !UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				SET_SHP_COLS(tmp_shpp,
					SHP_COLS(tmp_shpp) - 1);
				SET_SHP_COLS(tmp_shpp,
					SHP_COLS(tmp_shpp) * 2 );
				SET_SHP_MACH_DIM(tmp_shpp,0, 1);
				/* BUG set more things?  n_elts?  mach_dim? */
				auto_shape_flags(tmp_shpp,NO_OBJ);
			}
			copy_node_shape(enp,tmp_shpp);
			RELEASE_SHAPE_PTR(tmp_shpp)
			CHECK_UK_CHILD(enp,0);

			break;

		case T_CONJ:
		/* case T_WRAP: case T_SCROLL: case T_DILATE: case T_ERODE: case T_FILL: */
		ALL_UNARY_CASES				/* prelim_node_shape */

//#ifdef CAUTIOUS
//			if( insure_child_shape(enp,0) < 0 ){
//WARN("CAUTIOUS:  prelim_node_shape:  unary op child has no shape");
//				return;		/* an error */
//			}
//#endif /* CAUTIOUS */
			assert( VN_CHILD_SHAPE(enp,0) != NO_SHAPE );

			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
			CHECK_UK_CHILD(enp,0);

			break;


		case T_MAXVAL:
		case T_MINVAL:
		case T_SUM:
			copy_node_shape(enp, uk_shape(SHP_PREC(VN_CHILD_SHAPE(enp,0))) );

			/*
			POINT_NODE_SHAPE(enp,scalar_shape(SHP_PREC(VN_CHILD_SHAPE(enp,0))));
			*/
			/* Before we introduced generalized projection, sum was a scalar,
			 * but now it takes its shape from the LHS!?
			 */
			break;

		case T_INNER:				/* prelim_node_shape */
//#ifdef CAUTIOUS
//			insure_child_shape(enp,0);
//			insure_child_shape(enp,1);
//#endif /* CAUTIOUS */
			assert( VN_CHILD_SHAPE(enp,0) != NO_SHAPE );
			assert( VN_CHILD_SHAPE(enp,1) != NO_SHAPE );

			/* inner product */
			/*
			 * This can be a row dotted with a column, or a matrix
			 * times a column, or a pair of scalars...
			 */

			if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) || UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
				copy_node_shape(enp, uk_shape(SHP_PREC(VN_CHILD_SHAPE(enp,0))) );
				CHECK_UK_CHILD(enp,0);
				CHECK_UK_CHILD(enp,1);
				return;
			}

			/* Use copy_node_shape just to do the mem alloc */
			INIT_SHAPE_PTR(tmp_shpp)
			// COPY_SHAPE copies everything - is that what we want?
			COPY_SHAPE(tmp_shpp, VN_CHILD_SHAPE(enp,0) );
			SET_SHP_COLS(tmp_shpp,SHP_COLS(VN_CHILD_SHAPE(enp,1)) );
			auto_shape_flags(tmp_shpp,NO_OBJ);
			copy_node_shape(enp,tmp_shpp);
			RELEASE_SHAPE_PTR(tmp_shpp)
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

		case T_STRV_FN:
			POINT_NODE_SHAPE(enp, VN_CHILD_SHAPE(enp,0) );
			break;

		case T_CHAR_FN:
		case T_CHAR_VFN:
			// BUG output is a bitmap, not a string!?
			POINT_NODE_SHAPE(enp, VN_CHILD_SHAPE(enp,0) );
			break;


		case T_FILETYPE:	/* could check for valid type here... */
		case T_SAVE:
		case T_INFO:
			/* no-op */
			break;

		case T_LOAD:		/* prelim_node_shape */
			/* We must assign the shape to unkown before calling update node shape.  */
			POINT_NODE_SHAPE(enp,uk_shape(PREC_SP));
			update_node_shape(enp);
			break;

		case T_RETURN:		/* prelim_node_shape */

			/* why do we neet to remember the return nodes? */

			remember_node( & SR_RET_LIST(curr_srp),enp);

			/* This is our chance to figure out the size
			 * of a subroutine...
			 * at least if we are returning an object of known
			 * size...
			 */

			/* Hopeuflly we checked in vectree.y that all returns have
			 * the same type and match the subrt decl
			 */

			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){ /* a void subrt */
				/* leave the shape pointer null */
//#ifdef CAUTIOUS
//				if( VN_SHAPE(enp) != NO_SHAPE ){
//					sprintf(ERROR_STRING,
//				"CAUTIOUS:  prelim_node_shape:  %s has a shape!?",node_desc(enp));
//					WARN(ERROR_STRING);
//					VN_SHAPE(enp) = NO_SHAPE;
//				}
//#endif /* CAUTIOUS */
				assert( VN_SHAPE(enp) == NO_SHAPE );

				return;

			} else if( IS_CURDLED(VN_CHILD(enp,0)) ){
				/* an error expr */
				WARN("return expression is curdled!?");
				discard_node_shape(enp);
				return;
			} else {
				POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
//#ifdef CAUTIOUS
//if( VN_SHAPE(enp) == NO_SHAPE ){
//sprintf(ERROR_STRING,"CAUTIOUS:  prelim_node_shape:  return expression has no shape!?");
//WARN(ERROR_STRING);
//return;
//}
//#endif /* CAUTIOUS */
				assert( VN_SHAPE(enp) != NO_SHAPE );
			}

			/* Here we need to see if there are other
			 * returns, and if so we need to make sure
			 * all the returns match in size...
			 * We also need to communicate this info
			 * to the main subroutine body...
			 */

			srp = curr_srp;

			if( VN_SHAPE(enp) != NO_SHAPE && ! UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
				if( UNKNOWN_SHAPE(SR_SHAPE(srp)) ){
					SET_SR_SHAPE(srp, VN_SHAPE(enp) );
				} else if( !shapes_match(SR_SHAPE(srp), VN_SHAPE(enp)) ){
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

			srp=VN_SUBRT(enp);
			SET_SR_CALL_VN(srp, enp);

			/* We probably need a separate list! */
			/* link any unknown shape args to the callfunc node */
			if( ! NULL_CHILD(enp,0) )
				link_uk_args(QSP_ARG  enp,VN_CHILD(enp,0));

			/* void subrt's never need to have a shape */
			if( SR_PREC_CODE(srp) == PREC_VOID ){
				SET_VN_SHAPE(enp, NO_SHAPE);
				return;
			}

			/* we don't need to remember void subrts... */
			if( curr_srp != NO_SUBRT	/* WHY??? */
				&& enp != SR_BODY(curr_srp) &&
				VN_CODE(VN_PARENT(enp)) != T_STAT_LIST ){

				/* Why are we doing this?? */
				remember_callfunc_node(curr_srp,enp);
			}


			/* make sure the number of aruments is correct */
			if( arg_count(VN_CHILD(enp,0)) != SR_N_ARGS(srp) ){
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,
	"Subrt %s expects %d arguments (%d passed)",SR_NAME(srp), SR_N_ARGS(srp),
					arg_count(VN_CHILD(enp,0)));
				WARN(ERROR_STRING);
				CURDLE(enp)
				return;
			}



			copy_node_shape(enp,SR_SHAPE(srp));
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
//#ifdef CAUTIOUS
//			if( insure_child_shape(enp,0) < 0 ||
//				insure_child_shape(enp,1) ) return;
//#endif /* CAUTIOUS */
			assert( VN_CHILD_SHAPE(enp,0) != NO_SHAPE );
			assert( VN_CHILD_SHAPE(enp,1) != NO_SHAPE );

			/* top node of a list has a literal on the right side,
			 * and the lists descending on the left.
			 * BUG will this code work if the tree is balanced?
			 */

			/* if either child node has unknown shape, then we might as well give up... */
			if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
				copy_node_shape(enp,VN_CHILD_SHAPE(enp,1));
				break;
			}
			if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
				break;
			}

			/* Here we arbitrarily take the shape from the right-child;
			 * But what if the precisions differ?  This can arise if we
			 * omit the decimal point from float values with no decimal
			 * fraction, they are stored as literal long values...
			 */
			/*
			if( VN_CHILD_PREC(enp,0) != VN_CHILD_PREC(enp,1) )
				promote_child(enp);
			*/

			INIT_SHAPE_PTR(tmp_shpp)
			// COPY_SHAPE copies everything - is that what we want?
			COPY_SHAPE(tmp_shpp, VN_CHILD_SHAPE(enp,1) );

			SET_SHP_TYPE_DIM(tmp_shpp,SHP_MAXDIM(tmp_shpp)+1,
				SHP_TYPE_DIM(VN_CHILD_SHAPE(enp,0),SHP_MAXDIM(tmp_shpp)+1) +1 );

			// we need to set mach_dim also, because it
			// is used to set minmaxdim...
			SET_SHP_MACH_DIM(tmp_shpp,SHP_MAXDIM(tmp_shpp)+1,
				SHP_MACH_DIM(VN_CHILD_SHAPE(enp,0),SHP_MAXDIM(tmp_shpp)+1) +1 );

			auto_shape_flags(tmp_shpp,NO_OBJ);	/* auto_shape_flags sets maxdim?? */
			MULTIPLY_DIMENSIONS( SHP_N_TYPE_ELTS(tmp_shpp),SHP_TYPE_DIMS(tmp_shpp) )
			/* BUG?  n_mach_elts? */
			copy_node_shape(enp,tmp_shpp);
			RELEASE_SHAPE_PTR(tmp_shpp)

			break;

		case T_COMP_LIST:				/* prelim_node_shape */
			/* arises as a child node of T_LIST_OBJ (declaration)
			 * The latter case is why we need shape info...
			 */
//#ifdef CAUTIOUS
//			if( insure_child_shape(enp,0) < 0 ||
//				insure_child_shape(enp,1) ) return;
//#endif /* CAUTIOUS */
			assert( VN_CHILD_SHAPE(enp,0) != NO_SHAPE );
			assert( VN_CHILD_SHAPE(enp,1) != NO_SHAPE );

			/* top node of a list has a literal on the right side,
			 * and the lists descending on the left.
			 * BUG will this code work if the tree is balanced?
			 */

			/* if either child node has unknown shape, then we might as well give up... */
			if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
				copy_node_shape(enp,VN_CHILD_SHAPE(enp,1));
				break;
			}
			if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
				break;
			}

			/* Here we arbitrarily take the shape from the right-child;
			 * But what if the precisions differ?  This can arise if we
			 * omit the decimal point from float values with no decimal
			 * fraction, they are stored as literal long values...
			 */
			/*
			if( VN_CHILD_PREC(enp,0) != VN_CHILD_PREC(enp,1) )
				promote_child(enp);
			*/

			INIT_SHAPE_PTR(tmp_shpp)
			// COPY_SHAPE copies everything - is that what we want?
			COPY_SHAPE(tmp_shpp, VN_CHILD_SHAPE(enp,1) );


			if( SHP_MINDIM(tmp_shpp) == (N_DIMENSIONS-1) ){
				SET_SHP_MINDIM(tmp_shpp, 0);
				SET_SHP_MAXDIM(tmp_shpp, 0);
				SET_SHP_RANGE_MINDIM(tmp_shpp, 0);
				SET_SHP_RANGE_MAXDIM(tmp_shpp, 0);
			}

			/* BUG?  type_dim or mach_dim? */
			SET_SHP_TYPE_DIM(tmp_shpp,SHP_MINDIM(tmp_shpp),
				SHP_TYPE_DIM(VN_CHILD_SHAPE(enp,0),SHP_MINDIM(tmp_shpp)) +1 );

			auto_shape_flags(tmp_shpp,NO_OBJ);	/* auto_shape_flags sets maxdim?? */
			MULTIPLY_DIMENSIONS( SHP_N_TYPE_ELTS(tmp_shpp),SHP_TYPE_DIMS(tmp_shpp) )
			/* BUG?  n_mach_elts? */
			copy_node_shape(enp,tmp_shpp);
			RELEASE_SHAPE_PTR(tmp_shpp)
			break;


		/* Dereference nodes can't be assigned a shape until runtime,
		 * but we can still point to the child shape.
		 * We also set up a link for resolution.
		 */
		case T_DEREFERENCE:	/* prelim_node_shape */
//#ifdef CAUTIOUS
//			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,"dereference child %s node n%d has no shape!?",
//					NNAME(VN_CHILD(enp,0)),VN_SERIAL(VN_CHILD(enp,0)));
//				WARN(ERROR_STRING);
//			}
//#endif /* CAUTIOUS */
			assert( VN_CHILD_SHAPE(enp,0) != NO_SHAPE );
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
NODE_ERROR(enp);
sprintf(ERROR_STRING,"prelim_node_shape calling link_uk_nodes for %s and child %s",
node_desc(enp),node_desc(VN_CHILD(enp,0)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			LINK_UK_NODES(enp,VN_CHILD(enp,0));
			break;

		case T_STRING:			/* prelim_node_shape */
			copy_node_shape(enp,scalar_shape(PREC_CHAR));
			SET_SHP_TYPE_DIM(VN_SHAPE(enp),1, (dimension_t) strlen(VN_STRING(enp))+1 );
			SET_SHP_N_TYPE_ELTS(VN_SHAPE(enp), (dimension_t) strlen(VN_STRING(enp))+1 );
			auto_shape_flags(VN_SHAPE(enp),NO_OBJ);
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
			if( HAS_CONSTANT_VALUE(VN_CHILD(enp,0)) )
				SET_VN_FLAG_BITS(enp, NODE_HAS_CONST_VALUE);

			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE )
				return;		/* an error */
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
			CHECK_UK_CHILD(enp,0);

			break;

		case T_RECIP:
			if( HAS_CONSTANT_VALUE(VN_CHILD(enp,0)) )
				SET_VN_FLAG_BITS(enp, NODE_HAS_CONST_VALUE);

			if( VN_CHILD_SHAPE(enp,0) == NO_SHAPE )
				return;		/* an error */

//#ifdef CAUTIOUS
//			if( ! SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
//		WARN("CAUTIOUS:  T_RECIP arg should have scalar shape!?");
//				return;
//			}
//#endif /* CAUTIOUS */
			assert( SCALAR_SHAPE(VN_CHILD_SHAPE(enp,0)) );

			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
			CHECK_UK_CHILD(enp,0);

			break;

		case T_BOOL_NOT:				/* prelim_node_shape */
			if( HAS_CONSTANT_VALUE(VN_CHILD(enp,0)) )
				SET_VN_FLAG_BITS(enp, NODE_HAS_CONST_VALUE);

			copy_node_shape(enp,VN_CHILD_SHAPE(enp,0));
			CHECK_UK_CHILD(enp,0);
			break;

		case T_BITCOMP:
		case T_VCOMP:
			if( HAS_CONSTANT_VALUE(VN_CHILD(enp,0)) )
				SET_VN_FLAG_BITS(enp, NODE_HAS_CONST_VALUE);

			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
			if( VN_SHAPE(enp) != NO_SHAPE )
				CHECK_UK_CHILD(enp,0);
			break;

		case T_MATH0_VFN:	/* prelim_node_shape */
		case T_MATH0_FN:
			update_node_shape(enp);
			break;

		case T_MATH1_VFN:	/* prelim_node_shape */
		case T_MATH1_FN:
		case T_INT1_VFN:	/* prelim_node_shape */
		case T_INT1_FN:
			if( HAS_CONSTANT_VALUE(VN_CHILD(enp,0)) )
				SET_VN_FLAG_BITS(enp, NODE_HAS_CONST_VALUE);

			update_node_shape(enp);
			CHECK_UK_CHILD(enp,0);
			break;

		case T_MATH2_FN:			/* prelim_node_shape */
		case T_MATH2_VFN:
		case T_MATH2_VSFN:
			if( HAS_CONSTANT_VALUE(VN_CHILD(enp,0)) &&
				HAS_CONSTANT_VALUE(VN_CHILD(enp,1)) )
				SET_VN_FLAG_BITS(enp, NODE_HAS_CONST_VALUE);

/*
sprintf(ERROR_STRING,"prelim_node_shape %s: begin ",node_desc(enp));
advise(ERROR_STRING);
DUMP_TREE(enp);
*/
			update_node_shape(enp);
/*
sprintf(ERROR_STRING,"prelim_node_shape %s: after update_node_shape ",node_desc(enp));
advise(ERROR_STRING);
DUMP_TREE(enp);
*/
			check_binop_links(enp);
/*
sprintf(ERROR_STRING,"prelim_node_shape %s: done ",node_desc(enp));
advise(ERROR_STRING);
DUMP_TREE(enp);
*/
			break;



		case T_ASSIGN:			/* prelim_node_shape */

			/* The case for T_ASSIGN is much like that for
			 * the arithmetic operands below...
			 */
			if( IS_CURDLED(VN_CHILD(enp,0)) ||
				IS_CURDLED(VN_CHILD(enp,1)) ){
fprintf(stderr,"prelim_node_shape T_ASSIGN:  curdled child!?\n");
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

			count_lhs_refs(enp);

			break;

		case T_REFERENCE:	/* prelim_node_shape */
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
			LINK_UK_NODES(enp,VN_CHILD(enp,0));
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
			if( HAS_CONSTANT_VALUE(VN_CHILD(enp,0)) &&
				HAS_CONSTANT_VALUE(VN_CHILD(enp,1)) )
				SET_VN_FLAG_BITS(enp, NODE_HAS_CONST_VALUE);

			if( get_mating_shapes(enp,0,1) == NO_SHAPE )
{
WARN("prelim_node_shape (boolop):  no mating shapes!?");
DUMP_TREE(enp);
				return;
}
			if( ! SCALAR_SHAPE(VN_SHAPE(enp)) )
				set_bool_vecop_code(QSP_ARG  enp);
			check_binop_links(enp);
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
//sprintf(ERROR_STRING,"prelim_node_shape %s:  binop",node_desc(enp));
//advise(ERROR_STRING);
//dump_tree(enp);
			if( HAS_CONSTANT_VALUE(VN_CHILD(enp,0)) &&
				HAS_CONSTANT_VALUE(VN_CHILD(enp,1)) )
				SET_VN_FLAG_BITS(enp, NODE_HAS_CONST_VALUE);
#ifdef FOOBAR
			/* when are these nodes supposed to get their shapes set??? */
//#ifdef CAUTIOUS
//			if( VN_SHAPE(enp) == NO_SHAPE ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,"CAUTIOUS:  prelim_node_shape %s:  null shape ptr!?",node_desc(enp));
//				ERROR1(ERROR_STRING);
//			}
//#endif /* CAUTIOUS */
			assert( VN_SHAPE(enp) != NO_SHAPE );

#endif /* FOOBAR */
			/* shape pointer is generally uninitialized here??? */

			/* used to be a nop for vv_func in update_node_shape... */
			update_node_shape(enp);
			check_binop_links(enp);

			break;

		case T_OBJ_LOOKUP:	/* prelim_node_shape */
			if( ! executing ){
				POINT_NODE_SHAPE(enp,uk_shape(PREC_SP));
				break;
			}

			s=EVAL_STRING(VN_CHILD(enp,0));
			if( s == NULL )
				dp=NO_OBJ;
			else
				dp=DOBJ_OF(s);

//#ifdef CAUTIOUS
//			if( dp==NO_OBJ ){
//				if( s != NULL ){
//					sprintf(ERROR_STRING,
//		"CAUTIOUS:  prelim_node_shape:  missing lookup object %s",s);
//					WARN(ERROR_STRING);
//					DUMP_TREE(enp);
//				}
//				return;
//			}
//#endif /* CAUTIOUS */
			assert( dp != NO_OBJ );

			goto gen_obj_shape;

		case T_STATIC_OBJ:			/* prelim_node_shape */
			dp=VN_OBJ(enp);
			goto handle_obj;

		case T_DYN_OBJ:				/* prelim_node_shape */
			/* BUG we should use the identifier?? */
			dp=DOBJ_OF(VN_STRING(enp));

handle_obj:
			if( mode_is_matlab ){
				Identifier *idp;

				if( dp == NO_OBJ ){
					POINT_NODE_SHAPE(enp,uk_shape(PREC_DP));
					return;	/* not an error */
				}

				idp = ID_OF(VN_STRING(enp));
				if( idp == NO_IDENTIFIER ){
					POINT_NODE_SHAPE(enp,uk_shape(PREC_DP));
					return;	/* not an error */
				}

				POINT_NODE_SHAPE(enp,ID_SHAPE(idp));
				return;
			}

//#ifdef CAUTIOUS
//			if( dp==NO_OBJ ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,
//		"CAUTIOUS:  prelim_node_shape:  missing object %s",VN_STRING(enp));
//				WARN(ERROR_STRING);
//				DUMP_TREE(enp);
//				return;
//			}
//#endif /* CAUTIOUS */
			assert( dp != NO_OBJ );

gen_obj_shape:
			/* copy the size of this object to the node */

			/* If the object's shape is unknown, shouldn't
			 * we add it to the list here???
			 */
			decl_enp = (Vec_Expr_Node *)OBJ_EXTRA(dp);

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
	// decl_enp may be no good if it was an immediate
	// execution...
	// A warning used to print here, but it was
	// removed after cleaning up immediately executed nodes.

	// This is still problematic - if we have some immediate declarations
	// followed by a subroutine declaration, all in a file, then
	// the declarations will be released, leaving a dangling pointer...

	// It is not clear what the best solution is; we could keep
	// a reference count that would prohibit us from releasing
	// a referenced declaration node?

	// On the other hand, why do we want to point to the declaration
	// node at all?  Perhaps because for subroutine objects,
	// the objects come and go (hence "dynamic" objects)...

#ifdef FOOBAR
				if( (OBJ_FLAGS(dp)&DT_EXPORTED) == 0 && ! mode_is_matlab ){
sprintf(ERROR_STRING,"prelim_node_shape OBJECT %s (%s):  no decl node, and object was not exported!?",
VN_STRING(enp),node_desc(enp));
WARN(ERROR_STRING);
				}
#endif // FOOBAR
				POINT_NODE_SHAPE(enp,OBJ_SHAPE(dp));
			} else {
/*
sprintf(ERROR_STRING,"prelim_node_shape %s:  pointing to %s",node_desc(enp),node_desc(decl_enp));
advise(ERROR_STRING);
DESCRIBE_SHAPE(VN_SHAPE(decl_enp));
*/
//#ifdef CAUTIOUS
//				if( VN_SHAPE(decl_enp) == NO_SHAPE ){
//					NODE_ERROR(enp);
//					sprintf(ERROR_STRING,
//		"prelim_node_shape %s:  declaration node %s has no shape!?",node_desc(enp),
//						node_desc(decl_enp));
//					ERROR1(ERROR_STRING);
//				}
//#endif /* CAUTIOUS */
				assert( VN_SHAPE(decl_enp) != NO_SHAPE );

				POINT_NODE_SHAPE(enp,VN_SHAPE(decl_enp));
			}
			/* we link all references to the declaration node, that way
			 * when we resolve the object we can get to all the other occurrences.
			 */
			if( UNKNOWN_SHAPE(VN_SHAPE(enp)) )
				note_uk_objref(QSP_ARG  decl_enp,enp);

			break;

		case T_POINTER:		/* prelim_node_shape */
			{
			Identifier *idp;
//#ifdef CAUTIOUS
//			idp = get_named_ptr(QSP_ARG  enp);
//			if( idp == NO_IDENTIFIER ) break;
//
//			if( PTR_DECL_VN(ID_PTR(idp)) == NO_VEXPR_NODE ){
//				sprintf(ERROR_STRING,
//					"CAUTIOUS:  prelim_node_shape:  pointer %s has no decl node!?",ID_NAME(idp));
//				WARN(ERROR_STRING);
//				return;
//			}
//			if( VN_SHAPE(PTR_DECL_VN(ID_PTR(idp))) == NO_SHAPE ){
//				sprintf(ERROR_STRING,
//					"CAUTIOUS:  prelim_node_shape:  pointer decl node has no shape!?");
//				WARN(ERROR_STRING);
//			}
//#else
//			idp = ID_OF(VN_STRING(enp));
//#endif /* CAUTIOUS */
			idp = ID_OF(VN_STRING(enp));

			assert( idp != NULL );
			assert( IS_POINTER(idp) );
			assert( PTR_DECL_VN(ID_PTR(idp)) != NO_VEXPR_NODE );
			assert( VN_SHAPE(PTR_DECL_VN(ID_PTR(idp))) != NO_SHAPE );

			POINT_NODE_SHAPE(enp,VN_SHAPE(PTR_DECL_VN(ID_PTR(idp))));
			break;
			}

		case T_END:			/* prelim_node_shape (matlab) */
		case T_LIT_INT:
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_DI));
			SET_VN_FLAG_BITS(enp, NODE_HAS_CONST_VALUE);
			break;
		case T_LIT_DBL:			/* prelim_node_shape */
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_DP));
			SET_VN_FLAG_BITS(enp, NODE_HAS_CONST_VALUE);
			break;
		case T_FILE_EXISTS:
			POINT_NODE_SHAPE(enp,scalar_shape(PREC_IN));
			break;

		case T_FIX_SIZE:		/* prelim_node_shape */
			LINK_UK_NODES(enp,VN_CHILD(enp,0));
			POINT_NODE_SHAPE(enp,uk_shape(VN_CHILD_PREC(enp,0)));
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
			POINT_NODE_SHAPE(enp,SR_SHAPE(VN_SUBRT(enp)));
			break;

		case T_SET_FUNCPTR:		/* prelim_node_shape */
		case T_FUNCPTR:			/* prelim_node_shape */
		/* These get a shape when they are assigned */
			/* Do we know the return precision of the funcptr? */
			break;

		ALL_INCDEC_CASES
			POINT_NODE_SHAPE(enp,VN_CHILD_SHAPE(enp,0));
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
				copy_node_shape(enp,scalar_shape(PREC_DI));
				n1=EVAL_INT_EXP(VN_CHILD(enp,0));
				n2=EVAL_INT_EXP(VN_CHILD(enp,1));
				SET_SHP_COLS(VN_SHAPE(enp), floor( n2 - n1 ) );
				auto_shape_flags(VN_SHAPE(enp),NO_OBJ);
			} else {
				copy_node_shape(enp,uk_shape(PREC_DI));
			}
			break;

		case T_RANGE:
			copy_node_shape(enp,scalar_shape(PREC_DI));
			/* we need to figure out how many elements? */
			// children of a range node:  first, last, inc
			n1=EVAL_INT_EXP(VN_CHILD(enp,0));
			n2=EVAL_INT_EXP(VN_CHILD(enp,1));
			n3=EVAL_INT_EXP(VN_CHILD(enp,2));
			/* BUG? should we allow float start and stop? */
			// We store the number of samples in the column count
			// field of the range node, even though the range
			// could apply to any dimension of the object it is
			// subscripting...
			// 0:2:1	3 elements	3	3
			// 0:1:1	2 elements	2	2
			// 0:2:2	2 elements	3	1.5
			// 0:4:2	3 elements	5	2.5
			// 0:5:2	3 elements	6	3
			SET_SHP_COLS(VN_SHAPE(enp), ceil( ((n2-n1+1)/(double)n3 ) ) );
			/* NOT columns, could be any dimension... */
			auto_shape_flags(VN_SHAPE(enp),NO_OBJ);
			break;

		case T_STRING_LIST:	/* prelim_node_shape */
			{
			copy_node_shape(enp,scalar_shape(PREC_CHAR));
			SET_SHP_COLS(VN_SHAPE(enp), SHP_COLS(VN_CHILD_SHAPE(enp,0))
						+ SHP_COLS(VN_CHILD_SHAPE(enp,1))
						- 1 );
			auto_shape_flags(VN_SHAPE(enp),NO_OBJ);
			break;
			}

		case T_RET_LIST:		/* prelim_node_shape */
			LINK_UK_NODES(enp,VN_CHILD(enp,0));
			LINK_UK_NODES(enp,VN_CHILD(enp,1));
			/* when we have a list of matrices, who knows what we'll do...
			 * but for now, we assume it's a row of scalars!
			 */
			copy_node_shape(enp,uk_shape(PREC_NONE));
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

//#ifdef CAUTIOUS
//			verify_null_shape(QSP_ARG  enp);
//#endif /* CAUTIOUS */
			assert( VN_SHAPE(enp) == NO_SHAPE );
			break;

		NON_INIT_DECL_ITEM_CASES
			if( VN_DECL_PREC(enp) == NULL ){
				// parent could be T_DECL_STAT or list node?
				SET_VN_DECL_PREC(enp,parent_decl_prec(enp));
				//abort();
			}
			copy_node_shape(enp,uk_shape(PREC_CODE(VN_DECL_PREC(enp))));
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

			if( UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,0)) ){
				copy_node_shape(enp,VN_CHILD_SHAPE(enp,1));
				if( ! UNKNOWN_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
					copy_node_shape(VN_CHILD(enp,0),VN_CHILD_SHAPE(enp,1));
				} else {
					LINK_UK_NODES(enp,VN_CHILD(enp,1));
					LINK_UK_NODES(VN_CHILD(enp,0),VN_CHILD(enp,1));
				}
			}


			break;


		default:
			MISSING_CASE(enp,"prelim_node_shape");
			break;
	}
} /* end prelim_node_shape */

/* We have two nodes which can resolve each other - remember this! */

void link_uk_nodes(QSP_ARG_DECL  Vec_Expr_Node *enp1,Vec_Expr_Node *enp2)
{
	Node *np;

	if( VN_RESOLVERS(enp1) == NO_LIST )
		SET_VN_RESOLVERS(enp1, NEW_LIST );
	if( VN_RESOLVERS(enp2) == NO_LIST )
		SET_VN_RESOLVERS(enp2, NEW_LIST );

//#ifdef CAUTIOUS
//	/* When we started linking unknown args with the callfunc nodes,
//	 * we lost the parent relationship...
//	 * Does it matter???
//	 */
//
//	/* We make sure that these nodes aren't already linked before we make
//	 * a new link...
//	 */
//
//	if( nodeOf(VN_RESOLVERS(enp1),enp2) != NO_NODE ){
//		NODE_ERROR(enp1);
//		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  link_uk_nodes:  %s and %s are already linked!?",
//			node_desc(enp1),node_desc(enp2));
//		NWARN(DEFAULT_ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( nodeOf(VN_RESOLVERS(enp1),enp2) == NO_NODE );

	np = mk_node(enp1);
	addTail(VN_RESOLVERS(enp2),np);

	np = mk_node(enp2);
	addTail(VN_RESOLVERS(enp1),np);
}



int decl_count(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int i1,i2;

	if( enp==NO_VEXPR_NODE ) return(0);

	switch(VN_CODE(enp)){
		case T_DECL_STAT_LIST:
			i1=decl_count(QSP_ARG  VN_CHILD(enp,0));
			i2=decl_count(QSP_ARG  VN_CHILD(enp,1));
			return(i1+i2);

		case T_DECL_STAT:
			i1=decl_count(QSP_ARG  VN_CHILD(enp,0));
			return(i1);

		/* We don't need a case for T_EXTERN_DECL
		 * because this routine is only used for function args...
		 */

/* case T_SEQ_DECL: case T_VEC_DECL: case T_SCAL_DECL: case T_IMG_DECL: case T_PTR_DECL: case T_FUNCPTR_DECL: case T_CSEQ_DECL: case T_CIMG_DECL: case T_CVEC_DECL: case T_CSCAL_DECL: */
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

	while(enp!=NO_VEXPR_NODE){

		switch(VN_CODE(enp)){
			case T_RETURN:
				return(1);

			case T_STAT_LIST:
				enp = VN_CHILD(enp,1);
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
				if( ! final_return(QSP_ARG  VN_CHILD(enp,1)) ) return(0);
				if( VN_CHILD(enp,2) == NO_VEXPR_NODE ) return(1);
				return( final_return(QSP_ARG  VN_CHILD(enp,2)) );

			default:
				MISSING_CASE(enp,"final_return");
				return(0);
		}
	}
	return(0);
} /* final_return */

Shape_Info *calc_outer_shape(Vec_Expr_Node *enp1, Vec_Expr_Node *enp2)
{
	static Shape_Info * shpp1;
	int i;

	INIT_SHAPE_PTR(shpp1)

	/* we assume get_mating shapes has already been called,
	 * so we don't need to call check_typecast() again...
	 */

	for(i=0;i<N_DIMENSIONS;i++){
		dimension_t d1,d2;

		d1=SHP_TYPE_DIM(VN_SHAPE(enp1),i);
		d2=SHP_TYPE_DIM(VN_SHAPE(enp2),i);
		if( d1 == 1 )
			SET_SHP_TYPE_DIM(shpp1,i, d2);
		else if( d2 == 1 )
			SET_SHP_TYPE_DIM(shpp1,i, d1);
		else if( d1 == d2 )
			SET_SHP_TYPE_DIM(shpp1,i, d1);
		else
			/* do something else here? */
			return(NO_SHAPE);
	}
	/* Now we know the dimensions, we need to figure out the precision... */
	SET_SHP_PREC_PTR(shpp1, VN_PREC_PTR(enp1));

	/* BUG?  are mach_dim and type_dim both set? */
	SET_SHP_N_MACH_ELTS(shpp1, 1);
	SET_SHP_N_TYPE_ELTS(shpp1, 1);
	for(i=0;i<N_DIMENSIONS;i++){
		SET_SHP_N_MACH_ELTS(shpp1,
			SHP_N_MACH_ELTS(shpp1) * SHP_MACH_DIM(shpp1,i) );
		SET_SHP_N_TYPE_ELTS(shpp1,
			SHP_N_TYPE_ELTS(shpp1) * SHP_TYPE_DIM(shpp1,i) );
	}

	auto_shape_flags(shpp1,NO_OBJ);

	return(shpp1);
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

	exch_enp = VN_CHILD(enp,heavy_side);

/*		  a
 *		 / \
 *		1   b
 *		   / \
 *		  2   c
 *		     / \
 *		    3   4
 */
	SET_VN_CHILD(enp,heavy_side, VN_CHILD(exch_enp,light_side));
	SET_VN_PARENT(VN_CHILD(enp,heavy_side), enp);


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

	parent_enp = VN_PARENT(enp);

	/* There may be no parent node, if this is the root of a subrt body */
	if( parent_enp != NO_VEXPR_NODE ){
		index = (-1);
		for(i=0;i<MAX_CHILDREN(parent_enp);i++)
			if( VN_CHILD(parent_enp,i) == enp ){
				index=i;
				i=MAX_NODE_CHILDREN+1;
			}
//#ifdef CAUTIOUS
//		if( index < 0 ){
//			sprintf(DEFAULT_ERROR_STRING,"lighten_branch %s: is not a child of %s!?",node_desc(enp),node_desc(parent_enp));
//			NERROR1(DEFAULT_ERROR_STRING);
//		}
//#endif /* CAUTIOUS */
		assert( index >= 0 );

		SET_VN_CHILD(parent_enp,index, exch_enp);
		SET_VN_PARENT(exch_enp, parent_enp);
	} else {
		SET_VN_PARENT(exch_enp, NO_VEXPR_NODE);
	}

	SET_VN_CHILD(exch_enp,light_side, enp);
	SET_VN_PARENT(enp, exch_enp);

	/* BUG there is a more efficient way to compute these node counts! */
	SET_VN_N_ELTS(enp, leaf_count(enp,VN_CODE(enp)) );
	SET_VN_N_ELTS(exch_enp, leaf_count(exch_enp,VN_CODE(enp)) );
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

	if( enp == NO_VEXPR_NODE ){
		// This occurs if we have an empty subroutine...
		//NWARN("CAUTIOUS:  balance_list passed null node");
		return(enp);
	}

	if( VN_CODE(enp) != list_code ) return(enp);

	/* first balance the subtrees */
	if( balance_list(VN_CHILD(enp,0),list_code) == NO_VEXPR_NODE )
		return NO_VEXPR_NODE;
	if( balance_list(VN_CHILD(enp,1),list_code) == NO_VEXPR_NODE )
		return NO_VEXPR_NODE;

	n1=leaf_count(VN_CHILD(enp,0),list_code);
	n2=leaf_count(VN_CHILD(enp,1),list_code);

	if( n1 > (2*n2) ){	/* node is left_heavy */
		root_enp = lighten_branch(enp,0);
		balance_list(enp,list_code);
	} else if( n2 > (2*n1) ){	/* node is right-heavy */
		root_enp = lighten_branch(enp,1);
		balance_list(enp,list_code);
	} else root_enp = enp;
	return(root_enp);
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
sprintf(ERROR_STRING,"compile_subrt %s",SR_NAME(srp));
advise(ERROR_STRING);
*/

//#ifdef CAUTIOUS
//	if( IS_COMPILED(srp) ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  compile_subrt:  Subroutine %s has already been compiled",SR_NAME(srp));
//		WARN(ERROR_STRING);
//		abort();
//	}
//#endif /* CAUTIOUS */

	assert( ! IS_COMPILED(srp) );

//fprintf(stderr,"compile_subrt BEGIN\n");
//DUMP_TREE(SR_BODY(srp));

	SET_SR_FLAG_BITS(srp, SR_COMPILED);

/*
sprintf(ERROR_STRING,"compile_subrt %s",SR_NAME(srp));
advise(ERROR_STRING);
*/
	executing=0;

	save_srp = curr_srp;
	curr_srp = srp;

	SET_SR_N_ARGS(srp, decl_count(QSP_ARG  SR_ARG_DECLS(srp)) );	/* set # args */

/*
sprintf(ERROR_STRING,"compile_subrt %s setting SCANNING flag",SR_NAME(srp));
advise(ERROR_STRING);
*/
	SET_SR_FLAG_BITS(srp, SR_SCANNING);

	/* set the context */
	set_subrt_ctx(QSP_ARG  SR_NAME(srp));

	/* declare the arg variables */
	EVAL_DECL_TREE(SR_ARG_DECLS(srp));

	/* no values to assign, because we haven't been called! */


/*
sprintf(ERROR_STRING,"compile_subrt %s, before balance_list",SR_NAME(srp));
advise(ERROR_STRING);
DUMP_TREE(SR_BODY(srp));
*/
	SR_BODY(srp) = balance_list(SR_BODY(srp),T_STAT_LIST);
/*
sprintf(ERROR_STRING,"compile_subrt %s, after balance_list",SR_NAME(srp));
advise(ERROR_STRING);
DUMP_TREE(SR_BODY(srp));
*/

	COMPILE_TREE(SR_BODY(srp));

	delete_subrt_ctx(QSP_ARG  SR_NAME(srp));

	if( SR_PREC_CODE(srp) != PREC_VOID && ! final_return(QSP_ARG  SR_BODY(srp)) ){
		/* what node should we report the error at ? */
		sprintf(ERROR_STRING,"subroutine %s does not end with a return statement",SR_NAME(srp));
		WARN(ERROR_STRING);
	}

/*
sprintf(ERROR_STRING,"compile_subrt %s clearing SCANNING flag",SR_NAME(srp));
advise(ERROR_STRING);
*/
	CLEAR_SR_FLAG_BITS(srp,SR_SCANNING);

	curr_srp = save_srp;
}

/* compile_tree is used to compile the body of a subroutine,
 * it will evaluate declaration statements.
 */

void compile_tree(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	int i;
	Vec_Expr_Node *cenp;

	if( enp == NO_VEXPR_NODE ) return;

	/* BUG need to set & destroy the context! */

	/* How do we know whether or not to evaluate declarations?
	 * we do need to evaluate declarations within subroutines,
	 * but not global declarations...
	 */

	/* BUG?  we set up uk links in prelim_node_shape, but will
	 * we be ok for auto-init resolutions?
	 */

	if( VN_CODE(enp) == T_DECL_STAT || VN_CODE(enp) == T_EXTERN_DECL ){
		EVAL_TREE(enp,NO_OBJ);
		return;		/* BUG what if we don't return? */
	}

	for(i=0;i<MAX_CHILDREN(enp);i++){
		if( VN_CHILD(enp,i) != NO_VEXPR_NODE )
			COMPILE_TREE(VN_CHILD(enp,i));
	}

	/* now all the child nodes have been scanned, process this one */

	cenp=compile_node(enp);
	prelim_node_shape(cenp);
	/* should we check to see if cenp == enp??? */
} /* end compile_tree */

/* compile_prog is used to compile  a program with external decls and statements.
 * it will NOT evaluate declaration statements.
 * (why not??)
 */

Vec_Expr_Node * compile_prog(QSP_ARG_DECL   Vec_Expr_Node *enp)
{
	int i;
	//Vec_Expr_Node *orig_enp;

	//orig_enp = enp;
	executing=0;

	if( enp == NO_VEXPR_NODE ) return(enp);

	/* We didn't used to compile declarations, but after we started allowing
	 * auto-initialization with expressions, this became necessary.
	 */

	/*
	if( VN_CODE(enp) == T_DECL_STAT || VN_CODE(enp) == T_DECL_STAT_LIST )
		return;
	*/

	/* prototype declarations don't need to be compiled... */
	if( VN_CODE(enp) == T_PROTO )
		return(enp);

	for(i=0;i<MAX_CHILDREN(enp);i++){
		if( VN_CHILD(enp,i) != NO_VEXPR_NODE ){
			// compile_prog used to be passed a ptr to an enp (the child node),
			// which could be overwritten.
			// Now we use the return value instead...
			Vec_Expr_Node *cenp;
			cenp = COMPILE_PROG(VN_CHILD(enp,i));
			if( cenp != VN_CHILD(enp,i) )
				SET_VN_CHILD(enp,i,cenp);
		}
	}

	/* now all the child nodes have been scanned, process this one */

	//enp = compile_node(&enp);
	enp = compile_node(enp);
//sprintf(ERROR_STRING,"compile_prog %s calling prelim_node_shape",node_desc(enp));
//advise(ERROR_STRING);
	prelim_node_shape(enp);

	/* Here we used to overwrite the variable we called with... */
	//if( enp != *enpp )
	//	*enpp = enp;

	return(enp);
} /* end compile_prog */


#ifdef NOT_YET
/* enp is a T_ASSIGN node that has a vector subscript on the left ( a[coords] = samples )
 *
 * The goal here is to transform the node to one which is render(&a,&coords,&samples)
 */

static Vec_Expr_Node * fix_render_code(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Vec_Expr_Node *new_enp, *a1_enp, *lhs_enp;
	Vec_Expr_Node *l1_enp, *l2_enp;

	lhs_enp = VN_CHILD(enp,0);
	if( VN_CODE(lhs_enp) != T_SQUARE_SUBSCR ){
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
	a1_enp = NODE1(T_REFERENCE,VN_CHILD(lhs_enp,0));
	l1_enp = NODE2(T_ARGLIST,a1_enp,VN_CHILD(lhs_enp,1));
	l2_enp = NODE2(T_ARGLIST,l1_enp,VN_CHILD(enp,1));

	new_enp = NODE1(T_CALL_NATIVE,l2_enp);
	SET_VN_INTVAL(new_enp, NATIVE_RENDER);
	return(new_enp);
} /* end fix_render_code */
#endif /* NOT_YET */

#ifdef FOOBAR
static void check_vx_vx_condass_code(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,1)) ){
		if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,2)) ){
			/* vector-vector op */
			SET_VN_CODE(enp, T_VV_VV_CONDASS);
		} else {
			/* vector scalar operation,
			 * second operand is the scalar,
			 * so we leave the order unchanged.
			 */
			SET_VN_CODE(enp, T_VS_B_CONDASS);
		}
	} else {
		if( IS_VECTOR_SHAPE(VN_CHILD_SHAPE(enp,2)) ){
			/* vector scalar op, first node is
			 * the scalar, so we interchange the nodes.
			 * After we do this, however, we need to invert
			 * the sense of the boolean code!
			 */
			Vec_Expr_Node *tmp_enp;

			tmp_enp = VN_CHILD(enp,1);
			SET_VN_CHILD(enp,1, VN_CHILD(enp,2));
			SET_VN_CHILD(enp,2, tmp_enp);
			SET_VN_CODE(enp, T_VS_B_CONDASS);
			invert_bool_test(QSP_ARG VN_CHILD(enp,0));
			set_vs_bool_code(QSP_ARG  VN_CHILD(enp,0));
		} else {	/* scalar-scalar */
			SET_VN_CODE(enp, T_SS_B_CONDASS);
		}
	}
	check_bitmap_arg(enp,0);
} /* check_vx_vx_condass_code */
#endif /* FOOBAR */


void init_fixed_nodes(SINGLE_QSP_ARG_DECL)
{
	static int fixed_nodes_inited=0;

	int i;

	if( fixed_nodes_inited ) return;

	INIT_ENODE_PTR(minus_one_enp);

	// code must be set before calling init_expr_node!
	SET_VN_CODE(minus_one_enp, T_LIT_DBL);

	init_expr_node(QSP_ARG  minus_one_enp);
	SET_VN_DBLVAL(minus_one_enp, -1.0);
	POINT_NODE_SHAPE(minus_one_enp, scalar_shape(PREC_DP) );

	for(i=0;i<N_NAMED_PRECS;i++){
		_uk_shpp[i]=NO_SHAPE;
		_scalar_shpp[i]=NO_SHAPE;
	}
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

	//_scalar_shpp[index] = (Shape_Info *)getbuf(sizeof(Shape_Info));
	//SET_SHP_MACH_DIMS(_scalar_shpp[index],((Dimension_Set *)getbuf(sizeof(Dimension_Set))) );
	//SET_SHP_TYPE_DIMS(_scalar_shpp[index],((Dimension_Set *)getbuf(sizeof(Dimension_Set))) );
	_scalar_shpp[index] = alloc_shape();

	scalarize(_scalar_shpp[index]);

	/* should some of these things be put in scalarize? */
	SET_SHP_FLAGS(_scalar_shpp[index], 0);
	/* _scalar_shpp[index]->si_rowpad = 0; */

	SET_SHP_PREC_PTR(_scalar_shpp[index], PREC_FOR_CODE(prec) );

	auto_shape_flags(_scalar_shpp[index],NO_OBJ);

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
//#ifdef CAUTIOUS
	  /* Why are the complex cases CAUTIOUS??? */
	  else if( COMPLEX_PRECISION(prec) ){
		if( (prec& MACH_PREC_MASK) == PREC_SP )
			i_prec = N_MACHINE_PRECS + PP_CPX;
		else if( (prec&MACH_PREC_MASK) == PREC_DP )
			i_prec = N_MACHINE_PRECS + /* PP_DBLCPX */ PP_CPX ; /* BUG? double complex prec or type? */
		else {
//			sprintf(DEFAULT_ERROR_STRING,"uk_shape:  No complex support for machine precision %s",
//				NAME_FOR_PREC_CODE(prec&MACH_PREC_MASK));
//			NERROR1(DEFAULT_ERROR_STRING);
			assert( AERROR("uk_shape:  bad machine precision") );
			i_prec=0;	/* silence compiler warning NOTREACHED */
		}
	} else if( QUAT_PRECISION(prec) ){
		if( (prec& MACH_PREC_MASK) == PREC_SP )
			i_prec = N_MACHINE_PRECS + PP_QUAT;
		else if( (prec&MACH_PREC_MASK) == PREC_DP )
			i_prec = N_MACHINE_PRECS + /* PP_DBLCPX */ PP_QUAT ; /* BUG? double complex prec or type? */
		else {
//			sprintf(DEFAULT_ERROR_STRING,"uk_shape:  No quaternion support for machine precision %s",
//				NAME_FOR_PREC_CODE(prec&MACH_PREC_MASK));
//			NERROR1(DEFAULT_ERROR_STRING);
			assert( AERROR("uk_shape:  bad machine precision") );
			i_prec=0;	/* silence compiler warning NOTREACHED */
		}
	} else {
//		sprintf(DEFAULT_ERROR_STRING,"prec is %s (0x%x)",NAME_FOR_PREC_CODE(prec),prec);
//		NADVISE(DEFAULT_ERROR_STRING);
//		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  uk_shape:  don't know how to handle pseudo prec  %d (0%o 0x%x)",
//			prec,prec,prec);
//		NERROR1(DEFAULT_ERROR_STRING);
		assert( AERROR("uk_shape:  bad pseudo precision") );
		i_prec=0;	/* silence compiler warning NOTREACHED */
	}
//#endif /* CAUTIOUS */

	if( _uk_shpp[i_prec]!=NO_SHAPE ){
		return(_uk_shpp[i_prec]);
	}

	_uk_shpp[i_prec] = alloc_shape();
	for(i=0;i<N_DIMENSIONS;i++)
		SET_SHP_TYPE_DIM(_uk_shpp[i_prec],i,0);

	/* auto_shape_flags(_uk_shpp[i_prec],NO_OBJ); */
	SET_SHP_FLAGS(_uk_shpp[i_prec], DT_UNKNOWN_SHAPE);
	SET_SHP_PREC_PTR(_uk_shpp[i_prec], PREC_FOR_CODE(prec));
	/* Set BIT & COMPLEX flags if necessary */
	if( COMPLEX_PRECISION(prec) )
		SET_SHP_FLAG_BITS(_uk_shpp[i_prec], DT_COMPLEX);
	if( BITMAP_PRECISION(prec) )
		SET_SHP_FLAG_BITS(_uk_shpp[i_prec], DT_BIT);

	return(_uk_shpp[i_prec]);
}

int shapes_match(Shape_Info *shpp1,Shape_Info *shpp2)
{
	int i;

	if( shpp1 == NO_SHAPE ){
		if( shpp2 == NO_SHAPE ) return(1);
		else return(0);
	} else if( shpp2 == NO_SHAPE ) return(0);

	for(i=0;i<N_DIMENSIONS;i++)
		if( SHP_TYPE_DIM(shpp1,i) != SHP_TYPE_DIM(shpp2,i) )
			return(0);
	return(1);
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
			if( SHP_TYPE_DIM(shpp1,i) != SHP_TYPE_DIM(shpp2,i) )
				goto mismatch;
		}
		return(shpp1);
	} else if( COMPLEX_SHAPE(shpp2) && ! COMPLEX_SHAPE(shpp1) ){
		/* make sure the other dimensions match */
		int i;

		for(i=1;i<N_DIMENSIONS;i++){
			if( SHP_TYPE_DIM(shpp1,i) != SHP_TYPE_DIM(shpp2,i) )
				goto mismatch;
		}
		return(shpp2);
	}
#endif /* FOOBAR */

/*
DESCRIBE_SHAPE(shpp1);
DESCRIBE_SHAPE(shpp2);
	advise("product_shape:  should we fall through??");
	*/

	/* The shapes don't mate, but they may be ok for an outer binop...
	 * If so, we return the outer shape.
	 */
	{
		Shape_Info *shpp;
		int i;

		INIT_SHAPE_PTR(shpp)

		SET_SHP_N_TYPE_ELTS(shpp,1);
		for(i=0;i<N_DIMENSIONS;i++){
			if( SHP_TYPE_DIM(shpp1,i) == 1 ){
				SET_SHP_TYPE_DIM(shpp,i, SHP_TYPE_DIM(shpp2,i) );
			} else if( SHP_TYPE_DIM(shpp2,i) == 1 ){
				SET_SHP_TYPE_DIM(shpp,i, SHP_TYPE_DIM(shpp1,i) );
			} else if( SHP_TYPE_DIM(shpp1,i) == SHP_TYPE_DIM(shpp2,i) ){
				SET_SHP_TYPE_DIM(shpp,i, SHP_TYPE_DIM(shpp1,i) );
			} else {
				/* mismatch */
				goto mismatch;
			}
			SET_SHP_N_TYPE_ELTS(shpp,
				SHP_N_TYPE_ELTS(shpp) * SHP_TYPE_DIM(shpp,i) );
		}
		/* BUG - should we set si_n_mach_elts etc? */
		/* we assume the precisions match ...  is this correct?  BUG? */
		SET_SHP_PREC_PTR(shpp, SHP_PREC_PTR(shpp1) );
		auto_shape_flags(shpp,NO_OBJ);
		return(shpp);	/* BUG?  can we get away with a single static shape here??? */
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

	if( VN_CODE(enp) != T_ARGLIST ){
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

	if( n < (l=leaf_count(VN_CHILD(enp,0),T_ARGLIST)) ){
		return( nth_arg(QSP_ARG  VN_CHILD(enp,0),n) );
	} else {
		return( nth_arg(QSP_ARG  VN_CHILD(enp,1),n-l) );
	}
	/* NOTREACHED */
	return(NO_VEXPR_NODE);
}

//#ifdef CAUTIOUS
//void verify_null_shape(QSP_ARG_DECL  Vec_Expr_Node *enp)
//{
//	if( VN_SHAPE(enp) != NO_SHAPE ){
//		NODE_ERROR(enp);
//		DUMP_TREE(enp);
//		sprintf(ERROR_STRING,"CAUTIOUS:  verify_null_shape:  %s has a non-null shape ptr!?",node_desc(enp));
//		ERROR1(ERROR_STRING);
//	}
//}
//#endif /* CAUTIOUS */

void update_tree_shape(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int i;

	if( IS_CURDLED(enp) ) return;

	for(i=0;i<MAX_CHILDREN(enp);i++)
		if( VN_CHILD(enp,i) != NO_VEXPR_NODE ){
			UPDATE_TREE_SHAPE(VN_CHILD(enp,i));
		}

	update_node_shape(enp);
}


const char *get_lhs_name(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	switch(VN_CODE(enp)){
		case T_POINTER:
		case T_STR_PTR:
		case T_DYN_OBJ:
			return(VN_STRING(enp));
		case T_STATIC_OBJ:
			return(OBJ_NAME(VN_OBJ(enp)));
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
			return( GET_LHS_NAME(VN_CHILD(enp,0)) );

		case T_OBJ_LOOKUP:
			return( EVAL_STRING(VN_CHILD(enp,0)) );

		case T_UNDEF: return("undefined symbol");

		default:
			MISSING_CASE(enp,"get_lhs_name");
			break;
	}
	return(NULL);
}


void shapify(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	update_node_shape(enp);
}

#ifdef MAX_DEBUG
void show_shape_addrs(const char *s, Shape_Info *shpp)
{
	fprintf(stderr,"%s:\n",s);
	fprintf(stderr,"\tshpp = 0x%lx\n",(long)shpp);
	fprintf(stderr,"\tmach_dims = 0x%lx\n",(long)SHP_MACH_DIMS(shpp));
	fprintf(stderr,"\ttype_dims = 0x%lx\n",(long)SHP_TYPE_DIMS(shpp));
	fprintf(stderr,"\tmach_incs = 0x%lx\n",(long)SHP_MACH_INCS(shpp));
	fprintf(stderr,"\ttype_incs = 0x%lx\n",(long)SHP_TYPE_INCS(shpp));
}
#endif // MAX_DEBUG
