/*#define QUIP_DEBUG_ONLY */

#include "quip_config.h"

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
#include "data_obj.h"
#include "veclib_api.h"
#include "veclib/vl2_veclib_prot.h"
#include "vec_util.h"		/* dilate, erode */
//#include "../veclib/nvf.h"		/* show_obj_args, for debugging */
#include "fileck.h"
#include "vectree.h"
#include "vt_native.h"
#include "subrt.h"
#include "query_stack.h"		// BUG?
#include "platform.h"

//#include "mlab.h"

#define MAX_HIDDEN_CONTEXTS	32

#define push_cpair(cpp)		_push_cpair(QSP_ARG  cpp)

static inline void _push_cpair(QSP_ARG_DECL  Context_Pair *cpp)
{
//fprintf(stderr,"push_cpair:  pushing %s\n",CTX_NAME(CP_OBJ_CTX(cpp)));
	PUSH_ID_CONTEXT(CP_ID_CTX(cpp));
	push_dobj_context(CP_OBJ_CTX(cpp));
}

#define POP_CPAIR		POP_ID_CONTEXT;				\
				pop_dobj_context();

#define delete_local_objs() _delete_local_objs(SINGLE_QSP_ARG)

static void _delete_local_objs(SINGLE_QSP_ARG_DECL);

static void _insure_object_size(QSP_ARG_DECL  Data_Obj *dp,index_t index);
#define insure_object_size(dp,index) _insure_object_size(QSP_ARG  dp,index)

// BUG TOO MANY GLOBALS, NOT THREAD-SAFE!?
// NEED TO ADD TO QS PARSER DATA STRUCT!!!

// BUG not thread-safe!?
static Dimension_Set *scalar_dsp=NULL;

/* BUG use of this global list make this not thread-safe - should be an element of parser_data! ... */
static List *local_obj_lp=NULL;

// Not sure what these are used for ???  BUG not thread-safe!?
static Item_Context *hidden_context[MAX_HIDDEN_CONTEXTS];
static int n_hidden_contexts=0;

// BUG these global vars are not thread safe!!!
Subrt *curr_srp=NULL;
int scanning_args=0;
static Vec_Expr_Node *iteration_enp = NULL;
static Vec_Expr_Node *eval_enp=NULL;
static const char *goto_label;

static int interrupted=0;
int executing=0;	/* executing is a flag which we use to determine whether or not we
			 * expect pointers to have valid values.
			 */

/* temp storage of return values */
static int subrt_ret_type=0;

static int expect_objs_assigned=1;
static int continuing=0;
static int breaking=0;
static int going=0;

static int expect_perfection=0;		/* for mapping... */

#ifdef QUIP_DEBUG
debug_flag_t eval_debug=0;
debug_flag_t scope_debug=0;
#endif /* QUIP_DEBUG */

/* local prototypes needed because of recursion... */
static void _eval_obj_assignment(QSP_ARG_DECL Data_Obj *,Vec_Expr_Node *enp);
static int _eval_work_tree(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj *dst_dp);
static Data_Obj *_create_list_lhs(QSP_ARG_DECL Vec_Expr_Node *enp);

#define MAP_SUBSCRIPTS(src_dp,index_dp,enp)		map_subscripts(QSP_ARG  src_dp,index_dp,enp)
#define assign_obj_from_scalar(enp,dp,svp)		_assign_obj_from_scalar(QSP_ARG  enp,dp,svp)
#define eval_mixed_list(enp)			_eval_mixed_list(QSP_ARG  enp)
#define eval_typecast(enp,dst_dp)		_eval_typecast(QSP_ARG  enp,dst_dp)
#define eval_bitmap(dst_dp,enp)		_eval_bitmap(QSP_ARG  dst_dp,enp)
#define eval_subscript1(dp,enp)		_eval_subscript1(QSP_ARG  dp,enp)
#define exec_reffunc(enp)	_exec_reffunc(QSP_ARG  enp)
#define find_case(enp,lval)	_find_case(QSP_ARG  enp,lval)
#define eval_work_tree(enp,dst_dp)		_eval_work_tree(QSP_ARG  enp,dst_dp)
#define create_list_lhs(enp)			_create_list_lhs(QSP_ARG  enp)
#define CREATE_MATRIX(enp,shpp)			create_matrix(QSP_ARG  enp,shpp)
#define ASSIGN_ROW(dp,index,enp)		assign_row(QSP_ARG  dp,index,enp)
#define ASSIGN_ELEMENT(dp,ri,ci,enp)		assign_element(QSP_ARG  dp,ri,ci,enp)
#define MLAB_LHS(dp,enp)			mlab_lhs(QSP_ARG  dp,enp)
#define MLAB_TARGET(dp,enp)			mlab_target(QSP_ARG  dp,enp)
#define eval_obj_id(enp)			_eval_obj_id(QSP_ARG  enp)
#define eval_ref_tree(enp,dst_idp)		_eval_ref_tree(QSP_ARG  enp,dst_idp)
#define run_reffunc(srp,enp,dst_idp)		_run_reffunc(QSP_ARG  srp,enp,dst_idp)
#define eval_scalar(svp,enp,prec)		_eval_scalar(QSP_ARG  svp,enp,prec)
#define assign_subrt_args(scp,arg_enp,val_enp,cpp)	_assign_subrt_args(QSP_ARG  scp,arg_enp,val_enp,cpp)
#define assign_ptr_arg(arg_enp,val_enp,curr_cpp,prev_cpp)	_assign_ptr_arg(QSP_ARG  arg_enp,val_enp,curr_cpp,prev_cpp)
#define assign_obj_from_list(dp,enp,offset)	_assign_obj_from_list(QSP_ARG  dp,enp,offset)
#define eval_print_stat(enp)			_eval_print_stat(QSP_ARG  enp)
#define eval_obj_assignment(dp,enp)		_eval_obj_assignment(QSP_ARG  dp,enp)
#define eval_dim_assignment(dp,enp)		_eval_dim_assignment(QSP_ARG  dp,enp)
#define eval_decl_stat(prec,enp,decl_flags)		_eval_decl_stat(QSP_ARG  prec,enp,decl_flags)
#define eval_extern_decl(prec_p,enp,decl_flags)		_eval_extern_decl(QSP_ARG  prec_p,enp,decl_flags)

#define get_arg_ptr(enp)	_get_arg_ptr(QSP_ARG  enp)
#define parse_script_args(enp,index,max_args)		_parse_script_args(QSP_ARG  enp,index,max_args)
#define eval_info_stat(enp)		_eval_info_stat(QSP_ARG  enp)
#define eval_display_stat(enp)		_eval_display_stat(QSP_ARG  enp)
#define GET_2_OPERANDS(enp,dpp1,dpp2,dst_dp)		get_2_operands(QSP_ARG  enp,dpp1,dpp2,dst_dp)

#define SUBSCR_TYPE(enp)	(VN_CODE(enp)==T_SQUARE_SUBSCR?SQUARE:CURLY)

#define max( n1 , n2 )		(n1>n2?n1:n2)

const char *(*native_string_func)(QSP_ARG_DECL  Vec_Expr_Node *)=_eval_vt_native_string;
float (*native_flt_func)(QSP_ARG_DECL  Vec_Expr_Node *)=_eval_vt_native_flt;
void (*native_work_func)(QSP_ARG_DECL  Vec_Expr_Node *)=_eval_vt_native_work;
void (*native_assign_func)(QSP_ARG_DECL  Data_Obj *,Vec_Expr_Node *)=_eval_vt_native_assignment;

#ifdef NOT_USED
static void eval_native_assignment(Data_Obj *dp,Vec_Expr_Node *enp)
{
	(*native_assign_func)(dp,enp);
}

static float eval_native_flt(Vec_Expr_Node *enp)
{
	return( (*native_flt_func)(enp) );
}

static const char * eval_native_string(Vec_Expr_Node *enp)
{
	return( (*native_string_func)(enp) );
}
#endif // NOT_USED

#define assign_scalar_id(idp, val_enp) _assign_scalar_id(QSP_ARG  idp, val_enp)

static void _assign_scalar_id(QSP_ARG_DECL  Identifier *idp, Vec_Expr_Node *val_enp)
{
	double d;
	d = eval_flt_exp(val_enp);
	cast_dbl_to_scalar_value(ID_SVAL_PTR(idp),ID_PREC_PTR(idp),d);
}

#define eval_native_work(enp) _eval_native_work(QSP_ARG  enp)

static void _eval_native_work(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	(*native_work_func)(QSP_ARG enp);
}

#define unset_object_warning(enp, dp) _unset_object_warning(QSP_ARG  enp, dp)

static void _unset_object_warning(QSP_ARG_DECL  Vec_Expr_Node *enp, Data_Obj *dp)
{
	node_error(enp);

	if( HAS_SOME_VALUES(dp) ){
		sprintf(ERROR_STRING,
			"unset_object_warning:  Object %s may be used before all values have been set.",
			OBJ_NAME(dp));
		advise(ERROR_STRING);
	} else {
		sprintf(ERROR_STRING,
			"unset_object_warning:  Object %s is used before value has been set!?",
			OBJ_NAME(dp));
		warn(ERROR_STRING);
	}
}

static long get_long_scalar_value(Data_Obj *dp)
{
	Scalar_Value *svp;
	long lval=0.0;

	svp = (Scalar_Value *)OBJ_DATA_PTR(dp);
	switch(OBJ_MACH_PREC(dp)){
		case PREC_BY:  lval = svp->u_b; break;
		case PREC_IN:  lval = svp->u_s; break;
		case PREC_DI:  lval = svp->u_l; break;
		case PREC_LI:  lval = (long) svp->u_ll; break;
		case PREC_SP:  lval = (long) svp->u_f; break;
		case PREC_DP:  lval = (long) svp->u_d; break;
		case PREC_UBY:  lval = svp->u_ub; break;
		case PREC_UIN:  lval = svp->u_us; break;
		case PREC_UDI:  lval = svp->u_ul; break;
		case PREC_ULI:  lval = (long) svp->u_ull; break;
		/* shut up compiler */
		case PREC_INVALID:
		case PREC_NONE:
		case N_MACHINE_PRECS:
			assert( AERROR("get_long_scalar_value:  nonsense precision") );
			break;
	}
	return(lval);
}

static double get_dbl_scalar_value(Data_Obj *dp)
{
	Scalar_Value *svp;
	double dval=0.0;

	svp = (Scalar_Value *)OBJ_DATA_PTR(dp);
	switch(OBJ_MACH_PREC(dp)){
		case PREC_BY:  dval = svp->u_b; break;
		case PREC_IN:  dval = svp->u_s; break;
		case PREC_DI:  dval = svp->u_l; break;
		case PREC_LI:  dval = svp->u_ll; break;
		case PREC_SP:  dval = svp->u_f; break;
		case PREC_DP:  dval = svp->u_d; break;
		case PREC_UBY:  dval = svp->u_ub; break;
		case PREC_UIN:  dval = svp->u_us; break;
		case PREC_UDI:  dval = svp->u_ul; break;
		case PREC_ULI:  dval = svp->u_ull; break;
		/* shut up compiler */
		case PREC_NONE:
		case PREC_INVALID:
		case N_MACHINE_PRECS:
			assert( AERROR("get_dbl_scalar_value:  nonsense precision") );
			break;
	}
	return(dval);
}

#ifdef NOT_USED
void show_id(QSP_ARG_DECL  Identifier *idp)
{
	sprintf(msg_str,"Identifier %s at 0x%"PRIxPTR":  ",ID_NAME(idp), (uintptr_t)idp);
	prt_msg_frag(msg_str);
	switch(ID_TYPE(idp)){
		case ID_OBJ_REF:  prt_msg("reference"); break;
		case ID_POINTER:  prt_msg("pointer"); break;
		case ID_STRING:  prt_msg("string"); break;
		default:
			prt_msg("");
			sprintf(ERROR_STRING,"missing case in show_id (%d)",ID_TYPE(idp));
			warn(ERROR_STRING);
			break;
	}
}
#endif /* NOT_USED */


#define prototype_mismatch(enp1,enp2) _prototype_mismatch(QSP_ARG  enp1,enp2)

static void _prototype_mismatch(QSP_ARG_DECL  Vec_Expr_Node *enp1,Vec_Expr_Node *enp2)
{
	node_error(enp1);
	warn("declaration conflicts with earlier prototype");
	node_error(enp2);
	advise("original prototype");
}

static void assign_pointer(Pointer *ptrp, Reference *refp)
{
fprintf(stderr,"assign_pointer( 0x%lx, 0x%lx ) BEGIN\n",(long)ptrp,(long)refp);
	SET_PTR_REF(ptrp, refp);
	/* the pointer declaration carries around the shape of its current contents? */
	/* so we don't need to copy the shape? */
	SET_PTR_FLAG_BITS(ptrp, POINTER_SET);
fprintf(stderr,"assign_pointer( 0x%lx, 0x%lx ) DONE\n",(long)ptrp,(long)refp);
}

static void dbl_to_scalar(Scalar_Value *svp,double dblval,Precision *prec_p)
{
	switch( PREC_CODE(prec_p) ){
		case PREC_BY:  svp->u_b = (char) dblval; break;
		case PREC_IN:  svp->u_s = (short) dblval; break;
		case PREC_DI:  svp->u_l = (int32_t) dblval; break;
		case PREC_SP:  svp->u_f = (float) dblval; break;
		case PREC_DP:  svp->u_d = dblval; break;
		case PREC_UBY:  svp->u_ub = (u_char) dblval; break;
		case PREC_UIN:  svp->u_us = (u_short) dblval; break;
		case PREC_UDI:  svp->u_ul = (uint32_t) dblval; break;
		case PREC_CPX:
			svp->u_fc[0] = (float) dblval;
			svp->u_fc[1] = 0.0;
			break;

		case PREC_DBLCPX:
			svp->u_dc[0] = dblval;
			svp->u_dc[1] = 0.0;
			break;

		case PREC_QUAT:
			svp->u_fq[0] = (float) dblval;
			svp->u_fq[1] = 0.0;
			svp->u_fq[2] = 0.0;
			svp->u_fq[3] = 0.0;
			break;

		case PREC_DBLQUAT:
			svp->u_dq[0] = dblval;
			svp->u_dq[1] = 0.0;
			svp->u_dq[2] = 0.0;
			svp->u_dq[3] = 0.0;
			break;

		case PREC_BIT:
			if( dblval != 0 )
				svp->u_l = 1;
			else
				svp->u_l = 0;
			break;

		/* BUG add default case */
		default:
			sprintf(DEFAULT_ERROR_STRING,"dbl_to_scalar:  unhandled precision %s",
				PREC_NAME(prec_p));
			NERROR1(DEFAULT_ERROR_STRING);
			IOS_RETURN
	}
}


static void int_to_scalar(Scalar_Value *svp,long intval,Precision *prec_p)
{
	switch( PREC_CODE(prec_p) ){
		case PREC_BY:  svp->u_b = (char) intval; break;
		case PREC_IN:  svp->u_s = (short) intval; break;
		case PREC_DI:  svp->u_l = (int32_t) intval; break;
		case PREC_SP:  svp->u_f = intval; break;
		case PREC_DP:  svp->u_d = intval; break;
		case PREC_CHAR:
		case PREC_UBY:  svp->u_ub = (u_char) intval; break;
		case PREC_UIN:  svp->u_us = (u_short) intval; break;
		case PREC_UDI:  svp->u_ul = (uint32_t) intval; break;
		case PREC_CPX:
			svp->u_fc[0] = intval;
			svp->u_fc[1] = 0;
			break;
		case PREC_DBLCPX:
			svp->u_dc[0] = intval;
			svp->u_dc[1] = 0;
			break;
		case PREC_BIT:
			if( intval )
				svp->u_l = 1;
			else
				svp->u_l = 0;
			break;
		default:
			assert( AERROR("int_to_scalar:  unhandled target precision") );
			break;
	}
}

/* dp_const should be used for floating point assignments... */

/*
 * dp_const - set object dp to the value indicated by svp
 *
 * The scalar value gets copied into a scalar object...
 * WHY???
 */

#define dp_const(dp,svp) _dp_const(QSP_ARG  dp,svp)

static Data_Obj *_dp_const(QSP_ARG_DECL  Data_Obj *dp,Scalar_Value * svp)
{
	Vec_Obj_Args oa1, *oap=&oa1;
	int status;

	setvarg1(oap,dp);	// has to come first (clears *oap)
	SET_OA_SVAL(oap,0, svp );
	status = perf_vfunc(FVSET,oap);
	if( status < 0 )
		dp = NULL;

	return( dp );
} /* end dp_const() */

#define zero_dp(dp) _zero_dp(QSP_ARG  dp)

static int _zero_dp(QSP_ARG_DECL  Data_Obj *dp)
{
	Scalar_Value sval;

	switch(OBJ_PREC(dp)){
		case PREC_SP:  sval.u_f = 0.0; break;
		case PREC_DP:  sval.u_d = 0.0; break;
		case PREC_BY:  sval.u_b = 0; break;
		case PREC_IN:  sval.u_s = 0; break;
		case PREC_DI:  sval.u_l = 0; break;
		default:
			assert( AERROR("zero_dp:  unhandled precision") );
			break;
	}
	if( dp_const(dp,&sval) == NULL ) return -1;
	return 0;
}

static int _assign_obj_from_scalar(QSP_ARG_DECL  Vec_Expr_Node *enp,Data_Obj *dp,Scalar_Value *svp)
{
	if( dp_const(dp,svp) == NULL ){
		node_error(enp);
		sprintf(ERROR_STRING,"Error assigning object %s from scalar value",OBJ_NAME(dp));
		warn(ERROR_STRING);
		return -1;
	}
	return 0;
}

void _missing_case(QSP_ARG_DECL  Vec_Expr_Node *enp,const char *func_name)
{
	node_error(enp);
	sprintf(ERROR_STRING,
		"Code %s (%d) not handled by %s switch",
		NNAME(enp),VN_CODE(enp),func_name);
	warn(ERROR_STRING);
	dump_tree(enp);
	advise("");
}

#define xpose_data(dpto,dpfr) _xpose_data(QSP_ARG  dpto,dpfr)

static void _xpose_data(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	float *fr, *to;
	dimension_t i,j,k;

	if( OBJ_ROWS(dpto) != OBJ_COLS(dpfr) ){
		sprintf(ERROR_STRING,
	"xpose_data:  # destination rows object %s (%d) should match # source cols object %s (%d)",
			OBJ_NAME(dpto),OBJ_ROWS(dpto),OBJ_NAME(dpfr),OBJ_COLS(dpfr));
		warn(ERROR_STRING);
		return;
	}
	if( OBJ_COLS(dpto) != OBJ_ROWS(dpfr) ){
		sprintf(ERROR_STRING,
	"xpose_data:  # destination cols object %s (%d) should match # source rows object %s (%d)",
			OBJ_NAME(dpto),OBJ_COLS(dpto),OBJ_NAME(dpfr),OBJ_ROWS(dpfr));
		warn(ERROR_STRING);
		return;
	}
	if( OBJ_PREC(dpto) != PREC_SP ){
		warn("Sorry, now can only transpose float objects");
		return;
	}

	/* BUG if different prec's we could do the conversion... */

	for(i=0;i<OBJ_ROWS(dpto);i++){
		fr = ((float *)OBJ_DATA_PTR(dpfr)) + i*OBJ_PXL_INC(dpfr);
		to = ((float *)OBJ_DATA_PTR(dpto)) + i*OBJ_ROW_INC(dpto);
		for(j=0;j<OBJ_COLS(dpto);j++){
			for(k=0;k<OBJ_COMPS(dpfr);k++)
				*(to+k*OBJ_COMP_INC(dpto)) = *(fr+k*OBJ_COMP_INC(dpfr));
			to += OBJ_PXL_INC(dpto);
			fr += OBJ_ROW_INC(dpfr);
		}
	}
}

static Data_Obj *map_source_dp=NULL;

/* GET_MAP_OFFSET
 *
 * Read the index data from the index object into the array "indices"
 * There is one for each addressable dimension.
 * Once this is done, the overall offset is computed.
 *
 * For floating point indices, we have 2 indices for each dimension -
 * if there are 2 dimensions (as in an image) then we need to cache 4
 * offsets.
 */

/* Some static variables to support floating point indices */
static incr_t lower_index[N_DIMENSIONS];
static incr_t upper_index[N_DIMENSIONS];
static double weights[N_DIMENSIONS];
static incr_t sample_offset[32];		/* 32 = 2 ^ N_DIMENSIONS */
static double sample_weight[32];	/* 32 = 2 ^ N_DIMENSIONS */


#define GET_MAP_OFFSET(type)						\
									\
{									\
	type *ip;							\
									\
	ip = (type *)OBJ_DATA_PTR(index_dp);				\
	ip += i_index;							\
	while( i_dim >= OBJ_MINDIM(map_source_dp) ){			\
		indices[i_dim]= (index_t) (*ip);			\
		if( indices[i_dim] > OBJ_TYPE_DIM(map_source_dp,i_dim) ){ \
			node_error(iteration_enp);			\
			sprintf(ERROR_STRING,				\
"map_iteration:  GET_MAP_OFFSET:  index %d is out of range for %s dimension (%d) of source object %s",		\
	indices[i_dim],dimension_name[i_dim],				\
	OBJ_TYPE_DIM(map_source_dp,i_dim),				\
	OBJ_NAME(map_source_dp));					\
			warn(ERROR_STRING);				\
			indices[i_dim]=0;				\
		}							\
		ip += OBJ_TYPE_INC(index_dp,0);				\
		i_dim--;						\
	}								\
	offset=0;							\
	for(i=0;i<N_DIMENSIONS;i++)					\
		offset += indices[i] * OBJ_TYPE_INC(map_source_dp,i);	\
}

/* GET_MAP_WEIGHTS
 *
 * We use this in the case we have a floating point index, and the desired goal
 * is to linearly interpolate between neighboring values.  For each dimension,
 * we have upper_index and lower_index which tell us which values to interpolate,
 * and a weight (which stores the fractional part).  When we iterate through
 * all of the samples, we have to take the product of all the dimension weights
 * (complementing weights based on which sample we are doing).
 *
 * When we fall outside of the image, what should we do?
 *
 * i_dim starts out at OBJ_MAXDIM(source_dp)...
 * Here we assume that the type dimension of index_dp is appropriate for the source object.
 * Each dimension we are interpolating over adds a factor of two to the number
 * of samples we have to consider...  For example, if we are sampling an image
 * with two floating point indices, in general we need to dereference 4 pixels and
 * compute 4 corresponding weights.
 *
 * What should the convention be for the ordering of the indices?
 * If a pixel is indexed by {a,b}, which should be the row index and
 * which shoul dbe the column index?
 */


#define GET_MAP_WEIGHTS(type)						\
									\
{									\
	type *ip;							\
	int ns,j;							\
									\
	/* first put in some defaults */				\
	for(i=0;i<N_DIMENSIONS;i++)					\
		lower_index[i]=upper_index[i]=0;			\
									\
	ip = (type *)OBJ_DATA_PTR(index_dp);				\
	ip += i_index;							\
	n_samples=1;							\
	/* i_dim is initialized to maxdim, so this loop is foreach index */ \
	/* ip points to the first component of the index object...	\
	 * this gets applied to the highest dimension... (WHY)		\
	 * we want the opposite for x-y pairs?				\
	 */								\
	while( i_dim >= OBJ_MINDIM(map_source_dp) ){			\
		double d;						\
		d = *ip;						\
		lower_index[i_dim]= (incr_t) floor(d);			\
		upper_index[i_dim]= (incr_t) ceil(d);			\
		weights[i_dim]= upper_index[i_dim]-d;			\
		if( lower_index[i_dim] < 0 ){				\
			if( expect_perfection ){			\
				node_error(iteration_enp);		\
				sprintf(ERROR_STRING,			\
"map_iteration:  GET_MAP_WEIGHTS:  index %g (rounded to %d) out of range for %s dimension (%d) of src %s",	\
		d,lower_index[i_dim],dimension_name[i_dim],		\
		OBJ_TYPE_DIM(map_source_dp,i_dim),			\
		OBJ_NAME(map_source_dp));				\
				warn(ERROR_STRING);			\
			}						\
			lower_index[i_dim]=upper_index[i_dim]=0;	\
			weights[i_dim]=0.0;				\
		} else if( upper_index[i_dim] >= 			\
			(incr_t) OBJ_TYPE_DIM(map_source_dp,i_dim) ){	\
			if( expect_perfection ){			\
				node_error(iteration_enp);		\
				sprintf(ERROR_STRING,			\
"map_iteration:  GET_MAP_WEIGHTS:  index %g (rounded to %d) out of range for %s dimension (%d) of src %s",	\
		d,upper_index[i_dim],dimension_name[i_dim],		\
				OBJ_TYPE_DIM(map_source_dp,i_dim),	\
		OBJ_NAME(map_source_dp));				\
				warn(ERROR_STRING);			\
			}						\
			lower_index[i_dim]=upper_index[i_dim]=OBJ_TYPE_DIM(map_source_dp,i_dim)-1;\
			weights[i_dim]=1.0;				\
		}							\
		ip += OBJ_TYPE_INC(index_dp,0);				\
		n_samples *= 2;						\
		i_dim--;						\
	}								\
	offset=0;							\
	sample_offset[0] = 0;						\
	sample_weight[0] = 1.0;						\
									\
	ns=1;								\
	for(i=0;i<N_DIMENSIONS;i++){					\
		if( (short)i >= OBJ_MINDIM(map_source_dp) && 		\
			(short)i <= OBJ_MAXDIM(map_source_dp) ){	\
		/* We double the number of pts to interpolate here */	\
			for(j=ns;j<2*ns;j++){				\
				sample_offset[j] = sample_offset[j-ns] + \
					upper_index[i] * 		\
					OBJ_TYPE_INC(map_source_dp,i);	\
				sample_weight[j] = sample_weight[j-ns]	\
						* (1.0 - weights[i]);	\
			}						\
			for(j=0;j<ns;j++){				\
				sample_offset[j] += lower_index[i]	\
					* OBJ_TYPE_INC(map_source_dp,i); \
				sample_weight[j] *= weights[i];		\
			}						\
			ns *= 2;					\
		}							\
	}								\
}


#define MAP_IT( type )							\
									\
	{								\
		type *srcp,*dstp;					\
									\
		srcp=(type *)OBJ_DATA_PTR(map_source_dp);		\
		srcp+=offset;						\
		dstp=(type *)OBJ_DATA_PTR(dst_dp);			\
		dstp+=i_dst;						\
		*dstp = *srcp;						\
	}

#define MAP_BILINEAR( type )						\
									\
	{								\
		type *srcp, v, *dstp;					\
		index_t mb_i;						\
		v=0;							\
		for(mb_i=0;mb_i<n_samples;mb_i++){			\
			srcp=(type *)OBJ_DATA_PTR(map_source_dp);	\
			srcp += sample_offset[mb_i];			\
			v += (*srcp) * sample_weight[mb_i];		\
		}							\
		dstp=(type *)OBJ_DATA_PTR(dst_dp);			\
		dstp+=i_dst;						\
		*dstp = v;						\
	}

#define UNHANDLED_MAP_CASES										\
													\
			case PREC_BY:									\
			case PREC_IN:									\
			case PREC_DI:									\
			case PREC_LI:									\
			case PREC_UIN:									\
			case PREC_UDI:									\
			case PREC_ULI:									\
			case PREC_DP:									\
				sprintf(ERROR_STRING,"map_iteration:  unhandled precision %s",		\
					NAME_FOR_PREC_CODE(mp));						\
				warn(ERROR_STRING);							\
				break;

#define INVALID_MAP_CASES(dp)						\
									\
		/* shouldn't happen, but these are valid enum's */	\
		case PREC_NONE:						\
		case N_MACHINE_PRECS:					\
		default:						\
			assert(AERROR("map_iteration:  illegal machine precision!?"));	\
			break;


static void map_iteration(QSP_ARG_DECL  Data_Obj *dst_dp,index_t i_dst, Data_Obj *index_dp, index_t i_index)
{
	/* Here we assume that the index object has the same dimensions (except for type)
	 * as the destination.  The typedimension of the index object should match the
	 * number of indexable dimensions of the source object.
	 *
	 * If these are both images (and the source) is an image,
	 * Then we will need several indices for the source - e.g. row and column.
	 * Therefore the type dimension of the index object should match the number
	 * of subscriptable dimensions in the source.  Any unused dimensions should
	 * either default to zero or be taken from the destination...
	 * 
	 * If we only give one dimension for an image, then each output "pixel"
	 * should be a row of the source image...  But for now we don't allow this,
	 * it seems unnecessarily complicated for the time being.
	 */

	index_t indices[N_DIMENSIONS];
	mach_prec mp;
	index_t i, offset;
	int i_dim;
	dimension_t n_samples;

	n_samples=1;			/* default */

	/* Now fetch the indices from the index object */
	for(i=0;i<N_DIMENSIONS;i++) indices[i]=0;
	i_dim = OBJ_MAXDIM(map_source_dp);
	mp = OBJ_MACH_PREC(index_dp);
	switch( mp ){
		case PREC_DI:
		case PREC_UDI: GET_MAP_OFFSET(uint32_t) break;
		case PREC_LI:
		case PREC_ULI: GET_MAP_OFFSET(uint64_t) break;
		case PREC_IN:
		case PREC_UIN: GET_MAP_OFFSET(u_short) break;
		case PREC_BY:
		case PREC_UBY: GET_MAP_OFFSET(u_char) break;
		/* for floating point indices, we will need to interpolate */
		case PREC_SP:  GET_MAP_WEIGHTS(float) break;
		case PREC_DP:  GET_MAP_WEIGHTS(double) break;

		INVALID_MAP_CASES(index_dp)

		/*
		default:
			sprintf(ERROR_STRING,"map_iteration:  index object %s, unsupported precision %s",
				OBJ_NAME(index_dp),OBJ_PREC_NAME(index_dp));
			warn(ERROR_STRING);
			return;
			break;
			*/
	}

	if( n_samples > 1 ){				/* we have to interpolate! */
		mp = OBJ_MACH_PREC(dst_dp);
		switch( mp ){
			case PREC_SP:  MAP_BILINEAR(float); break;
			case PREC_UBY: MAP_BILINEAR(u_char) break;

			UNHANDLED_MAP_CASES
			INVALID_MAP_CASES(dst_dp)

			/*
			default:
				sprintf(ERROR_STRING,"map_iteration:  unhandled precision %s",
					OBJ_PREC_NAME(dst_dp));
				warn(ERROR_STRING);
				break;
				*/
		}
	} else {
		mp = OBJ_MACH_PREC(dst_dp);
		switch( mp ){
			case PREC_SP: MAP_IT(float); break;
			case PREC_UBY: MAP_IT(u_char) break;

			UNHANDLED_MAP_CASES
			INVALID_MAP_CASES(dst_dp)
			/*
			default:
				sprintf(ERROR_STRING,"map_iteration:  unhandled precision %s",
					OBJ_PREC_NAME(dst_dp));
				warn(ERROR_STRING);
				break;
				*/
		}
	}
}

static void note_partial_assignment(Data_Obj *dp)
{
	//uint64_t x;
	//x = DT_PARTIALLY_ASSIGNED;

	SET_OBJ_FLAG_BITS(dp, DT_PARTIALLY_ASSIGNED);
	if( OBJ_PARENT(dp) != NULL ) note_partial_assignment(OBJ_PARENT(dp));
}

void note_assignment(Data_Obj *dp)
{
	SET_OBJ_FLAG_BITS(dp, DT_ASSIGNED);
	if( OBJ_PARENT(dp) != NULL ) note_partial_assignment(OBJ_PARENT(dp));
}

static Data_Obj *map_subscripts(QSP_ARG_DECL  Data_Obj *src_dp, Data_Obj *index_dp, Vec_Expr_Node *enp )
{
	/* Not quite sure how we ought to do this...  We'd like to be able
	 * to permute the pixels of an image using a single subscript with
	 * an image of indices.  But we might like to also have separate
	 * vectors of row and column indices.
	 *
	 * Let f be an image (4x5), and i an image (6x7)
	 * Then g = f[i] should be a 6x7 image
	 *
	 * If col is a column of indices with 8 elements, then 
	 * g = f[col] is an 8x5 image; each row is a row from f, indexed by col
	 */
	Data_Obj *dst_dp;
	//Dimension_Set dimset={{1,1,1,1,1}};
	Dimension_Set ds1, *dsp=(&ds1);
	dimension_t i;

	SET_DIMENSION(dsp,0,1);
	SET_DIMENSION(dsp,1,1);
	SET_DIMENSION(dsp,2,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,4,1);

	if( SUBSCR_TYPE(enp) != SQUARE )
		warn("map_subscripts:  Sorry, curly subscripts are not correctly handled...");

	/* For now, we create dst_dp to have the same dimensions as the index array... */
	SET_DIMENSION(dsp,0, OBJ_TYPE_DIM(src_dp,0) );	/* copy tdim from src_dp */
	for(i=1;i<N_DIMENSIONS;i++)
		SET_DIMENSION(dsp,i, OBJ_TYPE_DIM(index_dp,i) );	/* BUG need to do something better */

	dst_dp=make_local_dobj(dsp,OBJ_PREC_PTR(src_dp),OBJ_PFDEV(src_dp));

	if( dst_dp == NULL )
		return(dst_dp);

	/* Now check the sizes - we might like to use dp_same_size(), but we allow tdim to differ  */

	if( !dp_same_dims(dst_dp,index_dp,1,N_DIMENSIONS-1,"map_subscripts") ){
		node_error(enp);
		sprintf(ERROR_STRING,"map_subscripts:  objects %s and %s should have the same shape",
			OBJ_NAME(dst_dp),OBJ_NAME(index_dp));
		warn(ERROR_STRING);
		return(NULL);
	}
	/* Now figure out which dimensions we need to iterate over.
	 * For now, we assume that the destination has to have the same shape
	 * as the index array.
	 *
	 *
	 */
	/* Now iterate */
	map_source_dp = src_dp;			/* Do we have to introduce this global var? */

	iteration_enp = enp;			/* pass info to map_iteration via this global */
	dpair_iterate(dst_dp,index_dp,map_iteration);
	iteration_enp = NULL;

	//SET_OBJ_FLAG_BITS(dst_dp, DT_ASSIGNED);
	note_assignment(dst_dp);
	return(dst_dp);
} /* end map_subscripts */

#define do_vvfunc(dpto,dpfr1,dpfr2,code) _do_vvfunc(QSP_ARG  dpto,dpfr1,dpfr2,code)

static int _do_vvfunc(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr1,Data_Obj *dpfr2,Vec_Func_Code code)
{
	Vec_Obj_Args oa1, *oap=&oa1;
	int retval;

	if( code == FVMUL && COMPLEX_PRECISION(OBJ_PREC(dpfr2)) && ! COMPLEX_PRECISION(OBJ_PREC(dpfr1)) ){
		setvarg3(oap,dpto,dpfr2,dpfr1);
	} else {
		setvarg3(oap,dpto,dpfr1,dpfr2);
	}
	retval = perf_vfunc(code,oap) ;

	return( retval );
}

#define do_vsfunc(dpto,dpfr,svp,code) _do_vsfunc(QSP_ARG  dpto,dpfr,svp,code)

static int _do_vsfunc(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Scalar_Value *svp,Vec_Func_Code code)
{
	Vec_Obj_Args oa1, *oap=&oa1;
	int retval;

	setvarg2(oap,dpto,dpfr);
	SET_OA_SVAL(oap,0, svp );

	retval = perf_vfunc(code,oap);

	return( retval );
} // do_vsfunc

#define do_un0func(dpto,code) _do_un0func(QSP_ARG dpto,code)

static int _do_un0func(QSP_ARG_DECL Data_Obj *dpto,Vec_Func_Code code)
{
	Vec_Obj_Args oa1, *oap=&oa1;
	int retval;

	setvarg1(oap,dpto);
	retval = perf_vfunc(code,oap);

	return( retval );
}

#define do_unfunc(dpto,dpfr,code) _do_unfunc(QSP_ARG  dpto,dpfr,code)

static int _do_unfunc(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Vec_Func_Code code)
{
	Vec_Obj_Args oa1;
	int retval;

	setvarg2(&oa1,dpto,dpfr);
	retval = perf_vfunc(code,&oa1) ;

	return( retval );
}

#ifdef UNUSED
static Scalar_Value * take_inner(Data_Obj *dp1,Data_Obj *dp2)
{
	static Scalar_Value sval;

	assert( dp1 != NULL && dp2 != NULL );

	sprintf(ERROR_STRING,"take_inner %s %s:  unimplemented",
		OBJ_NAME(dp1),OBJ_NAME(dp2));
	warn(ERROR_STRING);

	switch( OBJ_MACH_PREC(dp1) ){
		case PREC_BY:  sval.u_b = 0; break;
		case PREC_IN:  sval.u_s = 0; break;
		case PREC_DI:  sval.u_l = 0; break;
		case PREC_SP:  sval.u_f = 0.0; break;
		case PREC_DP:  sval.u_d = 0.0; break;
		case PREC_UBY:  sval.u_ub = 0; break;
		case PREC_UIN:  sval.u_us = 0; break;
		case PREC_UDI:  sval.u_ul = 0; break;
		/* just to shut the compiler up */
		case PREC_NONE:
		case N_MACHINE_PRECS:
			assert( AERROR("take_inner:  nonsense precision!?") );
			/* can't happen? */
			break;

	}
	return(&sval);
}
#endif /* UNUSED */

#define assign_string(idp, str, enp) _assign_string(QSP_ARG  idp, str, enp)

static void _assign_string(QSP_ARG_DECL  Identifier *idp, const char *str, Vec_Expr_Node *enp)
{
	if( ! IS_STRING_ID(idp) ){
		node_error(enp);
		sprintf(ERROR_STRING,"assign_string:  identifier %s (type %d) does not refer to a string",
			ID_NAME(idp),ID_TYPE(idp));
		warn(ERROR_STRING);
		return;
	}

	/* copy_string(idp->id_sbp,str); */
	copy_string(REF_SBUF(ID_REF(idp)),str);
}

#define ptr_for_string(s,enp) _ptr_for_string(QSP_ARG  s,enp)

static Identifier *_ptr_for_string(QSP_ARG_DECL  const char *s,Vec_Expr_Node *enp)
{
	static int n_auto_strs=1;
	char idname[LLEN];
	Identifier *idp;

	/* We need to make an object and a reference... */

	sprintf(idname,"Lstr.%d",n_auto_strs++);
	idp = new_identifier(idname);
sprintf(ERROR_STRING,"ptr_for_string:  creating id %s",idname);
advise(ERROR_STRING);
	SET_ID_TYPE(idp, ID_STRING);

	/* Can't do this, because refp is in a union w/ sbp... */
	SET_ID_REF(idp, NEW_REFERENCE );
	SET_REF_TYPE(ID_REF(idp), STR_REFERENCE );
	SET_REF_ID(ID_REF(idp), idp );
	SET_REF_DECL_VN(ID_REF(idp), NULL );
	/* SET_REF_OBJ(ID_REF(idp), NULL ); */
	SET_REF_SBUF(ID_REF(idp), new_stringbuf() );

	assign_string(idp,s,enp);

	return( idp );
}

/* Get the data object for this value node.
 * We use this routine for call-by-reference.
 */

static Identifier *_get_arg_ptr(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Identifier *idp;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_STATIC_OBJ:		/* get_arg_ptr */
			node_error(enp);
			sprintf(ERROR_STRING,"object %s not properly referenced, try prepending &",OBJ_NAME(VN_OBJ(enp)));
			advise(ERROR_STRING);
			idp = get_id(OBJ_NAME(VN_OBJ(enp)));
			return(idp);
			break;

		case T_DYN_OBJ:		/* get_arg_ptr */
			node_error(enp);
			sprintf(ERROR_STRING,"object %s not properly referenced, try prepending &",VN_STRING(enp));
			advise(ERROR_STRING);
			idp = get_id(VN_STRING(enp));
			return(idp);
			break;

		case T_REFERENCE:
		case T_POINTER:
		case T_STR_PTR:
			return( eval_ptr_expr(enp,EXPECT_PTR_SET) );


		case T_SET_STR:			/* get_arg_ptr */
			/* an assignment statement as a function arg...
			 * we need to execute it!
			 */
		case T_STRING:
			/* we need to make up an object for this string...
			 * BUG this is going to be a memory leak!?
			 */
			return( ptr_for_string( eval_string(enp), enp ) );
			break;

		default:
			missing_case(enp,"get_arg_ptr");
			break;
	}
	return(NULL);
}

#define get_id_obj(name, enp) _get_id_obj(QSP_ARG  name, enp)

static Data_Obj *_get_id_obj(QSP_ARG_DECL  const char *name, Vec_Expr_Node *enp)
{
	Identifier *idp;

	idp = /* get_id */ id_of(name);

	assert( idp != NULL );
	assert( IS_OBJ_REF(idp) );
	assert( ! strcmp(ID_NAME(idp),OBJ_NAME(REF_OBJ(ID_REF(idp)))) );

	{
		Data_Obj *dp;
		dp = dobj_of(ID_NAME(idp));
		assert( dp != NULL );
		assert( dp == REF_OBJ(ID_REF(idp)) );
	}

	return(REF_OBJ(ID_REF(idp)));
} /* get_id_obj */

#define eval_funcptr(enp) _eval_funcptr(QSP_ARG enp)

static Function_Ptr *_eval_funcptr(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Function_Ptr *fpp=NULL;
	Identifier *idp;

	switch(VN_CODE(enp)){
		case T_FUNCPTR_DECL:
		case T_FUNCPTR:
			idp=id_of(VN_STRING(enp));
			/* BUG chould check that type is funcptr */
			/* BUG chould check that idp is valid */
			assert( idp != NULL );

			fpp = ID_FUNC(idp);
			break;
		default:
			missing_case(enp,"eval_funcptr");
			break;
	}
	return(fpp);
}


#define eval_funcref(enp) _eval_funcref(QSP_ARG  enp)

static Subrt *_eval_funcref(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Subrt *srp;
	Function_Ptr *fpp;

	srp=NULL;
	switch(VN_CODE(enp) ){
		case T_FUNCREF:
			srp=VN_SUBRT(enp);
			break;
		case T_FUNCPTR:
			fpp = eval_funcptr(enp);
			srp = fpp->fp_srp;
			break;
		default:
			missing_case(enp,"eval_funcref");
			break;
	}
	return(srp);
}

/* assign_ptr_arg
 *
 * We call this from assign_subrt_args to assign a value to a pointer variable.
 * It is broken out as a separate subroutine, because we also need do this when we are
 * performing calltime resolution.
 *
 * curr_cpp should be the context of the subrutine which is about to be called,
 * and prev_cpp is the context from which we are calling (used to look up the arg vals).
 */

static int _assign_ptr_arg(QSP_ARG_DECL Vec_Expr_Node *arg_enp,Vec_Expr_Node *val_enp, Context_Pair *curr_cpp,Context_Pair *prev_cpp)
{
	Identifier *idp, *src_idp;

	/* we want this object to be equivalenced to the calling obj */

	pop_subrt_cpair(curr_cpp,SR_NAME(curr_srp));
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"assign_ptr_arg:  current contexts %s, %s popped",CTX_NAME(CP_ID_CTX(curr_cpp)),
CTX_NAME(CP_OBJ_CTX(curr_cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( prev_cpp != NULL ){
		push_cpair(prev_cpp);

#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"assign_ptr_arg:  previous contexts %s, %s pushed",CTX_NAME(CP_ID_CTX(prev_cpp)),
CTX_NAME(CP_OBJ_CTX(prev_cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	}

	src_idp = get_arg_ptr(val_enp);		/* what if the val_enp is a string?? */

	if( prev_cpp != NULL ){
		POP_ID_CONTEXT;
		pop_dobj_context();
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"assign_ptr_arg:  previous contexts %s, %s popped",CTX_NAME(CP_ID_CTX(prev_cpp)),
CTX_NAME(CP_OBJ_CTX(prev_cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	}

	push_cpair(curr_cpp);

#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"assign_ptr_arg:  current contexts %s, %s pushed",CTX_NAME(CP_ID_CTX(curr_cpp)),
CTX_NAME(CP_OBJ_CTX(curr_cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( src_idp == NULL ){
		warn("Missing source object!?");
		return -1;
	}

	idp = get_id(VN_STRING(arg_enp));
	if( idp==NULL ) return -1;


	/* assign_ptr_arg */

	switch(ID_TYPE(idp)){
		case ID_POINTER:
			if( IS_OBJ_REF(src_idp) ){
				assign_pointer(ID_PTR(idp), ID_REF(src_idp) );
				/* propagate_shape? */
				return 0;
			} else if( IS_POINTER(src_idp) ){
				assign_pointer(ID_PTR(idp), PTR_REF(ID_PTR(src_idp)) );
				/* propagate_shape? */
				return 0;
			} else if( IS_STRING_ID(src_idp) ){
				assign_pointer(ID_PTR(idp),ID_REF(src_idp));
				return 0;
			} else {
				node_error(val_enp);
				sprintf(ERROR_STRING,"argval %s is not a reference or a pointer!?",
					ID_NAME(src_idp));
				warn(ERROR_STRING);
				return -1;
			}
			/* NOTREACHED */
			return -1;
		case ID_STRING:
			if( ! IS_STRING_ID(src_idp) ){
				node_error(val_enp);
				sprintf(ERROR_STRING,"argval %s is not a string!?",
					ID_NAME(idp));
				warn(ERROR_STRING);
				return -1;
			}
			assert( sb_buffer(REF_SBUF(ID_REF(src_idp))) != NULL );

			copy_string(REF_SBUF(ID_REF(idp)),sb_buffer(REF_SBUF(ID_REF(src_idp))));
			/* BUG need to set string set flag */
			return 0;
		default:
			warn("unhandled case in assign_ptr_args");
			return -1;
	}
	/* NOTREACHED */
	return -1;
} /* assign_ptr_arg */

static void constant_bitmap(Data_Obj *dp,u_long lval)
{
	u_long *wp;
	int n_words;

	/* BUG here we assume the bitmap is contiguous */

	/* BUG what about padding? */
	n_words = (OBJ_N_TYPE_ELTS(dp) + BITS_PER_BITMAP_WORD - 1 ) / BITS_PER_BITMAP_WORD;
	wp = (u_long *)OBJ_DATA_PTR(dp);
	while(n_words--) *wp++ = lval;
}

#define complement_bitmap(dp) _complement_bitmap(QSP_ARG  dp)

static Data_Obj * _complement_bitmap(QSP_ARG_DECL  Data_Obj *dp)
{
	u_long *wp;
	int n_words;
	static u_long complement_bits=0;

	/* BUG here we assume the bitmap is contiguous */
	assert( IS_CONTIGUOUS(dp) );

	if( complement_bits == 0 ){
		u_short i;
		for(i=0;i<BITS_PER_BITMAP_WORD;i++)
			complement_bits |= NUMBERED_BIT(i);
	}

	/* If this bitmap is not a temporary object, the dup and copy it,
	 * and return the copy!
	 */

	if( !IS_TEMP(dp) ){
		Data_Obj *new_dp;
		const char *s;

		/* BUG possible string overflow */
		new_dp = dup_obj(dp,s=localname());
		assert( new_dp != NULL );

		dp_copy(new_dp,dp);
		dp = new_dp;
	}

	/* BUG what about padding? */
	n_words = (OBJ_N_TYPE_ELTS(dp) + BITS_PER_BITMAP_WORD - 1 ) / BITS_PER_BITMAP_WORD;
	/* BUG check offset (bit0) ... */
	wp = (u_long *)OBJ_DATA_PTR(dp);
	while(n_words--){
		*wp ^= complement_bits;
		wp++;
	}
	return(dp);
}

static void _eval_scalar(QSP_ARG_DECL Scalar_Value *svp, Vec_Expr_Node *enp, Precision *prec_p)
{
	eval_enp = enp;

	/* should we call eval_flt_exp for all??? */

	switch(PREC_CODE(prec_p)&MACH_PREC_MASK){
		case PREC_SP:  svp->u_f = (float) eval_flt_exp(enp); break;
		case PREC_DP:  svp->u_d = eval_flt_exp(enp); break;
		case PREC_BY:  svp->u_b = (char) eval_int_exp(enp); break;
		case PREC_IN:  svp->u_s = (short) eval_int_exp(enp); break;
		case PREC_DI:  svp->u_l = (int32_t) eval_int_exp(enp); break;
		case PREC_LI:  svp->u_ll = (int64_t) eval_int_exp(enp); break;
		case PREC_ULI:  svp->u_ull = (uint64_t) eval_int_exp(enp); break;
		case PREC_UDI:  svp->u_ul = (uint32_t) eval_int_exp(enp); break;
		case PREC_UIN:  svp->u_us = (u_short) eval_int_exp(enp); break;
		case PREC_UBY:  svp->u_ub = (u_char) eval_int_exp(enp); break;
		default:
			assert( AERROR("eval_scalar:  unhandled machine precision") );
			break;
	}
}

#define create_bitmap( src_dsp, pdp ) _create_bitmap( QSP_ARG  src_dsp, pdp )

static Data_Obj *_create_bitmap( QSP_ARG_DECL  Dimension_Set *src_dsp, Platform_Device *pdp )
{
	Dimension_Set ds1, *dsp=(&ds1);
	Data_Obj *bmdp;
//	int i;

	// Why do we copy the dimensions here?
	// does make_obj change them because it is a bitmap?

	COPY_DIMS(dsp,src_dsp);

	/* BUG? the bitmap code in veclib assumes that all the bits
	 * run into one another (i.e., no padding of rows
	 * as is done here.
	 *
	 * Perhaps this is ok, it just wastes memory...
	 */

	bmdp = make_local_dobj(dsp,prec_for_code(PREC_BIT), pdp);
// This seems to be leaked!?
	return(bmdp);
}

#define dup_bitmap(dp) _dup_bitmap(QSP_ARG  dp)

static Data_Obj *_dup_bitmap(QSP_ARG_DECL  Data_Obj *dp)
{
	Data_Obj *new_dp;

	assert( ! UNKNOWN_SHAPE(OBJ_SHAPE(dp)) );

	new_dp = create_bitmap(OBJ_TYPE_DIMS(dp), OBJ_PFDEV(dp) ) ;
	return new_dp;
}

/* vs_bitmap:  vsm_lt etc. */

#define vs_bitmap(dst_dp, dp,svp,code) _vs_bitmap(QSP_ARG  dst_dp, dp,svp,code)

static Data_Obj * _vs_bitmap(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *dp,Scalar_Value *svp,Vec_Func_Code code)
{
	Data_Obj *bmdp;
	Vec_Obj_Args oa1;
#ifdef FOOBAR
	static Data_Obj *vsbm_sclr_dp=NULL;
#endif // FOOBAR
	int status;

	assert( code == FVSMLT || code == FVSMGT || code == FVSMLE ||
	        code == FVSMGE || code == FVSMNE || code == FVSMEQ );

	if( dst_dp == NULL ){
		bmdp = dup_bitmap(dp);
		assert( bmdp != NULL );
	}
	else
		bmdp = dst_dp;

	setvarg2(&oa1,bmdp,dp);	// dbm is the same as dest...

	SET_OA_SVAL(&oa1,0, svp );

	status = perf_vfunc(code,&oa1);

	if( status )
		bmdp=NULL;

	return(bmdp);
	/* BUG? when do we delete the bitmap??? */

} /* end vs_bitmap() */

/* Like dup_bitmap, but we need to use this version w/ vv_bitmap because
 * the two operands might have different shape (outer op).
 * Can we use get_mating_shape?
 */

#define dup_bitmap2(dp1, dp2) _dup_bitmap2(QSP_ARG  dp1, dp2)

static Data_Obj *_dup_bitmap2(QSP_ARG_DECL  Data_Obj *dp1, Data_Obj *dp2)
{
	Shape_Info *shpp;
	Data_Obj *dp;

	shpp = product_shape(OBJ_SHAPE(dp1),OBJ_SHAPE(dp2));
	if( shpp == NULL ) return(NULL);

	dp = create_bitmap(SHP_TYPE_DIMS(shpp), OBJ_PFDEV(dp1) );
	return dp;
}


#define vv_bitmap(dst_dp,dp1,dp2,code) _vv_bitmap(QSP_ARG  dst_dp,dp1,dp2,code)

static Data_Obj * _vv_bitmap(QSP_ARG_DECL  Data_Obj *dst_dp,Data_Obj *dp1,Data_Obj *dp2,Vec_Func_Code code)
{
	Data_Obj *bmdp;
	Vec_Obj_Args oa1;
	int status;

	assert( code == FVVMLT || code == FVVMGT || code == FVVMLE ||
	        code == FVVMGE || code == FVVMNE || code == FVVMEQ );

	if( dst_dp != NULL )
		bmdp = dst_dp;
	else {
		bmdp = dup_bitmap2(dp1,dp2);	/* might be an outer op */
	}

	setvarg3(&oa1,bmdp,dp1,dp2);
	status = perf_vfunc(code,&oa1);

	if( status < 0 )
		bmdp=NULL;

	return(bmdp);
}

static Data_Obj *_eval_bitmap(QSP_ARG_DECL Data_Obj *dst_dp, Vec_Expr_Node *enp)
{
	Data_Obj *bm_dp1,*bm_dp2,*dp,*dp2;
	long ival;

	eval_enp = enp;

	// if dst_dp is non-null, then we return a new object, otherwise we use dst_dp

	switch( VN_CODE(enp) ){
		/* ALL_OBJREF_CASES??? */
		case T_STATIC_OBJ:		/* eval_bitmap */
		case T_DYN_OBJ:			/* eval_bitmap */
//fprintf(stderr,"eval_bitmap object BEGIN, dst_dp = 0x%lx\n",(long) dst_dp);
			dp = eval_obj_ref(enp);
			return(dp);
			break;

		case T_BOOL_AND:
			if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) ){
//fprintf(stderr,"eval_bitmap bool_and case 1  first child is a scalar\n");
				ival = eval_int_exp(VN_CHILD(enp,0));
				bm_dp1 = eval_bitmap(dst_dp,VN_CHILD(enp,1));
//fprintf(stderr,"eval_bitmap bool_and case 1  back from recursive call to eval_bitmap...\n");
				if( !ival )
					constant_bitmap(bm_dp1,0L);
//fprintf(stderr,"eval_bitmap bool_and DONE #1, will return 0x%lx\n",(long) bm_dp1);
				return(bm_dp1);
			} else if( SCALAR_SHAPE( VN_SHAPE(VN_CHILD(enp,1)) ) ){
//fprintf(stderr,"eval_bitmap bool_and case 2  second child is a scalar\n");
				ival = eval_int_exp(VN_CHILD(enp,1));
				bm_dp1 = eval_bitmap(dst_dp,VN_CHILD(enp,0));
//fprintf(stderr,"eval_bitmap bool_and case 2  back from recursive call to eval_bitmap...\n");
				if( !ival )
					constant_bitmap(bm_dp1,0L);
//fprintf(stderr,"eval_bitmap bool_and DONE #2, will return 0x%lx\n",(long) bm_dp1);
				return(bm_dp1);
			} else {
//fprintf(stderr,"eval_bitmap bool_and case 3  neither child is a scalar\n");
				bm_dp1 = eval_bitmap(dst_dp,VN_CHILD(enp,0));
//fprintf(stderr,"eval_bitmap bool_and case 2  back from first recursive call to eval_bitmap...\n");
				bm_dp2 = eval_bitmap(NULL,VN_CHILD(enp,1));
				if( do_vvfunc(bm_dp1,bm_dp1,bm_dp2,FVAND) < 0 ){
					node_error(enp);
					warn("Error evaluating bitmap");
					return(NULL);
				}
//fprintf(stderr,"eval_bitmap bool_and DONE #3, will return 0x%lx\n",(long) bm_dp1);
				return(bm_dp1);
			}
			break;
		case T_BOOL_OR:
//fprintf(stderr,"eval_bitmap bool_or BEGIN, dst_dp = 0x%lx\n",(long) dst_dp);
			if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) ){
				ival = eval_int_exp(VN_CHILD(enp,0));
				bm_dp1 = eval_bitmap(dst_dp,VN_CHILD(enp,1));
				if( ival )
					constant_bitmap(bm_dp1,0xffffffff);
				return(bm_dp1);
			} else if( SCALAR_SHAPE( VN_SHAPE(VN_CHILD(enp,1)) ) ){
				ival = eval_int_exp(VN_CHILD(enp,1));
				bm_dp1 = eval_bitmap(dst_dp,VN_CHILD(enp,0));
				if( ival )
					constant_bitmap(bm_dp1,0xffffffff);
				return(bm_dp1);
			} else {
				bm_dp1 = eval_bitmap(dst_dp,VN_CHILD(enp,0));
				bm_dp2 = eval_bitmap(NULL,VN_CHILD(enp,1));
				if( do_vvfunc(bm_dp1,bm_dp1,bm_dp2,FVOR) < 0 ){
					node_error(enp);
					warn("Error evaluating bitmap");
					return(NULL);
				}
				return(bm_dp1);
			}
			break;
		case T_BOOL_XOR:
//fprintf(stderr,"eval_bitmap bool_xor BEGIN, dst_dp = 0x%lx\n",(long) dst_dp);
			if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) ){
				ival = eval_int_exp(VN_CHILD(enp,0));
				bm_dp1 = eval_bitmap(dst_dp,VN_CHILD(enp,1));
				if( ival ){
					bm_dp1 = complement_bitmap(bm_dp1);
				}
				return(bm_dp1);
			} else if( SCALAR_SHAPE( VN_SHAPE(VN_CHILD(enp,1)) ) ){
				ival = eval_int_exp(VN_CHILD(enp,1));
				bm_dp1 = eval_bitmap(dst_dp,VN_CHILD(enp,0));
				if( ival ){
					bm_dp1 = complement_bitmap(bm_dp1);
				}
				return(bm_dp1);
			} else {
				bm_dp1 = eval_bitmap(dst_dp,VN_CHILD(enp,0));
				bm_dp2 = eval_bitmap(NULL,VN_CHILD(enp,1));
				if( do_vvfunc(bm_dp1,bm_dp1,bm_dp2,FVXOR) < 0 ){
					node_error(enp);
					warn("Error evaluating bitmap");
					return(NULL);
				}
				return(bm_dp1);
			}
			break;
		case T_BOOL_NOT:
//fprintf(stderr,"eval_bitmap bool_not BEGIN, dst_dp = 0x%lx\n",(long) dst_dp);
			bm_dp1 = eval_bitmap(dst_dp,VN_CHILD(enp,0));
			bm_dp1 = complement_bitmap(bm_dp1);
			return(bm_dp1);
			break;

		ALL_NUMERIC_COMPARISON_CASES			/* eval_bitmap */

			assert( ! SCALAR_SHAPE( VN_SHAPE(VN_CHILD(enp,0)) ) );

			if( SCALAR_SHAPE( VN_SHAPE(VN_CHILD(enp,1)) ) ){
				Scalar_Value sval;
				dp = eval_obj_exp(VN_CHILD(enp,0),NULL);
				ASSERT_NODE_DATA_TYPE(enp,ND_FUNC)
				assert( dp != NULL );
				eval_scalar(&sval,VN_CHILD(enp,1),OBJ_PREC_PTR(dp));
				bm_dp1 = vs_bitmap(dst_dp,dp,&sval,VN_BM_CODE(enp));

if( bm_dp1 == NULL ){
node_error(enp);
sprintf(ERROR_STRING,"bad vs_bitmap, %s",node_desc(enp));
error1(ERROR_STRING);
IOS_RETURN_VAL(NULL)
}
			} else {
				/* both vectors */
				dp = eval_obj_exp(VN_CHILD(enp,0),NULL);
				dp2 = eval_obj_exp(VN_CHILD(enp,1),NULL);
				bm_dp1 = vv_bitmap(dst_dp,dp,dp2,VN_BM_CODE(enp));
			}
//fprintf(stderr,"eval_bitmap numeric_comparison DONE, will return 0x%lx\n",(long) bm_dp1);
			return(bm_dp1);
			break;

		default:
			missing_case(enp,"eval_bitmap");
			break;
	}
	return(NULL);
} /* end eval_bitmap() */

void _easy_ramp2d(QSP_ARG_DECL  Data_Obj *dst_dp,double start,double dx,double dy)
{
	Vec_Obj_Args oa1;
	Scalar_Value sv1, sv2, sv3;

	cast_dbl_to_scalar_value(&sv1,OBJ_PREC_PTR(dst_dp),(double)start);
	cast_dbl_to_scalar_value(&sv2,OBJ_PREC_PTR(dst_dp),(double)dx);
	cast_dbl_to_scalar_value(&sv3,OBJ_PREC_PTR(dst_dp),(double)dy);

	clear_obj_args(&oa1);
	//SET_OA_SRC_OBJ(&oa1,0, dst_dp);			// why set this???
	SET_OA_DEST(&oa1, dst_dp);
	SET_OA_SVAL(&oa1,0, &sv1);
	SET_OA_SVAL(&oa1,1, &sv2);
	SET_OA_SVAL(&oa1,2, &sv3);

	set_obj_arg_flags(&oa1);

	platform_dispatch_by_code( FVRAMP2D, &oa1 );
}

static void assign_element(QSP_ARG_DECL Data_Obj *dp,dimension_t ri,dimension_t ci,Vec_Expr_Node *enp)
{
	double *dbl_p,d;

	//SET_OBJ_FLAG_BITS(dp, DT_ASSIGNED);
	note_assignment(dp);

	assert( OBJ_PREC(dp) == PREC_DP );

	dbl_p = (double *)OBJ_DATA_PTR(dp);
	/* assign_element uses matlab indexing */
	dbl_p += (ri-1) * OBJ_ROW_INC(dp);
	dbl_p += (ci-1) * OBJ_PXL_INC(dp);
	d = eval_flt_exp(enp);
	*dbl_p = d;
}

/* assign_row - matlab support */

static void assign_row(QSP_ARG_DECL Data_Obj *dp,dimension_t row_index,Vec_Expr_Node *enp)
{
	dimension_t j;
	Data_Obj *src_dp;

	switch(VN_CODE(enp)){
		case T_ROW:		/* really a list of elements */
			j=SHP_COLS(VN_SHAPE(enp));
			ASSIGN_ELEMENT(dp,row_index,j,VN_CHILD(enp,1));
			ASSIGN_ROW(dp,row_index,VN_CHILD(enp,0));
			break;
		case T_TIMES:
		case T_UMINUS:
		case T_LIT_DBL:
			ASSIGN_ELEMENT(dp,row_index,1,enp);
			break;
		case T_STATIC_OBJ:	/* assign_row */
			src_dp = VN_OBJ(enp);
			goto assign_row_from_dp;

		case T_DYN_OBJ:		/* assign_row */
			src_dp = dobj_of(VN_STRING(enp));
			/* fall thru */
assign_row_from_dp:
			for(j=0;j<OBJ_COLS(src_dp);j++){
				double *dbl_p1,*dbl_p2;
				/* BUG we need to get the value in a general way */
				dbl_p1 = (double *)OBJ_DATA_PTR(dp);
				dbl_p1 += (row_index-1) * OBJ_ROW_INC(dp);
				dbl_p1 += j * OBJ_PXL_INC(dp);
				dbl_p2 = (double *)OBJ_DATA_PTR(src_dp);
				dbl_p2 += j * OBJ_PXL_INC(src_dp);
				*dbl_p1 = *dbl_p2;
			}
			//SET_OBJ_FLAG_BITS(dp, DT_ASSIGNED);
			note_assignment(dp);
			break;
		default:
			missing_case(enp,"assign_row");
			break;
	}
}

/* Like dp_convert(), but if destination is complex then do the right thing.
 */

#define convert_any_type(dst_dp, dp) _convert_any_type(QSP_ARG  dst_dp, dp)

static int _convert_any_type(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *dp)
{
	Data_Obj *tmp_dp;


	// BUG need to update for quaternion case!
	if( IS_REAL(dst_dp) ){
		if( IS_REAL(dp) ){
			dp_convert(dst_dp,dp);
		} else {
			sprintf(ERROR_STRING,
		"convert_any_type:  can't convert complex/quaternion object %s to real object %s",
				OBJ_NAME(dp),OBJ_NAME(dst_dp));
			warn(ERROR_STRING);
			return -1;
		}
	} else if( IS_COMPLEX(dst_dp) ){
		if( IS_REAL(dp) ){
			tmp_dp = c_subscript(dst_dp,0);
			dp_convert(tmp_dp,dp);
			// BUG should set imaginary part to 0!
			tmp_dp = c_subscript(dst_dp,1);
			return zero_dp(tmp_dp);
		} else if( IS_COMPLEX(dp) ){
			dp_convert(dst_dp,dp);
		} else {
			sprintf(ERROR_STRING,
		"convert_any_type:  unhandled type combination, will not convert %s to %s",
				OBJ_NAME(dp),OBJ_NAME(dst_dp));
			warn(ERROR_STRING);
			return -1;
		}
	} else {
		sprintf(ERROR_STRING,
		"convert_any_type:  unhandled destination type, will not convert %s to %s",
			OBJ_NAME(dp),OBJ_NAME(dst_dp));
		warn(ERROR_STRING);
		return -1;
	}

	return 0;
}

/* We may need to treat this differently for eval_obj_exp and eval_obj_assignment...
 * eval_obj_exp doesn't use dst_dp unless it has to, and dst_dp can be null...
 * Here we assume that the compilation process will have removed any
 * unnecessary typecasts, so we assume that dst_dp is needed, and if it is
 * null we will create it.
 *
 * But we have a problem deciding when to release the local objects!  For simple
 * statements we have been calling delete_local_objs() at the top of eval_work_tree,
 * but this is problematic when the expression involves a subroutine call...
 */

static Data_Obj *_eval_typecast(QSP_ARG_DECL Vec_Expr_Node *enp, Data_Obj *dst_dp)
{
	Data_Obj *dp, *tmp_dp;

	assert( VN_SHAPE(VN_CHILD(enp,0)) != NULL );

	// Can we assert that this is not null???
	if( dst_dp != NULL ){
		//assert( ! UNKNOWN_SHAPE(OBJ_SHAPE(dst_dp)) );
		// This can happen with a declaration error...
		if( UNKNOWN_SHAPE(OBJ_SHAPE(dst_dp)) )
			return NULL;

		assert( OBJ_PREC(dst_dp) == VN_PREC(enp) );
	}

	/* It is not an error for the typecast to match the LHS -
	 * in fact it should!  compile_node may insert a typecast
	 * node to effect type conversion.
	 */

//#ifdef CAUTIOUS
//	if( dst_dp != NULL && OBJ_PREC(dst_dp) != VN_PREC(enp) /* same as VN_INTVAL(enp) */ ){
//		node_error(enp);
//		sprintf(ERROR_STRING,
//	"CAUTIOUS:  eval_typecast:  %s precision %s does not match target %s precision %s",
//			node_desc(enp),NAME_FOR_PREC_CODE(VN_PREC(enp)),OBJ_NAME(dst_dp),OBJ_PREC_NAME(dst_dp));
//		warn(ERROR_STRING);
//		advise("ignoring typecast");
//		eval_obj_assignment(dst_dp,VN_CHILD(enp,0));
//		return(dst_dp);
//	}
//#endif /* CAUTIOUS */

	if( VN_INTVAL(enp) == SHP_PREC(VN_SHAPE(VN_CHILD(enp,0))) ){
		/* the object already has the cast precision */
		node_error(enp);
		warn("typecast redundant w/ rhs");
		eval_obj_assignment(dst_dp,VN_CHILD(enp,0));
		return(dst_dp);
	}

	/* If the child node is an object, we simply do a conversion into the
	 * destination.  If it's an operator, we have to make a temporary object
	 * to hold the result, and then convert.
	 */

	switch(VN_CODE(VN_CHILD(enp,0))){
		ALL_OBJREF_CASES
			/* dp=eval_obj_ref(VN_CHILD(enp,0)); */
			dp=eval_obj_exp(VN_CHILD(enp,0),NULL);
			if( dp != NULL ){
				if( dst_dp == NULL ){
					dst_dp=make_local_dobj(
						SHP_TYPE_DIMS(VN_SHAPE(VN_CHILD(enp,0))),
						VN_PREC_PTR(enp),
						OBJ_PFDEV(dp));
				}
				if( convert_any_type(dst_dp,dp) < 0 ){
					node_error(enp);
					warn("Error performing conversion");
				}
			} else return(NULL);
			break;

		/*
		ALL_VECTOR_SCALAR_CASES
		ALL_VECTOR_VECTOR_CASES
		*/
		ALL_CONDASS_CASES
		case T_CALL_NATIVE:
		case T_VS_FUNC:
		case T_VV_FUNC:
		case T_MATH1_VFN:
		case T_MATH0_VFN:
		case T_RDFT:
		case T_RIDFT:
		case T_SUM:
		case T_CALLFUNC:
			goto handle_it;

		default:			// eval_typecast
			missing_case(VN_CHILD(enp,0),"eval_typecast");
			/* missing_case calls dump_tree?? */
			dump_tree(enp);

handle_it:
			/* We have been requested to convert
			 * to a different precision
			 */

			tmp_dp=make_local_dobj( SHP_TYPE_DIMS(VN_SHAPE(VN_CHILD(enp,0))),
					SHP_PREC_PTR(VN_SHAPE(VN_CHILD(enp,0))),
					NULL );

			eval_obj_assignment(tmp_dp,VN_CHILD(enp,0));

			if( dst_dp == NULL ){
				dst_dp=make_local_dobj( SHP_TYPE_DIMS(VN_SHAPE(VN_CHILD(enp,0))),
					VN_PREC_PTR(enp), NULL );
			}

			if( convert_any_type(dst_dp,tmp_dp) < 0 ){
				node_error(enp);
				warn("error performing conversion");
			}
			delvec(tmp_dp);
			break;
	}
	return(dst_dp);
} // eval_typecast


/* Call assign_subrt_args after the arguments have been declared...
 * This routine loops through and copies the values in.
 *
 * It is assumesd that the shapes match.
 *
 * assign_subrt_args is called with the context of the new subroutine, but we pass it
 * the previous context (which may be needed to look up the argument values).
 *
 * We may call this during the resolution process, in which case the SCANNING flag should
 * be set.  In this case, we don't want to actually copy any data.  The main point of
 * going through the motions is to do pointer assignments (to get shape information).
 */

static int _assign_subrt_args(QSP_ARG_DECL Subrt *srp,Vec_Expr_Node *arg_enp,Vec_Expr_Node *val_enp, Context_Pair *prev_cpp)
{
	int stat;
	Data_Obj *dp;
	Context_Pair *_curr_cpp;
	Function_Ptr *fpp;

	INIT_CPAIR_PTR(_curr_cpp);

	if( arg_enp==NULL ) return 0;

	switch(VN_CODE(arg_enp)){
		case T_DECL_STAT:
			/* en_decl_prec is the type (float,short,etc) */
			stat=assign_subrt_args(srp,VN_CHILD(arg_enp,0),val_enp,prev_cpp);
			return(stat);

		case T_DECL_STAT_LIST:
			/* descend the arg tree */
			/* VN_CODE(val_enp) should be T_ARGLIST */
			stat=assign_subrt_args(srp,VN_CHILD(arg_enp,0),VN_CHILD(val_enp,0),prev_cpp);
			if( stat < 0 ) return(stat);

			stat=assign_subrt_args(srp,VN_CHILD(arg_enp,1),VN_CHILD(val_enp,1),prev_cpp);
			return(stat);

		case T_FUNCPTR_DECL:		/* assign_subrt_args */
			/* we evaluate the argument */

			pop_subrt_cpair(_curr_cpp,SR_NAME(curr_srp));

			if( prev_cpp != NULL ){
				push_cpair(prev_cpp);
			}

			srp = eval_funcref(val_enp);

			if( prev_cpp != NULL ){
				POP_CPAIR;
			}

			/* Now we switch contexts back to the called subrt */

			push_cpair(_curr_cpp);

			/* the argument is a function ptr */
			fpp = eval_funcptr(arg_enp);

			if( srp == NULL ) {
				warn("assign_subrt_args:  error evaluating function ref");
				return -1;
			}

			if( fpp == NULL ){
				warn("assign_subrt_args:  missing function pointer");
				return -1;
			}

			fpp->fp_srp = srp;
			return 0;

		case T_PTR_DECL:		/* assign_subrt_args */
			return( assign_ptr_arg(arg_enp,val_enp,_curr_cpp,prev_cpp) );


		case T_SCAL_DECL:		/* assign_subrt_args */
			{
			Identifier *idp;
			idp = get_id(VN_STRING(arg_enp));
			assert(idp!=NULL);
			assert(ID_SHAPE(idp)!=NULL);
			assert(ID_PREC_PTR(idp)!=NULL);
			assign_scalar_id(idp, val_enp);
			}
			return 0;
			break;

		case T_VEC_DECL:
		case T_IMG_DECL:
		case T_SEQ_DECL:
		case T_CSCAL_DECL:		/* assign_subrt_args */
		case T_CVEC_DECL:
		case T_CIMG_DECL:
		case T_CSEQ_DECL:

			/* Don't copy any data if we're only scanning */

			if( IS_SCANNING(srp) ){
				/* break; */
				return 0;
			}

			dp = get_id_obj(VN_STRING(arg_enp),arg_enp);

			if( dp == NULL ){
sprintf(ERROR_STRING,"assign_subrt_args:  missing object %s",VN_STRING(arg_enp));
warn(ERROR_STRING);
				return -1;
			}

			assert( ! UNKNOWN_SHAPE(OBJ_SHAPE(dp)) );

			/* Tricky point:  we need to pop the subroutine context
			 * here, in case val_enp uses names which are also
			 * some of the new subrt arguments...  if there
			 * are name overlaps, we want to be sure we use
			 * the outer ones for the assignment value!
			 */

			pop_subrt_cpair(_curr_cpp,SR_NAME(curr_srp));

			if( prev_cpp != NULL ){

				push_cpair(prev_cpp);

			}

			eval_obj_assignment(dp, val_enp);

			if( prev_cpp != NULL ){
				Item_Context *icp;
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"assign_subrt_args T_IMG_DECL:  previous contexts %s, %s popped",CTX_NAME(CP_ID_CTX(prev_cpp)),
CTX_NAME(CP_OBJ_CTX(prev_cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
				/*icp=*/ POP_ID_CONTEXT;
				icp=pop_dobj_context();
				assert( icp == CP_OBJ_CTX(prev_cpp) );
			}

			/* restore it */
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"assign_subrt_args T_IMG_DECL:  pushing current context %s",prev_cpp==NULL?
	"(null previous context)":CTX_NAME(CP_OBJ_CTX(prev_cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			push_cpair(_curr_cpp);

			return 0;

		default:
			missing_case(arg_enp,"assign_subrt_args");
			break;

	}
	warn("assign_subrt_args:  shouldn't reach this point");
	return -1;
} /* end assign_subrt_args() */

// BUG?  here we store the arg vals and the call node in the subroutine struct itself.
// Better to have a call struct so we can support multi-threading...

Subrt *_runnable_subrt(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Subrt *srp;

	switch(VN_CODE(enp)){
		case T_CALLFUNC:
			srp=VN_SUBRT(enp);
			break;
		case T_INDIR_CALL:
			srp = eval_funcref(VN_CHILD(enp,0));
			assert( srp!=NULL );
			break;
		default:
			missing_case(enp,"runnable_subrt");
			return(NULL);
	}


	if( SR_BODY(srp) == NULL ){
		node_error(enp);
		sprintf(ERROR_STRING,"subroutine %s has not been defined!?",SR_NAME(srp));
		warn(ERROR_STRING);
		return(NULL);
	}

	// We don't allocate the dynamic object until we know there are no errors
	return srp;
}

/* exec_subrt is usually called on a T_CALLFUNC or T_INDIR_CALL node */

void _exec_subrt(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj *dst_dp)
{
	Subrt *srp;

	srp = runnable_subrt(enp);

	if( srp != NULL ){
		run_subrt(srp,dst_dp,enp);
	} else {
		sprintf(ERROR_STRING,"subroutine is not runnable!?");
		warn(ERROR_STRING);
		dump_tree(enp);
	}
}

Identifier *_make_named_reference(QSP_ARG_DECL  const char *name)
{
	Identifier *idp;

	idp = id_of(name);
	if( idp != NULL ) return(idp);

//sprintf(ERROR_STRING,"make_named_reference:  creating id %s",name);
//advise(ERROR_STRING);
	idp = new_identifier(name);
	SET_ID_TYPE(idp, ID_OBJ_REF);
	SET_ID_REF(idp, NEW_REFERENCE );
	SET_REF_OBJ(ID_REF(idp), NULL );
	SET_REF_ID(ID_REF(idp), idp );
	SET_REF_TYPE(ID_REF(idp), OBJ_REFERENCE );		/* override if string */
	SET_REF_DECL_VN(ID_REF(idp), NULL );
	return(idp);
}

static void _eval_display_stat(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp;
	const char *s;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_EXPR_LIST:
			eval_display_stat(VN_CHILD(enp,0));
			eval_display_stat(VN_CHILD(enp,1));
			break;
		case T_STR_PTR:
			s = eval_string(enp);
			sprintf(msg_str,"String %s:  \"%s\"",VN_STRING(enp),s);
			prt_msg(msg_str);
			break;
		case T_CURLY_SUBSCR:		/* eval_display_stat */
		case T_SQUARE_SUBSCR:
		case T_DEREFERENCE:
		case T_STATIC_OBJ:		/* eval_display_stat */
		case T_DYN_OBJ:			/* eval_display_stat */
		case T_POINTER:
		case T_SUBVEC:
		case T_SUBSAMP:			/* eval_display_stat */
		case T_CSUBSAMP:
			dp = eval_obj_ref(enp);
			if( dp==NULL ){
				warn("missing info object");
				// An informative message should have
				// been printed before we get here...
				break;
			} else {
				list_dobj(dp);
				pntvec(dp, tell_msgfile() );
			}
			break;
		default:
			missing_case(enp,"eval_display_stat");
			break;
	}
}

static Vec_Expr_Node *_find_case(QSP_ARG_DECL Vec_Expr_Node *enp,long lval)
{
	Vec_Expr_Node *ret_enp;
	long cval;

	switch(VN_CODE(enp)){
		case T_CASE_STAT:	/* case_list stat_list pair */
			if( find_case(VN_CHILD(enp,0),lval) != NULL )
				return(enp);
			else return(NULL);

		case T_CASE_LIST:
			ret_enp=find_case(VN_CHILD(enp,0),lval);
			if( ret_enp == NULL )
				ret_enp=find_case(VN_CHILD(enp,1),lval);
			return(ret_enp);

		case T_CASE:
			cval = eval_int_exp(VN_CHILD(enp,0));
			if( cval == lval ){
				return(VN_CHILD(enp,0));
			} else return(NULL);

		case T_DEFAULT:
			return(enp);

		case T_SWITCH_LIST:	/* list of case_stat's */
			ret_enp=find_case(VN_CHILD(enp,0),lval);
			if( ret_enp == NULL )
				ret_enp=find_case(VN_CHILD(enp,1),lval);
			return(ret_enp);

		default:
			missing_case(enp,"find_case");
			break;
	}
	return(NULL);
}

/* Find the first case of a switch statement.
 * used for goto scanning.
 */

static Vec_Expr_Node *first_case(Vec_Expr_Node *enp)
{
	Vec_Expr_Node *case_enp;

	assert( VN_CODE(enp) == T_SWITCH );

	case_enp = VN_CHILD(enp,1);

	while(VN_CODE(case_enp) == T_SWITCH_LIST )
		case_enp = VN_CHILD(case_enp,0);
	assert( VN_CODE(case_enp) == T_CASE_STAT );

	return(case_enp);
}

/* returns a child LABEL node whose name matches global goto_label,
 * or NULL if not found...
 */

static Vec_Expr_Node *goto_child(Vec_Expr_Node *enp)
{
	int i;

	for(i=0;i<MAX_CHILDREN(enp);i++){
		if( VN_CHILD(enp,i) != NULL ){
			if( VN_CODE(VN_CHILD(enp,i)) == T_LABEL && !strcmp(VN_STRING(VN_CHILD(enp,i)),goto_label) ){
				return(VN_CHILD(enp,i));
			}
			else if( goto_child(VN_CHILD(enp,i)) != NULL )
				return(VN_CHILD(enp,i));
		}
	}
	return(NULL);
}

static Vec_Expr_Node *next_case(Vec_Expr_Node *enp)
{
	if( VN_CODE(VN_PARENT(enp)) == T_SWITCH ){
		return(NULL);
	}

	assert( VN_CODE(VN_PARENT(enp) ) == T_SWITCH_LIST );

keep_looking:
	if( VN_CODE(VN_PARENT(enp)) == T_SWITCH_LIST ){
		if( enp == VN_CHILD(VN_PARENT(enp),0) ){
			/* descend the right branch */
			enp=VN_CHILD(VN_PARENT(enp),1);
			while( VN_CODE(enp) == T_SWITCH_LIST )
				enp=VN_CHILD(enp,0);
			assert( VN_CODE(enp) == T_CASE_STAT );

			return(enp);
		} else {
			/* our case is the right hand child... */
			enp = VN_PARENT(enp);
			goto keep_looking;
		}
	}
	return(NULL);
}

/* Traverse a string list tree, setting the query args starting
 * at index.  Returns the number of leaves.
 *
 * We use a static array to hold the string pointers while we are parsing the tree.
 * When this was first written, the query args were a fixed-sized
 * table q_arg;  But now they are a dynamically allocated array
 * of variable size, renamed q_args.  Normally these are allocated
 * when a macro is invoked and pushed onto the query
 * stack.  Here we are pushing a script function on the query stack.
 */

#define MAX_SCRIPT_ARGS	12
static const char *script_arg_tbl[MAX_SCRIPT_ARGS];	// BUG - not thread-safe!
static int n_stored_script_args=0;

static void store_script_arg( const char *s, int idx )
{
	assert(idx>=0 && idx < MAX_SCRIPT_ARGS);
	assert(n_stored_script_args==idx);

	script_arg_tbl[idx] = savestr(s);
	n_stored_script_args++;
}

static void pass_script_args(Query *qp)
{
	int i;

	for(i=0;i<n_stored_script_args;i++){
		set_query_arg_at_index(qp,i,script_arg_tbl[i]);
	}
}

static void clear_script_args(void)
{
	bzero(script_arg_tbl,MAX_SCRIPT_ARGS*sizeof(char *));
	n_stored_script_args = 0;
}

static void rls_script_args(void)
{
	int i;

	for(i=0;i<n_stored_script_args;i++){
		rls_str(script_arg_tbl[i]);
	}
	clear_script_args();
}

static int _parse_script_args(QSP_ARG_DECL Vec_Expr_Node *enp,int index,int max_args)
{
	Data_Obj *dp;

	if( enp==NULL ) return 0;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_DEREFERENCE:				/* parse_script_args */
			dp = eval_obj_ref(enp);
			if( dp==NULL ){
				node_error(enp);
				warn("missing script arg object");
				return 0;
			} else {
				store_script_arg(OBJ_NAME(dp),index);
				return(1);
			}

		case T_STR_PTR:
			{
			const char *s;
			s=eval_string(enp);
			store_script_arg(s,index);
			}
			return(1);

		case T_POINTER:
			/* do we dereference the pointer??? */
			node_error(enp);
			sprintf(ERROR_STRING,
				"parse_script_args:  not sure whether or not to dereference ptr %s",
				VN_STRING(enp));
			advise(ERROR_STRING);
			/* fall-thru */

		case T_STATIC_OBJ:		/* parse_script_args */
		case T_DYN_OBJ:			/* parse_script_args */
			/* maybe we could check the node shape instead of looking up the object? */
			dp=eval_obj_ref(enp);
			if( IS_SCALAR(dp) ){
				char buf[64];
				format_scalar_obj(buf,64,dp,OBJ_DATA_PTR(dp),NO_PADDING);
				store_script_arg( buf, index );
			} else {
				store_script_arg( OBJ_NAME(dp), index );
			}
			return 1;
			break;

		case T_STRING:
			/* add this string as one of the args */
			store_script_arg( VN_STRING(enp), index );
			return(1);

		case T_PRINT_LIST:
		case T_STRING_LIST:
		case T_MIXED_LIST:
			{
			int n1,n2;
			n1=parse_script_args(VN_CHILD(enp,0),index,max_args);
			n2=parse_script_args(VN_CHILD(enp,1),index+n1,max_args);
			return(n1+n2);
			}

		/* BUG there are more cases that need to go here
		 * in order to handle generic expressions
		 */

		case T_LIT_INT: case T_LIT_DBL:			/* parse_script_args */
		case T_PLUS: case T_MINUS: case T_TIMES: case T_DIVIDE:
			{
			double dval;
			dval=eval_flt_exp(enp);
			sprintf(msg_str,"%g",dval);
			}
			store_script_arg( msg_str, index );
			return(1);

		default:
			assert( AERROR("missing case in parse_script_args") );
			break;
	}
	return 0;
} /* end parse_script_args */

/*
 * When we call a script func, we want to be able to access all objects, not
 * just those in the current subroutine context.  (There are no scope rules
 * for the normal script interpreter.)  The top of the dobj context stack will
 * be the one for the current subroutine; we pop this, then push the "hidden"
 * contexts for the calling subroutines.  We push these from the bottom of the stack
 * to maintain the proper precedence.  There is a BUG in this approach, in that
 * a pointer may point to an object way down the stack which will be masked
 * by an identically named object higher up the stack.  Not a problem in the
 * vectree parser where things are scoped, but potentially a problem when
 * executing script funcs.  This would be a good place to scan for this and
 * issue a warning...
 */

static void set_script_context(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp;
	int i;

	icp = pop_dobj_context();
	assert( icp != NULL );

#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"set_script_context:  current context %s popped",CTX_NAME(icp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	for(i=0;i<n_hidden_contexts;i++){
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"set_script_context:  pushing hidden context %s",CTX_NAME(hidden_context[i]));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		push_dobj_context(hidden_context[i]);
	}
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"set_script_context:  pushing current context %s",CTX_NAME(icp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	push_dobj_context(icp);

	set_global_ctx(SINGLE_QSP_ARG);	/* we do this so any new items created will be global */
}

static void unset_script_context(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp,*top_icp;
	int i;

	unset_global_ctx(SINGLE_QSP_ARG);

	top_icp = pop_dobj_context();

#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"unset_script_context:  top context %s popped",CTX_NAME(top_icp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	assert( top_icp != NULL );

	for(i=0;i<n_hidden_contexts;i++){
		icp = pop_dobj_context();
		assert( icp == hidden_context[n_hidden_contexts-(1+i)] );

#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"unset_script_context:  hidden context %s popped",CTX_NAME(icp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	}

	push_dobj_context(top_icp);
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"unset_script_context:  top context %s pushed",CTX_NAME(top_icp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
}

static void inc_obj(Data_Obj *dp)
{
	Scalar_Value *svp;

	svp = (Scalar_Value *)OBJ_DATA_PTR(dp);
	switch( OBJ_MACH_PREC(dp) ){
		case PREC_BY:	svp->u_b  += 1;		break;
		case PREC_IN:	svp->u_s  += 1;		break;
		case PREC_DI:	svp->u_l  += 1;		break;
		case PREC_LI:	svp->u_ll  += 1;	break;
		case PREC_UBY:	svp->u_ub += 1;		break;
		case PREC_UIN:	svp->u_us += 1;		break;
		case PREC_UDI:	svp->u_ul += 1;		break;
		case PREC_ULI:	svp->u_ull += 1;	break;
		case PREC_SP:	svp->u_f  += 1;		break;
		case PREC_DP:	svp->u_d  += 1;		break;

		case PREC_NONE:
		case PREC_INVALID:
		case N_MACHINE_PRECS:
			break;
	}
}

static void dec_obj(Data_Obj *dp)
{
	Scalar_Value *svp;

	svp = (Scalar_Value *)OBJ_DATA_PTR(dp);
	switch( OBJ_MACH_PREC(dp) ){
		case PREC_BY:	svp->u_b -= 1;		break;
		case PREC_IN:	svp->u_s -= 1;		break;
		case PREC_DI:	svp->u_l -= 1;		break;
		case PREC_LI:	svp->u_ll -= 1;		break;
		case PREC_SP:	svp->u_f -= 1;		break;
		case PREC_DP:	svp->u_d -= 1;		break;
		case PREC_UBY:	svp->u_ub -= 1;		break;
		case PREC_UIN:	svp->u_us -= 1;		break;
		case PREC_UDI:	svp->u_ul -= 1;		break;
		case PREC_ULI:	svp->u_ull -= 1;	break;

		case PREC_NONE:
		case PREC_INVALID:
		case N_MACHINE_PRECS:
			break;
	}
}

#define STRING_FORMAT	"%s"
#define OBJECT_FORMAT	"%f"

static const char *_eval_mixed_list(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	char *s;
	const char *s1,*s2;
	char buf[128];
	int n;
	Data_Obj *dp;
	Identifier *idp;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_NAME_FUNC:
			if( dumping ) return(STRING_FORMAT);

			dp=eval_obj_ref(VN_CHILD(enp,0));
			assert( dp != NULL );
			return(OBJ_NAME(dp));

		case T_STRING:
			return( VN_STRING(enp) );
		case T_STR_PTR:
			if( dumping ) return(STRING_FORMAT);

			idp = eval_ptr_expr(enp,EXPECT_PTR_SET);
			if( idp==NULL ) return("");
			assert( IS_STRING_ID(idp) );

			if( sb_buffer(REF_SBUF(ID_REF(idp))) == NULL ){
				node_error(enp);
				sprintf(ERROR_STRING,"string pointer %s not set",ID_NAME(idp));
				advise(ERROR_STRING);
				break;
			}
			return(sb_buffer(REF_SBUF(ID_REF(idp))));

		case T_SET_STR:
			eval_work_tree(enp,NULL);	/* do the assignment! */
			return( eval_mixed_list(VN_CHILD(enp,0)) );

		case T_STRING_LIST:
		case T_PRINT_LIST:
			s1=eval_mixed_list(VN_CHILD(enp,0));
			s2=eval_mixed_list(VN_CHILD(enp,1));
			n=(int)(strlen(s1)+strlen(s2)+1);
			s=(char *)getbuf(n);
			strcpy(s,s1);
			strcat(s,s2);
			return(s);

		ALL_OBJREF_CASES			/* eval_mixed_list */
			if( dumping ) return(OBJECT_FORMAT);

			/* BUG need all expr nodes here */

			dp = eval_obj_ref(enp);
			if( dp==NULL ) return("(null)");
			if( IS_SCALAR(dp) )
				format_scalar_obj(buf,128,dp,OBJ_DATA_PTR(dp),NO_PADDING);
			else {
				/*
				node_error(enp);
				sprintf(ERROR_STRING,
					"eval_mixed_list:  object %s is not a scalar!?",OBJ_NAME(dp));
				warn(ERROR_STRING);
				return("");
				*/
				strcpy(buf,OBJ_NAME(dp));
			}
			n=(int)strlen(buf)+1;
			s=(char *)getbuf(n);
			strcpy(s,buf);
			/* BUG a memory leak because these strings are never freed!? */
			return(s);
		default:
			missing_case(enp,"eval_mixed_list");
			break;
	}
	return("");
} /* eval_mixed_list */

static void _eval_print_stat(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp;
	Identifier *idp;
	long n;
	double d;
	const char *s;

	eval_enp = enp;

	switch(VN_CODE(enp)){
#ifdef SCALARS_NOT_OBJECTS
		case T_SCALAR_VAR:			/* eval_print_stat */
			idp = get_id(VN_STRING(enp));
			assert(idp!=NULL);
			if( IS_FLOATING_PREC_CODE(PREC_CODE(ID_PREC_PTR(idp))) )
				goto print_float;
			else
				goto print_integer;
			break;
#endif // SCALARS_NOT_OBJECTS
		case T_CALLFUNC:			/* eval_print_stat */
			if( ! SCALAR_SHAPE(VN_SHAPE(enp)) ){
				prt_msg("");
				node_error(enp);
				advise("Can't print a non-scalar function");
				break;
			}
			/* Now we evaluate it... */
			switch(VN_PREC(enp)){
				case PREC_DI:
				case PREC_UDI:
				case PREC_IN:
				case PREC_UIN:
				case PREC_BY:
				case PREC_UBY:
					goto print_integer;
				case PREC_SP:
				case PREC_DP:
					goto print_float;
				default:
					error1("eval_print_stat:  missing CALLFUNC precision");
					IOS_RETURN
			}
			break;

		case T_LIT_DBL:
		/* BUG need all expression nodes here */
		ALL_SCALAR_BINOP_CASES
		case T_MATH0_FN:
		case T_MATH1_FN:
		case T_MATH2_FN:
print_float:
			d = eval_flt_exp(enp);
			sprintf(msg_str,"%g",d);
			prt_msg_frag(msg_str);
			break;
			break;


		case T_NAME_FUNC:
			dp = eval_obj_ref(VN_CHILD(enp,0));
			if( dp == NULL ){
				node_error(enp);
				advise("Missing object");
			} else
				prt_msg_frag(OBJ_NAME(dp));
			break;

		case T_STRING_LIST:
		case T_MIXED_LIST:
		case T_PRINT_LIST:
			eval_print_stat(VN_CHILD(enp,0));
			prt_msg_frag(" ");
			eval_print_stat(VN_CHILD(enp,1));
			break;
		case T_POINTER:
			idp = eval_ptr_expr(enp,EXPECT_PTR_SET);
			assert( IS_POINTER(idp) );
			assert( POINTER_IS_SET(idp) );
			assert( ID_PTR(idp) != NULL );
			assert( PTR_REF(ID_PTR(idp)) != NULL );
			if( IS_OBJECT_REF(PTR_REF(ID_PTR(idp))) ){
				assert( REF_OBJ(PTR_REF(ID_PTR(idp))) != NULL );

				/* what should we print here? */
				/* If the pointer points to a string, then print the string... */
				dp=REF_OBJ(PTR_REF(ID_PTR(idp))) ;
				if( OBJ_PREC(dp) == PREC_CHAR || OBJ_PREC(dp) == PREC_STR )
					prt_msg_frag((char *)OBJ_DATA_PTR(dp));
				else
					prt_msg_frag(ID_NAME(idp));
			} else if( IS_STRING_REF(PTR_REF(ID_PTR(idp))) ){
				prt_msg_frag(sb_buffer(REF_SBUF(PTR_REF(ID_PTR(idp)))));
			}
			  else {
				assert( AERROR("bad reference type") );
			}
			break;

		ALL_OBJREF_CASES
		case T_PREDEC:
		case T_PREINC:			/* eval_print_stat */
		case T_POSTDEC:
		case T_POSTINC:			/* eval_print_stat */
			dp = eval_obj_ref(enp);
			if( dp==NULL ) return;

			if( VN_CODE(enp) == T_PREINC ) inc_obj(dp);
			else if( VN_CODE(enp) == T_PREDEC ) dec_obj(dp);

			if( IS_SCALAR(dp) ){
				format_scalar_obj(msg_str,LLEN,dp,OBJ_DATA_PTR(dp),NO_PADDING);
				prt_msg_frag(msg_str);
			} else {
				/*
				node_error(enp);
				sprintf(ERROR_STRING,
					"eval_print_stat:  object %s is not a scalar!?",OBJ_NAME(dp));
				advise(ERROR_STRING);
				*/
				prt_msg_frag(OBJ_NAME(dp));
			}

			if( VN_CODE(enp) == T_POSTINC ) inc_obj(dp);
			else if( VN_CODE(enp) == T_POSTDEC ) dec_obj(dp);

			break;


		case T_SET_STR:
		case T_STR_PTR:
			s=eval_mixed_list(enp);
			prt_msg_frag(s);
			break;

		case T_STRING:
			prt_msg_frag(VN_STRING(enp));
			break;
		case T_LIT_INT:
		case T_SIZE_FN:		// eval_print_stat
			/* BUG need all expr nodes here */
print_integer:
			n=eval_int_exp(enp);
			sprintf(msg_str,"%ld",n);
			prt_msg_frag(msg_str);
			break;
		default:
			missing_case(enp,"eval_print_stat");
			break;
	}
} /* eval_print_stat */

/* eval_ref_tree - what is this used for??? */

static void _eval_ref_tree(QSP_ARG_DECL Vec_Expr_Node *enp,Identifier *dst_idp)
{
	Identifier *idp;

	switch(VN_CODE(enp)){
		case T_STAT_LIST:
			eval_ref_tree(VN_CHILD(enp,0),dst_idp);
			eval_ref_tree(VN_CHILD(enp,1),dst_idp);
			break;
		case T_ASSIGN:				/* eval_ref_tree */
			if( eval_work_tree(enp,NULL) == 0 )
				warn("CAUTIOUS:  eval_ref_tree:  eval_work_tree returned 0!?");
			break;
		case T_RETURN:	/* return a pointer */
			idp = eval_ptr_expr(VN_CHILD(enp,0),1);
			assert( idp != NULL );
			assert( IS_OBJ_REF(idp) );

			/* now copy over the identifier data */
			SET_PTR_REF(ID_PTR(dst_idp), ID_REF(idp));
			/* BUG? keep flags? */
			/*
			SET_PTR_FLAGS(ID_PTR(dst_idp), PTR_FLAGS(ID_PTR(idp)));
			*/
			break;
		default:
			missing_case(enp,"eval_ref_tree");
			break;
	}
}

/* wrapup_call
 *
 * forget any resolved shapes that are local to this subrt
 * Also pop the subrt context and restore the previous one, if any...
 */

#define wrapup_call(rip) _wrapup_call(QSP_ARG  rip)

static void _wrapup_call(QSP_ARG_DECL  Run_Info *rip)
{
	/* We need to forget both the uk shape arguments
	 * and uk shape automatic variables.
	 */
	forget_resolved_shapes(rip->ri_srp);
	wrapup_context(rip);
}

static void _run_reffunc(QSP_ARG_DECL Subrt *srp, Vec_Expr_Node *call_enp, Identifier *dst_idp)
{
	Run_Info *rip;

	executing=1;
	/* Run-time resolution of unknown shapes */

	rip = setup_subrt_call(srp,call_enp,NULL);
	if( rip == NULL ){
sprintf(ERROR_STRING,"run_reffunc %s:  no return info!?",SR_NAME(srp));
warn(ERROR_STRING);
		return;
	}

	if( rip->ri_arg_stat >= 0 ){
		eval_decl_tree(SR_BODY(srp));
		eval_ref_tree(SR_BODY(srp),dst_idp);
	}

	wrapup_call(rip);
}



/* a function that returns a pointer */

static Identifier * _exec_reffunc(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Identifier *idp;
	char name[LLEN];
	Subrt *srp;

	srp = runnable_subrt(enp);
	if( srp==NULL ) return(NULL);

	sprintf(name,"ref.%s",SR_NAME(srp));

	idp = make_named_reference(name);
	/* BUG set ptr_type?? */

	assert( idp != NULL ) ;

	/* need to check stuff */


	if( srp != NULL )
		run_reffunc(srp,enp,idp);

	return(idp);
}

static void push_hidden_context(Context_Pair *cpp)
{
	if( n_hidden_contexts >= MAX_HIDDEN_CONTEXTS ){
		NERROR1("too many hidden contexts (try increasing MAX_HIDDEN_CONTEXTS)");
		IOS_RETURN
	}
	hidden_context[n_hidden_contexts] = CP_OBJ_CTX(cpp);
	n_hidden_contexts++;
}

static void pop_hidden_context()
{
	assert( n_hidden_contexts > 0 );
	n_hidden_contexts--;
}

/* Pop the context of a previous subroutine.
 * We do this when we call another subroutine, so that
 * automatic variables in the calling subrt will have their
 * scope restricted to the caller.
 * However, we need to use the context when we assign subrt args...
 *
 * We can have a current subroutine even when the context has not been set, so
 * this has to be done carefully...
 */

Context_Pair *pop_previous(SINGLE_QSP_ARG_DECL)
{
	Context_Pair *cpp;

	/* If there is a previous subroutine context, pop it now */
	if( curr_srp == NULL ){
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
advise("pop_previous:  no current subroutine, nothing to pop");
}
#endif /* QUIP_DEBUG */
		cpp = NULL;
	} else {
		/* Before we go through with this, we should make sure that the context really
		 * is installed!
		 */
		cpp = (Context_Pair *)getbuf( sizeof(Context_Pair) );
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"pop_previous %s:  calling pop_subrt_cpair (context)",SR_NAME(curr_srp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		pop_subrt_cpair(cpp,SR_NAME(curr_srp));
		/* we remember this context so we can use it if we call a script func */
		push_hidden_context(cpp);
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"pop_previous:  previous contexts %s, %s popped",
CTX_NAME(CP_ID_CTX(cpp)),
CTX_NAME(CP_OBJ_CTX(cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	}
	return(cpp);
} // pop_previous

/* We call restore_previous when we return from a subroutine call to go back
 * the the original context.
 */

void _restore_previous(QSP_ARG_DECL  Context_Pair *cpp)
{
	pop_hidden_context();
	push_cpair(cpp);
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"restore_previous:  previous contexts %s, %s pushed",
CTX_NAME(CP_ID_CTX(cpp)),
CTX_NAME(CP_OBJ_CTX(cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	givbuf(cpp);
}

static Run_Info *new_rip()
{
	Run_Info *rip;

	rip=(Run_Info *)getbuf( sizeof(Run_Info) );
	rip->ri_prev_cpp = NULL;
	rip->ri_arg_stat = 0;
	rip->ri_srp = NULL;
	rip->ri_old_srp = NULL;
	return(rip);
}

/* setup_subrt_call
 *
 * does the following things:
 *	calls early_calltime_resolve
 *	calls check_args_shapes, exits if there's a mismatch
 *	switches subrt context to the new subrt
 *	evaluates the arg decls
 *	assigns the arg values
 *	returns a run_info struct
 */

Run_Info * _setup_subrt_call(QSP_ARG_DECL Subrt *srp, Vec_Expr_Node *call_enp, Data_Obj *dst_dp)
{
	Run_Info *rip;
	Vec_Expr_Node *args_enp;

	/*
	 * We call calltime resolve to resolve arg shapes and return shapes if we can.
	 * What is the expected context for early_calltime_resolve???
	 */

	early_calltime_resolve(srp,call_enp,dst_dp);

	/* BUG We'd like to pop the context of any calling subrts here, but it is tricky:
	 * We need to have the old context so we can find the arg values...  but we want
	 * to pop the context when we evaluate the arg decls to avoid warnings
	 * about names shadowing other names
	 * (which we only want for objects in the global context).
	 */

	rip = new_rip();
	rip->ri_srp = srp;

	args_enp = VN_CHILD(call_enp,0);
	if( check_arg_shapes(SR_ARG_DECLS(srp),args_enp,call_enp) < 0 )
		goto call_err;

	/* declare the arg variables */

	/* First, pop the context of the previous subroutine and push the new one */
	rip->ri_prev_cpp = pop_previous(SINGLE_QSP_ARG);	/* what does pop_previous() do??? */
	set_subrt_ctx(SR_NAME(srp));

	// We need to be sure that we use the correct platform when we
	// declare any objects that we need here...
	// Scalar objects are created for scalar arguments, that is a lot of overhead!?
	// Maybe we should allow id's to be scalars???

	eval_decl_tree(SR_ARG_DECLS(srp));

	rip->ri_old_srp = curr_srp;
	curr_srp = srp;

	rip->ri_arg_stat = assign_subrt_args(srp,SR_ARG_DECLS(srp),args_enp,rip->ri_prev_cpp);

	return(rip);

call_err:

	/* now we're back , restore the context of the caller , if any */
	if( rip->ri_prev_cpp != NULL ){
		restore_previous(rip->ri_prev_cpp);
	}
	return(NULL);
} // setup_subrt_call

/* wrapup_context
 *
 * called after subroutine execution to restore the context of the caller.
 */

void _wrapup_context(QSP_ARG_DECL  Run_Info *rip)
{

	curr_srp = rip->ri_old_srp;

	/* get rid of the context, restore the context of the caller , if any */

	delete_subrt_ctx(SR_NAME(rip->ri_srp));
	if( rip->ri_prev_cpp != NULL ){
		restore_previous(rip->ri_prev_cpp);
	}
}




// This is the function called from the menu to run a single function...
// We need to push the parser data BEFORE calling this so that we can
// get the args...

void _run_subrt_immed(QSP_ARG_DECL  Subrt *srp, Data_Obj *dst_dp, Vec_Expr_Node *call_enp)
{
	delete_local_objs();	// run_subrt_immed
	run_subrt(srp,dst_dp,call_enp);
}

#ifdef HAVE_ANY_GPU

#define pfdev_for_call(args_enp) _pfdev_for_call(QSP_ARG  args_enp)

static Platform_Device *_pfdev_for_call(QSP_ARG_DECL  Vec_Expr_Node *args_enp)
{
	// Normally, we determine this from the arg tree...
	if( VN_PFDEV( args_enp ) == NULL ){
		// try to figure it out
		update_pfdev_from_children(args_enp);
	}
	if( VN_PFDEV( args_enp ) == NULL ){
		fprintf(stderr,"Arg values do not have platform set!?\n");
		return NULL;
	}

//fprintf(stderr,"Call is targeted for platform device %s\n",PFDEV_NAME( VN_PFDEV( args_enp ) ) );
	return VN_PFDEV( args_enp );
}

#endif // HAVE_ANY_GPU

void _run_subrt(QSP_ARG_DECL Subrt *srp, Data_Obj *dst_dp, Vec_Expr_Node *call_enp)
{
	Run_Info *rip;
	Platform_Device *pdp;
	void * kp;
	Vec_Expr_Node *args_enp;

	executing=1;

	// BUG - we probably don't want to do all the work in setup_subrt_call
	// if we are calling a fused kernel???
	rip = setup_subrt_call(srp, call_enp, dst_dp);
	if( rip == NULL ){
		return;
	}

	args_enp = VN_CHILD(call_enp,0);

#ifdef HAVE_ANY_GPU
	// Has this subroutine been "fused" (compiled)?
	// Need to determine the platform...
	if( args_enp != NULL )
		pdp = pfdev_for_call(args_enp);
	else
		pdp = default_pfdev();

	if( pdp == NULL ){
		vl2_init_platform(SINGLE_QSP_ARG);
		pdp = default_pfdev();
	}

	assert(pdp!=NULL);

	push_pfdev(pdp);
#else
	pdp = NULL;
#endif // HAVE_ANY_GPU

	if( (kp=find_fused_kernel(srp,pdp)) != NULL ){
		run_fused_kernel(srp,args_enp,kp,pdp);
	} else {
		if( rip->ri_arg_stat >= 0 ){
			eval_decl_tree(SR_BODY(srp));
			/* eval_work_tree returns 0 if a return statement was executed,
			 * but not if there is an implied return.
			 *
			 * Uh, what is an "implied" return???
			 */

			// BUG - eval_work_tree calls delete_local_objs, but dst_dp
			// here may be a local object!?
			// We might test for dst_dp being local before making the call,
			// but would that be sufficient???
			eval_work_tree(SR_BODY(srp),dst_dp);
		} else {
sprintf(ERROR_STRING,"run_subrt %s:  arg_stat = %d",SR_NAME(srp),rip->ri_arg_stat);
warn(ERROR_STRING);
		}
	}

	wrapup_call(rip);
#ifdef HAVE_ANY_GPU
	pop_pfdev();
#endif // HAVE_ANY_GPU
}

/* A utility routine used when a declaration item has null
 * (or currently unevaluat-able)
 * children instead of intexp tree nodes.
 * The first time through, we set the dimensions to 0,
 * to flag to the data library that this is an unknown shape
 * object.
 *
 * After a run-time scan, the node will have its
 * shape struct set to the current shape context, so we
 * use those values...
 *
 * There is likely to be a more efficient way to get this done,
 * but for now we are focussing on the semantics...
 *
 * We used to maintain a per-subrt list of uk arg's, now we will
 * link each node to it's parent (some sort of T_DECL_ITEM_LIST).  If there
 * is no parent, we are at the top T_DECL_ITEM_LIST node; this means we
 * need to use the arg values to resolve...
 */

#define setup_unknown_shape(enp,dsp) _setup_unknown_shape(QSP_ARG  enp,dsp)

static void _setup_unknown_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,Dimension_Set *dsp)
{
	int i;
	if( VN_SHAPE(enp) == NULL ){
		for(i=0;i<N_DIMENSIONS;i++) SET_DIMENSION(dsp,i,0);
		copy_node_shape(enp,uk_shape(VN_DECL_PREC_CODE(enp)));
	} else {
		/* use node shape struct
		 * hopefully set to calling context
		 * by rt_scan()
		 */
		for(i=0;i<N_DIMENSIONS;i++)
			SET_DIMENSION(dsp,i, SHP_TYPE_DIM(VN_SHAPE(enp),i) );
	}

	assert( VN_PARENT(enp) != NULL );
}

#define finish_obj_decl(enp,dsp,prec_p, decl_flags) _finish_obj_decl(QSP_ARG  enp,dsp,prec_p, decl_flags)

static Data_Obj * _finish_obj_decl(QSP_ARG_DECL  Vec_Expr_Node *enp,Dimension_Set *dsp,Precision *prec_p, int decl_flags)
{
	Data_Obj *dp;

	eval_enp = enp;

	/* at one time we handled special (complex) precision here... */

	dp=make_dobj(VN_STRING(enp),dsp,prec_p);

	if( dp==NULL ){
		node_error(enp);
		sprintf(ERROR_STRING,
			"Error processing declaration for object %s",
			VN_STRING(enp));
		warn(ERROR_STRING);
		return(dp);
	}

	/* remember the declaration node for this object */
	/* However, if this declaration statement is deleted, then
	 * this will be a dangling pointer...  One solution might
	 * be to attach a list of objects to the declaration node...
	 *
	 * For global declarations, make then static - but how
	 * can we tell if we are in a subroutine or not?
	 * Try checking the object context...
	 */
	SET_OBJ_EXTRA(dp, enp);

	{
		Item_Context *icp;
		icp = current_dobj_context();
		if( !strcmp(CTX_NAME(icp),"Data_Obj.default") ){
//fprintf(stderr,"dobj context is %s, forcing static\n",CTX_NAME(icp));
			decl_flags |= DECL_IS_STATIC;
		}
	}

// The problem is not with recycling nodes...  it is with local vars in subroutines.
// The declaration statements are evaluated each time we call the subroutine.
// So we need to clear the field when the object is deleted!
// This could create an unwanted dependency between the dobj module and this one!?

if( VN_DECL_OBJ(enp) != NULL ){
sprintf(ERROR_STRING,"%s decl obj (%s) is not null!?",
node_desc(enp),OBJ_NAME(VN_DECL_OBJ(enp)));
warn(ERROR_STRING);
}

	// We are getting an error with an immediate declaration...
	// This is a static object, but it appears it gets deleted and then evaluated again?
	if( VN_DECL_OBJ(enp) != NULL ){
fprintf(stderr,"VN_DECL_OBJ = 0x%lx\n",(long)(VN_DECL_OBJ(enp)));
fprintf(stderr,"OBJ_NAME(VN_DECL_OBJ) = 0x%lx\n",(long)(OBJ_NAME(VN_DECL_OBJ(enp))));
		assert(OBJ_NAME(VN_DECL_OBJ(enp))!=NULL);

		if( !strncmp("Z.",OBJ_NAME(VN_DECL_OBJ(enp)),2) ){
			// the decl object is a zombie!?
			advise("memory leak from zombie object?");
		} else {
			assert( VN_DECL_OBJ(enp) == NULL );
		}
	}

	SET_VN_DECL_OBJ(enp,dp);

	copy_node_shape(enp,OBJ_SHAPE(dp));

/* BUG - now the const-ness is passed in the rpecision struct... */
	if( decl_flags & DECL_IS_CONST ) SET_OBJ_FLAG_BITS(dp, DT_RDONLY);
	if( decl_flags & DECL_IS_STATIC ) SET_OBJ_FLAG_BITS(dp, DT_STATIC);

	return(dp);
}	// finish_obj_decl


/* Evaluate a declaration statement, e.g.
 * float x,y;
 * When we get to here, we have already seen the root T_DECL_STAT node...
 *
 * For open-ended declarations like float x[], the child node
 * will be null, and we will make the object have unknown shape
 * by setting all dims to 0.
 *
 * A trickier example is float x[a], where the value of a needs to be
 * determined at runtime.  We make this a unknown object too...
 * But it won't be resolved by shape, it must be resolved by the value of a...
 * We might set a flag RESOLVE_AT_RUNTIME...
 */

static void _eval_decl_stat(QSP_ARG_DECL Precision * prec_p,Vec_Expr_Node *enp, int decl_flags)
{
	int i;
	Dimension_Set ds1, *dsp=(&ds1);
	int type/* =ID_OBJECT */;
	Identifier *idp;

	if( PREC_CODE(prec_p) == PREC_STR ){
		type = ID_STRING;
	} else {
		type = ID_OBJ_REF;	// default - refers to an object
	}

	eval_enp = enp;

	for(i=0;i<N_DIMENSIONS;i++)
		SET_DIMENSION(dsp,i,1 );

	switch(VN_CODE(enp)){
		case T_PROTO:
			{
			Subrt *srp;
			srp=subrt_of(VN_STRING(enp));
			if( srp != NULL ){
				/* subroutine already declared.
				 * We should check to make sure that the arg decls match BUG
				 * this gets done elsewhere, but here we make sure the return
				 * type is the same.
				 */
				/* This subroutine has already been declared...
				 * make sure the type matches
				 */
				if( PREC_CODE(prec_p) != SR_PREC_CODE(srp) )
					prototype_mismatch(SR_ARG_DECLS(srp),enp);
				break;
			}
			srp = remember_subrt(prec_p,VN_STRING(enp),VN_CHILD(enp,0),NULL);
			SET_SR_N_ARGS(srp, decl_count(SR_ARG_DECLS(srp)) );	/* set # args */
			SET_SR_FLAG_BITS(srp, SR_PROTOTYPE);
			return;
			}

		case T_BADNAME:
			return;
		case T_DECL_ITEM_LIST:
			eval_decl_stat(prec_p,VN_CHILD(enp,0),decl_flags);
			if( VN_CHILD(enp,1)!=NULL )
				eval_decl_stat(prec_p,VN_CHILD(enp,1),decl_flags);
			return;
		case T_DECL_INIT:		/* eval_decl_stat */
			{
			Scalar_Value sval;
			Data_Obj *dp;
			double dval;

			/* CURDLED? */
			if( IS_CURDLED(enp) ) return;

			eval_decl_stat(prec_p,VN_CHILD(enp,0),decl_flags);
			/* the next node is an expression */
			dp = get_id_obj(VN_STRING(VN_CHILD(enp,0)),enp);
			assert(dp!=NULL);
// Can this be legitimately NULL?
// BUG this is not CAUTIOUS because this can happen if you try to create
// an existing object that has not been exported.

			/* What if the rhs is unknown size - then we have to resolve now! */
			if( UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
				/* Can we assume the rhs has a shape? */
				if( UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) ){
					node_error(enp);
					warn("LHS and RHS are both unknown shape!?");
				} else {
					resolve_tree(enp,NULL);
					dump_tree(enp);
				}
			}

			if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) ){
				dval = eval_flt_exp(VN_CHILD(enp,1));
				dbl_to_scalar(&sval,dval,OBJ_PREC_PTR(dp));
				assign_obj_from_scalar(enp,dp,&sval);
			} else {
				eval_obj_assignment(dp,VN_CHILD(enp,1));
			}
			return;
			}
		case T_SCAL_DECL:
			SET_VN_DECL_PREC(enp, prec_p);
			/*type = ID_SCALAR;*/
			break;

		case T_CSCAL_DECL:					/* eval_decl_stat */
			// If this is a complex scalar, why allow unknown shapes???
			SET_VN_DECL_PREC(enp, prec_p);

			/* eg float x{3} */
			if( VN_CHILD(enp,0) == NULL ){
				/* float x{} */
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(enp,dsp);
				else
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp,SHP_TYPE_DIMS(VN_SHAPE(enp)));
			} else {
				SET_DIMENSION(dsp,0,eval_int_exp(VN_CHILD(enp,0)) );
				if( DIMENSION(dsp,0) == 0 ){
					setup_unknown_shape(enp,dsp);
				}
			}
			break;

		case T_VEC_DECL:			/* eval_decl_stat */
			SET_VN_DECL_PREC(enp, prec_p);

			if( VN_CHILD(enp,0) == NULL ){
				if( ! IS_RESOLVED(enp) ) {
					setup_unknown_shape(enp,dsp);
				} else {
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp, SHP_TYPE_DIMS(VN_SHAPE(enp)));
				}
			} else {
				SET_DIMENSION(dsp,1,eval_int_exp(VN_CHILD(enp,0)) );
				if( DIMENSION(dsp,1) == 0 ){
					setup_unknown_shape(enp,dsp);
				}
			}
			break;
		case T_CVEC_DECL:
			SET_VN_DECL_PREC(enp, prec_p);

			if( VN_CHILD(enp,0) == NULL ){
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(enp,dsp);
				else {
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp, SHP_TYPE_DIMS(VN_SHAPE(enp)));
				}
			} else {
				SET_DIMENSION(dsp,1,eval_int_exp(VN_CHILD(enp,0)) );
				SET_DIMENSION(dsp,0,eval_int_exp(VN_CHILD(enp,1)) );
				if( DIMENSION(dsp,2) == 0 || DIMENSION(dsp,1) == 0 ){
					setup_unknown_shape(enp,dsp);
				}
			}
			break;
		case T_IMG_DECL:
			SET_VN_DECL_PREC(enp, prec_p);

			if( VN_CHILD(enp,0) == NULL ){
				if( ! IS_RESOLVED(enp) ){
					setup_unknown_shape(enp,dsp);
				} else {
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp, SHP_TYPE_DIMS(VN_SHAPE(enp)));
				}
			} else {
				SET_DIMENSION(dsp,2,eval_int_exp(VN_CHILD(enp,0)) );
				SET_DIMENSION(dsp,1,eval_int_exp(VN_CHILD(enp,1)) );
				if( DIMENSION(dsp,2) == 0 || DIMENSION(dsp,1) == 0 ){
					setup_unknown_shape(enp,dsp);
				}

			}
			break;
		case T_CIMG_DECL:
			SET_VN_DECL_PREC(enp, prec_p);

			if( VN_CHILD(enp,0) == NULL ){
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(enp,dsp);
				else {
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp, SHP_TYPE_DIMS(VN_SHAPE(enp)));
				}
			} else {
				SET_DIMENSION(dsp,2,eval_int_exp(VN_CHILD(enp,0)) );
				SET_DIMENSION(dsp,1,eval_int_exp(VN_CHILD(enp,1)) );
				SET_DIMENSION(dsp,0,eval_int_exp(VN_CHILD(enp,2)) );
				if( DIMENSION(dsp,2) == 0 || DIMENSION(dsp,1) == 0 || DIMENSION(dsp,0) == 0 ){
					setup_unknown_shape(enp,dsp);
				}
			}
			break;
		case T_SEQ_DECL:
			SET_VN_DECL_PREC(enp, prec_p);

			if( VN_CHILD(enp,0) == NULL ){
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(enp,dsp);
				else {
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp, SHP_TYPE_DIMS(VN_SHAPE(enp)));
				}
			} else {
				SET_DIMENSION(dsp,3,eval_int_exp(VN_CHILD(enp,0)) );
				SET_DIMENSION(dsp,2,eval_int_exp(VN_CHILD(enp,1)) );
				SET_DIMENSION(dsp,1,eval_int_exp(VN_CHILD(enp,2)) );
				if( DIMENSION(dsp,3) == 0 || DIMENSION(dsp,2) == 0 || DIMENSION(dsp,1) == 0 ){
					setup_unknown_shape(enp,dsp);
				}
			}
			break;
		case T_CSEQ_DECL:
			SET_VN_DECL_PREC(enp, prec_p);

			if( VN_CHILD(enp,0) == NULL ){
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(enp,dsp);
				else {
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp, SHP_TYPE_DIMS(VN_SHAPE(enp)));
				}
			} else {
				Vec_Expr_Node *enp2;
				SET_DIMENSION(dsp,3,eval_int_exp(VN_CHILD(enp,0)) );
				SET_DIMENSION(dsp,2,eval_int_exp(VN_CHILD(enp,1)) );
				enp2 = VN_CHILD(enp,2);
				assert( VN_CODE(enp2) == T_EXPR_LIST );

				SET_DIMENSION(dsp,1,eval_int_exp(VN_CHILD(enp2,0)) );
				SET_DIMENSION(dsp,0,eval_int_exp(VN_CHILD(enp2,1)) );
				if( DIMENSION(dsp,3) == 0 || DIMENSION(dsp,2) == 0 || DIMENSION(dsp,1) == 0 || DIMENSION(dsp,0) == 0 ){
					setup_unknown_shape(enp,dsp);
				}
			}
			break;
		case T_PTR_DECL:			/* eval_decl_stat() */
			SET_VN_DECL_PREC(enp, prec_p);

			/* call by reference */
			if( type != ID_STRING )
				type = ID_POINTER;
			if( type == ID_STRING ){
				// what about PREC_STR???
				assert( PREC_CODE(prec_p) == PREC_CHAR );
			}
			break;
		case T_FUNCPTR_DECL:			/* eval_decl_stat */
			SET_VN_DECL_PREC(enp, prec_p);

			type = ID_FUNCPTR;
			break;
		default:
			missing_case(enp,"eval_decl_stat");
			break;
	}

	if( PREC_CODE(prec_p)==PREC_COLOR ){
		SET_DIMENSION(dsp,0,3 );
	} else if( PREC_CODE(prec_p)==PREC_VOID ){	/* determine from context? */
		if( VN_SHAPE(enp) != NULL )
			prec_p=SHP_PREC_PTR(VN_SHAPE(enp));
	}

	/* We allow name conflicts at levels above the current context.
	 * RESTRICT_ITEM_CONTEXT causes item lookup to only use the top context.
	 */

	restrict_id_context(1);

	assert( VN_STRING(enp) != NULL );

	// Make sure this name has not been used already...
	idp = id_of(VN_STRING(enp));
	if( idp != NULL ){
		node_error(enp);
		sprintf(ERROR_STRING,"identifier %s redeclared",VN_STRING(enp));
		advise(ERROR_STRING);
		/* BUG - we'd like to print the location of the original declaration... */
		/* it might have been declared previously because of an extern decl...
		 * if the old declaration didn't declare the size, and this one does,
		 * then we'd like to set the size.  If the size is already set, then we'd
		 * like to make sure they match.
		 */
		return;
	}

	// Now check in the global context, so we can print a shadowing warning
	// if it exists there.  In that case, we still create the var in the local
	// context...

	restrict_id_context(0);
	idp=id_of(VN_STRING(enp));
	if( idp != NULL ){
		/* only print this message once (the code seems to be
		 * executed 3 times!?
		 */
		if( ! WAS_WARNED(enp) ){
			node_error(enp);
			sprintf(ERROR_STRING,"declaration of %s masks previous declaration",VN_STRING(enp));
			advise(ERROR_STRING);
			MARK_WARNED(enp)
		}
		/* this stuff should all be debug only... */
		/*
		if( ID_TYPE(idp) == ID_OBJECT ){
			Vec_Expr_Node *decl_enp;
			sprintf(ERROR_STRING,"context of %s (%s) is %s",
				ID_NAME(idp),OBJ_NAME(idp->id_dp),CTX_NAME(ID_DOBJ_CTX(idp)));
			advise(ERROR_STRING);
			decl_enp = OBJ_EXTRA(idp->id_dp);
			if( decl_enp != NULL ){
				advise("previous object declaration at:");
				node_error(decl_enp);
			}
			sprintf(ERROR_STRING,"current context is %s",
				CTX_NAME(((Item_Context *)NODE_DATA(QLIST_HEAD(LIST_OF_DOBJ_CONTEXTS)))));
			advise(ERROR_STRING);
		}
		*/
	}

	/* remember the declaration identifier context */
	// Could we use CURRENT_CONTEXT equivalently???
	SET_VN_DECL_CTX(enp, (Item_Context *)NODE_DATA(QLIST_HEAD(LIST_OF_ID_CONTEXTS)) );

//sprintf(ERROR_STRING,"eval_decl_stat:  creating id %s, type = %d",
//VN_STRING(enp),type);
//advise(ERROR_STRING);
	// New items are always created in the top context.

	idp = new_identifier(VN_STRING(enp));		/* eval_decl_stat */
	SET_ID_TYPE(idp, type);


	assert( idp != NULL );

	switch( type ){
#ifdef SCALARS_NOT_OBJECTS
		case ID_SCALAR:
			SET_ID_SVAL_PTR( idp, getbuf(sizeof(Scalar_Value)) );
			copy_node_shape(enp,scalar_shape(PREC_CODE(prec_p)));
			set_id_shape(idp,VN_SHAPE(enp));
			break;
#endif // SCALARS_NOT_OBJECTS

		case ID_OBJ_REF:
			// Here we create an object...
			SET_ID_DOBJ_CTX(idp , (Item_Context *)NODE_DATA(QLIST_HEAD(LIST_OF_DOBJ_CONTEXTS)) );
			SET_ID_REF(idp, NEW_REFERENCE );
			SET_REF_ID(ID_REF(idp), idp );
			SET_REF_DECL_VN(ID_REF(idp), enp );	/* BUG? */
			SET_REF_TYPE(ID_REF(idp), OBJ_REFERENCE );
			// This would be a place to OR in the STATIC flag
			// if this declaration is not within a subroutine?
			// see dangling pointer problem...  subrt_ctx
			SET_REF_OBJ(ID_REF(idp), finish_obj_decl(enp,dsp,prec_p,decl_flags) );	/* eval_decl_stat */

			set_id_shape(idp,VN_SHAPE(enp));

			// This was CAUTIOUS before, but this can happen
			// if the user tries to create an object that
			// was already created outside of the expression parser

			if( REF_OBJ(ID_REF(idp)) == NULL ){
				// Need to clean up!
fprintf(stderr,"eval_decl_stat:  deleting identifier %s\n",ID_NAME(idp));
				delete_id((Item *)idp);

				node_error(enp);
				sprintf(ERROR_STRING,
			"eval_decl_stat:  unable to create object for id %s",ID_NAME(idp));
				warn(ERROR_STRING);
			}

			break;

		case ID_STRING:
			SET_ID_REF(idp, NEW_REFERENCE );
			SET_REF_ID(ID_REF(idp), idp );
			SET_REF_DECL_VN(ID_REF(idp), enp );	/* BUG? */
			SET_REF_TYPE(ID_REF(idp), STR_REFERENCE );
			SET_REF_SBUF(ID_REF(idp), new_stringbuf() );
			break;
		case ID_POINTER:
			SET_ID_PTR(idp, NEW_POINTER );
			SET_PTR_REF(ID_PTR(idp), NULL);
			SET_PTR_FLAGS(ID_PTR(idp), 0);
			SET_PTR_DECL_VN(ID_PTR(idp), enp);
			copy_node_shape(enp,uk_shape(PREC_CODE(prec_p)));
			break;
		case ID_FUNCPTR:
			//SET_ID_FUNC(idp, (Function_Ptr *)getbuf(sizeof(Function_Ptr)) );
			SET_ID_FUNC(idp, NEW_FUNC_PTR );
			ID_FUNC(idp)->fp_srp = NULL;
			copy_node_shape(enp,uk_shape(PREC_CODE(prec_p)));
			break;
		default:
			node_error(enp);
			sprintf(ERROR_STRING,
				"identifier type %d not handled by eval_decl_stat switch",
				type);
			warn(ERROR_STRING);
			break;
	}
} /* end eval_decl_stat */

static void _eval_extern_decl(QSP_ARG_DECL Precision * prec_p,Vec_Expr_Node *enp, int decl_flags)
{
	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_PROTO:
			{
			Subrt *srp;
			srp=subrt_of(VN_STRING(enp));
			if( srp == NULL ) eval_decl_stat(prec_p,enp,decl_flags);
			else {
				/* This subroutine has already been declared...
				 * make sure the type matches
				 */
				if( PREC_CODE(prec_p) != SR_PREC_CODE(srp) )
					prototype_mismatch(SR_ARG_DECLS(srp),enp);
			}

			/* BUG make sure arg decls match */
			return;
			}
		case T_BADNAME: return;
		case T_DECL_ITEM_LIST:
			eval_extern_decl(prec_p,VN_CHILD(enp,0),decl_flags);
			if( VN_CHILD(enp,1)!=NULL )
				eval_extern_decl(prec_p,VN_CHILD(enp,1),decl_flags);
			return;
		case T_DECL_INIT:
			node_error(enp);
			advise("no auto-initialization with extern declarations");
			eval_extern_decl(prec_p,VN_CHILD(enp,0),decl_flags);
			return;

		case T_SCAL_DECL:
		case T_CSCAL_DECL:			/* eval_extern_decl */
		case T_VEC_DECL:			/* eval_extern_decl */
		case T_CVEC_DECL:
		case T_IMG_DECL:
		case T_CIMG_DECL:
		case T_SEQ_DECL:
		case T_CSEQ_DECL:
			{
			Data_Obj *dp;

			dp=dobj_of(VN_STRING(enp));
			if( dp == NULL ){
fprintf(stderr,"eval_extern_decl\n");
				eval_decl_stat(prec_p,enp,decl_flags);
				return;
			}
			/* BUG should check that decl matches earlier one... */
			break;
			}
		case T_PTR_DECL:			/* eval_extern_decl */
			{
			Identifier *idp;
			idp = id_of(VN_STRING(enp));
			if( idp == NULL ){
				eval_decl_stat(prec_p,enp,decl_flags);
				return;
			}
			/* BUG chould check that type matches earlier type */
			break;
			}

		default:
			missing_case(enp,"eval_extern_decl");
			break;
	}
}

/* We call eval_tree when we may have declarations as well as statements */

int _eval_tree(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj *dst_dp)
{
	int ret_val=1;

	if( enp==NULL || IS_CURDLED(enp) ) return(ret_val);

	eval_enp = enp;
	if( interrupted ) return 0;

#ifdef FOOBAR
	/*
	if( signal(SIGINT,intr_evaluation) == SIG_ERR ){
		error1("error setting evaluation interrupt handler");
		IOS_RETURN
	}
		*/
#endif /* FOOBAR */

	if( going ) return(eval_work_tree(enp,dst_dp));

	switch(VN_CODE(enp)){
		case T_EXIT:
			{
			int status;
			status = (int) eval_int_exp( VN_CHILD(enp,0) );
			exit(status);
			}

		case T_STAT_LIST:			/* eval_tree */
		case T_GO_FWD:  case T_GO_BACK:		/* eval_tree */
		case T_BREAK:
		ALL_INCDEC_CASES
		case T_CALLFUNC:
		case T_ASSIGN:
		case T_RETURN:
		case T_WARN:
		case T_IFTHEN:
		case T_EXP_PRINT:
		case T_DISPLAY:
		case T_ADVISE:
			ret_val = eval_work_tree(enp,dst_dp);
			break;

		case T_DECL_STAT:
			/* why en_intval here, and not en_cast_prec??? */
			eval_decl_stat(VN_DECL_PREC(enp),VN_CHILD(enp,0),VN_DECL_FLAGS(enp));
			break;
		case T_EXTERN_DECL:
			eval_extern_decl(VN_DECL_PREC(enp),VN_CHILD(enp,0),VN_DECL_FLAGS(enp));
			break;
		default:
			missing_case(enp,"eval_tree");
			ret_val = eval_work_tree(enp,dst_dp);
			break;
	}
	return(ret_val);
} // eval_tree

#ifdef NOT_YET
static int _get_filetype_index(QSP_ARG_DECL  const char *name)
{
	int i;

	for(i=0;i<N_FILETYPE;i++){
		if( !strcmp(ft_tbl[i].ft_name,name) )
			return(i);
	}
	sprintf(ERROR_STRING,"Invalid filetype name %s",name);
	warn(ERROR_STRING);
	sprintf_ERROR_STRING,"Valid selections are:");
	advise(ERROR_STRING);
	for(i=0;i<N_FILETYPE;i++){
		sprintf(ERROR_STRING,"\t%s", ft_tbl[i].ft_name);
		advise(ERROR_STRING);
	}
	return -1;
}
#endif /* NOT_YET */

static void _eval_info_stat(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_UNDEF:
			break;

		case T_EXPR_LIST:
			eval_info_stat(VN_CHILD(enp,0));
			eval_info_stat(VN_CHILD(enp,1));
			break;

		case T_DEREFERENCE:			/* eval_info_stat */
		case T_STATIC_OBJ:	/* eval_info_stat */
		case T_DYN_OBJ:		/* eval_info_stat */
		case T_POINTER:
		case T_STR_PTR:
		case T_SUBVEC:
		case T_CSUBVEC:
		case T_SQUARE_SUBSCR:
		case T_CURLY_SUBSCR:
		case T_SUBSAMP:
		case T_CSUBSAMP:
			dp = eval_obj_ref(enp);
			if( dp==NULL )
				warn("missing info object");
			else {
				longlist(dp);
			}
			break;
		default:
			missing_case(enp,"eval_info_stat");
			break;
	}
}

/* I would have put FOOBAR here but we might need this for future debug */
#ifdef QUIP_DEBUG_ONLY

static void show_ref(Reference *refp)
{
	advise("show_ref:");
	if( REF_TYPE(refp) == OBJ_REFERENCE ){
		sprintf(ERROR_STRING,
			"show_ref:  ref at 0x%"PRIxPTR":  object %s",
			(uintptr_t)refp, OBJ_NAME(REF_OBJ(refp)));
		advise(ERROR_STRING);
	} else if( REF_TYPE(refp) == STR_REFERENCE ){
		sprintf(ERROR_STRING,"show_ref:  string");
		advise(ERROR_STRING);
	} else {
		sprintf(ERROR_STRING,"show_ref:  unexpected ref type %d",
			REF_TYPE(refp));
		warn(ERROR_STRING);
	}
}

static void show_ptr(Pointer *ptrp)
{
	sprintf(ERROR_STRING,"Pointer at 0x%"PRIxPTR,(uintptr_t)ptrp);
	advise(ERROR_STRING);
}

#endif /* QUIP_DEBUG_ONLY */

#ifdef FOOBAR		/* not used? */
static Vec_Expr_Node *find_goto(Vec_Expr_Node *enp)
{
	Vec_Expr_Node *ret_enp;

	switch(VN_CODE(enp)){
		case T_LABEL:
			if( !strcmp(VN_STRING(enp),goto_label) ){
				return(enp);
			}
			break;

		case T_STAT_LIST:			/* find_goto */
			ret_enp=find_goto(VN_CHILD(enp,0));
			if( ret_enp != NULL ) return(ret_enp);
			ret_enp=find_goto(VN_CHILD(enp,1));
			return(ret_enp);

		case T_DECL_STAT:
		case T_EXP_PRINT:
			return(NULL);

		default:
			missing_case(enp,"find_goto");
			break;
	}
	return(NULL);
}
#endif /* FOOBAR */

static const char *name_for_ref( Reference *ref_p )
{
	if( ref_p == NULL ) return "(null ref_p)";

	switch( REF_TYPE(ref_p) ){
		case OBJ_REFERENCE:
			return OBJ_NAME(REF_OBJ(ref_p));
		default:
			return "(unhandled ref type case in name_for_ref)";
	}
}

static void dump_ref( Identifier *idp )
{
	Reference *refp;
    assert(idp!=NULL);
	fprintf(stderr,"Showing reference info for identifier %s\n",ID_NAME(idp));
}

long _eval_int_exp(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	long lval,lval2;
	double dval1,dval2;
	Data_Obj *dp;
	Scalar_Value *svp,sval;
	Subrt *srp;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		/* case T_MATH1_FUNC: */	/* returns double - should have been typecast? */
#ifdef SCALARS_NOT_OBJECTS
		case T_SCALAR_VAR:		// eval_int_exp
			{
			Identifier *idp;
			idp = get_id(VN_STRING(enp));
			assert(idp!=NULL);
			lval = (long) cast_from_scalar_value(ID_SVAL_PTR(idp), ID_PREC_PTR(idp));
			return lval;
			}
#endif // SCALARS_NOT_OBJECTS

		case T_VS_FUNC:
			dp = eval_obj_exp(enp,NULL);
			if( dp == NULL ){
				node_error(enp);
				warn("unable to evaluate vector-scalar expression");
				return 0;
			}
			if( !IS_SCALAR(dp) ){
				node_error(enp);
				sprintf(ERROR_STRING,
	"eval_int_exp T_VS_FUNC:  object %s is not a scalar!?",OBJ_NAME(dp));
				warn(ERROR_STRING);
				return 0;
			}
			return( get_long_scalar_value(dp) );
			break;

		case T_RECIP:			/* eval_int_exp */
			lval = eval_int_exp(VN_CHILD(enp,0));
			if( lval == 0 ){
				node_error(enp);
				warn("divide by zero!?");
				return 0;
			}
			return(1/lval);
			
		case T_TYPECAST:	/* eval_int_exp */
			dval1 = eval_flt_exp(VN_CHILD(enp,0));
			switch(VN_PREC(enp)){
				case PREC_BY:   return( (long) ((char)     dval1 ) );
				case PREC_STR:
				case PREC_UBY:  return( (long) ((u_char)   dval1 ) );
				case PREC_IN:   return( (long) ((short)    dval1 ) );
				case PREC_UIN:  return( (long) ((u_short)  dval1 ) );
				case PREC_DI:   return( (long) ((int32_t)  dval1 ) );
				case PREC_UDI:  return( (long) ((uint32_t) dval1 ) );
				case PREC_LI:   return( (long) ((int64_t)  dval1 ) );
				case PREC_ULI:  return( (long) ((uint64_t) dval1 ) );
				case PREC_SP:   return( (long) ((float)    dval1 ) );
				case PREC_DP:   return( (long)             dval1   );
				case PREC_BIT:
					if( dval1 == 0.0 ) return 0;
					else return(1);

				default:
					dump_tree(enp);
					assert( AERROR("eval_int_exp:  unhandled precision") );
			}
			break;

		case T_CALLFUNC:			/* eval_int_exp */
			/* This could get called if we use a function inside a dimesion bracket... */
			if( ! executing ) return 0;

			srp=VN_SUBRT(enp);
			/* BUG SHould check and see if the return type is int... */

			/* BUG at least make sure that it's not void... */

			/* make a scalar object to hold the return value... */
			if( scalar_dsp == NULL ){
				INIT_DIMSET_PTR(scalar_dsp)
				SET_DIMENSION(scalar_dsp,0,1);
				SET_DIMENSION(scalar_dsp,1,1);
				SET_DIMENSION(scalar_dsp,2,1);
				SET_DIMENSION(scalar_dsp,3,1);
				SET_DIMENSION(scalar_dsp,4,1);
			}
			dp=make_local_dobj(scalar_dsp,SR_PREC_PTR(srp),NULL);
			exec_subrt(enp,dp);
			/* get the scalar value */
			lval = get_long_scalar_value(dp);
			delvec(dp);
			return(lval);
			break;

		case T_POSTINC:
			dp = eval_obj_ref(VN_CHILD(enp,0));
			lval = eval_int_exp(VN_CHILD(enp,0));
			inc_obj(dp);
			return(lval);

		case T_POSTDEC:
			dp = eval_obj_ref(VN_CHILD(enp,0));
			lval = eval_int_exp(VN_CHILD(enp,0));
			dec_obj(dp);
			return(lval);

		case T_PREDEC:
			dp = eval_obj_ref(VN_CHILD(enp,0));
			dec_obj(dp);
			return(eval_int_exp(VN_CHILD(enp,0)));

		case T_PREINC:		/* eval_int_exp */
			dp = eval_obj_ref(VN_CHILD(enp,0));
			inc_obj(dp);
			return(eval_int_exp(VN_CHILD(enp,0)));

		case T_ASSIGN:			/* eval_int_exp */
			dp = eval_obj_ref(VN_CHILD(enp,0));
			lval = eval_int_exp(VN_CHILD(enp,1));
			int_to_scalar(&sval,lval,OBJ_PREC_PTR(dp));
			if( assign_obj_from_scalar(enp,dp,&sval) < 0 )
				return 0;
			return(lval);

		case T_UNDEF:
			return 0;

		case T_FILE_EXISTS:
			{
				const char *s;
				s=eval_string(VN_CHILD(enp,0));
				if( s != NULL )
					return(file_exists(s));
				else
					return 0;
			}
			break;

		case T_SQUARE_SUBSCR:			/* eval_int_exp */
		case T_CURLY_SUBSCR:			/* eval_int_exp */
		case T_STATIC_OBJ:			/* eval_int_exp */
		case T_DYN_OBJ:				/* eval_int_exp */
			/* If we are preprocessing a declaration, don't bother
			 * to get the value - it will have to be determined at runtime.
			 * Just return 0...
			 */
			dp = eval_obj_ref(enp);
			assert( dp != NULL );

			if( ! IS_SCALAR(dp) ){
				node_error(enp);
				sprintf(ERROR_STRING,
	"eval_int_exp:  Object %s is not a scalar!?",OBJ_NAME(dp));
				warn(ERROR_STRING);
				longlist(dp);	// after an error message
				return 0;
			}
			/* has the object been set? */
			if( ! HAS_ALL_VALUES(dp) ){
				if( executing && expect_objs_assigned ){
					unset_object_warning(enp,dp);
				}
				
				return 0;			/* we don't print the warning unless we know
								 * that we aren't doing pre-evaluation...
								 */
			}
			svp = (Scalar_Value *)OBJ_DATA_PTR(dp);
			switch(OBJ_MACH_PREC(dp)){
				case PREC_BY:  lval = svp->u_b; break;
				case PREC_IN:  lval = svp->u_s; break;
				case PREC_DI:  lval = svp->u_l; break;
				case PREC_LI:  lval = (long)svp->u_ll; break;
				case PREC_SP:  lval = (long) svp->u_f; break;
				case PREC_DP:  lval = (long) svp->u_d; break;
				case PREC_UBY:  lval = svp->u_ub; break;
				case PREC_UIN:  lval = svp->u_us; break;
				case PREC_UDI:  lval = svp->u_ul; break;
				case PREC_ULI:  lval = (long)svp->u_ull; break;
				case PREC_NONE:
				case N_MACHINE_PRECS:
				default:
					assert( AERROR("eval_int_exp:  nonsense precision") );
					lval=0.0;	// quiet compiler
					break;
			}
			return(lval);
			break;

		case T_BOOL_OR:
			// BUG - not performing short-circuit
			// evaluation if the first subexpression
			// is true - see comment below...
			lval=eval_int_exp(VN_CHILD(enp,0));
			lval2=eval_int_exp(VN_CHILD(enp,1));
			if( lval || lval2 ) return(1);
			else return 0;
			break;
		case T_BOOL_AND:
			// Originally, we used short-circuit
			// evaluation here, where if the first
			// subexpression was false, then we didn't
			// need to evaluate the rest.
			// But that resulted in a memory leak.
			// BUG - there should be a way to
			// do the necessary to the subtree without
			// actually evaluating?
			lval=eval_int_exp(VN_CHILD(enp,0));
			lval2=eval_int_exp(VN_CHILD(enp,1));
			if( lval && lval2 ) return(1);
			else return 0;
			break;
		case T_BOOL_NOT:
			lval=eval_int_exp(VN_CHILD(enp,0));
			if( ! lval ) return(1);
			else return 0;
			break;
		case T_BOOL_GT:			/* eval_int_exp */
			dval1=eval_flt_exp(VN_CHILD(enp,0));
			dval2=eval_flt_exp(VN_CHILD(enp,1));
			if( dval1 > dval2 ) return(1);
			else return 0;
			break;
		case T_BOOL_LT:			/* eval_int_exp */
			dval1=eval_flt_exp(VN_CHILD(enp,0));
			dval2=eval_flt_exp(VN_CHILD(enp,1));
			if( dval1 < dval2 ) return(1);
			else return 0;
			break;
		case T_BOOL_GE:
			dval1=eval_flt_exp(VN_CHILD(enp,0));
			dval2=eval_flt_exp(VN_CHILD(enp,1));
			if( dval1 >= dval2 ) return(1);
			else return 0;
			break;
		case T_BOOL_LE:
			dval1=eval_flt_exp(VN_CHILD(enp,0));
			dval2=eval_flt_exp(VN_CHILD(enp,1));
			if( dval1 <= dval2 ) return(1);
			else return 0;
			break;
		case T_BOOL_NE:
			dval1=eval_flt_exp(VN_CHILD(enp,0));
			dval2=eval_flt_exp(VN_CHILD(enp,1));
			if( dval1 != dval2 ) return(1);
			else return 0;
			break;
		case T_BOOL_EQ:			/* eval_int_exp */
			dval1=eval_flt_exp(VN_CHILD(enp,0));
			dval2=eval_flt_exp(VN_CHILD(enp,1));
			if( dval1 == dval2 ) return(1);
			else return 0;
			break;
		/* Can we fuse these two???
		 * Would require examining types of child nodes?
		 */
		case T_BOOL_PTREQ:			/* eval_int_exp */
			{
			Identifier *idp1,*idp2;
			idp1=eval_ptr_expr(VN_CHILD(enp,0),EXPECT_PTR_SET);
if( idp1 == NULL ){
fprintf(stderr,"eval_int_exp T_BOOL_PTREQ:  idp1 is NULL!?\n");
} else {
fprintf(stderr,"eval_int_exp T_BOOL_PTREQ:  idp1 is %s\n",ID_NAME(idp1));
}
dump_ref(idp1);
			idp2=eval_ptr_expr(VN_CHILD(enp,1),EXPECT_PTR_SET);
if( idp2 == NULL ){
fprintf(stderr,"eval_int_exp T_BOOL_PTREQ:  idp2 is NULL!?\n");
} else {
fprintf(stderr,"eval_int_exp T_BOOL_PTREQ:  idp2 is %s\n",ID_NAME(idp2));
}
			/* CAUTIOUS check for ptrs? */
			/* BUG? any other test besides dp ptr identity? */
if( REF_OBJ(ID_REF(idp1)) == NULL ){
fprintf(stderr,"eval_int_exp T_BOOL_PTREQ:  ref obj 1 is NULL!?\n");
} else {
fprintf(stderr,"eval_int_exp T_BOOL_PTREQ:  ref obj 1 is %s!?\n",
name_for_ref(ID_REF(idp1))
);
}
if( REF_OBJ(ID_REF(idp2)) == NULL ){
fprintf(stderr,"eval_int_exp T_BOOL_PTREQ:  ref obj 2 is NULL!?\n");
} else {
fprintf(stderr,"eval_int_exp T_BOOL_PTREQ:  ref obj 2 is %s!?\n",
name_for_ref(ID_REF(idp2))
);
}
			if( REF_OBJ(ID_REF(idp1)) == REF_OBJ(ID_REF(idp2)) )
				return(1);
			else
				return 0;
			}

		case T_PLUS:
			lval=eval_int_exp(VN_CHILD(enp,0));
			lval2=eval_int_exp(VN_CHILD(enp,1));
			return(lval+lval2);
			break;
		case T_MINUS:
			lval=eval_int_exp(VN_CHILD(enp,0));
			lval2=eval_int_exp(VN_CHILD(enp,1));
			return(lval-lval2);
			break;
		case T_TIMES:
			lval=eval_int_exp(VN_CHILD(enp,0));
			lval2=eval_int_exp(VN_CHILD(enp,1));
			return(lval*lval2);
			break;
		case T_DIVIDE:
			lval=eval_int_exp(VN_CHILD(enp,0));
			lval2=eval_int_exp(VN_CHILD(enp,1));
			if( lval2==0 ){
				warn("integer division by 0!?");
				return(0L);
			}
			return(lval/lval2);
			break;
		case T_MODULO:
			lval=eval_int_exp(VN_CHILD(enp,0));
			lval2=eval_int_exp(VN_CHILD(enp,1));
			if( lval2==0 ){
				warn("integer division (modulo) by 0!?");
				return(0L);
			}
			return(lval%lval2);
			break;
		case T_BITOR:
			lval=eval_int_exp(VN_CHILD(enp,0));
			lval2=eval_int_exp(VN_CHILD(enp,1));
			return(lval|lval2);
			break;
		case T_BITAND:
			lval=eval_int_exp(VN_CHILD(enp,0));
			lval2=eval_int_exp(VN_CHILD(enp,1));
			return(lval&lval2);
			break;
		case T_BITXOR:
			lval=eval_int_exp(VN_CHILD(enp,0));
			lval2=eval_int_exp(VN_CHILD(enp,1));
			return(lval^lval2);
			break;
		case T_BITCOMP:
			lval=eval_int_exp(VN_CHILD(enp,0));
			return(~lval);
			break;
		case T_BITRSHIFT:
			lval=eval_int_exp(VN_CHILD(enp,0));
			lval2=eval_int_exp(VN_CHILD(enp,1));
			return(lval>>lval2);
			break;
		case T_BITLSHIFT:
			lval=eval_int_exp(VN_CHILD(enp,0));
			lval2=eval_int_exp(VN_CHILD(enp,1));
			return(lval<<lval2);
			break;
		case T_LIT_INT:			/* eval_int_exp */
			return(VN_INTVAL(enp));
			break;
		case T_UMINUS:
			lval=eval_int_exp(VN_CHILD(enp,0));
			return(-lval);
			break;

		case T_STR2_FN:			/* eval_int_exp */
		case T_LIT_DBL:
		case T_STR1_FN:	/* eval_int_exp */
		case T_SIZE_FN: 	/* eval_int_exp */
			lval= (long) eval_flt_exp(enp);
			return(lval);
			break;
		default:
			missing_case(enp,"eval_int_exp");
			break;
	}
	return(-1L);
} /* end eval_int_exp */

/* Process a tree, doing only declarations */

void _eval_decl_tree(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	if( enp==NULL )
		return;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_STAT_LIST:
			/* There should only be one T_STAT_LIST at
			 * the beginning of a subroutine, so we
			 * don't need to scan the second child.
			 * for declarations...
			 */
			eval_decl_tree(VN_CHILD(enp,0));
			break;
		case T_DECL_STAT_LIST:
			eval_decl_tree(VN_CHILD(enp,0));
			eval_decl_tree(VN_CHILD(enp,1));
			break;
		case T_DECL_STAT:
			eval_decl_stat(VN_DECL_PREC(enp),VN_CHILD(enp,0),VN_DECL_FLAGS(enp));
			break;
		case T_EXTERN_DECL:
			eval_extern_decl(VN_DECL_PREC(enp),VN_CHILD(enp,0),VN_DECL_FLAGS(enp));
			break;
		default:
			/* We will end up here with any code
			 * in a subroutine with no declarations.
			 */
			/*
			advise("You can safely ignore this warning???");
			missing_case(enp,"eval_decl_tree");
			*/
			break;
	}
}

static int compare_arg_decls(Vec_Expr_Node *enp1,Vec_Expr_Node *enp2)
{
	int i;

	if( enp1 == NULL ) {
		if( enp2 == NULL ) return 0;
		else return -1;
	} else if( enp2 == NULL ) return -1;

	if( VN_CODE(enp1) != VN_CODE(enp2) ) return -1;

	if( VN_CODE(enp1) == T_DECL_STAT ){
		if( VN_DECL_PREC(enp1) != VN_DECL_PREC(enp2) ) return -1;
	}

	for(i=0;i<MAX_CHILDREN(enp1);i++){
		if( compare_arg_decls(VN_CHILD(enp1,i),VN_CHILD(enp2,i)) < 0 )
			return -1;
	}
	return 0;
}

void _compare_arg_trees(QSP_ARG_DECL  Vec_Expr_Node *enp1,Vec_Expr_Node *enp2)
{
	if( compare_arg_decls(enp1,enp2) < 0 )
		prototype_mismatch(enp1,enp2);
}



/* This whole check is probably CAUTIOUS */

#define bad_reeval_shape(enp) _bad_reeval_shape(QSP_ARG  enp)

static int _bad_reeval_shape(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	eval_enp = enp;

	if( VN_SHAPE(enp) == NULL ){
		sprintf(ERROR_STRING,
	"reeval:  missing shape info for %s",VN_STRING(enp));
		warn(ERROR_STRING);
		return(1);
	}
	if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
		sprintf(ERROR_STRING,
	"reeval:  unknown shape info for %s",VN_STRING(enp));
		warn(ERROR_STRING);
		return(1);
	}
	return 0;
}

/* We call reeval_decl_stat when we think we know the sizes of all unknown
 * shape objects.
 *
 * We have a problem:  if we have unknown size global objects, they may be
 * resolved during the execution of a subroutine - they are destroyed and
 * created by reeval_decl_stat, but if they're created with the local subrt
 * context, then they will be destroyed when the subroutine exits...
 * Therefore we need to somehow carry the original context
 * around with us...
 */

void _reeval_decl_stat(QSP_ARG_DECL  Precision *prec_p,Vec_Expr_Node *enp,int decl_flags)
{
	int i;
	Dimension_Set ds1, *dsp=(&ds1);
	Data_Obj *dp;
	Identifier *idp;
	int context_pushed;

	eval_enp = enp;

	for(i=0;i<N_DIMENSIONS;i++)
		SET_DIMENSION(dsp,i,1 );

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"reeval_decl_stat, code is %d",VN_CODE(enp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	switch(VN_CODE(enp)){
		case T_BADNAME:
			return;
		case T_DECL_ITEM_LIST:
			reeval_decl_stat(prec_p,VN_CHILD(enp,0),decl_flags);
			if( VN_CHILD(enp,1)!=NULL )
				reeval_decl_stat(prec_p,VN_CHILD(enp,1),decl_flags);
			return;
			break;

		case T_SCAL_DECL:
fprintf(stderr,"reeval_decl_stat\n");
			return ;

		case T_IMG_DECL:			/* reeval_decl_stat */
		case T_VEC_DECL:
		case T_SEQ_DECL:
		case T_CSCAL_DECL:
		case T_CIMG_DECL:			/* reeval_decl_stat */
		case T_CVEC_DECL:
		case T_CSEQ_DECL:
			if( VN_CHILD(enp,0) == NULL ){
				/* the node should have the shape info */
				if( bad_reeval_shape(enp) ) return;

				/* We used to just copy in the dimensions we
				 * thought we needed, but that didn't work
				 * for column vectors!
				 * Can you see why not?
				 */

				for(i=0;i<N_DIMENSIONS;i++)
					SET_DIMENSION(dsp,i,SHP_TYPE_DIM(VN_SHAPE(enp),i) );
			} else {
				return;
			}
			break;

		default:
			missing_case(enp,"reeval_decl_stat");
			break;
	}

	/* First make sure that the context of this declaration is active */
	PUSH_ID_CONTEXT(VN_DECL_CTX(enp));
	idp = id_of(VN_STRING(enp));
	POP_ID_CONTEXT;

	assert( idp != NULL );
	assert( IS_OBJ_REF(idp) );

	dp=REF_OBJ(ID_REF(idp));
	assert( dp != NULL );

	/* the size may not be unknown, if we were able to determine
	 * it's size during the first scan, .e.g. LOAD, or a known obj
	 */
	if( ! UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
		return;
	}

	if( ID_DOBJ_CTX(idp) != NODE_DATA(QLIST_HEAD(LIST_OF_DOBJ_CONTEXTS)) ){
		context_pushed=1;
		push_dobj_context(ID_DOBJ_CTX(idp));
	} else context_pushed=0;

	delvec(dp);

	SET_REF_OBJ(ID_REF(idp), finish_obj_decl(enp,dsp,prec_p,decl_flags) );	/* reeval_decl_stat */

	if( context_pushed )
		pop_dobj_context();

} /* end reeval_decl_stat */

/* eval_obj_id
 *
 * returns a ptr to an Identifier.
 * This is called from eval_ptr_expr
 */

static Identifier *_eval_obj_id(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Identifier *idp;
	Data_Obj *dp;
	const char *s;

	switch(VN_CODE(enp)){
		case T_EQUIVALENCE:		/* eval_obj_id() */
		case T_SUBSAMP:			/* eval_obj_id() */
		case T_CSUBSAMP:		/* eval_obj_id() */
		case T_SUBVEC:			/* eval_obj_id() */
		case T_CSUBVEC:			/* eval_obj_id() */
		case T_SQUARE_SUBSCR:		/* eval_obj_id() */
		case T_CURLY_SUBSCR:		/* eval_obj_id() */
			/* Subvectors & subscripted objects normally don't have permenent identifiers.
			 * We'll make one here from the object itself, but this creates a big
			 * problem with memory getting eaten up.  This was solved in the data library
			 * by having a small pool of temporary objects - but how do we know how
			 * many is enough here???
			 */
			dp = eval_obj_ref(enp);
			assert( dp != NULL );

			/* now make an identifier to go with this thing */
			idp = make_named_reference(OBJ_NAME(dp));
			SET_REF_OBJ(ID_REF(idp), dp );
			set_id_shape(idp, OBJ_SHAPE(dp) );
			return(idp);

		case T_OBJ_LOOKUP:
			s = eval_string(VN_CHILD(enp,0));
			goto find_obj;

		case T_STATIC_OBJ:			/* eval_obj_id */
			s=OBJ_NAME(VN_OBJ(enp));
			goto find_obj;

		case T_DYN_OBJ:				/* eval_obj_id */
			s=VN_STRING(enp);
			/* fall-thru */
find_obj:
			idp = id_of(s);
			assert( idp != NULL );
			assert( IS_OBJ_REF(idp) );

			return(idp);

		case T_STRING:
			/* make a local string name */
			{
				Dimension_Set ds1, *dsp=(&ds1);

				STRING_DIMENSION(dsp,(dimension_t)strlen(VN_STRING(enp))+1);
				dp=make_local_dobj(dsp,prec_for_code(PREC_STR),NULL);
				if( dp == NULL ){
					warn("unable to make temporary object");
					return(NULL);
				}
				strcpy((char *)OBJ_DATA_PTR(dp),VN_STRING(enp));
				idp = make_named_reference(OBJ_NAME(dp));
				SET_REF_TYPE(ID_REF(idp), STR_REFERENCE );
				SET_REF_OBJ(ID_REF(idp), dp );
				if( idp == NULL ){
					error1("error making identifier for temp string obj");
					IOS_RETURN_VAL(NULL)
				}
				return(idp);
			}

		default:
			missing_case(enp,"eval_obj_id");
			break;
	}
	return(NULL);
}

/*
 * Two ways to call eval_ptr_expr:
 * when a ptr is dereferenced, or appears on the RHS, it must be set!
 */

Identifier *_eval_ptr_expr(QSP_ARG_DECL Vec_Expr_Node *enp,int expect_ptr_set)
{
	Identifier *idp;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_EQUIVALENCE:
			idp = eval_obj_id(enp);
			return(idp);
			break;

		case T_CALLFUNC:		/* a function that returns a pointer */
			return( exec_reffunc(enp) );

		case T_REFERENCE:
			idp = eval_obj_id(VN_CHILD(enp,0));
			assert( idp != NULL );

			return( idp );

		case T_UNDEF:
			return(NULL);

		case T_POINTER:		/* eval_ptr_expr */
		case T_STR_PTR:		/* eval_ptr_expr */
			idp = get_id(VN_STRING(enp));
			assert( idp != NULL );

			/* BUG this is not an error if the ptr is on the left hand side... */
			if( executing && expect_ptr_set ){
				if( IS_POINTER(idp) && !POINTER_IS_SET(idp) ){
					node_error(enp);
					sprintf(ERROR_STRING,"object pointer \"%s\" used before value is set",ID_NAME(idp));
					advise(ERROR_STRING);
				} else if( IS_STRING_ID(idp) && !STRING_IS_SET(idp) ){
					node_error(enp);
					sprintf(ERROR_STRING,"string pointer \"%s\" used before value is set",ID_NAME(idp));
					advise(ERROR_STRING);
				}
			}
			return(idp);
			break;
		default:
			missing_case(enp,"eval_ptr_expr");
			break;
	}
	return(NULL);
} /* end eval_ptr_rhs */

Identifier *_get_set_ptr(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Identifier *idp;

	idp = eval_ptr_expr(enp,EXPECT_PTR_SET);
	assert( idp != NULL );
	assert( IS_POINTER(idp) );

	if( ! POINTER_IS_SET(idp) )
		return(NULL);

	return(idp);
}

// Currently, it seems that we don't allow subvectors with curly braces?
// But that is T_CSUBVEC!?

#define eval_subvec(dp, index, i2 ) _eval_subvec(QSP_ARG  dp, index, i2 )

static Data_Obj *_eval_subvec(QSP_ARG_DECL  Data_Obj *dp, index_t index, index_t i2 )
{
	//Dimension_Set dimset;
	Dimension_Set ds1, *dsp=(&ds1);
	index_t offsets[N_DIMENSIONS];
	char newname[LLEN];
	Data_Obj *dp2;
	int i;

	COPY_DIMS(dsp,OBJ_TYPE_DIMS(dp));

	for(i=0;i<N_DIMENSIONS;i++){
		offsets[i]=0;
	}
	SET_DIMENSION(dsp, OBJ_RANGE_MAXDIM(dp) , i2+1-index );
	offsets[ OBJ_RANGE_MAXDIM(dp) ] = index;
	sprintf(newname,"%s[%d:%d]",OBJ_NAME(dp),index,i2);
	dp2=dobj_of(newname);
	if( dp2 != NULL ) return(dp2);

	dp2=mk_subseq(newname,dp,offsets,dsp);
	if( dp2 == NULL ) return(dp2);
	// We used to decrement maxdim here, but that is wrong...
	// Because when we do v[0:2]=[1,2,3], v must be subscripted...
	// We decrement here so that we can take a subimage of an image x 
	// thusly:  x[0:3][0:2]
	// What is the best solution?
	// We could let mindim and maxdim revert to being the first (last)
	// non-1 dimensions...  then we would need a new field for the next
	// subscriptable dimension.  When we copy an object, the subscriptable
	// dimensions would need to be reset to mindim/maxdim...
	//
	// Solution is to introduce si_range_maxdim and use it instead of si_maxdim

	SET_OBJ_RANGE_MAXDIM(dp2,OBJ_RANGE_MAXDIM(dp)-1);
	return(dp2);
}

// What does eval_subscript1 do???
// It appears that this is only used for matlab emulation?

static Data_Obj *_eval_subscript1(QSP_ARG_DECL Data_Obj *dp, Vec_Expr_Node *enp)
{
	index_t index,index2;
	Data_Obj *dp2;

	if( VN_CODE(enp) == T_RANGE ){
		/* T_RANGE has 3 children, and is used to specify subsamping:
		 * start : end : inc
		 */
		warn("eval_subscript1:  Sorry, not sure how to handle T_RANGE node");
		return(NULL);
	} else if( VN_CODE(enp) == T_RANGE2 ){
		index = (index_t) eval_int_exp(VN_CHILD(enp,0));
		index2 = (index_t) eval_int_exp(VN_CHILD(enp,1));
		/* Now we need to make a subvector */
		dp2 = eval_subvec(dp,index-1,index2-1) ;
		return(dp2);
	}

	/* index = eval_int_exp(VN_CHILD(enp,1)); */
	index = (index_t) eval_flt_exp(enp);

	/* d_subscript fails if the index is too large,
	 * but in matlab we want to automagically make the array larger
	 */
	insure_object_size(dp,index);

	dp2 = d_subscript(dp,index);
	return( dp2 );
}

/* make something new */

static Data_Obj *create_matrix(QSP_ARG_DECL Vec_Expr_Node *enp,Shape_Info *shpp)
{
	Data_Obj *dp;
	Identifier *idp;

	switch(VN_CODE(enp)){
		case T_RET_LIST:
			return( create_list_lhs(enp) );

		case T_DYN_OBJ:		/* create_matrix */
			/* we need to create an identifier too! */
			idp = make_named_reference(VN_STRING(enp));
			dp = make_dobj(VN_STRING(enp),SHP_TYPE_DIMS(shpp),SHP_PREC_PTR(shpp));
			assert( dp != NULL );

			SET_REF_OBJ(ID_REF(idp), dp );
			set_id_shape(idp, OBJ_SHAPE(dp) );
			return(dp);
		default:
			missing_case(enp,"create_matrix");
			break;
	}
	return(NULL);
}

/* For matlab, if the rhs shape is different, then we reshape the LHS to match.
 * (for other languages, this might be an error!)
 * The node passed is generally the assign node...
 */

static Data_Obj *mlab_target(QSP_ARG_DECL Data_Obj *dp, Vec_Expr_Node *enp)
{
	if( dp == NULL ){
		dp = CREATE_MATRIX(enp,VN_SHAPE(enp));
	}
	else {
	/* BUG should check reshape if already exists */
		sprintf(ERROR_STRING,"mlab_target %s:  not checking reshape",OBJ_NAME(dp));
	  	warn(ERROR_STRING);
	}
	return(dp);
}

static Data_Obj *_create_list_lhs(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp1,*dp2;
	List *lp;
	Node *np1,*np2;

	dp1 = eval_obj_ref(VN_CHILD(enp,0));
	dp1 = MLAB_TARGET(dp1,VN_CHILD(enp,0));
	dp2 = eval_obj_ref(VN_CHILD(enp,1));
	dp2 = MLAB_TARGET(dp2,VN_CHILD(enp,1));
	np1=mk_node(dp1);
	np2=mk_node(dp2);
	lp=new_list();
	addTail(lp,np1);
	addTail(lp,np2);
	dp1=make_obj_list(localname(),lp);
	return(dp1);
}

/* NUMLIST trees grow down to the left.  We do this recursively
 * for simplicity, assuming that we will never have a very
 * big depth...
 *
 * For a 2D array, we have a tree of ROW_LIST's of the rows, each row
 * is a LIST_OBJ, below each LIST_OBJ there are more ROW_LIST's,
 * eventually reaching a literal or an object.
 * To get this to work correctly, we need to subscript dp when we descend a LIST_OBJ...
 */

static dimension_t _assign_obj_from_list(QSP_ARG_DECL Data_Obj *dp,Vec_Expr_Node *enp,index_t index)
{
	dimension_t i1,i2;
	Data_Obj *sub_dp,*src_dp;
	double dval;
	Scalar_Value sval;

	eval_enp = enp;

	assert(dp!=NULL);

	if( UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
		/* does the rhs have a shape??? */
		/* Why haven't we resolved when we are here? */
		warn("assign_obj_from_list:  LHS has unknown shape!?");
	}

	switch(VN_CODE(enp)){
		case T_TYPECAST:			/* assign_obj_from_list */
			/* do we need to do anything??? BUG? */
			i1=assign_obj_from_list(dp,VN_CHILD(enp,0),index);
			return(i1);
			break;

		case T_COMP_OBJ:
		case T_LIST_OBJ:		/* assign_obj_from_list */
			/* Old comment: */
			/* this node occurs only once, and has T_ROW_LIST children */
			/* New comment:
			 * We see LIST_OBJ's for the rows of a 2D object... (and
			 * presumably higher dimensional objects as well)
			 * Each T_LIST_OBJ represents a dimension...
			 * To figure out the shape one must start at the bottom
			 * and count up.  (But what about components?)
			 */
			/* If the child is a row_list, then we don't need to subscript here */
			/*i1=*/ assign_obj_from_list(dp,VN_CHILD(enp,0),0);
			/* why return 1 and not i1??? */
			return(1);
			break;

		case T_ROW_LIST:			/* assign_obj_from_list */
			/* HERE is where we need to be subscripting dp->dt_parent... */
			/* Don't subscript if the child is another ROW_LIST node */
			/* If we knew that the tree grew to the right or the left,
			 * we could eliminate the child tests - I am so lazy!
			 */
//advise("assign_obj_from_list T_ROW_LIST");
			if( VN_CODE(VN_CHILD(enp,0)) == T_ROW_LIST ){
				i1=assign_obj_from_list(dp,VN_CHILD(enp,0),index);
			} else {
				// If it's not a row_list node, then what is it???
				sub_dp = d_subscript(dp,index);
				i1=assign_obj_from_list(sub_dp,VN_CHILD(enp,0),index);
				delvec(sub_dp);
			}


			if( VN_CODE(VN_CHILD(enp,1)) == T_ROW_LIST ){
				i2=assign_obj_from_list(dp,VN_CHILD(enp,1),index+i1);
			} else {
				sub_dp = d_subscript(dp,index+i1);
				i2=assign_obj_from_list(sub_dp,VN_CHILD(enp,1),index+i1);
				delvec(sub_dp);
			}
			return(i1+i2);
			break;

		case T_COMP_LIST:			/* assign_obj_from_list */
			/* HERE is where we need to be subscripting dp->dt_parent... */
			/* Don't subscript if the child is another COMP_LIST node */
			/* If we knew that the tree grew to the right or the left,
			 * we could eliminate the child tests - I am so lazy!
			 */
			if( VN_CODE(VN_CHILD(enp,0)) == T_COMP_LIST ){
				i1=assign_obj_from_list(dp,VN_CHILD(enp,0),index);
			} else {
				sub_dp = c_subscript(dp,index);
				i1=assign_obj_from_list(sub_dp,VN_CHILD(enp,0),index);
				delvec(sub_dp);
			}


			if( VN_CODE(VN_CHILD(enp,1)) == T_COMP_LIST ){
				i2=assign_obj_from_list(dp,VN_CHILD(enp,1),index+i1);
			} else {
				sub_dp = c_subscript(dp,index+i1);
				i2=assign_obj_from_list(sub_dp,VN_CHILD(enp,1),index+i1);
				delvec(sub_dp);
			}
			return(i1+i2);
			break;

		case T_LIT_DBL:			/* assign_obj_from_list */
			dbl_to_scalar(&sval,VN_DBLVAL(enp),OBJ_MACH_PREC_PTR(dp));
assign_literal:
			if( ! IS_SCALAR(dp) ){
				node_error(enp);
				sprintf(ERROR_STRING,
	"assign_obj_from_list:  %s[%d] is not a scalar",OBJ_NAME(dp),
					index);
				warn(ERROR_STRING);
				return(1);
			}
			assign_scalar_obj(dp,&sval);
			return(1);
			break;

		case T_LIT_INT:				/* assign_obj_from_list */
			int_to_scalar(&sval,VN_INTVAL(enp),OBJ_MACH_PREC_PTR(dp));
			goto assign_literal;

		ALL_SCALAR_FUNCTION_CASES
		ALL_SCALAR_BINOP_CASES
			/* we allow arbitrary expressions within braces. */
			dval = eval_flt_exp(enp);
			dbl_to_scalar(&sval,dval,OBJ_MACH_PREC_PTR(dp));
			goto assign_literal;

		case T_STATIC_OBJ:	/* assign_obj_from_list */
		case T_DYN_OBJ:		/* assign_obj_from_list */
		case T_VS_FUNC:
		case T_VV_FUNC:
			src_dp = eval_obj_exp(enp,NULL);
			if( src_dp==NULL){
				node_error(enp);
				sprintf(ERROR_STRING,
			"assign_obj_from_list:  error evaluating RHS");
				warn(ERROR_STRING);
				return 0;
			}
			/* do we need to make sure they are the same size??? */
			//setvarg2(oap,dp,src_dp);
			dp_convert(dp,src_dp);
			return(1);
			break;

		default:
			missing_case(enp,"assign_obj_from_list");
			break;
	}
warn("assign_obj_from_list returning 0!?");
	return 0;
}

Data_Obj *_eval_obj_ref(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp,*dp2;
	index_t index,i2;
	index_t offsets[N_DIMENSIONS]={0,0,0,0,0};
	Dimension_Set ds1, *dsp=(&ds1);
	int i;
	char newname[LLEN];
	const char *s;
	Identifier *idp;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_EQUIVALENCE:		/* eval_obj_ref() */
			if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
				resolve_tree(enp,NULL);
			}
			if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
				node_error(enp);
				warn("unable to determine shape of equivalence");
dump_tree(enp);
				return(NULL);
			}
			{
			Data_Obj *parent_dp;

			/* we assume that the shape is known... */
			parent_dp = eval_obj_ref(VN_CHILD(enp,0));
			if( parent_dp == NULL ){
				node_error(enp);
				warn("invalid parent object for equivalence");
				return(NULL);
			}
			dp = make_equivalence(localname(), parent_dp,SHP_TYPE_DIMS(VN_SHAPE(enp)),VN_DECL_PREC(enp));
			if( dp == NULL ){
				node_error(enp);
				warn("unable to create equivalence");
			}
			return(dp);
			}
			break;

		case T_RET_LIST:		/* eval_obj_ref() */
			{
			/* BUG - what if this is not the LHS??? */
			return( create_list_lhs(enp) );
			break;
			}
			
		case T_COMP_OBJ:	/* eval_obj_ref */
		case T_LIST_OBJ:	/* eval_obj_ref */
			/* we seem to need this when we have typecast a list object... */

			/* We declare a temporary object and then assign it.
			 * This is rather inefficient, but we don't expect to do it often
			 * or for large objects.
			 */
			dp=make_local_dobj(SHP_TYPE_DIMS(VN_SHAPE(VN_CHILD(enp,0))), VN_PREC_PTR(enp),NULL);
			assign_obj_from_list(dp,VN_CHILD(enp,0),0);
			//SET_OBJ_FLAG_BITS(dp, DT_ASSIGNED);
			note_assignment(dp);
			return(dp);
			break;

		/* matlab */
		case T_SUBSCRIPT1:	/* eval_obj_ref */
			/* it seems that d_subscript knows about indices starting at 1? */
			dp = eval_obj_ref(VN_CHILD(enp,0));
			if( dp == NULL ) return(dp);

			/* In matlab, we can have a list of indices inside
			 * the paren's...  We need to know if the list trees
			 * grow to the left or right??
			 * SHould this code be matlab only? BUG?
			 */
			while( VN_CODE(VN_CHILD(enp,1)) == T_INDEX_LIST ){
				enp=VN_CHILD(enp,1);
				dp2 = eval_subscript1(dp,VN_CHILD(enp,0));
				if( dp2 == NULL ){
					return(dp2);
				}
				dp=dp2;
			}
			/* BUG doesn't enforce reference to an existing object!? */
			dp2=eval_subscript1(dp,VN_CHILD(enp,1));
			return(dp2);
			break;

		case T_PREINC:			/* eval_obj_ref */
		case T_PREDEC:
		case T_POSTINC:
		case T_POSTDEC:
			return( eval_obj_ref( VN_CHILD(enp,0) ) );

		case T_DEREFERENCE:	/* eval_obj_ref */
			/* runtime resolution, we may not be able to do this until ptrs have been assigned */
			if( UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_obj_ref:  last ditch attempt at runtime resolution of %s",node_desc(VN_CHILD(enp,0)));
advise(ERROR_STRING);
dump_tree(enp);
}
#endif /* QUIP_DEBUG */
				/*
				resolve_one_uk_node(VN_CHILD(enp,0));
				*/
				resolve_tree(VN_CHILD(enp,0),NULL);
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_obj_ref:  after last ditch attempt at runtime resolution of %s",node_desc(VN_CHILD(enp,0)));
advise(ERROR_STRING);
dump_tree(enp);
}
#endif /* QUIP_DEBUG */
			}

			idp = get_set_ptr(VN_CHILD(enp,0));
			if( idp == NULL ){
				/* actually , this could just be used before set? */
				/* We also get here if we use an object expression in a declaration,
				 * which can't be evaluated until run-time...
				 */

/*
node_error(enp);
warn("eval_obj_ref DEREFERENCE:  missing identifier!?");
dump_tree(enp);
*/
				return(NULL);
			}
			/* At one time we did a CAUTIOUS check
			 * to see if the object was in the database,
			 * but that could fail because the pointed to object
			 * might be out of scope
			 * (e.g.  ptr to local object passed as subrt arg)
			 */

			assert( IS_POINTER(idp) );
			assert( PTR_REF(ID_PTR(idp)) != NULL );

			return(REF_OBJ(PTR_REF(ID_PTR(idp))));

		case T_OBJ_LOOKUP:
			s=eval_string(VN_CHILD(enp,0));
			if( s == NULL ) return(NULL);
			dp=get_id_obj(s,enp);
			return(dp);

		case T_UNDEF:
			return(NULL);

		case T_REFERENCE:
			return( eval_obj_ref(VN_CHILD(enp,0)) );

		case T_STATIC_OBJ:		/* eval_obj_ref() */
			return(VN_OBJ(enp));

		case T_DYN_OBJ:		/* eval_obj_ref */
			return( get_id_obj(VN_STRING(enp),enp) );

		case T_CURLY_SUBSCR:				/* eval_obj_ref */
			dp = eval_obj_ref(VN_CHILD(enp,0));
			if( dp == NULL ) return(dp);
			index = (index_t) eval_int_exp(VN_CHILD(enp,1));
			dp=c_subscript(dp,index);
			return(dp);

		case T_SQUARE_SUBSCR:				/* eval_obj_ref */
			dp = eval_obj_ref(VN_CHILD(enp,0));
			if( dp == NULL ) return(dp);
			/* Before we evaluate the subscript as an integer, check and
			 * see if it's a vector...
			 */
			if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) ){
				index = (index_t) eval_int_exp(VN_CHILD(enp,1));
				dp = d_subscript(dp,index);
				return( dp );
			} else {
				node_error(enp);
				warn("eval_obj_ref:  vector indices are not allowed");
			}
			break;

		case T_SUBSAMP:					/* eval_obj_ref */
			{
			//Dimension_Set *ss_dsp;
			//Dimension_Set ds1;	// BUG not good if these are ObjC objects...
			//index_t ss_offsets[N_DIMENSIONS]={0,0,0,0,0};
			incr_t incrs[N_DIMENSIONS]={1,1,1,1,1};
			incr_t inc;
			char tmp_name[LLEN];
			Data_Obj *sub_dp;

			dp = eval_obj_ref(VN_CHILD(enp,0));
			if( dp == NULL ) return(dp);

			/* is this matlab order??? */
			index = (index_t) eval_int_exp(VN_CHILD(VN_CHILD(enp,1),0));	/* start */
			if( mode_is_matlab ){
				/* start : inc : end */
				inc = (incr_t) eval_int_exp(VN_CHILD(VN_CHILD(enp,1),1));
				i2 = (index_t) eval_int_exp(VN_CHILD(VN_CHILD(enp,1),2));	/* end */
			} else {
				/* start : end : inc */
				i2 = (index_t) eval_int_exp(VN_CHILD(VN_CHILD(enp,1),1));	/* end */
				inc = (incr_t) eval_int_exp(VN_CHILD(VN_CHILD(enp,1),2));
			}

			sprintf(tmp_name,"%s[%d:%d:%d]",OBJ_NAME(dp),index,inc,i2);
			sub_dp = dobj_of(tmp_name);
			if( sub_dp != NULL )
				return(sub_dp);		/* already exists */

			dsp = &ds1;
			COPY_DIMS(dsp , OBJ_TYPE_DIMS(dp) );
			SET_DIMENSION(dsp,OBJ_RANGE_MAXDIM(dp),1+(dimension_t)floor((i2-index)/inc) );	/* BUG assumes not reversed */
			offsets[OBJ_RANGE_MAXDIM(dp)] = index;
			incrs[OBJ_RANGE_MAXDIM(dp)] = inc;
			/* If we have referred to this before, the object may still exist */
			sub_dp = make_subsamp(tmp_name,dp,dsp,offsets,incrs);

			if( sub_dp == NULL ) return( sub_dp );
			SET_OBJ_RANGE_MAXDIM(sub_dp,
				OBJ_RANGE_MAXDIM(sub_dp)-1 );
			/* BUG?  make sure not less than mindim? */
			return( sub_dp );
			}
			break;
		case T_SUBVEC:					/* eval_obj_ref */
			dp = eval_obj_ref(VN_CHILD(enp,0));
			if( dp == NULL ) return(dp);

			// First make sure that we have a dimension available
			if( OBJ_RANGE_MAXDIM(dp) < OBJ_RANGE_MINDIM(dp) ){
				node_error(VN_CHILD(enp,0));
				sprintf(ERROR_STRING,
		"Can't specify range for object %s!?",OBJ_NAME(dp));
				warn(ERROR_STRING);
				return(NULL);
			}

			if( VN_CHILD(enp,1) == NULL )
				index = 0;	/* first element by default */
			else
				index = (index_t) eval_int_exp(VN_CHILD(enp,1));

			if( VN_CHILD(enp,2) == NULL )
				i2 = OBJ_TYPE_DIM(dp,OBJ_RANGE_MAXDIM(dp)) - 1;	/* last elt. */
			else
				i2 = (index_t) eval_int_exp(VN_CHILD(enp,2));

			return( eval_subvec(dp,index,i2) );
			break;
		case T_CSUBVEC:					/* eval_obj_ref */
			// Why is this so different from T_SUBVEC???
			// because eval_subvec is shared w/ matlab?...
			dp = eval_obj_ref(VN_CHILD(enp,0));
			if( dp == NULL ) return(dp);
			index = (index_t) eval_int_exp(VN_CHILD(enp,1));
			i2 = (index_t) eval_int_exp(VN_CHILD(enp,2));
			COPY_DIMS(dsp,OBJ_TYPE_DIMS(dp));
			for(i=0;i<N_DIMENSIONS;i++){
				offsets[i]=0;
			}
			SET_DIMENSION(dsp, OBJ_RANGE_MINDIM(dp) , i2+1-index );
			offsets[ OBJ_RANGE_MINDIM(dp) ] = index;
			sprintf(newname,"%s{%d:%d}",OBJ_NAME(dp),index,i2);
			dp2=dobj_of(newname);
			if( dp2 != NULL ) return(dp2);

			dp2=mk_subseq(newname,dp,offsets,dsp);
			SET_OBJ_RANGE_MINDIM(dp2, OBJ_RANGE_MINDIM(dp)+1 );
			return(dp2);
			break;
		case T_REAL_PART:
			dp = eval_obj_ref(VN_CHILD(enp,0));
			if( dp == NULL ) return(dp);
			/* BUG make sure that the object is commplex! */
			return( c_subscript(dp,0) );
			break;
		case T_IMAG_PART:
			dp = eval_obj_ref(VN_CHILD(enp,0));
			if( dp == NULL ) return(dp);
			/* BUG make sure that the object is commplex! */
			return( c_subscript(dp,1) );
			break;

		default:
			missing_case(enp,"eval_obj_ref");
			break;
	}
	return(NULL);
} /* end eval_obj_ref() */

/*
 * get_2_operands
 *
 * enp is a binary operator node with two children that we must evaluate.
 * dst_dp points to the destination of the operator (or NULL), and can
 * be used for intermediate results, if the size is right...
 */

static void get_2_operands(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj **dpp1,Data_Obj **dpp2,Data_Obj *dst_dp)
{
	Data_Obj *dp1=NULL,*dp2;
	int is_bitmap=0;

	eval_enp = enp;

	/* first, a special case:  T_VS_B_CONDASS
	 * first arg is a bitmap.
	 */

	// BUG - bitmap ops use VN_BM_CODE, not VN_VFUNC_CODE...
	// But this is only called with T_VV_FUNC, T_MATH2_FN, and T_INNER...

	if( VN_VFUNC_CODE(enp) == FVVSSLCT ) is_bitmap=1;

	/* The argument dp will often be the destination...
	 * to avoid overwriting it before we are done with it,
	 * we check the lhs_ref counts...
	 *
	 * There is a not-so-horrible BUG here, in that if
	 * we have descended far into a subtree, we will
	 * continue to make these tests and create additional
	 * temp objs...  After doing it the first time, we don't
	 * need to again, but the results will be correct
	 * anyway, so we leave it wasteful until we can figure
	 * out smarter logic.
	 */

	if( VN_LHS_REFS(VN_CHILD(enp,1)) == 0 ){
		/* the right hand subtree makes no ref to the lhs,
		 * so we're ok
		 */
		if( is_bitmap ){
			dp1=eval_bitmap(NULL,VN_CHILD(enp,0));
		} else if( dst_dp!=NULL && same_shape(VN_SHAPE(VN_CHILD(enp,0)),OBJ_SHAPE(dst_dp)) ){
			dp1=eval_obj_exp(VN_CHILD(enp,0),dst_dp);
		} else {
			dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
		}

		if( dst_dp!=NULL && dp1!=dst_dp && same_shape(VN_SHAPE(VN_CHILD(enp,1)),OBJ_SHAPE(dst_dp)) ){
			dp2=eval_obj_exp(VN_CHILD(enp,1),dst_dp);
		} else {
			dp2=eval_obj_exp(VN_CHILD(enp,1),NULL);
		}
	} else if( VN_LHS_REFS(VN_CHILD(enp,0)) == 0 ){
		/* the right hand subtree  refers to the lhs...
		 * but the left-hand subtree does not.
		 * we can proceed as above, but with r & l
		 * interchanged.
		 */
		if( dst_dp!=NULL && same_shape(VN_SHAPE(VN_CHILD(enp,1)),OBJ_SHAPE(dst_dp)) ){
			dp2=eval_obj_exp(VN_CHILD(enp,1),dst_dp);
		} else {
			dp2=eval_obj_exp(VN_CHILD(enp,1),NULL);
		}

		if( dp2!=dst_dp ){
			if( is_bitmap ){
				dp1=eval_bitmap(NULL,VN_CHILD(enp,0));
			} else if( dst_dp!=NULL && same_shape(VN_SHAPE(VN_CHILD(enp,0)),OBJ_SHAPE(dst_dp)) ){
				dp1=eval_obj_exp(VN_CHILD(enp,0),dst_dp);
			}
		} else {
			if( is_bitmap ){
				dp1=eval_bitmap(NULL,VN_CHILD(enp,0));
			} else {
				dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
			}
		}
	} else {
		/* Both sides refer to the lhs */
		if( is_bitmap )
			dp1=eval_bitmap(NULL,VN_CHILD(enp,0));
		else
			dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);

		/* used to have dst_dp here, would have added a test here for shape match,
		 * but it seems that if this branch refers to the lhs then we probably don't
		 * want to use the destination object?
		 */

		dp2=eval_obj_exp(VN_CHILD(enp,1),NULL);
	}

	assert( dp1 != NULL && dp2 != NULL );

	*dpp1 = dp1;
	*dpp2 = dp2;
} /* end get_2_operands() */

/* This is like eval_obj_assignment, except that the rhs is smaller that the LHS,
 * so we figure out which dimensions we need to iterate over, and then call eval_obj_assignment.
 * a bit wasteful perhaps...
 *
 * THis is kind of like projection?
 */

static void _eval_dim_assignment(QSP_ARG_DECL Data_Obj *dp,Vec_Expr_Node *enp)
{
	int i;

	for(i=N_DIMENSIONS-1;i>=0;i--){
		/* If the source only has one element in this dimension, but the
		 * destination has more, then
		 * we have to iterate.  Iteration over matching dimensions is
		 * done in eval_obj_assignment.
		 */
		if( OBJ_TYPE_DIM(dp,i) > 1 && SHP_TYPE_DIM(VN_SHAPE(enp),i)==1 ){
			Dimension_Set ds1, *dsp=(&ds1);
			Data_Obj *sub_dp;
			index_t dst_offsets[N_DIMENSIONS]={0,0,0,0,0};
			incr_t dst_incrs[N_DIMENSIONS]={1,1,1,1,1};
			dimension_t j;
			char tmp_dst_name[LLEN];
			char *base;

			COPY_DIMS(dsp , OBJ_TYPE_DIMS(dp) );
			SET_DIMENSION(dsp,i,1);
			sprintf(tmp_dst_name,"eda.%s",OBJ_NAME(dp));
			sub_dp=make_subsamp(tmp_dst_name,dp,dsp,dst_offsets,dst_incrs);

			/* Now copy each row (or whatever).
			 * Instead of making a new subobject, we just reset the
			 * data pointer ourselves - is this risky?
			 */

			base = (char *)OBJ_DATA_PTR(sub_dp);
			for(j=0;j<OBJ_TYPE_DIM(dp,i);j++){
				char *cp;

				cp = base;
				cp += j * OBJ_TYPE_INC(dp,i) * OBJ_MACH_PREC_SIZE(dp);
				SET_OBJ_DATA_PTR(sub_dp, cp);

				eval_dim_assignment(sub_dp,enp);
			}
			delvec(sub_dp);
			/* all the work done in the recursive calls */
			return;
		}
	}
	/* now we're ready! */
	eval_obj_assignment(dp,enp);
} /* end eval_dim_assignment */

static double scalar_to_double(Scalar_Value *svp,Precision *prec_p)
{
	double dval=0.0;
	switch( PREC_CODE(prec_p) &MACH_PREC_MASK ){
		case PREC_BY:  dval = svp->u_b; break;
		case PREC_UBY:  dval = svp->u_ub; break;
		case PREC_IN:  dval = svp->u_s; break;
		case PREC_UIN:  dval = svp->u_us; break;
		case PREC_DI:  dval = svp->u_l; break;
		case PREC_UDI: dval = svp->u_ul; break;	/* BIT precision handled elsewhere */
		case PREC_SP:  dval = svp->u_f; break;
		case PREC_DP:  dval = svp->u_d; break;
		default:
			assert( AERROR("scalar_to_double:  unhandled precision") );
			break;
	}
	return(dval);
}

double _eval_flt_exp(QSP_ARG_DECL Vec_Expr_Node *enp)
{
#ifdef SCALARS_NOT_OBJECTS
	Identifier *idp;
#endif // SCALARS_NOT_OBJECTS
	Data_Obj *dp,*dp2;
	double dval;
	double dval2;
	index_t index;
	Scalar_Value *svp;
	Subrt *srp;
	Dimension_Set ds1, *dsp=(&ds1);
	Vec_Obj_Args oa1, *oap=&oa1;

	SET_DIMENSION(dsp,0,1);
	SET_DIMENSION(dsp,1,1);
	SET_DIMENSION(dsp,2,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,4,1);

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_SUM:
			// create a destination object...
			dp2=eval_obj_exp(VN_CHILD(enp,0),NULL);
			dp=mk_scalar("tmp_sum",OBJ_PREC_PTR(dp2));
			clear_obj_args(oap);
			setvarg2(oap,dp,dp2);
			platform_dispatch_by_code(FVSUM, oap);
			dval = cast_from_scalar_value(OBJ_DATA_PTR(dp),OBJ_PREC_PTR(dp));
			delvec(dp);
			break;

#ifdef SCALARS_NOT_OBJECTS
		case T_SCALAR_VAR:		// eval_flt_exp
			idp = get_id(VN_STRING(enp));
			assert(idp!=NULL);
			dval = cast_from_scalar_value(ID_SVAL_PTR(idp), ID_PREC_PTR(idp));
			break;
#endif // SCALARS_NOT_OBJECTS
			
		case T_MINVAL:
			dp2=eval_obj_exp(VN_CHILD(enp,0),NULL);
if( dp2 == NULL ){
node_error(VN_CHILD(enp,0));
warn("error evaluating arg to min");
return(0.0);
}
			/* make a scalar object to hold the answer */
			dp=make_local_dobj(dsp,OBJ_PREC_PTR(dp2),OBJ_PFDEV(dp2));
			clear_obj_args(oap);
			setvarg2(oap,dp,dp2);
			//vminv(oap);
			//vf_code=FVMINV;
			//h_vl2_vminv(HOST_CALL_ARGS);
			platform_dispatch_by_code(FVMINV, oap );
			dval = get_dbl_scalar_value(dp);
			delvec(dp);
			break;

		case T_CALLFUNC:			/* eval_flt_exp */
			/* This could get called if we use a function inside a dimesion bracket... */
			if( ! executing ) return 0;

			srp=VN_SUBRT(enp);
			/* BUG SHould check and see if the return type is double... */

			/* BUG at least make sure that it's not void... */

			/* make a scalar object to hold the return value... */
			dp=make_local_dobj(dsp,SR_PREC_PTR(srp),NULL);
			exec_subrt(enp,dp);
			/* get the scalar value */
			dval = get_dbl_scalar_value(dp);
			delvec(dp);
			break;


#ifdef NOT_YET
		case T_CALL_NATIVE:
			dval = eval_native_flt(enp) ;
			break;
#endif /* NOT_YET */

		/* matlab */
		case T_INNER:
			/* assume both children are scalars */
			dval = eval_flt_exp(VN_CHILD(enp,0));
			dval2 = eval_flt_exp(VN_CHILD(enp,1));
			dval *= dval2 ;
			break;

		case T_SUBSCRIPT1:			/* eval_flt_exp */
			dp=get_obj(VN_STRING(VN_CHILD(enp,0)));
			index = (index_t) eval_flt_exp(VN_CHILD(enp,1));
			dp2 = d_subscript(dp,index);
			if( dp2 == NULL ){
				sprintf(ERROR_STRING,
		"Couldn't form subobject %s[%d]",OBJ_NAME(dp),index);
				warn(ERROR_STRING);
				dval = 0.0;
			} else {
				svp = (Scalar_Value *)OBJ_DATA_PTR(dp2);
				dval = scalar_to_double(svp,OBJ_PREC_PTR(dp2));
			}
			break;

		/* end matlab */


		case T_POWER:
			dval = eval_flt_exp(VN_CHILD(enp,0));
			dval2 = eval_flt_exp(VN_CHILD(enp,1));
			dval = pow(dval,dval2) ;
			break;

		case T_POSTINC:
			dp = eval_obj_ref(VN_CHILD(enp,0));
			dval = eval_flt_exp(VN_CHILD(enp,0));
			inc_obj(dp);
			break;

		case T_POSTDEC:
			dp = eval_obj_ref(VN_CHILD(enp,0));
			dval = eval_flt_exp(VN_CHILD(enp,0));
			dec_obj(dp);
			break;

		case T_PREDEC:
			dp = eval_obj_ref(VN_CHILD(enp,0));
			dec_obj(dp);
			dval = eval_flt_exp(VN_CHILD(enp,0));
			break;

		case T_PREINC:					/* eval_flt_exp */
			dp = eval_obj_ref(VN_CHILD(enp,0));
			inc_obj(dp);
			dval = eval_flt_exp(VN_CHILD(enp,0));
			break;

		case T_TYPECAST:	/* eval_flt_exp */
			/* We could just call eval_flt_exp on the child node,
			 * But if we are casting a float to int, we need to round...
			 */
			dval = eval_flt_exp(VN_CHILD(enp,0));
			switch(VN_PREC(enp)){
				case PREC_BY:   return( (double) ((char)    dval ) );
				case PREC_UBY:  return( (double) ((u_char)  dval ) );
				case PREC_IN:   return( (double) ((short)   dval ) );
				case PREC_UIN:  return( (double) ((u_short) dval ) );
				case PREC_DI:   return( (double) ((long)    dval ) );
				case PREC_UDI:  return( (double) ((u_long)  dval ) );
				case PREC_SP:   return( (double) ((float)   dval ) );
				case PREC_DP:   return(                     dval   );
				case PREC_BIT:
					if( dval != 0 )
						dval = 1.0;
					else
						dval = 0.0;
					break;
				default:
					assert( AERROR("eval_flt_exp:  unhandled precision") );
					dval = 0.0;
					break;
			}
			break;
		case T_UNDEF:
			dval=0.0;
			break;

		case T_STR2_FN:
			{
				const char *s1,*s2;
				s1=eval_string(VN_CHILD(enp,0));
				s2=eval_string(VN_CHILD(enp,1));
				if( s1 != NULL && s2 != NULL ){
					dval = (*FUNC_STR2_FUNC(VN_FUNC_PTR(enp)))(s1,s2);
				} else	{
					dval = 1;	/* the default is unequal strings */
				}
			}
			break;

		case T_STR1_FN:	/* eval_flt_exp */
			/* BUG?  should this really be an int expression? */
			{
			const char *str;
			str = eval_string(VN_CHILD(enp,0));
			if( str == NULL ){
				warn("error evaluating string...");
				dval=0.0;
				break;
			}
			dval = (*FUNC_STR1_FUNC(VN_FUNC_PTR(enp)))(QSP_ARG  str);
			return(dval);
			}

		// The whole section for T_SIZE_FN was commented out with ifdef NOT_YET
		// - why???

		case T_SIZE_FN:	/* eval_flt_exp */
			if( VN_CODE(VN_CHILD(enp,0)) == T_STRING ){
				/* name of a sizable object */
				Item *ip;
				ip = find_sizable(VN_STRING(VN_CHILD(enp,0)));
				if(ip==NULL){
					sprintf(ERROR_STRING,
						"Couldn't find sizable object %s",
						VN_STRING(VN_CHILD(enp,0)));
					warn(ERROR_STRING);
					dval=0.0;
					break;
				}
				dval = (*(FUNC_SZ_FUNC(VN_FUNC_PTR(enp))))(QSP_ARG  ip);
			} else {
				/* an objref expressions */
				int save_e;	/* objs don't need values to query their size */

				save_e = expect_objs_assigned;
				expect_objs_assigned=0;
				dp = eval_obj_exp(VN_CHILD(enp,0),NULL);
				expect_objs_assigned=save_e;

				if( dp == NULL ){
					/* This might not be an error if we have used an object
					 * expression as a dimension, e.g.
					 * float x[ncols(v)];
					 * where v is a subroutine argument...
					 */

					if( executing ){
						node_error(enp);
						sprintf(ERROR_STRING,
				"bad object expression given for function %s",
							FUNC_NAME(VN_FUNC_PTR(enp)));
						warn(ERROR_STRING);
dump_tree(enp);
					}
					dval = 0.0;	/* eval_flt_exp T_SIZE_FN */
					break;
				}
				if( UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
					/* An old comment:
					 *
					 * we might use the size of an unknown shape
					 * object as a dimension in a subrt arg,
					 * e.g. the size of one of the arguments,
					 * which can only be known at run-time...
					 * We'd like to leave this off until run-time,
					 * our solution here is to return a non-zero
					 * value (to suppress zero-dim warnings),
					 * that is greater than 1 (to make sure
					 * the right shape is determined).  The
					 * problem with this approach is that it
					 * might lead to spurious shape conflicts
					 * or incorrect shape resolutions...
					 * But we'll wait and see.
					 */

					/* return 0 to indicate that we don't know yet */
					node_error(enp);
					sprintf(ERROR_STRING,"returning 0 for size of unknown shape object %s",
						OBJ_NAME(dp));
					advise(ERROR_STRING);
					dval=0.0;
					break;
				}
				/* Originally, we called the function from the size_functbl here...
				 * this works for any type of sizable object, but in order to do
				 * an itemtype-specific function call, it has to determine what
				 * type of object the name refers to by searching each of the sizable
				 * object databases...  This creates a problem, because there can
				 * be pointed-to objects that have had their contexts popped
				 * because they are not in the current scope.  Because we know this
				 * is a data object, we just call the appropriate dobj-specific function.
				 */

				assert( FUNC_DIM_INDEX(VN_FUNC_PTR(enp)) >= 0 );
				dval = get_dobj_size(dp, FUNC_DIM_INDEX(VN_FUNC_PTR(enp)));
			}
			break;

		case T_SCALMAX:
			dval = eval_flt_exp(VN_CHILD(enp,0));
			dval2 = eval_flt_exp(VN_CHILD(enp,1));
			if( dval < dval2 )
				dval = dval2;
			break;

		case T_SCALMIN:
			dval = eval_flt_exp(VN_CHILD(enp,0));
			dval2 = eval_flt_exp(VN_CHILD(enp,1));
			if( dval > dval2 )
				dval = dval2;
			break;

		case T_STATIC_OBJ:	/* eval_flt_exp */
			dp=VN_OBJ(enp);
			goto obj_flt_exp;

		case T_POINTER:
		case T_DYN_OBJ:		/* eval_flt_exp */
			dp=get_obj(VN_STRING(enp));
			assert( dp != NULL );

obj_flt_exp:

			/* check that this object is a scalar */
			if( OBJ_N_TYPE_ELTS(dp) != 1 ){
				/* what about a complex scalar? BUG */
				node_error(enp);
				sprintf(ERROR_STRING,
		"eval_flt_exp:  object %s is not a scalar!?",OBJ_NAME(dp));
				warn(ERROR_STRING);
			}
			svp=(Scalar_Value *)OBJ_DATA_PTR(dp);
			if( svp == NULL ){
				node_error(enp);
				sprintf(ERROR_STRING,"object %s has null data ptr!?",OBJ_NAME(dp));
				advise(ERROR_STRING);
				dval = 0.0;
				break;
			}

			if( IS_BITMAP(dp) ){
				if( svp->u_ul & 1<<OBJ_BIT0(dp) )
					dval=1.0;
				else
					dval=0.0;
			} else {
				dval = scalar_to_double(svp,OBJ_PREC_PTR(dp));
			}
			break;

		/* BUG need T_CURLY_SUBSCR too! */
		case T_SQUARE_SUBSCR:			/* eval_flt_exp */
			/* dp=get_obj(VN_STRING(VN_CHILD(enp,0))); */
			dp = eval_obj_ref(VN_CHILD(enp,0));
			index = (index_t) eval_int_exp(VN_CHILD(enp,1));
			dp2 = d_subscript(dp,index);
			if( dp2 == NULL ){
				sprintf(ERROR_STRING,
		"Couldn't form subobject %s[%d]",OBJ_NAME(dp),index);
				warn(ERROR_STRING);
				dval=0.0;
			} else {
				svp = (Scalar_Value *)OBJ_DATA_PTR(dp2);
				dval = scalar_to_double(svp,OBJ_PREC_PTR(dp2));
			}
			break;

		case T_CURLY_SUBSCR:
			dp = eval_obj_ref(VN_CHILD(enp,0));
			index = (index_t) eval_int_exp(VN_CHILD(enp,1));
			dp2 = c_subscript(dp,index);
			if( dp2 == NULL ){
				sprintf(ERROR_STRING,
		"Couldn't form subobject %s[%d]",OBJ_NAME(dp),index);
				warn(ERROR_STRING);
				dval=0.0;
			} else {
				svp = (Scalar_Value *)OBJ_DATA_PTR(dp2);
				dval = scalar_to_double(svp,OBJ_PREC_PTR(dp2));
			}
			break;

		case T_MATH0_FN:
		dval = (*FUNC_D0_FUNC(VN_FUNC_PTR(enp)))();
			break;

		case T_MATH1_FN:
			dval = eval_flt_exp(VN_CHILD(enp,0));
		dval = (*FUNC_D1_FUNC(VN_FUNC_PTR(enp)))(dval);
			break;
		case T_MATH2_FN:				/* eval_flt_exp */
			dval = eval_flt_exp(VN_CHILD(enp,0));
			dval2 = eval_flt_exp(VN_CHILD(enp,1));
	dval = (*FUNC_D2_FUNC(VN_FUNC_PTR(enp)))(dval,dval2);
			break;
		case T_UMINUS:
			dval = eval_flt_exp(VN_CHILD(enp,0));
			dval *= (-1);
			break;
		case T_RECIP:
			dval = eval_flt_exp(VN_CHILD(enp,0));
			if( dval == 0.0 ){
				node_error(enp);
				sprintf(ERROR_STRING,"Divide by 0!?");
				warn(ERROR_STRING);
				dval=(0.0);
			} else {
				dval = 1.0/dval;
			}
			break;
		case T_LIT_DBL:
			dval=VN_DBLVAL(enp);
			break;
		case T_LIT_INT:
			dval=VN_INTVAL(enp);
			break;
		ALL_SCALINT_BINOP_CASES
			dval=eval_int_exp(enp);
			break;
		case T_DIVIDE:
			dval=eval_flt_exp(VN_CHILD(enp,0));
			dval2=eval_flt_exp(VN_CHILD(enp,1));
			if( dval2==0.0 ){
				node_error(enp);
				sprintf(ERROR_STRING,"Divide by 0!?");
				warn(ERROR_STRING);
				dval=(0.0);
				break;
			}
			dval/=dval2;
			break;
		case T_PLUS:
			dval=eval_flt_exp(VN_CHILD(enp,0));
			dval2=eval_flt_exp(VN_CHILD(enp,1));
			dval+=dval2;
			break;
		case T_MINUS:
			dval=eval_flt_exp(VN_CHILD(enp,0));
			dval2=eval_flt_exp(VN_CHILD(enp,1));
			dval-=dval2;
			break;
		case T_TIMES:
			dval=eval_flt_exp(VN_CHILD(enp,0));
			dval2=eval_flt_exp(VN_CHILD(enp,1));
			dval*=dval2;
			break;

		default:
			missing_case(enp,"eval_flt_exp");
			dval=0.0;	// make sure there is a value to return
			break;
	}

	//return(0.0);		// why return 0.0?
	return(dval);
}

#define INSURE_DESTINATION						\
									\
			if( dst_dp == NULL ){				\
				dst_dp=make_local_dobj(			\
					SHP_TYPE_DIMS(VN_SHAPE(enp)),	\
					SHP_PREC_PTR(VN_SHAPE(enp)),	\
					NULL);		\
			}



/* Evalutate an object expression.
 * Unlike object assignments, if the expression is an object
 * reference, we don't copy, we just return the pointer.
 * If we need to store some results, we put them in dst_dp,
 * or if dst_dp is null we create a temporary object.
 */

Data_Obj *_eval_obj_exp(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj *dst_dp)
{
	Data_Obj *dp;
	long ldx,ldy;

	/* We don't do more complex things here, because we don't
	 * want to worry about creating temporary objects...
	 */

	eval_enp = enp;

	switch(VN_CODE(enp)){
		ALL_CONDASS_CASES
		case T_SUM:
		case T_INNER:
			INSURE_DESTINATION
			eval_obj_assignment(dst_dp,enp);
			return(dst_dp);
			break;

		case T_RAMP:
			{
			/* We should distinguish between float and int ramps?  BUG */
			float start, dx, dy;
			start=(float)eval_flt_exp(VN_CHILD(enp,0));
			dx=(float)eval_flt_exp(VN_CHILD(enp,1));
			dy=(float)eval_flt_exp(VN_CHILD(enp,2));
			INSURE_DESTINATION
			easy_ramp2d(dst_dp,start,dx,dy);
			return(dst_dp);
			break;
			}

		case T_TRANSPOSE:
			dp=eval_obj_exp(VN_CHILD(enp,0),NULL);
			if( dp == NULL ) break;
			/* BUG make sure valid??? */
			INSURE_DESTINATION
			xpose_data(dst_dp,dp);
			return(dst_dp);
			break;

		case T_SCROLL:		/* eval_obj_exp */
			dp = eval_obj_exp(VN_CHILD(enp,0),NULL);
			ldx=eval_int_exp(VN_CHILD(enp,1));
			ldy=eval_int_exp(VN_CHILD(enp,2));
			assert( dp != NULL );

			/* BUG? do we need to make sure that dp is not dst_dp? */
			INSURE_DESTINATION
			dp_scroll(dst_dp,dp,(incr_t)ldx,(incr_t)ldy);
			return(dst_dp);


		case T_WRAP:				/* eval_obj_exp */
			dp = eval_obj_exp(VN_CHILD(enp,0),NULL);
			assert( dp != NULL );

			/* BUG? do we need to make sure that dp is not dst_dp? */
			if( dst_dp == NULL ){
				dst_dp=make_local_dobj(
					SHP_TYPE_DIMS(VN_SHAPE(enp)),
					SHP_PREC_PTR(VN_SHAPE(enp)),
					OBJ_PFDEV(dp));
			}
			wrap(dst_dp,dp);
			return(dst_dp);

		case T_TYPECAST:			/* eval_obj_exp */
			/* The requested type should match the destination,
			 * and not match the child node...  this is supposed
			 * to be insured by the compilation process.
			 */
			return( eval_typecast(enp,dst_dp) );

		case T_UNDEF:
			return(NULL);


		/* We used to call eval_obj_ref() for all OBJREF cases,
		 * but that was wrong once we introduced the possibility
		 * of vector subscripts...
		 */

		case T_STATIC_OBJ:		/* eval_obj_exp */
			return(VN_OBJ(enp));


		case T_COMP_OBJ:		/* eval_obj_exp */
		case T_LIST_OBJ:
		case T_OBJ_LOOKUP:
		case T_DEREFERENCE:
		case T_DYN_OBJ:		/* eval_obj_exp */
		case T_SUBSAMP:
		case T_CSUBSAMP:
		case T_SUBVEC:
		case T_CSUBVEC:
		case T_SUBSCRIPT1:
		case T_REAL_PART:
		case T_IMAG_PART:

			return( eval_obj_ref(enp) );
			break;

		case T_SQUARE_SUBSCR:
		case T_CURLY_SUBSCR:
			/* BUG - need separate code for the two types of subscript!? */
			dp = eval_obj_ref(VN_CHILD(enp,0));
			if( dp == NULL ) return(dp);

			/* Before we evaluate the subscript as an integer, check and
			 * see if it's a vector...
			 */
			if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) ){
				index_t index;
				index = (index_t)eval_int_exp(VN_CHILD(enp,1));
				if( VN_CODE(enp) == T_SQUARE_SUBSCR )
					return( d_subscript(dp,index) );
				else
					return( c_subscript(dp,index) );
			} else {
				Data_Obj *index_dp;
				index_dp=eval_obj_ref(VN_CHILD(enp,1));
				if( index_dp == NULL ) break;	/* BUG?  print error here? */
				if( OBJ_COMPS(index_dp) != (dimension_t)(1+OBJ_MAXDIM(dp)-OBJ_MINDIM(dp)) ){
					node_error(enp);
					sprintf(ERROR_STRING,
	"Map source object %s needs %d indices, but index array %s has component dimension %d!?",
						OBJ_NAME(dp),1+OBJ_MAXDIM(dp)-OBJ_MINDIM(dp),
						OBJ_NAME(index_dp),OBJ_COMPS(index_dp));
					warn(ERROR_STRING);
				} else {
					return( MAP_SUBSCRIPTS(dp,index_dp,enp) );
				}
			}
			break;

#ifdef MATLAB_FOOBAR
		/* matlab */
		case T_SUBSCRIPT1:	/* eval_obj_exp */
			dp=get_obj(VN_STRING(VN_CHILD(enp,0)));
			index = eval_flt_exp(VN_CHILD(enp,1));
			dp2 = d_subscript(dp,index);
			return(dp2);
#endif /* MATLAB_FOOBAR */



		case T_CALLFUNC:

		case T_RIDFT:
		case T_RDFT:
		case T_VV_FUNC:
		case T_VS_FUNC:
		ALL_MATH_VFN_CASES			/* eval_obj_exp */
			if( dst_dp!=NULL ){
				eval_obj_assignment(dst_dp,enp);
				return(dst_dp);
			} else {
				/* We need to create a temporary object to
				 * hold the result...   hopefully we know
				 * the shape at this node!
				 */
				assert( VN_SHAPE(enp) != NULL );
if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
dump_tree(enp);
}
				assert( ! UNKNOWN_SHAPE(VN_SHAPE(enp)) );

				dst_dp=make_local_dobj(SHP_TYPE_DIMS(VN_SHAPE(enp)),
							SHP_PREC_PTR(VN_SHAPE(enp)),
							NULL);
				assert( dst_dp != NULL );

				eval_obj_assignment(dst_dp,enp);
				return(dst_dp);
			}
			break;


		default:
			missing_case(enp,"eval_obj_exp");
			break;
	}
	return(NULL);
} /* end eval_obj_exp() */

static const char *bad_string="XXXbad_stringXXX";

/* Construct a string from the given tree.
 * At runtime, it is an error to get a missing identifier,
 * but at other times (e.g. dump_tree) we may not be able
 * to expand everything...
 */

const char *_eval_string(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	const char *s;
	const char *s1,*s2;
	Identifier *idp;
	Data_Obj *dp;
	int n;

	eval_enp = enp;

	switch(VN_CODE(enp)){
#ifdef NOT_YET
		case T_CALL_NATIVE:
			/*
			sprintf(ERROR_STRING,"eval_string T_CALL_NATIVE:  function %s, not handled...",
					FUNC_NAME(VN_FUNC_PTR(enp)));
			warn(ERROR_STRING);
			return("FOO");
			*/
			return( eval_native_string(enp) );
			break;
#endif /* NOT_YET */

		case T_SQUARE_SUBSCR:			/* eval_string */
		case T_CURLY_SUBSCR:
		case T_STATIC_OBJ:			/* eval_string */
		case T_DYN_OBJ:				/* eval_string */
			dp = eval_obj_exp(enp,NULL);
			if( dp == NULL ){
				node_error(enp);
				sprintf(ERROR_STRING,"eval_string:  missing object %s",VN_STRING(enp));
				warn(ERROR_STRING);
				return(NULL);
			}
			assert( OBJ_PREC(dp) == PREC_CHAR );

			/* not exactly a BUG, but we might verify that the number of
			 * columns matches the string length?
			 */
			return((char *)OBJ_DATA_PTR(dp));
			break;

		case T_SET_STR:
			if( dumping ) return(STRING_FORMAT);

			s = eval_string(VN_CHILD(enp,1));
			idp = eval_ptr_expr(VN_CHILD(enp,0),UNSET_PTR_OK);
			if( idp == NULL ) break;
			assign_string(idp,s,enp);
			return(s);

		case T_PRINT_LIST:
			return(eval_mixed_list(enp));

		case T_STRING_LIST:
			{
			char *new_string;
			s1=eval_string(VN_CHILD(enp,0));
			s2=eval_string(VN_CHILD(enp,1));
			if( s1 == NULL || s2 == NULL ) return(NULL);
			/* BUG need some garbage collection!? */
			n=(int)(strlen(s1)+strlen(s2)+1);
			new_string=(char *)getbuf(n);
			strcpy(new_string,s1);
			strcat(new_string,s2);
			return(new_string);
			}
			break;

		case T_STRING:
			s=VN_STRING(enp);
			break;

		case T_STRV_FN:			// eval_string - like T_SIZE_FN
			s="bad_strv_result";		// set default value
			if( VN_CODE(VN_CHILD(enp,0)) == T_STRING ){
				/* name of a sizable object */
				/*
				Item *ip;
				ip = find_sizable(VN_STRING(VN_CHILD(enp,0)));
				if(ip==NULL){
					sprintf(ERROR_STRING,
						"Couldn't find sizable object %s",
						VN_STRING(VN_CHILD(enp,0)));
					warn(ERROR_STRING);
					break;
				}
				*/
				s = (*(FUNC_STRV_FUNC(VN_FUNC_PTR(enp))))(QSP_ARG  VN_STRING(VN_CHILD(enp,0)) );
			} else {
				/* an objref expressions */
				int save_e;	/* objs don't need values to query their size */

				save_e = expect_objs_assigned;
				expect_objs_assigned=0;
				dp = eval_obj_exp(VN_CHILD(enp,0),NULL);
				expect_objs_assigned=save_e;

				if( dp == NULL ){
					/* This might not be an error if we have used an object
					 * expression as a dimension, e.g.
					 * float x[ncols(v)];
					 * where v is a subroutine argument...
					 */

					if( executing ){
						node_error(enp);
						sprintf(ERROR_STRING,
				"bad object expression given for function %s",
							FUNC_NAME(VN_FUNC_PTR(enp)));
						warn(ERROR_STRING);
dump_tree(enp);
					}
					break;
				}
				if( UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
					/* see comments for T_SIZE_FN... */
					/* return 0 to indicate that we don't know yet */
					node_error(enp);
					sprintf(ERROR_STRING,"Unknown shape object %s passed to function %s",
						OBJ_NAME(dp),FUNC_NAME(VN_FUNC_PTR(enp)) );
					advise(ERROR_STRING);
					break;
				}
				/*
				 * Comment from T_SIZE_FN:
				 * Originally, we called the function from the size_functbl here...
				 * this works for any type of sizable object, but in order to do
				 * an itemtype-specific function call, it has to determine what
				 * type of object the name refers to by searching each of the sizable
				 * object databases...  This creates a problem, because there can
				 * be pointed-to objects that have had their contexts popped
				 * because they are not in the current scope.  Because we know this
				 * is a data object, we just call the appropriate dobj-specific function.
				 *
				 * At this writing, there is only one STRV function (precision).
				 * But that is not a safe assumption!?
				 */

				assert( ! strcmp(FUNC_NAME(VN_FUNC_PTR(enp)),"precision") );

				s = OBJ_PREC_NAME(dp);
			}
#ifdef FOOBAR
			// Old code for toupper etc.
			s=eval_string( VN_CHILD(enp,0) );
			if( s == NULL ) return(NULL);
			// Where do we get the destination string?
			// We need a temp object!?
			dp=make_local_dobj(SHP_TYPE_DIMS(VN_SHAPE(VN_CHILD(enp,0))),
				VN_PREC_PTR(enp), NULL);
			(*FUNC_STRV_FUNC(VN_FUNC_PTR(enp)))(OBJ_DATA_PTR(dp),s);
			return s;
#endif // FOOBAR
			break;

		case T_STR_PTR:			/* eval_string */
			if( dumping ) return(STRING_FORMAT);

			idp = eval_ptr_expr(enp,EXPECT_PTR_SET);

			if( idp == NULL ){
				node_error(enp);
				sprintf(ERROR_STRING,"missing string pointer object %s",VN_STRING(enp));
				advise(ERROR_STRING);
				return(NULL);
			}

			assert( IS_STRING_ID(idp) );

			if( sb_buffer(REF_SBUF(ID_REF(idp))) == NULL ){
				node_error(enp);
				sprintf(ERROR_STRING,"string pointer \"%s\" used before set!?",ID_NAME(idp));
				warn(ERROR_STRING);
				return(NULL);
			} else
				s=sb_buffer(REF_SBUF(ID_REF(idp)));
			break;

		default:
			missing_case(enp,"eval_string");
			return(bad_string);
			break;
	}
	return(s);
}


/* for matlab support */

static void _insure_object_size(QSP_ARG_DECL  Data_Obj *dp,index_t index)
{
	int which_dim;

	which_dim = OBJ_MINDIM(dp);

	if( OBJ_TYPE_DIM(dp,which_dim) <= index ){ /* index is too big, we need to resize */
		Dimension_Set ds1, *dsp=(&ds1);
		index_t offsets[N_DIMENSIONS];
		Scalar_Value sval;
		Data_Obj *new_dp,*sub_dp;
		void *tmp_data;
		int i;

		/* first, get the new data area */

		COPY_DIMS(dsp, OBJ_TYPE_DIMS(dp) );

		for(i=0;i<N_DIMENSIONS;i++){
		/*
			dims[i]=OBJ_TYPE_DIM(dp,i);
		*/
			offsets[i]=0;
		}
		SET_DIMENSION(dsp,which_dim,index);

		new_dp = make_dobj("tmpname",dsp,OBJ_PREC_PTR(dp));

		/* set new data area to all zeroes */
		sval.u_d = 0.0;	/* BUG assumes PREC_DP */
		dp_const(new_dp,&sval);

		/* copy in original data */
		sub_dp = mk_subseq("tmp_subseq",new_dp,offsets,OBJ_TYPE_DIMS(dp));
		dp_copy(sub_dp,dp);

		/* get rid of the subimage */
		delvec(sub_dp);

		/* now this is tricky...  we want to swap data areas, and dimensions
		 * between new_dp and dp...  here goes nothing
		 */
		tmp_data = OBJ_DATA_PTR(dp);
		SET_OBJ_DATA_PTR(dp, OBJ_DATA_PTR(new_dp));
		SET_OBJ_DATA_PTR(new_dp, tmp_data);

		SET_OBJ_TYPE_DIM(new_dp,which_dim, OBJ_TYPE_DIM(dp,which_dim) );
		SET_OBJ_TYPE_DIM(dp,which_dim, DIMENSION(dsp,which_dim) );

		delvec(new_dp);
	}
}

Data_Obj *mlab_reshape(QSP_ARG_DECL  Data_Obj *dp, Shape_Info *shpp, const char *name)
{
	Data_Obj *dp_new;
	Identifier *idp;

	/* BUG we need to make sure that this object
	 * is not referenced in the right hand expression.
	 * If it is, we are going to lose the old values...
	 * The safest thing to do would be to create a new
	 * object for the assignment, then rename it after
	 * we are done with the old one...
	 */
	dp_new = make_dobj("ass_tmp",SHP_TYPE_DIMS(shpp),SHP_PREC_PTR(shpp));
	/* BUG? we may have a problem with multiple return objects... */
	/* what should get_lhs_name() return when there are multiple
	 * objects on the lhs???
	 */
	if( dp != NULL ){
		delvec(dp);
	}
	obj_rename(dp_new,name);

	/* We also need to fix the identifier pointer */

	idp = get_id(name);
	assert( idp != NULL );
	assert( ID_TYPE(idp) == ID_OBJ_REF );

	SET_REF_OBJ(ID_REF(idp), dp_new );
	/* and update the shape! */
	set_id_shape(idp, OBJ_SHAPE(dp_new) );
	return(dp_new);
} // mlab_reshape
				
static Data_Obj * mlab_lhs(QSP_ARG_DECL Data_Obj *dp, Vec_Expr_Node *enp)
{
	assert( VN_CODE(enp) == T_ASSIGN );

	if( dp != NULL ){
		/* If the shape doesn't match,
		 * then we have to remake the object
		 */
if( VN_SHAPE(enp) == NULL ){
warn("mlab_eval_work_tree:  T_ASSIGN has null shape ptr");
} else {
if( VN_SHAPE(VN_CHILD(enp,1)) == NULL ){
warn("mlab_lhs:  rhs has null shape");
return NULL;
}

		if( OBJ_COLS(dp) != SHP_COLS(VN_SHAPE(VN_CHILD(enp,1))) ||
			OBJ_ROWS(dp) != SHP_ROWS(VN_SHAPE(VN_CHILD(enp,1))) ){

			Data_Obj *dp_new;
			const char *s;

			/* In matlab, you're allowed to reassign
			 * the shape of an object...
			 */

			s = get_lhs_name(VN_CHILD(enp,0));
			dp_new = mlab_reshape(QSP_ARG  dp,VN_SHAPE(VN_CHILD(enp,1)),s);
			/* We do this later! */
			/* eval_obj_assignment(dp_new,VN_CHILD(enp,1)); */
			return(dp_new);
		}
} /* end debug */
	} else {
		/* make a new object */
		assert( VN_CHILD(enp,0) != NULL );

		dp = CREATE_MATRIX(VN_CHILD(enp,0),VN_SHAPE(enp));
	}
	return(dp);
} /* end mlab_lhs */


#ifdef MATLAB_FOOBAR

static void exec_mlab_cmd(int code)
{
	switch(code){
		case MC_WHO:	mc_who();	break;
		case MC_HELP:	mc_help();	break;
		case MC_WHOS:	mc_whos();	break;
		case MC_LOOKFOR:mc_lookfor();	break;
		case MC_SAVE:	mc_save();	break;
		default:
			sprintf(ERROR_STRING,"Unexpected mlab cmd code %d",code);
			warn(ERROR_STRING);
			break;
	}
}
#endif /* MATLAB_FOOBAR */

void _eval_immediate(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	/* Why do we need to do this here??? */
	enp=compile_prog(enp);

	if( IS_CURDLED(enp) ) return;

	if( dumpit ) {
		print_shape_key(SINGLE_QSP_ARG);
		dump_tree(enp);
	}

	// call delete_local_objs() here???
	delete_local_objs();	// eval_immediate

	/* We need to do some run-time resolution for this case:
	 * float f[]=[1,2,3];
	 *
	 */

	switch(VN_CODE(enp)){
		case T_IFTHEN:		/* eval_immediate */
		case T_WHILE:		/* eval_immediate */
		case T_DO_WHILE:	/* eval_immediate */
		case T_DO_UNTIL:	/* eval_immediate */
		case T_STAT_LIST:
		case T_ASSIGN:
		case T_SET_STR:		/* eval_immediate */
		case T_SET_PTR:		/* eval_immediate */
		case T_DIM_ASSIGN:
		case T_CALLFUNC:
		case T_DISPLAY:
		case T_SCRIPT:
		case T_INFO:
		case T_CALL_NATIVE:
		ALL_PRINT_CASES		/* advise, warn, print_exp */
		ALL_INCDEC_CASES
		SOME_MATLAB_CASES
			eval_work_tree(enp,NULL);
			break;
		case T_DECL_STAT:
		case T_EXTERN_DECL:
			eval_decl_tree(enp);
			break;
		default:
			missing_case(enp,"eval_immediate");
dump_tree(enp);
			break;
	}
	executing=0;

	// Should we release the tree here??
} /* end eval_immediate */


Data_Obj *
_make_local_dobj(QSP_ARG_DECL  Dimension_Set *dsp,Precision *prec_p, Platform_Device *pdp)
{
	Data_Obj *dp;
	Node *np;
	const char *s;

	s=localname();	// localname() uses savestr, so we have to free or else there will be a leak

#ifdef HAVE_ANY_GPU
	if( pdp != NULL ) push_pfdev(pdp);
#endif // HAVE_ANY_GPU

	dp=make_dobj(s,dsp,prec_p);

#ifdef HAVE_ANY_GPU
	if( pdp != NULL ) pop_pfdev();
#endif // HAVE_ANY_GPU

	rls_str(s);

	//if( dp == NULL ) return(dp);	// does this ever happen?
	assert(dp!=NULL);

	/* remember this for later deletion... */
	if( local_obj_lp == NULL )
		local_obj_lp = new_list();

	/* We can't just store dp, because it could become a dangling pointer if someone
	 * else deletes him, and apparently some functions are good citizens and clean up
	 * after themselves but others do not or cannot.  So we have to save a new string with
	 * the name, and hope we delete the same one later...
	 */

	s=savestr(OBJ_NAME(dp));
	np = mk_node((void *)s);

	addTail(local_obj_lp,np);

	return(dp);
}

static void _delete_local_objs(SINGLE_QSP_ARG_DECL)
{
	Node *np;
	Data_Obj *dp;
	const char *s;

	if( local_obj_lp == NULL ) return;

	//np=QLIST_HEAD(local_obj_lp);
	np = remHead(local_obj_lp);
	while(np!=NULL){
		s = (char *)NODE_DATA(np);
		assert( ! strncmp(s,"L.",2) );	// assume all names begin L.

		dp = dobj_of(s);
		if( dp != NULL ){
			delvec(dp);
		}
		  else {
		}
		rls_str(s);
		rls_node(np);
		np = remHead(local_obj_lp);
	}
}


/*
 * Evaluate the right hand side of an assignment, with the
 * result in dp.
 * We would like to do this in a way which avoids unnecessary copying
 * of data...
 * This means that whenever possible, we will place the results in dp
 * as they are computed.  To do this correctly, however, we must be sure
 * that the object referred to does not appear too many times on the rhs:
 * for example, v = -v * v;
 * would use v as the destination, so that the first scalar multiply
 * would put the result there, which would overwrite the value of v used
 * by the ensuing vector multiply!
 * Similar problems arise for expressions like v=v*v*v; ...
 *
 * To deal with this properly, we need to see if the lhs object appears
 * on the rhs - and if so, how many times and where (this really seems
 * like a scan-tree issue!).
 *
 *			    =			0
 *			  /   \
 *		        v      *                1
 *			     /   \
 *			   v      *		2
 *                               /  \
 *                              v   -1		3
 *
 * We need to precompute (in scan_tree) the lowest depth in the parse tree
 * at which the rhs object appears; if the evaluation depth is greater than
 * that, then we cannot use the destination object for temporary storage.
 *
 * But this may not be correct:
 *
 *			    =			0
 *			  /   \
 *		        v      *                1
 *			     /   \
 *			   *      *		2
 *                        / \    /  \
 *                       v   v  v    v		3
 *
 * What we can see here is that the depth at which v appears has little
 * to do with what is safe; the first product at level 2 needs a temp obj,
 * while the second one can go ahead and use v.  Perhaps we could make a count
 * of the number of lhs references in the tree, and count them as they
 * are evaluated, and inhibit appropriately...
 *
 * How about this:
 * at every node, we record the number of lhs references...
 * then, if we are traversing a tree, and we come to a node w/ >0 refs
 * for both children, then we can't pass the lhs obj to the first one.
 * If there is only one side, then we are ok, but we have to evaluate
 * that side first!
 * This is probably handled best by tree manipulation in scan_tree()
 */

static void _eval_obj_assignment(QSP_ARG_DECL Data_Obj *dst_dp,Vec_Expr_Node *enp)
{
#ifdef SCALARS_NOT_OBJECTS
	Identifier *idp;
#endif // SCALARS_NOT_OBJECTS
	double start,dx,dy;
	double dval;
	Data_Obj *dp1,*dp2,*dp3,*dp4;
	long ldx,ldy;
#ifdef NOT_YET
	Image_File *ifp;
#endif /* NOT_YET */
	Scalar_Value sval,*svp;
	Vec_Obj_Args oa1, *oap=&oa1;
	//int i;
	const char *s;
	//int vf_code=(-1);

/*
*/
	eval_enp = enp;

	if( dst_dp == NULL ){
advise("eval_obj_assignment returning (NULL target)");
dump_tree(enp);
		return;	/* probably an undefined reference */
	}
	assert(OBJ_NAME(dst_dp)!=NULL);


#ifdef QUIP_DEBUG
if( debug & eval_debug ){
sprintf(ERROR_STRING,"eval_obj_assignment %s",OBJ_NAME(dst_dp));
advise(ERROR_STRING);
dump_tree(enp);
}
#endif /* QUIP_DEBUG */

	switch(VN_CODE(enp)){
#ifdef SCALARS_NOT_OBJECTS
		case T_SCALAR_VAR:	// eval_obj_assignment
			idp = get_id(VN_STRING(enp));
			assert(idp!=NULL);
			if( ID_PREC_CODE(idp) == OBJ_PREC(dst_dp) ){
				assign_obj_from_scalar(enp,dst_dp,ID_SVAL_PTR(idp));
			} else {
				dval = cast_from_scalar_value(ID_SVAL_PTR(idp),ID_PREC_PTR(idp));
				(*(OBJ_PREC_PTR(dst_dp)->cast_from_double_func))(&sval,dval);

				assign_obj_from_scalar(enp,dst_dp,&sval);
			}
			break;
#endif // SCALARS_NOT_OBJECTS

		case T_BOOL_EQ:		/* eval_obj_assignment */
		case T_BOOL_NE:
		case T_BOOL_LT:
		case T_BOOL_GT:
		case T_BOOL_LE:
		case T_BOOL_GE:
		case T_BOOL_AND:
		case T_BOOL_OR:
		case T_BOOL_XOR:
		case T_BOOL_NOT:
		case T_BOOL_PTREQ:		/* eval_obj_assignment */
			eval_bitmap(dst_dp,enp);
			break;

		case T_RANGE2:
			{
			double d1,d2;
			double delta;
			d1=eval_int_exp(VN_CHILD(enp,0));
			d2=eval_int_exp(VN_CHILD(enp,1));
			delta = (d2-d1)/(OBJ_N_TYPE_ELTS(dst_dp)-1);
			easy_ramp2d(dst_dp,d1,delta,0.0);
			}
			break;

		case T_STRING_LIST:
		case T_STRING:
			assert( OBJ_PREC(dst_dp) == PREC_CHAR );

			s = eval_string(enp);
			assert( OBJ_N_TYPE_ELTS(dst_dp) > strlen(s) );

			strcpy((char *)OBJ_DATA_PTR(dst_dp),s);
			break;

		/* matlab */
#ifdef FOOBAR
		case T_ROWLIST:		/* eval_obj_assignment */
			/* rowlist trees grow down to the left, so we start with the bottom row
			 * and work up
			 */
			i=SHP_ROWS(VN_SHAPE(enp));
			/* But this child could be a matrix object? */
			ASSIGN_ROW(dst_dp,i,VN_CHILD(enp,1));
			/* child[0] is either a ROWLIST node, or a ROW */
			eval_obj_assignment(dst_dp,VN_CHILD(enp,0));
			break;
#endif

		case T_ROW:
			/* do we need to subscript dst_dp?? */
			if( OBJ_ROWS(dst_dp) > 1 ){
				dp2 = d_subscript(dst_dp,1);
			} else {
				dp2 = dst_dp;
			}

			ASSIGN_ROW(dp2,1,enp);
			break;
		/* end matlab */

		case T_FIX_SIZE:
			/* fix_size() is a do-nothing function that we use
			 * to get around some otherwise difficult unknown
			 * size resolutions.  a real hack!
			 */
			break;

		case T_DILATE:
			dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
			dilate(dst_dp,dp1);
			break;
		case T_ERODE:
			dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
			erode(dst_dp,dp1);
			break;
		/* COnditional assignment:  a<b ? v : w
		 * If the conditional is a scalar, then
		 * this is easy, we just to the assignment
		 * to one or the other...
		 * But if the conditional is a vector, then
		 * we need to evaluate it into a scratch vector...
		 *
		 * For these tree codes, the first two keys indicate the result types,
		 * while the third is the test type.  Test can be S (scalar), or B (bitmap).
		 * The bitmap case handles all tests involving one or more vectors.
		 */
		case T_SS_S_CONDASS:		/* eval_obj_assignment */
			{
				index_t index;
				//Scalar_Value sval;

				/* I don't get this AT ALL??? */
				index = (index_t) eval_int_exp(VN_CHILD(enp,0));

				index = index!=0 ? 1 : 2;
				assert( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,index))) );

				eval_scalar(&sval,VN_CHILD(enp,index),OBJ_PREC_PTR(dst_dp));
				assign_obj_from_scalar(enp,dst_dp,&sval);
			}
			break;
		case T_VS_S_CONDASS:		/* eval_obj_assignment */
			{
				index_t index;
				//Scalar_Value sval;

				/* is a boolean expression and int expression? */
				index = (index_t) eval_int_exp(VN_CHILD(enp,0));

				index = index!=0 ? 1 : 2;

				if( index == 1 ){	/* first choice should be the vector */
					assert( ! SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,index))) );

					eval_obj_assignment(dst_dp,VN_CHILD(enp,index));
				} else {		/* second choice should be the scalar */
					assert( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,index))) );

					eval_scalar(&sval,VN_CHILD(enp,index),OBJ_PREC_PTR(dst_dp));
					assign_obj_from_scalar(enp,dst_dp,&sval);
				}
			}
			break;

		case T_VV_S_CONDASS:			/* eval_obj_assignment */
			{
				index_t index;

				index = (index_t) eval_int_exp(VN_CHILD(enp,0));

				index = index!=0 ? 1 : 2;
				assert( ! SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,index))) );

				eval_obj_assignment(dst_dp,VN_CHILD(enp,index));
			}
			break;

		case T_SS_B_CONDASS: /* eval_obj_assignment */
			{
				Data_Obj *bm_dp;
				Scalar_Value sval2;

				/* Neet to create a temp vector or bitmap,
				 * and then use the select vector function.
				 */

				bm_dp = eval_bitmap(NULL,VN_CHILD(enp,0));
				/* we need to know the type of the destination before
				 * we evaluate the scalars...
				 */
				eval_scalar(&sval,VN_CHILD(enp,1),OBJ_PREC_PTR(dst_dp));
				eval_scalar(&sval2,VN_CHILD(enp,2),OBJ_PREC_PTR(dst_dp));

				setvarg1(oap,dst_dp);

				SET_OA_SVAL(oap,0, &sval);
				SET_OA_SVAL(oap,1, &sval2);

				SET_OA_SBM(oap,bm_dp);
				if( perf_vfunc(FVSSSLCT,oap) < 0 ){
					node_error(enp);
					warn("Error evaluating VSS select operator");
				}
			}
			break;

		case T_VS_B_CONDASS:		/* eval_obj_assignment */
			{
				Data_Obj *bm_dp;

				bm_dp = eval_bitmap(NULL,VN_CHILD(enp,0));
				dp2=eval_obj_exp(VN_CHILD(enp,1),NULL);
				eval_scalar(&sval,VN_CHILD(enp,2),OBJ_PREC_PTR(dst_dp));

				setvarg2(oap,dst_dp,dp2);
				SET_OA_SVAL(oap,0, &sval);
				SET_OA_SBM(oap,bm_dp);

				if( perf_vfunc(FVVSSLCT,oap) < 0 ){
					node_error(enp);
					warn("Error evaluating VVS select operator");
				}
			}
			break;
		case T_VV_B_CONDASS:		/* eval_obj_assignment */
			{
				Data_Obj *bm_dp;
				bm_dp = eval_bitmap(NULL,VN_CHILD(enp,0));
				dp2=eval_obj_exp(VN_CHILD(enp,1),NULL);
				dp3=eval_obj_exp(VN_CHILD(enp,2),NULL);

				setvarg3(oap,dst_dp,dp2,dp3);
				SET_OA_SBM(oap,bm_dp);
				if( perf_vfunc(FVVVSLCT,oap) < 0 ){
					node_error(enp);
					warn("Error evaluating VVV select operator");
				}
			}
			break;

		case T_VV_VV_CONDASS:		/* eval_obj_assignment */
			{
				dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
				dp2=eval_obj_exp(VN_CHILD(enp,1),NULL);
				dp3=eval_obj_exp(VN_CHILD(enp,2),NULL);
				dp4=eval_obj_exp(VN_CHILD(enp,3),NULL);
				setvarg5(oap,dst_dp,dp1,dp2,dp3,dp4);
				if( perf_vfunc(VN_BM_CODE(enp), oap) < 0 ){
					node_error(enp);
					warn("Error evaluating VV_VV conditional");
				}
			}
			break;
		case T_VV_VS_CONDASS:		/* eval_obj_assignment */
			{
				dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
				dp2=eval_obj_exp(VN_CHILD(enp,1),NULL);
				dp3=eval_obj_exp(VN_CHILD(enp,2),NULL);
				eval_scalar(&sval,VN_CHILD(enp,3),OBJ_MACH_PREC_PTR(dp3));
				setvarg4(oap,dst_dp,dp1,dp2,dp3);
				SET_OA_SVAL(oap,0, &sval);
				if( perf_vfunc(VN_BM_CODE(enp), oap) < 0 ){
					node_error(enp);
					warn("Error evaluating VV_VS conditional");
				}
			}
			break;
		case T_VS_VV_CONDASS:		/* eval_obj_assignment */
			{
				dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
				eval_scalar(&sval,VN_CHILD(enp,1),OBJ_MACH_PREC_PTR(dp1));
				dp2=eval_obj_exp(VN_CHILD(enp,2),NULL);
				dp3=eval_obj_exp(VN_CHILD(enp,3),NULL);
				setvarg4(oap,dst_dp,dp1,dp2,dp3);
				SET_OA_SVAL(oap,0, &sval);
				if( perf_vfunc(VN_BM_CODE(enp), oap) < 0 ){
					node_error(enp);
					warn("Error evaluating VS_VV conditional");
				}
			}
			break;
		case T_VS_VS_CONDASS:		/* eval_obj_assignment */
			{
				Scalar_Value sval2;

				dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
				eval_scalar(&sval,VN_CHILD(enp,1),OBJ_MACH_PREC_PTR(dp1));
				dp2=eval_obj_exp(VN_CHILD(enp,2),NULL);
				eval_scalar(&sval2,VN_CHILD(enp,3),OBJ_MACH_PREC_PTR(dp2));
				setvarg3(oap,dst_dp,dp1,dp2);
				/* The first scalar is the source */
				SET_OA_SVAL(oap,0, &sval);
				SET_OA_SVAL(oap,1, &sval2);
				if( perf_vfunc(VN_BM_CODE(enp), oap) < 0 ){
					node_error(enp);
					warn("Error evaluating VS_VS conditional");
				}
			}
			break;
#ifdef NEED_TO_DO_THESE
	T_SS_VV_CONDASS,			/* new condass functions */
	T_SS_VS_CONDASS,			/* new condass functions */
	T_VV_SS_CONDASS,			/* new condass functions */
	T_VS_SS_CONDASS,			/* new condass functions */
	T_SS_SS_CONDASS,			/* new condass functions */
#endif /* NEED_TO_DO_THESE */


		case T_MAX_TIMES:
			/* we return the scalar number of times.
			 * There are two args passed by reference,
			 * for the indices, and the max value
			 */

			/* FVMAXG */

			dp1=eval_obj_ref(VN_CHILD(enp,0));		/* indices */
			dp2=eval_obj_ref(VN_CHILD(enp,1));		/* maxval */
			dp3=eval_obj_exp(VN_CHILD(enp,2),NULL);	/* input */
			setvarg2(oap,dp1,dp3);
			SET_OA_SRC1(oap,dp2);				/* destination maxval */
			SET_OA_SRC2(oap,dst_dp);			/* destination n */
			SET_OA_SVAL(oap,0, (Scalar_Value *)OBJ_DATA_PTR(dp2));
			SET_OA_SVAL(oap,1, (Scalar_Value *)OBJ_DATA_PTR(dst_dp));
			if( perf_vfunc(FVMAXG,oap) < 0 ){
				node_error(enp);
				warn("Error evaluating max_times operator");
			}
			break;

		case T_RDFT:						/* eval_obj_assignment */
			dp1 = eval_obj_exp(VN_CHILD(enp,0),NULL);
			//h_vl2_fft2d(VFCODE_ARG  dst_dp,dp1);
			error1("eval_obj_assignment:  Sorry, don't know how to call fft2d!?"); 
			break;

		case T_RIDFT:						/* eval_obj_assignment */
			dp1 = eval_obj_exp(VN_CHILD(enp,0),NULL);
			//h_vl2_ift2d(VFCODE_ARG  dst_dp,dp1);
			error1("eval_obj_assignment:  Sorry, don't know how to call ift2d!?"); 
			break;

		case T_REDUCE:						/* eval_obj_assignment */
			dp1 = eval_obj_exp(VN_CHILD(enp,0),NULL);
			reduce(dst_dp,dp1);
			break;

		case T_ENLARGE:						/* eval_obj_assignment */
			dp1 = eval_obj_exp(VN_CHILD(enp,0),NULL);
			enlarge(dst_dp,dp1);
			break;

		case T_TYPECAST:		/* eval_obj_assignment */
			eval_typecast(enp,dst_dp);
			break;

		/* use tabled functions here???
		 * Or at least write a macro for the repeated code...
		 */

		case T_MINVAL:
			dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
			clear_obj_args(oap);
			setvarg2(oap,dst_dp,dp1);
			//vminv(oap);
			//vf_code=FVMINV;
			//h_vl2_vminv(HOST_CALL_ARGS);
			platform_dispatch_by_code(FVMINV, oap);
			break;
		case T_MAXVAL:
			dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
			clear_obj_args(oap);
			setvarg2(oap,dst_dp,dp1);
			//vmaxv(oap);
			//vf_code=FVMAXV;
			//h_vl2_vmaxv(HOST_CALL_ARGS);
			platform_dispatch_by_code(FVMAXV, oap);
			break;
		case T_SUM:				/* eval_obj_assignment */
			dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
			clear_obj_args(oap);
			setvarg2(oap,dst_dp,dp1);
			platform_dispatch_by_code(FVSUM, oap);
			break;

#ifdef NOT_YET
		case T_LOAD:						/* eval_obj_assignment */
			s = eval_string(VN_CHILD(enp,0));
			if( s == NULL ) break;

			/* load image from file */
			/* Can we assume that the sizes have already
			 * been checked???
			 */

			ifp = img_file_of(s);

			/* BUG?  a lot of these checks should
			 * probably be done in scan_tree() ?
			 */
			if( ifp == NULL ){
				ifp = read_image_file(s);
				if( ifp==NULL ){
					node_error(enp);
					sprintf(ERROR_STRING,
	"eval_obj_assignment LOAD/READ:  Couldn't open image file %s",s);
					warn(ERROR_STRING);
					break;
				}
			}
			if( ! IS_READABLE(ifp) ){
				sprintf(ERROR_STRING,
		"File %s is not readable!?",s);
				warn(ERROR_STRING);
				break;
			}

			if( OBJ_PREC(ifp->if_dp) == PREC_ANY || OBJ_PREC(dst_dp) == OBJ_PREC(ifp->if_dp) ){
				/* no need to typecast */
				read_object_from_file(dst_dp,ifp);
				/* BUG?? do we know the whole object is assigned? */
				/* does it matter? */
				//SET_OBJ_FLAG_BITS(dst_dp, DT_ASSIGNED);
				// done below
				//note_assignment(dst_dp);
			} else {
				dp1=make_local_dobj(
					OBJ_SHAPE(dst_dp).si_type_dimset,
					OBJ_PREC_PTR(ifp->if_dp));
				read_object_from_file(dp1,ifp);
				//h_vl2_convert(QSP_ARG  dst_dp,dp1);
				dp_convert(dst_dp,dp1);
				delvec(dp1);	// doesn't need delete_local_objects?
			}
			break;
#endif /* NOT_YET */

		case T_ASSIGN:		/* x=y=z; */
			dp1 = eval_obj_ref(VN_CHILD(enp,0));
			if( dp1 == NULL )
				break;
			eval_obj_assignment(dp1,VN_CHILD(enp,1));
			/* now copy to the target of this call */
			if( do_unfunc(dst_dp,dp1,FVMOV) ){
				node_error(enp);
				warn("Error evaluating assignment");
			}
			break;


#ifdef NOT_YET
		case T_CALL_NATIVE:			/* eval_obj_assignment() */
			eval_native_assignment(dst_dp,enp);
			break;
#endif /* NOT_YET */

		case T_INDIR_CALL:
		case T_CALLFUNC:			/* eval_obj_assignment() */
#ifdef QUIP_DEBUG
if( debug & eval_debug ){
sprintf(ERROR_STRING,"eval_obj_assignment calling exec_subrt, dst = %s",OBJ_NAME(dst_dp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			exec_subrt(enp,dst_dp);
			break;

		ALL_OBJREF_CASES			/* eval_obj_assignment */
			if( VN_CODE(enp) == T_LIST_OBJ || VN_CODE(enp) == T_COMP_OBJ ){
				/* should be its own case... */
				/* a list of expressions, maybe literals... */
				/* We need to do something to handle 2D arrays... */
				/* assign_obj_from_list(dst_dp,VN_CHILD(enp,0),0); */

				assign_obj_from_list(dst_dp,enp,0);
				//SET_OBJ_FLAG_BITS(dst_dp, DT_ASSIGNED);
				// done below
				//note_assignment(dst_dp);
				break;
			}

			/* dp1=eval_obj_ref(enp); */
			dp1=eval_obj_exp(enp,dst_dp);

			if( dp1 == NULL ){
				node_error(enp);
				warn("Unable to evaluate RHS");
				break;
			}

			if( executing && expect_objs_assigned && ! HAS_ALL_VALUES(dp1) ){
				unset_object_warning(enp,dp1);
			}
			if( mode_is_matlab ){
				if( OBJ_ROWS(dp1) == 1 && OBJ_ROWS(dst_dp) > 1 ){
					dp2 = d_subscript(dst_dp,1);
					//setvarg2(oap,dst_dp,dp1);
					//h_vl2_convert(HOST_CALL_ARGS);
					dp_convert(dst_dp,dp1);
					break;
				}
			}

			/* BUG?  is this correct if we have multiple components??? */
			if( IS_SCALAR(dp1) ){
				svp = (Scalar_Value *)OBJ_DATA_PTR(dp1);
				/* BUG type conversion? */
				assign_obj_from_scalar(enp,dst_dp,svp);
			} else {
				/* object-to-object copy */
				if( dst_dp != dp1 ){
					//setvarg2(oap,dst_dp,dp1);
					//h_vl2_convert(HOST_CALL_ARGS);
					dp_convert(dst_dp,dp1);
				}
			}
			break;

		case T_INNER:		/* eval_obj_assignment */
			/* We might put this with the two-op funcs,
			 * but because of the shape differences,
			 * it is inlikely that...
			 *
			 * well, actually, what about:
			 *
			 * mat[i] = inner(mat,c1) + inner(mat,c2);
			 */
			if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) && 
				SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) ){
				double d2;
				dval=eval_flt_exp(VN_CHILD(enp,0));
				d2=eval_flt_exp(VN_CHILD(enp,1));
				dbl_to_scalar(&sval,dval*d2,OBJ_PREC_PTR(dst_dp));
				assign_obj_from_scalar(enp,dst_dp,&sval);
			} else {
				/* we don't pass the dst object, because it may not
				 * be the right shape - we could check this, but we're lazy!
				 */
				GET_2_OPERANDS(enp,&dp1,&dp2,NULL);	// T_INNER
				/* This assumes that the destination is the right size;
				 * it will be wrong if the dot product is a scalar...
				 */
				inner(dst_dp,dp1,dp2);
				//WARN("Sorry, inner is temporarily unavailable!?");
			}
			break;

		case T_DFT:			/* eval_obj_assignment */
			/* BUG if the types are difference, dst_dp may not be
			 * an appropriate arg for eval_obj_exp()
			 */
			dp1=eval_obj_exp(VN_CHILD(enp,0),dst_dp);
			/* BUG need to handle real fft's;
			 * for now, assume cpx to cpx
			 */

			/*
			if( do_unfunc(dst_dp,dp1,FVMOV) < 0 ){
				node_error(enp);
				warn("error moving data for fft");
				break;
			}
			h_vl2_fft2d(VFCODE_ARG  dst_dp,dst_dp);
			*/

			clear_obj_args(oap);
			setvarg2(oap,dst_dp,dp1);
			platform_dispatch_by_code(FVFFT2D, oap);

			break;

		case T_IDFT:
			dp1=eval_obj_exp(VN_CHILD(enp,0),dst_dp);
			/* BUG need to handle real fft's;
			 * for now, assume cpx to cpx
			 */
			/*
			if( do_unfunc(dst_dp,dp1,FVMOV) < 0 ){
				node_error(enp);
				warn("error moving data for ifft");
				break;
			}
			h_vl2_ift2d(VFCODE_ARG  dst_dp,dst_dp);
			*/
			clear_obj_args(oap);
			setvarg2(oap,dst_dp,dp1);
			platform_dispatch_by_code(FVIFT2D, oap);

			break;

		case T_WRAP:		/* eval_obj_assignment */
			/* We can't wrap in-place, so don't pass dst_dp
			 * to eval_obj_exp
			 */
			/* BUG?  will this catch a=wrap(a) ?? */
			dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
			assert( dp1 != NULL );

			wrap(dst_dp,dp1);
			break;

		case T_SCROLL:
			dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
			ldx=eval_int_exp(VN_CHILD(enp,1));
			ldy=eval_int_exp(VN_CHILD(enp,2));
			dp_scroll(dst_dp,dp1,(incr_t)ldx,(incr_t)ldy);
			break;

		/* 2 argument operations */

		case T_MATH2_VFN:		/* eval_obj_assignment */
		case T_VV_FUNC:
			GET_2_OPERANDS(enp,&dp1,&dp2,dst_dp);	// T_VV_FUNC
			if( dp1 == NULL || dp2 == NULL ){
				node_error(enp);
				advise("bad vector operand");
			} else
				if( do_vvfunc(dst_dp,dp1,dp2,VN_VFUNC_CODE(enp)) < 0 ){
					node_error(enp);
					warn("Expression error");
dump_tree(enp);	// expression error
				}
			break;

		case T_MATH2_VSFN:
		case T_VS_FUNC:
			dp1=eval_obj_exp(VN_CHILD(enp,0),dst_dp);
			if( dp1 == NULL ){
				node_error(enp);
				advise("vector operand does not exist");
				break;
			}
			eval_scalar(&sval,VN_CHILD(enp,1),OBJ_MACH_PREC_PTR(dp1));
			if( do_vsfunc(dst_dp,dp1,&sval,VN_VFUNC_CODE(enp)) < 0 ){
				node_error(enp);
				warn("Error assigning object");
			}
			break;

		case T_TRANSPOSE:	/* eval_obj_assignment */
			/* Why did we ever think this was correct? */
			/* dp1 = get_id_obj(VN_STRING(enp),enp); */
			dp1=eval_obj_exp(VN_CHILD(enp,0),NULL);
			if( dp1 == NULL ) break;
			/* BUG make sure valid */
			xpose_data(dst_dp,dp1);
			break;

		case T_RAMP:
			start=eval_flt_exp(VN_CHILD(enp,0));
			dx=eval_flt_exp(VN_CHILD(enp,1));
			dy=eval_flt_exp(VN_CHILD(enp,2));
			easy_ramp2d(dst_dp,start,dx,dy);
			break;

		case T_STR2_FN:	/* eval_obj_assignment */
		case T_STR1_FN:	/* eval_obj_assignment */
		case T_SIZE_FN: 	/* eval_obj_assignment */
			dval = eval_flt_exp(enp);
			dbl_to_scalar(&sval,dval,OBJ_PREC_PTR(dst_dp));
			assign_obj_from_scalar(enp,dst_dp,&sval);
			break;

		case T_LIT_INT:				/* eval_obj_assignment */
			/* BUG? we are doing a lot of unecessary conversions
			 * if the object is integer to begin with... but this
			 * will work.
			 */
			int_to_scalar(&sval,VN_INTVAL(enp),OBJ_PREC_PTR(dst_dp));
			assign_obj_from_scalar(enp,dst_dp,&sval);
			break;

		case T_LIT_DBL:
			dbl_to_scalar(&sval,VN_DBLVAL(enp),OBJ_PREC_PTR(dst_dp));
			assign_obj_from_scalar(enp,dst_dp,&sval);
			break;

		case T_BITRSHIFT:
		case T_BITLSHIFT:
		case T_BITAND:
		case T_BITOR:
		case T_BITXOR:
		case T_BITCOMP:
		case T_MODULO:
			int_to_scalar( &sval, eval_int_exp(enp), OBJ_PREC_PTR(dst_dp) );
			assign_obj_from_scalar(enp,dst_dp,&sval);
			break;

		case T_MATH0_FN:
		case T_MATH1_FN:
		case T_MATH2_FN:		/* eval_obj_assignment */
		case T_INT1_FN:
		case T_TIMES:
		case T_PLUS:
		case T_MINUS:
		case T_DIVIDE:
		case T_SCALMAX:
		case T_SCALMIN:
			dbl_to_scalar(&sval, eval_flt_exp(enp), OBJ_PREC_PTR(dst_dp) );
			assign_obj_from_scalar(enp,dst_dp,&sval);
			break;

		case T_MATH0_VFN:			/* eval_obj_assignment */
			/* unary math function */
			if( do_un0func(dst_dp,VN_VFUNC_CODE(enp)) ){
				node_error(enp);
				warn("Error evaluating math function");
			}
			break;

		case T_INT1_VFN:			/* eval_obj_assignment */
		case T_MATH1_VFN:			/* eval_obj_assignment */
		case T_CHAR_VFN:			/* eval_obj_assignment */
			/* unary math function */
			dp1=eval_obj_exp(VN_CHILD(enp,0),dst_dp);
			assert( dp1 != NULL );

			if( do_unfunc(dst_dp,dp1,VN_VFUNC_CODE(enp)) ){
				node_error(enp);
				warn("Error evaluating (math/int) function");
			}
			break;

		default:
			missing_case(enp,"eval_obj_assignment");
			break;
	}

	note_assignment(dst_dp);
}		/* end eval_obj_assignment() */

/****************** eval_work_tree helper funcs ********************/

#define execute_script_node(enp) _execute_script_node(QSP_ARG  enp)

static int _execute_script_node(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Macro *dummy_mp;
	Macro_Arg **ma_tbl;
	String_Buf *sbp;
	Subrt *srp;
	Query *qp;
	int n_args, start_level;

	srp = VN_SUBRT(enp);
	assert( IS_SCRIPT(srp) );
	assert(SR_N_ARGS(srp)<=MAX_SCRIPT_ARGS);
	assert(n_stored_script_args==0);	// BUG will fail if recursion
						// also not thread-safe!?  BUG

	n_args=parse_script_args(VN_CHILD(enp,0),0,SR_N_ARGS(srp));
	if( n_args < 0 ) return -1;

	// IN the objC implementation, the args are held in the query object as a list,
	// not an array.  This makes the recursive population a little tricker.
	// If we traverse the tree correctly, we may be able to simply add to the list
	if( n_args != SR_N_ARGS(srp) ){
		sprintf(ERROR_STRING,
	"Script subrt %s should have %d args, passed %d",
			SR_NAME(srp),SR_N_ARGS(srp),n_args);
		warn(ERROR_STRING);
		rls_script_args();
		return -1;
	}

	/* Set up dummy_mac so that the interpreter will
	 * think we are in a macro...
	 */
	ma_tbl = create_generic_macro_args(SR_N_ARGS(srp));
	sbp = create_stringbuf(SR_TEXT(srp));
	dummy_mp = create_macro(SR_NAME(srp), SR_N_ARGS(srp), ma_tbl, sbp,
		current_line_number(SINGLE_QSP_ARG) );

	/* Any arguments to a script function
	 * will be treated like macro args...
	 */

	sprintf(msg_str,"Script func %s",SR_NAME(srp));
	push_text((char *)SR_TEXT(srp), msg_str);

	qp=CURR_QRY(THIS_QSP);

	set_query_macro(qp, dummy_mp);
	// when are these freed???  BUG?  memory leak?
	set_query_args(qp, (const char **)getbuf( SR_N_ARGS(srp) * sizeof(char *) ) );

	pass_script_args(qp);

	/* If we pass object names to script functions by
	 * dereferencing pointers, we may end up with invisible objects
	 * whose contexts have been popped; here we restore those
	 * contexts.
	 */

	set_script_context(SINGLE_QSP_ARG);

	push_top_menu(SINGLE_QSP_ARG);	/* make sure at root menu */
	start_level = QLEVEL;
	enable_stripping_quotes(SINGLE_QSP_ARG);
	while( QLEVEL >= start_level ){
		// was do_cmd
		// We have a problem, if the script contains
		// a pause macro and a ^D is typed, do_cmd
		// tries to read more input...
		qs_do_cmd(THIS_QSP);
		lookahead(SINGLE_QSP_ARG);
	}
	do_pop_menu(SINGLE_QSP_ARG);		/* go back */

	unset_script_context(SINGLE_QSP_ARG);

	rls_macro(dummy_mp);
	// We don't need to call rls_script_args, because
	// when the macro is exited the args are released.
	// But we do need to zero n_stored_script_args!
	clear_script_args();

	return 0;
}

#define eval_assignment(enp) _eval_assignment(QSP_ARG  enp)

static void _eval_assignment(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Data_Obj *dp;

	/* we check runtime resolution here ...
	 * In preliminary shape analysis, we leave the assign
	 * node UK if either node is; but calltime resolution
	 * proceeds incrementally, we might get the assign node
	 * or even lower?)
	 */

	if( mode_is_matlab ){
		if( UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) )
			update_tree_shape(VN_CHILD(enp,0));
		if( UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) )
			update_tree_shape(VN_CHILD(enp,1));
		if( UNKNOWN_SHAPE(VN_SHAPE(enp)) &&
				! UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) ){
			resolve_tree(enp,NULL);
		}
	}

	if( UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_assignment:  last ditch attempt at runtime resolution of LHS %s",node_desc(VN_CHILD(enp,0)));
advise(ERROR_STRING);
dump_tree(enp);
}
#endif /* QUIP_DEBUG */
		/*
		resolve_one_uk_node(VN_CHILD(enp,0));
		*/
		resolve_tree(VN_CHILD(enp,0),NULL);

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_assignment:  after last ditch attempt at runtime resolution of LHS %s:",node_desc(VN_CHILD(enp,0)));
advise(ERROR_STRING);
dump_tree(enp);
}
#endif /* QUIP_DEBUG */
	}
	if( UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_assignment:  last ditch attempt at runtime resolution of RHS %s",node_desc(VN_CHILD(enp,1)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		resolve_tree(VN_CHILD(enp,1),NULL);
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_assignment:  after last ditch attempt at runtime resolution of RHS %s:",node_desc(VN_CHILD(enp,1)));
advise(ERROR_STRING);
dump_tree(enp);
}
#endif /* QUIP_DEBUG */
	}

	// if the LHS is a scalar var, we need to do something different...
#ifdef SCALARS_NOT_OBJECTS
	if( VN_CODE(VN_CHILD(enp,0)) == T_SCALAR_VAR ){
		Identifier *idp;
		idp = get_id(VN_STRING(VN_CHILD(enp,0)));
		assert(idp!=NULL);
		assign_scalar_id(idp, VN_CHILD(enp,1));
		return;
	}
#endif // SCALARS_NOT_OBJECTS

	dp = eval_obj_ref(VN_CHILD(enp,0));
	if( dp == NULL ){
		node_error(enp);
		warn("eval_assignment:  Invalid LHS");
		return;
	}

	if( mode_is_matlab ){
		dp=MLAB_LHS(dp,enp);
		assert( dp != NULL );

		eval_obj_assignment(dp,VN_CHILD(enp,1));
		return;
	}

#ifdef QUIP_DEBUG
if( debug & eval_debug ){
sprintf(ERROR_STRING,"eval_assignment:  calling eval_obj_assignment for target %s",OBJ_NAME(dp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( VN_CODE(enp) == T_DIM_ASSIGN )
		eval_dim_assignment(dp,VN_CHILD(enp,1));
	else {
		eval_obj_assignment(dp,VN_CHILD(enp,1));
	}
}

/* We return a 1 if we should keep working.
 * We return 0 if we encounter a return statement within a subroutine.
 *
 * what is "going" ???
 */

static int _eval_work_tree(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj *dst_dp)
{
	Data_Obj *dp;
	Subrt *srp;
	int intval;
#ifdef NOT_YET
	Image_File *ifp;
#endif /* NOT_YET */
	//Macro dummy_mac;
	const char *s;
	Identifier *idp,*idp2;
	Function_Ptr *fpp;
	int ret_val=1;			/* the default is to keep working */

	if( enp==NULL || IS_CURDLED(enp) ) return(ret_val);

#ifdef QUIP_DEBUG
if( debug & eval_debug ){
sprintf(ERROR_STRING,"eval_work_tree (dst = %s) %s",
dst_dp==NULL?"null":OBJ_NAME(dst_dp),
node_desc(enp));
advise(ERROR_STRING);
dump_tree(enp);
}
#endif /* QUIP_DEBUG */

#ifdef CAUTIOUS
	if( dst_dp != NULL ) {
		// this checks for a dangling pointer to a local object
		// that might have been deleted...
		assert(OBJ_NAME(dst_dp)!=NULL);
	}
#endif // CAUTIOUS

	eval_enp = enp;
	executing = 1;
	if( interrupted ) return 0;

	/* We need to do runtime resolution, but we don't want to descend entire
	 * statment trees here...  The top node may have an unknown leaf up until the
	 * time that the last statement is executed, but this may not be resolvable until
	 * the previous statement is executed...  therefore we only try to resolve selected
	 * nodes...
	 */

	/* We may have some unresolved shapes which depend on the current values of pointers */
	if( VN_SHAPE(enp) != NULL && UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_work_tree:  attemping to runtime resolution of %s",node_desc(enp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		resolve_tree(enp,NULL);
	}

	/* Where should we put this? */
	/* We want to do this at the END of each statement... */
	unlock_all_tmp_objs();

	/* BUG we'll do something more efficient eventually */

	/* We also need to remove the "local" objects... */

	// This appears to be a BUG because we can have a local object
	// at the root of a deep tree, and then call this multiple times...
	// Maybe local objects should have a node associated with them???

	//delete_local_objs();	// eval_work_tree

	switch(VN_CODE(enp)){

		case T_CALL_NATIVE:
			eval_native_work(enp);
			break;

		/* matlab ? */
		case T_MLFUNC:
			/* just a defn, do nothing */
			return(1);

		case T_DRAWNOW:			/* eval_work_tree */
			if( verbose )
				advise("Sorry, matlab drawnow not implemented yet");
			return(1);
			break;

		case T_CLF:
		case T_CLEAR:			/* eval_work_tree */
			warn("Sorry, matlab clear/clr not implemented yet");
			return(1);
			break;
#ifdef MATLAB_FOOBAR
		case T_MCMD:
			exec_mlab_cmd(VN_INTVAL(enp));
			break;
		/* this needs to be here for matlab (?)
		 * but we want this to be independent of the matlab module
		 * These should be matlab native functions...
		 */
		case T_MFILE:
			read_matlab_file(VN_STRING(enp));
			break;
		case T_PRINT:
			mlab_print_tree(VN_CHILD(enp,0));
			break;
#endif /* MATLAB_FOOBAR */
		case T_GLOBAL:
			/* what should we do? */
			break;

		/* end matlab */

		case T_GO_FWD:			/* eval_work_tree */
		case T_GO_BACK:
			if( !going ) {
				going=1;
				goto_label=VN_STRING(enp);
			}
			return(1);

		case T_LABEL:
			if( going && !strcmp(VN_STRING(enp),goto_label) ){
				going=0;
			}
			return(1);

		case T_BREAK:
			if( !going ) breaking=1;
			return(1);

		case T_CONTINUE:
			if( going ) return(1);
			/* We want to pop up to the enclosing for or while loop;
			 * BUT we have a small difficulty:  we might think that we can
			 * just search upwards in the tree for the node where we should
			 * go , but because of the recursive implementation of stat_list
			 * nodes, we have the whole stack to worry about...
			 * We really need to set a flag and return until we hit the loop node.
			 */
			continuing=1;
			return(1);

		case T_EXIT:
			if( going ) return(1);
			/* BUG what we'd really like to do here is break out to the interpreter,
			 * not exit the whole system...
			 */
			if( VN_CHILD(enp,0)!=NULL )
				exit( (int) eval_int_exp(VN_CHILD(enp,0)) );
			else
				exit(0);
			break;

		case T_FIX_SIZE:
			break;

		case T_DISPLAY:		/* eval_work_tree */
			if( going ) return(1);
			eval_display_stat(VN_CHILD(enp,0));
			break;

		case T_SET_FUNCPTR:	/* eval_work_tree */
			if( going ) return(1);
			srp = eval_funcref(VN_CHILD(enp,1));
			fpp = eval_funcptr(VN_CHILD(enp,0));
			/* BUG check for valid return values */
			fpp->fp_srp = srp;
			// The function may not have a shape until called!?
			//point_node_shape(enp,SC_SHAPE(scp));
			break;

		case T_SET_STR:		/* eval_work_tree */
			if( going ) return(1);
			s = eval_string(VN_CHILD(enp,1));
			idp = eval_ptr_expr(VN_CHILD(enp,0),UNSET_PTR_OK);
			if( idp == NULL ) break;
			assign_string(idp,s,enp);
			break;

		case T_SET_PTR:		/* eval_work_tree */
			if( going ) return(1);

			assert( dst_dp == NULL );

fprintf(stderr,"eval_work_tree T_SET_PTR:  BEGIN\n");
			idp2 = eval_ptr_expr(VN_CHILD(enp,1),EXPECT_PTR_SET);
			idp = eval_ptr_expr(VN_CHILD(enp,0),UNSET_PTR_OK);

			if( idp2 == NULL || idp == NULL ){
				node_error(enp);
				advise("eval_work_tree T_SET_PTR:  null object");
				break;
			}
fprintf(stderr,"eval_work_tree T_SET_PTR:  dst ptr = %s\n",ID_NAME(idp));
fprintf(stderr,"eval_work_tree T_SET_PTR:  src = %s\n",ID_NAME(idp2));
dump_tree(enp);

			assert( IS_POINTER(idp) );

			if( IS_POINTER(idp2) ){
fprintf(stderr,"eval_work_tree T_SET_PTR:  src %s is a pointer\n",ID_NAME(idp2));
				SET_PTR_REF(ID_PTR(idp), PTR_REF(ID_PTR(idp2)));
				SET_PTR_FLAG_BITS(ID_PTR(idp), POINTER_SET);
			} else if( IS_OBJ_REF(idp2) ){
fprintf(stderr,"eval_work_tree T_SET_PTR:  src %s is an object reference, will assign ptr\n",ID_NAME(idp2));
				assign_pointer(ID_PTR(idp),ID_REF(idp2));
fprintf(stderr,"eval_work_tree T_SET_PTR:  back from assign_pointer\n");
				/* can we do some runtime shape resolution here?? */
				/* We mark the node as unknown to force propagate_shape to do something
				 * even when the ptr was previously set to something else.
				 */
fprintf(stderr,"will make assertion, ID_PTR(idp) = 0x%lx\n",(long)ID_PTR(idp));
				/* We already asserted that idp identifies a pointer */
fprintf(stderr,"Pointer %s:\n",ID_NAME(idp));
fprintf(stderr,"\tflags: 0x%x\n",PTR_FLAGS(ID_PTR(idp)));
fprintf(stderr,"\tdecl_enp: 0x%lx\n",(long)PTR_DECL_VN(ID_PTR(idp)));
fprintf(stderr,"\treference: 0x%lx\n",(long)PTR_REF(ID_PTR(idp)));
				assert( PTR_DECL_VN(ID_PTR(idp)) != NULL );
				assert(VN_CHILD(enp,0) != NULL);
fprintf(stderr,"will make assertion about child 0 prec\n");
dump_tree(VN_CHILD(enp,0));
				assert(VN_SHAPE(VN_CHILD(enp,0)) != NULL );
describe_shape(VN_SHAPE(VN_CHILD(enp,0)));
				assert( VN_PREC_PTR(VN_CHILD(enp,0)) != NULL );
fprintf(stderr,"eval_work_tree T_SET_PTR:  will copy unknown node shape to %s\n",node_desc(PTR_DECL_VN(ID_PTR(idp))));
				copy_node_shape( PTR_DECL_VN(ID_PTR(idp)),uk_shape(VN_PREC(VN_CHILD(enp,0))));
fprintf(stderr,"eval_work_tree T_SET_PTR:  node shape copied with uk_shape...\n");
				assert( VN_SHAPE(VN_CHILD(enp,1)) != NULL );
				if( !UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) )
					resolve_pointer(VN_CHILD(enp,0),VN_SHAPE(VN_CHILD(enp,1)));
			}
			  else {
				assert( AERROR("eval_work_tree:  rhs is neither ptr nor reference") );
			}

			break;

#ifdef NOT_YET
		case T_OUTPUT_FILE:		/* eval_work_tree */
			if( going ) return(1);
			s=eval_string(VN_CHILD(enp,0));
			if( s!=NULL )
				set_output_file(s);
			break;
#endif /* NOT_YET */

		case T_STRCPY:		/* eval_work_tree */
			if( going ) return(1);
			idp=eval_ptr_expr(VN_CHILD(enp,0),UNSET_PTR_OK);
			s=eval_string(VN_CHILD(enp,1));
			if( idp != NULL && s != NULL )
				assign_string(idp,s,enp);
			break;

		case T_STRCAT:		/* eval_work_tree */
			if( going ) return(1);
			idp=eval_ptr_expr(VN_CHILD(enp,0),EXPECT_PTR_SET);
			s=eval_string(VN_CHILD(enp,1));
			if( idp != NULL && s != NULL )
				cat_string(REF_SBUF(ID_REF(idp)),s);
			break;

		case T_FOR:		/* eval_work_tree */
			do {
				/* evaluate the conditional */
				if( ! going )
					intval = (int) eval_int_exp(VN_CHILD(enp,0));
				else
					intval = 1;

				if( going || intval ){
					/* execute the body */
					ret_val=eval_tree(VN_CHILD(enp,1),NULL);
					if( ret_val == 0 ) return 0;
					continuing=0;
					if( going ) return(ret_val);
					ret_val=eval_tree(VN_CHILD(enp,2),NULL);
					if( ret_val == 0 ) return 0;
					if( going ) return(ret_val);
				}
				if( interrupted ) break;
			} while( intval );
			break;

		case T_SWITCH:			/* eval_work_tree */
			{
			Vec_Expr_Node *case_enp;
			long lval;

			if( ! going ){
				lval = eval_int_exp(VN_CHILD(enp,0));
				case_enp = find_case(VN_CHILD(enp,1),lval);
				if( case_enp == NULL ){
					/* It is not an error for there to be no case.
					 * We might want to have this warning controlled by a flag.
					 */
					node_error(enp);
					sprintf(ERROR_STRING,"No case for value %ld",lval);
					warn(ERROR_STRING);
					break;
				}

				assert( VN_CODE(case_enp) == T_CASE_STAT );
			} else {
				/* while we are looking for a goto label,
				 * we must examine all the cases...
				 */
				case_enp=first_case(enp);
			}

try_again:
			while( case_enp!=NULL && ! breaking ){
				ret_val=eval_tree(VN_CHILD(case_enp,1),NULL);
				/* BUG This test may get performed multiple times (harmlessly) */
				if( going ){
					/* first see if the target is in one of the cases at all */
					if( goto_child(VN_CHILD(enp,1)) == NULL ) {
						breaking=0;
						return(ret_val);
					}
				}
				if( going || ( ret_val && ! breaking ) ){
					/* this searches forward, how do we search backwards? */
					case_enp = next_case(case_enp);
				} else
					case_enp = NULL;
			}
			if( going ){
				/* If we get here, the label is in the case statements, but before the goto */
				case_enp = first_case(enp);
				goto try_again;
			}
			breaking=0;
			break;
			}

		case T_DO_UNTIL:		/* eval_work_tree */
			intval=0;	// quiet compiler
			do {
				ret_val=eval_tree(VN_CHILD(enp,0),NULL);
				if( ret_val == 0 ) return 0;
				continuing = 0;
				if( ! going ) intval = (int) eval_int_exp(VN_CHILD(enp,1));
			} while( (!going) && !intval);
			break;

		case T_DO_WHILE:		/* eval_work_tree */
			intval=0;	// quiet compiler
			do {
				ret_val=eval_tree(VN_CHILD(enp,0),NULL);
				if( ret_val == 0 ) return 0;
				continuing = 0;
				if( ! going ) intval = (int) eval_int_exp(VN_CHILD(enp,1));
			} while( (!going) && intval);
			break;

		case T_WHILE:			/* eval_work_tree */
			do {
				/* evaluate the conditional */
				if( !going )
					intval = (int) eval_int_exp(VN_CHILD(enp,0));
				else	intval = 1;
				if( intval ){
					/* execute the body */
					ret_val=eval_tree(VN_CHILD(enp,1),NULL);
					if( ret_val == 0 )
						return 0;
					continuing=0;
				}
				if( interrupted ) break;
				if( going ) return(1);
			} while( intval );
			break;

		case T_UNTIL:			/* eval_work_tree */
			do {
				/* evaluate the conditional */
				if( !going )
					intval = (int) eval_int_exp(VN_CHILD(enp,0));
				else	intval = 0;
				if( ! intval ){
					/* execute the body */
					ret_val=eval_tree(VN_CHILD(enp,1),NULL);
					if( ret_val == 0 )
						return 0;
					continuing=0;
				}
				if( interrupted ) break;
				if( going ) return(1);
			} while( ! intval );
			break;

		case T_PERFORM:		/* eval_work_tree */
			if( going ) return(1);
			intval = (int) eval_int_exp(VN_CHILD(enp,0));
			node_error(enp);
			if( intval )
				advise("enabling vector evaluation");
			else
				advise("disabling vector evaluation");

			set_perf(intval);
			break;

		case T_SCRIPT:		/* eval_work_tree */
			if( going ) return(1);
			if( execute_script_node(enp) < 0 )
				return 0;
			break;

#ifdef NOT_YET
		case T_SAVE:		/* eval_work_tree */
			if( going ) return(1);
			ifp=img_file_of(VN_STRING(enp));
			if( ifp == NULL ){
advise("evaltree:  save:");
describe_shape(VN_SHAPE(VN_CHILD(enp,0)));
				ifp = write_image_file(VN_STRING(enp),
					VN_SHAPE(VN_CHILD(enp,0))->si_frames);
				if( ifp==NULL ){
					/* permission error? */
					sprintf(ERROR_STRING,
		"Couldn't open image file %s",VN_STRING(enp));
					warn(ERROR_STRING);
					return 0;
				}
			}
		/* BUG we'd like to allow an arbitrary expression here!? */
			dp = eval_obj_ref(VN_CHILD(enp,0));
			if( dp == NULL ) return(1);

			write_image_to_file(ifp,dp);
			break;

		case T_FILETYPE:		/* eval_work_tree */
			if( going ) return(1);
			/* BUG? scan tree should maybe fetch this? */
			intval = get_filetype_index(VN_STRING(enp));
			if( intval < 0 ) return 0;
			set_filetype((filetype_code)intval);
			break;
#endif /* NOT_YET */

		case T_DECL_STAT_LIST:			/* eval_work_tree */
		case T_DECL_STAT:		/* eval_work_tree */
		case T_EXTERN_DECL:		/* eval_work_tree */
			if( going ) return(1);
#ifdef QUIP_DEBUG
if( debug & eval_debug ){
sprintf(ERROR_STRING,"eval_work_tree:  nothing to do for declarations");
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			return(ret_val);

		case T_STAT_LIST:				/* eval_work_tree */
			if( (ret_val=eval_work_tree(VN_CHILD(enp,0),dst_dp)) ){
				if( continuing || breaking ) return(ret_val);
				ret_val=eval_work_tree(VN_CHILD(enp,1),dst_dp);
			}
			if( ret_val && going ){
				/* The logic of how to do this depends to some extent
				 * on how the stat_list trees are structured...
				 *
				 * Currently, they are left-heavy.
				 * If we are going back, therefore, the label
				 * should be a descendent of the left-child of this node.
				 * (This will not be true if and when we allow
				 * restructuring/balancing of the trees.)
				 * In this case, we need to make the calls to
				 * get the stack right for the context of the label.
				 *
				 * If we are going ahead, we just need to pop.
				 */
				Vec_Expr_Node *goto_enp;
				goto_enp=goto_child(VN_CHILD(enp,0));
				if( goto_enp != NULL ){
					/* We don't need to pop any stack */
					eval_work_tree(enp,NULL);
				} else {
					/* The label must be ahead of us, which means that
					 * it is the right child of an ancestor node.
					 * we need to pop out until we find it
					 */
					return(1);
				}
			}
			/* BUG how do we know how far to pop??? */
			break;

		case T_INDIR_CALL:
		case T_CALLFUNC:				/* eval_work_tree */
			if( going ) return(1);
			subrt_ret_type=0;
			exec_subrt(enp,NULL);
			break;

		case T_IFTHEN:		/* eval_work_tree */
			if( ! going ){
				intval = (int) eval_int_exp(VN_CHILD(enp,0));
				if( intval )
					return( eval_tree(VN_CHILD(enp,1),dst_dp) );
				else if( VN_CHILD(enp,2) != NULL )
					return( eval_tree(VN_CHILD(enp,2),dst_dp) );
			} else {	// going
				ret_val = eval_tree(VN_CHILD(enp,1),dst_dp);
				// can eval_tree change going???
				// BUG?  changed these returns from 1 to ret_val
				// to eliminate an analyzer warning, but
				// I'm not sure if that is correct???
				if( ! going ) return(ret_val); // return(1);
				if( VN_CHILD(enp,1) != NULL )
					ret_val = eval_tree(VN_CHILD(enp,1),dst_dp);
				return ret_val; // return(1);
			}
			break;

		case T_SUBRT_DECL:		/* eval_work_tree */
			if( going ) return(1);
			error1("eval_work_tree encountered unexpected T_SUBRT_DECL???");
			break;

		case T_RETURN:		/* eval_work_tree */
			if( going ) return(1);
			if( VN_CHILD(enp,0) != NULL ){
				eval_obj_assignment(dst_dp,VN_CHILD(enp,0));
			}
			/* If we are returning from a subroutine before the end,
			 * we have to pop it now...
			 */
			return 0;

		case T_EXP_PRINT:		/* eval_work_tree */
			if( going ) return(1);
			eval_print_stat(VN_CHILD(enp,0));
			prt_msg("");	/* print newline after other expressions */
			break;

		case T_INFO:		/* eval_work_tree */
			if( going ) return(1);
			eval_info_stat(VN_CHILD(enp,0));
			break;

		case T_WARN:		/* eval_work_tree */
			if( going ) return(1);
			s=eval_string(VN_CHILD(enp,0));
			if( s != NULL ) warn(s);
			break;

		case T_ADVISE:		/* eval_work_tree */
			if( going ) return(1);
			s=eval_string(VN_CHILD(enp,0));
			if( s != NULL ) advise(s);
			break;

		case T_END:		/* eval_work_tree */
			if( going ) return(1);
			vecexp_ing=0;
			break;

		case T_DIM_ASSIGN:	/* eval_work_tree */
		case T_ASSIGN:		/* eval_work_tree */
			if( going ) return(1);

			eval_assignment(enp);
			break;

		case T_PREINC:		/* eval_work_tree */
		case T_POSTINC:		/* eval_work_tree */
			if( going ) return(1);
			dp = eval_obj_ref(VN_CHILD(enp,0));
			inc_obj(dp);
			break;

		case T_POSTDEC:
		case T_PREDEC:
			if( going ) return(1);
			dp = eval_obj_ref(VN_CHILD(enp,0));
			dec_obj(dp);
			break;



		default:		/* eval_work_tree */
			missing_case(enp,"eval_work_tree");
			break;
	}
	return(ret_val);
} /* end eval_work_tree */

