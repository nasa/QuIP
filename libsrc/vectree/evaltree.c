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
#include "../veclib/nvf.h"		/* show_obj_args, for debugging */
//#include "fio_api.h"
//#include "filetype.h"

#include "vectree.h"
#include "vt_native.h"

//#include "mlab.h"

#define MAX_HIDDEN_CONTEXTS	32

#define PUSH_CPAIR(cpp)		PUSH_ID_CONTEXT(CP_ID_CTX(cpp));	\
				PUSH_DOBJ_CONTEXT(CP_OBJ_CTX(cpp))

#define POP_CPAIR		POP_ID_CONTEXT;				\
				POP_DOBJ_CONTEXT

// BUG not thread-safe!?
static Dimension_Set *scalar_dsp=NULL;

/* BUG use of this global list make this not reentrant... */
static List *local_obj_lp=NO_LIST;
static void delete_local_objs(SINGLE_QSP_ARG_DECL);

static Item_Context *hidden_context[MAX_HIDDEN_CONTEXTS];
static int n_hidden_contexts=0;

Subrt *curr_srp=NO_SUBRT;
int scanning_args=0;
static Vec_Expr_Node *iteration_enp = NO_VEXPR_NODE;
static Vec_Expr_Node *eval_enp=NO_VEXPR_NODE;
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

#ifdef SGI
#include <alloca.h>
#endif

#ifdef QUIP_DEBUG
debug_flag_t eval_debug=0;
debug_flag_t scope_debug=0;
#endif /* QUIP_DEBUG */

/* local prototypes needed because of recursion... */
static void eval_obj_assignment(QSP_ARG_DECL Data_Obj *,Vec_Expr_Node *enp);
static int eval_work_tree(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj *dst_dp);
static Data_Obj *create_list_lhs(QSP_ARG_DECL Vec_Expr_Node *enp);


#define MAP_SUBSCRIPTS(src_dp,index_dp,enp)		map_subscripts(QSP_ARG  src_dp,index_dp,enp)
#define ASSIGN_OBJ_FROM_SCALAR(enp,dp,svp)		assign_obj_from_scalar(QSP_ARG  enp,dp,svp)
#define EVAL_MIXED_LIST(enp)			eval_mixed_list(QSP_ARG  enp)
#define EVAL_TYPECAST(enp,dst_dp)		eval_typecast(QSP_ARG  enp,dst_dp)
#define EVAL_BITMAP(dst_dp,enp)		eval_bitmap(QSP_ARG  dst_dp,enp)
#define EVAL_SUBSCRIPT1(dp,enp)		eval_subscript1(QSP_ARG  dp,enp)
#define EXEC_REFFUNC(enp)	exec_reffunc(QSP_ARG  enp)
#define FIND_CASE(enp,lval)	find_case(QSP_ARG  enp,lval)
#define EVAL_WORK_TREE(enp,dst_dp)		eval_work_tree(QSP_ARG  enp,dst_dp)
#define CREATE_LIST_LHS(enp)			create_list_lhs(QSP_ARG  enp)
#define CREATE_MATRIX(enp,shpp)			create_matrix(QSP_ARG  enp,shpp)
#define ASSIGN_ROW(dp,index,enp)		assign_row(QSP_ARG  dp,index,enp)
#define ASSIGN_ELEMENT(dp,ri,ci,enp)		assign_element(QSP_ARG  dp,ri,ci,enp)
#define MLAB_LHS(dp,enp)			mlab_lhs(QSP_ARG  dp,enp)
#define MLAB_TARGET(dp,enp)			mlab_target(QSP_ARG  dp,enp)
#define EVAL_OBJ_ID(enp)		eval_obj_id(QSP_ARG  enp)
#define EVAL_REF_TREE(enp,dst_idp)		eval_ref_tree(QSP_ARG  enp,dst_idp)
#define RUN_REFFUNC(srp,enp,dst_idp)		run_reffunc(QSP_ARG  srp,enp,dst_idp)
#define EVAL_SCALAR(svp,enp,prec)		eval_scalar(QSP_ARG  svp,enp,prec)
#define ASSIGN_SUBRT_ARGS(arg_enp,val_enp,srp,cpp)	assign_subrt_args(QSP_ARG  arg_enp,val_enp,srp,cpp)
#define ASSIGN_PTR_ARG(arg_enp,val_enp,curr_cpp,prev_cpp)	assign_ptr_arg(QSP_ARG  arg_enp,val_enp,curr_cpp,prev_cpp)
#define ASSIGN_OBJ_FROM_LIST(dp,enp,offset)	assign_obj_from_list(QSP_ARG  dp,enp,offset)
#define EVAL_PRINT_STAT(enp)			eval_print_stat(QSP_ARG  enp)
#define EVAL_OBJ_ASSIGNMENT(dp,enp)		eval_obj_assignment(QSP_ARG  dp,enp)
#define EVAL_DIM_ASSIGNMENT(dp,enp)		eval_dim_assignment(QSP_ARG  dp,enp)
#define EVAL_DECL_STAT(prec,enp,decl_flags)		eval_decl_stat(QSP_ARG  prec,enp,decl_flags)
#define EVAL_EXTERN_DECL(prec_p,enp,decl_flags)		eval_extern_decl(QSP_ARG  prec_p,enp,decl_flags)
#define D_SUBSCRIPT(dp,index)		d_subscript(QSP_ARG  dp , index )
#define C_SUBSCRIPT(dp,index)		c_subscript(QSP_ARG  dp , index )

#define GET_ARG_PTR(enp)	get_arg_ptr(QSP_ARG  enp)
#define SET_SCRIPT_ARGS(enp,index,qp,max_args)		set_script_args(QSP_ARG  enp,index,qp,max_args)
#define EVAL_INFO_STAT(enp)		eval_info_stat(QSP_ARG  enp)
#define EVAL_DISPLAY_STAT(enp)		eval_display_stat(QSP_ARG  enp)
#define GET_2_OPERANDS(enp,dpp1,dpp2,dst_dp)		get_2_operands(QSP_ARG  enp,dpp1,dpp2,dst_dp)

#define SUBSCR_TYPE(enp)	(VN_CODE(enp)==T_SQUARE_SUBSCR?SQUARE:CURLY)

#define max( n1 , n2 )		(n1>n2?n1:n2)

const char *(*native_string_func)(Vec_Expr_Node *)=eval_vt_native_string;
float (*native_flt_func)(Vec_Expr_Node *)=eval_vt_native_flt;
void (*native_work_func)(QSP_ARG_DECL  Vec_Expr_Node *)=eval_vt_native_work;
void (*native_assign_func)(Data_Obj *,Vec_Expr_Node *)=eval_vt_native_assignment;

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

static void eval_native_work(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	(*native_work_func)(QSP_ARG enp);
}

static void unset_object_warning(QSP_ARG_DECL  Vec_Expr_Node *enp, Data_Obj *dp)
{
	NODE_ERROR(enp);

	if( HAS_SOME_VALUES(dp) ){
		sprintf(ERROR_STRING,
			"unset_object_warning:  Object %s may be used before all values have been set.",
			OBJ_NAME(dp));
		ADVISE(ERROR_STRING);
	} else {
		sprintf(ERROR_STRING,
			"unset_object_warning:  Object %s is used before value has been set!?",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
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
//#ifdef CAUTIOUS
		/* shut up compiler */
		case PREC_INVALID:
		case PREC_NONE:
		case N_MACHINE_PRECS:
			//NWARN("CAUTIOUS:  get_long_scalar_value:  nonsense precision");
			assert( AERROR("get_long_scalar_value:  nonsense precision") );
			break;
//#endif /* CAUTIOUS */
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
//#ifdef CAUTIOUS
		/* shut up compiler */
		case PREC_NONE:
		case PREC_INVALID:
		case N_MACHINE_PRECS:
			//NWARN("CAUTIOUS:  get_dbl_scalar_value:  nonsense precision");
			assert( AERROR("get_dbl_scalar_value:  nonsense precision") );
			break;
//#endif /* CAUTIOUS */
	}
	return(dval);
}

#ifdef NOT_USED
void show_id(QSP_ARG_DECL  Identifier *idp)
{
	sprintf(msg_str,"Identifier %s at 0x%lx:  ",ID_NAME(idp), (int_for_addr)idp);
	prt_msg_frag(msg_str);
	switch(ID_TYPE(idp)){
		case ID_REFERENCE:  prt_msg("reference"); break;
		case ID_POINTER:  prt_msg("pointer"); break;
		case ID_STRING:  prt_msg("string"); break;
		default:
			prt_msg("");
			sprintf(DEFAULT_ERROR_STRING,"missing case in show_id (%d)",ID_TYPE(idp));
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
}
#endif /* NOT_USED */


static void prototype_mismatch(QSP_ARG_DECL  Vec_Expr_Node *enp1,Vec_Expr_Node *enp2)
{
	NODE_ERROR(enp1);
	NWARN("declaration conflicts with earlier prototype");
	NODE_ERROR(enp2);
	advise("original prototype");
}

static void assign_pointer(Pointer *ptrp, Reference *refp)
{
	SET_PTR_REF(ptrp, refp);
	/* the pointer declaration carries around the shape of its current contents? */
	/*
	copy_node_shape(QSP_ARG  PTR_DECL_VN(ptrp),OBJ_SHAPE(REF_OBJ(PTR_REF(ptrp))));
	*/
	SET_PTR_FLAG_BITS(ptrp, POINTER_SET);
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
//#ifdef CAUTIOUS
		default:
//			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  int_to_scalar:  unhandled target precision %s",
//				PREC_NAME(prec_p));
//			NERROR1(DEFAULT_ERROR_STRING);
//			IOS_RETURN
			assert( AERROR("int_to_scalar:  unhandled target precision") );
			break;
//#endif /* CAUTIOUS */
	}
}

#ifdef FOOBAR
static Data_Obj *make_global_scalar(QSP_ARG_DECL  const char *name,Precision *prec_p)
{
	Data_Obj *dp;

	set_global_ctx(SINGLE_QSP_ARG);
#ifdef HAVE_CUDA
	push_data_area(ram_area_p);
#endif // HAVE_CUDA
	dp = mk_scalar(QSP_ARG  name,prec_p);
#ifdef HAVE_CUDA
	pop_data_area();
#endif // HAVE_CUDA
	unset_global_ctx(SINGLE_QSP_ARG);
	return(dp);
}

static Data_Obj * check_global_scalar(QSP_ARG_DECL  const char *name,
					Data_Obj *prototype_dp,Data_Obj *dp)
{
	if( dp != NO_OBJ && OBJ_PREC(dp) != OBJ_PREC(prototype_dp) ){
		delvec(QSP_ARG  dp);
		dp=NO_OBJ;
	}

	if( dp == NO_OBJ ){
		/* We have to create this scalar in the global context,
		 * otherwise when the subroutine exits, and its context
		 * is deleted, this object will be deleted too -
		 * but our static pointer will still be dangling!?
		 */
		dp = make_global_scalar(QSP_ARG  name,OBJ_PREC_PTR(prototype_dp));
	}

	return(dp);
}
#endif // FOOBAR

/* dp_const should be used for floating point assignments... */

/*
 * dp_const - set object dp to the value indicated by svp
 *
 * The scalar value gets copied into a scalar object...
 * WHY???
 */

static Data_Obj *dp_const(QSP_ARG_DECL  Data_Obj *dp,Scalar_Value * svp)
{
//	static Data_Obj *const_dp=NO_OBJ;
	Vec_Obj_Args oa1, *oap=&oa1;
	int status;

	INIT_OBJ_ARG_PTR(oap)

#ifdef FOOBAR
	const_dp=check_global_scalar(QSP_ARG  "const_scalar",dp,const_dp);

	if( OBJ_PREC(const_dp) == PREC_BIT ){
		/* assign_scalar will only change 1 bit */
		*((bitmap_word *) OBJ_DATA_PTR(const_dp)) = 0;
	}

	/* now assign the value */
	assign_scalar(QSP_ARG  const_dp,svp);

	SET_OA_SRC1(oap,const_dp);
	SET_OA_SVAL(oap,0, (Scalar_Value *)OBJ_DATA_PTR(const_dp));
#endif // FOOBAR

	setvarg1(oap,dp);	// has to come first (clears *oap)
	SET_OA_SVAL(oap,0, svp );
	status = perf_vfunc(QSP_ARG  FVSET,oap);
	if( status < 0 )
		dp = NO_OBJ;

	return( dp );
} /* end dp_const() */

int zero_dp(QSP_ARG_DECL  Data_Obj *dp)
{
	Scalar_Value sval;

	switch(OBJ_PREC(dp)){
		case PREC_SP:  sval.u_f = 0.0; break;
		case PREC_DP:  sval.u_d = 0.0; break;
		case PREC_BY:  sval.u_b = 0; break;
		case PREC_IN:  sval.u_s = 0; break;
		case PREC_DI:  sval.u_l = 0; break;
		default:
//			ERROR1("CAUTIOUS:  unhandled machine precision in zero_dp()");
//			IOS_RETURN_VAL(-1)
			assert( AERROR("zero_dp:  unhandled precision") );
			break;
	}
	if( dp_const(QSP_ARG  dp,&sval) == NO_OBJ ) return(-1);
	return(0);
}

static int assign_obj_from_scalar(QSP_ARG_DECL  Vec_Expr_Node *enp,Data_Obj *dp,Scalar_Value *svp)
{
	if( dp_const(QSP_ARG  dp,svp) == NO_OBJ ){
		NODE_ERROR(enp);
		sprintf(ERROR_STRING,"Error assigning object %s from scalar value",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return(-1);
	}
	return(0);
}

void missing_case(QSP_ARG_DECL  Vec_Expr_Node *enp,const char *func_name)
{
	NODE_ERROR(enp);
	sprintf(ERROR_STRING,
		"Code %s (%d) not handled by %s switch",
		NNAME(enp),VN_CODE(enp),func_name);
	WARN(ERROR_STRING);
	DUMP_TREE(enp);
	advise("");
}

static void xpose_data(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	float *fr, *to;
	dimension_t i,j,k;

	if( OBJ_ROWS(dpto) != OBJ_COLS(dpfr) ){
		sprintf(ERROR_STRING,
	"xpose_data:  # destination rows object %s (%d) should match # source cols object %s (%d)",
			OBJ_NAME(dpto),OBJ_ROWS(dpto),OBJ_NAME(dpfr),OBJ_COLS(dpfr));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_COLS(dpto) != OBJ_ROWS(dpfr) ){
		sprintf(ERROR_STRING,
	"xpose_data:  # destination cols object %s (%d) should match # source rows object %s (%d)",
			OBJ_NAME(dpto),OBJ_COLS(dpto),OBJ_NAME(dpfr),OBJ_ROWS(dpfr));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_PREC(dpto) != PREC_SP ){
		WARN("Sorry, now can only transpose float objects");
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

static Data_Obj *map_source_dp=NO_OBJ;

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
			NODE_ERROR(iteration_enp);			\
			sprintf(ERROR_STRING,				\
"map_iteration:  GET_MAP_OFFSET:  index %d is out of range for %s dimension (%d) of source object %s",		\
	indices[i_dim],dimension_name[i_dim],				\
	OBJ_TYPE_DIM(map_source_dp,i_dim),				\
	OBJ_NAME(map_source_dp));					\
			WARN(ERROR_STRING);				\
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
				NODE_ERROR(iteration_enp);		\
				sprintf(ERROR_STRING,			\
"map_iteration:  GET_MAP_WEIGHTS:  index %g (rounded to %d) out of range for %s dimension (%d) of src %s",	\
		d,lower_index[i_dim],dimension_name[i_dim],		\
		OBJ_TYPE_DIM(map_source_dp,i_dim),			\
		OBJ_NAME(map_source_dp));				\
				WARN(ERROR_STRING);			\
			}						\
			lower_index[i_dim]=upper_index[i_dim]=0;	\
			weights[i_dim]=0.0;				\
		} else if( upper_index[i_dim] >= 			\
			(incr_t) OBJ_TYPE_DIM(map_source_dp,i_dim) ){	\
			if( expect_perfection ){			\
				NODE_ERROR(iteration_enp);		\
				sprintf(ERROR_STRING,			\
"map_iteration:  GET_MAP_WEIGHTS:  index %g (rounded to %d) out of range for %s dimension (%d) of src %s",	\
		d,upper_index[i_dim],dimension_name[i_dim],		\
				OBJ_TYPE_DIM(map_source_dp,i_dim),	\
		OBJ_NAME(map_source_dp));				\
				WARN(ERROR_STRING);			\
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
/*sprintf(ERROR_STRING,"GET_MAP_W:  i = %d     ui = %d   li = %d    w = %f",\
i,upper_index[i],lower_index[i],weights[i]);\
advise(ERROR_STRING);*/\
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
/*for(i=0;i<ns;i++){\
sprintf(ERROR_STRING,"%d, %d     i = %d   sample_offset[i] = %d   sample_weight[i] = %g",\
i_dst,i_index,i,sample_offset[i],sample_weight[i]);\
advise(ERROR_STRING);\
}*/\
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
				WARN(ERROR_STRING);							\
				break;
//#ifdef CAUTIOUS

#define INVALID_MAP_CASES(dp)						\
									\
		/* shouldn't happen, but these are valid enum's */	\
		case PREC_NONE:						\
		case N_MACHINE_PRECS:					\
		default:						\
			/*sprintf(ERROR_STRING,				\
		"CAUTIOUS:  map_interation:  illegal machine precision (object %s).",			\
				OBJ_NAME(dp));				\
			ERROR1(ERROR_STRING);				\
			IOS_RETURN					\
			offset = 0;*/	/* quiet compiler */		\
			assert(AERROR("map_iteration:  illegal machine precision!?"));	\
			break;

//#else /* ! CAUTIOUS */
//
//#define INVALID_MAP_CASES(dp)
//
//#endif /* ! CAUTIOUS */


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
			WARN(ERROR_STRING);
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
				WARN(ERROR_STRING);
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
				WARN(ERROR_STRING);
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
	if( OBJ_PARENT(dp) != NO_OBJ ) note_partial_assignment(OBJ_PARENT(dp));
}

void note_assignment(Data_Obj *dp)
{
	SET_OBJ_FLAG_BITS(dp, DT_ASSIGNED);
	if( OBJ_PARENT(dp) != NO_OBJ ) note_partial_assignment(OBJ_PARENT(dp));
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
		WARN("map_subscripts:  Sorry, curly subscripts are not correctly handled...");

	/* For now, we create dst_dp to have the same dimensions as the index array... */
	SET_DIMENSION(dsp,0, OBJ_TYPE_DIM(src_dp,0) );	/* copy tdim from src_dp */
	for(i=1;i<N_DIMENSIONS;i++)
		SET_DIMENSION(dsp,i, OBJ_TYPE_DIM(index_dp,i) );	/* BUG need to do something better */

	dst_dp=make_local_dobj(QSP_ARG  dsp,OBJ_PREC_PTR(src_dp));

	if( dst_dp == NO_OBJ )
		return(dst_dp);

	/* Now check the sizes - we might like to use dp_same_size(), but we allow tdim to differ  */

	if( !dp_same_dims(QSP_ARG  dst_dp,index_dp,1,N_DIMENSIONS-1,"map_subscripts") ){
		NODE_ERROR(enp);
		sprintf(ERROR_STRING,"map_subscripts:  objects %s and %s should have the same shape",
			OBJ_NAME(dst_dp),OBJ_NAME(index_dp));
		WARN(ERROR_STRING);
		return(NO_OBJ);
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
	dpair_iterate(QSP_ARG  dst_dp,index_dp,map_iteration);
	iteration_enp = NO_VEXPR_NODE;

	//SET_OBJ_FLAG_BITS(dst_dp, DT_ASSIGNED);
	note_assignment(dst_dp);
	return(dst_dp);
} /* end map_subscripts */

static int do_vvfunc(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr1,Data_Obj *dpfr2,Vec_Func_Code code)
{
	Vec_Obj_Args oa1, *oap=&oa1;
	int retval;

	if( code == FVMUL && COMPLEX_PRECISION(OBJ_PREC(dpfr2)) && ! COMPLEX_PRECISION(OBJ_PREC(dpfr1)) ){
		setvarg3(oap,dpto,dpfr2,dpfr1);
	} else {
		setvarg3(oap,dpto,dpfr1,dpfr2);
	}
	retval = perf_vfunc(QSP_ARG  code,oap) ;

	return( retval );
}

static int do_vsfunc(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Scalar_Value *svp,Vec_Func_Code code)
{
#ifdef FOOBAR
	static Data_Obj *scal_dp=NO_OBJ;
#endif // FOOBAR
	Vec_Obj_Args oa1, *oap=&oa1;
	int retval;

#ifdef FOOBAR
	scal_dp = check_global_scalar(QSP_ARG  "vsfunc_scalar",dpfr,scal_dp);

	assign_scalar(QSP_ARG  scal_dp,svp);
	setvarg2(oap,dpto,dpfr);
	//SET_OA_SRC1(oap, scal_dp);
	SET_OA_SVAL(oap,0, (Scalar_Value *)OBJ_DATA_PTR(scal_dp));
#endif // FOOBAR

	setvarg2(oap,dpto,dpfr);
	SET_OA_SVAL(oap,0, svp );

	retval = perf_vfunc(QSP_ARG  code,oap);

	return( retval );
} // do_vsfunc

static int do_un0func(QSP_ARG_DECL Data_Obj *dpto,Vec_Func_Code code)
{
	Vec_Obj_Args oa1, *oap=&oa1;
	int retval;

	setvarg1(oap,dpto);
	retval = perf_vfunc(QSP_ARG  code,oap);

	return( retval );
}

static int do_unfunc(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Vec_Func_Code code)
{
	Vec_Obj_Args oa1;
	int retval;

	setvarg2(&oa1,dpto,dpfr);
	retval = perf_vfunc(QSP_ARG  code,&oa1) ;

	return( retval );
}

#ifdef UNUSED
static Scalar_Value * take_inner(Data_Obj *dp1,Data_Obj *dp2)
{
	static Scalar_Value sval;

//#ifdef CAUTIOUS
//	if( dp1==NO_OBJ || dp2==NO_OBJ ){
//		sprintf(ERROR_STRING,"CAUTIOUS: take_inner: passed null arg!?");
//		WARN(ERROR_STRING);
//		return(NULL);
//	}
//#endif /* CAUTIOUS */
	assert( dp1 != NO_OBJ && dp2 != NO_OBJ );

	sprintf(ERROR_STRING,"take_inner %s %s:  unimplemented",
		OBJ_NAME(dp1),OBJ_NAME(dp2));
	WARN(ERROR_STRING);

	switch( OBJ_MACH_PREC(dp1) ){
		case PREC_BY:  sval.u_b = 0; break;
		case PREC_IN:  sval.u_s = 0; break;
		case PREC_DI:  sval.u_l = 0; break;
		case PREC_SP:  sval.u_f = 0.0; break;
		case PREC_DP:  sval.u_d = 0.0; break;
		case PREC_UBY:  sval.u_ub = 0; break;
		case PREC_UIN:  sval.u_us = 0; break;
		case PREC_UDI:  sval.u_ul = 0; break;
//#ifdef CAUTIOUS
		/* just to shut the compiler up */
		case PREC_NONE:
		case N_MACHINE_PRECS:
//			sprintf(ERROR_STRING,
//				"CAUTIOUS:  take_inner:  %s has nonsense machine precision",
//				OBJ_NAME(dp1));
//			WARN(ERROR_STRING);
			assert( AERROR("take_inner:  nonsense precision!?") );
			/* can't happen? */
			break;
//#endif /* CAUTIOUS */

	}
	return(&sval);
}
#endif /* UNUSED */

static void assign_string(QSP_ARG_DECL  Identifier *idp, const char *str, Vec_Expr_Node *enp)
{
	if( ! IS_STRING_ID(idp) ){
		NODE_ERROR(enp);
		sprintf(DEFAULT_ERROR_STRING,"assign_string:  identifier %s (type %d) does not refer to a string",
			ID_NAME(idp),ID_TYPE(idp));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	/* copy_string(idp->id_sbp,str); */
	copy_string(REF_SBUF(ID_REF(idp)),str);
}

static Identifier *ptr_for_string(QSP_ARG_DECL  const char *s,Vec_Expr_Node *enp)
{
	static int n_auto_strs=1;
	char idname[LLEN];
	Identifier *idp;

	/* We need to make an object and a reference... */

	sprintf(idname,"Lstr.%d",n_auto_strs++);
	idp = new_id(QSP_ARG  idname);
sprintf(ERROR_STRING,"ptr_for_string:  creating id %s",idname);
advise(ERROR_STRING);
	SET_ID_TYPE(idp, ID_STRING);
#ifdef FOOBAR
	//idp->id_sbp = getbuf(sizeof(String_Buf));
	//idp->id_sbp->sb_buf = NULL;
	//idp->id_sbp->sb_size = 0;
#endif /* FOOBAR */

	/* Can't do this, because refp is in a union w/ sbp... */
	SET_ID_REF(idp, NEW_REFERENCE );
	SET_REF_TYPE(ID_REF(idp), STR_REFERENCE );
	SET_REF_ID(ID_REF(idp), idp );
	SET_REF_DECL_VN(ID_REF(idp), NO_VEXPR_NODE );
	/* SET_REF_OBJ(ID_REF(idp), NO_OBJ ); */
	SET_REF_SBUF(ID_REF(idp), NEW_STRINGBUF );
	REF_SBUF(ID_REF(idp))->sb_buf = NULL;
	REF_SBUF(ID_REF(idp))->sb_size = 0;

	assign_string(QSP_ARG  idp,s,enp);

	return( idp );
}

/* Get the data object for this value node.
 * We use this routine for call-by-reference.
 */

static Identifier *get_arg_ptr(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Identifier *idp;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_STATIC_OBJ:		/* get_arg_ptr */
			NODE_ERROR(enp);
			sprintf(ERROR_STRING,"object %s not properly referenced, try prepending &",OBJ_NAME(VN_OBJ(enp)));
			advise(ERROR_STRING);
			idp = GET_ID(OBJ_NAME(VN_OBJ(enp)));
			return(idp);
			break;

		case T_DYN_OBJ:		/* get_arg_ptr */
			NODE_ERROR(enp);
			sprintf(ERROR_STRING,"object %s not properly referenced, try prepending &",VN_STRING(enp));
			advise(ERROR_STRING);
			idp = GET_ID(VN_STRING(enp));
			return(idp);
			break;

		case T_REFERENCE:
		case T_POINTER:
		case T_STR_PTR:
			return( EVAL_PTR_REF(enp,EXPECT_PTR_SET) );


		case T_SET_STR:			/* get_arg_ptr */
			/* an assignment statement as a function arg...
			 * we need to execute it!
			 */
		case T_STRING:
			/* we need to make up an object for this string...
			 * BUG this is going to be a memory leak!?
			 */
			return( ptr_for_string( QSP_ARG  EVAL_STRING(enp), enp ) );
			break;

		default:
			MISSING_CASE(enp,"get_arg_ptr");
			break;
	}
	return(NO_IDENTIFIER);
}

static Data_Obj *get_id_obj(QSP_ARG_DECL  const char *name, Vec_Expr_Node *enp)
{
	Identifier *idp;

	idp = /* GET_ID */ ID_OF(name);

//#ifdef CAUTIOUS
//	if( idp==NO_IDENTIFIER ){
//		if( mode_is_matlab ) return(NO_OBJ);	/* not an error in matlab */
//		NODE_ERROR(enp);
//		sprintf(ERROR_STRING,"CAUTIOUS:  missing identifier object (%s) #2!?",name);
//		WARN(ERROR_STRING);
//		return(NO_OBJ);
//	}
	assert( idp != NO_IDENTIFIER );

//	if( ! IS_REFERENCE(idp) ){
//		NODE_ERROR(enp);
//		sprintf(ERROR_STRING,"CAUTIOUS:  identifier %s is not an object!?",
//			ID_NAME(idp));
//		WARN(ERROR_STRING);
//		return(NO_OBJ);
//	}
	assert( IS_REFERENCE(idp) );

//	if( strcmp(ID_NAME(idp),OBJ_NAME(REF_OBJ(ID_REF(idp)))) ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  get_id_obj:  identifier %s points to object %s!?",
//			ID_NAME(idp),OBJ_NAME(REF_OBJ(ID_REF(idp))));
//		WARN(ERROR_STRING);
//	}
	assert( ! strcmp(ID_NAME(idp),OBJ_NAME(REF_OBJ(ID_REF(idp)))) );

	{
		Data_Obj *dp;
		dp = DOBJ_OF(ID_NAME(idp));
//		if( dp == NO_OBJ ){
//			NODE_ERROR(enp);
//			sprintf(ERROR_STRING,
//		"CAUTIOUS:  get_id_obj:  object identifier %s exists but object is missing!?",
//				ID_NAME(idp));
//			WARN(ERROR_STRING);
////show_context_stack(QSP_ARG  id_itp);
////show_context_stack(QSP_ARG  dobj_itp);
//list_dobjs(SINGLE_QSP_ARG);
//sprintf(ERROR_STRING,"object pointed to by identifier %s:",ID_NAME(idp));
//advise(ERROR_STRING);
//LONGLIST(REF_OBJ(ID_REF(idp)));
//			return(dp);
//		}
		assert( dp != NO_OBJ );

//		if( dp != REF_OBJ(ID_REF(idp)) ){
//			sprintf(ERROR_STRING,
//		"CAUTIOUS:  identifier %s pointer 0x%lx does not match object %s addr 0x%lx",
//				ID_NAME(idp),(int_for_addr)REF_OBJ(ID_REF(idp)),OBJ_NAME(dp),(int_for_addr)dp);
//			WARN(ERROR_STRING);
//		}
		assert( dp == REF_OBJ(ID_REF(idp)) );
	}
//#endif /* CAUTIOUS */

	return(REF_OBJ(ID_REF(idp)));
} /* get_id_obj */

static Function_Ptr *eval_funcptr(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Function_Ptr *fpp=NO_FUNC_PTR;
	Identifier *idp;

	switch(VN_CODE(enp)){
		case T_FUNCPTR_DECL:
		case T_FUNCPTR:
			idp=ID_OF(VN_STRING(enp));
			/* BUG chould check that type is funcptr */
			/* BUG chould check that idp is valid */
//#ifdef CAUTIOUS
//			if( idp == NO_IDENTIFIER ){
//				sprintf(ERROR_STRING,"CAUTIOUS:  eval_funcptr:  missing identifier %s",VN_STRING(enp));
//				WARN(ERROR_STRING);
//				DUMP_TREE(enp);
//				return NULL;
//			}
//#endif /* CAUTIOUS */
			assert( idp != NO_IDENTIFIER );

			fpp = ID_FUNC(idp);
			break;
		default:
			MISSING_CASE(enp,"eval_funcptr");
			break;
	}
	return(fpp);
}


static Subrt *eval_funcref(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Subrt *srp;
	Function_Ptr *fpp;

	srp=NO_SUBRT;
	switch(VN_CODE(enp) ){
		case T_FUNCREF:
			srp=VN_SUBRT(enp);
			break;
		case T_FUNCPTR:
			fpp = eval_funcptr(QSP_ARG  enp);
			srp = fpp->fp_srp;
			break;
		default:
			MISSING_CASE(enp,"eval_funcref");
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

static int assign_ptr_arg(QSP_ARG_DECL Vec_Expr_Node *arg_enp,Vec_Expr_Node *val_enp, Context_Pair *curr_cpp,Context_Pair *prev_cpp)
{
	Identifier *idp, *src_idp;

	/* we want this object to be equivalenced to the calling obj */

/*
sprintf(ERROR_STRING,"assign_ptr_arg %s %s:  calling pop_subrt_pair",node_desc(arg_enp),node_desc(val_enp));
advise(ERROR_STRING);
*/

	POP_SUBRT_CPAIR(curr_cpp,SR_NAME(curr_srp));
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"assign_ptr_arg:  current contexts %s, %s popped",CONTEXT_NAME(CP_ID_CTX(curr_cpp)),
CONTEXT_NAME(CP_OBJ_CTX(curr_cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( prev_cpp != NO_CONTEXT_PAIR ){
/*
sprintf(ERROR_STRING,"assign_ptr_arg %s %s:  restoring previous context",node_desc(arg_enp),node_desc(val_enp));
advise(ERROR_STRING);
*/

		//PUSH_ITEM_CONTEXT(id_itp,CP_ID_CTX(prev_cpp));
		//PUSH_ITEM_CONTEXT(dobj_itp,CP_OBJ_CTX(prev_cpp));
		PUSH_CPAIR(prev_cpp);

#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"assign_ptr_arg:  previous contexts %s, %s pushed",CONTEXT_NAME(CP_ID_CTX(prev_cpp)),
CONTEXT_NAME(CP_OBJ_CTX(prev_cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	}

	src_idp = GET_ARG_PTR(val_enp);		/* what if the val_enp is a string?? */

	if( prev_cpp != NO_CONTEXT_PAIR ){
/*
sprintf(ERROR_STRING,"assign_ptr_arg %s %s:  popping previous context",node_desc(arg_enp),node_desc(val_enp));
advise(ERROR_STRING);
*/
		//pop_item_context(QSP_ARG  id_itp);
		//pop_item_context(QSP_ARG  dobj_itp);
		POP_ID_CONTEXT;
		POP_DOBJ_CONTEXT;
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"assign_ptr_arg:  previous contexts %s, %s popped",CONTEXT_NAME(CP_ID_CTX(prev_cpp)),
CONTEXT_NAME(CP_OBJ_CTX(prev_cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	}

/*
sprintf(ERROR_STRING,"assign_ptr_arg %s %s:  pushing current context",node_desc(arg_enp),node_desc(val_enp));
advise(ERROR_STRING);
*/
	//PUSH_ITEM_CONTEXT(id_itp,CP_ID_CTX(curr_cpp));
	//PUSH_ITEM_CONTEXT(dobj_itp,CP_OBJ_CTX(curr_cpp));
	PUSH_CPAIR(curr_cpp);

#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"assign_ptr_arg:  current contexts %s, %s pushed",CONTEXT_NAME(CP_ID_CTX(curr_cpp)),
CONTEXT_NAME(CP_OBJ_CTX(curr_cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( src_idp == NO_IDENTIFIER ){
		WARN("Missing source object!?");
		return(-1);
	}

	idp = GET_ID(VN_STRING(arg_enp));
	if( idp==NO_IDENTIFIER ) return(-1);


	/* assign_ptr_arg */

	switch(ID_TYPE(idp)){
		case ID_POINTER:
			if( IS_REFERENCE(src_idp) ){
				assign_pointer(ID_PTR(idp), ID_REF(src_idp) );
				/* propagate_shape? */
				return(0);
			} else if( IS_POINTER(src_idp) ){
				assign_pointer(ID_PTR(idp), PTR_REF(ID_PTR(src_idp)) );
				/* propagate_shape? */
				return(0);
			} else if( IS_STRING_ID(src_idp) ){
				assign_pointer(ID_PTR(idp),ID_REF(src_idp));
				return(0);
			} else {
				NODE_ERROR(val_enp);
				sprintf(ERROR_STRING,"argval %s is not a reference or a pointer!?",
					ID_NAME(src_idp));
				WARN(ERROR_STRING);
				return(-1);
			}
			/* NOTREACHED */
			return(-1);
		case ID_STRING:
			if( ! IS_STRING_ID(src_idp) ){
				NODE_ERROR(val_enp);
				sprintf(ERROR_STRING,"argval %s is not a string!?",
					ID_NAME(idp));
				WARN(ERROR_STRING);
				return(-1);
			}
//#ifdef CAUTIOUS
//			if( REF_SBUF(ID_REF(src_idp))->sb_buf == NULL ){
//				NODE_ERROR(val_enp);
//				sprintf(ERROR_STRING,
//			"CAUTIOUS:  assign_ptr_arg STRING %s:  source buffer from %s is NULL!?",
//					node_desc(arg_enp),node_desc(val_enp));
//				WARN(ERROR_STRING);
//				return(-1);
//			}
//#endif /* CAUTIOUS */
			assert( REF_SBUF(ID_REF(src_idp))->sb_buf != NULL );

			copy_string(REF_SBUF(ID_REF(idp)),REF_SBUF(ID_REF(src_idp))->sb_buf);
			/* BUG need to set string set flag */
			return(0);
		default:
			WARN("unhandled case in assign_ptr_args");
			return(-1);
	}
	/* NOTREACHED */
	return(-1);
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

static Data_Obj * complement_bitmap(QSP_ARG_DECL  Data_Obj *dp)
{
	u_long *wp;
	int n_words;
	static u_long complement_bits=0;

	/* BUG here we assume the bitmap is contiguous */
//#ifdef CAUTIOUS
//	if( ! IS_CONTIGUOUS(dp) ){
//		LONGLIST(dp);
//		sprintf(ERROR_STRING,"complement_bitmap:  CAUTIOUS:  arg %s is not contiguous",OBJ_NAME(dp));
//		ERROR1(ERROR_STRING);
//		IOS_RETURN_VAL(NULL)
//	}
//#endif /* CAUTIOUS */
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
		new_dp = dup_obj(QSP_ARG  dp,s=localname());

//#ifdef CAUTIOUS
//		if( new_dp == NO_OBJ ){
//			sprintf(ERROR_STRING,"CAUTIOUS:  complement_bitmap:  Unable to create object %s",s);
//			ERROR1(ERROR_STRING);
//			IOS_RETURN_VAL(NULL)
//		}
//#endif /* CAUTIOUS */
		assert( new_dp != NO_OBJ );

		dp_copy(QSP_ARG  new_dp,dp);
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

static void eval_scalar(QSP_ARG_DECL Scalar_Value *svp, Vec_Expr_Node *enp, Precision *prec_p)
{
	eval_enp = enp;

	/* should we call eval_flt_exp for all??? */

	switch(PREC_CODE(prec_p)&MACH_PREC_MASK){
		case PREC_SP:  svp->u_f = (float) EVAL_FLT_EXP(enp); break;
		case PREC_DP:  svp->u_d = EVAL_FLT_EXP(enp); break;
		case PREC_BY:  svp->u_b = (char) EVAL_INT_EXP(enp); break;
		case PREC_IN:  svp->u_s = (short) EVAL_INT_EXP(enp); break;
		case PREC_DI:  svp->u_l = (int32_t) EVAL_INT_EXP(enp); break;
		case PREC_LI:  svp->u_ll = (int64_t) EVAL_INT_EXP(enp); break;
		case PREC_ULI:  svp->u_ull = (uint64_t) EVAL_INT_EXP(enp); break;
		case PREC_UDI:  svp->u_ul = (uint32_t) EVAL_INT_EXP(enp); break;
		case PREC_UIN:  svp->u_us = (u_short) EVAL_INT_EXP(enp); break;
		case PREC_UBY:  svp->u_ub = (u_char) EVAL_INT_EXP(enp); break;
		default:
//			WARN("CAUTIOUS:  unhandled machine precision in eval_scalar()");
			assert( AERROR("eval_scalar:  unhandled machine precision") );
			break;
	}
}

static Data_Obj *create_bitmap( QSP_ARG_DECL  Dimension_Set *src_dsp )
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

	bmdp = make_local_dobj(QSP_ARG  dsp,prec_for_code(PREC_BIT));
	return(bmdp);
}

static Data_Obj *dup_bitmap(QSP_ARG_DECL  Data_Obj *dp)
{
//#ifdef CAUTIOUS
//	if( UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
//		sprintf(ERROR_STRING,"dup_bitmap:  can't dup from unknown shape object %s",OBJ_NAME(dp));
//		ERROR1(ERROR_STRING);
//		IOS_RETURN_VAL(NULL)
//	}
//#endif /* CAUTIOUS */
	assert( ! UNKNOWN_SHAPE(OBJ_SHAPE(dp)) );

	return( create_bitmap(QSP_ARG  OBJ_TYPE_DIMS(dp) ) );
}

/* vs_bitmap:  vsm_lt etc. */

static Data_Obj * vs_bitmap(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *dp,Scalar_Value *svp,Vec_Func_Code code)
{
	Data_Obj *bmdp;
	Vec_Obj_Args oa1;
#ifdef FOOBAR
	static Data_Obj *vsbm_sclr_dp=NO_OBJ;
#endif // FOOBAR
	int status;

//#ifdef CAUTIOUS
//	switch(code){
//		case FVSMLT:
//		case FVSMGT:
//		case FVSMLE:
//		case FVSMGE:
//		case FVSMNE:
//		case FVSMEQ:
//			break;
//		default:
//			sprintf(ERROR_STRING,
//				"CAUTIOUS:  unexpected code (%d) in vs_bitmap",code);
//			WARN(ERROR_STRING);
//			return(NO_OBJ);
//	}
//#endif /* CAUTIOUS */
	assert( code == FVSMLT || code == FVSMGT || code == FVSMLE ||
	        code == FVSMGE || code == FVSMNE || code == FVSMEQ );

	if( dst_dp == NO_OBJ ){
		bmdp = dup_bitmap(QSP_ARG  dp);
//#ifdef CAUTIOUS
//	if( bmdp == NO_OBJ ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  vs_bitmap:  unable to dup bitmap for obj %s",OBJ_NAME(dp));
//		ERROR1(ERROR_STRING);
//		IOS_RETURN_VAL(NULL)
//	}
//#endif /* CAUTIOUS */
		assert( bmdp != NO_OBJ );
	}
	else
		bmdp = dst_dp;

	setvarg2(&oa1,bmdp,dp);	// dbm is the same as dest...

	SET_OA_SVAL(&oa1,0, svp );

	status = perf_vfunc(QSP_ARG  code,&oa1);

	if( status )
		bmdp=NO_OBJ;

	return(bmdp);
	/* BUG? when do we delete the bitmap??? */

} /* end vs_bitmap() */

/* Like dup_bitmap, but we need to use this version w/ vv_bitmap because
 * the two operands might have different shape (outer op).
 * Can we use get_mating_shape?
 */

static Data_Obj *dup_bitmap2(QSP_ARG_DECL  Data_Obj *dp1, Data_Obj *dp2)
{
	Shape_Info *shpp;

	shpp = product_shape(OBJ_SHAPE(dp1),OBJ_SHAPE(dp2));
	if( shpp == NO_SHAPE ) return(NO_OBJ);

	return( create_bitmap(QSP_ARG  SHP_TYPE_DIMS(shpp) ) );
}


static Data_Obj * vv_bitmap(QSP_ARG_DECL  Data_Obj *dst_dp,Data_Obj *dp1,Data_Obj *dp2,Vec_Func_Code code)
{
	Data_Obj *bmdp;
	Vec_Obj_Args oa1;
	int status;

//#ifdef CAUTIOUS
//	switch(code){
//		case FVVMLT:
//		case FVVMGT:
//		case FVVMLE:
//		case FVVMGE:
//		case FVVMNE:
//		case FVVMEQ:
//			break;
//		default:
//			WARN("CAUTIOUS:  unexpected code in vv_bitmap");
//			return(NO_OBJ);
//	}
//#endif /* CAUTIOUS */
	assert( code == FVVMLT || code == FVVMGT || code == FVVMLE ||
	        code == FVVMGE || code == FVVMNE || code == FVVMEQ );

	if( dst_dp != NO_OBJ )
		bmdp = dst_dp;
	else {
		bmdp = dup_bitmap2(QSP_ARG  dp1,dp2);	/* might be an outer op */
	}

	setvarg3(&oa1,bmdp,dp1,dp2);
	status = perf_vfunc(QSP_ARG  code,&oa1);

	if( status < 0 )
		bmdp=NO_OBJ;

	return(bmdp);
}

static Data_Obj *eval_bitmap(QSP_ARG_DECL Data_Obj *dst_dp, Vec_Expr_Node *enp)
{
	Data_Obj *bm_dp1,*bm_dp2,*dp,*dp2;
	long ival;

	eval_enp = enp;

	switch( VN_CODE(enp) ){
		/* ALL_OBJREF_CASES??? */
		case T_STATIC_OBJ:		/* eval_bitmap */
		case T_DYN_OBJ:			/* eval_bitmap */
			dp = EVAL_OBJ_REF(enp);
			return(dp);
			break;

		case T_BOOL_AND:
			if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) ){
				ival = EVAL_INT_EXP(VN_CHILD(enp,0));
				bm_dp1 = EVAL_BITMAP(dst_dp,VN_CHILD(enp,1));
				if( !ival )
					constant_bitmap(bm_dp1,0L);
				return(bm_dp1);
			} else if( SCALAR_SHAPE( VN_SHAPE(VN_CHILD(enp,1)) ) ){
				ival = EVAL_INT_EXP(VN_CHILD(enp,1));
				bm_dp1 = EVAL_BITMAP(dst_dp,VN_CHILD(enp,0));
				if( !ival )
					constant_bitmap(bm_dp1,0L);
				return(bm_dp1);
			} else {
				bm_dp1 = EVAL_BITMAP(dst_dp,VN_CHILD(enp,0));
				bm_dp2 = EVAL_BITMAP(NO_OBJ,VN_CHILD(enp,1));
				if( do_vvfunc(QSP_ARG  bm_dp1,bm_dp1,bm_dp2,FVAND) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating bitmap");
					return(NO_OBJ);
				}
				return(bm_dp1);
			}
			break;
		case T_BOOL_OR:
			if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) ){
				ival = EVAL_INT_EXP(VN_CHILD(enp,0));
				bm_dp1 = EVAL_BITMAP(dst_dp,VN_CHILD(enp,1));
				if( ival )
					constant_bitmap(bm_dp1,0xffffffff);
				return(bm_dp1);
			} else if( SCALAR_SHAPE( VN_SHAPE(VN_CHILD(enp,1)) ) ){
				ival = EVAL_INT_EXP(VN_CHILD(enp,1));
				bm_dp1 = EVAL_BITMAP(dst_dp,VN_CHILD(enp,0));
				if( ival )
					constant_bitmap(bm_dp1,0xffffffff);
				return(bm_dp1);
			} else {
				bm_dp1 = EVAL_BITMAP(dst_dp,VN_CHILD(enp,0));
				bm_dp2 = EVAL_BITMAP(NO_OBJ,VN_CHILD(enp,1));
				if( do_vvfunc(QSP_ARG  bm_dp1,bm_dp1,bm_dp2,FVOR) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating bitmap");
					return(NO_OBJ);
				}
				return(bm_dp1);
			}
			break;
		case T_BOOL_XOR:
			if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) ){
				ival = EVAL_INT_EXP(VN_CHILD(enp,0));
				bm_dp1 = EVAL_BITMAP(dst_dp,VN_CHILD(enp,1));
				if( ival ){
					bm_dp1 = complement_bitmap(QSP_ARG  bm_dp1);
				}
				return(bm_dp1);
			} else if( SCALAR_SHAPE( VN_SHAPE(VN_CHILD(enp,1)) ) ){
				ival = EVAL_INT_EXP(VN_CHILD(enp,1));
				bm_dp1 = EVAL_BITMAP(dst_dp,VN_CHILD(enp,0));
				if( ival ){
					bm_dp1 = complement_bitmap(QSP_ARG  bm_dp1);
				}
				return(bm_dp1);
			} else {
				bm_dp1 = EVAL_BITMAP(dst_dp,VN_CHILD(enp,0));
				bm_dp2 = EVAL_BITMAP(NO_OBJ,VN_CHILD(enp,1));
				if( do_vvfunc(QSP_ARG  bm_dp1,bm_dp1,bm_dp2,FVXOR) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating bitmap");
					return(NO_OBJ);
				}
				return(bm_dp1);
			}
			break;
		case T_BOOL_NOT:
			bm_dp1 = EVAL_BITMAP(dst_dp,VN_CHILD(enp,0));
			bm_dp1 = complement_bitmap(QSP_ARG  bm_dp1);
			return(bm_dp1);
			break;

		ALL_NUMERIC_COMPARISON_CASES			/* eval_bitmap */

//#ifdef CAUTIOUS
//			if( SCALAR_SHAPE( VN_SHAPE(VN_CHILD(enp,0)) ) ){
//		WARN("CAUTIOUS:  scalar comparison operand should have been swapped!?");
//				return(NO_OBJ);
//			}
//#endif /* CAUTIOUS */
			assert( ! SCALAR_SHAPE( VN_SHAPE(VN_CHILD(enp,0)) ) );

			if( SCALAR_SHAPE( VN_SHAPE(VN_CHILD(enp,1)) ) ){
				Scalar_Value sval;
				dp = EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
//#ifdef CAUTIOUS
				//VERIFY_DATA_TYPE(enp,ND_FUNC,"eval_bitmap")
				ASSERT_NODE_DATA_TYPE(enp,ND_FUNC)

//				if( dp == NO_OBJ ){
//					NODE_ERROR(enp);
//					advise("CAUTIOUS:  missing object");
//					return(NO_OBJ);
//				}
				assert( dp != NO_OBJ );
//#endif /* CAUTIOUS */
				EVAL_SCALAR(&sval,VN_CHILD(enp,1),OBJ_PREC_PTR(dp));
				bm_dp1 = vs_bitmap(QSP_ARG  dst_dp,dp,&sval,VN_BM_CODE(enp));

if( bm_dp1 == NO_OBJ ){
NODE_ERROR(enp);
sprintf(ERROR_STRING,"bad vs_bitmap, %s",node_desc(enp));
ERROR1(ERROR_STRING);
IOS_RETURN_VAL(NULL)
}
			} else {
				/* both vectors */
				dp = EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
				dp2 = EVAL_OBJ_EXP(VN_CHILD(enp,1),NO_OBJ);
				bm_dp1 = vv_bitmap(QSP_ARG  dst_dp,dp,dp2,VN_BM_CODE(enp));
			}
			return(bm_dp1);
			break;

		default:
			MISSING_CASE(enp,"eval_bitmap");
			break;
	}
	return(NO_OBJ);
} /* end eval_bitmap() */

static void easy_ramp2d(QSP_ARG_DECL  Data_Obj *dst_dp,double start,double dx,double dy)
{
	Vec_Obj_Args oa1;
	Scalar_Value sv1, sv2, sv3;

	cast_to_scalar_value(QSP_ARG  &sv1,OBJ_PREC_PTR(dst_dp),(double)start);
	cast_to_scalar_value(QSP_ARG  &sv2,OBJ_PREC_PTR(dst_dp),(double)dx);
	cast_to_scalar_value(QSP_ARG  &sv3,OBJ_PREC_PTR(dst_dp),(double)dy);

	clear_obj_args(&oa1);
	//SET_OA_SRC_OBJ(&oa1,0, dst_dp);			// why set this???
	SET_OA_DEST(&oa1, dst_dp);
	SET_OA_SVAL(&oa1,0, &sv1);
	SET_OA_SVAL(&oa1,1, &sv2);
	SET_OA_SVAL(&oa1,2, &sv3);

	set_obj_arg_flags(&oa1);

	platform_dispatch_by_code( QSP_ARG  FVRAMP2D, &oa1 );
}

static void assign_element(QSP_ARG_DECL Data_Obj *dp,dimension_t ri,dimension_t ci,Vec_Expr_Node *enp)
{
	double *dbl_p,d;

	//SET_OBJ_FLAG_BITS(dp, DT_ASSIGNED);
	note_assignment(dp);

//#ifdef CAUTIOUS
//	if( OBJ_PREC(dp) != PREC_DP ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  assign_element:  object %s is not double precision!?",OBJ_NAME(dp));
//		WARN(ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( OBJ_PREC(dp) == PREC_DP );

	dbl_p = (double *)OBJ_DATA_PTR(dp);
	/* assign_element uses matlab indexing */
	dbl_p += (ri-1) * OBJ_ROW_INC(dp);
	dbl_p += (ci-1) * OBJ_PXL_INC(dp);
	d = EVAL_FLT_EXP(enp);
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
			src_dp = DOBJ_OF(VN_STRING(enp));
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
			MISSING_CASE(enp,"assign_row");
			break;
	}
}

/* Like dp_convert(), but if destination is complex then do the right thing.
 */

static int c_convert(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *dp)
{
	//Vec_Obj_Args oa1, *oap=(&oa1);
	Data_Obj *tmp_dp;

	//clear_obj_args(oap);

	if( IS_COMPLEX(dst_dp) && ! IS_COMPLEX(dp) ){
		tmp_dp = C_SUBSCRIPT(dst_dp,0);
		//setvarg2(oap,tmp_dp,dp);
		dp_convert(QSP_ARG  tmp_dp,dp);
	} else {
		/* Can we put an error check on convert??? */
		//setvarg2(oap,dst_dp,dp);
		dp_convert(QSP_ARG  dst_dp,dp);
	}

#ifdef FOOBAR
	if( OBJ_MACH_PREC(dst_dp) == PREC_SP ){
		platform_dispatch_by_code(QSP_ARG FVCONV2SP, oap);
	} else if( OBJ_MACH_PREC(dst_dp) == PREC_DP ){
		platform_dispatch_by_code(QSP_ARG FVCONV2DP, oap);
	}
#ifdef CAUTIOUS
	  else {
		sprintf(ERROR_STRING,
"CAUTIOUS:  c_convert:  complex destination (%s) has bad machine precision (%s)!?",
			OBJ_NAME(dst_dp),NAME_FOR_PREC_CODE(OBJ_MACH_PREC(dst_dp)));
		WARN(ERROR_STRING);
	}
#endif // CAUTIOUS
#endif // FOOBAR

	if( IS_COMPLEX(dst_dp) && ! IS_COMPLEX(dp) ){
		tmp_dp = C_SUBSCRIPT(dst_dp,1);
		return( zero_dp(QSP_ARG  tmp_dp) );
	}
	return(0);
}

/* We may need to treat this differently for eval_obj_exp and eval_obj_assignment...
 * eval_obj_exp doesn't use dst_dp unless it has to, and dst_dp can be null...
 * Here we assume that the compilation process will have removed any
 * unnecessary typecasts, so we assume that dst_dp is needed, and if it is
 * null we will create it.
 */

static Data_Obj *eval_typecast(QSP_ARG_DECL Vec_Expr_Node *enp, Data_Obj *dst_dp)
{
	Data_Obj *dp, *tmp_dp;

/*
if( dst_dp != NO_OBJ ){
sprintf(ERROR_STRING,"eval_typecast:  dst_dp %s at 0x%lx",OBJ_NAME(dst_dp),(u_long)dst_dp);
advise(ERROR_STRING);
}
*/

//#ifdef CAUTIOUS
//	if( VN_SHAPE(VN_CHILD(enp,0)) == NO_SHAPE ){
//		NODE_ERROR(VN_CHILD(enp,0));
//		sprintf(ERROR_STRING,"CAUTIOUS:  eval TYPECAST:  %s has no shape!?",
//			node_desc(VN_CHILD(enp,0)));
//		ERROR1(ERROR_STRING);
//		IOS_RETURN_VAL(NULL)
//	}
	assert( VN_SHAPE(VN_CHILD(enp,0)) != NO_SHAPE );

//	if( dst_dp!= NO_OBJ && UNKNOWN_SHAPE(OBJ_SHAPE(dst_dp)) ){
//		NODE_ERROR(enp);
//		sprintf(ERROR_STRING,"CAUTIOUS:  eval_typecast %s:  destination Object %s has uknown shape!?",
//			node_desc(enp),OBJ_NAME(dst_dp));
//		ERROR1(ERROR_STRING);
//		IOS_RETURN_VAL(NULL)
//	}
//#endif /* CAUTIOUS */
	// Can we assert that this is not null???
	if( dst_dp != NO_OBJ ){
		assert( ! UNKNOWN_SHAPE(OBJ_SHAPE(dst_dp)) );
		assert( OBJ_PREC(dst_dp) == VN_PREC(enp) );
	}

	/* It is not an error for the typecast to match the LHS -
	 * in fact it should!  compile_node may insert a typecast
	 * node to effect type conversion.
	 */

//#ifdef CAUTIOUS
//	if( dst_dp != NO_OBJ && OBJ_PREC(dst_dp) != VN_PREC(enp) /* same as VN_INTVAL(enp) */ ){
//		NODE_ERROR(enp);
//		sprintf(ERROR_STRING,
//	"CAUTIOUS:  eval_typecast:  %s precision %s does not match target %s precision %s",
//			node_desc(enp),NAME_FOR_PREC_CODE(VN_PREC(enp)),OBJ_NAME(dst_dp),OBJ_PREC_NAME(dst_dp));
//		WARN(ERROR_STRING);
//		advise("ignoring typecast");
//		EVAL_OBJ_ASSIGNMENT(dst_dp,VN_CHILD(enp,0));
//		return(dst_dp);
//	}
//#endif /* CAUTIOUS */

	if( VN_INTVAL(enp) == SHP_PREC(VN_SHAPE(VN_CHILD(enp,0))) ){
		/* the object already has the cast precision */
		NODE_ERROR(enp);
		WARN("typecast redundant w/ rhs");
		EVAL_OBJ_ASSIGNMENT(dst_dp,VN_CHILD(enp,0));
		return(dst_dp);
	}

	/* If the child node is an object, we simply do a conversion into the
	 * destination.  If it's an operator, we have to make a temporary object
	 * to hold the result, and then convert.
	 */

	switch(VN_CODE(VN_CHILD(enp,0))){
		ALL_OBJREF_CASES
			/* dp=EVAL_OBJ_REF(VN_CHILD(enp,0)); */
			dp=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			if( dp != NO_OBJ ){
				if( dst_dp == NO_OBJ ){
					dst_dp=make_local_dobj(QSP_ARG  
						SHP_TYPE_DIMS(VN_SHAPE(VN_CHILD(enp,0))),
						VN_PREC_PTR(enp));
				}
/*
advise("eval_typecast calling convert");
LONGLIST(dst_dp);
LONGLIST(dp);
*/
				if( c_convert(QSP_ARG  dst_dp,dp) < 0 ){
					NODE_ERROR(enp);
					WARN("Error performing conversion");
				}
			} else return(NO_OBJ);
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
			goto handle_it;

		default:
			MISSING_CASE(VN_CHILD(enp,0),"eval_typecast");
			/* missing_case calls dump_tree?? */
			DUMP_TREE(enp);

handle_it:
			/* We have been requested to convert
			 * to a different precision
			 */

			tmp_dp=make_local_dobj(QSP_ARG  
					SHP_TYPE_DIMS(VN_SHAPE(VN_CHILD(enp,0))),
					SHP_PREC_PTR(VN_SHAPE(VN_CHILD(enp,0))) );

			EVAL_OBJ_ASSIGNMENT(tmp_dp,VN_CHILD(enp,0));

			if( dst_dp == NO_OBJ )
				dst_dp=make_local_dobj(QSP_ARG  
					SHP_TYPE_DIMS(VN_SHAPE(VN_CHILD(enp,0))),
					VN_PREC_PTR(enp));

			if( c_convert(QSP_ARG  dst_dp,tmp_dp) < 0 ){
				NODE_ERROR(enp);
				WARN("error performing conversion");
			}
			delvec(QSP_ARG  tmp_dp);
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

static int assign_subrt_args(QSP_ARG_DECL Vec_Expr_Node *arg_enp,Vec_Expr_Node *val_enp,Subrt *srp,Context_Pair *prev_cpp)
{
	int stat;
	Data_Obj *dp;
	Context_Pair *_curr_cpp;
	Function_Ptr *fpp;

	INIT_CPAIR_PTR(_curr_cpp);

	if( arg_enp==NO_VEXPR_NODE ) return(0);

	switch(VN_CODE(arg_enp)){
		case T_DECL_STAT:
			/* en_decl_prec is the type (float,short,etc) */
			stat=ASSIGN_SUBRT_ARGS(VN_CHILD(arg_enp,0),
						val_enp,srp,prev_cpp);
			return(stat);

		case T_DECL_STAT_LIST:
			/* descend the arg tree */
			/* VN_CODE(val_enp) should be T_ARGLIST */
			stat=ASSIGN_SUBRT_ARGS(VN_CHILD(arg_enp,0),
				VN_CHILD(val_enp,0),srp,prev_cpp);
			if( stat < 0 ) return(stat);

			stat=ASSIGN_SUBRT_ARGS(VN_CHILD(arg_enp,1),
				VN_CHILD(val_enp,1),srp,prev_cpp);
			return(stat);

		case T_FUNCPTR_DECL:		/* assign_subrt_args */
			/* we evaluate the argument */

			POP_SUBRT_CPAIR(_curr_cpp,SR_NAME(curr_srp));

			if( prev_cpp != NO_CONTEXT_PAIR ){
				PUSH_CPAIR(prev_cpp);
			}

			srp = eval_funcref(QSP_ARG  val_enp);

			if( prev_cpp != NO_CONTEXT_PAIR ){
				POP_CPAIR;
			}

			/* Now we switch contexts back to the called subrt */

			PUSH_CPAIR(_curr_cpp);

			/* the argument is a function ptr */
			fpp = eval_funcptr(QSP_ARG  arg_enp);

			if( srp == NO_SUBRT ) {
				WARN("assign_subrt_args:  error evaluating function ref");
				return(-1);
			}

			if( fpp == NO_FUNC_PTR ){
				WARN("assign_subrt_args:  missing function pointer");
				return(-1);
			}

			fpp->fp_srp = srp;
			return(0);

		case T_PTR_DECL:		/* assign_subrt_args */
			return( ASSIGN_PTR_ARG(arg_enp,val_enp,_curr_cpp,prev_cpp) );


		case T_SCAL_DECL:		/* assign_subrt_args */
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
				return(0);
			}

			dp = get_id_obj(QSP_ARG  VN_STRING(arg_enp),arg_enp);

			if( dp == NO_OBJ ){
sprintf(ERROR_STRING,"assign_subrt_args:  missing object %s",VN_STRING(arg_enp));
WARN(ERROR_STRING);
				return(-1);
			}

//#ifdef CAUTIOUS
//			if( UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
//				NODE_ERROR(arg_enp);
//				sprintf(ERROR_STRING,
//	"CAUTIOUS:  assign_subrt_args:  subrt %s, arg %s has unknown shape!?",
//		SR_NAME(srp),OBJ_NAME(dp));
//				WARN(ERROR_STRING);
//			}
//#endif /* CAUTIOUS */
			assert( ! UNKNOWN_SHAPE(OBJ_SHAPE(dp)) );

			/* Tricky point:  we need to pop the subroutine context
			 * here, in case val_enp uses names which are also
			 * some of the new subrt arguments...  if there
			 * are name overlaps, we want to be sure we use
			 * the outer ones for the assignment value!
			 */

			POP_SUBRT_CPAIR(_curr_cpp,SR_NAME(curr_srp));

			if( prev_cpp != NO_CONTEXT_PAIR ){

				PUSH_CPAIR(prev_cpp);

			}

			EVAL_OBJ_ASSIGNMENT(dp, val_enp);

			if( prev_cpp != NO_CONTEXT_PAIR ){
				Item_Context *icp;
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"assign_subrt_args T_IMG_DECL:  previous contexts %s, %s popped",CONTEXT_NAME(CP_ID_CTX(prev_cpp)),
CONTEXT_NAME(CP_OBJ_CTX(prev_cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
				/*icp=*/ POP_ID_CONTEXT;
				icp=POP_DOBJ_CONTEXT;
//#ifdef CAUTIOUS
//				if( icp != CP_OBJ_CTX(prev_cpp) ){
//					sprintf(ERROR_STRING,
//		"CAUTIOUS:  popped context %s does not match expected context %s!?",CONTEXT_NAME(icp),
//						CONTEXT_NAME(CP_OBJ_CTX(prev_cpp)));
//					WARN(ERROR_STRING);
//				}
//#endif /* CAUTIOUS */
				assert( icp == CP_OBJ_CTX(prev_cpp) );
			}

			/* restore it */
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"assign_subrt_args T_IMG_DECL:  pushing current context %s",prev_cpp==NULL?
	"(null previous context)":CONTEXT_NAME(CP_OBJ_CTX(prev_cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			PUSH_CPAIR(_curr_cpp);

			return(0);

		default:
			MISSING_CASE(arg_enp,"assign_subrt_args");
			break;

	}
	WARN("assign_subrt_args:  shouldn't reach this point");
	return(-1);
} /* end assign_subrt_args() */

Subrt *runnable_subrt(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Subrt *srp;

	switch(VN_CODE(enp)){
		case T_CALLFUNC:
			srp=VN_CALL_SUBRT(enp);
			SET_SR_ARG_VALS(srp, VN_CHILD(enp,0) );
			break;
		case T_INDIR_CALL:
			srp = eval_funcref(QSP_ARG  VN_CHILD(enp,0));
//#ifdef CAUTIOUS
//			if( srp==NO_SUBRT ){
//				NODE_ERROR(enp);
//				WARN("CAUTIOUS:  Missing function reference");
//				return(srp);
//			}
//#endif /* CAUTIOUS */
			assert( srp!=NO_SUBRT );

			SET_SR_ARG_VALS(srp, VN_CHILD(enp,1) );
			break;
		default:
			MISSING_CASE(enp,"runnable_subrt");
			return(NO_SUBRT);
	}

	SET_SR_CALL_VN(srp, enp); /* what is this used for??? */

	if( SR_BODY(srp) == NO_VEXPR_NODE ){
		NODE_ERROR(enp);
		sprintf(ERROR_STRING,"subroutine %s has not been defined!?",SR_NAME(srp));
		WARN(ERROR_STRING);
		return(NO_SUBRT);
	}
	return(srp);
}

/* exec_subrt is usually called on a T_CALLFUNC or T_INDIR_CALL node */

void exec_subrt(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj *dst_dp)
{
	Subrt *srp;

	srp = runnable_subrt(QSP_ARG  enp);

	if( srp != NO_SUBRT ){
		RUN_SUBRT(srp,enp,dst_dp);
	} else {
		sprintf(ERROR_STRING,"subroutine is not runnable!?");
		WARN(ERROR_STRING);
		DUMP_TREE(enp);
	}
}

Identifier *make_named_reference(QSP_ARG_DECL  const char *name)
{
	Identifier *idp;

	idp = ID_OF(name);
	if( idp != NO_IDENTIFIER ) return(idp);

//sprintf(ERROR_STRING,"make_named_reference:  creating id %s",name);
//advise(ERROR_STRING);
	idp = new_id(QSP_ARG  name);
	SET_ID_TYPE(idp, ID_REFERENCE);
	SET_ID_REF(idp, NEW_REFERENCE );
	SET_REF_OBJ(ID_REF(idp), NO_OBJ );
	SET_REF_ID(ID_REF(idp), idp );
	SET_REF_TYPE(ID_REF(idp), OBJ_REFERENCE );		/* override if string */
	SET_REF_DECL_VN(ID_REF(idp), NO_VEXPR_NODE );
	return(idp);
}

static void eval_display_stat(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp;
	const char *s;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_EXPR_LIST:
			EVAL_DISPLAY_STAT(VN_CHILD(enp,0));
			EVAL_DISPLAY_STAT(VN_CHILD(enp,1));
			break;
		case T_STR_PTR:
			s = EVAL_STRING(enp);
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
			dp = EVAL_OBJ_REF(enp);
			if( dp==NO_OBJ ){
				WARN("missing info object");
				// An informative message should have
				// been printed before we get here...
				break;
			} else {
				list_dobj(QSP_ARG  dp);
				/* set_output_file */
				/* pntvec(dp,stdout); */
				pntvec(QSP_ARG  dp,
					tell_msgfile(SINGLE_QSP_ARG) );
			}
			break;
		default:
			MISSING_CASE(enp,"eval_display_stat");
			break;
	}
}

static Vec_Expr_Node *find_case(QSP_ARG_DECL Vec_Expr_Node *enp,long lval)
{
	Vec_Expr_Node *ret_enp;
	long cval;

	switch(VN_CODE(enp)){
		case T_CASE_STAT:	/* case_list stat_list pair */
			if( FIND_CASE(VN_CHILD(enp,0),lval) != NO_VEXPR_NODE )
				return(enp);
			else return(NO_VEXPR_NODE);

		case T_CASE_LIST:
			ret_enp=FIND_CASE(VN_CHILD(enp,0),lval);
			if( ret_enp == NO_VEXPR_NODE )
				ret_enp=FIND_CASE(VN_CHILD(enp,1),lval);
			return(ret_enp);

		case T_CASE:
			cval = EVAL_INT_EXP(VN_CHILD(enp,0));
			if( cval == lval ){
				return(VN_CHILD(enp,0));
			} else return(NO_VEXPR_NODE);

		case T_DEFAULT:
			return(enp);

		case T_SWITCH_LIST:	/* list of case_stat's */
			ret_enp=FIND_CASE(VN_CHILD(enp,0),lval);
			if( ret_enp == NO_VEXPR_NODE )
				ret_enp=FIND_CASE(VN_CHILD(enp,1),lval);
			return(ret_enp);

		default:
			MISSING_CASE(enp,"find_case");
			break;
	}
	return(NO_VEXPR_NODE);
}

/* Find the first case of a switch statement.
 * used for goto scanning.
 */

static Vec_Expr_Node *first_case(Vec_Expr_Node *enp)
{
	Vec_Expr_Node *case_enp;

//#ifdef CAUTIOUS
//	if( VN_CODE(enp) != T_SWITCH ){
//		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  first_case %s:  expected switch node",node_desc(enp));
//		NERROR1(DEFAULT_ERROR_STRING);
//		IOS_RETURN_VAL(NULL)
//	}
//#endif /* CAUTIOUS */
	assert( VN_CODE(enp) == T_SWITCH );

	case_enp = VN_CHILD(enp,1);

	while(VN_CODE(case_enp) == T_SWITCH_LIST )
		case_enp = VN_CHILD(case_enp,0);
//#ifdef CAUTIOUS
//	if( VN_CODE(case_enp) != T_CASE_STAT ){
//		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  eval_work_tree:  expected to find case_stat while searching for goto label, found %s",node_desc(case_enp));
//		NERROR1(DEFAULT_ERROR_STRING);
//		IOS_RETURN_VAL(NULL)
//	}
//#endif /* CAUTIOUS */
	assert( VN_CODE(case_enp) == T_CASE_STAT );

	return(case_enp);
}

/* returns a child LABEL node whose name matches global goto_label,
 * or NO_VEXPR_NODE if not found...
 */

static Vec_Expr_Node *goto_child(Vec_Expr_Node *enp)
{
	int i;

	for(i=0;i<MAX_CHILDREN(enp);i++){
		if( VN_CHILD(enp,i) != NO_VEXPR_NODE ){
			if( VN_CODE(VN_CHILD(enp,i)) == T_LABEL && !strcmp(VN_STRING(VN_CHILD(enp,i)),goto_label) ){
				return(VN_CHILD(enp,i));
			}
			else if( goto_child(VN_CHILD(enp,i)) != NO_VEXPR_NODE )
				return(VN_CHILD(enp,i));
		}
	}
	return(NO_VEXPR_NODE);
}

static Vec_Expr_Node *next_case(Vec_Expr_Node *enp)
{
	if( VN_CODE(VN_PARENT(enp)) == T_SWITCH ){
		return(NO_VEXPR_NODE);
	}

//#ifdef CAUTIOUS
//	if( VN_CODE(VN_PARENT(enp) ) != T_SWITCH_LIST ){
//		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  next_case %s:  expected switch_list parent, found %s!?",
//			node_desc(enp), node_desc(VN_PARENT(enp)));
//		NERROR1(DEFAULT_ERROR_STRING);
//		IOS_RETURN_VAL(NULL)
//	}
//#endif /* CAUTIOUS */
	assert( VN_CODE(VN_PARENT(enp) ) == T_SWITCH_LIST );

keep_looking:
	if( VN_CODE(VN_PARENT(enp)) == T_SWITCH_LIST ){
		if( enp == VN_CHILD(VN_PARENT(enp),0) ){
			/* descend the right branch */
			enp=VN_CHILD(VN_PARENT(enp),1);
			while( VN_CODE(enp) == T_SWITCH_LIST )
				enp=VN_CHILD(enp,0);
//#ifdef CAUTIOUS
//			if( VN_CODE(enp) != T_CASE_STAT ){
//				sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  next_case %s: leaf node should have been case_stat",
//					node_desc(enp));
//				NERROR1(DEFAULT_ERROR_STRING);
//				IOS_RETURN_VAL(NULL)
//			}
//#endif /* CAUTIOUS */
			assert( VN_CODE(enp) == T_CASE_STAT );

			return(enp);
		} else {
			/* our case is the right hand child... */
			enp = VN_PARENT(enp);
			goto keep_looking;
		}
	}
	return(NO_VEXPR_NODE);
}

/* Traverse a string list tree, setting the query args starting
 * at index.  Returns the number of leaves.
 *
 * When this was first written, the query args were a fixed-sized
 * table q_arg;  But now they are a dynamically allocated array
 * of variable size, renamed q_args.  Normally these are allocated
 * when a macro is invoked and pushed onto the query
 * stack.  Here we are pushing a script function on the query stack.
 */

#define STORE_QUERY_ARG( s )						\
{									\
	if( index < max_args ){						\
		SET_QRY_ARG_AT_IDX(qp,index,s);			\
	} else {								\
		sprintf(ERROR_STRING,"set_script_args:  can't assign arg %d (max %d)",\
			index+1,max_args);				\
		WARN(ERROR_STRING);					\
	}								\
}

static int set_script_args(QSP_ARG_DECL Vec_Expr_Node *enp,int index,Query *qp,int max_args)
{
	int n1,n2;
	double dval;
	const char *s;
	Data_Obj *dp;
	char buf[64];

	if( enp==NO_VEXPR_NODE ) return(0);

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_DEREFERENCE:				/* set_script_args */
			dp = EVAL_OBJ_REF(enp);
			if( dp==NO_OBJ ){
				NODE_ERROR(enp);
				WARN("missing script arg object");
				return(0);
			} else {
				STORE_QUERY_ARG(OBJ_NAME(dp));
				return(1);
			}

		case T_STR_PTR:
			s=EVAL_STRING(enp);
			STORE_QUERY_ARG(s)
			return(1);

		case T_POINTER:
			/* do we dereference the pointer??? */
			NODE_ERROR(enp);
			sprintf(ERROR_STRING,
				"set_script_args:  not sure whether or not to dereference ptr %s",
				VN_STRING(enp));
			advise(ERROR_STRING);
			/* fall-thru */

		case T_STATIC_OBJ:		/* set_script_args */
		case T_DYN_OBJ:			/* set_script_args */
			/* maybe we could check the node shape instead of looking up the object? */
			dp=EVAL_OBJ_REF(enp);
			if( IS_SCALAR(dp) ){
				format_scalar_obj(QSP_ARG  buf,dp,OBJ_DATA_PTR(dp));
				STORE_QUERY_ARG( savestr(buf) )
				return(1);
			}
			/* else fall-thru */
		case T_STRING:
			/* add this string as one of the args */
			STORE_QUERY_ARG( savestr(VN_STRING(enp)) )
			return(1);

		case T_PRINT_LIST:
		case T_STRING_LIST:
		case T_MIXED_LIST:
			n1=SET_SCRIPT_ARGS(VN_CHILD(enp,0),index,qp,max_args);
			n2=SET_SCRIPT_ARGS(VN_CHILD(enp,1),index+n1,qp,max_args);
			return(n1+n2);

		/* BUG there are more cases that need to go here
		 * in order to handle generic expressions
		 */

		case T_LIT_INT: case T_LIT_DBL:			/* set_script_args */
		case T_PLUS: case T_MINUS: case T_TIMES: case T_DIVIDE:
			dval=EVAL_FLT_EXP(enp);
			sprintf(msg_str,"%g",dval);
			STORE_QUERY_ARG( savestr(msg_str) )
			return(1);

		default:
//#ifdef CAUTIOUS
//			MISSING_CASE(enp,"set_script_args");
//#endif /* CAUTIOUS */
			assert( AERROR("missing case in set_script_args") );
			break;
	}
	return(0);
} /* end set_script_args */

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

	icp = POP_DOBJ_CONTEXT;
//#ifdef CAUTIOUS
//	if( icp == NO_ITEM_CONTEXT ){
//		ERROR1("CAUTIOUS:  set_script_context:  no current dobj context!?");
//		IOS_RETURN
//	}
//#endif /* CAUTIOUS */
	assert( icp != NO_ITEM_CONTEXT );

#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"set_script_context:  current context %s popped",CONTEXT_NAME(icp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	for(i=0;i<n_hidden_contexts;i++){
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"set_script_context:  pushing hidden context %s",CONTEXT_NAME(hidden_context[i]));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		PUSH_DOBJ_CONTEXT(hidden_context[i]);
	}
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"set_script_context:  pushing current context %s",CONTEXT_NAME(icp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	PUSH_DOBJ_CONTEXT(icp);

	set_global_ctx(SINGLE_QSP_ARG);	/* we do this so any new items created will be global */
}

static void unset_script_context(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp,*top_icp;
	int i;

	unset_global_ctx(SINGLE_QSP_ARG);

	top_icp = POP_DOBJ_CONTEXT;

#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"unset_script_context:  top context %s popped",CONTEXT_NAME(top_icp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
//#ifdef CAUTIOUS
//	if( top_icp == NO_ITEM_CONTEXT ){
//		ERROR1("CAUTIOUS:  unset_script_context:  no current dobj context!?");
//		IOS_RETURN
//	}
//#endif /* CAUTIOUS */
	assert( top_icp != NO_ITEM_CONTEXT );

	for(i=0;i<n_hidden_contexts;i++){
		icp = POP_DOBJ_CONTEXT;
//#ifdef CAUTIOUS
//		if( icp != hidden_context[n_hidden_contexts-(1+i)] ){
//			sprintf(ERROR_STRING,
//"CAUTIOUS:  unset_script_context:  popped context %d %s does not match hidden stack context %s!?",
//				i+1,CONTEXT_NAME(icp),CONTEXT_NAME(hidden_context[n_hidden_contexts-(i+1)]));
//			WARN(ERROR_STRING);
//		}
//#endif /* CAUTIOUS */
		assert( icp == hidden_context[n_hidden_contexts-(1+i)] );

#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"unset_script_context:  hidden context %s popped",CONTEXT_NAME(icp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	}

	PUSH_DOBJ_CONTEXT(top_icp);
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"unset_script_context:  top context %s pushed",CONTEXT_NAME(top_icp));
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

static const char *eval_mixed_list(QSP_ARG_DECL Vec_Expr_Node *enp)
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

			dp=EVAL_OBJ_REF(VN_CHILD(enp,0));
//#ifdef CAUTIOUS
//			if( dp == NO_OBJ ){
//				NODE_ERROR(enp);
//				WARN("CAUTIOUS:  bad namefunc node");
//				return "bad_name";
//			}
//#endif /* CAUTIOUS */
			assert( dp != NO_OBJ );
			return(OBJ_NAME(dp));

		case T_STRING:
			return( VN_STRING(enp) );
		case T_STR_PTR:
			if( dumping ) return(STRING_FORMAT);

			idp = EVAL_PTR_REF(enp,EXPECT_PTR_SET);
			if( idp==NO_IDENTIFIER ) return("");
//#ifdef CAUTIOUS
//			if( ! IS_STRING_ID(idp) ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,"CAUTIOUS:  identifier %s is not a string ptr!?",ID_NAME(idp));
//				WARN(ERROR_STRING);
//				break;
//			}
//#endif /* CAUTIOUS */
			assert( IS_STRING_ID(idp) );

			if( REF_SBUF(ID_REF(idp))->sb_buf == NULL ){
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,"string pointer %s not set",ID_NAME(idp));
				advise(ERROR_STRING);
				break;
			}
			return(REF_SBUF(ID_REF(idp))->sb_buf);

		case T_SET_STR:
			EVAL_WORK_TREE(enp,NO_OBJ);	/* do the assignment! */
			return( EVAL_MIXED_LIST(VN_CHILD(enp,0)) );

		case T_STRING_LIST:
		case T_PRINT_LIST:
			s1=EVAL_MIXED_LIST(VN_CHILD(enp,0));
			s2=EVAL_MIXED_LIST(VN_CHILD(enp,1));
			n=(int)(strlen(s1)+strlen(s2)+1);
			s=(char *)getbuf(n);
			strcpy(s,s1);
			strcat(s,s2);
			return(s);

		ALL_OBJREF_CASES			/* eval_mixed_list */
			if( dumping ) return(OBJECT_FORMAT);

			/* BUG need all expr nodes here */

			dp = EVAL_OBJ_REF(enp);
			if( dp==NO_OBJ ) return("(null)");
			if( IS_SCALAR(dp) )
				format_scalar_obj(QSP_ARG  buf,dp,OBJ_DATA_PTR(dp));
			else {
				/*
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,
					"eval_mixed_list:  object %s is not a scalar!?",OBJ_NAME(dp));
				WARN(ERROR_STRING);
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
			MISSING_CASE(enp,"eval_mixed_list");
			break;
	}
	return("");
} /* eval_mixed_list */

static void eval_print_stat(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp;
	Identifier *idp;
	long n;
	double d;
	const char *s;


	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_CALLFUNC:			/* eval_print_stat */
			if( ! SCALAR_SHAPE(VN_SHAPE(enp)) ){
				prt_msg("");
				NODE_ERROR(enp);
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
					ERROR1("eval_print_stat:  missing CALLFUNC precision");
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
			d = EVAL_FLT_EXP(enp);
			sprintf(msg_str,"%g",d);
			prt_msg_frag(msg_str);
			break;
			break;


		case T_NAME_FUNC:
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( dp == NO_OBJ ){
				NODE_ERROR(enp);
				advise("Missing object");
			} else
				prt_msg_frag(OBJ_NAME(dp));
			break;

		case T_STRING_LIST:
		case T_MIXED_LIST:
		case T_PRINT_LIST:
			EVAL_PRINT_STAT(VN_CHILD(enp,0));
			prt_msg_frag(" ");
			EVAL_PRINT_STAT(VN_CHILD(enp,1));
			break;
		case T_POINTER:
			idp = EVAL_PTR_REF(enp,EXPECT_PTR_SET);
//#ifdef CAUTIOUS
//			if( ! IS_POINTER(idp) ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,"CAUTIOUS:  identifier %s is not a pointer",ID_NAME(idp));
//				advise(ERROR_STRING);
//				break;
//			}
			assert( IS_POINTER(idp) );

//			if( ! POINTER_IS_SET(idp) ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,"CAUTIOUS:  pointer %s is not set",ID_NAME(idp));
//				advise(ERROR_STRING);
//				break;
//			}
			assert( POINTER_IS_SET(idp) );

			/* If it's a pointer, should be id_ptrp, not id_refp!? */
			/*
			if( ID_REF(idp) == NO_REFERENCE ){
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,"CAUTIOUS:  id %s, id_refp is null",ID_NAME(idp));
				advise(ERROR_STRING);
				break;
			}
			*/

//			if( ID_PTR(idp) == NO_POINTER ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,"CAUTIOUS:  id %s, id_ptrp is null",ID_NAME(idp));
//				advise(ERROR_STRING);
//				break;
//			}
			assert( ID_PTR(idp) != NO_POINTER );

//			if( PTR_REF(ID_PTR(idp)) == NO_REFERENCE ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,"CAUTIOUS:  id %s, PTR_REF(id_ptrp) is null",ID_NAME(idp));
//				advise(ERROR_STRING);
//				break;
//			}
			assert( PTR_REF(ID_PTR(idp)) != NO_REFERENCE );

//#endif /* CAUTIOUS */
			if( IS_OBJECT_REF(PTR_REF(ID_PTR(idp))) ){
//#ifdef CAUTIOUS
//				if( REF_OBJ(PTR_REF(ID_PTR(idp))) == NO_OBJ ){
//					NODE_ERROR(enp);
//					sprintf(ERROR_STRING,"CAUTIOUS:  id %s, ref_dp is null",ID_NAME(idp));
//					advise(ERROR_STRING);
//					break;
//				}
//#endif /* CAUTIOUS */
				assert( REF_OBJ(PTR_REF(ID_PTR(idp))) != NO_OBJ );

				/* what should we print here? */
				/* If the pointer points to a string, then print the string... */
				dp=REF_OBJ(PTR_REF(ID_PTR(idp))) ;
				if( OBJ_PREC(dp) == PREC_CHAR || OBJ_PREC(dp) == PREC_STR )
					prt_msg_frag((char *)OBJ_DATA_PTR(dp));
				else
					prt_msg_frag(ID_NAME(idp));
			} else if( IS_STRING_REF(PTR_REF(ID_PTR(idp))) ){
				prt_msg_frag(REF_SBUF(PTR_REF(ID_PTR(idp)))->sb_buf);
			}
//#ifdef CAUTIOUS
			  else {
//			  	ERROR1("CAUTIOUS:  bad reference type");
//				IOS_RETURN
				assert( AERROR("bad reference type") );
			}
//#endif /* CAUTIOUS */
			break;

		ALL_OBJREF_CASES
		case T_PREDEC:
		case T_PREINC:			/* eval_print_stat */
		case T_POSTDEC:
		case T_POSTINC:			/* eval_print_stat */
			dp = EVAL_OBJ_REF(enp);
			if( dp==NO_OBJ ) return;

			if( VN_CODE(enp) == T_PREINC ) inc_obj(dp);
			else if( VN_CODE(enp) == T_PREDEC ) dec_obj(dp);

			if( IS_SCALAR(dp) ){
				format_scalar_obj(QSP_ARG  msg_str,dp,OBJ_DATA_PTR(dp));
				prt_msg_frag(msg_str);
			} else {
				/*
				NODE_ERROR(enp);
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
			s=EVAL_MIXED_LIST(enp);
			prt_msg_frag(s);
			break;

		case T_STRING:
			prt_msg_frag(VN_STRING(enp));
			break;
		case T_LIT_INT:
		case T_SIZE_FN:		// eval_print_stat
			/* BUG need all expr nodes here */
print_integer:
			n=EVAL_INT_EXP(enp);
			sprintf(msg_str,"%ld",n);
			prt_msg_frag(msg_str);
			break;
		default:
			MISSING_CASE(enp,"eval_print_stat");
			break;
	}
} /* eval_print_stat */

/* eval_ref_tree - what is this used for??? */

static void eval_ref_tree(QSP_ARG_DECL Vec_Expr_Node *enp,Identifier *dst_idp)
{
	Identifier *idp;

	switch(VN_CODE(enp)){
		case T_STAT_LIST:
			EVAL_REF_TREE(VN_CHILD(enp,0),dst_idp);
			EVAL_REF_TREE(VN_CHILD(enp,1),dst_idp);
			break;
		case T_ASSIGN:				/* eval_ref_tree */
			if( EVAL_WORK_TREE(enp,NO_OBJ) == 0 )
				WARN("CAUTIOUS:  eval_ref_tree:  eval_work_tree returned 0!?");
			break;
		case T_RETURN:	/* return a pointer */
			idp = EVAL_PTR_REF(VN_CHILD(enp,0),1);
//#ifdef CAUTIOUS
//			if( idp == NO_IDENTIFIER ){
//				NODE_ERROR(enp);
//				WARN("CAUTIOUS:  missing reference");
//				break;
//			}
			assert( idp != NO_IDENTIFIER );

//			if( ! IS_REFERENCE(idp) ){
//				sprintf(ERROR_STRING,"CAUTIOUS:  eval_ref_tree:  return val is not a reference");
//				ERROR1(ERROR_STRING);
//				IOS_RETURN
//			}
//#endif /* CAUTIOUS */
			assert( IS_REFERENCE(idp) );

			/* now copy over the identifier data */
			SET_PTR_REF(ID_PTR(dst_idp), ID_REF(idp));
			/* BUG? keep flags? */
			/*
			SET_PTR_FLAGS(ID_PTR(dst_idp), PTR_FLAGS(ID_PTR(idp)));
			*/
			break;
		default:
			MISSING_CASE(enp,"eval_ref_tree");
			break;
	}
}

/* wrapup_call
 *
 * forget any resolved shapes that are local to this subrt
 * Also pop the subrt context and restore the previous one, if any...
 */

static void wrapup_call(QSP_ARG_DECL  Run_Info *rip)
{
	/* We need to forget both the uk shape arguments
	 * and uk shape automatic variables.
	 */
	forget_resolved_shapes(QSP_ARG  rip->ri_srp);
	wrapup_context(QSP_ARG  rip);
}

static void run_reffunc(QSP_ARG_DECL Subrt *srp, Vec_Expr_Node *enp, Identifier *dst_idp)
{
	Run_Info *rip;

	executing=1;
	/* Run-time resolution of unknown shapes */

/*
sprintf(ERROR_STRING,"run_reffunc %s:  calling setup_call",SR_NAME(srp));
advise(ERROR_STRING);
*/
	rip = SETUP_CALL(srp,NO_OBJ);
	if( rip == NO_RUN_INFO ){
sprintf(ERROR_STRING,"run_reffunc %s:  no return info!?",SR_NAME(srp));
WARN(ERROR_STRING);
		return;
	}

	if( rip->ri_arg_stat >= 0 ){
		EVAL_DECL_TREE(SR_BODY(srp));
		EVAL_REF_TREE(SR_BODY(srp),dst_idp);
	}

	wrapup_call(QSP_ARG  rip);
}



/* a function that returns a pointer */

static Identifier * exec_reffunc(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Identifier *idp;
	char name[LLEN];
	Subrt *srp;

	srp = runnable_subrt(QSP_ARG  enp);
	if( srp==NO_SUBRT ) return(NO_IDENTIFIER);

	sprintf(name,"ref.%s",SR_NAME(srp));

	idp = make_named_reference(QSP_ARG  name);
	/* BUG set ptr_type?? */

//#ifdef CAUTIOUS
//	if( idp == NO_IDENTIFIER ) {
//		ERROR1("CAUTIOUS:  unable to make named identifier");
//		IOS_RETURN_VAL(NULL)
//	}
//#endif /* CAUTIOUS */
	assert( idp != NO_IDENTIFIER ) ;

	/* need to check stuff */


	if( srp != NO_SUBRT )
		RUN_REFFUNC(srp,enp,idp);

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
//#ifdef CAUTIOUS
//	if( n_hidden_contexts <= 0 ){
//		NERROR1("CAUTIOUS:  no hidden context to pop");
//		IOS_RETURN
//	}
//#endif /* CAUTIOUS */
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
	if( curr_srp == NO_SUBRT ){
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
advise("pop_previous:  no current subroutine, nothing to pop");
}
#endif /* QUIP_DEBUG */
		cpp = NO_CONTEXT_PAIR;
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
		POP_SUBRT_CPAIR(cpp,SR_NAME(curr_srp));
		/* we remember this context so we can use it if we call a script func */
		push_hidden_context(cpp);
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"pop_previous:  previous contexts %s, %s popped",
CONTEXT_NAME(CP_ID_CTX(cpp)),
CONTEXT_NAME(CP_OBJ_CTX(cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	}
	return(cpp);
}

/* We call restore_previous when we return from a subroutine call to go back
 * the the original context.
 */

void restore_previous(QSP_ARG_DECL  Context_Pair *cpp)
{
	pop_hidden_context();
	PUSH_CPAIR(cpp);
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"restore_previous:  previous contexts %s, %s pushed",
CONTEXT_NAME(CP_ID_CTX(cpp)),
CONTEXT_NAME(CP_OBJ_CTX(cpp)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	givbuf(cpp);
}

static Run_Info *new_rip()
{
	Run_Info *rip;

	rip=(Run_Info *)getbuf( sizeof(Run_Info) );
	rip->ri_prev_cpp = NO_CONTEXT_PAIR;
	rip->ri_arg_stat = 0;
	rip->ri_srp = NO_SUBRT;
	rip->ri_old_srp = NO_SUBRT;
	return(rip);
}

/* setup_call
 *
 * does the following things:
 *	calls early_calltime_resolve
 *	calls check_args_shapes, exits if there's a mismatch
 *	switches subrt context to the new subrt
 *	evaluates the arg decls
 *	assigns the arg values
 *	returns a run_info struct
 */

Run_Info * setup_call(QSP_ARG_DECL Subrt *srp,Data_Obj *dst_dp)
{
	Run_Info *rip;

	/*
	 * We call calltime resolve to resolve arg shapes and return shapes if we can.
	 * What is the expected context for early_calltime_resolve???
	 */
/*
if( dst_dp != NO_OBJ ){
sprintf(ERROR_STRING,"setup_call %s:  calling early_calltime_resolve, dst_dp = %s",SR_NAME(srp),OBJ_NAME(dst_dp));
advise(ERROR_STRING);
describe_shape(OBJ_SHAPE(dst_dp));
} else {
sprintf(ERROR_STRING,"setup_call %s:  calling early_calltime_resolve, dst_dp = NULL",SR_NAME(srp));
advise(ERROR_STRING);
}
*/
/* advise("setup_call calling early_calltime_resolve"); */
	EARLY_CALLTIME_RESOLVE(srp,dst_dp);
/* advise("setup_call back from early_calltime_resolve"); */

/*
advise("setup_call:  after early_calltime_resolve:");
DUMP_TREE(SR_BODY(srp));
*/
	/* BUG We'd like to pop the context of any calling subrts here, but it is tricky:
	 * We need to have the old context so we can find the arg values...  but we want
	 * to pop the context when we evaluate the arg decls to avoid warnings
	 * about names shadowing other names
	 * (which we only want for objects in the global context).
	 */

	rip = new_rip();
	rip->ri_srp = srp;

/*
sprintf(ERROR_STRING,"setup_call %s:  calling pop_previous #1 (context)",SR_NAME(srp));
advise(ERROR_STRING);
*/
/* advise("setup_call calling check_arg_shapes"); */
	if( CHECK_ARG_SHAPES(SR_ARG_DECLS(srp),SR_ARG_VALS(srp),srp) < 0 )
		goto call_err;

	/* declare the arg variables */

	/* First, pop the context of the previous subroutine and push the new one */
	rip->ri_prev_cpp = POP_PREVIOUS();	/* what does pop_previous() do??? */
	set_subrt_ctx(QSP_ARG  SR_NAME(srp));

	EVAL_DECL_TREE(SR_ARG_DECLS(srp));

	rip->ri_old_srp = curr_srp;
	curr_srp = srp;

/* advise("setup_call calling assign_subrt_args"); */
	rip->ri_arg_stat = ASSIGN_SUBRT_ARGS(SR_ARG_DECLS(srp),SR_ARG_VALS(srp),srp,rip->ri_prev_cpp);

	return(rip);

call_err:

	/* now we're back , restore the context of the caller , if any */
	if( rip->ri_prev_cpp != NO_CONTEXT_PAIR ){
/*
sprintf(ERROR_STRING,"setup_call %s:  restoring previous context",SR_NAME(srp));
advise(ERROR_STRING);
*/
		RESTORE_PREVIOUS(rip->ri_prev_cpp);
	}
	return(NO_RUN_INFO);
}

/* wrapup_context
 *
 * called after subroutine execution to restore the context of the caller.
 */

void wrapup_context(QSP_ARG_DECL  Run_Info *rip)
{

	curr_srp = rip->ri_old_srp;

/*
sprintf(ERROR_STRING,"wrapup_context %s:  calling delete_subrt_ctx",SR_NAME(rip->ri_srp));
advise(ERROR_STRING);
*/
	/* get rid of the context, restore the context of the caller , if any */

	delete_subrt_ctx(QSP_ARG  SR_NAME(rip->ri_srp));
	if( rip->ri_prev_cpp != NO_CONTEXT_PAIR ){
/*
sprintf(ERROR_STRING,"wrapup_context %s:  restoring previous context",SR_NAME(rip->ri_srp));
advise(ERROR_STRING);
*/
		RESTORE_PREVIOUS(rip->ri_prev_cpp);
	}
}






void run_subrt(QSP_ARG_DECL Subrt *srp, Vec_Expr_Node *enp, Data_Obj *dst_dp)
{
	Run_Info *rip;

	executing=1;

	rip = SETUP_CALL(srp,dst_dp);
	if( rip == NO_RUN_INFO ){
		return;
	}

	if( rip->ri_arg_stat >= 0 ){
		EVAL_DECL_TREE(SR_BODY(srp));
		/* eval_work_tree returns 0 if a return statement was executed,
		 * but not if there is an implied return.
		 *
		 * Uh, what is an "implied" return???
		 */
		EVAL_WORK_TREE(SR_BODY(srp),dst_dp);
	} else {
sprintf(ERROR_STRING,"run_subrt %s:  arg_stat = %d",SR_NAME(srp),rip->ri_arg_stat);
WARN(ERROR_STRING);
	}

	wrapup_call(QSP_ARG  rip);
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

static void setup_unknown_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,Dimension_Set *dsp)
{
	int i;
	if( VN_SHAPE(enp) == NO_SHAPE ){
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

//#ifdef CAUTIOUS
//	if( VN_PARENT(enp) == NO_VEXPR_NODE ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  setup_unknown_shape:  %s has no parent",
//			node_desc(enp));
//		WARN(ERROR_STRING);
//	}
//	/* We don't link this node to anything, because it's a declaration node */
//#endif /* ! CAUTIOUS */
	assert( VN_PARENT(enp) != NO_VEXPR_NODE );
}

static Data_Obj * finish_obj_decl(QSP_ARG_DECL  Vec_Expr_Node *enp,Dimension_Set *dsp,Precision *prec_p, int decl_flags)
{
	Data_Obj *dp;

	eval_enp = enp;

	/* at one time we handled special (complex) precision here... */

	dp=make_dobj(QSP_ARG  VN_STRING(enp),dsp,prec_p);

	if( dp==NO_OBJ ){
		NODE_ERROR(enp);
		sprintf(ERROR_STRING,
			"Error processing declaration for object %s",
			VN_STRING(enp));
		WARN(ERROR_STRING);
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
		icp = current_dobj_context(SINGLE_QSP_ARG);
		if( !strcmp(CTX_NAME(icp),"Data_Obj.default") ){
//fprintf(stderr,"dobj context is %s, forcing static\n",CTX_NAME(icp));
			decl_flags |= DECL_IS_STATIC;
		}
	}

// The problem is not with recycling nodes...  it is with local vars in subroutines.
// The declaration statements are evaluated each time we call the subroutine.
// So we need to clear the field when the object is deleted!
// This could create an unwanted dependency between the dobj module and this one!?

if( VN_DECL_OBJ(enp) != NO_OBJ ){
sprintf(ERROR_STRING,"%s decl obj (%s) is not null!?",
node_desc(enp),OBJ_NAME(VN_DECL_OBJ(enp)));
WARN(ERROR_STRING);
}

	assert( VN_DECL_OBJ(enp) == NO_OBJ );

	SET_VN_DECL_OBJ(enp,dp);

	copy_node_shape(enp,OBJ_SHAPE(dp));

/* BUG - now the const-ness is passed in the rpecision struct... */
	if( decl_flags & DECL_IS_CONST ) SET_OBJ_FLAG_BITS(dp, DT_RDONLY);
	if( decl_flags & DECL_IS_STATIC ) SET_OBJ_FLAG_BITS(dp, DT_STATIC);

	return(dp);
}


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

static void eval_decl_stat(QSP_ARG_DECL Precision * prec_p,Vec_Expr_Node *enp, int decl_flags)
{
	int i;
	Dimension_Set ds1, *dsp=(&ds1);
	int type/* =ID_OBJECT */;
	Identifier *idp;

	if( PREC_CODE(prec_p) == PREC_STR ){
		type = ID_STRING;
	} else {
		type = ID_REFERENCE;
	}

	eval_enp = enp;

	for(i=0;i<N_DIMENSIONS;i++)
		SET_DIMENSION(dsp,i,1 );

/*
sprintf(ERROR_STRING,"eval_decl_stat %s:",node_desc(enp));
advise(ERROR_STRING);
if( VN_SHAPE(enp) != NO_SHAPE ) describe_shape( VN_SHAPE(enp));
else prt_msg("\t(no shape)");
DUMP_TREE(enp);
*/

	switch(VN_CODE(enp)){
		case T_PROTO:
			{
			Subrt *srp;
			srp=subrt_of(QSP_ARG  VN_STRING(enp));
			if( srp != NO_SUBRT ){
				/* subroutine already declared.
				 * We should check to make sure that the arg decls match BUG
				 * this gets done elsewhere, but here we make sure the return
				 * type is the same.
				 */
				/* This subroutine has already been declared...
				 * make sure the type matches
				 */
				if( PREC_CODE(prec_p) != SR_PREC_CODE(srp) )
					prototype_mismatch(QSP_ARG  SR_ARG_DECLS(srp),enp);
				break;
			}
			srp = remember_subrt(QSP_ARG  prec_p,VN_STRING(enp),VN_CHILD(enp,0),NO_VEXPR_NODE);
			SET_SR_N_ARGS(srp, decl_count(QSP_ARG  SR_ARG_DECLS(srp)) );	/* set # args */
			SET_SR_FLAG_BITS(srp, SR_PROTOTYPE);
			return;
			}

		case T_BADNAME:
			return;
		case T_DECL_ITEM_LIST:
			EVAL_DECL_STAT(prec_p,VN_CHILD(enp,0),decl_flags);
			if( VN_CHILD(enp,1)!=NO_VEXPR_NODE )
				EVAL_DECL_STAT(prec_p,VN_CHILD(enp,1),decl_flags);
			return;
		case T_DECL_INIT:		/* eval_decl_stat */
			{
			Scalar_Value sval;
			Data_Obj *dp;
			double dval;

			/* CURDLED? */
			if( IS_CURDLED(enp) ) return;

			EVAL_DECL_STAT(prec_p,VN_CHILD(enp,0),decl_flags);
			/* the next node is an expression */
			dp = get_id_obj(QSP_ARG  VN_STRING(VN_CHILD(enp,0)),enp);
// BUG this is not CAUTIOUS because this can happen if you try to create
// an existing object that has not been exported.
// But then the message should not contain the string CAUTIOUS, and this should not
// be un-indented!?
if( dp==NO_OBJ ) ERROR1("CAUTIOUS:  eval_decl_stat:  Null object to initialize!?");

			/* What if the rhs is unknown size - then we have to resolve now! */
			if( UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
				/* Can we assume the rhs has a shape? */
				if( UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) ){
					NODE_ERROR(enp);
					WARN("LHS and RHS are both unknown shape!?");
				} else {
advise("attempting resolution");
					RESOLVE_TREE(enp,NO_VEXPR_NODE);
					DUMP_TREE(enp);
				}
			}

			if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) ){
				dval = EVAL_FLT_EXP(VN_CHILD(enp,1));
				dbl_to_scalar(&sval,dval,OBJ_PREC_PTR(dp));
				ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			} else {
				EVAL_OBJ_ASSIGNMENT(dp,VN_CHILD(enp,1));
			}
			return;
			}
		case T_SCAL_DECL:
			SET_VN_DECL_PREC(enp, prec_p);

			break;
		case T_CSCAL_DECL:					/* eval_decl_stat */
			SET_VN_DECL_PREC(enp, prec_p);

			/* eg float x{3} */
			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
				/* float x{} */
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(QSP_ARG  enp,dsp);
				else
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp,SHP_TYPE_DIMS(VN_SHAPE(enp)));
			} else {
				SET_DIMENSION(dsp,0,EVAL_INT_EXP(VN_CHILD(enp,0)) );
				if( DIMENSION(dsp,0) == 0 ){
					setup_unknown_shape(QSP_ARG  enp,dsp);
				}
			}
			break;
		case T_VEC_DECL:			/* eval_decl_stat */
			SET_VN_DECL_PREC(enp, prec_p);

			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
				if( ! IS_RESOLVED(enp) ) {
					setup_unknown_shape(QSP_ARG  enp,dsp);
				} else {
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp, SHP_TYPE_DIMS(VN_SHAPE(enp)));
				}
			} else {
				SET_DIMENSION(dsp,1,EVAL_INT_EXP(VN_CHILD(enp,0)) );
				if( DIMENSION(dsp,1) == 0 ){
					setup_unknown_shape(QSP_ARG  enp,dsp);
				}
			}
			break;
		case T_CVEC_DECL:
			SET_VN_DECL_PREC(enp, prec_p);

			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(QSP_ARG  enp,dsp);
				else {
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp, SHP_TYPE_DIMS(VN_SHAPE(enp)));
				}
			} else {
				SET_DIMENSION(dsp,1,EVAL_INT_EXP(VN_CHILD(enp,0)) );
				SET_DIMENSION(dsp,0,EVAL_INT_EXP(VN_CHILD(enp,1)) );
				if( DIMENSION(dsp,2) == 0 || DIMENSION(dsp,1) == 0 ){
					setup_unknown_shape(QSP_ARG  enp,dsp);
				}
			}
			break;
		case T_IMG_DECL:
			SET_VN_DECL_PREC(enp, prec_p);

			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
				if( ! IS_RESOLVED(enp) ){
					setup_unknown_shape(QSP_ARG  enp,dsp);
				} else {
/*
sprintf(ERROR_STRING,"resolved, nr = %d  nc = %d",SHP_TYPE_DIM(VN_SHAPE(enp),2),
SHP_TYPE_DIM(VN_SHAPE(enp),1));
advise(ERROR_STRING);
*/
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp, SHP_TYPE_DIMS(VN_SHAPE(enp)));
				}
			} else {
				SET_DIMENSION(dsp,2,EVAL_INT_EXP(VN_CHILD(enp,0)) );
				SET_DIMENSION(dsp,1,EVAL_INT_EXP(VN_CHILD(enp,1)) );
				if( DIMENSION(dsp,2) == 0 || DIMENSION(dsp,1) == 0 ){
					setup_unknown_shape(QSP_ARG  enp,dsp);
				}

			}
			break;
		case T_CIMG_DECL:
			SET_VN_DECL_PREC(enp, prec_p);

			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(QSP_ARG  enp,dsp);
				else {
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp, SHP_TYPE_DIMS(VN_SHAPE(enp)));
				}
			} else {
				SET_DIMENSION(dsp,2,EVAL_INT_EXP(VN_CHILD(enp,0)) );
				SET_DIMENSION(dsp,1,EVAL_INT_EXP(VN_CHILD(enp,1)) );
				SET_DIMENSION(dsp,0,EVAL_INT_EXP(VN_CHILD(enp,2)) );
				if( DIMENSION(dsp,2) == 0 || DIMENSION(dsp,1) == 0 || DIMENSION(dsp,0) == 0 ){
					setup_unknown_shape(QSP_ARG  enp,dsp);
				}
			}
			break;
		case T_SEQ_DECL:
			SET_VN_DECL_PREC(enp, prec_p);

			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(QSP_ARG  enp,dsp);
				else {
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp, SHP_TYPE_DIMS(VN_SHAPE(enp)));
				}
			} else {
				SET_DIMENSION(dsp,3,EVAL_INT_EXP(VN_CHILD(enp,0)) );
				SET_DIMENSION(dsp,2,EVAL_INT_EXP(VN_CHILD(enp,1)) );
				SET_DIMENSION(dsp,1,EVAL_INT_EXP(VN_CHILD(enp,2)) );
				if( DIMENSION(dsp,3) == 0 || DIMENSION(dsp,2) == 0 || DIMENSION(dsp,1) == 0 ){
					setup_unknown_shape(QSP_ARG  enp,dsp);
				}
			}
			break;
		case T_CSEQ_DECL:
			SET_VN_DECL_PREC(enp, prec_p);

			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(QSP_ARG  enp,dsp);
				else {
					/* BUG?  type_dimset or mach_dimset? */
					COPY_DIMS(dsp, SHP_TYPE_DIMS(VN_SHAPE(enp)));
				}
			} else {
				Vec_Expr_Node *enp2;
				SET_DIMENSION(dsp,3,EVAL_INT_EXP(VN_CHILD(enp,0)) );
				SET_DIMENSION(dsp,2,EVAL_INT_EXP(VN_CHILD(enp,1)) );
				enp2 = VN_CHILD(enp,2);
//#ifdef CAUTIOUS
//				if( VN_CODE(enp2) != T_EXPR_LIST )
//					WARN("CAUTIOUS:  node should be T_EXPR_LIST!?");
//#endif /* CAUTIOUS */
				assert( VN_CODE(enp2) == T_EXPR_LIST );

				SET_DIMENSION(dsp,1,EVAL_INT_EXP(VN_CHILD(enp2,0)) );
				SET_DIMENSION(dsp,0,EVAL_INT_EXP(VN_CHILD(enp2,1)) );
				if( DIMENSION(dsp,3) == 0 || DIMENSION(dsp,2) == 0 || DIMENSION(dsp,1) == 0 || DIMENSION(dsp,0) == 0 ){
					setup_unknown_shape(QSP_ARG  enp,dsp);
				}
			}
			break;
		case T_PTR_DECL:			/* eval_decl_stat() */
			SET_VN_DECL_PREC(enp, prec_p);

			/* call by reference */
			if( type != ID_STRING )
				type = ID_POINTER;
//#ifdef CAUTIOUS
//			if( type == ID_STRING && PREC_CODE(prec_p) != PREC_CHAR ){
//				NODE_ERROR(enp);
//				WARN("CAUTIOUS:  string object does not have string prec!?");
//			}
//#endif /* CAUTIOUS */
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
			MISSING_CASE(enp,"eval_decl_stat");
			break;
	}

	if( PREC_CODE(prec_p)==PREC_COLOR ){
		SET_DIMENSION(dsp,0,3 );
	} else if( PREC_CODE(prec_p)==PREC_VOID ){	/* determine from context? */
		if( VN_SHAPE(enp) != NO_SHAPE )
			prec_p=SHP_PREC_PTR(VN_SHAPE(enp));
	}

	/* We allow name conflicts at levels above the current context.
	 * RESTRICT_ITEM_CONTEXT causes item lookup to only use the top context.
	 */

	RESTRICT_ID_CONTEXT(1);

//#ifdef CAUTIOUS
//	if( VN_STRING(enp) == NULL ){
//		NODE_ERROR(enp);
//		sprintf(ERROR_STRING,"CAUTIOUS:  eval_decl_stat:  %s has null string!?",node_desc(enp));
//		WARN(ERROR_STRING);
//	}
//#endif /* CAUTIOUS */
	assert( VN_STRING(enp) != NULL );

	// Make sure this name has not been used already...
	idp = ID_OF(VN_STRING(enp));
	if( idp != NO_IDENTIFIER ){
		NODE_ERROR(enp);
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

	RESTRICT_ID_CONTEXT(0);
	idp=ID_OF(VN_STRING(enp));
	if( idp != NO_IDENTIFIER ){
		/* only print this message once (the code seems to be
		 * executed 3 times!?
		 */
		if( ! WAS_WARNED(enp) ){
			NODE_ERROR(enp);
			sprintf(ERROR_STRING,"declaration of %s masks previous declaration",VN_STRING(enp));
			advise(ERROR_STRING);
			MARK_WARNED(enp)
		}
/*
show_context_stack(QSP_ARG  dobj_itp);
*/
		/* this stuff should all be debug only... */
		/*
		if( ID_TYPE(idp) == ID_OBJECT ){
			Vec_Expr_Node *decl_enp;
			sprintf(ERROR_STRING,"context of %s (%s) is %s",
				ID_NAME(idp),OBJ_NAME(idp->id_dp),CONTEXT_NAME(ID_DOBJ_CTX(idp)));
			advise(ERROR_STRING);
			decl_enp = OBJ_EXTRA(idp->id_dp);
			if( decl_enp != NO_VEXPR_NODE ){
				advise("previous object declaration at:");
				NODE_ERROR(decl_enp);
			}
			sprintf(ERROR_STRING,"current context is %s",
				CONTEXT_NAME(((Item_Context *)NODE_DATA(QLIST_HEAD(DOBJ_CONTEXT_LIST)))));
			advise(ERROR_STRING);
		}
		*/
	}

	/* remember the declaration identifier context */
	// Could we use CURRENT_CONTEXT equivalently???
	SET_VN_DECL_CTX(enp, (Item_Context *)NODE_DATA(QLIST_HEAD(ID_CONTEXT_LIST)) );

//sprintf(ERROR_STRING,"eval_decl_stat:  creating id %s, type = %d",
//VN_STRING(enp),type);
//advise(ERROR_STRING);
	// New items are always created in the top context.

	idp = new_id(QSP_ARG  VN_STRING(enp));		/* eval_decl_stat */
	SET_ID_TYPE(idp, type);


//#ifdef CAUTIOUS
//	if( idp == NO_IDENTIFIER ){
//		NODE_ERROR(enp);
//		sprintf(ERROR_STRING,"CAUTIOUS:  unable to create new identifier %s",VN_STRING(enp));
//		WARN(ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( idp != NO_IDENTIFIER );

	switch( type ){

		case ID_REFERENCE:
			SET_ID_DOBJ_CTX(idp , (Item_Context *)NODE_DATA(QLIST_HEAD(DOBJ_CONTEXT_LIST)) );
			SET_ID_REF(idp, NEW_REFERENCE );
			SET_REF_ID(ID_REF(idp), idp );
			SET_REF_DECL_VN(ID_REF(idp), enp );	/* BUG? */
			SET_REF_TYPE(ID_REF(idp), OBJ_REFERENCE );
			// This would be a place to OR in the STATIC flag
			// if this declaration is not within a subroutine?
			// see dangling pointer problem...  subrt_ctx
			SET_REF_OBJ(ID_REF(idp), finish_obj_decl(QSP_ARG  enp,dsp,prec_p,decl_flags) );	/* eval_decl_stat */

			// This was CAUTIOUS before, but this can happen
			// if the user tries to create an object that
			// was already created outside of the expression parser

			if( REF_OBJ(ID_REF(idp)) == NO_OBJ ){
				// Need to clean up!
				del_id(QSP_ARG  idp);

				NODE_ERROR(enp);
				sprintf(ERROR_STRING,
			"eval_decl_stat:  unable to create object for id %s",ID_NAME(idp));
				WARN(ERROR_STRING);
			}

			break;

		case ID_STRING:
			SET_ID_REF(idp, NEW_REFERENCE );
			SET_REF_ID(ID_REF(idp), idp );
			SET_REF_DECL_VN(ID_REF(idp), enp );	/* BUG? */
			SET_REF_TYPE(ID_REF(idp), STR_REFERENCE );
			SET_REF_SBUF(ID_REF(idp), NEW_STRINGBUF );
			REF_SBUF(ID_REF(idp))->sb_buf = NULL;
			REF_SBUF(ID_REF(idp))->sb_size = 0;
			break;
		case ID_POINTER:
			SET_ID_PTR(idp, NEW_POINTER );
			SET_PTR_REF(ID_PTR(idp), NO_REFERENCE);
			SET_PTR_FLAGS(ID_PTR(idp), 0);
			SET_PTR_DECL_VN(ID_PTR(idp), enp);
			copy_node_shape(enp,uk_shape(PREC_CODE(prec_p)));
			break;
		case ID_FUNCPTR:
			//SET_ID_FUNC(idp, (Function_Ptr *)getbuf(sizeof(Function_Ptr)) );
			SET_ID_FUNC(idp, NEW_FUNC_PTR );
			ID_FUNC(idp)->fp_srp = NO_SUBRT;
			copy_node_shape(enp,uk_shape(PREC_CODE(prec_p)));
			break;
		default:
			NODE_ERROR(enp);
			sprintf(ERROR_STRING,
				"identifier type %d not handled by eval_decl_stat switch",
				type);
			WARN(ERROR_STRING);
			break;
	}
} /* end eval_decl_stat */

static void eval_extern_decl(QSP_ARG_DECL Precision * prec_p,Vec_Expr_Node *enp, int decl_flags)
{
	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_PROTO:
			{
			Subrt *srp;
			srp=subrt_of(QSP_ARG  VN_STRING(enp));
			if( srp == NO_SUBRT ) EVAL_DECL_STAT(prec_p,enp,decl_flags);
			else {
				/* This subroutine has already been declared...
				 * make sure the type matches
				 */
				if( PREC_CODE(prec_p) != SR_PREC_CODE(srp) )
					prototype_mismatch(QSP_ARG  SR_ARG_DECLS(srp),enp);
			}

			/* BUG make sure arg decls match */
			return;
			}
		case T_BADNAME: return;
		case T_DECL_ITEM_LIST:
			EVAL_EXTERN_DECL(prec_p,VN_CHILD(enp,0),decl_flags);
			if( VN_CHILD(enp,1)!=NO_VEXPR_NODE )
				EVAL_EXTERN_DECL(prec_p,VN_CHILD(enp,1),decl_flags);
			return;
		case T_DECL_INIT:
			NODE_ERROR(enp);
			advise("no auto-initialization with extern declarations");
			EVAL_EXTERN_DECL(prec_p,VN_CHILD(enp,0),decl_flags);
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

			dp=DOBJ_OF(VN_STRING(enp));
			if( dp == NO_OBJ ){
				EVAL_DECL_STAT(prec_p,enp,decl_flags);
				return;
			}
			/* BUG should check that decl matches earlier one... */
			break;
			}
		case T_PTR_DECL:			/* eval_extern_decl */
			{
			Identifier *idp;
			idp = ID_OF(VN_STRING(enp));
			if( idp == NO_IDENTIFIER ){
				EVAL_DECL_STAT(prec_p,enp,decl_flags);
				return;
			}
			/* BUG chould check that type matches earlier type */
			break;
			}

		default:
			MISSING_CASE(enp,"eval_extern_decl");
			break;
	}
}

/* We call eval_tree when we may have declarations as well as statements */

int eval_tree(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj *dst_dp)
{
	int ret_val=1;

	if( enp==NO_VEXPR_NODE || IS_CURDLED(enp) ) return(ret_val);

	eval_enp = enp;
	if( interrupted ) return(0);

#ifdef FOOBAR
	/*
	if( signal(SIGINT,intr_evaluation) == SIG_ERR ){
		ERROR1("error setting evaluation interrupt handler");
		IOS_RETURN
	}
		*/
#endif /* FOOBAR */

	if( going ) return(EVAL_WORK_TREE(enp,dst_dp));

	switch(VN_CODE(enp)){
		case T_EXIT:
			{
			int status;
			status = (int) EVAL_INT_EXP( VN_CHILD(enp,0) );
			exit(status);
			}

		case T_STAT_LIST:			/* eval_tree */
			/* used to call eval_tree on children here - WHY? */
			ret_val=EVAL_WORK_TREE(enp,dst_dp);
			break;

		case T_GO_FWD:  case T_GO_BACK:		/* eval_tree */
sprintf(ERROR_STRING,"eval_tree GOTO, dst_dp = %s",
dst_dp==NULL?"null":OBJ_NAME(dst_dp));
advise(ERROR_STRING);
		case T_BREAK:
		ALL_INCDEC_CASES
		case T_CALLFUNC:
		case T_ASSIGN:
		case T_RETURN:
		case T_WARN:
		case T_IFTHEN:
		case T_EXP_PRINT:
		case T_DISPLAY:
			ret_val = EVAL_WORK_TREE(enp,dst_dp);
			break;

		case T_DECL_STAT:
			/* why en_intval here, and not en_cast_prec??? */
			EVAL_DECL_STAT(VN_DECL_PREC(enp),VN_CHILD(enp,0),VN_DECL_FLAGS(enp));
			break;
		case T_EXTERN_DECL:
			EVAL_EXTERN_DECL(VN_DECL_PREC(enp),VN_CHILD(enp,0),VN_DECL_FLAGS(enp));
			break;
		default:
			MISSING_CASE(enp,"eval_tree");
			ret_val = EVAL_WORK_TREE(enp,dst_dp);
			break;
	}
	return(ret_val);
} // eval_tree

#ifdef NOT_YET
static int get_filetype_index(const char *name)
{
	int i;

	for(i=0;i<N_FILETYPE;i++){
		if( !strcmp(ft_tbl[i].ft_name,name) )
			return(i);
	}
	sprintf(DEFAULT_ERROR_STRING,"Invalid filetype name %s",name);
	NWARN(DEFAULT_ERROR_STRING);
	sprintf(DEFAULT_ERROR_STRING,"Valid selections are:");
	advise(DEFAULT_ERROR_STRING);
	for(i=0;i<N_FILETYPE;i++){
		sprintf(DEFAULT_ERROR_STRING,"\t%s", ft_tbl[i].ft_name);
		advise(DEFAULT_ERROR_STRING);
	}
	return(-1);
}
#endif /* NOT_YET */

static void eval_info_stat(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_UNDEF:
			break;

		case T_EXPR_LIST:
			EVAL_INFO_STAT(VN_CHILD(enp,0));
			EVAL_INFO_STAT(VN_CHILD(enp,1));
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
			dp = EVAL_OBJ_REF(enp);
			if( dp==NO_OBJ )
				WARN("missing info object");
			else {
				LONGLIST(dp);
			}
			break;
		default:
			MISSING_CASE(enp,"eval_info_stat");
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
			"show_ref:  ref at 0x%lx:  object %s",
			(int_for_addr)refp, OBJ_NAME(REF_OBJ(refp)));
		advise(ERROR_STRING);
	} else if( REF_TYPE(refp) == STR_REFERENCE ){
		sprintf(ERROR_STRING,"show_ref:  string");
		advise(ERROR_STRING);
	} else {
		sprintf(ERROR_STRING,"show_ref:  unexpected ref type %d",
			REF_TYPE(refp));
		WARN(ERROR_STRING);
	}
}

static void show_ptr(Pointer *ptrp)
{
	sprintf(ERROR_STRING,"Pointer at 0x%lx",(int_for_addr)ptrp);
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
			if( ret_enp != NO_VEXPR_NODE ) return(ret_enp);
			ret_enp=find_goto(VN_CHILD(enp,1));
			return(ret_enp);

		case T_DECL_STAT:
		case T_EXP_PRINT:
			return(NO_VEXPR_NODE);

		default:
			MISSING_CASE(enp,"find_goto");
			break;
	}
	return(NO_VEXPR_NODE);
}
#endif /* FOOBAR */

long eval_int_exp(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	long lval,lval2;
	double dval1,dval2;
	Data_Obj *dp;
	Scalar_Value *svp,sval;
	Subrt *srp;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		/* case T_MATH1_FUNC: */	/* returns double - should have been typecast? */
		case T_VS_FUNC:
			dp = EVAL_OBJ_EXP(enp,NO_OBJ);
			if( dp == NO_OBJ ){
				NODE_ERROR(enp);
				WARN("unable to evaluate vector-scalar expression");
				return(0);
			}
			if( !IS_SCALAR(dp) ){
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,
	"eval_int_exp T_VS_FUNC:  object %s is not a scalar!?",OBJ_NAME(dp));
				WARN(ERROR_STRING);
				return(0);
			}
			return( get_long_scalar_value(dp) );
			break;

		case T_RECIP:
			lval = EVAL_INT_EXP(VN_CHILD(enp,0));
			if( lval == 0 ){
				NODE_ERROR(enp);
				WARN("divide by zero!?");
				return(0);
			}
			return(1/lval);
			
		case T_TYPECAST:	/* eval_int_exp */
			dval1 = EVAL_FLT_EXP(VN_CHILD(enp,0));
			switch(VN_PREC(enp)){
				case PREC_BY:   return( (long) ((char)     dval1 ) );
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
					if( dval1 == 0.0 ) return(0);
					else return(1);

//#ifdef CAUTIOUS
				default:
//					NODE_ERROR(enp);
//					sprintf(ERROR_STRING,
//		"CAUTIOUS:  eval_int_exp:  unhandled precision %s (%"PREC_FMT_D", 0x%"PREC_FMT_X") in TYPECAST switch",
//						NAME_FOR_PREC_CODE(VN_PREC(enp)),VN_PREC(enp),VN_PREC(enp));
//					ERROR1(ERROR_STRING);
//					IOS_RETURN_VAL(0)
//#endif /* CAUTIOUS */
					assert( AERROR("eval_int_exp:  unhandled precision") );
			}
			break;

		case T_CALLFUNC:			/* eval_int_exp */
			/* This could get called if we use a function inside a dimesion bracket... */
			if( ! executing ) return(0);

			srp=VN_CALL_SUBRT(enp);
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
			dp=make_local_dobj(QSP_ARG  scalar_dsp,SR_PREC_PTR(srp));
			EXEC_SUBRT(enp,dp);
			/* get the scalar value */
			lval = get_long_scalar_value(dp);
			delvec(QSP_ARG  dp);
			return(lval);
			break;

		case T_BOOL_PTREQ:
			{
			Identifier *idp1,*idp2;
			idp1=EVAL_PTR_REF(VN_CHILD(enp,0),EXPECT_PTR_SET);
			idp2=EVAL_PTR_REF(VN_CHILD(enp,1),EXPECT_PTR_SET);
			/* CAUTIOUS check for ptrs? */
			/* BUG? any other test besides dp ptr identity? */
			if( REF_OBJ(ID_REF(idp1)) == REF_OBJ(ID_REF(idp2)) )
				return(1);
			else
				return(0);
			}

		case T_POSTINC:
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			lval = EVAL_INT_EXP(VN_CHILD(enp,0));
			inc_obj(dp);
			return(lval);

		case T_POSTDEC:
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			lval = EVAL_INT_EXP(VN_CHILD(enp,0));
			dec_obj(dp);
			return(lval);

		case T_PREDEC:
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			dec_obj(dp);
			return(EVAL_INT_EXP(VN_CHILD(enp,0)));

		case T_PREINC:		/* eval_int_exp */
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			inc_obj(dp);
			return(EVAL_INT_EXP(VN_CHILD(enp,0)));

		case T_ASSIGN:
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			lval = EVAL_INT_EXP(VN_CHILD(enp,1));
			int_to_scalar(&sval,lval,OBJ_PREC_PTR(dp));
			if( ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval) < 0 )
				return(0);
			return(lval);

		case T_UNDEF:
			return(0);

		case T_FILE_EXISTS:
			{
				const char *s;
				s=EVAL_STRING(VN_CHILD(enp,0));
				if( s != NULL )
					return(file_exists(QSP_ARG  s));
				else
					return(0);
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
			dp = EVAL_OBJ_REF(enp);
//#ifdef CAUTIOUS
//			if( dp==NO_OBJ ){
//				sprintf(ERROR_STRING,
//	"CAUTIOUS:  eval_int_exp:  missing dobj %s",VN_STRING(enp));
//				WARN(ERROR_STRING);
//				return(0);
//			}
//#endif /* CAUTIOUS */
			assert( dp != NO_OBJ );

			if( ! IS_SCALAR(dp) ){
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,
	"eval_int_exp:  Object %s is not a scalar!?",OBJ_NAME(dp));
				WARN(ERROR_STRING);
				LONGLIST(dp);
				return(0);
			}
			/* has the object been set? */
			if( ! HAS_ALL_VALUES(dp) ){
				if( executing && expect_objs_assigned ){
					unset_object_warning(QSP_ARG  enp,dp);
				}
				
				return(0);			/* we don't print the warning unless we know
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
//#ifdef CAUTIOUS
				case PREC_NONE:
				case N_MACHINE_PRECS:
				default:
//					sprintf(ERROR_STRING,
//			"CAUTIOUS: eval_int_exp:  %s has nonsense precision",
//					OBJ_NAME(dp));
//					WARN(ERROR_STRING);
					assert( AERROR("eval_int_exp:  nonsense precision") );
					lval=0.0;	// quiet compiler
					break;
//#endif /* CAUTIOUS */
			}
			return(lval);
			break;

		case T_BOOL_OR:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			lval2=EVAL_INT_EXP(VN_CHILD(enp,1));
			if( lval || lval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_AND:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			lval2=EVAL_INT_EXP(VN_CHILD(enp,1));
			if( lval && lval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_NOT:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			if( ! lval ) return(1);
			else return(0);
			break;
		case T_BOOL_GT:			/* eval_int_exp */
			dval1=EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2=EVAL_FLT_EXP(VN_CHILD(enp,1));
			if( dval1 > dval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_LT:			/* eval_int_exp */
			dval1=EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2=EVAL_FLT_EXP(VN_CHILD(enp,1));
			if( dval1 < dval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_GE:
			dval1=EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2=EVAL_FLT_EXP(VN_CHILD(enp,1));
			if( dval1 >= dval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_LE:
			dval1=EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2=EVAL_FLT_EXP(VN_CHILD(enp,1));
			if( dval1 <= dval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_NE:
			dval1=EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2=EVAL_FLT_EXP(VN_CHILD(enp,1));
			if( dval1 != dval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_EQ:
			dval1=EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2=EVAL_FLT_EXP(VN_CHILD(enp,1));
			if( dval1 == dval2 ) return(1);
			else return(0);
			break;
		case T_PLUS:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			lval2=EVAL_INT_EXP(VN_CHILD(enp,1));
			return(lval+lval2);
			break;
		case T_MINUS:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			lval2=EVAL_INT_EXP(VN_CHILD(enp,1));
			return(lval-lval2);
			break;
		case T_TIMES:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			lval2=EVAL_INT_EXP(VN_CHILD(enp,1));
			return(lval*lval2);
			break;
		case T_DIVIDE:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			lval2=EVAL_INT_EXP(VN_CHILD(enp,1));
			if( lval2==0 ){
				WARN("integer division by 0!?");
				return(0L);
			}
			return(lval/lval2);
			break;
		case T_MODULO:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			lval2=EVAL_INT_EXP(VN_CHILD(enp,1));
			if( lval2==0 ){
				WARN("integer division (modulo) by 0!?");
				return(0L);
			}
			return(lval%lval2);
			break;
		case T_BITOR:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			lval2=EVAL_INT_EXP(VN_CHILD(enp,1));
			return(lval|lval2);
			break;
		case T_BITAND:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			lval2=EVAL_INT_EXP(VN_CHILD(enp,1));
			return(lval&lval2);
			break;
		case T_BITXOR:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			lval2=EVAL_INT_EXP(VN_CHILD(enp,1));
			return(lval^lval2);
			break;
		case T_BITCOMP:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			return(~lval);
			break;
		case T_BITRSHIFT:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			lval2=EVAL_INT_EXP(VN_CHILD(enp,1));
			return(lval>>lval2);
			break;
		case T_BITLSHIFT:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			lval2=EVAL_INT_EXP(VN_CHILD(enp,1));
			return(lval<<lval2);
			break;
		case T_LIT_INT:			/* eval_int_exp */
			return(VN_INTVAL(enp));
			break;
		case T_UMINUS:
			lval=EVAL_INT_EXP(VN_CHILD(enp,0));
			return(-lval);
			break;

		case T_STR2_FN:			/* eval_int_exp */
		case T_LIT_DBL:
		case T_STR1_FN:	/* eval_int_exp */
		case T_SIZE_FN: 	/* eval_int_exp */
			lval= (long) EVAL_FLT_EXP(enp);
			return(lval);
			break;
		default:
			MISSING_CASE(enp,"eval_int_exp");
			break;
	}
	return(-1L);
} /* end eval_int_exp */

/* Process a tree, doing only declarations */

void eval_decl_tree(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	if( enp==NO_VEXPR_NODE )
		return;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_STAT_LIST:
			/* There should only be one T_STAT_LIST at
			 * the beginning of a subroutine, so we
			 * don't need to scan the second child.
			 * for declarations...
			 */
			EVAL_DECL_TREE(VN_CHILD(enp,0));
			break;
		case T_DECL_STAT_LIST:
			EVAL_DECL_TREE(VN_CHILD(enp,0));
			EVAL_DECL_TREE(VN_CHILD(enp,1));
			break;
		case T_DECL_STAT:
			EVAL_DECL_STAT(VN_DECL_PREC(enp),VN_CHILD(enp,0),VN_DECL_FLAGS(enp));
			break;
		case T_EXTERN_DECL:
			EVAL_EXTERN_DECL(VN_DECL_PREC(enp),VN_CHILD(enp,0),VN_DECL_FLAGS(enp));
			break;
		default:
			/* We will end up here with any code
			 * in a subroutine with no declarations.
			 */
			/*
			advise("You can safely ignore this warning???");
			MISSING_CASE(enp,"eval_decl_tree");
			*/
			break;
	}
}

static int compare_arg_decls(Vec_Expr_Node *enp1,Vec_Expr_Node *enp2)
{
	int i;

	if( enp1 == NO_VEXPR_NODE ) {
		if( enp2 == NO_VEXPR_NODE ) return(0);
		else return(-1);
	} else if( enp2 == NO_VEXPR_NODE ) return(-1);

	if( VN_CODE(enp1) != VN_CODE(enp2) ) return(-1);

	if( VN_CODE(enp1) == T_DECL_STAT ){
		if( VN_DECL_PREC(enp1) != VN_DECL_PREC(enp2) ) return(-1);
	}

	for(i=0;i<MAX_CHILDREN(enp1);i++){
		if( compare_arg_decls(VN_CHILD(enp1,i),VN_CHILD(enp2,i)) < 0 )
			return(-1);
	}
	return(0);
}

void compare_arg_trees(QSP_ARG_DECL  Vec_Expr_Node *enp1,Vec_Expr_Node *enp2)
{
	if( compare_arg_decls(enp1,enp2) < 0 )
		prototype_mismatch(QSP_ARG  enp1,enp2);
}



/* This whole check is probably CAUTIOUS */

static int bad_reeval_shape(Vec_Expr_Node *enp)
{
	eval_enp = enp;

	if( VN_SHAPE(enp) == NO_SHAPE ){
		sprintf(DEFAULT_ERROR_STRING,
	"reeval:  missing shape info for %s",VN_STRING(enp));
		NWARN(DEFAULT_ERROR_STRING);
		return(1);
	}
	if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
		sprintf(DEFAULT_ERROR_STRING,
	"reeval:  unknown shape info for %s",VN_STRING(enp));
		NWARN(DEFAULT_ERROR_STRING);
		return(1);
	}
	return(0);
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

void reeval_decl_stat(QSP_ARG_DECL  Precision *prec_p,Vec_Expr_Node *enp,int decl_flags)
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
			reeval_decl_stat(QSP_ARG  prec_p,VN_CHILD(enp,0),decl_flags);
			if( VN_CHILD(enp,1)!=NO_VEXPR_NODE )
				reeval_decl_stat(QSP_ARG  prec_p,VN_CHILD(enp,1),decl_flags);
			return;
			break;

		case T_SCAL_DECL:
			return ;

		case T_IMG_DECL:			/* reeval_decl_stat */
		case T_VEC_DECL:
		case T_SEQ_DECL:
		case T_CSCAL_DECL:
		case T_CIMG_DECL:			/* reeval_decl_stat */
		case T_CVEC_DECL:
		case T_CSEQ_DECL:
			if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
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
			MISSING_CASE(enp,"reeval_decl_stat");
			break;
	}

	/* First make sure that the context of this declaration is active */
	PUSH_ID_CONTEXT(VN_DECL_CTX(enp));
	idp = ID_OF(VN_STRING(enp));
	POP_ID_CONTEXT;
	//pop_item_context(QSP_ARG  id_itp);

//#ifdef CAUTIOUS
//	if( idp == NO_IDENTIFIER ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  missing id obj %s",VN_STRING(enp));
//		ERROR1(ERROR_STRING);
//		IOS_RETURN
//	}
	assert( idp != NO_IDENTIFIER );

//	if( ! IS_REFERENCE(idp) ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  reeval_decl_stat:  identifier %s does not refer to an object!?",
//			ID_NAME(idp));
//		ERROR1(ERROR_STRING);
//		IOS_RETURN
//	}
//#endif /* CAUTIOUS */
	assert( IS_REFERENCE(idp) );

	dp=REF_OBJ(ID_REF(idp));

//#ifdef CAUTIOUS
//	if( dp == NO_OBJ ){
//		NODE_ERROR(enp);
//		sprintf(ERROR_STRING,"CAUTIOUS:  expected to find uk obj %s!?",
//			VN_STRING(enp));
//		ERROR1(ERROR_STRING);
//		IOS_RETURN
//	}
//#endif /* CAUTIOUS */
	assert( dp != NO_OBJ );

	/* the size may not be unknown, if we were able to determine
	 * it's size during the first scan, .e.g. LOAD, or a known obj
	 */
	if( ! UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
		return;
	}

	if( ID_DOBJ_CTX(idp) != NODE_DATA(QLIST_HEAD(DOBJ_CONTEXT_LIST)) ){
		context_pushed=1;
		PUSH_DOBJ_CONTEXT(ID_DOBJ_CTX(idp));
	} else context_pushed=0;

	delvec(QSP_ARG  dp);

	SET_REF_OBJ(ID_REF(idp), finish_obj_decl(QSP_ARG  enp,dsp,prec_p,decl_flags) );	/* reeval_decl_stat */

	if( context_pushed )
		POP_DOBJ_CONTEXT;

} /* end reeval_decl_stat */

/* eval_obj_id
 *
 * returns a ptr to an Identifier.
 * This is called from eval_ptr_ref
 */

static Identifier *eval_obj_id(QSP_ARG_DECL Vec_Expr_Node *enp)
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
			/* Subvectors & subscripted objects normally don't have permenent identifiers.
			 * We'll make one here from the object itself, but this creates a big
			 * problem with memory getting eaten up.  This was solved in the data library
			 * by having a small pool of temporary objects - but how do we know how
			 * many is enough here???
			 */
			dp = EVAL_OBJ_REF(enp);
//#ifdef CAUTIOUS
//			if( dp == NO_OBJ ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,
//						"CAUTIOUS:  eval_obj_id %s:  missing object!?",node_desc(enp));
//				WARN(ERROR_STRING);
//				break;
//			}
//#endif /* CAUTIOUS */
			assert( dp != NO_OBJ );

			/* now make an identifier to go with this thing */
			idp = make_named_reference(QSP_ARG  OBJ_NAME(dp));
			SET_REF_OBJ(ID_REF(idp), dp );
			SET_ID_SHAPE(idp, OBJ_SHAPE(dp) );
			return(idp);

		case T_OBJ_LOOKUP:
			s = EVAL_STRING(VN_CHILD(enp,0));
			goto find_obj;

		case T_STATIC_OBJ:			/* eval_obj_id */
			s=OBJ_NAME(VN_OBJ(enp));
			goto find_obj;

		case T_DYN_OBJ:				/* eval_obj_id */
			s=VN_STRING(enp);
			/* fall-thru */
find_obj:
			idp = ID_OF(s);
//#ifdef CAUTIOUS
//			if( idp == NO_IDENTIFIER ){
//				sprintf(ERROR_STRING,"CAUTIOUS:  eval_obj_id:  missing identifier %s",s);
//				ERROR1(ERROR_STRING);
//				IOS_RETURN_VAL(NULL)
//			}
			assert( idp != NO_IDENTIFIER );

//			if( ! IS_REFERENCE(idp) ){
//				sprintf(ERROR_STRING,"CAUTIOUS:  eval_obj_id %s:  identifier is not a reference",
//					ID_NAME(idp));
//				ERROR1(ERROR_STRING);
//				IOS_RETURN_VAL(NULL)
//			}
//#endif /* CAUTIOUS */
			assert( IS_REFERENCE(idp) );

			return(idp);

		case T_STRING:
			/* make a local string name */
			{
				Dimension_Set ds1, *dsp=(&ds1);

				STRING_DIMENSION(dsp,(dimension_t)strlen(VN_STRING(enp))+1);
				dp=make_local_dobj(QSP_ARG  dsp,prec_for_code(PREC_STR));
				if( dp == NO_OBJ ){
					WARN("unable to make temporary object");
					return(NO_IDENTIFIER);
				}
				strcpy((char *)OBJ_DATA_PTR(dp),VN_STRING(enp));
				idp = make_named_reference(QSP_ARG  OBJ_NAME(dp));
				SET_REF_TYPE(ID_REF(idp), STR_REFERENCE );
				SET_REF_OBJ(ID_REF(idp), dp );
				if( idp == NO_IDENTIFIER ){
					ERROR1("error making identifier for temp string obj");
					IOS_RETURN_VAL(NULL)
				}
				return(idp);
			}

		default:
			MISSING_CASE(enp,"eval_obj_id");
			break;
	}
	return(NO_IDENTIFIER);
}

/*
 * Two ways to call eval_ptr_ref:
 * when a ptr is dereferenced, or appears on the RHS, it must be set!
 */

Identifier *eval_ptr_ref(QSP_ARG_DECL Vec_Expr_Node *enp,int expect_ptr_set)
{
	Identifier *idp;

	eval_enp = enp;

	switch(VN_CODE(enp)){
		case T_EQUIVALENCE:
			idp = EVAL_OBJ_ID(enp);
			return(idp);
			break;

		case T_CALLFUNC:		/* a function that returns a pointer */
			return( EXEC_REFFUNC(enp) );

		case T_REFERENCE:
			idp = EVAL_OBJ_ID(VN_CHILD(enp,0));
//#ifdef CAUTIOUS
//			if( idp == NO_IDENTIFIER ){
//				NODE_ERROR(enp);
//				DUMP_TREE(enp);
//				ERROR1("CAUTIOUS:  eval_ptr_ref:  missing reference target");
//				IOS_RETURN_VAL(NULL)
//			}
//#endif	/* CAUTIOUS */
			assert( idp != NO_IDENTIFIER );

			return( idp );

		case T_UNDEF:
			return(NO_IDENTIFIER);

		case T_POINTER:		/* eval_ptr_ref */
		case T_STR_PTR:		/* eval_ptr_ref */
			idp = GET_ID(VN_STRING(enp));
//#ifdef CAUTIOUS
//			if( idp==NO_IDENTIFIER ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,"CAUTIOUS:  missing identifier object (%s) #1!?",VN_STRING(enp));
//				WARN(ERROR_STRING);
//				return(idp);
//			}
//#endif /* CAUTIOUS */
			assert( idp != NO_IDENTIFIER );

			/* BUG this is not an error if the ptr is on the left hand side... */
			if( executing && expect_ptr_set ){
				if( IS_POINTER(idp) && !POINTER_IS_SET(idp) ){
					NODE_ERROR(enp);
					sprintf(ERROR_STRING,"object pointer \"%s\" used before value is set",ID_NAME(idp));
					advise(ERROR_STRING);
				} else if( IS_STRING_ID(idp) && !STRING_IS_SET(idp) ){
					NODE_ERROR(enp);
					sprintf(ERROR_STRING,"string pointer \"%s\" used before value is set",ID_NAME(idp));
					advise(ERROR_STRING);
				}
			}
			return(idp);
			break;
		default:
			MISSING_CASE(enp,"eval_ptr_ref");
			break;
	}
	return(NO_IDENTIFIER);
} /* end eval_ptr_rhs */

Identifier *get_set_ptr(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Identifier *idp;

	idp = EVAL_PTR_REF(enp,EXPECT_PTR_SET);

//#ifdef CAUTIOUS
//	if( idp==NO_IDENTIFIER ){
//		NODE_ERROR(enp);
//		WARN("CAUTIOUS:  missing pointer identifier!?");
//		return(NO_IDENTIFIER);
//	}
	assert( idp != NO_IDENTIFIER );

//	if( ! IS_POINTER(idp) ){
//		NODE_ERROR(enp);
//		sprintf(ERROR_STRING,
//			"CAUTIOUS:  eval_obj_ref:  id %s does not refer to a pointer!?",
//			ID_NAME(idp));
//		WARN(ERROR_STRING);
//		return(NO_IDENTIFIER);
//	}
//#endif /* CAUTIOUS */
	assert( IS_POINTER(idp) );


	if( ! POINTER_IS_SET(idp) )
		return(NO_IDENTIFIER);

	return(idp);
}

// Currently, it seems that we don't allow subvectors with curly braces?
// But that is T_CSUBVEC!?

static Data_Obj *eval_subvec(QSP_ARG_DECL  Data_Obj *dp, index_t index, index_t i2 )
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
	dp2=DOBJ_OF(newname);
	if( dp2 != NO_OBJ ) return(dp2);

	dp2=mk_subseq(QSP_ARG  newname,dp,offsets,dsp);
	if( dp2 == NO_OBJ ) return(dp2);
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

static Data_Obj *eval_subscript1(QSP_ARG_DECL Data_Obj *dp, Vec_Expr_Node *enp)
{
	index_t index,index2;
	Data_Obj *dp2;

	if( VN_CODE(enp) == T_RANGE ){
		/* T_RANGE has 3 children, and is used to specify subsamping:
		 * start : end : inc
		 */
		WARN("eval_subscript1:  Sorry, not sure how to handle T_RANGE node");
		return(NO_OBJ);
	} else if( VN_CODE(enp) == T_RANGE2 ){
		index = (index_t) EVAL_INT_EXP(VN_CHILD(enp,0));
		index2 = (index_t) EVAL_INT_EXP(VN_CHILD(enp,1));
		/* Now we need to make a subvector */
		dp2 = eval_subvec(QSP_ARG  dp,index-1,index2-1) ;
		return(dp2);
	}

	/* index = EVAL_INT_EXP(VN_CHILD(enp,1)); */
	index = (index_t) EVAL_FLT_EXP(enp);
sprintf(ERROR_STRING,"eval_subscript1:  index is %d",index);
advise(ERROR_STRING);

	/* d_subscript fails if the index is too large,
	 * but in matlab we want to automagically make the array larger
	 */
	insure_object_size(QSP_ARG  dp,index);

	dp2 = D_SUBSCRIPT(dp,index);
	return( dp2 );
}

/* make something new */

static Data_Obj *create_matrix(QSP_ARG_DECL Vec_Expr_Node *enp,Shape_Info *shpp)
{
	Data_Obj *dp;
	Identifier *idp;

	switch(VN_CODE(enp)){
		case T_RET_LIST:
			return( CREATE_LIST_LHS(enp) );

		case T_DYN_OBJ:		/* create_matrix */
			/* we need to create an identifier too! */
			idp = make_named_reference(QSP_ARG  VN_STRING(enp));
			dp = make_dobj(QSP_ARG  VN_STRING(enp),SHP_TYPE_DIMS(shpp),SHP_PREC_PTR(shpp));

//#ifdef CAUTIOUS
//			if( dp == NO_OBJ ){
//				NODE_ERROR(enp);
//				ERROR1("CAUTIOUS:  create_matrix:  make_dobj failed");
//				IOS_RETURN_VAL(NULL)
//			}
//#endif /* CAUTIOUS */
			assert( dp != NO_OBJ );

			SET_REF_OBJ(ID_REF(idp), dp );
			SET_ID_SHAPE(idp, OBJ_SHAPE(dp) );
			return(dp);
		default:
			MISSING_CASE(enp,"create_matrix");
			break;
	}
	return(NO_OBJ);
}

/* For matlab, if the rhs shape is different, then we reshape the LHS to match.
 * (for other languages, this might be an error!)
 * The node passed is generally the assign node...
 */

static Data_Obj *mlab_target(QSP_ARG_DECL Data_Obj *dp, Vec_Expr_Node *enp)
{
	if( dp == NO_OBJ ){
		dp = CREATE_MATRIX(enp,VN_SHAPE(enp));
	}
	else {
	/* BUG should check reshape if already exists */
		sprintf(ERROR_STRING,"mlab_target %s:  not checking reshape",OBJ_NAME(dp));
	  	WARN(ERROR_STRING);
	}
	return(dp);
}

static Data_Obj *create_list_lhs(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp1,*dp2;
	List *lp;
	Node *np1,*np2;

	dp1 = EVAL_OBJ_REF(VN_CHILD(enp,0));
	dp1 = MLAB_TARGET(dp1,VN_CHILD(enp,0));
	dp2 = EVAL_OBJ_REF(VN_CHILD(enp,1));
	dp2 = MLAB_TARGET(dp2,VN_CHILD(enp,1));
	np1=mk_node(dp1);
	np2=mk_node(dp2);
	lp=new_list();
	addTail(lp,np1);
	addTail(lp,np2);
	dp1=make_obj_list(QSP_ARG  localname(),lp);
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

static dimension_t assign_obj_from_list(QSP_ARG_DECL Data_Obj *dp,Vec_Expr_Node *enp,index_t index)
{
	dimension_t i1,i2;
	Data_Obj *sub_dp,*src_dp;
	double dval;
	Scalar_Value sval;

	eval_enp = enp;

	assert(dp!=NO_OBJ);

	if( UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
		/* does the rhs have a shape??? */
		/* Why haven't we resolved when we are here? */
		WARN("assign_obj_from_list:  LHS has unknown shape!?");
	}

/*
sprintf(ERROR_STRING,"assign_obj_from_list  dp = %s, enp = %s, index = %d",
OBJ_NAME(dp),node_desc(enp),index);
advise(ERROR_STRING);
LONGLIST(dp);
DUMP_TREE(enp);
*/
	switch(VN_CODE(enp)){
		case T_TYPECAST:			/* assign_obj_from_list */
			/* do we need to do anything??? BUG? */
			i1=ASSIGN_OBJ_FROM_LIST(dp,VN_CHILD(enp,0),index);
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
			/*i1=*/ ASSIGN_OBJ_FROM_LIST(dp,VN_CHILD(enp,0),0);
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
				i1=ASSIGN_OBJ_FROM_LIST(dp,VN_CHILD(enp,0),index);
			} else {
				// If it's not a row_list node, then what is it???
				sub_dp = D_SUBSCRIPT(dp,index);
				i1=ASSIGN_OBJ_FROM_LIST(sub_dp,VN_CHILD(enp,0),index);
				delvec(QSP_ARG  sub_dp);
			}


			if( VN_CODE(VN_CHILD(enp,1)) == T_ROW_LIST ){
				i2=ASSIGN_OBJ_FROM_LIST(dp,VN_CHILD(enp,1),index+i1);
			} else {
				sub_dp = D_SUBSCRIPT(dp,index+i1);
				i2=ASSIGN_OBJ_FROM_LIST(sub_dp,VN_CHILD(enp,1),index+i1);
				delvec(QSP_ARG  sub_dp);
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
				i1=ASSIGN_OBJ_FROM_LIST(dp,VN_CHILD(enp,0),index);
			} else {
				sub_dp = C_SUBSCRIPT(dp,index);
				i1=ASSIGN_OBJ_FROM_LIST(sub_dp,VN_CHILD(enp,0),index);
				delvec(QSP_ARG  sub_dp);
			}


			if( VN_CODE(VN_CHILD(enp,1)) == T_COMP_LIST ){
				i2=ASSIGN_OBJ_FROM_LIST(dp,VN_CHILD(enp,1),index+i1);
			} else {
				sub_dp = C_SUBSCRIPT(dp,index+i1);
				i2=ASSIGN_OBJ_FROM_LIST(sub_dp,VN_CHILD(enp,1),index+i1);
				delvec(QSP_ARG  sub_dp);
			}
			return(i1+i2);
			break;

		case T_LIT_DBL:			/* assign_obj_from_list */
			dbl_to_scalar(&sval,VN_DBLVAL(enp),OBJ_MACH_PREC_PTR(dp));
assign_literal:
			if( ! IS_SCALAR(dp) ){
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,
	"assign_obj_from_list:  %s[%d] is not a scalar",OBJ_NAME(dp),
					index);
				WARN(ERROR_STRING);
				return(1);
			}
			assign_scalar(QSP_ARG  dp,&sval);
			return(1);
			break;

		case T_LIT_INT:				/* assign_obj_from_list */
			int_to_scalar(&sval,VN_INTVAL(enp),OBJ_MACH_PREC_PTR(dp));
			goto assign_literal;

		ALL_SCALAR_FUNCTION_CASES
		ALL_SCALAR_BINOP_CASES
			/* we allow arbitrary expressions within braces. */
			dval = EVAL_FLT_EXP(enp);
			dbl_to_scalar(&sval,dval,OBJ_MACH_PREC_PTR(dp));
			goto assign_literal;

		case T_STATIC_OBJ:	/* assign_obj_from_list */
		case T_DYN_OBJ:		/* assign_obj_from_list */
		case T_VS_FUNC:
		case T_VV_FUNC:
			src_dp = EVAL_OBJ_EXP(enp,NO_OBJ);
			if( src_dp==NO_OBJ){
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,
			"assign_obj_from_list:  error evaluating RHS");
				WARN(ERROR_STRING);
				return(0);
			}
			/* do we need to make sure they are the same size??? */
			//setvarg2(oap,dp,src_dp);
			dp_convert(QSP_ARG  dp,src_dp);
			return(1);
			break;

		default:
			MISSING_CASE(enp,"assign_obj_from_list");
			break;
	}
WARN("assign_obj_from_list returning 0!?");
	return(0);
}

Data_Obj *eval_obj_ref(QSP_ARG_DECL Vec_Expr_Node *enp)
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
				RESOLVE_TREE(enp,NO_VEXPR_NODE);
			}
			if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
				NODE_ERROR(enp);
				WARN("unable to determine shape of equivalence");
DUMP_TREE(enp);
				return(NO_OBJ);
			}
			{
			Data_Obj *parent_dp;

			/* we assume that the shape is known... */
			parent_dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( parent_dp == NO_OBJ ){
				NODE_ERROR(enp);
				WARN("invalid parent object for equivalence");
				return(NO_OBJ);
			}
			dp = make_equivalence(QSP_ARG  localname(), parent_dp,SHP_TYPE_DIMS(VN_SHAPE(enp)),VN_DECL_PREC(enp));
			if( dp == NO_OBJ ){
				NODE_ERROR(enp);
				WARN("unable to create equivalence");
			}
			return(dp);
			}
			break;

		case T_RET_LIST:		/* eval_obj_ref() */
			{
			/* BUG - what if this is not the LHS??? */
			return( CREATE_LIST_LHS(enp) );
			break;
			}
			
		case T_COMP_OBJ:	/* eval_obj_ref */
		case T_LIST_OBJ:	/* eval_obj_ref */
			/* we seem to need this when we have typecast a list object... */

			/* We declare a temporary object and then assign it.
			 * This is rather inefficient, but we don't expect to do it often
			 * or for large objects.
			 */
			dp=make_local_dobj(QSP_ARG   SHP_TYPE_DIMS(VN_SHAPE(VN_CHILD(enp,0))), VN_PREC_PTR(enp));
			ASSIGN_OBJ_FROM_LIST(dp,VN_CHILD(enp,0),0);
			//SET_OBJ_FLAG_BITS(dp, DT_ASSIGNED);
			note_assignment(dp);
/*
advise("eval_obj_ref returning:");
LONGLIST(dp);
pntvec(QSP_ARG  dp, tell_msgfile() );
*/
			return(dp);
			break;

		/* matlab */
		case T_SUBSCRIPT1:	/* eval_obj_ref */
			/* it seems that d_subscript knows about indices starting at 1? */
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( dp == NO_OBJ ) return(dp);

			/* In matlab, we can have a list of indices inside
			 * the paren's...  We need to know if the list trees
			 * grow to the left or right??
			 * SHould this code be matlab only? BUG?
			 */
			while( VN_CODE(VN_CHILD(enp,1)) == T_INDEX_LIST ){
				enp=VN_CHILD(enp,1);
				dp2 = EVAL_SUBSCRIPT1(dp,VN_CHILD(enp,0));
				if( dp2 == NO_OBJ ){
					return(dp2);
				}
				dp=dp2;
			}
			/* BUG doesn't enforce reference to an existing object!? */
			dp2=EVAL_SUBSCRIPT1(dp,VN_CHILD(enp,1));
			return(dp2);
			break;

		case T_PREINC:			/* eval_obj_ref */
		case T_PREDEC:
		case T_POSTINC:
		case T_POSTDEC:
			return( EVAL_OBJ_REF( VN_CHILD(enp,0) ) );

		case T_DEREFERENCE:	/* eval_obj_ref */
			/* runtime resolution, we may not be able to do this until ptrs have been assigned */
			if( UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_obj_ref:  last ditch attempt at runtime resolution of %s",node_desc(VN_CHILD(enp,0)));
advise(ERROR_STRING);
DUMP_TREE(enp);
}
#endif /* QUIP_DEBUG */
				/*
				resolve_one_uk_node(VN_CHILD(enp,0));
				*/
				RESOLVE_TREE(VN_CHILD(enp,0),NO_VEXPR_NODE);
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_obj_ref:  after last ditch attempt at runtime resolution of %s",node_desc(VN_CHILD(enp,0)));
advise(ERROR_STRING);
DUMP_TREE(enp);
}
#endif /* QUIP_DEBUG */
			}

			idp = GET_SET_PTR(VN_CHILD(enp,0));
			if( idp == NO_IDENTIFIER ){
				/* actually , this could just be used before set? */
				/* We also get here if we use an object expression in a declaration,
				 * which can't be evaluated until run-time...
				 */

/*
NODE_ERROR(enp);
WARN("eval_obj_ref DEREFERENCE:  missing identifier!?");
DUMP_TREE(enp);
*/
				return(NO_OBJ);
			}
			/* At one time we did a CAUTIOUS check to see if the object was in the database,
			 * but that could fail because the pointed to object might be out of scope
			 * (e.g.  ptr to local object passed as subrt arg)
			 */

//#ifdef CAUTIOUS
//			if( ! IS_POINTER(idp) ){
//				sprintf(ERROR_STRING,"CAUTIOUS:  eval_obj_ref:  identifier %s isn't a pointer!?",ID_NAME(idp));
//				ERROR1(ERROR_STRING);
//				IOS_RETURN_VAL(NULL)
//			}
			assert( IS_POINTER(idp) );

//			if( PTR_REF(ID_PTR(idp)) == NO_REFERENCE ){
//				sprintf(ERROR_STRING,"CAUTIOUS:  eval_obj_ref:  target of pointer %s isn't set!?",ID_NAME(idp));
//				ERROR1(ERROR_STRING);
//				IOS_RETURN_VAL(NULL)
//			}
//#endif /* CAUTIOUS */
			assert( PTR_REF(ID_PTR(idp)) != NO_REFERENCE );

			return(REF_OBJ(PTR_REF(ID_PTR(idp))));

		case T_OBJ_LOOKUP:
			s=EVAL_STRING(VN_CHILD(enp,0));
			if( s == NULL ) return(NO_OBJ);
			dp=get_id_obj(QSP_ARG  s,enp);
			return(dp);

		case T_UNDEF:
			return(NO_OBJ);

		case T_REFERENCE:
			return( EVAL_OBJ_REF(VN_CHILD(enp,0)) );

		case T_STATIC_OBJ:		/* eval_obj_ref() */
			return(VN_OBJ(enp));

		case T_DYN_OBJ:		/* eval_obj_ref */
			return( get_id_obj(QSP_ARG  VN_STRING(enp),enp) );

		case T_CURLY_SUBSCR:				/* eval_obj_ref */
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( dp == NO_OBJ ) return(dp);
			index = (index_t) EVAL_INT_EXP(VN_CHILD(enp,1));
			dp=C_SUBSCRIPT(dp,index);
			return(dp);

		case T_SQUARE_SUBSCR:				/* eval_obj_ref */
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( dp == NO_OBJ ) return(dp);
			/* Before we evaluate the subscript as an integer, check and
			 * see if it's a vector...
			 */
			if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) ){
				index = (index_t) EVAL_INT_EXP(VN_CHILD(enp,1));
				dp = D_SUBSCRIPT(dp,index);
				return( dp );
			} else {
				NODE_ERROR(enp);
				WARN("eval_obj_ref:  vector indices are not allowed");
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

			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( dp == NO_OBJ ) return(dp);

			/* is this matlab order??? */
			index = (index_t) EVAL_INT_EXP(VN_CHILD(VN_CHILD(enp,1),0));	/* start */
			if( mode_is_matlab ){
				/* start : inc : end */
				inc = (incr_t) EVAL_INT_EXP(VN_CHILD(VN_CHILD(enp,1),1));
				i2 = (index_t) EVAL_INT_EXP(VN_CHILD(VN_CHILD(enp,1),2));	/* end */
			} else {
				/* start : end : inc */
				i2 = (index_t) EVAL_INT_EXP(VN_CHILD(VN_CHILD(enp,1),1));	/* end */
				inc = (incr_t) EVAL_INT_EXP(VN_CHILD(VN_CHILD(enp,1),2));
			}

			sprintf(tmp_name,"%s[%d:%d:%d]",OBJ_NAME(dp),index,inc,i2);
			sub_dp = DOBJ_OF(tmp_name);
			if( sub_dp != NO_OBJ )
				return(sub_dp);		/* already exists */

			dsp = &ds1;
			COPY_DIMS(dsp , OBJ_TYPE_DIMS(dp) );
			SET_DIMENSION(dsp,OBJ_RANGE_MAXDIM(dp),1+(dimension_t)floor((i2-index)/inc) );	/* BUG assumes not reversed */
			offsets[OBJ_RANGE_MAXDIM(dp)] = index;
			incrs[OBJ_RANGE_MAXDIM(dp)] = inc;
			/* If we have referred to this before, the object may still exist */
			sub_dp = make_subsamp(QSP_ARG  tmp_name,dp,dsp,offsets,incrs);

			if( sub_dp == NO_OBJ ) return( sub_dp );
			SET_OBJ_RANGE_MAXDIM(sub_dp,
				OBJ_RANGE_MAXDIM(sub_dp)-1 );
			/* BUG?  make sure not less than mindim? */
			return( sub_dp );
			}
			break;
		case T_SUBVEC:					/* eval_obj_ref */
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( dp == NO_OBJ ) return(dp);

			// First make sure that we have a dimension available
			if( OBJ_RANGE_MAXDIM(dp) < OBJ_RANGE_MINDIM(dp) ){
				NODE_ERROR(VN_CHILD(enp,0));
				sprintf(ERROR_STRING,
		"Can't specify range for object %s!?",OBJ_NAME(dp));
				WARN(ERROR_STRING);
				return(NO_OBJ);
			}

			if( VN_CHILD(enp,1) == NO_VEXPR_NODE )
				index = 0;	/* first element by default */
			else
				index = (index_t) EVAL_INT_EXP(VN_CHILD(enp,1));

			if( VN_CHILD(enp,2) == NO_VEXPR_NODE )
				i2 = OBJ_TYPE_DIM(dp,OBJ_RANGE_MAXDIM(dp)) - 1;	/* last elt. */
			else
				i2 = (index_t) EVAL_INT_EXP(VN_CHILD(enp,2));

			return( eval_subvec(QSP_ARG  dp,index,i2) );
			break;
		case T_CSUBVEC:					/* eval_obj_ref */
			// Why is this so different from T_SUBVEC???
			// because eval_subvec is shared w/ matlab?...
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( dp == NO_OBJ ) return(dp);
			index = (index_t) EVAL_INT_EXP(VN_CHILD(enp,1));
			i2 = (index_t) EVAL_INT_EXP(VN_CHILD(enp,2));
			COPY_DIMS(dsp,OBJ_TYPE_DIMS(dp));
			for(i=0;i<N_DIMENSIONS;i++){
				offsets[i]=0;
			}
			SET_DIMENSION(dsp, OBJ_RANGE_MINDIM(dp) , i2+1-index );
			offsets[ OBJ_RANGE_MINDIM(dp) ] = index;
			sprintf(newname,"%s{%d:%d}",OBJ_NAME(dp),index,i2);
			dp2=DOBJ_OF(newname);
			if( dp2 != NO_OBJ ) return(dp2);

			dp2=mk_subseq(QSP_ARG  newname,dp,offsets,dsp);
			SET_OBJ_RANGE_MINDIM(dp2, OBJ_RANGE_MINDIM(dp)+1 );
			return(dp2);
			break;
		case T_REAL_PART:
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( dp == NO_OBJ ) return(dp);
			/* BUG make sure that the object is commplex! */
			return( C_SUBSCRIPT(dp,0) );
			break;
		case T_IMAG_PART:
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( dp == NO_OBJ ) return(dp);
			/* BUG make sure that the object is commplex! */
			return( C_SUBSCRIPT(dp,1) );
			break;

		default:
			MISSING_CASE(enp,"eval_obj_ref");
			break;
	}
	return(NO_OBJ);
} /* end eval_obj_ref() */

/*
 * get_2_operands
 *
 * enp is a binary operator node with two children that we must evaluate.
 * dst_dp points to the destination of the operator (or NO_OBJ), and can
 * be used for intermediate results, if the size is right...
 */

static void get_2_operands(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj **dpp1,Data_Obj **dpp2,Data_Obj *dst_dp)
{
	Data_Obj *dp1=NO_OBJ,*dp2;
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
			dp1=EVAL_BITMAP(NO_OBJ,VN_CHILD(enp,0));
		} else if( dst_dp!=NO_OBJ && same_shape(VN_SHAPE(VN_CHILD(enp,0)),OBJ_SHAPE(dst_dp)) ){
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),dst_dp);
		} else {
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
		}

		if( dst_dp!=NO_OBJ && dp1!=dst_dp && same_shape(VN_SHAPE(VN_CHILD(enp,1)),OBJ_SHAPE(dst_dp)) ){
			dp2=EVAL_OBJ_EXP(VN_CHILD(enp,1),dst_dp);
		} else {
			dp2=EVAL_OBJ_EXP(VN_CHILD(enp,1),NO_OBJ);
		}
	} else if( VN_LHS_REFS(VN_CHILD(enp,0)) == 0 ){
		/* the right hand subtree  refers to the lhs...
		 * but the left-hand subtree does not.
		 * we can proceed as above, but with r & l
		 * interchanged.
		 */
		if( dst_dp!=NO_OBJ && same_shape(VN_SHAPE(VN_CHILD(enp,1)),OBJ_SHAPE(dst_dp)) ){
			dp2=EVAL_OBJ_EXP(VN_CHILD(enp,1),dst_dp);
		} else {
			dp2=EVAL_OBJ_EXP(VN_CHILD(enp,1),NO_OBJ);
		}

		if( dp2!=dst_dp ){
			if( is_bitmap ){
				dp1=EVAL_BITMAP(NO_OBJ,VN_CHILD(enp,0));
			} else if( dst_dp!=NO_OBJ && same_shape(VN_SHAPE(VN_CHILD(enp,0)),OBJ_SHAPE(dst_dp)) ){
				dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),dst_dp);
			}
		} else {
			if( is_bitmap ){
				dp1=EVAL_BITMAP(NO_OBJ,VN_CHILD(enp,0));
			} else {
				dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			}
		}
	} else {
		/* Both sides refer to the lhs */
		if( is_bitmap )
			dp1=EVAL_BITMAP(NO_OBJ,VN_CHILD(enp,0));
		else
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);

		/* used to have dst_dp here, would have added a test here for shape match,
		 * but it seems that if this branch refers to the lhs then we probably don't
		 * want to use the destination object?
		 */

		dp2=EVAL_OBJ_EXP(VN_CHILD(enp,1),NO_OBJ);
	}

//#ifdef CAUTIOUS
//	if( dp1==NO_OBJ ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  get_2_operands:  null operand on left subtree!?");
//		WARN(ERROR_STRING);
//		DUMP_TREE(enp);
//	}
//	if( dp2 == NO_OBJ ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  get_2_operands:  null operand on right subtree!?");
//		WARN(ERROR_STRING);
//		DUMP_TREE(enp);
//	}
//#endif /* CAUTIOUS */
	assert( dp1 != NO_OBJ && dp2 != NO_OBJ );

	*dpp1 = dp1;
	*dpp2 = dp2;
} /* end get_2_operands() */

/* This is like eval_obj_assignment, except that the rhs is smaller that the LHS,
 * so we figure out which dimensions we need to iterate over, and then call eval_obj_assignment.
 * a bit wasteful perhaps...
 *
 * THis is kind of like projection?
 */

static void eval_dim_assignment(QSP_ARG_DECL Data_Obj *dp,Vec_Expr_Node *enp)
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
			sub_dp=make_subsamp(QSP_ARG  tmp_dst_name,dp,dsp,dst_offsets,dst_incrs);

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

				EVAL_DIM_ASSIGNMENT(sub_dp,enp);
			}
			delvec(QSP_ARG  sub_dp);
			/* all the work done in the recursive calls */
			return;
		}
	}
	/* now we're ready! */
	EVAL_OBJ_ASSIGNMENT(dp,enp);
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
//#ifdef CAUTIOUS
		default:
//			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  unhandled precision \"%s\" in scalar_to_double()",PREC_NAME(PREC_MACH_PREC_PTR(prec_p)));
//			NWARN(DEFAULT_ERROR_STRING);
			assert( AERROR("scalar_to_double:  unhandled precision") );
			break;
//#endif /* CAUTIOUS */
	}
	return(dval);
}

double eval_flt_exp(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp,*dp2;
	double dval;
	double dval2;
	index_t index;
	Scalar_Value *svp;
	Subrt *srp;
	//Dimension_Set dimset={{1,1,1,1,1}};
	Dimension_Set ds1, *dsp=(&ds1);
	Vec_Obj_Args oa1, *oap=&oa1;

	SET_DIMENSION(dsp,0,1);
	SET_DIMENSION(dsp,1,1);
	SET_DIMENSION(dsp,2,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,4,1);

	eval_enp = enp;

/*
sprintf(ERROR_STRING,"eval_flt_exp, code is %d",VN_CODE(enp));
advise(ERROR_STRING);
*/
	switch(VN_CODE(enp)){
		case T_MINVAL:
			dp2=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			if( dp2 == NO_OBJ ){
NODE_ERROR(VN_CHILD(enp,0));
WARN("error evaluating arg to min");
return(0.0);
}
			/* make a scalar object to hold the answer */
			dp=make_local_dobj(QSP_ARG  dsp,OBJ_PREC_PTR(dp2));
			clear_obj_args(oap);
			setvarg2(oap,dp,dp2);
			//vminv(oap);
			//vf_code=FVMINV;
			//h_vl2_vminv(HOST_CALL_ARGS);
			platform_dispatch_by_code(QSP_ARG  FVMINV, oap );
			dval = get_dbl_scalar_value(dp);
			delvec(QSP_ARG  dp);
			break;

		case T_CALLFUNC:			/* eval_flt_exp */
			/* This could get called if we use a function inside a dimesion bracket... */
			if( ! executing ) return(0);

			srp=VN_CALL_SUBRT(enp);
			/* BUG SHould check and see if the return type is double... */

			/* BUG at least make sure that it's not void... */

			/* make a scalar object to hold the return value... */
			dp=make_local_dobj(QSP_ARG  dsp,SR_PREC_PTR(srp));
			EXEC_SUBRT(enp,dp);
			/* get the scalar value */
			dval = get_dbl_scalar_value(dp);
			delvec(QSP_ARG  dp);
			break;


#ifdef NOT_YET
		case T_CALL_NATIVE:
			dval = eval_native_flt(enp) ;
			break;
#endif /* NOT_YET */

		/* matlab */
		case T_INNER:
			/* assume both children are scalars */
			dval = EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2 = EVAL_FLT_EXP(VN_CHILD(enp,1));
			dval *= dval2 ;
			break;

		case T_SUBSCRIPT1:			/* eval_flt_exp */
			dp=GET_OBJ(VN_STRING(VN_CHILD(enp,0)));
			index = (index_t) EVAL_FLT_EXP(VN_CHILD(enp,1));
			dp2 = D_SUBSCRIPT(dp,index);
			if( dp2 == NO_OBJ ){
				sprintf(ERROR_STRING,
		"Couldn't form subobject %s[%d]",OBJ_NAME(dp),index);
				WARN(ERROR_STRING);
				dval = 0.0;
			} else {
				svp = (Scalar_Value *)OBJ_DATA_PTR(dp2);
				dval = scalar_to_double(svp,OBJ_PREC_PTR(dp2));
			}
			break;

		/* end matlab */


		case T_POWER:
			dval = EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2 = EVAL_FLT_EXP(VN_CHILD(enp,1));
			dval = pow(dval,dval2) ;
			break;

		case T_POSTINC:
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			dval = EVAL_FLT_EXP(VN_CHILD(enp,0));
			inc_obj(dp);
			break;

		case T_POSTDEC:
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			dval = EVAL_FLT_EXP(VN_CHILD(enp,0));
			dec_obj(dp);
			break;

		case T_PREDEC:
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			dec_obj(dp);
			dval = EVAL_FLT_EXP(VN_CHILD(enp,0));
			break;

		case T_PREINC:					/* eval_flt_exp */
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			inc_obj(dp);
			dval = EVAL_FLT_EXP(VN_CHILD(enp,0));
			break;

		case T_TYPECAST:	/* eval_flt_exp */
			/* We could just call eval_flt_exp on the child node,
			 * But if we are casting a float to int, we need to round...
			 */
			dval = EVAL_FLT_EXP(VN_CHILD(enp,0));
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
//#ifdef CAUTIOUS
				default:
//					NODE_ERROR(enp);
//					sprintf(ERROR_STRING,
//		"CAUTIOUS:  eval_flt_exp:  unhandled precision (%s) in TYPECAST switch",
//						NAME_FOR_PREC_CODE(VN_PREC(enp)));
//					ERROR1(ERROR_STRING);
					assert( AERROR("eval_flt_exp:  unhandled precision") );
					dval = 0.0;
					break;
//#endif /* CAUTIOUS */
			}
			break;
		case T_UNDEF:
			dval=0.0;
			break;

		case T_STR2_FN:
			{
				const char *s1,*s2;
				s1=EVAL_STRING(VN_CHILD(enp,0));
				s2=EVAL_STRING(VN_CHILD(enp,1));
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
			str = EVAL_STRING(VN_CHILD(enp,0));
			if( str == NULL ){
				WARN("error evaluating string...");
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
				ip = find_sizable(QSP_ARG  VN_STRING(VN_CHILD(enp,0)));
				if(ip==NO_ITEM){
					sprintf(ERROR_STRING,
						"Couldn't find sizable object %s",
						VN_STRING(VN_CHILD(enp,0)));
					WARN(ERROR_STRING);
					dval=0.0;
					break;
				}
				dval = (*(FUNC_SZ_FUNC(VN_FUNC_PTR(enp))))(QSP_ARG  ip);
			} else {
				/* an objref expressions */
				int save_e;	/* objs don't need values to query their size */

				save_e = expect_objs_assigned;
				expect_objs_assigned=0;
				dp = EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
				expect_objs_assigned=save_e;

				if( dp == NO_OBJ ){
					/* This might not be an error if we have used an object
					 * expression as a dimension, e.g.
					 * float x[ncols(v)];
					 * where v is a subroutine argument...
					 */

					if( executing ){
						NODE_ERROR(enp);
						sprintf(ERROR_STRING,
				"bad object expression given for function %s",
							FUNC_NAME(VN_FUNC_PTR(enp)));
						WARN(ERROR_STRING);
DUMP_TREE(enp);
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
					NODE_ERROR(enp);
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

//				if( FUNC_DIM_INDEX(VN_FUNC_PTR(enp)) < 0 ){
//					sprintf(ERROR_STRING,
//				"CAUTIOUS:  eval_flt_exp:  bad size function dimension index!?");
//					WARN(ERROR_STRING);
//					dval = 0.0;
//				} else {
//					dval = get_dobj_size(QSP_ARG  dp,
//							FUNC_DIM_INDEX(VN_FUNC_PTR(enp)));
//				}

				assert( FUNC_DIM_INDEX(VN_FUNC_PTR(enp)) >= 0 );
				dval = get_dobj_size(QSP_ARG  dp,
							FUNC_DIM_INDEX(VN_FUNC_PTR(enp)));
			}
			break;

		case T_SCALMAX:
			dval = EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2 = EVAL_FLT_EXP(VN_CHILD(enp,1));
			if( dval < dval2 )
				dval = dval2;
			break;

		case T_SCALMIN:
			dval = EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2 = EVAL_FLT_EXP(VN_CHILD(enp,1));
			if( dval > dval2 )
				dval = dval2;
			break;

		case T_STATIC_OBJ:	/* eval_flt_exp */
			dp=VN_OBJ(enp);
			goto obj_flt_exp;

		case T_POINTER:
		case T_DYN_OBJ:		/* eval_flt_exp */
			dp=GET_OBJ(VN_STRING(enp));
//#ifdef CAUTIOUS
//			if( dp == NO_OBJ ){
//				sprintf(ERROR_STRING,"CAUTIOUS:  eval_flt_exp:  missing object %s",VN_STRING(enp));
//				WARN(ERROR_STRING);
//				dval=0.0;
//				break;
//			}
//#endif /* CAUTIOUS */
			assert( dp != NO_OBJ );

obj_flt_exp:

			/* check that this object is a scalar */
			if( OBJ_N_TYPE_ELTS(dp) != 1 ){
				/* what about a complex scalar? BUG */
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,
		"eval_flt_exp:  object %s is not a scalar!?",OBJ_NAME(dp));
				WARN(ERROR_STRING);
			}
			svp=(Scalar_Value *)OBJ_DATA_PTR(dp);
			if( svp == NO_SCALAR_VALUE ){
				NODE_ERROR(enp);
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
			/* dp=GET_OBJ(VN_STRING(VN_CHILD(enp,0))); */
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			index = (index_t) EVAL_INT_EXP(VN_CHILD(enp,1));
			dp2 = D_SUBSCRIPT(dp,index);
			if( dp2 == NO_OBJ ){
				sprintf(ERROR_STRING,
		"Couldn't form subobject %s[%d]",OBJ_NAME(dp),index);
				WARN(ERROR_STRING);
				dval=0.0;
			} else {
				svp = (Scalar_Value *)OBJ_DATA_PTR(dp2);
				dval = scalar_to_double(svp,OBJ_PREC_PTR(dp2));
			}
			break;

		case T_CURLY_SUBSCR:
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			index = (index_t) EVAL_INT_EXP(VN_CHILD(enp,1));
			dp2 = C_SUBSCRIPT(dp,index);
			if( dp2 == NO_OBJ ){
				sprintf(ERROR_STRING,
		"Couldn't form subobject %s[%d]",OBJ_NAME(dp),index);
				WARN(ERROR_STRING);
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
			dval = EVAL_FLT_EXP(VN_CHILD(enp,0));
		dval = (*FUNC_D1_FUNC(VN_FUNC_PTR(enp)))(dval);
			break;
		case T_MATH2_FN:				/* eval_flt_exp */
			dval = EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2 = EVAL_FLT_EXP(VN_CHILD(enp,1));
	dval = (*FUNC_D2_FUNC(VN_FUNC_PTR(enp)))(dval,dval2);
			break;
		case T_UMINUS:
			dval = EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval *= (-1);
			break;
		case T_RECIP:
			dval = EVAL_FLT_EXP(VN_CHILD(enp,0));
			if( dval == 0.0 ){
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,"Divide by 0!?");
				WARN(ERROR_STRING);
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
			dval=EVAL_INT_EXP(enp);
			break;
		case T_DIVIDE:
			dval=EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2=EVAL_FLT_EXP(VN_CHILD(enp,1));
			if( dval2==0.0 ){
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,"Divide by 0!?");
				WARN(ERROR_STRING);
				dval=(0.0);
				break;
			}
			dval/=dval2;
			break;
		case T_PLUS:
			dval=EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2=EVAL_FLT_EXP(VN_CHILD(enp,1));
			dval+=dval2;
			break;
		case T_MINUS:
			dval=EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2=EVAL_FLT_EXP(VN_CHILD(enp,1));
			dval-=dval2;
			break;
		case T_TIMES:
			dval=EVAL_FLT_EXP(VN_CHILD(enp,0));
			dval2=EVAL_FLT_EXP(VN_CHILD(enp,1));
			dval*=dval2;
			break;

		default:
			MISSING_CASE(enp,"eval_flt_exp");
			dval=0.0;	// make sure there is a value to return
			break;
	}

	//return(0.0);		// why return 0.0?
	return(dval);
}

#define INSURE_DESTINATION						\
									\
			if( dst_dp == NO_OBJ ){				\
				dst_dp=make_local_dobj(QSP_ARG 		\
					SHP_TYPE_DIMS(VN_SHAPE(enp)),	\
					SHP_PREC_PTR(VN_SHAPE(enp)));		\
			}



/* Evalutate an object expression.
 * Unlike object assignments, if the expression is an object
 * reference, we don't copy, we just return the pointer.
 * If we need to store some results, we put them in dst_dp,
 * or if dst_dp is null we create a temporary object.
 */

Data_Obj *eval_obj_exp(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj *dst_dp)
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
			EVAL_OBJ_ASSIGNMENT(dst_dp,enp);
			return(dst_dp);
			break;

		case T_RAMP:
			{
			/* We should distinguish between float and int ramps?  BUG */
			float start, dx, dy;
			start=(float)EVAL_FLT_EXP(VN_CHILD(enp,0));
			dx=(float)EVAL_FLT_EXP(VN_CHILD(enp,1));
			dy=(float)EVAL_FLT_EXP(VN_CHILD(enp,2));
			INSURE_DESTINATION
			easy_ramp2d(QSP_ARG  dst_dp,start,dx,dy);
			return(dst_dp);
			break;
			}

		case T_TRANSPOSE:
			dp=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			if( dp == NO_OBJ ) break;
			/* BUG make sure valid??? */
			INSURE_DESTINATION
			xpose_data(QSP_ARG  dst_dp,dp);
			return(dst_dp);
			break;

		case T_SCROLL:		/* eval_obj_exp */
			dp = EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			ldx=EVAL_INT_EXP(VN_CHILD(enp,1));
			ldy=EVAL_INT_EXP(VN_CHILD(enp,2));
//#ifdef CAUTIOUS
//			if( dp == NO_OBJ ){
//				WARN("CAUTIOUS:  eval_obj_exp:  missing scroll arg");
//				break;
//			}
//#endif /* CAUTIOUS */
			assert( dp != NO_OBJ );

			/* BUG? do we need to make sure that dp is not dst_dp? */
			INSURE_DESTINATION
			dp_scroll(QSP_ARG  dst_dp,dp,(incr_t)ldx,(incr_t)ldy);
			return(dst_dp);


		case T_WRAP:				/* eval_obj_exp */
			dp = EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
//#ifdef CAUTIOUS
//			if( dp == NO_OBJ ){
//				WARN("CAUTIOUS:  eval_obj_exp:  missing wrap arg");
//				break;
//			}
//#endif /* CAUTIOUS */
			assert( dp != NO_OBJ );

			/* BUG? do we need to make sure that dp is not dst_dp? */
			if( dst_dp == NO_OBJ ){
				dst_dp=make_local_dobj(QSP_ARG  
					SHP_TYPE_DIMS(VN_SHAPE(enp)),
					SHP_PREC_PTR(VN_SHAPE(enp)));
			}
			wrap(QSP_ARG  dst_dp,dp);
			return(dst_dp);

		case T_TYPECAST:			/* eval_obj_exp */
			/* The requested type should match the destination,
			 * and not match the child node...  this is supposed
			 * to be insured by the compilation process.
			 */
			return( EVAL_TYPECAST(enp,dst_dp) );

		case T_UNDEF:
			return(NO_OBJ);


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

			return( EVAL_OBJ_REF(enp) );
			break;

		case T_SQUARE_SUBSCR:
		case T_CURLY_SUBSCR:
			/* BUG - need separate code for the two types of subscript!? */
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( dp == NO_OBJ ) return(dp);

			/* Before we evaluate the subscript as an integer, check and
			 * see if it's a vector...
			 */
			if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) ){
				index_t index;
				index = (index_t)EVAL_INT_EXP(VN_CHILD(enp,1));
				if( VN_CODE(enp) == T_SQUARE_SUBSCR )
					return( D_SUBSCRIPT(dp,index) );
				else
					return( C_SUBSCRIPT(dp,index) );
			} else {
				Data_Obj *index_dp;
				index_dp=EVAL_OBJ_REF(VN_CHILD(enp,1));
				if( index_dp == NO_OBJ ) break;	/* BUG?  print error here? */
				if( OBJ_COMPS(index_dp) != (dimension_t)(1+OBJ_MAXDIM(dp)-OBJ_MINDIM(dp)) ){
					NODE_ERROR(enp);
					sprintf(ERROR_STRING,
	"Map source object %s needs %d indices, but index array %s has component dimension %d!?",
						OBJ_NAME(dp),1+OBJ_MAXDIM(dp)-OBJ_MINDIM(dp),
						OBJ_NAME(index_dp),OBJ_COMPS(index_dp));
					WARN(ERROR_STRING);
				} else {
					return( MAP_SUBSCRIPTS(dp,index_dp,enp) );
				}
			}
			break;

#ifdef MATLAB_FOOBAR
		/* matlab */
		case T_SUBSCRIPT1:	/* eval_obj_exp */
			dp=GET_OBJ(VN_STRING(VN_CHILD(enp,0)));
			index = EVAL_FLT_EXP(VN_CHILD(enp,1));
			dp2 = D_SUBSCRIPT(dp,index);
			return(dp2);
#endif /* MATLAB_FOOBAR */



		case T_CALLFUNC:

		case T_RIDFT:
		case T_RDFT:
		case T_VV_FUNC:
		case T_VS_FUNC:
		ALL_MATH_VFN_CASES			/* eval_obj_exp */
			if( dst_dp!=NO_OBJ ){
				EVAL_OBJ_ASSIGNMENT(dst_dp,enp);
				return(dst_dp);
			} else {
				/* We need to create a temporary object to
				 * hold the result...   hopefully we know
				 * the shape at this node!
				 */
//#ifdef CAUTIOUS
//				if( VN_SHAPE(enp)==NO_SHAPE ){
//					WARN(ERROR_STRING);
//					sprintf(ERROR_STRING,
//		"CAUTIOUS:  no shape at node %d, need for proto!?",VN_SERIAL(enp));
//					DUMP_TREE(SR_BODY(curr_srp));
//					return(NO_OBJ);
//				}
				assert( VN_SHAPE(enp) != NO_SHAPE );

//				if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
//					NODE_ERROR(enp);
//					WARN("CAUTIOUS:  eval_obj_exp:  proto node shape is UNKNOWN!?");
//					DUMP_TREE(enp);
//					return(NO_OBJ);
//				}
//#endif /* CAUTIOUS */
				assert( ! UNKNOWN_SHAPE(VN_SHAPE(enp)) );

				dst_dp=make_local_dobj(QSP_ARG   SHP_TYPE_DIMS(VN_SHAPE(enp)),
							SHP_PREC_PTR(VN_SHAPE(enp)));
//#ifdef CAUTIOUS
//				if( dst_dp == NO_OBJ ){
//			WARN("CAUTIOUS:  couldn't make shaped copy!?");
//					return(dst_dp);
//				}
//#endif /* CAUTIOUS */
				assert( dst_dp != NO_OBJ );

				EVAL_OBJ_ASSIGNMENT(dst_dp,enp);
				return(dst_dp);
			}
			break;


		default:
			MISSING_CASE(enp,"eval_obj_exp");
			break;
	}
	return(NO_OBJ);
} /* end eval_obj_exp() */

static const char *bad_string="XXXbad_stringXXX";

/* Construct a string from the given tree.
 * At runtime, it is an error to get a missing identifier,
 * but at other times (e.g. dump_tree) we may not be able
 * to expand everything...
 */

const char *eval_string(QSP_ARG_DECL Vec_Expr_Node *enp)
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
			WARN(ERROR_STRING);
			return("FOO");
			*/
			return( eval_native_string(enp) );
			break;
#endif /* NOT_YET */

		case T_SQUARE_SUBSCR:			/* eval_string */
		case T_CURLY_SUBSCR:
		case T_STATIC_OBJ:			/* eval_string */
		case T_DYN_OBJ:				/* eval_string */
			dp = EVAL_OBJ_EXP(enp,NO_OBJ);
			if( dp == NO_OBJ ){
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,"eval_string:  missing object %s",VN_STRING(enp));
				WARN(ERROR_STRING);
				return(NULL);
			}
//#ifdef CAUTIOUS
//			if( OBJ_PREC(dp) != PREC_CHAR ){
//				sprintf(ERROR_STRING,
//			"CAUTIOUS:  eval_string:  object %s (%s) should have %s precision",
//					OBJ_NAME(dp),OBJ_PREC_NAME(dp),
//						NAME_FOR_PREC_CODE(PREC_CHAR));
//				WARN(ERROR_STRING);
//				return(NULL);
//			}
//#endif /* CAUTIOUS */
			assert( OBJ_PREC(dp) == PREC_CHAR );

			/* not exactly a BUG, but we might verify that the number of
			 * columns matches the string length?
			 */
			return((char *)OBJ_DATA_PTR(dp));
			break;

		case T_SET_STR:
			if( dumping ) return(STRING_FORMAT);

			s = EVAL_STRING(VN_CHILD(enp,1));
			idp = EVAL_PTR_REF(VN_CHILD(enp,0),UNSET_PTR_OK);
			if( idp == NO_IDENTIFIER ) break;
			assign_string(QSP_ARG  idp,s,enp);
			return(s);

		case T_PRINT_LIST:
			return(EVAL_MIXED_LIST(enp));

		case T_STRING_LIST:
			{
			char *new_string;
			s1=EVAL_STRING(VN_CHILD(enp,0));
			s2=EVAL_STRING(VN_CHILD(enp,1));
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
				Item *ip;
				ip = find_sizable(QSP_ARG  VN_STRING(VN_CHILD(enp,0)));
				if(ip==NO_ITEM){
					sprintf(ERROR_STRING,
						"Couldn't find sizable object %s",
						VN_STRING(VN_CHILD(enp,0)));
					WARN(ERROR_STRING);
					break;
				}
				s = (*(FUNC_STRV_FUNC(VN_FUNC_PTR(enp))))(QSP_ARG  ip);
			} else {
				/* an objref expressions */
				int save_e;	/* objs don't need values to query their size */

				save_e = expect_objs_assigned;
				expect_objs_assigned=0;
				dp = EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
				expect_objs_assigned=save_e;

				if( dp == NO_OBJ ){
					/* This might not be an error if we have used an object
					 * expression as a dimension, e.g.
					 * float x[ncols(v)];
					 * where v is a subroutine argument...
					 */

					if( executing ){
						NODE_ERROR(enp);
						sprintf(ERROR_STRING,
				"bad object expression given for function %s",
							FUNC_NAME(VN_FUNC_PTR(enp)));
						WARN(ERROR_STRING);
DUMP_TREE(enp);
					}
					break;
				}
				if( UNKNOWN_SHAPE(OBJ_SHAPE(dp)) ){
					/* see comments for T_SIZE_FN... */
					/* return 0 to indicate that we don't know yet */
					NODE_ERROR(enp);
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

//#ifdef CAUTIOUS
//				if( strcmp(FUNC_NAME(VN_FUNC_PTR(enp)),"precision") ){
//					sprintf(ERROR_STRING,"CAUTIOUS:  Not properly handling function %s!?",
//						FUNC_NAME(VN_FUNC_PTR(enp)) );
//					WARN(ERROR_STRING);
//				}
//#endif // CAUTIOUS
				assert( ! strcmp(FUNC_NAME(VN_FUNC_PTR(enp)),"precision") );

				s = OBJ_PREC_NAME(dp);
			}
#ifdef FOOBAR
			// Old code for toupper etc.
			s=EVAL_STRING( VN_CHILD(enp,0) );
			if( s == NULL ) return(NULL);
			// Where do we get the destination string?
			// We need a temp object!?
			dp=make_local_dobj(QSP_ARG   SHP_TYPE_DIMS(VN_SHAPE(VN_CHILD(enp,0))), VN_PREC_PTR(enp));
			(*FUNC_STRV_FUNC(VN_FUNC_PTR(enp)))(OBJ_DATA_PTR(dp),s);
			return s;
#endif // FOOBAR
			break;

		case T_STR_PTR:			/* eval_string */
			if( dumping ) return(STRING_FORMAT);

			idp = EVAL_PTR_REF(enp,EXPECT_PTR_SET);

			if( idp == NO_IDENTIFIER ){
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,"missing string pointer object %s",VN_STRING(enp));
				advise(ERROR_STRING);
				return(NULL);
			}

//#ifdef CAUTIOUS
//			if( ! IS_STRING_ID(idp) ){
//				WARN("CAUTIOUS:  eval-string:  ptr not a string!?");
//				return(NULL);
//			}
//#endif /* CAUTIOUS */
			assert( IS_STRING_ID(idp) );

			if( REF_SBUF(ID_REF(idp))->sb_buf == NULL ){
				NODE_ERROR(enp);
				sprintf(ERROR_STRING,"string pointer \"%s\" used before set!?",ID_NAME(idp));
				WARN(ERROR_STRING);
				return(NULL);
			} else
				s=REF_SBUF(ID_REF(idp))->sb_buf;
			break;

		default:
			MISSING_CASE(enp,"eval_string");
			return(bad_string);
			break;
	}
	return(s);
}

//#ifdef FOOBAR
//void intr_evaluation(int arg)
//{
//	/* use setjmp/longjmp to get back to the interpreter */
//	if( eval_enp != NO_VEXPR_NODE )
//		NODE_ERROR(eval_enp);
//#ifdef CAUTIOUS
//	else NWARN("CAUTIOUS:  no current eval_enp!?");
//#endif /* CAUTIOUS */
//
//	advise("execution halted by SIGINTR");
//	interrupted=1;
//	sleep(2);
//	/* signal(SIGINT,intr_evaluation); */
//}
//#endif /* FOOBAR */

/* for matlab support */

void insure_object_size(QSP_ARG_DECL  Data_Obj *dp,index_t index)
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

		new_dp = make_dobj(QSP_ARG  "tmpname",dsp,OBJ_PREC_PTR(dp));

		/* set new data area to all zeroes */
		sval.u_d = 0.0;	/* BUG assumes PREC_DP */
		dp_const(QSP_ARG  new_dp,&sval);

		/* copy in original data */
		sub_dp = mk_subseq(QSP_ARG  "tmp_subseq",new_dp,offsets,OBJ_TYPE_DIMS(dp));
		dp_copy(QSP_ARG  sub_dp,dp);

		/* get rid of the subimage */
		delvec(QSP_ARG  sub_dp);

		/* now this is tricky...  we want to swap data areas, and dimensions
		 * between new_dp and dp...  here goes nothing
		 */
		tmp_data = OBJ_DATA_PTR(dp);
		SET_OBJ_DATA_PTR(dp, OBJ_DATA_PTR(new_dp));
		SET_OBJ_DATA_PTR(new_dp, tmp_data);

		SET_OBJ_TYPE_DIM(new_dp,which_dim, OBJ_TYPE_DIM(dp,which_dim) );
		SET_OBJ_TYPE_DIM(dp,which_dim, DIMENSION(dsp,which_dim) );

		delvec(QSP_ARG  new_dp);
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
	dp_new = make_dobj(QSP_ARG  "ass_tmp",SHP_TYPE_DIMS(shpp),SHP_PREC_PTR(shpp));
	/* BUG? we may have a problem with multiple return objects... */
	/* what should get_lhs_name() return when there are multiple
	 * objects on the lhs???
	 */
	if( dp != NO_OBJ ){
		delvec(QSP_ARG  dp);
	}
	obj_rename(QSP_ARG  dp_new,name);

	/* We also need to fix the identifier pointer */

	idp = GET_ID(name);

//	if( idp == NO_IDENTIFIER ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  missing matlab identifier %s",name);
//		WARN(ERROR_STRING);
//	} else if( ID_TYPE(idp) != ID_REFERENCE ){
//		sprintf(ERROR_STRING,"CAUTIOUS:  identifier %s is not a reference",name);
//		WARN(ERROR_STRING);
//	} else {
//		SET_REF_OBJ(ID_REF(idp), dp_new );
//		/* and update the shape! */
//		SET_ID_SHAPE(idp, OBJ_SHAPE(dp_new) );
//	}
	assert( idp != NO_IDENTIFIER );
	assert( ID_TYPE(idp) == ID_REFERENCE );

	SET_REF_OBJ(ID_REF(idp), dp_new );
	/* and update the shape! */
	SET_ID_SHAPE(idp, OBJ_SHAPE(dp_new) );
	return(dp_new);
}
				
static Data_Obj * mlab_lhs(QSP_ARG_DECL Data_Obj *dp, Vec_Expr_Node *enp)
{
//#ifdef CAUTIOUS
//	if( VN_CODE(enp) != T_ASSIGN ){
//		NODE_ERROR(enp);
//		sprintf(ERROR_STRING,"CAUTIOUS mlab_lhs:  %s is not an assign node",
//			node_desc(enp));
//		ERROR1(ERROR_STRING);
//		IOS_RETURN_VAL(NULL)
//	}
//#endif /* CAUTIOUS */
	assert( VN_CODE(enp) == T_ASSIGN );

	if( dp != NO_OBJ ){
		/* If the shape doesn't match,
		 * then we have to remake the object
		 */
if( VN_SHAPE(enp) == NO_SHAPE ){
WARN("mlab_eval_work_tree:  T_ASSIGN has null shape ptr");
} else {
if( VN_SHAPE(VN_CHILD(enp,1)) == NO_SHAPE ){
WARN("mlab_lhs:  rhs has null shape");
return NULL;
}

		if( OBJ_COLS(dp) != SHP_COLS(VN_SHAPE(VN_CHILD(enp,1))) ||
			OBJ_ROWS(dp) != SHP_ROWS(VN_SHAPE(VN_CHILD(enp,1))) ){

			Data_Obj *dp_new;
			const char *s;

			/* In matlab, you're allowed to reassign
			 * the shape of an object...
			 */

			s = GET_LHS_NAME(VN_CHILD(enp,0));
			dp_new = mlab_reshape(QSP_ARG  dp,VN_SHAPE(VN_CHILD(enp,1)),s);
			/* We do this later! */
			/* EVAL_OBJ_ASSIGNMENT(dp_new,VN_CHILD(enp,1)); */
			return(dp_new);
		}
} /* end debug */
	} else {
		/* make a new object */
//#ifdef CAUTIOUS
//		if( VN_CHILD(enp,0) == NO_VEXPR_NODE ){
//			sprintf(ERROR_STRING,"CAUTIOUS:  mlab_lhs:  %s has null child",node_desc(enp));
//			WARN(ERROR_STRING);
//			DUMP_TREE(enp);
//			ERROR1("CAUTIOUS:  giving up");
//			IOS_RETURN_VAL(NULL)
//		}
//#endif /* CAUTIOUS */
		assert( VN_CHILD(enp,0) != NO_VEXPR_NODE );

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
			WARN(ERROR_STRING);
			break;
	}
}
#endif /* MATLAB_FOOBAR */

void eval_immediate(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	/* Why do we need to do this here??? */
	enp=COMPILE_PROG(enp);

	if( IS_CURDLED(enp) ) return;

	if( dumpit ) {
		print_shape_key(SINGLE_QSP_ARG);
		DUMP_TREE(enp);
	}

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
			EVAL_WORK_TREE(enp,NO_OBJ);
			break;
		case T_DECL_STAT:
		case T_EXTERN_DECL:
			EVAL_DECL_TREE(enp);
			break;
		default:
			MISSING_CASE(enp,"eval_immediate");
DUMP_TREE(enp);
			break;
	}
	executing=0;

	// Should we release the tree here??
} /* end eval_immediate */


Data_Obj *
make_local_dobj(QSP_ARG_DECL  Dimension_Set *dsp,Precision *prec_p)
{
	Data_Obj *dp;
	Node *np;
	const char *s;

	dp=make_dobj(QSP_ARG  localname(),dsp,prec_p);
	if( dp == NO_OBJ ) return(dp);

	/* remember this for later deletion... */
	if( local_obj_lp == NO_LIST )
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

static void delete_local_objs(SINGLE_QSP_ARG_DECL)
{
	Node *np;
	Data_Obj *dp;
	const char *s;

	if( local_obj_lp == NO_LIST ) return;

	np=QLIST_HEAD(local_obj_lp);
	while(np!=NO_NODE){
		s = (char *)NODE_DATA(np);
		dp = DOBJ_OF(s);

//#ifdef CAUTIOUS
//		if( strncmp(s,"L.",2) ){
//			sprintf(ERROR_STRING,
//				"CAUTIOUS:  delete_local_objs:  Oops, object %s is on local object list!?",
//				OBJ_NAME(dp));
//			ERROR1(ERROR_STRING);
//			IOS_RETURN
//		}
//#endif /* CAUTIOUS */
		assert( ! strncmp(s,"L.",2) );

		if( dp != NO_OBJ ){
			delvec(QSP_ARG  dp);
		}
		np = NODE_NEXT(np);
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
 * bu the ensuing vector multiply!
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

static void eval_obj_assignment(QSP_ARG_DECL Data_Obj *dp,Vec_Expr_Node *enp)
{
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
	int vf_code=(-1);

/*
*/
	eval_enp = enp;

	if( dp == NO_OBJ ){
advise("eval_obj_assignment returning (NULL target)");
DUMP_TREE(enp);
		return;	/* probably an undefined reference */
	}

#ifdef QUIP_DEBUG
if( debug & eval_debug ){
sprintf(ERROR_STRING,"eval_obj_assignment %s",OBJ_NAME(dp));
advise(ERROR_STRING);
DUMP_TREE(enp);
}
#endif /* QUIP_DEBUG */

	switch(VN_CODE(enp)){
		case T_BOOL_EQ:
		case T_BOOL_NE:
		case T_BOOL_LT:
		case T_BOOL_GT:
		case T_BOOL_LE:
		case T_BOOL_GE:
		case T_BOOL_AND:
		case T_BOOL_OR:
		case T_BOOL_XOR:
		case T_BOOL_NOT:
		case T_BOOL_PTREQ:
			EVAL_BITMAP(dp,enp);
			break;

		case T_RANGE2:
			{
			double d1,d2;
			double delta;
			d1=EVAL_INT_EXP(VN_CHILD(enp,0));
			d2=EVAL_INT_EXP(VN_CHILD(enp,1));
			delta = (d2-d1)/(OBJ_N_TYPE_ELTS(dp)-1);
			easy_ramp2d(QSP_ARG  dp,d1,delta,0.0);
			}
			break;

		case T_STRING_LIST:
		case T_STRING:
//#ifdef CAUTIOUS
//			if( OBJ_PREC(dp) != PREC_CHAR ){
//				WARN("CAUTIOUS:");
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,"LHS (%s, %s) does not have %s precision, but RHS is a string",
//					OBJ_NAME(dp),OBJ_PREC_NAME(dp),NAME_FOR_PREC_CODE(PREC_CHAR));
//				advise(ERROR_STRING);
//				break;
//			}
//#endif /* CAUTIOUS */
			assert( OBJ_PREC(dp) == PREC_CHAR );

			s = EVAL_STRING(enp);

//#ifdef CAUTIOUS
//			if( OBJ_N_TYPE_ELTS(dp) <= strlen(s) ){
//				WARN("CAUTIOUS:");
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,"LHS (%s, %d) does not have space for RHS string",
//					OBJ_NAME(dp),OBJ_N_TYPE_ELTS(dp));
//				advise(ERROR_STRING);
//				break;
//			}
//#endif /* CAUTIOUS */
			assert( OBJ_N_TYPE_ELTS(dp) > strlen(s) );

			strcpy((char *)OBJ_DATA_PTR(dp),s);
			break;

		/* matlab */
#ifdef FOOBAR
		case T_ROWLIST:		/* eval_obj_assignment */
			/* rowlist trees grow down to the left, so we start with the bottom row
			 * and work up
			 */
			i=SHP_ROWS(VN_SHAPE(enp));
			/* But this child could be a matrix object? */
			ASSIGN_ROW(dp,i,VN_CHILD(enp,1));
			/* child[0] is either a ROWLIST node, or a ROW */
			EVAL_OBJ_ASSIGNMENT(dp,VN_CHILD(enp,0));
			break;
#endif

		case T_ROW:
			/* do we need to subscript dp?? */
			if( OBJ_ROWS(dp) > 1 ){
				dp2 = D_SUBSCRIPT(dp,1);
			} else {
				dp2 = dp;
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
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			dilate(QSP_ARG  dp,dp1);
			break;
		case T_ERODE:
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			erode(QSP_ARG  dp,dp1);
			break;
		/* COnditional assignment:  a<b ? v : w
		 * If the conditional is a scalar, then
		 * this is easy, we just to the assignment
		 * to one or the other...
		 * But if the conditional is a vector, then
		 * we need to evaluate it into a scratch vector...
		 */
		case T_SS_S_CONDASS:		/* eval_obj_assignment */
			{
				index_t index;
				//Scalar_Value sval;

				/* I don't get this AT ALL??? */
				index = (index_t) EVAL_INT_EXP(VN_CHILD(enp,0));

				index = index!=0 ? 1 : 2;

//#ifdef CAUTIOUS
//				if( ! SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,index))) ){
//					NODE_ERROR(enp);
//					sprintf(ERROR_STRING,
//				"CAUTIOUS:  eval_obj_assignment %s:  %s does not have scalar shape!?",
//						node_desc(enp),node_desc(VN_CHILD(enp,index)));
//					ERROR1(ERROR_STRING);
//					IOS_RETURN
//				}
//#endif /* CAUTIOUS */
				assert( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,index))) );

				EVAL_SCALAR(&sval,VN_CHILD(enp,index),OBJ_PREC_PTR(dp));
				ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			}
			break;
		case T_VS_S_CONDASS:		/* eval_obj_assignment */
			{
				index_t index;
				//Scalar_Value sval;

				/* is a boolean expression and int expression? */
				index = (index_t) EVAL_INT_EXP(VN_CHILD(enp,0));

				index = index!=0 ? 1 : 2;

				if( index == 1 ){	/* first choice should be the vector */
//#ifdef CAUTIOUS
//					if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,index))) ){
//						NODE_ERROR(enp);
//						sprintf(ERROR_STRING,
//				"CAUTIOUS:  eval_obj_assignment %s:  %s does not have vector shape!?",
//				node_desc(enp),node_desc(VN_CHILD(enp,index)));
//						ERROR1(ERROR_STRING);
//						IOS_RETURN
//					}
//#endif /* CAUTIOUS */
					assert( ! SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,index))) );

					EVAL_OBJ_ASSIGNMENT(dp,VN_CHILD(enp,index));
				} else {		/* second choice should be the scalar */
//#ifdef CAUTIOUS
//					if( ! SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,index))) ){
//						NODE_ERROR(enp);
//						sprintf(ERROR_STRING,
//				"CAUTIOUS:  eval_obj_assignment %s:  %s does not have scalar shape!?",
//							node_desc(enp),node_desc(VN_CHILD(enp,index)));
//						ERROR1(ERROR_STRING);
//						IOS_RETURN
//					}
//#endif /* CAUTIOUS */
					assert( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,index))) );

					EVAL_SCALAR(&sval,VN_CHILD(enp,index),OBJ_PREC_PTR(dp));
					ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
				}
			}
			break;

		case T_VV_S_CONDASS:			/* eval_obj_assignment */
			{
				index_t index;

				index = (index_t) EVAL_INT_EXP(VN_CHILD(enp,0));

				index = index!=0 ? 1 : 2;

//#ifdef CAUTIOUS
//				if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,index))) ){
//					NODE_ERROR(enp);
//					sprintf(ERROR_STRING,
//				"CAUTIOUS:  eval_obj_assignment %s:  %s does not have vector shape!?",
//				node_desc(enp),node_desc(VN_CHILD(enp,index)));
//					ERROR1(ERROR_STRING);
//					IOS_RETURN
//				}
//#endif /* CAUTIOUS */
				assert( ! SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,index))) );

				EVAL_OBJ_ASSIGNMENT(dp,VN_CHILD(enp,index));
			}
			break;

		case T_SS_B_CONDASS: /* eval_obj_assignment */
			{
				Data_Obj *bm_dp;
				Scalar_Value sval2;

				/* Neet to create a temp vector or bitmap,
				 * and then use the select vector function.
				 */

				bm_dp = EVAL_BITMAP(NO_OBJ,VN_CHILD(enp,0));
				/* we need to know the type of the destination before
				 * we evaluate the scalars...
				 */
				EVAL_SCALAR(&sval,VN_CHILD(enp,1),OBJ_PREC_PTR(dp));
				EVAL_SCALAR(&sval2,VN_CHILD(enp,2),OBJ_PREC_PTR(dp));

				setvarg1(oap,dp);

				SET_OA_SVAL(oap,0, &sval);
				SET_OA_SVAL(oap,1, &sval2);

				SET_OA_SBM(oap,bm_dp);
				if( perf_vfunc(QSP_ARG  FVSSSLCT,oap) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VSS select operator");
				}
			}
			break;

		case T_VS_B_CONDASS:		/* eval_obj_assignment */
			{
				Data_Obj *bm_dp;

				bm_dp = EVAL_BITMAP(NO_OBJ,VN_CHILD(enp,0));
				dp2=EVAL_OBJ_EXP(VN_CHILD(enp,1),NO_OBJ);
				EVAL_SCALAR(&sval,VN_CHILD(enp,2),OBJ_PREC_PTR(dp));

				setvarg2(oap,dp,dp2);
				SET_OA_SVAL(oap,0, &sval);
				SET_OA_SBM(oap,bm_dp);

				if( perf_vfunc(QSP_ARG  FVVSSLCT,oap) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VVS select operator");
				}
			}
			break;
		case T_VV_B_CONDASS:		/* eval_obj_assignment */
			{
				Data_Obj *bm_dp;
				bm_dp = EVAL_BITMAP(NO_OBJ,VN_CHILD(enp,0));
				dp2=EVAL_OBJ_EXP(VN_CHILD(enp,1),NO_OBJ);
				dp3=EVAL_OBJ_EXP(VN_CHILD(enp,2),NO_OBJ);

				setvarg3(oap,dp,dp2,dp3);
				SET_OA_SBM(oap,bm_dp);
				if( perf_vfunc(QSP_ARG  FVVVSLCT,oap) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VVV select operator");
				}
			}
			break;

		case T_VV_VV_CONDASS:		/* eval_obj_assignment */
			{
				dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
				dp2=EVAL_OBJ_EXP(VN_CHILD(enp,1),NO_OBJ);
				dp3=EVAL_OBJ_EXP(VN_CHILD(enp,2),NO_OBJ);
				dp4=EVAL_OBJ_EXP(VN_CHILD(enp,3),NO_OBJ);
				setvarg5(oap,dp,dp1,dp2,dp3,dp4);
				if( perf_vfunc(QSP_ARG  VN_BM_CODE(enp), oap) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VV_VV conditional");
				}
			}
			break;
		case T_VV_VS_CONDASS:		/* eval_obj_assignment */
			{
				dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
				dp2=EVAL_OBJ_EXP(VN_CHILD(enp,1),NO_OBJ);
				dp3=EVAL_OBJ_EXP(VN_CHILD(enp,2),NO_OBJ);
				EVAL_SCALAR(&sval,VN_CHILD(enp,3),OBJ_MACH_PREC_PTR(dp3));
				setvarg4(oap,dp,dp1,dp2,dp3);
				SET_OA_SVAL(oap,0, &sval);
				if( perf_vfunc(QSP_ARG  VN_BM_CODE(enp), oap) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VV_VS conditional");
				}
			}
			break;
		case T_VS_VV_CONDASS:		/* eval_obj_assignment */
			{
				dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
				EVAL_SCALAR(&sval,VN_CHILD(enp,1),OBJ_MACH_PREC_PTR(dp1));
				dp2=EVAL_OBJ_EXP(VN_CHILD(enp,2),NO_OBJ);
				dp3=EVAL_OBJ_EXP(VN_CHILD(enp,3),NO_OBJ);
				setvarg4(oap,dp,dp1,dp2,dp3);
				SET_OA_SVAL(oap,0, &sval);
				if( perf_vfunc(QSP_ARG  VN_BM_CODE(enp), oap) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VS_VV conditional");
				}
			}
			break;
		case T_VS_VS_CONDASS:		/* eval_obj_assignment */
			{
				Scalar_Value sval2;

				dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
				EVAL_SCALAR(&sval,VN_CHILD(enp,1),OBJ_MACH_PREC_PTR(dp1));
				dp2=EVAL_OBJ_EXP(VN_CHILD(enp,2),NO_OBJ);
				EVAL_SCALAR(&sval2,VN_CHILD(enp,3),OBJ_MACH_PREC_PTR(dp2));
				setvarg3(oap,dp,dp1,dp2);
				/* The first scalar is the source */
				SET_OA_SVAL(oap,0, &sval);
				SET_OA_SVAL(oap,1, &sval2);
				if( perf_vfunc(QSP_ARG  VN_BM_CODE(enp), oap) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VS_VS conditional");
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

			dp1=EVAL_OBJ_REF(VN_CHILD(enp,0));		/* indices */
			dp2=EVAL_OBJ_REF(VN_CHILD(enp,1));		/* maxval */
			dp3=EVAL_OBJ_EXP(VN_CHILD(enp,2),NO_OBJ);	/* input */
			setvarg2(oap,dp1,dp3);
			SET_OA_SRC1(oap,dp2);				/* destination maxval */
			SET_OA_SRC2(oap,dp);					/* destination n */
			SET_OA_SVAL(oap,0, (Scalar_Value *)OBJ_DATA_PTR(dp2));
			SET_OA_SVAL(oap,1, (Scalar_Value *)OBJ_DATA_PTR(dp));
			if( perf_vfunc(QSP_ARG  FVMAXG,oap) < 0 ){
				NODE_ERROR(enp);
				WARN("Error evaluating max_times operator");
			}
			break;

		case T_RDFT:						/* eval_obj_assignment */
			dp1 = EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			h_vl2_fft2d(VFCODE_ARG  dp,dp1);
			break;

		case T_RIDFT:						/* eval_obj_assignment */
			dp1 = EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			h_vl2_ift2d(VFCODE_ARG  dp,dp1);
			break;

		case T_REDUCE:						/* eval_obj_assignment */
			dp1 = EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			reduce(QSP_ARG  dp,dp1);
			break;

		case T_ENLARGE:						/* eval_obj_assignment */
			dp1 = EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			enlarge(QSP_ARG  dp,dp1);
			break;

		case T_TYPECAST:		/* eval_obj_assignment */
			EVAL_TYPECAST(enp,dp);
			break;

		/* use tabled functions here???
		 * Or at least write a macro for the repeated code...
		 */

		case T_MINVAL:
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			clear_obj_args(oap);
			setvarg2(oap,dp,dp1);
			//vminv(oap);
			//vf_code=FVMINV;
			//h_vl2_vminv(HOST_CALL_ARGS);
			platform_dispatch_by_code(QSP_ARG  FVMINV, oap);
			break;
		case T_MAXVAL:
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			clear_obj_args(oap);
			setvarg2(oap,dp,dp1);
			//vmaxv(oap);
			//vf_code=FVMAXV;
			//h_vl2_vmaxv(HOST_CALL_ARGS);
			platform_dispatch_by_code(QSP_ARG  FVMAXV, oap);
			break;
		case T_SUM:				/* eval_obj_assignment */
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			clear_obj_args(oap);
			setvarg2(oap,dp,dp1);
			//vsum(oap);
			//vf_code=FVSUM;
			//h_vl2_vsum(HOST_CALL_ARGS);
			platform_dispatch_by_code(QSP_ARG  FVSUM, oap);
			break;

#ifdef NOT_YET
		case T_LOAD:						/* eval_obj_assignment */
			s = EVAL_STRING(VN_CHILD(enp,0));
			if( s == NULL ) break;

			/* load image from file */
			/* Can we assume that the sizes have already
			 * been checked???
			 */

			ifp = img_file_of(QSP_ARG  s);

			/* BUG?  a lot of these checks should
			 * probably be done in scan_tree() ?
			 */
			if( ifp == NO_IMAGE_FILE ){
				ifp = read_image_file(QSP_ARG  s);
				if( ifp==NO_IMAGE_FILE ){
					NODE_ERROR(enp);
					sprintf(ERROR_STRING,
	"eval_obj_assignment LOAD/READ:  Couldn't open image file %s",s);
					WARN(ERROR_STRING);
					break;
				}
			}
			if( ! IS_READABLE(ifp) ){
				sprintf(ERROR_STRING,
		"File %s is not readable!?",s);
				WARN(ERROR_STRING);
				break;
			}

			if( OBJ_PREC(ifp->if_dp) == PREC_ANY || OBJ_PREC(dp) == OBJ_PREC(ifp->if_dp) ){
				/* no need to typecast */
				read_object_from_file(QSP_ARG  dp,ifp);
				/* BUG?? do we know the whole object is assigned? */
				/* does it matter? */
				//SET_OBJ_FLAG_BITS(dp, DT_ASSIGNED);
				// done below
				//note_assignment(dp);
			} else {
				dp1=make_local_dobj(QSP_ARG  
					OBJ_SHAPE(dp).si_type_dimset,
					OBJ_PREC_PTR(ifp->if_dp));
				read_object_from_file(QSP_ARG  dp1,ifp);
				//h_vl2_convert(QSP_ARG  dp,dp1);
				dp_convert(QSP_ARG  dp,dp1);
				delvec(QSP_ARG  dp1);
			}
			break;
#endif /* NOT_YET */

		case T_ASSIGN:		/* x=y=z; */
			dp1 = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( dp1 == NO_OBJ )
				break;
			EVAL_OBJ_ASSIGNMENT(dp1,VN_CHILD(enp,1));
			/* now copy to the target of this call */
			if( do_unfunc(QSP_ARG  dp,dp1,FVMOV) ){
				NODE_ERROR(enp);
				WARN("Error evaluating assignment");
			}
			break;


#ifdef NOT_YET
		case T_CALL_NATIVE:			/* eval_obj_assignment() */
			eval_native_assignment(dp,enp);
			break;
#endif /* NOT_YET */

		case T_INDIR_CALL:
		case T_CALLFUNC:			/* eval_obj_assignment() */
#ifdef QUIP_DEBUG
if( debug & eval_debug ){
sprintf(ERROR_STRING,"eval_obj_assignment calling exec_subrt, dst = %s",OBJ_NAME(dp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			EXEC_SUBRT(enp,dp);
			break;

		ALL_OBJREF_CASES			/* eval_obj_assignment */
			if( VN_CODE(enp) == T_LIST_OBJ || VN_CODE(enp) == T_COMP_OBJ ){
				/* should be its own case... */
				/* a list of expressions, maybe literals... */
				/* We need to do something to handle 2D arrays... */
				/* ASSIGN_OBJ_FROM_LIST(dp,VN_CHILD(enp,0),0); */

				ASSIGN_OBJ_FROM_LIST(dp,enp,0);
				//SET_OBJ_FLAG_BITS(dp, DT_ASSIGNED);
				// done below
				//note_assignment(dp);
				break;
			}

			/* dp1=EVAL_OBJ_REF(enp); */
			dp1=EVAL_OBJ_EXP(enp,dp);

			if( dp1 == NO_OBJ ){
				NODE_ERROR(enp);
				WARN("Unable to evaluate RHS");
				break;
			}

			if( executing && expect_objs_assigned && ! HAS_ALL_VALUES(dp1) ){
				unset_object_warning(QSP_ARG  enp,dp1);
			}
			if( mode_is_matlab ){
				if( OBJ_ROWS(dp1) == 1 && OBJ_ROWS(dp) > 1 ){
					dp2 = D_SUBSCRIPT(dp,1);
					//setvarg2(oap,dp,dp1);
					//h_vl2_convert(HOST_CALL_ARGS);
					dp_convert(QSP_ARG  dp,dp1);
					break;
				}
			}

			/* BUG?  is this correct if we have multiple components??? */
			if( IS_SCALAR(dp1) ){
				svp = (Scalar_Value *)OBJ_DATA_PTR(dp1);
				/* BUG type conversion? */
				ASSIGN_OBJ_FROM_SCALAR(enp,dp,svp);
			} else {
				/* object-to-object copy */
				if( dp != dp1 ){
					//setvarg2(oap,dp,dp1);
					//h_vl2_convert(HOST_CALL_ARGS);
					dp_convert(QSP_ARG  dp,dp1);
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
				dval=EVAL_FLT_EXP(VN_CHILD(enp,0));
				d2=EVAL_FLT_EXP(VN_CHILD(enp,1));
				dbl_to_scalar(&sval,dval*d2,OBJ_PREC_PTR(dp));
				ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			} else {
				/* we don't pass the dst object, because it may not
				 * be the right shape - we could check this, but we're lazy!
				 */
				GET_2_OPERANDS(enp,&dp1,&dp2,NO_OBJ);	// T_INNER
				/* This assumes that the destination is the right size;
				 * it will be wrong if the dot product is a scalar...
				 */
				inner(QSP_ARG  dp,dp1,dp2);
				//WARN("Sorry, inner is temporarily unavailable!?");
			}
			break;

		case T_DFT:			/* eval_obj_assignment */
			/* BUG if the types are difference, dp may not be
			 * an appropriate arg for eval_obj_exp()
			 */
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),dp);
			/* BUG need to handle real fft's;
			 * for now, assume cpx to cpx
			 */
			if( do_unfunc(QSP_ARG  dp,dp1,FVMOV) < 0 ){
				NODE_ERROR(enp);
				WARN("error moving data for fft");
				break;
			}
			h_vl2_fft2d(VFCODE_ARG  dp,dp);
			break;

		case T_IDFT:
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),dp);
			/* BUG need to handle real fft's;
			 * for now, assume cpx to cpx
			 */
			if( do_unfunc(QSP_ARG  dp,dp1,FVMOV) < 0 ){
				NODE_ERROR(enp);
				WARN("error moving data for ifft");
				break;
			}
			h_vl2_ift2d(VFCODE_ARG  dp,dp);
			break;

		case T_WRAP:		/* eval_obj_assignment */
			/* We can't wrap in-place, so don't pass dp
			 * to eval_obj_exp
			 */
			/* BUG?  will this catch a=wrap(a) ?? */
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
//#ifdef CAUTIOUS
//			if( dp1 == NO_OBJ ){
//				WARN("CAUTIOUS:  eval_obj_assignemnt:  missing wrap arg");
//				break;
//			}
//#endif /* CAUTIOUS */
			assert( dp1 != NO_OBJ );

			wrap(QSP_ARG  dp,dp1);
			break;

		case T_SCROLL:
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			ldx=EVAL_INT_EXP(VN_CHILD(enp,1));
			ldy=EVAL_INT_EXP(VN_CHILD(enp,2));
			dp_scroll(QSP_ARG  dp,dp1,(incr_t)ldx,(incr_t)ldy);
			break;

		/* 2 argument operations */

		case T_MATH2_VFN:		/* eval_obj_assignment */
		case T_VV_FUNC:
			GET_2_OPERANDS(enp,&dp1,&dp2,dp);	// T_VV_FUNC
			if( dp1 == NO_OBJ || dp2 == NO_OBJ ){
				NODE_ERROR(enp);
				advise("bad vector operand");
			} else
				if( do_vvfunc(QSP_ARG  dp,dp1,dp2,VN_VFUNC_CODE(enp)) < 0 ){
					NODE_ERROR(enp);
					WARN("Expression error");
dump_tree(QSP_ARG  enp);	// expression error
				}
			break;

		case T_MATH2_VSFN:
		case T_VS_FUNC:
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),dp);
			if( dp1 == NO_OBJ ){
				NODE_ERROR(enp);
				advise("vector operand does not exist");
				break;
			}
			EVAL_SCALAR(&sval,VN_CHILD(enp,1),OBJ_MACH_PREC_PTR(dp1));
			if( do_vsfunc(QSP_ARG  dp,dp1,&sval,VN_VFUNC_CODE(enp)) < 0 ){
				NODE_ERROR(enp);
				WARN("Error assigning object");
			}
			break;

		case T_TRANSPOSE:	/* eval_obj_assignment */
			/* Why did we ever think this was correct? */
			/* dp1 = get_id_obj(QSP_ARG  VN_STRING(enp),enp); */
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),NO_OBJ);
			if( dp1 == NO_OBJ ) break;
			/* BUG make sure valid */
			xpose_data(QSP_ARG  dp,dp1);
			break;

		case T_RAMP:
			start=EVAL_FLT_EXP(VN_CHILD(enp,0));
			dx=EVAL_FLT_EXP(VN_CHILD(enp,1));
			dy=EVAL_FLT_EXP(VN_CHILD(enp,2));
			easy_ramp2d(QSP_ARG  dp,start,dx,dy);
			break;

		case T_STR2_FN:	/* eval_obj_assignment */
		case T_STR1_FN:	/* eval_obj_assignment */
		case T_SIZE_FN: 	/* eval_obj_assignment */
			dval = EVAL_FLT_EXP(enp);
			dbl_to_scalar(&sval,dval,OBJ_PREC_PTR(dp));
			ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			break;

		case T_LIT_INT:				/* eval_obj_assignment */
			/* BUG? we are doing a lot of unecessary conversions
			 * if the object is integer to begin with... but this
			 * will work.
			 */
			int_to_scalar(&sval,VN_INTVAL(enp),OBJ_PREC_PTR(dp));
			ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			break;

		case T_LIT_DBL:
			dbl_to_scalar(&sval,VN_DBLVAL(enp),OBJ_PREC_PTR(dp));
			ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			break;

		case T_BITRSHIFT:
		case T_BITLSHIFT:
		case T_BITAND:
		case T_BITOR:
		case T_BITXOR:
		case T_BITCOMP:
		case T_MODULO:
			int_to_scalar( &sval, EVAL_INT_EXP(enp), OBJ_PREC_PTR(dp) );
			ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
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
			dbl_to_scalar(&sval, EVAL_FLT_EXP(enp), OBJ_PREC_PTR(dp) );
			ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			break;

		case T_MATH0_VFN:			/* eval_obj_assignment */
			/* unary math function */
			if( do_un0func(QSP_ARG  dp,VN_VFUNC_CODE(enp)) ){
				NODE_ERROR(enp);
				WARN("Error evaluating math function");
			}
			break;

		case T_INT1_VFN:			/* eval_obj_assignment */
		case T_MATH1_VFN:			/* eval_obj_assignment */
		case T_CHAR_VFN:			/* eval_obj_assignment */
			/* unary math function */
			dp1=EVAL_OBJ_EXP(VN_CHILD(enp,0),dp);
//#ifdef CAUTIOUS
//			if( dp1 == NO_OBJ ){
//				WARN("CAUTIOUS:  eval_obj_exp:  missing (math/int) vfn arg");
//				DUMP_TREE(VN_CHILD(enp,0));
//				break;
//			}
//#endif /* CAUTIOUS */
			assert( dp1 != NO_OBJ );

			if( do_unfunc(QSP_ARG  dp,dp1,VN_VFUNC_CODE(enp)) ){
				NODE_ERROR(enp);
				WARN("Error evaluating (math/int) function");
			}
			break;

		default:
			MISSING_CASE(enp,"eval_obj_assignment");
			break;
	}
/*
sprintf(ERROR_STRING,"eval_obj_assignment %s DONE!",OBJ_NAME(dp));
advise(ERROR_STRING);
LONGLIST(dp);
*/


	note_assignment(dp);
}		/* end eval_obj_assignment() */

/* We return a 1 if we should keep working.
 * We return 0 if we encounter a return statement within a subroutine.
 *
 * what is "going" ???
 */

static int eval_work_tree(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj *dst_dp)
{
	Data_Obj *dp;
	Subrt *srp;
	int intval;
#ifdef NOT_YET
	Image_File *ifp;
#endif /* NOT_YET */
	//Macro dummy_mac;
	Query *qp;
#ifdef OLD_LOOKAHEAD
	int la_level;
#endif /* OLD_LOOKAHEAD */
	const char *s;
	Identifier *idp,*idp2;
	Function_Ptr *fpp;
	int ret_val=1;			/* the default is to keep working */

	if( enp==NO_VEXPR_NODE || IS_CURDLED(enp) ) return(ret_val);

#ifdef QUIP_DEBUG
if( debug & eval_debug ){
sprintf(ERROR_STRING,"eval_work_tree (dst = %s) %s",
dst_dp==NO_OBJ?"null":OBJ_NAME(dst_dp),
node_desc(enp));
advise(ERROR_STRING);
DUMP_TREE(enp);
}
#endif /* QUIP_DEBUG */

	eval_enp = enp;
	executing = 1;
	if( interrupted ) return(0);

	/* We need to do runtime resolution, but we don't want to descend entire
	 * statment trees here...  The top node may have an unknown leaf up until the
	 * time that the last statement is executed, but this may not be resolvable until
	 * the previous statement is executed...  therefore we only try to resolve selected
	 * nodes...
	 */

	/* We may have some unresolved shapes which depend on the current values of pointers */
	if( VN_SHAPE(enp) != NO_SHAPE && UNKNOWN_SHAPE(VN_SHAPE(enp)) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_work_tree:  attemping to runtime resolution of %s",node_desc(enp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		/*
		resolve_one_uk_node(enp);
		*/
		RESOLVE_TREE(enp,NO_VEXPR_NODE);
/*
advise("eval_work_tree after runtime resolution:");
DUMP_TREE(enp);
*/
	}

	/* Where should we put this? */
	/* We want to do this at the END of each statement... */
	unlock_all_tmp_objs(SINGLE_QSP_ARG);

	/* BUG we'll do something more efficient eventually */

	/* We also need to remove the "local" objects... */

	delete_local_objs(SINGLE_QSP_ARG);

	switch(VN_CODE(enp)){

		case T_CALL_NATIVE:
			eval_native_work(QSP_ARG  enp);
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
			WARN("Sorry, matlab clear/clr not implemented yet");
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
			if( VN_CHILD(enp,0)!=NO_VEXPR_NODE )
				exit( (int) EVAL_INT_EXP(VN_CHILD(enp,0)) );
			else
				exit(0);
			break;

		case T_FIX_SIZE:
			break;

		case T_DISPLAY:		/* eval_work_tree */
			if( going ) return(1);
			EVAL_DISPLAY_STAT(VN_CHILD(enp,0));
			break;

		case T_SET_FUNCPTR:
			if( going ) return(1);
			srp = eval_funcref(QSP_ARG  VN_CHILD(enp,1));
			fpp = eval_funcptr(QSP_ARG  VN_CHILD(enp,0));
			/* BUG check for valid return values */
			fpp->fp_srp = srp;
			point_node_shape(QSP_ARG  enp,SR_SHAPE(srp));
			break;

		case T_SET_STR:		/* eval_work_tree */
			if( going ) return(1);
			s = EVAL_STRING(VN_CHILD(enp,1));
			idp = EVAL_PTR_REF(VN_CHILD(enp,0),UNSET_PTR_OK);
			if( idp == NO_IDENTIFIER ) break;
			assign_string(QSP_ARG  idp,s,enp);
			break;

		case T_SET_PTR:		/* eval_work_tree */
			if( going ) return(1);

//#ifdef CAUTIOUS
//			if( dst_dp != NO_OBJ ){
//				sprintf(ERROR_STRING,"CAUTIOUS:  eval_work_tree:  T_SET_PTR, dst_dp (%s) not null!?",
//					OBJ_NAME(dst_dp));
//				WARN(ERROR_STRING);
//			}
//#endif /* CAUTIOUS */
			assert( dst_dp == NO_OBJ );

			idp2 = EVAL_PTR_REF(VN_CHILD(enp,1),EXPECT_PTR_SET);
			idp = EVAL_PTR_REF(VN_CHILD(enp,0),UNSET_PTR_OK);

			if( idp2 == NO_IDENTIFIER || idp == NO_IDENTIFIER ){
				NODE_ERROR(enp);
				advise("eval_work_tree T_SET_PTR:  null object");
				break;
			}

//#ifdef CAUTIOUS
//			if( ! IS_POINTER(idp) ){
//				NODE_ERROR(enp);
//				sprintf(ERROR_STRING,"CAUTIOUS:  identifier %s does not refer to a pointer!?",
//					ID_NAME(idp));
//				WARN(ERROR_STRING);
//				return(0);
//			}
//#endif /* CAUTIOUS */
			assert( IS_POINTER(idp) );

			if( IS_POINTER(idp2) ){
				SET_PTR_REF(ID_PTR(idp), PTR_REF(ID_PTR(idp2)));
				SET_PTR_FLAG_BITS(ID_PTR(idp), POINTER_SET);
			} else if( IS_REFERENCE(idp2) ){
				assign_pointer(ID_PTR(idp),ID_REF(idp2));
				/* can we do some runtime shape resolution here?? */
				/* We mark the node as unknown to force propagate_shape to do something
				 * even when the ptr was previously set to something else.
				 */
				copy_node_shape( PTR_DECL_VN(ID_PTR(idp)),uk_shape(VN_PREC(VN_CHILD(enp,0))));
				if( !UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) )
					RESOLVE_POINTER(VN_CHILD(enp,0),VN_SHAPE(VN_CHILD(enp,1)));
			}
//#ifdef CAUTIOUS
			  else {
//				sprintf(ERROR_STRING,"CAUTIOUS:  eval_work_tree T_SET_PTR:  rhs is neither a pointer nor a reference!?");
//				ERROR1(ERROR_STRING);
//				IOS_RETURN_VAL(0)
				assert( AERROR("eval_work_tree:  rhs is neither ptr nor reference") );
			}
//#endif /* CAUTIOUS */

			break;

#ifdef NOT_YET
		case T_OUTPUT_FILE:		/* eval_work_tree */
			if( going ) return(1);
			s=EVAL_STRING(VN_CHILD(enp,0));
			if( s!=NULL )
				set_output_file(QSP_ARG  s);
			break;
#endif /* NOT_YET */

		case T_STRCPY:		/* eval_work_tree */
			if( going ) return(1);
			idp=EVAL_PTR_REF(VN_CHILD(enp,0),UNSET_PTR_OK);
			s=EVAL_STRING(VN_CHILD(enp,1));
			if( idp != NO_IDENTIFIER && s != NULL )
				assign_string(QSP_ARG  idp,s,enp);
			break;

		case T_STRCAT:		/* eval_work_tree */
			if( going ) return(1);
			idp=EVAL_PTR_REF(VN_CHILD(enp,0),EXPECT_PTR_SET);
			s=EVAL_STRING(VN_CHILD(enp,1));
			if( idp != NO_IDENTIFIER && s != NULL )
				cat_string(REF_SBUF(ID_REF(idp)),s);
			break;

		case T_FOR:		/* eval_work_tree */
			do {
				/* evaluate the conditional */
				if( ! going )
					intval = (int) EVAL_INT_EXP(VN_CHILD(enp,0));
				else
					intval = 1;

				if( going || intval ){
					/* execute the body */
					ret_val=EVAL_TREE(VN_CHILD(enp,1),NO_OBJ);
					if( ret_val == 0 ) return(0);
					continuing=0;
					if( going ) return(ret_val);
					ret_val=EVAL_TREE(VN_CHILD(enp,2),NO_OBJ);
					if( ret_val == 0 ) return(0);
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
				lval = EVAL_INT_EXP(VN_CHILD(enp,0));
				case_enp = FIND_CASE(VN_CHILD(enp,1),lval);
				if( case_enp == NO_VEXPR_NODE ){
					/* It is not an error for there to be no case.
					 * We might want to have this warning controlled by a flag.
					 */
					NODE_ERROR(enp);
					sprintf(ERROR_STRING,"No case for value %ld",lval);
					WARN(ERROR_STRING);
					break;
				}

//#ifdef CAUTIOUS
//				if( VN_CODE(case_enp) != T_CASE_STAT ){
//					NODE_ERROR(enp);
//					sprintf(ERROR_STRING,
//	"CAUTIOUS:  eval_work_tree:  find_case value (%s) not a case_stat node!?",
//						node_desc(case_enp));
//					ERROR1(ERROR_STRING);
//					IOS_RETURN_VAL(0)
//				}
//#endif /* CAUTIOUS */
				assert( VN_CODE(case_enp) == T_CASE_STAT );
			} else {
				/* while we are looking for a goto label,
				 * we must examine all the cases...
				 */
				case_enp=first_case(enp);
			}

try_again:
			while( case_enp!=NO_VEXPR_NODE && ! breaking ){
				ret_val=EVAL_TREE(VN_CHILD(case_enp,1),NO_OBJ);
				/* BUG This test may get performed multiple times (harmlessly) */
				if( going ){
					/* first see if the target is in one of the cases at all */
					if( goto_child(VN_CHILD(enp,1)) == NO_VEXPR_NODE ) {
						breaking=0;
						return(ret_val);
					}
				}
				if( going || ( ret_val && ! breaking ) ){
					/* this searches forward, how do we search backwards? */
					case_enp = next_case(case_enp);
				} else
					case_enp = NO_VEXPR_NODE;
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
				ret_val=EVAL_TREE(VN_CHILD(enp,0),NO_OBJ);
				if( ret_val == 0 ) return(0);
				continuing = 0;
				if( ! going ) intval = (int) EVAL_INT_EXP(VN_CHILD(enp,1));
			} while( (!going) && !intval);
			break;

		case T_DO_WHILE:		/* eval_work_tree */
			intval=0;	// quiet compiler
			do {
				ret_val=EVAL_TREE(VN_CHILD(enp,0),NO_OBJ);
				if( ret_val == 0 ) return(0);
				continuing = 0;
				if( ! going ) intval = (int) EVAL_INT_EXP(VN_CHILD(enp,1));
			} while( (!going) && intval);
			break;

		case T_WHILE:			/* eval_work_tree */
			do {
				/* evaluate the conditional */
				if( !going )
					intval = (int) EVAL_INT_EXP(VN_CHILD(enp,0));
				else	intval = 1;
				if( intval ){
					/* execute the body */
					ret_val=EVAL_TREE(VN_CHILD(enp,1),NO_OBJ);
					if( ret_val == 0 )
						return(0);
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
					intval = (int) EVAL_INT_EXP(VN_CHILD(enp,0));
				else	intval = 0;
				if( ! intval ){
					/* execute the body */
					ret_val=EVAL_TREE(VN_CHILD(enp,1),NO_OBJ);
					if( ret_val == 0 )
						return(0);
					continuing=0;
				}
				if( interrupted ) break;
				if( going ) return(1);
			} while( ! intval );
			break;

		case T_PERFORM:		/* eval_work_tree */
			if( going ) return(1);
			intval = (int) EVAL_INT_EXP(VN_CHILD(enp,0));
			NODE_ERROR(enp);
			if( intval )
				advise("enabling vector evaluation");
			else
				advise("disabling vector evaluation");

			set_perf(intval);
			break;

		case T_SCRIPT:		/* eval_work_tree */
			{
			Macro *dummy_mp;

			if( going ) return(1);
			srp = VN_SUBRT(enp);

//#ifdef CAUTIOUS
//			if( ! IS_SCRIPT(srp) ){
//				sprintf(ERROR_STRING,
//	"Subrt %s is not a script subroutine!?",SR_NAME(srp));
//				WARN(ERROR_STRING);
//				return(0);
//			}
//#endif /* CAUTIOUS */
			assert( IS_SCRIPT(srp) );

			/* Set up dummy_mac so that the interpreter will
			 * think we are in a macro...
			 */
			INIT_MACRO_PTR(dummy_mp)
			SET_MACRO_NAME(dummy_mp, SR_NAME(srp) );
			SET_MACRO_N_ARGS(dummy_mp, SR_N_ARGS(srp) );
			SET_MACRO_TEXT(dummy_mp, (char *) SR_BODY(srp) );
			SET_MACRO_FLAGS(dummy_mp, 0 ); /* disallow recursion */

			/* Any arguments to a script function
			 * will be treated like macro args...
			 */

			sprintf(msg_str,"Script func %s",SR_NAME(srp));
			//push_input_file(QSP_ARG  msg_str);

#ifdef OLD_LOOKAHEAD
	la_level=enable_lookahead(QLEVEL);
#endif /* OLD_LOOKAHEAD */
			push_text(QSP_ARG  (char *)SR_BODY(srp), msg_str);

			//qp=(&THIS_QSP->qs_query[QLEVEL]);
			qp=CURR_QRY(THIS_QSP);

			SET_QUERY_MACRO(qp, dummy_mp);

			SET_QRY_ARGS(qp, (const char **)getbuf( SR_N_ARGS(srp) * sizeof(char *) ) );

			/* BUG?  we have to make sure than we never try to assign more than sr_nargs args! */

			intval=SET_SCRIPT_ARGS(VN_CHILD(enp,0),0,qp,SR_N_ARGS(srp));
			// IN the objC implementation, the args are held in the query object as a list,
			// not an array.  This makes the recursive population a little tricker.
			// If we traverse the tree correctly, we may be able to simply add to the list
			if( intval != SR_N_ARGS(srp) ){
				sprintf(ERROR_STRING,
	"Script subrt %s should have %d args, passed %d",
					SR_NAME(srp),SR_N_ARGS(srp),intval);
				WARN(ERROR_STRING);
				/* BUG? poptext? */
				givbuf(dummy_mp);
				return(0);
			}
			/* If we pass object names to script functions by
			 * dereferencing pointers, we may end up with invisible objects
			 * whose contexts have been popped; here we restore those
			 * contexts.
			 */

			set_script_context(SINGLE_QSP_ARG);

			push_top_menu(SINGLE_QSP_ARG);	/* make sure at root menu */
			intval = QLEVEL;
			enable_stripping_quotes(SINGLE_QSP_ARG);
			while( QLEVEL >= intval ){
				// was do_cmd
				qs_do_cmd(THIS_QSP);
lookahead(SINGLE_QSP_ARG);
			}
			//popcmd(SINGLE_QSP_ARG);		/* go back */
			do_pop_menu(SINGLE_QSP_ARG);		/* go back */

			unset_script_context(SINGLE_QSP_ARG);

			givbuf(dummy_mp);
			}

			break;

#ifdef NOT_YET
		case T_SAVE:		/* eval_work_tree */
			if( going ) return(1);
			ifp=img_file_of(QSP_ARG  VN_STRING(enp));
			if( ifp == NO_IMAGE_FILE ){
advise("evaltree:  save:");
describe_shape(VN_SHAPE(VN_CHILD(enp,0)));
				ifp = write_image_file(QSP_ARG  VN_STRING(enp),
					VN_SHAPE(VN_CHILD(enp,0))->si_frames);
				if( ifp==NO_IMAGE_FILE ){
					/* permission error? */
					sprintf(ERROR_STRING,
		"Couldn't open image file %s",VN_STRING(enp));
					WARN(ERROR_STRING);
					return(0);
				}
			}
		/* BUG we'd like to allow an arbitrary expression here!? */
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( dp == NO_OBJ ) return(1);

			write_image_to_file(QSP_ARG  ifp,dp);
			break;

		case T_FILETYPE:		/* eval_work_tree */
			if( going ) return(1);
			/* BUG? scan tree should maybe fetch this? */
			intval = get_filetype_index(VN_STRING(enp));
			if( intval < 0 ) return(0);
			set_filetype(QSP_ARG  (filetype_code)intval);
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
			if( (ret_val=EVAL_WORK_TREE(VN_CHILD(enp,0),dst_dp)) ){
				if( continuing || breaking ) return(ret_val);
				ret_val=EVAL_WORK_TREE(VN_CHILD(enp,1),dst_dp);
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
				if( goto_enp != NO_VEXPR_NODE ){
					/* We don't need to pop any stack */
					EVAL_WORK_TREE(enp,NO_OBJ);
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
			EXEC_SUBRT(enp,NO_OBJ);
			break;

		case T_IFTHEN:		/* eval_work_tree */
			if( ! going ){
				intval = (int) EVAL_INT_EXP(VN_CHILD(enp,0));
				if( intval )
					return( EVAL_TREE(VN_CHILD(enp,1),dst_dp) );
				else if( VN_CHILD(enp,2) != NO_VEXPR_NODE )
					return( EVAL_TREE(VN_CHILD(enp,2),dst_dp) );
			} else {	// going
				ret_val = EVAL_TREE(VN_CHILD(enp,1),dst_dp);
				// can eval_tree change going???
				// BUG?  changed these returns from 1 to ret_val
				// to eliminate an analyzer warning, but
				// I'm not sure if that is correct???
				if( ! going ) return(ret_val); // return(1);
				if( VN_CHILD(enp,1) != NO_VEXPR_NODE )
					ret_val = EVAL_TREE(VN_CHILD(enp,1),dst_dp);
				return ret_val; // return(1);
			}
			break;

		case T_SUBRT:		/* eval_work_tree */
			if( going ) return(1);
			srp=VN_SUBRT(enp);
			/* if there are args, need to pass */
			WARN("eval_work_tree T_SUBRT - NOT calling subroutine???");
#ifdef QUIP_DEBUG
if( debug & eval_debug ){
sprintf(ERROR_STRING,"eval_work_tree:  what do we do for subrt %s!?",SR_NAME(srp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			break;

		case T_RETURN:		/* eval_work_tree */
			if( going ) return(1);
			if( VN_CHILD(enp,0) != NO_VEXPR_NODE ){
				EVAL_OBJ_ASSIGNMENT(dst_dp,VN_CHILD(enp,0));
			}
			/* If we are returning from a subroutine before the end,
			 * we have to pop it now...
			 */
			return(0);

		case T_EXP_PRINT:		/* eval_work_tree */
			if( going ) return(1);
			EVAL_PRINT_STAT(VN_CHILD(enp,0));
			prt_msg("");	/* print newline after other expressions */
			break;

		case T_INFO:		/* eval_work_tree */
			if( going ) return(1);
			EVAL_INFO_STAT(VN_CHILD(enp,0));
			break;

		case T_WARN:		/* eval_work_tree */
			if( going ) return(1);
			s=EVAL_STRING(VN_CHILD(enp,0));
			if( s != NULL ) WARN(s);
			break;

		case T_ADVISE:		/* eval_work_tree */
			if( going ) return(1);
			s=EVAL_STRING(VN_CHILD(enp,0));
			if( s != NULL ) advise(s);
			break;

		case T_END:		/* eval_work_tree */
			if( going ) return(1);
			vecexp_ing=0;
			break;

		case T_DIM_ASSIGN:	/* eval_work_tree */
		case T_ASSIGN:		/* eval_work_tree */
			if( going ) return(1);
			/* we check runtime resolution here ...
			 * In preliminary shape analysis, we leave the assign
			 * node UK if either node is; but calltime resolution
			 * proceeds incrementally, we might get the assign node
			 * or even lower?)
			 */

			if( mode_is_matlab ){
				if( UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) )
					UPDATE_TREE_SHAPE(VN_CHILD(enp,0));
				if( UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) )
					UPDATE_TREE_SHAPE(VN_CHILD(enp,1));
				if( UNKNOWN_SHAPE(VN_SHAPE(enp)) &&
						! UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) ){
					RESOLVE_TREE(enp,NO_VEXPR_NODE);
				}
			}

			if( UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,0))) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_work_tree ASSIGN:  last ditch attempt at runtime resolution of LHS %s",node_desc(VN_CHILD(enp,0)));
advise(ERROR_STRING);
DUMP_TREE(enp);
}
#endif /* QUIP_DEBUG */
				/*
				resolve_one_uk_node(VN_CHILD(enp,0));
				*/
				RESOLVE_TREE(VN_CHILD(enp,0),NO_VEXPR_NODE);

#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_work_tree ASSIGN:  after last ditch attempt at runtime resolution of LHS %s:",node_desc(VN_CHILD(enp,0)));
advise(ERROR_STRING);
DUMP_TREE(enp);
}
#endif /* QUIP_DEBUG */
			}
			if( UNKNOWN_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) ){
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_work_tree ASSIGN:  last ditch attempt at runtime resolution of RHS %s",node_desc(VN_CHILD(enp,1)));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
				/*
				resolve_one_uk_node(VN_CHILD(enp,1));
				*/
				RESOLVE_TREE(VN_CHILD(enp,1),NO_VEXPR_NODE);
#ifdef QUIP_DEBUG
if( debug & resolve_debug ){
sprintf(ERROR_STRING,"eval_work_tree ASSIGN:  after last ditch attempt at runtime resolution of RHS %s:",node_desc(VN_CHILD(enp,1)));
advise(ERROR_STRING);
DUMP_TREE(enp);
}
#endif /* QUIP_DEBUG */
			}

			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			if( dp == NO_OBJ ){
				NODE_ERROR(enp);
				WARN("Invalid LHS");
				break;
			}

			if( mode_is_matlab ){
				dp=MLAB_LHS(dp,enp);
//#ifdef CAUTIOUS
//if( dp == NO_OBJ ){
//sprintf(ERROR_STRING,"CAUTIOUS:  mlab_lhs returned a null ptr!?");
//WARN(ERROR_STRING);
//break;
//}
//#endif /* CAUTIOUS */
				assert( dp != NO_OBJ );

				EVAL_OBJ_ASSIGNMENT(dp,VN_CHILD(enp,1));
				break;
			}

#ifdef QUIP_DEBUG
if( debug & eval_debug ){
sprintf(ERROR_STRING,"eval_work_tree:  calling eval_obj_assignment for target %s",OBJ_NAME(dp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

			if( VN_CODE(enp) == T_DIM_ASSIGN )
				EVAL_DIM_ASSIGNMENT(dp,VN_CHILD(enp,1));
			else
				EVAL_OBJ_ASSIGNMENT(dp,VN_CHILD(enp,1));
			break;

		case T_PREINC:		/* eval_work_tree */
		case T_POSTINC:		/* eval_work_tree */
			if( going ) return(1);
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			inc_obj(dp);
			break;

		case T_POSTDEC:
		case T_PREDEC:
			if( going ) return(1);
			dp = EVAL_OBJ_REF(VN_CHILD(enp,0));
			dec_obj(dp);
			break;



		default:		/* eval_work_tree */
			MISSING_CASE(enp,"eval_work_tree");
			break;
	}
	return(ret_val);
} /* end eval_work_tree */

