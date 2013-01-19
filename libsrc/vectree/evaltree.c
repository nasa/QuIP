/*#define DEBUG_ONLY */

#include "quip_config.h"

char VersionId_vectree_evaltree[] = QUIP_VERSION_STRING;

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
#include "query.h"
#include "macros.h"
#include "chewtext.h"
#include "vec_util.h"		/* dilate, erode */
#include "fio_api.h"
#include "filetype.h"
#include "fileck.h"		/* file exists */
//#include "bi_cmds.h"		/* set_output_file */

#include "vectree.h"
#include "vt_native.h"

#include "nvf_api.h"

#include "mlab.h"

#define MAX_HIDDEN_CONTEXTS	32

/* BUG use of this global list make this not reentrant... */
static List *local_obj_lp=NO_LIST;
static void delete_local_objs(SINGLE_QSP_ARG_DECL);

Subrt *curr_srp=NO_SUBRT;
int scanning_args=0;
static Vec_Expr_Node *iteration_enp = NO_VEXPR_NODE;
static Vec_Expr_Node *eval_enp=NO_VEXPR_NODE;
static const char *goto_label;
#ifdef FOOBAR
void intr_evaluation(int arg);
#endif /* FOOBAR */

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

#ifdef DEBUG
debug_flag_t eval_debug=0;
debug_flag_t scope_debug=0;
#endif /* DEBUG */

/* local prototypes */


#ifdef DEBUG_ONLY
static void show_ptr(Pointer *ptrp);
static void show_ref(Reference *refp);
#endif /* DEBUG_ONLY */

static Data_Obj *map_subscripts(QSP_ARG_DECL  Data_Obj *src_dp, Data_Obj *index_dp, Vec_Expr_Node *enp );
#define MAP_SUBSCRIPTS(src_dp,index_dp,enp)		map_subscripts(QSP_ARG  src_dp,index_dp,enp)
static void map_iteration(QSP_ARG_DECL  Data_Obj *dst_dp,index_t i_dst, Data_Obj *index_dp, index_t i_index);
static int assign_obj_from_scalar(QSP_ARG_DECL  Vec_Expr_Node *enp,Data_Obj *dp,Scalar_Value *svp);
#define ASSIGN_OBJ_FROM_SCALAR(enp,dp,svp)		assign_obj_from_scalar(QSP_ARG  enp,dp,svp)

static const char *eval_mixed_list(QSP_ARG_DECL Vec_Expr_Node *enp);
#define EVAL_MIXED_LIST(enp)			eval_mixed_list(QSP_ARG  enp)
static Data_Obj *eval_typecast(QSP_ARG_DECL Vec_Expr_Node *enp, Data_Obj *dst_dp);
#define EVAL_TYPECAST(enp,dst_dp)		eval_typecast(QSP_ARG  enp,dst_dp)
static Data_Obj *eval_bitmap(QSP_ARG_DECL Data_Obj *dst_dp, Vec_Expr_Node *enp);
#define EVAL_BITMAP(dst_dp,enp)		eval_bitmap(QSP_ARG  dst_dp,enp)
static Data_Obj *eval_subscript1(QSP_ARG_DECL Data_Obj *dp, Vec_Expr_Node *enp);
#define EVAL_SUBSCRIPT1(dp,enp)		eval_subscript1(QSP_ARG  dp,enp)
static Identifier * exec_reffunc(QSP_ARG_DECL Vec_Expr_Node *enp);
#define EXEC_REFFUNC(enp)	exec_reffunc(QSP_ARG  enp)

static Vec_Expr_Node *find_case(QSP_ARG_DECL Vec_Expr_Node *enp,long lval);
#define FIND_CASE(enp,lval)	find_case(QSP_ARG  enp,lval)
static int eval_work_tree(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj *dst_dp);
#define EVAL_WORK_TREE(enp,dst_dp)		eval_work_tree(QSP_ARG  enp,dst_dp)
static Data_Obj *create_list_lhs(QSP_ARG_DECL Vec_Expr_Node *enp);
#define CREATE_LIST_LHS(enp)			create_list_lhs(QSP_ARG  enp)
/*static void exec_mlab_cmd(int code); */
static Data_Obj *create_matrix(QSP_ARG_DECL Vec_Expr_Node *enp,Shape_Info *shpp);
#define CREATE_MATRIX(enp,shpp)			create_matrix(QSP_ARG  enp,shpp)
static void assign_row(QSP_ARG_DECL Data_Obj *dp,dimension_t index,Vec_Expr_Node *enp);
#define ASSIGN_ROW(dp,index,enp)		assign_row(QSP_ARG  dp,index,enp)
static void assign_element(QSP_ARG_DECL Data_Obj *dp,dimension_t ri,dimension_t ci,Vec_Expr_Node *enp);
#define ASSIGN_ELEMENT(dp,ri,ci,enp)		assign_element(QSP_ARG  dp,ri,ci,enp)
static Data_Obj * mlab_lhs(QSP_ARG_DECL Data_Obj *dp, Vec_Expr_Node *enp);
#define MLAB_LHS(dp,enp)			mlab_lhs(QSP_ARG  dp,enp)
static Data_Obj *mlab_target(QSP_ARG_DECL Data_Obj *dp, Vec_Expr_Node *enp);
#define MLAB_TARGET(dp,enp)			mlab_target(QSP_ARG  dp,enp)
static Identifier * eval_obj_id(QSP_ARG_DECL Vec_Expr_Node *enp);
#define EVAL_OBJ_ID(enp)		eval_obj_id(QSP_ARG  enp)

static void eval_ref_tree(QSP_ARG_DECL Vec_Expr_Node *enp,Identifier *dst_idp);
#define EVAL_REF_TREE(enp,dst_idp)		eval_ref_tree(QSP_ARG  enp,dst_idp)
static void run_reffunc(QSP_ARG_DECL Subrt *srp, Vec_Expr_Node *enp, Identifier *dst_idp);
#define RUN_REFFUNC(srp,enp,dst_idp)		run_reffunc(QSP_ARG  srp,enp,dst_idp)
static void inc_obj(Data_Obj *dp);
static void dec_obj(Data_Obj *dp);
static void assign_string(QSP_ARG_DECL  Identifier *idp, const char *str, Vec_Expr_Node *enp);
static Identifier *ptr_for_string(QSP_ARG_DECL  const char *s,Vec_Expr_Node *enp);
static void eval_scalar(QSP_ARG_DECL Scalar_Value *svp, Vec_Expr_Node *enp, prec_t prec);
#define EVAL_SCALAR(svp,enp,prec)		eval_scalar(QSP_ARG  svp,enp,prec)
static int assign_subrt_args(QSP_ARG_DECL Vec_Expr_Node *arg_enp,Vec_Expr_Node *val_enp,Subrt *srp,Context_Pair *cpp);
#define ASSIGN_SUBRT_ARGS(arg_enp,val_enp,srp,cpp)	assign_subrt_args(QSP_ARG  arg_enp,val_enp,srp,cpp)
static int assign_ptr_arg(QSP_ARG_DECL Vec_Expr_Node *arg_enp,Vec_Expr_Node *val_enp, Context_Pair *curr_cpp,Context_Pair *prev_cpp);
#define ASSIGN_PTR_ARG(arg_enp,val_enp,curr_cpp,prev_cpp)	assign_ptr_arg(QSP_ARG  arg_enp,val_enp,curr_cpp,prev_cpp)
static Data_Obj * finish_obj_decl(QSP_ARG_DECL  Vec_Expr_Node *enp,Dimension_Set *dsp,prec_t prec,int decl_flags);

static dimension_t assign_obj_from_list( QSP_ARG_DECL  Data_Obj *dp,Vec_Expr_Node *, index_t );
#define ASSIGN_OBJ_FROM_LIST(dp,enp,offset)	assign_obj_from_list(QSP_ARG  dp,enp,offset)
static void eval_print_stat(QSP_ARG_DECL  Vec_Expr_Node *);
#define EVAL_PRINT_STAT(enp)			eval_print_stat(QSP_ARG  enp)
/* static Data_Obj *eval_obj_exp(Vec_Expr_Node *enp,Data_Obj *); */
static void eval_obj_assignment(QSP_ARG_DECL Data_Obj *,Vec_Expr_Node *enp);
#define EVAL_OBJ_ASSIGNMENT(dp,enp)		eval_obj_assignment(QSP_ARG  dp,enp)
static void eval_dim_assignment(QSP_ARG_DECL  Data_Obj *,Vec_Expr_Node *enp);
#define EVAL_DIM_ASSIGNMENT(dp,enp)		eval_dim_assignment(QSP_ARG  dp,enp)
/* static Data_Obj *eval_obj_ref(Vec_Expr_Node *enp); */
static void eval_decl_stat(QSP_ARG_DECL prec_t prec,Vec_Expr_Node *,int decl_flags);
#define EVAL_DECL_STAT(prec,enp,decl_flags)		eval_decl_stat(QSP_ARG  prec,enp,decl_flags)
static void eval_extern_decl(QSP_ARG_DECL  prec_t,Vec_Expr_Node *,int decl_flags);
#define EVAL_EXTERN_DECL(prec,enp,decl_flags)		eval_extern_decl(QSP_ARG  prec,enp,decl_flags)

#define D_SUBSCRIPT(dp,index)		d_subscript(QSP_ARG  dp , index )
#define C_SUBSCRIPT(dp,index)		c_subscript(QSP_ARG  dp , index )

static void xpose_data(QSP_ARG_DECL  Data_Obj *,Data_Obj *);
static int do_vsfunc(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Scalar_Value * ,Vec_Func_Code code);
static int do_vvfunc(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *,Data_Obj *,Vec_Func_Code code);
static int do_unfunc(QSP_ARG_DECL  Data_Obj *,Data_Obj *,Vec_Func_Code);
static int do_un0func(QSP_ARG_DECL  Data_Obj *,Vec_Func_Code);
static Data_Obj * check_global_scalar(QSP_ARG_DECL  const char *name,Data_Obj *,Data_Obj *);

/* these are functions that ought to be in the library somewhere... */

static void easy_ramp2d(QSP_ARG_DECL  Data_Obj *dst_dp,double start,double dx,double dy);

#ifdef UNUSED
static Scalar_Value * take_inner(Data_Obj *,Data_Obj *);
#endif /* UNUSED */
static Data_Obj *dp_const(QSP_ARG_DECL  Data_Obj *,Scalar_Value *);
static Data_Obj *make_global_scalar(QSP_ARG_DECL  const char *name,prec_t prec);
static Identifier *get_arg_ptr(QSP_ARG_DECL Vec_Expr_Node *enp);
#define GET_ARG_PTR(enp)	get_arg_ptr(QSP_ARG  enp)
static int set_script_args(QSP_ARG_DECL Vec_Expr_Node *enp,int index,Query *qp,int max_args);
#define SET_SCRIPT_ARGS(enp,index,qp,max_args)		set_script_args(QSP_ARG  enp,index,qp,max_args)
static void eval_info_stat(QSP_ARG_DECL Vec_Expr_Node *enp);
#define EVAL_INFO_STAT(enp)		eval_info_stat(QSP_ARG  enp)
static void eval_display_stat(QSP_ARG_DECL Vec_Expr_Node *enp);
#define EVAL_DISPLAY_STAT(enp)		eval_display_stat(QSP_ARG  enp)
static int get_filetype_index(const char *name);
static int bad_reeval_shape(Vec_Expr_Node *enp);
static void setup_unknown_shape(QSP_ARG_DECL  Vec_Expr_Node *enp,Dimension_Set *dsp);
static void get_2_operands(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj **dpp1,
				Data_Obj **dpp2,Data_Obj *dst_dp);
#define GET_2_OPERANDS(enp,dpp1,dpp2,dst_dp)		get_2_operands(QSP_ARG  enp,dpp1,dpp2,dst_dp)



#define SUBSCR_TYPE(enp)	(enp->en_code==T_SQUARE_SUBSCR?SQUARE:CURLY)

#define max( n1 , n2 )		(n1>n2?n1:n2)

const char *(*native_string_func)(Vec_Expr_Node *)=eval_vt_native_string;
float (*native_flt_func)(Vec_Expr_Node *)=eval_vt_native_flt;
void (*native_work_func)(QSP_ARG_DECL  Vec_Expr_Node *)=eval_vt_native_work;
void (*native_assign_func)(Data_Obj *,Vec_Expr_Node *)=eval_vt_native_assignment;

void eval_native_assignment(Data_Obj *dp,Vec_Expr_Node *enp)
{
	(*native_assign_func)(dp,enp);
}

float eval_native_flt(Vec_Expr_Node *enp)
{
	return( (*native_flt_func)(enp) );
}

const char * eval_native_string(Vec_Expr_Node *enp)
{
	return( (*native_string_func)(enp) );
}

static void eval_native_work(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	(*native_work_func)(QSP_ARG enp);
}

static long get_long_scalar_value(Data_Obj *dp)
{
	Scalar_Value *svp;
	long lval=0.0;

	svp = (Scalar_Value *)dp->dt_data;
	switch(MACHINE_PREC(dp)){
		case PREC_BY:  lval = svp->u_b; break;
		case PREC_IN:  lval = svp->u_s; break;
		case PREC_DI:  lval = svp->u_l; break;
		case PREC_LI:  lval = svp->u_ll; break;
		case PREC_SP:  lval = svp->u_f; break;
		case PREC_DP:  lval = svp->u_d; break;
		case PREC_UBY:  lval = svp->u_ub; break;
		case PREC_UIN:  lval = svp->u_us; break;
		case PREC_UDI:  lval = svp->u_ul; break;
		case PREC_ULI:  lval = svp->u_ull; break;
#ifdef CAUTIOUS
		/* shut up compiler */
		case PREC_NONE:
		case N_MACHINE_PRECS:
			NWARN("CAUTIOUS:  get_long_scalar_value:  nonsense precision");
			break;
#endif /* CAUTIOUS */
	}
	return(lval);
}

static double get_dbl_scalar_value(Data_Obj *dp)
{
	Scalar_Value *svp;
	double dval=0.0;

	svp = (Scalar_Value *)dp->dt_data;
	switch(MACHINE_PREC(dp)){
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
#ifdef CAUTIOUS
		/* shut up compiler */
		case PREC_NONE:
		case N_MACHINE_PRECS:
			NWARN("CAUTIOUS:  get_dbl_scalar_value:  nonsense precision");
			break;
#endif /* CAUTIOUS */
	}
	return(dval);
}

void show_id(Identifier *idp)
{
	sprintf(msg_str,"Identifier %s at 0x%lx:  ",idp->id_name, (int_for_addr)idp);
	prt_msg_frag(msg_str);
	switch(idp->id_type){
		case ID_REFERENCE:  prt_msg("reference"); break;
		case ID_POINTER:  prt_msg("pointer"); break;
		case ID_STRING:  prt_msg("string"); break;
		default:
			prt_msg("");
			sprintf(DEFAULT_ERROR_STRING,"missing case in show_id (%d)",idp->id_type);
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
}


static void prototype_mismatch(QSP_ARG_DECL  Vec_Expr_Node *enp1,Vec_Expr_Node *enp2)
{
	NODE_ERROR(enp1);
	NWARN("declaration conflicts with earlier prototype");
	NODE_ERROR(enp2);
	advise("original prototype");
}

void assign_pointer(Pointer *ptrp, Reference *refp)
{
	ptrp->ptr_refp = refp;
	/* the pointer declaration carries around the shape of its current contents? */
	/*
	copy_node_shape(QSP_ARG  ptrp->ptr_decl_enp,&ptrp->ptr_refp->ref_dp->dt_shape);
	*/
	ptrp->ptr_flags |= POINTER_SET;
}

static void dbl_to_scalar(Scalar_Value *svp,double dblval,prec_t prec)
{
	switch( prec ){
		case PREC_BY:  svp->u_b = dblval; break;
		case PREC_IN:  svp->u_s = dblval; break;
		case PREC_DI:  svp->u_l = dblval; break;
		case PREC_SP:  svp->u_f = dblval; break;
		case PREC_DP:  svp->u_d = dblval; break;
		case PREC_UBY:  svp->u_ub = dblval; break;
		case PREC_UIN:  svp->u_us = dblval; break;
		case PREC_UDI:  svp->u_ul = dblval; break;
		case PREC_CPX:
			svp->u_fc[0] = dblval;
			svp->u_fc[1] = 0.0;
			break;

		case PREC_DBLCPX:
			svp->u_dc[0] = dblval;
			svp->u_dc[1] = 0.0;
			break;

		case PREC_QUAT:
			svp->u_fq[0] = dblval;
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
			if( dblval )
				svp->u_l = 1;
			else
				svp->u_l = 0;
			break;

		/* BUG add default case */
		default:
			sprintf(DEFAULT_ERROR_STRING,"dbl_to_scalar:  unhandled precision %s",
				name_for_prec(prec));
			NERROR1(DEFAULT_ERROR_STRING);
	}
}


static void int_to_scalar(Scalar_Value *svp,long intval,prec_t prec)
{
	switch( prec ){
		case PREC_BY:  svp->u_b = intval; break;
		case PREC_IN:  svp->u_s = intval; break;
		case PREC_DI:  svp->u_l = intval; break;
		case PREC_SP:  svp->u_f = intval; break;
		case PREC_DP:  svp->u_d = intval; break;
		case PREC_CHAR:
		case PREC_UBY:  svp->u_ub = intval; break;
		case PREC_UIN:  svp->u_us = intval; break;
		case PREC_UDI:  svp->u_ul = intval; break;
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
#ifdef CAUTIOUS
		default:
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  int_to_scalar:  unhandled target precision %s",
				name_for_prec(prec));
			NERROR1(DEFAULT_ERROR_STRING);
			break;
#endif /* CAUTIOUS */
	}
}

int zero_dp(QSP_ARG_DECL  Data_Obj *dp)
{
	Scalar_Value sval;

	switch(dp->dt_prec){
		case PREC_SP:  sval.u_f = 0.0; break;
		case PREC_DP:  sval.u_d = 0.0; break;
		case PREC_BY:  sval.u_b = 0; break;
		case PREC_IN:  sval.u_s = 0; break;
		case PREC_DI:  sval.u_l = 0; break;
		default:
			ERROR1("CAUTIOUS:  unhandled machine precision in zero_dp()");
	}
	if( dp_const(QSP_ARG  dp,&sval) == NO_OBJ ) return(-1);
	return(0);
}

static int assign_obj_from_scalar(QSP_ARG_DECL  Vec_Expr_Node *enp,Data_Obj *dp,Scalar_Value *svp)
{
	if( dp_const(QSP_ARG  dp,svp) == NO_OBJ ){
		NODE_ERROR(enp);
		sprintf(error_string,"Error assigning object %s from scalar value",dp->dt_name);
		WARN(error_string);
		return(-1);
	}
	return(0);
}

void missing_case(QSP_ARG_DECL  Vec_Expr_Node *enp,const char *func_name)
{
	NODE_ERROR(enp);
	sprintf(error_string,
		"Code %s (%d) not handled by %s switch",
		NNAME(enp),enp->en_code,func_name);
	WARN(error_string);
	DUMP_TREE(enp);
	advise("");
}

static void xpose_data(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	float *fr, *to;
	dimension_t i,j,k;

	if( dpto->dt_rows != dpfr->dt_cols ){
		sprintf(error_string,
	"xpose_data:  # destination rows object %s (%d) should match # source cols object %s (%d)",
			dpto->dt_name,dpto->dt_rows,dpfr->dt_name,dpfr->dt_cols);
		WARN(error_string);
		return;
	}
	if( dpto->dt_cols != dpfr->dt_rows ){
		sprintf(error_string,
	"xpose_data:  # destination cols object %s (%d) should match # source rows object %s (%d)",
			dpto->dt_name,dpto->dt_cols,dpfr->dt_name,dpfr->dt_rows);
		WARN(error_string);
		return;
	}
	if( dpto->dt_prec != PREC_SP ){
		WARN("Sorry, now can only transpose float objects");
		return;
	}

	/* BUG if different prec's we could do the conversion... */

	for(i=0;i<dpto->dt_rows;i++){
		fr = ((float *)dpfr->dt_data) + i*dpfr->dt_pinc;
		to = ((float *)dpto->dt_data) + i*dpto->dt_rowinc;
		for(j=0;j<dpto->dt_cols;j++){
			for(k=0;k<dpfr->dt_comps;k++)
				*(to+k*dpto->dt_cinc) = *(fr+k*dpfr->dt_cinc);
			to += dpto->dt_pinc;
			fr += dpfr->dt_rowinc;
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
static float weights[N_DIMENSIONS];
static incr_t sample_offset[32];		/* 32 = 2 ^ N_DIMENSIONS */
static double sample_weight[32];	/* 32 = 2 ^ N_DIMENSIONS */


#define GET_MAP_OFFSET(type)								\
											\
{											\
	type *ip;									\
											\
	ip = (type *)index_dp->dt_data;							\
	ip += i_index;									\
	while( i_dim >= map_source_dp->dt_mindim ){					\
		indices[i_dim]= *ip;							\
		if( indices[i_dim] > map_source_dp->dt_type_dim[i_dim] ){		\
			NODE_ERROR(iteration_enp);					\
			sprintf(error_string,						\
"map_iteration:  GET_MAP_OFFSET:  index %d is out of range for %s dimension (%d) of source object %s",		\
	indices[i_dim],dimension_name[i_dim],map_source_dp->dt_type_dim[i_dim],	\
	map_source_dp->dt_name);							\
			WARN(error_string);						\
			indices[i_dim]=0;						\
		}									\
		ip += index_dp->dt_type_inc[0];					\
		i_dim--;								\
	}										\
	offset=0;									\
	for(i=0;i<N_DIMENSIONS;i++)							\
		offset += indices[i] * map_source_dp->dt_type_inc[i];			\
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
 * i_dim starts out at source_dp->dt_maxdim...
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


#define GET_MAP_WEIGHTS(type)								\
											\
{											\
	type *ip;									\
	int ns,j;									\
											\
	/* first put in some defaults */						\
	for(i=0;i<N_DIMENSIONS;i++)							\
		lower_index[i]=upper_index[i]=0;					\
											\
	ip = (type *)index_dp->dt_data;							\
	ip += i_index;									\
	n_samples=1;									\
	/* i_dim is initialized to maxdim, so this loop is foreach index */		\
	/* ip points to the first component of the index object...  this gets applied	\
	 * to the highest dimension... (WHY)  we want the opposite for x-y pairs?	\
	 */										\
	while( i_dim >= map_source_dp->dt_mindim ){					\
		double d;								\
		d = *ip;								\
		lower_index[i_dim]= floor(d);						\
		upper_index[i_dim]= ceil(d);						\
		weights[i_dim]= upper_index[i_dim]-d;					\
		if( lower_index[i_dim] < 0 ){						\
			if( expect_perfection ){					\
				NODE_ERROR(iteration_enp);				\
				sprintf(error_string,					\
"map_iteration:  GET_MAP_WEIGHTS:  index %g (rounded to %d) out of range for %s dimension (%d) of src %s",	\
		d,lower_index[i_dim],dimension_name[i_dim],map_source_dp->dt_type_dim[i_dim],	\
		map_source_dp->dt_name);						\
				WARN(error_string);					\
			}								\
			lower_index[i_dim]=upper_index[i_dim]=0;			\
			weights[i_dim]=0.0;						\
		} else if( upper_index[i_dim] >= (incr_t) map_source_dp->dt_type_dim[i_dim] ){	\
			if( expect_perfection ){					\
				NODE_ERROR(iteration_enp);				\
				sprintf(error_string,					\
"map_iteration:  GET_MAP_WEIGHTS:  index %g (rounded to %d) out of range for %s dimension (%d) of src %s",	\
		d,upper_index[i_dim],dimension_name[i_dim],map_source_dp->dt_type_dim[i_dim],	\
		map_source_dp->dt_name);						\
				WARN(error_string);					\
			}								\
			lower_index[i_dim]=upper_index[i_dim]=map_source_dp->dt_type_dim[i_dim]-1;\
			weights[i_dim]=1.0;						\
		}									\
		ip += index_dp->dt_type_inc[0];					\
		n_samples *= 2;								\
		i_dim--;								\
	}										\
	offset=0;									\
	sample_offset[0] = 0;								\
	sample_weight[0] = 1.0;								\
											\
	ns=1;										\
	for(i=0;i<N_DIMENSIONS;i++){							\
/*sprintf(error_string,"GET_MAP_W:  i = %d     ui = %d   li = %d    w = %f",\
i,upper_index[i],lower_index[i],weights[i]);\
advise(error_string);*/\
		if( (short)i >= map_source_dp->dt_mindim && (short)i <= map_source_dp->dt_maxdim ){	\
			/* We double the number of pts to interpolate here */		\
			for(j=ns;j<2*ns;j++){						\
				sample_offset[j] = sample_offset[j-ns] +		\
					upper_index[i] * map_source_dp->dt_type_inc[i]; \
				sample_weight[j] = sample_weight[j-ns] * (1.0 - weights[i]);	\
			}								\
			for(j=0;j<ns;j++){						\
				sample_offset[j] += lower_index[i] * map_source_dp->dt_type_inc[i]; \
				sample_weight[j] *= weights[i];				\
			}								\
			ns *= 2;							\
		}									\
	}										\
/*for(i=0;i<ns;i++){\
sprintf(error_string,"%d, %d     i = %d   sample_offset[i] = %d   sample_weight[i] = %g",\
i_dst,i_index,i,sample_offset[i],sample_weight[i]);\
advise(error_string);\
}*/\
}


#define MAP_IT( type )								\
										\
	{									\
		type *srcp,*dstp;						\
										\
		srcp=(type *)map_source_dp->dt_data;				\
		srcp+=offset;							\
		dstp=(type *)dst_dp->dt_data;					\
		dstp+=i_dst;							\
		*dstp = *srcp;							\
	}

#define MAP_BILINEAR( type )							\
										\
	{									\
		type *srcp, v, *dstp;						\
		index_t i;							\
		v=0;								\
		for(i=0;i<n_samples;i++){					\
			srcp=(type *)map_source_dp->dt_data;			\
			srcp += sample_offset[i];				\
			v += (*srcp) * sample_weight[i];			\
		}								\
		dstp=(type *)dst_dp->dt_data;					\
		dstp+=i_dst;							\
		*dstp = v;							\
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
				sprintf(error_string,"map_iteration:  unhandled precision %s",		\
					name_for_prec(mp));						\
				WARN(error_string);							\
				break;
#ifdef CAUTIOUS

#define INVALID_MAP_CASES(dp)										\
													\
		/* shouldn't happen, but these are valid enum's */					\
		case PREC_NONE:										\
		case N_MACHINE_PRECS:									\
		default:										\
			sprintf(error_string,								\
		"CAUTIOUS:  map_interation:  illegal machine precision (object %s).",			\
				dp->dt_name);								\
			ERROR1(error_string);								\
			offset = 0;	/* quiet compiler */						\
			break;

#else /* ! CAUTIOUS */

#define INVALID_MAP_CASES(dp)

#endif /* ! CAUTIOUS */


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
	i_dim = map_source_dp->dt_maxdim;
	mp = MACHINE_PREC(index_dp);
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
			sprintf(error_string,"map_iteration:  index object %s, unsupported precision %s",
				index_dp->dt_name,name_for_prec(index_dp->dt_prec));
			WARN(error_string);
			return;
			break;
			*/
	}

	if( n_samples > 1 ){				/* we have to interpolate! */
		mp = MACHINE_PREC(dst_dp);
		switch( mp ){
			case PREC_SP:  MAP_BILINEAR(float); break;
			case PREC_UBY: MAP_BILINEAR(u_char) break;

			UNHANDLED_MAP_CASES
			INVALID_MAP_CASES(dst_dp)

			/*
			default:
				sprintf(error_string,"map_iteration:  unhandled precision %s",
					name_for_prec(dst_dp->dt_prec));
				WARN(error_string);
				break;
				*/
		}
	} else {
		mp = MACHINE_PREC(dst_dp);
		switch( mp ){
			case PREC_SP: MAP_IT(float); break;
			case PREC_UBY: MAP_IT(u_char) break;

			UNHANDLED_MAP_CASES
			INVALID_MAP_CASES(dst_dp)
			/*
			default:
				sprintf(error_string,"map_iteration:  unhandled precision %s",
					name_for_prec(dst_dp->dt_prec));
				WARN(error_string);
				break;
				*/
		}
	}
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
	Dimension_Set dimset={{1,1,1,1,1}};
	dimension_t i;

	if( SUBSCR_TYPE(enp) != SQUARE )
		WARN("map_subscripts:  Sorry, curly subscripts are not correctly handled...");

	/* For now, we create dst_dp to have the same dimensions as the index array... */
	dimset.ds_dimension[0] = src_dp->dt_type_dim[0];	/* copy tdim from src_dp */
	for(i=1;i<N_DIMENSIONS;i++)
		dimset.ds_dimension[i] = index_dp->dt_type_dim[i];	/* BUG need to do something better */

	dst_dp=make_local_dobj(QSP_ARG  &dimset,src_dp->dt_prec);

	if( dst_dp == NO_OBJ )
		return(dst_dp);

	/* Now check the sizes - we might like to use dp_same_size(), but we allow tdim to differ  */

	if( !dp_same_dims(QSP_ARG  dst_dp,index_dp,1,N_DIMENSIONS-1,"map_subscripts") ){
		NODE_ERROR(enp);
		sprintf(error_string,"map_subscripts:  objects %s and %s should have the same shape",
			dst_dp->dt_name,index_dp->dt_name);
		WARN(error_string);
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

	dst_dp->dt_flags |= DT_ASSIGNED;
	return(dst_dp);
} /* end map_subscripts */

static int do_vvfunc(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr1,Data_Obj *dpfr2,Vec_Func_Code code)
{
	Vec_Obj_Args oargs;
	if( code == FVMUL && COMPLEX_PRECISION(dpfr2->dt_prec) && ! COMPLEX_PRECISION(dpfr1->dt_prec) ){
		setvarg3(&oargs,dpto,dpfr2,dpfr1);
	} else {
		setvarg3(&oargs,dpto,dpfr1,dpfr2);
	}
	return( perf_vfunc(QSP_ARG  code,&oargs) );
}

static Data_Obj *make_global_scalar(QSP_ARG_DECL  const char *name,prec_t prec)
{
	Data_Obj *dp;

	set_global_ctx(SINGLE_QSP_ARG);
	dp = mk_scalar(QSP_ARG  name,prec);
	unset_global_ctx(SINGLE_QSP_ARG);
	return(dp);
}

static Data_Obj * check_global_scalar(QSP_ARG_DECL  const char *name,Data_Obj *prototype_dp,Data_Obj *dp)
{
	if( dp != NO_OBJ && dp->dt_prec != prototype_dp->dt_prec ){
		delvec(QSP_ARG  dp);
		dp=NO_OBJ;
	}

	if( dp == NO_OBJ ){
		/* We have to create this scalar in the global context,
		 * otherwise when the subroutine exits, and its context
		 * is deleted, this object will be deleted too -
		 * but our static pointer will still be dangling!?
		 */
		dp = make_global_scalar(QSP_ARG  name,prototype_dp->dt_prec);
	}

	return(dp);
}

static int do_vsfunc(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Scalar_Value *valp,Vec_Func_Code code)
{
	static Data_Obj *scal_dp=NO_OBJ;
	Vec_Obj_Args oargs;

	scal_dp = check_global_scalar(QSP_ARG  "vsfunc_scalar",dpfr,scal_dp);

	assign_scalar(QSP_ARG  scal_dp,valp);
	setvarg2(&oargs,dpto,dpfr);
	oargs.oa_s1 = scal_dp;
	oargs.oa_svp[0] = (Scalar_Value *)scal_dp->dt_data;
	return( perf_vfunc(QSP_ARG  code,&oargs) );
}

static int do_un0func(QSP_ARG_DECL Data_Obj *dpto,Vec_Func_Code code)
{
	Vec_Obj_Args oargs;
	setvarg1(&oargs,dpto);
	return( perf_vfunc(QSP_ARG  code,&oargs) );
}

static int do_unfunc(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Vec_Func_Code code)
{
	Vec_Obj_Args oargs;
	setvarg2(&oargs,dpto,dpfr);
	return( perf_vfunc(QSP_ARG  code,&oargs) );
}

/* dp_const should be used for floating point assignments... */

/*
 * dp_const - set object dp to the value indicated by svp
 *
 * The scalar value gets copied into a scalar object...
 */

static Data_Obj *dp_const(QSP_ARG_DECL  Data_Obj *dp,Scalar_Value * svp)
{
	static Data_Obj *const_dp=NO_OBJ;
	Vec_Obj_Args oargs;

	const_dp=check_global_scalar(QSP_ARG  "const_scalar",dp,const_dp);

	if( const_dp->dt_prec == PREC_BIT ){
		/* assign_scalar will only change 1 bit */
		*((bitmap_word *) const_dp->dt_data) = 0;
	}

	/* now assign the value */
	assign_scalar(QSP_ARG  const_dp,svp);

	setvarg1(&oargs,dp);
	oargs.oa_s1=const_dp;
	oargs.oa_svp[0] = (Scalar_Value *)const_dp->dt_data;
	if( perf_vfunc(QSP_ARG  FVSET,&oargs) < 0 )
		return(NO_OBJ);
	return( dp );
} /* end dp_const() */

#ifdef UNUSED
static Scalar_Value * take_inner(Data_Obj *dp1,Data_Obj *dp2)
{
	static Scalar_Value sval;

#ifdef CAUTIOUS
	if( dp1==NO_OBJ || dp2==NO_OBJ ){
		sprintf(error_string,"CAUTIOUS: take_inner: passed null arg!?");
		WARN(error_string);
		return(NULL);
	}
#endif /* CAUTIOUS */

	sprintf(error_string,"take_inner %s %s:  unimplemented",
		dp1->dt_name,dp2->dt_name);
	WARN(error_string);

	switch( MACHINE_PREC(dp1) ){
		case PREC_BY:  sval.u_b = 0; break;
		case PREC_IN:  sval.u_s = 0; break;
		case PREC_DI:  sval.u_l = 0; break;
		case PREC_SP:  sval.u_f = 0.0; break;
		case PREC_DP:  sval.u_d = 0.0; break;
		case PREC_UBY:  sval.u_ub = 0; break;
		case PREC_UIN:  sval.u_us = 0; break;
		case PREC_UDI:  sval.u_ul = 0; break;
#ifdef CAUTIOUS
		/* just to shut the compiler up */
		case PREC_NONE:
		case N_MACHINE_PRECS:
			sprintf(error_string,
				"CAUTIOUS:  take_inner:  %s has nonsense machine precision",
				dp1->dt_name);
			WARN(error_string);
			/* can't happen? */
			break;
#endif /* CAUTIOUS */

	}
	return(&sval);
}
#endif /* UNUSED */

/* Get the data object for this value node.
 * We use this routine for call-by-reference.
 */

static Identifier *get_arg_ptr(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Identifier *idp;

	eval_enp = enp;

	switch(enp->en_code){
		case T_STATIC_OBJ:		/* get_arg_ptr */
			NODE_ERROR(enp);
			sprintf(error_string,"object %s not properly referenced, try prepending &",enp->en_dp->dt_name);
			advise(error_string);
			idp = GET_ID(enp->en_dp->dt_name);
			return(idp);
			break;

		case T_DYN_OBJ:		/* get_arg_ptr */
			NODE_ERROR(enp);
			sprintf(error_string,"object %s not properly referenced, try prepending &",enp->en_string);
			advise(error_string);
			idp = GET_ID(enp->en_string);
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

#ifdef CAUTIOUS
	if( idp==NO_IDENTIFIER ){
		if( mode_is_matlab ) return(NO_OBJ);	/* not an error in matlab */
		NODE_ERROR(enp);
		sprintf(error_string,"CAUTIOUS:  missing identifier object (%s) #2!?",name);
		WARN(error_string);
		return(NO_OBJ);
	}
	if( ! IS_REFERENCE(idp) ){
		NODE_ERROR(enp);
		sprintf(error_string,"CAUTIOUS:  identifier %s is not an object!?",
			idp->id_name);
		WARN(error_string);
		return(NO_OBJ);
	}
	if( strcmp(idp->id_name,idp->id_refp->ref_dp->dt_name) ){
		sprintf(error_string,"CAUTIOUS:  get_id_obj:  identifier %s points to object %s!?",
			idp->id_name,idp->id_refp->ref_dp->dt_name);
		WARN(error_string);
	}
	{
		Data_Obj *dp;
		dp = DOBJ_OF(idp->id_name);
		if( dp == NO_OBJ ){
			NODE_ERROR(enp);
			sprintf(error_string,
		"CAUTIOUS:  get_id_obj:  object identifier %s exists but object is missing!?",
				idp->id_name);
			WARN(error_string);
show_context_stack(QSP_ARG  id_itp);
show_context_stack(QSP_ARG  dobj_itp);
list_dobjs(SINGLE_QSP_ARG);
sprintf(error_string,"object pointed to by identifier %s:",idp->id_name);
advise(error_string);
LONGLIST(idp->id_refp->ref_dp);
			return(dp);
		}
		if( dp != idp->id_refp->ref_dp ){
			sprintf(error_string,
		"CAUTIOUS:  identifier %s pointer 0x%lx does not match object %s addr 0x%lx",
				idp->id_name,(int_for_addr)idp->id_refp->ref_dp,dp->dt_name,(int_for_addr)dp);
			WARN(error_string);
		}
	}
#endif /* CAUTIOUS */

	return(idp->id_refp->ref_dp);
} /* get_id_obj */

Function_Ptr *eval_funcptr(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Function_Ptr *fpp=NO_FUNC_PTR;
	Identifier *idp;

	switch(enp->en_code){
		case T_FUNCPTR_DECL:
		case T_FUNCPTR:
			idp=ID_OF(enp->en_string);
			/* BUG chould check that type is funcptr */
			/* BUG chould check that idp is valid */
#ifdef CAUTIOUS
			if( idp == NO_IDENTIFIER ){
				sprintf(error_string,"CAUTIOUS:  eval_funcptr:  missing identifier %s",enp->en_string);
				WARN(error_string);
				DUMP_TREE(enp);
			}
#endif /* CAUTIOUS */
			fpp = idp->id_fpp;
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
	switch(enp->en_code ){
		case T_FUNCREF:
			srp=enp->en_srp;
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
sprintf(error_string,"assign_ptr_arg %s %s:  calling pop_subrt_pair",node_desc(arg_enp),node_desc(val_enp));
advise(error_string);
*/

	POP_SUBRT_CPAIR(curr_cpp,curr_srp->sr_name);
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"assign_ptr_arg:  current contexts %s, %s popped",curr_cpp->cp_id_icp->ic_name,
curr_cpp->cp_dobj_icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */

	if( prev_cpp != NO_CONTEXT_PAIR ){
/*
sprintf(error_string,"assign_ptr_arg %s %s:  restoring previous context",node_desc(arg_enp),node_desc(val_enp));
advise(error_string);
*/

		PUSH_ITEM_CONTEXT(id_itp,prev_cpp->cp_id_icp);
		PUSH_ITEM_CONTEXT(dobj_itp,prev_cpp->cp_dobj_icp);
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"assign_ptr_arg:  previous contexts %s, %s pushed",prev_cpp->cp_id_icp->ic_name,
prev_cpp->cp_dobj_icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */
	}

	src_idp = GET_ARG_PTR(val_enp);		/* what if the val_enp is a string?? */

	if( prev_cpp != NO_CONTEXT_PAIR ){
/*
sprintf(error_string,"assign_ptr_arg %s %s:  popping previous context",node_desc(arg_enp),node_desc(val_enp));
advise(error_string);
*/
		pop_item_context(QSP_ARG  id_itp);
		pop_item_context(QSP_ARG  dobj_itp);
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"assign_ptr_arg:  previous contexts %s, %s popped",prev_cpp->cp_id_icp->ic_name,
prev_cpp->cp_dobj_icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */
	}

/*
sprintf(error_string,"assign_ptr_arg %s %s:  pushing current context",node_desc(arg_enp),node_desc(val_enp));
advise(error_string);
*/
	PUSH_ITEM_CONTEXT(id_itp,curr_cpp->cp_id_icp);
	PUSH_ITEM_CONTEXT(dobj_itp,curr_cpp->cp_dobj_icp);
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"assign_ptr_arg:  current contexts %s, %s pushed",curr_cpp->cp_id_icp->ic_name,
curr_cpp->cp_dobj_icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */

	if( src_idp == NO_IDENTIFIER ){
		WARN("Missing source object!?");
		return(-1);
	}

	idp = GET_ID(arg_enp->en_string);
	if( idp==NO_IDENTIFIER ) return(-1);


	/* assign_ptr_arg */

	switch(idp->id_type){
		case ID_POINTER:
			if( IS_REFERENCE(src_idp) ){
				assign_pointer(idp->id_ptrp, src_idp->id_refp );
				/* propagate_shape? */
				return(0);
			} else if( IS_POINTER(src_idp) ){
				assign_pointer(idp->id_ptrp, src_idp->id_ptrp->ptr_refp );
				/* propagate_shape? */
				return(0);
			} else if( IS_STRING_ID(src_idp) ){
				assign_pointer(idp->id_ptrp,src_idp->id_refp);
				return(0);
			} else {
				NODE_ERROR(val_enp);
				sprintf(error_string,"argval %s is not a reference or a pointer!?",
					src_idp->id_name);
				WARN(error_string);
				return(-1);
			}
			/* NOTREACHED */
			return(-1);
		case ID_STRING:
			if( ! IS_STRING_ID(src_idp) ){
				NODE_ERROR(val_enp);
				sprintf(error_string,"argval %s is not a string!?",
					idp->id_name);
				WARN(error_string);
				return(-1);
			}
#ifdef CAUTIOUS
			if( src_idp->id_refp->ref_sbp->sb_buf == NULL ){
				NODE_ERROR(val_enp);
				sprintf(error_string,
			"CAUTIOUS:  assign_ptr_arg STRING %s:  source buffer from %s is NULL!?",
					node_desc(arg_enp),node_desc(val_enp));
				WARN(error_string);
				return(-1);
			}
#endif /* CAUTIOUS */

			copy_string(idp->id_refp->ref_sbp,src_idp->id_refp->ref_sbp->sb_buf);
			/* BUG need to set string set flag */
			return(0);
		default:
			WARN("unhandled case in assign_ptr_args");
			return(-1);
	}
	/* NOTREACHED */
	return(-1);
} /* assign_ptr_arg */


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
	Context_Pair curr_cpair;
	Function_Ptr *fpp;

	if( arg_enp==NO_VEXPR_NODE ) return(0);

	switch(arg_enp->en_code){
		case T_DECL_STAT:
			/* en_decl_prec is the type (float,short,etc) */
			stat=ASSIGN_SUBRT_ARGS(arg_enp->en_child[0],
						val_enp,srp,prev_cpp);
			return(stat);

		case T_DECL_STAT_LIST:
			/* descend the arg tree */
			/* val_enp->en_code should be T_ARGLIST */
			stat=ASSIGN_SUBRT_ARGS(arg_enp->en_child[0],
				val_enp->en_child[0],srp,prev_cpp);
			if( stat < 0 ) return(stat);

			stat=ASSIGN_SUBRT_ARGS(arg_enp->en_child[1],
				val_enp->en_child[1],srp,prev_cpp);
			return(stat);

		case T_FUNCPTR_DECL:		/* assign_subrt_args */
			/* we evaluate the argument */

			POP_SUBRT_CPAIR(&curr_cpair,curr_srp->sr_name);

			if( prev_cpp != NO_CONTEXT_PAIR ){
				PUSH_ITEM_CONTEXT(id_itp,prev_cpp->cp_id_icp);
				PUSH_ITEM_CONTEXT(dobj_itp,prev_cpp->cp_dobj_icp);
			}

			srp = eval_funcref(QSP_ARG  val_enp);

			if( prev_cpp != NO_CONTEXT_PAIR ){
				pop_item_context(QSP_ARG  id_itp);
				pop_item_context(QSP_ARG  dobj_itp);
			}

			/* Now we switch contexts back to the called subrt */

			PUSH_ITEM_CONTEXT(id_itp,curr_cpair.cp_id_icp);
			PUSH_ITEM_CONTEXT(dobj_itp,curr_cpair.cp_dobj_icp);

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
			return( ASSIGN_PTR_ARG(arg_enp,val_enp,&curr_cpair,prev_cpp) );


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

			dp = get_id_obj(QSP_ARG  arg_enp->en_string,arg_enp);

			if( dp == NO_OBJ ){
sprintf(error_string,"assign_subrt_args:  missing object %s",arg_enp->en_string);
WARN(error_string);
				return(-1);
			}

#ifdef CAUTIOUS
			if( UNKNOWN_SHAPE(&dp->dt_shape) ){
				NODE_ERROR(arg_enp);
				sprintf(error_string,
	"CAUTIOUS:  assign_subrt_args:  subrt %s, arg %s has unknown shape!?",
		srp->sr_name,dp->dt_name);
				WARN(error_string);
			}
#endif /* CAUTIOUS */

			/* Tricky point:  we need to pop the subroutine context
			 * here, in case val_enp uses names which are also
			 * some of the new subrt arguments...  if there
			 * are name overlaps, we want to be sure we use
			 * the outer ones for the assignment value!
			 */

			POP_SUBRT_CPAIR(&curr_cpair,curr_srp->sr_name);

			if( prev_cpp != NO_CONTEXT_PAIR ){
				PUSH_ITEM_CONTEXT(id_itp,prev_cpp->cp_id_icp);
				PUSH_ITEM_CONTEXT(dobj_itp,prev_cpp->cp_dobj_icp);
			}

			EVAL_OBJ_ASSIGNMENT(dp, val_enp);

			if( prev_cpp != NO_CONTEXT_PAIR ){
				Item_Context *icp;
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"assign_subrt_args T_IMG_DECL:  previous contexts %s, %s popped",prev_cpp->cp_id_icp->ic_name,
prev_cpp->cp_dobj_icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */
				icp=pop_item_context(QSP_ARG  id_itp);
				icp=pop_item_context(QSP_ARG  dobj_itp);
#ifdef CAUTIOUS
				if( icp != prev_cpp->cp_dobj_icp ){
					sprintf(error_string,
		"CAUTIOUS:  popped context %s does not match expected context %s!?",icp->ic_name,
						prev_cpp->cp_dobj_icp->ic_name);
					WARN(error_string);
				}
#endif /* CAUTIOUS */
			}

			/* restore it */
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"assign_subrt_args T_IMG_DECL:  pushing current context %s",prev_cpp->cp_dobj_icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */
			PUSH_ITEM_CONTEXT(id_itp,curr_cpair.cp_id_icp);
			PUSH_ITEM_CONTEXT(dobj_itp,curr_cpair.cp_dobj_icp);

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

	switch(enp->en_code){
		case T_CALLFUNC:
			srp=enp->en_call_srp;
			srp->sr_arg_vals = enp->en_child[0];
			break;
		case T_INDIR_CALL:
			srp = eval_funcref(QSP_ARG  enp->en_child[0]);
#ifdef CAUTIOUS
			if( srp==NO_SUBRT ){
				NODE_ERROR(enp);
				WARN("CAUTIOUS:  Missing function reference");
				return(srp);
			}
#endif /* CAUTIOUS */
			srp->sr_arg_vals = enp->en_child[1];
			break;
		default:
			MISSING_CASE(enp,"runnable_subrt");
			return(NO_SUBRT);
	}

	srp->sr_call_enp = enp; /* what is this used for??? */

	if( srp->sr_body == NO_VEXPR_NODE ){
		NODE_ERROR(enp);
		sprintf(error_string,"subroutine %s has not been defined!?",srp->sr_name);
		WARN(error_string);
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
		sprintf(error_string,"subroutine is not runnable!?");
		WARN(error_string);
		DUMP_TREE(enp);
	}
}

Identifier *make_named_reference(QSP_ARG_DECL  const char *name)
{
	Identifier *idp;

	idp = ID_OF(name);
	if( idp != NO_IDENTIFIER ) return(idp);

//sprintf(error_string,"make_named_reference:  creating id %s",name);
//advise(error_string);
	idp = new_id(QSP_ARG  name);
	idp->id_type = ID_REFERENCE;
	idp->id_refp = (Reference *)getbuf( sizeof(Reference) );
	idp->id_refp->ref_dp = NO_OBJ;
	idp->id_refp->ref_idp = idp;
	idp->id_refp->ref_typ = OBJ_REFERENCE;		/* override if string */
	idp->id_refp->ref_decl_enp = NO_VEXPR_NODE;
	return(idp);
}



/* a function that returns a pointer */

static Identifier * exec_reffunc(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Identifier *idp;
	char name[LLEN];
	Subrt *srp;

	srp = runnable_subrt(QSP_ARG  enp);
	if( srp==NO_SUBRT ) return(NO_IDENTIFIER);

	sprintf(name,"ref.%s",srp->sr_name);

	idp = make_named_reference(QSP_ARG  name);
	/* BUG set ptr_type?? */

#ifdef CAUTIOUS
	if( idp == NO_IDENTIFIER ) ERROR1("CAUTIOUS:  unable to make named identifier");
#endif /* CAUTIOUS */

	/* need to check stuff */


	if( srp != NO_SUBRT )
		RUN_REFFUNC(srp,enp,idp);

	return(idp);
}

Item_Context *hidden_context[MAX_HIDDEN_CONTEXTS];
static int n_hidden_contexts=0;

void push_hidden_context(Context_Pair *cpp)
{
	if( n_hidden_contexts >= MAX_HIDDEN_CONTEXTS )
		NERROR1("too many hidden contexts (try increasing MAX_HIDDEN_CONTEXTS)");
	hidden_context[n_hidden_contexts] = cpp->cp_dobj_icp;
	n_hidden_contexts++;
}

void pop_hidden_context()
{
#ifdef CAUTIOUS
	if( n_hidden_contexts <= 0 )
		NERROR1("CAUTIOUS:  no hidden context to pop");
#endif /* CAUTIOUS */
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
#ifdef DEBUG
if( debug & scope_debug ){
advise("pop_previous:  no current subroutine, nothing to pop");
}
#endif /* DEBUG */
		cpp = NO_CONTEXT_PAIR;
	} else {
		/* Before we go through with this, we should make sure that the context really
		 * is installed!
		 */
		cpp = (Context_Pair *)getbuf( sizeof(Context_Pair) );
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"pop_previous %s:  calling pop_subrt_cpair (context)",curr_srp->sr_name);
advise(error_string);
}
#endif /* DEBUG */
		POP_SUBRT_CPAIR(cpp,curr_srp->sr_name);
		/* we remember this context so we can use it if we call a script func */
		push_hidden_context(cpp);
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"pop_previous:  previous contexts %s, %s popped",
cpp->cp_id_icp->ic_name,
cpp->cp_dobj_icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */
	}
	return(cpp);
}

/* We call restore_previous when we return from a subroutine call to go back
 * the the original context.
 */

void restore_previous(QSP_ARG_DECL  Context_Pair *cpp)
{
	pop_hidden_context();
	PUSH_ITEM_CONTEXT(id_itp,cpp->cp_id_icp);
	PUSH_ITEM_CONTEXT(dobj_itp,cpp->cp_dobj_icp);
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"restore_previous:  previous contexts %s, %s pushed",
cpp->cp_id_icp->ic_name,
cpp->cp_dobj_icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */
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
sprintf(error_string,"setup_call %s:  calling early_calltime_resolve, dst_dp = %s",srp->sr_name,dst_dp->dt_name);
advise(error_string);
describe_shape(&dst_dp->dt_shape);
} else {
sprintf(error_string,"setup_call %s:  calling early_calltime_resolve, dst_dp = NULL",srp->sr_name);
advise(error_string);
}
*/
/* advise("setup_call calling early_calltime_resolve"); */
	EARLY_CALLTIME_RESOLVE(srp,dst_dp);
/* advise("setup_call back from early_calltime_resolve"); */

/*
advise("setup_call:  after early_calltime_resolve:");
DUMP_TREE(srp->sr_body);
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
sprintf(error_string,"setup_call %s:  calling pop_previous #1 (context)",srp->sr_name);
advise(error_string);
*/
/* advise("setup_call calling check_arg_shapes"); */
	if( CHECK_ARG_SHAPES(srp->sr_arg_decls,srp->sr_arg_vals,srp) < 0 )
		goto call_err;

	/* declare the arg variables */

	/* First, pop the context of the previous subroutine and push the new one */
	rip->ri_prev_cpp = POP_PREVIOUS();	/* what does pop_previous() do??? */
	set_subrt_ctx(QSP_ARG  srp->sr_name);

	EVAL_DECL_TREE(srp->sr_arg_decls);

	rip->ri_old_srp = curr_srp;
	curr_srp = srp;

/* advise("setup_call calling assign_subrt_args"); */
	rip->ri_arg_stat = ASSIGN_SUBRT_ARGS(srp->sr_arg_decls,srp->sr_arg_vals,srp,rip->ri_prev_cpp);

	return(rip);

call_err:

	/* now we're back , restore the context of the caller , if any */
	if( rip->ri_prev_cpp != NO_CONTEXT_PAIR ){
/*
sprintf(error_string,"setup_call %s:  restoring previous context",srp->sr_name);
advise(error_string);
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
sprintf(error_string,"wrapup_context %s:  calling delete_subrt_ctx",rip->ri_srp->sr_name);
advise(error_string);
*/
	/* get rid of the context, restore the context of the caller , if any */

	delete_subrt_ctx(QSP_ARG  rip->ri_srp->sr_name);
	if( rip->ri_prev_cpp != NO_CONTEXT_PAIR ){
/*
sprintf(error_string,"wrapup_context %s:  restoring previous context",rip->ri_srp->sr_name);
advise(error_string);
*/
		RESTORE_PREVIOUS(rip->ri_prev_cpp);
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


/* eval_ref_tree - what is this used for??? */

static void eval_ref_tree(QSP_ARG_DECL Vec_Expr_Node *enp,Identifier *dst_idp)
{
	Identifier *idp;

	switch(enp->en_code){
		case T_STAT_LIST:
			EVAL_REF_TREE(enp->en_child[0],dst_idp);
			EVAL_REF_TREE(enp->en_child[1],dst_idp);
			break;
		case T_ASSIGN:				/* eval_ref_tree */
			if( EVAL_WORK_TREE(enp,NO_OBJ) == 0 )
				WARN("CAUTIOUS:  eval_ref_tree:  eval_work_tree returned 0!?");
			break;
		case T_RETURN:	/* return a pointer */
			idp = EVAL_PTR_REF(enp->en_child[0],1);
#ifdef CAUTIOUS
			if( idp == NO_IDENTIFIER ){
				NODE_ERROR(enp);
				WARN("CAUTIOUS:  missing reference");
				break;
			}
			if( ! IS_REFERENCE(idp) ){
				sprintf(error_string,"CAUTIOUS:  eval_ref_tree:  return val is not a reference");
				ERROR1(error_string);
			}
#endif /* CAUTIOUS */
			/* now copy over the identifier data */
			dst_idp->id_ptrp->ptr_refp = idp->id_refp;
			/* BUG? keep flags? */
			/*
			dst_idp->id_ptrp->ptr_flags = idp->id_ptrp->ptr_flags;
			*/
			break;
		default:
			MISSING_CASE(enp,"eval_ref_tree");
			break;
	}
}


static void run_reffunc(QSP_ARG_DECL Subrt *srp, Vec_Expr_Node *enp, Identifier *dst_idp)
{
	Run_Info *rip;

	executing=1;
	/* Run-time resolution of unknown shapes */

/*
sprintf(error_string,"run_reffunc %s:  calling setup_call",srp->sr_name);
advise(error_string);
*/
	rip = SETUP_CALL(srp,NO_OBJ);
	if( rip == NO_RUN_INFO ){
sprintf(error_string,"run_reffunc %s:  no return info!?",srp->sr_name);
WARN(error_string);
		return;
	}

	if( rip->ri_arg_stat >= 0 ){
		EVAL_DECL_TREE(srp->sr_body);
		EVAL_REF_TREE(srp->sr_body,dst_idp);
	}

	wrapup_call(QSP_ARG  rip);
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
		EVAL_DECL_TREE(srp->sr_body);
		/* eval_work_tree returns 0 if a return statement was executed,
		 * but not if there is an implied return.
		 *
		 * Uh, what is an "implied" return???
		 */
		EVAL_WORK_TREE(srp->sr_body,dst_dp);
	} else {
sprintf(error_string,"run_subrt %s:  arg_stat = %d",srp->sr_name,rip->ri_arg_stat);
WARN(error_string);
	}

	wrapup_call(QSP_ARG  rip);
}

/* returns a child LABEL node whose name matches global goto_label,
 * or NO_VEXPR_NODE if not found...
 */

static Vec_Expr_Node *goto_child(Vec_Expr_Node *enp)
{
	int i;

	for(i=0;i<MAX_CHILDREN(enp);i++){
		if( enp->en_child[i] != NO_VEXPR_NODE ){
			if( enp->en_child[i]->en_code == T_LABEL && !strcmp(enp->en_child[i]->en_string,goto_label) ){
				return(enp->en_child[i]);
			}
			else if( goto_child(enp->en_child[i]) != NO_VEXPR_NODE )
				return(enp->en_child[i]);
		}
	}
	return(NO_VEXPR_NODE);
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
	if( signal(SIGINT,intr_evaluation) == SIG_ERR )
		ERROR1("error setting evaluation interrupt handler");
		*/
#endif /* FOOBAR */

	if( going ) return(EVAL_WORK_TREE(enp,dst_dp));

	switch(enp->en_code){
		case T_EXIT:
			{
			int status;
			status = EVAL_INT_EXP( enp->en_child[0] );
			exit(status);
			}

		case T_STAT_LIST:			/* eval_tree */
			/* used to call eval_tree on children here - WHY? */
			ret_val=EVAL_WORK_TREE(enp,dst_dp);
			break;

		case T_GO_FWD:  case T_GO_BACK:		/* eval_tree */
sprintf(error_string,"eval_tree GOTO, dst_dp = %s",
dst_dp==NULL?"null":dst_dp->dt_name);
advise(error_string);
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
			EVAL_DECL_STAT(enp->en_decl_prec,enp->en_child[0],enp->en_decl_flags);
			break;
		case T_EXTERN_DECL:
			EVAL_EXTERN_DECL(enp->en_decl_prec,enp->en_child[0],enp->en_decl_flags);
			break;
		default:
			MISSING_CASE(enp,"eval_tree");
			ret_val = EVAL_WORK_TREE(enp,dst_dp);
			break;
	}
	return(ret_val);
}

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

/* Traverse a string list tree, setting the query args starting
 * at index.  Returns the number of leaves.
 *
 * When this was first written, the query args were a fixed-sized table q_arg;
 * But now they are a dynamically allocated array of variable size, renamed q_args.
 * Normally these are allocated when a macro is invoked and pushed onto the query
 * stack.  Here we are pushing a script function on the query stack.
 */

#define STORE_ARG( s )							\
									\
	if( index < max_args )						\
		qp->q_args[index]=savestr( s );				\
	else {								\
		sprintf(error_string,"set_script_args:  can't assign arg %d (max %d)",\
			index+1,max_args);				\
		WARN(error_string);					\
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

	switch(enp->en_code){
		case T_DEREFERENCE:				/* set_script_args */
			dp = EVAL_OBJ_REF(enp);
			if( dp==NO_OBJ ){
				NODE_ERROR(enp);
				WARN("missing script arg object");
				return(0);
			} else {
				STORE_ARG(dp->dt_name)
				return(1);
			}

		case T_STR_PTR:
			s=EVAL_STRING(enp);
			STORE_ARG(s)
			return(1);

		case T_POINTER:
			/* do we dereference the pointer??? */
			NODE_ERROR(enp);
			sprintf(error_string,
				"set_script_args:  not sure whether or not to dereference ptr %s",
				enp->en_string);
			advise(error_string);
			/* fall-thru */

		case T_STATIC_OBJ:		/* set_script_args */
		case T_DYN_OBJ:			/* set_script_args */
			/* maybe we could check the node shape instead of looking up the object? */
			dp=EVAL_OBJ_REF(enp);
			if( IS_SCALAR(dp) ){
				format_scalar_obj(buf,dp,dp->dt_data);
				STORE_ARG( savestr(buf) )
				return(1);
			}
			/* else fall-thru */
		case T_STRING:
			/* add this string as one of the args */
			STORE_ARG( savestr(enp->en_string) )
			return(1);

		case T_PRINT_LIST:
		case T_STRING_LIST:
		case T_MIXED_LIST:
			n1=SET_SCRIPT_ARGS(enp->en_child[0],index,qp,max_args);
			n2=SET_SCRIPT_ARGS(enp->en_child[1],index+n1,qp,max_args);
			return(n1+n2);

		/* BUG there are more cases that need to go here
		 * in order to handle generic expressions
		 */

		case T_LIT_INT: case T_LIT_DBL:			/* set_script_args */
		case T_PLUS: case T_MINUS: case T_TIMES: case T_DIVIDE:
			dval=EVAL_FLT_EXP(enp);
			sprintf(msg_str,"%g",dval);
			STORE_ARG( savestr(msg_str) )
			return(1);

		default:
#ifdef CAUTIOUS
			MISSING_CASE(enp,"set_script_args");
#endif /* CAUTIOUS */
			break;
	}
	return(0);
} /* end set_script_args */

static void eval_info_stat(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp;

	eval_enp = enp;

	switch(enp->en_code){
		case T_UNDEF:
			break;

		case T_EXPR_LIST:
			EVAL_INFO_STAT(enp->en_child[0]);
			EVAL_INFO_STAT(enp->en_child[1]);
			break;

		case T_DEREFERENCE:			/* eval_info_stat */
		case T_STATIC_OBJ:	/* eval_info_stat */
		case T_DYN_OBJ:		/* eval_info_stat */
		case T_POINTER:
		case T_STR_PTR:
		case T_SUBVEC:
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

static void eval_display_stat(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp;
	const char *s;

	eval_enp = enp;

	switch(enp->en_code){
		case T_EXPR_LIST:
			EVAL_DISPLAY_STAT(enp->en_child[0]);
			EVAL_DISPLAY_STAT(enp->en_child[1]);
			break;
		case T_STR_PTR:
			s = EVAL_STRING(enp);
			sprintf(msg_str,"String %s:  \"%s\"",enp->en_string,s);
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
			if( dp==NO_OBJ )
				WARN("missing info object");
			else {
				listone(dp);
				/* set_output_file */
				/* pntvec(dp,stdout); */
				pntvec(QSP_ARG  dp, tell_msgfile() );
			}
			break;
		default:
			MISSING_CASE(enp,"eval_display_stat");
			break;
	}
}

/* I would have put FOOBAR here but we might need this for future debug */
#ifdef DEBUG_ONLY

static void show_ref(Reference *refp)
{
	advise("show_ref:");
	if( refp->ref_typ == OBJ_REFERENCE ){
		sprintf(error_string,
			"show_ref:  ref at 0x%lx:  object %s",
			(int_for_addr)refp, refp->ref_dp->dt_name);
		advise(error_string);
	} else if( refp->ref_typ == STR_REFERENCE ){
		sprintf(error_string,"show_ref:  string");
		advise(error_string);
	} else {
		sprintf(error_string,"show_ref:  unexpected ref type %d",
			refp->ref_typ);
		WARN(error_string);
	}
}

static void show_ptr(Pointer *ptrp)
{
	sprintf(error_string,"Pointer at 0x%lx",(int_for_addr)ptrp);
	advise(error_string);
}

#endif /* DEBUG_ONLY */

static Identifier *ptr_for_string(QSP_ARG_DECL  const char *s,Vec_Expr_Node *enp)
{
	static int n_auto_strs=1;
	char idname[LLEN];
	Identifier *idp;

	/* We need to make an object and a reference... */

	sprintf(idname,"Lstr.%d",n_auto_strs++);
	idp = new_id(QSP_ARG  idname);
sprintf(error_string,"ptr_for_string:  creating id %s",idname);
advise(error_string);
	idp->id_type = ID_STRING;
#ifdef FOOBAR
	//idp->id_sbp = getbuf(sizeof(String_Buf));
	//idp->id_sbp->sb_buf = NULL;
	//idp->id_sbp->sb_size = 0;
#endif /* FOOBAR */

	/* Can't do this, because refp is in a union w/ sbp... */
	idp->id_refp = (Reference *)getbuf( sizeof(Reference) );
	idp->id_refp->ref_typ = STR_REFERENCE;
	idp->id_refp->ref_idp = idp;
	idp->id_refp->ref_decl_enp = NO_VEXPR_NODE;
	/* idp->id_refp->ref_dp = NO_OBJ; */
	idp->id_refp->ref_sbp = (String_Buf *)getbuf(sizeof(String_Buf));
	idp->id_refp->ref_sbp->sb_buf = NULL;
	idp->id_refp->ref_sbp->sb_size = 0;

	assign_string(QSP_ARG  idp,s,enp);

	return( idp );
}

static void assign_string(QSP_ARG_DECL  Identifier *idp, const char *str, Vec_Expr_Node *enp)
{
	if( ! IS_STRING_ID(idp) ){
		NODE_ERROR(enp);
		sprintf(DEFAULT_ERROR_STRING,"assign_string:  identifier %s (type %d) does not refer to a string",
			idp->id_name,idp->id_type);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	/* copy_string(idp->id_sbp,str); */
	copy_string(idp->id_refp->ref_sbp,str);
}

static Vec_Expr_Node *find_case(QSP_ARG_DECL Vec_Expr_Node *enp,long lval)
{
	Vec_Expr_Node *ret_enp;
	long cval;

	switch(enp->en_code){
		case T_CASE_STAT:	/* case_list stat_list pair */
			if( FIND_CASE(enp->en_child[0],lval) != NO_VEXPR_NODE )
				return(enp);
			else return(NO_VEXPR_NODE);

		case T_CASE_LIST:
			ret_enp=FIND_CASE(enp->en_child[0],lval);
			if( ret_enp == NO_VEXPR_NODE )
				ret_enp=FIND_CASE(enp->en_child[1],lval);
			return(ret_enp);

		case T_CASE:
			cval = EVAL_INT_EXP(enp->en_child[0]);
			if( cval == lval ){
				return(enp->en_child[0]);
			} else return(NO_VEXPR_NODE);

		case T_DEFAULT:
			return(enp);

		case T_SWITCH_LIST:	/* list of case_stat's */
			ret_enp=FIND_CASE(enp->en_child[0],lval);
			if( ret_enp == NO_VEXPR_NODE )
				ret_enp=FIND_CASE(enp->en_child[1],lval);
			return(ret_enp);

		default:
			MISSING_CASE(enp,"find_case");
			break;
	}
	return(NO_VEXPR_NODE);
}

static Vec_Expr_Node *next_case(Vec_Expr_Node *enp)
{
	if( enp->en_parent->en_code == T_SWITCH ){
		return(NO_VEXPR_NODE);
	}

#ifdef CAUTIOUS
	if( enp->en_parent ->en_code != T_SWITCH_LIST ){
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  next_case %s:  expected switch_list parent, found %s!?",
			node_desc(enp), node_desc(enp->en_parent));
		NERROR1(DEFAULT_ERROR_STRING);
	}
#endif /* CAUTIOUS */

keep_looking:
	if( enp->en_parent->en_code == T_SWITCH_LIST ){
		if( enp == enp->en_parent->en_child[0] ){
			/* descend the right branch */
			enp=enp->en_parent->en_child[1];
			while( enp->en_code == T_SWITCH_LIST )
				enp=enp->en_child[0];
#ifdef CAUTIOUS
			if( enp->en_code != T_CASE_STAT ){
				sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  next_case %s: leaf node should have been case_stat",
					node_desc(enp));
				NERROR1(DEFAULT_ERROR_STRING);
			}
#endif /* CAUTIOUS */
			return(enp);
		} else {
			/* our case is the right hand child... */
			enp = enp->en_parent;
			goto keep_looking;
		}
	}
	return(NO_VEXPR_NODE);
}

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

	icp = pop_item_context(QSP_ARG  dobj_itp);
#ifdef CAUTIOUS
	if( icp == NO_ITEM_CONTEXT )
		ERROR1("CAUTIOUS:  set_script_context:  no current dobj context!?");
#endif /* CAUTIOUS */

#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"set_script_context:  current context %s popped",icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */

	for(i=0;i<n_hidden_contexts;i++){
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"set_script_context:  pushing hidden context %s",hidden_context[i]->ic_name);
advise(error_string);
}
#endif /* DEBUG */
		PUSH_ITEM_CONTEXT(dobj_itp,hidden_context[i]);
	}
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"set_script_context:  pushing current context %s",icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */
	PUSH_ITEM_CONTEXT(dobj_itp,icp);

	set_global_ctx(SINGLE_QSP_ARG);	/* we do this so any new items created will be global */
}

static void unset_script_context(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp,*top_icp;
	int i;

	unset_global_ctx(SINGLE_QSP_ARG);

	top_icp = pop_item_context(QSP_ARG  dobj_itp);

#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"unset_script_context:  top context %s popped",top_icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */
#ifdef CAUTIOUS
	if( top_icp == NO_ITEM_CONTEXT )
		ERROR1("CAUTIOUS:  unset_script_context:  no current dobj context!?");
#endif /* CAUTIOUS */

	for(i=0;i<n_hidden_contexts;i++){
		icp = pop_item_context(QSP_ARG  dobj_itp);
#ifdef CAUTIOUS
		if( icp != hidden_context[n_hidden_contexts-(1+i)] ){
			sprintf(error_string,
"CAUTIOUS:  unset_script_context:  popped context %d %s does not match hidden stack context %s!?",
				i+1,icp->ic_name,hidden_context[n_hidden_contexts-(i+1)]->ic_name);
			WARN(error_string);
		}
#endif /* CAUTIOUS */
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"unset_script_context:  hidden context %s popped",icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */
	}

	PUSH_ITEM_CONTEXT(dobj_itp,top_icp);
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"unset_script_context:  top context %s pushed",top_icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */
}

#ifdef FOOBAR		/* not used? */
static Vec_Expr_Node *find_goto(Vec_Expr_Node *enp)
{
	Vec_Expr_Node *ret_enp;

	switch(enp->en_code){
		case T_LABEL:
			if( !strcmp(enp->en_string,goto_label) ){
				return(enp);
			}
			break;

		case T_STAT_LIST:			/* find_goto */
			ret_enp=find_goto(enp->en_child[0]);
			if( ret_enp != NO_VEXPR_NODE ) return(ret_enp);
			ret_enp=find_goto(enp->en_child[1]);
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

/* Find the first case of a switch statement.
 * used for goto scanning.
 */

static Vec_Expr_Node *first_case(Vec_Expr_Node *enp)
{
	Vec_Expr_Node *case_enp;

#ifdef CAUTIOUS
	if( enp->en_code != T_SWITCH ){
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  first_case %s:  expected switch node",node_desc(enp));
		NERROR1(DEFAULT_ERROR_STRING);
	}
#endif /* CAUTIOUS */

	case_enp = enp->en_child[1];

	while(case_enp->en_code == T_SWITCH_LIST )
		case_enp = case_enp->en_child[0];
#ifdef CAUTIOUS
	if( case_enp->en_code != T_CASE_STAT ){
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  eval_work_tree:  expected to find case_stat while searching for goto label, found %s",node_desc(case_enp));
		NERROR1(DEFAULT_ERROR_STRING);
	}
#endif /* CAUTIOUS */

	return(case_enp);
}

/* We return a 1 if we should keep working.
 * We return 0 if we encounter a return statement within a subroutine.
 */

static int eval_work_tree(QSP_ARG_DECL Vec_Expr_Node *enp,Data_Obj *dst_dp)
{
	Data_Obj *dp;
	Subrt *srp;
	int intval;
	Image_File *ifp;
	Macro dummy_mac;
	Query *qp;
#ifdef OLD_LOOKAHEAD
	int la_level;
#endif /* OLD_LOOKAHEAD */
	const char *s;
	Identifier *idp,*idp2;
	Function_Ptr *fpp;
	int ret_val=1;			/* the default is to keep working */

	if( enp==NO_VEXPR_NODE || IS_CURDLED(enp) ) return(ret_val);

#ifdef DEBUG
if( debug & eval_debug ){
sprintf(error_string,"eval_work_tree (dst = %s) %s",
dst_dp==NO_OBJ?"null":dst_dp->dt_name,
node_desc(enp));
advise(error_string);
DUMP_TREE(enp);
}
#endif /* DEBUG */

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
	if( enp->en_shpp != NO_SHAPE && UNKNOWN_SHAPE(enp->en_shpp) ){
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"eval_work_tree:  attemping to runtime resolution of %s",node_desc(enp));
advise(error_string);
}
#endif /* DEBUG */
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
	unlock_all_tmp_objs();

	/* BUG we'll do something more efficient eventually */

	/* We also need to remove the "local" objects... */

	delete_local_objs(SINGLE_QSP_ARG);

	switch(enp->en_code){
		/* matlab */
		case T_CALL_NATIVE:
			eval_native_work(QSP_ARG  enp);
			break;

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
			exec_mlab_cmd(enp->en_intval);
			break;
		/* this needs to be here for matlab (?)
		 * but we want this to be independent of the matlab module
		 * These should be matlab native functions...
		 */
		case T_MFILE:
			read_matlab_file(enp->en_string);
			break;
		case T_PRINT:
			mlab_print_tree(enp->en_child[0]);
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
				goto_label=enp->en_string;
			}
			return(1);

		case T_LABEL:
			if( going && !strcmp(enp->en_string,goto_label) ){
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
			if( enp->en_child[0]!=NO_VEXPR_NODE )
				exit( EVAL_INT_EXP(enp->en_child[0]) );
			else
				exit(0);
			break;

		case T_FIX_SIZE:
			break;

		case T_DISPLAY:		/* eval_work_tree */
			if( going ) return(1);
			EVAL_DISPLAY_STAT(enp->en_child[0]);
			break;

		case T_SET_FUNCPTR:
			if( going ) return(1);
			srp = eval_funcref(QSP_ARG  enp->en_child[1]);
			fpp = eval_funcptr(QSP_ARG  enp->en_child[0]);
			/* BUG check for valid return values */
			fpp->fp_srp = srp;
			point_node_shape(QSP_ARG  enp,srp->sr_shpp);
			break;

		case T_SET_STR:		/* eval_work_tree */
			if( going ) return(1);
			s = EVAL_STRING(enp->en_child[1]);
			idp = EVAL_PTR_REF(enp->en_child[0],UNSET_PTR_OK);
			if( idp == NO_IDENTIFIER ) break;
			assign_string(QSP_ARG  idp,s,enp);
			break;

		case T_SET_PTR:		/* eval_work_tree */
			if( going ) return(1);

#ifdef CAUTIOUS
			if( dst_dp != NO_OBJ ){
				sprintf(error_string,"CAUTIOUS:  eval_work_tree:  T_SET_PTR, dst_dp (%s) not null!?",
					dst_dp->dt_name);
				WARN(error_string);
			}
#endif /* CAUTIOUS */
			idp2 = EVAL_PTR_REF(enp->en_child[1],EXPECT_PTR_SET);
			idp = EVAL_PTR_REF(enp->en_child[0],UNSET_PTR_OK);

			if( idp2 == NO_IDENTIFIER || idp == NO_IDENTIFIER ){
				NODE_ERROR(enp);
				advise("eval_work_tree T_SET_PTR:  null object");
				break;
			}

#ifdef CAUTIOUS
			if( ! IS_POINTER(idp) ){
				NODE_ERROR(enp);
				sprintf(error_string,"CAUTIOUS:  identifier %s does not refer to a pointer!?",
					idp->id_name);
				WARN(error_string);
				return(0);
			}
#endif /* CAUTIOUS */

			if( IS_POINTER(idp2) ){
				idp->id_ptrp->ptr_refp = idp2->id_ptrp->ptr_refp;
				idp->id_ptrp->ptr_flags |= POINTER_SET;
			} else if( IS_REFERENCE(idp2) ){
				assign_pointer(idp->id_ptrp,idp2->id_refp);
				/* can we do some runtime shape resolution here?? */
				/* We mark the node as unknown to force propagate_shape to do something
				 * even when the ptr was previously set to something else.
				 */
				copy_node_shape(QSP_ARG  idp->id_ptrp->ptr_decl_enp,uk_shape(enp->en_child[0]->en_prec));
				if( !UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) )
					RESOLVE_POINTER(enp->en_child[0],enp->en_child[1]->en_shpp);
			}
#ifdef CAUTIOUS
			  else {
				sprintf(error_string,"CAUTIOUS:  eval_work_tree T_SET_PTR:  rhs is neither a pointer nor a reference!?");
				ERROR1(error_string);
			}
#endif /* CAUTIOUS */

			break;

		case T_OUTPUT_FILE:		/* eval_work_tree */
			if( going ) return(1);
			s=EVAL_STRING(enp->en_child[0]);
			if( s!=NULL )
				set_output_file(QSP_ARG  s);
			break;

		case T_STRCPY:		/* eval_work_tree */
			if( going ) return(1);
			idp=EVAL_PTR_REF(enp->en_child[0],UNSET_PTR_OK);
			s=EVAL_STRING(enp->en_child[1]);
			if( idp != NO_IDENTIFIER && s != NULL )
				assign_string(QSP_ARG  idp,s,enp);
			break;

		case T_STRCAT:		/* eval_work_tree */
			if( going ) return(1);
			idp=EVAL_PTR_REF(enp->en_child[0],EXPECT_PTR_SET);
			s=EVAL_STRING(enp->en_child[1]);
			if( idp != NO_IDENTIFIER && s != NULL )
				cat_string(idp->id_refp->ref_sbp,s);
			break;

		case T_FOR:		/* eval_work_tree */
			do {
				/* evaluate the conditional */
				if( ! going )
					intval = EVAL_INT_EXP(enp->en_child[0]);
				else
					intval = 1;

				if( going || intval ){
					/* execute the body */
					ret_val=EVAL_TREE(enp->en_child[1],NO_OBJ);
					if( ret_val == 0 ) return(0);
					continuing=0;
					if( going ) return(ret_val);
					ret_val=EVAL_TREE(enp->en_child[2],NO_OBJ);
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
				lval = EVAL_INT_EXP(enp->en_child[0]);
				case_enp = FIND_CASE(enp->en_child[1],lval);
				if( case_enp == NO_VEXPR_NODE ){
					/* It is not an error for there to be no case.
					 * We might want to have this warning controlled by a flag.
					 */
					NODE_ERROR(enp);
					sprintf(error_string,"No case for value %ld",lval);
					WARN(error_string);
					break;
				}

#ifdef CAUTIOUS
				if( case_enp->en_code != T_CASE_STAT ){
					NODE_ERROR(enp);
					sprintf(error_string,
	"CAUTIOUS:  eval_work_tree:  find_case value (%s) not a case_stat node!?",
						node_desc(case_enp));
					ERROR1(error_string);
				}
#endif /* CAUTIOUS */
			} else {
				/* while we are looking for a goto label,
				 * we must examine all the cases...
				 */
				case_enp=first_case(enp);
			}

try_again:
			while( case_enp!=NO_VEXPR_NODE && ! breaking ){
				ret_val=EVAL_TREE(case_enp->en_child[1],NO_OBJ);
				/* BUG This test may get performed multiple times (harmlessly) */
				if( going ){
					/* first see if the target is in one of the cases at all */
					if( goto_child(enp->en_child[1]) == NO_VEXPR_NODE ) {
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
				ret_val=EVAL_TREE(enp->en_child[0],NO_OBJ);
				if( ret_val == 0 ) return(0);
				continuing = 0;
				if( ! going ) intval = EVAL_INT_EXP(enp->en_child[1]);
			} while( (!going) && !intval);
			break;

		case T_DO_WHILE:		/* eval_work_tree */
			intval=0;	// quiet compiler
			do {
				ret_val=EVAL_TREE(enp->en_child[0],NO_OBJ);
				if( ret_val == 0 ) return(0);
				continuing = 0;
				if( ! going ) intval = EVAL_INT_EXP(enp->en_child[1]);
			} while( (!going) && intval);
			break;

		case T_WHILE:			/* eval_work_tree */
			do {
				/* evaluate the conditional */
				if( !going )
					intval = EVAL_INT_EXP(enp->en_child[0]);
				else	intval = 1;
				if( intval ){
					/* execute the body */
					ret_val=EVAL_TREE(enp->en_child[1],NO_OBJ);
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
					intval = EVAL_INT_EXP(enp->en_child[0]);
				else	intval = 0;
				if( ! intval ){
					/* execute the body */
					ret_val=EVAL_TREE(enp->en_child[1],NO_OBJ);
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
			intval = EVAL_INT_EXP(enp->en_child[0]);
			NODE_ERROR(enp);
			if( intval )
				advise("enabling vector evaluation");
			else
				advise("disabling vector evaluation");

			set_perf(intval);
			break;

		case T_SCRIPT:		/* eval_work_tree */
			if( going ) return(1);
			srp = enp->en_srp;

#ifdef CAUTIOUS
			if( ! IS_SCRIPT(srp) ){
				sprintf(error_string,
	"Subrt %s is not a script subroutine!?",srp->sr_name);
				WARN(error_string);
				return(0);
			}
#endif /* CAUTIOUS */
			/* Set up dummy_mac so that the interpreter will
			 * think we are in a macro...
			 */
			dummy_mac.m_name = srp->sr_name;
			dummy_mac.m_nargs = srp->sr_nargs;
			dummy_mac.m_text = (char *) srp->sr_body;
			dummy_mac.m_flags = 0; /* disallow recursion */

			/* Any arguments to a script function
			 * will be treated like macro args...
			 */

			sprintf(msg_str,"Script func %s",srp->sr_name);
			push_input_file(QSP_ARG  msg_str);

#ifdef OLD_LOOKAHEAD
	la_level=enable_lookahead(tell_qlevel());
#endif /* OLD_LOOKAHEAD */
			pushtext(QSP_ARG  (char *)srp->sr_body);

			qp=(&THIS_QSP->qs_query[tell_qlevel(SINGLE_QSP_ARG)]);
			qp->q_macro = &dummy_mac;
			qp->q_args = (const char **)getbuf( srp->sr_nargs * sizeof(char *) );
			/* BUG?  we have to make sure than we never try to assign more than sr_nargs args! */

			intval=SET_SCRIPT_ARGS(enp->en_child[0],0,qp,srp->sr_nargs);
			if( intval != srp->sr_nargs ){
				sprintf(error_string,
	"Script subrt %s should have %d args, passed %d",
					srp->sr_name,srp->sr_nargs,intval);
				WARN(error_string);
				/* BUG? poptext? */
				return(0);
			}
			/* If we pass object names to script functions by
			 * dereferencing pointers, we may end up with invisible objects
			 * whose contexts have been popped; here we restore those
			 * contexts.
			 */

			set_script_context(SINGLE_QSP_ARG);

			push_top_menu(SINGLE_QSP_ARG);	/* make sure at root menu */
			intval = tell_qlevel(SINGLE_QSP_ARG);
			enable_stripping_quotes(SINGLE_QSP_ARG);
			while( tell_qlevel(SINGLE_QSP_ARG) >= intval ){
				do_cmd(SINGLE_QSP_ARG);
lookahead(SINGLE_QSP_ARG);
			}
			popcmd(SINGLE_QSP_ARG);		/* go back */

			unset_script_context(SINGLE_QSP_ARG);

			break;

		case T_SAVE:		/* eval_work_tree */
			if( going ) return(1);
			ifp=img_file_of(QSP_ARG  enp->en_string);
			if( ifp == NO_IMAGE_FILE ){
advise("evaltree:  save:");
describe_shape(enp->en_child[0]->en_shpp);
				ifp = write_image_file(QSP_ARG  enp->en_string,
					enp->en_child[0]->en_shpp->si_frames);
				if( ifp==NO_IMAGE_FILE ){
					/* permission error? */
					sprintf(error_string,
		"Couldn't open image file %s",enp->en_string);
					WARN(error_string);
					return(0);
				}
			}
		/* BUG we'd like to allow an arbitrary expression here!? */
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			if( dp == NO_OBJ ) return(1);

			write_image_to_file(QSP_ARG  ifp,dp);
			break;

		case T_FILETYPE:		/* eval_work_tree */
			if( going ) return(1);
			/* BUG? scan tree should maybe fetch this? */
			intval = get_filetype_index(enp->en_string);
			if( intval < 0 ) return(0);
			set_filetype(QSP_ARG  (filetype_code)intval);
			break;

		case T_DECL_STAT_LIST:			/* eval_work_tree */
		case T_DECL_STAT:		/* eval_work_tree */
		case T_EXTERN_DECL:		/* eval_work_tree */
			if( going ) return(1);
#ifdef DEBUG
if( debug & eval_debug ){
sprintf(error_string,"eval_work_tree:  nothing to do for declarations");
advise(error_string);
}
#endif /* DEBUG */
			return(ret_val);

		case T_STAT_LIST:				/* eval_work_tree */
			if( (ret_val=EVAL_WORK_TREE(enp->en_child[0],dst_dp)) ){
				if( continuing || breaking ) return(ret_val);
				ret_val=EVAL_WORK_TREE(enp->en_child[1],dst_dp);
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
				goto_enp=goto_child(enp->en_child[0]);
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
				intval = EVAL_INT_EXP(enp->en_child[0]);
				if( intval )
					return( EVAL_TREE(enp->en_child[1],dst_dp) );
				else if( enp->en_child[2] != NO_VEXPR_NODE )
					return( EVAL_TREE(enp->en_child[2],dst_dp) );
			} else {
				ret_val = EVAL_TREE(enp->en_child[1],dst_dp);
				if( ! going ) return(1);
				if( enp->en_child[1] != NO_VEXPR_NODE )
					ret_val = EVAL_TREE(enp->en_child[1],dst_dp);
				return(1);
			}
			break;

		case T_SUBRT:		/* eval_work_tree */
			if( going ) return(1);
			srp=enp->en_srp;
			/* if there are args, need to pass */
			WARN("eval_work_tree T_SUBRT - NOT calling subroutine???");
#ifdef DEBUG
if( debug & eval_debug ){
sprintf(error_string,"eval_work_tree:  what do we do for subrt %s!?",srp->sr_name);
advise(error_string);
}
#endif /* DEBUG */
			break;

		case T_RETURN:		/* eval_work_tree */
			if( going ) return(1);
			if( enp->en_child[0] != NO_VEXPR_NODE ){
				EVAL_OBJ_ASSIGNMENT(dst_dp,enp->en_child[0]);
			}
			/* If we are returning from a subroutine before the end,
			 * we have to pop it now...
			 */
			return(0);

		case T_EXP_PRINT:		/* eval_work_tree */
			if( going ) return(1);
			EVAL_PRINT_STAT(enp->en_child[0]);
			prt_msg("");	/* print newline after other expressions */
			break;

		case T_INFO:		/* eval_work_tree */
			if( going ) return(1);
			EVAL_INFO_STAT(enp->en_child[0]);
			break;

		case T_WARN:		/* eval_work_tree */
			if( going ) return(1);
			s=EVAL_STRING(enp->en_child[0]);
			if( s != NULL ) WARN(s);
			break;

		case T_ADVISE:		/* eval_work_tree */
			if( going ) return(1);
			s=EVAL_STRING(enp->en_child[0]);
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
				if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) )
					UPDATE_TREE_SHAPE(enp->en_child[0]);
				if( UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) )
					UPDATE_TREE_SHAPE(enp->en_child[1]);
				if( UNKNOWN_SHAPE(enp->en_shpp) &&
						! UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){
					RESOLVE_TREE(enp,NO_VEXPR_NODE);
				}
			}

			if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"eval_work_tree ASSIGN:  last ditch attempt at runtime resolution of LHS %s",node_desc(enp->en_child[0]));
advise(error_string);
DUMP_TREE(enp);
}
#endif /* DEBUG */
				/*
				resolve_one_uk_node(enp->en_child[0]);
				*/
				RESOLVE_TREE(enp->en_child[0],NO_VEXPR_NODE);

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"eval_work_tree ASSIGN:  after last ditch attempt at runtime resolution of LHS %s:",node_desc(enp->en_child[0]));
advise(error_string);
DUMP_TREE(enp);
}
#endif /* DEBUG */
			}
			if( UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"eval_work_tree ASSIGN:  last ditch attempt at runtime resolution of RHS %s",node_desc(enp->en_child[1]));
advise(error_string);
}
#endif /* DEBUG */
				/*
				resolve_one_uk_node(enp->en_child[1]);
				*/
				RESOLVE_TREE(enp->en_child[1],NO_VEXPR_NODE);
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"eval_work_tree ASSIGN:  after last ditch attempt at runtime resolution of RHS %s:",node_desc(enp->en_child[1]));
advise(error_string);
DUMP_TREE(enp);
}
#endif /* DEBUG */
			}

			dp = EVAL_OBJ_REF(enp->en_child[0]);
			if( dp == NO_OBJ ){
				NODE_ERROR(enp);
				WARN("Invalid LHS");
				break;
			}

			if( mode_is_matlab ){
				dp=MLAB_LHS(dp,enp);
#ifdef CAUTIOUS
if( dp == NO_OBJ ){
sprintf(error_string,"CAUTIOUS:  mlab_lhs returned a null ptr!?");
WARN(error_string);
break;
}
#endif /* CAUTIOUS */
				EVAL_OBJ_ASSIGNMENT(dp,enp->en_child[1]);
				break;
			}

#ifdef DEBUG
if( debug & eval_debug ){
sprintf(error_string,"eval_work_tree:  calling eval_obj_assignment for target %s",dp->dt_name);
advise(error_string);
}
#endif /* DEBUG */

			if( enp->en_code == T_DIM_ASSIGN )
				EVAL_DIM_ASSIGNMENT(dp,enp->en_child[1]);
			else
				EVAL_OBJ_ASSIGNMENT(dp,enp->en_child[1]);
			break;

		case T_PREINC:		/* eval_work_tree */
		case T_POSTINC:		/* eval_work_tree */
			if( going ) return(1);
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			inc_obj(dp);
			break;

		case T_POSTDEC:
		case T_PREDEC:
			if( going ) return(1);
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			dec_obj(dp);
			break;



		default:		/* eval_work_tree */
			MISSING_CASE(enp,"eval_work_tree");
			break;
	}
	return(ret_val);
} /* end eval_work_tree */

long eval_int_exp(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	long lval,lval2;
	double dval1,dval2;
	Data_Obj *dp;
	Scalar_Value *svp,sval;
	Dimension_Set dimset={{1,1,1,1,1}};
	Subrt *srp;

	eval_enp = enp;

	switch(enp->en_code){
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
				sprintf(error_string,
	"eval_int_exp T_VS_FUNC:  object %s is not a scalar!?",dp->dt_name);
				WARN(error_string);
				return(0);
			}
			return( get_long_scalar_value(dp) );
			break;

		case T_RECIP:
			lval = EVAL_INT_EXP(enp->en_child[0]);
			if( lval == 0 ){
				NODE_ERROR(enp);
				WARN("divide by zero!?");
				return(0);
			}
			return(1/lval);
			
		case T_TYPECAST:	/* eval_int_exp */
			dval1 = EVAL_FLT_EXP(enp->en_child[0]);
			switch(enp->en_prec){
				case PREC_BY:   return( (long) ((char)     dval1 ) );
				case PREC_UBY:  return( (long) ((u_char)   dval1 ) );
				case PREC_IN:   return( (long) ((short)    dval1 ) );
				case PREC_UIN:  return( (long) ((u_short)  dval1 ) );
				case PREC_DI:   return( (long) ((int32_t)  dval1 ) );
				case PREC_UDI:  return( (long) ((uint32_t) dval1 ) );
				case PREC_LI:   return( (long) ((int64_t)  dval1 ) );
				case PREC_ULI:  return( (long) ((uint64_t) dval1 ) );
				case PREC_SP:   return( (long) ((float)    dval1 ) );
				case PREC_DP:   return(                    dval1   );
				case PREC_BIT:
					if( dval1 == 0.0 ) return(0);
					else return(1);

#ifdef CAUTIOUS
				default:
					NODE_ERROR(enp);
					sprintf(error_string,
		"CAUTIOUS:  eval_int_exp:  unhandled precision %s (%d, 0x%x) in TYPECAST switch",
						name_for_prec(enp->en_prec),enp->en_prec,enp->en_prec);
					ERROR1(error_string);
#endif /* CAUTIOUS */
			}
			break;

		case T_CALLFUNC:			/* eval_int_exp */
			/* This could get called if we use a function inside a dimesion bracket... */
			if( ! executing ) return(0);

			srp=enp->en_call_srp;
			/* BUG SHould check and see if the return type is int... */

			/* BUG at least make sure that it's not void... */

			/* make a scalar object to hold the return value... */
			dp=make_local_dobj(QSP_ARG  &dimset,srp->sr_prec);
			EXEC_SUBRT(enp,dp);
			/* get the scalar value */
			lval = get_long_scalar_value(dp);
			delvec(QSP_ARG  dp);
			return(lval);
			break;

		case T_BOOL_PTREQ:
			{
			Identifier *idp1,*idp2;
			idp1=EVAL_PTR_REF(enp->en_child[0],EXPECT_PTR_SET);
			idp2=EVAL_PTR_REF(enp->en_child[1],EXPECT_PTR_SET);
			/* CAUTIOUS check for ptrs? */
			/* BUG? any other test besides dp ptr identity? */
			if( idp1->id_refp->ref_dp == idp2->id_refp->ref_dp )
				return(1);
			else
				return(0);
			}

		case T_POSTINC:
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			lval = EVAL_INT_EXP(enp->en_child[0]);
			inc_obj(dp);
			return(lval);

		case T_POSTDEC:
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			lval = EVAL_INT_EXP(enp->en_child[0]);
			dec_obj(dp);
			return(lval);

		case T_PREDEC:
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			dec_obj(dp);
			return(EVAL_INT_EXP(enp->en_child[0]));

		case T_PREINC:		/* eval_int_exp */
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			inc_obj(dp);
			return(EVAL_INT_EXP(enp->en_child[0]));

		case T_ASSIGN:
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			lval = EVAL_INT_EXP(enp->en_child[1]);
			int_to_scalar(&sval,lval,dp->dt_prec);
			if( ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval) < 0 )
				return(0);
			return(lval);

		case T_UNDEF:
			return(0);

		case T_FILE_EXISTS:
			{
				const char *s;
				s=EVAL_STRING(enp->en_child[0]);
				if( s != NULL )
					return(file_exists(s));
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
#ifdef CAUTIOUS
			if( dp==NO_OBJ ){
				sprintf(error_string,
	"CAUTIOUS:  eval_int_exp:  missing dobj %s",enp->en_string);
				WARN(error_string);
				return(0);
			}
#endif /* CAUTIOUS */
			if( ! IS_SCALAR(dp) ){
				NODE_ERROR(enp);
				sprintf(error_string,
	"eval_int_exp:  Object %s is not a scalar!?",dp->dt_name);
				WARN(error_string);
				LONGLIST(dp);
				return(0);
			}
			/* has the object been set? */
			if( ! HAS_VALUES(dp) ){
				if( executing && expect_objs_assigned ){
					NODE_ERROR(enp);
					sprintf(error_string,"Object %s is used before value has been set",
						dp->dt_name);
					WARN(error_string);
				}
				
				return(0);			/* we don't print the warning unless we know
								 * that we aren't doing pre-evaluation...
								 */
			}
			svp = (Scalar_Value *)dp->dt_data;
			switch(MACHINE_PREC(dp)){
				case PREC_BY:  lval = svp->u_b; break;
				case PREC_IN:  lval = svp->u_s; break;
				case PREC_DI:  lval = svp->u_l; break;
				case PREC_LI:  lval = svp->u_ll; break;
				case PREC_SP:  lval = svp->u_f; break;
				case PREC_DP:  lval = svp->u_d; break;
				case PREC_UBY:  lval = svp->u_ub; break;
				case PREC_UIN:  lval = svp->u_us; break;
				case PREC_UDI:  lval = svp->u_ul; break;
				case PREC_ULI:  lval = svp->u_ull; break;
#ifdef CAUTIOUS
				case PREC_NONE:
				case N_MACHINE_PRECS:
				default:
					sprintf(error_string,
			"CAUTIOUS: eval_int_exp:  %s has nonsense precision",
					dp->dt_name);
					WARN(error_string);
					lval=0.0;	// quiet compiler
					break;
#endif /* CAUTIOUS */
			}
			return(lval);
			break;

		case T_BOOL_OR:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			lval2=EVAL_INT_EXP(enp->en_child[1]);
			if( lval || lval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_AND:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			lval2=EVAL_INT_EXP(enp->en_child[1]);
			if( lval && lval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_NOT:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			if( ! lval ) return(1);
			else return(0);
			break;
		case T_BOOL_GT:			/* eval_int_exp */
			dval1=EVAL_FLT_EXP(enp->en_child[0]);
			dval2=EVAL_FLT_EXP(enp->en_child[1]);
			if( dval1 > dval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_LT:			/* eval_int_exp */
			dval1=EVAL_FLT_EXP(enp->en_child[0]);
			dval2=EVAL_FLT_EXP(enp->en_child[1]);
			if( dval1 < dval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_GE:
			dval1=EVAL_FLT_EXP(enp->en_child[0]);
			dval2=EVAL_FLT_EXP(enp->en_child[1]);
			if( dval1 >= dval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_LE:
			dval1=EVAL_FLT_EXP(enp->en_child[0]);
			dval2=EVAL_FLT_EXP(enp->en_child[1]);
			if( dval1 <= dval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_NE:
			dval1=EVAL_FLT_EXP(enp->en_child[0]);
			dval2=EVAL_FLT_EXP(enp->en_child[1]);
			if( dval1 != dval2 ) return(1);
			else return(0);
			break;
		case T_BOOL_EQ:
			dval1=EVAL_FLT_EXP(enp->en_child[0]);
			dval2=EVAL_FLT_EXP(enp->en_child[1]);
			if( dval1 == dval2 ) return(1);
			else return(0);
			break;
		case T_PLUS:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			lval2=EVAL_INT_EXP(enp->en_child[1]);
			return(lval+lval2);
			break;
		case T_MINUS:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			lval2=EVAL_INT_EXP(enp->en_child[1]);
			return(lval-lval2);
			break;
		case T_TIMES:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			lval2=EVAL_INT_EXP(enp->en_child[1]);
			return(lval*lval2);
			break;
		case T_DIVIDE:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			lval2=EVAL_INT_EXP(enp->en_child[1]);
			if( lval2==0 ){
				WARN("integer division by 0!?");
				return(0L);
			}
			return(lval/lval2);
			break;
		case T_MODULO:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			lval2=EVAL_INT_EXP(enp->en_child[1]);
			if( lval2==0 ){
				WARN("integer division (modulo) by 0!?");
				return(0L);
			}
			return(lval%lval2);
			break;
		case T_BITOR:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			lval2=EVAL_INT_EXP(enp->en_child[1]);
			return(lval|lval2);
			break;
		case T_BITAND:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			lval2=EVAL_INT_EXP(enp->en_child[1]);
			return(lval&lval2);
			break;
		case T_BITXOR:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			lval2=EVAL_INT_EXP(enp->en_child[1]);
			return(lval^lval2);
			break;
		case T_BITCOMP:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			return(~lval);
			break;
		case T_BITRSHIFT:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			lval2=EVAL_INT_EXP(enp->en_child[1]);
			return(lval>>lval2);
			break;
		case T_BITLSHIFT:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			lval2=EVAL_INT_EXP(enp->en_child[1]);
			return(lval<<lval2);
			break;
		case T_LIT_INT:			/* eval_int_exp */
			return(enp->en_intval);
			break;
		case T_UMINUS:
			lval=EVAL_INT_EXP(enp->en_child[0]);
			return(-lval);
			break;

		case T_STR2_FN:			/* eval_int_exp */
		case T_LIT_DBL:
		case T_STR1_FN:	/* eval_int_exp */
		case T_SIZE_FN: 	/* eval_int_exp */
			lval=EVAL_FLT_EXP(enp);
			return(lval);
			break;
		default:
			MISSING_CASE(enp,"eval_int_exp");
			break;
	}
	return(-1L);
} /* end eval_int_exp */

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
	if( enp->en_shpp == NO_SHAPE ){
		for(i=0;i<N_DIMENSIONS;i++) dsp->ds_dimension[i]=0;
		copy_node_shape(QSP_ARG  enp,uk_shape(enp->en_decl_prec));
	} else {
		/* use node shape struct
		 * hopefully set to calling context
		 * by rt_scan()
		 */
		for(i=0;i<N_DIMENSIONS;i++)
			dsp->ds_dimension[i]= enp->en_shpp->si_type_dim[i];
	}

#ifdef CAUTIOUS
	if( enp->en_parent == NO_VEXPR_NODE ){
		sprintf(error_string,"CAUTIOUS:  setup_unknown_shape:  %s has no parent",
			node_desc(enp));
		WARN(error_string);
	}
	/* We don't link this node to anything, because it's a declaration node */
#endif /* ! CAUTIOUS */
}

/* Process a tree, doing only declarations */

void eval_decl_tree(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	if( enp==NO_VEXPR_NODE )
		return;

	eval_enp = enp;

	switch(enp->en_code){
		case T_STAT_LIST:
			/* There should only be one T_STAT_LIST at
			 * the beginning of a subroutine, so we
			 * don't need to scan the second child.
			 * for declarations...
			 */
			EVAL_DECL_TREE(enp->en_child[0]);
			break;
		case T_DECL_STAT_LIST:
			EVAL_DECL_TREE(enp->en_child[0]);
			EVAL_DECL_TREE(enp->en_child[1]);
			break;
		case T_DECL_STAT:
			EVAL_DECL_STAT(enp->en_decl_prec,enp->en_child[0],enp->en_decl_flags);
			break;
		case T_EXTERN_DECL:
			EVAL_EXTERN_DECL(enp->en_decl_prec,enp->en_child[0],enp->en_decl_flags);
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

	if( enp1->en_code != enp2->en_code ) return(-1);

	if( enp1->en_code == T_DECL_STAT ){
		if( enp1->en_decl_prec != enp2->en_decl_prec ) return(-1);
	}

	for(i=0;i<MAX_CHILDREN(enp1);i++){
		if( compare_arg_decls(enp1->en_child[i],enp2->en_child[i]) < 0 )
			return(-1);
	}
	return(0);
}

void compare_arg_trees(QSP_ARG_DECL  Vec_Expr_Node *enp1,Vec_Expr_Node *enp2)
{
	if( compare_arg_decls(enp1,enp2) < 0 )
		prototype_mismatch(QSP_ARG  enp1,enp2);
}


static void eval_extern_decl(QSP_ARG_DECL prec_t prec,Vec_Expr_Node *enp, int decl_flags)
{
	eval_enp = enp;

	switch(enp->en_code){
		case T_PROTO:
			{
			Subrt *srp;
			srp=subrt_of(QSP_ARG  enp->en_string);
			if( srp == NO_SUBRT ) EVAL_DECL_STAT(prec,enp,decl_flags);
			else {
				/* This subroutine has already been declared...
				 * make sure the type matches
				 */
				if( prec != srp->sr_prec )
					prototype_mismatch(QSP_ARG  srp->sr_arg_decls,enp);
			}

			/* BUG make sure arg decls match */
			return;
			}
		case T_BADNAME: return;
		case T_DECL_ITEM_LIST:
			EVAL_EXTERN_DECL(prec,enp->en_child[0],decl_flags);
			if( enp->en_child[1]!=NO_VEXPR_NODE )
				EVAL_EXTERN_DECL(prec,enp->en_child[1],decl_flags);
			return;
		case T_DECL_INIT:
			NODE_ERROR(enp);
			advise("no auto-initialization with extern declarations");
			EVAL_EXTERN_DECL(prec,enp->en_child[0],decl_flags);
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

			dp=DOBJ_OF(enp->en_string);
			if( dp == NO_OBJ ){
				EVAL_DECL_STAT(prec,enp,decl_flags);
				return;
			}
			/* BUG should check that decl matches earlier one... */
			break;
			}
		case T_PTR_DECL:			/* eval_extern_decl */
			{
			Identifier *idp;
			idp = ID_OF(enp->en_string);
			if( idp == NO_IDENTIFIER ){
				EVAL_DECL_STAT(prec,enp,decl_flags);
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

static void eval_decl_stat(QSP_ARG_DECL prec_t prec,Vec_Expr_Node *enp, int decl_flags)
{
	int i;
	Dimension_Set dimset;
	int type/* =ID_OBJECT */;
	Identifier *idp;

	if( prec == PREC_STR ){
		type = ID_STRING;
	} else {
		type = ID_REFERENCE;
	}

	eval_enp = enp;

	for(i=0;i<N_DIMENSIONS;i++)
		dimset.ds_dimension[i]=1;

/*
sprintf(error_string,"eval_decl_stat %s:",node_desc(enp));
advise(error_string);
if( enp->en_shpp != NO_SHAPE ) describe_shape( enp->en_shpp);
else prt_msg("\t(no shape)");
DUMP_TREE(enp);
*/

	switch(enp->en_code){
		case T_PROTO:
			{
			Subrt *srp;
			srp=subrt_of(QSP_ARG  enp->en_string);
			if( srp != NO_SUBRT ){
				/* subroutine already declared.
				 * We should check to make sure that the arg decls match BUG
				 * this gets done elsewhere, but here we make sure the return
				 * type is the same.
				 */
				/* This subroutine has already been declared...
				 * make sure the type matches
				 */
				if( prec != srp->sr_prec )
					prototype_mismatch(QSP_ARG  srp->sr_arg_decls,enp);
				break;
			}
			srp = remember_subrt(QSP_ARG  prec,enp->en_string,enp->en_child[0],NO_VEXPR_NODE);
			srp->sr_nargs = decl_count(QSP_ARG  srp->sr_arg_decls);	/* set # args */
			srp->sr_flags |= SR_PROTOTYPE;
			return;
			}

		case T_BADNAME:
			return;
		case T_DECL_ITEM_LIST:
			EVAL_DECL_STAT(prec,enp->en_child[0],decl_flags);
			if( enp->en_child[1]!=NO_VEXPR_NODE )
				EVAL_DECL_STAT(prec,enp->en_child[1],decl_flags);
			return;
		case T_DECL_INIT:		/* eval_decl_stat */
			{
			Scalar_Value sval;
			Data_Obj *dp;
			double dval;

			/* CURDLED? */
			if( IS_CURDLED(enp) ) return;

			EVAL_DECL_STAT(prec,enp->en_child[0],decl_flags);
			/* the next node is an expression */
			dp = get_id_obj(QSP_ARG  enp->en_child[0]->en_string,enp);
if( dp==NO_OBJ ) WARN("eval_decl_stat:  Null object to initialize!?");

			/* What if the rhs is unknown size - then we have to resolve now! */
			if( UNKNOWN_SHAPE(&dp->dt_shape) ){
				/* Can we assume the rhs has a shape? */
				if( UNKNOWN_SHAPE(enp->en_child[1]->en_shpp) ){
					NODE_ERROR(enp);
					WARN("LHS and RHS are both unknown shape!?");
				} else {
advise("attempting resolution");
					RESOLVE_TREE(enp,NO_VEXPR_NODE);
					DUMP_TREE(enp);
				}
			}

			if( SCALAR_SHAPE(enp->en_child[1]->en_shpp) ){
				dval = EVAL_FLT_EXP(enp->en_child[1]);
				dbl_to_scalar(&sval,dval,dp->dt_prec);
				ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			} else {
				EVAL_OBJ_ASSIGNMENT(dp,enp->en_child[1]);
			}
			return;
			}
		case T_SCAL_DECL:
			enp->en_decl_prec = prec;

			break;
		case T_CSCAL_DECL:					/* eval_decl_stat */
			enp->en_decl_prec = prec;

			/* eg float x{3} */
			if( enp->en_child[0] == NO_VEXPR_NODE ){
				/* float x{} */
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				else
					/* BUG?  type_dimset or mach_dimset? */
					dimset = enp->en_shpp->si_type_dimset;
			} else {
				dimset.ds_dimension[0]=EVAL_INT_EXP(enp->en_child[0]);
				if( dimset.ds_dimension[0] == 0 ){
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				}
			}
			break;
		case T_VEC_DECL:					/* eval_decl_stat */
			enp->en_decl_prec = prec;

			if( enp->en_child[0] == NO_VEXPR_NODE ){
				if( ! IS_RESOLVED(enp) ) {
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				} else {
					/* BUG?  type_dimset or mach_dimset? */
					dimset = enp->en_shpp->si_type_dimset;
				}
			} else {
				dimset.ds_dimension[1]=EVAL_INT_EXP(enp->en_child[0]);
				if( dimset.ds_dimension[1] == 0 ){
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				}
			}
			break;
		case T_CVEC_DECL:
			enp->en_decl_prec = prec;

			if( enp->en_child[0] == NO_VEXPR_NODE ){
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				else {
					/* BUG?  type_dimset or mach_dimset? */
					dimset = enp->en_shpp->si_type_dimset;
				}
			} else {
				dimset.ds_dimension[1]=EVAL_INT_EXP(enp->en_child[0]);
				dimset.ds_dimension[0]=EVAL_INT_EXP(enp->en_child[1]);
				if( dimset.ds_dimension[2] == 0 || dimset.ds_dimension[1] == 0 ){
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				}
			}
			break;
		case T_IMG_DECL:
			enp->en_decl_prec = prec;

			if( enp->en_child[0] == NO_VEXPR_NODE ){
				if( ! IS_RESOLVED(enp) ){
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				} else {
/*
sprintf(error_string,"resolved, nr = %d  nc = %d",enp->en_shpp->si_type_dim[2],
enp->en_shpp->si_type_dim[1]);
advise(error_string);
*/
					/* BUG?  type_dimset or mach_dimset? */
					dimset = enp->en_shpp->si_type_dimset;
				}
			} else {
				dimset.ds_dimension[2]=EVAL_INT_EXP(enp->en_child[0]);
				dimset.ds_dimension[1]=EVAL_INT_EXP(enp->en_child[1]);
				if( dimset.ds_dimension[2] == 0 || dimset.ds_dimension[1] == 0 ){
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				}

			}
			break;
		case T_CIMG_DECL:
			enp->en_decl_prec = prec;

			if( enp->en_child[0] == NO_VEXPR_NODE ){
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				else {
					/* BUG?  type_dimset or mach_dimset? */
					dimset = enp->en_shpp->si_type_dimset;
				}
			} else {
				dimset.ds_dimension[2]=EVAL_INT_EXP(enp->en_child[0]);
				dimset.ds_dimension[1]=EVAL_INT_EXP(enp->en_child[1]);
				dimset.ds_dimension[0]=EVAL_INT_EXP(enp->en_child[2]);
				if( dimset.ds_dimension[2] == 0 || dimset.ds_dimension[1] == 0 || dimset.ds_dimension[0] == 0 ){
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				}
			}
			break;
		case T_SEQ_DECL:
			enp->en_decl_prec = prec;

			if( enp->en_child[0] == NO_VEXPR_NODE ){
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				else {
					/* BUG?  type_dimset or mach_dimset? */
					dimset = enp->en_shpp->si_type_dimset;
				}
			} else {
				dimset.ds_dimension[3]=EVAL_INT_EXP(enp->en_child[0]);
				dimset.ds_dimension[2]=EVAL_INT_EXP(enp->en_child[1]);
				dimset.ds_dimension[1]=EVAL_INT_EXP(enp->en_child[2]);
				if( dimset.ds_dimension[3] == 0 || dimset.ds_dimension[2] == 0 || dimset.ds_dimension[1] == 0 ){
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				}
			}
			break;
		case T_CSEQ_DECL:
			enp->en_decl_prec = prec;

			if( enp->en_child[0] == NO_VEXPR_NODE ){
				if( ! IS_RESOLVED(enp) )
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				else {
					/* BUG?  type_dimset or mach_dimset? */
					dimset = enp->en_shpp->si_type_dimset;
				}
			} else {
				Vec_Expr_Node *enp2;
				dimset.ds_dimension[3]=EVAL_INT_EXP(enp->en_child[0]);
				dimset.ds_dimension[2]=EVAL_INT_EXP(enp->en_child[1]);
				enp2 = enp->en_child[2];
#ifdef CAUTIOUS
				if( enp2->en_code != T_EXPR_LIST )
					WARN("CAUTIOUS:  node should be T_EXPR_LIST!?");
#endif /* CAUTIOUS */
				dimset.ds_dimension[1]=EVAL_INT_EXP(enp2->en_child[0]);
				dimset.ds_dimension[0]=EVAL_INT_EXP(enp2->en_child[1]);
				if( dimset.ds_dimension[3] == 0 || dimset.ds_dimension[2] == 0 || dimset.ds_dimension[1] == 0 || dimset.ds_dimension[0] == 0 ){
					setup_unknown_shape(QSP_ARG  enp,&dimset);
				}
			}
			break;
		case T_PTR_DECL:			/* eval_decl_stat() */
			enp->en_decl_prec = prec;

			/* call by reference */
			if( type != ID_STRING )
				type = ID_POINTER;
#ifdef CAUTIOUS
			if( type == ID_STRING && prec != PREC_CHAR ){
				NODE_ERROR(enp);
				WARN("CAUTIOUS:  string object does not have string prec!?");
			}
#endif /* CAUTIOUS */
			break;
		case T_FUNCPTR_DECL:			/* eval_decl_stat */
			enp->en_decl_prec = prec;

			type = ID_FUNCPTR;
			break;
		default:
			MISSING_CASE(enp,"eval_decl_stat");
			break;
	}


	if( prec==PREC_COLOR ){
		dimset.ds_dimension[0]=3;
	} else if( prec==PREC_VOID ){	/* determine from context? */
		if( enp->en_shpp != NO_SHAPE )
			prec=enp->en_shpp->si_prec;
	}

	/* We allow name conflicts at levels above the current context.
	 * RESTRICT_ITEM_CONTEXT causes item lookup to only use the top context.
	 */

	RESTRICT_ITEM_CONTEXT(id_itp, 1);

#ifdef CAUTIOUS
	if( enp->en_string == NULL ){
		NODE_ERROR(enp);
		sprintf(error_string,"CAUTIOUS:  eval_decl_stat:  %s has null string!?",node_desc(enp));
		WARN(error_string);
	}
#endif /* CAUTIOUS */

	idp = ID_OF(enp->en_string);
	if( idp != NO_IDENTIFIER ){
		NODE_ERROR(enp);
		sprintf(error_string,"identifier %s redeclared",enp->en_string);
		advise(error_string);
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

	RESTRICT_ITEM_CONTEXT(id_itp, 0);
	idp=ID_OF(enp->en_string);
	if( idp != NO_IDENTIFIER ){
		/* only print this message once (the code seems to be
		 * executed 3 times!?
		 */
		if( ! WAS_WARNED(enp) ){
			NODE_ERROR(enp);
			sprintf(error_string,"declaration of %s masks previous declaration",enp->en_string);
			advise(error_string);
			MARK_WARNED(enp)
		}
/*
show_context_stack(QSP_ARG  dobj_itp);
*/
		/* this stuff should all be debug only... */
		/*
		if( idp->id_type == ID_OBJECT ){
			Vec_Expr_Node *decl_enp;
			sprintf(error_string,"context of %s (%s) is %s",
				idp->id_name,idp->id_dp->dt_name,idp->id_dobj_icp->ic_name);
			advise(error_string);
			decl_enp = idp->id_dp->dt_extra;
			if( decl_enp != NO_VEXPR_NODE ){
				advise("previous object declaration at:");
				NODE_ERROR(decl_enp);
			}
			sprintf(error_string,"current context is %s",
				((Item_Context *)CONTEXT_LIST(dobj_itp)->l_head->n_data)->ic_name);
			advise(error_string);
		}
		*/
	}

	/* remember the declaration identifier context */
	// Could we use CURRENT_CONTEXT equivalently???
	enp->en_decl_icp = (Item_Context *)CONTEXT_LIST(id_itp)->l_head->n_data;

//sprintf(error_string,"eval_decl_stat:  creating id %s, type = %d",
//enp->en_string,type);
//advise(error_string);
	// New items are always created in the top context.

	idp = new_id(QSP_ARG  enp->en_string);		/* eval_decl_stat */
	idp->id_type = type;


#ifdef CAUTIOUS
	if( idp == NO_IDENTIFIER ){
		NODE_ERROR(enp);
		sprintf(error_string,"CAUTIOUS:  unable to create new identifier %s",enp->en_string);
		WARN(error_string);
		return;
	}
#endif /* CAUTIOUS */

	switch( type ){

		case ID_REFERENCE:
			idp->id_dobj_icp = (Item_Context *)CONTEXT_LIST(dobj_itp)->l_head->n_data;
			idp->id_refp = (Reference *)getbuf( sizeof(Reference) );
			idp->id_refp->ref_idp = idp;
			idp->id_refp->ref_decl_enp = enp;	/* BUG? */
			idp->id_refp->ref_typ = OBJ_REFERENCE;
			idp->id_refp->ref_dp = finish_obj_decl(QSP_ARG  enp,&dimset,prec,decl_flags);	/* eval_decl_stat */

#ifdef CAUTIOUS
			if( idp->id_refp->ref_dp == NO_OBJ ){
				NODE_ERROR(enp);
				sprintf(error_string,
			"CAUTIOUS:  eval_decl_stat:  unable to create object for id %s",idp->id_name);
				WARN(error_string);
			}
#endif /* CAUTIOUS */
			break;

		case ID_STRING:
			idp->id_refp = (Reference *)getbuf( sizeof(Reference) );
			idp->id_refp->ref_idp = idp;
			idp->id_refp->ref_decl_enp = enp;	/* BUG? */
			idp->id_refp->ref_typ = STR_REFERENCE;
			idp->id_refp->ref_sbp = (String_Buf *)getbuf(sizeof(String_Buf));
			idp->id_refp->ref_sbp->sb_buf = NULL;
			idp->id_refp->ref_sbp->sb_size = 0;
			break;
		case ID_POINTER:
			idp->id_ptrp = (Pointer *)getbuf(sizeof(Pointer));
			idp->id_ptrp->ptr_refp = NO_REFERENCE;
			idp->id_ptrp->ptr_flags = 0;
			idp->id_ptrp->ptr_decl_enp = enp;
			copy_node_shape(QSP_ARG  enp,uk_shape(prec));
			break;
		case ID_FUNCPTR:
			idp->id_fpp = (Function_Ptr *)getbuf(sizeof(Function_Ptr));
			idp->id_fpp->fp_srp = NO_SUBRT;
			copy_node_shape(QSP_ARG  enp,uk_shape(prec));
			break;
		default:
			NODE_ERROR(enp);
			sprintf(error_string,
				"identifier type %d not handled by eval_decl_stat switch",
				type);
			WARN(error_string);
			break;
	}
} /* end eval_decl_stat */

static Data_Obj * finish_obj_decl(QSP_ARG_DECL  Vec_Expr_Node *enp,Dimension_Set *dsp,prec_t prec, int decl_flags)
{
	Data_Obj *dp;

	eval_enp = enp;

	/* at one time we handled special (complex) precision here... */

	dp=make_dobj(QSP_ARG  enp->en_string,dsp,prec);

	if( dp==NO_OBJ ){
		NODE_ERROR(enp);
		sprintf(error_string,
			"Error processing declaration for object %s",
			enp->en_string);
		WARN(error_string);
		return(dp);
	}

	/* remember the declaration node for this object */
	dp->dt_extra = enp;

	copy_node_shape(QSP_ARG  enp,&dp->dt_shape);

	if( decl_flags & DECL_IS_CONST ) dp->dt_flags |= DT_RDONLY;
	if( decl_flags & DECL_IS_STATIC ) dp->dt_flags |= DT_STATIC;

	return(dp);
}

/* This whole check is probably CAUTIOUS */

static int bad_reeval_shape(Vec_Expr_Node *enp)
{
	eval_enp = enp;

	if( enp->en_shpp == NO_SHAPE ){
		sprintf(DEFAULT_ERROR_STRING,
	"reeval:  missing shape info for %s",enp->en_string);
		NWARN(DEFAULT_ERROR_STRING);
		return(1);
	}
	if( UNKNOWN_SHAPE(enp->en_shpp) ){
		sprintf(DEFAULT_ERROR_STRING,
	"reeval:  unknown shape info for %s",enp->en_string);
		NWARN(DEFAULT_ERROR_STRING);
		return(1);
	}
	return(0);
}

/* We call reeval_decl_stat when we think we know the sizes of all unknown
 * shape objects.
 *
 * We have a problem:  if we have unknown size global objects, they may be resolved during
 * the execution of a subroutine - they are destroyed and created by reeval_decl_stat,
 * but if they're created with the local subrt context, then they will be destroyed
 * when the subroutine exits...  Therefore we need to somehow carry the original context
 * arount with us...
 */

void reeval_decl_stat(QSP_ARG_DECL  prec_t prec,Vec_Expr_Node *enp,int decl_flags)
{
	int i;
	Dimension_Set dimset;
	Data_Obj *dp;
	Identifier *idp;
	int context_pushed;

	eval_enp = enp;

	for(i=0;i<N_DIMENSIONS;i++)
		dimset.ds_dimension[i]=1;

#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"reeval_decl_stat, code is %d",enp->en_code);
advise(error_string);
}
#endif /* DEBUG */
	switch(enp->en_code){
		case T_BADNAME:
			return;
		case T_DECL_ITEM_LIST:
			reeval_decl_stat(QSP_ARG  prec,enp->en_child[0],decl_flags);
			if( enp->en_child[1]!=NO_VEXPR_NODE )
				reeval_decl_stat(QSP_ARG  prec,enp->en_child[1],decl_flags);
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
			if( enp->en_child[0] == NO_VEXPR_NODE ){
				/* the node should have the shape info */
				if( bad_reeval_shape(enp) ) return;

				/* We used to just copy in the dimensions we
				 * thought we needed, but that didn't work
				 * for column vectors!
				 * Can you see why not?
				 */

				for(i=0;i<N_DIMENSIONS;i++)
					dimset.ds_dimension[i]=enp->en_shpp->si_type_dim[i];
			} else {
				return;
			}
			break;

		default:
			MISSING_CASE(enp,"reeval_decl_stat");
			break;
	}

	/* First make sure that the context of this declaration is active */
	PUSH_ITEM_CONTEXT(id_itp,enp->en_decl_icp);
	idp = ID_OF(enp->en_string);
	pop_item_context(QSP_ARG  id_itp);

#ifdef CAUTIOUS
	if( idp == NO_IDENTIFIER ){
		sprintf(error_string,"CAUTIOUS:  missing id obj %s",enp->en_string);
		ERROR1(error_string);
	}
	if( ! IS_REFERENCE(idp) ){
		sprintf(error_string,"CAUTIOUS:  reeval_decl_stat:  identifier %s does not refer to an object!?",
			idp->id_name);
		ERROR1(error_string);
	}
#endif /* CAUTIOUS */

	dp=idp->id_refp->ref_dp;

#ifdef CAUTIOUS
	if( dp == NO_OBJ ){
		NODE_ERROR(enp);
		sprintf(error_string,"CAUTIOUS:  expected to find uk obj %s!?",
			enp->en_string);
		ERROR1(error_string);
	}
#endif /* CAUTIOUS */

	/* the size may not be unknown, if we were able to determine
	 * it's size during the first scan, .e.g. LOAD, or a known obj
	 */
	if( ! UNKNOWN_SHAPE(&dp->dt_shape) ){
		return;
	}

	if( idp->id_dobj_icp != CONTEXT_LIST(dobj_itp)->l_head->n_data ){
		context_pushed=1;
		PUSH_ITEM_CONTEXT(dobj_itp,idp->id_dobj_icp);
	} else context_pushed=0;

	delvec(QSP_ARG  dp);

	idp->id_refp->ref_dp = finish_obj_decl(QSP_ARG  enp,&dimset,prec,decl_flags);	/* reeval_decl_stat */

	if( context_pushed )
		pop_item_context(QSP_ARG  dobj_itp);

} /* end reeval_decl_stat */



/*
 * Two ways to call eval_ptr_ref:
 * when a ptr is dereferenced, or appears on the RHS, it must be set!
 */

Identifier *eval_ptr_ref(QSP_ARG_DECL Vec_Expr_Node *enp,int expect_ptr_set)
{
	Identifier *idp;

	eval_enp = enp;

	switch(enp->en_code){
		case T_EQUIVALENCE:
			idp = EVAL_OBJ_ID(enp);
			return(idp);
			break;

		case T_CALLFUNC:		/* a function that returns a pointer */
			return( EXEC_REFFUNC(enp) );

		case T_REFERENCE:
			idp = EVAL_OBJ_ID(enp->en_child[0]);
#ifdef CAUTIOUS
			if( idp == NO_IDENTIFIER ){
				NODE_ERROR(enp);
				DUMP_TREE(enp);
				ERROR1("CAUTIOUS:  eval_ptr_ref:  missing reference target");
			}
#endif	/* CAUTIOUS */
			return( idp );

		case T_UNDEF:
			return(NO_IDENTIFIER);

		case T_POINTER:		/* eval_ptr_ref */
		case T_STR_PTR:		/* eval_ptr_ref */
			idp = GET_ID(enp->en_string);
#ifdef CAUTIOUS
			if( idp==NO_IDENTIFIER ){
				NODE_ERROR(enp);
				sprintf(error_string,"CAUTIOUS:  missing identifier object (%s) #1!?",enp->en_string);
				WARN(error_string);
				return(idp);
			}
#endif /* CAUTIOUS */

			/* BUG this is not an error if the ptr is on the left hand side... */
			if( executing && expect_ptr_set ){
				if( IS_POINTER(idp) && !POINTER_IS_SET(idp) ){
					NODE_ERROR(enp);
					sprintf(error_string,"object pointer \"%s\" used before value is set",idp->id_name);
					advise(error_string);
				} else if( IS_STRING_ID(idp) && !STRING_IS_SET(idp) ){
					NODE_ERROR(enp);
					sprintf(error_string,"string pointer \"%s\" used before value is set",idp->id_name);
					advise(error_string);
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

#ifdef CAUTIOUS
	if( idp==NO_IDENTIFIER ){
		NODE_ERROR(enp);
		WARN("CAUTIOUS:  missing pointer identifier!?");
		return(NO_IDENTIFIER);
	}
	if( ! IS_POINTER(idp) ){
		NODE_ERROR(enp);
		sprintf(error_string,
			"CAUTIOUS:  eval_obj_ref:  id %s does not refer to a pointer!?",
			idp->id_name);
		WARN(error_string);
		return(NO_IDENTIFIER);
	}
#endif /* CAUTIOUS */

	if( ! POINTER_IS_SET(idp) )
		return(NO_IDENTIFIER);

	return(idp);
}

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

	switch(enp->en_code){
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
#ifdef CAUTIOUS
			if( dp == NO_OBJ ){
				NODE_ERROR(enp);
				sprintf(error_string,
						"CAUTIOUS:  eval_obj_id %s:  missing object!?",node_desc(enp));
				WARN(error_string);
				break;
			}
#endif /* CAUTIOUS */

			/* now make an identifier to go with this thing */
			idp = make_named_reference(QSP_ARG  dp->dt_name);
			idp->id_refp->ref_dp = dp;
			idp->id_shape = dp->dt_shape;
			return(idp);

		case T_OBJ_LOOKUP:
			s = EVAL_STRING(enp->en_child[0]);
			goto find_obj;

		case T_STATIC_OBJ:			/* eval_obj_id */
			s=enp->en_dp->dt_name;
			goto find_obj;

		case T_DYN_OBJ:				/* eval_obj_id */
			s=enp->en_string;
			/* fall-thru */
find_obj:
			idp = ID_OF(s);
#ifdef CAUTIOUS
			if( idp == NO_IDENTIFIER ){
				sprintf(error_string,"CAUTIOUS:  eval_obj_id:  missing identifier %s",s);
				ERROR1(error_string);
			}
			if( ! IS_REFERENCE(idp) ){
				sprintf(error_string,"CAUTIOUS:  eval_obj_id %s:  identifier is not a reference",
					idp->id_name);
				ERROR1(error_string);
			}
#endif /* CAUTIOUS */
			return(idp);

		case T_STRING:
			/* make a local string name */
			{
				Dimension_Set ds;
advise("eval_obj_id making a string object");
				STRING_DIMENSION(ds,strlen(enp->en_string)+1);
				dp=make_local_dobj(QSP_ARG  &ds,PREC_STR);
				if( dp == NO_OBJ ){
					WARN("unable to make temporary object");
					return(NO_IDENTIFIER);
				}
				strcpy((char *)dp->dt_data,enp->en_string);
LONGLIST(dp);
				idp = make_named_reference(QSP_ARG  dp->dt_name);
				idp->id_refp->ref_typ = STR_REFERENCE;
				idp->id_refp->ref_dp = dp;
				if( idp == NO_IDENTIFIER ){
					ERROR1("error making identifier for temp string obj");
				}
				return(idp);
			}

		default:
			MISSING_CASE(enp,"eval_obj_id");
			break;
	}
	return(NO_IDENTIFIER);
}

static Data_Obj *eval_subvec(QSP_ARG_DECL  Data_Obj *dp, index_t index, index_t i2 )
{
	Dimension_Set dimset;
	index_t offsets[N_DIMENSIONS];
	char newname[LLEN];
	Data_Obj *dp2;
	int i;

	dimset=dp->dt_type_dimset;
	for(i=0;i<N_DIMENSIONS;i++){
		offsets[i]=0;
	}
	dimset.ds_dimension[ dp->dt_maxdim ] = i2+1-index;
	offsets[ dp->dt_maxdim ] = index;
	sprintf(newname,"%s[%d:%d]",dp->dt_name,index,i2);
	dp2=DOBJ_OF(newname);
	if( dp2 != NO_OBJ ) return(dp2);

	dp2=mk_subseq(QSP_ARG  newname,dp,offsets,&dimset);
	if( dp2 == NO_OBJ ) return(dp2);
	dp2->dt_maxdim = dp->dt_maxdim-1;
	return(dp2);
}

static Data_Obj *eval_subscript1(QSP_ARG_DECL Data_Obj *dp, Vec_Expr_Node *enp)
{
	index_t index,index2;
	Data_Obj *dp2;

	if( enp->en_code == T_RANGE ){
		/* T_RANGE has 3 children, and is used to specify subsamping:
		 * start : end : inc
		 */
		WARN("eval_subscript1:  Sorry, not sure how to handle T_RANGE node");
		return(NO_OBJ);
	} else if( enp->en_code == T_RANGE2 ){
		index = EVAL_INT_EXP(enp->en_child[0]);
		index2 = EVAL_INT_EXP(enp->en_child[1]);
		/* Now we need to make a subvector */
		dp2 = eval_subvec(QSP_ARG  dp,index-1,index2-1) ;
		return(dp2);
	}

	/* index = EVAL_INT_EXP(enp->en_child[1]); */
	index = EVAL_FLT_EXP(enp);
sprintf(error_string,"eval_subscript1:  index is %d",index);
advise(error_string);

	/* d_subscript fails if the index is too large,
	 * but in matlab we want to automagically make the array larger
	 */
	insure_object_size(QSP_ARG  dp,index);

	dp2 = D_SUBSCRIPT(dp,index);
	return( dp2 );
}

Data_Obj *eval_obj_ref(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp,*dp2;
	index_t index,i2;
	index_t offsets[N_DIMENSIONS];
	Dimension_Set dimset;
	int i;
	char newname[LLEN];
	const char *s;
	Identifier *idp;

	eval_enp = enp;

	switch(enp->en_code){
		case T_EQUIVALENCE:		/* eval_obj_ref() */
			if( UNKNOWN_SHAPE(enp->en_shpp) ){
				RESOLVE_TREE(enp,NO_VEXPR_NODE);
			}
			if( UNKNOWN_SHAPE(enp->en_shpp) ){
				NODE_ERROR(enp);
				WARN("unable to determine shape of equivalence");
DUMP_TREE(enp);
				return(NO_OBJ);
			}
			{
			Data_Obj *parent_dp;

			/* we assume that the shape is known... */
			parent_dp = EVAL_OBJ_REF(enp->en_child[0]);
			if( parent_dp == NO_OBJ ){
				NODE_ERROR(enp);
				WARN("invalid parent object for equivalence");
				return(NO_OBJ);
			}
			dp = make_equivalence(QSP_ARG  localname(), parent_dp,&enp->en_shpp->si_type_dimset,enp->en_decl_prec);
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
			dp=make_local_dobj(QSP_ARG   &enp->en_child[0]->en_shpp->si_type_dimset, enp->en_prec);
			ASSIGN_OBJ_FROM_LIST(dp,enp->en_child[0],0);
			dp->dt_flags |= DT_ASSIGNED;
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
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			if( dp == NO_OBJ ) return(dp);

			/* In matlab, we can have a list of indices inside
			 * the paren's...  We need to know if the list trees
			 * grow to the left or right??
			 * SHould this code be matlab only? BUG?
			 */
			while( enp->en_child[1]->en_code == T_INDEX_LIST ){
				enp=enp->en_child[1];
				dp2 = EVAL_SUBSCRIPT1(dp,enp->en_child[0]);
				if( dp2 == NO_OBJ ){
					return(dp2);
				}
				dp=dp2;
			}
			/* BUG doesn't enforce reference to an existing object!? */
			dp2=EVAL_SUBSCRIPT1(dp,enp->en_child[1]);
			return(dp2);
			break;

		case T_PREINC:			/* eval_obj_ref */
		case T_PREDEC:
		case T_POSTINC:
		case T_POSTDEC:
			return( EVAL_OBJ_REF( enp->en_child[0] ) );

		case T_DEREFERENCE:	/* eval_obj_ref */
			/* runtime resolution, we may not be able to do this until ptrs have been assigned */
			if( UNKNOWN_SHAPE(enp->en_child[0]->en_shpp) ){
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"eval_obj_ref:  last ditch attempt at runtime resolution of %s",node_desc(enp->en_child[0]));
advise(error_string);
DUMP_TREE(enp);
}
#endif /* DEBUG */
				/*
				resolve_one_uk_node(enp->en_child[0]);
				*/
				RESOLVE_TREE(enp->en_child[0],NO_VEXPR_NODE);
#ifdef DEBUG
if( debug & resolve_debug ){
sprintf(error_string,"eval_obj_ref:  after last ditch attempt at runtime resolution of %s",node_desc(enp->en_child[0]));
advise(error_string);
DUMP_TREE(enp);
}
#endif /* DEBUG */
			}

			idp = GET_SET_PTR(enp->en_child[0]);
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

#ifdef CAUTIOUS
			if( ! IS_POINTER(idp) ){
				sprintf(error_string,"CAUTIOUS:  eval_obj_ref:  identifier %s isn't a pointer!?",idp->id_name);
				ERROR1(error_string);
			}
			if( idp->id_ptrp->ptr_refp == NO_REFERENCE ){
				sprintf(error_string,"CAUTIOUS:  eval_obj_ref:  target of pointer %s isn't set!?",idp->id_name);
				ERROR1(error_string);
			}
#endif /* CAUTIOUS */

			return(idp->id_ptrp->ptr_refp->ref_dp);

		case T_OBJ_LOOKUP:
			s=EVAL_STRING(enp->en_child[0]);
			if( s == NULL ) return(NO_OBJ);
			dp=get_id_obj(QSP_ARG  s,enp);
			return(dp);

		case T_UNDEF:
			return(NO_OBJ);

		case T_REFERENCE:
			return( EVAL_OBJ_REF(enp->en_child[0]) );

		case T_STATIC_OBJ:		/* eval_obj_ref() */
			return(enp->en_dp);

		case T_DYN_OBJ:		/* eval_obj_ref */
			return( get_id_obj(QSP_ARG  enp->en_string,enp) );

		case T_CURLY_SUBSCR:				/* eval_obj_ref */
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			if( dp == NO_OBJ ) return(dp);
			index = EVAL_INT_EXP(enp->en_child[1]);
			dp=C_SUBSCRIPT(dp,index);
			return(dp);

		case T_SQUARE_SUBSCR:				/* eval_obj_ref */
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			if( dp == NO_OBJ ) return(dp);
			/* Before we evaluate the subscript as an integer, check and
			 * see if it's a vector...
			 */
			if( SCALAR_SHAPE(enp->en_child[1]->en_shpp) ){
				index = EVAL_INT_EXP(enp->en_child[1]);
				dp = D_SUBSCRIPT(dp,index);
				return( dp );
			} else {
				NODE_ERROR(enp);
				WARN("eval_obj_ref:  vector indices are not allowed");
			}
			break;

		case T_SUBSAMP:					/* eval_obj_ref */
			{
			Dimension_Set ds;
			index_t offsets[N_DIMENSIONS]={0,0,0,0,0};
			incr_t incrs[N_DIMENSIONS]={1,1,1,1,1};
			incr_t inc;
			char tmp_name[LLEN];
			Data_Obj *sub_dp;

			dp = EVAL_OBJ_REF(enp->en_child[0]);
			if( dp == NO_OBJ ) return(dp);

			/* is this matlab order??? */
			index = EVAL_INT_EXP(enp->en_child[1]->en_child[0]);	/* start */
			if( mode_is_matlab ){
				/* start : inc : end */
				inc = EVAL_INT_EXP(enp->en_child[1]->en_child[1]);
				i2 = EVAL_INT_EXP(enp->en_child[1]->en_child[2]);	/* end */
			} else {
				/* start : end : inc */
				i2 = EVAL_INT_EXP(enp->en_child[1]->en_child[1]);	/* end */
				inc = EVAL_INT_EXP(enp->en_child[1]->en_child[2]);
			}

			sprintf(tmp_name,"%s[%d:%d:%d]",dp->dt_name,index,inc,i2);
			sub_dp = DOBJ_OF(tmp_name);
			if( sub_dp != NO_OBJ )
				return(sub_dp);		/* already exists */

			ds = dp->dt_type_dimset;
			ds.ds_dimension[dp->dt_maxdim]=1+floor((i2-index)/inc);	/* BUG assumes not reversed */
			offsets[dp->dt_maxdim] = index;
			incrs[dp->dt_maxdim] = inc;
			/* If we have referred to this before, the object may still exist */
			sub_dp = make_subsamp(QSP_ARG  tmp_name,dp,&ds,offsets,incrs);

			if( sub_dp == NO_OBJ ) return( sub_dp );
			sub_dp->dt_maxdim --;
			/* BUG?  make sure not less than mindim? */
			return( sub_dp );
			}
			break;
		case T_SUBVEC:					/* eval_obj_ref */
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			if( dp == NO_OBJ ) return(dp);

			if( enp->en_child[1] == NO_VEXPR_NODE )
				index = 0;	/* first element by default */
			else
				index = EVAL_INT_EXP(enp->en_child[1]);

			if( enp->en_child[2] == NO_VEXPR_NODE )
				i2 = dp->dt_type_dim[dp->dt_maxdim] - 1;	/* last elt. */
			else
				i2 = EVAL_INT_EXP(enp->en_child[2]);

			return( eval_subvec(QSP_ARG  dp,index,i2) );
			break;
		case T_CSUBVEC:					/* eval_obj_ref */
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			if( dp == NO_OBJ ) return(dp);
			index = EVAL_INT_EXP(enp->en_child[1]);
			i2 = EVAL_INT_EXP(enp->en_child[2]);
			dimset=dp->dt_type_dimset;
			for(i=0;i<N_DIMENSIONS;i++){
				offsets[i]=0;
			}
			dimset.ds_dimension[ dp->dt_mindim ] = i2+1-index;
			offsets[ dp->dt_mindim ] = index;
			sprintf(newname,"%s{%d:%d}",dp->dt_name,index,i2);
			dp2=DOBJ_OF(newname);
			if( dp2 != NO_OBJ ) return(dp2);

			dp2=mk_subseq(QSP_ARG  newname,dp,offsets,&dimset);
			dp2->dt_mindim = dp->dt_mindim+1;
			return(dp2);
			break;
		case T_REAL_PART:
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			if( dp == NO_OBJ ) return(dp);
			/* BUG make sure that the object is commplex! */
			return( C_SUBSCRIPT(dp,0) );
			break;
		case T_IMAG_PART:
			dp = EVAL_OBJ_REF(enp->en_child[0]);
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

	if( UNKNOWN_SHAPE(&dp->dt_shape) ){
		/* does the rhs have a shape??? */
		/* Why haven't we resolved when we are here? */
		WARN("assign_obj_from_list:  LHS has unknown shape!?");
	}

/*
sprintf(error_string,"assign_obj_from_list  dp = %s, enp = %s, index = %d",
dp->dt_name,node_desc(enp),index);
advise(error_string);
LONGLIST(dp);
DUMP_TREE(enp);
*/
	switch(enp->en_code){
		case T_TYPECAST:			/* assign_obj_from_list */
			/* do we need to do anything??? BUG? */
			i1=ASSIGN_OBJ_FROM_LIST(dp,enp->en_child[0],index);
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
			i1=ASSIGN_OBJ_FROM_LIST(dp,enp->en_child[0],0);
			/* why return 1 and not i1??? */
			return(1);
			break;

		case T_ROW_LIST:			/* assign_obj_from_list */
			/* HERE is where we need to be subscripting dp->dt_parent... */
			/* Don't subscript if the child is another ROW_LIST node */
			/* If we knew that the tree grew to the right or the left,
			 * we could eliminate the child tests - I am so lazy!
			 */
			if( enp->en_child[0]->en_code == T_ROW_LIST ){
				i1=ASSIGN_OBJ_FROM_LIST(dp,enp->en_child[0],index);
			} else {
				sub_dp = D_SUBSCRIPT(dp,index);
				i1=ASSIGN_OBJ_FROM_LIST(sub_dp,enp->en_child[0],index);
				delvec(QSP_ARG  sub_dp);
			}


			if( enp->en_child[1]->en_code == T_ROW_LIST ){
				i2=ASSIGN_OBJ_FROM_LIST(dp,enp->en_child[1],index+i1);
			} else {
				sub_dp = D_SUBSCRIPT(dp,index+i1);
				i2=ASSIGN_OBJ_FROM_LIST(sub_dp,enp->en_child[1],index+i1);
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
			if( enp->en_child[0]->en_code == T_COMP_LIST ){
				i1=ASSIGN_OBJ_FROM_LIST(dp,enp->en_child[0],index);
			} else {
				sub_dp = C_SUBSCRIPT(dp,index);
				i1=ASSIGN_OBJ_FROM_LIST(sub_dp,enp->en_child[0],index);
				delvec(QSP_ARG  sub_dp);
			}


			if( enp->en_child[1]->en_code == T_COMP_LIST ){
				i2=ASSIGN_OBJ_FROM_LIST(dp,enp->en_child[1],index+i1);
			} else {
				sub_dp = C_SUBSCRIPT(dp,index+i1);
				i2=ASSIGN_OBJ_FROM_LIST(sub_dp,enp->en_child[1],index+i1);
				delvec(QSP_ARG  sub_dp);
			}
			return(i1+i2);
			break;

		case T_LIT_DBL:			/* assign_obj_from_list */
			dbl_to_scalar(&sval,enp->en_dblval,MACHINE_PREC(dp));
assign_literal:
			if( ! IS_SCALAR(dp) ){
				NODE_ERROR(enp);
				sprintf(error_string,
	"assign_obj_from_list:  %s[%d] is not a scalar",dp->dt_name,
					index);
				WARN(error_string);
				return(1);
			}
			assign_scalar(QSP_ARG  dp,&sval);
			return(1);
			break;

		case T_LIT_INT:				/* assign_obj_from_list */
			int_to_scalar(&sval,enp->en_intval,MACHINE_PREC(dp));
			goto assign_literal;

		ALL_SCALAR_FUNCTION_CASES
		ALL_SCALAR_BINOP_CASES
			/* we allow arbitrary expressions within braces. */
			dval = EVAL_FLT_EXP(enp);
			dbl_to_scalar(&sval,dval,MACHINE_PREC(dp));
			goto assign_literal;

		case T_STATIC_OBJ:	/* assign_obj_from_list */
		case T_DYN_OBJ:		/* assign_obj_from_list */
		case T_VS_FUNC:
		case T_VV_FUNC:
			src_dp = EVAL_OBJ_EXP(enp,NO_OBJ);
			if( src_dp==NO_OBJ){
				NODE_ERROR(enp);
				sprintf(error_string,
			"assign_obj_from_list:  error evaluating RHS");
				WARN(error_string);
				return(0);
			}
			/* do we need to make sure they are the same size??? */
			convert(QSP_ARG  dp,src_dp);
			return(1);
			break;

		default:
			MISSING_CASE(enp,"assign_obj_from_list");
			break;
	}
WARN("assign_obj_from_list returning 0!?");
	return(0);
}

static void constant_bitmap(Data_Obj *dp,u_long lval)
{
	u_long *wp;
	int n_words;

	/* BUG here we assume the bitmap is contiguous */

	/* BUG what about padding? */
	n_words = (dp->dt_n_type_elts + BITS_PER_BITMAP_WORD - 1 ) / BITS_PER_BITMAP_WORD;
	wp = (u_long *)dp->dt_data;
	while(n_words--) *wp++ = lval;
}


static Data_Obj * complement_bitmap(QSP_ARG_DECL  Data_Obj *dp)
{
	u_long *wp;
	int n_words;
	static u_long complement_bits=0;

	/* BUG here we assume the bitmap is contiguous */
#ifdef CAUTIOUS
	if( ! IS_CONTIGUOUS(dp) ){
		LONGLIST(dp);
		sprintf(error_string,"complement_bitmap:  CAUTIOUS:  arg %s is not contiguous",dp->dt_name);
		ERROR1(error_string);
	}
#endif /* CAUTIOUS */

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

#ifdef CAUTIOUS
		if( new_dp == NO_OBJ ){
			sprintf(error_string,"CAUTIOUS:  complement_bitmap:  Unable to create object %s",s);
			ERROR1(error_string);
		}
#endif /* CAUTIOUS */

		dp_copy(QSP_ARG  new_dp,dp);
		dp = new_dp;
	}

	/* BUG what about padding? */
	n_words = (dp->dt_n_type_elts + BITS_PER_BITMAP_WORD - 1 ) / BITS_PER_BITMAP_WORD;
	/* BUG check offset (bit0) ... */
	wp = (u_long *)dp->dt_data;
	while(n_words--){
		*wp ^= complement_bits;
		wp++;
	}
	return(dp);
}

static Data_Obj *create_bitmap( QSP_ARG_DECL  Dimension_Set *dsp )
{
	Dimension_Set dimset;
	Data_Obj *bmdp;
	int i;

	for(i=0;i<N_DIMENSIONS;i++)
		dimset.ds_dimension[i] = dsp->ds_dimension[i];

	/* BUG? the bitmap code in veclib assumes that all the bits
	 * run into one another (i.e., no padding of rows
	 * as is done here.
	 *
	 * Perhaps this is ok, it just wastes memory...
	 */

	bmdp = make_local_dobj(QSP_ARG  &dimset,PREC_BIT);
	return(bmdp);
}

static Data_Obj *dup_bitmap(QSP_ARG_DECL  Data_Obj *dp)
{
#ifdef CAUTIOUS
	if( UNKNOWN_SHAPE(&dp->dt_shape) ){
		sprintf(error_string,"dup_bitmap:  can't dup from unknown shape object %s",dp->dt_name);
		ERROR1(error_string);
	}
#endif /* CAUTIOUS */

	return( create_bitmap(QSP_ARG  &dp->dt_type_dimset ) );
}

/* Like dup_bitmap, but we need to use this version w/ vv_bitmap because
 * the two operands might have different shape (outer op).
 * Can we use get_mating_shape?
 */

static Data_Obj *dup_bitmap2(QSP_ARG_DECL  Data_Obj *dp1, Data_Obj *dp2)
{
	Shape_Info *shpp;

	shpp = product_shape(&dp1->dt_shape,&dp2->dt_shape);
	if( shpp == NO_SHAPE ) return(NO_OBJ);

	return( create_bitmap(QSP_ARG  &shpp->si_type_dimset ) );
}

/* vs_bitmap:  vsm_lt etc. */

static Data_Obj * vs_bitmap(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *dp,Scalar_Value *svp,Vec_Func_Code code)
{
	Data_Obj *bmdp;
	Vec_Obj_Args oargs;
	static Data_Obj *vsbm_sclr_dp=NO_OBJ;

#ifdef CAUTIOUS
	switch(code){
		case FVSMLT:
		case FVSMGT:
		case FVSMLE:
		case FVSMGE:
		case FVSMNE:
		case FVSMEQ:
			break;
		default:
			sprintf(error_string,
				"CAUTIOUS:  unexpected code (%d) in vs_bitmap",code);
			WARN(error_string);
			return(NO_OBJ);
			break;
	}
#endif /* CAUTIOUS */

	if( dst_dp == NO_OBJ ){
		bmdp = dup_bitmap(QSP_ARG  dp);
#ifdef CAUTIOUS
	if( bmdp == NO_OBJ ){
		sprintf(error_string,"CAUTIOUS:  vs_bitmap:  unable to dup bitmap for obj %s",dp->dt_name);
		ERROR1(error_string);
	}
#endif /* CAUTIOUS */
	}
	else
		bmdp = dst_dp;

	/* set_bitmap(bmdp); */
	setvarg2(&oargs,bmdp,dp);	/* Need bitmap dest... */

	vsbm_sclr_dp=check_global_scalar(QSP_ARG  "vs_bitmap_scalar",dp,vsbm_sclr_dp);
#ifdef CAUTIOUS
	if( vsbm_sclr_dp == NO_OBJ ){
		sprintf(error_string,"CAUTIOUS:  vs_bitmap:  check_global_scalar failed");
		ERROR1(error_string);
	}
#endif /* CAUTIOUS */
	assign_scalar(QSP_ARG  vsbm_sclr_dp,svp);
	oargs.oa_s1=vsbm_sclr_dp;
	oargs.oa_svp[0] = (Scalar_Value *)vsbm_sclr_dp->dt_data;

	oargs.oa_bmap=bmdp;		/* why was this commented out? */
	/* oargs.oa_bmap=bmdp; */

	if( perf_vfunc(QSP_ARG  code,&oargs) )
		return(NO_OBJ);

	return(bmdp);
	/* BUG? when do we delete the bitmap??? */

} /* end vs_bitmap() */

static Data_Obj * vv_bitmap(QSP_ARG_DECL  Data_Obj *dst_dp,Data_Obj *dp1,Data_Obj *dp2,Vec_Func_Code code)
{
	Data_Obj *bmdp;
	Vec_Obj_Args oargs;

#ifdef CAUTIOUS
	switch(code){
		case FVVMLT:
		case FVVMGT:
		case FVVMLE:
		case FVVMGE:
		case FVVMNE:
		case FVVMEQ:
			break;
		default:
			WARN("CAUTIOUS:  unexpected code in vv_bitmap");
			break;
	}
#endif /* CAUTIOUS */
	if( dst_dp != NO_OBJ )
		bmdp = dst_dp;
	else {
		bmdp = dup_bitmap2(QSP_ARG  dp1,dp2);	/* might be an outer op */
	}

	setvarg3(&oargs,bmdp,dp1,dp2);
	if( perf_vfunc(QSP_ARG  code,&oargs) < 0 )
		return(NO_OBJ);

	return(bmdp);
}

/* Call eval_bitmap with dst_dp = NO_OBJ for automatic allocation
 *
 * For a compound test such as a<b || c>d, we would allocate a temp
 * bitmap for the first comparison, another for the second comparison,
 * but reuse one of these as the destination object for the final
 * comparison.  But if we have cached bitmaps b1 || b2 then we
 * have to allocate a new bitmap to hold the result...
 *
 * The allocation is done later, in vs_bitmap, vv_bitmap etc.
 */

static Data_Obj *eval_bitmap(QSP_ARG_DECL Data_Obj *dst_dp, Vec_Expr_Node *enp)
{
	Data_Obj *bm_dp1,*bm_dp2,*dp,*dp2;
	long ival;

	eval_enp = enp;

	switch( enp->en_code ){
		/* ALL_OBJREF_CASES??? */
		case T_STATIC_OBJ:		/* eval_bitmap */
		case T_DYN_OBJ:			/* eval_bitmap */
			dp = EVAL_OBJ_REF(enp);
			return(dp);
			break;

		case T_BOOL_AND:		/* eval_bitmap */
			if( SCALAR_SHAPE(enp->en_child[0]->en_shpp) ){
				ival = EVAL_INT_EXP(enp->en_child[0]);
				bm_dp1 = EVAL_BITMAP(dst_dp,enp->en_child[1]);
				if( !ival )
					constant_bitmap(bm_dp1,0L);
				return(bm_dp1);
			} else if( SCALAR_SHAPE( enp->en_child[1]->en_shpp ) ){
				ival = EVAL_INT_EXP(enp->en_child[1]);
				bm_dp1 = EVAL_BITMAP(dst_dp,enp->en_child[0]);
				if( !ival )
					constant_bitmap(bm_dp1,0L);
				return(bm_dp1);
			} else {
				dst_dp =
				bm_dp1 = EVAL_BITMAP(dst_dp,enp->en_child[0]);
				bm_dp2 = EVAL_BITMAP(NO_OBJ,enp->en_child[1]);
				if( do_vvfunc(QSP_ARG  dst_dp,bm_dp1,bm_dp2,FVAND) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating bitmap");
					return(NO_OBJ);
				}
				return(dst_dp);
			}
			break;
		case T_BOOL_OR:		/* eval_bitmap */
			if( SCALAR_SHAPE(enp->en_child[0]->en_shpp) ){
				ival = EVAL_INT_EXP(enp->en_child[0]);
				bm_dp1 = EVAL_BITMAP(dst_dp,enp->en_child[1]);
				if( ival )
					constant_bitmap(bm_dp1,0xffffffff);
				return(bm_dp1);
			} else if( SCALAR_SHAPE( enp->en_child[1]->en_shpp ) ){
				ival = EVAL_INT_EXP(enp->en_child[1]);
				bm_dp1 = EVAL_BITMAP(dst_dp,enp->en_child[0]);
				if( ival )
					constant_bitmap(bm_dp1,0xffffffff);
				return(bm_dp1);
			} else {
				dst_dp =
				bm_dp1 = EVAL_BITMAP(dst_dp,enp->en_child[0]);
				bm_dp2 = EVAL_BITMAP(NO_OBJ,enp->en_child[1]);
				if( do_vvfunc(QSP_ARG  dst_dp,bm_dp1,bm_dp2,FVOR) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating bitmap");
					return(NO_OBJ);
				}
				return(dst_dp);
			}
			break;
		case T_BOOL_XOR:		/* eval_bitmap */
			if( SCALAR_SHAPE(enp->en_child[0]->en_shpp) ){
				ival = EVAL_INT_EXP(enp->en_child[0]);
				bm_dp1 = EVAL_BITMAP(dst_dp,enp->en_child[1]);
				if( ival ){
					bm_dp1 = complement_bitmap(QSP_ARG  bm_dp1);
				}
				return(bm_dp1);
			} else if( SCALAR_SHAPE( enp->en_child[1]->en_shpp ) ){
				ival = EVAL_INT_EXP(enp->en_child[1]);
				bm_dp1 = EVAL_BITMAP(dst_dp,enp->en_child[0]);
				if( ival ){
					bm_dp1 = complement_bitmap(QSP_ARG  bm_dp1);
				}
				return(bm_dp1);
			} else {
				dst_dp =
				bm_dp1 = EVAL_BITMAP(dst_dp,enp->en_child[0]);
				bm_dp2 = EVAL_BITMAP(NO_OBJ,enp->en_child[1]);
				if( do_vvfunc(QSP_ARG  dst_dp,bm_dp1,bm_dp2,FVXOR) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating bitmap");
					return(NO_OBJ);
				}
				return(dst_dp);
			}
			break;
		case T_BOOL_NOT:		/* eval_bitmap */
			bm_dp1 = EVAL_BITMAP(dst_dp,enp->en_child[0]);
			bm_dp1 = complement_bitmap(QSP_ARG  bm_dp1);
			return(bm_dp1);
			break;

		ALL_NUMERIC_COMPARISON_CASES			/* eval_bitmap */

#ifdef CAUTIOUS
			if( SCALAR_SHAPE( enp->en_child[0]->en_shpp ) ){
		WARN("CAUTIOUS:  scalar comparison operand should have been swapped!?");
				return(NO_OBJ);
			}
#endif /* CAUTIOUS */
			if( SCALAR_SHAPE( enp->en_child[1]->en_shpp ) ){
				Scalar_Value sval;
				dp = EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
#ifdef CAUTIOUS
				VERIFY_DATA_TYPE(enp,ND_FUNC,"eval_bitmap")

				if( dp == NO_OBJ ){
					NODE_ERROR(enp);
					advise("CAUTIOUS:  missing object");
					return(NO_OBJ);
				}
#endif /* CAUTIOUS */
				EVAL_SCALAR(&sval,enp->en_child[1],dp->dt_prec);
				bm_dp1 = vs_bitmap(QSP_ARG  dst_dp,dp,&sval,enp->en_bm_code);
if( bm_dp1 == NO_OBJ ){
NODE_ERROR(enp);
sprintf(error_string,"bad vs_bitmap, %s",node_desc(enp));
ERROR1(error_string);
}
			} else {
				/* both vectors */
				dp = EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
				dp2 = EVAL_OBJ_EXP(enp->en_child[1],NO_OBJ);
				bm_dp1 = vv_bitmap(QSP_ARG  dst_dp,dp,dp2,enp->en_bm_code);
			}
			return(bm_dp1);
			break;

		default:
			MISSING_CASE(enp,"eval_bitmap");
			break;
	}
	return(NO_OBJ);
} /* end eval_bitmap() */

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

	if( enp->en_vfunc_code == FVVSSLCT ) is_bitmap=1;

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

	if( enp->en_child[1]->en_lhs_refs == 0 ){
		/* the right hand subtree makes no ref to the lhs,
		 * so we're ok
		 */
		if( is_bitmap ){
ADVISE("get_2_operands calling eval_bitmap #1");
			dp1=EVAL_BITMAP(NO_OBJ,enp->en_child[0]);
		} else if( dst_dp!=NO_OBJ && same_shape(enp->en_child[0]->en_shpp,&dst_dp->dt_shape) ){
			dp1=EVAL_OBJ_EXP(enp->en_child[0],dst_dp);
		} else {
			dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
		}

		if( dst_dp!=NO_OBJ && dp1!=dst_dp && same_shape(enp->en_child[1]->en_shpp,&dst_dp->dt_shape) ){
			dp2=EVAL_OBJ_EXP(enp->en_child[1],dst_dp);
		} else {
			dp2=EVAL_OBJ_EXP(enp->en_child[1],NO_OBJ);
		}
	} else if( enp->en_child[0]->en_lhs_refs == 0 ){
		/* the right hand subtree  refers to the lhs...
		 * but the left-hand subtree does not.
		 * we can proceed as above, but with r & l
		 * interchanged.
		 */
		if( dst_dp!=NO_OBJ && same_shape(enp->en_child[1]->en_shpp,&dst_dp->dt_shape) ){
			dp2=EVAL_OBJ_EXP(enp->en_child[1],dst_dp);
		} else {
			dp2=EVAL_OBJ_EXP(enp->en_child[1],NO_OBJ);
		}

		if( dp2!=dst_dp ){
			if( is_bitmap ){
ADVISE("get_2_operands calling eval_bitmap #2");
				dp1=EVAL_BITMAP(NO_OBJ,enp->en_child[0]);
			} else if( dst_dp!=NO_OBJ && same_shape(enp->en_child[0]->en_shpp,&dst_dp->dt_shape) ){
				dp1=EVAL_OBJ_EXP(enp->en_child[0],dst_dp);
			}
		} else {
			if( is_bitmap ){
ADVISE("get_2_operands calling eval_bitmap #3");
				dp1=EVAL_BITMAP(NO_OBJ,enp->en_child[0]);
			} else {
				dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			}
		}
	} else {
		/* Both sides refer to the lhs */
		if( is_bitmap ){
ADVISE("get_2_operands calling eval_bitmap #3");
			dp1=EVAL_BITMAP(NO_OBJ,enp->en_child[0]);
		} else
			dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);

		/* used to have dst_dp here, would have added a test here for shape match,
		 * but it seems that if this branch refers to the lhs then we probably don't
		 * want to use the destination object?
		 */

		dp2=EVAL_OBJ_EXP(enp->en_child[1],NO_OBJ);
	}

#ifdef CAUTIOUS
	if( dp1==NO_OBJ ){
		sprintf(error_string,"CAUTIOUS:  get_2_operands:  null operand on left subtree!?");
		WARN(error_string);
		DUMP_TREE(enp);
	}
	if( dp2 == NO_OBJ ){
		sprintf(error_string,"CAUTIOUS:  get_2_operands:  null operand on right subtree!?");
		WARN(error_string);
		DUMP_TREE(enp);
	}
#endif /* CAUTIOUS */

	*dpp1 = dp1;
	*dpp2 = dp2;
} /* end get_2_operands() */

static void eval_scalar(QSP_ARG_DECL Scalar_Value *svp, Vec_Expr_Node *enp, prec_t prec)
{
	eval_enp = enp;

	/* should we call eval_flt_exp for all??? */

	switch(prec&MACH_PREC_MASK){
		case PREC_SP:  svp->u_f = EVAL_FLT_EXP(enp); break;
		case PREC_DP:  svp->u_d = EVAL_FLT_EXP(enp); break;
		case PREC_BY:  svp->u_b = EVAL_INT_EXP(enp); break;
		case PREC_IN:  svp->u_s = EVAL_INT_EXP(enp); break;
		case PREC_DI:  svp->u_l = EVAL_INT_EXP(enp); break;
		case PREC_UDI:  svp->u_ul = (u_long) EVAL_INT_EXP(enp); break;
		case PREC_UIN:  svp->u_us = EVAL_INT_EXP(enp); break;
		case PREC_UBY:  svp->u_ub = EVAL_INT_EXP(enp); break;
		default:
			WARN("CAUTIOUS:  unhandled machine precision in eval_scalar()");
			break;
	}
}

/* Like convert(), but if destination is complex then do the right thing.
 */

static int c_convert(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *dp)
{
	if( IS_COMPLEX(dst_dp) && ! IS_COMPLEX(dp) ){
		Data_Obj *tmp_dp;

		tmp_dp = C_SUBSCRIPT(dst_dp,0);
		convert(QSP_ARG  tmp_dp,dp);

		tmp_dp = C_SUBSCRIPT(dst_dp,1);
		return( zero_dp(QSP_ARG  tmp_dp) );
	} else {
		/* Can we put an error check on convert??? */
		convert(QSP_ARG  dst_dp,dp);
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
sprintf(error_string,"eval_typecast:  dst_dp %s at 0x%lx",dst_dp->dt_name,(u_long)dst_dp);
advise(error_string);
}
*/

#ifdef CAUTIOUS
	if( enp->en_child[0]->en_shpp == NO_SHAPE ){
		NODE_ERROR(enp->en_child[0]);
		sprintf(error_string,"CAUTIOUS:  eval TYPECAST:  %s has no shape!?",
			node_desc(enp->en_child[0]));
		ERROR1(error_string);
	}
	if( dst_dp!= NO_OBJ && UNKNOWN_SHAPE(&dst_dp->dt_shape) ){
		NODE_ERROR(enp);
		sprintf(error_string,"CAUTIOUS:  eval_typecast %s:  destination Object %s has uknown shape!?",
			node_desc(enp),dst_dp->dt_name);
		ERROR1(error_string);
	}
#endif /* CAUTIOUS */

	/* It is not an error for the typecast to match the LHS -
	 * in fact it should!  compile_node may insert a typecast
	 * node to effect type conversion.
	 */

#ifdef CAUTIOUS
	if( dst_dp != NO_OBJ && dst_dp->dt_prec != enp->en_prec /* same as enp->en_intval */ ){
		NODE_ERROR(enp);
		sprintf(error_string,
	"CAUTIOUS:  eval_typecast:  %s precision %s does not match target %s precision %s",
			node_desc(enp),name_for_prec(enp->en_prec),dst_dp->dt_name,name_for_prec(dst_dp->dt_prec));
		WARN(error_string);
		advise("ignoring typecast");
		EVAL_OBJ_ASSIGNMENT(dst_dp,enp->en_child[0]);
		return(dst_dp);
	}
#endif /* CAUTIOUS */

	if( enp->en_intval == enp->en_child[0]->en_shpp->si_prec ){
		/* the object already has the cast precision */
		NODE_ERROR(enp);
		WARN("typecast redundant w/ rhs");
		EVAL_OBJ_ASSIGNMENT(dst_dp,enp->en_child[0]);
		return(dst_dp);
	}

	/* If the child node is an object, we simply do a conversion into the
	 * destination.  If it's an operator, we have to make a temporary object
	 * to hold the result, and then convert.
	 */

	switch(enp->en_child[0]->en_code){
		ALL_OBJREF_CASES
			/* dp=EVAL_OBJ_REF(enp->en_child[0]); */
			dp=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			if( dp != NO_OBJ ){
				if( dst_dp == NO_OBJ ){
					dst_dp=make_local_dobj(QSP_ARG  
						&enp->en_child[0]->en_shpp->si_type_dimset,
						enp->en_prec);
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
			MISSING_CASE(enp->en_child[0],"eval_typecast");
			/* missing_case calls dump_tree?? */
			DUMP_TREE(enp);

handle_it:
			/* We have been requested to convert
			 * to a different precision
			 */

			tmp_dp=make_local_dobj(QSP_ARG  
					&enp->en_child[0]->en_shpp->si_type_dimset,
					enp->en_child[0]->en_shpp->si_prec );

			EVAL_OBJ_ASSIGNMENT(tmp_dp,enp->en_child[0]);

			if( dst_dp == NO_OBJ )
				dst_dp=make_local_dobj(QSP_ARG  
					&enp->en_child[0]->en_shpp->si_type_dimset,
					enp->en_prec);

			if( c_convert(QSP_ARG  dst_dp,tmp_dp) < 0 ){
				NODE_ERROR(enp);
				WARN("error performing conversion");
			}
			delvec(QSP_ARG  tmp_dp);
			break;
	}
	return(dst_dp);
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
	Image_File *ifp;
	Scalar_Value sval,*svp;
	Vec_Obj_Args oargs;
	//int i;
	const char *s;

/*
*/
	eval_enp = enp;

	if( dp == NO_OBJ ){
advise("eval_obj_assignment returning (NULL target)");
DUMP_TREE(enp);
		return;	/* probably an undefined reference */
	}

#ifdef DEBUG
if( debug & eval_debug ){
sprintf(error_string,"eval_obj_assignment %s",dp->dt_name);
advise(error_string);
DUMP_TREE(enp);
}
#endif /* DEBUG */

	switch(enp->en_code){
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
ADVISE("eval_obj_assignment calling eval_bitmap");
			EVAL_BITMAP(dp,enp);
			break;

		case T_RANGE2:
			{
			double d1,d2;
			double delta;
			d1=EVAL_INT_EXP(enp->en_child[0]);
			d2=EVAL_INT_EXP(enp->en_child[1]);
			delta = (d2-d1)/(dp->dt_n_type_elts-1);
			easy_ramp2d(QSP_ARG  dp,d1,delta,0.0);
			}
			break;

		case T_STRING_LIST:
		case T_STRING:
#ifdef CAUTIOUS
			if( dp->dt_prec != PREC_CHAR ){
				WARN("CAUTIOUS:");
				NODE_ERROR(enp);
				sprintf(error_string,"LHS (%s, %s) does not have %s precision, but RHS is a string",
					dp->dt_name,name_for_prec(dp->dt_prec),name_for_prec(PREC_CHAR));
				advise(error_string);
				break;
			}
#endif /* CAUTIOUS */

			s = EVAL_STRING(enp);

#ifdef CAUTIOUS
			if( dp->dt_n_type_elts <= strlen(s) ){
				WARN("CAUTIOUS:");
				NODE_ERROR(enp);
				sprintf(error_string,"LHS (%s, %d) does not have space for RHS string",
					dp->dt_name,dp->dt_n_type_elts);
				advise(error_string);
				break;
			}
#endif /* CAUTIOUS */

			strcpy((char *)dp->dt_data,s);
			break;

		/* matlab */
#ifdef FOOBAR
		case T_ROWLIST:		/* eval_obj_assignment */
			/* rowlist trees grow down to the left, so we start with the bottom row
			 * and work up
			 */
advise("rowlist");
			i=enp->en_shpp->si_rows;
			/* But this child could be a matrix object? */
			ASSIGN_ROW(dp,i,enp->en_child[1]);
			/* child[0] is either a ROWLIST node, or a ROW */
			EVAL_OBJ_ASSIGNMENT(dp,enp->en_child[0]);
			break;
#endif

		case T_ROW:
advise("row!");
			/* do we need to subscript dp?? */
			if( dp->dt_rows > 1 ){
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
			return;

		case T_DILATE:
			dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			dilate(QSP_ARG  dp,dp1);
			break;
		case T_ERODE:
			dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
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
				int index;
				Scalar_Value sval;

				/* I don't get this AT ALL??? */
				index = EVAL_INT_EXP(enp->en_child[0]);

				index = index!=0 ? 1 : 2;

#ifdef CAUTIOUS
				if( ! SCALAR_SHAPE(enp->en_child[index]->en_shpp) ){
					NODE_ERROR(enp);
					sprintf(error_string,
				"CAUTIOUS:  eval_obj_assignment %s:  %s does not have scalar shape!?",
						node_desc(enp),node_desc(enp->en_child[index]));
					ERROR1(error_string);
				}
#endif /* CAUTIOUS */
				EVAL_SCALAR(&sval,enp->en_child[index],dp->dt_prec);
				ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			}
			break;
		case T_VS_S_CONDASS:		/* eval_obj_assignment */
			{
				int index;
				Scalar_Value sval;

				/* is a boolean expression and int expression? */
				index = EVAL_INT_EXP(enp->en_child[0]);

				index = index!=0 ? 1 : 2;

				if( index == 1 ){	/* first choice should be the vector */
#ifdef CAUTIOUS
					if( SCALAR_SHAPE(enp->en_child[index]->en_shpp) ){
						NODE_ERROR(enp);
						sprintf(error_string,
				"CAUTIOUS:  eval_obj_assignment %s:  %s does not have vector shape!?",
				node_desc(enp),node_desc(enp->en_child[index]));
						ERROR1(error_string);
					}
#endif /* CAUTIOUS */
					EVAL_OBJ_ASSIGNMENT(dp,enp->en_child[index]);
				} else {		/* second choice should be the scalar */
#ifdef CAUTIOUS
					if( ! SCALAR_SHAPE(enp->en_child[index]->en_shpp) ){
						NODE_ERROR(enp);
						sprintf(error_string,
				"CAUTIOUS:  eval_obj_assignment %s:  %s does not have scalar shape!?",
							node_desc(enp),node_desc(enp->en_child[index]));
						ERROR1(error_string);
					}
#endif /* CAUTIOUS */
					EVAL_SCALAR(&sval,enp->en_child[index],dp->dt_prec);
					ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
				}
			}
			break;

		case T_VV_S_CONDASS:			/* eval_obj_assignment */
			{
				int index;

				index = EVAL_INT_EXP(enp->en_child[0]);

				index = index!=0 ? 1 : 2;

#ifdef CAUTIOUS
				if( SCALAR_SHAPE(enp->en_child[index]->en_shpp) ){
					NODE_ERROR(enp);
					sprintf(error_string,
				"CAUTIOUS:  eval_obj_assignment %s:  %s does not have vector shape!?",
				node_desc(enp),node_desc(enp->en_child[index]));
					ERROR1(error_string);
				}
#endif /* CAUTIOUS */
				EVAL_OBJ_ASSIGNMENT(dp,enp->en_child[index]);
			}
			break;

		case T_SS_B_CONDASS: /* eval_obj_assignment */
			{
				Data_Obj *bm_dp;
				Scalar_Value sval2;
				static Data_Obj *s1dp=NO_OBJ,*s2dp=NO_OBJ;

				/* Neet to create a temp vector or bitmap,
				 * and then use the select vector function.
				 */

				bm_dp = EVAL_BITMAP(NO_OBJ,enp->en_child[0]);
				/* we need to know the type of the destination before
				 * we evaluate the scalars...
				 */
				EVAL_SCALAR(&sval,enp->en_child[1],dp->dt_prec);
				EVAL_SCALAR(&sval2,enp->en_child[2],dp->dt_prec);

				s1dp=check_global_scalar(QSP_ARG  "vss_scalar1",dp,s1dp);
				s2dp=check_global_scalar(QSP_ARG  "vss_scalar2",dp,s2dp);

				assign_scalar(QSP_ARG  s1dp,&sval);
				assign_scalar(QSP_ARG  s2dp,&sval2);

				setvarg1(&oargs,dp);
				oargs.oa_s1=s1dp;
				oargs.oa_svp[0] = (Scalar_Value *)s1dp->dt_data;
				oargs.oa_s2=s2dp;
				oargs.oa_svp[1] = (Scalar_Value *)s2dp->dt_data;

				oargs.oa_bmap=bm_dp;
				if( perf_vfunc(QSP_ARG  FVSSSLCT,&oargs) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VSS select operator");
				}
			}
			break;

		case T_VS_B_CONDASS:		/* eval_obj_assignment */
			{
				Data_Obj *bm_dp;
				static Data_Obj *sdp=NO_OBJ;;

				bm_dp = EVAL_BITMAP(NO_OBJ,enp->en_child[0]);
				dp2=EVAL_OBJ_EXP(enp->en_child[1],NO_OBJ);
				EVAL_SCALAR(&sval,enp->en_child[2],dp->dt_prec);

				sdp=check_global_scalar(QSP_ARG  "vvs_scalar",dp,sdp);
				assign_scalar(QSP_ARG  sdp,&sval);
				setvarg2(&oargs,dp,dp2);
				oargs.oa_s1=sdp;
				oargs.oa_svp[0] = (Scalar_Value *)sdp->dt_data;
				oargs.oa_bmap=bm_dp;

				if( perf_vfunc(QSP_ARG  FVVSSLCT,&oargs) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VVS select operator");
				}
			}
			break;
		case T_VV_B_CONDASS:		/* eval_obj_assignment */
			{
				Data_Obj *bm_dp;
				bm_dp = EVAL_BITMAP(NO_OBJ,enp->en_child[0]);
				dp2=EVAL_OBJ_EXP(enp->en_child[1],NO_OBJ);
				dp3=EVAL_OBJ_EXP(enp->en_child[2],NO_OBJ);

				setvarg3(&oargs,dp,dp2,dp3);
				oargs.oa_bmap=bm_dp;
				if( perf_vfunc(QSP_ARG  FVVVSLCT,&oargs) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VVV select operator");
				}
			}
			break;

		case T_VV_VV_CONDASS:		/* eval_obj_assignment */
			{
				dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
				dp2=EVAL_OBJ_EXP(enp->en_child[1],NO_OBJ);
				dp3=EVAL_OBJ_EXP(enp->en_child[2],NO_OBJ);
				dp4=EVAL_OBJ_EXP(enp->en_child[3],NO_OBJ);
				setvarg5(&oargs,dp,dp1,dp2,dp3,dp4);
				if( perf_vfunc(QSP_ARG  enp->en_vfunc_code, &oargs) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VV_VV conditional");
				}
			}
			break;
		case T_VV_VS_CONDASS:		/* eval_obj_assignment */
			{
				dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
				dp2=EVAL_OBJ_EXP(enp->en_child[1],NO_OBJ);
				dp3=EVAL_OBJ_EXP(enp->en_child[2],NO_OBJ);
				EVAL_SCALAR(&sval,enp->en_child[3],MACHINE_PREC(dp3));
				setvarg4(&oargs,dp,dp1,dp2,dp3);
				oargs.oa_svp[0] = &sval;
				if( perf_vfunc(QSP_ARG  enp->en_vfunc_code, &oargs) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VV_VV conditional");
				}
			}
			break;
		case T_VS_VV_CONDASS:		/* eval_obj_assignment */
			{
				dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
				EVAL_SCALAR(&sval,enp->en_child[1],MACHINE_PREC(dp1));
				dp2=EVAL_OBJ_EXP(enp->en_child[2],NO_OBJ);
				dp3=EVAL_OBJ_EXP(enp->en_child[3],NO_OBJ);
				setvarg4(&oargs,dp,dp1,dp2,dp3);
				oargs.oa_svp[0] = &sval;
				if( perf_vfunc(QSP_ARG  enp->en_vfunc_code, &oargs) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VV_VV conditional");
				}
			}
			break;
		case T_VS_VS_CONDASS:		/* eval_obj_assignment */
			{
				Scalar_Value sval2;

				dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
				EVAL_SCALAR(&sval,enp->en_child[1],MACHINE_PREC(dp1));
				dp2=EVAL_OBJ_EXP(enp->en_child[2],NO_OBJ);
				EVAL_SCALAR(&sval2,enp->en_child[3],MACHINE_PREC(dp2));
				setvarg3(&oargs,dp,dp1,dp2);
				/* The first scalar is the source */
				oargs.oa_svp[0] = &sval;
				oargs.oa_svp[1] = &sval2;
				if( perf_vfunc(QSP_ARG  enp->en_vfunc_code, &oargs) < 0 ){
					NODE_ERROR(enp);
					WARN("Error evaluating VV_VV conditional");
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

			dp1=EVAL_OBJ_REF(enp->en_child[0]);		/* indices */
			dp2=EVAL_OBJ_REF(enp->en_child[1]);		/* maxval */
			dp3=EVAL_OBJ_EXP(enp->en_child[2],NO_OBJ);	/* input */
			setvarg2(&oargs,dp1,dp3);
			oargs.oa_s1=dp2;				/* destination maxval */
			oargs.oa_s2=dp;					/* destination n */
			oargs.oa_svp[0] = (Scalar_Value *)dp2->dt_data;
			oargs.oa_svp[1] = (Scalar_Value *)dp->dt_data;
			if( perf_vfunc(QSP_ARG  FVMAXG,&oargs) < 0 ){
				NODE_ERROR(enp);
				WARN("Error evaluating max_times operator");
			}
			break;

		case T_RDFT:						/* eval_obj_assignment */
			dp1 = EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			fft2d(dp,dp1);
			break;

		case T_RIDFT:						/* eval_obj_assignment */
			dp1 = EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			ift2d(dp,dp1);
			break;

		case T_REDUCE:						/* eval_obj_assignment */
			dp1 = EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			reduce(QSP_ARG  dp,dp1);
			break;

		case T_ENLARGE:						/* eval_obj_assignment */
			dp1 = EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			enlarge(QSP_ARG  dp,dp1);
			break;

		case T_TYPECAST:		/* eval_obj_assignment */
			EVAL_TYPECAST(enp,dp);
			break;

		/* use tabled functions here???
		 * Or at least write a macro for the repeated code...
		 */

		case T_MINVAL:
			dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			setvarg2(&oargs,dp,dp1);
			vminv(&oargs);
			break;
		case T_MAXVAL:
			dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			setvarg2(&oargs,dp,dp1);
			vmaxv(&oargs);
			break;
		case T_SUM:				/* eval_obj_assignment */
			dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			setvarg2(&oargs,dp,dp1);
			vsum(&oargs);
			break;

		case T_LOAD:						/* eval_obj_assignment */
			s = EVAL_STRING(enp->en_child[0]);
			if( s == NULL ) return;

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
					sprintf(error_string,
	"eval_obj_assignment LOAD/READ:  Couldn't open image file %s",s);
					WARN(error_string);
					return;
				}
			}
			if( ! IS_READABLE(ifp) ){
				sprintf(error_string,
		"File %s is not readable!?",s);
				WARN(error_string);
				return;
			}

			if( ifp->if_dp->dt_prec == PREC_ANY || dp->dt_prec == ifp->if_dp->dt_prec ){
				/* no need to typecast */
				read_object_from_file(QSP_ARG  dp,ifp);
				/* BUG?? do we know the whole object is assigned? */
				/* does it matter? */
				dp->dt_flags |= DT_ASSIGNED;
			} else {
				dp1=make_local_dobj(QSP_ARG  
					&dp->dt_shape.si_type_dimset,
					ifp->if_dp->dt_prec);
				read_object_from_file(QSP_ARG  dp1,ifp);
				convert(QSP_ARG  dp,dp1);
				delvec(QSP_ARG  dp1);
			}
			break;

		case T_ASSIGN:		/* x=y=z; */
			dp1 = EVAL_OBJ_REF(enp->en_child[0]);
			if( dp1 == NO_OBJ ) return;
			EVAL_OBJ_ASSIGNMENT(dp1,enp->en_child[1]);
			/* now copy to the target of this call */
			if( do_unfunc(QSP_ARG  dp,dp1,FVMOV) ){
				NODE_ERROR(enp);
				WARN("Error evaluating assignment");
			}
			break;


		case T_CALL_NATIVE:			/* eval_obj_assignment() */
			eval_native_assignment(dp,enp);
			break;

		case T_INDIR_CALL:
		case T_CALLFUNC:			/* eval_obj_assignment() */
#ifdef DEBUG
if( debug & eval_debug ){
sprintf(error_string,"eval_obj_assignment calling exec_subrt, dst = %s",dp->dt_name);
advise(error_string);
}
#endif /* DEBUG */
			EXEC_SUBRT(enp,dp);
			break;

		ALL_OBJREF_CASES			/* eval_obj_assignemnt */
			if( enp->en_code == T_LIST_OBJ || enp->en_code == T_COMP_OBJ ){
				/* should be its own case... */
				/* a list of expressions, maybe literals... */
				/* We need to do something to handle 2D arrays... */
				/* ASSIGN_OBJ_FROM_LIST(dp,enp->en_child[0],0); */

				ASSIGN_OBJ_FROM_LIST(dp,enp,0);
				dp->dt_flags |= DT_ASSIGNED;
				return;
			}

			/* dp1=EVAL_OBJ_REF(enp); */
			dp1=EVAL_OBJ_EXP(enp,dp);

			if( dp1 == NO_OBJ ){
				NODE_ERROR(enp);
				WARN("Unable to evaluate RHS");
				return;
			}

			if( executing && expect_objs_assigned && ! HAS_VALUES(dp1) ){
				NODE_ERROR(enp);
				sprintf(error_string,"Object %s is used before value has been set",dp1->dt_name);
				WARN(error_string);
			}
			if( mode_is_matlab ){
				if( dp1->dt_rows == 1 && dp->dt_rows > 1 ){
					dp2 = D_SUBSCRIPT(dp,1);
					convert(QSP_ARG  dp2,dp1);
					break;
				}
			}

			/* BUG?  is this correct if we have multiple components??? */
			if( IS_SCALAR(dp1) ){
				svp = (Scalar_Value *)dp1->dt_data;
				/* BUG type conversion? */
				ASSIGN_OBJ_FROM_SCALAR(enp,dp,svp);
			} else {
				/* object-to-object copy */
				if( dp != dp1 )
					convert(QSP_ARG  dp,dp1);
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
			if( SCALAR_SHAPE(enp->en_child[0]->en_shpp) && 
				SCALAR_SHAPE(enp->en_child[1]->en_shpp) ){
				double d2;
				dval=EVAL_FLT_EXP(enp->en_child[0]);
				d2=EVAL_FLT_EXP(enp->en_child[1]);
				dbl_to_scalar(&sval,dval*d2,dp->dt_prec);
				ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			} else {
				/* we don't pass the dst object, because it may not
				 * be the right shape - we could check this, but we're lazy!
				 */
				GET_2_OPERANDS(enp,&dp1,&dp2,NO_OBJ);
				/* This assumes that the destination is the right size;
				 * it will be wrong if the dot product is a scalar...
				 */
				inner(QSP_ARG  dp,dp1,dp2);
			}
			break;

		case T_DFT:			/* eval_obj_assignment */
			/* BUG if the types are difference, dp may not be
			 * an appropriate arg for eval_obj_exp()
			 */
			dp1=EVAL_OBJ_EXP(enp->en_child[0],dp);
			/* BUG need to handle real fft's;
			 * for now, assume cpx to cpx
			 */
			if( do_unfunc(QSP_ARG  dp,dp1,FVMOV) < 0 ){
				NODE_ERROR(enp);
				WARN("error moving data for fft");
				break;
			}
			fft2d(dp,dp);
			break;

		case T_IDFT:
			dp1=EVAL_OBJ_EXP(enp->en_child[0],dp);
			/* BUG need to handle real fft's;
			 * for now, assume cpx to cpx
			 */
			if( do_unfunc(QSP_ARG  dp,dp1,FVMOV) < 0 ){
				NODE_ERROR(enp);
				WARN("error moving data for ifft");
				break;
			}
			ift2d(dp,dp);
			break;

		case T_WRAP:		/* eval_obj_assignment */
			/* We can't wrap in-place, so don't pass dp
			 * to eval_obj_exp
			 */
			/* BUG?  will this catch a=wrap(a) ?? */
			dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
#ifdef CAUTIOUS
			if( dp1 == NO_OBJ ){
				WARN("CAUTIOUS:  eval_obj_assignemnt:  missing wrap arg");
				break;
			}
#endif /* CAUTIOUS */
			wrap(QSP_ARG  dp,dp1);
			break;

		case T_SCROLL:
			dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			ldx=EVAL_INT_EXP(enp->en_child[1]);
			ldy=EVAL_INT_EXP(enp->en_child[2]);
			dp_scroll(QSP_ARG  dp,dp1,ldx,ldy);
			break;

		/* 2 argument operations */

		case T_MATH2_VFN:		/* eval_obj_assignment */
		case T_VV_FUNC:
			GET_2_OPERANDS(enp,&dp1,&dp2,dp);
			if( dp1 == NO_OBJ || dp2 == NO_OBJ ){
				NODE_ERROR(enp);
				advise("bad vector operand");
			} else
				if( do_vvfunc(QSP_ARG  dp,dp1,dp2,enp->en_vfunc_code) < 0 ){
					NODE_ERROR(enp);
					WARN("Expression error");
				}
			break;

		case T_MATH2_VSFN:
		case T_VS_FUNC:
			dp1=EVAL_OBJ_EXP(enp->en_child[0],dp);
			if( dp1 == NO_OBJ ){
				NODE_ERROR(enp);
				advise("vector operand does not exist");
				break;
			}
			EVAL_SCALAR(&sval,enp->en_child[1],MACHINE_PREC(dp1));
			if( do_vsfunc(QSP_ARG  dp,dp1,&sval,enp->en_vfunc_code) < 0 ){
				NODE_ERROR(enp);
				WARN("Error assigning object");
			}
			break;

		case T_TRANSPOSE:	/* eval_obj_assignment */
			/* Why did we ever think this was correct? */
			/* dp1 = get_id_obj(QSP_ARG  enp->en_string,enp); */
			dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			if( dp1 == NO_OBJ ) break;
			/* BUG make sure valid */
			xpose_data(QSP_ARG  dp,dp1);
			break;

		case T_VCOMP:		/* eval_obj_assignment */
			dp1=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			if( dp1 == NO_OBJ ) break;
			if( do_unfunc(QSP_ARG  dp,dp1,FVCOMP) < 0 ){
				NODE_ERROR(enp);
				WARN("error computing bit complement");
				break;
			}
			break;

		case T_RAMP:
			start=EVAL_FLT_EXP(enp->en_child[0]);
			dx=EVAL_FLT_EXP(enp->en_child[1]);
			dy=EVAL_FLT_EXP(enp->en_child[2]);
			easy_ramp2d(QSP_ARG  dp,start,dx,dy);
			break;

		case T_STR2_FN:	/* eval_obj_assignment */
		case T_STR1_FN:	/* eval_obj_assignment */
		case T_SIZE_FN: 	/* eval_obj_assignment */
			dval = EVAL_FLT_EXP(enp);
			dbl_to_scalar(&sval,dval,dp->dt_prec);
			ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			break;

		case T_LIT_INT:				/* eval_obj_assignment */
			/* BUG? we are doing a lot of unecessary conversions
			 * if the object is integer to begin with... but this
			 * will work.
			 */
			int_to_scalar(&sval,enp->en_intval,dp->dt_prec);
			ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			break;

		case T_LIT_DBL:
			dbl_to_scalar(&sval,enp->en_dblval,dp->dt_prec);
			ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			break;

		case T_BITRSHIFT:
		case T_BITLSHIFT:
		case T_BITAND:
		case T_BITOR:
		case T_BITXOR:
		case T_BITCOMP:
		case T_MODULO:
			int_to_scalar( &sval, EVAL_INT_EXP(enp), dp->dt_prec );
			ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			break;

		case T_MATH0_FN:
		case T_MATH1_FN:
		case T_MATH2_FN:		/* eval_obj_assignment */
		case T_TIMES:
		case T_PLUS:
		case T_MINUS:
		case T_DIVIDE:
		case T_SCALMAX:
		case T_SCALMIN:
			dbl_to_scalar(&sval, EVAL_FLT_EXP(enp), dp->dt_prec );
			ASSIGN_OBJ_FROM_SCALAR(enp,dp,&sval);
			break;

		case T_MATH0_VFN:			/* eval_obj_assignment */
			/* unary math function */
			if( do_un0func(QSP_ARG  dp,enp->/*en_intval*//*en_func_index*/en_vfunc_code) ){
				NODE_ERROR(enp);
				WARN("Error evaluating math function");
			}
			break;

		case T_MATH1_VFN:			/* eval_obj_assignment */
			/* unary math function */
			dp1=EVAL_OBJ_EXP(enp->en_child[0],dp);
#ifdef CAUTIOUS
			if( dp1 == NO_OBJ ){
				WARN("CAUTIOUS:  eval_obj_exp:  missing math vfn arg");
				DUMP_TREE(enp->en_child[0]);
				break;
			}
#endif /* CAUTIOUS */
			if( do_unfunc(QSP_ARG  dp,dp1,enp->/*en_intval*/en_vfunc_code) ){
				NODE_ERROR(enp);
				WARN("Error evaluating math function");
			}
			break;

		default:
			MISSING_CASE(enp,"eval_obj_assignment");
			break;
	}
/*
sprintf(error_string,"eval_obj_assignment %s DONE!",dp->dt_name);
advise(error_string);
LONGLIST(dp);
*/

}		/* end eval_obj_assignment() */

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
		if( dp->dt_type_dim[i] > 1 && enp->en_shpp->si_type_dim[i]==1 ){
			Dimension_Set ds;
			Data_Obj *sub_dp;
			index_t dst_offsets[N_DIMENSIONS]={0,0,0,0,0};
			incr_t dst_incrs[N_DIMENSIONS]={1,1,1,1,1};
			dimension_t j;
			char tmp_dst_name[LLEN];
			char *base;

			ds = dp->dt_type_dimset;
			ds.ds_dimension[i]=1;
			sprintf(tmp_dst_name,"eda.%s",dp->dt_name);
			sub_dp=make_subsamp(QSP_ARG  tmp_dst_name,dp,&ds,dst_offsets,dst_incrs);

			/* Now copy each row (or whatever).
			 * Instead of making a new subobject, we just reset the
			 * data pointer ourselves - is this risky?
			 */

			base = (char *)sub_dp->dt_data;
			for(j=0;j<dp->dt_type_dim[i];j++){
				char *cp;

				cp = base;
				cp += j * dp->dt_type_inc[i] * siztbl[ MACHINE_PREC(dp) ];
				sub_dp->dt_data = cp;

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

static double scalar_to_double(Scalar_Value *svp,prec_t prec)
{
	double dval=0.0;
	switch( prec&MACH_PREC_MASK ){
		case PREC_BY:  dval = svp->u_b; break;
		case PREC_UBY:  dval = svp->u_ub; break;
		case PREC_IN:  dval = svp->u_s; break;
		case PREC_UIN:  dval = svp->u_us; break;
		case PREC_DI:  dval = svp->u_l; break;
		case PREC_UDI: dval = svp->u_ul; break;	/* BIT precision handled elsewhere */
		case PREC_SP:  dval = svp->u_f; break;
		case PREC_DP:  dval = svp->u_d; break;
#ifdef CAUTIOUS
		default:
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  unhandled precision \"%s\" in scalar_to_double()",name_for_prec(prec&MACH_PREC_MASK));
			NWARN(DEFAULT_ERROR_STRING);
			break;
#endif /* CAUTIOUS */
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
	Dimension_Set dimset={{1,1,1,1,1}};
	Vec_Obj_Args oargs;

	eval_enp = enp;

/*
sprintf(error_string,"eval_flt_exp, code is %d",enp->en_code);
advise(error_string);
*/
	switch(enp->en_code){
		case T_MINVAL:
			dp2=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			if( dp2 == NO_OBJ ){
NODE_ERROR(enp->en_child[0]);
WARN("error evaluating arg to min");
return(0.0);
}
			/* make a scalar object to hold the answer */
			dp=make_local_dobj(QSP_ARG  &dimset,dp2->dt_prec);
			setvarg2(&oargs,dp,dp2);
			vminv(&oargs);
			dval = get_dbl_scalar_value(dp);
			delvec(QSP_ARG  dp);
			return(dval);
			break;

		case T_CALLFUNC:			/* eval_flt_exp */
			/* This could get called if we use a function inside a dimesion bracket... */
			if( ! executing ) return(0);

			srp=enp->en_call_srp;
			/* BUG SHould check and see if the return type is double... */

			/* BUG at least make sure that it's not void... */

			/* make a scalar object to hold the return value... */
			dp=make_local_dobj(QSP_ARG  &dimset,srp->sr_prec);
			EXEC_SUBRT(enp,dp);
			/* get the scalar value */
			dval = get_dbl_scalar_value(dp);
			delvec(QSP_ARG  dp);
			return(dval);
			break;


		case T_CALL_NATIVE:
			return( eval_native_flt(enp) );
			break;

		/* matlab */
		case T_INNER:
			/* assume both children are scalars */
			dval = EVAL_FLT_EXP(enp->en_child[0]);
			dval2 = EVAL_FLT_EXP(enp->en_child[1]);
			return( dval * dval2 );
			break;

		case T_SUBSCRIPT1:			/* eval_flt_exp */
			dp=GET_OBJ(enp->en_child[0]->en_string);
			index = EVAL_FLT_EXP(enp->en_child[1]);
			dp2 = D_SUBSCRIPT(dp,index);
			if( dp2 == NO_OBJ ){
				sprintf(error_string,
		"Couldn't form subobject %s[%d]",dp->dt_name,index);
				WARN(error_string);
				return(0.0);
			}
			svp = (Scalar_Value *)dp2->dt_data;
			dval = scalar_to_double(svp,dp2->dt_prec);
			return(dval);
			break;

		/* end matlab */


		case T_POWER:
			dval = EVAL_FLT_EXP(enp->en_child[0]);
			dval2 = EVAL_FLT_EXP(enp->en_child[1]);
			return( pow(dval,dval2) );
			break;

		case T_POSTINC:
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			dval = EVAL_FLT_EXP(enp->en_child[0]);
			inc_obj(dp);
			return(dval);

		case T_POSTDEC:
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			dval = EVAL_FLT_EXP(enp->en_child[0]);
			dec_obj(dp);
			return(dval);

		case T_PREDEC:
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			dec_obj(dp);
			return(EVAL_FLT_EXP(enp->en_child[0]));

		case T_PREINC:					/* eval_flt_exp */
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			inc_obj(dp);
			return(EVAL_FLT_EXP(enp->en_child[0]));

		case T_TYPECAST:	/* eval_flt_exp */
			/* We could just call eval_flt_exp on the child node,
			 * But if we are casting a float to int, we need to round...
			 */
			dval = EVAL_FLT_EXP(enp->en_child[0]);
			switch(enp->en_prec){
				case PREC_BY:   return( (double) ((char)    dval ) );
				case PREC_UBY:  return( (double) ((u_char)  dval ) );
				case PREC_IN:   return( (double) ((short)   dval ) );
				case PREC_UIN:  return( (double) ((u_short) dval ) );
				case PREC_DI:   return( (double) ((long)    dval ) );
				case PREC_UDI:  return( (double) ((u_long)  dval ) );
				case PREC_SP:   return( (double) ((float)   dval ) );
				case PREC_DP:   return(                     dval   );
				case PREC_BIT:
					if( dval )
						return 1.0;
					else
						return 0.0;
#ifdef CAUTIOUS
				default:
					NODE_ERROR(enp);
					sprintf(error_string,
		"CAUTIOUS:  eval_flt_exp:  unhandled precision (%s) in TYPECAST switch",
						name_for_prec(enp->en_prec));
					ERROR1(error_string);
#endif /* CAUTIOUS */
			}
		case T_UNDEF:
			return(0.0);

		case T_STR2_FN:
			{
			const char *s1,*s2;
			s1=EVAL_STRING(enp->en_child[0]);
			s2=EVAL_STRING(enp->en_child[1]);
			if( s1 != NULL && s2 != NULL ){
				dval = (*str2_functbl[enp->en_func_index].fn_func.str2_func)(s1,s2);
				return(dval);
			} else	return(1);	/* the default is unequal strings */
			}

		case T_STR1_FN:	/* eval_flt_exp */
			/* BUG?  should this really be an int expression? */
			{
			const char *str;
			str = EVAL_STRING(enp->en_child[0]);
			if( str == NULL ){
				WARN("error evaluating string...");
				return(0.0);
			}
			dval = (*str1_functbl[enp->en_func_index].str1f_func)(QSP_ARG  str);
			return(dval);
			}

		case T_SIZE_FN:	/* eval_flt_exp */
			if( enp->en_child[0]->en_code == T_STRING ){
				/* name of a sizable object */
				Item *ip;
				ip = find_sizable(QSP_ARG  enp->en_child[0]->en_string);
				if(ip==NO_ITEM){
					sprintf(error_string,
						"Couldn't find sizable object %s",
						enp->en_child[0]->en_string);
					WARN(error_string);
					return(0.0);
				}
		dval = (*size_functbl[enp->en_func_index].szf_func)(QSP_ARG  ip);
			} else {
				/* an objref expressions */
				int save_e;	/* objs don't need values to query their size */

				save_e = expect_objs_assigned;
				expect_objs_assigned=0;
				dp = EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
				expect_objs_assigned=save_e;

				if( dp == NO_OBJ ){
					/* This might not be an error if we have used an object
					 * expression as a dimension, e.g.
					 * float x[ncols(v)];
					 * where v is a subroutine argument...
					 */

					if( executing ){
						NODE_ERROR(enp);
						sprintf(error_string,
				"bad object expression given for function %s",
							size_functbl[enp->en_func_index].fn_name);
						WARN(error_string);
DUMP_TREE(enp);
					}
					return(0.0);	/* eval_flt_exp T_SIZE_FN */
				}
				if( UNKNOWN_SHAPE(&dp->dt_shape) ){
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
					sprintf(error_string,"returning 0 for size of unknown shape object %s",
						dp->dt_name);
					advise(error_string);
					return(0.0);
				}
				/* Originally, we called the function from the size_functbl here...
				 * this works for any type of sizable object, but in order to do
				 * an itemtype-specific function call, it has to determine what
				 * type of object the name refers to by searching each of the sizable
				 * object databases...  This creates a problem, because there can
				 * be pointed-to objects that have had their contexts popped
				 * because they are not in the current scope.  Because we know this
				 * is a data object, we just call the appropriate dobj-sepcific function.
				 */
				/* intval is the index of the function in the size function table,
				 * not the dimension index.  We might fix this by fixing the order
				 * of the functions in the table, or by calling a mapping
				 * function...  either way it's not clean!?
				 */
				dval = get_dobj_size((Item *)dp,enp->en_func_index);
			}
			return(dval);

		case T_SCALMAX:
			dval = EVAL_FLT_EXP(enp->en_child[0]);
			dval2 = EVAL_FLT_EXP(enp->en_child[1]);
			if( dval > dval2 )
				return(dval);
			else
				return(dval2);

		case T_SCALMIN:
			dval = EVAL_FLT_EXP(enp->en_child[0]);
			dval2 = EVAL_FLT_EXP(enp->en_child[0]);
			if( dval < dval2 )
				return(dval);
			else
				return(dval2);

		case T_STATIC_OBJ:	/* eval_flt_exp */
			dp=enp->en_dp;
			goto obj_flt_exp;

		case T_POINTER:
		case T_DYN_OBJ:		/* eval_flt_exp */
			dp=GET_OBJ(enp->en_string);
#ifdef CAUTIOUS
			if( dp == NO_OBJ ){
				sprintf(error_string,"CAUTIOUS:  eval_flt_exp:  missing object %s",enp->en_string);
				WARN(error_string);
				return(0.0);
			}
#endif /* CAUTIOUS */

obj_flt_exp:

			/* check that this object is a scalar */
			if( dp->dt_n_type_elts != 1 ){
				/* what about a complex scalar? BUG */
				NODE_ERROR(enp);
				sprintf(error_string,
		"eval_flt_exp:  object %s is not a scalar!?",dp->dt_name);
				WARN(error_string);
			}
			svp=(Scalar_Value *)dp->dt_data;
			if( svp == NO_SCALAR_VALUE ){
				NODE_ERROR(enp);
				sprintf(error_string,"object %s has null data ptr!?",dp->dt_name);
				advise(error_string);
				return(0.0);
			}

			if( IS_BITMAP(dp) ){
				if( svp->u_ul & 1<<dp->dt_bit0 )
					dval=1.0;
				else
					dval=0.0;
			} else {
				dval = scalar_to_double(svp,dp->dt_prec);
			}
			return( dval );

		/* BUG need T_CURLY_SUBSCR too! */
		case T_SQUARE_SUBSCR:			/* eval_flt_exp */
			/* dp=GET_OBJ(enp->en_child[0]->en_string); */
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			index = EVAL_INT_EXP(enp->en_child[1]);
			dp2 = D_SUBSCRIPT(dp,index);
			if( dp2 == NO_OBJ ){
				sprintf(error_string,
		"Couldn't form subobject %s[%d]",dp->dt_name,index);
				WARN(error_string);
				return(0.0);
			}
			svp = (Scalar_Value *)dp2->dt_data;
			dval = scalar_to_double(svp,dp2->dt_prec);
			return(dval);

		case T_CURLY_SUBSCR:
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			index = EVAL_INT_EXP(enp->en_child[1]);
			dp2 = C_SUBSCRIPT(dp,index);
			if( dp2 == NO_OBJ ){
				sprintf(error_string,
		"Couldn't form subobject %s[%d]",dp->dt_name,index);
				WARN(error_string);
				return(0.0);
			}
			svp = (Scalar_Value *)dp2->dt_data;
			dval = scalar_to_double(svp,dp2->dt_prec);
			return(dval);

		case T_MATH0_FN:
		dval = (*math0_functbl[enp->en_func_index].d0f_func)();
			return(dval);

		case T_MATH1_FN:
			dval = EVAL_FLT_EXP(enp->en_child[0]);
		dval = (*math1_functbl[enp->en_func_index].d1f_func)(dval);
			return(dval);
		case T_MATH2_FN:				/* eval_flt_exp */
			dval = EVAL_FLT_EXP(enp->en_child[0]);
			dval2 = EVAL_FLT_EXP(enp->en_child[1]);
	dval = (*math2_functbl[enp->en_func_index].d2f_func)(dval,dval2);
			return(dval);
		case T_UMINUS:
			dval = EVAL_FLT_EXP(enp->en_child[0]);
			return(-dval);
		case T_RECIP:
			dval = EVAL_FLT_EXP(enp->en_child[0]);
			return(1.0/dval);
		case T_LIT_DBL:
			return(enp->en_dblval);
		case T_LIT_INT:
			dval=enp->en_intval;
			return(dval);
		ALL_SCALINT_BINOP_CASES
			dval=EVAL_INT_EXP(enp);
			return(dval);
		case T_DIVIDE:
			dval=EVAL_FLT_EXP(enp->en_child[0]);
			dval2=EVAL_FLT_EXP(enp->en_child[1]);
			if( dval2==0.0 ){
				NODE_ERROR(enp);
				sprintf(error_string,"Divide by 0!?");
				WARN(error_string);
				return(0.0);
			}
			return(dval/dval2);
		case T_PLUS:
			dval=EVAL_FLT_EXP(enp->en_child[0]);
			dval2=EVAL_FLT_EXP(enp->en_child[1]);
			return(dval+dval2);
		case T_MINUS:
			dval=EVAL_FLT_EXP(enp->en_child[0]);
			dval2=EVAL_FLT_EXP(enp->en_child[1]);
			return(dval-dval2);
		case T_TIMES:
			dval=EVAL_FLT_EXP(enp->en_child[0]);
			dval2=EVAL_FLT_EXP(enp->en_child[1]);
			return(dval*dval2);

		default:
			MISSING_CASE(enp,"eval_flt_exp");
			break;
	}
	return(0.0);
}

static void easy_ramp2d(QSP_ARG_DECL  Data_Obj *dst_dp,double start,double dx,double dy)
{
	Vec_Obj_Args oargs;
	Scalar_Value sv1, sv2, sv3;

	cast_to_scalar_value(QSP_ARG  &sv1,dst_dp->dt_prec,(double)start);
	cast_to_scalar_value(QSP_ARG  &sv2,dst_dp->dt_prec,(double)dx);
	cast_to_scalar_value(QSP_ARG  &sv3,dst_dp->dt_prec,(double)dy);

	oargs.oa_dp[0] = oargs.oa_dest = dst_dp;
	oargs.oa_svp[0] = &sv1;
	oargs.oa_svp[1] = &sv2;
	oargs.oa_svp[2] = &sv3;

	vramp2d(&oargs);
}

#define INSURE_DESTINATION						\
									\
			if( dst_dp == NO_OBJ ){				\
				dst_dp=make_local_dobj(QSP_ARG 		\
					&enp->en_shpp->si_type_dimset,	\
					enp->en_shpp->si_prec);		\
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

	switch(enp->en_code){
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
			start=EVAL_FLT_EXP(enp->en_child[0]);
			dx=EVAL_FLT_EXP(enp->en_child[1]);
			dy=EVAL_FLT_EXP(enp->en_child[2]);
			INSURE_DESTINATION
			easy_ramp2d(QSP_ARG  dst_dp,start,dx,dy);
			return(dst_dp);
			break;
			}

		case T_TRANSPOSE:
			dp=EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			if( dp == NO_OBJ ) break;
			/* BUG make sure valid??? */
			INSURE_DESTINATION
			xpose_data(QSP_ARG  dst_dp,dp);
			return(dst_dp);
			break;

		case T_SCROLL:		/* eval_obj_exp */
			dp = EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
			ldx=EVAL_INT_EXP(enp->en_child[1]);
			ldy=EVAL_INT_EXP(enp->en_child[2]);
#ifdef CAUTIOUS
			if( dp == NO_OBJ ){
				WARN("CAUTIOUS:  eval_obj_exp:  missing scroll arg");
				break;
			}
#endif /* CAUTIOUS */
			/* BUG? do we need to make sure that dp is not dst_dp? */
			INSURE_DESTINATION
			dp_scroll(QSP_ARG  dst_dp,dp,ldx,ldy);
			return(dst_dp);


		case T_WRAP:				/* eval_obj_exp */
			dp = EVAL_OBJ_EXP(enp->en_child[0],NO_OBJ);
#ifdef CAUTIOUS
			if( dp == NO_OBJ ){
				WARN("CAUTIOUS:  eval_obj_exp:  missing wrap arg");
				break;
			}
#endif /* CAUTIOUS */
			/* BUG? do we need to make sure that dp is not dst_dp? */
			if( dst_dp == NO_OBJ ){
				dst_dp=make_local_dobj(QSP_ARG  
					&enp->en_shpp->si_type_dimset,
					enp->en_shpp->si_prec);
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
			return(enp->en_dp);


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
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			if( dp == NO_OBJ ) return(dp);

			/* Before we evaluate the subscript as an integer, check and
			 * see if it's a vector...
			 */
			if( SCALAR_SHAPE(enp->en_child[1]->en_shpp) ){
				dimension_t index;
				index = EVAL_INT_EXP(enp->en_child[1]);
				if( enp->en_code == T_SQUARE_SUBSCR )
					return( D_SUBSCRIPT(dp,index) );
				else
					return( C_SUBSCRIPT(dp,index) );
			} else {
				Data_Obj *index_dp;
				index_dp=EVAL_OBJ_REF(enp->en_child[1]);
				if( index_dp == NO_OBJ ) break;	/* BUG?  print error here? */
				if( index_dp->dt_comps != (dimension_t)(1+dp->dt_maxdim-dp->dt_mindim) ){
					NODE_ERROR(enp);
					sprintf(error_string,
	"Map source object %s needs %d indices, but index array %s has component dimension %d!?",
						dp->dt_name,1+dp->dt_maxdim-dp->dt_mindim,
						index_dp->dt_name,index_dp->dt_comps);
					WARN(error_string);
				} else {
					return( MAP_SUBSCRIPTS(dp,index_dp,enp) );
				}
			}
			break;

#ifdef MATLAB_FOOBAR
		/* matlab */
		case T_SUBSCRIPT1:	/* eval_obj_exp */
			dp=GET_OBJ(enp->en_child[0]->en_string);
			index = EVAL_FLT_EXP(enp->en_child[1]);
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
#ifdef CAUTIOUS
				if( enp->en_shpp==NO_SHAPE ){
					WARN(error_string);
					sprintf(error_string,
		"CAUTIOUS:  no shape at node %d, need for proto!?",enp->en_serial);
					DUMP_TREE(curr_srp->sr_body);
					return(NO_OBJ);
				}
				if( UNKNOWN_SHAPE(enp->en_shpp) ){
					NODE_ERROR(enp);
					WARN("CAUTIOUS:  eval_obj_exp:  proto node shape is UNKNOWN!?");
					DUMP_TREE(enp);
					return(NO_OBJ);
				}
#endif /* CAUTIOUS */
				dst_dp=make_local_dobj(QSP_ARG   &enp->en_shpp->si_type_dimset,
							enp->en_shpp->si_prec);
#ifdef CAUTIOUS
				if( dst_dp == NO_OBJ ){
			WARN("CAUTIOUS:  couldn't make shaped copy!?");
					return(dst_dp);
				}
#endif /* CAUTIOUS */
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

	switch(enp->en_code){
		case T_NAME_FUNC:
			if( dumping ) return(STRING_FORMAT);

			dp=EVAL_OBJ_REF(enp->en_child[0]);
#ifdef CAUTIOUS
			if( dp == NO_OBJ ){
				NODE_ERROR(enp);
				WARN("CAUTIOUS:  bad namefunc node");
			}
#endif /* CAUTIOUS */
			return(dp->dt_name);

		case T_STRING:
			return( enp->en_string );
		case T_STR_PTR:
			if( dumping ) return(STRING_FORMAT);

			idp = EVAL_PTR_REF(enp,EXPECT_PTR_SET);
			if( idp==NO_IDENTIFIER ) return("");
#ifdef CAUTIOUS
			if( ! IS_STRING_ID(idp) ){
				NODE_ERROR(enp);
				sprintf(error_string,"CAUTIOUS:  identifier %s is not a string ptr!?",idp->id_name);
				WARN(error_string);
				break;
			}
#endif /* CAUTIOUS */
			if( idp->id_refp->ref_sbp->sb_buf == NULL ){
				NODE_ERROR(enp);
				sprintf(error_string,"string pointer %s not set",idp->id_name);
				advise(error_string);
				break;
			}
			return(idp->id_refp->ref_sbp->sb_buf);

		case T_SET_STR:
			EVAL_WORK_TREE(enp,NO_OBJ);	/* do the assignment! */
			return( EVAL_MIXED_LIST(enp->en_child[0]) );

		case T_STRING_LIST:
		case T_PRINT_LIST:
			s1=EVAL_MIXED_LIST(enp->en_child[0]);
			s2=EVAL_MIXED_LIST(enp->en_child[1]);
			n=strlen(s1)+strlen(s2)+1;
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
				format_scalar_obj(buf,dp,dp->dt_data);
			else {
				/*
				NODE_ERROR(enp);
				sprintf(error_string,
					"eval_mixed_list:  object %s is not a scalar!?",dp->dt_name);
				WARN(error_string);
				return("");
				*/
				strcpy(buf,dp->dt_name);
			}
			n=strlen(buf)+1;
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

static void dec_obj(Data_Obj *dp)
{
	Scalar_Value *svp;

	svp = (Scalar_Value *)dp->dt_data;
	switch( MACHINE_PREC(dp) ){
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
		case N_MACHINE_PRECS:
			break;
	}
}

static void inc_obj(Data_Obj *dp)
{
	Scalar_Value *svp;

	svp = (Scalar_Value *)dp->dt_data;
	switch( MACHINE_PREC(dp) ){
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
		case N_MACHINE_PRECS:
			break;
	}
}

static void eval_print_stat(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp;
	Identifier *idp;
	long n;
	double d;
	const char *s;


	eval_enp = enp;

	switch(enp->en_code){
		case T_CALLFUNC:			/* eval_print_stat */
			if( ! SCALAR_SHAPE(enp->en_shpp) ){
				prt_msg("");
				NODE_ERROR(enp);
				advise("Can't print a non-scalar function");
				break;
			}
			/* Now we evaluate it... */
			switch(enp->en_prec){
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
			dp = EVAL_OBJ_REF(enp->en_child[0]);
			if( dp == NO_OBJ ){
				NODE_ERROR(enp);
				advise("Missing object");
			} else
				prt_msg_frag(dp->dt_name);
			break;

		case T_STRING_LIST:
		case T_MIXED_LIST:
		case T_PRINT_LIST:
			EVAL_PRINT_STAT(enp->en_child[0]);
			prt_msg_frag(" ");
			EVAL_PRINT_STAT(enp->en_child[1]);
			break;
		case T_POINTER:
			idp = EVAL_PTR_REF(enp,EXPECT_PTR_SET);
#ifdef CAUTIOUS
			if( ! IS_POINTER(idp) ){
				NODE_ERROR(enp);
				sprintf(error_string,"CAUTIOUS:  identifier %s is not a pointer",idp->id_name);
				advise(error_string);
				break;
			}
			if( ! POINTER_IS_SET(idp) ){
				NODE_ERROR(enp);
				sprintf(error_string,"CAUTIOUS:  pointer %s is not set",idp->id_name);
				advise(error_string);
				break;
			}
			/* If it's a pointer, should be id_ptrp, not id_refp!? */
			/*
			if( idp->id_refp == NO_REFERENCE ){
				NODE_ERROR(enp);
				sprintf(error_string,"CAUTIOUS:  id %s, id_refp is null",idp->id_name);
				advise(error_string);
				break;
			}
			*/
			if( idp->id_ptrp == NO_POINTER ){
				NODE_ERROR(enp);
				sprintf(error_string,"CAUTIOUS:  id %s, id_ptrp is null",idp->id_name);
				advise(error_string);
				break;
			}
			if( idp->id_ptrp->ptr_refp == NO_REFERENCE ){
				NODE_ERROR(enp);
				sprintf(error_string,"CAUTIOUS:  id %s, id_ptrp->ptr_refp is null",idp->id_name);
				advise(error_string);
				break;
			}
#endif /* CAUTIOUS */
			if( IS_OBJECT_REF(idp->id_ptrp->ptr_refp) ){
#ifdef CAUTIOUS
				if( idp->id_ptrp->ptr_refp->ref_dp == NO_OBJ ){
					NODE_ERROR(enp);
					sprintf(error_string,"CAUTIOUS:  id %s, ref_dp is null",idp->id_name);
					advise(error_string);
					break;
				}
#endif /* CAUTIOUS */

				/* what should we print here? */
				/* If the pointer points to a string, then print the string... */
				dp=idp->id_ptrp->ptr_refp->ref_dp ;
				if( dp->dt_prec == PREC_CHAR || dp->dt_prec == PREC_STR )
					prt_msg_frag((char *)dp->dt_data);
				else
					prt_msg_frag(idp->id_name);
			} else if( IS_STRING_REF(idp->id_ptrp->ptr_refp) ){
				prt_msg_frag(idp->id_ptrp->ptr_refp->ref_sbp->sb_buf);
			}
#ifdef CAUTIOUS
			  else ERROR1("CAUTIOUS:  bad reference type");
#endif /* CAUTIOUS */
			break;

		ALL_OBJREF_CASES
		case T_PREDEC:
		case T_PREINC:			/* eval_print_stat */
		case T_POSTDEC:
		case T_POSTINC:			/* eval_print_stat */
			dp = EVAL_OBJ_REF(enp);
			if( dp==NO_OBJ ) return;

			if( enp->en_code == T_PREINC ) inc_obj(dp);
			else if( enp->en_code == T_PREDEC ) dec_obj(dp);

			if( IS_SCALAR(dp) ){
				format_scalar_obj(msg_str,dp,dp->dt_data);
				prt_msg_frag(msg_str);
			} else {
				/*
				NODE_ERROR(enp);
				sprintf(error_string,
					"eval_print_stat:  object %s is not a scalar!?",dp->dt_name);
				advise(error_string);
				*/
				prt_msg_frag(dp->dt_name);
			}

			if( enp->en_code == T_POSTINC ) inc_obj(dp);
			else if( enp->en_code == T_POSTDEC ) dec_obj(dp);

			break;


		case T_SET_STR:
		case T_STR_PTR:
			s=EVAL_MIXED_LIST(enp);
			prt_msg_frag(s);
			break;

		case T_STRING:
			prt_msg_frag(enp->en_string);
			break;
		case T_LIT_INT:
		case T_SIZE_FN:
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

	switch(enp->en_code){
		case T_CALL_NATIVE:
			/*
			sprintf(error_string,"eval_string T_CALL_NATIVE:  function %s, not handled...",
					native_func_tbl[enp->en_func_index].kw_token);
			WARN(error_string);
			return("FOO");
			*/
			return( eval_native_string(enp) );
			break;

		case T_SQUARE_SUBSCR:			/* eval_string */
		case T_CURLY_SUBSCR:
		case T_STATIC_OBJ:			/* eval_string */
		case T_DYN_OBJ:				/* eval_string */
			dp = EVAL_OBJ_EXP(enp,NO_OBJ);
			if( dp == NO_OBJ ){
				NODE_ERROR(enp);
				sprintf(error_string,"eval_string:  missing object %s",enp->en_string);
				WARN(error_string);
				return(NULL);
			}
#ifdef CAUTIOUS
			if( dp->dt_prec != PREC_CHAR ){
				sprintf(error_string,
			"CAUTIOUS:  eval_string:  object %s (%s) should have %s precision",
					dp->dt_name,name_for_prec(dp->dt_prec),
						name_for_prec(PREC_CHAR));
				WARN(error_string);
				return(NULL);
			}
#endif /* CAUTIOUS */
			/* not exactly a BUG, but we might verify that the number of
			 * columns matches the string length?
			 */
			return((char *)dp->dt_data);
			break;

		case T_SET_STR:
			if( dumping ) return(STRING_FORMAT);

			s = EVAL_STRING(enp->en_child[1]);
			idp = EVAL_PTR_REF(enp->en_child[0],UNSET_PTR_OK);
			if( idp == NO_IDENTIFIER ) break;
			assign_string(QSP_ARG  idp,s,enp);
			return(s);

		case T_PRINT_LIST:
			return(EVAL_MIXED_LIST(enp));

		case T_STRING_LIST:
			{
			char *new_string;
			s1=EVAL_STRING(enp->en_child[0]);
			s2=EVAL_STRING(enp->en_child[1]);
			if( s1 == NULL || s2 == NULL ) return(NULL);
			/* BUG need some garbage collection!? */
			n=strlen(s1)+strlen(s2)+1;
			new_string=(char *)getbuf(n);
			strcpy(new_string,s1);
			strcat(new_string,s2);
			return(new_string);
			}
			break;

		case T_STRING:
			s=enp->en_string;
			break;

		case T_STR_PTR:			/* eval_string */
			if( dumping ) return(STRING_FORMAT);

			idp = EVAL_PTR_REF(enp,EXPECT_PTR_SET);

			if( idp == NO_IDENTIFIER ){
				NODE_ERROR(enp);
				sprintf(error_string,"missing string pointer object %s",enp->en_string);
				advise(error_string);
				return(NULL);
			}

#ifdef CAUTIOUS
			if( ! IS_STRING_ID(idp) ){
				WARN("CAUTIOUS:  eval-string:  ptr not a string!?");
				return(NULL);
			}
#endif /* CAUTIOUS */
			if( idp->id_refp->ref_sbp->sb_buf == NULL ){
				NODE_ERROR(enp);
				sprintf(error_string,"string pointer \"%s\" used before set!?",idp->id_name);
				WARN(error_string);
				return(NULL);
			} else
				s=idp->id_refp->ref_sbp->sb_buf;
			break;

		default:
			MISSING_CASE(enp,"eval_string");
			return(bad_string);
			break;
	}
	return(s);
}

#ifdef FOOBAR
void intr_evaluation(int arg)
{
	/* use setjmp/longjmp to get back to the interpreter */
	if( eval_enp != NO_VEXPR_NODE )
		NODE_ERROR(eval_enp);
#ifdef CAUTIOUS
	else NWARN("CAUTIOUS:  no current eval_enp!?");
#endif /* CAUTIOUS */

	advise("execution halted by SIGINTR");
	interrupted=1;
	sleep(2);
	/* signal(SIGINT,intr_evaluation); */
}
#endif /* FOOBAR */

/* for matlab support */

void insure_object_size(QSP_ARG_DECL  Data_Obj *dp,index_t index)
{
	int which_dim;

	which_dim = dp->dt_mindim;

	if( dp->dt_type_dim[which_dim] <= index ){ /* index is too big, we need to resize */
		Dimension_Set dims;
		index_t offsets[N_DIMENSIONS];
		Scalar_Value sval;
		Data_Obj *new_dp,*sub_dp;
		void *tmp_data;
		int i;

		/* first, get the new data area */

		dims = dp->dt_type_dimset;
		for(i=0;i<N_DIMENSIONS;i++){
		/*
			dims[i]=dp->dt_type_dim[i];
		*/
			offsets[i]=0;
		}
		dims.ds_dimension[which_dim]=index;

		new_dp = make_dobj(QSP_ARG  "tmpname",&dims,dp->dt_prec);

		/* set new data area to all zeroes */
		sval.u_d = 0.0;	/* BUG assumes PREC_DP */
		dp_const(QSP_ARG  new_dp,&sval);

		/* copy in original data */
		sub_dp = mk_subseq(QSP_ARG  "tmp_subseq",new_dp,offsets,&dp->dt_type_dimset);
		dp_copy(QSP_ARG  sub_dp,dp);

		/* get rid of the subimage */
		delvec(QSP_ARG  sub_dp);

		/* now this is tricky...  we want to swap data areas, and dimensions
		 * between new_dp and dp...  here goes nothing
		 */
		tmp_data = dp->dt_data;
		dp->dt_data = new_dp->dt_data;
		new_dp->dt_data = tmp_data;

		new_dp->dt_type_dim[which_dim] = dp->dt_type_dim[which_dim];
		dp->dt_type_dim[which_dim] = dims.ds_dimension[which_dim];

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
	dp_new = make_dobj(QSP_ARG  "ass_tmp",&shpp->si_type_dimset,shpp->si_prec);
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
	if( idp == NO_IDENTIFIER ){
		sprintf(error_string,"CAUTIOUS:  missing matlab identifier %s",name);
		WARN(error_string);
	} else if( idp->id_type != ID_REFERENCE ){
		sprintf(error_string,"CAUTIOUS:  identifier %s is not a reference",name);
		WARN(error_string);
	} else {
		idp->id_refp->ref_dp = dp_new;
		/* and update the shape! */
		idp->id_shape = dp_new->dt_shape;
	}
	return(dp_new);
}
				
static Data_Obj * mlab_lhs(QSP_ARG_DECL Data_Obj *dp, Vec_Expr_Node *enp)
{
#ifdef CAUTIOUS
	if( enp->en_code != T_ASSIGN ){
		NODE_ERROR(enp);
		sprintf(error_string,"CAUTIOUS mlab_lhs:  %s is not an assign node",
			node_desc(enp));
		ERROR1(error_string);
	}
#endif /* CAUTIOUS */

	if( dp != NO_OBJ ){
		/* If the shape doesn't match,
		 * then we have to remake the object
		 */
if( enp->en_shpp == NO_SHAPE ){
WARN("mlab_eval_work_tree:  T_ASSIGN has null shape ptr");
} else {
if( enp->en_child[1]->en_shpp == NO_SHAPE ){
WARN("mlab_lhs:  rhs has null shape");
}

		if( dp->dt_cols != enp->en_child[1]->en_shpp->si_cols ||
			dp->dt_rows != enp->en_child[1]->en_shpp->si_rows ){

			Data_Obj *dp_new;
			const char *s;

			/* In matlab, you're allowed to reassign
			 * the shape of an object...
			 */

			s = GET_LHS_NAME(enp->en_child[0]);
			dp_new = mlab_reshape(QSP_ARG  dp,enp->en_child[1]->en_shpp,s);
			/* We do this later! */
			/* EVAL_OBJ_ASSIGNMENT(dp_new,enp->en_child[1]); */
			return(dp_new);
		}
} /* end debug */
	} else {
		/* make a new object */
#ifdef CAUTIOUS
		if( enp->en_child[0] == NO_VEXPR_NODE ){
			sprintf(error_string,"CAUTIOUS:  mlab_lhs:  %s has null child",node_desc(enp));
			WARN(error_string);
			DUMP_TREE(enp);
			ERROR1("CAUTIOUS:  giving up");
		}
#endif /* CAUTIOUS */
		dp = CREATE_MATRIX(enp->en_child[0],enp->en_shpp);
	}
	return(dp);
} /* end mlab_lhs */


/* For matlab, if the rhs shape is different, then we reshape the LHS to match.
 * (for other languages, this might be an error!)
 * The node passed is generally the assign node...
 */

static Data_Obj *mlab_target(QSP_ARG_DECL Data_Obj *dp, Vec_Expr_Node *enp)
{
	if( dp == NO_OBJ ){
		dp = CREATE_MATRIX(enp,enp->en_shpp);
	}
	else {
	/* BUG should check reshape if already exists */
		sprintf(error_string,"mlab_target %s:  not checking reshape",dp->dt_name);
	  	WARN(error_string);
	}
	return(dp);
}


static Data_Obj *create_list_lhs(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	Data_Obj *dp1,*dp2;
	List *lp;
	Node *np1,*np2;

	dp1 = EVAL_OBJ_REF(enp->en_child[0]);
	dp1 = MLAB_TARGET(dp1,enp->en_child[0]);
	dp2 = EVAL_OBJ_REF(enp->en_child[1]);
	dp2 = MLAB_TARGET(dp2,enp->en_child[1]);
	np1=mk_node(dp1);
	np2=mk_node(dp2);
	lp=new_list();
	addTail(lp,np1);
	addTail(lp,np2);
	dp1=make_obj_list(QSP_ARG  localname(),lp);
	return(dp1);
}

/* make something new */

static Data_Obj *create_matrix(QSP_ARG_DECL Vec_Expr_Node *enp,Shape_Info *shpp)
{
	Data_Obj *dp;
	Identifier *idp;

	switch(enp->en_code){
		case T_RET_LIST:
			return( CREATE_LIST_LHS(enp) );

		case T_DYN_OBJ:		/* create_matrix */
			/* we need to create an identifier too! */
			idp = make_named_reference(QSP_ARG  enp->en_string);
			dp = make_dobj(QSP_ARG  enp->en_string,&shpp->si_type_dimset,shpp->si_prec);

#ifdef CAUTIOUS
			if( dp == NO_OBJ ){
				NODE_ERROR(enp);
				ERROR1("CAUTIOUS:  create_matrix:  make_dobj failed");
			}
#endif /* CAUTIOUS */

			idp->id_refp->ref_dp = dp;
			idp->id_shape = dp->dt_shape;
			return(dp);
		default:
			MISSING_CASE(enp,"create_matrix");
			break;
	}
	return(NO_OBJ);
}

static void assign_element(QSP_ARG_DECL Data_Obj *dp,dimension_t ri,dimension_t ci,Vec_Expr_Node *enp)
{
	double *dbl_p,d;

	dp->dt_flags |= DT_ASSIGNED;

#ifdef CAUTIOUS
	if( dp->dt_prec != PREC_DP ){
		sprintf(error_string,"CAUTIOUS:  assign_element:  object %s is not double precision!?",dp->dt_name);
		WARN(error_string);
		return;
	}
#endif /* CAUTIOUS */
	dbl_p = (double *)dp->dt_data;
	/* assign_element uses matlab indexing */
	dbl_p += (ri-1) * dp->dt_rinc;
	dbl_p += (ci-1) * dp->dt_pinc;
	d = EVAL_FLT_EXP(enp);
	*dbl_p = d;
}

/* assign_row - matlab support */

static void assign_row(QSP_ARG_DECL Data_Obj *dp,dimension_t row_index,Vec_Expr_Node *enp)
{
	dimension_t j;
	Data_Obj *src_dp;

	switch(enp->en_code){
		case T_ROW:		/* really a list of elements */
			j=enp->en_shpp->si_cols;
			ASSIGN_ELEMENT(dp,row_index,j,enp->en_child[1]);
			ASSIGN_ROW(dp,row_index,enp->en_child[0]);
			break;
		case T_TIMES:
		case T_UMINUS:
		case T_LIT_DBL:
			ASSIGN_ELEMENT(dp,row_index,1,enp);
			break;
		case T_STATIC_OBJ:	/* assign_row */
			src_dp = enp->en_dp;
			goto assign_row_from_dp;

		case T_DYN_OBJ:		/* assign_row */
			src_dp = DOBJ_OF(enp->en_string);
			/* fall thru */
assign_row_from_dp:
			for(j=0;j<src_dp->dt_cols;j++){
				double *dbl_p1,*dbl_p2;
				/* BUG we need to get the value in a general way */
				dbl_p1 = (double *)dp->dt_data;
				dbl_p1 += (row_index-1) * dp->dt_rinc;
				dbl_p1 += j * dp->dt_pinc;
				dbl_p2 = (double *)src_dp->dt_data;
				dbl_p2 += j * src_dp->dt_pinc;
				*dbl_p1 = *dbl_p2;
			}
			dp->dt_flags |= DT_ASSIGNED;
			break;
		default:
			MISSING_CASE(enp,"assign_row");
			break;
	}
}

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
			sprintf(error_string,"Unexpected mlab cmd code %d",code);
			WARN(error_string);
			break;
	}
}
#endif /* MATLAB_FOOBAR */

void eval_immediate(QSP_ARG_DECL Vec_Expr_Node *enp)
{
//sprintf(error_string,"eval_immediate BEGIN");
//advise(error_string);
//DUMP_TREE(enp);
	/* Why do we need to do this here??? */
	enp=COMPILE_PROG(&enp);
/*
sprintf(error_string,"eval_immediate after compile_prog...");
advise(error_string);
DUMP_TREE(enp);
*/
	if( IS_CURDLED(enp) ) return;

	if( dumpit ) {
		print_shape_key();
		DUMP_TREE(enp);
	}

	/* We need to do some run-time resolution for this case:
	 * float f[]=[1,2,3];
	 *
	 */

	switch(enp->en_code){
		case T_WHILE:			/* eval_immediate */
		case T_DO_WHILE:			/* eval_immediate */
		case T_DO_UNTIL:			/* eval_immediate */
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
} /* end eval_immediate */


Data_Obj *
make_local_dobj(QSP_ARG_DECL  Dimension_Set *dsp,prec_t p)
{
	Data_Obj *dp;
	Node *np;
	const char *s;

	dp=make_dobj(QSP_ARG  localname(),dsp,p);
	if( dp == NO_OBJ ) return(dp);

	/* remember this for later deletion... */
	if( local_obj_lp == NO_LIST )
		local_obj_lp = new_list();

	/* We can't just store dp, because it could become a dangling pointer if someone
	 * else deletes him, and apparently some functions are good citizens and clean up
	 * after themselves but others do not or cannot.  So we have to save a new string with
	 * the name, and hope we delete the same one later...
	 */

	s=savestr(dp->dt_name);
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

	np=local_obj_lp->l_head;
	while(np!=NO_NODE){
		s = (char *)np->n_data;
		dp = DOBJ_OF(s);

#ifdef CAUTIOUS
		if( strncmp(s,"L.",2) ){
			sprintf(error_string,
				"CAUTIOUS:  delete_local_objs:  Oops, object %s is on local object list!?",
				dp->dt_name);
			ERROR1(error_string);
		}
#endif /* CAUTIOUS */
		if( dp != NO_OBJ ){
			delvec(QSP_ARG  dp);
		}
		np = np->n_next;
	}
}

