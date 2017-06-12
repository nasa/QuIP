#include "quip_config.h"

#include "quip_prot.h"
#include "dobj_prot.h"

Item_Type *prec_itp=NULL;

#define INIT_VOID_PREC(name,type,code)			\
	INIT_GENERIC_PREC(name,type,code)		\
	SET_PREC_SIZE(prec_p, 0);

#define INIT_PREC(name,type,code)			\
	INIT_GENERIC_PREC(name,type,code)		\
	SET_PREC_SIZE(prec_p, sizeof(type));

#define INIT_GENERIC_PREC(name,type,code)		\
	prec_p = new_prec(QSP_ARG  #name);		\
	SET_PREC_CODE(prec_p, code);			\
	SET_PREC_ASSIGN_SCALAR_FUNC(prec_p,name##_assign_scalar);	\
	SET_PREC_CAST_TO_DOUBLE_FUNC(prec_p,cast_##name##_to_double);	\
	SET_PREC_CAST_FROM_DOUBLE_FUNC(prec_p,cast_##name##_from_double);	\
	SET_PREC_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(prec_p,cast_indexed_##name##_from_double);	\
	if( (code & PSEUDO_PREC_MASK) == 0 )		\
		SET_PREC_MACH_PREC_PTR(prec_p, prec_p);	\
	else {						\
		SET_PREC_MACH_PREC_PTR(prec_p,		\
		prec_for_code( code & MACH_PREC_MASK ) ); \
	}

static Precision *new_prec(QSP_ARG_DECL  const char *name)
{
	Precision *prec_p;

	prec_p = (Precision *) new_item(QSP_ARG  prec_itp, name, sizeof(Precision) );
	assert( prec_p != NULL );

	return(prec_p);
}

Precision *get_prec(QSP_ARG_DECL  const char *name)
{
	return (Precision *)get_item(QSP_ARG  prec_itp, name);
}

#define DECLARE_ASSIGN_REAL_SCALAR_FUNC(stem,type,member)		\
									\
static int stem##_assign_scalar(Data_Obj *dp, Scalar_Value *svp)	\
{									\
	*((type     *)OBJ_DATA_PTR(dp)) = svp->member ;			\
	return 0;							\
}

DECLARE_ASSIGN_REAL_SCALAR_FUNC(char,char,u_b)
DECLARE_ASSIGN_REAL_SCALAR_FUNC(byte,char,u_b)
DECLARE_ASSIGN_REAL_SCALAR_FUNC(short,short,u_s)
DECLARE_ASSIGN_REAL_SCALAR_FUNC(int32,int32_t,u_l)
DECLARE_ASSIGN_REAL_SCALAR_FUNC(int64,int64_t,u_ll)
DECLARE_ASSIGN_REAL_SCALAR_FUNC(u_byte,u_char,u_ub)	// also PREC_CHAR
DECLARE_ASSIGN_REAL_SCALAR_FUNC(u_short,u_short,u_us)
DECLARE_ASSIGN_REAL_SCALAR_FUNC(uint32,uint32_t,u_ul)
DECLARE_ASSIGN_REAL_SCALAR_FUNC(uint64,uint64_t,u_ull)
DECLARE_ASSIGN_REAL_SCALAR_FUNC(float,float,u_f)
DECLARE_ASSIGN_REAL_SCALAR_FUNC(double,double,u_d)

#define DECLARE_ASSIGN_CPX_SCALAR_FUNC(stem,type,member)		\
									\
static int stem##_assign_scalar(Data_Obj *dp, Scalar_Value *svp)	\
{									\
	*( (type  *)OBJ_DATA_PTR(dp)  ) = svp->member[0];		\
	*(((type *)OBJ_DATA_PTR(dp))+1) = svp->member[1];		\
	return 0;							\
}

DECLARE_ASSIGN_CPX_SCALAR_FUNC(complex,float,u_fc)
DECLARE_ASSIGN_CPX_SCALAR_FUNC(dblcpx,double,u_dc)

#define DECLARE_ASSIGN_QUAT_SCALAR_FUNC(stem,type,member)		\
									\
static int stem##_assign_scalar(Data_Obj *dp, Scalar_Value *svp)	\
{									\
	*( (type  *)OBJ_DATA_PTR(dp)  ) = svp->member[0];		\
	*(((type *)OBJ_DATA_PTR(dp))+1) = svp->member[1];		\
	*(((type *)OBJ_DATA_PTR(dp))+2) = svp->member[2];		\
	*(((type *)OBJ_DATA_PTR(dp))+3) = svp->member[3];		\
	return 0;							\
}

DECLARE_ASSIGN_QUAT_SCALAR_FUNC(quaternion,float,u_fq)
DECLARE_ASSIGN_QUAT_SCALAR_FUNC(dblquat,double,u_dq)

static int string_assign_scalar(Data_Obj *dp, Scalar_Value *svp)
{
	return -1;
}

static int void_assign_scalar(Data_Obj *dp, Scalar_Value *svp)
{
	return -1;
}

static int color_assign_scalar(Data_Obj *dp, Scalar_Value *svp)
{
	*( (float  *)OBJ_DATA_PTR(dp)  ) = svp->u_color_comp[0];
	*(((float *)OBJ_DATA_PTR(dp))+1) = svp->u_color_comp[1];
	*(((float *)OBJ_DATA_PTR(dp))+2) = svp->u_color_comp[2];
	return 0;
}

static int bit_assign_scalar(Data_Obj *dp,Scalar_Value *svp)
{
	if( svp->u_l )
		*( (u_long *)OBJ_DATA_PTR(dp) ) |= 1 << OBJ_BIT0(dp) ;
	else
		*( (u_long *)OBJ_DATA_PTR(dp) ) &= ~( 1 << OBJ_BIT0(dp) );
	return 0;
}

////////////////////////////

#define DECLARE_CAST_TO_DOUBLE_FUNC(stem,member)			\
									\
static double cast_##stem##_to_double(Scalar_Value *svp)		\
{									\
	return (double) svp->u_b;					\
}

DECLARE_CAST_TO_DOUBLE_FUNC(char,u_b)
DECLARE_CAST_TO_DOUBLE_FUNC(byte,u_b)
DECLARE_CAST_TO_DOUBLE_FUNC(short,u_s)
DECLARE_CAST_TO_DOUBLE_FUNC(int32,u_l)
DECLARE_CAST_TO_DOUBLE_FUNC(int64,u_ll)
DECLARE_CAST_TO_DOUBLE_FUNC(u_byte,u_ub)
DECLARE_CAST_TO_DOUBLE_FUNC(u_short,u_us)
DECLARE_CAST_TO_DOUBLE_FUNC(uint32,u_ul)
DECLARE_CAST_TO_DOUBLE_FUNC(uint64,u_ull)
DECLARE_CAST_TO_DOUBLE_FUNC(float,u_f)
DECLARE_CAST_TO_DOUBLE_FUNC(double,u_d)

#define DECLARE_BAD_CAST_TO_DOUBLE_FUNC(stem)				\
									\
static double cast_##stem##_to_double(Scalar_Value *svp)		\
{									\
	sprintf(DEFAULT_ERROR_STRING,					\
		"Can't cast %s to double!?",#stem);			\
	NWARN(DEFAULT_ERROR_STRING);					\
	return 0.0;							\
}

DECLARE_BAD_CAST_TO_DOUBLE_FUNC(complex)
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(dblcpx)
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(quaternion)
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(dblquat)
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(color)
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(string)
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(void)

static double cast_bit_to_double(Scalar_Value *svp)
{
	if( svp->u_l )  return 1;
	else		return 0;
}

////////////////////////////////

#define DECLARE_CAST_FROM_DOUBLE_FUNC(stem,type,member)			\
									\
static void cast_##stem##_from_double(Scalar_Value *svp, double val)	\
{									\
	svp->member = (type) val;					\
}


DECLARE_CAST_FROM_DOUBLE_FUNC(char,char,u_b)
DECLARE_CAST_FROM_DOUBLE_FUNC(byte,char,u_b)
DECLARE_CAST_FROM_DOUBLE_FUNC(short,short,u_s)
DECLARE_CAST_FROM_DOUBLE_FUNC(int32,int32_t,u_l)
DECLARE_CAST_FROM_DOUBLE_FUNC(int64,int64_t,u_ll)
DECLARE_CAST_FROM_DOUBLE_FUNC(u_byte,u_char,u_ub)	// also PREC_CHAR
DECLARE_CAST_FROM_DOUBLE_FUNC(u_short,u_short,u_us)
DECLARE_CAST_FROM_DOUBLE_FUNC(uint32,uint32_t,u_ul)
DECLARE_CAST_FROM_DOUBLE_FUNC(uint64,uint64_t,u_ull)
DECLARE_CAST_FROM_DOUBLE_FUNC(float,float,u_f)
DECLARE_CAST_FROM_DOUBLE_FUNC(double,double,u_d)

#define DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(stem)				\
									\
static void cast_##stem##_from_double(Scalar_Value *svp, double val)	\
{									\
	sprintf(DEFAULT_ERROR_STRING,					\
		"Can't cast %s from double!?",#stem);			\
	NWARN(DEFAULT_ERROR_STRING);					\
}

// BUG?  use bitmap word type member???

static void cast_bit_from_double(Scalar_Value *svp,double val)
{
	if( val == 0.0 )
		svp->u_l = 0;
	else
		svp->u_l = 1;
}

DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(complex)
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(dblcpx)
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(quaternion)
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(dblquat)
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(string)
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(color)
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(void)

////////////////////////////

#define DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(stem,type,member)	\
									\
static void cast_indexed_##stem##_from_double				\
			(Scalar_Value *svp, int idx, double val)	\
{									\
	svp->member[idx] = (type) val;					\
}

DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(complex,float,u_fc)
DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(dblcpx,double,u_dc)
DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(quaternion,float,u_fq)
DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(dblquat,double,u_dq)
DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(color,float,u_color_comp)

#define DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(stem)		\
									\
static void cast_indexed_##stem##_from_double				\
			(Scalar_Value *svp, int idx, double val)	\
{									\
	sprintf(DEFAULT_ERROR_STRING,					\
		"Can't cast to %s with an index!?",#stem);		\
	NWARN(DEFAULT_ERROR_STRING);					\
}

DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(char)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(byte)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(short)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(int32)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(int64)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(u_byte)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(u_short)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(uint32)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(uint64)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(float)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(double)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(bit)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(void)
// BUG - we could do string!!!
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(string)

/////////////////////////////

void init_precisions(SINGLE_QSP_ARG_DECL)
{
	Precision *prec_p;

	prec_itp = new_item_type(QSP_ARG  "Precision", LIST_CONTAINER);	// used to be hashed, but not many of these?
									// should sort based on access?

	INIT_PREC(byte,char,PREC_BY)
	INIT_PREC(u_byte,u_char,PREC_UBY)
	INIT_PREC(short,short,PREC_IN)
	INIT_PREC(u_short,u_short,PREC_UIN)
	INIT_PREC(int32,int32_t,PREC_DI)
	INIT_PREC(uint32,uint32_t,PREC_UDI)
	INIT_PREC(int64,int64_t,PREC_LI)
	INIT_PREC(uint64,uint64_t,PREC_ULI)
	INIT_PREC(float,float,PREC_SP)
	INIT_PREC(double,double,PREC_DP)

	INIT_PREC(complex,SP_Complex,PREC_CPX)
	INIT_PREC(dblcpx,DP_Complex,PREC_DBLCPX)
	INIT_PREC(quaternion,SP_Quaternion,PREC_QUAT)
	INIT_PREC(dblquat,DP_Quaternion,PREC_DBLQUAT)
	INIT_PREC(char,char,PREC_CHAR)
	INIT_PREC(string,char,PREC_STR)
	INIT_PREC(color,SP_Color,PREC_COLOR)
	INIT_PREC(bit,BITMAP_DATA_TYPE,PREC_BIT)
	INIT_VOID_PREC(void,void,PREC_VOID)

#ifdef USE_LONG_DOUBLE
	INIT_PREC(long_dbl,long double,PREC_LP)
	INIT_PREC(ldblcpx,LP_Complex,PREC_LDBLCPX)
	INIT_PREC(ldblquat,LP_Quaternion,PREC_LDBLQUAT)
#endif // USE_LONG_DOUBLE
}


List *prec_list(SINGLE_QSP_ARG_DECL)
{
	if( prec_itp == NULL ){
		init_precisions(SINGLE_QSP_ARG);
	}

	return item_list(QSP_ARG  prec_itp);
}

Precision *const_precision(Precision *prec_p)
{
	NERROR1("Sorry, const_precision not implemented yet.");
	return NULL;
}

Precision *complex_precision(Precision *prec_p)
{
	assert( ! COMPLEX_PRECISION(PREC_CODE(prec_p)) );

	return PREC_FOR_CODE( PREC_CODE(prec_p) | DT_COMPLEX );
}

