#include "quip_config.h"

#include <math.h>	// FLT_MIN etc
#include "quip_prot.h"
#include "dobj_prot.h"
#include "dobj_private.h"
#include "ascii_fmts.h"
#include "query_stack.h"	// like to eliminate this dependency...
#include "debug.h"

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
	SET_PREC_SET_VALUE_FROM_INPUT_FUNC(prec_p,name##_set_value_from_input);	\
	SET_PREC_INDEXED_DATA_FUNC(prec_p,name##_indexed_data);	\
	SET_PREC_IS_NUMERIC_FUNC(prec_p,name##_is_numeric);	\
	SET_PREC_ASSIGN_SCALAR_FUNC(prec_p,name##_assign_scalar);	\
	SET_PREC_EXTRACT_SCALAR_FUNC(prec_p,name##_extract_scalar);	\
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

/////////////////////////////////

#define DECLARE_BAD_SET_VALUE_FROM_INPUT_FUNC(stem)									\
															\
static void stem##_set_value_from_input(QSP_ARG_DECL  void *vp)								\
{															\
	assert( AERROR(#stem"_set_value_from_input should never be called, not a machine precision!?") );		\
}

// BUG need special case for bitmap!

#define DECLARE_SET_VALUE_FROM_INPUT_FUNC(stem,type,read_type,query_func,prompt,next_input_func,type_min,type_max)	\
															\
static void stem##_set_value_from_input(QSP_ARG_DECL  void *vp)								\
{															\
	read_type val;													\
															\
	if( ! HAS_FORMAT_LIST )												\
		val = query_func(QSP_ARG  prompt );									\
	else														\
		val = next_input_func(QSP_ARG  prompt);									\
															\
	if( val < type_min || val > type_max ){										\
		sprintf(ERROR_STRING,"Truncation error converting to %s",#stem);					\
		WARN(ERROR_STRING);											\
	}														\
															\
	* ((type *)vp) = (type) val;											\
}

DECLARE_SET_VALUE_FROM_INPUT_FUNC(float,float,double,how_much,"real data",next_input_flt_with_format,__FLT_MIN__,__FLT_MAX__)
DECLARE_SET_VALUE_FROM_INPUT_FUNC(double,double,double,how_much,"real data",next_input_flt_with_format,__DBL_MIN__,__DBL_MAX__)

DECLARE_SET_VALUE_FROM_INPUT_FUNC(byte,char,long,how_many,"integer data",next_input_int_with_format,MIN_BYTE,MAX_BYTE)
DECLARE_SET_VALUE_FROM_INPUT_FUNC(short,short,long,how_many,"integer data",next_input_int_with_format,MIN_SHORT,MAX_SHORT)
DECLARE_SET_VALUE_FROM_INPUT_FUNC(int32,int32_t,long,how_many,"integer data",next_input_int_with_format,MIN_INT32,MAX_INT32)
DECLARE_SET_VALUE_FROM_INPUT_FUNC(int64,int64_t,long,how_many,"integer data",next_input_int_with_format,MIN_INT64,MAX_INT64)

DECLARE_SET_VALUE_FROM_INPUT_FUNC(u_byte,u_char,long,how_many,"integer data",next_input_int_with_format,MIN_UBYTE,MAX_UBYTE)
DECLARE_SET_VALUE_FROM_INPUT_FUNC(u_short,u_short,long,how_many,"integer data",next_input_int_with_format,MIN_USHORT,MAX_USHORT)
DECLARE_SET_VALUE_FROM_INPUT_FUNC(uint32,int32_t,long,how_many,"integer data",next_input_int_with_format,MIN_UINT32,MAX_UINT32)
DECLARE_SET_VALUE_FROM_INPUT_FUNC(uint64,int64_t,long,how_many,"integer data",next_input_int_with_format,MIN_UINT64,MAX_UINT64)

/////////////////////////////////

#define DECLARE_INDEXED_DATA_FUNC(stem,type)				\
									\
static double stem##_indexed_data(Data_Obj *dp, int index)		\
{									\
	return (double) (* (((type *)OBJ_DATA_PTR(dp))+index) );	\
}

static inline double fetch_bit(Data_Obj *dp, bitnum_t bitnum)
{
	bitmap_word bit, *word_p;

	bitnum += OBJ_BIT0(dp);
	word_p = (bitmap_word *)OBJ_DATA_PTR(dp);
	word_p += bitnum/BITS_PER_BITMAP_WORD;
	bitnum %= BITS_PER_BITMAP_WORD;
	bit = 1 << bitnum;
	if( *word_p & bit )
		return 1.0;
	else
		return 0.0;
}

#define DECLARE_POSSIBLY_BITMAP_INDEXED_DATA_FUNC(stem,type)			\
										\
static double stem##_indexed_data(Data_Obj *dp, int index)			\
{										\
	if( IS_BITMAP(dp) ){							\
		return fetch_bit( dp, OBJ_BIT0(dp)+index );			\
	} else {								\
		return (double) (* (((type *)OBJ_DATA_PTR(dp))+index) );	\
	}									\
}


#define DECLARE_BAD_INDEXED_DATA_FUNC(stem)				\
									\
static double stem##_indexed_data(Data_Obj *dp, int index)		\
{									\
	assert( AERROR(#stem" indexed data function does not exist (not a machine precision)!?") );	\
}


/////////////////////////////////

#define DECLARE_IS_NUMERIC_FUNC(stem)		\
						\
static int stem##_is_numeric(void)		\
{						\
	return 1;				\
}

#define DECLARE_NOT_NUMERIC_FUNC(stem)		\
						\
static int stem##_is_numeric(void)		\
{						\
	return 0;				\
}

/////////////////////////////////

#define DECLARE_ASSIGN_REAL_SCALAR_FUNC(stem,type,member)		\
									\
static int stem##_assign_scalar(Data_Obj *dp, Scalar_Value *svp)		\
{									\
	*((type *)OBJ_DATA_PTR(dp)) = svp->member ;					\
	return 0;							\
}

#define DECLARE_ASSIGN_CPX_SCALAR_FUNC(stem,type,member)		\
									\
static int stem##_assign_scalar(Data_Obj *dp, Scalar_Value *svp)		\
{									\
	*( (type *)(OBJ_DATA_PTR(dp))  ) = svp->member[0];				\
	*(((type *)(OBJ_DATA_PTR(dp)))+1) = svp->member[1];				\
	return 0;							\
}

#define DECLARE_ASSIGN_QUAT_SCALAR_FUNC(stem,type,member)		\
									\
static int stem##_assign_scalar(Data_Obj *dp, Scalar_Value *svp)		\
{									\
	*( (type *)(OBJ_DATA_PTR(dp))  ) = svp->member[0];				\
	*(((type *)(OBJ_DATA_PTR(dp)))+1) = svp->member[1];				\
	*(((type *)(OBJ_DATA_PTR(dp)))+2) = svp->member[2];				\
	*(((type *)(OBJ_DATA_PTR(dp)))+3) = svp->member[3];				\
	return 0;							\
}

#define DECLARE_ASSIGN_COLOR_SCALAR_FUNC(stem,type,member)		\
									\
static int stem##_assign_scalar(Data_Obj *dp, Scalar_Value *svp)		\
{									\
	*( (type *)(OBJ_DATA_PTR(dp))   ) = svp->member[0];				\
	*(((type *)(OBJ_DATA_PTR(dp)))+1) = svp->member[1];				\
	*(((type *)(OBJ_DATA_PTR(dp)))+2) = svp->member[2];				\
	return 0;							\
}

#define DECLARE_BAD_ASSIGN_SCALAR_FUNC(stem)				\
									\
static int stem##_assign_scalar(Data_Obj *dp, Scalar_Value *svp)		\
{									\
	return -1;							\
}

//////////////////////////////////////////////

#define DECLARE_EXTRACT_REAL_SCALAR_FUNC(stem,type,member)		\
									\
static void stem##_extract_scalar(Scalar_Value *svp, Data_Obj *dp)		\
{									\
	svp->member = *((type     *)OBJ_DATA_PTR(dp));				\
}

#define DECLARE_EXTRACT_CPX_SCALAR_FUNC(stem,type,member)		\
									\
static void stem##_extract_scalar(Scalar_Value *svp, Data_Obj *dp)		\
{									\
	svp->member[0] = *(((type *)OBJ_DATA_PTR(dp)));				\
	svp->member[1] = *(((type *)OBJ_DATA_PTR(dp))+1);				\
}

#define DECLARE_EXTRACT_QUAT_SCALAR_FUNC(stem,type,member)		\
									\
static void stem##_extract_scalar(Scalar_Value *svp, Data_Obj *dp)		\
{									\
	svp->member[0] = *(((type *)OBJ_DATA_PTR(dp)));				\
	svp->member[1] = *(((type *)OBJ_DATA_PTR(dp))+1);				\
	svp->member[2] = *(((type *)OBJ_DATA_PTR(dp))+2);				\
	svp->member[3] = *(((type *)OBJ_DATA_PTR(dp))+3);				\
}

#define DECLARE_EXTRACT_COLOR_SCALAR_FUNC(stem,type,member)		\
									\
static void stem##_extract_scalar(Scalar_Value *svp, Data_Obj *dp)		\
{									\
	svp->member[0] = *(((type *)OBJ_DATA_PTR(dp)));				\
	svp->member[1] = *(((type *)OBJ_DATA_PTR(dp))+1);				\
	svp->member[2] = *(((type *)OBJ_DATA_PTR(dp))+2);				\
}

#define DECLARE_BAD_EXTRACT_SCALAR_FUNC(stem)				\
									\
static void stem##_extract_scalar(Scalar_Value *svp, Data_Obj *dp)		\
{									\
	sprintf(DEFAULT_ERROR_STRING,"Nonsensical call to %s_extract_scalar!?", #stem);\
	NWARN(DEFAULT_ERROR_STRING);					\
}

////////////////////////////

#define DECLARE_CAST_TO_DOUBLE_FUNC(stem,member)			\
									\
static double cast_##stem##_to_double(Scalar_Value *svp)		\
{									\
	return (double) svp->member;					\
}

#define DECLARE_BAD_CAST_TO_DOUBLE_FUNC(stem)				\
									\
static double cast_##stem##_to_double(Scalar_Value *svp)		\
{									\
	sprintf(DEFAULT_ERROR_STRING,					\
		"Can't cast %s to double!?",#stem);			\
	NWARN(DEFAULT_ERROR_STRING);					\
	return 0.0;							\
}

////////////////////////////////

#define DECLARE_CAST_FROM_DOUBLE_FUNC(stem,type,member)			\
									\
static void cast_##stem##_from_double(Scalar_Value *svp, double val)	\
{									\
	svp->member = (type) val;					\
}

#define DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(stem)				\
									\
static void cast_##stem##_from_double(Scalar_Value *svp, double val)	\
{									\
	sprintf(DEFAULT_ERROR_STRING,					\
		"Can't cast %s from double!?",#stem);			\
	NWARN(DEFAULT_ERROR_STRING);					\
}

////////////////////////////

#define DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(stem,type,member)	\
									\
static void cast_indexed_##stem##_from_double				\
			(Scalar_Value *svp, int idx, double val)	\
{									\
	svp->member[idx] = (type) val;					\
}


#define DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(stem)		\
									\
static void cast_indexed_##stem##_from_double				\
			(Scalar_Value *svp, int idx, double val)	\
{									\
	sprintf(DEFAULT_ERROR_STRING,					\
		"Can't cast to %s with an index!?",#stem);		\
	NWARN(DEFAULT_ERROR_STRING);					\
}

/////////////////////////////

#define DECLARE_REAL_SCALAR_FUNCS(stem,type,member)			\
									\
DECLARE_ALMOST_REAL_SCALAR_FUNCS(stem,type,member)			\
DECLARE_INDEXED_DATA_FUNC(stem,type)

#define DECLARE_BITMAP_REAL_SCALAR_FUNCS(stem,type,member)		\
									\
DECLARE_ALMOST_REAL_SCALAR_FUNCS(stem,type,member)			\
DECLARE_POSSIBLY_BITMAP_INDEXED_DATA_FUNC(stem,type)

#define DECLARE_ALMOST_REAL_SCALAR_FUNCS(stem,type,member)			\
									\
DECLARE_IS_NUMERIC_FUNC(stem)				\
DECLARE_CAST_FROM_DOUBLE_FUNC(stem,type,member)				\
DECLARE_CAST_TO_DOUBLE_FUNC(stem,member)				\
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(stem)			\
DECLARE_ASSIGN_REAL_SCALAR_FUNC(stem,type,member)			\
DECLARE_EXTRACT_REAL_SCALAR_FUNC(stem,type,member)


#define DECLARE_CPX_SCALAR_FUNCS(stem,type,member)			\
									\
DECLARE_BAD_SET_VALUE_FROM_INPUT_FUNC(stem)				\
DECLARE_BAD_INDEXED_DATA_FUNC(stem)				\
DECLARE_IS_NUMERIC_FUNC(stem)				\
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(stem)					\
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(stem)					\
DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(stem,type,member)		\
DECLARE_ASSIGN_CPX_SCALAR_FUNC(stem,type,member)			\
DECLARE_EXTRACT_CPX_SCALAR_FUNC(stem,type,member)


#define DECLARE_QUAT_SCALAR_FUNCS(stem,type,member)			\
									\
DECLARE_BAD_SET_VALUE_FROM_INPUT_FUNC(stem)				\
DECLARE_BAD_INDEXED_DATA_FUNC(stem)				\
DECLARE_IS_NUMERIC_FUNC(stem)				\
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(stem)					\
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(stem)					\
DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(stem,type,member)		\
DECLARE_ASSIGN_QUAT_SCALAR_FUNC(stem,type,member)			\
DECLARE_EXTRACT_QUAT_SCALAR_FUNC(stem,type,member)


#define DECLARE_COLOR_SCALAR_FUNCS(stem,type,member)			\
									\
DECLARE_BAD_SET_VALUE_FROM_INPUT_FUNC(stem)				\
DECLARE_BAD_INDEXED_DATA_FUNC(stem)				\
DECLARE_IS_NUMERIC_FUNC(stem)				\
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(stem)					\
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(stem)					\
DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(stem,type,member)		\
DECLARE_ASSIGN_COLOR_SCALAR_FUNC(stem,type,member)			\
DECLARE_EXTRACT_COLOR_SCALAR_FUNC(stem,type,member)

////////////////////////////////

static int bit_assign_scalar(Data_Obj *dp,Scalar_Value *svp)
{
	if( svp->bitmap_scalar )
		*( (BITMAP_DATA_TYPE *)OBJ_DATA_PTR(dp) ) |= 1 << OBJ_BIT0(dp) ;
	else
		*( (BITMAP_DATA_TYPE *)OBJ_DATA_PTR(dp) ) &= ~( 1 << OBJ_BIT0(dp) );
	return 0;
}

static void bit_extract_scalar(Scalar_Value *svp, Data_Obj *dp)
{
	svp->bitmap_scalar = *( (BITMAP_DATA_TYPE *)OBJ_DATA_PTR(dp) ) & (1 << OBJ_BIT0(dp)) ;
}


static void cast_bit_from_double(Scalar_Value *svp,double val)
{
	if( val == 0.0 )
		svp->bitmap_scalar = 0;
	else
		svp->bitmap_scalar = 1;
}

static double cast_bit_to_double(Scalar_Value *svp)
{
	if( svp->bitmap_scalar )  return 1;
	else		return 0;
}

DECLARE_IS_NUMERIC_FUNC(bit)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(bit)
DECLARE_BAD_INDEXED_DATA_FUNC(bit)
DECLARE_BAD_SET_VALUE_FROM_INPUT_FUNC(bit)

/////////////////////////////////

// The machine precisions
DECLARE_REAL_SCALAR_FUNCS(byte,char,u_b)
DECLARE_REAL_SCALAR_FUNCS(short,short,u_s)
DECLARE_REAL_SCALAR_FUNCS(int32,int32_t,u_l)
DECLARE_REAL_SCALAR_FUNCS(int64,int64_t,u_ll)
DECLARE_REAL_SCALAR_FUNCS(u_byte,u_char,u_ub)
DECLARE_REAL_SCALAR_FUNCS(u_short,u_short,u_us)
DECLARE_REAL_SCALAR_FUNCS(float,float,u_f)
DECLARE_REAL_SCALAR_FUNCS(double,double,u_d)

#ifdef BITMAP_WORD_IS_64_BITS
DECLARE_REAL_SCALAR_FUNCS(uint32,uint32_t,u_ul)
DECLARE_BITMAP_REAL_SCALAR_FUNCS(uint64,uint64_t,u_ull)
#else // ! BITMAP_WORD_IS_64_BITS
DECLARE_BITMAP_REAL_SCALAR_FUNCS(uint32,uint32_t,u_ul)
DECLARE_REAL_SCALAR_FUNCS(uint64,uint64_t,u_ull)
#endif // ! BITMAP_WORD_IS_64_BITS

DECLARE_ALMOST_REAL_SCALAR_FUNCS(char,char,u_b)
DECLARE_BAD_INDEXED_DATA_FUNC(char)
DECLARE_BAD_SET_VALUE_FROM_INPUT_FUNC(char)

DECLARE_CPX_SCALAR_FUNCS(complex,float,u_fc)
DECLARE_CPX_SCALAR_FUNCS(dblcpx,double,u_dc)

DECLARE_QUAT_SCALAR_FUNCS(quaternion,float,u_fq)
DECLARE_QUAT_SCALAR_FUNCS(dblquat,double,u_dq)

DECLARE_COLOR_SCALAR_FUNCS(color,float,u_color_comp)

DECLARE_NOT_NUMERIC_FUNC(string)
DECLARE_BAD_ASSIGN_SCALAR_FUNC(string)
DECLARE_BAD_EXTRACT_SCALAR_FUNC(string)
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(string)
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(string)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(string) // BUG - we could do string!!!
DECLARE_BAD_INDEXED_DATA_FUNC(string)
DECLARE_BAD_SET_VALUE_FROM_INPUT_FUNC(string)

DECLARE_NOT_NUMERIC_FUNC(void)
DECLARE_BAD_ASSIGN_SCALAR_FUNC(void)
DECLARE_BAD_EXTRACT_SCALAR_FUNC(void)
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(void)
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(void)
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(void)
DECLARE_BAD_INDEXED_DATA_FUNC(void)
DECLARE_BAD_SET_VALUE_FROM_INPUT_FUNC(void)


////////////////////////////////

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

