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

#define new_prec(name)	_new_prec(QSP_ARG  name)

#define INIT_GENERIC_PREC(quip_prec_name,type,code)					\
	prec_p = new_prec(#quip_prec_name);						\
	SET_PREC_CODE(prec_p, code);							\
	SET_PREC_N_COMPS(prec_p, 1);							\
	SET_PREC_SET_VALUE_FROM_INPUT_FUNC(prec_p,quip_prec_name##_set_value_from_input);	\
	SET_PREC_INDEXED_DATA_FUNC(prec_p,quip_prec_name##_indexed_data);		\
	SET_PREC_IS_NUMERIC_FUNC(prec_p,quip_prec_name##_is_numeric);			\
	SET_PREC_ASSIGN_SCALAR_FUNC(prec_p,quip_prec_name##_assign_scalar_obj);		\
	SET_PREC_EXTRACT_SCALAR_FUNC(prec_p,_##quip_prec_name##_extract_scalar);	\
	SET_PREC_CAST_TO_DOUBLE_FUNC(prec_p,_cast_##quip_prec_name##_to_double);	\
	SET_PREC_CAST_FROM_DOUBLE_FUNC(prec_p,_cast_##quip_prec_name##_from_double);	\
	SET_PREC_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(prec_p,_cast_indexed_##quip_prec_name##_from_double);	\
	SET_PREC_COPY_VALUE_FUNC(prec_p,copy_##quip_prec_name##_value);			\
	SET_PREC_FORMAT_FUNC(prec_p,_format_##quip_prec_name##_value);			\
	SET_PREC_VAL_COMP_FUNC(prec_p,compare_##quip_prec_name##_values);		\
	SET_PREC_IDX_COMP_FUNC(prec_p,compare_indexed_##quip_prec_name##s);		\
	if( (code & PSEUDO_PREC_MASK) == 0 )						\
		SET_PREC_MACH_PREC_PTR(prec_p, prec_p);					\
	else { /* special handling for pseudo-precisions */				\
		SET_PREC_MACH_PREC_PTR(prec_p,						\
			prec_for_code( code & MACH_PREC_MASK ) );			\
		if( code == PREC_CPX || code == PREC_DBLCPX )				\
			SET_PREC_N_COMPS(prec_p,2);					\
		if( code == PREC_QUAT || code == PREC_DBLQUAT )				\
			SET_PREC_N_COMPS(prec_p,4);					\
		if( code == PREC_COLOR )						\
			SET_PREC_N_COMPS(prec_p,3);					\
	}

static Precision *_new_prec(QSP_ARG_DECL  const char *name)
{
	Precision *prec_p;

	prec_p = (Precision *) new_item(prec_itp, name, sizeof(Precision) );
	assert( prec_p != NULL );

	return(prec_p);
}

Precision *_get_prec(QSP_ARG_DECL  const char *name)
{
	return (Precision *)get_item(prec_itp, name);
}

// We generally use the long format
// We would like to show the sign for decimal display,
// but just the bits for hex and octal...

#define DECLARE_INT_FORMAT_FUNC(type_str)								\
													\
static void _format_##type_str##_value(QSP_ARG_DECL  char *buf, Scalar_Value *svp, int pad_flag)	\
{													\
	Integer_Output_Fmt *iof_p;									\
													\
	iof_p = curr_output_int_fmt_p;									\
	(*(iof_p->iof_fmt_##type_str##_func))(QSP_ARG  buf,svp,pad_flag);				\
													\
}

DECLARE_INT_FORMAT_FUNC(string)
DECLARE_INT_FORMAT_FUNC(char)
DECLARE_INT_FORMAT_FUNC(byte)
DECLARE_INT_FORMAT_FUNC(short)
DECLARE_INT_FORMAT_FUNC(int)
DECLARE_INT_FORMAT_FUNC(long)
DECLARE_INT_FORMAT_FUNC(u_byte)
DECLARE_INT_FORMAT_FUNC(u_short)
DECLARE_INT_FORMAT_FUNC(u_int)
DECLARE_INT_FORMAT_FUNC(u_long)

#define DECLARE_FLT_FORMAT_FUNC(type_str,member)							\
													\
static void _format_##type_str##_value(QSP_ARG_DECL  char *buf, Scalar_Value *svp, int pad_flag)	\
{													\
	if( pad_flag )											\
		sprintf(buf,padded_flt_fmt_str,svp->member);						\
	else												\
		sprintf(buf,"%g",svp->member);								\
}

DECLARE_FLT_FORMAT_FUNC(float,u_f)
DECLARE_FLT_FORMAT_FUNC(double,u_d)

#define DECLARE_INVALID_FORMAT_FUNC(type_str,member)							\
													\
static void _format_##type_str##_value(QSP_ARG_DECL  char *buf, Scalar_Value *svp, int pad_flag)	\
{													\
	sprintf(ERROR_STRING,"CAUTIOUS:  '%s' is not a machine precision, can't format value!?",#type_str);	\
	error1(ERROR_STRING);										\
}

DECLARE_INVALID_FORMAT_FUNC(complex,u_f)
DECLARE_INVALID_FORMAT_FUNC(dblcpx,u_d)
DECLARE_INVALID_FORMAT_FUNC(quaternion,u_f)
DECLARE_INVALID_FORMAT_FUNC(dblquat,u_d)
DECLARE_INVALID_FORMAT_FUNC(color,u_f)
DECLARE_INVALID_FORMAT_FUNC(bit,u_b)
DECLARE_INVALID_FORMAT_FUNC(void,u_b)

/////////////////////////////////

#define DECLARE_VAL_COMPARISON_FUNC(prec_name,type)						\
static int compare_##prec_name##_values(const void *p1, const void *p2)			\
{											\
	if( *((const type *)p1) > *((const type *)p2) ) return 1;			\
	else if( *((const type *)p1) < *((const type *)p2) ) return -1;			\
	else return 0;									\
}

DECLARE_VAL_COMPARISON_FUNC(byte,char)
DECLARE_VAL_COMPARISON_FUNC(u_byte,unsigned char)
DECLARE_VAL_COMPARISON_FUNC(short,short)
DECLARE_VAL_COMPARISON_FUNC(u_short,unsigned short)
DECLARE_VAL_COMPARISON_FUNC(int,int32_t)
DECLARE_VAL_COMPARISON_FUNC(u_int,uint32_t)
DECLARE_VAL_COMPARISON_FUNC(long,int64_t)
DECLARE_VAL_COMPARISON_FUNC(u_long,uint64_t)
DECLARE_VAL_COMPARISON_FUNC(float,float)
DECLARE_VAL_COMPARISON_FUNC(double,double)

#define INVALID_VAL_COMPARISON_FUNC(prec_name)						\
static int compare_##prec_name##_values(const void *p1, const void *p2)			\
{											\
	_error1(DEFAULT_QSP_ARG  "CAUTIOUS:  Illegal sort attempt on " #prec_name " data!?");		\
	return 0;									\
}

INVALID_VAL_COMPARISON_FUNC(complex)
INVALID_VAL_COMPARISON_FUNC(dblcpx)
INVALID_VAL_COMPARISON_FUNC(quaternion)
INVALID_VAL_COMPARISON_FUNC(dblquat)
INVALID_VAL_COMPARISON_FUNC(color)
INVALID_VAL_COMPARISON_FUNC(bit)
INVALID_VAL_COMPARISON_FUNC(char)
INVALID_VAL_COMPARISON_FUNC(string)
INVALID_VAL_COMPARISON_FUNC(void)

/////////////////////////////////

#define DECLARE_IDX_COMPARISON_FUNC(prec_name,type)					\
											\
static int compare_indexed_##prec_name##s(INDEX_SORT_DATA_DP_ARG const void *ptr1, const void *ptr2)	\
{											\
	INDEX_TYPE i1, i2, inc;								\
	type *p1, *p2;									\
	INDEX_SORT_DATA_DP_DECL								\
											\
	assert(dp!=NULL);								\
											\
	i1 = *((const INDEX_TYPE *)ptr1);						\
	i2 = *((const INDEX_TYPE *)ptr2);						\
											\
	inc = OBJ_TYPE_INC(index_sort_data_dp, OBJ_MINDIM(index_sort_data_dp) );	\
											\
	p1 = ((type *)OBJ_DATA_PTR(index_sort_data_dp)) + i1*inc;			\
	p2 = ((type *)OBJ_DATA_PTR(index_sort_data_dp)) + i2*inc;			\
											\
	if( *p1 > *p2 ) return(1);							\
	else if( *p1 < *p2 ) return(-1);						\
	else return(0);									\
}

DECLARE_IDX_COMPARISON_FUNC(byte,char)
DECLARE_IDX_COMPARISON_FUNC(u_byte,unsigned char)
DECLARE_IDX_COMPARISON_FUNC(short,short)
DECLARE_IDX_COMPARISON_FUNC(u_short,unsigned short)
DECLARE_IDX_COMPARISON_FUNC(int,int32_t)
DECLARE_IDX_COMPARISON_FUNC(u_int,uint32_t)
DECLARE_IDX_COMPARISON_FUNC(long,int64_t)
DECLARE_IDX_COMPARISON_FUNC(u_long,uint64_t)
DECLARE_IDX_COMPARISON_FUNC(float,float)
DECLARE_IDX_COMPARISON_FUNC(double,double)

#define INVALID_IDX_COMPARISON_FUNC(prec_name)						\
											\
static int compare_indexed_##prec_name##s(void *dp, const void *p1, const void *p2)	\
{											\
	_error1(DEFAULT_QSP_ARG  "CAUTIOUS:  Illegal sort attempt on indices of " #prec_name " data!?");		\
	return 0;									\
}

INVALID_IDX_COMPARISON_FUNC(complex)
INVALID_IDX_COMPARISON_FUNC(dblcpx)
INVALID_IDX_COMPARISON_FUNC(quaternion)
INVALID_IDX_COMPARISON_FUNC(dblquat)
INVALID_IDX_COMPARISON_FUNC(color)
INVALID_IDX_COMPARISON_FUNC(bit)
INVALID_IDX_COMPARISON_FUNC(char)
INVALID_IDX_COMPARISON_FUNC(string)
INVALID_IDX_COMPARISON_FUNC(void)


/////////////////////////////////

#define DECLARE_BAD_SET_VALUE_FROM_INPUT_FUNC(quip_type_name)						\
												\
static void quip_type_name##_set_value_from_input(QSP_ARG_DECL  void *vp, const char *prompt)		\
{												\
assert( AERROR(#quip_type_name"_set_value_from_input should never be called, not a machine precision!?") );\
}

// BUG need special case for bitmap!
// Floating point values aren't signed...

#define DECLARE_SET_INT_VALUE_FROM_INPUT_FUNC(quip_type_name,mem_type,read_type,query_func,next_input_func,type_min,type_max)	\
												\
static void quip_type_name##_set_value_from_input(QSP_ARG_DECL  void *vp, const char *prompt)		\
{												\
	read_type val;										\
												\
	if( ! HAS_FORMAT_LIST )									\
		val = query_func(prompt );							\
	else											\
		val = next_input_func(QSP_ARG  prompt);						\
												\
	if( val < type_min || val > type_max ){							\
		sprintf(ERROR_STRING,								\
			"%s_set_value_from_input:  Truncation error converting %s to %s (%s)",	\
			#quip_type_name,#read_type,#quip_type_name,#mem_type);						\
		warn(ERROR_STRING);								\
		sprintf(ERROR_STRING,"val = %ld, type_min = %ld, type_max = %ld",		\
			(long)val,(long)type_min,(long)type_max);				\
		advise(ERROR_STRING);								\
	}											\
												\
	if( vp != NULL )									\
		* ((mem_type *)vp) = (mem_type) val;							\
}

#define DECLARE_SET_FLT_VALUE_FROM_INPUT_FUNC(quip_type_name,type,read_type,query_func,next_input_func,type_min,type_max)	\
												\
static void quip_type_name##_set_value_from_input(QSP_ARG_DECL  void *vp, const char *prompt)		\
{												\
	read_type val;										\
												\
	if( ! HAS_FORMAT_LIST )									\
		val = query_func(prompt );							\
	else											\
		val = next_input_func(QSP_ARG  prompt);						\
												\
	if( val < (-type_max) || val > type_max ){						\
		sprintf(ERROR_STRING,"%s_set_value_from_input:  Truncation error converting %s to %s (%s)",#quip_type_name,#read_type,#quip_type_name,#type);		\
		warn(ERROR_STRING);								\
	}											\
												\
	if( (val < (type_min) && val > 0) || (val > (-type_min) && val < 0) ){			\
		sprintf(ERROR_STRING,"Rounding error converting %s to %s (%s)",#read_type,#quip_type_name,#type);			\
		warn(ERROR_STRING);								\
	}											\
												\
	if( vp != NULL )									\
		* ((type *)vp) = (type) val;							\
}

DECLARE_SET_FLT_VALUE_FROM_INPUT_FUNC(float,float,double,how_much,next_input_flt_with_format,__FLT_MIN__,__FLT_MAX__)
DECLARE_SET_FLT_VALUE_FROM_INPUT_FUNC(double,double,double,how_much,next_input_flt_with_format,__DBL_MIN__,__DBL_MAX__)

DECLARE_SET_INT_VALUE_FROM_INPUT_FUNC(byte,char,howmany_type,how_many,next_input_int_with_format,MIN_BYTE,MAX_BYTE)
DECLARE_SET_INT_VALUE_FROM_INPUT_FUNC(short,short,howmany_type,how_many,next_input_int_with_format,MIN_SHORT,MAX_SHORT)
DECLARE_SET_INT_VALUE_FROM_INPUT_FUNC(int,int32_t,howmany_type,how_many,next_input_int_with_format,MIN_INT32,MAX_INT32)
// This one generates warnings when building for iOS?
DECLARE_SET_INT_VALUE_FROM_INPUT_FUNC(long,int64_t,howmany_type,how_many,next_input_int_with_format,MIN_INT64,MAX_INT64)

DECLARE_SET_INT_VALUE_FROM_INPUT_FUNC(u_byte,u_char,howmany_type,how_many,next_input_int_with_format,MIN_UBYTE,MAX_UBYTE)
DECLARE_SET_INT_VALUE_FROM_INPUT_FUNC(u_short,u_short,howmany_type,how_many,next_input_int_with_format,MIN_USHORT,MAX_USHORT)
DECLARE_SET_INT_VALUE_FROM_INPUT_FUNC(u_int,int32_t,howmany_type,how_many,next_input_int_with_format,MIN_UINT32,MAX_UINT32)
DECLARE_SET_INT_VALUE_FROM_INPUT_FUNC(u_long,int64_t,howmany_type,how_many,next_input_int_with_format,MIN_UINT64,MAX_UINT64)

/////////////////////////////////

#define DECLARE_INDEXED_DATA_FUNC(quip_type_name,type)				\
									\
static double quip_type_name##_indexed_data(Data_Obj *dp, int index)		\
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

#define DECLARE_POSSIBLY_BITMAP_INDEXED_DATA_FUNC(quip_type_name,type)			\
										\
static double quip_type_name##_indexed_data(Data_Obj *dp, int index)			\
{										\
	if( IS_BITMAP(dp) ){							\
		return fetch_bit( dp, OBJ_BIT0(dp)+index );			\
	} else {								\
		return (double) (* (((type *)OBJ_DATA_PTR(dp))+index) );	\
	}									\
}


#define DECLARE_BAD_INDEXED_DATA_FUNC(quip_type_name)				\
									\
static double quip_type_name##_indexed_data(Data_Obj *dp, int index)		\
{									\
	assert( AERROR(#quip_type_name" indexed data function does not exist (not a machine precision)!?") );	\
}


/////////////////////////////////

#define DECLARE_IS_NUMERIC_FUNC(quip_type_name)		\
						\
static int quip_type_name##_is_numeric(void)		\
{						\
	return 1;				\
}

#define DECLARE_NOT_NUMERIC_FUNC(quip_type_name)		\
						\
static int quip_type_name##_is_numeric(void)		\
{						\
	return 0;				\
}

/////////////////////////////////

#define DECLARE_ASSIGN_REAL_SCALAR_FUNC(quip_type_name,type,member)		\
									\
static int quip_type_name##_assign_scalar_obj(Data_Obj *dp, Scalar_Value *svp)		\
{									\
	*((type *)OBJ_DATA_PTR(dp)) = svp->member ;					\
	return 0;							\
}

#define DECLARE_ASSIGN_CPX_SCALAR_FUNC(quip_type_name,type,member)		\
									\
static int quip_type_name##_assign_scalar_obj(Data_Obj *dp, Scalar_Value *svp)		\
{									\
	*( (type *)(OBJ_DATA_PTR(dp))  ) = svp->member[0];				\
	*(((type *)(OBJ_DATA_PTR(dp)))+1) = svp->member[1];				\
	return 0;							\
}

#define DECLARE_ASSIGN_QUAT_SCALAR_FUNC(quip_type_name,type,member)		\
									\
static int quip_type_name##_assign_scalar_obj(Data_Obj *dp, Scalar_Value *svp)		\
{									\
	*( (type *)(OBJ_DATA_PTR(dp))  ) = svp->member[0];				\
	*(((type *)(OBJ_DATA_PTR(dp)))+1) = svp->member[1];				\
	*(((type *)(OBJ_DATA_PTR(dp)))+2) = svp->member[2];				\
	*(((type *)(OBJ_DATA_PTR(dp)))+3) = svp->member[3];				\
	return 0;							\
}

#define DECLARE_ASSIGN_COLOR_SCALAR_FUNC(quip_type_name,type,member)		\
									\
static int quip_type_name##_assign_scalar_obj(Data_Obj *dp, Scalar_Value *svp)		\
{									\
	*( (type *)(OBJ_DATA_PTR(dp))   ) = svp->member[0];				\
	*(((type *)(OBJ_DATA_PTR(dp)))+1) = svp->member[1];				\
	*(((type *)(OBJ_DATA_PTR(dp)))+2) = svp->member[2];				\
	return 0;							\
}

#define DECLARE_BAD_ASSIGN_SCALAR_FUNC(quip_type_name)				\
									\
static int quip_type_name##_assign_scalar_obj(Data_Obj *dp, Scalar_Value *svp)		\
{									\
	return -1;							\
}

//////////////////////////////////////////////

#define DECLARE_EXTRACT_REAL_SCALAR_FUNC(quip_type_name,type,member)		\
									\
static void _##quip_type_name##_extract_scalar(QSP_ARG_DECL  Scalar_Value *svp, Data_Obj *dp)		\
{									\
	svp->member = *((type     *)OBJ_DATA_PTR(dp));				\
}

#define DECLARE_EXTRACT_CPX_SCALAR_FUNC(quip_type_name,type,member)		\
									\
static void _##quip_type_name##_extract_scalar(QSP_ARG_DECL  Scalar_Value *svp, Data_Obj *dp)		\
{									\
	svp->member[0] = *(((type *)OBJ_DATA_PTR(dp)));				\
	svp->member[1] = *(((type *)OBJ_DATA_PTR(dp))+1);				\
}

#define DECLARE_EXTRACT_QUAT_SCALAR_FUNC(quip_type_name,type,member)		\
									\
static void _##quip_type_name##_extract_scalar(QSP_ARG_DECL  Scalar_Value *svp, Data_Obj *dp)		\
{									\
	svp->member[0] = *(((type *)OBJ_DATA_PTR(dp)));				\
	svp->member[1] = *(((type *)OBJ_DATA_PTR(dp))+1);				\
	svp->member[2] = *(((type *)OBJ_DATA_PTR(dp))+2);				\
	svp->member[3] = *(((type *)OBJ_DATA_PTR(dp))+3);				\
}

#define DECLARE_EXTRACT_COLOR_SCALAR_FUNC(quip_type_name,type,member)		\
									\
static void _##quip_type_name##_extract_scalar(QSP_ARG_DECL  Scalar_Value *svp, Data_Obj *dp)	\
{									\
	svp->member[0] = *(((type *)OBJ_DATA_PTR(dp)));			\
	svp->member[1] = *(((type *)OBJ_DATA_PTR(dp))+1);		\
	svp->member[2] = *(((type *)OBJ_DATA_PTR(dp))+2);		\
}

#define DECLARE_BAD_EXTRACT_SCALAR_FUNC(quip_type_name)				\
									\
static void _##quip_type_name##_extract_scalar(QSP_ARG_DECL  Scalar_Value *svp, Data_Obj *dp)	\
{									\
	sprintf(ERROR_STRING,"Nonsensical call to %s_extract_scalar!?", #quip_type_name);\
	warn(ERROR_STRING);						\
}

////////////////////////////

#define DECLARE_CAST_TO_DOUBLE_FUNC(quip_type_name,member)			\
									\
static double _cast_##quip_type_name##_to_double(QSP_ARG_DECL  Scalar_Value *svp)	\
{									\
	return (double) svp->member;					\
}

#define DECLARE_BAD_CAST_TO_DOUBLE_FUNC(quip_type_name)				\
									\
static double _cast_##quip_type_name##_to_double(QSP_ARG_DECL  Scalar_Value *svp)	\
{									\
	sprintf(ERROR_STRING,						\
		"Can't cast %s to double!?",#quip_type_name);			\
	warn(ERROR_STRING);						\
	return 0.0;							\
}

////////////////////////////////

#define DECLARE_CAST_FROM_DOUBLE_FUNC(quip_type_name,type,member)			\
									\
static void _cast_##quip_type_name##_from_double(QSP_ARG_DECL  Scalar_Value *svp, double val)	\
{									\
	svp->member = (type) val;					\
}

#define DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(quip_type_name)				\
									\
static void _cast_##quip_type_name##_from_double(QSP_ARG_DECL  Scalar_Value *svp, double val)	\
{									\
	sprintf(ERROR_STRING,						\
		"Can't cast %s from double!?",#quip_type_name);			\
	warn(ERROR_STRING);						\
}

////////////////////////////

#define DECLARE_COPY_VALUE_FUNC(quip_type_name,member)				\
									\
static void copy_##quip_type_name##_value						\
			(Scalar_Value *dst_svp, Scalar_Value *src_svp)	\
{									\
	dst_svp->member = src_svp->member;				\
}

DECLARE_COPY_VALUE_FUNC(string,u_b)
DECLARE_COPY_VALUE_FUNC(bit,u_bit)

static void copy_void_value(Scalar_Value *dst_svp, Scalar_Value *src_svp) {}


#define DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(quip_type_name,type,member)	\
									\
static void _cast_indexed_##quip_type_name##_from_double				\
			(QSP_ARG_DECL  Scalar_Value *svp, int idx, double val)	\
{									\
	svp->member[idx] = (type) val;					\
}


#define DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(quip_type_name)		\
									\
static void _cast_indexed_##quip_type_name##_from_double				\
			(QSP_ARG_DECL  Scalar_Value *svp, int idx, double val)	\
{									\
	sprintf(ERROR_STRING,						\
		"cast_indexed_%s_from_double:  Can't cast to %s with an index (%d)!?",#quip_type_name,#quip_type_name,idx);		\
	warn(ERROR_STRING);						\
}

/////////////////////////////

#define DECLARE_REAL_SCALAR_FUNCS(quip_type_name,type,member)			\
									\
DECLARE_ALMOST_REAL_SCALAR_FUNCS(quip_type_name,type,member)			\
DECLARE_INDEXED_DATA_FUNC(quip_type_name,type)

#define DECLARE_BITMAP_REAL_SCALAR_FUNCS(quip_type_name,type,member)		\
									\
DECLARE_ALMOST_REAL_SCALAR_FUNCS(quip_type_name,type,member)			\
DECLARE_POSSIBLY_BITMAP_INDEXED_DATA_FUNC(quip_type_name,type)

#define DECLARE_ALMOST_REAL_SCALAR_FUNCS(quip_type_name,type,member)		\
									\
DECLARE_IS_NUMERIC_FUNC(quip_type_name)						\
DECLARE_COPY_VALUE_FUNC(quip_type_name,member)					\
DECLARE_CAST_FROM_DOUBLE_FUNC(quip_type_name,type,member)				\
DECLARE_CAST_TO_DOUBLE_FUNC(quip_type_name,member)				\
DECLARE_BAD_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(quip_type_name)			\
DECLARE_ASSIGN_REAL_SCALAR_FUNC(quip_type_name,type,member)			\
DECLARE_EXTRACT_REAL_SCALAR_FUNC(quip_type_name,type,member)


#define DECLARE_CPX_SCALAR_FUNCS(quip_type_name,type,indexable_member,copyable_member) \
									\
DECLARE_COPY_VALUE_FUNC(quip_type_name,copyable_member)				\
DECLARE_BAD_SET_VALUE_FROM_INPUT_FUNC(quip_type_name)				\
DECLARE_BAD_INDEXED_DATA_FUNC(quip_type_name)					\
DECLARE_IS_NUMERIC_FUNC(quip_type_name)						\
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(quip_type_name)					\
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(quip_type_name)					\
DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(quip_type_name,type,indexable_member)	\
DECLARE_ASSIGN_CPX_SCALAR_FUNC(quip_type_name,type,indexable_member)		\
DECLARE_EXTRACT_CPX_SCALAR_FUNC(quip_type_name,type,indexable_member)


#define DECLARE_QUAT_SCALAR_FUNCS(quip_type_name,type,indexable_member,copyable_member) \
									\
DECLARE_COPY_VALUE_FUNC(quip_type_name,copyable_member)				\
DECLARE_BAD_SET_VALUE_FROM_INPUT_FUNC(quip_type_name)				\
DECLARE_BAD_INDEXED_DATA_FUNC(quip_type_name)					\
DECLARE_IS_NUMERIC_FUNC(quip_type_name)						\
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(quip_type_name)					\
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(quip_type_name)					\
DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(quip_type_name,type,indexable_member)	\
DECLARE_ASSIGN_QUAT_SCALAR_FUNC(quip_type_name,type,indexable_member)		\
DECLARE_EXTRACT_QUAT_SCALAR_FUNC(quip_type_name,type,indexable_member)


#define DECLARE_COLOR_SCALAR_FUNCS(quip_type_name,type,indexable_member,copyable_member) \
									\
DECLARE_COPY_VALUE_FUNC(quip_type_name,copyable_member)				\
DECLARE_BAD_SET_VALUE_FROM_INPUT_FUNC(quip_type_name)				\
DECLARE_BAD_INDEXED_DATA_FUNC(quip_type_name)					\
DECLARE_IS_NUMERIC_FUNC(quip_type_name)						\
DECLARE_BAD_CAST_FROM_DOUBLE_FUNC(quip_type_name)					\
DECLARE_BAD_CAST_TO_DOUBLE_FUNC(quip_type_name)					\
DECLARE_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(quip_type_name,type,indexable_member)	\
DECLARE_ASSIGN_COLOR_SCALAR_FUNC(quip_type_name,type,indexable_member)		\
DECLARE_EXTRACT_COLOR_SCALAR_FUNC(quip_type_name,type,indexable_member)

////////////////////////////////

static int bit_assign_scalar_obj(Data_Obj *dp,Scalar_Value *svp)
{
	if( svp->bitmap_scalar )
		*( (BITMAP_DATA_TYPE *)OBJ_DATA_PTR(dp) ) |= 1 << OBJ_BIT0(dp) ;
	else
		*( (BITMAP_DATA_TYPE *)OBJ_DATA_PTR(dp) ) &= ~( 1 << OBJ_BIT0(dp) );
	return 0;
}

static void _bit_extract_scalar(QSP_ARG_DECL  Scalar_Value *svp, Data_Obj *dp)
{
	svp->bitmap_scalar = *( (BITMAP_DATA_TYPE *)OBJ_DATA_PTR(dp) ) & (1 << OBJ_BIT0(dp)) ;
}


static void _cast_bit_from_double(QSP_ARG_DECL  Scalar_Value *svp,double val)
{
	if( val == 0.0 )
		svp->bitmap_scalar = 0;
	else
		svp->bitmap_scalar = 1;
}

static double _cast_bit_to_double(QSP_ARG_DECL  Scalar_Value *svp)
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
DECLARE_REAL_SCALAR_FUNCS(int,int32_t,u_l)
DECLARE_REAL_SCALAR_FUNCS(long,int64_t,u_ll)
DECLARE_REAL_SCALAR_FUNCS(u_byte,u_char,u_ub)
DECLARE_REAL_SCALAR_FUNCS(u_short,u_short,u_us)
DECLARE_REAL_SCALAR_FUNCS(float,float,u_f)
DECLARE_REAL_SCALAR_FUNCS(double,double,u_d)

#ifdef BITMAP_WORD_IS_64_BITS
DECLARE_REAL_SCALAR_FUNCS(u_int,uint32_t,u_ul)
DECLARE_BITMAP_REAL_SCALAR_FUNCS(u_long,uint64_t,u_ull)
#else // ! BITMAP_WORD_IS_64_BITS
DECLARE_BITMAP_REAL_SCALAR_FUNCS(u_int,uint32_t,u_ul)
DECLARE_REAL_SCALAR_FUNCS(u_long,uint64_t,u_ull)
#endif // ! BITMAP_WORD_IS_64_BITS

DECLARE_ALMOST_REAL_SCALAR_FUNCS(char,char,u_b)
DECLARE_BAD_INDEXED_DATA_FUNC(char)
DECLARE_BAD_SET_VALUE_FROM_INPUT_FUNC(char)

DECLARE_CPX_SCALAR_FUNCS(complex,float,u_fc,u_spc)
DECLARE_CPX_SCALAR_FUNCS(dblcpx,double,u_dc,u_dpc)

DECLARE_QUAT_SCALAR_FUNCS(quaternion,float,u_fq,u_spq)
DECLARE_QUAT_SCALAR_FUNCS(dblquat,double,u_dq,u_dpq)

DECLARE_COLOR_SCALAR_FUNCS(color,float,u_color_comp,u_color)

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

void _init_precisions(SINGLE_QSP_ARG_DECL)
{
	Precision *prec_p;

	prec_itp = new_item_type("Precision", LIST_CONTAINER);	// used to be hashed, but not many of these?
									// should sort based on access?

	INIT_PREC(byte,char,PREC_BY)
	INIT_PREC(u_byte,u_char,PREC_UBY)
	INIT_PREC(short,short,PREC_IN)
	INIT_PREC(u_short,u_short,PREC_UIN)
	INIT_PREC(int,int32_t,PREC_DI)
	INIT_PREC(u_int,uint32_t,PREC_UDI)
	INIT_PREC(long,int64_t,PREC_LI)
	INIT_PREC(u_long,uint64_t,PREC_ULI)
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
		init_precisions();
	}

	return item_list(prec_itp);
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

