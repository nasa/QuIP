
#include "quip_config.h"

#ifdef HAVE_CUDA

char VersionId_cuda_conversions[] = QUIP_VERSION_STRING;

#include <stdio.h>

#include <cutil.h>
#include <cutil_inline.h>

#include "my_cuda.h"
#include "cuda_supp.h"		// describe_cuda_error
#include "vecgen.h"

#include "my_vector_functions.h"	// declare all host prototypes

#include "gpu_call_utils.h"
#include "host_call_utils.h"
#include "veclib/fast_test.h"


#define KERN_CONVERSION( name, src_prec, dst_prec )			\
		_KERN_CONVERSION( name , src_prec, dst_prec )


#define _KERN_CONVERSION( func_name, src_prec, dst_prec )		\
									\
	__global__ void g_##func_name(dst_prec* a, src_prec* b)		\
	{								\
		INIT_INDICES_2						\
		dst = src1 ;						\
	}								\
									\
	__global__ void g_##func_name##_eqsp(	dst_prec * a, int inc1,	\
						src_prec * b, int inc2 )\
	{								\
		INIT_INDICES_2						\
		SCALE_INDICES_2						\
		dst = src1 ;						\
	}								\
									\
	__global__ void g_##func_name##_len				\
				(dst_prec * a, src_prec * b, int len)	\
	{								\
		INIT_INDICES_2						\
		if( index1.x < len ){					\
			dst = src1 ;					\
		}							\
	}								\
									\
	__global__ void g_##func_name##_eqsp_len			\
	(dst_prec * a, int inc1, src_prec * b, int inc2, int len)	\
	{								\
		INIT_INDICES_2						\
		if( index1.x < len ){					\
			SCALE_INDICES_2					\
			dst = src1 ;					\
		}							\
	}								\
									\
	__global__ void g_##func_name##_incs				\
	(dst_prec * a, dim3 inc1, src_prec * b, dim3 inc2, dim3 len)	\
	{								\
		INIT_INDICES_XY_2					\
		if( index1.x < len.x && index1.y < len.y ){		\
			SCALE_INDICES_XY_2				\
			dst = src1 ;					\
		}							\
	}

#define ALL_SIGNED_GPU_CONVERSIONS( src_code, src_type)		\
								\
KERN_CONVERSION( src_code##2by, src_type, char)			\
KERN_CONVERSION( src_code##2in, src_type, short)		\
KERN_CONVERSION( src_code##2di, src_type, int32_t)		\
KERN_CONVERSION( src_code##2li, src_type, int64_t)

#define ALL_UNSIGNED_GPU_CONVERSIONS( src_code, src_type)	\
								\
KERN_CONVERSION( src_code##2uby, src_type, u_char)		\
KERN_CONVERSION( src_code##2uin, src_type, u_short)		\
KERN_CONVERSION( src_code##2udi, src_type, uint32_t)		\
KERN_CONVERSION( src_code##2uli, src_type, uint64_t)

#define ALL_FLOAT_GPU_CONVERSIONS( src_code, src_type)		\
								\
KERN_CONVERSION( src_code##2sp, src_type, float)		\
KERN_CONVERSION( src_code##2dp, src_type, double)

ALL_SIGNED_GPU_CONVERSIONS( sp, float)
ALL_UNSIGNED_GPU_CONVERSIONS( sp, float)
KERN_CONVERSION(sp2dp, float, double)

ALL_SIGNED_GPU_CONVERSIONS( dp, double)
ALL_UNSIGNED_GPU_CONVERSIONS( dp, double)
KERN_CONVERSION( dp2sp, double, float)

KERN_CONVERSION( by2in, char, short)
KERN_CONVERSION( by2di, char, int32_t)
KERN_CONVERSION( by2li, char, int64_t)
ALL_UNSIGNED_GPU_CONVERSIONS( by, char)
ALL_FLOAT_GPU_CONVERSIONS( by, char)

KERN_CONVERSION( in2by, short, char)
KERN_CONVERSION( in2di, short, int32_t)
KERN_CONVERSION( in2li, short, int64_t)
ALL_UNSIGNED_GPU_CONVERSIONS( in, short)
ALL_FLOAT_GPU_CONVERSIONS( in, short)

KERN_CONVERSION( di2by, int32_t, char)
KERN_CONVERSION( di2in, int32_t, short)
KERN_CONVERSION( di2li, int32_t, int64_t)
ALL_UNSIGNED_GPU_CONVERSIONS( di, int32_t)
ALL_FLOAT_GPU_CONVERSIONS( di, int32_t)

KERN_CONVERSION( li2by, int64_t, char)
KERN_CONVERSION( li2in, int64_t, short)
KERN_CONVERSION( li2di, int64_t, int32_t)
ALL_UNSIGNED_GPU_CONVERSIONS( li, int64_t)
ALL_FLOAT_GPU_CONVERSIONS( li, int64_t)


KERN_CONVERSION( uby2uin, u_char, u_short)
KERN_CONVERSION( uby2udi, u_char, uint32_t)
KERN_CONVERSION( uby2uli, u_char, uint64_t)
ALL_SIGNED_GPU_CONVERSIONS( uby, u_char)
ALL_FLOAT_GPU_CONVERSIONS( uby, u_char)

KERN_CONVERSION( uin2uby, u_short, u_char)
KERN_CONVERSION( uin2udi, u_short, uint32_t)
KERN_CONVERSION( uin2uli, u_short, uint64_t)
ALL_SIGNED_GPU_CONVERSIONS( uin, u_short)
ALL_FLOAT_GPU_CONVERSIONS( uin, u_short)

KERN_CONVERSION( udi2uby, uint32_t, u_char)
KERN_CONVERSION( udi2uin, uint32_t, u_short)
KERN_CONVERSION( udi2uli, uint32_t, uint64_t)
ALL_SIGNED_GPU_CONVERSIONS( udi, uint32_t)
ALL_FLOAT_GPU_CONVERSIONS( udi, uint32_t)

KERN_CONVERSION( uli2uby, uint64_t, u_char)
KERN_CONVERSION( uli2uin, uint64_t, u_short)
KERN_CONVERSION( uli2udi, uint64_t, uint32_t)
ALL_SIGNED_GPU_CONVERSIONS( uli, uint64_t)
ALL_FLOAT_GPU_CONVERSIONS( uli, uint64_t)

#define H_CALL_CONV(host_func_name, gpu_func_name, dst_type, src_type )	\
									\
	void host_func_name(Vec_Obj_Args * oap ) {			\
		BLOCK_VARS_DECLS					\
		DEST_DECL( dst_type )					\
		src_type *arg2;						\
		dim3 inc2;						\
									\
		GET_DEST(dst_type)					\
		arg2 = (src_type *)oap->oa_dp[0]->dt_data;		\
									\
	CLEAR_CUDA_ERROR2(host_func_name,gpu_func_name)			\
		if( FAST_TEST_2 ){					\
			GET_THREADS_PER_BLOCK				\
			if (extra.x != 0) {				\
				n_blocks.x++;				\
				gpu_func_name##_len<<< NN_GPU >>>	\
					(arg1, arg2, len.x);		\
			} else {					\
				gpu_func_name<<< NN_GPU >>>		\
					(arg1, arg2);			\
			}						\
		} else if( EQSP_TEST_2 ){				\
			GET_THREADS_PER_BLOCK				\
			SETUP_SIMPLE_INCS2				\
			if (extra.x != 0) {				\
				n_blocks.x++;				\
				gpu_func_name##_eqsp_len<<< NN_GPU >>>	\
					(arg1, inc1.x, arg2, inc2.x, len.x);\
			} else {					\
				gpu_func_name##_eqsp<<< NN_GPU >>>	\
					(arg1, inc1.x, arg2, inc2.x );	\
			}						\
		} else {						\
			SETUP_2_INCS( host_func_name )			\
			gpu_func_name##_incs<<< NN_GPU >>> 		\
				(arg1, inc1, arg2, inc2, len); 		\
		}							\
	CHECK_CUDA_ERROR(host_func_name,gpu_func_name)			\
	}					



#define ALL_SIGNED_HOST_CONVERSIONS( src_code , src_type )		\
									\
H_CALL_CONV( h_##src_code##2by, g_##src_code##2by, char, src_type )	\
H_CALL_CONV( h_##src_code##2in, g_##src_code##2in, short, src_type )	\
H_CALL_CONV( h_##src_code##2di, g_##src_code##2di, int32_t, src_type )	\
H_CALL_CONV( h_##src_code##2li, g_##src_code##2li, int64_t, src_type )

#define ALL_UNSIGNED_HOST_CONVERSIONS( src_code , src_type )		\
									\
H_CALL_CONV( h_##src_code##2uby, g_##src_code##2uby, u_char, src_type )	\
H_CALL_CONV( h_##src_code##2uin, g_##src_code##2uin, u_short, src_type )\
H_CALL_CONV( h_##src_code##2udi, g_##src_code##2udi, uint32_t, src_type )\
H_CALL_CONV( h_##src_code##2uli, g_##src_code##2uli, uint64_t, src_type )

#define ALL_FLOAT_HOST_CONVERSIONS( src_code , src_type )		\
									\
H_CALL_CONV( h_##src_code##2sp, g_##src_code##2sp, float, src_type )	\
H_CALL_CONV( h_##src_code##2dp, g_##src_code##2dp, double, src_type )

// From sp
H_CALL_CONV( h_sp2dp, g_sp2dp, double, float )
ALL_SIGNED_HOST_CONVERSIONS( sp , float )
ALL_UNSIGNED_HOST_CONVERSIONS( sp , float )

// From dp
H_CALL_CONV( h_dp2sp, g_dp2sp, float , double)
ALL_SIGNED_HOST_CONVERSIONS( dp , double )
ALL_UNSIGNED_HOST_CONVERSIONS( dp , double )

//from by
H_CALL_CONV( h_by2in, g_by2in, short, char )
H_CALL_CONV( h_by2di, g_by2di, int32_t, char )
H_CALL_CONV( h_by2li, g_by2li, int64_t, char )
ALL_UNSIGNED_HOST_CONVERSIONS( by , char )
ALL_FLOAT_HOST_CONVERSIONS( by , char )

//from in
H_CALL_CONV( h_in2by, g_in2by, char, short )
H_CALL_CONV( h_in2di, g_in2di, int32_t, short )
H_CALL_CONV( h_in2li, g_in2li, int64_t, short )
ALL_UNSIGNED_HOST_CONVERSIONS( in , short )
ALL_FLOAT_HOST_CONVERSIONS( in , short )

//from di
H_CALL_CONV( h_di2by, g_di2by, char, int32_t )
H_CALL_CONV( h_di2in, g_di2in, short, int32_t )
H_CALL_CONV( h_di2li, g_di2li, int64_t, int32_t )
ALL_UNSIGNED_HOST_CONVERSIONS( di , int32_t )
ALL_FLOAT_HOST_CONVERSIONS( di , int32_t )

//from li
H_CALL_CONV( h_li2by, g_li2by, char, int64_t )
H_CALL_CONV( h_li2in, g_li2in, short, int64_t )
H_CALL_CONV( h_li2di, g_li2di, int32_t, int64_t )
ALL_UNSIGNED_HOST_CONVERSIONS( li , int64_t )
ALL_FLOAT_HOST_CONVERSIONS( li , int64_t )


//from uby
H_CALL_CONV( h_uby2uin, g_uby2uin, u_short, u_char )
H_CALL_CONV( h_uby2udi, g_uby2udi, uint32_t, u_char )
H_CALL_CONV( h_uby2uli, g_uby2uli, uint64_t, u_char )
ALL_SIGNED_HOST_CONVERSIONS( uby , u_char )
ALL_FLOAT_HOST_CONVERSIONS( uby , u_char )

//from uin
H_CALL_CONV( h_uin2uby, g_uin2uby, u_char, u_short )
H_CALL_CONV( h_uin2udi, g_uin2udi, uint32_t, u_short )
H_CALL_CONV( h_uin2uli, g_uin2uli, uint64_t, u_short )
ALL_SIGNED_HOST_CONVERSIONS( uin , u_short )
ALL_FLOAT_HOST_CONVERSIONS( uin , u_short )

//from udi
H_CALL_CONV( h_udi2uby, g_udi2uby, u_char, uint32_t )
H_CALL_CONV( h_udi2uin, g_udi2uin, u_short, uint32_t )
H_CALL_CONV( h_udi2uli, g_udi2uli, uint64_t, uint32_t )
ALL_SIGNED_HOST_CONVERSIONS( udi , uint32_t )
ALL_FLOAT_HOST_CONVERSIONS( udi , uint32_t )

//from uli
H_CALL_CONV( h_uli2uby, g_uli2uby, u_char, uint64_t )
H_CALL_CONV( h_uli2uin, g_uli2uin, u_short, uint64_t )
H_CALL_CONV( h_uli2udi, g_uli2udi, uint32_t, uint64_t )
ALL_SIGNED_HOST_CONVERSIONS( uli , uint64_t )
ALL_FLOAT_HOST_CONVERSIONS( uli , uint64_t )


#endif /* HAVE_CUDA */

