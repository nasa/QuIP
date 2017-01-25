#include "veclib/gen_gpu_calls.h"

#define KERNEL_ARG_QUALIFIER
#define KERNEL_FUNC_QUALIFIER	__global__

#define KERNEL_FUNC_PRELUDE

// These can't be defined generically, because for openCL we define them
// to declare a string variable containing the kernel source code.

#define _GENERIC_FAST_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)	\
	__GENERIC_FAST_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)

#define _GENERIC_EQSP_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)	\
	__GENERIC_EQSP_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)

#define _GENERIC_SLOW_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)	\
	__GENERIC_SLOW_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)

#define _GENERIC_FLEN_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)	\
	__GENERIC_FLEN_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)

#define _GENERIC_ELEN_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)	\
	__GENERIC_ELEN_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)

#define _GENERIC_SLEN_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)	\
	__GENERIC_SLEN_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)

/************** conversions **************/

#define _GENERIC_FAST_CONV_FUNC(name,src_type,dst_type)	\
	__GENERIC_FAST_CONV_FUNC(name,src_type,dst_type)

#define _GENERIC_EQSP_CONV_FUNC(name,src_type,dst_type)	\
	__GENERIC_EQSP_CONV_FUNC(name,src_type,dst_type)

#define _GENERIC_SLOW_CONV_FUNC(name,src_type,dst_type)	\
	__GENERIC_SLOW_CONV_FUNC(name,src_type,dst_type)

#define _GENERIC_FLEN_CONV_FUNC(name,src_type,dst_type)	\
	__GENERIC_FLEN_CONV_FUNC(name,src_type,dst_type)

#define _GENERIC_ELEN_CONV_FUNC(name,src_type,dst_type)	\
	__GENERIC_ELEN_CONV_FUNC(name,src_type,dst_type)

#define _GENERIC_SLEN_CONV_FUNC(name,src_type,dst_type)	\
	__GENERIC_SLEN_CONV_FUNC(name,src_type,dst_type)



#define _GENERIC_FAST_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)	\
	__GENERIC_FAST_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)

#define _GENERIC_EQSP_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)	\
	__GENERIC_EQSP_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)

#define _GENERIC_FLEN_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)	\
	__GENERIC_FLEN_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)

#define _GENERIC_ELEN_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)	\
	__GENERIC_ELEN_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)

#define _GENERIC_SLOW_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)	\
	__GENERIC_SLOW_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)

#define _GENERIC_SLEN_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)	\
	__GENERIC_SLEN_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)

#define __VEC_FUNC_FAST_2V_PROJ_SETUP( func_name, gpu_expr )		\
	___VEC_FUNC_FAST_2V_PROJ_SETUP( func_name, gpu_expr )

#define __VEC_FUNC_FAST_2V_PROJ_HELPER( func_name, gpu_expr )		\
	___VEC_FUNC_FAST_2V_PROJ_HELPER( func_name, gpu_expr )

#define __VEC_FUNC_CPX_FAST_2V_PROJ_SETUP( func_name, re_expr, im_expr )		\
	___VEC_FUNC_CPX_FAST_2V_PROJ_SETUP( func_name, re_expr, im_expr )

#define __VEC_FUNC_CPX_FAST_2V_PROJ_HELPER( func_name, re_expr, im_expr )		\
	___VEC_FUNC_CPX_FAST_2V_PROJ_HELPER( func_name, re_expr, im_expr )

#define __VEC_FUNC_QUAT_FAST_2V_PROJ_SETUP( func_name, re_expr, im_expr1, im_expr2, im_expr3 )	\
	___VEC_FUNC_QUAT_FAST_2V_PROJ_SETUP( func_name, re_expr, im_expr1, im_expr2, im_expr3 )		

#define __VEC_FUNC_QUAT_FAST_2V_PROJ_HELPER( func_name, re_expr, im_expr1, im_expr2, im_expr3 )	\
	___VEC_FUNC_QUAT_FAST_2V_PROJ_HELPER( func_name, re_expr, im_expr1, im_expr2, im_expr3 )		



#define __VEC_FUNC_FAST_2V_PROJ_IDX( func_name, statement1, statement2 )	\
	___VEC_FUNC_FAST_2V_PROJ_IDX( func_name, statement1, statement2 )

#define __VEC_FUNC_FAST_3V_PROJ( func_name )		\
	___VEC_FUNC_FAST_3V_PROJ( func_name )		\

#define __VEC_FUNC_CPX_FAST_3V_PROJ( func_name )		\
	___VEC_FUNC_CPX_FAST_3V_PROJ( func_name )		\

#define __VEC_FUNC_FAST_MM_NOCC( func_name, test1, test2 )		\
								\
	___VEC_FUNC_FAST_MM_NOCC_SETUP( func_name, test1, test2 )	\
	___VEC_FUNC_FAST_MM_NOCC_HELPER( func_name, test1, test2 )

