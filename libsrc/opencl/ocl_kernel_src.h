#ifndef _OCL_GPU_CALLS_H_
#define _OCL_GPU_CALLS_H_
// 
// macros for creating OpenCL kernels...
//

#include "veclib/gen_gpu_calls.h"

//#define QUOTE_CHAR	\"
#define QUOTE_CHAR	'"'

#define QUOTE_IT(string)	_QUOTE_IT(string)
#define _QUOTE_IT(string)	#string

// PORT
#define KERN_SOURCE_NAME(name,stem)	_KERN_SOURCE_NAME(name,stem,type_code,pf_str)
#define _KERN_SOURCE_NAME(name,stem,tc,pf)	__KERN_SOURCE_NAME(name,stem,tc,pf)
#define __KERN_SOURCE_NAME(name,stem,tc,pf)	kernel_source_##pf##_##stem##_##tc##_##name

//#define KERN_CONV_SOURCE_NAME(name,stem)	kern_source_##stem##_##name

#define _GENERIC_FAST_CONV_FUNC(name,from_type,to_type) \
char KERN_SOURCE_NAME(name,fast)[]=QUOTE_IT( __GENERIC_FAST_CONV_FUNC(name,from_type,to_type) );

#define _GENERIC_EQSP_CONV_FUNC(name,from_type,to_type) \
char KERN_SOURCE_NAME(name,eqsp)[]=QUOTE_IT( __GENERIC_EQSP_CONV_FUNC(name,from_type,to_type) );

#define _GENERIC_SLOW_CONV_FUNC(name,from_type,to_type) \
char KERN_SOURCE_NAME(name,slow)[]=QUOTE_IT( __GENERIC_SLOW_CONV_FUNC(name,from_type,to_type) );


#define _GENERIC_FAST_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)	\
char KERN_SOURCE_NAME(name,fast)[]= QUOTE_IT( __GENERIC_FAST_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra) );

#define _GENERIC_EQSP_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)	\
char KERN_SOURCE_NAME(name,eqsp)[]= QUOTE_IT( __GENERIC_EQSP_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra) );

#define _GENERIC_SLOW_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra)	\
char KERN_SOURCE_NAME(name,slow)[]= QUOTE_IT( __GENERIC_SLOW_VEC_FUNC(name,statement,bm,typ,scalars,vectors,extra) );


// For cuda, this is __global__
#define KERNEL_FUNC_QUALIFIER __kernel


#define GENERIC_FAST_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)	\
	_GENERIC_FAST_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)

#define GENERIC_EQSP_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)	\
	_GENERIC_EQSP_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)

#define GENERIC_SLOW_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)	\
	_GENERIC_SLOW_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)

#define _GENERIC_FAST_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)	\
char KERN_SOURCE_NAME(name,fast)[] = QUOTE_IT( __GENERIC_FAST_VEC_FUNC_DBM(name,statement,typ,scalars,vectors) );

#define _GENERIC_EQSP_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)	\
char KERN_SOURCE_NAME(name,eqsp)[] = QUOTE_IT( __GENERIC_EQSP_VEC_FUNC_DBM(name,statement,typ,scalars,vectors) );

#define _GENERIC_SLOW_VEC_FUNC_DBM(name,statement,typ,scalars,vectors)	\
char KERN_SOURCE_NAME(name,slow)[] = QUOTE_IT( __GENERIC_SLOW_VEC_FUNC_DBM(name,statement,typ,scalars,vectors) );



/* FIXME still need to convert these to generic macros if possible */

#define _VEC_FUNC_MM( func_name, statement )		\
__VEC_FUNC_MM( func_name, statement );

#define __VEC_FUNC_MM( func_name, statement )		\
char KERN_SOURCE_NAME(func_name,mm)[]=QUOTE_IT(___VEC_FUNC_MM( func_name, statement ) );

#define _VEC_FUNC_MM_IND( func_name, statement1, statement2 )\
__VEC_FUNC_MM_IND( func_name, statement1, statement2 )

#define __VEC_FUNC_MM_IND( func_name, statement1, statement2 )\
char KERN_SOURCE_NAME(func_name,mm_ind)[] = QUOTE_IT( ___VEC_FUNC_MM_IND( func_name, statement1, statement2 ) );

/* For nocc_setup, we index directly into the value and count temp arrays
 * (a and b, respectively), but we have to double the index for the source
 * array c, and the index array d.  Because we are comparing adjacent pairs, 
 */

#define __VEC_FUNC_FAST_MM_NOCC( func_name, test1, test2 )			\
									\
char KERN_SOURCE_NAME(func_name##_nocc_setup,fast)[]= QUOTE_IT( ___VEC_FUNC_FAST_MM_NOCC_SETUP( func_name, test1, test2 ) );	\
char KERN_SOURCE_NAME(func_name##_nocc_helper,fast)[]= QUOTE_IT( ___VEC_FUNC_FAST_MM_NOCC_HELPER( func_name, test1, test2 ) );

// vsum, vdot, etc
// BUG this is hard-coded for vsum!?
//
// The idea is that , because all threads cannot access the destination simultaneously,
// we have to make the source smaller recursively...  But when we project
// to a vector instead of a scalar, we can to the elements of the vector in parallel...
// This is quite tricky.
//
// Example:  col=sum(m)
//
// m = | 1 2 3 4 |
//     | 5 6 7 8 |
//
// tmp = | 4  6  |
//       | 12 14 |
//
// col = | 10 |
//       | 26 |
     

// BUG - we need to make this do vmaxv and vminv as well.
// It's the same except for the sum line, which would be replaced with
//

#ifdef FOOBAR
#define psrc1	s1[index1.x]	// FOOBAR
#define psrc2	s2[index1.x]	// FOOBAR
#else // ! FOOBAR
//#define psrc1	s1[index1.d5_dim[0]]
//#define psrc2	s2[index1.d5_dim[0]]
#endif  // ! FOOBAR

// for vsum:   psrc1 + psrc2
// for vmaxv:  psrc1 > psrc2 ? psrc1 : psrc2

#define __VEC_FUNC_FAST_2V_PROJ( func_name, expr )					\
char KERN_SOURCE_NAME(func_name,fast)[]= QUOTE_IT( ___VEC_FUNC_FAST_2V_PROJ( func_name, expr ) );

#define __VEC_FUNC_CPX_FAST_2V_PROJ( func_name, expr_re, expr_im )		\
char KERN_SOURCE_NAME(func_name,fast)[]= QUOTE_IT( ___VEC_FUNC_CPX_FAST_2V_PROJ( func_name, expr_re, expr_im ) );

#define __VEC_FUNC_QUAT_FAST_2V_PROJ( func_name, expr_re, expr_im1, expr_im2, expr_im3 )		\
char KERN_SOURCE_NAME(func_name,fast)[]= QUOTE_IT( ___VEC_FUNC_QUAT_FAST_2V_PROJ( func_name, expr_re, expr_im1, expr_im2, expr_im3 ) );

#define __VEC_FUNC_FAST_2V_PROJ_IDX( func_name, gpu_s1, gpu_s2 )	\
char KERN_SOURCE_NAME(func_name,fast)[]=QUOTE_IT(___VEC_FUNC_FAST_2V_PROJ_IDX( func_name, gpu_s1, gpu_s2 ) );

#define __VEC_FUNC_FAST_3V_PROJ( func_name )		\
char KERN_SOURCE_NAME(func_name,fast)[]=QUOTE_IT(___VEC_FUNC_FAST_3V_PROJ( func_name ) );

#define __VEC_FUNC_CPX_FAST_3V_PROJ( func_name )		\
char KERN_SOURCE_NAME(func_name,fast)[]=QUOTE_IT(___VEC_FUNC_CPX_FAST_3V_PROJ( func_name ) );


#endif // ! _OCL_GPU_CALLS_H_
