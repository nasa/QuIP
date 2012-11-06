
char VersionId_cuda_myproject_kernel[] = QUIP_VERSION_STRING;

#ifndef _MYPROJECT_KERN_
#define _MYPROJECT_KERN_

#include "my_vector_functions.h"				// declare all the prototypes for the host

#include "gpu_calls.h"

#define GPU_FUNCTION	// to disable special cases in shared defn files

// SP stuff

#define absfunc fabs
#define std_type float
#define std_cpx SP_Complex
#define dest_type float
#define dest_cpx SP_Complex
#define type_code sp

#include "call_defs.h"
#include "veclib/all_vec.c"
#include "veclib/math_funcs.c"
#include "veclib/cpx_vec.c"
#include "veclib/signed_vec.c"
#include "gpu_all.cu"
#include "gpu_signed.cu"
#include "undefs2.h"

// DP stuff

#define absfunc fabs
#define std_type double
#define std_cpx DP_Complex
#define dest_type double
#define dest_cpx DP_Complex
#define type_code dp

#include "call_defs.h"
#include "veclib/all_vec.c"
#include "veclib/math_funcs.c"
#include "veclib/cpx_vec.c"
#include "veclib/signed_vec.c"
#include "gpu_all.cu"
#include "gpu_signed.cu"
#include "undefs2.h"

// BY stuff

#define std_type char
#define dest_type char
#define ALL_ONES 0xff
#define absfunc abs
#define type_code by
#include "call_defs.h"
#include "veclib/all_vec.c"
#include "gpu_all.cu"
//#include "int_prec_funcs.cu"
#include "veclib/intvec.c"
#include "gpu_int.cu"
#include "veclib/signed_vec.c"
#include "gpu_signed.cu"
#include "undefs2.h"


// IN stuff

#define std_type short
#define dest_type short
#define ALL_ONES 0xffff
#define absfunc abs
#define type_code in
#include "call_defs.h"
#include "veclib/all_vec.c"
#include "gpu_all.cu"
#include "veclib/intvec.c"
#include "gpu_int.cu"
#include "veclib/signed_vec.c"
#include "gpu_signed.cu"
#include "undefs2.h"


// DI stuff

#define std_type int32_t
#define dest_type int32_t
#define ALL_ONES 0xffffffff
#define absfunc abs
#define type_code di
#include "call_defs.h"
#include "veclib/all_vec.c"
#include "gpu_all.cu"
#include "veclib/intvec.c"
#include "gpu_int.cu"
#include "veclib/signed_vec.c"
#include "gpu_signed.cu"
#include "undefs2.h"


// LI stuff

#define std_type int64_t
#define dest_type int64_t
#define ALL_ONES 0xffffffffffffffff
#define absfunc abs
#define type_code li
#include "call_defs.h"
#include "veclib/all_vec.c"
#include "gpu_all.cu"
#include "veclib/intvec.c"
#include "gpu_int.cu"
#include "veclib/signed_vec.c"
#include "gpu_signed.cu"
#include "undefs2.h"


// UBY stuff

#define std_type u_char
#define std_signed char
#define dest_type u_char
#define ALL_ONES 0xff
#define absfunc abs
#define type_code uby
#include "call_defs.h"
#include "veclib/all_vec.c"
#include "gpu_all.cu"
#include "veclib/intvec.c"
#include "gpu_int.cu"
#include "veclib/unsigned_vec.c"
#include "undefs2.h"


// UIN stuff

#define std_type u_short
#define std_signed short
#define dest_type u_short
#define ALL_ONES 0xffff
#define absfunc abs
#define type_code uin
#include "call_defs.h"
#include "veclib/all_vec.c"
#include "gpu_all.cu"
#include "veclib/intvec.c"
#include "gpu_int.cu"
#include "veclib/unsigned_vec.c"
#include "undefs2.h"


// UDI stuff

#define std_type uint32_t
#define std_signed int32_t
#define dest_type uint32_t
#define ALL_ONES 0xffffffff
#define absfunc abs
#define type_code udi
#include "call_defs.h"
#include "veclib/all_vec.c"
#include "gpu_all.cu"
#include "veclib/intvec.c"
#include "gpu_int.cu"
#include "veclib/unsigned_vec.c"
#include "undefs2.h"


// ULI stuff

#define std_type uint64_t
#define std_signed int64_t
#define dest_type uint64_t
#define ALL_ONES 0xffffffffffffffff
#define absfunc abs
#define type_code uli
#include "call_defs.h"
#include "veclib/all_vec.c"
#include "gpu_all.cu"
#include "veclib/intvec.c"
#include "gpu_int.cu"
#include "veclib/unsigned_vec.c"
#include "undefs2.h"

// bit stuff

#define std_type bitmap_word
#define dst_type bitmap_word
#define ALL_ONES	1
#define type_code bit
#include "call_defs.h"
#include "gpu_bit.cu"
#include "undefs2.h"


__constant__ unsigned char cmem[CONST_MEM_SIZE];

#endif // #ifndef _MYPROJECT_KERN_
