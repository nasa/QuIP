//#include "kern_both_defs.h"
//#include "gpu_calls.h"
#ifndef _OCL_KERN_CALL_DEFS_H_
#define _OCL_KERN_CALL_DEFS_H_

// These definitions are expanded during C compilation, but are not
// compiled - they are stored to strings and compiled on the fly.

#define KERNEL_FUNC_PRELUDE						\
									\
typedef unsigned long bitmap_word;					\
typedef unsigned char u_char;						\
typedef unsigned long uint64_t;						\
typedef long int64_t;							\
typedef unsigned int uint32_t;						\
typedef int int32_t;							\
typedef struct { int x; int y; int z; } dim3 ;				\
typedef struct { float re; float im; } SP_Complex;			\
typedef struct { double re; double im; } DP_Complex;			\
typedef struct { float re; float _i; float _j; float _k; } SP_Quaternion;\
typedef struct { double re; double _i; double _j; double _k; } DP_Quaternion; 


#endif // _OCL_KERN_CALL_DEFS_H_
