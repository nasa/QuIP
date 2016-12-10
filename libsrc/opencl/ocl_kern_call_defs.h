//#include "kern_both_defs.h"
//#include "gpu_calls.h"
#ifndef _OCL_KERN_CALL_DEFS_H_
#define _OCL_KERN_CALL_DEFS_H_

// These definitions are expanded during C compilation, but are not
// compiled - they are stored to strings and compiled on the fly.

// We have a problem because some gpu's can't handle double (e.g. Iris Pro)
// The typedefs here cause problems, even when the kernel doesn't use them.
// So we take them out and include only when needed...

#define KERNEL_FUNC_PRELUDE						\
									\
typedef unsigned long bitmap_word;					\
typedef unsigned char u_char;						\
typedef unsigned long uint64_t;						\
typedef long int64_t;							\
typedef unsigned int uint32_t;						\
typedef int int32_t;							\
/*typedef struct { int x; int y; int z; } dim3 ;*/			\
typedef struct { int d5_dim[5]; } dim5 ;				\
typedef struct { float re; float im; } SP_Complex;			\
typedef struct { float re; float _i; float _j; float _k; } SP_Quaternion;\
EXTRA_PRELUDE(type_code)

#define EXTRA_PRELUDE(t)	_EXTRA_PRELUDE(t)
#define _EXTRA_PRELUDE(t)	__EXTRA_PRELUDE(t)
#define __EXTRA_PRELUDE(t)	EXTRA_PRELUDE_##t

#define EXTRA_PRELUDE_sp
#define EXTRA_PRELUDE_by
#define EXTRA_PRELUDE_in
#define EXTRA_PRELUDE_di
#define EXTRA_PRELUDE_li
#define EXTRA_PRELUDE_uby
#define EXTRA_PRELUDE_uin
#define EXTRA_PRELUDE_udi
#define EXTRA_PRELUDE_uli
#define EXTRA_PRELUDE_ubyin
#define EXTRA_PRELUDE_uindi
#define EXTRA_PRELUDE_inby
#define EXTRA_PRELUDE_bit
#define EXTRA_PRELUDE_spdp	EXTRA_PRELUDE_dp

#define EXTRA_PRELUDE_dp						\
									\
typedef struct { double re; double im; } DP_Complex;			\
typedef struct { double re; double _i; double _j; double _k; } DP_Quaternion;


#endif // _OCL_KERN_CALL_DEFS_H_
