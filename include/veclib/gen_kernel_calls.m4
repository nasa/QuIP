dnl used for host calls and kernels, should be renamed...

dnl SP stuff

include(`../../include/veclib/sp_defs.m4')
// including all_float_kernels for float
include(`../../include/veclib/all_float_kernels.m4')

// DP stuff

include(`../../include/veclib/dp_defs.m4')
// including all_float_kernels for double
include(`../../include/veclib/all_float_kernels.m4')

// BY stuff

include(`../../include/veclib/by_defs.m4')
include(`../../include/veclib/all_int_kernels.m4')


// IN stuff

include(`../../include/veclib/in_defs.m4')
include(`../../include/veclib/all_int_kernels.m4')


// DI stuff

include(`../../include/veclib/di_defs.m4')
include(`../../include/veclib/all_int_kernels.m4')


// LI stuff

include(`../../include/veclib/li_defs.m4')
include(`../../include/veclib/all_int_kernels.m4')


// UBY stuff

include(`../../include/veclib/uby_defs.m4')
include(`../../include/veclib/all_uint_kernels.m4')


// UIN stuff

include(`../../include/veclib/uin_defs.m4')
include(`../../include/veclib/all_uint_kernels.m4')


// UDI stuff

include(`../../include/veclib/udi_defs.m4')
include(`../../include/veclib/all_uint_kernels.m4')


// ULI stuff

include(`../../include/veclib/uli_defs.m4')
include(`../../include/veclib/all_uint_kernels.m4')

dnl	// Now mixed precision functions...
dnl	// We currently implement 4:
dnl	// inby, spdp, uindi, and ubyin
dnl	// The second code indicates the destination precision,
dnl	// Most common use is vsum

define(`MIXED_PRECISION',`')

include(`../../include/veclib/inby_defs.m4')
include(`../../include/veclib/all_mixed_int_kernels.m4')

include(`../../include/veclib/uindi_defs.m4')
include(`../../include/veclib/all_mixed_uint_kernels.m4')

include(`../../include/veclib/ubyin_defs.m4')
include(`../../include/veclib/all_mixed_uint_kernels.m4')

// including all_mixed_float_kernels for float/double
include(`../../include/veclib/spdp_defs.m4')
include(`../../include/veclib/all_mixed_float_kernels.m4')

undefine(`MIXED_PRECISION')

// bit stuff
dnl	//
dnl	// This may not work on a GPU, because different threads will need to read
dnl	// and write the same word!?
dnl	We handle GPU word contention by having one thread per word, with a special data structure
dnl	that specifies which bits should be manipulated.

dnl	On the CPU, we also need different loops for bitmaps...

/* gen_kernel_calls.m4  inclding bit_defs.m4 */

include(`../../include/veclib/bit_defs.m4')

/* gen_kernel_calls.m4  inclding all_bit_kernels.m4 */

include(`../../include/veclib/all_bit_kernels.m4')

