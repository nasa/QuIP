dnl used for host calls and kernels, should be renamed...

dnl SP stuff

include(`../../include/veclib/sp_defs.m4')
include(`../../include/veclib/all_float_kernels.m4')

// DP stuff

include(`../../include/veclib/dp_defs.m4')
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

// bit stuff
//
// This may not work on a GPU, because different threads will need to read
// and write the same word!?

include(`../../include/veclib/bit_defs.m4')
include(`../../include/veclib/all_bit_kernels.m4')


// Why is this only for not building kernels?

dnl ifdef(`BUILDING_KERNELS',`',` dnl ifndef BUILDING_KERNELS

// Now mixed precision functions...
// We currently implement 4:
// inby, spdp, uindi, and ubyin
// The second code indicates the destination precision,
// Most common use is vsum

define(`MIXED_PRECISION',`')

include(`../../include/veclib/inby_defs.m4')
include(`../../include/veclib/all_mixed_int_kernels.m4')

include(`../../include/veclib/uindi_defs.m4')
include(`../../include/veclib/all_mixed_uint_kernels.m4')

include(`../../include/veclib/ubyin_defs.m4')
include(`../../include/veclib/all_mixed_uint_kernels.m4')

include(`../../include/veclib/spdp_defs.m4')
include(`../../include/veclib/all_mixed_float_kernels.m4')

undefine(`MIXED_PRECISION')

dnl ') dnl endif // ! BUILDING_KERNELS




