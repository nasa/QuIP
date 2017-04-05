// gen_host_calls.m4 BEGIN - used for host calls NOT kernels 

define(`INCLUDE_FFT_FUNCS',`')
// SP stuff

// gen_host_calls.m4:  CALLING sp_defs.m4
flush_all_output
my_include(`../../include/veclib/sp_defs.m4')
// BACK FROM sp_defs.m4
flush_all_output

// gen_host_calls.m4:  CALLING gen_float_calls.m4 for float
flush_all_output
my_include(`../../include/veclib/gen_float_calls.m4')
// BACK FROM gen_float_calls.m4.m4
flush_all_output

// DONE with sp stuff
flush_all_output

// DP stuff

my_include(`../../include/veclib/dp_defs.m4')
// gen_host_calls.m4:  CALLING gen_float_calls.m4 for double
my_include(`../../include/veclib/gen_float_calls.m4')

undefine(`INCLUDE_FFT_FUNCS')

// BY stuff

my_include(`../../include/veclib/by_defs.m4')
my_include(`../../include/veclib/gen_int_calls.m4')


// IN stuff

my_include(`../../include/veclib/in_defs.m4')
my_include(`../../include/veclib/gen_int_calls.m4')


// DI stuff

my_include(`../../include/veclib/di_defs.m4')
my_include(`../../include/veclib/gen_int_calls.m4')


// LI stuff

my_include(`../../include/veclib/li_defs.m4')
my_include(`../../include/veclib/gen_int_calls.m4')


// UBY stuff

my_include(`../../include/veclib/uby_defs.m4')
my_include(`../../include/veclib/gen_uint_calls.m4')


// UIN stuff

my_include(`../../include/veclib/uin_defs.m4')
my_include(`../../include/veclib/gen_uint_calls.m4')


// UDI stuff

my_include(`../../include/veclib/udi_defs.m4')
my_include(`../../include/veclib/gen_uint_calls.m4')


// ULI stuff

my_include(`../../include/veclib/uli_defs.m4')
my_include(`../../include/veclib/gen_uint_calls.m4')


dnl Why is this only for not building kernels?

dnl ifndef(`BUILDING_KERNELS',`

dnl Now mixed precision functions...
dnl We currently implement 4:
dnl inby, spdp, uindi, and ubyin
dnl The second code indicates the destination precision,
dnl Most common use is vsum

define(`MIXED_PRECISION',`')
my_include(`../../include/veclib/inby_defs.m4')
my_include(`../../include/veclib/gen_mixed_int_calls.m4')

my_include(`../../include/veclib/uindi_defs.m4')
my_include(`../../include/veclib/gen_mixed_uint_calls.m4')

my_include(`../../include/veclib/ubyin_defs.m4')
my_include(`../../include/veclib/gen_mixed_uint_calls.m4')

my_include(`../../include/veclib/spdp_defs.m4')
// gen_host_calls.m4:  CALLING gen_mixed_float_calls.m4 for float/double
my_include(`../../include/veclib/gen_mixed_float_calls.m4')
undefine(`MIXED_PRECISION')

dnl	// gpu_int.cl contains special case for left-shift!?  cuda bug?

dnl	// bit stuff
dnl	//
dnl	// This may not work on a GPU, because different threads will need to read
dnl	// and write the same word!?

define(`BIT_PRECISION',`')
my_include(`../../include/veclib/bit_defs.m4')
my_include(`../../include/veclib/gen_bit_calls.m4')
my_include(`../../include/veclib/all_same_prec_vec.m4')
undefine(`BIT_PRECISION')

dnl ',`') dnl endif // BUILDING_KERNELS


// gen_host_calls.m4 DONE


