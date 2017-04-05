dnl this file in read for both kernels and host calls?
dnl also read for fast, eqsp, and slow kernels!

dnl	// These are the float funcs that can mix sp and dp

// gen_float_calls.m4:  CALLING gen_mixed_float_calls.m4
my_include(`../../include/veclib/gen_mixed_float_calls.m4')



divert(0) dnl enable output

dnl	// These are the float funcs where all operands must be either sp or dp, not mixed
my_include(`../../include/veclib/all_same_prec_vec.m4')

dnl fft_funcs are not implemented in a platform-independent way!?
dnl - it ought to be possible to write the host function platform-independent,
dnl but have different kernels?
ifdef(`INCLUDE_FFT_FUNCS',`
/* gen_float_calls.m4:  including fft_funcs.m4 */
my_include(`../../include/veclib/fft_funcs.m4')
',`
/* gen_float_calls.m4:  NOT including fft_funcs.m4 */
')

my_include(`../../include/veclib/new_conv.m4')
suppress_if dnl suppress output

