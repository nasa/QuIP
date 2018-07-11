suppress_if
dnl this file in read for both kernels and host calls?
dnl also read for fast, eqsp, and slow kernels!

dnl	// These are the float funcs that can mix sp and dp

// gen_float_calls.m4:  CALLING gen_mixed_float_calls.m4
my_include(`veclib/gen_mixed_float_calls.m4')

dnl	// These are the float funcs where all operands must be either sp or dp, not mixed
dnl	only vset & vmov ???
my_include(`veclib/all_same_prec_vec.m4')

dnl	dnl fft_funcs are not implemented in a platform-independent way!?
dnl	dnl - it ought to be possible to write the host function platform-independent,
dnl	dnl but have different kernels?
dnl	ifdef(`INCLUDE_FFT_FUNCS',`
dnl	suppress_no
dnl	/* gen_float_calls.m4:  including fft_funcs.m4 */
dnl	my_include(`veclib/fft_funcs.m4')
dnl	',`
dnl	suppress_no
dnl	/* gen_float_calls.m4:  NOT including fft_funcs.m4 */
dnl	')

my_include(`veclib/new_conv.m4')

suppress_no
