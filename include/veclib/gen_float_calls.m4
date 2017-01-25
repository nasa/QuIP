dnl this file in read for both kernels and host calls?

// These are the float funcs that can mix sp and dp

// CALLING gen_mixed_float_calls.m4
my_include(`../../include/veclib/gen_mixed_float_calls.m4')


// These are the float funcs where all operands must be either sp or dp, not mixed

divert(0) dnl enable output
my_include(`../../include/veclib/all_same_prec_vec.m4')
dnl fft_funcs are not implemented in a platform-independent way!?
dnl - it ought to be possible to write the host function platform-independent,
dnl but have different kernels?
my_include(`../../include/veclib/fft_funcs.m4')
my_include(`../../include/veclib/new_conv.m4')
suppress_if dnl suppress output

