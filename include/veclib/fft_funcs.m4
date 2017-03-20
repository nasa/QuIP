/* fft_funcs.m4 what should we put here??? */

dnl call platform-specific file...

/* fft_funcs.m4 `pf_str' = pf_str */
define(`include_platform_fft',`my_include(pf_str`'_fft_funcs.m4)')

/* BEGIN including platform fft funcs */
include_platform_fft
/* DONE including platform fft funcs */


