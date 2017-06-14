/* sp_defs.m4 BEGIN */
define(`absfunc',`fabsf($1)')
define(`std_type',`float')
define(`simd_type',`v4sf')
define(`std_cpx',`SP_Complex')
define(`std_quat',`SP_Quaternion')
define(`dest_type',`float')
define(`dest_cpx',`SP_Complex')
define(`dest_quat',`SP_Quaternion')
define(`type_code',`sp')
define(`std_scalar',`u_f')
define(`std_cpx_scalar',`u_spc')
define(`std_quat_scalar',`u_spq')

define(`ASSIGN_CPX',`$1 = $2')
define(`ASSIGN_QUAT',`$1 = $2')

define(`REQUIRED_DST_PREC',`PREC_SP')
define(`REQUIRED_SRC_PREC',`PREC_SP')

define(`MY_CLFFT_PRECISION',`CLFFT_SINGLE')

my_include(`veclib/sp_func_defs.m4')

/* `TYPE_CODE =' type_code   `dest_type ='dest_type */
/* sp_defs.m4 DONE */

dnl	definitions for CUDA cuFFT
define(`cuda_cpx_fft_func',`cufftExecC2C')
define(`cuda_real_fft_func',`cufftExecR2C')
define(`cuda_real_ift_func',`cufftExecC2R')
define(`cuda_cpx_fft_type',`cufftComplex')
define(`cuda_real_fft_type',`cufftReal')

my_include(`veclib/sp_simd_funcs.m4')

