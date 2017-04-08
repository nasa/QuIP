
ifdef(`BUILD_FOR_OPENCL',`
my_include(`../../include/veclib/dp_func_defs.m4')
',` dnl else // ! BUILD_FOR_OPENCL
suppress_if
define(`floor_func',`floorf')
define(`trunc_func',`truncf')
define(`ceil_func',`ceilf')
define(`round_func',`roundf')
define(`sqrt_func',`sqrtf')
define(`rint_func',`rintf')
define(`cos_func',`cosf')
define(`sin_func',`sinf')
define(`acos_func',`acosf')
define(`asin_func',`asinf')
define(`tan_func',`tanf')
define(`atan_func',`atanf')
define(`atan2_func',`atan2f')
define(`exp_func',`expf')
define(`erf_func',`erff')
define(`erfinv_func',`erfinvf')
define(`log10_func',`log10f')
define(`log_func',`logf')
define(`pow_func',`powf')
dnl these are double precision funcs...
dnl define gamma_func,gsl_sf_gamma
dnl define lngamma_func,gsl_sf_lngamma

dnl these are single-prec versions...
define(`gamma_func',`tgammaf')
define(`lngamma_func',`lgammaf')

') dnl endif // ! BUILD_FOR_OPENCL

suppress_no

