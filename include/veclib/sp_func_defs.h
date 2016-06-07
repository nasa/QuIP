
#ifdef BUILD_FOR_OPENCL

#include "dp_func_defs.h"

#else // ! BUILD_FOR_OPENCL

#define floor_func	floorf
#define trunc_func	truncf
#define ceil_func	ceilf
#define round_func	roundf
#define sqrt_func	sqrtf
#define rint_func	rintf
#define cos_func	cosf
#define sin_func	sinf
#define acos_func	acosf
#define asin_func	asinf
#define tan_func	tanf
#define atan_func	atanf
#define atan2_func	atan2f
#define exp_func	expf
#define erf_func	erff
#define erfinv_func	erfinvf
#define log10_func	log10f
#define log_func	logf
#define pow_func	powf
// these are double precision funcs...
//#define gamma_func	gsl_sf_gamma
//#define lngamma_func	gsl_sf_lngamma

// these are single-prec versions...
#define gamma_func	tgammaf
#define lngamma_func	lgammaf

#endif // ! BUILD_FOR_OPENCL
