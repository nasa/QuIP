
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

#endif // ! BUILD_FOR_OPENCL
