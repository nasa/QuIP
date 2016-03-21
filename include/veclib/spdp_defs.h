
#define absfunc fabs
#define std_type float
#define std_cpx SP_Complex
#define std_quat SP_Quaternion
#define dest_type double
#define dest_cpx DP_Complex
#define dest_quat DP_Quaternion
#define type_code spdp
#define std_scalar	u_f
#define std_cpx_scalar	u_spc
#define std_quat_scalar	u_spq

#define ASSIGN_CPX(to,fr)	to.re = fr.re; to.im = fr.im

#define ASSIGN_QUAT(to,fr)	to.re = fr.re; \
				to._i = fr._i; \
				to._j = fr._j; \
				to._k = fr._k

#define MIXED_PRECISION

// float input and double output -
#include "veclib/dp_func_defs.h"

