
#define absfunc fabsf
#define std_type float
#define std_cpx SP_Complex
#define std_quat SP_Quaternion
#define dest_type float
#define dest_cpx SP_Complex
#define dest_quat SP_Quaternion
#define type_code sp
#define std_scalar	u_f
#define std_cpx_scalar	u_spc
#define std_quat_scalar	u_spq

#define ASSIGN_CPX(to,fr)	to = fr
#define ASSIGN_QUAT(to,fr)	to = fr

#define REQUIRED_DST_PREC	PREC_SP
#define REQUIRED_SRC_PREC	PREC_SP

#define MY_CLFFT_PRECISION	CLFFT_SINGLE

#include "veclib/sp_func_defs.h"

