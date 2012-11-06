
#include "all_prot.h"

#define dest_type	double
#define std_type	float
#define std_cpx		SP_Complex
#define dest_cpx	DP_Complex
#define std_quat	SP_Quaternion
#define dest_quat	DP_Quaternion
#define TYP		spdp

#include "method_prot.h"
#include "veclib/math_funcs.c"
#include "veclib/all_vec.c"
#include "veclib/cpx_vec.c"
#include "veclib/signed_vec.c"
#include "nv_undefs.h"

ALL_FP_DECLS( spdp )
ALL_PROT_DECLS( spdp )

