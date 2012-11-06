#include "nvf.h"

#include "dp_prot.h"

#define absfunc		fabs

#include <math.h>

#define twiddle		dp_twiddle
#define _sinfact	dp_sinfact
#define _isinfact	dp_isinfact
#define init_twiddle	init_dp_twiddle
#define init_sinfact	init_dp_sinfact

#include "new_ops.h"

#define std_type	double
#define std_cpx		DP_Complex
#define std_quat	DP_Quaternion
#define dest_type	double
#define dest_cpx	DP_Complex
#define dest_quat	DP_Quaternion
#define std_scalar	u_d		/* member of Scalar_Val union, see data_obj.h */
#define std_cpx_scalar	u_dpc		/* member of Scalar_Val union, see data_obj.h */
#define std_quat_scalar	u_dpq		/* member of Scalar_Val union, see data_obj.h */

#define TYP	dp
#define SIGNED_SRC_PRECISION
#define REQUIRED_SRC_PREC	PREC_DP
#define REQUIRED_DST_PREC	PREC_DP

