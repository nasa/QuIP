
#include "quip_config.h"

char VersionId_newvec_spdpfuncs[] = QUIP_VERSION_STRING;

#include "nvf.h"

#include <math.h>
#define absfunc		fabs

#define twiddle		spdp_twiddle
#define _sinfact	spdp_sinfact
#define _isinfact	spdp_isinfact
#define init_twiddle	init_spdp_twiddle
#define init_sinfact	init_spdp_sinfact


#include "spdp_prot.h"

#include "new_ops.h"

#define std_type	float
#define dest_type	double
#define std_scalar	u_f		/* member of Scalar_Val union, see data_obj.h */
#define std_cpx_scalar	u_spc		/* member of Scalar_Val union, see data_obj.h */
#define std_quat_scalar	u_spq		/* member of Scalar_Val union, see data_obj.h */
#define std_cpx		SP_Complex
#define dest_cpx	DP_Complex
#define std_quat	SP_Quaternion
#define dest_quat	DP_Quaternion

#define TYP	spdp
#define SIGNED_SRC_PRECISION
#define REQUIRED_DST_PREC	PREC_DP
#define REQUIRED_SRC_PREC	PREC_SP
#define REQUIRED_CPX_SRC_PREC	PREC_CPX
#define REQUIRED_CPX_DST_PREC	PREC_DBLCPX

static std_type std_tmp;
static std_type tmp_denom;

#ifdef QUATERNION_SUPPORT
static dest_quat tmpq;	/* BUG not thread safe, should not be global */
#endif /* QUATERNION_SUPPORT */

#include "linear.c"
#include "veclib/all_vec.c"
#include "veclib/cpx_vec.c"
#include "veclib/signed_vec.c"
#include "veclib/math_funcs.c"

