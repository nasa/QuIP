
#include "nvf.h"

#include "sp_prot.h"

#define absfunc		fabs

#include <math.h>

#define twiddle		sp_twiddle
#define _sinfact	sp_sinfact
#define _isinfact	sp_isinfact
#define init_twiddle	init_sp_twiddle
#define init_sinfact	init_sp_sinfact

#include "new_ops.h"

#define std_type	float
#define std_cpx		SP_Complex
#define std_quat	SP_Quaternion
#define dest_type	float
#define dest_cpx	SP_Complex
#define dest_quat	SP_Quaternion
#define std_scalar	u_f		/* member of Scalar_Val union, see data_obj.h */
#define std_cpx_scalar	u_spc		/* member of Scalar_Val union, see data_obj.h */
#define std_quat_scalar	u_spq		/* member of Scalar_Val union, see data_obj.h */

#define TYP	sp
#define SIGNED_SRC_PRECISION
#define REQUIRED_SRC_PREC	PREC_SP
#define REQUIRED_DST_PREC	PREC_SP
/*#define REQUIRED_CPX_PREC	PREC_CPX */

