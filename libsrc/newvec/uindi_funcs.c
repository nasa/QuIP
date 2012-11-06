#include "quip_config.h"

char VersionId_newvec_indifuncs[] = QUIP_VERSION_STRING;

#include "nvf.h"

#include "uindi_prot.h"

#define std_type	u_short
#define std_signed	short
#define dest_type	int32_t
#define std_scalar	u_s		/* member of Scalar_Val union, see data_obj.h */

#include <math.h>
#define absfunc		fabs

#include "new_ops.h"

#define TYP	uindi
#define UNSIGNED_SRC_PRECISION
#define REQUIRED_DST_PREC	PREC_DI
#define REQUIRED_SRC_PREC	PREC_UIN

#include "veclib/all_vec.c"
#include "veclib/intvec.c"
#include "veclib/unsigned_vec.c"
/* #include "linear.c" */

