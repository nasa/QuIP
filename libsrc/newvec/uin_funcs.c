#include "quip_config.h"

char VersionId_newvec_uinfuncs[] = QUIP_VERSION_STRING;

#include "nvf.h"

#include "uin_prot.h"

#define std_type	u_short
#define std_signed	short
#define dest_type	u_short
#define std_scalar	u_us		/* member of Scalar_Val union, see data_obj.h */

#define absfunc		abs

#include "new_ops.h"

#define TYP	uin
#define UNSIGNED_SRC_PRECISION
#define REQUIRED_DST_PREC	PREC_UIN
#define REQUIRED_SRC_PREC	PREC_UIN

#include "veclib/all_vec.c"
#include "veclib/intvec.c"
#include "veclib/unsigned_vec.c"
/* #include "linear.c" */

