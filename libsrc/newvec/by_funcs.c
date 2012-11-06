#include "quip_config.h"

char VersionId_newvec_byfuncs[] = QUIP_VERSION_STRING;

#include "nvf.h"

#include "by_prot.h"

#define std_type	char
#define dest_type	char
#define std_scalar	u_b		/* member of Scalar_Val union, see data_obj.h */

#define absfunc		abs

#include "new_ops.h"

#define TYP	by
#define SIGNED_SRC_PRECISION
#define REQUIRED_DST_PREC	PREC_BY
#define REQUIRED_SRC_PREC	PREC_BY

#include "veclib/all_vec.c"
#include "veclib/signed_vec.c"
#include "veclib/intvec.c"
/* #include "linear.c" */

