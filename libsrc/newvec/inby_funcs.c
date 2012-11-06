#include "quip_config.h"

char VersionId_newvec_inbyfuncs[] = QUIP_VERSION_STRING;

#include "nvf.h"

#include "inby_prot.h"

#define std_type	short
#define dest_type	char
#define std_scalar	u_s		/* member of Scalar_Val union, see data_obj.h */

#include <math.h>
#define absfunc		fabs

#include "new_ops.h"

#define TYP	inby
#define SIGNED_SRC_PRECISION
#define REQUIRED_DST_PREC	PREC_IN
#define REQUIRED_SRC_PREC	PREC_BY

#include "veclib/all_vec.c"
#include "veclib/intvec.c"
#include "veclib/signed_vec.c"
/* #include "linear.c" */

