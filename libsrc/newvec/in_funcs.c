#include "quip_config.h"

char VersionId_newvec_infuncs[] = QUIP_VERSION_STRING;

#include "nvf.h"

#include "in_prot.h"

#define std_type	short
#define dest_type	short
#define std_scalar	u_s		/* member of Scalar_Val union, see data_obj.h */

#define absfunc		abs

#include "new_ops.h"


#define TYP	in
#define SIGNED_SRC_PRECISION
#define REQUIRED_DST_PREC	PREC_IN
#define REQUIRED_SRC_PREC	PREC_IN

#include "veclib/all_vec.c"
#include "veclib/signed_vec.c"
#include "veclib/intvec.c"
/* #include "linear.c" */

