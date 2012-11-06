#include "quip_config.h"

char VersionId_newvec_difuncs[] = QUIP_VERSION_STRING;

#include "nvf.h"

#include "di_prot.h"

#define std_type	int32_t
#define dest_type	int32_t
#define std_scalar	u_l		/* member of Scalar_Val union, see data_obj.h */


#define absfunc		abs

#include "new_ops.h"

#define TYP	di
#define SIGNED_SRC_PRECISION
#define REQUIRED_DST_PREC	PREC_DI
#define REQUIRED_SRC_PREC	PREC_DI

#include "veclib/all_vec.c"
#include "veclib/signed_vec.c"
#include "veclib/intvec.c"
/* #include "linear.c" */

