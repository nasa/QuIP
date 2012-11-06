#include "quip_config.h"

char VersionId_newvec_lifuncs[] = QUIP_VERSION_STRING;

#include "nvf.h"

#include "li_prot.h"

#define std_type	int64_t
#define dest_type	int64_t
#define std_scalar	u_ll		/* member of Scalar_Val union, see data_obj.h */

/* BUG put test for llabs in configure.ac */
#include <stdlib.h>
#define absfunc		llabs

#include "new_ops.h"

#define TYP	li
#define SIGNED_SRC_PRECISION
#define REQUIRED_DST_PREC	PREC_LI
#define REQUIRED_SRC_PREC	PREC_LI

#include "veclib/all_vec.c"
#include "veclib/signed_vec.c"
#include "veclib/intvec.c"
/* #include "linear.c" */

