#include "quip_config.h"

char VersionId_newvec_byinfuncs[] = QUIP_VERSION_STRING;

#include "nvf.h"

#include "ubyin_prot.h"

#define std_type	unsigned char
#define std_signed	char
#define dest_type	short
#define std_scalar	u_b		/* member of Scalar_Val union, see data_obj.h */

#define absfunc		abs

#include "new_ops.h"

#define TYP	ubyin
#define UNSIGNED_SRC_PRECISION
#define REQUIRED_DST_PREC	PREC_IN
#define REQUIRED_SRC_PREC	PREC_UBY

#include "veclib/all_vec.c"
#include "veclib/intvec.c"
#include "veclib/unsigned_vec.c"
/* #include "linear.c" */

