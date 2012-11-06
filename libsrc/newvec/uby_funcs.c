#include "quip_config.h"

char VersionId_newvec_ubyfuncs[] = QUIP_VERSION_STRING;

#include "nvf.h"

#include "uby_prot.h"

#include "new_ops.h"

#define std_type	u_char
#define std_signed	char
#define dest_type	u_char
#define std_scalar	u_ub		/* member of Scalar_Val union, see data_obj.h */

#define TYP	uby
#define UNSIGNED_SRC_PRECISION
#define REQUIRED_DST_PREC	PREC_UBY
#define REQUIRED_SRC_PREC	PREC_UBY

#include "linear.c"
#include "veclib/all_vec.c"
#include "veclib/intvec.c"
#include "veclib/unsigned_vec.c"

