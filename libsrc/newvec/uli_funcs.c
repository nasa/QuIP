#include "quip_config.h"

char VersionId_newvec_ulifuncs[] = QUIP_VERSION_STRING;

#include "nvf.h"

#include "uli_prot.h"

#include "new_ops.h"

#define std_type	uint64_t
#define std_signed	int64_t
#define dest_type	uint64_t
#define std_scalar	u_ull		/* member of Scalar_Val union, see data_obj.h */

#define TYP	uli
#define UNSIGNED_SRC_PRECISION
#define REQUIRED_DST_PREC	PREC_ULI
#define REQUIRED_SRC_PREC	PREC_ULI

#include "veclib/all_vec.c"
#include "veclib/intvec.c"
#include "veclib/unsigned_vec.c"
/* #include "linear.c" */

