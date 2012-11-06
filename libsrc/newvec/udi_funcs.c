#include "quip_config.h"

char VersionId_newvec_udifuncs[] = QUIP_VERSION_STRING;

#include "nvf.h"

#include "udi_prot.h"

#include "new_ops.h"

#define std_type	uint32_t
#define std_signed	int32_t
#define dest_type	uint32_t
#define std_scalar	u_ul		/* member of Scalar_Val union, see data_obj.h */

#define TYP	udi
#define UNSIGNED_SRC_PRECISION
#define REQUIRED_DST_PREC	PREC_UDI
#define REQUIRED_SRC_PREC	PREC_UDI

#include "veclib/all_vec.c"
#include "veclib/intvec.c"
#include "veclib/unsigned_vec.c"
/* #include "linear.c" */

