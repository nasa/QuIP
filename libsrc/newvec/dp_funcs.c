
#include "quip_config.h"

char VersionId_newvec_dpfuncs[] = QUIP_VERSION_STRING;

#include "dp_common.h"

static std_type std_tmp;
static std_type tmp_denom;

#ifdef QUATERNION_SUPPORT
static std_quat tmpq;
#endif /* QUATERNION_SUPPORT */

#include "linear.c"
#include "veclib/all_vec.c"
#include "veclib/signed_vec.c"
#include "veclib/cpx_vec.c"
#include "veclib/math_funcs.c"

