
#include "quip_config.h"

char VersionId_newvec_spfuncs[] = QUIP_VERSION_STRING;

#include "sp_common.h"

// BUG these are globals, so not thread-safe
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

