#include "quip_config.h"

/* this table defines all of the functions.
 * It gives their name, a mask that tells the supported precisions,
 * and an entry point...
 */

#include <stdlib.h>	/* qsort */
//#include "nvf.h"
//#include "item.h"
//#include "version.h"
//#include "rn.h"		/* set_random_seed */
//#include "debug.h"
//#include "warn.h"
#include "quip_prot.h"
#include "platform.h"
#include "veclib/vl2_veclib_prot.h"
#include "vl2_func_tbl.h"

/* This used to be an initialized table,
 * but in Objective C we have to do it differently...
 *
 * This table contains information about the functions, but isn't
 * specific to a particular platform.  The elements are the name,
 * the code, a code indicating the argument types, a mask indicating
 * what precisions (and combinations) are allowed, and a mask
 * indicating what types (real/complex/quaternion/mixed/etc)
 * are allowed.
 */

#define ADD_FUNC_TO_TBL( func, idx )	{ idx, h_vl2_##func },
#define ADD_CPU_FUNC_TO_TBL( func, idx )	{ idx, h_vl2_##func },
//BEGIN_VL2_VFUNC_DECLS
Dispatch_Function vl2_func_tbl[]={

#include "veclib/dispatch_tbl.c"

};
//END_VL2_VFUNC_DECLS


// In the old coding, we created items and copied their values from the table...
// But we don't need to have two copies - we just have to insert the names

#define N_VL2_FUNCS		(sizeof(vl2_func_tbl)/sizeof(Dispatch_Entry))


