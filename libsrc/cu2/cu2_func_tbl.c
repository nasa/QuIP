#include "quip_config.h"

/* this table defines all of the functions.
 * It gives their name, a mask that tells the supported precisions,
 * and an entry point...
 */

#include <stdlib.h>	/* qsort */
//#include "nvf.h"
////#include "item.h"
////#include "version.h"
//#include "rn.h"		/* set_random_seed */
//#include "debug.h"
//#include "warn.h"

//#include "veclib_prot.h"
#ifdef HAVE_CUDA
#include "veclib/cu2_veclib_prot.h"

#include "quip_prot.h"
#include "cu2_func_tbl.h"

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

//BEGIN_CU2_VFUNC_DECLS

#define ADD_FUNC_TO_TBL( func, index )	{	index,	h_cu2_##func	},
#define ADD_CPU_FUNC_TO_TBL( func, index )	{	index,	NULL	},

Dispatch_Function cu2_func_tbl[]={

#include "veclib/dispatch_tbl.c"

//END_CU2_VFUNC_DECLS
};


// In the old coding, we created items and copied their values from the table...
// But we don't need to have two copies - we just have to insert the names

#define N_CU2_FUNCS		(sizeof(cu2_func_tbl)/sizeof(Dispatch_Function))

#endif // HAVE_CUDA

