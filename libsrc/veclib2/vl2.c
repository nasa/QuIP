
/* jbm's interface to cuda devices */

/* This file contains the menu-callable functions, which in turn call
 * host functions which are typed and take an oap argument.
 * These host functions then call the gpu kernel functions...
 */

#include "quip_config.h"

#ifdef HAVE_STRING_H
#include <string.h>
#endif // HAVE_STRING_H

#include "quip_prot.h"
#include "my_vl2.h"	// 
#include "veclib_api.h"

#include "veclib/vl2_veclib_prot.h"
//#include "veclib/vl2_menu_prot.h"	// prototypes

// BUG name conflict!?
// universal function, or needs porting???


void *TMPVEC_NAME(Platform_Device *pdp, size_t size, size_t len, const char *whence)
{
	return getbuf( size * len );
}

void FREETMP_NAME(void *a, const char *whence)
{
	givbuf(a);
}


