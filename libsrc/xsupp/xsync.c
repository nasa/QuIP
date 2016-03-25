#include "quip_config.h"

#ifdef HAVE_X11

#include <stdio.h>
#include "xsupp.h"

void x_sync_off()
{
	Disp_Obj *dop;

	dop = curr_dop();
	if( dop == NO_DISP_OBJ ) return;

	XSynchronize(dop->do_dpy,False);
}


void x_sync_on()
{
	Disp_Obj *dop;

	dop = curr_dop();
	if( dop == NO_DISP_OBJ ) return;

	XSynchronize(dop->do_dpy,True);
}

#endif /* HAVE_X11 */

