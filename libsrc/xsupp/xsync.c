#include "quip_config.h"

char VersionId_xsupp_xsync[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

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

