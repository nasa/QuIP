#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#include "quip_prot.h"
#include "xsupp.h"

static const char *display_name=NULL;

/* read the name of the preferred display from the environment */
/* This function could have a much more descriptive name... */

const char *_check_display(SINGLE_QSP_ARG_DECL)
{
	if( display_name != NULL ) return(display_name);

/* this is now done in viewmenu... */

	display_name=getenv("LOCAL_DISPLAY");
	if( display_name == NULL )
		display_name=getenv("DISPLAY");

	if( display_name != NULL && *display_name != 0 ){
		if( verbose ) {
			sprintf(ERROR_STRING,"Using display %s\n",display_name);
			advise(ERROR_STRING);
		}
		return(display_name);
	}

	if( display_name == NULL )
		warn("environment variable DISPLAY not set, using :0");
	else if( *display_name == 0 )
		warn("environment variable DISPLAY set to null string, using :0");
	display_name=":0";

	return(display_name);
}

