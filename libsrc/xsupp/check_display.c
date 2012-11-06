#include "quip_config.h"

char VersionId_xsupp_check_display[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#include "query.h"	// error_string
#include "debug.h"
#include "version.h"

static const char *display_name=NULL;

/* read the name of the preferred display from the environment */
/* This function could have a much more descriptive name... */

const char *check_display()
{
	if( display_name != NULL ) return(display_name);

/* this is now done in viewmenu... */

	display_name=getenv("LOCAL_DISPLAY");
	if( display_name == NULL )
		display_name=getenv("DISPLAY");

	if( display_name != NULL && *display_name != 0 ){
		if( verbose ) {
			sprintf(DEFAULT_ERROR_STRING,"Using display %s\n",display_name);
			advise(DEFAULT_ERROR_STRING);
		}
		return(display_name);
	}

	if( display_name == NULL )
		NWARN("environment variable DISPLAY not set, using :0");
	else if( *display_name == 0 )
		NWARN("environment variable DISPLAY set to null string, using :0");
	display_name=":0";

	return(display_name);
}

