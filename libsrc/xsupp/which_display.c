#include "quip_config.h"

char VersionId_xsupp_which_display[] = QUIP_VERSION_STRING;

#ifdef HAVE_X11

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "debug.h"
#include "version.h"
#include "xsupp.h"

static const char *display_name=NULL;

const char *which_display( VOID )
{
	display_name = check_display();
	if( !strncmp(display_name, ":0", 1 ))
		display_name=getenv("HOSTNAME");

	if( display_name != NULL && *display_name != 0 ){
	
		if( verbose ) {
			sprintf(DEFAULT_ERROR_STRING,"Using display %s\n",display_name);
			advise(DEFAULT_ERROR_STRING);
		}
		return(display_name);
	}

	if( display_name == NULL )
		NWARN("environment variable HOSTNAME not set, using :0");
	else if( *display_name == 0 )
		NWARN("environment variable HOSTNAME set to null string, using :0");
	display_name=":0";

	return(display_name);
}

#endif /* HAVE_X11 */

