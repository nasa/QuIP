#include "quip_config.h"

#ifdef HAVE_X11

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "xsupp.h"

static const char *display_name=NULL;

const char *_which_display( SINGLE_QSP_ARG_DECL )
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
		WARN("environment variable HOSTNAME not set, using :0");
	else if( *display_name == 0 )
		WARN("environment variable HOSTNAME set to null string, using :0");
	display_name=":0";

	return(display_name);
}

#endif /* HAVE_X11 */

