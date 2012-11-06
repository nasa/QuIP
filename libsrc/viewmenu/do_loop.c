#include "quip_config.h"

char VersionId_viewmenu_do_loop[] = QUIP_VERSION_STRING;

/* this sure looks like xlib dependent stuff!!! BUG move to view_xlib.c */

#include "view_cmds.h"
#include "view_util.h"
#include "debug.h"
#include "xsupp.h"


/*
 * The purpose of these functions is to allow concurrent processing
 * of console terminal input and window system events.
 *
 * The current scheme works well when we are basically typing
 * stuff in, but want to catch window events also; this way
 * we check for events before processing each command...
 *
 * A problem arises with the program guimenu:
 * Window events generate calls to pushtext(), placing
 * text in a buffer to be executed by the next do_cmd().
 * But when one of the commands includes a call to do_loop(),
 * then execution can get to here, and the call to do_cmd()
 * from within do_loop() will cause a blocking attempt
 * to get tty input - perhaps because lookahead is disabled?
 *
 */

static int looping=0;

COMMAND_FUNC( stop_loop ) { looping=0; }

COMMAND_FUNC( do_loop )
{
	insure_x11_server(SINGLE_QSP_ARG);

	if( looping ) return;		/* avoid recursion */

	looping=1;

	while( looping ){
		while( event_loop(SINGLE_QSP_ARG) != -1 )	/* i_loop(); */
			;

		do_cmd(SINGLE_QSP_ARG);	/* process a command */
	}
}

static int doing_event_redir=0;

COMMAND_FUNC( event_redir )
{
	insure_x11_server(SINGLE_QSP_ARG);

	if( doing_event_redir ) return;		/* avoid recursion */

	doing_event_redir = 1;
	
	while( doing_event_redir ){
		event_loop(SINGLE_QSP_ARG);
	}
}

COMMAND_FUNC( event_unredir )
{
	doing_event_redir=0;
}

