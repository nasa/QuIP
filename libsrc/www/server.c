#include "quip_config.h"

/* This file was set up to use the w3c libraries, but is no longer
 * maintained, and should be considered obsolete.
 */

#ifdef HAVE_W3C

#include <stdio.h>

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_NETDB_H
#include <netdb.h>
#endif

#ifdef HAVE_DIRENT_H
#include <dirent.h>
#endif

#include "w3c-libwww/WWWLib.h"
#include "w3c-libwww/WWWHTTP.h"
#include "w3c-libwww/WWWInit.h"

#include "server.h"
#include "query.h"
#include "debug.h"			/* verbose */

ITEM_INTERFACE_DECLARATIONS(Server,svr)

static HTRequest * request;	/* should this be per-server?  This way is not thread-safe... */

static COMMAND_FUNC( do_server_read )
{
	const char *s;
	const char *varname;
	char * cwd;
	char * absolute_url;
	HTAnchor * anchor;
	HTChunk * chunk = NULL;

	varname=NAMEOF("target variable name");
	s=NAMEOF("remote file name");

	cwd = HTGetCurrentDirectoryURL();	/* client? */
	absolute_url = HTParse(s, cwd, PARSE_ALL);
if( verbose ){
advise("VERBOSE");
if( strlen(s) > LLEN ){
sprintf(error_string,"length of s is %d",strlen(s));
warn(error_string);
}

sprintf(error_string,"s = %s\n\ncwd = %s\n\nabsolute_url = %s\n",s,cwd,absolute_url);
advise(error_string);
}
	anchor = HTAnchor_findAddress(absolute_url);
if( anchor == NULL ) advise("got null anchor!?");
	chunk = HTLoadAnchorToChunk(anchor, request);
if( chunk == NULL ) advise("got null chunk!?");
	HT_FREE(absolute_url);
	HT_FREE(cwd);


	/* If chunk != NULL then we have the data */
	if (chunk) {
		char * string;
		/* wait until the request is over */
		HTEventList_loop (request);
		string = HTChunk_toCString(chunk);
		/* HTPrint("%s", string ? string : "no text"); */
		ASSIGN_VAR(varname,string);
		HT_FREE(string);
	} else {
		ASSIGN_VAR(varname,"(null)");
	}

	/* svr_finish() stuff used to follow here in standalone program... */
}

void svr_finish()
{
	/* Clean up the request */
	HTRequest_delete(request);	/* jbm - probably shouldn't do this... */
	HTFormat_deleteAll();		/* jbm - ??? defer this also? */

	/* On windows, shut down eventloop as well */
#ifdef WWW_WIN_ASYNC
	HTEventTerminate();
#endif

	/* Terminate the Library */
	HTLibTerminate(); 		/* jbm - don't do this now */
}

//static COMMAND_FUNC( svr_info )
//{
//	advise("svr_info not implemented");
//}

/* stuff lifted from chunkbody.c */

PRIVATE int printer (const char * fmt, va_list pArgs)
{
	return (vfprintf(stdout, fmt, pArgs));
}

PRIVATE int tracer (const char * fmt, va_list pArgs)
{
	return (vfprintf(stderr, fmt, pArgs));
}

PRIVATE int terminate_handler (HTRequest * request, HTResponse * response,
					void * param, int status)
{
	/* Check for status */
	/* HTPrint("Load resulted in status %d\n", status); */

	/* we're not handling other requests */
	HTEventList_stopLoop ();

	/* stop here */
	return HT_ERROR;
}

static void www_lib_init(void)
{
	HTList * converters;
	HTList * encodings;

	converters = HTList_new();		/* List of converters */
	encodings = HTList_new();		/* List of encoders */
	/* Initialize libwww core */
	HTLibInit("TestApp", "1.0");

	/* Gotta set up our own traces */
	HTPrint_setCallback(printer);
	HTTrace_setCallback(tracer);

	/* Turn on TRACE so we can see what is going on */
#if 0
	HTSetTraceMessageMask("sop");
#endif

	/* On windows we must always set up the eventloop */
#ifdef WWW_WIN_ASYNC
	HTEventInit();
#endif

	/* Register the default set of transport protocols */
	HTTransportInit();

	/* Register the default set of protocol modules */
	HTProtocolInit();

	/* Register the default set of BEFORE and AFTER callback functions */
	HTNetInit();

	/* Register the default set of converters */
	HTConverterInit(converters);
	HTFormat_setConversion(converters);

	/* Register the default set of transfer encoders and decoders */
	HTTransferEncoderInit(encodings);
	HTFormat_setTransferCoding(encodings);

	/* Register the default set of MIME header parsers */
	HTMIMEInit();

	/* Add our own filter to handle termination */
	HTNet_addAfter(terminate_handler, NULL, NULL, HT_ALL, HT_FILTER_LAST);

	/* Set up the request and pass it to the Library */
	request = HTRequest_new();
	HTRequest_setOutputFormat(request, WWW_SOURCE);
	HTRequest_setPreemptive(request, YES);
} /* end www_lib_init */

//static COMMAND_FUNC( do_list_svrs ){ list_svrs(); }

static Command svr_ctbl[]={
{	"read",		do_server_read,	"read a file from a server"	},
//{	"list",		do_list_svrs,	"list all connected servers"		},
//{	"info",		svr_info,	"report information about a connected server"		},
{	"quit",		popcmd,		"exit submenu"				},
{ NULL_COMMAND }
};

COMMAND_FUNC( svr_menu )
{
	static int inited=0;

	if( ! inited ){
		www_lib_init();
		inited=1;
	}

	PUSHCMD(svr_ctbl,"servers");
}


#endif /* HAVE_W3C */

