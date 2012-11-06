#include "quip_config.h"

#ifdef HAVE_LIBCURL

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

#ifdef HAVE_CURL_CURL_H
#include <curl/curl.h>
#endif

#include "server.h"
#include "query.h"
#include "debug.h"			/* verbose */

ITEM_INTERFACE_DECLARATIONS(Server,svr)

static CURL * easy_handle;
static FILE *fp_out=NULL;

size_t my_write_data(void *buffer, size_t size, size_t nmemb, void *userp)
{
	int n;

	//sprintf(error_string,"my_write_data:  received a buffer with size = %d bytes",size*nmemb);
	//advise(error_string);
	prt_msg_frag(".");

	if( fp_out != NULL ){
		n=fwrite(buffer,size,nmemb,fp_out);
		if( n != (int) (size*nmemb) ){
			sprintf(error_string,"Requested %ld bytes but only wrote %d",
						size*nmemb,n);
			warn(error_string);
		}
	}

	return(size*nmemb);
}

static char my_curl_error_buffer[1024];

static COMMAND_FUNC( do_server_read )
{
	const char *url;
	const char *s;
	CURLcode success;

	url=NAMEOF("remote file name or URL");
	s=NAMEOF("output filename");

	fp_out = try_open(s,"w");
	if( fp_out == NULL ) return;

	/* BUG?  make sure this is a sensible URL? */
	curl_easy_setopt(easy_handle,CURLOPT_URL,url);

	curl_easy_setopt(easy_handle,CURLOPT_ERRORBUFFER,my_curl_error_buffer);

	curl_easy_setopt(easy_handle, CURLOPT_WRITEFUNCTION, my_write_data);

	success = curl_easy_perform(easy_handle);
	prt_msg("");

	if( success != 0 ){
		warn("curl_easy_perform error...");
		advise(my_curl_error_buffer);
	}

	if( fp_out != NULL ){
		fclose(fp_out);
		fp_out=NULL;
	}
}


//static COMMAND_FUNC( do_list_svrs ){ list_svrs(); }

static Command curl_ctbl[]={
{	"read",		do_server_read,	"read a file from a server"	},
{	"quit",		popcmd,		"exit submenu"				},
{ NULL_COMMAND }
};

COMMAND_FUNC( svr_menu )
{
	static int inited=0;

	if( ! inited ){
		curl_version_info_data *vdp;

		curl_global_init(CURL_GLOBAL_SSL);

		vdp = curl_version_info(CURLVERSION_NOW);

		/* need matching call to curl_global_cleanup() */

		easy_handle = curl_easy_init();

		inited=1;
	}

	PUSHCMD(curl_ctbl,"curl");
}


#endif /* HAVE_LIBCURL */

