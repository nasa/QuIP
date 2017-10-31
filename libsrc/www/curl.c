#include "quip_config.h"
#include "server.h"

#ifdef HAVE_LIBCURL

#include <stdio.h>
#include "my_curl.h"
#include "quip_prot.h"
#include "server.h"
#include "my_encryption.h"
#include "query_stack.h"

static void add_text_to_buffer(QSP_ARG_DECL  void *data, size_t nbytes )
{
	u_int n_need;
	u_int n_have;
	u_int i_start;
	String_Buf *sbp;

	sbp = CURL_STRINGBUF;
	assert( sbp != NULL );
	/*
	if( CURL_STRINGBUF == NULL ){
		CURL_STRINGBUF = new_stringbuf();
		n_have = 0;
	} else {
		n_have = strlen(CURL_STRINGBUF->sb_buf);
	}
	*/
	if(sb_buffer(sbp) == NULL)
		n_have=0;
	else
		n_have = strlen(sb_buffer(sbp));


	if( (n_need=n_have+nbytes+1) > sb_size(sbp) ){
		enlarge_buffer(sbp,n_need);
	}

	i_start = strlen(sb_buffer(sbp));
	memcpy(sb_buffer(sbp)+i_start,data,nbytes);
	(sb_buffer(sbp))[i_start+nbytes]=0;
}

static size_t buffer_url_text(void *buffer, size_t size, size_t nmemb, void *userp)
{
	Query_Stack *qsp;

	qsp = (Query_Stack *) userp;

	// The data comes in chunks of unknown size...

//sprintf(ERROR_STRING,"buffer_url_text:  %ld elements of size %ld",nmemb,size);
//advise(ERROR_STRING);

	// Make sure that all characters are printable.
	// Data from libcurl is not null terminated, we want to count lines
	// so we shouldn't replace the final newline with a null.
	// The safest thing would be to load all the data to a buffer,
	// which might be larger than the individual chunks.

	add_text_to_buffer(QSP_ARG  buffer,size*nmemb);

	return(size*nmemb);
}


String_Buf *get_url_contents(QSP_ARG_DECL  const char *url)
{
	CURLcode success;

	/* BUG?  should we make sure this is a sensible URL? */
	curl_easy_setopt(EASY_HANDLE,CURLOPT_URL,url);
	curl_easy_setopt(EASY_HANDLE,CURLOPT_ERRORBUFFER,MY_CURL_ERROR_BUFFER);
	curl_easy_setopt(EASY_HANDLE, CURLOPT_WRITEDATA, THIS_QSP);
	curl_easy_setopt(EASY_HANDLE, CURLOPT_WRITEFUNCTION, buffer_url_text);

	success = curl_easy_perform(EASY_HANDLE);
	prt_msg("");		// newline after the dots we have been printing...

	if( success != 0 ){
		warn("curl_easy_perform error...");
		advise(MY_CURL_ERROR_BUFFER);
		return NULL;
	}

	// Now all the data is in the string buffer...
#ifdef CAUTIOUS
	if( CURL_STRINGBUF == NULL ){
		WARN("CAUTIOUS:  read_url:  null string buffer!?");
		return NULL;
	}
#endif /* CAUTIOUS */

	return(CURL_STRINGBUF);
}

void write_file_from_url( QSP_ARG_DECL  FILE *fp_out, const char *url )
{
	CURLcode success;

	/* BUG?  should we make sure this is a sensible URL? */
	curl_easy_setopt(EASY_HANDLE,CURLOPT_URL,url);
	/* BUG?  do we know that the error buffer size is sufficient? */
	curl_easy_setopt(EASY_HANDLE,CURLOPT_ERRORBUFFER,MY_CURL_ERROR_BUFFER);
	curl_easy_setopt(EASY_HANDLE, CURLOPT_WRITEDATA, fp_out);
	// Set this to NULL to use default write function,
	// which write to fp set by CURLOPT_WRITEDATA
	curl_easy_setopt(EASY_HANDLE, CURLOPT_WRITEFUNCTION, NULL);

	success = curl_easy_perform(EASY_HANDLE);
	prt_msg("");

	if( success != 0 ){
		warn("curl_easy_perform error...");
		advise(MY_CURL_ERROR_BUFFER);
	}
}

void init_http_subsystem(SINGLE_QSP_ARG_DECL)
{
	curl_version_info_data *vdp;
	curl_global_init(CURL_GLOBAL_SSL);
	vdp = curl_version_info(CURLVERSION_NOW);
	/* BUG need matching call to curl_global_cleanup() */
	THIS_QSP->qs_curl_info = getbuf( sizeof(*(THIS_QSP->qs_curl_info)));
	CURL_STRINGBUF = NULL;
	CURL_VERSION_INFO = vdp;
	EASY_HANDLE = curl_easy_init();
}

#else /* !HAVE_LIBCURL */

String_Buf *get_url_contents(QSP_ARG_DECL  const char *url)
{
	WARN("get_url_contents:  program not compiled with http support!?");
	return NULL;
}

void write_file_from_url( QSP_ARG_DECL  FILE *fp_out, const char *url )
{
	WARN("write_file_from_url:  program not compiled with http support!?");
}

void init_http_subsystem(SINGLE_QSP_ARG_DECL)
{
	WARN("init_http_subsystem:  program not compiled with http support!?");
}

#endif /* ! HAVE_LIBCURL */
