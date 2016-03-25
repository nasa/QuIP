#ifndef _MY_CURL_H_

#ifdef HAVE_LIBCURL

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

typedef struct curl_info {
	CURL *				ci_easy_handle;
	String_Buf *			ci_sbp;
	char				ci_error_buffer[1024];
	curl_version_info_data *	ci_vdp;
} Curl_Info;

#endif // HAVE_LIBCURL

#define _MY_CURL_H_
#endif // ! _MY_CURL_H_
