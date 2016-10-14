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

#define _CURL_STRINGBUF(ci_p)		(ci_p)->ci_sbp
#define _CURL_EASY_HANDLE(ci_p)		(ci_p)->ci_easy_handle
#define _CURL_ERROR_BUFFER(ci_p)	(ci_p)->ci_error_buffer
#define _CURL_VERSION_INFO(ci_p)	(ci_p)->ci_vdp

#define CURL_STRINGBUF		_CURL_STRINGBUF(QS_CURL_INFO)
#define EASY_HANDLE		_CURL_EASY_HANDLE(QS_CURL_INFO)
//#define CURL_STRINGBUF	(THIS_QSP->qs_curl_info->ci_sbp)
#define MY_CURL_ERROR_BUFFER	_CURL_ERROR_BUFFER(QS_CURL_INFO)
#define CURL_VERSION_INFO	_CURL_VERSION_INFO(QS_CURL_INFO)

#endif // HAVE_LIBCURL

#define _MY_CURL_H_
#endif // ! _MY_CURL_H_
