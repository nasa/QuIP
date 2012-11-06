#include "quip_config.h"
char VersionId_qutil_substr[] = QUIP_VERSION_STRING;

#include <string.h>
#include "substr.h"

int is_a_substring(const char* s, const char* w)		/** true if str is substring of word */
{
	char *substr;

	substr=strstr(w,s);
	if( substr == w ) return(1);
	return(0);
}

