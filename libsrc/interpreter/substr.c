#include "quip_config.h"
#include <string.h>
#include "query_prot.h"

int is_a_substring(const char* s, const char* w)		/** true if str is substring of word */
{
	char *substr;

	// we need this here because we haven't initialized all the precision strings...
	if( w == NULL || s == NULL ) return(0);

	substr=strstr(w,s);
	if( substr == w ) return(1);
	return(0);
}

