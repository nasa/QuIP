#include "quip_config.h"

char VersionId_interpreter_howmuch[] = QUIP_VERSION_STRING;



#include <ctype.h>

#include "query.h"
#include "nexpr.h"

double how_much(QSP_ARG_DECL  const char* s)        /**/
{
	const char *estr;

	estr=nameof(QSP_ARG  s);
	return( pexpr(QSP_ARG  estr) );
}

