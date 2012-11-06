#include "quip_config.h"

char VersionId_interpreter_filerd[] = QUIP_VERSION_STRING;

#undef _WINDLL /* Need stdin to be defined */

#include <string.h>

#include "query.h"
#include "filerd.h"
#include "debug.h"


void interp_file(QSP_ARG_DECL  const char *filename)
{
	FILE *fp;

	if( !strcmp(filename,"-") ){
		push_input_file(QSP_ARG "-");
		redir(QSP_ARG stdin);
	} else {
		fp=TRY_OPEN( filename,"r");
		if( fp ){
			push_input_file(QSP_ARG filename);
			redir(QSP_ARG fp);
		}
	}
}

void filerd(SINGLE_QSP_ARG_DECL)
{
	const char *s;

	s=nameof(QSP_ARG "input file");
	interp_file(QSP_ARG s);
}

void copycmd(SINGLE_QSP_ARG_DECL)
{
	FILE *fp;

	fp=TRYNICE( nameof(QSP_ARG  "transcript file"), "w" );
	if( fp ) {
		if(dupout(QSP_ARG fp)==(-1))
			fclose(fp);
	}
}

