#include "quip_config.h"

char VersionId_interpreter_trynice[] = QUIP_VERSION_STRING;

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "query.h"

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	// needed on MAC
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

static int cautious=1;

COMMAND_FUNC( togclobber )
{
	if( cautious ){
		cautious=0;
		WARN("allowing file overwrites without explicit confirmation");
	} else {
		cautious=1;
		WARN("requiring confirmation for file overwrites");
	}
}

FILE *trynice(QSP_ARG_DECL  const char *fnam, const char *mode)
{
        FILE *fp;
        char pstr[128];
	struct stat statb;

	if( fnam[0]==0 ){
		sprintf(ERROR_STRING,"null file name");
		WARN(ERROR_STRING);
		return(NULL);
	}
	if( !strcmp(mode,"w") ){
		if( cautious && stat(fnam,&statb) != (-1) ){	/* file found */
                       	sprintf(pstr, "file %s exists, overwrite",fnam);
                       	if( !confirm(QSP_ARG pstr) ) return(NULL);
                }
        }
        fp=TRY_OPEN(fnam, mode);
	return(fp);
}

