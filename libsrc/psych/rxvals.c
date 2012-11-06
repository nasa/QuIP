#include "quip_config.h"

char VersionId_psych_rxvals[] = QUIP_VERSION_STRING;

#include <stdio.h>

#include "stc.h"
#include "query.h"

void rdxvals(QSP_ARG_DECL  const char *fnam)
{
        FILE *fp;
	float ftmp;
	char str[32];

        fp=TRY_OPEN(fnam,"r");
	if(fp){
		sprintf(error_string,"reading x values from file %s",fnam);
		advise(error_string);
	} else {
		WARN("Must specify a valid file for xvalues before running!!!");
		_nvals = -1 ;
		return;
	}

	_nvals=0;
	while( _nvals < MAXVALS && fscanf(fp,"%f",&xval_array[_nvals]) == 1 )
		_nvals++;
	if( _nvals == MAXVALS ){
		if( fscanf(fp,"%f",&ftmp) == 1 )
			WARN("warning: extra x values");
	}
	sprintf(str,"%d",_nvals);
	ASSIGN_VAR("nxvals",str);

	fclose(fp);
}


