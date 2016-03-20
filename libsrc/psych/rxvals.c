#include "quip_config.h"

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
		sprintf(ERROR_STRING,"reading x values from file %s",fnam);
		advise(ERROR_STRING);
	} else {
		WARN("Must specify a valid file for xvalues before running!!!");
		_nvals = -1 ;
		return;
	}

	_nvals=0;
	while( _nvals < MAX_X_VALUES && fscanf(fp,"%f",&xval_array[_nvals]) == 1 )
		_nvals++;
	if( _nvals == MAX_X_VALUES ){
		if( fscanf(fp,"%f",&ftmp) == 1 )
			WARN("warning: extra x values");
	}
	sprintf(str,"%d",_nvals);
	// BUG should be a reserved var!
	ASSIGN_VAR("nxvals",str);

	fclose(fp);
}


