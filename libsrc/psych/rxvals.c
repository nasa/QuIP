#include "quip_config.h"

#include <stdio.h>

#include "stc.h"

void rdxvals(QSP_ARG_DECL  const char *filename)
{
        FILE *fp;
	float ftmp;
	char str[32];
	int n;

	// It would make more sense to determine the number from
	// the file!
	if( insure_xval_array() < 0 ) return;

        fp=try_open(filename,"r");
	if(fp){
		sprintf(ERROR_STRING,"reading x values from file %s",filename);
		advise(ERROR_STRING);
	} else {
		warn("Must specify a valid file for xvalues before running!!!");
		return;
	}

	n=0;
	while( n < _nvals && fscanf(fp,"%f",&xval_array[n]) == 1 )
		n++;

	if( n == _nvals ){
		if( fscanf(fp,"%f",&ftmp) == 1 ){
			sprintf(ERROR_STRING,
				"File %s contains more than %d values!?",
				filename,_nvals);
			warn(ERROR_STRING);
		}
	}

	fclose(fp);

	if( n < _nvals ){
		sprintf(ERROR_STRING,"File %s contains %d values, expected %d!?",
			filename,n,_nvals);
		warn(ERROR_STRING);

		set_n_xvals(n);
		rdxvals(QSP_ARG  filename);
	} else {
		sprintf(str,"%d",n);
		assign_var("nxvals",str);
	}
}


