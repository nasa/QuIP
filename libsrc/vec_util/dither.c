#include "quip_config.h"

char VersionId_vec_util_dither[] = QUIP_VERSION_STRING;

/* construct ordered dither matrix  */
#include <stdio.h>
#include "vec_util.h"

static int matrix[2][2]={{0,2},{3,1}};

void odither(Data_Obj *dp,int size)	/* ordered dither matrix order n */
{
	dimension_t x,y;
	int i;
	int factor;
	float *buf;
	int logsiz;

	if( dp->dt_prec != PREC_SP ){
		NWARN("target image must be float precision");
		return;
	}
	logsiz=0;
	factor=1;
	while( size != 1 ){
		if( size & 1 ){
			NWARN("size must be a power of 2");
			return;
		}
		size >>= 1;
		logsiz++;
		factor *= 4;
	}
	factor/=4;

	for(y=0;y<dp->dt_rows;y++){
		buf = (float *) dp->dt_data;
		buf += y * dp->dt_rowinc;
		for(x=0;x<dp->dt_cols;x++){
			*buf=0.0;
			for(i=0;i<logsiz;i++)
				*buf += ((factor>>(i*2)) *
					matrix[(x>>i)&1][(y>>i)&1]);
			buf += dp->dt_pinc;
		}
	}
}

