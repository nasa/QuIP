#include "quip_config.h"

/* construct ordered dither matrix  */
#include <stdio.h>
#include "vec_util.h"
#include "quip_prot.h"

static int matrix[2][2]={{0,2},{3,1}};

void odither(QSP_ARG_DECL  Data_Obj *dp,int size)/* ordered dither matrix order n */
{
	dimension_t x,y;
	int i;
	int factor;
	float *buf;
	int logsiz;

	INSIST_RAM_OBJ(dp,odither)

	if( OBJ_PREC(dp) != PREC_SP ){
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

	for(y=0;y<OBJ_ROWS(dp);y++){
		buf = (float *) OBJ_DATA_PTR(dp);
		buf += y * OBJ_ROW_INC(dp);
		for(x=0;x<OBJ_COLS(dp);x++){
			*buf=0.0;
			for(i=0;i<logsiz;i++)
				*buf += ((factor>>(i*2)) *
					matrix[(x>>i)&1][(y>>i)&1]);
			buf += OBJ_PXL_INC(dp);
		}
	}
}

