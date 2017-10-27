#include "quip_config.h"

#include <stdio.h>
#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "quip_prot.h"
#include "qlevel.h"
#include "ctone.h"
#include "vec_util.h"	/* spread_debug - BUG should be moved elsewhere! */

static float err_dist[MAX_QUANT_LEVELS];
//float quant_level[MAX_QUANT_LEVELS];
Dither_Params dp1={
	.dp_n_columns = 	0,
	.dp_n_rows = 	0,
	.dp_desired = 	{NULL,NULL,NULL}
};


/* get the best level for phosphor: minimize absval of difference */
float lum_weight=1.0;
float rg_weight=1.0;
float by_weight=1.0;

void showvec(float *p)
{
sprintf(DEFAULT_ERROR_STRING,"vector:  %g %g %g",p[0],p[1],p[2]);
NADVISE(DEFAULT_ERROR_STRING);
}

/* red in most sig. bits, then green, then blue */

int getbest(QSP_ARG_DECL  int col_index)
{
	int red_level,grn_level,blu_level;
	int n,best;
	float least_err;
	float vec[3];

	n=0;
	for(red_level=0;red_level<nlevels;red_level++){
		for(grn_level=0;grn_level<nlevels;grn_level++){
			for(blu_level=0;blu_level<nlevels;blu_level++){
/*
sprintf(DEFAULT_ERROR_STRING,"getbest:  desired = %g %g %g",
desired[0][col_index],desired[1][col_index],desired[2][col_index]);
advise(DEFAULT_ERROR_STRING);
*/
				vec[0]=desired[0][col_index]-quant_level[red_level];
				vec[1]=desired[1][col_index]-quant_level[grn_level];
				vec[2]=desired[2][col_index]-quant_level[blu_level];
				rgb2o(QSP_ARG  vec);
				vec[0] *= rg_weight;
				vec[1] *= by_weight;
				vec[2] *= lum_weight;
				err_dist[n] =	  vec[0]*vec[0]
						+ vec[1]*vec[1]
						+ vec[2]*vec[2];
				n++;
			}
		}
	}
	least_err=(float) 1000000000000.0;	// BUG use scientific notation
	best=(-1);
	n=0;
	for(red_level=0;red_level<nlevels;red_level++){
		for(grn_level=0;grn_level<nlevels;grn_level++){
			for(blu_level=0;blu_level<nlevels;blu_level++){
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"getbest %d %d %d:  err_dist[%d] = %g",red_level,grn_level,blu_level,n,err_dist[n]);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
				if( err_dist[n] < least_err ){
					least_err=err_dist[n];
					thebest[0]=red_level;
					thebest[1]=grn_level;
					thebest[2]=blu_level;
					best=n;
				}
				n++;
			}
		}
	}
	if( best== -1 ) error1("error too big!!!");
#ifdef QUIP_DEBUG
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"getbest RETURNING %d",best);
advise(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	return(best);
}

