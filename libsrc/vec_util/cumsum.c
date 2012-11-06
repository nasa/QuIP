#include "quip_config.h"

char VersionId_vec_util_cumsum[] = QUIP_VERSION_STRING;

#include "data_obj.h"
#include "vec_util.h"


/*
 * Compute the cumulative sum.
 * For the present, this is only defined on row vectors...
 */

void war_cumsum(QSP_ARG_DECL  Data_Obj *dp_to,Data_Obj *dp_fr)
{
	int i;

	if( dp_fr->dt_prec != PREC_SP && dp_fr->dt_prec != PREC_DP ){
		NWARN("Sorry, can only accumulate float or double precision vectors");
		return;
	}
	/* check precision & type dimension */
	if( !dp_same_pixel_type(QSP_ARG  dp_to,dp_fr,"war_cumsum") ) return;
	if( ! dp_same_size(QSP_ARG  dp_to,dp_fr,"war_cumsum") ) return;

	/*
	if( ! IS_ROWVEC(dp_to) ){
		NWARN("Sorry, accumulation operation only defined for row vectors");
		return;
	}
	if( dp_to->dt_tdim != 1 ){
		NWARN("Sorry, can only accumulate vectors with 1 component/pixel");
		return;
	}
	*/
	if( (! IS_CONTIGUOUS(dp_to)) || ! IS_CONTIGUOUS(dp_fr) ){
		NWARN("Sorry, accumulation operation requires contiguous source and destination objects");
		return;
	}

	if( dp_fr->dt_prec == PREC_SP ){
		float *srcp, *dstp, *last_dstp;

		srcp=(float *)dp_fr->dt_data;
		dstp=(float *)dp_to->dt_data;

		*dstp = *srcp;		/* copy the first element */
		i=dp_to->dt_n_type_elts-1;	/* do nothing to the first element */ 
		while(i--){
			last_dstp = dstp;
			dstp ++;
			srcp ++;
			*dstp = *last_dstp + *srcp;
		}
	} else if( dp_fr->dt_prec == PREC_DP ){
		double *srcp, *dstp, *last_dstp;

		srcp=(double *)dp_fr->dt_data;
		dstp=(double *)dp_to->dt_data;

		*dstp = *srcp;		/* copy the first element */
		i=dp_to->dt_n_type_elts-1;	/* do nothing to the first element */ 
		while(i--){
			last_dstp = dstp;
			dstp ++;
			srcp ++;
			*dstp = *last_dstp + *srcp;
		}
	}
}
