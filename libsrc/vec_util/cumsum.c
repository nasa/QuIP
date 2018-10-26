#include "quip_config.h"

#include "quip_prot.h"
#include "data_obj.h"
#include "vec_util.h"


/*
 * Compute the cumulative sum.
 * For the present, this is only defined on row vectors...
 */

void _war_cumsum(QSP_ARG_DECL  Data_Obj *dp_to,Data_Obj *dp_fr)
{
	int i;

	INSIST_RAM_OBJ(dp_to,war_cumsum);
	INSIST_RAM_OBJ(dp_fr,war_cumsum);

	if( OBJ_PREC(dp_fr) != PREC_SP && OBJ_PREC(dp_fr) != PREC_DP ){
		warn("Sorry, can only accumulate float or double precision vectors");
		return;
	}
	/* check precision & type dimension */
	if( !dp_same_pixel_type(dp_to,dp_fr,"war_cumsum") ) return;
	if( ! dp_same_size(dp_to,dp_fr,"war_cumsum") ) return;

	/*
	if( ! IS_ROWVEC(dp_to) ){
		warn("Sorry, accumulation operation only defined for row vectors");
		return;
	}
	if( OBJ_COMPS(dp_to) != 1 ){
		warn("Sorry, can only accumulate vectors with 1 component/pixel");
		return;
	}
	*/
	if( (! IS_CONTIGUOUS(dp_to)) || ! IS_CONTIGUOUS(dp_fr) ){
		warn("Sorry, accumulation operation requires contiguous source and destination objects");
		return;
	}

	if( OBJ_PREC(dp_fr) == PREC_SP ){
		float *srcp, *dstp, *last_dstp;

		srcp=(float *)OBJ_DATA_PTR(dp_fr);
		dstp=(float *)OBJ_DATA_PTR(dp_to);

		*dstp = *srcp;		/* copy the first element */
		i=OBJ_N_TYPE_ELTS(dp_to)-1;	/* do nothing to the first element */ 
		while(i--){
			last_dstp = dstp;
			dstp ++;
			srcp ++;
			*dstp = *last_dstp + *srcp;
		}
	} else if( OBJ_PREC(dp_fr) == PREC_DP ){
		double *srcp, *dstp, *last_dstp;

		srcp=(double *)OBJ_DATA_PTR(dp_fr);
		dstp=(double *)OBJ_DATA_PTR(dp_to);

		*dstp = *srcp;		/* copy the first element */
		i=OBJ_N_TYPE_ELTS(dp_to)-1;	/* do nothing to the first element */ 
		while(i--){
			last_dstp = dstp;
			dstp ++;
			srcp ++;
			*dstp = *last_dstp + *srcp;
		}
	}
}

