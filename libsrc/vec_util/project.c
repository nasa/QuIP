#include "quip_config.h"

char VersionId_vec_util_project[] = QUIP_VERSION_STRING;

#include "vec_util.h"
#include "vecgen.h"

/*
 * Project onto an object of lower dimension
 */

void war_project(dp_to,dp_fr)
Data_Obj *dp_to, *dp_fr;
{
	float *srcp, *dstp;
	int i;
	Vec_Args args;

	/* make sure precisions are good... */

	if( ! IS_IMAGE(dp_fr) ){
		warn("Sorry, can only project from images");
		return;
	}
	if( dp_fr->dt_prec != PREC_SP ){
		warn("Sorry, can only project float images");
		return;
	}
	/* check precision & type dimension */
	if( !dp_same_pixel_type(dp_to,dp_fr) ) return;

	if( IS_COMPLEX(dp_fr) )	args.arg_argstype = COMPLEX_ARGS;
	else			args.arg_argstype = REAL_ARGS;

	args.arg_prec1 = MACHINE_PREC(dp_fr);	/* include complex bit? */

	/* now figure out which dimension to project */

	if( dp_to->dt_rows == 1 ){	/* project columns */
		if( dp_to->dt_cols != dp_fr->dt_cols ){
			sprintf(error_string,
"Objects %s and %s must have the same number of columns to project",
				dp_fr->dt_name,dp_to->dt_name);
			warn(error_string);
			return;
		}
		srcp=dp_fr->dt_data;
		dstp=dp_to->dt_data;

		args.arg_n1 = dp_fr->dt_rows;
		args.arg_inc1 = dp_fr->dt_rinc;

		for(i=0;i<dp_to->dt_cols;i++){
			/* add up each column */
			args.arg_scalar1 = dstp;
			args.arg_v1 = srcp;
			if( for_real )
				vsum(&args);
			srcp += dp_fr->dt_pinc;
			dstp += dp_to->dt_pinc;
		}
	} else if( dp_to->dt_cols == 1 ){
		if( dp_to->dt_rows != dp_fr->dt_rows ){
			sprintf(error_string,
"Objects %s and %s must have the same number of rows to project",
				dp_fr->dt_name,dp_to->dt_name);
			warn(error_string);
			return;
		}
		srcp=dp_fr->dt_data;
		dstp=dp_to->dt_data;

		args.arg_n1 = dp_fr->dt_cols;
		args.arg_inc1 = dp_fr->dt_pinc;

		for(i=0;i<dp_to->dt_rows;i++){
			/* add up each column */
			args.arg_scalar1 = dstp;
			args.arg_v1 = srcp;
			if( for_real )
				vsum(&args);
			srcp += dp_fr->dt_rinc;
			dstp += dp_to->dt_rinc;
		}
	} else {
		warn("Sorry, don't know how to project anything except rows and columns");
	}
}

