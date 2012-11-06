#include "quip_config.h"

char VersionId_vec_util_radavg[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <math.h>

#include "vec_util.h"
#include "data_obj.h"

/*
 * This version expects a "wrapped" spectrum...
 */

int rad_avg(QSP_ARG_DECL  Data_Obj *mean_dp,Data_Obj *var_dp,Data_Obj *count_dp,Data_Obj *img_dp)
{
	dimension_t i, j;
	/* dimension_t */ float x0, y0;
	uint32_t h;
	uint32_t nbins;
	float distx, disty;
	float f;
	float *mean_ptr, *var_ptr, *count_ptr, *img_ptr;
	double rad,maxrad;

	if( (!IS_VECTOR(mean_dp)) || (!IS_VECTOR(var_dp)) ){
		WARN("data objects for mean, variance and counts must be vectors");
		return(-1);
	}
	if( mean_dp->dt_prec != PREC_SP || var_dp->dt_prec != PREC_SP ||
		count_dp->dt_prec != PREC_SP || img_dp->dt_prec != PREC_SP ){
		WARN("object precisions must be float");
		return(-1);
	}
	if( mean_dp->dt_cols != var_dp->dt_cols ||
		mean_dp->dt_cols != count_dp->dt_cols ){
		WARN("all vectors must have same dimension");
		return(-1);
	}
	if( mean_dp->dt_pinc != 1 || var_dp->dt_pinc != 1 || count_dp->dt_pinc != 1 ){
		WARN("result vectors must be contiguous");
		return(-1);
	}

	nbins = mean_dp->dt_cols;

	mean_ptr = (float *) mean_dp->dt_data;
	var_ptr = (float *) var_dp->dt_data;
	count_ptr = (float *) count_dp->dt_data;

	x0 = img_dp->dt_cols/2;
	y0 = img_dp->dt_rows/2;

	/* pi=4.0*atan(1.0); */


	for(i=0;i<nbins;i++){
		mean_ptr[i]=var_ptr[i]=0.0;
		count_ptr[i]=0.0;
	}

	maxrad = sqrt( ((float)x0)*((float)x0) + ((float)y0)*((float)y0) );

	for(i=0; i<img_dp->dt_rows; i++) {
		img_ptr = ((float *) img_dp->dt_data) + i * img_dp->dt_rowinc;
		for(j=0; j< img_dp->dt_cols; j++) {
			distx = (float)(j - x0) ;
			disty = (float)(i - y0) ;

			rad = sqrt( distx*distx + disty*disty ) / maxrad;
/*
sprintf(error_string,"maxrad = %g, distx = %g, disty = %g, rad = %g",maxrad,distx,disty,rad);
advise(error_string);
*/
			h = floor(0.5 + (((double)(nbins-1))*rad));
/*
sprintf(error_string,"rad = %g, nbins = %ld, h = %d",rad,nbins,h);
advise(error_string);
*/
						/* 0 <= h < nbins */
#ifdef CAUTIOUS
			if( h < 0 ){
				sprintf(error_string,"h (%d) was neg! (nbins = %d, i=%d, j = %d, distx=%g, idsty=%g)",h,nbins,i,j,distx,disty);
				WARN(error_string);
				h=nbins-1;
			} else if( h > (nbins-1) ){
				h=0;
				sprintf(error_string,"h was too big!");
				WARN(error_string);
			}
#endif /* CAUTIOUS */

			f = *img_ptr;
			mean_ptr[h] += f;
			var_ptr[h] += f*f;
			count_ptr[h]++;

			img_ptr += img_dp->dt_pinc;
		}
	}
	for(h=0; h < nbins; h++) {
		if( count_ptr[h] != 0 ){
			mean_ptr[h] /= count_ptr[h];
			var_ptr[h] /= count_ptr[h];
		}
		var_ptr[h] -= mean_ptr[h]*mean_ptr[h];

		if(count_ptr[h] > 1)
			var_ptr[h] *= count_ptr[h]/(count_ptr[h] - 1.);
		else
			var_ptr[h] = 0.;

	}
	return(0);
}

/* This version averages in radial strips */

int ori_avg(QSP_ARG_DECL  Data_Obj *mean_dp,Data_Obj *var_dp,Data_Obj *count_dp,Data_Obj *img_dp)
{
	dimension_t i, j;
	/* dimension_t */ float x0, y0;
	u_long h;
	u_long nbins;
	float distx, disty;
	float f;
	float *mean_ptr, *var_ptr, *count_ptr, *img_ptr;
	double ang,pi,pi_over_two;

	if( (!IS_VECTOR(mean_dp)) || (!IS_VECTOR(var_dp)) ){
		WARN("data objects for mean, variance and counts must be vectors");
		return(-1);
	}
	if( mean_dp->dt_prec != PREC_SP || var_dp->dt_prec != PREC_SP ||
		count_dp->dt_prec != PREC_SP || img_dp->dt_prec != PREC_SP ){
		WARN("object precisions must be float");
		return(-1);
	}
	if( mean_dp->dt_cols != var_dp->dt_cols ||
		mean_dp->dt_cols != count_dp->dt_cols ){
		WARN("all vectors must have same dimension");
		return(-1);
	}
	if( mean_dp->dt_pinc != 1 || var_dp->dt_pinc != 1 || count_dp->dt_pinc != 1 ){
		WARN("result vectors must be contiguous");
		return(-1);
	}

	nbins = mean_dp->dt_cols;

	mean_ptr = (float *) mean_dp->dt_data;
	var_ptr = (float *) var_dp->dt_data;
	count_ptr = (float *) count_dp->dt_data;

	x0 = img_dp->dt_cols/2;
	y0 = img_dp->dt_rows/2;

	/* pi=4.0*atan(1.0); */


	for(i=0;i<nbins;i++){
		mean_ptr[i]=var_ptr[i]=0.0;
		count_ptr[i]=0.0;
	}

	pi = 4*atan(1.0);
	pi_over_two = 2*atan(1.0);

	for(i=0; i<img_dp->dt_rows; i++) {
		img_ptr = ((float *) img_dp->dt_data) + i * img_dp->dt_rowinc;
		for(j=0; j< img_dp->dt_cols; j++) {
			distx = (float)(j - x0) ;
			disty = (float)(i - y0) ;

			if( distx == 0 && disty == 0 ) continue;

			/* is this always positive or not?? */
			ang = atan( disty/distx ) + pi_over_two;
/*
sprintf(error_string,"ang = %g, distx = %g, disty = %g",ang,distx,disty);
advise(error_string);
*/
			h = floor(0.5 + (((double)(nbins-1))*ang/pi));
/*
sprintf(error_string,"rad = %g, nbins = %ld, h = %d",rad,nbins,h);
advise(error_string);
*/
						/* 0 <= h < nbins */
#ifdef CAUTIOUS
			/* Now h is unsigned, don't have to check against 0 */
			if( h > (nbins-1) ){
				h=0;
				sprintf(error_string,"CAUTIOUS:  h (%ld) was too big!",h);
				WARN(error_string);
			}
#endif /* CAUTIOUS */

			f = *img_ptr;
			mean_ptr[h] += f;
			var_ptr[h] += f*f;
			count_ptr[h]++;

			img_ptr += img_dp->dt_pinc;
		}
	}
	for(h=0; h < nbins; h++) {
		if( count_ptr[h] != 0 ){
			mean_ptr[h] /= count_ptr[h];
			var_ptr[h] /= count_ptr[h];
		}
		var_ptr[h] -= mean_ptr[h]*mean_ptr[h];

		if(count_ptr[h] > 1)
			var_ptr[h] *= count_ptr[h]/(count_ptr[h] - 1.);
		else
			var_ptr[h] = 0.;

	}
	return(0);
}
