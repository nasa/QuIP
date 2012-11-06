#include "quip_config.h"

char VersionId_vec_util_resample[] = QUIP_VERSION_STRING;

/* image warping */

/*
 * Two kinds of sample maps:
 *
 * Output source maps give a pair of coordinates for each output
 * pixel, which tell where to get the input data from.
 *
 * Input destination maps give a pair of coordinates for each input
 * pixel, which tell where to accumulate each sample
 */

#include "data_obj.h"
#include "vec_util.h"
#include <math.h>

/* local prototype */
static int resamp_check(QSP_ARG_DECL  Data_Obj *,Data_Obj *,Data_Obj *);

static int wrap_resample=1;

void set_resample_wrap(int flag)
{
	if( flag )
		advise("enabling wrap-around during resample ops");
	else
		advise("disabling wrap-around during resample ops");

	wrap_resample=flag;
}

static int resamp_check(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp)
{
	if( (dpto->dt_prec != PREC_SP ) || (dpfr->dt_prec != PREC_SP) ){
		WARN("source and destination must be float");
		return(-1);
	}
	if( dpto->dt_comps != dpfr->dt_comps ){
		sprintf(ERROR_STRING,"resamp_check:  type dimension mismatch between %s (%d) and %s (%d)",
				dpto->dt_name,dpto->dt_comps,dpfr->dt_name,dpfr->dt_comps);
		WARN(ERROR_STRING);
		return(-1);
	}
	if( (dpto->dt_rows != dpwarp->dt_rows) ||
		(dpto->dt_cols != dpwarp->dt_cols)){
sprintf(ERROR_STRING,"target %s, %d rows by %d cols",
dpto->dt_name,dpto->dt_rows,dpto->dt_cols);
advise(ERROR_STRING);
sprintf(ERROR_STRING,"map %s, %d rows by %d cols",
dpwarp->dt_name,dpwarp->dt_rows,dpwarp->dt_cols);
advise(ERROR_STRING);
		WARN("size mismatch between target and resample map");
		return(-1);
	}
	if( (MACHINE_PREC(dpwarp) != PREC_SP) ||
		(dpwarp->dt_comps != 2) ){
		sprintf(ERROR_STRING,"warp control image %s must be float, complex",
			dpwarp->dt_name);
		WARN(ERROR_STRING);
		return(-1);
	}
	return(0);
}

void resample(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp)
{
	float x_fr,y_fr;
	u_long ii,jj;
	float *ptrto, *ptrfr, *wp;
	u_long i,j;
	u_long out_rows, out_cols;
	u_long src_rows, src_cols;
	incr_t src_rowinc, src_pinc;

	if( resamp_check(QSP_ARG  dpto,dpfr,dpwarp) < 0 ) return;

	out_rows=dpto->dt_rows;
	out_cols=dpto->dt_cols;
	src_rows=dpfr->dt_rows;
	src_cols=dpfr->dt_cols;
	src_rowinc=dpfr->dt_rowinc;
	src_pinc=dpfr->dt_pinc;

	wp = (float *)dpwarp->dt_data;
	ptrto = (float *)dpto->dt_data;
	ptrfr = (float *)dpfr->dt_data;
	for(i=0;i<out_rows;i++){
		for(j=0;j<out_cols;j++){

			x_fr = ((float)j) + *wp++;
			while( x_fr < 0.0 )
				x_fr += (float)src_cols;
			while( x_fr >= (float) src_cols )
				x_fr -= (float) src_cols;
			jj = (u_long)x_fr; /* truncate fraction */

			y_fr = ((float)i) + *wp++;
			while( y_fr < 0.0 )
				y_fr += (float) src_rows;
			while( y_fr >= (float) src_rows )
				y_fr -= (float) src_rows;
			ii = (u_long)y_fr;

			/* no interpolation at this time */
			/* BUG does src have to be contiguous? no check... */
			*ptrto++ = *(ptrfr + ii*src_rowinc + jj);
		}
	}
}

#define PUT_IN_RANGE( coord, limit )				\
								\
			if( coord < 0.0 ){			\
				do {				\
				coord += (float)limit;		\
				} while ( coord < 0.0 );	\
			}					\
								\
			if ( coord >= (float)limit ){		\
				do {				\
					coord -= (float)limit;	\
				} while ( coord >= (float)limit );\
			}

#define CHECK_RANGE	if( wrap_resample ){					\
				PUT_IN_RANGE(x_fr,src_cols)			\
				PUT_IN_RANGE(y_fr,src_rows)			\
				map_sample=1;					\
			} else {						\
				if( x_fr < 0.0 || x_fr >= (float)src_cols ||	\
					y_fr < 0.0 || y_fr >= (float)src_rows ){\
					map_sample=0;				\
				} else {					\
					map_sample=1;				\
				}						\
			}

#define MAP_PIXEL	jj = x_fr; /* truncate fraction */				\
			ii = y_fr;							\
											\
			dx=x_fr-jj;							\
			dy=y_fr-ii;							\
			dxy = dx*dy;							\
											\
			ii2=ii+1;							\
			if( ii2 == src_rows ) ii2=0;					\
			ii *= src_rowinc;						\
			ii2 *= src_rowinc;						\
			jj2=jj+1;							\
			if( jj2 == src_cols ) jj2=0;					\
			jj *= src_pinc;							\
			jj2 *= src_pinc;						\
											\
			for(c=0;c<src_comps;c++){					\
	*(ptrto+c*dst_cinc) = (*(ptrfr + c*src_cinc + ii +  jj  )) * (1-dx-dy+dxy)	\
		 + (*(ptrfr + c*src_cinc + ii +  jj2 )) * (dx -dxy)			\
		 + (*(ptrfr + c*src_cinc + ii2 + jj  )) * (dy - dxy)			\
		 + (*(ptrfr + c*src_cinc + ii2 + jj2 )) * dxy;				\
		 	}

/* this old bilinear warp uses the warp map for perturbations */

void bilinear_warp(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp)
{
	float x_fr,y_fr;
	register dimension_t ii,jj,ii2,jj2;
	float *ptrto, *ptrfr, *wp;
	dimension_t i,j;
	dimension_t c;
	dimension_t out_rows, out_cols;
	dimension_t src_rows, src_cols;
	dimension_t src_comps;
	incr_t src_rowinc, src_pinc;
	incr_t dst_cinc, src_cinc;
	float dx,dy,dxy;
	int map_sample;

	if( resamp_check(QSP_ARG  dpto,dpfr,dpwarp) < 0 ) return;

	out_rows=dpto->dt_rows;
	out_cols=dpto->dt_cols;
	src_rows=dpfr->dt_rows;
	src_cols=dpfr->dt_cols;
	src_comps=dpfr->dt_comps;
	src_rowinc=dpfr->dt_rowinc;
	src_pinc=dpfr->dt_pinc;

	dst_cinc = dpto->dt_cinc;
	src_cinc = dpfr->dt_cinc;

	ptrto = (float *)dpto->dt_data;
	ptrfr = (float *)dpfr->dt_data;
	for(i=0;i<out_rows;i++){
		wp = (float *)dpwarp->dt_data + i * dpwarp->dt_rinc;
		for(j=0;j<out_cols;j++){

			x_fr = (float)j + *wp;
			y_fr = (float)i + *(wp+dpwarp->dt_cinc);
			wp += dpwarp->dt_pinc;

			CHECK_RANGE

			if( map_sample ){
				MAP_PIXEL
			}
			ptrto++;
		}
	}
}

/* this new bilinear warp uses the warp map for coords */

void new_bilinear_warp(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp)
{
	float x_fr,y_fr;
	register dimension_t ii,jj,ii2,jj2;
	float *ptrto, *ptrfr, *wp;
	dimension_t i,j;
	dimension_t c;
	dimension_t out_rows, out_cols;
	dimension_t src_rows, src_cols;
	dimension_t src_comps;
	incr_t src_rowinc, src_pinc;
	incr_t dst_cinc, src_cinc;
	float dx,dy,dxy;
	int map_sample;

	if( resamp_check(QSP_ARG  dpto,dpfr,dpwarp) < 0 ) return;


	out_rows=dpto->dt_rows;
	out_cols=dpto->dt_cols;
	src_rows=dpfr->dt_rows;
	src_cols=dpfr->dt_cols;
	src_comps=dpfr->dt_comps;
	src_rowinc=dpfr->dt_rowinc;
	src_pinc=dpfr->dt_pinc;

	dst_cinc = dpto->dt_cinc;
	src_cinc = dpfr->dt_cinc;

	ptrfr = (float *)dpfr->dt_data;
	for(i=0;i<out_rows;i++){
		ptrto = (float *)dpto->dt_data;
		ptrto += i*dpto->dt_rowinc;
		wp = (float *)dpwarp->dt_data + i * dpwarp->dt_rinc;
		for(j=0;j<out_cols;j++){

			x_fr = *wp;
			y_fr = *(wp+dpwarp->dt_cinc);
			wp += dpwarp->dt_pinc;

			CHECK_RANGE

			if( map_sample ){
				MAP_PIXEL
			}
			ptrto += dpto->dt_pinc;
		}
	}
}

