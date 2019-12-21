#include "quip_config.h"

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

#include <math.h>
#include "quip_prot.h"
#include "vec_util.h"

// BUG global var not thread-safe
static int wrap_resample=0;

void set_resample_wrap(int flag)
{
	if( flag )
		NADVISE("enabling wrap-around during resample ops");
	else
		NADVISE("disabling wrap-around during resample ops");

	wrap_resample=flag;
}

#define resamp_check(dpto,dpfr,dpwarp) _resamp_check(QSP_ARG  dpto,dpfr,dpwarp)

static int _resamp_check(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp)
{
	VINSIST_RAM_OBJ(dpto,resample,-1)
	VINSIST_RAM_OBJ(dpfr,resample,-1)
	VINSIST_RAM_OBJ(dpwarp,resample,-1)

	if( (OBJ_PREC(dpto) != PREC_SP ) || (OBJ_PREC(dpfr) != PREC_SP) ){
		WARN("source and destination must be float");
		return(-1);
	}
	if( OBJ_COMPS(dpto) != OBJ_COMPS(dpfr) ){
		sprintf(ERROR_STRING,"resamp_check:  type dimension mismatch between %s (%d) and %s (%d)",
				OBJ_NAME(dpto),OBJ_COMPS(dpto),OBJ_NAME(dpfr),OBJ_COMPS(dpfr));
		WARN(ERROR_STRING);
		return(-1);
	}
	if( (OBJ_ROWS(dpto) != OBJ_ROWS(dpwarp)) ||
		(OBJ_COLS(dpto) != OBJ_COLS(dpwarp))){
sprintf(ERROR_STRING,"target %s, %d rows by %d cols",
OBJ_NAME(dpto),OBJ_ROWS(dpto),OBJ_COLS(dpto));
advise(ERROR_STRING);
sprintf(ERROR_STRING,"map %s, %d rows by %d cols",
OBJ_NAME(dpwarp),OBJ_ROWS(dpwarp),OBJ_COLS(dpwarp));
advise(ERROR_STRING);
		WARN("size mismatch between target and resample map");
		return(-1);
	}
	// We allow 1-component complex or two-component float
	if( (!(OBJ_PREC(dpwarp) == PREC_CPX && OBJ_COMPS(dpwarp)==1)) &&
	    (!(OBJ_PREC(dpwarp) == PREC_SP  && OBJ_COMPS(dpwarp)==2)) ){
		sprintf(ERROR_STRING,
"warp control image %s (%ld component %s) must be complex or 2-component float",
			OBJ_NAME(dpwarp),(long)OBJ_COMPS(dpwarp),
			PREC_NAME(OBJ_PREC_PTR(dpwarp)));
		WARN(ERROR_STRING);
		return(-1);
	}
	return(0);
}

void _resample(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp)
{
	float x_fr,y_fr;
	u_long ii,jj;
	float *ptrto, *ptrfr, *wp;
	u_long i,j;
	u_long out_rows, out_cols;
	u_long src_rows, src_cols;
	incr_t src_rowinc;
    //incr_t src_pinc;

	if( resamp_check(dpto,dpfr,dpwarp) < 0 ) return;

	out_rows=OBJ_ROWS(dpto);
	out_cols=OBJ_COLS(dpto);
	src_rows=OBJ_ROWS(dpfr);
	src_cols=OBJ_COLS(dpfr);
	src_rowinc=OBJ_ROW_INC(dpfr);
	//src_pinc=OBJ_PXL_INC(dpfr);

	wp = (float *)OBJ_DATA_PTR(dpwarp);
	ptrto = (float *)OBJ_DATA_PTR(dpto);
	ptrfr = (float *)OBJ_DATA_PTR(dpfr);
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

#define CHECK_RANGE	if( wrap_resample ){				\
				PUT_IN_RANGE(x_fr,src_cols)		\
				PUT_IN_RANGE(y_fr,src_rows)		\
				map_sample=1;				\
			} else {					\
				if( x_fr < 0.0 || x_fr >= (float)src_cols ||	\
					y_fr < 0.0 || y_fr >= (float)src_rows ){\
					map_sample=0;			\
				} else {				\
					map_sample=1;			\
				}					\
			}

#define MAP_PIXEL	jj = (dimension_t) x_fr; /* truncate fraction */\
			ii = (dimension_t) y_fr;			\
									\
			dx=x_fr-jj;					\
			dy=y_fr-ii;					\
			dxy = dx*dy;					\
									\
			ii2=ii+1;					\
			if( ii2 == src_rows ) ii2=0;			\
			ii *= src_rowinc;				\
			ii2 *= src_rowinc;				\
			jj2=jj+1;					\
			if( jj2 == src_cols ) jj2=0;			\
			jj *= src_pinc;					\
			jj2 *= src_pinc;				\
									\
			for(c=0;c<src_comps;c++){			\
	*(ptrto+c*dst_cinc) = (*(ptrfr + c*src_cinc + ii +  jj  )) * (1-dx-dy+dxy)	\
		 + (*(ptrfr + c*src_cinc + ii +  jj2 )) * (dx -dxy)	\
		 + (*(ptrfr + c*src_cinc + ii2 + jj  )) * (dy - dxy)	\
		 + (*(ptrfr + c*src_cinc + ii2 + jj2 )) * dxy;		\
		 	}

/* this old bilinear warp uses the warp map for perturbations */

void _bilinear_warp(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp)
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

	if( resamp_check(dpto,dpfr,dpwarp) < 0 ) return;

	out_rows=OBJ_ROWS(dpto);
	out_cols=OBJ_COLS(dpto);
	src_rows=OBJ_ROWS(dpfr);
	src_cols=OBJ_COLS(dpfr);
	src_comps=OBJ_COMPS(dpfr);
	src_rowinc=OBJ_ROW_INC(dpfr);
	src_pinc=OBJ_PXL_INC(dpfr);

	dst_cinc = OBJ_COMP_INC(dpto);
	src_cinc = OBJ_COMP_INC(dpfr);

	ptrto = (float *)OBJ_DATA_PTR(dpto);
	ptrfr = (float *)OBJ_DATA_PTR(dpfr);
	for(i=0;i<out_rows;i++){
		wp = (float *)OBJ_DATA_PTR(dpwarp) + i * OBJ_ROW_INC(dpwarp);
		for(j=0;j<out_cols;j++){

			x_fr = (float)j + *wp;
			y_fr = (float)i + *(wp+OBJ_COMP_INC(dpwarp));
			wp += OBJ_PXL_INC(dpwarp);

			CHECK_RANGE

			if( map_sample ){
				MAP_PIXEL
			}
			ptrto++;
		}
	}
}

/* this new bilinear warp uses the warp map for coords */

void _new_bilinear_warp(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,Data_Obj *dpwarp)
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

	if( resamp_check(dpto,dpfr,dpwarp) < 0 ) return;


	out_rows=OBJ_ROWS(dpto);
	out_cols=OBJ_COLS(dpto);
	src_rows=OBJ_ROWS(dpfr);
	src_cols=OBJ_COLS(dpfr);
	src_comps=OBJ_COMPS(dpfr);
	src_rowinc=OBJ_ROW_INC(dpfr);
	src_pinc=OBJ_PXL_INC(dpfr);

	dst_cinc = OBJ_COMP_INC(dpto);
	src_cinc = OBJ_COMP_INC(dpfr);

	ptrfr = (float *)OBJ_DATA_PTR(dpfr);
	for(i=0;i<out_rows;i++){
		ptrto = (float *)OBJ_DATA_PTR(dpto);
		ptrto += i*OBJ_ROW_INC(dpto);
		wp = (float *)OBJ_DATA_PTR(dpwarp) + i * OBJ_ROW_INC(dpwarp);
		for(j=0;j<out_cols;j++){

			x_fr = *wp;
			y_fr = *(wp+OBJ_COMP_INC(dpwarp));
			wp += OBJ_PXL_INC(dpwarp);

			CHECK_RANGE

			if( map_sample ){
				MAP_PIXEL
			}
			ptrto += OBJ_PXL_INC(dpto);
		}
	}
}

