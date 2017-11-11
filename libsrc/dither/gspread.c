
#include "quip_config.h"

/*
 * utility routines for optimal halftoning
 */

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "quip_prot.h"
#include "vec_util.h"

/* get the filtered error at one point.
 * We sum the product of the filter impulse response with a subimage
 * of the input centered at x,y.
 */

double get_ferror(Data_Obj *edp,Data_Obj *fdp,dimension_t x,dimension_t y)
{
	incr_t i,j;
	dimension_t hx,hy;	/* half distances */
	double err;
	incr_t foffset, eoffset, yos;
	incr_t ix,iy;
	float *eptr, *fptr;

	/*
	 * the filter should have odd sizes;
	 * the center is at the middle of the image
	 */

	hy=OBJ_ROWS(fdp)/2;
	hx=OBJ_COLS(fdp)/2;
	/*
	x -= hx;
	y -= hy;
	*/
	err=0.0;
	eptr = (float *)OBJ_DATA_PTR(edp);
	fptr = (float *)OBJ_DATA_PTR(fdp);

	j=(incr_t)OBJ_ROWS(fdp);
	while(j--){
		iy=y+j-hy;
#ifdef NOWRAP
		if( iy < 0 ) continue;
		else if( iy >= OBJ_ROWS(edp) ) continue;
#else
		while( iy < 0 ) iy += OBJ_ROWS(edp);
		while( iy >= (incr_t)OBJ_ROWS(edp) ) iy -= OBJ_ROWS(edp);
#endif /* NOWRAP */

		yos = iy*(incr_t)OBJ_ROW_INC(edp);

		foffset = (j+1) * (incr_t)OBJ_ROW_INC(fdp);

		i=(incr_t)OBJ_COLS(fdp);
		while(i--){
			foffset--;

			ix=x+i-hx;
#ifdef NOWRAP
			if( ix < 0 ) continue;
			else if( ix >= OBJ_COLS(edp) ) continue;
#else
			while( ix < 0 ) ix += OBJ_COLS(edp);
			while( ix >= (incr_t)OBJ_COLS(edp) ) ix -= OBJ_COLS(edp);
#endif /* NOWRAP */
			eoffset = yos + ix;

			err += *(eptr+eoffset)
				* *(fptr+foffset);

#ifdef QUIP_DEBUG
/*
if( debug & spread_debug ){
sprintf(ERROR_STRING,"get_ferror %d %d:  %d %d, err = %g, filt = %g, running total %g",
x,y,i,j,*(eptr+eoffset),*(fptr+foffset),err);
advise(ERROR_STRING);
}
*/
#endif /* QUIP_DEBUG */
		}
	}
#ifdef QUIP_DEBUG
/*
if( debug & spread_debug ){
sprintf(ERROR_STRING,"get_ferror %d %d:  TOTAL err = %g",x,y,err);
advise(ERROR_STRING);
}
*/
#endif /* QUIP_DEBUG */

	return(err);
}

/* really the mean square, after we divide by the number of pixels! */

double get_sos(Data_Obj *edp,Data_Obj *fdp)		/* get the total sq'd error */
{
	incr_t i,j;
	double sos, rowsos,err;

	sos = 0.0;
	for(j=0;j<(incr_t)OBJ_ROWS(edp);j++){
		rowsos=0.0;
		for(i=0;i<(incr_t)OBJ_COLS(edp);i++){
			err = get_ferror(edp,fdp,i,j);
			rowsos += err * err;
		}
		sos += rowsos;
	}

	/* normalize by number of pixels */
	/* why rowinc and not ncols??? */
	sos /= OBJ_ROWS(edp)*OBJ_ROW_INC(edp);
	return(sos);
}

/* add_to_sos
 * This version recalculates the sum of squared filtered error, with an adjustment
 * of strength factor at x,y.  The filtered error is not stored, we call get_ferror
 * over the support of the filter.  (get_ferror does [inefficient] space domain convolution.)
 * factor is always 1 or -1, so we use this routine to exclude or include the effect of
 * a particular pixel.  Typically, we remove the effect of a pixel, then recalculate for a new
 * value.
 */

double _add_to_sos(QSP_ARG_DECL  dimension_t x,dimension_t y,Data_Obj *edp,Data_Obj *fdp,int factor)
{
	dimension_t i,j;
	double err,adj;
	incr_t xx,yy;

	/*
	if( the_sos == NO_VALUE )
		the_sos = get_sos(edp,fdp);
	*/

	adj =0.0;
	for(j=0;j<OBJ_ROWS(fdp);j++){
		yy = (incr_t)(y + j) - (incr_t)OBJ_ROWS(fdp)/2;
#ifdef NOWRAP
		if( yy >= 0 && yy < OBJ_ROWS(edp) ){
			for(i=0;i<OBJ_COLS(fdp);i++){
				xx = (incr_t)(x + i) - (incr_t)(OBJ_COLS(fdp)/2);
				if( xx >= 0 && xx < OBJ_COLS(edp) ){
					err = get_ferror(edp,fdp,xx,yy);
					adj += err*err;
				}
			}
		}
#else
		while( yy < 0 ) yy += OBJ_ROWS(edp);
		while( yy >= (incr_t)OBJ_ROWS(edp) ) yy -= OBJ_ROWS(edp);
		for(i=0;i<OBJ_COLS(fdp);i++){
			xx = x + i - OBJ_COLS(fdp)/2;
			while( xx < 0 ) xx += OBJ_COLS(edp);
			while( xx >= (incr_t)OBJ_COLS(edp) ) xx -= OBJ_COLS(edp);
			err = get_ferror(edp,fdp,xx,yy);
			adj += err*err;
		}
#endif /* NOWRAP */
	}
	/* normalize by number of pixels */
	if( factor == 1 )
		adj /= (OBJ_COLS(edp) * OBJ_ROWS(edp));
	else if( factor == -1 )
		adj /= - (OBJ_COLS(edp) * OBJ_ROWS(edp));
#ifdef CAUTIOUS
	else {
		sprintf(ERROR_STRING,"CAUTIOUS:  add_to_sos:  factor (%d) is not 1 or -1 !?",factor);
		warn(ERROR_STRING);
		return(0.0);
	}
#endif /* CAUTIOUS */
	/* the_sos += adj; */
	return(adj);
}
	
void _get_xy_scattered_point(QSP_ARG_DECL  dimension_t n,dimension_t xsize,dimension_t ysize,dimension_t *xp,dimension_t *yp)
{
	int x,y;
	/* size is (square) linear dimension */

	x = y = 0;
	/* n is the index in the scan of this point.
	 * For raster scan, x=n%cols, y=n/cols.
	 * We'd like to do the same thing, but on a bit-reversed n...
	 */

	/* This method does something, but doesn't work for non-square images... */

	if( xsize!=ysize ){
		sprintf(ERROR_STRING,"get_xy_scattered_point:  width (%d) and height (%d) should match!?",
			xsize,ysize);
		warn(ERROR_STRING);
		get_xy_random_point(n,xsize,ysize,xp,yp);
		return;
	}

	while( xsize > 1 ){
		x <<= 1;
		y <<= 1;
		switch( n & 3 ){
			case 0: x |= 1; break;
			case 1: y |= 1; break;
			case 2: x |= 1; y |= 1; break;
		}
		n >>= 2;
		xsize >>= 1;
	}

	*xp = x;
	*yp = y;
}

void _get_xy_raster_point(QSP_ARG_DECL  dimension_t n,dimension_t xsize,dimension_t ysize,dimension_t *xp,dimension_t *yp)
{
	*xp = n % xsize;
	*yp = n / xsize;
}

void _get_xy_random_point(QSP_ARG_DECL  dimension_t n,dimension_t xsize,dimension_t ysize,dimension_t *xp,dimension_t *yp)
{
	*xp = (dimension_t)(drand48() * (double)xsize);
	if( *xp == xsize ) *xp=0;
	*yp = (dimension_t)(drand48() * (double)ysize);
	if( *yp == ysize ) *yp=0;
}


