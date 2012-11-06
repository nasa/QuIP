
#include "quip_config.h"

char VersionId_dither_gspread[] = QUIP_VERSION_STRING;

/*
 * utility routines for optimal halftoning
 */

#ifdef HAVE_MATH_H
#include <math.h>
#endif

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

	hy=fdp->dt_rows/2;
	hx=fdp->dt_cols/2;
	/*
	x -= hx;
	y -= hy;
	*/
	err=0.0;
	eptr = (float *)edp->dt_data;
	fptr = (float *)fdp->dt_data;

	j=(long)fdp->dt_rows;
	while(j--){
		iy=y+j-hy;
#ifdef NOWRAP
		if( iy < 0 ) continue;
		else if( iy >= edp->dt_rows ) continue;
#else
		while( iy < 0 ) iy += edp->dt_rows;
		while( iy >= (incr_t)edp->dt_rows ) iy -= edp->dt_rows;
#endif /* NOWRAP */

		yos = iy*(long)edp->dt_rowinc;

		foffset = (j+1) * (long)fdp->dt_rowinc;

		i=(long)fdp->dt_cols;
		while(i--){
			foffset--;

			ix=x+i-hx;
#ifdef NOWRAP
			if( ix < 0 ) continue;
			else if( ix >= edp->dt_cols ) continue;
#else
			while( ix < 0 ) ix += edp->dt_cols;
			while( ix >= (incr_t)edp->dt_cols ) ix -= edp->dt_cols;
#endif /* NOWRAP */
			eoffset = yos + ix;

			err += *(eptr+eoffset)
				* *(fptr+foffset);

#ifdef DEBUG
/*
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"get_ferror %d %d:  %d %d, err = %g, filt = %g, running total %g",
x,y,i,j,*(eptr+eoffset),*(fptr+foffset),err);
advise(DEFAULT_ERROR_STRING);
}
*/
#endif /* DEBUG */
		}
	}
#ifdef DEBUG
/*
if( debug & spread_debug ){
sprintf(DEFAULT_ERROR_STRING,"get_ferror %d %d:  TOTAL err = %g",x,y,err);
advise(DEFAULT_ERROR_STRING);
}
*/
#endif /* DEBUG */

	return(err);
}

/* really the mean square, after we divide by the number of pixels! */

double get_sos(Data_Obj *edp,Data_Obj *fdp)		/* get the total sq'd error */
{
	incr_t i,j;
	double sos, rowsos,err;

	sos = 0.0;
	for(j=0;j<(incr_t)edp->dt_rows;j++){
		rowsos=0.0;
		for(i=0;i<(incr_t)edp->dt_cols;i++){
			err = get_ferror(edp,fdp,i,j);
			rowsos += err * err;
		}
		sos += rowsos;
	}

	/* normalize by number of pixels */
	/* why rowinc and not ncols??? */
	sos /= edp->dt_rows*edp->dt_rowinc;
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

double add_to_sos(dimension_t x,dimension_t y,Data_Obj *edp,Data_Obj *fdp,int factor)
{
	dimension_t i,j;
	double err,adj;
	incr_t xx,yy;

	/*
	if( the_sos == NO_VALUE )
		the_sos = get_sos(edp,fdp);
	*/

	adj =0.0;
	for(j=0;j<fdp->dt_rows;j++){
		yy = (incr_t)(y + j) - (incr_t)fdp->dt_rows/2;
#ifdef NOWRAP
		if( yy >= 0 && yy < edp->dt_rows ){
			for(i=0;i<fdp->dt_cols;i++){
				xx = (incr_t)(x + i) - (incr_t)(fdp->dt_cols/2);
				if( xx >= 0 && xx < edp->dt_cols ){
					err = get_ferror(edp,fdp,xx,yy);
					adj += err*err;
				}
			}
		}
#else
		while( yy < 0 ) yy += edp->dt_rows;
		while( yy >= (incr_t)edp->dt_rows ) yy -= edp->dt_rows;
		for(i=0;i<fdp->dt_cols;i++){
			xx = x + i - fdp->dt_cols/2;
			while( xx < 0 ) xx += edp->dt_cols;
			while( xx >= (incr_t)edp->dt_cols ) xx -= edp->dt_cols;
			err = get_ferror(edp,fdp,xx,yy);
			adj += err*err;
		}
#endif /* NOWRAP */
	}
	/* normalize by number of pixels */
	if( factor == 1 )
		adj /= (edp->dt_cols * edp->dt_rows);
	else if( factor == -1 )
		adj /= - (edp->dt_cols * edp->dt_rows);
#ifdef CAUTIOUS
	else {
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  add_to_sos:  factor (%d) is not 1 or -1 !?",factor);
		NWARN(DEFAULT_ERROR_STRING);
		return(0.0);
	}
#endif /* CAUTIOUS */
	/* the_sos += adj; */
	return(adj);
}
	
void get_xy_scattered_point(dimension_t n,dimension_t xsize,dimension_t ysize,dimension_t *xp,dimension_t *yp)
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
		sprintf(DEFAULT_ERROR_STRING,"get_xy_scattered_point:  width (%d) and height (%d) should match!?",
			xsize,ysize);
		NWARN(DEFAULT_ERROR_STRING);
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

void get_xy_raster_point(dimension_t n,dimension_t xsize,dimension_t ysize,dimension_t *xp,dimension_t *yp)
{
	*xp = n % xsize;
	*yp = n / xsize;
}

void get_xy_random_point(dimension_t n,dimension_t xsize,dimension_t ysize,dimension_t *xp,dimension_t *yp)
{
	*xp = (dimension_t)(drand48() * (double)xsize);
	if( *xp == xsize ) *xp=0;
	*yp = (dimension_t)(drand48() * (double)ysize);
	if( *yp == ysize ) *yp=0;
}


