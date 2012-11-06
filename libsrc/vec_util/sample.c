#include "quip_config.h"

char VersionId_vec_util_sample[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include "data_obj.h"
#include "debug.h"
#include "vec_util.h"

/* local prototypes */
static int render_check(QSP_ARG_DECL
			Data_Obj *image_dp,
			Data_Obj *coord_dp,
			Data_Obj *intens_dp);

static int render_check(QSP_ARG_DECL  Data_Obj *image_dp,Data_Obj *coord_dp,Data_Obj *intens_dp)
{
	if( image_dp->dt_prec != PREC_SP ){
		sprintf(ERROR_STRING,"render_check:  source image %s must be float",image_dp->dt_name);
		WARN(ERROR_STRING);
		return(-1);
	}
	if( coord_dp->dt_comps != 2 ){
		sprintf(ERROR_STRING,"render_check:  coordinate list %s (%d) must have two components",coord_dp->dt_name,
			coord_dp->dt_comps);
		WARN(ERROR_STRING);
		return(-1);
	}
	if( MACHINE_PREC(intens_dp) != PREC_SP ){
		sprintf(ERROR_STRING,"render_check:  intensity list %s must be float",intens_dp->dt_name);
		WARN(ERROR_STRING);
		return(-1);
	}
	if( MACHINE_PREC(coord_dp) != PREC_SP ){
		sprintf(ERROR_STRING,"render_check:  coordinate list %s must be float",
			coord_dp->dt_name);
		WARN(ERROR_STRING);
		return(-1);
	}
	if( !IS_CONTIGUOUS(image_dp) || !IS_CONTIGUOUS(coord_dp) ||
			!IS_CONTIGUOUS(intens_dp) ){
		WARN("render_check:  Sorry, objects must be contiguous for render_samples");
		return(-1);
	}
	if( image_dp->dt_comps != intens_dp->dt_comps ){
		sprintf(ERROR_STRING,"render_check:  intensity list %s (%d) must have same number of components as target image %s (%d)",
			intens_dp->dt_name,intens_dp->dt_comps,image_dp->dt_name,image_dp->dt_comps);
		WARN(ERROR_STRING);
		return(-1);
	}
	if( image_dp->dt_comps>1 && image_dp->dt_cinc != 1 ){
		sprintf(ERROR_STRING,"render_check:  Sorry, target image %s has component increment (%d) != 1",
				image_dp->dt_name,image_dp->dt_cinc);
		WARN(ERROR_STRING);
		advise("Target image component increment must be 1 for rendering and sampling");
		return(-1);
	}


	/* the number of coordinates must match the number of samples */

	/* like to use dp_same_size() here, but type dim is unequal!? */

	if( coord_dp->dt_rows != intens_dp->dt_rows ||
		/* BUG should check all dims? */
		coord_dp->dt_cols != intens_dp->dt_cols ){
		sprintf(ERROR_STRING,
	"Coordinate object %s (%d x %d) differs in size from intensity object %s (%d x %d)",
			coord_dp->dt_name,
			coord_dp->dt_rows,
			coord_dp->dt_cols,
			intens_dp->dt_name,
			intens_dp->dt_rows,
			intens_dp->dt_cols);
		WARN(ERROR_STRING);
		return(-1);
	}
	return(0);
}

void render_samples(QSP_ARG_DECL  Data_Obj *image_dp, Data_Obj *coord_dp, Data_Obj *intens_dp)
{
	float *image, *coord, *intens;
	dimension_t width, height;
	float x,y;
	incr_t ix,iy;
	dimension_t i,k;

	if( render_check(QSP_ARG  image_dp,coord_dp,intens_dp) < 0 )
		return;

	image  = (float *) image_dp->dt_data;
	coord  = (float *) coord_dp->dt_data;
	intens = (float *) intens_dp->dt_data;
	width  = image_dp->dt_cols;
	height = image_dp->dt_rows;

	x=y=0;

	i=coord_dp->dt_n_type_elts/2;

	while(i--){
		x = *coord++;
		y = *coord++;
		ix = (incr_t)( x + 0.5 );
		iy = (incr_t)( y + 0.5 );

		if( iy < (incr_t)height && iy >= 0 && ix >= 0 && ix < (incr_t)width ){
			for(k=0;k<image_dp->dt_comps;k++)
				image[ iy * image_dp->dt_rinc + ix * image_dp->dt_pinc + k * image_dp->dt_cinc ] =
					*intens++;
		} else {
			if( verbose ){
				sprintf(msg_str,"render_samples:  clipping point at %d %d",ix,iy);
				prt_msg(msg_str);
			}
			for(k=0;k<image_dp->dt_comps;k++)
				intens++;
		}
	}
} /* end render_samples() */

/* Uses pixel quadrature method.
 *
 * The coordinates specify the location in the target image where
 * to put the energy from the corresponding source pixel.
 */

void render_samples2(QSP_ARG_DECL  Data_Obj *image_dp, Data_Obj *coord_dp, Data_Obj *intens_dp)
{
	float *image, *coord, *intens;
	dimension_t width, height;
	float x,y;
	incr_t ix,iy,ix2,iy2;
	dimension_t i,k;
	float dx,dy,dxy;

	if( render_check(QSP_ARG  image_dp,coord_dp,intens_dp) < 0 )
		return;

	image  = (float *) image_dp->dt_data;
	coord  = (float *) coord_dp->dt_data;
	intens = (float *) intens_dp->dt_data;
	width  = image_dp->dt_cols;
	height = image_dp->dt_rows;

	x=y=0;

	i=coord_dp->dt_n_type_elts/2;

	while(i--){			/* foreach destination coordinate pair */
		x = *coord++;
		y = *coord++;

		if ( (x > (float)(width  - 1)) || (x < 0) ||
		     (y > (float)(height - 1)) || (y < 0) )
		  {
			intens += image_dp->dt_comps;	/* same as intens_dp */
			continue;
		  }

		ix = (incr_t)x; /* truncate fraction */
		iy = (incr_t)y;

		/* make sure that this sample is not the largest x or y coord */
		if( ix == (incr_t)(width-1) ) ix--;
		if( iy == (incr_t)(height-1) ) iy--;

		dx=x-(float)ix;
		dy=y-(float)iy;
		dxy = dx*dy;

		ix2=ix+1;
		iy2=iy+1;

		ix  *= image_dp -> dt_pinc;
		ix2 *= image_dp -> dt_pinc;
		iy  *= image_dp -> dt_rowinc;
		iy2 *= image_dp -> dt_rowinc;

		/* BUG this code relies on the component increment == 1 */
		for(k=0;k<image_dp->dt_comps;k++){
			*(image + ix  + iy + k ) += ((*intens) * (1-dx-dy+dxy));
			*(image + ix  + iy2 + k ) += ((*intens) * (dy - dxy));
			*(image + ix2 + iy + k ) += ((*intens) * (dx - dxy));
			*(image + ix2 + iy2 +k ) += ((*intens) * dxy);
			intens++;
		}
	}
}

void sample_image(QSP_ARG_DECL  Data_Obj *intens_dp, Data_Obj *image_dp, Data_Obj *coord_dp)
{
	float *image, *coord, *intens;
	dimension_t width, height;
	float x,y;
	incr_t ix,iy,ix2,iy2;
	dimension_t i,k;
	float dx,dy,dxy;

	if( render_check(QSP_ARG  image_dp,coord_dp,intens_dp) < 0 )
		return;

	image  = (float *) image_dp->dt_data;
	coord  = (float *) coord_dp->dt_data;
	intens = (float *) intens_dp->dt_data;
	width  = image_dp->dt_cols;
	height = image_dp->dt_rows;

	x=y=0;

	i=coord_dp->dt_n_type_elts/2;

	while(i--){
		x = *coord++;
		y = *coord++;

		if ( (x > (float)(width  - 1)) || (x < 0) ||
		     (y > (float)(height - 1)) || (y < 0) )
		  {
			intens += image_dp->dt_comps;
			continue;
		  }

		ix = (incr_t)x;		/* round down (floor) */
		iy = (incr_t)y;

		/* make sure that this sample is not the largest x or y coord */
		if( ix == (incr_t)(width-1) ) ix--;
		if( iy == (incr_t)(height-1) ) iy--;

		dx=x-(float)ix;
		dy=y-(float)iy;
		dxy = dx*dy;

		ix2=ix+1;
		iy2=iy+1;

		ix  *= image_dp -> dt_pinc;
		ix2 *= image_dp -> dt_pinc;
		iy  *= image_dp -> dt_rowinc;
		iy2 *= image_dp -> dt_rowinc;

		for(k=0;k<image_dp->dt_comps;k++){
			*intens++ = (*(image + ix  + iy + k ) * (1-dx-dy+dxy))
				  + (*(image + ix  + iy2 + k ) * (dy - dxy))
				  + (*(image + ix2 + iy + k ) * (dx - dxy))
				  + (*(image + ix2 + iy2 + k ) * dxy);
		}
	}
}

