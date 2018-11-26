#include "quip_config.h"

#include <stdio.h>
#include <math.h>

#include "vec_util.h"
#include "debug.h"
#include "quip_prot.h"

#define render_check(image_dp,coord_dp,intens_dp) _render_check(QSP_ARG  image_dp,coord_dp,intens_dp)

static int _render_check(QSP_ARG_DECL  Data_Obj *image_dp,Data_Obj *coord_dp,Data_Obj *intens_dp)
{
	if( OBJ_PREC(image_dp) != PREC_SP ){
		sprintf(ERROR_STRING,"render_check:  source image %s must be float",OBJ_NAME(image_dp));
		WARN(ERROR_STRING);
		return(-1);
	}
	if( OBJ_COMPS(coord_dp) != 2 ){
		sprintf(ERROR_STRING,"render_check:  coordinate list %s (%d) must have two components",OBJ_NAME(coord_dp),
			OBJ_COMPS(coord_dp));
		WARN(ERROR_STRING);
		return(-1);
	}
	if( OBJ_MACH_PREC(intens_dp) != PREC_SP ){
		sprintf(ERROR_STRING,"render_check:  intensity list %s must be float",OBJ_NAME(intens_dp));
		WARN(ERROR_STRING);
		return(-1);
	}
	if( OBJ_MACH_PREC(coord_dp) != PREC_SP ){
		sprintf(ERROR_STRING,"render_check:  coordinate list %s must be float",
			OBJ_NAME(coord_dp));
		WARN(ERROR_STRING);
		return(-1);
	}
	if( !IS_CONTIGUOUS(image_dp) || !IS_CONTIGUOUS(coord_dp) ||
			!IS_CONTIGUOUS(intens_dp) ){
		WARN("render_check:  Sorry, objects must be contiguous for render_samples");
		return(-1);
	}
	if( OBJ_COMPS(image_dp) != OBJ_COMPS(intens_dp) ){
		sprintf(ERROR_STRING,"render_check:  intensity list %s (%d) must have same number of components as target image %s (%d)",
			OBJ_NAME(intens_dp),OBJ_COMPS(intens_dp),OBJ_NAME(image_dp),OBJ_COMPS(image_dp));
		WARN(ERROR_STRING);
		return(-1);
	}
	if( OBJ_COMPS(image_dp)>1 && OBJ_COMP_INC(image_dp) != 1 ){
		sprintf(ERROR_STRING,"render_check:  Sorry, target image %s has component increment (%d) != 1",
				OBJ_NAME(image_dp),OBJ_COMP_INC(image_dp));
		WARN(ERROR_STRING);
		advise("Target image component increment must be 1 for rendering and sampling");
		return(-1);
	}


	/* the number of coordinates must match the number of samples */

	/* like to use dp_same_size() here, but type dim is unequal!? */

	if( OBJ_ROWS(coord_dp) != OBJ_ROWS(intens_dp) ||
		/* BUG should check all dims? */
		OBJ_COLS(coord_dp) != OBJ_COLS(intens_dp) ){
		sprintf(ERROR_STRING,
	"Coordinate object %s (%d x %d) differs in size from intensity object %s (%d x %d)",
			OBJ_NAME(coord_dp),
			OBJ_ROWS(coord_dp),
			OBJ_COLS(coord_dp),
			OBJ_NAME(intens_dp),
			OBJ_ROWS(intens_dp),
			OBJ_COLS(intens_dp));
		WARN(ERROR_STRING);
		return(-1);
	}
	return(0);
}

void _render_samples(QSP_ARG_DECL  Data_Obj *image_dp, Data_Obj *coord_dp, Data_Obj *intens_dp)
{
	float *image, *coord, *intens;
	dimension_t width, height;
	float x,y;
	incr_t ix,iy;
	dimension_t i,k;

	if( render_check(image_dp,coord_dp,intens_dp) < 0 )
		return;

	image  = (float *) OBJ_DATA_PTR(image_dp);
	coord  = (float *) OBJ_DATA_PTR(coord_dp);
	intens = (float *) OBJ_DATA_PTR(intens_dp);
	width  = OBJ_COLS(image_dp);
	height = OBJ_ROWS(image_dp);

	//x=0;
    //y=0;

	i=OBJ_N_TYPE_ELTS(coord_dp)/2;

	while(i--){
		x = *coord++;
		y = *coord++;
		ix = (incr_t)( x + 0.5 );
		iy = (incr_t)( y + 0.5 );

		if( iy < (incr_t)height && iy >= 0 && ix >= 0 && ix < (incr_t)width ){
			for(k=0;k<OBJ_COMPS(image_dp);k++)
				image[ iy * OBJ_ROW_INC(image_dp) + ix * OBJ_PXL_INC(image_dp) + k * OBJ_COMP_INC(image_dp) ] =
					*intens++;
		} else {
			if( verbose ){
				sprintf(msg_str,"render_samples:  clipping point at %d %d",ix,iy);
				prt_msg(msg_str);
			}
			for(k=0;k<OBJ_COMPS(image_dp);k++)
				intens++;
		}
	}
} /* end render_samples() */

/* Uses pixel quadrature method.
 *
 * The coordinates specify the location in the target image where
 * to put the energy from the corresponding source pixel.
 */

void _render_samples2(QSP_ARG_DECL  Data_Obj *image_dp, Data_Obj *coord_dp, Data_Obj *intens_dp)
{
	float *image, *coord, *intens;
	dimension_t width, height;
	float x,y;
	incr_t ix,iy,ix2,iy2;
	dimension_t i,k;
	float dx,dy,dxy;

	if( render_check(image_dp,coord_dp,intens_dp) < 0 )
		return;

fprintf(stderr,"render_samples2 %s %s %s BEGIN\n",
OBJ_NAME(image_dp),OBJ_NAME(coord_dp),OBJ_NAME(intens_dp));

	image  = (float *) OBJ_DATA_PTR(image_dp);
	coord  = (float *) OBJ_DATA_PTR(coord_dp);
	intens = (float *) OBJ_DATA_PTR(intens_dp);
	width  = OBJ_COLS(image_dp);
	height = OBJ_ROWS(image_dp);

	//x=0;
    //y=0;

	i=OBJ_N_TYPE_ELTS(coord_dp)/2;
	while(i--){			/* foreach destination coordinate pair */
		x = *coord++;
		y = *coord++;

		if( isnan(x) || isnan(y) ){
			sprintf(ERROR_STRING,
	"render_samples2:  nan value passed in coordinate list %s",
				OBJ_NAME(coord_dp));
			error1(ERROR_STRING);
			// IOS_RETURN?
		}
			
		if ( (x > (float)(width  - 1)) || (x < 0) ||
		     (y > (float)(height - 1)) || (y < 0) )
		  {
//fprintf(stderr,"coordinate %g %g is out of range\n",x,y);
			intens += OBJ_COMPS(image_dp);	/* same as intens_dp */
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

		ix  *= OBJ_PXL_INC(image_dp);
		ix2 *= OBJ_PXL_INC(image_dp);
		iy  *= OBJ_ROW_INC(image_dp );
		iy2 *= OBJ_ROW_INC(image_dp );

		/* BUG this code relies on the component increment == 1 */
		for(k=0;k<OBJ_COMPS(image_dp);k++){
			*(image + ix  + iy + k ) += ((*intens) * (1-dx-dy+dxy));
			*(image + ix  + iy2 + k ) += ((*intens) * (dy - dxy));
			*(image + ix2 + iy + k ) += ((*intens) * (dx - dxy));
			*(image + ix2 + iy2 +k ) += ((*intens) * dxy);
			intens++;
		}
	}
}

void _sample_image(QSP_ARG_DECL  Data_Obj *intens_dp, Data_Obj *image_dp, Data_Obj *coord_dp)
{
	float *image, *coord, *intens;
	dimension_t width, height;
	float x,y;
	incr_t ix,iy,ix2,iy2;
	dimension_t i,k;
	float dx,dy,dxy;

	if( render_check(image_dp,coord_dp,intens_dp) < 0 )
		return;

	image  = (float *) OBJ_DATA_PTR(image_dp);
	coord  = (float *) OBJ_DATA_PTR(coord_dp);
	intens = (float *) OBJ_DATA_PTR(intens_dp);
	width  = OBJ_COLS(image_dp);
	height = OBJ_ROWS(image_dp);

	//x=0;
    //y=0;

	i=OBJ_N_TYPE_ELTS(coord_dp)/2;

	while(i--){
		x = *coord++;
		y = *coord++;

		if ( (x > (float)(width  - 1)) || (x < 0) ||
		     (y > (float)(height - 1)) || (y < 0) )
		  {
			intens += OBJ_COMPS(image_dp);
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

		ix  *= OBJ_PXL_INC(image_dp);
		ix2 *= OBJ_PXL_INC(image_dp);
		iy  *= OBJ_ROW_INC(image_dp);
		iy2 *= OBJ_ROW_INC(image_dp);

		for(k=0;k<OBJ_COMPS(image_dp);k++){
			*intens++ = (*(image + ix  + iy + k ) * (1-dx-dy+dxy))
				  + (*(image + ix  + iy2 + k ) * (dy - dxy))
				  + (*(image + ix2 + iy + k ) * (dx - dxy))
				  + (*(image + ix2 + iy2 + k ) * dxy);
		}
	}
}

