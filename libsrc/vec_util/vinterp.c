#include "quip_config.h"

/* utility funcs for vinterp() */

#include "quip_prot.h"
#include "data_obj.h"
#include "debug.h"
#include "veclib/vecgen.h"
#include "vec_util.h"

#define MAX_AVG	5
// BUG global var not thread-safe!?
static int max_avg=MAX_AVG;	/* BUG this should be a menu-settable param */

static float get_end_val(Data_Obj *source,Data_Obj *control,dimension_t index)
{
	float *f,*c,sum;
	int i=0;

	f = (float *)OBJ_DATA_PTR(source);
	f += index*OBJ_PXL_INC(source);
	c = (float *)OBJ_DATA_PTR(control);
	c += index*OBJ_PXL_INC(control);

	sum = 0.0;
	while( (index+i) < OBJ_COLS(source) && *c == 1.0 && i < max_avg ){
		sum += *f;
		f += OBJ_PXL_INC(source);
		c += OBJ_PXL_INC(control);
		i++;
	}
	sum /= i;
	return(sum);
}

static float get_start_val(Data_Obj *source,Data_Obj *control,dimension_t index)
{
	float *f,*c,sum;
	int i=0;

	f = (float *)OBJ_DATA_PTR(source);
	f += index*OBJ_PXL_INC(source);
	c = (float *)OBJ_DATA_PTR(control);
	c += index*OBJ_PXL_INC(control);

	sum = 0.0;
	while( (((int)index)-i) >= 0 && *c == 1.0 && i < max_avg ){
		sum += *f;
		f -= OBJ_PXL_INC(source);
		c -= OBJ_PXL_INC(control);
		i++;
	}
#ifdef DEBUG
if( debug ){
sprintf(DEFAULT_ERROR_STRING,"get_start_val:  index = %d  n = %d  sum = %f",index,i,sum);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif
	sum /= i;
	return(sum);
}


/*
 * this "interpolation" function is useful for saccade removal of eye-movement
 * records when we don't want to shift the records; we put it here for no
 * good reason except that is is somewhat like stitching!?;
 * we interpolate the data from a target vector based on
 * a control input (saccade detector), if the control is 1 we copy the
 * data, otherwise we interpolate linearly...
 */

void vinterp(QSP_ARG_DECL  Data_Obj *target,Data_Obj *source,Data_Obj *control)
{
	float *to, *fr, *c;
	dimension_t i;
	int32_t n_to_interpolate=0;
	float *interp_dest;
	int32_t start_index=(-1);

	assert( target != NULL );
	assert( source != NULL );
	assert( control != NULL );

	INSIST_RAM_OBJ(target,vinterp)
	INSIST_RAM_OBJ(source,vinterp)
	INSIST_RAM_OBJ(control,vinterp)

	if( not_prec(target,PREC_SP) )  return;
	if( not_prec(source,PREC_SP) )  return;
	if( not_prec(control,PREC_SP) ) return;


	if( !dp_same_size(target,source,"vinterp") ){
		NWARN("vinterp:  target/source length mismatch");
		return;
	}
	if( !dp_same_size(target,control,"vinterp") ){
		NWARN("vinterp:  target/control length mismatch");
		return;
	}
	if( OBJ_COMPS(source) != 1 || OBJ_COMPS(target) != 1 ){
		NWARN("vinterp:  component dimensions must be 1");
		return;
	}

	/* could check that they are all vectors... */

	to=(float *)OBJ_DATA_PTR(target);
	fr=(float *)OBJ_DATA_PTR(source);
	c=(float *)OBJ_DATA_PTR(control);

	interp_dest=to;

	for(i=0;i<OBJ_COLS(target);i++){
		if( *c == 1.0 ){			/* copy data */
			if( n_to_interpolate > 0 ){	/* end of gap? */
				int j; float start_val, end_val;

				end_val = get_end_val(source,control,i);

				/* if we haven't seen any good values yet,
				 * just fill in with the first good value.
				 */

				if( start_index < 0 ) start_val=end_val;

				/*
				 * Otherwise, use a starting value which
				 * is an average of the last N good values...
				 */

				else start_val = get_start_val(source,control,start_index);

#ifdef DEBUG
if( debug ){
sprintf(ERROR_STRING,
"vinterp:  %d values at index %d (start_i = %d), start = %f end = %f",
n_to_interpolate,i,start_index,start_val,end_val);
advise(ERROR_STRING);
}
#endif /* DEBUG */
				for(j=0;j<n_to_interpolate;j++){
					float factor;
					factor=((float)j+1)/((float)n_to_interpolate+1);
					*interp_dest = factor*end_val
						+ (1-factor)*start_val;
					interp_dest += OBJ_PXL_INC(target);
				}
			}
			*to = *fr;
			start_index = i;		/* always the last good one seen */
			n_to_interpolate=0;
		} else {				/* control is 0 */
			if( n_to_interpolate == 0 )	/* remember start */
				interp_dest = to;
			n_to_interpolate++;
		}
		to += OBJ_PXL_INC(target);
		c += OBJ_PXL_INC(control);
		fr += OBJ_PXL_INC(source);
	}
	if( n_to_interpolate > 0 ){		/* fill in at end? */
		float fill_val;
		int j;
		if( start_index < 0 ){
			NWARN("vinterp:  no valid data!?");
			fill_val=0.0;
		} else fill_val = get_start_val(source,control,start_index);
		for(j=0;j<n_to_interpolate;j++){
			*interp_dest = fill_val;
			interp_dest += OBJ_PXL_INC(target);
		}
	}
}

/* There was a wartbl implementation of vflip here...
 * We really don't need this, since we can implement a flip operation
 * with an equivalenced data object with a -1 increment...
 */

