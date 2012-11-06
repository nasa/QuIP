#include "quip_config.h"

char VersionId_vec_util_local_max[] = QUIP_VERSION_STRING;

/* find local maxima - used on output of hough transform.
 * This version is naive and doesn't worry about wrap-around...
 * How should we handle this?
 *
 * For now, we assume that the hough transform is padded so we don't need to check
 * the boundary pixels or deal with wrap-around.  This simplifies things considerably...
 */

#include "data_obj.h"
#include "vec_util.h"

long local_maxima(QSP_ARG_DECL  Data_Obj *val_dp, Data_Obj *coord_dp, Data_Obj *src_dp )
{
	float *nw_p, *w_p, *sw_p;
	float v,*src_p;
	float *base_p;
	u_long *coord_p;
	u_long x,y;
	dimension_t n_maxima;
	float *val_p;

	/* For now, we only support contiguous float */
	if( src_dp->dt_prec != PREC_SP ){
		sprintf(error_string,"local_maxima:  source image %s (%s) should have %s precision!?",
			src_dp->dt_name,name_for_prec(src_dp->dt_prec),name_for_prec(PREC_SP));
		advise(error_string);
		return(-1);
	}
	if( val_dp->dt_prec != PREC_SP ){
		sprintf(error_string,"local_maxima:  max value vector %s (%s) should have %s precision!?",
			val_dp->dt_name,name_for_prec(val_dp->dt_prec),name_for_prec(PREC_SP));
		advise(error_string);
		return(-1);
	}
	if( ! IS_CONTIGUOUS(src_dp) ){
		sprintf(error_string,"local_maxima:  source image %s should be contiguous!?",
			src_dp->dt_name);
		advise(error_string);
		return(-1);
	}
	if( coord_dp->dt_prec != PREC_UDI ){
		sprintf(error_string,"local_maxima:  coordinate array %s (%s) should have %s precision!?",
			coord_dp->dt_name,name_for_prec(coord_dp->dt_prec),name_for_prec(PREC_UDI));
		advise(error_string);
		return(-1);
	}
	if( ! IS_CONTIGUOUS(coord_dp) ){
		sprintf(error_string,"local_maxima:  coordinate vector %s should be contiguous!?",
			coord_dp->dt_name);
		advise(error_string);
		return(-1);
	}
	if( ! IS_CONTIGUOUS(val_dp) ){
		sprintf(error_string,"local_maxima:  max value vector %s should be contiguous!?",
			val_dp->dt_name);
		advise(error_string);
		return(-1);
	}

	base_p = (float *)src_dp->dt_data;
	coord_p = (u_long *)coord_dp->dt_data;
	val_p = (float *)val_dp->dt_data;

	n_maxima=0;

	for(y=1;y<(src_dp->dt_rows-1);y++){
		nw_p = base_p;
		w_p = nw_p + src_dp->dt_cols;
		sw_p = w_p + src_dp->dt_cols;
		src_p = w_p + 1;
		for(x=1;x<(src_dp->dt_cols-1);x++){
			v=(*src_p);
//sprintf(error_string,"%g at %d %d",v,x,y);
//prt_msg(error_string);
			if(
				   v >  *(nw_p  )
				&& v >  *(nw_p+1)
				&& v >  *(nw_p+2)
				&& v >  *( w_p  )
				&& v >= *( w_p+2)
				&& v >= *(sw_p  )
				&& v >= *(sw_p+1)
				&& v >= *(sw_p+2)
								){

//sprintf(error_string,"local max %g at %d %d",v,x,y);
//advise(error_string);
				/* this is a local max! */
				n_maxima++;
				if( n_maxima > coord_dp->dt_cols ){
					sprintf(error_string,"local_maxima:  coord vector %s (%d columns) needs to be enlarged!?",
						coord_dp->dt_name,coord_dp->dt_cols);
					NWARN(error_string);
					return(coord_dp->dt_cols);
				}
				*val_p++ = v;
				*coord_p++ = x;
				*coord_p++ = y;
			}

			src_p++;
			nw_p++;
			w_p++;
			sw_p++;
		}
		base_p += src_dp->dt_cols;
	}
	return(n_maxima);
}

