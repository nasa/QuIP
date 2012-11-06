#include "quip_config.h"

char VersionId_vec_util_lutmap[] = QUIP_VERSION_STRING;

#include "data_obj.h"
#include "vec_util.h"

#define CLIP_INDEX							\
									\
	if( si >= lut_dp->dt_cols ){					\
		sprintf(error_string,					\
"Index value %d exceeds size of lookup table %s (%d), clipping.",	\
			si,lut_dp->dt_name,lut_dp->dt_cols);		\
		WARN(error_string);					\
		si = lut_dp->dt_cols - 1;				\
	}

/* Originally there was a call to chain_breaks here, probably from when the
 * function was part of lib warf...  NONE of the functions in vec_util
 * are chainable, because they are outside of the veclib framework...
 * This could probably be fixed up if it were deemed worthwhile, but
 * that would really require integrating these functions into the veclib
 * framework somehow...  not obvious how to do it cleanly.
 */

/* do like a lookup table */

#define DO_MAPPING(dst_type,src_type)					\
		u_long npix;						\
		dstptr = (dst_type *)dest_dp->dt_data;			\
		srcptr = (src_type *)src_dp->dt_data;			\
		lutptr = (dst_type *)lut_dp->dt_data;			\
									\
		npix = src_dp->dt_n_type_elts;				\
		if( lut_dp->dt_pinc == 1 ){				\
			for(i=0;i<npix;i++){				\
				si = *srcptr++;				\
				CLIP_INDEX				\
				*dstptr++ = lutptr[ si ];		\
			}						\
		} else {						\
			for(i=0;i<npix;i++){				\
				si = *srcptr++;				\
				CLIP_INDEX				\
				si *= lut_dp->dt_pinc;			\
				for(j=0;j<dest_dp->dt_comps;j++)		\
					*dstptr++ = lutptr[ si+j*lut_dp->dt_cinc ];	\
			}						\
		} 

int lutmap( QSP_ARG_DECL  Data_Obj *dest_dp, Data_Obj *src_dp, Data_Obj *lut_dp )
{
	dimension_t i,j;

	if( dest_dp->dt_prec != lut_dp->dt_prec ){
		sprintf(error_string,
	"lutmap:  precision mismatch destination %s (%s) and lut %s (%s)",
			dest_dp->dt_name,PNAME(dest_dp),
			lut_dp->dt_name,PNAME(lut_dp));
		WARN(error_string);
		return(-1);
	}
	if( dest_dp->dt_comps != lut_dp->dt_comps*src_dp->dt_comps ){
		sprintf(error_string,
		"lutmap:  destination (%s) depth (%d) should be product of",
			dest_dp->dt_name, dest_dp->dt_comps);
		WARN(error_string);
		sprintf(error_string,
		"\tsource (%s) depth (%d) and lut (%s) depth (%d)",
			src_dp->dt_name,  src_dp->dt_comps,
			lut_dp->dt_name,  lut_dp->dt_comps
			);
		WARN(error_string);
		return(-1);
	}
	if( dest_dp->dt_n_type_elts != src_dp->dt_n_type_elts * lut_dp->dt_comps ){
		sprintf(error_string,"lutmap:  destination %s and source %s must match in size",
			dest_dp->dt_name,src_dp->dt_name);
		WARN(error_string);
		return(-1);
	}
	if( src_dp->dt_prec != PREC_UIN && src_dp->dt_prec != PREC_UBY ){
		sprintf(error_string,
"lutmap:  source %s precision (%s) must be unsigned byte or unsigned short",
			src_dp->dt_name,name_for_prec(src_dp->dt_prec));
		WARN(error_string);
		return(-1);
	}
	if( src_dp->dt_prec == PREC_UBY && lut_dp->dt_cols != 256 ){
		WARN("lutmap:  lut size must be 256 for byte indexing");
		return(-1);
	}
	/* BUG?  if we use short indices, we may not want to have a full 64k lookup table... */
	if( !IS_CONTIGUOUS(src_dp) ){
		sprintf(error_string,"lutmap:  source image %s must be contiguous",
			src_dp->dt_name);
		WARN(error_string);
		return(-1);
	}
	if( !IS_CONTIGUOUS(dest_dp) ){
		sprintf(error_string,"lutmap:  destination image %s must be contiguous",
			dest_dp->dt_name);
		WARN(error_string);
		return(-1);
	}

	if( dest_dp->dt_prec == PREC_SP ){
		float *dstptr, *lutptr;
		dimension_t si;

		if( src_dp->dt_prec == PREC_UBY ){
			u_char *srcptr;
			DO_MAPPING(float,u_char)
		} else if( src_dp->dt_prec == PREC_UIN ){
			u_short *srcptr;
			DO_MAPPING(float,u_short)
		}
	} else if( MACHINE_PREC(dest_dp) == PREC_BY || 
		MACHINE_PREC(dest_dp) == PREC_UBY ){
		u_char *dstptr, *lutptr;
		dimension_t si;

		if( src_dp->dt_prec == PREC_UBY ){
			u_char *srcptr;
			DO_MAPPING(u_char,u_char)
		} else if( src_dp->dt_prec == PREC_UIN ){
			u_short *srcptr;
			DO_MAPPING(u_char,u_short)
		}
	} else {
		sprintf(error_string,
			"Sorry, unhandled destination precision (%s) for lutmap",name_for_prec(dest_dp->dt_prec));
		WARN(error_string);
		return(-1);
	}
	return(0);
}

