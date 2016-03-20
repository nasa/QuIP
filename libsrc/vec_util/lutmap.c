#include "quip_config.h"

#include "vec_util.h"
#include "quip_prot.h"

#define CLIP_INDEX							\
									\
	if( si >= OBJ_COLS(lut_dp) ){					\
		sprintf(ERROR_STRING,					\
"Index value %d exceeds size of lookup table %s (%d), clipping.",	\
			si,OBJ_NAME(lut_dp),OBJ_COLS(lut_dp));		\
		WARN(ERROR_STRING);					\
		si = OBJ_COLS(lut_dp) - 1;				\
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
		dstptr = (dst_type *)OBJ_DATA_PTR(dest_dp);			\
		srcptr = (src_type *)OBJ_DATA_PTR(src_dp);			\
		lutptr = (dst_type *)OBJ_DATA_PTR(lut_dp);			\
									\
		npix = OBJ_N_TYPE_ELTS(src_dp);				\
		if( OBJ_PXL_INC(lut_dp) == 1 ){				\
			for(i=0;i<npix;i++){				\
				si = *srcptr++;				\
				CLIP_INDEX				\
				*dstptr++ = lutptr[ si ];		\
			}						\
		} else {						\
			for(i=0;i<npix;i++){				\
				si = *srcptr++;				\
				CLIP_INDEX				\
				si *= OBJ_PXL_INC(lut_dp);			\
				for(j=0;j<OBJ_COMPS(dest_dp);j++)		\
					*dstptr++ = lutptr[ si+j*OBJ_COMP_INC(lut_dp) ];	\
			}						\
		} 

int lutmap( QSP_ARG_DECL  Data_Obj *dest_dp, Data_Obj *src_dp, Data_Obj *lut_dp )
{
	dimension_t i,j;

	// BUG we will need lut mapping on the GPU for gamma correction!!!

	VINSIST_RAM_OBJ(dest_dp,lutmap,-1)
	VINSIST_RAM_OBJ(src_dp,lutmap,-1)
	VINSIST_RAM_OBJ(lut_dp,lutmap,-1)

	if( OBJ_PREC(dest_dp) != OBJ_PREC(lut_dp) ){
		sprintf(ERROR_STRING,
	"lutmap:  precision mismatch destination %s (%s) and lut %s (%s)",
			OBJ_NAME(dest_dp),OBJ_PREC_NAME(dest_dp),
			OBJ_NAME(lut_dp),OBJ_PREC_NAME(lut_dp));
		WARN(ERROR_STRING);
		return(-1);
	}
	if( OBJ_COMPS(dest_dp) != OBJ_COMPS(lut_dp)*OBJ_COMPS(src_dp) ){
		sprintf(ERROR_STRING,
		"lutmap:  destination (%s) depth (%d) should be product of",
			OBJ_NAME(dest_dp), OBJ_COMPS(dest_dp));
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,
		"\tsource (%s) depth (%d) and lut (%s) depth (%d)",
			OBJ_NAME(src_dp),  OBJ_COMPS(src_dp),
			OBJ_NAME(lut_dp),  OBJ_COMPS(lut_dp)
			);
		WARN(ERROR_STRING);
		return(-1);
	}
	if( OBJ_N_TYPE_ELTS(dest_dp) != OBJ_N_TYPE_ELTS(src_dp) * OBJ_COMPS(lut_dp) ){
		sprintf(ERROR_STRING,"lutmap:  destination %s and source %s must match in size",
			OBJ_NAME(dest_dp),OBJ_NAME(src_dp));
		WARN(ERROR_STRING);
		return(-1);
	}
	if( OBJ_PREC(src_dp) != PREC_UIN && OBJ_PREC(src_dp) != PREC_UBY ){
		sprintf(ERROR_STRING,
"lutmap:  source %s precision (%s) must be unsigned byte or unsigned short",
			OBJ_NAME(src_dp),OBJ_PREC_NAME(src_dp));
		WARN(ERROR_STRING);
		return(-1);
	}
	if( OBJ_PREC(src_dp) == PREC_UBY && OBJ_COLS(lut_dp) != 256 ){
		sprintf(ERROR_STRING,
"lutmap:  lut %s size (%d) must be 256 for byte indexing",OBJ_NAME(lut_dp),OBJ_COLS(lut_dp));
		WARN(ERROR_STRING);
		return(-1);
	}
	/* BUG?  if we use short indices, we may not want to have a full 64k lookup table... */
	if( !IS_CONTIGUOUS(src_dp) ){
		sprintf(ERROR_STRING,"lutmap:  source image %s must be contiguous",
			OBJ_NAME(src_dp));
		WARN(ERROR_STRING);
		return(-1);
	}
	if( !IS_CONTIGUOUS(dest_dp) ){
		sprintf(ERROR_STRING,"lutmap:  destination image %s must be contiguous",
			OBJ_NAME(dest_dp));
		WARN(ERROR_STRING);
		return(-1);
	}

	if( OBJ_PREC(dest_dp) == PREC_SP ){
		float *dstptr, *lutptr;
		dimension_t si;

		if( OBJ_PREC(src_dp) == PREC_UBY ){
			u_char *srcptr;
			DO_MAPPING(float,u_char)
		} else if( OBJ_PREC(src_dp) == PREC_UIN ){
			u_short *srcptr;
			DO_MAPPING(float,u_short)
		}
	} else if( OBJ_PREC(dest_dp) == PREC_DP ){
		double *dstptr, *lutptr;
		dimension_t si;

		if( OBJ_PREC(src_dp) == PREC_UBY ){
			u_char *srcptr;
			DO_MAPPING(double,u_char)
		} else if( OBJ_PREC(src_dp) == PREC_UIN ){
			u_short *srcptr;
			DO_MAPPING(double,u_short)
		}
	} else if( OBJ_MACH_PREC(dest_dp) == PREC_BY || 
		OBJ_MACH_PREC(dest_dp) == PREC_UBY ){
		u_char *dstptr, *lutptr;
		dimension_t si;

		if( OBJ_PREC(src_dp) == PREC_UBY ){
			u_char *srcptr;
			DO_MAPPING(u_char,u_char)
		} else if( OBJ_PREC(src_dp) == PREC_UIN ){
			u_short *srcptr;
			DO_MAPPING(u_char,u_short)
		}
	} else {
		sprintf(ERROR_STRING,
			"Sorry, unhandled destination precision (%s) for lutmap",OBJ_PREC_NAME(dest_dp));
		WARN(ERROR_STRING);
		return(-1);
	}
	return(0);
}

