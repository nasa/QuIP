#include "quip_config.h"

char VersionId_newvec_cksiz[] = QUIP_VERSION_STRING;


#include "nvf.h"

/* In the old veclib, operands were always expected to have matching sizes,
 * and outer ops were handled separately.
 *
 * Now all binary ops handle outer ops automatically; in the new cksiz,
 * the dimensions should either match, or have one equal to 1.
 *
 * The destination can be smaller only for ops like vmax, vsum, etc.
 * We let those go here, but we really should be passing the function also.
 * NO - we can use argtyp to determine this...
 *
 * How is the return value used? 
 */

int cksiz(QSP_ARG_DECL  int argtyp,Data_Obj *src_dp,Data_Obj *dst_dp)
{
	int i;

	/* allow exception for vmaxg, etc.
	 * These functions return two scalars, the max value,
	 * and number of occurrences, PLUS a vector with the
	 * indices of the occurrences.
	 *
	 * But where do we check them?  Do we need to do any checking for them?
	 */
	if( argtyp == V_SCALRET2 )
		return(0);

	for(i=0;i<N_DIMENSIONS;i++){
		if( src_dp->dt_type_dim[i] != dst_dp->dt_type_dim[i] ){

			/* special case for real/cpx fft */
			if( i==1 ){
				if( (argtyp & FWD_FT) && IS_REAL(src_dp) && IS_COMPLEX(dst_dp) ){
					if( dst_dp->dt_cols == (1+src_dp->dt_cols/2) )
						continue;
				} else if( (argtyp & INV_FT) && IS_COMPLEX(src_dp) && IS_REAL(dst_dp) ){
					if( src_dp->dt_cols == (1+dst_dp->dt_cols/2) )
						continue;
				}
			}

			/* if we get to here, the dimensions don't match... */
			/* if the source dimension is 1, it may be an outer op */
			/* if the destination dimension is 1, it may be a projection op */
			if( src_dp->dt_type_dim[i] == 1 /* && (argtyp&VV_SOURCES) == VV_SOURCES */ ){
				/* vmul, vadd, vsub, vatan2 */
				/* vvm_gt etc also */
				/* don't need VV_SOURCES... */
				continue;
			} else if( dst_dp->dt_type_dim[i] == 1 && CAN_PROJECT(argtyp) ){
				/* vsum, vmaxv, vmainv, etc */
				continue;
			} else {
				/* if we get to here, we're not happy... */
				sprintf(error_string,
					"cksiz:  %s count mismatch, %s (%d) & %s (%d)",
					dimension_name[i],
					src_dp->dt_name,src_dp->dt_type_dim[i],
					dst_dp->dt_name,dst_dp->dt_type_dim[i]);
				WARN(error_string);
				return(-1);
			}
		}
	}
	return(0);
} /* end cksiz() */


/* This is the old cksiz, which expects an exact match (except for type dim).
 * This is still used so we should find a better name - UGLY
 */

int old_cksiz(QSP_ARG_DECL  int argtyp,Data_Obj *src_dp,Data_Obj *dst_dp)
{
	int i;

	/* allow exception for vmaxg, etc. */
	if( argtyp == V_SCALRET2 )
		return(0);

	for(i=0;i<N_DIMENSIONS;i++){
		if( src_dp->dt_type_dim[i] != dst_dp->dt_type_dim[i] ){
			/* if we get to here, we're not happy... */
			sprintf(error_string,
				"old_cksiz:  %s count mismatch, %s (%d) & %s (%d)",
				dimension_name[i],
				src_dp->dt_name,src_dp->dt_type_dim[i],
				dst_dp->dt_name,dst_dp->dt_type_dim[i]);
			WARN(error_string);
			return(-1);
		}
	}
	return(0);
} /* end old_cksiz() */

int check_bitmap(QSP_ARG_DECL  Data_Obj *bitmap,Data_Obj *dst_dp)
{
	if( bitmap==NO_OBJ ) ERROR1("no bitmap???");
	if( (bitmap->dt_n_type_elts * BITS_PER_BITMAP_WORD ) < dst_dp->dt_n_type_elts ){
		WARN("bitmap size too small");
		return(-1);
	}
	return(0);
}

