#include "quip_config.h"

#include "quip_prot.h"
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
		if( OBJ_TYPE_DIM(src_dp,i) != OBJ_TYPE_DIM(dst_dp,i) ){

			/* special case for real/cpx fft */

			/* This is usually a row, but for a 1-D fft of a column, this test wouldn't
			 * be triggered !?  BUG
			 */
			if( i==1 ){
				if( (argtyp & FWD_FT) && IS_REAL(src_dp) && IS_COMPLEX(dst_dp) ){
					if( OBJ_COLS(dst_dp) == (1+OBJ_COLS(src_dp)/2) )
						continue;
				} else if( (argtyp & INV_FT) && IS_COMPLEX(src_dp) && IS_REAL(dst_dp) ){
					if( OBJ_COLS(src_dp) == (1+OBJ_COLS(dst_dp)/2) )
						continue;
				}
			}

			/* if we get to here, the dimensions don't match... */
			/* if the source dimension is 1, it may be an outer op */
			/* if the destination dimension is 1, it may be a projection op */
fprintf(stderr,"checking dimension of source object %s\n",OBJ_NAME(src_dp));
			if( OBJ_TYPE_DIM(src_dp,i) == 1 /* && (argtyp&VV_SOURCES) == VV_SOURCES */ ){
				/* vmul, vadd, vsub, vatan2 */
				/* vvm_gt etc also */
				/* don't need VV_SOURCES... */
				continue;
			} else if( OBJ_TYPE_DIM(dst_dp,i) == 1 && CAN_PROJECT(argtyp) ){
				/* vsum, vmaxv, vmainv, etc */
				continue;
			} else {
				/* if we get to here, we're not happy... */
				sprintf(ERROR_STRING,
					"cksiz:  %s count mismatch, %s (%d) & %s (%d)",
					dimension_name[i],
					OBJ_NAME(src_dp),OBJ_TYPE_DIM(src_dp,i),
					OBJ_NAME(dst_dp),OBJ_TYPE_DIM(dst_dp,i));
				warn(ERROR_STRING);
				return(-1);
			}
		} else {	/* dimensions match */

			/* special case for real/cpx fft */

			/* This is usually a row, but for a 1-D fft of a column, this test wouldn't
			 * be triggered !?  BUG
			 */
			if( i==1 ){
				if( (argtyp & FWD_FT) && IS_REAL(src_dp) && IS_COMPLEX(dst_dp) ){
					if( OBJ_COLS(dst_dp) != (1+OBJ_COLS(src_dp)/2) ){
						sprintf(ERROR_STRING,
"For FFT, number of columns of transform %s (%d) should 1 plus the half number of columns of the destination %s (%d)",
OBJ_NAME(dst_dp),OBJ_COLS(dst_dp),OBJ_NAME(src_dp),OBJ_COLS(src_dp));
						warn(ERROR_STRING);
						return -1;
					}
				} else if( (argtyp & INV_FT) && IS_COMPLEX(src_dp) && IS_REAL(dst_dp) ){
					if( OBJ_COLS(src_dp) == (1+OBJ_COLS(dst_dp)/2) ){
						sprintf(ERROR_STRING,
"For inverse FFT, number of columns of transform %s (%d) should 1 plus the half number of columns of the destination %s (%d)",
OBJ_NAME(src_dp),OBJ_COLS(src_dp),OBJ_NAME(dst_dp),OBJ_COLS(dst_dp));
						warn(ERROR_STRING);
						return -1;
					}
				}
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
		if( OBJ_TYPE_DIM(src_dp,i) != OBJ_TYPE_DIM(dst_dp,i) ){
			/* if we get to here, we're not happy... */
			sprintf(ERROR_STRING,
				"old_cksiz:  %s count mismatch, %s (%d) & %s (%d)",
				dimension_name[i],
				OBJ_NAME(src_dp),OBJ_TYPE_DIM(src_dp,i),
				OBJ_NAME(dst_dp),OBJ_TYPE_DIM(dst_dp,i));
			warn(ERROR_STRING);
			return(-1);
		}
	}
	return(0);
} /* end old_cksiz() */

// BUG - Please add a comment concerning the purpose of this function!

int check_bitmap(QSP_ARG_DECL  Data_Obj *bitmap,Data_Obj *dst_dp)
{
	if( bitmap==NULL ) {
		error1("no bitmap???");
		IOS_RETURN_VAL(-1)
	}
	// BUG?  Is this code correct for a "gappy" bitmap?
	if( (OBJ_N_TYPE_ELTS(bitmap) * BITS_PER_BITMAP_WORD ) < OBJ_N_TYPE_ELTS(dst_dp) ){
		warn("bitmap size too small");
		return(-1);
	}
	return(0);
}

