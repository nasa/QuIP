#include "quip_config.h"

char VersionId_dataf_contig[] = QUIP_VERSION_STRING;
/**		get_obj.c	user interface for interactive progs	*/

#include "data_obj.h"
#include "debug.h"
#include <stdio.h>

int is_evenly_spaced(Data_Obj *dp)
{
	/* returns 1 if all the data can be accessed with a single increment */
	int i,n,spacing;

	/* New logic to handle case when increments are 0
	 * if the corresponding dimension is 1
	 */

	/* mindim is the smallest indexable dimension - but for complex,
	 * it is always equal to 1 with an increment of 0...
	 */

	/*
	 * Do we need a special case for bitmaps???
	 * We do this with type sizes and increments,
	 * should we instead be using machine sizes and increments?
	 */

	if( IS_SCALAR(dp) ) return(1);

	/* Find the smallest spacing */
	spacing = 0;
	i=dp->dt_mindim-1;
	while(spacing==0){
		i++;
#ifdef CAUTIOUS
		if( i >= N_DIMENSIONS ){
			sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  is_evenly_spaced %s:  spacing is 0!?",
				dp->dt_name);
			NERROR1(DEFAULT_ERROR_STRING);
		}
#endif /* CAUTIOUS */
		spacing = dp->dt_type_inc[i];
	}
	n=dp->dt_type_dim[i];
	i++;
	for(;i<=dp->dt_maxdim;i++){
		if( dp->dt_type_inc[i] != 0 ){
			if( dp->dt_type_inc[i] != spacing * n ){
				return(0);
			}
		}
		n *= dp->dt_type_dim[i];
	}

	return(1);
}

/* Why is this function CAUTIOUS? */

int is_contiguous(Data_Obj *dp)
{
#ifdef CAUTIOUS
	if( (dp->dt_flags&DT_CHECKED) == 0 ){
		sprintf(DEFAULT_ERROR_STRING,
		"CAUTIOUS:  object \"%s\" not checked for contiguity!?",dp->dt_name);
		advise(DEFAULT_ERROR_STRING);
		check_contiguity(dp);
	}
#endif /* CAUTIOUS */
	return(IS_CONTIGUOUS(dp));
}

/* We have a special case for bitmaps:
 * A bitmap may be padded so that each row has an integral number of words.
 * This makes it non-contiguous from the point of view of bits,
 * but it is still contiguous from the point of view of transferring data.
 */

int has_contiguous_data(Data_Obj *dp)
{
	if( IS_BITMAP(dp) ){
		int i_dim,n,n_words,inc;
		/* We should cache the status instead
		 * of recomputing every time, but we're
		 * running out of flag bits...
		 */
		if( dp->dt_type_inc[dp->dt_mindim] != 1 ) return(0);
		n=dp->dt_type_dim[dp->dt_mindim];
		n_words = (dp->dt_bit0 + n + BITS_PER_BITMAP_WORD -1 )/BITS_PER_BITMAP_WORD;
		for(i_dim=dp->dt_mindim+1;i_dim<N_DIMENSIONS;i_dim++){
			if( dp->dt_type_dim[i_dim] != 1 ){
				inc = dp->dt_type_inc[i_dim];
				if( inc != n_words * BITS_PER_BITMAP_WORD ) return(0);
				n_words *= dp->dt_type_dim[i_dim];
			}
		}
		return(1);
	} else {
		return(IS_CONTIGUOUS(dp));
	}
}

void check_contiguity(Data_Obj *dp)
{
	int i,inc;

	dp->dt_flags |= DT_CHECKED;
	dp->dt_flags &= ~(DT_EVENLY|DT_CONTIG);

	if( !is_evenly_spaced(dp) ) return;

	dp->dt_flags |= DT_EVENLY;

	/* if the base increment is -1 the object may still be contiguous,
	 * but here we will take contiguity to mean that it is contiguous
	 * AND is addressed upward from the base.
	 *
	 * This test fails for scalars, so we make then a special case.
	 * Why does it fail for scalars???  mindim/maxdim set to crazy values?
	 */


	/* Find the first non-zero increment
	 *
	 * This used to be mindim, but with separate type and machine increments
	 * that is no longer true.
	 */
	i=dp->dt_mindim;
	inc=0;
	while( inc==0 && i<N_DIMENSIONS ){
		inc = dp->dt_type_inc[i];
		if( inc > 0 && inc != 1 ) return;	/* not contiguous */
		i++;
	}

	dp->dt_flags |= DT_CONTIG;
}

