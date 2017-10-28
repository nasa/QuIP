#include "quip_config.h"

/**		get_obj.c	user interface for interactive progs	*/

#include <stdio.h>
#include "quip_prot.h"
#include "data_obj.h"

// what about bitmaps???

int is_evenly_spaced(Data_Obj *dp)
{
	/* returns 1 if all the data can be accessed with a single increment */
	/* This means we can treat a multi-dimensional object as a 1-D vector */
	int i,n,spacing;

	/* New logic to handle case when increments are 0 if the corresponding dimension is 1 */

	/* mindim is the smallest indexable dimension - but for complex,
	 * it is always equal to 1 with an increment of 0...
	 */
	if( IS_SCALAR(dp) ){
		SET_SHP_EQSP_INC(OBJ_SHAPE(dp),1);
		return 1;	// true
	}


	/* a complex vector with length of 1 is not flagged as a scalar... */
	if( OBJ_N_TYPE_ELTS(dp) == 1 ){
		SET_SHP_EQSP_INC(OBJ_SHAPE(dp),1);
		return 1;	// true
	}

	spacing = 0;
	i=OBJ_MINDIM(dp)-1;
	while(spacing==0){
		i++;
		assert( i < N_DIMENSIONS );
		spacing = OBJ_TYPE_INC(dp,i);
	}
	n=OBJ_TYPE_DIM(dp,i);
	i++;
	for(;i<=OBJ_MAXDIM(dp);i++){
		if( OBJ_TYPE_INC(dp,i) != 0 ){
			if( OBJ_TYPE_INC(dp,i) != spacing * n ){
				SET_SHP_EQSP_INC(OBJ_SHAPE(dp),0);
				return 0;
			}
		}
		n *= OBJ_TYPE_DIM(dp,i);
	}

#ifdef PAD_BITMAP
	if( IS_BITMAP(dp) ){
		// bitmaps can seem contiguous, but are not if the row length is not a multiple of the word size (in bits)
		if( (OBJ_TYPE_DIM(dp,1)*OBJ_TYPE_INC(dp,1)) % BITS_PER_BITMAP_WORD != 0 ){
			SET_SHP_EQSP_INC(OBJ_SHAPE(dp),0);
			return 0;
		}
	}
#endif // PAD_BITMAP
	SET_SHP_EQSP_INC(OBJ_SHAPE(dp),spacing);
	return 1;
}

int _is_contiguous(QSP_ARG_DECL  Data_Obj *dp)
{
	assert( OBJ_FLAGS(dp) & DT_CHECKED );
	return(IS_CONTIGUOUS(dp));
}

static inline int bitmap_is_contiguous(Data_Obj *dp)
{
	int i_dim;
	dimension_t n;
	bitnum_t n_words;
	int inc;

	if( OBJ_TYPE_INC(dp,OBJ_MINDIM(dp)) != 1 ) return 0;
	n=OBJ_TYPE_DIM(dp,OBJ_MINDIM(dp));
	n_words = (bitnum_t)(( OBJ_BIT0(dp) + n + BITS_PER_BITMAP_WORD - 1 )/BITS_PER_BITMAP_WORD);
	for(i_dim=OBJ_MINDIM(dp)+1;i_dim<N_DIMENSIONS;i_dim++){
		if( OBJ_TYPE_DIM(dp,i_dim) != 1 ){
			inc = OBJ_TYPE_INC(dp,i_dim);
			if( inc != n_words * BITS_PER_BITMAP_WORD ) return 0;
			n_words *= OBJ_TYPE_DIM(dp,i_dim);
		}
	}
	/* We cache the status when called from
	 * check_contiguity()...
	 */
	return 1;
}

/* We have a special case for bitmaps:
 * A bitmap may be padded so that each row has an integral number of words.
 * This makes it non-contiguous from the point of view of bits,
 * but it is still contiguous from the point of view of transferring data.
 */

static int has_contiguous_data(QSP_ARG_DECL  Data_Obj *dp)
{
	if( IS_BITMAP(dp) ){
		return bitmap_is_contiguous(dp);
	} else {
		return(IS_CONTIGUOUS(dp));
	}
}

void check_contiguity(Data_Obj *dp)
{
	int i,inc;

	SET_OBJ_FLAG_BITS(dp, DT_CHECKED);
	CLEAR_OBJ_FLAG_BITS(dp,DT_EVENLY|DT_CONTIG);

	if( !is_evenly_spaced(dp) ){
		if( IS_BITMAP(dp) && has_contiguous_data(DEFAULT_QSP_ARG  dp) )
			SET_OBJ_FLAG_BITS(dp, DT_CONTIG_BITMAP_DATA);
		return;
	}

	SET_OBJ_FLAG_BITS(dp, DT_EVENLY);

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
	i=OBJ_MINDIM(dp);
	inc=0;

	// Most non-contiguous cases will get caught by the non-evenly spaced test above.
	// The only way an evenly-spaced object can be non-contiguous is if the smallest
	// increment is not 1, so we only have to check the smallest increment.
	// Note that if the dimension is equal to 1, then the increment is 0.
	while( inc==0 && i<N_DIMENSIONS ){
		inc = OBJ_TYPE_INC(dp,i);
		if( inc > 0 && inc != 1 ){
			return;	/* not contiguous */
		}
		i++;
	}

	SET_OBJ_FLAG_BITS(dp, DT_CONTIG);
}

