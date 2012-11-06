
/* This is a file which gets included in other files...
 * To implement each precision, we first include a file
 * defining all the macros, then we include this file.
 */

/* allvec.c
 *
 * these are functions which are implemented for ALL precisions.
 */

#include "debug.h"
#include "rn.h"		/* rn() */

/* Real only */

TWO_VEC_SCALAR_METHOD( vscmp , dst=src1>=scalar1_val?1:0 )
TWO_VEC_SCALAR_METHOD( vscmp2, dst=src1<=scalar1_val?1:0 )
THREE_VEC_METHOD( vcmp , dst=(src1>src2?1:0) )



/* With our new loop logic (which accepts as a shape match vector pairs where
 * one has been collapsed along a dimension), there is really no need for separate
 * vector-scalar operations (since a scalar is the extreme case of all dimensions
 * collapsed), although by retaining them there is an efficiency advantage since
 * we don't have to add the 0 increment to the scalar pointer each time through
 * the loop...
 */

/* EXTREMA_LOCATIONS_METHOD describes a method with multiple return vectors:
 * vmaxg returns the value of the max, the number of occurrences, and a vector
 * of indices of all the occurrences.  It is not obvious how we should generalize
 * this with our new projection operator logic used with vmax.  Here is the problem:
 * if we have an image, and we call vmaxv with a row vector for the destination vector,
 * then each element of the destination vector will be set to the max value of the corresponding
 * column of the input image.  That will work just fine for number of occurrences and
 * max values, but what dimension do we use to store the indices of the multiple occurrences?
 *
 * Here is one possibility:  say we have an image which is our input, and we pass row vectors
 * to hold the value and n_occurrences.  We require that these two have the same shape, and
 * this determines the nature of the projection.  In this example, the index array should have
 * the same number of columns (in general it should match on all non-collapsed dimensions),
 * and some number of rows which determines how many indices might be stored.
 *
 * Another example is a movie, where we might want to compute this for each frame,
 * we pass a movie of scalars for maxv and nocc, then the index array could either be
 * a regular movie or a movie of row vectors.  We will be permissive and not restrict
 * the values of non-collapsed dimensions...  Being this flexible does complicate
 * the index calculations...
 */


EXTREMA_LOCATIONS_METHOD( vmaxg, src1==extval, src1>extval, extval=src1 )
EXTREMA_LOCATIONS_METHOD( vming, src1==extval, src1<extval, extval=src1 )


/* used to be EXTREME_VALUE_METHOD, but logic incorporating projection operation
 * (dimension collapsing) was brought in from the java macros.
 * Now would be a nice time to merge some of the macros...
 * m4 may be preferred over cpp because we lose the backslashes...
 *
 * BUG we need to do the same thing for other EXTREME methods, as above...
 */

THREE_VEC_METHOD( vmax ,  dst = ( src1 > src2 ? src1 : src2 ) )
THREE_VEC_METHOD( vmin , dst = ( src1 < src2 ? src1 : src2 ) )

TWO_VEC_SCALAR_METHOD( vsmax , dst = (scalar1_val > src1 ? scalar1_val : src1 ) )
TWO_VEC_SCALAR_METHOD( vsmin , dst = (scalar1_val < src1 ? scalar1_val : src1 ) )



/* Implemented for real and complex
 *
 * Here are the real versions.
 */

#ifdef NOT_YET
TWO_VEC_METHOD( vnot , if( src1 == 0 ) dst =1; else dst=0; )
#endif
TWO_VEC_MOV_METHOD( rvmov , dst = src1 )
TWO_VEC_METHOD( rvsqr , dst = src1 * src1 )

THREE_VEC_METHOD( rvadd , dst = src1 + src2 )
THREE_VEC_METHOD( rvsub , dst = src1 - src2 )
THREE_VEC_METHOD( rvmul , dst = src1 * src2 )
THREE_VEC_METHOD( rvdiv , dst = src1 / src2 )

RAMP2D_METHOD( vramp2d )
ONE_VEC_2SCALAR_METHOD( vramp1d , dst = scalar1_val; scalar1_val+=scalar2_val )

TWO_VEC_SCALAR_METHOD( rvsadd , dst = scalar1_val + src1 )
TWO_VEC_SCALAR_METHOD( rvssub , dst = scalar1_val - src1 )
TWO_VEC_SCALAR_METHOD( rvsmul , dst = src1 * scalar1_val )
TWO_VEC_SCALAR_METHOD( rvsdiv , dst = scalar1_val / src1 )
TWO_VEC_SCALAR_METHOD( rvsdiv2 , dst = src1 / scalar1_val )

ONE_VEC_SCALAR_METHOD( rvset , dst = scalar1_val )
/* don't need this for all types */
/* SCALAR_BIT_METHOD( bvset , SETBIT( scalar1_val ) ) */

/* FUNC_DECL( rvsum ) { RSINIT1; *scalar = 0; V1LOOP( *scalar += src1 ) */

BITMAP_DST_TWO_VEC_METHOD( vvm_le , <= )
BITMAP_DST_TWO_VEC_METHOD( vvm_ge , >= )
BITMAP_DST_TWO_VEC_METHOD( vvm_lt , <  )
BITMAP_DST_TWO_VEC_METHOD( vvm_gt , >  )
BITMAP_DST_TWO_VEC_METHOD( vvm_ne , != )
BITMAP_DST_TWO_VEC_METHOD( vvm_eq , == )


/* New conditional assignments */

FIVE_VEC_METHOD( vv_vv_lt, dst = src3 < src4 ? src1 : src2 )
FIVE_VEC_METHOD( vv_vv_gt, dst = src3 > src4 ? src1 : src2 )
FIVE_VEC_METHOD( vv_vv_le, dst = src3 <= src4 ? src1 : src2 )
FIVE_VEC_METHOD( vv_vv_ge, dst = src3 >= src4 ? src1 : src2 )
FIVE_VEC_METHOD( vv_vv_eq, dst = src3 == src4 ? src1 : src2 )
FIVE_VEC_METHOD( vv_vv_ne, dst = src3 != src4 ? src1 : src2 )

FOUR_VEC_SCALAR_METHOD( vv_vs_lt, dst = src3 < scalar1_val ? src1 : src2 )
FOUR_VEC_SCALAR_METHOD( vv_vs_gt, dst = src3 > scalar1_val ? src1 : src2 )
FOUR_VEC_SCALAR_METHOD( vv_vs_le, dst = src3 <= scalar1_val ? src1 : src2 )
FOUR_VEC_SCALAR_METHOD( vv_vs_ge, dst = src3 >= scalar1_val ? src1 : src2 )
FOUR_VEC_SCALAR_METHOD( vv_vs_eq, dst = src3 == scalar1_val ? src1 : src2 )
FOUR_VEC_SCALAR_METHOD( vv_vs_ne, dst = src3 != scalar1_val ? src1 : src2 )

FOUR_VEC_SCALAR_METHOD( vs_vv_lt, dst = src2 < src3 ? src1 : scalar1_val )
FOUR_VEC_SCALAR_METHOD( vs_vv_gt, dst = src2 > src3 ? src1 : scalar1_val )
FOUR_VEC_SCALAR_METHOD( vs_vv_le, dst = src2 <= src3 ? src1 : scalar1_val )
FOUR_VEC_SCALAR_METHOD( vs_vv_ge, dst = src2 >= src3 ? src1 : scalar1_val )
FOUR_VEC_SCALAR_METHOD( vs_vv_eq, dst = src2 == src3 ? src1 : scalar1_val )
FOUR_VEC_SCALAR_METHOD( vs_vv_ne, dst = src2 != src3 ? src1 : scalar1_val )

THREE_VEC_2SCALAR_METHOD( vs_vs_lt, dst = src2 < scalar2_val ? src1 : scalar1_val )
THREE_VEC_2SCALAR_METHOD( vs_vs_gt, dst = src2 > scalar2_val ? src1 : scalar1_val )
THREE_VEC_2SCALAR_METHOD( vs_vs_le, dst = src2 <= scalar2_val ? src1 : scalar1_val )
THREE_VEC_2SCALAR_METHOD( vs_vs_ge, dst = src2 >= scalar2_val ? src1 : scalar1_val )
THREE_VEC_2SCALAR_METHOD( vs_vs_eq, dst = src2 == scalar2_val ? src1 : scalar1_val )
THREE_VEC_2SCALAR_METHOD( vs_vs_ne, dst = src2 != scalar2_val ? src1 : scalar1_val )

THREE_VEC_2SCALAR_METHOD( ss_vv_lt, dst = src1 < src2 ? scalar1_val : scalar2_val )
THREE_VEC_2SCALAR_METHOD( ss_vv_gt, dst = src1 > src2 ? scalar1_val : scalar2_val )
THREE_VEC_2SCALAR_METHOD( ss_vv_le, dst = src1 <= src2 ? scalar1_val : scalar2_val )
THREE_VEC_2SCALAR_METHOD( ss_vv_ge, dst = src1 >= src2 ? scalar1_val : scalar2_val )
THREE_VEC_2SCALAR_METHOD( ss_vv_eq, dst = src1 == src2 ? scalar1_val : scalar2_val )
THREE_VEC_2SCALAR_METHOD( ss_vv_ne, dst = src1 != src2 ? scalar1_val : scalar2_val )

TWO_VEC_3SCALAR_METHOD( ss_vs_lt, dst = src1 < scalar3_val ? scalar1_val : scalar2_val )
TWO_VEC_3SCALAR_METHOD( ss_vs_gt, dst = src1 > scalar3_val ? scalar1_val : scalar2_val )
TWO_VEC_3SCALAR_METHOD( ss_vs_le, dst = src1 <= scalar3_val ? scalar1_val : scalar2_val )
TWO_VEC_3SCALAR_METHOD( ss_vs_ge, dst = src1 >= scalar3_val ? scalar1_val : scalar2_val )
TWO_VEC_3SCALAR_METHOD( ss_vs_eq, dst = src1 == scalar3_val ? scalar1_val : scalar2_val )
TWO_VEC_3SCALAR_METHOD( ss_vs_ne, dst = src1 != scalar3_val ? scalar1_val : scalar2_val )


PROJECTION_METHOD_2( vmaxv , dst = src1 , if( src1 > dst ) dst = src1; )
PROJECTION_METHOD_2( vminv , dst = src1 , if( src1 < dst ) dst = src1; )

PROJECTION_METHOD_IDX_2( vmaxi ,
	dst = index_base[0] ,
	tmp_ptr = INDEX_VDATA(dst); if ( src1 > *tmp_ptr ) dst=index_base[0] )

PROJECTION_METHOD_IDX_2( vmini ,
	dst = index_base[0] ,
	tmp_ptr = INDEX_VDATA(dst); if( src1 < *tmp_ptr ) dst=index_base[0] )


PROJECTION_METHOD_2( rvsum, dst = 0, dst += src1 )

PROJECTION_METHOD_3( rvdot, dst = 0, dst += src1 * src2 )

#ifndef GPU_FUNCTION
TWO_VEC_METHOD( rvrand , dst = rn(src1)	)
#endif /* ! GPU_FUNCTION */

/* bitmap, scalar magnitude compare */

BITMAP_DST_ONE_VEC_SCALAR_METHOD( vsm_gt , > )
BITMAP_DST_ONE_VEC_SCALAR_METHOD( vsm_lt , < )
BITMAP_DST_ONE_VEC_SCALAR_METHOD( vsm_ge , >= )
BITMAP_DST_ONE_VEC_SCALAR_METHOD( vsm_le , <= )
BITMAP_DST_ONE_VEC_SCALAR_METHOD( vsm_ne , != ) 
BITMAP_DST_ONE_VEC_SCALAR_METHOD( vsm_eq , == )



VV_SELECTION_METHOD( rvvv_slct , dst = srcbit ? src1 : src2 )
VS_SELECTION_METHOD( rvvs_slct , dst = srcbit ? src1 : scalar1_val )
SS_SELECTION_METHOD( rvss_slct , dst = srcbit ? scalar1_val : scalar2_val )

BITMAP_DST_ONE_VEC_METHOD( vconv_to_bit, SETBIT( src1!=0 ) )
BITMAP_SRC_CONVERSION_METHOD( vconv_from_bit, if( srcbit ){ dst = 1; } else { dst = 0; } )


