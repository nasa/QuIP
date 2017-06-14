/* This is a file which gets included in other files...
 * To implement each precision, we first include a file
 * defining all the macros, then we include this file.
 *
 * For gpu implementation, some functions need different definitions...
 */

my_include(`veclib/real_args.m4')


// For sum, we may want to accumulate to a higher precision destination...

/* all_vec.m4  `type_code =' type_code */
_VEC_FUNC_2V_PROJ( rvsum, dst = (dest_type)  0, dst += (dest_type)  src1, psrc1 + psrc2)

_VEC_FUNC_3V_SSE( rvmul, src1 * src2 )
_VEC_FUNC_3V_SSE( rvadd, src1 + src2 )
_VEC_FUNC_3V_SSE( rvsub, src1 - src2 )
_VEC_FUNC_3V_SSE( rvdiv, src1 / src2 )

_VEC_FUNC_2V( rvsqr, dst = (dest_type)(src1 * src1) )

_VEC_FUNC_2V_SCAL( rvsadd, dst = (dest_type)(scalar1_val + src1) )
_VEC_FUNC_2V_SCAL( rvssub, dst = (dest_type)(scalar1_val - src1) )
_VEC_FUNC_2V_SCAL( rvssub2, dst = (dest_type)(src1 - scalar1_val) )
_VEC_FUNC_2V_SCAL( rvsmul, dst = (dest_type)(src1 * scalar1_val) )
_VEC_FUNC_2V_SCAL( rvsdiv, dst = (dest_type)(scalar1_val / src1) )
_VEC_FUNC_2V_SCAL( rvsdiv2, dst = (dest_type)(src1 / scalar1_val) )

_VEC_FUNC_3V_PROJ( rvdot, dst = (dest_type)  0, dst += (dest_type)  src1 * src2, psrc1 * psrc2, psrc1 + psrc2)

_VEC_FUNC_3V( vcmp, dst= (dest_type) (src1>src2?1:0) )
_VEC_FUNC_2V_SCAL( vscmp, dst= (dest_type) src1>=scalar1_val?1:0 )
_VEC_FUNC_2V_SCAL( vscmp2, dst= (dest_type) src1<=scalar1_val?1:0 )



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
 *
 * Later comment:  I don't know what was actually done - does that comment describe an 
 * idea, or an implementation?
 *
 * These functions are handled differently in parallel GPU implementations...
 * It would be nice to unify this.  The gpu implementation works recursively,
 * we do one pass where pairs of elements are compared, with the result stored
 * in a half-size temporary object.  This repeats until we are left with one value.
 * The number of tests is the same, but there's more data movement, and more temporary
 * storage.
 */

// Do these functions have fast and slow versions?

_VEC_FUNC_MM_NOCC( vmaxg, src1==extval, src1>extval, extval= src1, src_vals[IDX2]>src_vals[IDX2+1],src_vals[IDX2]<src_vals[IDX2+1])
_VEC_FUNC_MM_NOCC( vming, src1==extval, src1<extval, extval= src1, src_vals[IDX2]<src_vals[IDX2+1],src_vals[IDX2]>src_vals[IDX2+1])

dnl	VEC_FUNC_MM_NOCC(vmaxg,src_vals[IDX2]>src_vals[IDX2+1],src_vals[IDX2]<src_vals[IDX2+1])
dnl	VEC_FUNC_MM_NOCC(vming,src_vals[IDX2]<src_vals[IDX2+1],src_vals[IDX2]>src_vals[IDX2+1])

/* used to be EXTREME_VALUE_METHOD, but logic incorporating projection operation
 * (dimension collapsing) was brought in from the java macros.
 * Now would be a nice time to merge some of the macros...
 * m4 may be preferred over cpp because we lose the backslashes...
 *
 * BUG we need to do the same thing for other EXTREME methods, as above...
 */

_VEC_FUNC_3V( vmax,  dst = (dest_type)( src1 > src2 ? src1 : src2 ) )
_VEC_FUNC_3V( vmin, dst = (dest_type)( src1 < src2 ? src1 : src2 ) )

_VEC_FUNC_2V_SCAL( vsmax, dst = (dest_type)(scalar1_val > src1 ? scalar1_val : src1 ) )
_VEC_FUNC_2V_SCAL( vsmin, dst = (dest_type)(scalar1_val < src1 ? scalar1_val : src1 ) )



/* Implemented for real and complex
 *
 * Here are the real versions.
 */

dnl	// vnot used to be here, but now it is integer-only...

dnl	// With the new style conversions, we don't need vmov any more,
dnl	// as we have same type conversions... 

dnl	// vmov used to be defined differently, presumably to do
dnl	// fast moves of contiguous objects...
dnl	// We should bring that back, if possible

dnl	// Ramp functions are slow - only...

dnl	This is wrong, because IDX1 is used to index the destination, but can
dnl	increase by an increment other than 1 (evenly-spaced, subimage, etc)

_VEC_FUNC_1V_2SCAL( vramp1d, dst = (dest_type)scalar1_val; scalar1_val+=scalar2_val,
				dst = scalar1_val + RAMP_IDX * scalar2_val )

// Why are stat1, stat2 not used?
// cpu implementation?

_VEC_FUNC_1V_3SCAL( vramp2d, stat1, stat2, dst = scalar1_val + scalar2_val * (IDX1_1 / INC1_1 ) + scalar3_val * (IDX1_2 / INC1_2 ))

// How do we handle bit precision?

// rvset moved to all_same_prec_vec.c

/* New conditional assignments */

_VEC_FUNC_5V( vv_vv_lt, dst = (dest_type) ( src3 < src4 ? src1 : src2 ) )
_VEC_FUNC_5V( vv_vv_gt, dst = (dest_type) ( src3 > src4 ? src1 : src2 ) )
_VEC_FUNC_5V( vv_vv_le, dst = (dest_type) ( src3 <= src4 ? src1 : src2 ) )
_VEC_FUNC_5V( vv_vv_ge, dst = (dest_type) ( src3 >= src4 ? src1 : src2 ) )
_VEC_FUNC_5V( vv_vv_eq, dst = (dest_type) ( src3 == src4 ? src1 : src2 ) )
_VEC_FUNC_5V( vv_vv_ne, dst = (dest_type) ( src3 != src4 ? src1 : src2 ) )

_VEC_FUNC_4V_SCAL( vv_vs_lt, dst = (dest_type) ( src3 < scalar1_val ? src1 : src2 ) )
_VEC_FUNC_4V_SCAL( vv_vs_gt, dst = (dest_type) ( src3 > scalar1_val ? src1 : src2 ) )
_VEC_FUNC_4V_SCAL( vv_vs_le, dst = (dest_type) ( src3 <= scalar1_val ? src1 : src2 ) )
_VEC_FUNC_4V_SCAL( vv_vs_ge, dst = (dest_type) ( src3 >= scalar1_val ? src1 : src2 ) )
_VEC_FUNC_4V_SCAL( vv_vs_eq, dst = (dest_type) ( src3 == scalar1_val ? src1 : src2 ) )
_VEC_FUNC_4V_SCAL( vv_vs_ne, dst = (dest_type) ( src3 != scalar1_val ? src1 : src2 ) )

_VEC_FUNC_4V_SCAL( vs_vv_lt, dst = (dest_type) ( src2 < src3 ? src1 : scalar1_val ) )
_VEC_FUNC_4V_SCAL( vs_vv_gt, dst = (dest_type) ( src2 > src3 ? src1 : scalar1_val ) )
_VEC_FUNC_4V_SCAL( vs_vv_le, dst = (dest_type) ( src2 <= src3 ? src1 : scalar1_val ) )
_VEC_FUNC_4V_SCAL( vs_vv_ge, dst = (dest_type) ( src2 >= src3 ? src1 : scalar1_val ) )
_VEC_FUNC_4V_SCAL( vs_vv_eq, dst = (dest_type) ( src2 == src3 ? src1 : scalar1_val ) )
_VEC_FUNC_4V_SCAL( vs_vv_ne, dst = (dest_type) ( src2 != src3 ? src1 : scalar1_val ) )

_VEC_FUNC_3V_2SCAL( vs_vs_lt, dst = (dest_type)  ( src2 < scalar2_val ? src1 : scalar1_val ) )
_VEC_FUNC_3V_2SCAL( vs_vs_gt, dst = (dest_type)  ( src2 > scalar2_val ? src1 : scalar1_val ) )
_VEC_FUNC_3V_2SCAL( vs_vs_le, dst = (dest_type)  ( src2 <= scalar2_val ? src1 : scalar1_val ) )
_VEC_FUNC_3V_2SCAL( vs_vs_ge, dst = (dest_type)  ( src2 >= scalar2_val ? src1 : scalar1_val ) )
_VEC_FUNC_3V_2SCAL( vs_vs_eq, dst = (dest_type)  ( src2 == scalar2_val ? src1 : scalar1_val ) )
_VEC_FUNC_3V_2SCAL( vs_vs_ne, dst = (dest_type)  ( src2 != scalar2_val ? src1 : scalar1_val ) )

_VEC_FUNC_3V_2SCAL( ss_vv_lt, dst = (dest_type) ( src1 < src2 ? scalar1_val : scalar2_val ) )
_VEC_FUNC_3V_2SCAL( ss_vv_gt, dst = (dest_type) ( src1 > src2 ? scalar1_val : scalar2_val ) )
_VEC_FUNC_3V_2SCAL( ss_vv_le, dst = (dest_type) ( src1 <= src2 ? scalar1_val : scalar2_val ) )
_VEC_FUNC_3V_2SCAL( ss_vv_ge, dst = (dest_type) ( src1 >= src2 ? scalar1_val : scalar2_val ) )
_VEC_FUNC_3V_2SCAL( ss_vv_eq, dst = (dest_type) ( src1 == src2 ? scalar1_val : scalar2_val ) )
_VEC_FUNC_3V_2SCAL( ss_vv_ne, dst = (dest_type) ( src1 != src2 ? scalar1_val : scalar2_val ) )

_VEC_FUNC_2V_3SCAL( ss_vs_lt, dst = (dest_type) ( src1 < scalar3_val ? scalar1_val : scalar2_val ) )
_VEC_FUNC_2V_3SCAL( ss_vs_gt, dst = (dest_type) ( src1 > scalar3_val ? scalar1_val : scalar2_val ) )
_VEC_FUNC_2V_3SCAL( ss_vs_le, dst = (dest_type) ( src1 <= scalar3_val ? scalar1_val : scalar2_val ) )
_VEC_FUNC_2V_3SCAL( ss_vs_ge, dst = (dest_type) ( src1 >= scalar3_val ? scalar1_val : scalar2_val ) )
_VEC_FUNC_2V_3SCAL( ss_vs_eq, dst = (dest_type) ( src1 == scalar3_val ? scalar1_val : scalar2_val ) )
_VEC_FUNC_2V_3SCAL( ss_vs_ne, dst = (dest_type) ( src1 != scalar3_val ? scalar1_val : scalar2_val ) )


_VEC_FUNC_2V_PROJ( vmaxv, dst = (dest_type) src1, if( src1 > dst ) dst = (dest_type)  src1, psrc1 > psrc2 ? psrc1 : psrc2 )

_VEC_FUNC_2V_PROJ( vminv, dst = (dest_type) src1, if( src1 < dst ) dst = (dest_type)  src1, psrc1 < psrc2 ? psrc1 : psrc2 )


dnl VEC_FUNC_MM_IND(vmaxi, dst = (src1 > src2 ? IDX2 : IDX3+len1), dst = (orig[src1] > orig[src2] ? src1 : src2) )
dnl VEC_FUNC_MM_IND(vmini, dst = (src1 < src2 ? IDX2 : IDX3+len1), dst = (orig[src1] < orig[src2] ? src1 : src2) )


dnl _VF_2V_PROJ_IDX( name, cpu_s1, cpu_s2, gpu_s1, gpu_s2 )

_VEC_FUNC_2V_PROJ_IDX( vmaxi, dst = index_base[0], tmp_ptr = INDEX_VDATA(dst); if ( src1 > *tmp_ptr ) dst=index_base[0], dst = (src1 > src2 ? IDX2 : IDX3+len1), dst = (orig[src1] > orig[src2] ? src1 : src2))

_VEC_FUNC_2V_PROJ_IDX( vmini, dst = index_base[0], tmp_ptr = INDEX_VDATA(dst); if( src1 < *tmp_ptr ) dst=index_base[0], dst = (src1 < src2 ? IDX2 : IDX3+len1), dst = (orig[src1] < orig[src2] ? src1 : src2))

/* vmini DONE */


ifdef(`BUILD_FOR_GPU',`',`	dnl	ifndef BUILD_FOR_GPU
_VEC_FUNC_2V( rvrand, dst = (dest_type) rn((u_long)src1)	)
')				dnl	endif /* ! BUILD_FOR_GPU */

/* vvm_le BEGIN */
_VEC_FUNC_VVMAP( vvm_le, <= )
/* vvm_le DONE */
_VEC_FUNC_VVMAP( vvm_ge, >= )
_VEC_FUNC_VVMAP( vvm_lt, <  )
_VEC_FUNC_VVMAP( vvm_gt, >  )
_VEC_FUNC_VVMAP( vvm_ne, != )
_VEC_FUNC_VVMAP( vvm_eq, == )


/* bitmap, scalar magnitude compare */

_VEC_FUNC_VSMAP( vsm_gt, > )
_VEC_FUNC_VSMAP( vsm_lt, < )
_VEC_FUNC_VSMAP( vsm_ge, >= )
_VEC_FUNC_VSMAP( vsm_le, <= )
_VEC_FUNC_VSMAP( vsm_ne, != ) 
_VEC_FUNC_VSMAP( vsm_eq, == )

// gpu versions appear compatible...
dnl VEC_FUNC_VVSLCT( vvv_slct, dst = srcbit ? src1 : src2 )
dnl VEC_FUNC_VSSLCT( vvs_slct, dst = srcbit ? src1 : scalar1_val )
dnl VEC_FUNC_SSSLCT( vss_slct, dst = srcbit ? scalar1_val : scalar2_val )




dnl #define VV_SELECTION_METHOD( name, stat )	_VF_VVSLCT( name, type_code, stat)
_VEC_FUNC_VVSLCT( rvvv_slct, dst = (dest_type) ( srcbit ? src1 : src2 ) )
_VEC_FUNC_VSSLCT( rvvs_slct, dst = (dest_type) ( srcbit ? src1 : scalar1_val ) )
_VEC_FUNC_SSSLCT( rvss_slct, dst = (dest_type) ( srcbit ? scalar1_val : scalar2_val ) )

