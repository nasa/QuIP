/* These functions work for integers AND bitmaps */

/* This is a file which gets included in other files...
 * To implement each precision, we first include a file
 * defining all the macros, then we include this file.
 */

/* intvec.c
 *
 * these are functions which are implemented for ALL integer precisions.
 */

/* Real only */

_VEC_FUNC_3V( vand ,	dst = (dest_type) ( src1 & src2	)	)
_VEC_FUNC_3V( vnand ,	dst = (dest_type) ( ~(src1 & src2) )	)
_VEC_FUNC_3V( vor ,		dst = (dest_type) ( src1 | src2	)	)
_VEC_FUNC_3V( vxor ,	dst = (dest_type) ( src1 ^ src2	)	)
_VEC_FUNC_2V( vnot ,		dst = (dest_type) ( ~src1 )		)
_VEC_FUNC_2V( vcomp ,		dst = (dest_type) ( ~src1 )		)
_VEC_FUNC_2V_SCAL( vsand ,	dst = (dest_type) ( src1 & scalar1_val ) )
dnl /*_VEC_FUNC_2V_SCAL( vsnand , dst = (dest_type) ( ~(src1 & scalar1_val))) */
_VEC_FUNC_2V_SCAL( vsor ,	dst = (dest_type) ( src1 | scalar1_val ) )
_VEC_FUNC_2V_SCAL( vsxor ,	dst = (dest_type) ( src1 ^ scalar1_val ) )


