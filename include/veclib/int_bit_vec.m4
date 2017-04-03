dnl	/* These functions work for integers AND bitmaps */
dnl
dnl	/* This is a file which gets included in other files...
dnl	 * To implement each precision, we first include a file
dnl	 * defining all the macros, then we include this file.
dnl	 */
dnl
dnl	/* intvec.c
dnl	 *
dnl	 * these are functions which are implemented for ALL integer precisions.
dnl	 */

dnl	/* Real only */

/* testing: bit_precision_kernel = BIT_PRECISION_KERNEL */

ifdef(`BIT_PRECISION_KERNEL',`

dnl	Special kernels for bit precision

/* bit precision vand */

_VEC_FUNC_DBM_2SBM(vand,SET_DBM_BIT(srcbit1 && srcbit2) )
_VEC_FUNC_DBM_2SBM(vnand,SET_DBM_BIT(!(srcbit1 && srcbit2)))
_VEC_FUNC_DBM_2SBM(vor,SET_DBM_BIT(srcbit1 || srcbit2) )
_VEC_FUNC_DBM_2SBM(vxor,SET_DBM_BIT((srcbit1&&!srcbit2)||((!srcbit1)&&srcbit2)))
_VEC_FUNC_DBM_1SBM(vnot,SET_DBM_BIT(!srcbit1))
_VEC_FUNC_DBM_1SBM(vcomp,SET_DBM_BIT(!srcbit1))
_VEC_FUNC_DBM_1SBM_1S(vsand,SET_DBM_BIT(srcbit1&&scalar1_val))
dnl	vsnand???
_VEC_FUNC_DBM_1SBM_1S(vsor,SET_DBM_BIT(srcbit1||scalar1_val))
_VEC_FUNC_DBM_1SBM_1S(vsxor,SET_DBM_BIT((srcbit1&&!scalar1_val)||((!srcbit1)&&scalar1_val)))

',`	dnl	! BIT_PRECISION_KERNEL

dnl	Normal kernels for all integer types

/* integer precision vand */

_VEC_FUNC_3V( vand ,		dst = (dest_type) ( src1 & src2	)	)
_VEC_FUNC_3V( vnand ,		dst = (dest_type) ( ~(src1 & src2) )	)
_VEC_FUNC_3V( vor ,		dst = (dest_type) ( src1 | src2	)	)
_VEC_FUNC_3V( vxor ,		dst = (dest_type) ( src1 ^ src2	)	)
_VEC_FUNC_2V( vnot ,		dst = (dest_type) ( ~src1 )		)
_VEC_FUNC_2V( vcomp ,		dst = (dest_type) ( ~src1 )		)
_VEC_FUNC_2V_SCAL( vsand ,	dst = (dest_type) ( src1 & scalar1_val ) )
dnl /*_VEC_FUNC_2V_SCAL( vsnand , dst = (dest_type) ( ~(src1 & scalar1_val))) */
_VEC_FUNC_2V_SCAL( vsor ,	dst = (dest_type) ( src1 | scalar1_val ) )
_VEC_FUNC_2V_SCAL( vsxor ,	dst = (dest_type) ( src1 ^ scalar1_val ) )

')

