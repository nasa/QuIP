
/* This is a file which gets included in other files...
 * To implement each precision, we first include a file
 * defining all the macros, then we include this file.
 */

/* intvec.c
 *
 * these are functions which are implemented for ALL integer precisions.
 */

/* Real only */

THREE_VEC_METHOD( vand ,	dst = src1 & src2		)
THREE_VEC_METHOD( vnand ,	dst = ~(src1 & src2)		)
THREE_VEC_METHOD( vor ,		dst = src1 | src2		)
THREE_VEC_METHOD( vxor ,	dst = src1 ^ src2		)
THREE_VEC_METHOD( vmod ,	dst = src1 % src2		)
THREE_VEC_METHOD( vshr ,	dst = src1 >> src2		)

TWO_VEC_METHOD( vnot ,		dst = ~src1			)
TWO_VEC_METHOD( vcomp ,		dst = ~src1			)

TWO_VEC_SCALAR_METHOD( vsand ,	dst = src1 & scalar1_val		)
TWO_VEC_SCALAR_METHOD( vsnand ,	dst = ~(src1 & scalar1_val)	)
TWO_VEC_SCALAR_METHOD( vsor ,	dst = src1 | scalar1_val		)
TWO_VEC_SCALAR_METHOD( vsxor ,	dst = src1 ^ scalar1_val		)
TWO_VEC_SCALAR_METHOD( vsmod ,	dst = src1 % scalar1_val		)
TWO_VEC_SCALAR_METHOD( vsmod2 ,	dst = scalar1_val % src1		)
TWO_VEC_SCALAR_METHOD( vsshr ,	dst = src1 >> scalar1_val	)
TWO_VEC_SCALAR_METHOD( vsshr2 ,	dst = scalar1_val >> src1	)

#ifndef GPU_FUNCTION
THREE_VEC_METHOD( vshl ,	dst = src1 << src2		)
TWO_VEC_SCALAR_METHOD( vsshl ,	dst = src1 << scalar1_val	)
TWO_VEC_SCALAR_METHOD( vsshl2 ,	dst = scalar1_val << src1	)
#endif /* ! GPU_FUNCTION */

