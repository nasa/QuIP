
/* This is a file which gets included in other files...
 * To implement each precision, we first include a file
 * defining all the macros, then we include this file.
 */

/* intvec.c
 *
 * these are functions which are implemented for ALL integer precisions.
 */

#include "veclib/int_bit_vec.c"

/* Real only */

_VEC_FUNC_3V( vmod ,	dst = (dest_type) ( src1 % src2	)	)
_VEC_FUNC_3V( vshr ,	dst = (dest_type) ( src1 >> src2 )	)

_VEC_FUNC_2V_SCAL( vsmod ,	dst = (dest_type) ( src1 % scalar1_val ) )
_VEC_FUNC_2V_SCAL( vsmod2 ,	dst = (dest_type) ( scalar1_val % src1 ) )
_VEC_FUNC_2V_SCAL( vsshr ,	dst = (dest_type) ( src1 >> scalar1_val	) )
_VEC_FUNC_2V_SCAL( vsshr2 ,	dst = (dest_type) ( scalar1_val >> src1	) )

#ifdef BUILD_FOR_GPU

/* why no left-shift on GPU?? */
/* CUDA impementation is broken */
#ifdef BUILD_FOR_CUDA
/*
KERN_CALL_VV_LS(vshl, dst , src1 , src2)
KERN_CALL_VS_LS(vsshl, dst , src1 , scalar1_val)
KERN_CALL_VS_LS(vsshl2, dst , scalar1_val , src1)
*/
_VEC_FUNC_3V( vshl ,		dst = (dest_type)(src1 << src2)		)
_VEC_FUNC_2V_SCAL( vsshl ,	dst = (dest_type)(src1 << scalar1_val)	)
_VEC_FUNC_2V_SCAL( vsshl2 ,	dst = (dest_type)(scalar1_val << src1)	)
#else // ! BUILD_FOR_CUDA
_VEC_FUNC_3V( vshl ,		LSHIFT_SWITCH_32(dst,src1,src2)		)
_VEC_FUNC_2V_SCAL( vsshl ,	LSHIFT_SWITCH_32(dst,src1,scalar1_val)	)
_VEC_FUNC_2V_SCAL( vsshl2 ,	LSHIFT_SWITCH_32(dst,scalar1_val,src1)	)
#endif // ! BUILD_FOR_CUDA

/* ctype stuff... */
_VEC_FUNC_2V( vtolower ,	dst = (dest_type) ( src1 >= 'A' && src1 <= 'Z' ? src1 + /* ('a'-'A') */ 32 : src1 ) )
_VEC_FUNC_2V( vtoupper ,	dst = (dest_type) ( src1 >= 'a' && src1 <= 'z' ? src1 - /* ('a'-'A') */ 32 : src1 ) )
_VEC_FUNC_2V( vislower ,	dst = (dest_type) ( src1 >= 'a' && src1 <= 'z' ? 1 : 0 ) )
_VEC_FUNC_2V( visupper ,	dst = (dest_type) ( src1 >= 'A' && src1 <= 'Z' ? 1 : 0 ) )
_VEC_FUNC_2V( visalpha ,	dst = (dest_type) ( ((src1 >= 'A' && src1 <= 'Z')||(src1 >= 'a' && src1 <= 'z')) ? 1 : 0 ) )
_VEC_FUNC_2V( visdigit ,	dst = (dest_type) ( src1 >= '0' && src1 <= '9' ? 1 : 0 ) )
_VEC_FUNC_2V( visalnum ,	dst = (dest_type) ( ((src1>='0'&&src1<='9')||(src1 >= 'A' && src1 <= 'Z')||(src1 >= 'a' && src1 <= 'z')) ? 1 : 0 ) )
_VEC_FUNC_2V( viscntrl ,	dst = (dest_type) ( (((src1&0x7f) <= 0x1f)||(src1 == 0x7f )) ? 1 : 0 ) )
_VEC_FUNC_2V( visspace ,	dst = (dest_type) ( ((src1>=0x9&&src1<=0xd)||(src1 == 0x20)) ? 1 : 0 ) )
_VEC_FUNC_2V( visblank ,	dst = (dest_type) ( ((src1==0x9)||(src1 == 0x20)) ? 1 : 0 ) )

#else // ! BUILD_FOR_GPU

_VEC_FUNC_3V( vshl ,	dst = (dest_type)(src1 << src2)		)
_VEC_FUNC_2V_SCAL( vsshl ,	dst = (dest_type)(src1 << scalar1_val)	)
_VEC_FUNC_2V_SCAL( vsshl2 ,	dst = (dest_type)(scalar1_val << src1)	)

_VEC_FUNC_2V( vtolower ,	dst = (dest_type) tolower( (int) src1 )	)
_VEC_FUNC_2V( vtoupper ,	dst = (dest_type) toupper( (int) src1 )	)

_VEC_FUNC_2V( vislower ,	dst = (dest_type) islower( (int) src1 )	)
_VEC_FUNC_2V( visupper ,	dst = (dest_type) isupper( (int) src1 )	)
_VEC_FUNC_2V( visalpha ,	dst = (dest_type) isalpha( (int) src1 )	)
_VEC_FUNC_2V( visalnum ,	dst = (dest_type) isalnum( (int) src1 )	)
_VEC_FUNC_2V( visdigit ,	dst = (dest_type) isdigit( (int) src1 )	)
_VEC_FUNC_2V( visspace ,	dst = (dest_type) isspace( (int) src1 )	)
_VEC_FUNC_2V( visblank ,	dst = (dest_type) isblank( (int) src1 )	)
_VEC_FUNC_2V( viscntrl ,	dst = (dest_type) iscntrl( (int) src1 )	)

#endif /* ! BUILD_FOR_GPU */

