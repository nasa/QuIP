
/* This is a file which gets included in other files...
 * To implement each precision, we first include a file
 * defining all the macros, then we include this file.
 */

/* intvec.c
 *
 * these are functions which are implemented for ALL integer precisions.
 */

include(`../../include/veclib/int_bit_vec.m4')

/* Real only */

_VEC_FUNC_3V( vmod,	dst = (dest_type) ( src1 % src2	)	)
_VEC_FUNC_2V_SCAL( vsmod,	dst = (dest_type) ( src1 % scalar1_val ) )
_VEC_FUNC_2V_SCAL( vsmod2,	dst = (dest_type) ( scalar1_val % src1 ) )

_VEC_FUNC_3V( vshr,	dst = (dest_type) ( src1 >> src2 )	)
_VEC_FUNC_2V_SCAL( vsshr,	dst = (dest_type) ( src1 >> scalar1_val	) )
_VEC_FUNC_2V_SCAL( vsshr2,	dst = (dest_type) ( scalar1_val >> src1	) )

dnl vshl was a GPU special case because of a cuda bug - still relevant?
_VEC_FUNC_3V( vshl,		dst = (dest_type)(src1 << src2)		)
_VEC_FUNC_2V_SCAL( vsshl,	dst = (dest_type)(src1 << scalar1_val)	)
_VEC_FUNC_2V_SCAL( vsshl2,	dst = (dest_type)(scalar1_val << src1)	)
	

dnl Do this define outside of the ifdef
changequote(`[',`]')
define([CHAR_CONST],['$1'])
changequote([`],['])

/* done defining char_const */

dnl no ctype.h on GPU - could implement table?

ifdef(`BUILD_FOR_GPU',`
_VEC_FUNC_2V( vtolower,	dst = (dest_type) ( src1 >= CHAR_CONST(A) && src1 <= CHAR_CONST(Z) ? src1 + 32 : src1 ) )
_VEC_FUNC_2V( vtoupper,	dst = (dest_type) ( src1 >= CHAR_CONST(a) && src1 <= CHAR_CONST(z) ? src1 - 32 : src1 ) )
	
_VEC_FUNC_2V( vislower,	dst = (dest_type) ( src1 >= CHAR_CONST(a) && src1 <= CHAR_CONST(z) ? 1 : 0 ) )
_VEC_FUNC_2V( visupper,	dst = (dest_type) ( src1 >= CHAR_CONST(A) && src1 <= CHAR_CONST(Z) ? 1 : 0 ) )
_VEC_FUNC_2V( visalpha,	dst = (dest_type) ( ((src1 >= CHAR_CONST(A) && src1 <= CHAR_CONST(Z))||(src1 >= CHAR_CONST(a) && src1 <= CHAR_CONST(z))) ? 1 : 0 ) )
_VEC_FUNC_2V( visdigit,	dst = (dest_type) ( src1 >= CHAR_CONST(0) && src1 <= CHAR_CONST(9) ? 1 : 0 ) )
_VEC_FUNC_2V( visalnum,	dst = (dest_type) ( ((src1>=CHAR_CONST(0)&&src1<=CHAR_CONST(9))||(src1 >= CHAR_CONST(A) && src1 <= CHAR_CONST(Z))||(src1 >= CHAR_CONST(a) && src1 <= CHAR_CONST(z))) ? 1 : 0 ) )

dnl THIS LINE ALONE IS ENOUGH TO CAUSE TROUBLE FOR HOST CALL!?
_VEC_FUNC_2V( viscntrl,	dst = (dest_type) ( (((src1&0x7f) <= 0x1f)||(src1 == 0x7f )) ? 1 : 0 ) )
_VEC_FUNC_2V( visspace,	dst = (dest_type) ( ((src1>=0x9&&src1<=0xd)||(src1 == 0x20)) ? 1 : 0 ) )
_VEC_FUNC_2V( visblank,	dst = (dest_type) ( ((src1==0x9)||(src1 == 0x20)) ? 1 : 0 ) )
',` dnl else ! BUILD_FOR_GPU
_VEC_FUNC_2V( vtolower,	dst = (dest_type) tolower( (int) src1 )	)
_VEC_FUNC_2V( vtoupper,	dst = (dest_type) toupper( (int) src1 )	)

_VEC_FUNC_2V( vislower,	dst = (dest_type) islower( (int) src1 )	)
_VEC_FUNC_2V( visupper,	dst = (dest_type) isupper( (int) src1 )	)
_VEC_FUNC_2V( visalpha,	dst = (dest_type) isalpha( (int) src1 )	)
_VEC_FUNC_2V( visalnum,	dst = (dest_type) isalnum( (int) src1 )	)
_VEC_FUNC_2V( visdigit,	dst = (dest_type) isdigit( (int) src1 )	)
_VEC_FUNC_2V( visspace,	dst = (dest_type) isspace( (int) src1 )	)
_VEC_FUNC_2V( visblank,	dst = (dest_type) isblank( (int) src1 )	)
_VEC_FUNC_2V( viscntrl,	dst = (dest_type) iscntrl( (int) src1 )	)
') dnl endif ! BUILD_FOR_GPU

dnl/* build_for_gpu BUILD_FOR_GPU block BEGIN */
dnl
dnl/* why no left-shift on GPU?? */
dnl/* CUDA impementation is broken */
dnl	ifdef(`BUILD_FOR_CUDA',`
dnl	dnl KERN_CALL_VV_LS(vshl, dst, src1, src2)
dnl	dnl KERN_CALL_VS_LS(vsshl, dst, src1, scalar1_val)
dnl	dnl KERN_CALL_VS_LS(vsshl2, dst, scalar1_val, src1)
dnl	_VEC_FUNC_3V( vshl,		dst = (dest_type)(src1 << src2)		)
dnl	_VEC_FUNC_2V_SCAL( vsshl,	dst = (dest_type)(src1 << scalar1_val)	)
dnl	_VEC_FUNC_2V_SCAL( vsshl2,	dst = (dest_type)(scalar1_val << src1)	)
dnl	',` dnl else // ! BUILD_FOR_CUDA
dnl	dnl _VEC_FUNC_3V( vshl,		LSHIFT_SWITCH_32(dst,src1,src2)		)
dnl	dnl _VEC_FUNC_2V_SCAL( vsshl,	LSHIFT_SWITCH_32(dst,src1,scalar1_val)	)
dnl	dnl _VEC_FUNC_2V_SCAL( vsshl2,	LSHIFT_SWITCH_32(dst,scalar1_val,src1)	)
dnl	_VEC_FUNC_3V( vshl,		dst = (dest_type)(src1 << src2)		)
dnl	_VEC_FUNC_2V_SCAL( vsshl,	dst = (dest_type)(src1 << scalar1_val)	)
dnl	_VEC_FUNC_2V_SCAL( vsshl2,	dst = (dest_type)(scalar1_val << src1)	)
dnl	') dnl endif // ! BUILD_FOR_CUDA
dnl	/* end of cuda block */
dnl
dnl/* ctype stuff... */
dnl 32 = 'a'-'A'
dnl quote this block to escape single quotes in char constants
dnl
dnl	_VEC_FUNC_2V( vtolower,	dst = (dest_type) ( src1 >= CHAR_CONST(A) && src1 <= CHAR_CONST(Z) ? src1 + 32 : src1 ) )
dnl	_VEC_FUNC_2V( vtoupper,	dst = (dest_type) ( src1 >= CHAR_CONST(a) && src1 <= CHAR_CONST(z) ? src1 - 32 : src1 ) )
dnl	
dnl	_VEC_FUNC_2V( vislower,	dst = (dest_type) ( src1 >= CHAR_CONST(a) && src1 <= CHAR_CONST(z) ? 1 : 0 ) )
dnl	_VEC_FUNC_2V( visupper,	dst = (dest_type) ( src1 >= CHAR_CONST(A) && src1 <= CHAR_CONST(Z) ? 1 : 0 ) )
dnl	_VEC_FUNC_2V( visalpha,	dst = (dest_type) ( ((src1 >= CHAR_CONST(A) && src1 <= CHAR_CONST(Z))||(src1 >= CHAR_CONST(a) && src1 <= CHAR_CONST(z))) ? 1 : 0 ) )
dnl	_VEC_FUNC_2V( visdigit,	dst = (dest_type) ( src1 >= CHAR_CONST(0) && src1 <= CHAR_CONST(9) ? 1 : 0 ) )
dnl	_VEC_FUNC_2V( visalnum,	dst = (dest_type) ( ((src1>=CHAR_CONST(0)&&src1<=CHAR_CONST(9))||(src1 >= CHAR_CONST(A) && src1 <= CHAR_CONST(Z))||(src1 >= CHAR_CONST(a) && src1 <= CHAR_CONST(z))) ? 1 : 0 ) )
dnl
dnl THIS LINE ALONE IS ENOUGH TO CAUSE TROUBLE FOR HOST CALL!?
dnl _VEC_FUNC_2V( viscntrl,	dst = (dest_type) ( (((src1&0x7f) <= 0x1f)||(src1 == 0x7f )) ? 1 : 0 ) )
dnl	_VEC_FUNC_2V( visspace,	dst = (dest_type) ( ((src1>=0x9&&src1<=0xd)||(src1 == 0x20)) ? 1 : 0 ) )
dnl	_VEC_FUNC_2V( visblank,	dst = (dest_type) ( ((src1==0x9)||(src1 == 0x20)) ? 1 : 0 ) )
dnl
dnl/* build_for_gpu BUILD_FOR_GPU block done */
dnl
dnl',` dnl else // ! BUILD_FOR_GPU
dnl/* ! build_for_gpu ! BUILD_FOR_GPU block BEGIN */
dnl
dnl	_VEC_FUNC_3V( vshl,		dst = (dest_type)(src1 << src2)		)
dnl	_VEC_FUNC_2V_SCAL( vsshl,	dst = (dest_type)(src1 << scalar1_val)	)
dnl	_VEC_FUNC_2V_SCAL( vsshl2,	dst = (dest_type)(scalar1_val << src1)	)
dnl	
dnl	_VEC_FUNC_2V( vtolower,	dst = (dest_type) tolower( (int) src1 )	)
dnl	_VEC_FUNC_2V( vtoupper,	dst = (dest_type) toupper( (int) src1 )	)
dnl
dnl	_VEC_FUNC_2V( vislower,	dst = (dest_type) islower( (int) src1 )	)
dnl	_VEC_FUNC_2V( visupper,	dst = (dest_type) isupper( (int) src1 )	)
dnl	_VEC_FUNC_2V( visalpha,	dst = (dest_type) isalpha( (int) src1 )	)
dnl	_VEC_FUNC_2V( visalnum,	dst = (dest_type) isalnum( (int) src1 )	)
dnl	_VEC_FUNC_2V( visdigit,	dst = (dest_type) isdigit( (int) src1 )	)
dnl	_VEC_FUNC_2V( visspace,	dst = (dest_type) isspace( (int) src1 )	)
dnl	_VEC_FUNC_2V( visblank,	dst = (dest_type) isblank( (int) src1 )	)
dnl	_VEC_FUNC_2V( viscntrl,	dst = (dest_type) iscntrl( (int) src1 )	)
dnl
dnl/* ! build_for_gpu ! BUILD_FOR_GPU block DONE */
dnl
dnl') dnl endif /* ! BUILD_FOR_GPU */
/* after build_for_gpu conditional block */

