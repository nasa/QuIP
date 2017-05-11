
/* This is a file which gets included in other files...
 * To implement each precision, we first include a file
 * defining all the macros, then we include this file.
 */

/* mathvec.c
 *
 * these are functions which are implemented for float and double.
 */


// BUILD_FOR_GPU means that we are building for openCL or cuda...

ifdef(`BUILD_FOR_GPU',`

// We need to have stub host functions to populate the table of entries???

',` dnl else // ! BUILD_FOR_GPU

dnl	All of the functions in this section are not available as gpu subroutines...
dnl	At least as far as cuda is concerned - what about openCL?

dnl	These 3 are not really math functions, but they only
dnl	are applicable to floating point arguments.

_VEC_FUNC_2V(		visinf,	dst = (dest_type) isinf( src1 )	)
_VEC_FUNC_2V(		visnan,	dst = (dest_type) isnan( src1 )	)
_VEC_FUNC_2V(		visnorm,	dst = (dest_type) isnormal( src1 )	)

_VEC_FUNC_2V(		vj0,		dst = (dest_type)j0	( src1 )		)
_VEC_FUNC_2V(		vj1,		dst = (dest_type)j1	( src1 )		)
_VEC_FUNC_2V(		vgamma,		dst = gamma_func ( src1 )		)
_VEC_FUNC_2V(		vlngamma,	dst = lngamma_func ( src1 )		)

// vuni has no source, so mixed precision makes no sense
ifdef(`MIXED_PRECISION',`',` dnl ifndef MIXED_PRECISION
_VEC_FUNC_1V(		vuni,		dst = (dest_type)drand48()	)
') dnl endif // ! MIXED_PRECISION

') dnl endif /* ! BUILD_FOR_GPU */

_VEC_FUNC_2V(		vrint,		dst = rint_func	( src1 )		)

_VEC_FUNC_2V(		vfloor,	dst = floor_func	( src1 )		)
_VEC_FUNC_2V(		vtrunc,	dst = trunc_func	( src1 )		)
/* BUG should use roundf for float!? */
_VEC_FUNC_2V(		vround,	dst = round_func	( src1 )		)
_VEC_FUNC_2V(		vceil,		dst = ceil_func	( src1 )		)
_VEC_FUNC_2V(		vsqrt,		dst = sqrt_func	( src1 )		)
_VEC_FUNC_2V(		vlog,		dst = log_func	( src1 )		)
_VEC_FUNC_2V(		vlog10,	dst = log10_func	( src1 )		)
_VEC_FUNC_2V(		rvexp,		dst = exp_func	( src1 )		)
_VEC_FUNC_2V(		vatan,		dst = atan_func	( src1 )		)
_VEC_FUNC_2V(		vtan,		dst = tan_func	( src1 )		)
_VEC_FUNC_2V(		vcos,		dst = cos_func	( src1 )		)
_VEC_FUNC_2V(		verf,		dst = erf_func	( src1 )		)
_VEC_FUNC_2V(		verfinv,	dst = erfinv_func ( src1 )		)
_VEC_FUNC_2V(		vacos,		dst = acos_func	( src1 )		)
_VEC_FUNC_2V(		vsin,		dst = sin_func	( src1 )		)
_VEC_FUNC_2V(		vasin,		dst = asin_func	( src1 )		)
// does openCL have atan2f? NO!
_VEC_FUNC_2V_MIXED(	vatn2,		dst = atan2_func (csrc1.im,csrc1.re)	)
_VEC_FUNC_3V( 	rvpow,		dst = pow_func (src1,src2)		)
_VEC_FUNC_3V(	vatan2,	dst = atan2_func ( src2, src1 )		)

_VEC_FUNC_2V_SCAL(	vsatan2,	dst = atan2_func ( src1, scalar1_val )	)
_VEC_FUNC_2V_SCAL(	vsatan22,	dst = atan2_func ( scalar1_val, src1 )	)

_VEC_FUNC_2V_SCAL(	vspow2,	dst = pow_func ( scalar1_val, src1 )	)
_VEC_FUNC_2V_SCAL(	vspow,		dst = pow_func ( src1, scalar1_val )	)

/* Complex powers - need to go to polar coords!
 *
 * (a+bi)^(c+di)
 * (r e^it)^(c+di)		r^2 = a^2 + b^2  t=atan2(b,a) (check order!)
 * r^(c+di) (e^it)^(c+di)
 * r^c r^di e^(-dt+i ct)
 * r^c e^(-dt) r^id e^ict
 *
 * r = e^log(r)
 * r^c = (e ^ log(r))^c
 *     = e ^ ( c log r )
 *
 * r^c e^(-dt) r^id e^ict
 * e^(c log(r) - dt ) e^(i(d log(r) + ct))
 *
 * How many temp vars?
 *       a   b
 *	 |\/|
 *	 |/\|
 *       r  t
 *	 |
 *	log(r)		t
 *      exp(c log(r) - dt)        cos(d log(r) + ct)	sin(d log(r) + ct)
 *
 */

// We put this in brackets because there is a comma in the call to atan2

_VEC_FUNC_CPX_3V_T1( cvpow,					\
	r = sqrt_func (csrc1.re*csrc1.re+csrc1.im*csrc1.im);	\
	theta=atan2_func (csrc1.re,csrc1.im);			\
	arg = csrc2.im * log_func (r) + csrc2.re * theta;	\
	r = csrc2.re * log_func (r) - csrc2.im * theta;		\
	cdst.re = r*cos_func (arg);				\
	cdst.im = r*sin_func (arg);				\
)

