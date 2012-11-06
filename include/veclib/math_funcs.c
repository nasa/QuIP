#include "quip_config.h"

/* This is a file which gets included in other files...
 * To implement each precision, we first include a file
 * defining all the macros, then we include this file.
 */

/* mathvec.c
 *
 * these are functions which are implemented for float and double.
 */

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "rn.h"		/* rninit() */

#ifdef SGI
#define round	rint
#endif /* SGI */

#ifndef GPU_FUNCTION
TWO_VEC_METHOD(		vj0 ,		dst = j0	( src1 )		)
TWO_VEC_METHOD(		vj1 ,		dst = j1	( src1 )		)
#endif /* GPU_FUNCTION */

TWO_VEC_METHOD(		vrint ,		dst = rint	( src1 )		)
TWO_VEC_METHOD(		vfloor ,	dst = floor	( src1 )		)
/* BUG should use roundf for float!? */
TWO_VEC_METHOD(		vround ,	dst = round	( src1 )		)
TWO_VEC_METHOD(		vceil ,		dst = ceil	( src1 )		)
TWO_VEC_METHOD(		vsqrt ,		dst = sqrt	( src1 )		)
ONE_VEC_METHOD(		vuni ,		dst = drand48()		)
TWO_VEC_METHOD(		vlog ,		dst = log	( src1 )		)
TWO_VEC_METHOD(		vlog10 ,	dst = log10	( src1 )		)
TWO_VEC_METHOD(		vexp ,		dst = exp	( src1 )		)
TWO_VEC_METHOD(		vatan ,		dst = atan	( src1 )		)
TWO_VEC_METHOD(		vtan ,		dst = tan	( src1 )		)
TWO_VEC_METHOD(		vcos ,		dst = cos	( src1 )		)
TWO_VEC_METHOD(		verf ,		dst = erf	( src1 )		)
TWO_VEC_METHOD(		vacos ,		dst = acos	( src1 )		)
TWO_VEC_METHOD(		vsin ,		dst = sin	( src1 )		)
TWO_VEC_METHOD(		vasin ,		dst = asin	( src1 )		)
TWO_MIXED_RC_VEC_METHOD(	vatn2 ,		dst = atan2(csrc1.im,csrc1.re)	)
THREE_VEC_METHOD( 	rvpow ,		dst = pow(src1,src2)		)
THREE_VEC_METHOD(	vatan2 ,	dst = atan2( src2 , src1 )		)

TWO_VEC_SCALAR_METHOD(	vsatan2 ,	dst = atan2( src1 , scalar1_val )	)
TWO_VEC_SCALAR_METHOD(	vsatan22 ,	dst = atan2( scalar1_val, src1 )	)

TWO_VEC_SCALAR_METHOD(	vspow ,		dst = pow( src1 , scalar1_val )	)
TWO_VEC_SCALAR_METHOD(	vspow2 ,	dst = pow( scalar1_val, src1 )	)

/* Complex powers - need to go to polar coords!
 *
 * We only use the real part of the exponent...
 * This is a subtle BUG, it will work ok if we just pass
 * a real scalar, but the argument-getting routine in
 * libwarmenu will want to get a complex scalar.
 */

#ifdef NOT_YET
TWO_VEC_METHOD( cvpow ,
{
	std_type r,theta;
	CSINIT2;
V2LOOP( r = sqrt(src1.re*src1.re+src1.im*src1.im); theta=atan2(src1.re,src1.im); r=pow(r,scalar1_val.re); theta *= scalar1_val.re; v2a.re = r*cos(theta); v2a.im = r*sin(theta) )
}
#endif /* NOT_YET */


