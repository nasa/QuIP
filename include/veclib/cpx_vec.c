/* complex number stuff */

#include "veclib/cpx_args.h"

/* Real only */

// for debugging...
#define SHOW_CPX_3							\
	sprintf(error_string,"csrc1 = %g  %g",csrc1.re,csrc1.im);	\
	advise(error_string);						\
	sprintf(error_string,"csrc2 = %g  %g",csrc2.re,csrc2.im);	\
	advise(error_string);

#define CPX_PROD_RE(c1,c2)	( c1.re*c2.re - c1.im*c2.im )
#define CPX_PROD_IM(c1,c2)	( c1.re*c2.im + c1.im*c2.re )

_VEC_FUNC_CPX_2V( cvmov ,		cdst.re = csrc1.re ; cdst.im = csrc1.im )
_VEC_FUNC_CPX_2V( cvneg ,		cdst.re = - csrc1.re ; cdst.im = - csrc1.im )
_VEC_FUNC_CPX_2V_T2( cvsqr ,		tmpc.re = CPX_PROD_RE(csrc1,csrc1);
					tmpc.im = csrc1.re*csrc1.im*2 ;
					ASSIGN_CPX(cdst,tmpc); )

_VEC_FUNC_CPX_2V( vconj ,		cdst.re = csrc1.re;
					cdst.im = - csrc1.im )

// complex exponential - e ^ a + bi = e^a (cos b + i sin b)
// This can work in-place...
_VEC_FUNC_CPX_2V_T2( cvexp ,		tmpc.re = exp_func(csrc1.re);		\
					cdst.re = tmpc.re * cos_func(csrc1.im);	\
					cdst.im = tmpc.re * sin_func(csrc1.im);	\
					)

_VEC_FUNC_CPX_3V( cvadd , 	cdst.re = csrc1.re + csrc2.re ; cdst.im = csrc1.im + csrc2.im )
_VEC_FUNC_CPX_3V( cvsub ,	cdst.re = csrc2.re - csrc1.re ; cdst.im = csrc2.im - csrc1.im )

#define ASSIGN_CPX_PROD(cd,c1,c2)	{ cd.re = CPX_PROD_RE(c1,c2);		\
					cd.im = CPX_PROD_IM(c1,c2); }

_VEC_FUNC_CPX_3V_T2( cvmul ,	ASSIGN_CPX_PROD(tmpc,csrc1,csrc2)
					ASSIGN_CPX(cdst,tmpc) )

#define CPX_NORM(c)		( c.re*c.re +c.im*c.im )

// conjugate second operand
#define CPX_CPROD_RE(c1,c2)	( c1.re*c2.re + c1.im*c2.im )
#define CPX_CPROD_IM(c1,c2)	( c2.re*c1.im - c2.im*c1.re )

#define ASSIGN_CPX_CPROD(cd,c1,c2)		{ cd.re = CPX_CPROD_RE(c1,c2);		\
						cd.im = CPX_CPROD_IM(c1,c2); }

#define ASSIGN_CPX_CPROD_NORM(cd,c1,c2,denom)	{ cd.re = CPX_CPROD_RE(c1,c2)/denom;		\
						cd.im = CPX_CPROD_IM(c1,c2)/denom; }

// are we dividing by src1 or src2 ???

_VEC_FUNC_CPX_3V_T3( cvdiv ,	tmp_denom = CPX_NORM(csrc2);
					ASSIGN_CPX_CPROD_NORM(tmpc,csrc1,csrc2,tmp_denom)
					ASSIGN_CPX(cdst,tmpc) )

/* vcmul seems to be redundant with cvmul, but actually it is conjugate multiplication? */
_VEC_FUNC_CPX_3V_T2( vcmul , 	ASSIGN_CPX_CPROD(tmpc,csrc1,csrc2)
					ASSIGN_CPX(cdst,tmpc) )

/* float times complex */

_VEC_FUNC_CCR_3V( mvadd , cdst.re = csrc1.re + src2; cdst.im = csrc1.im )
_VEC_FUNC_CCR_3V( mvsub , cdst.re = csrc1.re - src2; cdst.im = csrc1.im )
_VEC_FUNC_CCR_3V( mvmul , cdst.re = csrc1.re * src2; cdst.im = csrc1.im * src2 )
_VEC_FUNC_CCR_3V( mvdiv , cdst.re=csrc1.re/src2; cdst.im = csrc1.im / src2 )

_VEC_FUNC_2V_MIXED( vmgsq , dst = csrc1.re*csrc1.re + csrc1.im*csrc1.im )


/* BUG check for divide by zero */
// These appear to have a complex destination, but a real source,
// and real scalar...  what's the point?
_VEC_FUNC_CR_1S_2V( mvsdiv ,	cdst.re = scalar1_val / src1 ;
					cdst.im = 0
					)

_VEC_FUNC_CR_1S_2V( mvsdiv2 ,	cdst.re = src1 / scalar1_val ;
					cdst.im = 0
					)

_VEC_FUNC_CR_1S_2V( mvsmul ,	cdst.re = scalar1_val * src1 ;
					cdst.im = 0
					)

_VEC_FUNC_CR_1S_2V( mvssub ,	cdst.re = scalar1_val - src1 ;
					cdst.im = 0
					)

_VEC_FUNC_CR_1S_2V( mvsadd ,	cdst.re = src1 + scalar1_val ;
					cdst.im = 0
					)

_VEC_FUNC_CPX_1S_2V( cvsadd ,	cdst.re = csrc1.re + cscalar1_val.re ;
					cdst.im = csrc1.im + cscalar1_val.im
					)

_VEC_FUNC_CPX_1S_2V( cvssub ,	cdst.re = cscalar1_val.re - csrc1.re ;
					cdst.im = cscalar1_val.im - csrc1.im
					)
_VEC_FUNC_CPX_1S_2V_T2( cvsmul ,	ASSIGN_CPX_PROD(tmpc,csrc1,cscalar1_val)
					ASSIGN_CPX(cdst,tmpc)
					)

#define QUAT_NORM(q)	( q.re * q.re + q._i * q._i + q._j * q._j + q._k * q._k )

// dst = scalar / src
_VEC_FUNC_CPX_1S_2V_T3( cvsdiv ,	tmp_denom=CPX_NORM(csrc1);
					ASSIGN_CPX_CPROD_NORM(tmpc,cscalar1_val,csrc1,tmp_denom)
					ASSIGN_CPX(cdst,tmpc)
					)

_VEC_FUNC_CPX_1S_2V_T3( cvsdiv2 ,	tmp_denom=CPX_NORM(cscalar1_val);
					ASSIGN_CPX_CPROD_NORM(tmpc,csrc1,cscalar1_val,tmp_denom)
					ASSIGN_CPX(cdst,tmpc)
					)

/* for mixed (MDECLS2), v1 is complex and v2 is real? */


_VEC_FUNC_CPX_1S_1V( cvset , cdst.re = cscalar1_val.re ; cdst.im = cscalar1_val.im )

#ifdef NOT_YET

/* dot product with complex conjugate */

FUNC_DECL( vcdot )
{
	RCSCALAR_DECL;
	CINIT2;
	RSCALAR_SETUP;

	scalar->re = scalar->im = 0.0;

	V2LOOP( scalar->re += csrc1.re * csrc2.re + csrc1.im * csrc2.im ; scalar->im += csrc1.re * csrc2.im - csrc1.im * csrc2.re )
}
#endif /* NOT_YET */

/* We use std_tmp here so this will work if called w/ in-place argument (v3 = v1 or v2)
 */



/* conjugated vsmul ? */

_VEC_FUNC_CPX_1S_2V_T2( vscml ,	ASSIGN_CPX_CPROD(tmpc,csrc1,cscalar1_val)
					ASSIGN_CPX(cdst,tmpc) )


/* These are clean, but don't work when destination and source have different precisions */
/*
_VEC_FUNC_SBM_CPX_3V( cvvv_slct ,	cdst = srcbit ? csrc1 : csrc2 )
CPX_VS_SELECTION_METHOD( cvvs_slct ,	cdst = srcbit ? csrc1 : cscalar1_val )
CPX_SS_SELECTION_METHOD( cvss_slct ,	cdst = srcbit ? cscalar1_val : cscalar2_val )
*/

// why not structure assign?
#define COPY_CPX( c )	{ cdst.re=c.re; cdst.im=c.im; }

_VEC_FUNC_SBM_CPX_3V( cvvv_slct ,	if( srcbit ) COPY_CPX(csrc1) else COPY_CPX(csrc2) )

_VEC_FUNC_SBM_CPX_1S_2V( cvvs_slct ,	if( srcbit ) COPY_CPX(csrc1) else COPY_CPX(cscalar1_val) )
_VEC_FUNC_SBM_CPX_2S_1V( cvss_slct ,	if( srcbit ) COPY_CPX(cscalar1_val) else COPY_CPX(cscalar2_val) )

// no CPX in this macro?
_VEC_FUNC_CPX_2V_PROJ( cvsum,
	cdst.re = 0 ; cdst.im = 0 ,
	cdst.re += csrc1.re; cdst.im += csrc1.im ,
	psrc1.re + psrc2.re,
	psrc1.im + psrc2.im
	)

// Need to implement cvdot as composite of cvmul and cvsum BUG
//_VEC_FUNC_CPX_3V_PROJ( cvdot,
//	cdst.re = 0; cdst.im = 0 ,
//	cdst.re += csrc1.re * csrc2.re - csrc1.im * csrc2.im; cdst.im += csrc1.re * csrc2.im + csrc1.im * csrc2.re ,
//	csrc1.re*csrc2.re-csrc1.im*csrc2.im,
//	csrc1.re*csrc2.im+csrc1.im*csrc2.re,
//	csrc1.re+csrc2.re,
//	csrc1.im+csrc2.im
//	)

#ifndef BUILD_FOR_GPU
_VEC_FUNC_CPX_2V( cvrand , cdst.re = rn((u_long)csrc1.re); cdst.im = rn((u_long)csrc1.im)	)
#endif /* ! BUILD_FOR_GPU */


#ifdef QUATERNION_SUPPORT

#include "veclib/quat_args.h"

/* Quaternions */

/* Here is the multiplication chart:
 * note that multiplication of the imaginary terms is non-commutative...
 *
 *	X		1	i	j	k
 *
 *	1		1	i	j	k
 *	i		i	-1	k	-j
 *	j		j	-k	-1	i
 *	k		k	j	-i	-1
 */

#define QUAT_PROD_RE(q1,q2)	( q1.re * q2.re - q1._i * q2._i - q1._j * q2._j - q1._k * q2._k )
#define QUAT_PROD_I(q1,q2)	( q1.re * q2._i + q1._i * q2.re + q1._j * q2._k - q1._k * q2._j )
#define QUAT_PROD_J(q1,q2)	( q1.re * q2._j + q1._j * q2.re + q1._k * q2._i - q1._i * q2._k )
#define QUAT_PROD_K(q1,q2)	( q1.re * q2._k + q1._k * q2.re + q1._i * q2._j - q1._j * q2._i )

_VEC_FUNC_QUAT_2V( qvmov ,		qdst.re = qsrc1.re ;
					qdst._i = qsrc1._i ;
					qdst._j = qsrc1._j ;
					qdst._k = qsrc1._k )

_VEC_FUNC_QUAT_2V( qvneg ,		qdst.re = - qsrc1.re ;
					qdst._i = - qsrc1._i ;
					qdst._j = - qsrc1._j ;
					qdst._k = - qsrc1._k )

_VEC_FUNC_QUAT_2V_T4( qvsqr ,		tmpq.re = QUAT_PROD_RE(qsrc1,qsrc1);
					tmpq._i =	2 * qsrc1.re * qsrc1._i;
					tmpq._j =	2 * qsrc1.re * qsrc1._j;
					tmpq._k =	2 * qsrc1.re * qsrc1._k;
					ASSIGN_QUAT(qdst,tmpq) )

_VEC_FUNC_QUAT_3V( qvadd , 		qdst.re = qsrc1.re + qsrc2.re ;
					qdst._i = qsrc1._i + qsrc2._i ;
					qdst._j = qsrc1._j + qsrc2._j ;
					qdst._k = qsrc1._k + qsrc2._k )

_VEC_FUNC_QUAT_3V( qvsub ,		qdst.re = qsrc2.re - qsrc1.re ;
					qdst._i = qsrc2._i - qsrc1._i ;
					qdst._j = qsrc2._j - qsrc1._j ;
					qdst._k = qsrc2._k - qsrc1._k )

#define ASSIGN_QUAT_PROD(qd,q1,q2)	{ qd.re = QUAT_PROD_RE(q1,q2);	\
					qd._i = QUAT_PROD_I(q1,q2);	\
					qd._j = QUAT_PROD_J(q1,q2);	\
					qd._k = QUAT_PROD_K(q1,q2); }

_VEC_FUNC_QUAT_3V_T4( qvmul ,	ASSIGN_QUAT_PROD(tmpq,qsrc1,qsrc2)
					ASSIGN_QUAT(qdst,tmpq) )

// Quaternion division is like complex division:  q1 / q2 = q1 * conj(q2) / maqsq(q2)
// Conjugation negates all three imaginary components...

// T4 declares tmpq, T5 tmpq plus tmp_denom

#define QUAT_NORM(q)	( q.re * q.re + q._i * q._i + q._j * q._j + q._k * q._k )

// QUAT_CPROD conjugates the second operand...

#define QUAT_CPROD_RE(q1,q2)	( q1.re * q2.re + q1._i * q2._i + q1._j * q2._j + q1._k * q2._k )
#define QUAT_CPROD_I(q1,q2)	( - q1.re * q2._i + q1._i * q2.re - q1._j * q2._k + q1._k * q2._j )
#define QUAT_CPROD_J(q1,q2)	( - q1.re * q2._j + q1._j * q2.re - q1._k * q2._i + q1._i * q2._k )
#define QUAT_CPROD_K(q1,q2)	( - q1.re * q2._k + q1._k * q2.re - q1._i * q2._j + q1._j * q2._i )

#define ASSIGN_QUAT_CPROD_NORM(qd,q1,q2,denom)	{ qd.re = QUAT_CPROD_RE(q1,q2)/denom;	\
						qd._i = QUAT_CPROD_I(q1,q2)/denom;	\
						qd._j = QUAT_CPROD_J(q1,q2)/denom;	\
						qd._k = QUAT_CPROD_K(q1,q2)/denom; }

_VEC_FUNC_QUAT_3V_T5( qvdiv ,	tmp_denom =	QUAT_NORM(qsrc2);
					ASSIGN_QUAT_CPROD_NORM(tmpq,qsrc1,qsrc2,tmp_denom)
					ASSIGN_QUAT(qdst,tmpq) )

// dst = scalar / src1

_VEC_FUNC_QUAT_1S_2V_T5( qvsdiv ,	tmp_denom =	QUAT_NORM(qsrc1);
					ASSIGN_QUAT_CPROD_NORM(tmpq,qscalar1_val,qsrc1,tmp_denom)
					ASSIGN_QUAT(qdst,tmpq) )

// dst = src1 / scalar

_VEC_FUNC_QUAT_1S_2V_T5( qvsdiv2 ,	tmp_denom =	QUAT_NORM(qscalar1_val);
					ASSIGN_QUAT_CPROD_NORM(tmpq,qsrc1,qscalar1_val,tmp_denom)
					ASSIGN_QUAT(qdst,tmpq) )


/* float times quaternion */

// These appear to have a quaternion destination, but a real source,
// and real scalar...  what's the point?

_VEC_FUNC_QQR_3V( pvadd ,		qdst.re = qsrc1.re + src2;
					qdst._i = qsrc1._i ;
					qdst._j = qsrc1._j ;
					qdst._k = qsrc1._k )
					
_VEC_FUNC_QQR_3V( pvsub ,		qdst.re = qsrc1.re - src2;
					qdst._i = qsrc1._i ;
					qdst._j = qsrc1._j ;
					qdst._k = qsrc1._k )

_VEC_FUNC_QQR_3V( pvmul ,		qdst.re = qsrc1.re * src2;
					qdst._i = qsrc1._i * src2 ;
					qdst._j = qsrc1._j * src2 ;
					qdst._k = qsrc1._k * src2 )

_VEC_FUNC_QQR_3V( pvdiv ,		qdst.re = qsrc1.re / src2;
					qdst._i = qsrc1._i / src2 ;
					qdst._j = qsrc1._j / src2 ;
					qdst._k = qsrc1._k / src2 )

/* BUG check for divide by zero */
// 
//_VF_QR_1S_2V( name, type_code, stat)
_VEC_FUNC_QR_1S_2V( pvsdiv ,	qdst.re = scalar1_val / src1 ;
					qdst._i = 0;
					qdst._j = 0;
					qdst._k = 0
					)

_VEC_FUNC_QR_1S_2V( pvsdiv2 ,	qdst.re = src1 / scalar1_val ;
					qdst._i = 0;
					qdst._j = 0;
					qdst._k = 0
					)

_VEC_FUNC_QR_1S_2V( pvsmul ,	qdst.re = scalar1_val * src1 ;
					qdst._i = 0;
					qdst._j = 0;
					qdst._k = 0
					)

_VEC_FUNC_QR_1S_2V( pvssub ,	qdst.re = scalar1_val - src1 ;
					qdst._i = 0;
					qdst._j = 0;
					qdst._k = 0
					)

_VEC_FUNC_QR_1S_2V( pvsadd ,	qdst.re = src1 + scalar1_val ;
					qdst._i = 0;
					qdst._j = 0;
					qdst._k = 0
					)

_VEC_FUNC_QUAT_1S_2V( qvsadd ,	qdst.re = qsrc1.re + qscalar1_val.re ;
					qdst._i = qsrc1._i + qscalar1_val._i ;
					qdst._j = qsrc1._j + qscalar1_val._j ;
					qdst._k = qsrc1._k + qscalar1_val._k
					)

_VEC_FUNC_QUAT_1S_2V( qvssub ,	qdst.re = qscalar1_val.re - qsrc1.re ;
					qdst._i = qscalar1_val._i - qsrc1._i ;
					qdst._j = qscalar1_val._j - qsrc1._j ;
					qdst._k = qscalar1_val._k - qsrc1._k
					)

/* For real or complex, multiplication is commutative, but not for quaternions!? */

_VEC_FUNC_QUAT_1S_2V_T4( qvsmul ,	ASSIGN_QUAT_PROD(tmpq,qsrc1,qscalar1_val)
					ASSIGN_QUAT(qdst,tmpq) )

/* not yet...
_VEC_FUNC_QUAT_1S_2V_T4( qvsmul2 ,ASSIGN_QUAT_PROD(tmpq,qscalar1_val,qsrc1)
					ASSIGN_QUAT(qdst,tmpq) )
*/

_VEC_FUNC_QUAT_1S_1V( qvset ,	qdst.re = qscalar1_val.re ;
					qdst._i = qscalar1_val._i ;
					qdst._j = qscalar1_val._j ;
					qdst._k = qscalar1_val._k )


/*
QUAT_VV_SELECTION_METHOD( qvvv_slct ,	qdst = srcbit ? qsrc1 : qsrc2 )
QUAT_VS_SELECTION_METHOD( qvvs_slct ,	qdst = srcbit ? qsrc1 : qscalar1_val )
QUAT_SS_SELECTION_METHOD( qvss_slct ,	qdst = srcbit ? qscalar1_val : qscalar2_val )
*/
#define COPY_QUAT( q )	{ qdst.re=q.re; qdst._i=q._i; qdst._j=q._j; qdst._k=q._k; }

_VEC_FUNC_SBM_QUAT_3V( qvvv_slct ,	if( srcbit ) COPY_QUAT(qsrc1) else COPY_QUAT(qsrc2) )
_VEC_FUNC_SBM_QUAT_1S_2V( qvvs_slct ,	if( srcbit ) COPY_QUAT(qsrc1) else COPY_QUAT(qscalar1_val) )
_VEC_FUNC_SBM_QUAT_2S_1V( qvss_slct ,	if( srcbit ) COPY_QUAT(qscalar1_val) else COPY_QUAT(qscalar2_val) )

/* BUG - gpu implementation? */

_VEC_FUNC_QUAT_2V_PROJ( qvsum,
					qdst.re = 0 ;
					qdst._i = 0 ;
					qdst._j = 0 ;
					qdst._k = 0
					,
					qdst.re += qsrc1.re ;
					qdst._i += qsrc1._i ;
					qdst._j += qsrc1._j ;
					qdst._k += qsrc1._k
					,
					psrc1.re + psrc2.re,
					psrc1._i + psrc2._i,
					psrc1._j + psrc2._j,
					psrc1._k + psrc2._k
					)


#endif /* QUATERNION_SUPPORT */

