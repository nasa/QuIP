/* complex number stuff */

#include <stdlib.h>		/* abs() */
#include "debug.h"

/* Real only */

// for debugging...
#define SHOW_CPX_3							\
	sprintf(error_string,"csrc1 = %g  %g",csrc1.re,csrc1.im);	\
	advise(error_string);						\
	sprintf(error_string,"csrc2 = %g  %g",csrc2.re,csrc2.im);	\
	advise(error_string);

TWO_CPX_VEC_METHOD( cvmov ,		cdst.re = csrc1.re ; cdst.im = csrc1.im )
TWO_CPX_VEC_METHOD( cvneg ,		cdst.re = - csrc1.re ; cdst.im = - csrc1.im )
TWO_CPXT_VEC_METHOD( cvsqr ,		std_tmp = csrc1.re*csrc1.re-csrc1.im*csrc1.im;
					cdst.im=csrc1.re*csrc1.im*2; cdst.re = (std_type) std_tmp )
TWO_CPX_VEC_METHOD( vconj ,		cdst.re = csrc1.re; cdst.im = - csrc1.im )

THREE_CPX_VEC_METHOD( cvadd , 	cdst.re = csrc1.re + csrc2.re ; cdst.im = csrc1.im + csrc2.im )
THREE_CPX_VEC_METHOD( cvsub ,	cdst.re = csrc2.re - csrc1.re ; cdst.im = csrc2.im - csrc1.im )
THREE_CPXT_VEC_METHOD( cvmul ,	std_tmp = csrc1.re*csrc2.re - csrc1.im*csrc2.im;
					cdst.im = csrc1.re*csrc2.im + csrc1.im*csrc2.re;
					cdst.re = std_tmp )
THREE_CPXD_VEC_METHOD( cvdiv ,	tmp_denom = csrc1.re*csrc1.re +csrc1.im*csrc1.im;
					std_tmp=(csrc1.re*csrc2.re+csrc1.im*csrc2.im)/tmp_denom;
					cdst.im = (csrc2.im*csrc1.re-csrc2.re*csrc1.im)/ tmp_denom;
					cdst.re = std_tmp )

/* vcmul seems to be redundant with cvmul, but actually it is conjugate multiplication? */
THREE_CPXT_VEC_METHOD( vcmul , 	std_tmp = csrc1.re * csrc2.re + csrc1.im * csrc2.im ;
					cdst.im = csrc2.re * csrc1.im - csrc2.im * csrc1.re;
					cdst.re = std_tmp )

/* float times complex */

THREE_MIXED_VEC_METHOD( mvadd , cdst.re = csrc1.re + src2; cdst.im = csrc1.im )
THREE_MIXED_VEC_METHOD( mvsub , cdst.re = csrc1.re - src2; cdst.im = csrc1.im )
THREE_MIXED_VEC_METHOD( mvmul , cdst.re = csrc1.re * src2; cdst.im = csrc1.im * src2 )
THREE_MIXED_VEC_METHOD( mvdiv , cdst.re=csrc1.re/src2; cdst.im = csrc1.im / src2 )

TWO_MIXED_RC_VEC_METHOD( vmgsq , dst = csrc1.re*csrc1.re + csrc1.im*csrc1.im )


/* BUG check for divide by zero */
TWO_MIXED_CR_VEC_SCALAR_METHOD( mvsdiv ,	cdst.re = scalar1_val / src1 ;
					cdst.im = 0
					)

TWO_MIXED_CR_VEC_SCALAR_METHOD( mvsdiv2 ,	cdst.re = src1 / scalar1_val ;
					cdst.im = 0
					)

TWO_MIXED_CR_VEC_SCALAR_METHOD( mvsmul ,	cdst.re = scalar1_val * src1 ;
					cdst.im = 0
					)

TWO_MIXED_CR_VEC_SCALAR_METHOD( mvssub ,	cdst.re = scalar1_val - src1 ;
					cdst.im = 0
					)

TWO_MIXED_CR_VEC_SCALAR_METHOD( mvsadd ,	cdst.re = src1 + scalar1_val ;
					cdst.im = 0
					)

TWO_CPX_VEC_SCALAR_METHOD( cvsadd ,	cdst.re = csrc1.re + cscalar1_val.re ;
					cdst.im = csrc1.im + cscalar1_val.im
					)

TWO_CPX_VEC_SCALAR_METHOD( cvssub ,	cdst.re = cscalar1_val.re - csrc1.re ;
					cdst.im = cscalar1_val.im - csrc1.im
					)
TWO_CPXT_VEC_SCALAR_METHOD( cvsmul ,	std_tmp = csrc1.re * cscalar1_val.re - csrc1.im * cscalar1_val.im;
					cdst.im = csrc1.im * cscalar1_val.re + csrc1.re * cscalar1_val.im;
					cdst.re = std_tmp
					)

TWO_CPXD_VEC_SCALAR_METHOD( cvsdiv ,	tmp_denom=(csrc1.re*csrc1.re+csrc1.im*csrc1.im);
					std_tmp = (csrc1.re * cscalar1_val.re
							+ csrc1.im * cscalar1_val.im)/tmp_denom;
					cdst.im = (cscalar1_val.im * csrc1.re 
							- cscalar1_val.re * csrc1.im )/tmp_denom;
					cdst.re = std_tmp
					)

TWO_CPXD_VEC_SCALAR_METHOD( cvsdiv2 ,	tmp_denom=(cscalar1_val.re*cscalar1_val.re+cscalar1_val.im*cscalar1_val.im);
					std_tmp = (csrc1.re * cscalar1_val.re
							+ csrc1.im * cscalar1_val.im)/tmp_denom;
					cdst.im = (cscalar1_val.re * csrc1.im 
							- cscalar1_val.im * csrc1.re )/tmp_denom;
					cdst.re = std_tmp
					)

/* for mixed (MDECLS2), v1 is complex and v2 is real? */


ONE_CPX_VEC_SCALAR_METHOD( cvset , cdst.re = cscalar1_val.re ; cdst.im = cscalar1_val.im )

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



/* complex vsmul */

TWO_CPXT_VEC_SCALAR_METHOD( vscml ,	std_tmp = csrc1.re*cscalar1_val.re + csrc1.im*cscalar1_val.im ;
					cdst.im = csrc1.re*cscalar1_val.im - csrc1.im*cscalar1_val.re;
					cdst.re = std_tmp )


/* These are clean, but don't work when destination and source have different precisions */
/*
CPX_VV_SELECTION_METHOD( cvvv_slct ,	cdst = srcbit ? csrc1 : csrc2 )
CPX_VS_SELECTION_METHOD( cvvs_slct ,	cdst = srcbit ? csrc1 : cscalar1_val )
CPX_SS_SELECTION_METHOD( cvss_slct ,	cdst = srcbit ? cscalar1_val : cscalar2_val )
*/

#define COPY_CPX( c )	{ cdst.re=c.re; cdst.im=c.im; }

CPX_VV_SELECTION_METHOD( cvvv_slct ,	if( srcbit ) COPY_CPX(csrc1) else COPY_CPX(csrc2) )
CPX_VS_SELECTION_METHOD( cvvs_slct ,	if( srcbit ) COPY_CPX(csrc1) else COPY_CPX(cscalar1_val) )
CPX_SS_SELECTION_METHOD( cvss_slct ,	if( srcbit ) COPY_CPX(cscalar1_val) else COPY_CPX(cscalar2_val) )

/* ONE_CPX_VEC_SCALRET_METHOD( cvsum, retval.re += csrc1.re; retval.im += csrc1.im ) */
CPX_PROJECTION_METHOD_2( cvsum, cdst.re = 0 ; cdst.im = 0 , cdst.re += csrc1.re; cdst.im += csrc1.im )
/* TWO_CPX_VEC_SCALRET_METHOD( cvdot, retval.re += csrc1.re * csrc2.re - csrc1.im * csrc2.im; retval.im += csrc1.re * csrc2.im + csrc1.im * csrc2.re  ) */
CPX_PROJECTION_METHOD_3( cvdot, cdst.re = 0; cdst.im = 0 , cdst.re += csrc1.re * csrc2.re - csrc1.im * csrc2.im; cdst.im += csrc1.re * csrc2.im + csrc1.im * csrc2.re  )

#ifndef GPU_FUNCTION
TWO_CPX_VEC_METHOD( cvrand , cdst.re = rn(csrc1.re); cdst.im = rn(csrc1.im)	)
#endif /* ! GPU_FUNCTION */


#ifdef QUATERNION_SUPPORT

/* Quaternions */

TWO_QUAT_VEC_METHOD( qvmov ,		qdst.re = qsrc1.re ;
					qdst._i = qsrc1._i ;
					qdst._j = qsrc1._j ;
					qdst._k = qsrc1._k )

TWO_QUAT_VEC_METHOD( qvneg ,		qdst.re = - qsrc1.re ;
					qdst._i = - qsrc1._i ;
					qdst._j = - qsrc1._j ;
					qdst._k = - qsrc1._k )

TWO_QUAT_VEC_METHOD( qvsqr ,		tmpq.re =	  qsrc1.re * qsrc1.re
							- qsrc1._i * qsrc1._i
							- qsrc1._j * qsrc1._j
							- qsrc1._k * qsrc1._k;
					tmpq._i =	2 * qsrc1.re * qsrc1._i;
					tmpq._j =	2 * qsrc1.re * qsrc1._j;
					tmpq._k =	2 * qsrc1.re * qsrc1._k;
					qdst = tmpq )

THREE_QUAT_VEC_METHOD( qvadd , 		qdst.re = qsrc1.re + qsrc2.re ;
					qdst._i = qsrc1._i + qsrc2._i ;
					qdst._j = qsrc1._j + qsrc2._j ;
					qdst._k = qsrc1._k + qsrc2._k )

THREE_QUAT_VEC_METHOD( qvsub ,		qdst.re = qsrc2.re - qsrc1.re ;
					qdst._i = qsrc2._i - qsrc1._i ;
					qdst._j = qsrc2._j - qsrc1._j ;
					qdst._k = qsrc2._k - qsrc1._k )

/* Here is the multiplication chart:
 *
 *	X	1	_i	_j	_k
 *	1	1	_i	_j	_k
 *	_i	_i	-1	_k	-_j
 *	_j	_j	-_k	-1	_i
 *	_k	_k	_j	-_i	-1
 */

THREE_QUAT_VEC_METHOD( qvmul ,		tmpq.re = 	  qsrc1.re * qsrc2.re
							- qsrc1._i * qsrc2._i
							- qsrc1._j * qsrc2._j
							- qsrc1._k * qsrc2._k;
					tmpq._i =	  qsrc1.re * qsrc2._i
							+ qsrc1._i * qsrc2.re
							+ qsrc1._j * qsrc2._k
							- qsrc1._k * qsrc2._j;
					tmpq._j =	  qsrc1.re * qsrc2._j
							+ qsrc1._j * qsrc2.re
							+ qsrc1._k * qsrc2._i
							- qsrc1._i * qsrc2._k;
					tmpq._k =	  qsrc1.re * qsrc2._k
							+ qsrc1._k * qsrc2.re
							+ qsrc1._i * qsrc2._j
							- qsrc1._j * qsrc2._i;
					qdst = tmpq )


/* float times quaternion */

THREE_QMIXD_VEC_METHOD( pvadd ,		qdst.re = qsrc1.re + src2;
					qdst._i = qsrc1._i ;
					qdst._j = qsrc1._j ;
					qdst._k = qsrc1._k )
					
THREE_QMIXD_VEC_METHOD( pvsub ,		qdst.re = qsrc1.re - src2;
					qdst._i = qsrc1._i ;
					qdst._j = qsrc1._j ;
					qdst._k = qsrc1._k )

THREE_QMIXD_VEC_METHOD( pvmul ,		qdst.re = qsrc1.re * src2;
					qdst._i = qsrc1._i * src2 ;
					qdst._j = qsrc1._j * src2 ;
					qdst._k = qsrc1._k * src2 )

THREE_QMIXD_VEC_METHOD( pvdiv ,		qdst.re = qsrc1.re / src2;
					qdst._i = qsrc1._i / src2 ;
					qdst._j = qsrc1._j / src2 ;
					qdst._k = qsrc1._k / src2 )

/* BUG check for divide by zero */
TWO_QMIXD_QR_VEC_SCALAR_METHOD( pvsdiv ,	qdst.re = scalar1_val / src1 ;
					qdst._i = 0;
					qdst._j = 0;
					qdst._k = 0
					)

TWO_QMIXD_QR_VEC_SCALAR_METHOD( pvsdiv2 ,	qdst.re = src1 / scalar1_val ;
					qdst._i = 0;
					qdst._j = 0;
					qdst._k = 0
					)

TWO_QMIXD_QR_VEC_SCALAR_METHOD( pvsmul ,	qdst.re = scalar1_val * src1 ;
					qdst._i = 0;
					qdst._j = 0;
					qdst._k = 0
					)

TWO_QMIXD_QR_VEC_SCALAR_METHOD( pvssub ,	qdst.re = scalar1_val - src1 ;
					qdst._i = 0;
					qdst._j = 0;
					qdst._k = 0
					)

TWO_QMIXD_QR_VEC_SCALAR_METHOD( pvsadd ,	qdst.re = src1 + scalar1_val ;
					qdst._i = 0;
					qdst._j = 0;
					qdst._k = 0
					)

TWO_QUAT_VEC_SCALAR_METHOD( qvsadd ,	qdst.re = qsrc1.re + qscalar1_val.re ;
					qdst._i = qsrc1._i + qscalar1_val._i ;
					qdst._j = qsrc1._j + qscalar1_val._j ;
					qdst._k = qsrc1._k + qscalar1_val._k
					)

TWO_QUAT_VEC_SCALAR_METHOD( qvssub ,	qdst.re = qscalar1_val.re - qsrc1.re ;
					qdst._i = qscalar1_val._i - qsrc1._i ;
					qdst._j = qscalar1_val._j - qsrc1._j ;
					qdst._k = qscalar1_val._k - qsrc1._k
					)

/* For real or complex, multiplication is commutative, but not for quaternions!? */

TWO_QUAT_VEC_SCALAR_METHOD( qvsmul ,	tmpq.re = 	  qsrc1.re * qscalar1_val.re
							- qsrc1._i * qscalar1_val._i
							- qsrc1._j * qscalar1_val._j
							- qsrc1._k * qscalar1_val._k;
					tmpq._i =	  qsrc1.re * qscalar1_val._i
							+ qsrc1._i * qscalar1_val.re
							+ qsrc1._j * qscalar1_val._k
							- qsrc1._k * qscalar1_val._j;
					tmpq._j =	  qsrc1.re * qscalar1_val._j
							+ qsrc1._j * qscalar1_val.re
							+ qsrc1._k * qscalar1_val._i
							- qsrc1._i * qscalar1_val._k;
					tmpq._k =	  qsrc1.re * qscalar1_val._k
							+ qsrc1._k * qscalar1_val.re
							+ qsrc1._i * qscalar1_val._j
							- qsrc1._j * qscalar1_val._i;
					qdst = tmpq )

TWO_QUAT_VEC_SCALAR_METHOD( qvsmul2 ,	tmpq.re = 	  qscalar1_val.re * qsrc1.re
							- qscalar1_val._i * qsrc1._i
							- qscalar1_val._j * qsrc1._j
							- qscalar1_val._k * qsrc1._k;
					tmpq._i =	  qscalar1_val.re * qsrc1._i
							+ qscalar1_val._i * qsrc1.re
							+ qscalar1_val._j * qsrc1._k
							- qscalar1_val._k * qsrc1._j;
					tmpq._j =	  qscalar1_val.re * qsrc1._j
							+ qscalar1_val._j * qsrc1.re
							+ qscalar1_val._k * qsrc1._i
							- qscalar1_val._i * qsrc1._k;
					tmpq._k =	  qscalar1_val.re * qsrc1._k
							+ qscalar1_val._k * qsrc1.re
							+ qscalar1_val._i * qsrc1._j
							- qscalar1_val._j * qsrc1._i;
					qdst = tmpq )

ONE_QUAT_VEC_SCALAR_METHOD( qvset ,	qdst.re = qscalar1_val.re ;
					qdst._i = qscalar1_val._i ;
					qdst._j = qscalar1_val._j ;
					qdst._k = qscalar1_val._k )


/*
QUAT_VV_SELECTION_METHOD( qvvv_slct ,	qdst = srcbit ? qsrc1 : qsrc2 )
QUAT_VS_SELECTION_METHOD( qvvs_slct ,	qdst = srcbit ? qsrc1 : qscalar1_val )
QUAT_SS_SELECTION_METHOD( qvss_slct ,	qdst = srcbit ? qscalar1_val : qscalar2_val )
*/
#define COPY_QUAT( q )	{ qdst.re=q.re; qdst._i=q._i; qdst._j=q._j; qdst._k=q._k; }

QUAT_VV_SELECTION_METHOD( qvvv_slct ,	if( srcbit ) COPY_QUAT(qsrc1) else COPY_QUAT(qsrc2) )
QUAT_VS_SELECTION_METHOD( qvvs_slct ,	if( srcbit ) COPY_QUAT(qsrc1) else COPY_QUAT(qscalar1_val) )
QUAT_SS_SELECTION_METHOD( qvss_slct ,	if( srcbit ) COPY_QUAT(qscalar1_val) else COPY_QUAT(qscalar2_val) )


QUAT_PROJECTION_METHOD_2( qvsum,
					qdst.re = 0 ;
					qdst._i = 0 ;
					qdst._j = 0 ;
					qdst._k = 0
					,
					qdst.re += qsrc1.re ;
					qdst._i += qsrc1._i ;
					qdst._j += qsrc1._j ;
					qdst._k += qsrc1._k
					)


IMPOSSIBLE_METHOD( qvdiv )
IMPOSSIBLE_METHOD( qvsdiv )
IMPOSSIBLE_METHOD( qvsdiv2 )

#endif /* QUATERNION_SUPPORT */

