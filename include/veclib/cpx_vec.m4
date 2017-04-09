/* complex number stuff */

my_include(`veclib/cpx_args.m4')

/* Real only */

// for debugging...
define(`SHOW_CPX_3',
	sprintf(error_string,"csrc1 = %g  %g",csrc1.re,csrc1.im);
	advise(error_string);
	sprintf(error_string,"csrc2 = %g  %g",csrc2.re,csrc2.im);
	advise(error_string);
)

define(`CPX_PROD_RE',	( $1.re*$2.re - $1.im*$2.im ) )
define(`CPX_PROD_IM',	( $1.re*$2.im + $1.im*$2.re ) )

_VEC_FUNC_CPX_2V( cvmov ,		cdst.re = csrc1.re ; cdst.im = csrc1.im )
_VEC_FUNC_CPX_2V( cvneg ,		cdst.re = - csrc1.re ; cdst.im = - csrc1.im )
_VEC_FUNC_CPX_2V_T2( cvsqr ,		tmpc.re = CPX_PROD_RE(csrc1,csrc1);		\
					tmpc.im = csrc1.re*csrc1.im*2 ;			\
					ASSIGN_CPX(cdst,tmpc); )

_VEC_FUNC_CPX_2V( vconj ,		cdst.re = csrc1.re;				\
					cdst.im = - csrc1.im )

// complex exponential - e ^ a + bi = e^a (cos b + i sin b)
// This can work in-place...
_VEC_FUNC_CPX_2V_T2( cvexp ,		tmpc.re = exp_func (csrc1.re);			\
					cdst.re = tmpc.re * cos_func (csrc1.im);			\
					cdst.im = tmpc.re * sin_func (csrc1.im);			\
					)

_VEC_FUNC_CPX_3V( cvadd , 	cdst.re = csrc1.re + csrc2.re ; cdst.im = csrc1.im + csrc2.im )
_VEC_FUNC_CPX_3V( cvsub ,	cdst.re = csrc2.re - csrc1.re ; cdst.im = csrc2.im - csrc1.im )

define(`ASSIGN_CPX_PROD',{ $1.re = CPX_PROD_RE($2,$3); $1.im = CPX_PROD_IM($2,$3); } )

_VEC_FUNC_CPX_3V_T2( cvmul ,	ASSIGN_CPX_PROD(tmpc,csrc1,csrc2)			\
				ASSIGN_CPX(cdst,tmpc) )

define(`CPX_NORM',( $1.re*$1.re +$1.im*$1.im ))

// conjugate second operand
define(`CPX_CPROD_RE',( $1.re*$2.re + $1.im*$2.im ))
define(`CPX_CPROD_IM',( $2.re*$1.im - $2.im*$1.re ))

define(`ASSIGN_CPX_CPROD',{ $1.re = CPX_CPROD_RE($2,$3); $1.im = CPX_CPROD_IM($2,$3); })

define(`ASSIGN_CPX_CPROD_NORM',{ $1.re = CPX_CPROD_RE($2,$3)/$4; $1.im = CPX_CPROD_IM($2,$3)/$4; })

// are we dividing by src1 or src2 ???

_VEC_FUNC_CPX_3V_T3( cvdiv ,	tmp_denom = CPX_NORM(csrc2);				\
				ASSIGN_CPX_CPROD_NORM(tmpc,csrc1,csrc2,tmp_denom)				\
				ASSIGN_CPX(cdst,tmpc) )

/* vcmul seems to be redundant with cvmul, but actually it is conjugate multiplication? */
_VEC_FUNC_CPX_3V_T2( vcmul , 	ASSIGN_CPX_CPROD(tmpc,csrc1,csrc2)				\
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
_VEC_FUNC_CR_2V_1S( mvsdiv ,	cdst.re = scalar1_val / src1 ; cdst.im = 0)

_VEC_FUNC_CR_2V_1S( mvsdiv2 ,	cdst.re = src1 / scalar1_val ; cdst.im = 0)

_VEC_FUNC_CR_2V_1S( mvsmul ,	cdst.re = scalar1_val * src1 ; cdst.im = 0)

_VEC_FUNC_CR_2V_1S( mvssub ,	cdst.re = scalar1_val - src1 ; cdst.im = 0)
_VEC_FUNC_CR_2V_1S( mvssub2 ,	cdst.re = src1 - scalar1_val ; cdst.im = 0)

_VEC_FUNC_CR_2V_1S( mvsadd ,	cdst.re = src1 + scalar1_val ; cdst.im = 0)

_VEC_FUNC_CPX_2V_1S( cvsadd ,	cdst.re = csrc1.re + cscalar1_val.re ; cdst.im = csrc1.im + cscalar1_val.im)

_VEC_FUNC_CPX_2V_1S( cvssub ,	cdst.re = cscalar1_val.re - csrc1.re ; cdst.im = cscalar1_val.im - csrc1.im )
_VEC_FUNC_CPX_2V_1S( cvssub2 ,	cdst.re = csrc1.re - cscalar1_val.re ; cdst.im = csrc1.im - cscalar1_val.im )
_VEC_FUNC_CPX_2V_1S_T2( cvsmul ,	ASSIGN_CPX_PROD(tmpc,csrc1,cscalar1_val) ASSIGN_CPX(cdst,tmpc))

define(`QUAT_NORM',( $1.re * $1.re + $1._i * $1._i + $1._j * $1._j + $1._k * $1._k ))

// dst = scalar / src
_VEC_FUNC_CPX_2V_1S_T3( cvsdiv ,	tmp_denom=CPX_NORM(csrc1);					\
					ASSIGN_CPX_CPROD_NORM(tmpc,cscalar1_val,csrc1,tmp_denom)					\
					ASSIGN_CPX(cdst,tmpc)					\
					)

_VEC_FUNC_CPX_2V_1S_T3( cvsdiv2 ,	tmp_denom=CPX_NORM(cscalar1_val);					\
					ASSIGN_CPX_CPROD_NORM(tmpc,csrc1,cscalar1_val,tmp_denom)					\
					ASSIGN_CPX(cdst,tmpc)					\
					)

/* for mixed (MDECLS2), v1 is complex and v2 is real? */


_VEC_FUNC_CPX_1V_1S( cvset , cdst.re = cscalar1_val.re ; cdst.im = cscalar1_val.im )

/* We use std_tmp here so this will work if called w/ in-place argument (v3 = v1 or v2)
 */



/* conjugated vsmul ? */

_VEC_FUNC_CPX_2V_1S_T2( vscml ,	ASSIGN_CPX_CPROD(tmpc,csrc1,cscalar1_val) ASSIGN_CPX(cdst,tmpc) )


dnl /* These are clean, but don't work when destination and source have different precisions */
dnl /*
dnl _VEC_FUNC_SBM_CPX_3V( cvvv_slct ,	cdst = srcbit ? csrc1 : csrc2 )
dnl CPX_VS_SELECTION_METHOD( cvvs_slct ,	cdst = srcbit ? csrc1 : cscalar1_val )
dnl CPX_SS_SELECTION_METHOD( cvss_slct ,	cdst = srcbit ? cscalar1_val : cscalar2_val )
dnl */

// why not structure assign?
define(`COPY_CPX',{ cdst.re=$1.re; cdst.im=$1.im; })

_VEC_FUNC_SBM_CPX_3V( cvvv_slct ,	if( srcbit ) COPY_CPX(csrc1) else COPY_CPX(csrc2) )

_VEC_FUNC_SBM_CPX_2V_1S( cvvs_slct ,	if( srcbit ) COPY_CPX(csrc1) else COPY_CPX(cscalar1_val) )
_VEC_FUNC_SBM_CPX_1V_2S( cvss_slct ,	if( srcbit ) COPY_CPX(cscalar1_val) else COPY_CPX(cscalar2_val) )

// no CPX in this macro?
_VEC_FUNC_CPX_2V_PROJ( cvsum,						\
	cdst.re = 0 ; cdst.im = 0 ,						\
	cdst.re += csrc1.re; cdst.im += csrc1.im ,						\
	psrc1.re + psrc2.re,						\
	psrc1.im + psrc2.im						\
	)

dnl Need to implement cvdot as composite of cvmul and cvsum BUG
_VEC_FUNC_CPX_3V_PROJ( cvdot,
	cdst.re = 0; cdst.im = 0 ,
	cdst.re += csrc1.re * csrc2.re - csrc1.im * csrc2.im; cdst.im += csrc1.re * csrc2.im + csrc1.im * csrc2.re ,
	csrc1.re*csrc2.re-csrc1.im*csrc2.im,
	csrc1.re*csrc2.im+csrc1.im*csrc2.re,
	csrc1.re+csrc2.re,
	csrc1.im+csrc2.im
	)

ifdef(`BUILD_FOR_GPU',`',`	dnl	ifndef BUILD_FOR_GPU
_VEC_FUNC_CPX_2V( cvrand , cdst.re = rn((u_long)csrc1.re); cdst.im = rn((u_long)csrc1.im)	)
')				dnl	endif /* ! BUILD_FOR_GPU */


ifdef(`QUATERNION_SUPPORT',`

my_include(`veclib/quat_args.m4')

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

define(`QUAT_PROD_RE',`( $1.re * $2.re - $1._i * $2._i - $1._j * $2._j - $1._k * $2._k )')
define(`QUAT_PROD_I',`( $1.re * $2._i + $1._i * $2.re + $1._j * $2._k - $1._k * $2._j )')
define(`QUAT_PROD_J',`( $1.re * $2._j + $1._j * $2.re + $1._k * $2._i - $1._i * $2._k )')
define(`QUAT_PROD_K',`( $1.re * $2._k + $1._k * $2.re + $1._i * $2._j - $1._j * $2._i )')

_VEC_FUNC_QUAT_2V( qvmov ,		qdst.re = qsrc1.re ;			\
					qdst._i = qsrc1._i ;			\
					qdst._j = qsrc1._j ;			\
					qdst._k = qsrc1._k )

_VEC_FUNC_QUAT_2V( qvneg ,		qdst.re = - qsrc1.re ;			\
					qdst._i = - qsrc1._i ;			\
					qdst._j = - qsrc1._j ;			\
					qdst._k = - qsrc1._k )

_VEC_FUNC_QUAT_2V_T4( qvsqr ,		tmpq.re = QUAT_PROD_RE(qsrc1,qsrc1);			\
					tmpq._i =	2 * qsrc1.re * qsrc1._i;			\
					tmpq._j =	2 * qsrc1.re * qsrc1._j;			\
					tmpq._k =	2 * qsrc1.re * qsrc1._k;			\
					ASSIGN_QUAT(qdst,tmpq) )

_VEC_FUNC_QUAT_3V( qvadd , 		qdst.re = qsrc1.re + qsrc2.re ;			\
					qdst._i = qsrc1._i + qsrc2._i ;			\
					qdst._j = qsrc1._j + qsrc2._j ;			\
					qdst._k = qsrc1._k + qsrc2._k )

_VEC_FUNC_QUAT_3V( qvsub ,		qdst.re = qsrc2.re - qsrc1.re ;			\
					qdst._i = qsrc2._i - qsrc1._i ;			\
					qdst._j = qsrc2._j - qsrc1._j ;			\
					qdst._k = qsrc2._k - qsrc1._k )

define(`ASSIGN_QUAT_PROD',{ $1.re = QUAT_PROD_RE($2,$3); $1._i = QUAT_PROD_I($2,$3); $1._j = QUAT_PROD_J($2,$3); $1._k = QUAT_PROD_K($2,$3); })

_VEC_FUNC_QUAT_3V_T4( qvmul ,	ASSIGN_QUAT_PROD(tmpq,qsrc1,qsrc2)			\
					ASSIGN_QUAT(qdst,tmpq) )

// Quaternion division is like complex division:  q1 / q2 = q1 * conj(q2) / maqsq(q2)
// Conjugation negates all three imaginary components...

// T4 declares tmpq, T5 tmpq plus tmp_denom

define(`QUAT_NORM',	( $1.re * $1.re + $1._i * $1._i + $1._j * $1._j + $1._k * $1._k ) )

// QUAT_CPROD conjugates the second operand...

define(`QUAT_CPROD_RE',	( $1.re * $2.re + $1._i * $2._i + $1._j * $2._j + $1._k * $2._k ) )
define(`QUAT_CPROD_I',	( - $1.re * $2._i + $1._i * $2.re - $1._j * $2._k + $1._k * $2._j ) )
define(`QUAT_CPROD_J',	( - $1.re * $2._j + $1._j * $2.re - $1._k * $2._i + $1._i * $2._k ) )
define(`QUAT_CPROD_K',	( - $1.re * $2._k + $1._k * $2.re - $1._i * $2._j + $1._j * $2._i ) )

define(`ASSIGN_QUAT_CPROD_NORM',`			\
	{ $1.re = QUAT_CPROD_RE($2,$3)/$4;			\
	$1._i = QUAT_CPROD_I($2,$3)/$4;			\
	$1._j = QUAT_CPROD_J($2,$3)/$4;			\
	$1._k = QUAT_CPROD_K($2,$3)/$4; }			\
')

_VEC_FUNC_QUAT_3V_T5( qvdiv ,	tmp_denom =	QUAT_NORM(qsrc2);			\
					ASSIGN_QUAT_CPROD_NORM(tmpq,qsrc1,qsrc2,tmp_denom)			\
					ASSIGN_QUAT(qdst,tmpq) )

// dst = scalar / src1

_VEC_FUNC_QUAT_2V_1S_T5( qvsdiv ,	tmp_denom =	QUAT_NORM(qsrc1);			\
					ASSIGN_QUAT_CPROD_NORM(tmpq,qscalar1_val,qsrc1,tmp_denom)			\
					ASSIGN_QUAT(qdst,tmpq) )

// dst = src1 / scalar

_VEC_FUNC_QUAT_2V_1S_T5( qvsdiv2 ,	tmp_denom =	QUAT_NORM(qscalar1_val);			\
					ASSIGN_QUAT_CPROD_NORM(tmpq,qsrc1,qscalar1_val,tmp_denom)			\
					ASSIGN_QUAT(qdst,tmpq) )


/* float times quaternion */

// These appear to have a quaternion destination, but a real source,
// and real scalar...  whats the point?

_VEC_FUNC_QQR_3V( pvadd ,		qdst.re = qsrc1.re + src2;			\
					qdst._i = qsrc1._i ;			\
					qdst._j = qsrc1._j ;			\
					qdst._k = qsrc1._k )
					
_VEC_FUNC_QQR_3V( pvsub ,		qdst.re = qsrc1.re - src2;			\
					qdst._i = qsrc1._i ;			\
					qdst._j = qsrc1._j ;			\
					qdst._k = qsrc1._k )

_VEC_FUNC_QQR_3V( pvmul ,		qdst.re = qsrc1.re * src2;			\
					qdst._i = qsrc1._i * src2 ;			\
					qdst._j = qsrc1._j * src2 ;			\
					qdst._k = qsrc1._k * src2 )

_VEC_FUNC_QQR_3V( pvdiv ,		qdst.re = qsrc1.re / src2;			\
					qdst._i = qsrc1._i / src2 ;			\
					qdst._j = qsrc1._j / src2 ;			\
					qdst._k = qsrc1._k / src2 )

/* BUG check for divide by zero */
// 
//_VF_QR_2V_1S( name, type_code, stat)
_VEC_FUNC_QR_2V_1S( pvsdiv ,	qdst.re = scalar1_val / src1 ;			\
					qdst._i = 0;			\
					qdst._j = 0;			\
					qdst._k = 0			\
					)

_VEC_FUNC_QR_2V_1S( pvsdiv2 ,	qdst.re = src1 / scalar1_val ;			\
					qdst._i = 0;			\
					qdst._j = 0;			\
					qdst._k = 0			\
					)

_VEC_FUNC_QR_2V_1S( pvsmul ,	qdst.re = scalar1_val * src1 ;			\
					qdst._i = 0;			\
					qdst._j = 0;			\
					qdst._k = 0			\
					)

_VEC_FUNC_QR_2V_1S( pvssub ,	qdst.re = scalar1_val - src1 ;		\
					qdst._i = 0;			\
					qdst._j = 0;			\
					qdst._k = 0			\
					)

_VEC_FUNC_QR_2V_1S( pvssub2 ,	qdst.re = src1 - scalar1_val ;		\
					qdst._i = 0;			\
					qdst._j = 0;			\
					qdst._k = 0			\
					)

_VEC_FUNC_QR_2V_1S( pvsadd ,	qdst.re = src1 + scalar1_val ;		\
					qdst._i = 0;			\
					qdst._j = 0;			\
					qdst._k = 0			\
					)

_VEC_FUNC_QUAT_2V_1S( qvsadd ,	qdst.re = qsrc1.re + qscalar1_val.re ;			\
				qdst._i = qsrc1._i + qscalar1_val._i ;			\
				qdst._j = qsrc1._j + qscalar1_val._j ;			\
				qdst._k = qsrc1._k + qscalar1_val._k			\
				)

_VEC_FUNC_QUAT_2V_1S( qvssub ,	qdst.re = qscalar1_val.re - qsrc1.re ;			\
				qdst._i = qscalar1_val._i - qsrc1._i ;			\
				qdst._j = qscalar1_val._j - qsrc1._j ;			\
				qdst._k = qscalar1_val._k - qsrc1._k			\
				)

_VEC_FUNC_QUAT_2V_1S( qvssub2 ,	qdst.re = qsrc1.re - qscalar1_val.re ;			\
				qdst._i = qsrc1._i - qscalar1_val._i ;			\
				qdst._j = qsrc1._j - qscalar1_val._j ;			\
				qdst._k = qsrc1._k - qscalar1_val._k 			\
				)

/* For real or complex, multiplication is commutative, but not for quaternions!? */

_VEC_FUNC_QUAT_2V_1S_T4( qvsmul ,	ASSIGN_QUAT_PROD(tmpq,qsrc1,qscalar1_val)			\
					ASSIGN_QUAT(qdst,tmpq) )

dnl not yet...
dnl _VEC_FUNC_QUAT_2V_1S_T4( qvsmul2 ,ASSIGN_QUAT_PROD(tmpq,qscalar1_val,qsrc1)			\
dnl 					ASSIGN_QUAT(qdst,tmpq) )

_VEC_FUNC_QUAT_1V_1S( qvset ,	qdst.re = qscalar1_val.re ;			\
					qdst._i = qscalar1_val._i ;			\
					qdst._j = qscalar1_val._j ;			\
					qdst._k = qscalar1_val._k )


/*
QUAT_VV_SELECTION_METHOD( qvvv_slct ,	qdst = srcbit ? qsrc1 : qsrc2 )
QUAT_VS_SELECTION_METHOD( qvvs_slct ,	qdst = srcbit ? qsrc1 : qscalar1_val )
QUAT_SS_SELECTION_METHOD( qvss_slct ,	qdst = srcbit ? qscalar1_val : qscalar2_val )
*/
define(`COPY_QUAT',{ qdst.re=$1.re; qdst._i=$1._i; qdst._j=$1._j; qdst._k=$1._k; })

_VEC_FUNC_SBM_QUAT_3V( qvvv_slct ,	if( srcbit ) COPY_QUAT(qsrc1) else COPY_QUAT(qsrc2) )
_VEC_FUNC_SBM_QUAT_2V_1S( qvvs_slct ,	if( srcbit ) COPY_QUAT(qsrc1) else COPY_QUAT(qscalar1_val) )
_VEC_FUNC_SBM_QUAT_1V_2S( qvss_slct ,	if( srcbit ) COPY_QUAT(qscalar1_val) else COPY_QUAT(qscalar2_val) )

/* BUG - gpu implementation? */

dnl	_VEC_FUNC_QUAT_2V_PROJ( name, init_stat, loop_stat, gpu_e1, gpu_e2, gpu_e3, gpu_e4 )

_VEC_FUNC_QUAT_2V_PROJ( qvsum,			\
					qdst.re = 0 ;			\
					qdst._i = 0 ;			\
					qdst._j = 0 ;			\
					qdst._k = 0			\
					,
					qdst.re += qsrc1.re ;			\
					qdst._i += qsrc1._i ;			\
					qdst._j += qsrc1._j ;			\
					qdst._k += qsrc1._k			\
					,
					psrc1.re + psrc2.re,			\
					psrc1._i + psrc2._i,			\
					psrc1._j + psrc2._j,			\
					psrc1._k + psrc2._k			\
					)

',`') dnl endif /* QUATERNION_SUPPORT */

