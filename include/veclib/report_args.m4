
/**************** Things that do the actual printing *************/

ifdef(`MORE_DEBUG',`

/* `MORE_DEBUG' is defined */

dnl  These can be defined to print for debugging...
define(`REPORT_DIM3_VAR',`						\
	sprintf(DEFAULT_ERROR_STRING,"%s:  %d %d %d\n",#$1,$1.x,$1.y,$1.z);	\
	NADVISE(DEFAULT_ERROR_STRING);\
')

define(`REPORT_DIM5_VAR',`						\
	sprintf(DEFAULT_ERROR_STRING,"%s:  %d %d %d %d %d\n",#$1,$1.d5_dim[0],$1.d5_dim[1],$1.d5_dim[2],$1.d5_dim[3],$1.d5_dim[4]);	\
	NADVISE(DEFAULT_ERROR_STRING);	\
')

define(`REPORT_DST_PTR',`								\
										\
sprintf(DEFAULT_ERROR_STRING,"dst = 0x%lx",					\
(int_for_addr)VA_DEST_PTR(vap));						\
NADVISE(DEFAULT_ERROR_STRING);							\
')

dnl  BUG VA_ARGSET_PREC appears to be invalid - not copied from oap?

define(`REPORT_SVAL1',`								\
										\
sprintf(DEFAULT_ERROR_STRING,"sval1 = %s",					\
string_for_scalar(DEFAULT_QSP_ARG  &(SVAL_STD(VA_SVAL1(vap))),					\
src_prec_for_argset_prec(VA_ARGSET_PREC(vap),VA_ARGSET_TYPE(vap)) ) );		\
NADVISE(DEFAULT_ERROR_STRING);							\
')

',`	dnl #else // ! MORE_DEBUG

/* `MORE_DEBUG' is not defined */

define(`REPORT_DIM3_VAR',`')
define(`REPORT_DST_PTR',`')
define(`REPORT_SVAL1',`')

') dnl #endif // ! MORE_DEBUG

/************* Fast Args ***********/

define(`REPORT_FAST_ARGS_',`')
define(`REPORT_FAST_ARGS_1',`')
define(`REPORT_FAST_ARGS_2',`')
define(`REPORT_FAST_ARGS_CONV',`')
define(`REPORT_FAST_ARGS_3',`')
define(`REPORT_FAST_ARGS_CPX_3',`')
define(`REPORT_FAST_ARGS_CPX_1S_1',`')
define(`REPORT_FAST_ARGS_4',`')
define(`REPORT_FAST_ARGS_5',`')
define(`REPORT_FAST_ARGS_1S_1',`')
define(`REPORT_FAST_ARGS_1S_2',`')
define(`REPORT_FAST_ARGS_2S_1',`')
define(`REPORT_FAST_ARGS_3S_1',`')
define(`REPORT_FAST_ARGS_1S_4',`')
define(`REPORT_FAST_ARGS_2S_3',`')
define(`REPORT_FAST_ARGS_3S_2',`')
define(`REPORT_FAST_ARGS_SBM_3',`')
define(`REPORT_FAST_ARGS_SBM_1S_2',`')
dnl define(`REPORT_FAST_ARGS_SBM_2S_1

define(`REPORT_FAST_ARGS_SBM_2S_1',`')
define(`REPORT_FAST_ARGS_1S_2',`')

dnl  bit_rvset
define(`REPORT_FAST_ARGS_DBM_1S_',`REPORT_DST_PTR REPORT_SVAL1')

define(`REPORT_FAST_ARGS_DBM_1S_1SRC',`')
define(`REPORT_FAST_ARGS_DBM_2SRCS',`')
define(`REPORT_FAST_ARGS_DBM_SBM',`')
define(`REPORT_FAST_ARGS_RC_2',`')

define(`REPORT_FAST_ARGS_CCR_3',`')
define(`REPORT_FAST_ARGS_CPX_2',`')
define(`REPORT_FAST_ARGS_CR_1S_2',`')
define(`REPORT_FAST_ARGS_CPX_1S_2',`')
define(`REPORT_FAST_ARGS_QUAT_2',`')
define(`REPORT_FAST_ARGS_SBM_CPX_1S_2',`')
define(`REPORT_FAST_ARGS_SBM_CPX_2S_1',`')
define(`REPORT_FAST_ARGS_SBM_CPX_3',`')
define(`REPORT_FAST_ARGS_QUAT_3',`')
define(`REPORT_FAST_ARGS_QQR_3',`')
define(`REPORT_FAST_ARGS_QR_1S_2',`')
define(`REPORT_FAST_ARGS_QUAT_1S_2',`')
define(`REPORT_FAST_ARGS_QUAT_1S_1',`')
define(`REPORT_FAST_ARGS_SBM_QUAT_1S_2',`')
define(`REPORT_FAST_ARGS_SBM_QUAT_2S_1',`')
define(`REPORT_FAST_ARGS_SBM_QUAT_3',`')


/************* EqSp Args ***********/

dnl REPORT_EQSP_ARGS(bitmap,typ,scalars,vectors)
define(`REPORT_EQSP_ARGS',REPORT_EQSP_ARGS_$1$2$3$4)

define(`REPORT_EQSP_ARGS_',`')
define(`REPORT_EQSP_ARGS_1',`')
define(`REPORT_EQSP_ARGS_2',`')
define(`REPORT_EQSP_ARGS_CONV',`')
define(`REPORT_EQSP_ARGS_3',`')
define(`REPORT_EQSP_ARGS_CPX_3',`')
define(`REPORT_EQSP_ARGS_CPX_1S_1',`')
define(`REPORT_EQSP_ARGS_4',`')
define(`REPORT_EQSP_ARGS_5',`')
define(`REPORT_EQSP_ARGS_1S_1',`')
define(`REPORT_EQSP_ARGS_1S_2',`')
define(`REPORT_EQSP_ARGS_2S_1',`')
define(`REPORT_EQSP_ARGS_3S_1',`')
define(`REPORT_EQSP_ARGS_1S_4',`')
define(`REPORT_EQSP_ARGS_2S_3',`')
define(`REPORT_EQSP_ARGS_3S_2',`')
define(`REPORT_EQSP_ARGS_SBM_3',`')
define(`REPORT_EQSP_ARGS_SBM_1S_2',`')
dnl define(`REPORT_EQSP_ARGS_SBM_2S_1',`')

define(`REPORT_EQSP_ARGS_SBM_2S_1',`')
define(`REPORT_EQSP_ARGS_1S_2',`')
define(`REPORT_EQSP_ARGS_DBM_1S_',`')
define(`REPORT_EQSP_ARGS_DBM_1S_1SRC',`')
define(`REPORT_EQSP_ARGS_DBM_2SRCS',`')
define(`REPORT_EQSP_ARGS_DBM_SBM',`')
define(`REPORT_EQSP_ARGS_RC_2',`')

define(`REPORT_EQSP_ARGS_CCR_3',`')
define(`REPORT_EQSP_ARGS_CPX_2',`')
define(`REPORT_EQSP_ARGS_CR_1S_2',`')
define(`REPORT_EQSP_ARGS_CPX_1S_2',`')
define(`REPORT_EQSP_ARGS_QUAT_2',`')
define(`REPORT_EQSP_ARGS_SBM_CPX_1S_2',`')
define(`REPORT_EQSP_ARGS_SBM_CPX_2S_1',`')
define(`REPORT_EQSP_ARGS_SBM_CPX_3',`')
define(`REPORT_EQSP_ARGS_QUAT_3',`')
define(`REPORT_EQSP_ARGS_QQR_3',`')
define(`REPORT_EQSP_ARGS_QR_1S_2',`')
define(`REPORT_EQSP_ARGS_QUAT_1S_2',`')
define(`REPORT_EQSP_ARGS_QUAT_1S_1',`')
define(`REPORT_EQSP_ARGS_SBM_QUAT_1S_2',`')
define(`REPORT_EQSP_ARGS_SBM_QUAT_2S_1',`')
define(`REPORT_EQSP_ARGS_SBM_QUAT_3',`')



/************* Slow Args ***********/

dnl REPORT_SLOW_ARGS(bitmap,typ,scalars,vectors)
define(`REPORT_SLOW_ARGS',REPORT_SLOW_ARGS_$1$2$3$4)
define(`REPORT_FAST_ARGS',`/* report_fast_args "$1" "$2" "$3" "$4" */ REPORT_FAST_ARGS_$1$2$3$4 /* report_fast_args "$1" "$2" "$3" "$4"  */')

define(`REPORT_SLOW_ARGS_',`')
define(`REPORT_SLOW_ARGS_1',`')
define(`REPORT_SLOW_ARGS_2',`')
define(`REPORT_SLOW_ARGS_CONV',`')
define(`REPORT_SLOW_ARGS_3',`')
define(`REPORT_SLOW_ARGS_CPX_3',`')
define(`REPORT_SLOW_ARGS_CPX_1S_1',`')
define(`REPORT_SLOW_ARGS_4',`')
define(`REPORT_SLOW_ARGS_5',`')


define(`REPORT_SLOW_ARGS_1S_1',`REPORT_DST_PTR REPORT_SVAL1')

define(`REPORT_SLOW_ARGS_1S_2',`')
define(`REPORT_SLOW_ARGS_2S_1',`')
define(`REPORT_SLOW_ARGS_3S_1',`')
define(`REPORT_SLOW_ARGS_1S_4',`')
define(`REPORT_SLOW_ARGS_2S_3',`')
define(`REPORT_SLOW_ARGS_3S_2',`')
define(`REPORT_SLOW_ARGS_SBM_3',`')
define(`REPORT_SLOW_ARGS_SBM_1S_2',`')
dnl define(`REPORT_SLOW_ARGS_SBM_2S_1',`')

define(`REPORT_SLOW_ARGS_SBM_2S_1',`')
define(`REPORT_SLOW_ARGS_1S_2',`')
define(`REPORT_SLOW_ARGS_DBM_1S_',`')
define(`REPORT_SLOW_ARGS_DBM_1S_1SRC',`')
define(`REPORT_SLOW_ARGS_DBM_2SRCS',`')
define(`REPORT_SLOW_ARGS_DBM_SBM',`')
define(`REPORT_SLOW_ARGS_RC_2',`')

define(`REPORT_SLOW_ARGS_CCR_3',`')
define(`REPORT_SLOW_ARGS_CPX_2',`')
define(`REPORT_SLOW_ARGS_CR_1S_2',`')
define(`REPORT_SLOW_ARGS_CPX_1S_2',`')
define(`REPORT_SLOW_ARGS_QUAT_2',`')
define(`REPORT_SLOW_ARGS_SBM_CPX_1S_2',`')
define(`REPORT_SLOW_ARGS_SBM_CPX_2S_1',`')
define(`REPORT_SLOW_ARGS_SBM_CPX_3',`')
define(`REPORT_SLOW_ARGS_QUAT_3',`')
define(`REPORT_SLOW_ARGS_QQR_3',`')
define(`REPORT_SLOW_ARGS_QR_1S_2',`')
define(`REPORT_SLOW_ARGS_QUAT_1S_2',`')
define(`REPORT_SLOW_ARGS_QUAT_1S_1',`')
define(`REPORT_SLOW_ARGS_SBM_QUAT_1S_2',`')
define(`REPORT_SLOW_ARGS_SBM_QUAT_2S_1',`')
define(`REPORT_SLOW_ARGS_SBM_QUAT_3',`')



/************* Slen Args ***********/

define(`REPORT_SLEN_ARGS_',`')
define(`REPORT_SLEN_ARGS_1',`')
define(`REPORT_SLEN_ARGS_2',`')
define(`REPORT_SLEN_ARGS_CONV',`')
define(`REPORT_SLEN_ARGS_3',`')
define(`REPORT_SLEN_ARGS_CPX_3',`')
define(`REPORT_SLEN_ARGS_CPX_1S_1',`')
define(`REPORT_SLEN_ARGS_4',`')
define(`REPORT_SLEN_ARGS_5',`')

define(`REPORT_SLEN_ARGS_1S_1',`REPORT_SLOW_ARGS_1S_1 REPORT_DIM3_VAR(VA_XYZ_LEN(vap))')

define(`REPORT_SLEN_ARGS_1S_2',`')
define(`REPORT_SLEN_ARGS_2S_1',`')
define(`REPORT_SLEN_ARGS_3S_1',`')
define(`REPORT_SLEN_ARGS_1S_4',`')
define(`REPORT_SLEN_ARGS_2S_3',`')
define(`REPORT_SLEN_ARGS_3S_2',`')
define(`REPORT_SLEN_ARGS_SBM_3',`')
define(`REPORT_SLEN_ARGS_SBM_1S_2',`')
dnl define(`REPORT_SLEN_ARGS_SBM_2S_1',`')

define(`REPORT_SLEN_ARGS_SBM_2S_1',`')
define(`REPORT_SLEN_ARGS_1S_2',`')
define(`REPORT_SLEN_ARGS_DBM_1S_',`')
define(`REPORT_SLEN_ARGS_DBM_1S_1SRC',`')
define(`REPORT_SLEN_ARGS_DBM_2SRCS',`')
define(`REPORT_SLEN_ARGS_DBM_SBM',`')
define(`REPORT_SLEN_ARGS_RC_2',`')

define(`REPORT_SLEN_ARGS_CCR_3',`')
define(`REPORT_SLEN_ARGS_CPX_2',`')
define(`REPORT_SLEN_ARGS_CR_1S_2',`')
define(`REPORT_SLEN_ARGS_CPX_1S_2',`')
define(`REPORT_SLEN_ARGS_QUAT_2',`')
define(`REPORT_SLEN_ARGS_SBM_CPX_1S_2',`')
define(`REPORT_SLEN_ARGS_SBM_CPX_2S_1',`')
define(`REPORT_SLEN_ARGS_SBM_CPX_3',`')
define(`REPORT_SLEN_ARGS_QUAT_3',`')
define(`REPORT_SLEN_ARGS_QQR_3',`')
define(`REPORT_SLEN_ARGS_QR_1S_2',`')
define(`REPORT_SLEN_ARGS_QUAT_1S_2',`')
define(`REPORT_SLEN_ARGS_QUAT_1S_1',`')
define(`REPORT_SLEN_ARGS_SBM_QUAT_1S_2',`')
define(`REPORT_SLEN_ARGS_SBM_QUAT_2S_1',`')
define(`REPORT_SLEN_ARGS_SBM_QUAT_3',`')


/* end of report_args.m4 */
