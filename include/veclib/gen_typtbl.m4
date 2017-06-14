// This file lays out the tables
// of typed functions.  Unfortunately, it makes assumptions
// about the order of the precisions, so that if those are
// changed, all of this is wrong...  It might be better
// to populate the table programmatically to eliminate
// this problem.

define(`NULL_5',`nullobjf, nullobjf, nullobjf, nullobjf, nullobjf')

define(`NULL_4',`nullobjf, nullobjf, nullobjf, nullobjf')

define(`NULL_3',`nullobjf, nullobjf, nullobjf')

define(`NULL_2',`nullobjf, nullobjf')

define(`DUP_4', $1`,' $1`,' $1`,' $1 )
	
define(`DUP_2', $1`,' $1 )

define(`DUP_3', $1`,' $1`,' $1 )

define(`ALL_NULL',`NULL_4, NULL_2, NULL_4, NULL_5')

define(`DUP_ALL',`DUP_4($1), DUP_2($1), DUP_4($1), DUP_4($1), $2' )

define(`FLT_ALL',`
	NULL_4,
	HOST_TYPED_CALL_NAME($1,sp),
	HOST_TYPED_CALL_NAME($1,dp),
	NULL_4,
	NULL_3,
	HOST_TYPED_CALL_NAME($1,spdp),
	nullobjf
')

define(`FLT_SAME_PREC_ALL',`
	NULL_4,
	HOST_TYPED_CALL_NAME($1,sp),
	HOST_TYPED_CALL_NAME($1,dp),
	NULL_4,
	NULL_4,
	nullobjf
')

define(`RFLT_ALL',`
	NULL_4,
	HOST_TYPED_CALL_NAME_REAL($1,sp),
	HOST_TYPED_CALL_NAME_REAL($1,dp),
	NULL_4,
	NULL_3,
	HOST_TYPED_CALL_NAME_REAL($1,spdp),
	nullobjf
')

define(`RFLT_SAME_PREC_ALL',`
	NULL_4,
	HOST_TYPED_CALL_NAME_REAL($1,sp),
	HOST_TYPED_CALL_NAME_REAL($1,dp),
	NULL_4,
	NULL_4,
	nullobjf
')

// NO_MIXED is the same as SAME_PREC...

define(`RFLT_NO_MIXED',`
	NULL_4,
	HOST_TYPED_CALL_NAME_REAL($1,sp),
	HOST_TYPED_CALL_NAME_REAL($1,dp),
	NULL_4,
	NULL_5
')

define(`CFLT_ALL',`
	NULL_4,
	HOST_TYPED_CALL_NAME_CPX($1,sp),
	HOST_TYPED_CALL_NAME_CPX($1,dp),
	NULL_4,
	NULL_3,
	HOST_TYPED_CALL_NAME_CPX($1,spdp),
	nullobjf
')

define(`CFLT_NO_MIXED',`
	NULL_4,
	HOST_TYPED_CALL_NAME_CPX($1,sp),
	HOST_TYPED_CALL_NAME_CPX($1,dp),
	NULL_4,
	NULL_5
')

define(`MFLT_ALL',`
	NULL_4,
	HOST_TYPED_CALL_NAME_MIXED($1,sp),
	HOST_TYPED_CALL_NAME_MIXED($1,dp),
	NULL_4,
	NULL_3,
	HOST_TYPED_CALL_NAME_MIXED($1,spdp)
')

define(`INT_ALL',`
	HOST_TYPED_CALL_NAME($1,by),
	HOST_TYPED_CALL_NAME($1,in),
	HOST_TYPED_CALL_NAME($1,di),
	HOST_TYPED_CALL_NAME($1,li),
	nullobjf,
	nullobjf,
	HOST_TYPED_CALL_NAME($1,uby),
	HOST_TYPED_CALL_NAME($1,uin),
	HOST_TYPED_CALL_NAME($1,udi),
	HOST_TYPED_CALL_NAME($1,uli),
	HOST_TYPED_CALL_NAME($1,ubyin),
	HOST_TYPED_CALL_NAME($1,inby),
	HOST_TYPED_CALL_NAME($1,uindi),
	nullobjf,
	HOST_TYPED_CALL_NAME($1,bit)
')


/* this is for vneg */

define(`SIGNED_ALL_REAL',`
	HOST_TYPED_CALL_NAME_REAL($1,by),
	HOST_TYPED_CALL_NAME_REAL($1,in),
	HOST_TYPED_CALL_NAME_REAL($1,di),
	HOST_TYPED_CALL_NAME_REAL($1,li),
	HOST_TYPED_CALL_NAME_REAL($1,sp),
	HOST_TYPED_CALL_NAME_REAL($1,dp),
	nullobjf,	/* uby */
	nullobjf,	/* uin */
	nullobjf,	/* udi */
	nullobjf,	/* uli */
	nullobjf,	/* ubyin */
	HOST_TYPED_CALL_NAME_REAL($1,inby),
	nullobjf,	/* uindi */
	HOST_TYPED_CALL_NAME_REAL($1,spdp),
	nullobjf	/* bit */
')

/* 12B means no Bitmap */

define(`INT_ALL_NO_BITMAP',`
	HOST_TYPED_CALL_NAME($1,by),
	HOST_TYPED_CALL_NAME($1,in),
	HOST_TYPED_CALL_NAME($1,di),
	HOST_TYPED_CALL_NAME($1,li),
	nullobjf,
	nullobjf,
	HOST_TYPED_CALL_NAME($1,uby),
	HOST_TYPED_CALL_NAME($1,uin),
	HOST_TYPED_CALL_NAME($1,udi),
	HOST_TYPED_CALL_NAME($1,uli),
	HOST_TYPED_CALL_NAME($1,ubyin),
	HOST_TYPED_CALL_NAME($1,inby),
	HOST_TYPED_CALL_NAME($1,uindi),
	nullobjf,
	nullobjf
')


define(`REAL_INT_ALL',`
	byr$1,  inr$1,  dir$1,  lir$1,
	nullobjf, nullobjf,
	ubyr$1, uinr$1, udir$1, uli$1,
	ubyinr$1, inbyr$1, uindir$1, nullobjf, nullobjf
')


define(`ALL_NO_BITMAP',`
	HOST_TYPED_CALL_NAME($1,by),
	HOST_TYPED_CALL_NAME($1,in),
	HOST_TYPED_CALL_NAME($1,di),
	HOST_TYPED_CALL_NAME($1,li),
	HOST_TYPED_CALL_NAME($1,sp),
	HOST_TYPED_CALL_NAME($1,dp),
	HOST_TYPED_CALL_NAME($1,uby),
	HOST_TYPED_CALL_NAME($1,uin),
	HOST_TYPED_CALL_NAME($1,udi),
	HOST_TYPED_CALL_NAME($1,uli),
	HOST_TYPED_CALL_NAME($1,ubyin),
	HOST_TYPED_CALL_NAME($1,inby),
	HOST_TYPED_CALL_NAME($1,uindi),
	HOST_TYPED_CALL_NAME($1,spdp),
	nullobjf
')

define(`ALL_SAME_PREC_NO_BITMAP',`
	HOST_TYPED_CALL_NAME($1,by),
	HOST_TYPED_CALL_NAME($1,in),
	HOST_TYPED_CALL_NAME($1,di),
	HOST_TYPED_CALL_NAME($1,li),
	HOST_TYPED_CALL_NAME($1,sp),
	HOST_TYPED_CALL_NAME($1,dp),
	HOST_TYPED_CALL_NAME($1,uby),
	HOST_TYPED_CALL_NAME($1,uin),
	HOST_TYPED_CALL_NAME($1,udi),
	HOST_TYPED_CALL_NAME($1,uli),
	NULL_4,
	nullobjf
')

define(`ALL_REAL_SAME_PREC_NO_BITMAP',`
	HOST_TYPED_CALL_NAME_REAL($1,by),
	HOST_TYPED_CALL_NAME_REAL($1,in),
	HOST_TYPED_CALL_NAME_REAL($1,di),
	HOST_TYPED_CALL_NAME_REAL($1,li),
	HOST_TYPED_CALL_NAME_REAL($1,sp),
	HOST_TYPED_CALL_NAME_REAL($1,dp),
	HOST_TYPED_CALL_NAME_REAL($1,uby),
	HOST_TYPED_CALL_NAME_REAL($1,uin),
	HOST_TYPED_CALL_NAME_REAL($1,udi),
	HOST_TYPED_CALL_NAME_REAL($1,uli),
	NULL_4,
	nullobjf
')

ifdef(`BUILD_FOR_GPU',`
define(`NO_GPUBITMAP_FUNC',nullobjf)
',` dnl else // ! BUILD_FOR_GPU
define(`NO_GPUBITMAP_FUNC',`HOST_TYPED_CALL_NAME_REAL($1,bit)')
') dnl endif // ! BUILD_FOR_GPU

define(`ALL_REAL_SAME_PREC_NO_GPUBITMAP',`
	HOST_TYPED_CALL_NAME_REAL($1,by),
	HOST_TYPED_CALL_NAME_REAL($1,in),
	HOST_TYPED_CALL_NAME_REAL($1,di),
	HOST_TYPED_CALL_NAME_REAL($1,li),
	HOST_TYPED_CALL_NAME_REAL($1,sp),
	HOST_TYPED_CALL_NAME_REAL($1,dp),
	HOST_TYPED_CALL_NAME_REAL($1,uby),
	HOST_TYPED_CALL_NAME_REAL($1,uin),
	HOST_TYPED_CALL_NAME_REAL($1,udi),
	HOST_TYPED_CALL_NAME_REAL($1,uli),
	NULL_4,
	NO_GPUBITMAP_FUNC($1)
')


define(`ALL_REAL',`
	HOST_TYPED_CALL_NAME_REAL($1,by),
	HOST_TYPED_CALL_NAME_REAL($1,in),
	HOST_TYPED_CALL_NAME_REAL($1,di),
	HOST_TYPED_CALL_NAME_REAL($1,li),
	HOST_TYPED_CALL_NAME_REAL($1,sp),
	HOST_TYPED_CALL_NAME_REAL($1,dp),
	HOST_TYPED_CALL_NAME_REAL($1,uby),
	HOST_TYPED_CALL_NAME_REAL($1,uin),
	HOST_TYPED_CALL_NAME_REAL($1,udi),
	HOST_TYPED_CALL_NAME_REAL($1,uli),
	HOST_TYPED_CALL_NAME_REAL($1,ubyin),
	HOST_TYPED_CALL_NAME_REAL($1,inby),
	HOST_TYPED_CALL_NAME_REAL($1,uindi),
	HOST_TYPED_CALL_NAME_REAL($1,spdp),
	HOST_TYPED_CALL_NAME_REAL($1,bit)
')

define(`ALL_REAL_SAME_PREC',`
	HOST_TYPED_CALL_NAME_REAL($1,by),
	HOST_TYPED_CALL_NAME_REAL($1,in),
	HOST_TYPED_CALL_NAME_REAL($1,di),
	HOST_TYPED_CALL_NAME_REAL($1,li),
	HOST_TYPED_CALL_NAME_REAL($1,sp),
	HOST_TYPED_CALL_NAME_REAL($1,dp),
	HOST_TYPED_CALL_NAME_REAL($1,uby),
	HOST_TYPED_CALL_NAME_REAL($1,uin),
	HOST_TYPED_CALL_NAME_REAL($1,udi),
	HOST_TYPED_CALL_NAME_REAL($1,uli),
	NULL_4,
	HOST_TYPED_CALL_NAME_REAL($1,bit)
')

define(`ALL_REAL_NO_BITMAP',`
	HOST_TYPED_CALL_NAME_REAL($1,by),
	HOST_TYPED_CALL_NAME_REAL($1,in),
	HOST_TYPED_CALL_NAME_REAL($1,di),
	HOST_TYPED_CALL_NAME_REAL($1,li),
	HOST_TYPED_CALL_NAME_REAL($1,sp),
	HOST_TYPED_CALL_NAME_REAL($1,dp),
	HOST_TYPED_CALL_NAME_REAL($1,uby),
	HOST_TYPED_CALL_NAME_REAL($1,uin),
	HOST_TYPED_CALL_NAME_REAL($1,udi),
	HOST_TYPED_CALL_NAME_REAL($1,uli),
	HOST_TYPED_CALL_NAME_REAL($1,ubyin),
	HOST_TYPED_CALL_NAME_REAL($1,inby),
	HOST_TYPED_CALL_NAME_REAL($1,uindi),
	HOST_TYPED_CALL_NAME_REAL($1,spdp),
	nullobjf
')

define(`ALL_COMPLEX',`
	NULL_4,
	HOST_TYPED_CALL_NAME_CPX($1,sp),
	HOST_TYPED_CALL_NAME_CPX($1,dp),
	NULL_4,
	NULL_3,
	HOST_TYPED_CALL_NAME_CPX($1,spdp),
	nullobjf
')


define(`ALL_MIXED',`
	NULL_4,
	HOST_TYPED_CALL_NAME_MIXED($1,sp),
	HOST_TYPED_CALL_NAME_MIXED($1,dp),
	NULL_4,
	NULL_3,
	HOST_TYPED_CALL_NAME_MIXED($1,spdp),
	nullobjf
')

define(`ALL_QUAT',`
	NULL_4,
	HOST_TYPED_CALL_NAME_QUAT($1,sp),
	HOST_TYPED_CALL_NAME_QUAT($1,dp),
	NULL_4,
	NULL_3,
	HOST_TYPED_CALL_NAME_QUAT($1,spdp),
	nullobjf
')

define(`ALL_QMIXD',`
	NULL_4,
	HOST_TYPED_CALL_NAME_QMIXD($1,sp),
	HOST_TYPED_CALL_NAME_QMIXD($1,dp),
	NULL_4,
	NULL_3,
	HOST_TYPED_CALL_NAME_QMIXD($1,spdp),
	nullobjf
')


define(`NULL_ARR',`{ $2, { ALL_NULL , ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }')

define(`CONV_ARR',`{ $2, { DUP_ALL($1,$3), DUP_ALL($1,$3), ALL_NULL, DUP_ALL($1,$3), ALL_NULL } }')

define(`CFLT_ARR',`{ $2, { ALL_NULL, FLT_ALL($1), ALL_NULL, ALL_NULL, ALL_NULL } }')

define(`RFLT_ARR',`{ $2, { FLT_ALL($1), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }')

define(`RFLT_SAME_PREC_ARR',`{ $2, { FLT_SAME_PREC_ALL($1), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }')

define(`RCFLT_ARR2',`{ $2, { RFLT_NO_MIXED($1), CFLT_NO_MIXED($1), ALL_NULL, ALL_NULL, ALL_NULL } }')

define(`RCFLT_ARR',`{ $2, { RFLT_ALL($1), CFLT_ALL($1), ALL_NULL, ALL_NULL, ALL_NULL } }')

define(`FLT_ARR',`{ $2, { RFLT_ALL($1), CFLT_ALL($1), MFLT_ALL($1), ALL_NULL, ALL_NULL } }')

define(`ALL_ARR',`{ $2, { ALL_REAL_NO_BITMAP($1), ALL_COMPLEX($1), ALL_MIXED($1), ALL_NULL, ALL_NULL } }')

define(`RALL_ARR',`{ $2, { ALL_NO_BITMAP($1), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL	} }')

define(`RALL_SAME_PREC_ARR',`{ $2, { ALL_SAME_PREC_NO_BITMAP($1), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL	} }')

define(`RCALL_ARR',`{ $2, { ALL_REAL_NO_BITMAP($1), ALL_COMPLEX($1), ALL_NULL, ALL_NULL, ALL_NULL } }')

define(`QALL_ARR',`{ $2, { ALL_REAL_NO_BITMAP($1), ALL_COMPLEX($1), ALL_MIXED($1), ALL_QUAT($1), ALL_NULL } }')

define(`RCQALL_ARR',`{ $2, { ALL_REAL_NO_BITMAP($1), ALL_COMPLEX($1), ALL_NULL, ALL_QUAT($1), ALL_NULL } }')

define(`RCQALL_SAME_PREC_ARR',`{ $2, { ALL_REAL_SAME_PREC($1), ALL_COMPLEX($1), ALL_NULL, ALL_QUAT($1), ALL_NULL } }')

define(`RCQPALL_ARR',`{ $2, { ALL_REAL_NO_BITMAP($1), ALL_COMPLEX($1), ALL_NULL, ALL_QUAT($1), ALL_QMIXD($1) } }')

define(`CMALL_ARR',`{ $2, { ALL_NULL, ALL_COMPLEX($1), ALL_MIXED($1), ALL_NULL, ALL_NULL } }')

define(`QPALL_ARR'.`{ $2, { ALL_NULL, ALL_NULL, ALL_NULL, ALL_QUAT($1), ALL_QMIXD($1) } }')

define(`RCMQPALL_ARR',`{ $2, { ALL_REAL_NO_BITMAP($1), ALL_COMPLEX($1), ALL_MIXED($1), ALL_QUAT($1), ALL_QMIXD($1) } }')

define(`RC_FIXED_ARR',`{ $2, { DUP_ALL($1,$3), DUP_ALL($1,nullobjf), ALL_NULL, ALL_NULL, ALL_NULL } }')

define(`INT_ARR',`{ $2, { REAL_INT_ALL($1), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }')

define(`REAL_INT_ARR',`{ $2, { INT_ALL($1), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }')

define(`REAL_INT_ARR_NO_BITMAP',`{ $2, { INT_ALL_NO_BITMAP($1), ALL_NULL, ALL_NULL, ALL_NULL, ALL_NULL } }')



