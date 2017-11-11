dnl	 This file is kind of the opposite of the type_tbl approach...
dnl	 here we have a bunch of switches and if/then's...

dnl	This file defines the untyped functions
dnl	BUT it doesn't seem to be included???

dnl	my_include(`veclib/gen_entries.m4')

ifdef(`BUILD_FOR_GPU',`

dnl	Need GPU definitions here!!!

',` dnl ifndef BUILD_FOR_GPU


dnl	REAL_FLOAT_SWITCH(func,prot_dp)
define(`REAL_FLOAT_SWITCH',`

switch( OBJ_MACH_PREC($2) ){
	case PREC_SP:  HOST_TYPED_CALL_NAME_REAL($1,sp)(HOST_CALL_ARGS); break;
	case PREC_DP:  HOST_TYPED_CALL_NAME_REAL($1,dp)(HOST_CALL_ARGS); break;
	default:
		sprintf(ERROR_STRING,
	"real_float_switch (%s):  missing precision case (obj %s, prec %s)",
		"$1", OBJ_NAME($2), OBJ_MACH_PREC_NAME($2) );
		warn(ERROR_STRING);
		break;
}
')

// FFT_SWITCH is used for complex fft functions, that have the inverse flag...
dnl	FFT_SWITCH( func, inverse_factor )
define(`FFT_SWITCH',`

switch( OBJ_MACH_PREC(OA_SRC1(oap)) ){
	case PREC_SP:
		HOST_TYPED_CALL_NAME_CPX($1,sp)(HOST_CALL_ARGS, $2);
		break;
	case PREC_DP:
		HOST_TYPED_CALL_NAME_CPX($1,dp)(HOST_CALL_ARGS, $2);
		break;
	default:	sprintf(ERROR_STRING,
"fft_switch (%s):  object %s has bad machine precision %s",
"$1",OBJ_NAME(OA_SRC1(oap)),OBJ_MACH_PREC_NAME(OA_SRC1(oap)) );
		warn(ERROR_STRING);
		break;
}
')

dnl	These are the un-typed calls...
dnl THESE NEXT TWO ARE NOT FFT FUNCS!?!?

void HOST_CALL_NAME(xform_list)(HOST_CALL_ARG_DECLS) { REAL_FLOAT_SWITCH( xform_list, OA_DEST(oap) ) }
void HOST_CALL_NAME(vec_xform)(HOST_CALL_ARG_DECLS) { REAL_FLOAT_SWITCH( vec_xform, OA_DEST(oap) ) }

') dnl endif // ! BUILD_FOR_GPU

