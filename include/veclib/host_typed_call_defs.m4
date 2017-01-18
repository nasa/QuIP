
define(`HOST_CALL_VAR_DECLS',`

	Vector_Args va1;
	Vector_Args *vap=(&va1);
')

// These definitions mostly turn into a GENERIC_HOST_TYPED_CALL...
//
// Somewhere we have a declaration like VFUNC_PROT_2V...

include(`../../include/veclib/fast_test.m4')
include(`../../include/veclib/xfer_args.m4')

// args n, c, s  are func_name, type_code, statement

// CONV replaces 2
define(`_VEC_FUNC_2V_CONV',	GENERIC_HOST_TYPED_CONV($1,,,$2))


// 5 args
define(`_VEC_FUNC_5V',	GENERIC_HOST_TYPED_CALL($1,,,,5))
define(`_VEC_FUNC_4V_SCAL',	GENERIC_HOST_TYPED_CALL($1,,,1S_,4))
define(`_VEC_FUNC_3V_2SCAL',	GENERIC_HOST_TYPED_CALL($1,,,2S_,3))
define(`_VEC_FUNC_2V_3SCAL',	GENERIC_HOST_TYPED_CALL($1,,,3S_,2))

// this is vramp2d
define(`_VEC_FUNC_1V_3SCAL',	SLOW_HOST_CALL($1,,,3S_,1))

// 3 args
define(`_VEC_FUNC_3V',		GENERIC_HOST_TYPED_CALL($1,,,,3))
define(`_VEC_FUNC_CPX_3V',		GENERIC_HOST_TYPED_CALL($1,,CPX_,,3))
define(`_VEC_FUNC_QUAT_3V',		GENERIC_HOST_TYPED_CALL($1,,QUAT_,,3))
define(`_VEC_FUNC_1V_2SCAL',	GENERIC_HOST_TYPED_CALL($1,,,2S_,1))
define(`_VEC_FUNC_2V_SCAL',		GENERIC_HOST_TYPED_CALL($1,,,1S_,2))

define(`_VEC_FUNC_VVSLCT',		GENERIC_HOST_TYPED_CALL($1,SBM_,,,3))
define(`_VEC_FUNC_VSSLCT',		GENERIC_HOST_TYPED_CALL($1,SBM_,,1S_,2))
define(`_VEC_FUNC_SSSLCT',		GENERIC_HOST_TYPED_CALL($1,SBM_,,2S_,1))

define(`_VEC_FUNC_SBM_1',		GENERIC_HOST_TYPED_CALL($1,SBM_,,,1) )

define(`_VEC_FUNC_1V',		GENERIC_HOST_TYPED_CALL($1,,,,1))
define(`_VEC_FUNC_2V',`GENERIC_HOST_TYPED_CALL($1,,,,2)')
define(`_VEC_FUNC_2V_MIXED',		GENERIC_HOST_TYPED_CALL($1,,RC_,,2))
define(`_VEC_FUNC_CPX_2V',		GENERIC_HOST_TYPED_CALL($1,,CPX_,,2)) 
define(`_VEC_FUNC_QUAT_2V',		GENERIC_HOST_TYPED_CALL($1,,QUAT_,,2)) 

define(`_VEC_FUNC_VVMAP',		GENERIC_HOST_TYPED_CALL($1,DBM_,,,2SRCS))
// vsm_gt etc
define(`_VEC_FUNC_VSMAP',		GENERIC_HOST_TYPED_CALL($1,DBM_,,1S_,1SRC))

// this is vset
define(`_VEC_FUNC_1V_SCAL',		GENERIC_HOST_TYPED_CALL($1,,,1S_,1))
// where is cpx vset??

define(`_VEC_FUNC_CPX_1V_SCAL',	GENERIC_HOST_TYPED_CALL($1,,CPX_,1S_,1))
// this is bit_vset
// What is up with 1S vs 1S_ ???
// bit_vset
define(`_VEC_FUNC_DBM_1S_',		GENERIC_HOST_TYPED_CALL($1,DBM_,,1S_,))
define(`_VEC_FUNC_DBM_1S',		GENERIC_HOST_TYPED_CALL($1,DBM_,,1S_,))

define(`_VEC_FUNC_DBM_1V',		GENERIC_HOST_TYPED_CALL($1,DBM_,,,1))
// bit_vmov
define(`_VEC_FUNC_DBM_SBM',		GENERIC_HOST_TYPED_CALL($1,DBM_SBM,,,))

define(`_VEC_FUNC_SBM_CPX_3V',	GENERIC_HOST_TYPED_CALL($1,SBM_,CPX_,,3) )
define(`_VEC_FUNC_SBM_CPX_1S_2V',	GENERIC_HOST_TYPED_CALL($1,SBM_,CPX_,1S_,2) )
define(`_VEC_FUNC_SBM_CPX_2S_1V',	GENERIC_HOST_TYPED_CALL($1,SBM_,CPX_,2S_,1) )
define(`_VEC_FUNC_SBM_QUAT_3V',	GENERIC_HOST_TYPED_CALL($1,SBM_,QUAT_,,3) )
define(`_VEC_FUNC_SBM_QUAT_1S_2V',	GENERIC_HOST_TYPED_CALL($1,SBM_,QUAT_,1S_,2) )
define(`_VEC_FUNC_SBM_QUAT_2S_1V',	GENERIC_HOST_TYPED_CALL($1,SBM_,QUAT_,2S_,1) )
define(`_VEC_FUNC_CPX_2V_T2',	GENERIC_HOST_TYPED_CALL($1,,CPX_,,2) )
define(`_VEC_FUNC_CPXT_2V',		GENERIC_HOST_TYPED_CALL($1,,CPX_,,2) )
define(`_VEC_FUNC_CPXT_3V',		GENERIC_HOST_TYPED_CALL($1,,CPX_,,3) )
define(`_VEC_FUNC_CPXD_3V',		GENERIC_HOST_TYPED_CALL($1,,CPX_,,3) )
define(`_VEC_FUNC_CPX_1S_2V',	GENERIC_HOST_TYPED_CALL($1,,CPX_,1S_,2) )
define(`_VEC_FUNC_QUAT_1S_2V',	GENERIC_HOST_TYPED_CALL($1,,QUAT_,1S_,2) )
define(`_VEC_FUNC_CPX_1S_2V_T2',	GENERIC_HOST_TYPED_CALL($1,,CPX_,1S_,2) )
define(`_VEC_FUNC_CPX_1S_2V_T3',	GENERIC_HOST_TYPED_CALL($1,,CPX_,1S_,2) )
define(`_VEC_FUNC_QUAT_1S_2V_T4',	GENERIC_HOST_TYPED_CALL($1,,QUAT_,1S_,2) )
define(`_VEC_FUNC_CPXT_1S_2V',	GENERIC_HOST_TYPED_CALL($1,,CPX_,1S_,2) )
define(`_VEC_FUNC_CPXD_1S_2V',	GENERIC_HOST_TYPED_CALL($1,,CPX_,1S_,2) )
define(`_VEC_FUNC_CPX_1S_1V',	GENERIC_HOST_TYPED_CALL($1,,CPX_,1S_,1) )
define(`_VEC_FUNC_QUAT_1S_1V',	GENERIC_HOST_TYPED_CALL($1,,QUAT_,1S_,1) )
define(`_VEC_FUNC_CPX_3V_T1',	GENERIC_HOST_TYPED_CALL($1,,CPX_,,3) )
define(`_VEC_FUNC_CPX_3V_T2',	GENERIC_HOST_TYPED_CALL($1,,CPX_,,3) )
define(`_VEC_FUNC_CPX_3V_T3',	GENERIC_HOST_TYPED_CALL($1,,CPX_,,3) )
define(`_VEC_FUNC_QUAT_2V_T4',	GENERIC_HOST_TYPED_CALL($1,,QUAT_,,2) )
define(`_VEC_FUNC_QUAT_3V_T4',	GENERIC_HOST_TYPED_CALL($1,,QUAT_,,3) )
define(`_VEC_FUNC_QUAT_3V_T5',	GENERIC_HOST_TYPED_CALL($1,,QUAT_,,3) )
define(`_VEC_FUNC_QUAT_1S_2V_T5',	GENERIC_HOST_TYPED_CALL($1,,QUAT_,1S_,2) )
define(`_VEC_FUNC_CCR_3V',		GENERIC_HOST_TYPED_CALL($1,,CCR_,,3) )
define(`_VEC_FUNC_QQR_3V',		GENERIC_HOST_TYPED_CALL($1,,QQR_,,3) )
define(`_VEC_FUNC_CR_1S_2V',		GENERIC_HOST_TYPED_CALL($1,,CR_,1S_,2) )
define(`_VEC_FUNC_QR_1S_2V',		GENERIC_HOST_TYPED_CALL($1,,QR_,1S_,2) )
// args d,s1,s2 are dst_arg, src_arg1, src_arg2
define(`_VEC_FUNC_VS_LS',	GENERIC_HOST_TYPED_CALL($1,,,1S_,2))
define(`_VEC_FUNC_VV_LS',	GENERIC_HOST_TYPED_CALL($1,,,,3))

// GENERIC_HOST_TYPED_CALL declares four functions:
// First, fast, equally-spaced, and slow versions of the typed call
// Then a typed call which performs the fast test prior to calling
// one of the previously defined functions.
// The typed ones can be static?

// These are special cases that need to be coded by hand for gpu...

include(`../../include/veclib/host_calls_special.m4')

ifdef(`BUILD_FOR_GPU',`

include(`../../include/veclib/host_calls_gpu.m4')

',`

include(`../../include/veclib/host_calls_cpu.m4')

')

/* The fast/slow functions are declared along with the generic typed call,
 * but they really platform-specific, as they call the kernel, so the
 * definitions are elsewhere.
 */

define(`GENERIC_HOST_TYPED_CONV',`

GENERIC_HOST_FAST_CONV($1,$2,$3,$4)
GENERIC_HOST_EQSP_CONV($1,$2,$3,$4)
GENERIC_HOST_SLOW_CONV($1,$2,$3,$4)

GENERIC_HOST_FAST_SWITCH($1,$2,$3,,2)
')

define(`GENERIC_HOST_TYPED_CALL',`

GENERIC_HOST_FAST_CALL($1,$2,$3,$4,$5)
GENERIC_HOST_EQSP_CALL($1,$2,$3,$4,$5)
GENERIC_HOST_SLOW_CALL($1,$2,$3,$4,$5)

GENERIC_HOST_FAST_SWITCH($1,$2,$3,$4,$5)
')



define(`CHAIN_CHECK',`

	if( is_chaining ){
		if( insure_static(oap) < 0 ) return;
		add_link( & $1, LINK_FUNC_ARGS );
		return;
	} else {
		$1(LINK_FUNC_ARGS);
		SET_ASSIGNED_FLAG( OA_DEST(oap) )
	}
')



// This is really only necessary for debugging, or if we want to print
// out the arguments, without knowing which are supposed to be set...
define(`CLEAR_VEC_ARGS', bzero($1,sizeof(*$1));)

dnl SLOW_HOST_CALL(name,bitmaps,typ,scalars,vectors)

define(`SLOW_HOST_CALL',`

GENERIC_HOST_SLOW_CALL($1,$2,$3,$4,$5)

static void HOST_TYPED_CALL_NAME($1,type_code)(HOST_CALL_ARG_DECLS)
{
	HOST_CALL_VAR_DECLS

	CLEAR_VEC_ARGS(vap)
	SET_VA_PFDEV(vap,OA_PFDEV(oap));
	/* where do the increments get set? */
	dnl `XFER_SLOW_ARGS_'$2$3$4$5
	XFER_SLOW_ARGS($2,$3,$4,$5)
	/* setup_slow_len must go here! */
	dnl `SETUP_SLOW_LEN_'$2$5
	SETUP_SLOW_LEN($2,$3,$5)
	CHAIN_CHECK( HOST_SLOW_CALL_NAME($1) )
}
')


// BUG!! these need to be filled in...
define(`MISSING_CALL',`fprintf(stderr,"Missing code body for function %s!?\n","$1");')

dnl define(`MORE_DEBUG',`')		dnl this line will enable the debugging version...

ifdef(`MORE_DEBUG',`

define(`REPORT_SWITCH',`sprintf(DEFAULT_ERROR_STRING,"Calling %s version of %s","$2","$1"); NADVISE(DEFAULT_ERROR_STRING);')
',`
define(`REPORT_SWITCH',`')
')

define(`GENERIC_HOST_FAST_SWITCH',`

static void HOST_TYPED_CALL_NAME($1,type_code)(HOST_CALL_ARG_DECLS)
{
	HOST_CALL_VAR_DECLS

	CLEAR_VEC_ARGS(vap)
	SET_VA_PFDEV(vap,OA_PFDEV(oap));
	if( FAST_TEST($2,$3,$5) ){
REPORT_SWITCH(STRINGIFY(HOST_TYPED_CALL_NAME($1,type_code)),fast)
		dnl `XFER_FAST_ARGS_'$2$3$4$5
		XFER_FAST_ARGS($2,$3,$4,$5)
		CHAIN_CHECK( HOST_FAST_CALL_NAME($1) )
	} else if( EQSP_TEST($2,$3,$5) ){
REPORT_SWITCH($1,eqsp)
		dnl `XFER_EQSP_ARGS_'$2$3$4$5
		XFER_EQSP_ARGS($2,$3,$4,$5)
/*sprintf(DEFAULT_ERROR_STRING,"showing vec args for eqsp %s","$1");
NADVISE(DEFAULT_ERROR_STRING);*/
/*show_vec_args(vap);*/
		CHAIN_CHECK( HOST_EQSP_CALL_NAME($1) )
	} else {
REPORT_SWITCH($1,slow)
		dnl `XFER_SLOW_ARGS_'$2$3$4$5
		XFER_SLOW_ARGS($2,$3,$4,$5)
		/* setup_slow_len must go here! */
		dnl `SETUP_SLOW_LEN_'$2$5
		SETUP_SLOW_LEN($2,$3,$5)
		CHAIN_CHECK( HOST_SLOW_CALL_NAME($1) )
	}
}
')

