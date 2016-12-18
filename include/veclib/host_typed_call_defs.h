#ifndef _HOST_TYPED_CALL_DEFS_H_
#define _HOST_TYPED_CALL_DEFS_H_

//#define MORE_DEBUG	// uncomment this for extra debugging

#define HOST_NAME_STRING	STRINGIFY(HOST_TYPED_CALL_NAME(name,type_code))

#define HOST_CALL_VAR_DECLS						\
									\
	Vector_Args va1;						\
	Vector_Args *vap=(&va1);					\

// These definitions mostly turn into a GENERIC_HOST_TYPED_CALL...
//
// Somewhere we have a declaration like VFUNC_PROT_2V...

//#include "kern_both_defs.h"
#include "veclib/fast_test.h"
#include "veclib/xfer_args.h"

// args n, c, s  are func_name, type_code, statement

// CONV replaces 2
#define _VEC_FUNC_2V_CONV(n,type,s)	GENERIC_HOST_TYPED_CONV(n,,,type)


// 5 args
#define _VEC_FUNC_5V(n,s)	GENERIC_HOST_TYPED_CALL(n,,,,5)
#define _VEC_FUNC_4V_SCAL(n,s)	GENERIC_HOST_TYPED_CALL(n,,,1S_,4)
#define _VEC_FUNC_3V_2SCAL(n,s)	GENERIC_HOST_TYPED_CALL(n,,,2S_,3)
#define _VEC_FUNC_2V_3SCAL(n,s)	GENERIC_HOST_TYPED_CALL(n,,,3S_,2)

// this is vramp2d
#define _VEC_FUNC_1V_3SCAL(n,cpu_stat1,cpu_stat2,gpu_stat)	SLOW_HOST_CALL(n,,,3S_,1)

// 3 args
#define _VEC_FUNC_3V(n,s)		GENERIC_HOST_TYPED_CALL(n,,,,3)
#define _VEC_FUNC_CPX_3V(n,s)		GENERIC_HOST_TYPED_CALL(n,,CPX_,,3)
#define _VEC_FUNC_QUAT_3V(n,s)		GENERIC_HOST_TYPED_CALL(n,,QUAT_,,3)
#define _VEC_FUNC_1V_2SCAL(n,cpu_s,gpu_s)	GENERIC_HOST_TYPED_CALL(n,,,2S_,1)
#define _VEC_FUNC_2V_SCAL(n,s)		GENERIC_HOST_TYPED_CALL(n,,,1S_,2)

#define _VEC_FUNC_VVSLCT(n,s)		GENERIC_HOST_TYPED_CALL(n,SBM_,,,3)
#define _VEC_FUNC_VSSLCT(n,s)		GENERIC_HOST_TYPED_CALL(n,SBM_,,1S_,2)
#define _VEC_FUNC_SSSLCT(n,s)		GENERIC_HOST_TYPED_CALL(n,SBM_,,2S_,1)

#define _VEC_FUNC_SBM_1(n,s)		GENERIC_HOST_TYPED_CALL(n,SBM_,,,1) 

#define _VEC_FUNC_1V(n,s)		GENERIC_HOST_TYPED_CALL(n,,,,1)
#define _VEC_FUNC_2V(n,s)		GENERIC_HOST_TYPED_CALL(n,,,,2)
#define _VEC_FUNC_2V_MIXED(n,s)		GENERIC_HOST_TYPED_CALL(n,,RC_,,2)
#define _VEC_FUNC_CPX_2V(n,s)		GENERIC_HOST_TYPED_CALL(n,,CPX_,,2) 
#define _VEC_FUNC_QUAT_2V(n,s)		GENERIC_HOST_TYPED_CALL(n,,QUAT_,,2) 

#define _VEC_FUNC_VVMAP(n,s)		GENERIC_HOST_TYPED_CALL(n,DBM_,,,2SRCS)
// vsm_gt etc
#define _VEC_FUNC_VSMAP(n,s)		GENERIC_HOST_TYPED_CALL(n,DBM_,,1S_,1SRC)

// this is vset
#define _VEC_FUNC_1V_SCAL(n,s)		GENERIC_HOST_TYPED_CALL(n,,,1S_,1)
// where is cpx vset??

#define _VEC_FUNC_CPX_1V_SCAL(n,s)	GENERIC_HOST_TYPED_CALL(n,,CPX_,1S_,1)
// this is bit_vset
// What is up with 1S vs 1S_ ???
// bit_vset
#define _VEC_FUNC_DBM_1S_(n,s)		GENERIC_HOST_TYPED_CALL(n,DBM_,,1S_,)
#define _VEC_FUNC_DBM_1S(n,s)		GENERIC_HOST_TYPED_CALL(n,DBM_,,1S_,)

#define _VEC_FUNC_DBM_1V(n,s)		GENERIC_HOST_TYPED_CALL(n,DBM_,,,1)
// bit_vmov
#define _VEC_FUNC_DBM_SBM(n,s)		GENERIC_HOST_TYPED_CALL(n,DBM_SBM,,,)

#define _VEC_FUNC_SBM_CPX_3V(n,s)	GENERIC_HOST_TYPED_CALL(n,SBM_,CPX_,,3) 
#define _VEC_FUNC_SBM_CPX_1S_2V(n,s)	GENERIC_HOST_TYPED_CALL(n,SBM_,CPX_,1S_,2) 
#define _VEC_FUNC_SBM_CPX_2S_1V(n,s)	GENERIC_HOST_TYPED_CALL(n,SBM_,CPX_,2S_,1) 
#define _VEC_FUNC_SBM_QUAT_3V(n,s)	GENERIC_HOST_TYPED_CALL(n,SBM_,QUAT_,,3) 
#define _VEC_FUNC_SBM_QUAT_1S_2V(n,s)	GENERIC_HOST_TYPED_CALL(n,SBM_,QUAT_,1S_,2) 
#define _VEC_FUNC_SBM_QUAT_2S_1V(n,s)	GENERIC_HOST_TYPED_CALL(n,SBM_,QUAT_,2S_,1) 
#define _VEC_FUNC_CPX_2V_T2(n,s)	GENERIC_HOST_TYPED_CALL(n,,CPX_,,2) 
#define _VEC_FUNC_CPXT_2V(n,s)		GENERIC_HOST_TYPED_CALL(n,,CPX_,,2) 
#define _VEC_FUNC_CPXT_3V(n,s)		GENERIC_HOST_TYPED_CALL(n,,CPX_,,3) 
#define _VEC_FUNC_CPXD_3V(n,s)		GENERIC_HOST_TYPED_CALL(n,,CPX_,,3) 
#define _VEC_FUNC_CPX_1S_2V(n,s)	GENERIC_HOST_TYPED_CALL(n,,CPX_,1S_,2) 
#define _VEC_FUNC_QUAT_1S_2V(n,s)	GENERIC_HOST_TYPED_CALL(n,,QUAT_,1S_,2) 
#define _VEC_FUNC_CPX_1S_2V_T2(n,s)	GENERIC_HOST_TYPED_CALL(n,,CPX_,1S_,2) 
#define _VEC_FUNC_CPX_1S_2V_T3(n,s)	GENERIC_HOST_TYPED_CALL(n,,CPX_,1S_,2) 
#define _VEC_FUNC_QUAT_1S_2V_T4(n,s)	GENERIC_HOST_TYPED_CALL(n,,QUAT_,1S_,2) 
#define _VEC_FUNC_CPXT_1S_2V(n,s)	GENERIC_HOST_TYPED_CALL(n,,CPX_,1S_,2) 
#define _VEC_FUNC_CPXD_1S_2V(n,s)	GENERIC_HOST_TYPED_CALL(n,,CPX_,1S_,2) 
#define _VEC_FUNC_CPX_1S_1V(n,s)	GENERIC_HOST_TYPED_CALL(n,,CPX_,1S_,1) 
#define _VEC_FUNC_QUAT_1S_1V(n,s)	GENERIC_HOST_TYPED_CALL(n,,QUAT_,1S_,1) 
#define _VEC_FUNC_CPX_3V_T1(n,s)	GENERIC_HOST_TYPED_CALL(n,,CPX_,,3) 
#define _VEC_FUNC_CPX_3V_T2(n,s)	GENERIC_HOST_TYPED_CALL(n,,CPX_,,3) 
#define _VEC_FUNC_CPX_3V_T3(n,s)	GENERIC_HOST_TYPED_CALL(n,,CPX_,,3) 
#define _VEC_FUNC_QUAT_2V_T4(n,s)	GENERIC_HOST_TYPED_CALL(n,,QUAT_,,2) 
#define _VEC_FUNC_QUAT_3V_T4(n,s)	GENERIC_HOST_TYPED_CALL(n,,QUAT_,,3) 
#define _VEC_FUNC_QUAT_3V_T5(n,s)	GENERIC_HOST_TYPED_CALL(n,,QUAT_,,3) 
#define _VEC_FUNC_QUAT_1S_2V_T5(n,s)	GENERIC_HOST_TYPED_CALL(n,,QUAT_,1S_,2) 
#define _VEC_FUNC_CCR_3V(n,s)		GENERIC_HOST_TYPED_CALL(n,,CCR_,,3) 
#define _VEC_FUNC_QQR_3V(n,s)		GENERIC_HOST_TYPED_CALL(n,,QQR_,,3) 
#define _VEC_FUNC_CR_1S_2V(n,s)		GENERIC_HOST_TYPED_CALL(n,,CR_,1S_,2) 
#define _VEC_FUNC_QR_1S_2V(n,s)		GENERIC_HOST_TYPED_CALL(n,,QR_,1S_,2) 
// args d,s1,s2 are dst_arg, src_arg1, src_arg2
#define _VEC_FUNC_VS_LS(n,d,s1,s2)	GENERIC_HOST_TYPED_CALL(n,,,1S_,2)
#define _VEC_FUNC_VV_LS(n,d,s1,s2)	GENERIC_HOST_TYPED_CALL(n,,,,3)

// GENERIC_HOST_TYPED_CALL declares four functions:
// First, fast, equally-spaced, and slow versions of the typed call
// Then a typed call which performs the fast test prior to calling
// one of the previously defined functions.
// The typed ones can be static?

// These are special cases that need to be coded by hand for gpu...

#include "veclib/host_calls_special.h"

#ifdef BUILD_FOR_GPU

// BUG - need to merge these files!
//#ifdef BUILD_FOR_OPENCL
//#include "veclib/host_calls_ocl.h"
//#else
#include "veclib/host_calls_gpu.h"
//#endif // ! BUILD_FOR_OPENCL

#else // ! BUILD_FOR_GPU

#include "veclib/host_calls_cpu.h"

#endif // ! BUILD_FOR_GPU


/* The fast/slow functions are declared along with the generic typed call,
 * but they really platform-specific, as they call the kernel, so the
 * definitions are elsewhere.
 */

#define GENERIC_HOST_TYPED_CONV(name,bitmap,typ,type)	\
							\
GENERIC_HOST_FAST_CONV(name,bitmap,typ,type)		\
GENERIC_HOST_EQSP_CONV(name,bitmap,typ,type)		\
GENERIC_HOST_SLOW_CONV(name,bitmap,typ,type)		\
							\
GENERIC_HOST_FAST_SWITCH(name,bitmap,typ,,2)

#define GENERIC_HOST_TYPED_CALL(name,bitmap,typ,scalars,vectors)	\
									\
GENERIC_HOST_FAST_CALL(name,bitmap,typ,scalars,vectors)			\
GENERIC_HOST_EQSP_CALL(name,bitmap,typ,scalars,vectors)			\
GENERIC_HOST_SLOW_CALL(name,bitmap,typ,scalars,vectors)			\
									\
GENERIC_HOST_FAST_SWITCH(name,bitmap,typ,scalars,vectors)



#define CHAIN_CHECK( func )					\
								\
	if( is_chaining ){					\
		if( insure_static(oap) < 0 ) return;		\
		add_link( & func , LINK_FUNC_ARGS );		\
		return;						\
	} else {						\
		func(LINK_FUNC_ARGS);				\
		SET_ASSIGNED_FLAG( OA_DEST(oap) )		\
	}



// This is really only necessary for debugging, or if we want to print
// out the arguments, without knowing which are supposed to be set...
#define CLEAR_VEC_ARGS(vap)						\
	bzero(vap,sizeof(*vap));

#define SLOW_HOST_CALL(name,bitmap,typ,scalars,vectors)			\
									\
GENERIC_HOST_SLOW_CALL(name,bitmap,typ,scalars,vectors)			\
									\
static void HOST_TYPED_CALL_NAME(name,type_code)(HOST_CALL_ARG_DECLS)	\
{									\
	HOST_CALL_VAR_DECLS						\
									\
	CLEAR_VEC_ARGS(vap)						\
	SET_VA_PFDEV(vap,OA_PFDEV(oap));				\
	/* where do the increments get set? */				\
	XFER_SLOW_ARGS_##bitmap##typ##scalars##vectors			\
	/* setup_slow_len must go here! */				\
	SETUP_SLOW_LEN_##bitmap##vectors				\
	CHAIN_CHECK( HOST_SLOW_CALL_NAME(name) )			\
}

// BUG!! these need to be filled in...
#define MISSING_CALL(func)	_MISSING_CALL(func)
#define _MISSING_CALL(func)	fprintf(stderr,"Missing code body for function %s!?\n",#func);

#ifdef MORE_DEBUG

#define REPORT_SWITCH(name,sw)		_REPORT_SWITCH(name,sw)

#define _REPORT_SWITCH(name,sw)						\
sprintf(DEFAULT_ERROR_STRING,"Calling %s version of %s",#sw,#name);	\
NADVISE(DEFAULT_ERROR_STRING);

#else /* ! MORE_DEBUG */
#define REPORT_SWITCH(name,sw)
#endif /* ! MORE_DEBUG */

#define GENERIC_HOST_FAST_SWITCH(name,bitmap,typ,scalars,vectors)		\
									\
static void HOST_TYPED_CALL_NAME(name,type_code)(HOST_CALL_ARG_DECLS)	\
{									\
	HOST_CALL_VAR_DECLS						\
									\
	CLEAR_VEC_ARGS(vap)						\
	SET_VA_PFDEV(vap,OA_PFDEV(oap));				\
	if( FAST_TEST_##bitmap##typ##vectors ){				\
REPORT_SWITCH(STRINGIFY(HOST_TYPED_CALL_NAME(name,type_code)),fast)	\
		XFER_FAST_ARGS_##bitmap##typ##scalars##vectors		\
		CHAIN_CHECK( HOST_FAST_CALL_NAME(name) )		\
	} else if( EQSP_TEST_##bitmap##typ##vectors ){			\
REPORT_SWITCH(name,eqsp)						\
		XFER_EQSP_ARGS_##bitmap##typ##scalars##vectors		\
/*sprintf(DEFAULT_ERROR_STRING,"showing vec args for eqsp %s",#name);\
NADVISE(DEFAULT_ERROR_STRING);*/\
/*show_vec_args(vap);*/\
		CHAIN_CHECK( HOST_EQSP_CALL_NAME(name) )		\
	} else {							\
REPORT_SWITCH(name,slow)						\
		XFER_SLOW_ARGS_##bitmap##typ##scalars##vectors		\
		/* setup_slow_len must go here! */			\
		SETUP_SLOW_LEN_##bitmap##vectors			\
		CHAIN_CHECK( HOST_SLOW_CALL_NAME(name) )		\
	}								\
}

#endif // ! _HOST_TYPED_CALL_DEFS_H_
