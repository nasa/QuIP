#ifndef _GEN_PORT_H_
#define _GEN_PORT_H_

#include "platform.h"

#define TMPVEC_NAME	_TMPVEC_NAME(pf_str)
#define _TMPVEC_NAME(pf)	__TMPVEC_NAME(pf)
#define __TMPVEC_NAME(pf)	pf##_tmp_vec

#define FREETMP_NAME	_FREETMP_NAME(pf_str)
#define _FREETMP_NAME(pf)	__FREETMP_NAME(pf)
#define __FREETMP_NAME(pf)	pf##_free_tmp

#define PF_COMMAND_FUNC(f)	COMMAND_FUNC( MENU_FUNC_NAME(f) )

#define PF_FUNC_NAME(f)		PLATFORM_SYMBOL_NAME(f)

#define PLATFORM_SYMBOL_NAME(n)	PF_SYM_NAME(pf_str,n)
#define PF_SYM_NAME(pf,n)	_PF_SYM_NAME(pf,n)
#define _PF_SYM_NAME(pf,n)	pf##_##n


#define MENU_FUNC_NAME(f)			_MENU_FUNC_NAME(pf_str,f)
#define _MENU_FUNC_NAME(pf,f)			__MENU_FUNC_NAME(pf,f)
#define __MENU_FUNC_NAME(pf,f)			do_##pf##_##f


// this is really a host call...
#define HOST_CALL_NAME(name)		_XXX_CALL_NAME(h,pf_str,name)
#define HOST_CALL_NAME_REAL(name)	_XXX_CALL_NAME_REAL(h,pf_str,name)
#define HOST_CALL_NAME_CPX(name)	_XXX_CALL_NAME_CPX(h,pf_str,name)
#define HOST_CALL_NAME_QUAT(name)	_XXX_CALL_NAME_QUAT(h,pf_str,name)
#define HOST_CALL_NAME_MIXED(name)	_XXX_CALL_NAME_MIXED(h,pf_str,name)
#define HOST_CALL_NAME_QMIXD(name)	_XXX_CALL_NAME_QMIXD(h,pf_str,name)

#define CPU_CALL_NAME(name)		_XXX_CALL_NAME(c,pf_str,name)
#define CPU_CALL_NAME_REAL(name)	_XXX_CALL_NAME_REAL(c,pf_str,name)
#define CPU_CALL_NAME_CPX(name)		_XXX_CALL_NAME_CPX(c,pf_str,name)
#define CPU_CALL_NAME_QUAT(name)	_XXX_CALL_NAME_QUAT(c,pf_str,name)
#define CPU_CALL_NAME_MIXED(name)	_XXX_CALL_NAME_MIXED(c,pf_str,name)
#define CPU_CALL_NAME_QMIXD(name)	_XXX_CALL_NAME_QMIXD(c,pf_str,name)

#define GPU_CALL_NAME(name)		GPU_TYPED_CALL_NAME(name,type_code)

#define CPU_TYPED_CALL_NAME(name,tc)	_XXX_TYPED_CALL_NAME(c,pf_str,name,tc)
#define GPU_TYPED_CALL_NAME(name,tc)	_XXX_TYPED_CALL_NAME(g,pf_str,name,tc)

#define PF_TYPED_CALL_NAME(name)	_PF_TYPED_CALL_NAME(pf_str,type_code,name)
#define _PF_TYPED_CALL_NAME(pf,tc,n)	__PF_TYPED_CALL_NAME(pf,tc,n)
#define __PF_TYPED_CALL_NAME(pf,tc,n)	h_##pf##_##tc##_##n

#define PF_FFT_CALL_NAME(name)		_PF_FFT_CALL_NAME(pf_str,type_code,name)
#define _PF_FFT_CALL_NAME(pf,tc,n)	__PF_FFT_CALL_NAME(pf,tc,n)
#define __PF_FFT_CALL_NAME(pf,tc,n)	pf##_fft_##tc##_##n

#define PF_TYPED_CALL_NAME_CPX(name,tc)		_PF_TYPED_CALL_NAME_CPX(pf_str,name,tc)
#define PF_TYPED_CALL_NAME_REAL(name,tc)	_PF_TYPED_CALL_NAME_REAL(pf_str,name,tc)

#define HOST_TYPED_CALL_NAME(name,tc)		_XXX_TYPED_CALL_NAME(h,pf_str,name,tc)
#define HOST_TYPED_CALL_NAME_REAL(name,tc)	_HOST_TYPED_CALL_NAME_REAL(pf_str,name,tc)
#define HOST_TYPED_CALL_NAME_CPX(name,tc)	_HOST_TYPED_CALL_NAME_CPX(pf_str,name,tc)
#define HOST_TYPED_CALL_NAME_MIXED(name,tc)	_HOST_TYPED_CALL_NAME_MIXED(pf_str,name,tc)
#define HOST_TYPED_CALL_NAME_QUAT(name,tc)	_HOST_TYPED_CALL_NAME_QUAT(pf_str,name,tc)
#define HOST_TYPED_CALL_NAME_QMIXD(name,tc)	_HOST_TYPED_CALL_NAME_QMIXD(pf_str,name,tc)

#define INDEX_SETUP_NAME(name)	_INDEX_SETUP_NAME(pf_str,name,type_code)
#define INDEX_HELPER_NAME(name)	_INDEX_HELPER_NAME(pf_str,name,type_code)

#define MM_HELPER_NAME(name)	_MM_HELPER_NAME(pf_str,name,type_code)

#define NOCC_SETUP_NAME(name)	_NOCC_SETUP_NAME(pf_str,name,type_code)
#define NOCC_HELPER_NAME(name)	_NOCC_HELPER_NAME(pf_str,name,type_code)

#define CONV_FUNC_NAME(prec_from,prec_to)	_CONV_FUNC_NAME(pf_str,prec_from,prec_to)

// these are the kernel names
#define GPU_FAST_CALL_NAME(name)	_XXX_FAST_CALL_NAME(g,pf_str,type_code,name)
#define GPU_EQSP_CALL_NAME(name)	_XXX_EQSP_CALL_NAME(g,pf_str,type_code,name)
#define GPU_SLOW_CALL_NAME(name)	_XXX_SLOW_CALL_NAME(g,pf_str,type_code,name)
#define GPU_FLEN_CALL_NAME(name)	_XXX_FLEN_CALL_NAME(g,pf_str,type_code,name)
#define GPU_ELEN_CALL_NAME(name)	_XXX_ELEN_CALL_NAME(g,pf_str,type_code,name)
#define GPU_SLEN_CALL_NAME(name)	_XXX_SLEN_CALL_NAME(g,pf_str,type_code,name)

#define HOST_FAST_CALL_NAME(stem)	_XXX_FAST_CALL_NAME(h,pf_str,type_code,stem)
#define HOST_EQSP_CALL_NAME(stem)	_XXX_EQSP_CALL_NAME(h,pf_str,type_code,stem)
#define HOST_SLOW_CALL_NAME(stem)	_XXX_SLOW_CALL_NAME(h,pf_str,type_code,stem)
#define HOST_FLEN_CALL_NAME(stem)	_XXX_FLEN_CALL_NAME(h,pf_str,type_code,stem)
#define HOST_ELEN_CALL_NAME(stem)	_XXX_ELEN_CALL_NAME(h,pf_str,type_code,stem)
#define HOST_SLEN_CALL_NAME(stem)	_XXX_SLEN_CALL_NAME(h,pf_str,type_code,stem)

#define CPU_FAST_CALL_NAME(stem)	_XXX_FAST_CALL_NAME(c,pf_str,type_code,stem)
#define CPU_EQSP_CALL_NAME(stem)	_XXX_EQSP_CALL_NAME(c,pf_str,type_code,stem)
#define CPU_SLOW_CALL_NAME(stem)	_XXX_SLOW_CALL_NAME(c,pf_str,type_code,stem)
#define CPU_FLEN_CALL_NAME(stem)	_XXX_FLEN_CALL_NAME(c,pf_str,type_code,stem)
#define CPU_ELEN_CALL_NAME(stem)	_XXX_ELEN_CALL_NAME(c,pf_str,type_code,stem)
#define CPU_SLEN_CALL_NAME(stem)	_XXX_SLEN_CALL_NAME(c,pf_str,type_code,stem)


#define _XXX_CALL_NAME(pre,pf,name)		__XXX_CALL_NAME(pre,pf,name)
#define _XXX_CALL_NAME_REAL(pre,pf,name)	__XXX_CALL_NAME_REAL(pre,pf,name)
#define _XXX_CALL_NAME_CPX(pre,pf,name)		__XXX_CALL_NAME_CPX(pre,pf,name)
#define _XXX_CALL_NAME_QUAT(pre,pf,name)	__XXX_CALL_NAME_QUAT(pre,pf,name)
#define _XXX_CALL_NAME_MIXED(pre,pf,name)	__XXX_CALL_NAME_MIXED(pre,pf,name)
#define _XXX_CALL_NAME_QMIXD(pre,pf,name)	__XXX_CALL_NAME_QMIXD(pre,pf,name)

#define _XXX_TYPED_CALL_NAME(pre,pf,n,typ)	__XXX_TYPED_CALL_NAME(pre,pf,n,typ)
#define _PF_TYPED_CALL_NAME_CPX(pf,n,typ)	__PF_TYPED_CALL_NAME_CPX(pf,n,typ)
#define _PF_TYPED_CALL_NAME_REAL(pf,n,typ)	__PF_TYPED_CALL_NAME_REAL(pf,n,typ)

#define _HOST_TYPED_CALL_NAME_REAL(pf,n,typ)	__HOST_TYPED_CALL_NAME_REAL(pf,n,typ)
#define _HOST_TYPED_CALL_NAME_CPX(pf,n,typ)	__HOST_TYPED_CALL_NAME_CPX(pf,n,typ)
#define _HOST_TYPED_CALL_NAME_MIXED(pf,n,typ)	__HOST_TYPED_CALL_NAME_MIXED(pf,n,typ)
#define _HOST_TYPED_CALL_NAME_QUAT(pf,n,typ)	__HOST_TYPED_CALL_NAME_QUAT(pf,n,typ)
#define _HOST_TYPED_CALL_NAME_QMIXD(pf,n,typ)	__HOST_TYPED_CALL_NAME_QMIXD(pf,n,typ)

#define _INDEX_SETUP_NAME(pf,name,typ)	__INDEX_SETUP_NAME(pf,name,typ)
#define _INDEX_HELPER_NAME(pf,name,typ)	__INDEX_HELPER_NAME(pf,name,typ)

#define _MM_HELPER_NAME(pf,name,typ)	__MM_HELPER_NAME(pf,name,typ)

#define _NOCC_SETUP_NAME(pf,name,typ)	__NOCC_SETUP_NAME(pf,name,typ)
#define _NOCC_HELPER_NAME(pf,name,typ)	__NOCC_HELPER_NAME(pf,name,typ)

#define _CONV_FUNC_NAME(pf,prec_from,prec_to)	__CONV_FUNC_NAME(pf,prec_from,prec_to)

#define _XXX_FAST_CALL_NAME(pre,pf,ty,stem)	__XXX_FAST_CALL_NAME(pre,pf,ty,stem)
#define _XXX_EQSP_CALL_NAME(pre,pf,ty,stem)	__XXX_EQSP_CALL_NAME(pre,pf,ty,stem)
#define _XXX_SLOW_CALL_NAME(pre,pf,ty,stem)	__XXX_SLOW_CALL_NAME(pre,pf,ty,stem)
#define _XXX_FLEN_CALL_NAME(pre,pf,ty,stem)	__XXX_FLEN_CALL_NAME(pre,pf,ty,stem)
#define _XXX_ELEN_CALL_NAME(pre,pf,ty,stem)	__XXX_ELEN_CALL_NAME(pre,pf,ty,stem)
#define _XXX_SLEN_CALL_NAME(pre,pf,ty,stem)	__XXX_SLEN_CALL_NAME(pre,pf,ty,stem)

#define __XXX_CALL_NAME(prefix,pf,name)			prefix##_##pf##_##name
#define __XXX_CALL_NAME_REAL(prefix,pf,name)		prefix##_##pf##_r##name
#define __XXX_CALL_NAME_CPX(prefix,pf,name)		prefix##_##pf##_c##name
#define __XXX_CALL_NAME_QUAT(prefix,pf,name)		prefix##_##pf##_q##name
#define __XXX_CALL_NAME_MIXED(prefix,pf,name)		prefix##_##pf##_m##name
#define __XXX_CALL_NAME_QMIXD(prefix,pf,name)		prefix##_##pf##_p##name

#define __XXX_TYPED_CALL_NAME(pre,pf,name,typ)		pre##_##pf##_##typ##_##name
#define __PF_TYPED_CALL_NAME_CPX(pf,name,typ)		pf##_##typ##_c##name
#define __PF_TYPED_CALL_NAME_REAL(pf,name,typ)		pf##_##typ##_r##name

#define __HOST_TYPED_CALL_NAME_REAL(pf,name,typ)	h_##pf##_##typ##_r##name
#define __HOST_TYPED_CALL_NAME_CPX(pf,name,typ)	h_##pf##_##typ##_c##name
#define __HOST_TYPED_CALL_NAME_MIXED(pf,name,typ)	h_##pf##_##typ##_m##name
#define __HOST_TYPED_CALL_NAME_QUAT(pf,name,typ)	h_##pf##_##typ##_q##name
#define __HOST_TYPED_CALL_NAME_QMIXD(pf,name,typ)	h_##pf##_##typ##_p##name

#define __INDEX_SETUP_NAME(pf,name,typ)		h_##pf##_##name##_##typ##_setup
#define __INDEX_HELPER_NAME(pf,name,typ)		h_##pf##_##name##_##typ##_helper

#define __MM_HELPER_NAME(pf,name,typ)		h_##pf##_##name##_##typ##_helper

#define __NOCC_SETUP_NAME(pf,name,typ)		h_##pf##_##name##_##typ##_setup
#define __NOCC_HELPER_NAME(pf,name,typ)		h_##pf##_##name##_##typ##_helper

#define __CONV_FUNC_NAME(pf,prec_from,prec_to)	h_##pf##_v##prec_from##2##prec_to

#define __XXX_FAST_CALL_NAME(pre,pf,ty,stem)	pre##_##pf##_fast_##ty##_##stem
#define __XXX_EQSP_CALL_NAME(pre,pf,ty,stem)	pre##_##pf##_eqsp_##ty##_##stem
#define __XXX_SLOW_CALL_NAME(pre,pf,ty,stem)	pre##_##pf##_slow_##ty##_##stem
#define __XXX_FLEN_CALL_NAME(pre,pf,ty,stem)	pre##_##pf##_flen_##ty##_##stem
#define __XXX_ELEN_CALL_NAME(pre,pf,ty,stem)	pre##_##pf##_elen_##ty##_##stem
#define __XXX_SLEN_CALL_NAME(pre,pf,ty,stem)	pre##_##pf##_slen_##ty##_##stem

#endif // ! _GEN_PORT_H_
