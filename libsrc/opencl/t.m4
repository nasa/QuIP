divert(-1)	dnl suppress output

include(`../../include/veclib/gen_port.m4')
include(`../../include/veclib/vecgen.m4')
include(`../../include/veclib/gen_host_calls.m4')
include(`../../include/veclib/host_typed_call_defs.m4')
include(`../../include/veclib/host_calls_gpu.m4')
include(`ocl_host_call_defs.m4')

define(`foo',`')
define(`pf_str',`ocl')
define(`VFUNC_NAME',``g_'pf_str`_'type_code`_'$1')
define(`type_code',`sp')
VFUNC_NAME(vsum)

changequote(`[',`]')
define([CHAR_CONST],['$1'])
changequote([`],['])

define(`func',`f2(a = CHAR_CONST(a);)')
define(`f2',`f3($1,x)')
define(`f3',`/* hi */f4($1,$2,b)')
define(`f4',`f5($1,$2,$3,c)')
define(`f5',`f6($1,$2,$3,$4,d)')
define(`f6',`F $1 $2 $3 $4 $5')

define(`_VEC_FUNC_2V',`GENERIC_HOST_TYPED_CALL($1,,,,2)')
define(`DECLARE_OCL_EVENT',`cl_event event;')

dnl	define(`GENERIC_HOST_TYPED_CALL',`			\
dnl								\
dnl	GENERIC_HOST_FAST_CALL($1,$2,$3,$4,$5)			\
dnl	dnl GENERIC_HOST_EQSP_CALL($1,$2,$3,$4,$5)			\
dnl	dnl GENERIC_HOST_SLOW_CALL($1,$2,$3,$4,$5)			\
dnl								\
dnl	dnl GENERIC_HOST_FAST_SWITCH($1,$2,$3,$4,$5)		\
dnl	')

dnl	define(`HOST_FAST_CALL_NAME',	_XXX_FAST_CALL_NAME(h,$1))

dnl	define(`_XXX_SPEED_CALL_NAME',``$1`_'pf_str`_'$3`_'type_code`_'$2'')
dnl	define(`_XXX_FAST_CALL_NAME',_XXX_SPEED_CALL_NAME($1,$2,fast))

dnl	define(`DECLARE_OCL_VARS',`
dnl		static cl_kernel kernel[MAX_OPENCL_DEVICES] = {NULL,NULL,NULL,NULL};
dnl		DECLARE_OCL_COMMON_VARS
dnl	')

dnl	define(`DECLARE_OCL_COMMON_VARS',`
dnl	
dnl		static cl_program program = NULL;
dnl		cl_int status;
dnl		DECLARE_OCL_EVENT
dnl		int ki_idx=0;
dnl		int pd_idx; /* need to set! */
dnl		const char *ksrc;
dnl		/* define the global size and local size
dnl		 * (grid size and block size in CUDA) */
dnl		size_t global_work_size[3] = {1, 1, 1};
dnl		/* size_t local_work_size[3]  = {0, 0, 0}; */
dnl	')

dnl	define(`DECLARE_PLATFORM_FAST_VARS',`DECLARE_PLATFORM_VARS')

dnl	define(`GENERIC_HOST_FAST_CALL',`
dnl	
dnl	static void HOST_FAST_CALL_NAME($1)(LINK_FUNC_ARG_DECLS)
dnl	{
dnl		DECLARE_HOST_FAST_CALL_VARS($2,$5)
dnl	dnl	SETUP_KERNEL_FAST_CALL($1,$2,$3,$4,$5)
dnl	dnl	CALL_FAST_KERNEL($1,$2,$3,$4,$5)
dnl	}
dnl	')

define(`DECLARE_PLATFORM_VARS',`DECLARE_OCL_VARS')

define(`DECLARE_HOST_FAST_CALL_VARS',`DECLARE_PLATFORM_FAST_VARS')

ifdef(`foo',`

f2(a)

divert(0)

dnl _VEC_FUNC_2V( viscntrl,	dst = (dest_type) ( (((src1&0x7f) <= 0x1f)||(src1 == 0x7f )) ? 1 : 0 ) )
include(`../../include/veclib/intvec.m4')

',` dnl else ! foo
/* not foo */
')
