/* ocl_kernel_src.m4 BEGIN */

my_include(`veclib/gen_gpu_calls.m4')
my_include(`veclib/gpu_special_defs.m4')

define(`QUOTE_CHAR',`"')

define(`QUOTE_IT',QUOTE_CHAR$1QUOTE_CHAR)

dnl all these quotes because type_code is not defined until later...
dnl type_code not expanded???
dnl	KERN_SOURCE_NAME( func_name, speed_string )
define(`KERN_SOURCE_NAME',`/* kern_source_name $1 $2 */ `kernel_source_'pf_str`_'$2`_'type_code`_'$1')

dnl	_GENERIC_FAST_CONV_FUNC(name,dest_type)
define(`_GENERIC_FAST_CONV_FUNC',`char KERN_SOURCE_NAME($1,fast)[]=QUOTE_IT( __GENERIC_FAST_CONV_FUNC($1,$2) );')
define(`_GENERIC_EQSP_CONV_FUNC',`char KERN_SOURCE_NAME($1,eqsp)[]=QUOTE_IT( __GENERIC_EQSP_CONV_FUNC($1,$2) );')
define(`_GENERIC_SLOW_CONV_FUNC',`char KERN_SOURCE_NAME($1,slow)[]=QUOTE_IT( __GENERIC_SLOW_CONV_FUNC($1,$2) );')


define(`_GENERIC_FAST_VEC_FUNC',`char KERN_SOURCE_NAME($1,fast)[]= QUOTE_IT( __GENERIC_FAST_VEC_FUNC($1,$2,$3,$4,$5,$6,$7) );')
define(`_GENERIC_EQSP_VEC_FUNC',`char KERN_SOURCE_NAME($1,eqsp)[]= QUOTE_IT( __GENERIC_EQSP_VEC_FUNC($1,$2,$3,$4,$5,$6,$7) );')
define(`_GENERIC_SLOW_VEC_FUNC',`char KERN_SOURCE_NAME($1,slow)[]= QUOTE_IT( __GENERIC_SLOW_VEC_FUNC($1,$2,$3,$4,$5,$6,$7) );')


// For cuda, this is __global__
define(`KERNEL_FUNC_QUALIFIER',`__kernel')


dnl	What are the args???
define(`GENERIC_FAST_VEC_FUNC_DBM',`_GENERIC_FAST_VEC_FUNC_DBM($1,$2,$3,$4,$5)')
define(`GENERIC_EQSP_VEC_FUNC_DBM',`_GENERIC_EQSP_VEC_FUNC_DBM($1,$2,$3,$4,$5)')
define(`GENERIC_SLOW_VEC_FUNC_DBM',`_GENERIC_SLOW_VEC_FUNC_DBM($1,$2,$3,$4,$5)')

define(`_GENERIC_FAST_VEC_FUNC_DBM',`
char KERN_SOURCE_NAME($1,fast)[] = QUOTE_IT( __GENERIC_FAST_VEC_FUNC_DBM($1,$2,$3,$4,$5) );')

define(`_GENERIC_EQSP_VEC_FUNC_DBM',`
char KERN_SOURCE_NAME($1,eqsp)[] = QUOTE_IT( __GENERIC_EQSP_VEC_FUNC_DBM($1,$2,$3,$4,$5) );')

define(`_GENERIC_SLOW_VEC_FUNC_DBM',`
char KERN_SOURCE_NAME($1,slow)[] = QUOTE_IT( __GENERIC_SLOW_VEC_FUNC_DBM($1,$2,$3,$4,$5) );')



/* FIXME still need to convert these to generic macros if possible */

define(`_VEC_FUNC_MM',`__VEC_FUNC_MM($1,$2 );')

define(`__VEC_FUNC_MM',`char KERN_SOURCE_NAME($1,mm)[]=QUOTE_IT(___VEC_FUNC_MM($1,$2 ) );')

define(`_VEC_FUNC_MM_IND',__VEC_FUNC_MM_IND($1,$2,$3)')

define(`__VEC_FUNC_MM_IND',`char KERN_SOURCE_NAME($1,mm_ind)[] = QUOTE_IT( ___VEC_FUNC_MM_IND($1,$2,$3 ) );')

/* For nocc_setup, we index directly into the value and count temp arrays
 * (a and b, respectively), but we have to double the index for the source
 * array c, and the index array d.  Because we are comparing adjacent pairs, 
 */

dnl define(`__VEC_FUNC_FAST_MM_NOCC',`
define(`_VEC_FUNC_FAST_MM_NOCC',`
char KERN_SOURCE_NAME($1`_setup',fast)[]= QUOTE_IT( ___VEC_FUNC_FAST_MM_NOCC_SETUP($1,$2,$3) );
char KERN_SOURCE_NAME($1`_nocc_helper',fast)[]= QUOTE_IT( ___VEC_FUNC_FAST_MM_NOCC_HELPER($1,$2,$3) );
')

define(`_VEC_FUNC_EQSP_MM_NOCC',`')

// vsum, vdot, etc
// BUG this is hard-coded for vsum!?
//
// The idea is that , because all threads cannot access the destination simultaneously,
// we have to make the source smaller recursively...  But when we project
// to a vector instead of a scalar, we can to the elements of the vector in parallel...
// This is quite tricky.
//
// Example:  col=sum(m)
//
// m = | 1 2 3 4 |
//     | 5 6 7 8 |
//
// tmp = | 4  6  |
//       | 12 14 |
//
// col = | 10 |
//       | 26 |
     

// BUG - we need to make this do vmaxv and vminv as well.
// It's the same except for the sum line, which would be replaced with
//

dnl	Moved these to include/veclib/gpu_special_defs.m4
dnl	dnl	Some no-ops
dnl	define(`_VEC_FUNC_SLOW_2V_PROJ',`')
dnl	define(`_VEC_FUNC_SLOW_2V_PROJ_IDX',`')
dnl	define(`_VEC_FUNC_SLOW_CPX_2V_PROJ',`')
dnl	define(`_VEC_FUNC_SLOW_CPX_2V_PROJ_IDX',`')
dnl	define(`_VEC_FUNC_SLOW_QUAT_2V_PROJ',`')
dnl	define(`_VEC_FUNC_SLOW_QUAT_2V_PROJ_IDX',`')
dnl	define(`_VEC_FUNC_SLOW_3V_PROJ',`')
dnl	define(`_VEC_FUNC_SLOW_MM_NOCC',`')

dnl	define(`_VEC_FUNC_DBM_2SBM',`')	dnl	BUG - need to implement!
dnl	define(`_VEC_FUNC_DBM_1SBM',`')	dnl	BUG - need to implement!
dnl	define(`_VEC_FUNC_DBM_1SBM_1S',`')	dnl	BUG - need to implement!

define(`_VEC_FUNC_SLOW_1V_3SCAL',`SLOW_GPU_FUNC_CALL($1,$4,,,_3S,1,)')

// for vsum:   psrc1 + psrc2
// ocl_kernel_src.m4:  for vmaxv:  psrc1 > psrc2 ? psrc1 : psrc2


/* `VEC_FUNC_FAST_2V_PROJ_SETUP' */

define(`__VEC_FUNC_FAST_2V_PROJ_SETUP',`
char KERN_SOURCE_NAME($1`_setup',fast)[]= QUOTE_IT( ___VEC_FUNC_FAST_2V_PROJ_SETUP($1,$2) );
')

/* `VEC_FUNC_FAST_2V_PROJ_HELPER' */

define(`__VEC_FUNC_FAST_2V_PROJ_HELPER',`
char KERN_SOURCE_NAME($1`_helper',fast)[]= QUOTE_IT( ___VEC_FUNC_FAST_2V_PROJ_HELPER($1,$2) );
')

dnl	We don't need kernels for vdot any more because we use the vmul and vsum kernels to build the
dnl	operation!

dnl define(`__VEC_FUNC_FAST_3V_PROJ_SETUP',`
dnl char KERN_SOURCE_NAME($1`_setup',fast)[]=QUOTE_IT(___VEC_FUNC_FAST_3V_PROJ_SETUP($1));
dnl ')
dnl 
dnl define(`__VEC_FUNC_FAST_3V_PROJ_HELPER',`
dnl char KERN_SOURCE_NAME($1`_helper',fast)[]=QUOTE_IT(___VEC_FUNC_FAST_3V_PROJ_HELPER($1));
dnl ')
dnl 
dnl define(`__VEC_FUNC_CPX_FAST_3V_PROJ_SETUP',`
dnl char KERN_SOURCE_NAME($1`_setup',fast)[]= QUOTE_IT(___VEC_FUNC_CPX_FAST_3V_PROJ_SETUP($1));
dnl ')
dnl 
dnl define(`__VEC_FUNC_CPX_FAST_3V_PROJ_HELPER',`
dnl char KERN_SOURCE_NAME($1`_helper',fast)[]=QUOTE_IT(___VEC_FUNC_CPX_FAST_3V_PROJ_HELPER($1));
dnl ')

/* `VEC_FUNC_CPX_FAST_2V_PROJ_SETUP' */

define(`__VEC_FUNC_CPX_FAST_2V_PROJ_SETUP',`
char KERN_SOURCE_NAME($1`_setup',fast)[]= QUOTE_IT( ___VEC_FUNC_CPX_FAST_2V_PROJ_SETUP($1, $2, $3 ) );
')

/* `VEC_FUNC_CPX_FAST_2V_PROJ_HELPER' */

define(`__VEC_FUNC_CPX_FAST_2V_PROJ_HELPER',`
char KERN_SOURCE_NAME($1`_helper',fast)[]= QUOTE_IT( ___VEC_FUNC_CPX_FAST_2V_PROJ_HELPER($1, $2, $3 ) );
')

/* `VEC_FUNC_QUAT_FAST_2V_PROJ_SETUP' */

define(`__VEC_FUNC_QUAT_FAST_2V_PROJ_SETUP',`
char KERN_SOURCE_NAME($1`_setup',fast)[]= QUOTE_IT( ___VEC_FUNC_QUAT_FAST_2V_PROJ_SETUP($1, $2, $3, $4, $5 ) );
')

/* `VEC_FUNC_QUAT_FAST_2V_PROJ_HELPER' */

define(`__VEC_FUNC_QUAT_FAST_2V_PROJ_HELPER',`
char KERN_SOURCE_NAME($1`_helper',fast)[]= QUOTE_IT( ___VEC_FUNC_QUAT_FAST_2V_PROJ_HELPER($1, $2, $3, $4, $5 ) );
')

/* `VEC_FUNC_FAST_2V_PROJ_IDX' */

define(`__VEC_FUNC_FAST_2V_PROJ_IDX',`
char KERN_SOURCE_NAME($1,fast)[]=QUOTE_IT(___VEC_FUNC_FAST_2V_PROJ_IDX($1, $2, $3 ) );
')


/* ocl_kernel_src.m4 END */
