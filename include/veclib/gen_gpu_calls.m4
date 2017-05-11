/* gen_gpu_calls.m4 BEGIN */
// 
// This file contains macros that are useful for writing kernels...
//
// A lot of this stuff is not platform specific!?

my_include(`veclib/gpu_call_utils.m4')

/**********************************************************************/

// args n, s  are func_name, statement

// 5 args
define(`_VEC_FUNC_5V',`GENERIC_GPU_FUNC_CALL($1,$2,,,,5,)')
define(`_VEC_FUNC_4V_SCAL',`GENERIC_GPU_FUNC_CALL($1,$2,,,_1S,4,)')
define(`_VEC_FUNC_3V_2SCAL',`GENERIC_GPU_FUNC_CALL($1,$2,,,_2S,3,)')
define(`_VEC_FUNC_2V_3SCAL',`GENERIC_GPU_FUNC_CALL($1,$2,,,_3S,2,)')

// this is vramp2d
define(`_VEC_FUNC_1V_3SCAL',`SLOW_GPU_FUNC_CALL($1,$4,,,_3S,1,)')

// 3 args
define(`_VEC_FUNC_3V',`GENERIC_GPU_FUNC_CALL($1,$2,,,,3,) ')
define(`_VEC_FUNC_CPX_3V',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,,3,) ')
define(`_VEC_FUNC_QUAT_3V',`GENERIC_GPU_FUNC_CALL($1,$2,,QUAT_,,3,) ')
dnl	this is only used by ramp1d
define(`_VEC_FUNC_1V_2SCAL',`GENERIC_GPU_FUNC_CALL($1,$3,,,_2S,1,RAMP1D) ')
define(`_VEC_FUNC_2V_SCAL',`GENERIC_GPU_FUNC_CALL($1,$2,,,_1S,2,) ')
define(`_VEC_FUNC_VVSLCT',`GENERIC_GPU_FUNC_CALL($1,$2,SBM_,,,3,) ')
define(`_VEC_FUNC_VSSLCT',`GENERIC_GPU_FUNC_CALL($1,$2,SBM_,,_1S,2,) ')
define(`_VEC_FUNC_SSSLCT',`GENERIC_GPU_FUNC_CALL($1,$2,SBM_,,_2S,1,) ')

define(`_VEC_FUNC_SBM_1',`GENERIC_GPU_FUNC_CALL($1,$2,SBM_,,,1,) ')

define(`_VEC_FUNC_1V',`GENERIC_GPU_FUNC_CALL($1,$2,,,,1,) ')
define(`_VEC_FUNC_2V',`GENERIC_GPU_FUNC_CALL($1,$2,,,,2,) ')
define(`_VEC_FUNC_2V_MIXED',`GENERIC_GPU_FUNC_CALL($1,$2,,RC_,,2,) ')
define(`_VEC_FUNC_CPX_2V',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,,2,) ')
define(`_VEC_FUNC_QUAT_2V',`GENERIC_GPU_FUNC_CALL($1,$2,,QUAT_,,2,) ')

define(`_VEC_FUNC_VVMAP',`GENERIC_VEC_FUNC_DBM($1,SET_DBM_BIT( src1 $2 src2 ),,,2SRCS)')
// vsm_gt etc
define(`_VEC_FUNC_VSMAP',`GENERIC_VEC_FUNC_DBM($1,SET_DBM_BIT(src1 $2 scalar1_val),,_1S,1SRC)')

// this is vset
define(`_VEC_FUNC_1V_SCAL',`GENERIC_GPU_FUNC_CALL($1,$2,,,_1S,1,) ')
// where is cpx vset??

// Are these two redundant?
// this is bit_vset
define(`_VEC_FUNC_DBM_1S',`GENERIC_VEC_FUNC_DBM($1,$2,,_1S,)')
// bit_vmov
define(`_VEC_FUNC_DBM_SBM',`GENERIC_VEC_FUNC_DBM($1,$2,,,SBM)')

// vand etc
define(`_VEC_FUNC_DBM_2SBM',`GENERIC_VEC_FUNC_DBM($1,$2,,,2SBM)')
define(`_VEC_FUNC_DBM_1SBM',`GENERIC_VEC_FUNC_DBM($1,$2,,,1SBM)')
// vsand etc
define(`_VEC_FUNC_DBM_1SBM_1S',`GENERIC_VEC_FUNC_DBM($1,$2,,_1S,1SBM)')

define(`_VEC_FUNC_DBM_1V',`GENERIC_VEC_FUNC_DBM($1,$2,,,1SRC)')

define(`_VEC_FUNC_SBM_CPX_3V',`GENERIC_GPU_FUNC_CALL($1,$2,SBM_,CPX_,,3,) ')
define(`_VEC_FUNC_SBM_CPX_2V_1S',`GENERIC_GPU_FUNC_CALL($1,$2,SBM_,CPX_,_1S,2,) ')
define(`_VEC_FUNC_SBM_CPX_1V_2S',`GENERIC_GPU_FUNC_CALL($1,$2,SBM_,CPX_,_2S,1,) ')
define(`_VEC_FUNC_SBM_QUAT_3V',`GENERIC_GPU_FUNC_CALL($1,$2,SBM_,QUAT_,,3,) ')
define(`_VEC_FUNC_SBM_QUAT_2V_1S',`GENERIC_GPU_FUNC_CALL($1,$2,SBM_,QUAT_,_1S,2,) ')
define(`_VEC_FUNC_SBM_QUAT_1V_2S',`GENERIC_GPU_FUNC_CALL($1,$2,SBM_,QUAT_,_2S,1,) ')
define(`_VEC_FUNC_CPX_2V_T2',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,,2,T2) ')
define(`_VEC_FUNC_CPXT_2V',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,,2,T) ')
define(`_VEC_FUNC_CPXT_3V',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,,3,T) ')
define(`_VEC_FUNC_CPXD_3V',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,,3,D) ')
define(`_VEC_FUNC_CPX_2V_1S',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,_1S,2,) ')
define(`_VEC_FUNC_QUAT_2V_1S',`GENERIC_GPU_FUNC_CALL($1,$2,,QUAT_,_1S,2,) ')
define(`_VEC_FUNC_CPX_2V_1S_T2',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,_1S,2,T2) ')
define(`_VEC_FUNC_CPX_2V_1S_T3',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,_1S,2,T3) ')
define(`_VEC_FUNC_QUAT_2V_1S_T4',`GENERIC_GPU_FUNC_CALL($1,$2,,QUAT_,_1S,2,T4) ')
define(`_VEC_FUNC_QUAT_2V_1S_T5',`GENERIC_GPU_FUNC_CALL($1,$2,,QUAT_,_1S,2,T5) ')
define(`_VEC_FUNC_CPXT_2V_1S',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,_1S,2,T) ')
define(`_VEC_FUNC_CPXD_2V_1S',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,_1S,2,D) ')
define(`_VEC_FUNC_CPX_1V_1S',`/* _vec_func_cpx_1v_1s /$1/ /$2/ */GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,_1S,1,) ')
define(`_VEC_FUNC_QUAT_1V_1S',`GENERIC_GPU_FUNC_CALL($1,$2,,QUAT_,_1S,1,)') 
define(`_VEC_FUNC_CPX_3V_T1',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,,3,T1) ')
define(`_VEC_FUNC_CPX_3V_T2',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,,3,T2) ')
define(`_VEC_FUNC_CPX_3V_T3',`GENERIC_GPU_FUNC_CALL($1,$2,,CPX_,,3,T3) ')
define(`_VEC_FUNC_QUAT_2V_T4',`GENERIC_GPU_FUNC_CALL($1,$2,,QUAT_,,2,T4) ')
define(`_VEC_FUNC_QUAT_3V_T4',`GENERIC_GPU_FUNC_CALL($1,$2,,QUAT_,,3,T4) ')
define(`_VEC_FUNC_QUAT_3V_T5',`GENERIC_GPU_FUNC_CALL($1,$2,,QUAT_,,3,T5) ')
define(`_VEC_FUNC_CCR_3V',`GENERIC_GPU_FUNC_CALL($1,$2,,CCR_,,3,) ')
define(`_VEC_FUNC_QQR_3V',`GENERIC_GPU_FUNC_CALL($1,$2,,QQR_,,3,) ')
define(`_VEC_FUNC_CR_2V_1S',`GENERIC_GPU_FUNC_CALL($1,$2,,CR_,_1S,2,) ')
define(`_VEC_FUNC_QR_2V_1S',`GENERIC_GPU_FUNC_CALL($1,$2,,QR_,_1S,2,) ')
// args d,s1,s2 are dst_arg, src_arg1, src_arg2
define(`_VEC_FUNC_VS_LS',`GENERIC_LS_GPU_FUNC_CALL($1,_1S,2,$2,$3,$4)')
define(`_VEC_FUNC_VV_LS',`GENERIC_LS_GPU_FUNC_CALL($1,,3,$2,$3,$4)')

// special case for left shift
// is just for cuda???
define(`GENERIC_LS_GPU_FUNC_CALL',`GENERIC_GPU_FUNC_CALL($1,LSHIFT_SWITCH_32($4,$5,$6),,,$2,$3,)')




// PORT ?

dnl	GENERIC_FAST_VEC_FUNC(name,statement,bitmaps,rc_type,scalars,vectors,extra)

define(`GENERIC_FAST_VEC_FUNC',`_GENERIC_FAST_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')
define(`GENERIC_EQSP_VEC_FUNC',`_GENERIC_EQSP_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')
define(`GENERIC_SLOW_VEC_FUNC',`_GENERIC_SLOW_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')
define(`GENERIC_FLEN_VEC_FUNC',`_GENERIC_FLEN_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')
define(`GENERIC_ELEN_VEC_FUNC',`_GENERIC_ELEN_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')
define(`GENERIC_SLEN_VEC_FUNC',`_GENERIC_SLEN_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)')



// What is this???
define(` GENERIC_FAST_GPU_FUNC_DECL',`
void name(DECLARE_KERN_ARGS_FAST($3,$4,$5,$6))
{
}
')


/* These are for calls with a destination bitmap (vvm_lt etc)
 *
 * Here we cannot vectorize over all the pixels, because multiple
 * pixels share the same bitmap word.  Each thread has to set all the bits
 * in a given word.
 */


// This loops over all of the bits in one word.  We have a problem here if
// all of the bits are not used - there is no harm in reading or setting
// unused bits, but it might cause a seg violation when trying to access
// corresponding non-bit args???  BUG?

dnl	FAST_DBM_LOOP(statement,advance)

define(`FAST_DBM_LOOP',`							\
										\
	/* fast_dbm_loop /$1/ /$2/ */						\
	for(i_dbm_bit=0;i_dbm_bit<BITS_PER_BITMAP_WORD;i_dbm_bit++){		\
		dbm_bit = NUMBERED_BIT(i_dbm_bit);				\
		$1 ;								\
		$2								\
	}									\
')

/* `FLEN_DBM_LOOP' */

define(`FLEN_DBM_LOOP', EQSP_DBM_LOOP($1,$2))

/* `EQSP_DBM_LOOP' */
define(`EQSP_DBM_LOOP',`							\
										\
	for(i_dbm_bit=0;i_dbm_bit<BITS_PER_BITMAP_WORD;i_dbm_bit++){		\
		dbm_bit = NUMBERED_BIT(i_dbm_bit);				\
		if( dbm_info_p->word_tbl[tbl_idx].valid_bits & dbm_bit ){	\
			$1 ;							\
			$2;							\
		}								\
	}									\
')

define(`SLEN_DBM_LOOP', SLOW_DBM_LOOP( $1, $2 ) )

// len is a different type, but here we don't check the other len dimensions!?  BUG?
// We don't necessarily want to set all of the bits in the word, if there is
// a skipping increment?  So this probably won't work for subsamples?  BUG?

dnl define(`SLOW_DBM_LOOP',	FAST_DBM_LOOP( $1, $2 ))

/* `SLOW_DBM_LOOP' */
define(`SLOW_DBM_LOOP',`							\
										\
	for(i_dbm_bit=0;i_dbm_bit<BITS_PER_BITMAP_WORD;i_dbm_bit++){		\
		dbm_bit = NUMBERED_BIT(i_dbm_bit);				\
		if( dbm_info_p->word_tbl[tbl_idx].valid_bits & dbm_bit ){	\
			$1;							\
			$2;							\
		}								\
	}									\
')

dnl	// PORT ?
dnl	// BUG these seem to be re-defined in ocl...

define(`ADVANCE_FAST__1S',`')
define(`ADVANCE_FAST_SBM',`')
define(`ADVANCE_FAST_2SBM',`')
define(`ADVANCE_FAST_1SBM',`')
define(`ADVANCE_FAST_1SBM_1S',`')
define(`ADVANCE_FAST_1SRC_1S',`ADVANCE_FAST_SRC1')
define(`ADVANCE_FAST_1SRC',`ADVANCE_FAST_SRC1')
define(`ADVANCE_FAST_2SRCS',`ADVANCE_FAST_SRC1 ADVANCE_FAST_SRC2')

define(`ADVANCE_EQSP__1S',`')
define(`ADVANCE_EQSP_SBM',`')
define(`ADVANCE_EQSP_SBM1',`')
define(`ADVANCE_EQSP_SBM2',`')
define(`ADVANCE_EQSP_2SBM',`')
define(`ADVANCE_EQSP_1SBM',`')
define(`ADVANCE_EQSP_1SRC_1S',`ADVANCE_EQSP_SRC1')
define(`ADVANCE_EQSP_1SBM_1S',`ADVANCE_EQSP_SBM1')
define(`ADVANCE_EQSP_1SRC',`ADVANCE_EQSP_SRC1')
define(`ADVANCE_EQSP_2SRCS',`ADVANCE_EQSP_SRC1 ADVANCE_EQSP_SRC2')

define(`ADVANCE_SLOW__1S',`')
define(`ADVANCE_SLOW_SBM',`')
define(`ADVANCE_SLOW_2SBM',`')
define(`ADVANCE_SLOW_1SBM',`')
define(`ADVANCE_SLOW_1SBM_1S',`')
define(`ADVANCE_SLOW_1SRC_1S',`ADVANCE_SLOW_SRC1')
define(`ADVANCE_SLOW_1SRC',`ADVANCE_SLOW_SRC1')
define(`ADVANCE_SLOW_2SRCS',`ADVANCE_SLOW_SRC1 ADVANCE_SLOW_SRC2')

define(`ADVANCE_FAST_SRC1',`index2++;')
define(`ADVANCE_FAST_SRC2',`index3++;')
define(`ADVANCE_FAST_SRC3',`index4++;')
define(`ADVANCE_FAST_SRC4',`index5++;')
dnl	define(`ADVANCE_FAST_DBM',`')
dnl	define(`ADVANCE_FAST_SBM',`')

define(`ADVANCE_EQSP_SRC1',`index2 += inc2;')
define(`ADVANCE_EQSP_SRC2',`index3 += inc3;')
define(`ADVANCE_EQSP_SRC3',`index4 += inc4;')
define(`ADVANCE_EQSP_SRC4',`index5 += inc5;')
dnl	define(`ADVANCE_EQSP_DBM',`')
dnl	define(`ADVANCE_EQSP_SBM',`')

dnl BUG - why dim[1] and not dim[0] ???
define(`ADVANCE_SLOW_SRC1',`index2.d5_dim[1]+=inc2.d5_dim[1];')
define(`ADVANCE_SLOW_SRC2',`index3.d5_dim[1]+=inc3.d5_dim[1];')
define(`ADVANCE_SLOW_SRC3',`index4.d5_dim[1]+=inc4.d5_dim[1];')
define(`ADVANCE_SLOW_SRC4',`index5.d5_dim[1]+=inc5.d5_dim[1];')

dnl	define(`ADVANCE_SLOW_DBM',`')
dnl	define(`ADVANCE_SLOW_SBM',`')

dnl	define(`ADVANCE_FAST_DBM_',ADVANCE_FAST_DBM)
dnl	define(`ADVANCE_FAST_DBM_1SRC',ADVANCE_FAST_DBM ADVANCE_FAST_SRC1)
dnl	define(`ADVANCE_FAST_DBM_2SRCS',ADVANCE_FAST_DBM_1SRC ADVANCE_FAST_SRC2)
dnl	define(`ADVANCE_FAST_DBM_SBM',ADVANCE_FAST_DBM ADVANCE_FAST_SBM)

dnl	define(`ADVANCE_EQSP_DBM_',ADVANCE_EQSP_DBM)
dnl	define(`ADVANCE_EQSP_DBM_1SRC',ADVANCE_EQSP_DBM ADVANCE_EQSP_SRC1)
dnl	define(`ADVANCE_EQSP_DBM_1SRC_1S',ADVANCE_EQSP_DBM_1SRC)
dnl	define(`ADVANCE_EQSP_DBM__1S',`')
dnl	define(`ADVANCE_EQSP_DBM_2SRCS',ADVANCE_EQSP_DBM_1SRC ADVANCE_EQSP_SRC2)
dnl	define(`ADVANCE_EQSP_DBM_SBM',ADVANCE_EQSP_DBM ADVANCE_EQSP_SBM)

dnl	define(`ADVANCE_SLOW_DBM_',ADVANCE_SLOW_DBM)
dnl	define(`ADVANCE_SLOW_DBM_1SRC',ADVANCE_SLOW_DBM ADVANCE_SLOW_SRC1)
dnl	define(`ADVANCE_SLOW_DBM_2SRCS',ADVANCE_SLOW_DBM_1SRC ADVANCE_SLOW_SRC2)
dnl	define(`ADVANCE_SLOW_DBM_SBM',ADVANCE_SLOW_DBM ADVANCE_SLOW_SBM)

dnl	GENERIC_FAST_VEC_FUNC_DBM( name, statement, typ, scalars, vectors )
define(`GENERIC_FAST_VEC_FUNC_DBM',`_GENERIC_FAST_VEC_FUNC_DBM($1,$2,$3,$4,$5)')
define(`GENERIC_EQSP_VEC_FUNC_DBM',`_GENERIC_EQSP_VEC_FUNC_DBM($1,$2,$3,$4,$5)')
define(`GENERIC_SLOW_VEC_FUNC_DBM',`_GENERIC_SLOW_VEC_FUNC_DBM($1,$2,$3,$4,$5)')
define(`GENERIC_FLEN_VEC_FUNC_DBM',`_GENERIC_FLEN_VEC_FUNC_DBM($1,$2,$3,$4,$5)')
define(`GENERIC_ELEN_VEC_FUNC_DBM',`_GENERIC_ELEN_VEC_FUNC_DBM($1,$2,$3,$4,$5)')
define(`GENERIC_SLEN_VEC_FUNC_DBM',`_GENERIC_SLEN_VEC_FUNC_DBM($1,$2,$3,$4,$5)')




/* FIXME still need to convert these to generic macros if possible */

define(`_VEC_FUNC_MM',`__VEC_FUNC_MM($1,$2)')
define(`_VEC_FUNC_MM_IND',`__VEC_FUNC_MM_IND($1,$2,$3)')

// rvdot - we need temporary space for the products!?
// The first step should be a normal vmul...

dnl	define(`___VEC_FUNC_FAST_3V_PROJ_SETUP',`			\
dnl	KERNEL_FUNC_PRELUDE						\
dnl	KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_setup')	\
dnl		( DECLARE_KERN_ARGS_FAST_3V_PROJ_SETUP )		\
dnl		FAST_3V_PROJ_BODY(std_type)				\
dnl	')
dnl	
dnl	define(`___VEC_FUNC_FAST_3V_PROJ_HELPER',`			\
dnl									\
dnl	KERNEL_FUNC_PRELUDE						\
dnl									\
dnl	KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_helper')	\
dnl		( DECLARE_KERN_ARGS_FAST_3V_PROJ_HELPER )		\
dnl		FAST_3V_PROJ_BODY(dest_type)				\
dnl	')
dnl	
dnl	define(`FAST_3V_PROJ_BODY',`					\
dnl	{								\
dnl	dnl	INIT_INDICES_1
dnl	dnl
dnl	dnl	if( IDX1 < len2 ){
dnl	dnl		KERNEL_ARG_QUALIFIER $2 *s2;
dnl	dnl		s2 = s1 + len1;
dnl	dnl		dest[IDX1] = $1 ;
dnl	dnl	} else if( IDX1 < len1 ){
dnl	dnl		dest[IDX1] = s1[IDX1];
dnl	dnl	}
dnl		/* do nothing until we figure this out */		\
dnl	}								\
dnl	')
dnl	
dnl	
dnl	/* `CPX_FAST_3V_PROJ_SETUP' */
dnl	
dnl	define(`___VEC_FUNC_CPX_FAST_3V_PROJ_SETUP',`			\
dnl									\
dnl	KERNEL_FUNC_PRELUDE						\
dnl									\
dnl	KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_setup')	\
dnl		( DECLARE_KERN_ARGS_CPX_FAST_3V_PROJ_SETUP )		\
dnl	FAST_CPX_3V_PROJ_BODY($2,$3,std_cpx)				\
dnl	')
dnl	
dnl	/* `CPX_FAST_3V_PROJ_HELPER' */
dnl	define(`___VEC_FUNC_CPX_FAST_3V_PROJ_HELPER',`				\
dnl										\
dnl	KERNEL_FUNC_PRELUDE							\
dnl										\
dnl		KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_helper')	\
dnl			( DECLARE_KERN_ARGS_CPX_FAST_3V_PROJ_HELPER )		\
dnl		FAST_CPX_3V_PROJ_BODY($2,$3,dest_cpx)				\
dnl	')
dnl	
dnl	define(`FAST_CPX_3V_PROJ_BODY',`		\
dnl	{						\
dnl	dnl what should we do here ?  BUG
dnl	}						\
dnl	')


/* `CPX_FAST_2V_PROJ_SETUP' */

dnl needs backslashes so that it can be included in a quoted string for OpenCL...

define(`___VEC_FUNC_CPX_FAST_2V_PROJ_SETUP',`			\
								\
KERNEL_FUNC_PRELUDE						\
								\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_setup')	\
	( DECLARE_KERN_ARGS_CPX_FAST_2V_PROJ_SETUP )		\
FAST_CPX_2V_PROJ_BODY($2,$3,std_cpx)				\
')

/* `CPX_FAST_2V_PROJ_HELPER' */
define(`___VEC_FUNC_CPX_FAST_2V_PROJ_HELPER',`				\
									\
KERNEL_FUNC_PRELUDE							\
									\
	KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_helper')	\
		( DECLARE_KERN_ARGS_CPX_FAST_2V_PROJ_HELPER )		\
	FAST_CPX_2V_PROJ_BODY($2,$3,dest_cpx)				\
')

dnl  needs backslashes for quoting for OpenCL
define(`FAST_CPX_2V_PROJ_BODY',`			\
{							\
	INIT_INDICES_1					\
							\
	if( IDX1 < len2 ){				\
		$3 *s2;					\
		s2 = s1 + len1;				\
		dest[IDX1].re = $1 ;			\
		dest[IDX1].im = $2 ;			\
	} else if( IDX1 < len1 ){			\
		dest[IDX1].re = s1[IDX1].re;		\
		dest[IDX1].im = s1[IDX1].im;		\
	}						\
}							\
')

// `2V_PROJ SETUP' and HELPER do the same thing, but have different input types
// (only relevant for mixed operations, e.g. summing float to double

dnl need backslashes so that OpenCL can quote as string...
define(`___VEC_FUNC_FAST_2V_PROJ_SETUP',`			\
KERNEL_FUNC_PRELUDE						\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_setup')	\
	( DECLARE_KERN_ARGS_FAST_2V_PROJ_SETUP )		\
	FAST_2V_PROJ_BODY($2,std_type)				\
')

define(`___VEC_FUNC_FAST_2V_PROJ_HELPER',`			\
								\
KERNEL_FUNC_PRELUDE						\
								\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_helper')	\
	( DECLARE_KERN_ARGS_FAST_2V_PROJ_HELPER )		\
	FAST_2V_PROJ_BODY($2,dest_type)				\
')

dnl  backslashes needed for quoting!

define(`FAST_2V_PROJ_BODY',`				\
{							\
	INIT_INDICES_1					\
							\
	if( IDX1 < len2 ){				\
		KERNEL_ARG_QUALIFIER $2 *s2;		\
		s2 = s1 + len1;				\
		dest[IDX1] = $1 ;			\
	} else if( IDX1 < len1 ){			\
		dest[IDX1] = s1[IDX1];			\
	}						\
}							\
')

/* `CPX_FAST_2V_PROJ_SETUP' */

define(`___VEC_FUNC_CPX_FAST_2V_PROJ_SETUP',`			\
								\
KERNEL_FUNC_PRELUDE						\
								\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_setup')	\
	( DECLARE_KERN_ARGS_CPX_FAST_2V_PROJ_SETUP )		\
FAST_CPX_2V_PROJ_BODY($2,$3,std_cpx)				\
')

/* `CPX_FAST_2V_PROJ_HELPER' */
define(`___VEC_FUNC_CPX_FAST_2V_PROJ_HELPER',`				\
									\
KERNEL_FUNC_PRELUDE							\
									\
	KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_helper')	\
		( DECLARE_KERN_ARGS_CPX_FAST_2V_PROJ_HELPER )		\
	FAST_CPX_2V_PROJ_BODY($2,$3,dest_cpx)				\
')

define(`FAST_CPX_2V_PROJ_BODY',`			\
{							\
	INIT_INDICES_1					\
							\
	if( IDX1 < len2 ){				\
		$3 *s2;					\
		s2 = s1 + len1;				\
		dest[IDX1].re = $1 ;			\
		dest[IDX1].im = $2 ;			\
	} else if( IDX1 < len1 ){			\
		dest[IDX1].re = s1[IDX1].re;		\
		dest[IDX1].im = s1[IDX1].im;		\
	}						\
}							\
')


/* `QUAT_FAST_2V_PROJ_SETUP' */

define(`___VEC_FUNC_QUAT_FAST_2V_PROJ_SETUP',`			\
								\
KERNEL_FUNC_PRELUDE						\
								\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_setup')	\
	( DECLARE_KERN_ARGS_QUAT_FAST_2V_PROJ_SETUP )		\
FAST_QUAT_2V_PROJ_BODY($2,$3,$4,$5,std_quat)			\
')

define(`___VEC_FUNC_QUAT_FAST_2V_PROJ_HELPER',`			\
								\
KERNEL_FUNC_PRELUDE						\
								\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_helper')	\
	( DECLARE_KERN_ARGS_QUAT_FAST_2V_PROJ_HELPER )		\
FAST_QUAT_2V_PROJ_BODY($2,$3,$4,$5,dest_quat)			\
')


define(`FAST_QUAT_2V_PROJ_BODY',`				\
{								\
	INIT_INDICES_1						\
								\
	if( IDX1 < len2 ){					\
		$5 *s2;						\
		s2 = s1 + len1;					\
		dest[IDX1].re = $1 ;				\
		dest[IDX1]._i = $2 ;				\
		dest[IDX1]._j = $3 ;				\
		dest[IDX1]._k = $4 ;				\
	} else if( IDX1 < len1 ){				\
		dest[IDX1].re = s1[IDX1].re;			\
		dest[IDX1]._i = s1[IDX1]._i;			\
		dest[IDX1]._j = s1[IDX1]._j;			\
		dest[IDX1]._k = s1[IDX1]._k;			\
	}							\
}								\
')



// BUG? does this need to be two macros, one for setup and one for helper?


define(`___VEC_FUNC_FAST_2V_PROJ_IDX',`				\
								\
KERNEL_FUNC_PRELUDE						\
								\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_setup')	\
	( DECLARE_KERN_ARGS_FAST_IDX_SETUP )			\
{								\
	INIT_INDICES_3						\
	if( index3 < len2 )					\
		$2 ;						\
	else if( IDX1 < len1 )					\
		dst = index2 ;					\
}								\
								\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1`_helper')	\
	( DECLARE_KERN_ARGS_FAST_IDX_HELPER )			\
{								\
	INIT_INDICES_3						\
	if( index3 < len2 )					\
		$3 ;						\
	else if( IDX1 < len1 )					\
		dst = src1 ;					\
}								\
								\
')


/* For nocc_setup, we index directly into the value and count temp arrays
 * (a and b, respectively), but we have to double the index for the source
 * array c, and the index array d.  Because we are comparing adjacent pairs, 
 */


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

define(`psrc1',s1[IDX1])
define(`psrc2',s2[IDX1])

// for vsum:   psrc1 + psrc2
// `for vmaxv:  psrc1 > psrc2 ? psrc1 : psrc2'

/* after comment? */

// left shift was broken on cuda, what about now?

define(`LSHIFT_SWITCH_32',`
	switch($3) {
		case 0:  $1 = $2;     break;
		case 1:  $1 = $2<<1;  break;
		case 2:  $1 = $2<<2;  break;
		case 3:  $1 = $2<<3;  break;
		case 4:  $1 = $2<<4;  break;
		case 5:  $1 = $2<<5;  break;
		case 6:  $1 = $2<<6;  break;
		case 7:  $1 = $2<<7;  break;
		case 8:  $1 = $2<<8;  break;
		case 9:  $1 = $2<<9;  break;
		case 10: $1 = $2<<10; break;
		case 11: $1 = $2<<11; break;
		case 12: $1 = $2<<12; break;
		case 13: $1 = $2<<13; break;
		case 14: $1 = $2<<14; break;
		case 15: $1 = $2<<15; break;
		case 16: $1 = $2<<16; break;
		case 17: $1 = $2<<17; break;
		case 18: $1 = $2<<18; break;
		case 19: $1 = $2<<19; break;
		case 20: $1 = $2<<20; break;
		case 21: $1 = $2<<21; break;
		case 22: $1 = $2<<22; break;
		case 23: $1 = $2<<23; break;
		case 24: $1 = $2<<24; break;
		case 25: $1 = $2<<25; break;
		case 26: $1 = $2<<26; break;
		case 27: $1 = $2<<27; break;
		case 28: $1 = $2<<28; break;
		case 29: $1 = $2<<29; break;
		case 30: $1 = $2<<30; break;
		case 31: $1 = $2<<31; break;
	}
')

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

define(`___GPU_FUNC_CALL_2V_PROJ',`				\
								\
KERNEL_FUNC_PRELUDE						\
								\
KERNEL_FUNC_QUALIFIER void `g_'type_code`_'$1			\
	( DECLARE_KERN_ARGS_2V_PROJ )				\
{								\
	INIT_INDICES_1						\
								\
	if( IDX1 < len2 ){					\
		std_type *s2;					\
		s2 = s1 + len1;					\
		dest[IDX1] = $3 ;				\
	} else if( IDX1 < len1 ){				\
		dest[IDX1] = s1[IDX1];				\
	}							\
}								\
')

define(`___VEC_FUNC_MM',`					\
								\
KERNEL_FUNC_PRELUDE						\
								\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_HELPER_NAME($1,$2)		\
( DECLARE_KERN_ARGS_MM )					\
{								\
	INIT_INDICES_3						\
	if( index3.d5_dim[1] < len2 )				\
		$3 ;						\
	else if( IDX1 < len1 )					\
		dst = src1 ;					\
}								\
')


/* For nocc_setup, we index directly into the value and count temp arrays
 * (a and b, respectively), but we have to double the index for the source
 * array c, and the index array d.  Because we are comparing adjacent pairs, 
 *
 * We assume that the data are contiguous, and use fast (single) indices.
 */


define(`___GPU_FUNC_CALL_FAST_MM_NOCC', ___VEC_FUNC_FAST_MM_NOCC( $1, $2, $3 ))


/* For nocc_setup, we index directly into the value and count temp arrays
 * (a and b, respectively), but we have to double the index for the source
 * array c, and the index array d.  Because we are comparing adjacent pairs, 
 */

define(`___VEC_FUNC_FAST_MM_NOCC',`				\
	___VEC_FUNC_FAST_MM_NOCC_SETUP( $1, $2, $3 )		\
	___VEC_FUNC_FAST_MM_NOCC_HELPER( $1, $2, $3 )		\
')


define(`IDX2',index2)

// How are we handling the indices???

define(`___VEC_FUNC_FAST_MM_NOCC_SETUP',`			\
								\
KERNEL_FUNC_PRELUDE						\
								\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_SETUP_NAME($1)		\
	( DECLARE_KERN_ARGS_FAST_NOCC_SETUP )			\
{								\
	INIT_INDICES_2						\
	IDX2 *= 2;						\
	if( IDX1 < len2 ){					\
		if( $2 ){					\
			dst_extrema[IDX1] = src_vals[IDX2];	\
			dst_counts[IDX1]=1;			\
			dst_indices[IDX2]=IDX2;			\
		} else if( $3 ){				\
			dst_extrema[IDX1] = src_vals[IDX2+1];	\
			dst_counts[IDX1]=1;			\
			dst_indices[IDX2]=IDX2+1;		\
		} else {					\
			dst_extrema[IDX1] = src_vals[IDX2];	\
			dst_counts[IDX1]=2;			\
			dst_indices[IDX2]=IDX2;			\
			dst_indices[IDX2+1]=IDX2+1;		\
		}						\
	} else {						\
		/* Nothing to compare */			\
		dst_extrema[IDX1] = src_vals[IDX2];		\
		dst_counts[IDX1]=1;				\
		dst_indices[IDX2]=IDX2;				\
	}							\
}								\
')

// indices and stride example:
//
// src data			ext_val			indices				counts
// 0  1  5  5  5  2  2  2	1   5   5   2		1  X  2   3   4  X  6   7	1  2  1  2		setup, n=4
// 1  5  5  2			5   5			2  3 (2) (3)  4  X (6) (7)	2  1			helper, n=2, stride=4
// 5  5				5			2  3  4  (3) (4) X (6) (7)	3			helper, n=1, stride=8

define(`___VEC_FUNC_FAST_MM_NOCC_HELPER',`				\
									\
KERNEL_FUNC_PRELUDE							\
									\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_HELPER_NAME($1)		\
	( DECLARE_KERN_ARGS_FAST_NOCC_HELPER )				\
{									\
	int i;								\
	INIT_INDICES_2							\
	IDX2 *= 2;							\
	if( IDX1 < len2 ){						\
		if( $2 ){						\
			dst_extrema[IDX1]=src_vals[IDX2];		\
			dst_counts[IDX1]=src_counts[IDX2];		\
			/* No copy necessary */				\
		} else if( $3 ){					\
			dst_extrema[IDX1]=src_vals[IDX2+1];		\
			dst_counts[IDX1]=src_counts[IDX2+1];		\
			/* Now copy the indices down */			\
			for(i=0;i<dst_counts[IDX1];i++){		\
				dst_indices[IDX1*stride+i] =		\
		dst_indices[IDX1*stride+stride/2+i];			\
			}						\
		} else {						\
			dst_extrema[IDX1]=src_vals[IDX2];		\
			dst_counts[IDX1] = src_counts[IDX2] + 		\
				src_counts[IDX2+1];			\
			/* Now copy the second half of the indices */	\
			for(i=0;i<src_counts[IDX2+1];i++){		\
	dst_indices[IDX1*stride+src_counts[IDX2]+i] =			\
		dst_indices[IDX1*stride+stride/2+i];			\
			}						\
		}							\
	} else {							\
		dst_extrema[IDX1]=src_vals[IDX2];			\
		dst_counts[IDX1]=src_counts[IDX2];			\
		/* No copy necessary */					\
	}								\
}									\
')

/* `GENERIC_FAST_VEC_FUNC' */
/* 2V_PROJ is OK but not this??? */
/* that uses GPU_FUNC_FAST_NAME... */
/* used to use GPU_FAST_CALL_NAME... */

dnl	__GENERIC_FAST_VEC_FUNC(name,statement,bitmaps,typ,scalars,vectors,extra)

dnl /* generic_fast_vec_func /$1/ /$2/ /$3/ /$4/ /$5/ /$6/ /$7/ */		\

dnl	/* generic_fast_vec_func statement */				\
dnl	/* statement = $2 */						\
dnl	/* generic_fast_vec_func done */				\


define(`__GENERIC_FAST_VEC_FUNC',`					\
									\
KERNEL_FUNC_PRELUDE							\
									\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1)			\
	(DECLARE_KERN_ARGS_FAST($3,$4,$5,$6))				\
{									\
	DECL_EXTRA($7)							\
	INIT_INDICES($3,$6)						\
	SET_EXTRA_INDICES($7)						\
	$2;								\
}									\
')

dnl	__GENERIC_EQSP_VEC_FUNC( name, statements, bitmaps, typ, scalars, vectors, extra )

define(`__GENERIC_EQSP_VEC_FUNC',`							\
											\
KERNEL_FUNC_PRELUDE									\
											\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_EQSP_NAME($1)(DECLARE_KERN_ARGS_EQSP($3,$4,$5,$6))	\
{											\
	DECL_EXTRA($7)									\
	/* generic_eqsp_vec_func to invoke init_indices */				\
	INIT_INDICES($3,$6)								\
	SET_EXTRA_INDICES($7)								\
	/* generic_eqsp_vec_func to invoke scale_indices */				\
	SCALE_INDICES($3,$6)								\
	$2;										\
}											\
')

// BUG change to CAN_INDEX_THREE_DIMS

define(`__GENERIC_SLOW_VEC_FUNC',`							\
											\
KERNEL_FUNC_PRELUDE									\
											\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_SLOW_NAME($1)(DECLARE_KERN_ARGS_SLOW($3,$4,$5,$6))	\
{											\
	DECL_EXTRA($7)									\
	INIT_INDICES($3,$6)								\
	SET_EXTRA_INDICES($7)								\
	SCALE_INDICES($3,$6)								\
	$2;										\
}											\
')

define(`__GENERIC_FLEN_VEC_FUNC',`							\
											\
KERNEL_FUNC_PRELUDE									\
											\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FLEN_NAME($1)(DECLARE_KERN_ARGS_FLEN($3,$4,$5,$6))	\
{											\
	DECL_EXTRA($7)									\
	INIT_INDICES($3,$6)								\
	SET_EXTRA_INDICES($7)								\
	if( IDX1 < len) {								\
		$2 ;									\
	}										\
}											\
')

define(`__GENERIC_ELEN_VEC_FUNC',`							\
											\
KERNEL_FUNC_PRELUDE									\
											\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_ELEN_NAME($1)(DECLARE_KERN_ARGS_ELEN($3,$4,$5,$6))	\
{											\
	DECL_EXTRA($7)									\
	INIT_INDICES($3,$6)								\
	SET_EXTRA_INDICES($7)								\
	if( IDX1 < len ){								\
		SCALE_INDICES($3,$6)							\
		$2;									\
	}										\
}											\
')

// Does OpenCL have a limit (like CUDA) on the number of dimensions (3)?

define(`SLEN_SUBTST', $1.d5_dim[$3] < $2.d5_dim[$3])

define(`SLEN_IDX_TEST',`( SLEN_SUBTST($1,$2,0) && SLEN_SUBTST($1,$2,1) && SLEN_SUBTST($1,$2,2) && SLEN_SUBTST($1,$2,3) && SLEN_SUBTST($1,$2,4) )')

define(`__GENERIC_SLEN_VEC_FUNC',`							\
											\
KERNEL_FUNC_PRELUDE									\
/* generic_slen_vec_func /$1/ /$2/ /$3/ /$4/ /$5/ /$6/ /$7/ */				\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_SLEN_NAME($1)(DECLARE_KERN_ARGS_SLEN($3,$4,$5,$6))	\
{											\
	DECL_EXTRA($7)									\
	INIT_INDICES($3,$6)								\
	SET_EXTRA_INDICES($7)								\
	if( SLEN_IDX_TEST(index1,szarr) ){						\
		SCALE_INDICES($3,$6)							\
		$2;									\
	}										\
}											\
')


dnl	/* CUDA definitions? */

dnl	define(`REAL_CONVERSION',`
dnl	FAST_CONVERSION($1,$2,$3,$4)
dnl	EQSP_CONVERSION($1,$2,$3,$4)
dnl	SLOW_CONVERSION($1,$2,$3,$4)
dnl	FLEN_CONVERSION($1,$2,$3,$4)
dnl	ELEN_CONVERSION($1,$2,$3,$4)
dnl	SLEN_CONVERSION($1,$2,$3,$4)
dnl	')


dnl	define(`REAL_CONVERSION',`
dnl	FAST_CONVERSION($1,$2,$3,$4)
dnl	EQSP_CONVERSION($1,$2,$3,$4)
dnl	SLOW_CONVERSION($1,$2,$3,$4)
dnl	')

dnl	define(`FAST_CONVERSION',`_GENERIC_FAST_CONV_FUNC( `v'$1`2'$3, $2, $4 )')
dnl	define(`EQSP_CONVERSION',`_GENERIC_EQSP_CONV_FUNC( `v'$1`2'$3, $2, $4 )')
dnl	define(`SLOW_CONVERSION',`_GENERIC_SLOW_CONV_FUNC( `v'$1`2'$3, $2, $4 )')
dnl	define(`FLEN_CONVERSION',`_GENERIC_FLEN_CONV_FUNC( `v'$1`2'$3, $2, $4 )')
dnl	define(`ELEN_CONVERSION',`_GENERIC_ELEN_CONV_FUNC( `v'$1`2'$3, $2, $4 )')
dnl	define(`SLEN_CONVERSION',`_GENERIC_SLEN_CONV_FUNC( `v'$1`2'$3, $2, $4 )')

dnl	__GENERIC_FAST_CONV_FUNC(name,dest_type)

define(`__GENERIC_FAST_CONV_FUNC',`							\
											\
KERNEL_FUNC_PRELUDE									\
											\
KERNEL_FUNC_QUALIFIER void GPU_FAST_CALL_NAME($1)(DECLARE_KERN_ARGS_FAST_CONV($2))	\
{											\
	INIT_INDICES_2									\
	dst = ($2) src1 ;								\
}											\
')

dnl	__GENERIC_EQSP_CONV_FUNC(name,dest_prec)

define(`__GENERIC_EQSP_CONV_FUNC',`							\
											\
KERNEL_FUNC_PRELUDE									\
											\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_EQSP_NAME($1)(DECLARE_KERN_ARGS_EQSP_CONV($2))	\
{											\
	INIT_INDICES_2									\
	SCALE_INDICES_2									\
	dst = ($2) src1 ;								\
}											\
')

dnl	__GENERIC_SLOW_CONV_FUNC(name,dest_prec)

define(`__GENERIC_SLOW_CONV_FUNC',`							\
											\
KERNEL_FUNC_PRELUDE									\
											\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_SLOW_NAME($1)(DECLARE_KERN_ARGS_SLOW_CONV($2))	\
{											\
	INIT_INDICES_2									\
	SCALE_INDICES_2									\
	dst = ($2) src1 ;								\
}											\
')

define(`__GENERIC_FLEN_CONV_FUNC',`							\
											\
KERNEL_FUNC_PRELUDE									\
											\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FLEN_NAME($1)(DECLARE_KERN_ARGS_FLEN_CONV($2))	\
{											\
	INIT_INDICES_2									\
	if( IDX1 < len) {								\
		dst = ($2) src1 ;							\
	}										\
}											\
')

define(`__GENERIC_ELEN_CONV_FUNC',`							\
											\
KERNEL_FUNC_PRELUDE									\
											\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_ELEN_NAME($1)(DECLARE_KERN_ARGS_ELEN_CONV($2))	\
{											\
	INIT_INDICES_2									\
	if( IDX1 < len) {								\
		SCALE_INDICES_2								\
		dst = ($2) src1 ;							\
	}										\
}											\
')

define(`__GENERIC_SLEN_CONV_FUNC',`							\
											\
KERNEL_FUNC_PRELUDE									\
											\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_SLEN_NAME($1)(DECLARE_KERN_ARGS_SLEN_CONV($2))	\
{											\
	INIT_INDICES_2									\
	if( SLEN_IDX_TEST(index1,szarr) ){						\
		SCALE_INDICES_2								\
		dst = ($2) src1 ;							\
	}										\
}											\
')

dnl	__GENERIC_FAST_VEC_FUNC_DBM( name, statement, typ, scalars, vectors )

define(`__GENERIC_FAST_VEC_FUNC_DBM',`								\
												\
KERNEL_FUNC_PRELUDE										\
												\
/* generic_fast_vec_func_dbm /$1/ /$2/ /$3/ /$4/ /$5/ */					\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FAST_NAME($1)(DECLARE_KERN_ARGS_FAST(`DBM_',$3,$4,$5))	\
{												\
	/* generic_fast_vec_func_dbm */								\
	INIT_INDICES(`DBM_',$5)									\
	/* generic_fast_vec_func dbm calling fast_dbm_loop /$2/ /$5/ */				\
	FAST_DBM_LOOP( $2, ADVANCE_FAST_$5$4)							\
}												\
')

// EQSP is tricky because the number of relevant bits in a word is no
// longer all of the bits - so the LOOP should just loop over the bits
// in a single word!?  BUG?

define(`__GENERIC_EQSP_VEC_FUNC_DBM',`								\
												\
KERNEL_FUNC_PRELUDE										\
												\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_EQSP_NAME($1)(DECLARE_KERN_ARGS_EQSP(`DBM_',$3,$4,$5))	\
{												\
	/* generic_eqsp_vec_func_dbm /$1/ /$2/ /$3/ /$4/ /$5/ */				\
	INIT_INDICES(`DBM_',$5)									\
	/* generic_eqsp_vec_func_dbm 2 */							\
	SCALE_INDICES(`DBM_',$5)								\
	/* generic_eqsp_vec_func_dbm 3 */							\
	EQSP_DBM_LOOP($2,ADVANCE_EQSP_$5$4)							\
}												\
')


dnl	__GENERIC_SLOW_VEC_FUNC_DBM( name, statement, typ, scalars, vectors )

define(`__GENERIC_SLOW_VEC_FUNC_DBM',`								\
												\
KERNEL_FUNC_PRELUDE										\
												\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_SLOW_NAME($1)(DECLARE_KERN_ARGS_SLOW(`DBM_',$3,$4,$5))	\
{												\
	/* generic_slow_vec_func_dbm */								\
	INIT_INDICES(`DBM_',$5)									\
	SET_EXTRA_INDICES($7)									\
	/* scale_indices /DBM_/ /$5/ */								\
	SCALE_INDICES(`DBM_',$5)								\
												\
	/* slow_dbm_loop /$2/ /$5/ */								\
	SLOW_DBM_LOOP( $2 , ADVANCE_SLOW_$5$4)							\
}												\
												\
')

dnl	__GENERIC_FLEN_VEC_FUNC_DBM(name,statement,rc_type,scalars,vectors)
define(`__GENERIC_FLEN_VEC_FUNC_DBM',`								\
												\
KERNEL_FUNC_PRELUDE										\
												\
/* __generic_flen_vec_func_dbm /$1/ /$2/ /$3/ /$4/ /$5/ */					\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_FLEN_NAME($1)(DECLARE_KERN_ARGS_FLEN(`DBM_',$3,$4,$5))	\
{												\
	INIT_INDICES(`DBM_',$5)									\
	FLEN_DBM_LOOP($2,ADVANCE_FAST_$5$4)							\
}												\
')

define(`__GENERIC_ELEN_VEC_FUNC_DBM',`								\
												\
KERNEL_FUNC_PRELUDE										\
												\
dnl KERNEL_FUNC_QUALIFIER void GPU_FUNC_ELEN_NAME($1)(DECLARE_KERN_ARGS_ELEN($3,$4,$5,$6))
KERNEL_FUNC_QUALIFIER void GPU_FUNC_ELEN_NAME($1)(DECLARE_KERN_ARGS_ELEN(`DBM_',$3,$4,$5))	\
{												\
	INIT_INDICES(`DBM_',$5)									\
	SCALE_INDICES(`DBM_',$5)								\
	FLEN_DBM_LOOP( $2,ADVANCE_EQSP_$5$4)							\
}												\
')



define(`__GENERIC_SLEN_VEC_FUNC_DBM',`								\
												\
KERNEL_FUNC_PRELUDE										\
												\
KERNEL_FUNC_QUALIFIER void GPU_FUNC_SLEN_NAME($1)(DECLARE_KERN_ARGS_SLEN(`DBM_',$3,$4,$5))	\
{												\
	INIT_INDICES(`DBM_',$5)									\
	SCALE_INDICES(`DBM_',$5)								\
												\
	/* BUG need to put len test in statement */						\
	/* BUG can we test before scaling??? */							\
	SLEN_DBM_LOOP( $2 , ADVANCE_SLOW_$5$4)							\
}												\
')


// BUG use macro for helper name

define(`___GPU_FUNC_CALL_MM',`				\
							\
KERNEL_FUNC_PRELUDE					\
							\
KERNEL_FUNC_QUALIFIER void `g_'$2`_'$1`_helper'		\
( DECLARE_KERN_ARGS_MM )				\
{							\
	INIT_INDICES_3					\
	if( index3.d5_dim[1] < len2 )			\
		$3 ;					\
	else if( IDX1 < len1 )				\
		dst = src1 ;				\
}							\
							\
')


dnl fast or slow???

define(`___GPU_FUNC_CALL_MM_IND',`						\
										\
KERNEL_FUNC_PRELUDE								\
										\
KERNEL_FUNC_QUALIFIER void `g_'$2`_'$1`_setup'					\
(index_type* a, std_type* b, std_type* c, uint32_t len1, uint32_t len2)		\
{										\
	INIT_INDICES_3								\
	if( index3.d5_dim[1] < len2 )						\
		$3 ;								\
	else if( IDX1 < len1 )							\
		dst = index2.d5_dim[1] ;					\
}										\
										\
KERNEL_FUNC_QUALIFIER void `g_'$2`_'$1`_helper'					\
	(index_type* a, index_type* b, index_type* c,				\
			std_type *orig, uint32_t len1, uint32_t len2)		\
{										\
	INIT_INDICES_3								\
	if( index3.d5_dim[1] < len2 )						\
		$3 ;								\
	else if( IDX1 < len1 )							\
		dst = src1 ;							\
}										\
										\
')



/* gen_gpu_calls.m4 END */
