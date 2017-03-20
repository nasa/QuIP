/* This file has the following sections:
 *
 * 1 - definitions
 * 2 - slow loop bodies
 * 3 - slow functions
 * 4 - obsolete FOOBAR loops
 * 5 - fast functions
 * 6 - fast loop bodies
 * 7 - fast/slow switches
 * 8 - object methods
 */

define(`BUILD_FOR_CPU',`')
dnl #undef BUILD_FOR_HOST

/* use 64 bit words for data movement when possible.  Pointers have
 * to be aligned to 8 byte boundaries.
 */

dnl	L_ALIGNMENT(a)
define(`L_ALIGNMENT',`(((int_for_addr)$1) & 7)')

define(`XFER_EQSP_DBM_GPU_INFO',`')	dnl	/* nop on cpu */
define(`XFER_SLOW_DBM_GPU_INFO',`')	dnl	/* nop on cpu */

/***************** Section 1 - definitions **********************/

dnl #include "calling_args.h"	dnl  declaration args, shared

dnl  some of this stuff is obsolete...
dnl #define FFT_METHOD_NAME(stem)	TYPED_NAME( _##_fft_##stem )
dnl #define OBJ_METHOD_NAME(stem)	HOST_TYPED_CALL_NAME( stem , type_code )

dnl	CONV_METHOD_NAME(stem)
define(`CONV_METHOD_NAME',`CPU_CALL_NAME($1)')
define(`FAST_NAME',`CPU_FAST_CALL_NAME($1)')
define(`SLOW_NAME',`CPU_SLOW_CALL_NAME($1)')
define(`FAST_CONV_NAME',`CPU_FAST_CALL_NAME($1)')
define(`SLOW_CONV_NAME',`CPU_SLOW_CALL_NAME($1)')
define(`EQSP_NAME',`CPU_EQSP_CALL_NAME($1)')

dnl	TYPED_NAME(s)
define(`TYPED_NAME',`type_code`_'$1')

define(`TYPED_STRING',`"TYPED_NAME($1)"')

define(`dst_dp',`OA_DEST(oap)')
dnl /*
dnl #define src1_dp		OA_SRC1(oap)
dnl #define src2_dp		OA_SRC2(oap)
dnl #define src3_dp		OA_SRC3(oap)
dnl #define src4_dp		OA_SRC4(oap)
dnl */
define(`SRC_DP(idx)',`OA_SRC_OBJ(oap,idx)')
define(`SRC1_DP',`SRC_DP(0)')
define(`SRC2_DP',`SRC_DP(1)')
define(`SRC3_DP',`SRC_DP(2)')
define(`SRC4_DP',`SRC_DP(3)')
define(`SRC5_DP',`SRC_DP(4)')

define(`bitmap_dst_dp',`OA_DEST(oap)')
define(`bitmap_src_dp',`OA_SBM(oap)')

define(`MAX_DEBUG',`')

ifdef(`MAX_DEBUG',`

define(`ANNOUNCE_FUNCTION',`

	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,"BEGIN function %s",db_func_name);
		NADVISE(DEFAULT_ERROR_STRING);
	}
')

define(`REPORT_OBJ_METHOD_DONE',`

	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,"Function %s DONE",db_func_name);
		NADVISE(DEFAULT_ERROR_STRING);
	}
')

define(`REPORT_FAST_CALL',`

	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,"Function %s calling fast func",db_func_name);
		NADVISE(DEFAULT_ERROR_STRING);
	}
')

define(`REPORT_EQSP_CALL',`

	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,"Function %s calling eqsp func",db_func_name);
		NADVISE(DEFAULT_ERROR_STRING);
	}
')

define(`REPORT_SLOW_CALL',`

	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,"Function %s calling slow func",db_func_name);
		NADVISE(DEFAULT_ERROR_STRING);
	}
')


dnl	DECLARE_FUNC_NAME(name)
define(`DECLARE_FUNC_NAME',`const char * db_func_name=STRINGIFY( OBJ_METHOD_NAME($1) );')

',` dnl else /* ! MAX_DEBUG */

define(`ANNOUNCE_FUNCTION',`')
define(`REPORT_OBJ_METHOD_DONE',`')
define(`REPORT_FAST_CALL',`')
define(`REPORT_EQSP_CALL',`')
define(`REPORT_SLOW_CALL',`')
define(`DECLARE_FUNC_NAME',`')
define(`db_func_name',`NULL')

') dnl endif /* ! MAX_DEBUG */



dnl	CHECK_MATCH(dp1,dp2)
define(`CHECK_MATCH',`

	if( ! dp_compatible( $1, $2 ) ){
		if( UNKNOWN_SHAPE(&$1->dt_shape) &&
				! UNKNOWN_SHAPE($2) ){
			install_shape($1,$2);
		} else if( UNKNOWN_SHAPE($2) &&
				! UNKNOWN_SHAPE($1) ){
			install_shape($2,$1);
		} else {
			sprintf(DEFAULT_ERROR_STRING,
"Shape mismatch between objects %s and %s",OBJ_NAME($1),OBJ_NAME($2));
			NWARN(DEFAULT_ERROR_STRING);
			return;
		}
	}
')


define(`OBJ_ARG_CHK_DBM',`

	ANNOUNCE_FUNCTION
	if( bitmap_dst_dp == NO_OBJ ){
NWARN("OBJ_ARG_CHK_DBM:  Null bitmap destination object!?");
		return;
	}
')

define(`OBJ_ARG_CHK_DBM_',`OBJ_ARG_CHK_DBM')

define(`OBJ_ARG_CHK_SBM',`

	if( bitmap_src_dp == NO_OBJ ){
		NWARN("Null bitmap source object!?");
		return;
	}
')


ifdef(`CAUTIOUS',`

define(`OBJ_ARG_CHK_1',`

	ANNOUNCE_FUNCTION
	OBJ_ARG_CHK(dst_dp,"destination")
')

dnl	OBJ_ARG_CHK(dp,string)
define(`OBJ_ARG_CHK',`

	if( $1==NO_OBJ ){
		sprintf(DEFAULT_ERROR_STRING,
			"CAUTIOUS:  Null %s object!?",$2);
		NERROR1(DEFAULT_ERROR_STRING);
		IOS_RETURN
	}
')


define(`OBJ_ARG_CHK_SRC1',`OBJ_ARG_CHK(SRC1_DP,"first source")')
define(`OBJ_ARG_CHK_SRC2',`OBJ_ARG_CHK(SRC2_DP,"second source")')
define(`OBJ_ARG_CHK_SRC3',`OBJ_ARG_CHK(SRC3_DP,"third source")')
define(`OBJ_ARG_CHK_SRC4',`OBJ_ARG_CHK(SRC4_DP,"fourth source")')

',` dnl else /* ! CAUTIOUS */

define(`OBJ_ARG_CHK_1',`ANNOUNCE_FUNCTION')

define(`OBJ_ARG_CHK_SRC1',`')
define(`OBJ_ARG_CHK_SRC2',`')
define(`OBJ_ARG_CHK_SRC3',`')
define(`OBJ_ARG_CHK_SRC4',`')

') dnl endif /* CAUTIOUS */

define(`OBJ_ARG_CHK_2SRCS',`OBJ_ARG_CHK_SRC1 OBJ_ARG_CHK_SRC2')
define(`OBJ_ARG_CHK_DBM_2SRCS',`OBJ_ARG_CHK_DBM OBJ_ARG_CHK_2SRCS')
define(`OBJ_ARG_CHK_DBM_1SRC',`OBJ_ARG_CHK_DBM OBJ_ARG_CHK_SRC1')
define(`OBJ_ARG_CHK_DBM_SBM',`OBJ_ARG_CHK_DBM OBJ_ARG_CHK_SBM')

define(`OBJ_ARG_CHK_',`')
define(`OBJ_ARG_CHK_2',`OBJ_ARG_CHK_1 OBJ_ARG_CHK_SRC1')
define(`OBJ_ARG_CHK_3',`OBJ_ARG_CHK_2 OBJ_ARG_CHK_SRC2')
define(`OBJ_ARG_CHK_4',`OBJ_ARG_CHK_3 OBJ_ARG_CHK_SRC3')
define(`OBJ_ARG_CHK_5',`OBJ_ARG_CHK_4 OBJ_ARG_CHK_SRC4')

define(`OBJ_ARG_CHK_SBM_1',`OBJ_ARG_CHK_SBM OBJ_ARG_CHK_1')
define(`OBJ_ARG_CHK_SBM_2',`OBJ_ARG_CHK_SBM OBJ_ARG_CHK_2')
define(`OBJ_ARG_CHK_SBM_3',`OBJ_ARG_CHK_SBM OBJ_ARG_CHK_3')

dnl

dnl	FAST_ADVANCE(typ,suffix)
define(`FAST_ADVANCE',`_FAST_ADVANCE($1$2)')
define(`_FAST_ADVANCE',FAST_ADVANCE_$1)

define(`FAST_ADVANCE_SRC1',`s1_ptr++ ;')
define(`FAST_ADVANCE_SRC2',`s2_ptr++ ;')
define(`FAST_ADVANCE_SRC3',`s3_ptr++ ;')
define(`FAST_ADVANCE_SRC4',`s4_ptr++ ;')
define(`FAST_ADVANCE_1',`dst_ptr++ ;')
define(`FAST_ADVANCE_CPX_1',`cdst_ptr++ ;')
define(`FAST_ADVANCE_CPX_SRC1',`cs1_ptr++ ;')
define(`FAST_ADVANCE_CPX_SRC2',`cs2_ptr++ ;')
define(`FAST_ADVANCE_CPX_SRC3',`cs3_ptr++ ;')
define(`FAST_ADVANCE_CPX_SRC4',`cs4_ptr++ ;')
define(`FAST_ADVANCE_QUAT_1',`qdst_ptr++ ;')
define(`FAST_ADVANCE_QUAT_SRC1',`qs1_ptr++ ;')
define(`FAST_ADVANCE_QUAT_SRC2',`qs2_ptr++ ;')
define(`FAST_ADVANCE_QUAT_SRC3',`qs3_ptr++ ;')
define(`FAST_ADVANCE_QUAT_SRC4',`qs4_ptr++ ;')
dnl #define FAST_ADVANCE_BITMAP	which_bit ++ ;
define(`FAST_ADVANCE_DBM',`dbm_bit ++ ;')
define(`FAST_ADVANCE_SBM',`sbm_bit ++ ;')
define(`FAST_ADVANCE_DBM_SBM',`FAST_ADVANCE_DBM FAST_ADVANCE_SBM')

define(`FAST_ADVANCE_2',`FAST_ADVANCE_1 FAST_ADVANCE_SRC1')
define(`FAST_ADVANCE_3',`FAST_ADVANCE_2 FAST_ADVANCE_SRC2')
define(`FAST_ADVANCE_4',`FAST_ADVANCE_3 FAST_ADVANCE_SRC3')
define(`FAST_ADVANCE_5',`FAST_ADVANCE_4 FAST_ADVANCE_SRC4')

define(`FAST_ADVANCE_CPX_2',`FAST_ADVANCE_CPX_1 FAST_ADVANCE_CPX_SRC1')
define(`FAST_ADVANCE_CPX_3',`FAST_ADVANCE_CPX_2 FAST_ADVANCE_CPX_SRC2')
define(`FAST_ADVANCE_CPX_4',`FAST_ADVANCE_CPX_3 FAST_ADVANCE_CPX_SRC3')
define(`FAST_ADVANCE_CPX_5',`FAST_ADVANCE_CPX_4 FAST_ADVANCE_CPX_SRC4')
define(`FAST_ADVANCE_CCR_3',`FAST_ADVANCE_CPX_2 FAST_ADVANCE_SRC2')
define(`FAST_ADVANCE_CR_2',`FAST_ADVANCE_CPX_1 FAST_ADVANCE_SRC1')
define(`FAST_ADVANCE_RC_2',`FAST_ADVANCE_1 FAST_ADVANCE_CPX_SRC1')

define(`FAST_ADVANCE_QUAT_2',`FAST_ADVANCE_QUAT_1 FAST_ADVANCE_QUAT_SRC1')
define(`FAST_ADVANCE_QUAT_3',`FAST_ADVANCE_QUAT_2 FAST_ADVANCE_QUAT_SRC2')
define(`FAST_ADVANCE_QUAT_4',`FAST_ADVANCE_QUAT_3 FAST_ADVANCE_QUAT_SRC3')
define(`FAST_ADVANCE_QUAT_5',`FAST_ADVANCE_QUAT_4 FAST_ADVANCE_QUAT_SRC4')
define(`FAST_ADVANCE_QQR_3',`FAST_ADVANCE_QUAT_2 FAST_ADVANCE_SRC2')
define(`FAST_ADVANCE_QR_2',`FAST_ADVANCE_QUAT_1 FAST_ADVANCE_SRC1')
define(`FAST_ADVANCE_RQ_2',`FAST_ADVANCE_1 FAST_ADVANCE_QUAT_SRC1')

define(`FAST_ADVANCE_SBM_1',`FAST_ADVANCE_SBM FAST_ADVANCE_1')
define(`FAST_ADVANCE_SBM_2',`FAST_ADVANCE_SBM FAST_ADVANCE_2')
define(`FAST_ADVANCE_SBM_3',`FAST_ADVANCE_SBM FAST_ADVANCE_3')
define(`FAST_ADVANCE_SBM_CPX_1',`FAST_ADVANCE_SBM FAST_ADVANCE_CPX_1')
define(`FAST_ADVANCE_SBM_CPX_2',`FAST_ADVANCE_SBM FAST_ADVANCE_CPX_2')
define(`FAST_ADVANCE_SBM_CPX_3',`FAST_ADVANCE_SBM FAST_ADVANCE_CPX_3')
define(`FAST_ADVANCE_SBM_QUAT_1',`FAST_ADVANCE_SBM FAST_ADVANCE_QUAT_1')
define(`FAST_ADVANCE_SBM_QUAT_2',`FAST_ADVANCE_SBM FAST_ADVANCE_QUAT_2')
define(`FAST_ADVANCE_SBM_QUAT_3',`FAST_ADVANCE_SBM FAST_ADVANCE_QUAT_3')

define(`FAST_ADVANCE_DBM_',`FAST_ADVANCE_DBM')
define(`FAST_ADVANCE_DBM_1SRC',`FAST_ADVANCE_DBM FAST_ADVANCE_SRC1')
define(`FAST_ADVANCE_DBM_SBM',`FAST_ADVANCE_DBM FAST_ADVANCE_SBM')
define(`FAST_ADVANCE_DBM_2SRCS',`FAST_ADVANCE_DBM FAST_ADVANCE_2SRCS')
define(`FAST_ADVANCE_2SRCS',`FAST_ADVANCE_SRC1 FAST_ADVANCE_SRC2')


dnl	EQSP_ADVANCE(typ,suffix)
define(`EQSP_ADVANCE',`_EQSP_ADVANCE($1$2)')
define(`_EQSP_ADVANCE',EQSP_ADVANCE_$1)

define(`EQSP_ADVANCE_SRC1',`s1_ptr += eqsp_src1_inc ;')
define(`EQSP_ADVANCE_SRC2',`s2_ptr += eqsp_src2_inc ;')
define(`EQSP_ADVANCE_SRC3',`s3_ptr += eqsp_src3_inc ;')
define(`EQSP_ADVANCE_SRC4',`s4_ptr += eqsp_src4_inc ;')
define(`EQSP_ADVANCE_1',`dst_ptr += eqsp_dest_inc ;')
define(`EQSP_ADVANCE_CPX_1',`cdst_ptr += eqsp_dest_inc ;')
define(`EQSP_ADVANCE_CPX_SRC1',`cs1_ptr += eqsp_src1_inc ;')
define(`EQSP_ADVANCE_CPX_SRC2',`cs2_ptr += eqsp_src2_inc ;')
define(`EQSP_ADVANCE_CPX_SRC3',`cs3_ptr += eqsp_src3_inc ;')
define(`EQSP_ADVANCE_CPX_SRC4',`cs4_ptr += eqsp_src4_inc ;')
define(`EQSP_ADVANCE_QUAT_1',`qdst_ptr += eqsp_dest_inc ;')
define(`EQSP_ADVANCE_QUAT_SRC1',`qs1_ptr += eqsp_src1_inc ;')
define(`EQSP_ADVANCE_QUAT_SRC2',`qs2_ptr += eqsp_src2_inc ;')
define(`EQSP_ADVANCE_QUAT_SRC3',`qs3_ptr += eqsp_src3_inc ;')
define(`EQSP_ADVANCE_QUAT_SRC4',`qs4_ptr += eqsp_src4_inc ;')
dnl #define EQSP_ADVANCE_BITMAP	which_bit  += eqsp_bit_inc ;
define(`EQSP_ADVANCE_DBM',`dbm_bit  += eqsp_dbm_inc ;')
define(`EQSP_ADVANCE_SBM',`sbm_bit  += eqsp_sbm_inc ;')
define(`EQSP_ADVANCE_DBM_SBM',`EQSP_ADVANCE_DBM EQSP_ADVANCE_SBM')

define(`EQSP_ADVANCE_2',`EQSP_ADVANCE_1 EQSP_ADVANCE_SRC1')
define(`EQSP_ADVANCE_3',`EQSP_ADVANCE_2 EQSP_ADVANCE_SRC2')
define(`EQSP_ADVANCE_4',`EQSP_ADVANCE_3 EQSP_ADVANCE_SRC3')
define(`EQSP_ADVANCE_5',`EQSP_ADVANCE_4 EQSP_ADVANCE_SRC4')

define(`EQSP_ADVANCE_CPX_2',`EQSP_ADVANCE_CPX_1 EQSP_ADVANCE_CPX_SRC1')
define(`EQSP_ADVANCE_CPX_3',`EQSP_ADVANCE_CPX_2 EQSP_ADVANCE_CPX_SRC2')
define(`EQSP_ADVANCE_CPX_4',`EQSP_ADVANCE_CPX_3 EQSP_ADVANCE_CPX_SRC3')
define(`EQSP_ADVANCE_CPX_5',`EQSP_ADVANCE_CPX_4 EQSP_ADVANCE_CPX_SRC4')
define(`EQSP_ADVANCE_CCR_3',`EQSP_ADVANCE_CPX_2 EQSP_ADVANCE_SRC2')
define(`EQSP_ADVANCE_CR_2',`EQSP_ADVANCE_CPX_1 EQSP_ADVANCE_SRC1')
define(`EQSP_ADVANCE_RC_2',`EQSP_ADVANCE_1 EQSP_ADVANCE_CPX_SRC1')

define(`EQSP_ADVANCE_QUAT_2',`EQSP_ADVANCE_QUAT_1 EQSP_ADVANCE_QUAT_SRC1')
define(`EQSP_ADVANCE_QUAT_3',`EQSP_ADVANCE_QUAT_2 EQSP_ADVANCE_QUAT_SRC2')
define(`EQSP_ADVANCE_QUAT_4',`EQSP_ADVANCE_QUAT_3 EQSP_ADVANCE_QUAT_SRC3')
define(`EQSP_ADVANCE_QUAT_5',`EQSP_ADVANCE_QUAT_4 EQSP_ADVANCE_QUAT_SRC4')
define(`EQSP_ADVANCE_QQR_3',`EQSP_ADVANCE_QUAT_2 EQSP_ADVANCE_SRC2')
define(`EQSP_ADVANCE_QR_2',`EQSP_ADVANCE_QUAT_1 EQSP_ADVANCE_SRC1')
define(`EQSP_ADVANCE_RQ_2',`EQSP_ADVANCE_1 EQSP_ADVANCE_QUAT_SRC1')

define(`EQSP_ADVANCE_SBM_1',`EQSP_ADVANCE_SBM EQSP_ADVANCE_1')
define(`EQSP_ADVANCE_SBM_2',`EQSP_ADVANCE_SBM EQSP_ADVANCE_2')
define(`EQSP_ADVANCE_SBM_3',`EQSP_ADVANCE_SBM EQSP_ADVANCE_3')
define(`EQSP_ADVANCE_SBM_CPX_1',`EQSP_ADVANCE_SBM EQSP_ADVANCE_CPX_1')
define(`EQSP_ADVANCE_SBM_CPX_2',`EQSP_ADVANCE_SBM EQSP_ADVANCE_CPX_2')
define(`EQSP_ADVANCE_SBM_CPX_3',`EQSP_ADVANCE_SBM EQSP_ADVANCE_CPX_3')
define(`EQSP_ADVANCE_SBM_QUAT_1',`EQSP_ADVANCE_SBM EQSP_ADVANCE_QUAT_1')
define(`EQSP_ADVANCE_SBM_QUAT_2',`EQSP_ADVANCE_SBM EQSP_ADVANCE_QUAT_2')
define(`EQSP_ADVANCE_SBM_QUAT_3',`EQSP_ADVANCE_SBM EQSP_ADVANCE_QUAT_3')

define(`EQSP_ADVANCE_DBM_',`EQSP_ADVANCE_DBM')
define(`EQSP_ADVANCE_DBM_1SRC',`EQSP_ADVANCE_DBM EQSP_ADVANCE_SRC1')
define(`EQSP_ADVANCE_DBM_SBM',`EQSP_ADVANCE_DBM EQSP_ADVANCE_SBM')
define(`EQSP_ADVANCE_DBM_2SRCS',`EQSP_ADVANCE_DBM EQSP_ADVANCE_2SRCS')
define(`EQSP_ADVANCE_2SRCS',`EQSP_ADVANCE_SRC1 EQSP_ADVANCE_SRC2')


dnl	EXTRA_DECLS(extra_suffix)
define(`EXTRA_DECLS',EXTRA_DECLS_$1)
define(`EXTRA_DECLS_',`')	dnl /* nothing */
define(`EXTRA_DECLS_T1',`dest_type r,theta,arg;')
define(`EXTRA_DECLS_T2',`dest_cpx tmpc;')
define(`EXTRA_DECLS_T3',`dest_cpx tmpc; dest_type tmp_denom;')
define(`EXTRA_DECLS_T4',`dest_quat tmpq;')
define(`EXTRA_DECLS_T5',`dest_quat tmpq; dest_type tmp_denom;')

/* Stuff for projection loops */

/* was:
 *	count_type loop_count[N_DIMENSIONS];
 */

dnl  BUG - why are we allocating with NEW_DIMSET?

define(`DECLARE_LOOP_COUNT',`
	int i_dim;
	/*Dimension_Set *loop_count=NEW_DIMSET;*/
	Dimension_Set lc_ds, *loop_count=(&lc_ds);
')

dnl	PROJ_LOOP_DECLS(typ,vectors)
define(`PROJ_LOOP_DECLS',`_PROJ_LOOP_DECLS($1$2)')
define(`_PROJ_LOOP_DECLS',PROJ_LOOP_DECLS_$1)

define(`PROJ_LOOP_DECLS_2',`
DECLARE_BASES_2
DECLARE_LOOP_COUNT
')

define(`PROJ_LOOP_DECLS_CPX_2',`
DECLARE_BASES_CPX_2
DECLARE_LOOP_COUNT
')

define(`PROJ_LOOP_DECLS_QUAT_2',`
DECLARE_BASES_QUAT_2
DECLARE_LOOP_COUNT
')

define(`PROJ_LOOP_DECLS_IDX_2',`
DECLARE_BASES_IDX_2
DECLARE_LOOP_COUNT
	std_type *tmp_ptr;
	std_type *orig_s1_ptr;
	index_type this_index;
')

define(`PROJ_LOOP_DECLS_3',`
DECLARE_BASES_3
DECLARE_LOOP_COUNT
')

define(`PROJ_LOOP_DECLS_CPX_3',`
DECLARE_BASES_CPX_3
DECLARE_LOOP_COUNT
')

define(`PROJ_LOOP_DECLS_QUAT_3',`
DECLARE_BASES_QUAT_3
DECLARE_LOOP_COUNT
')

define(`PROJ_LOOP_DECLS_IDX_3',`
DECLARE_BASES_IDX_3
DECLARE_LOOP_COUNT
	std_type *tmp_ptr;
	index_type this_index;
')


define(`INIT_LOOP_COUNT',`
	for(i_dim=0;i_dim<N_DIMENSIONS;i_dim++) ASSIGN_IDX_COUNT(loop_count,i_dim,1);
')


dnl	INC_BASE(which_dim,base_array,inc_array)
define(`INC_BASE',`

	$2[$1-1] += IDX_INC($3,$1);
')

dnl	INC_BASES(bitmap,typ,suffix)
define(`INC_BASES',`_INC_BASES($1$2$3)')
define(`_INC_BASES',INC_BASES_$1)

dnl	INC_BASES_1(which_dim)
define(`INC_BASES_1',`INC_BASE($1,dst_base,dinc)')
define(`INC_BASES_SRC1',`INC_BASE($1,s1_base,s1inc)')
define(`INC_BASES_SRC2',`INC_BASE($1,s2_base,s2inc)')
define(`INC_BASES_SRC3',`INC_BASE($1,s3_base,s3inc)')
define(`INC_BASES_SRC4',`INC_BASE($1,s4_base,s4inc)')

define(`INC_BASES_CPX_1',`INC_BASE($1,cdst_base,dinc)')
define(`INC_BASES_CPX_SRC1',`INC_BASE($1,cs1_base,s1inc)')
define(`INC_BASES_CPX_SRC2',`INC_BASE($1,cs2_base,s2inc)')
define(`INC_BASES_CPX_SRC3',`INC_BASE($1,cs3_base,s3inc)')
define(`INC_BASES_CPX_SRC4',`INC_BASE($1,cs4_base,s4inc)')

define(`INC_BASES_QUAT_1',`INC_BASE($1,qdst_base,dinc)')
define(`INC_BASES_QUAT_SRC1',`INC_BASE($1,qs1_base,s1inc)')
define(`INC_BASES_QUAT_SRC2',`INC_BASE($1,qs2_base,s2inc)')
define(`INC_BASES_QUAT_SRC3',`INC_BASE($1,qs3_base,s3inc)')
define(`INC_BASES_QUAT_SRC4',`INC_BASE($1,qs4_base,s4inc)')

define(`INC_BASES_SBM',`INC_BASE($1,sbm_base,sbminc)')
define(`INC_BASES_DBM',`INC_BASE($1,dbm_base,dbminc)')
define(`INC_BASES_DBM_',`INC_BASES_DBM($1)')

define(`INC_BASES_2',`INC_BASES_1($1) INC_BASES_SRC1($1)')
define(`INC_BASES_3',`/* inc_bases_3 /$1/ */ INC_BASES_2($1) INC_BASES_SRC2($1)')
define(`INC_BASES_4',`INC_BASES_3($1) INC_BASES_SRC3($1)')
define(`INC_BASES_5',`INC_BASES_4($1) INC_BASES_SRC4($1)')

define(`INC_BASES_CPX_2',`INC_BASES_CPX_1($1) INC_BASES_CPX_SRC1($1)')
define(`INC_BASES_CPX_3',`INC_BASES_CPX_2($1) INC_BASES_CPX_SRC2($1)')
define(`INC_BASES_CPX_4',`INC_BASES_CPX_3($1) INC_BASES_CPX_SRC3($1)')
define(`INC_BASES_CPX_5',`INC_BASES_CPX_4($1) INC_BASES_CPX_SRC4($1)')
define(`INC_BASES_CCR_3',`INC_BASES_CPX_2($1) INC_BASES_SRC2($1)')
define(`INC_BASES_CR_2',`INC_BASES_CPX_1($1) INC_BASES_SRC1($1)')
define(`INC_BASES_RC_2',`INC_BASES_1($1) INC_BASES_CPX_SRC1($1)')

define(`INC_BASES_QUAT_2',`INC_BASES_QUAT_1($1) INC_BASES_QUAT_SRC1($1)')
define(`INC_BASES_QUAT_3',`INC_BASES_QUAT_2($1) INC_BASES_QUAT_SRC2($1)')
define(`INC_BASES_QUAT_4',`INC_BASES_QUAT_3($1) INC_BASES_QUAT_SRC3($1)')
define(`INC_BASES_QUAT_5',`INC_BASES_QUAT_4($1) INC_BASES_QUAT_SRC4($1)')
define(`INC_BASES_QQR_3',`INC_BASES_QUAT_2($1) INC_BASES_SRC2($1)')
define(`INC_BASES_QR_2',`INC_BASES_QUAT_1($1) INC_BASES_SRC1($1)')
define(`INC_BASES_RQ_2',`INC_BASES_1($1) INC_BASES_QUAT_SRC1($1)')

define(`INC_BASES_IDX_2',`INC_BASES_2($1) INC_BASES_IDX($1)')
define(`INC_BASES_IDX',`index_base[$1-1] += INDEX_COUNT(s1_count,$1-1);')

define(`INC_BASES_X_2',`INC_BASES_2')

dnl	INC_XXX_BASE(which_dim,base_array,inc_array,count_array)
define(`INC_XXX_BASE',`
	$2[$1-1] += $3[$1] / $4[0];
')

dnl	INC_XXX_DST_BASE(which_dim)
define(`INC_XXX_DST_BASE',`INC_XXX_BASE($1,dst_base,dinc,count)')
define(`INC_XXX_SRC1_BASE',`INC_XXX_BASE($1,s1_base,s1inc,s1_count)')
define(`INC_XXX_SRC2_BASE',`INC_XXX_BASE($1,s2_base,s2inc,s2_count)')
define(`INC_XXX_SRC3_BASE',`INC_XXX_BASE($1,s3_base,s3inc,s3_count)')
define(`INC_XXX_SRC4_BASE',`INC_XXX_BASE($1,s4_base,s4inc,s4_count)')

define(`INC_BASES_XXX_1',`INC_XXX_DST_BASE($1)')
define(`INC_BASES_XXX_2',`INC_BASES_XXX_1($1) INC_XXX_SRC1_BASE($1)')

define(`INC_BASES_XXX_3',`INC_BASES_XXX_2($1) INC_XXX_SRC2_BASE($1)')


dnl	INC_BASES_2SRCS(index)
define(`INC_BASES_2SRCS',`INC_BASES_SRC1($1) INC_BASES_SRC2($1)')
define(`INC_BASES_DBM_1SRC',`INC_BASES_DBM($1) INC_BASES_SRC1($1)')
define(`INC_BASES_DBM_SBM',`INC_BASES_DBM($1) INC_BASES_SBM($1)')
define(`INC_BASES_DBM_2SRCS',`INC_BASES_DBM($1) INC_BASES_2SRCS($1)')

dnl	COPY_BASES(bitmap,typ,suffix)
define(`COPY_BASES',`/* copy_bases /$1/ /$2/ */_COPY_BASES($1$2$3)')
define(`_COPY_BASES',COPY_BASES_$1)

define(`COPY_BASES_DBM_1SRC',`COPY_BASES_DBM($1) COPY_BASES_SRC1($1)')
define(`COPY_BASES_DBM_SBM',`COPY_BASES_DBM($1) COPY_BASES_SBM($1)')
define(`COPY_BASES_2SRCS',`COPY_BASES_SRC1($1) COPY_BASES_SRC2($1)')
define(`COPY_BASES_DBM_2SRCS',`COPY_BASES_DBM($1) COPY_BASES_2SRCS($1)')

define(`COPY_BASES_DBM_',`COPY_BASES_DBM($1)')


define(`INC_BASES_SBM_1',`INC_BASES_1($1) INC_BASES_SBM($1)')
define(`INC_BASES_SBM_2',`INC_BASES_2($1) INC_BASES_SBM($1)')
define(`INC_BASES_SBM_3',`INC_BASES_3($1) INC_BASES_SBM($1)')
define(`INC_BASES_SBM_CPX_1',`INC_BASES_CPX_1($1) INC_BASES_SBM($1)')
define(`INC_BASES_SBM_CPX_2',`INC_BASES_CPX_2($1) INC_BASES_SBM($1)')
define(`INC_BASES_SBM_CPX_3',`INC_BASES_CPX_3($1) INC_BASES_SBM($1)')
define(`INC_BASES_SBM_QUAT_1',`INC_BASES_QUAT_1($1) INC_BASES_SBM($1)')
define(`INC_BASES_SBM_QUAT_2',`INC_BASES_QUAT_2($1) INC_BASES_SBM($1)')
define(`INC_BASES_SBM_QUAT_3',`INC_BASES_QUAT_3($1) INC_BASES_SBM($1)')
define(`INC_BASES_SBM_XXX_1',`INC_BASES_XXX_1($1) INC_BASES_SBM($1)')
define(`INC_BASES_SBM_XXX_2',`INC_BASES_XXX_2($1) INC_BASES_SBM($1)')
define(`INC_BASES_SBM_XXX_3',`INC_BASES_XXX_3($1) INC_BASES_SBM($1)')


dnl	INIT_BASE( type, which_dim, base_array, dp )
define(`INIT_BASE',`

	if( ($2) < 4 )
		($3)[$2-1] = ($3)[$2];
	else
		($3)[$2-1] = ($1 *)OBJ_DATA_PTR($4);
')


dnl	INIT_INDEX_BASE( which_dim )
define(`INIT_INDEX_BASE',`

	if( ($1) < 4 )
		index_base[$1-1] = index_base[$1];
	else
		index_base[$1-1] = 0;
')



/* #define INIT_CPX_INDEX_3 */		/* what is this for? */

dnl	INIT_PTRS(bitmap,typ,suffix)
define(`INIT_PTRS',`_INIT_PTRS($1$2$3)')
define(`_INIT_PTRS',INIT_PTRS_$1)


define(`INIT_PTRS_1',`dst_ptr = dst_base[0];')
define(`INIT_PTRS_SRC1',`s1_ptr = s1_base[0];')
define(`INIT_PTRS_SRC2',`s2_ptr = s2_base[0];')
define(`INIT_PTRS_SRC3',`s3_ptr = s3_base[0];')
define(`INIT_PTRS_SRC4',`s4_ptr = s4_base[0];')

define(`INIT_PTRS_CPX_1',`cdst_ptr = cdst_base[0];')
define(`INIT_PTRS_CPX_SRC1',`cs1_ptr = cs1_base[0];')
define(`INIT_PTRS_CPX_SRC2',`cs2_ptr = cs2_base[0];')
define(`INIT_PTRS_CPX_SRC3',`cs3_ptr = cs3_base[0];')
define(`INIT_PTRS_CPX_SRC4',`cs4_ptr = cs4_base[0];')

define(`INIT_PTRS_QUAT_1',`qdst_ptr = qdst_base[0];')
define(`INIT_PTRS_QUAT_SRC1',`qs1_ptr = qs1_base[0];')
define(`INIT_PTRS_QUAT_SRC2',`qs2_ptr = qs2_base[0];')
define(`INIT_PTRS_QUAT_SRC3',`qs3_ptr = qs3_base[0];')
define(`INIT_PTRS_QUAT_SRC4',`qs4_ptr = qs4_base[0];')

define(`INIT_PTRS_SBM',`sbm_bit = sbm_base[0];')
define(`INIT_PTRS_DBM',`dbm_bit = dbm_base[0];')
define(`INIT_PTRS_DBM_',`INIT_PTRS_DBM')

define(`INIT_PTRS_DBM_1SRC',`INIT_PTRS_DBM INIT_PTRS_SRC1')
define(`INIT_PTRS_DBM_SBM',`INIT_PTRS_DBM INIT_PTRS_SBM')
define(`INIT_PTRS_2SRCS',`INIT_PTRS_SRC1 INIT_PTRS_SRC2')
define(`INIT_PTRS_DBM_2SRCS',`INIT_PTRS_DBM INIT_PTRS_2SRCS')

define(`INIT_PTRS_SBM_1',`INIT_PTRS_1 INIT_PTRS_SBM')
define(`INIT_PTRS_SBM_2',`INIT_PTRS_2 INIT_PTRS_SBM')
define(`INIT_PTRS_SBM_3',`INIT_PTRS_3 INIT_PTRS_SBM')
define(`INIT_PTRS_SBM_CPX_1',`INIT_PTRS_CPX_1 INIT_PTRS_SBM')
define(`INIT_PTRS_SBM_CPX_2',`INIT_PTRS_CPX_2 INIT_PTRS_SBM')
define(`INIT_PTRS_SBM_CPX_3',`INIT_PTRS_CPX_3 INIT_PTRS_SBM')
define(`INIT_PTRS_SBM_QUAT_1',`INIT_PTRS_QUAT_1 INIT_PTRS_SBM')
define(`INIT_PTRS_SBM_QUAT_2',`INIT_PTRS_QUAT_2 INIT_PTRS_SBM')
define(`INIT_PTRS_SBM_QUAT_3',`INIT_PTRS_QUAT_3 INIT_PTRS_SBM')

define(`INIT_PTRS_2',`INIT_PTRS_1 INIT_PTRS_SRC1')
define(`INIT_PTRS_3',`INIT_PTRS_2 INIT_PTRS_SRC2')
define(`INIT_PTRS_4',`INIT_PTRS_3 INIT_PTRS_SRC3')
define(`INIT_PTRS_5',`INIT_PTRS_4 INIT_PTRS_SRC4')

define(`INIT_PTRS_CPX_2',`INIT_PTRS_CPX_1 INIT_PTRS_CPX_SRC1')
define(`INIT_PTRS_CPX_3',`INIT_PTRS_CPX_2 INIT_PTRS_CPX_SRC2')
define(`INIT_PTRS_CPX_4',`INIT_PTRS_CPX_3 INIT_PTRS_CPX_SRC3')
define(`INIT_PTRS_CPX_5',`INIT_PTRS_CPX_4 INIT_PTRS_CPX_SRC4')
define(`INIT_PTRS_CCR_3',`INIT_PTRS_CPX_2 INIT_PTRS_SRC2')
define(`INIT_PTRS_CR_2',`INIT_PTRS_CPX_1 INIT_PTRS_SRC1')
define(`INIT_PTRS_RC_2',`INIT_PTRS_1 INIT_PTRS_CPX_SRC1')

define(`INIT_PTRS_QUAT_2',`INIT_PTRS_QUAT_1 INIT_PTRS_QUAT_SRC1')
define(`INIT_PTRS_QUAT_3',`INIT_PTRS_QUAT_2 INIT_PTRS_QUAT_SRC2')
define(`INIT_PTRS_QUAT_4',`INIT_PTRS_QUAT_3 INIT_PTRS_QUAT_SRC3')
define(`INIT_PTRS_QUAT_5',`INIT_PTRS_QUAT_4 INIT_PTRS_QUAT_SRC4')
define(`INIT_PTRS_QQR_3',`INIT_PTRS_QUAT_2 INIT_PTRS_SRC2')
define(`INIT_PTRS_QR_2',`INIT_PTRS_QUAT_1 INIT_PTRS_SRC1')
define(`INIT_PTRS_RQ_2',`INIT_PTRS_1 INIT_PTRS_QUAT_SRC1')

define(`INIT_PTRS_IDX_1',`INIT_PTRS_1 this_index = index_base[0];')
define(`INIT_PTRS_IDX_2',`INIT_PTRS_IDX_1 INIT_PTRS_SRC1')


define(`INIT_PTRS_XXX_1',`INIT_PTRS_1')
define(`INIT_PTRS_XXX_2',`INIT_PTRS_2')
define(`INIT_PTRS_X_2',`INIT_PTRS_2')
define(`INIT_PTRS_XXX_3',`INIT_PTRS_3')
define(`INIT_PTRS_SBM_XXX_3',`INIT_PTRS_SBM_3')
define(`INIT_PTRS_SBM_XXX_2',`INIT_PTRS_SBM_2')
define(`INIT_PTRS_SBM_XXX_1',`INIT_PTRS_SBM_1')

dnl	INC_PTRS(typ,suffix)
define(`INC_PTRS',`_INC_PTRS($1$2)')
define(`_INC_PTRS',INC_PTRS_$1)

define(`INC_PTRS_SRC1',`s1_ptr += IDX_INC(s1inc,0);')
define(`INC_PTRS_SRC2',`s2_ptr += IDX_INC(s2inc,0);')
define(`INC_PTRS_SRC3',`s3_ptr += IDX_INC(s3inc,0);')
define(`INC_PTRS_SRC4',`s4_ptr += IDX_INC(s4inc,0);')
/* BUG? here we seem to assume that all bitmaps are contiguous - but
 * dobj allows bitmap subimages...
 */
define(`INC_PTRS_SBM',`sbm_bit++;')
define(`INC_PTRS_DBM',`dbm_bit++;')
define(`INC_PTRS_DBM_',`INC_PTRS_DBM')

define(`INC_PTRS_DBM_1SRC',`INC_PTRS_DBM INC_PTRS_SRC1')
define(`INC_PTRS_DBM_SBM',`INC_PTRS_DBM INC_PTRS_SBM')
define(`INC_PTRS_2SRCS',`INC_PTRS_SRC1 INC_PTRS_SRC2')
define(`INC_PTRS_DBM_2SRCS',`INC_PTRS_DBM INC_PTRS_2SRCS')

define(`INC_PTRS_SBM_1',`INC_PTRS_1 INC_PTRS_SBM')
define(`INC_PTRS_SBM_2',`INC_PTRS_2 INC_PTRS_SBM')
define(`INC_PTRS_SBM_3',`INC_PTRS_3 INC_PTRS_SBM')

dnl	define(`INC_PTRS_SBM_CPX_1',`INC_PTRS_CPX_1 INC_PTRS_SBM
dnl	define(`INC_PTRS_SBM_CPX_2',`INC_PTRS_CPX_2 INC_PTRS_SBM
dnl	define(`INC_PTRS_SBM_CPX_3',`INC_PTRS_CPX_3 INC_PTRS_SBM
dnl	define(`INC_PTRS_SBM_QUAT_1',`INC_PTRS_QUAT_1 INC_PTRS_SBM
dnl	define(`INC_PTRS_SBM_QUAT_2',`INC_PTRS_QUAT_2 INC_PTRS_SBM
dnl	define(`INC_PTRS_SBM_QUAT_3',`INC_PTRS_QUAT_3 INC_PTRS_SBM

define(`INC_PTRS_1',`dst_ptr += IDX_INC(dinc,0);')
define(`INC_PTRS_2',`INC_PTRS_1 INC_PTRS_SRC1')
define(`INC_PTRS_3',`INC_PTRS_2 INC_PTRS_SRC2')
define(`INC_PTRS_4',`INC_PTRS_3 INC_PTRS_SRC3')
define(`INC_PTRS_5',`INC_PTRS_4 INC_PTRS_SRC4')

define(`INC_PTRS_IDX_2',`INC_PTRS_2 this_index++;')
define(`INC_PTRS_X_2',`INC_PTRS_2')

/* compiler BUG?
 * These macros were originally written with left shifts, but
 * even when a u_long is 64 bits, we cannot shift left by more than 31!?
 * SOLVED - 1<<n assumes that 1 is an "int" e.g. 32 bits
 * Use 1L instead!
 *
 * dbm_bit counts the bits from the start of the object
 */

dnl	SET_DBM_BIT( condition )
define(`SET_DBM_BIT',`

	if( $1 )
		*(dbm_ptr + (dbm_bit/BITS_PER_BITMAP_WORD)) |=
			NUMBERED_BIT(dbm_bit); 
	else
		*(dbm_ptr + (dbm_bit/BITS_PER_BITMAP_WORD)) &=
			~ NUMBERED_BIT(dbm_bit);
')

define(`DEBUG_SBM_',`
sprintf(DEFAULT_ERROR_STRING,"sbm_ptr = 0x%lx   sbm_bit = %d",
(int_for_addr)sbm_ptr,sbm_bit);
NADVISE(DEFAULT_ERROR_STRING);
')

define(`DEBUG_DBM_',`
sprintf(DEFAULT_ERROR_STRING,"dbm_ptr = 0x%lx   dbm_bit = %d",
(int_for_addr)dbm_ptr,dbm_bit);
NADVISE(DEFAULT_ERROR_STRING);
')

define(`DEBUG_DBM_1SRC',`DEBUG_DBM_ DEBUG_SRC1')

define(`srcbit',`((*(sbm_ptr + (sbm_bit/BITS_PER_BITMAP_WORD))) & NUMBERED_BIT(sbm_bit))')


dnl	INIT_BASES(bitmap,typ,suffix)
define(`INIT_BASES',`_INIT_BASES($1$2$3)')
define(`_INIT_BASES',INIT_BASES_$1)

dnl	INIT_BASES_X_1(dsttyp)
define(`INIT_BASES_X_1',`dst_base[3]=($1 *)VA_DEST_PTR(vap);')

dnl	INIT_BASES_X_SRC1(srctyp)
define(`INIT_BASES_X_SRC1',`s1_base[3]=($1 *)VA_SRC_PTR(vap,0);')

dnl	INIT_BASES_CONV_1(type)
define(`INIT_BASES_CONV_1',`dst_base[3]=($1 *)VA_DEST_PTR(vap);')
define(`INIT_BASES_1',`dst_base[3]=(dest_type *)VA_DEST_PTR(vap);')
define(`INIT_BASES_SRC1',`s1_base[3]=(std_type *)VA_SRC_PTR(vap,0);')
define(`INIT_BASES_SRC2',`s2_base[3]=(std_type *)VA_SRC_PTR(vap,1);')
define(`INIT_BASES_SRC3',`s3_base[3]=(std_type *)VA_SRC_PTR(vap,2);')
define(`INIT_BASES_SRC4',`s4_base[3]=(std_type *)VA_SRC_PTR(vap,3);')

define(`INIT_BASES_CPX_1',`cdst_base[3]=(dest_cpx *)VA_DEST_PTR(vap);')
define(`INIT_BASES_CPX_SRC1',`cs1_base[3]=(std_cpx *)VA_SRC_PTR(vap,0);')
define(`INIT_BASES_CPX_SRC2',`cs2_base[3]=(std_cpx *)VA_SRC_PTR(vap,1);')
define(`INIT_BASES_CPX_SRC3',`cs3_base[3]=(std_cpx *)VA_SRC_PTR(vap,2);')
define(`INIT_BASES_CPX_SRC4',`cs4_base[3]=(std_cpx *)VA_SRC_PTR(vap,3);')

define(`INIT_BASES_QUAT_1',`qdst_base[3]=(dest_quat *)VA_DEST_PTR(vap);')
define(`INIT_BASES_QUAT_SRC1',`qs1_base[3]=(std_quat *)VA_SRC_PTR(vap,0);')
define(`INIT_BASES_QUAT_SRC2',`qs2_base[3]=(std_quat *)VA_SRC_PTR(vap,1);')
define(`INIT_BASES_QUAT_SRC3',`qs3_base[3]=(std_quat *)VA_SRC_PTR(vap,2);')
define(`INIT_BASES_QUAT_SRC4',`qs4_base[3]=(std_quat *)VA_SRC_PTR(vap,3);')

define(`INIT_BASES_IDX_1',`INIT_BASES_X_1(index_type)  index_base[3]=0;')
define(`INIT_BASES_IDX_2',`INIT_BASES_IDX_1 INIT_BASES_SRC1')


dnl	INIT_BASES_X_2(dsttyp,srctyp)
define(`INIT_BASES_X_2',`INIT_BASES_X_1($1) INIT_BASES_X_SRC1($2)')

dnl	INIT_BASES_CONV_2(type)
define(`INIT_BASES_CONV_2',`INIT_BASES_CONV_1($1) INIT_BASES_SRC1')

define(`INIT_BASES_2',`INIT_BASES_1 INIT_BASES_SRC1')

define(`INIT_BASES_3',`INIT_BASES_2 INIT_BASES_SRC2')
define(`INIT_BASES_4',`INIT_BASES_3 INIT_BASES_SRC3')
define(`INIT_BASES_5',`INIT_BASES_4 INIT_BASES_SRC4')

define(`INIT_BASES_CPX_2',`INIT_BASES_CPX_1 INIT_BASES_CPX_SRC1')
define(`INIT_BASES_CPX_3',`INIT_BASES_CPX_2 INIT_BASES_CPX_SRC2')
define(`INIT_BASES_CPX_4',`INIT_BASES_CPX_3 INIT_BASES_CPX_SRC3')
define(`INIT_BASES_CPX_5',`INIT_BASES_CPX_4 INIT_BASES_CPX_SRC4')
define(`INIT_BASES_CCR_3',`INIT_BASES_CPX_2 INIT_BASES_SRC2')
define(`INIT_BASES_CR_2',`INIT_BASES_CPX_1 INIT_BASES_SRC1')
define(`INIT_BASES_RC_2',`INIT_BASES_1 INIT_BASES_CPX_SRC1')

define(`INIT_BASES_QUAT_2',`INIT_BASES_QUAT_1 INIT_BASES_QUAT_SRC1')
define(`INIT_BASES_QUAT_3',`INIT_BASES_QUAT_2 INIT_BASES_QUAT_SRC2')
define(`INIT_BASES_QUAT_4',`INIT_BASES_QUAT_3 INIT_BASES_QUAT_SRC3')
define(`INIT_BASES_QUAT_5',`INIT_BASES_QUAT_4 INIT_BASES_QUAT_SRC4')
define(`INIT_BASES_QQR_3',`INIT_BASES_QUAT_2 INIT_BASES_SRC2')
define(`INIT_BASES_QR_2',`INIT_BASES_QUAT_1 INIT_BASES_SRC1')
define(`INIT_BASES_RQ_2',`INIT_BASES_1 INIT_BASES_QUAT_SRC1')

define(`INIT_BASES_DBM_',`INIT_BASES_DBM')

/* We don't actually use the bases for destination bitmaps...
 * Should we?
 *
 * dbm_base used to be the pointer, but now it is bit0
 *
 * Why are the indices three and not 4?
 */

define(`INIT_BASES_DBM',`
	dbm_ptr= VA_DEST_PTR(vap);
	dbm_base[3]=VA_DBM_BIT0(vap);
')

define(`INIT_BASES_SBM',`
	sbm_ptr= VA_SRC_PTR(vap,4);
	sbm_base[3]=VA_SBM_BIT0(vap);
')

define(`INIT_BASES_SBM_1',`INIT_BASES_1 INIT_BASES_SBM')
define(`INIT_BASES_SBM_2',`INIT_BASES_2 INIT_BASES_SBM')
define(`INIT_BASES_SBM_3',`INIT_BASES_3 INIT_BASES_SBM')
define(`INIT_BASES_SBM_CPX_1',`INIT_BASES_CPX_1 INIT_BASES_SBM')
define(`INIT_BASES_SBM_CPX_2',`INIT_BASES_CPX_2 INIT_BASES_SBM')
define(`INIT_BASES_SBM_CPX_3',`INIT_BASES_CPX_3 INIT_BASES_SBM')
define(`INIT_BASES_SBM_QUAT_1',`INIT_BASES_QUAT_1 INIT_BASES_SBM')
define(`INIT_BASES_SBM_QUAT_2',`INIT_BASES_QUAT_2 INIT_BASES_SBM')
define(`INIT_BASES_SBM_QUAT_3',`INIT_BASES_QUAT_3 INIT_BASES_SBM')

define(`INIT_BASES_2SRCS',`INIT_BASES_SRC1 INIT_BASES_SRC2')
define(`INIT_BASES_DBM_1SRC',`INIT_BASES_DBM INIT_BASES_SRC1')
define(`INIT_BASES_DBM_SBM',`INIT_BASES_DBM INIT_BASES_SBM')
define(`INIT_BASES_DBM_2',`INIT_BASES_DBM INIT_BASES_2SRCS')
define(`INIT_BASES_DBM_2SRCS',`INIT_BASES_DBM INIT_BASES_2SRCS')

dnl	INIT_COUNT( var, index )
define(`INIT_COUNT',`_INIT_COUNT($1,count,$2)')

dnl #define _INIT_COUNT( var, array, index )
define(`_INIT_COUNT',`$1=INDEX_COUNT($2,$3);')

dnl	COPY_BASES_1(index)
define(`COPY_BASES_1',`dst_base[$1] = dst_base[$1+1];')
define(`COPY_BASES_CPX_1',`cdst_base[$1] = cdst_base[$1+1];')
define(`COPY_BASES_QUAT_1',`qdst_base[$1] = qdst_base[$1+1];')

define(`COPY_BASES_IDX_1',`COPY_BASES_1($1) index_base[$1] = index_base[$1+1];')


define(`COPY_BASES_IDX_2',`COPY_BASES_IDX_1($1) COPY_BASES_SRC1($1)')


define(`COPY_BASES_SRC1',`s1_base[$1] = s1_base[$1+1];')
define(`COPY_BASES_SRC2',`s2_base[$1] = s2_base[$1+1];')
define(`COPY_BASES_SRC3',`s3_base[$1] = s3_base[$1+1];')
define(`COPY_BASES_SRC4',`s4_base[$1] = s4_base[$1+1];')

define(`COPY_BASES_CPX_SRC1',`cs1_base[$1] = cs1_base[$1+1];')
define(`COPY_BASES_CPX_SRC2',`cs2_base[$1] = cs2_base[$1+1];')
define(`COPY_BASES_CPX_SRC3',`cs3_base[$1] = cs3_base[$1+1];')
define(`COPY_BASES_CPX_SRC4',`cs4_base[$1] = cs4_base[$1+1];')

define(`COPY_BASES_QUAT_SRC1',`qs1_base[$1] = qs1_base[$1+1];')
define(`COPY_BASES_QUAT_SRC2',`qs2_base[$1] = qs2_base[$1+1];')
define(`COPY_BASES_QUAT_SRC3',`qs3_base[$1] = qs3_base[$1+1];')
define(`COPY_BASES_QUAT_SRC4',`qs4_base[$1] = qs4_base[$1+1];')

define(`COPY_BASES_2',`COPY_BASES_1($1) COPY_BASES_SRC1($1)')
define(`COPY_BASES_3',`COPY_BASES_2($1) COPY_BASES_SRC2($1)')
define(`COPY_BASES_4',`COPY_BASES_3($1) COPY_BASES_SRC3($1)')
define(`COPY_BASES_5',`COPY_BASES_4($1) COPY_BASES_SRC4($1)')

define(`COPY_BASES_CPX_2',`COPY_BASES_CPX_1($1) COPY_BASES_CPX_SRC1($1)')
define(`COPY_BASES_CPX_3',`COPY_BASES_CPX_2($1) COPY_BASES_CPX_SRC2($1)')
define(`COPY_BASES_CPX_4',`COPY_BASES_CPX_3($1) COPY_BASES_CPX_SRC3($1)')
define(`COPY_BASES_CPX_5',`COPY_BASES_CPX_4($1) COPY_BASES_CPX_SRC4($1)')
define(`COPY_BASES_CCR_3',`COPY_BASES_CPX_2($1) COPY_BASES_SRC2($1)')
define(`COPY_BASES_CR_2',`COPY_BASES_CPX_1($1) COPY_BASES_SRC1($1)')
define(`COPY_BASES_RC_2',`COPY_BASES_1($1) COPY_BASES_CPX_SRC1($1)')

define(`COPY_BASES_QUAT_2',`COPY_BASES_QUAT_1($1) COPY_BASES_QUAT_SRC1($1)')
define(`COPY_BASES_QUAT_3',`COPY_BASES_QUAT_2($1) COPY_BASES_QUAT_SRC2($1)')
define(`COPY_BASES_QUAT_4',`COPY_BASES_QUAT_3($1) COPY_BASES_QUAT_SRC3($1)')
define(`COPY_BASES_QUAT_5',`COPY_BASES_QUAT_4($1) COPY_BASES_QUAT_SRC4($1)')
define(`COPY_BASES_QQR_3',`COPY_BASES_QUAT_2($1) COPY_BASES_SRC2($1)')
define(`COPY_BASES_QR_2',`COPY_BASES_QUAT_1($1) COPY_BASES_SRC1($1)')
define(`COPY_BASES_RQ_2',`COPY_BASES_1($1) COPY_BASES_QUAT_SRC1($1)')

define(`COPY_BASES_SBM_1',`COPY_BASES_1($1) COPY_BASES_SBM($1)')
define(`COPY_BASES_SBM_2',`COPY_BASES_2($1) COPY_BASES_SBM($1)')
define(`COPY_BASES_SBM_3',`COPY_BASES_3($1) COPY_BASES_SBM($1)')
define(`COPY_BASES_SBM_CPX_1',`COPY_BASES_CPX_1($1) COPY_BASES_SBM($1)')
define(`COPY_BASES_SBM_CPX_2',`COPY_BASES_CPX_2($1) COPY_BASES_SBM($1)')
define(`COPY_BASES_SBM_CPX_3',`COPY_BASES_CPX_3($1) COPY_BASES_SBM($1)')
define(`COPY_BASES_SBM_QUAT_1',`COPY_BASES_QUAT_1($1) COPY_BASES_SBM($1)')
define(`COPY_BASES_SBM_QUAT_2',`COPY_BASES_QUAT_2($1) COPY_BASES_SBM($1)')
define(`COPY_BASES_SBM_QUAT_3',`COPY_BASES_QUAT_3($1) COPY_BASES_SBM($1)')


define(`COPY_BASES_DBM',`

	dbm_base[$1] = dbm_base[$1+1];
')

define(`COPY_BASES_SBM',`

	sbm_base[$1] = sbm_base[$1+1];
')

define(`COPY_BASES_DBM_SBM',`
COPY_BASES_DBM($1)
COPY_BASES_SBM($1)
')

define(`COPY_BASES_XXX_1',`COPY_BASES_1')
define(`COPY_BASES_XXX_2',`COPY_BASES_2')
define(`COPY_BASES_XXX_3',`COPY_BASES_3')
define(`COPY_BASES_SBM_XXX_1',`COPY_BASES_SBM_1')
define(`COPY_BASES_SBM_XXX_2',`COPY_BASES_SBM_2')
define(`COPY_BASES_SBM_XXX_3',`COPY_BASES_SBM_3')
define(`COPY_BASES_X_2',`COPY_BASES_2')

dnl	DECLARE_BASES(bitmap,typ,suffix)
define(`DECLARE_BASES',`_DECLARE_BASES($1$2$3)')
define(`_DECLARE_BASES',DECLARE_BASES_$1)

define(`DECLARE_BASES_SBM',`
	int sbm_base[N_DIMENSIONS-1];
	bitmap_word *sbm_ptr;
	int sbm_bit;
')

define(`DECLARE_BASES_DBM_',`DECLARE_BASES_DBM')

/* base is not a bit number, not a pointer */

define(`DECLARE_BASES_DBM',`
	int dbm_base[N_DIMENSIONS-1];
	int dbm_bit;
	bitmap_word *dbm_ptr;
DECLARE_FIVE_LOOP_INDICES
')

define(`DECLARE_FOUR_LOOP_INDICES',`
	dimension_t i;
	dimension_t j;
	dimension_t k;
	dimension_t l;
')

define(`DECLARE_FIVE_LOOP_INDICES',`
DECLARE_FOUR_LOOP_INDICES
	dimension_t m;
')

dnl	DECLARE_BASES_CONV_1(type)
define(`DECLARE_BASES_CONV_1',`
	$1 *dst_base[N_DIMENSIONS-1];
	$1 *dst_ptr;
DECLARE_FIVE_LOOP_INDICES
')

define(`DECLARE_BASES_1',`
	dest_type *dst_base[N_DIMENSIONS-1];
	dest_type *dst_ptr;
DECLARE_FIVE_LOOP_INDICES
')

define(`DECLARE_BASES_IDX_1',`
	index_type *dst_base[N_DIMENSIONS-1];
	index_type index_base[N_DIMENSIONS-1];
	index_type *dst_ptr;
DECLARE_FIVE_LOOP_INDICES
')

define(`DECLARE_BASES_CPX_1',`
	dest_cpx *cdst_base[N_DIMENSIONS-1];
	dest_cpx *cdst_ptr;
DECLARE_FOUR_LOOP_INDICES
')

define(`DECLARE_BASES_QUAT_1',`
	dest_quat *qdst_base[N_DIMENSIONS-1];
	dest_quat *qdst_ptr;
DECLARE_FOUR_LOOP_INDICES
')

dnl	DECLARE_X_DST_VBASE(type)
define(`DECLARE_X_DST_VBASE',`
	$1 *dst_base[N_DIMENSIONS-1];
	$1 *dst_ptr;
	DECLARE_FIVE_LOOP_INDICES
')

dnl	DECLARE_XXX_DST_VBASE(type,keychar)
define(`DECLARE_XXX_DST_VBASE',`
	type *$2`dst_base'[N_DIMENSIONS-1];
	type *$2`dst_ptr';
	DECLARE_FOUR_LOOP_INDICES
')

dnl	DECLARE_BASES_CONV_2(type)
define(`DECLARE_BASES_CONV_2',`
	DECLARE_BASES_CONV_1($1)
	DECLARE_VBASE_SRC1
')

define(`DECLARE_BASES_2',`
	DECLARE_BASES_1
	DECLARE_VBASE_SRC1
')

define(`DECLARE_BASES_IDX_2',`
	DECLARE_BASES_IDX_1
	DECLARE_VBASE_SRC1
')

define(`DECLARE_BASES_CPX_2',`
	DECLARE_BASES_CPX_1
	DECLARE_VBASE_CPX_SRC1
')

define(`DECLARE_BASES_QUAT_2',`
	DECLARE_BASES_QUAT_1
	DECLARE_VBASE_QUAT_SRC1
')

dnl	DECLARE_VBASE(typ,prefix)
define(`DECLARE_VBASE',`
	$1 *$2`_base'[N_DIMENSIONS-1];
	$1 *$2`_ptr';
')

dnl	DECLARE_XXX_SRC1_VBASE(type)
define(`DECLARE_XXX_SRC1_VBASE',`DECLARE_VBASE($1,s1)')
define(`DECLARE_XXX_SRC2_VBASE',`DECLARE_VBASE($1,s2)')

define(`DECLARE_VBASE_SRC1',`DECLARE_VBASE(std_type,s1)')
define(`DECLARE_VBASE_CPX_SRC1',`DECLARE_VBASE(std_cpx,cs1)')
define(`DECLARE_VBASE_QUAT_SRC1',`DECLARE_VBASE(std_quat,qs1)')
define(`DECLARE_VBASE_SRC2',`DECLARE_VBASE(std_type,s2)')
define(`DECLARE_VBASE_CPX_SRC2',`DECLARE_VBASE(std_cpx,cs2)')
define(`DECLARE_VBASE_QUAT_SRC2',`DECLARE_VBASE(std_quat,qs2)')

define(`DECLARE_BASES_3',`DECLARE_BASES_2 DECLARE_VBASE_SRC2')

define(`DECLARE_BASES_SBM_1',`DECLARE_BASES_SBM
DECLARE_BASES_1')
define(`DECLARE_BASES_SBM_2',`DECLARE_BASES_SBM
DECLARE_BASES_2')
define(`DECLARE_BASES_SBM_3',`DECLARE_BASES_SBM
DECLARE_BASES_3')

define(`DECLARE_BASES_CPX_T_3',`DECLARE_BASES_CPX_3')

define(`DECLARE_BASES_CPX_3',`DECLARE_BASES_CPX_2
DECLARE_VBASE_CPX_SRC2
')

define(`DECLARE_BASES_QUAT_3',`DECLARE_BASES_QUAT_2
DECLARE_VBASE_QUAT_SRC2
')

define(`DECLARE_BASES_SBM_CPX_1',`DECLARE_BASES_CPX_1
DECLARE_BASES_SBM
')

define(`DECLARE_BASES_SBM_CPX_2',`
	DECLARE_BASES_CPX_2
	DECLARE_BASES_SBM
')

define(`DECLARE_BASES_SBM_CPX_3',`
	DECLARE_BASES_CPX_3
	DECLARE_BASES_SBM
')

define(`DECLARE_BASES_SBM_QUAT_1',`
	DECLARE_BASES_QUAT_1
	DECLARE_BASES_SBM
')

define(`DECLARE_BASES_SBM_QUAT_2',`
	DECLARE_BASES_QUAT_2
	DECLARE_BASES_SBM
')

define(`DECLARE_BASES_SBM_QUAT_3',`
	DECLARE_BASES_QUAT_3
	DECLARE_BASES_SBM
')


define(`DECLARE_BASES_CR_2',`
	DECLARE_BASES_CPX_1
	DECLARE_VBASE_SRC1
')

define(`DECLARE_BASES_QR_2',`
	DECLARE_BASES_QUAT_1
	DECLARE_VBASE_SRC1
')

define(`DECLARE_BASES_CCR_3',`
	DECLARE_BASES_CPX_2
	DECLARE_VBASE_SRC2
')

define(`DECLARE_BASES_QQR_3',`
	DECLARE_BASES_QUAT_2
	DECLARE_VBASE_SRC2
')

define(`DECLARE_BASES_RC_2',`
	dest_type *dst_base[N_DIMENSIONS-1];
	dest_type *dst_ptr;
	DECLARE_VBASE_CPX_SRC1
	DECLARE_FOUR_LOOP_INDICES
')

define(`DECLARE_BASES_4',`
	DECLARE_BASES_3
	std_type *s3_ptr;
	std_type *s3_base[N_DIMENSIONS-1];
')

define(`DECLARE_BASES_5',`
	DECLARE_BASES_4
	std_type *s4_ptr;
	std_type *s4_base[N_DIMENSIONS-1];
')

define(`DECLARE_BASES_DBM_1SRC',`DECLARE_BASES_DBM DECLARE_VBASE_SRC1')
define(`DECLARE_BASES_DBM_SBM',`DECLARE_BASES_DBM DECLARE_BASES_SBM')
define(`DECLARE_BASES_DBM_2SRCS',`DECLARE_BASES_DBM DECLARE_BASES_2SRCS')
define(`DECLARE_BASES_2SRCS',`DECLARE_VBASE_SRC1 DECLARE_VBASE_SRC2')

dnl	DECLARE_BASES_X_2(dsttyp,srctyp)
define(`DECLARE_BASES_X_2',`
DECLARE_X_DST_VBASE($1)
DECLARE_XXX_SRC1_VBASE($2)
')

dnl  for debugging...
define(`INIT_SPACING( spi )',`
	spi.spi_dst_isp = NULL
')

define(`INIT_VEC_ARGS(vap)',`INIT_SPACING( VA_SPACING(vap) );')


dnl  BUG - we should avoid dynamic allocation...
dnl	is this still needed???

define(`RELEASE_VEC_ARGS_STRUCT',`
	/*givbuf( VA_SPACING(vap) );*/
	/*givbuf( VA_SIZE_INFO(vap) );*/
	/*givbuf(vap);*/
')

dnl	DECL_VEC_ARGS_STRUCT(name)
define(`DECL_VEC_ARGS_STRUCT($1)',`
	/*Vector_Args *vap=NEW_VEC_ARGS;*/
	Vector_Args va1, *vap=(&va1);
DECLARE_FUNC_NAME($1)
INIT_VEC_ARGS(vap)
')


dnl  MIN is defined in the iOS Foundation headers...
dnl  We should make sure it is the same!
dnl#ifndef MIN
dnl	MIN( n1, n2 )
define(`MIN',`(($1)<($2)?($1):($2))')
dnl#endif /* undef MIN */

/* The loop_arr array holds the max of all the dimensions - either 1 or N_i.
 * If we encounter a dimension which is not N_i or 1, then it's an error.
 */

dnl	ADJ_COUNTS(loop_arr,obj_arr)
define(`ADJ_COUNTS',`

	for(i_dim=0;i_dim<N_DIMENSIONS;i_dim++){
		if( INDEX_COUNT($2,i_dim) > 1 ){
			if( INDEX_COUNT($1,i_dim) == 1 ){
				ASSIGN_IDX_COUNT($1,i_dim,INDEX_COUNT($2,i_dim));
			} else {
				if( INDEX_COUNT($2,i_dim) != INDEX_COUNT($1,i_dim) ){
					count_type n;

					n = MIN(INDEX_COUNT($1,i_dim),
						INDEX_COUNT($2,i_dim));
					sprintf(DEFAULT_ERROR_STRING,
	"Oops: %s count mismatch, (%d != %d), using %d",
						dimension_name[i_dim],
						INDEX_COUNT($1,i_dim),
						INDEX_COUNT($2,i_dim),n);
					ASSIGN_IDX_COUNT($1,i_dim,n);
				}
			}
		}
	}
')

define(`SHOW_BASES',`
sprintf(DEFAULT_ERROR_STRING,"s1_ptr:  0x%lx",(int_for_addr)s1_ptr);
NADVISE(DEFAULT_ERROR_STRING);
/*sprintf(DEFAULT_ERROR_STRING,"bm_ptr:  0x%lx, which_bit = %d",(int_for_addr)bm_ptr,which_bit);
NADVISE(DEFAULT_ERROR_STRING);*/
sprintf(DEFAULT_ERROR_STRING,"s1_base:  0x%lx  0x%lx  0x%lx  0x%lx",(int_for_addr)s1_base[0],(int_for_addr)s1_base[1],(int_for_addr)s1_base[2],(int_for_addr)s1_base[3]);
NADVISE(DEFAULT_ERROR_STRING);
')



dnl	CONV_METHOD_DECL(name)
define(`CONV_METHOD_DECL',`
static void CONV_METHOD_NAME($1)( Vec_Obj_Args *oap )
')


include(`../../include/veclib/fast_test.m4')


define(`dst',`(*dst_ptr)')
define(`src1',`(*s1_ptr)')
define(`src2',`(*s2_ptr)')
define(`src3',`(*s3_ptr)')
define(`src4',`(*s4_ptr)')

define(`cdst',`(*cdst_ptr)')
define(`csrc1',`(*cs1_ptr)')
define(`csrc2',`(*cs2_ptr)')
define(`csrc3',`(*cs3_ptr)')
define(`csrc4',`(*cs4_ptr)')

define(`qdst',`(*qdst_ptr)')
define(`qsrc1',`(*qs1_ptr)')
define(`qsrc2',`(*qs2_ptr)')
define(`qsrc3',`(*qs3_ptr)')
define(`qsrc4',`(*qs4_ptr)')

/* This implementation of vmaxg vming requires that the destination
 * index array be contiguous...
 */

dnl	EXTLOC_DOIT(assignment)
define(`EXTLOC_DOIT',`
		$1;
		dst = index;
		dst_ptr++;
		nocc++;
')

dnl	EXTLOC_STATEMENT( augment_condition, restart_condition, assignment )
define(`EXTLOC_STATEMENT',`

	if( $2 ) {
		nocc=0;
		dst_ptr = orig_dst;
		EXTLOC_DOIT($3)
	} else if( $1 ){
		CHECK_IDXVEC_OVERFLOW
		EXTLOC_DOIT($3)
	}
	index++;
')



define(`CHECK_IDXVEC_OVERFLOW',`

	if( nocc >= idx_len ){
		if( verbose && ! overflow_warned ){
			sprintf(DEFAULT_ERROR_STRING,
"%s:  index vector has %d elements, more occurrences of extreme value", 
				func_name, idx_len);
			NADVISE(DEFAULT_ERROR_STRING);
			overflow_warned=1;
		}
		dst_ptr--;
		nocc--;
	}
')



/******************* Section 2 - slow loop bodies *******************/

define(`SHOW_SLOW_COUNT',`
sprintf(DEFAULT_ERROR_STRING,"count = %d %d %d %d %d",
INDEX_COUNT(count,0),
INDEX_COUNT(count,1),
INDEX_COUNT(count,2),
INDEX_COUNT(count,3),
INDEX_COUNT(count,4));
NADVISE(DEFAULT_ERROR_STRING);
')

dnl	GENERIC_SLOW_BODY( name, statement, decls, inits, copy_macro, ptr_init, comp_inc, inc_macro, debug_it )
define(`GENERIC_SLOW_BODY',`

dnl /* generic_slow_body copy_macro = /$5/  ptr_init = /$6/ */
{
	$3
	$4

	INIT_COUNT(i,4)
	while(i-- > 0){
		$5(2)
		INIT_COUNT(j,3)
		while(j-- > 0){
			$5(1)
			INIT_COUNT(k,2)
			while(k-- > 0){
				$5(0)
				INIT_COUNT(l,1)
				while(l-- > 0){
					$6
					INIT_COUNT(m,0)
					while(m-- > 0){
						$9
						$2 ;
						$7
					}
					$8(1)
				}
				$8(2)
			}
			$8(3)
		}
		$8(4)
	}
}
')

define(`DEBUG_2',`DEBUG_DST DEBUG_SRC1')

define(`DEBUG_2SRCS',`DEBUG_SRC1 DEBUG_SRC2')

define(`DEBUG_DST',`
	sprintf(DEFAULT_ERROR_STRING,"tdst = 0x%lx",(int_for_addr)dst_ptr);
	NADVISE(DEFAULT_ERROR_STRING);
')

define(`DEBUG_SRC1',`
	sprintf(DEFAULT_ERROR_STRING,"tsrc1 = 0x%lx",(int_for_addr)s1_ptr);
	NADVISE(DEFAULT_ERROR_STRING);
')

define(`DEBUG_SRC2',`
	sprintf(DEFAULT_ERROR_STRING,"tsrc2 = 0x%lx",(int_for_addr)s2_ptr);
	NADVISE(DEFAULT_ERROR_STRING);
')


dnl	GENERIC_XXX_SLOW_BODY( name, statement, decls, inits, copy_macro, ptr_init, inc_macro, debugit )
define(`GENERIC_XXX_SLOW_BODY',`

{
	$3
	$4

	INIT_COUNT(i,4)
	while(i-- > 0){
		$5(2)
		INIT_COUNT(j,3)
		while(j-- > 0){
			$5(1)
			INIT_COUNT(k,2)
			while(k-- > 0){
				$5(0)
				INIT_COUNT(l,1)
				while(l-- > 0){
					$6
					$8
					$2 ;
					$7(1)
				}
				$7(2)
			}
			$7(3)
		}
		$7(4)
	}
}
')

dnl	SIMPLE_SLOW_BODY(name,statement,typ,suffix,debug_it)
define(`SIMPLE_SLOW_BODY',`
/* simple_slow_body typ = /$3/ suffix = /$4/ */

		dnl DECLARE_BASES_##typ##suffix,
		dnl INIT_BASES_##typ##suffix,
		dnl COPY_BASES_##suffix,
		dnl INIT_PTRS_##suffix,
		dnl INC_PTRS_##typ##suffix,
		dnl INC_BASES_##typ##suffix,
GENERIC_SLOW_BODY($1,$2,`DECLARE_BASES(`',$3,$4)',`INIT_BASES(`',$3,$4)',`COPY_BASES(`',$3,$4)',`INIT_PTRS($4)',`INC_PTRS($3,$4)',`INC_BASES(`',$3,$4)',$5)
')

dnl	SIMPLE_EQSP_BODY(name, statement,typ,suffix,extra,debugit)
define(`SIMPLE_EQSP_BODY',`

{
	dnl EQSP_DECLS_##typ##suffix
	dnl EQSP_INIT_##typ##suffix
	dnl EXTRA_DECLS_##extra
	EQSP_DECLS($3,$4)
	EQSP_INIT($3,$4)
	EXTRA_DECLS($5)
	while(fl_ctr-- > 0){
		$6
		$2 ;
		dnl EQSP_ADVANCE_##typ##suffix
		EQSP_ADVANCE($3,$4)
	}
}
')

define(`DEBUG_CPX_3',`
sprintf(DEFAULT_ERROR_STRING,"executing dst = 0x%lx   src1 = 0x%lx  src2 = 0x%lx",
(int_for_addr)cdst_ptr,
(int_for_addr)cs1_ptr,
(int_for_addr)cs2_ptr);
NADVISE(DEFAULT_ERROR_STRING);
')


dnl  eqsp bodies

dnl	EQSP_BODY(bitmap,typ,vectors,extra)
define(`EQSP_BODY',`_EQSP_BODY($1$2$3$4)')
define(`_EQSP_BODY',EQSP_BODY_$1)

dnl	EQSP_BODY_2(name, statement)
define(`EQSP_BODY_2',`SIMPLE_EQSP_BODY($1,$2,`',2,`',`')')
define(`EQSP_BODY_3',`SIMPLE_EQSP_BODY($1,$2,`',3,`',`')')
define(`EQSP_BODY_4',`SIMPLE_EQSP_BODY($1,$2,`',4,`',`')')
define(`EQSP_BODY_5',`SIMPLE_EQSP_BODY($1,$2,`',5,`',`')')
define(`EQSP_BODY_SBM_1',`SIMPLE_EQSP_BODY($1,$2,`',SBM_1,`',`')')
define(`EQSP_BODY_SBM_2',`SIMPLE_EQSP_BODY($1,$2,`',SBM_2,`',`')')
define(`EQSP_BODY_SBM_3',`SIMPLE_EQSP_BODY($1,$2,`',SBM_3,`',`')')
define(`EQSP_BODY_SBM_CPX_1',`SIMPLE_EQSP_BODY($1,$2,`',SBM_CPX_1,`',`')')
define(`EQSP_BODY_SBM_CPX_2',`SIMPLE_EQSP_BODY($1,$2,`',SBM_CPX_2,`',`')')
define(`EQSP_BODY_SBM_CPX_3',`SIMPLE_EQSP_BODY($1,$2,`',SBM_CPX_3,`',`')')
define(`EQSP_BODY_SBM_QUAT_1',`SIMPLE_EQSP_BODY($1,$2,`',SBM_QUAT_1,`',`')')
define(`EQSP_BODY_SBM_QUAT_2',`SIMPLE_EQSP_BODY($1,$2,`',SBM_QUAT_2,`',`')')
define(`EQSP_BODY_SBM_QUAT_3',`SIMPLE_EQSP_BODY($1,$2,`',SBM_QUAT_3,`',`')')
define(`EQSP_BODY_BM_1',`SIMPLE_EQSP_BODY($1,$2,`',BM_1,`',`')')
define(`EQSP_BODY_DBM_1SRC',`SIMPLE_EQSP_BODY($1,$2,`',DBM_1SRC,`',`')')
define(`EQSP_BODY_DBM_SBM_',`SIMPLE_EQSP_BODY($1,$2,`',DBM_SBM,`',`')')
define(`EQSP_BODY_DBM_2SRCS',`SIMPLE_EQSP_BODY($1,$2,`',DBM_2SRCS,`',`')')
define(`EQSP_BODY_DBM_',`SIMPLE_EQSP_BODY($1,$2,`',DBM_,`',`')')
define(`EQSP_BODY_1',`SIMPLE_EQSP_BODY($1,$2,`',1,`',`')')
define(`EQSP_BODY_CPX_1',`SIMPLE_EQSP_BODY($1,$2,CPX_,1,`',`')')
define(`EQSP_BODY_CPX_2',`SIMPLE_EQSP_BODY($1,$2,CPX_,2,`',`')')
define(`EQSP_BODY_CPX_2_T2',`SIMPLE_EQSP_BODY($1,$2,CPX_,2,T2,`')')
define(`EQSP_BODY_CPX_2_T3',`SIMPLE_EQSP_BODY($1,$2,CPX_,2,T3,`')')
define(`EQSP_BODY_CPX_3',`SIMPLE_EQSP_BODY($1,$2,CPX_,3,`',`')')
define(`EQSP_BODY_CPX_3_T1',`SIMPLE_EQSP_BODY($1,$2,CPX_,3,T1,`')')
define(`EQSP_BODY_CPX_3_T2',`SIMPLE_EQSP_BODY($1,$2,CPX_,3,T2,`')')
define(`EQSP_BODY_CPX_3_T3',`SIMPLE_EQSP_BODY($1,$2,CPX_,3,T3,`')')
define(`EQSP_BODY_CPX_4',`SIMPLE_EQSP_BODY($1,$2,CPX_,4,`',`')')
define(`EQSP_BODY_CPX_5',`SIMPLE_EQSP_BODY($1,$2,CPX_,5,`',`')')
define(`EQSP_BODY_CCR_3',`SIMPLE_EQSP_BODY($1,$2,CCR_,3,`',`')')
define(`EQSP_BODY_CR_2',`SIMPLE_EQSP_BODY($1,$2,CR_,2,`',`')')
define(`EQSP_BODY_RC_2',`SIMPLE_EQSP_BODY($1,$2,RC_,2,`',`')')
define(`EQSP_BODY_QUAT_1',`SIMPLE_EQSP_BODY($1,$2,QUAT_,1,`',`')')
define(`EQSP_BODY_QUAT_2',`SIMPLE_EQSP_BODY($1,$2,QUAT_,2,`',`')')
define(`EQSP_BODY_QUAT_2_T4',`SIMPLE_EQSP_BODY($1,$2,QUAT_,2,T4,`')')
define(`EQSP_BODY_QUAT_2_T5',`SIMPLE_EQSP_BODY($1,$2,QUAT_,2,T5,`')')
define(`EQSP_BODY_QUAT_3',`SIMPLE_EQSP_BODY($1,$2,QUAT_,3,`',`')')
define(`EQSP_BODY_QUAT_3_T4',`SIMPLE_EQSP_BODY($1,$2,QUAT_,3,T4,`')')
define(`EQSP_BODY_QUAT_3_T5',`SIMPLE_EQSP_BODY($1,$2,QUAT_,3,T5,`')')
define(`EQSP_BODY_QUAT_4',`SIMPLE_EQSP_BODY($1,$2,QUAT_,4,`',`')')
define(`EQSP_BODY_QUAT_5',`SIMPLE_EQSP_BODY($1,$2,QUAT_,5,`',`')')
define(`EQSP_BODY_QQR_3',`SIMPLE_EQSP_BODY($1,$2,QQR_,3,`',`')')
define(`EQSP_BODY_QR_2',`SIMPLE_EQSP_BODY($1,$2,QR_,2,`',`')')
define(`EQSP_BODY_RQ_2',`SIMPLE_EQSP_BODY($1,$2,RQ_,2,`',`')')


dnl  slow bodies

define(`SLOW_BODY_1',`SIMPLE_SLOW_BODY($1,$2,`',1,`')')
define(`SLOW_BODY_2',`SIMPLE_SLOW_BODY($1,$2,`',2,`')')
define(`SLOW_BODY_3',`SIMPLE_SLOW_BODY($1,$2,`',3,`')')
define(`SLOW_BODY_CPX_1',`SIMPLE_XXX_SLOW_BODY($1,$2,`',CPX_,1,`',`')')
define(`SLOW_BODY_CPX_2',`SIMPLE_XXX_SLOW_BODY($1,$2,`',CPX_,2,`',`')')
define(`SLOW_BODY_CPX_2_T2',`SIMPLE_XXX_SLOW_BODY($1,$2,`',CPX_,2,T2,`')')
define(`SLOW_BODY_CPX_2_T3',`SIMPLE_XXX_SLOW_BODY($1,$2,`',CPX_,2,T3,`')')
define(`SLOW_BODY_CPX_3',`SIMPLE_XXX_SLOW_BODY($1,$2,`',CPX_,3,`',`')')
define(`SLOW_BODY_CPX_3_T1',`SIMPLE_XXX_SLOW_BODY($1,$2,`',CPX_,3,T1,`')')
define(`SLOW_BODY_CPX_3_T2',`SIMPLE_XXX_SLOW_BODY($1,$2,`',CPX_,3,T2,`')')
define(`SLOW_BODY_CPX_3_T3',`SIMPLE_XXX_SLOW_BODY($1,$2,`',CPX_,3,T3,`')')
define(`SLOW_BODY_QUAT_1',`SIMPLE_XXX_SLOW_BODY($1,$2,`',QUAT_,1,`',`')')
define(`SLOW_BODY_QUAT_2',`SIMPLE_XXX_SLOW_BODY($1,$2,`',QUAT_,2,`',`')')
define(`SLOW_BODY_QUAT_2_T4',`SIMPLE_XXX_SLOW_BODY($1,$2,`',QUAT_,2,T4,`')')
define(`SLOW_BODY_QUAT_2_T5',`SIMPLE_XXX_SLOW_BODY($1,$2,`',QUAT_,2,T5,`')')
define(`SLOW_BODY_QUAT_3',`SIMPLE_XXX_SLOW_BODY($1,$2,`',QUAT_,3,`',`')')
define(`SLOW_BODY_QUAT_3_T4',`SIMPLE_XXX_SLOW_BODY($1,$2,`',QUAT_,3,T4,`')')
define(`SLOW_BODY_QUAT_3_T5',`SIMPLE_XXX_SLOW_BODY($1,$2,`',QUAT_,3,T5,`')')
define(`SLOW_BODY_SBM_CPX_1',`SIMPLE_XXX_SLOW_BODY($1,$2,SBM_,CPX_,1,`',`')')
define(`SLOW_BODY_SBM_CPX_2',`SIMPLE_XXX_SLOW_BODY($1,$2,SBM_,CPX_,2,`',`')')
define(`SLOW_BODY_SBM_CPX_3',`SIMPLE_XXX_SLOW_BODY($1,$2,SBM_,CPX_,3,`',`')')
define(`SLOW_BODY_SBM_QUAT_1',`SIMPLE_XXX_SLOW_BODY($1,$2,SBM_,QUAT_,1,`',`')')
define(`SLOW_BODY_SBM_QUAT_2',`SIMPLE_XXX_SLOW_BODY($1,$2,SBM_,QUAT_,2,`',`')')
define(`SLOW_BODY_SBM_QUAT_3',`SIMPLE_XXX_SLOW_BODY($1,$2,SBM_,QUAT_,3,`',`')')
define(`SLOW_BODY_CR_2',`SIMPLE_XXX_SLOW_BODY($1,$2,`',CR_,2,`',`')')
define(`SLOW_BODY_QR_2',`SIMPLE_XXX_SLOW_BODY($1,$2,`',QR_,2,`',`')')
define(`SLOW_BODY_CCR_3',`SIMPLE_XXX_SLOW_BODY($1,$2,`',CCR_,3,`',`')')
define(`SLOW_BODY_QQR_3',`SIMPLE_XXX_SLOW_BODY($1,$2,`',QQR_,3,`',`')')
define(`SLOW_BODY_RC_2',`SIMPLE_XXX_SLOW_BODY($1,$2,`',RC_,2,`',`')')
define(`SLOW_BODY_4',`SIMPLE_SLOW_BODY($1,$2,`',4,`')')
define(`SLOW_BODY_5',`SIMPLE_SLOW_BODY($1,$2,`',5,`')')
define(`SLOW_BODY_DBM_',`SIMPLE_SLOW_BODY($1,$2,`',DBM_,`')')
dnl  put DEBUG_DBM_1SRC in last position to see debug info
define(`SLOW_BODY_DBM_1SRC',`SIMPLE_SLOW_BODY($1,$2,`',DBM_1SRC,/*DEBUG_DBM_1SRC*/)')
define(`SLOW_BODY_DBM_SBM_',`SIMPLE_SLOW_BODY($1,$2,`',DBM_SBM,`')')
define(`SLOW_BODY_SBM_3',`SIMPLE_SLOW_BODY($1,$2,`',SBM_3,`')')
define(`SLOW_BODY_SBM_2',`SIMPLE_SLOW_BODY($1,$2,`',SBM_2,`')')
define(`SLOW_BODY_SBM_1',`SIMPLE_SLOW_BODY($1,$2,`',SBM_1,`')')
define(`SLOW_BODY_DBM',`SIMPLE_SLOW_BODY($1,$2,`',DBM,`')')

define(`SLOW_BODY_DBM_2SRCS',`SIMPLE_SLOW_BODY($1,$2,`',DBM_2SRCS,`')')



dnl	SLOW_BODY_XX_2( name, statement,dsttyp,srctyp )
define(`SLOW_BODY_XX_2',`
GENERIC_SLOW_BODY($1,$2,`DECLARE_BASES_X_2($3,$4)',`INIT_BASES_X_2($3,$4)',`COPY_BASES_2',`INIT_PTRS_2',`INC_PTRS_2',`INC_BASES_2',`')
')


dnl  XXX is complex or quaternion

dnl	SIMPLE_XXX_SLOW_BODY(name, stat,bitmap,typ,suffix,extra,debugit)
define(`SIMPLE_XXX_SLOW_BODY',`
GENERIC_XXX_SLOW_BODY( $1, $2,`EXTRA_DECLS($6) DECLARE_BASES($3,$4,$5)',`INIT_BASES($3,$4,$5)',`COPY_BASES($3,$4,$5)',`INIT_PTRS($3,$4,$5)',`INC_BASES($3,$4,$5)',$7)
')

dnl	SLOW_BODY_XXX_2( name, stat, dsttyp, srctyp )
define(`SLOW_BODY_XXX_2',`
SIMPLE_XXX_SLOW_BODY($1,$2,$3,$4,`',2,`',`')
')

dnl	SLOW_BODY_SBM_XXX_3( name, stat, dsttyp, srctyp )
define(`SLOW_BODY_SBM_XXX_3',`SIMPLE_XXX_SLOW_BODY($1,$2,$3,$4,`SBM_',`3',`',`')')


dnl	SLOW_BODY_SBM_XXX_2( name, stat, dsttyp, srctyp )
define(`SLOW_BODY_SBM_XXX_2',`SIMPLE_XXX_SLOW_BODY($1,$2,$3,$4,`SBM_',`2',`',`')')

dnl	SLOW_BODY_SBM_XXX_1( name, stat, dsttyp, srctyp )
define(`SLOW_BODY_SBM_XXX_1',`GENERIC_XXX_SLOW_BODY($1,$2,DECL_INIT_SBM_XXX_1($3),COPY_BASES_SBM_XXX_1,INIT_PTRS_SBM_XXX_1,INC_BASES_SBM_XXX_1,`')')


dnl	SLOW_BODY_XXX_1( name, stat, dsttyp, srctyp )
define(`SLOW_BODY_XXX_1',`GENERIC_XXX_SLOW_BODY($1,$2,DECL_INIT_XXX_1($3),COPY_BASES_XXX_1,INIT_PTRS_XXX_1,INC_BASES_XXX_1,`')')


dnl	SLOW_BODY_XXX_3( name, stat, dsttyp, src1typ, src2typ )
define(`SLOW_BODY_XXX_3',`GENERIC_XXX_SLOW_BODY($1,$2,DECL_INIT_XXX_3($3,$4,$5),COPY_BASES_XXX_3,INIT_PTRS_XXX_3,INC_BASES_XXX_3,`')')




/* A "projection" loop is for the situation where the destination collapses
 * one or more dimensions of the input, e.g.  row=max(image), each element
 * in the destination row is the max of all the values in the corresponding
 * column of image.  In this case, we usually initialize with the first
 * value of the column, but it may be tricky to know this when we don't
 * know which dimensions will be collapsed...
 *
 * The ADJ_COUNTS macro initializes the loop_count array, to be the max
 * of all of the dimensions
 */


dnl	SLOW_BODY_PROJ_2( name, init_statement, statement )
define(`SLOW_BODY_PROJ_2',`

{
	/* slow_body_proj_2 */
	PROJ_LOOP_DECLS_2

	INIT_LOOP_COUNT	/* init loop_count to all 1s */

	ADJ_COUNTS(loop_count,count)
	ADJ_COUNTS(loop_count,s1_count)

	/* for the init loop, we dont need to loop over all the src */
	/* We just scan the destination once */
	NEW_PLOOP_2( $2, count )
	NEW_PLOOP_2( $3, loop_count )
}
')




dnl	SLOW_BODY_PROJ_IDX_2( name, init_statement, statement )
define(`SLOW_BODY_PROJ_IDX_2',`

{
PROJ_LOOP_DECLS_IDX_2
	orig_s1_ptr= (std_type *)VA_SRC_PTR(vap,0);

INIT_LOOP_COUNT

ADJ_COUNTS(loop_count,count)
ADJ_COUNTS(loop_count,s1_count)

	/* for the init loop, we dont need to loop over all the src */
	/* We just scan the destination once */
NEW_PLOOP_IDX_2( $2, count )
NEW_PLOOP_IDX_2( $3, loop_count )
}
')


dnl	PROJ3_SLOW_BODY( name, typ, init_statement, statement )
define(`PROJ3_SLOW_BODY',`

{
	dnl PROJ_LOOP_DECLS_##typ##3
	PROJ_LOOP_DECLS($2,`3')

	INIT_LOOP_COUNT

	ADJ_COUNTS(loop_count,count)
	ADJ_COUNTS(loop_count,s1_count)
	ADJ_COUNTS(loop_count,s2_count)

dnl	NEW_PLOOP_##typ##3( init_statement, count )
dnl	NEW_PLOOP_##typ##3( statement, loop_count )
	NEW_PLOOP($2,`3')($3,count)
	NEW_PLOOP($2,`3')($4,loop_count)
}
')


dnl	SLOW_BODY_PROJ_CPX_2( name, statement, init_statement )
define(`SLOW_BODY_PROJ_CPX_2',`SLOW_BODY_PROJ_XXX_2($1,$2,$3,`CPX_')')

dnl	SLOW_BODY_PROJ_QUAT_2( name, statement, init_statement )
define(`SLOW_BODY_PROJ_QUAT_2',`SLOW_BODY_PROJ_XXX_2($1,$2,$3,`QUAT_')')

dnl	SLOW_BODY_PROJ_CPX_3( name, statement, init_statement )
define(`SLOW_BODY_PROJ_CPX_3',`SLOW_BODY_PROJ_XXX_3($1,$2,$3,`CPX_')')

dnl	SLOW_BODY_PROJ_QUAT_3( name, statement, init_statement )
define(`SLOW_BODY_PROJ_QUAT_3',`SLOW_BODY_PROJ_XXX_3($1,$2,$3,`QUAT_')')

dnl	NEW_PLOOP(typ,vectors)
define(`NEW_PLOOP',`_NEW_PLOOP($1$2)')
define(`_NEW_PLOOP',NEW_PLOOP_$1)

dnl	NEW_PLOOP_CPX_2(statement,loop_count)
define(`NEW_PLOOP_CPX_2',`NEW_PLOOP_XXX_2($1,$2,`CPX_')')
define(`NEW_PLOOP_CPX_3',`NEW_PLOOP_XXX_3($1,$2,`CPX_')')
define(`NEW_PLOOP_QUAT_2',`NEW_PLOOP_XXX_2($1,$2,`QUAT_')')
define(`NEW_PLOOP_QUAT_3',`NEW_PLOOP_XXX_3($1,$2,`QUAT_')')

dnl	SLOW_BODY_PROJ_XXX_2( name, statement, init_statement, typ )
define(`SLOW_BODY_PROJ_XXX_2',`

{
	dnl PROJ_LOOP_DECLS_##typ##2
	PROJ_LOOP_DECLS($4,`2')

	INIT_LOOP_COUNT

	ADJ_COUNTS(loop_count,count)
	ADJ_COUNTS(loop_count,s1_count)

	dnl NEW_PLOOP_##typ##2( init_statement, count )
	dnl NEW_PLOOP_##typ##2( statement, loop_count )
	NEW_PLOOP($4,`2')($3,`count')
	NEW_PLOOP($4,`2')($2,`loop_count')
}
')

dnl	SLOW_BODY_PROJ_XXX_3( name, statement, init_statement, typ )
define(`SLOW_BODY_PROJ_XXX_3',`

{
	dnl PROJ_LOOP_DECLS_##typ##3
	PROJ_LOOP_DECLS($4,`3')

	INIT_LOOP_COUNT

	ADJ_COUNTS(loop_count,count)
	ADJ_COUNTS(loop_count,s1_count)
	ADJ_COUNTS(loop_count,s2_count)

	dnl NEW_PLOOP_##typ##3( init_statement, count )
	dnl NEW_PLOOP_##typ##3( statement, loop_count )
	NEW_PLOOP($4,`3')($3,`count')
	NEW_PLOOP($4,`3')($2,`loop_count')
}
')

/* Projection loop
 *
 * We typically call this once with the counts of the destination vector, to initialize,
 * and then again with the source counts to perform the projection...
 */


dnl	NEW_PLOOP_2( statement,count_arr )
define(`NEW_PLOOP_2',`

	INIT_BASES_2
	_INIT_COUNT(i,$2,4)
	while(i-- > 0){
		COPY_BASES_2(2)
		_INIT_COUNT(j,$2,3)
		while(j-- > 0){
			COPY_BASES_2(1)
			_INIT_COUNT(k,$2,2)
			while(k-- > 0){
				COPY_BASES_2(0)
				_INIT_COUNT(l,$2,1)
				while(l-- > 0){
					INIT_PTRS_2
					_INIT_COUNT(m,$2,0)
					while(m-- > 0){
						$1 ;
						INC_PTRS_2
					}
					INC_BASES_2(1)
				}
				INC_BASES_2(2)
			}
			INC_BASES_2(3)
		}
		INC_BASES_2(4)
	}
')




dnl	(`NEW_PLOOP_IDX_2( statement,count_arr )
define(`NEW_PLOOP_IDX_2',`

	INIT_BASES_IDX_2
	_INIT_COUNT(i,$2,4)
	while(i-- > 0){
		COPY_BASES_IDX_2(2)
		_INIT_COUNT(j,$2,3)
		while(j-- > 0){
			COPY_BASES_IDX_2(1)
			_INIT_COUNT(k,$2,2)
			while(k-- > 0){
				COPY_BASES_IDX_2(0) /* sets index_base[0] */
				_INIT_COUNT(l,$2,1)
				while(l-- > 0){
					INIT_PTRS_IDX_2
					_INIT_COUNT(m,$2,0)
					while(m-- > 0){
						$1 ;
						INC_PTRS_IDX_2
					}
					INC_BASES_IDX_2(1)
				}
				INC_BASES_IDX_2(2)
			}
			INC_BASES_IDX_2(3)
		}
		INC_BASES_IDX_2(4)
	}
')



dnl	NEW_PLOOP_3( statement, count_arr )
define(`NEW_PLOOP_3',`

	INIT_BASES_3
	_INIT_COUNT(i,$2,4)
	while(i-- > 0){
		COPY_BASES_3(2)
		_INIT_COUNT(j,$2,3)
		while(j-- > 0){
			COPY_BASES_3(1)
			_INIT_COUNT(k,$2,2)
			while(k-- > 0){
				COPY_BASES_3(0)
				_INIT_COUNT(l,$2,1)
				while(l-- > 0){
					INIT_PTRS_3
					_INIT_COUNT(m,$2,0)
					while(m-- > 0){
						$1 ;
						INC_PTRS_3
					}
					INC_BASES_3(1)
				}
				INC_BASES_3(2)
			}
			INC_BASES_3(3)
		}
		INC_BASES_3(4)
	}
')

dnl	NEW_PLOOP_XXX_2( statement,count_arr,typ )
define(`NEW_PLOOP_XXX_2',`

	dnl INIT_BASES_##typ##2
	INIT_BASES(`',$3,`2')
	_INIT_COUNT(i,$2,4)
	while(i-- > 0){
		dnl COPY_BASES_##typ##2(2)
		COPY_BASES(`',$3,`2')(2)
		_INIT_COUNT(j,$2,3)
		while(j-- > 0){
			dnl COPY_BASES_##typ##2(1)
			COPY_BASES(`',$3,`2')(1)
			_INIT_COUNT(k,$2,2)
			while(k-- > 0){
				dnl COPY_BASES_##typ##2(0)
				COPY_BASES(`',$3,`2')(0)
				_INIT_COUNT(l,$2,1)
				while(l-- > 0){
					dnl INIT_PTRS_##typ##2
					INIT_PTRS($3,`2')
						$1 ;
					dnl INC_BASES_##typ##2(1)
					INC_BASES(`',$3,`2')(1)
				}
				dnl INC_BASES_##typ##2(2)
				INC_BASES(`',$3,`2')(2)
			}
			dnl INC_BASES_##typ##2(3)
			INC_BASES(`',$3,`2')(3)
		}
		dnl INC_BASES_##typ##2(4)
		INC_BASES(`',$3,`2')(4)
	}
')


dnl	NEW_PLOOP_XXX_3( statement, count_arr, typ )
define(`NEW_PLOOP_XXX_3',`

	INIT_BASES(`',$3,`3')
	_INIT_COUNT(i,$2,4)
	while(i-- > 0){
		COPY_BASES(`',$3,`3')(2)
		_INIT_COUNT(j,$2,3)
		while(j-- > 0){
			COPY_BASES(`',$3,`3')(1)
			_INIT_COUNT(k,$2,2)
			while(k-- > 0){
				COPY_BASES(`',$3,`3')(0)
				_INIT_COUNT(l,$2,1)
				while(l-- > 0){
					INIT_PTRS($3,`3')
						$1 ;
					INC_BASES(`',$3,`3')(1)
				}
				INC_BASES(`',$3,`3')(2)
			}
			INC_BASES(`',$3,`3')(3)
		}
		INC_BASES(`',$3,`3')(4)
	}
')


dnl	EXTLOC_SLOW_FUNC(name, statement)
define(`EXTLOC_SLOW_FUNC',`

static void SLOW_NAME($1)( LINK_FUNC_ARG_DECLS )
{
	DECLARE_VBASE_SRC1
	DECLARE_FIVE_LOOP_INDICES
	EXTLOC_DECLS
	const char * func_name="$1";

	INIT_BASES_SRC1
	s1_ptr = (std_type *) VA_SRC_PTR(vap,0);
	EXTLOC_INITS
	_INIT_COUNT(i,s1_count,4)
	while(i-- > 0){
		COPY_BASES_SRC1(2)
		_INIT_COUNT(j,s1_count,3)
		while(j-- > 0){
			COPY_BASES_SRC1(1)
			_INIT_COUNT(k,s1_count,2)
			while(k-- > 0){
				COPY_BASES_SRC1(0)
				_INIT_COUNT(l,s1_count,1)
				while(l-- > 0){
					INIT_PTRS_SRC1
					_INIT_COUNT(m,s1_count,0)
					while(m-- > 0){
						$2 ;
						INC_PTRS_SRC1
					}
					INC_BASES_SRC1(1)
				}
				INC_BASES_SRC1(2)
			}
			INC_BASES_SRC1(3)
		}
		INC_BASES_SRC1(4)
	}
	SET_EXTLOC_RETURN_SCALARS
}
')


/*************** Section 3 - slow functions ***********************/

/******************* Section 4 - obsolete loops?  **********************/

/********************* Section 5 - fast functions ****************/

dnl	FAST_BODY(bitmap,typ,vectors,extra)
define(`FAST_BODY',`_FAST_BODY($1$2$3$4)')
define(`_FAST_BODY',FAST_BODY_$1)

dnl	SLOW_BODY(bitmap,typ,vectors,extra)
define(`SLOW_BODY',`_SLOW_BODY($1$2$3$4)')
define(`_SLOW_BODY',SLOW_BODY_$1)

dnl	FF_DECL(name)
define(`FF_DECL',`/* ff_decl /$1/ */static void FAST_NAME($1)')
define(`EF_DECL',`static void EQSP_NAME($1)')
define(`SF_DECL',`static void SLOW_NAME($1)')

dnl	MOV_FF_DECL(name, statement,bitmap,typ,scalars,vectors)
define(`MOV_FF_DECL',`

	FF_DECL($1)( LINK_FUNC_ARG_DECLS )
dnl	FAST_BODY_##typ##MOV( $1, $4 )
	FAST_BODY(`',$4,`',`MOV')( $1, $4 )
')

dnl	GENERIC_FF_DECL(name, statement,bitmap,typ,scalars,vectors,extra)
define(`GENERIC_FF_DECL',`

	FF_DECL($1)( LINK_FUNC_ARG_DECLS )
dnl	FAST_BODY_##bitmap##typ##vectors##extra( name, statement )
	FAST_BODY($3,$4,$6,$7)($1,$2)
')

dnl	GENERIC_EF_DECL(name, statement,bitmap,typ,scalars,vectors,extra)
define(`GENERIC_EF_DECL',`

	EF_DECL($1)( LINK_FUNC_ARG_DECLS )
dnl	EQSP_BODY_##bitmap##typ##vectors##extra( name, statement )
	EQSP_BODY($3,$4,$6,$7)($1,$2)
')

dnl	GENERIC_SF_DECL(name,statement,bitmap,typ,scalars,vectors,extra)
define(`GENERIC_SF_DECL',`

	SF_DECL($1)( LINK_FUNC_ARG_DECLS )
dnl	SLOW_BODY_##bitmap##typ##vectors##extra( name, statement )
	SLOW_BODY($3,$4,$6,$7)($1,$2)
')


dnl	GENERIC_FUNC_DECLS(name,statement,bitmap,typ,scalars,vectors,extra)
define(`GENERIC_FUNC_DECLS',`
GENERIC_FF_DECL($1,$2,$3,$4,$5,$6,$7)
GENERIC_EF_DECL($1,$2,$3,$4,$5,$6,$7)
GENERIC_SF_DECL($1,$2,$3,$4,$5,$6,$7)
')


dnl	MOV_FUNC_DECLS(name,statement,bitmap,typ,scalars,vectors)
define(`MOV_FUNC_DECLS',`
MOV_FF_DECL($1,$2,$3,$4,$5,$6)
GENERIC_EF_DECL($1,$2,$3,$4,$5,$6,`')
GENERIC_SF_DECL($1,$2,$3,$4,$5,$6,`')
')


dnl	IDXRES_FAST_FUNC(name,init_statement,statement)
define(`IDXRES_FAST_FUNC',`

FF_DECL($1)(IDX_PTR_ARG,PTR_ARGS_SRC1,COUNT_ARG)
{
	sprintf(DEFAULT_ERROR_STRING,"Sorry, Function %s is not implemented.","$1");
	NWARN(DEFAULT_ERROR_STRING);
}
')

define(`EXTLOC_DECLS',`
	index_type index;
	index_type *dst_ptr;
	index_type *orig_dst;
	std_type extval;
	dimension_t nocc;
	dimension_t idx_len;
	int overflow_warned=0;
')

define(`EXTLOC_INITS',`
	dst_ptr = (index_type *)VA_DEST_PTR(vap);
	orig_dst = dst_ptr;
	nocc = 0;
	index = 0;
	extval = src1;
	/*idx_len = VA_SCALAR_VAL_UDI(vap,0);*/
	idx_len = VA_DEST_LEN(vap);
')


define(`SET_EXTLOC_RETURN_SCALARS',`
	SET_VA_SCALAR_VAL_STD(vap,0,extval);
	SET_VA_SCALAR_VAL_UDI(vap,1,nocc);
')


dnl	EXTLOC_FAST_FUNC(name, statement)
define(`EXTLOC_FAST_FUNC',`

FF_DECL($1)( LINK_FUNC_ARG_DECLS )
{
	FAST_DECLS_SRC1
	FAST_INIT_SRC1
	EXTLOC_DECLS
	dimension_t fl_ctr;
	const char * func_name="$1";

	FAST_INIT_COUNT
	FAST_INIT_SRC1
	EXTLOC_INITS

	while(fl_ctr-- > 0){
		$2 ;
		FAST_ADVANCE_SRC1
	}

	/* Now return nocc & extval */
	SET_EXTLOC_RETURN_SCALARS
}
')


dnl	EXTLOC_EQSP_FUNC(name, statement)
define(`EXTLOC_EQSP_FUNC',`

EF_DECL($1)( LINK_FUNC_ARG_DECLS )
{
	EQSP_DECLS_SRC1
	EQSP_INIT_SRC1
	EXTLOC_DECLS
	dimension_t fl_ctr;
	const char * func_name="$1";

	EQSP_INIT_COUNT
	EQSP_INIT_SRC1
	EXTLOC_INITS

	while(fl_ctr-- > 0){
		$2 ;
		EQSP_ADVANCE_SRC1
	}

	/* Now return nocc & extval */
	SET_EXTLOC_RETURN_SCALARS
}
')


/***************** Section 6 - fast loop bodies *************/


/* FAST_DECLS declare variables at the top of the body */

dnl	FAST_DECLS(typ,suffix)
define(`FAST_DECLS',`_FAST_DECLS($1$2)')
define(`_FAST_DECLS',FAST_DECLS_$1)

define(`FAST_DECLS_1',`dest_type *dst_ptr; dimension_t fl_ctr;')
define(`FAST_DECLS_SRC1',`std_type *s1_ptr;')
define(`FAST_DECLS_SRC2',`std_type *s2_ptr;')
define(`FAST_DECLS_SRC3',`std_type *s3_ptr;')
define(`FAST_DECLS_SRC4',`std_type *s4_ptr;')
define(`FAST_DECLS_2SRCS',`FAST_DECLS_SRC1 FAST_DECLS_SRC2')
define(`FAST_DECLS_2',`FAST_DECLS_1 FAST_DECLS_SRC1')
define(`FAST_DECLS_3',`FAST_DECLS_2 FAST_DECLS_SRC2')
define(`FAST_DECLS_4',`FAST_DECLS_3 FAST_DECLS_SRC3')
define(`FAST_DECLS_5',`FAST_DECLS_4 FAST_DECLS_SRC4')
define(`FAST_DECLS_SBM',`int sbm_bit; bitmap_word *sbm_ptr;')
define(`FAST_DECLS_DBM',`int dbm_bit; bitmap_word *dbm_ptr; dimension_t fl_ctr;')
define(`FAST_DECLS_DBM_1SRC',`FAST_DECLS_DBM FAST_DECLS_SRC1')
define(`FAST_DECLS_DBM_SBM',`FAST_DECLS_DBM FAST_DECLS_SBM')
define(`FAST_DECLS_SBM_1',`FAST_DECLS_SBM FAST_DECLS_1')
define(`FAST_DECLS_SBM_2',`FAST_DECLS_SBM FAST_DECLS_2')
define(`FAST_DECLS_SBM_3',`FAST_DECLS_SBM FAST_DECLS_3')
define(`FAST_DECLS_SBM_CPX_1',`FAST_DECLS_SBM FAST_DECLS_CPX_1')
define(`FAST_DECLS_SBM_CPX_2',`FAST_DECLS_SBM FAST_DECLS_CPX_2')
define(`FAST_DECLS_SBM_CPX_3',`FAST_DECLS_SBM FAST_DECLS_CPX_3')
define(`FAST_DECLS_SBM_QUAT_1',`FAST_DECLS_SBM FAST_DECLS_QUAT_1')
define(`FAST_DECLS_SBM_QUAT_2',`FAST_DECLS_SBM FAST_DECLS_QUAT_2')
define(`FAST_DECLS_SBM_QUAT_3',`FAST_DECLS_SBM FAST_DECLS_QUAT_3')
define(`FAST_DECLS_DBM_',`FAST_DECLS_DBM')
define(`FAST_DECLS_DBM_2SRCS',`FAST_DECLS_DBM FAST_DECLS_2SRCS')

define(`CPX_TMP_DECLS',`std_type r; std_type theta; std_type arg;')
define(`FAST_DECLS_CPX_1',`dest_cpx *cdst_ptr; dimension_t fl_ctr;')
define(`FAST_DECLS_CPX_SRC1',`std_cpx *cs1_ptr;')
define(`FAST_DECLS_CPX_SRC2',`std_cpx *cs2_ptr;')
define(`FAST_DECLS_CPX_SRC3',`std_cpx *cs3_ptr;')
define(`FAST_DECLS_CPX_SRC4',`std_cpx *cs4_ptr;')
define(`FAST_DECLS_CPX_2',`FAST_DECLS_CPX_1 FAST_DECLS_CPX_SRC1')
define(`FAST_DECLS_CPX_3',`FAST_DECLS_CPX_2 FAST_DECLS_CPX_SRC2')
define(`FAST_DECLS_CPX_4',`FAST_DECLS_CPX_3 FAST_DECLS_CPX_SRC3')
define(`FAST_DECLS_CPX_5',`FAST_DECLS_CPX_4 FAST_DECLS_CPX_SRC4')
define(`FAST_DECLS_CCR_3',`FAST_DECLS_CPX_2 FAST_DECLS_SRC2')
define(`FAST_DECLS_CR_2',`FAST_DECLS_CPX_1 FAST_DECLS_SRC1')
define(`FAST_DECLS_RC_2',`FAST_DECLS_1 FAST_DECLS_CPX_SRC1')

define(`FAST_DECLS_QUAT_1',`dest_quat *qdst_ptr; dimension_t fl_ctr;')
define(`FAST_DECLS_QUAT_SRC1',`std_quat *qs1_ptr;')
define(`FAST_DECLS_QUAT_SRC2',`std_quat *qs2_ptr;')
define(`FAST_DECLS_QUAT_SRC3',`std_quat *qs3_ptr;')
define(`FAST_DECLS_QUAT_SRC4',`std_quat *qs4_ptr;')
define(`FAST_DECLS_QUAT_2',`FAST_DECLS_QUAT_1 FAST_DECLS_QUAT_SRC1')
define(`FAST_DECLS_QUAT_3',`FAST_DECLS_QUAT_2 FAST_DECLS_QUAT_SRC2')
define(`FAST_DECLS_QUAT_4',`FAST_DECLS_QUAT_3 FAST_DECLS_QUAT_SRC3')
define(`FAST_DECLS_QUAT_5',`FAST_DECLS_QUAT_4 FAST_DECLS_QUAT_SRC4')
define(`FAST_DECLS_QQR_3',`FAST_DECLS_QUAT_2 FAST_DECLS_SRC2')
define(`FAST_DECLS_QR_2',`FAST_DECLS_QUAT_1 FAST_DECLS_SRC1')
define(`FAST_DECLS_RQ_2',`FAST_DECLS_1 FAST_DECLS_QUAT_SRC1')

dnl  eqsp decls

dnl EQSP_DECLS(typ,suffix)
define(`EQSP_DECLS',`_EQSP_DECLS($1$2)')
define(`_EQSP_DECLS',EQSP_DECLS_$1)

define(`EQSP_DECLS_1',`dest_type *dst_ptr; dimension_t fl_ctr;')
define(`EQSP_DECLS_SRC1',`std_type *s1_ptr;')
define(`EQSP_DECLS_SRC2',`std_type *s2_ptr;')
define(`EQSP_DECLS_SRC3',`std_type *s3_ptr;')
define(`EQSP_DECLS_SRC4',`std_type *s4_ptr;')
define(`EQSP_DECLS_2SRCS',`EQSP_DECLS_SRC1 EQSP_DECLS_SRC2')
define(`EQSP_DECLS_2',`EQSP_DECLS_1 EQSP_DECLS_SRC1')
define(`EQSP_DECLS_3',`EQSP_DECLS_2 EQSP_DECLS_SRC2')
define(`EQSP_DECLS_4',`EQSP_DECLS_3 EQSP_DECLS_SRC3')
define(`EQSP_DECLS_5',`EQSP_DECLS_4 EQSP_DECLS_SRC4')
define(`EQSP_DECLS_SBM',`int sbm_bit; bitmap_word *sbm_ptr;')
define(`EQSP_DECLS_DBM',`int dbm_bit; bitmap_word *dbm_ptr; dimension_t fl_ctr;')
define(`EQSP_DECLS_DBM_1SRC',`EQSP_DECLS_DBM EQSP_DECLS_SRC1')
define(`EQSP_DECLS_DBM_SBM',`EQSP_DECLS_DBM EQSP_DECLS_SBM')
define(`EQSP_DECLS_SBM_1',`EQSP_DECLS_SBM EQSP_DECLS_1')
define(`EQSP_DECLS_SBM_2',`EQSP_DECLS_SBM EQSP_DECLS_2')
define(`EQSP_DECLS_SBM_3',`EQSP_DECLS_SBM EQSP_DECLS_3')
define(`EQSP_DECLS_SBM_CPX_1',`EQSP_DECLS_SBM EQSP_DECLS_CPX_1')
define(`EQSP_DECLS_SBM_CPX_2',`EQSP_DECLS_SBM EQSP_DECLS_CPX_2')
define(`EQSP_DECLS_SBM_CPX_3',`EQSP_DECLS_SBM EQSP_DECLS_CPX_3')
define(`EQSP_DECLS_SBM_QUAT_1',`EQSP_DECLS_SBM EQSP_DECLS_QUAT_1')
define(`EQSP_DECLS_SBM_QUAT_2',`EQSP_DECLS_SBM EQSP_DECLS_QUAT_2')
define(`EQSP_DECLS_SBM_QUAT_3',`EQSP_DECLS_SBM EQSP_DECLS_QUAT_3')
define(`EQSP_DECLS_DBM_',`EQSP_DECLS_DBM')
define(`EQSP_DECLS_DBM_2SRCS',`EQSP_DECLS_DBM EQSP_DECLS_2SRCS')

define(`EQSP_DECLS_CPX_1',`dest_cpx *cdst_ptr; dimension_t fl_ctr;')
define(`EQSP_DECLS_CPX_SRC1',`std_cpx *cs1_ptr;')
define(`EQSP_DECLS_CPX_SRC2',`std_cpx *cs2_ptr;')
define(`EQSP_DECLS_CPX_SRC3',`std_cpx *cs3_ptr;')
define(`EQSP_DECLS_CPX_SRC4',`std_cpx *cs4_ptr;')
define(`EQSP_DECLS_CPX_2',`EQSP_DECLS_CPX_1 EQSP_DECLS_CPX_SRC1')
define(`EQSP_DECLS_CPX_3',`EQSP_DECLS_CPX_2 EQSP_DECLS_CPX_SRC2')
define(`EQSP_DECLS_CPX_4',`EQSP_DECLS_CPX_3 EQSP_DECLS_CPX_SRC3')
define(`EQSP_DECLS_CPX_5',`EQSP_DECLS_CPX_4 EQSP_DECLS_CPX_SRC4')
define(`EQSP_DECLS_CCR_3',`EQSP_DECLS_CPX_2 EQSP_DECLS_SRC2')
define(`EQSP_DECLS_CR_2',`EQSP_DECLS_CPX_1 EQSP_DECLS_SRC1')
define(`EQSP_DECLS_RC_2',`EQSP_DECLS_1 EQSP_DECLS_CPX_SRC1')

define(`EQSP_DECLS_QUAT_1',`dest_quat *qdst_ptr; dimension_t fl_ctr;')
define(`EQSP_DECLS_QUAT_SRC1',`std_quat *qs1_ptr;')
define(`EQSP_DECLS_QUAT_SRC2',`std_quat *qs2_ptr;')
define(`EQSP_DECLS_QUAT_SRC3',`std_quat *qs3_ptr;')
define(`EQSP_DECLS_QUAT_SRC4',`std_quat *qs4_ptr;')
define(`EQSP_DECLS_QUAT_2',`EQSP_DECLS_QUAT_1 EQSP_DECLS_QUAT_SRC1')
define(`EQSP_DECLS_QUAT_3',`EQSP_DECLS_QUAT_2 EQSP_DECLS_QUAT_SRC2')
define(`EQSP_DECLS_QUAT_4',`EQSP_DECLS_QUAT_3 EQSP_DECLS_QUAT_SRC3')
define(`EQSP_DECLS_QUAT_5',`EQSP_DECLS_QUAT_4 EQSP_DECLS_QUAT_SRC4')
define(`EQSP_DECLS_QQR_3',`EQSP_DECLS_QUAT_2 EQSP_DECLS_SRC2')
define(`EQSP_DECLS_QR_2',`EQSP_DECLS_QUAT_1 EQSP_DECLS_SRC1')
define(`EQSP_DECLS_RQ_2',`EQSP_DECLS_1 EQSP_DECLS_QUAT_SRC1')


/* FAST_INIT sets up the local vars from the arguments passed */


dnl	FAST_INIT(typ,suffix)
define(`FAST_INIT',`/* fast_init /$1/ /$2/ */_FAST_INIT($1$2)')
define(`_FAST_INIT',FAST_INIT_$1)

define(`FAST_INIT_1',`
	dst_ptr = (dest_type *)VA_DEST_PTR(vap);
FAST_INIT_COUNT
')

/* We used to divide by 2 or 4 for cpx and quat, but no longer needed. */
define(`FAST_INIT_COUNT',`fl_ctr = VA_LENGTH(vap);')
dnl /*
dnl define(`FAST_INIT_COUNT_CPX	fl_ctr = VA_LENGTH(vap);
dnl define(`FAST_INIT_COUNT_QUAT	fl_ctr = VA_LENGTH(vap);
dnl */

define(`FAST_INIT_SRC1',`s1_ptr = (std_type *)VA_SRC_PTR(vap,0);')
define(`FAST_INIT_SRC2',`s2_ptr = (std_type *)VA_SRC_PTR(vap,1);')
define(`FAST_INIT_SRC3',`s3_ptr = (std_type *)VA_SRC_PTR(vap,2);')
define(`FAST_INIT_SRC4',`s4_ptr = (std_type *)VA_SRC_PTR(vap,3);')
define(`FAST_INIT_2',`FAST_INIT_1	FAST_INIT_SRC1')
define(`FAST_INIT_3',`FAST_INIT_2	FAST_INIT_SRC2')
define(`FAST_INIT_4',`FAST_INIT_3	FAST_INIT_SRC3')
define(`FAST_INIT_5',`FAST_INIT_4	FAST_INIT_SRC4')
define(`FAST_INIT_2SRCS',`FAST_INIT_SRC1	FAST_INIT_SRC2')


define(`FAST_INIT_CPX_1',`
	cdst_ptr = (dest_cpx *)VA_DEST_PTR(vap);
FAST_INIT_COUNT
')
define(`FAST_INIT_CPX_SRC1',`cs1_ptr = (std_cpx *)VA_SRC_PTR(vap,0);')
define(`FAST_INIT_CPX_SRC2',`cs2_ptr = (std_cpx *)VA_SRC_PTR(vap,1);')
define(`FAST_INIT_CPX_SRC3',`cs3_ptr = (std_cpx *)VA_SRC_PTR(vap,2);')
define(`FAST_INIT_CPX_SRC4',`cs4_ptr = (std_cpx *)VA_SRC_PTR(vap,3);')

define(`FAST_INIT_CPX_2',`FAST_INIT_CPX_1 FAST_INIT_CPX_SRC1')
define(`FAST_INIT_CPX_3',`FAST_INIT_CPX_2 FAST_INIT_CPX_SRC2')
define(`FAST_INIT_CPX_4',`FAST_INIT_CPX_3 FAST_INIT_CPX_SRC3')
define(`FAST_INIT_CPX_5',`FAST_INIT_CPX_4 FAST_INIT_CPX_SRC4')
define(`FAST_INIT_CCR_3',`FAST_INIT_CPX_2 FAST_INIT_SRC2')
define(`FAST_INIT_CR_2',`FAST_INIT_CPX_1 FAST_INIT_SRC1')
define(`FAST_INIT_RC_2',`FAST_INIT_1 FAST_INIT_CPX_SRC1')

define(`FAST_INIT_QUAT_1',`
	qdst_ptr = (dest_quat *)VA_DEST_PTR(vap);
FAST_INIT_COUNT
')

define(`FAST_INIT_QUAT_SRC1',`qs1_ptr = (std_quat *)VA_SRC_PTR(vap,0);')
define(`FAST_INIT_QUAT_SRC2',`qs2_ptr = (std_quat *)VA_SRC_PTR(vap,1);')
define(`FAST_INIT_QUAT_SRC3',`qs3_ptr = (std_quat *)VA_SRC_PTR(vap,2);')
define(`FAST_INIT_QUAT_SRC4',`qs4_ptr = (std_quat *)VA_SRC_PTR(vap,3);')

define(`FAST_INIT_QUAT_2',`FAST_INIT_QUAT_1 FAST_INIT_QUAT_SRC1')
define(`FAST_INIT_QUAT_3',`FAST_INIT_QUAT_2 FAST_INIT_QUAT_SRC2')
define(`FAST_INIT_QUAT_4',`FAST_INIT_QUAT_3 FAST_INIT_QUAT_SRC3')
define(`FAST_INIT_QUAT_5',`FAST_INIT_QUAT_4 FAST_INIT_QUAT_SRC4')
define(`FAST_INIT_QQR_3',`FAST_INIT_QUAT_2 FAST_INIT_SRC2')
define(`FAST_INIT_QR_2',`FAST_INIT_QUAT_1 FAST_INIT_SRC1')
define(`FAST_INIT_RQ_2',`FAST_INIT_1 FAST_INIT_QUAT_SRC1')

define(`FAST_INIT_DBM_',`FAST_INIT_DBM')

define(`FAST_INIT_DBM',`
	dbm_bit = VA_DBM_BIT0(vap);
	dbm_ptr=VA_DEST_PTR(vap);
FAST_INIT_COUNT
')

define(`FAST_INIT_SBM',`
	sbm_bit = VA_SBM_BIT0(vap);
	sbm_ptr=VA_SRC_PTR(vap,4);
')

define(`FAST_INIT_DBM_2SRCS',`FAST_INIT_DBM FAST_INIT_2SRCS')
define(`FAST_INIT_DBM_1SRC',`FAST_INIT_DBM FAST_INIT_SRC1')
define(`FAST_INIT_DBM_SBM',`FAST_INIT_DBM FAST_INIT_SBM')
define(`FAST_INIT_SBM_1',`FAST_INIT_SBM FAST_INIT_1')
define(`FAST_INIT_SBM_2',`FAST_INIT_SBM FAST_INIT_2')
define(`FAST_INIT_SBM_3',`FAST_INIT_SBM FAST_INIT_3')
define(`FAST_INIT_SBM_CPX_1',`FAST_INIT_SBM FAST_INIT_CPX_1')
define(`FAST_INIT_SBM_CPX_2',`FAST_INIT_SBM FAST_INIT_CPX_2')
define(`FAST_INIT_SBM_CPX_3',`FAST_INIT_SBM FAST_INIT_CPX_3')
define(`FAST_INIT_SBM_QUAT_1',`FAST_INIT_SBM FAST_INIT_QUAT_1')
define(`FAST_INIT_SBM_QUAT_2',`FAST_INIT_SBM FAST_INIT_QUAT_2')
define(`FAST_INIT_SBM_QUAT_3',`FAST_INIT_SBM FAST_INIT_QUAT_3')

dnl
dnl  eqsp init


dnl EQSP_INIT(typ,suffix)
define(`EQSP_INIT',`_EQSP_INIT($1$2)')
define(`_EQSP_INIT',EQSP_INIT_$1)

define(`EQSP_INIT_1',`
	dst_ptr = (dest_type *)VA_DEST_PTR(vap);
EQSP_INIT_COUNT
')

/* We used to divide by 2 or 4 for cpx and quat, but no longer needed. */
define(`EQSP_INIT_COUNT',`fl_ctr = VA_LENGTH(vap);')
dnl /*
dnl define(`EQSP_INIT_COUNT_CPX	fl_ctr = VA_LENGTH(vap);
dnl define(`EQSP_INIT_COUNT_QUAT	fl_ctr = VA_LENGTH(vap);
dnl */

define(`EQSP_INIT_SRC1',`s1_ptr = (std_type *)VA_SRC_PTR(vap,0);')
define(`EQSP_INIT_SRC2',`s2_ptr = (std_type *)VA_SRC_PTR(vap,1);')
define(`EQSP_INIT_SRC3',`s3_ptr = (std_type *)VA_SRC_PTR(vap,2);')
define(`EQSP_INIT_SRC4',`s4_ptr = (std_type *)VA_SRC_PTR(vap,3);')
define(`EQSP_INIT_2',`EQSP_INIT_1	EQSP_INIT_SRC1')
define(`EQSP_INIT_3',`EQSP_INIT_2	EQSP_INIT_SRC2')
define(`EQSP_INIT_4',`EQSP_INIT_3	EQSP_INIT_SRC3')
define(`EQSP_INIT_5',`EQSP_INIT_4	EQSP_INIT_SRC4')
define(`EQSP_INIT_2SRCS',`EQSP_INIT_SRC1	EQSP_INIT_SRC2')


define(`EQSP_INIT_CPX_1',`
	cdst_ptr = (dest_cpx *)VA_DEST_PTR(vap);
EQSP_INIT_COUNT
')
define(`EQSP_INIT_CPX_SRC1',`cs1_ptr = (std_cpx *)VA_SRC_PTR(vap,0);')
define(`EQSP_INIT_CPX_SRC2',`cs2_ptr = (std_cpx *)VA_SRC_PTR(vap,1);')
define(`EQSP_INIT_CPX_SRC3',`cs3_ptr = (std_cpx *)VA_SRC_PTR(vap,2);')
define(`EQSP_INIT_CPX_SRC4',`cs4_ptr = (std_cpx *)VA_SRC_PTR(vap,3);')

define(`EQSP_INIT_CPX_2',`EQSP_INIT_CPX_1 EQSP_INIT_CPX_SRC1')
define(`EQSP_INIT_CPX_3',`EQSP_INIT_CPX_2 EQSP_INIT_CPX_SRC2')
define(`EQSP_INIT_CPX_4',`EQSP_INIT_CPX_3 EQSP_INIT_CPX_SRC3')
define(`EQSP_INIT_CPX_5',`EQSP_INIT_CPX_4 EQSP_INIT_CPX_SRC4')
define(`EQSP_INIT_CCR_3',`EQSP_INIT_CPX_2 EQSP_INIT_SRC2')
define(`EQSP_INIT_CR_2',`EQSP_INIT_CPX_1 EQSP_INIT_SRC1')
define(`EQSP_INIT_RC_2',`EQSP_INIT_1 EQSP_INIT_CPX_SRC1')

define(`EQSP_INIT_QUAT_1',`
	qdst_ptr = (dest_quat *)VA_DEST_PTR(vap);
EQSP_INIT_COUNT
')

define(`EQSP_INIT_QUAT_SRC1',`qs1_ptr = (std_quat *)VA_SRC_PTR(vap,0);')
define(`EQSP_INIT_QUAT_SRC2',`qs2_ptr = (std_quat *)VA_SRC_PTR(vap,1);')
define(`EQSP_INIT_QUAT_SRC3',`qs3_ptr = (std_quat *)VA_SRC_PTR(vap,2);')
define(`EQSP_INIT_QUAT_SRC4',`qs4_ptr = (std_quat *)VA_SRC_PTR(vap,3);')

define(`EQSP_INIT_QUAT_2',`EQSP_INIT_QUAT_1	EQSP_INIT_QUAT_SRC1')
define(`EQSP_INIT_QUAT_3',`EQSP_INIT_QUAT_2	EQSP_INIT_QUAT_SRC2')
define(`EQSP_INIT_QUAT_4',`EQSP_INIT_QUAT_3	EQSP_INIT_QUAT_SRC3')
define(`EQSP_INIT_QUAT_5',`EQSP_INIT_QUAT_4	EQSP_INIT_QUAT_SRC4')
define(`EQSP_INIT_QQR_3',`EQSP_INIT_QUAT_2	EQSP_INIT_SRC2')
define(`EQSP_INIT_QR_2',`EQSP_INIT_QUAT_1	EQSP_INIT_SRC1')
define(`EQSP_INIT_RQ_2',`EQSP_INIT_1	EQSP_INIT_QUAT_SRC1')

define(`EQSP_INIT_DBM_',`EQSP_INIT_DBM')

define(`EQSP_INIT_DBM',`
	dbm_bit = VA_DBM_BIT0(vap);
	dbm_ptr=VA_DEST_PTR(vap);
EQSP_INIT_COUNT
')

define(`EQSP_INIT_SBM',`
	sbm_bit = VA_SBM_BIT0(vap);
	sbm_ptr=VA_SRC_PTR(vap,4);
')

define(`EQSP_INIT_DBM_2SRCS',`EQSP_INIT_DBM EQSP_INIT_2SRCS')
define(`EQSP_INIT_DBM_1SRC',`EQSP_INIT_DBM EQSP_INIT_SRC1')
define(`EQSP_INIT_DBM_SBM',`EQSP_INIT_DBM EQSP_INIT_SBM')
define(`EQSP_INIT_SBM_1',`EQSP_INIT_SBM EQSP_INIT_1')
define(`EQSP_INIT_SBM_2',`EQSP_INIT_SBM EQSP_INIT_2')
define(`EQSP_INIT_SBM_3',`EQSP_INIT_SBM EQSP_INIT_3')
define(`EQSP_INIT_SBM_CPX_1',`EQSP_INIT_SBM EQSP_INIT_CPX_1')
define(`EQSP_INIT_SBM_CPX_2',`EQSP_INIT_SBM EQSP_INIT_CPX_2')
define(`EQSP_INIT_SBM_CPX_3',`EQSP_INIT_SBM EQSP_INIT_CPX_3')
define(`EQSP_INIT_SBM_QUAT_1',`EQSP_INIT_SBM EQSP_INIT_QUAT_1')
define(`EQSP_INIT_SBM_QUAT_2',`EQSP_INIT_SBM EQSP_INIT_QUAT_2')
define(`EQSP_INIT_SBM_QUAT_3',`EQSP_INIT_SBM EQSP_INIT_QUAT_3')


dnl

/* The fast body is pretty simple...  Should we try to unroll loops
 * to take advantage of SSE?  How do we help the compiler to do this?
 */

dnl	SIMPLE_FAST_BODY(name, statement,typ,suffix,extra,debugit)
define(`SIMPLE_FAST_BODY',`

{
	/* simple_fast_body typ = /$3/  suffix = /$4/ */
	dnl FAST_DECLS_##typ##suffix
	dnl FAST_INIT_##typ##suffix
	dnl EXTRA_DECLS_##extra
	FAST_DECLS($3,$4)
	FAST_INIT($3,$4)
	EXTRA_DECLS($5)
	while(fl_ctr-- > 0){
		$6
		$2 ;
dnl		FAST_ADVANCE_##typ##suffix
		FAST_ADVANCE($3,$4)
	}
}
')


dnl	FAST_BODY_CONV_2(name, statement,dsttyp,srctyp)
define(`FAST_BODY_CONV_2',`

{
	$3 *dst_ptr;
	$4 *s1_ptr;
	dimension_t fl_ctr;
	dst_ptr = ($3 *)VA_DEST_PTR(vap);
	s1_ptr = ($4 *)VA_SRC_PTR(vap,0);
	fl_ctr = VA_LENGTH(vap);
	while(fl_ctr-- > 0){
		$2 ;
		FAST_ADVANCE_2
	}
}
')

/* There ought to be a more compact way to do all of this? */

dnl	FAST_BODY_2(name, statement)
define(`FAST_BODY_2',`SIMPLE_FAST_BODY($1,$2,`',2,`',`')')
define(`FAST_BODY_3',`SIMPLE_FAST_BODY($1,$2,`',3,`',`')')
define(`FAST_BODY_4',`SIMPLE_FAST_BODY($1,$2,`',4,`',`')')
define(`FAST_BODY_5',`SIMPLE_FAST_BODY($1,$2,`',5,`',`')')
define(`FAST_BODY_SBM_1',`SIMPLE_FAST_BODY($1,$2,`',SBM_1,`',`')')
define(`FAST_BODY_SBM_2',`SIMPLE_FAST_BODY($1,$2,`',SBM_2,`',`')')
define(`FAST_BODY_SBM_3',`SIMPLE_FAST_BODY($1,$2,`',SBM_3,`',`')')
define(`FAST_BODY_SBM_CPX_1',`SIMPLE_FAST_BODY($1,$2,`',SBM_CPX_1,`',`')')
define(`FAST_BODY_SBM_CPX_2',`SIMPLE_FAST_BODY($1,$2,`',SBM_CPX_2,`',`')')
define(`FAST_BODY_SBM_CPX_3',`SIMPLE_FAST_BODY($1,$2,`',SBM_CPX_3,`',`')')
define(`FAST_BODY_SBM_QUAT_1',`SIMPLE_FAST_BODY($1,$2,`',SBM_QUAT_1,`',`')')
define(`FAST_BODY_SBM_QUAT_2',`SIMPLE_FAST_BODY($1,$2,`',SBM_QUAT_2,`',`')')
define(`FAST_BODY_SBM_QUAT_3',`SIMPLE_FAST_BODY($1,$2,`',SBM_QUAT_3,`',`')')
define(`FAST_BODY_BM_1',`SIMPLE_FAST_BODY($1,$2,`',BM_1,`',`')')
define(`FAST_BODY_DBM_1SRC',`SIMPLE_FAST_BODY($1,$2,`',DBM_1SRC,`',`')')
define(`FAST_BODY_DBM_SBM_',`SIMPLE_FAST_BODY($1,$2,`',DBM_SBM,`',`')')

define(`FAST_BODY_DBM_2SRCS',`SIMPLE_FAST_BODY($1,$2,`',DBM_2SRCS,`',`')')
define(`FAST_BODY_DBM_',`SIMPLE_FAST_BODY($1,$2,`',DBM_,`',`')')
define(`FAST_BODY_1',`SIMPLE_FAST_BODY($1,$2,`',1,`',`')')
define(`FAST_BODY_CPX_1',`SIMPLE_FAST_BODY($1,$2,CPX_,1,`',`')')
define(`FAST_BODY_CPX_2',`SIMPLE_FAST_BODY($1,$2,CPX_,2,`',`')')
define(`FAST_BODY_CPX_2_T2',`SIMPLE_FAST_BODY($1,$2,CPX_,2,T2,`')')
define(`FAST_BODY_CPX_2_T3',`SIMPLE_FAST_BODY($1,$2,CPX_,2,T3,`')')
define(`FAST_BODY_CPX_3',`SIMPLE_FAST_BODY($1,$2,CPX_,3,`',`')')
define(`FAST_BODY_CPX_3_T1',`SIMPLE_FAST_BODY($1,$2,CPX_,3,T1,`')')
define(`FAST_BODY_CPX_3_T2',`SIMPLE_FAST_BODY($1,$2,CPX_,3,T2,`')')
define(`FAST_BODY_CPX_3_T3',`SIMPLE_FAST_BODY($1,$2,CPX_,3,T3,`')')
define(`FAST_BODY_CPX_4',`SIMPLE_FAST_BODY($1,$2,CPX_,4,`',`')')
define(`FAST_BODY_CPX_5',`SIMPLE_FAST_BODY($1,$2,CPX_,5,`',`')')
define(`FAST_BODY_CCR_3',`SIMPLE_FAST_BODY($1,$2,CCR_,3,`',`')')
define(`FAST_BODY_CR_2',`SIMPLE_FAST_BODY($1,$2,CR_,2,`',`')')
define(`FAST_BODY_RC_2',`/* fast_body_rc_2 */SIMPLE_FAST_BODY($1,$2,RC_,2,`',`')')
define(`FAST_BODY_QUAT_1',`SIMPLE_FAST_BODY($1,$2,QUAT_,1,`',`')')
define(`FAST_BODY_QUAT_2',`SIMPLE_FAST_BODY($1,$2,QUAT_,2,`',`')')
define(`FAST_BODY_QUAT_2_T4',`SIMPLE_FAST_BODY($1,$2,QUAT_,2,T4,`')')
define(`FAST_BODY_QUAT_2_T5',`SIMPLE_FAST_BODY($1,$2,QUAT_,2,T5,`')')
define(`FAST_BODY_QUAT_3',`SIMPLE_FAST_BODY($1,$2,QUAT_,3,`',`')')
define(`FAST_BODY_QUAT_3_T4',`SIMPLE_FAST_BODY($1,$2,QUAT_,3,T4,`')')
define(`FAST_BODY_QUAT_3_T5',`SIMPLE_FAST_BODY($1,$2,QUAT_,3,T5,`')')
define(`FAST_BODY_QUAT_4',`SIMPLE_FAST_BODY($1,$2,QUAT_,4,`',`')')
define(`FAST_BODY_QUAT_5',`SIMPLE_FAST_BODY($1,$2,QUAT_,5,`',`')')
define(`FAST_BODY_QQR_3',`SIMPLE_FAST_BODY($1,$2,QQR_,3,`',`')')
define(`FAST_BODY_QR_2',`SIMPLE_FAST_BODY($1,$2,QR_,2,`',`')')
define(`FAST_BODY_RQ_2',`SIMPLE_FAST_BODY($1,$2,RQ_,2,`',`')')


/********************** Section 7 - fast/slow switches **************/

dnl	SHOW_VEC_ARGS(speed)
define(`SHOW_VEC_ARGS',`
	sprintf(DEFAULT_ERROR_STRING,"%s:  vap.va_dst_vp = 0x%lx","$1",(int_for_addr)VA_DEST_PTR(vap);
	NADVISE(DEFAULT_ERROR_STRING);
	sprintf(DEFAULT_ERROR_STRING,"%s:  vap.va_src_vp[0] = 0x%lx","$1",(int_for_addr)VA_SRC_PTR(vap,0) );
	NADVISE(DEFAULT_ERROR_STRING);
	sprintf(DEFAULT_ERROR_STRING,"%s:  vap.va_src_vp[1] = 0x%lx","$1",(int_for_addr)VA_SRC_PTR(vap,1) );
	NADVISE(DEFAULT_ERROR_STRING);
')

dnl  BUG?  no equally-spaced case???

dnl	FAST_SWITCH_CONV( name )
define(`FAST_SWITCH_CONV',`

if( FAST_TEST_2 ){
	XFER_FAST_ARGS_2
	CHAIN_CHECK( FAST_CONV_NAME($1) )
} else if( EQSP_TEST_2 ){
	XFER_EQSP_ARGS_2
	CHAIN_CHECK( EQSP_CONV_NAME($1) )
} else {
	XFER_SLOW_ARGS_2
	CHAIN_CHECK( SLOW_CONV_NAME($1) )
}
')

define(`SHOW_FAST_TEST_',`')
define(`SHOW_FAST_TEST_2SRCS',`')
define(`SHOW_FAST_TEST_1SRC',`')
define(`SHOW_FAST_TEST_4',`')
define(`SHOW_FAST_TEST_5',`')

define(`SHOW_FAST_TEST_1',`
	sprintf(DEFAULT_ERROR_STRING,"fast_test_1:  %d",FAST_TEST_1?1:0);
	NADVISE(DEFAULT_ERROR_STRING);
')

dnl  These macros that use IS_CONTIGUOUS have qsp problems...

define(`SHOW_FAST_TEST_2',`
SHOW_FAST_TEST_1
	sprintf(DEFAULT_ERROR_STRING,"fast_test_src1:  %d",IS_CONTIGUOUS(SRC1_DP)?1:0);
	NADVISE(DEFAULT_ERROR_STRING);
')

define(`SHOW_FAST_TEST_3',`
SHOW_FAST_TEST_2
	sprintf(DEFAULT_ERROR_STRING,"fast_test_src2:  %d",IS_CONTIGUOUS(SRC2_DP)?1:0);
	NADVISE(DEFAULT_ERROR_STRING);
')

dnl	GENERIC_FAST_SWITCH(name,bitmap,typ,scalars,vectors)
define(`GENERIC_FAST_SWITCH',`

	/* generic_fast_switch, bitmap = /$2/ vectors = /$5/ */

dnl if( FAST_TEST_##bitmap##vectors ){
if( FAST_TEST($2,$5) ){
	REPORT_FAST_CALL
	dnl XFER_FAST_ARGS_##bitmap##typ##scalars##vectors
	XFER_FAST_ARGS($2,$3,$4,$5)
	CHAIN_CHECK( FAST_NAME($1) )
dnl } else if( EQSP_TEST_##bitmap##vectors ){
} else if( EQSP_TEST($2,$5) ){
	REPORT_EQSP_CALL
	dnl XFER_EQSP_ARGS_##bitmap##typ##scalars##vectors
	XFER_EQSP_ARGS($2,$3,$4,$5)
	CHAIN_CHECK( EQSP_NAME($1) )
} else {
	REPORT_SLOW_CALL
	XFER_SLOW_ARGS_##bitmap##typ##scalars##vectors
	CHAIN_CHECK( SLOW_NAME($1) )
}
')

/* Why do we need these??? */
dnl	FAST_SWITCH_2( name )
define(`FAST_SWITCH_2',`GENERIC_FAST_SWITCH($1,`',`',`',2)')
define(`FAST_SWITCH_3',`GENERIC_FAST_SWITCH($1,`',`',`',3)')
define(`FAST_SWITCH_4',`GENERIC_FAST_SWITCH($1,`',`',`',4)')
define(`FAST_SWITCH_CPX_3',`GENERIC_FAST_SWITCH($1,`',CPX_,`',3)')
define(`FAST_SWITCH_5',`GENERIC_FAST_SWITCH($1,`',`',`',5)')

#include "veclib/xfer_args.h"

/*********************** Section 8 - object methods ***************/

dnl	_VEC_FUNC_2V_CONV(name,type,statement)
define(`_VEC_FUNC_2V_CONV',`

FF_DECL($1)( LINK_FUNC_ARG_DECLS )
{
	/* FAST_DECLS_1 */
	/* use passed `type' instead of `dest_type' */
	$2 *dst_ptr; dimension_t fl_ctr;
	FAST_DECLS_SRC1
	dst_ptr = ($2 *)VA_DEST_PTR(vap);
	FAST_INIT_COUNT
	FAST_INIT_SRC1
	/* no extra decls */
	while(fl_ctr-- > 0){
		$3 ;
		FAST_ADVANCE_2
	}
}

EF_DECL($1)( LINK_FUNC_ARG_DECLS )
{
	/* EQSP_DECLS_1 */
	$2 *dst_ptr; dimension_t fl_ctr;
	EQSP_DECLS_SRC1
	/* EQSP_INIT_2 */
	/* EQSP_INIT_1 */
	dst_ptr = ($2 *)VA_DEST_PTR(vap);
	EQSP_INIT_COUNT
	EQSP_INIT_SRC1
	/* no extra decls */
	while(fl_ctr-- > 0){
		$3 ;
		EQSP_ADVANCE_2
	}
}

dnl /* GENERIC_SF_DECL(name,statement,bitmap,typ,scalars,vectors,extra) */
SF_DECL($1)( LINK_FUNC_ARG_DECLS )
GENERIC_SLOW_BODY( $1, $3,`DECLARE_BASES_CONV_2($2)',`INIT_BASES_CONV_2($2)',`COPY_BASES_2',`INIT_PTRS_2',`INC_PTRS_2',`INC_BASES_2',`')')


dnl	OBJ_METHOD(name,statement,bitmap,typ,scalars,vectors,extra)
dnl define(`_VEC_FUNC_2V_MIXED',`OBJ_METHOD($1,$2,`',RC_,`',2,`')')
define(`OBJ_METHOD',`
/* obj_method /$1/ BEGIN */
GENERIC_FUNC_DECLS($1,$2,$3,$4,$5,$6,$7)
/* obj_method /$1/ DONE */
')



dnl	OBJ_MOV_METHOD(name,statement,bitmap,typ,scalars,vectors)
define(`OBJ_MOV_METHOD',`
MOV_FUNC_DECLS($1,$2,$3,$4,$5,$6)
')


dnl	_VEC_FUNC_2V_SCAL( name, statement )
define(`_VEC_FUNC_2V_SCAL',`OBJ_METHOD($1,$2,`',`',1S_,2,`')')

define(`_VEC_FUNC_3V',`OBJ_METHOD($1,$2,`',`',`',3,`')')


dnl  These are the kernels...

dnl	_VEC_FUNC_MM_NOCC( name, augment_condition,
dnl					restart_condition, assignment,
dnl					gpu_c1, gpu_c2 )

define(`_VEC_FUNC_MM_NOCC',`
/* vec_func_mm_nocc /$1/ /$2/ /$3/ /$4/ BEGIN */

EXTLOC_FAST_FUNC($1,EXTLOC_STATEMENT($2,$3,$4))
EXTLOC_EQSP_FUNC($1,EXTLOC_STATEMENT($2,$3,$4))
EXTLOC_SLOW_FUNC($1,EXTLOC_STATEMENT($2,$3,$4)) 
/* vec_func_mm_nocc /$1/ /$2/ /$3/ /$4/ DONE */
')


dnl	_VEC_FUNC_2V( name, statement )
define(`_VEC_FUNC_2V',`OBJ_METHOD($1,$2,`',`',`',2,`')')

/* e.g. ramp1d */

dnl	_VEC_FUNC_1V_2SCAL( name, statement, gpu_stat )
define(`_VEC_FUNC_1V_2SCAL',`OBJ_METHOD($1,$2,`',`',2S_,1,`')')

/* e.g. vset */

dnl	_VEC_FUNC_1V_SCAL( name, statement )
define(`_VEC_FUNC_1V_SCAL',`OBJ_METHOD($1,$2,`',`',1S_,1,`')')

/* used to set a bitmap based on a vector test */
/* vvm_gt etc */

dnl	_VEC_FUNC_VVMAP( name, op )
define(`_VEC_FUNC_VVMAP',`OBJ_METHOD($1,SET_DBM_BIT(src1 $2 src2),DBM_,`',`',2SRCS,`')')

dnl	_VEC_FUNC_5V( name, statement )
define(`_VEC_FUNC_5V',`OBJ_METHOD($1,$2,`',`',`',5,`')')

define(`_VEC_FUNC_4V_SCAL',`OBJ_METHOD($1,$2,`',`',1S_,4,`')')

define(`_VEC_FUNC_3V_2SCAL',`OBJ_METHOD($1,$2,`',`',2S_,3,`')')

define(`_VEC_FUNC_2V_3SCAL',`OBJ_METHOD($1,$2,`',`',3S_,2,`')')


define(`_VEC_FUNC_VVSLCT',`OBJ_METHOD($1,$2,SBM_,`',`',3,`')')

define(`_VEC_FUNC_VSSLCT',`OBJ_METHOD($1,$2,SBM_,`',1S_,2,`')')

define(`_VEC_FUNC_SSSLCT',`OBJ_METHOD($1,$2,SBM_,`',2S_,1,`')')

define(`_VEC_FUNC_1V',`OBJ_METHOD($1,$2,`',`',`',1,`')')

/* this is for vmagsq, vatn2:  real result, cpx source */
define(`_VEC_FUNC_2V_MIXED',`OBJ_METHOD($1,$2,`',RC_,`',2,`')')

dnl #define THREE_CPX_VEC_METHOD_T1( name, statement )
define(`_VEC_FUNC_CPX_3V_T1',`OBJ_METHOD($1,$2,`',CPX_,`',3,_T1)')

define(`_VEC_FUNC_CPX_2V',`OBJ_METHOD($1,$2,`',CPX_,1S_,2,`')')

define(`_VEC_FUNC_CPX_2V_T2',`OBJ_METHOD($1,$2,`',CPX_,1S_,2,_T2)')

define(`_VEC_FUNC_CPX_3V_T2',`OBJ_METHOD($1,$2,`',CPX_,`',3,_T2)')

define(`_VEC_FUNC_CPX_3V_T3',`OBJ_METHOD($1,$2,`',CPX_,`',3,_T3)')

define(`_VEC_FUNC_CPX_3V_T1',`OBJ_METHOD($1,$2,`',CPX_,`',3,_T1)')

define(`_VEC_FUNC_CPX_3V',`OBJ_METHOD($1,$2,`',CPX_,`',3,`')')

define(`_VEC_FUNC_CCR_3V',`OBJ_METHOD($1,$2,`',CCR_,`',3,`')')

define(`_VEC_FUNC_CR_1S_2V',`OBJ_METHOD($1,$2,`',CR_,1S_,2,`')')

define(`_VEC_FUNC_CPX_1S_2V',`OBJ_METHOD($1,$2,`',CPX_,1S_,2,`')')

define(`_VEC_FUNC_CPX_1S_2V',`OBJ_METHOD($1,$2,`',CPX_,1S_,2,`')')

define(`_VEC_FUNC_CPX_1S_2V_T2',`OBJ_METHOD($1,$2,`',CPX_,1S_,2,_T2)')

define(`_VEC_FUNC_CPX_1S_2V_T3',`OBJ_METHOD($1,$2,`',CPX_,1S_,2,_T3)')

define(`_VEC_FUNC_CPX_1S_1V',`OBJ_METHOD($1,$2,`',CPX_,1S_,1,`')')

define(`_VEC_FUNC_SBM_CPX_3V',`OBJ_METHOD($1,$2,SBM_,CPX_,`',3,`')')

define(`_VEC_FUNC_SBM_CPX_1S_2V',`OBJ_METHOD($1,$2,SBM_,CPX_,1S_,2,`')')

define(`_VEC_FUNC_SBM_CPX_2S_1V',`OBJ_METHOD($1,$2,SBM_,CPX_,2S_,1,`')')

define(`_VEC_FUNC_SBM_QUAT_3V',`OBJ_METHOD($1,$2,SBM_,QUAT_,`',3,`')')

define(`_VEC_FUNC_SBM_QUAT_1S_2V',`OBJ_METHOD($1,$2,SBM_,QUAT_,1S_,2,`')')

define(`_VEC_FUNC_SBM_QUAT_2S_1V',`OBJ_METHOD($1,$2,SBM_,QUAT_,2S_,1,`')')

dnl #define TWO_QUAT_VEC_METHOD( name,stat )
define(`_VEC_FUNC_QUAT_2V',`OBJ_METHOD($1,$2,`',QUAT_,`',2,`')')

define(`_VEC_FUNC_QUAT_2V_T4',`OBJ_METHOD($1,$2,`',QUAT_,`',2,_T4)')

dnl #define THREE_QUAT_VEC_METHOD( name,stat )
define(`_VEC_FUNC_QUAT_3V',`OBJ_METHOD($1,$2,`',QUAT_,`',3,`')')

define(`_VEC_FUNC_QUAT_3V_T4',`OBJ_METHOD($1,$2,`',QUAT_,`',3,_T4)')

define(`_VEC_FUNC_QUAT_3V_T5',`OBJ_METHOD($1,$2,`',QUAT_,`',3,_T5)')

define(`_VEC_FUNC_QQR_3V',`OBJ_METHOD($1,$2,`',QQR_,`',3,`')')

define(`_VEC_FUNC_QR_1S_2V',`OBJ_METHOD($1,$2,`',QR_,1S_,2,`')')

define(`_VEC_FUNC_QUAT_1S_2V',`OBJ_METHOD($1,$2,`',QUAT_,1S_,2,`')')

define(`_VEC_FUNC_QUAT_1S_2V_T4',`OBJ_METHOD($1,$2,`',QUAT_,1S_,2,_T4)')

define(`_VEC_FUNC_QUAT_1S_2V_T5',`OBJ_METHOD($1,$2,`',QUAT_,1S_,2,_T5)')

define(`_VEC_FUNC_QUAT_1S_1V',`OBJ_METHOD($1,$2,`',QUAT_,1S_,1,`')')


/* PROJECTION_METHOD_2 is for vmaxv, vminv, vsum
 * Destination can be a scalar or we can collapse along any dimension...
 *
 * We have a similar issue for vmaxi, where we wish to return the index
 * of the max...
 */

/* We could do a fast loop when the destination is a scalar... */

/* INDEX_VDATA gets a pointer to the nth element in the array... */
/* It is a linear index, so we have to take it apart... */

dnl	INDEX_VDATA(index)
define(`INDEX_VDATA',`(orig_s1_ptr+($1%INDEX_COUNT(s1_count,0))*IDX_INC(s1inc,0)
+ (($1/INDEX_COUNT(s1_count,0))%INDEX_COUNT(s1_count,1))*IDX_INC(s1inc,1)
+ (($1/(INDEX_COUNT(s1_count,0)*INDEX_COUNT(s1_count,1)))%INDEX_COUNT(s1_count,2))*IDX_INC(s1inc,2) 
+ (($1/(INDEX_COUNT(s1_count,0)*INDEX_COUNT(s1_count,1)*INDEX_COUNT(s1_count,2)))%INDEX_COUNT(s1_count,3))*IDX_INC(s1inc,3) 
+ (($1/(INDEX_COUNT(s1_count,0)*INDEX_COUNT(s1_count,1)*INDEX_COUNT(s1_count,2)*INDEX_COUNT(s1_count,3)))%INDEX_COUNT(s1_count,4))*IDX_INC(s1inc,4))
')

dnl	_VEC_FUNC_2V_PROJ( name, init_statement, statement, gpu_expr )
define(`_VEC_FUNC_2V_PROJ',`

static void SLOW_NAME($1)(LINK_FUNC_ARG_DECLS)
SLOW_BODY_PROJ_2($1,$2,$3)
')




dnl	_VEC_FUNC_CPX_2V_PROJ( name, init_statement, statement, gpu_expr_re, gpu_expr_im )
define(`_VEC_FUNC_CPX_2V_PROJ',`

static void SLOW_NAME($1)(LINK_FUNC_ARG_DECLS)
SLOW_BODY_PROJ_CPX_2($1,$2,$3)
')


dnl	_VEC_FUNC_QUAT_2V_PROJ( name, init_statement, statement, expr_re, expr_im1, expr_im2, expr_im3 )
define(`_VEC_FUNC_QUAT_2V_PROJ',`

static void SLOW_NAME($1)(LINK_FUNC_ARG_DECLS)
SLOW_BODY_PROJ_QUAT_2($1,$2,$3)
')



dnl #define RAMP2D_METHOD(name)
dnl  BUG? can we use the statement args?  now they are not used

dnl	_VEC_FUNC_1V_3SCAL(name,s1,s2,s3)
define(`_VEC_FUNC_1V_3SCAL',`


static void SLOW_NAME($1)( LINK_FUNC_ARG_DECLS )
{
	dimension_t i,j;
	std_type val,row_start_val;
	dest_type *dst_ptr;	/* BUG init me */
	dest_type *row_ptr;

	dst_ptr = (dest_type *)VA_DEST_PTR(vap);
	row_start_val = scalar1_val;
	row_ptr = dst_ptr;
	for(i=0; i < INDEX_COUNT(count,2); i ++ ){
		val = row_start_val;
		dst_ptr = row_ptr;
		for(j=0; j < INDEX_COUNT(count,1); j++ ){
			*dst_ptr = (dest_type) val;

			dst_ptr += IDX_INC(dinc,1);
			val += scalar2_val;
		}
		row_ptr += IDX_INC(dinc,2);
		row_start_val += scalar3_val;
	}
}
')



dnl	_VEC_FUNC_2V_PROJ_IDX( name, init_statement, statement, gpu_s1, gpu_s2 )
define(`_VEC_FUNC_2V_PROJ_IDX',`

static void SLOW_NAME($1)(LINK_FUNC_ARG_DECLS)
SLOW_BODY_PROJ_IDX_2($1,$2,$3)
')





/* PROJECTION_METHOD_3 is for vdot
 * Destination can be a scalar or we can collapse along any dimension...
 */

dnl  complex really not different?
dnl  where is DECLARE_BASES?

dnl	_VEC_FUNC_CPX_3V_PROJ( name, init_statement, statement, gpu_r1, gpu_i1, gpu_r2, gpu_i2 )
define(`_VEC_FUNC_CPX_3V_PROJ',`__VEC_FUNC_3V_PROJ($1,CPX_,$2,$3 )')

dnl	__VEC_FUNC_3V_PROJ( name, typ, init_statement, statement )
define(`__VEC_FUNC_3V_PROJ',`

static void SLOW_NAME($1)(LINK_FUNC_ARG_DECLS)
PROJ3_SLOW_BODY($1,$2,$3,$4)
')



dnl	_VEC_FUNC_3V_PROJ( name, init_statement, statement, gpu_e1, gpu_e2 )
define(`_VEC_FUNC_3V_PROJ',`__VEC_FUNC_3V_PROJ($1,`',$2,$3 )')

/* bitmap conversion from another type */

dnl #define _VEC_FUNC_DBM_1V( name, statement )
dnl 	OBJ_METHOD(name,statement,DBM_,,,1SRC,)

dnl	_VEC_FUNC_DBM_SBM( name, statement )
define(`_VEC_FUNC_DBM_SBM',`OBJ_METHOD($1,$2,DBM_SBM_,`',`',`',`')')

/* bitmap set from a constant */
dnl	_VEC_FUNC_DBM_1S( name, statement )
define(`_VEC_FUNC_DBM_1S',`OBJ_METHOD($1,$2,DBM_,`',1S_,`',`')')

/* used to set a bitmap based on a vector test */
/* vsm_gt etc */

dnl	_VEC_FUNC_VSMAP( name, op )
define(`_VEC_FUNC_VSMAP',`OBJ_METHOD($1,SET_DBM_BIT(src1 $2 scalar1_val),DBM_,`',1S_,1SRC,`')')


define(`scalar1_val',`VA_SCALAR_VAL_STD(vap,0)')
define(`scalar2_val',`VA_SCALAR_VAL_STD(vap,1)')
define(`scalar3_val',`VA_SCALAR_VAL_STD(vap,2)')
define(`cscalar1_val',`VA_SCALAR_VAL_STDCPX(vap,0)')
define(`cscalar2_val',`VA_SCALAR_VAL_STDCPX(vap,1)')
define(`cscalar3_val',`VA_SCALAR_VAL_STDCPX(vap,2)')
define(`qscalar1_val',`VA_SCALAR_VAL_STDQUAT(vap,0)')
define(`qscalar2_val',`VA_SCALAR_VAL_STDQUAT(vap,1)')
define(`qscalar3_val',`VA_SCALAR_VAL_STDQUAT(vap,2)')
define(`count',`VA_DEST_DIMSET(vap)')
define(`s1_count',`VA_SRC1_DIMSET(vap)')
define(`s2_count',`VA_SRC2_DIMSET(vap)')
define(`dbminc',`VA_DEST_INCSET(vap)')
define(`sbminc',`VA_SRC5_INCSET(vap)')
define(`dinc',`VA_DEST_INCSET(vap)')
define(`s1inc',`VA_SRC1_INCSET(vap)')
define(`s2inc',`VA_SRC2_INCSET(vap)')
define(`s3inc',`VA_SRC3_INCSET(vap)')
define(`s4inc',`VA_SRC4_INCSET(vap)')

