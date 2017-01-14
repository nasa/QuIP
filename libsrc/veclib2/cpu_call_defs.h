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

#define BUILD_FOR_CPU
#undef BUILD_FOR_HOST

/* use 64 bit words for data movement when possible.  Pointers have
 * to be aligned to 8 byte boundaries.
 */

#define L_ALIGNMENT(a)		(((int_for_addr)a) & 7)

#define XFER_EQSP_DBM_GPU_INFO	/* nop on cpu */
#define XFER_SLOW_DBM_GPU_INFO	/* nop on cpu */

/***************** Section 1 - definitions **********************/

//#include "calling_args.h"	// declaration args, shared

// some of this stuff is obsolete...
//#define FFT_METHOD_NAME(stem)	TYPED_NAME( _##_fft_##stem )
//#define OBJ_METHOD_NAME(stem)	HOST_TYPED_CALL_NAME( stem , type_code )

#define CONV_METHOD_NAME(stem)	CPU_CALL_NAME( stem )
#define FAST_NAME(stem)		CPU_FAST_CALL_NAME(stem)
#define SLOW_NAME(stem)		CPU_SLOW_CALL_NAME(stem)
#define FAST_CONV_NAME(stem)	CPU_FAST_CALL_NAME(stem)
#define SLOW_CONV_NAME(stem)	CPU_SLOW_CALL_NAME(stem)
#define EQSP_NAME(stem)		CPU_EQSP_CALL_NAME(stem)

#define TYPED_NAME(s)			_TYPED_NAME(type_code,s)
#define _TYPED_NAME(t,s)		__TYPED_NAME(t,s)
#define __TYPED_NAME(t,s)		t##_##s

#define TYPED_STRING(s)		STRINGIFY( TYPED_NAME(s) )

#define dst_dp		OA_DEST(oap)
/*
#define src1_dp		OA_SRC1(oap)
#define src2_dp		OA_SRC2(oap)
#define src3_dp		OA_SRC3(oap)
#define src4_dp		OA_SRC4(oap)
*/
#define SRC_DP(idx)	OA_SRC_OBJ(oap,idx)
#define SRC1_DP		SRC_DP(0)
#define SRC2_DP		SRC_DP(1)
#define SRC3_DP		SRC_DP(2)
#define SRC4_DP		SRC_DP(3)
#define SRC5_DP		SRC_DP(4)

#define bitmap_dst_dp	OA_DEST(oap)
#define bitmap_src_dp	OA_SBM(oap)

#define MAX_DEBUG

#ifdef MAX_DEBUG

#define ANNOUNCE_FUNCTION						\
									\
	if( verbose ){							\
		sprintf(DEFAULT_ERROR_STRING,"BEGIN function %s",db_func_name);	\
		NADVISE(DEFAULT_ERROR_STRING);					\
	}

#define REPORT_OBJ_METHOD_DONE						\
									\
	if( verbose ){							\
		sprintf(DEFAULT_ERROR_STRING,"Function %s DONE",db_func_name);	\
		NADVISE(DEFAULT_ERROR_STRING);					\
	}

#define REPORT_FAST_CALL						\
									\
	if( verbose ){							\
		sprintf(DEFAULT_ERROR_STRING,"Function %s calling fast func",db_func_name);	\
		NADVISE(DEFAULT_ERROR_STRING);					\
	}

#define REPORT_EQSP_CALL						\
									\
	if( verbose ){							\
		sprintf(DEFAULT_ERROR_STRING,"Function %s calling eqsp func",db_func_name);	\
		NADVISE(DEFAULT_ERROR_STRING);					\
	}

#define REPORT_SLOW_CALL						\
									\
	if( verbose ){							\
		sprintf(DEFAULT_ERROR_STRING,"Function %s calling slow func",db_func_name);	\
		NADVISE(DEFAULT_ERROR_STRING);					\
	}


#define DECLARE_FUNC_NAME(name)		const char * db_func_name=STRINGIFY( OBJ_METHOD_NAME(name) );

#else /* ! MAX_DEBUG */

#define ANNOUNCE_FUNCTION
#define REPORT_OBJ_METHOD_DONE
#define REPORT_FAST_CALL
#define REPORT_EQSP_CALL
#define REPORT_SLOW_CALL
#define DECLARE_FUNC_NAME(name)
#define db_func_name NULL

#endif /* ! MAX_DEBUG */



#define CHECK_MATCH(dp1,dp2)						\
									\
	if( ! dp_compatible( dp1, dp2 ) ){				\
		if( UNKNOWN_SHAPE(&dp1->dt_shape) &&			\
				! UNKNOWN_SHAPE(dp2) ){			\
			install_shape(dp1,dp2);				\
		} else if( UNKNOWN_SHAPE(dp2) &&			\
				! UNKNOWN_SHAPE(dp1) ){			\
			install_shape(dp2,dp1);				\
		} else {						\
			sprintf(DEFAULT_ERROR_STRING,				\
"Shape mismatch between objects %s and %s",OBJ_NAME(dp1),OBJ_NAME(dp2));	\
			NWARN(DEFAULT_ERROR_STRING);				\
			return;						\
		}							\
	}


#define OBJ_ARG_CHK_DBM							\
									\
	ANNOUNCE_FUNCTION						\
	if( bitmap_dst_dp == NO_OBJ ){					\
NWARN("OBJ_ARG_CHK_DBM:  Null bitmap destination object!?");		\
		return;							\
	}

#define OBJ_ARG_CHK_DBM_	OBJ_ARG_CHK_DBM

#define OBJ_ARG_CHK_SBM							\
									\
	if( bitmap_src_dp == NO_OBJ ){					\
		NWARN("Null bitmap source object!?");			\
		return;							\
	}


#ifdef CAUTIOUS

#define OBJ_ARG_CHK_1							\
									\
	ANNOUNCE_FUNCTION						\
	OBJ_ARG_CHK(dst_dp,"destination")

#define OBJ_ARG_CHK(dp,string)						\
									\
	if( dp==NO_OBJ ){						\
		sprintf(DEFAULT_ERROR_STRING,				\
			"CAUTIOUS:  Null %s object!?",string);		\
		NERROR1(DEFAULT_ERROR_STRING);				\
		IOS_RETURN						\
	}


#define OBJ_ARG_CHK_SRC1	OBJ_ARG_CHK(SRC1_DP,"first source")
#define OBJ_ARG_CHK_SRC2	OBJ_ARG_CHK(SRC2_DP,"second source")
#define OBJ_ARG_CHK_SRC3	OBJ_ARG_CHK(SRC3_DP,"third source")
#define OBJ_ARG_CHK_SRC4	OBJ_ARG_CHK(SRC4_DP,"fourth source")

#else /* ! CAUTIOUS */

#define OBJ_ARG_CHK_1							\
									\
	ANNOUNCE_FUNCTION

#define OBJ_ARG_CHK_SRC1
#define OBJ_ARG_CHK_SRC2
#define OBJ_ARG_CHK_SRC3
#define OBJ_ARG_CHK_SRC4

#endif /* CAUTIOUS */

#define OBJ_ARG_CHK_2SRCS	OBJ_ARG_CHK_SRC1 OBJ_ARG_CHK_SRC2
#define OBJ_ARG_CHK_DBM_2SRCS	OBJ_ARG_CHK_DBM OBJ_ARG_CHK_2SRCS
#define OBJ_ARG_CHK_DBM_1SRC	OBJ_ARG_CHK_DBM OBJ_ARG_CHK_SRC1
#define OBJ_ARG_CHK_DBM_SBM	OBJ_ARG_CHK_DBM OBJ_ARG_CHK_SBM

#define OBJ_ARG_CHK_
#define OBJ_ARG_CHK_2		OBJ_ARG_CHK_1 OBJ_ARG_CHK_SRC1
#define OBJ_ARG_CHK_3		OBJ_ARG_CHK_2 OBJ_ARG_CHK_SRC2
#define OBJ_ARG_CHK_4		OBJ_ARG_CHK_3 OBJ_ARG_CHK_SRC3
#define OBJ_ARG_CHK_5		OBJ_ARG_CHK_4 OBJ_ARG_CHK_SRC4

#define OBJ_ARG_CHK_SBM_1	OBJ_ARG_CHK_SBM OBJ_ARG_CHK_1
#define OBJ_ARG_CHK_SBM_2	OBJ_ARG_CHK_SBM OBJ_ARG_CHK_2
#define OBJ_ARG_CHK_SBM_3	OBJ_ARG_CHK_SBM OBJ_ARG_CHK_3

/* DONE with CHECK definitions */

/////////////

#define FAST_ADVANCE_SRC1	s1_ptr++ ;
#define FAST_ADVANCE_SRC2	s2_ptr++ ;
#define FAST_ADVANCE_SRC3	s3_ptr++ ;
#define FAST_ADVANCE_SRC4	s4_ptr++ ;
#define FAST_ADVANCE_1		dst_ptr++ ;
#define FAST_ADVANCE_CPX_1	cdst_ptr++ ;
#define FAST_ADVANCE_CPX_SRC1	cs1_ptr++ ;
#define FAST_ADVANCE_CPX_SRC2	cs2_ptr++ ;
#define FAST_ADVANCE_CPX_SRC3	cs3_ptr++ ;
#define FAST_ADVANCE_CPX_SRC4	cs4_ptr++ ;
#define FAST_ADVANCE_QUAT_1	qdst_ptr++ ;
#define FAST_ADVANCE_QUAT_SRC1	qs1_ptr++ ;
#define FAST_ADVANCE_QUAT_SRC2	qs2_ptr++ ;
#define FAST_ADVANCE_QUAT_SRC3	qs3_ptr++ ;
#define FAST_ADVANCE_QUAT_SRC4	qs4_ptr++ ;
//#define FAST_ADVANCE_BITMAP	which_bit ++ ;
#define FAST_ADVANCE_DBM	dbm_bit ++ ;
#define FAST_ADVANCE_SBM	sbm_bit ++ ;
#define FAST_ADVANCE_DBM_SBM	FAST_ADVANCE_DBM FAST_ADVANCE_SBM

#define FAST_ADVANCE_2		FAST_ADVANCE_1 FAST_ADVANCE_SRC1
#define FAST_ADVANCE_3		FAST_ADVANCE_2 FAST_ADVANCE_SRC2
#define FAST_ADVANCE_4		FAST_ADVANCE_3 FAST_ADVANCE_SRC3
#define FAST_ADVANCE_5		FAST_ADVANCE_4 FAST_ADVANCE_SRC4

#define FAST_ADVANCE_CPX_2	FAST_ADVANCE_CPX_1 FAST_ADVANCE_CPX_SRC1
#define FAST_ADVANCE_CPX_3	FAST_ADVANCE_CPX_2 FAST_ADVANCE_CPX_SRC2
#define FAST_ADVANCE_CPX_4	FAST_ADVANCE_CPX_3 FAST_ADVANCE_CPX_SRC3
#define FAST_ADVANCE_CPX_5	FAST_ADVANCE_CPX_4 FAST_ADVANCE_CPX_SRC4
#define FAST_ADVANCE_CCR_3	FAST_ADVANCE_CPX_2 FAST_ADVANCE_SRC2
#define FAST_ADVANCE_CR_2	FAST_ADVANCE_CPX_1 FAST_ADVANCE_SRC1
#define FAST_ADVANCE_RC_2	FAST_ADVANCE_1 FAST_ADVANCE_CPX_SRC1

#define FAST_ADVANCE_QUAT_2	FAST_ADVANCE_QUAT_1 FAST_ADVANCE_QUAT_SRC1
#define FAST_ADVANCE_QUAT_3	FAST_ADVANCE_QUAT_2 FAST_ADVANCE_QUAT_SRC2
#define FAST_ADVANCE_QUAT_4	FAST_ADVANCE_QUAT_3 FAST_ADVANCE_QUAT_SRC3
#define FAST_ADVANCE_QUAT_5	FAST_ADVANCE_QUAT_4 FAST_ADVANCE_QUAT_SRC4
#define FAST_ADVANCE_QQR_3	FAST_ADVANCE_QUAT_2 FAST_ADVANCE_SRC2
#define FAST_ADVANCE_QR_2	FAST_ADVANCE_QUAT_1 FAST_ADVANCE_SRC1
#define FAST_ADVANCE_RQ_2	FAST_ADVANCE_1 FAST_ADVANCE_QUAT_SRC1

#define FAST_ADVANCE_SBM_1	FAST_ADVANCE_SBM FAST_ADVANCE_1
#define FAST_ADVANCE_SBM_2	FAST_ADVANCE_SBM FAST_ADVANCE_2
#define FAST_ADVANCE_SBM_3	FAST_ADVANCE_SBM FAST_ADVANCE_3
#define FAST_ADVANCE_SBM_CPX_1	FAST_ADVANCE_SBM FAST_ADVANCE_CPX_1
#define FAST_ADVANCE_SBM_CPX_2	FAST_ADVANCE_SBM FAST_ADVANCE_CPX_2
#define FAST_ADVANCE_SBM_CPX_3	FAST_ADVANCE_SBM FAST_ADVANCE_CPX_3
#define FAST_ADVANCE_SBM_QUAT_1	FAST_ADVANCE_SBM FAST_ADVANCE_QUAT_1
#define FAST_ADVANCE_SBM_QUAT_2	FAST_ADVANCE_SBM FAST_ADVANCE_QUAT_2
#define FAST_ADVANCE_SBM_QUAT_3	FAST_ADVANCE_SBM FAST_ADVANCE_QUAT_3

#define FAST_ADVANCE_DBM_	FAST_ADVANCE_DBM
#define FAST_ADVANCE_DBM_1SRC	FAST_ADVANCE_DBM FAST_ADVANCE_SRC1
#define FAST_ADVANCE_DBM_SBM	FAST_ADVANCE_DBM FAST_ADVANCE_SBM
#define FAST_ADVANCE_DBM_2SRCS	FAST_ADVANCE_DBM FAST_ADVANCE_2SRCS
#define FAST_ADVANCE_2SRCS	FAST_ADVANCE_SRC1 FAST_ADVANCE_SRC2

/////////////

#define EQSP_ADVANCE_SRC1	s1_ptr += eqsp_src1_inc ;
#define EQSP_ADVANCE_SRC2	s2_ptr += eqsp_src2_inc ;
#define EQSP_ADVANCE_SRC3	s3_ptr += eqsp_src3_inc ;
#define EQSP_ADVANCE_SRC4	s4_ptr += eqsp_src4_inc ;
#define EQSP_ADVANCE_1		dst_ptr += eqsp_dest_inc ;
#define EQSP_ADVANCE_CPX_1	cdst_ptr += eqsp_dest_inc ;
#define EQSP_ADVANCE_CPX_SRC1	cs1_ptr += eqsp_src1_inc ;
#define EQSP_ADVANCE_CPX_SRC2	cs2_ptr += eqsp_src2_inc ;
#define EQSP_ADVANCE_CPX_SRC3	cs3_ptr += eqsp_src3_inc ;
#define EQSP_ADVANCE_CPX_SRC4	cs4_ptr += eqsp_src4_inc ;
#define EQSP_ADVANCE_QUAT_1	qdst_ptr += eqsp_dest_inc ;
#define EQSP_ADVANCE_QUAT_SRC1	qs1_ptr += eqsp_src1_inc ;
#define EQSP_ADVANCE_QUAT_SRC2	qs2_ptr += eqsp_src2_inc ;
#define EQSP_ADVANCE_QUAT_SRC3	qs3_ptr += eqsp_src3_inc ;
#define EQSP_ADVANCE_QUAT_SRC4	qs4_ptr += eqsp_src4_inc ;
//#define EQSP_ADVANCE_BITMAP	which_bit  += eqsp_bit_inc ;
#define EQSP_ADVANCE_DBM	dbm_bit  += eqsp_dbm_inc ;
#define EQSP_ADVANCE_SBM	sbm_bit  += eqsp_sbm_inc ;
#define EQSP_ADVANCE_DBM_SBM	EQSP_ADVANCE_DBM EQSP_ADVANCE_SBM

#define EQSP_ADVANCE_2		EQSP_ADVANCE_1 EQSP_ADVANCE_SRC1
#define EQSP_ADVANCE_3		EQSP_ADVANCE_2 EQSP_ADVANCE_SRC2
#define EQSP_ADVANCE_4		EQSP_ADVANCE_3 EQSP_ADVANCE_SRC3
#define EQSP_ADVANCE_5		EQSP_ADVANCE_4 EQSP_ADVANCE_SRC4

#define EQSP_ADVANCE_CPX_2	EQSP_ADVANCE_CPX_1 EQSP_ADVANCE_CPX_SRC1
#define EQSP_ADVANCE_CPX_3	EQSP_ADVANCE_CPX_2 EQSP_ADVANCE_CPX_SRC2
#define EQSP_ADVANCE_CPX_4	EQSP_ADVANCE_CPX_3 EQSP_ADVANCE_CPX_SRC3
#define EQSP_ADVANCE_CPX_5	EQSP_ADVANCE_CPX_4 EQSP_ADVANCE_CPX_SRC4
#define EQSP_ADVANCE_CCR_3	EQSP_ADVANCE_CPX_2 EQSP_ADVANCE_SRC2
#define EQSP_ADVANCE_CR_2	EQSP_ADVANCE_CPX_1 EQSP_ADVANCE_SRC1
#define EQSP_ADVANCE_RC_2	EQSP_ADVANCE_1 EQSP_ADVANCE_CPX_SRC1

#define EQSP_ADVANCE_QUAT_2	EQSP_ADVANCE_QUAT_1 EQSP_ADVANCE_QUAT_SRC1
#define EQSP_ADVANCE_QUAT_3	EQSP_ADVANCE_QUAT_2 EQSP_ADVANCE_QUAT_SRC2
#define EQSP_ADVANCE_QUAT_4	EQSP_ADVANCE_QUAT_3 EQSP_ADVANCE_QUAT_SRC3
#define EQSP_ADVANCE_QUAT_5	EQSP_ADVANCE_QUAT_4 EQSP_ADVANCE_QUAT_SRC4
#define EQSP_ADVANCE_QQR_3	EQSP_ADVANCE_QUAT_2 EQSP_ADVANCE_SRC2
#define EQSP_ADVANCE_QR_2	EQSP_ADVANCE_QUAT_1 EQSP_ADVANCE_SRC1
#define EQSP_ADVANCE_RQ_2	EQSP_ADVANCE_1 EQSP_ADVANCE_QUAT_SRC1

#define EQSP_ADVANCE_SBM_1	EQSP_ADVANCE_SBM EQSP_ADVANCE_1
#define EQSP_ADVANCE_SBM_2	EQSP_ADVANCE_SBM EQSP_ADVANCE_2
#define EQSP_ADVANCE_SBM_3	EQSP_ADVANCE_SBM EQSP_ADVANCE_3
#define EQSP_ADVANCE_SBM_CPX_1	EQSP_ADVANCE_SBM EQSP_ADVANCE_CPX_1
#define EQSP_ADVANCE_SBM_CPX_2	EQSP_ADVANCE_SBM EQSP_ADVANCE_CPX_2
#define EQSP_ADVANCE_SBM_CPX_3	EQSP_ADVANCE_SBM EQSP_ADVANCE_CPX_3
#define EQSP_ADVANCE_SBM_QUAT_1	EQSP_ADVANCE_SBM EQSP_ADVANCE_QUAT_1
#define EQSP_ADVANCE_SBM_QUAT_2	EQSP_ADVANCE_SBM EQSP_ADVANCE_QUAT_2
#define EQSP_ADVANCE_SBM_QUAT_3	EQSP_ADVANCE_SBM EQSP_ADVANCE_QUAT_3

#define EQSP_ADVANCE_DBM_	EQSP_ADVANCE_DBM
#define EQSP_ADVANCE_DBM_1SRC	EQSP_ADVANCE_DBM EQSP_ADVANCE_SRC1
#define EQSP_ADVANCE_DBM_SBM	EQSP_ADVANCE_DBM EQSP_ADVANCE_SBM
#define EQSP_ADVANCE_DBM_2SRCS	EQSP_ADVANCE_DBM EQSP_ADVANCE_2SRCS
#define EQSP_ADVANCE_2SRCS	EQSP_ADVANCE_SRC1 EQSP_ADVANCE_SRC2

/////////////

#define EXTRA_DECLS_	/* nothing */
#define EXTRA_DECLS_T1	dest_type r,theta,arg;
#define EXTRA_DECLS_T2	dest_cpx tmpc;
#define EXTRA_DECLS_T3	dest_cpx tmpc; dest_type tmp_denom;
#define EXTRA_DECLS_T4	dest_quat tmpq;
#define EXTRA_DECLS_T5	dest_quat tmpq; dest_type tmp_denom;

/* Stuff for projection loops */

/* was:
 *	count_type loop_count[N_DIMENSIONS];		\
 */

// BUG - why are we allocating with NEW_DIMSET?

#define DECLARE_LOOP_COUNT				\
	int i_dim;					\
	/*Dimension_Set *loop_count=NEW_DIMSET;*/	\
	Dimension_Set lc_ds, *loop_count=(&lc_ds);

#define PROJ_LOOP_DECLS_2				\
							\
	DECLARE_BASES_2					\
	DECLARE_LOOP_COUNT

#define PROJ_LOOP_DECLS_CPX_2				\
							\
	DECLARE_BASES_CPX_2				\
	DECLARE_LOOP_COUNT

#define PROJ_LOOP_DECLS_QUAT_2				\
							\
	DECLARE_BASES_QUAT_2				\
	DECLARE_LOOP_COUNT

#define PROJ_LOOP_DECLS_IDX_2				\
							\
	DECLARE_BASES_IDX_2				\
	DECLARE_LOOP_COUNT				\
	std_type *tmp_ptr;				\
	std_type *orig_s1_ptr;				\
	index_type this_index;

#define PROJ_LOOP_DECLS_3				\
							\
	DECLARE_BASES_3					\
	DECLARE_LOOP_COUNT

#define PROJ_LOOP_DECLS_CPX_3				\
							\
	DECLARE_BASES_CPX_3				\
	DECLARE_LOOP_COUNT

#define PROJ_LOOP_DECLS_QUAT_3				\
							\
	DECLARE_BASES_QUAT_3				\
	DECLARE_LOOP_COUNT

#define PROJ_LOOP_DECLS_IDX_3				\
							\
	DECLARE_BASES_IDX_3				\
	DECLARE_LOOP_COUNT				\
	std_type *tmp_ptr;				\
	index_type this_index;


#define INIT_LOOP_COUNT					\
	for(i_dim=0;i_dim<N_DIMENSIONS;i_dim++) ASSIGN_IDX_COUNT(loop_count,i_dim,1);

/* INC_BASE is what we do at the end of a nested loop... */

#define INC_BASE(which_dim,base_array,inc_array)		\
								\
	base_array[which_dim-1] += IDX_INC(inc_array,which_dim);

#define INC_BASES_1(which_dim)		INC_BASE(which_dim,dst_base,dinc)
#define INC_BASES_SRC1(which_dim)	INC_BASE(which_dim,s1_base,s1inc)
#define INC_BASES_SRC2(which_dim)	INC_BASE(which_dim,s2_base,s2inc)
#define INC_BASES_SRC3(which_dim)	INC_BASE(which_dim,s3_base,s3inc)
#define INC_BASES_SRC4(which_dim)	INC_BASE(which_dim,s4_base,s4inc)

#define INC_BASES_CPX_1(which_dim)	INC_BASE(which_dim,cdst_base,dinc)
#define INC_BASES_CPX_SRC1(which_dim)	INC_BASE(which_dim,cs1_base,s1inc)
#define INC_BASES_CPX_SRC2(which_dim)	INC_BASE(which_dim,cs2_base,s2inc)
#define INC_BASES_CPX_SRC3(which_dim)	INC_BASE(which_dim,cs3_base,s3inc)
#define INC_BASES_CPX_SRC4(which_dim)	INC_BASE(which_dim,cs4_base,s4inc)

#define INC_BASES_QUAT_1(which_dim)	INC_BASE(which_dim,qdst_base,dinc)
#define INC_BASES_QUAT_SRC1(which_dim)	INC_BASE(which_dim,qs1_base,s1inc)
#define INC_BASES_QUAT_SRC2(which_dim)	INC_BASE(which_dim,qs2_base,s2inc)
#define INC_BASES_QUAT_SRC3(which_dim)	INC_BASE(which_dim,qs3_base,s3inc)
#define INC_BASES_QUAT_SRC4(which_dim)	INC_BASE(which_dim,qs4_base,s4inc)

//#define INC_BASES_SBM(which_dim)	INC_BASE(which_dim,sbm_bit0,sbminc)
//#define INC_BASES_DBM(which_dim)	INC_BASE(which_dim,dbm_bit0,dbminc)
#define INC_BASES_SBM(which_dim)	INC_BASE(which_dim,sbm_base,sbminc)
#define INC_BASES_DBM(which_dim)	INC_BASE(which_dim,dbm_base,dbminc)
#define INC_BASES_DBM_			INC_BASES_DBM

#define INC_BASES_2(which_dim)	INC_BASES_1(which_dim) INC_BASES_SRC1(which_dim)
#define INC_BASES_3(which_dim)	INC_BASES_2(which_dim) INC_BASES_SRC2(which_dim)
#define INC_BASES_4(which_dim)	INC_BASES_3(which_dim) INC_BASES_SRC3(which_dim)
#define INC_BASES_5(which_dim)	INC_BASES_4(which_dim) INC_BASES_SRC4(which_dim)

#define INC_BASES_CPX_2(which_dim)	INC_BASES_CPX_1(which_dim) INC_BASES_CPX_SRC1(which_dim)
#define INC_BASES_CPX_3(which_dim)	INC_BASES_CPX_2(which_dim) INC_BASES_CPX_SRC2(which_dim)
#define INC_BASES_CPX_4(which_dim)	INC_BASES_CPX_3(which_dim) INC_BASES_CPX_SRC3(which_dim)
#define INC_BASES_CPX_5(which_dim)	INC_BASES_CPX_4(which_dim) INC_BASES_CPX_SRC4(which_dim)
#define INC_BASES_CCR_3(which_dim)	INC_BASES_CPX_2(which_dim) INC_BASES_SRC2(which_dim)
#define INC_BASES_CR_2(which_dim)	INC_BASES_CPX_1(which_dim) INC_BASES_SRC1(which_dim)
#define INC_BASES_RC_2(which_dim)	INC_BASES_1(which_dim) INC_BASES_CPX_SRC1(which_dim)

#define INC_BASES_QUAT_2(which_dim)	INC_BASES_QUAT_1(which_dim) INC_BASES_QUAT_SRC1(which_dim)
#define INC_BASES_QUAT_3(which_dim)	INC_BASES_QUAT_2(which_dim) INC_BASES_QUAT_SRC2(which_dim)
#define INC_BASES_QUAT_4(which_dim)	INC_BASES_QUAT_3(which_dim) INC_BASES_QUAT_SRC3(which_dim)
#define INC_BASES_QUAT_5(which_dim)	INC_BASES_QUAT_4(which_dim) INC_BASES_QUAT_SRC4(which_dim)
#define INC_BASES_QQR_3(which_dim)	INC_BASES_QUAT_2(which_dim) INC_BASES_SRC2(which_dim)
#define INC_BASES_QR_2(which_dim)	INC_BASES_QUAT_1(which_dim) INC_BASES_SRC1(which_dim)
#define INC_BASES_RQ_2(which_dim)	INC_BASES_1(which_dim) INC_BASES_QUAT_SRC1(which_dim)

#define INC_BASES_IDX_2(which_dim)	INC_BASES_2(which_dim) INC_BASES_IDX(which_dim)
#define INC_BASES_IDX(which_dim)	index_base[which_dim-1] += INDEX_COUNT(s1_count,which_dim-1);

#define INC_BASES_X_2		INC_BASES_2

#define INC_XXX_BASE(which_dim,base_array,inc_array,count_array)	\
									\
	base_array[which_dim-1] += inc_array[which_dim] / count_array[0];

#define INC_XXX_DST_BASE(which_dim)	INC_XXX_BASE(which_dim,dst_base,dinc,count)
#define INC_XXX_SRC1_BASE(which_dim)	INC_XXX_BASE(which_dim,s1_base,s1inc,s1_count)
#define INC_XXX_SRC2_BASE(which_dim)	INC_XXX_BASE(which_dim,s2_base,s2inc,s2_count)
#define INC_XXX_SRC3_BASE(which_dim)	INC_XXX_BASE(which_dim,s3_base,s3inc,s3_count)
#define INC_XXX_SRC4_BASE(which_dim)	INC_XXX_BASE(which_dim,s4_base,s4inc,s4_count)

#define INC_BASES_XXX_1(which_dim)	INC_XXX_DST_BASE(which_dim)
#define INC_BASES_XXX_2(which_dim)	INC_BASES_XXX_1(which_dim)	\
						INC_XXX_SRC1_BASE(which_dim)

#define INC_BASES_XXX_3(which_dim)	INC_BASES_XXX_2(which_dim)	\
						INC_XXX_SRC2_BASE(which_dim)


#define INC_BASES_2SRCS(index)		INC_BASES_SRC1(index) INC_BASES_SRC2(index)
#define INC_BASES_DBM_1SRC( index )	INC_BASES_DBM(index) INC_BASES_SRC1(index)
#define INC_BASES_DBM_SBM( index )	INC_BASES_DBM(index) INC_BASES_SBM(index)
#define INC_BASES_DBM_2SRCS( index )	INC_BASES_DBM(index) INC_BASES_2SRCS(index)
#define COPY_BASES_DBM_1SRC(index)	COPY_BASES_DBM(index) COPY_BASES_SRC1(index)
#define COPY_BASES_DBM_SBM(index)	COPY_BASES_DBM(index) COPY_BASES_SBM(index)
#define COPY_BASES_2SRCS(index)		COPY_BASES_SRC1(index) COPY_BASES_SRC2(index)
#define COPY_BASES_DBM_2SRCS(index)	COPY_BASES_DBM(index) COPY_BASES_2SRCS(index)

#define COPY_BASES_DBM_			COPY_BASES_DBM


#define INC_BASES_SBM_1(index)		INC_BASES_1(index) INC_BASES_SBM(index)
#define INC_BASES_SBM_2(index)		INC_BASES_2(index) INC_BASES_SBM(index)
#define INC_BASES_SBM_3(index)		INC_BASES_3(index) INC_BASES_SBM(index)
#define INC_BASES_SBM_CPX_1(index)	INC_BASES_CPX_1(index) INC_BASES_SBM(index)
#define INC_BASES_SBM_CPX_2(index)	INC_BASES_CPX_2(index) INC_BASES_SBM(index)
#define INC_BASES_SBM_CPX_3(index)	INC_BASES_CPX_3(index) INC_BASES_SBM(index)
#define INC_BASES_SBM_QUAT_1(index)	INC_BASES_QUAT_1(index) INC_BASES_SBM(index)
#define INC_BASES_SBM_QUAT_2(index)	INC_BASES_QUAT_2(index) INC_BASES_SBM(index)
#define INC_BASES_SBM_QUAT_3(index)	INC_BASES_QUAT_3(index) INC_BASES_SBM(index)
#define INC_BASES_SBM_XXX_1(index)	INC_BASES_XXX_1(index) INC_BASES_SBM(index)
#define INC_BASES_SBM_XXX_2(index)	INC_BASES_XXX_2(index) INC_BASES_SBM(index)
#define INC_BASES_SBM_XXX_3(index)	INC_BASES_XXX_3(index) INC_BASES_SBM(index)



#define INIT_BASE( type, which_dim, base_array, dp )			\
									\
	if( (which_dim) < 4 )						\
		(base_array)[which_dim-1] = (base_array)[which_dim];	\
	else								\
		(base_array)[which_dim-1] = (type *)OBJ_DATA_PTR(dp);


#define INIT_INDEX_BASE( which_dim )					\
									\
	if( (which_dim) < 4 )						\
		index_base[which_dim-1] = index_base[which_dim];	\
	else								\
		index_base[which_dim-1] = 0;



/* #define INIT_CPX_INDEX_3 */		/* what is this for? */

#define INIT_PTRS_1		dst_ptr = dst_base[0];
#define INIT_PTRS_SRC1		s1_ptr = s1_base[0];	/* pixel base */
#define INIT_PTRS_SRC2		s2_ptr = s2_base[0];	/* pixel base */
#define INIT_PTRS_SRC3		s3_ptr = s3_base[0];	/* pixel base */
#define INIT_PTRS_SRC4		s4_ptr = s4_base[0];	/* pixel base */

#define INIT_PTRS_CPX_1		cdst_ptr = cdst_base[0];
#define INIT_PTRS_CPX_SRC1	cs1_ptr = cs1_base[0];	/* pixel base */
#define INIT_PTRS_CPX_SRC2	cs2_ptr = cs2_base[0];	/* pixel base */
#define INIT_PTRS_CPX_SRC3	cs3_ptr = cs3_base[0];	/* pixel base */
#define INIT_PTRS_CPX_SRC4	cs4_ptr = cs4_base[0];	/* pixel base */

#define INIT_PTRS_QUAT_1	qdst_ptr = qdst_base[0];
#define INIT_PTRS_QUAT_SRC1	qs1_ptr = qs1_base[0];	/* pixel base */
#define INIT_PTRS_QUAT_SRC2	qs2_ptr = qs2_base[0];	/* pixel base */
#define INIT_PTRS_QUAT_SRC3	qs3_ptr = qs3_base[0];	/* pixel base */
#define INIT_PTRS_QUAT_SRC4	qs4_ptr = qs4_base[0];	/* pixel base */

//#define INIT_PTRS_SBM		sbm_bit = sbm_bit0[0];
//#define INIT_PTRS_DBM		dbm_bit = dbm_bit0[0];
#define INIT_PTRS_SBM		sbm_bit = sbm_base[0];
#define INIT_PTRS_DBM		dbm_bit = dbm_base[0];
#define INIT_PTRS_DBM_		INIT_PTRS_DBM

#define INIT_PTRS_DBM_1SRC	INIT_PTRS_DBM INIT_PTRS_SRC1
#define INIT_PTRS_DBM_SBM	INIT_PTRS_DBM INIT_PTRS_SBM
#define INIT_PTRS_2SRCS		INIT_PTRS_SRC1 INIT_PTRS_SRC2
#define INIT_PTRS_DBM_2SRCS	INIT_PTRS_DBM INIT_PTRS_2SRCS

#define INIT_PTRS_SBM_1		INIT_PTRS_1 INIT_PTRS_SBM
#define INIT_PTRS_SBM_2		INIT_PTRS_2 INIT_PTRS_SBM
#define INIT_PTRS_SBM_3		INIT_PTRS_3 INIT_PTRS_SBM
#define INIT_PTRS_SBM_CPX_1	INIT_PTRS_CPX_1 INIT_PTRS_SBM
#define INIT_PTRS_SBM_CPX_2	INIT_PTRS_CPX_2 INIT_PTRS_SBM
#define INIT_PTRS_SBM_CPX_3	INIT_PTRS_CPX_3 INIT_PTRS_SBM
#define INIT_PTRS_SBM_QUAT_1	INIT_PTRS_QUAT_1 INIT_PTRS_SBM
#define INIT_PTRS_SBM_QUAT_2	INIT_PTRS_QUAT_2 INIT_PTRS_SBM
#define INIT_PTRS_SBM_QUAT_3	INIT_PTRS_QUAT_3 INIT_PTRS_SBM

#define INIT_PTRS_2		INIT_PTRS_1 INIT_PTRS_SRC1
#define INIT_PTRS_3		INIT_PTRS_2 INIT_PTRS_SRC2
#define INIT_PTRS_4		INIT_PTRS_3 INIT_PTRS_SRC3
#define INIT_PTRS_5		INIT_PTRS_4 INIT_PTRS_SRC4

#define INIT_PTRS_CPX_2		INIT_PTRS_CPX_1 INIT_PTRS_CPX_SRC1
#define INIT_PTRS_CPX_3		INIT_PTRS_CPX_2 INIT_PTRS_CPX_SRC2
#define INIT_PTRS_CPX_4		INIT_PTRS_CPX_3 INIT_PTRS_CPX_SRC3
#define INIT_PTRS_CPX_5		INIT_PTRS_CPX_4 INIT_PTRS_CPX_SRC4
#define INIT_PTRS_CCR_3		INIT_PTRS_CPX_2 INIT_PTRS_SRC2
#define INIT_PTRS_CR_2		INIT_PTRS_CPX_1 INIT_PTRS_SRC1
#define INIT_PTRS_RC_2		INIT_PTRS_1 INIT_PTRS_CPX_SRC1

#define INIT_PTRS_QUAT_2	INIT_PTRS_QUAT_1 INIT_PTRS_QUAT_SRC1
#define INIT_PTRS_QUAT_3	INIT_PTRS_QUAT_2 INIT_PTRS_QUAT_SRC2
#define INIT_PTRS_QUAT_4	INIT_PTRS_QUAT_3 INIT_PTRS_QUAT_SRC3
#define INIT_PTRS_QUAT_5	INIT_PTRS_QUAT_4 INIT_PTRS_QUAT_SRC4
#define INIT_PTRS_QQR_3		INIT_PTRS_QUAT_2 INIT_PTRS_SRC2
#define INIT_PTRS_QR_2		INIT_PTRS_QUAT_1 INIT_PTRS_SRC1
#define INIT_PTRS_RQ_2		INIT_PTRS_1 INIT_PTRS_QUAT_SRC1

#define INIT_PTRS_IDX_1		INIT_PTRS_1 this_index = index_base[0];
#define INIT_PTRS_IDX_2		INIT_PTRS_IDX_1 INIT_PTRS_SRC1


#define INIT_PTRS_XXX_1		INIT_PTRS_1
#define INIT_PTRS_XXX_2		INIT_PTRS_2
#define INIT_PTRS_X_2		INIT_PTRS_2
#define INIT_PTRS_XXX_3		INIT_PTRS_3
#define INIT_PTRS_SBM_XXX_3	INIT_PTRS_SBM_3
#define INIT_PTRS_SBM_XXX_2	INIT_PTRS_SBM_2
#define INIT_PTRS_SBM_XXX_1	INIT_PTRS_SBM_1

#define INC_PTRS_SRC1		s1_ptr += IDX_INC(s1inc,0);
#define INC_PTRS_SRC2		s2_ptr += IDX_INC(s2inc,0);
#define INC_PTRS_SRC3		s3_ptr += IDX_INC(s3inc,0);
#define INC_PTRS_SRC4		s4_ptr += IDX_INC(s4inc,0);
/* BUG? here we seem to assume that all bitmaps are contiguous - but
 * dobj allows bitmap subimages...
 */
#define INC_PTRS_SBM		sbm_bit++;
#define INC_PTRS_DBM		dbm_bit++;
#define INC_PTRS_DBM_		INC_PTRS_DBM

#define INC_PTRS_DBM_1SRC	INC_PTRS_DBM INC_PTRS_SRC1
#define INC_PTRS_DBM_SBM	INC_PTRS_DBM INC_PTRS_SBM
#define INC_PTRS_2SRCS		INC_PTRS_SRC1 INC_PTRS_SRC2
#define INC_PTRS_DBM_2SRCS	INC_PTRS_DBM INC_PTRS_2SRCS

#define INC_PTRS_SBM_1		INC_PTRS_1 INC_PTRS_SBM
#define INC_PTRS_SBM_2		INC_PTRS_2 INC_PTRS_SBM
#define INC_PTRS_SBM_3		INC_PTRS_3 INC_PTRS_SBM
#ifdef NOT_USED_FOO
#define INC_PTRS_SBM_CPX_1	INC_PTRS_CPX_1 INC_PTRS_SBM
#define INC_PTRS_SBM_CPX_2	INC_PTRS_CPX_2 INC_PTRS_SBM
#define INC_PTRS_SBM_CPX_3	INC_PTRS_CPX_3 INC_PTRS_SBM
#define INC_PTRS_SBM_QUAT_1	INC_PTRS_QUAT_1 INC_PTRS_SBM
#define INC_PTRS_SBM_QUAT_2	INC_PTRS_QUAT_2 INC_PTRS_SBM
#define INC_PTRS_SBM_QUAT_3	INC_PTRS_QUAT_3 INC_PTRS_SBM
#endif /* NOT_USED_FOO */

#define INC_PTRS_1		dst_ptr += IDX_INC(dinc,0);
#define INC_PTRS_2		INC_PTRS_1 INC_PTRS_SRC1
#define INC_PTRS_3		INC_PTRS_2 INC_PTRS_SRC2
#define INC_PTRS_4		INC_PTRS_3 INC_PTRS_SRC3
#define INC_PTRS_5		INC_PTRS_4 INC_PTRS_SRC4

#define INC_PTRS_IDX_2		INC_PTRS_2 this_index++;
#define INC_PTRS_X_2		INC_PTRS_2

/* compiler BUG?
 * These macros were originally written with left shifts, but
 * even when a u_long is 64 bits, we cannot shift left by more than 31!?
 * SOLVED - 1<<n assumes that 1 is an "int" e.g. 32 bits
 * Use 1L instead!
 *
 * dbm_bit counts the bits from the start of the object
 */

#define SET_DBM_BIT( condition )					\
									\
	if( condition )							\
		*(dbm_ptr + (dbm_bit/BITS_PER_BITMAP_WORD)) |=		\
			NUMBERED_BIT(dbm_bit); 				\
	else								\
		*(dbm_ptr + (dbm_bit/BITS_PER_BITMAP_WORD)) &=		\
			~ NUMBERED_BIT(dbm_bit);

#define DEBUG_SBM_	\
sprintf(DEFAULT_ERROR_STRING,"sbm_ptr = 0x%lx   sbm_bit = %d",\
(int_for_addr)sbm_ptr,sbm_bit);\
NADVISE(DEFAULT_ERROR_STRING);

#define DEBUG_DBM_	\
sprintf(DEFAULT_ERROR_STRING,"dbm_ptr = 0x%lx   dbm_bit = %d",\
(int_for_addr)dbm_ptr,dbm_bit);\
NADVISE(DEFAULT_ERROR_STRING);

#define DEBUG_DBM_1SRC		DEBUG_DBM_		\
				DEBUG_SRC1

#define srcbit		((*(sbm_ptr + (sbm_bit/BITS_PER_BITMAP_WORD)))\
			& NUMBERED_BIT(sbm_bit))


#define INIT_BASES_X_1(dsttyp)	dst_base[3]=(dsttyp *)VA_DEST_PTR(vap);

#define INIT_BASES_X_SRC1(srctyp)	s1_base[3]=(srctyp *)VA_SRC_PTR(vap,0);

#define INIT_BASES_CONV_1(type)	dst_base[3]=(type *)VA_DEST_PTR(vap);
#define INIT_BASES_1		dst_base[3]=(dest_type *)VA_DEST_PTR(vap);
#define INIT_BASES_SRC1		s1_base[3]=(std_type *)VA_SRC_PTR(vap,0);
#define INIT_BASES_SRC2		s2_base[3]=(std_type *)VA_SRC_PTR(vap,1);
#define INIT_BASES_SRC3		s3_base[3]=(std_type *)VA_SRC_PTR(vap,2);
#define INIT_BASES_SRC4		s4_base[3]=(std_type *)VA_SRC_PTR(vap,3);

#define INIT_BASES_CPX_1	cdst_base[3]=(dest_cpx *)VA_DEST_PTR(vap);
#define INIT_BASES_CPX_SRC1	cs1_base[3]=(std_cpx *)VA_SRC_PTR(vap,0);
#define INIT_BASES_CPX_SRC2	cs2_base[3]=(std_cpx *)VA_SRC_PTR(vap,1);
#define INIT_BASES_CPX_SRC3	cs3_base[3]=(std_cpx *)VA_SRC_PTR(vap,2);
#define INIT_BASES_CPX_SRC4	cs4_base[3]=(std_cpx *)VA_SRC_PTR(vap,3);

#define INIT_BASES_QUAT_1	qdst_base[3]=(dest_quat *)VA_DEST_PTR(vap);
#define INIT_BASES_QUAT_SRC1	qs1_base[3]=(std_quat *)VA_SRC_PTR(vap,0);
#define INIT_BASES_QUAT_SRC2	qs2_base[3]=(std_quat *)VA_SRC_PTR(vap,1);
#define INIT_BASES_QUAT_SRC3	qs3_base[3]=(std_quat *)VA_SRC_PTR(vap,2);
#define INIT_BASES_QUAT_SRC4	qs4_base[3]=(std_quat *)VA_SRC_PTR(vap,3);

#define INIT_BASES_IDX_1	INIT_BASES_X_1(index_type)  index_base[3]=0;
#define INIT_BASES_IDX_2	INIT_BASES_IDX_1 INIT_BASES_SRC1


#define INIT_BASES_X_2(dsttyp,srctyp)		INIT_BASES_X_1(dsttyp) INIT_BASES_X_SRC1(srctyp)

#define INIT_BASES_CONV_2(type)	INIT_BASES_CONV_1(type) INIT_BASES_SRC1
#define INIT_BASES_2		INIT_BASES_1 INIT_BASES_SRC1
#define INIT_BASES_3		INIT_BASES_2 INIT_BASES_SRC2
#define INIT_BASES_4		INIT_BASES_3 INIT_BASES_SRC3
#define INIT_BASES_5		INIT_BASES_4 INIT_BASES_SRC4

#define INIT_BASES_CPX_2	INIT_BASES_CPX_1 INIT_BASES_CPX_SRC1
#define INIT_BASES_CPX_3	INIT_BASES_CPX_2 INIT_BASES_CPX_SRC2
#define INIT_BASES_CPX_4	INIT_BASES_CPX_3 INIT_BASES_CPX_SRC3
#define INIT_BASES_CPX_5	INIT_BASES_CPX_4 INIT_BASES_CPX_SRC4
#define INIT_BASES_CCR_3	INIT_BASES_CPX_2 INIT_BASES_SRC2
#define INIT_BASES_CR_2		INIT_BASES_CPX_1 INIT_BASES_SRC1
#define INIT_BASES_RC_2		INIT_BASES_1 INIT_BASES_CPX_SRC1

#define INIT_BASES_QUAT_2	INIT_BASES_QUAT_1 INIT_BASES_QUAT_SRC1
#define INIT_BASES_QUAT_3	INIT_BASES_QUAT_2 INIT_BASES_QUAT_SRC2
#define INIT_BASES_QUAT_4	INIT_BASES_QUAT_3 INIT_BASES_QUAT_SRC3
#define INIT_BASES_QUAT_5	INIT_BASES_QUAT_4 INIT_BASES_QUAT_SRC4
#define INIT_BASES_QQR_3	INIT_BASES_QUAT_2 INIT_BASES_SRC2
#define INIT_BASES_QR_2		INIT_BASES_QUAT_1 INIT_BASES_SRC1
#define INIT_BASES_RQ_2		INIT_BASES_1 INIT_BASES_QUAT_SRC1

#define INIT_BASES_DBM_	INIT_BASES_DBM

/* We don't actually use the bases for destination bitmaps...
 * Should we?
 *
 * dbm_base used to be the pointer, but now it is bit0
 *
 * Why are the indices three and not 4?
 */

#define INIT_BASES_DBM			\
	dbm_ptr= VA_DEST_PTR(vap);	\
	/*dbm_base[3]= VA_DEST_PTR(vap);*/	\
	dbm_base[3]=VA_DBM_BIT0(vap);	\
	/*dbm_bit0[3]=VA_DBM_BIT0(vap);*/	\
	dbm_bit0=VA_DBM_BIT0(vap);

#define INIT_BASES_SBM				\
	sbm_ptr= VA_SRC_PTR(vap,4);		\
	/*sbm_base[3]= VA_SRC_PTR(vap,4);*/	\
	sbm_base[3]=VA_SBM_BIT0(vap);		\
	/*sbm_bit0[3]=VA_SBM_BIT0(vap);*/	\
	sbm_bit0=VA_SBM_BIT0(vap);

#define INIT_BASES_SBM_1	INIT_BASES_1 INIT_BASES_SBM
#define INIT_BASES_SBM_2	INIT_BASES_2 INIT_BASES_SBM
#define INIT_BASES_SBM_3	INIT_BASES_3 INIT_BASES_SBM
#define INIT_BASES_SBM_CPX_1	INIT_BASES_CPX_1 INIT_BASES_SBM
#define INIT_BASES_SBM_CPX_2	INIT_BASES_CPX_2 INIT_BASES_SBM
#define INIT_BASES_SBM_CPX_3	INIT_BASES_CPX_3 INIT_BASES_SBM
#define INIT_BASES_SBM_QUAT_1	INIT_BASES_QUAT_1 INIT_BASES_SBM
#define INIT_BASES_SBM_QUAT_2	INIT_BASES_QUAT_2 INIT_BASES_SBM
#define INIT_BASES_SBM_QUAT_3	INIT_BASES_QUAT_3 INIT_BASES_SBM

#define INIT_BASES_2SRCS	INIT_BASES_SRC1	INIT_BASES_SRC2
#define INIT_BASES_DBM_1SRC	INIT_BASES_DBM INIT_BASES_SRC1
#define INIT_BASES_DBM_SBM	INIT_BASES_DBM INIT_BASES_SBM
#define INIT_BASES_DBM_2	INIT_BASES_DBM INIT_BASES_2SRCS
#define INIT_BASES_DBM_2SRCS	INIT_BASES_DBM INIT_BASES_2SRCS

#define INIT_COUNT( var, index ) _INIT_COUNT(var,count,index)

//#define _INIT_COUNT( var, array, index ) var=array[index];
#define _INIT_COUNT( var, array, index )	var=INDEX_COUNT(array,index);

#define COPY_BASES_1(index)		dst_base[index] = dst_base[index+1];
#define COPY_BASES_CPX_1(index)		cdst_base[index] = cdst_base[index+1];
#define COPY_BASES_QUAT_1(index)	qdst_base[index] = qdst_base[index+1];

#define COPY_BASES_IDX_1(index)	COPY_BASES_1(index) index_base[index] = index_base[index+1];


#define COPY_BASES_IDX_2(index)			\
	COPY_BASES_IDX_1(index)			\
	COPY_BASES_SRC1(index)


#define COPY_BASES_SRC1(index)		s1_base[index] = s1_base[index+1];
#define COPY_BASES_SRC2(index)		s2_base[index] = s2_base[index+1];
#define COPY_BASES_SRC3(index)		s3_base[index] = s3_base[index+1];
#define COPY_BASES_SRC4(index)		s4_base[index] = s4_base[index+1];

#define COPY_BASES_CPX_SRC1(index)	cs1_base[index] = cs1_base[index+1];
#define COPY_BASES_CPX_SRC2(index)	cs2_base[index] = cs2_base[index+1];
#define COPY_BASES_CPX_SRC3(index)	cs3_base[index] = cs3_base[index+1];
#define COPY_BASES_CPX_SRC4(index)	cs4_base[index] = cs4_base[index+1];

#define COPY_BASES_QUAT_SRC1(index)	qs1_base[index] = qs1_base[index+1];
#define COPY_BASES_QUAT_SRC2(index)	qs2_base[index] = qs2_base[index+1];
#define COPY_BASES_QUAT_SRC3(index)	qs3_base[index] = qs3_base[index+1];
#define COPY_BASES_QUAT_SRC4(index)	qs4_base[index] = qs4_base[index+1];

#define COPY_BASES_2(index)	COPY_BASES_1(index) COPY_BASES_SRC1(index)
#define COPY_BASES_3(index)	COPY_BASES_2(index) COPY_BASES_SRC2(index)
#define COPY_BASES_4(index)	COPY_BASES_3(index) COPY_BASES_SRC3(index)
#define COPY_BASES_5(index)	COPY_BASES_4(index) COPY_BASES_SRC4(index)

#define COPY_BASES_CPX_2(index)	COPY_BASES_CPX_1(index) COPY_BASES_CPX_SRC1(index)
#define COPY_BASES_CPX_3(index)	COPY_BASES_CPX_2(index) COPY_BASES_CPX_SRC2(index)
#define COPY_BASES_CPX_4(index)	COPY_BASES_CPX_3(index) COPY_BASES_CPX_SRC3(index)
#define COPY_BASES_CPX_5(index)	COPY_BASES_CPX_4(index) COPY_BASES_CPX_SRC4(index)
#define COPY_BASES_CCR_3(index)	COPY_BASES_CPX_2(index) COPY_BASES_SRC2(index)
#define COPY_BASES_CR_2(index)	COPY_BASES_CPX_1(index) COPY_BASES_SRC1(index)
#define COPY_BASES_RC_2(index)	COPY_BASES_1(index) COPY_BASES_CPX_SRC1(index)

#define COPY_BASES_QUAT_2(index)	COPY_BASES_QUAT_1(index) COPY_BASES_QUAT_SRC1(index)
#define COPY_BASES_QUAT_3(index)	COPY_BASES_QUAT_2(index) COPY_BASES_QUAT_SRC2(index)
#define COPY_BASES_QUAT_4(index)	COPY_BASES_QUAT_3(index) COPY_BASES_QUAT_SRC3(index)
#define COPY_BASES_QUAT_5(index)	COPY_BASES_QUAT_4(index) COPY_BASES_QUAT_SRC4(index)
#define COPY_BASES_QQR_3(index)	COPY_BASES_QUAT_2(index) COPY_BASES_SRC2(index)
#define COPY_BASES_QR_2(index)	COPY_BASES_QUAT_1(index) COPY_BASES_SRC1(index)
#define COPY_BASES_RQ_2(index)	COPY_BASES_1(index) COPY_BASES_QUAT_SRC1(index)

#define COPY_BASES_SBM_1(index)	COPY_BASES_1(index) COPY_BASES_SBM(index)
#define COPY_BASES_SBM_2(index)	COPY_BASES_2(index) COPY_BASES_SBM(index)
#define COPY_BASES_SBM_3(index)	COPY_BASES_3(index) COPY_BASES_SBM(index)
#define COPY_BASES_SBM_CPX_1(index)	COPY_BASES_CPX_1(index) COPY_BASES_SBM(index)
#define COPY_BASES_SBM_CPX_2(index)	COPY_BASES_CPX_2(index) COPY_BASES_SBM(index)
#define COPY_BASES_SBM_CPX_3(index)	COPY_BASES_CPX_3(index) COPY_BASES_SBM(index)
#define COPY_BASES_SBM_QUAT_1(index)	COPY_BASES_QUAT_1(index) COPY_BASES_SBM(index)
#define COPY_BASES_SBM_QUAT_2(index)	COPY_BASES_QUAT_2(index) COPY_BASES_SBM(index)
#define COPY_BASES_SBM_QUAT_3(index)	COPY_BASES_QUAT_3(index) COPY_BASES_SBM(index)


#define COPY_BASES_DBM(index)			\
						\
	dbm_base[index] = dbm_base[index+1];	\
	/*dbm_bit0[index] = dbm_bit0[index+1];*/

#define COPY_BASES_SBM(index)			\
						\
	sbm_base[index] = sbm_base[index+1];	\
	/*sbm_bit0[index] = sbm_bit0[index+1];*/

#define COPY_BASES_DBM_SBM(index)	COPY_BASES_DBM(index)	\
					COPY_BASES_SBM(index)

#define COPY_BASES_XXX_1	COPY_BASES_1
#define COPY_BASES_XXX_2	COPY_BASES_2
#define COPY_BASES_XXX_3	COPY_BASES_3
#define COPY_BASES_SBM_XXX_1	COPY_BASES_SBM_1
#define COPY_BASES_SBM_XXX_2	COPY_BASES_SBM_2
#define COPY_BASES_SBM_XXX_3	COPY_BASES_SBM_3
#define COPY_BASES_X_2		COPY_BASES_2


#define DECLARE_BASES_SBM			\
	/*bitmap_word *sbm_base[N_DIMENSIONS-1];*/	\
	int sbm_base[N_DIMENSIONS-1];		\
	bitmap_word *sbm_ptr;			\
	int sbm_bit;				\
	/*int sbm_bit0[N_DIMENSIONS-1];*/	\
	int sbm_bit0;

#define DECLARE_BASES_DBM_	DECLARE_BASES_DBM

/* base is not a bit number, not a pointer */

#define DECLARE_BASES_DBM			\
	/*bitmap_word *dbm_base[N_DIMENSIONS-1];*/	\
	int dbm_base[N_DIMENSIONS-1];		\
	/*int dbm_bit0[N_DIMENSIONS-1];*/		\
	int dbm_bit0;				\
	int dbm_bit;				\
	bitmap_word *dbm_ptr;			\
	DECLARE_FIVE_LOOP_INDICES

#define DECLARE_FOUR_LOOP_INDICES		\
	dimension_t i;				\
	dimension_t j;				\
	dimension_t k;				\
	dimension_t l;

#define DECLARE_FIVE_LOOP_INDICES		\
	DECLARE_FOUR_LOOP_INDICES		\
	dimension_t m;

#define DECLARE_BASES_CONV_1(type)		\
	type *dst_base[N_DIMENSIONS-1];		\
	type *dst_ptr;				\
	DECLARE_FIVE_LOOP_INDICES

#define DECLARE_BASES_1				\
	dest_type *dst_base[N_DIMENSIONS-1];	\
	dest_type *dst_ptr;			\
	DECLARE_FIVE_LOOP_INDICES

#define DECLARE_BASES_IDX_1			\
	index_type *dst_base[N_DIMENSIONS-1];	\
	index_type index_base[N_DIMENSIONS-1];	\
	index_type *dst_ptr;			\
	DECLARE_FIVE_LOOP_INDICES

#define DECLARE_BASES_CPX_1			\
	dest_cpx *cdst_base[N_DIMENSIONS-1];	\
	dest_cpx *cdst_ptr;			\
	DECLARE_FOUR_LOOP_INDICES

#define DECLARE_BASES_QUAT_1			\
	dest_quat *qdst_base[N_DIMENSIONS-1];	\
	dest_quat *qdst_ptr;			\
	DECLARE_FOUR_LOOP_INDICES

#define DECLARE_X_DST_VBASE(type)		\
	type *dst_base[N_DIMENSIONS-1];		\
	type *dst_ptr;				\
	DECLARE_FIVE_LOOP_INDICES

#define DECLARE_XXX_DST_VBASE(type,keychar)	\
	type *keychar##dst_base[N_DIMENSIONS-1];\
	type *keychar##dst_ptr;			\
	DECLARE_FOUR_LOOP_INDICES

#define DECLARE_BASES_CONV_2(type)		\
	DECLARE_BASES_CONV_1(type)		\
	DECLARE_VBASE_SRC1

#define DECLARE_BASES_2				\
	DECLARE_BASES_1				\
	DECLARE_VBASE_SRC1

#define DECLARE_BASES_IDX_2			\
	DECLARE_BASES_IDX_1			\
	DECLARE_VBASE_SRC1

#define DECLARE_BASES_CPX_2				\
	DECLARE_BASES_CPX_1				\
	DECLARE_VBASE_CPX_SRC1

#define DECLARE_BASES_QUAT_2				\
	DECLARE_BASES_QUAT_1				\
	DECLARE_VBASE_QUAT_SRC1

#define DECLARE_VBASE(typ,prefix)	typ *prefix##_base[N_DIMENSIONS-1];	\
					typ *prefix##_ptr;

#define DECLARE_VBASE_SRC1		DECLARE_VBASE(std_type,s1)
#define DECLARE_VBASE_CPX_SRC1		DECLARE_VBASE(std_cpx,cs1)
#define DECLARE_VBASE_QUAT_SRC1		DECLARE_VBASE(std_quat,qs1)
#define DECLARE_XXX_SRC1_VBASE(type)	DECLARE_VBASE(type,s1)
#define DECLARE_XXX_SRC2_VBASE(type)	DECLARE_VBASE(type,s2)
#define DECLARE_VBASE_SRC2		DECLARE_VBASE(std_type,s2)
#define DECLARE_VBASE_CPX_SRC2		DECLARE_VBASE(std_cpx,cs2)
#define DECLARE_VBASE_QUAT_SRC2		DECLARE_VBASE(std_quat,qs2)

#define DECLARE_BASES_3		DECLARE_BASES_2	DECLARE_VBASE_SRC2

#define DECLARE_BASES_SBM_1	DECLARE_BASES_SBM DECLARE_BASES_1
#define DECLARE_BASES_SBM_2	DECLARE_BASES_SBM DECLARE_BASES_2
#define DECLARE_BASES_SBM_3	DECLARE_BASES_SBM DECLARE_BASES_3

#define DECLARE_BASES_CPX_T_3				\
	DECLARE_BASES_CPX_3				\

#define DECLARE_BASES_CPX_3				\
	DECLARE_BASES_CPX_2				\
	DECLARE_VBASE_CPX_SRC2

#define DECLARE_BASES_QUAT_3				\
	DECLARE_BASES_QUAT_2				\
	DECLARE_VBASE_QUAT_SRC2

#define DECLARE_BASES_SBM_CPX_1				\
	DECLARE_BASES_CPX_1				\
	DECLARE_BASES_SBM

#define DECLARE_BASES_SBM_CPX_2				\
	DECLARE_BASES_CPX_2				\
	DECLARE_BASES_SBM

#define DECLARE_BASES_SBM_CPX_3				\
	DECLARE_BASES_CPX_3				\
	DECLARE_BASES_SBM

#define DECLARE_BASES_SBM_QUAT_1				\
	DECLARE_BASES_QUAT_1				\
	DECLARE_BASES_SBM

#define DECLARE_BASES_SBM_QUAT_2				\
	DECLARE_BASES_QUAT_2				\
	DECLARE_BASES_SBM

#define DECLARE_BASES_SBM_QUAT_3				\
	DECLARE_BASES_QUAT_3				\
	DECLARE_BASES_SBM


#define DECLARE_BASES_CR_2				\
	DECLARE_BASES_CPX_1				\
	DECLARE_VBASE_SRC1

#define DECLARE_BASES_QR_2				\
	DECLARE_BASES_QUAT_1				\
	DECLARE_VBASE_SRC1

#define DECLARE_BASES_CCR_3				\
	DECLARE_BASES_CPX_2				\
	DECLARE_VBASE_SRC2

#define DECLARE_BASES_QQR_3				\
	DECLARE_BASES_QUAT_2				\
	DECLARE_VBASE_SRC2

#define DECLARE_BASES_RC_2			\
	dest_type *dst_base[N_DIMENSIONS-1];	\
	dest_type *dst_ptr;			\
	DECLARE_VBASE_CPX_SRC1			\
	DECLARE_FOUR_LOOP_INDICES

#define DECLARE_BASES_4				\
	DECLARE_BASES_3				\
	std_type *s3_ptr;			\
	std_type *s3_base[N_DIMENSIONS-1];

#define DECLARE_BASES_5				\
	DECLARE_BASES_4				\
	std_type *s4_ptr;			\
	std_type *s4_base[N_DIMENSIONS-1];

#define DECLARE_BASES_DBM_1SRC		DECLARE_BASES_DBM DECLARE_VBASE_SRC1
#define DECLARE_BASES_DBM_SBM		DECLARE_BASES_DBM DECLARE_BASES_SBM
#define DECLARE_BASES_DBM_2SRCS		DECLARE_BASES_DBM DECLARE_BASES_2SRCS
#define DECLARE_BASES_2SRCS		DECLARE_VBASE_SRC1 DECLARE_VBASE_SRC2

#define DECLARE_BASES_X_2(dsttyp,srctyp)	\
		DECLARE_X_DST_VBASE(dsttyp)	\
		DECLARE_XXX_SRC1_VBASE(srctyp)

// for debugging...
#define INIT_SPACING( spi )					\
								\
	spi.spi_dst_isp = NULL

#define INIT_VEC_ARGS(vap)					\
								\
	INIT_SPACING( VA_SPACING(vap) );


// BUG - we should avoid dynamic allocation...

#define RELEASE_VEC_ARGS_STRUCT					\
	givbuf( VA_SPACING(vap) );				\
	givbuf( VA_SIZE_INFO(vap) );				\
	givbuf(vap);

#define DECL_VEC_ARGS_STRUCT(name)				\
	Vector_Args *vap=NEW_VEC_ARGS;				\
	DECLARE_FUNC_NAME(name)					\
	INIT_VEC_ARGS(vap)

// MIN is defined in the iOS Foundation headers...
// We should make sure it is the same!
#ifndef MIN
#define MIN( n1, n2 )		((n1)<(n2)?(n1):(n2))
#endif /* undef MIN */

/* The loop_arr array holds the max of all the dimensions - either 1 or N_i.
 * If we encounter a dimension which is not N_i or 1, then it's an error.
 */

#define ADJ_COUNTS(loop_arr,obj_arr)				\
								\
for(i_dim=0;i_dim<N_DIMENSIONS;i_dim++){			\
	if( INDEX_COUNT(obj_arr,i_dim) > 1 ){				\
		if( INDEX_COUNT(loop_arr,i_dim) == 1 ){			\
			ASSIGN_IDX_COUNT(loop_arr,i_dim,INDEX_COUNT(obj_arr,i_dim));	\
		} else {					\
			if( INDEX_COUNT(obj_arr,i_dim) != INDEX_COUNT(loop_arr,i_dim) ){\
				count_type n;			\
								\
				n = MIN(INDEX_COUNT(loop_arr,i_dim),	\
						INDEX_COUNT(obj_arr,i_dim));\
				sprintf(DEFAULT_ERROR_STRING,		\
	"Oops: %s count mismatch, (%d != %d), using %d",	\
					dimension_name[i_dim],	\
					INDEX_COUNT(loop_arr,i_dim),	\
					INDEX_COUNT(obj_arr,i_dim),n);	\
				ASSIGN_IDX_COUNT(loop_arr,i_dim,n);		\
			}					\
			/* else loop_arr already has value */	\
		}						\
	}							\
}

#define SHOW_BASES			\
sprintf(DEFAULT_ERROR_STRING,"s1_ptr:  0x%lx",(int_for_addr)s1_ptr);\
NADVISE(DEFAULT_ERROR_STRING);\
/*sprintf(DEFAULT_ERROR_STRING,"bm_ptr:  0x%lx, which_bit = %d",(int_for_addr)bm_ptr,which_bit);\
NADVISE(DEFAULT_ERROR_STRING);*/\
sprintf(DEFAULT_ERROR_STRING,"s1_base:  0x%lx  0x%lx  0x%lx  0x%lx",(int_for_addr)s1_base[0],(int_for_addr)s1_base[1],(int_for_addr)s1_base[2],(int_for_addr)s1_base[3]);\
NADVISE(DEFAULT_ERROR_STRING);



#define CONV_METHOD_DECL(name)					\
								\
static void CONV_METHOD_NAME(name)( Vec_Obj_Args *oap )


#include "veclib/fast_test.h"


#define dst		(*dst_ptr)
#define src1		(*s1_ptr)
#define src2		(*s2_ptr)
#define src3		(*s3_ptr)
#define src4		(*s4_ptr)

#define cdst		(*cdst_ptr)
#define csrc1		(*cs1_ptr)
#define csrc2		(*cs2_ptr)
#define csrc3		(*cs3_ptr)
#define csrc4		(*cs4_ptr)

#define qdst		(*qdst_ptr)
#define qsrc1		(*qs1_ptr)
#define qsrc2		(*qs2_ptr)
#define qsrc3		(*qs3_ptr)
#define qsrc4		(*qs4_ptr)

/* This implementation of vmaxg vming requires that the destination
 * index array be contiguous...
 */

#define EXTLOC_DOIT(assignment)					\
		assignment;					\
		dst = index;					\
		dst_ptr++;					\
		nocc++;

#define EXTLOC_STATEMENT( augment_condition,			\
				restart_condition, assignment )	\
								\
	if(restart_condition) {					\
		nocc=0;						\
		dst_ptr = orig_dst;				\
		EXTLOC_DOIT(assignment)				\
	} else if( augment_condition ){				\
		CHECK_IDXVEC_OVERFLOW				\
		EXTLOC_DOIT(assignment)				\
	}							\
	index++;



#define CHECK_IDXVEC_OVERFLOW					\
								\
	if( nocc >= idx_len ){					\
		if( verbose && ! overflow_warned ){		\
			sprintf(DEFAULT_ERROR_STRING,			\
"%s:  index vector has %d elements, more occurrences of extreme value", \
				func_name, idx_len);		\
			NADVISE(DEFAULT_ERROR_STRING);			\
			overflow_warned=1;			\
		}						\
		dst_ptr--;					\
		nocc--;						\
	}



/******************* Section 2 - slow loop bodies *******************/

#define SHOW_SLOW_COUNT		\
sprintf(DEFAULT_ERROR_STRING,"count = %d %d %d %d %d",\
INDEX_COUNT(count,0),		\
INDEX_COUNT(count,1),		\
INDEX_COUNT(count,2),		\
INDEX_COUNT(count,3),		\
INDEX_COUNT(count,4));		\
NADVISE(DEFAULT_ERROR_STRING);

#define GENERIC_SLOW_BODY( name, statement, decls, inits,	\
	copy_macro, ptr_init, comp_inc, inc_macro, debug_it )	\
								\
{								\
	decls							\
	inits							\
								\
	INIT_COUNT(i,4)						\
	while(i-- > 0){						\
		copy_macro(2)					\
		INIT_COUNT(j,3)					\
		while(j-- > 0){					\
			copy_macro(1)				\
			INIT_COUNT(k,2)				\
			while(k-- > 0){				\
				copy_macro(0)			\
				INIT_COUNT(l,1)			\
				while(l-- > 0){			\
					ptr_init		\
					INIT_COUNT(m,0)		\
					while(m-- > 0){		\
						debug_it	\
						statement ;	\
						comp_inc	\
					}			\
					inc_macro(1)		\
				}				\
				inc_macro(2)			\
			}					\
			inc_macro(3)				\
		}						\
		inc_macro(4)					\
	}							\
}

#define DEBUG_2		DEBUG_DST DEBUG_SRC1

#define DEBUG_2SRCS	DEBUG_SRC1 DEBUG_SRC2
				
#define DEBUG_DST								\
	sprintf(DEFAULT_ERROR_STRING,"\tdst = 0x%lx",(int_for_addr)dst_ptr);	\
	NADVISE(DEFAULT_ERROR_STRING);

#define DEBUG_SRC1								\
	sprintf(DEFAULT_ERROR_STRING,"\tsrc1 = 0x%lx",(int_for_addr)s1_ptr);	\
	NADVISE(DEFAULT_ERROR_STRING);

#define DEBUG_SRC2								\
	sprintf(DEFAULT_ERROR_STRING,"\tsrc2 = 0x%lx",(int_for_addr)s2_ptr);	\
	NADVISE(DEFAULT_ERROR_STRING);


#define GENERIC_XXX_SLOW_BODY( name, statement, decls,	\
	inits, copy_macro, ptr_init, inc_macro, debugit )\
							\
{							\
	decls						\
	inits						\
							\
	INIT_COUNT(i,4)					\
	while(i-- > 0){					\
		copy_macro(2)				\
		INIT_COUNT(j,3)				\
		while(j-- > 0){				\
			copy_macro(1)			\
			INIT_COUNT(k,2)			\
			while(k-- > 0){			\
				copy_macro(0)		\
				INIT_COUNT(l,1)		\
				while(l-- > 0){		\
					ptr_init	\
					debugit		\
					statement ;	\
					inc_macro(1)	\
				}			\
				inc_macro(2)		\
			}				\
			inc_macro(3)			\
		}					\
		inc_macro(4)				\
	}						\
}

#define SIMPLE_SLOW_BODY(name,statement,typ,suffix,debug_it)	\
							\
	GENERIC_SLOW_BODY( name, statement,		\
		DECLARE_BASES_##typ##suffix,		\
		INIT_BASES_##typ##suffix,		\
		COPY_BASES_##suffix,			\
		INIT_PTRS_##suffix,			\
		INC_PTRS_##typ##suffix,			\
		INC_BASES_##typ##suffix,		\
		debug_it)

#define SIMPLE_EQSP_BODY(name, statement,typ,suffix,extra,debugit)	\
							\
{							\
	EQSP_DECLS_##typ##suffix			\
	EQSP_INIT_##typ##suffix				\
	EXTRA_DECLS_##extra				\
	while(fl_ctr-- > 0){				\
		debugit					\
		statement ;				\
		EQSP_ADVANCE_##typ##suffix			\
	}						\
}

#define DEBUG_CPX_3	\
sprintf(DEFAULT_ERROR_STRING,"executing dst = 0x%lx   src1 = 0x%lx  src2 = 0x%lx",\
(int_for_addr)cdst_ptr,\
(int_for_addr)cs1_ptr,\
(int_for_addr)cs2_ptr);\
NADVISE(DEFAULT_ERROR_STRING);


// eqsp bodies


#define EQSP_BODY_2(name, statement)		SIMPLE_EQSP_BODY(name, statement,,2,,)
#define EQSP_BODY_3( name, statement )		SIMPLE_EQSP_BODY(name, statement,,3,,)
#define EQSP_BODY_4( name, statement )		SIMPLE_EQSP_BODY(name, statement,,4,,)
#define EQSP_BODY_5( name, statement )		SIMPLE_EQSP_BODY(name, statement,,5,,)
#define EQSP_BODY_SBM_1(name, statement)	SIMPLE_EQSP_BODY(name, statement,,SBM_1,,)
#define EQSP_BODY_SBM_2(name, statement)	SIMPLE_EQSP_BODY(name, statement,,SBM_2,,)
#define EQSP_BODY_SBM_3(name, statement)	SIMPLE_EQSP_BODY(name, statement,,SBM_3,,)
#define EQSP_BODY_SBM_CPX_1(name, statement)	SIMPLE_EQSP_BODY(name, statement,,SBM_CPX_1,,)
#define EQSP_BODY_SBM_CPX_2(name, statement)	SIMPLE_EQSP_BODY(name, statement,,SBM_CPX_2,,)
#define EQSP_BODY_SBM_CPX_3(name, statement)	SIMPLE_EQSP_BODY(name, statement,,SBM_CPX_3,,)
#define EQSP_BODY_SBM_QUAT_1(name, statement)	SIMPLE_EQSP_BODY(name, statement,,SBM_QUAT_1,,)
#define EQSP_BODY_SBM_QUAT_2(name, statement)	SIMPLE_EQSP_BODY(name, statement,,SBM_QUAT_2,,)
#define EQSP_BODY_SBM_QUAT_3(name, statement)	SIMPLE_EQSP_BODY(name, statement,,SBM_QUAT_3,,)
#define EQSP_BODY_BM_1(name, statement)		SIMPLE_EQSP_BODY(name, statement,,BM_1,,)
#define EQSP_BODY_DBM_1SRC(name, statement)	SIMPLE_EQSP_BODY(name, statement,,DBM_1SRC,,)
#define EQSP_BODY_DBM_SBM_(name, statement)	SIMPLE_EQSP_BODY(name, statement,,DBM_SBM,,)
#define EQSP_BODY_DBM_2SRCS(name, statement)	SIMPLE_EQSP_BODY(name, statement,,DBM_2SRCS,,)
#define EQSP_BODY_DBM_(name, statement)		SIMPLE_EQSP_BODY(name, statement,,DBM_,,)
#define EQSP_BODY_1( name, statement )		SIMPLE_EQSP_BODY(name, statement,,1,,)
#define EQSP_BODY_CPX_1( name, statement )	SIMPLE_EQSP_BODY(name, statement,CPX_,1,,)
#define EQSP_BODY_CPX_2(name, statement)	SIMPLE_EQSP_BODY(name, statement,CPX_,2,,)
#define EQSP_BODY_CPX_2_T2(name, statement)	SIMPLE_EQSP_BODY(name, statement,CPX_,2,T2,)
#define EQSP_BODY_CPX_2_T3(name, statement)	SIMPLE_EQSP_BODY(name, statement,CPX_,2,T3,)
#define EQSP_BODY_CPX_3( name, statement )	SIMPLE_EQSP_BODY(name, statement,CPX_,3,,)
#define EQSP_BODY_CPX_3_T1( name, statement )	SIMPLE_EQSP_BODY(name, statement,CPX_,3,T1,)
#define EQSP_BODY_CPX_3_T2( name, statement )	SIMPLE_EQSP_BODY(name, statement,CPX_,3,T2,)
#define EQSP_BODY_CPX_3_T3( name, statement )	SIMPLE_EQSP_BODY(name, statement,CPX_,3,T3,)
#define EQSP_BODY_CPX_4( name, statement )	SIMPLE_EQSP_BODY(name, statement,CPX_,4,,)
#define EQSP_BODY_CPX_5( name, statement )	SIMPLE_EQSP_BODY(name, statement,CPX_,5,,)
#define EQSP_BODY_CCR_3( name, statement )	SIMPLE_EQSP_BODY(name, statement,CCR_,3,,)
#define EQSP_BODY_CR_2( name, statement )	SIMPLE_EQSP_BODY(name, statement,CR_,2,,)
#define EQSP_BODY_RC_2( name, statement )	SIMPLE_EQSP_BODY(name, statement,RC_,2,,)
#define EQSP_BODY_QUAT_1( name, statement )	SIMPLE_EQSP_BODY(name, statement,QUAT_,1,,)
#define EQSP_BODY_QUAT_2(name, statement)	SIMPLE_EQSP_BODY(name, statement,QUAT_,2,,)
#define EQSP_BODY_QUAT_2_T4(name, statement)	SIMPLE_EQSP_BODY(name, statement,QUAT_,2,T4,)
#define EQSP_BODY_QUAT_2_T5(name, statement)	SIMPLE_EQSP_BODY(name, statement,QUAT_,2,T5,)
#define EQSP_BODY_QUAT_3( name, statement )	SIMPLE_EQSP_BODY(name, statement,QUAT_,3,,)
#define EQSP_BODY_QUAT_3_T4( name, statement )	SIMPLE_EQSP_BODY(name, statement,QUAT_,3,T4,)
#define EQSP_BODY_QUAT_3_T5( name, statement )	SIMPLE_EQSP_BODY(name, statement,QUAT_,3,T5,)
#define EQSP_BODY_QUAT_4( name, statement )	SIMPLE_EQSP_BODY(name, statement,QUAT_,4,,)
#define EQSP_BODY_QUAT_5( name, statement )	SIMPLE_EQSP_BODY(name, statement,QUAT_,5,,)
#define EQSP_BODY_QQR_3( name, statement )	SIMPLE_EQSP_BODY(name, statement,QQR_,3,,)
#define EQSP_BODY_QR_2( name, statement )	SIMPLE_EQSP_BODY(name, statement,QR_,2,,)
#define EQSP_BODY_RQ_2( name, statement )	SIMPLE_EQSP_BODY(name, statement,RQ_,2,,)


// slow bodies

#define SLOW_BODY_1(name,statement)	SIMPLE_SLOW_BODY(name,statement,,1,)
#define SLOW_BODY_2(name,statement)	SIMPLE_SLOW_BODY(name,statement,,2,)
#define SLOW_BODY_3(name,statement)	SIMPLE_SLOW_BODY(name,statement,,3,)
#define SLOW_BODY_CPX_1(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CPX_,1,,)
#define SLOW_BODY_CPX_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CPX_,2,,)
#define SLOW_BODY_CPX_2_T2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CPX_,2,T2,)
#define SLOW_BODY_CPX_2_T3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CPX_,2,T3,)
#define SLOW_BODY_CPX_3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CPX_,3,,)
#define SLOW_BODY_CPX_3_T1(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CPX_,3,T1,)
#define SLOW_BODY_CPX_3_T2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CPX_,3,T2,)
#define SLOW_BODY_CPX_3_T3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CPX_,3,T3,)
#define SLOW_BODY_QUAT_1(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QUAT_,1,,)
#define SLOW_BODY_QUAT_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QUAT_,2,,)
#define SLOW_BODY_QUAT_2_T4(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QUAT_,2,T4,)
#define SLOW_BODY_QUAT_2_T5(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QUAT_,2,T5,)
#define SLOW_BODY_QUAT_3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QUAT_,3,,)
#define SLOW_BODY_QUAT_3_T4(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QUAT_,3,T4,)
#define SLOW_BODY_QUAT_3_T5(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QUAT_,3,T5,)
#define SLOW_BODY_SBM_CPX_1(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,SBM_,CPX_,1,,)
#define SLOW_BODY_SBM_CPX_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,SBM_,CPX_,2,,)
#define SLOW_BODY_SBM_CPX_3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,SBM_,CPX_,3,,)
#define SLOW_BODY_SBM_QUAT_1(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,SBM_,QUAT_,1,,)
#define SLOW_BODY_SBM_QUAT_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,SBM_,QUAT_,2,,)
#define SLOW_BODY_SBM_QUAT_3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,SBM_,QUAT_,3,,)
#define SLOW_BODY_CR_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CR_,2,,)
#define SLOW_BODY_QR_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QR_,2,,)
#define SLOW_BODY_CCR_3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CCR_,3,,)
#define SLOW_BODY_QQR_3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QQR_,3,,)
#define SLOW_BODY_RC_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,RC_,2,,)
#define SLOW_BODY_4(name,statement)	SIMPLE_SLOW_BODY(name,statement,,4,)
#define SLOW_BODY_5(name,statement)	SIMPLE_SLOW_BODY(name,statement,,5,)
#define SLOW_BODY_DBM_(name,statement)	SIMPLE_SLOW_BODY(name,statement,,DBM_,)
// put DEBUG_DBM_1SRC in last position to see debug info
#define SLOW_BODY_DBM_1SRC(name,statement)	SIMPLE_SLOW_BODY(name,statement,,DBM_1SRC,/*DEBUG_DBM_1SRC*/)
#define SLOW_BODY_DBM_SBM_(name,statement)	SIMPLE_SLOW_BODY(name,statement,,DBM_SBM,)
#define SLOW_BODY_SBM_3(name,statement)	SIMPLE_SLOW_BODY(name,statement,,SBM_3,)
#define SLOW_BODY_SBM_2(name,statement)	SIMPLE_SLOW_BODY(name,statement,,SBM_2,)
#define SLOW_BODY_SBM_1(name,statement)	SIMPLE_SLOW_BODY(name,statement,,SBM_1,)
#define SLOW_BODY_DBM(name,statement)	SIMPLE_SLOW_BODY(name,statement,,DBM,)

#define SLOW_BODY_DBM_2SRCS(name,statement)	\
				SIMPLE_SLOW_BODY(name,statement,,DBM_2SRCS,)



#define SLOW_BODY_XX_2( name, statement,dsttyp,srctyp )		\
								\
	GENERIC_SLOW_BODY( name, statement,			\
		DECLARE_BASES_X_2(dsttyp,srctyp),		\
		INIT_BASES_X_2(dsttyp,srctyp),			\
		COPY_BASES_2,					\
		INIT_PTRS_2,					\
		INC_PTRS_2,					\
		INC_BASES_2,					\
		)


// XXX is complex or quaternion

#define SIMPLE_XXX_SLOW_BODY(name, stat,bitmap,typ,suffix,extra,debugit)	\
								\
	GENERIC_XXX_SLOW_BODY( name, stat,			\
		EXTRA_DECLS_##extra				\
		DECLARE_BASES_##bitmap##typ##suffix,		\
		INIT_BASES_##bitmap##typ##suffix,		\
		COPY_BASES_##bitmap##typ##suffix,		\
		INIT_PTRS_##bitmap##typ##suffix,		\
		INC_BASES_##bitmap##typ##suffix,		\
		debugit	)

#define SLOW_BODY_XXX_2( name, stat, dsttyp, srctyp )		\
								\
	SIMPLE_XXX_SLOW_BODY(name, stat,dsttyp,srctyp,,2,,)

#define SLOW_BODY_SBM_XXX_3( name, stat, dsttyp, srctyp )	\
								\
	SIMPLE_XXX_SLOW_BODY(name, stat,dsttyp,srctyp,SBM_,3,,)


#define SLOW_BODY_SBM_XXX_2( name, stat, dsttyp, srctyp )	\
								\
	SIMPLE_XXX_SLOW_BODY(name, stat,dsttyp,srctyp,SBM_,2,,)

#define SLOW_BODY_SBM_XXX_1( name, stat, dsttyp, srctyp )	\
								\
	GENERIC_XXX_SLOW_BODY( name, stat,			\
		DECL_INIT_SBM_XXX_1(dsttyp),			\
		COPY_BASES_SBM_XXX_1,				\
		INIT_PTRS_SBM_XXX_1,				\
		INC_BASES_SBM_XXX_1,			)


#define SLOW_BODY_XXX_1( name, stat, dsttyp, srctyp )		\
								\
	GENERIC_XXX_SLOW_BODY( name, stat,			\
		DECL_INIT_XXX_1(dsttyp),			\
		COPY_BASES_XXX_1,				\
		INIT_PTRS_XXX_1,				\
		INC_BASES_XXX_1,	)


#define SLOW_BODY_XXX_3( name, stat, dsttyp, src1typ, src2typ )	\
								\
	GENERIC_XXX_SLOW_BODY( name, stat,			\
		DECL_INIT_XXX_3(dsttyp,src1typ,src2typ),	\
		COPY_BASES_XXX_3,				\
		INIT_PTRS_XXX_3,				\
		INC_BASES_XXX_3,		)




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


#define SLOW_BODY_PROJ_2( name, init_statement, statement )		\
									\
{									\
	PROJ_LOOP_DECLS_2						\
									\
	INIT_LOOP_COUNT	/* init loop_count to all 1's */					\
									\
	ADJ_COUNTS(loop_count,count)					\
	ADJ_COUNTS(loop_count,s1_count)					\
									\
	/* for the init loop, we don't need to loop over all the src */	\
	/* We just scan the destination once */				\
	NEW_PLOOP_2( init_statement, count )				\
	NEW_PLOOP_2( statement, loop_count )				\
}




#define SLOW_BODY_PROJ_IDX_2( name, init_statement, statement )		\
									\
{									\
	PROJ_LOOP_DECLS_IDX_2						\
	orig_s1_ptr= (std_type *)VA_SRC_PTR(vap,0);	\
									\
	INIT_LOOP_COUNT							\
									\
	ADJ_COUNTS(loop_count,count)					\
	ADJ_COUNTS(loop_count,s1_count)					\
									\
	/* for the init loop, we don't need to loop over all the src */	\
	/* We just scan the destination once */				\
	NEW_PLOOP_IDX_2( init_statement, count )			\
	NEW_PLOOP_IDX_2( statement, loop_count )			\
}


#define PROJ3_SLOW_BODY( name, typ, init_statement, statement )		\
									\
{									\
	PROJ_LOOP_DECLS_##typ##3					\
									\
	INIT_LOOP_COUNT							\
									\
	ADJ_COUNTS(loop_count,count)					\
	ADJ_COUNTS(loop_count,s1_count)					\
	ADJ_COUNTS(loop_count,s2_count)					\
									\
	NEW_PLOOP_##typ##3( init_statement, count )				\
	NEW_PLOOP_##typ##3( statement, loop_count )				\
}


#define SLOW_BODY_PROJ_CPX_2( name, statement, init_statement )		\
	SLOW_BODY_PROJ_XXX_2(name,statement,init_statement,CPX_)

#define SLOW_BODY_PROJ_QUAT_2( name, statement, init_statement )	\
	SLOW_BODY_PROJ_XXX_2(name,statement,init_statement,QUAT_)

#define SLOW_BODY_PROJ_CPX_3( name, statement, init_statement )		\
	SLOW_BODY_PROJ_XXX_3(name,statement,init_statement,CPX_)

#define SLOW_BODY_PROJ_QUAT_3( name, statement, init_statement )	\
	SLOW_BODY_PROJ_XXX_3(name,statement,init_statement,QUAT_)


#define NEW_PLOOP_CPX_2(statement,loop_count)	NEW_PLOOP_XXX_2(statement,loop_count,CPX_)
#define NEW_PLOOP_CPX_3(statement,loop_count)	NEW_PLOOP_XXX_3(statement,loop_count,CPX_)
#define NEW_PLOOP_QUAT_2(statement,loop_count)	NEW_PLOOP_XXX_2(statement,loop_count,QUAT_)
#define NEW_PLOOP_QUAT_3(statement,loop_count)	NEW_PLOOP_XXX_3(statement,loop_count,QUAT_)

#define SLOW_BODY_PROJ_XXX_2( name, statement, init_statement, typ )	\
									\
{									\
	PROJ_LOOP_DECLS_##typ##2					\
									\
	INIT_LOOP_COUNT							\
									\
	ADJ_COUNTS(loop_count,count)					\
	ADJ_COUNTS(loop_count,s1_count)					\
									\
	NEW_PLOOP_##typ##2( init_statement, count )			\
	NEW_PLOOP_##typ##2( statement, loop_count )			\
}

#define SLOW_BODY_PROJ_XXX_3( name, statement, init_statement, typ )	\
									\
{									\
	PROJ_LOOP_DECLS_##typ##3					\
									\
	INIT_LOOP_COUNT							\
									\
	ADJ_COUNTS(loop_count,count)					\
	ADJ_COUNTS(loop_count,s1_count)					\
	ADJ_COUNTS(loop_count,s2_count)					\
									\
	NEW_PLOOP_##typ##3( init_statement, count )			\
	NEW_PLOOP_##typ##3( statement, loop_count )			\
}

/* Projection loop
 *
 * We typically call this once with the counts of the destination vector, to initialize,
 * and then again with the source counts to perform the projection...
 */


#define NEW_PLOOP_2( statement,count_arr )				\
									\
	INIT_BASES_2							\
	_INIT_COUNT(i,count_arr,4)					\
	while(i-- > 0){							\
		COPY_BASES_2(2)						\
		_INIT_COUNT(j,count_arr,3)				\
		while(j-- > 0){						\
			COPY_BASES_2(1)					\
			_INIT_COUNT(k,count_arr,2)			\
			while(k-- > 0){					\
				COPY_BASES_2(0)				\
				_INIT_COUNT(l,count_arr,1)		\
				while(l-- > 0){				\
					INIT_PTRS_2			\
					_INIT_COUNT(m,count_arr,0)	\
					while(m-- > 0){			\
						statement ;		\
						INC_PTRS_2		\
					}				\
					INC_BASES_2(1)			\
				}					\
				INC_BASES_2(2)				\
			}						\
			INC_BASES_2(3)					\
		}							\
		INC_BASES_2(4)						\
	}




#define NEW_PLOOP_IDX_2( statement,count_arr )				\
									\
	INIT_BASES_IDX_2						\
	_INIT_COUNT(i,count_arr,4)					\
	while(i-- > 0){							\
		COPY_BASES_IDX_2(2)					\
		_INIT_COUNT(j,count_arr,3)				\
		while(j-- > 0){						\
			COPY_BASES_IDX_2(1)				\
			_INIT_COUNT(k,count_arr,2)			\
			while(k-- > 0){					\
				COPY_BASES_IDX_2(0) /* sets index_base[0] */	\
				_INIT_COUNT(l,count_arr,1)		\
				while(l-- > 0){				\
					INIT_PTRS_IDX_2			\
					_INIT_COUNT(m,count_arr,0)	\
					while(m-- > 0){			\
						statement ;		\
						INC_PTRS_IDX_2		\
					}				\
					INC_BASES_IDX_2(1)		\
				}					\
				INC_BASES_IDX_2(2)			\
			}						\
			INC_BASES_IDX_2(3)				\
		}							\
		INC_BASES_IDX_2(4)					\
	}



#define NEW_PLOOP_3( statement, count_arr )			\
									\
	INIT_BASES_3						\
	_INIT_COUNT(i,count_arr,4)					\
	while(i-- > 0){							\
		COPY_BASES_3(2)						\
		_INIT_COUNT(j,count_arr,3)				\
		while(j-- > 0){						\
			COPY_BASES_3(1)					\
			_INIT_COUNT(k,count_arr,2)			\
			while(k-- > 0){					\
				COPY_BASES_3(0)				\
				_INIT_COUNT(l,count_arr,1)		\
				while(l-- > 0){				\
					INIT_PTRS_3			\
					_INIT_COUNT(m,count_arr,0)	\
					while(m-- > 0){			\
						statement ;		\
						INC_PTRS_3		\
					}				\
					INC_BASES_3(1)			\
				}					\
				INC_BASES_3(2)				\
			}						\
			INC_BASES_3(3)					\
		}							\
		INC_BASES_3(4)						\
	}

#define NEW_PLOOP_XXX_2( statement,count_arr,typ )			\
									\
	INIT_BASES_##typ##2						\
	_INIT_COUNT(i,count_arr,4)					\
	while(i-- > 0){							\
		COPY_BASES_##typ##2(2)					\
		_INIT_COUNT(j,count_arr,3)				\
		while(j-- > 0){						\
			COPY_BASES_##typ##2(1)				\
			_INIT_COUNT(k,count_arr,2)			\
			while(k-- > 0){					\
				COPY_BASES_##typ##2(0)			\
				_INIT_COUNT(l,count_arr,1)		\
				while(l-- > 0){				\
					INIT_PTRS_##typ##2		\
						statement ;		\
					INC_BASES_##typ##2(1)			\
				}					\
				INC_BASES_##typ##2(2)			\
			}						\
			INC_BASES_##typ##2(3)				\
		}							\
		INC_BASES_##typ##2(4)					\
	}


#define NEW_PLOOP_XXX_3( statement, count_arr, typ )			\
									\
	INIT_BASES_##typ##3						\
	_INIT_COUNT(i,count_arr,4)					\
	while(i-- > 0){							\
		COPY_BASES_##typ##3(2)					\
		_INIT_COUNT(j,count_arr,3)				\
		while(j-- > 0){						\
			COPY_BASES_##typ##3(1)				\
			_INIT_COUNT(k,count_arr,2)			\
			while(k-- > 0){					\
				COPY_BASES_##typ##3(0)			\
				_INIT_COUNT(l,count_arr,1)		\
				while(l-- > 0){				\
					INIT_PTRS_##typ##3		\
						statement ;		\
					INC_BASES_##typ##3(1)		\
				}					\
				INC_BASES_##typ##3(2)			\
			}						\
			INC_BASES_##typ##3(3)				\
		}							\
		INC_BASES_##typ##3(4)					\
	}


#define EXTLOC_SLOW_FUNC(name, statement)				\
									\
static void SLOW_NAME(name)( LINK_FUNC_ARG_DECLS )			\
{									\
	DECLARE_VBASE_SRC1						\
	DECLARE_FIVE_LOOP_INDICES					\
	EXTLOC_DECLS							\
	const char * func_name=#name;					\
									\
	INIT_BASES_SRC1							\
	s1_ptr = (std_type *) VA_SRC_PTR(vap,0);			\
	EXTLOC_INITS							\
	_INIT_COUNT(i,s1_count,4)					\
	while(i-- > 0){							\
		COPY_BASES_SRC1(2)					\
		_INIT_COUNT(j,s1_count,3)				\
		while(j-- > 0){						\
			COPY_BASES_SRC1(1)				\
			_INIT_COUNT(k,s1_count,2)			\
			while(k-- > 0){					\
				COPY_BASES_SRC1(0)			\
				_INIT_COUNT(l,s1_count,1)		\
				while(l-- > 0){				\
					INIT_PTRS_SRC1			\
					_INIT_COUNT(m,s1_count,0)	\
					while(m-- > 0){			\
						statement ;		\
						INC_PTRS_SRC1		\
					}				\
					INC_BASES_SRC1(1)		\
				}					\
				INC_BASES_SRC1(2)			\
			}						\
			INC_BASES_SRC1(3)				\
		}							\
		INC_BASES_SRC1(4)					\
	}								\
	SET_EXTLOC_RETURN_SCALARS					\
}


/*************** Section 3 - slow functions ***********************/

/******************* Section 4 - obsolete loops?  **********************/

/********************* Section 5 - fast functions ****************/

#define FF_DECL(name)	static void FAST_NAME(name)
#define EF_DECL(name)	static void EQSP_NAME(name)
#define SF_DECL(name)	static void SLOW_NAME(name)

#define MOV_FF_DECL(name, statement,bitmap,typ,scalars,vectors)	\
								\
	FF_DECL(name)( LINK_FUNC_ARG_DECLS )			\
	FAST_BODY_##typ##MOV( name, typ )

#define GENERIC_FF_DECL(name, statement,bitmap,typ,scalars,vectors,extra)\
									\
	FF_DECL(name)( LINK_FUNC_ARG_DECLS )				\
	FAST_BODY_##bitmap##typ##vectors##extra( name, statement )

#define GENERIC_EF_DECL(name, statement,bitmap,typ,scalars,vectors,extra)\
									\
	EF_DECL(name)( LINK_FUNC_ARG_DECLS )				\
	EQSP_BODY_##bitmap##typ##vectors##extra( name, statement )

#define GENERIC_SF_DECL(name,statement,bitmap,typ,scalars,vectors,extra)\
									\
	SF_DECL(name)( LINK_FUNC_ARG_DECLS )				\
	SLOW_BODY_##bitmap##typ##vectors##extra( name, statement )


#define GENERIC_FUNC_DECLS(name,statement,bitmap,typ,scalars,vectors,extra)\
									\
	GENERIC_FF_DECL(name,statement,bitmap,typ,scalars,vectors,extra)\
	GENERIC_EF_DECL(name,statement,bitmap,typ,scalars,vectors,extra)\
	GENERIC_SF_DECL(name,statement,bitmap,typ,scalars,vectors,extra)


#define MOV_FUNC_DECLS(name,statement,bitmap,typ,scalars,vectors)	\
									\
	MOV_FF_DECL(name,statement,bitmap,typ,scalars,vectors)		\
	GENERIC_EF_DECL(name,statement,bitmap,typ,scalars,vectors,)	\
	GENERIC_SF_DECL(name,statement,bitmap,typ,scalars,vectors,)


#define IDXRES_FAST_FUNC(name,init_statement,statement)		\
								\
FF_DECL(name)(IDX_PTR_ARG,PTR_ARGS_SRC1,COUNT_ARG)		\
{								\
	sprintf(DEFAULT_ERROR_STRING,"Sorry, Function %s is not implemented.",#name);\
	NWARN(DEFAULT_ERROR_STRING);					\
}

#define EXTLOC_DECLS				\
	index_type index;			\
	index_type *dst_ptr;			\
	index_type *orig_dst;			\
	std_type extval;			\
	dimension_t nocc;			\
	dimension_t idx_len;			\
	int overflow_warned=0;

#define EXTLOC_INITS					\
	dst_ptr = (index_type *)VA_DEST_PTR(vap);		\
	orig_dst = dst_ptr;				\
	nocc = 0;					\
	index = 0;					\
	extval = src1;					\
	/*idx_len = VA_SCALAR_VAL_UDI(vap,0);*/		\
	idx_len = VA_DEST_LEN(vap);


#define SET_EXTLOC_RETURN_SCALARS			\
	SET_VA_SCALAR_VAL_STD(vap,0,extval);		\
	SET_VA_SCALAR_VAL_UDI(vap,1,nocc);


#define EXTLOC_FAST_FUNC(name, statement)				\
									\
FF_DECL(name)( LINK_FUNC_ARG_DECLS )					\
{									\
	FAST_DECLS_SRC1							\
	FAST_INIT_SRC1							\
	EXTLOC_DECLS							\
	dimension_t fl_ctr;						\
	const char * func_name=#name;					\
									\
	FAST_INIT_COUNT							\
	FAST_INIT_SRC1							\
	EXTLOC_INITS							\
									\
	while(fl_ctr-- > 0){						\
		statement ;						\
		FAST_ADVANCE_SRC1					\
	}								\
									\
	/* Now return nocc & extval */					\
	SET_EXTLOC_RETURN_SCALARS					\
}


#define EXTLOC_EQSP_FUNC(name, statement)				\
									\
EF_DECL(name)( LINK_FUNC_ARG_DECLS )					\
{									\
	EQSP_DECLS_SRC1							\
	EQSP_INIT_SRC1							\
	EXTLOC_DECLS							\
	dimension_t fl_ctr;						\
	const char * func_name=#name;					\
									\
	EQSP_INIT_COUNT							\
	EQSP_INIT_SRC1							\
	EXTLOC_INITS							\
									\
	while(fl_ctr-- > 0){						\
		statement ;						\
		EQSP_ADVANCE_SRC1						\
	}								\
									\
	/* Now return nocc & extval */					\
	SET_EXTLOC_RETURN_SCALARS					\
}


/***************** Section 6 - fast loop bodies *************/


/* FAST_DECLS declare variables at the top of the body */

#define FAST_DECLS_1		dest_type *dst_ptr; dimension_t fl_ctr;
#define FAST_DECLS_SRC1		std_type *s1_ptr;
#define FAST_DECLS_SRC2		std_type *s2_ptr;
#define FAST_DECLS_SRC3		std_type *s3_ptr;
#define FAST_DECLS_SRC4		std_type *s4_ptr;
#define FAST_DECLS_2SRCS	FAST_DECLS_SRC1	FAST_DECLS_SRC2
#define FAST_DECLS_2		FAST_DECLS_1	FAST_DECLS_SRC1
#define FAST_DECLS_3		FAST_DECLS_2	FAST_DECLS_SRC2
#define FAST_DECLS_4		FAST_DECLS_3	FAST_DECLS_SRC3
#define FAST_DECLS_5		FAST_DECLS_4	FAST_DECLS_SRC4
#define FAST_DECLS_SBM		int sbm_bit; bitmap_word *sbm_ptr;
#define FAST_DECLS_DBM		int dbm_bit; bitmap_word *dbm_ptr; dimension_t fl_ctr;
#define FAST_DECLS_DBM_1SRC	FAST_DECLS_DBM FAST_DECLS_SRC1
#define FAST_DECLS_DBM_SBM	FAST_DECLS_DBM FAST_DECLS_SBM
#define FAST_DECLS_SBM_1	FAST_DECLS_SBM FAST_DECLS_1
#define FAST_DECLS_SBM_2	FAST_DECLS_SBM FAST_DECLS_2
#define FAST_DECLS_SBM_3	FAST_DECLS_SBM FAST_DECLS_3
#define FAST_DECLS_SBM_CPX_1	FAST_DECLS_SBM FAST_DECLS_CPX_1
#define FAST_DECLS_SBM_CPX_2	FAST_DECLS_SBM FAST_DECLS_CPX_2
#define FAST_DECLS_SBM_CPX_3	FAST_DECLS_SBM FAST_DECLS_CPX_3
#define FAST_DECLS_SBM_QUAT_1	FAST_DECLS_SBM FAST_DECLS_QUAT_1
#define FAST_DECLS_SBM_QUAT_2	FAST_DECLS_SBM FAST_DECLS_QUAT_2
#define FAST_DECLS_SBM_QUAT_3	FAST_DECLS_SBM FAST_DECLS_QUAT_3
#define FAST_DECLS_DBM_		FAST_DECLS_DBM
#define FAST_DECLS_DBM_2SRCS	FAST_DECLS_DBM FAST_DECLS_2SRCS

#define CPX_TMP_DECLS		std_type r; std_type theta; std_type arg;
#define FAST_DECLS_CPX_1	dest_cpx *cdst_ptr; dimension_t fl_ctr;
#define FAST_DECLS_CPX_SRC1	std_cpx *cs1_ptr;
#define FAST_DECLS_CPX_SRC2	std_cpx *cs2_ptr;
#define FAST_DECLS_CPX_SRC3	std_cpx *cs3_ptr;
#define FAST_DECLS_CPX_SRC4	std_cpx *cs4_ptr;
#define FAST_DECLS_CPX_2	FAST_DECLS_CPX_1	FAST_DECLS_CPX_SRC1
#define FAST_DECLS_CPX_3	FAST_DECLS_CPX_2	FAST_DECLS_CPX_SRC2
#define FAST_DECLS_CPX_4	FAST_DECLS_CPX_3	FAST_DECLS_CPX_SRC3
#define FAST_DECLS_CPX_5	FAST_DECLS_CPX_4	FAST_DECLS_CPX_SRC4
#define FAST_DECLS_CCR_3	FAST_DECLS_CPX_2	FAST_DECLS_SRC2
#define FAST_DECLS_CR_2		FAST_DECLS_CPX_1	FAST_DECLS_SRC1
#define FAST_DECLS_RC_2		FAST_DECLS_1	FAST_DECLS_CPX_SRC1

#define FAST_DECLS_QUAT_1	dest_quat *qdst_ptr; dimension_t fl_ctr;
#define FAST_DECLS_QUAT_SRC1	std_quat *qs1_ptr;
#define FAST_DECLS_QUAT_SRC2	std_quat *qs2_ptr;
#define FAST_DECLS_QUAT_SRC3	std_quat *qs3_ptr;
#define FAST_DECLS_QUAT_SRC4	std_quat *qs4_ptr;
#define FAST_DECLS_QUAT_2	FAST_DECLS_QUAT_1	FAST_DECLS_QUAT_SRC1
#define FAST_DECLS_QUAT_3	FAST_DECLS_QUAT_2	FAST_DECLS_QUAT_SRC2
#define FAST_DECLS_QUAT_4	FAST_DECLS_QUAT_3	FAST_DECLS_QUAT_SRC3
#define FAST_DECLS_QUAT_5	FAST_DECLS_QUAT_4	FAST_DECLS_QUAT_SRC4
#define FAST_DECLS_QQR_3	FAST_DECLS_QUAT_2	FAST_DECLS_SRC2
#define FAST_DECLS_QR_2		FAST_DECLS_QUAT_1	FAST_DECLS_SRC1
#define FAST_DECLS_RQ_2		FAST_DECLS_1	FAST_DECLS_QUAT_SRC1

// eqsp decls

#define EQSP_DECLS_1		dest_type *dst_ptr; dimension_t fl_ctr;
#define EQSP_DECLS_SRC1		std_type *s1_ptr;
#define EQSP_DECLS_SRC2		std_type *s2_ptr;
#define EQSP_DECLS_SRC3		std_type *s3_ptr;
#define EQSP_DECLS_SRC4		std_type *s4_ptr;
#define EQSP_DECLS_2SRCS	EQSP_DECLS_SRC1	EQSP_DECLS_SRC2
#define EQSP_DECLS_2		EQSP_DECLS_1	EQSP_DECLS_SRC1
#define EQSP_DECLS_3		EQSP_DECLS_2	EQSP_DECLS_SRC2
#define EQSP_DECLS_4		EQSP_DECLS_3	EQSP_DECLS_SRC3
#define EQSP_DECLS_5		EQSP_DECLS_4	EQSP_DECLS_SRC4
#define EQSP_DECLS_SBM		int sbm_bit; bitmap_word *sbm_ptr;
#define EQSP_DECLS_DBM		int dbm_bit; bitmap_word *dbm_ptr; dimension_t fl_ctr;
#define EQSP_DECLS_DBM_1SRC	EQSP_DECLS_DBM EQSP_DECLS_SRC1
#define EQSP_DECLS_DBM_SBM	EQSP_DECLS_DBM EQSP_DECLS_SBM
#define EQSP_DECLS_SBM_1	EQSP_DECLS_SBM EQSP_DECLS_1
#define EQSP_DECLS_SBM_2	EQSP_DECLS_SBM EQSP_DECLS_2
#define EQSP_DECLS_SBM_3	EQSP_DECLS_SBM EQSP_DECLS_3
#define EQSP_DECLS_SBM_CPX_1	EQSP_DECLS_SBM EQSP_DECLS_CPX_1
#define EQSP_DECLS_SBM_CPX_2	EQSP_DECLS_SBM EQSP_DECLS_CPX_2
#define EQSP_DECLS_SBM_CPX_3	EQSP_DECLS_SBM EQSP_DECLS_CPX_3
#define EQSP_DECLS_SBM_QUAT_1	EQSP_DECLS_SBM EQSP_DECLS_QUAT_1
#define EQSP_DECLS_SBM_QUAT_2	EQSP_DECLS_SBM EQSP_DECLS_QUAT_2
#define EQSP_DECLS_SBM_QUAT_3	EQSP_DECLS_SBM EQSP_DECLS_QUAT_3
#define EQSP_DECLS_DBM_		EQSP_DECLS_DBM
#define EQSP_DECLS_DBM_2SRCS	EQSP_DECLS_DBM EQSP_DECLS_2SRCS

#define EQSP_DECLS_CPX_1	dest_cpx *cdst_ptr; dimension_t fl_ctr;
#define EQSP_DECLS_CPX_SRC1	std_cpx *cs1_ptr;
#define EQSP_DECLS_CPX_SRC2	std_cpx *cs2_ptr;
#define EQSP_DECLS_CPX_SRC3	std_cpx *cs3_ptr;
#define EQSP_DECLS_CPX_SRC4	std_cpx *cs4_ptr;
#define EQSP_DECLS_CPX_2	EQSP_DECLS_CPX_1	EQSP_DECLS_CPX_SRC1
#define EQSP_DECLS_CPX_3	EQSP_DECLS_CPX_2	EQSP_DECLS_CPX_SRC2
#define EQSP_DECLS_CPX_4	EQSP_DECLS_CPX_3	EQSP_DECLS_CPX_SRC3
#define EQSP_DECLS_CPX_5	EQSP_DECLS_CPX_4	EQSP_DECLS_CPX_SRC4
#define EQSP_DECLS_CCR_3	EQSP_DECLS_CPX_2	EQSP_DECLS_SRC2
#define EQSP_DECLS_CR_2		EQSP_DECLS_CPX_1	EQSP_DECLS_SRC1
#define EQSP_DECLS_RC_2		EQSP_DECLS_1	EQSP_DECLS_CPX_SRC1

#define EQSP_DECLS_QUAT_1	dest_quat *qdst_ptr; dimension_t fl_ctr;
#define EQSP_DECLS_QUAT_SRC1	std_quat *qs1_ptr;
#define EQSP_DECLS_QUAT_SRC2	std_quat *qs2_ptr;
#define EQSP_DECLS_QUAT_SRC3	std_quat *qs3_ptr;
#define EQSP_DECLS_QUAT_SRC4	std_quat *qs4_ptr;
#define EQSP_DECLS_QUAT_2	EQSP_DECLS_QUAT_1	EQSP_DECLS_QUAT_SRC1
#define EQSP_DECLS_QUAT_3	EQSP_DECLS_QUAT_2	EQSP_DECLS_QUAT_SRC2
#define EQSP_DECLS_QUAT_4	EQSP_DECLS_QUAT_3	EQSP_DECLS_QUAT_SRC3
#define EQSP_DECLS_QUAT_5	EQSP_DECLS_QUAT_4	EQSP_DECLS_QUAT_SRC4
#define EQSP_DECLS_QQR_3	EQSP_DECLS_QUAT_2	EQSP_DECLS_SRC2
#define EQSP_DECLS_QR_2		EQSP_DECLS_QUAT_1	EQSP_DECLS_SRC1
#define EQSP_DECLS_RQ_2		EQSP_DECLS_1		EQSP_DECLS_QUAT_SRC1


/* FAST_INIT sets up the local vars from the arguments passed */


#define FAST_INIT_1		dst_ptr = (dest_type *)VA_DEST_PTR(vap);\
				FAST_INIT_COUNT

/* We used to divide by 2 or 4 for cpx and quat, but no longer needed. */
#define FAST_INIT_COUNT		fl_ctr = VA_LENGTH(vap);
/*
#define FAST_INIT_COUNT_CPX	fl_ctr = VA_LENGTH(vap);
#define FAST_INIT_COUNT_QUAT	fl_ctr = VA_LENGTH(vap);
*/

#define FAST_INIT_SRC1		s1_ptr = (std_type *)VA_SRC_PTR(vap,0);
#define FAST_INIT_SRC2		s2_ptr = (std_type *)VA_SRC_PTR(vap,1);
#define FAST_INIT_SRC3		s3_ptr = (std_type *)VA_SRC_PTR(vap,2);
#define FAST_INIT_SRC4		s4_ptr = (std_type *)VA_SRC_PTR(vap,3);
#define FAST_INIT_2		FAST_INIT_1	FAST_INIT_SRC1
#define FAST_INIT_3		FAST_INIT_2	FAST_INIT_SRC2
#define FAST_INIT_4		FAST_INIT_3	FAST_INIT_SRC3
#define FAST_INIT_5		FAST_INIT_4	FAST_INIT_SRC4
#define FAST_INIT_2SRCS		FAST_INIT_SRC1	FAST_INIT_SRC2


#define FAST_INIT_CPX_1		cdst_ptr = (dest_cpx *)VA_DEST_PTR(vap);	\
				FAST_INIT_COUNT
#define FAST_INIT_CPX_SRC1	cs1_ptr = (std_cpx *)VA_SRC_PTR(vap,0);
#define FAST_INIT_CPX_SRC2	cs2_ptr = (std_cpx *)VA_SRC_PTR(vap,1);
#define FAST_INIT_CPX_SRC3	cs3_ptr = (std_cpx *)VA_SRC_PTR(vap,2);
#define FAST_INIT_CPX_SRC4	cs4_ptr = (std_cpx *)VA_SRC_PTR(vap,3);

#define FAST_INIT_CPX_2		FAST_INIT_CPX_1	FAST_INIT_CPX_SRC1
#define FAST_INIT_CPX_3		FAST_INIT_CPX_2	FAST_INIT_CPX_SRC2
#define FAST_INIT_CPX_4		FAST_INIT_CPX_3	FAST_INIT_CPX_SRC3
#define FAST_INIT_CPX_5		FAST_INIT_CPX_4	FAST_INIT_CPX_SRC4
#define FAST_INIT_CCR_3		FAST_INIT_CPX_2	FAST_INIT_SRC2
#define FAST_INIT_CR_2		FAST_INIT_CPX_1	FAST_INIT_SRC1
#define FAST_INIT_RC_2		FAST_INIT_1	FAST_INIT_CPX_SRC1

#define FAST_INIT_QUAT_1	qdst_ptr = (dest_quat *)VA_DEST_PTR(vap);	\
				FAST_INIT_COUNT

#define FAST_INIT_QUAT_SRC1	qs1_ptr = (std_quat *)VA_SRC_PTR(vap,0);
#define FAST_INIT_QUAT_SRC2	qs2_ptr = (std_quat *)VA_SRC_PTR(vap,1);
#define FAST_INIT_QUAT_SRC3	qs3_ptr = (std_quat *)VA_SRC_PTR(vap,2);
#define FAST_INIT_QUAT_SRC4	qs4_ptr = (std_quat *)VA_SRC_PTR(vap,3);

#define FAST_INIT_QUAT_2	FAST_INIT_QUAT_1	FAST_INIT_QUAT_SRC1
#define FAST_INIT_QUAT_3	FAST_INIT_QUAT_2	FAST_INIT_QUAT_SRC2
#define FAST_INIT_QUAT_4	FAST_INIT_QUAT_3	FAST_INIT_QUAT_SRC3
#define FAST_INIT_QUAT_5	FAST_INIT_QUAT_4	FAST_INIT_QUAT_SRC4
#define FAST_INIT_QQR_3		FAST_INIT_QUAT_2	FAST_INIT_SRC2
#define FAST_INIT_QR_2		FAST_INIT_QUAT_1	FAST_INIT_SRC1
#define FAST_INIT_RQ_2		FAST_INIT_1	FAST_INIT_QUAT_SRC1

#define FAST_INIT_DBM_		FAST_INIT_DBM

#define FAST_INIT_DBM		dbm_bit = VA_DBM_BIT0(vap);		\
				dbm_ptr=VA_DEST_PTR(vap);	\
				FAST_INIT_COUNT

#define FAST_INIT_SBM		sbm_bit = VA_SBM_BIT0(vap);		\
				sbm_ptr=VA_SRC_PTR(vap,4);

#define FAST_INIT_DBM_2SRCS	FAST_INIT_DBM FAST_INIT_2SRCS
#define FAST_INIT_DBM_1SRC	FAST_INIT_DBM FAST_INIT_SRC1
#define FAST_INIT_DBM_SBM	FAST_INIT_DBM FAST_INIT_SBM
#define FAST_INIT_SBM_1		FAST_INIT_SBM FAST_INIT_1
#define FAST_INIT_SBM_2		FAST_INIT_SBM FAST_INIT_2
#define FAST_INIT_SBM_3		FAST_INIT_SBM FAST_INIT_3
#define FAST_INIT_SBM_CPX_1	FAST_INIT_SBM FAST_INIT_CPX_1
#define FAST_INIT_SBM_CPX_2	FAST_INIT_SBM FAST_INIT_CPX_2
#define FAST_INIT_SBM_CPX_3	FAST_INIT_SBM FAST_INIT_CPX_3
#define FAST_INIT_SBM_QUAT_1	FAST_INIT_SBM FAST_INIT_QUAT_1
#define FAST_INIT_SBM_QUAT_2	FAST_INIT_SBM FAST_INIT_QUAT_2
#define FAST_INIT_SBM_QUAT_3	FAST_INIT_SBM FAST_INIT_QUAT_3

///////////////////////////////////////
// eqsp init


#define EQSP_INIT_1		dst_ptr = (dest_type *)VA_DEST_PTR(vap);\
				EQSP_INIT_COUNT

/* We used to divide by 2 or 4 for cpx and quat, but no longer needed. */
#define EQSP_INIT_COUNT		fl_ctr = VA_LENGTH(vap);
/*
#define EQSP_INIT_COUNT_CPX	fl_ctr = VA_LENGTH(vap);
#define EQSP_INIT_COUNT_QUAT	fl_ctr = VA_LENGTH(vap);
*/

#define EQSP_INIT_SRC1		s1_ptr = (std_type *)VA_SRC_PTR(vap,0);
#define EQSP_INIT_SRC2		s2_ptr = (std_type *)VA_SRC_PTR(vap,1);
#define EQSP_INIT_SRC3		s3_ptr = (std_type *)VA_SRC_PTR(vap,2);
#define EQSP_INIT_SRC4		s4_ptr = (std_type *)VA_SRC_PTR(vap,3);
#define EQSP_INIT_2		EQSP_INIT_1	EQSP_INIT_SRC1
#define EQSP_INIT_3		EQSP_INIT_2	EQSP_INIT_SRC2
#define EQSP_INIT_4		EQSP_INIT_3	EQSP_INIT_SRC3
#define EQSP_INIT_5		EQSP_INIT_4	EQSP_INIT_SRC4
#define EQSP_INIT_2SRCS		EQSP_INIT_SRC1	EQSP_INIT_SRC2


#define EQSP_INIT_CPX_1		cdst_ptr = (dest_cpx *)VA_DEST_PTR(vap);	\
				EQSP_INIT_COUNT
#define EQSP_INIT_CPX_SRC1	cs1_ptr = (std_cpx *)VA_SRC_PTR(vap,0);
#define EQSP_INIT_CPX_SRC2	cs2_ptr = (std_cpx *)VA_SRC_PTR(vap,1);
#define EQSP_INIT_CPX_SRC3	cs3_ptr = (std_cpx *)VA_SRC_PTR(vap,2);
#define EQSP_INIT_CPX_SRC4	cs4_ptr = (std_cpx *)VA_SRC_PTR(vap,3);

#define EQSP_INIT_CPX_2		EQSP_INIT_CPX_1	EQSP_INIT_CPX_SRC1
#define EQSP_INIT_CPX_3		EQSP_INIT_CPX_2	EQSP_INIT_CPX_SRC2
#define EQSP_INIT_CPX_4		EQSP_INIT_CPX_3	EQSP_INIT_CPX_SRC3
#define EQSP_INIT_CPX_5		EQSP_INIT_CPX_4	EQSP_INIT_CPX_SRC4
#define EQSP_INIT_CCR_3		EQSP_INIT_CPX_2	EQSP_INIT_SRC2
#define EQSP_INIT_CR_2		EQSP_INIT_CPX_1	EQSP_INIT_SRC1
#define EQSP_INIT_RC_2		EQSP_INIT_1	EQSP_INIT_CPX_SRC1

#define EQSP_INIT_QUAT_1	qdst_ptr = (dest_quat *)VA_DEST_PTR(vap);	\
				EQSP_INIT_COUNT

#define EQSP_INIT_QUAT_SRC1	qs1_ptr = (std_quat *)VA_SRC_PTR(vap,0);
#define EQSP_INIT_QUAT_SRC2	qs2_ptr = (std_quat *)VA_SRC_PTR(vap,1);
#define EQSP_INIT_QUAT_SRC3	qs3_ptr = (std_quat *)VA_SRC_PTR(vap,2);
#define EQSP_INIT_QUAT_SRC4	qs4_ptr = (std_quat *)VA_SRC_PTR(vap,3);

#define EQSP_INIT_QUAT_2	EQSP_INIT_QUAT_1	EQSP_INIT_QUAT_SRC1
#define EQSP_INIT_QUAT_3	EQSP_INIT_QUAT_2	EQSP_INIT_QUAT_SRC2
#define EQSP_INIT_QUAT_4	EQSP_INIT_QUAT_3	EQSP_INIT_QUAT_SRC3
#define EQSP_INIT_QUAT_5	EQSP_INIT_QUAT_4	EQSP_INIT_QUAT_SRC4
#define EQSP_INIT_QQR_3		EQSP_INIT_QUAT_2	EQSP_INIT_SRC2
#define EQSP_INIT_QR_2		EQSP_INIT_QUAT_1	EQSP_INIT_SRC1
#define EQSP_INIT_RQ_2		EQSP_INIT_1	EQSP_INIT_QUAT_SRC1

#define EQSP_INIT_DBM_		EQSP_INIT_DBM

#define EQSP_INIT_DBM		dbm_bit = VA_DBM_BIT0(vap);		\
				dbm_ptr=VA_DEST_PTR(vap);	\
				EQSP_INIT_COUNT

#define EQSP_INIT_SBM		sbm_bit = VA_SBM_BIT0(vap);		\
				sbm_ptr=VA_SRC_PTR(vap,4);

#define EQSP_INIT_DBM_2SRCS	EQSP_INIT_DBM EQSP_INIT_2SRCS
#define EQSP_INIT_DBM_1SRC	EQSP_INIT_DBM EQSP_INIT_SRC1
#define EQSP_INIT_DBM_SBM	EQSP_INIT_DBM EQSP_INIT_SBM
#define EQSP_INIT_SBM_1		EQSP_INIT_SBM EQSP_INIT_1
#define EQSP_INIT_SBM_2		EQSP_INIT_SBM EQSP_INIT_2
#define EQSP_INIT_SBM_3		EQSP_INIT_SBM EQSP_INIT_3
#define EQSP_INIT_SBM_CPX_1	EQSP_INIT_SBM EQSP_INIT_CPX_1
#define EQSP_INIT_SBM_CPX_2	EQSP_INIT_SBM EQSP_INIT_CPX_2
#define EQSP_INIT_SBM_CPX_3	EQSP_INIT_SBM EQSP_INIT_CPX_3
#define EQSP_INIT_SBM_QUAT_1	EQSP_INIT_SBM EQSP_INIT_QUAT_1
#define EQSP_INIT_SBM_QUAT_2	EQSP_INIT_SBM EQSP_INIT_QUAT_2
#define EQSP_INIT_SBM_QUAT_3	EQSP_INIT_SBM EQSP_INIT_QUAT_3


/////////////////////////////////////

/* The fast body is pretty simple...  Should we try to unroll loops
 * to take advantage of SSE?  How do we help the compiler to do this?
 */

#define SIMPLE_FAST_BODY(name, statement,typ,suffix,extra,debugit)	\
							\
{							\
	FAST_DECLS_##typ##suffix			\
	FAST_INIT_##typ##suffix				\
	EXTRA_DECLS_##extra				\
	while(fl_ctr-- > 0){				\
		debugit					\
		statement ;				\
		FAST_ADVANCE_##typ##suffix			\
	}						\
}


#define FAST_BODY_CONV_2(name, statement,dsttyp,srctyp)	\
							\
{							\
	dsttyp *dst_ptr;				\
	srctyp *s1_ptr;					\
	dimension_t fl_ctr;				\
	dst_ptr = (dsttyp *)VA_DEST_PTR(vap);		\
	s1_ptr = (srctyp *)VA_SRC_PTR(vap,0);		\
	fl_ctr = VA_LENGTH(vap);				\
	while(fl_ctr-- > 0){				\
		statement ;				\
		FAST_ADVANCE_2				\
	}						\
}

/* There ought to be a more compact way to do all of this? */

#define FAST_BODY_2(name, statement)		SIMPLE_FAST_BODY(name, statement,,2,,)
#define FAST_BODY_3( name, statement )		SIMPLE_FAST_BODY(name, statement,,3,,)
#define FAST_BODY_4( name, statement )		SIMPLE_FAST_BODY(name, statement,,4,,)
#define FAST_BODY_5( name, statement )		SIMPLE_FAST_BODY(name, statement,,5,,)
#define FAST_BODY_SBM_1(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_1,,)
#define FAST_BODY_SBM_2(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_2,,)
#define FAST_BODY_SBM_3(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_3,,)
#define FAST_BODY_SBM_CPX_1(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_CPX_1,,)
#define FAST_BODY_SBM_CPX_2(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_CPX_2,,)
#define FAST_BODY_SBM_CPX_3(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_CPX_3,,)
#define FAST_BODY_SBM_QUAT_1(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_QUAT_1,,)
#define FAST_BODY_SBM_QUAT_2(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_QUAT_2,,)
#define FAST_BODY_SBM_QUAT_3(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_QUAT_3,,)
#define FAST_BODY_BM_1(name, statement)		SIMPLE_FAST_BODY(name, statement,,BM_1,,)
#define FAST_BODY_DBM_1SRC(name, statement)	SIMPLE_FAST_BODY(name, statement,,DBM_1SRC,,)
#define FAST_BODY_DBM_SBM_(name, statement)	SIMPLE_FAST_BODY(name, statement,,DBM_SBM,,)

#define FAST_BODY_DBM_2SRCS(name, statement)	SIMPLE_FAST_BODY(name, statement,,DBM_2SRCS,,)
#define FAST_BODY_DBM_(name, statement)		SIMPLE_FAST_BODY(name, statement,,DBM_,,)
#define FAST_BODY_1( name, statement )		SIMPLE_FAST_BODY(name, statement,,1,,)
#define FAST_BODY_CPX_1( name, statement )	SIMPLE_FAST_BODY(name, statement,CPX_,1,,)
#define FAST_BODY_CPX_2(name, statement)	SIMPLE_FAST_BODY(name, statement,CPX_,2,,)
#define FAST_BODY_CPX_2_T2(name, statement)	SIMPLE_FAST_BODY(name, statement,CPX_,2,T2,)
#define FAST_BODY_CPX_2_T3(name, statement)	SIMPLE_FAST_BODY(name, statement,CPX_,2,T3,)
#define FAST_BODY_CPX_3( name, statement )	SIMPLE_FAST_BODY(name, statement,CPX_,3,,)
#define FAST_BODY_CPX_3_T1( name, statement )	SIMPLE_FAST_BODY(name, statement,CPX_,3,T1,)
#define FAST_BODY_CPX_3_T2( name, statement )	SIMPLE_FAST_BODY(name, statement,CPX_,3,T2,)
#define FAST_BODY_CPX_3_T3( name, statement )	SIMPLE_FAST_BODY(name, statement,CPX_,3,T3,)
#define FAST_BODY_CPX_4( name, statement )	SIMPLE_FAST_BODY(name, statement,CPX_,4,,)
#define FAST_BODY_CPX_5( name, statement )	SIMPLE_FAST_BODY(name, statement,CPX_,5,,)
#define FAST_BODY_CCR_3( name, statement )	SIMPLE_FAST_BODY(name, statement,CCR_,3,,)
#define FAST_BODY_CR_2( name, statement )	SIMPLE_FAST_BODY(name, statement,CR_,2,,)
#define FAST_BODY_RC_2( name, statement )	SIMPLE_FAST_BODY(name, statement,RC_,2,,)
#define FAST_BODY_QUAT_1( name, statement )	SIMPLE_FAST_BODY(name, statement,QUAT_,1,,)
#define FAST_BODY_QUAT_2(name, statement)	SIMPLE_FAST_BODY(name, statement,QUAT_,2,,)
#define FAST_BODY_QUAT_2_T4(name, statement)	SIMPLE_FAST_BODY(name, statement,QUAT_,2,T4,)
#define FAST_BODY_QUAT_2_T5(name, statement)	SIMPLE_FAST_BODY(name, statement,QUAT_,2,T5,)
#define FAST_BODY_QUAT_3( name, statement )	SIMPLE_FAST_BODY(name, statement,QUAT_,3,,)
#define FAST_BODY_QUAT_3_T4( name, statement )	SIMPLE_FAST_BODY(name, statement,QUAT_,3,T4,)
#define FAST_BODY_QUAT_3_T5( name, statement )	SIMPLE_FAST_BODY(name, statement,QUAT_,3,T5,)
#define FAST_BODY_QUAT_4( name, statement )	SIMPLE_FAST_BODY(name, statement,QUAT_,4,,)
#define FAST_BODY_QUAT_5( name, statement )	SIMPLE_FAST_BODY(name, statement,QUAT_,5,,)
#define FAST_BODY_QQR_3( name, statement )	SIMPLE_FAST_BODY(name, statement,QQR_,3,,)
#define FAST_BODY_QR_2( name, statement )	SIMPLE_FAST_BODY(name, statement,QR_,2,,)
#define FAST_BODY_RQ_2( name, statement )	SIMPLE_FAST_BODY(name, statement,RQ_,2,,)


/********************** Section 7 - fast/slow switches **************/

#define SHOW_VEC_ARGS(speed)			\
	sprintf(DEFAULT_ERROR_STRING,"%s:  vap.va_dst_vp = 0x%lx",#speed,(int_for_addr)VA_DEST_PTR(vap);\
	NADVISE(DEFAULT_ERROR_STRING);					\
	sprintf(DEFAULT_ERROR_STRING,"%s:  vap.va_src_vp[0] = 0x%lx",#speed,(int_for_addr)VA_SRC_PTR(vap,0) );\
	NADVISE(DEFAULT_ERROR_STRING);					\
	sprintf(DEFAULT_ERROR_STRING,"%s:  vap.va_src_vp[1] = 0x%lx",#speed,(int_for_addr)VA_SRC_PTR(vap,1) );\
	NADVISE(DEFAULT_ERROR_STRING);					\

// BUG?  no equally-spaced case???

#define FAST_SWITCH_CONV( name )			\
							\
if( FAST_TEST_2 ){					\
	XFER_FAST_ARGS_2				\
	CHAIN_CHECK( FAST_CONV_NAME(name) )		\
} else if( EQSP_TEST_2 ){				\
	XFER_EQSP_ARGS_2				\
	CHAIN_CHECK( EQSP_CONV_NAME(name) )		\
} else {						\
	XFER_SLOW_ARGS_2				\
	CHAIN_CHECK( SLOW_CONV_NAME(name) )		\
}

#define SHOW_FAST_TEST_
#define SHOW_FAST_TEST_2SRCS
#define SHOW_FAST_TEST_1SRC
#define SHOW_FAST_TEST_4
#define SHOW_FAST_TEST_5

#define SHOW_FAST_TEST_1						\
	sprintf(DEFAULT_ERROR_STRING,"FAST_TEST_1:  %d",FAST_TEST_1?1:0);	\
	NADVISE(DEFAULT_ERROR_STRING);

// These macros that use IS_CONTIGUOUS have qsp problems...

#define SHOW_FAST_TEST_2						\
	SHOW_FAST_TEST_1						\
	sprintf(DEFAULT_ERROR_STRING,"FAST_TEST_SRC1:  %d",IS_CONTIGUOUS(SRC1_DP)?1:0);	\
	NADVISE(DEFAULT_ERROR_STRING);

#define SHOW_FAST_TEST_3						\
	SHOW_FAST_TEST_2						\
	sprintf(DEFAULT_ERROR_STRING,"FAST_TEST_SRC2:  %d",IS_CONTIGUOUS(SRC2_DP)?1:0);	\
	NADVISE(DEFAULT_ERROR_STRING);

#define GENERIC_FAST_SWITCH(name,bitmap,typ,scalars,vectors)	\
								\
								\
if( FAST_TEST_##bitmap##vectors ){				\
	REPORT_FAST_CALL					\
	XFER_FAST_ARGS_##bitmap##typ##scalars##vectors		\
	CHAIN_CHECK( FAST_NAME(name) )				\
} else if( EQSP_TEST_##bitmap##vectors ){			\
	REPORT_EQSP_CALL					\
	XFER_EQSP_ARGS_##bitmap##typ##scalars##vectors		\
	CHAIN_CHECK( EQSP_NAME(name) )				\
} else {							\
	REPORT_SLOW_CALL					\
	XFER_SLOW_ARGS_##bitmap##typ##scalars##vectors		\
	CHAIN_CHECK( SLOW_NAME(name) )				\
}

/* Why do we need these??? */
#define FAST_SWITCH_2( name )		GENERIC_FAST_SWITCH(name,,,,2)
#define FAST_SWITCH_3( name )		GENERIC_FAST_SWITCH(name,,,,3)
#define FAST_SWITCH_4( name )		GENERIC_FAST_SWITCH(name,,,,4)
#define FAST_SWITCH_CPX_3( name )	GENERIC_FAST_SWITCH(name,,CPX_,,3)
#define FAST_SWITCH_5( name )		GENERIC_FAST_SWITCH(name,,,,5)

#include "veclib/xfer_args.h"

/*********************** Section 8 - object methods ***************/

#define _VEC_FUNC_2V_CONV(name,type,statement)			\
								\
FF_DECL(name)( LINK_FUNC_ARG_DECLS )			\
{							\
	/* FAST_DECLS_1 */				\
	/* use passed type instead of dest_type */	\
	type *dst_ptr; dimension_t fl_ctr;		\
	FAST_DECLS_SRC1					\
	/* FAST_INIT_2 */				\
	/* FAST_INIT_1 */				\
	dst_ptr = (type *)VA_DEST_PTR(vap);		\
	FAST_INIT_COUNT					\
	FAST_INIT_SRC1					\
	/* no extra decls */				\
	while(fl_ctr-- > 0){				\
		statement ;				\
		FAST_ADVANCE_2				\
	}						\
}							\
							\
EF_DECL(name)( LINK_FUNC_ARG_DECLS )			\
{							\
	/* EQSP_DECLS_1 */				\
	/* use passed type instead of dest_type */	\
	type *dst_ptr; dimension_t fl_ctr;		\
	EQSP_DECLS_SRC1					\
	/* EQSP_INIT_2 */				\
	/* EQSP_INIT_1 */				\
	dst_ptr = (type *)VA_DEST_PTR(vap);		\
	EQSP_INIT_COUNT					\
	EQSP_INIT_SRC1					\
	/* no extra decls */				\
	while(fl_ctr-- > 0){				\
		statement ;				\
		EQSP_ADVANCE_2				\
	}						\
}							\
							\
/* GENERIC_SF_DECL(name,statement,bitmap,typ,scalars,vectors,extra) */	\
SF_DECL(name)( LINK_FUNC_ARG_DECLS )					\
	GENERIC_SLOW_BODY( name, statement,		\
		DECLARE_BASES_CONV_2(type),		\
		INIT_BASES_CONV_2(type),		\
		COPY_BASES_2,			\
		INIT_PTRS_2,			\
		INC_PTRS_2,			\
		INC_BASES_2,		\
		)


#define OBJ_METHOD(name,statement,bitmap,typ,scalars,vectors,extra)	\
									\
GENERIC_FUNC_DECLS(name, statement,bitmap,typ,scalars,vectors,extra)		// OBJ_METHOD



#define OBJ_MOV_METHOD(name,statement,bitmap,typ,scalars,vectors)	\
									\
MOV_FUNC_DECLS(name, statement,bitmap,typ,scalars,vectors)


#define _VEC_FUNC_2V_SCAL( name, statement )			\
	OBJ_METHOD(name,statement,,,1S_,2,)

#define _VEC_FUNC_3V( name, statement )	\
	OBJ_METHOD(name,statement,,,,3,)


// These are the kernels...

#define _VEC_FUNC_MM_NOCC( name, augment_condition,		\
					restart_condition, assignment,	\
					gpu_c1, gpu_c2 )		\
									\
EXTLOC_FAST_FUNC(name,EXTLOC_STATEMENT(augment_condition,restart_condition,assignment))	\
EXTLOC_EQSP_FUNC(name,EXTLOC_STATEMENT(augment_condition,restart_condition,assignment))	\
EXTLOC_SLOW_FUNC(name,EXTLOC_STATEMENT(augment_condition,restart_condition,assignment)) \
									\

#define _VEC_FUNC_2V( name, statement )				\
	OBJ_METHOD(name,statement,,,,2,)

/* e.g. ramp1d */

#define _VEC_FUNC_1V_2SCAL( name, statement, gpu_stat )		\
	OBJ_METHOD(name,statement,,,2S_,1,)

/* e.g. vset */

#define _VEC_FUNC_1V_SCAL( name, statement )			\
	OBJ_METHOD(name,statement,,,1S_,1,)

/* used to set a bitmap based on a vector test */
/* vvm_gt etc */

#define _VEC_FUNC_VVMAP( name, op )			\
	OBJ_METHOD(name,SET_DBM_BIT(src1 op src2),DBM_,,,2SRCS,)

#define _VEC_FUNC_5V( name, statement )	\
	OBJ_METHOD(name,statement,,,,5,)

#define _VEC_FUNC_4V_SCAL( name, statement )			\
									\
	OBJ_METHOD(name,statement,,,1S_,4,)

#define _VEC_FUNC_3V_2SCAL( name, statement )			\
	OBJ_METHOD(name,statement,,,2S_,3,)

#define _VEC_FUNC_2V_3SCAL( name, statement )			\
	OBJ_METHOD(name,statement,,,3S_,2,)
									\

#define _VEC_FUNC_VVSLCT(name, statement)	\
	OBJ_METHOD(name,statement,SBM_,,,3,)

#define _VEC_FUNC_VSSLCT(name, statement)			\
	OBJ_METHOD(name,statement,SBM_,,1S_,2,)

#define _VEC_FUNC_SSSLCT(name, statement)			\
	OBJ_METHOD(name,statement,SBM_,,2S_,1,)

#define _VEC_FUNC_1V( name, statement )				\
	OBJ_METHOD(name,statement,,,,1,)

/* this is for vmagsq, vatn2:  real result, cpx source */
#define _VEC_FUNC_2V_MIXED( name, statement )			\
	OBJ_METHOD(name,statement,,RC_,,2,)

//#define THREE_CPX_VEC_METHOD_T1( name, statement )
#define _VEC_FUNC_CPX_3V_T1( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,,3,_T1)

#define _VEC_FUNC_CPX_2V( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,1S_,2,)

#define _VEC_FUNC_CPX_2V_T2( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,1S_,2,_T2)

#define _VEC_FUNC_CPX_3V_T2( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,,3,_T2)

#define _VEC_FUNC_CPX_3V_T3( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,,3,_T3)

#define _VEC_FUNC_CPX_3V_T1( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,,3,_T1)

#define _VEC_FUNC_CPX_3V( name, statement )				\
	OBJ_METHOD(name,statement,,CPX_,,3,)

#define _VEC_FUNC_CCR_3V( name, statement )			\
	OBJ_METHOD(name,statement,,CCR_,,3,)

#define _VEC_FUNC_CR_1S_2V( name, statement )	\
	OBJ_METHOD(name,statement,,CR_,1S_,2,)

#define _VEC_FUNC_CPX_1S_2V( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,1S_,2,)

#define _VEC_FUNC_CPX_1S_2V( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,1S_,2,)

#define _VEC_FUNC_CPX_1S_2V_T2( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,1S_,2,_T2)

#define _VEC_FUNC_CPX_1S_2V_T3( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,1S_,2,_T3)

#define _VEC_FUNC_CPX_1S_1V( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,1S_,1,)

#define _VEC_FUNC_SBM_CPX_3V(name, statement)		\
	OBJ_METHOD(name,statement,SBM_,CPX_,,3,)

#define _VEC_FUNC_SBM_CPX_1S_2V(name, statement)		\
	OBJ_METHOD(name,statement,SBM_,CPX_,1S_,2,)

#define _VEC_FUNC_SBM_CPX_2S_1V(name, statement)			\
	OBJ_METHOD(name,statement,SBM_,CPX_,2S_,1,)

#define _VEC_FUNC_SBM_QUAT_3V(name, statement)		\
	OBJ_METHOD(name,statement,SBM_,QUAT_,,3,)

#define _VEC_FUNC_SBM_QUAT_1S_2V(name, statement)		\
	OBJ_METHOD(name,statement,SBM_,QUAT_,1S_,2,)

#define _VEC_FUNC_SBM_QUAT_2S_1V(name, statement)			\
	OBJ_METHOD(name,statement,SBM_,QUAT_,2S_,1,)

//#define TWO_QUAT_VEC_METHOD( name,stat )	_VF_QUAT_2V( name, type_code, stat)
#define _VEC_FUNC_QUAT_2V( name, statement )				\
	OBJ_METHOD(name,statement,,QUAT_,,2,)

#define _VEC_FUNC_QUAT_2V_T4( name, statement )			\
	OBJ_METHOD(name,statement,,QUAT_,,2,_T4)

//#define THREE_QUAT_VEC_METHOD( name,stat )	_VF_QUAT_3V( name, type_code, stat)
#define _VEC_FUNC_QUAT_3V( name, statement )		\
	OBJ_METHOD(name,statement,,QUAT_,,3,)

#define _VEC_FUNC_QUAT_3V_T4( name, statement )		\
	OBJ_METHOD(name,statement,,QUAT_,,3,_T4)

#define _VEC_FUNC_QUAT_3V_T5( name, statement )		\
	OBJ_METHOD(name,statement,,QUAT_,,3,_T5)

#define _VEC_FUNC_QQR_3V( name, statement )			\
	OBJ_METHOD(name,statement,,QQR_,,3,)

#define _VEC_FUNC_QR_1S_2V( name, statement )	\
	OBJ_METHOD(name,statement,,QR_,1S_,2,)

#define _VEC_FUNC_QUAT_1S_2V( name, statement )			\
	OBJ_METHOD(name,statement,,QUAT_,1S_,2,)

#define _VEC_FUNC_QUAT_1S_2V_T4( name, statement )		\
	OBJ_METHOD(name,statement,,QUAT_,1S_,2,_T4)

#define _VEC_FUNC_QUAT_1S_2V_T5( name, statement )		\
	OBJ_METHOD(name,statement,,QUAT_,1S_,2,_T5)

#define _VEC_FUNC_QUAT_1S_1V( name, statement )			\
	OBJ_METHOD(name,statement,,QUAT_,1S_,1,)


/* PROJECTION_METHOD_2 is for vmaxv, vminv, vsum
 * Destination can be a scalar or we can collapse along any dimension...
 *
 * We have a similar issue for vmaxi, where we wish to return the index
 * of the max...
 */

/* We could do a fast loop when the destination is a scalar... */

/* INDEX_VDATA gets a pointer to the nth element in the array... */
/* It is a linear index, so we have to take it apart... */

#define INDEX_VDATA(index)	(orig_s1_ptr+	\
	(index%INDEX_COUNT(s1_count,0))*IDX_INC(s1inc,0)	\
+ ((index/INDEX_COUNT(s1_count,0))%INDEX_COUNT(s1_count,1))*IDX_INC(s1inc,1)	\
+ ((index/(INDEX_COUNT(s1_count,0)*INDEX_COUNT(s1_count,1)))%INDEX_COUNT(s1_count,2))*IDX_INC(s1inc,2) \
+ ((index/(INDEX_COUNT(s1_count,0)*INDEX_COUNT(s1_count,1)*INDEX_COUNT(s1_count,2)))%INDEX_COUNT(s1_count,3))*IDX_INC(s1inc,3) \
+ ((index/(INDEX_COUNT(s1_count,0)*INDEX_COUNT(s1_count,1)*INDEX_COUNT(s1_count,2)*INDEX_COUNT(s1_count,3)))%INDEX_COUNT(s1_count,4))*IDX_INC(s1inc,4))

#define _VEC_FUNC_2V_PROJ( name, init_statement, statement, gpu_expr )	\
									\
static void SLOW_NAME(name)(LINK_FUNC_ARG_DECLS)			\
SLOW_BODY_PROJ_2(name,init_statement,statement)




#define _VEC_FUNC_CPX_2V_PROJ( name, init_statement, statement, gpu_expr_re, gpu_expr_im )	\
								\
static void SLOW_NAME(name)(LINK_FUNC_ARG_DECLS)		\
SLOW_BODY_PROJ_CPX_2(name,init_statement,statement)


#define _VEC_FUNC_QUAT_2V_PROJ( name, init_statement, statement, expr_re, expr_im1, expr_im2, expr_im3 )	\
								\
static void SLOW_NAME(name)(LINK_FUNC_ARG_DECLS)		\
SLOW_BODY_PROJ_QUAT_2(name,init_statement,statement)



//#define RAMP2D_METHOD(name)
// BUG? can we use the statement args?  now they are not used
#define _VEC_FUNC_1V_3SCAL(name,s1,s2,s3)			\
								\
								\
static void SLOW_NAME(name)( LINK_FUNC_ARG_DECLS )		\
{								\
	dimension_t i,j;					\
	std_type val,row_start_val;				\
	dest_type *dst_ptr;	/* BUG init me */		\
	dest_type *row_ptr;					\
								\
	dst_ptr = (dest_type *)VA_DEST_PTR(vap);	\
	row_start_val = scalar1_val;				\
	row_ptr = dst_ptr;					\
	for(i=0; i < INDEX_COUNT(count,2); i ++ ){		\
		val = row_start_val;				\
		dst_ptr = row_ptr;				\
		for(j=0; j < INDEX_COUNT(count,1); j++ ){	\
			*dst_ptr = (dest_type) val;		\
								\
			dst_ptr += IDX_INC(dinc,1);		\
			val += scalar2_val;			\
		}						\
		row_ptr += IDX_INC(dinc,2);			\
		row_start_val += scalar3_val;			\
	}							\
}



#define _VEC_FUNC_2V_PROJ_IDX( name, init_statement, statement, gpu_s1, gpu_s2 )	\
								\
static void SLOW_NAME(name)(LINK_FUNC_ARG_DECLS)		\
SLOW_BODY_PROJ_IDX_2(name,init_statement,statement)





/* PROJECTION_METHOD_3 is for vdot
 * Destination can be a scalar or we can collapse along any dimension...
 */

// complex really not different?
// where is DECLARE_BASES?
#define _VEC_FUNC_CPX_3V_PROJ( name, init_statement, statement, gpu_r1, gpu_i1, gpu_r2, gpu_i2 )	\
	__VEC_FUNC_3V_PROJ(name,CPX_,init_statement,statement )

#define __VEC_FUNC_3V_PROJ( name, typ, init_statement, statement )	\
								\
static void SLOW_NAME(name)(LINK_FUNC_ARG_DECLS)		\
PROJ3_SLOW_BODY(name,typ,init_statement,statement)



#define _VEC_FUNC_3V_PROJ( name, init_statement, statement, gpu_e1, gpu_e2 )	\
								\
	__VEC_FUNC_3V_PROJ(name,,init_statement,statement )

/* bitmap conversion from another type */

//#define _VEC_FUNC_DBM_1V( name, statement )
//	OBJ_METHOD(name,statement,DBM_,,,1SRC,)

#define _VEC_FUNC_DBM_SBM( name, statement )			\
	OBJ_METHOD(name,statement,DBM_SBM_,,,,)

/* bitmap set from a constant */
#define _VEC_FUNC_DBM_1S( name, statement )			\
	OBJ_METHOD(name,statement,DBM_,,1S_,,)

/* used to set a bitmap based on a vector test */
/* vsm_gt etc */

#define _VEC_FUNC_VSMAP( name, op )		\
	OBJ_METHOD(name,SET_DBM_BIT(src1 op scalar1_val),DBM_,,1S_,1SRC,)


#define scalar1_val	VA_SCALAR_VAL_STD(vap,0)
#define scalar2_val	VA_SCALAR_VAL_STD(vap,1)
#define scalar3_val	VA_SCALAR_VAL_STD(vap,2)
#define cscalar1_val	VA_SCALAR_VAL_STDCPX(vap,0)
#define cscalar2_val	VA_SCALAR_VAL_STDCPX(vap,1)
#define cscalar3_val	VA_SCALAR_VAL_STDCPX(vap,2)
#define qscalar1_val	VA_SCALAR_VAL_STDQUAT(vap,0)
#define qscalar2_val	VA_SCALAR_VAL_STDQUAT(vap,1)
#define qscalar3_val	VA_SCALAR_VAL_STDQUAT(vap,2)
#define count		VA_DEST_DIMSET(vap)		// ->va_szi_p->szi_dst_dim
#define s1_count	VA_SRC1_DIMSET(vap)		// ->va_szi_p->szi_src_dim[0]
#define s2_count	VA_SRC2_DIMSET(vap)		// ->va_szi_p->szi_src_dim[1]
#define dbminc		VA_DEST_INCSET(vap)		// ->va_spi_p->spi_dst_incr
#define sbminc		VA_SRC5_INCSET(vap)		// ->va_spi_p->spi_src_incr[4]
#define dinc		VA_DEST_INCSET(vap)		// ->va_spi_p->spi_dst_incr
#define s1inc		VA_SRC1_INCSET(vap)		// ->va_spi_p->spi_src_incr[0]
#define s2inc		VA_SRC2_INCSET(vap)		// ->va_spi_p->spi_src_incr[1]
#define s3inc		VA_SRC3_INCSET(vap)		// ->va_spi_p->spi_src_incr[2]
#define s4inc		VA_SRC4_INCSET(vap)		// ->va_spi_p->spi_src_incr[3]

