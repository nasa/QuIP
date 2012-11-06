
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

/***************** Section 1 - definitions **********************/

#include "calling_args.h"	// declaration args, shared

#define FFT_METHOD_NAME(stem)	TYPED_NAME( _##stem )
#define OBJ_METHOD_NAME(stem)	TYPED_NAME( obj_##stem )
#define FAST_NAME(stem)		TYPED_NAME( fast_##stem )
#define SLOW_NAME(stem)		TYPED_NAME( slow_##stem )
#define EQSP_NAME(stem)		TYPED_NAME( eqsp_##stem )

#define TYPED_NAME(s)			_TYPED_NAME(TYP,s)
#define _TYPED_NAME(t,s)		__TYPED_NAME(t,s)
#define __TYPED_NAME(t,s)		t##_##s

#define TYPED_STRING(s)		STRINGIFY( TYPED_NAME(s) )
#define STRINGIFY(s)		_STRINGIFY(s)
#define _STRINGIFY(s)		#s

#define dst_dp		oap->oa_dest
#define src1_dp		oap->oa_dp[0]
#define src2_dp		oap->oa_dp[1]
#define src3_dp		oap->oa_dp[2]
#define src4_dp		oap->oa_dp[3]
#define bitmap_dst_dp	oap->oa_dest
#define bitmap_src_dp	oap->oa_bmap

//#define MAX_DEBUG

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

#define REPORT_SLOW_CALL						\
									\
	if( verbose ){							\
		sprintf(DEFAULT_ERROR_STRING,"Function %s calling slow func",db_func_name);	\
		NADVISE(DEFAULT_ERROR_STRING);					\
	}


#define FUNC_NAME(name)		const char * db_func_name=STRINGIFY( OBJ_METHOD_NAME(name) );

#else /* ! MAX_DEBUG */

#define ANNOUNCE_FUNCTION
#define REPORT_OBJ_METHOD_DONE
#define REPORT_FAST_CALL
#define REPORT_SLOW_CALL
#define FUNC_NAME(name)
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
"Shape mismatch between objects %s and %s",dp1->dt_name,dp2->dt_name);	\
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
		sprintf(DEFAULT_ERROR_STRING,					\
			"CAUTIOUS:  Null %s object!?",string);		\
		NERROR1(DEFAULT_ERROR_STRING);					\
	}


#define OBJ_ARG_CHK_SRC1	OBJ_ARG_CHK(src1_dp,"first source")
#define OBJ_ARG_CHK_SRC2	OBJ_ARG_CHK(src2_dp,"second source")
#define OBJ_ARG_CHK_SRC3	OBJ_ARG_CHK(src3_dp,"third source")
#define OBJ_ARG_CHK_SRC4	OBJ_ARG_CHK(src4_dp,"fourth source")

#define OBJ_ARG_CHK_2SRCS	OBJ_ARG_CHK_SRC1 OBJ_ARG_CHK_SRC2
#define OBJ_ARG_CHK_DBM_2SRCS	OBJ_ARG_CHK_DBM OBJ_ARG_CHK_2SRCS
#define OBJ_ARG_CHK_DBM_1SRC	OBJ_ARG_CHK_DBM OBJ_ARG_CHK_SRC1

#else /* ! CAUTIOUS */

#define OBJ_ARG_CHK_1							\
									\
	ANNOUNCE_FUNCTION

#define OBJ_ARG_CHK_SRC1
#define OBJ_ARG_CHK_SRC2
#define OBJ_ARG_CHK_SRC3
#define OBJ_ARG_CHK_SRC4

#endif /* CAUTIOUS */


#define OBJ_ARG_CHK_
#define OBJ_ARG_CHK_2		OBJ_ARG_CHK_1 OBJ_ARG_CHK_SRC1
#define OBJ_ARG_CHK_3		OBJ_ARG_CHK_2 OBJ_ARG_CHK_SRC2
#define OBJ_ARG_CHK_4		OBJ_ARG_CHK_3 OBJ_ARG_CHK_SRC3
#define OBJ_ARG_CHK_5		OBJ_ARG_CHK_4 OBJ_ARG_CHK_SRC4

#define OBJ_ARG_CHK_SBM_1		OBJ_ARG_CHK_SBM OBJ_ARG_CHK_1
#define OBJ_ARG_CHK_SBM_2		OBJ_ARG_CHK_SBM OBJ_ARG_CHK_2
#define OBJ_ARG_CHK_SBM_3		OBJ_ARG_CHK_SBM OBJ_ARG_CHK_3

/* DONE with CHECK definitions */


#define ADVANCE_SRC1		s1_ptr++ ;
#define ADVANCE_SRC2		s2_ptr++ ;
#define ADVANCE_SRC3		s3_ptr++ ;
#define ADVANCE_SRC4		s4_ptr++ ;
#define ADVANCE_1		dst_ptr++ ;
#define ADVANCE_CPX_1		cdst_ptr++ ;
#define ADVANCE_CPX_SRC1	cs1_ptr++ ;
#define ADVANCE_CPX_SRC2	cs2_ptr++ ;
#define ADVANCE_CPX_SRC3	cs3_ptr++ ;
#define ADVANCE_CPX_SRC4	cs4_ptr++ ;
#define ADVANCE_QUAT_1		qdst_ptr++ ;
#define ADVANCE_QUAT_SRC1	qs1_ptr++ ;
#define ADVANCE_QUAT_SRC2	qs2_ptr++ ;
#define ADVANCE_QUAT_SRC3	qs3_ptr++ ;
#define ADVANCE_QUAT_SRC4	qs4_ptr++ ;
#define ADVANCE_BITMAP		which_bit ++ ;

#define ADVANCE_2		ADVANCE_1 ADVANCE_SRC1
#define ADVANCE_3		ADVANCE_2 ADVANCE_SRC2
#define ADVANCE_4		ADVANCE_3 ADVANCE_SRC3
#define ADVANCE_5		ADVANCE_4 ADVANCE_SRC4

#define ADVANCE_CPX_2		ADVANCE_CPX_1 ADVANCE_CPX_SRC1
#define ADVANCE_CPX_3		ADVANCE_CPX_2 ADVANCE_CPX_SRC2
#define ADVANCE_CPX_4		ADVANCE_CPX_3 ADVANCE_CPX_SRC3
#define ADVANCE_CPX_5		ADVANCE_CPX_4 ADVANCE_CPX_SRC4
#define ADVANCE_CCR_3		ADVANCE_CPX_2 ADVANCE_SRC2
#define ADVANCE_CR_2		ADVANCE_CPX_1 ADVANCE_SRC1
#define ADVANCE_RC_2		ADVANCE_1 ADVANCE_CPX_SRC1

#define ADVANCE_QUAT_2		ADVANCE_QUAT_1 ADVANCE_QUAT_SRC1
#define ADVANCE_QUAT_3		ADVANCE_QUAT_2 ADVANCE_QUAT_SRC2
#define ADVANCE_QUAT_4		ADVANCE_QUAT_3 ADVANCE_QUAT_SRC3
#define ADVANCE_QUAT_5		ADVANCE_QUAT_4 ADVANCE_QUAT_SRC4
#define ADVANCE_QQR_3		ADVANCE_QUAT_2 ADVANCE_SRC2
#define ADVANCE_QR_2		ADVANCE_QUAT_1 ADVANCE_SRC1
#define ADVANCE_RQ_2		ADVANCE_1 ADVANCE_QUAT_SRC1

#define ADVANCE_SBM_1		ADVANCE_BITMAP ADVANCE_1
#define ADVANCE_SBM_2		ADVANCE_BITMAP ADVANCE_2
#define ADVANCE_SBM_3		ADVANCE_BITMAP ADVANCE_3
#define ADVANCE_SBM_CPX_1	ADVANCE_BITMAP ADVANCE_CPX_1
#define ADVANCE_SBM_CPX_2	ADVANCE_BITMAP ADVANCE_CPX_2
#define ADVANCE_SBM_CPX_3	ADVANCE_BITMAP ADVANCE_CPX_3
#define ADVANCE_SBM_QUAT_1	ADVANCE_BITMAP ADVANCE_QUAT_1
#define ADVANCE_SBM_QUAT_2	ADVANCE_BITMAP ADVANCE_QUAT_2
#define ADVANCE_SBM_QUAT_3	ADVANCE_BITMAP ADVANCE_QUAT_3

#define ADVANCE_DBM_		ADVANCE_BITMAP
#define ADVANCE_DBM_1SRC	ADVANCE_BITMAP ADVANCE_SRC1
#define ADVANCE_DBM_2SRCS	ADVANCE_BITMAP ADVANCE_2SRCS
#define ADVANCE_2SRCS		ADVANCE_SRC1 ADVANCE_SRC2

/* Stuff for projection loops */


#define DECLARE_LOOP_COUNT				\
	count_type loop_count[N_DIMENSIONS];		\
	int i_dim;

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
	for(i_dim=0;i_dim<N_DIMENSIONS;i_dim++) loop_count[i_dim]=1;

/* INC_BASE is what we do at the end of a nested loop... */

#define INC_BASE(which_dim,base_array,inc_array)		\
								\
	base_array[which_dim-1] += inc_array[which_dim];

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

#define INC_BASES_SBM(which_dim)	INC_BASE(which_dim,sbm_bit0,sbminc)
#define INC_BASES_DBM(which_dim)	INC_BASE(which_dim,dbm_bit0,dbminc)
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
#define INC_BASES_IDX(which_dim)	index_base[which_dim-1] += s1_count[which_dim-1];

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
#define INC_BASES_DBM_2SRCS( index )	INC_BASES_DBM(index) INC_BASES_2SRCS(index)
#define COPY_BASES_DBM_1SRC(index)	COPY_BASES_DBM(index) COPY_BASES_SRC1(index)
#define COPY_BASES_2SRCS(index)		COPY_BASES_SRC1(index) COPY_BASES_SRC2(index)
#define COPY_BASES_DBM_2SRCS(index)	COPY_BASES_DBM_1SRC(index) COPY_BASES_2SRCS(index)

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
		(base_array)[which_dim-1] = (type *)(dp)->dt_data;


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

#define INIT_PTRS_SBM		which_bit = sbm_bit0[0];
#define INIT_PTRS_DBM		which_bit = dbm_bit0[0];
#define INIT_PTRS_DBM_		INIT_PTRS_DBM

#define INIT_PTRS_DBM_1SRC	INIT_PTRS_DBM INIT_PTRS_SRC1
#define INIT_PTRS_2SRCS		INIT_PTRS_SRC1 INIT_PTRS_SRC2
#define INIT_PTRS_DBM_2SRCS	INIT_PTRS_DBM_1SRC INIT_PTRS_2SRCS

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

#define INC_PTRS_SRC1		s1_ptr += s1inc[0];
#define INC_PTRS_SRC2		s2_ptr += s2inc[0];
#define INC_PTRS_SRC3		s3_ptr += s3inc[0];
#define INC_PTRS_SRC4		s4_ptr += s4inc[0];
/* BUG? here we seem to assume that all bitmaps are contiguous - but
 * dobj allows bitmap subimages...
 */
#define INC_PTRS_SBM		which_bit++;
#define INC_PTRS_DBM		which_bit++;
#define INC_PTRS_DBM_		INC_PTRS_DBM

#define INC_PTRS_DBM_1SRC	INC_PTRS_DBM INC_PTRS_SRC1
#define INC_PTRS_2SRCS		INC_PTRS_SRC1 INC_PTRS_SRC2
#define INC_PTRS_DBM_2SRCS	INC_PTRS_DBM_1SRC INC_PTRS_2SRCS

#define INC_PTRS_SBM_1		INC_PTRS_1 INC_PTRS_SBM
#define INC_PTRS_SBM_2		INC_PTRS_2 INC_PTRS_SBM
#define INC_PTRS_SBM_3		INC_PTRS_3 INC_PTRS_SBM
#ifdef FOOBAR
#define INC_PTRS_SBM_CPX_1	INC_PTRS_CPX_1 INC_PTRS_SBM
#define INC_PTRS_SBM_CPX_2	INC_PTRS_CPX_2 INC_PTRS_SBM
#define INC_PTRS_SBM_CPX_3	INC_PTRS_CPX_3 INC_PTRS_SBM
#define INC_PTRS_SBM_QUAT_1	INC_PTRS_QUAT_1 INC_PTRS_SBM
#define INC_PTRS_SBM_QUAT_2	INC_PTRS_QUAT_2 INC_PTRS_SBM
#define INC_PTRS_SBM_QUAT_3	INC_PTRS_QUAT_3 INC_PTRS_SBM
#endif /* FOOBAR */

#define INC_PTRS_1		dst_ptr += dinc[0];
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
 */

#define SETBIT( value )							\
									\
	if( value )							\
		*(dbm_ptr + (which_bit/BITS_PER_BITMAP_WORD)) |=	\
			NUMBERED_BIT(which_bit); \
	else								\
		*(dbm_ptr + (which_bit/BITS_PER_BITMAP_WORD)) &=	\
			~ NUMBERED_BIT(which_bit);

#define DEBUG_SBM_	\
sprintf(DEFAULT_ERROR_STRING,"sbm_ptr = 0x%lx   which_bit = %d",\
(int_for_addr)sbm_ptr,which_bit);\
NADVISE(DEFAULT_ERROR_STRING);

#define DEBUG_DBM_	\
sprintf(DEFAULT_ERROR_STRING,"dbm_ptr = 0x%lx   which_bit = %d",\
(int_for_addr)dbm_ptr,which_bit);\
NADVISE(DEFAULT_ERROR_STRING);

#define srcbit		((*(sbm_ptr + (which_bit/BITS_PER_BITMAP_WORD)))\
			& NUMBERED_BIT(which_bit))


#define INIT_BASES_X_1(dsttyp)	dst_base[3]=(dsttyp *)vap->va_dst_vp;

#define INIT_BASES_X_SRC1(srctyp)	s1_base[3]=(srctyp *)vap->va_src_vp[0];

#define INIT_BASES_1		dst_base[3]=(dest_type *)vap->va_dst_vp;
#define INIT_BASES_SRC1		s1_base[3]=(std_type *)vap->va_src_vp[0];
#define INIT_BASES_SRC2		s2_base[3]=(std_type *)vap->va_src_vp[1];
#define INIT_BASES_SRC3		s3_base[3]=(std_type *)vap->va_src_vp[2];
#define INIT_BASES_SRC4		s4_base[3]=(std_type *)vap->va_src_vp[3];

#define INIT_BASES_CPX_1	cdst_base[3]=(dest_cpx *)vap->va_dst_vp;
#define INIT_BASES_CPX_SRC1	cs1_base[3]=(std_cpx *)vap->va_src_vp[0];
#define INIT_BASES_CPX_SRC2	cs2_base[3]=(std_cpx *)vap->va_src_vp[1];
#define INIT_BASES_CPX_SRC3	cs3_base[3]=(std_cpx *)vap->va_src_vp[2];
#define INIT_BASES_CPX_SRC4	cs4_base[3]=(std_cpx *)vap->va_src_vp[3];

#define INIT_BASES_QUAT_1	qdst_base[3]=(dest_quat *)vap->va_dst_vp;
#define INIT_BASES_QUAT_SRC1	qs1_base[3]=(std_quat *)vap->va_src_vp[0];
#define INIT_BASES_QUAT_SRC2	qs2_base[3]=(std_quat *)vap->va_src_vp[1];
#define INIT_BASES_QUAT_SRC3	qs3_base[3]=(std_quat *)vap->va_src_vp[2];
#define INIT_BASES_QUAT_SRC4	qs4_base[3]=(std_quat *)vap->va_src_vp[3];

#define INIT_BASES_IDX_1	INIT_BASES_X_1(index_type)  index_base[3]=0;
#define INIT_BASES_IDX_2	INIT_BASES_IDX_1 INIT_BASES_SRC1


#define INIT_BASES_X_2(dsttyp,srctyp)		INIT_BASES_X_1(dsttyp) INIT_BASES_X_SRC1(srctyp)

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

/* We don't actually use the bases for destination bitmaps... */

#define INIT_BASES_DBM				\
	dbm_ptr= vap->va_dst_vp;	\
	dbm_base[3]= vap->va_dst_vp;	\
	dbm_bit0[3]=vap->va_bit0;

#define INIT_BASES_SBM				\
	sbm_ptr= vap->va_src_vp[4];	\
	sbm_base[3]= vap->va_src_vp[4];	\
	sbm_bit0[3]=vap->va_bit0;

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
#define INIT_BASES_DBM_2	INIT_BASES_DBM INIT_BASES_2SRCS
#define INIT_BASES_DBM_2SRCS	INIT_BASES_DBM INIT_BASES_2SRCS

#define INIT_COUNT( var, index ) _INIT_COUNT(var,count,index)

#define _INIT_COUNT( var, array, index ) var=array[index];

#define COPY_BASES_1(index)	dst_base[index] = dst_base[index+1];
#define COPY_BASES_CPX_1(index)	cdst_base[index] = cdst_base[index+1];
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
	dbm_bit0[index] = dbm_bit0[index+1];

#define COPY_BASES_SBM(index)			\
						\
	sbm_base[index] = sbm_base[index+1];	\
	sbm_bit0[index] = sbm_bit0[index+1];


#define COPY_BASES_XXX_1	COPY_BASES_1
#define COPY_BASES_XXX_2	COPY_BASES_2
#define COPY_BASES_XXX_3	COPY_BASES_3
#define COPY_BASES_SBM_XXX_1	COPY_BASES_SBM_1
#define COPY_BASES_SBM_XXX_2	COPY_BASES_SBM_2
#define COPY_BASES_SBM_XXX_3	COPY_BASES_SBM_3
#define COPY_BASES_X_2		COPY_BASES_2


#define DECLARE_BASES_SBM			\
	bitmap_word *sbm_base[N_DIMENSIONS-1];	\
	bitmap_word *sbm_ptr;			\
	int which_bit;				\
	int sbm_bit0[N_DIMENSIONS-1];

#define DECLARE_BASES_DBM_	DECLARE_BASES_DBM

#define DECLARE_BASES_DBM			\
	bitmap_word *dbm_base[N_DIMENSIONS-1];	\
	int dbm_bit0[N_DIMENSIONS-1];		\
	int which_bit;				\
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
#define DECLARE_BASES_DBM_2SRCS		DECLARE_BASES_DBM DECLARE_BASES_2SRCS
#define DECLARE_BASES_2SRCS		DECLARE_VBASE_SRC1 DECLARE_VBASE_SRC2

#define DECLARE_BASES_X_2(dsttyp,srctyp)	\
		DECLARE_X_DST_VBASE(dsttyp)	\
		DECLARE_XXX_SRC1_VBASE(srctyp)

#define DECL_VEC_ARGS_STRUCT(name)				\
	Vector_Args va1;					\
	Spacing_Info spi1;					\
	Size_Info szi1;						\
	FUNC_NAME(name)

#define MIN( n1, n2 )		((n1)<(n2)?(n1):(n2))

/* The loop_arr array holds the max of all the dimensions - either 1 or N_i.
 * If we encounter a dimension which is not N_i or 1, then it's an error.
 */

#define ADJ_COUNTS(loop_arr,obj_arr)				\
								\
for(i_dim=0;i_dim<N_DIMENSIONS;i_dim++){			\
	if( obj_arr[i_dim] > 1 ){				\
		if( loop_arr[i_dim] == 1 ){			\
			loop_arr[i_dim] = obj_arr[i_dim];	\
		} else {					\
			if( obj_arr[i_dim] != loop_arr[i_dim] ){\
				count_type n;			\
								\
				n = MIN(loop_arr[i_dim],	\
						obj_arr[i_dim]);\
				sprintf(DEFAULT_ERROR_STRING,		\
	"Oops: %s count mismatch, (%d != %d), using %d",	\
					dimension_name[i_dim],	\
					loop_arr[i_dim],	\
					obj_arr[i_dim],n);	\
				loop_arr[i_dim] = n;		\
			}					\
			/* else loop_arr already has value */	\
		}						\
	}							\
}

#define SHOW_BASES			\
sprintf(DEFAULT_ERROR_STRING,"s1_ptr:  0x%lx",(int_for_addr)s1_ptr);\
NADVISE(DEFAULT_ERROR_STRING);\
sprintf(DEFAULT_ERROR_STRING,"bm_ptr:  0x%lx, which_bit = %d",(int_for_addr)bm_ptr,which_bit);\
NADVISE(DEFAULT_ERROR_STRING);\
sprintf(DEFAULT_ERROR_STRING,"s1_base:  0x%lx  0x%lx  0x%lx  0x%lx",(int_for_addr)s1_base[0],(int_for_addr)s1_base[1],(int_for_addr)s1_base[2],(int_for_addr)s1_base[3]);\
NADVISE(DEFAULT_ERROR_STRING);


#define OBJ_METHOD_DECL(name)					\
								\
	void OBJ_METHOD_NAME(name)( Vec_Obj_Args *oap )


#include "veclib/fast_test.h"



#define IMPOSSIBLE_METHOD( name )					\
									\
	OBJ_METHOD_DECL(name)						\
	{								\
		/*FUNC_NAME(name)*/					\
		sprintf(DEFAULT_ERROR_STRING,					\
	"%s:  Sorry, this operation is impossible.",			\
			STRINGIFY( TYPED_NAME(obj_##name) ) );		\
		NWARN(DEFAULT_ERROR_STRING);					\
	}



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

#define EXTLOC_STATEMENT( augment_condition,			\
				restart_condition, assignment )	\
								\
	if(restart_condition) {					\
		nocc=0;						\
		dst_ptr = orig_dst;				\
		assignment;					\
	}							\
	if( augment_condition ){				\
		CHECK_IDXVEC_OVERFLOW				\
		assignment;					\
		dst = index;					\
		dst_ptr++;					\
		nocc++;						\
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
count[0],count[1],count[2],count[3],count[4]);\
NADVISE(DEFAULT_ERROR_STRING);\
sprintf(DEFAULT_ERROR_STRING,"sbminc = %d %d %d %d %d",\
sbminc[0],sbminc[1],sbminc[2],sbminc[3],sbminc[4]);\
NADVISE(DEFAULT_ERROR_STRING);\
sprintf(DEFAULT_ERROR_STRING,"dbminc = %d %d %d %d %d",\
dbminc[0],dbminc[1],dbminc[2],dbminc[3],dbminc[4]);\
NADVISE(DEFAULT_ERROR_STRING);

#define GENERIC_SLOW_BODY( name, statement, decls, inits,	\
	copy_macro, ptr_init, comp_inc, inc_macro, debug_it )	\
								\
{								\
	decls							\
	inits							\
								\
/*SHOW_SLOW_COUNT*/ 						\
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

#define DEBUG_2							\
sprintf(DEFAULT_ERROR_STRING,"executing dst = 0x%lx   src = 0x%lx",\
(int_for_addr)dst_ptr,\
(int_for_addr)s1_ptr);\
NADVISE(DEFAULT_ERROR_STRING);

#define DEBUG_2SRCS							\
sprintf(DEFAULT_ERROR_STRING,"executing src1 = 0x%lx   src2 = 0x%lx",\
(int_for_addr)s1_ptr,\
(int_for_addr)s2_ptr);\
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

#define DEBUG_CPX_3	\
sprintf(DEFAULT_ERROR_STRING,"executing dst = 0x%lx   src1 = 0x%lx  src2 = 0x%lx",\
(int_for_addr)cdst_ptr,\
(int_for_addr)cs1_ptr,\
(int_for_addr)cs2_ptr);\
NADVISE(DEFAULT_ERROR_STRING);

#define SLOW_BODY_1(name,statement)	SIMPLE_SLOW_BODY(name,statement,,1,)
#define SLOW_BODY_2(name,statement)	SIMPLE_SLOW_BODY(name,statement,,2,)
#define SLOW_BODY_3(name,statement)	SIMPLE_SLOW_BODY(name,statement,,3,)
#define SLOW_BODY_CPX_1(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CPX_,1,)
#define SLOW_BODY_CPX_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CPX_,2,)
#define SLOW_BODY_CPX_3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CPX_,3,)
#define SLOW_BODY_QUAT_1(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QUAT_,1,)
#define SLOW_BODY_QUAT_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QUAT_,2,)
#define SLOW_BODY_QUAT_3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QUAT_,3,)
#define SLOW_BODY_SBM_CPX_1(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,SBM_,CPX_,1,)
#define SLOW_BODY_SBM_CPX_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,SBM_,CPX_,2,)
#define SLOW_BODY_SBM_CPX_3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,SBM_,CPX_,3,)
#define SLOW_BODY_SBM_QUAT_1(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,SBM_,QUAT_,1,)
#define SLOW_BODY_SBM_QUAT_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,SBM_,QUAT_,2,)
#define SLOW_BODY_SBM_QUAT_3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,SBM_,QUAT_,3,)
#define SLOW_BODY_CR_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CR_,2,)
#define SLOW_BODY_QR_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QR_,2,)
#define SLOW_BODY_CCR_3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,CCR_,3,)
#define SLOW_BODY_QQR_3(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,QQR_,3,)
#define SLOW_BODY_RC_2(name,statement)	SIMPLE_XXX_SLOW_BODY(name,statement,,RC_,2,)
#define SLOW_BODY_4(name,statement)	SIMPLE_SLOW_BODY(name,statement,,4,)
#define SLOW_BODY_5(name,statement)	SIMPLE_SLOW_BODY(name,statement,,5,)
#define SLOW_BODY_DBM_(name,statement)	SIMPLE_SLOW_BODY(name,statement,,DBM_,)
#define SLOW_BODY_DBM_1SRC(name,statement)	SIMPLE_SLOW_BODY(name,statement,,DBM_1SRC,)
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

#define SIMPLE_XXX_SLOW_BODY(name, stat,bitmap,typ,suffix,debugit)	\
								\
	GENERIC_XXX_SLOW_BODY( name, stat,			\
		DECLARE_BASES_##bitmap##typ##suffix,		\
		INIT_BASES_##bitmap##typ##suffix,		\
		COPY_BASES_##bitmap##typ##suffix,		\
		INIT_PTRS_##bitmap##typ##suffix,		\
		INC_BASES_##bitmap##typ##suffix,		\
		debugit	)

#define SLOW_BODY_XXX_2( name, stat, dsttyp, srctyp )		\
								\
	SIMPLE_XXX_SLOW_BODY(name, stat,dsttyp,srctyp,,2,)

#define SLOW_BODY_SBM_XXX_3( name, stat, dsttyp, srctyp )	\
								\
	SIMPLE_XXX_SLOW_BODY(name, stat,dsttyp,srctyp,SBM_,3,)


#define SLOW_BODY_SBM_XXX_2( name, stat, dsttyp, srctyp )	\
								\
	SIMPLE_XXX_SLOW_BODY(name, stat,dsttyp,srctyp,SBM_,2,)

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
 */


#define SLOW_BODY_PROJ_2( name, init_statement, statement )		\
									\
{									\
	PROJ_LOOP_DECLS_2						\
									\
	INIT_LOOP_COUNT							\
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
	orig_s1_ptr= (std_type *)vap->va_src_vp[0];	\
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


#define PROJ3_SLOW_BODY( name, init_statement, statement )		\
									\
{									\
	PROJ_LOOP_DECLS_3						\
									\
	INIT_LOOP_COUNT							\
									\
	ADJ_COUNTS(loop_count,count)					\
	ADJ_COUNTS(loop_count,s1_count)					\
	ADJ_COUNTS(loop_count,s2_count)					\
									\
	NEW_PLOOP_3( init_statement, count )				\
	NEW_PLOOP_3( statement, loop_count )				\
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



#define NEW_PLOOP_3( statement, count_arr )				\
									\
	INIT_BASES_3							\
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
void SLOW_NAME(name)( SLOW_ARGS_PTR )					\
{									\
	DECLARE_VBASE_SRC1						\
	DECLARE_FIVE_LOOP_INDICES					\
	EXTLOC_DECLS							\
	const char * func_name=#name;					\
									\
	INIT_BASES_SRC1							\
	s1_ptr = (std_type *) vap->va_src1;				\
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

#define FF_DECL(name)	void FAST_NAME(name)
#define SF_DECL(name)	void SLOW_NAME(name)


#define MOV_FF_DECL(name, statement,bitmap,typ,scalars,vectors)	\
								\
	FF_DECL(name)( FAST_ARGS_PTR )				\
	FAST_BODY_##typ##MOV( name, typ )

#define GENERIC_FF_DECL(name, statement,bitmap,typ,scalars,vectors)	\
								\
	FF_DECL(name)( FAST_ARGS_PTR )	\
	FAST_BODY_##bitmap##typ##vectors( name, statement )

#define GENERIC_SF_DECL(name,statement,bitmap,typ,scalars,vectors)	\
								\
	SF_DECL(name)( SLOW_ARGS_PTR )	\
	SLOW_BODY_##bitmap##typ##vectors( name, statement )


#define GENERIC_FUNC_DECLS(name,statement,bitmap,typ,scalars,vectors)	\
								\
	GENERIC_FF_DECL(name,statement,bitmap,typ,scalars,vectors)	\
	GENERIC_SF_DECL(name,statement,bitmap,typ,scalars,vectors)


#define MOV_FUNC_DECLS(name,statement,bitmap,typ,scalars,vectors)	\
								\
	MOV_FF_DECL(name,statement,bitmap,typ,scalars,vectors)	\
	GENERIC_SF_DECL(name,statement,bitmap,typ,scalars,vectors)


#define IDXRES_FAST_FUNC(name,init_statement,statement)		\
								\
FF_DECL(name)(IDX_PTR_ARG,PTR_ARGS_SRC1,COUNT_ARG)	\
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
	dst_ptr = (index_type *)vap->va_dst_vp;		\
	orig_dst = dst_ptr;				\
	nocc = 0;					\
	index = 0;					\
	extval = src1;					\
	idx_len = vap->va_sval[0].u_ul;


#define SET_EXTLOC_RETURN_SCALARS			\
	vap->va_sval[0].std_scalar = extval;		\
	vap->va_sval[1].u_ul = nocc;


#define EXTLOC_FAST_FUNC(name, statement)				\
									\
FF_DECL(name)( FAST_ARGS_PTR )						\
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
		ADVANCE_SRC1						\
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
#define FAST_DECLS_SBM		int which_bit; bitmap_word *sbm_ptr;
#define FAST_DECLS_DBM		int which_bit; bitmap_word *dbm_ptr; dimension_t fl_ctr;
#define FAST_DECLS_DBM_1SRC	FAST_DECLS_DBM FAST_DECLS_SRC1
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


/* FAST_INIT sets up the local vars from the arguments passed */

#define FAST_INIT_1		dst_ptr = (dest_type *)vap->va_dst_vp;\
				FAST_INIT_COUNT

/* We used to divide by 2 or 4 for cpx and quat, but no longer needed. */
#define FAST_INIT_COUNT		fl_ctr = vap->va_len;
/*
#define FAST_INIT_COUNT_CPX	fl_ctr = vap->va_len;
#define FAST_INIT_COUNT_QUAT	fl_ctr = vap->va_len;
*/

#define FAST_INIT_SRC1		s1_ptr = (std_type *)vap->va_src_vp[0];
#define FAST_INIT_SRC2		s2_ptr = (std_type *)vap->va_src_vp[1];
#define FAST_INIT_SRC3		s3_ptr = (std_type *)vap->va_src_vp[2];
#define FAST_INIT_SRC4		s4_ptr = (std_type *)vap->va_src_vp[3];
#define FAST_INIT_2		FAST_INIT_1	FAST_INIT_SRC1
#define FAST_INIT_3		FAST_INIT_2	FAST_INIT_SRC2
#define FAST_INIT_4		FAST_INIT_3	FAST_INIT_SRC3
#define FAST_INIT_5		FAST_INIT_4	FAST_INIT_SRC4
#define FAST_INIT_2SRCS		FAST_INIT_SRC1	FAST_INIT_SRC2

/* use 64 bit words for data movement when possible.  Pointers have
 * to be aligned to 8 byte boundaries.
 */

#define L_ALIGNMENT(a)		(((int_for_addr)a) & 7)

#define FAST_INIT_CPX_1		cdst_ptr = (dest_cpx *)vap->va_dst_vp;	\
				FAST_INIT_COUNT
#define FAST_INIT_CPX_SRC1	cs1_ptr = (std_cpx *)vap->va_src_vp[0];
#define FAST_INIT_CPX_SRC2	cs2_ptr = (std_cpx *)vap->va_src_vp[1];
#define FAST_INIT_CPX_SRC3	cs3_ptr = (std_cpx *)vap->va_src_vp[2];
#define FAST_INIT_CPX_SRC4	cs4_ptr = (std_cpx *)vap->va_src_vp[3];

#define FAST_INIT_CPX_2		FAST_INIT_CPX_1	FAST_INIT_CPX_SRC1
#define FAST_INIT_CPX_3		FAST_INIT_CPX_2	FAST_INIT_CPX_SRC2
#define FAST_INIT_CPX_4		FAST_INIT_CPX_3	FAST_INIT_CPX_SRC3
#define FAST_INIT_CPX_5		FAST_INIT_CPX_4	FAST_INIT_CPX_SRC4
#define FAST_INIT_CCR_3		FAST_INIT_CPX_2	FAST_INIT_SRC2
#define FAST_INIT_CR_2		FAST_INIT_CPX_1	FAST_INIT_SRC1
#define FAST_INIT_RC_2		FAST_INIT_1	FAST_INIT_CPX_SRC1

#define FAST_INIT_QUAT_1	qdst_ptr = (dest_quat *)vap->va_dst_vp;	\
				FAST_INIT_COUNT

#define FAST_INIT_QUAT_SRC1	qs1_ptr = (std_quat *)vap->va_src_vp[0];
#define FAST_INIT_QUAT_SRC2	qs2_ptr = (std_quat *)vap->va_src_vp[1];
#define FAST_INIT_QUAT_SRC3	qs3_ptr = (std_quat *)vap->va_src_vp[2];
#define FAST_INIT_QUAT_SRC4	qs4_ptr = (std_quat *)vap->va_src_vp[3];

#define FAST_INIT_QUAT_2	FAST_INIT_QUAT_1	FAST_INIT_QUAT_SRC1
#define FAST_INIT_QUAT_3	FAST_INIT_QUAT_2	FAST_INIT_QUAT_SRC2
#define FAST_INIT_QUAT_4	FAST_INIT_QUAT_3	FAST_INIT_QUAT_SRC3
#define FAST_INIT_QUAT_5	FAST_INIT_QUAT_4	FAST_INIT_QUAT_SRC4
#define FAST_INIT_QQR_3		FAST_INIT_QUAT_2	FAST_INIT_SRC2
#define FAST_INIT_QR_2		FAST_INIT_QUAT_1	FAST_INIT_SRC1
#define FAST_INIT_RQ_2		FAST_INIT_1	FAST_INIT_QUAT_SRC1

#define FAST_INIT_DBM_		FAST_INIT_DBM

#define FAST_INIT_DBM		which_bit = vap->va_bit0;		\
				dbm_ptr=vap->va_dst_vp;	\
				FAST_INIT_COUNT

#define FAST_INIT_SBM		which_bit = vap->va_bit0;		\
				sbm_ptr=vap->va_src_vp[4];

#define FAST_INIT_DBM_2SRCS	FAST_INIT_DBM FAST_INIT_2SRCS
#define FAST_INIT_DBM_1SRC	FAST_INIT_DBM FAST_INIT_SRC1
#define FAST_INIT_SBM_1		FAST_INIT_SBM FAST_INIT_1
#define FAST_INIT_SBM_2		FAST_INIT_SBM FAST_INIT_2
#define FAST_INIT_SBM_3		FAST_INIT_SBM FAST_INIT_3
#define FAST_INIT_SBM_CPX_1	FAST_INIT_SBM FAST_INIT_CPX_1
#define FAST_INIT_SBM_CPX_2	FAST_INIT_SBM FAST_INIT_CPX_2
#define FAST_INIT_SBM_CPX_3	FAST_INIT_SBM FAST_INIT_CPX_3
#define FAST_INIT_SBM_QUAT_1	FAST_INIT_SBM FAST_INIT_QUAT_1
#define FAST_INIT_SBM_QUAT_2	FAST_INIT_SBM FAST_INIT_QUAT_2
#define FAST_INIT_SBM_QUAT_3	FAST_INIT_SBM FAST_INIT_QUAT_3

/* The fast body is pretty simple...  Should we try to unroll loops
 * to take advantage of SSE?  How do we help the compiler to do this?
 */

#define SIMPLE_FAST_BODY(name, statement,typ,suffix,debugit)	\
							\
{							\
	FAST_DECLS_##typ##suffix			\
	FAST_INIT_##typ##suffix				\
	while(fl_ctr-- > 0){				\
		debugit					\
		statement ;				\
		ADVANCE_##typ##suffix			\
	}						\
}

/* We don't want to do the special case if that already is the type? */

#define FAST_BODY_MOV(name,typ)				\
{							\
	FAST_DECLS_##typ##2				\
	FAST_INIT_##typ##2				\
							\
	TYPED_NAME(FAST_ESCAPE)				\
	while(fl_ctr-- > 0){				\
		dst = src1 ;				\
		ADVANCE_##typ##2			\
	}						\
}

#define spdp_FAST_ESCAPE	/* no fast move for mixed types */
#define ubyin_FAST_ESCAPE	/* no fast move for mixed types */
#define uindi_FAST_ESCAPE	/* no fast move for mixed types */
#define inby_FAST_ESCAPE	/* no fast move for mixed types */
#define li_FAST_ESCAPE		/* no fast move for int64 */
#define uli_FAST_ESCAPE		/* no fast move for int64 */
#define dp_FAST_ESCAPE		/* no fast move for dp */

#define by_FAST_ESCAPE		GEN_FAST_ESCAPE(8)
#define uby_FAST_ESCAPE		GEN_FAST_ESCAPE(8)
#define in_FAST_ESCAPE		GEN_FAST_ESCAPE(4)
#define uin_FAST_ESCAPE		GEN_FAST_ESCAPE(4)

#if __WORDSIZE == 64
#define di_FAST_ESCAPE		GEN_FAST_ESCAPE(2)
#define udi_FAST_ESCAPE		GEN_FAST_ESCAPE(2)
#define sp_FAST_ESCAPE		GEN_FAST_ESCAPE(2)
#else	/* __WORDSIZE == 32 */
#define di_FAST_ESCAPE		/* this is the machine size */
#define udi_FAST_ESCAPE		/* this is the machine size */
#define sp_FAST_ESCAPE		/* this is the machine size */
#endif	/* __WORDSIZE == 32 */

#define GEN_FAST_ESCAPE( elts_per_word )		\
							\
	if( fl_ctr > (elts_per_word*4) && 		\
	L_ALIGNMENT(dst_ptr) == L_ALIGNMENT(s1_ptr) ){	\
		uint64_t *ldp, *lsp;			\
		int n_pre,n_post;			\
		if( L_ALIGNMENT(dst_ptr) != 0 ){	\
			n_pre = elts_per_word-L_ALIGNMENT(dst_ptr);	\
			fl_ctr -= n_pre;		\
			while(n_pre--)			\
				*dst_ptr++ = *s1_ptr++;	\
		}					\
		n_post = fl_ctr % elts_per_word;	\
		fl_ctr = fl_ctr / elts_per_word;	\
		ldp = (uint64_t *)dst_ptr;		\
		lsp = (uint64_t *)s1_ptr;		\
		while( fl_ctr-- )			\
			*ldp++ = *lsp++;		\
		if( n_post > 0 ){			\
			dst_ptr = (std_type *) ldp;	\
			s1_ptr = (std_type *) lsp;	\
			while( n_post-- )		\
				*dst_ptr++ = *s1_ptr++;	\
		}					\
		return;					\
	}


#define FAST_BODY_CONV_2(name, statement,dsttyp,srctyp)	\
							\
{							\
	dsttyp *dst_ptr;				\
	srctyp *s1_ptr;					\
	dimension_t fl_ctr;				\
	dst_ptr = (dsttyp *)vap->va_dst_vp;		\
	s1_ptr = (srctyp *)vap->va_src_vp[0];		\
	fl_ctr = vap->va_len;				\
	while(fl_ctr-- > 0){				\
		statement ;				\
		ADVANCE_2				\
	}						\
}

/* There ought to be a more compact way to do all of this? */

#define FAST_BODY_2(name, statement)		SIMPLE_FAST_BODY(name, statement,,2,)
#define FAST_BODY_3( name, statement )		SIMPLE_FAST_BODY(name, statement,,3,)
#define FAST_BODY_4( name, statement )		SIMPLE_FAST_BODY(name, statement,,4,)
#define FAST_BODY_5( name, statement )		SIMPLE_FAST_BODY(name, statement,,5,)
#define FAST_BODY_SBM_1(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_1,)
#define FAST_BODY_SBM_2(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_2,)
#define FAST_BODY_SBM_3(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_3,)
#define FAST_BODY_SBM_CPX_1(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_CPX_1,)
#define FAST_BODY_SBM_CPX_2(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_CPX_2,)
#define FAST_BODY_SBM_CPX_3(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_CPX_3,)
#define FAST_BODY_SBM_QUAT_1(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_QUAT_1,)
#define FAST_BODY_SBM_QUAT_2(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_QUAT_2,)
#define FAST_BODY_SBM_QUAT_3(name, statement)	SIMPLE_FAST_BODY(name, statement,,SBM_QUAT_3,)
#define FAST_BODY_BM_1(name, statement)		SIMPLE_FAST_BODY(name, statement,,BM_1,)
#define FAST_BODY_DBM_1SRC(name, statement)	SIMPLE_FAST_BODY(name, statement,,DBM_1SRC,)
#define FAST_BODY_DBM_2SRCS(name, statement)	SIMPLE_FAST_BODY(name, statement,,DBM_2SRCS,)
#define FAST_BODY_DBM_(name, statement)		SIMPLE_FAST_BODY(name, statement,,DBM_,)
#define FAST_BODY_1( name, statement )		SIMPLE_FAST_BODY(name, statement,,1,)
#define FAST_BODY_CPX_1( name, statement )	SIMPLE_FAST_BODY(name, statement,CPX_,1,)
#define FAST_BODY_CPX_2(name, statement)	SIMPLE_FAST_BODY(name, statement,CPX_,2,)
#define FAST_BODY_CPX_3( name, statement )	SIMPLE_FAST_BODY(name, statement,CPX_,3,)
#define FAST_BODY_CPX_4( name, statement )	SIMPLE_FAST_BODY(name, statement,CPX_,4,)
#define FAST_BODY_CPX_5( name, statement )	SIMPLE_FAST_BODY(name, statement,CPX_,5,)
#define FAST_BODY_CCR_3( name, statement )	SIMPLE_FAST_BODY(name, statement,CCR_,3,)
#define FAST_BODY_CR_2( name, statement )	SIMPLE_FAST_BODY(name, statement,CR_,2,)
#define FAST_BODY_RC_2( name, statement )	SIMPLE_FAST_BODY(name, statement,RC_,2,)
#define FAST_BODY_QUAT_1( name, statement )	SIMPLE_FAST_BODY(name, statement,QUAT_,1,)
#define FAST_BODY_QUAT_2(name, statement)	SIMPLE_FAST_BODY(name, statement,QUAT_,2,)
#define FAST_BODY_QUAT_3( name, statement )	SIMPLE_FAST_BODY(name, statement,QUAT_,3,)
#define FAST_BODY_QUAT_4( name, statement )	SIMPLE_FAST_BODY(name, statement,QUAT_,4,)
#define FAST_BODY_QUAT_5( name, statement )	SIMPLE_FAST_BODY(name, statement,QUAT_,5,)
#define FAST_BODY_QQR_3( name, statement )	SIMPLE_FAST_BODY(name, statement,QQR_,3,)
#define FAST_BODY_QR_2( name, statement )	SIMPLE_FAST_BODY(name, statement,QR_,2,)
#define FAST_BODY_RQ_2( name, statement )	SIMPLE_FAST_BODY(name, statement,RQ_,2,)


/********************** Section 7 - fast/slow switches **************/

#define SHOW_VEC_ARGS(speed)			\
	sprintf(DEFAULT_ERROR_STRING,"%s:  va1.va_dst_vp = 0x%lx",#speed,(int_for_addr)va1.va_dst_vp);\
	NADVISE(DEFAULT_ERROR_STRING);					\
	sprintf(DEFAULT_ERROR_STRING,"%s:  va1.va_src_vp[0] = 0x%lx",#speed,(int_for_addr)va1.va_src_vp[0]);\
	NADVISE(DEFAULT_ERROR_STRING);					\
	sprintf(DEFAULT_ERROR_STRING,"%s:  va1.va_src_vp[1] = 0x%lx",#speed,(int_for_addr)va1.va_src_vp[1]);\
	NADVISE(DEFAULT_ERROR_STRING);					\

#define FAST_SWITCH_CONV( name, type1, type2 )		\
							\
if( FAST_TEST_2 ){					\
	XFER_FAST_ARGS_2				\
	CHAIN_CHECK( fast_##name )			\
} else {						\
	XFER_SLOW_ARGS_2				\
	CHAIN_CHECK( slow_##name )			\
}

#define SHOW_FAST_TEST_
#define SHOW_FAST_TEST_2SRCS
#define SHOW_FAST_TEST_1SRC
#define SHOW_FAST_TEST_4
#define SHOW_FAST_TEST_5

#define SHOW_FAST_TEST_1						\
	sprintf(DEFAULT_ERROR_STRING,"FAST_TEST_1:  %d",FAST_TEST_1?1:0);	\
	NADVISE(DEFAULT_ERROR_STRING);

#define SHOW_FAST_TEST_2						\
	SHOW_FAST_TEST_1						\
	sprintf(DEFAULT_ERROR_STRING,"FAST_TEST_SRC1:  %d",IS_CONTIGUOUS(src1_dp)?1:0);	\
	NADVISE(DEFAULT_ERROR_STRING);

#define SHOW_FAST_TEST_3						\
	SHOW_FAST_TEST_2						\
	sprintf(DEFAULT_ERROR_STRING,"FAST_TEST_SRC2:  %d",IS_CONTIGUOUS(src2_dp)?1:0);	\
	NADVISE(DEFAULT_ERROR_STRING);

#define GENERIC_FAST_SWITCH(name,bitmap,typ,scalars,vectors)	\
								\
if( FAST_TEST_##bitmap##vectors ){				\
	REPORT_FAST_CALL					\
	XFER_FAST_ARGS_##bitmap##typ##scalars##vectors		\
	CHAIN_CHECK( FAST_NAME(name) )				\
} else {							\
	REPORT_SLOW_CALL					\
	XFER_SLOW_ARGS_##bitmap##typ##scalars##vectors		\
	CHAIN_CHECK( SLOW_NAME(name) )				\
}

/* Why do we need these??? */
#define FAST_SWITCH_3( name )		GENERIC_FAST_SWITCH(name,,,,3)
#define FAST_SWITCH_4( name )		GENERIC_FAST_SWITCH(name,,,,4)
#define FAST_SWITCH_CPX_3( name )	GENERIC_FAST_SWITCH(name,,CPX_,,3)
#define FAST_SWITCH_5( name )		GENERIC_FAST_SWITCH(name,,,,5)

#include "veclib/xfer_args.h"

/*********************** Section 8 - object methods ***************/

#define GENERIC_OBJ_METHOD(name,bitmap,typ,scalars,vectors)	\
								\
OBJ_METHOD_DECL(name)						\
{								\
	DECL_VEC_ARGS_STRUCT(name)				\
	OBJ_ARG_CHK_##bitmap##vectors				\
	GENERIC_FAST_SWITCH(name,bitmap,typ,scalars,vectors)	\
	REPORT_OBJ_METHOD_DONE					\
}

#define OBJ_METHOD(name,statement,bitmap,typ,scalars,vectors)	\
								\
GENERIC_OBJ_METHOD(name,bitmap,typ,scalars,vectors)		\
								\
GENERIC_FUNC_DECLS(name, statement,bitmap,typ,scalars,vectors)

#define OBJ_MOV_METHOD(name,statement,bitmap,typ,scalars,vectors)\
								\
GENERIC_OBJ_METHOD(name,bitmap,typ,scalars,vectors)		\
								\
MOV_FUNC_DECLS(name, statement,bitmap,typ,scalars,vectors)



#define VV_SELECTION_METHOD(name, statement)	\
	OBJ_METHOD(name,statement,SBM_,,,3)



#define CPX_VV_SELECTION_METHOD(name, statement)		\
	OBJ_METHOD(name,statement,SBM_,CPX_,,3)


#define QUAT_VV_SELECTION_METHOD(name, statement)			\
	OBJ_METHOD(name,statement,SBM_,QUAT_,,3)


#define VS_SELECTION_METHOD(name, statement)			\
	OBJ_METHOD(name,statement,SBM_,,1S_,2)

#define SS_SELECTION_METHOD(name, statement)			\
	OBJ_METHOD(name,statement,SBM_,,2S_,1)


#define CPX_SS_SELECTION_METHOD(name, statement)			\
	OBJ_METHOD(name,statement,SBM_,CPX_,2S_,1)



#define QUAT_SS_SELECTION_METHOD(name, statement)			\
	OBJ_METHOD(name,statement,SBM_,QUAT_,2S_,1)



#define CPX_VS_SELECTION_METHOD(name, statement)		\
	OBJ_METHOD(name,statement,SBM_,CPX_,1S_,2)



#define QUAT_VS_SELECTION_METHOD(name, statement)		\
	OBJ_METHOD(name,statement,SBM_,QUAT_,1S_,2)


/* Start FIX here... */


#define ONE_VEC_METHOD( name, statement )				\
	OBJ_METHOD(name,statement,,,,1)


/* e.g. vset */

#define ONE_VEC_SCALAR_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,,1S_,1)

/* e.g. ramp1d */

#define ONE_VEC_2SCALAR_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,,2S_,1)


#define ONE_CPX_VEC_SCALAR_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,1S_,1)


#define ONE_QUAT_VEC_SCALAR_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,QUAT_,1S_,1)


/* this is for vmagsq, vatn2:  real result, cpx source */
#define TWO_MIXED_RC_VEC_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,RC_,,2)



#define TWO_MIXED_CR_VEC_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,CR_,,2)


#define THREE_VEC_2SCALAR_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,,2S_,3)


#define THREE_MIXED_VEC_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,CCR_,,3)

/* real and quaternion sources for a quaternion result */

#define THREE_QMIXD_VEC_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,QQR_,,3)


#define TWO_VEC_SCALAR_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,,1S_,2)


#define TWO_CPX_VEC_SCALAR_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,1S_,2)

#define TWO_CPXT_VEC_SCALAR_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,1S_,2)

#define TWO_CPXD_VEC_SCALAR_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,1S_,2)


#define TWO_QUAT_VEC_SCALAR_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,QUAT_,1S_,2)



#define TWO_VEC_3SCALAR_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,,3S_,2)
									\

#define TWO_VEC_MOV_METHOD( name, statement )				\
	OBJ_MOV_METHOD(name,statement,,,,2)

#define TWO_VEC_METHOD( name, statement )				\
	OBJ_METHOD(name,statement,,,,2)


#define TWO_QUAT_VEC_METHOD( name, statement )				\
	OBJ_METHOD(name,statement,,QUAT_,,2)

#define THREE_VEC_METHOD( name, statement )	\
	OBJ_METHOD(name,statement,,,,3)


#define FIVE_VEC_METHOD( name, statement )	\
	OBJ_METHOD(name,statement,,,,5)

#define THREE_QUAT_VEC_METHOD( name, statement )		\
	OBJ_METHOD(name,statement,,QUAT_,,3)



// BUG CPXT and CPXD use global vars, should be local to be thread-safe...

#define THREE_CPXT_VEC_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,,3)

#define THREE_CPXD_VEC_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,,CPX_,,3)

#define THREE_CPX_VEC_METHOD( name, statement )				\
	OBJ_METHOD(name,statement,,CPX_,,3)

#define TWO_CPX_VEC_METHOD( name, statement )				\
	OBJ_METHOD(name,statement,,CPX_,,2)

#define TWO_CPXT_VEC_METHOD( name, statement )				\
	OBJ_METHOD(name,statement,,CPX_,,2)

#define FOUR_VEC_SCALAR_METHOD( name, statement )			\
									\
	OBJ_METHOD(name,statement,,,1S_,4)


/* complex destination, real source; im part of dst set to zero... */

#define TWO_MIXED_CR_VEC_SCALAR_METHOD( name, statement )	\
	OBJ_METHOD(name,statement,,CR_,1S_,2)


/* quaternion destination, real source; im part of dst set to zero... */

#define TWO_QMIXD_QR_VEC_SCALAR_METHOD( name, statement )	\
	OBJ_METHOD(name,statement,,QR_,1S_,2)


/* PROJECTION_METHOD_2 is for vmaxv, vminv, vsum
 * Destination can be a scalar or we can collapse along any dimension...
 *
 * We have a similar issue for vmaxi, where we wish to return the index
 * of the max...
 */

/* We could do a fast loop when the destination is a scalar... */

/* INDEX_VDATA gets a pointer to the nth element in the array... */
/* It is a linear index, so we have to take it apart... */

#define INDEX_VDATA(index)	(orig_s1_ptr+(index%s1_count[0])*s1inc[0]	\
				+ ((index/s1_count[0])%s1_count[1])*s1inc[1]	\
			+ ((index/(s1_count[0]*s1_count[1]))%s1_count[2])*s1inc[2] \
		+ ((index/(s1_count[0]*s1_count[1]*s1_count[2]))%s1_count[3])*s1inc[3] \
	+ ((index/(s1_count[0]*s1_count[1]*s1_count[2]*s1_count[3]))%s1_count[4])*s1inc[4])



#define PROJECTION_METHOD_2( name, init_statement, statement )	\
								\
OBJ_METHOD_DECL(name)						\
{								\
	DECL_VEC_ARGS_STRUCT(name)				\
	OBJ_ARG_CHK_SRC1					\
	OBJ_ARG_CHK_1						\
	XFER_SLOW_ARGS_2					\
	CHAIN_CHECK( SLOW_NAME(name) )				\
}								\
void SLOW_NAME(name)(SLOW_ARGS_PTR)				\
SLOW_BODY_PROJ_2(name,init_statement,statement)




#define PROJECTION_METHOD_IDX_2( name, init_statement, statement )	\
								\
OBJ_METHOD_DECL(name)						\
{								\
	DECL_VEC_ARGS_STRUCT(name)				\
	OBJ_ARG_CHK_SRC1					\
	OBJ_ARG_CHK_1						\
	XFER_SLOW_ARGS_2					\
	CHAIN_CHECK( SLOW_NAME(name) )				\
}								\
void SLOW_NAME(name)(SLOW_ARGS_PTR)				\
SLOW_BODY_PROJ_IDX_2(name,init_statement,statement)




/* PROJECTION_METHOD_3 is for vdot
 * Destination can be a scalar or we can collapse along any dimension...
 */

#define PROJECTION_METHOD_3( name, init_statement, statement )	\
								\
OBJ_METHOD_DECL(name)						\
{								\
	DECL_VEC_ARGS_STRUCT(name)				\
	OBJ_ARG_CHK_SRC1					\
	OBJ_ARG_CHK_SRC2					\
	OBJ_ARG_CHK_1						\
	XFER_SLOW_ARGS_3					\
	CHAIN_CHECK( SLOW_NAME(name) )				\
}								\
void SLOW_NAME(name)(SLOW_ARGS_PTR)				\
PROJ3_SLOW_BODY(name,init_statement,statement)



#define CPX_PROJECTION_METHOD_2( name, init_statement, statement )	\
									\
OBJ_METHOD_DECL(name)							\
{									\
	DECL_VEC_ARGS_STRUCT(name)					\
	OBJ_ARG_CHK_SRC1						\
	OBJ_ARG_CHK_1							\
	XFER_SLOW_ARGS_CPX_2						\
	CHAIN_CHECK( SLOW_NAME(name) )				\
}									\
void SLOW_NAME(name)(SLOW_ARGS_PTR)					\
SLOW_BODY_PROJ_CPX_2(name,init_statement,statement)





#define CPX_PROJECTION_METHOD_3( name, init_statement, statement )	\
									\
OBJ_METHOD_DECL(name)							\
{									\
	DECL_VEC_ARGS_STRUCT(name)					\
	OBJ_ARG_CHK_SRC2						\
	OBJ_ARG_CHK_SRC1						\
	OBJ_ARG_CHK_1							\
	XFER_SLOW_ARGS_CPX_3						\
	CHAIN_CHECK( SLOW_NAME(name) )				\
}									\
void SLOW_NAME(name)(SLOW_ARGS_PTR)					\
SLOW_BODY_PROJ_CPX_3(name,init_statement,statement)



#define QUAT_PROJECTION_METHOD_2( name, init_statement, statement )	\
									\
OBJ_METHOD_DECL(name)							\
{									\
	DECL_VEC_ARGS_STRUCT(name)					\
	OBJ_ARG_CHK_SRC1						\
	OBJ_ARG_CHK_1							\
	XFER_SLOW_ARGS_QUAT_2						\
	CHAIN_CHECK( SLOW_NAME(name) )				\
}									\
void SLOW_NAME(name)(SLOW_ARGS_PTR)					\
SLOW_BODY_PROJ_QUAT_2(name,init_statement,statement)


/* vset for bitmap type */

#define SCALAR_BIT_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,DBM_,,1S,)

/* bitmap conversion from another type */

#define BITMAP_DST_ONE_VEC_METHOD( name, statement )			\
	OBJ_METHOD(name,statement,DBM_,,,1SRC)


/* bitmap conversion to another type */

#define BITMAP_SRC_CONVERSION_METHOD( name, statement )		\
	OBJ_METHOD(name,statement,SBM_,,,1)


/* used to set a bitmap based on a vector test */
/* vsm_gt etc */

#define BITMAP_DST_ONE_VEC_SCALAR_METHOD( name, op )		\
	OBJ_METHOD(name,SETBIT(src1 op scalar1_val),DBM_,,1S_,1SRC)


/* used to set a bitmap based on a vector test */
/* vvm_gt etc */

#define BITMAP_DST_TWO_VEC_METHOD( name, op )			\
	OBJ_METHOD(name,SETBIT(src1 op src2),DBM_,,,2SRCS)


#define RAMP2D_METHOD(name)					\
								\
OBJ_METHOD_DECL(name)						\
{								\
	DECL_VEC_ARGS_STRUCT(name)				\
	OBJ_ARG_CHK_1						\
								\
	if( dst_dp->dt_comps > 1 ){				\
		sprintf(DEFAULT_ERROR_STRING,				\
"%s:  Sorry, object %s has depth %d, can only handle depth 1 currently...",\
			#name, dst_dp->dt_name,dst_dp->dt_comps);\
		NWARN(DEFAULT_ERROR_STRING);				\
		return;						\
	}							\
	if( dst_dp->dt_frames > 1 ){				\
		sprintf(DEFAULT_ERROR_STRING,				\
"%s:  Sorry, object %s has %d frames, can only handle 1 frame currently...",\
			#name ,					\
			dst_dp->dt_name,dst_dp->dt_frames);	\
		NWARN(DEFAULT_ERROR_STRING);				\
		return;						\
	}							\
	XFER_SLOW_ARGS_3S_1					\
	CHAIN_CHECK( SLOW_NAME(name) )				\
}								\
								\
								\
void SLOW_NAME(name)( SLOW_ARGS_PTR )				\
{								\
	dimension_t i,j;					\
	std_type val,row_start_val;				\
	dest_type *dst_ptr;	/* BUG init me */		\
	dest_type *row_ptr;					\
								\
	dst_ptr = (dest_type *)vap->va_dst_vp;	\
	row_start_val = scalar1_val;				\
	row_ptr = dst_ptr;					\
	for(i=0; i < count[2]; i ++ ){				\
		val = row_start_val;				\
		dst_ptr = row_ptr;				\
		for(j=0; j < count[1]; j++ ){			\
			*dst_ptr = val;				\
								\
			dst_ptr += dinc[1];			\
			val += scalar2_val;			\
		}						\
		row_ptr += dinc[2];				\
		row_start_val += scalar3_val;			\
	}							\
}


#ifdef FOOBAR
/* This is the ramp1d object method - but where is the vector method? */

#define RAMP1D_METHOD(name)					\
								\
OBJ_METHOD_DECL(name)						\
{								\
	dimension_t i,n;					\
	incr_t inc;						\
	std_type val;						\
	std_type *ptr;						\
	FUNC_NAME(name)						\
								\
	OBJ_ARG_CHK_1						\
								\
	if( dst_dp->dt_comps > 1 ){				\
		sprintf(DEFAULT_ERROR_STRING,				\
"%s:  Sorry, object %s has depth %d, can only handle depth 1 currently...",\
			#name ,					\
			dst_dp->dt_name,dst_dp->dt_comps);	\
		NWARN(DEFAULT_ERROR_STRING);				\
		return;						\
	}							\
	if( dst_dp->dt_frames > 1 ){				\
		sprintf(DEFAULT_ERROR_STRING,				\
"%s:  Sorry, object %s has %d frames, can only handle 1 frame currently...",\
			#name ,					\
			dst_dp->dt_name,dst_dp->dt_frames);	\
		NWARN(DEFAULT_ERROR_STRING);				\
		return;						\
	}							\
								\
	if( dst_dp->dt_rows==1 ){				\
		n = dst_dp->dt_cols;				\
		inc = dst_dp->dt_pinc;				\
	} else if( dst_dp->dt_cols==1 ){			\
		n = dst_dp->dt_rows;				\
		inc = dst_dp->dt_rinc;				\
	} else {						\
		sprintf(DEFAULT_ERROR_STRING,				\
"vramp:  destination object %s is neither a row nor a column!?",	\
			dst_dp->dt_name);			\
		NWARN(DEFAULT_ERROR_STRING);				\
		return;						\
	}							\
								\
	val = scalar1_val;					\
	ptr = (std_type *)dst_dp->dt_data;			\
	for(i=0; i < n; i ++ ){					\
		*ptr = val;					\
								\
		ptr += inc;					\
		val += scalar2_val;				\
	}							\
	dst_dp->dt_flags |= DT_ASSIGNED;			\
}

#endif /* FOOBAR */

#define _REAL_CONVERSION(name,dsttyp,srctyp)			\
								\
void name ( Vec_Obj_Args *oap )					\
{								\
	DECL_VEC_ARGS_STRUCT(name)				\
	OBJ_ARG_CHK_2						\
	FAST_SWITCH_CONV(name,dsttyp,srctyp)			\
}								\
								\
void fast_##name(FAST_ARGS_PTR)					\
FAST_BODY_CONV_2(name,dst=src1,dsttyp,srctyp)			\
								\
void slow_##name(SLOW_ARGS_PTR)					\
SLOW_BODY_XX_2(name,dst=src1,dsttyp,srctyp)



#define REAL_CONVERSION( key1, type1, key2, type2 )			\
_REAL_CONVERSION( v##key2##2##key1, type1, type2 )

#define ALL_SIGNED_CONVERSIONS( key, type )			\
REAL_CONVERSION( key, type, by,  char    )			\
REAL_CONVERSION( key, type, in,  short   )			\
REAL_CONVERSION( key, type, di,  int32_t    )			\
REAL_CONVERSION( key, type, li,  int64_t    )

#define ALL_FLOAT_CONVERSIONS( key, type )			\
REAL_CONVERSION( key, type, sp, float   )			\
REAL_CONVERSION( key, type, dp, double  )

#define ALL_UNSIGNED_CONVERSIONS( key, type )			\
REAL_CONVERSION( key, type, uby, u_char  )			\
REAL_CONVERSION( key, type, uin, u_short )			\
REAL_CONVERSION( key, type, udi, uint32_t  )			\
REAL_CONVERSION( key, type, uli, uint64_t  )

#define FAST_SWITCH_EXTLOC(name)		\
						\
if( FAST_TEST_1SRC ){				\
	XFER_FAST_ARGS_EXTLOC			\
	CHAIN_CHECK( FAST_NAME(name) )		\
} else {					\
	XFER_SLOW_ARGS_EXTLOC			\
	CHAIN_CHECK( SLOW_NAME(name) )		\
}



#define EXTREMA_LOCATIONS_METHOD( name, augment_condition,		\
					restart_condition, assignment )	\
									\
OBJ_METHOD_DECL(name)							\
{									\
	DECL_VEC_ARGS_STRUCT(name)					\
	ANNOUNCE_FUNCTION						\
	OBJ_ARG_CHK_2							\
	/* Pass the index len in scalar1 */				\
	va1.va_sval[0].u_ul = oap->oa_dest->dt_n_type_elts;		\
	FAST_SWITCH_EXTLOC(name)					\
									\
	/* return extval in scalar1 */					\
	*((std_type *)oap->oa_sdp[0]->dt_data) = va1.va_sval[0].std_scalar;\
									\
	/* return nocc in scalar2 */					\
	*((index_type *)oap->oa_sdp[1]->dt_data) = va1.va_sval[1].u_ul;	\
									\
	oap->oa_sdp[0]->dt_flags |= DT_ASSIGNED;			\
	oap->oa_sdp[1]->dt_flags |= DT_ASSIGNED;			\
}									\
EXTLOC_FAST_FUNC(name,EXTLOC_STATEMENT(augment_condition,restart_condition,assignment))	\
EXTLOC_SLOW_FUNC(name,EXTLOC_STATEMENT(augment_condition,restart_condition,assignment))


#define scalar1_val	vap->va_sval[0].std_scalar
#define scalar2_val	vap->va_sval[1].std_scalar
#define scalar3_val	vap->va_sval[2].std_scalar
#define cscalar1_val	vap->va_sval[0].std_cpx_scalar
#define cscalar2_val	vap->va_sval[1].std_cpx_scalar
#define cscalar3_val	vap->va_sval[2].std_cpx_scalar
#define qscalar1_val	vap->va_sval[0].std_quat_scalar
#define qscalar2_val	vap->va_sval[1].std_quat_scalar
#define qscalar3_val	vap->va_sval[2].std_quat_scalar
#define count		vap->va_szi_p->szi_dst_dim
#define s1_count	vap->va_szi_p->szi_src_dim[0]
#define s2_count	vap->va_szi_p->szi_src_dim[1]
#define dbminc		vap->va_spi_p->spi_dst_incr
#define sbminc		vap->va_spi_p->spi_src_incr[4]
#define dinc		vap->va_spi_p->spi_dst_incr
#define s1inc		vap->va_spi_p->spi_src_incr[0]
#define s2inc		vap->va_spi_p->spi_src_incr[1]
#define s3inc		vap->va_spi_p->spi_src_incr[2]
#define s4inc		vap->va_spi_p->spi_src_incr[3]

