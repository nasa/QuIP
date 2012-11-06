
/*
 * We use the fast method only if everything is contiguous, and if the shapes
 * are identical.
 */

#define FAST_TEST_	0

#define FAST_TEST_CPX_1		FAST_TEST_1
#define EQSP_TEST_CPX_1		EQSP_TEST_1

#define FAST_TEST_1		( IS_CONTIGUOUS(dst_dp) )
#define EQSP_TEST_1		( IS_EVENLY_SPACED(dst_dp) )

#define FAST_TEST_1SRC		( IS_CONTIGUOUS(src1_dp) )
#define EQSP_TEST_1SRC		( IS_EVENLY_SPACED(src1_dp) )

#define FAST_TEST_2SRCS		( FAST_TEST_1SRC && IS_CONTIGUOUS(src2_dp) \
				&& dp_same_size_query(src1_dp,src2_dp) )

#define EQSP_TEST_2SRCS		( EQSP_TEST_1SRC && IS_EVENLY_SPACED(src2_dp) \
				&& dp_same_size_query(src1_dp,src2_dp) )

#define FAST_TEST_2		( FAST_TEST_1			\
				&& FAST_TEST_1SRC		\
				&& dp_same_size_query(src1_dp,dst_dp) )

#define FAST_TEST_CPX_2		FAST_TEST_2
#define EQSP_TEST_CPX_2		EQSP_TEST_2

#define FAST_TEST_RC_2		( FAST_TEST_1			\
				&& FAST_TEST_1SRC		\
				&& dp_same_size_query(dst_dp,src1_dp) )

#define EQSP_TEST_2		( EQSP_TEST_1			\
				&& EQSP_TEST_1SRC		\
				&& dp_same_size_query(src1_dp,dst_dp) )

#define EQSP_TEST_RC_2		( EQSP_TEST_1			\
				&& EQSP_TEST_1SRC		\
				&& dp_same_size_query(dst_dp,src1_dp) )

/* this is not used anywhere??? */
#define FAST_TEST_M_2		( FAST_TEST_1			\
				&& IS_CONTIGUOUS(src1_dp)	\
				&& dp_equal_dims(src1_dp,dst_dp,1,4) )

#define FAST_TEST_SBM_1		( IS_CONTIGUOUS(bitmap_src_dp)	\
				&& IS_CONTIGUOUS(dst_dp)	\
				&& dp_same_size_query(dst_dp,bitmap_src_dp) )

#define EQSP_TEST_SBM_1		( IS_EVENLY_SPACED(bitmap_src_dp)	\
				&& IS_EVENLY_SPACED(dst_dp)	\
				&& dp_same_size_query(dst_dp,bitmap_src_dp) )

/* we no longer assume contiguous bitmaps??? */

#define FAST_TEST_DBM_		( IS_CONTIGUOUS(bitmap_dst_dp) )

#define FAST_TEST_DBM_1SRC		( IS_CONTIGUOUS(bitmap_dst_dp)	\
					&& IS_CONTIGUOUS(src1_dp)	\
					&& dp_same_size_query(src1_dp,bitmap_dst_dp) )

#define FAST_TEST_DBM_2SRCS	( IS_CONTIGUOUS(bitmap_dst_dp)	\
					&& IS_CONTIGUOUS(src1_dp)	\
					&& IS_CONTIGUOUS(src2_dp)	\
					&& dp_same_size_query(src1_dp,bitmap_dst_dp) \
					&& dp_same_size_query(src1_dp,bitmap_dst_dp))

#define EQSP_TEST_DBM_		( IS_EVENLY_SPACED(bitmap_dst_dp) )

#define EQSP_TEST_DBM_1SRC		( IS_EVENLY_SPACED(bitmap_dst_dp)	\
					&& IS_EVENLY_SPACED(src1_dp)	\
					&& dp_same_size_query(src1_dp,bitmap_dst_dp) )

#define EQSP_TEST_DBM_2SRCS	( IS_EVENLY_SPACED(bitmap_dst_dp)	\
					&& IS_EVENLY_SPACED(src1_dp)	\
					&& IS_EVENLY_SPACED(src2_dp)	\
					&& dp_same_size_query(src1_dp,bitmap_dst_dp) \
					&& dp_same_size_query(src1_dp,bitmap_dst_dp))

#define FAST_TEST_3		( FAST_TEST_2			\
					&& IS_CONTIGUOUS(src2_dp)	\
					&& dp_same_size_query(src2_dp,dst_dp) )

#define EQSP_TEST_3		( EQSP_TEST_2 && IS_EVENLY_SPACED(src2_dp) \
					&& dp_same_size_query(src2_dp,dst_dp) )

#define FAST_TEST_CPX_3		( FAST_TEST_CPX_2		\
					&& IS_CONTIGUOUS(src2_dp)	\
					&& dp_same_size_query(src2_dp,dst_dp) )

#define EQSP_TEST_CPX_3		( EQSP_TEST_CPX_2 && IS_EVENLY_SPACED(src2_dp) \
					&& dp_same_size_query(src2_dp,dst_dp) )

#define FAST_TEST_QUAT_3	( FAST_TEST_2	\
					&& IS_CONTIGUOUS(src2_dp)	\
					&& dp_same_size_query(src2_dp,dst_dp) )

#define FAST_TEST_CCR_3		( FAST_TEST_2		\
					&& IS_CONTIGUOUS(src2_dp)	\
					&& dp_same_len(src2_dp,dst_dp) )

#define FAST_TEST_QQR_3		( FAST_TEST_2	\
					&& IS_CONTIGUOUS(src2_dp)	\
					&& dp_same_len(src2_dp,dst_dp) )

#define FAST_TEST_SBM_2		( FAST_TEST_2			\
					&& IS_CONTIGUOUS(bitmap_src_dp)	\
					&& dp_same_size_query(bitmap_src_dp,dst_dp) )

#define EQSP_TEST_SBM_2		( EQSP_TEST_2			\
					&& IS_EVENLY_SPACED(bitmap_src_dp)	\
					&& dp_same_size_query(bitmap_src_dp,dst_dp) )

#define FAST_TEST_CPX_SBM_2	( FAST_TEST_2		\
					&& IS_CONTIGUOUS(bitmap_src_dp)	\
					&& dp_same_size_query(bitmap_src_dp,dst_dp) )

#define FAST_TEST_QUAT_SBM_2	( FAST_TEST_2	\
					&& IS_CONTIGUOUS(bitmap_src_dp)	\
					&& dp_same_size_query(bitmap_src_dp,dst_dp) )

/* not used? */
#define FAST_TEST_SBM_3		( FAST_TEST_3		\
					&& IS_CONTIGUOUS(bitmap_src_dp)	\
					&& dp_same_size_query(bitmap_src_dp,dst_dp) )

#define EQSP_TEST_SBM_3		( EQSP_TEST_3		\
					&& IS_EVENLY_SPACED(bitmap_src_dp)	\
					&& dp_same_size_query(bitmap_src_dp,dst_dp) )

#define FAST_TEST_4		( FAST_TEST_3		\
					&& IS_CONTIGUOUS(src3_dp)	\
					&& dp_same_size_query(src3_dp,dst_dp) )

#define FAST_TEST_5		( FAST_TEST_4		\
					&& IS_CONTIGUOUS(src4_dp)	\
					&& dp_same_size_query(src4_dp,dst_dp) )

#define EQSP_TEST_4		( EQSP_TEST_3		\
					&& IS_EVENLY_SPACED(src3_dp)	\
					&& dp_same_size_query(src3_dp,dst_dp) )

#define EQSP_TEST_5		( EQSP_TEST_4		\
					&& IS_EVENLY_SPACED(src4_dp)	\
					&& dp_same_size_query(src4_dp,dst_dp) )


#ifndef dst_dp

#define dst_dp	oap->oa_dest
#define src1_dp	oap->oa_dp[0]
#define src2_dp	oap->oa_dp[1]
#define src3_dp	oap->oa_dp[2]
#define src4_dp	oap->oa_dp[3]
#define bitmap_src_dp	oap->oa_dp[4]
#define bitmap_dst_dp	oap->oa_dest

#endif /* ! dst_dp */


