
#include "veclib/speed_undefs.h"

#define dst	fast_dst
#define src1	fast_src1
#define src2	fast_src2
#define src3	fast_src3
#define src4	fast_src4

#define cdst	fast_cdst
#define csrc1	fast_csrc1
#define csrc2	fast_csrc2
#define csrc3	fast_csrc3
#define csrc4	fast_csrc4

#define qdst	fast_qdst
#define qsrc1	fast_qsrc1
#define qsrc2	fast_qsrc2
#define qsrc3	fast_qsrc3
#define qsrc4	fast_qsrc4

#define GENERIC_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)			\
										\
	GENERIC_FLEN_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)

#define SLOW_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)

#define GENERIC_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)		\
								\
	GENERIC_FLEN_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)

#define _VEC_FUNC_2V_CONV(n,type,statement)		\
							\
	_GENERIC_FLEN_CONV_FUNC(n,std_type,type)




#define DECL_BASIC_INDICES_DBM	unsigned int i_dbm_bit;				\
				int i_dbm_word; bitmap_word dbm_bit;		\
				int tbl_idx;



// slow defn - almost
#define SET_INDICES_DBM		SET_DBM_TBL_INDEX							\
				SET_DBM_INDEX_ARRAY

#define SET_INDICES_DBM_1S_	SET_DBM_TBL_INDEX

#define SET_DBM_TBL_INDEX	tbl_idx = THREAD_INDEX_X;						\
  				i_dbm_word = dbm_info_p->word_tbl[tbl_idx].word_offset;

#define SET_DBM_INDEX_ARRAY										\
				dbmi = dbm_info_p->word_tbl[tbl_idx].first_bit_num;

// Because the fast version has lengths, we don't need flen version...
// No-ops
#define _VEC_FUNC_MM_NOCC( func_name, c1, c2, s1, gpu_c1, gpu_c2 )
#define _VEC_FUNC_2V_PROJ( func_name, s1, s2, gpu_expr )
#define _VEC_FUNC_CPX_2V_PROJ( func_name, s1, s2, gpu_re_expr, gpu_im_expr )
#define _VEC_FUNC_QUAT_2V_PROJ( func_name, s1, s2, gpu_r1, gpu_i1, gpu_i2, gpu_i3 )
#define _VEC_FUNC_2V_PROJ_IDX( func_name, s1, s2, gpu_s1, gpu_s2 )
#define _VEC_FUNC_3V_PROJ( func_name, s1, s2, gpu_e1, gpu_e2 )
#define _VEC_FUNC_CPX_3V_PROJ( func_name, s1, s2, gpu_r1, gpu_i1, gpu_r2, gpu_i2 )

#include "veclib/fast_eqsp_defs.h"

