
#include "veclib/speed_undefs.h"

#define dst	eqsp_dst
#define src1	eqsp_src1
#define src2	eqsp_src2
#define src3	eqsp_src3
#define src4	eqsp_src4

#define cdst	eqsp_cdst
#define csrc1	eqsp_csrc1
#define csrc2	eqsp_csrc2
#define csrc3	eqsp_csrc3
#define csrc4	eqsp_csrc4

#define qdst	eqsp_qdst
#define qsrc1	eqsp_qsrc1
#define qsrc2	eqsp_qsrc2
#define qsrc3	eqsp_qsrc3
#define qsrc4	eqsp_qsrc4

#ifdef BUILD_FOR_CUDA

#define GENERIC_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)		\
									\
	GENERIC_EQSP_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)		\
	GENERIC_ELEN_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)

#define SLOW_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)

#define GENERIC_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)		\
								\
	GENERIC_EQSP_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)	\
	GENERIC_ELEN_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)

#define _VEC_FUNC_2V_CONV(n,type,statement)		\
							\
	_GENERIC_EQSP_CONV_FUNC(n,std_type,type)	\
	_GENERIC_ELEN_CONV_FUNC(n,std_type,type)

#else // ! BUILD_FOR_CUDA

// Why is it that only CUDA needs the len versions???

#define GENERIC_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)			\
										\
	GENERIC_EQSP_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)

#define SLOW_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)

#define GENERIC_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)		\
								\
	GENERIC_EQSP_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)	\


#define _VEC_FUNC_2V_CONV(n,type,statement)		\
							\
	_GENERIC_EQSP_CONV_FUNC(n,std_type,type)


#endif // ! BUILD_FOR_CUDA

// slow defn - almost
#define SET_INDICES_DBM		SET_DBM_TBL_INDEX							\
				SET_DBM_INDEX_ARRAY

#define SET_INDICES_DBM_1S_	SET_DBM_TBL_INDEX

#define SET_DBM_TBL_INDEX	tbl_idx = THREAD_INDEX_X;						\
  				i_dbm_word = dbm_info_p->word_tbl[tbl_idx].word_offset;

#define SET_DBM_INDEX_ARRAY										\
				dbmi = dbm_info_p->word_tbl[tbl_idx].first_bit_num;

#define DBM_EQSP_LEN_TEST	dbmi >= dbm_bit0  && dbmi < dbm_bit0+len

#define DECL_BASIC_INDICES_DBM	unsigned int i_dbm_bit;				\
				int i_dbm_word; bitmap_word dbm_bit;	\
				int tbl_idx;

#include "veclib/fast_eqsp_defs.h"


