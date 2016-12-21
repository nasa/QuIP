
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

#ifdef BUILD_FOR_CUDA

#define GENERIC_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)			\
										\
	GENERIC_FAST_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)			\
	GENERIC_FLEN_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)

#define SLOW_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)

#define GENERIC_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)		\
								\
	GENERIC_FAST_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)	\
	GENERIC_FLEN_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)

#define _VEC_FUNC_2V_CONV(n,type,statement)		\
							\
	_GENERIC_FAST_CONV_FUNC(n,std_type,type)	\
	_GENERIC_FLEN_CONV_FUNC(n,std_type,type)

#else // ! BUILD_FOR_CUDA

// Why is it that only CUDA needs the len versions???

#define GENERIC_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)			\
										\
	GENERIC_FAST_VEC_FUNC(fn,stat,bm,typ,sclrs,vecs,extra)

#define SLOW_VFUNC_CALL(fn,stat,bm,typ,sclrs,vecs,extra)

#define GENERIC_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)		\
								\
	GENERIC_FAST_VEC_FUNC_DBM(fn,stat,typ,sclrs,vecs)


#define _VEC_FUNC_2V_CONV(n,type,statement)		\
							\
	_GENERIC_FAST_CONV_FUNC(n,std_type,type)


#endif // ! BUILD_FOR_CUDA

#define SET_INDICES_DBM		SET_INDEX(dbmi)			\
				i_dbm_word = dbmi;		\
				dbmi *= BITS_PER_BITMAP_WORD;

#define DBM_FAST_LEN_TEST	dbmi >= dbm_bit0  && dbmi < dbm_bit0+len

#define DECL_BASIC_INDICES_DBM	unsigned int i_dbm_bit;				\
				int i_dbm_word; bitmap_word dbm_bit;

#define SET_INDICES_DBM_1S_	i_dbm_word = THREAD_INDEX_X;

#include "veclib/fast_eqsp_defs.h"

