
dnl the subsidiary defns need to come first!

/* slow_defs defining gpu_index_type */
define(`GPU_INDEX_TYPE',`SLOW_GPU_INDEX_TYPE')
dnl define(`DECLARE_KERN_ARGS_DBM',`KERNEL_ARG_QUALIFIER bitmap_word *dbm, DECLARE_KERN_ARGS_DBM_GPU_INFO')

define(`SCALE_INDEX',`					\
	$1.d5_dim[0] *= $2.d5_dim[0];					\
	$1.d5_dim[1] *= $2.d5_dim[1];					\
	$1.d5_dim[2] *= $2.d5_dim[2];					\
	$1.d5_dim[3] *= $2.d5_dim[3];					\
	$1.d5_dim[4] *= $2.d5_dim[4];					\
')


define(`dst',`slow_dst')
define(`src1',`slow_src1')
define(`src2',`slow_src2')
define(`src3',`slow_src3')
define(`src4',`slow_src4')

define(`cdst',`slow_cdst')
define(`csrc1',`slow_csrc1')
define(`csrc2',`slow_csrc2')
define(`csrc3',`slow_csrc3')
define(`csrc4',`slow_csrc4')

define(`qdst',`slow_qdst')
define(`qsrc1',`slow_qsrc1')
define(`qsrc2',`slow_qsrc2')
define(`qsrc3',`slow_qsrc3')
define(`qsrc4',`slow_qsrc4')

dnl	SET_INDEX(this_index)

// What if we have to have blocks in 2 or more dims?

define(`SET_INDEX',`							\
									\
	$1.d5_dim[0] = THREAD_INDEX_X;					\
	$1.d5_dim[1] = $1.d5_dim[0] / szarr.d5_dim[0];			\
	$1.d5_dim[2] = $1.d5_dim[1] / szarr.d5_dim[1];			\
	$1.d5_dim[3] = $1.d5_dim[2] / szarr.d5_dim[2];			\
	$1.d5_dim[4] = $1.d5_dim[3] / szarr.d5_dim[3];			\
	$1.d5_dim[0] %= szarr.d5_dim[0];				\
	$1.d5_dim[1] %= szarr.d5_dim[1];				\
	$1.d5_dim[2] %= szarr.d5_dim[2];				\
	$1.d5_dim[3] %= szarr.d5_dim[3];				\
	$1.d5_dim[4] %= szarr.d5_dim[4];				\
')

dnl this is for bitmaps with info...
define(`SET_INDICES_1SRC',`index2.d5_dim[0] = tbl_idx; /* BUG need to initialize index2! */')


// Slow versions of these not implemented yet...
define(`_VEC_FUNC_MM_NOCC',`')
define(`_VEC_FUNC_2V_PROJ',`')
define(`_VEC_FUNC_2V_PROJ',`')
define(`_VEC_FUNC_CPX_2V_PROJ',`')
define(`_VEC_FUNC_QUAT_2V_PROJ',`')
define(`_VEC_FUNC_2V_PROJ_IDX',`')
define(`_VEC_FUNC_3V_PROJ',`')
define(`_VEC_FUNC_CPX_3V_PROJ',`')

define(`IDX1',`(INDEX_SUM(index1))')
define(`IDX1_1',`(index1.d5_dim[1])')
define(`IDX1_2',`(index1.d5_dim[2])')
define(`INC1_1',`inc1.d5_dim[1]')
define(`INC1_2',`inc1.d5_dim[2]')

define(`BITMAP_ROW_IDX',`(i_dbm_word/words_per_row)')
define(`IDX_WITHIN_ROW',`((i_dbm_word % words_per_row)*BITS_PER_BITMAP_WORD)')
define(`WORDS_PER_FRAME',`(words_per_row * szarr.d5_dim[2])')
define(`WORDS_PER_SEQ',`(words_per_row * szarr.d5_dim[2] * szarr.d5_dim[3])')

define(`SET_INDICES_DBM_1S_',`SET_BASIC_INDICES_DBM')

define(`SET_INDICES_DBM',`				\
	SET_BASIC_INDICES_DBM				\
dnl	dbmi.d5_dim[0] = dbm_info_p->word_tbl[tbl_idx].first_indices[0];				\
dnl	dbmi.d5_dim[1] = dbm_info_p->word_tbl[tbl_idx].first_indices[1];				\
dnl	dbmi.d5_dim[2] = dbm_info_p->word_tbl[tbl_idx].first_indices[2];				\
dnl	dbmi.d5_dim[3] = dbm_info_p->word_tbl[tbl_idx].first_indices[3];				\
dnl	dbmi.d5_dim[4] = dbm_info_p->word_tbl[tbl_idx].first_indices[4];				\
')

define(`SET_BASIC_INDICES_DBM',`				\
	tbl_idx = THREAD_INDEX_X;				\
  	i_dbm_word = dbm_info_p->word_tbl[tbl_idx].word_offset;				\
')

define(`DECL_BASIC_INDICES_DBM',`				\
	unsigned int i_dbm_bit;					\
	int i_dbm_word; bitmap_word dbm_bit;			\
	int tbl_idx;						\
')


// We need to know if we should do this bit...
// From these definitions, it is not clear whether the rows are padded to be an 
// integral number of words...
//
// We assume that i_dbm_word is initilized to dbmi.x, before upscaling to the bit index.
// Here we add the row offset
// But when adjust is called, the y increment has already been scaled.
// should dbmi have more than one dimension or not???
define(`SET_SBM_WORD_IDX',`i_sbm_word=(sbmi.d5_dim[1]+sbmi.d5_dim[2])/BITS_PER_BITMAP_WORD;')

define(`srcbit',`(sbm[(INDEX_SUM(sbmi)+sbm_bit0)>>LOG2_BITS_PER_BITMAP_WORD] & NUMBERED_BIT((INDEX_SUM(sbmi)+sbm_bit0)&(BITS_PER_BITMAP_WORD-1)))')


ifdef(`BUILD_FOR_CUDA',`

define(`GENERIC_GPU_FUNC_CALL',`
	GENERIC_SLOW_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)
	GENERIC_SLEN_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)
')

define(`SLOW_GPU_FUNC_CALL',`
	GENERIC_SLOW_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)
	GENERIC_SLEN_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)
')


define(`GENERIC_VEC_FUNC_DBM',`
	GENERIC_SLOW_VEC_FUNC_DBM($1,$2,$3,$4,$5)
	GENERIC_SLEN_VEC_FUNC_DBM($1,$2,$3,$4,$5)
')

dnl _VEC_FUNC_2V_CONV(name,dest_type
define(`_VEC_FUNC_2V_CONV',`
	/* cuda vec_func_2v_conv */
	_GENERIC_SLOW_CONV_FUNC($1,std_type,$2)
	_GENERIC_SLEN_CONV_FUNC($1,std_type,$2)
')

dnl test it
dnl // slow_defs.m4 testing vec_func_2v_conv
dnl #ifdef FOOBAR
dnl // testing!
dnl _VEC_FUNC_2V_CONV(testconv,char)
dnl #endif // FOOBAR

',` dnl else // ! BUILD_FOR_CUDA

// Why is it that only CUDA needs the len versions???

define(`GENERIC_GPU_FUNC_CALL',`
	GENERIC_SLOW_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)
')

define(`SLOW_GPU_FUNC_CALL',`
	GENERIC_SLOW_VEC_FUNC($1,$2,$3,$4,$5,$6,$7)
')

define(`GENERIC_VEC_FUNC_DBM',`
	GENERIC_SLOW_VEC_FUNC_DBM($1,$2,$3,$4,$5)
')

define(`_VEC_FUNC_2V_CONV',`
	_GENERIC_SLOW_CONV_FUNC($1,std_type,$2)
')


') dnl endif // ! BUILD_FOR_CUDA
