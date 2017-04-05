
dnl	// This appears to be mostly gpu-related stuff???

dnl	// These definitions are shared between fast and eqsp functions

dnl	/* fast_eqsp_defs setting gpu_index_type */
define(`GPU_INDEX_TYPE',`int')

dnl ifdef BUILD_FOR_CUDA
dnl
dnl#define SET_INDEX( this_index )					\
dnl								\
dnl	this_index = blockIdx.x * blockDim.x + threadIdx.x;
dnl
dnl endif // BUILD_FOR_CUDA

define(`IDX1',`index1')
define(`SCALE_INDEX',$1 *= $2;)

dnl SET_INDEX(this_index)
define(`SET_INDEX',`$1 = THREAD_INDEX_X;')


dnl#define SET_BITMAP_WORD		i_word=(bmi.x+bmi.y)/BITS_PER_BITMAP_WORD;
dnl From these definitions, it is not clear whether the rows are padded to be an 
dnl integral number of words...
dnl
dnl We assume that i_dbm_word is initilized to dbm_bit_idx.x, before upscaling to the bit index.
dnl Here we add the row offset
dnl But when adjust is called, the y increment has already been scaled.
dnl should dbm_bit_idx have more than one dimension or not???
dnl#define ADJUST_DBM_WORD_IDX	i_dbm_word += ((dbm_bit_idx /* * dbm_inc.y */)/BITS_PER_BITMAP_WORD);
define(`SET_SBM_WORD_IDX',i_sbm_word=sbm_bit_idx/BITS_PER_BITMAP_WORD;)

define(`srcbit',(sbm_ptr[this_sbm_bit>>LOG2_BITS_PER_BITMAP_WORD] & NUMBERED_BIT(this_sbm_bit&(BITS_PER_BITMAP_WORD-1))))

