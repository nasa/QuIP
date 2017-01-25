
// These definitions are shared between fast and eqsp functions

#define GPU_INDEX_TYPE	int

#ifdef BUILD_FOR_OPENCL

#define SET_INDEX( this_index )			\
						\
	this_index = get_global_id(0);		\

#endif // BUILD_FOR_OPENCL

#ifdef BUILD_FOR_CUDA

#define SET_INDEX( this_index )					\
								\
	this_index = blockIdx.x * blockDim.x + threadIdx.x;

#endif // BUILD_FOR_CUDA

#define IDX1	index1		// used to be index1.x

#define SCALE_INDEX(idx,inc)	idx *= inc;


//#define SET_BITMAP_WORD		i_word=(bmi.x+bmi.y)/BITS_PER_BITMAP_WORD;
// From these definitions, it is not clear whether the rows are padded to be an 
// integral number of words...
//
// We assume that i_dbm_word is initilized to dbmi.x, before upscaling to the bit index.
// Here we add the row offset
// But when adjust is called, the y increment has already been scaled.
// should dbmi have more than one dimension or not???
//#define ADJUST_DBM_WORD_IDX	i_dbm_word += ((dbmi /* * dbm_inc.y */)/BITS_PER_BITMAP_WORD);
#define SET_SBM_WORD_IDX	i_sbm_word=sbmi/BITS_PER_BITMAP_WORD;

#define srcbit	(sbm[(sbmi+sbm_bit0)>>LOG2_BITS_PER_BITMAP_WORD] & \
		NUMBERED_BIT((sbmi+sbm_bit0)&(BITS_PER_BITMAP_WORD-1)))



