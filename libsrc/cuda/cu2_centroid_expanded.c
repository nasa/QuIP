/* cu2_port.m4 BEGIN */
/* gen_port.m4 BEGIN */
#include "quip_prot.h"
#include "shape_bits.h"

/* NOT Suppressing ! */


/* gen_port.m4 DONE */



#define BUILD_FOR_CUDA
#define BUILD_FOR_GPU


/* Suppressing ! */

/* NOT Suppressing ! */


extern void *cu2_tmp_vec(Platform_Device *pdp, size_t size, size_t len, const char *whence);
extern void cu2_free_tmp(void *a, const char *whence);

/* cu2_port.m4 DONE */






/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/gpu_args.m4
/* gpu_args.m4 BEGIN */




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/gen_kern_args.m4

/* Suppressing ! */

/* NOT Suppressing ! */





/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/gen_kern_args.m4

































/* These are the arguments used in the declarations
 * of the kernel functions: DECLARE_KERN_ARGS_FAST_2 etc
 * When we call the kernel (cuda), we use KERN_ARGS_FAST_2,
 * and when we pass them one-by-one (openCL) we use SET_KERNEL_ARGS_2
 *
 * Really, we need macros to declare macros!?
 */

/******************* declarations and calling kernel arguments  ***************************/
/* used to be VFUNC_ARGS (for declarations), changed to DECLARE_KERN_ARGS */ 
// Special cases




// BUG - how can we insure that the declarations are consistent!?











// MM_ARGS or helper args?
/*#define KERN_ARGS_MM	dest,s1,s2,len1,len2*/



































// BUG?  where is KERN_ARGS_FAST_IDX_HELPER?








////////////// end of special cases

////////////// generic args for kernel declaration

// BUG?  we ran into problems with vector-scalar ops and spdp mixed precision type,
// because of inconsistency between the cpu kernel (which assumed std_type)
// and the arg fetching code (which assumed dest_type).  We tried fixing that
// by switching the kernel to use dest_type, but that broke vsatan2 (cuda),
// and doesn't make sense anyway...





















// complex stuff







// quaternion stuff








// EQSP real







// real, slow










//#define DECLARE_KERN_ARGS_SLOW_LEN',`GPU_INDEX_TYPE xyz_len')	// BUG vwxyz_len

// Now use szarr!
//#define DECLARE_KERN_ARGS_SLOW_LEN',`GPU_INDEX_TYPE vwxyz_len')

































/****************************************/


































/////////////////////////////























// quaternion, eqsp















////////////////////////////////////////

































////////////////

























// complex, with len











































// quaternion, with len








































//////////////////////////





























































///////////////////































































// real, source bitmap






// complex, source bitmap





// quaternion, source bitmap





/////////////////////

// real, source bitmap






// complex, source bitmap





// quaternion, source bitmap





/////////////////////

// real, source bitmap






// complex, source bitmap





// quaternion, source bitmap






/////////////////////

// real, source bitmap







// complex, source bitmap





// quaternion, source bitmap





/////////////////////

// real, source bitmap







// complex, source bitmap





// quaternion, source bitmap





/////////////////////

// real, source bitmap







// complex, source bitmap





// quaternion, source bitmap






/******************* KERN_ARGS ***************************/









///////////////////
// Building blocks











//









// complex stuff







// quaternion stuff







// EQSP real









// complex, eqsp

// real, slow

// these are dim3's, not the object increment sets...




















// Compound arg lists
/****************************************/


































/////////////////////////////























// quaternion, eqsp















////////////////////////////////////////

































////////////////

























// complex, with len











































// quaternion, with len








































//////////////////////////






















































































/////////////////////

// real, source bitmap






// complex, source bitmap





// quaternion, source bitmap





/////////////////////

// real, source bitmap






// complex, source bitmap





// quaternion, source bitmap





/////////////////////

// real, source bitmap






// complex, source bitmap





// quaternion, source bitmap






/////////////////////

// real, source bitmap






// complex, source bitmap





// quaternion, source bitmap





/////////////////////

// real, source bitmap






// complex, source bitmap





// quaternion, source bitmap





/////////////////////

// real, source bitmap






// complex, source bitmap





// quaternion, source bitmap






/* gpu_args.m4 END */




/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/gpu_args.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/gpu_call_utils.m4


/* NOT Suppressing ! */

/* gpu_call_utils.m4 BEGIN */

/* Suppressing ! */

/* NOT Suppressing ! */

/* gpu_call_utils.m4 END */




/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/gpu_call_utils.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/slow_defs.m4


/* slow_defs defining gpu_index_type */
























// What if we have to have blocks in 2 or more dims?


































// We need to know if we should do this bit...
// From these definitions, it is not clear whether the rows are padded to be an 
// integral number of words...
//
// We assume that i_dbm_word is initilized to dbm_bit_idx.x, before upscaling to the bit index.
// Here we add the row offset
// But when adjust is called, the y increment has already been scaled.
// should dbm_bit_idx have more than one dimension or not???























 


/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/slow_defs.m4





/* NOT Suppressing ! */

// BEGIN INCLUDED FILE cu2_centroid_defs.m4















/* NOT Suppressing ! */

// END INCLUDED FILE cu2_centroid_defs.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../cu2/cu2_kern_call_defs.m4
/* gen_gpu_calls.m4 BEGIN */
// 
// This file contains macros that are useful for writing kernels...
//
// A lot of this stuff is not platform specific!?




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/gpu_call_utils.m4


/* NOT Suppressing ! */

/* gpu_call_utils.m4 BEGIN */

/* Suppressing ! */

/* NOT Suppressing ! */

/* gpu_call_utils.m4 END */




/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/gpu_call_utils.m4


/**********************************************************************/

// args n, s  are func_name, statement

// 5 args





// this is vramp2d


// 3 args


















// vsm_gt etc


// this is vset

// where is cpx vset??

// Are these two redundant?
// this is bit_vset

// bit_vmov


// vand etc


// vsand etc























 










// args d,s1,s2 are dst_arg, src_arg1, src_arg2



// special case for left shift
// is just for cuda???





// PORT ?











// What is this???



/* These are for calls with a destination bitmap (vvm_lt etc)
 *
 * Here we cannot vectorize over all the pixels, because multiple
 * pixels share the same bitmap word.  Each thread has to set all the bits
 * in a given word.
 */


// This loops over all of the bits in one word.  We have a problem here if
// all of the bits are not used - there is no harm in reading or setting
// unused bits, but it might cause a seg violation when trying to access
// corresponding non-bit args???  BUG?




/* FLEN_DBM_LOOP */



/* EQSP_DBM_LOOP */




// len is a different type, but here we don't check the other len dimensions!?  BUG?
// We don't necessarily want to set all of the bits in the word, if there is
// a skipping increment?  So this probably won't work for subsamples?  BUG?


/* SLOW_DBM_LOOP */





























































/* FIXME still need to convert these to generic macros if possible */




// rvdot - we need temporary space for the products!?
// The first step should be a normal vmul...



/* CPX_FAST_2V_PROJ_SETUP */




/* CPX_FAST_2V_PROJ_HELPER */




// 2V_PROJ SETUP and HELPER do the same thing, but have different input types
// (only relevant for mixed operations, e.g. summing float to double








/* CPX_FAST_2V_PROJ_SETUP */



/* CPX_FAST_2V_PROJ_HELPER */





/* QUAT_FAST_2V_PROJ_SETUP */










// BUG? does this need to be two macros, one for setup and one for helper?





/* For nocc_setup, we index directly into the value and count temp arrays
 * (a and b, respectively), but we have to double the index for the source
 * array c, and the index array d.  Because we are comparing adjacent pairs, 
 */


// vsum, vdot, etc
// BUG this is hard-coded for vsum!?
//
// The idea is that , because all threads cannot access the destination simultaneously,
// we have to make the source smaller recursively...  But when we project
// to a vector instead of a scalar, we can to the elements of the vector in parallel...
// This is quite tricky.
//
// Example:  col=sum(m)
//
// m = | 1 2 3 4 |
//     | 5 6 7 8 |
//
// tmp = | 4  6  |
//       | 12 14 |
//
// col = | 10 |
//       | 26 |
     

// BUG - we need to make this do vmaxv and vminv as well.
// It's the same except for the sum line, which would be replaced with
//




// for vsum:   s1[(index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4])] + s2[(index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4])]
// for vmaxv:  psrc1 > psrc2 ? psrc1 : psrc2

/* after comment? */

// left shift was broken on cuda, what about now?



// vsum, vdot, etc
// BUG this is hard-coded for vsum!?
//
// The idea is that , because all threads cannot access the destination simultaneously,
// we have to make the source smaller recursively...  But when we project
// to a vector instead of a scalar, we can to the elements of the vector in parallel...
// This is quite tricky.
//
// Example:  col=sum(m)
//
// m = | 1 2 3 4 |
//     | 5 6 7 8 |
//
// tmp = | 4  6  |
//       | 12 14 |
//
// col = | 10 |
//       | 26 |
     

// BUG - we need to make this do vmaxv and vminv as well.
// It's the same except for the sum line, which would be replaced with
//






/* For nocc_setup, we index directly into the value and count temp arrays
 * (a and b, respectively), but we have to double the index for the source
 * array c, and the index array d.  Because we are comparing adjacent pairs, 
 *
 * We assume that the data are contiguous, and use fast (single) indices.
 */





/* For nocc_setup, we index directly into the value and count temp arrays
 * (a and b, respectively), but we have to double the index for the source
 * array c, and the index array d.  Because we are comparing adjacent pairs, 
 */






// How are we handling the indices???



// indices and stride example:
//
// src data			ext_val			indices				counts
// 0  1  5  5  5  2  2  2	1   5   5   2		1  X  2   3   4  X  6   7	1  2  1  2		setup, n=4
// 1  5  5  2			5   5			2  3 (2) (3)  4  X (6) (7)	2  1			helper, n=2, stride=4
// 5  5				5			2  3  4  (3) (4) X (6) (7)	3			helper, n=1, stride=8



/* GENERIC_FAST_VEC_FUNC */
/* 2V_PROJ is OK but not this??? */
/* that uses g_cu2_fast_type_code_... */
/* used to use g_cu2_fast_type_code_... */







// BUG change to CAN_INDEX_THREE_DIMS







// Does OpenCL have a limit (like CUDA) on the number of dimensions (3)?































// EQSP is tricky because the number of relevant bits in a word is no
// longer all of the bits - so the LOOP should just loop over the bits
// in a single word!?  BUG?
















// BUG use macro for helper name









/* gen_gpu_calls.m4 END */

// gpu_special_defs.m4 BEGIN

// vmaxg etc - require contiguous, fast only



// vmaxv, vminv, vsum

// on gpu only fast version, but on cpu only slow version!?




















// gpu_special_defs.m4 DONE


















/************** conversions **************/























































/* NOT Suppressing ! */

// END INCLUDED FILE ../cu2/cu2_kern_call_defs.m4





/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/sp_defs.m4
/* sp_defs.m4 BEGIN */























/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/sp_func_defs.m4

 
/* Suppressing ! */

/* NOT Suppressing ! */





/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/sp_func_defs.m4


/* TYPE_CODE = sp   dest_type =float */
/* sp_defs.m4 DONE */



/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/sp_defs.m4



__global__ void sp_slow_cent_helper
( /*float *x_array, dim3 inc1, float *y_array, dim3 inc2,
	float *input, dim3 inc3, dim3 len*/ dim5 szarr ,  float* a  , dim5 inc1 ,  float* b  , dim5 inc2 ,  float* c  , dim5 inc3  )

{
	/*dim3 index;*/
	/*uint32_t offset1, offset2, offset3;*/
	/* decl_indices_2 */ dim5 index1; dim5 index2; dim5 index3;
	float p;

	/*index.x = blockIdx.x * blockDim.x + threadIdx.x;*/
	/*index.y = blockIdx.y * blockDim.y + threadIdx.y;*/
								\
									\
	index1.d5_dim[0] = blockIdx.x * blockDim.x + threadIdx.x;					\
	index1.d5_dim[1] = index1.d5_dim[0] / szarr.d5_dim[0];			\
	index1.d5_dim[2] = index1.d5_dim[1] / szarr.d5_dim[1];			\
	index1.d5_dim[3] = index1.d5_dim[2] / szarr.d5_dim[2];			\
	index1.d5_dim[4] = index1.d5_dim[3] / szarr.d5_dim[3];			\
	index1.d5_dim[0] %= szarr.d5_dim[0];				\
	index1.d5_dim[1] %= szarr.d5_dim[1];				\
	index1.d5_dim[2] %= szarr.d5_dim[2];				\
	index1.d5_dim[3] %= szarr.d5_dim[3];				\
	index1.d5_dim[4] %= szarr.d5_dim[4];				\
 index2 = index1; index3 = index2;

	/*offset1 = index.y * inc1.x + index.x;*/
	/*offset2 = index.y * inc2.x + index.x;*/
	/*offset3 = index.y * inc3.x + index.x;*/

	/*p = *( c + offset3);*/
	p = c[index3.d5_dim[0]+index3.d5_dim[1]+index3.d5_dim[2]+index3.d5_dim[3]+index3.d5_dim[4]  ];	/* third arg, no first source */
	a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] = p * index3.d5_dim[1]; /* x */
	b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] = p * index3.d5_dim[2]; /* y */
	/* *(a+offset1) = p * index.x; */
	/* *(b+offset2) = p * index.y; */
}
			



/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/dp_defs.m4
























/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/dp_func_defs.m4


























/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/dp_func_defs.m4





/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/dp_defs.m4



__global__ void dp_slow_cent_helper
( /*double *x_array, dim3 inc1, double *y_array, dim3 inc2,
	double *input, dim3 inc3, dim3 len*/ dim5 szarr ,  double* a  , dim5 inc1 ,  double* b  , dim5 inc2 ,  double* c  , dim5 inc3  )

{
	/*dim3 index;*/
	/*uint32_t offset1, offset2, offset3;*/
	/* decl_indices_2 */ dim5 index1; dim5 index2; dim5 index3;
	double p;

	/*index.x = blockIdx.x * blockDim.x + threadIdx.x;*/
	/*index.y = blockIdx.y * blockDim.y + threadIdx.y;*/
								\
									\
	index1.d5_dim[0] = blockIdx.x * blockDim.x + threadIdx.x;					\
	index1.d5_dim[1] = index1.d5_dim[0] / szarr.d5_dim[0];			\
	index1.d5_dim[2] = index1.d5_dim[1] / szarr.d5_dim[1];			\
	index1.d5_dim[3] = index1.d5_dim[2] / szarr.d5_dim[2];			\
	index1.d5_dim[4] = index1.d5_dim[3] / szarr.d5_dim[3];			\
	index1.d5_dim[0] %= szarr.d5_dim[0];				\
	index1.d5_dim[1] %= szarr.d5_dim[1];				\
	index1.d5_dim[2] %= szarr.d5_dim[2];				\
	index1.d5_dim[3] %= szarr.d5_dim[3];				\
	index1.d5_dim[4] %= szarr.d5_dim[4];				\
 index2 = index1; index3 = index2;

	/*offset1 = index.y * inc1.x + index.x;*/
	/*offset2 = index.y * inc2.x + index.x;*/
	/*offset3 = index.y * inc3.x + index.x;*/

	/*p = *( c + offset3);*/
	p = c[index3.d5_dim[0]+index3.d5_dim[1]+index3.d5_dim[2]+index3.d5_dim[3]+index3.d5_dim[4]  ];	/* third arg, no first source */
	a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] = p * index3.d5_dim[1]; /* x */
	b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] = p * index3.d5_dim[2]; /* y */
	/* *(a+offset1) = p * index.x; */
	/* *(b+offset2) = p * index.y; */
}
			



/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/host_typed_call_defs.m4



// These definitions mostly turn into a GENERIC_HOST_TYPED_CALL...
//
// Somewhere we have a declaration like VFUNC_PROT_2V...




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/fast_test.m4











// When are these used???








































/* this is not used anywhere??? */



























































// BUG one of these should go...
// This is the new one


// quat tests



// This one was here before








/* not used? */




































/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/fast_test.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/xfer_args.m4

// The NOCC functions have a destination array of indices, and a source vector of inputs,
// plus two return scalars

// Can't use XFER_FAST_ARGS_2, because it uses oap->oa_dest for the count...
// But we need to know the length of the index array (dest)



































































// for rvmov, sbm is passed as b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ]?





















































































































































































































// XFER_DBM_GPU_INFO needs to be defined differently for cpu & gpu !!!


// can be shared with CUDA, should be moved?
// moved back to veclib/xfer_args.h, with  guard...













 














	/* no-op, mem is allocated at object creation... */
	




// Why don't we transfer the destination dimset???










































/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/xfer_args.m4


// args n, c, s  are func_name, dp, statement

// CONV replaces 2



// 5 args





// this is vramp2d


// 3 args















 
 


// vsm_gt etc


// this is vset

// where is cpx vset??


// this is bit_vset
// What is up with 1S vs 1S_ ???
// bit_vset




// bit_vmov



































// args d,s1,s2 are dst_arg, src_arg1, src_arg2



// GENERIC_HOST_TYPED_CALL declares four functions:
// First, fast, equally-spaced, and slow versions of the typed call
// Then a typed call which performs the fast test prior to calling
// one of the previously defined functions.
// The typed ones can be static?

// These are special cases that need to be coded by hand for gpu...




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/host_calls_special.m4
// These macros are used to build the host-side function calls, which typically
// involve a speed test, then a branch to a speed-specific host function that
// calls the appropriate kernel.
//
// This file contains "special" definitions that don't follow the usual pattern...
















// This was H_CALL_MM - ???
// The statements are used in the kernels, this just declares the function that fixes the args
// and then calls the kernel...










/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/host_calls_special.m4







/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/gpu_host_call_defs.m4


/* NOT Suppressing ! */

/* gpu_host_call_defs.m4 BEGIN */

/* Suppressing ! */

/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/report_args.m4

/**************** Things that do the actual printing *************/

	
/* MORE_DEBUG is not defined */





 
/************* Fast Args ***********/



















































/************* EqSp Args ***********/




















































/************* Slow Args ***********/
























































/************* Slen Args ***********/



















































/* end of report_args.m4 */



/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/report_args.m4












/* SETUP_KERNEL is a nop for CUDA */











/* setup_slow_len must be called outside of here, because the
 * lengths are inside vap...
 */






// is this used?













/* For vmaxg, the temp arrays don't double for the destination vector...
 * oa_sdp[0] is the extreme value, oa_sdp[1] is the count.
 */








// The index functions wants to return the index of the max/min.
// After the first pass, we will have a half-length array with the
// max indices.  The subsequent passes need to use these indices to
// lookup data to make the comparison...





/* Can we use the recursive strategy for vmaxg?
 *
 * We need a vector of occurrences for each thread...
 *
 * Let's assume that the destination vector (index array) has the same dimension
 * as the input.  So we can use portions of it for the temporary storage.
 * First do all the pairs.  We need an occurrence count array equal in size to
 * the number of pairs.  The first pass sets this - the number of occurrences
 * will be either 1 or 2.
 * The second pass will compare the values accessed indirectly (as in vmaxi).
 *
 * Unlike vmaxi, we don't compare far-away pairs, we do adjacent pairs.
 *
 *
 * How do we do the merge in general?  Let's try an example with 8 elements [ 6 6 3 6 5 6 7 8 ]
 * After the first pass:
 *	index_list	[ 0 1 3 . 5 . 7 . ]
 *	n_tmp		[ 2 1 1 1 ]
 *	v_tmp		[ 6 6 6 8 ]
 *
 * The nocc_helper function has a thread for each pair of values in v_tmp;
 * it compares the values, and then updates the new result n and v accordingly.
 * The hard part is copying the indices (if necessary).
 * We have to pass the addresses of the source and destination n and v arrays,
 * plus the address of the index_list.  As len gets smaller, so do n_tmp and
 * v_tmp, but the index_list is always the same length, so we need to pass
 * a multiplier to get the offset...
 *
 * The setup function is like the helper function, but it uses the original
 * input instead of the temp array for the source of the comparisons.
 * AND it doesn't have to worry about doing any merging.
 *
 * After the second pass:
 *	index_list	[ 0 1 3 . 7 . . . ]
 *	n_tmp		[ 3 1 ]
 *	v_tmp		[ 6 8 ]
 *
 * What if we initialize like this before any passes:
 * 	index_list	[ 0 1 2 3 4 5 6 7 ]
 * 	n_tmp		[ 1 1 1 1 1 1 1 1 ]
 * 	v_tmp		[ 6 6 3 6 5 6 7 8 ]
 *
 * Testing with [ 1 2 3 4 5 6 7 8 ]
 *
 * index_list:	[ 1 . 3 . 5 . 7 . ]
 * 		[ 3 . . . 7 . . . ]
 * 		[ 7 . . . . . . . ]
 *
 * max_list:	[ 2 4 6 8 ]
 * 		[ 4 8 ]
 * 		[ 8 ]
 *
 * n_list	[ 1 1 1 1 ]
 * 		[ 1 1 ]
 * 		[ 1 ]
 *
 * What about with 9 elements?
 * input:	[ 1 2 3 4 5 6 7 8 9 ]
 *
 * index_list:	[ 1 . 3 . 5 . 7 . 8 ]
 * 		[ 3 . . . 7 . . . 8 ]
 * 		[ 7 . . . . . . . 8 ]
 * 		[ 8 . . . . . . . . ]
 *
 *
 * max_list:	[ 2 4 6 8 9 ]
 * 		[ 4 8 9 ]
 * 		[ 8 9 ]
 * 		[ 9 ]
 *
 *
 * What about with 7?
 *
 * input:	[ 1 2 3 4 5 6 7 ]
 *
 * max_v:	[ 2 4 6 7 ]
 * 		[ 4 7 ]
 * 		[ 7 ]
 *
 * indices:	[ 1 . 3 . 5 . 6 ]
 * 		[ 3 . . . 6 . . ]
 * 		[ 6 . . . . . . ]
 */

// The g functions (vmaxg etc)  want to return an array of indices, along
// with the value and the occurrence count.
// After the first pass, we will have a full-length array with the pairwise
// max indices, and two half-length temp arrays with the occurrence counts
// and the values.  (Although we don't necessarily need to store the values
// separately, because they can be obtained from the original input array.)
// The subsequent passes need to use temp max values (or use the indices
// to lookup) and then coalesce the indices.

// BUG - where do the gpu tests get used???

// This one takes an oap argument...  we need a vap!?

// These are unusual because the output array (indices) is a linear array,
// regardless of the shape of the input...
// Actually, that not need be true - this could be a form of a projection operator,
// where we could operate on all the rows of an image.  But let's defer that
// complication for the time being.
//
// The input array can be any shape, and shouldn't have to be contiguous...
//
// Here we insist that the index array have the same number of elements as the input.
// We could insist that it have the same shape?  Most of the time the number of occurrences
// will be 1...  Insisting on the same shape would simplify generalizing to projection,
// but might complicate other things...
//
// The index array *should* have the same shape as the input, with the indices accumulating
// in the dimensions(s) that are collapsed in the output extremum target.  (What about when multiple
// dimensions are collapsed?)

// The basic idea is recursive subdivision - each kernel compares two inputs, so the initial
// number of threads is half the number of inputs (rounded up).  We use temp objects for the max's...
//
// example:
//
// input:	10   11   12   13   14   15   16   17
//
// itr. 1	11       13       15       17		indices:  1   3   5   7		count:  1   1   1   1
// itr. 2	13                17         		indices:  3   7			count:  1   1
// itr. 3	17                           		indices:  7			count:  1

// BUG len is declared uint32_t, but later treated as index_type???

// What is stride???




// vmaxv, vsum, vdot, etc
//
// Called "projection" because on host can compute entire sum if dest
// is a scalar, or row sums if dest is a column, etc...
//
// To compute the sum to a scalar, we have to divide and conquer
// as we did with vmaxv...
//
// This looks wrong - the destination does not have to be a scalar!?
//
// We can have a fast version when the source is contiguous and the destination is a scalar,
// we don't need anything except lengths...
//
// For non-scalar outputs (projections) it is more complicated...
// We do the same things, but with more complicated indexing calculations.
//
// i11 i12 i13 i14 i15	->	max(i11,i21)	max(i12,i22)	max(i13,i23)	max(i14,i24)	max(i15,i25)
// i21 i22 i23 i24 i25		max(i31,i41)	max(i32,i42)	max(i33,i43)	max(i34,i44)	max(i35,i45)
// i31 i32 i33 i34 i35
// i41 i42 i43 i44 i45
//
//				d11		d12		d13		d14		d15
//				d21		d22		d23		d24		d25
//
// We could collapse dimensions one by one...

// BUG? gpu_expr? used?



// We can't have vmaxv for complex, no ordering relation!?
// But we do have cvsum!


// This started as the cuda version...
// where do we set dst_values for the first iteration???  ans:  in SETUP_PROJ_ITERATION...
//
// The idea of the original implementation was this:  we can't find the max in parallel, so instead
// we divide and conquer:  we split the input data in two, and then store the pairwise maxima
// to a temporary array.  We repeat on the temporary array, until we have a single value.
//
// That works fine if the final target is a scalar, but what if we are projecting an image
// to a row or column?  Let's analyze the case of projecting an image to a row...  Following the
// analogy, we would divide the image into the top and bottom halves, and then store the pairwise
// maxima to a temporary half-size image, and so on...  We have a problem because the Vector_Args
// struct doesn't contain shape information!?

// We need setup and helper functions in order to support mixed-precision versions (taking sum
// to higher precision).




// vdot, cvdot, etc
// The following is messed up!
// The "setup" function should to the element-wise products to a temp area (which may be large),
// then the "helper" function should sum those.  Best approach would be to use vmul and vsum...







/* NOT Suppressing ! */

/* gpu_host_call_defs.m4 END */




/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/gpu_host_call_defs.m4




/* The fast/slow functions are declared along with the generic typed call,
 * but they really platform-specific, as they call the kernel, so the
 * definitions are elsewhere.
 */











// This is really only necessary for debugging, or if we want to print
// out the arguments, without knowing which are supposed to be set...






// BUG!! these need to be filled in...












/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/host_typed_call_defs.m4




/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../cu2/cu2_host_call_defs.m4


// These length macros used to be called XFER_..._LEN, but that is
// confusing because the other XFER macros are used to transfer
// information from a Vec_Obj_Args struct to a Vector_Args struct.
// Here, we are taking info FROM Vector_Args, to get dimensions
// to be used for GPU gridding...

// SETUP_SLOW_LEN takes the dimension arrays (up to 5 dieensions)
// and figures out which 3 to use, putting the dimensions in len
// and which were chosen in dim_indices.
// There is a problem with complex numbers, as the type dimension
// is 2...
//
// These are only the source dimensions.










































// Not sure if these are correct...







// dest bitmap is like a normal dest vector?


// BUG should use a symbolic constant instead of 4 here?





// Not sure about how to handle source bitmaps?
















/* Suppressing ! */

/* NOT Suppressing ! */


// Declare increments

// Now we set the increments in the vector_args struct?











































// cuda uses dim3...



// cudaGetLastError not available before 5.0 ...












// What is the point of this - where does it occur?


























/* call_fast_kernel */


/* call_fast_kernel defn DONE */
























/* For slow loops, we currently only iterate over two dimensions (x and y),
 * although in principle we should be able to handle 3...
 * We need to determine which 2 by examining the dimensions of the vectors.
 */









// this is 80 columns
//345678901234567890123456789012345678901234567890123456789012345678901234567890





// PORT - insure_gpu_device ???

// We should have an intelligent way of designing blocks...
// On wheatstone, the card can have a maximum of 512 threads/block.
// But this doesn't divide typical video image dimensions
// of 640x480.  But 640 is 128x5, so we could have blocks
// that are 128x4 and not have any leftover.  Using our dumb
// strategy, we have blocks that are 16x32... 480/32=15,
// so we are OK here.
//
// We found a bug doing interlaces of 640/480, the 240 line half-frames
// came out to 7.5 32 row blocks in height.  In that case it's better
// to have the blocks wider and less tall.
//
// It might be nice to adjust things like this automatically, but
// it's kind of tricky!?

// This is kind of kludgy, but let's try it










/* If we have a destination bitmap, we handle all the bits in one word
 * in a single thread.
 *
 * BUG - here we ignore bit0 ???
 *
 * 32 is 32...  for a 512 pixel wide image, the nmber
 * of bitmap words is either 8 (64 bit words) or 16 (32 bit words).
 * So we need to 
 */




















 



 












/* get_threads_per_block defn */


/* get_threads_per_block defn DONE */


// This used to be called GET_THREADS_PER_BITMAP_BLOCK





















 






 




// MM_IND vmaxi etc

// CUDA definitions
// BUG we probably want the passed vap to have constant data...

// BUG use symbolic constant for kernel args!



// BUG use symbolic constant for kernel args!




// CUDA only!






// CUDA only!













/* NOT Suppressing ! */

// END INCLUDED FILE ../cu2/cu2_host_call_defs.m4








/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/sp_defs.m4
/* sp_defs.m4 BEGIN */























/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/sp_func_defs.m4

 
/* Suppressing ! */

/* NOT Suppressing ! */





/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/sp_func_defs.m4


/* TYPE_CODE = sp   dest_type =float */
/* sp_defs.m4 DONE */



/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/sp_defs.m4



void sp_cent(LINK_FUNC_ARG_DECLS)
{
	
	cudaError_t e;


	dim3 n_blocks, n_threads_per_block;
	dim3 extra;


	/*

	dim3 n_blocks, n_threads_per_block;
	dim3 extra;
*/
	dim3 len;
	/*int max_threads_per_block;*/
	dim5 dst_vwxyz_incr; dim5 s1_vwxyz_incr; dim5 s2_vwxyz_incr;

	/*max_threads_per_block =	curr_cdp->cudev_prop.maxThreadsPerBlock;*/
	

	e = cudaGetLastError();
	if( e != cudaSuccess ){
		describe_cuda_driver_error("sp_slow_cent_helper",e);
	}

	/*XFER_SLOW_LEN_3*/
	/*

	

	if( (VA_ITERATION_TOTAL(vap)) < 32 ) {
		n_threads_per_block.x = VA_ITERATION_TOTAL(vap);
		n_blocks.x = 1;
		extra.x = 0;
	} else {
		n_blocks.x = (VA_ITERATION_TOTAL(vap)) / 32;
		n_threads_per_block.x = 32;
		extra.x = (VA_ITERATION_TOTAL(vap)) % 32;
	}

	

	assert(n_threads_per_block.x>0);
	n_threads_per_block.y = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(VA_PFDEV(vap)) 
				/ n_threads_per_block.x;
	if( VA_LEN_Y(vap) < n_threads_per_block.y ){
		n_threads_per_block.y = VA_LEN_Y(vap);
		n_blocks.y = 1;
		extra.y = 0;
	} else {
		assert(n_threads_per_block.y>0);
		n_blocks.y = VA_LEN_Y(vap) / n_threads_per_block.y;
		extra.y = VA_LEN_Y(vap) % n_threads_per_block.y;
	}
	if( extra.x > 0 ) n_blocks.x++;
	if( extra.y > 0 ) n_blocks.y++;

	

	assert(n_threads_per_block.x>0);
	assert(n_threads_per_block.y>0);
	n_threads_per_block.z = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(VA_PFDEV(vap)) 
		/ (n_threads_per_block.x*n_threads_per_block.y);
	if( VA_LEN_Z(vap) < n_threads_per_block.z ){
		n_threads_per_block.z = VA_LEN_Z(vap);
		n_blocks.z = 1;
		extra.z = 0;
	} else {
		assert(n_threads_per_block.z>0);
		n_blocks.z = VA_LEN_Z(vap) / n_threads_per_block.z;
		extra.z = VA_LEN_Z(vap) % n_threads_per_block.z;
	}
	if( extra.z > 0 ) n_blocks.z++;

*/
	

	dst_vwxyz_incr.d5_dim[0] = INCREMENT(VA_DEST_INCSET(vap),0);
	dst_vwxyz_incr.d5_dim[1] = INCREMENT(VA_DEST_INCSET(vap),1);
	dst_vwxyz_incr.d5_dim[2] = INCREMENT(VA_DEST_INCSET(vap),2);
	dst_vwxyz_incr.d5_dim[3] = INCREMENT(VA_DEST_INCSET(vap),3);
	dst_vwxyz_incr.d5_dim[4] = INCREMENT(VA_DEST_INCSET(vap),4);
/*REPORT_INCS(dst_vwxyz_incr)*/
 

	s1_vwxyz_incr.d5_dim[0] = INCREMENT(VA_SRC1_INCSET(vap),0);
	s1_vwxyz_incr.d5_dim[1] = INCREMENT(VA_SRC1_INCSET(vap),1);
	s1_vwxyz_incr.d5_dim[2] = INCREMENT(VA_SRC1_INCSET(vap),2);
	s1_vwxyz_incr.d5_dim[3] = INCREMENT(VA_SRC1_INCSET(vap),3);
	s1_vwxyz_incr.d5_dim[4] = INCREMENT(VA_SRC1_INCSET(vap),4);
 

	s2_vwxyz_incr.d5_dim[0] = INCREMENT(VA_SRC2_INCSET(vap),0);
	s2_vwxyz_incr.d5_dim[1] = INCREMENT(VA_SRC2_INCSET(vap),1);
	s2_vwxyz_incr.d5_dim[2] = INCREMENT(VA_SRC2_INCSET(vap),2);
	s2_vwxyz_incr.d5_dim[3] = INCREMENT(VA_SRC2_INCSET(vap),3);
	s2_vwxyz_incr.d5_dim[4] = INCREMENT(VA_SRC2_INCSET(vap),4);

	
/*REPORT_ARGS_3*/
	sp_slow_cent_helper<<< n_blocks, n_threads_per_block >>>
		(VA_SLOW_SIZE(vap) , (float *) VA_DEST_PTR(vap)  , dst_vwxyz_incr , (float *) VA_SRC1_PTR(vap)  , s1_vwxyz_incr , (float *) VA_SRC2_PTR(vap)  , s2_vwxyz_incr  );
    	CUDA_ERROR_CHECK("kernel launch failure");
}

/* Now the entry point */

void sp_cuda_centroid(HOST_CALL_ARG_DECLS)
{
	Vector_Args va1;
	Vector_Args *vap=(&va1);
	/*Spacing_Info spi1;*/
	/*Size_Info szi1;*/

	/*SET_VA_SPACING(vap,&spi1);*/
	/*SET_VA_SIZE_INFO(vap,&szi1);*/
	insure_cuda_device( oap->oa_dest );
	

	SET_VA_DEST_PTR(vap, OBJ_DATA_PTR(oap->oa_dest) );
	SET_VA_DEST_OFFSET(vap,OBJ_OFFSET(oap->oa_dest));

	SET_VA_DEST_INCSET(vap, OBJ_TYPE_INCS(oap->oa_dest) );
	SET_VA_COUNT(vap,OBJ_TYPE_DIMS(oap->oa_dest) );
 
	SET_VA_SRC_PTR(vap,0, OBJ_DATA_PTR(OA_SRC_OBJ(oap,0)) );
	SET_VA_SRC_OFFSET(vap,0,OBJ_OFFSET(OA_SRC_OBJ(oap,0)) );

	
	SET_VA_SRC_INCSET(vap,0,OBJ_TYPE_INCS(OA_SRC_OBJ(oap,0)) );
	SET_VA_SRC_DIMSET(vap,0,OBJ_TYPE_DIMS(OA_SRC_OBJ(oap,0)) );
 
	SET_VA_SRC_PTR(vap,1, OBJ_DATA_PTR(OA_SRC_OBJ(oap,1)) );
	SET_VA_SRC_OFFSET(vap,1,OBJ_OFFSET(OA_SRC_OBJ(oap,1)) );

	
	SET_VA_SRC_INCSET(vap,1,OBJ_TYPE_INCS(OA_SRC_OBJ(oap,1)) );
	SET_VA_SRC_DIMSET(vap,1,OBJ_TYPE_DIMS(OA_SRC_OBJ(oap,1)) );

	if( setup_slow_len(vap,0,0,3,VA_PFDEV(vap)) < 0 ) return;
	

	if( is_chaining ){
		if( insure_static(oap) < 0 ) return;
		add_link( & sp_cent , LINK_FUNC_ARGS );
		return;
	} else {
		sp_cent (LINK_FUNC_ARGS);
		SET_ASSIGNED_FLAG( OA_DEST(oap) )
	}

	if( is_chaining ){
		if( insure_static(oap) < 0 ) return;
		add_link( & sp_cent , LINK_FUNC_ARGS );
		return;
	} else {
		sp_cent(LINK_FUNC_ARGS);
		SET_OBJ_FLAG_BITS(OA_DEST(oap), DT_ASSIGNED);
		/* WHY set assigned flag on a source obj??? */
		/* Maybe because its really a second destination? */
		SET_OBJ_FLAG_BITS(OA_SRC1(oap), DT_ASSIGNED);
	}
}





/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/dp_defs.m4
























/* NOT Suppressing ! */

// BEGIN INCLUDED FILE ../../include/veclib/dp_func_defs.m4


























/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/dp_func_defs.m4





/* NOT Suppressing ! */

// END INCLUDED FILE ../../include/veclib/dp_defs.m4



void dp_cent(LINK_FUNC_ARG_DECLS)
{
	
	cudaError_t e;


	dim3 n_blocks, n_threads_per_block;
	dim3 extra;


	/*

	dim3 n_blocks, n_threads_per_block;
	dim3 extra;
*/
	dim3 len;
	/*int max_threads_per_block;*/
	dim5 dst_vwxyz_incr; dim5 s1_vwxyz_incr; dim5 s2_vwxyz_incr;

	/*max_threads_per_block =	curr_cdp->cudev_prop.maxThreadsPerBlock;*/
	

	e = cudaGetLastError();
	if( e != cudaSuccess ){
		describe_cuda_driver_error("dp_slow_cent_helper",e);
	}

	/*XFER_SLOW_LEN_3*/
	/*

	

	if( (VA_ITERATION_TOTAL(vap)) < 32 ) {
		n_threads_per_block.x = VA_ITERATION_TOTAL(vap);
		n_blocks.x = 1;
		extra.x = 0;
	} else {
		n_blocks.x = (VA_ITERATION_TOTAL(vap)) / 32;
		n_threads_per_block.x = 32;
		extra.x = (VA_ITERATION_TOTAL(vap)) % 32;
	}

	

	assert(n_threads_per_block.x>0);
	n_threads_per_block.y = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(VA_PFDEV(vap)) 
				/ n_threads_per_block.x;
	if( VA_LEN_Y(vap) < n_threads_per_block.y ){
		n_threads_per_block.y = VA_LEN_Y(vap);
		n_blocks.y = 1;
		extra.y = 0;
	} else {
		assert(n_threads_per_block.y>0);
		n_blocks.y = VA_LEN_Y(vap) / n_threads_per_block.y;
		extra.y = VA_LEN_Y(vap) % n_threads_per_block.y;
	}
	if( extra.x > 0 ) n_blocks.x++;
	if( extra.y > 0 ) n_blocks.y++;

	

	assert(n_threads_per_block.x>0);
	assert(n_threads_per_block.y>0);
	n_threads_per_block.z = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(VA_PFDEV(vap)) 
		/ (n_threads_per_block.x*n_threads_per_block.y);
	if( VA_LEN_Z(vap) < n_threads_per_block.z ){
		n_threads_per_block.z = VA_LEN_Z(vap);
		n_blocks.z = 1;
		extra.z = 0;
	} else {
		assert(n_threads_per_block.z>0);
		n_blocks.z = VA_LEN_Z(vap) / n_threads_per_block.z;
		extra.z = VA_LEN_Z(vap) % n_threads_per_block.z;
	}
	if( extra.z > 0 ) n_blocks.z++;

*/
	

	dst_vwxyz_incr.d5_dim[0] = INCREMENT(VA_DEST_INCSET(vap),0);
	dst_vwxyz_incr.d5_dim[1] = INCREMENT(VA_DEST_INCSET(vap),1);
	dst_vwxyz_incr.d5_dim[2] = INCREMENT(VA_DEST_INCSET(vap),2);
	dst_vwxyz_incr.d5_dim[3] = INCREMENT(VA_DEST_INCSET(vap),3);
	dst_vwxyz_incr.d5_dim[4] = INCREMENT(VA_DEST_INCSET(vap),4);
/*REPORT_INCS(dst_vwxyz_incr)*/
 

	s1_vwxyz_incr.d5_dim[0] = INCREMENT(VA_SRC1_INCSET(vap),0);
	s1_vwxyz_incr.d5_dim[1] = INCREMENT(VA_SRC1_INCSET(vap),1);
	s1_vwxyz_incr.d5_dim[2] = INCREMENT(VA_SRC1_INCSET(vap),2);
	s1_vwxyz_incr.d5_dim[3] = INCREMENT(VA_SRC1_INCSET(vap),3);
	s1_vwxyz_incr.d5_dim[4] = INCREMENT(VA_SRC1_INCSET(vap),4);
 

	s2_vwxyz_incr.d5_dim[0] = INCREMENT(VA_SRC2_INCSET(vap),0);
	s2_vwxyz_incr.d5_dim[1] = INCREMENT(VA_SRC2_INCSET(vap),1);
	s2_vwxyz_incr.d5_dim[2] = INCREMENT(VA_SRC2_INCSET(vap),2);
	s2_vwxyz_incr.d5_dim[3] = INCREMENT(VA_SRC2_INCSET(vap),3);
	s2_vwxyz_incr.d5_dim[4] = INCREMENT(VA_SRC2_INCSET(vap),4);

	
/*REPORT_ARGS_3*/
	dp_slow_cent_helper<<< n_blocks, n_threads_per_block >>>
		(VA_SLOW_SIZE(vap) , (double *) VA_DEST_PTR(vap)  , dst_vwxyz_incr , (double *) VA_SRC1_PTR(vap)  , s1_vwxyz_incr , (double *) VA_SRC2_PTR(vap)  , s2_vwxyz_incr  );
    	CUDA_ERROR_CHECK("kernel launch failure");
}

/* Now the entry point */

void dp_cuda_centroid(HOST_CALL_ARG_DECLS)
{
	Vector_Args va1;
	Vector_Args *vap=(&va1);
	/*Spacing_Info spi1;*/
	/*Size_Info szi1;*/

	/*SET_VA_SPACING(vap,&spi1);*/
	/*SET_VA_SIZE_INFO(vap,&szi1);*/
	insure_cuda_device( oap->oa_dest );
	

	SET_VA_DEST_PTR(vap, OBJ_DATA_PTR(oap->oa_dest) );
	SET_VA_DEST_OFFSET(vap,OBJ_OFFSET(oap->oa_dest));

	SET_VA_DEST_INCSET(vap, OBJ_TYPE_INCS(oap->oa_dest) );
	SET_VA_COUNT(vap,OBJ_TYPE_DIMS(oap->oa_dest) );
 
	SET_VA_SRC_PTR(vap,0, OBJ_DATA_PTR(OA_SRC_OBJ(oap,0)) );
	SET_VA_SRC_OFFSET(vap,0,OBJ_OFFSET(OA_SRC_OBJ(oap,0)) );

	
	SET_VA_SRC_INCSET(vap,0,OBJ_TYPE_INCS(OA_SRC_OBJ(oap,0)) );
	SET_VA_SRC_DIMSET(vap,0,OBJ_TYPE_DIMS(OA_SRC_OBJ(oap,0)) );
 
	SET_VA_SRC_PTR(vap,1, OBJ_DATA_PTR(OA_SRC_OBJ(oap,1)) );
	SET_VA_SRC_OFFSET(vap,1,OBJ_OFFSET(OA_SRC_OBJ(oap,1)) );

	
	SET_VA_SRC_INCSET(vap,1,OBJ_TYPE_INCS(OA_SRC_OBJ(oap,1)) );
	SET_VA_SRC_DIMSET(vap,1,OBJ_TYPE_DIMS(OA_SRC_OBJ(oap,1)) );

	if( setup_slow_len(vap,0,0,3,VA_PFDEV(vap)) < 0 ) return;
	

	if( is_chaining ){
		if( insure_static(oap) < 0 ) return;
		add_link( & dp_cent , LINK_FUNC_ARGS );
		return;
	} else {
		dp_cent (LINK_FUNC_ARGS);
		SET_ASSIGNED_FLAG( OA_DEST(oap) )
	}

	if( is_chaining ){
		if( insure_static(oap) < 0 ) return;
		add_link( & dp_cent , LINK_FUNC_ARGS );
		return;
	} else {
		dp_cent(LINK_FUNC_ARGS);
		SET_OBJ_FLAG_BITS(OA_DEST(oap), DT_ASSIGNED);
		/* WHY set assigned flag on a source obj??? */
		/* Maybe because its really a second destination? */
		SET_OBJ_FLAG_BITS(OA_SRC1(oap), DT_ASSIGNED);
	}
}



