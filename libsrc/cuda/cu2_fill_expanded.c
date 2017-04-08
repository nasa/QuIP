
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


//#include(`my_vector_functions.h"	// max_threads_per_block




// The fill routine kernel




__global__ void zeroit(dim5 szarr, unsigned char* a, dim5 len )
{
	//int x,y;
	dim5 index1;

	//x = blockIdx.x * blockDim.x + threadIdx.x;
	//y = blockIdx.y * blockDim.y + threadIdx.y;
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


	/*
	if( x < len.x && y < len.y ){
		a[x+y*len.y] = 0;
	}
	*/
	a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] = 0;
}

__global__ void g_sp_ifl_incs(dim5 szarr, float* a, dim5 inc1,
			unsigned char* b, dim5 inc2,
			dim5 len,
			float v, float tol, float fill_val, int *flag)
{
	/* decl_indices_2 */ dim5 index1; dim5 index2;

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
 index2 = index1;

		if( b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] == 0 ){	// not filled yet
			if( index2.d5_dim[1] > 0 ){	// in-bounds
				index2.d5_dim[1] -= inc2.d5_dim[1];
				if( b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] > 0 ){	// neighbor filled?
					index2.d5_dim[1] += inc2.d5_dim[1];
					
	if( fabs( a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] - v ) < tol ){
		b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] = 1;
		a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] = fill_val;
		*flag = 1;
		return;
	}

				}
				index2.d5_dim[1] += inc2.d5_dim[1];
			}
			if( index2.d5_dim[1] < (len.d5_dim[1]-1)*inc2.d5_dim[1] ){
				index2.d5_dim[1] += inc2.d5_dim[1];
				if( b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] > 0 ){
					index2.d5_dim[1] -= inc2.d5_dim[1];
					
	if( fabs( a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] - v ) < tol ){
		b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] = 1;
		a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] = fill_val;
		*flag = 1;
		return;
	}

				}
				index2.d5_dim[1] -= inc2.d5_dim[1];
			}
			if( index2.d5_dim[2] > 0 ){
				index2.d5_dim[2] -= inc2.d5_dim[2];
				if( b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] > 0 ){
					index2.d5_dim[2] += inc2.d5_dim[2];
					
	if( fabs( a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] - v ) < tol ){
		b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] = 1;
		a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] = fill_val;
		*flag = 1;
		return;
	}

				}
				index2.d5_dim[2] += inc2.d5_dim[2];
			}
			if( index2.d5_dim[2] < (len.d5_dim[2]-1)*inc2.d5_dim[2] ){
				index2.d5_dim[2] += inc2.d5_dim[2];
				if( b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] > 0 ){
					index2.d5_dim[2] -= inc2.d5_dim[2];
					
	if( fabs( a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] - v ) < tol ){
		b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] = 1;
		a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] = fill_val;
		*flag = 1;
		return;
	}

				}
				index2.d5_dim[2] -= inc2.d5_dim[2];
			}
		}
}

__constant__ float test_value[1];
__constant__ float tolerance[1];
__constant__ float fill_value[1];



__global__ void g_sp_ifl2_incs(dim5 szarr, float* a, dim5 inc1,
			unsigned char* b, dim5 inc2,
			dim5 len)
{
	/* decl_indices_2 */ dim5 index1; dim5 index2;

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
 index2 = index1;

	if (index1.d5_dim[1] < len.d5_dim[1] && index1.d5_dim[2] < len.d5_dim[2] ) {
		if( b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] == 0 ){	// not filled yet
			// check each neighbor if filled
			if( index2.d5_dim[1] > 0 ){	// in-bounds
				index2.d5_dim[1] -= inc2.d5_dim[1];
				if( b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] > 0 ){	// neighbor filled?
					index2.d5_dim[1] += inc2.d5_dim[1];
					
	if( fabs( a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] - test_value[0] ) < tolerance[0] ){
		b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] = 1;
		a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] = fill_value[0];
		return;
	}

				}
				index2.d5_dim[1] += inc2.d5_dim[1];
			}
			if( index2.d5_dim[1] < (len.d5_dim[1]-1)*inc2.d5_dim[1] ){
				index2.d5_dim[1] += inc2.d5_dim[1];
				if( b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] > 0 ){
					index2.d5_dim[1] -= inc2.d5_dim[1];
					
	if( fabs( a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] - test_value[0] ) < tolerance[0] ){
		b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] = 1;
		a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] = fill_value[0];
		return;
	}

				}
				index2.d5_dim[1] -= inc2.d5_dim[1];
			}
			if( index2.d5_dim[2] > 0 ){
				index2.d5_dim[2] -= inc2.d5_dim[2];
				if( b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] > 0 ){
					index2.d5_dim[2] += inc2.d5_dim[2];
					
	if( fabs( a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] - test_value[0] ) < tolerance[0] ){
		b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] = 1;
		a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] = fill_value[0];
		return;
	}

				}
				index2.d5_dim[2] += inc2.d5_dim[2];
			}
			if( index2.d5_dim[2] < (len.d5_dim[2]-1)*inc2.d5_dim[2] ){
				index2.d5_dim[2] += inc2.d5_dim[2];
				if( b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] > 0 ){
					index2.d5_dim[2] -= inc2.d5_dim[2];
					
	if( fabs( a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] - test_value[0] ) < tolerance[0] ){
		b[index2.d5_dim[0]+index2.d5_dim[1]+index2.d5_dim[2]+index2.d5_dim[3]+index2.d5_dim[4]  ] = 1;
		a[index1.d5_dim[0]+index1.d5_dim[1]+index1.d5_dim[2]+index1.d5_dim[3]+index1.d5_dim[4]  ] = fill_value[0];
		return;
	}

				}
				index2.d5_dim[2] -= inc2.d5_dim[2];
			}
		}
	}
}

void h_sp_ifl( Data_Obj *dp, int x, int y, float tol, float fill_val )
{
	

	dim3 n_blocks, n_threads_per_block;
	dim3 extra;

	cudaError_t e;
	dim5 len, inc1, inc2;
	unsigned char *filled, b_one;
	float *f_p, v;
	int h_flag, *flag_p;
	int n_iterations;
	Vector_Args va1, *vap=(&va1);
	dim5 szarr;

	len.d5_dim[1] = OBJ_COLS(dp);
	len.d5_dim[2] = OBJ_ROWS(dp);

	

	

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
	n_threads_per_block.y = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(OBJ_PFDEV(dp)) 
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
	n_threads_per_block.z = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(OBJ_PFDEV(dp)) 
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



	inc1.d5_dim[1] = OBJ_TYPE_INC(dp,1);
	inc1.d5_dim[2] = OBJ_TYPE_INC(dp,2);
	inc1.d5_dim[0] = inc1.d5_dim[3] = inc1.d5_dim[4] = 0;
	inc2 = inc1;

	if( cudaMalloc(&flag_p,sizeof(*flag_p)) != cudaSuccess ){
		NERROR1("cuda malloc error getting flag word");
	}

	/* use 2d allocator for better stride? */
	if( cudaMalloc(&filled,len.d5_dim[1]*len.d5_dim[2]) != cudaSuccess ){
		NERROR1("cuda malloc error getting filled array");
	}

	/* set filled to zero */
	

	e = cudaGetLastError();
	if( e != cudaSuccess ){
		describe_cuda_driver_error("zeroit",e);
	}

	zeroit<<< n_blocks, n_threads_per_block >>>(szarr,filled,len);
	

	e = cudaGetLastError();
	if( e != cudaSuccess ){
		describe_cuda_driver_error("h_sp_ifl: zeroit",e);
	}


	// Get the value at the seed point
	f_p = (float *)OBJ_DATA_PTR(dp);
	f_p += x + y * inc1.d5_dim[2];

	e = cudaMemcpy(&v, f_p, sizeof(v), cudaMemcpyDeviceToHost);
	
	if( e != cudaSuccess ){
		NWARN("cudaMemcpy device to host");
	}
;

	// Fill the seed point
	b_one = 1;
	e = cudaMemcpy(filled+x+y*len.d5_dim[1], &b_one, sizeof(b_one),
						cudaMemcpyHostToDevice);
	
	if( e != cudaSuccess ){
		NWARN("cudaMemcpy host to device");
	}
;

	e = cudaMemcpy(f_p, &fill_val, sizeof(fill_val),
						cudaMemcpyHostToDevice);
	
	if( e != cudaSuccess ){
		NWARN("cudaMemcpy host to device");
	}
;


	n_iterations=0;
	do {
		h_flag = 0;
		e = cudaMemcpy(flag_p, &h_flag, sizeof(h_flag),
						cudaMemcpyHostToDevice);
		
	if( e != cudaSuccess ){
		NWARN("cudaMemcpy host to device");
	}
;

		

	e = cudaGetLastError();
	if( e != cudaSuccess ){
		describe_cuda_driver_error("g_sp_ifl_incs",e);
	}

		g_sp_ifl_incs<<< n_blocks, n_threads_per_block >>>
		(szarr,(float *)OBJ_DATA_PTR(dp),inc1,filled,inc2,len,v,tol,fill_val,flag_p);
		

	e = cudaGetLastError();
	if( e != cudaSuccess ){
		describe_cuda_driver_error("h_sp_ifl: g_sp_ifl_incs",e);
	}


		// download flag to see what happened.
		e = cudaMemcpy(&h_flag, flag_p, 1,
						cudaMemcpyDeviceToHost);
		
	if( e != cudaSuccess ){
		NWARN("cudaMemcpy device to host");
	}
;
		n_iterations++;
	} while( h_flag );

	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,"Fill completed after %d iterations",n_iterations);
		NADVISE(DEFAULT_ERROR_STRING);
	}
}

void h_sp_ifl2( Data_Obj *dp, int seed_x, int seed_y, float tol, float fill_val )
{
	

	dim3 n_blocks, n_threads_per_block;
	dim3 extra;

	cudaError_t e;
	dim5 len, inc1, inc2;
	unsigned char *filled, b_one;
	float *f_p, v;
	int n_iterations;
	Vector_Args va1, *vap=(&va1);
	dim5 szarr;

	len.d5_dim[1] = OBJ_COLS(dp);
	len.d5_dim[2] = OBJ_ROWS(dp);

	

	

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
	n_threads_per_block.y = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(OBJ_PFDEV(dp)) 
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
	n_threads_per_block.z = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(OBJ_PFDEV(dp)) 
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



	inc1.d5_dim[1] = OBJ_TYPE_INC(dp,1);
	inc1.d5_dim[2] = OBJ_TYPE_INC(dp,2);
	inc1.d5_dim[0] = inc1.d5_dim[3] = inc1.d5_dim[4] = 0;
	inc2 = inc1;

	if( cudaMalloc(&filled,len.d5_dim[1]*len.d5_dim[2]) != cudaSuccess ){
		NERROR1("cuda malloc error getting filled array");
	}

	

	e = cudaGetLastError();
	if( e != cudaSuccess ){
		describe_cuda_driver_error("zeroit",e);
	}

	zeroit<<< n_blocks, n_threads_per_block >>>(szarr,filled,len);
	

	e = cudaGetLastError();
	if( e != cudaSuccess ){
		describe_cuda_driver_error("h_sp_ifl2: zeroit",e);
	}


	f_p = (float *)OBJ_DATA_PTR(dp);
	f_p += seed_x + seed_y * inc1.d5_dim[2];

	e = cudaMemcpy(&v, f_p, sizeof(v), cudaMemcpyDeviceToHost);
	
	if( e != cudaSuccess ){
		NWARN("cudaMemcpy device to host");
	}
;

	b_one = 1;
	e = cudaMemcpy(filled+seed_x+seed_y*len.d5_dim[1], &b_one, sizeof(b_one),
						cudaMemcpyHostToDevice);
	
	if( e != cudaSuccess ){
		NWARN("cudaMemcpy host to device");
	}
;
	e = cudaMemcpy(f_p, &fill_val, sizeof(fill_val),
						cudaMemcpyHostToDevice);
	
	if( e != cudaSuccess ){
		NWARN("cudaMemcpy host to device");
	}
;

	e = cudaMemcpyToSymbol(fill_value, &fill_val, sizeof(float));
	
	if( e != cudaSuccess ){
		NWARN("cudaMemcpyToSymbol");
	}
;
	e = cudaMemcpyToSymbol(tolerance, &tol, sizeof(float));
	
	if( e != cudaSuccess ){
		NWARN("cudaMemcpyToSymbol");
	}
;
	e = cudaMemcpyToSymbol(test_value, &v, sizeof(float));
	
	if( e != cudaSuccess ){
		NWARN("cudaMemcpyToSymbol");
	}
;

	n_iterations=0;
	for( n_iterations = 0 ; n_iterations < 300 ; n_iterations++ ){

		

	e = cudaGetLastError();
	if( e != cudaSuccess ){
		describe_cuda_driver_error("g_sp_ifl2_incs",e);
	}

		g_sp_ifl2_incs<<< n_blocks, n_threads_per_block >>>
		(szarr,(float *)OBJ_DATA_PTR(dp),inc1,filled,inc2,len);
		

	e = cudaGetLastError();
	if( e != cudaSuccess ){
		describe_cuda_driver_error("h_sp_ifl2:  g_sp_ifl2_incs",e);
	}


	}


	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,"Fill completed after %d iterations",n_iterations);
		NADVISE(DEFAULT_ERROR_STRING);
	}
}

