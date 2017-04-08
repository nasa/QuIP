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

















static __shared__ unsigned int  ng_yuv_gray[256];
static __shared__ unsigned int  ng_yuv_red[256];
static __shared__ unsigned int  ng_yuv_blue[256];
static __shared__ unsigned int  ng_yuv_g1[256];
static __shared__ unsigned int  ng_yuv_g2[256];
static __shared__ unsigned int  ng_clip[256 + 2 * 320];







static int tbls_inited=0;


static void init_tables(void);


__global__ void decode_two_pixels_yuv2rgb(unsigned char *dst_p,unsigned char *yuyv_p)
{
	int index,gray;
	unsigned char *d_p, *y_p, *v_p, *u_p;

	index = blockIdx.x * blockDim.x + threadIdx.x;

	d_p = dst_p  + 6*index;		// two RGB pixels
	y_p = yuyv_p + 4*index;
	u_p = y_p + 1;
	v_p = y_p + 3;

	gray   = ng_yuv_gray[*y_p];
	d_p[0] = ng_clip[ 320 + gray + ng_yuv_blue[*u_p] ];
	d_p[1] = ng_clip[ 320 + gray + ng_yuv_g1[*v_p] + ng_yuv_g2[*v_p] ];
	d_p[2] = ng_clip[ 320 + gray + ng_yuv_red[*v_p] ];

	y_p += 2;

	gray   = ng_yuv_gray[*y_p];
	d_p[3] = ng_clip[ 320 + gray + ng_yuv_blue[*u_p] ];
	d_p[4] = ng_clip[ 320 + gray + ng_yuv_g1[*v_p] + ng_yuv_g2[*v_p] ];
	d_p[5] = ng_clip[ 320 + gray + ng_yuv_red[*v_p] ];
}





/* This function assumes that src_dp points to an image w/ YUYV samples... */

void cuda_yuv422_to_rgb24(Data_Obj *dst_dp, Data_Obj * src_dp )
{
	unsigned char *y_p, *dst_p;
	//int max_threads_per_block, n_blocks, n_thr_need, n_extra;
	Vector_Args va1, *vap=(&va1);

	

	dim3 n_blocks, n_threads_per_block;
	dim3 extra;

	if( !tbls_inited ) init_tables();
	

	

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
	n_threads_per_block.y = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(OBJ_PFDEV(dst_dp)) 
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
	n_threads_per_block.z = PFDEV_CUDA_MAX_THREADS_PER_BLOCK(OBJ_PFDEV(dst_dp)) 
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



	dst_p = (unsigned char *)OBJ_DATA_PTR(dst_dp);
	y_p  = (unsigned char *)OBJ_DATA_PTR(src_dp);

	decode_two_pixels_yuv2rgb<<< /*n_blocks , max_threads_per_block*/ n_blocks, n_threads_per_block >>>
		(dst_p,y_p);
}

/* ------------------------------------------------------------------- */

__global__ void init_tbl_entries(void)
{
	int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

        ng_yuv_gray[i] = i * 256 >> 8;
        ng_yuv_red[i]  = ((-128  * 512)    + i * 512)    >> 8;
        ng_yuv_blue[i] = ((-128 * 512)   + i * 512)   >> 8;
        ng_yuv_g1[i]   = ((-(-128  * 512)/2) + i * (-512/2)) >> 8;
        ng_yuv_g2[i]   = ((-(-128 * 512)/6) + i * (-512/6)) >> 8;
        ng_clip[i+320] = i ;
}

__global__ void const_tbl_entries(unsigned int index, unsigned int value)
{
	int i;

	i = blockIdx.x * blockDim.x + threadIdx.x;

        ng_clip[index+i] = value;
}

static void init_tables(void)
{
	// BUG make sure we can have this many threads in a block.

	init_tbl_entries<<< 1 , 256  >>>();
	// can't take the address of a shared variable, cuda 5 warning???
	const_tbl_entries<<< 1 , 320  >>>( 0, 0 );
	const_tbl_entries<<< 1 , 320  >>>( 320+256, 255 );
	tbls_inited=1;
}


