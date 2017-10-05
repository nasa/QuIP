dnl	/* Flood fill
dnl	 *
dnl	 * We perform a breadth-first search on all of the pixels.
dnl	 * We do this with the aid of a depth array (initialized to -1,
dnl	 * everywhere except the seed point, which we initialize with 0).
dnl	 * The kernel has in iteration counter which we also start at 0,
dnl	 * each iteration we only operate on pixels whose depth matches
dnl	 * the counter value.  When a pixel's turn comes, we first perform
dnl	 * the fill test.  If it passes, we update the pixel value, note that
dnl	 * something changed, and then proceed to propagate the depth
dnl	 * to the four neighbors.  We only update the depth for neighbors
dnl	 * with depth -1 (UNEXAMINED).	This is repeated until there is no
dnl	 * change.
dnl	 *
dnl	 */

include(`../../include/veclib/cu2_port.m4')

my_include(`../../include/veclib/gpu_call_utils.m4')
my_include(`../../include/veclib/slow_defs.m4')
my_include(`../cu2/cu2_host_call_defs.m4')
my_include(`../cu2/cu2_kern_call_defs.m4')

//#include(`my_vector_functions.h"	// max_threads_per_block

dnl	 CHECK_CUDA_RETURN_VAL(msg)

define(`CHECK_CUDA_RETURN_VAL',`
	if( e != cudaSuccess ){
		NWARN($1);
	}
')


define(`UNEXAMINED',`-1')
define(`EXAMINED',`-2')		dnl EXAMINED but not filled

define(`FILL_IF',`
	if( absfunc( src1 - v ) < tol ){
		src1 = fill_val;
		flooding = 1;
	} else {
		dst = EXAMINED;
	}
')

dnl	FLOOD_IF(neighbor)

define(`FLOOD_IF',`
	if( $1 == UNEXAMINED ){
		$1 = depth + 1;
	}
')

define(`LEFT_NEIGHBOR', `a[ index1 - 1       OFFSET_A ]')
define(`RIGHT_NEIGHBOR',`a[ index1 + 1       OFFSET_A ]')
define(`UPPER_NEIGHBOR',`a[ index1 - row_len OFFSET_A ]')
define(`LOWER_NEIGHBOR',`a[ index1 + row_len OFFSET_A ]')


dnl better to pass the size array...

KERNEL_FUNC_QUALIFIER void new_fill(short *a, std_type *b,
			std_type v, std_type tol, std_type fill_val, int seed_x, int seed_y )
{
	DECL_INDICES_2
	int depth=0;
	__shared__ flooding;

	SET_INDICES_2

	dst = UNEXAMINED;
	if( index1.x == seed_x && index1.y == seed_y )
		dst = 0;

	do {
		// Better to do this once or in every thread?
		if( index1.x == seed_x && index1.y == seed_y )
			flooding = 0;
		__syncthreads();

		if( dst == depth ){
			FILL_IF
			// Only check neighbors if filled
			if( dst >= 0 ){
				if( index1.x > 0 ){	// check left
					FLOOD_IF(LEFT_NEIGHBOR)
				}
				if( index1.x < (szarr.x-1) ){	// check right
					FLOOD_IF(RIGHT_NEIGHBOR)
				}
				if( index1.y > 0 ){ 
					FLOOD_IF(UPPER_NEIGHBOR)
				}
				if( index1.y < (szarr.y-1) ){	// check right
					FLOOD_IF(LOWER_NEIGHBOR)
				}
			}
		}
		depth++;
		__syncthreads();
	} while(flooding);
}

__constant__ float test_value[1];
__constant__ float tolerance[1];
__constant__ float fill_value[1];

void h_sp_ifl( Data_Obj *dp, int x, int y, float tol, float fill_val )
{
	BLOCK_VARS_DECLS
	cudaError_t e;
	dim5 len, inc1, inc2;
	short *depth;
	float *f_p, v;
	int h_flag, *flag_p;
	int n_iterations;
	Vector_Args va1, *vap=(&va1);
	dim5 szarr;

	len.d5_dim[1] = OBJ_COLS(dp);
	len.d5_dim[2] = OBJ_ROWS(dp);

dnl	//GET_MAX_THREADS(dp)
	SETUP_BLOCKS_XYZ(OBJ_PFDEV(dp))

	inc1.d5_dim[1] = OBJ_TYPE_INC(dp,1);
	inc1.d5_dim[2] = OBJ_TYPE_INC(dp,2);
	inc1.d5_dim[0] = inc1.d5_dim[3] = inc1.d5_dim[4] = 0;
	inc2 = inc1;

	/* use 2d allocator for better stride? */
	if( cudaMalloc(&depth,sizeof(short)*len.d5_dim[1]*len.d5_dim[2]) != cudaSuccess ){
		NERROR1("h_sp_ifl:  cuda malloc error getting depth array");
	}

	// BUG need to set other elements of va1 !?
	if( setup_slow_len(vap,0,0,1,VA_PFDEV(vap)) < 0 )
		return;

	// Get the value at the seed point
	f_p = (float *)OBJ_DATA_PTR(dp);
	f_p += x + y * inc1.d5_dim[2];

	e = cudaMemcpy(&v, f_p, sizeof(v), cudaMemcpyDeviceToHost);
	CHECK_CUDA_RETURN_VAL("cudaMemcpy device to host");

	CLEAR_CUDA_ERROR(new_fill)
	new_fill<<< NN_GPU >>>
		(szarr,depth,(float *)OBJ_DATA_PTR(dp),inc1,inc2,len,v,tol,fill_val,x,y);
	CHECK_CUDA_ERROR(h_sp_ifl: g_sp_ifl_incs)

}


