dnl	/* Flood fill
dnl	 *
dnl	 * The basic idea is very simple; each iteration we examine each pixel;
dnl	 * it the pixel borders a filled pixel, we test it, and if the test
dnl	 * succeeds, then we fill it, also setting a global flag that something
dnl	 * has changed.  We repeat this until nothing changes.
dnl	 *
dnl	 * That implementation turned out to be very slow... One problem is
dnl	 * that unfilled pixels have to examine all of their neighbors.
dnl	 * We will try a second implementation in which when a pixel is
dnl	 * filled, it marks it's un-filled neighbors.
dnl	 *
dnl	 * No difference.  Eliminating the flag checks after each kernel
dnl	 * launch reduces the time (for 100 iterations) from 11 msec to 7 msec!
dnl	 * This could probably be speeded up quite a bit if the control
dnl	 * logic could be run on the device instead of on the host...
dnl	 * [should be possible with sync-threads!]
dnl	 *
dnl	 * But can we launch a thread array from a device function?
dnl	 * Or should we launch the whole grid and have one special thread
dnl	 * which is the master?
dnl	 * The slow implementation has one thread per pixel in the image;
dnl	 * but many iterations are required... better perhaps to have one
dnl	 * thread per filled pixel with unchecked neighbors?
dnl	 *
dnl	 * We can only synchronize threads within a block, so we would have to
dnl	 * do this with a single block.  Let's say we have one thread per
dnl	 * filled pixel...  Each pixel has up to 4 fillable neighbors (although
dnl	 * only the first seed pixel with have all 4 unfilled).  So we have
dnl	 * an array in shared memory that we fill with the pixel values. (Need
dnl	 * to check how to avoid bank conflicts!)  Then we have a table of
dnl	 * of future pixels.  Each thread gets 4 slots.  After these have
dnl	 * been filled, we would like to prune duplicates; we won't have many
dnl	 * when filling parallel to a coordinate axis, but there will be lots
dnl	 * for an oblique front...  we could use a hash function?  Or use the
dnl	 * flag image.  We could use these values:
dnl	 * 0 - unchecked
dnl	 * 1 - filled
dnl	 * 2 - queued
dnl	 * 3 - rejected
dnl	 *
dnl	 *	0 0 0 0 0    0 0 0 0 0    0 0 2 0 0
dnl	 *	0 0 0 0 0    0 0 2 0 0    0 2 1 2 0
dnl	 *	0 0 2 0 0 -> 0 2 1 2 0 -> 2 1 1 1 2
dnl	 *	0 0 0 0 0    0 0 2 0 0    0 2 1 2 0
dnl	 *	0 0 0 0 0    0 0 0 0 0    0 0 2 0 0
dnl	 *
dnl	 * Shared memory per block is only 16k, so we can't put the whole image
dnl	 * there...
dnl	 *
dnl	 * We have an array of pixels to check, sized 4 times the max number
dnl	 * of threads in a block.  We have an array of active pixels, sized
dnl	 * the max number of threads.  After syncing the threads, we need to make
dnl	 * up the new active pixel list.  We may not have enough threads to do all
dnl	 * of the pixels, so we have several lists.  After processing each list,
dnl	 * we transfer new pixels to be checked to the list, marking them as queued.
dnl	 * If we run out of space, we will have to set a flag that says we
dnl	 * have unrecorded pixels that need to be queued; if that is set when
dnl	 * we are all done, we should scan the entire image again looking for them,
dnl	 * maybe using a special flag value to indicated un-fulfilled queue request?
dnl	 * If we can allocate 2048 queue request slots it ought to be enough
dnl	 * for a 512x512 image...
dnl	 *
dnl	 * We probably want to have the shared memory allocated at launch time...
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
		warn($1);
	}
')

// The fill routine kernel

define(`FILL_IF',`
	if( fabs( dst - v ) < tol ){
		src1 = 1;
		dst = fill_val;
		*flag = 1;
		return;
	}
')

dnl	// why not use vset ???

KERNEL_FUNC_QUALIFIER void zeroit(dim5 szarr, unsigned char* a, dim5 len )
{
	//int x,y;
	DECL_INDICES_1

	//x = blockIdx.x * blockDim.x + threadIdx.x;
	//y = blockIdx.y * blockDim.y + threadIdx.y;
	SET_INDICES_1

	/*
	if( x < len.x && y < len.y ){
		a[x+y*len.y] = 0;
	}
	*/
	dst = 0;
}

KERNEL_FUNC_QUALIFIER void g_sp_ifl_incs(dim5 szarr, float* a, dim5 inc1,
			unsigned char* b, dim5 inc2,
			dim5 len,
			float v, float tol, float fill_val, int *flag)
{
dnl	/* BLOCK_VARS_DECLS */
dnl	//INIT_INDICES_XYZ_2
	DECL_INDICES_2

	SET_INDICES_2

dnl	//if (index1.x < len.x && index1.y < len.y ) {
dnl		//SCALE_INDICES_XYZ_2
		if( src1 == 0 ){	// not filled yet
dnl			// check each neighbor if filled
			if( index2.d5_dim[1] > 0 ){	// in-bounds
				index2.d5_dim[1] -= inc2.d5_dim[1];
				if( src1 > 0 ){	// neighbor filled?
					index2.d5_dim[1] += inc2.d5_dim[1];
					FILL_IF
				}
				index2.d5_dim[1] += inc2.d5_dim[1];
			}
			if( index2.d5_dim[1] < (len.d5_dim[1]-1)*inc2.d5_dim[1] ){
				index2.d5_dim[1] += inc2.d5_dim[1];
				if( src1 > 0 ){
					index2.d5_dim[1] -= inc2.d5_dim[1];
					FILL_IF
				}
				index2.d5_dim[1] -= inc2.d5_dim[1];
			}
			if( index2.d5_dim[2] > 0 ){
				index2.d5_dim[2] -= inc2.d5_dim[2];
				if( src1 > 0 ){
					index2.d5_dim[2] += inc2.d5_dim[2];
					FILL_IF
				}
				index2.d5_dim[2] += inc2.d5_dim[2];
			}
			if( index2.d5_dim[2] < (len.d5_dim[2]-1)*inc2.d5_dim[2] ){
				index2.d5_dim[2] += inc2.d5_dim[2];
				if( src1 > 0 ){
					index2.d5_dim[2] -= inc2.d5_dim[2];
					FILL_IF
				}
				index2.d5_dim[2] -= inc2.d5_dim[2];
			}
		}
dnl	//}
}

__constant__ float test_value[1];
__constant__ float tolerance[1];
__constant__ float fill_value[1];

define(`FILL_IF2',`
	if( fabs( dst - test_value[0] ) < tolerance[0] ){
		src1 = 1;
		dst = fill_value[0];
		return;
	}
')

KERNEL_FUNC_QUALIFIER void g_sp_ifl2_incs(dim5 szarr, float* a, dim5 inc1,
			unsigned char* b, dim5 inc2,
			dim5 len)
{
dnl	/* BLOCK_VARS_DECLS */
dnl	//INIT_INDICES_XYZ_2
	DECL_INDICES_2

	SET_INDICES_2

	if (index1.d5_dim[1] < len.d5_dim[1] && index1.d5_dim[2] < len.d5_dim[2] ) {
dnl		//SCALE_INDICES_XYZ_2
		if( src1 == 0 ){	// not filled yet
			// check each neighbor if filled
			if( index2.d5_dim[1] > 0 ){	// in-bounds
				index2.d5_dim[1] -= inc2.d5_dim[1];
				if( src1 > 0 ){	// neighbor filled?
					index2.d5_dim[1] += inc2.d5_dim[1];
					FILL_IF2
				}
				index2.d5_dim[1] += inc2.d5_dim[1];
			}
			if( index2.d5_dim[1] < (len.d5_dim[1]-1)*inc2.d5_dim[1] ){
				index2.d5_dim[1] += inc2.d5_dim[1];
				if( src1 > 0 ){
					index2.d5_dim[1] -= inc2.d5_dim[1];
					FILL_IF2
				}
				index2.d5_dim[1] -= inc2.d5_dim[1];
			}
			if( index2.d5_dim[2] > 0 ){
				index2.d5_dim[2] -= inc2.d5_dim[2];
				if( src1 > 0 ){
					index2.d5_dim[2] += inc2.d5_dim[2];
					FILL_IF2
				}
				index2.d5_dim[2] += inc2.d5_dim[2];
			}
			if( index2.d5_dim[2] < (len.d5_dim[2]-1)*inc2.d5_dim[2] ){
				index2.d5_dim[2] += inc2.d5_dim[2];
				if( src1 > 0 ){
					index2.d5_dim[2] -= inc2.d5_dim[2];
					FILL_IF2
				}
				index2.d5_dim[2] -= inc2.d5_dim[2];
			}
		}
	}
}

void h_sp_ifl( QSP_ARG_DECL   Data_Obj *dp, int x, int y, float tol, float fill_val )
{
	BLOCK_VARS_DECLS
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

dnl	//GET_MAX_THREADS(dp)
	SETUP_BLOCKS_XYZ(OBJ_PFDEV(dp))

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

	// BUG need to set other elements of va1 !?
	if( setup_slow_len(vap,0,0,1,VA_PFDEV(vap)) < 0 )
		return;

	/* set filled to zero */
	CLEAR_CUDA_ERROR(zeroit)
	zeroit<<< NN_GPU >>>(szarr,filled,len);
	CHECK_CUDA_ERROR(h_sp_ifl: zeroit)

	// Get the value at the seed point
	f_p = (float *)OBJ_DATA_PTR(dp);
	f_p += x + y * inc1.d5_dim[2];

	e = cudaMemcpy(&v, f_p, sizeof(v), cudaMemcpyDeviceToHost);
	CHECK_CUDA_RETURN_VAL("cudaMemcpy device to host");

	// Fill the seed point
	b_one = 1;
	e = cudaMemcpy(filled+x+y*len.d5_dim[1], &b_one, sizeof(b_one),
						cudaMemcpyHostToDevice);
	CHECK_CUDA_RETURN_VAL("cudaMemcpy host to device");

	e = cudaMemcpy(f_p, &fill_val, sizeof(fill_val),
						cudaMemcpyHostToDevice);
	CHECK_CUDA_RETURN_VAL("cudaMemcpy host to device");


	n_iterations=0;
	do {
dnl		/* Clear the flag */
		h_flag = 0;
		e = cudaMemcpy(flag_p, &h_flag, sizeof(h_flag),
						cudaMemcpyHostToDevice);
		CHECK_CUDA_RETURN_VAL("cudaMemcpy host to device");

		CLEAR_CUDA_ERROR(g_sp_ifl_incs)
		g_sp_ifl_incs<<< NN_GPU >>>
		(szarr,(float *)OBJ_DATA_PTR(dp),inc1,filled,inc2,len,v,tol,fill_val,flag_p);
		CHECK_CUDA_ERROR(h_sp_ifl: g_sp_ifl_incs)

		// download flag to see what happened.
		e = cudaMemcpy(&h_flag, flag_p, 1,
						cudaMemcpyDeviceToHost);
		CHECK_CUDA_RETURN_VAL("cudaMemcpy device to host");
		n_iterations++;
	} while( h_flag );

	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,"Fill completed after %d iterations",n_iterations);
		NADVISE(DEFAULT_ERROR_STRING);
	}
}

void h_sp_ifl2( QSP_ARG_DECL  Data_Obj *dp, int seed_x, int seed_y, float tol, float fill_val )
{
	BLOCK_VARS_DECLS
	cudaError_t e;
	dim5 len, inc1, inc2;
	unsigned char *filled, b_one;
	float *f_p, v;
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

dnl	/* use 2d allocator for better stride? */
	if( cudaMalloc(&filled,len.d5_dim[1]*len.d5_dim[2]) != cudaSuccess ){
		NERROR1("cuda malloc error getting filled array");
	}

dnl	/* set filled to zero */
	CLEAR_CUDA_ERROR(zeroit)
	zeroit<<< NN_GPU >>>(szarr,filled,len);
	CHECK_CUDA_ERROR(h_sp_ifl2: zeroit)

dnl	// Get the value at the seed point
	f_p = (float *)OBJ_DATA_PTR(dp);
	f_p += seed_x + seed_y * inc1.d5_dim[2];

	e = cudaMemcpy(&v, f_p, sizeof(v), cudaMemcpyDeviceToHost);
	CHECK_CUDA_RETURN_VAL("cudaMemcpy device to host");

dnl	// Fill the seed point
	b_one = 1;
	e = cudaMemcpy(filled+seed_x+seed_y*len.d5_dim[1], &b_one, sizeof(b_one),
						cudaMemcpyHostToDevice);
	CHECK_CUDA_RETURN_VAL("cudaMemcpy host to device");
	e = cudaMemcpy(f_p, &fill_val, sizeof(fill_val),
						cudaMemcpyHostToDevice);
	CHECK_CUDA_RETURN_VAL("cudaMemcpy host to device");

	e = cudaMemcpyToSymbol(fill_value, &fill_val, sizeof(float));
	CHECK_CUDA_RETURN_VAL("cudaMemcpyToSymbol");
	e = cudaMemcpyToSymbol(tolerance, &tol, sizeof(float));
	CHECK_CUDA_RETURN_VAL("cudaMemcpyToSymbol");
	e = cudaMemcpyToSymbol(test_value, &v, sizeof(float));
	CHECK_CUDA_RETURN_VAL("cudaMemcpyToSymbol");

	n_iterations=0;
	for( n_iterations = 0 ; n_iterations < 300 ; n_iterations++ ){

		CLEAR_CUDA_ERROR(g_sp_ifl2_incs)
		g_sp_ifl2_incs<<< NN_GPU >>>
		(szarr,(float *)OBJ_DATA_PTR(dp),inc1,filled,inc2,len);
		CHECK_CUDA_ERROR(h_sp_ifl2:  g_sp_ifl2_incs)

	}


	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,"Fill completed after %d iterations",n_iterations);
		NADVISE(DEFAULT_ERROR_STRING);
	}
}

