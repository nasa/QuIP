/* special helper function for computing centroids */

// What is the strategy here?
// 
// We have two output arrays (for the weighted coordinates) and one input array
// with the values.  For each pixel, we compute the x and y coordinates from the
// thread index, and store the product of these with the pixel value.
// Later we will compute the sum...
//
// This could probably be composed as two vmul's followed by a vsum?

#define KERNEL_FUNC_QUALIFIER __global__

#define CENT_KERNEL( typ )						\
									\
KERNEL_FUNC_QUALIFIER void typ##_slow_cent_helper					\
( /*std_type *x_array, dim3 inc1, std_type *y_array, dim3 inc2,		\
	std_type *input, dim3 inc3, dim3 len*/ DECLARE_KERN_ARGS_SLEN_3 )	\
									\
{									\
	/*dim3 index;*/							\
	/*uint32_t offset1, offset2, offset3;*/				\
	DECL_INDICES_3							\
	std_type p;							\
									\
	/*index.x = blockIdx.x * blockDim.x + threadIdx.x;*/		\
	/*index.y = blockIdx.y * blockDim.y + threadIdx.y;*/		\
	SET_INDICES_3							\
									\
	/*offset1 = index.y * inc1.x + index.x;*/				\
	/*offset2 = index.y * inc2.x + index.x;*/				\
	/*offset3 = index.y * inc3.x + index.x;*/				\
									\
	/*p = *( c + offset3);*/						\
	p = slow_src2;	/* third arg, no first source */		\
	slow_dst1 = p * index3.d5_dim[1]; /* x */			\
	slow_dst2 = p * index3.d5_dim[2]; /* y */			\
	/* *(a+offset1) = p * index.x; */				\
	/* *(b+offset2) = p * index.y; */				\
}

#define CK( c )		CENT_KERNEL( c )

CK( type_code )

