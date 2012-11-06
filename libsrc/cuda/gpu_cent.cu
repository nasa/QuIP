/* special helper function for computing centroids */

#define CENT_KERNEL( typ )						\
									\
__global__ void typ##_slow_cent_helper					\
( std_type *x_array, dim3 inc1, std_type *y_array, dim3 inc2,		\
	std_type *input, dim3 inc3, dim3 len )				\
									\
{									\
	dim3 index;							\
	uint32_t offset1, offset2, offset3;				\
	std_type p;							\
									\
	index.x = blockIdx.x * blockDim.x + threadIdx.x;		\
	index.y = blockIdx.y * blockDim.y + threadIdx.y;		\
									\
	offset1 = index.y * inc1.x + index.x;				\
	offset2 = index.y * inc2.x + index.x;				\
	offset3 = index.y * inc3.x + index.x;				\
									\
	p = *(input + offset3);						\
	*(x_array+offset1) = p * index.x;				\
	*(y_array+offset2) = p * index.y;				\
}

#define CK( c )		CENT_KERNEL( c )

CK( type_code )

