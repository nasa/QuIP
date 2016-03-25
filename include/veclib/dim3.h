
#ifndef DIM3

#define DECLARE_DIM3_STRUCT			\
typedef struct {				\
	unsigned int	x;			\
	unsigned int	y;			\
	unsigned int	z;			\
} dim3;

//#ifndef HAVE_CUDA
#ifdef BUILD_FOR_CUDA
// include the cuda header that defines dim3 here...
#include <vector_types.h>	// cuda header that defines dim3?

#else
DECLARE_DIM3_STRUCT
#endif // BUILD_FOR_CUDA
//#endif // ! HAVE_CUDA

#define DIM3 dim3

#endif // ! DIM3

