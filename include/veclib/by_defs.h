
#define type_code by
#define std_type char
#define std_scalar u_b
#define dest_type char
#define ALL_ONES 0xff

// Broken on CUDA 6?
//#define absfunc abs
#ifdef BUILD_FOR_CUDA
#define absfunc(val)	(val<0?(-val):val)
#else // ! BUILD_FOR_CUDA
#define absfunc abs
#endif // ! BUILD_FOR_CUDA


