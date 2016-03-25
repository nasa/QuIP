
#define std_type short
#define std_scalar u_s
#define dest_type short
#define ALL_ONES 0xffff
// Broken on CUDA 6?
//#define absfunc abs
#define absfunc(val)	(val<0?(-val):val)
#define type_code in

