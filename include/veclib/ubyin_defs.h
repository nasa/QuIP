
#define std_type u_char
#define std_scalar u_ub
#define std_signed char
#define dest_type short
#define ALL_ONES 0xffff
// Broken on CUDA 6?
//#define absfunc abs
#define absfunc(val)	(val<0?(-val):val)
#define type_code ubyin

#define MIXED_PRECISION

