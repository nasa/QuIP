#ifndef _SLOW_LEN_H_
#define _SLOW_LEN_H_

// These length macros used to be called XFER_..._LEN, but that is
// confusing because the other XFER macros are used to transfer
// information from a Vec_Obj_Args struct to a Vector_Args struct.
// Here, we are taking info FROM Vector_Args, to get dimensions
// to be used for GPU gridding...

#ifdef __cplusplus
extern "C" {
#endif

#include "platform.h"
#include "veclib/dim3.h"

extern int setup_slow_len(	/* DIM3 *len_p, */ /* use vap */
				/* Size_Info *szi_p, */
				Vector_Args *vap,
				dimension_t start_dim,
				/* int *dim_indices, */	/* now in vap */
				int i_first,
				int n_vec,
				Platform_Device *pdp);

#ifdef FOOBAR
// now in vap
#define SETUP_FAST_LEN		len=VA_LENGTH(vap);
#endif // FOOBAR

// SETUP_SLOW_LEN takes the dimension arrays (up to 5 dieensions)
// and figures out which 3 to use, putting the dimensions in len
// and which were chosen in dim_indices.
// There is a problem with complex numbers, as the type dimension
// is 2...
//
// These are only the source dimensions.

#define SETUP_SLOW_LEN_RC_N(n)					\
								\
if( setup_slow_len(vap,1,0,n,VA_PFDEV(vap)) < 0 )	\
	return;

#define SETUP_SLOW_LEN_QUAT_N(n)	SETUP_SLOW_LEN_CPX_N(n)

#define SETUP_SLOW_LEN_CPX_N(n)					\
								\
if( setup_slow_len(vap,1,0,n,VA_PFDEV(vap)) < 0 )	\
	return;

#define SETUP_SLOW_LEN_N(n)					\
								\
if( setup_slow_len(vap,0,0,n,VA_PFDEV(vap)) < 0 )	\
	return;

// The way this is used, it appears we skip the first source - why???
// 1SRC and 2SRCS are used with destination bitmaps???

#define SETUP_SLOW_LEN_F(first,n)					\
									\
if( setup_slow_len(vap,0,first,n,VA_PFDEV(vap)) < 0 )	\
	return;


#define SETUP_SLOW_LEN_1	SETUP_SLOW_LEN_N(1)
#define SETUP_SLOW_LEN_2	SETUP_SLOW_LEN_N(2)
#define SETUP_SLOW_LEN_CONV	SETUP_SLOW_LEN_N(2)
#define SETUP_SLOW_LEN_3	SETUP_SLOW_LEN_N(3)
#define SETUP_SLOW_LEN_4	SETUP_SLOW_LEN_N(4)
#define SETUP_SLOW_LEN_5	SETUP_SLOW_LEN_N(5)

#define SETUP_SLOW_LEN_CPX_1	SETUP_SLOW_LEN_CPX_N(1)
#define SETUP_SLOW_LEN_CPX_2	SETUP_SLOW_LEN_CPX_N(2)
#define SETUP_SLOW_LEN_CPX_3	SETUP_SLOW_LEN_CPX_N(3)
#define SETUP_SLOW_LEN_CPX_4	SETUP_SLOW_LEN_CPX_N(4)
#define SETUP_SLOW_LEN_CPX_5	SETUP_SLOW_LEN_CPX_N(5)

#define SETUP_SLOW_LEN_QUAT_1	SETUP_SLOW_LEN_CPX_N(1)
#define SETUP_SLOW_LEN_QUAT_2	SETUP_SLOW_LEN_CPX_N(2)
#define SETUP_SLOW_LEN_QUAT_3	SETUP_SLOW_LEN_CPX_N(3)
#define SETUP_SLOW_LEN_QUAT_4	SETUP_SLOW_LEN_CPX_N(4)
#define SETUP_SLOW_LEN_QUAT_5	SETUP_SLOW_LEN_CPX_N(5)

#define SETUP_SLOW_LEN_		SETUP_SLOW_LEN_F(1,1)
#define SETUP_SLOW_LEN_1SRC	SETUP_SLOW_LEN_F(1,1)
#define SETUP_SLOW_LEN_2SRCS	SETUP_SLOW_LEN_F(1,2)

#define SETUP_SLOW_LEN_RC_2	SETUP_SLOW_LEN_RC_N(2)
#define SETUP_SLOW_LEN_RQ_2	SETUP_SLOW_LEN_RC_N(2)

// Not sure if these are correct...
#define SETUP_SLOW_LEN_CCR_3	SETUP_SLOW_LEN_CPX_3
#define SETUP_SLOW_LEN_CR_2	SETUP_SLOW_LEN_CPX_2

#define SETUP_SLOW_LEN_QQR_3	SETUP_SLOW_LEN_CPX_3
#define SETUP_SLOW_LEN_QR_2	SETUP_SLOW_LEN_CPX_2


// dest bitmap is like a normal dest vector?
#define SETUP_SLOW_LEN_DBM_2SRCS	SETUP_SLOW_LEN_3
#define SETUP_SLOW_LEN_DBM_1SRC		SETUP_SLOW_LEN_2
// BUG should use a symbolic constant instead of 4 here?
#define SETUP_SLOW_LEN_DBM_SBM		SETUP_SLOW_LEN_F(4,1)
#define SETUP_SLOW_LEN_DBM_		SETUP_SLOW_LEN_1

// Not sure about how to handle source bitmaps?
#define SETUP_SLOW_LEN_SBM_1		SETUP_SLOW_LEN_1
#define SETUP_SLOW_LEN_SBM_2		SETUP_SLOW_LEN_2
#define SETUP_SLOW_LEN_SBM_3		SETUP_SLOW_LEN_3

#ifdef __cplusplus
}
#endif


#endif // ! _SLOW_LEN_H_


