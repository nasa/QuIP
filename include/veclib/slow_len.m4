
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

dnl SETUP_SLOW_LEN(bitmaps,typ,vectors)
define(`SETUP_SLOW_LEN',SETUP_SLOW_LEN_$1$2$3)

dnl SETUP_SLOW_LEN_RC_N(n)
define(`SETUP_SLOW_LEN_RC_N',`if( setup_slow_len(vap,1,0,$1,VA_PFDEV(vap)) < 0 ) return;')

define(`SETUP_SLOW_LEN_QUAT_N',`SETUP_SLOW_LEN_CPX_N($1)')

define(`SETUP_SLOW_LEN_CPX_N',`if( setup_slow_len(vap,1,0,$1,VA_PFDEV(vap)) < 0 ) return;')

define(`SETUP_SLOW_LEN_N',`if( setup_slow_len(vap,0,0,$1,VA_PFDEV(vap)) < 0 ) return;')

dnl The way this is used, it appears we skip the first source - why???
dnl 1SRC and 2SRCS are used with destination bitmaps???

dnl SETUP_SLOW_LEN_F(first,n)

define(`SETUP_SLOW_LEN_F',`if( setup_slow_len(vap,0,$1,$2,VA_PFDEV(vap)) < 0 ) return;')


define(`SETUP_SLOW_LEN_1',`SETUP_SLOW_LEN_N(1)')
define(`SETUP_SLOW_LEN_2',`SETUP_SLOW_LEN_N(2)')
define(`SETUP_SLOW_LEN_CONV',`SETUP_SLOW_LEN_N(2)')
define(`SETUP_SLOW_LEN_3',`SETUP_SLOW_LEN_N(3)')
define(`SETUP_SLOW_LEN_4',`SETUP_SLOW_LEN_N(4)')
define(`SETUP_SLOW_LEN_5',`SETUP_SLOW_LEN_N(5)')

define(`SETUP_SLOW_LEN_CPX_1',`SETUP_SLOW_LEN_CPX_N(1)')
define(`SETUP_SLOW_LEN_CPX_2',`SETUP_SLOW_LEN_CPX_N(2)')
define(`SETUP_SLOW_LEN_CPX_3',`SETUP_SLOW_LEN_CPX_N(3)')
define(`SETUP_SLOW_LEN_CPX_4',`SETUP_SLOW_LEN_CPX_N(4)')
define(`SETUP_SLOW_LEN_CPX_5',`SETUP_SLOW_LEN_CPX_N(5)')

define(`SETUP_SLOW_LEN_QUAT_1',`SETUP_SLOW_LEN_CPX_N(1)')
define(`SETUP_SLOW_LEN_QUAT_2',`SETUP_SLOW_LEN_CPX_N(2)')
define(`SETUP_SLOW_LEN_QUAT_3',`SETUP_SLOW_LEN_CPX_N(3)')
define(`SETUP_SLOW_LEN_QUAT_4',`SETUP_SLOW_LEN_CPX_N(4)')
define(`SETUP_SLOW_LEN_QUAT_5',`SETUP_SLOW_LEN_CPX_N(5)')

define(`SETUP_SLOW_LEN_',`SETUP_SLOW_LEN_F(1,1)')
define(`SETUP_SLOW_LEN_1SRC',`SETUP_SLOW_LEN_F(1,1)')
define(`SETUP_SLOW_LEN_2SRCS',`SETUP_SLOW_LEN_F(1,2)')

define(`SETUP_SLOW_LEN_RC_2',`SETUP_SLOW_LEN_RC_N(2)')
define(`SETUP_SLOW_LEN_RQ_2',`SETUP_SLOW_LEN_RC_N(2)')

// Not sure if these are correct...
define(`SETUP_SLOW_LEN_CCR_3',`SETUP_SLOW_LEN_CPX_3')
define(`SETUP_SLOW_LEN_CR_2',`SETUP_SLOW_LEN_CPX_2')

define(`SETUP_SLOW_LEN_QQR_3',`SETUP_SLOW_LEN_CPX_3')
define(`SETUP_SLOW_LEN_QR_2',`SETUP_SLOW_LEN_CPX_2')


// dest bitmap is like a normal dest vector?
define(`SETUP_SLOW_LEN_DBM_2SRCS',`SETUP_SLOW_LEN_3')
define(`SETUP_SLOW_LEN_DBM_1SRC',`SETUP_SLOW_LEN_2')
// BUG should use a symbolic constant instead of 4 here?
define(`SETUP_SLOW_LEN_DBM_SBM',`SETUP_SLOW_LEN_F(4,1)')
define(`SETUP_SLOW_LEN_DBM_',`SETUP_SLOW_LEN_1')

// Not sure about how to handle source bitmaps?
define(`SETUP_SLOW_LEN_SBM_1',`SETUP_SLOW_LEN_1')
define(`SETUP_SLOW_LEN_SBM_2',`SETUP_SLOW_LEN_2')
define(`SETUP_SLOW_LEN_SBM_3',`SETUP_SLOW_LEN_3')

define(`SETUP_SLOW_LEN_SBM_CPX_1',`SETUP_SLOW_LEN_CPX_1')
define(`SETUP_SLOW_LEN_SBM_CPX_2',`SETUP_SLOW_LEN_CPX_2')
define(`SETUP_SLOW_LEN_SBM_CPX_3',`SETUP_SLOW_LEN_CPX_3')

define(`SETUP_SLOW_LEN_SBM_QUAT_1',`SETUP_SLOW_LEN_QUAT_1')
define(`SETUP_SLOW_LEN_SBM_QUAT_2',`SETUP_SLOW_LEN_QUAT_2')
define(`SETUP_SLOW_LEN_SBM_QUAT_3',`SETUP_SLOW_LEN_QUAT_3')


