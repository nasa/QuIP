suppress_if		dnl	this file is just definitions

dnl	// These are used in the declarations of the kernels,
dnl	// and the function calls of the kernels.

define(`GEN_ARGS_1S',$1`_1S')
define(`GEN_ARGS_2S',$1`_2S')
define(`GEN_ARGS_3S',$1`_3S')

define(`GEN_ARGS_CPX_1S',$1`_CPX_1S')
define(`GEN_ARGS_CPX_2S',$1`_CPX_2S')
define(`GEN_ARGS_CPX_3S',$1`_CPX_3S')

define(`GEN_ARGS_QUAT_1S',$1`_QUAT_1S')
define(`GEN_ARGS_QUAT_2S',$1`_QUAT_2S')
define(`GEN_ARGS_QUAT_3S',$1`_QUAT_3S')



define(`GEN_SEP',$1`_SEPARATOR')
define(`DECLARE_KERN_ARGS_SEPARATOR',`,')
define(`KERN_ARGS_SEPARATOR',`,')
define(`SET_KERNEL_ARGS_SEPARATOR',`')

define(`GEN_FAST_ARG_LEN',$1`_FAST_LEN')
define(`GEN_EQSP_ARG_LEN',$1`_EQSP_LEN')
define(`GEN_SLOW_ARG_LEN',$1`_SLOW_LEN')

define(`GEN_ARGS_EQSP_SBM',$1`_EQSP_SBM')
define(`GEN_ARGS_EQSP_SBM1',$1`_EQSP_SBM1')
define(`GEN_ARGS_EQSP_SBM2',$1`_EQSP_SBM2')
define(`GEN_ARGS_SLOW_SBM',$1`_SLOW_SBM')
define(`GEN_ARGS_SLOW_SBM1',$1`_SLOW_SBM1')
define(`GEN_ARGS_SLOW_SBM2',$1`_SLOW_SBM2')

define(`GEN_ARGS_FLEN_DBM',`GEN_DBM_GPU_INFO($1) GEN_SEP($1) GEN_FLEN_DBM($1)')

dnl	GEN_ARGS_EQSP_DBM(prefix)
define(`GEN_ARGS_EQSP_DBM',`GEN_DBM_GPU_INFO($1) GEN_SEP($1) GEN_EQSP_DBM($1)')

dnl	GEN_ARGS_ELEN(prefix,bitmaps,typ,scalars,vectors)
define(`GEN_ARGS_ELEN',`_GEN_ARGS_ELEN($1,$2$3$5$4)')
define(`_GEN_ARGS_ELEN',`GEN_ARGS_ELEN_'$2($1))

define(`GEN_ARGS_ELEN_DBM',`GEN_DBM_GPU_INFO($1) GEN_SEP($1) GEN_ELEN_DBM($1)')

define(`GEN_FLEN_DBM',$1`_FLEN_DBM')
define(`GEN_EQSP_DBM',$1`_EQSP_DBM')
define(`GEN_ELEN_DBM',$1`_ELEN_DBM')
define(`GEN_SLOW_DBM',$1`_SLOW_DBM')
define(`GEN_SLEN_DBM',$1`_SLEN_DBM')

// BUG?  need to make sure that GPU_INFO gets inserted everywhere that's necessary!?

dnl define(`GEN_ARGS_SLOW_DBM',`GEN_SLOW_SIZE($1) GEN_SEP($1) GEN_DBM_GPU_INFO($1) GEN_SEP($1) GEN_SLOW_DBM($1)')
define(`GEN_ARGS_SLOW_DBM',`GEN_DBM_GPU_INFO($1) GEN_SEP($1) GEN_SLOW_DBM($1)')
define(`GEN_ARGS_SLEN_DBM',`GEN_DBM_GPU_INFO($1) GEN_SEP($1) GEN_SLOW_DBM($1)')

dnl define(`GEN_ARGS_SLOW_DBM_BASIC',`GEN_SLOW_DBM_GPU_INFO($1) GEN_SEP($1) $1`_SLOW_DBM'')

dnl // define(`GEN_ARGS_NOCC_SETUP',`GEN_SLOW_SIZE($1)
dnl // GEN_SEP($1)
dnl // $1`_NOCC_SETUP')

////////// FAST

define(`GEN_FAST_CONV_DEST',$1`_FAST_CONV_DEST'($2) OS_ARG($1,DEST))

define(`GEN_ARGS_FAST_1',$1`_FAST_1' OS_ARG($1,DEST))
define(`GEN_ARGS_FAST_CPX_1',$1`_FAST_CPX_1' OS_ARG($1,DEST))
define(`GEN_ARGS_FAST_QUAT_1',$1`_FAST_QUAT_1' OS_ARG($1,DEST))

define(`GEN_ARGS_FAST_SRC1',$1`_FAST_SRC1' OS_ARG($1,SRC1))
define(`GEN_ARGS_FAST_BSRC1',$1`_FAST_BSRC1' OS_ARG($1,BSRC1))
define(`GEN_ARGS_FAST_SSRC1',$1`_FAST_SSRC1' OS_ARG($1,SSRC1))
define(`GEN_ARGS_FAST_CPX_SRC1',$1`_FAST_CPX_SRC1' OS_ARG($1,SRC1))
define(`GEN_ARGS_FAST_QUAT_SRC1',$1`_FAST_QUAT_SRC1' OS_ARG($1,SRC1))

define(`GEN_ARGS_FAST_SRC2',$1`_FAST_SRC2' OS_ARG($1,SRC2))
define(`GEN_ARGS_FAST_MAP_B',$1`_FAST_MAP_B' OS_ARG($1,MAP))
define(`GEN_ARGS_FAST_MAP_S',$1`_FAST_MAP_S' OS_ARG($1,MAP))
define(`GEN_ARGS_FAST_CPX_SRC2',$1`_FAST_CPX_SRC2' OS_ARG($1,SRC2))
define(`GEN_ARGS_FAST_QUAT_SRC2',$1`_FAST_QUAT_SRC2' OS_ARG($1,SRC2))

define(`GEN_ARGS_FAST_SRC3',$1`_FAST_SRC3' OS_ARG($1,SRC3))
define(`GEN_ARGS_FAST_CPX_SRC3',$1`_FAST_CPX_SRC3' OS_ARG($1,SRC3))
define(`GEN_ARGS_FAST_QUAT_SRC3',$1`_FAST_QUAT_SRC3' OS_ARG($1,SRC3))

define(`GEN_ARGS_FAST_SRC4',$1`_FAST_SRC4' OS_ARG($1,SRC4))
define(`GEN_ARGS_FAST_CPX_SRC4',$1`_FAST_CPX_SRC4' OS_ARG($1,SRC4))
define(`GEN_ARGS_FAST_QUAT_SRC4',$1`_FAST_QUAT_SRC4' OS_ARG($1,SRC4))

define(`GEN_ARGS_FAST_SBM',$1`_FAST_SBM' OS_ARG($1,SBM))
define(`GEN_ARGS_FAST_SBM1',$1`_FAST_SBM1' OS_ARG($1,SBM1))
define(`GEN_ARGS_FAST_SBM2',$1`_FAST_SBM2' OS_ARG($1,SBM2))
define(`GEN_ARGS_FAST_DBM',$1`_FAST_DBM' OS_ARG($1,DBM))

////// FLEN

dnl	GEN_ARGS_FAST(prefix,bitmaps,typ,scalars,vectors)
define(`GEN_ARGS_FAST',`_GEN_ARGS_FAST($1,$2$3$5$4)')
define(`_GEN_ARGS_FAST',`GEN_ARGS_FAST_$2($1)')

dnl	GEN_ARGS_FLEN(prefix,bitmaps,typ,scalars,vectors)
define(`GEN_ARGS_FLEN',`_GEN_ARGS_FLEN($1,$2$3$5$4)')
define(`_GEN_ARGS_FLEN',`GEN_ARGS_FLEN_$2($1)')

dnl	GEN_ARGS_EQSP(prefix,bitmaps,typ,scalars,vectors)
define(`GEN_ARGS_EQSP',`_GEN_ARGS_EQSP($1,$2$3$5$4)')
define(`_GEN_ARGS_EQSP',`GEN_ARGS_EQSP_$2($1)')

dnl	GEN_ARGS_SLOW(prefix,bitmaps,typ,scalars,vectors)
define(`GEN_ARGS_SLOW',`_GEN_ARGS_SLOW($1,$2$3$5$4)')
define(`_GEN_ARGS_SLOW',`GEN_ARGS_SLOW_$2($1)')

dnl	GEN_ARGS_SLEN(prefix,bitmaps,typ,scalars,vectors)
define(`GEN_ARGS_SLEN',`_GEN_ARGS_SLEN($1,$2$3$5$4)')
define(`_GEN_ARGS_SLEN',`GEN_ARGS_SLEN_$2($1)')

define(`GEN_ARGS_FLEN_1',`GEN_ARGS_FAST_1($1)')
define(`GEN_ARGS_FLEN_SRC1',`GEN_ARGS_FAST_SRC1($1)')
define(`GEN_ARGS_FLEN_SRC2',`GEN_ARGS_FAST_SRC2($1)')
define(`GEN_ARGS_FLEN_SRC3',`GEN_ARGS_FAST_SRC3($1)')
define(`GEN_ARGS_FLEN_SRC4',`GEN_ARGS_FAST_SRC4($1)')

///// EQSP

define(`GEN_EQSP_ARG_INC1',$1`_EQSP_INC1')
define(`GEN_EQSP_ARG_INC2',$1`_EQSP_INC2')
define(`GEN_EQSP_ARG_INC3',$1`_EQSP_INC3')
define(`GEN_EQSP_ARG_INC4',$1`_EQSP_INC4')
define(`GEN_EQSP_ARG_INC5',$1`_EQSP_INC5')

define(`GEN_SLOW_ARG_INC1',$1`_SLOW_INC1')
define(`GEN_SLOW_ARG_INC2',$1`_SLOW_INC2')
define(`GEN_SLOW_ARG_INC3',$1`_SLOW_INC3')
define(`GEN_SLOW_ARG_INC4',$1`_SLOW_INC4')
define(`GEN_SLOW_ARG_INC5',$1`_SLOW_INC5')

//////////  Everything after this point recombines stuff in this file

////////// FAST

define(`GEN_ARGS_FAST__1S',`GEN_ARGS_1S($1)')

define(`GEN_ARGS_FAST_1_1S',`GEN_ARGS_FAST_1($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_FAST_CPX_1_1S',`GEN_ARGS_FAST_CPX_1($1) GEN_SEP($1) GEN_ARGS_CPX_1S($1)')

define(`GEN_ARGS_FAST_QUAT_1_1S',`GEN_ARGS_FAST_QUAT_1($1) GEN_SEP($1) GEN_ARGS_QUAT_1S($1)')

define(`GEN_ARGS_FAST_1_2S',`GEN_ARGS_FAST_1($1) GEN_SEP($1) GEN_ARGS_2S($1)')

define(`GEN_ARGS_FAST_1_3S',`GEN_ARGS_FAST_1($1) GEN_SEP($1) GEN_ARGS_3S($1)')

define(`GEN_ARGS_FAST_SBM_1_2S',`GEN_ARGS_FAST_SBM_1($1) GEN_SEP($1) GEN_ARGS_2S($1)')

define(`GEN_ARGS_FAST_SBM_CPX_1_2S',`GEN_ARGS_FAST_SBM_CPX_1($1) GEN_SEP($1) GEN_ARGS_CPX_2S($1)')

define(`GEN_ARGS_FAST_SBM_QUAT_1_2S',`GEN_ARGS_FAST_SBM_QUAT_1($1) GEN_SEP($1) GEN_ARGS_QUAT_2S($1)')

define(`GEN_ARGS_FAST_RC_2',`GEN_ARGS_FAST_1($1) GEN_SEP($1) GEN_ARGS_FAST_CPX_SRC1($1)')

define(`GEN_ARGS_FAST_DBM_SBM',`GEN_ARGS_FAST_DBM($1) GEN_SEP($1) GEN_ARGS_FAST_SBM($1)')
define(`GEN_ARGS_FAST_DBM_2SBM',`GEN_ARGS_FAST_DBM($1) GEN_SEP($1) GEN_ARGS_FAST_SBM1($1) GEN_SEP($1) GEN_ARGS_FAST_SBM2($1)')
define(`GEN_ARGS_FAST_DBM_1SBM',`GEN_ARGS_FAST_DBM($1) GEN_SEP($1) GEN_ARGS_FAST_SBM1($1)')
define(`GEN_ARGS_FAST_DBM_1SBM_1S',`GEN_ARGS_FAST_DBM($1) GEN_SEP($1) GEN_ARGS_FAST_SBM1($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_FAST_DBM_2SRCS',`GEN_ARGS_FAST_DBM($1) GEN_SEP($1) GEN_ARGS_FAST_2SRCS($1)')

define(`GEN_ARGS_FAST_DBM_1SRC_1S',`GEN_ARGS_FAST_1SRC_1S($1) GEN_SEP($1) GEN_ARGS_FAST_DBM($1)')

define(`GEN_ARGS_FAST_DBM__1S',`GEN_ARGS_1S($1) GEN_SEP($1) GEN_ARGS_FAST_DBM($1)')

define(`GEN_ARGS_FAST_CONV',`GEN_FAST_CONV_DEST($1,$2) GEN_SEP($1) GEN_ARGS_FAST_SRC1($1)')

define(`GEN_ARGS_FAST_2',`GEN_ARGS_FAST_1($1) GEN_SEP($1) GEN_ARGS_FAST_SRC1($1)')

define(`GEN_ARGS_FAST_CPX_2',`GEN_ARGS_FAST_CPX_1($1) GEN_SEP($1) GEN_ARGS_FAST_CPX_SRC1($1)')

define(`GEN_ARGS_FAST_CR_2',`GEN_ARGS_FAST_CPX_1($1) GEN_SEP($1) GEN_ARGS_FAST_SRC1($1)')

define(`GEN_ARGS_FAST_QR_2',`GEN_ARGS_FAST_QUAT_1($1) GEN_SEP($1) GEN_ARGS_FAST_SRC1($1)')

define(`GEN_ARGS_FAST_2SRCS',`GEN_ARGS_FAST_SRC1($1) GEN_SEP($1) GEN_ARGS_FAST_SRC2($1)')

define(`GEN_ARGS_FAST_1SRC_1S',`GEN_ARGS_FAST_SRC1($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_FAST_ARG_DBM_2SRCS',`GEN_ARGS_FAST_2SRCS($1) GEN_SEP($1) GEN_ARGS_FAST_DBM($1)')

define(`GEN_ARGS_FAST_SBM_1',`GEN_ARGS_FAST_1($1) GEN_SEP($1) GEN_ARGS_FAST_SBM($1)')

define(`GEN_ARGS_FAST_SBM_CPX_1',`GEN_ARGS_FAST_CPX_1($1) GEN_SEP($1) GEN_ARGS_FAST_SBM($1)')

define(`GEN_ARGS_FAST_SBM_QUAT_1',`GEN_ARGS_FAST_QUAT_1($1) GEN_SEP($1) GEN_ARGS_FAST_SBM($1)')

define(`GEN_ARGS_FAST_SBM_2',`GEN_ARGS_FAST_2($1) GEN_SEP($1) GEN_ARGS_FAST_SBM($1)')

define(`GEN_ARGS_FAST_SBM_CPX_2',`GEN_ARGS_FAST_CPX_2($1) GEN_SEP($1) GEN_ARGS_FAST_SBM($1)')

define(`GEN_ARGS_FAST_SBM_QUAT_2',`GEN_ARGS_FAST_QUAT_2($1) GEN_SEP($1) GEN_ARGS_FAST_SBM($1)')

define(`GEN_ARGS_FAST_QUAT_2',`GEN_ARGS_FAST_QUAT_1($1) GEN_SEP($1) GEN_ARGS_FAST_QUAT_SRC1($1)')

define(`GEN_ARGS_FAST_QR_2',`GEN_ARGS_FAST_QUAT_1($1) GEN_SEP($1) GEN_ARGS_FAST_SRC1($1)')

define(`GEN_ARGS_FAST_4',`GEN_ARGS_FAST_3($1) GEN_SEP($1) GEN_ARGS_FAST_SRC3($1)')

define(`GEN_ARGS_FAST_5',`GEN_ARGS_FAST_4($1) GEN_SEP($1) GEN_ARGS_FAST_SRC4($1)')

define(`GEN_ARGS_FAST_4_1S',`GEN_ARGS_FAST_4($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_FAST_3_2S',`GEN_ARGS_FAST_3($1) GEN_SEP($1) GEN_ARGS_2S($1)')

define(`GEN_ARGS_FAST_2_3S',`GEN_ARGS_FAST_2($1) GEN_SEP($1) GEN_ARGS_3S($1)')

define(`GEN_ARGS_FAST_3',`GEN_ARGS_FAST_2($1) GEN_SEP($1) GEN_ARGS_FAST_SRC2($1)')

define(`GEN_ARGS_FAST_LUTMAP_B',`GEN_ARGS_FAST_1($1) GEN_SEP($1) GEN_ARGS_FAST_BSRC1($1)  GEN_SEP($1) GEN_ARGS_FAST_MAP_B($1)')
define(`GEN_ARGS_FAST_LUTMAP_S',`GEN_ARGS_FAST_1($1) GEN_SEP($1) GEN_ARGS_FAST_SSRC1($1)  GEN_SEP($1) GEN_ARGS_FAST_MAP_S($1)')

define(`GEN_ARGS_FAST_CPX_3',`GEN_ARGS_FAST_CPX_2($1) GEN_SEP($1) GEN_ARGS_FAST_CPX_SRC2($1)')

define(`GEN_ARGS_FAST_CCR_3',`GEN_ARGS_FAST_CPX_2($1) GEN_SEP($1) GEN_ARGS_FAST_SRC2($1)')

define(`GEN_ARGS_FAST_QUAT_3',`GEN_ARGS_FAST_QUAT_2($1) GEN_SEP($1) GEN_ARGS_FAST_QUAT_SRC2($1)')

define(`GEN_ARGS_FAST_QQR_3',`GEN_ARGS_FAST_QUAT_2($1) GEN_SEP($1) GEN_ARGS_FAST_SRC2($1)')



define(`GEN_ARGS_FAST_SBM_3',`GEN_ARGS_FAST_3($1) GEN_SEP($1) GEN_ARGS_FAST_SBM($1)')

define(`GEN_ARGS_FAST_SBM_CPX_3',`GEN_ARGS_FAST_CPX_3($1) GEN_SEP($1) GEN_ARGS_FAST_SBM($1)')

define(`GEN_ARGS_FAST_SBM_QUAT_3',`GEN_ARGS_FAST_QUAT_3($1) GEN_SEP($1) GEN_ARGS_FAST_SBM($1)')

define(`GEN_ARGS_FAST_2_1S',`GEN_ARGS_FAST_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_FAST_SBM_2_1S',`GEN_ARGS_FAST_SBM_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_FAST_CPX_2_1S',`GEN_ARGS_FAST_CPX_2($1) GEN_SEP($1) GEN_ARGS_CPX_1S($1)')

define(`GEN_ARGS_FAST_SBM_CPX_2_1S',`GEN_ARGS_FAST_SBM_CPX_2($1) GEN_SEP($1) GEN_ARGS_CPX_1S($1)')

define(`GEN_ARGS_FAST_SBM_QUAT_2_1S',`GEN_ARGS_FAST_SBM_QUAT_2($1) GEN_SEP($1) GEN_ARGS_QUAT_1S($1)')

define(`GEN_ARGS_FAST_QUAT_2_1S',`GEN_ARGS_FAST_QUAT_2($1) GEN_SEP($1) GEN_ARGS_QUAT_1S($1)')

define(`GEN_ARGS_FAST_CR_2_1S',`GEN_ARGS_FAST_CR_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_FAST_QUAT_2_1S',`GEN_ARGS_FAST_QUAT_2($1) GEN_SEP($1) GEN_ARGS_QUAT_1S($1)')

define(`GEN_ARGS_FAST_QR_2_1S',`GEN_ARGS_FAST_QR_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')





///////////////// EQSP


define(`GEN_ARGS_EQSP__1S',`GEN_ARGS_1S($1)')
define(`GEN_ARGS_EQSP_1_1S',`GEN_ARGS_EQSP_1($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_EQSP_CPX_1_1S',`GEN_ARGS_EQSP_CPX_1($1) GEN_SEP($1) GEN_ARGS_CPX_1S($1)')

define(`GEN_ARGS_EQSP_QUAT_1_1S',`GEN_ARGS_EQSP_QUAT_1($1) GEN_SEP($1) GEN_ARGS_QUAT_1S($1)')

define(`GEN_ARGS_EQSP_1_2S',`GEN_ARGS_EQSP_1($1) GEN_SEP($1) GEN_ARGS_2S($1)')

define(`GEN_ARGS_EQSP_1_3S',`GEN_ARGS_EQSP_1($1) GEN_SEP($1) GEN_ARGS_3S($1)')

define(`GEN_ARGS_EQSP_SBM_1_2S',`GEN_ARGS_EQSP_SBM_1($1) GEN_SEP($1) GEN_ARGS_2S($1)')

define(`GEN_ARGS_EQSP_SBM_CPX_1_2S',`GEN_ARGS_EQSP_SBM_CPX_1($1) GEN_SEP($1) GEN_ARGS_CPX_2S($1)')

define(`GEN_ARGS_EQSP_SBM_QUAT_1_2S',`GEN_ARGS_EQSP_SBM_QUAT_1($1) GEN_SEP($1) GEN_ARGS_QUAT_2S($1)')

define(`GEN_ARGS_EQSP_RC_2',`GEN_ARGS_EQSP_1($1) GEN_SEP($1) GEN_ARGS_EQSP_CPX_SRC1($1)')

define(`GEN_ARGS_EQSP_DBM_SBM',`GEN_ARGS_EQSP_DBM($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM($1)')
define(`GEN_ARGS_EQSP_DBM_2SBM',`GEN_ARGS_EQSP_DBM($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM1($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM2($1)')
define(`GEN_ARGS_EQSP_DBM_1SBM',`GEN_ARGS_EQSP_DBM($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM1($1)')
dnl	Needed???
define(`GEN_ARGS_EQSP_DBM_1SBM_1S',`GEN_ARGS_EQSP_DBM($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM1($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_EQSP_DBM_2SRCS',`GEN_ARGS_EQSP_2SRCS($1) GEN_SEP($1) GEN_ARGS_EQSP_DBM($1)')

define(`GEN_ARGS_EQSP_DBM_1SRC_1S',`GEN_ARGS_EQSP_1SRC_1S($1) GEN_SEP($1) GEN_ARGS_EQSP_DBM($1)')

define(`GEN_ARGS_EQSP_DBM__1S',`GEN_ARGS_1S($1) GEN_SEP($1) GEN_ARGS_EQSP_DBM($1)')

define(`GEN_EQSP_CONV_DEST',`GEN_FAST_CONV_DEST($1,$2) GEN_SEP($1) GEN_EQSP_ARG_INC1($1)')

define(`GEN_ARGS_EQSP_1',`GEN_ARGS_FAST_1($1) GEN_SEP($1) GEN_EQSP_ARG_INC1($1)')

define(`GEN_ARGS_EQSP_CPX_1',`GEN_ARGS_FAST_CPX_1($1) GEN_SEP($1) GEN_EQSP_ARG_INC1($1)')

define(`GEN_ARGS_EQSP_QUAT_1',`GEN_ARGS_FAST_QUAT_1($1) GEN_SEP($1) GEN_EQSP_ARG_INC1($1)')

define(`GEN_ARGS_EQSP_CR_2',`GEN_ARGS_EQSP_CPX_1($1) GEN_SEP($1) GEN_ARGS_EQSP_SRC1($1)')

define(`GEN_ARGS_EQSP_QR_2',`GEN_ARGS_EQSP_QUAT_1($1) GEN_SEP($1) GEN_ARGS_EQSP_SRC1($1)')

define(`GEN_ARGS_EQSP_2SRCS',`GEN_ARGS_EQSP_SRC1($1) GEN_SEP($1) GEN_ARGS_EQSP_SRC2($1)')

define(`GEN_ARGS_EQSP_1SRC_1S',`GEN_ARGS_EQSP_SRC1($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_EQSP_SBM_1',`GEN_ARGS_EQSP_1($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM($1)')

define(`GEN_ARGS_EQSP_SBM_CPX_1',`GEN_ARGS_EQSP_CPX_1($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM($1)')

define(`GEN_ARGS_EQSP_SBM_QUAT_1',`GEN_ARGS_EQSP_QUAT_1($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM($1)')

define(`GEN_ARGS_EQSP_SBM_2',`GEN_ARGS_EQSP_2($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM($1)')

define(`GEN_ARGS_EQSP_SBM_CPX_2',`GEN_ARGS_EQSP_CPX_2($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM($1)')

define(`GEN_ARGS_EQSP_SBM_QUAT_2',`GEN_ARGS_EQSP_QUAT_2($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM($1)')

define(`GEN_ARGS_EQSP_4',`GEN_ARGS_EQSP_3($1) GEN_SEP($1) GEN_ARGS_EQSP_SRC3($1)')

define(`GEN_ARGS_EQSP_5',`GEN_ARGS_EQSP_4($1) GEN_SEP($1) GEN_ARGS_EQSP_SRC4($1)')

define(`GEN_ARGS_EQSP_4_1S',`GEN_ARGS_EQSP_4($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_EQSP_3_2S',`GEN_ARGS_EQSP_3($1) GEN_SEP($1) GEN_ARGS_2S($1)')

define(`GEN_ARGS_EQSP_2_3S',`GEN_ARGS_EQSP_2($1) GEN_SEP($1) GEN_ARGS_3S($1)')

define(`GEN_ARGS_EQSP_SBM_3',`GEN_ARGS_EQSP_3($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM($1)')

define(`GEN_ARGS_EQSP_SBM_CPX_3',`GEN_ARGS_EQSP_CPX_3($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM($1)')

define(`GEN_ARGS_EQSP_SBM_QUAT_3',`GEN_ARGS_EQSP_QUAT_3($1) GEN_SEP($1) GEN_ARGS_EQSP_SBM($1)')

define(`GEN_ARGS_EQSP_2_1S',`GEN_ARGS_EQSP_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_EQSP_SBM_2_1S',`GEN_ARGS_EQSP_SBM_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_EQSP_CPX_2_1S',`GEN_ARGS_EQSP_CPX_2($1) GEN_SEP($1) GEN_ARGS_CPX_1S($1)')

define(`GEN_ARGS_EQSP_SBM_CPX_2_1S',`GEN_ARGS_EQSP_SBM_CPX_2($1) GEN_SEP($1) GEN_ARGS_CPX_1S($1)')

define(`GEN_ARGS_EQSP_SBM_QUAT_2_1S',`GEN_ARGS_EQSP_SBM_QUAT_2($1) GEN_SEP($1) GEN_ARGS_QUAT_1S($1)')

define(`GEN_ARGS_EQSP_CR_2_1S',`GEN_ARGS_EQSP_CR_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_EQSP_CR_2_1S',`GEN_ARGS_EQSP_CR_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_EQSP_QUAT_2_1S',`GEN_ARGS_EQSP_QUAT_2($1) GEN_SEP($1) GEN_ARGS_QUAT_1S($1)')

define(`GEN_ARGS_EQSP_QR_2_1S',`GEN_ARGS_EQSP_QR_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_EQSP_SRC1',`GEN_ARGS_FAST_SRC1($1) GEN_SEP($1) GEN_EQSP_ARG_INC2($1)')
define(`GEN_ARGS_EQSP_BSRC1',`GEN_ARGS_FAST_BSRC1($1) GEN_SEP($1) GEN_EQSP_ARG_INC2($1)')
define(`GEN_ARGS_EQSP_SSRC1',`GEN_ARGS_FAST_SSRC1($1) GEN_SEP($1) GEN_EQSP_ARG_INC2($1)')

define(`GEN_ARGS_EQSP_CPX_SRC1',`GEN_ARGS_FAST_CPX_SRC1($1) GEN_SEP($1) GEN_EQSP_ARG_INC2($1)')

define(`GEN_ARGS_EQSP_QUAT_SRC1',`GEN_ARGS_FAST_QUAT_SRC1($1) GEN_SEP($1) GEN_EQSP_ARG_INC2($1)')

define(`GEN_ARGS_EQSP_SRC2',`GEN_ARGS_FAST_SRC2($1) GEN_SEP($1) GEN_EQSP_ARG_INC3($1)')

define(`GEN_ARGS_EQSP_CPX_SRC2',`GEN_ARGS_FAST_CPX_SRC2($1) GEN_SEP($1) GEN_EQSP_ARG_INC3($1)')

define(`GEN_ARGS_EQSP_QUAT_SRC2',`GEN_ARGS_FAST_QUAT_SRC2($1) GEN_SEP($1) GEN_EQSP_ARG_INC3($1)')

define(`GEN_ARGS_EQSP_SRC3',`GEN_ARGS_FAST_SRC3($1) GEN_SEP($1) GEN_EQSP_ARG_INC4($1)')

define(`GEN_ARGS_EQSP_CPX_SRC3',`GEN_ARGS_FAST_CPX_SRC3($1) GEN_SEP($1) GEN_EQSP_ARG_INC4($1)')

define(`GEN_ARGS_EQSP_QUAT_SRC3',`GEN_ARGS_FAST_QUAT_SRC3($1) GEN_SEP($1) GEN_EQSP_ARG_INC4($1)')

define(`GEN_ARGS_EQSP_SRC4',`GEN_ARGS_FAST_SRC4($1) GEN_SEP($1) GEN_EQSP_ARG_INC5($1)')

define(`GEN_ARGS_EQSP_CPX_SRC4',`GEN_ARGS_FAST_CPX_SRC4($1) GEN_SEP($1) GEN_EQSP_ARG_INC5($1)')

define(`GEN_ARGS_EQSP_QUAT_SRC4',`GEN_ARGS_FAST_QUAT_SRC4($1) GEN_SEP($1) GEN_EQSP_ARG_INC5($1)')

define(`GEN_ARGS_EQSP_CPX_1',`GEN_ARGS_FAST_CPX_1($1) GEN_SEP($1) GEN_EQSP_ARG_INC1($1)')

define(`GEN_ARGS_EQSP_CONV',`GEN_EQSP_CONV_DEST($1,$2) GEN_SEP($1) GEN_ARGS_EQSP_SRC1($1)')

define(`GEN_ARGS_EQSP_2',`GEN_ARGS_EQSP_1($1) GEN_SEP($1) GEN_ARGS_EQSP_SRC1($1)')

define(`GEN_ARGS_EQSP_CPX_2',`GEN_ARGS_EQSP_CPX_1($1) GEN_SEP($1) GEN_ARGS_EQSP_CPX_SRC1($1)')

define(`GEN_ARGS_EQSP_QUAT_2',`GEN_ARGS_EQSP_QUAT_1($1) GEN_SEP($1) GEN_ARGS_EQSP_QUAT_SRC1($1)')

define(`GEN_ARGS_EQSP_3',`GEN_ARGS_EQSP_2($1) GEN_SEP($1) GEN_ARGS_EQSP_SRC2($1)')

define(`GEN_ARGS_EQSP_LUTMAP_B',`GEN_ARGS_EQSP_1($1) GEN_SEP($1) GEN_ARGS_EQSP_BSRC1($1)  GEN_SEP($1) GEN_ARGS_FAST_MAP_B($1)')
define(`GEN_ARGS_EQSP_LUTMAP_S',`GEN_ARGS_EQSP_1($1) GEN_SEP($1) GEN_ARGS_EQSP_SSRC1($1)  GEN_SEP($1) GEN_ARGS_FAST_MAP_S($1)')

define(`GEN_ARGS_EQSP_CPX_3',`GEN_ARGS_EQSP_CPX_2($1) GEN_SEP($1) GEN_ARGS_EQSP_CPX_SRC2($1)')

define(`GEN_ARGS_EQSP_QUAT_3',`GEN_ARGS_EQSP_QUAT_2($1) GEN_SEP($1) GEN_ARGS_EQSP_QUAT_SRC2($1)')

define(`GEN_ARGS_EQSP_CCR_3',`GEN_ARGS_EQSP_CPX_2($1) GEN_SEP($1) GEN_ARGS_EQSP_SRC2($1)')

define(`GEN_ARGS_EQSP_QQR_3',`GEN_ARGS_EQSP_QUAT_2($1) GEN_SEP($1) GEN_ARGS_EQSP_SRC2($1)')



//// SLOW

define(`GEN_ARGS_SLOW__1S',`GEN_ARGS_1S($1)')
define(`GEN_ARGS_SLOW_1_1S',`GEN_ARGS_SLOW_1($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_SLOW_CPX_1_1S',`GEN_ARGS_SLOW_CPX_1($1) GEN_SEP($1) GEN_ARGS_CPX_1S($1)')

define(`GEN_ARGS_SLOW_QUAT_1_1S',`GEN_ARGS_SLOW_QUAT_1($1) GEN_SEP($1) GEN_ARGS_QUAT_1S($1)')



define(`GEN_ARGS_SLOW_1_2S',`GEN_ARGS_SLOW_1($1) GEN_SEP($1) GEN_ARGS_2S($1)')



define(`GEN_ARGS_SLOW_1_3S',`GEN_ARGS_SLOW_1($1) GEN_SEP($1) GEN_ARGS_3S($1)')



define(`GEN_ARGS_SLOW_SBM_1_2S',`GEN_ARGS_SLOW_SBM_1($1) GEN_SEP($1) GEN_ARGS_2S($1)')



define(`GEN_ARGS_SLOW_SBM_CPX_1_2S',`GEN_ARGS_SLOW_SBM_CPX_1($1) GEN_SEP($1) GEN_ARGS_CPX_2S($1)')



define(`GEN_ARGS_SLOW_SBM_QUAT_1_2S',`GEN_ARGS_SLOW_SBM_QUAT_1($1) GEN_SEP($1) GEN_ARGS_QUAT_2S($1)')



define(`GEN_ARGS_SLOW_RC_2',`GEN_ARGS_SLOW_1($1) GEN_SEP($1) GEN_ARGS_SLOW_CPX_SRC1($1)')



define(`GEN_ARGS_SLOW_DBM_SBM',`GEN_ARGS_SLOW_DBM($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM($1)')
define(`GEN_ARGS_SLOW_DBM_2SBM',`GEN_ARGS_SLOW_DBM($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM1($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM2($1)')
define(`GEN_ARGS_SLOW_DBM_1SBM',`GEN_ARGS_SLOW_DBM($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM1($1)')
define(`GEN_ARGS_SLOW_DBM_1SBM_1S',`GEN_ARGS_SLOW_DBM($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM1($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_SLOW_DBM_2SRCS',`GEN_ARGS_SLOW_2SRCS($1) GEN_SEP($1) GEN_ARGS_SLOW_DBM($1)')



define(`GEN_ARGS_SLOW_DBM_1SRC_1S',`GEN_ARGS_SLOW_1SRC_1S($1) GEN_SEP($1) GEN_ARGS_SLOW_DBM($1)')

define(`GEN_ARGS_SLOW_DBM__1S',`GEN_ARGS_1S($1) GEN_SEP($1) GEN_ARGS_SLOW_DBM($1)')


define(`GEN_SLOW_CONV_DEST',`GEN_FAST_CONV_DEST($1,$2) GEN_SEP($1) GEN_SLOW_ARG_INC1($1)')

define(`GEN_SLOW_SIZE',$1`_SLOW_SIZE')
define(`GEN_DBM_GPU_INFO',$1`_DBM_GPU_INFO')

define(`GEN_ARGS_SLOW_1',`GEN_SLOW_SIZE($1) GEN_SEP($1) GEN_ARGS_FAST_1($1) GEN_SEP($1) GEN_SLOW_ARG_INC1($1)')

define(`GEN_ARGS_SLOW_CPX_1',`GEN_SLOW_SIZE($1) GEN_SEP($1) GEN_ARGS_FAST_CPX_1($1) GEN_SEP($1) GEN_SLOW_ARG_INC1($1)')

define(`GEN_ARGS_SLOW_QUAT_1',`GEN_SLOW_SIZE($1) GEN_SEP($1) GEN_ARGS_FAST_QUAT_1($1) GEN_SEP($1) GEN_SLOW_ARG_INC1($1)')


define(`GEN_ARGS_SLOW_CR_2',`GEN_ARGS_SLOW_CPX_1($1) GEN_SEP($1) GEN_ARGS_SLOW_SRC1($1)')



define(`GEN_ARGS_SLOW_QR_2',`GEN_ARGS_SLOW_QUAT_1($1) GEN_SEP($1) GEN_ARGS_SLOW_SRC1($1)')






define(`GEN_ARGS_SLOW_2SRCS',`GEN_SLOW_SIZE($1) GEN_SEP($1) GEN_ARGS_SLOW_SRC1($1) GEN_SEP($1) GEN_ARGS_SLOW_SRC2($1)')

define(`GEN_ARGS_SLOW_1SRC_1S',`GEN_ARGS_SLOW_SRC1($1) GEN_SEP($1) GEN_ARGS_1S($1)')




define(`GEN_ARGS_SLOW_SBM_1',`GEN_ARGS_SLOW_1($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM($1)')

define(`GEN_ARGS_SLOW_SBM_CPX_1',`GEN_ARGS_SLOW_CPX_1($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM($1)')

define(`GEN_ARGS_SLOW_SBM_QUAT_1',`GEN_ARGS_SLOW_QUAT_1($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM($1)')



define(`GEN_ARGS_SLOW_SBM_2',`GEN_ARGS_SLOW_2($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM($1)')

define(`GEN_ARGS_SLOW_SBM_CPX_2',`GEN_ARGS_SLOW_CPX_2($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM($1)')

define(`GEN_ARGS_SLOW_SBM_QUAT_2',`GEN_ARGS_SLOW_QUAT_2($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM($1)')

define(`GEN_ARGS_SLOW_4',`GEN_ARGS_SLOW_3($1) GEN_SEP($1) GEN_ARGS_SLOW_SRC3($1)')

define(`GEN_ARGS_SLOW_5',`GEN_ARGS_SLOW_4($1) GEN_SEP($1) GEN_ARGS_SLOW_SRC4($1)')

define(`GEN_ARGS_SLOW_4_1S',`GEN_ARGS_SLOW_4($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_SLOW_3_2S',`GEN_ARGS_SLOW_3($1) GEN_SEP($1) GEN_ARGS_2S($1)')

define(`GEN_ARGS_SLOW_2_3S',`GEN_ARGS_SLOW_2($1) GEN_SEP($1) GEN_ARGS_3S($1)')

define(`GEN_ARGS_SLOW_SBM_3',`GEN_ARGS_SLOW_3($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM($1)')

define(`GEN_ARGS_SLOW_SBM_CPX_3',`GEN_ARGS_SLOW_CPX_3($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM($1)')

define(`GEN_ARGS_SLOW_SBM_QUAT_3',`GEN_ARGS_SLOW_QUAT_3($1) GEN_SEP($1) GEN_ARGS_SLOW_SBM($1)')

define(`GEN_ARGS_SLOW_2_1S',`GEN_ARGS_SLOW_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_SLOW_SBM_2_1S',`GEN_ARGS_SLOW_SBM_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_SLOW_CPX_2_1S',`GEN_ARGS_SLOW_CPX_2($1) GEN_SEP($1) GEN_ARGS_CPX_1S($1)')

define(`GEN_ARGS_SLOW_SBM_CPX_2_1S',`GEN_ARGS_SLOW_SBM_CPX_2($1) GEN_SEP($1) GEN_ARGS_CPX_1S($1)')

define(`GEN_ARGS_SLOW_SBM_QUAT_2_1S',`GEN_ARGS_SLOW_SBM_QUAT_2($1) GEN_SEP($1) GEN_ARGS_QUAT_1S($1)')

define(`GEN_ARGS_SLOW_CR_2_1S',`GEN_ARGS_SLOW_CR_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_SLOW_CR_2_1S',`GEN_ARGS_SLOW_CR_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')

define(`GEN_ARGS_SLOW_QUAT_2_1S',`GEN_ARGS_SLOW_QUAT_2($1) GEN_SEP($1) GEN_ARGS_QUAT_1S($1)')

define(`GEN_ARGS_SLOW_QR_2_1S',`GEN_ARGS_SLOW_QR_2($1) GEN_SEP($1) GEN_ARGS_1S($1)')




// SRC1


define(`GEN_ARGS_SLOW_SRC1',`GEN_ARGS_FAST_SRC1($1) GEN_SEP($1) GEN_SLOW_ARG_INC2($1)')
define(`GEN_ARGS_SLOW_BSRC1',`GEN_ARGS_FAST_BSRC1($1) GEN_SEP($1) GEN_SLOW_ARG_INC2($1)')
define(`GEN_ARGS_SLOW_SSRC1',`GEN_ARGS_FAST_SSRC1($1) GEN_SEP($1) GEN_SLOW_ARG_INC2($1)')



define(`GEN_ARGS_SLOW_CPX_SRC1',`GEN_ARGS_FAST_CPX_SRC1($1) GEN_SEP($1) GEN_SLOW_ARG_INC2($1)')



define(`GEN_ARGS_SLOW_QUAT_SRC1',`GEN_ARGS_FAST_QUAT_SRC1($1) GEN_SEP($1) GEN_SLOW_ARG_INC2($1)')


// SRC2

define(`GEN_ARGS_SLOW_SRC2',`GEN_ARGS_FAST_SRC2($1) GEN_SEP($1) GEN_SLOW_ARG_INC3($1)')


define(`GEN_ARGS_SLOW_CPX_SRC2',`GEN_ARGS_FAST_CPX_SRC2($1) GEN_SEP($1) GEN_SLOW_ARG_INC3($1)')


define(`GEN_ARGS_SLOW_QUAT_SRC2',`GEN_ARGS_FAST_QUAT_SRC2($1) GEN_SEP($1) GEN_SLOW_ARG_INC3($1)')


// SRC3

define(`GEN_ARGS_SLOW_SRC3',`GEN_ARGS_FAST_SRC3($1) GEN_SEP($1) GEN_SLOW_ARG_INC4($1)')


define(`GEN_ARGS_SLOW_CPX_SRC3',`GEN_ARGS_FAST_CPX_SRC3($1) GEN_SEP($1) GEN_SLOW_ARG_INC4($1)')


define(`GEN_ARGS_SLOW_QUAT_SRC3',`GEN_ARGS_FAST_QUAT_SRC3($1) GEN_SEP($1) GEN_SLOW_ARG_INC4($1)')



// SRC4

define(`GEN_ARGS_SLOW_SRC4',`GEN_ARGS_FAST_SRC4($1) GEN_SEP($1) GEN_SLOW_ARG_INC5($1)')



define(`GEN_ARGS_SLOW_CPX_SRC4',`GEN_ARGS_FAST_CPX_SRC4($1) GEN_SEP($1) GEN_SLOW_ARG_INC5($1)')



define(`GEN_ARGS_SLOW_QUAT_SRC4',`GEN_ARGS_FAST_QUAT_SRC4($1) GEN_SEP($1) GEN_SLOW_ARG_INC5($1)')


/////


define(`GEN_ARGS_SLOW_CONV',`GEN_SLOW_SIZE($1) GEN_SEP($1) GEN_SLOW_CONV_DEST($1,$2) GEN_SEP($1) GEN_ARGS_SLOW_SRC1($1)')

define(`GEN_ARGS_SLOW_2',`GEN_ARGS_SLOW_1($1) GEN_SEP($1) GEN_ARGS_SLOW_SRC1($1)')
define(`GEN_ARGS_SLOW_CPX_2',`GEN_ARGS_SLOW_CPX_1($1) GEN_SEP($1) GEN_ARGS_SLOW_CPX_SRC1($1)')

define(`GEN_ARGS_SLOW_QUAT_2',`GEN_ARGS_SLOW_QUAT_1($1) GEN_SEP($1) GEN_ARGS_SLOW_QUAT_SRC1($1)')

define(`GEN_ARGS_SLOW_3',`GEN_ARGS_SLOW_2($1) GEN_SEP($1) GEN_ARGS_SLOW_SRC2($1)')

define(`GEN_ARGS_SLOW_LUTMAP_B',`GEN_ARGS_SLOW_1($1) GEN_SEP($1) GEN_ARGS_SLOW_BSRC1($1) GEN_SEP($1) GEN_ARGS_FAST_MAP_B($1)')
define(`GEN_ARGS_SLOW_LUTMAP_S',`GEN_ARGS_SLOW_1($1) GEN_SEP($1) GEN_ARGS_SLOW_SSRC1($1) GEN_SEP($1) GEN_ARGS_FAST_MAP_S($1)')

define(`GEN_ARGS_SLOW_CPX_3',`GEN_ARGS_SLOW_CPX_2($1) GEN_SEP($1) GEN_ARGS_SLOW_CPX_SRC2($1)')

define(`GEN_ARGS_SLOW_CCR_3',`GEN_ARGS_SLOW_CPX_2($1) GEN_SEP($1) GEN_ARGS_SLOW_SRC2($1)')

define(`GEN_ARGS_SLOW_QUAT_3',`GEN_ARGS_SLOW_QUAT_2($1) GEN_SEP($1) GEN_ARGS_SLOW_QUAT_SRC2($1)')

define(`GEN_ARGS_SLOW_QQR_3',`GEN_ARGS_SLOW_QUAT_2($1) GEN_SEP($1) GEN_ARGS_SLOW_SRC2($1)')

// Now the len versions

define(`GEN_ADD_FAST_LEN',`GEN_SEP($1) GEN_FAST_ARG_LEN($1)')
define(`GEN_ADD_EQSP_LEN',`GEN_SEP($1) GEN_EQSP_ARG_LEN($1)')
dnl	define GEN_ADD_SLOW_LEN',`GEN_SEP($1) GEN_SLOW_ARG_LEN($1)')
define(`GEN_ADD_SLOW_LEN',`') dnl// nop - we now use szarr (SLOW_SIZE)
// FLEN


define(`GEN_ARGS_FLEN_SBM',`GEN_ARGS_FAST_SBM($1)')
define(`GEN_ARGS_FLEN__1S',`GEN_ARGS_FAST__1S($1)')
define(`GEN_ARGS_FLEN_1_1S',`GEN_ARGS_FAST_1_1S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_CPX_1_1S',`GEN_ARGS_FAST_CPX_1_1S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_QUAT_1_1S',`GEN_ARGS_FAST_QUAT_1_1S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_1_2S',`GEN_ARGS_FAST_1_2S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_1_3S',`GEN_ARGS_FAST_1_3S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_1_2S',`GEN_ARGS_FAST_SBM_1_2S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_CPX_1_2S',`GEN_ARGS_FAST_SBM_CPX_1_2S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_QUAT_1_2S',`GEN_ARGS_FAST_SBM_QUAT_1_2S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_RC_2',`GEN_ARGS_FAST_RC_2($1) GEN_ADD_FAST_LEN($1)')

dnl	For bitmaps with a length parameter, we need to be able to specify the bits in the word
dnl	that will be affected...
define(`GEN_ARGS_FLEN_DBM_SBM',`GEN_ARGS_ELEN_DBM_SBM($1)')
define(`GEN_ARGS_FLEN_DBM_2SBM',`GEN_ARGS_ELEN_DBM_2SBM($1)')
define(`GEN_ARGS_FLEN_DBM_1SBM',`GEN_ARGS_ELEN_DBM_1SBM($1)')
define(`GEN_ARGS_FLEN_DBM_1SBM_1S',`GEN_ARGS_ELEN_DBM_1SBM_1S($1)')

//prefix##_EQSP_DBM_SBM

dnl //define(`GEN_ARGS_FLEN_DBM_SBM',`GEN_DBM_GPU_INFO($1)
dnl //						GEN_SEP($1)
dnl //						$1`_EQSP_DBM_SBM' )

define(`GEN_ARGS_FLEN_DBM_2SRCS',`GEN_ARGS_FLEN_DBM($1) GEN_SEP($1) GEN_ARGS_FLEN_2SRCS($1)')

define(`GEN_ARGS_SLOW_DBM_1SRC_1S',`GEN_ARGS_SLOW_1SRC_1S($1) GEN_SEP($1) GEN_ARGS_SLOW_DBM($1)')
define(`GEN_ARGS_FLEN_DBM_1SRC_1S',`GEN_ARGS_FLEN_DBM($1) GEN_SEP($1) GEN_ARGS_FLEN_1SRC_1S($1)')
define(`GEN_ARGS_FLEN_DBM__1S',`GEN_ARGS_FLEN_DBM($1) GEN_SEP($1) GEN_ARGS_FLEN__1S($1)')

dnl // dfn						$1`_EQSP_DBM__1S'

dnl //define(`GEN_ARGS_FLEN_DBM__1S',`GEN_DBM_GPU_INFO($1)
dnl //						GEN_SEP($1)
dnl //						$1`_EQSP_DBM__1S')

define(`GEN_ARGS_FLEN_CONV',`GEN_ARGS_FAST_CONV($1,$2) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_2',`GEN_ARGS_FAST_2($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_CPX_2',`GEN_ARGS_FAST_CPX_2($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_CR_2',`GEN_ARGS_FAST_CR_2($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_QR_2',`GEN_ARGS_FAST_QR_2($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_2SRCS',`GEN_ARGS_FAST_2SRCS($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_1SRC_1S',`GEN_ARGS_FAST_1SRC_1S($1) GEN_ADD_FAST_LEN($1)')

dnl define(`GEN_FLEN_ARG_DBM_2SRCS',`GEN_FAST_ARG_DBM_2SRCS($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_1',`GEN_ARGS_FAST_SBM_1($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_CPX_1',`GEN_ARGS_FAST_SBM_CPX_1($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_QUAT_1',`GEN_ARGS_FAST_SBM_QUAT_1($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_2',`GEN_ARGS_FAST_SBM_2($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_CPX_2',`GEN_ARGS_FAST_SBM_CPX_2($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_QUAT_2',`GEN_ARGS_FAST_SBM_QUAT_2($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_QUAT_2',`GEN_ARGS_FAST_QUAT_2($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_QR_2',`GEN_ARGS_FAST_QR_2($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_4',`GEN_ARGS_FAST_4($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_5',`GEN_ARGS_FAST_5($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_4_1S',`GEN_ARGS_FAST_4_1S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_3_2S',`GEN_ARGS_FAST_3_2S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_2_3S',`GEN_ARGS_FAST_2_3S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_3',`GEN_ARGS_FAST_3($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_LUTMAP_B',`GEN_ARGS_FAST_LUTMAP_B($1) GEN_ADD_FAST_LEN($1)')
define(`GEN_ARGS_FLEN_LUTMAP_S',`GEN_ARGS_FAST_LUTMAP_S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_CPX_3',`GEN_ARGS_FAST_CPX_3($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_CCR_3',`GEN_ARGS_FAST_CCR_3($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_QUAT_3',`GEN_ARGS_FAST_QUAT_3($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_QQR_3',`GEN_ARGS_FAST_QQR_3($1) GEN_ADD_FAST_LEN($1)')



define(`GEN_ARGS_FLEN_SBM_3',`GEN_ARGS_FAST_SBM_3($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_CPX_3',`GEN_ARGS_FAST_SBM_CPX_3($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_QUAT_3',`GEN_ARGS_FAST_SBM_QUAT_3($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_2_1S',`GEN_ARGS_FAST_2_1S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_2_1S',`GEN_ARGS_FAST_SBM_2_1S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_CPX_2_1S',`GEN_ARGS_FAST_CPX_2_1S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_CPX_2_1S',`GEN_ARGS_FAST_SBM_CPX_2_1S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_SBM_QUAT_2_1S',`GEN_ARGS_FAST_SBM_QUAT_2_1S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_QUAT_2_1S',`GEN_ARGS_FAST_QUAT_2_1S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_CR_2_1S',`GEN_ARGS_FAST_CR_2_1S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_QUAT_2_1S',`GEN_ARGS_FAST_QUAT_2_1S($1) GEN_ADD_FAST_LEN($1)')

define(`GEN_ARGS_FLEN_QR_2_1S',`GEN_ARGS_FAST_QR_2_1S($1) GEN_ADD_FAST_LEN($1)')




// ELEN


define(`GEN_ARGS_ELEN__1S',`GEN_ARGS_EQSP__1S($1)')
define(`GEN_ARGS_ELEN_1_1S',`GEN_ARGS_EQSP_1_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_CPX_1_1S',`GEN_ARGS_EQSP_CPX_1_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_QUAT_1_1S',`GEN_ARGS_EQSP_QUAT_1_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_1_2S',`GEN_ARGS_EQSP_1_2S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_1_3S',`GEN_ARGS_EQSP_1_3S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM',`GEN_ARGS_EQSP_SBM($1)')
define(`GEN_ARGS_ELEN_SBM_1_2S',`GEN_ARGS_EQSP_SBM_1_2S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM_CPX_1_2S',`GEN_ARGS_EQSP_SBM_CPX_1_2S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM_QUAT_1_2S',`GEN_ARGS_EQSP_SBM_QUAT_1_2S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_RC_2',`GEN_ARGS_EQSP_RC_2($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_DBM_SBM',`GEN_ARGS_EQSP_DBM_SBM($1) GEN_ADD_EQSP_LEN($1)')
define(`GEN_ARGS_ELEN_DBM_2SBM',`GEN_ARGS_EQSP_DBM_2SBM($1) GEN_ADD_EQSP_LEN($1)')
define(`GEN_ARGS_ELEN_DBM_1SBM',`GEN_ARGS_EQSP_DBM_1SBM($1) GEN_ADD_EQSP_LEN($1)')
define(`GEN_ARGS_ELEN_DBM_1SBM_1S',`GEN_ARGS_EQSP_DBM_1SBM_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_DBM_2SRCS',`GEN_ARGS_EQSP_DBM_2SRCS($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_DBM_1SRC_1S',`GEN_ARGS_EQSP_DBM_1SRC_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_DBM__1S',`GEN_ARGS_EQSP_DBM__1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_CONV',`GEN_ARGS_EQSP_CONV($1,$2) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_2',`GEN_ARGS_EQSP_2($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_CPX_2',`GEN_ARGS_EQSP_CPX_2($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_CR_2',`GEN_ARGS_EQSP_CR_2($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_QR_2',`GEN_ARGS_EQSP_QR_2($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_2SRCS',`GEN_ARGS_EQSP_2SRCS($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_1SRC_1S',`GEN_ARGS_EQSP_1SRC_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ELEN_ARG_DBM_2SRCS',`GEN_EQSP_ARG_DBM_2SRCS($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM_1',`GEN_ARGS_EQSP_SBM_1($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM_CPX_1',`GEN_ARGS_EQSP_SBM_CPX_1($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM_QUAT_1',`GEN_ARGS_EQSP_SBM_QUAT_1($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM_2',`GEN_ARGS_EQSP_SBM_2($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM_CPX_2',`GEN_ARGS_EQSP_SBM_CPX_2($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM_QUAT_2',`GEN_ARGS_EQSP_SBM_QUAT_2($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_QUAT_2',`GEN_ARGS_EQSP_QUAT_2($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_QR_2',`GEN_ARGS_EQSP_QR_2($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_4',`GEN_ARGS_EQSP_4($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_5',`GEN_ARGS_EQSP_5($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_4_1S',`GEN_ARGS_EQSP_4_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_3_2S',`GEN_ARGS_EQSP_3_2S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_2_3S',`GEN_ARGS_EQSP_2_3S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_3',`GEN_ARGS_EQSP_3($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_LUTMAP_B',`GEN_ARGS_EQSP_LUTMAP_B($1) GEN_ADD_EQSP_LEN($1)')
define(`GEN_ARGS_ELEN_LUTMAP_S',`GEN_ARGS_EQSP_LUTMAP_S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_CPX_3',`GEN_ARGS_EQSP_CPX_3($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_CCR_3',`GEN_ARGS_EQSP_CCR_3($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_QUAT_3',`GEN_ARGS_EQSP_QUAT_3($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_QQR_3',`GEN_ARGS_EQSP_QQR_3($1) GEN_ADD_EQSP_LEN($1)')



define(`GEN_ARGS_ELEN_SBM_3',`GEN_ARGS_EQSP_SBM_3($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM_CPX_3',`GEN_ARGS_EQSP_SBM_CPX_3($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM_QUAT_3',`GEN_ARGS_EQSP_SBM_QUAT_3($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_2_1S',`GEN_ARGS_EQSP_2_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM_2_1S',`GEN_ARGS_EQSP_SBM_2_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_CPX_2_1S',`GEN_ARGS_EQSP_CPX_2_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM_CPX_2_1S',`GEN_ARGS_EQSP_SBM_CPX_2_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_SBM_QUAT_2_1S',`GEN_ARGS_EQSP_SBM_QUAT_2_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_QUAT_2_1S',`GEN_ARGS_EQSP_QUAT_2_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_CR_2_1S',`GEN_ARGS_EQSP_CR_2_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_QUAT_2_1S',`GEN_ARGS_EQSP_QUAT_2_1S($1) GEN_ADD_EQSP_LEN($1)')

define(`GEN_ARGS_ELEN_QR_2_1S',`GEN_ARGS_EQSP_QR_2_1S($1) GEN_ADD_EQSP_LEN($1)')





// SLEN



define(`GEN_ARGS_SLEN__1S',`GEN_ARGS_SLOW__1S($1)')
define(`GEN_ARGS_SLEN_1_1S',`GEN_ARGS_SLOW_1_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_CPX_1_1S',`GEN_ARGS_SLOW_CPX_1_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_QUAT_1_1S',`GEN_ARGS_SLOW_QUAT_1_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_1_2S',`GEN_ARGS_SLOW_1_2S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_1_3S',`GEN_ARGS_SLOW_1_3S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM_1_2S',`GEN_ARGS_SLOW_SBM_1_2S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM_CPX_1_2S',`GEN_ARGS_SLOW_SBM_CPX_1_2S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM_QUAT_1_2S',`GEN_ARGS_SLOW_SBM_QUAT_1_2S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_RC_2',`GEN_ARGS_SLOW_RC_2($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_DBM_SBM',`GEN_ARGS_SLOW_DBM_SBM($1) GEN_ADD_SLOW_LEN($1)')
define(`GEN_ARGS_SLEN_DBM_2SBM',`GEN_ARGS_SLOW_DBM_2SBM($1) GEN_ADD_SLOW_LEN($1)')
define(`GEN_ARGS_SLEN_DBM_1SBM',`GEN_ARGS_SLOW_DBM_1SBM($1) GEN_ADD_SLOW_LEN($1)')
define(`GEN_ARGS_SLEN_DBM_1SBM_1S',`GEN_ARGS_SLOW_DBM_1SBM_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_DBM_2SRCS',`GEN_ARGS_SLOW_DBM_2SRCS($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_DBM_1SRC_1S',`GEN_ARGS_SLOW_DBM_1SRC_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_DBM__1S',`GEN_ARGS_SLOW_DBM__1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_CONV',`GEN_ARGS_SLOW_CONV($1,$2) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_2',`GEN_ARGS_SLOW_2($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_CPX_2',`GEN_ARGS_SLOW_CPX_2($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_CR_2',`GEN_ARGS_SLOW_CR_2($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_QR_2',`GEN_ARGS_SLOW_QR_2($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_2SRCS',`GEN_ARGS_SLOW_2SRCS($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_1SRC_1S',`GEN_ARGS_SLOW_1SRC_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_SLEN_ARG_DBM_2SRCS',`GEN_SLOW_ARG_DBM_2SRCS($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM',`GEN_ARGS_SLOW_SBM($1)')
define(`GEN_ARGS_SLEN_SBM_1',`GEN_ARGS_SLOW_SBM_1($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM_CPX_1',`GEN_ARGS_SLOW_SBM_CPX_1($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM_QUAT_1',`GEN_ARGS_SLOW_SBM_QUAT_1($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM_2',`GEN_ARGS_SLOW_SBM_2($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM_CPX_2',`GEN_ARGS_SLOW_SBM_CPX_2($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM_QUAT_2',`GEN_ARGS_SLOW_SBM_QUAT_2($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_QUAT_2',`GEN_ARGS_SLOW_QUAT_2($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_QR_2',`GEN_ARGS_SLOW_QR_2($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_4',`GEN_ARGS_SLOW_4($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_5',`GEN_ARGS_SLOW_5($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_4_1S',`GEN_ARGS_SLOW_4_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_3_2S',`GEN_ARGS_SLOW_3_2S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_2_3S',`GEN_ARGS_SLOW_2_3S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_3',`GEN_ARGS_SLOW_3($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_LUTMAP_B',`GEN_ARGS_SLOW_LUTMAP_B($1) GEN_ADD_SLOW_LEN($1)')
define(`GEN_ARGS_SLEN_LUTMAP_S',`GEN_ARGS_SLOW_LUTMAP_S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_CPX_3',`GEN_ARGS_SLOW_CPX_3($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_CCR_3',`GEN_ARGS_SLOW_CCR_3($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_QUAT_3',`GEN_ARGS_SLOW_QUAT_3($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_QQR_3',`GEN_ARGS_SLOW_QQR_3($1) GEN_ADD_SLOW_LEN($1)')



define(`GEN_ARGS_SLEN_SBM_3',`GEN_ARGS_SLOW_SBM_3($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM_CPX_3',`GEN_ARGS_SLOW_SBM_CPX_3($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM_QUAT_3',`GEN_ARGS_SLOW_SBM_QUAT_3($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_2_1S',`GEN_ARGS_SLOW_2_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM_2_1S',`GEN_ARGS_SLOW_SBM_2_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_CPX_2_1S',`GEN_ARGS_SLOW_CPX_2_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM_CPX_2_1S',`GEN_ARGS_SLOW_SBM_CPX_2_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_SBM_QUAT_2_1S',`GEN_ARGS_SLOW_SBM_QUAT_2_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_QUAT_2_1S',`GEN_ARGS_SLOW_QUAT_2_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_CR_2_1S',`GEN_ARGS_SLOW_CR_2_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_QUAT_2_1S',`GEN_ARGS_SLOW_QUAT_2_1S($1) GEN_ADD_SLOW_LEN($1)')

define(`GEN_ARGS_SLEN_QR_2_1S',`GEN_ARGS_SLOW_QR_2_1S($1) GEN_ADD_SLOW_LEN($1)')




suppress_no

