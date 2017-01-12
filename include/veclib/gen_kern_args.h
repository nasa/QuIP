// These are used in the declarations of the kernels,
// and the function calls of the kernels.

#ifndef _GEN_KERN_ARGS_H_
#define _GEN_KERN_ARGS_H_

///  map to elsewhere

#define GEN_ARGS_1S(prefix)		prefix##_1S
#define GEN_ARGS_2S(prefix)		prefix##_2S
#define GEN_ARGS_3S(prefix)		prefix##_3S

#define GEN_ARGS_CPX_1S(prefix)		prefix##_CPX_1S
#define GEN_ARGS_CPX_2S(prefix)		prefix##_CPX_2S
#define GEN_ARGS_CPX_3S(prefix)		prefix##_CPX_3S

#define GEN_ARGS_QUAT_1S(prefix)	prefix##_QUAT_1S
#define GEN_ARGS_QUAT_2S(prefix)	prefix##_QUAT_2S
#define GEN_ARGS_QUAT_3S(prefix)	prefix##_QUAT_3S



#define GEN_SEP(prefix)			prefix##_SEPARATOR
#define DECLARE_KERN_ARGS_SEPARATOR	,
#define KERN_ARGS_SEPARATOR		,
#define SET_KERNEL_ARGS_SEPARATOR

#define GEN_FAST_ARG_LEN(prefix)	prefix##_FAST_LEN
#define GEN_EQSP_ARG_LEN(prefix)	prefix##_EQSP_LEN
#define GEN_SLOW_ARG_LEN(prefix)	prefix##_SLOW_LEN

#define GEN_ARGS_EQSP_SBM(prefix)	prefix##_EQSP_SBM
#define GEN_ARGS_SLOW_SBM(prefix)	prefix##_SLOW_SBM

#define GEN_ARGS_EQSP_DBM(prefix)					\
					GEN_SLOW_DBM_GPU_INFO(prefix)	\
					GEN_SEP(prefix)			\
					prefix##_EQSP_DBM

// BUG?  need to make sure that GPU_INFO gets inserted everywhere that's necessary!?

#define GEN_ARGS_SLOW_DBM(prefix)					\
					GEN_SLOW_SIZE(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_DBM_BASIC(prefix)

#define GEN_ARGS_SLOW_DBM_BASIC(prefix)					\
					GEN_SLOW_DBM_GPU_INFO(prefix)	\
					GEN_SEP(prefix)			\
					prefix##_SLOW_DBM	// SLOW_SIZE

//#define GEN_ARGS_NOCC_SETUP(prefix)	GEN_SLOW_SIZE(prefix)		\
//					GEN_SEP(prefix)			\
//					prefix##_NOCC_SETUP

////////// FAST

// OS_ARG is the offset arg...  that is how we do indexing with opencl

#ifdef BUILD_FOR_OPENCL
#define OS_ARG(p,wh)	GEN_SEP(p) p##_##wh##_OFFSET
#else // ! BUILD_FOR_OPENCL
#define OS_ARG(p,wh)
#endif // ! BUILD_FOR_OPENCL

#define GEN_FAST_CONV_DEST(p,to_type)	p##_FAST_CONV_DEST(to_type)	OS_ARG(p,DEST)

#define GEN_ARGS_FAST_1(p)		p##_FAST_1		OS_ARG(p,DEST)
#define GEN_ARGS_FAST_CPX_1(p)		p##_FAST_CPX_1		OS_ARG(p,DEST)
#define GEN_ARGS_FAST_QUAT_1(p)		p##_FAST_QUAT_1		OS_ARG(p,DEST)

#define GEN_ARGS_FAST_SRC1(p)		p##_FAST_SRC1		OS_ARG(p,SRC1)
#define GEN_ARGS_FAST_CPX_SRC1(p)	p##_FAST_CPX_SRC1	OS_ARG(p,SRC1)
#define GEN_ARGS_FAST_QUAT_SRC1(p)	p##_FAST_QUAT_SRC1	OS_ARG(p,SRC1)

#define GEN_ARGS_FAST_SRC2(p)		p##_FAST_SRC2		OS_ARG(p,SRC2)
#define GEN_ARGS_FAST_CPX_SRC2(p)	p##_FAST_CPX_SRC2	OS_ARG(p,SRC2)
#define GEN_ARGS_FAST_QUAT_SRC2(p)	p##_FAST_QUAT_SRC2	OS_ARG(p,SRC2)

#define GEN_ARGS_FAST_SRC3(p)		p##_FAST_SRC3		OS_ARG(p,SRC3)
#define GEN_ARGS_FAST_CPX_SRC3(p)	p##_FAST_CPX_SRC3	OS_ARG(p,SRC3)
#define GEN_ARGS_FAST_QUAT_SRC3(p)	p##_FAST_QUAT_SRC3	OS_ARG(p,SRC3)

#define GEN_ARGS_FAST_SRC4(p)		p##_FAST_SRC4		OS_ARG(p,SRC4)
#define GEN_ARGS_FAST_CPX_SRC4(p)	p##_FAST_CPX_SRC4	OS_ARG(p,SRC4)
#define GEN_ARGS_FAST_QUAT_SRC4(p)	p##_FAST_QUAT_SRC4	OS_ARG(p,SRC4)

#define GEN_ARGS_FAST_SBM(p)		p##_FAST_SBM		OS_ARG(p,SBM)
#define GEN_ARGS_FAST_DBM(p)		p##_FAST_DBM		OS_ARG(p,DBM)

////// FLEN

#define GEN_ARGS_FLEN_1(prefix)		GEN_ARGS_FAST_1(prefix)
#define GEN_ARGS_FLEN_SRC1(prefix)	GEN_ARGS_FAST_SRC1(prefix)
#define GEN_ARGS_FLEN_SRC2(prefix)	GEN_ARGS_FAST_SRC2(prefix)
#define GEN_ARGS_FLEN_SRC3(prefix)	GEN_ARGS_FAST_SRC3(prefix)
#define GEN_ARGS_FLEN_SRC4(prefix)	GEN_ARGS_FAST_SRC4(prefix)

///// EQSP

#define GEN_EQSP_ARG_INC1(prefix)	prefix##_EQSP_INC1
#define GEN_EQSP_ARG_INC2(prefix)	prefix##_EQSP_INC2
#define GEN_EQSP_ARG_INC3(prefix)	prefix##_EQSP_INC3
#define GEN_EQSP_ARG_INC4(prefix)	prefix##_EQSP_INC4
#define GEN_EQSP_ARG_INC5(prefix)	prefix##_EQSP_INC5

#define GEN_SLOW_ARG_INC1(prefix)	prefix##_SLOW_INC1
#define GEN_SLOW_ARG_INC2(prefix)	prefix##_SLOW_INC2
#define GEN_SLOW_ARG_INC3(prefix)	prefix##_SLOW_INC3
#define GEN_SLOW_ARG_INC4(prefix)	prefix##_SLOW_INC4
#define GEN_SLOW_ARG_INC5(prefix)	prefix##_SLOW_INC5

//////////  Everything after this point recombines stuff in this file

////////// FAST

#define GEN_ARGS_FAST_1S_(prefix)	GEN_ARGS_1S(prefix)

#define GEN_ARGS_FAST_1S_1(prefix)	GEN_ARGS_FAST_1(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_FAST_CPX_1S_1(prefix)	GEN_ARGS_FAST_CPX_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_CPX_1S(prefix)

#define GEN_ARGS_FAST_QUAT_1S_1(prefix)	GEN_ARGS_FAST_QUAT_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_QUAT_1S(prefix)

#define GEN_ARGS_FAST_2S_1(prefix)	GEN_ARGS_FAST_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_2S(prefix)

#define GEN_ARGS_FAST_3S_1(prefix)	GEN_ARGS_FAST_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_3S(prefix)

#define GEN_ARGS_FAST_SBM_2S_1(prefix)	GEN_ARGS_FAST_SBM_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_2S(prefix)

#define GEN_ARGS_FAST_SBM_CPX_2S_1(prefix)	GEN_ARGS_FAST_SBM_CPX_1(prefix)	\
						GEN_SEP(prefix)			\
						GEN_ARGS_CPX_2S(prefix)

#define GEN_ARGS_FAST_SBM_QUAT_2S_1(prefix)	GEN_ARGS_FAST_SBM_QUAT_1(prefix)	\
						GEN_SEP(prefix)			\
						GEN_ARGS_QUAT_2S(prefix)

#define GEN_ARGS_FAST_RC_2(prefix)	GEN_ARGS_FAST_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_FAST_CPX_SRC1(prefix)

#define GEN_ARGS_FAST_DBM_SBM(prefix)		GEN_ARGS_FAST_DBM(prefix)	\
						GEN_SEP(prefix)			\
						GEN_ARGS_FAST_SBM(prefix)

#define GEN_ARGS_FAST_DBM_2SRCS(prefix)		GEN_ARGS_FAST_2SRCS(prefix)	\
						GEN_SEP(prefix)			\
						GEN_ARGS_FAST_DBM(prefix)

#define GEN_ARGS_FAST_DBM_1S_1SRC(prefix)	GEN_ARGS_FAST_1S_1SRC(prefix)	\
						GEN_SEP(prefix)			\
						GEN_ARGS_FAST_DBM(prefix)

#define GEN_ARGS_FAST_DBM_1S_(prefix)		GEN_ARGS_1S(prefix)		\
						GEN_SEP(prefix)			\
						GEN_ARGS_FAST_DBM(prefix)

#define GEN_ARGS_FAST_CONV(prefix,to_type)	GEN_FAST_CONV_DEST(prefix,to_type)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SRC1(prefix)

#define GEN_ARGS_FAST_2(prefix)		GEN_ARGS_FAST_1(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SRC1(prefix)

#define GEN_ARGS_FAST_CPX_2(prefix)	GEN_ARGS_FAST_CPX_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_CPX_SRC1(prefix)

#define GEN_ARGS_FAST_CR_2(prefix)	GEN_ARGS_FAST_CPX_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SRC1(prefix)

#define GEN_ARGS_FAST_QR_2(prefix)	GEN_ARGS_FAST_QUAT_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SRC1(prefix)

#define GEN_ARGS_FAST_2SRCS(prefix)	GEN_ARGS_FAST_SRC1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SRC2(prefix)

#define GEN_ARGS_FAST_1S_1SRC(prefix)	GEN_ARGS_FAST_SRC1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_FAST_ARG_DBM_2SRCS(prefix)	GEN_ARGS_FAST_2SRCS(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_DBM(prefix)

#define GEN_ARGS_FAST_SBM_1(prefix)	GEN_ARGS_FAST_1(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SBM(prefix)

#define GEN_ARGS_FAST_SBM_CPX_1(prefix)	GEN_ARGS_FAST_CPX_1(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SBM(prefix)

#define GEN_ARGS_FAST_SBM_QUAT_1(prefix)	GEN_ARGS_FAST_QUAT_1(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SBM(prefix)

#define GEN_ARGS_FAST_SBM_2(prefix)	GEN_ARGS_FAST_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SBM(prefix)

#define GEN_ARGS_FAST_SBM_CPX_2(prefix)	GEN_ARGS_FAST_CPX_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SBM(prefix)

#define GEN_ARGS_FAST_SBM_QUAT_2(prefix)	GEN_ARGS_FAST_QUAT_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SBM(prefix)

#define GEN_ARGS_FAST_QUAT_2(prefix)	GEN_ARGS_FAST_QUAT_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_QUAT_SRC1(prefix)

#define GEN_ARGS_FAST_QR_2(prefix)	GEN_ARGS_FAST_QUAT_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SRC1(prefix)

#define GEN_ARGS_FAST_4(prefix)		GEN_ARGS_FAST_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SRC3(prefix)

#define GEN_ARGS_FAST_5(prefix)		GEN_ARGS_FAST_4(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SRC4(prefix)

#define GEN_ARGS_FAST_1S_4(prefix)	GEN_ARGS_FAST_4(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_FAST_2S_3(prefix)	GEN_ARGS_FAST_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_2S(prefix)

#define GEN_ARGS_FAST_3S_2(prefix)	GEN_ARGS_FAST_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_3S(prefix)

#define GEN_ARGS_FAST_3(prefix)		GEN_ARGS_FAST_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SRC2(prefix)

#define GEN_ARGS_FAST_CPX_3(prefix)	GEN_ARGS_FAST_CPX_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_CPX_SRC2(prefix)

#define GEN_ARGS_FAST_CCR_3(prefix)	GEN_ARGS_FAST_CPX_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SRC2(prefix)

#define GEN_ARGS_FAST_QUAT_3(prefix)	GEN_ARGS_FAST_QUAT_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_QUAT_SRC2(prefix)

#define GEN_ARGS_FAST_QQR_3(prefix)	GEN_ARGS_FAST_QUAT_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SRC2(prefix)



#define GEN_ARGS_FAST_SBM_3(prefix)	GEN_ARGS_FAST_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SBM(prefix)

#define GEN_ARGS_FAST_SBM_CPX_3(prefix)	GEN_ARGS_FAST_CPX_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SBM(prefix)

#define GEN_ARGS_FAST_SBM_QUAT_3(prefix)	GEN_ARGS_FAST_QUAT_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_FAST_SBM(prefix)

#define GEN_ARGS_FAST_1S_2(prefix)	GEN_ARGS_FAST_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_FAST_SBM_1S_2(prefix)	GEN_ARGS_FAST_SBM_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_FAST_CPX_1S_2(prefix)	GEN_ARGS_FAST_CPX_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_CPX_1S(prefix)

#define GEN_ARGS_FAST_SBM_CPX_1S_2(prefix)	GEN_ARGS_FAST_SBM_CPX_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_CPX_1S(prefix)

#define GEN_ARGS_FAST_SBM_QUAT_1S_2(prefix)	GEN_ARGS_FAST_SBM_QUAT_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_QUAT_1S(prefix)

#define GEN_ARGS_FAST_QUAT_1S_2(prefix)	GEN_ARGS_FAST_QUAT_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_QUAT_1S(prefix)

#define GEN_ARGS_FAST_CR_1S_2(prefix)	GEN_ARGS_FAST_CR_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_FAST_QUAT_1S_2(prefix)	GEN_ARGS_FAST_QUAT_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_QUAT_1S(prefix)

#define GEN_ARGS_FAST_QR_1S_2(prefix)	GEN_ARGS_FAST_QR_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)





///////////////// EQSP


#define GEN_ARGS_EQSP_1S_1(prefix)	GEN_ARGS_EQSP_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_EQSP_CPX_1S_1(prefix)	GEN_ARGS_EQSP_CPX_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_CPX_1S(prefix)

#define GEN_ARGS_EQSP_QUAT_1S_1(prefix)	GEN_ARGS_EQSP_QUAT_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_QUAT_1S(prefix)

#define GEN_ARGS_EQSP_2S_1(prefix)	GEN_ARGS_EQSP_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_2S(prefix)

#define GEN_ARGS_EQSP_3S_1(prefix)	GEN_ARGS_EQSP_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_3S(prefix)

#define GEN_ARGS_EQSP_SBM_2S_1(prefix)	GEN_ARGS_EQSP_SBM_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_2S(prefix)

#define GEN_ARGS_EQSP_SBM_CPX_2S_1(prefix)	GEN_ARGS_EQSP_SBM_CPX_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_CPX_2S(prefix)

#define GEN_ARGS_EQSP_SBM_QUAT_2S_1(prefix)	GEN_ARGS_EQSP_SBM_QUAT_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_QUAT_2S(prefix)

#define GEN_ARGS_EQSP_RC_2(prefix)	GEN_ARGS_EQSP_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_EQSP_CPX_SRC1(prefix)

#define GEN_ARGS_EQSP_DBM_SBM(prefix)		GEN_ARGS_EQSP_DBM(prefix)	\
						GEN_SEP(prefix)			\
						GEN_ARGS_EQSP_SBM(prefix)

#define GEN_ARGS_EQSP_DBM_2SRCS(prefix)		GEN_ARGS_EQSP_2SRCS(prefix)	\
						GEN_SEP(prefix)			\
						GEN_ARGS_EQSP_DBM(prefix)

#define GEN_ARGS_EQSP_DBM_1S_1SRC(prefix)	GEN_ARGS_EQSP_1S_1SRC(prefix)	\
						GEN_SEP(prefix)			\
						GEN_ARGS_EQSP_DBM(prefix)

#define GEN_ARGS_EQSP_DBM_1S_(prefix)		GEN_ARGS_1S(prefix)	\
						GEN_SEP(prefix)			\
						GEN_ARGS_EQSP_DBM(prefix)

#define GEN_EQSP_CONV_DEST(prefix,t)	GEN_FAST_CONV_DEST	(prefix,t)	\
				GEN_SEP			(prefix)	\
				GEN_EQSP_ARG_INC1	(prefix)

#define GEN_ARGS_EQSP_1(prefix)	GEN_ARGS_FAST_1		(prefix)	\
				GEN_SEP			(prefix)	\
				GEN_EQSP_ARG_INC1	(prefix)

#define GEN_ARGS_EQSP_CPX_1(prefix)	GEN_ARGS_FAST_CPX_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_EQSP_ARG_INC1(prefix)

#define GEN_ARGS_EQSP_QUAT_1(prefix)	GEN_ARGS_FAST_QUAT_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_EQSP_ARG_INC1(prefix)

#define GEN_ARGS_EQSP_CR_2(prefix)	GEN_ARGS_EQSP_CPX_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SRC1(prefix)

#define GEN_ARGS_EQSP_QR_2(prefix)	GEN_ARGS_EQSP_QUAT_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SRC1(prefix)

#define GEN_ARGS_EQSP_2SRCS(prefix)	GEN_ARGS_EQSP_SRC1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SRC2(prefix)

#define GEN_ARGS_EQSP_1S_1SRC(prefix)	GEN_ARGS_EQSP_SRC1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_EQSP_SBM_1(prefix)	GEN_ARGS_EQSP_1(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SBM(prefix)

#define GEN_ARGS_EQSP_SBM_CPX_1(prefix)	GEN_ARGS_EQSP_CPX_1(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SBM(prefix)

#define GEN_ARGS_EQSP_SBM_QUAT_1(prefix)	GEN_ARGS_EQSP_QUAT_1(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SBM(prefix)

#define GEN_ARGS_EQSP_SBM_2(prefix)	GEN_ARGS_EQSP_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SBM(prefix)

#define GEN_ARGS_EQSP_SBM_CPX_2(prefix)	GEN_ARGS_EQSP_CPX_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SBM(prefix)

#define GEN_ARGS_EQSP_SBM_QUAT_2(prefix)	GEN_ARGS_EQSP_QUAT_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SBM(prefix)

#define GEN_ARGS_EQSP_4(prefix)		GEN_ARGS_EQSP_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SRC3(prefix)

#define GEN_ARGS_EQSP_5(prefix)		GEN_ARGS_EQSP_4(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SRC4(prefix)

#define GEN_ARGS_EQSP_1S_4(prefix)	GEN_ARGS_EQSP_4(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_EQSP_2S_3(prefix)	GEN_ARGS_EQSP_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_2S(prefix)

#define GEN_ARGS_EQSP_3S_2(prefix)	GEN_ARGS_EQSP_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_3S(prefix)

#define GEN_ARGS_EQSP_SBM_3(prefix)	GEN_ARGS_EQSP_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SBM(prefix)

#define GEN_ARGS_EQSP_SBM_CPX_3(prefix)	GEN_ARGS_EQSP_CPX_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SBM(prefix)

#define GEN_ARGS_EQSP_SBM_QUAT_3(prefix)	GEN_ARGS_EQSP_QUAT_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_EQSP_SBM(prefix)

#define GEN_ARGS_EQSP_1S_2(prefix)	GEN_ARGS_EQSP_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_EQSP_SBM_1S_2(prefix)	GEN_ARGS_EQSP_SBM_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_EQSP_CPX_1S_2(prefix)	GEN_ARGS_EQSP_CPX_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_CPX_1S(prefix)

#define GEN_ARGS_EQSP_SBM_CPX_1S_2(prefix)	GEN_ARGS_EQSP_SBM_CPX_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_CPX_1S(prefix)

#define GEN_ARGS_EQSP_SBM_QUAT_1S_2(prefix)	GEN_ARGS_EQSP_SBM_QUAT_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_QUAT_1S(prefix)

#define GEN_ARGS_EQSP_CR_1S_2(prefix)	GEN_ARGS_EQSP_CR_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_EQSP_CR_1S_2(prefix)	GEN_ARGS_EQSP_CR_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_EQSP_QUAT_1S_2(prefix)	GEN_ARGS_EQSP_QUAT_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_QUAT_1S(prefix)

#define GEN_ARGS_EQSP_QR_1S_2(prefix)	GEN_ARGS_EQSP_QR_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_EQSP_SRC1(prefix)	GEN_ARGS_FAST_SRC1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_EQSP_ARG_INC2(prefix)

#define GEN_ARGS_EQSP_CPX_SRC1(prefix)	GEN_ARGS_FAST_CPX_SRC1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_EQSP_ARG_INC2(prefix)

#define GEN_ARGS_EQSP_QUAT_SRC1(prefix)	GEN_ARGS_FAST_QUAT_SRC1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_EQSP_ARG_INC2(prefix)

#define GEN_ARGS_EQSP_SRC2(prefix)	GEN_ARGS_FAST_SRC2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_EQSP_ARG_INC3(prefix)

#define GEN_ARGS_EQSP_CPX_SRC2(prefix)	GEN_ARGS_FAST_CPX_SRC2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_EQSP_ARG_INC3(prefix)

#define GEN_ARGS_EQSP_QUAT_SRC2(prefix)	GEN_ARGS_FAST_QUAT_SRC2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_EQSP_ARG_INC3(prefix)

#define GEN_ARGS_EQSP_SRC3(prefix)	GEN_ARGS_FAST_SRC3(prefix)	\
					GEN_SEP(prefix)			\
					GEN_EQSP_ARG_INC4(prefix)

#define GEN_ARGS_EQSP_CPX_SRC3(prefix)	GEN_ARGS_FAST_CPX_SRC3(prefix)	\
					GEN_SEP(prefix)			\
					GEN_EQSP_ARG_INC4(prefix)

#define GEN_ARGS_EQSP_QUAT_SRC3(prefix)	GEN_ARGS_FAST_QUAT_SRC3(prefix)	\
					GEN_SEP(prefix)			\
					GEN_EQSP_ARG_INC4(prefix)

#define GEN_ARGS_EQSP_SRC4(prefix)	GEN_ARGS_FAST_SRC4(prefix)	\
					GEN_SEP(prefix)			\
					GEN_EQSP_ARG_INC5(prefix)

#define GEN_ARGS_EQSP_CPX_SRC4(prefix)	GEN_ARGS_FAST_CPX_SRC4(prefix)	\
					GEN_SEP(prefix)			\
					GEN_EQSP_ARG_INC5(prefix)

#define GEN_ARGS_EQSP_QUAT_SRC4(prefix)	GEN_ARGS_FAST_QUAT_SRC4(prefix)	\
					GEN_SEP(prefix)			\
					GEN_EQSP_ARG_INC5(prefix)

#define GEN_ARGS_EQSP_CPX_1(prefix)	GEN_ARGS_FAST_CPX_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_EQSP_ARG_INC1(prefix)

#define GEN_ARGS_EQSP_CONV(prefix,t)	GEN_EQSP_CONV_DEST(prefix,t) \
					GEN_SEP(prefix)		\
					GEN_ARGS_EQSP_SRC1(prefix)

#define GEN_ARGS_EQSP_2(prefix)		GEN_ARGS_EQSP_1(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_EQSP_SRC1(prefix)

#define GEN_ARGS_EQSP_CPX_2(prefix)	GEN_ARGS_EQSP_CPX_1(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_EQSP_CPX_SRC1(prefix)

#define GEN_ARGS_EQSP_QUAT_2(prefix)	GEN_ARGS_EQSP_QUAT_1(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_EQSP_QUAT_SRC1(prefix)

#define GEN_ARGS_EQSP_3(prefix)	GEN_ARGS_EQSP_2(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_EQSP_SRC2(prefix)

#define GEN_ARGS_EQSP_CPX_3(prefix)	GEN_ARGS_EQSP_CPX_2(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_EQSP_CPX_SRC2(prefix)

#define GEN_ARGS_EQSP_QUAT_3(prefix)	GEN_ARGS_EQSP_QUAT_2(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_EQSP_QUAT_SRC2(prefix)

#define GEN_ARGS_EQSP_CCR_3(prefix)	GEN_ARGS_EQSP_CPX_2(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_EQSP_SRC2(prefix)

#define GEN_ARGS_EQSP_QQR_3(prefix)	GEN_ARGS_EQSP_QUAT_2(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_EQSP_SRC2(prefix)



//// SLOW

#define GEN_ARGS_SLOW_1S_1(prefix)	GEN_ARGS_SLOW_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_SLOW_CPX_1S_1(prefix)	GEN_ARGS_SLOW_CPX_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_CPX_1S(prefix)

#define GEN_ARGS_SLOW_QUAT_1S_1(prefix)	GEN_ARGS_SLOW_QUAT_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_QUAT_1S(prefix)



#define GEN_ARGS_SLOW_2S_1(prefix)	GEN_ARGS_SLOW_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_2S(prefix)



#define GEN_ARGS_SLOW_3S_1(prefix)	GEN_ARGS_SLOW_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_3S(prefix)



#define GEN_ARGS_SLOW_SBM_2S_1(prefix)	GEN_ARGS_SLOW_SBM_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_2S(prefix)



#define GEN_ARGS_SLOW_SBM_CPX_2S_1(prefix)	GEN_ARGS_SLOW_SBM_CPX_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_CPX_2S(prefix)



#define GEN_ARGS_SLOW_SBM_QUAT_2S_1(prefix)	GEN_ARGS_SLOW_SBM_QUAT_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_QUAT_2S(prefix)



#define GEN_ARGS_SLOW_RC_2(prefix)	GEN_ARGS_SLOW_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_ARGS_SLOW_CPX_SRC1(prefix)



#define GEN_ARGS_SLOW_DBM_SBM(prefix)		GEN_ARGS_SLOW_DBM(prefix)	\
						GEN_SEP(prefix)			\
						GEN_ARGS_SLOW_SBM(prefix)

#define GEN_ARGS_SLOW_DBM_2SRCS(prefix)		GEN_ARGS_SLOW_2SRCS(prefix)	\
						GEN_SEP(prefix)			\
						GEN_ARGS_SLOW_DBM(prefix)



#define GEN_ARGS_SLOW_DBM_1S_1SRC(prefix)	GEN_ARGS_SLOW_1S_1SRC(prefix)	\
						GEN_SEP(prefix)			\
						GEN_ARGS_SLOW_DBM(prefix)

#define GEN_ARGS_SLOW_DBM_1S_(prefix)		GEN_ARGS_1S(prefix)		\
						GEN_SEP(prefix)			\
						GEN_ARGS_SLOW_DBM_BASIC(prefix)


#define GEN_SLOW_CONV_DEST(prefix,t)	GEN_FAST_CONV_DEST(prefix,t)	\
					GEN_SEP(prefix)		\
					GEN_SLOW_ARG_INC1(prefix)

//#define GEN_ARGS_FAST_1(p)		p##_FAST_1		OS_ARG(p,DEST)
//
//#define GEN_ARGS_SLOW_1(prefix)		GEN_ARGS_FAST_1(prefix)	\
//					GEN_SEP(prefix)		\
//					GEN_SLOW_ARG_INC1(prefix)

#define GEN_SLOW_SIZE(p)		p##_SLOW_SIZE
#define GEN_SLOW_DBM_GPU_INFO(p)	p##_DBM_GPU_INFO

#define GEN_ARGS_SLOW_1(prefix)		GEN_SLOW_SIZE(prefix)		\
					GEN_SEP(prefix) 		\
					GEN_ARGS_FAST_1(prefix)		\
					GEN_SEP(prefix) 		\
					GEN_SLOW_ARG_INC1(prefix)

#define GEN_ARGS_SLOW_CPX_1(prefix)					\
					GEN_SLOW_SIZE(prefix)		\
					GEN_SEP(prefix) 		\
					GEN_ARGS_FAST_CPX_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_SLOW_ARG_INC1(prefix)

#define GEN_ARGS_SLOW_QUAT_1(prefix)					\
					GEN_SLOW_SIZE(prefix)		\
					GEN_SEP(prefix) 		\
					GEN_ARGS_FAST_QUAT_1(prefix)	\
					GEN_SEP(prefix)		\
					GEN_SLOW_ARG_INC1(prefix)


#define GEN_ARGS_SLOW_CR_2(prefix)	GEN_ARGS_SLOW_CPX_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SRC1(prefix)



#define GEN_ARGS_SLOW_QR_2(prefix)	GEN_ARGS_SLOW_QUAT_1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SRC1(prefix)






#define GEN_ARGS_SLOW_2SRCS(prefix)	GEN_ARGS_SLOW_SRC1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SRC2(prefix)

#define GEN_ARGS_SLOW_1S_1SRC(prefix)	GEN_ARGS_SLOW_SRC1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)




#define GEN_ARGS_SLOW_SBM_1(prefix)	GEN_ARGS_SLOW_1(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SBM(prefix)

#define GEN_ARGS_SLOW_SBM_CPX_1(prefix)	GEN_ARGS_SLOW_CPX_1(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SBM(prefix)

#define GEN_ARGS_SLOW_SBM_QUAT_1(prefix)	GEN_ARGS_SLOW_QUAT_1(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SBM(prefix)



#define GEN_ARGS_SLOW_SBM_2(prefix)	GEN_ARGS_SLOW_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SBM(prefix)

#define GEN_ARGS_SLOW_SBM_CPX_2(prefix)	GEN_ARGS_SLOW_CPX_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SBM(prefix)

#define GEN_ARGS_SLOW_SBM_QUAT_2(prefix)	GEN_ARGS_SLOW_QUAT_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SBM(prefix)

#define GEN_ARGS_SLOW_4(prefix)		GEN_ARGS_SLOW_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SRC3(prefix)

#define GEN_ARGS_SLOW_5(prefix)		GEN_ARGS_SLOW_4(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SRC4(prefix)

#define GEN_ARGS_SLOW_1S_4(prefix)	GEN_ARGS_SLOW_4(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_SLOW_2S_3(prefix)	GEN_ARGS_SLOW_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_2S(prefix)

#define GEN_ARGS_SLOW_3S_2(prefix)	GEN_ARGS_SLOW_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_3S(prefix)

#define GEN_ARGS_SLOW_SBM_3(prefix)	GEN_ARGS_SLOW_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SBM(prefix)

#define GEN_ARGS_SLOW_SBM_CPX_3(prefix)	GEN_ARGS_SLOW_CPX_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SBM(prefix)

#define GEN_ARGS_SLOW_SBM_QUAT_3(prefix)	GEN_ARGS_SLOW_QUAT_3(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SBM(prefix)

#define GEN_ARGS_SLOW_1S_2(prefix)	GEN_ARGS_SLOW_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_SLOW_SBM_1S_2(prefix)	GEN_ARGS_SLOW_SBM_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_SLOW_CPX_1S_2(prefix)	GEN_ARGS_SLOW_CPX_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_CPX_1S(prefix)

#define GEN_ARGS_SLOW_SBM_CPX_1S_2(prefix)	GEN_ARGS_SLOW_SBM_CPX_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_CPX_1S(prefix)

#define GEN_ARGS_SLOW_SBM_QUAT_1S_2(prefix)	GEN_ARGS_SLOW_SBM_QUAT_2(prefix)		\
					GEN_SEP(prefix)			\
					GEN_ARGS_QUAT_1S(prefix)

#define GEN_ARGS_SLOW_CR_1S_2(prefix)	GEN_ARGS_SLOW_CR_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_SLOW_CR_1S_2(prefix)	GEN_ARGS_SLOW_CR_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)

#define GEN_ARGS_SLOW_QUAT_1S_2(prefix)	GEN_ARGS_SLOW_QUAT_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_QUAT_1S(prefix)

#define GEN_ARGS_SLOW_QR_1S_2(prefix)	GEN_ARGS_SLOW_QR_2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_1S(prefix)




// SRC1


#define GEN_ARGS_SLOW_SRC1(prefix)	GEN_ARGS_FAST_SRC1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_SLOW_ARG_INC2(prefix)



#define GEN_ARGS_SLOW_CPX_SRC1(prefix)	GEN_ARGS_FAST_CPX_SRC1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_SLOW_ARG_INC2(prefix)



#define GEN_ARGS_SLOW_QUAT_SRC1(prefix)	GEN_ARGS_FAST_QUAT_SRC1(prefix)	\
					GEN_SEP(prefix)			\
					GEN_SLOW_ARG_INC2(prefix)


// SRC2

#define GEN_ARGS_SLOW_SRC2(prefix)	GEN_ARGS_FAST_SRC2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_SLOW_ARG_INC3(prefix)


#define GEN_ARGS_SLOW_CPX_SRC2(prefix)	GEN_ARGS_FAST_CPX_SRC2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_SLOW_ARG_INC3(prefix)


#define GEN_ARGS_SLOW_QUAT_SRC2(prefix)	GEN_ARGS_FAST_QUAT_SRC2(prefix)	\
					GEN_SEP(prefix)			\
					GEN_SLOW_ARG_INC3(prefix)


// SRC3

#define GEN_ARGS_SLOW_SRC3(prefix)	GEN_ARGS_FAST_SRC3(prefix)	\
					GEN_SEP(prefix)			\
					GEN_SLOW_ARG_INC4(prefix)


#define GEN_ARGS_SLOW_CPX_SRC3(prefix)	GEN_ARGS_FAST_CPX_SRC3(prefix)	\
					GEN_SEP(prefix)			\
					GEN_SLOW_ARG_INC4(prefix)


#define GEN_ARGS_SLOW_QUAT_SRC3(prefix)	GEN_ARGS_FAST_QUAT_SRC3(prefix)	\
					GEN_SEP(prefix)			\
					GEN_SLOW_ARG_INC4(prefix)



// SRC4

#define GEN_ARGS_SLOW_SRC4(prefix)	GEN_ARGS_FAST_SRC4(prefix)	\
					GEN_SEP(prefix)			\
					GEN_SLOW_ARG_INC5(prefix)



#define GEN_ARGS_SLOW_CPX_SRC4(prefix)	GEN_ARGS_FAST_CPX_SRC4(prefix)	\
					GEN_SEP(prefix)			\
					GEN_SLOW_ARG_INC5(prefix)



#define GEN_ARGS_SLOW_QUAT_SRC4(prefix)	GEN_ARGS_FAST_QUAT_SRC4(prefix)	\
					GEN_SEP(prefix)			\
					GEN_SLOW_ARG_INC5(prefix)


/////


#define GEN_ARGS_SLOW_CONV(prefix,t)					\
					GEN_SLOW_SIZE(prefix)		\
					GEN_SEP(prefix)			\
					GEN_SLOW_CONV_DEST(prefix,t)	\
					GEN_SEP(prefix)			\
					GEN_ARGS_SLOW_SRC1(prefix)

#define GEN_ARGS_SLOW_2(prefix)		GEN_ARGS_SLOW_1(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_SLOW_SRC1(prefix)

#define GEN_ARGS_SLOW_CPX_2(prefix)	GEN_ARGS_SLOW_CPX_1(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_SLOW_CPX_SRC1(prefix)

#define GEN_ARGS_SLOW_QUAT_2(prefix)	GEN_ARGS_SLOW_QUAT_1(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_SLOW_QUAT_SRC1(prefix)

#define GEN_ARGS_SLOW_3(prefix)		GEN_ARGS_SLOW_2(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_SLOW_SRC2(prefix)

#define GEN_ARGS_SLOW_CPX_3(prefix)	GEN_ARGS_SLOW_CPX_2(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_SLOW_CPX_SRC2(prefix)

#define GEN_ARGS_SLOW_CCR_3(prefix)	GEN_ARGS_SLOW_CPX_2(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_SLOW_SRC2(prefix)

#define GEN_ARGS_SLOW_QUAT_3(prefix)	GEN_ARGS_SLOW_QUAT_2(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_SLOW_QUAT_SRC2(prefix)

#define GEN_ARGS_SLOW_QQR_3(prefix)	GEN_ARGS_SLOW_QUAT_2(prefix) \
					GEN_SEP(prefix)		\
					GEN_ARGS_SLOW_SRC2(prefix)

// Now the len versions

#define GEN_ADD_FAST_LEN(prefix)	GEN_SEP(prefix)	GEN_FAST_ARG_LEN(prefix)
#define GEN_ADD_EQSP_LEN(prefix)	GEN_SEP(prefix)	GEN_EQSP_ARG_LEN(prefix)
//#define GEN_ADD_SLOW_LEN(prefix)	GEN_SEP(prefix)	GEN_SLOW_ARG_LEN(prefix)
#define GEN_ADD_SLOW_LEN(prefix)	// nop - we now use szarr (SLOW_SIZE)
// FLEN


#define GEN_ARGS_FLEN_1S_1(prefix)	GEN_ARGS_FAST_1S_1(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_CPX_1S_1(prefix)	GEN_ARGS_FAST_CPX_1S_1(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_QUAT_1S_1(prefix)	GEN_ARGS_FAST_QUAT_1S_1(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_2S_1(prefix)	GEN_ARGS_FAST_2S_1(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_3S_1(prefix)	GEN_ARGS_FAST_3S_1(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_2S_1(prefix)	GEN_ARGS_FAST_SBM_2S_1(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_CPX_2S_1(prefix)	GEN_ARGS_FAST_SBM_CPX_2S_1(prefix)	\
						GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_QUAT_2S_1(prefix)	GEN_ARGS_FAST_SBM_QUAT_2S_1(prefix)	\
						GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_RC_2(prefix)	GEN_ARGS_FAST_RC_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_DBM_SBM(prefix)		GEN_SLOW_DBM_GPU_INFO(prefix)	\
						GEN_SEP(prefix)			\
						prefix##_EQSP_DBM_SBM

#define GEN_ARGS_FLEN_DBM_2SRCS(prefix)		prefix##_EQSP_DBM_2SRCS

#define GEN_ARGS_FLEN_DBM_1S_1SRC(prefix)	prefix##_EQSP_DBM_1S_1SRC

#define GEN_ARGS_FLEN_DBM_1S_(prefix)		GEN_SLOW_DBM_GPU_INFO(prefix)	\
						GEN_SEP(prefix)			\
						prefix##_EQSP_DBM_1S_

#define GEN_ARGS_FLEN_CONV(prefix,t)	GEN_ARGS_FAST_CONV(prefix,t)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_2(prefix)	GEN_ARGS_FAST_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_CPX_2(prefix)	GEN_ARGS_FAST_CPX_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_CR_2(prefix)	GEN_ARGS_FAST_CR_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_QR_2(prefix)	GEN_ARGS_FAST_QR_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_2SRCS(prefix)	GEN_ARGS_FAST_2SRCS(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_1S_1SRC(prefix)	GEN_ARGS_FAST_1S_1SRC(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_FLEN_ARG_DBM_2SRCS(prefix)	GEN_FAST_ARG_DBM_2SRCS(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_1(prefix)	GEN_ARGS_FAST_SBM_1(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_CPX_1(prefix)	GEN_ARGS_FAST_SBM_CPX_1(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_QUAT_1(prefix)	GEN_ARGS_FAST_SBM_QUAT_1(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_2(prefix)	GEN_ARGS_FAST_SBM_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_CPX_2(prefix)	GEN_ARGS_FAST_SBM_CPX_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_QUAT_2(prefix)	GEN_ARGS_FAST_SBM_QUAT_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_QUAT_2(prefix)	GEN_ARGS_FAST_QUAT_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_QR_2(prefix)	GEN_ARGS_FAST_QR_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_4(prefix)	GEN_ARGS_FAST_4(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_5(prefix)	GEN_ARGS_FAST_5(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_1S_4(prefix)	GEN_ARGS_FAST_1S_4(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_2S_3(prefix)	GEN_ARGS_FAST_2S_3(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_3S_2(prefix)	GEN_ARGS_FAST_3S_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_3(prefix)	GEN_ARGS_FAST_3(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_CPX_3(prefix)	GEN_ARGS_FAST_CPX_3(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_CCR_3(prefix)	GEN_ARGS_FAST_CCR_3(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_QUAT_3(prefix)	GEN_ARGS_FAST_QUAT_3(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_QQR_3(prefix)	GEN_ARGS_FAST_QQR_3(prefix)	\
					GEN_ADD_FAST_LEN(prefix)



#define GEN_ARGS_FLEN_SBM_3(prefix)	GEN_ARGS_FAST_SBM_3(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_CPX_3(prefix)	GEN_ARGS_FAST_SBM_CPX_3(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_QUAT_3(prefix)	GEN_ARGS_FAST_SBM_QUAT_3(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_1S_2(prefix)	GEN_ARGS_FAST_1S_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_1S_2(prefix)	GEN_ARGS_FAST_SBM_1S_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_CPX_1S_2(prefix)	GEN_ARGS_FAST_CPX_1S_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_CPX_1S_2(prefix)	GEN_ARGS_FAST_SBM_CPX_1S_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_SBM_QUAT_1S_2(prefix)	GEN_ARGS_FAST_SBM_QUAT_1S_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_QUAT_1S_2(prefix)	GEN_ARGS_FAST_QUAT_1S_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_CR_1S_2(prefix)	GEN_ARGS_FAST_CR_1S_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_QUAT_1S_2(prefix)	GEN_ARGS_FAST_QUAT_1S_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)

#define GEN_ARGS_FLEN_QR_1S_2(prefix)	GEN_ARGS_FAST_QR_1S_2(prefix)	\
					GEN_ADD_FAST_LEN(prefix)




// ELEN


#define GEN_ARGS_ELEN_1S_1(prefix)	GEN_ARGS_EQSP_1S_1(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_CPX_1S_1(prefix)	GEN_ARGS_EQSP_CPX_1S_1(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_QUAT_1S_1(prefix)	GEN_ARGS_EQSP_QUAT_1S_1(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_2S_1(prefix)	GEN_ARGS_EQSP_2S_1(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_3S_1(prefix)	GEN_ARGS_EQSP_3S_1(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_2S_1(prefix)	GEN_ARGS_EQSP_SBM_2S_1(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_CPX_2S_1(prefix)	GEN_ARGS_EQSP_SBM_CPX_2S_1(prefix)	\
						GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_QUAT_2S_1(prefix)	GEN_ARGS_EQSP_SBM_QUAT_2S_1(prefix)	\
						GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_RC_2(prefix)	GEN_ARGS_EQSP_RC_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_DBM_SBM(prefix)	GEN_ARGS_EQSP_DBM_SBM(prefix)	\
						GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_DBM_2SRCS(prefix)	GEN_ARGS_EQSP_DBM_2SRCS(prefix)	\
						GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_DBM_1S_1SRC(prefix)	GEN_ARGS_EQSP_DBM_1S_1SRC(prefix)	\
						GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_DBM_1S_(prefix)	GEN_ARGS_EQSP_DBM_1S_(prefix)	\
						GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_CONV(prefix,t)	GEN_ARGS_EQSP_CONV(prefix,t)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_2(prefix)	GEN_ARGS_EQSP_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_CPX_2(prefix)	GEN_ARGS_EQSP_CPX_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_CR_2(prefix)	GEN_ARGS_EQSP_CR_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_QR_2(prefix)	GEN_ARGS_EQSP_QR_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_2SRCS(prefix)	GEN_ARGS_EQSP_2SRCS(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_1S_1SRC(prefix)	GEN_ARGS_EQSP_1S_1SRC(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ELEN_ARG_DBM_2SRCS(prefix)	GEN_EQSP_ARG_DBM_2SRCS(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_1(prefix)	GEN_ARGS_EQSP_SBM_1(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_CPX_1(prefix)	GEN_ARGS_EQSP_SBM_CPX_1(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_QUAT_1(prefix)	GEN_ARGS_EQSP_SBM_QUAT_1(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_2(prefix)	GEN_ARGS_EQSP_SBM_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_CPX_2(prefix)	GEN_ARGS_EQSP_SBM_CPX_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_QUAT_2(prefix)	GEN_ARGS_EQSP_SBM_QUAT_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_QUAT_2(prefix)	GEN_ARGS_EQSP_QUAT_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_QR_2(prefix)	GEN_ARGS_EQSP_QR_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_4(prefix)	GEN_ARGS_EQSP_4(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_5(prefix)	GEN_ARGS_EQSP_5(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_1S_4(prefix)	GEN_ARGS_EQSP_1S_4(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_2S_3(prefix)	GEN_ARGS_EQSP_2S_3(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_3S_2(prefix)	GEN_ARGS_EQSP_3S_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_3(prefix)	GEN_ARGS_EQSP_3(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_CPX_3(prefix)	GEN_ARGS_EQSP_CPX_3(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_CCR_3(prefix)	GEN_ARGS_EQSP_CCR_3(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_QUAT_3(prefix)	GEN_ARGS_EQSP_QUAT_3(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_QQR_3(prefix)	GEN_ARGS_EQSP_QQR_3(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)



#define GEN_ARGS_ELEN_SBM_3(prefix)	GEN_ARGS_EQSP_SBM_3(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_CPX_3(prefix)	GEN_ARGS_EQSP_SBM_CPX_3(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_QUAT_3(prefix)	GEN_ARGS_EQSP_SBM_QUAT_3(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_1S_2(prefix)	GEN_ARGS_EQSP_1S_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_1S_2(prefix)	GEN_ARGS_EQSP_SBM_1S_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_CPX_1S_2(prefix)	GEN_ARGS_EQSP_CPX_1S_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_CPX_1S_2(prefix)	GEN_ARGS_EQSP_SBM_CPX_1S_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_SBM_QUAT_1S_2(prefix)	GEN_ARGS_EQSP_SBM_QUAT_1S_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_QUAT_1S_2(prefix)	GEN_ARGS_EQSP_QUAT_1S_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_CR_1S_2(prefix)	GEN_ARGS_EQSP_CR_1S_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_QUAT_1S_2(prefix)	GEN_ARGS_EQSP_QUAT_1S_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)

#define GEN_ARGS_ELEN_QR_1S_2(prefix)	GEN_ARGS_EQSP_QR_1S_2(prefix)	\
					GEN_ADD_EQSP_LEN(prefix)





// SLEN



#define GEN_ARGS_SLEN_1S_1(prefix)	GEN_ARGS_SLOW_1S_1(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_CPX_1S_1(prefix)	GEN_ARGS_SLOW_CPX_1S_1(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_QUAT_1S_1(prefix)	GEN_ARGS_SLOW_QUAT_1S_1(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_2S_1(prefix)	GEN_ARGS_SLOW_2S_1(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_3S_1(prefix)	GEN_ARGS_SLOW_3S_1(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_2S_1(prefix)	GEN_ARGS_SLOW_SBM_2S_1(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_CPX_2S_1(prefix)	GEN_ARGS_SLOW_SBM_CPX_2S_1(prefix)	\
						GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_QUAT_2S_1(prefix)	GEN_ARGS_SLOW_SBM_QUAT_2S_1(prefix)	\
						GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_RC_2(prefix)	GEN_ARGS_SLOW_RC_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_DBM_SBM(prefix)	GEN_ARGS_SLOW_DBM_SBM(prefix)	\
						GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_DBM_2SRCS(prefix)	GEN_ARGS_SLOW_DBM_2SRCS(prefix)	\
						GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_DBM_1S_1SRC(prefix)	GEN_ARGS_SLOW_DBM_1S_1SRC(prefix)	\
						GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_DBM_1S_(prefix)	GEN_ARGS_SLOW_DBM_1S_(prefix)	\
						GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_CONV(prefix,t)	GEN_ARGS_SLOW_CONV(prefix,t)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_2(prefix)	GEN_ARGS_SLOW_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_CPX_2(prefix)	GEN_ARGS_SLOW_CPX_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_CR_2(prefix)	GEN_ARGS_SLOW_CR_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_QR_2(prefix)	GEN_ARGS_SLOW_QR_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_2SRCS(prefix)	GEN_ARGS_SLOW_2SRCS(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_1S_1SRC(prefix)	GEN_ARGS_SLOW_1S_1SRC(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_SLEN_ARG_DBM_2SRCS(prefix)	GEN_SLOW_ARG_DBM_2SRCS(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_1(prefix)	GEN_ARGS_SLOW_SBM_1(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_CPX_1(prefix)	GEN_ARGS_SLOW_SBM_CPX_1(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_QUAT_1(prefix)	GEN_ARGS_SLOW_SBM_QUAT_1(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_2(prefix)	GEN_ARGS_SLOW_SBM_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_CPX_2(prefix)	GEN_ARGS_SLOW_SBM_CPX_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_QUAT_2(prefix)	GEN_ARGS_SLOW_SBM_QUAT_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_QUAT_2(prefix)	GEN_ARGS_SLOW_QUAT_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_QR_2(prefix)	GEN_ARGS_SLOW_QR_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_4(prefix)	GEN_ARGS_SLOW_4(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_5(prefix)	GEN_ARGS_SLOW_5(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_1S_4(prefix)	GEN_ARGS_SLOW_1S_4(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_2S_3(prefix)	GEN_ARGS_SLOW_2S_3(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_3S_2(prefix)	GEN_ARGS_SLOW_3S_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_3(prefix)	GEN_ARGS_SLOW_3(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_CPX_3(prefix)	GEN_ARGS_SLOW_CPX_3(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_CCR_3(prefix)	GEN_ARGS_SLOW_CCR_3(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_QUAT_3(prefix)	GEN_ARGS_SLOW_QUAT_3(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_QQR_3(prefix)	GEN_ARGS_SLOW_QQR_3(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)



#define GEN_ARGS_SLEN_SBM_3(prefix)	GEN_ARGS_SLOW_SBM_3(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_CPX_3(prefix)	GEN_ARGS_SLOW_SBM_CPX_3(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_QUAT_3(prefix)	GEN_ARGS_SLOW_SBM_QUAT_3(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_1S_2(prefix)	GEN_ARGS_SLOW_1S_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_1S_2(prefix)	GEN_ARGS_SLOW_SBM_1S_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_CPX_1S_2(prefix)	GEN_ARGS_SLOW_CPX_1S_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_CPX_1S_2(prefix)	GEN_ARGS_SLOW_SBM_CPX_1S_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_SBM_QUAT_1S_2(prefix)	GEN_ARGS_SLOW_SBM_QUAT_1S_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_QUAT_1S_2(prefix)	GEN_ARGS_SLOW_QUAT_1S_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_CR_1S_2(prefix)	GEN_ARGS_SLOW_CR_1S_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_QUAT_1S_2(prefix)	GEN_ARGS_SLOW_QUAT_1S_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)

#define GEN_ARGS_SLEN_QR_1S_2(prefix)	GEN_ARGS_SLOW_QR_1S_2(prefix)	\
					GEN_ADD_SLOW_LEN(prefix)





#endif /* _GEN_KERN_ARGS_H_ */

