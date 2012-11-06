// Turn this on to get maximum debugging output
//#define MORE_DEBUG

// this is 80 columns
//345678901234567890123456789012345678901234567890123456789012345678901234567890

#include "host_call_utils.h"
#include "veclib/fast_test.h"
#include "veclib/xfer_args.h"

#ifdef HAVE_CUDA

#define HOST_CALL_NAME(name,code)		h_##code##_##name
#define GPU_FAST_CALL_NAME(name,code)	g_fast_##code##_##name
#define GPU_EQSP_CALL_NAME(name,code)	g_eqsp_##code##_##name
#define GPU_SLOW_CALL_NAME(name,code)	g_slow_##code##_##name

#define GPU_ARGS_1S		vap->va_sval[0].std_scalar
#define GPU_ARGS_2S		GPU_ARGS_1S, vap->va_sval[1].std_scalar
#define GPU_ARGS_3S		GPU_ARGS_2S, vap->va_sval[2].std_scalar

#define GPU_ARGS_CPX_1S		vap->va_sval[0].std_cpx_scalar
#define GPU_ARGS_CPX_2S		GPU_ARGS_CPX_1S, vap->va_sval[1].std_cpx_scalar
#define GPU_ARGS_CPX_3S		GPU_ARGS_CPX_2S, vap->va_sval[2].std_cpx_scalar

#define GPU_FAST_ARGS_1		(dest_type*)vap->va_dst_vp
#define GPU_FAST_ARGS_SRC1	(std_type*)vap->va_src_vp[0]
#define GPU_FAST_ARGS_SRC2	(std_type*)vap->va_src_vp[1]
#define GPU_FAST_ARGS_SRC3	(std_type*)vap->va_src_vp[2]
#define GPU_FAST_ARGS_SRC4	(std_type*)vap->va_src_vp[3]
#define GPU_FAST_ARGS_CPX_1	(dest_cpx*)vap->va_dst_vp
#define GPU_FAST_ARGS_CPX_SRC1	(std_cpx*)vap->va_src_vp[0]
#define GPU_FAST_ARGS_CPX_SRC2	(std_cpx*)vap->va_src_vp[1]
#define GPU_FAST_ARGS_CPX_SRC3	(std_cpx*)vap->va_src_vp[2]
#define GPU_FAST_ARGS_CPX_SRC4	(std_cpx*)vap->va_src_vp[3]
#define GPU_FAST_LEN_ARG	vap->va_len
#define GPU_FAST_ARGS_CPX_SRC1	(std_cpx*)vap->va_src_vp[0]

#define GPU_ARGS_SBM		(bitmap_word*)vap->va_src_vp[4],vap->va_bit0

#define GPU_FAST_ARGS_SBM	GPU_ARGS_SBM
#define GPU_EQSP_ARGS_SBM	GPU_ARGS_SBM,bm_incr
#define GPU_SLOW_ARGS_SBM	GPU_ARGS_SBM,bm_incr

#define GPU_ARGS_DBM		(bitmap_word*)vap->va_dst_vp,vap->va_bit0

#define GPU_FAST_ARGS_DBM	GPU_ARGS_DBM
#define GPU_EQSP_ARGS_DBM	GPU_ARGS_DBM,bm_incr
#define GPU_SLOW_ARGS_DBM	GPU_ARGS_DBM,bm_incr

#define GPU_FAST_ARGS_2		GPU_FAST_ARGS_1, GPU_FAST_ARGS_SRC1
#define GPU_FAST_ARGS_3		GPU_FAST_ARGS_2, GPU_FAST_ARGS_SRC2
#define GPU_FAST_ARGS_4		GPU_FAST_ARGS_3, GPU_FAST_ARGS_SRC3
#define GPU_FAST_ARGS_5		GPU_FAST_ARGS_4, GPU_FAST_ARGS_SRC4
#define GPU_FAST_ARGS_2SRCS	GPU_FAST_ARGS_SRC1, GPU_FAST_ARGS_SRC2

#define GPU_FAST_ARGS_CPX_2	GPU_FAST_ARGS_CPX_1, GPU_FAST_ARGS_CPX_SRC1
#define GPU_FAST_ARGS_CPX_3	GPU_FAST_ARGS_CPX_2, GPU_FAST_ARGS_CPX_SRC2
#define GPU_FAST_ARGS_CPX_4	GPU_FAST_ARGS_CPX_3, GPU_FAST_ARGS_CPX_SRC3
#define GPU_FAST_ARGS_CPX_5	GPU_FAST_ARGS_CPX_4, GPU_FAST_ARGS_CPX_SRC4

#define GPU_FAST_ARGS_SBM_2S_1	GPU_FAST_ARGS_2S_1, GPU_FAST_ARGS_SBM
#define GPU_EQSP_ARGS_SBM_2S_1	GPU_EQSP_ARGS_2S_1, GPU_EQSP_ARGS_SBM
#define GPU_SLOW_ARGS_SBM_2S_1	GPU_SLOW_ARGS_2S_1, GPU_SLOW_ARGS_SBM

#define GPU_FAST_ARGS_SBM_1S_2	GPU_FAST_ARGS_1S_2, GPU_FAST_ARGS_SBM
#define GPU_EQSP_ARGS_SBM_1S_2	GPU_EQSP_ARGS_1S_2, GPU_EQSP_ARGS_SBM
#define GPU_SLOW_ARGS_SBM_1S_2	GPU_SLOW_ARGS_1S_2, GPU_SLOW_ARGS_SBM

#define GPU_FAST_ARGS_SBM_3	GPU_FAST_ARGS_3, GPU_FAST_ARGS_SBM
#define GPU_EQSP_ARGS_SBM_3	GPU_EQSP_ARGS_3, GPU_EQSP_ARGS_SBM
#define GPU_SLOW_ARGS_SBM_3	GPU_SLOW_ARGS_3, GPU_SLOW_ARGS_SBM

#define GPU_FAST_ARGS_DBM_1S_	GPU_FAST_ARGS_DBM, GPU_ARGS_1S
#define GPU_EQSP_ARGS_DBM_1S_	GPU_EQSP_ARGS_DBM, GPU_ARGS_1S
#define GPU_SLOW_ARGS_DBM_1S_	GPU_SLOW_ARGS_DBM, GPU_ARGS_1S

#define GPU_FAST_ARGS_DBM_1S_1SRC	GPU_FAST_ARGS_DBM, GPU_FAST_ARGS_SRC1, GPU_ARGS_1S
#define GPU_EQSP_ARGS_DBM_1S_1SRC	GPU_EQSP_ARGS_DBM, GPU_EQSP_ARGS_SRC1, GPU_ARGS_1S
#define GPU_SLOW_ARGS_DBM_1S_1SRC	GPU_SLOW_ARGS_DBM, GPU_SLOW_ARGS_SRC1, GPU_ARGS_1S

#define GPU_FAST_ARGS_DBM_2SRCS	GPU_FAST_ARGS_DBM, GPU_FAST_ARGS_2SRCS
#define GPU_EQSP_ARGS_DBM_2SRCS	GPU_EQSP_ARGS_DBM, GPU_EQSP_ARGS_2SRCS
#define GPU_SLOW_ARGS_DBM_2SRCS	GPU_SLOW_ARGS_DBM, GPU_SLOW_ARGS_2SRCS

#define GPU_FAST_ARGS_RC_2	GPU_FAST_ARGS_1, GPU_FAST_ARGS_CPX_SRC1
#define GPU_EQSP_ARGS_RC_2	GPU_EQSP_ARGS_1, GPU_EQSP_ARGS_CPX_SRC1
#define GPU_SLOW_ARGS_RC_2	GPU_SLOW_ARGS_1, GPU_SLOW_ARGS_CPX_SRC1

#define GPU_FLEN_ARGS_1		GPU_FAST_ARGS_1, GPU_FAST_LEN_ARG
#define GPU_FLEN_ARGS_2		GPU_FAST_ARGS_2, GPU_FAST_LEN_ARG
#define GPU_FLEN_ARGS_3		GPU_FAST_ARGS_3, GPU_FAST_LEN_ARG
#define GPU_FLEN_ARGS_4		GPU_FAST_ARGS_4, GPU_FAST_LEN_ARG
#define GPU_FLEN_ARGS_5		GPU_FAST_ARGS_5, GPU_FAST_LEN_ARG

#define GPU_FLEN_ARGS_CPX_1		GPU_FAST_ARGS_CPX_1, GPU_FAST_LEN_ARG
#define GPU_FLEN_ARGS_CPX_2		GPU_FAST_ARGS_CPX_2, GPU_FAST_LEN_ARG
#define GPU_FLEN_ARGS_CPX_3		GPU_FAST_ARGS_CPX_3, GPU_FAST_LEN_ARG
#define GPU_FLEN_ARGS_CPX_4		GPU_FAST_ARGS_CPX_4, GPU_FAST_LEN_ARG
#define GPU_FLEN_ARGS_CPX_5		GPU_FAST_ARGS_CPX_5, GPU_FAST_LEN_ARG

#define GPU_FLEN_ARGS_SBM_2S_1	GPU_FAST_ARGS_SBM_2S_1, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_SBM_2S_1	GPU_EQSP_ARGS_SBM_2S_1, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_SBM_2S_1	GPU_SLOW_ARGS_SBM_2S_1, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_SBM_1S_2	GPU_FAST_ARGS_SBM_1S_2, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_SBM_1S_2	GPU_EQSP_ARGS_SBM_1S_2, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_SBM_1S_2	GPU_SLOW_ARGS_SBM_1S_2, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_SBM_3	GPU_FAST_ARGS_SBM_3, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_SBM_3	GPU_EQSP_ARGS_SBM_3, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_SBM_3	GPU_SLOW_ARGS_SBM_3, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_DBM_1S_	GPU_FAST_ARGS_DBM_1S_, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_DBM_1S_	GPU_EQSP_ARGS_DBM_1S_, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_DBM_1S_	GPU_SLOW_ARGS_DBM_1S_, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_DBM_1S_1SRC	GPU_FAST_ARGS_DBM_1S_1SRC, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_DBM_1S_1SRC	GPU_EQSP_ARGS_DBM_1S_1SRC, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_DBM_1S_1SRC	GPU_SLOW_ARGS_DBM_1S_1SRC, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_CPX_1S_1	GPU_FAST_ARGS_CPX_1S_1, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_CPX_1S_1	GPU_EQSP_ARGS_CPX_1S_1, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_CPX_1S_1	GPU_SLOW_ARGS_CPX_1S_1, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_DBM_2SRCS	GPU_FAST_ARGS_DBM_2SRCS, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_DBM_2SRCS	GPU_EQSP_ARGS_DBM_2SRCS, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_DBM_2SRCS	GPU_SLOW_ARGS_DBM_2SRCS, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_RC_2	GPU_FAST_ARGS_RC_2, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_RC_2	GPU_EQSP_ARGS_RC_2, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_RC_2	GPU_SLOW_ARGS_RC_2, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_1S_1	GPU_FAST_ARGS_1S_1, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_1S_1	GPU_EQSP_ARGS_1S_1, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_1S_1	GPU_SLOW_ARGS_1S_1, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_1S_2	GPU_FAST_ARGS_1S_2, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_1S_2	GPU_EQSP_ARGS_1S_2, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_1S_2	GPU_SLOW_ARGS_1S_2, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_1S_4	GPU_FAST_ARGS_1S_4, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_1S_4	GPU_EQSP_ARGS_1S_4, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_1S_4	GPU_SLOW_ARGS_1S_4, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_2S_1	GPU_FAST_ARGS_2S_1, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_2S_1	GPU_EQSP_ARGS_2S_1, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_2S_1	GPU_SLOW_ARGS_2S_1, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_2S_3	GPU_FAST_ARGS_2S_3, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_2S_3	GPU_EQSP_ARGS_2S_3, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_2S_3	GPU_SLOW_ARGS_2S_3, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_3S_1	GPU_FAST_ARGS_3S_1, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_3S_1	GPU_EQSP_ARGS_3S_1, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_3S_1	GPU_SLOW_ARGS_3S_1, GPU_SLOW_LEN_ARG

#define GPU_FLEN_ARGS_3S_2	GPU_FAST_ARGS_3S_2, GPU_FAST_LEN_ARG
#define GPU_ELEN_ARGS_3S_2	GPU_EQSP_ARGS_3S_2, GPU_EQSP_LEN_ARG
#define GPU_SLEN_ARGS_3S_2	GPU_SLOW_ARGS_3S_2, GPU_SLOW_LEN_ARG

#define GPU_FAST_ARGS_1S_1	GPU_FAST_ARGS_1, GPU_ARGS_1S
#define GPU_EQSP_ARGS_1S_1	GPU_EQSP_ARGS_1, GPU_ARGS_1S
#define GPU_SLOW_ARGS_1S_1	GPU_SLOW_ARGS_1, GPU_ARGS_1S

#define GPU_FAST_ARGS_CPX_1S_1	GPU_FAST_ARGS_CPX_1, GPU_ARGS_CPX_1S
#define GPU_EQSP_ARGS_CPX_1S_1	GPU_EQSP_ARGS_CPX_1, GPU_ARGS_CPX_1S
#define GPU_SLOW_ARGS_CPX_1S_1	GPU_SLOW_ARGS_CPX_1, GPU_ARGS_CPX_1S

#define GPU_FAST_ARGS_1S_2	GPU_FAST_ARGS_2, GPU_ARGS_1S
#define GPU_EQSP_ARGS_1S_2	GPU_EQSP_ARGS_2, GPU_ARGS_1S
#define GPU_SLOW_ARGS_1S_2	GPU_SLOW_ARGS_2, GPU_ARGS_1S

#define GPU_FAST_ARGS_1S_4	GPU_FAST_ARGS_4, GPU_ARGS_1S
#define GPU_EQSP_ARGS_1S_4	GPU_EQSP_ARGS_4, GPU_ARGS_1S
#define GPU_SLOW_ARGS_1S_4	GPU_SLOW_ARGS_4, GPU_ARGS_1S

#define GPU_FAST_ARGS_2S_1	GPU_FAST_ARGS_1, GPU_ARGS_2S
#define GPU_EQSP_ARGS_2S_1	GPU_EQSP_ARGS_1, GPU_ARGS_2S
#define GPU_SLOW_ARGS_2S_1	GPU_SLOW_ARGS_1, GPU_ARGS_2S

#define GPU_FAST_ARGS_2S_3	GPU_FAST_ARGS_3, GPU_ARGS_2S
#define GPU_EQSP_ARGS_2S_3	GPU_EQSP_ARGS_3, GPU_ARGS_2S
#define GPU_SLOW_ARGS_2S_3	GPU_SLOW_ARGS_3, GPU_ARGS_2S

#define GPU_FAST_ARGS_3S_1	GPU_FAST_ARGS_1, GPU_ARGS_3S
#define GPU_EQSP_ARGS_3S_1	GPU_EQSP_ARGS_1, GPU_ARGS_3S
#define GPU_SLOW_ARGS_3S_1	GPU_SLOW_ARGS_1, GPU_ARGS_3S

#define GPU_FAST_ARGS_3S_2	GPU_FAST_ARGS_2, GPU_ARGS_3S
#define GPU_EQSP_ARGS_3S_2	GPU_EQSP_ARGS_2, GPU_ARGS_3S
#define GPU_SLOW_ARGS_3S_2	GPU_SLOW_ARGS_2, GPU_ARGS_3S


#define SHOW_FAST_ARGS_3				\
sprintf(DEFAULT_ERROR_STRING,"\tdst = 0x%lx   s1 = 0x%lx   s2 = 0x%lx   len = %d",\
(int_for_addr)vap->va_dst_vp,(int_for_addr)vap->va_src_vp[0],\
(int_for_addr)vap->va_src_vp[1],vap->va_len);\
advise(DEFAULT_ERROR_STRING);

#define DECL_EQSP_INCRS_1	int dst_incr;
#define DECL_EQSP_INCRS_SRC1	int s1_incr;
#define DECL_EQSP_INCRS_SRC2	int s2_incr;
#define DECL_EQSP_INCRS_SRC3	int s3_incr;
#define DECL_EQSP_INCRS_SRC4	int s4_incr;
#define DECL_EQSP_INCRS_SBM	int bm_incr;
#define DECL_EQSP_INCRS_DBM	int bm_incr;
#define DECL_EQSP_INCRS_2	DECL_EQSP_INCRS_1 DECL_EQSP_INCRS_SRC1
#define DECL_EQSP_INCRS_3	DECL_EQSP_INCRS_2 DECL_EQSP_INCRS_SRC2
#define DECL_EQSP_INCRS_4	DECL_EQSP_INCRS_3 DECL_EQSP_INCRS_SRC3
#define DECL_EQSP_INCRS_5	DECL_EQSP_INCRS_4 DECL_EQSP_INCRS_SRC4
#define DECL_EQSP_INCRS_2SRCS	DECL_EQSP_INCRS_SRC1 DECL_EQSP_INCRS_SRC2

#define DECL_EQSP_INCRS_SBM_1	DECL_EQSP_INCRS_1 DECL_EQSP_INCRS_SBM
#define DECL_EQSP_INCRS_SBM_2	DECL_EQSP_INCRS_2 DECL_EQSP_INCRS_SBM
#define DECL_EQSP_INCRS_SBM_3	DECL_EQSP_INCRS_3 DECL_EQSP_INCRS_SBM
#define DECL_EQSP_INCRS_DBM_		DECL_EQSP_INCRS_DBM
#define DECL_EQSP_INCRS_DBM_1SRC	DECL_EQSP_INCRS_SRC1 DECL_EQSP_INCRS_DBM
#define DECL_EQSP_INCRS_DBM_2SRCS	DECL_EQSP_INCRS_2SRCS DECL_EQSP_INCRS_DBM

#define DECL_SLOW_INCRS_1	dim3 dst_incr;		\
				int dim_indices[3];
#define DECL_SLOW_INCRS_SRC1	dim3 s1_incr;
#define DECL_SLOW_INCRS_SRC2	dim3 s2_incr;
#define DECL_SLOW_INCRS_SRC3	dim3 s3_incr;
#define DECL_SLOW_INCRS_SRC4	dim3 s4_incr;
#define DECL_SLOW_INCRS_SBM	dim3 bm_incr;
#define DECL_SLOW_INCRS_DBM	dim3 bm_incr;		\
				int dim_indices[3];
#define DECL_SLOW_INCRS_2	DECL_SLOW_INCRS_1 DECL_SLOW_INCRS_SRC1
#define DECL_SLOW_INCRS_3	DECL_SLOW_INCRS_2 DECL_SLOW_INCRS_SRC2
#define DECL_SLOW_INCRS_4	DECL_SLOW_INCRS_3 DECL_SLOW_INCRS_SRC3
#define DECL_SLOW_INCRS_5	DECL_SLOW_INCRS_4 DECL_SLOW_INCRS_SRC4
#define DECL_SLOW_INCRS_2SRCS	DECL_SLOW_INCRS_SRC1 DECL_SLOW_INCRS_SRC2
#define DECL_SLOW_INCRS_1SRC	DECL_SLOW_INCRS_SRC1
#define DECL_SLOW_INCRS_SBM_1	DECL_SLOW_INCRS_1 DECL_SLOW_INCRS_SBM
#define DECL_SLOW_INCRS_SBM_2	DECL_SLOW_INCRS_2 DECL_SLOW_INCRS_SBM
#define DECL_SLOW_INCRS_SBM_3	DECL_SLOW_INCRS_3 DECL_SLOW_INCRS_SBM
#define DECL_SLOW_INCRS_DBM_		DECL_SLOW_INCRS_DBM
#define DECL_SLOW_INCRS_DBM_1SRC	DECL_SLOW_INCRS_1SRC DECL_SLOW_INCRS_DBM
#define DECL_SLOW_INCRS_DBM_2SRCS	DECL_SLOW_INCRS_2SRCS DECL_SLOW_INCRS_DBM

#define GET_EQSP_INCR(inc_var,inc_array)		\
	for(i_dim=0;i_dim<N_DIMENSIONS;i_dim++){	\
		if( inc_array[i_dim] != 0 ){		\
			inc_var = inc_array[i_dim];	\
			i_dim=N_DIMENSIONS+1;		\
		}					\
	}

#define SHOW_EQSP_INCRS_3				\
sprintf(DEFAULT_ERROR_STRING,"dst_incr = %d   s1_incr = %d   s2_incr = %d",\
dst_incr,s1_incr,s2_incr);\
advise(DEFAULT_ERROR_STRING);

#define GET_EQSP_INCR_1		GET_EQSP_INCR(dst_incr,vap->va_spi_p->spi_dst_incr)
#define GET_EQSP_INCR_SRC1	GET_EQSP_INCR(s1_incr,vap->va_spi_p->spi_src_incr[0])
#define GET_EQSP_INCR_SRC2	GET_EQSP_INCR(s2_incr,vap->va_spi_p->spi_src_incr[1])
#define GET_EQSP_INCR_SRC3	GET_EQSP_INCR(s3_incr,vap->va_spi_p->spi_src_incr[2])
#define GET_EQSP_INCR_SRC4	GET_EQSP_INCR(s4_incr,vap->va_spi_p->spi_src_incr[3])
#define GET_EQSP_INCR_SBM	GET_EQSP_INCR(bm_incr,vap->va_spi_p->spi_src_incr[4])
#define GET_EQSP_INCR_DBM	GET_EQSP_INCR(bm_incr,vap->va_spi_p->spi_dst_incr)

#define GET_EQSP_INCR_2		GET_EQSP_INCR_1 GET_EQSP_INCR_SRC1
#define GET_EQSP_INCR_3		GET_EQSP_INCR_2 GET_EQSP_INCR_SRC2
#define GET_EQSP_INCR_4		GET_EQSP_INCR_3 GET_EQSP_INCR_SRC3
#define GET_EQSP_INCR_5		GET_EQSP_INCR_4 GET_EQSP_INCR_SRC4
#define GET_EQSP_INCR_2SRCS	GET_EQSP_INCR_SRC1 GET_EQSP_INCR_SRC2
#define GET_EQSP_INCR_1SRC	GET_EQSP_INCR_SRC1

#define GET_EQSP_INCR_SBM_1	GET_EQSP_INCR_1 GET_EQSP_INCR_SBM
#define GET_EQSP_INCR_SBM_2	GET_EQSP_INCR_2 GET_EQSP_INCR_SBM
#define GET_EQSP_INCR_SBM_3	GET_EQSP_INCR_3 GET_EQSP_INCR_SBM
#define GET_EQSP_INCR_DBM_	GET_EQSP_INCR_DBM
#define GET_EQSP_INCR_DBM_1SRC	GET_EQSP_INCR_1SRC GET_EQSP_INCR_DBM
#define GET_EQSP_INCR_DBM_2SRCS	GET_EQSP_INCR_2SRCS GET_EQSP_INCR_DBM


#define GPU_EQSP_ARGS_1		(dest_type*)vap->va_dst_vp, dst_incr
#define GPU_EQSP_ARGS_SRC1	(std_type*)vap->va_src_vp[0], s1_incr
#define GPU_EQSP_ARGS_SRC2	(std_type*)vap->va_src_vp[1], s2_incr
#define GPU_EQSP_ARGS_SRC3	(std_type*)vap->va_src_vp[2], s3_incr
#define GPU_EQSP_ARGS_SRC4	(std_type*)vap->va_src_vp[3], s4_incr

#define GPU_EQSP_ARGS_CPX_1	(dest_cpx*)vap->va_dst_vp, dst_incr
#define GPU_EQSP_ARGS_CPX_SRC1	(std_cpx*)vap->va_src_vp[0], s1_incr
#define GPU_EQSP_ARGS_CPX_SRC2	(std_cpx*)vap->va_src_vp[1], s2_incr
#define GPU_EQSP_ARGS_CPX_SRC3	(std_cpx*)vap->va_src_vp[2], s3_incr
#define GPU_EQSP_ARGS_CPX_SRC4	(std_cpx*)vap->va_src_vp[3], s4_incr

#define GPU_EQSP_ARGS_2		GPU_EQSP_ARGS_1, GPU_EQSP_ARGS_SRC1
#define GPU_EQSP_ARGS_3		GPU_EQSP_ARGS_2, GPU_EQSP_ARGS_SRC2
#define GPU_EQSP_ARGS_4		GPU_EQSP_ARGS_3, GPU_EQSP_ARGS_SRC3
#define GPU_EQSP_ARGS_5		GPU_EQSP_ARGS_4, GPU_EQSP_ARGS_SRC4

#define GPU_EQSP_ARGS_CPX_2	GPU_EQSP_ARGS_CPX_1, GPU_EQSP_ARGS_CPX_SRC1
#define GPU_EQSP_ARGS_CPX_3	GPU_EQSP_ARGS_CPX_2, GPU_EQSP_ARGS_CPX_SRC2
#define GPU_EQSP_ARGS_CPX_4	GPU_EQSP_ARGS_CPX_3, GPU_EQSP_ARGS_CPX_SRC3
#define GPU_EQSP_ARGS_CPX_5	GPU_EQSP_ARGS_CPX_4, GPU_EQSP_ARGS_CPX_SRC4

#define GPU_EQSP_ARGS_2SRCS	GPU_EQSP_ARGS_SRC1, GPU_EQSP_ARGS_SRC2

#define GPU_EQSP_LEN_ARG	vap->va_len
#define GPU_ELEN_ARGS_1	GPU_EQSP_ARGS_1, GPU_EQSP_LEN_ARG
#define GPU_ELEN_ARGS_2	GPU_EQSP_ARGS_2, GPU_EQSP_LEN_ARG
#define GPU_ELEN_ARGS_3	GPU_EQSP_ARGS_3, GPU_EQSP_LEN_ARG
#define GPU_ELEN_ARGS_4	GPU_EQSP_ARGS_4, GPU_EQSP_LEN_ARG
#define GPU_ELEN_ARGS_5	GPU_EQSP_ARGS_5, GPU_EQSP_LEN_ARG

#define GPU_ELEN_ARGS_CPX_1	GPU_EQSP_ARGS_CPX_1, GPU_EQSP_LEN_ARG
#define GPU_ELEN_ARGS_CPX_2	GPU_EQSP_ARGS_CPX_2, GPU_EQSP_LEN_ARG
#define GPU_ELEN_ARGS_CPX_3	GPU_EQSP_ARGS_CPX_3, GPU_EQSP_LEN_ARG
#define GPU_ELEN_ARGS_CPX_4	GPU_EQSP_ARGS_CPX_4, GPU_EQSP_LEN_ARG
#define GPU_ELEN_ARGS_CPX_5	GPU_EQSP_ARGS_CPX_5, GPU_EQSP_LEN_ARG

#define GPU_SLOW_ARGS_1		(dest_type*)vap->va_dst_vp, dst_incr
#define GPU_SLOW_ARGS_SRC1	(std_type*)vap->va_src_vp[0], s1_incr
#define GPU_SLOW_ARGS_SRC2	(std_type*)vap->va_src_vp[1], s2_incr
#define GPU_SLOW_ARGS_SRC3	(std_type*)vap->va_src_vp[2], s3_incr
#define GPU_SLOW_ARGS_SRC4	(std_type*)vap->va_src_vp[3], s4_incr

#define GPU_SLOW_ARGS_CPX_1	(dest_cpx*)vap->va_dst_vp, dst_incr
#define GPU_SLOW_ARGS_CPX_SRC1	(std_cpx*)vap->va_src_vp[0], s1_incr
#define GPU_SLOW_ARGS_CPX_SRC2	(std_cpx*)vap->va_src_vp[1], s2_incr
#define GPU_SLOW_ARGS_CPX_SRC3	(std_cpx*)vap->va_src_vp[2], s3_incr
#define GPU_SLOW_ARGS_CPX_SRC4	(std_cpx*)vap->va_src_vp[3], s4_incr

#define GPU_SLOW_ARGS_2		GPU_SLOW_ARGS_1, GPU_SLOW_ARGS_SRC1
#define GPU_SLOW_ARGS_3		GPU_SLOW_ARGS_2, GPU_SLOW_ARGS_SRC2
#define GPU_SLOW_ARGS_4		GPU_SLOW_ARGS_3, GPU_SLOW_ARGS_SRC3
#define GPU_SLOW_ARGS_5		GPU_SLOW_ARGS_4, GPU_SLOW_ARGS_SRC4

#define GPU_SLOW_ARGS_CPX_2	GPU_SLOW_ARGS_CPX_1, GPU_SLOW_ARGS_CPX_SRC1
#define GPU_SLOW_ARGS_CPX_3	GPU_SLOW_ARGS_CPX_2, GPU_SLOW_ARGS_CPX_SRC2
#define GPU_SLOW_ARGS_CPX_4	GPU_SLOW_ARGS_CPX_3, GPU_SLOW_ARGS_CPX_SRC3
#define GPU_SLOW_ARGS_CPX_5	GPU_SLOW_ARGS_CPX_4, GPU_SLOW_ARGS_CPX_SRC4

#define GPU_SLOW_ARGS_2SRCS	GPU_SLOW_ARGS_SRC1, GPU_SLOW_ARGS_SRC2

#define GPU_SLOW_LEN_ARG	len
#define GPU_SLEN_ARGS_1	GPU_SLOW_ARGS_1, GPU_SLOW_LEN_ARG
#define GPU_SLEN_ARGS_2	GPU_SLOW_ARGS_2, GPU_SLOW_LEN_ARG
#define GPU_SLEN_ARGS_3	GPU_SLOW_ARGS_3, GPU_SLOW_LEN_ARG
#define GPU_SLEN_ARGS_4	GPU_SLOW_ARGS_4, GPU_SLOW_LEN_ARG
#define GPU_SLEN_ARGS_5	GPU_SLOW_ARGS_5, GPU_SLOW_LEN_ARG

#define GPU_SLEN_ARGS_CPX_1	GPU_SLOW_ARGS_CPX_1, GPU_SLOW_LEN_ARG
#define GPU_SLEN_ARGS_CPX_2	GPU_SLOW_ARGS_CPX_2, GPU_SLOW_LEN_ARG
#define GPU_SLEN_ARGS_CPX_3	GPU_SLOW_ARGS_CPX_3, GPU_SLOW_LEN_ARG
#define GPU_SLEN_ARGS_CPX_4	GPU_SLOW_ARGS_CPX_4, GPU_SLOW_LEN_ARG
#define GPU_SLEN_ARGS_CPX_5	GPU_SLOW_ARGS_CPX_5, GPU_SLOW_LEN_ARG

#define XFER_FAST_LEN		len.x=vap->va_len;

// XFER_SLOW_LEN takes the dimension arrays (up to 5 dieensions)
// and figures out which 3 to use, putting the dimensions in len
// and which were chosen in dim_indices.
// There is a problem with complex numbers, as the type dimension
// is 2...
//
// These are only the source dimensions.

#define XFER_SLOW_LEN_RC_N(n)					\
								\
if( setup_slow_len(&len,vap->va_szi_p,1,dim_indices,0,n) < 0 )	\
	return;

#define XFER_SLOW_LEN_CPX_N(n)					\
								\
if( setup_slow_len(&len,vap->va_szi_p,1,dim_indices,0,n) < 0 )	\
	return;

#define XFER_SLOW_LEN_N(n)					\
								\
if( setup_slow_len(&len,vap->va_szi_p,0,dim_indices,0,n) < 0 )	\
	return;

// The way this is used, it appears we skip the first source - why???
// 1SRC and 2SRCS are used with destination bitmaps???

#define XFER_SLOW_LEN_F(first,n)					\
									\
if( setup_slow_len(&len,vap->va_szi_p,0,dim_indices,first,n) < 0 )	\
	return;

#define XFER_SLOW_LEN_CPX_1	XFER_SLOW_LEN_1

#define XFER_SLOW_LEN_1		XFER_SLOW_LEN_N(1)
#define XFER_SLOW_LEN_2		XFER_SLOW_LEN_N(2)
#define XFER_SLOW_LEN_RC_2	XFER_SLOW_LEN_RC_N(2)
#define XFER_SLOW_LEN_3		XFER_SLOW_LEN_N(3)
#define XFER_SLOW_LEN_CPX_3	XFER_SLOW_LEN_CPX_N(3)
#define XFER_SLOW_LEN_4		XFER_SLOW_LEN_N(4)
#define XFER_SLOW_LEN_5		XFER_SLOW_LEN_N(5)
#define XFER_SLOW_LEN_		XFER_SLOW_LEN_F(1,1)
#define XFER_SLOW_LEN_1SRC	XFER_SLOW_LEN_F(1,1)
#define XFER_SLOW_LEN_2SRCS	XFER_SLOW_LEN_F(1,2)

/* For slow loops, we currently only iterate over two dimensions (x and y),
 * although in principle we should be able to handle 3...
 * We need to determine which 2 by examining the dimensions of the vectors.
 */

#define SETUP_SLOW_INCRS(var,inc_array)			\
	SETUP_INC_IF(var.x,inc_array,0)		\
	SETUP_INC_IF(var.y,inc_array,1)		\
	SETUP_INC_IF(var.z,inc_array,2)

#define SETUP_INC_IF( var ,inc_array, which_index )	\
	if( dim_indices[which_index] < 0 )		\
		var = 0;				\
	else						\
		var = inc_array[dim_indices[which_index]];

/*
sprintf(DEFAULT_ERROR_STRING,"SETUP_INC_IF:  %s = %d, dim_indices[%d] = %d",	\
#var,var,which_index,dim_indices[which_index]);				\
advise(DEFAULT_ERROR_STRING);
*/


#define SETUP_SLOW_INCRS_1		SETUP_SLOW_INCRS(dst_incr,vap->va_spi_p->spi_dst_incr)
#define SETUP_SLOW_INCRS_SRC1		SETUP_SLOW_INCRS(s1_incr,vap->va_spi_p->spi_src_incr[0])
#define SETUP_SLOW_INCRS_SRC2		SETUP_SLOW_INCRS(s2_incr,vap->va_spi_p->spi_src_incr[1])
#define SETUP_SLOW_INCRS_SRC3		SETUP_SLOW_INCRS(s2_incr,vap->va_spi_p->spi_src_incr[2])
#define SETUP_SLOW_INCRS_SRC4		SETUP_SLOW_INCRS(s2_incr,vap->va_spi_p->spi_src_incr[3])
#define SETUP_SLOW_INCRS_SBM		SETUP_SLOW_INCRS(bm_incr,vap->va_spi_p->spi_src_incr[4])
#define SETUP_SLOW_INCRS_DBM		SETUP_SLOW_INCRS(bm_incr,vap->va_spi_p->spi_dst_incr)

#define SETUP_SLOW_INCRS_2		SETUP_SLOW_INCRS_1 SETUP_SLOW_INCRS_SRC1
#define SETUP_SLOW_INCRS_3		SETUP_SLOW_INCRS_2 SETUP_SLOW_INCRS_SRC2
#define SETUP_SLOW_INCRS_4		SETUP_SLOW_INCRS_3 SETUP_SLOW_INCRS_SRC3
#define SETUP_SLOW_INCRS_5		SETUP_SLOW_INCRS_4 SETUP_SLOW_INCRS_SRC4
#define SETUP_SLOW_INCRS_2SRCS		SETUP_SLOW_INCRS_SRC1 SETUP_SLOW_INCRS_SRC2
#define SETUP_SLOW_INCRS_1SRC		SETUP_SLOW_INCRS_SRC1
#define SETUP_SLOW_INCRS_SBM_1		SETUP_SLOW_INCRS_1 SETUP_SLOW_INCRS_SBM
#define SETUP_SLOW_INCRS_SBM_2		SETUP_SLOW_INCRS_2 SETUP_SLOW_INCRS_SBM
#define SETUP_SLOW_INCRS_SBM_3		SETUP_SLOW_INCRS_3 SETUP_SLOW_INCRS_SBM
#define SETUP_SLOW_INCRS_DBM_		SETUP_SLOW_INCRS_DBM
#define SETUP_SLOW_INCRS_DBM_1SRC	SETUP_SLOW_INCRS_1SRC SETUP_SLOW_INCRS_DBM
#define SETUP_SLOW_INCRS_DBM_2SRCS	SETUP_SLOW_INCRS_2SRCS SETUP_SLOW_INCRS_DBM

#define SHOW_INCS_1		SHOW_INCS(dst_incr);
#define SHOW_INCS_SRC1		SHOW_INCS(s1_incr);
#define SHOW_INCS_SRC2		SHOW_INCS(s2_incr);
#define SHOW_INCS_SRC3		SHOW_INCS(s2_incr);
#define SHOW_INCS_SRC4		SHOW_INCS(s2_incr);
#define SHOW_INCS_2		SHOW_INCS_1 SHOW_INCS_SRC1
#define SHOW_INCS_3		SHOW_INCS_2 SHOW_INCS_SRC2
#define SHOW_INCS_4		SHOW_INCS_3 SHOW_INCS_SRC3
#define SHOW_INCS_5		SHOW_INCS_4 SHOW_INCS_SRC4

#define SHOW_INCS(v)		\
sprintf(DEFAULT_ERROR_STRING,"%s = %d  %d  %d",#v,v.x,v.y,v.z);\
advise(DEFAULT_ERROR_STRING);


#define GENERIC_HOST_CALL(name,code,bitmap,typ,scalars,vectors)		\
									\
GENERIC_HOST_FAST_CALL(name,code,bitmap,typ,scalars,vectors)		\
GENERIC_HOST_EQSP_CALL(name,code,bitmap,typ,scalars,vectors)		\
GENERIC_HOST_SLOW_CALL(name,code,bitmap,typ,scalars,vectors)		\
									\
GENERIC_HOST_SWITCH(name,code,bitmap,typ,scalars,vectors)

#define SLOW_HOST_CALL(name,code,bitmap,typ,scalars,vectors)	\
								\
GENERIC_HOST_SLOW_CALL(name,code,bitmap,typ,scalars,vectors)	\
								\
void HOST_CALL_NAME(name,code)(Vec_Obj_Args *oap)		\
{								\
	Vector_Args va1;					\
	Spacing_Info spi1;					\
	Size_Info szi1;						\
								\
	va1.va_spi_p = &spi1;					\
	va1.va_szi_p = &szi1;					\
	XFER_SLOW_ARGS_##bitmap##typ##scalars##vectors		\
	CHAIN_CHECK( GPU_SLOW_CALL_NAME(name,code) )		\
}


#define GENERIC_HOST_FAST_CALL(name,code,bitmap,typ,scalars,vectors)	\
									\
void GPU_FAST_CALL_NAME(name,code)(Vector_Args *vap)			\
{									\
	BLOCK_VARS_DECLS						\
	dim3 len;							\
	DECLS_##bitmap							\
									\
	CLEAR_CUDA_ERROR(GPU_FAST_CALL_NAME(name,code))			\
	XFER_FAST_LEN							\
	GET_THREADS_PER_##bitmap##BLOCK					\
REPORT_FAST_ARGS_##bitmap##typ##scalars##vectors			\
	if (extra.x != 0) {						\
		n_blocks.x++;						\
		REPORT_THREAD_INFO					\
		KERN_FLEN_NAME(name,code)<<< NN_GPU >>> 		\
			(GPU_FLEN_ARGS_##bitmap##typ##scalars##vectors );\
	} else {							\
		REPORT_THREAD_INFO					\
		KERN_FAST_NAME(name,code)<<< NN_GPU >>>			\
			(GPU_FAST_ARGS_##bitmap##typ##scalars##vectors );\
		/* BUG?  should we put this check everywhere? */	\
    		cutilCheckMsg("kernel launch failure #1");		\
	}								\
}

#define GENERIC_HOST_EQSP_CALL(name,code,bitmap,typ,scalars,vectors)	\
									\
void GPU_EQSP_CALL_NAME(name,code)(Vector_Args *vap)			\
{									\
	BLOCK_VARS_DECLS						\
	dim3 len;							\
	DECL_EQSP_INCRS_##bitmap##vectors				\
	int i_dim;							\
									\
	CLEAR_CUDA_ERROR(GPU_EQSP_CALL_NAME(name,code))			\
	XFER_FAST_LEN							\
	GET_THREADS_PER_##bitmap##BLOCK					\
	/*SETUP_SIMPLE_INCS_##vectors*/					\
	GET_EQSP_INCR_##bitmap##vectors					\
	if (extra.x != 0) {						\
		n_blocks.x++;						\
		REPORT_THREAD_INFO					\
		KERN_ELEN_NAME(name,code)<<< NN_GPU >>> 		\
			(GPU_ELEN_ARGS_##bitmap##typ##scalars##vectors );\
	} else {							\
		REPORT_THREAD_INFO					\
		KERN_EQSP_NAME(name,code)<<< NN_GPU >>>			\
			(GPU_EQSP_ARGS_##bitmap##typ##scalars##vectors );\
		/* BUG?  should we put this check everywhere? */	\
    		cutilCheckMsg("kernel launch failure #2");		\
	}								\
}

#define REPORT_FAST_ARGS_1
#define REPORT_FAST_ARGS_2
#define REPORT_FAST_ARGS_3
#define REPORT_FAST_ARGS_CPX_3
#define REPORT_FAST_ARGS_CPX_1S_1
#define REPORT_FAST_ARGS_4
#define REPORT_FAST_ARGS_5
#define REPORT_FAST_ARGS_1S_1
#define REPORT_FAST_ARGS_2S_1
#define REPORT_FAST_ARGS_3S_1
#define REPORT_FAST_ARGS_1S_4
#define REPORT_FAST_ARGS_2S_3
#define REPORT_FAST_ARGS_3S_2
#define REPORT_FAST_ARGS_SBM_3
#define REPORT_FAST_ARGS_SBM_1S_2
#define REPORT_FAST_ARGS_SBM_2S_1

#ifdef MORE_DEBUG

#define REPORT_ARGS_1S_1		\
sprintf(DEFAULT_ERROR_STRING,"dst = 0x%lx sval = %g",\
(int_for_addr)vap->va_dst_vp,vap->va_sval[0].std_scalar);\
advise(DEFAULT_ERROR_STRING);


#define REPORT_FAST_ARGS_1S_2		\
sprintf(DEFAULT_ERROR_STRING,"dst = 0x%lx src = 0x%lx  sval = %g",\
(int_for_addr)vap->va_dst_vp,(int_for_addr)vap->va_src_vp[0],vap->va_sval[0].std_scalar);\
advise(DEFAULT_ERROR_STRING);


#define REPORT_FAST_ARGS_DBM_1S_	\
sprintf(DEFAULT_ERROR_STRING,"dbm_ptr = 0x%lx bit0 = %d  sval = 0x%x",\
(int_for_addr)vap->va_dst_vp,vap->va_bit0,vap->va_sval[0].bitmap_scalar);\
advise(DEFAULT_ERROR_STRING);

#define REPORT_FAST_ARGS_DBM_1S_1SRC		\
sprintf(DEFAULT_ERROR_STRING,"dbm_ptr = 0x%lx  s1_ptr = 0x%lx",\
(int_for_addr)vap->va_dst_vp,vap->va_src_vp[0]);\
advise(DEFAULT_ERROR_STRING);

#define REPORT_FAST_ARGS_DBM_2SRCS		\
sprintf(DEFAULT_ERROR_STRING,"dbm_ptr = 0x%lx  s1_ptr = 0x%lx  s2_ptr = 0x%lx",\
(int_for_addr)vap->va_dst_vp,vap->va_src_vp[0],vap->va_src_vp[1]);\
advise(DEFAULT_ERROR_STRING);

#define REPORT_FAST_ARGS_RC_2		\
sprintf(DEFAULT_ERROR_STRING,"dbm_ptr = 0x%lx  s1_ptr = 0x%lx",\
(int_for_addr)vap->va_dst_vp,(int_for_addr)vap->va_src_vp[0]);\
advise(DEFAULT_ERROR_STRING);

#else /* ! MORE_DEBUG */

#define REPORT_ARGS_1S_1
#define REPORT_FAST_ARGS_1S_2
#define REPORT_FAST_ARGS_DBM_1S_
#define REPORT_FAST_ARGS_DBM_1S_1SRC
#define REPORT_FAST_ARGS_DBM_2SRCS
#define REPORT_FAST_ARGS_RC_2

#endif /* ! MORE_DEBUG */

#define REPORT_ARGS_1
#define REPORT_ARGS_2
#define REPORT_ARGS_3
#define REPORT_ARGS_CPX_3
#define REPORT_ARGS_CPX_1S_1
#define REPORT_ARGS_4
#define REPORT_ARGS_5
#define REPORT_ARGS_2S_1
#define REPORT_ARGS_3S_1
#define REPORT_ARGS_1S_2
#define REPORT_ARGS_1S_4
#define REPORT_ARGS_2S_3
#define REPORT_ARGS_3S_2
#define REPORT_ARGS_SBM_3
#define REPORT_ARGS_SBM_1S_2
#define REPORT_ARGS_SBM_2S_1

#ifdef MORE_DEBUG


#define REPORT_ARGS_DBM_1S_	\
sprintf(DEFAULT_ERROR_STRING,"dbm_ptr = 0x%lx bit0 = %d  sval = 0x%x",\
(int_for_addr)vap->va_dst_vp,vap->va_bit0,vap->va_sval[0].bitmap_scalar);\
advise(DEFAULT_ERROR_STRING);\
REPORT_INCR(bm_incr) \
REPORT_INCR(len)

#define REPORT_ARGS_DBM_1S_1SRC		\
sprintf(DEFAULT_ERROR_STRING,"dbm_ptr = 0x%lx  s1_ptr = 0x%lx",\
(int_for_addr)vap->va_dst_vp,vap->va_src_vp[0]);\
advise(DEFAULT_ERROR_STRING);\
REPORT_INCR(bm_incr)\
REPORT_INCR(s1_incr)

#define REPORT_ARGS_DBM_2SRCS		\
sprintf(DEFAULT_ERROR_STRING,"dbm_ptr = 0x%lx  s1_ptr = 0x%lx  s2_ptr = 0x%lx",\
(int_for_addr)vap->va_dst_vp,vap->va_src_vp[0],vap->va_src_vp[1]);\
advise(DEFAULT_ERROR_STRING);\
REPORT_INCR(bm_incr)\
REPORT_INCR(s1_incr)\
REPORT_INCR(s2_incr)

#define REPORT_ARGS_RC_2		\
sprintf(DEFAULT_ERROR_STRING,"dbm_ptr = 0x%lx  s1_ptr = 0x%lx",\
(int_for_addr)vap->va_dst_vp,(int_for_addr)vap->va_src_vp[0]);\
advise(DEFAULT_ERROR_STRING);\
REPORT_INCR(dst_incr)\
REPORT_INCR(s1_incr)\

#define REPORT_INCR(v)		\
sprintf(DEFAULT_ERROR_STRING,"%s = %d %d %d",#v,v.x,v.y,v.z);\
advise(DEFAULT_ERROR_STRING);

#else /* ! MORE_DEBUG */

#define REPORT_ARGS_DBM_1S_
#define REPORT_ARGS_DBM_1S_1SRC
#define REPORT_ARGS_DBM_2SRCS
#define REPORT_ARGS_RC_2
#define REPORT_INCR(v)

#endif /* ! MORE_DEBUG */


#define GENERIC_HOST_SLOW_CALL(name,code,bitmap,typ,scalars,vectors)	\
									\
void GPU_SLOW_CALL_NAME(name,code)(Vector_Args *vap)			\
{									\
	BLOCK_VARS_DECLS						\
	DECL_SLOW_INCRS_##bitmap##vectors				\
	dim3 len;							\
									\
	CLEAR_CUDA_ERROR(GPU_SLOW_CALL_NAME(name,code))			\
	XFER_SLOW_LEN_##typ##vectors					\
	SETUP_BLOCKS_XYZ_##bitmap	/* using len - was _XY */	\
	SETUP_SLOW_INCRS_##bitmap##vectors				\
	REPORT_THREAD_INFO						\
REPORT_ARGS_##bitmap##typ##scalars##vectors				\
	/* We don't bother with separate non-len function... */		\
	KERN_SLEN_NAME(name,code)<<< NN_GPU >>>				\
		(GPU_SLEN_ARGS_##bitmap##typ##scalars##vectors );	\
    	cutilCheckMsg("kernel launch failure #3");			\
}

#ifdef MORE_DEBUG
#define REPORT_SWITCH(name,sw)						\
sprintf(DEFAULT_ERROR_STRING,"Calling %s version of %s",#sw,#name);		\
advise(DEFAULT_ERROR_STRING);
#else /* ! MORE_DEBUG */
#define REPORT_SWITCH(name,sw)
#endif /* ! MORE_DEBUG */

#define GENERIC_HOST_SWITCH(name,code,bitmap,typ,scalars,vectors)	\
									\
void HOST_CALL_NAME(name,code)(Vec_Obj_Args *oap)			\
{									\
	Vector_Args va1;						\
	Spacing_Info spi1;						\
	Size_Info szi1;							\
									\
	if( FAST_TEST_##bitmap##typ##vectors ){				\
REPORT_SWITCH(name,fast)						\
		XFER_FAST_ARGS_##bitmap##typ##scalars##vectors		\
		CHAIN_CHECK( GPU_FAST_CALL_NAME(name,code) )		\
	} else if( EQSP_TEST_##bitmap##typ##vectors ){			\
REPORT_SWITCH(name,eqsp)						\
		va1.va_spi_p = &spi1;					\
		XFER_EQSP_ARGS_##bitmap##typ##scalars##vectors		\
		CHAIN_CHECK( GPU_EQSP_CALL_NAME(name,code) )		\
	} else {							\
REPORT_SWITCH(name,slow)						\
		va1.va_spi_p = &spi1;					\
		va1.va_szi_p = &szi1;					\
		XFER_SLOW_ARGS_##bitmap##typ##scalars##vectors		\
		CHAIN_CHECK( GPU_SLOW_CALL_NAME(name,code) )		\
	}								\
}

#define DECLS_
#define DECLS_SBM_	DECLS_SBM
#define DECLS_DBM_	DECLS_DBM
#define DECLS_SBM
#define DECLS_DBM

#define DECLS_SBM_1	DECLS_SBM DECL_TMP_ARGS_1
#define DECLS_SBM_2	DECLS_SBM DECL_TMP_ARGS_2
#define DECLS_SBM_3	DECLS_SBM DECL_TMP_ARGS_3

/* These args are used for destination bitmaps... */
#define DECL_TMP_ARGS_1		std_type *arg1;
#define DECL_TMP_ARGS_SRC1	std_type *arg2;
#define DECL_TMP_ARGS_SRC2	std_type *arg3;
#define DECL_TMP_ARGS_2		DECL_TMP_ARGS_1 DECL_TMP_ARGS_SRC1
#define DECL_TMP_ARGS_3		DECL_TMP_ARGS_2 DECL_TMP_ARGS_SRC2

#define ADVANCE_FAST_ARGS_1	arg1 ++;
#define ADVANCE_FAST_ARGS_SRC1	arg2 ++;
#define ADVANCE_FAST_ARGS_SRC2	arg3 ++;
#define ADVANCE_FAST_ARGS_2	ADVANCE_FAST_ARGS_1 ADVANCE_FAST_ARGS_SRC1
#define ADVANCE_FAST_ARGS_3	ADVANCE_FAST_ARGS_2 ADVANCE_FAST_ARGS_SRC2

#define ADVANCE_EQSP_ARGS_1	arg1 += dst_incr;
#define ADVANCE_EQSP_ARGS_SRC1	arg2 += s1_incr;
#define ADVANCE_EQSP_ARGS_SRC2	arg3 += s2_incr;
#define ADVANCE_EQSP_ARGS_2	ADVANCE_EQSP_ARGS_1 ADVANCE_EQSP_ARGS_SRC1
#define ADVANCE_EQSP_ARGS_3	ADVANCE_EQSP_ARGS_2 ADVANCE_EQSP_ARGS_SRC2

#define ADVANCE_SLOW_ARGS_1	arg1 += dst_incr.x;
#define ADVANCE_SLOW_ARGS_SRC1	arg2 += s1_incr.x;
#define ADVANCE_SLOW_ARGS_SRC2	arg3 += s2_incr.x;
#define ADVANCE_SLOW_ARGS_2	ADVANCE_SLOW_ARGS_1 ADVANCE_SLOW_ARGS_SRC1
#define ADVANCE_SLOW_ARGS_3	ADVANCE_SLOW_ARGS_2 ADVANCE_SLOW_ARGS_SRC2


// Minmax functions
//
// the basic GPU operation is to compare two input pixels and put the result in a third
// There is only one input object, but we treat it as two.
// For now, we assume contiguous - or at least evenly spaced
//
// Here is the algorithm:
// Say we have 9 elements...  make a temp vector (t) with 5 elements.
// Split the input into two halves, a (5) and b (4)
// then t = max( a , b ), or just a for the last element...
// Then we do the same thing on t.


// MM

#define HELPER_FUNC_PRELUDE						\
									\
	BLOCK_VARS_DECLS						\
	DEFAULT_YZ							\
									\
	/* compare *arg2, *arg3, results in *arg1 */			\
	/* Allocate a temporary vector for the intermediate results, */	\
	/* put in arg1 */						\
									\
	if( len1 < max_threads_per_block ) {				\
		n_threads_per_block.x = len1;				\
		n_blocks.x = 1;						\
		extra.x = 0; /* quiet compiler */			\
	} else {							\
		n_blocks.x = len1 / max_threads_per_block;		\
		n_threads_per_block.x = max_threads_per_block;		\
		extra.x = len1 % max_threads_per_block;			\
		if (extra.x != 0) n_blocks.x++;				\
	}


#define DECLARE_MM_HELPER(host_func_name,gpu_func_name,type)		\
									\
void host_func_name##_helper( type *arg1, type *arg2,			\
			type * arg3, u_long len1, u_long len2 )		\
{									\
	HELPER_FUNC_PRELUDE						\
	CLEAR_CUDA_ERROR2(host_func_name,gpu_func_name)			\
	REPORT_THREAD_INFO2						\
	gpu_func_name##_helper<<< NN_GPU >>>				\
					(arg1, arg2, arg3, len1, len2);	\
	CHECK_CUDA_ERROR(host_func_name,gpu_func_name)			\
}


#define DECLARE_MM_IND_SETUP(host_func_name,gpu_func_name,type)		\
									\
void host_func_name##_index_setup( index_type *arg1, type *arg2,	\
				type * arg3, u_long len1, u_long len2 )	\
{									\
	HELPER_FUNC_PRELUDE						\
	CLEAR_CUDA_ERROR2(host_func_name,gpu_func_name)			\
sprintf(DEFAULT_ERROR_STRING,							\
"calling gpu kernel %s_index_setup with %d blocks and %d threads per block...",\
#gpu_func_name,n_blocks.x,n_threads_per_block.x);advise(DEFAULT_ERROR_STRING);	\
	REPORT_THREAD_INFO2						\
	gpu_func_name##_index_setup<<< NN_GPU >>>			\
					(arg1, arg2, arg3, len1, len2);	\
	CHECK_CUDA_ERROR(host_func_name,gpu_func_name)			\
}

#define DECLARE_MM_IND_HELPER(host_func_name,gpu_func_name,type)	\
									\
void host_func_name##_index_helper( index_type *arg1, index_type *arg2,	\
	index_type * arg3, type * orig, u_long len1, u_long len2 )	\
{									\
	HELPER_FUNC_PRELUDE						\
	CLEAR_CUDA_ERROR2(host_func_name,gpu_func_name)			\
sprintf(DEFAULT_ERROR_STRING,"calling %s_index_helper...",#gpu_func_name);	\
advise(DEFAULT_ERROR_STRING);							\
	REPORT_THREAD_INFO2						\
	gpu_func_name##_index_helper<<< NN_GPU >>>			\
				(arg1, arg2, arg3, orig, len1, len2);	\
	CHECK_CUDA_ERROR(host_func_name,gpu_func_name)			\
}



#define DECLARE_MM_NOCC_SETUP(host_func_name,gpu_func_name,type)	\
									\
void host_func_name##_nocc_setup( type *dst_values, index_type *dst_counts,\
				type *src_values, index_type * indices,	\
				u_long len1, u_long len2 )		\
{									\
	HELPER_FUNC_PRELUDE						\
	CLEAR_CUDA_ERROR2(host_func_name,gpu_func_name)			\
sprintf(DEFAULT_ERROR_STRING,							\
"calling gpu kernel %s_nocc_setup with %d blocks and %d threads per block...",\
#gpu_func_name,n_blocks.x,n_threads_per_block.x);advise(DEFAULT_ERROR_STRING);	\
	REPORT_THREAD_INFO2						\
	gpu_func_name##_nocc_setup<<< NN_GPU >>>			\
		(dst_values, dst_counts, src_values, indices, len1, len2);\
	CHECK_CUDA_ERROR(host_func_name,gpu_func_name)			\
}

#define DECLARE_MM_NOCC_HELPER(host_func_name,gpu_func_name,type)	\
									\
void host_func_name##_nocc_helper(					\
			type *dst_values, index_type *dst_counts,	\
			type *src_values, index_type *src_counts,	\
			index_type *indices, u_long len1, u_long len2,	\
			u_long stride )					\
{									\
	HELPER_FUNC_PRELUDE						\
	CLEAR_CUDA_ERROR2(host_func_name,gpu_func_name)			\
sprintf(DEFAULT_ERROR_STRING,"calling %s_nocc_helper...",#gpu_func_name);	\
advise(DEFAULT_ERROR_STRING);							\
	REPORT_THREAD_INFO2						\
	gpu_func_name##_nocc_helper<<< NN_GPU >>>			\
		(dst_values, dst_counts, src_values, src_counts,	\
					indices, len1, len2, stride);	\
	CHECK_CUDA_ERROR(host_func_name,gpu_func_name)			\
}



#define FINISH_MM_ITERATION(whence)					\
									\
	/* Each temp vector gets used twice,				\
	 * first as result, then as source */				\
	if( src_to_free != NULL ){					\
		freetmp(src_to_free,whence);				\
		src_to_free=NULL;					\
	}								\
									\
	/* Now roll things over... */					\
	arg2 = arg1;							\
	len = len1;							\
	src_to_free = dst_to_free;					\
	dst_to_free = NULL;

#define SETUP_MM_ITERATION(type,a2,a3,whence)				\
									\
	len1 = (len.x+1)/2;						\
	len2 = len.x - len1;						\
									\
	a3 = a2 + len1;							\
									\
	if( len1 == 1 ){						\
		arg1 = (type *) oap->oa_dest->dt_data;			\
		dst_to_free = NULL;					\
	} else {							\
		arg1 = (type *) tmpvec(sizeof(type),len1,whence);	\
		dst_to_free = arg1;					\
	}




#define FINISH_NOCC_ITERATION(whence)					\
									\
	/* Each temp vector gets used twice,				\
	 * first as result, then as source */				\
	if( src_vals_to_free != NULL ){					\
		freetmp(src_vals_to_free,whence);			\
		src_vals_to_free=NULL;					\
	}								\
	if( src_counts_to_free != NULL ){				\
		freetmp(src_counts_to_free,whence);			\
		src_counts_to_free=NULL;				\
	}								\
									\
	/* Now roll things over... */					\
	src_values = dst_values;					\
	src_counts = dst_counts;					\
	len.x = len1;							\
	src_vals_to_free = dst_vals_to_free;				\
	src_counts_to_free = dst_counts_to_free;			\
	dst_vals_to_free = NULL;					\
	dst_counts_to_free = NULL;

/* For vmaxg, the temp arrays don't double for the destination vector...
 * oa_sdp[0] is the extreme value, oa_sdp[1] is the count.
 */

#define SETUP_NOCC_ITERATION(type,index_type,whence)			\
									\
	len1 = (len.x+1)/2;						\
	len2 = len.x - len1;						\
									\
	if( len1 == 1 ){						\
		dst_values = (type *) oap->oa_sdp[0]->dt_data;		\
		dst_vals_to_free = NULL;				\
		dst_counts = (index_type *) oap->oa_sdp[1]->dt_data;	\
		dst_counts_to_free = NULL;				\
	} else {							\
		dst_values = (type *) tmpvec(sizeof(type),len1,whence);	\
		dst_vals_to_free = dst_values;				\
		dst_counts = (index_type *) tmpvec(sizeof(index_type),len1,whence);\
		dst_counts_to_free = dst_counts;			\
	}


#define INIT_MM(type,a2)						\
									\
	len.x = oap->oa_dp[0]->dt_n_type_elts;				\
	len.y = len.z = 1;						\
	a2 = (type *)oap->oa_dp[0]->dt_data;				\
									\
	/* need to set max_threads_per_block */				\
	GET_MAX_THREADS( oap->oa_dest )					\
	INSIST_CONTIG( oap->oa_dp[0] , "min/max" )			\
	INSIST_LENGTH( len.x , "min/max" , oap->oa_dp[0]->dt_name )



// The index functions wants to return the index of the max/min.
// After the first pass, we will have a half-length array with the
// max indices.  The subsequent passes need to use these indices to
// lookup data to make the comparison...

#define H_CALL_MM_IND(host_func_name, gpu_func_name, type )		\
									\
DECLARE_MM_IND_SETUP(host_func_name,gpu_func_name,type)			\
DECLARE_MM_IND_HELPER(host_func_name,gpu_func_name,type)		\
									\
void host_func_name(Vec_Obj_Args * oap )				\
{									\
	type *sarg2;							\
	type *sarg3;							\
	type *orig_sarg;						\
	index_type *arg1,*arg2,*arg3;					\
	index_type *dst_to_free=NULL;					\
	index_type *src_to_free=NULL;					\
	dim3 len;							\
	u_long len1, len2;						\
									\
	if( MACHINE_PREC(oap->oa_dest) != INDEX_PREC ){			\
		sprintf(DEFAULT_ERROR_STRING,					\
"%s:  destination index %s has %s precision, should be %s",		\
			#host_func_name,oap->oa_dest->dt_name,		\
			name_for_prec(MACHINE_PREC(oap->oa_dest)),	\
			name_for_prec(INDEX_PREC) );			\
		NWARN(DEFAULT_ERROR_STRING);					\
		return;							\
	}								\
									\
	INIT_MM(type,sarg2)						\
									\
	SETUP_MM_ITERATION(index_type,sarg2,sarg3,#host_func_name)	\
	orig_sarg = sarg2;						\
	host_func_name##_index_setup(arg1,sarg2,sarg3,len1,len2);	\
	FINISH_MM_ITERATION(#host_func_name)				\
									\
	while( len.x > 1 ){						\
		SETUP_MM_ITERATION(index_type,arg2,arg3,#host_func_name)\
		host_func_name##_index_helper(arg1,arg2,arg3,		\
					orig_sarg,len1,len2);		\
		FINISH_MM_ITERATION(#host_func_name)			\
	}								\
	if( src_to_free != NULL ){					\
		freetmp(src_to_free,#host_func_name);			\
		src_to_free=NULL;					\
	}								\
}


/* Can we use the recursive strategy for vmaxg?
 *
 * We need a vector of occurrences for each thread...
 * 
 * Let's assume that the destination vector (index array) has the same dimension
 * as the input.  So we can use portions of it for the temporary storage.
 * First do all the pairs.  We need an occurrence count array equal in size to
 * the number of pairs.  The first pass sets this - the number of occurrences
 * will be either 1 or 2.
 * The second pass will compare the values accessed indirectly (as in vmaxi).
 *
 * Unlike vmaxi, we don't compare far-away pairs, we do adjacent pairs.
 *
 *
 * How do we do the merge in general?  Let's try an example with 8 elements [ 6 6 3 6 5 6 7 8 ]
 * After the first pass:
 *	index_list	[ 0 1 3 . 5 . 7 . ]
 *	n_tmp		[ 2 1 1 1 ]
 *	v_tmp		[ 6 6 6 8 ]
 *
 * The nocc_helper function has a thread for each pair of values in v_tmp;
 * it compares the values, and then updates the new result n and v accordingly.
 * The hard part is copying the indices (if necessary).
 * We have to pass the addresses of the source and destination n and v arrays,
 * plus the address of the index_list.  As len gets smaller, so do n_tmp and
 * v_tmp, but the index_list is always the same length, so we need to pass
 * a multiplier to get the offset...
 *
 * The setup function is like the helper function, but it uses the original
 * input instead of the temp array for the source of the comparisons.
 * AND it doesn't have to worry about doing any merging.
 *
 * After the second pass:
 *	index_list	[ 0 1 3 . 7 . . . ]
 *	n_tmp		[ 3 1 ]
 *	v_tmp		[ 6 8 ]
 *
 * What if we initialize like this before any passes:
 * 	index_list	[ 0 1 2 3 4 5 6 7 ]
 * 	n_tmp		[ 1 1 1 1 1 1 1 1 ]
 * 	v_tmp		[ 6 6 3 6 5 6 7 8 ]
 *
 * Testing with [ 1 2 3 4 5 6 7 8 ]
 *
 * index_list:	[ 1 . 3 . 5 . 7 . ]
 * 		[ 3 . . . 7 . . . ]
 * 		[ 7 . . . . . . . ]
 *
 * max_list:	[ 2 4 6 8 ]
 * 		[ 4 8 ]
 * 		[ 8 ]
 *
 * n_list	[ 1 1 1 1 ]
 * 		[ 1 1 ]
 * 		[ 1 ]
 *
 * What about with 9 elements?
 * input:	[ 1 2 3 4 5 6 7 8 9 ]
 *
 * index_list:	[ 1 . 3 . 5 . 7 . 8 ]
 * 		[ 3 . . . 7 . . . 8 ]
 * 		[ 7 . . . . . . . 8 ]
 * 		[ 8 . . . . . . . . ]
 *
 *
 * max_list:	[ 2 4 6 8 9 ]
 * 		[ 4 8 9 ]
 * 		[ 8 9 ]
 * 		[ 9 ]
 *
 *
 * What about with 7?
 *
 * input:	[ 1 2 3 4 5 6 7 ]
 *
 * max_v:	[ 2 4 6 7 ]
 * 		[ 4 7 ]
 * 		[ 7 ]
 *
 * indices:	[ 1 . 3 . 5 . 6 ]
 * 		[ 3 . . . 6 . . ]
 * 		[ 6 . . . . . . ]
 */

// The g functions (vmaxg etc)  want to return an array of indices, along
// with the value and the occurrence count.
// After the first pass, we will have a full-length array with the pairwise
// max indices, and two half-length temp arrays with the occurrence counts
// and the values.  (Although we don't necessarily need to store the values
// separately, because they can be obtained from the original input array.)
// The subsequent passes need to use temp max values (or use the indices
// to lookup) and then coalesce the indices.

#define H_CALL_MM_NOCC(host_func_name, gpu_func_name, type )		\
									\
DECLARE_MM_NOCC_SETUP(host_func_name,gpu_func_name,type)		\
DECLARE_MM_NOCC_HELPER(host_func_name,gpu_func_name,type)		\
									\
void host_func_name(Vec_Obj_Args * oap )				\
{									\
	type *sarg;							\
	/* dst_values, dst_counts are destinations */			\
	type *dst_values,*src_values;					\
	index_type *dst_counts,*src_counts;				\
	index_type *indices;						\
	type *dst_vals_to_free=NULL;					\
	type *src_vals_to_free=NULL;					\
	index_type *dst_counts_to_free=NULL;				\
	index_type *src_counts_to_free=NULL;				\
	dim3 len;							\
	u_long stride;							\
	u_long len1, len2;						\
									\
	if( MACHINE_PREC(oap->oa_dest) != INDEX_PREC ){			\
		sprintf(DEFAULT_ERROR_STRING,					\
	"%s:  destination index %s has %s precision, should be %s",	\
	#host_func_name,oap->oa_dest->dt_name,				\
	name_for_prec(MACHINE_PREC(oap->oa_dest)),			\
			name_for_prec(INDEX_PREC) );			\
		NWARN(DEFAULT_ERROR_STRING);					\
		return;							\
	}								\
	if( oap->oa_dest->dt_n_type_elts !=				\
		oap->oa_dp[0]->dt_n_type_elts ){			\
		sprintf(DEFAULT_ERROR_STRING,					\
"%s:  number of elements of index array %s (%d) must match source %s (%d)",\
			#host_func_name, oap->oa_dest->dt_name,		\
			oap->oa_dest->dt_n_type_elts,			\
			oap->oa_dp[0]->dt_name,				\
			oap->oa_dp[0]->dt_n_type_elts);			\
		NWARN(DEFAULT_ERROR_STRING);					\
		return;							\
	}								\
									\
	/* Set len; sets sarg2 to the input */				\
	INIT_MM(type,sarg)						\
	indices = (index_type *) oap->oa_dest->dt_data;			\
									\
	/* Set len1, len2 */						\
	/* sets sarg3 to second half of input data */			\
	/* Set dst_values, dst_counts to temp vectors */		\
	SETUP_NOCC_ITERATION(type,index_type,#host_func_name)		\
	host_func_name##_nocc_setup(dst_values,dst_counts,		\
				sarg,indices,len1,len2);		\
	FINISH_NOCC_ITERATION(#host_func_name) 				\
									\
	stride = 4;							\
	while( len.x > 1 ){						\
		SETUP_NOCC_ITERATION(type,index_type,#host_func_name)	\
		host_func_name##_nocc_helper(dst_values,		\
			dst_counts,src_values,src_counts,		\
			indices,len1,len2,stride);			\
		FINISH_NOCC_ITERATION(#host_func_name)			\
		stride *= 2;						\
	}								\
	if( src_vals_to_free != NULL ){					\
		freetmp(src_vals_to_free,#host_func_name);		\
		src_vals_to_free=NULL;					\
	}								\
	if( src_counts_to_free != NULL ){				\
		freetmp(src_counts_to_free,#host_func_name);		\
		src_counts_to_free=NULL;				\
	}								\
}

// H_CALL_MM is for min/max value
// This should really be a 2-arg projection, like vsum
//
// We have two lengths to worry about:  we initialize the destination
// in parallel, then we recursively get down to each destination element...
//

#define H_CALL_MM(host_func_name, gpu_func_name, type )			\
									\
DECLARE_MM_HELPER(host_func_name,gpu_func_name,type)			\
									\
void host_func_name(Vec_Obj_Args * oap ) {				\
	type *arg1;							\
	type *arg2;							\
	type *arg3;							\
	type *dst_to_free=NULL;						\
	type *src_to_free=NULL;						\
	dim3 len;							\
	u_long len1, len2;						\
									\
	INIT_MM(type,arg2)						\
									\
	while( len.x > 1 ){						\
		SETUP_MM_ITERATION(type,arg2,arg3,#host_func_name)	\
		host_func_name##_helper(arg1,arg2,arg3,len1,len2);	\
		FINISH_MM_ITERATION(#host_func_name)			\
	}								\
}


// vsum, vdot, etc
//
// Called "projection" because on host can compute entire sum if dest
// is a scalar, or row sums if dest is a column, etc...
//
// To compute the sum to a scalar, we have to divide and conquer
// as we did with vmaxv...

#define H_CALL_PROJ_2V( host_func_name , gpu_func_name , type )		\
									\
void host_func_name( Vec_Obj_Args *oap )				\
{									\
	type *s1, *arg1;						\
	dim3 len;							\
	u_long len1, len2;						\
	type *src_to_free, *dst_to_free;				\
									\
	len = oap->oa_dp[0]->dt_n_type_elts;				\
	s1 = (type *)oap->oa_dp[0]->dt_data;				\
									\
	GET_MAX_THREADS( oap->oa_dest )					\
									\
	INSIST_CONTIG( oap->oa_dp[0] , "sum" )				\
	INSIST_LENGTH( len.x , "sum" , oap->oa_dp[0]->dt_name )		\
									\
	src_to_free=NULL;						\
	while( len.x > 1 ){						\
		len1 = (len.x+1)/2;					\
		len2 = len.x - len1;					\
									\
		if( len1 == 1 ){					\
			arg1 = (type *) oap->oa_dest->dt_data;		\
			dst_to_free = NULL;				\
		} else {						\
			arg1 = (type *) tmpvec(sizeof(type),		\
					len1,#host_func_name);		\
			dst_to_free = arg1;				\
		}							\
									\
		/* set up thread counts */				\
		HELPER_FUNC_PRELUDE					\
									\
		CLEAR_CUDA_ERROR2(host_func_name,gpu_func_name)		\
		REPORT_THREAD_INFO					\
		gpu_func_name<<< NN_GPU >>>( arg1 , s1 , len1 , len2 );	\
		CHECK_CUDA_ERROR(host_func_name,gpu_func_name)		\
		len.x = len1;						\
		s1 = arg1;						\
		/* Each temp vector gets used twice,			\
		 * first as result, then as source */			\
		if( src_to_free != NULL ){				\
			freetmp(src_to_free,#host_func_name);		\
			src_to_free=NULL;				\
		}							\
		src_to_free = dst_to_free;				\
		dst_to_free = NULL;					\
	}								\
	if( src_to_free != NULL ){					\
		freetmp(src_to_free,#host_func_name);			\
		src_to_free=NULL;					\
	}								\
									\
}


#define H_CALL_PROJ_3V( host_func_name , gpu_func_name , type )		\
									\
void host_func_name( Vec_Obj_Args *oap )				\
{									\
	type *arg1;							\
	dim3 len;							\
	type *s1, *s2;							\
	u_long len1, len2;						\
	type *src_to_free, *dst_to_free;				\
									\
	len = oap->oa_dp[0]->dt_n_type_elts;				\
	s1 = (type *)oap->oa_dp[0]->dt_data;				\
	s2 = (type *)oap->oa_dp[1]->dt_data;				\
									\
	GET_MAX_THREADS( oap->oa_dest )					\
	INSIST_CONTIG( oap->oa_dp[0] , "vdot" )				\
	INSIST_LENGTH( len.x , "vdot" , oap->oa_dp[0]->dt_name )	\
									\
	src_to_free=NULL;						\
	while( len.x > 1 ){						\
		len1 = (len.x+1)/2;					\
		len2 = len.x - len1;					\
									\
		if( len1 == 1 ){					\
			arg1 = (type *) oap->oa_dest->dt_data;		\
			dst_to_free = NULL;				\
		} else {						\
			arg1 = (type *) tmpvec(sizeof(type),		\
					len1,#host_func_name);		\
			dst_to_free = arg1;				\
		}							\
									\
		/* set up thread counts */				\
		HELPER_FUNC_PRELUDE					\
		CLEAR_CUDA_ERROR2(host_func_name,gpu_func_name)		\
									\
		REPORT_THREAD_INFO					\
		gpu_func_name<<< NN_GPU >>>				\
			( arg1 , s1 , s2 , len1 , len2 );		\
		CHECK_CUDA_ERROR(host_func_name,gpu_func_name)		\
		len.x = len1;						\
		s1 = arg1;						\
		/* Each temp vector gets used twice,			\
		 * first as result, then as source */			\
		if( src_to_free != NULL ){				\
			freetmp(src_to_free,#host_func_name);		\
			src_to_free=NULL;				\
		}							\
		src_to_free = dst_to_free;				\
		dst_to_free = NULL;					\
	}								\
	if( src_to_free != NULL ){					\
		freetmp(src_to_free,#host_func_name);			\
		src_to_free=NULL;					\
	}								\
									\
}


#else /* ! HAVE_CUDA */

#define GENERIC_HOST_CALL(name,code,bitmap,type,scalars,vectors)
#define SLOW_HOST_CALL(name,code,bitmap,type,scalars,vectors)
#define GENERIC_SBM_GPU_CALL(name,code,type,scalars,vectors)


#define H_CALL_PROJ_3V(host_func_name,gpu_func_name,type)	\
								\
	void host_func_name(Vec_Obj_Args * oap ) {		\
		NWARN("No CUDA support."); }

#define H_CALL_PROJ_2V(host_func_name,gpu_func_name,type)	\
								\
	void host_func_name(Vec_Obj_Args * oap ) {		\
		NWARN("No CUDA support."); }

#define H_CALL_5V(host_func_name, gpu_func_name, type)		\
								\
	void host_func_name(Vec_Obj_Args * oap ) {		\
		NWARN("No CUDA support."); }

#define H_CALL_4V_SCAL(host_func_name, gpu_func_name, type)	\
								\
	void host_func_name(Vec_Obj_Args * oap ) {		\
		NWARN("No CUDA support."); }

#define H_CALL_3V_2SCAL(host_func_name, gpu_func_name, type)	\
								\
	void host_func_name(Vec_Obj_Args * oap ) {		\
		NWARN("No CUDA support."); }

#define H_CALL_2V_3SCAL(host_func_name, gpu_func_name, type)	\
								\
	void host_func_name(Vec_Obj_Args * oap ) {		\
		NWARN("No CUDA support."); }

#define H_CALL_3V(host_func_name, gpu_func_name, type)		\
								\
	void host_func_name(Vec_Obj_Args * oap ) {		\
		NWARN("No CUDA support."); }

#define H_CALL_2V(host_func_name, gpu_func_name, type)		\
									\
	void host_func_name(Vec_Obj_Args * oap ) {			\
		NWARN("No CUDA support."); }

#define H_CALL_2V_MIXED(host_func_name, gpu_func_name, type, ctyp)	\
									\
	void host_func_name(Vec_Obj_Args * oap ) {			\
		NWARN("No CUDA support."); }

#define H_CALL_CONV(host_func_name, gpu_func_name, dst_type, src_type )\
									\
	void host_func_name(Vec_Obj_Args * oap ) {			\
		NWARN("Sorry, quip configured without CUDA support."); }

#define H_CALL_2V_SCAL(host_func_name, gpu_func_name, type)	\
									\
	void host_func_name(Vec_Obj_Args * oap) {			\
		NWARN("Sorry, quip configured without CUDA support."); }

#ifdef FOOBAR
#define H_CALL_1V_SCAL(host_func_name, gpu_func_name, type)	\
									\
	void host_func_name(Vec_Obj_Args * oap) {			\
		NWARN("Sorry, quip configured without CUDA support."); }
#endif

#define H_CALL_1V_2SCAL(host_func_name, gpu_func_name, type)	\
									\
	void host_func_name(Vec_Obj_Args * oap) {			\
		NWARN("Sorry, quip configured without CUDA support."); }

#define H_CALL_1V_3SCAL(host_func_name, gpu_func_name, type)	\
									\
	void host_func_name(Vec_Obj_Args * oap) {			\
		NWARN("Sorry, quip configured without CUDA support."); }

#define H_CALL_VVSLCT(host_func_name, gpu_func_name, type)	\
									\
	void host_func_name(Vec_Obj_Args * oap) {			\
		NWARN("Sorry, quip configured without CUDA support."); }

#define H_CALL_VSSLCT(host_func_name, gpu_func_name, type)	\
									\
	void host_func_name(Vec_Obj_Args * oap) {			\
		NWARN("Sorry, quip configured without CUDA support."); }

#define H_CALL_SSSLCT(host_func_name, gpu_func_name, type)	\
									\
	void host_func_name(Vec_Obj_Args * oap) {			\
		NWARN("Sorry, quip configured without CUDA support."); }

#define H_CALL_VVMAP(host_func_name, gpu_func_name, type)	\
									\
	void host_func_name(Vec_Obj_Args * oap) {			\
		NWARN("Sorry, quip configured without CUDA support."); }

#define H_CALL_VSMAP(host_func_name, gpu_func_name, type)	\
									\
	void host_func_name(Vec_Obj_Args * oap) {			\
		NWARN("Sorry, quip configured without CUDA support."); }

#define INIT_MM(type,a2)
#define SETUP_MM_ITERATION(type,a2,a3,hfn)
#define FINISH_MM_ITERATION(hfn)
#define H_CALL_MM_NOCC(host_func_name, gpu_func_name, type )	\
								\
	void host_func_name(Vec_Obj_Args * oap ) {}

#define H_CALL_MM_IND(host_func_name, gpu_func_name, type )	\
								\
	void host_func_name(Vec_Obj_Args * oap ) {}

#define H_CALL_MM(host_func_name, gpu_func_name, type )		\
								\
	void host_func_name(Vec_Obj_Args * oap ) {}


#define DECLARE_MM_HELPER(host_func_name,gpu_func_name,type)			\
											\
	void host_func_name##_helper( type *arg1, type *arg2, type * arg3, u_long len1,	\
								u_long len2 ) {}

#define H_CALL_UNI(host_func_name, gpu_func_name, type)			\
									\
void host_func_name(Vec_Obj_Args * oap ) {				\
sprintf(DEFAULT_ERROR_STRING,"%s:  not implemented",#host_func_name);		\
NWARN(DEFAULT_ERROR_STRING);\
}






#endif /* ! HAVE_CUDA */

#define GPU_EQSP_ARGS_3		GPU_EQSP_ARGS_2, GPU_EQSP_ARGS_SRC2

