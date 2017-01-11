
#ifndef _OCL_KERN_ARGS_H_
#define _OCL_KERN_ARGS_H_

#include "veclib/gen_kern_args.h"

#define SET_KERNEL_SEPARATOR

#define SHOW_KERNEL_ARG(type,value)							\
										\
fprintf(stderr,"Setting kernel arg %d, with size %ld\n",			\
ki_idx,sizeof(type));								\
fprintf(stderr,"SHOW_KERNEL_ARG( %s , %s )\n",#type,#value);			\
if( !strcmp(#type,"float") ) fprintf(stderr,"\tfloat arg = %g\n",		\
* ((float *)value));								\
else if( !strcmp(#type,"int") ) fprintf(stderr,"\tint arg = %d\n",		\
* ((int *)value));								\
else if( !strcmp(#type,"uint32_t") ) fprintf(stderr,"\tuint32_t arg = %d\n",		\
* ((uint32_t *)value));								\
else if( !strcmp(#type,"void *") ) fprintf(stderr,"\tptr arg = 0x%lx\n",	\
(u_long)value);									\
else if( !strcmp(#type,"bitmap_word") ) fprintf(stderr,"\tbitmap word arg = 0x%lx\n",\
(/*bitmap_word*/u_long)value);								\
else if( !strcmp(#type,"dim5") ) fprintf(stderr,"\tdim5 arg = %d %d %d %d %d\n",\
((dim5 *)value)->d5_dim[0],((dim5 *)value)->d5_dim[1],((dim5 *)value)->d5_dim[2],((dim5 *)value)->d5_dim[3],((dim5 *)value)->d5_dim[4]); \
else fprintf(stderr,"\tSHOW_KERNEL_ARG:  unhandled case for type %s\n",#type);

/*else if( !strcmp(#type,"dim3") ) fprintf(stderr,"\tdim3 arg = %d %d %d\n",\
((dim3 *)value)->x,((dim3 *)value)->y,((dim3 *)value)->z);			\
*/

#define SET_KERNEL_ARG(type,value)	_SET_KERNEL_ARG(kernel[pd_idx],type,value)
#define SET_KERNEL_ARG_1(type,value)	_SET_KERNEL_ARG(kernel1[pd_idx],type,value)
#define SET_KERNEL_ARG_2(type,value)	_SET_KERNEL_ARG(kernel2[pd_idx],type,value)

#define _SET_KERNEL_ARG(kernel,type,value)				\
	/*SHOW_KERNEL_ARG(type,value)*/					\
	status = clSetKernelArg(kernel,	ki_idx++, sizeof(type), value);	\
	if( status != CL_SUCCESS )					\
		report_ocl_error(DEFAULT_QSP_ARG  status, "clSetKernelArg" );


#define SET_KERNEL_ARGS_DEST_OFFSET				\
	SET_KERNEL_ARG( OCL_OFFSET_TYPE, &VA_DEST_OFFSET(vap) )

#define SET_KERNEL_ARGS_SRC1_OFFSET				\
	SET_KERNEL_ARG( OCL_OFFSET_TYPE, &VA_SRC1_OFFSET(vap) )

#define SET_KERNEL_ARGS_SRC2_OFFSET				\
	SET_KERNEL_ARG( OCL_OFFSET_TYPE, &VA_SRC2_OFFSET(vap) )

#define SET_KERNEL_ARGS_SRC3_OFFSET				\
	SET_KERNEL_ARG( OCL_OFFSET_TYPE, &VA_SRC3_OFFSET(vap) )

#define SET_KERNEL_ARGS_SRC4_OFFSET				\
	SET_KERNEL_ARG( OCL_OFFSET_TYPE, &VA_SRC4_OFFSET(vap) )

#define SET_KERNEL_ARGS_SBM_OFFSET				\
	SET_KERNEL_ARG( OCL_OFFSET_TYPE, &VA_SBM_OFFSET(vap) )

#define SET_KERNEL_ARGS_DBM_OFFSET				\
	SET_KERNEL_ARG( OCL_OFFSET_TYPE, &VA_DBM_OFFSET(vap) )

#define SET_KERNEL_ARGS_DBM_GPU_INFO				\
	SET_KERNEL_ARG( void *, &VA_DBM_GPU_INFO_PTR(vap) )

#define SET_KERNEL_ARGS_FAST_SRC2				\
	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_SRC2(vap))) )

#define SET_KERNEL_ARGS_FAST_CONV_DEST(t)	SET_KERNEL_ARGS_FAST_1

#define SET_KERNEL_ARGS_SLOW_SIZE				\
	SET_KERNEL_ARG(DIM5,VA_SLOW_SIZE(vap))

#define SET_KERNEL_ARGS_SLOW_SIZE_OFFSET	/* nop */

#define SET_KERNEL_ARGS_FAST_1					\
	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_DEST(vap))) )

#define SET_KERNEL_ARGS_FAST_CPX_1				\
	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_DEST(vap))) )

#define SET_KERNEL_ARGS_FAST_QUAT_1				\
	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_DEST(vap))) )

// BUG - need to make sure consistent with expected args???

#define SET_KERNEL_ARGS_FAST_NOCC_SETUP				\
								\
	SET_KERNEL_ARG_1(void *,&dst_values)			\
	SET_KERNEL_ARG_1(void *,&dst_counts)			\
	SET_KERNEL_ARG_1(void *,&src_values)			\
	SET_KERNEL_ARG_1(void *,&indices)			\
	SET_KERNEL_ARG_1( uint32_t, &len1 )			\
	SET_KERNEL_ARG_1( uint32_t, &len2 )
	/*GPU_CALL_NAME(name##_nocc_setup)(dst_values, dst_counts, src_values, indices, len1, len2); */

#define SET_KERNEL_ARGS_FAST_NOCC_HELPER			\
								\
	SET_KERNEL_ARG_2(void *,&dst_values)			\
	SET_KERNEL_ARG_2(void *,&dst_counts)			\
	SET_KERNEL_ARG_2(void *,&src_values)			\
	SET_KERNEL_ARG_2(void *,&src_counts)			\
	SET_KERNEL_ARG_2(void *,&indices)			\
	SET_KERNEL_ARG_2( uint32_t, &len1 )			\
	SET_KERNEL_ARG_2( uint32_t, &len2 )			\
	SET_KERNEL_ARG_2( uint32_t, &stride )
	/*(GPU_CALL_NAME(name##_nocc_helper) (dst_values, dst_counts, src_values, src_counts, indices, len1, len2, stride); */

#define SET_KERNEL_ARGS_FAST_PROJ_2V					\
								\
fprintf(stderr,"SET_KERNEL_ARGS_PROJ_2V:  len1 = %d, len2 = %d\n",len1,len2);\
	SET_KERNEL_ARG(void *,&dst_values)		\
	SET_KERNEL_ARG(void *,&src_values)		\
	SET_KERNEL_ARG( uint32_t, &len1 )		\
	SET_KERNEL_ARG( uint32_t, &len2 )
	/* GPU_CALL_NAME(name)arg1 , s1 , len1 , len2 ); */

#define SET_KERNEL_ARGS_FAST_PROJ_3V					\
								\
	SET_KERNEL_ARG(void *,&dst_values)		\
	SET_KERNEL_ARG(void *,&src1_values)		\
	SET_KERNEL_ARG(void *,&src2_values)		\
	SET_KERNEL_ARG( uint32_t, &len1 )		\
	SET_KERNEL_ARG( uint32_t, &len2 )
		/*	( arg1 , s1 , s2 , len1 , len2 ); */

#define SET_KERNEL_ARGS_FAST_INDEX_SETUP				\
								\
	SET_KERNEL_ARG_1(void *,&indices)			\
	SET_KERNEL_ARG_1(void *,&src1_values)		\
	SET_KERNEL_ARG_1(void *,&src2_values)		\
	SET_KERNEL_ARG_1( uint32_t, &len1 )		\
	SET_KERNEL_ARG_1( uint32_t, &len2 )
		/* (idx_arg1,std_arg2,std_arg3,len1,len2,max_threads_per_block) */	\


// should this be src1_values???  or orig_values?

#define SET_KERNEL_ARGS_FAST_INDEX_HELPER				\
								\
	SET_KERNEL_ARG_2(void *,&indices)			\
	SET_KERNEL_ARG_2(void *,&idx1_values)		\
	SET_KERNEL_ARG_2(void *,&idx2_values)		\
	SET_KERNEL_ARG_2(void *,&src1_values)		\
	SET_KERNEL_ARG_2( uint32_t, &len1 )		\
	SET_KERNEL_ARG_2( uint32_t, &len2 )
	/*(arg1, arg2, arg3, orig, len1, len2) */
/* ( index_type *arg1, index_type *arg2, index_type * arg3, std_type * orig, u_long len1, u_long len2, int max_threads_per_block ) */




// BUG - can we insure that the arg order matches VFUNC???

#define SET_KERNEL_ARGS_FAST_SBM	SET_KERNEL_ARG(void *,&(VA_SBM_PTR(vap)))	\
					SET_KERNEL_ARG(int,&(VA_SBM_BIT0(vap)))

// BUG incset is not increment!?
#define SET_KERNEL_ARGS_EQSP_SBM	SET_KERNEL_ARG(void *,&(VA_SBM_PTR(vap)))	\
					SET_KERNEL_ARG(int,&(VA_SBM_BIT0(vap)))	\
					SET_KERNEL_ARG(int,&VA_SBM_EQSP_INC(vap))

#define SET_KERNEL_ARGS_SLOW_SBM	SET_KERNEL_ARG(void *,&(VA_SBM_PTR(vap)))	\
					SET_KERNEL_ARG(int,&(VA_SBM_BIT0(vap)))	\
					/* SET_KERNEL_ARG(DIM3,&sbm_xyz_incr) */ \
					SET_KERNEL_ARG(DIM5,&sbm_vwxyz_incr)



// I don't think that "fast" bitmaps can include a bit0 parameter, unless it is a multiple of the word size!?
// If so, then they will need the gpu_info arg, which is no longer "fast" !?

// BUG - how can we be sure that these definitions are consistent with the kernels?

//	SET_KERNEL_ARG( void *, &VA_DBM_GPU_INFO_PTR(vap) )

#define SET_KERNEL_ARGS_FAST_DBM	SET_KERNEL_ARG(void *,&(VA_DBM_PTR(vap)))

#define SET_KERNEL_ARGS_EQSP_DBM	SET_KERNEL_ARG(void *,&(VA_DBM_PTR(vap)))

//#define SET_KERNEL_ARGS_SLOW_DBM	SET_KERNEL_ARG(void *,&(VA_DBM_PTR(vap)))	\
//					SET_KERNEL_ARG(int,&(VA_DBM_BIT0(vap)))		\
//					SET_KERNEL_ARG(DIM5,&dbm_vwxyz_incr)

// BUG - there doesn't seem to be anything that enforces these definitions to match what is done elsewhere?
#define SET_KERNEL_ARGS_SLOW_DBM	SET_KERNEL_ARG(void *,&(VA_DBM_PTR(vap)))


// SRC1

#define SET_KERNEL_ARGS_FAST_SRC1	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_SRC1(vap))) )
#define SET_KERNEL_ARGS_FAST_CPX_SRC1	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_SRC1(vap))) )
#define SET_KERNEL_ARGS_FAST_QUAT_SRC1	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_SRC1(vap))) )

#define SET_KERNEL_ARGS_FAST_SRC2	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_SRC2(vap))) )
#define SET_KERNEL_ARGS_FAST_CPX_SRC2	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_SRC2(vap))) )
#define SET_KERNEL_ARGS_FAST_QUAT_SRC2	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_SRC2(vap))) )

#define SET_KERNEL_ARGS_FAST_SRC3	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_SRC3(vap))) )
#define SET_KERNEL_ARGS_FAST_CPX_SRC3	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_SRC3(vap))) )
#define SET_KERNEL_ARGS_FAST_QUAT_SRC3	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_SRC3(vap))) )

#define SET_KERNEL_ARGS_FAST_SRC4	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_SRC4(vap))) )
#define SET_KERNEL_ARGS_FAST_CPX_SRC4	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_SRC4(vap))) )
#define SET_KERNEL_ARGS_FAST_QUAT_SRC4	SET_KERNEL_ARG(void *,&(VARG_PTR( VA_SRC4(vap))) )

#define SET_KERNEL_ARGS_EQSP_SRC1	GEN_ARGS_EQSP_SRC1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_CPX_SRC1	GEN_ARGS_EQSP_SRC1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_QUAT_SRC1	GEN_ARGS_EQSP_SRC1(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_SLOW_SRC1	GEN_ARGS_SLOW_SRC1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_CPX_SRC1	GEN_ARGS_SLOW_SRC1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_QUAT_SRC1	GEN_ARGS_SLOW_SRC1(SET_KERNEL_ARGS)


#define SET_KERNEL_ARGS_EQSP_CONV(t)	GEN_ARGS_EQSP_CONV(SET_KERNEL_ARGS,t)
/* #define SET_KERNEL_ELEN_ARGS_CONV(t)	GEN_ELEN_ARGS_CONV(SET_KERNEL_ARGS,t) */
#define SET_KERNEL_ARGS_EQSP_1		GEN_ARGS_EQSP_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_2		GEN_ARGS_EQSP_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_3		GEN_ARGS_EQSP_3(SET_KERNEL_ARGS)
/*#define SET_KERNEL_ELEN_ARGS_1		GEN_ELEN_ARGS_1(SET_KERNEL_ARGS) */
/*#define SET_KERNEL_ELEN_ARGS_2		GEN_ELEN_ARGS_2(SET_KERNEL_ARGS) */
/*#define SET_KERNEL_ELEN_ARGS_3		GEN_ELEN_ARGS_3(SET_KERNEL_ARGS) */

#define SET_KERNEL_ARGS_EQSP_CPX_1	GEN_ARGS_EQSP_CPX_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_CPX_2	GEN_ARGS_EQSP_CPX_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_CPX_3	GEN_ARGS_EQSP_CPX_3(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_EQSP_QUAT_1	GEN_ARGS_EQSP_QUAT_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_QUAT_2	GEN_ARGS_EQSP_QUAT_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_QUAT_3	GEN_ARGS_EQSP_QUAT_3(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_FAST_CPX_3	GEN_ARGS_FAST_CPX_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_CPX_3 	GEN_ARGS_EQSP_CPX_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_CPX_3 	GEN_ARGS_SLOW_CPX_3(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_FAST_QUAT_3	GEN_ARGS_FAST_QUAT_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_QUAT_3 	GEN_ARGS_EQSP_QUAT_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_QUAT_3 	GEN_ARGS_SLOW_QUAT_3(SET_KERNEL_ARGS)


#define SET_KERNEL_ARGS_FAST_CCR_3 	GEN_ARGS_FAST_CCR_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_CCR_3	GEN_ARGS_EQSP_CCR_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_CCR_3	GEN_ARGS_SLOW_CCR_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_QQR_3 	GEN_ARGS_FAST_QQR_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_QQR_3	GEN_ARGS_EQSP_QQR_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_QQR_3	GEN_ARGS_SLOW_QQR_3(SET_KERNEL_ARGS)



#define SET_KERNEL_ARGS_FAST_CR_1S_2	GEN_ARGS_FAST_CR_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_CR_1S_2	GEN_ARGS_EQSP_CR_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_CR_1S_2	GEN_ARGS_SLOW_CR_1S_2(SET_KERNEL_ARGS)

#ifdef FOOBAR
#define SET_KERNEL_FLEN_ARGS_CR_1S_2	GEN_FLEN_ARGS_CR_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ELEN_ARGS_CR_1S_2	GEN_ELEN_ARGS_CR_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_CR_1S_2	GEN_SLEN_ARGS_CR_1S_2(SET_KERNEL_ARGS)

#define SET_KERNEL_FLEN_ARGS_QR_1S_2	GEN_FLEN_ARGS_QR_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ELEN_ARGS_QR_1S_2	GEN_ELEN_ARGS_QR_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_QR_1S_2	GEN_SLEN_ARGS_QR_1S_2(SET_KERNEL_ARGS)

#define SET_KERNEL_FLEN_ARGS_RC_2	GEN_FLEN_ARGS_RC_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ELEN_ARGS_RC_2	GEN_ELEN_ARGS_RC_2(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_RC_2	GEN_SLEN_ARGS_RC_2(SET_KERNEL_ARGS)

#define SET_KERNEL_FLEN_ARGS_RQ_2	GEN_FLEN_ARGS_RQ_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ELEN_ARGS_RQ_2	GEN_ELEN_ARGS_RQ_2(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_RQ_2	GEN_SLEN_ARGS_RQ_2(SET_KERNEL_ARGS)

#define SET_KERNEL_FLEN_ARGS_1S_1	GEN_FLEN_ARGS_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ELEN_ARGS_1S_1	GEN_ELEN_ARGS_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_1S_1	GEN_SLEN_ARGS_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_FLEN_ARGS_CPX_1S_1	GEN_FLEN_ARGS_CPX_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ELEN_ARGS_CPX_1S_1	GEN_ELEN_ARGS_CPX_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_CPX_1S_1	GEN_SLEN_ARGS_CPX_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_FLEN_ARGS_QUAT_1S_1	GEN_FLEN_ARGS_QUAT_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ELEN_ARGS_QUAT_1S_1	GEN_ELEN_ARGS_QUAT_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_QUAT_1S_1	GEN_SLEN_ARGS_QUAT_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_FLEN_ARGS_2S_1	GEN_FLEN_ARGS_2S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ELEN_ARGS_2S_1	GEN_ELEN_ARGS_2S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_2S_1	GEN_SLEN_ARGS_2S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_FLEN_ARGS_CPX_1S_2	GEN_FLEN_ARGS_CPX_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ELEN_ARGS_CPX_1S_2	GEN_ELEN_ARGS_CPX_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_CPX_1S_2	GEN_SLEN_ARGS_CPX_1S_2(SET_KERNEL_ARGS)

#define SET_KERNEL_FLEN_ARGS_3	GEN_FLEN_ARGS_3(SET_KERNEL_ARGS)
#define SET_KERNEL_FLEN_ARGS_2	GEN_FLEN_ARGS_2(SET_KERNEL_ARGS)
#define SET_KERNEL_FLEN_ARGS_1	GEN_FLEN_ARGS_1(SET_KERNEL_ARGS)

#define SET_KERNEL_FLEN_ARGS_1S_2	GEN_FLEN_ARGS_1S_2(SET_KERNEL_ARGS)

#endif // FOOBAR

#define SET_KERNEL_ARGS_FAST_QR_1S_2	GEN_ARGS_FAST_QR_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_QR_1S_2	GEN_ARGS_EQSP_QR_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_QR_1S_2	GEN_ARGS_SLOW_QR_1S_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_FAST_RC_2	GEN_ARGS_FAST_RC_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_RC_2	GEN_ARGS_EQSP_RC_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_RC_2	GEN_ARGS_SLOW_RC_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_FAST_RQ_2	GEN_ARGS_FAST_RQ_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_RQ_2	GEN_ARGS_EQSP_RQ_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_RQ_2	GEN_ARGS_SLOW_RQ_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_SLOW_CONV(t)	GEN_ARGS_SLOW_CONV(SET_KERNEL_ARGS,t)
#define SET_KERNEL_ARGS_SLOW_1		GEN_ARGS_SLOW_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_2		GEN_ARGS_SLOW_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_3		GEN_ARGS_SLOW_3(SET_KERNEL_ARGS)

#ifdef FOOBAR
#define SET_KERNEL_SLEN_ARGS_CONV(t)	GEN_SLEN_ARGS_CONV(SET_KERNEL_ARGS,t)
#define SET_KERNEL_SLEN_ARGS_1		GEN_SLEN_ARGS_1(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_2		GEN_SLEN_ARGS_2(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_3		GEN_SLEN_ARGS_3(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_CPX_1	GEN_SLEN_ARGS_CPX_1(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_CPX_2	GEN_SLEN_ARGS_CPX_2(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_CPX_3	GEN_SLEN_ARGS_CPX_3(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_CCR_3	GEN_SLEN_ARGS_CCR_3(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_QQR_3	GEN_SLEN_ARGS_QQR_3(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_QUAT_1	GEN_SLEN_ARGS_QUAT_1(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_QUAT_2	GEN_SLEN_ARGS_QUAT_2(SET_KERNEL_ARGS)
#define SET_KERNEL_SLEN_ARGS_QUAT_3	GEN_SLEN_ARGS_QUAT_3(SET_KERNEL_ARGS)
#endif // FOOBAR

#define SET_KERNEL_ARGS_SLOW_CPX_1	GEN_ARGS_SLOW_CPX_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_CPX_2	GEN_ARGS_SLOW_CPX_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_CPX_3	GEN_ARGS_SLOW_CPX_3(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_SLOW_CCR_3	GEN_ARGS_SLOW_CCR_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_QQR_3	GEN_ARGS_SLOW_QQR_3(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_SLOW_QUAT_1	GEN_ARGS_SLOW_QUAT_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_QUAT_2	GEN_ARGS_SLOW_QUAT_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_QUAT_3	GEN_ARGS_SLOW_QUAT_3(SET_KERNEL_ARGS)


#define SET_KERNEL_ARGS_FAST_1S_1	GEN_ARGS_FAST_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_1S_1	GEN_ARGS_EQSP_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_1S_1	GEN_ARGS_SLOW_1S_1(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_FAST_CPX_1S_1	GEN_ARGS_FAST_CPX_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_CPX_1S_1	GEN_ARGS_EQSP_CPX_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_CPX_1S_1	GEN_ARGS_SLOW_CPX_1S_1(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_FAST_QUAT_1S_1	GEN_ARGS_FAST_QUAT_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_QUAT_1S_1	GEN_ARGS_EQSP_QUAT_1S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_QUAT_1S_1	GEN_ARGS_SLOW_QUAT_1S_1(SET_KERNEL_ARGS)


#define SET_KERNEL_ARGS_FAST_2S_1	GEN_ARGS_FAST_2S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_2S_1	GEN_ARGS_EQSP_2S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_2S_1	GEN_ARGS_SLOW_2S_1(SET_KERNEL_ARGS)

#define UNIMP_SK( mname )	\
fprintf(stderr,"Oops:  Need to implement %s!?\n",#mname);


#define SET_KERNEL_ARGS_1S				\
	SET_KERNEL_SCALAR_ARG(0)
	
#define SET_KERNEL_ARGS_CPX_1S				\
	SET_KERNEL_SCALAR_ARG(0)
	
#define SET_KERNEL_ARGS_QUAT_1S				\
	SET_KERNEL_SCALAR_ARG(0)
	
#define SET_KERNEL_ARGS_2S				\
	SET_KERNEL_SCALAR_ARG(0)			\
	SET_KERNEL_SCALAR_ARG(1)

#define SET_KERNEL_ARGS_CPX_2S				\
	SET_KERNEL_CPX_SCALAR_ARG(0)			\
	SET_KERNEL_CPX_SCALAR_ARG(1)

#define SET_KERNEL_ARGS_QUAT_2S				\
	SET_KERNEL_QUAT_SCALAR_ARG(0)			\
	SET_KERNEL_QUAT_SCALAR_ARG(1)

#define SET_KERNEL_ARGS_3S				\
	SET_KERNEL_SCALAR_ARG(0)			\
	SET_KERNEL_SCALAR_ARG(1)			\
	SET_KERNEL_SCALAR_ARG(2)

#define SET_KERNEL_SCALAR_ARG(scalar_idx)				\
	SET_KERNEL_ARG(std_type,vap->va_sval[scalar_idx] )


#define SET_KERNEL_CPX_SCALAR_ARG(scalar_idx)				\
	SET_KERNEL_ARG(std_cpx,vap->va_sval[scalar_idx] )


#define SET_KERNEL_QUAT_SCALAR_ARG(scalar_idx)				\
	SET_KERNEL_ARG(std_quat,vap->va_sval[scalar_idx] )


// BUG need to figure out which dimension is the one?
#define SET_KERNEL_ARGS_EQSP_INC1	SET_KERNEL_ARG(int,&VA_DEST_EQSP_INC(vap))
#define SET_KERNEL_ARGS_EQSP_INC2	SET_KERNEL_ARG(int,&VA_SRC1_EQSP_INC(vap))
#define SET_KERNEL_ARGS_EQSP_INC3	SET_KERNEL_ARG(int,&VA_SRC2_EQSP_INC(vap))
#define SET_KERNEL_ARGS_EQSP_INC4	SET_KERNEL_ARG(int,&VA_SRC3_EQSP_INC(vap))
#define SET_KERNEL_ARGS_EQSP_INC5	SET_KERNEL_ARG(int,&VA_SRC4_EQSP_INC(vap))

// BUG?  do we need DIM3 increments in Vector_Args ???
// Now we have DIM5 sizes and increments in Vector_Args!

/*
#define SET_KERNEL_ARGS_SLOW_INC1	SET_KERNEL_ARG(DIM3,&dst_xyz_incr)
#define SET_KERNEL_ARGS_SLOW_INC2	SET_KERNEL_ARG(DIM3,&s1_xyz_incr)
#define SET_KERNEL_ARGS_SLOW_INC3	SET_KERNEL_ARG(DIM3,&s2_xyz_incr)
#define SET_KERNEL_ARGS_SLOW_INC4	SET_KERNEL_ARG(DIM3,&s3_xyz_incr)
#define SET_KERNEL_ARGS_SLOW_INC5	SET_KERNEL_ARG(DIM3,&s4_xyz_incr)
*/
#define SET_KERNEL_ARGS_SLOW_INC1	SET_KERNEL_ARG(DIM5,&dst_vwxyz_incr)
#define SET_KERNEL_ARGS_SLOW_INC2	SET_KERNEL_ARG(DIM5,&s1_vwxyz_incr)
#define SET_KERNEL_ARGS_SLOW_INC3	SET_KERNEL_ARG(DIM5,&s2_vwxyz_incr)
#define SET_KERNEL_ARGS_SLOW_INC4	SET_KERNEL_ARG(DIM5,&s3_vwxyz_incr)
#define SET_KERNEL_ARGS_SLOW_INC5	SET_KERNEL_ARG(DIM5,&s4_vwxyz_incr)

#define SET_KERNEL_ARGS_FAST_1S_2	GEN_ARGS_FAST_1S_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_FAST_CPX_1S_2	GEN_ARGS_FAST_CPX_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_CPX_1S_2	GEN_ARGS_EQSP_CPX_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_CPX_1S_2	GEN_ARGS_SLOW_CPX_1S_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_FAST_QUAT_1S_2	GEN_ARGS_FAST_QUAT_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_QUAT_1S_2	GEN_ARGS_EQSP_QUAT_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_QUAT_1S_2	GEN_ARGS_SLOW_QUAT_1S_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_FAST_3		GEN_ARGS_FAST_3(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_FAST_5		GEN_ARGS_FAST_5(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_1S_4	GEN_ARGS_FAST_1S_4(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_2S_3	GEN_ARGS_FAST_2S_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_3S_2	GEN_ARGS_FAST_3S_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_EQSP_5		GEN_ARGS_EQSP_5(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_1S_4	GEN_ARGS_EQSP_1S_4(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_2S_3	GEN_ARGS_EQSP_2S_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_3S_2	GEN_ARGS_EQSP_3S_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_SLOW_5		GEN_ARGS_SLOW_5(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_1S_4	GEN_ARGS_SLOW_1S_4(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_2S_3	GEN_ARGS_SLOW_2S_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_3S_2	GEN_ARGS_SLOW_3S_2(SET_KERNEL_ARGS)


#define SET_KERNEL_ARGS_FAST_SBM_3 		GEN_ARGS_FAST_SBM_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_SBM_CPX_3 		GEN_ARGS_FAST_SBM_CPX_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_SBM_QUAT_3 	GEN_ARGS_FAST_SBM_QUAT_3(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_EQSP_SBM_3 		GEN_ARGS_EQSP_SBM_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_SBM_CPX_3 		GEN_ARGS_EQSP_SBM_CPX_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_SBM_QUAT_3 	GEN_ARGS_EQSP_SBM_QUAT_3(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_SLOW_SBM_3 		GEN_ARGS_SLOW_SBM_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_SBM_CPX_3 		GEN_ARGS_SLOW_SBM_CPX_3(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_SBM_QUAT_3 	GEN_ARGS_SLOW_SBM_QUAT_3(SET_KERNEL_ARGS)


#define SET_KERNEL_ARGS_FAST_SBM_1S_2 		GEN_ARGS_FAST_SBM_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_SBM_CPX_1S_2 	GEN_ARGS_FAST_SBM_CPX_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_SBM_QUAT_1S_2 	GEN_ARGS_FAST_SBM_QUAT_1S_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_EQSP_SBM_1S_2 		GEN_ARGS_EQSP_SBM_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_SBM_CPX_1S_2 	GEN_ARGS_EQSP_SBM_CPX_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_SBM_QUAT_1S_2 	GEN_ARGS_EQSP_SBM_QUAT_1S_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_SLOW_SBM_1S_2 		GEN_ARGS_SLOW_SBM_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_SBM_CPX_1S_2 	GEN_ARGS_SLOW_SBM_CPX_1S_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_SBM_QUAT_1S_2 	GEN_ARGS_SLOW_SBM_QUAT_1S_2(SET_KERNEL_ARGS)



#define SET_KERNEL_ARGS_FAST_SBM_2S_1 		GEN_ARGS_FAST_SBM_2S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_SBM_CPX_2S_1	GEN_ARGS_FAST_SBM_CPX_2S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_SBM_QUAT_2S_1 	GEN_ARGS_FAST_SBM_QUAT_2S_1(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_EQSP_SBM_2S_1 		GEN_ARGS_EQSP_SBM_2S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_SBM_CPX_2S_1	GEN_ARGS_EQSP_SBM_CPX_2S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_SBM_QUAT_2S_1 	GEN_ARGS_EQSP_SBM_QUAT_2S_1(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_SLOW_SBM_2S_1 		GEN_ARGS_SLOW_SBM_2S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_SBM_CPX_2S_1	GEN_ARGS_SLOW_SBM_CPX_2S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_SBM_QUAT_2S_1 	GEN_ARGS_SLOW_SBM_QUAT_2S_1(SET_KERNEL_ARGS)




#define SET_KERNEL_ARGS_FAST_CONV(t)	GEN_ARGS_FAST_CONV(SET_KERNEL_ARGS,t)
#define SET_KERNEL_ARGS_FAST_2		GEN_ARGS_FAST_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_CPX_2	GEN_ARGS_FAST_CPX_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_QUAT_2	GEN_ARGS_FAST_QUAT_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_FAST_CR_2	GEN_ARGS_FAST_CR_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_QR_2	GEN_ARGS_FAST_QR_2(SET_KERNEL_ARGS)

//#define SET_KERNEL_ARGS_EQSP_2		GEN_ARGS_EQSP_2(SET_KERNEL_ARGS)
//#define SET_KERNEL_ARGS_EQSP_CPX_2	GEN_ARGS_EQSP_CPX_2(SET_KERNEL_ARGS)
//#define SET_KERNEL_ARGS_EQSP_QUAT_2	GEN_ARGS_EQSP_QUAT_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_EQSP_CR_2	GEN_ARGS_EQSP_CR_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_QR_2	GEN_ARGS_EQSP_QR_2(SET_KERNEL_ARGS)

//#define SET_KERNEL_ARGS_SLOW_2		GEN_ARGS_SLOW_2(SET_KERNEL_ARGS)
//#define SET_KERNEL_ARGS_SLOW_CPX_2	GEN_ARGS_SLOW_CPX_2(SET_KERNEL_ARGS)
//#define SET_KERNEL_ARGS_SLOW_QUAT_2	GEN_ARGS_SLOW_QUAT_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_SLOW_CR_2	GEN_ARGS_SLOW_CR_2(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_QR_2	GEN_ARGS_SLOW_QR_2(SET_KERNEL_ARGS)


// All eqsp args


#define SET_KERNEL_ARGS_EQSP_1S_2	GEN_ARGS_EQSP_1S_2(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_SLOW_3S_1	GEN_ARGS_SLOW_3S_1(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_1S_2	GEN_ARGS_SLOW_1S_2(SET_KERNEL_ARGS)

// All flen args


#define SET_KERNEL_ARGS_FAST_DBM_2SRCS		GEN_ARGS_FAST_DBM_2SRCS(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_DBM_1S_1SRC	GEN_ARGS_FAST_DBM_1S_1SRC(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_FAST_DBM_1S_		GEN_ARGS_FAST_DBM_1S_(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_EQSP_DBM_2SRCS		GEN_ARGS_EQSP_DBM_2SRCS(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_DBM_1S_1SRC	GEN_ARGS_EQSP_DBM_1S_1SRC(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_DBM_1S_		GEN_ARGS_EQSP_DBM_1S_(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_SLOW_DBM_2SRCS		GEN_ARGS_SLOW_DBM_2SRCS(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_DBM_1S_1SRC	GEN_ARGS_SLOW_DBM_1S_1SRC(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_DBM_1S_		GEN_ARGS_SLOW_DBM_1S_(SET_KERNEL_ARGS)

#define SET_KERNEL_ARGS_FAST_DBM_SBM		GEN_ARGS_FAST_DBM_SBM(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_EQSP_DBM_SBM		GEN_ARGS_EQSP_DBM_SBM(SET_KERNEL_ARGS)
#define SET_KERNEL_ARGS_SLOW_DBM_SBM		GEN_ARGS_SLOW_DBM_SBM(SET_KERNEL_ARGS)


#define SET_KERNEL_ARGS_FAST_CPX_3V_T1	SET_KERNEL_ARGS_FAST_CPX_3

#endif /* _OCL_KERN_ARGS_H_ */

