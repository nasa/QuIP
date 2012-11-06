#define MENU_FUNC_NAME(f)	do_gpu_##f

#define CHECK_GPU_OBJ( dp )						\
									\
	if( IS_RAM(dp) )						\
	{								\
		sprintf(DEFAULT_ERROR_STRING,					\
	"%s:  object %s lives in %s data area, expected a GPU.",	\
			func_name,					\
			dp->dt_name, dp->dt_ap->da_name);		\
		NWARN(DEFAULT_ERROR_STRING);					\
		return;							\
	}

#define CHECK_SAME_GPU( dp1, dp2 )					\
									\
	if( dp1->dt_ap != dp2->dt_ap )					\
	{								\
		sprintf(DEFAULT_ERROR_STRING,					\
	"%s:  objects %s (%s) and %s (%s) are not on the same GPU!?",	\
	func_name,dp1->dt_name, dp1->dt_ap->da_name, dp2->dt_name,	\
					dp2->dt_ap->da_name);		\
		NWARN(DEFAULT_ERROR_STRING);					\
		return;							\
	}	
	
#define CHECK_SAME_SIZE( dp1, dp2, whence )				\
									\
	if( !dp_same_size(QSP_ARG  dp1,dp2,whence) )			\
		return;
	
#define CHECK_SAME_PREC( dp1, dp2, whence )				\
									\
	if( !dp_same_prec(QSP_ARG  dp1,dp2,whence) )			\
		return;
	
#define CHECK_CONTIGUITY( dp )						\
									\
	if (!is_contiguous(dp))						\
	{								\
		sprintf(DEFAULT_ERROR_STRING,					\
	"Sorry, object %s is not contiguous.", dp->dt_name);		\
		NWARN(DEFAULT_ERROR_STRING);					\
		return;							\
	}

#define NEW_CHECK_CONTIGUITY( dp )					\
									\
	if (!is_contiguous(dp)) {					\
		all_contig = 0;						\
		if( !is_evenly_spaced(dp) )				\
			all_eqsp = 0;					\
	}

#ifdef HAVE_CUDA

#define CUDA_MENU_FUNC_DECLS						\
									\
	/*Capture event start time */					\
	cudaEvent_t start, stop;					\
	cudaEventCreate(&start);					\
	cudaEventCreate(&stop);						\
	cudaEventRecord(start, 0);
	
#define FINISH_CUDA_MENU_FUNC						\
	/*Capture event stop time*/					\
	cudaEventRecord(stop, 0);					\
	cudaEventSynchronize(stop);					\
									\
	float elapsedTime;						\
									\
	cudaEventElapsedTime(&elapsedTime, start, stop);		\
	/*printf("Compute Time: %3.1fms\n", elapsedTime);*/		\
	cudaEventDestroy(start);					\
	cudaEventDestroy(stop);


#else /* ! HAVE_CUDA */

#define CUDA_MENU_FUNC_DECLS
#define FINISH_CUDA_MENU_FUNC

#endif /* ! HAVE_CUDA */

#define GPU_PREC_SWITCH_RC(name,dst_dp,oap)				\
									\
	SET_MAX_THREADS( dst_dp )					\
	if( IS_COMPLEX(dst_dp) ){					\
		GPU_PREC_SWITCH_C(c##name,dst_dp,oap)			\
	} else {							\
		GPU_PREC_SWITCH(r##name,dst_dp,oap)			\
	}

#define GPU_PREC_SWITCH(name, dst_dp, oap )				\
									\
	SET_MAX_THREADS( dst_dp )					\
	switch( dst_dp->dt_prec ){					\
		case PREC_SP :						\
			h_sp_##name( oap );				\
			break;						\
		case PREC_DP :						\
			h_dp_##name( oap );				\
			break;						\
		case PREC_BY :						\
			h_by_##name( oap );				\
			break;						\
		case PREC_IN :						\
			h_in_##name( oap );				\
			break;						\
		case PREC_DI :						\
			h_di_##name( oap );				\
			break;						\
		case PREC_LI :						\
			h_li_##name( oap );				\
			break;						\
		case PREC_UBY :						\
			h_uby_##name( oap );				\
			break;						\
		case PREC_UIN :						\
			h_uin_##name( oap );				\
			break;						\
		case PREC_UDI :						\
			h_udi_##name( oap );				\
			break;						\
		case PREC_ULI :						\
			h_uli_##name( oap );				\
			break;						\
		default:						\
			/* GPU_PREC_SWITCH */				\
			sprintf(DEFAULT_ERROR_STRING,				\
	"Sorry, %s precision is not supported on the GPU for function %s.",\
				name_for_prec(dst_dp->dt_prec),#name);	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}

/* All the usual types PLUS bitmap */

#define GPU_PREC_SWITCH_B(name, dst_dp, oap )				\
									\
	SET_MAX_THREADS( dst_dp )					\
	switch( dst_dp->dt_prec ){					\
		case PREC_SP :						\
			h_sp_##name( oap );				\
			break;						\
		case PREC_DP :						\
			h_dp_##name( oap );				\
			break;						\
		case PREC_BY :						\
			h_by_##name( oap );				\
			break;						\
		case PREC_IN :						\
			h_in_##name( oap );				\
			break;						\
		case PREC_DI :						\
			h_di_##name( oap );				\
			break;						\
		case PREC_LI :						\
			h_li_##name( oap );				\
			break;						\
		case PREC_UBY :						\
			h_uby_##name( oap );				\
			break;						\
		case PREC_UIN :						\
			h_uin_##name( oap );				\
			break;						\
		case PREC_UDI :						\
			h_udi_##name( oap );				\
			break;						\
		case PREC_ULI :						\
			h_uli_##name( oap );				\
			break;						\
		case PREC_BIT :						\
			h_bit_##name( oap );				\
			break;						\
		default:						\
			/* GPU_PREC_SWITCH */				\
			sprintf(DEFAULT_ERROR_STRING,				\
	"Sorry, %s precision is not supported on the GPU for function %s.",\
				name_for_prec(dst_dp->dt_prec),#name);	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}


// Signed types only

#define GPU_PREC_SWITCH_S(name, dst_dp, oap )				\
									\
	SET_MAX_THREADS( dst_dp )					\
	switch( dst_dp->dt_prec ){					\
		case PREC_SP :						\
			h_sp_##name( oap );				\
			break;						\
		case PREC_DP :						\
			h_dp_##name( oap );				\
			break;						\
		case PREC_BY :						\
			h_by_##name( oap );				\
			break;						\
		case PREC_IN :						\
			h_in_##name( oap );				\
			break;						\
		case PREC_DI :						\
			h_di_##name( oap );				\
			break;						\
		case PREC_LI :						\
			h_li_##name( oap );				\
			break;						\
		default:						\
			/* GPU_PREC_SWITCH_S */				\
			sprintf(DEFAULT_ERROR_STRING,				\
	"Sorry, %s precision is not supported on the GPU for function %s.",\
				name_for_prec(dst_dp->dt_prec),#name);	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}


#define GPU_PREC_SWITCH_I(name, dst_dp, oap)				\
									\
	SET_MAX_THREADS( dst_dp )					\
	switch( dst_dp->dt_prec ){					\
		case PREC_BY :						\
			h_by_##name(oap);				\
			break;						\
		case PREC_IN :						\
			h_in_##name(oap);				\
			break;						\
		case PREC_DI :						\
			h_di_##name(oap);				\
			break;						\
		case PREC_LI :						\
			h_li_##name(oap);				\
			break;						\
		case PREC_UBY :						\
			h_uby_##name(oap);				\
			break;						\
		case PREC_UIN :						\
			h_uin_##name(oap);				\
			break;						\
		case PREC_UDI :						\
			h_udi_##name(oap);				\
			break;						\
		case PREC_ULI :						\
			h_uli_##name(oap);				\
			break;						\
		default:						\
			/* GPU_PREC_SWITCH_I */				\
			sprintf(DEFAULT_ERROR_STRING,				\
	"Sorry, %s precision is not supported on the GPU for function %s.",\
				name_for_prec(dst_dp->dt_prec),#name);	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}


#define GPU_PREC_SWITCH_F(name, dst_dp, oap)				\
	SET_MAX_THREADS( dst_dp )					\
	switch( dst_dp->dt_prec ){					\
		case PREC_SP :						\
			h_sp_##name(oap);				\
			break;						\
		case PREC_DP :						\
			h_dp_##name(oap);				\
			break;						\
		default:						\
			/* GPU_PREC_SWITCH_F */				\
			sprintf(DEFAULT_ERROR_STRING,				\
	"Sorry, %s precision is not supported on the GPU for function %s.",\
				name_for_prec(dst_dp->dt_prec),#name);	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}																	



#define GPU_PREC_SWITCH_C(name, dst_dp, oap)				\
	SET_MAX_THREADS( dst_dp )					\
	switch( dst_dp->dt_prec ){					\
		case PREC_CPX :						\
			h_sp_##name(oap);				\
			break;						\
		case PREC_DBLCPX :					\
			h_dp_##name(oap);				\
			break;						\
		default:						\
			/* GPU_PREC_SWITCH_F */				\
			sprintf(DEFAULT_ERROR_STRING,				\
	"Sorry, %s precision is not supported on the GPU for function %s.",\
				name_for_prec(dst_dp->dt_prec),#name);	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}																	


#define GPU_PREC_SWITCH_CONV(name, dst_dp, src1_dp, oap)		\
									\
	SET_MAX_THREADS( dst_dp )					\
	switch( dst_dp->dt_prec ){					\
		case PREC_SP :						\
			switch(src1_dp->dt_prec){			\
				case PREC_SP:				\
					h_sp_rvmov(oap);		\
					break;				\
				case PREC_DP:				\
					h_dp2sp(oap);			\
					break;				\
				case PREC_BY:				\
					h_by2sp(oap);			\
					break;				\
				case PREC_IN:				\
					h_in2sp(oap);			\
					break;				\
				case PREC_DI:				\
					h_di2sp(oap);			\
					break;				\
				case PREC_LI:				\
					h_li2sp(oap);			\
					break;				\
				case PREC_UBY:				\
					h_uby2sp(oap);			\
					break;				\
				case PREC_UIN:				\
					h_uin2sp(oap);			\
					break;				\
				case PREC_UDI:				\
					h_udi2sp(oap);			\
					break;				\
				case PREC_ULI:				\
					h_uli2sp(oap);			\
					break;				\
				default:				\
					sprintf(DEFAULT_ERROR_STRING,		\
			"Unexpected source precision %s (function %s)",	\
		name_for_prec(src1_dp->dt_prec),#name);			\
					NWARN(DEFAULT_ERROR_STRING);		\
					break;				\
			}						\
			break;						\
		case PREC_DP :						\
			switch(src1_dp->dt_prec){			\
				case PREC_SP:				\
					h_sp2dp(oap);			\
					break;				\
				case PREC_DP:				\
					h_dp_rvmov(oap);		\
					break;				\
				case PREC_BY:				\
					h_by2dp(oap);			\
					break;				\
				case PREC_IN:				\
					h_in2dp(oap);			\
					break;				\
				case PREC_DI:				\
					h_di2dp(oap);			\
					break;				\
				case PREC_LI:				\
					h_li2dp(oap);			\
					break;				\
				case PREC_UBY:				\
					h_uby2dp(oap);			\
					break;				\
				case PREC_UIN:				\
					h_uin2dp(oap);			\
					break;				\
				case PREC_UDI:				\
					h_udi2dp(oap);			\
					break;				\
				case PREC_ULI:				\
					h_uli2dp(oap);			\
					break;				\
				default:				\
					sprintf(DEFAULT_ERROR_STRING,		\
			"Unexpected source precision %s (function %s)",	\
		name_for_prec(src1_dp->dt_prec),#name);			\
					NWARN(DEFAULT_ERROR_STRING);		\
					break;				\
			}						\
			break;						\
		case PREC_BY :						\
			switch(src1_dp->dt_prec){			\
				case PREC_SP:				\
					h_sp2by(oap);			\
					break;				\
				case PREC_DP:				\
					h_dp2by(oap);			\
					break;				\
				case PREC_BY:				\
					h_by_rvmov(oap);		\
					break;				\
				case PREC_IN:				\
					h_in2by(oap);			\
					break;				\
				case PREC_DI:				\
					h_di2by(oap);			\
					break;				\
				case PREC_LI:				\
					h_li2by(oap);			\
					break;				\
				case PREC_UBY:				\
					h_uby2by(oap);			\
					break;				\
				case PREC_UIN:				\
					h_uin2by(oap);			\
					break;				\
				case PREC_UDI:				\
					h_udi2by(oap);			\
					break;				\
				case PREC_ULI:				\
					h_uli2by(oap);			\
					break;				\
				default:				\
					sprintf(DEFAULT_ERROR_STRING,		\
			"Unexpected source precision %s (function %s)",	\
		name_for_prec(src1_dp->dt_prec),#name);			\
					NWARN(DEFAULT_ERROR_STRING);		\
					break;				\
			}						\
			break;						\
		case PREC_IN :						\
			switch(src1_dp->dt_prec){			\
				case PREC_SP:				\
					h_sp2in(oap);			\
					break;				\
				case PREC_DP:				\
					h_dp2in(oap);			\
					break;				\
				case PREC_BY:				\
					h_by2in(oap);			\
					break;				\
				case PREC_IN:				\
					h_in_rvmov(oap);		\
					break;				\
				case PREC_DI:				\
					h_di2in(oap);			\
					break;				\
				case PREC_LI:				\
					h_li2in(oap);			\
					break;				\
				case PREC_UBY:				\
					h_uby2in(oap);			\
					break;				\
				case PREC_UIN:				\
					h_uin2in(oap);			\
					break;				\
				case PREC_UDI:				\
					h_udi2in(oap);			\
					break;				\
				case PREC_ULI:				\
					h_uli2in(oap);			\
					break;				\
				default:				\
					sprintf(DEFAULT_ERROR_STRING,		\
			"Unexpected source precision %s (function %s)",	\
		name_for_prec(src1_dp->dt_prec),#name);			\
					NWARN(DEFAULT_ERROR_STRING);		\
					break;				\
			}						\
			break;						\
		case PREC_DI :						\
			switch(src1_dp->dt_prec){			\
				case PREC_SP:				\
					h_sp2di(oap);			\
					break;				\
				case PREC_DP:				\
					h_dp2di(oap);			\
					break;				\
				case PREC_BY:				\
					h_by2di(oap);			\
					break;				\
				case PREC_IN:				\
					h_in2di(oap);			\
					break;				\
				case PREC_DI:				\
					h_di_rvmov(oap);		\
					break;				\
				case PREC_LI:				\
					h_li2di(oap);			\
					break;				\
				case PREC_UBY:				\
					h_uby2di(oap);			\
					break;				\
				case PREC_UIN:				\
					h_uin2di(oap);			\
					break;				\
				case PREC_UDI:				\
					h_udi2di(oap);			\
					break;				\
				case PREC_ULI:				\
					h_uli2di(oap);			\
					break;				\
				default:				\
					sprintf(DEFAULT_ERROR_STRING,		\
			"Unexpected source precision %s (function %s)",	\
		name_for_prec(src1_dp->dt_prec),#name);			\
					NWARN(DEFAULT_ERROR_STRING);		\
					break;				\
			}						\
			break;						\
		case PREC_LI :						\
			switch(src1_dp->dt_prec){			\
				case PREC_SP:				\
					h_sp2li(oap);			\
					break;				\
				case PREC_DP:				\
					h_dp2li(oap);			\
					break;				\
				case PREC_BY:				\
					h_by2li(oap);			\
					break;				\
				case PREC_IN:				\
					h_in2li(oap);			\
					break;				\
				case PREC_DI:				\
					h_di2li(oap);			\
					break;				\
				case PREC_LI:				\
					h_li_rvmov(oap);		\
					break;				\
				case PREC_UBY:				\
					h_uby2li(oap);			\
					break;				\
				case PREC_UIN:				\
					h_uin2li(oap);			\
					break;				\
				case PREC_UDI:				\
					h_udi2li(oap);			\
					break;				\
				case PREC_ULI:				\
					h_uli2li(oap);			\
					break;				\
				default:				\
					sprintf(DEFAULT_ERROR_STRING,		\
			"Unexpected source precision %s (function %s)",	\
		name_for_prec(src1_dp->dt_prec),#name);			\
					NWARN(DEFAULT_ERROR_STRING);		\
					break;				\
			}						\
			break;						\
		case PREC_UBY :						\
			switch(src1_dp->dt_prec){			\
				case PREC_SP:				\
					h_sp2uby(oap);			\
					break;				\
				case PREC_DP:				\
					h_dp2uby(oap);			\
					break;				\
				case PREC_BY:				\
					h_by2uby(oap);			\
					break;				\
				case PREC_IN:				\
					h_in2uby(oap);			\
					break;				\
				case PREC_DI:				\
					h_di2uby(oap);			\
					break;				\
				case PREC_LI:				\
					h_li2uby(oap);			\
					break;				\
				case PREC_UBY:				\
					h_uby_rvmov(oap);		\
					break;				\
				case PREC_UIN:				\
					h_uin2uby(oap);			\
					break;				\
				case PREC_UDI:				\
					h_udi2uby(oap);			\
					break;				\
				case PREC_ULI:				\
					h_uli2uby(oap);			\
					break;				\
				default:				\
					sprintf(DEFAULT_ERROR_STRING,		\
			"Unexpected source precision %s (function %s)",	\
		name_for_prec(src1_dp->dt_prec),#name);			\
					NWARN(DEFAULT_ERROR_STRING);		\
					break;				\
			}						\
			break;						\
		case PREC_UIN :						\
			switch(src1_dp->dt_prec){			\
				case PREC_SP:				\
					h_sp2uin(oap);			\
					break;				\
				case PREC_DP:				\
					h_dp2uin(oap);			\
					break;				\
				case PREC_BY:				\
					h_by2uin(oap);			\
					break;				\
				case PREC_IN:				\
					h_in2uin(oap);			\
					break;				\
				case PREC_DI:				\
					h_di2uin(oap);			\
					break;				\
				case PREC_LI:				\
					h_li2uin(oap);			\
					break;				\
				case PREC_UBY:				\
					h_uby2uin(oap);			\
					break;				\
				case PREC_UIN:				\
					h_uin_rvmov(oap);		\
					break;				\
				case PREC_UDI:				\
					h_udi2uin(oap);			\
					break;				\
				case PREC_ULI:				\
					h_uli2uin(oap);			\
					break;				\
				default:				\
					sprintf(DEFAULT_ERROR_STRING,		\
			"Unexpected source precision %s (function %s)",	\
		name_for_prec(src1_dp->dt_prec),#name);			\
					NWARN(DEFAULT_ERROR_STRING);		\
					break;				\
			}						\
			break;						\
		case PREC_UDI :						\
			switch(src1_dp->dt_prec){			\
				case PREC_SP:				\
					h_sp2udi(oap);			\
					break;				\
				case PREC_DP:				\
					h_dp2udi(oap);			\
					break;				\
				case PREC_BY:				\
					h_by2udi(oap);			\
					break;				\
				case PREC_IN:				\
					h_in2udi(oap);			\
					break;				\
				case PREC_DI:				\
					h_di2udi(oap);			\
					break;				\
				case PREC_LI:				\
					h_li2udi(oap);			\
					break;				\
				case PREC_UBY:				\
					h_uby2udi(oap);			\
					break;				\
				case PREC_UIN:				\
					h_uin2udi(oap);			\
					break;				\
				case PREC_UDI:				\
					h_udi_rvmov(oap);		\
					break;				\
				case PREC_ULI:				\
					h_uli2udi(oap);			\
					break;				\
				default:				\
					sprintf(DEFAULT_ERROR_STRING,		\
			"Unexpected source precision %s (function %s)",	\
		name_for_prec(src1_dp->dt_prec),#name);			\
					NWARN(DEFAULT_ERROR_STRING);		\
					break;				\
			}						\
			break;						\
		case PREC_ULI :						\
			switch(src1_dp->dt_prec){			\
				case PREC_SP:				\
					h_sp2uli(oap);			\
					break;				\
				case PREC_DP:				\
					h_dp2uli(oap);			\
					break;				\
				case PREC_BY:				\
					h_by2uli(oap);			\
					break;				\
				case PREC_IN:				\
					h_in2uli(oap);			\
					break;				\
				case PREC_DI:				\
					h_di2uli(oap);			\
					break;				\
				case PREC_LI:				\
					h_li2uli(oap);			\
					break;				\
				case PREC_UBY:				\
					h_uby2uli(oap);			\
					break;				\
				case PREC_UIN:				\
					h_uin2uli(oap);			\
					break;				\
				case PREC_UDI:				\
					h_udi2uli(oap);			\
					break;				\
				case PREC_ULI:				\
					h_uli_rvmov(oap);		\
					break;				\
				default:				\
					sprintf(DEFAULT_ERROR_STRING,		\
			"Unexpected source precision %s (function %s)",	\
		name_for_prec(src1_dp->dt_prec),#name);			\
					NWARN(DEFAULT_ERROR_STRING);		\
					break;				\
			}						\
			break;						\
		default:						\
			sprintf(DEFAULT_ERROR_STRING,				\
	"Unexpected destination precision %s (function %s).",		\
				name_for_prec(dst_dp->dt_prec),#name);	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}


// Host call macros

// Here RC means real or complex (or quaternion!?)

#define H_CALL_ALL_RC(gpu_func)						\
									\
	void g_##gpu_func(Vec_Obj_Args *oap)				\
	{								\
		GPU_PREC_SWITCH_RC( gpu_func, oap->oa_dest, oap )	\
	}


#define H_CALL_ALL(gpu_func)						\
									\
	void g_##gpu_func(Vec_Obj_Args *oap)				\
	{								\
		GPU_PREC_SWITCH( gpu_func, oap->oa_dest, oap )		\
	}


#define H_CALL_ALL_B(gpu_func)						\
									\
	void g_##gpu_func(Vec_Obj_Args *oap)				\
	{								\
		GPU_PREC_SWITCH_B( gpu_func, oap->oa_dest, oap )	\
	}

#define H_CALL_MAP(gpu_func)						\
									\
	void g_##gpu_func(Vec_Obj_Args *oap)				\
	{								\
		GPU_PREC_SWITCH( gpu_func, oap->oa_dp[0], oap )		\
	}

#define H_CALL_F(gpu_func)						\
									\
	void g_##gpu_func(Vec_Obj_Args *oap)				\
	{								\
		GPU_PREC_SWITCH_F( gpu_func, oap->oa_dest, oap )	\
	}

#define H_CALL_S(gpu_func)						\
									\
	void g_##gpu_func(Vec_Obj_Args *oap)				\
	{								\
		GPU_PREC_SWITCH_S( gpu_func, oap->oa_dest, oap )	\
	}

#define H_CALL_I(gpu_func)						\
									\
	void g_##gpu_func(Vec_Obj_Args *oap)				\
	{								\
		GPU_PREC_SWITCH_I( gpu_func, oap->oa_dest, oap )	\
	}


// Menu calls declare the host call and a menu-callable wrapper

#define MENU_CALL_3V_F(gpu_func )				\
									\
	H_CALL_F( gpu_func )						\
									\
	GENERIC_CALL_3V( gpu_func )

#define GENERIC_CALL_3V( gpu_func )			\
									\
	START_MENU_CALL_3V( MENU_FUNC_NAME(gpu_func) )					\
	g_##gpu_func( & oargs );					\
	FINISH_CUDA_MENU_FUNC						\
	}

#define MENU_CALL_3V_I(gpu_func )				\
									\
	H_CALL_I( gpu_func )						\
									\
	GENERIC_CALL_3V( gpu_func )

#define MENU_CALL_3V_S(gpu_func )					\
									\
	H_CALL_S( gpu_func )						\
									\
	GENERIC_CALL_3V( gpu_func )

#define MENU_CALL_3V(gpu_func )						\
									\
	H_CALL_ALL( gpu_func )						\
									\
	GENERIC_CALL_3V( gpu_func )

#define MENU_CALL_3V_RC(gpu_func )					\
									\
	H_CALL_ALL_RC( gpu_func )					\
									\
	GENERIC_CALL_3V( gpu_func )

#define MENU_CALL_1S_1V_RC(gpu_func )					\
									\
	H_CALL_ALL_RC( gpu_func )					\
									\
	GENERIC_CALL_1S_1( gpu_func )

#define MENU_CALL_1S_1V_F(gpu_func )					\
									\
	H_CALL_F( gpu_func )						\
									\
	GENERIC_CALL_1S_1( gpu_func )

#define FUNC_NAME(name)		const char *func_name=#name;

#define START_MENU_CALL_3V( name )					\
									\
	COMMAND_FUNC(name) 						\
	{								\
		FUNC_NAME(name)						\
		CUDA_MENU_FUNC_DECLS					\
		Vec_Obj_Args oargs;					\
		Data_Obj *dst_dp, *src1_dp, *src2_dp;			\
									\
		dst_dp = PICK_OBJ("destination object");		\
		src1_dp = PICK_OBJ("first source object");		\
		src2_dp = PICK_OBJ("second source object");		\
									\
		if( dst_dp == NO_OBJ || src1_dp == NO_OBJ || src2_dp == NO_OBJ ){				\
			return;						\
		}							\
									\
		CHECK_GPU_OBJ(dst_dp)					\
		CHECK_GPU_OBJ(src1_dp)					\
		CHECK_GPU_OBJ(src2_dp)					\
									\
		CHECK_SAME_GPU(dst_dp, src1_dp)				\
		CHECK_SAME_GPU(src1_dp, src2_dp)			\
									\
		CHECK_CONTIGUITY(dst_dp)				\
		CHECK_CONTIGUITY(src1_dp)				\
		CHECK_CONTIGUITY(src2_dp)				\
									\
		CHECK_SAME_SIZE(dst_dp, src1_dp, func_name)		\
		CHECK_SAME_SIZE(src1_dp, src2_dp, func_name)		\
		CHECK_SAME_PREC(dst_dp, src1_dp,func_name)		\
		CHECK_SAME_PREC(src1_dp, src2_dp,func_name)		\
									\
		/* BUG need to check that all operands have same type */\
		/* ALSO there are a few special cases			\
		 * where mixed types are required...			\
		 * see lib newvec... */					\
		setvarg3(&oargs,dst_dp,src1_dp,src2_dp);


#define H_CALL_CONV(gpu_func)						\
									\
	void g_##gpu_func( Vec_Obj_Args *oap )				\
	{								\
	GPU_PREC_SWITCH_CONV( gpu_func, oap->oa_dest, oap->oa_dp[0], oap )	\
	}


#define MENU_CALL_2V_CONV(gpu_func )				\
								\
	H_CALL_CONV(gpu_func)					\
	START_MENU_CALL_2V_CONV( MENU_FUNC_NAME(gpu_func) )	\
	FINISH_CUDA_MENU_FUNC					\
	}

/* vset */
#define MENU_CALL_1S_1_B(gpu_func )			\
							\
	H_CALL_ALL_B(gpu_func)				\
	GENERIC_CALL_1S_1(gpu_func)

#define MENU_CALL_1S_1(gpu_func )			\
							\
	H_CALL_ALL(gpu_func)				\
	GENERIC_CALL_1S_1(gpu_func)

#define MENU_CALL_1V_2SCALAR(gpu_func )			\
							\
	H_CALL_ALL(gpu_func)				\
	GENERIC_CALL_1V_2SCALAR(gpu_func)

#define MENU_CALL_1V_3SCALAR(gpu_func )			\
							\
	H_CALL_ALL(gpu_func)				\
	GENERIC_CALL_1V_3SCALAR(gpu_func)

#define MENU_CALL_2V_S(gpu_func )			\
							\
	H_CALL_S(gpu_func)				\
	GENERIC_CALL_2V(gpu_func)

#define MENU_CALL_2V_I(gpu_func )			\
							\
	H_CALL_I(gpu_func)				\
	GENERIC_CALL_2V(gpu_func)

#define MENU_CALL_2V_F(gpu_func )			\
							\
	H_CALL_F(gpu_func)				\
	GENERIC_CALL_2V(gpu_func)

#define GENERIC_CALL_2V(gpu_func)			\
							\
	START_MENU_CALL_2V( MENU_FUNC_NAME(gpu_func) )	\
	g_##gpu_func( & oargs );			\
	FINISH_CUDA_MENU_FUNC				\
	}

#define GENERIC_CALL_1V(gpu_func)				\
									\
	START_MENU_CALL_1V( MENU_FUNC_NAME(gpu_func) )					\
	g_##gpu_func( & oargs );					\
	FINISH_CUDA_MENU_FUNC						\
	}


#define GENERIC_CALL_1S_1(gpu_func)				\
									\
	START_MENU_CALL_1S_1( MENU_FUNC_NAME(gpu_func) )		\
	g_##gpu_func( & oargs );					\
	FINISH_CUDA_MENU_FUNC						\
	}

#define GENERIC_CALL_1V_2SCALAR(gpu_func)				\
									\
	START_MENU_CALL_1V_2SCALAR( MENU_FUNC_NAME(gpu_func) )				\
	g_##gpu_func( & oargs );					\
	FINISH_CUDA_MENU_FUNC						\
	}


#define GENERIC_CALL_1V_3SCALAR(gpu_func)				\
									\
	START_MENU_CALL_1V_3SCALAR( MENU_FUNC_NAME(gpu_func) )				\
	g_##gpu_func( & oargs );					\
	FINISH_CUDA_MENU_FUNC						\
	}


#define MENU_CALL_2V(gpu_func )				\
									\
	H_CALL_ALL(gpu_func)						\
	GENERIC_CALL_2V(gpu_func)

#define MENU_CALL_2V_SCALAR(gpu_func )			\
									\
	H_CALL_ALL(gpu_func)						\
	GENERIC_CALL_2V_SCALAR( gpu_func )

#define GENERIC_CALL_2V_SCALAR( gpu_func )			\
									\
	START_MENU_CALL_2V_SCALAR( MENU_FUNC_NAME(gpu_func) )				\
	g_##gpu_func( & oargs );					\
	FINISH_CUDA_MENU_FUNC						\
	}
	
#define MENU_CALL_2V_SCALAR_I(gpu_func )			\
									\
	H_CALL_I(gpu_func)						\
	GENERIC_CALL_2V_SCALAR( gpu_func )
	
#define MENU_CALL_2V_SCALAR_F(gpu_func )			\
									\
	H_CALL_F(gpu_func)						\
	GENERIC_CALL_2V_SCALAR( gpu_func )

#define MENU_CALL_2V_SCALAR_S(gpu_func )			\
									\
	H_CALL_S(gpu_func)						\
	GENERIC_CALL_2V_SCALAR( gpu_func )

	
#define START_MENU_CALL_2V( name )					\
									\
	COMMAND_FUNC(name)						\
	{								\
		FUNC_NAME(name)						\
		CUDA_MENU_FUNC_DECLS					\
									\
		Vec_Obj_Args oargs;					\
		Data_Obj *dst_dp, *src1_dp;				\
									\
		dst_dp = PICK_OBJ("destination object");		\
		src1_dp = PICK_OBJ("source object");			\
									\
		if( dst_dp == NO_OBJ || src1_dp == NO_OBJ ) return;	\
									\
		/* make sure these objects are not RAM objects */	\
		CHECK_GPU_OBJ(dst_dp);					\
		CHECK_GPU_OBJ(src1_dp);					\
									\
		/* make sure these objects are all on the same GPU */	\
		CHECK_SAME_GPU(dst_dp,src1_dp)				\
									\
		/* check for matching sizes and types */		\
		CHECK_SAME_SIZE(dst_dp,src1_dp,func_name)		\
		CHECK_SAME_PREC(dst_dp,src1_dp,func_name)		\
									\
		setvarg2(&oargs,dst_dp,src1_dp);

#ifdef HAVE_CUDA

#define SET_MAX_THREADS(dp)						\
									\
		/* Get the max_threads_per_block */			\
		insure_cuda_device( dp );				\
		max_threads_per_block = curr_cdp->cudev_prop.maxThreadsPerBlock;

#else

#define SET_MAX_THREADS(dp)

#endif

#define START_MENU_CALL_2V_CONV( name )					\
									\
	COMMAND_FUNC(name)						\
	{								\
		FUNC_NAME(name)						\
		CUDA_MENU_FUNC_DECLS					\
									\
		Vec_Obj_Args oargs;					\
		Data_Obj *dst_dp, *src1_dp;				\
									\
		dst_dp = PICK_OBJ("destination object");		\
		src1_dp = PICK_OBJ("source object");			\
									\
		if( dst_dp == NO_OBJ || src1_dp == NO_OBJ ) return;	\
									\
		/* make sure these objects are not RAM objects */	\
		CHECK_GPU_OBJ(dst_dp);					\
		CHECK_GPU_OBJ(src1_dp);					\
									\
		/* make sure these objects are all on the same GPU */	\
		CHECK_SAME_GPU(dst_dp,src1_dp)				\
									\
		/* check for matching sizes but NOT types */		\
		CHECK_SAME_SIZE(dst_dp,src1_dp,func_name)				\
									\
		setvarg2(&oargs,dst_dp,src1_dp);




#define START_MENU_CALL_2V_SCALAR( name )				\
									\
	COMMAND_FUNC( name )						\
	{								\
		FUNC_NAME(name)						\
		CUDA_MENU_FUNC_DECLS					\
		Vec_Obj_Args oargs;					\
		Data_Obj *dst_dp, *src1_dp;				\
		Scalar_Value sv;					\
		double scalar_op;					\
									\
		dst_dp = PICK_OBJ("destination object");		\
		src1_dp = PICK_OBJ("source object");			\
		scalar_op = HOW_MUCH((char *)"scalar operand");		\
									\
		if( dst_dp == NO_OBJ || src1_dp == NO_OBJ ) return;	\
									\
		/* make sure these objects are not RAM objects */	\
		CHECK_GPU_OBJ(dst_dp);					\
		CHECK_GPU_OBJ(src1_dp);					\
									\
		/* make sure these objects are all on the same GPU */	\
		CHECK_SAME_GPU(dst_dp,src1_dp)				\
									\
		/*check if contiguous */				\
		CHECK_CONTIGUITY(dst_dp)				\
		CHECK_CONTIGUITY(src1_dp)				\
									\
		CHECK_SAME_SIZE(dst_dp,src1_dp,func_name)		\
		CHECK_SAME_PREC(dst_dp,src1_dp,func_name)		\
									\
		setvarg2(&oargs,dst_dp,src1_dp);			\
		oargs.oa_svp[0] = &sv;					\
		cast_to_scalar_value(QSP_ARG  &sv,dst_dp->dt_prec,scalar_op);
		


#define START_MENU_CALL_1S_1( name )				\
									\
	COMMAND_FUNC( name )						\
	{								\
		FUNC_NAME(name)						\
		CUDA_MENU_FUNC_DECLS					\
		Vec_Obj_Args oargs;					\
		Data_Obj *dst_dp;					\
		Scalar_Value sv;					\
		double scalar_op;					\
									\
		dst_dp = PICK_OBJ("destination object");		\
		scalar_op = HOW_MUCH((char *)"scalar operand");		\
									\
		if( dst_dp == NO_OBJ ) return;				\
									\
		/* make sure these objects are not RAM objects */	\
		CHECK_GPU_OBJ(dst_dp);					\
									\
		/*check if contiguous */				\
		CHECK_CONTIGUITY(dst_dp)				\
									\
		setvarg1(&oargs,dst_dp);				\
		oargs.oa_svp[0] = &sv;					\
		cast_to_scalar_value(QSP_ARG  &sv,dst_dp->dt_prec,scalar_op);
		


// BUG these prompts are for vramp, not generic...
#define START_MENU_CALL_1V_2SCALAR( name )				\
									\
	COMMAND_FUNC( name )						\
	{								\
		FUNC_NAME(name)						\
		CUDA_MENU_FUNC_DECLS					\
		Vec_Obj_Args oargs;					\
		Data_Obj *dst_dp;					\
		Scalar_Value sv1,sv2;					\
		double scal_op1, scal_op2;				\
									\
		dst_dp = PICK_OBJ("destination object");		\
		scal_op1 = HOW_MUCH((char *)"start value");		\
		scal_op2 = HOW_MUCH((char *)"increment");		\
									\
		if( dst_dp == NO_OBJ ) return;				\
									\
		/* make sure these objects are not RAM objects */	\
		CHECK_GPU_OBJ(dst_dp);					\
									\
		/*check if contiguous */				\
		CHECK_CONTIGUITY(dst_dp)				\
									\
		setvarg1(&oargs,dst_dp);				\
		oargs.oa_svp[0] = &sv1;					\
		cast_to_scalar_value(QSP_ARG  &sv1,dst_dp->dt_prec,scal_op1);	\
		oargs.oa_svp[1] = &sv2;					\
		cast_to_scalar_value(QSP_ARG  &sv2,dst_dp->dt_prec,scal_op2);


#define START_MENU_CALL_1V_3SCALAR( name )				\
									\
	COMMAND_FUNC( name )						\
	{								\
		FUNC_NAME(name)						\
		CUDA_MENU_FUNC_DECLS					\
		Vec_Obj_Args oargs;					\
		Data_Obj *dst_dp;					\
		Scalar_Value sv1,sv2,sv3;				\
		double scal_op1, scal_op2, scal_op3;			\
									\
		dst_dp = PICK_OBJ("destination object");		\
		scal_op1 = HOW_MUCH((char *)"start value");		\
		scal_op2 = HOW_MUCH((char *)"H increment");		\
		scal_op3 = HOW_MUCH((char *)"V increment");		\
									\
		if( dst_dp == NO_OBJ ) return;				\
									\
		/* make sure these objects are not RAM objects */	\
		CHECK_GPU_OBJ(dst_dp);					\
									\
		/*check if contiguous */				\
		CHECK_CONTIGUITY(dst_dp)				\
									\
		setvarg1(&oargs,dst_dp);				\
		oargs.oa_svp[0] = &sv1;					\
		cast_to_scalar_value(QSP_ARG  &sv1,dst_dp->dt_prec,scal_op1);	\
		oargs.oa_svp[1] = &sv2;					\
		cast_to_scalar_value(QSP_ARG  &sv2,dst_dp->dt_prec,scal_op2);	\
		oargs.oa_svp[2] = &sv3;					\
		cast_to_scalar_value(QSP_ARG  &sv3,dst_dp->dt_prec,scal_op3);

