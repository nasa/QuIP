// Copied from cuda

// BUG the test here should be that they are
// cuda devices, and live on the same cuda device!?

#define CHECK_GPU_OBJ( dp )						\
									\
	if( OBJ_IS_RAM(dp) )						\
	{								\
		sprintf(DEFAULT_ERROR_STRING,				\
	"%s:  object %s lives in %s data area, expected a GPU.",	\
			func_name,					\
			OBJ_NAME(dp), AREA_NAME(OBJ_AREA(dp)));		\
		NWARN(DEFAULT_ERROR_STRING);				\
		return;							\
	}

#define CHECK_SAME_GPU( dp1, dp2 )					\
									\
	if( OBJ_AREA(dp1) != OBJ_AREA(dp2) )					\
	{								\
		sprintf(DEFAULT_ERROR_STRING,				\
	"%s:  objects %s (%s) and %s (%s) are not on the same GPU!?",	\
	func_name,OBJ_NAME(dp1), AREA_NAME(OBJ_AREA(dp1)), OBJ_NAME(dp2),	\
					AREA_NAME(OBJ_AREA(dp2)));		\
		NWARN(DEFAULT_ERROR_STRING);				\
		return;							\
	}	
	
#define CHECK_SAME_SIZE( dp1, dp2, whence )				\
									\
	if( !dp_same_size(DEFAULT_QSP_ARG  dp1,dp2,whence) )			\
		return;
	
#define CHECK_SAME_PREC( dp1, dp2, whence )				\
									\
	if( !dp_same_prec(DEFAULT_QSP_ARG  dp1,dp2,whence) )			\
		return;
	
#define CHECK_CONTIGUITY( dp )						\
									\
	if (!is_contiguous(DEFAULT_QSP_ARG  dp))				\
	{								\
		sprintf(DEFAULT_ERROR_STRING,				\
	"Sorry, object %s is not contiguous.", OBJ_NAME(dp));		\
		NWARN(DEFAULT_ERROR_STRING);				\
		return;							\
	}

#define NEW_CHECK_CONTIGUITY( dp )					\
									\
	if (!is_contiguous(DEFAULT_QSP_ARG  dp)) {					\
		all_contig = 0;						\
		if( !is_evenly_spaced(dp) )				\
			all_eqsp = 0;					\
	}


#define GPU_PREC_SWITCH_RC(name,dst_dp,oap)				\
									\
fprintf(stderr,"GPU_PREC_SWITCH_RC %s\n",OBJ_NAME(dst_dp));\
longlist(DEFAULT_QSP_ARG  dst_dp);\
	SET_MAX_THREADS_FROM_OBJ( dst_dp )					\
	if( IS_COMPLEX(dst_dp) ){					\
		GPU_PREC_SWITCH_C(c##name,dst_dp,oap)			\
	} else {							\
		GPU_PREC_SWITCH(r##name,dst_dp,oap)			\
	}



// uc_prec - upper case prec

#define PREC_CASE(uc_prec,lc_prec,name)					\
									\
		case PREC_##uc_prec :					\
			HOST_TYPED_CALL_NAME(name,lc_prec)(oap);	\
			 break;



#define GPU_PREC_SWITCH(name, dst_dp, oap )				\
									\
fprintf(stderr,"GPU_PREC_SWITCH %s\n",OBJ_NAME(dst_dp));\
longlist(DEFAULT_QSP_ARG  dst_dp);\
	SET_MAX_THREADS_FROM_OBJ( dst_dp )					\
	switch( OBJ_PREC(dst_dp) ){					\
		PREC_CASE(SP,sp,name)					\
		PREC_CASE(DP,dp,name)					\
		PREC_CASE(BY,by,name)					\
		PREC_CASE(IN,in,name)					\
		PREC_CASE(DI,di,name)					\
		PREC_CASE(LI,li,name)					\
		PREC_CASE(UBY,uby,name)					\
		PREC_CASE(UIN,uin,name)					\
		PREC_CASE(UDI,udi,name)					\
		PREC_CASE(ULI,uli,name)					\
		default:						\
			/* GPU_PREC_SWITCH */				\
			sprintf(DEFAULT_ERROR_STRING,				\
	"Sorry, %s precision is not supported on the GPU for function %s.",\
				PREC_NAME(OBJ_PREC_PTR(dst_dp)),#name);	\
			NWARN(DEFAULT_ERROR_STRING);				\
			break;						\
	}

/* All the usual types PLUS bitmap */

#define GPU_PREC_SWITCH_B(name, dst_dp, oap )				\
									\
fprintf(stderr,"GPU_PREC_SWITCH_B %s\n",OBJ_NAME(dst_dp));\
longlist(DEFAULT_QSP_ARG  dst_dp);\
	SET_MAX_THREADS_FROM_OBJ( dst_dp )					\
	switch( OBJ_PREC(dst_dp) ){					\
		PREC_CASE(SP,sp,name)					\
		PREC_CASE(DP,dp,name)					\
		PREC_CASE(BY,by,name)					\
		PREC_CASE(IN,in,name)					\
		PREC_CASE(DI,di,name)					\
		PREC_CASE(LI,li,name)					\
		PREC_CASE(UBY,uby,name)					\
		PREC_CASE(UIN,uin,name)					\
		PREC_CASE(UDI,udi,name)					\
		PREC_CASE(ULI,uli,name)					\
		PREC_CASE(BIT,bit,name)					\
		default:						\
			/* GPU_PREC_SWITCH */				\
			sprintf(DEFAULT_ERROR_STRING,			\
	"Sorry, %s precision is not supported on the GPU for function %s.",\
				PREC_NAME(OBJ_PREC_PTR(dst_dp)),#name);	\
			NWARN(DEFAULT_ERROR_STRING);			\
			break;						\
	}


// Signed types only

#define GPU_PREC_SWITCH_S(name, dst_dp, oap )				\
									\
fprintf(stderr,"GPU_PREC_SWITCH_S %s\n",OBJ_NAME(dst_dp));\
longlist(DEFAULT_QSP_ARG  dst_dp);\
	SET_MAX_THREADS_FROM_OBJ( dst_dp )					\
	switch( OBJ_PREC(dst_dp) ){					\
		PREC_CASE(SP,sp,name)					\
		PREC_CASE(DP,dp,name)					\
		PREC_CASE(BY,by,name)					\
		PREC_CASE(IN,in,name)					\
		PREC_CASE(DI,di,name)					\
		PREC_CASE(LI,li,name)					\
		default:						\
			/* GPU_PREC_SWITCH_S */				\
			sprintf(DEFAULT_ERROR_STRING,			\
	"Sorry, %s precision is not supported on the GPU for function %s.",\
				PREC_NAME(OBJ_PREC_PTR(dst_dp)),#name);	\
			NWARN(DEFAULT_ERROR_STRING);			\
			break;						\
	}


// Int types only

#define GPU_PREC_SWITCH_I(name, dst_dp, oap)				\
									\
fprintf(stderr,"GPU_PREC_SWITCH_I %s\n",OBJ_NAME(dst_dp));\
longlist(DEFAULT_QSP_ARG  dst_dp);\
	SET_MAX_THREADS_FROM_OBJ( dst_dp )					\
	switch( OBJ_PREC(dst_dp) ){					\
		PREC_CASE(BY,by,name)					\
		PREC_CASE(IN,in,name)					\
		PREC_CASE(DI,di,name)					\
		PREC_CASE(LI,li,name)					\
		PREC_CASE(UBY,uby,name)					\
		PREC_CASE(UIN,uin,name)					\
		PREC_CASE(UDI,udi,name)					\
		PREC_CASE(ULI,uli,name)					\
		default:						\
			/* GPU_PREC_SWITCH_I */				\
			sprintf(DEFAULT_ERROR_STRING,			\
	"Sorry, %s precision is not supported on the GPU for function %s.",\
				PREC_NAME(OBJ_PREC_PTR(dst_dp)),#name);	\
			NWARN(DEFAULT_ERROR_STRING);			\
			break;						\
	}

// float types only

#define GPU_PREC_SWITCH_F(name, dst_dp, oap)				\
fprintf(stderr,"GPU_PREC_SWITCH_F %s\n",OBJ_NAME(dst_dp));\
longlist(DEFAULT_QSP_ARG  dst_dp);\
	SET_MAX_THREADS_FROM_OBJ( dst_dp )					\
	switch( OBJ_PREC(dst_dp) ){					\
		PREC_CASE(SP,sp,name)					\
		PREC_CASE(DP,dp,name)					\
		default:						\
			/* GPU_PREC_SWITCH_F */				\
			sprintf(DEFAULT_ERROR_STRING,			\
	"Sorry, %s precision is not supported on the GPU for function %s.",\
				PREC_NAME(OBJ_PREC_PTR(dst_dp)),#name);	\
			NWARN(DEFAULT_ERROR_STRING);			\
			break;						\
	}																	



#define GPU_PREC_SWITCH_C(name, dst_dp, oap)				\
fprintf(stderr,"GPU_PREC_SWITCH_C %s\n",OBJ_NAME(dst_dp));\
longlist(DEFAULT_QSP_ARG  dst_dp);\
	SET_MAX_THREADS_FROM_OBJ( dst_dp )					\
	switch( OBJ_PREC(dst_dp) ){					\
		PREC_CASE(CPX,sp,name)					\
		PREC_CASE(DBLCPX,dp,name)				\
		default:						\
			/* GPU_PREC_SWITCH_F */				\
			sprintf(DEFAULT_ERROR_STRING,			\
	"Sorry, %s precision is not supported on the GPU for function %s.",\
				PREC_NAME(OBJ_PREC_PTR(dst_dp)),#name);	\
			NWARN(DEFAULT_ERROR_STRING);			\
			break;						\
	}																	

#define CONV_CASE(uc_prec,lc_prec_from,lc_prec_to)			\
									\
	case PREC_##uc_prec:						\
		CONV_FUNC_NAME(lc_prec_from,lc_prec_to)(oap);		\
		break;


#define SIGNED_CONV_CASES(prec_to)				\
								\
		CONV_CASE(BY,by,prec_to)			\
		CONV_CASE(IN,in,prec_to)			\
		CONV_CASE(DI,di,prec_to)			\
		CONV_CASE(LI,li,prec_to)

#define UNSIGNED_CONV_CASES(prec_to)				\
								\
		CONV_CASE(UBY,uby,prec_to)			\
		CONV_CASE(UIN,uin,prec_to)			\
		CONV_CASE(UDI,udi,prec_to)			\
		CONV_CASE(ULI,uli,prec_to)

#define FLOAT_CONV_CASES(prec_to)				\
								\
		CONV_CASE(SP,sp,prec_to)			\
		CONV_CASE(DP,dp,prec_to)


#define BAD_SRC_DEFAULT(src_dp,func_name)				\
									\
	default:							\
		sprintf(DEFAULT_ERROR_STRING,				\
			"Unexpected source precision %s (function %s)",	\
			PREC_NAME(OBJ_PREC_PTR(src_dp)),#func_name);	\
		NWARN(DEFAULT_ERROR_STRING);				\
		break;

#define GPU_PREC_SWITCH_CONV(name, dst_dp, src1_dp, oap)		\
									\
fprintf(stderr,"GPU_PREC_SWITCH_CONV %s\n",OBJ_NAME(dst_dp));\
longlist(DEFAULT_QSP_ARG  dst_dp);\
	SET_MAX_THREADS_FROM_OBJ( dst_dp )					\
	switch( OBJ_PREC(dst_dp) ){					\
		case PREC_SP :						\
			switch(OBJ_PREC(src1_dp)){			\
				PREC_CASE(SP,sp,rvmov)			\
				CONV_CASE(DP,dp,sp)			\
				SIGNED_CONV_CASES(sp)			\
				UNSIGNED_CONV_CASES(sp)			\
				BAD_SRC_DEFAULT(src1_dp,name)		\
			}						\
			break;						\
		case PREC_DP :						\
			switch(OBJ_PREC(src1_dp)){			\
				CONV_CASE(SP,sp,dp)			\
				PREC_CASE(DP,dp,rvmov)			\
				SIGNED_CONV_CASES(dp)			\
				UNSIGNED_CONV_CASES(dp)			\
				BAD_SRC_DEFAULT(src1_dp,name)		\
			}						\
			break;						\
		case PREC_BY :						\
			switch(OBJ_PREC(src1_dp)){			\
				FLOAT_CONV_CASES(by)			\
				PREC_CASE(BY,by,rvmov)			\
				CONV_CASE(IN,in,by)			\
				CONV_CASE(DI,di,by)			\
				CONV_CASE(LI,li,by)			\
				UNSIGNED_CONV_CASES(by)			\
				BAD_SRC_DEFAULT(src1_dp,name)		\
			}						\
			break;						\
		case PREC_IN :						\
			switch(OBJ_PREC(src1_dp)){			\
				FLOAT_CONV_CASES(in)			\
				CONV_CASE(BY,by,in)			\
				PREC_CASE(IN,in,rvmov)			\
				CONV_CASE(DI,di,in)			\
				CONV_CASE(LI,li,in)			\
				UNSIGNED_CONV_CASES(in)			\
				BAD_SRC_DEFAULT(src1_dp,name)		\
			}						\
			break;						\
		case PREC_DI :						\
			switch(OBJ_PREC(src1_dp)){			\
				FLOAT_CONV_CASES(di)			\
				CONV_CASE(BY,by,di)			\
				CONV_CASE(IN,in,di)			\
				PREC_CASE(DI,di,rvmov)			\
				CONV_CASE(LI,li,di)			\
				UNSIGNED_CONV_CASES(di)			\
				BAD_SRC_DEFAULT(src1_dp,name)		\
			}						\
			break;						\
		case PREC_LI :						\
			switch(OBJ_PREC(src1_dp)){			\
				FLOAT_CONV_CASES(li)			\
				CONV_CASE(BY,by,li)			\
				CONV_CASE(IN,in,li)			\
				CONV_CASE(DI,di,li)			\
				PREC_CASE(LI,li,rvmov)			\
				UNSIGNED_CONV_CASES(li)			\
				BAD_SRC_DEFAULT(src1_dp,name)		\
			}						\
			break;						\
		case PREC_UBY :						\
			switch(OBJ_PREC(src1_dp)){			\
				FLOAT_CONV_CASES(uby)			\
				SIGNED_CONV_CASES(uby)			\
				PREC_CASE(UBY,uby,rvmov)		\
				CONV_CASE(UIN,uin,uby)			\
				CONV_CASE(UDI,udi,uby)			\
				CONV_CASE(ULI,uli,uby)			\
				BAD_SRC_DEFAULT(src1_dp,name)		\
			}						\
			break;						\
		case PREC_UIN :						\
			switch(OBJ_PREC(src1_dp)){			\
				FLOAT_CONV_CASES(uin)			\
				SIGNED_CONV_CASES(uin)			\
				CONV_CASE(UBY,uby,uin)			\
				PREC_CASE(UIN,uin,rvmov)		\
				CONV_CASE(UDI,udi,uin)			\
				CONV_CASE(ULI,uli,uin)			\
				BAD_SRC_DEFAULT(src1_dp,name)		\
			}						\
			break;						\
		case PREC_UDI :						\
			switch(OBJ_PREC(src1_dp)){			\
				FLOAT_CONV_CASES(udi)			\
				SIGNED_CONV_CASES(udi)			\
				CONV_CASE(UBY,uby,udi)			\
				CONV_CASE(UIN,uin,udi)			\
				PREC_CASE(UDI,udi,rvmov)		\
				CONV_CASE(ULI,uli,udi)			\
				BAD_SRC_DEFAULT(src1_dp,name)		\
			}						\
			break;						\
		case PREC_ULI :						\
			switch(OBJ_PREC(src1_dp)){			\
				FLOAT_CONV_CASES(uli)			\
				SIGNED_CONV_CASES(uli)			\
				CONV_CASE(UBY,uby,uli)			\
				CONV_CASE(UIN,uin,uli)			\
				CONV_CASE(UDI,udi,uli)			\
				PREC_CASE(ULI,uli,rvmov)		\
				BAD_SRC_DEFAULT(src1_dp,name)		\
			}						\
			break;						\
		default:						\
			sprintf(DEFAULT_ERROR_STRING,			\
	"Unexpected destination precision %s (function %s).",		\
				PREC_NAME(OBJ_PREC_PTR(dst_dp)),#name);	\
			NWARN(DEFAULT_ERROR_STRING);			\
			break;						\
	}


// Host call macros
//
// But what is the difference between the h_ prefix and the g_ prefix?

// Here RC means real or complex (or quaternion!?)
//
// eg vadd

#define H_CALL_ALL_RC(gpu_func)						\
									\
	void HOST_CALL_NAME(gpu_func)(HOST_CALL_ARG_DECLS)		\
	{								\
		GPU_PREC_SWITCH_RC( gpu_func, oap->oa_dest, oap )	\
	}


#define H_CALL_ALL(gpu_func)						\
									\
	void HOST_CALL_NAME(gpu_func)(HOST_CALL_ARG_DECLS)			\
	{								\
		GPU_PREC_SWITCH( gpu_func, oap->oa_dest, oap )		\
	}


#define H_CALL_ALL_B(gpu_func)						\
									\
	void HOST_CALL_NAME(gpu_func)(HOST_CALL_ARG_DECLS)			\
	{								\
		GPU_PREC_SWITCH_B( gpu_func, oap->oa_dest, oap )	\
	}

#define H_CALL_MAP(gpu_func)						\
									\
	void HOST_CALL_NAME(gpu_func)(HOST_CALL_ARG_DECLS)			\
	{								\
		GPU_PREC_SWITCH( gpu_func, oap->oa_dp[0], oap )		\
	}

#define H_CALL_F(gpu_func)						\
									\
	void HOST_CALL_NAME(gpu_func)(HOST_CALL_ARG_DECLS)			\
	{								\
		GPU_PREC_SWITCH_F( gpu_func, oap->oa_dest, oap )	\
	}

#define H_CALL_S(gpu_func)						\
									\
	void HOST_CALL_NAME(gpu_func)(HOST_CALL_ARG_DECLS)			\
	{								\
		GPU_PREC_SWITCH_S( gpu_func, oap->oa_dest, oap )	\
	}

#define H_CALL_I(gpu_func)						\
									\
	void HOST_CALL_NAME(gpu_func)(HOST_CALL_ARG_DECLS)			\
	{								\
		GPU_PREC_SWITCH_I( gpu_func, oap->oa_dest, oap )	\
	}


// Menu calls declare the host call and a menu-callable wrapper

#define MENU_CALL_3V_F(gpu_func )	H_CALL_F( gpu_func )
#define MENU_CALL_3V_I(gpu_func )	H_CALL_I( gpu_func )
#define MENU_CALL_3V_S(gpu_func )	H_CALL_S( gpu_func )
#define MENU_CALL_3V(gpu_func )		H_CALL_ALL( gpu_func )
#define MENU_CALL_3V_RC(gpu_func )	H_CALL_ALL_RC( gpu_func )
#define MENU_CALL_1S_1V_RC(gpu_func )	H_CALL_ALL_RC( gpu_func )

// cvset, and ?
#define MENU_CALL_1S_1V_F(gpu_func )	H_CALL_F( gpu_func )

#define H_CALL_CONV(gpu_func)						\
									\
	void HOST_CALL_NAME(gpu_func)( HOST_CALL_ARG_DECLS )				\
	{								\
	GPU_PREC_SWITCH_CONV( gpu_func, oap->oa_dest, oap->oa_dp[0], oap )	\
	}



/* vset */
#define MENU_CALL_1S_1_B(gpu_func )	H_CALL_ALL_B(gpu_func)
#define MENU_CALL_1S_1(gpu_func )	H_CALL_ALL(gpu_func)
#define MENU_CALL_1V_2SCALAR(gpu_func )	H_CALL_ALL(gpu_func)
// ramp2d
#define MENU_CALL_1V_3SCALAR(gpu_func )		H_CALL_ALL(gpu_func)
#define MENU_CALL_2V_S(gpu_func )		H_CALL_S(gpu_func)
#define MENU_CALL_2V_I(gpu_func )		H_CALL_I(gpu_func)
#define MENU_CALL_2V_F(gpu_func )		H_CALL_F(gpu_func)
#define MENU_CALL_2V(gpu_func )			H_CALL_ALL(gpu_func)
#define MENU_CALL_2V_SCALAR(gpu_func )		H_CALL_ALL(gpu_func)
#define MENU_CALL_2V_SCALAR_I(gpu_func )	H_CALL_I(gpu_func)
#define MENU_CALL_2V_SCALAR_F(gpu_func )	H_CALL_F(gpu_func)
#define MENU_CALL_2V_SCALAR_S(gpu_func )	H_CALL_S(gpu_func)
	
#define MENU_CALL_2V_CONV(gpu_func )		H_CALL_CONV(gpu_func)



