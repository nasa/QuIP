#define INSIST_CONTIG( dp , msg )					\
									\
	if( ! is_contiguous( DEFAULT_QSP_ARG  dp ) ){			\
		sprintf(DEFAULT_ERROR_STRING,				\
	"Sorry, object %s must be contiguous for %s (gpu only?).",	\
			OBJ_NAME(dp),msg);				\
		NWARN(DEFAULT_ERROR_STRING);				\
		return;							\
	}



#define INSIST_LENGTH( n , msg , name )					\
									\
		if( (n) == 1 ){						\
			sprintf(DEFAULT_ERROR_STRING,			\
	"Oops, kind of silly to do %s of 1-len vector %s!?",msg,name);	\
			NWARN(DEFAULT_ERROR_STRING);			\
			return;						\
		}


#define _VEC_FUNC_2V_PROJ_IDX(name,cpu_s1,cpu_s2,gpu_s1,gpu_s2)	H_CALL_PROJ_2V_IDX( name)
#define _VEC_FUNC_MM_NOCC(name,cpu_t1,cpu_t2,cpu_s,gpu_test1,gpu_test2)	\
						H_CALL_MM_NOCC( name )

#define _VEC_FUNC_MM(name,statement)		H_CALL_MM( name )
#define _VEC_FUNC_CPX_2V_PROJ(name,cpu_init_stat,cpu_loop_stat,gpu_expr_re,gpu_expr_im)	\
	H_CALL_PROJ_2V( name, std_cpx )

#define _VEC_FUNC_QUAT_2V_PROJ(name,cpu_init_stat,cpu_loop_stat,gpu_expr_re,gpu_expr_im1,gpu_expr_im2,gpu_expr_im3)	H_CALL_PROJ_2V( name, std_quat )

// This was H_CALL_MM - ???
// The statements are used in the kernels, this just declares the function that fixes the args
// and then calls the kernel...
#define _VEC_FUNC_2V_PROJ(name,cpu_init_stat,cpu_loop_stat,gpu_expr)	\
						H_CALL_PROJ_2V( name, std_type )

#define _VEC_FUNC_3V_PROJ(name,s1,s2,e1,e2)		H_CALL_PROJ_3V( name, std_type )
#define _VEC_FUNC_CPX_3V_PROJ(name,s1,s2,r1,i1,r2,i2)	H_CALL_PROJ_3V( name, std_cpx )
//#define _VEC_FUNC_QUAT_3V_PROJ(name,s1,s2,e1,e2)	H_CALL_PROJ_3V( name, std_quat )


//#define _VEC_FUNC_MM_IND(name,stat1,stat2)	H_CALL_MM_IND( name )
// Is VEC_FUNC_MM_IND not used any more???
//#define H_CALL_PROJ_2V_IDX(name)

