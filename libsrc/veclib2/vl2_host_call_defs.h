
#ifndef _VL2_HOST_CALL_UTILS_H_
#define _VL2_HOST_CALL_UTILS_H_ 

// copied from opencl...
// probably not specific, should be moved to include/veclib BUG

#define SETUP_EQSP_LEN						\
								\
	i=OBJ_MINDIM(OA_DEST(oap));				\
	len = OBJ_TYPE_DIM(OA_DEST(oap),i);			\
	i++;							\
	while( i <= OBJ_MAXDIM(OA_DEST(oap)) ){			\
		len *= OBJ_TYPE_DIM(OA_DEST(oap),i);		\
		i++;						\
	}

#define SETUP_EQSP_INCS1			\
						\
	i=OBJ_MINDIM(OA_DEST(oap));		\
	inc1 = OBJ_TYPE_INC(OA_DEST(oap),i);

#define SETUP_EQSP_INCS2			\
						\
	SETUP_EQSP_INCS1			\
	inc2 = OBJ_TYPE_INC(OA_SRC1(oap),i);

#define SETUP_EQSP_INCS3			\
						\
	SETUP_EQSP_INCS2			\
	inc2 = OBJ_TYPE_INC(OA_SRC2(oap),i);

#define SETUP_EQSP_INCS4			\
						\
	SETUP_EQSP_INCS3			\
	inc2 = OBJ_TYPE_INC(OA_SRC3(oap),i);

#define SETUP_EQSP_INCS5			\
						\
	SETUP_EQSP_INCS4			\
	inc2 = OBJ_TYPE_INC(OA_SRC4(oap),i);

// above stuff was cribbed from cuda code...
// How much can we use?

// The kernels need to be per-device...


#define DECLARE_EQSP_VARS_1	/* int inc1; */
#define DECLARE_EQSP_VARS_2	/* DECLARE_EQSP_VARS_1 int inc2; */
#define DECLARE_EQSP_VARS_3	/* DECLARE_EQSP_VARS_2 int inc3; */
#define DECLARE_EQSP_VARS_4	/* DECLARE_EQSP_VARS_3 int inc4; */
#define DECLARE_EQSP_VARS_5	/* DECLARE_EQSP_VARS_4 int inc5; */

#define DECLARE_EQSP_VARS_1SRC	/* int inc2; */
#define DECLARE_EQSP_VARS_2SRCS	DECLARE_EQSP_VARS_1SRC /* int inc3; */

#define DECLARE_EQSP_VARS_SBM_1	DECLARE_EQSP_VARS_1 /* int bm_inc; */
#define DECLARE_EQSP_VARS_SBM_2	DECLARE_EQSP_VARS_2 /* int bm_inc; */
#define DECLARE_EQSP_VARS_SBM_3	DECLARE_EQSP_VARS_3 /* int bm_inc; */

#define DECLARE_EQSP_VARS_DBM_2SRCS	DECLARE_EQSP_VARS_2SRCS /* int bm_inc; */
#define DECLARE_EQSP_VARS_DBM_1SRC	DECLARE_EQSP_VARS_1SRC /* int bm_inc; */

#define DECLARE_SLOW_VARS_1	DIM3 inc1;
#define DECLARE_SLOW_VARS_2	DECLARE_SLOW_VARS_1 DIM3 inc2;
#define DECLARE_SLOW_VARS_3	DECLARE_SLOW_VARS_2 DIM3 inc3;
#define DECLARE_SLOW_VARS_4	DECLARE_SLOW_VARS_3 DIM3 inc4;
#define DECLARE_SLOW_VARS_5	DECLARE_SLOW_VARS_4 DIM3 inc5;
#define DECLARE_SLOW_VARS_2SRCS	DIM3 inc2; DIM3 inc3;
#define DECLARE_SLOW_VARS_1SRC	DIM3 inc2;

#define DECLARE_SLOW_VARS_SBM_3	DECLARE_SLOW_VARS_3 DIM3 bm_inc;
#define DECLARE_SLOW_VARS_SBM_2	DECLARE_SLOW_VARS_2 DIM3 bm_inc;
#define DECLARE_SLOW_VARS_SBM_1	DECLARE_SLOW_VARS_1 DIM3 bm_inc;

#define DECLARE_SLOW_VARS_DBM_2SRCS	DECLARE_SLOW_VARS_2SRCS DIM3 bm_inc;
#define DECLARE_SLOW_VARS_DBM_1SRC	DECLARE_SLOW_VARS_1SRC DIM3 bm_inc;

#define GENERIC_HOST_FAST_CONV(name,bitmap,typ,type)		\
static void HOST_FAST_CALL_NAME(name)( LINK_FUNC_ARG_DECLS )	\
{	CPU_FAST_CALL_NAME(name)(LINK_FUNC_ARGS);	}

#define GENERIC_HOST_FAST_CALL(name,bitmap,typ,scalars,vectors)		\
static void HOST_FAST_CALL_NAME(name)( LINK_FUNC_ARG_DECLS )	\
{	CPU_FAST_CALL_NAME(name)(LINK_FUNC_ARGS);	}

#define GENERIC_HOST_EQSP_CONV(name,bitmap,typ,type)	\
static void HOST_EQSP_CALL_NAME(name) ( LINK_FUNC_ARG_DECLS )	\
{	CPU_EQSP_CALL_NAME(name)(LINK_FUNC_ARGS);	}

#define GENERIC_HOST_EQSP_CALL(name,bitmap,typ,scalars,vectors)	\
static void HOST_EQSP_CALL_NAME(name) ( LINK_FUNC_ARG_DECLS )	\
{	CPU_EQSP_CALL_NAME(name)(LINK_FUNC_ARGS);	}

#define GENERIC_HOST_SLOW_CONV(name,bitmap,typ,type)	\
static void HOST_SLOW_CALL_NAME(name)( LINK_FUNC_ARG_DECLS )	\
{	CPU_SLOW_CALL_NAME(name)(LINK_FUNC_ARGS);	}

#define GENERIC_HOST_SLOW_CALL(name,bitmap,typ,scalars,vectors)	\
static void HOST_SLOW_CALL_NAME(name)( LINK_FUNC_ARG_DECLS )	\
{	CPU_SLOW_CALL_NAME(name)(LINK_FUNC_ARGS);	}


// a bunch of nop's

#define SETUP_SLOW_LEN_5
#define SETUP_SLOW_LEN_4
#define SETUP_SLOW_LEN_3
#define SETUP_SLOW_LEN_2
#define SETUP_SLOW_LEN_1
#define SETUP_SLOW_LEN_NOCC
#define SETUP_SLOW_LEN_DBM_2SRCS
#define SETUP_SLOW_LEN_DBM_1SRC
#define SETUP_SLOW_LEN_DBM_
#define SETUP_SLOW_LEN_DBM_SBM
#define SETUP_SLOW_LEN_SBM_3
#define SETUP_SLOW_LEN_SBM_2
#define SETUP_SLOW_LEN_SBM_1

#endif /* _VL2_HOST_CALL_UTILS_H_ */

