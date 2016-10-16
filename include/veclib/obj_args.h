#ifndef _OBJ_ARGS_H_
#define _OBJ_ARGS_H_

#include "data_obj.h"
#include "veclib/dim3.h"

/* MAX_N_ARGS originally was 3.
 * Increased to 4 to accomodate bitmaps.
 * Increased to 5 when new conditional-assignment ops were added,
 * so that the bitmap arg didn't have to share space with src4.
 */

#define MAX_SRC_OBJECTS	4
#define MAX_N_ARGS	(MAX_SRC_OBJECTS+1)	// up to 4 sources, bitmap arg doesn't share
#define MAX_RETSCAL_ARGS	2		// objects used for scalar returns
#define MAX_SRCSCAL_ARGS	3		// values used for scalar operands



// What are these used for?

/* Not the machine precisions, because we include psuedo precisions (bit) and some mixed combos */

typedef enum {
	BY_ARGS,	/* 0 */
	IN_ARGS,	/* 1 */
	DI_ARGS,	/* 2 */
	LI_ARGS,	/* 3 */
	SP_ARGS,	/* 4 */
	DP_ARGS,	/* 5 */
	UBY_ARGS,	/* 6 */
	UIN_ARGS,	/* 7 */
	UDI_ARGS,	/* 8 */
	ULI_ARGS,	/* 9 */
	BYIN_ARGS,	/* 10 */		/* uby args, short result */
	INBY_ARGS,	/* 11 */		/* what is this for??? */
	INDI_ARGS,	/* 12 */		/* uin args, int32 result */
	SPDP_ARGS,	/* 13 */		/* sp args, dp result */
	BIT_ARGS,	/* 14 */
	N_ARGSET_PRECISIONS,/* 15 */
	INVALID_ARGSET_PREC
} argset_prec;	/* argsprec */

#define ARGPREC_UNSPECIFIED	(-1)

#define FUNCTYPE_FOR( argsprec, arg_type )				\
		( argsprec + (arg_type-1) * N_ARGSET_PRECISIONS )

#define TELL_FUNCTYPE(argsprec,arg_type)				\
sprintf(ERROR_STRING,"functype = %d, argsprec = %d (%s), arg_type = %d (%s)",\
( argsprec + (arg_type-1) * N_ARGSET_PRECISIONS ), \
argsprec, NAME_FOR_ARGSPREC(argsprec), \
arg_type, NAME_FOR_ARGTYPE(arg_type) );\
advise(ERROR_STRING);

#define NAME_FOR_ARGSPREC(i)	name_for_argsprec(i)

typedef enum {
	UNKNOWN_ARGS,
	REAL_ARGS,
	COMPLEX_ARGS,
	MIXED_ARGS,		/* real/complex */
	QUATERNION_ARGS,
	QMIXED_ARGS,		/* real/quaternion */
	N_ARGSET_TYPES,
	INVALID_ARGSET_TYPE
} argset_type;	/* argstype */


struct platform_device;	// incomplete type, defined later...

struct vec_obj_args {
	struct platform_device *	oa_pdp;
	Data_Obj *	oa_dest;			// result vector
	Data_Obj *	oa_sdp[MAX_RETSCAL_ARGS];	// for return scalars

	Data_Obj *	oa_dp[MAX_N_ARGS];		// source objects
	Scalar_Value *	oa_svp[MAX_SRCSCAL_ARGS];

	int		oa_flags;
	int		oa_functype;
	argset_prec	oa_argsprec;
	argset_type	oa_argstype;
} ;

#define NO_VEC_OBJ_ARGS	((Vec_Obj_Args *) NULL)

/* Flag values */
#define OARGS_CHECKED	1
//#define OARGS_RAM	2
//#define OARGS_GPU	4

#define HAS_CHECKED_ARGS(oap)	( (oap)->oa_flags & OARGS_CHECKED )
//#define HAS_RAM_ARGS(oap)	( (oap)->oa_flags & OARGS_CHECKED ? (oap)->oa_flags & OARGS_RAM : are_ram_args(oap) )
//#define HAS_GPU_ARGS(oap)	( (oap)->oa_flags & OARGS_CHECKED ? (oap)->oa_flags & OARGS_GPU : are_gpu_args(oap) )

#define HAS_MIXED_ARGS(oap)	( OA_ARGSTYPE(oap) == MIXED_ARGS )
#define HAS_QMIXED_ARGS(oap)	( OA_ARGSTYPE(oap) == QMIXED_ARGS )


/* In OpenCL, the data pointers aren't memory addresses, they
 * are pointers to a memory buffer struct.  So, we can just
 * add offsets to them, we have to pass the offset to the kernel.
 * And so we have to 
 * [...]
 *
 *  It is not clear why we need this struct - all of the information seems
 * to be present in the dp.  The one advantage is that we have dropped
 * the unnecessary information - but at the expense of copying data around.
 * That might be important if we did a lot of link execution ("display list"
 * - or, rather, "compute list"), but historically we have not.
 *
 */

typedef struct vector_arg {
	void *		varg_vp;	// data or struct ptr
	union {
		int	u_inc;
		Increment_Set *	u_isp;
	} varg_u;
	Dimension_Set *	varg_dsp;

#ifdef HAVE_OPENCL
#define OCL_OFFSET_TYPE	int
	OCL_OFFSET_TYPE	varg_offset;
#endif // HAVE_OPENCL

} Vector_Arg;

#define VARG_DIMSET(vap)	((vap).varg_dsp)
#define VARG_PTR(vap)		((vap).varg_vp)
#define VARG_INCSET(vap)	((vap).varg_u.u_isp)
#define VARG_INC(vap)		((vap).varg_u.u_inc)
#define VARG_OFFSET(vap)	((vap).varg_offset)

#define VARG_LEN(vap)		DS_N_ELTS(VARG_DIMSET(vap))

#define SET_VARG_DIMSET(vap,v)	VARG_DIMSET(vap) = v
#define SET_VARG_PTR(vap,v)	VARG_PTR(vap) = v
#define SET_VARG_INCSET(vap,v)	VARG_INCSET(vap) = v
#define SET_VARG_INC(vap,v)	VARG_INC(vap) = v

#ifdef HAVE_OPENCL
#define SET_VARG_OFFSET(vap,v)	VARG_OFFSET(vap) = v
#else // ! HAVE_OPENCL
#define SET_VARG_OFFSET(vap,v)
#endif // ! HAVE_OPENCL

// BUG we let CUDA and OPENCL do multidimensional iterations,
// but that limits us to 3 dimensions!?

typedef struct vector_args {
	Vector_Arg	va_dst;
	Vector_Arg	va_src[MAX_N_ARGS];
	Scalar_Value *	va_sval[3];		// BUG use symbolic constant
						// used for return scalars also?
	dimension_t	va_dbm_bit0;
	dimension_t	va_sbm_bit0;
	dimension_t	va_len;
/*#ifdef BUILD_FOR_GPU*/
// This really should be conditionally compiled, but currently
// BUILD_FOR_GPU isn't defined in the right spot...
// Really BUILD_FOR_GPU get set and unset during the build,
// but we need the struct to have a constant size.  We really
// should define a new symbol BUILD_WITH_GPU...  or HAVE_ANY_GPU
	DIM3		va_xyz_len;
	int		va_dim_indices[3];
/*#endif // BUILD_FOR_GPU*/
	argset_type	va_argstype;
	argset_prec	va_argsprec;
	int		va_functype;
	int		va_flags;
	struct platform_device *	va_pfdev;
	// For a projecting version of vmaxg,
	// we need more return vector args for values and n_occ...
	Vector_Arg	va_values;
	Vector_Arg	va_counts;
} Vector_Args;

#define VA_ARGSET_PREC(vap)	(vap)->va_argsprec
#define VA_ARGSET_TYPE(vap)	(vap)->va_argstype

#define VA_DEST(vap)	(vap)->va_dst
#define VA_SRC(vap,idx)	(vap)->va_src[idx]
#define VA_SRC1(vap)	VA_SRC(vap,0)
#define VA_SRC2(vap)	VA_SRC(vap,1)
#define VA_SRC3(vap)	VA_SRC(vap,2)
#define VA_SRC4(vap)	VA_SRC(vap,3)
#define VA_SRC5(vap)	VA_SRC(vap,4)
#define VA_DBM(vap)	VA_DEST(vap)
#define VA_SBM(vap)	VA_SRC5(vap)

#define VA_DEST_LEN(vap)	VARG_LEN( VA_DEST(vap) )
#define VA_SRC1_LEN(vap)	VARG_LEN( VA_SRC1(vap) )

#define VA_XYZ_LEN(val)		(vap)->va_xyz_len

#define VA_LEN_X(vap)		(vap)->va_xyz_len.x
#define VA_LEN_Y(vap)		(vap)->va_xyz_len.y
#define VA_LEN_Z(vap)		(vap)->va_xyz_len.z
#define SET_VA_LEN_X(vap,v)	(vap)->va_xyz_len.x = v
#define SET_VA_LEN_Y(vap,v)	(vap)->va_xyz_len.y = v
#define SET_VA_LEN_Z(vap,v)	(vap)->va_xyz_len.z = v

#define VA_DIM_INDEX(vap,which)		(vap)->va_dim_indices[which]
#define SET_VA_DIM_INDEX(vap,which,v)	(vap)->va_dim_indices[which] = v

// va_flags
#define VA_FAST_ARGS	1
#define VA_EQSP_ARGS	2
#define VA_SLOW_ARGS	4

#define VA_FLAGS(vap)			(vap)->va_flags
#define SET_VA_FLAGS(vap,v)		VA_FLAGS(vap) = v
#define SET_VA_FLAG_BIT(vap,bit)	VA_FLAGS(vap) |= bit
#define CLEAR_VA_FLAG_BIT(vap,bit)	VA_FLAGS(vap) &= ~(bit)

#define ARE_FAST_ARGS(vap)		(VA_FLAGS(vap) & VA_FAST_ARGS)
#define ARE_EQSP_ARGS(vap)		(VA_FLAGS(vap) & VA_EQSP_ARGS)
#define ARE_SLOW_ARGS(vap)		(VA_FLAGS(vap) & VA_SLOW_ARGS)

#define VA_PFDEV(vap)			(vap)->va_pfdev
#define SET_VA_PFDEV(vap,v)		(vap)->va_pfdev = v

extern void show_vec_args(Vector_Args *vap);	// for debug


/* Now we subtract 1 because the 0 code is "unknown" */

#define REAL_ARG_MASK	VL_TYPE_MASK(REAL_ARGS)
#define CPX_ARG_MASK	VL_TYPE_MASK(COMPLEX_ARGS)
#define MIXED_ARG_MASK	VL_TYPE_MASK(MIXED_ARGS)
#define QUAT_ARG_MASK	VL_TYPE_MASK(QUATERNION_ARGS)
#define QMIXD_ARG_MASK	VL_TYPE_MASK(QMIXED_ARGS)	/* lmnop, Q, R ...   P!  (well, what else?) */

#define RC_MASK		(REAL_ARG_MASK|CPX_ARG_MASK)
#define RCQ_MASK	(REAL_ARG_MASK|CPX_ARG_MASK|QUAT_ARG_MASK)
#define RCM_MASK	(REAL_ARG_MASK|CPX_ARG_MASK|MIXED_ARG_MASK)
#define CM_MASK		(CPX_ARG_MASK|MIXED_ARG_MASK)
#define RCMQ_MASK	(REAL_ARG_MASK|CPX_ARG_MASK|MIXED_ARG_MASK|QUAT_ARG_MASK)
#define RCMQP_MASK	(REAL_ARG_MASK|CPX_ARG_MASK|MIXED_ARG_MASK|QUAT_ARG_MASK|QMIXD_ARG_MASK)
#define QP_MASK		(QUAT_ARG_MASK|QMIXD_ARG_MASK)

// What is the type mask applied to???
// argset_type  REAL_ARGS etc...
#ifdef CAUTIOUS
#define VL_TYPE_MASK(code)		(code>=0&&code<N_ARGSET_TYPES?1<<(code-1):0)
#else // ! CAUTIOUS
#define VL_TYPE_MASK(code)		(1<<((code)-1))
#endif // ! CAUTIOUS
#define NAME_FOR_ARGTYPE(i)	name_for_argtype(i)

/* masks for the allowable machine precisions */
#define M_BY	(1<<PREC_BY)
#define M_IN	(1<<PREC_IN)
#define M_DI	(1<<PREC_DI)
#define M_LI	(1<<PREC_LI)
#define M_SP	(1<<PREC_SP)
#define M_DP	(1<<PREC_DP)
#define M_UBY	(1<<PREC_UBY)
#define M_UIN	(1<<PREC_UIN)
#define M_UDI	(1<<PREC_UDI)
#define M_ULI	(1<<PREC_ULI)
#define M_MM	(1<<N_MACHINE_PRECS)

#define M_BP	(M_SP|M_DP)
#define M_ALL	(M_BY|M_IN|M_DI|M_LI|M_SP|M_DP|M_UBY|M_UIN|M_UDI|M_ULI)
#define M_ALLMM	(M_ALL|M_MM)
#define M_BPDI	(M_SP|M_DP|M_DI)
#define M_AI	(M_BY|M_IN|M_DI|M_LI|M_UBY|M_UIN|M_UDI|M_ULI)
#define M_BPMM	(M_BP|M_MM)
#define NULLARG	0

/* (MISPLACED COMMENT - goes where?) some flags defining what types of scalars are used */

/* Macros for Vector_Args */

#define NEW_VEC_ARGS			((Vector_Args *)getbuf(sizeof(Vector_Args)))
#define VA_SRC_PTR(vap,idx)		VARG_PTR( VA_SRC(vap,idx) )
#define VA_DEST_PTR(vap)		VARG_PTR( VA_DEST(vap) )

#define VA_SRC1_PTR(vap)		VARG_PTR( VA_SRC(vap,0) )
#define VA_SRC2_PTR(vap)		VARG_PTR( VA_SRC(vap,1) )
#define VA_SRC3_PTR(vap)		VARG_PTR( VA_SRC(vap,2) )
#define VA_SRC4_PTR(vap)		VARG_PTR( VA_SRC(vap,3) )

#define VA_DEST_OFFSET(vap)		VARG_OFFSET( VA_DEST(vap) )
#define VA_SRC_OFFSET(vap,idx)		VARG_OFFSET( VA_SRC(vap,idx) )
#define VA_SRC1_OFFSET(vap)		VA_SRC_OFFSET(vap,0)
#define VA_SRC2_OFFSET(vap)		VA_SRC_OFFSET(vap,1)
#define VA_SRC3_OFFSET(vap)		VA_SRC_OFFSET(vap,2)
#define VA_SRC4_OFFSET(vap)		VA_SRC_OFFSET(vap,3)
#define VA_SRC5_OFFSET(vap)		VA_SRC_OFFSET(vap,4)
#define VA_SBM_OFFSET(vap)		VARG_OFFSET( VA_SBM(vap) )
#define VA_DBM_OFFSET(vap)		VARG_OFFSET( VA_DBM(vap) )

#define VA_LENGTH(vap)			(vap)->va_len

#define VA_DEST_DIMSET(vap)		VARG_DIMSET( VA_DEST(vap) )
#define VA_SRC_DIMSET(vap,idx)		VARG_DIMSET( VA_SRC(vap,idx) )
#define VA_SRC1_DIMSET(vap)		VARG_DIMSET( VA_SRC1(vap) )
#define VA_SRC2_DIMSET(vap)		VARG_DIMSET( VA_SRC2(vap) )

#define VA_DEST_INCSET(vap)		VARG_INCSET( VA_DEST(vap) )
#define VA_SRC_INCSET(vap,idx)		VARG_INCSET( VA_SRC(vap,idx) )
#define VA_SRC1_INCSET(vap)		VA_SRC_INCSET(vap,0)
#define VA_SRC2_INCSET(vap)		VA_SRC_INCSET(vap,1)
#define VA_SRC3_INCSET(vap)		VA_SRC_INCSET(vap,2)
#define VA_SRC4_INCSET(vap)		VA_SRC_INCSET(vap,3)
#define VA_SRC5_INCSET(vap)		VA_SRC_INCSET(vap,4)

#define VA_DEST_INC(vap)		VARG_INC( VA_DEST(vap) )
#define VA_SRC_INC(vap,idx)		VARG_INC( VA_SRC(vap,idx) )
#define VA_SRC1_INC(vap)		VA_SRC_INC(vap,0)
#define VA_SRC2_INC(vap)		VA_SRC_INC(vap,1)
#define VA_SRC3_INC(vap)		VA_SRC_INC(vap,2)
#define VA_SRC4_INC(vap)		VA_SRC_INC(vap,3)
#define VA_SRC5_INC(vap)		VA_SRC_INC(vap,4)
#define VA_SBM_INC(vap)			VARG_INC( VA_SRC5(vap) )
#define VA_DBM_INC(vap)			VA_DEST_INC( vap )

#define eqsp_dest_inc			VA_DEST_INC(vap)
#define eqsp_src1_inc			VA_SRC1_INC(vap)
#define eqsp_src2_inc			VA_SRC2_INC(vap)
#define eqsp_src3_inc			VA_SRC3_INC(vap)
#define eqsp_src4_inc			VA_SRC4_INC(vap)
#define eqsp_src5_inc			VA_SRC5_INC(vap)
#define eqsp_bit_inc			VA_SRC5_INC(vap)

#define VA_SBM_BIT0(vap)		(vap)->va_sbm_bit0
#define VA_DBM_BIT0(vap)		(vap)->va_dbm_bit0
#define VA_SBM_PTR(vap)			VARG_PTR( VA_SBM(vap) )
#define VA_DBM_PTR(vap)			VARG_PTR( VA_DBM(vap) )
#define VA_SBM_INCSET(vap)		VA_SRC5_INCSET(vap)

#define VA_SVAL(vap,idx)		(vap)->va_sval[idx]
#define VA_SVAL1(vap)			VA_SVAL(vap,0)
#define VA_SVAL2(vap)			VA_SVAL(vap,1)
#define VA_SVAL3(vap)			VA_SVAL(vap,2)

// BUG? use one or the other?
#define SET_VA_LEN(vap,l)		(vap)->va_len = l
#define SET_VA_LENGTH(vap,v)		(vap)->va_len = v

#define SET_VA_DEST_PTR(vap,ptr)	SET_VARG_PTR( VA_DEST(vap) , ptr )
#define SET_VA_SRC_PTR(vap,idx,ptr)	SET_VARG_PTR( VA_SRC(vap,idx) , ptr )
#define SET_VA_SRC1_PTR(vap,ptr)	SET_VA_SRC_PTR(vap,0,ptr)
#define SET_VA_SRC2_PTR(vap,ptr)	SET_VA_SRC_PTR(vap,1,ptr)
#define SET_VA_SRC3_PTR(vap,ptr)	SET_VA_SRC_PTR(vap,2,ptr)
#define SET_VA_SRC4_PTR(vap,ptr)	SET_VA_SRC_PTR(vap,3,ptr)
#define SET_VA_SRC5_PTR(vap,ptr)	SET_VA_SRC_PTR(vap,4,ptr)
#define SET_VA_SBM_PTR(vap,ptr)		SET_VARG_PTR( VA_SRC5(vap) , ptr )
#define SET_VA_DBM_PTR(vap,ptr)		SET_VARG_PTR( VA_DEST(vap) , ptr )

// We didn't have the destination dimset here?
// Why not???
#define SET_VA_DEST_DIMSET(vap,v)	SET_VARG_DIMSET( VA_DEST(vap), v )
#define SET_VA_SRC_DIMSET(vap,idx,v)	SET_VARG_DIMSET( VA_SRC(vap,idx), v )
#define SET_VA_SRC1_DIMSET(vap,v)	SET_VA_SRC_DIMSET(vap,0,v)
#define SET_VA_SRC2_DIMSET(vap,v)	SET_VA_SRC_DIMSET(vap,1,v)
#define SET_VA_SRC3_DIMSET(vap,v)	SET_VA_SRC_DIMSET(vap,2,v)
#define SET_VA_SRC4_DIMSET(vap,v)	SET_VA_SRC_DIMSET(vap,3,v)
#define SET_VA_SRC5_DIMSET(vap,v)	SET_VA_SRC_DIMSET(vap,4,v)
#define SET_VA_SBM_DIMSET(vap,v)	SET_VARG_DIMSET( VA_SRC5(vap), v )

#define SET_VA_DEST_INCSET(vap,v)	SET_VARG_INCSET( VA_DEST(vap), v )
#define SET_VA_SRC_INCSET(vap,idx,v)	SET_VARG_INCSET( VA_SRC(vap,idx), v )
#define SET_VA_SRC1_INCSET(vap,v)	SET_VA_SRC_INCSET(vap,idx,v)
#define SET_VA_SRC2_INCSET(vap,v)	SET_VARG_INCSET( VA_SRC2(vap), v )
#define SET_VA_SRC3_INCSET(vap,v)	SET_VARG_INCSET( VA_SRC3(vap), v )
#define SET_VA_SRC4_INCSET(vap,v)	SET_VARG_INCSET( VA_SRC4(vap), v )
#define SET_VA_SRC5_INCSET(vap,v)	SET_VARG_INCSET( VA_SRC4(vap), v )
#define SET_VA_SBM_INCSET(vap,v)	SET_VARG_INCSET( VA_SRC5(vap), v )

#define SET_VA_DEST_INC(vap,v)	SET_VARG_INC( VA_DEST(vap), v )
#define SET_VA_SRC_INC(vap,idx,v)	SET_VARG_INC( VA_SRC(vap,idx), v )
#define SET_VA_SRC1_INC(vap,v)		SET_VA_SRC_INC(vap,0,v)
#define SET_VA_SRC2_INC(vap,v)		SET_VA_SRC_INC(vap,1,v)
#define SET_VA_SRC3_INC(vap,v)		SET_VA_SRC_INC(vap,2,v)
#define SET_VA_SRC4_INC(vap,v)		SET_VA_SRC_INC(vap,3,v)
#define SET_VA_SRC5_INC(vap,v)		SET_VA_SRC_INC(vap,4,v)

#define SET_VA_SBM_INC(vap,v)	SET_VARG_INC( VA_SRC5(vap), v )

//#define SET_VA_COUNT(vap,v)		SET_SZI_DST_DIMS(VA_SIZE_INFO(vap),v)
#define SET_VA_COUNT(vap,v)		SET_VARG_DIMSET(VA_DEST(vap),v)
#define SET_VA_SBM_BIT0(vap,v)		(vap)->va_sbm_bit0 = v
#define SET_VA_DBM_BIT0(vap,v)		(vap)->va_dbm_bit0 = v
#define SET_VA_SVAL(vap,idx,v)		(vap)->va_sval[idx] =  v
#define SET_VA_SVAL1(vap,v)		SET_VA_SVAL(vap,0,v)
#define SET_VA_SVAL2(vap,v)		SET_VA_SVAL(vap,1,v)
#define SET_VA_SVAL3(vap,v)		SET_VA_SVAL(vap,2,v)
#define SET_VA_CVAL1(vap,v)		SET_VA_SVAL(vap,0,v)
#define SET_VA_CVAL2(vap,v)		SET_VA_SVAL(vap,1,v)
#define SET_VA_QVAL1(vap,v)		SET_VA_SVAL(vap,0,v)
#define SET_VA_QVAL2(vap,v)		SET_VA_SVAL(vap,1,v)
//#define SET_VA_SCALAR_VAL_STD(vap,idx,v)	*((std_type *)(vap)->va_sval[idx]) = v
#define SET_VA_SCALAR_VAL_STD(vap,idx,v)	*((dest_type *)(vap)->va_sval[idx]) = v
#define SET_VA_SCALAR_VAL_UDI(vap,idx,v)	*((uint32_t *)(vap)->va_sval[idx]) = v
//#define VA_SCALAR_VAL_STD(vap,idx)	(*((std_type *)((vap)->va_sval[idx])))
#define VA_SCALAR_VAL_STD(vap,idx)	(*((dest_type *)((vap)->va_sval[idx])))
#define VA_SCALAR_VAL_UDI(vap,idx)	(*((uint32_t *)((vap)->va_sval[idx])))
//#define VA_SCALAR_VAL_STDCPX(vap,idx)	(*((std_cpx *)((vap)->va_sval[idx])))
//#define VA_SCALAR_VAL_STDQUAT(vap,idx)	(*((std_quat *)((vap)->va_sval[idx])))
#define VA_SCALAR_VAL_STDCPX(vap,idx)	(*((dest_cpx *)((vap)->va_sval[idx])))
#define VA_SCALAR_VAL_STDQUAT(vap,idx)	(*((dest_quat *)((vap)->va_sval[idx])))

#define SET_VA_DEST_OFFSET(vap,os)	SET_VARG_OFFSET( VA_DEST(vap), os );

#define SET_VA_SRC_OFFSET(vap,idx,os)	SET_VARG_OFFSET( VA_SRC(vap,idx), os )
#define SET_VA_SRC1_OFFSET(vap,os)	SET_VA_SRC_OFFSET(vap,0,os)
#define SET_VA_SRC2_OFFSET(vap,os)	SET_VA_SRC_OFFSET(vap,1,os)
#define SET_VA_SRC3_OFFSET(vap,os)	SET_VA_SRC_OFFSET(vap,2,os)
#define SET_VA_SRC4_OFFSET(vap,os)	SET_VA_SRC_OFFSET(vap,3,os)
#define SET_VA_SRC5_OFFSET(vap,os)	SET_VA_SRC_OFFSET(vap,4,os)
#define SET_VA_SBM_OFFSET(vap,os)	SET_VARG_OFFSET( VA_SBM(vap), os )
#define SET_VA_DBM_OFFSET(vap,os)	SET_VARG_OFFSET( VA_DBM(vap), os )

#define DECLARE_VA_SCALARS						\
									\
	Scalar_Value sv1, sv2, sv3;					\
	SET_VA_SVAL1(vap,&sv1);						\
	SET_VA_SVAL2(vap,&sv2);						\
	SET_VA_SVAL3(vap,&sv3);

typedef struct fft_args {
	void *		fft_src_addr;
	void *		fft_dst_addr;
	incr_t		fft_src_inc;
	incr_t		fft_dst_inc;
	dimension_t	fft_len;
	int		fft_isi;	// inverse flag
	struct platform_device *	fft_pdp;
#ifdef HAVE_OPENCL
	dimension_t	fft_src_offset;
	dimension_t	fft_dst_offset;
#endif // HAVE_OPENCL
} FFT_Args;


/* FFT args */

#define FFT_LEN(fap)			(fap)->fft_len
#define FFT_ISI(fap)			(fap)->fft_isi
#define FFT_SRC(fap)			(fap)->fft_src_addr
#define FFT_DST(fap)			(fap)->fft_dst_addr
#define FFT_DINC(fap)			(fap)->fft_dst_inc
#define FFT_SINC(fap)			(fap)->fft_src_inc
#define FFT_PFDEV(fap)			(fap)->fft_pdp

#define SET_FFT_LEN(fap,v)		(fap)->fft_len = v
#define SET_FFT_ISI(fap,v)		(fap)->fft_isi = v
#define SET_FFT_SRC(fap,v)		(fap)->fft_src_addr = v
#define SET_FFT_DST(fap,v)		(fap)->fft_dst_addr = v
#define SET_FFT_DINC(fap,v)		(fap)->fft_dst_inc = v
#define SET_FFT_SINC(fap,v)		(fap)->fft_src_inc = v
#define SET_FFT_PFDEV(fap,v)		(fap)->fft_pdp = v

#ifdef HAVE_OPENCL
#define FFT_SRC_OFFSET(fap)		(fap)->fft_src_offset
#define FFT_DST_OFFSET(fap)		(fap)->fft_dst_offset
#define SET_FFT_SRC_OFFSET(fap,v)	(fap)->fft_src_offset = v
#define SET_FFT_DST_OFFSET(fap,v)	(fap)->fft_dst_offset = v
#endif // HAVE_OPENCL

/* Obj_Args */

#define OA_ARGSTYPE(oap)		(oap)->oa_argstype
#define OA_ARGSPREC(oap)		(oap)->oa_argsprec
#define OA_FUNCTYPE(oap)		(oap)->oa_functype
#define OA_DEST(oap)			(oap)->oa_dest
#define OA_DBM(oap)			OA_DEST(oap)
#define INIT_OBJ_ARG_PTR(oap)		(oap)=((Vec_Obj_Args *)getbuf(sizeof(Vec_Obj_Args)));
#define RELEASE_OBJ_ARG_PTR(oap)	givbuf(oap);
#define OA_SRC_OBJ(oap,idx)		(oap)->oa_dp[idx]
#define OA_SRC1(oap)			OA_SRC_OBJ(oap,0)
#define OA_SRC2(oap)			OA_SRC_OBJ(oap,1)
#define OA_SRC3(oap)			OA_SRC_OBJ(oap,2)
#define OA_SRC4(oap)			OA_SRC_OBJ(oap,3)
//#define OA_BMAP(oap)			OA_SRC_OBJ(oap,4)
#define OA_SBM(oap)			OA_SRC_OBJ(oap,4)
#define OA_PFDEV(oap)			(oap)->oa_pdp
#define SET_OA_PFDEV(oap,v)		(oap)->oa_pdp = v
//#define OA_DISPATCH_FUNC(oap)		PFDEV_DISPATCH_FUNC( OA_PFDEV(oap) )
#define SET_OA_DEST(oap,dp)		(oap)->oa_dest=dp
#define SET_OA_DBM(oap,v)		SET_OA_DEST(oap,v)
#define SET_OA_SRC_OBJ(oap,idx,dp)	OA_SRC_OBJ(oap,idx) =  dp
#define SET_OA_SBM(oap,dp)		OA_SRC_OBJ(oap,4) = dp

/*#define OA_OBJ(oap,idx)			[oap getDataObjectAtIndex : idx] */
/*#define SET_OA_OBJ(oap,idx,dp)		[oap setDataObjectAtIndex : idx withValue : dp] */
#define SET_OA_SRC1(oap,dp)		OA_SRC_OBJ(oap,0) = dp
#define SET_OA_SRC2(oap,dp)		OA_SRC_OBJ(oap,1) = dp
#define SET_OA_SRC3(oap,dp)		OA_SRC_OBJ(oap,2) = dp
#define SET_OA_SRC4(oap,dp)		OA_SRC_OBJ(oap,3) = dp

#define OA_SCLR_OBJ(oap,idx)		(oap)->oa_sdp[idx]
#define OA_SCLR1(oap)			OA_SCLR_OBJ(oap,0)
#define OA_SCLR2(oap)			OA_SCLR_OBJ(oap,1)
#define SET_OA_SCLR_OBJ(oap,idx,dp)	OA_SCLR_OBJ(oap,idx) = dp
#define SET_OA_SCLR1(oap,dp)		OA_SCLR_OBJ(oap,0) = dp
#define SET_OA_SCLR2(oap,dp)		OA_SCLR_OBJ(oap,1) = dp

#define OA_SVAL(oap,idx)		(oap)->oa_svp[idx]
#define OA_SVAL1(oap)			(oap)->oa_svp[0]
#define OA_SVAL2(oap)			(oap)->oa_svp[1]
#define OA_CPX_SVAL(oap,idx)		(oap)->oa_svp[idx]
#define OA_QUAT_SVAL(oap,idx)		(oap)->oa_svp[idx]
#define SET_OA_SVAL1(oap,val)		(oap)->oa_svp[0]=val
#define SET_OA_ARGSPREC(oap,val)	(oap)->oa_argsprec=val
#define SET_OA_ARGSTYPE(oap,val)	(oap)->oa_argstype=val
#define SET_OA_FUNCTYPE(oap,val)	(oap)->oa_functype=val
#define SET_OA_FLAGS(oap,v)		(oap)->oa_flags = v
#define SET_OA_FLAG_BITS(oap,v)		(oap)->oa_flags |= v
#define SET_OA_SVAL(oap,idx,v)		(oap)->oa_svp[idx] = v

extern void clear_oargs(Vec_Obj_Args *oap);

extern Precision *src_prec_for_argset_prec(argset_prec ap,argset_type at);

#endif /* ! _OBJ_ARGS_H_ */
