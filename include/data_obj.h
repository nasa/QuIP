#ifndef _DATA_OBJ_H_
#define _DATA_OBJ_H_

#include "quip_config.h"
#ifdef FOOBAR
#include "quip_menu.h"
#endif // FOOBAR
#include "query_api.h"
#include "item_type.h"
#include "shape_info.h"
#include "freel.h"
#include "list.h"
#include "debug.h"
#include "dobj_basic.h"

struct platform_device;

typedef union {
	double		u_d;	// this is first to be the default initializer
	long double	u_ld;
	char		u_b;
	short		u_s;
	int32_t		u_l;
	int64_t		u_ll;
	unsigned char	u_ub;
	unsigned short	u_us;
	uint32_t	u_ul;
	uint64_t	u_ull;
	float		u_f;
	/* Do we need both of these??? */
	float		u_fc[2];
	SP_Complex	u_spc;
	float		u_fq[4];
	SP_Quaternion	u_spq;

	double		u_dc[2];
	DP_Complex	u_dpc;
	double		u_dq[4];
	DP_Quaternion	u_dpq;

	long double	u_lc[2];
	LP_Complex	u_lpc;
	long double	u_lq[4];
	LP_Quaternion	u_lpq;

	bitmap_word	u_bit;	/* should be boolean type... */
	void *		u_vp;	// for string type
} Scalar_Value;

#define NO_SCALAR_VALUE	((Scalar_Value *)NULL)

#define SVAL_FLOAT(svp)		(svp)->u_f
#define SVAL_STD(svp)		(svp)->std_scalar
#define SVAL_STD_CPX(svp)	(svp)->std_cpx_scalar
#define SVAL_STD_QUAT(svp)	(svp)->std_quat_scalar

/* Data areas were a largely obsolete construct, which were
 * originally introduced to allow objects to be placed in a particular
 * protion of physical ram, as needed by the sky warrior hardware.
 *
 * However, they have come back with the extension of the system
 * to CUDA, where the device has its own memory space.  Today,
 * however, we don't do much of the allocation ourselves, and so
 * the free list management has been broken out...
 */

typedef struct memory_area {
	void *			ma_base;
	FreeList		ma_freelist;
	uint32_t		ma_memsiz;
	uint32_t		ma_memfree;
} Memory_Area;

#define NO_MEMORY_AREA ((Memory_Area *)NULL)



typedef struct data_area {
	Item			da_item;
	Memory_Area *		da_ma_p;
	uint32_t		da_flags;
	Data_Obj *		da_dp;	// scratch scalar
	struct platform_device *	da_pdp;
	void *			da_extra_p;	// use this for both OpenCL and CUDA!
} Data_Area;

// could use a union...
#define da_cd_p		da_extra_p
#define da_ocldev_p	da_extra_p

#define AREA_PFDEV(ap)		(ap)->da_pdp
#define SET_AREA_PFDEV(ap,v)	(ap)->da_pdp = v
#define AREA_PLATFORM(ap)	PFDEV_PLATFORM( AREA_PFDEV(ap) )

#define SET_AREA_OCL_DEV(ap,odp)	(ap)->da_ocldev_p = odp

#define AREA_SCALAR_OBJ( ap )	(ap)->da_dp

#define da_name		da_item.item_name

#define da_base		da_ma_p->ma_base
#define da_freelist	da_ma_p->ma_freelist
#define da_memsiz	da_ma_p->ma_memsiz
#define da_memfree	da_ma_p->ma_memfree


/* Data area flag bits */
enum data_area_type {
	DA_RAM_BIT,
	DA_CUDA_GLOBAL_BIT,
	DA_CUDA_HOST_BIT,
	DA_CUDA_HOST_MAPPED_BIT,
	/* DA_CUDA_CONSTANT_BIT, */	// not currently used...
	DA_OCL_GLOBAL_BIT,
	DA_OCL_HOST_BIT,
	DA_OCL_HOST_MAPPED_BIT,
	N_DATA_AREA_TYPES
};

// What is the logic of these???
#define PF_GLOBAL_AREA_INDEX		0
#define PF_HOST_AREA_INDEX		1
#define PF_HOST_MAPPED_AREA_INDEX	2
//#define PF_CONSTANT_AREA_INDEX		3	// not used at present...

#define DA_RAM			(1<<DA_RAM_BIT)
#define DA_CUDA_GLOBAL		(1<<DA_CUDA_GLOBAL_BIT)
#define DA_CUDA_HOST		(1<<DA_CUDA_HOST_BIT)
#define DA_CUDA_HOST_MAPPED	(1<<DA_CUDA_HOST_MAPPED_BIT)

#define DA_OCL_GLOBAL		(1<<DA_OCL_GLOBAL_BIT)
#define DA_OCL_HOST		(1<<DA_OCL_HOST_BIT)
#define DA_OCL_HOST_MAPPED	(1<<DA_OCL_HOST_MAPPED_BIT)

#define DA_TYPE_MASK	((1<<N_DATA_AREA_TYPES)-1)

#define AREA_TYPE(dp)	(dp->dt_ap->da_flags & DA_TYPE_MASK)
#define OBJ_IS_RAM(dp)	(dp->dt_ap->da_flags&(DA_RAM|DA_CUDA_HOST))

#define MAX_RAM_CHUNKS	256	// size of free list


#define INSIST_OBJ_PREC(dp,code,whence)			\
	if( OBJ_PREC(dp) != code ){			\
		sprintf(ERROR_STRING,			\
"%s:  object %s (%s) should have %s precision.",	\
			#whence,OBJ_NAME(dp),		\
			PREC_NAME(OBJ_PREC_PTR(dp)),	\
			PREC_NAME(PREC_FOR_CODE(code)));\
		WARN(ERROR_STRING);			\
		return;					\
	}

#ifdef HAVE_CUDA

#define RAM_OBJ_ERROR_MSG(dp,whence)			\
		sprintf(ERROR_STRING,			\
"%s:  object %s must be a host ram object or cuda host object.", \
			#whence,OBJ_NAME(dp));		\
		WARN(ERROR_STRING);

#define INSIST_RAM_OBJ(dp,whence)			\
	if( ! OBJ_IS_RAM(dp) ){				\
		RAM_OBJ_ERROR_MSG(dp,whence)		\
		return;					\
	}

#define VINSIST_RAM_OBJ(dp,whence,retval)		\
	if( ! OBJ_IS_RAM(dp) ){				\
		RAM_OBJ_ERROR_MSG(dp,whence)		\
		return retval;				\
	}

#else // ! HAVE_CUDA
#define INSIST_RAM_OBJ(dp,whence)		// nop
#define VINSIST_RAM_OBJ(dp,whence,retval)	// nop
#endif // ! HAVE_CUDA
		

extern Data_Area *def_area_p, *ram_area_p;

extern debug_flag_t debug_data;

#define MAX_AREAS	4

#define NO_AREA	((Data_Area *) NULL )

// this may be pointed to by dt_unaligned_ptr...
struct gl_info;

struct data_obj {
	Item			dt_item;
	Shape_Info *		dt_shpp;
	void *			dt_data_ptr;
	void *			dt_unaligned_ptr;
	void *			dt_extra;
	int			dt_bit0;
	Data_Obj *		dt_parent;
	List *			dt_children;
	index_t			dt_offset;	// data offset of subobjects - in bytes
	int			dt_refcount;
	const char *		dt_declfile;
	Data_Area *		dt_ap;
	//int			dt_flags;	// are the flags the same as the shape flags?
};

#define NO_OBJ		((Data_Obj *)NULL)


#define OWNS_DATA(dp)		((OBJ_FLAGS(dp) & DT_NO_DATA)==0)


/* Shape macros */

#define IS_STRING( dp )		STRING_PRECISION( (OBJ_SHAPE(dp))->si_prec_p->prec_code )
#define IS_IMAGE( dp )		IMAGE_SHAPE( OBJ_SHAPE(dp) )
#define IS_BITMAP( dp )		BITMAP_SHAPE( OBJ_SHAPE(dp) )
#define IS_VECTOR( dp )		VECTOR_SHAPE( OBJ_SHAPE(dp) )
#define IS_COLVEC( dp )		COLVEC_SHAPE( OBJ_SHAPE(dp) )
#define IS_ROWVEC( dp )		ROWVEC_SHAPE( OBJ_SHAPE(dp) )
#define IS_SEQUENCE( dp )	SEQUENCE_SHAPE( OBJ_SHAPE(dp) )
#define IS_HYPER_SEQ( dp )	HYPER_SEQ_SHAPE( OBJ_SHAPE(dp) )
#define IS_SCALAR( dp )		SCALAR_SHAPE( OBJ_SHAPE(dp) )


#define IS_QUAT( dp )		QUAT_SHAPE( OBJ_SHAPE(dp) )
#define IS_COMPLEX( dp )	COMPLEX_SHAPE(  OBJ_SHAPE(dp) )
#define IS_CPX_OR_QUAT(dp)	( IS_COMPLEX(dp) || IS_QUAT(dp) )
#define IS_REAL( dp )		REAL_SHAPE( OBJ_SHAPE(dp) )
#define IS_MULTIDIM( dp )	MULTIDIM_SHAPE( OBJ_SHAPE(dp) )
/* #define IS_UNSIGNED( dp )	UNSIGNED_SHAPE( OBJ_SHAPE(dp) ) */
/* This definition depends on the unsigned precisions coming last! */
#define IS_UNSIGNED( dp )	( OBJ_MACH_PREC( dp ) >= PREC_UBY )

#define IS_STATIC( dp )		( OBJ_FLAGS(dp) & DT_STATIC )

#define IS_TEMP( dp )		( OBJ_FLAGS(dp) & DT_TEMP )

#define IS_ZOMBIE(dp)		( OBJ_FLAGS(dp) & DT_ZOMBIE )

#ifdef CAUTIOUS
#define IS_CONTIGUOUS(dp)	(  ( OBJ_FLAGS(dp) & DT_CONTIG ) || 		\
				( (!(OBJ_FLAGS(dp) & DT_CHECKED)) && is_contiguous(QSP_ARG  dp) ) )

#define N_IS_CONTIGUOUS(dp)	(  ( OBJ_FLAGS(dp) & DT_CONTIG ) || 		\
				( (!(OBJ_FLAGS(dp) & DT_CHECKED)) &&		\
				is_contiguous(DEFAULT_QSP_ARG  dp) ) )

#define IS_EVENLY_SPACED(dp)	( OBJ_FLAGS(dp) & DT_EVENLY )

#else /* ! CAUTIOUS */
#define IS_CONTIGUOUS(dp)	(  ( OBJ_FLAGS(dp) & DT_CONTIG ) )
#define IS_EVENLY_SPACED(dp)	( OBJ_FLAGS(dp) & DT_EVENLY )
#endif /* ! CAUTIOUS */

#define HAS_CONTIGUOUS_DATA(dp)	( IS_CONTIGUOUS(dp) || (OBJ_FLAGS(dp) & DT_CONTIG_BITMAP_DATA) )

#define IS_ALIGNED(dp)		( OBJ_FLAGS(dp) & DT_ALIGNED )
#define DOBJ_IS_LOCKED(dp)	( OBJ_FLAGS(dp) & DT_LOCKED )
#define HAS_ALL_VALUES(dp)	( OBJ_FLAGS(dp) & DT_ASSIGNED )
#define HAS_SOME_VALUES(dp)	( OBJ_FLAGS(dp) & (DT_ASSIGNED|DT_PARTIALLY_ASSIGNED) )

#define IS_GL_BUFFER(dp)	( OBJ_FLAGS(dp) & DT_GL_BUF )
#define BUF_IS_MAPPED(dp)	( OBJ_FLAGS(dp) & DT_BUF_MAPPED )

#define IS_EXPORTED(dp)		( OBJ_FLAGS(dp) & DT_EXPORTED )


#define DNAME_VALID	"_.-/"

/* codes for subscript strings */
#define SQUARE	1
#define CURLY	2

extern int max_vectorizable;

#include "dobj_prot.h"

/* Data_Obj macros */

#define NEW_DOBJ		((Data_Obj *)getbuf(sizeof(Data_Obj)))
#define INDEX_COUNT(dsp,i)		(dsp)->ds_dimension[i]
#define ASSIGN_IDX_COUNT(dsp,i,v)	(dsp)->ds_dimension[i]=v
#define IDX_INC(isp,i)			(isp)->is_increment[i]

#define OBJ_SHAPE(dp)		(dp)->dt_shpp
#define SET_OBJ_SHAPE(dp,shpp)	(dp)->dt_shpp = shpp
#define OBJ_EXTRA(dp)		(dp)->dt_extra
#define SET_ASSIGNED_FLAG(dp)	SET_OBJ_FLAG_BITS(dp,DT_ASSIGNED);
#define OBJ_DATA_PTR(dp)	(dp)->dt_data_ptr
#define OBJ_UNALIGNED_PTR(dp)	(dp)->dt_unaligned_ptr

#ifdef HAVE_OPENGL
#define OBJ_GL_INFO(dp)		((struct gl_info *)(dp)->dt_unaligned_ptr)
//#define SET_OBJ_GL_INFO(dp,p)	(dp)->dt_unaligned_ptr = p
#define SET_OBJ_GL_INFO(dp,p)	{ fprintf(stderr,"SET_OBJ_GL_INFO(%s,0x%lx)\n",OBJ_NAME(dp),(long)p); (dp)->dt_unaligned_ptr = p; }
#else // ! HAVE_OPENGL
#define OBJ_GL_INFO(dp)		((dp)->dt_unaligned_ptr)
#define SET_OBJ_GL_INFO(dp,p)
#endif // HAVE_OPENGL

#define OBJ_N_MACH_ELTS(dp)	SHP_N_MACH_ELTS(OBJ_SHAPE(dp))
#define OBJ_N_TYPE_ELTS(dp)	SHP_N_TYPE_ELTS(OBJ_SHAPE(dp))
#define OBJ_NAME(dp)		(dp)->dt_item.item_name
#define SET_OBJ_NAME(dp,s)	(dp)->dt_item.item_name = s
#define OBJ_DECLFILE(dp)	(dp)->dt_declfile
#define OBJ_BIT0(dp)		(dp)->dt_bit0
#define OBJ_PREC(dp)		SHP_PREC(OBJ_SHAPE(dp))
#define OBJ_PREC_PTR(dp)	SHP_PREC_PTR(OBJ_SHAPE(dp))
#define OBJ_PREC_NAME(dp)	PREC_NAME( OBJ_PREC_PTR(dp) )
#define OBJ_TYPE_DIMS(dp)	SHP_TYPE_DIMS(OBJ_SHAPE(dp))
#define OBJ_TYPE_INCS(dp)	SHP_TYPE_INCS(OBJ_SHAPE(dp))
#define OBJ_MACH_DIMS(dp)	SHP_MACH_DIMS(OBJ_SHAPE(dp))
#define OBJ_MACH_INCS(dp)	SHP_MACH_INCS(OBJ_SHAPE(dp))
#define OBJ_AREA(dp)		(dp)->dt_ap
#define OBJ_PFDEV(dp)		AREA_PFDEV( OBJ_AREA(dp) )
#define OBJ_PLATFORM(dp)	AREA_PLATFORM( OBJ_AREA(dp) )
#define OBJ_COMPS(dp)		SHP_COMPS(OBJ_SHAPE(dp))
#define OBJ_COLS(dp)		SHP_COLS(OBJ_SHAPE(dp))
#define OBJ_ROWS(dp)		SHP_ROWS(OBJ_SHAPE(dp))
#define OBJ_FRAMES(dp)		SHP_FRAMES(OBJ_SHAPE(dp))
#define OBJ_SEQS(dp)		SHP_SEQS(OBJ_SHAPE(dp))
#define OBJ_DIMENSION(dp,idx)	SHP_DIMENSION(OBJ_SHAPE(dp),idx)

/*
#define OBJ_RANGE_MAXDIM(dp)	(dp)->dt_range_maxdim
#define OBJ_RANGE_MINDIM(dp)	(dp)->dt_range_mindim
#define SET_OBJ_RANGE_MAXDIM(dp,v)	(dp)->dt_range_maxdim = v
#define SET_OBJ_RANGE_MINDIM(dp,v)	(dp)->dt_range_mindim = v
*/
#define OBJ_RANGE_MAXDIM(dp)	SHP_RANGE_MAXDIM(OBJ_SHAPE(dp))
#define OBJ_RANGE_MINDIM(dp)	SHP_RANGE_MINDIM(OBJ_SHAPE(dp))
#define SET_OBJ_RANGE_MAXDIM(dp,v)	SET_SHP_RANGE_MAXDIM(OBJ_SHAPE(dp),v)
#define SET_OBJ_RANGE_MINDIM(dp,v)	SET_SHP_RANGE_MINDIM(OBJ_SHAPE(dp),v)

#define SET_OBJ_COMPS(dp,v)	SET_SHP_COMPS(OBJ_SHAPE(dp),v)
#define SET_OBJ_COLS(dp,v)	SET_SHP_COLS(OBJ_SHAPE(dp),v)
#define SET_OBJ_ROWS(dp,v)	SET_SHP_ROWS(OBJ_SHAPE(dp),v)
#define SET_OBJ_FRAMES(dp,v)	SET_SHP_FRAMES(OBJ_SHAPE(dp),v)
#define SET_OBJ_SEQS(dp,v)	SET_SHP_SEQS(OBJ_SHAPE(dp),v)

#define OBJ_COMP_INC(dp)	SHP_COMP_INC(OBJ_SHAPE(dp))
#define OBJ_PXL_INC(dp)		SHP_PXL_INC(OBJ_SHAPE(dp))
#define OBJ_ROW_INC(dp)		SHP_ROW_INC(OBJ_SHAPE(dp))
#define OBJ_FRM_INC(dp)		SHP_FRM_INC(OBJ_SHAPE(dp))
#define OBJ_SEQ_INC(dp)		SHP_SEQ_INC(OBJ_SHAPE(dp))

#define SET_OBJ_COMP_INC(dp,v)	SET_SHP_COMP_INC(OBJ_SHAPE(dp),v)
#define SET_OBJ_PXL_INC(dp,v)	SET_SHP_PXL_INC(OBJ_SHAPE(dp),v)
#define SET_OBJ_ROW_INC(dp,v)	SET_SHP_ROW_INC(OBJ_SHAPE(dp),v)
#define SET_OBJ_FRM_INC(dp,v)	SET_SHP_FRM_INC(OBJ_SHAPE(dp),v)
#define SET_OBJ_SEQ_INC(dp,v)	SET_SHP_SEQ_INC(OBJ_SHAPE(dp),v)

#define SET_OBJ_SHAPE_FLAGS(dp)	set_shape_flags(OBJ_SHAPE(dp),dp,AUTO_SHAPE)

#define SET_SHP_COMP_INC(shpp,v)	SET_INCREMENT(shpp->si_type_incs,0,v)
#define SET_SHP_PXL_INC(shpp,v)		SET_INCREMENT(shpp->si_type_incs,1,v)
#define SET_SHP_ROW_INC(shpp,v)		SET_INCREMENT(shpp->si_type_incs,2,v)
#define SET_SHP_FRM_INC(shpp,v)		SET_INCREMENT(shpp->si_type_incs,3,v)
#define SET_SHP_SEQ_INC(shpp,v)		SET_INCREMENT(shpp->si_type_incs,4,v)

#define SHP_COMP_INC(shpp)	INCREMENT(shpp->si_type_incs,0)
#define SHP_PXL_INC(shpp)	INCREMENT(shpp->si_type_incs,1)
#define SHP_ROW_INC(shpp)	INCREMENT(shpp->si_type_incs,2)
#define SHP_FRM_INC(shpp)	INCREMENT(shpp->si_type_incs,3)
#define SHP_SEQ_INC(shpp)	INCREMENT(shpp->si_type_incs,4)

#define OBJ_TYPE_DIM(dp,idx)	SHP_TYPE_DIM(OBJ_SHAPE(dp),idx)
#define OBJ_MACH_DIM(dp,idx)	SHP_MACH_DIM(OBJ_SHAPE(dp),idx)
#define OBJ_CHILDREN(dp)	(dp)->dt_children
#define OBJ_PARENT(dp)		((dp)->dt_parent)
#define OBJ_REFCOUNT(dp)	(dp)->dt_refcount
#define OBJ_FLAGS(dp)		SHP_FLAGS(OBJ_SHAPE(dp))
#define OBJ_MAXDIM(dp)		SHP_MAXDIM(OBJ_SHAPE(dp))
#define OBJ_MINDIM(dp)		SHP_MINDIM(OBJ_SHAPE(dp))
#define OBJ_MACH_INC(dp,idx)	SHP_MACH_INC(OBJ_SHAPE(dp),idx)
#define OBJ_TYPE_INC(dp,idx)	SHP_TYPE_INC(OBJ_SHAPE(dp),idx)
#define OBJ_OFFSET(dp)		(dp)->dt_offset

#define OBJ_PREC_MACH_SIZE(dp)	SHP_PREC_MACH_SIZE(OBJ_SHAPE(dp))
#define OBJ_MACH_PREC_NAME(dp)	PREC_NAME(OBJ_MACH_PREC_PTR(dp))
#define OBJ_MACH_PREC_SIZE(dp)	PREC_SIZE(OBJ_MACH_PREC_PTR(dp))
#define OBJ_MACH_PREC_PTR(dp)	PREC_MACH_PREC_PTR(OBJ_PREC_PTR(dp))

/* This is not so good now that the shape info is pointed to... */
#define OBJ_COPY_FROM(dpto,dpfr)	*dpto = *dpfr

/* Use this to give the copy its own shape */
#define DUP_OBJ_SHAPE(dpto,dpfr)			\
							\
	{						\
	SET_OBJ_SHAPE(dpto, ALLOC_SHAPE );		\
	COPY_SHAPE( OBJ_SHAPE(dpto), OBJ_SHAPE(dpfr) );	\
	}

#define SET_OBJ_EXTRA(dp,p)		(dp)->dt_extra = p
#define SET_OBJ_FLAGS(dp,f)		SET_SHP_FLAGS(OBJ_SHAPE(dp),f)
#define SET_OBJ_PARENT(dp,parent)	(dp)->dt_parent = parent
#define SET_OBJ_CHILDREN(dp,lp)		(dp)->dt_children = lp
#define SET_OBJ_AREA(dp,ap)		(dp)->dt_ap = ap 
#define SET_OBJ_PREC_PTR(dp,p)		SET_SHP_PREC_PTR(OBJ_SHAPE(dp),p)
#define SET_OBJ_OFFSET(dp,v)		(dp)->dt_offset = v
#define SET_OBJ_DATA_PTR(dp,ptr)	(dp)->dt_data_ptr = ptr
#define SET_OBJ_UNALIGNED_PTR(dp,ptr)	(dp)->dt_unaligned_ptr = ptr
#define SET_OBJ_N_MACH_ELTS(dp,n)	SET_SHP_N_MACH_ELTS(OBJ_SHAPE(dp),n)
#define SET_OBJ_N_TYPE_ELTS(dp,n)	SET_SHP_N_TYPE_ELTS(OBJ_SHAPE(dp),n)
#define SET_OBJ_BIT0(dp,v)		(dp)->dt_bit0 = v
/* BUG?  we we save here, or when calling??? */
#define SET_OBJ_DECLFILE(dp,str)	(dp)->dt_declfile = str
#define SET_OBJ_MACH_DIM(dp,idx,v)	SET_SHP_MACH_DIM(OBJ_SHAPE(dp),idx,v)
#define SET_OBJ_TYPE_DIM(dp,idx,v)	SET_SHP_TYPE_DIM(OBJ_SHAPE(dp),idx,v)
#define SET_OBJ_MACH_INC(dp,idx,v)	SET_SHP_MACH_INC(OBJ_SHAPE(dp),idx,v)
#define SET_OBJ_TYPE_INC(dp,idx,v)	SET_SHP_TYPE_INC(OBJ_SHAPE(dp),idx,v)
#define SET_OBJ_REFCOUNT(dp,c)		(dp)->dt_refcount = c
#define SET_OBJ_FLAG_BITS(dp,f)		SET_SHP_FLAG_BITS(OBJ_SHAPE(dp),f)
#define SET_OBJ_MINDIM(dp,v)		SET_SHP_MINDIM(OBJ_SHAPE(dp),v)
#define SET_OBJ_MAXDIM(dp,v)		SET_SHP_MAXDIM(OBJ_SHAPE(dp),v)
//#define SET_OBJ_LAST_SUBI(dp,v)		SET_SHP_LAST_SUBI(OBJ_SHAPE(dp),v)
#define CLEAR_OBJ_FLAG_BITS(dp,f)	CLEAR_SHP_FLAG_BITS(OBJ_SHAPE(dp),f)
#define CLEAR_OBJ_PREC_BITS(dp,f)	CLEAR_SHP_PREC_BITS(OBJ_SHAPE(dp),f)

/* Data_Area */
#define AREA_NAME(ap)			(ap)->da_item.item_name
#define AREA_FLAGS(ap)			(ap)->da_flags
#define AREA_CUDA_DEV(ap)		(ap)->da_cd_p
#define SET_AREA_CUDA_DEV(ap,cdp)	(ap)->da_cd_p = cdp
#define AREA_OCL_DEV(ap)		(ap)->da_ocldev_p
#define SET_AREA_OCL_DEV(ap,odp)	(ap)->da_ocldev_p = odp

#define CHECK_CONTIG_DATA(whence,which_obj,dp)				\
									\
	if( ! HAS_CONTIGUOUS_DATA(dp) ){			\
		sprintf(ERROR_STRING,					\
	"%s:  %s object %s must have contiguous data.",			\
			whence,which_obj,OBJ_NAME(dp));			\
		WARN(ERROR_STRING);					\
		return;							\
	}


#define CHECK_NOT_RAM(whence,which_obj,dp)				\
									\
	if( OBJ_IS_RAM(dp) ){						\
		sprintf(ERROR_STRING,					\
	"%s:  %s object %s lives in %s data area, expected a GPU.",	\
			whence,which_obj,OBJ_NAME(dpto),		\
			AREA_NAME(OBJ_AREA(dpto)));			\
		WARN(ERROR_STRING);					\
		return;							\
	}

#define CHECK_RAM(whence,which_obj,dp)					\
									\
	if( ! OBJ_IS_RAM(dp) ){						\
		sprintf(ERROR_STRING,					\
	"%s:  %s object %s lives in %s data area, expected ram.",	\
			whence,which_obj,OBJ_NAME(dpto),		\
			AREA_NAME(OBJ_AREA(dpto)));			\
		WARN(ERROR_STRING);					\
		return;							\
	}

/* BUG should go elsewhere */
extern const char *dimension_name[];
extern void dobj_init(SINGLE_QSP_ARG_DECL);

extern void	init_dobj_menu(void);
extern int set_obj_shape_flags(Data_Obj *dp);



extern Data_Area *curr_ap;
void describe_shape(QSP_ARG_DECL  Shape_Info *shpp);
#define DESCRIBE_SHAPE(shpp)	describe_shape(QSP_ARG  shpp)

extern Item_Context *create_dobj_context(QSP_ARG_DECL  const char *);

extern Data_Obj *pick_dobj(QSP_ARG_DECL  const char *pmpt);
extern Data_Area *pick_data_area(QSP_ARG_DECL  const char *pmpt);
extern void push_dobj_context(QSP_ARG_DECL  Item_Context *icp);
extern Item_Context * pop_dobj_context(SINGLE_QSP_ARG_DECL);
extern Item_Context * current_dobj_context(SINGLE_QSP_ARG_DECL);
extern void init_asc_menu(void);
extern void init_ops_menu(void);
//extern int siztbl[];
extern Item_Type *dobj_itp;
extern Item_Type *prec_itp;
extern void xfer_dobj_flag(Data_Obj *dpto, Data_Obj *dpfr, uint32_t flagbit);

ITEM_INIT_PROT(Data_Obj,dobj)
ITEM_LIST_PROT(Data_Obj,dobj)
ITEM_CHECK_PROT(Data_Obj,dobj)
ITEM_NEW_PROT(Data_Obj,dobj)
ITEM_DEL_PROT(Data_Obj,dobj)

// areas.c

ITEM_INTERFACE_PROTOTYPES(Data_Area,data_area)

// sub_obj.c
extern void default_offset_data_func(QSP_ARG_DECL  Data_Obj *dp, index_t pix_offset );

// dobj_util.c

extern void propagate_flag(Data_Obj *dp,uint32_t flagbit);

// dobj_expr.c
void init_dobj_expr_funcs(SINGLE_QSP_ARG_DECL);

// Something in here breaks some old cuda code...
#include "platform.h"

#endif /* ! _DATA_OBJ_H_ */

