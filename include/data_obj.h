
// this is a test change

#ifndef _DATA_OBJ_H_
#define _DATA_OBJ_H_

#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>
#include "typedefs.h"
#include "items.h"
#include "query.h"

#ifdef HAVE_CUDA
#include <GL/gl.h>		// needed for BUF_ID
#endif

/* data types */

/* prec codes */

/* 4 bits for machine precision */
/* 4 bits for psuedo-precision  */

#define BAD_PREC	(-1)

typedef enum {
	/* The machine precisions have to come first */
	PREC_NONE,		/* 0 not really a machine prec */
	PREC_BY,		/* 1 char */
	PREC_IN,		/* 2 short */
	PREC_DI,		/* 3 long (int32) */
	PREC_LI,		/* 4 longlong (int64) */
	PREC_SP,		/* 5 float */
	PREC_DP,		/* 6 double */
	/* add the unsigned types */
	PREC_UBY,		/* 7 u_char */
	PREC_UIN,		/* 8 u_short */
	PREC_UDI,		/* 9 uint32 */
	PREC_ULI,		/* 10 uint64 */
	/* PREC_FR, PREC_DF, */	/* unsupported sky warrior types */
	N_MACHINE_PRECS		/* 11 */
} mach_prec;

#define N_MACHPREC_BITS		4

#define FIRST_MACH_PREC	((mach_prec)0)		/* make sure this is the first line of the enum! */

#define N_REAL_MACHINE_PRECS	(N_MACHINE_PRECS-1)	/* don't count PREC_NONE */

#define PREC_MM	N_MACHINE_PRECS

#define N_PREC_STRINGS	(N_MACHINE_PRECS+1)


// BUG? we'd like to have the precision be the machine's native long...
// This is int32 or int64, which have different PREC codes...


#if __WORDSIZE == 64
#define BITMAP_DATA_TYPE		uint64_t
#define BITMAP_MACH_PREC		PREC_ULI
#define bitmap_scalar			u_ull
#define BITMAP_ALL_ONES			0xffffffffffffffff	/* not currently used? */
#define LOG2_BITS_PER_BITMAP_WORD	6
#define BITMAP_WORD_MSB			0x8000000000000000	/* This is ugly, but
								 * gcc does not allow
								 * l = 1 << 63
								 * even when l has 64 bits.
								 * a bug in gcc?
								 */

#else	/* __WORDSIZE == 32 */

#define BITMAP_DATA_TYPE		uint32_t
#define BITMAP_MACH_PREC		PREC_UDI
#define bitmap_scalar			u_ul
#define BITMAP_ALL_ONES			0xffffffff		/* not currently used? */
#define LOG2_BITS_PER_BITMAP_WORD	5
#define BITMAP_WORD_MSB			(1<<(BITS_PER_BITMAP_WORD-1))

#endif

typedef BITMAP_DATA_TYPE bitmap_word;
#define BITS_PER_BYTE			8
#define BYTES_PER_BITMAP_WORD		(sizeof(BITMAP_DATA_TYPE))
#define BITS_PER_BITMAP_WORD		(BYTES_PER_BITMAP_WORD*BITS_PER_BYTE)
#define BIT_NUMBER_MASK			(BITS_PER_BITMAP_WORD-1)

/* Should we allow multi-component bitmaps?
 * To facilitate processing bitmaps in CUDA, we don't want bitmap
 * words to cross dimension boundaries.  If we have 3-component
 * pixels then we might make an exception...  But is it worth
 * the trouble?  But we might want to allow multi-component
 * pixels for arrays of data...  So we insist that mindim has
 * to be made up of an integral number of bitmap words.
 */

/* This macro just divides by bits per word and rounds up to the nearest integer */
#define N_BITMAP_WORDS(n)	(((n)+BITS_PER_BITMAP_WORD-1)/BITS_PER_BITMAP_WORD)

#define BITMAP_WORD_COUNT(dp)	(N_BITMAP_WORDS(dp->dt_type_dim[dp->dt_mindim]+dp->dt_bit0)*(dp->dt_n_type_elts/dp->dt_type_dim[dp->dt_mindim]))

#define NUMBERED_BIT(n)	( 1L << (n) )



typedef struct sp_complex {
	float	re;
	float	im;
} SP_Complex;

typedef struct dp_complex {
	double	re;
	double	im;
} DP_Complex;

typedef struct sp_quaternion {
	float	re;
	float	_i;
	float	_j;
	float	_k;
} SP_Quaternion;

typedef struct dp_quaternion {
	double	re;
	double	_i;
	double	_j;
	double	_k;
} DP_Quaternion;

typedef union {
	char		u_b;
	short		u_s;
	int32_t		u_l;
	int64_t		u_ll;
	u_char		u_ub;
	u_short		u_us;
	uint32_t	u_ul;
	uint64_t	u_ull;
	float		u_f;
	float		u_fc[2];
	float		u_fq[4];	/* quaternion */
	double		u_d;
	double		u_dc[2];
	float		u_dq[4];	/* quaternion */
	SP_Complex	u_spc;
	DP_Complex	u_dpc;
	SP_Quaternion	u_spq;
	DP_Quaternion	u_dpq;
	bitmap_word	u_bit;	/* should be boolean type... */
} Scalar_Value;

#define NO_SCALAR_VALUE ((Scalar_Value *)NULL)


/*
 * The following are not real precisions,
 * but are sometimes used in the same context.
 *
 * These would never be encountered in a data object.
 */

typedef enum {
	PP_NORM,	/* don't want to use 0 as an actual pseudo prec, this means the machine precision is the type */
	PP_COLOR,		/*  01 */
	PP_MIXED,		/*  02 */
	PP_VOID,		/*  03 */
	PP_CPX,			/*  04 */
	PP_STRING,		/*  05 */
	PP_BIT,			/*  06 */
	PP_ANY,			/*  07 */
	PP_QUAT,		/* 010 */
	PP_LIST,		/* 011 */
	PP_CHAR,		/* 012 */
	PP_DBLCPX,		/* 013 */	/* not used! but we need this to have a name... */
	PP_DBLQUAT,		/* 014 */	/* not used! but we need this to have a name... */
	N_PSEUDO_PRECS	/* must be last */
} pseudo_prec;

#define N_PSEUDOPREC_BITS	4

#define FIRST_PSEUDO_PREC	((pseudo_prec)0)

typedef int prec_t;

#define PSEUDO_PREC_BITS( pseudo_prec )			( (pseudo_prec) << N_MACHPREC_BITS )
#define PRECISION_CODE(mach_prec,pseudo_prec)		( (mach_prec) | PSEUDO_PREC_BITS(pseudo_prec) )

#define PREC_COLOR	PRECISION_CODE(	PREC_SP,   PP_COLOR )
#define PREC_VOID	PRECISION_CODE(	PREC_SP,   PP_VOID )		/* why PREC_SP??? */
#define PREC_STR	PRECISION_CODE(	PREC_BY,   PP_STRING )
#define PREC_CHAR	PRECISION_CODE(	PREC_BY,   PP_CHAR )
#define PREC_ANY	PRECISION_CODE(	PREC_NONE, PP_ANY )


/* complex flag is OR'd in */

#define LIST_PREC_BITS		PSEUDO_PREC_BITS(PP_LIST)
#define QUAT_PREC_BITS		PSEUDO_PREC_BITS(PP_QUAT)
#define COMPLEX_PREC_BITS	PSEUDO_PREC_BITS(PP_CPX)
#define BITMAP_PREC_BITS	PSEUDO_PREC_BITS(PP_BIT)
#define COLOR_PREC_BITS		PSEUDO_PREC_BITS(PP_COLOR)
#define STRING_PREC_BITS	PSEUDO_PREC_BITS(PP_STRING)
#define CHAR_PREC_BITS		PSEUDO_PREC_BITS(PP_CHAR)
#define QC_PREC_BITS		(QUAT_PREC_BITS|COMPLEX_PREC_BITS)

#define PSEUDO_PREC_MASK	PSEUDO_PREC_BITS((1<<N_PSEUDOPREC_BITS)-1)

#define PSEUDO_PREC_INDEX(prec)	(((prec) & PSEUDO_PREC_MASK)>>N_MACHPREC_BITS)


/* not all of these really have names, but this guarantees that the tbls are big enough */
#define N_NAMED_PRECS		(N_MACHINE_PRECS+N_PSEUDO_PRECS)
#define PREC_INDEX(prec)	(((prec)&PSEUDO_PREC_MASK)==0?prec:PSEUDO_PREC_INDEX(prec))

#define PREC_LIST		(/* does a list obj have a machine prec? */ QUAT_PREC_BITS)
#define PREC_QUAT		(PREC_SP|QUAT_PREC_BITS)
#define PREC_DBLQUAT		(PREC_DP|QUAT_PREC_BITS)
#define PREC_CPX		(PREC_SP|COMPLEX_PREC_BITS)
#define PREC_DBLCPX		(PREC_DP|COMPLEX_PREC_BITS)
#define PREC_BIT		(BITMAP_MACH_PREC|BITMAP_PREC_BITS)

#define MACH_PREC_MASK		((1<<N_MACHPREC_BITS)-1)
#define MACHINE_PREC(dp)	((mach_prec)((dp)->dt_prec & MACH_PREC_MASK))
#define MACH_PREC_NAME(dp)	prec_name[ MACHINE_PREC(dp) ]
#define PNAME(dp)		MACH_PREC_NAME(dp)

#define ELEMENT_SIZE(dp)	(siztbl[ MACHINE_PREC(dp) ])
#define ELEMENT_INC_SIZE(dp)	(IS_BITMAP(dp) ? 1 : siztbl[ MACHINE_PREC(dp) ])

#define QUAT_PRECISION(prec)	(((prec)&PSEUDO_PREC_MASK)==QUAT_PREC_BITS)
#define COMPLEX_PRECISION(prec)	(((prec)&PSEUDO_PREC_MASK)==COMPLEX_PREC_BITS)
#define NORMAL_PRECISION(prec)	((((prec)&PSEUDO_PREC_MASK)&QC_PREC_BITS)==0)
#define BITMAP_PRECISION(prec)	(((prec)&PSEUDO_PREC_MASK)==BITMAP_PREC_BITS)
#define COLOR_PRECISION(prec)	(((prec)&PSEUDO_PREC_MASK)==COLOR_PREC_BITS)
#define STRING_PRECISION(prec)	(((prec)&PSEUDO_PREC_MASK)==STRING_PREC_BITS)
#define CHAR_PRECISION(prec)	(((prec)&PSEUDO_PREC_MASK)==CHAR_PREC_BITS)

#define PRECISIONS_DIFFER(dp1,dp2)	( (dp1)->dt_prec != (dp2)->dt_prec )

#define PREC_OF( dp )		( ( dp )->dt_prec )


#define FLOATING_OBJ( dp )	(((dp->dt_prec&MACH_PREC_MASK)==PREC_SP)||((dp->dt_prec&MACH_PREC_MASK)==PREC_DP))
#define FLOATING_PREC( prec )	((((prec)&MACH_PREC_MASK)==PREC_SP)||(((prec)&MACH_PREC_MASK)==PREC_DP))

#define INTEGER_PREC( prec )	(   (((prec)&MACH_PREC_MASK)==PREC_BY)  \
				 || (((prec)&MACH_PREC_MASK)==PREC_IN)  \
				 || (((prec)&MACH_PREC_MASK)==PREC_DI)  \
				 || (((prec)&MACH_PREC_MASK)==PREC_UBY) \
				 || (((prec)&MACH_PREC_MASK)==PREC_UIN) \
				 || (((prec)&MACH_PREC_MASK)==PREC_UDI) \
				)


/*
 * A 5th dimension was added to allow conversion between interleaved color
 * representations (tdim=ncolors) and HIPS2 style frame sequential color.
 * In this latter representation, nframes becomes the number of color frames,
 * and n_seqs is the number of color frames.
 * jbm 9-11-94
 */

#define N_DIMENSIONS	5	/* color, x, y, t, hyper_t */

#include "freel.h"
#include "node.h"

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


typedef union {
#ifdef HAVE_CUDA
	/* This will hold a cuda device ptr,
	 * but we don't have the defn for that
	 * struct here at the moment...
	 */
	void *		dai_cd_p;
#define da_cd_p		da_dai.dai_cd_p;
#endif /* HAVE_CUDA */
} Data_Area_Info;

typedef struct data_area {
	Item			da_item;
	Memory_Area *		da_ma_p;
	Data_Area_Info		da_dai;
	uint32_t		da_flags;
} Data_Area;

#define da_name		da_item.item_name

#define da_base		da_ma_p->ma_base
#define da_freelist	da_ma_p->ma_freelist
#define da_memsiz	da_ma_p->ma_memsiz
#define da_memfree	da_ma_p->ma_memfree


/* Data area flag bits */

#define DA_RAM_BIT		0
#define DA_CUDA_GLOBAL_BIT	1
#define DA_CUDA_HOST_BIT	2
#define DA_CUDA_HOST_MAPPED_BIT	3
#ifdef FOOBAR
#define DA_CUDA_CONSTANT_BIT	4	// not currently used...
#endif /* FOOBAR */

#define CUDA_GLOBAL_AREA_INDEX		0
#define CUDA_HOST_AREA_INDEX		(DA_CUDA_HOST_BIT-DA_CUDA_GLOBAL_BIT)
#define CUDA_HOST_MAPPED_AREA_INDEX	(DA_CUDA_HOST_MAPPED_BIT-DA_CUDA_GLOBAL_BIT)

#define DA_RAM			(1<<DA_RAM_BIT)
#define DA_CUDA_GLOBAL		(1<<DA_CUDA_GLOBAL_BIT)
#define DA_CUDA_HOST		(1<<DA_CUDA_HOST_BIT)
#define DA_CUDA_HOST_MAPPED	(1<<DA_CUDA_HOST_MAPPED_BIT)

#ifdef FOOBAR
#define CUDA_CONSTANT_AREA_INDEX	(DA_CUDA_CONSTANT_BIT-DA_CUDA_GLOBAL_BIT)
#define DA_CUDA_CONSTANT	(1<<DA_CUDA_CONSTANT_BIT)
#endif /* FOOBAR */

#ifdef DA_CUDA_CONSTANT
#define DA_TYPE_MASK	(DA_RAM|DA_CUDA_GLOBAL|DA_CUDA_CONSTANT|DA_CUDA_HOST|DA_CUDA_HOST_MAPPED)
#else
#define DA_TYPE_MASK	(DA_RAM|DA_CUDA_GLOBAL|DA_CUDA_HOST|DA_CUDA_HOST_MAPPED)
#endif

#define AREA_TYPE(dp)	(dp->dt_ap->da_flags & DA_TYPE_MASK)
#define IS_RAM(dp)	(dp->dt_ap->da_flags&(DA_RAM|DA_CUDA_HOST))

extern Data_Area *def_area, *ram_area;
extern debug_flag_t debug_data;

#define MAX_AREAS	4

#define NO_AREA	((Data_Area *) NULL )
/* #define NO_DATA	((unsigned char *) NULL ) */

extern Item_Type *dobj_itp;	/* for non-context objects */

typedef uint32_t dimension_t;
typedef uint32_t index_t;
typedef int32_t incr_t;

typedef struct dimension_set {
	dimension_t	ds_dimension[N_DIMENSIONS];
} Dimension_Set;

/*#define ds_tdim		ds_dimension[0]*/
#define ds_comps	ds_dimension[0]
#define ds_cols		ds_dimension[1]
#define ds_rows		ds_dimension[2]
#define ds_frames	ds_dimension[3]
#define ds_seqs		ds_dimension[4]

#define STRING_DIMENSION(ds,len)					\
									\
	(ds).ds_comps=1;						\
	(ds).ds_cols=len;						\
	(ds).ds_rows=1;							\
	(ds).ds_frames=1;						\
	(ds).ds_seqs=1;

/* We have a problem with complex objects - in order to be able to
 * treat the real and complex parts as components, the dimensions
 * and increments reflect the machine precision of the elements.
 * But in vector operations we use a complex pointer, so they
 * are too much...  We could have the pointer be to the basic machine
 * type instead???
 */

typedef struct shape_info {
	prec_t		si_prec;

	/* We reluctantly introduce two sets of dimensions to handle complex cleanly. */

	Dimension_Set	si_mach_dimset;
	uint32_t	si_n_mach_elts;	/* total number of elements */

	Dimension_Set	si_type_dimset;
	uint32_t	si_n_type_elts;	/* total number of type elements */
					/* (product of all dims) */
	short		si_maxdim;	/* largest index of a dimension > 1 */
	short		si_mindim;	/* smallest index of a dimension > 1 */
	uint32_t	si_flags;
	int		si_last_subi;	/* used in evaltree.c */
					/* this allows us to do something with multiply-subscripted objects;
					 * but shouldn't we also need last_subj for curly-brace subscripts?
					 */
} Shape_Info;

#define NO_SHAPE		((Shape_Info *)NULL)

#define AUTO_SHAPE	0

// si_dimension is now the machine type dimension - deprecated
/* #define si_dimension	si_mach_dimset.ds_dimension */
#define si_mach_dim	si_mach_dimset.ds_dimension
#define si_type_dim	si_type_dimset.ds_dimension

#define si_comps	si_type_dimset.ds_comps
#define si_cols		si_type_dimset.ds_cols
#define si_rows		si_type_dimset.ds_rows
#define si_frames	si_type_dimset.ds_frames
#define si_seqs		si_type_dimset.ds_seqs

/* Shape macros */

#define VOID_SHAPE( shpp )		( ( shpp )->si_flags & DT_VOID )
#define IMAGE_SHAPE( shpp )		( ( shpp )->si_flags & DT_IMAGE )
#define BITMAP_SHAPE( shpp )		( ( shpp )->si_flags & DT_BIT )
#define VECTOR_SHAPE( shpp )		( ( shpp )->si_flags & DT_VECTOR )
#define COLVEC_SHAPE( shpp )		( ( shpp )->si_flags & DT_COLVEC )
#define ROWVEC_SHAPE( shpp )		( ( shpp )->si_flags & DT_ROWVEC )
#define SEQUENCE_SHAPE( shpp )		( ( shpp )->si_flags & DT_SEQUENCE )
#define HYPER_SEQ_SHAPE( shpp )		( ( shpp )->si_flags & DT_HYPER_SEQ )
/* BUG?  is this correct for complex/quaternion??? */
#define SCALAR_SHAPE( shpp )		( ( ( shpp )->si_flags & DT_SCALAR ) && ( (shpp)->si_type_dim[0] == 1 ) )
#define PIXEL_SHAPE( shpp )		( ( ( shpp )->si_flags & DT_SCALAR ) )
#define UNKNOWN_SHAPE( shpp )		( ( shpp )->si_flags & DT_UNKNOWN_SHAPE )
#define INTERLACED_SHAPE( shpp )	( ( shpp )->si_flags & DT_INTERLACED )

#define COMPLEX_SHAPE( shpp )		( COMPLEX_PRECISION( ( shpp )->si_prec ) )
#define QUAT_SHAPE( shpp )		( QUAT_PRECISION( ( shpp )->si_prec ) )
#define REAL_SHAPE( shpp )		( (( shpp )->si_flags & (SHAPE_TYPE_MASK&~DT_MULTIDIM)) == 0 )
#define MULTIDIM_SHAPE( shpp )		( ( shpp )->si_flags & DT_MULTIDIM )

#define SHAPE_DIM_MASK	(DT_UNKNOWN_SHAPE|DT_HYPER_SEQ|DT_SEQUENCE|DT_IMAGE|DT_VECTOR|DT_SCALAR)
/* don't care about unsigned... */
#define SHAPE_TYPE_MASK	(DT_COMPLEX|DT_MULTIDIM|DT_QUAT)
#define SHAPE_MASK	(SHAPE_DIM_MASK|SHAPE_TYPE_MASK)



struct data_obj {
	Item		 dt_item;

	Shape_Info	 dt_shape;

#define dt_type_dimset	dt_shape.si_type_dimset
#define dt_mach_dimset	dt_shape.si_mach_dimset
#define dt_type_dim	dt_type_dimset.ds_dimension
#define dt_mach_dim	dt_mach_dimset.ds_dimension
#define dt_prec		dt_shape.si_prec
#define dt_maxdim	dt_shape.si_maxdim
#define dt_mindim	dt_shape.si_mindim
#define dt_flags	dt_shape.si_flags
#define dt_last_subi	dt_shape.si_last_subi
#define dt_n_mach_elts	dt_shape.si_n_mach_elts
#define dt_n_type_elts	dt_shape.si_n_type_elts

	void *		dt_extra;	/* points to decl node in vectree module */

	void *		dt_data;	/* this should be union!?, */
					/* for now we'll cast  */
	void *		dt_unaligned_data;	/* free this one */
	int		dt_bit0;	/* for subimages of bitmaps */

	Data_Area *	dt_ap;		/* ptr to area where this lives */
	struct data_obj *dt_parent;	/* for subimages only */
	List *		dt_children;	/* for recursive deletion */
	incr_t		dt_type_inc[N_DIMENSIONS];
	incr_t		dt_mach_inc[N_DIMENSIONS];
	index_t	 	dt_offset;	/* data offset of child from parent */
					/* needed to handle relocation of
					 * grandchildren - see sub_obj.c */

	int		dt_refcount;	/* number of viewers depending on it */
	const char *	 dt_declfile;	/* file or macro name where declared */
};

#ifdef HAVE_CUDA
typedef struct gl_info {
	GLuint		buf_id;
	GLuint		tex_id;
} GL_Info;
#endif /* HAVE_CUDA */

#define dt_gl_info_p	dt_unaligned_data		/* overloading this field - used in cuda/cuda_viewer.cpp */
#define BUF_ID(dp)	( (((GL_Info *)(dp->dt_gl_info_p))->buf_id))
#define BUF_ID_P(dp)	( & (((GL_Info *)(dp->dt_gl_info_p))->buf_id))
#define TEX_ID(dp)	( (((GL_Info *)(dp->dt_gl_info_p))->tex_id))
#define TEX_ID_P(dp)	( & (((GL_Info *)(dp->dt_gl_info_p))->tex_id))

#define dt_name		dt_item.item_name

#define dt_comps	dt_shape.si_comps
#define dt_cols		dt_shape.si_cols
#define dt_rows		dt_shape.si_rows
#define dt_frames	dt_shape.si_frames
#define dt_seqs		dt_shape.si_seqs

#define dt_cinc		dt_type_inc[0]		/* to get to next component */
#define dt_pinc		dt_type_inc[1]		/* to get to next column */
#define dt_rowinc	dt_type_inc[2]		/* to get to next row */
#define dt_rinc		dt_type_inc[2]		/* to get to next row */
#define dt_finc		dt_type_inc[3]		/* to get to next frame */
#define dt_sinc		dt_type_inc[4]		/* to get to next sequence */

#define dt_mach_cinc	dt_mach_inc[0]		/* to get to next component */
#define dt_mach_pinc	dt_mach_inc[1]		/* to get to next column */
#define dt_mach_rowinc	dt_mach_inc[2]		/* to get to next row */
#define dt_mach_rinc	dt_mach_inc[2]		/* to get to next row */
#define dt_mach_finc	dt_mach_inc[3]		/* to get to next frame */
#define dt_mach_sinc	dt_mach_inc[4]		/* to get to next sequence */

typedef struct data_obj Data_Obj;

/* flag bits */

enum {
	DT_SEQ_BIT,		/* 0:6 object types */
	DT_IMG_BIT,
	DT_ROW_BIT,
	DT_COL_BIT,
	DT_SCL_BIT,
	DT_HYP_BIT,
	DT_UNK_BIT,
	DT_CPX_BIT,		/* 7:10 pixel types */
	DT_MLT_BIT,
	DT_BIT_BIT,
	DT_QTR_BIT,
	DT_ZMB_BIT,		/* 11 zombie */
	DT_CNT_BIT,		/* 12:14 contiguity */
	DT_CHK_BIT,
	DT_EVN_BIT,
	DT_ALN_BIT,		/* 15 aligned */
	DT_LCK_BIT,		/* 16 locked */
	DT_ASS_BIT,		/* 17 assigned */
	DT_TMP_BIT,		/* 18 temporary */
	DT_VOI_BIT,		/* 19 void */
	DT_EXP_BIT,		/* 20 exported */
	DT_RDO_BIT,		/* 21 read-only */
	DT_VOL_BIT,		/* 22 volatile */
	DT_NOD_BIT,		/* 23 no data */
	DT_INT_BIT,		/* 24 interlaced */
	DT_LST_BIT,		/* 25 list object */
	DT_STR_BIT,		/* 26 string */
	DT_CHR_BIT,		/* 27 character */
	DT_STA_BIT,		/* 28 static object */
	DT_GLB_BIT,		/* 29 object is GL buffer */
	DT_MAP_BIT,		/* 30 GL buffer is mapped */
	N_DP_FLAGS		/* must be last */
};

#define DT_SEQUENCE	(1<<DT_SEQ_BIT)
#define DT_IMAGE	(1<<DT_IMG_BIT)
#define DT_ROWVEC	(1<<DT_ROW_BIT)
#define DT_COLVEC	(1<<DT_COL_BIT)
#define DT_SCALAR	(1<<DT_SCL_BIT)
#define DT_HYPER_SEQ	(1<<DT_HYP_BIT)
#define DT_UNKNOWN_SHAPE (1<<DT_UNK_BIT)
#define DT_COMPLEX	(1<<DT_CPX_BIT)
#define DT_MULTIDIM	(1<<DT_MLT_BIT)
#define DT_STRING	(1<<DT_STR_BIT)
#define DT_BIT		(1<<DT_BIT_BIT)
#define DT_QUAT		(1<<DT_QTR_BIT)
#define DT_CHAR		(1<<DT_CHR_BIT)
#define DT_ZOMBIE	(1<<DT_ZMB_BIT)		/* set if an application needs to keep this data */
#define	DT_CONTIG	(1<<DT_CNT_BIT)		/* object is known to be contiguous */
#define	DT_CHECKED	(1<<DT_CHK_BIT)		/* contiguity checked */
#define	DT_EVENLY	(1<<DT_EVN_BIT)		/* evenly spaced data */
#define	DT_ALIGNED	(1<<DT_ALN_BIT)		/* data area from memalign */
#define DT_LOCKED	(1<<DT_LCK_BIT)		/* don't delete (for tmpobjs?) */
#define DT_ASSIGNED	(1<<DT_ASS_BIT)		/* Object has had data assigned */
#define DT_TEMP		(1<<DT_TMP_BIT)		/* temporary object flag */
#define DT_VOID		(1<<DT_VOI_BIT)		/* another object type */
#define DT_EXPORTED	(1<<DT_EXP_BIT)		/* exported (to vectree) */
#define	DT_RDONLY	(1<<DT_RDO_BIT)		/* declared w/ const qualifier */
#define	DT_VOLATILE	(1<<DT_VOL_BIT)		/* can unlock temp object at end of cmd */
#define DT_NO_DATA	(1<<DT_NOD_BIT)		/* whether or not the data was gotten from getbuf */
#define DT_INTERLACED	(1<<DT_INT_BIT)		/* if this is a frame from an interlaced camera */
#define DT_OBJ_LIST	(1<<DT_LST_BIT)		/* if this is a list object */
#define DT_STATIC	(1<<DT_STA_BIT)		/* static (persistent) object */
#define DT_GL_BUF	(1<<DT_GLB_BIT)		/* refers to a GL buffer */
#define DT_BUF_MAPPED	(1<<DT_MAP_BIT)		/* GL buffer is mapped */

#define DT_VECTOR	(DT_ROWVEC|DT_COLVEC)



#define OWNS_DATA(dp)		(((dp)->dt_flags & DT_NO_DATA)==0)


/* Shape macros */

#define IS_STRING( dp )		STRING_PRECISION( ( dp )->dt_prec )
#define IS_IMAGE( dp )		IMAGE_SHAPE( &( dp )->dt_shape )
#define IS_BITMAP( dp )		BITMAP_SHAPE( &( dp )->dt_shape )
#define IS_VECTOR( dp )		VECTOR_SHAPE( &( dp )->dt_shape )
#define IS_COLVEC( dp )		COLVEC_SHAPE( &( dp )->dt_shape )
#define IS_ROWVEC( dp )		ROWVEC_SHAPE( &( dp )->dt_shape )
#define IS_SEQUENCE( dp )	SEQUENCE_SHAPE( &( dp )->dt_shape )
#define IS_HYPER_SEQ( dp )	HYPER_SEQ_SHAPE( &( dp )->dt_shape )
#define IS_SCALAR( dp )		SCALAR_SHAPE( &( dp )->dt_shape )


#define IS_QUAT( dp )		QUAT_SHAPE( &( dp )->dt_shape )
#define IS_COMPLEX( dp )	COMPLEX_SHAPE( &(( dp )->dt_shape) )
#define IS_CPX_OR_QUAT(dp)	( IS_COMPLEX(dp) || IS_QUAT(dp) )
#define IS_REAL( dp )		REAL_SHAPE( (&( dp )->dt_shape) )
#define IS_MULTIDIM( dp )	MULTIDIM_SHAPE( &( dp )->dt_shape )
/* #define IS_UNSIGNED( dp )	UNSIGNED_SHAPE( &( dp )->dt_shape ) */
/* This definition depends on the unsigned precisions coming last! */
#define IS_UNSIGNED( dp )	( MACHINE_PREC( dp ) >= PREC_UBY )

#define IS_STATIC( dp )		( (dp)->dt_flags & DT_STATIC )

#define IS_TEMP( dp )		( ( dp )->dt_flags & DT_TEMP )

#define IS_ZOMBIE(dp)		( (dp)->dt_flags & DT_ZOMBIE )

#ifdef CAUTIOUS
#define IS_CONTIGUOUS(dp)	(  ( (dp)->dt_flags & DT_CONTIG ) || 		\
				( (!(dp->dt_flags & DT_CHECKED)) && is_contiguous(dp) ) )
#define IS_EVENLY_SPACED(dp)	( (dp)->dt_flags & DT_EVENLY )

#else /* ! CAUTIOUS */
#define IS_CONTIGUOUS(dp)	(  ( (dp)->dt_flags & DT_CONTIG ) )
#endif /* ! CAUTIOUS */

#define IS_ALIGNED(dp)		( (dp)->dt_flags & DT_ALIGNED )
#define DOBJ_IS_LOCKED(dp)	( (dp)->dt_flags & DT_LOCKED )
#define HAS_VALUES(dp)		( (dp)->dt_flags & DT_ASSIGNED )

#define IS_GL_BUFFER(dp)	( (dp)->dt_flags & DT_GL_BUF )
#define BUF_IS_MAPPED(dp)	( (dp)->dt_flags & DT_BUF_MAPPED )

#define NO_OBJ		((struct data_obj *) NULL)

#define MAX_RAM_OBJS	256

#define MAXDIM	(2*512*512)

#define DNAME_VALID	"_.-/"

/* codes for subscript strings */
#define SQUARE	1
#define CURLY	2


#ifdef HAVE_CUDA

#define INSIST_RAM(dp,whence)						\
									\
	if( ! IS_RAM(dp) ){						\
		sprintf(error_string,"%s:  object %s is not in ram data area.",\
			whence,dp->dt_name);				\
		if( INTRACTIVE() )					\
			WARN(error_string);				\
		else	ERROR1(error_string);				\
	}

#else /* ! HAVE_CUDA */

#define INSIST_RAM(dp,whence)

#endif /* ! HAVE_CUDA */



/* global vars */
extern const char *dim_names[N_DIMENSIONS];
extern short siztbl[N_MACHINE_PRECS];
extern const char *prec_name[N_NAMED_PRECS];	/* descriptive strings */
extern const char *dimension_name[N_DIMENSIONS];
extern Data_Area *curr_ap;
extern int max_vectorizable;





/* contig.c */

extern int is_evenly_spaced(Data_Obj *);
extern int is_contiguous(Data_Obj *);
extern int has_contiguous_data(Data_Obj *);
extern void check_contiguity(Data_Obj *);

/* data_obj.c */

extern Data_Obj *	dobj_of(QSP_ARG_DECL  const char *);
#define DOBJ_OF(s)	dobj_of(QSP_ARG s)
extern void		list_dobjs(SINGLE_QSP_ARG_DECL);
extern Data_Obj *	new_dobj(QSP_ARG_DECL  const char *);
extern Data_Obj *	del_dobj(QSP_ARG_DECL  const char *);
extern List *		dobj_list(SINGLE_QSP_ARG_DECL);
extern Data_Obj *	pick_obj(QSP_ARG_DECL const char *pmpt);
extern void		disown_child(Data_Obj * dp);
extern void		delvec(QSP_ARG_DECL  Data_Obj * dp);
extern void		info_area(QSP_ARG_DECL  Data_Area *ap);
extern void		info_all_dps(SINGLE_QSP_ARG_DECL);
extern void		sizinit(void);
extern void		make_complex(Shape_Info *shpp);
extern int		set_shape_flags(Shape_Info *shpp,Data_Obj * dp,uint32_t shape_flag);
extern void		show_space_used(Data_Obj * dp);
extern void		dobj_iterate(Data_Obj * dp,void (*func)(Data_Obj * ,uint32_t));
extern void		dpair_iterate(QSP_ARG_DECL  Data_Obj * dp,Data_Obj * dp2,
				void (*func)(QSP_ARG_DECL  Data_Obj * ,uint32_t,Data_Obj * ,uint32_t));
extern void		gen_xpose(Data_Obj * dp,int dim1,int dim2);
extern double		get_dobj_size(Item * dp,int index);
extern double		get_dobj_il_flg(Item * dp);
extern void		dataobj_init(SINGLE_QSP_ARG_DECL);
extern void		init_dfuncs(void);
extern int		same_shape(Shape_Info *,Shape_Info *);

/* dplist.c */

/* apparently , this file was split off from data_obj.c, bacause
 * some of the prototypes are still listed above!?
 */

extern const char *	name_for_prec(int prec);
extern void		dump_shape(Shape_Info *shpp);
extern void		listone(Data_Obj * dp);
extern void		longlist(QSP_ARG_DECL  Data_Obj * dp);
extern void		describe_shape(Shape_Info *shpp);

#define LONGLIST(dp)	longlist(QSP_ARG  dp)

/* pars_obj.c */
extern Data_Obj * pars_obj(const char *);

/* arrays.c */
/* formerly in arrays.h */

extern void release_tmp_obj(Data_Obj *);
extern void unlock_all_tmp_objs(void);
extern void set_array_base_index(QSP_ARG_DECL  int);
extern void init_tmp_dps(void);
extern Data_Obj * find_free_temp_dp(Data_Obj *dp);
extern Data_Obj * temp_child(const char *name,Data_Obj * dp);
extern void make_array_name(QSP_ARG_DECL  char *target_str,Data_Obj * dp,
   index_t index,int which_dim,int subscr_type);
extern Data_Obj * gen_subscript(QSP_ARG_DECL  Data_Obj * dp,
   int which_dim,index_t index,int subscr_type);
extern Data_Obj * reduce_from_end(QSP_ARG_DECL  Data_Obj * dp,
   index_t index,int subscr_type);
extern Data_Obj * d_subscript(QSP_ARG_DECL  Data_Obj * dp,index_t index);
extern Data_Obj * c_subscript(QSP_ARG_DECL  Data_Obj * dp,index_t index);
extern int is_in_string(int c,const char *s);
extern void reindex(QSP_ARG_DECL  Data_Obj * ,int,index_t);
extern void list_temp_dps(void);

/* get_obj.c */
/* formerly in get_obj.h */

extern Data_Obj * hunt_obj(QSP_ARG_DECL  const char *s);
extern Data_Obj * get_obj(QSP_ARG_DECL  const char *s);
#define GET_OBJ(s)	get_obj(QSP_ARG  s)
extern Data_Obj * get_vec(QSP_ARG_DECL  const char *s);
extern Data_Obj * img_of(QSP_ARG_DECL  const char *s);
extern Data_Obj * get_seq(QSP_ARG_DECL  const char *s);
extern Data_Obj * get_img(QSP_ARG_DECL  const char *s );


/* data_fns.c */

extern void *		multiply_indexed_data(Data_Obj *dp, dimension_t *offset );
extern void *		indexed_data(Data_Obj *dp, dimension_t offset );
extern void		make_contiguous(Data_Obj *);
extern int		set_shape_dimensions(QSP_ARG_DECL  Shape_Info *shpp,Dimension_Set *dimensions,prec_t);
extern int		set_obj_dimensions(QSP_ARG_DECL  Data_Obj *dp,Dimension_Set *dimensions,prec_t);
extern int		obj_rename(QSP_ARG_DECL  Data_Obj *dp,const char *newname);
extern Data_Obj *	make_obj_list(QSP_ARG_DECL  const char *name,List *lp);
extern Data_Obj *	make_obj(QSP_ARG_DECL  const char *name,dimension_t frames,
	dimension_t rows,dimension_t cols,dimension_t type_dim,prec_t prec);
extern Data_Obj *	mk_scalar(QSP_ARG_DECL  const char *name,prec_t prec);
extern void		assign_scalar(QSP_ARG_DECL  Data_Obj *,Scalar_Value *);
extern void		extract_scalar_value(Scalar_Value *, Data_Obj *);
extern double		cast_from_scalar_value(QSP_ARG_DECL  Scalar_Value *, prec_t prec);
extern void		cast_to_scalar_value(QSP_ARG_DECL  Scalar_Value *, prec_t prec, double val);
extern const char *	string_for_scalar_value(Scalar_Value *, prec_t prec);
extern void		cast_to_cpx_scalar(QSP_ARG_DECL  int index, Scalar_Value *, prec_t prec, double val);
extern void		cast_to_quat_scalar(QSP_ARG_DECL  int index, Scalar_Value *, prec_t prec, double val);
extern Data_Obj *	mk_cscalar(QSP_ARG_DECL  const char *name,double rval, double ival);
extern Data_Obj *	mk_img(QSP_ARG_DECL  const char *,dimension_t,dimension_t,dimension_t ,prec_t);
extern Data_Obj *	mk_vec(QSP_ARG_DECL  const char *,dimension_t, dimension_t,prec_t prec);
extern Data_Obj *	comp_replicate(QSP_ARG_DECL  Data_Obj *dp,int n,int allocate_data);
extern Data_Obj *	dup_half(QSP_ARG_DECL  Data_Obj *dp,const char *name);
extern Data_Obj *	dup_dbl(QSP_ARG_DECL  Data_Obj *dp,const char *name);
extern Data_Obj *	dup_obj(QSP_ARG_DECL  Data_Obj *dp,const char *name);
extern const char *	localname(void);
extern Data_Obj *	dupdp(QSP_ARG_DECL  Data_Obj *dp);
extern int		is_valid_dname(QSP_ARG_DECL  const char *name);


/* makedobj.c */
extern void	  set_dp_alignment(int);
extern Data_Obj * make_dobj_with_shape(QSP_ARG_DECL  const char *name,Dimension_Set *,prec_t,uint32_t);
//extern Data_Obj * _make_dp_with_maxdim(QSP_ARG_DECL  const char *name,Dimension_Set *,prec_t,int);
extern Data_Obj * make_dobj(QSP_ARG_DECL  const char *name,Dimension_Set *,prec_t);
extern Data_Obj * setup_dp(QSP_ARG_DECL  Data_Obj *dp,prec_t);
extern Data_Obj * _make_dp(QSP_ARG_DECL  const char *name,Dimension_Set *,prec_t);
extern Data_Obj * init_dp(QSP_ARG_DECL  Data_Obj *dp,Dimension_Set *,prec_t);


/* formerly in areas.h */
/* areas.c */

ITEM_INTERFACE_PROTOTYPES(Data_Area,data_area)

void			push_data_area(Data_Area *);
void			pop_data_area(void);
extern List *		da_list(SINGLE_QSP_ARG_DECL);
extern void		a_init(void);
extern Data_Area *	default_data_area(SINGLE_QSP_ARG_DECL);
extern void		set_data_area(Data_Area *);
extern Data_Obj *	search_areas(const char *name);
extern Data_Area *	area_init(QSP_ARG_DECL  const char *name,u_char *buffer,uint32_t siz,int nobjs,uint32_t flags);
extern Data_Area *	new_area(QSP_ARG_DECL  const char *s,uint32_t siz,int n);
extern void		list_area(Data_Area *ap);
extern void		data_area_info(Data_Area *ap);
extern int		dp_addr_cmp(CONST void *dpp1,CONST void *dpp2);
extern void		show_area_space(QSP_ARG_DECL  Data_Area *ap);

/* formerly in index.h */
extern Data_Obj * index_data( QSP_ARG_DECL  Data_Obj *dp, const char *index_str );

/* memops.c */
extern int not_prec(QSP_ARG_DECL  Data_Obj *,prec_t);
extern void check_vectorization(Data_Obj *dp);
extern void dp1_vectorize(QSP_ARG_DECL  int,Data_Obj *,void (*func)(Data_Obj *) );
extern void dp2_vectorize(QSP_ARG_DECL  int,Data_Obj *,Data_Obj *,
				void (*func)(Data_Obj *,Data_Obj *) );
extern void getmean(QSP_ARG_DECL  Data_Obj *dp);
extern void dp_equate(QSP_ARG_DECL  Data_Obj *dp, double v);
extern void dp_copy(QSP_ARG_DECL  Data_Obj *dp_to, Data_Obj *dp_fr);
extern void i_rnd(QSP_ARG_DECL  Data_Obj *dp, int imin, int imax);
extern void dp_uni(QSP_ARG_DECL  Data_Obj *dp);
extern void mxpose(Data_Obj *dp_to, Data_Obj *dp_fr);

extern int dp_same_prec(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);
extern int dp_same_mach_prec(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);
extern int dp_same_pixel_type(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);
extern int dp_same_size(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);
extern int dp_same_dim(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,int index,const char *whence);
extern int dp_same_dims(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,int index1,int index2,const char *whence);
extern int dp_same(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,const char *whence);

extern int dp_same_size_query(Data_Obj *dp1,Data_Obj *dp2);
extern int dp_same_size_query_rc(QSP_ARG_DECL  Data_Obj *real_dp,Data_Obj *cpx_dp);
extern int dp_equal_dim(Data_Obj *dp1,Data_Obj *dp2,int index);
extern int dp_equal_dims(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2,int index1,int index2);
extern int dp_same_len(QSP_ARG_DECL  Data_Obj *dp1,Data_Obj *dp2);

/* dfuncs.c  support for pexpr */
extern double obj_exists(QSP_ARG_DECL  const char *);
extern double comp_func(Data_Obj *,index_t);
extern double val_func(Data_Obj *);
/*
extern double val_func(Data_Obj *);
extern double re_func(Data_Obj *);
extern double im_func(Data_Obj *);
*/
extern double row_func(Data_Obj *);
extern double depth_func(Data_Obj *);
extern double col_func(Data_Obj *);
extern double frm_func(Data_Obj *);
extern double seq_func(Data_Obj *);

/* sub_obj.c */

extern void parent_relationship(Data_Obj *parent,Data_Obj *child);
extern Data_Obj *mk_subseq(QSP_ARG_DECL  const char *name, Data_Obj *parent,
 index_t *offsets, Dimension_Set *sizes);
extern Data_Obj *mk_ilace(QSP_ARG_DECL  Data_Obj *parent, const char *name, int parity);
extern int __relocate(QSP_ARG_DECL  Data_Obj *dp,index_t *offsets);
extern int _relocate(QSP_ARG_DECL  Data_Obj *dp,index_t xos,index_t yos,index_t tos);
extern Data_Obj *mk_subimg(QSP_ARG_DECL  Data_Obj *parent, index_t xos, index_t yos,
 const char *name, dimension_t rows, dimension_t cols);
extern Data_Obj *nmk_subimg(QSP_ARG_DECL  Data_Obj *parent, index_t xos, index_t yos, 
 const char *name, dimension_t rows, dimension_t cols, dimension_t tdim);
extern Data_Obj *make_equivalence(QSP_ARG_DECL  const char *name, Data_Obj *dp,
 Dimension_Set *dsp, prec_t prec);
extern Data_Obj *make_subsamp(QSP_ARG_DECL  const char *name, Data_Obj *dp,
 Dimension_Set *sizes, index_t *offsets, incr_t *incrs );
extern void propagate_flag_to_children(Data_Obj *dp, uint32_t flags );


/* verdata.c */
extern void verdata(SINGLE_QSP_ARG_DECL);


/* ascii.c */
extern void format_scalar_obj(char *buf,Data_Obj *dp,void *data);
extern void format_scalar_value(char *buf,void *data,prec_t prec);
extern void pntvec(QSP_ARG_DECL  Data_Obj *dp, FILE *fp);
extern void read_ascii_data(QSP_ARG_DECL Data_Obj *dp, FILE *fp, const char *s, int expect_exact_count);
extern void read_obj(QSP_ARG_DECL Data_Obj *dp);
extern void dptrace(Data_Obj *);
extern void set_integer_print_fmt(QSP_ARG_DECL  Number_Fmt fmt_code);
extern void set_max_per_line(QSP_ARG_DECL  int n);
extern void set_input_format_string(QSP_ARG_DECL  const char *s);
extern void set_display_precision(int);

/* datamenu.c */
extern int get_precision(SINGLE_QSP_ARG_DECL);

#ifdef __cplusplus
}
#endif


#endif /* ! _DATA_OBJ_H_ */

