
#ifndef N_DIMENSIONS

#include <stdint.h>
#include <sys/types.h>

//typedef unsigned char u_char;
//typedef unsigned short u_short;


/* flag bits */
// These used to be data_obj flags (hence the DT prefix), but now they are all shape_info flags

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
	DT_SHP_BIT,		/* 31 shape checked */
	DT_PAS_BIT,		/* 32 partially assigned */
	DT_CBT_BIT,		/* 33 contiguous bitmap data */
	DT_BMI_BIT,		/* 34 Bitmap_GPU_Info present */
	N_DP_FLAGS		/* must be last */
};

#define AUTO_SHAPE	0

#define SHIFT_IT(bit_no)	(((shape_flag_t)1)<<bit_no)

#define DT_SEQUENCE	SHIFT_IT(DT_SEQ_BIT)
#define DT_IMAGE	SHIFT_IT(DT_IMG_BIT)
#define DT_ROWVEC	SHIFT_IT(DT_ROW_BIT)
#define DT_COLVEC	SHIFT_IT(DT_COL_BIT)
#define DT_SCALAR	SHIFT_IT(DT_SCL_BIT)
#define DT_HYPER_SEQ	SHIFT_IT(DT_HYP_BIT)
#define DT_UNKNOWN_SHAPE SHIFT_IT(DT_UNK_BIT)
#define DT_COMPLEX	SHIFT_IT(DT_CPX_BIT)
#define DT_MULTIDIM	SHIFT_IT(DT_MLT_BIT)
#define DT_STRING	SHIFT_IT(DT_STR_BIT)
#define DT_BIT		SHIFT_IT(DT_BIT_BIT)
#define DT_QUAT		SHIFT_IT(DT_QTR_BIT)
#define DT_CHAR		SHIFT_IT(DT_CHR_BIT)
#define DT_ZOMBIE	SHIFT_IT(DT_ZMB_BIT)		/* set if an application needs to keep this data */
#define	DT_CONTIG	SHIFT_IT(DT_CNT_BIT)		/* object is known to be contiguous */
#define	DT_CONTIG_BITMAP_DATA	SHIFT_IT(DT_CBT_BIT)		/* bitmap with contiguous enclosing data */
#define	DT_HAS_BITMAP_GPU_INFO	SHIFT_IT(DT_BMI_BIT)		/* non-contiguous bitmap gpu info present */
#define	DT_CHECKED	SHIFT_IT(DT_CHK_BIT)		/* contiguity checked */
#define	DT_EVENLY	SHIFT_IT(DT_EVN_BIT)		/* evenly spaced data */
#define	DT_ALIGNED	SHIFT_IT(DT_ALN_BIT)		/* data area from memalign */
#define DT_LOCKED	SHIFT_IT(DT_LCK_BIT)		/* don't delete (for tmpobjs?) */
#define DT_ASSIGNED	SHIFT_IT(DT_ASS_BIT)		/* Object has had data assigned */
#define DT_PARTIALLY_ASSIGNED	SHIFT_IT(DT_PAS_BIT)	/* At least one subobject has had data assigned */
#define DT_TEMP		SHIFT_IT(DT_TMP_BIT)		/* temporary object flag */
#define DT_VOID		SHIFT_IT(DT_VOI_BIT)		/* another object type */
#define DT_EXPORTED	SHIFT_IT(DT_EXP_BIT)		/* exported (to vectree) */
#define	DT_RDONLY	SHIFT_IT(DT_RDO_BIT)		/* declared w/ const qualifier */
#define	DT_VOLATILE	SHIFT_IT(DT_VOL_BIT)		/* can unlock temp object at end of cmd */
#define DT_NO_DATA	SHIFT_IT(DT_NOD_BIT)		/* whether or not the data was gotten from getbuf */
#define DT_INTERLACED	SHIFT_IT(DT_INT_BIT)		/* if this is a frame from an interlaced camera */
#define DT_OBJ_LIST	SHIFT_IT(DT_LST_BIT)		/* if this is a list object */
#define DT_STATIC	SHIFT_IT(DT_STA_BIT)		/* static (persistent) object */
#define DT_GL_BUF	SHIFT_IT(DT_GLB_BIT)		/* refers to a GL buffer */
#define DT_BUF_MAPPED	SHIFT_IT(DT_MAP_BIT)		/* GL buffer is mapped */
#define DT_SHAPE_CHECKED	SHIFT_IT(DT_SHP_BIT)		/* shape has been checked */

#define DT_VECTOR	(DT_ROWVEC|DT_COLVEC)



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

#ifdef USE_LONG_DOUBLE
	PREC_LP,		/* 11 long double */
#endif // USE_LONG_DOUBLE
	/* PREC_FR, PREC_DF, */	/* unsupported sky warrior types */
	N_MACHINE_PRECS,	/* 12 */
	PREC_INVALID		/* because we may not be able to use -1 */
} mach_prec;

#define N_MACHPREC_BITS		4

#define FIRST_MACH_PREC	((mach_prec)0)		/* make sure this is the first line of the enum! */

#define N_REAL_MACHINE_PRECS	(N_MACHINE_PRECS-1)	/* don't count PREC_NONE */

#define PREC_MM	N_MACHINE_PRECS

#define N_PREC_STRINGS	(N_MACHINE_PRECS+1)

#define PREC_BY_NAME	NAME_FOR_PREC_CODE(PREC_BY)
#define PREC_IN_NAME	NAME_FOR_PREC_CODE(PREC_IN)
#define PREC_DI_NAME	NAME_FOR_PREC_CODE(PREC_DI)
#define PREC_LI_NAME	NAME_FOR_PREC_CODE(PREC_LI)
#define PREC_UBY_NAME	NAME_FOR_PREC_CODE(PREC_UBY)
#define PREC_UIN_NAME	NAME_FOR_PREC_CODE(PREC_UIN)
#define PREC_UDI_NAME	NAME_FOR_PREC_CODE(PREC_UDI)
#define PREC_ULI_NAME	NAME_FOR_PREC_CODE(PREC_ULI)
#define PREC_SP_NAME	NAME_FOR_PREC_CODE(PREC_SP)
#define PREC_DP_NAME	NAME_FOR_PREC_CODE(PREC_DP)
#ifdef USE_LONG_DOUBLE
#define PREC_LP_NAME	NAME_FOR_PREC_CODE(PREC_LP)
#endif // USE_LONG_DOUBLE

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
#define BITMAP_WORD_MSB			(SHIFT_IT(BITS_PER_BITMAP_WORD)-1)

#endif

typedef BITMAP_DATA_TYPE bitmap_word;
typedef uint64_t bit_count_t;		// even if bitmap word is 32, we can have larger bit numbers...

#ifdef FOOBAR
#define BITNUM_64	// comment out for 32 bit bit numbers
#ifdef BITNUM_64
typedef uint64_t	bitnum_t;	// could be uint32_t?
#else
typedef uint32_t	bitnum_t;	// should be uint64_t?
#endif // ! BITNUM_64
#endif // FOOBAR

#define BITS_PER_BYTE			8
#define BYTES_PER_BITMAP_WORD		((int)sizeof(BITMAP_DATA_TYPE))
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

//#define BITMAP_WORD_COUNT(dp)	(N_BITMAP_WORDS(OBJ_TYPE_DIM(dp,OBJ_MINDIM(dp))+OBJ_BIT0(dp))*(OBJ_N_TYPE_ELTS(dp)/OBJ_TYPE_DIM(dp,OBJ_MINDIM(dp))))
#define BITMAP_WORD_COUNT(dp)	bitmap_obj_word_count(dp)

// The mask wasn't needed before - that is this worked with n greater than
// the number of bits, it just rolled around.  But on iOS devices, this failed.
// Although it worked fine in the simulator!

#define NUMBERED_BIT(n)	( 1L << (n&BIT_NUMBER_MASK) )



typedef struct sp_complex {
	float	re;
	float	im;
} SP_Complex;

typedef struct dp_complex {
	double	re;
	double	im;
} DP_Complex;

typedef struct lp_complex {
	long double	re;
	long double	im;
} LP_Complex;

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

typedef struct lp_quaternion {
	long double	re;
	long double	_i;
	long double	_j;
	long double	_k;
} LP_Quaternion;

typedef struct sp_color {
	float r;
	float g;
	float b;
} SP_Color;

// No double or long double colors?

/*
 * The following are not real precisions,
 * but are sometimes used in the same context.
 *
 * These would never be encountered in a data object.
 *
 * How many bits do we have available???
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

	N_PSEUDO_PRECS	/* must be last */
} pseudo_prec;

#define N_PSEUDOPREC_BITS	4	// Must be large enough for N_PSEUDO_PRECS
					// BUG should have a software check to insure.

#define FIRST_PSEUDO_PREC	((pseudo_prec)0)

// used to be uint32_t, but now have 33 flag bits...
typedef uint64_t shape_flag_t;

//typedef int prec_t;
typedef int32_t prec_t;

#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#define PREC_FMT_X	PRIx32
#define PREC_FMT_D	PRId32
#else // ! HAVE_INTTYPES_H
#define PREC_FMT_X	"x"
#define PREC_FMT_D	"d"	// should be lld on some systems???
#endif // ! HAVE_INTTYPES_H


#define PSEUDO_PREC_BITS( pseudo_prec )			( (pseudo_prec) << N_MACHPREC_BITS )
#define PRECISION_CODE(mach_prec,pseudo_prec)		( (mach_prec) | PSEUDO_PREC_BITS(pseudo_prec) )

#define PREC_COLOR	PRECISION_CODE(	PREC_SP,   PP_COLOR )
#define PREC_VOID	PRECISION_CODE(	PREC_SP,   PP_VOID )		/* why PREC_SP??? */
#define PREC_STR	PRECISION_CODE(	PREC_BY,   PP_STRING )
#define PREC_CHAR	PRECISION_CODE(	PREC_BY,   PP_CHAR )
#define PREC_ANY	PRECISION_CODE(	PREC_NONE, PP_ANY )


/* complex flag is OR'd in */
/* These are all mutually-exclusive, so they are not flag bits
 * that can be set independently.
 */

#define LIST_PREC_BITS		PSEUDO_PREC_BITS(PP_LIST)
#define QUAT_PREC_BITS		PSEUDO_PREC_BITS(PP_QUAT)
#define COMPLEX_PREC_BITS	PSEUDO_PREC_BITS(PP_CPX)
#define BITMAP_PREC_BITS	PSEUDO_PREC_BITS(PP_BIT)
#define COLOR_PREC_BITS		PSEUDO_PREC_BITS(PP_COLOR)
#define STRING_PREC_BITS	PSEUDO_PREC_BITS(PP_STRING)
#define CHAR_PREC_BITS		PSEUDO_PREC_BITS(PP_CHAR)

#define PSEUDO_PREC_MASK	PSEUDO_PREC_BITS(SHIFT_IT(N_PSEUDOPREC_BITS)-1)
#define PP_BITS(prec)		((prec)&PSEUDO_PREC_MASK)

#define PSEUDO_PREC_INDEX(prec)	(PP_BITS(prec)>>N_MACHPREC_BITS)


/* not all of these really have names, but this guarantees that the tbls are big enough */
#define N_NAMED_PRECS		(N_MACHINE_PRECS+N_PSEUDO_PRECS)
#define PREC_INDEX(prec)	(PP_BITS(prec)==0?prec:PSEUDO_PREC_INDEX(prec))

#define PREC_LIST		(/* does a list obj have a machine prec? */ QUAT_PREC_BITS)
#define PREC_QUAT		(PREC_SP|QUAT_PREC_BITS)
#define PREC_DBLQUAT		(PREC_DP|QUAT_PREC_BITS)

#define PREC_CPX		(PREC_SP|COMPLEX_PREC_BITS)
#define PREC_DBLCPX		(PREC_DP|COMPLEX_PREC_BITS)

#ifdef USE_LONG_DOUBLE
#define PREC_LDBLCPX		(PREC_LP|COMPLEX_PREC_BITS)
#define PREC_LDBLQUAT		(PREC_LP|QUAT_PREC_BITS)
#endif // USE_LONG_DOUBLE

#define PREC_BIT		(BITMAP_MACH_PREC|BITMAP_PREC_BITS)

#define MACH_PREC_MASK		(SHIFT_IT(N_MACHPREC_BITS)-1)
#define MP_BITS(prec)		((prec)&MACH_PREC_MASK)
#define OBJ_MACH_PREC(dp)	((mach_prec)(MP_BITS(OBJ_PREC(dp))))

#define ELEMENT_INC_SIZE(dp)	(IS_BITMAP(dp) ? 1 : ELEMENT_SIZE(dp) )

#define QUAT_PRECISION(prec)	(PP_BITS(prec)==QUAT_PREC_BITS)
#define COMPLEX_PRECISION(prec)	(PP_BITS(prec)==COMPLEX_PREC_BITS)
#define NORMAL_PRECISION(prec)	( ! ( QUAT_PRECISION(prec) || COMPLEX_PRECISION(prec) ) )
#define BITMAP_PRECISION(prec)	(PP_BITS(prec)==BITMAP_PREC_BITS)
#define COLOR_PRECISION(prec)	(PP_BITS(prec)==COLOR_PREC_BITS)
#define STRING_PRECISION(prec)	(PP_BITS(prec)==STRING_PREC_BITS)
#define CHAR_PRECISION(prec)	(PP_BITS(prec)==CHAR_PREC_BITS)

#define PREC_IS_COMPLEX(prec_p)	COMPLEX_PRECISION(PREC_CODE(prec_p))

#define PRECISIONS_DIFFER(dp1,dp2)	( OBJ_PREC(dp1) != OBJ_PREC(dp2) )

// BUG redundant with OBJ_PREC...
#define PREC_OF( dp )		( OBJ_PREC( dp ) )


#define FLOATING_OBJ( dp )	FLOATING_PREC( OBJ_PREC(dp) )
#ifdef USE_LONG_DOUBLE
#define FLOATING_PREC( prec )	((MP_BITS(prec)==PREC_SP)||(MP_BITS(prec)==PREC_DP)||(MP_BITS(prec)==PREC_LP))
#else // ! USE_LONG_DOUBLE
#define FLOATING_PREC( prec )	((MP_BITS(prec)==PREC_SP)||(MP_BITS(prec)==PREC_DP))
#endif // USE_LONG_DOUBLE

#define INTEGER_PREC( prec )	(   (MP_BITS(prec)==PREC_BY)  \
				 || (MP_BITS(prec)==PREC_IN)  \
				 || (MP_BITS(prec)==PREC_DI)  \
				 || (MP_BITS(prec)==PREC_UBY) \
				 || (MP_BITS(prec)==PREC_UIN) \
				 || (MP_BITS(prec)==PREC_UDI) \
				)

#define CHAR_PREC( prec )	(   (MP_BITS(prec)==PREC_BY)  \
				 || (MP_BITS(prec)==PREC_UBY) \
				)


/*
 * A 5th dimension was added to allow conversion between interleaved color
 * representations (tdim=ncolors) and HIPS2 style frame sequential color.
 * In this latter representation, nframes becomes the number of color frames,
 * and n_seqs is the number of color frames.
 * jbm 9-11-94
 */

#define N_DIMENSIONS	5	/* color, x, y, t, hyper_t */

typedef uint32_t dimension_t;
typedef uint32_t index_t;
typedef int32_t incr_t;

#endif /* undef N_DIMENSIONS */

