#ifndef _SHAPE_INFO_H_
#define _SHAPE_INFO_H_

#include "item_type.h"
#include "shape_bits.h"

/*#define ELEMENT_SIZE(dp)	(siztbl[ MACHINE_PREC(dp) ]) */
#define ELEMENT_SIZE(dp)	OBJ_PREC_MACH_SIZE(dp)

struct dimension_set {
	dimension_t	ds_dimension[N_DIMENSIONS];
	uint32_t	ds_n_elts;	/* total number of elements */
} ;

#define DS_N_ELTS(dsp)			(dsp)->ds_n_elts
#define DS_DIM(dsp,idx)			(dsp)->ds_dimension[idx]
#define SET_DS_N_ELTS(dsp,v)		(dsp)->ds_n_elts = v
#define SET_DS_DIM(dsp,idx,v)		(dsp)->ds_dimension[idx] = v

/* Dimension_Set macros */

//#define ALLOC_DIMSET			((Dimension_Set *)getbuf(sizeof(Dimension_Set)))
#define NEW_DIMSET			((Dimension_Set *)getbuf(sizeof(Dimension_Set)))
#define INIT_DIMSET_PTR(dsp)		dsp=NEW_DIMSET;
#define RELEASE_DIMSET(dsp)		givbuf(dsp);

/* Deep or shallow copy??? */
#define DIMSET_COPY(dsp_to,dsp_fr)	*dsp_to = *dsp_fr

#define NEW_INCSET			((Increment_Set *)getbuf(sizeof(Increment_Set)))

#define DIMENSION(dsp,idx)		(dsp)->ds_dimension[idx]
#define SET_DIMENSION(dsp,idx,v)	(dsp)->ds_dimension[idx]=(dimension_t)v

#define DIMENSION_NAME(idx)	dimension_name[idx]


#define STRING_DIMENSION(dsp,len)		\
{						\
	dsp->ds_dimension[0]=1;			\
	dsp->ds_dimension[1]=len;		\
	dsp->ds_dimension[2]=1;			\
	dsp->ds_dimension[3]=1;			\
	dsp->ds_dimension[4]=1;			\
}


/* We have a problem with complex objects - in order to be able to
 * treat the real and complex parts as components, the dimensions
 * and increments reflect the machine precision of the elements.
 * But in vector operations we use a complex pointer, so they
 * are too much...  We could have the pointer be to the basic machine
 * type instead???
 */

struct increment_set {
	incr_t		is_increment[N_DIMENSIONS];
} ;

/* IncrementSet macros */
#define INCREMENT(isp,idx)		isp->is_increment[idx]
#define SET_INCREMENT(isp,idx,v)	isp->is_increment[idx]=v

struct vfunc_tbl;	// fwd declaration

struct precision {
	Item			prec_item;
	prec_t			prec_code;
	int			prec_size;	// type size in bytes
	struct precision *	prec_mach_p;	// NULL for machine precisions
	struct vfunc_tbl *	prec_vf_tbl;

	// methods

	// read script input with appropriate function, and assign scalar
	void			(*set_value_from_input_func)(QSP_ARG_DECL  void *ptr, const char *prompt);

	double			(*indexed_data_func)(Data_Obj *dp, int index);

	int			(*is_numeric_func)(void);

	// assign a scalar data object from a scalar value constant
	int			(*assign_scalar_obj_func)(Data_Obj *dp, Scalar_Value *svp);

	// extract a value from a scalar data object
	void			(*extract_scalar_func)(Scalar_Value *svp, Data_Obj *dp);

	double			(*cast_to_double_func)(Scalar_Value *svp);
	void			(*cast_from_double_func)
					(Scalar_Value *,double val);
	void			(*cast_indexed_type_from_double_func)
					(Scalar_Value *,int idx,double val);
	void			(*copy_value_func)
					(Scalar_Value *,Scalar_Value *);
} ;

#define prec_name	prec_item.item_name


#define PREC_NAME(prec_p)		(prec_p)->prec_item.item_name

#define PREC_CODE(prec_p)		(prec_p)->prec_code
#define PREC_SIZE(prec_p)		(prec_p)->prec_size
#define PREC_MACH_PREC_PTR(prec_p)	(prec_p)->prec_mach_p
#define SET_PREC_CODE(prec_p,c)		(prec_p)->prec_code = c
#define SET_PREC_CODE_BITS(prec_p,b)	(prec_p)->prec_code |= b
#define CLEAR_PREC_CODE_BITS(prec_p,b)	(prec_p)->prec_code &= (b)

#define SET_PREC_SIZE(prec_p,s)		(prec_p)->prec_size = s
#define SET_PREC_MACH_PREC_PTR(prec_p,p)	(prec_p)->prec_mach_p = p

#define PREC_MACH_CODE(prec_p)		PREC_CODE(PREC_MACH_PREC_PTR(prec_p))
#define PREC_MACH_SIZE(prec_p)		PREC_SIZE(PREC_MACH_PREC_PTR(prec_p))

#define PREC_FOR_CODE(p)		prec_for_code(p)

#define NAME_FOR_PREC_CODE(p)		PREC_NAME(PREC_FOR_CODE(p))
#define SIZE_FOR_PREC_CODE(p)		PREC_SIZE(PREC_FOR_CODE(p))

#define PREC_VFUNC_TBL(prec_p)		(prec_p)->prec_vf_tbl
#define SET_PREC_VFUNC_TBL(prec_p,v)	(prec_p)->prec_vf_tbl = v

#define PREC_SET_VALUE_FROM_INPUT_FUNC(prec_p)	(prec_p)->set_value_from_input_func
#define SET_PREC_SET_VALUE_FROM_INPUT_FUNC(prec_p,v)	(prec_p)->set_value_from_input_func = v

#define PREC_INDEXED_DATA_FUNC(prec_p)	(prec_p)->indexed_data_func
#define SET_PREC_INDEXED_DATA_FUNC(prec_p,v)	(prec_p)->indexed_data_func = v

#define PREC_IS_NUMERIC_FUNC(prec_p)	(prec_p)->is_numeric_func
#define SET_PREC_IS_NUMERIC_FUNC(prec_p,v)	(prec_p)->is_numeric_func = v

#define PREC_ASSIGN_SCALAR_FUNC(prec_p)	(prec_p)->assign_scalar_obj_func
#define SET_PREC_ASSIGN_SCALAR_FUNC(prec_p,v)	(prec_p)->assign_scalar_obj_func = v

#define PREC_EXTRACT_SCALAR_FUNC(prec_p)	(prec_p)->extract_scalar_func
#define SET_PREC_EXTRACT_SCALAR_FUNC(prec_p,v)	(prec_p)->extract_scalar_func = v

#define PREC_CAST_TO_DOUBLE_FUNC(prec_p)	(prec_p)->cast_to_double_func
#define SET_PREC_CAST_TO_DOUBLE_FUNC(prec_p,v)	(prec_p)->cast_to_double_func = v
#define PREC_CAST_FROM_DOUBLE_FUNC(prec_p)	(prec_p)->cast_from_double_func
#define SET_PREC_CAST_FROM_DOUBLE_FUNC(prec_p,v)	(prec_p)->cast_from_double_func = v

#define PREC_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(prec_p)	(prec_p)->cast_indexed_type_from_double_func
#define SET_PREC_CAST_INDEXED_TYPE_FROM_DOUBLE_FUNC(prec_p,v)	(prec_p)->cast_indexed_type_from_double_func = v

#define PREC_COPY_VALUE_FUNC(prec_p)	(prec_p)->copy_value_func
#define SET_PREC_COPY_VALUE_FUNC(prec_p,v)	(prec_p)->copy_value_func = v


//#ifdef HAVE_ANY_GPU
//#ifdef HAVE_ANY_GPU

// We use this struct to count the number of words in a bitmap, which is used
// in a non-gpu function...

// This struct allows us to simplify gpu kernels that deal with bitmaps.
// Contiguous bitmaps that have no offset and exactly fill an integral number
// of words are easy, but for bitmaps with unused word bits (including gaps
// due to increments > 1), we compute a mask of which bits are active,
// and the offset relative to the base of each word making up the bitmap.

typedef struct bitmap_gpu_word_info {
	dimension_t	word_offset;			// relative to the start of the base pointer, in words
	dimension_t	first_indices[N_DIMENSIONS];	// indices of the first valid bit
	uint64_t	first_bit_num;
	bitmap_word	valid_bits;
} Bitmap_GPU_Word_Info;

// We had problems passing a pointer to an array of word_info's inside a struct,
// so instead we put the table inside this struct definition.  We declare the array size
// to be 1, but in practice we will allocate what we need.  This way there is only 1 allocation.
// The down-side is that the struct no longer has a fixed size.

typedef struct bitmap_gpu_info {
	dimension_t			n_bitmap_words;
	uint32_t			total_size;	// size of this struct in bytes
	dimension_t			next_word_idx;
	dimension_t			this_word_idx;
	dimension_t			last_word_idx;
	Bitmap_GPU_Word_Info 		word_tbl[1];
} Bitmap_GPU_Info;

#define BITMAP_GPU_INFO_SIZE(n_words)	(sizeof(Bitmap_GPU_Info)+(n_words-1)*sizeof(Bitmap_GPU_Word_Info))

#define BMI_N_WORDS(bmi_p)		(bmi_p)->n_bitmap_words
#define SET_BMI_N_WORDS(bmi_p,n)	(bmi_p)->n_bitmap_words = n
#define BMI_WORD_TBL(bmi_p)		(bmi_p)->word_tbl
#define BMI_WORD_INFO_P(bmi_p,idx)	(&((bmi_p)->word_tbl[idx]))

#define BMI_LAST_WORD_IDX(bmi_p)	(bmi_p)->last_word_idx
#define SET_BMI_LAST_WORD_IDX(bmi_p,v)	(bmi_p)->last_word_idx = v

#define BMI_THIS_WORD_IDX(bmi_p)	(bmi_p)->this_word_idx
#define SET_BMI_THIS_WORD_IDX(bmi_p,v)	(bmi_p)->this_word_idx = v

#define BMI_NEXT_WORD_IDX(bmi_p)	(bmi_p)->next_word_idx
#define SET_BMI_NEXT_WORD_IDX(bmi_p,v)	(bmi_p)->next_word_idx = v

#define BMI_STRUCT_SIZE(bmi_p)		(bmi_p)->total_size
#define SET_BMI_STRUCT_SIZE(bmi_p,s)	(bmi_p)->total_size = s

#define BMWI_OFFSET(bmwi_p)		(bmwi_p)->word_offset
#define BMWI_FIRST_INDICES(bmwi_p)	(bmwi_p)->first_indices
#define BMWI_FIRST_INDEX(bmwi_p,which)		BMWI_FIRST_INDICES(bmwi_p)[which]
#define SET_BMWI_FIRST_INDEX(bmwi_p,which,v)	BMWI_FIRST_INDICES(bmwi_p)[which] = v

#define BMWI_FIRST_BIT_NUM(bmwi_p)		(bmwi_p)->first_bit_num
#define SET_BMWI_FIRST_BIT_NUM(bmwi_p,n)	(bmwi_p)->first_bit_num = n

#define BMWI_VALID_BITS(bmwi_p)			(bmwi_p)->valid_bits
#define SET_BMWI_VALID_BITS(bmwi_p,bits)	(bmwi_p)->valid_bits = bits
#define SET_BMWI_VALID_BIT(bmwi_p,bits)		(bmwi_p)->valid_bits |= bits

#define SET_BMWI_OFFSET(bmwi_p,v)	(bmwi_p)->word_offset = v

#define UNLIKELY_INDEX			((dimension_t) ((int32_t)-1))

//#endif // HAVE_ANY_GPU


struct shape_info {
	Dimension_Set *		si_mach_dims;
	Dimension_Set *		si_type_dims;	// what if these are the same???
	Increment_Set *		si_mach_incs;
	Increment_Set *		si_type_incs;	// what if these are the same???
	Precision *		si_prec_p;
	int32_t			si_maxdim;
	int32_t			si_mindim;
	int32_t			si_range_maxdim;
	int32_t			si_range_mindim;
	shape_flag_t		si_flags;
	incr_t			si_eqsp_inc;
	/*int32_t			si_last_subi; */
	// only used for bitmaps - candidate for union
	Bitmap_GPU_Info *	si_bitmap_gpu_info_h;	// address in host memory
							// We may not need to keep this around once we have things working,
							// but for now it's helpful for debugging.
#ifdef HAVE_ANY_GPU
	Bitmap_GPU_Info *	si_bitmap_gpu_info_g;	// address in device memory
#endif // HAVE_ANY_GPU
} ;

/* Shape macros */

#define VOID_SHAPE( shpp )		( SHP_FLAGS( shpp ) & DT_VOID )
#define IMAGE_SHAPE( shpp )		( SHP_FLAGS( shpp ) & DT_IMAGE )
#define BITMAP_SHAPE( shpp )		( SHP_FLAGS( shpp ) & DT_BIT )
#define VECTOR_SHAPE( shpp )		( SHP_FLAGS( shpp ) & DT_VECTOR )
#define COLVEC_SHAPE( shpp )		( SHP_FLAGS( shpp ) & DT_COLVEC )
#define ROWVEC_SHAPE( shpp )		( SHP_FLAGS( shpp ) & DT_ROWVEC )
#define SEQUENCE_SHAPE( shpp )		( SHP_FLAGS( shpp ) & DT_SEQUENCE )
#define HYPER_SEQ_SHAPE( shpp )		( SHP_FLAGS( shpp ) & DT_HYPER_SEQ )
/* BUG?  is this correct for complex/quaternion??? */
#define SCALAR_SHAPE( shpp )		( ( SHP_FLAGS( shpp ) & DT_SCALAR ) && ( DIMENSION(SHP_TYPE_DIMS(shpp),0) == 1 ) )
#define PIXEL_SHAPE( shpp )		( ( SHP_FLAGS( shpp ) & DT_SCALAR ) )
#define UNKNOWN_SHAPE( shpp )		( SHP_FLAGS( shpp ) & DT_UNKNOWN_SHAPE )

#define UNKNOWN_OBJ_SHAPE(dp)		UNKNOWN_SHAPE(OBJ_SHAPE(dp))

#define INTERLACED_SHAPE( shpp )	( SHP_FLAGS( shpp ) & DT_INTERLACED )

#define COMPLEX_SHAPE( shpp )		( COMPLEX_PRECISION( SHP_PREC(shpp) ) )
#define QUAT_SHAPE( shpp )		( QUAT_PRECISION( SHP_PREC(shpp) ) )
#define REAL_SHAPE( shpp )		( (SHP_FLAGS( shpp ) & (SHAPE_TYPE_MASK&~DT_MULTIDIM)) == 0 )
#define MULTIDIM_SHAPE( shpp )		( SHP_FLAGS( shpp ) & DT_MULTIDIM )

#define SHAPE_DIM_MASK	(DT_UNKNOWN_SHAPE|DT_HYPER_SEQ|DT_SEQUENCE|DT_IMAGE|DT_VECTOR|DT_SCALAR)
/* don't care about unsigned... */
#define SHAPE_TYPE_MASK	(DT_COMPLEX|DT_MULTIDIM|DT_QUAT)
#define SHAPE_MASK	(SHAPE_DIM_MASK|SHAPE_TYPE_MASK)

/* ShapeInfo macros */
#define COPY_DIMS(dsp1,dsp2)	*dsp1 = *dsp2
#define COPY_INCS(isp1,isp2)	*isp1 = *isp2

/* This line was put in COPY_SHAPE for extra debugging... */
/*show_shape_addrs("dst",dst_shpp); show_shape_addrs("src",src_shpp); */

// COPY_SHAPE used to be a simple structure copy, but now the shape
// contains pointers to stuff...
// We could encapsulate the local data into a struct?

#define COPY_SHAPE(dst_shpp,src_shpp)	{				\
									\
	SET_SHP_PREC_PTR(dst_shpp, SHP_PREC_PTR(src_shpp) );		\
	SET_SHP_MAXDIM(dst_shpp, SHP_MAXDIM(src_shpp) );		\
	SET_SHP_MINDIM(dst_shpp, SHP_MINDIM(src_shpp) );		\
	SET_SHP_RANGE_MAXDIM(dst_shpp, SHP_RANGE_MAXDIM(src_shpp) );	\
	SET_SHP_RANGE_MINDIM(dst_shpp, SHP_RANGE_MINDIM(src_shpp) );	\
	SET_SHP_FLAGS(dst_shpp, SHP_FLAGS(src_shpp) );			\
	/*SET_SHP_LAST_SUBI(dst_shpp, SHP_LAST_SUBI(src_shpp) );*/	\
	COPY_DIMS((SHP_TYPE_DIMS(dst_shpp)),(SHP_TYPE_DIMS(src_shpp)));	\
	COPY_DIMS((SHP_MACH_DIMS(dst_shpp)),(SHP_MACH_DIMS(src_shpp)));	\
	COPY_INCS((SHP_TYPE_INCS(dst_shpp)),(SHP_TYPE_INCS(src_shpp)));	\
	COPY_INCS((SHP_MACH_INCS(dst_shpp)),(SHP_MACH_INCS(src_shpp)));	\
	}

#define NEW_SHAPE_INFO		((Shape_Info *)getbuf(sizeof(Shape_Info)))
#define SHP_PREC(shp)		PREC_CODE( (shp)->si_prec_p )
#define SHP_PREC_PTR(shp)	(shp)->si_prec_p
/* We may not want to diddle these bits - because now the precisions */
/* are pointed-to structures?!  BUG? */
#define SET_SHP_PREC_BITS(shp,b)	SET_PREC_CODE_BITS( (shp)->si_prec_p, b )
#define CLEAR_SHP_PREC_BITS(shp,b)	CLEAR_PREC_CODE_BITS( (shp)->si_prec_p, b )
#define SHP_MACH_PREC_PTR(shp)	(shp)->si_prec_p->prec_mach_p
#define SHP_PREC_MACH_SIZE(shp)	PREC_MACH_SIZE(shp->si_prec_p)
#define SHP_MAXDIM(shp)		(shp)->si_maxdim
#define SHP_MINDIM(shp)		(shp)->si_mindim
#define SHP_RANGE_MAXDIM(shp)	(shp)->si_range_maxdim
#define SHP_RANGE_MINDIM(shp)	(shp)->si_range_mindim
#define SHP_MACH_DIMS(shp)	(shp)->si_mach_dims
#define SHP_TYPE_DIMS(shp)	(shp)->si_type_dims
#define SHP_TYPE_DIM(shp,idx)	DS_DIM((shp)->si_type_dims,idx)
#define SHP_MACH_DIM(shp,idx)	DS_DIM((shp)->si_mach_dims,idx)
#define SHP_N_MACH_ELTS(shpp)	DS_N_ELTS((shpp)->si_mach_dims)
#define SHP_N_TYPE_ELTS(shpp)	DS_N_ELTS((shpp)->si_type_dims)
//#define SHP_LAST_SUBI(shpp)	(shpp)->si_last_subi
#define SHP_FLAGS(shpp)		(shpp)->si_flags
#define SET_SHP_FLAGS(shpp,f)	(shpp)->si_flags = f

#define SHP_SEQS(shpp)		DS_SEQS(   (shpp)->si_type_dims )
#define SHP_FRAMES(shpp)	DS_FRAMES( (shpp)->si_type_dims )
#define SHP_ROWS(shpp)		DS_ROWS(   (shpp)->si_type_dims )
#define SHP_COLS(shpp)		DS_COLS(   (shpp)->si_type_dims )
#define SHP_COMPS(shpp)		DS_COMPS(  (shpp)->si_type_dims )
#define SHP_DIMENSION(shpp,idx)	DIMENSION( (shpp)->si_type_dims, idx )

#define SET_SHP_SEQS(shpp,v)	SET_DS_SEQS(   (shpp)->si_type_dims,v)
#define SET_SHP_FRAMES(shpp,v)	SET_DS_FRAMES( (shpp)->si_type_dims, v)
#define SET_SHP_ROWS(shpp,v)	SET_DS_ROWS(   (shpp)->si_type_dims, v )
#define SET_SHP_COLS(shpp,v)	SET_DS_COLS(   (shpp)->si_type_dims, v )
#define SET_SHP_COMPS(shpp,v)	SET_DS_COMPS(  (shpp)->si_type_dims, v)

#define DS_SEQS(dsp)		DIMENSION(dsp,4)
#define DS_FRAMES(dsp)		DIMENSION(dsp,3)
#define DS_ROWS(dsp)		DIMENSION(dsp,2)
#define DS_COLS(dsp)		DIMENSION(dsp,1)
#define DS_COMPS(dsp)		DIMENSION(dsp,0)

#define SET_DS_SEQS(dsp,v)	SET_DIMENSION(dsp,4,v)
#define SET_DS_FRAMES(dsp,v)	SET_DIMENSION(dsp,3,v)
#define SET_DS_ROWS(dsp,v)	SET_DIMENSION(dsp,2,v)
#define SET_DS_COLS(dsp,v)	SET_DIMENSION(dsp,1,v)
#define SET_DS_COMPS(dsp,v)	SET_DIMENSION(dsp,0,v)

#define SHP_MACH_INCS(shp)		(shp)->si_mach_incs
#define SHP_TYPE_INCS(shp)		(shp)->si_type_incs
#define SET_SHP_MACH_INCS(shp,isp)	(shp)->si_mach_incs = isp
#define SET_SHP_TYPE_INCS(shp,isp)	(shp)->si_type_incs = isp
#define SHP_MACH_INC(shp,idx)		(shp)->si_mach_incs->is_increment[idx]
#define SHP_TYPE_INC(shp,idx)		(shp)->si_type_incs->is_increment[idx]
#define SET_SHP_MACH_INC(shp,idx,v)	(shp)->si_mach_incs->is_increment[idx] = v
#define SET_SHP_TYPE_INC(shp,idx,v)	(shp)->si_type_incs->is_increment[idx] = v

#define SET_SHP_MACH_DIMS(shp,dsp)	(shp)->si_mach_dims = dsp
#define SET_SHP_TYPE_DIMS(shp,dsp)	(shp)->si_type_dims = dsp
#define SET_SHP_TYPE_DIM(shp,idx,v)	SET_DS_DIM((shp)->si_type_dims,idx,v)
#define SET_SHP_MACH_DIM(shp,idx,v)	SET_DS_DIM((shp)->si_mach_dims,idx,v)
#define SET_SHP_PREC_PTR(shp,prec_p)	(shp)->si_prec_p = prec_p
#define SET_SHP_MAXDIM(shp,v)		(shp)->si_maxdim = v
#define SET_SHP_MINDIM(shp,v)		(shp)->si_mindim = v
#define SET_SHP_RANGE_MAXDIM(shp,v)	(shp)->si_range_maxdim = v
#define SET_SHP_RANGE_MINDIM(shp,v)	(shp)->si_range_mindim = v
#define SET_SHP_N_MACH_ELTS(shp,v)	SET_DS_N_ELTS((shp)->si_mach_dims,v)
#define SET_SHP_N_TYPE_ELTS(shp,v)	SET_DS_N_ELTS((shp)->si_type_dims,v)
//#define SET_SHP_LAST_SUBI(shp,v)	(shp)->si_last_subi = v
#define SET_SHP_FLAG_BITS(shp,v)	(shp)->si_flags |= v
#define CLEAR_SHP_FLAG_BITS(shp,v)	(shp)->si_flags &= ~(v)

#define SHP_EQSP_INC(shp)		(shp)->si_eqsp_inc
#define SET_SHP_EQSP_INC(shp,v)		(shp)->si_eqsp_inc = v

#define SHP_BITMAP_GPU_INFO_H(shp)		(shp)->si_bitmap_gpu_info_h
#define SET_SHP_BITMAP_GPU_INFO_H(shp,p)	(shp)->si_bitmap_gpu_info_h = p

#define SHP_BITMAP_GPU_INFO_G(shp)		(shp)->si_bitmap_gpu_info_g
#define SET_SHP_BITMAP_GPU_INFO_G(shp,p)	(shp)->si_bitmap_gpu_info_g = p

// BUG currently in vectree/comptree.c, but should be moved!
extern Shape_Info *_make_outer_shape(QSP_ARG_DECL  Shape_Info *,Shape_Info *);
#define make_outer_shape(shpp1,shpp2) _make_outer_shape(QSP_ARG  shpp1,shpp2)

#endif /* ! _SHAPE_INFO_H_ */

