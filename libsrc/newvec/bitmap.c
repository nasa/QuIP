#include "vecgen.h"
#include "bitmap.h"

/* Bitmaps are normally represented in u_longs, which can be 32 or 64 bits,
 * depending on the host architecture.
 * That makes it most efficient for doing bitwise ops on the host.
 * But for GPU computations, where we do each bit of the word
 * sequentially, and all of the words in parallel, it might be more
 * efficient to use a shorter word...
 *
 * The necessary constants are defined in data_obj.h, and include:
 * BITS_PER_BITMAP_WORD			32 or 64
 *
 * In the original implementation, the bits were all contiguous...  But to
 * support the GPU, it was necessary to make each image row be an integral
 * number of words.  Therefore, the old logic of having a single loop to
 * do all the bits is no longer valid.  We need an outer loop to iterate
 * over rows, and an inner loop to iterate over bits.
 */


#define ADVANCE_BIT( bit, lp, l )					\
									\
		if( bit == BITMAP_WORD_MSB ){				\
			bit = 1;					\
			lp++;						\
			l = *lp;					\
		} else {						\
			bit <<= 1;					\
		}

// BUG for now we haven't implemented sequences or hyperseqs...

#define BITMAP_OBJ_BINARY_FUNC( funcname, statement )			\
									\
void funcname( Vec_Obj_Args *oap )					\
{									\
	int i,j;							\
	bitmap_word *src1_ptr,*src2_ptr,*dst_ptr;			\
	bitmap_word l1,l2,l3;						\
	bitmap_word src1_bit,src2_bit,dst_bit;				\
									\
	for(i=0;i<oap->oa_dest->dt_rows;i++){				\
		src1_ptr=(bitmap_word *)oap->oa_dp[0]->dt_data;		\
		src2_ptr=(bitmap_word *)oap->oa_dp[1]->dt_data;		\
		dst_ptr=(bitmap_word *)oap->oa_dest->dt_data;		\
									\
		src1_ptr+= i * oap->oa_dp[0]->dt_mach_inc[2];		\
		src2_ptr+= i * oap->oa_dp[1]->dt_mach_inc[2];		\
		dst_ptr+= i * oap->oa_dest->dt_mach_inc[2];		\
									\
		src1_bit = 1 << oap->oa_dp[0]->dt_bit0;			\
		src2_bit = 1 << oap->oa_dp[1]->dt_bit0;			\
		dst_bit = 1 << oap->oa_dest->dt_bit0;			\
									\
		l1=*src1_ptr;						\
		l2=*src2_ptr;						\
		l3=*dst_ptr;						\
									\
		for(j=0;j<oap->oa_dest->dt_cols;j++){			\
			statement					\
									\
			ADVANCE_BIT(src1_bit,src1_ptr,l1)		\
			ADVANCE_BIT(src2_bit,src2_ptr,l2)		\
			ADVANCE_BIT(dst_bit,dst_ptr,l3)			\
		}							\
	}								\
}

#ifdef FOOBAR

#define BITMAP_OBJ_BINARY_FUNC( funcname, statement )			\
									\
void funcname( Vec_Obj_Args *oap )					\
{									\
	bitmap_word *src1_ptr,*src2_ptr,*dst_ptr;			\
	bitmap_word l1,l2,l3;						\
	bitmap_word src1_bit,src2_bit,dst_bit;				\
	dimension_t n;							\
									\
	n = oap->oa_dest->dt_n_type_elts;				\
									\
	src1_ptr=(bitmap_word *)oap->oa_dp[0]->dt_data;			\
	src2_ptr=(bitmap_word *)oap->oa_dp[1]->dt_data;			\
	dst_ptr=(bitmap_word *)oap->oa_dest->dt_data;			\
									\
	src1_bit = 1 << oap->oa_dp[0]->dt_bit0;				\
	src2_bit = 1 << oap->oa_dp[1]->dt_bit0;				\
	dst_bit = 1 << oap->oa_dest->dt_bit0;				\
									\
	l1=*src1_ptr;							\
	l2=*src2_ptr;							\
	l3=*dst_ptr;							\
	while(n--){							\
		statement						\
									\
		ADVANCE_BIT(src1_bit,src1_ptr,l1)			\
		ADVANCE_BIT(src2_bit,src2_ptr,l2)			\
		ADVANCE_BIT(dst_bit,dst_ptr,l3)				\
	}								\
}

#endif /* ! FOOBAR */

#define BITMAP_OBJ_VS_FUNC( funcname, statement )			\
void funcname( Vec_Obj_Args *oap )					\
{									\
	bitmap_word *src1_ptr,*dst_ptr;					\
	bitmap_word l1,l3;						\
	bitmap_word src1_bit,dst_bit;					\
	bitmap_word scalar_bit; /* really just a boolean */		\
	dimension_t n;							\
									\
	n = oap->oa_dest->dt_n_type_elts;				\
									\
	src1_ptr=(bitmap_word *)oap->oa_dp[0]->dt_data;			\
	scalar_bit=oap->oa_svp[0]->u_bit;				\
	dst_ptr=(bitmap_word *)oap->oa_dest->dt_data;			\
									\
	src1_bit = 1 << oap->oa_dp[0]->dt_bit0;				\
	dst_bit = 1 << oap->oa_dest->dt_bit0;				\
									\
	l1=*src1_ptr;							\
	l3=*dst_ptr;							\
	while(n--){							\
		statement						\
									\
		ADVANCE_BIT(src1_bit,src1_ptr,l1)			\
		ADVANCE_BIT(dst_bit,dst_ptr,l3)				\
	}								\
}

#define BITMAP_OBJ_UNARY_FUNC( funcname, statement )			\
									\
void funcname( Vec_Obj_Args *oap )					\
{									\
	bitmap_word *src1_ptr,*dst_ptr;					\
	bitmap_word l1,l2;						\
	bitmap_word src1_bit,dst_bit;					\
	dimension_t n;							\
									\
	n = oap->oa_dp[0]->dt_n_type_elts;				\
									\
	src1_ptr=(bitmap_word *)oap->oa_dp[0]->dt_data;			\
	dst_ptr=(bitmap_word *)oap->oa_dest->dt_data;			\
									\
	src1_bit = 1 << oap->oa_dp[0]->dt_bit0;				\
	dst_bit = 1 << oap->oa_dest->dt_bit0;				\
									\
									\
	l1=*src1_ptr;							\
	l2=*dst_ptr;							\
	while(n--){							\
		statement						\
									\
		ADVANCE_BIT(src1_bit,src1_ptr,l1)			\
		ADVANCE_BIT(dst_bit,dst_ptr,l2)				\
	}								\
}


BITMAP_OBJ_UNARY_FUNC( bitmap_obj_vnot, 
		if( l1 & src1_bit ) *dst_ptr &= ~dst_bit;
		else *dst_ptr |= dst_bit; )

BITMAP_OBJ_BINARY_FUNC( bitmap_obj_vand, 
	if( (l1 & src1_bit) && (l2 & src2_bit) ) *dst_ptr |= dst_bit; else *dst_ptr &= ~dst_bit; )

BITMAP_OBJ_BINARY_FUNC( bitmap_obj_vnand,
	if( (l1 & src1_bit) && (l2 & src2_bit) ) *dst_ptr &= ~dst_bit; else *dst_ptr |= dst_bit; )

BITMAP_OBJ_BINARY_FUNC( bitmap_obj_vor,
	if( (l1 & src1_bit) || (l2 & src2_bit) ) *dst_ptr |= dst_bit; else *dst_ptr &= ~dst_bit; )

BITMAP_OBJ_BINARY_FUNC( bitmap_obj_vxor,
	if( ( (l1 & src1_bit) && !(l2 & src2_bit) ) ||
	    (!(l1 & src1_bit) &&  (l2 & src2_bit) ) ) *dst_ptr |= dst_bit; else *dst_ptr &= ~dst_bit; )

/* void bitmap_obj_vsand( Vec_Obj_Args *oap ) { NERROR1("unimplemented bitmap function bitmap_obj_vsand"); } */

BITMAP_OBJ_VS_FUNC( bitmap_obj_vsand,
	if( (l1 & src1_bit) && scalar_bit ) *dst_ptr |= dst_bit; else *dst_ptr &= ~dst_bit; )

void bitmap_obj_vsor( Vec_Obj_Args *oap ) { NERROR1("unimplemented bitmap function bitmap_obj_vsor"); }
void bitmap_obj_vsxor( Vec_Obj_Args *oap ) { NERROR1("unimplemented bitmap function bitmap_obj_vsxor"); }


void bitmap_obj_vmod( Vec_Obj_Args *oap ) { NERROR1("unimplemented bitmap function bitmap_obj_vmod"); }
void bitmap_obj_vsmod( Vec_Obj_Args *oap ) { NERROR1("unimplemented bitmap function bitmap_obj_vsmod"); }
void bitmap_obj_vsmod2( Vec_Obj_Args *oap ) { NERROR1("unimplemented bitmap function bitmap_obj_vsmod2"); }
void bitmap_obj_vshr( Vec_Obj_Args *oap ) { NERROR1("unimplemented bitmap function bitmap_obj_vshr"); }
void bitmap_obj_vsshr( Vec_Obj_Args *oap ) { NERROR1("unimplemented bitmap function bitmap_obj_vsshr"); }
void bitmap_obj_vsshr2( Vec_Obj_Args *oap ) { NERROR1("unimplemented bitmap function bitmap_obj_vsshr2"); }
void bitmap_obj_vshl( Vec_Obj_Args *oap ) { NERROR1("unimplemented bitmap function bitmap_obj_vshl"); }
void bitmap_obj_vsshl( Vec_Obj_Args *oap ) { NERROR1("unimplemented bitmap function bitmap_obj_vsshl"); }
void bitmap_obj_vsshl2( Vec_Obj_Args *oap ) { NERROR1("unimplemented bitmap function bitmap_obj_vsshl2"); }

/* This is the same as vnot - much more efficient to complement the long words! */

BITMAP_OBJ_UNARY_FUNC( bitmap_obj_vcomp, 
		if( l1 & src1_bit ) *dst_ptr &= ~dst_bit;
		else *dst_ptr |= dst_bit; )

