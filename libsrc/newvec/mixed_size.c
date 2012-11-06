#include "quip_config.h"


char VersionId_newvec_mixed_size[] = QUIP_VERSION_STRING;

#include "vec_util.h"
#include "nvf.h"

/*
 * Mixed size ops.
 *
 * We define mixed size ops as follows:
 *
 * if we add a column to an image, then we do a row-by-row vector scalar
 * addition of the row of the image with the corresponding value of the scalar.
 * In general, the following constraint must be satisfied:
 *   foreach dimension:
 *     dimensions match
 *     OR
 *     at least one dimension is one.
 *
 * When this is satisfied, we vectorize over the non-one dimensions, and pick
 * scalars out of the one dimensions.
 *
 * The outer product is an example of this for multiplication.  By analogy,
 * we define the outer sum, outer difference, outer quotient...
 * In general we can do this with ANY binary op!
 *
 * Note that for a row, column outer product, there is no vectorization,
 * We have to do all scalar ops...  If we have a sequence of columns,
 * and an equal length sequence of rows, we can vectorize over time,
 * but nothing else...  In that case we compute the product of the scalars,
 * and then do a VSET.
 */

#define MAX(n1,n2)	(n1>n2?n1:n2)

#define ADVANCE_DATA_PTR(sub_dp,parent_dp)				\
									\
	cp = sub_dp->dt_data;						\
	cp += parent_dp->dt_increment[i]				\
				* siztbl[ MACHINE_PREC(parent_dp) ];	\
	sub_dp->dt_data = cp;


#define ADVANCE_BITMAP_PTR(sub_dp,parent_dp)				\
									\
	cp = sub_dp->dt_data;						\
	cp += (sub_dp->dt_bit0 + parent_dp->dt_increment[i])>>5;	\
	sub_dp->dt_bit0 =						\
		( sub_dp->dt_bit0 + parent_dp->dt_increment[i] ) & 31;	\
	sub_dp->dt_data = cp;


#define SUB_ITERATE( src_dp, small_src_dp, reverse_order )		\
									\
	Dimension_Set dst_sizes;					\
	Dimension_Set src_sizes;					\
	doff_t src_offsets[N_DIMENSIONS]={0,0,0,0,0};			\
	doff_t dst_offsets[N_DIMENSIONS]={0,0,0,0,0};			\
	incr_t src_incrs[N_DIMENSIONS]={1,1,1,1,1};			\
	incr_t dst_incrs[N_DIMENSIONS]={1,1,1,1,1};			\
	Data_Obj *sub_dst_dp, *sub_src_dp;				\
	char tmp_dst_name[LLEN];					\
	char tmp_src_name[LLEN];					\
	Vec_Obj_Args _oa;						\
									\
	/* iterate over src_dp */					\
	dst_sizes=oap->oa_dest->dt_dimset;				\
	dst_sizes.ds_dimension[i]=1;					\
	sprintf(tmp_dst_name,"dsb.%s",oap->oa_dest->dt_name);		\
	sub_dst_dp=make_subsamp(tmp_dst_name,oap->oa_dest,		\
				&dst_sizes,dst_offsets,dst_incrs);	\
									\
	src_sizes=src_dp->dt_dimset;					\
	src_sizes.ds_dimension[i]=1;					\
	sprintf(tmp_src_name,"ssb.%s",src_dp->dt_name);			\
	sub_src_dp=make_subsamp(tmp_src_name,src_dp,			\
				&src_sizes,src_offsets,src_incrs);	\
									\
	_oa.oa_dest = _oa.oa_3 = sub_dst_dp;				\
	if( reverse_order ){						\
		_oa.oa_1 = sub_src_dp;					\
		_oa.oa_2 = small_src_dp;				\
	} else {							\
		_oa.oa_2 = sub_src_dp;					\
		_oa.oa_1 = small_src_dp;				\
	}								\
	/* BUG?  oa_3?  oa_4? */					\
	_oa.oa_bmap = _oa.oa_s1 = _oa.oa_s2 = NO_OBJ;			\
									\
	for(j=0;j<src_dp->dt_dimension[i];j++){				\
		char *cp;						\
									\
	/* now the objects all have matching sizes in THIS dimension! */\
		outer_binop(&_oa,code);					\
									\
		/* now step the offset... */				\
		ADVANCE_DATA_PTR(sub_dst_dp,oap->oa_dest)		\
		ADVANCE_DATA_PTR(sub_src_dp,src_dp)			\
	}								\
	delvec(sub_dst_dp);						\
	delvec(sub_src_dp);


void outer_binop(QSP_ARG_DECL  Vec_Obj_Args *oap,Vec_Func_Code code)
{
	int i,j;

	/* find the first dimension we have to iterate over */

	for(i=0;i<N_DIMENSIONS;i++){
		if( oap->oa_1->dt_dimension[i] == 1 && oap->oa_2->dt_dimension[i] > 1 ){
			SUB_ITERATE(oap->oa_2,oap->oa_1,1)
			return;	/* all the work done in recursion */
		} else if( oap->oa_2->dt_dimension[i] == 1 && oap->oa_1->dt_dimension[i] > 1 ){
			SUB_ITERATE(oap->oa_1,oap->oa_2,0)
			return;	/* all the work done in recursion */
		}
		/* if we are here, the dimensions should match */
		if( oap->oa_1->dt_dimension[i] != oap->oa_2->dt_dimension[i] ){
			sprintf(error_string,"outer_binop:  %s count mismatch, objects %s (%ld) and %s (%ld)",
				dimension_name[i],oap->oa_1->dt_name,oap->oa_1->dt_dimension[i],
				oap->oa_2->dt_name,oap->oa_2->dt_dimension[i]);
			NWARN(error_string);
			return;
		}
		/* now we know this dimension matches */
	}
	/* now we know ALL dimensions match.
	 * Call the appropriate vector-vector op.
	 */

	perf_vfunc(QSP_ARG  code,oap);
}

/* If the bitmap is smaller, we have to make subobjects for the dest and src */

#define SUB_ITERATE_BMAP1( src_dp, small_bmap_dp )			\
									\
	Dimension_Set dst_sizes;					\
	Dimension_Set src_sizes;					\
	doff_t src_offsets[N_DIMENSIONS]={0,0,0,0,0};			\
	doff_t dst_offsets[N_DIMENSIONS]={0,0,0,0,0};			\
	incr_t src_incrs[N_DIMENSIONS]={1,1,1,1,1};			\
	incr_t dst_incrs[N_DIMENSIONS]={1,1,1,1,1};			\
	Data_Obj *sub_dst_dp, *sub_src_dp;				\
	char tmp_dst_name[LLEN];					\
	char tmp_src_name[LLEN];					\
	Vec_Obj_Args _oa;						\
									\
	/* iterate over src_dp */					\
	dst_sizes=oap->oa_dest->dt_dimset;				\
	dst_sizes.ds_dimension[i]=1;					\
	sprintf(tmp_dst_name,"dsb.%s",oap->oa_dest->dt_name);		\
	sub_dst_dp=make_subsamp(tmp_dst_name,oap->oa_dest,&dst_sizes,	\
						dst_offsets,dst_incrs);	\
									\
	src_sizes=src_dp->dt_dimset;					\
	src_sizes.ds_dimension[i]=1;					\
	sprintf(tmp_src_name,"ssb.%s",src_dp->dt_name);			\
	sub_src_dp=make_subsamp(tmp_src_name,src_dp,&src_sizes,		\
					src_offsets,src_incrs);		\
									\
	/* oa_3?? this code was from binop... */			\
	_oa.oa_dest = _oa.oa_3 = sub_dst_dp;				\
	_oa.oa_1 = sub_src_dp;						\
	_oa.oa_bmap = small_bmap_dp;					\
	_oa.oa_s1 = oap->oa_s1;						\
	_oa.oa_s2 = NO_OBJ;						\
									\
	for(j=0;j<src_dp->dt_dimension[i];j++){				\
		char *cp;						\
									\
/* now the objects all have matching sizes in THIS dimension! */	\
		outer_vvs(&_oa,code);					\
									\
		/* now step the offset... */				\
		ADVANCE_DATA_PTR(sub_dst_dp,oap->oa_dest)		\
		ADVANCE_DATA_PTR(sub_src_dp,src_dp)			\
	}								\
	delvec(sub_dst_dp);						\
	delvec(sub_src_dp);



/* If the bitmap is larger, we have to make subobjects for the dest and bitmap */

#define SUB_ITERATE_BMAP2( bmap_dp, small_src_dp )			\
									\
	Dimension_Set dst_sizes;					\
	Dimension_Set bmap_sizes;					\
	doff_t bmap_offsets[N_DIMENSIONS]={0,0,0,0,0};			\
	doff_t dst_offsets[N_DIMENSIONS]={0,0,0,0,0};			\
	incr_t bmap_incrs[N_DIMENSIONS]={1,1,1,1,1};			\
	incr_t dst_incrs[N_DIMENSIONS]={1,1,1,1,1};			\
	Data_Obj *sub_dst_dp, *sub_bmap_dp;				\
	char tmp_dst_name[LLEN];					\
	char tmp_bmap_name[LLEN];					\
	Vec_Obj_Args _oa;						\
	dimension_t bm_n;						\
									\
	/* iterate over bmap_dp */					\
	dst_sizes=oap->oa_dest->dt_dimset;				\
	dst_sizes.ds_dimension[i]=1;					\
	sprintf(tmp_dst_name,"dsb.%s",oap->oa_dest->dt_name);		\
	sub_dst_dp=make_subsamp(tmp_dst_name,oap->oa_dest,&dst_sizes,dst_offsets,dst_incrs);	\
									\
	bmap_sizes=bmap_dp->dt_dimset;					\
	bmap_sizes.ds_dimension[i]=1;					\
									\
	/* if( i != 1 )							\
bmap_sizes.ds_dimension[1] = N_BITMAP_COLS(&bmap_dp->dt_shape);*/	\
									\
	sprintf(tmp_bmap_name,"ssb.%s",bmap_dp->dt_name);		\
	sub_bmap_dp=make_subsamp(tmp_bmap_name,bmap_dp,&bmap_sizes,	\
					bmap_offsets,bmap_incrs);	\
									\
	/* oa_3?? this code was from binop... */			\
	_oa.oa_dest = _oa.oa_2 = sub_dst_dp;				\
	_oa.oa_bmap = sub_bmap_dp;					\
	_oa.oa_1 = small_src_dp;					\
	_oa.oa_s1 = oap->oa_s1;						\
	_oa.oa_4 = _oa.oa_3 = _oa.oa_s2 = NO_OBJ;			\
									\
	bm_n = bmap_dp->dt_dimension[i];				\
									\
	for(j=0;j<bm_n;j++){						\
		char *cp;						\
									\
/* now the objects all have matching sizes in THIS dimension! */	\
		outer_vvs(&_oa,code);					\
									\
		/* now step the offset... */				\
		ADVANCE_DATA_PTR(sub_dst_dp,oap->oa_dest)		\
		ADVANCE_BITMAP_PTR(sub_bmap_dp,bmap_dp)			\
	}								\
	delvec(sub_dst_dp);						\
	delvec(sub_bmap_dp);


void outer_vvs(QSP_ARG_DECL  Vec_Obj_Args *oap,Vec_Func_Code code)
{
	int i,j;
	dimension_t d1;

	/* find the first dimension we have to iterate over */

	for(i=0;i<N_DIMENSIONS;i++){
		d1 = oap->oa_bmap->dt_dimension[i];

		if( d1 == 1 && oap->oa_1->dt_dimension[i] > 1 ){
			/* bitmap is smaller */

			SUB_ITERATE_BMAP1(oap->oa_1,oap->oa_bmap)
			return;	/* all the work done in recursion */
		} else if( oap->oa_1->dt_dimension[i] == 1 && d1 > 1 ){
			/* bitmap is larger */
			SUB_ITERATE_BMAP2(oap->oa_bmap,oap->oa_1)
			return;	/* all the work done in recursion */
		}
		/* if we are here, the dimensions should match */
		if( d1 != oap->oa_1->dt_dimension[i] ){
			sprintf(error_string,
	"outer_vvs:  %s count mismatch, objects %s (%ld) and %s (%ld)",
				dimension_name[i],oap->oa_bmap->dt_name,
				oap->oa_bmap->dt_dimension[i],
				oap->oa_1->dt_name,oap->oa_1->dt_dimension[i]);
			NWARN(error_string);
			return;
		}
		/* now we know this dimension matches */
	}
	/* now we know ALL dimensions match.
	 * Call the appropriate vector-vector op.
	 */

	perf_vfunc(QSP_ARG  code,oap);
}

#define SUB_ITERATE_BMD( src_dp, small_src_dp, reverse_order )		\
									\
	Dimension_Set dst_sizes;					\
	Dimension_Set src_sizes;					\
	doff_t src_offsets[N_DIMENSIONS]={0,0,0,0,0};			\
	doff_t dst_offsets[N_DIMENSIONS]={0,0,0,0,0};			\
	incr_t src_incrs[N_DIMENSIONS]={1,1,1,1,1};			\
	incr_t dst_incrs[N_DIMENSIONS]={1,1,1,1,1};			\
	Data_Obj *sub_dst_dp, *sub_src_dp;				\
	char tmp_dst_name[LLEN];					\
	char tmp_src_name[LLEN];					\
	Vec_Obj_Args _oa;						\
									\
	/* iterate over src_dp */					\
	dst_sizes=oap->oa_bmap->dt_dimset;				\
	dst_sizes.ds_dimension[i]=1;					\
	sprintf(tmp_dst_name,"dsb.%s",oap->oa_dest->dt_name);		\
	sub_dst_dp=make_subsamp(tmp_dst_name,oap->oa_bmap,		\
				&dst_sizes,dst_offsets,dst_incrs);	\
									\
	src_sizes=src_dp->dt_dimset;					\
	src_sizes.ds_dimension[i]=1;					\
	sprintf(tmp_src_name,"ssb.%s",src_dp->dt_name);			\
	sub_src_dp=make_subsamp(tmp_src_name,src_dp,&src_sizes,		\
					src_offsets,src_incrs);		\
									\
	_oa.oa_4 = _oa.oa_3 = _oa.oa_dest = _oa.oa_bmap = sub_dst_dp;	\
	if( reverse_order ){						\
		_oa.oa_1 = sub_src_dp;					\
		_oa.oa_2 = small_src_dp;				\
	} else {							\
		_oa.oa_2 = sub_src_dp;					\
		_oa.oa_1 = small_src_dp;				\
	}								\
	_oa.oa_s1 = _oa.oa_s2 = NO_OBJ;					\
									\
	for(j=0;j<src_dp->dt_dimension[i];j++){				\
		char *cp;						\
									\
/* now the objects all have matching sizes in THIS dimension! */	\
		outer_vv_bitmap(&_oa,code);				\
									\
		/* now step the offset... */				\
		ADVANCE_BITMAP_PTR(sub_dst_dp,oap->oa_dest)		\
		ADVANCE_DATA_PTR(sub_src_dp,src_dp)			\
	}								\
	delvec(sub_dst_dp);						\
	delvec(sub_src_dp);


void outer_vv_bitmap(QSP_ARG_DECL  Vec_Obj_Args *oap,Vec_Func_Code code)
{
	int i,j;
	dimension_t d1,d2;

	/* two args of different sizes, bitmap result */

	/* find the first dimension we have to iterate over */

	for(i=0;i<N_DIMENSIONS;i++){
		d1 = oap->oa_1->dt_dimension[i];
		d2 = oap->oa_2->dt_dimension[i];

		if( d1 == 1 && d2 > 1 ){
			/* first arg is smaller */
			SUB_ITERATE_BMD(oap->oa_2,oap->oa_1,0)
			return;	/* all the work done in recursion */
		} else if( d2 == 1 && d1 > 1 ){
			/* first arg is larger */
			SUB_ITERATE_BMD(oap->oa_1,oap->oa_2,1)
			return;	/* all the work done in recursion */
		}
		/* if we are here, the dimensions should match */
		if( d1 != oap->oa_1->dt_dimension[i] ){
			sprintf(error_string,
	"outer_vvs:  %s count mismatch, objects %s (%ld) and %s (%ld)",
				dimension_name[i],oap->oa_bmap->dt_name,
				oap->oa_bmap->dt_dimension[i],
				oap->oa_1->dt_name,oap->oa_1->dt_dimension[i]);
			NWARN(error_string);
			return;
		}
		/* now we know this dimension matches */
	}
	/* now we know ALL dimensions match.
	 * Call the appropriate vector-vector op.
	 */

	perf_vfunc(QSP_ARG  code,oap);
}

void outer_vs_bitmap(QSP_ARG_DECL  Vec_Obj_Args *oap,Vec_Func_Code code)
{
	int i,j;
	dimension_t d1;

sprintf(error_string,"outer_vs_bitmap:  func = %s",vec_func_tbl[code].vf_name);
advise(error_string);
advise("arg1");
LONGLIST(oap->oa_1);
advise("bmap");
LONGLIST(oap->oa_bmap);
advise("scalar");
LONGLIST(oap->oa_sdp[0]);

	/* two args of different sizes, bitmap result */

	/* find the first dimension we have to iterate over */

	for(i=0;i<N_DIMENSIONS;i++){
#ifdef FOOBAR
		if( i==1 )
			d1 = N_BITMAP_COLS(&oap->oa_bmap->dt_shape);
		else
			d1 = oap->oa_bmap->dt_dimension[i];
#endif /* FOOBAR */

advise("arg1");
LONGLIST(oap->oa_1);
sprintf(error_string,"i=%d, d1 = %ld",i,d1);
advise(error_string);
		if( d1 == 1 && oap->oa_1->dt_dimension[i] > 1 ){
			/* bitmap is smaller */
			SUB_ITERATE_BMAP1(oap->oa_1,oap->oa_bmap)
			return;	/* all the work done in recursion */
		} else if( oap->oa_1->dt_dimension[i] == 1 && d1 > 1 ){
			/* bitmap is larger */
			SUB_ITERATE_BMAP2(oap->oa_bmap,oap->oa_1)
			return;	/* all the work done in recursion */
		}
		/* if we are here, the dimensions should match */
		if( d1 != oap->oa_1->dt_dimension[i] ){
			sprintf(error_string,
	"outer_vvs:  %s count mismatch, objects %s (%ld) and %s (%ld)",
				dimension_name[i],oap->oa_bmap->dt_name,
				oap->oa_bmap->dt_dimension[i],
				oap->oa_1->dt_name,oap->oa_1->dt_dimension[i]);
			NWARN(error_string);
			return;
		}
		/* now we know this dimension matches */
	}
	/* now we know ALL dimensions match.
	 * Call the appropriate vector-vector op.
	 */

	perf_vfunc(QSP_ARG  code,oap);
}

/* unary operation(s) - vmov and ??? */

/* We can have a target with higher dimensionality,
 * but we can't have the source having higher dimensionality...
 */

#define USUB_ITERATE( src_dp, dst_dp )					\
									\
	Dimension_Set dst_sizes;					\
	doff_t dst_offsets[N_DIMENSIONS]={0,0,0,0,0};			\
	incr_t dst_incrs[N_DIMENSIONS]={1,1,1,1,1};			\
	Data_Obj *sub_dst_dp;						\
	char tmp_dst_name[LLEN];					\
	Vec_Obj_Args _oa;						\
									\
	/* iterate over dst_dp */					\
	dst_sizes=oap->oa_dest->dt_dimset;				\
	dst_sizes.ds_dimension[i]=1;					\
	sprintf(tmp_dst_name,"dsb.%s",oap->oa_dest->dt_name);		\
	sub_dst_dp=make_subsamp(tmp_dst_name,oap->oa_dest,&dst_sizes,dst_offsets,dst_incrs);	\
									\
	_oa.oa_dest = _oa.oa_2 = sub_dst_dp;				\
	_oa.oa_1 = src_dp;						\
	_oa.oa_3 = _oa.oa_bmap = _oa.oa_s1 = _oa.oa_s2 = NO_OBJ;	\
									\
	for(j=0;j<dst_dp->dt_dimension[i];j++){				\
		char *cp;						\
									\
		/* now the objects all have matching sizes in THIS dimension! */	\
		outer_unop(&_oa,code);					\
									\
		/* now step the offset... */				\
		ADVANCE_DATA_PTR(sub_dst_dp,oap->oa_dest)		\
	}								\
	delvec(sub_dst_dp);

void outer_unop(QSP_ARG_DECL  Vec_Obj_Args *oap,Vec_Func_Code code)
{
	int i,j;

	/* find the first dimension we have to iterate over */

	for(i=0;i<N_DIMENSIONS;i++){
		if( oap->oa_1->dt_dimension[i] == 1 &&
				oap->oa_2->dt_dimension[i] > 1 ){
			USUB_ITERATE(oap->oa_1,oap->oa_2)
			return;	/* all the work done in recursion */
		}
		/* if we are here, the dimensions should match */
		if( oap->oa_1->dt_dimension[i] != oap->oa_2->dt_dimension[i] ){
			sprintf(error_string,
	"outer_unop:  %s count mismatch, objects %s (%ld) and %s (%ld)",
				dimension_name[i],oap->oa_1->dt_name,
				oap->oa_1->dt_dimension[i],
				oap->oa_2->dt_name,oap->oa_2->dt_dimension[i]);
			NWARN(error_string);
			return;
		}
		/* now we know this dimension matches */
	}
	/* now we know ALL dimensions match.
	 * Call the appropriate vector-vector op.
	 */

	perf_vfunc(QSP_ARG  code,oap);
}

