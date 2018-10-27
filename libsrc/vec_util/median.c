#include "quip_config.h"

/* median filter */

#include <stdlib.h>
#include "vec_util.h"
#include "quip_prot.h"

// BUG static vars are not thread-safe
static void (*median_pix_func)(u_char *,u_char *);
static int n_neighborhood;
static int median_index;
static long s_rinc,d_rinc;


#define DELTA		3
#define N_NEIGHBORHOOD	9	/* DELTA*DELTA */
#define MEDIAN_INDEX	4	/* floor(N_NEIGHBORHOOD/2) */

typedef struct {
	int	ival;
	int	jval;
	union {
		u_char	ucv;
		float	fv;
	} pval;
} Pixel;

// BUG static vars are not thread-safe
static Pixel pix_arr[N_NEIGHBORHOOD];
static int order[N_NEIGHBORHOOD];

static int pix_comp(const void *ip1,const void *ip2) /* args are pointers into the order array */
{
	if( pix_arr[*(const int *)ip1].pval.ucv >
		pix_arr[*(const int *)ip2].pval.ucv ) return(1);
	else if( pix_arr[*(const int *)ip1].pval.ucv <
		pix_arr[*(const int *)ip2].pval.ucv ) return(-1);
	else return(0);
}

static void median_of_pix(u_char *dst,u_char *src)
{
	/* sort the pixel array */
	qsort(order,(size_t)n_neighborhood,sizeof(order[0]),pix_comp);
	/* output the target pixel! */
	*dst = pix_arr[ order[median_index] ].pval.ucv;
}

static void setup_pix_arr(int start_col,int ncols,int start_row,int nrows,u_char *src)
{
	int i,j,k;

	k=0;
	for(i=0;i<nrows;i++){
		for(j=0;j<ncols;j++){
			int ii,jj;

			ii = start_row + k/ncols;		/* relative row index */
			jj = start_col + k%ncols;		/* relative column index */
			pix_arr[k].pval.ucv = *(src+jj+ii*s_rinc);
			pix_arr[k].ival = ii;
			pix_arr[k].jval = jj;
			order[k] = k;
			k++;
		}
	}
	n_neighborhood=k;
	median_index = (n_neighborhood-1) / 2;
}

static void median_row(u_char *dst,u_char *src,u_long ncols)
{
	dimension_t j;
	int k;

	for(j=1;j<(ncols-1);){
		(*median_pix_func)(dst,src);

		src++;
		dst++;
		j++;

		/* now move everything over one */

		if( j<(ncols-1) )
			for(k=0;k<n_neighborhood;k++){
				pix_arr[k].jval -= 1;
				if( pix_arr[k].jval == (-2) ){
					pix_arr[k].jval = 1;
					pix_arr[k].pval.ucv = *(src+1+pix_arr[k].ival*s_rinc);
				}
			}
	}
}

static void median_col(u_char *dst,u_char *src,u_long nrows)
{
	u_long i;
	int k;

	for(i=1;i<(nrows-1);){
		(*median_pix_func)(dst,src);

		src += s_rinc;
		dst += d_rinc;
		i++;

		/* now move everything over one */

		if( i<(nrows-1) )
			for(k=0;k<n_neighborhood;k++){
				pix_arr[k].ival -= 1;
				if( pix_arr[k].ival == (-2) ){
					pix_arr[k].ival = 1;
					pix_arr[k].pval.ucv = *(src+s_rinc+pix_arr[k].jval);
				}
			}
	}
}


/* BUG this subroutine works over 3x3 neighborhoods,
 * but we'd like the size to be a parameter...
 */

static void uby_median(Data_Obj *dpto,Data_Obj *dpfr)
{
	u_long i;
	u_char *dst,*src;

	s_rinc = OBJ_ROW_INC(dpfr);
	d_rinc = OBJ_ROW_INC(dpto);

	/* The trick here is that we want to remember what we already know
	 * as we scan accross the image.  We need to sort the pixel values for
	 * the nine pixels.  Then we will remove 3 and add 3 pixels.  We
	 * would like to preserve the work we have already done sorting...
	 * One way to do this is to maintain two arrays, one of the pixel
	 * values, and another of the corresponding indices.
	 */

	/* Do the easy part first */

	dst=(u_char *)OBJ_DATA_PTR(dpto);
	src=(u_char *)OBJ_DATA_PTR(dpfr);
	for(i=1;i<(OBJ_ROWS(dpto)-1);i++){
		dst += d_rinc;
		src += s_rinc;
		setup_pix_arr(-1,3,-1,3,src+1);
		median_row(dst+1,src+1,OBJ_COLS(dpto));
	}


	/* do the top and botton rows */

	dst=(u_char *)OBJ_DATA_PTR(dpto);
	src=(u_char *)OBJ_DATA_PTR(dpfr);
	src++;
	dst++;
	setup_pix_arr(-1,3,0,2,src);
	median_row(dst,src,OBJ_COLS(dpto));


	dst=(u_char *)OBJ_DATA_PTR(dpto);
	src=(u_char *)OBJ_DATA_PTR(dpfr);
	dst += 1 + (OBJ_ROWS(dpto)-1)*d_rinc;
	src += 1 + (OBJ_ROWS(dpfr)-1)*s_rinc;
	setup_pix_arr(-1,3,-1,2,src);
	median_row(dst,src,OBJ_COLS(dpto));

	/* do the left column */

	dst=(u_char *)OBJ_DATA_PTR(dpto);
	src=(u_char *)OBJ_DATA_PTR(dpfr);
	dst += d_rinc;
	src += s_rinc;
	setup_pix_arr(0,2,-1,3,src);
	median_col(dst,src,OBJ_COLS(dpto));

	/* do the right column */

	dst=(u_char *)OBJ_DATA_PTR(dpto);
	src=(u_char *)OBJ_DATA_PTR(dpfr);
	dst += 2*d_rinc-1;
	src += 2*s_rinc-1;
	setup_pix_arr(-1,2,-1,3,src);
	median_col(dst,src,OBJ_COLS(dpto));


	/* what about the corners? */

	/* upper left */

	dst=(u_char *)OBJ_DATA_PTR(dpto);
	src=(u_char *)OBJ_DATA_PTR(dpfr);
	setup_pix_arr(0,2,0,2,src);
	(*median_pix_func)(dst,src);

	/* upper right */

	dst=(u_char *)OBJ_DATA_PTR(dpto);
	src=(u_char *)OBJ_DATA_PTR(dpfr);
	dst += d_rinc-1;
	src += s_rinc-1;
	setup_pix_arr(-1,2,0,2,src);
	(*median_pix_func)(dst,src);

	/* lower left */

	dst=(u_char *)OBJ_DATA_PTR(dpto);
	src=(u_char *)OBJ_DATA_PTR(dpfr);
	dst += (OBJ_ROWS(dpto)-1)*d_rinc;
	src += (OBJ_ROWS(dpfr)-1)*s_rinc;
	setup_pix_arr(0,2,-1,2,src);
	(*median_pix_func)(dst,src);

	/* lower right */

	dst=(u_char *)OBJ_DATA_PTR(dpto);
	src=(u_char *)OBJ_DATA_PTR(dpfr);
	dst += (OBJ_ROWS(dpto)-1)*d_rinc;
	src += (OBJ_ROWS(dpfr)-1)*s_rinc;
	dst += d_rinc-1;
	src += s_rinc-1;
	setup_pix_arr(-1,2,-1,2,src);
	(*median_pix_func)(dst,src);
}

static void process_median(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	INSIST_RAM_OBJ(dpto,process_median)
	INSIST_RAM_OBJ(dpfr,process_median)

	if( dpto == dpfr ){
		WARN("target and source must be distinct for median filter");
		return;
		/* BUG this will not catch subobjects that share data, so be careful! */
	}
	if( !IS_CONTIGUOUS(dpto) || !IS_CONTIGUOUS(dpfr) ){
		sprintf(ERROR_STRING,
			"%s and %s must be contiguous for median filter",
			OBJ_NAME(dpto),OBJ_NAME(dpfr));
		WARN(ERROR_STRING);
		return;
	}
	if( !dp_same_prec(dpto,dpfr,"process_median") ){
		WARN("source and target must have same precision for median filter");
		return;
	}
	switch(OBJ_PREC(dpto)){
		// Why is this commented out???
		/* case PREC_SP: sp_median(dpto,dpfr); break; */
		case PREC_UBY: uby_median(dpto,dpfr); break;
		default:
			sprintf(ERROR_STRING,
			"sorry, %s precision not supported by median filter",
				PREC_NAME(OBJ_PREC_PTR(dpto)));
			WARN(ERROR_STRING);
			advise("supported precisions:");
			sprintf(ERROR_STRING,"\t%s",NAME_FOR_PREC_CODE(PREC_UBY));
			advise(ERROR_STRING);
			break;
	}
}

void _median(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	median_pix_func=median_of_pix;
	process_median(QSP_ARG  dpto,dpfr);
}

static int fpix_comp(const void *ip1,const void *ip2) /* args are pointers into the order array */
{
	if( pix_arr[*(const int *)ip1].pval.fv >
		pix_arr[*(const int *)ip2].pval.fv ) return(1);
	else if( pix_arr[*(const int *)ip1].pval.fv <
		pix_arr[*(const int *)ip2].pval.fv ) return(-1);
	else return(0);
}

static void median_clip_pix(u_char *dst,u_char *src)
{
	u_char v;

	/* sort the pixel array */

	/* BUG since the array is so small we could probably do something
	 * faster, sort on insertion of new items?  Do some
	 * profiling first...
	 */

	qsort(order,(size_t)n_neighborhood,sizeof(order[0]),pix_comp);
	/* output the target pixel! */
	if( *src >  (v = pix_arr[ order[median_index] ].pval.ucv) )
		*dst = v;
	else
		*dst = *src;
}

void _median_clip(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	median_pix_func=median_clip_pix;
	process_median(QSP_ARG  dpto,dpfr);
}


/* one dimensional median filter */
/* This may be useful for saccade elimination! */

void _median_1D(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,int median_radius)
{
	incr_t i,j;
	int k;
	float *dst,*src;

	INSIST_RAM_OBJ(dpto,median_1D)
	INSIST_RAM_OBJ(dpfr,median_1D)

	if( dpto == dpfr ){
		WARN("target and source must be distinct for median filter");
		return;
		/* BUG this will not catch subobjects that share data, so be careful! */
	}

	INSIST_OBJ_PREC(dpto,PREC_SP,median_1D)
	INSIST_OBJ_PREC(dpfr,PREC_SP,median_1D)

	if( ! dp_same_size(dpto,dpfr,"median_1D") ){
		WARN("median_1D:  object sizes must match!?");
		return;
	}

	/* Do the easy part first */

	dst=(float *)OBJ_DATA_PTR(dpto);
	src=(float *)OBJ_DATA_PTR(dpfr);

	k=0;
	for(j=0;j<=(dimension_t)(2*median_radius);j++){
		pix_arr[k].pval.fv = *(src+j);
		pix_arr[k].ival = 0;
		pix_arr[k].jval = j-median_radius;
		order[k] = k;
		k++;
	}
	n_neighborhood=k;

	src+=median_radius;
	dst+=median_radius;
	for(j=median_radius;j<(OBJ_COLS(dpto)-median_radius);){
		/* sort the pixel array */
		qsort(order,(size_t)n_neighborhood,sizeof(order[0]),fpix_comp);
		/* output the target pixel! */
		*dst = pix_arr[ order[median_radius] ].pval.fv;

		src++;
		dst++;
		j++;

		/* now move everything over one */

		if( j<(OBJ_COLS(dpto)-median_radius) )
			for(k=0;k<n_neighborhood;k++){
				pix_arr[k].jval -= 1;
				if( pix_arr[k].jval < (-median_radius) ){
					pix_arr[k].jval = median_radius;
					pix_arr[k].pval.fv =
						*(src+median_radius);
				}
			}
	}

	/* now work out way out to the edges */
	for(i=median_radius-1;i>=0;i--){
		/* i is the index of the pixel we will do */
		dst=(float *)OBJ_DATA_PTR(dpto);
		src=(float *)OBJ_DATA_PTR(dpfr);
		k=0;
		for(j=0;j<=i+median_radius;j++){
			pix_arr[k].pval.fv = *(src+j);
			order[k] = k;
			k++;
		}
		n_neighborhood=k;

		//src+=i;
		dst+=i;

		/* sort the pixel array */
		qsort(order,(size_t)n_neighborhood,sizeof(order[0]),fpix_comp);
		/* output the target pixel! */
		*dst = pix_arr[ order[median_radius] ].pval.fv;

		dst=(float *)OBJ_DATA_PTR(dpto);
		src=(float *)OBJ_DATA_PTR(dpfr);
		dst+=OBJ_COLS(dpto)-1;		/* point to last one */
		src+=OBJ_COLS(dpto)-1;

		k=0;
		for(j=0;j<=i+median_radius;j++){
			pix_arr[k].pval.fv = *(src-j);
			order[k] = k;
			k++;
		}
		n_neighborhood=k;

		//src-=i;
		dst-=i;

		/* sort the pixel array */
		qsort(order,(size_t)n_neighborhood,sizeof(order[0]),fpix_comp);
		/* output the target pixel! */
		*dst = pix_arr[ order[median_radius] ].pval.fv;
	}
}

#define good_for_sorting(dp) _good_for_sorting(QSP_ARG  dp)

static int _good_for_sorting(QSP_ARG_DECL  Data_Obj *dp)
{
	if( IS_COMPLEX(dp) || IS_QUAT(dp) || IS_COLOR(dp) ){
		sprintf(ERROR_STRING,"sort_data:  Sorry, can't sort complex/quaternion/color object %s",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return 0;
	}
	return 1;
}

/* This function just sorts the pixels in-place.  This is useful for determining
 * the median value of an array, we can sort in place and then sample the middle value.
 */

void _sort_data(QSP_ARG_DECL  Data_Obj *dp)
{
	INSIST_RAM_OBJ(dp,sort_data)

	if( ! good_for_sorting(dp) )
		return;

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"sort_data:  object %s must be contiguous",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}
	if( IS_BITMAP(dp) ){
		sprintf(ERROR_STRING,"sort_data:  Sorry, can't sort bitmap object %s",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}
	// BUG it is also illegal to sort chars and strings, although we might like to be
	// able to sort string tables using strcmp!
	//
	// Because we use the machine precision below, chars and strings will sort as bytes.
	// This may cause a problem with null-terminated strings!

	qsort(OBJ_DATA_PTR(dp),(size_t)OBJ_N_MACH_ELTS(dp),OBJ_MACH_PREC_SIZE(dp),PREC_VAL_COMP_FUNC(OBJ_MACH_PREC_PTR(dp)));
}

#ifndef HAVE_QSORT_R
// Too bad this needs a global var...
// not thread safe unless moved to query_stack.
Data_Obj *index_sort_data_dp=NULL;
#endif // ! HAVE_QSORT_R


void _sort_indices(QSP_ARG_DECL  Data_Obj *index_dp,Data_Obj *data_dp)
{
	INSIST_RAM_OBJ(index_dp,sort_indices)
	INSIST_RAM_OBJ(data_dp,sort_indices)

	if( ! good_for_sorting(data_dp) )
		return;

	if( ! IS_CONTIGUOUS(index_dp) ){
		sprintf(ERROR_STRING,"sort_data:  object %s must be contiguous",OBJ_NAME(index_dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_MACH_PREC(index_dp) != PREC_UDI && OBJ_MACH_PREC(index_dp) != PREC_DI ){
		sprintf(ERROR_STRING,"sort_indices:  index array %s (%s) should have %s or %s precision",
			OBJ_NAME(index_dp),OBJ_PREC_NAME(index_dp),
			NAME_FOR_PREC_CODE(PREC_UDI),NAME_FOR_PREC_CODE(PREC_DI));
		WARN(ERROR_STRING);
		return;
	}

	/* We index the data using min_dim - if that is not the only dimension, then
	 * we print a warning, but sort anyway.
	 */
	if( OBJ_N_TYPE_ELTS(data_dp) != OBJ_TYPE_DIM(data_dp, OBJ_MINDIM(data_dp) ) ){
		sprintf(ERROR_STRING,"sort_indices:  only sorting first %s of object %s",
			dimension_name[ OBJ_MINDIM(data_dp) ], OBJ_NAME(data_dp) );
		WARN(ERROR_STRING);
	}

	/* make sure the index of the array matches what we are sorting */
	if( OBJ_N_TYPE_ELTS(index_dp) != OBJ_TYPE_DIM(data_dp, OBJ_MINDIM(data_dp) ) ){
		sprintf(ERROR_STRING,
	"sort_indices:  size of index array %s (%d) does not match %s dimension of object %s (%d)",
			OBJ_NAME(index_dp),OBJ_N_TYPE_ELTS(index_dp),dimension_name[OBJ_MINDIM(data_dp)],
			OBJ_NAME(data_dp),OBJ_TYPE_DIM(data_dp, OBJ_MINDIM(data_dp) ) );
		WARN(ERROR_STRING);
		return;
	}


	/* BUG perhaps should check that tdim matches # dimensions of data_dp? */

	SET_GLOBAL_THUNK_ARG			// only if no qsort_r

	// Should we make sure that we have qsort_r ???
	INDEX_SORT_FUNC(	OBJ_DATA_PTR(index_dp),
				(size_t)OBJ_N_MACH_ELTS(index_dp),
				SIZE_FOR_PREC_CODE( PREC_DI ),
				INDEX_SORT_THUNK_ARG
				PREC_IDX_COMP_FUNC(OBJ_MACH_PREC_PTR(data_dp)));

}

