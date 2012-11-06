#include "quip_config.h"


char VersionId_vec_util_median[] = QUIP_VERSION_STRING;

/* median filter */

#include <stdlib.h>
#include "data_obj.h"
#include "vec_util.h"

/* local prototypes */


static void uby_median(Data_Obj *,Data_Obj *);
static int pix_comp(const void *,const void *);
static int fpix_comp(const void *,const void *);
static void median_of_pix(u_char *,u_char *);
static void median_clip_pix(u_char *,u_char *);
static void process_median(QSP_ARG_DECL  Data_Obj *,Data_Obj *);
static void setup_pix_arr(int, int, int, int, u_char *src);
static void median_row(u_char *dst,u_char *src,u_long ncols);
static void median_col(u_char *dst,u_char *src,u_long ncols);

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

static Pixel pix_arr[N_NEIGHBORHOOD];
static int order[N_NEIGHBORHOOD];

void median(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	median_pix_func=median_of_pix;
	process_median(QSP_ARG  dpto,dpfr);
}

void median_clip(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	median_pix_func=median_clip_pix;
	process_median(QSP_ARG  dpto,dpfr);
}

static void process_median(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr)
{
	if( dpto == dpfr ){
		WARN("target and source must be distinct for median filter");
		return;
		/* BUG this will not catch subobjects that share data, so be careful! */
	}
	if( dpto->dt_prec != dpfr->dt_prec ){
		sprintf(error_string,
			"%s and %s must have same precision for median filter",
			dpto->dt_name,dpfr->dt_name);
		WARN(error_string);
		return;
	}
	if( !IS_CONTIGUOUS(dpto) || !IS_CONTIGUOUS(dpfr) ){
		sprintf(error_string,
			"%s and %s must be contiguous for median filter",
			dpto->dt_name,dpfr->dt_name);
		WARN(error_string);
		return;
	}
	if( !dp_same_prec(QSP_ARG  dpto,dpfr,"process_median") ){
		WARN("source and target must have same precision for median filter");
		return;
	}
	switch(dpto->dt_prec){
		/* case PREC_SP: sp_median(dpto,dpfr); break; */
		case PREC_UBY: uby_median(dpto,dpfr); break;
		default:
			sprintf(error_string,
			"sorry, %s precision not supported by median filter",
				prec_name[dpto->dt_prec]);
			WARN(error_string);
			advise("supported precisions:");
			sprintf(error_string,"\t%s",name_for_prec(PREC_UBY));
			advise(error_string);
			break;
	}
}

static int pix_comp(CONST void *ip1,CONST void *ip2) /* args are pointers into the order array */
{
	if( pix_arr[*(CONST int *)ip1].pval.ucv >
		pix_arr[*(CONST int *)ip2].pval.ucv ) return(1);
	else if( pix_arr[*(CONST int *)ip1].pval.ucv <
		pix_arr[*(CONST int *)ip2].pval.ucv ) return(-1);
	else return(0);
}

static int fpix_comp(CONST void *ip1,CONST void *ip2) /* args are pointers into the order array */
{
	if( pix_arr[*(CONST int *)ip1].pval.fv >
		pix_arr[*(CONST int *)ip2].pval.fv ) return(1);
	else if( pix_arr[*(CONST int *)ip1].pval.fv <
		pix_arr[*(CONST int *)ip2].pval.fv ) return(-1);
	else return(0);
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

/* BUG this subroutine works over 3x3 neighborhoods, but we'd like the size to be a parameter...
 */

static void uby_median(Data_Obj *dpto,Data_Obj *dpfr)
{
	u_long i;
	u_char *dst,*src;

	s_rinc = dpfr->dt_rinc;
	d_rinc = dpto->dt_rinc;

	/* The trick here is that we want to remember what we already know
	 * as we scan accross the image.  We need to sort the pixel values for
	 * the nine pixels.  Then we will remove 3 and add 3 pixels.  We
	 * would like to preserve the work we have already done sorting...
	 * One way to do this is to maintain two arrays, one of the pixel
	 * values, and another of the corresponding indices.
	 */

	/* Do the easy part first */

	dst=(u_char *)dpto->dt_data;
	src=(u_char *)dpfr->dt_data;
	for(i=1;i<(dpto->dt_rows-1);i++){
		dst += d_rinc;
		src += s_rinc;
		setup_pix_arr(-1,3,-1,3,src+1);
		median_row(dst+1,src+1,dpto->dt_cols);
	}


	/* do the top and botton rows */

	dst=(u_char *)dpto->dt_data;
	src=(u_char *)dpfr->dt_data;
	src++;
	dst++;
	setup_pix_arr(-1,3,0,2,src);
	median_row(dst,src,dpto->dt_cols);


	dst=(u_char *)dpto->dt_data;
	src=(u_char *)dpfr->dt_data;
	dst += 1 + (dpto->dt_rows-1)*d_rinc;
	src += 1 + (dpfr->dt_rows-1)*s_rinc;
	setup_pix_arr(-1,3,-1,2,src);
	median_row(dst,src,dpto->dt_cols);

	/* do the left column */

	dst=(u_char *)dpto->dt_data;
	src=(u_char *)dpfr->dt_data;
	dst += d_rinc;
	src += s_rinc;
	setup_pix_arr(0,2,-1,3,src);
	median_col(dst,src,dpto->dt_cols);

	/* do the right column */

	dst=(u_char *)dpto->dt_data;
	src=(u_char *)dpfr->dt_data;
	dst += 2*d_rinc-1;
	src += 2*s_rinc-1;
	setup_pix_arr(-1,2,-1,3,src);
	median_col(dst,src,dpto->dt_cols);


	/* what about the corners? */

	/* upper left */

	dst=(u_char *)dpto->dt_data;
	src=(u_char *)dpfr->dt_data;
	setup_pix_arr(0,2,0,2,src);
	(*median_pix_func)(dst,src);

	/* upper right */

	dst=(u_char *)dpto->dt_data;
	src=(u_char *)dpfr->dt_data;
	dst += d_rinc-1;
	src += s_rinc-1;
	setup_pix_arr(-1,2,0,2,src);
	(*median_pix_func)(dst,src);

	/* lower left */

	dst=(u_char *)dpto->dt_data;
	src=(u_char *)dpfr->dt_data;
	dst += (dpto->dt_rows-1)*d_rinc;
	src += (dpfr->dt_rows-1)*s_rinc;
	setup_pix_arr(0,2,-1,2,src);
	(*median_pix_func)(dst,src);

	/* lower right */

	dst=(u_char *)dpto->dt_data;
	src=(u_char *)dpfr->dt_data;
	dst += (dpto->dt_rows-1)*d_rinc;
	src += (dpfr->dt_rows-1)*s_rinc;
	dst += d_rinc-1;
	src += s_rinc-1;
	setup_pix_arr(-1,2,-1,2,src);
	(*median_pix_func)(dst,src);
}


/* one dimensional median filter */
/* This may be useful for saccade elimination! */

void median_1D(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,int median_radius)
{
	dimension_t i,j;
	int k;
	float *dst,*src;

	/* Do the easy part first */

	dst=(float *)dpto->dt_data;
	src=(float *)dpfr->dt_data;

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
	for(j=median_radius;j<(dpto->dt_cols-median_radius);){
		/* sort the pixel array */
		qsort(order,(size_t)n_neighborhood,sizeof(order[0]),fpix_comp);
		/* output the target pixel! */
		*dst = pix_arr[ order[median_radius] ].pval.fv;

		src++;
		dst++;
		j++;

		/* now move everything over one */

		if( j<(dpto->dt_cols-median_radius) )
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
		dst=(float *)dpto->dt_data;
		src=(float *)dpfr->dt_data;
		k=0;
		for(j=0;j<=i+median_radius;j++){
			pix_arr[k].pval.fv = *(src+j);
			order[k] = k;
			k++;
		}
		n_neighborhood=k;

		src+=i;
		dst+=i;

		/* sort the pixel array */
		qsort(order,(size_t)n_neighborhood,sizeof(order[0]),fpix_comp);
		/* output the target pixel! */
		*dst = pix_arr[ order[median_radius] ].pval.fv;

		dst=(float *)dpto->dt_data;
		src=(float *)dpfr->dt_data;
		dst+=dpto->dt_cols-1;		/* point to last one */
		src+=dpto->dt_cols-1;

		k=0;
		for(j=0;j<=i+median_radius;j++){
			pix_arr[k].pval.fv = *(src-j);
			order[k] = k;
			k++;
		}
		n_neighborhood=k;

		src-=i;
		dst-=i;

		/* sort the pixel array */
		qsort(order,(size_t)n_neighborhood,sizeof(order[0]),fpix_comp);
		/* output the target pixel! */
		*dst = pix_arr[ order[median_radius] ].pval.fv;
	}
}

static int comp_dbl_pix(CONST void *ptr1,CONST void *ptr2) /* args are pointers into the order array */
{
	if( *((CONST double *)ptr1) > *((CONST double *)ptr2) ) return(1);
	else if( *((CONST double *)ptr1) < *((CONST double *)ptr2)  ) return(-1);
	else return(0);
}

static int comp_flt_pix(CONST void *ptr1,CONST void *ptr2) /* args are pointers into the order array */
{
	if( *((CONST float *)ptr1) > *((CONST float *)ptr2) ) return(1);
	else if( *((CONST float *)ptr1) < *((CONST float *)ptr2)  ) return(-1);
	else return(0);
}

/* This function just sorts the pixels in-place.  This is useful for determining
 * the median value of an array, we can sort in place and then sample the middle value.
 */

void sort_data(QSP_ARG_DECL  Data_Obj *dp)
{
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(error_string,"sort_data:  object %s must be contiguous",dp->dt_name);
		WARN(error_string);
		return;
	}
	if( IS_COMPLEX(dp) ){
		sprintf(error_string,"sort_data:  Sorry, can't sort complex object %s",dp->dt_name);
		WARN(error_string);
		return;
	}
	switch( MACHINE_PREC(dp) ){
		case PREC_SP:
			qsort(dp->dt_data,(size_t)dp->dt_n_mach_elts,siztbl[MACHINE_PREC(dp)],comp_flt_pix);
			break;
		case PREC_DP:
			qsort(dp->dt_data,(size_t)dp->dt_n_mach_elts,siztbl[MACHINE_PREC(dp)],comp_dbl_pix);
			break;
		default:
			sprintf(error_string,"sort_data:  Sorry, %s precision not supported",
				prec_name[MACHINE_PREC(dp)]);
			WARN(error_string);
			break;
	}
}

static Data_Obj *index_sort_data_dp;

static int comp_indexed_flt_pix(CONST void *ptr1,CONST void *ptr2) /* args are pointers into the order array */
{
	u_long i1, i2, inc;
	float *p1, *p2;

	i1 = *((CONST u_long *)ptr1);
	i2 = *((CONST u_long *)ptr2); 

	inc = index_sort_data_dp->dt_type_inc[ index_sort_data_dp->dt_mindim ];
	p1 = ((float *)index_sort_data_dp->dt_data) + i1*inc;
	p2 = ((float *)index_sort_data_dp->dt_data) + i2*inc;

	if( *p1 > *p2 ) return(1);
	else if( *p1 < *p2 ) return(-1);
	else return(0);
}

static int comp_indexed_dbl_pix(CONST void *ptr1,CONST void *ptr2) /* args are pointers into the order array */
{
	u_long i1, i2, inc;
	double *p1, *p2;

	i1 = *((CONST u_long *)ptr1);
	i2 = *((CONST u_long *)ptr2); 

	inc = index_sort_data_dp->dt_type_inc[ index_sort_data_dp->dt_mindim ];
	p1 = ((double *)index_sort_data_dp->dt_data) + i1*inc;
	p2 = ((double *)index_sort_data_dp->dt_data) + i2*inc;

	if( *p1 > *p2 ) return(1);
	else if( *p1 < *p2 ) return(-1);
	else return(0);
}

void sort_indices(QSP_ARG_DECL  Data_Obj *index_dp,Data_Obj *data_dp)
{
	if( ! IS_CONTIGUOUS(index_dp) ){
		sprintf(error_string,"sort_data:  object %s must be contiguous",index_dp->dt_name);
		WARN(error_string);
		return;
	}
	if( MACHINE_PREC(index_dp) != PREC_UDI && MACHINE_PREC(index_dp) != PREC_DI ){
		sprintf(error_string,"sort_indices:  index array %s (%s) should have %s or %s precision",
			index_dp->dt_name,name_for_prec(index_dp->dt_prec),
			name_for_prec(PREC_UDI),name_for_prec(PREC_DI));
		WARN(error_string);
		return;
	}
	if( MACHINE_PREC(data_dp) != PREC_SP && MACHINE_PREC(data_dp) != PREC_DP ){
		WARN("Sorry, right now only know how to do index sort for float/double data");
		sprintf(error_string,"Data array %s has %s precision",data_dp->dt_name,
			name_for_prec(data_dp->dt_prec));
		advise(error_string);
		return;
	}
	/* We index the data using min_dim - if that is not the only dimension, then
	 * we print a warning, but sort anyway.
	 */
	if( data_dp->dt_n_type_elts != data_dp->dt_type_dim[ data_dp->dt_mindim ] ){
		sprintf(error_string,"sort_indices:  only sorting first %s of object %s",
			dimension_name[ data_dp->dt_mindim ], data_dp->dt_name );
		WARN(error_string);
	}

	/* make sure the index of the array matches what we are sorting */
	if( index_dp->dt_n_type_elts != data_dp->dt_type_dim[ data_dp->dt_mindim ] ){
		sprintf(error_string,
	"sort_indices:  size of index array %s (%d) does not match %s dimension of object %s (%d)",
			index_dp->dt_name,index_dp->dt_n_type_elts,dimension_name[data_dp->dt_mindim],
			data_dp->dt_name,data_dp->dt_type_dim[ data_dp->dt_mindim ] );
		WARN(error_string);
		return;
	}


	/* BUG perhaps should check that tdim matches # dimensions of data_dp? */

	index_sort_data_dp = data_dp;

	switch( MACHINE_PREC(data_dp) ){
		case PREC_SP:
			qsort(index_dp->dt_data,
				(size_t)index_dp->dt_n_mach_elts,siztbl[PREC_DI],
				comp_indexed_flt_pix);
			break;
		case PREC_DP:
			qsort(index_dp->dt_data,
				(size_t)index_dp->dt_n_mach_elts,siztbl[PREC_DI],
				comp_indexed_dbl_pix);
			break;
#ifdef CAUTIOUS
		default:
			WARN("CAUTIOUS:  sort_indices:  unexpected precision");
			break;
#endif /* CAUTIOUS */
	}
}

