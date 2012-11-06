
#include "quip_config.h"

char VersionId_nrmenu_numrec[] = QUIP_VERSION_STRING;

/* interface to numerical recipies code */

#include <stdio.h>
#include "data_obj.h"
/* #include "warproto.h" */
#include "nrm_api.h"
#include "data_obj.h"

#define MAX_DIM	0x10000		/* 64k */

#ifdef HAVE_NUMREC

#include "numrec.h"

/*

	The numerical recipes code does everything using
	fortran style indexing (1 - N instead of 0 - N-1).

		C style indices
		0,0	0,1	0,2	0,3
		1,0	1,1	1,2	1,3
		2,0	2,1	2,2	2,3

		Fortran style indices
		1,1	1,2	1,3	1,4
		2,1	2,2	2,3	2,4
		3,1	3,2	3,3	3,4

		C style row ptrs array
		base
		base+4
		base+8

		C/fortran row pointers 

		Don't care
		base-1
		(base+4)-1
		(base+8)-1

*/

void float_init_rowlist(float **list, Data_Obj *dp)
{
	unsigned i;
	float *fbase;

	fbase = ((float *)dp->dt_data);
	fbase --;				/* for numrec fortran indices */

	if( dp->dt_rows > MAX_DIM ){
		sprintf(DEFAULT_ERROR_STRING,"Sorry, object %s has %d rows but MAX_DIM is %d",
			dp->dt_name,dp->dt_rows,MAX_DIM);
		NERROR1(DEFAULT_ERROR_STRING);
	}

	for(i=0;i<dp->dt_rows;i++)
		*list++ = fbase + i*dp->dt_rowinc; /* ??? *dp->dt_pinc; */
}

void double_init_rowlist(double **list, Data_Obj *dp)
{
	unsigned i;
	double *fbase;

	fbase = ((double *)dp->dt_data);
	fbase --;				/* for numrec fortran indices */
	for(i=0;i<dp->dt_rows;i++)
		*list++ = fbase + i*dp->dt_rowinc; /* ??? *dp->dt_pinc; */
}

#ifdef FOOBAR
static void float_sort_svd_eigenvectors(Data_Obj *u_dp, Data_Obj *w_dp, Data_Obj *v_dp)
{
	//SORT_EIGENVECTORS(float)
	dimension_t i,n;
	u_long *index_data;

	n = w_dp->dt_cols;

	index_dp = mkvec("svd_tmp_indices",n,1,PREC_UDI);
	if( index_dp == NO_OBJ ){
		NWARN("Unable to create index object for sorting SVD eigenvalues");
		return;
	}

	sort_indices(index_dp,w_dp);
	/* sorting is done from smallest to largest */

	j=0;
	index_data = index_dp->dt_data;

	/* We should only have to have 1 column of storage to permute things,
	 * but life is simplified if we copy the data...
	 */

	for(i=n-1;i>=0;i--){
		k= *(index_data+i);
	}
}

static void double_sort_svd_eigenvectors(Data_Obj *u_dp, Data_Obj *w_dp, Data_Obj *v_dp)
{
	//SORT_EIGENVECTORS(double)
	NWARN("double_sort_svd_eigenvectors not implemented");
}
#endif /* FOOBAR */

void dp_choldc(Data_Obj *a_dp, Data_Obj *p_dp)
{
	unsigned m, n;
	void *a_rowlist[MAX_DIM];
	/*
	n=a_dp->dt_rows;

	if( n > MAX_DIM ){
		NWARN("Sorry, MAX dimension exceeded in dp_choldc");
		sprintf(DEFAULT_ERROR_STRING,"dp_choldc:  MAX_DIM = %d, n = %d", MAX_DIM,n);
		advise(DEFAULT_ERROR_STRING);
		return;
	}
	*/
	n=a_dp->dt_cols;
	m=a_dp->dt_rows;

	if( n > MAX_DIM || m > MAX_DIM ){
		NWARN("Sorry, MAX dimension exceeded in dp_choldc");
		sprintf(DEFAULT_ERROR_STRING,"dp_choldc:  MAX_DIM = %d, n = %d, m = %d",
			MAX_DIM,n,m);
		advise(DEFAULT_ERROR_STRING);
		return;
	}

	printf("nrmenu:numrec.c data %f\n", *((float *)a_dp->dt_data));
	
	
	if( MACHINE_PREC(a_dp) == PREC_SP ){
		float_init_rowlist((float **)(void *)a_rowlist,a_dp);

		float_choldc(((float **)(void *)a_rowlist)-1,n,((float *)p_dp->dt_data)-1);
		
	} else if( MACHINE_PREC(a_dp) == PREC_DP ){
		double_init_rowlist((double **)(void *)a_rowlist,a_dp);

		double_choldc(((double **)(void *)a_rowlist)-1,n,((double *)p_dp->dt_data)-1);
	}
	else {
		NWARN("bad machine precision in dp_choldc");
	}
	
}


/*
 * Compute the singular value decomposition of the matrix a.
 * results are in matrix v, and eigen values w, and matrix u
 * which replaces a.
 */

void dp_svd(Data_Obj *a_dp, Data_Obj *w_dp, Data_Obj *v_dp)
{
	unsigned m,n;
	void *a_rowlist[MAX_DIM], *v_rowlist[MAX_DIM];

	n=a_dp->dt_cols;
	m=a_dp->dt_rows;

	if( n > MAX_DIM || m > MAX_DIM ){
		NWARN("Sorry, MAX dimension exceeded in dp_svd");
		sprintf(DEFAULT_ERROR_STRING,"dp_svdcmp:  MAX_DIM = %d, n = %d, m = %d",
			MAX_DIM,n,m);
		advise(DEFAULT_ERROR_STRING);
		return;
	}

	/*
	if( m < n ){
		sprintf(DEFAULT_ERROR_STRING,"dp_svdcmp:  input matrix %s (%d x %d) cannot be wider than tall!?",
			a_dp->dt_name,m,n);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	*/
	if( w_dp->dt_cols != n ){
		sprintf(DEFAULT_ERROR_STRING,"dp_svdcmp:  weight vector %s should have %d columns, to match input matrix %s!?",
			w_dp->dt_name,n,a_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if(w_dp->dt_rows != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"dp_svdcmp:  weight vector %s should be a vector, (rows = %d)!?",
			w_dp->dt_name,w_dp->dt_rows);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( v_dp->dt_cols != n || v_dp->dt_rows != n ){
		sprintf(DEFAULT_ERROR_STRING,
			"V matrix %s should be square with dimension %d, to match # columns of input %s",
			v_dp->dt_name,n,a_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	/* BUG make sure all vectors have same precision */

	if( MACHINE_PREC(a_dp) == PREC_SP ){
		float_init_rowlist((float **)(void *)a_rowlist,a_dp);
		float_init_rowlist((float **)(void *)v_rowlist,v_dp);

//advise("calling float_svdcmp");
		float_svdcmp(((float **)(void *)a_rowlist)-1,m,n,((float *)w_dp->dt_data)-1,((float **)(void *)v_rowlist)-1);
//advise("back from float_svdcmp");
		/* The eigenvectors aren't sorted by numerical recipes... */
		//float_sort_svd_eigenvectors(a_dp,w_dp,v_dp);
	} else if( MACHINE_PREC(a_dp) == PREC_DP ){
		double_init_rowlist((double **)(void *)a_rowlist,a_dp);
		double_init_rowlist((double **)(void *)v_rowlist,v_dp);

//advise("calling double_svdcmp");
		double_svdcmp(((double **)(void *)a_rowlist)-1,m,n,((double *)w_dp->dt_data)-1,((double **)(void *)v_rowlist)-1);
//advise("back from double_svdcmp");
		/* The eigenvectors aren't sorted by numerical recipes... */
		//double_sort_svd_eigenvectors(a_dp,w_dp,v_dp);
	}
	else {
		NWARN("bad machine precision in dp_svd");
	}


}

void dp_zroots(Data_Obj *r_dp, Data_Obj *a_dp, int polish )
{
	int m,n;

	n=a_dp->dt_cols;	/* polynomial degree + 1 */
	m=r_dp->dt_cols;

	if( m != n-1 ){
		sprintf(DEFAULT_ERROR_STRING,
	"dp_zroots:  len of root vector %s (%d) inconsistent with coefficients vector %s (%d)",
			r_dp->dt_name,r_dp->dt_cols,a_dp->dt_name,a_dp->dt_cols);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	/* BUG make sure are row vectors */

	if(a_dp->dt_rows != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"dp_zroots:  coefficient vector %s should be a row vector, (rows = %d)!?",
			a_dp->dt_name,a_dp->dt_rows);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( r_dp->dt_rows != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"dp_zroots:  root vector %s should be a row vector, (rows = %d)!?",
			r_dp->dt_name,r_dp->dt_rows);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	/* BUG make sure all vectors have same precision */

	if( MACHINE_PREC(a_dp) == PREC_SP ){
		/* Why do we subtract 1 from the address of the roots, but not
		 * the coefficients!?
		 */

		float_zroots(
			((fcomplex *)a_dp->dt_data)/*-1*/,
			m,
			((fcomplex *)r_dp->dt_data)-1,
			polish);

	} else if( MACHINE_PREC(a_dp) == PREC_DP ){
		double_zroots( ((dcomplex *)a_dp->dt_data)-1,m,((dcomplex *)r_dp->dt_data)-1,polish);
	}
	else {
		NWARN("bad machine precision in dp_zroots");
	}
}


/*
 * Compute the solution parameters x.
 * Inputs are the input data b, and u,w, and v returned from svdcmp.
 * see pg. 64
 */

/* args:	coeffs U eigenvalues V datain	*/

void dp_svbksb(Data_Obj *x_dp, Data_Obj *u_dp, Data_Obj *w_dp, Data_Obj *v_dp, Data_Obj *b_dp)
{
	unsigned m,n;
	void *u_rowlist[MAX_DIM], *v_rowlist[MAX_DIM];

	n=u_dp->dt_cols;
	m=u_dp->dt_rows;
	if( m < n ){
		sprintf(DEFAULT_ERROR_STRING,"dp_svbksb:  matrix %s (%d x %d) cannot be wider than tall",
			u_dp->dt_name,m,n);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( w_dp->dt_cols != n ){
		sprintf(DEFAULT_ERROR_STRING,
			"dimension of eigenvalue vector %s (%d) must be match # of columns of matrix %s (%d)",
			w_dp->dt_name,w_dp->dt_cols,u_dp->dt_name,n);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if(w_dp->dt_rows != 1){
		sprintf(DEFAULT_ERROR_STRING,"dp_svbksb:  eigenvalue vector %s (%d rows) should be a row vector!?",
			w_dp->dt_name,w_dp->dt_rows);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if(b_dp->dt_rows != 1){
		sprintf(DEFAULT_ERROR_STRING,"dp_svbksb:  data vector %s (%d rows) should be a row vector!?",
			b_dp->dt_name,b_dp->dt_rows);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( x_dp->dt_rows != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"dp_svbksb:  weight vector %s (%d rows) should be a row vector!?",
			x_dp->dt_name,x_dp->dt_rows);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( v_dp->dt_cols != n || v_dp->dt_rows != n ){
		sprintf(DEFAULT_ERROR_STRING,"dp_svbksb:  V matrix %s (%d x %d) should be square with dimension %d",
			v_dp->dt_name,v_dp->dt_rows,v_dp->dt_cols,n);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( u_dp->dt_rows > MAX_DIM ){
		sprintf(DEFAULT_ERROR_STRING,"dp_svbksb:  matrix %s has %d rows, max is %d",
			u_dp->dt_name,u_dp->dt_rows,MAX_DIM);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( v_dp->dt_rows > MAX_DIM ){
		sprintf(DEFAULT_ERROR_STRING,"dp_svbksb:  matrix %s has %d rows, max is %d",
			v_dp->dt_name,u_dp->dt_rows,MAX_DIM);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	/* BUG make sure precisions match */

	if( MACHINE_PREC(u_dp) == PREC_SP ){
		float_init_rowlist((float **)(void *)u_rowlist,u_dp);
		float_init_rowlist((float **)(void *)v_rowlist,v_dp);

//advise("calling float_svbksb...");
		float_svbksb(((float **)(void *)u_rowlist)-1,((float *)w_dp->dt_data)-1,((float **)(void *)v_rowlist)-1,
			m,n,(((float *)b_dp->dt_data)-1),(((float *)x_dp->dt_data)-1));
//advise("back from float_svbksb...");
	} else if( MACHINE_PREC(u_dp) == PREC_DP ){
		double_init_rowlist((double **)(void *)u_rowlist,u_dp);
		double_init_rowlist((double **)(void *)v_rowlist,v_dp);

		double_svbksb(((double **)(void *)u_rowlist)-1,((double *)w_dp->dt_data)-1,((double **)(void *)v_rowlist)-1,
			m,n,(((double *)b_dp->dt_data)-1),(((double *)x_dp->dt_data)-1));
	} else {
		NWARN("bad precision in dp_svbksb");
	}

}

void dp_jacobi(Data_Obj *v_dp, Data_Obj *d_dp, Data_Obj *a_dp, int *nrotp)
{
	void *a_rowlist[MAX_DIM], *v_rowlist[MAX_DIM];
	int n;

	if( a_dp->dt_cols != a_dp->dt_rows ){
		sprintf(DEFAULT_ERROR_STRING,"dp_jacobi:  matrix %s must be square for jacobi",a_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( v_dp->dt_cols != v_dp->dt_rows ){
		sprintf(DEFAULT_ERROR_STRING,"dp_jacobi:  matrix %s must be square for jacobi",v_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( v_dp->dt_cols != a_dp->dt_cols ){
		sprintf(DEFAULT_ERROR_STRING,"dp_jacobi:  size of eigenvector matrix %s must match input matrix %s for jacobi",
			v_dp->dt_name,a_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( d_dp->dt_cols != a_dp->dt_cols ){
		sprintf(DEFAULT_ERROR_STRING,"dp_jacobi:  size of eigenvalue vector %s must match input matrix %s for jacobi",
			d_dp->dt_name,a_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( ! IS_CONTIGUOUS(a_dp) ){
		sprintf(DEFAULT_ERROR_STRING,"dp_jacobi:  Object %s must be contiguous for jacobi",a_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( ! IS_CONTIGUOUS(d_dp) ){
		sprintf(DEFAULT_ERROR_STRING,"dp_jacobi:  Object %s must be contiguous for jacobi",d_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( ! IS_CONTIGUOUS(v_dp) ){
		sprintf(DEFAULT_ERROR_STRING,"dp_jacobi:  Object %s must be contiguous for jacobi",v_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	n = a_dp->dt_cols;
	/* BUG make sure types match */

	if( MACHINE_PREC(a_dp) == PREC_SP ){
		float_init_rowlist((float **)(void *)a_rowlist,a_dp);
		float_init_rowlist((float **)(void *)v_rowlist,v_dp);

		float_jacobi(((float **)(void *)a_rowlist)-1,n,((float *)(d_dp->dt_data))-1,((float **)(void *)v_rowlist)-1,nrotp);
	} else if( MACHINE_PREC(a_dp) == PREC_DP ){
		double_init_rowlist((double **)(void *)a_rowlist,a_dp);
		double_init_rowlist((double **)(void *)v_rowlist,v_dp);

		double_jacobi(((double **)(void *)a_rowlist)-1,n,((double *)(d_dp->dt_data))-1,((double **)(void *)v_rowlist)-1,nrotp);
	} else {
		NWARN("bad precision in dp_jacobi");
	}
}

void dp_eigsrt(Data_Obj *v_dp, Data_Obj *d_dp)
{
	void *v_rowlist[MAX_DIM];
	int n;

	if( v_dp->dt_cols != v_dp->dt_rows ){
		sprintf(DEFAULT_ERROR_STRING,"dp_eigsrt:  matrix %s must be square for eigsrt",v_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( v_dp->dt_cols != d_dp->dt_cols ){
		sprintf(DEFAULT_ERROR_STRING,"dp_eigsrt:  size of eigenvalue vector %s must match input matrix %s for eigsrt",
			v_dp->dt_name,d_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( ! IS_CONTIGUOUS(d_dp) ){
		sprintf(DEFAULT_ERROR_STRING,"dp_eigsrt:  Object %s must be contiguous for eigsrt",d_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	if( ! IS_CONTIGUOUS(v_dp) ){
		sprintf(DEFAULT_ERROR_STRING,"dp_eigsrt:  Object %s must be contiguous for eigsrt",v_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	n = v_dp->dt_cols;

	/* BUG make sure all types match */

	if( MACHINE_PREC(v_dp) == PREC_SP ){
		float_init_rowlist((float **)(void *)v_rowlist,v_dp);
		float_eigsrt(((float *)(d_dp->dt_data))-1,((float **)(void *)v_rowlist)-1,n);
	} else if( MACHINE_PREC(v_dp) == PREC_DP ){
		double_init_rowlist((double **)(void *)v_rowlist,v_dp);
		double_eigsrt(((double *)(d_dp->dt_data))-1,((double **)(void *)v_rowlist)-1,n);
	} else {
		NWARN("bad precision in dp_eigsrt");
	}
}

void dp_moment(Data_Obj *d_dp)
{
	/* std_type *d_rowlist[MAX_DIM]; */
	int n;
	if( ! IS_CONTIGUOUS(d_dp) ){
           	sprintf(DEFAULT_ERROR_STRING,"dp_moment:  Object %s must be contiguous for eigsrt",d_dp->dt_name);
                NWARN(DEFAULT_ERROR_STRING);
                return;
        }
	n = d_dp->dt_cols;

	if( MACHINE_PREC(d_dp) == PREC_SP ){
       		float adev, var, skew, curt;
		float ave, sdev;
	
		/* BUG - the results don't get passed out anywhere!?
		 */

		float_moment(((float *)(d_dp->dt_data)),n,&ave,&adev,&sdev,&var,&skew,&curt);
	} else if( MACHINE_PREC(d_dp) == PREC_DP ){
       		double adev, var, skew, curt;
		double ave, sdev;
	
		double_moment(((double *)(d_dp->dt_data)),n,&ave,&adev,&sdev,&var,&skew,&curt);
	}
}


#endif /* HAVE_NUMREC */

