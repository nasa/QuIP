#include "quip_config.h"

#ifdef HAVE_GSL

#include <stdio.h>
#include "quip_prot.h"
#include "data_obj.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>
#include "gslprot.h"


static void export_matrix_data_to_gsl( QSP_ARG_DECL  gsl_matrix *mp, Data_Obj *dp )
{
	/* BUG?  check for matching sizes? */

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,
"export_matrix_data_to_gsl:  Sorry, for now object %s should be contiguous",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}

	switch( OBJ_MACH_PREC(dp) ){
		case PREC_SP:			/* single precision float */
			{
			float *fptr;
			dimension_t i,j;
			fptr = OBJ_DATA_PTR(dp);
			for(i=0;i<OBJ_ROWS(dp);i++){
				for(j=0;j<OBJ_COLS(dp);j++){
					gsl_matrix_set(mp,i,j,*fptr);
					/* Object assumed to be contiguous;
					*/
					fptr++;
				}
			}
			}
			break;

		case PREC_DP:			/* double precision float */
			{
			double *dptr;
			dimension_t i,j;
			dptr = OBJ_DATA_PTR(dp);
			for(i=0;i<OBJ_ROWS(dp);i++){
				for(j=0;j<OBJ_COLS(dp);j++){
					gsl_matrix_set(mp,i,j,*dptr);
					/* Object assumed to be contiguous;
					*/
					dptr++;
				}
			}
			}
			break;

		default:
	WARN("export_matrix_data_to_gsl: Oops, unhandled precision");
			return;
			break;
	}
}

static void export_vector_data_to_gsl( QSP_ARG_DECL  gsl_vector *vp, Data_Obj *dp )
{
	/* BUG?  check for matching sizes? */

	if(! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"export_matrix_data_to_gsl:  Sorry, for now object %s should be contiguous",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}

	switch( OBJ_MACH_PREC(dp) ){
		case PREC_SP:			/* single precision float */
			{
			float *fptr;
			dimension_t i;
			fptr = OBJ_DATA_PTR(dp);
			for(i=0;i<OBJ_N_MACH_ELTS(dp);i++){
				gsl_vector_set(vp,i,*fptr);
				/* Object assumed to be contiguous; */
				fptr++;
			}
			}
			break;

		case PREC_DP:			/* double precision float */
			{
			double *dptr;
			dimension_t i;
			dptr = OBJ_DATA_PTR(dp);
			for(i=0;i<OBJ_N_MACH_ELTS(dp);i++){
				gsl_vector_set(vp,i,*dptr);
				/* Object assumed to be contiguous; */
				dptr++;
			}
			}
			break;

		default:
	WARN("export_matrix_data_to_gsl: Oops, unhandled precision");
			return;
			break;
	}
}

static void import_matrix_data_from_gsl( QSP_ARG_DECL  Data_Obj *dp, gsl_matrix *mp )
{
	/* BUG?  check for matching sizes? */

	if(! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,
"import_matrix_data_from_gsl:  Sorry, for now object %s should be contiguous",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}

	switch( OBJ_MACH_PREC(dp) ){
		case PREC_SP:			/* single precision float */
			{
			float *fptr;
			dimension_t i,j;
			fptr = OBJ_DATA_PTR(dp);
			for(i=0;i<OBJ_ROWS(dp);i++){
				for(j=0;j<OBJ_COLS(dp);j++){
					*fptr = gsl_matrix_get(mp,i,j);
					/* Object assumed to be contiguous */
					fptr++;
				}
			}
			}
			break;

		case PREC_DP:			/* double precision float */
			{
			double *dptr;
			dimension_t i,j;
			dptr = OBJ_DATA_PTR(dp);
			for(i=0;i<OBJ_ROWS(dp);i++){
				for(j=0;j<OBJ_COLS(dp);j++){
					*dptr = gsl_matrix_get(mp,i,j);
					/* Object assumed to be contiguous */
					dptr++;
				}
			}
			}
			break;

		default:
	WARN("import_matrix_data_from_gsl: Oops, unhandled precision");
			return;
			break;
	}
}

static void import_vector_data_from_gsl( QSP_ARG_DECL  Data_Obj *dp, gsl_vector *vp )
{
	/* BUG?  check for matching sizes? */

	if(! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"import_vector_data_from_gsl:  Sorry, for now object %s should be contiguous",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}

	switch( OBJ_MACH_PREC(dp) ){
		case PREC_SP:			/* single precision float */
			{
			float *fptr;
			dimension_t i;
			fptr = OBJ_DATA_PTR(dp);
			for(i=0;i<OBJ_N_MACH_ELTS(dp);i++){
				*fptr = gsl_vector_get(vp,i);
				/* Object assumed to be contiguous */
				fptr++;
			}
			}
			break;

		case PREC_DP:			/* double precision float */
			{
			double *dptr;
			dimension_t i;
			dptr = OBJ_DATA_PTR(dp);
			for(i=0;i<OBJ_N_MACH_ELTS(dp);i++){
				*dptr = gsl_vector_get(vp,i);
				/* Object assumed to be contiguous */
				dptr++;
			}
			}
			break;

		default:
			WARN("import_matrix_data_from_gsl: Oops, unhandled precision case");
			return;
			break;
	}
}

static gsl_matrix *gsl_init_matrix(QSP_ARG_DECL  Data_Obj *dp)
{
	gsl_matrix *mp;

	/* We should check the type??? */

	mp=gsl_matrix_alloc(OBJ_ROWS(dp),OBJ_COLS(dp));
	/* BUG?  test return value for success? */

	export_matrix_data_to_gsl(QSP_ARG  mp,dp);

	return(mp);
}

static gsl_vector *gsl_init_vector(QSP_ARG_DECL  Data_Obj *dp)
{
	gsl_vector *vp;

	vp=gsl_vector_alloc(OBJ_N_MACH_ELTS(dp));
	/* BUG?  test return value for success? */

	export_vector_data_to_gsl(QSP_ARG  vp,dp);

	return(vp);
}

#define MAX_DIM 1024

void gsl_svd (QSP_ARG_DECL  Data_Obj *a_dp, Data_Obj *w_dp, Data_Obj *v_dp)
{
	unsigned m,n;
	gsl_matrix *a_matrix,*v_matrix;
	gsl_vector *w_vector;
        gsl_vector *work;

	n=OBJ_COLS(a_dp);		/* number of params */
	m=OBJ_ROWS(a_dp);		/* number of pts */

        if ( n > MAX_DIM || m > MAX_DIM) {
		WARN("Sorry, MAX dimension exceeded in gsl_svd");
		return;
	}
	if( n > m ){
        	sprintf(ERROR_STRING,
	"gsl_svd: input matrix %s (%d x %d) cannot be wider than tall!?",
			OBJ_NAME(a_dp),m,n);
		WARN(ERROR_STRING);
		return;
	}

	if( OBJ_COLS(w_dp) != n ){
		sprintf(ERROR_STRING,
"gsl_svd: weight vector %s should have %d columns, to match input matrix %s!?",
			OBJ_NAME(w_dp),n,OBJ_NAME(a_dp));
		WARN(ERROR_STRING);
		return;
        }

        a_matrix = gsl_init_matrix(QSP_ARG  a_dp);
        w_vector = gsl_init_vector(QSP_ARG  w_dp);
        v_matrix = gsl_init_matrix(QSP_ARG  v_dp);

        work = gsl_vector_alloc(n);
		
	gsl_linalg_SV_decomp(a_matrix,v_matrix,w_vector,work);	

	/* Now move the data back into our objects... */
	import_matrix_data_from_gsl(QSP_ARG  a_dp,a_matrix);
	import_vector_data_from_gsl(QSP_ARG  w_dp,w_vector);
	import_matrix_data_from_gsl(QSP_ARG  v_dp,v_matrix);
}

void gsl_solve(QSP_ARG_DECL  Data_Obj *x_dp, Data_Obj *u_dp, Data_Obj *w_dp, Data_Obj *v_dp,
		Data_Obj *b_dp )
{
	dimension_t m,n;
	gsl_matrix *u_matrix,*v_matrix;
	gsl_vector *w_vector;
	gsl_vector *x_vector,*b_vector;

	// checks copied from nrmenu/numrec.c

	n=OBJ_COLS(u_dp);
	m=OBJ_ROWS(u_dp);
	if( m < n ){
		sprintf(ERROR_STRING,"do_gsl_solve:  matrix %s (%d x %d) cannot be wider than tall",
			OBJ_NAME(u_dp),m,n);
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_COLS(w_dp) != n ){
		sprintf(ERROR_STRING,
			"do_gsl_solve:  dimension of eigenvalue vector %s (%d) must be match # of columns of matrix %s (%d)",
			OBJ_NAME(w_dp),OBJ_COLS(w_dp),OBJ_NAME(u_dp),n);
		WARN(ERROR_STRING);
		return;
	}
	if(OBJ_ROWS(w_dp) != 1){
		sprintf(ERROR_STRING,"do_gsl_solve:  eigenvalue vector %s (%d rows) should be a row vector!?",
			OBJ_NAME(w_dp),OBJ_ROWS(w_dp));
		WARN(ERROR_STRING);
		return;
	}
	if(OBJ_ROWS(b_dp) != 1){
		sprintf(ERROR_STRING,"do_gsl_solve:  data vector %s (%d rows) should be a row vector!?",
			OBJ_NAME(b_dp),OBJ_ROWS(b_dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_ROWS(x_dp) != 1 ){
		sprintf(ERROR_STRING,"do_gsl_solve:  weight vector %s (%d rows) should be a row vector!?",
			OBJ_NAME(x_dp),OBJ_ROWS(x_dp));
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_COLS(v_dp) != n || OBJ_ROWS(v_dp) != n ){
		sprintf(ERROR_STRING,"do_gsl_solve:  V matrix %s (%d x %d) should be square with dimension %d",
			OBJ_NAME(v_dp),OBJ_ROWS(v_dp),OBJ_COLS(v_dp),n);
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_ROWS(u_dp) > MAX_DIM ){
		sprintf(ERROR_STRING,"do_gsl_solve:  matrix %s has %d rows, max is %d",
			OBJ_NAME(u_dp),OBJ_ROWS(u_dp),MAX_DIM);
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_ROWS(v_dp) > MAX_DIM ){
		sprintf(ERROR_STRING,"do_gsl_solve:  matrix %s has %d rows, max is %d",
			OBJ_NAME(v_dp),OBJ_ROWS(u_dp),MAX_DIM);
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_COLS(b_dp) != OBJ_ROWS(u_dp) ){
		sprintf(ERROR_STRING,"do_gsl_solve:  Number of elements of data vector %s (%ld) should match number of rows of U matrix %s (%ld)",
			OBJ_NAME(b_dp),(long)OBJ_COLS(b_dp),OBJ_NAME(u_dp),(long)OBJ_ROWS(u_dp));
		WARN(ERROR_STRING);
		return;
	}
	/* BUG make sure precisions match */

        u_matrix = gsl_init_matrix(QSP_ARG  u_dp);
        w_vector = gsl_init_vector(QSP_ARG  w_dp);
        v_matrix = gsl_init_matrix(QSP_ARG  v_dp);
        b_vector = gsl_init_vector(QSP_ARG  b_dp);
        x_vector = gsl_init_vector(QSP_ARG  x_dp);	// no need to copy data?

	gsl_linalg_SV_solve(u_matrix,v_matrix,w_vector,b_vector,x_vector);

	import_vector_data_from_gsl(QSP_ARG  x_dp,x_vector);
}

#endif /* HAVE_GSL */
