#include "quip_config.h"

char VersionId_gslmenu_gsl[] = QUIP_VERSION_STRING;

#ifdef HAVE_GSL


#include <stdio.h>
#include "data_obj.h"
/* #include "warproto.h" */
#include "data_obj.h"
#include "query.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_linalg.h>


static void export_matrix_data_to_gsl( gsl_matrix *mp, Data_Obj *dp )
{
	/* BUG?  check for matching sizes? */

	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(DEFAULT_ERROR_STRING,
"export_matrix_data_to_gsl:  Sorry, for now object %s should be contiguous",
			dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	switch( MACHINE_PREC(dp) ){
		case PREC_SP:			/* single precision float */
			{
			float *fptr;
			dimension_t i,j;
			fptr = dp->dt_data;
			for(i=0;i<dp->dt_rows;i++){
				for(j=0;j<dp->dt_cols;j++){
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
			dptr = dp->dt_data;
			for(i=0;i<dp->dt_rows;i++){
				for(j=0;j<dp->dt_cols;j++){
					gsl_matrix_set(mp,i,j,*dptr);
					/* Object assumed to be contiguous;
					*/
					dptr++;
				}
			}
			}
			break;

		default:
	NWARN("export_matrix_data_to_gsl: Oops, unhandled precision");
			return;
			break;
	}
}

static void export_vector_data_to_gsl( gsl_vector *vp, Data_Obj *dp )
{
	/* BUG?  check for matching sizes? */

	if(! IS_CONTIGUOUS(dp) ){
		sprintf(DEFAULT_ERROR_STRING,"export_matrix_data_to_gsl:  Sorry, for now object %s should be contiguous",dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	switch( MACHINE_PREC(dp) ){
		case PREC_SP:			/* single precision float */
			{
			float *fptr;
			dimension_t i;
			fptr = dp->dt_data;
			for(i=0;i<dp->dt_n_mach_elts;i++){
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
			dptr = dp->dt_data;
			for(i=0;i<dp->dt_n_mach_elts;i++){
				gsl_vector_set(vp,i,*dptr);
				/* Object assumed to be contiguous; */
				dptr++;
			}
			}
			break;

		default:
	NWARN("export_matrix_data_to_gsl: Oops, unhandled precision");
			return;
			break;
	}
}

void import_matrix_data_from_gsl( Data_Obj *dp, gsl_matrix *mp )
{
	/* BUG?  check for matching sizes? */

	if(! IS_CONTIGUOUS(dp) ){
		sprintf(DEFAULT_ERROR_STRING,
"import_matrix_data_from_gsl:  Sorry, for now object %s should be contiguous",
			dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	switch( MACHINE_PREC(dp) ){
		case PREC_SP:			/* single precision float */
			{
			float *fptr;
			dimension_t i,j;
			fptr = dp->dt_data;
			for(i=0;i<dp->dt_rows;i++){
				for(j=0;j<dp->dt_cols;j++){
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
			dptr = dp->dt_data;
			for(i=0;i<dp->dt_rows;i++){
				for(j=0;j<dp->dt_cols;j++){
					*dptr = gsl_matrix_get(mp,i,j);
					/* Object assumed to be contiguous */
					dptr++;
				}
			}
			}
			break;

		default:
	NWARN("import_matrix_data_from_gsl: Oops, unhandled precision");
			return;
			break;
	}
}

void import_vector_data_from_gsl( Data_Obj *dp, gsl_vector *vp )
{
	/* BUG?  check for matching sizes? */

	if(! IS_CONTIGUOUS(dp) ){
		sprintf(DEFAULT_ERROR_STRING,"import_vector_data_from_gsl:  Sorry, for now object %s should be contiguous",dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	switch( MACHINE_PREC(dp) ){
		case PREC_SP:			/* single precision float */
			{
			float *fptr;
			dimension_t i;
			fptr = dp->dt_data;
			for(i=0;i<dp->dt_n_mach_elts;i++){
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
			dptr = dp->dt_data;
			for(i=0;i<dp->dt_n_mach_elts;i++){
				*dptr = gsl_vector_get(vp,i);
				/* Object assumed to be contiguous */
				dptr++;
			}
			}
			break;

		default:
			NWARN("import_matrix_data_from_gsl: Oops, unhandled precision case");
			return;
			break;
	}
}

gsl_matrix *gsl_init_matrix(Data_Obj *dp)
{
	gsl_matrix *mp;

	/* We should check the type??? */

	mp=gsl_matrix_alloc(dp->dt_rows,dp->dt_cols);
	/* BUG?  test return value for success? */

	export_matrix_data_to_gsl(mp,dp);

	return(mp);
}

gsl_vector *gsl_init_vector(Data_Obj *dp)
{
	gsl_vector *vp;

	vp=gsl_vector_alloc(dp->dt_n_mach_elts);
	/* BUG?  test return value for success? */

	export_vector_data_to_gsl(vp,dp);

	return(vp);
}

#define MAX_DIM 1024

void gsl_svd (Data_Obj *a_dp, Data_Obj *w_dp, Data_Obj *v_dp)
{
	unsigned m,n;
	gsl_matrix *a_matrix,*v_matrix;
	gsl_vector *w_vector;
        gsl_vector *work;

	n=a_dp->dt_cols;		/* number of params */
	m=a_dp->dt_rows;		/* number of pts */

        if ( n > MAX_DIM || m > MAX_DIM) {
		NWARN("Sorry, MAX dimension exceeded in gsl_svd");
		return;
	}
	if( n > m ){
        	sprintf(DEFAULT_ERROR_STRING,
	"gsl_svd: input matrix %s (%d x %d) cannot be wider than tall!?",
			a_dp->dt_name,m,n);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	if( w_dp->dt_cols != n ){
		sprintf(DEFAULT_ERROR_STRING,
"gsl_svd: weight vector %s should have %d columns, to match input matrix %s!?",
			w_dp->dt_name,n,a_dp->dt_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
        }

        a_matrix = gsl_init_matrix(a_dp);
        w_vector = gsl_init_vector(w_dp);
        v_matrix = gsl_init_matrix(v_dp);

        work = gsl_vector_alloc(n);
		
	gsl_linalg_SV_decomp(a_matrix,v_matrix,w_vector,work);	

	/* Now move the data back into our objects... */
	import_matrix_data_from_gsl(a_dp,a_matrix);
	import_vector_data_from_gsl(w_dp,w_vector);
	import_matrix_data_from_gsl(v_dp,v_matrix);
}

#endif /* HAVE_GSL */
