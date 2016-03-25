#include "quip_config.h"
#include "quip_prot.h"

/**	invert.c	subroutine for general matrix inversion */

/* this procedure seems to suck, and is generally inferior
 * to Gauss-Jordan!?
 */

#include "vec_util.h"

double dt_invert(QSP_ARG_DECL  Data_Obj *dp)
{
	dimension_t size;
	float *matrix;

	if( OBJ_ROWS(dp) != OBJ_COLS(dp) ){
		NWARN("matrix must be square");
		return(0.0);
	}
	if( OBJ_PREC(dp) != PREC_SP ){
		NWARN("matrix precision must be float");
		return(0.0);
	}
	if( ! IS_CONTIGUOUS(dp) ){
		NWARN("matrix object must be contiguous");
		return(0.0);
	}
	if( OBJ_COMPS(dp) != 1 ){
		NWARN("matrix componenet dimension must be 1");
		return(0.0);
	}

	size=OBJ_COLS(dp);
	matrix=(float *)OBJ_DATA_PTR(dp);

	return( invert_sq_matrix(QSP_ARG  matrix,size) );
}

/* BUG it would be nice to have a comment here saying what algorithm
 * is implemented, plus a reference with page numbers!!!
 */

double invert_sq_matrix(QSP_ARG_DECL  float *matrix,dimension_t size)
{
	dimension_t	j,k,l;
	double	pivot;
	double	temp;
	double	det = 1.0;

if( verbose ){
prt_msg("invert_sq_matrix:  input matrix");
/* show_matrix(matrix); */
}
	for (j = 0; j < size; j++) {
		pivot = matrix[j*size+j];
		det *= pivot;
		matrix[j*size+j] = 1.0;
		if( pivot == 0.0 ) {
			sprintf(ERROR_STRING,"zero pivot, j=%d",j);
			WARN(ERROR_STRING);
			return (0.0);
		}
		for (k = 0; k < size; k++)
			matrix[j*size+k] /= pivot;
		for (k = 0; k < size; k++)
			if (k != j) {
				temp = matrix[k*size+j];
				matrix[k*size+j] = 0.0;
				for (l = 0; l < size; l++)
					matrix[k*size+l] -=
						matrix[j*size+l] * temp;
			}
	}
if( verbose ){
prt_msg("invert_sq_matrix:  output matrix");
/* show_matrix(matrix); */
}
	return (det);
}

