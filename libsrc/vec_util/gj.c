#include "quip_config.h"

char VersionId_vec_util_gj[] = QUIP_VERSION_STRING;

/* Gauss-Jordan elimination matrix inversion
 *
 * This looks suspiciously like Numerical Recipes code!?
 * should macro-ize the 2 versions...
 */

#include <math.h>
#include <stdio.h>
#include "debug.h"

#include "vec_util.h"

#define MAXSIZE	256

#define SWAP(a,b)	{ float tmp=(a); (a)=(b); (b)=tmp; }

/* assume no right-hand side vectors */

/* changed from void to int to return status:
 * 0 all is cool
 * -1 singular matrix
 * -2 out of memory
 */

int gauss_jordan(float *matrix,dimension_t size)
{
	int indxc[MAXSIZE];
	int indxr[MAXSIZE];
	int ipiv[MAXSIZE];
	/* These used to be dimension_t, but l (at least) needs to be able to go negative... */
	incr_t l;
	dimension_t i,j,k;
	dimension_t icol=0, irow=0;	/* initialize to elim compiler warning */
	incr_t rowdex, rowdex2;
	dimension_t ll;
	double big, tmp, pivinv;

	for(j=0;j<size;j++) ipiv[j]=0;

	for(i=0;i<size;i++){	/* main loop over columns to be reduced */
		big=0.0;
		for(j=0;j<size;j++)	/* outer loop of pivot search */
			/* is this one an index??? */
			if( ipiv[j] != 1 )
				/* find the largest element in the jth row */
				for(k=0;k<size;k++){
					if( ipiv[k] == 0 ){
						if((tmp=fabs(matrix[j*size+k]))
						>= big ){
							big=tmp;
							irow=j;
							icol=k;
						}
					} else if( ipiv[k] > 1 ){
						NWARN("G-J:  singular, returning -1");
						return(-1);
					}
				}
		++(ipiv[icol]);

	/* We now have the pivot element, so we interchange rows
	 * (if needed) to put the pivot element on the diagonal.
	 * The columns are not physically interchanged, only relabeled;
	 * indx[i], the column of the ith pivot element,
	 * is in the ith column that is reduced,
	 * while indxr[i] is the row in which that pivot element was
	 * originally located.  If indxr[i]!=indxc[i] there is an
	 * implied column interchange.  With this form of bookkeeping,
	 * the solution b's will end up in the correct order, and the
	 * inverse matrix will be scrambled by columns.
	 */

		if( irow != icol ){
			for(l=0;l<(incr_t)size;l++)
				SWAP(matrix[irow*size+l],matrix[icol*size+l])
		}
		indxr[i]=irow;
		indxc[i]=icol;
		if( (tmp=matrix[icol*size+icol]) == 0.0 ){
			NWARN("G-J:  singular matrix");
			return(-1);
		}
		pivinv = 1/tmp;
		matrix[icol*size+icol]=1.0;
		rowdex = icol * size;
		for(l=0;l<(incr_t)size;l++){
			matrix[rowdex+l] *= pivinv;
		}
		/* now reduce the rows (except the pivot one) */
		for(ll=0;ll<size;ll++){
			rowdex2=ll*size;
			if( ll != icol ){
				tmp=matrix[rowdex2+icol];	/* the element we want to zero */
				matrix[rowdex2+icol]=0.0;
				for(l=0;l<(incr_t)size;l++)
					matrix[rowdex2+l] -=
						matrix[rowdex+l]*tmp;
			}
		}
	}
	/* end of main loop over columns of the reduction */

	/* it remains only to unscramble the solution in view
	 * of the column interchanges.  We do this by interchanging
	 * pairs of columns in the reverse order that the permutaion
	 * was built up
	 */

	for(l=size-1;l>=0;l--){
		if( indxr[l] != indxc[l] ){
			for(k=0;k<size;k++){
				rowdex=k*size;
			SWAP(matrix[rowdex+indxr[l]],matrix[rowdex+indxc[l]])
			}
		}
	}

	return(0);
}

int dp_gauss_jordan(double *matrix,dimension_t size)
{
	int indxc[MAXSIZE];
	int indxr[MAXSIZE];
	int ipiv[MAXSIZE];
	/* These used to be dimension_t, but l (at least) needs to be able to go negative... */
	incr_t l;
	dimension_t i,j,k;
	dimension_t icol=0, irow=0;	/* initialize to elim compiler warning */
	incr_t rowdex, rowdex2;
	dimension_t ll;
	double big, tmp, pivinv;


	for(j=0;j<size;j++) ipiv[j]=0.0;

	for(i=0;i<size;i++){	/* main loop over columns to be reduced */
		big=0.0;
		for(j=0;j<size;j++)	/* outer loop of pivot search */
			/* is this one an index??? */
			if( ipiv[j] != 1 )
				for(k=0;k<size;k++){
					if( ipiv[k] == 0 ){
						if((tmp=fabs(matrix[j*size+k]))
						>= big ){
							big=tmp;
							irow=j;
							icol=k;
						}
					} else if( ipiv[k] > 1 ){
						NWARN("G-J:  singular, returning -1");
						return(-1);
					}
				}
		++(ipiv[icol]);

	/* We now have the pivot element, so we interchange rows
	 * (if needed) to put the pivot element on the diagonal.
	 * The columns are not physically interchanged, only relabeled;
	 * indx[i], the column of the ith pivot element,
	 * is in the ith column that is reduced,
	 * while indxr[i] is the row in which that pivot element was
	 * originally located.  If indxr[i]!=indxc[i] there is an
	 * implied column interchange.  With this form of bookkeeping,
	 * the solution b's will end up in the correct order, and the
	 * inverse matrix will be scrambled by columns.
	 */

		if( irow != icol ){
			for(l=0;l<(incr_t)size;l++)
				SWAP(matrix[irow*size+l],matrix[icol*size+l])
		}
		indxr[i]=irow;
		indxc[i]=icol;
		if( (tmp=matrix[icol*size+icol]) == 0.0 ){
			NWARN("G-J:  singular matrix");
			return(-1);
		}
		pivinv = 1/tmp;
		matrix[icol*size+icol]=1.0;
		rowdex = icol * size;
		for(l=0;l<(incr_t)size;l++)
			matrix[rowdex+l] *= pivinv;
		/* now reduce the rows (except the pivot one) */
		for(ll=0;ll<size;ll++){
			rowdex2=ll*size;
			if( ll != icol ){
				tmp=matrix[rowdex2+icol];
				matrix[rowdex2+icol]=0.0;
				for(l=0;l<(incr_t)size;l++)
					matrix[rowdex2+l] -=
						matrix[rowdex+l]*tmp;
			}
		}
	}
	/* end of main loop over columns of the reduction */

	/* it remains only to unscramble the solution in view
	 * of the column interchanges.  We do this by interchanging
	 * pairs of columns in the reverse order that the permutaion
	 * was built up
	 */

	for(l=size-1;l>=0;l--){
		if( indxr[l] != indxc[l] ){
			for(k=0;k<size;k++){
				rowdex=k*size;
			SWAP(matrix[rowdex+indxr[l]],matrix[rowdex+indxc[l]])
			}
		}
	}

	return(0);
}

