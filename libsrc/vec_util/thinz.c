#include "quip_config.h"

char VersionId_vec_util_thinz[] = QUIP_VERSION_STRING;

#include "data_obj.h"
#include "ggem.h"
#include "vecgen.h"

/* local prototypes */
static int range ( Data_Obj *x,int n,int m);
static int nay8(Data_Obj *im,int i,int j,double val);
/* BUG naming conventions violated (by Sylvain?) */
int crossing_index(Data_Obj *x,int ii,int jj);

int crossing_index(Data_Obj *x,int ii,int jj)
{
/* 	Compute the crossing index for pixel X[ii][jj] and return it	*/
	
	int i,j,count;
	float *dstp, *k, *tmp;
	long ri;

	if( (ii<=0) || (ii>=(int)x->dt_rows-1) || (jj<=0) || (jj>=(int)(x->dt_cols-1)) ){
		/* Should write an error here */
		return -1;
	}
	count = 0;
	i = ii-1; j = jj-1; k = (float *)x->dt_data;
	k += i*x->dt_rowinc + j*x->dt_pinc;
	tmp = k;
/*	Move clockwise around the (ii,jj) pixel, couting level changes	*/

	ri = x->dt_rowinc;
	dstp = (float *)x->dt_data;
	tmp = tmp+1;	/* Move to (i-1,j)      */  
	if (*k != *tmp) 	{ k = tmp; count++; }
	tmp = tmp+1;	/* Move to (i-1,j+1)    */
	if (*k != *tmp) 	{ k = tmp; count++; }
	tmp = tmp+ri;	/* Move to (i,j+1)      */
	if (*k != *tmp) 	{ k = tmp; count++; }
	tmp = tmp+ri;	/* Move to (i+1,j+1)	*/
	if (*k != *tmp) 	{ k = tmp; count++; }
	tmp = tmp-1;	/* Move to (i+1,j)	*/
	if (*k != *tmp) 	{ k = tmp; count++; }
	tmp = tmp-1;	/* Move to (i+1,j-1)	*/
	if (*k != *tmp) 	{ k = tmp; count++; }
	tmp = tmp-ri;	/* Move to (i,j-1)	*/
	if (*k != *tmp) 	{ k = tmp; count++; }
	tmp = tmp-ri;	/* Move to (i-1,j-1)	*/
	if (*k != *tmp) 	{ k = tmp; count++; }
	return count/2;
}



static int range (Data_Obj *dp,int n,int m)
{
/* Return 1 if (n,m) are legal (row, column) indices for image X */
	if (n < 0 || n >= (int)dp->dt_rows) return 0;
	if (m < 0 || m >= (int)dp->dt_cols) return 0;
	return 1;
}

static int nay8(Data_Obj *im,int i,int j,double val)
/* Return the number of 8-connected neighbors of (i,j) having value VAL */
{
	int n,m,k;
	float *tp;
	long ri,ci;
	tp = (float *)im->dt_data;
	tp += i*im->dt_rowinc + j*im->dt_pinc;
	if (*tp != val) return 0;
	k = 0;
	for (n=-1; n<=1; n++){
	  	for (m=-1; m<=1; m++){
			tp = (float *)im->dt_data;
			tp += i*im->dt_rowinc + j*im->dt_pinc;
			ri = n*im->dt_rowinc;
			ci = m*im->dt_pinc;
			tp += ri + ci;
	  		if (range(im,i+n,j+m))
		  	  if (*tp == val) k++;
		  }
	}
	return k-1;
}


/* Zhang-Suen type of thinning procedure. This thin the region value */
void thinzs (QSP_ARG_DECL  Data_Obj *x,double value)
{
	Data_Obj *y;
	dimension_t i,j;
	int n,again;
	long ri;
	float *dstp, *srcp;
	float bckgrd;
	
	if( MACHINE_PREC(x) != PREC_SP){
		WARN("precision must be float for squeletonization");
		return;
	}
	bckgrd = 0.0;
	y = dup_obj(QSP_ARG  x,"x.bak");
	dp_copy(QSP_ARG  y,x);
	do{
		again = 0;
		for (i=1; i<(x->dt_rows-1); i++){
			for (j=1; j<(x->dt_cols-1); j++){
				ri = y->dt_rowinc;
				dstp=(float *)x->dt_data;
				srcp=(float *)y->dt_data;
				dstp += i*x->dt_rowinc + j*x->dt_pinc;
				srcp += i*y->dt_rowinc + j*y->dt_pinc;
				if (*srcp != value) continue;
				n = nay8(y,i,j,value);
				if( (n>=2) && (n<=6) ){
					if (crossing_index(y,i,j)==1) {
				  		if( ( *(srcp-ri) == bckgrd ) ||
				     		    ( *(srcp+1) == bckgrd ) ||
				     		    ( *(srcp+ri) == bckgrd ) ) {
				    			if( ( *(srcp+1) == bckgrd ) ||
				       			    ( *(srcp+ri) == bckgrd ) ||
				       			    ( *(srcp-1) == bckgrd ) ) {
								*dstp = bckgrd;
				       				again = 1;
				       			}
				  		}
					} 
				} 
			}
		}
		dp_copy(QSP_ARG  y,x);
		ri = x->dt_rowinc;

		for (i=1; i<(x->dt_rows-1); i++){
			for (j=1; j<(x->dt_cols-1); j++){
			dstp=(float *)x->dt_data;
			srcp=(float *)y->dt_data;
			dstp += i*x->dt_rowinc + j*x->dt_pinc;
			srcp += i*y->dt_rowinc + j*y->dt_pinc;
				if (*dstp != value) continue;
				n = nay8(x,i,j,value);
				if( (n>=2) && (n<=6) ){
					if (crossing_index(x,i,j)==1)  {
			  			if( ( *(dstp-ri) == bckgrd ) ||
			     			    ( *(dstp+1) == bckgrd ) ||
			     			    ( *(dstp-1) == bckgrd ) ) {
			    				if( ( *(dstp-ri) == bckgrd ) ||
			       				    ( *(dstp+ri) == bckgrd ) ||
			       				    ( *(dstp-1) == bckgrd ) ) {
			       					*srcp = bckgrd;
			       					again = 1;
			       				}
			  			}
					}
				}
			}
		}
		dp_copy (QSP_ARG  x,y);
	} 
	while (again);
	delvec(QSP_ARG  y);
}



