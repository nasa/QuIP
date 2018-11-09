#include "quip_config.h"

#include "quip_prot.h"
#include "data_obj.h"
#include "ggem.h"
#include "veclib/vecgen.h"
#include "vec_util.h"

static int crossing_index(Data_Obj *x,int ii,int jj)
{
/* 	Compute the crossing index for pixel X[ii][jj] and return it	*/
	
	int i,j,count;
	//float *dstp;
    float *k, *tmp;
	long ri;

	if( (ii<=0) || (ii>=(int)OBJ_ROWS(x)-1) || (jj<=0) || (jj>=(int)(OBJ_COLS(x)-1)) ){
		/* Should write an error here */
		return -1;
	}
	count = 0;
	i = ii-1; j = jj-1; k = (float *)OBJ_DATA_PTR(x);
	k += i*OBJ_ROW_INC(x) + j*OBJ_PXL_INC(x);
	tmp = k;
/*	Move clockwise around the (ii,jj) pixel, couting level changes	*/

	ri = OBJ_ROW_INC(x);
	//dstp = (float *)OBJ_DATA_PTR(x);
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
	if (*k != *tmp) 	{ /*k = tmp;*/ count++; }
	return count/2;
}



static int range (Data_Obj *dp,int n,int m)
{
/* Return 1 if (n,m) are legal (row, column) indices for image X */
	if (n < 0 || n >= (int)OBJ_ROWS(dp)) return 0;
	if (m < 0 || m >= (int)OBJ_COLS(dp)) return 0;
	return 1;
}

static int nay8(Data_Obj *im,int i,int j,double val)
/* Return the number of 8-connected neighbors of (i,j) having value VAL */
{
	int n,m,k;
	float *tp;
	long ri,ci;
	tp = (float *)OBJ_DATA_PTR(im);
	tp += i*OBJ_ROW_INC(im) + j*OBJ_PXL_INC(im);
	if (*tp != val) return 0;
	k = 0;
	for (n=-1; n<=1; n++){
	  	for (m=-1; m<=1; m++){
			tp = (float *)OBJ_DATA_PTR(im);
			tp += i*OBJ_ROW_INC(im) + j*OBJ_PXL_INC(im);
			ri = n*OBJ_ROW_INC(im);
			ci = m*OBJ_PXL_INC(im);
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
	
	if( OBJ_MACH_PREC(x) != PREC_SP){
		WARN("precision must be float for squeletonization");
		return;
	}
	bckgrd = 0.0;
	y = dup_obj(x,"x.bak");
	dp_copy(y,x);
	do{
		again = 0;
		for (i=1; i<(OBJ_ROWS(x)-1); i++){
			for (j=1; j<(OBJ_COLS(x)-1); j++){
				ri = OBJ_ROW_INC(y);
				dstp=(float *)OBJ_DATA_PTR(x);
				srcp=(float *)OBJ_DATA_PTR(y);
				dstp += i*OBJ_ROW_INC(x) + j*OBJ_PXL_INC(x);
				srcp += i*OBJ_ROW_INC(y) + j*OBJ_PXL_INC(y);
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
		dp_copy(y,x);
		ri = OBJ_ROW_INC(x);

		for (i=1; i<(OBJ_ROWS(x)-1); i++){
			for (j=1; j<(OBJ_COLS(x)-1); j++){
			dstp=(float *)OBJ_DATA_PTR(x);
			srcp=(float *)OBJ_DATA_PTR(y);
			dstp += i*OBJ_ROW_INC(x) + j*OBJ_PXL_INC(x);
			srcp += i*OBJ_ROW_INC(y) + j*OBJ_PXL_INC(y);
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
		dp_copy(x,y);
	} 
	while (again);
	delvec(y);
}



