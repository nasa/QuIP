#include "quip_config.h"

/* Floyd-Steinberg error diffusion */

#include <math.h>

#include "quip_prot.h"
#include "vec_util.h"

#define MAXLVLS	256
/* BUG this should by dynamically allocated, no max val */
#define MAXCOLS	1024

// BUG globals are not thread-safe!?
static int nlevels;
static float quantlevel[MAXLVLS];
static float desired[MAXCOLS];
static float ierror[MAXCOLS];
static u_char image[MAXCOLS];
/* static float qerror[MAXCOLS]; */

static void oddline(dimension_t cols)
{
	u_char i;
	long j;		/* has to be signed because we count down... */
			/* could fix this with do { } while(j>0) */
	/* float orig; */
	float derror;
	float err1,besterr;
	u_char best;

	for(j=(long)cols-1;j>=0;j--){
		best=0;
		besterr=(float)fabs( desired[j] - quantlevel[0] );
		for(i=1;i<nlevels;i++){
			err1=(float)fabs( desired[j] - quantlevel[i] );
			if( err1<besterr ){
				besterr=err1;
				best=i;
			}
		}
		/* orig=image[j]; */
		image[j]=best;
		derror=desired[j]-quantlevel[best];
		if( j>0 ){
			desired[j-1]+=.375*derror;
			ierror[j-1]+=.25*derror;
		}
		ierror[j]+=.375*derror;
		/* qerror[j]=quantlevel[best]-orig; */
	}
}

static void evenline(dimension_t cols)
{
	dimension_t j;
	u_char i;
	/* float orig; */
	float derror;
	float err1;
	float besterr;
	u_char best;

	for(j=0;j<cols;j++){
		/* starting guess is index 0 */
		best=0;
		besterr=(float)fabs( desired[j] - quantlevel[0] );
		for(i=1;i<nlevels;i++){
			err1=(float)fabs( desired[j] - quantlevel[i] );
			if( err1<besterr ){
				besterr=err1;
				best=i;
			}
		}
		/* orig=image[j]; */
		image[j]=best;
		derror=desired[j]-quantlevel[best];
		if( j < (cols-1) ){
			desired[j+1]+=.375*derror;
			ierror[j+1]+=.25*derror;
		}
		ierror[j]+=.375*derror;
		/* qerror[j]=quantlevel[best]-orig; */
	}
}

void dp_halftone(QSP_ARG_DECL  Data_Obj *dpto,Data_Obj *dpfr,dimension_t n,float *levels)
{
	dimension_t i,j;
	dimension_t rows, cols;
	u_char *byteptr, *destptr;
	float *fltptr;

	nlevels=n;
	for(i=0;i<n;i++) quantlevel[i]=levels[i];

	if( (OBJ_ROWS(dpto) != OBJ_ROWS(dpfr))
	  || (OBJ_COLS(dpto) != OBJ_COLS(dpfr)) ){
		WARN("size mismatch");
		return;
	}
	if( OBJ_COLS(dpto) > MAXCOLS ){
		sprintf(ERROR_STRING,"sp_halftone:  Sorry, image %s has %d columns, max is %d",OBJ_NAME(dpto),
			OBJ_COLS(dpto),MAXCOLS);
		WARN(ERROR_STRING);
		return;
	}
	if( OBJ_PREC(dpto) != PREC_UBY ){
		sprintf(ERROR_STRING,"dp_halftone:  target image %s (%s) must be unsigned byte precision",
			OBJ_NAME(dpto),OBJ_PREC_NAME(dpto));
		WARN(ERROR_STRING);
		return;
	}
	/* what about source precision??? */

	cols=OBJ_COLS(dpto);
	rows=OBJ_ROWS(dpto);
	byteptr=(u_char *)OBJ_DATA_PTR(dpfr);
	fltptr=(float *)byteptr;
	destptr = (u_char *)OBJ_DATA_PTR(dpto);

	for(i=0;i<cols;i++) ierror[i]=0.0;
	for(i=0;i<rows;i++){		/* i is row index */
		if( OBJ_PREC(dpfr) == PREC_UBY ){
			for(j=0;j< cols;j++)
				desired[j]= byteptr[j*OBJ_PXL_INC(dpfr)];
			byteptr += OBJ_ROW_INC(dpfr);
		} else if( OBJ_PREC(dpfr) == PREC_SP ){
			for(j=0;j< cols;j++)
				desired[j]=fltptr[j*OBJ_PXL_INC(dpfr)];
			fltptr += OBJ_ROW_INC(dpfr);
		} else {
			WARN("bad source precision; must be float or byte");
			return;
		}

		for(j=0;j< cols;j++){	/* j is column index */
			desired[j]+=ierror[j];
			ierror[j]=0.0;
		}
		if( i & 1 ) oddline(cols);
		else evenline(cols);

		for(j=0;j<cols;j++)
			destptr[j*OBJ_PXL_INC(dpto)] = image[j];
		destptr += OBJ_ROW_INC(dpto);

		if( verbose ){
			sprintf(msg_str,"line %d done",i);
			prt_msg(msg_str);
		}
	}
}

