#include "quip_config.h"

char VersionId_vec_util_dptone[] = QUIP_VERSION_STRING;

/* Floyd-Steinberg error diffusion */

#include <math.h>

#include "vec_util.h"

static int nlevels;
#define MAXLVLS	256
static float quantlevel[MAXLVLS];


/* BUG this should by dynamically allocated, no max val */
#define MAXCOLS	1024

static float desired[MAXCOLS];
static float ierror[MAXCOLS];
/* static float qerror[MAXCOLS]; */
static u_char image[MAXCOLS];

/* local prototypes */
static void oddline(dimension_t cols);
static void evenline(dimension_t cols);

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
		besterr=fabs( desired[j] - quantlevel[0] );
		for(i=1;i<nlevels;i++){
			err1=fabs( desired[j] - quantlevel[i] );
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
		besterr=fabs( desired[j] - quantlevel[0] );
		for(i=1;i<nlevels;i++){
			err1=fabs( desired[j] - quantlevel[i] );
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
	u_long rows, cols;
	u_char *byteptr, *destptr;
	float *fltptr;

	nlevels=n;
	for(i=0;i<n;i++) quantlevel[i]=levels[i];

	if( (dpto->dt_rows != dpfr->dt_rows)
	  || (dpto->dt_cols != dpfr->dt_cols) ){
		WARN("size mismatch");
		return;
	}
	if( dpto->dt_cols > MAXCOLS ){
		sprintf(error_string,"sp_halftone:  Sorry, image %s has %d columns, max is %d",dpto->dt_name,
			dpto->dt_cols,MAXCOLS);
		WARN(error_string);
		return;
	}
	if( dpto->dt_prec != PREC_UBY ){
		sprintf(error_string,"dp_halftone:  target image %s (%s) must be unsigned byte precision",
			dpto->dt_name,name_for_prec(dpto->dt_prec));
		WARN(error_string);
		return;
	}
	/* what about source precision??? */

	cols=dpto->dt_cols;
	rows=dpto->dt_rows;
	byteptr=(u_char *)dpfr->dt_data;
	fltptr=(float *)byteptr;
	destptr = (u_char *)dpto->dt_data;

	for(i=0;i<cols;i++) ierror[i]=0.0;
	for(i=0;i<rows;i++){		/* i is row index */
		if( dpfr->dt_prec == PREC_UBY ){
			for(j=0;j< cols;j++)
				desired[j]= byteptr[j*dpfr->dt_pinc];
			byteptr += dpfr->dt_rowinc;
		} else if( dpfr->dt_prec == PREC_SP ){
			for(j=0;j< cols;j++)
				desired[j]=fltptr[j*dpfr->dt_pinc];
			fltptr += dpfr->dt_rowinc;
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
			destptr[j*dpto->dt_pinc] = image[j];
		destptr += dpto->dt_rowinc;

		if( verbose ){
			sprintf(msg_str,"line %d done",i);
			prt_msg(msg_str);
		}
	}
}

