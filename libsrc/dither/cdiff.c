#include "quip_config.h"

char VersionId_dither_cdiff[] = QUIP_VERSION_STRING;

/* computation of quantization errors due to scan conversion */

#include <stdio.h>
#include "qlevel.h"
#include "ctone.h"
#include "data_obj.h"
#include "debug.h"
#include "vec_util.h"

#define MAXCOLS	512

float desired[3][MAXCOLS];
static float ierror1[3][MAXCOLS];
static float ierror2[3][MAXCOLS];
static float ierror3[3][MAXCOLS];
unsigned char image[3][MAXCOLS];
float qerror[3][MAXCOLS];	/* the actual pointwise error */

int _ncols;
int _nrows;

float raw[3], rbraw[3], lerr[3], rgerr[3], byerr[3];
int zz;

/* pseudo-impulse responses */

float s1_pir[2][2]={
	{ 	0.0,	0.375	},
	{	0.375,	0.25	}
};

float s2_pir[3][3]={
	{	0.0,	0.0,	0.25	},
	{	0.0,	0.0,	0.2	},
	{	0.25,	0.2,	0.1	}
};

float s3_pir[4][4]={
	{	0.0,	0.0,	0.0,	0.2	},
	{	0.0,	0.0,	0.0,	0.1	},
	{	0.0,	0.0,	0.0,	0.1	},
	{	0.2,	0.15,	0.1,	0.1	}
};

static void get_raw_err(QSP_ARG_DECL  int col)		/* get raw rgb error */
{
	int zz;
	for(zz=0;zz<3;zz++)
		rbraw[zz]=raw[zz]=desired[zz][col]-quant_level[ thebest[zz] ];
	rgb2o(QSP_ARG  rbraw);	/* tranform to r,b,l */
}


static void get_err_comp(QSP_ARG_DECL  float * cvec,int which)
{
	cvec[0]=cvec[1]=cvec[2]=0.0;
	cvec[which]=rbraw[which];
	o2rgb(QSP_ARG  cvec);
}


static void sprd_o1(float * cvec,int col)	/* diffuse with the smallest spread function */
{
	int comp_index;

	for(comp_index=0;comp_index<3;comp_index++){
		ierror1[comp_index][col] += s1_pir[1][0] * cvec[comp_index];
		if( col>0 ){
			desired[comp_index][col-1] += s1_pir[0][1] * cvec[comp_index];
			ierror1[comp_index][col-1] += s1_pir[1][1] * cvec[comp_index];
		}
	}
}

static void sprd_e1(float * cvec,int col)	/* diffuse with the smallest spread function */
{
	int comp_index;

	for(comp_index=0;comp_index<3;comp_index++){
		ierror1[comp_index][col] += s1_pir[1][0] * cvec[comp_index];
		if( col<(_ncols-1) ){
			desired[comp_index][col+1] += s1_pir[0][1] * cvec[comp_index];
			ierror1[comp_index][col+1] += s1_pir[1][1] * cvec[comp_index];
		}
	}
}

static void sprd_o2(float * cvec,int col)	/* diffuse with the middle spread function */
{
	int comp_index;

	for(comp_index=0;comp_index<3;comp_index++){
		ierror1[comp_index][col] +=/*.25*/ s2_pir[1][0] * cvec[comp_index];
		ierror2[comp_index][col] +=/*.125*/ s2_pir[2][0] * cvec[comp_index];
		if( col>0 ){
			desired[comp_index][col-1] +=/*.25*/ s2_pir[0][1] * cvec[comp_index];
			ierror1[comp_index][col-1] +=/*.125*/ s2_pir[1][1] * cvec[comp_index];
			ierror2[comp_index][col-1] +=/*.0625*/ s2_pir[2][1] * cvec[comp_index];
			if( col>1 ){
				desired[comp_index][col-2] += s2_pir[0][2] * cvec[comp_index];
				ierror1[comp_index][col-2] += s2_pir[1][2] * cvec[comp_index];
				ierror2[comp_index][col-2] += s2_pir[2][2] * cvec[comp_index];
			}
		}
	}
}

static void sprd_e2(float * cvec,int col)	/* diffuse with the middle spread function */
{
	int comp_index;

	for(comp_index=0;comp_index<3;comp_index++){
		ierror1[comp_index][col] += s2_pir[1][0] * cvec[comp_index];
		ierror2[comp_index][col] += s2_pir[2][0] * cvec[comp_index];
		if( col<(_ncols-1) ){
			desired[comp_index][col+1] += s2_pir[0][1] * cvec[comp_index];
			ierror1[comp_index][col+1] += s2_pir[1][1] * cvec[comp_index];
			ierror2[comp_index][col+1] += s2_pir[2][1] * cvec[comp_index];
			if( col<(_ncols-2) ){
				desired[comp_index][col+2] += s2_pir[0][2] * cvec[comp_index];
				ierror1[comp_index][col+2] += s2_pir[1][2] * cvec[comp_index];
				ierror2[comp_index][col+2] += s2_pir[2][2] * cvec[comp_index];
			}
		}
	}
}

static void sprd_o3(float * cvec,int col)
{
	int comp_index;

	for(comp_index=0;comp_index<3;comp_index++){
		ierror1[comp_index][col] += s3_pir[1][0] * cvec[comp_index];
		ierror2[comp_index][col] += s3_pir[2][0] * cvec[comp_index];
		ierror3[comp_index][col] += s3_pir[3][0] * cvec[comp_index];
		if( col>0 ){
			desired[comp_index][col-1] += s3_pir[0][1] * cvec[comp_index];
			ierror1[comp_index][col-1] += s3_pir[1][1] * cvec[comp_index];
			ierror2[comp_index][col-1] += s3_pir[2][1] * cvec[comp_index];
			ierror3[comp_index][col-1] += s3_pir[3][1] * cvec[comp_index];
			if( col>1 ){
				desired[comp_index][col-2] += s3_pir[0][2] * cvec[comp_index];
				ierror1[comp_index][col-2] += s3_pir[1][2] * cvec[comp_index];
				ierror2[comp_index][col-2] += s3_pir[2][2] * cvec[comp_index];
				ierror3[comp_index][col-2] += s3_pir[3][2] * cvec[comp_index];
				if( col>2 ){
					desired[comp_index][col-3] += s3_pir[0][3] * cvec[comp_index];
					ierror1[comp_index][col-3] += s3_pir[1][3] * cvec[comp_index];
					ierror2[comp_index][col-3] += s3_pir[2][3] * cvec[comp_index];
					ierror3[comp_index][col-3] += s3_pir[3][3] * cvec[comp_index];
				}
			}
		}
	}
}

static void sprd_e3(float * cvec,int col)
{
	int comp_index;

	for(comp_index=0;comp_index<3;comp_index++){
		ierror1[comp_index][col] += s3_pir[1][0] * cvec[comp_index];
		ierror2[comp_index][col] += s3_pir[2][0] * cvec[comp_index];
		ierror3[comp_index][col] += s3_pir[3][0] * cvec[comp_index];
		if( col<(_ncols-1) ){
			desired[comp_index][col+1] += s3_pir[0][1] * cvec[comp_index];
			ierror1[comp_index][col+1] += s3_pir[1][1] * cvec[comp_index];
			ierror2[comp_index][col+1] += s3_pir[2][1] * cvec[comp_index];
			ierror3[comp_index][col+1] += s3_pir[3][1] * cvec[comp_index];
			if( col<(_ncols-2) ){
				desired[comp_index][col+2] += s3_pir[0][2] * cvec[comp_index];
				ierror1[comp_index][col+2] += s3_pir[1][2] * cvec[comp_index];
				ierror2[comp_index][col+2] += s3_pir[2][2] * cvec[comp_index];
				ierror3[comp_index][col+2] += s3_pir[3][2] * cvec[comp_index];
				if( col<(_ncols-3) ){
					desired[comp_index][col+3] += s3_pir[0][3] * cvec[comp_index];
					ierror1[comp_index][col+3] += s3_pir[1][3] * cvec[comp_index];
					ierror2[comp_index][col+3] += s3_pir[2][3] * cvec[comp_index];
					ierror3[comp_index][col+3] += s3_pir[3][3] * cvec[comp_index];
				}
			}
		}
	}
}

static int not_float(Data_Obj *dp)
{
	if( dp->dt_prec != PREC_SP ){
		sprintf(DEFAULT_ERROR_STRING,"object %s (%s) must have float precision",
			dp->dt_name,name_for_prec(dp->dt_prec));
		NWARN(DEFAULT_ERROR_STRING);
		return(1);
	}
	return(0);
}

void ctoneit(QSP_ARG_DECL  Data_Obj *dst_dp, Data_Obj *src_dp)
{
	dimension_t row_index, col_index, comp_index;
	float cvec[3];
	float *src_ptr;

sprintf(ERROR_STRING,"BEGIN ctoneit, dst = %s",dst_dp->dt_name);
advise(ERROR_STRING);

	if( dst_dp == NO_OBJ || src_dp == NO_OBJ ){
		NWARN("ctoneit:  missing object");
		return;
	}
	if( not_float(dst_dp) ) return;
	if( not_float(src_dp) ) return;

	if( nlevels <= 0 ){
		NWARN("ctoneit:  need to specify number of quantization levels first");
		return;
	}
		
	_ncols = src_dp->dt_cols;
	_nrows = src_dp->dt_rows;

	for(col_index=0;col_index<src_dp->dt_cols;col_index++){
		for(comp_index=0;comp_index<3;comp_index++){
			ierror1[comp_index][col_index]=0.0;
			ierror2[comp_index][col_index]=0.0;
			ierror3[comp_index][col_index]=0.0;
		}
	}
	for(row_index=0;row_index<src_dp->dt_rows;row_index++){		/* k is row index */
		src_ptr = (float *)src_dp->dt_data;
		src_ptr += row_index * src_dp->dt_rowinc;
		/* copy over a line of desired data */
		for(comp_index=0;comp_index<3;comp_index++){
			src_ptr += comp_index * src_dp->dt_cinc;
			/* process this row */
			for(col_index=0;col_index< src_dp->dt_cols;col_index++){
				desired[comp_index][col_index] = *src_ptr;
				desired[comp_index][col_index] += ierror1[comp_index][col_index];
				ierror1[comp_index][col_index] = ierror2[comp_index][col_index];
				ierror2[comp_index][col_index] = ierror3[comp_index][col_index];
				ierror3[comp_index][col_index] = 0.0;
				src_ptr += src_dp->dt_pinc;
			}
		}

		if( row_index & 1 ) {	/* odd line */
			for(col_index=(src_dp->dt_cols-1);col_index>=0;col_index--){

				getbest(QSP_ARG  col_index);
				get_raw_err(QSP_ARG  col_index);

				get_err_comp(QSP_ARG  cvec,2);	/* luma */
				sprd_o1(cvec,col_index);

				get_err_comp(QSP_ARG  cvec,0);	/* chroma, R */
				sprd_o2(cvec,col_index);

				get_err_comp(QSP_ARG  cvec,1);	/* chroma, B */
				sprd_o3(cvec,col_index);

				for(comp_index=0;comp_index<3;comp_index++)
					image[comp_index][col_index]=thebest[comp_index];
			}
		} else {	/* even line */
			for(col_index=0;col_index<src_dp->dt_cols;col_index++){
				getbest(QSP_ARG  col_index);
				get_raw_err(QSP_ARG  col_index);

				get_err_comp(QSP_ARG  cvec,2);
				sprd_e1(cvec,col_index);

				get_err_comp(QSP_ARG  cvec,0);
				sprd_e2(cvec,col_index);

				get_err_comp(QSP_ARG  cvec,1);
				sprd_e3(cvec,col_index);

				for(comp_index=0;comp_index<3;comp_index++)
					image[comp_index][col_index]=thebest[comp_index];
			}
		}
		for(comp_index=0;comp_index<3;comp_index++){
			float *dst_ptr;

			dst_ptr = (float *)dst_dp->dt_data;
			dst_ptr += row_index * dst_dp->dt_rowinc + comp_index * dst_dp->dt_cinc;
			for(col_index=0;col_index<dst_dp->dt_cols;col_index++){
#ifdef DEBUG
if( debug & spread_debug ){
sprintf(ERROR_STRING,"Setting component %d of image %s at %d %d (addr = 0x%lx) to value %d",
comp_index,dst_dp->dt_name,col_index,row_index,(u_long)dst_ptr,image[comp_index][col_index]);
advise(ERROR_STRING);
}
#endif /* DEBUG */
				*dst_ptr = image[comp_index][col_index];
				dst_ptr += dst_dp->dt_pinc;
			}
		}
		fprintf(stderr,"line %d done\n",row_index);
		fflush(stderr);
	}
}


