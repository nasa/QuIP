#include "quip_config.h"

#include <stdio.h>
#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "quip_prot.h"
#include "data_obj.h"
#include "ctone.h"
#include "vec_util.h"
#include "cie.h"

				/* we multiply relative luminances by
					lumscal to get machine settings */
				/* we divide machine settings by lumscal
					to get rel. luminances */
#include "phosmax.h"
/*extern float maxlum; */

float lumscal[3];

static int install_white(SINGLE_QSP_ARG_DECL);

/* r-b coordinates of phosphors */
/* where do these numbers come from??? */
#define X1	0.825101f
#define Y1	0.001364f
#define X2	0.619213f
#define Y2	0.002498f
#define X3	0.535606f
#define Y3	0.157105f

/* phosphor to opponent (cardinal direction) matrix */
static float p2o_mat[3][3];
static float o2p_mat[3][3];

/* phosphor to chromaticity matrix */
/* this takes rgb (in frac. lum. units) to r,b chromaticities, 1 */

static float p2c_mat[3][3]={
	{	X1,	X2,	X3	},
	{	Y1,	Y2,	Y3	},
	{	1.0,	1.0,	1.0	}
};

/* chromaticity to phosphors matrix */
/* need to invert this one before using!!! */

static float c2p_mat[3][3];

static Data_Obj *o2p_dp=NULL;
static Data_Obj *p2o_dp=NULL;
static Data_Obj *c2p_dp=NULL;
static Data_Obj *p2c_dp=NULL;

int know_white=0;
int know_lumscal=0;


/* transformation from rgb to chromaticities is a projective transformation
 */

COMMAND_FUNC( set_matrices )
{
	c2p_dp=pick_obj("matrix for chromaticity to phosphor transformation");
	p2c_dp=pick_obj("matrix for phosphor to chromaticity transformation");
	o2p_dp=pick_obj("matrix for opponent to phosphor transformation");
	p2o_dp=pick_obj("matrix for phosphor to opponent transformation");

	install_white(SINGLE_QSP_ARG);	/* BUG should be renamed */
}

static int init_matrices(SINGLE_QSP_ARG_DECL)
{
	float *ptr;
	int i,j;

	if( c2p_dp == NULL ){
		warn("init_matrices: need to specify object for c2p_mat");
		return(-1);
	}
	if( p2c_dp == NULL ){
		warn("init_matrices: need to specify object for p2c_mat");
		return(-1);
	}
	ptr = (float *)OBJ_DATA_PTR(p2c_dp);
	for(i=0;i<3;i++)
		for(j=0;j<3;j++)
			*ptr++ = p2c_mat[i][j];
	dp_copy(c2p_dp,p2c_dp);
	dt_invert(c2p_dp);
	ptr = (float *)OBJ_DATA_PTR(c2p_dp);
	for(i=0;i<3;i++)
		for(j=0;j<3;j++)
			c2p_mat[i][j] = *ptr++;
	return(0);
}

/* we want a linear trans which approximates this about the white point */

/* somehow we need to specify the white direction for the lum basis vec... */

/* now, for a linear space about the white point, we define
  delta = color-white
  Mp2o del_r = 1,0,0
  Mp2o del_b = 0,1,0
  Mp2o white = 0,0,1

	How do we get del_r, del_b??
	We define these to be rgb vectors corresponding to unit changes
	in the chromaticites at the white luminance.
*/

#ifdef NOT_USED
void show_mat( float matrix[3][3], char *name )
{
	printf("%s:\n\t%f\t%f\t%f\n", name,
		matrix[0][0], matrix[0][1], matrix[0][2]);
	printf("\t%f\t%f\t%f\n",
		matrix[1][0], matrix[1][1], matrix[1][2]);
	printf("\t%f\t%f\t%f\n",
		matrix[2][0], matrix[2][1], matrix[2][2]);
}
#endif /* NOT_USED */

#define rgb_norm(vec) _rgb_norm(QSP_ARG  vec)

static float _rgb_norm(QSP_ARG_DECL  float *vec)	/* un-normalized rgb */
{
	int i;
	float totlum;

#ifdef CAUTIOUS
	if( !know_white )
		error1("CAUTIOUS:  rgb_norm:  don't know white!?");

	if( !know_lumscal )
		error1("CAUTIOUS:  rgb_norm:  don't know lumscal!?");
#endif /* CAUTIOUS */

	/* convert rgb settings to relative rgb luminances, get total lum */

advise("lumscal");
showvec(lumscal);
	totlum=0.0;
	for(i=0;i<3;i++){
		vec[i]/=lumscal[i];
		/* take abs so we can normalize color vectors */
		totlum+=fabs(vec[i]);
	}

	/* convert to fractional luminances */

	if( totlum == 0.0 ){
		vec[0]=vec[1]=vec[2]=0.0;
		return(0.0);
	}
	for(i=0;i<3;i++) vec[i] /= totlum;

	/* vec now contains fractional luminance from each gun */

	return(totlum);
}

static void apply_mat3x3(float *vec,float mat[3][3])
{
	float tvec[3];
	int i;

	for(i=0;i<3;i++){
		tvec[i] = mat[i][0] * vec[0];
		tvec[i] += mat[i][1] * vec[1];
		tvec[i] += mat[i][2] * vec[2];
	}
	for(i=0;i<3;i++)
		vec[i]=tvec[i];
}

/* convert rgb settings to rb chromaticities and relative luminance */

#define rgb2rb(vec) _rgb2rb(QSP_ARG  vec)

static void _rgb2rb(QSP_ARG_DECL  float *vec)			/** returns luminance in vec[2] */
{
	float totlum;

#ifdef CAUTIOUS
	if( !know_white )
		error1("CAUTIOUS:  rgb2rb:  don't know white!?");
#endif /* CAUTIOUS */

showvec(vec);
	totlum=rgb_norm(vec);
sprintf(ERROR_STRING,"rgb2rb:  totlum = %g",totlum);
advise(ERROR_STRING);
	/* vec is now fractional luminances (sum to 1) */
showvec(vec);
advise("p2c_mat");
showvec(p2c_mat[0]);
showvec(p2c_mat[1]);
showvec(p2c_mat[2]);
	apply_mat3x3(vec,p2c_mat);	/* now contains rb and lum */
advise("rgb2rb:  after p2c_mat");
showvec(vec);
	vec[2]=totlum;
showvec(vec);
}

/* convert chromaticity coordinates & luminance to an rgb triple */

#define rb2rgb(vec) _rb2rgb(QSP_ARG  vec)

static void _rb2rgb(QSP_ARG_DECL  float *vec)			/** luminance given in vec[2] */
{
	float totlum;
	int i;

#ifdef CAUTIOUS
	if( !know_white )
		error1("CAUTIOUS:  rb2rgb:  don't know white!?");
#endif /* CAUTIOUS */

	totlum=vec[2];
	vec[2]=1.0;

	apply_mat3x3(vec,c2p_mat);

	/* values are now fractional luminances */

	for(i=0;i<3;i++)
		vec[i]*=totlum;

	/* values are now relative luminances */

	for(i=0;i<3;i++)
		vec[i] *= lumscal[i];

	/* values are now machine units */
}

#define wnorm(vec) _wnorm(QSP_ARG  vec)

static void _wnorm(QSP_ARG_DECL  float *vec)
{
	int j;
	float fact, maxswing, factor[3];

sprintf(ERROR_STRING,"%g %g %g",vec[0],vec[1],vec[2]);
advise(ERROR_STRING);
	for(j=0;j<3;j++){
		maxswing=_white[j];	/* maximum allowable decrement */
		if( (PHOSMAX-_white[j]) < maxswing )
			maxswing=PHOSMAX-_white[j];
		factor[j] = (float) fabs(vec[j]) / maxswing ;
	}
	/* find the biggest factor */
	fact=0.0;
	for(j=0;j<3;j++)
		if( factor[j] > fact ) fact=factor[j];
#ifdef CAUTIOUS
	if( fact == 0.0 )
		error1("CAUTIOUS:  wnorm:  factor is 0!?");
#endif /* CAUTIOUS */
	for(j=0;j<3;j++)
		vec[j] /= fact;
	
}


static int install_white(SINGLE_QSP_ARG_DECL)
{
	int j,k;
	float wc[3];	/* white chromaticity */
	float uv[3];	/* unit vector */
	float *ptr;

	if( init_matrices(SINGLE_QSP_ARG) < 0 ) return(-1);

	/* we assume we already have the white point */
	if( ! know_white )
		error1("install_white:  white point not defined!?");


	for(j=0;j<3;j++) wc[j] = _white[j];
showvec(wc);
	rgb2rb(wc);	/* now we have the chromaticity of the white point */
advise("white transformed to opponent space");
showvec(wc);

	for(j=0;j<3;j++) uv[j] = wc[j];
	uv[0]+= 0.1;	/* take a step in the red cone direction */
	rb2rgb(uv);
	for(j=0;j<3;j++) uv[j] -= _white[j];

	/* now have an rgb vector for an red cone step */
	/* normalize this relative to the white point */
	/* this is to guarantee that amp. of +-1 won't overflow */
advise("red cone vector");
showvec(wc);

	wnorm(uv);

	/* now set the entries of the o2p matrix */
	for(j=0;j<3;j++) o2p_mat[j][0] = uv[j];

	for(j=0;j<3;j++) uv[j] = wc[j];
	uv[1]+= 0.1;	/* take a step in the blue cone  direction */
	rb2rgb(uv);
	for(j=0;j<3;j++) uv[j] -= _white[j];

	/* now have an rgb vector for a blue cone step */

	wnorm(uv);

	for(j=0;j<3;j++) o2p_mat[j][1] = uv[j];

	for(j=0;j<3;j++) uv[j] = _white[j];
	wnorm(uv);
	for(j=0;j<3;j++) o2p_mat[j][2] = uv[j];

	ptr = (float *)OBJ_DATA_PTR(o2p_dp);
	for(j=0;j<3;j++)
		for(k=0;k<3;k++)
			*ptr++ = o2p_mat[j][k];
	dp_copy(p2o_dp,o2p_dp);
	dt_invert(p2o_dp);
	ptr = (float *)OBJ_DATA_PTR(p2o_dp);
	for(j=0;j<3;j++)
		for(k=0;k<3;k++)
			p2o_mat[j][k] = *ptr++;

	return(0);
}

void rgb2o(QSP_ARG_DECL  float *vec)
{
	if( !know_white ) {
		if( install_white(SINGLE_QSP_ARG) < 0 )
			return;
	}

	apply_mat3x3(vec,p2o_mat);
}

void o2rgb(QSP_ARG_DECL  float *vec)
{
	if( !know_white ) {
		if( install_white(SINGLE_QSP_ARG) < 0 )
			return;
	}
	apply_mat3x3(vec,o2p_mat);
}

COMMAND_FUNC( set_lumscal )
{
	lumscal[0] = (float) HOW_MUCH("relative red luminance");
	lumscal[1] = (float) HOW_MUCH("relative green luminance");
	lumscal[2] = (float) HOW_MUCH("relative blue luminance");
advise("SETTING know_lumscal");
	know_lumscal = 1;
}
