#include "quip_config.h"

char VersionId_gui_crt_cal[] = QUIP_VERSION_STRING;

/*
 * Calibrate gamma & tau for workstation screen
 *
 *  Check windows:
 *	compare halftone ramp to continuous tone ramp
 *	compare contrast modulated checkerboard to continuous tone
 *
 *  Match window:
 *	match continuous tone to halftone pattern
 */

#include <stdio.h>

#include <math.h>

#include "cmds.h"
#include "howmuch.h"

/*
 *  Color map allocation:
 */

#define CMAP_BASE	8

#define HT_BASE		CMAP_BASE			/* 8 */
#define HT_DARK		HT_BASE
#define HT_LIGHT	(HT_BASE+1)
#define N_HT_COLORS	2

#define N_RAMP_LEVELS	32
#define RAMP_MODEL_BASE	(HT_BASE+N_HT_COLORS)			/* 10 */
#define RAMP_MATCH_BASE	(RAMP_MODEL_BASE+N_RAMP_LEVELS)		/* 42 */

#define N_CMOD_LEVELS	(N_RAMP_LEVELS)

#define N_MEAN_COLORS	N_CMOD_LEVELS
#define MEAN_MODEL_BASE	(RAMP_MATCH_BASE+N_RAMP_LEVELS)		/* 74 */
#define MEAN_MATCH_BASE	(MEAN_MODEL_BASE+N_MEAN_COLORS)		/* 106 */

#define CMOD_BASE	(MEAN_MATCH_BASE+N_MEAN_COLORS)		/* 138 */
#define N_CMOD_COLORS	(2*N_CMOD_LEVELS)

/*
 *	total colors: 2+32+32+32+32+64=194
 */

static int n_cmod_levels=N_CMOD_LEVELS;
static int n_ramp_levels=N_RAMP_LEVELS;

static double mean_level[3][N_CMOD_LEVELS];

static double ramp_level[3][N_RAMP_LEVELS];
static double rate_tbl[N_RAMP_LEVELS];

static int ramp_match_data[3][N_RAMP_LEVELS];

/* mean match data needs both the matches, and the programmed gray levels */
static int mean_match_data[3][N_CMOD_LEVELS];

/*
 * I = k * (v-vzero) ^ gamma	v >= vzero
 * I = 0			v < vzero
 *
 * 0 <= v <= 255
 *
 * Normalizing constant k is chosen so that 0 <= I <= 255
 *
 * v = exp( log(I/k)/gamma ) + vzero
 *
 */

#define I_MAX	255.0
#define I_MIN	0.0
#define I_MID	((I_MAX+I_MIN)/2.0)
#define V_MAX	255.0
#define V_MIN	0.0

#define N_GAMMA	256
static double gamma_tbl[3][N_GAMMA];

#define N_LIN_TBL	256
unsigned char lin_tbl[3][N_LIN_TBL];
static int which_phos=0;

#define SIG_LEN		256
#define FILT_LEN	16
#define OUT_LEN		256

static double signal[SIG_LEN],filter[FILT_LEN],output[OUT_LEN];

#define DEF_TAU		0.02
#define DEF_GAMMA	2.5

static double crt_gamma[3]={DEF_GAMMA,DEF_GAMMA,DEF_GAMMA};
static double tau[3]={DEF_TAU,DEF_TAU,DEF_TAU};
static int vzero[3]={0,0,0};

static int have_a_lintbl[3]={0,0,0};
int lindone=0;

static int disp_phos=(-1);
static int changed[3]={1,1,1};


/* stepit stuff */

static int n_step_params=0;
static int step_phos=0;

/* stepit fortran float type */
#define FTYPE	float
#define MAX_STEP_PARAMS	5

static char *param_name[MAX_STEP_PARAMS];
static FTYPE ans[MAX_STEP_PARAMS];	/* the answers!? */
static FTYPE xmax[MAX_STEP_PARAMS];
static FTYPE xmin[MAX_STEP_PARAMS];
static FTYPE deltx[MAX_STEP_PARAMS];
static FTYPE delmn[MAX_STEP_PARAMS];
static int ntrac=(-1);
#define PREC	.005

#define I_VZERO	0
#define I_GAMMA	1
#define I_TAU	2
#define I_RED_VZERO	I_VZERO
#define I_GRN_VZERO	3
#define I_BLU_VZERO	4

void enter_ramp_match()
{
	int i,p,d;
	int r,g,b,color[3];

	p=howmany("phosphor");
	i=howmany("index");
	d=howmany("match value");

	if( p < 0 || p >= n_ramp_levels ){
		warn("bad phosphor index");
		return;
	}
	if( i < 0 || i >= n_ramp_levels ){
		warn("bad level index");
		return;
	}

	ramp_match_data[p][i] = d;

	r=ramp_match_data[0][i];
	g=ramp_match_data[1][i];
	b=ramp_match_data[2][i];
	set_em(color,r,g,b);
	vpoke(RAMP_MATCH_BASE+i,color[0],color[1],color[2]);
	update_colors();
}

void enter_cmod_match()
{
	int i,p,d;
	int r,g,b,color[3];

	p=howmany("phosphor");
	i=howmany("index");
	d=howmany("match value");

	if( p < 0 || p >= n_ramp_levels ){
		warn("bad phosphor index");
		return;
	}
	if( i < 0 || i >= n_ramp_levels ){
		warn("bad level index");
		return;
	}

	mean_match_data[p][i] = d;

	r=mean_match_data[0][i];
	g=mean_match_data[1][i];
	b=mean_match_data[2][i];
	set_em(color,r,g,b);
	vpoke(MEAN_MATCH_BASE+i,color[0],color[1],color[2]);
	update_colors();
}

compute_lintbl(phos)
int phos;
{
	double logi;
	double vmax;
	double intens,v,k;
	int iv;
	int i;

	/* figure out normalizing constant k */

	logi = log(I_MAX);
	vmax = V_MAX-vzero[phos];

	k = exp( log(I_MAX) - crt_gamma[phos]*log(V_MAX-vzero[phos]) );

#ifdef CAUTIOUS
	/* now check to make sure it's right! */
	vmax = k * exp( log(V_MAX-vzero[phos]) * crt_gamma[phos] );
	fprintf(stderr,"max intensity = %f\n",vmax);
#endif CAUTIOUS

	/* compute gamma table */

	for(i=0;i<N_GAMMA;i++){
		v=i;
		if( v <= vzero[phos] ) gamma_tbl[phos][i] = 0.0;
		else gamma_tbl[phos][i] = k * exp( log(v-vzero[phos]) * crt_gamma[phos] );
	}

/*
 * v = exp( log(I/k)/gamma ) + vzero
 */

	lin_tbl[phos][0]=0;
	for(i=1;i<N_LIN_TBL;i++){
		intens = (I_MAX*i)/(N_LIN_TBL-1);
		intens /= k;
		logi = log(intens);
		logi /= crt_gamma[phos];
		v = exp(logi);
		v += vzero[phos];
		iv = v;
		if( iv < 0 || iv > 255 )
		  fprintf(stderr,"computed voltage %d out of range: %d\n",i,iv);
		lin_tbl[phos][i]=iv;
	}
	have_a_lintbl[phos]=1;

	if( have_a_lintbl[0] && have_a_lintbl[1] && have_a_lintbl[2] )
		lindone=1;
}

compute_color_map(phos)
int phos;
{
	if( !changed[phos] ) return;

#ifdef DEBUG
if( debug )
	advise("computing linearizing tables");
#endif DEBUG
	compute_lintbl(phos);

#ifdef DEBUG
if( debug )
	advise("computing comtrast ramp");
#endif DEBUG
	compute_contrast_ramp(phos);

#ifdef DEBUG
if( debug )
	advise("computing gray ramp");
#endif DEBUG
	compute_gray_ramp(phos);

	changed[phos]=0;
}

double compute_mean_intens(phos,rate,min,max)
int phos; double rate,min,max;
{
	int i,iv;
	double sum;
	static double old_tau=(-1);

	make_pulse_train(signal,SIG_LEN,rate,min,max);

	if( tau[phos] != old_tau ){
		make_impulse(filter,FILT_LEN,tau[phos]);
		old_tau=tau[phos];
	}

	s_convolve(output,OUT_LEN,signal,SIG_LEN,filter,FILT_LEN);

	/* map output through gamma table */

	for(i=0;i<OUT_LEN;i++){
		iv = output[i];
		if( iv < 0 || iv >= N_GAMMA ){
			fprintf(stderr,"voltage out of range:  %d\n",iv);
		} else {
			output[i] = gamma_tbl[phos][iv];
		}
	}

	/* now take the mean */

	sum=0.0;
	for(i=0;i<OUT_LEN;i++) sum += output[i];
	sum /= OUT_LEN;

	return(sum);
}

compute_contrast_ramp(phos)
{
	double contrast,vmin,vmax,imin,imax;
	int i,index1,index2;

	/* compute the averages for the contrast ramp */

	/*
	 * don't bother to do contrast 0 
	 */

	for(i=0;i<n_cmod_levels;i++){
		contrast = ((double)(i+1))/((double)n_cmod_levels);

		imin = I_MID * (1-contrast);
		imax = I_MID * (1+contrast);

		index1 = (imin/I_MAX)*(N_LIN_TBL-1);
		index2 = (imax/I_MAX)*(N_LIN_TBL-1);

		/*
		vmin = lin_tbl[phos][index1];
		vmax = lin_tbl[phos][index2];

		cmod_color_pairs[phos][i*2]=vmin;
		cmod_color_pairs[phos][i*2+1]=vmax;

		mean_level[phos][i] = compute_mean_intens(phos,0.5,vmin,vmax);
		*/

		mean_level[phos][i] =
		compute_mean_intens(phos,0.5,(double)index1,(double)index2);
	}
}

compute_gray_ramp(phos)
{
	double rate,vmin,vmax,imin,imax;
	int i,index;

	/* compute the averages for the contrast ramp */
	/*
	 * don't bother to do contrasts 0 & 1
	 * that makes n+1 steps
	 */

	for(i=0;i<n_ramp_levels;i++){
		ramp_level[phos][i] =
			compute_mean_intens(phos,rate_tbl[i],V_MIN,V_MAX);
	}
}

linear_ramp()		/* the default */
{
	int i;

	for(i=0;i<n_ramp_levels;i++)
		rate_tbl[i] = ((double)(i+1))/((double)(n_ramp_levels+1));
}

make_pulse_train(buf,n,rate,min,max)
double *buf, rate,min,max;
int n;
{
	int i;
	double mean, want;

	want = rate*max + (1-rate)*min;
	mean=0;

	for(i=0;i<n;i++){
		if( mean < want ) buf[i]=max;
		else buf[i]=min;

		/* update mean */
		mean *= i;
		mean += buf[i];
		mean /= i+1;
	}
}

make_impulse(buf,n,tau)
double *buf,tau; int n;
{
	int i;
	double x;
	double sum;

	sum=0.0;
	for(i=0;i<n;i++){
		x=i/tau;
		buf[i] = exp(-x);
		sum += buf[i];
	}

	/* normalize */
	for(i=0;i<n;i++) buf[i] /= sum;
}

s_convolve(output,on,signal,sn,filter,fn)
double *output, *signal, *filter;
int on, sn, fn;
{
	int i,j;

	for(i=0;i<on;i++){
		output[i]=0.0;
		for(j=fn-1;j>=0;j--){
			if( i-j < 0 ) continue;
			if( i-j >= sn ) continue;

			output[i] += filter[j] * signal[i-j];
		}
	}
}

void set_gamma()
{
	int i;
	double g;

	i=howmany("phosphor index");
	g = howmuch("gamma");
	if( g <= 0.0 ){
		warn("gamma must be positive");
		return;
	}

	if( i>=0 && i<3 ){
		crt_gamma[i] = g;
		changed[i]=1;
	} else if( i== -1 ){
		crt_gamma[0] =
		crt_gamma[1] =
		crt_gamma[2] = g;
		changed[0]=changed[1]=changed[2]=1;
	} else warn("bad index");
}

void set_tau()
{
	int i;
	double t;

	i=howmany("phosphor index");
	t=howmuch("tau");
	if( t <= 0.0 ){
		warn("tau must be positive");
		return;
	}

	if( i>=0 && i<3 ){
		tau[i] = t;
		changed[i]=1;
	} else if( i== -1 ){
		tau[0] =
		tau[1] =
		tau[2] = t;
		changed[0]=changed[1]=changed[2]=1;
	} else warn("bad index");
}

void set_vzero()
{
	int i;

	i=howmany("phosphor index");
	if( i>=0 && i<3 ){
		vzero[i] = howmany("vzero");
		changed[i]=1;
	} else if( i== -1 ){
		vzero[0] =
		vzero[1] =
		vzero[2] = howmany("vzero");
		changed[0]=changed[1]=changed[2]=1;
	} else warn("bad index");
}

void do_compute()
{
	int p;

	p=howmany("phosphor index (-1 for all three)");

	if( p>=0 && p<= 2)
		compute_color_map(p);

	else if( p == -1 ){
		compute_color_map(0);
		compute_color_map(1);
		compute_color_map(2);
	}
}

void dump_gray()
{
	int i;

	printf("\nComputed levels\t\tMatch Data\n\n");
	for(i=0;i<n_ramp_levels;i++)
		printf("%f\t%f\t%f\t%d\t%d\t%d\n",
			ramp_level[0][i], ramp_level[1][i], ramp_level[2][i],
			ramp_match_data[0][i], ramp_match_data[1][i],
			ramp_match_data[2][i]);
}

void dump_cont()
{
	int i;

	for(i=0;i<n_cmod_levels;i++)
		printf("%f\t%f\t%f\n",
			mean_level[0][i], mean_level[1][i], mean_level[2][i] );
}

void dump_filter()
{
	int i;

	for(i=0;i<FILT_LEN;i++)
		printf("%f\n",filter[i]);
}

void dump_lintbl()
{
	int i;

	fprintf(stderr,"\nGamma table:\n\n");

	for(i=0;i<N_GAMMA;i++)
		printf("%d\t%f\t%f\t%f\n",
			i, gamma_tbl[0][i], gamma_tbl[1][i], gamma_tbl[2][i] );
	fprintf(stderr,"\nLinearizing table:\n\n");
	for(i=0;i<N_LIN_TBL;i++)
		printf("%d\t%d\t%d\t%d\n",
			i, lin_tbl[0][i], lin_tbl[1][i], lin_tbl[2][i] );
}

default_lintbl()
{
	compute_color_map(0);
	compute_color_map(1);
	compute_color_map(2);
	lindone=1;
}

set_em(buf,r,g,b)
int *buf;
{
	int i;

	buf[0]=r;
	buf[1]=g;
	buf[2]=b;
	if( disp_phos == -1 ) return;
	for(i=0;i<3;i++)
		if( i!=disp_phos ) buf[i]=0;
}

set_ht_levels()
{
	int color[3],r,g,b;

	r=g=b=0;
	set_em(color,r,g,b);
	vpoke( HT_DARK, color[0], color[1], color[2] );
	r=g=b=255;
	set_em(color,r,g,b);
	vpoke( HT_LIGHT, color[0], color[1], color[2] );
}

set_ramp_levels()
{
	int i,color[3],r,g,b;

	for(i=0;i<n_ramp_levels;i++){
		r=ramp_level[0][i]+0.5;
		g=ramp_level[1][i]+0.5;
		b=ramp_level[2][i]+0.5;
		set_em(color,r,g,b);
		setcolor(RAMP_MODEL_BASE+i,color[0],color[1],color[2]);

		r=ramp_match_data[0][i];
		g=ramp_match_data[1][i];
		b=ramp_match_data[2][i];
		set_em(color,r,g,b);
		vpoke(RAMP_MATCH_BASE+i,color[0],color[1],color[2]);
	}
}

void set_color_maps()
{
	int i;

	i=howmany("index of phosphor for display (-1 for all three)");
	if( i < -1 || i > 2 ){
		warn("bad phosphor index");
		return;
	}
	disp_phos=i;

	set_ht_levels();

	set_ramp_levels();

	set_cmod_levels();
}

set_cmod_levels()
{
	int i,color[3],r,g,b;
	double contrast;

	for(i=0;i<(n_cmod_levels);i++){
		contrast = ((double)(i+1))/((double)n_cmod_levels);
		r=g=b=I_MID*(1-contrast)+0.5;
		set_em(color,r,g,b);
		/*
		setcolor(CMOD_BASE+i*2,color[0],color[1],color[2]);
		*/
		vpoke(CMOD_BASE+i*2,color[0],color[1],color[2]);
		r=g=b=I_MID*(1+contrast)+0.5;
		set_em(color,r,g,b);
		/*
		setcolor(CMOD_BASE+i*2+1,color[0],color[1],color[2]);
		*/
		vpoke(CMOD_BASE+i*2+1,color[0],color[1],color[2]);

		r=mean_level[0][i]+0.5;
		g=mean_level[1][i]+0.5;
		b=mean_level[2][i]+0.5;
		set_em(color,r,g,b);
		setcolor(MEAN_MODEL_BASE+i,color[0],color[1],color[2]);

		r=mean_match_data[0][i];
		g=mean_match_data[1][i];
		b=mean_match_data[2][i];
		set_em(color,r,g,b);
		vpoke(MEAN_MATCH_BASE+i,color[0],color[1],color[2]);
	}
}

void set_levels()
{
	int l;

	l=howmany("number of levels");
	if( l<2 || l>N_CMOD_LEVELS || l>N_RAMP_LEVELS ){
		warn("number of levels out of range");
		return;
	}
	n_cmod_levels = n_ramp_levels = l;
}

void set_rates()
{
	int i;

	for(i=0;i<n_ramp_levels;i++){
		rate_tbl[i] = howmuch("ramp level");
	}
}

/* stepit stuff */

residual_error()
{
	char str[128];
	FTYPE	err;
	FTYPE	x[MAX_STEP_PARAMS];
	int i,phos,ie;
	double sos;

	getvals_(x,&n_step_params);		/* get the parameter estimates */
	vzero[step_phos]=x[I_VZERO];
	crt_gamma[step_phos]=x[I_GAMMA];
	tau[step_phos]=x[I_TAU];

fprintf(stderr,"Trying:\t\t%d\t%f\t%f",vzero[step_phos],crt_gamma[step_phos],
tau[step_phos]);

	compute_lintbl(step_phos);
	compute_gray_ramp(step_phos);
	compute_contrast_ramp(step_phos);

	set_ramp_levels();
	set_cmod_levels();
	update_colors();

	sos=0.0;
	for(i=0;i<n_ramp_levels;i++){
		int index;

		index = ramp_level[step_phos][i] + 0.5;

		if( index < 0 || index > 255 ){
			warn("bad ramp_level index");
		} else {
			ie = lin_tbl[step_phos][ index ]
				- ramp_match_data[step_phos][i] ;
			sos += ie * ie;
		}
	}


	for(i=0;i<n_cmod_levels;i++){
		int index;

		index = mean_level[step_phos][i] + 0.5;

		if( index < 0 || index > 255 ){
			warn("bad mean_level index");
		} else {
			ie = lin_tbl[step_phos][ index ]
				- mean_match_data[step_phos][i] ;
			sos += ie * ie;
		}
	}
	err = sos;
fprintf(stderr,"\t%f\n",err);

	setfobj_(&err);
}

all_residual_error()
{
	char str[128];
	FTYPE	err;
	FTYPE	x[MAX_STEP_PARAMS];
	int i,p,phos,ie;
	double sos;

	getvals_(x,&n_step_params);		/* get the parameter estimates */
	vzero[0]=x[I_RED_VZERO];
	vzero[1]=x[I_GRN_VZERO];
	vzero[2]=x[I_BLU_VZERO];
	crt_gamma[0]=crt_gamma[1]=crt_gamma[2]=x[I_GAMMA];
	tau[0]=tau[1]=tau[2]=x[I_TAU];

fprintf(stderr,"Trying:\t%d %d %d\t%f\t%f",vzero[0],vzero[1],vzero[2],
crt_gamma[0], tau[0]);

	compute_lintbl(0);
	compute_lintbl(1);
	compute_lintbl(2);
	compute_gray_ramp(0);
	compute_gray_ramp(1);
	compute_gray_ramp(2);
	compute_contrast_ramp(0);
	compute_contrast_ramp(1);
	compute_contrast_ramp(2);

	set_ramp_levels();
	set_cmod_levels();
	update_colors();

	sos=0.0;
	for(p=0;p<3;p++){
		for(i=0;i<n_ramp_levels;i++){
			int index;

			index = ramp_level[p][i] + 0.5;

			if( index < 0 || index > 255 ){
				warn("bad ramp_level index");
			} else {
				ie = lin_tbl[p][ index ]
					- ramp_match_data[p][i] ;
				sos += ie * ie;
			}
		}
		for(i=0;i<n_cmod_levels;i++){
			int index;

			index = mean_level[p][i] + 0.5;

			if( index < 0 || index > 255 ){
				warn("bad mean_level index");
			} else {
				ie = lin_tbl[p][ index ]
					- mean_match_data[p][i] ;
				sos += ie * ie;
			}
		}
	}


	err = sos;
fprintf(stderr,"\t%f\n",err);

	setfobj_(&err);
}

void do_fit()
{
	step_phos=how_many("phosphor to fit");
	if( step_phos < -1 || step_phos > 2 ){
		warn("bad phosphor specification");
		return;
	}
	if( step_phos >= 0 )
		fit_one_phosphor();
	else
		fit_all_phosphors();
}

fit_all_phosphors()
{
	int nfmax;		/* max. # function calls */
	FTYPE	x[MAX_STEP_PARAMS];

	n_step_params=5;

	ans[I_RED_VZERO] = vzero[0];
	xmin[I_RED_VZERO] = 0.0;
	xmax[I_RED_VZERO] = 200.0;
	deltx[I_RED_VZERO] = 10.0;
	delmn[I_RED_VZERO] = 0.05;

	ans[I_GRN_VZERO] = vzero[1];
	xmin[I_GRN_VZERO] = 0.0;
	xmax[I_GRN_VZERO] = 200.0;
	deltx[I_GRN_VZERO] = 10.0;
	delmn[I_GRN_VZERO] = 0.05;

	ans[I_BLU_VZERO] = vzero[2];
	xmin[I_BLU_VZERO] = 0.0;
	xmax[I_BLU_VZERO] = 200.0;
	deltx[I_BLU_VZERO] = 10.0;
	delmn[I_BLU_VZERO] = 0.05;

	ans[I_GAMMA] = crt_gamma[0];
	xmin[I_GAMMA] = 0.1;
	xmax[I_GAMMA] = 4.0;
	deltx[I_GAMMA] = 0.2;
	delmn[I_GAMMA] = 0.005;

	ans[I_TAU] = tau[0];
	xmin[I_TAU] = 0.01;
	xmax[I_TAU] = 20.0;
	deltx[I_TAU] = 0.02;
	delmn[I_TAU] = 0.0005;


	/* copy to fortran */

	setvals_(ans,&n_step_params);
	setminmax_(xmin,xmax,&n_step_params);
	setdelta_(delmn,deltx,&n_step_params);
	settrace_(&ntrac);

	nfmax = 200;
	setmaxcalls_(&nfmax);

	stept_(all_residual_error);

	getvals_(x,&n_step_params);		/* get the parameter estimates */
	vzero[0]=x[I_RED_VZERO];
	vzero[1]=x[I_GRN_VZERO];
	vzero[2]=x[I_BLU_VZERO];
	crt_gamma[0]=crt_gamma[1]=crt_gamma[2]=x[I_GAMMA];
	tau[0]=tau[1]=tau[2]=x[I_TAU];

	printf("\nFit Parameters:\n\n");
	printf("\tvzero\t\t%d %d %d\n",vzero[0],vzero[1],vzero[2]);
	printf("\tcrt_gamma\t\t%f\n",crt_gamma[0]);
	printf("\ttau\t\t%f\n",tau[0]);
}

fit_one_phosphor()
{
	int nfmax;		/* max. # function calls */
	FTYPE	x[MAX_STEP_PARAMS];

	n_step_params=3;

	ans[I_VZERO] = vzero[step_phos];
	xmin[I_VZERO] = 0.0;
	xmax[I_VZERO] = 200.0;
	deltx[I_VZERO] = 10.0;
	delmn[I_VZERO] = 0.05;

	ans[I_GAMMA] = crt_gamma[step_phos];
	xmin[I_GAMMA] = 0.1;
	xmax[I_GAMMA] = 4.0;
	deltx[I_GAMMA] = 0.2;
	delmn[I_GAMMA] = 0.005;

	ans[I_TAU] = tau[step_phos];
	xmin[I_TAU] = 0.01;
	xmax[I_TAU] = 20.0;
	deltx[I_TAU] = 0.02;
	delmn[I_TAU] = 0.0005;


	/* copy to fortran */

	setvals_(ans,&n_step_params);
	setminmax_(xmin,xmax,&n_step_params);
	setdelta_(delmn,deltx,&n_step_params);
	settrace_(&ntrac);

	nfmax = 200;
	setmaxcalls_(&nfmax);

	stept_(residual_error);

	getvals_(x,&n_step_params);		/* get the parameter estimates */
	vzero[step_phos]=x[I_VZERO];
	crt_gamma[step_phos]=x[I_GAMMA];
	tau[step_phos]=x[I_TAU];

	printf("\nFit Parameters:\n\n");
	printf("\tvzero\t\t%d\n",vzero[step_phos]);
	printf("\tcrt_gamma\t\t%f\n",crt_gamma[step_phos]);
	printf("\ttau\t\t%f\n",tau[step_phos]);
}

static char *match_choices[]={"ramp","cmod"};

void save_matches(SINGLE_QSP_ARG_DECL)
{
	FILE *fp;
	int t,i;

	fp=TRYNICE( nameof("filename"), "w" );

	if( !fp ) return;

	for(i=0;i<n_ramp_levels;i++)
		fprintf(fp,"%d\t%d\t%d\n",
			ramp_match_data[0][i],
			ramp_match_data[1][i],
			ramp_match_data[2][i]);
	for(i=0;i<n_cmod_levels;i++)
		fprintf(fp,"%d\t%d\t%d\n",
			mean_match_data[0][i],
			mean_match_data[1][i],
			mean_match_data[2][i]);
}

void read_matches()
{
	FILE *fp;
	int t,i,p;
	int d[3];

	p=howmany("phosphor index");
	fp=try_open( nameof("filename"), "r" );
	if( !fp ) return;

	for(i=0;i<n_ramp_levels;i++){
		if( fscanf(fp,"%d %d %d",&d[0],&d[1],&d[2]) != 3 ){
			warn("error reading data file");
			goto bad;
		}
		ramp_match_data[p][i] = d[p];
	}
	for(i=0;i<n_cmod_levels;i++){
		if( fscanf(fp,"%d %d %d",&d[0],&d[1],&d[2]) != 3 ){
			warn("error reading data file");
			goto bad;
		}
		mean_match_data[p][i] = d[p];
	}
bad:
	fclose(fp);
}

Command cal_ctbl[]={
	"gamma",set_gamma,"set gamma parameter",
	"tau",set_tau,"set tau parameter",
	"vzero",set_vzero,"set vzero parameter",
	"compute",do_compute,"compute color maps",
	"install",set_color_maps,"install color map for calibration",
	"ramp_match",enter_ramp_match,"enter data for one ramp match",
	"cmod_match",enter_cmod_match,"enter data for one cmod match",
	"gray_ramp",dump_gray,"dump grayscale ramp data to screen",
	"cont_ramp",dump_cont,"dump contrast ramp data to screen",
	"lintbl",dump_lintbl,"dump linearizing table to screen",
	"levels",set_levels,"set number of levels",
	"read",read_matches,"read match data from a file",
	"save",save_matches,"save match data to a file",
	"rates",set_rates,"specify ramp levels in test image",
	"fit",do_fit,"estaimate parameters from match data",
	"filter",dump_filter,"dump exponental impulse response",
	"quit",popcmd,"exit submenu",
	NULL,NULL,NULL
};

void cal_menu()
{
	pushcmd(cal_ctbl,"calib");
}

