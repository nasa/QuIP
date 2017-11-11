#include "quip_config.h"

/* print confidence intervals for psychometric functions */


#include <math.h>

#include "stc.h"
#include "debug.h"
#include "optimize.h"

static int n_obs;
static int n_seen;
static float alpha;
#define ALPHA	0.05		/* for 95% conf. interval */

#define TP_NAME	"true_prob"

#define PREC	.005
#define MAXTRIES	20


/* local prototypes */
static void get_bar(QSP_ARG_DECL  int index, float (*func)(SINGLE_QSP_ARG_DECL), float *ptr );
static float u_confidence(SINGLE_QSP_ARG_DECL);
static float l_confidence(SINGLE_QSP_ARG_DECL);

static FTYPE fit_err, pk, num, sum, denom;
static int numfact, denomfact;

static float u_confidence(SINGLE_QSP_ARG_DECL)	/* return sq. deviation from .975 */
{
	int k,j;

	float t_p;		/* trial probability */
	Opt_Param *opp;

	sum = 0.0;

	opp=get_opt_param(TP_NAME);
	assert( opp != NULL );

	t_p = opp->ans;


	if( n_obs > 0 ){
		for(k=n_seen+1;k<=n_obs;k++){
			pk = 1;
			for(j=0;j<k;j++) pk *= t_p;
			for(j=0;j<(n_obs-k);j++) pk *= (1-t_p);
			/* binomial coefficient */
			num = 1;
			denom = 1;
			numfact = n_obs;
			denomfact = k;
			for(j=0;j<k;j++){
				num *= numfact;
				denom *= denomfact;
				numfact --;
				denomfact --;
			}
			pk *= num / denom;
			sum += pk;
		}
	} else WARN("zero observations!?");
	/* sum is the probability of observing n_obs or more */

	fit_err = ((1-alpha)-sum)*((1-alpha)-sum);

	return(fit_err);
}

static float l_confidence(SINGLE_QSP_ARG_DECL)
{

	int k,j;

	float t_p;		/* trial probability */
	Opt_Param *opp;

	sum = 0.0;

	opp=get_opt_param(TP_NAME);
	assert( opp != NULL );

	t_p = opp->ans;

	if( n_obs > 0 ){
		for(k=0;k<n_seen;k++){
			pk = 1;
			for(j=0;j<k;j++) pk *= t_p;
			for(j=0;j<(n_obs-k);j++) pk *= (1-t_p);
			/* binomial coefficient */
			num = 1;
			denom = 1;
			numfact = n_obs;
			denomfact = k;
			for(j=0;j<k;j++){
				num *= numfact;
				denom *= denomfact;
				numfact --;
				denomfact --;
			}
			pk *= num / denom;
			sum += pk;
		}
	}
	/* sum is the probability of observing n_obs or more */

	fit_err = ((1-alpha)-sum)*((1-alpha)-sum);
	return(fit_err);
}

static void get_bar(QSP_ARG_DECL  int index, float (*func)(SINGLE_QSP_ARG_DECL), float *ptr )
{
	Opt_Param op1, *opp=(&op1);

	/* initialize optimization vars */

	delete_opt_params(SINGLE_QSP_ARG);

	opp->op_name = TP_NAME;
	opp->ans= 0.5;
	opp->maxv= 1.0;
	opp->minv= 0.0;
	opp->delta= (float) 0.2;
	opp->mindel= (float) 1.0e-30;

	opp=add_opt_param(QSP_ARG  opp);

	optimize(QSP_ARG  func);

	*ptr = opp->ans;
}

void pnt_bars(QSP_ARG_DECL  FILE *fp, Trial_Class *tcp)
{
        int j;
        Data_Tbl *dtp;
	float upper_bar, lower_bar;

	if( tcp == NULL ) return;

	dtp=CLASS_DATA_TBL(tcp);
	for(j=0;j<_nvals;j++){
		if( DATUM_NTOTAL(DTBL_ENTRY(dtp,j)) > 0 ){
			fprintf(fp,"%f\t", xval_array[ j ]);
			n_obs = DATUM_NTOTAL(DTBL_ENTRY(dtp,j));
			n_seen = DATUM_NCORR(DTBL_ENTRY(dtp,j));
			fprintf(fp,"%f\t",(double) n_seen / (double) n_obs );
			if( n_obs == n_seen ) upper_bar=1.0;
			else {
				if( n_seen == 0 ) alpha = (float) ALPHA;
				else alpha = (float) ALPHA/2;
				get_bar(QSP_ARG  j,u_confidence,&upper_bar);
			}
			if( n_seen == 0 ) lower_bar=0.0;
			else {
				if( n_obs == n_seen ) alpha = (float) ALPHA;
				else alpha = (float) ALPHA/2;
				get_bar(QSP_ARG  j,l_confidence,&lower_bar);
			}
			fprintf(fp,"%f\t%f\n",lower_bar,upper_bar);
		}
	}
	fclose(fp);
}


