#include "quip_config.h"

/*
 * Maximum liklihood Weibull fits
 * See appendix in Watson, "Probability summation over time"
 * Vis. Res. v. 19 pp. 515-522
 */

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "stc.h"
#include "debug.h"

/*
 * Two methods for Weibull fit optimization:
 *
 * Original method called stepit (fortran).
 *
 * For greater portability, we now also support simplex method
 * from numerical recipes in C.
 */

#include "optimize.h"

static double	alpha,		/* threshold parameter */
		beta,		/* slope parameter */
		w_gamma;	/* lower asymptote or chance rate */

#define ALPHA_NAME	"alpha"
#define BETA_NAME	"beta"

#define DELTA		0.01		/* default finger error rate */
#define MIN_DELTA	0.0001		/* minumum finger error rate */
static double error_rate=DELTA;

static Summary_Data_Tbl *the_dtp;

#define ALPHA_INDEX	0
#define BETA_INDEX	1
#define N_WPARMS	2	/* number of variable paramters */

/* local prototypes */
static void weibull_fit(QSP_ARG_DECL  Summary_Data_Tbl *dp, int ntrac);

static float w_likelihood(SINGLE_QSP_ARG_DECL)		/* called from optimize; return likelihood of guess */
{

	float lh=0.0,lhinc;
	int i;
	int ntt,		/* number of total trials */
	    nc;			/* number "correct" */
	float pc,xv;

	float t_alpha, t_beta;	/* trial slope and int */
	Opt_Param *opp;
	int n_xvals;

	/* compute the likelihood for this guess */

	opp=get_opt_param(ALPHA_NAME);
	assert( opp != NULL );

	t_alpha = opp->ans;

	opp=get_opt_param(BETA_NAME);
	assert( opp != NULL );

	t_beta = opp->ans;

	assert(global_xval_dp!=NULL);
	n_xvals = OBJ_COLS(global_xval_dp);
	assert(n_xvals>1);

	for(i=0;i<n_xvals;i++){
		float *xv_p;

		/* calculate theoretical percent correct with this guess */

		if( (ntt=DATUM_NTOTAL(SUMM_DTBL_ENTRY(the_dtp,i)) ) <= 0 )
			continue;

		nc=DATUM_NCORR( SUMM_DTBL_ENTRY(the_dtp,i) );
		xv_p = indexed_data(global_xval_dp,i);
		xv = *xv_p;

		if( xv == 0.0 ) pc = (float) w_gamma;
		else {
			pc = (float) (1 - (1-w_gamma)*exp(-pow( xv/t_alpha, t_beta ) ));
			if( pc > (1-error_rate) ) pc =(float) ( 1-error_rate);
		}

		/* pc is the theoretical % correct at this xval */

		lhinc = (float) (nc * log( pc ) + ( ntt - nc ) * log( 1 - pc ));
		lh -= lhinc;
	}

	/* return the answer */

	return(lh);
}

static void weibull_fit(QSP_ARG_DECL  Summary_Data_Tbl *dp,int ntrac)		/** maximum liklihood fit */
{
	Opt_Param tmp_param;
	Opt_Param *alpha_param_p=NULL;
	Opt_Param *beta_param_p=NULL;
	int n_xvals;
	float *first_xv_p, *last_xv_p, *mid_xv_p, *mid2_xv_p;

	/* initialize global */

	the_dtp = dp;

	/* initialize the parameters */

	delete_opt_params(SINGLE_QSP_ARG);	/* clear any existing parameters */

	assert(global_xval_dp!=NULL);
	n_xvals = OBJ_COLS(global_xval_dp);
	assert(n_xvals>1);

	tmp_param.op_name = ALPHA_NAME;

	mid_xv_p = indexed_data(global_xval_dp,n_xvals/2);
	mid2_xv_p = indexed_data(global_xval_dp,n_xvals/2+1);
	first_xv_p = indexed_data(global_xval_dp,0);
	last_xv_p = indexed_data(global_xval_dp,n_xvals-1);

	tmp_param.ans = *mid_xv_p;
	if( *first_xv_p < *last_xv_p ){
		tmp_param.maxv =  *last_xv_p;
		tmp_param.minv =  *first_xv_p;
	} else {
		tmp_param.maxv = *first_xv_p;
		tmp_param.minv = *last_xv_p;
	}
	if( tmp_param.minv < 0.0 ){
		warn("wiebull fit will blow up for negative x values");
		return;
	}
	tmp_param.delta = (float) fabs( *mid_xv_p - *mid2_xv_p );
	tmp_param.mindel = (float) 1.0e-30;

	alpha_param_p = add_opt_param(&tmp_param);


	tmp_param.op_name = BETA_NAME;
	tmp_param.ans = 2;
	tmp_param.maxv = 10000.0;
	tmp_param.minv = 0.0;
	tmp_param.delta = 0.5;
	tmp_param.mindel = (float) 1.0e-30;

	beta_param_p = add_opt_param(&tmp_param);

	if( fc_flag ){
		w_gamma = 0.5;
	} else {
		w_gamma = error_rate;
	}


	optimize(w_likelihood);

	alpha_param_p=get_opt_param(ALPHA_NAME);
	assert( alpha_param_p != NULL );

	alpha = alpha_param_p->ans;

	beta_param_p=get_opt_param(BETA_NAME);
	assert( beta_param_p != NULL );

	beta = beta_param_p->ans;

	/* clean up */
	del_opt_param(beta_param_p);
	del_opt_param(alpha_param_p);
}

void _w_analyse( QSP_ARG_DECL  Trial_Class *tcp )		/** do a regression on the ith table */
{
	int ntrac=(-1);

	weibull_fit( QSP_ARG  CLASS_SUMM_DTBL(tcp), ntrac );
}

void _weibull_out(QSP_ARG_DECL  Trial_Class * tcp)			/** verbose analysis report */
{
        sprintf(msg_str,"\nTrial_Class %s\n",CLASS_NAME(tcp));
	prt_msg(msg_str);

	sprintf(msg_str,"alpha (threshold):\t\t%f",alpha);
	prt_msg(msg_str);
	sprintf(msg_str,"beta (slope):\t\t%f",beta);
	prt_msg(msg_str);
	/* BUG print out chi-square like statistic */
}

void _w_tersout(QSP_ARG_DECL  Trial_Class * tcp)
{
	sprintf(msg_str,"%s\t%f\t%f",CLASS_NAME(tcp),alpha,beta);
	prt_msg(msg_str);
	/* BUG should print out chi sq stat */
}

void _w_set_error_rate(QSP_ARG_DECL  double er)
{
	if( er < 0 ){
		warn("error rate must be non-negative");
		return;
	} else if( er >= 1 ){
		warn("error rate cannot be >= 1");
		return;
	} else if( er < MIN_DELTA ){
		if( verbose ){
			sprintf(DEFAULT_ERROR_STRING,
	"Setting error rate to minimum permissable value:  %g", MIN_DELTA);
			NADVISE(DEFAULT_ERROR_STRING);
		}
		er = MIN_DELTA;
	}
	error_rate=er;
}


