#include "quip_config.h"

// maximum liklihood fit suggested by DIAM */

/*
 * Fit a normal ogive by doing linear regression on
 * the zscores.
 */

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "stc.h"
#include "debug.h"
#include "optimize.h"
#include "function.h"	// ptoz

#define PREC		.005
#define MAXTRIES	20

// BUG - get rid of these globals!!!

int fc_flag=0;			// forced choice

static Opt_Param *slope_param_p=NULL;
static Opt_Param *intercept_param_p=NULL;
static Opt_Param *thresh_param_p=NULL;
static Opt_Param *siqd_param_p=NULL;

/* For 2afc, we use the 2afc flag, and do the ogive fit by reflecting the points in
 * the origin - assuming that we have a linear x value scale with zero representing zero
 * signal.  But for 4afc, we have a chance rate of 0.25...  for now we just transform
 * the probabilities - but what is the correct thing to do?
 *
 * ALSO:  it is non-ideal to reflect the points in the origin, because for something
 * like contrast threshold, we expect the psychometric functions to have the same
 * shape on a log axis, in which case the xvals never get to 0...  We might like to fit
 * by assuming an ogive with one asymptote at 1 and the other at the chance rate...
 */

// regression parameters
#define SLOPE_NAME	"slope"
#define INTERCEPT_NAME	"intercept"

// ogive fit parameters
#define THRESHOLD_NAME	"threshold"
#define SIQD_NAME	"siqd"

//static Summary_Data_Tbl *the_dtbl=NULL;
static Fit_Data *the_fdp=NULL;

// local prototypes
//static float likelihood(SINGLE_QSP_ARG_DECL);

#define set_float_var(name, v) _set_float_var(QSP_ARG  name, v)

static void _set_float_var(QSP_ARG_DECL  const char *name, double v)
{
	char val_str[LLEN];
	sprintf(val_str,"%g",v);
	assign_reserved_var(name,val_str);
}

void set_fcflag(int flg)
{ fc_flag=flg; }

#define NO_GOOD		(-2.0)		// special flag value...

#define regr(fdp,first) _regr(QSP_ARG  fdp,first)

static double _regr(QSP_ARG_DECL  Fit_Data *fdp,int first)
// =1 if the first iteration
{
	int i;
	double sx,sy,sxx,syy,sxy;
	double xvar, yvar, xyvar;
	double r, nt;
	double x[MAX_X_VALUES], y[MAX_X_VALUES];
	double pc;
	double n[MAX_X_VALUES];
	double f1,f2,yt;
	short nsamps=0;
	int n_xvals;
	double chance_rate;
	Summary_Data_Tbl *dtp;

	dtp = CLASS_SUMM_DTBL(FIT_CLASS(fdp));
	assert(dtp!=NULL);
	assert(SUMM_DTBL_XVAL_OBJ(dtp)!=NULL);
	n_xvals = OBJ_COLS( SUMM_DTBL_XVAL_OBJ(dtp) );
	assert(n_xvals>1);
	for(i=0;i<n_xvals;i++) n[i]=0.0;
	chance_rate = FIT_CHANCE_RATE(fdp);
	for(i=0;i<n_xvals;i++){
		if( DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,i)) > 0 ){
			float *xv_p;

			pc= (double) DATUM_NCORR(SUMM_DTBL_ENTRY(dtp,i))
				/ (double) DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,i));

			/* BUG need to make sure that chance_rate is 0 if the 2afc
			 * flag is set - We need a better way to do this!
			 */
			if( chance_rate != 0.0 ){
				// transform the percent correct to the probability seen
				// p_correct = p_seen + p_guess * ( 1 - p_seen )
				//           = p_seen * ( 1 - p_guess ) + p_guess
				// p_seen = (p_correct-p_guess)/(1-p_guess)
				//
				// That is what we have here, but it seems incorrect
				// because the binomial variability is based on the p_correct,
				// not p_seen.  Therefore, the correct way to do this is to fit
				// a curve to the real data...
				//
				// However, this linear regression is just used for a first cut...
				pc -= chance_rate;
				pc *= 1/(1-chance_rate);
				if( pc < 0 ) pc = 0;
			}

			if( pc == 0.0 ) pc = .01;
			else if( pc == 1.0 ) pc = .99;
			y[nsamps]=ptoz(pc);
			xv_p = indexed_data(SUMM_DTBL_XVAL_OBJ(dtp),i);
			x[nsamps] = *xv_p;
			if( first ) yt=y[nsamps];
			else {
				yt= FIT_Y_INT(fdp) + FIT_SLOPE(fdp) * x[nsamps];
				pc = ztop( yt );
				if( pc == 1.0 ) pc=.99;
				else if( pc==0.0 ) pc=.01;
			}
			f1 = exp( - yt * yt );
			f2 = (pc*(1-pc));
			n[nsamps]= f1 * DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,i)) / f2;
			nsamps++;
		}
	}
	if( nsamps <= 1 ) {
		if( nsamps == 1 ) warn("sorry, can't fit a line to 1 point");
		else warn("sorry, can't file a line to 0 points");
		return(NO_GOOD);
	}
	nt=sx=sy=sxx=syy=sxy=0.0;
	for(i=0;i<nsamps;i++){
		sx += n[i] * x[i];
		sy += n[i] * y[i];
		sxx += n[i] * x[i] * x[i];
		syy += n[i] * y[i] *y[i] ;
		sxy += n[i] * x[i] * y[i] ;
		nt += n[i];
	}
	xvar = nt * sxx;
	yvar = nt * syy;
	xyvar = nt * sxy;

	// fc_flag=1 is for forced choice

	if( !fc_flag ){
		xvar -= sx * sx ;
		yvar -= sy * sy ;
		xyvar -= sx * sy ;
	}
	if( xvar == 0.0 ){
		warn("zero xvar");
		return(0.0);
	}
	SET_FIT_SLOPE(fdp, xyvar / xvar );

	if( fc_flag ) {
		SET_FIT_Y_INT(fdp,0.0);
	} else {
		SET_FIT_Y_INT(fdp,(sy-sx*FIT_SLOPE(fdp))/nt);
	}

	if( yvar==0.0 ){
		warn("zero yvar");
		return(0.0);
	}
	r=xyvar/sqrt(xvar*yvar);
	return(r);
}

/*
static float likelihood(SINGLE_QSP_ARG_DECL)	// called from optimization routine; return likelihood of guess
{
	float lh=0.0,lhinc;
	int i;
	int ntt,		// number of total trials
	    nc;			// number "correct"
	float pc,xv;
	float t_slope, t_int;	// trial slope and int
	int n_xvals;
	// Opt_Param *opp;

	// compute the likelihood for this guess
	assert( EXPT_XVAL_OBJ(&expt1) != NULL );
	n_xvals = OBJ_COLS(EXPT_XVAL_OBJ(&expt1));

	t_slope = get_opt_param_value(SLOPE_NAME);

	if( !fc_flag )
		t_int = get_opt_param_value(INTERCEPT_NAME);
	else
		t_int = 0.0;

	for(i=0;i<n_xvals;i++){
		float *xv_p;

		// calculate theoretical percent correct with this guess

		if( (ntt=DATUM_NTOTAL(SUMM_DTBL_ENTRY(the_dtbl,i))) <= 0 )
			continue;

		nc=DATUM_NCORR(SUMM_DTBL_ENTRY(the_dtbl,i));
		xv_p = indexed_data(EXPT_XVAL_OBJ(&expt1),i);
		xv = *xv_p;
		// This is the crux of the model - we ought to be able to put any other function here???
		pc = (float) ztop( t_int + t_slope * xv );
		// This seems like a hack...
		if( pc == 1.0 ) pc = (float) 0.99;
		else if( pc == 0.0 ) pc = (float) 0.01;

		// pc is the theoretical % correct at this xval

		lhinc = (float)( nc * log( pc ) + ( ntt - nc ) * log( 1 - pc ) );
		lh -= lhinc;
	}

	return(lh);
}
*/

static float _get_lin_ptoz(QSP_ARG_DECL  Fit_Data *fdp, float xv)
{
	float pc, t_slope, t_int;

	t_slope = get_opt_param_value(SLOPE_NAME);

	if( !fc_flag )
		t_int = get_opt_param_value(INTERCEPT_NAME);
	else
		t_int = 0.0;

	pc = (float) ztop( t_int + t_slope * xv );
	return pc;
}

// We want our function to go from p_guess to 1, with p_guess=0 being a normal case
// for a yes-no experiment.  In that case we would take p = (erf+1)/2
// with an original range of 2 and a transformed range of 1.
// When p_guess != 0, we want a transformed range of 1-p_guess,
// so the factor is (1-p_guess)/2.
// In general, p = (erf+1) * (1-p_guess)/2 + p_guess;

static float _ogive_prediction(QSP_ARG_DECL  Fit_Data *fdp, float xv)
{
	double thresh, siqd;
	float pc;

	thresh = get_opt_param_value(THRESHOLD_NAME);
	siqd = get_opt_param_value(SIQD_NAME);

	pc = erf( (xv - thresh)/siqd );

	pc = (pc+1)*(1-FIT_CHANCE_RATE(fdp))/2 + FIT_CHANCE_RATE(fdp);
	return pc;
}

static float (*prediction_func)(QSP_ARG_DECL  Fit_Data *fdp, float xv) = _get_lin_ptoz; 

static float generic_likelihood(SINGLE_QSP_ARG_DECL)	// return likelihood of trial params
{
	float lh=0.0,lhinc;
	int i;
	int ntt,		// number of total trials
	    nc;			// number "correct"
	float pc,xv;
//	float t_slope, t_int;	// trial slope and int
	int n_xvals;
	// Opt_Param *opp;
	Summary_Data_Tbl *the_dtbl;

	the_dtbl = CLASS_SUMM_DTBL( FIT_CLASS(the_fdp) );

	// compute the likelihood for this guess
	assert( EXPT_XVAL_OBJ(&expt1) != NULL );
	n_xvals = OBJ_COLS(EXPT_XVAL_OBJ(&expt1));

	for(i=0;i<n_xvals;i++){
		float *xv_p;

		// calculate theoretical percent correct with this guess

		if( (ntt=DATUM_NTOTAL(SUMM_DTBL_ENTRY(the_dtbl,i))) <= 0 )
			// no trials at this xval...
			continue;

		nc=DATUM_NCORR(SUMM_DTBL_ENTRY(the_dtbl,i));
		xv_p = indexed_data(EXPT_XVAL_OBJ(&expt1),i);
		xv = *xv_p;

		if( FIT_LOG_FLAG(the_fdp) ){
			xv = log10(xv);
		}

		pc = (*prediction_func)(QSP_ARG  the_fdp, xv);
//fprintf(stderr,"original xv = %g, xv = %g, predicted pc = %g\n",*xv_p,xv,pc);

		// This is the crux of the model - we ought to be able to put any other function here???
		// This seems like a hack...
		if( pc == 1.0 ) pc = (float) 0.99;
		else if( pc == 0.0 ) pc = (float) 0.01;

		// pc is the theoretical % correct at this xval

		lhinc = (float)( nc * log( pc ) + ( ntt - nc ) * log( 1 - pc ) );
		lh -= lhinc;
	}

	return(lh);
}

#define init_regression_opt_params(fdp) _init_regression_opt_params(QSP_ARG  fdp)

static void _init_regression_opt_params(QSP_ARG_DECL  Fit_Data *fdp)
{
	Opt_Param tmp_param;

	delete_opt_params(SINGLE_QSP_ARG);

	tmp_param.op_name=SLOPE_NAME;
	tmp_param.maxv=10000.0;
	tmp_param.minv=(-10000.0);
	tmp_param.ans = (float) FIT_SLOPE(fdp);
	if( FIT_SLOPE_CONSTRAINT(fdp) < 0 ){
		tmp_param.maxv = 0.0;
		if( FIT_SLOPE(fdp) > 0 ) tmp_param.ans = 0.0;
	} else if( FIT_SLOPE_CONSTRAINT(fdp) > 0 ){
		tmp_param.minv = 0.0;
		if( FIT_SLOPE(fdp) < 0 ) tmp_param.ans = 0.0;
	}
	tmp_param.delta = (float) fabs(FIT_SLOPE(fdp)/10.0);
	tmp_param.mindel = (float) 1.0e-30;

	assert(slope_param_p==NULL);
	slope_param_p = add_opt_param(&tmp_param);



	/* If we are fitting forced choice data with the normal ogive,
	 * we constrain the psychometric function to pass through
	 * the origin (on a z-score plot).  This means that
	 * the x value scale must have a meaningful zero, i.e.
	 * that xval=0 means no signal to detect.
	 *
	 * It seems likely that whether the x axis values are a linear
	 * representation of some physical quantity, or a (log) transform
	 * thereof, may make a difference on the result.  (note that the
	 * original zero is sent to minus infinity by the log transform,
	 * and that the new 0 is determined by a scale factor.
	 *
	 * So for things like contrast threshold, better NOT to set the fc flag,
	 * and set the chance rate instead, and regress against the log values...
	 */

	if( !fc_flag ){
		tmp_param.op_name=INTERCEPT_NAME;
		tmp_param.ans = (float) FIT_Y_INT(fdp);
		tmp_param.delta = (float) fabs(FIT_Y_INT(fdp)/10.0);
		tmp_param.mindel = (float) 1.0e-30;
		tmp_param.maxv = 10000.0;
		tmp_param.minv = -10000.0;

		assert(intercept_param_p == NULL);
		intercept_param_p = add_opt_param(&tmp_param);
	}
}

#define init_ogive_opt_params() _init_ogive_opt_params(SINGLE_QSP_ARG)

static void _init_ogive_opt_params(SINGLE_QSP_ARG_DECL)
{
	Opt_Param tmp_param;

	delete_opt_params(SINGLE_QSP_ARG);

	tmp_param.op_name=THRESHOLD_NAME;
	tmp_param.maxv=10000.0;
	tmp_param.minv=(-10000.0);
	tmp_param.ans = 0.0;		// user should have some input here!?
	tmp_param.delta = 0.1;
	tmp_param.mindel = (float) 1.0e-30;

	assert(thresh_param_p==NULL);
fprintf(stderr,"init_ogive_opt_params:  creating threshold param\n");
	thresh_param_p = add_opt_param(&tmp_param);

	tmp_param.op_name=SIQD_NAME;
	tmp_param.ans = (float) 1;
	tmp_param.delta = 0.1;
	tmp_param.mindel = (float) 1.0e-30;
	tmp_param.maxv = 10000.0;
	tmp_param.minv = 0.0;

	assert(siqd_param_p == NULL);
fprintf(stderr,"init_ogive_opt_params:  creating SIQD param\n");
	siqd_param_p = add_opt_param(&tmp_param);
}

#define finish_regression_optimization(fdp) _finish_regression_optimization(QSP_ARG  fdp)

static void _finish_regression_optimization(QSP_ARG_DECL  Fit_Data *fdp)
{
	SET_FIT_SLOPE(fdp, get_opt_param_value(SLOPE_NAME) );
	del_opt_param(slope_param_p);
	slope_param_p=NULL;

	if( !fc_flag ){
		SET_FIT_Y_INT(fdp, get_opt_param_value(INTERCEPT_NAME) );
		del_opt_param(intercept_param_p);
		intercept_param_p = NULL;
	} else {
		SET_FIT_Y_INT(fdp, 0.0);
	}
}

#define finish_ogive_optimization(fdp) _finish_ogive_optimization(QSP_ARG  fdp)

static void _finish_ogive_optimization(QSP_ARG_DECL  Fit_Data *fdp)
{
	SET_FIT_THRESH( fdp, get_opt_param_value(THRESHOLD_NAME) );
	del_opt_param(thresh_param_p);
	thresh_param_p=NULL;

	SET_FIT_SIQD(fdp, get_opt_param_value(SIQD_NAME) );
	del_opt_param(siqd_param_p);
	siqd_param_p = NULL;
}

#define ml_fit(fdp,ntrac) _ml_fit(QSP_ARG  fdp,ntrac)

static void _ml_fit(QSP_ARG_DECL  Fit_Data *fdp,int ntrac)		// maximum liklihood fit
{
	// initialize globals
	the_fdp = fdp;
	//the_dtbl = CLASS_SUMM_DTBL(FIT_CLASS(fdp));

	optimize(generic_likelihood);
}

void _old_ogive_fit( QSP_ARG_DECL  Fit_Data *fdp )		// do a regression on the ith table
{
	double _slope, _y_int;

	// ntrac = how_many("trace stepit output (-1,0,1)");

	retabulate_one_class(FIT_CLASS(fdp),NULL);

	SET_FIT_R_INITIAL(fdp, regr( fdp, 1 ) );
	if( FIT_R_INITIAL(fdp) == NO_GOOD){
                advise("\n");
                return;
        }

	prediction_func = _get_lin_ptoz; 
	init_regression_opt_params(fdp);
	ml_fit( fdp, /* ntrac */ -1 );
	finish_regression_optimization(fdp);

	// now we want to compute the correlation coefficient
	// for the final fit
	
	
	// need to remember M-L slope & int
	_slope = FIT_SLOPE(fdp);
	_y_int = FIT_Y_INT(fdp);

	SET_FIT_R(fdp, regr( fdp, 0 ) );

	// BUG?  it is not at all clear how the local variables _slope & _y_int are updated by regr???

	SET_FIT_SLOPE(fdp, _slope);
	SET_FIT_Y_INT(fdp, _y_int);

        if( FIT_SLOPE(fdp) == 0.0 ) warn("zero slope");
        else if(!fc_flag) {
                SET_FIT_THRESH(fdp, ( ptoz( .5 ) - FIT_Y_INT(fdp) )/ FIT_SLOPE(fdp) );
                SET_FIT_SIQD( fdp, ( ptoz(.25) - FIT_Y_INT(fdp) )/FIT_SLOPE(fdp) );
                if( FIT_SIQD(fdp) > FIT_THRESH(fdp) ){
			SET_FIT_SIQD( fdp, FIT_SIQD(fdp) - FIT_THRESH(fdp) );
		} else {
			SET_FIT_SIQD( fdp, FIT_THRESH(fdp) - FIT_SIQD(fdp) );
		}
		set_float_var("ogive_siqd", FIT_SIQD( fdp ) );
		set_float_var("ogive_threshold",FIT_THRESH(fdp));
        } else {
		SET_FIT_THRESH( fdp, ptoz(.75)/FIT_SLOPE(fdp) );
		set_float_var("ogive_threshold",FIT_THRESH(fdp));
	}
}

// The "new" ogive fit doesn't mess with the trick of transforming to z scores
// and fitting a line - rather, we just synthesize a function with 2 parameters:
// ogive_threshold, and ogive_siqd;
// A third parameter, p_guess (the chance rate) is not varied by the optimization.
//
// The math library erf() goes from -1 to 1, with inflection at 0 (and siqd=1 ???)
//
// We want our function to go from p_guess to 1, with p_guess=0 being a normal case
// for a yes-no experiment.  In that case we would take p = (erf+1)/2
// with an original range of 2 and a transformed range of 1.
// When p_guess != 0, we want a transformed range of 1-p_guess,
// so the factor is (1-p_guess)/2.
// In general, p = (erf+1) * (1-p_guess)/2 + p_guess;

void _new_ogive_fit( QSP_ARG_DECL  Fit_Data *fdp )		// do a regression on the ith table
{
	// ntrac = how_many("trace stepit output (-1,0,1)");

	retabulate_one_class(FIT_CLASS(fdp),NULL);

	prediction_func = _ogive_prediction; 
	init_ogive_opt_params();
	ml_fit( fdp, /* ntrac */ -1 );
	finish_ogive_optimization(fdp);

	set_float_var("ogive_siqd",FIT_SIQD( fdp ) );
	set_float_var("ogive_threshold",FIT_THRESH( fdp ) );
}

#ifdef QUIK

void pntquic(FILE *fp,Trial_Class * tcp,int in_db)
{
        int j;
	float v;
        Summary_Data_Tbl *dtp;

	dtp=(&dt[cl]);
	// first count the number of records
	j=0;
	while( DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,j)) && j<n_xvals )
		j++;
	fprintf(fp,"%d\n",j);
	j=0;
	for(j=0;j<n_xvals;j++)
		if( DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,j)) > 0 ){
			if( in_db )
			v= 20.0*log10( xval_array[ j ] );
			else v=xval_array[ j ];
			fprintf(fp,"%f\t%d\t%d\n", v,
				DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,j)),
				DATUM_NCORR(SUMM_DTBL_ENTRY(dtp,j)));
		}
	fflush(fp);
}
#endif // QUIK

// The split function was introduced to analyze the two limbs of a U-shaped function
// separately...
// We scan the summary table, looking at entries with data.  If we want the upper limb,
// then we zero all the entries before we see the first 0 correct, at which point we are done.
// If we want the lower limb, then we do nothing until we see the first zero

void _split(QSP_ARG_DECL  Trial_Class * tcp,int wantupper)
{
        int j;
        Summary_Data_Tbl *dtp;
	int havzero=0;
	int n_xvals;

	dtp=CLASS_SUMM_DTBL(tcp);
	assert(dtp!=NULL);

	assert(CLASS_XVAL_OBJ(tcp)!=NULL);
	n_xvals = OBJ_COLS( CLASS_XVAL_OBJ(tcp) );
	assert(n_xvals>1);

	for(j=0;j<n_xvals;j++){
		if( DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,j)) > 0 ){
			if( DATUM_NCORR(SUMM_DTBL_ENTRY(dtp,j)) == 0 ){
				if( wantupper ) return;
				havzero=1;
			} else {
				if( wantupper || havzero ){
					DATUM_NTOTAL(SUMM_DTBL_ENTRY(dtp,j)) = 0;
				}
			}
		}
	}
	if( !havzero ) warn("split:  no zero found!");
}

