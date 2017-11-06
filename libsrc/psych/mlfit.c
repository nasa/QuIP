#include "quip_config.h"

/* maximum liklihood fit suggested by DIAM */

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

int fc_flag=0;			/* forced choice */

static double slope, y_int;
static int slope_constraint=0;
static double _r_, _r_initial, _x_, siqd;
static double chance_rate=0.0;	/* for yes-no */

/* For 2afc, we use the 2afc flag, and do the ogive fit by reflecting the points in
 * the origin - assuming that we have a linear x value scale with zero representing zero
 * signal.  But for 4afc, we have a chance rate of 0.25...  for now we just transform
 * the probabilities - but what is the correct thing to do?
 */

#define SLOPE_NAME	"slope"
#define INTERCEPT_NAME	"intercept"

Data_Tbl *the_dtbl;

/* local prototypes */
static float likelihood(SINGLE_QSP_ARG_DECL);

void set_chance_rate( double r )
{
	chance_rate = r;
}

void set_fcflag(int flg)
{ fc_flag=flg; }

double regr(Data_Tbl *dtp,int first)
/* =1 if the first iteration */
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

	for(i=0;i<_nvals;i++) n[i]=0.0;
	for(i=0;i<_nvals;i++){
		if( DATUM_NTOTAL(DTBL_ENTRY(dtp,i)) > 0 ){
			pc= (double) DATUM_NCORR(DTBL_ENTRY(dtp,i))
				/ (double) DATUM_NTOTAL(DTBL_ENTRY(dtp,i));

			/* BUG need to make sure that chance_rate is 0 if the 2afc
			 * flag is set - We need a better way to do this!
			 */
			if( chance_rate != 0.0 ){
				pc -= chance_rate;
				pc *= 1/(1-chance_rate);
				if( pc < 0 ) pc = 0;
			}

			if( pc == 0.0 ) pc = .01;
			else if( pc == 1.0 ) pc = .99;
			y[nsamps]=ptoz(pc);
			x[nsamps]=xval_array[i];
			if( first ) yt=y[nsamps];
			else {
				yt= y_int + slope * x[nsamps];
				pc = ztop( yt );
				if( pc == 1.0 ) pc=.99;
				else if( pc==0.0 ) pc=.01;
			}
			f1 = exp( - yt * yt );
			f2 = (pc*(1-pc));
			n[nsamps]= f1 * DATUM_NTOTAL(DTBL_ENTRY(dtp,i)) / f2;
			nsamps++;
		}
	}
	if( nsamps <= 1 ) {
		if( nsamps == 1 ) NWARN("sorry, can't fit a line to 1 point");
		else NWARN("sorry, can't file a line to 0 points");
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

	/* fc_flag=1 is for forced choice */

	if( !fc_flag ){
		xvar -= sx * sx ;
		yvar -= sy * sy ;
		xyvar -= sx * sy ;
	}
	if( xvar == 0.0 ){
		NWARN("zero xvar");
		return(0.0);
	}
	slope= xyvar/ xvar;

	if( fc_flag ) y_int=0.0;
	else y_int= (sy-sx*slope)/nt;

	if( yvar==0.0 ){
		NWARN("zero yvar");
		return(0.0);
	}
	r=xyvar/sqrt(xvar*yvar);
	return(r);
}

static float likelihood(SINGLE_QSP_ARG_DECL)	/* called from optimization routine; return likelihood of guess */
{

	float lh=0.0,lhinc;
	int i;
	int ntt,		/* number of total trials */
	    nc;			/* number "correct" */
	float pc,xv;

	float t_slope, t_int;	/* trial slope and int */
	/* Opt_Param *opp; */

	/* compute the likelihood for this guess */

	t_slope = get_opt_param_value(QSP_ARG  SLOPE_NAME);

	if( !fc_flag )
		t_int = get_opt_param_value(QSP_ARG  INTERCEPT_NAME);
	else
		t_int = 0.0;

	for(i=0;i<_nvals;i++){

		/* calculate theoretical percent correct with this guess */

		if( (ntt=DATUM_NTOTAL(DTBL_ENTRY(the_dtbl,i))) <= 0 )
			continue;

		nc=DATUM_NCORR(DTBL_ENTRY(the_dtbl,i));
		xv = xval_array[ i ];
		pc = (float) ztop( t_int + t_slope * xv );
		if( pc == 1.0 ) pc = (float) 0.99;
		else if( pc == 0.0 ) pc = (float) 0.01;

		/* pc is the theoretical % correct at this xval */

		lhinc = (float)( nc * log( pc ) + ( ntt - nc ) * log( 1 - pc ) );
		lh -= lhinc;
	}

	return(lh);
}

void ml_fit(QSP_ARG_DECL  Data_Tbl *dtp,int ntrac)		/** maximum liklihood fit */
{
	Opt_Param op1, *slope_opp=(&op1);
	Opt_Param op2, *int_opp=(&op2);

	/* initialize global */
	the_dtbl = dtp;

	delete_opt_params(SINGLE_QSP_ARG);

	slope_opp->op_name=SLOPE_NAME;
	slope_opp->maxv=10000.0;
	slope_opp->minv=(-10000.0);
	slope_opp->ans = (float) slope;
	if( slope_constraint < 0 ){
		slope_opp->maxv = 0.0;
		if( slope > 0 ) slope_opp->ans = 0.0;
	} else if( slope_constraint > 0 ){
		slope_opp->minv = 0.0;
		if( slope < 0 ) slope_opp->ans = 0.0;
	}
	slope_opp->delta = (float) fabs(slope/10.0);
	slope_opp->mindel = (float) 1.0e-30;

	add_opt_param(QSP_ARG  slope_opp);



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
	 */

	if( !fc_flag ){
		int_opp->op_name=INTERCEPT_NAME;
		int_opp->ans = (float) y_int;
		int_opp->delta = (float) fabs(y_int/10.0);
		int_opp->mindel = (float) 1.0e-30;
		int_opp->maxv = 10000.0;
		int_opp->minv = -10000.0;
		add_opt_param(QSP_ARG  int_opp);
	}


	optimize(QSP_ARG  likelihood);

	slope = get_opt_param_value(QSP_ARG  SLOPE_NAME);
	del_opt_param(slope_opp);

	if( !fc_flag ){
		y_int = get_opt_param_value(QSP_ARG  INTERCEPT_NAME);
		del_opt_param(int_opp);
	} else
		y_int = 0.0;
}

void analyse( QSP_ARG_DECL  Trial_Class *tcp )		/** do a regression on the ith table */
{
	double _slope, _y_int;

	/* ntrac = how_many("trace stepit output (-1,0,1)"); */

	_r_initial = regr( CLASS_DATA_TBL(tcp), 1 );
	if(_r_initial == NO_GOOD){
                advise("\n");
                return;
        }

	ml_fit( QSP_ARG  CLASS_DATA_TBL(tcp), /* ntrac */ -1 );

	/* now we want to compute the correlation coefficient
	 * for the final fit
	 */
	
	/* need to remember M-L slope & int */
	_slope = slope;
	_y_int = y_int;

	_r_ = regr( CLASS_DATA_TBL(tcp), 0 );

	slope = _slope;
	y_int = _y_int;

        if( slope == 0.0 ) NWARN("zero slope");
        else if(!fc_flag) {
                _x_ = ( ptoz( .5 ) - y_int )/ slope ;
                siqd = ( ptoz(.25) - y_int )/slope;
                if( siqd > _x_ ) siqd-=_x_;
                else siqd = _x_-siqd;
        }
	else {
		_x_=ptoz(.75)/slope;
	}
}

void longout(QSP_ARG_DECL  Trial_Class *tcp)	/** verbose analysis report */
{
        sprintf(msg_str,"\nTrial_Class %s\n",CLASS_NAME(tcp));
	prt_msg(msg_str);
        sprintf(msg_str,"initial correlation:\t\t\t%f", _r_initial );
	prt_msg(msg_str);
        sprintf(msg_str,"final correlation:\t\t\t%f", _r_ );
	prt_msg(msg_str);

        if(!fc_flag) {
                sprintf(msg_str,"x value at inflection pt:\t\t%f",_x_);
		prt_msg(msg_str);
                sprintf(msg_str,"semi-interquartile difference:\t\t%f",siqd);
		prt_msg(msg_str);
        } else {
		sprintf(msg_str,"x value for 75%%:\t%f",_x_);
		prt_msg(msg_str);
	}
}

void tersout(QSP_ARG_DECL  Trial_Class *tcp)
{
	if( !fc_flag ) 
		sprintf(msg_str,"%d\t%f\t%f\t%f", CLASS_INDEX(tcp),_r_, _x_,siqd);
	else
		sprintf(msg_str,"%d\t%f\t%f", CLASS_INDEX(tcp),_r_, _x_);
	prt_msg(msg_str);
}

#ifdef QUIK

void pntquic(FILE *fp,Trial_Class * tcp,int in_db)
{
        int j;
	float v;
        Data_Tbl *dtp;

	dtp=(&dt[cl]);
	/* first count the number of records */
	j=0;
	while( DATUM_NTOTAL(DTBL_ENTRY(dtp,j)) && j<_nvals )
		j++;
	fprintf(fp,"%d\n",j);
	j=0;
	for(j=0;j<_nvals;j++)
		if( DATUM_NTOTAL(DTBL_ENTRY(dtp,j)) > 0 ){
			if( in_db )
			v= 20.0*log10( xval_array[ j ] );
			else v=xval_array[ j ];
			fprintf(fp,"%f\t%d\t%d\n", v,
				DATUM_NTOTAL(DTBL_ENTRY(dtp,j)),
				DATUM_NCORR(DTBL_ENTRY(dtp,j)));
		}
	fflush(fp);
}
#endif /* QUIK */

void print_raw_data(QSP_ARG_DECL  Trial_Class * tcp)
{
        int j;
        Data_Tbl *dtp;

	assert( tcp != NULL );

	dtp=CLASS_DATA_TBL(tcp);

	assert( dtp != NULL );

	if( verbose ){
		sprintf(msg_str,"class = %s, %d points",
			CLASS_NAME(tcp),DTBL_N(CLASS_DATA_TBL(tcp)));
		prt_msg(msg_str);
		sprintf(msg_str,"val\txval\t\tntot\tncorr\t%% corr\n");
		prt_msg(msg_str);
	}
	//j=0;
	for(j=0;j<_nvals;j++){
		if( DATUM_NTOTAL(DTBL_ENTRY(dtp,j)) > 0 ){
			sprintf(msg_str,"%d\t%f\t%d\t%d\t%f",
				j,
				xval_array[ j ],
				DATUM_NTOTAL(DTBL_ENTRY(dtp,j)),
				DATUM_NCORR(DTBL_ENTRY(dtp,j)),
				(double) DATUM_NCORR(DTBL_ENTRY(dtp,j)) /
					(double) DATUM_NTOTAL(DTBL_ENTRY(dtp,j)) );
			prt_msg(msg_str);
		}
	}
	prt_msg("\n");
}

void split(QSP_ARG_DECL  Trial_Class * tcp,int wantupper)
{
        int j;
        Data_Tbl *dtp;
	int havzero=0;

	//tcp=index_class(QSP_ARG  cl);
	dtp=CLASS_DATA_TBL(tcp);
	//j=0;
	for(j=0;j<_nvals;j++){
		if( DATUM_NTOTAL(DTBL_ENTRY(dtp,j)) > 0 ){
			if( DATUM_NCORR(DTBL_ENTRY(dtp,j)) == 0 ){
				if( wantupper ){
					return;
				}
				havzero=1;
			} else {
				if( wantupper ){
					DATUM_NTOTAL(DTBL_ENTRY(dtp,j)) = 0;
				} else if( havzero ){
					DATUM_NTOTAL(DTBL_ENTRY(dtp,j)) = 0;
				}
			}
		}
	}
	if( !havzero ) NWARN("split:  no zero found!");
}

static const char *clist[]={"negative","unconstrained","positive"};

COMMAND_FUNC( constrain_slope )
{
	int ctype;

	ctype = which_one("constraint for slope",3,clist);
	if( ctype < 0 ) return;

	slope_constraint = ctype -1;
}


