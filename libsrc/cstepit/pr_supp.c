/* char SccsId_stepit_am_supp[] = "@(#) am_supp.c ver: 1.3 4/29/98"; */

#include "quip_config.h"

char VersionId_cstepit_pr_supp[] = QUIP_VERSION_STRING;

#ifdef HAVE_NUMREC

/*
 * linkage to numerical recipes Polak Ribiere routine
 */

#include "version.h"	/* for some reason, the compiler chokes if this is after math.h!? */

#include <math.h>

#include "savestr.h"
#include "query.h"
#include "fitsine.h"
#include "debug.h"

#include "optimize.h"
#include "chewtext.h"		/* digest */

static Query_Stream *pr_qsp=NULL;

/* local prototypes */

static float frprmn_scr_funk(float *p);
static void run_frprmn(float (*func)(float *));
static void dfunc(float *, float *);

static int n_prms;


static float (*user_c_func)(void);

static float (*pr_error_func)(float *);

static void dfunc(float *p,float *df)
{
	int i;
	float result;
	float save_p;

	result = pr_error_func(p);
	/* We compute the gradient by taking a tiny step in each direction */
	/* We ought to use min_inc for each parameter as the step size? */

	/* use fortran style index */
	for(i=1;i<=n_prms;i++){
		save_p = p[i];

#define DELTA 0.001

		p[i] += /* min_inc[i] */ DELTA ;
		df[i] = ( pr_error_func(p) - result ) / DELTA ;
		p[i] = save_p;
	}
}

void halt_frprmn(void)
{
	NWARN("Sorry, don't know how to halt frprmn");
}

static float frprmn_scr_funk(float *p)
{
	char str[128];
	float	err;
	Var *vp;
	int i;
	List *lp;
	Node *np;
	Query_Stream *qsp;

	qsp = pr_qsp;

	lp=opt_param_list(SGL_DEFAULT_QSP_ARG);
	if( lp==NO_LIST ) {
		NWARN("no parameters!?");
		return(0.0);
	}

	np=lp->l_head;

	i=0;
	while(np!=NO_NODE){
		Opt_Param *opp;

		opp=(Opt_Param *)np->n_data;
		sprintf(str,"%g",p[i+1]);
		ASSIGN_VAR(opp->op_name,str);
		i++;
		np=np->n_next;
	}

	DIGEST(opt_func_string);	/* used to call pushtext here */

	vp=var__of(QSP_ARG  "error");
	if( vp == NO_VAR ) {
		NWARN("variable \"error\" not set!!!");
		err=0.0;
	} else sscanf(vp->v_value,"%g",&err);

	return(err);
}

static float frprmn_c_funk(float *p)
{
	float	err;
	int i;
	List *lp;
	Opt_Param *opp;
	Node *np;

	lp=opt_param_list(SGL_DEFAULT_QSP_ARG);
	np=lp->l_head;

	i=0;
	while(np!=NO_NODE){
		opp=(Opt_Param *)np->n_data;
		opp->ans=p[i+1];
		np=np->n_next;
		i++;
	}

	err = (*user_c_func)();

	return(err);
}

static float p[MAX_OPT_PARAMS];

void run_frprmn_scr(SINGLE_QSP_ARG_DECL)
{
	pr_qsp = THIS_QSP;
	run_frprmn( frprmn_scr_funk );
}

void run_frprmn_c( float (*func)(void) )
{
	user_c_func = func;
	run_frprmn( frprmn_c_funk );
}


static void run_frprmn( float (*func)(float *) )
{
	float ftol;
	int iter;
	float fret;
	List *lp;
	Node *np;
	Opt_Param *opp;
	int i;

	/* initialize the number of parameters */

	n_prms = eltcount( lp=opt_param_list(SGL_DEFAULT_QSP_ARG) );

	/* init the values */
	np=lp->l_head;
	i=0;
	while(np!=NO_NODE){
		opp = (Opt_Param *)np->n_data;
		p[i] = opp->ans;
		np=np->n_next;
		i++;
	}

	/* call frprmn */

	/* not sure how to properly set this!? */
	ftol=0.001;

/*
if( verbose ){
printf("calling frprmn:\n");
printf("ndim = %d\n",n);
printf("ftol = %f\n",ftol);
printf("funk = 0x%x\n",&func);
printf("&iter = 0x%x\n",&iter);
}
*/

	pr_error_func = func;

	frprmn(p-1,n_prms,ftol,&iter,&fret,func,dfunc);

	if( verbose ){
		sprintf(msg_str,"run_frprmn:  %d iterations, final value %g",
			iter,fret);
		prt_msg(msg_str);
	}
}

#endif /* HAVE_NUMREC */

