/* char SccsId_stepit_am_supp[] = "@(#) am_supp.c ver: 1.3 4/29/98"; */

#include "quip_config.h"

#ifdef HAVE_NUMREC
#ifdef USE_NUMREC

/*
 * linkage to numerical recipes Polak Ribiere routine
 */

#include <math.h>

#include "quip_prot.h"
//#include "fitsine.h"
#include "optimize.h"
#include "list.h"
#include "variable.h"

static Query_Stack *pr_qsp=NULL;

/* local prototypes */

static float frprmn_scr_funk(float *p);
static void dfunc(float *, float *);

static int n_prms;


static float (*user_c_func)(SINGLE_QSP_ARG_DECL);

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

#define DELTA 0.001f

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
	Variable *vp;
	int i;
	List *lp;
	Node *np;
#ifdef THREAD_SAFE_QUERY
	Query_Stack *qsp;

	qsp = pr_qsp;
#endif // THREAD_SAFE_QUERY
	lp=opt_param_list();
	if( lp==NULL ) {
		NWARN("no parameters!?");
		return(0.0);
	}

	np=QLIST_HEAD(lp);

	i=0;
	while(np!=NULL){
		Opt_Param *opp;

		opp=(Opt_Param *)np->n_data;
		sprintf(str,"%g",p[i+1]);
		assign_var(opp->op_name,str);
		i++;
		np=np->n_next;
	}

	digest(opt_func_string, OPTIMIZER_FILENAME);
	
	vp=var__of("error");
	if( vp == NULL ) {
		WARN("variable \"error\" not set!!!");
		err=0.0;
	} else sscanf(VAR_VALUE(vp),"%g",&err);

	return(err);
}

static float frprmn_c_funk(float *p)
{
	float	err;
	int i;
	List *lp;
	Opt_Param *opp;
	Node *np;

	lp=_opt_param_list(SGL_DEFAULT_QSP_ARG);
	np=QLIST_HEAD(lp);

	i=0;
	while(np!=NULL){
		opp=(Opt_Param *)np->n_data;
		opp->ans=p[i+1];
		np=np->n_next;
		i++;
	}

	err = (*user_c_func)(SGL_DEFAULT_QSP_ARG);

	return(err);
}

static float p[MAX_OPT_PARAMS];

static void run_frprmn( QSP_ARG_DECL  float (*func)(float *) )
{
	float ftol;
	int iter;
	float fret;
	List *lp;
	Node *np;
	Opt_Param *opp;
	int i;

	/* initialize the number of parameters */

	n_prms = eltcount( lp=opt_param_list() );

	/* init the values */
	np=QLIST_HEAD(lp);
	i=0;
	while(np!=NULL){
		opp = (Opt_Param *)np->n_data;
		p[i] = opp->ans;
		np=np->n_next;
		i++;
	}

	/* call frprmn */

	/* not sure how to properly set this!? */
	ftol=0.001f;

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
		sprintf(DEFAULT_MSG_STR,"run_frprmn:  %d iterations, final value %g",
			iter,fret);
		prt_msg(DEFAULT_MSG_STR);
	}
}

void run_frprmn_scr(SINGLE_QSP_ARG_DECL)
{
	pr_qsp = THIS_QSP;
	run_frprmn( QSP_ARG  frprmn_scr_funk );
}

void run_frprmn_c( QSP_ARG_DECL  float (*func)(SINGLE_QSP_ARG_DECL) )
{
	user_c_func = func;
	run_frprmn( QSP_ARG  frprmn_c_funk );
}


#endif /* USE_NUMREC */
#endif /* HAVE_NUMREC */

