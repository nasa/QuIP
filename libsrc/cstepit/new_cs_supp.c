#include "quip_config.h"

/*
 * linkage to C stepit
 */

#include "version.h"
	/* for some reason, the compiler chokes if this is after math.h!? */
	/* but which compiler??? */

#include <math.h>

#include "savestr.h"
#include "fitsine.h"
#include "debug.h"

#include "new_cstepit.h"
#include "optimize.h"

static int n_prms;
static Query_Stream *cs_qsp=NULL;

/* local variables */
static float (*new_stept_user_func)(void);



/* local prototypes */

static void new_cstepit_scr_funk(void);
static void new_init_cstepit_params(void);

static void new_cstepit_scr_funk()
{
	char str[128];
	float	err;
	Var *vp;
	int i;
	List *lp;
	Node *np;
	double ans[MAX_OPT_PARAMS];
	Query_Stream *qsp;

	qsp = cs_qsp;

	new_getvals(ans,n_prms);

	if( opt_func_string==NO_STR ){
		NWARN("No optimization string defined");
		return;
	}

	lp=opt_param_list();
	if( lp == NO_LIST ){
		NWARN("No optimization parameters to vary!?");
		err=0.0;
		new_setfobj((double)err);
		return;
	}
	np=lp->l_head;

	i=0;
	while(np!=NO_NODE && i < n_prms ){
		Opt_Param *opp;

		opp = (Opt_Param *)( np->n_data);
		sprintf(str,"%g",ans[i]);	/* why add 1?  fortan? */
		ASSIGN_VAR(opp->op_name,str);
		i++;
		np=np->n_next;
	}

	digest(DEFAULT_QSP_ARG  opt_func_string, OPTIMIZER_FILENAME);

	vp=var__of("error");
	if( vp == NO_VAR ) {
		NWARN(error_string);
		sprintf(error_string,
	"variable \"error\" not set by script fragment \"%s\"!?",
			opt_func_string);
		err=0.0;
	} else sscanf(vp->v_value,"%g",&err);

	new_setfobj((double)err);
}

COMMAND_FUNC( new_run_cstepit_scr )
{
	new_init_cstepit_params();

	cs_qsp = THIS_QSP;
	new_stepit(new_cstepit_scr_funk);
}

void new_evaluate_error_c()
{
	double	err;
	double	x[MAX_OPT_PARAMS];
	int i;
	List *lp;
	Node *np;

	new_getvals(x,n_prms);		/* get the parameter estimates */

	lp=opt_param_list();
	np=lp->l_head;
	i=0;
	while(np!=NO_NODE && i < n_prms ){
		Opt_Param *opp;

		opp = (Opt_Param *)(np->n_data);
		opp->ans = x[i];
		i++;
		np=np->n_next;
	}

	err=(*new_stept_user_func)();

	new_setfobj(err);
}

void new_run_cstepit_c(float (*func)())
{
	new_init_cstepit_params();

	new_stept_user_func = func;

	new_stepit(new_evaluate_error_c);
}

static void new_init_cstepit_params()
{
	double xmin[MAX_OPT_PARAMS];
	double xmax[MAX_OPT_PARAMS];
	double deltx[MAX_OPT_PARAMS];
	double delmn[MAX_OPT_PARAMS];
	double ans[MAX_OPT_PARAMS];
	List *lp;
	Node *np;
	Opt_Param *opp;
	int i,n;
	int nfmax;		/* max. # function calls */

	lp = opt_param_list();
	if( lp == NO_LIST ) return;

	n_prms=eltcount(lp);
	n=new_reset_n_params(n_prms);
	if( n != n_prms ) n_prms = n;

	np=lp->l_head;
	i=0;
	while( np!= NO_NODE && i < n_prms ){
		opp = (Opt_Param *)(np->n_data);

		xmin[i]=opp->minv;
		xmax[i]=opp->maxv;
		deltx[i]=opp->delta;
		delmn[i]=opp->mindel;
		ans[i]=opp->ans;

		i++;
		np=np->n_next;
	}
	nfmax=100000;

	/* copy to fortran */

	new_setvals(ans,n_prms);
	new_setminmax(xmin,xmax,n_prms);
	new_setdelta(deltx,delmn,n_prms);
	/*
	new_settrace(ntrac);
	*/
	new_setmaxcalls(nfmax);
}
