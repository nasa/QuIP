#include "quip_config.h"

/*
 * Interpreter interface to stepit
 *
 *  Script computes error value and places in variable "error".
 *  Stepit places parameter values in script variables.
 */

#include <math.h>

#include "quip_prot.h"
#include "data_obj.h"	/* unlock_all_tmp_objs */
//#include "fitsine.h"
#include "optimize.h"
#include "sparse.h"	// do_sparse_menu

static COMMAND_FUNC( set_params )
{
	int i;
	int n;

	n=(int)how_many("number of paramters");
	if( n <= 0 ){
		warn("ridiculous number of paramters");
		return;
	}

	/* first, delete old parameters */
	delete_opt_params(SINGLE_QSP_ARG);

	for(i=0;i<n;i++){
		Opt_Param *opp;
		const char *s;
		float a,mnv,mxv,del,mndl;

		s=NAMEOF("parameter name");

		a=(float)how_much("starting value");
		mnv=(float)how_much("minimum value");
		mxv=(float)how_much("maximum value");
		del=(float)how_much("starting increment");
		mndl=(float)how_much("minimum increment");

		opp = new_opt_param(s);
		if( opp != NULL ){
			opp->ans = a;
			opp->minv = mnv;
			opp->maxv = mxv;
			opp->delta = del;
			opp->mindel = mndl;
		}

		/* in case we have a lot of params, and the expressions
		 * involve subsripted objects...
		 */
		unlock_all_tmp_objs();
	}
/*
sprintf(error_string,"%d parameters read",n);
advise(error_string);
*/
}

static COMMAND_FUNC( set_function )
{
	const char *s;

	if( opt_func_string != NULL ){
		rls_str(opt_func_string);
		opt_func_string=NULL;
	}
	s=NAMEOF("string to interpret to compute error");
	if( s != NULL )
		opt_func_string = savestr(s);
}

static COMMAND_FUNC( select_package )
{
	Opt_Pkg *pkp;

	insure_opt_pkg(SINGLE_QSP_ARG);

	pkp=pick_opt_pkg("");
	if( pkp!=NULL )

	curr_opt_pkg = pkp;
}

static COMMAND_FUNC( run_opt )
{
	insure_opt_pkg(SINGLE_QSP_ARG);

	/*lookahead(); */
	(*curr_opt_pkg->pkg_scr_func)(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( halt_opt )
{
	insure_opt_pkg(SINGLE_QSP_ARG);

	(*curr_opt_pkg->pkg_halt_func)();
}

static COMMAND_FUNC( do_opt_param_info )
{
	Opt_Param *opp;

	opp = pick_opt_param("");
	if( opp == NULL ) return;
	opt_param_info(opp);
}

static COMMAND_FUNC(do_list_opt_params){list_opt_params(tell_msgfile());}

#define ADD_CMD(s,f,h)	ADD_COMMAND(stepit_menu,s,f,h)

MENU_BEGIN(stepit)
ADD_CMD( package,	select_package,	select optimization package )
ADD_CMD( function,	set_function,	specify command which computes error )
ADD_CMD( parameters,	set_params,	specify parameters to be varied )
ADD_CMD( optimize,	run_opt,	search for minimum error )
ADD_CMD( halt,		halt_opt,	halt optimization )
ADD_CMD( list,		do_list_opt_params,	list current parameters )
// BUG?  shouldn't we configure for presence of this library???
// and parse the commands even if the lib isn't present?
#ifndef BUILD_FOR_OBJC
ADD_CMD( sparse,	do_sparse_menu,	sparse L-M sover )
#endif // ! BUILD_FOR_OBJC
ADD_CMD( info,		do_opt_param_info,	report information about a parameter )

MENU_END(stepit)

COMMAND_FUNC( do_step_menu )
{
	CHECK_AND_PUSH_MENU(stepit);
}


