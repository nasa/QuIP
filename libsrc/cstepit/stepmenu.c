#include "quip_config.h"

char VersionId_cstepit_stepmenu[] = QUIP_VERSION_STRING;

/*
 * Interpreter interface to stepit
 *
 *  Script computes error value and places in variable "error".
 *  Stepit places parameter values in script variables.
 */

#include "version.h"	/* for some reason, the compiler chokes if this is after math.h!? */

#include <math.h>

#include "data_obj.h"	/* unlock_all_tmp_objs */
#include "savestr.h"
#include "query.h"
#include "fitsine.h"
#include "items.h"

#include "optimize.h"
#include "submenus.h"

static COMMAND_FUNC( set_params )
{
	int i;
	int n;

	n=HOW_MANY("number of paramters");
	if( n <= 0 ){
		NWARN("ridiculous number of paramters");
		return;
	}

	/* first, delete old parameters */
	delete_opt_params(SINGLE_QSP_ARG);

	for(i=0;i<n;i++){
		Opt_Param *opp;
		const char *s;
		float a,mnv,mxv,del,mndl;

		s=NAMEOF("parameter name");

		a=HOW_MUCH("starting value");
		mnv=HOW_MUCH("minimum value");
		mxv=HOW_MUCH("maximum value");
		del=HOW_MUCH("starting increment");
		mndl=HOW_MUCH("minimum increment");

		opp = new_opt_param(QSP_ARG  s);
		if( opp != NO_OPT_PARAM ){
			opp->ans = a;
			opp->minv = mnv;
			opp->maxv = mxv;
			opp->delta = del;
			opp->mindel = mndl;
		}

		unlock_all_tmp_objs();	/* in case we have a lot of params, and the expressions
					 * involve subsripted objects... */
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

	pkp=PICK_OPT_PKG("");
	if( pkp!=NO_OPT_PKG )

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

	opp = PICK_OPT_PARAM("");
	if( opp == NO_OPT_PARAM ) return;
	opt_param_info(opp);
}

static COMMAND_FUNC(do_list_opt_params){list_opt_params(SINGLE_QSP_ARG);}

Command step_ctbl[]={
{ "package",	select_package,	"select optimization package"		},
{ "function",	set_function,	"specify command which computes error"	},
{ "parameters",	set_params,	"specify parameters to be varied"	},
{ "optimize",	run_opt,	"search for minimum error"		},
{ "halt",	halt_opt,	"halt optimization"			},

{ "list",	do_list_opt_params,	"list current parameters"	},
{ "info",	do_opt_param_info,	"report information about a parameter"	},

{ "quit",	popcmd,		"exit"					},
{ NULL_COMMAND								}
};



COMMAND_FUNC( stepmenu )
{
	static int inited=0;

	if( !inited ){
		auto_version(QSP_ARG  "CSTEPIT","VersionId_cstepit");
		inited=1;
	}
	PUSHCMD(step_ctbl,"stepit");
}


