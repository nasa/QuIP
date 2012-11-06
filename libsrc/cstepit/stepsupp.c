#include "quip_config.h"

char VersionId_cstepit_stepsupp[] = QUIP_VERSION_STRING;

#include <math.h>

#include "savestr.h"
#include "query.h"
#include "fitsine.h"
#include "items.h"
#include "optimize.h"

ITEM_INTERFACE_DECLARATIONS(Opt_Param,opt_param)

const char *opt_func_string=NULL;

Opt_Pkg *curr_opt_pkg=NO_OPT_PKG;


List *opt_param_list(SINGLE_QSP_ARG_DECL)
{
	if( opt_param_itp==NO_ITEM_TYPE ){
		/* NWARN("opt_param_list:  no parameters defined"); */
		return(NO_LIST);
	}

	return( item_list(QSP_ARG  opt_param_itp) );
}

void opt_param_info(Opt_Param *opp)
{
	sprintf(msg_str,"Parameter %s:",opp->op_name);
	prt_msg(msg_str);
	sprintf(msg_str,"\tvalue:\t%f",opp->ans);
	prt_msg(msg_str);
	sprintf(msg_str,"\tmin:\t%f",opp->minv);
	prt_msg(msg_str);
	sprintf(msg_str,"\tmax:\t%f",opp->maxv);
	prt_msg(msg_str);
	sprintf(msg_str,"\tdelta:\t%f",opp->delta);
	prt_msg(msg_str);
	sprintf(msg_str,"\tmindel:\t%f",opp->mindel);
	prt_msg(msg_str);
}

/* C language interface */

void delete_opt_params(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;

	lp = opt_param_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ) return;
	np=lp->l_head;
	while( np!= NO_NODE ){
		Opt_Param *opp;
		Node *next;

		opp = (Opt_Param *)(np->n_data);
		next=np->n_next;
		del_opt_param(QSP_ARG  opp->op_name);
		rls_str(opp->op_name);
		np=next;
	}
}

Opt_Param * add_opt_param(QSP_ARG_DECL  Opt_Param *opp)
{
	Opt_Param *new_opp;

	new_opp = new_opt_param(QSP_ARG  opp->op_name);
	if( new_opp != NO_OPT_PARAM ){
		new_opp->ans = opp->ans;
		new_opp->maxv = opp->maxv;
		new_opp->minv = opp->minv;
		new_opp->delta = opp->delta;
		new_opp->mindel = opp->mindel;
	}
	return(new_opp);
}

void optimize(QSP_ARG_DECL  float (*opt_func)())
{
	insure_opt_pkg(SINGLE_QSP_ARG);
	(*curr_opt_pkg->pkg_c_func)(opt_func);
}

float get_opt_param_value(QSP_ARG_DECL  const char *name)
{
	Opt_Param *opp;

	opp=get_opt_param(QSP_ARG  name);
	if( opp==NO_OPT_PARAM ){
		sprintf(error_string,"No optimization parameter \"%s\"",name);
		NWARN(error_string);
		return(-1.0);
	}
	return(opp->ans);
}

