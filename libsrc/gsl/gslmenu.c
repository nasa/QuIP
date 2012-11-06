
#include "quip_config.h"

char VersionId_gslmenu_gslmenu[] = QUIP_VERSION_STRING;

#ifdef HAVE_GSL

#include <stdio.h>
#include "data_obj.h"
#include "debug.h"
#include "query.h"
#include "menuname.h"
#include "version.h"
#include "submenus.h"

#include "gslprot.h"

static COMMAND_FUNC( do_gsl_svd );

#ifdef DEBUG
static void gsl_debug_init(SINGLE_QSP_ARG_DECL);
#endif /* DEBUG */


static COMMAND_FUNC( do_gsl_svd )
{
	Data_Obj *a_dp, *w_dp, *v_dp;

	a_dp=PICK_OBJ( "input matrix" );
	w_dp=PICK_OBJ( "vector for singular values" );
	v_dp=PICK_OBJ( "output v matrix" );

	if( a_dp == NO_OBJ || w_dp == NO_OBJ || v_dp == NO_OBJ )
		return;

	gsl_svd(a_dp,w_dp,v_dp);
}


Command gsl_ctbl[]={
{ "gslsvd",	do_gsl_svd,		"singular value decomposition"			},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"					},
#endif
{ NULL_COMMAND									}
};

#ifdef DEBUG
int gsl_debug=0;

static void gsl_debug_init(SINGLE_QSP_ARG_DECL)
{
	if( ! gsl_debug ){
		gsl_debug = add_debug_module(QSP_ARG  "gsl");
	}
}
#endif /* DEBUG */


COMMAND_FUNC( gsl_menu )
{
	static int inited=0;


	if( !inited ){
#ifdef DEBUG
		gsl_debug_init(SINGLE_QSP_ARG);
#endif /* DEBUG */
		auto_version(QSP_ARG  "GSLMENU","VersionId_gslmenu");
		inited=1;
	}

	PUSHCMD(gsl_ctbl,"gsl");
}

#endif /* HAVE_GSL */


