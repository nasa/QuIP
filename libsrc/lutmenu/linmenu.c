#include "quip_config.h"

char VersionId_lutmenu_linmenu[] = QUIP_VERSION_STRING;

#include "linear.h"
#include "lut_cmds.h"
#include "menuname.h"
#include "check_dpy.h"

double crt_gamma	= DEF_GAM;
double crt_vzero	= DEF_VZ;

static COMMAND_FUNC( do_set_n_linear )
{
	int n;

	n=(int)HOW_MANY("number of linearization table entries");
	set_n_linear(QSP_ARG  n);
	CHECK_DPYP("do_set_n_linear")
#ifdef HAVE_X11
	lin_setup(QSP_ARG  current_dpyp->c_lt_dp,crt_gamma,crt_vzero);
#endif /* HAVE_X11 */
}

static COMMAND_FUNC( do_set_gamma )
{
	crt_gamma=HOW_MUCH("exponent for linearization");
	CHECK_DPYP("do_set_gamma")
#ifdef HAVE_X11
	lin_setup(QSP_ARG  current_dpyp->c_lt_dp,crt_gamma,crt_vzero);
#endif /* HAVE_X11 */
}

static COMMAND_FUNC( do_set_vzero )
{
	crt_vzero=(float)HOW_MUCH("voltage offset for linearization");
	CHECK_DPYP("do_set_vzero")
#ifdef HAVE_X11
	lin_setup(QSP_ARG  current_dpyp->c_lt_dp,crt_gamma,crt_vzero);
#endif /* HAVE_X11 */
}

static COMMAND_FUNC( do_default_lin )
{
	crt_gamma= DEF_GAM;
	crt_vzero= DEF_VZ;
	CHECK_DPYP("do_default_lin")
#ifdef HAVE_X11
	lin_setup(QSP_ARG  current_dpyp->c_lt_dp,crt_gamma,crt_vzero);
#endif /* HAVE_X11 */
}

static COMMAND_FUNC( do_new_lintbl )
{
	Data_Obj *lt_dp;
	const char *s;

	s=NAMEOF("name for new linearization table");
	lt_dp = new_lintbl(QSP_ARG  s);
	if( lt_dp == NO_OBJ ) return;

	CHECK_DPYP("do_new_lintbl")
#ifdef HAVE_X11
	current_dpyp->c_lt_dp = lt_dp;
#endif /* HAVE_X11 */
	do_default_lin(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_sel_lintbl )
{
	Data_Obj *lt_dp;

	lt_dp = PICK_OBJ("name of linearization table");
	if( lt_dp == NO_OBJ ) return;

	CHECK_DPYP("do_sel_lintbl")
#ifdef HAVE_X11
	current_dpyp->c_lt_dp = lt_dp;
#endif /* HAVE_X11 */
}

Command gamma_ctbl[]={
{ "new",	do_new_lintbl,	"create new linearization table"	},
{ "select",	do_sel_lintbl,	"select linearization table for subsequent operations"	},
{ "default",	do_default_lin,	"default linearization values"		},
{ "gamma",	do_set_gamma,	"set linearization exponent"		},
{ "vzero",	do_set_vzero,	"set linearization voltage offset"	},
{ "nlevels",	do_set_n_linear,"set number of entries in lin. table"	},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif /* ! MAC */
{ NULL,		NULL,		NULL					}
};

COMMAND_FUNC( do_linearize )
{
	PUSHCMD(gamma_ctbl,GAMMA_MENU_NAME);
}

