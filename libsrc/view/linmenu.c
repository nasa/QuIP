#include "quip_config.h"

#include "quip_prot.h"
#include "linear.h"
#include "lut_cmds.h"
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
	lin_setup(QSP_ARG  DPA_LINTBL_OBJ(current_dpyp),crt_gamma,crt_vzero);
#endif /* HAVE_X11 */
}

static COMMAND_FUNC( do_set_gamma )
{
	crt_gamma=HOW_MUCH("exponent for linearization");
	CHECK_DPYP("do_set_gamma")
#ifdef HAVE_X11
	lin_setup(QSP_ARG  DPA_LINTBL_OBJ(current_dpyp),crt_gamma,crt_vzero);
#endif /* HAVE_X11 */
}

static COMMAND_FUNC( do_set_vzero )
{
	crt_vzero=(float)HOW_MUCH("voltage offset for linearization");
	CHECK_DPYP("do_set_vzero")
#ifdef HAVE_X11
	lin_setup(QSP_ARG  DPA_LINTBL_OBJ(current_dpyp),crt_gamma,crt_vzero);
#endif /* HAVE_X11 */
}

static COMMAND_FUNC( do_default_lin )
{
	crt_gamma= DEF_GAM;
	crt_vzero= DEF_VZ;
	CHECK_DPYP("do_default_lin")
#ifdef HAVE_X11
	lin_setup(QSP_ARG  DPA_LINTBL_OBJ(current_dpyp),crt_gamma,crt_vzero);
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
	DPA_LINTBL_OBJ(current_dpyp) = lt_dp;
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
	DPA_LINTBL_OBJ(current_dpyp) = lt_dp;
#endif /* HAVE_X11 */
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(gamma_menu,s,f,h)

MENU_BEGIN(gamma)
ADD_CMD( new,		do_new_lintbl,	create new linearization table )
ADD_CMD( select,	do_sel_lintbl,	select linearization table for subsequent operations )
ADD_CMD( default,	do_default_lin,	default linearization values )
ADD_CMD( gamma,		do_set_gamma,	set linearization exponent )
ADD_CMD( vzero,		do_set_vzero,	set linearization voltage offset )
ADD_CMD( nlevels,	do_set_n_linear,	set number of entries in lin. table )
MENU_END(gamma)

COMMAND_FUNC( do_linearize )
{
	PUSH_MENU(gamma);
}

