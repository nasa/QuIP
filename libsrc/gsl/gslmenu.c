
#include "quip_config.h"

#ifdef HAVE_GSL

#include <stdio.h>
#include "quip_prot.h"
#include "data_obj.h"
#include "gslprot.h"

#ifdef QUIP_DEBUG
static void gsl_debug_init(SINGLE_QSP_ARG_DECL);
#endif /* QUIP_DEBUG */


static COMMAND_FUNC( do_gsl_svd )
{
	Data_Obj *a_dp, *w_dp, *v_dp;

	a_dp=PICK_OBJ( "input matrix" );
	w_dp=PICK_OBJ( "vector for singular values" );
	v_dp=PICK_OBJ( "output v matrix" );

	if( a_dp == NO_OBJ || w_dp == NO_OBJ || v_dp == NO_OBJ )
		return;

	gsl_svd(QSP_ARG  a_dp,w_dp,v_dp);
}

static COMMAND_FUNC( do_gsl_solve )
{
	Data_Obj *u_dp, *v_dp, *w_dp, *x_dp, *b_dp;

	x_dp = PICK_OBJ("Vector of unknown coefficients");
	u_dp = PICK_OBJ("U matrix");
	w_dp = PICK_OBJ("Singular values");
	v_dp = PICK_OBJ("V matrix");
	b_dp = PICK_OBJ("Vector of input data");

	if( u_dp == NO_OBJ || w_dp == NO_OBJ || v_dp == NO_OBJ ||
		x_dp == NO_OBJ || b_dp == NO_OBJ )
		return;

	gsl_solve(QSP_ARG  x_dp,u_dp,w_dp,v_dp,b_dp);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(gsl_menu,s,f,h)

MENU_BEGIN(gsl)
ADD_CMD( gslsvd,	do_gsl_svd,		singular value decomposition )
ADD_CMD( solve,		do_gsl_solve,		solve linear system )
MENU_END(gsl)

#ifdef QUIP_DEBUG
int gsl_debug=0;

static void gsl_debug_init(SINGLE_QSP_ARG_DECL)
{
	if( ! gsl_debug ){
		gsl_debug = add_debug_module(QSP_ARG  "gsl");
	}
}
#endif /* QUIP_DEBUG */


COMMAND_FUNC( do_gsl_menu )
{
	static int inited=0;


	if( !inited ){
#ifdef QUIP_DEBUG
		gsl_debug_init(SINGLE_QSP_ARG);
#endif /* QUIP_DEBUG */
		inited=1;
	}

	PUSH_MENU(gsl);
}
	
#endif /* HAVE_GSL */


