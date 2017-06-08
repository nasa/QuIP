
#include "quip_config.h"

#include <stdio.h>
#include "quip_prot.h"
#include "data_obj.h"
#include "nrm_api.h"

#ifdef HAVE_NUMREC

#include "numrec.h"


#ifdef DEBUG
static void numrec_debug_init(SINGLE_QSP_ARG_DECL);
#endif /* DEBUG */


static COMMAND_FUNC( do_choldc )
{
	Data_Obj *a_dp, *p_dp;

	a_dp=PICK_OBJ( "input/output matrix" );
	p_dp=PICK_OBJ( "elements diagonal matrix" );

	if ( a_dp == NULL || p_dp == NULL )
		return;

	printf("nrmenu:nrmenu.c:pickobj %f\n", *((float *)OBJ_DATA_PTR(a_dp)));

	dp_choldc(a_dp,p_dp);
}


static COMMAND_FUNC( do_svd )
{
	Data_Obj *a_dp, *w_dp, *v_dp;

	a_dp=PICK_OBJ( "input matrix" );
	w_dp=PICK_OBJ( "vector for singular values" );
	v_dp=PICK_OBJ( "output v matrix" );

	if( a_dp == NULL || w_dp == NULL || v_dp == NULL )
		return;

	dp_svd(a_dp,w_dp,v_dp);
}

static COMMAND_FUNC( do_svbksb )
{
	Data_Obj *u_dp, *w_dp, *v_dp;
	Data_Obj *x_dp, *b_dp;

	x_dp=PICK_OBJ( "result vector for coefficients" );
	u_dp=PICK_OBJ( "U matrix" );
	w_dp=PICK_OBJ( "vector of singular values" );
	v_dp=PICK_OBJ( "V matrix" );
	b_dp=PICK_OBJ( "data vector" );

	if( u_dp == NULL || w_dp == NULL || v_dp == NULL ||
		x_dp == NULL || b_dp == NULL )
		return;

	dp_svbksb(x_dp,u_dp,w_dp,v_dp,b_dp);
}

static COMMAND_FUNC( do_jacobi )
{
	Data_Obj *a_dp,*d_dp,*v_dp;
	int nrot;

	v_dp = PICK_OBJ("destination matrix for eigenvectors");
	d_dp = PICK_OBJ("destination vector for eigenvalues");
	a_dp = PICK_OBJ("input matrix");

	if( v_dp == NULL || d_dp == NULL || a_dp == NULL ) return;

	dp_jacobi(QSP_ARG  v_dp,d_dp,a_dp,&nrot);

	// sprintf(msg_str,"%d rotations performed",nrot);
	// prt_msg(msg_str);
}

static COMMAND_FUNC( do_eigsrt )
{
	Data_Obj *v_dp, *d_dp;
	
	v_dp = PICK_OBJ("matrix of eigenvectors from Jacobi");
	d_dp = PICK_OBJ("vector of eigenvalues from Jacobi");

	if( v_dp == NULL || d_dp == NULL ) return;

	dp_eigsrt(QSP_ARG  v_dp,d_dp);
}

static COMMAND_FUNC( do_moment )
{
	Data_Obj *d_dp;
	
	d_dp = PICK_OBJ("array of data");
	/* How to get the values of ave and sdev??? */
	dp_moment(QSP_ARG  d_dp);
}

static COMMAND_FUNC( do_plgndr )
{
	float x,r;
	int l,m;

	l=(int)HOW_MANY("l");
	m=(int)HOW_MANY("m");
	x=(float)HOW_MUCH("x");

	if( m < 0 || m > l ){
		sprintf(ERROR_STRING,"parameter m (%d) must be between 0 and l (%d)",m,l);
		WARN(ERROR_STRING);
		return;
	}
	if( x < -1 || x > 1 ){
		sprintf(ERROR_STRING,"parameter x (%g) must be between -1 and 1",x);
		WARN(ERROR_STRING);
		return;
	}

	r = plgndr(l,m,x);

	sprintf(msg_str,"P_%d,%d(%g) = %g",l,m,x,r);
	prt_msg(msg_str);
}

static int polish_roots=1;

static COMMAND_FUNC( do_set_polish )
{
	polish_roots = ASKIF( "polish roots with laguerre method" );
}

static COMMAND_FUNC( do_zroots )
{
	Data_Obj *a_dp, *r_dp;

	r_dp=PICK_OBJ( "complex destination vector for roots" );
	a_dp=PICK_OBJ( "polynomial coefficient vector" );

	if( r_dp == NULL || a_dp == NULL ) return;

	dp_zroots(r_dp,a_dp,polish_roots);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(numrec_menu,s,f,h)

MENU_BEGIN(numrec)
ADD_CMD( svd,		do_svd,		singular value decomposition )
ADD_CMD( svbk,		do_svbksb,	back substitute into SVD )
ADD_CMD( jacobi,	do_jacobi,	compute eigenvectors & eigenvalues )
ADD_CMD( eigsrt,	do_eigsrt,	sort the eigenvalues into descending order )
ADD_CMD( moment,	do_moment,	mean value & standard deviation of a vector )
ADD_CMD( plgndr,	do_plgndr,	compute legendre polynomial )
ADD_CMD( zroots,	do_zroots,	compute roots of a polynomial )
ADD_CMD( polish,	do_set_polish,	enable (default) or disable root polishing in zroots() )
ADD_CMD( choldc,	do_choldc,	cholesky decomposition )
MENU_END(numrec)

#ifdef DEBUG
int numrec_debug=0;

static void numrec_debug_init(SINGLE_QSP_ARG_DECL)
{
	if( ! numrec_debug ){
		numrec_debug = add_debug_module(QSP_ARG  "numrec");
	}
}
#endif /* DEBUG */


COMMAND_FUNC( do_nr_menu )
{
	static int inited=0;


	if( !inited ){
#ifdef DEBUG
		numrec_debug_init(SINGLE_QSP_ARG);
#endif /* DEBUG */
		inited=1;
	}

	PUSH_MENU(numrec);
}

#endif /* HAVE_NUMREC */


