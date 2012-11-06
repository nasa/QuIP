
#include "quip_config.h"

char VersionId_nrmenu_nrmenu[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include "data_obj.h"
#include "debug.h"
#include "menuname.h"
#include "version.h"

#include "nrm_api.h"

#ifdef HAVE_NUMREC

#include "numrec.h"


static COMMAND_FUNC( do_choldc );
static COMMAND_FUNC( do_svd );
static COMMAND_FUNC( do_svbksb );
static COMMAND_FUNC( do_jacobi );
static COMMAND_FUNC( do_eigsrt );
static COMMAND_FUNC( do_moment );

#ifdef DEBUG
static void numrec_debug_init(SINGLE_QSP_ARG_DECL);
#endif /* DEBUG */


static COMMAND_FUNC( do_choldc )
{
	Data_Obj *a_dp, *p_dp;

	a_dp=PICK_OBJ( "input/output matrix" );
	p_dp=PICK_OBJ( "elements diagonal matrix" );

	if ( a_dp == NO_OBJ || p_dp == NO_OBJ )
		return;

	printf("nrmenu:nrmenu.c:pickobj %f\n", *((float *)a_dp->dt_data));

	dp_choldc(a_dp,p_dp);
}


static COMMAND_FUNC( do_svd )
{
	Data_Obj *a_dp, *w_dp, *v_dp;

	a_dp=PICK_OBJ( "input matrix" );
	w_dp=PICK_OBJ( "vector for singular values" );
	v_dp=PICK_OBJ( "output v matrix" );

	if( a_dp == NO_OBJ || w_dp == NO_OBJ || v_dp == NO_OBJ )
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

	if( u_dp == NO_OBJ || w_dp == NO_OBJ || v_dp == NO_OBJ ||
		x_dp == NO_OBJ || b_dp == NO_OBJ )
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

	if( v_dp == NO_OBJ || d_dp == NO_OBJ || a_dp == NO_OBJ ) return;

	dp_jacobi(v_dp,d_dp,a_dp,&nrot);

	// sprintf(msg_str,"%d rotations performed",nrot);
	// prt_msg(msg_str);
}

static COMMAND_FUNC( do_eigsrt )
{
	Data_Obj *v_dp, *d_dp;
	
	v_dp = PICK_OBJ("matrix of eigenvectors from Jacobi");
	d_dp = PICK_OBJ("vector of eigenvalues from Jacobi");

	if( v_dp == NO_OBJ || d_dp == NO_OBJ ) return;

	dp_eigsrt(v_dp,d_dp);
}

static COMMAND_FUNC( do_moment )
{
	Data_Obj *d_dp;
	
	d_dp = PICK_OBJ("array of data");
	/* How to get the values of ave and sdev??? */
	dp_moment(d_dp);
}

static COMMAND_FUNC( do_plgndr )
{
	float x,r;
	int l,m;

	l=HOW_MANY("l");
	m=HOW_MANY("m");
	x=HOW_MUCH("x");

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

	if( r_dp == NO_OBJ || a_dp == NO_OBJ ) return;

	dp_zroots(r_dp,a_dp,polish_roots);
}

Command nr_ctbl[]={
{ "svd",	do_svd,		"singular value decomposition"			},
{ "svbk",	do_svbksb,	"back substitute into SVD"			},
{ "jacobi",	do_jacobi,	"compute eigenvectors & eigenvalues"		},
{ "eigsrt",	do_eigsrt,	"sort the eigenvalues into descending order"	}, 
{ "moment",	do_moment,	"mean value & standard deviation of a vector"	},
{ "plgndr",	do_plgndr,	"compute legendre polynomial"			},
{ "zroots",	do_zroots,	"compute roots of a polynomial"			},
{ "polish",	do_set_polish,	"enable (default) or disable root polishing in zroots()"	},
{ "choldc",     do_choldc,      "cholesky decomposition"			},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"					},
#endif
{ NULL_COMMAND									}
};

#ifdef DEBUG
int numrec_debug=0;

static void numrec_debug_init(SINGLE_QSP_ARG_DECL)
{
	if( ! numrec_debug ){
		numrec_debug = add_debug_module(QSP_ARG  "numrec");
	}
}
#endif /* DEBUG */


COMMAND_FUNC( nrmenu )
{
	static int inited=0;


	if( !inited ){
#ifdef DEBUG
		numrec_debug_init(SINGLE_QSP_ARG);
#endif /* DEBUG */
		auto_version(QSP_ARG  "NRMENU","VersionId_nrmenu");
		inited=1;
	}

	PUSHCMD(nr_ctbl,"numrec");
}

#endif /* HAVE_NUMREC */


