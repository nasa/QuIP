
#include "quip_config.h"

/*	this is a general package for running experiments
 *
 *	The responsibilities of the caller of exprmnt() :
 *
 *		define the following global fuction ptrs:
 *		int (*stim_func)(), (*modrt)();
 *		optional:  resp_func...
 *		initrt points to a routine which is called before each run
 *
 *	stimrt pts to a routine called with two integer args: class, val
 *	modrt pts to a routine to modify stimulus parameters
 */

#include <stdio.h>
#include <math.h>

#include "stc.h"
#include "getbuf.h"
#include "data_obj.h"
#include "quip_menu.h"
#include "veclib_api.h"


/* globals */
// BUG not thread-safe, but probably OK
static float xval_1=1.0, xval_n=0.0;
//float *xval_array=NULL;
static int global_n_xvals=0;

#define LINEAR_STEPS	0
#define LOG_STEPS	1

static const char *step_types[]={"linear","logarithmic"};
static int log_steps=0;


// This is ramp1D !?

#define set_n_xvals(n) _set_n_xvals(QSP_ARG  n)

static void _set_n_xvals(QSP_ARG_DECL  int n)
{
	assert( n > 0 && n <= MAX_X_VALUES );

	if( global_xval_dp != NULL ){
		if( n != OBJ_COLS(global_xval_dp) ){
			sprintf(ERROR_STRING,
	"set_n_xvals:  requested value %d does not match object %s!?",
				n,OBJ_NAME(global_xval_dp));
			warn(ERROR_STRING);
			return;
		}
	}
	global_n_xvals = n;
fprintf(stderr,"number of x-values set to %d\n",n);
}

int _insure_xval_array(SINGLE_QSP_ARG_DECL)
{
	if( global_xval_dp == NULL ){
		if( verbose )
			advise("insure_xval_array:  creating x-value object");
		if( global_n_xvals <= 0 ){
			sprintf(ERROR_STRING,
	"insure_xval_array:  number of x-values not specified, defaulting to %d",MAX_X_VALUES);
			advise(ERROR_STRING);
			set_n_xvals(MAX_X_VALUES);
		}
	} else {
		if( verbose )
			advise("insure_xval_array:  x-value object already exists...");
		return 0;
	}
	global_xval_dp = mk_vec("default_x_values",global_n_xvals,1,prec_for_code(PREC_SP));
	assert(global_xval_dp!=NULL);
	return 0;
}

#define linsteps() _linsteps(SINGLE_QSP_ARG)

static void _linsteps(SINGLE_QSP_ARG_DECL)	/** make linear steps */
{
	float inc;
	int n_xvals;

	assert(global_xval_dp!=NULL);
	n_xvals = OBJ_COLS(global_xval_dp);

	inc=xval_n - xval_1;
	inc /= (n_xvals-1);
	easy_ramp2d(global_xval_dp, xval_1, inc, 0);
}

#define make_steps() _make_steps(SINGLE_QSP_ARG)

static void _make_steps(SINGLE_QSP_ARG_DECL)
{
	if( insure_xval_array() < 0 ) return;

	if( log_steps ){
		xval_1 = (float) log( xval_1 );
		xval_n = (float) log( xval_n );
	}

	linsteps();

	if( log_steps ){
		Vec_Obj_Args oa1;

		clear_obj_args(&oa1);
		SET_OA_DEST(&oa1, global_xval_dp);
		SET_OA_SRC1(&oa1, global_xval_dp);
		set_obj_arg_flags(&oa1);

		platform_dispatch_by_code( FVEXP, &oa1 );
	}
}

static COMMAND_FUNC( do_import_xvals )
{
	Data_Obj *dp;

	dp = pick_obj("float object for x-values");
	if( dp == NULL ) return;

	if( OBJ_PREC(dp) != PREC_SP ){
		sprintf(ERROR_STRING,"import_xvals:  object %s (%s) should have %s precision!?",
			OBJ_NAME(dp),PREC_NAME(OBJ_PREC_PTR(dp)),NAME_FOR_PREC_CODE(PREC_SP));
		warn(ERROR_STRING);
		return;
	}
	if( OBJ_COMPS(dp) != 1 ){
		sprintf(ERROR_STRING,"import_xvals:  object %s should have 1 component!?", OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}
	if( OBJ_ROWS(dp) != 1 ){
		sprintf(ERROR_STRING,"import_xvals:  object %s should have 1 row!?", OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}
	if( OBJ_FRAMES(dp) != 1 ){
		sprintf(ERROR_STRING,"import_xvals:  object %s should have 1 frame!?", OBJ_NAME(dp));
		warn(ERROR_STRING);
		return;
	}
	if( OBJ_COLS(dp) < 2 || OBJ_COLS(dp) > MAX_X_VALUES ){
		sprintf(ERROR_STRING,"import_xvals:  object %s has %d columns, should be in range 2-%d!?",
			OBJ_NAME(dp),OBJ_COLS(dp),MAX_X_VALUES);
		warn(ERROR_STRING);
		return;
	}
	SET_EXPT_XVAL_OBJ(&expt1,dp);
}

static COMMAND_FUNC( do_set_nxvals )
{
	int n;

	n = (int) how_many("number of x values");
	if( n <= 0 || n > MAX_X_VALUES ){
		sprintf(ERROR_STRING,
			"Number of x values must be between 0 and %d",MAX_X_VALUES);
		warn(ERROR_STRING);
		return;
	}
	set_n_xvals(n);

	make_steps();
}

static COMMAND_FUNC( do_set_range )
{
	xval_1 = (int) how_much("zeroeth value");
	xval_n = (int) how_much("last value");

	make_steps();
}

static COMMAND_FUNC( do_set_step_type )
{
	int i;

	i = which_one("step type (linear/logarithmic)",2,step_types);
	if( i < 0 ) warn("invalid step type");
	else log_steps=i;

	make_steps();
}

static COMMAND_FUNC( do_save_xvals )
{
	FILE *fp;
	int i;
	int n_xvals;

	fp=try_nice( nameof("output file"), "w" );
	if( !fp ) return;

	assert(global_xval_dp!=NULL);
	n_xvals = OBJ_COLS(global_xval_dp);

	for(i=0;i<n_xvals;i++){
		float *xv_p;
		xv_p = indexed_data(global_xval_dp,i);
		fprintf(fp,"%f\n",*xv_p);
	}

	fclose(fp);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(xvals_menu,s,f,h)

MENU_BEGIN(xvals)
ADD_CMD( import,	do_import_xvals,	load x values from a data object )
ADD_CMD( save,		do_save_xvals,	save x values to a file )
ADD_CMD( n_vals,	do_set_nxvals,	set number of x values )
ADD_CMD( range,		do_set_range,	set range of x values )
ADD_CMD( step_type,	do_set_step_type,	select linear/logarithmic steps )
MENU_END(xvals)

COMMAND_FUNC( xval_menu )	/** play around with an experiment */
{
	CHECK_AND_PUSH_MENU(xvals);
}

