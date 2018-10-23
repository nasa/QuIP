
#include "quip_config.h"

/*	this is a general package for running experiments
 *
 *	The responsibilities of the caller of exprmnt() :
 *
 *		define the following global fuction ptrs:
 *		int (*stmrt)(), (*modrt)();
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


/* local prototypes */

static COMMAND_FUNC( do_load_xvals );

/* globals */
// BUG not thread-safe, but probably OK
static float xval_1=1.0, xval_n=0.0;
float *xval_array=NULL;
int _nvals=0;

#define LINEAR_STEPS	0
#define LOG_STEPS	1

static const char *step_types[]={"linear","logarithmic"};
static int log_steps=0;


// This is ramp1D !?

void set_n_xvals(int n)
{
	assert( n > 0 && n <= MAX_X_VALUES );

	if( n != _nvals && xval_array != NULL ){
		givbuf(xval_array);
		xval_array = getbuf( n * sizeof(float) );
	}
	_nvals = n;
}

int _insure_xval_array(SINGLE_QSP_ARG_DECL)
{
	if( xval_array == NULL ){
advise("insure_xval_array:  creating x-value array");
		if( _nvals <= 0 ){
			sprintf(ERROR_STRING,
	"insure_xval_array:  number of x-values not specified, defaulting to %d",MAX_X_VALUES);
			advise(ERROR_STRING);
			set_n_xvals(MAX_X_VALUES);
		}
	} else {
advise("insure_xval_array:  freeing existing x-value array...");
		givbuf(xval_array);
	}
	xval_array = (float *) getbuf( _nvals * sizeof(float) );
	return 0;
}

static void linsteps(void)	/** make linear steps */
{
	float inc;
	int i;

	inc=xval_n - xval_1;
	inc /= (_nvals-1);
	xval_array[0] = xval_1;
	for(i=0;i<_nvals;i++)
		xval_array[i]=xval_1+i*inc;
}

#define make_steps() _make_steps(SINGLE_QSP_ARG)

static void _make_steps(SINGLE_QSP_ARG_DECL)
{
	if( insure_xval_array() < 0 ) return;

	if( log_steps ){
		int i;

		xval_1 = (float) log( xval_1 );
		xval_n = (float) log( xval_n );
		linsteps();
		for(i=0;i<_nvals;i++)
			xval_array[i]=(float) exp( xval_array[i] );
	} else linsteps();
}


static COMMAND_FUNC( do_load_xvals )
{
	rdxvals( QSP_ARG  nameof("x value file") );
}

static COMMAND_FUNC( do_import_xvals )
{
	Data_Obj *dp;
	int i; float *p;

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

	set_n_xvals( OBJ_COLS(dp) );
	if( insure_xval_array() < 0 ) return;
	p = OBJ_DATA_PTR(dp);
	for(i=0;i<_nvals;i++){
		xval_array[i] = *p;
		p += OBJ_PXL_INC(dp);
	}
}

static COMMAND_FUNC( do_set_nxvals )
{
	int n;

	n = (int) HOW_MANY("number of x values");
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
	xval_1 = (int) HOW_MUCH("zeroeth value");
	xval_n = (int) HOW_MUCH("last value");

	make_steps();
}

static COMMAND_FUNC( do_set_steps )
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

	fp=try_nice( nameof("output file"), "w" );
	if( !fp ) return;

	for(i=0;i<_nvals;i++)
		fprintf(fp,"%f\n",xval_array[i]);

	fclose(fp);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(xvals_menu,s,f,h)

MENU_BEGIN(xvals)
ADD_CMD( load,		do_load_xvals,	load x values from a file )
ADD_CMD( import,	do_import_xvals,	load x values from a data object )
ADD_CMD( save,		do_save_xvals,	save x values to a file )
ADD_CMD( n_vals,	do_set_nxvals,	set number of x values )
ADD_CMD( range,		do_set_range,	set range of x values )
ADD_CMD( step_type,	do_set_steps,	select linear/logarithmic steps )
MENU_END(xvals)

COMMAND_FUNC( xval_menu )	/** play around with an experiment */
{
	CHECK_AND_PUSH_MENU(xvals);
}

