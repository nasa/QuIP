
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
			advise("insure_xval_array:  no x-value array, creating with maximum size");
			set_n_xvals(MAX_X_VALUES);
		}
		xval_array = (float *) getbuf( _nvals * sizeof(float) );
	} else {
advise("insure_xval_array:  x-value array already exists");
	}
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
ADD_CMD( save,		do_save_xvals,	save x values to a file )
ADD_CMD( n_vals,	do_set_nxvals,	set number of x values )
ADD_CMD( range,		do_set_range,	set range of x values )
ADD_CMD( step_type,	do_set_steps,	select linear/logarithmic steps )
MENU_END(xvals)

COMMAND_FUNC( xval_menu )	/** play around with an experiment */
{
	CHECK_AND_PUSH_MENU(xvals);
}

