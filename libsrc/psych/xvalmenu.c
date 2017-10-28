
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
static COMMAND_FUNC( make_steps );
static void linsteps(void);

static float xval_1=1.0, xval_n=0.0;

/* globals */
float *xval_array=NULL;
int _nvals=0;


static COMMAND_FUNC( do_load_xvals )
{
	rdxvals( QSP_ARG  NAMEOF("x value file") );
}

static COMMAND_FUNC( do_set_nxvals )
{
	_nvals= (int) HOW_MANY("number of x values");
	if( _nvals<=0 || _nvals >MAX_X_VALUES ){
		sprintf(ERROR_STRING,
			"Number of x values must be between 0 and %d",MAX_X_VALUES);
		WARN(ERROR_STRING);
	}
}

static COMMAND_FUNC( do_set_range )
{
	xval_1 = (int) HOW_MUCH("zeroeth value");
	xval_n = (int) HOW_MUCH("last value");

	make_steps(SINGLE_QSP_ARG);
}

#define LINEAR_STEPS	0
#define LOG_STEPS	1

static const char *step_types[]={"linear","logarithmic"};
static int log_steps=0;

static COMMAND_FUNC( do_set_steps )
{
	int i;

	i = WHICH_ONE("step type (linear/logarithmic)",2,step_types);
	if( i < 0 ) WARN("invalid step type");
	else log_steps=i;
}

static COMMAND_FUNC( make_steps )
{
	if( xval_array == NULL )
		xval_array = (float *) getbuf( _nvals * sizeof(float) );

	if( log_steps ){
		int i;

		xval_1 = (float) log( xval_1 );
		xval_n = (float) log( xval_n );
		linsteps();
		for(i=0;i<_nvals;i++)
			xval_array[i]=(float) exp( xval_array[i] );
	} else linsteps();
}

static COMMAND_FUNC( do_save_xvals )
{
	FILE *fp;
	int i;

	fp=TRYNICE( NAMEOF("output file"), "w" );
	if( !fp ) return;

	for(i=0;i<_nvals;i++)
		fprintf(fp,"%f\n",xval_array[i]);

	fclose(fp);
}

static void linsteps(void)	/** make linear steps */
{
	float inc;
	int i;

	if( xval_array == NULL )
		xval_array = (float *) getbuf( _nvals * sizeof(float) );

	inc=xval_n - xval_1;
	inc /= (_nvals-1);
	xval_array[0] = xval_1;
	for(i=0;i<_nvals;i++)
		xval_array[i]=xval_1+i*inc;
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

