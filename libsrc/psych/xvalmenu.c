
#include "quip_config.h"

char VersionId_psych_xvalmenu[] = QUIP_VERSION_STRING;

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


/* local prototypes */

static COMMAND_FUNC( do_load_xvals );
static COMMAND_FUNC( make_steps );
static void linsteps(void);

static float xval_1=1.0, xval_n=0.0;

/* globals */
float xval_array[MAXVALS];
int _nvals;


static COMMAND_FUNC( do_load_xvals )
{
	rdxvals( QSP_ARG  NAMEOF("x value file") );
}

static COMMAND_FUNC( do_set_nxvals )
{
	_nvals=HOW_MANY("number of x values");
	if( _nvals<=0 || _nvals >MAXVALS ){
		sprintf(error_string,
			"Number of x values must be between 0 and %d",MAXVALS);
		WARN(error_string);
	}
}

static COMMAND_FUNC( do_set_range )
{
	xval_1 = HOW_MUCH("zeroeth value");
	xval_n = HOW_MUCH("last value");

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
	if( log_steps ){
		int i;

		xval_1 = log( xval_1 );
		xval_n = log( xval_n );
		linsteps();
		for(i=0;i<_nvals;i++)
			xval_array[i]=exp( xval_array[i] );
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

	inc=xval_n - xval_1;
	inc /= (_nvals-1);
	xval_array[0] = xval_1;
	for(i=0;i<_nvals;i++)
		xval_array[i]=xval_1+i*inc;
}

Command xval_ctbl[]={
{ "load",	do_load_xvals,	"load x values from a file"		},
{ "save",	do_save_xvals,	"save x values to a file"		},
{ "n_vals",	do_set_nxvals,	"set number of x values"		},
{ "range",	do_set_range,	"set range of x values"			},
{ "step_type",	do_set_steps,	"select linear/logarithmic steps"	},
{ "quit",	popcmd,		"quit"					},
{ NULL,		NULL,		NULL					}
};

COMMAND_FUNC( xval_menu )	/** play around with an experiment */
{
	PUSHCMD(xval_ctbl,"xvals");
}

