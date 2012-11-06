#include "quip_config.h"

char VersionId_interpreter_adjust[] = QUIP_VERSION_STRING;

/**/
/**			adaptive adjustment from keyboard	**/
/*
Revisions:
	1-3-94	P. Stassart		Kbhit flushing
*/
/**/

#include <stdio.h>

#include "query.h"		/* push_input_file() */
#include "boolean.h"
#include "savestr.h"
#include "param_api.h"
#include "adjust.h"
#include "menuname.h"
#include "submenus.h"

/* local prototypes */
static COMMAND_FUNC( increment );
static COMMAND_FUNC( decrement );
static COMMAND_FUNC( quit_adj );
static COMMAND_FUNC( doachng );
static COMMAND_FUNC( set_a_sels );

static void end_adj_redir(SINGLE_QSP_ARG_DECL);
static void do_nothing(QSP_ARG_DECL float);

static float _value, _incr, _maxincr, _minincr, _startincr, _hival, _loval;
static int _lasttr;
int adjusting;		/* a global */

#define DEV_KEYBOARD		1
#define DEV_MOUSE		2
#define DEV_STDIN		3
static int input_device=DEV_KEYBOARD;
static int device_in_use=DEV_STDIN;
static int _adj_ql;

Param adjptbl[]={
{ "value",	"current value",	FLT_PARAM(&_value)	},
{ "incr",	"current increment",	FLT_PARAM(&_incr)	},
{ "maxincr",	"max. incr step size",	FLT_PARAM(&_maxincr)	},
{ "minincr",	"min. incr step size",	FLT_PARAM(&_minincr)	},
{ "startincr",	"start incr step size",	FLT_PARAM(&_startincr)	},
{ "hival",	"upper limit",		FLT_PARAM(&_hival)	},
{ "loval",	"lower limit",		FLT_PARAM(&_loval)	},
{ NULL,	 NULL,		(long)NULL, NULL	}
};


static void do_nothing(QSP_ARG_DECL float v)
{
	WARN("no adjustment function specified");
}

#define STEP_INCR	(float)1.8 /* increase incr by this fact after NUM_SAME */
#define NUM_SAME	2
#define WHITTLE_FACTOR	(float)2.0	/* divide incr by this at reversals */

static void (*adj_func)(QSP_ARG_DECL float)=do_nothing;

static COMMAND_FUNC( increment )
{
	if( _lasttr < 0 ){
		_incr /= WHITTLE_FACTOR;
		_lasttr=0;
	}
	_lasttr++;
	if( _lasttr > NUM_SAME ) _incr *= STEP_INCR;
	if( _incr > _maxincr ) _incr = _maxincr;
	if( _incr < _minincr ) _incr = _minincr;
	_value += _incr;
	if( _value > _hival ) _value=_hival;

	(*adj_func)(QSP_ARG _value);
}

static COMMAND_FUNC( decrement )
{
	if( _lasttr > 0 ){
		_incr /= WHITTLE_FACTOR;
		_lasttr=0;
	}
	_lasttr--;
	if( _lasttr < (-NUM_SAME) ) _incr *= STEP_INCR;
	if( _incr > _maxincr ) _incr = _maxincr;
	if( _incr < _minincr ) _incr = _minincr;
	_value -= _incr;
	if( _value < _loval ) _value=_loval;

	(*adj_func)(QSP_ARG _value);
}

static COMMAND_FUNC( quit_adj )
{
	if( device_in_use != DEV_STDIN )
		end_adj_redir(SINGLE_QSP_ARG);

	popcmd(SINGLE_QSP_ARG);
	adjusting=0;
}

static COMMAND_FUNC( doachng )
{
	chngp( QSP_ARG adjptbl);
}

/* the body of set_a_sels() has to come after the command table declaration,
 * because it modifies the contents of the command table.
 */

#define N_INPUT_DEVICES		4
static const char *input_device_names[N_INPUT_DEVICES]={
	"keyboard","mouse","stdin","-"
};

static COMMAND_FUNC( do_select_dev )
{
	int i;

	i=which_one( QSP_ARG "input device",N_INPUT_DEVICES,input_device_names);
	if( i < 0 ) return;

	switch(i){
		case 0:  input_device=DEV_KEYBOARD; break;
		case 1:  input_device=DEV_MOUSE; break;
		case 2:
		case 3:  input_device=DEV_STDIN; break;
	}
}


/* Redirect the input based on device_in_use */

static void push_adj_input(SINGLE_QSP_ARG_DECL)
{
	if( device_in_use == DEV_MOUSE ){
		WARN("Sorry, can only redirect to the mouse on the PC");
		device_in_use = DEV_STDIN;
	} else if( device_in_use == DEV_KEYBOARD ){
		push_input_file( QSP_ARG "/dev/tty");
		redir(  QSP_ARG  tfile(SINGLE_QSP_ARG) );
	}
}

static COMMAND_FUNC( do_adj_redir )
{
	/* we often call this from a script, but we want the user
	 * to make adjustments until satisfied...  therefore we redirect
	 * to the tty file (or possibly the mouse on the PC)
	 */

	if( input_device != DEV_STDIN ){
		device_in_use = input_device;
		push_adj_input(SINGLE_QSP_ARG);
		_adj_ql=tell_qlevel(SINGLE_QSP_ARG);
sprintf(ERROR_STRING,"qlevel = %d",_adj_ql);
advise(ERROR_STRING);
	}
}


Command adjctbl[]={
{ "change",	doachng,	"change adjustment paramters"		},
{ "increment",	increment,	"increment"				},
{ "decrement",	decrement,	"decrement"				},
{ "accept",	quit_adj,	"quit"					},
{ "selectors",	set_a_sels,	"set adjustor selector words"		},
{ "device",	do_select_dev,	"select user input device"		},
{ "redir",	do_adj_redir,	"redirect to selected input device"	},
{ NULL_COMMAND								}
};


COMMAND_FUNC( do_adjust )
{

	_incr = _startincr;
	adjusting=1;
	PUSHCMD(adjctbl,ADJUST_MENU_NAME);
}

static void end_adj_redir(SINGLE_QSP_ARG_DECL)
{
	/* now pop the input file if necessary */
	if( device_in_use != DEV_STDIN ){
		while( tell_qlevel(SINGLE_QSP_ARG) >= _adj_ql ) popfile(SINGLE_QSP_ARG);
	}
	device_in_use=DEV_STDIN;
}


/* BUG?  why incr *and* startincr??? */

void setaps(float loval,float hival,float start,float incr,
	    float maxincr,float minincr,float startincr)
     /** set adjustment parameters */
{
	_loval	= (float)loval;
	_hival	= (float)hival;
	_value	= (float)start;
	_incr		= (float)incr;
	_maxincr	= (float)maxincr;
	_minincr	= (float)minincr;
	_startincr	= (float)startincr;
	_lasttr	= 0;
}

void setup_adjuster(float *flist,int n)
{
	if( n-- > 0 ) _loval			= *flist++;
	if( n-- > 0 ) _hival			= *flist++;
	if( n-- > 0 ) _value			= *flist++;
	if( n-- > 0 ) _incr			= *flist++;
	if( n-- > 0 ) _maxincr		= *flist++;
	if( n-- > 0 ) _minincr		= *flist++;
	if( n-- > 0 ) _startincr	= *flist++;
	_lasttr							= 0;
}

float adj_val()					/** current value */
{
	return(_value);
}

void set_adj_func(void (*func)(QSP_ARG_DECL float))
{
	adj_func=func;
}


/* this declaration has to come after the declaration of adjctbl[] ... */

static COMMAND_FUNC( set_a_sels )		/* set adjustment selectors */
{
	const char *s;

	s=savestr( nameof( QSP_ARG "increment command word") );
	adjctbl[1].cmd_sel = s;
	s=savestr( nameof( QSP_ARG "decrement command word") );
	adjctbl[2].cmd_sel = s;
	s=savestr( nameof( QSP_ARG "accept command word") );
	adjctbl[3].cmd_sel = s;

	/* BUG with the new implementation of command menus,
	 * the selector words won't get reset!?
	 * Need a reload table function...
	 */
	reload_menu(QSP_ARG  ADJUST_MENU_NAME,adjctbl);

}

