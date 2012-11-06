#include "quip_config.h"

char VersionId_interpreter_do_cmd[] = QUIP_VERSION_STRING;

/*
 * Possible #define's
 *
 * MY_INTR		user-settable interrupt handler - see builtin.c
 */

#include <stdio.h>

#include "bi_cmds.h"
#include "debug.h"
#include "query.h"
#include "sigpush.h"
#include "callback.h"

#ifdef MAC
extern void do_mac_cmd(void);
#endif

#ifdef MY_INTR
static char *intr_str="";
static char *new_intr_str="";
#include <signal.h>
#endif /* MY_INTR */

static List *cmd_callback_lp=NO_LIST;
static int processing_callbacks=1;

// this is the signal handler, so we don't have a good way to pass qsp?

void comp_cmd(int asig)
{
	/* When we get an interrupt - perhaps we should call this for ALL qsp's???   BUG??? */
	qgivup(SGL_DEFAULT_QSP_ARG);	/* force qword() not to wait for a word */
}

void do_cmd(SINGLE_QSP_ARG_DECL)
{
	if( cmd_callback_lp == NO_LIST )	/* first time? */
		cmd_callback_lp = new_callback_list();

#ifdef MAC
	do_mac_cmd();
#else
	/* this is now done in pushcmd() */
	/* bi_init(); */	/* initialize builtins */

	if( cmd_depth(SINGLE_QSP_ARG) < MIN_CMD_DEPTH ){
		advise("Command stack empty, exiting program");
		nice_exit(0);
	}

#ifdef MY_INTR
	if( *intr_str ) sigpush(SIGINT,comp_cmd);
#endif

	getwcmd(SINGLE_QSP_ARG);

#ifdef MY_INTR
	if( *intr_str ) sigpop(SIGINT);
	if( *new_intr_str ){
		intr_str=new_intr_str;
		new_intr_str="";
	}
	if( HAD_INTERRUPT )
	{
#ifdef DEBUG
if( debug ) advise("suicide by SIGINT");
#endif /* DEBUG */
		/* kill( getpid(), SIGINT ); */
		my_onintr(0);		/* BUG - what should the arg be??? */
	}
#endif /* MY_INTR */

#endif /* ! MAC */

	if( processing_callbacks ){
		call_callback_list(cmd_callback_lp);
	}
}

void add_cmd_callback(void (*func)(VOID))
{
	if( cmd_callback_lp == NO_LIST )
		cmd_callback_lp = new_callback_list();

	add_callback_func(cmd_callback_lp,func);
}

void callbacks_on(SINGLE_QSP_ARG_DECL)
{
	if( processing_callbacks )
		WARN("callbacks_on:  callbacks are already being processed");
	processing_callbacks = 1;
}

void callbacks_off(SINGLE_QSP_ARG_DECL)
{
	if( ! processing_callbacks )
		WARN("callbacks_off:  callbacks are already being inhibited");
	processing_callbacks = 0;
}
