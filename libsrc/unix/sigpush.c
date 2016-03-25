#include "quip_config.h"

/* nested signal handlers */

#include <stdio.h>

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

#include "sigpush.h"
#include "quip_prot.h"

/* use sigpush() instead of signal() */
/* sigpop() restores previous handler */

#define MAXDEPTH	4

static void (*handler_stack[NSIG][MAXDEPTH])(int sig);

static int sdepth[NSIG];
static int spinited=0;
static int no_sigs=0;

/*
 * Disable action on calls to sigpush() and sigpop()
 */

void inhibit_sigs()
{ no_sigs=1; }

/*
 * like signal(), but calls stack when sigpush() is used with sigpop()
 */

/* jnguyen: Neither version seems to be right.
 *
 * void sigpush(int sig, void (*action)(int asig))
 */

void sigpush(int sig, void (*action)(int asig))
{
	int i;

	if( no_sigs ) return;

	if( !spinited ){
		for(i=0;i<NSIG;i++) sdepth[i]=(-1);
		spinited=1;
	}

	sdepth[sig]++;
	while( sdepth[sig] >= MAXDEPTH ){
		sdepth[sig]--;
		NWARN("sigpush:  too many pushes");
	}
	handler_stack[sig][sdepth[sig]] = action;
	signal(sig,action);
}

#ifdef NOT_USED
void sigreinstate(int sig)		/* give another signal() call to top of stack */
{
	signal( sig, handler_stack[sig][sdepth[sig]] );
}
#endif /* NOT_USED */

/*
 * Pop current handler from the signal handler stack
 */

void sigpop(int sig)
		/* which signal */
{
	int i;

	if( no_sigs ) return;

	if( !spinited ){
		for(i=0;i<NSIG;i++) sdepth[i]=(-1);
		spinited=1;
	}

	if( sdepth[sig] < 0 ){
		NWARN("sigpop:  nothing to pop");
		return;
	}
	sdepth[sig]--;
	if( sdepth[sig] < 0 ) signal(sig,SIG_DFL);
	else signal(sig,handler_stack[sig][sdepth[sig]]);

	/* return( handler_stack[sig][ sdepth[sig]+1 ] ); */
}


