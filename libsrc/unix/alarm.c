
#include "quip_config.h"

#ifndef BUILD_FOR_IOS

// This is for a unix implementation using setitimer

#include <stdio.h>
#include "quip_prot.h"

#ifdef HAVE_SETITIMER
#ifdef HAVE_SYS_TIME_H
#include <math.h>
#include <signal.h>
#include <sys/time.h>
#endif /* HAVE_SYS_TIME_H */
#endif /* HAVE_SETITIMER */

/* Unix style alarm implementation - put this somewhere else */

static const char *timer_script = NULL;

void _set_alarm_script(QSP_ARG_DECL  const char *s)
{
	if( timer_script != NULL ) rls_str(timer_script);
	timer_script = savestr(s);
}

void _set_alarm_time(QSP_ARG_DECL  float f)
{
#ifdef HAVE_SETITIMER
	struct itimerval itv;
	int status;

	if( f <= 0 ){
		// cancel any pending alarm...
		f=0;
	}

	// make it_interval non-zero for recurring alarms
	itv.it_interval.tv_sec = 0;
	itv.it_interval.tv_usec = 0;

	itv.it_value.tv_sec = floor(f);
	itv.it_value.tv_usec = floor( 1000000 * ( f - floor(f) ) );

/*	if( signal(SIGARLM,my_alarm) == SIG_ERR ){
		tell_sys_error("signal");
		return;
	}
*/
	if( (status=setitimer(ITIMER_REAL,&itv,NULL)) < 0 ){
		tell_sys_error("setitimer");
	}
#else /* !HAVE_SETITIMER */
	warn("set_alarm_time:  no support for setitimer!?");
#endif /* !HAVE_SETITIMER */
}

#endif /* ! BUILD_FOR_IOS */

