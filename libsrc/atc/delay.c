#include "quip_config.h"

char VersionId_atc_delay[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>		/* gettimeofday() */
#include <signal.h>		/* gettimeofday() */

//#include "myerror.h"		/* warn() */
#include "query.h"
#include "debug.h"		/* verbose */

/* local prototypes */
static void end_delay(int signum);

static int delay_over;

static void end_delay(int signum)
{
	delay_over=1;
	signal(SIGALRM,SIG_IGN);
}

void delay(u_long msec)
{
	struct timeval tv1,tv2;
	struct timezone tz1;
	struct itimerval itv,o_itv;
	u_long actual;

	/* record the time at the beginning so we can see how well we did... */

	if( gettimeofday(&tv1,&tz1) < 0 ){
		perror("gettimeofday");
		NWARN("delay() unable to measure actual delay");
	}

	itv.it_value.tv_sec = msec/1000;
	itv.it_value.tv_usec = (msec%1000) * 1000;
	/* what should interval be set to??? */
	itv.it_interval.tv_sec = 0;
	itv.it_interval.tv_usec = 1000;

	delay_over = 0;
	signal(SIGALRM,end_delay);

	if( setitimer( ITIMER_REAL, &itv, &o_itv ) < 0 ){
		perror("setitimer");
		NWARN("unable to set interval timer for delay()");
		return;
	}
	while( ! delay_over )
		pause();	/* wait for signal */

	/* now see how much time actually elapsed */
	if( gettimeofday(&tv2,&tz1) < 0 ){
		perror("gettimeofday");
		NWARN("delay() unable to measure actual delay");
	}

	tv2.tv_sec -= tv1.tv_sec;
	if( tv2.tv_usec < tv1.tv_usec ){
		tv2.tv_sec -= 1;
		tv2.tv_usec += (1000000-tv1.tv_usec);
	} else {
		tv2.tv_usec -= tv1.tv_usec;
	}

	actual = tv2.tv_sec * 1000;
	actual += tv2.tv_usec/1000;

	if( verbose ){
		sprintf(DEFAULT_ERROR_STRING,
			"delay(%ld) consumed %ld msec",msec,actual);
		advise(DEFAULT_ERROR_STRING);
	}
}


