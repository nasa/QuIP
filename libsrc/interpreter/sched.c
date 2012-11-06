#include "quip_config.h"

char VersionId_interpreter_sched[] = QUIP_VERSION_STRING;

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#ifdef HAVE_SYS_RESOURCE_H
#include <sys/resource.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_SCHED_H
#include <sched.h>
#endif

#include "rt_sched.h"
#include "query.h"
#include "submenus.h"

#ifdef ALLOW_RT_SCHED
int try_rt_sched=1;
int rt_is_on=0;

static int curr_policy=SCHED_OTHER;
#endif

void rt_sched(QSP_ARG_DECL  int flag)
{
#ifdef HAVE_SCHED_SETSCHEDULER
#ifdef ALLOW_RT_SCHED
	pid_t pid;
	struct sched_param p;

	pid = getpid();

	if( flag ){		/* enable real-time scheduling */

		p.sched_priority = 1;
		if( sched_setscheduler(pid,SCHED_FIFO,&p) < 0 ){
			perror("sched_setscheduler");
			WARN("Unable to set real-time priority (run as root!)");

			rt_is_on = 0;
		} else {
			rt_is_on = 1;
			curr_policy=SCHED_FIFO;
		}
	} else {		/* disable real-time scheduling */
		if( rt_is_on ){
			p.sched_priority = 0;
			if( sched_setscheduler(pid,SCHED_OTHER,&p) < 0 ){
				perror("sched_setscheduler");
			
				WARN("Unable to reset real-time priority???");
			}
			rt_is_on = 0;
			curr_policy=SCHED_OTHER;
		}
	}
#endif /* ALLOW_RT_SCHED */
#else
	NWARN("rt_sched:  no scheduler support on this system.");
#endif
}

static COMMAND_FUNC( do_rt_sched )
{
	int yn;

	yn = ASKIF("enable real-time scheduling");
	rt_sched(QSP_ARG  yn);
}

static COMMAND_FUNC( do_get_pri )
{
#ifdef HAVE_SCHED_GETPARAM
#ifdef ALLOW_RT_SCHED
	int min,max;
	struct sched_param p;

	if( sched_getparam(getpid(),&p) < 0 ){
		perror("sched_getparam");
		WARN("unable to get scheduler params");
	}

	min = sched_get_priority_min(curr_policy);
	max = sched_get_priority_max(curr_policy);
	sprintf(msg_str,"Priority is %d (range %d-%d" ,p.sched_priority,min,max);
	prt_msg(msg_str);
#endif /* ALLOW_RT_SCHED */
#else
	WARN("do_get_pri:  no scheduler support on this system.");
#endif
}

static COMMAND_FUNC( do_set_pri )
{
#ifdef HAVE_SCHED_SETPARAM
#ifdef ALLOW_RT_SCHED
	int pri,min,max;
	struct sched_param p;

	min = sched_get_priority_min(curr_policy);
	max = sched_get_priority_max(curr_policy);
	sprintf(msg_str,"priority (%d-%d)",min,max);
	pri = HOW_MANY(msg_str);
	p.sched_priority = pri;
	if( sched_setparam(getpid(), &p) < 0 ){
		perror("sched_setparam");
		WARN("unable to set priority");
	}
	/*sched_set_pri(pri);*/
#endif /* ALLOW_RT_SCHED */
#else
	WARN("do_set_pri:  no scheduler support on this system.");
#endif
}


static Command sched_ctbl[]={
{ "rt_sched",	do_rt_sched,	"enable/disable real-time scheduling"	},
{ "get_pri",	do_get_pri,	"report current priority"		},
{ "set_pri",	do_set_pri,	"set priority"				},
{ "quit",	popcmd,		"exit submenu"				},
{ NULL_COMMAND								}
};


COMMAND_FUNC( sched_menu )
{
	PUSHCMD(sched_ctbl,"sched");
}

