
#include "quip_config.h"

/*
 * Possible #define's
 *
 * MY_INTR	allow user to specify interrupt handler (UNIX only)
 * DYNAMIC_LOAD	allow loading of code files
 * HELPFUL	include help submenu
 * SUBMENU	allow addition of submenus
 */

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>		/* gettimeofday */
#endif


#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* system() */
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_TIME_H
#include <time.h>		/* time(), ctime() */
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* getuid(), usleep() */
#endif

#include "quip_prot.h"
#include "sigpush.h"
#include "unix_prot.h"
#include "dobj_prot.h"
#include "stack.h"	// BUG
#include "query_stack.h"	// BUG

#ifdef BUILD_FOR_IOS
#include "ios_prot.h"
#endif /* BUILD_FOR_IOS */

#ifdef MY_INTR

#include <signal.h>
static const char *intr_str="";

/* when we get an interrupt, do we associate it with a particular query_stream? */

static void my_onintr(int asig /* what is this arg??? */)
{
	if( *intr_str == 0 ) _nice_exit(DEFAULT_QSP_ARG  0);

	// like top_menu - but pops everything
	// Maybe we should then flush any pending input too?
	while( STACK_DEPTH(QS_MENU_STACK(DEFAULT_QSP)) > MIN_CMD_DEPTH )
		_pop_menu(SGL_DEFAULT_QSP_ARG);

	_push_text(DEFAULT_QSP_ARG  intr_str, "intr_handler" );

	//curr_qsp->qs_flags &= ~QS_HAD_INTR;	/* clear flag */
	CLEAR_QS_FLAG_BITS(DEFAULT_QSP,QS_HAD_INTR);
}

static COMMAND_FUNC( set_onintr )
{
	const char *s;

	s=NAMEOF("text to interpret upon interrupt");
	if( *intr_str == 0 ) {
		sigpush((int)SIGINT,my_onintr);
		/*warn("no previous interrupt action"); */
	} else {
		if( verbose ){
			sprintf(ERROR_STRING,
		"former interrupt action string was \"%s\"",
				intr_str);
			advise(ERROR_STRING);
		}
		rls_str(intr_str);
	}
	intr_str = savestr(s);
}
#endif /* MY_INTR */

// moved to builtin menu
//static COMMAND_FUNC( my_quick_exit ){ exit(0); }

#ifdef MALLOC_DEBUG
static COMMAND_FUNC( do_m_verify )
{
	malloc_verify();
}
#endif /* MALLOC_DEBUG */

#ifdef USE_GETBUF
extern COMMAND_FUNC( heap_report );
#endif

static COMMAND_FUNC( do_report_node_data ){ report_node_data(SINGLE_QSP_ARG); }

static COMMAND_FUNC( do_showmaps )
{
#ifdef USE_GETBUF
	showmaps();
#else /* ! USE_GETBUF */
	warn("do_showmaps:  program not configured with USE_GETBUF, nothing to show.");
#endif /* ! USE_GETBUF */
}


/* we use a function vector to avoid a dependency on libxsupp */

static void (*discard_event_func_vec)(SINGLE_QSP_ARG_DECL)=NULL;

void set_discard_func(void (*func)(SINGLE_QSP_ARG_DECL) )
{
	discard_event_func_vec=func;
}

static COMMAND_FUNC( do_flush_events )
{ 
	if( discard_event_func_vec == NULL ){
		warn("do_flush_events:  no discard function specified!?");
	} else {
		(*discard_event_func_vec)(SINGLE_QSP_ARG);
	}
}

#ifdef HAVE_HISTORY
static COMMAND_FUNC( set_history )
{
	if( ASKIF("retain input history") )
		history_flag=1;
	else history_flag=0;
}
#endif /* HAVE_HISTORY */

static Data_Obj *ckpt_tbl_dp=NULL;
static Data_Obj *ckpt_msg_dp=NULL;
static int n_ckpts=0;
#define MAX_CKPTS	512
#define MAX_MSG_LEN	60
#define CKPT_TBL_NAME	"__ckpt_tbl"
#define CKPT_MSG_NAME	"__ckpt_msg"

static COMMAND_FUNC( do_ckpt )
{
#ifdef HAVE_GETTIMEOFDAY
	const char *s;
	struct timeval tv;
	char *ptr;
	time_t *t_ptr;

	s = NAMEOF("tag for this checkpoint");

	if( ckpt_tbl_dp == NULL ){
		int siz,siz2;
		prec_t prec;

		siz=sizeof(time_t);
		if( siz == 8 ) prec=PREC_ULI;
		else if( siz == 4 ) prec=PREC_UDI;
#ifdef CAUTIOUS
		else {
			error1("CAUTIOUS:  do_ckpt:  unhandled size of time_t");
			prec=PREC_ULI;	// silence compiler - doesn't know error1 never returns
		}
		
		siz2 = sizeof(suseconds_t);
		if( siz2 > siz ) {
			sprintf(ERROR_STRING,
"CAUTIOUS:  do_ckpt:  size of suseconds_t (%d) is greater than that of time_t (%d).",siz2,siz);
			error1(ERROR_STRING);
		}

#endif /* CAUTIOUS */

#ifdef HAVE_CUDA
		push_data_area(ram_area_p);
#endif /* HAVE_CUDA */

		ckpt_tbl_dp = mk_vec(QSP_ARG  CKPT_TBL_NAME, MAX_CKPTS, 2, PREC_FOR_CODE(prec) );
		if( ckpt_tbl_dp == NULL ) error1("Error creating checkpoint table");

		ckpt_msg_dp = mk_img(CKPT_MSG_NAME, MAX_CKPTS, MAX_MSG_LEN, 1,
			PREC_FOR_CODE(PREC_STR) );
		if( ckpt_msg_dp == NULL ) error1("Error creating checkpoint messages");

#ifdef HAVE_CUDA
		pop_data_area();
#endif /* HAVE_CUDA */

	}

	if( strlen(s) >= MAX_MSG_LEN ){
		sprintf(ERROR_STRING,"Sorry, checkpoint tag has too many characters (%ld, max %d), truncating...",
			(long)strlen(s),MAX_MSG_LEN-1);
		warn(ERROR_STRING);
	}
	if( n_ckpts >= MAX_CKPTS ){
		sprintf(ERROR_STRING,"Sorry, %d checkpoints already placed, can't place '%s'.",n_ckpts,s);
		warn(ERROR_STRING);
		return;
	}

	if( gettimeofday(&tv,NULL) < 0 ){
		perror("gettimeofday");
		warn("error reading system time");
		return;
	}

	t_ptr = (time_t *)OBJ_DATA_PTR(ckpt_tbl_dp);
	t_ptr += n_ckpts * 2 ;
	* t_ptr = tv.tv_sec;
	t_ptr ++;
	* t_ptr = tv.tv_usec;

	ptr = OBJ_DATA_PTR(ckpt_msg_dp);
	ptr += n_ckpts * MAX_MSG_LEN;

	strncpy(ptr,s,MAX_MSG_LEN-1);
	*(ptr+MAX_MSG_LEN-1) = 0;	// insure null-terminated

	n_ckpts++;
#else // ! HAVE_GETTIMEOFDAY
	warn("Sorry, no checkpointing available in this build!?");
#endif // ! HAVE_GETTIMEOFDAY
}

static COMMAND_FUNC( do_tell_ckpts )
{
	int i;
	char *mptr;
	time_t *tptr;
	long secs, usecs;
	long secs0, usecs0;
	double delta_ms, cum_ms;

	if( n_ckpts == 0 ){
		warn("do_tell_ckpts:  no checkpoints set.");
		return;
	}

	tptr = (time_t *) OBJ_DATA_PTR(ckpt_tbl_dp);
	mptr = (char *) OBJ_DATA_PTR(ckpt_msg_dp);

	secs0 = (long) *tptr;
	usecs0 = (long) *(tptr+1);

	secs  = (long) *(tptr+  2*(n_ckpts-1));
	usecs = (long) *(tptr+1+2*(n_ckpts-1));
	delta_ms = 1000.0*(secs-secs0) + (usecs-usecs0)/1000.0;
	sprintf(msg_str,"Total HOST time:  %12.3f",delta_ms);
	prt_msg(msg_str);

	cum_ms = 0.0;
	for(i=0;i<n_ckpts;i++){
		secs = (long) *tptr;
		usecs = (long) *(tptr+1);

		delta_ms = 1000.0*(secs-secs0) + (usecs-usecs0)/1000.0;
		cum_ms += delta_ms;

		sprintf(msg_str,"HOST %3d  %12.3f  %12.3f  %s",i+1,delta_ms,
			cum_ms,mptr);
		prt_msg(msg_str);

		tptr += 2;
		mptr += MAX_MSG_LEN;

		secs0 = secs;
		usecs0 = usecs;
	}
	n_ckpts=0;	// clear for next time
}


static COMMAND_FUNC( get_time_of_day )
{
	const char *s1,*s2;
#ifdef HAVE_GETTIMEOFDAY
	struct timeval tv;
#endif // HAVE_GETTIMEOFDAY

	s1=NAMEOF("variable name for seconds");
	s2=NAMEOF("variable name for microseconds");

#ifdef HAVE_GETTIMEOFDAY
	if( gettimeofday(&tv,NULL) < 0 ){
		perror("gettimeofday");
		warn("error reading system time");
		return;
	}

	sprintf(msg_str,"%ld",tv.tv_sec);
	assign_var(s1,msg_str);
	// on mac, tv_usec has a wierd type?
	sprintf(msg_str,"%ld",(long)tv.tv_usec);
	assign_var(s2,msg_str);
#else // ! HAVE_GETTIMEOFDAY
	assign_var(s1,"0");
	assign_var(s2,"0");
	warn("Sorry, no gettimeofday!?");
#endif // ! HAVE_GETTIMEOFDAY
}

static COMMAND_FUNC( get_time )
{
	const char *s;
	time_t t;

	s=NAMEOF("variable name");
	t=time(NULL);
	if( t == (time_t)(-1) ){
		tell_sys_error("time");
		t = (time_t) 0;
	}
	sprintf(msg_str,"%ld",t);
	assign_var(s,msg_str);
}

static COMMAND_FUNC( do_system )				/** execute a shell command */
{
	const char *s;
	int stat;
	int euid;
	int ruid;
	
	s=NAMEOF("command");
	
#ifdef HAVE_GETUID
	euid = geteuid();
	ruid = getuid();
	
	if( euid == 0 && ruid != 0 ){
		warn("Sorry, shell commands not allowed for set-uid root programs");
		return;
	}
#endif // HAVE_GETUID


#ifndef BUILD_FOR_IOS
	// On IOS, there is no stdout, so we don't see any output!?
	stat=system(s);
	
	if( stat == -1 )
		tell_sys_error("system");
	else if( verbose ){
		sprintf(ERROR_STRING,"Exit status %d",stat);
		advise(ERROR_STRING);
	}
#else // ! BUILD_FOR_IOS
	warn("Sorry, system command is temporarily unavailable for iOS!?");
	stat=(-1);
#endif // ! BUILD_FOR_IOS

	sprintf(msg_str,"%d",stat);
	assign_reserved_var("exit_status",msg_str);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(os_menu,s,f,h)

MENU_BEGIN(os)

ADD_CMD( system,	do_system,	  execute a shell command )

#ifdef BUILD_FOR_IOS
ADD_CMD( accelerometer,	do_accel_menu,	  accelerometer submenu )
ADD_CMD( camera,	do_cam_menu,	  camera submenu )
#endif /* BUILD_FOR_IOS */

#ifdef HAVE_ADJTIMEX
ADD_CMD( adjtimex,	do_timex_menu,	linux clock adjustment submenu )
#endif /* HAVE_ADJTIMEX */

ADD_CMD( get_time,	get_time,	set var to current time )
ADD_CMD( usecs,		get_time_of_day,	set vars to current seconds and usecs )
ADD_CMD( checkpoint,	do_ckpt,	record time to checkpoint table )
ADD_CMD( report,	do_tell_ckpts,	report checkpoints and reset )
ADD_CMD( serial,	do_ser_menu,	send/receive serial data )
ADD_CMD( mouse,		do_mouse_menu,	talk to mouse on serial port )

#ifdef HAVE_POPEN
ADD_CMD( pipes,		do_pipe_menu,	subprocess pipe submenu )
#endif /* HAVE_POPEN */

// BUG these two commands are not really OS!?
#ifdef HAVE_HISTORY
ADD_CMD( history,	set_history,	enable/disable history )
#ifdef TTY_CTL
ADD_CMD( complete,	set_completion,	set/clr command completion )
#endif /* TTY_CTL */
#endif /* HAVE_HISTORY */

ADD_CMD( events,	call_event_funcs,	call event handler(s) )
ADD_CMD( flush_events,	do_flush_events,	discard pending events )
ADD_CMD( showmem,	do_showmaps,		show free heap memory )
ADD_CMD( nodes,		do_report_node_data,	show node statistics )

#ifdef MY_INTR
ADD_CMD( interrupt_action,	set_onintr,	set interrupt command string )
#endif /* MY_INTR */

ADD_CMD( prompts,	tog_pmpt,	toggle prompt printing flag )
ADD_CMD( clobber,	togclobber,	toggle file clobber caution )

#ifdef MALLOC_DEBUG
ADD_CMD( malloc_verify,	do_m_verify,	heap consistency check )
#endif /* MALLOC_DEBUG */

#ifdef DYNAMIC_LOAD
ADD_CMD( load,		do_load,	load object module )
ADD_CMD( bind,		do_bind,	bind new menu item to function )
#endif /* DYNAMIC_LOAD */

#ifdef HAVE_FORK
ADD_CMD( fork,		do_fork,	fork interpreter )
ADD_CMD( wait,		do_wait_child,	wait for child to finish )
#endif /* HAVE_FORK */

#ifdef HAVE_PTHREADS
ADD_CMD( threads,	do_thread_menu,	threads submenu )
#endif /* HAVE_PTHREADS */

ADD_CMD( scheduler,	do_sched_menu,	scheduler submenu )
// move to builtin menu...
//ADD_CMD( fast_exit,	my_quick_exit,	exit without resetting tty )
MENU_END(os)



COMMAND_FUNC( do_unix_menu )
{
	CHECK_AND_PUSH_MENU(os);
}

