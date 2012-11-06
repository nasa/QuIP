/* Support for independent interpreter threads */

#include "quip_config.h"

char VersionId_interpreter_threads[] = QUIP_VERSION_STRING;

/*
 * Query_Streams were introduced to make the interpreter thread-safe
 * with a single memory space.
 *
 * Note that thread-safe is not required for a subprocess created
 * with fork(), because the forked subprocess has its own copy of the memory
 * space!?  However, there are many applications where we do want multiple
 * threads with shared memory space.  One example is an eye tracking
 * application where we might have one thread for each camera, another
 * thread handling stimuli...  We would like to have the individual threads
 * have their own context for script variables, so that we can write using
 * a variable like $i_cam, and each thread could have a different value...
 *
 */

#include "query.h"
#include "submenus.h"

#ifdef THREAD_SAFE_QUERY

#ifdef HAVE_PTHREADS

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

static Command *top_ctbl=NULL;

void set_top_ctbl(Command *tc)
{
	top_ctbl = tc;
}

void *thread_exec(void *argp)
{
	Query_Stream *qsp;

	qsp = argp;

	//pushcmd(qsp,quip_ctbl,"quip");

#ifdef CAUTIOUS
	if( top_ctbl == NULL ) error1(DEFAULT_QSP_ARG  "CAUTIOUS:  thread_exec:  top_ctbl is not set!?");
#endif /* CAUTIOUS */

	pushcmd(qsp,top_ctbl,"quip");

	// Threads can't call exit() without killing the whole
	// process, so they set a flag instead.
	while( ! IS_HALTING(qsp) ) do_cmd(qsp);
advise("thread_exec DONE!!!");

	return(NULL);
}

static COMMAND_FUNC( do_new_thread )
{
	const char *s, *c;
	Query_Stream *new_qsp;
	pthread_attr_t attr1;

	s=NAMEOF("name for thread");
	c=NAMEOF("script to execute");

	new_qsp = new_query_stream(QSP_ARG  s);
	if( new_qsp == NULL ) return;

	// change the flags from the default values
	new_qsp->qs_flags &= ~(QS_INTERACTIVE_TTYS|QS_FORMAT_PROMPT|QS_COMPLETING);

//if( verbose ){
//sprintf(ERROR_STRING,"do_new_thread %s:  qs_flags = 0x%x",
//new_qsp->qs_name,new_qsp->qs_flags);
//advise(ERROR_STRING);
//}
	// Does the query stream have an input already???
	push_input_file(new_qsp,  "thread text");
	// We have to copy the text, because the buffer returned
	// by nameof can get recycled, and pushtext doesn't make
	// a copy.

	// This is a potential memory leak?
	pushtext(new_qsp, savestr(c) );
//qdump(new_qsp);

	// We want to create a new variable context for this thread...
	// and if we have cuda, a cuda execution context...

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);

	pthread_create(&new_qsp->qs_thr,&attr1,thread_exec,new_qsp);
}

static COMMAND_FUNC( do_list_threads )
{
	list_qstreams(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_tell_thread )
{
	sprintf(MSG_STR,"Current thread is %s",THIS_QSP->qs_name);
	prt_msg(MSG_STR);
}

static Command thread_ctbl[]={
{ "new_thread",	do_new_thread,		"create a new thread"		},
{ "list",	do_list_threads,	"list all active threads"	},
{ "tell",	do_tell_thread,		"report name of current thread"	},
{ "quit",	popcmd,			"exit submenu"			},
{ NULL_COMMAND								}
};

COMMAND_FUNC( thread_menu )
{
	PUSHCMD( thread_ctbl, "threads" );
}


#else /* ! HAVE_PTHREADS */

COMMAND_FUNC( thread_menu )
{
	error1("Sorry, no support for threads in this build (libpthreads is missing).");
}

#endif /* ! HAVE_PTHREADS */

#else /* ! THREAD_SAFE_QUERY */

COMMAND_FUNC( thread_menu )
{
	error1("Sorry, no support for threads in this build (configure --enable-thread-safe-query).");
}

#endif /* ! THREAD_SAFE_QUERY */
