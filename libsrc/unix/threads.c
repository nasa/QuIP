/* Support for independent interpreter threads */

#include "quip_config.h"

/*
 * Query_Stacks were introduced to make the interpreter thread-safe
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

#include "quip_prot.h"
#include "query_stack.h"

#ifdef THREAD_SAFE_QUERY

#ifdef HAVE_PTHREADS

#ifdef HAVE_PTHREAD_H
#include <pthread.h>
#endif

static void *thread_exec(void *argp)
{
	Query_Stack *qsp;

	qsp = argp;

	//push_menu(qsp,FIRST_MENU(qsp));
	//push_quip_menu(qsp);
	push_first_menu(qsp);	// may not be the quip menu!

	init_aux_menus(qsp);

	// Threads can't call exit() without killing the whole
	// process, so they set a flag instead.
	while( ! IS_HALTING(qsp) ){
		qs_do_cmd(qsp);
	}
	// return is like pthread_exit...
	return(NULL);
}

static COMMAND_FUNC( do_new_thread )
{
	const char *s, *c;
	Query_Stack *new_qsp;
	pthread_attr_t attr1;

	s=NAMEOF("name for thread");
	c=NAMEOF("script to execute");

	new_qsp = new_qstk(QSP_ARG  s);
	if( new_qsp == NULL ) return;

	// change the flags from the default values
	CLEAR_QS_FLAG_BITS(new_qsp, QS_INTERACTIVE_TTYS|QS_FORMAT_PROMPT|QS_COMPLETING );

	SET_QS_PARENT_SERIAL(new_qsp,_QS_SERIAL(THIS_QSP));

	// The new thread should inherit context stacks from the parent thread,
	// but we don't want to bother to do all that now - ?

//if( verbose ){
//sprintf(ERROR_STRING,"do_new_thread %s:  qs_flags = 0x%x",
//QS_NAME(new_qsp),new_qsp->qs_flags);
//advise(ERROR_STRING);
//}
	// We have to copy the text, because the buffer returned
	// by nameof can get recycled, and pushtext doesn't make
	// a copy.

	// This is a potential memory leak?
	push_text(new_qsp, savestr(c), "(new thread)" );
	SET_QRY_FILENAME( CURR_QRY(new_qsp), "thread text" );
//qdump(new_qsp);

	// We want to create a new variable context for this thread...
	// and if we have cuda, a cuda execution context...

	pthread_attr_init(&attr1);	/* initialize to default values */
	pthread_attr_setinheritsched(&attr1,PTHREAD_INHERIT_SCHED);

	pthread_create(&new_qsp->qs_thr,&attr1,thread_exec,new_qsp);
}

static COMMAND_FUNC( do_list_threads )
{
	prt_msg("All threads:");
	list_query_stacks(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_tell_thread )
{
	sprintf(MSG_STR,"Current thread is %s",QS_NAME(THIS_QSP));
	prt_msg(MSG_STR);
}

static COMMAND_FUNC( do_wait_thread )
{
	int status;
	void **val_ptr=NULL;
	Query_Stack *thread_qsp;

	thread_qsp = pick_query_stack(QSP_ARG  "thread name");
	if( thread_qsp == NULL ) return;

	if( _QS_SERIAL(thread_qsp) == 0 ){
		WARN("do_wait_thread:  can't wait for main thread!?");
		return;
	}

	status = pthread_join( thread_qsp->qs_thr, val_ptr );
	if( status != 0 )
		WARN("pthread_join returned an error status!?");
}

#define ADD_CMD(s,f,h)		ADD_COMMAND(threads_menu,s,f,h)

MENU_BEGIN(threads)
ADD_CMD( new_thread,	do_new_thread,		create a new thread )
ADD_CMD( list,		do_list_threads,	list all active threads )
ADD_CMD( tell,		do_tell_thread,		report name of current thread )
ADD_CMD( wait,		do_wait_thread,		wait for thread to exit )
MENU_END(threads)

COMMAND_FUNC( do_thread_menu )
{
	PUSH_MENU( threads );
}


#else /* ! HAVE_PTHREADS */

COMMAND_FUNC( do_thread_menu )
{
	error1("Sorry, no support for threads in this build (libpthreads is missing).");
}

#endif /* ! HAVE_PTHREADS */

#else /* ! THREAD_SAFE_QUERY */

COMMAND_FUNC( do_thread_menu )
{
	error1("Sorry, no support for threads in this build (configure --enable-thread-safe-query).");
}

#endif /* ! THREAD_SAFE_QUERY */
