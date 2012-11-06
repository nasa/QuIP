/*
 *  Asynchronous tasks using fork(2)
 */

#include "quip_config.h"

char VersionId_interpreter_fork[] = QUIP_VERSION_STRING;

/* If HAVE_FORK is defined, we allow an async subprocess to be created.
 * This won't crash the program even when THREAD_SAFE_QUERY is not defined,
 * because the child process has its own copy of the memory space...
 *
 * Query_Streams were introduced to make the interpreter thread-safe
 * with a single memory space.
 *
 * Note that thread-safe is not required for a subprocess created
 * with fork(), because the forked subprocess has its own copy of the memory
 * space!?  See threads.c for true threads.
 *
 */

#define NO_FORK_MESSAGE							\
									\
	warn("Sorry, no support for asynchronous threads in this build."); \
	advise("Reonfigure program with -XXX option.");

#include "query.h"

#ifdef HAVE_FORK

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>		/* needed on cray */
#endif

#ifdef HAVE_SYS_WAIT_H
#include <sys/wait.h>
#endif


/* BUG this implemenation only allows one child process... */
static int child_pid=(-1);

#endif /* HAVE_FORK */

#include "submenus.h"

COMMAND_FUNC( do_fork )
{
	const char *s;

#ifdef HAVE_FORK

	s=NAMEOF("string to interpret in forked task");

	if( child_pid != -1 ){
		WARN("Sorry, can't have more than 1 child");
		return;
	}
	if( (child_pid=fork()) == 0 ){
		/* child process */
		/* Don't need a new query stream when we are forking, only for a shared-memory thread! */
		/* qsp = new_query_stream(); */
		push_input_file(QSP_ARG  "(do_fork)");
		PUSHTEXT("fast_exit");
		push_input_file(QSP_ARG  "(do_fork)");
		PUSHTEXT(s);

		while(1) do_cmd(SINGLE_QSP_ARG);

	} else if( child_pid == -1 ){
		tell_sys_error("fork");
	}
#else	/* ! HAVE_FORK */
	s=NAMEOF("dummy word");
	NO_FORK_MESSAGE
#endif	/* ! HAVE_FORK */
}

COMMAND_FUNC( wait_child )
{
#ifdef HAVE_FORK
	if( child_pid == -1 ){
		WARN("No child to wait for");
		return;
	}
	if( wait((void *)0) == -1 )
		tell_sys_error("wait");
	child_pid = (-1);
#else	/* ! HAVE_FORK */
	NO_FORK_MESSAGE
#endif	/* ! HAVE_FORK */
}

