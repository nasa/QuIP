
#include "quip_config.h"
#include "query_prot.h"
#include "data_obj.h"
#include "veclib_api.h"
#include "vt_api.h"
#include "camera_api.h"
#include "view_cmds.h"
#include "server.h"	// do_http_menu
#include "polh_menu.h"	// do_polh
#ifdef BUILD_FOR_IOS
#include "camera_api.h"
#endif // BUILD_FOR_IOS

#ifdef HAVE_CUDA
#include "cuda_api.h"
#endif /* HAVE_CUDA */

#ifdef BUILD_FOR_MACOS
//#define MINIMAL_BUILD
#endif // BUILD_FOR_MACOS

static Menu *first_menu=NULL;

void push_first_menu(Query_Stack *qsp)
{
	assert( first_menu != NULL );
	PUSH_MENU_PTR(first_menu);
}

void start_quip_with_menu(int argc, char **argv, Menu *initial_menu_p )
{
	Query_Stack *qsp;

	set_progname(argv[0]);
	first_menu = initial_menu_p;

	//debug |= CTX_DEBUG_MASK;
	//debug |= GETBUF_DEBUG_MASK;

	qsp=init_first_query_stack();	// reads stdin?

	init_builtins();
	init_variables(SINGLE_QSP_ARG);	// specify dynamic variables
	declare_functions(SINGLE_QSP_ARG);

	//PUSH_MENU(quip);
	PUSH_MENU_PTR(initial_menu_p);

	set_args(QSP_ARG  argc,argv);
	rcfile(qsp,argv[0]);

	// If we have commands to create a widget in the startup file,
	// we get an error, so don't call exec_quip until after the appDelegate
	// has started...
	
} // end start_quip_with_menu

// start_quip executes on the main thread...

// This seems redundant with exec_pending_cmds ???

static void exec_qs_cmds( void *_qsp )
{
	Query_Stack *qsp=(Query_Stack *)_qsp;

	while( lookahead_til(QSP_ARG  0) ){
		while( QS_HAS_SOMETHING(qsp) && ! IS_HALTING(qsp) ){
			QS_DO_CMD(qsp);
		}
	}
}

void exec_this_level(SINGLE_QSP_ARG_DECL)
{
	exec_at_level(QSP_ARG  QLEVEL);
}

// BUG if we push a macro which contains only comments, then we could
// end up presenting an interactive prompt instead of popping back...

void exec_at_level(QSP_ARG_DECL  int level)
{
//#ifdef CAUTIOUS
//	if( level < 0 ){
//		fprintf(stderr,
//	"CAUTIOUS:  exec_at_level:  bad level %d!?\n",level);
//		abort();
//	}
//#endif // CAUTIOUS

	assert( level >= 0 );

	// We thought a lookahead here might help, but it didn't, probably
	// because lookahead does not skip comments?

	//lookahead(SINGLE_QSP_ARG);	// in case an empty macro was pushed?
	while( QLEVEL >= level ){
		qs_do_cmd(THIS_QSP);
		if( IS_HALTING(THIS_QSP) ){
			return;
		}

		/* The command may have disabled lookahead;
		 * We need to lookahead here to make sure
		 * the end of the text is properly detected.
		 */

		lookahead(SINGLE_QSP_ARG);
	}

	// BUG?  what happens if we halt execution when an alert is delivered?
}

// exec_quip executes commands as long as there is something to do.
// We might want to execute this in a queue other than the main queue
// in order to catch events while we are executing?
// This is important for display updates, but could cause problems for
// touch events which call exec_quip again?

void exec_quip(SINGLE_QSP_ARG_DECL)
{
#ifdef USE_QS_QUEUE

	// This is iOS only!

	//dispatch_async_f(QS_QUEUE(THIS_QSP),qsp,exec_qs_cmds);

	if( QS_QUEUE(THIS_QSP) == dispatch_get_current_queue() ){
		exec_qs_cmds(THIS_QSP);
	} else {
		dispatch_async_f(QS_QUEUE(THIS_QSP),THIS_QSP,exec_qs_cmds);
	}

#else /* ! USE_QS_QUEUE */

	exec_qs_cmds(THIS_QSP);

#endif /* ! USE_QS_QUEUE */
}

