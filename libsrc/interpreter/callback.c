
#include "quip_config.h"

char VersionId_interpreter_callback[] = QUIP_VERSION_STRING;

#include "query.h"
#include "callback.h"	/* prototypes are here */
#include "callback_api.h"	/* prototypes are here */
#include "submenus.h"
#include "void.h"

#define MAX_EVENT_FUNCS	4
static int n_event_funcs=0;

static void (*	event_func[MAX_EVENT_FUNCS])(SINGLE_QSP_ARG_DECL);

void add_event_func( void (*func)(SINGLE_QSP_ARG_DECL) )	/** set event processing function */
{
	if( n_event_funcs >= MAX_EVENT_FUNCS ){
		NWARN("too many event functions");
		return;
	}
	event_func[n_event_funcs++] = func;
}

COMMAND_FUNC( call_event_funcs )
{
	int i;

	for(i=0;i<n_event_funcs;i++)
		(*event_func[i])(SINGLE_QSP_ARG);

}

int rem_event_func( void (*func)(SINGLE_QSP_ARG_DECL) )
{
	int i;
	int the_one=(-1);

	for(i=0;i<n_event_funcs;i++)
		if( event_func[i] == func ) the_one = i;
	
	if( the_one < 0 ) return(-1);

	/* shift table up */

	for(i=the_one+1;i<n_event_funcs;i++)
		event_func[i-1] = event_func[i];

	n_event_funcs--;
	return(0);
}

List *new_callback_list()
{
	List *lp;

	lp = new_list();
	return(lp);
}

void add_callback_func(List *lp,void (*func)(VOID))
{
	Node *np;

	/* BUG? without the cast, gcc reports that assigning a function pointer
	 * to void * is a violation of ANSI - we put the cast in to
	 * eliminate the warning, but will this cause a problem on
	 * some systems?  Why the ANSI prohibition??
	 */

	np = mk_node((void *) func);
	addTail(lp,np);
}

void call_callback_list(List *lp)
{
	Node *np;
	void (*func)(VOID);

	np=lp->l_head;
	while( np != NO_NODE ){
		func = (void (*)(VOID)) np->n_data;
		(*func)();
		np = np->n_next;
	}
}

