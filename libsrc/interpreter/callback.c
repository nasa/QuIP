#include "quip_config.h"
#include "quip_prot.h"
#include "history.h"
#include "query_prot.h"

void call_funcs_from_list(QSP_ARG_DECL  List *lp )
{
	Node *np;
	void (*func)(SINGLE_QSP_ARG_DECL);

	np=QLIST_HEAD(lp);

//#ifdef CAUTIOUS
//	if( np == NO_NODE ){
//		WARN("CAUTIOUS:  call_funcs_from_list:  list is empty!?");
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( np != NO_NODE );

	while( np != NO_NODE ){
		func = (void (*)(SINGLE_QSP_ARG_DECL)) NODE_DATA(np);
		(*func)(SINGLE_QSP_ARG);
		np = NODE_NEXT(np);
	}
}

void call_event_funcs(SINGLE_QSP_ARG_DECL)
{
#ifdef BUILD_FOR_IOS
	static int relinquishing=0;

	if( ! relinquishing ){
		add_event_func(QSP_ARG  relinquish_to_ios);
		relinquishing=1;
	}
#endif // BUILD_FOR_IOS

	if( QS_EVENT_LIST(THIS_QSP) == NO_LIST )
		return;

	call_funcs_from_list(QSP_ARG  QS_EVENT_LIST(THIS_QSP) );
}

