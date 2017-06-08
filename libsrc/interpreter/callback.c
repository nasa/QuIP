#include "quip_config.h"
#include "quip_prot.h"
#include "history.h"
#include "query_prot.h"

#ifdef BUILD_FOR_IOS

static void relinquish_to_ios(SINGLE_QSP_ARG_DECL)
{
	// We halt the interpreter to let ios take over
	// and do whatever, using the same mechanism we
	// use for alerts.  In the case of alerts, however,
	// the users dismissal of the alert serves to send
	// control back to the interpreter.  Here, on the other
	// hand, we don't know if anything at all is going to happen
	// (although it probably will, given that this will
	// be called when the script executes os/events).
	//
	// We set a timer based on the display refresh to wake up
	// the interpreter...

//#ifdef CAUTIOUS
//	if( IS_HALTING(THIS_QSP) ){
//		// The fact that this warning statement was commented out
//		// suggests that the code traverses this path routinely???
//
//		//WARN("CAUTIOUS:  relinquish_to_ios:  already halting!?");
//		return;
//	}
//#endif // CAUTIOUS

	// If this assertion fails, that means that we've
	// already called this?
	assert( ! IS_HALTING( THIS_QSP ) );

	SET_QS_FLAG_BITS(THIS_QSP,QS_HALTING);

	// Now the interpreter will stop reading input.
	// But we now need to schedule a wakeup so that it can pick up
	// again after the OS has done it's stuff...

	sync_with_ios();
}

#endif // BUILD_FOR_IOS

void call_funcs_from_list(QSP_ARG_DECL  List *lp )
{
	Node *np;
	void (*func)(SINGLE_QSP_ARG_DECL);

	np=QLIST_HEAD(lp);
	assert( np != NULL );

	while( np != NULL ){
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

	if( QS_EVENT_LIST(THIS_QSP) == NULL )
		return;

	call_funcs_from_list(QSP_ARG  QS_EVENT_LIST(THIS_QSP) );
}

