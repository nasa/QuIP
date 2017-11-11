#include "quipAlarm.h"
#include "quip_prot.h"
#include "query_stack.h"

// Code found on the web - thanks!

static void my_alarm(void);

@implementation quipAlarm

@synthesize timer;
@synthesize script;
@synthesize ticking;

- (id)initWithTimeout:(NSTimeInterval)timeout {
	self = [super init];

	// for testing
	if (self) {			
		script = NULL;

// We get the global queue - why?
// other choice might be the main queue, but I dimly
// remember that that can lock things up...
// But remember that UI interactions have to come
// from the main queue?

		dispatch_queue_t queue = dispatch_get_global_queue(
			DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);

		// create our timer source
		timer = dispatch_source_create(
			DISPATCH_SOURCE_TYPE_TIMER, 0, 0,
			queue);

		// set the time to fire (we're only going to fire once,
		// so just fill in the initial time).
		dispatch_source_set_timer(timer,
			dispatch_time(DISPATCH_TIME_NOW,
					timeout * NSEC_PER_SEC),
			DISPATCH_TIME_FOREVER, 0);

		// Hey, let's actually do something when the timer fires!
		dispatch_source_set_event_handler(timer,
#ifdef FOOBAR
		^{
			NSLog(@"WATCHDOG: task took longer than %f seconds",
				timeout);
			// ensure we never fire again
			dispatch_source_cancel(_timer);
		}
#endif /* FOOBAR */
				^{
					// ensure we don't fire again
					// just turn it off for now
					dispatch_suspend(timer);
					ticking = NO;
					my_alarm();
				}
			);
/*
}
*/
		// Don't start the timer in init...
		// now that our timer is all set to go, start it
		// dispatch_resume(timer);

		ticking = NO;
	}
	return self;
}

-(void) setDelay : (float) timeout
{
	dispatch_source_set_timer(timer,
		dispatch_time(DISPATCH_TIME_NOW, timeout * NSEC_PER_SEC),
		DISPATCH_TIME_FOREVER, 0);
}

- (void)dealloc {
	dispatch_source_cancel(timer);
	//dispatch_release(timer);
	
	
//	[super dealloc];
}

//- (void)invalidate {
//	_dispatch_source_cancel(_timer);
//}

@end

static quipAlarm *qap=NULL;

void set_alarm_script(QSP_ARG_DECL  const char *s)
{
	if( qap == NULL ){
		qap = [[quipAlarm alloc] initWithTimeout: (NSTimeInterval) 0.0];
	}

	if( qap.script != NULL )
		rls_str(qap.script);

	qap.script = savestr(s);
}

void set_alarm_time(QSP_ARG_DECL  float f)
{
	if( qap == NULL ){
		qap = [[quipAlarm alloc] initWithTimeout: (NSTimeInterval) f];
	} else {
		if( qap.ticking ){
			dispatch_suspend( qap.timer );
			qap.ticking = NO;
		}
		[qap setDelay:f];
	}

	// now that our timer is all set to go, start it
	if( qap.script != NULL ){
		// BUG we can't resume if it is already ticking!?
		dispatch_resume( qap.timer );
		qap.ticking = YES;	// BUG? race problem???
	} else {
		sprintf(ERROR_STRING,"set_alarm_time:  please specify alarm script before setting alarm time");
		WARN(ERROR_STRING);
	}
}

#ifdef FOOBAR
//static void run_in_main_queue(const char *script)
//{
//	push_text(DEFAULT_QSP_ARG  script);
//	dispatch_sync_f(dispatch_get_main_queue(),DEFAULT_QSP,
//		(void (*)(void *))check_quip);
//}
#endif /* FOOBAR */

static void my_alarm(void)
{
	if( qap.script == NULL ){
		NWARN("my_alarm:  null alarm script!?");
		return;
	}
	//run_in_main_queue(alarm_script);

	// Should we store the filename and line number as part of qap???
	// What if the alarm goes off while we are executing?
	// If we are waiting in an alert, we are OK because the chewing
	// flag is set???

	// Not doing the dispatch_sync_f here broke the refresh timing...

	// I think that we originally switched to chew text because if the
	// alarm occurs while we are doing something else, push_text
	// causes the execution to get messed up...  also there
	// is the issue of stopping execution after the pushed stuff
	// has been interpreted (exec_quip exhausts the input)

#ifdef FOOBAR
	push_text(DEFAULT_QSP_ARG  qap.script, "(alarm handler)");

	dispatch_sync_f(dispatch_get_main_queue(),DEFAULT_QSP,
		(void (*)(void *))exec_quip);
#endif // FOOBAR

	static Mouthful *mfp = NULL;
	if( mfp == NULL ){
		mfp = new_mouthful(qap.script,"(alarm handler)");
	} else {
		rls_str(mfp->text);
		rls_str(mfp->filename);
		mfp->text = savestr(qap.script);
		mfp->filename = savestr("(alarm handler)");
	}

	// BUG - memory leak - where should we release this mouthful?
	// Solution:  don't release, re-use.
	// We assume that there is only one alarm, so it shouldn't
	// interrupt itself.

	dispatch_sync_f(dispatch_get_main_queue(),mfp,
		(void (*)(void *))chew_mouthful);

	// We want to call chew_text, but we need to call it
	// from dispatch_sync_f - this was forgotten once and caused
	// a world of grief...

#ifdef FOOBAR
	// we use chew_text instead of check_quip and it seems to work fine...
	chew_text(DEFAULT_QSP_ARG  qap.script, "alarm handler");
#endif // FOOBAR

}

