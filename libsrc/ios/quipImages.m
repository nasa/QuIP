//
//  quipImages.m
//
#include <QuartzCore/QuartzCore.h>

// for mach_absolute_time()
//#include <CoreServices/CoreServices.h>
#include <mach/mach.h>
#include <mach/mach_time.h>

#include "quipImages.h"
//#include "quipImageView.h"
#include "quip_prot.h"
#include "viewer.h"

uint64_t my_absolute_to_nanoseconds( uint64_t *t )
{
	static mach_timebase_info_data_t	timebase_info;

	if( timebase_info.denom == 0 ){	// relies on bss initialized to 0?
		mach_timebase_info(&timebase_info);
//fprintf(stderr,"timebase_info:  numer = %u, denom = %u\n",timebase_info.numer,
//timebase_info.denom);
	}
	uint64_t ns = (*t) * timebase_info.numer / timebase_info.denom;
	return ns;
}

@implementation quipImages

#ifdef BUILD_FOR_IOS
@synthesize _updateTimer;
#endif // BUILD_FOR_IOS

@synthesize _time0;
@synthesize _time0_2;
@synthesize _flags;
@synthesize _vbl_count;
@synthesize _frame_duration;
@synthesize _queue_idx;
@synthesize refresh_func;
@synthesize frameQueue;
@synthesize afterAnimation;	// Can't combine user-provided setter with synthesized getter...
@synthesize animationStarted;

// refresh_func is a string containing a script fra(nonatomic) gment?
// If it is set, then the text is interpreted every refresh event
//
// afterAnimation is a string that is interpreted at the end of a one-shot event

-(void) set_refresh_func:(const char *)s
{
	if( refresh_func != NULL ) rls_str(refresh_func);
	//[self setCycleFunc:s];
	refresh_func = savestr(s);

	[self enableRefreshEventProcessing];
}

-(const char *) afterAnimation
{
    return afterAnimation;
}

// We have a custom setter not just to release the old string (which could
// be done automatically if we used an NSString), but to set the flag
// and enable refresh event processing...

-(void) setAfterAnimation:(const char *)s
{
    if( afterAnimation != NULL ) rls_str(afterAnimation);
    afterAnimation = savestr(s);
	animationStarted = 0;
	[self enableRefreshEventProcessing];
}

#ifdef BUILD_FOR_IOS

-(void) setRefreshTime
{
	char time_buf[64];
	uint64_t ltime;
	ltime = mach_absolute_time();
	CFTimeInterval t;
	

	// This is not what we need for PVT, because
	// This is called at the start of the trial...
	t = [_updateTimer timestamp];
	if( (QI_QV(self)).baseTime == 0 ){
		[QI_QV(self) setBaseTime:t];
	}
//sprintf(DEFAULT_ERROR_STRING,"_onScreenRefresh:  t = %g, base_time = %g",t,(QI_QV(self)).baseTime);
//sprintf(DEFAULT_ERROR_STRING,"_onScreenRefresh:  _vbl_count = %d, _frame_duration = %d",_vbl_count,_frame_duration);
//NADVISE(DEFAULT_ERROR_STRING);
	sprintf(time_buf,"%g",t - (QI_QV(self)).baseTime);
	assign_var(DEFAULT_QSP_ARG  "refresh_time", time_buf );

	//ltime -= _time0_2;
	ltime -= (QI_QV(self)).baseTime_2;
	uint64_t ns;
	//ns = AbsoluteToNanoseconds( *(AbsoluteTime *) &ltime );
	ns = my_absolute_to_nanoseconds( &ltime );
	sprintf(time_buf,"%g",round(ns/100000)/10.0);
	assign_var(DEFAULT_QSP_ARG  "refresh_time2", time_buf );
}

#endif // BUILD_FOR_IOS

-(void) exec_refresh_func
{
	// This is a one-shot
	[self disableRefreshEventProcessing];

#ifdef BUILD_FOR_IOS
	[self setRefreshTime];
#endif // BUILD_FOR_IOS

	chew_text(DEFAULT_QSP_ARG  refresh_func, "(refresh event)");
}

-(void) animationDone
{
	// Called at the end of a one-shot
	[self disableRefreshEventProcessing];
    if( afterAnimation != NULL ){
        chew_text(DEFAULT_QSP_ARG  afterAnimation, "(refresh event)");
	}
}

// This is called when _updateTimer is added to the run loop...

// We need to support a variety of animation modes;
// The default is continuous cycling of the image stack,
// but another important mode is display of a single frame
// for a fixed interval.  For example, for PVT we need
// to wait for a variable delay, and then swap images.
// We want to take the timestamp at the time of the swap.
//
// Yet another mode that we need to support is one-shot animation
// of a sequence...
//
// The original PVT used a single stimulus image, but when we analyzed Kenji's
// app that had a traditional time counter, we discovered that sometimes the first
// frame was not displayed.  So we would like to display a counter in our PVT as well
// so that we can see if we have the same problem!

-(void) _onScreenRefresh
{
	/* This used to be after the first block - not sure what it is for??? */
	/*
	if( _flags & QI_CHECK_TIMESTAMP ){
		t = [_updateTimer timestamp];
		fprintf(stderr,"_onScreenRefresh:  elapsed time is %g\n",t-_time0);
		_time0 = t;
	}
	*/

	if( refresh_func != NULL ){
		[self exec_refresh_func];
	} else {
        if( afterAnimation != NULL && animationStarted ){
			if( ! self.animating ){
				[self animationDone];
			}
		}
	}
}

-(void) startAnimation
{
	animationStarted = 1;
	// enableRefreshEventProcessing is called if an when we set _afterAnimation
	[self startAnimating];
}

-(NSInteger) subviewCount
{
#ifdef BUILD_FOR_IOS
	NSArray *a;

	a=self.subviews;
	if( a == NULL ) return -1;
	return a.count;
#else // ! BUILD_FOR_IOS
	return 0;	
#endif // ! BUILD_FOR_IOS
}

-(void) hide
{
#ifdef BUILD_FOR_IOS
	[self.superview sendSubviewToBack:self];
#endif // BUILD_FOR_IOS
}

-(void) reveal
{
#ifdef BUILD_FOR_IOS
	[self.superview bringSubviewToFront:self];
#endif
}

-(void) discard_subviews
{
#ifdef BUILD_FOR_IOS
	NSArray *a;
	a=self.subviews;

	while( a.count > 0 ){
		UIView *v;
		v= [a objectAtIndex:0];
		[v removeFromSuperview];
		// We shouldn't need to release the objects,
		// ARC should take care of it.
   		 a=self.subviews;
	}
#endif // BUILD_FOR_IOS

}

-(void) clearQueue
{
	if( self.frameQueue != NULL ){
		[self.frameQueue removeAllObjects];
	} else {
	}
	_queue_idx = 0;
}

-(void) queueFrame: (UIImage *)uii_p
{
	//assert( [self hasSubview:qiv_p] );

	if( self.frameQueue == NULL ){
		// The 60 is somewhat arbitrary, but should be enough for 1 second of animation.
		// Presumably the array can grow if we queue more than 60 frames.
		self.frameQueue = [NSMutableArray arrayWithCapacity:60]; 
	}
	assert( self.frameQueue != NULL );
	[ self.frameQueue addObject:uii_p];	// adds at end of array
}

-(void) enableRefreshEventProcessing
{
	if( _flags & QI_TRAP_REFRESH ){
		NADVISE("quipImages enableRefreshEventProcessing:  WARNING refresh processing is already enabled!?");
		return;
	}
#ifdef BUILD_FOR_IOS

	// What does _updateTimer do???

//fprintf(stderr,"enableRefreshEventProcessing:  adding updateTimer to run loop?\n");
	[_updateTimer addToRunLoop:[NSRunLoop currentRunLoop]
			forMode:NSDefaultRunLoopMode];
#endif // BUILD_FOR_IOS
	_flags |= QI_TRAP_REFRESH;
}

-(void) disableRefreshEventProcessing
{
#ifdef BUILD_FOR_IOS
	[_updateTimer
		removeFromRunLoop:[NSRunLoop currentRunLoop]
		forMode:NSDefaultRunLoopMode];
#endif // BUILD_FOR_IOS
	_flags &= ~QI_TRAP_REFRESH;
}

-(id)initWithSize:(CGSize) size
{
	CGRect r;

	r.origin.x = 0;
	r.origin.y = 0;
	r.size = size;
#ifdef BUILD_FOR_IOS
	self = [super initWithFrame:r];
	//self =[super initWithImage:myimg];
#endif // BUILD_FOR_IOS

	_flags = 0;	// make sure we're not refreshing
	_vbl_count = 0;
	_frame_duration = 15;	// Default is 4 fps for debugging
	refresh_func=NULL;
    afterAnimation=NULL;
	frameQueue = NULL;

#ifdef BUILD_FOR_IOS

	// initialize the timer here,
	// but we don't add it to the run queue
	// until there is an explicit request
	// wasteful?

	_updateTimer = [CADisplayLink
			displayLinkWithTarget:self
			selector:@selector(_onScreenRefresh)];
//fprintf(stderr,"initWithSize:  created updateTimer\n");
#endif // BUILD_FOR_IOS

	self.opaque = YES;
	self.alpha = 1.0;
	self.hidden=NO;

	//self.imageScaling = NSScaleNone;	// not with iOS?

	self.contentMode = UIViewContentModeTopLeft;

	return self;
}  // end initWithSize

@end		// end quipImages implementation

