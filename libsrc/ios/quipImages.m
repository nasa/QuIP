//
//  quipImages.m
//
#include <QuartzCore/QuartzCore.h>

// for mach_absolute_time()
//#include <CoreServices/CoreServices.h>
#include <mach/mach.h>
#include <mach/mach_time.h>

#include "quipImages.h"
#include "quipImageView.h"
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
@synthesize cycle_func;

-(void) set_cycle_func:(const char *)s
{
	if( cycle_func != NULL ) rls_str(cycle_func);
	//[self setCycleFunc:s];
	cycle_func = savestr(s);

	[self enableUpdates];
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
//sprintf(DEFAULT_ERROR_STRING,"_refresh:  t = %g, base_time = %g",t,(QI_QV(self)).baseTime);
//sprintf(DEFAULT_ERROR_STRING,"_refresh:  _vbl_count = %d, _frame_duration = %d",_vbl_count,_frame_duration);
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

-(void) run_cycle_func
{
	// This is a one-shot
	[self disableUpdates];

#ifdef BUILD_FOR_IOS
	[self setRefreshTime];
#endif // BUILD_FOR_IOS

	chew_text(DEFAULT_QSP_ARG  cycle_func, "(refresh event)");
}

-(void) check_cycle
{
	/* If there is no cycle_func, we count up until we cycle frames */

	if( _vbl_count < _frame_duration ){
		// Give this frame more time
		_vbl_count ++;
	} else {
		// Time to cycle
		_vbl_count = 1;		// We count the new frame
//fprintf(stderr,"refresh:  cycling image\n",_vbl_count);
		[self cycle_images];
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
// The original PVT used a single stimulus image, but when we analyzed Kenji's
// app that had a traditional time counter, we discovered that sometimes the first
// frame was not displayed.  So we would like to display a counter in our PVT as well
// so that we can see if we have the same problem!

-(void) _refresh
{
	/* This used to be after the first block - not sure what it is for??? */
	/*
	if( _flags & QI_CHECK_TIMESTAMP ){
		t = [_updateTimer timestamp];
		fprintf(stderr,"_refresh:  elapsed time is %g\n",t-_time0);
		_time0 = t;
	}
	*/

	if( cycle_func != NULL ){
		[self run_cycle_func];
	} else {
		[self check_cycle];
	}
}

-(void) cycle_images
{
	// Rotate the subviews.
	// the highest index in the array is in the front,
	// 0 is the rear-most, we bring it to the front.
#ifdef BUILD_FOR_IOS
	NSArray *a;
	a=self.subviews;
	if( a == NULL ) return;

	UIView *v;
	v= [a objectAtIndex:0];
	if( v == NULL ) return;


/*
sprintf(DEFAULT_ERROR_STRING,"cycle_images:  bringing view at 0x%lx to front, superview = 0x%lx, supersuper = 0x%lx",
(long)v,(long)self,(long) self.superview);
NADVISE(DEFAULT_ERROR_STRING);
*/

	[self bringSubviewToFront:v];
#endif // BUILD_FOR_IOS
}

-(int) indexForDataObject:(Data_Obj *)dp
{
#ifdef BUILD_FOR_IOS
	NSArray *a;

	a=self.subviews;
	if( a == NULL ) return -1;

	int i;
	for(i=0;i<a.count;i++){
		quipImageView *qiv_p;
		qiv_p = [a objectAtIndex:i];
		if( qiv_p.qiv_dp == dp ) return i;
	}
#endif // BUILD_FOR_IOS
	return -1;
}

-(int) hasImageFromDataObject:(Data_Obj *)dp
{
	int i;

	i=[self indexForDataObject:dp];
	if( i < 0 ) return NO;
	return YES;
}

-(void) removeImageFromDataObject:(Data_Obj *)dp
{
	int i;

	i=[self indexForDataObject:dp];
#ifdef CAUTIOUS
	if( i < 0 ){
		sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  removeImageFromDataObject:  object %s not found in image list!?",
			OBJ_NAME(dp));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

#ifdef BUILD_FOR_IOS
	quipImageView *qiv_p;
	qiv_p = [self.subviews objectAtIndex:i];
	[qiv_p removeFromSuperview];

	// now we want to deallocate the imageView...
	// what is the reference count???
	// are there any other references???

#endif // BUILD_FOR_IOS
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
fprintf(stderr,"reveal:  bringing subview to front...\n");
	[self.superview bringSubviewToFront:self];
#endif
}

-(void) bring_to_front : (Data_Obj *) dp
{
#ifdef BUILD_FOR_IOS
	NSArray *a;
	int i;

	a=self.subviews;
	if( a == NULL ){
		NWARN("bring_to_front:  no subviews!?");
		return;
	}

	for ( i=0; i<a.count; i++ ){
		quipImageView * qiv_p;

		qiv_p = [a objectAtIndex:i];
		if( dp == qiv_p.qiv_dp ){	// found a match!
			[self bringSubviewToFront:qiv_p];
			return;
		}
	}
	NWARN("bring_to_front:  no subview found matching image");
#endif // BUILD_FOR_IOS
}

-(void) send_to_back : (Data_Obj *) dp
{
#ifdef BUILD_FOR_IOS
	NSArray *a;
	int i;

	a=self.subviews;
	if( a == NULL ){
		NWARN("send_to_back:  no subviews!?");
		return;
	}

	for ( i=0; i<a.count; i++ ){
		quipImageView * qiv_p;

		qiv_p = [a objectAtIndex:i];
		if( dp == qiv_p.qiv_dp ){	// found a match!
			[self sendSubviewToBack:qiv_p];
			return;
		}
	}
	NWARN("send_to_back:  no subview found matching image");
#endif // BUILD_FOR_IOS
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

-(void) enableUpdates
{
	if( _flags & QI_TRAP_REFRESH ){
		//NWARN("quipImages enable_refresh:  refresh processing is already enabled!?");
		NADVISE("quipImages enableUpdates:  WARNING refresh processing is already enabled!?");
		return;
	}
#ifdef BUILD_FOR_IOS

	// What does _updateTimer do???

//fprintf(stderr,"enableUpdates:  adding updateTimer to run loop?\n");
	[_updateTimer addToRunLoop:[NSRunLoop currentRunLoop]
			forMode:NSDefaultRunLoopMode];
#endif // BUILD_FOR_IOS
	_flags |= QI_TRAP_REFRESH;
}

-(void) disableUpdates
{
#ifdef BUILD_FOR_IOS
	[_updateTimer
		removeFromRunLoop:[NSRunLoop currentRunLoop]
		forMode:NSDefaultRunLoopMode];
#endif // BUILD_FOR_IOS
	_flags &= ~QI_TRAP_REFRESH;
}

// We don't do this by default to avoid bogging things down...
// Disable with a requested duration <= 0

-(void) set_refresh:(int) duration
{
	if( duration > 0 ){
fprintf(stderr,"set_refresh(%d):  enabling updates\n",duration);
		_frame_duration = duration;
		[self enableUpdates];
	} else {
		if( (_flags & QI_TRAP_REFRESH) == 0 ){
//NWARN("quipImages enable_refresh:  refresh processing is already disabled!?");
#ifdef BUILD_FOR_IOS
			sprintf(DEFAULT_ERROR_STRING,
		"set_refresh:  refresh processing is already disabled for viewer %s!?",
				VW_NAME(QI_VW(self)));
			NADVISE(DEFAULT_ERROR_STRING);
			// We get this message when we flip windows around - why?
#endif // BUILD_FOR_IOS
			return;
		}
		[self disableUpdates];
		_frame_duration = 0;
	}
}

-(id)initWithSize:(CGSize) size
{
	CGRect r;

	r.origin.x = 0;
	r.origin.y = 0;
	r.size = size;
#ifdef BUILD_FOR_IOS
	self = [super initWithFrame:r];
#endif // BUILD_FOR_IOS

	_flags = 0;	// make sure we're not refreshing
	_vbl_count = 0;
	_frame_duration = 15;	// Default is 4 fps for debugging
	cycle_func=NULL;
#ifdef BUILD_FOR_IOS

	// initialize the timer here,
	// but we don't add it to the run queue
	// until there is an explicit request
	// wasteful?

	_updateTimer = [CADisplayLink
			displayLinkWithTarget:self
			selector:@selector(_refresh)];
//fprintf(stderr,"initWithSize:  created updateTimer\n");
#endif // BUILD_FOR_IOS

	return self;
}  // end initWithSize

@end		// end quipImages implementation

