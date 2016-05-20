//
//  quipCanvas.m
//
#import <QuartzCore/QuartzCore.h>

#import "quipCanvas.h"
#import "quip_prot.h"
#include "viewer.h"

@implementation quipCanvas

-(id) initWithSize: (CGSize) size
{
	CGRect r;

	r.origin.x = 0;
	r.origin.y = 0;
	r.size = size;
#ifdef BUILD_FOR_IOS
	self = [super initWithFrame:r];
#endif // BUILD_FOR_IOS
	return self;
}

// The documentation says that the timer selector function
// takes a timer argument, but the one that works is the one
// with no arg.  (The other throws a missing selector error!?)

#ifdef FOOBAR
-(void) canvasFireTimer:(NSTimer *) timer
{
	MAKE_NEEDY(CANVAS_VW(self));
}
#endif /* FOOBAR */

// This is called from drawRect, and would seem to request another call to drawRect???

-(void) canvasFire2
{
	MAKE_NEEDY(CANVAS_VW(self));
}

/*
 // Only override drawRect: if you perform custom drawing.
 // An empty implementation adversely affects performance during animation.
 */

 - (void)drawRect:(CGRect)rect
{
#ifdef BUILD_FOR_IOS
	SET_VW_GFX_CTX( CANVAS_VW(self), UIGraphicsGetCurrentContext());

//fprintf(stderr,"quipCanvas:drawRect:  viewer context set to 0x%lx\n",
//(long)VW_GFX_CTX(CANVAS_VW(self)));

#ifdef CAUTIOUS
	if( VW_GFX_CTX( CANVAS_VW(self) ) == NULL ){
		fprintf(stderr,"UIGraphicsGetCurrentContext returns 0x%lx\n",
			(long)UIGraphicsGetCurrentContext());
		NERROR1("CAUTIOUS:  drawRect:  Unable to obtain current graphics context!?");
		return;
	}
#endif /* CAUTIOUS */
	// Drawing code

//sprintf(DEFAULT_ERROR_STRING,"drawRect:  canvas = 0x%lx",(long)self);
//advise(DEFAULT_ERROR_STRING);
	// This is a quip viewer
	if( VW_DRAW_LIST(CANVAS_VW(self)) != NO_IOS_LIST ){
//		fprintf(stderr,"drawRect %s calling exec_drawlist\n",VW_NAME(CANVAS_VW(self)));
		// erasure is implemented by merely clearing the drawlist.
		// But nothing gets erased until drawRect is called again!?
		// How can we force drawRect to be called??
//sprintf(DEFAULT_ERROR_STRING,"drawRect:  canvas = 0x%lx, calling exec_drawlist",(long)self);
//advise(DEFAULT_ERROR_STRING);
		if( exec_drawlist(CANVAS_VW(self)) < 0 ){
			// negative return value indicates erasure requested...

			// Calling MAKE_NEEDY here seems to have no effect, it is probably
			// cleared when the delegate returns.
			//FLAG_NEEDY(CANVAS_VW(self));

			// What is canvasFire2 for???

			/*NSTimer *nst;
			nst =*/ [NSTimer scheduledTimerWithTimeInterval:0.050
				target:self
				selector:@selector(canvasFire2)
				userInfo:NULL
				repeats:NO
				];

		}
	} else {
		fprintf(stderr,"drawRect %s:  NULL draw list!?\n",VW_NAME(CANVAS_VW(self)));
	}
#endif // BUILD_FOR_IOS
	// Need to do more stuff here??
}

@end
