//
//  quipCanvas.m
//
//  The canvas is intended to be a layer that sits on top of images, that we can draw overlays on.
//  We are having trouble getting it to erase!?

#import <QuartzCore/QuartzCore.h>

#include "quip_config.h"
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
    self.opaque = NO;
#endif // BUILD_FOR_IOS


	return self;
}

// The documentation says that the timer selector function
// takes a timer argument, but the one that works is the one
// with no arg.  (The other throws a missing selector error!?)

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
//fprintf(stderr,"quipCanvas drawRect BEGIN\n");
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

	// This is a quip viewer
	if( VW_DRAW_LIST(CANVAS_VW(self)) != NULL ){
		if( exec_drawlist(CANVAS_VW(self)) < 0 ){
			// Before, a negative return meant that an erase and redraw was needed -
			// But now erase is executed, so it shouldn't need to be called twice!?

			// canvasFire2 causes drawrect to be called again???

			[NSTimer scheduledTimerWithTimeInterval:0.050
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
//fprintf(stderr,"drawRect DONE\n");
}

@end
