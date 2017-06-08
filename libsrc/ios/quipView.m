//
//  quipView.m
//
//  Created by Jeff Mulligan on 7/29/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#import "quipView.h"
#import "quip_prot.h"
#include "viewer.h"

#include <mach/mach.h>
#include <mach/mach_time.h>

@implementation quipView
@synthesize _size;
#ifdef BUILD_FOR_IOS
@synthesize qvc;
@synthesize bgImageView;
#endif // BUILD_FOR_IOS
@synthesize canvas;
@synthesize images;
@synthesize baseTime;
@synthesize baseTime_2;

-(id)initWithSize:(CGSize) size
{
	CGRect r;

	r.origin.x = 0;
	r.origin.y = 0;
	r.size = size;
	baseTime=0.0;
	baseTime_2=0;
	self = [self initWithFrame:r];

	// These lines were added in an attempt
	// to get the background to resize on
	// device rotation, but it didn't work!?
	// HOWEVER - it does make image viewers stretch, which
	// we do not want...
	//[self setAutoresizingMask:UIViewAutoresizingFlexibleWidth];
#ifdef BUILD_FOR_IOS
	self.autoresizesSubviews = NO;
#endif // BUILD_FOR_IOS
	
	return self;
}

- (id)initWithFrame:(CGRect)frame
{
	self = [super initWithFrame:frame];
	if (!self) return self;

	// Initialization code
#ifdef BUILD_FOR_IOS
	[self setMultipleTouchEnabled:YES];
	self.autoresizesSubviews = NO;
#endif // BUILD_FOR_IOS
	SET_QV_SIZE(self,frame.size);
	// Should we disable scrolling by default?
#ifdef BUILD_FOR_IOS
    
#ifdef SCROLLABLE_QUIP_VIEW
	[self setContentSize: frame.size];
#else // ! SCROLLABLE_QUIP_VIEW
    NADVISE("quipView:initWithFrame:  NOT setting content size, not scrollable in this build.");
#endif // ! SCROLLABLE_QUIP_VIEW
    
#endif // BUILD_FOR_IOS
	return self;
}

#ifdef BUILD_FOR_IOS
-(void) addDefaultBG
{
fprintf(stderr,"addDefaultBG calling make_bg_image, w = %f, h = %f\n",
_size.width,_size.height);

	bgImageView = make_bg_image(_size);
	[self addSubview: bgImageView];
}

-(void) process_action: (Canvas_Event_Code) code forTouches: (NSSet *) touches
{

	// Now look for the action text...
	if( QVC_GW(qvc) != NULL ){
		if( QVC_GW(qvc).event_tbl == NULL ) {
			if( verbose ){
				sprintf(DEFAULT_ERROR_STRING,"viewer %s has null event table",
					VW_NAME(QVC_VW(qvc)) );
				NADVISE(DEFAULT_ERROR_STRING);
			}
			return;
		}
		NSString *s = [QVC_GW(qvc).event_tbl objectAtIndex: code];

#ifdef CAUTIOUS
		if( s == NULL ) {
			return;
		}
#endif // CAUTIOUS

		if( *(s.UTF8String) == 0 ){	// empty string?
			// This is the default if nothing has
			// been specified for an event...
			return;
		}

#ifdef BUILD_FOR_IOS
		if( IS_TOUCH_EVENT(code) ){
            char buf[32];
			int n_touches=0;
            
			for( UITouch *t in touches ){
				char var_name[32];

				n_touches++;
				CGPoint p = [t locationInView:self];

				sprintf(var_name,"touch%d_x",n_touches);
				sprintf(buf,"%d",(int)p.x);
				assign_var(DEFAULT_QSP_ARG  var_name,buf);

				sprintf(var_name,"touch%d_y",n_touches);
				sprintf(buf,"%d",(int)p.y);
				assign_var(DEFAULT_QSP_ARG  var_name,buf);
			}
			sprintf(buf,"%d",n_touches);
			assign_var(DEFAULT_QSP_ARG  "n_touches",buf);

			chew_text( DEFAULT_QSP_ARG  s.UTF8String, "(touch event)" );
		} else {
			NWARN("process_action:  unhandled event code!?");
		}
#else // ! BUILD_FOR_IOS
		
		NWARN("process_action:  unhandled event code!?");
		
#endif // ! BUILD_FOR_IOS
		
	} else {
		// We get here for panels, and the console...
		// We should probably ignore these events!?

		//advise("No viewer linked with this quipView");
	}
}	// end process_action

-(void) touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
	[self getEventTime:event];
	[self process_action:CE_TOUCH_DOWN forTouches: touches];
}

-(void) getEventTime:(UIEvent *)event
{
	// How do we read the time here?
	// gettimeofday or something else?
	char time_buf[64];

	uint64_t now_time = mach_absolute_time();

	if( baseTime==0.0 )
		baseTime = event.timestamp;

	if( baseTime_2==0 )
		baseTime_2 = mach_absolute_time();

	sprintf(time_buf,"%g",event.timestamp-baseTime);
	assign_var(DEFAULT_QSP_ARG  "event_time",time_buf);

	now_time -= baseTime_2;
	//uint64_t ns = AbsoluteToNanoseconds( *(AbsoluteTime *) &now_time );
	uint64_t ns = my_absolute_to_nanoseconds( &now_time );
	sprintf(time_buf,"%g",round(ns/100000)/10.0);
	assign_var(DEFAULT_QSP_ARG  "event_time_2",time_buf);
}

-(void) touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event
{
	[self process_action:CE_TOUCH_MOVE forTouches: touches];
}

-(void) touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event
{
	[self process_action:CE_TOUCH_UP forTouches: touches];
}

// This gets called when the touches are taken by a scrollView???

-(void) touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event
{
	//for( UITouch *t in touches ){
	//	printf("touchesCancelled.\n");
	//}
}

#endif // BUILD_FOR_IOS

-(BOOL) canBecomeFirstResponder
{
	return YES;
}

@end

#ifdef BUILD_FOR_IOS

// This creates an image but we'd like to get rid of this eventually...

static QUIP_IMAGE_TYPE * CreateDefaultBG(int pixelsWide, int pixelsHigh)
{
	static IOS_List *lp=NULL;
	IOS_Node *np;
	QUIP_IMAGE_TYPE *ip;

	if( lp != NULL ){
		// Check the list for images of the correct size
		np=IOS_LIST_HEAD(lp);
		while( np != NULL ){
			ip = (QUIP_IMAGE_TYPE *) IOS_NODE_DATA(np);
			// How do we get the dimensions?
			if( ip.size.width== pixelsWide &&
					ip.size.height == pixelsHigh ){

//fprintf(stderr,"CreateDefaultBG:  reusing %d x %d image (0x%lx)\n",
//pixelsWide,pixelsHigh,(long)ip);

				return ip;
			}
			np = IOS_NODE_NEXT(np);
		}
	}

	CGImageRef myimg=NULL;
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	//CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();

	unsigned char *my_buffer = malloc(4*pixelsWide*pixelsHigh);
	unsigned char b=127;
	unsigned char a=255;
	unsigned char r,g, _r, _g, _b;
	float f_r, f_g, dr, dg;
	float r0,g0;
	int i,j;
    float g_frac, r_frac;

	// blue by itself is too dark for black text...
	r0=0;
	g0=0;
	// Try to lighten the whole thing...
	// But this just made it more yellow...
	//r0=160;
	//g0=160;

	dr = (float)((255.0-r0) / (pixelsWide-1));
	dg = (float)((255.0-g0) / (pixelsHigh-1));

	// Originally we had a L-R red ramp, a top-bottom green ramp, and blue filled
	// in the upper left hand corner.  But the blue by itself was too dark for black
	// text to show up, so we want the upper left hand corner to be cyan...

	f_g=g0;
	for(i=0;i<pixelsHigh;i++){
		f_r=255;
		g=(u_char)f_g;
		g_frac = (g-g0)/(255-g0);
		f_g += dg;
		for(j=0;j<pixelsWide;j++){
			r=(u_char)f_r;
			r_frac = (r-r0)/(255-r0);
			f_r-=dr;
			b=(u_char)(255*(1-(r_frac>g_frac?r_frac:g_frac)));
//			float tmp_g = 255-(2*b);
  //          if( tmp_g < 0 ) tmp_g = 0;
    //        g=tmp_g;

#define COLOR_FRACTION 0.2
#define BLEND(c)	(u_char)(COLOR_FRACTION*((float)c)+(1-COLOR_FRACTION)*255.0)

			_r = BLEND(r);
			_g = BLEND(g);
			_b = BLEND(b);
			my_buffer[ (i*pixelsWide+j)*4 + 0 ] = a;
			my_buffer[ (i*pixelsWide+j)*4 + 1 ] = _b;	// B
			my_buffer[ (i*pixelsWide+j)*4 + 2 ] = _g;	// G
			my_buffer[ (i*pixelsWide+j)*4 + 3 ] = _r;	// R
		}
	}

	CGContextRef cref = CGBitmapContextCreateWithData(my_buffer,
					pixelsWide, pixelsHigh, 8,
					4* pixelsWide , colorSpace,
		(kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedLast ),
					NULL,		// release callback
					NULL			// release callback data arg
													);
    CGColorSpaceRelease(colorSpace);  // to prevent memory leak...
	if ( cref == NULL ){
		printf("error creating bitmap context!?\n");
		return NULL;
	}
	myimg = CGBitmapContextCreateImage(cref);
    CGContextRelease(cref);

#ifdef BUILD_FOR_IOS
	ip = [QUIP_IMAGE_TYPE imageWithCGImage:myimg];
#endif // BUILD_FOR_IOS
	
	np = mk_ios_node(ip);
	if( lp == NULL ) lp = new_ios_list();
	ios_addTail(lp,np);

//fprintf(stderr,"CreateDefaultBG:  created %d x %d image (0x%lx)\n",
//pixelsWide,pixelsHigh,(long)ip);
	// myimg is retained by the property setting above,
	// so we can release the original
	// BUT do we really need to do this if we are using ARC?
	CGImageRelease(myimg);

	return ip;
} // end createDefaultBG

// BUG we make a new image for every viewer - we should share between
// viewers of the same size

QUIP_IMAGE_VIEW_TYPE *make_bg_image(CGSize siz)
{
	QUIP_IMAGE_VIEW_TYPE *iv;

	QUIP_IMAGE_TYPE *uip=CreateDefaultBG((int)siz.width,(int)siz.height);
    

	iv=[[QUIP_IMAGE_VIEW_TYPE alloc] initWithImage:uip];
	iv.alpha = 1.0;
	iv.hidden=NO;
	iv.contentMode = UIViewContentModeScaleToFill;	// should be default...
	return iv;
}
#endif // BUILD_FOR_IOS

