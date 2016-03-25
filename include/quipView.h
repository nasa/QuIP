//
//  quipView.h
//
//  Created by Jeff Mulligan on 7/29/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef _QUIPVIEW_H_
#define _QUIPVIEW_H_

#ifdef BUILD_FOR_IOS
#import <UIKit/UIKit.h>
#endif // BUILD_FOR_IOS


#include "quipCanvas.h"
#include "quipImages.h"

typedef struct device_prefs {
	int	type_code;
	int	font_size;
	int	text_width;
} Device_Prefs;

#define DEV_TYPE_DEFAULT	DEV_TYPE_IPAD2

@class Viewer;
@class Panel_Obj;

// We changed from UIView to UIScrollView to enable scrolling automatically...
// But that is a problem when we simply want to catch tap events, because
// the scroll view catches touch events first to see if they are scroll swipes.
// But maybe this can be fixed simply by disabling scrolling?

// enable this #define for the normal behavior
// comment out to test reaction times without scrolling...

// But in our test, the PVT added latency was the same whether or not scrolling
// was enabled, so we leave it in.
#define SCROLLABLE_QUIP_VIEW

#ifdef BUILD_FOR_IOS

@class quipViewController;

#ifdef SCROLLABLE_QUIP_VIEW
@interface quipView : UIScrollView
#else // ! SCROLLABLE_QUIP_VIEW
@interface quipView : UIView
#endif // ! SCROLLABLE_QUIP_VIEW

-(void) getEventTime:(UIEvent *)event;
-(void) addDefaultBG;
@property (nonatomic, retain) QUIP_IMAGE_VIEW_TYPE *	bgImageView;
@property (retain) quipViewController *	qvc;
#define QV_QVC(qv)		(qv).qvc
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
@class quipWindowController;
@interface quipView : NSView
@property (retain) quipWindowController *	qwc;
#define QV_QWC(qv)		(qv).qwc
#endif // BUILD_FOR_MACOS


// Here are the subviews, from back to front:
// every quipView has a background image...
@property (nonatomic, retain) quipImages *	images;		// for movies
@property (nonatomic, retain) quipCanvas *	canvas;		// for drawing

// We tried having a subview for the controls, but it blocked touch events...
// where do the controls go???

// More properties...
@property CGSize				_size;
@property NSTimeInterval			baseTime;

-(id) initWithSize:(CGSize)size;

#define QV_SIZE(qv)		(qv)._size
#define QV_BG_IMG(qv)		(qv).bgImageView
#define QV_CANVAS(qv)		(qv).canvas
#define QV_IMAGES(qv)		(qv).images
#define SET_QV_QVC(qv,c)	(qv).qvc = c

#define SET_QV_SIZE(qv,s)	(qv)._size = s
#define SET_QV_BG_IMG(qv,v)	(qv).bgImageView = v
#define SET_QV_CANVAS(qv,v)	(qv).canvas = v
#define SET_QV_IMAGES(qv,v)	(qv).images = v

@end

extern QUIP_IMAGE_VIEW_TYPE *make_bg_image(CGSize siz);

#endif /* ! _QUIPVIEW_H_ */

