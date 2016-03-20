//
//  quipCanvas.h
//

#ifndef _QUIPCANVAS_H_
#define _QUIPCANVAS_H_

#ifdef BUILD_FOR_IOS
#import <UIKit/UIKit.h>
#endif // BUILD_FOR_IOS


// A subview of the main quipView that gets drawRect calls to redraw graphics...

#ifdef BUILD_FOR_IOS
@interface quipCanvas : UIView
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
@interface quipCanvas : NSObject
#endif // BUILD_FOR_MACOS


// We don't keep a pointer to the parent quipView, we can get it as superview...
-(id) initWithSize:(CGSize)size;
@end

#define CANVAS_QV(cp)		((quipView *)(cp).superview)
#define CANVAS_VW(cp)		QVC_VW(QV_QVC(CANVAS_QV(cp)))

#define SET_CANVAS_VW(cp,vp)	SET_QV_VW(CANVAS_QV(cp),vp)

#define NO_CANVAS	((quipCanvas *)NULL)


#endif /* ! _QUIPCANVAS_H_ */

