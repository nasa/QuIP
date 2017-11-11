//
//  quipImages.h
//

#ifndef _QUIPIMAGES_H_
#define _QUIPIMAGES_H_

#ifdef BUILD_FOR_IOS
#import <UIKit/UIKit.h>
#endif // BUILD_FOR_IOS


#include "data_obj.h"

// Should this be a subclass of UIView or quipView???
// We'd like to inherit the event-handling routines from quipView,
// but we don't need (or want) background image stuff...

// Do we have a retention cycle if this view points to its parent UIView?
// (we do have a retention cycle between viewers and quipViews...)
//
@class Viewer;

#ifdef BUILD_FOR_IOS
@interface quipImages : UIView
// Should this timer be per-viewer/images?
// Why not have one for the whole app?
@property (retain) CADisplayLink *	_updateTimer;
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
#include <AppKit/NSView.h>
@interface quipImages : NSView
#endif // BUILD_FOR_MACOS


@property CFTimeInterval		_time0;
@property uint64_t			_time0_2;	// for mach_absolute_time()
@property long				_flags;
@property int				_vbl_count;
@property int				_frame_duration;
@property const char *			cycle_func;

-(id) initWithSize:(CGSize)size;
-(void) set_refresh:(int)duration;
-(void) cycle_images;
-(void) hide;
-(void) reveal;
-(void) set_cycle_func:(const char *)s;
-(void) discard_subviews;
-(void) enableUpdates;
-(void) disableUpdates;
-(void) bring_to_front:(struct data_obj *)dp;
-(void) send_to_back:(struct data_obj *)dp;
-(int) hasImageFromDataObject:(struct data_obj *)dp;
-(void) removeImageFromDataObject:(struct data_obj *)dp;
@end

// flag bits
#define QI_CHECK_TIMESTAMP	1
#define QI_TRAP_REFRESH		2

// The image stack is a subview of the quipView's topView (a UIView),
// which is in turn
#define QI_QV(qip)		((quipView *)(qip).superview)
#define QI_VW(qip)		QVC_VW(QV_QVC(QI_QV(qip)))

#define SET_QI_VW(qip,vp)	SET_QV_VW(QI_QV(qip),vp)

#endif /* ! _QUIPIMAGES_H_ */

