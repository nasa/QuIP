//
//  quipViewController.h
//
//  Created by Jeff Mulligan on 7/29/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//
#ifndef _QUIPVIEWCONTROLLER_H_
#define _QUIPVIEWCONTROLLER_H_

#ifdef BUILD_FOR_IOS
#include <UIKit/UIKit.h>
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
#include "quipWindowController.h"
#endif // BUILD_FOR_MACOS

#include "quipAppDelegate.h"

@class Gen_Win;

@interface quipViewController : QUIP_VIEW_CONTROLLER_TYPE

@property (nonatomic, retain) quipAppDelegate *	qadp;
@property CGSize				_size;
@property (retain) Gen_Win *			qvc_gwp;	// retain because this points back...
@property int					qvc_flags;
@property const char *				qvc_done_action;

-(id) initWithSize : (CGSize) size  withDelegate:(quipAppDelegate *)adp;
//-(void) attach_view:(quipView *)qv;
//-(void) attach_uiview:(UIView *)uiv;

#ifdef BUILD_FOR_IOS
-(void) addConsoleWithSize:(CGSize)size;
-(BOOL) didBlockAutorotation;
-(void) clearBlockedAutorotation;
-(void) hideBackButton:(bool) yesno;
#endif // BUILD_FOR_IOS

-(void) qvcExitProgram;
-(void) releaseView;
-(void) setDoneAction:(const char *) action;
-(void) qvcDoneButtonPressed;
@end

// bits for qvc_flags
#ifdef BUILD_FOR_IOS
#define QVC_ALLOWS_AUTOROTATION		1
#define QVC_BLOCKED_AUTOROTATION	2
#define QVC_HIDE_BACK_BUTTON		4
#define QVC_QV(qvc)		((quipView *)(qvc).view)
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
#define QVC_QV(qvc)		((quipView *)(qvc).view)
#endif // BUILD_FOR_MACOS

extern void addConsoleToQVC(quipViewController *qvc_p);

#define QVC_GW(qvc)		(qvc).qvc_gwp

// We keep these because of the casts
#define QVC_VW(qvc)		GW_VW(QVC_GW(qvc))
#define QVC_PO(qvc)		GW_PO(QVC_GW(qvc))

#define SET_QVC_GW(qvc,v)		(qvc).qvc_gwp = v

#endif /* _QUIPVIEWCONTROLLER_H_ */
