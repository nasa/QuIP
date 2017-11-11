//
//  quipAppDelegate.h
//  oq4t
//
//  Created by Jeff Mulligan on 7/29/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#ifndef _QUIPAPPDELEGATE_H_
#define _QUIPAPPDELEGATE_H_

#include "quipNavController.h"

#ifdef BUILD_FOR_IOS
#import <UIKit/UIKit.h>
#import <CoreMotion/CoreMotion.h>

#include "ios_list.h"

typedef enum {
	DEV_TYPE_NONE,	// we don't use this, so we know a value of 0 has not been set!
	DEV_TYPE_IPAD2,
	DEV_TYPE_IPOD_RETINA,
	DEV_TYPE_IPAD_PRO_9_7,
	DEV_TYPE_IPAD_PRO_12_9,
	DEV_TYPE_UNKNOWN
} known_device_type ;
#endif // BUILD_FOR_IOS


@class quipViewController;
@class quipView;
@class Panel_Obj;

#ifdef BUILD_FOR_IOS
@interface quipAppDelegate : NSObject <	UIApplicationDelegate,
					UITextFieldDelegate,
					UITextInputDelegate,
					UITextViewDelegate,
					UIPickerViewDelegate,
					UIPickerViewDataSource,
					UITableViewDelegate,
					UITableViewDataSource,
					/* UIAccelerometerDelegate, */
					UINavigationControllerDelegate,
					UIImagePickerControllerDelegate >

@property (nonatomic, retain) UIWindow *	window;
@property (nonatomic, retain) UINavigationController *nvc;
@property known_device_type 			dev_type;
@property NSTimeInterval			accelerometerUpdateInterval;
@property NSTimeInterval			deviceMotionUpdateInterval;

// jbm added to support slave device
@property (retain) CADisplayLink *		wakeup_timer;
@property (retain) IOS_List *			wakeup_lp;

// Gavin added this property
@property (readonly) CMMotionManager *		motionManager;
@property (readonly) CMAccelerometerData *	accelerometerData;
@property (readonly) CMAcceleration		acceleration;

-(void) genericButtonAction:(id) obj;
-(void) genericSwitchAction:(id) obj;
-(void) insure_wakeup;

#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
#include <AppKit/NSTextView.h>
#include <AppKit/NSMatrix.h>
#include <AppKit/NSTextField.h>
@interface quipAppDelegate : NSObject <NSTextViewDelegate,
					NSMatrixDelegate,
					NSTextFieldDelegate>
-(void) init_main_menu;
-(void) populateApplicationMenu:(NSMenu *)submenu;
-(void) populateFileMenu:(NSMenu *)submenu;
-(void) populateEditMenu:(NSMenu *)submenu;
-(void) populateWindowMenu:(NSMenu *)submenu;
-(void) populateHelpMenu:(NSMenu *)submenu;
-(void) genericChooserAction:(id) obj;

#endif // BUILD_FOR_MACOS

// Used by both
-(void) genericSliderAction:(id) obj;

@property (nonatomic, retain) quipViewController *qvc;
//@property (nonatomic, retain) Panel_Obj *console_panel;
@property CGSize dev_size;

@end


extern quipAppDelegate *globalAppDelegate;
extern void finish_launching(void *);

extern quipNavController *root_view_controller;

#endif /* _QUIPAPPDELEGATE_H_ */
