#ifndef _QUIPTABLEVIEWCONTROLLER_H_
#define _QUIPTABLEVIEWCONTROLLER_H_

@class quipAppDelegate;
@class Nav_Panel;
@class Nav_Item;

#ifdef BUILD_FOR_IOS
//#include <NSString.h>

@interface quipTableViewController : UITableViewController <UIAlertViewDelegate>
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
#include <AppKit/NSViewController.h>
@interface quipTableViewController : NSViewController
#endif // BUILD_FOR_MACOS

// This is not needed now???
//@property (nonatomic, retain) NSMutableArray * menuList;

// BUG circular refs, need to use weak reference...
// Not critical if we're not deleting panels...
@property (nonatomic, retain) Nav_Panel *nav_panel;
@property (nonatomic, retain) NSString *done_action;

-(id) initWithSize:(CGSize) size withDelegate:(quipAppDelegate*) adg withPanel:(Nav_Panel *) nav_p;
//-(void) add_nav_item:(Nav_Item *) nav_i;
-(void) qtvcExitProgram;
+(void) exitProgram;
-(void) qtvcDoneButtonPressed;
-(void) addDoneButton: (const char *) action;
-(BOOL) didBlockAutorotation;
@end


extern quipTableViewController *first_quip_controller;

#define QTVC_NAME(qnc)		(qnc).nav_panel.name.UTF8String
#define QTVC_NAMEOBJ(qnc)	(qnc).nav_panel.name

// BUG UGLY these are global vars that belong somewhere else
extern int done_button_pushed;
extern int xcode_debug;

#endif /* ! _QUIPTABLEVIEWCONTROLLER_H_ */

