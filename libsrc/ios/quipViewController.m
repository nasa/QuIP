//
//  quipViewController.m
//
//  Created by Jeff Mulligan on 7/29/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//
#include "quip_config.h"

#import "quipViewController.h"
#import "quipTableViewController.h"	// first_quip_controller
#import "quipView.h"
#import "ios_gui.h"

// At one time we thought that we might like to have one view
// controller instance per window - but the system doesn't
// like multiple instances of the same controller
// pushed onto the nav stack?
//
// How are we doing this now???

@implementation quipViewController
@synthesize qadp;	// quip App Delegate ptr
@synthesize _size;
@synthesize qvc_gwp;
@synthesize qvc_flags;
@synthesize qvc_done_action;

#ifdef BUILD_FOR_IOS
-(void) clearBlockedAutorotation
{
	qvc_flags &= ~QVC_BLOCKED_AUTOROTATION;
}

// This makes the status bar go away...

-(BOOL) prefersStatusBarHidden {
	return YES;
}

#endif // BUILD_FOR_IOS


#ifdef OLD
/********** UIAlertView delegate methods ************/

- (void)alertView:(QUIP_ALERT_OBJ_TYPE *)alertView didDismissWithButtonIndex:(NSInteger)buttonIndex
{
	dismiss_quip_alert(alertView,buttonIndex);
}

- (void)didPresentAlertView:(QUIP_ALERT_OBJ_TYPE *)alertView
{
	quip_alert_shown(alertView);
}

/********** end UIAlertView delegate methods ************/
#endif // OLD

#ifdef BUILD_FOR_IOS
- (void)didReceiveMemoryWarning
{
	// Releases the view if it doesn't have a superview.
	[super didReceiveMemoryWarning];

	// Release any cached data, images, etc that aren't in use.
}

- (void) hideBackButton:(bool) yesno
{
	if( yesno )
		qvc_flags |= QVC_HIDE_BACK_BUTTON;
	else
		qvc_flags &= ~QVC_HIDE_BACK_BUTTON;
}
#endif // BUILD_FOR_IOS

-(void) qvcDoneButtonPressed
{
	done_button_pushed=1;
	assert( qvc_done_action != NULL );
	chew_text(DEFAULT_QSP_ARG  qvc_done_action, "(done button)" );
}

-(void) setDoneAction:(const char *) action
{
	if( qvc_done_action != NULL )
		rls_str(qvc_done_action);
#ifdef BUILD_FOR_IOS
	if( !strcmp(action,"no") ){
		qvc_done_action=NULL;
		self.navigationItem.rightBarButtonItem = NULL;
	} else {
		qvc_done_action=savestr(action);

		// Now set the property on the navigation item...
		UIBarButtonItem *item = [[UIBarButtonItem alloc]
			initWithBarButtonSystemItem:UIBarButtonSystemItemDone
			target:self
			action:@selector(qvcDoneButtonPressed)];
			self.navigationItem.rightBarButtonItem = item;
	}
#endif // BUILD_FOR_IOS
}

#pragma mark - View lifecycle

// Implement viewDidLoad to do additional setup after loading the view, typically from a nib.
//
// We've discovered a problem that subviews can be deleted after a memory warning!?
// The view controller's view can be released if a memory warning is received
// when the view is off-screen.  When we navigate back to the page, loadView is called
// again, BUT the interpreter commands to create the widgets are not re-called,
// and so the panel appears blank.  One solution to this problem would be to not
// call super.didReceiveMemoryWarning (above), but that seems not in the spirit of iOS...

#ifdef BUILD_FOR_IOS

- (void)viewDidLoad
{
	[super viewDidLoad];

	// This assignment causes the name to appear in the nav bar.
	self.title = self.qvc_gwp.name;

	// hide the navigation bar  
	// use setNavigationBarHidden:animated: if you need animation  
	//self.navigationController.navigationBarHidden = YES;  
}

#endif // BUILD_FOR_IOS

-(void) qvcExitProgram
{
	exit(0);
}


#ifdef FOOBAR

- (void)viewDidUnload
{
	[super viewDidUnload];
	// Release any retained subviews of the main view.
	// e.g. self.myOutlet = nil;
}
#endif /* FOOBAR */

#ifdef BUILD_FOR_IOS

-(void) addConsoleWithSize:(CGSize) size
{
	addConsoleToQVC(self);
}

#endif // BUILD_FOR_IOS

#ifdef FOOBAR
// hoped this would fix the wait_fences problem, but it didn't...
// AND it never got called so there was no console!?

-(void) viewDidAppear
{
	//[super viewDidAppear];
}
#endif /* FOOBAR */

#ifdef BUILD_FOR_IOS

- (BOOL) shouldAutorotate
{
	if( qvc_flags & QVC_ALLOWS_AUTOROTATION )
		return YES;
	else {
		qvc_flags |= QVC_BLOCKED_AUTOROTATION;
		return NO;
	}
}

- (BOOL) didBlockAutorotation
{
	if( qvc_flags & QVC_BLOCKED_AUTOROTATION )
		return YES;
	else	return NO;
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations
{
	if( qvc_flags & QVC_ALLOWS_AUTOROTATION )
		return UIInterfaceOrientationMaskAll;
	else {
		/* BUG - we should remember the orientation when
		 * the view controller is created...
		 */
		qvc_flags |= QVC_BLOCKED_AUTOROTATION;
		return UIInterfaceOrientationMaskPortrait;
	}
}
/*
 Deprecated
 
 Override the supportedInterfaceOrientations and preferredInterfaceOrientationForPresentation methods instead.
 */

- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
{
	if( qvc_flags & QVC_ALLOWS_AUTOROTATION )
		return YES;
	else {
		qvc_flags |= QVC_BLOCKED_AUTOROTATION;
		return NO;
	}
}

#endif // BUILD_FOR_IOS

// When is loadView called?
// Load view is called when the view property of the controller is requested.
// If we initialized from a nib file, then it would get done then,
// but since we are not doing that we have to do it here.
//
// loadView initialized the view controllers internal view property...
// This can get freed when there is a low memory warning, and loadView will
// be called again later... (by the navigation controller?)
//
// quipView is a UIScrollView...

-(void) loadView
{
	quipView *qv;

//fprintf(stderr,"quipViewController.loadView creating a quipView\n");
	qv = [[quipView alloc] initWithSize:_size];
	SET_QV_QVC(qv,self);	// BUG?  needs to be a weak reference?
	// set the background image
#ifdef BUILD_FOR_IOS
	[qv addDefaultBG];
#endif // BUILD_FOR_IOS
	self.view = qv;
}

-(void) releaseView
{
	if( self.view != NULL )
        // This line generates a compiler warning???  Null passed to a callee that requires a non-null argument
		self.view = NULL;	// remove the ref, should free the view?
}

-(id) initWithSize : (CGSize) size withDelegate:(quipAppDelegate *)adp
{
#ifdef BUILD_FOR_IOS
	self=[super init];
#endif // BUILD_FOR_IOS
#ifdef BUILD_FOR_MACOS
	self=[super initWithNibName:nil bundle:nil];
#endif // BUILD_FOR_MACOS

//fprintf(stderr,"quipViewController initWithSize delegate = 0x%lx\n",(long)adp);
	qadp=adp;
	CGRect r;
	// This is the view controller???
	r.origin.x = 0;
	r.origin.y = 0;
	r.size = size;
	_size = size;

#ifdef BUILD_FOR_IOS
	//qvc_flags = QVC_ALLOWS_AUTOROTATION;
	qvc_flags = 0;
#endif // BUILD_FOR_IOS

	// The view property is set by loadView

//fprintf(stderr,"quipViewController initWithSize DONE\n");
	return self;
}

@end

