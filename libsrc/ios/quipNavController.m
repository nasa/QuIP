#include "quipNavController.h"
#include "quip_prot.h"
#include "ios_gui.h"
#include "nav_panel.h"
#include "screen_obj.h"

// global var for whole app...
// default is all
static UIInterfaceOrientationMask quipSupportedInterfaceOrientations=UIInterfaceOrientationMaskAll;
static BOOL quipShouldAutorotate=YES;

void set_supported_orientations( UIInterfaceOrientationMask m )
{
	quipSupportedInterfaceOrientations=m;
}

void set_autorotation_allowed( BOOL yesno )
{
	quipShouldAutorotate = yesno;
}

@implementation quipNavController

// This makes the status bar go away...

-(BOOL) prefersStatusBarHidden {
	return YES;
}

// is this needed??

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


/***************** UINavigationControllerDelegate methods ********/

// We need to call this before the controller is shown...

-(void) navigationController: (QUIP_NAV_CONTROLLER_TYPE *) nc
	willShowViewController: (QUIP_VIEW_CONTROLLER_TYPE *) vc
	animated: (BOOL) yesno
{
//fprintf(stderr,"navigationController 0x%lx  didShowViewController 0x%lx\n",
//(long)nc,(long)vc);
	Gen_Win *gwp;

	gwp = find_genwin_for_vc(vc);
	if( gwp == NULL ){
fprintf(stderr,"No genwin found associated with view controller 0x%lx\n",
(long)vc);
		return;
	}
//fprintf(stderr,"Found genwin '%s' associated with view controller 0x%lx\n",
//GW_NAME(gwp),(long)vc);

	IOS_Item_Context *icp;

	icp = GW_CONTEXT(gwp);

	if( icp != NULL ){
		if( [Screen_Obj contextStackDepth] > 1 )
			pop_scrnobj_context();
		push_scrnobj_context(GW_CONTEXT(gwp));
	}
#ifdef CAUTIOUS
	  else {
		sprintf(DEFAULT_ERROR_STRING,
		"Genwin '%s' has no item context!?\n",GW_NAME(gwp));
		NWARN(DEFAULT_ERROR_STRING);
	}
#endif // CAUTIOUS
}

/***************** end UINavigationControllerDelegate methods ********/

#ifdef BUILD_FOR_IOS

// method override to fix some auto-rotation things...

-() popViewControllerAnimated:(BOOL) animate checkOrientation:(BOOL) check_rot
{
	UIViewController *vc;
	vc=[super popViewControllerAnimated:animate];

	if( check_rot )
		[ [NSNotificationCenter defaultCenter]
		postNotificationName:UIDeviceOrientationDidChangeNotification
		object:nil];

	return vc;
}

#endif // BUILD_FOR_IOS

-(quipNavController *) initWithRootViewController:
				(quipTableViewController *) qtvc
{
#ifdef BUILD_FOR_IOS
	self = [super initWithRootViewController:qtvc];

	self.toolbarHidden = YES;
	self.navigationBarHidden = NO;

	self.navigationBar.tintColor = [UIColor   colorWithRed:102.0/255
						green:52.0/255
						blue:133.0/255
						alpha:1];

	// what is backItem???
	//nvc.navigationBar.backItem.title = @"Custom";
    
	// inherit delegate from parent/super?
	//self.delegate = self;
#endif // BUILD_FOR_IOS

	return self;
}

#ifdef BUILD_FOR_IOS

// BUG - these things need to be user-settable!

- (BOOL) shouldAutorotate
{
	if( [self topViewController] == NULL ){
		return quipShouldAutorotate;
	}
	BOOL yn;
	yn=[[self topViewController] shouldAutorotate];
	return yn;
}


- (UIInterfaceOrientationMask)supportedInterfaceOrientations
{
	// BUG find out the good mask...
	if( [self topViewController] == NULL ){
		return UIInterfaceOrientationMaskAll;
	//	return UIInterfaceOrientationMaskPortrait;
    } else {
		return [[self topViewController] supportedInterfaceOrientations];
    }
}

#ifdef FOOBAR
- (UIInterfaceOrientation)preferredInterfaceOrientationForPresentation
{
	return UIInterfaceOrientationPortrait;	// FOOBAR
}
#endif // FOOBAR

- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
{
	if( [self topViewController] == NULL ) return YES;
	//if( [self topViewController] == NULL ) return NO;

	// The view controller might have a flag that says one thing
	// or another?

	return [[self topViewController]
		shouldAutorotateToInterfaceOrientation:interfaceOrientation];
}

#endif // BUILD_FOR_IOS

// Calling this from a script right after push_nav doesn't always work!?
// The first time, it hides the back button on the previous pane, not the pushed pane...
// But the back button that is erroneously displayed is not active.
// But there is no back function from the pane where the button is erroneously NOT displayed!?

-(void) hideBackButton:(bool) yesno
{
#ifdef BUILD_FOR_IOS
	QUIP_NAV_ITEM_TYPE *nip;
    
	nip=self.navigationBar.topItem;
	[nip setHidesBackButton: yesno animated:NO];
#endif // BUILD_FOR_IOS
    
	// calling setNeedsDisplay does not help with the bug.
	//[self.navigationBar setNeedsDisplay];

#ifdef FOOBAR
	// This is a hack - we pop and re-push to see if
	// button displays correctly???
	UIViewController *vcp, *p;
	vcp = self.topViewController;
	p = [ self popViewControllerAnimated:NO ];
	[ self pushViewController:vcp animated:NO];
#endif // FOOBAR
}

@end


