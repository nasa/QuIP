#ifndef _QUIPNAVCONTROLLER_H_
#define _QUIPNAVCONTROLLER_H_

//@class quipAppDelegate;
//@class Nav_Panel;
//@class Nav_Item;

#include "quipTableViewController.h"

#ifdef BUILD_FOR_IOS
@interface quipNavController : UINavigationController
	<UIAlertViewDelegate, UINavigationControllerDelegate>
-(UIViewController *) popViewControllerAnimated:(BOOL) anim checkOrientation:(BOOL) check;
#endif //BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
@interface quipNavController : NSViewController
#endif // BUILD_FOR_MACOS

-(quipNavController *) initWithRootViewController:(quipTableViewController *) qtc;
-(void) hideBackButton:(bool)yesno;

@end

#endif /* ! _QUIPNAVCONTROLLER_H_ */

