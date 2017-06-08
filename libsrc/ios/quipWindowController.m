#include "quip_config.h"
#include "quipWindowController.h"
#include "quipAppDelegate.h"
#include "screen_obj.h"
#include "ios_gui.h"

@implementation quipWindowController

// This makes the status bar go away...

-(BOOL) prefersStatusBarHidden {
	return YES;
}

/*********** NSWindowDeleget methods *************/

-(void) windowDidResize:(NSNotification *) notif
{
//fprintf(stderr,"windowDidResize:  %s, obj 0x%lx user_info 0x%lx\n",
//notif.name.UTF8String,
//(long)notif.object,
//(long)notif.userInfo);
//	NSWindow *win = notif.object;

//	NSRect f = win.frame;
//fprintf(stderr,"width = %f, height = %f\n",f.size.width,f.size.height);
}


-(void) windowDidMove:(NSNotification *) notif
{
//fprintf(stderr,"windowDidMove:  %s, obj 0x%lx user_info 0x%lx\n",
//notif.name.UTF8String,
//(long)notif.object,
//(long)notif.userInfo);
	
//	NSWindow *win = notif.object;

//	NSRect f = win.frame;
//fprintf(stderr,"x = %f, y = %f\n",f.origin.x,f.origin.y);
}

/*********** end of NSWindowDeleget methods *************/

-(void) windowButtonAction: (id) obj
{
	Screen_Obj *sop;
	sop = find_scrnobj(obj);
	assert( sop != NULL );
	//fprintf(stderr,"Found %s\n",SOB_NAME(sop));
	chew_text(DEFAULT_QSP_ARG  SOB_ACTION(sop), "(button)" );
}

-(void) windowChooserAction: (id) obj
{
	[globalAppDelegate genericChooserAction:obj];
}

-(void) windowSliderAction: (id) obj
{
	[globalAppDelegate genericSliderAction:obj];
}

-(id) initWithWindow:(NSWindow *) window
{
//fprintf(stderr,"quipWindowController initializing\n");
	return [super initWithWindow:window];
}

-(void) windowDidLoad
{
	fprintf(stderr,"quipWindowController:  window did load!\n");
}

-(void) windowWillLoad
{
	fprintf(stderr,"quipWindowController:  window will load!\n");
}

-(NSString *)windowTitleForDocumentDisplayName: (NSString *) dname
{
	fprintf(stderr,"quipWindowController:  dname = %s\n",
		dname.UTF8String);
	return dname;
}

@end

