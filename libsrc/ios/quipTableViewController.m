// the table view controller for the main menu

#include "quipTableViewController.h"
#include "quipView.h"
#include "quip_prot.h"
#include "nav_panel.h"
#include "ios_gui.h"

//int done_button_pushed=0;
int done_button_pushed=0;

// Is this something that is not supposed to be done?
#ifdef BUILD_FOR_IOS
// This is almost certain to get us kicked out of the app store!?
@interface UIApplication (Private)
- (void)suspend;
@end

static NSString *kCellIdentifier = @"MyIdentifier";

#endif // BUILD_FOR_IOS


@implementation quipTableViewController

//@synthesize menuList;
@synthesize nav_panel;
@synthesize done_action;

// This makes the status bar go away...

-(BOOL) prefersStatusBarHidden {
	return YES;
}

// method override to fix some auto-rotation things...
-(void) clearBlockedAutorotation
{
	// nop
}

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

-(void) qtvcExitProgram
{
	done_button_pushed=1;
#ifdef BUILD_FOR_IOS
	[[UIApplication sharedApplication] suspend];
#endif // BUILD_FOR_IOS
}

+(void) exitProgram
{
	done_button_pushed=1;
#ifdef BUILD_FOR_IOS
	[[UIApplication sharedApplication] suspend];
#endif // BUILD_FOR_IOS
}

- (BOOL) didBlockAutorotation
{
	return NO;
}


-(id) initWithSize: (CGSize) size withDelegate:dgp withPanel:(Nav_Panel *)nav_p
{
#ifdef BUILD_FOR_IOS
	self = [super initWithStyle: UITableViewStyleGrouped];

fprintf(stderr,"quipTableViewControoler initWithSize delegate = 0x%lx\n",(long)dgp);

//fprintf(stderr,"quipTableViewControoler delegate set to self = 0x%lx\n",(long)self);
	//self.tableView.delegate = self;
	self.tableView.delegate = dgp;

	self.tableView.dataSource = self;
	// BUG - this makes a new background image for every panel.
	// After rewriting make_bg_image to reuse the UIImageView,
	// we hang!?
	self.tableView.backgroundView = make_bg_image(size);
	self.nav_panel = nav_p;
// we used to do this in viewDidLoad, but it was called too soon...
//fprintf(stderr,"Setting title to %s, nav_panel = 0x%lx\n",nav_p.name.UTF8String,(long)nav_p);
	self.title = nav_p.name;
	//menuList = [NSMutableArray array];

	//if( dgp

	// set nav bar to not hidden here?
#endif // BUILD_FOR_IOS
	return self;

}

// delegate methods
#ifdef BUILD_FOR_IOS

- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView
{
	NSInteger n;
	n = ios_eltcount(nav_panel.groups);
	return n;
}

- (NSIndexPath *)tableView:(UITableView *)tv willSelectRowAtIndexPath:(NSIndexPath *)indexPath
{
//	fprintf(stderr,
//"will select row %d from section %d\n",indexPath.row,indexPath.section);
	return indexPath;
}

// Data source methods

// this is a private method

-(Nav_Group *) groupForSection:(int) section
{
	// look up the group...
	IOS_Node *np;
	np = ios_nth_elt(nav_panel.groups,section);
	return (Nav_Group *) IOS_NODE_DATA(np);
}

// How do we change the font for the header title??


- (UIView *)tableView:(UITableView *)tableView viewForHeaderInSection:(NSInteger)section
{
	// create the parent view that will hold header Label
	// BUG - should we get the dimensions from device_size?
	UIView* customView = [[UIView alloc] initWithFrame:CGRectMake(10.0, 0.0, 300.0, 44.0)];

	// create the button object
	UILabel * headerLabel = [[UILabel alloc] initWithFrame:CGRectZero];
	headerLabel.backgroundColor = [UIColor clearColor];
	headerLabel.opaque = NO;
	headerLabel.textColor = [UIColor blackColor];
	headerLabel.highlightedTextColor = [UIColor whiteColor];
	headerLabel.font = [UIFont boldSystemFontOfSize:20];
	headerLabel.frame = CGRectMake(10.0, 0.0, 300.0, 44.0);

	// If you want to align the header text as centered
	// headerLabel.frame = CGRectMake(150.0, 0.0, 300.0, 44.0);

	Nav_Group *nav_g = [self groupForSection:(int)section];
	headerLabel.text = nav_g.name;
	[customView addSubview:headerLabel];

	return customView;
}

- (CGFloat) tableView:(UITableView *)tableView heightForHeaderInSection:(NSInteger)section
{
		return 44.0;
}


/*
- (NSString *)tableView:(UITableView *)tableView titleForHeaderInSection:(NSInteger)section
{
	Nav_Group *nav_g = [self groupForSection:section];
	return nav_g.name;
}
*/

// the table's selection has changed, switch to that item's UIViewController
- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath
{
	Nav_Group *nav_g = [self groupForSection:(int)indexPath.section];
	Nav_Item *nav_i = [nav_g.items objectAtIndex: indexPath.row];

	chew_text(DEFAULT_QSP_ARG  nav_i.action, "(nav selection)" );

	if( nav_i.type == TABLE_ITEM_TYPE_PLAIN ){
		// deselect the cell...
		// We originally did this at the level of the cell -
		// That worked on the simulator, but not on the device!?
		//[nav_i.cell setSelected:NO animated:YES];

		// This works on the device!
		[ tableView deselectRowAtIndexPath:indexPath animated:YES ];
	}
}


#pragma mark -
#pragma mark UITableViewDataSource

// tell our table how many rows it will have,

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section
{
	Nav_Group *nav_g = [self groupForSection:(int)section];
	return [nav_g.items count];
}

// tell our table what kind of cell to use and its title for the given row

- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath
{
	UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:kCellIdentifier];

	if (cell == nil) {
		cell = [[UITableViewCell alloc]
			initWithStyle:UITableViewCellStyleSubtitle
			reuseIdentifier:kCellIdentifier];
		// this line produces the right-arrow.
	}

	Nav_Group *nav_g;
	nav_g = [self groupForSection:(int)indexPath.section];
	Nav_Item *nip;
	nip = [nav_g.items objectAtIndex:indexPath.row];

	nip.cell = cell;

	// Our cells have right arrows, when they push another
	// view controller (navigation), but not if they do some other action
	// without changing the screen.

	if( nip.type == TABLE_ITEM_TYPE_NAV )
		cell.accessoryType = UITableViewCellAccessoryDisclosureIndicator;
	else
		cell.accessoryType = UITableViewCellAccessoryNone;

	cell.textLabel.text = nip.name;
	cell.detailTextLabel.text = nip.explanation;
	return cell;
}

- (BOOL) shouldAutorotate
{
	return YES;
}


- (UIInterfaceOrientationMask)supportedInterfaceOrientations
{
	//return UIInterfaceOrientationMaskPortrait;
	return UIInterfaceOrientationMaskAll;
}

- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
{
	return YES;
}



/*
- (CGFloat)tableView:(UITableView *)tableView heightForRowAtIndexPath:(NSIndexPath *)indexPath
{
    return [tableView rowHeight];
}
*/

#endif // BUILD_FOR_IOS

-(void) addDoneButton:(const char *) action
{
	// ARC takes care of disposing of any old action string
	done_action = STRINGOBJ(action);
#ifdef BUILD_FOR_IOS
	// This button is labelled 'Done'
	UIBarButtonItem *item = [[UIBarButtonItem alloc]
                             initWithBarButtonSystemItem:UIBarButtonSystemItemDone
                             target:self
                             action:@selector(qtvcDoneButtonPressed)];
    
	self.navigationItem.rightBarButtonItem = item;
#endif // BUILD_FOR_IOS
}


-(void) qtvcDoneButtonPressed
{
	done_button_pushed=1;
#ifdef CAUTIOUS
	if( done_action == NULL ){
		NWARN("qtvcDoneButtonPushed:  no action!?");
		return;
	}
#endif // CAUTIOUS
    
	chew_text(DEFAULT_QSP_ARG  done_action.UTF8String, "(done button)" );
}

@end

void ios_exit_program(void)
{
#ifdef BUILD_FOR_IOS
	[quipTableViewController exitProgram];
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
	// BUG?  why should this be in this file???
	exit(0);
#endif // BUILD_FOR_MACOS
}


