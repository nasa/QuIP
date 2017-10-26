#include "quip_config.h"

#ifdef BUILD_FOR_IOS

#define DEFAULT_UI_COLOR	blackColor

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>		/* strdup */
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* free */
#endif

#include "ios_item.h"

#include "quip_prot.h"
#include "data_obj.h"
#include "dispobj.h"
#include "gui_prot.h"
#include "ios_gui.h"
#include "nav_panel.h"
#include "screen_obj.h"
// screen_obj.h defines OBJECT_GAP to be 5...
#include "viewer.h"
#include "cmaps.h"

static QUIP_ALERT_OBJ_TYPE *fatal_alert_view=NULL;
static QUIP_ALERT_OBJ_TYPE *busy_alert_p=NULL;		// the active busy alert
static QUIP_ALERT_OBJ_TYPE *suspended_busy_p=NULL;	// suspended
static QUIP_ALERT_OBJ_TYPE *final_ending_busy_p=NULL;	// set when active goes away forever

inline static void remember_alert(QUIP_ALERT_OBJ_TYPE *alert, int code )
{
	[Alert_Info rememberAlert:alert withType:code];
}

inline static void remember_normal_alert(QUIP_ALERT_OBJ_TYPE *alert)
{
	remember_alert(alert, QUIP_ALERT_NORMAL);
}

inline static void remember_busy_alert(QUIP_ALERT_OBJ_TYPE *alert)
{
	remember_alert(alert, QUIP_ALERT_BUSY);
}

inline static void remember_confirmation_alert(QUIP_ALERT_OBJ_TYPE *alert)
{
	remember_alert(alert, QUIP_ALERT_CONFIRMATION);
}

static void suspend_quip_interpreter(SINGLE_QSP_ARG_DECL)
{
	SET_QS_FLAG_BITS(THIS_QSP,QS_HALTING);
}

static void show_alert( QSP_ARG_DECL   QUIP_ALERT_OBJ_TYPE *alert_p )
{
#ifdef OLD
	[alert_p show];
#else // ! OLD

	[ root_view_controller presentViewController:alert_p animated:YES completion:^(void){
		dispatch_after(0, dispatch_get_main_queue(), ^{
			if( alert_p == busy_alert_p ){
				resume_quip(DEFAULT_QSP_ARG);
			}
		    });
	}
	];

#endif // ! OLD

	// The alert won't be shown until we relinquish control
	// back to the system...
	// So we need to inform the system not to interpret any more commands!

	suspend_quip_interpreter(SINGLE_QSP_ARG);
}


static void resume_busy(void)
{
	if( suspended_busy_p == NULL ){
		NWARN("resume_busy:  no suspended busy alert!?");
		return;
	}
	busy_alert_p=suspended_busy_p;
	suspended_busy_p=NULL;
	// Isn't this alert already remembered?
	remember_busy_alert(busy_alert_p);
	show_alert(QSP_ARG  busy_alert_p);
}

void set_allowed_orientations( Quip_Allowed_Orientations o )
{
	UIInterfaceOrientationMask m;

	switch(o){
		case QUIP_ORI_ALL:
			m=UIInterfaceOrientationMaskAll;
			break;
		case QUIP_ORI_PORTRAIT_BOTH:
			m=UIInterfaceOrientationMaskPortrait | UIInterfaceOrientationMaskPortraitUpsideDown;
			break;
		case QUIP_ORI_PORTRAIT_UP:
			m=UIInterfaceOrientationMaskPortrait;
			break;
		case QUIP_LANDSCAPE_BOTH:
			m=UIInterfaceOrientationMaskLandscape;
			break;
		case QUIP_LANDSCAPE_RIGHT:
			m=UIInterfaceOrientationMaskLandscapeRight;
			break;
		case QUIP_LANDSCAPE_LEFT:
			m=UIInterfaceOrientationMaskLandscapeLeft;
			break;
		default:
			m=UIInterfaceOrientationMaskAll;	// silence compiler warning
			// use assertion instead?
			NERROR1("set_allowed_orientations:  illegal orienation code!?");
			break;
	}
	set_supported_orientations(m);
}

#define PIXELS_PER_CHAR 9	// BUG should depend on font, may not be fixed-width!?
#define EXTRA_WIDTH	20	// pad for string length...

#define LABEL_HEIGHT	BUTTON_HEIGHT

void window_sys_init(SINGLE_QSP_ARG_DECL)
{
	// nop
}

void reposition(Screen_Obj *sop)
{
	/* Reposition frame if it has one */
	NWARN("need to implement reposition in ios.c");
}


Panel_Obj *find_panel(QSP_ARG_DECL  quipView *qv)
{
	IOS_List *lp;
	IOS_Node *np;
	Panel_Obj *po;

	lp=panel_obj_list();
	if( lp == NULL ) return(NULL);
	np=IOS_LIST_HEAD(lp);
	while( np!=NULL ){
		po = (Panel_Obj *)IOS_NODE_DATA(np);
		if( PO_QV(po) == qv ){
			return(po);
		}
		np=IOS_NODE_NEXT(np);
	}
	return(NULL);
}


/* make the panel use the given lutbuffer */

void panel_cmap(Panel_Obj *po,Data_Obj *cm_dp)
{
}

void label_panel(Panel_Obj *po, const char *s)
{
		// in X11, this redraws the string on the window title bar...
}

// Originally, a panel was just a quipView (Gen_Win).  But we'd like
// to have it be a scrollView!
//
// We call make_panel to create the device objects, after we
// have created our own database object

void make_panel(QSP_ARG_DECL  Panel_Obj *po,int w,int h)
{

	// Before we create the panel, we might first
	// check and see if there already is a viewer
	// with this name
	Gen_Win *gwp = genwin_of(QSP_ARG  PO_NAME(po) );
	if( gwp == NULL ){
		gwp = make_genwin(QSP_ARG  PO_NAME(po),w,h);
	}
	SET_PO_GW(po,gwp);
	SET_GW_PO(gwp,po);

	// We'd like panels to have activity indicators -
	// should they be created automatically or scripted?

	// We tried making a subview to hold the controls;
	// That made it easy to bring them all to the front,
	// but it made it impossible to mix buttons with
	// touches to a viewer.

	// This flag may not be what we need...
	//
	CLEAR_PANEL_FLAG_BITS(po,PANEL_SHOWN);
} /* end make_panel */

static UIView *find_first_responder(UIView *uivp)
{
sprintf(DEFAULT_ERROR_STRING,"find_first_responder:  testing 0x%lx",(long)uivp);
NADVISE(DEFAULT_ERROR_STRING);
	if( [uivp isFirstResponder] ) return(uivp);
	for (UIView *subView in uivp.subviews) {
		UIView *svp;
		svp = find_first_responder(subView);
		if( svp != NULL ) return(svp);
	}
	return NULL;
}

// Added this because next trial button in PVT app doesn't work
// after first trial - but this didn't fix it!?!?

void activate_panel(QSP_ARG_DECL  Panel_Obj *po, int yesno)
{
	// "activate" the panel by bringing it to the front
	//
	// Actually, the panel container is already in the front -
	// we need to bring all the widgets to the front instead.
	// This is a problem when we start cycling quipImages -
	// they seem to be getting in front of the widgets.
	// It would be better to have an intermediate view container
	// that holds all the images, but always stays behind the widgets,
	// so that when we bring an image to the front it doesn't hide the widgets.
	//

	IOS_List *lp;
	IOS_Node *np;

	lp = PO_CHILDREN(po);
	if( lp == NULL ){
		WARN("activate_panel:  null widget list!?");
		return;
	}
	np = IOS_LIST_HEAD(lp);

	// It would be better to have a single subview that is the parent
	// view for all the controls...
	while(np!=NULL){
		Screen_Obj *sop;

		sop = IOS_NODE_DATA(np);

		if( yesno ){
			[PO_QV(po) bringSubviewToFront:SOB_CONTROL(sop)];
		} else {
			[PO_QV(po) sendSubviewToBack:SOB_CONTROL(sop)];
		}

		np = IOS_NODE_NEXT(np);
	}
}

void check_first(Panel_Obj *po)
{
	UIView *uiv;

	uiv = find_first_responder( PO_QV(po).superview.superview );
	sprintf(DEFAULT_ERROR_STRING,"check_first:  found 0x%lx",(long)uiv);
	NADVISE(DEFAULT_ERROR_STRING);
}


/* Creates Popup Menu, popup happens in post_menu_handler */
/* In the old code, pop-ups were only supported on SGI machines... */

void make_menu(QSP_ARG_DECL  Screen_Obj *sop, Screen_Obj *mip)
{
}

void make_menu_choice(QSP_ARG_DECL  Screen_Obj *mip, Screen_Obj *parent)
{
}

void make_pullright(QSP_ARG_DECL  Screen_Obj *mip, Screen_Obj *pr,
	Screen_Obj *parent)
{
	fprintf(stderr, "menus not implemented in this version\n");
}


void make_separator(QSP_ARG_DECL  Screen_Obj *so)
{
}

void delete_widget(QSP_ARG_DECL  Screen_Obj *sop)
{
	WARN("delete_widget:  not implemented yet for IOS!?");
}

void make_button(QSP_ARG_DECL  Screen_Obj *sop)
{
	/* We need the app delegate (to set event handling),
	 * and the view (to render the button)
	 */

	int extra_width=20;	// BUG should depend on font size, button height...

//#define MY_BUTTON_TYPE	UIButtonTypeRoundedRect	// deprecated
#define MY_BUTTON_TYPE		UIButtonTypeSystem
//#define MY_BUTTON_TYPE	UIButtonTypeCustom
//#define MY_BUTTON_TYPE	UIButtonTypeInfoLight	// with info icon
//#define MY_BUTTON_TYPE	UIButtonTypeDetailDisclosure	// info icon
//#define MY_BUTTON_TYPE	UIButtonTypeContactAdd	// with plus icon

	// To make a button we can see, we need to use custom
	// and call setBackgroundImage:forState:

	UIButton *newButton = [UIButton buttonWithType:MY_BUTTON_TYPE ];
	// in iOS 7 and later, the system button is just text???

	// BUG?  should the button size depend on the text label?
	// BUG this should depend on the font size...
	int button_width = (int) strlen(SOB_NAME(sop))*PIXELS_PER_CHAR + extra_width;

	SET_SOB_WIDTH(sop,button_width);
	SET_SOB_HEIGHT(sop,BUTTON_HEIGHT);

//	newButton.frame = CGRectMake(PO_CURR_X(curr_panel), PO_CURR_Y(curr_panel),
//		SOB_WIDTH(sop), SOB_HEIGHT(sop) );

	[newButton setTitle:STRINGOBJ(SOB_NAME(sop)) forState:UIControlStateNormal];

	[newButton.titleLabel setFont:[UIFont boldSystemFontOfSize:(newButton.titleLabel.font.pointSize * 1.5)]];

	[newButton sizeToFit];	// does this work?

//fprintf(stderr,"Button '%s' has width %g and height %g\n",SOB_NAME(sop),
//newButton.frame.size.width,newButton.frame.size.height);

//	newButton.frame.origin.x = PO_CURR_X(curr_panel);
//	newButton.frame.origin.y = PO_CURR_Y(curr_panel);

	newButton.frame = CGRectMake(PO_CURR_X(curr_panel),PO_CURR_Y(curr_panel),
				   newButton.frame.size.width,newButton.frame.size.height);

	[newButton addTarget:globalAppDelegate
			action:@selector(genericButtonAction:)
			forControlEvents:UIControlEventTouchUpInside];

	SET_SOB_CONTROL(sop,newButton);
}

// the UISwitch is just the switch - no descriptive text...
// Maybe a UIText view for the label?

//#define TOGGLE_LABEL_WIDTH	130
#define SWITCH_HEIGHT		22

void make_toggle(QSP_ARG_DECL  Screen_Obj *sop)
{
	UILabel *l;
	CGRect f;
	int w;

	// How do we know how wide to make the label?
	w=(int)(PIXELS_PER_CHAR*strlen(SOB_NAME(sop))+EXTRA_WIDTH);

	f=CGRectMake(PO_CURR_X(curr_panel),PO_CURR_Y(curr_panel),w,SWITCH_HEIGHT);
	l=[[UILabel alloc] initWithFrame:f];
	l.text = STRINGOBJ( SOB_NAME(sop) );
	l.backgroundColor = [UIColor clearColor];
	l.textColor = [UIColor DEFAULT_UI_COLOR];
//	l.textAlignment = UITextAlignmentCenter;
	l.textAlignment = NSTextAlignmentCenter;

	// width and height are don't care's for switches...
	f=CGRectMake(PO_CURR_X(curr_panel)+w,PO_CURR_Y(curr_panel),0,0);
	UISwitch *newSwitch = [[UISwitch alloc ] initWithFrame: f];

//	[newSwitch setTitle:STRINGOBJ(SOB_NAME(sop)) forState:UIControlStateNormal];

	[newSwitch addTarget:globalAppDelegate
		action:@selector(genericSwitchAction:)
		forControlEvents:UIControlEventTouchUpInside];

	[ PO_QV(curr_panel) addSubview:l ];

	// The switch appears, but where is the addSubview???
	SET_SOB_CONTROL(sop,newSwitch);
}

#define RIGHT_LABEL_MARGIN	10

void make_label(QSP_ARG_DECL  Screen_Obj *sop)
{
	UILabel *l;
	CGRect f;
	int w;

	// How do we know how wide to make the label?
	// It is not a fixed-width font...
	//
	//w=PIXELS_PER_CHAR*strlen(SOB_CONTENT(sop))+EXTRA_WIDTH;
	// leave some margins...
	w=globalAppDelegate.dev_size.width-PO_CURR_X(curr_panel)-RIGHT_LABEL_MARGIN;
	f=CGRectMake(PO_CURR_X(curr_panel),PO_CURR_Y(curr_panel),w,BUTTON_HEIGHT);
	l=[[UILabel alloc] initWithFrame:f];
	l.text = STRINGOBJ( SOB_CONTENT(sop) );
	l.backgroundColor = [UIColor clearColor];

	// BUG?  we would like to center single lines,
	// But left-justify wrapped text!?
	// We could use strlen to get a crude idea of whether
	// or not there is more than one line, but a correct
	// solution must incorporate the font and size etc.

#define LABEL_CENTER_THRESHOLD	30

	if( strlen(SOB_CONTENT(sop)) < LABEL_CENTER_THRESHOLD )
		l.textAlignment = NSTextAlignmentCenter;
	else
		l.textAlignment = NSTextAlignmentLeft;

	l.numberOfLines = 0; //for word wrapping
	//l.lineBreakMode = UILineBreakModeWordWrap; for word wrapping
	[l setLineBreakMode:NSLineBreakByWordWrapping];
	// what does this do about the width?
	[l sizeToFit];

	//l.frame.size.width = w;	// override...
	f = CGRectMake(l.frame.origin.x,l.frame.origin.y,l.frame.size.width,
							l.frame.size.height);
	f.size.width = w;
	l.frame = f;

	// BUG?  should we add the height to PO_CURR_Y?

	[ PO_QV(curr_panel) addSubview:l ];
	SET_SOB_CONTROL(sop,l);
	SET_SOB_HEIGHT(sop,l.frame.size.height);
	SET_SOB_WIDTH(sop,l.frame.size.width);
}

void make_message(QSP_ARG_DECL  Screen_Obj *sop)
{
	UILabel *l;
	CGRect f;
	int w;
	char s[LLEN];

	sprintf(s,"%s:  %s",SOB_NAME(sop),SOB_CONTENT(sop));
	//w=PIXELS_PER_CHAR*strlen(s)+EXTRA_WIDTH;
	w=globalAppDelegate.dev_size.width,
	f=CGRectMake(PO_CURR_X(curr_panel),PO_CURR_Y(curr_panel),w,BUTTON_HEIGHT);
	l=[[UILabel alloc] initWithFrame:f];
	l.text = STRINGOBJ( s );
	l.backgroundColor = [UIColor clearColor];
	l.textAlignment = NSTextAlignmentCenter;

	// font?
//	l.font = [UIFont systemFontOfSize:SOB_FONT_SIZE(sop)];

	[ PO_QV(curr_panel) addSubview:l ];
	SET_SOB_CONTROL(sop,l);
}

// works for choosers AND pickers?

void reload_chooser(Screen_Obj *sop)
{
	UITableView *t=(UITableView *)SOB_CONTROL(sop);

	[t reloadData];
}

void reload_picker(Screen_Obj *sop)
{
	UIPickerView *p=(UIPickerView *)SOB_CONTROL(sop);
	[p reloadAllComponents];
}

void get_choice(Screen_Obj *sop)
{
	if( IS_CHOOSER(sop) ){
		NWARN("get_choice:  Sorry, not implemented yet for choosers...");
	} else if( SOB_TYPE(sop) == SOT_PICKER ){
		if( 0 == SOB_N_SELECTORS_AT_IDX(sop,0) ){
			assign_var(DEFAULT_QSP_ARG "choice", "no_choices_available" );
			return;
		}

		UIPickerView *p;
		int idx;
		p=(UIPickerView *)SOB_CONTROL(sop);
		idx = [p selectedRowInComponent:0];
fprintf(stderr,"get_choice:  idx = %d\n",idx);
		assert( idx>=0 && idx < SOB_N_SELECTORS_AT_IDX(sop,0) );
		assign_var(DEFAULT_QSP_ARG "choice", SOB_SELECTOR_AT_IDX(sop,0,idx) );
		return;
	} else {
		sprintf(DEFAULT_ERROR_STRING,
"get_choice:  object %s is neither a picker nor a chooser!?",SOB_NAME(sop));
		NWARN(DEFAULT_ERROR_STRING);
	}
	assign_var(DEFAULT_QSP_ARG "choice", "oops" );
}

void set_choice(Screen_Obj *sop, int which)
{
	if( IS_CHOOSER(sop) ){
		UITableView *t;
		NSIndexPath *path;
		NSUInteger idxs[2];

		t=(UITableView *)SOB_CONTROL(sop);

		idxs[0]=0;	// section
		idxs[1]=which;
		path = [[ NSIndexPath alloc ] initWithIndexes:idxs length:2 ];

		[ t	selectRowAtIndexPath:path
			animated:YES
			scrollPosition:UITableViewScrollPositionMiddle ];
	} else if( SOB_TYPE(sop) == SOT_PICKER ){
		UIPickerView *p;
		p=(UIPickerView *)SOB_CONTROL(sop);
		[p selectRow:which inComponent:0 animated:NO];
	} else {
		sprintf(DEFAULT_ERROR_STRING,
"set_choice:  object %s is neither a picker nor a chooser!?",SOB_NAME(sop));
		NWARN(DEFAULT_ERROR_STRING);
	}
}

void set_pick(Screen_Obj *sop, int cyl, int which)
{
	if( SOB_TYPE(sop) == SOT_PICKER ){
		UIPickerView *p;
		p=(UIPickerView *)SOB_CONTROL(sop);

		if( cyl < 0 || cyl >= SOB_N_CYLINDERS(sop) ){
			sprintf(DEFAULT_ERROR_STRING,
	"set_pick:  cylinder %d is out of range for picker %s (0-%d)",
				cyl,SOB_NAME(sop),SOB_N_CYLINDERS(sop));
			NWARN(DEFAULT_ERROR_STRING);
			return;
		}

		[p selectRow:which inComponent:cyl animated:NO];
	} else {
		sprintf(DEFAULT_ERROR_STRING,
"set_pick:  object %s is not a chooser!?",SOB_NAME(sop));
		NWARN(DEFAULT_ERROR_STRING);
	}
}

// This is the new experimental make_chooser that uses a UITableView...

void make_chooser(QSP_ARG_DECL  Screen_Obj *sop, int n, const char **stringlist)
{
	UITableView *t;
	CGRect f;

	f=CGRectMake(PO_CURR_X(curr_panel),PO_CURR_Y(curr_panel),
		PO_WIDTH(curr_panel)-2*OBJECT_GAP,
		200);
	t=[[UITableView alloc] initWithFrame:f style:UITableViewStyleGrouped ];

	t.delegate = globalAppDelegate;
	t.dataSource = globalAppDelegate;
	t.backgroundColor = [UIColor clearColor];
	t.opaque = NO;

	if (SOB_TYPE(sop) == SOT_MLT_CHOOSER) {
		t.allowsMultipleSelection = YES;
	}

	SET_SOB_CONTROL(sop,t);	// has to come before layoutIfNeeded?

	[t layoutIfNeeded];
	CGFloat tHeight = [t contentSize].height;
	CGRect fitF = CGRectMake(f.origin.x, f.origin.y, f.size.width, tHeight);
	[t setFrame:fitF];

	// decide in the script whether or not a larger-than-device
	// page is necessary - depends on device size and presence of other
	// widgets...

// The background view is not null...
//fprintf(stderr,"background view = 0x%lx\n",(long)(t.backgroundView));
	//t.backgroundView = [[UIView alloc] initWithFrame:

	// default should be nil = clear
	//t.backgroundView.backgroundColor = [UIColor clearColor];
	t.backgroundView.opaque = NO;

	t.backgroundView = NULL;

	SET_SOB_WIDTH(sop,(int) f.size.width );
	SET_SOB_HEIGHT(sop,(int) tHeight );
}

void make_picker(QSP_ARG_DECL  Screen_Obj *sop )
{
	UIPickerView *p;
	CGRect f;

	// most widgets have a small gap from the left hand edge (10 pixels?)
	// but ios UIPickerView's take up the full screen width on the iPhone...
	// BUG make left hand gap device-type-dependent
	//
	// Update:  on iOS 9, UIPickerView is resizable and adaptive...
	// Now defaults to a width of 320 points on all devices...
#define MY_DEFAULT_PICKER_HEIGHT	200
#define GAP_FACTOR	6

	//f=CGRectMake(PO_CURR_X(curr_panel),PO_CURR_Y(curr_panel),PO_WIDTH(curr_panel)-2*OBJECT_GAP,0);
	f=CGRectMake(	PO_CURR_X(curr_panel)+GAP_FACTOR*OBJECT_GAP,
			PO_CURR_Y(curr_panel),
			PO_WIDTH(curr_panel)-2*GAP_FACTOR*OBJECT_GAP,
			MY_DEFAULT_PICKER_HEIGHT );
	p=[[UIPickerView alloc] initWithFrame:f];

	p.delegate = globalAppDelegate;
	p.dataSource = globalAppDelegate;
	p.backgroundColor = [UIColor clearColor];

	p.showsSelectionIndicator = YES;

	SET_SOB_CONTROL(sop,p);
	SET_SOB_HEIGHT(sop,MY_DEFAULT_PICKER_HEIGHT);

	// BUG?  set width and height of sop here?
}

// Originally, the action string is called when the Done key is entered,
// which can be confusing...  it would be better to have a way to pull the text
// from the widget, or call the action every time that a keystroke is typed...

void make_text_field(QSP_ARG_DECL  Screen_Obj *sop)
{
	// BUG - make sure so_type is text field or password

	// BUG should set width based on width of device!
	// We should be able to get that from the View...

	UITextField *textField = [[UITextField alloc]
		initWithFrame:CGRectMake(PO_CURR_X(curr_panel), PO_CURR_Y(curr_panel),
		SOB_WIDTH(sop), SOB_HEIGHT(sop) ) ];
	textField.borderStyle = UITextBorderStyleRoundedRect;
	textField.font = [UIFont systemFontOfSize:SOB_FONT_SIZE(sop)];

	// Setting the default text causes a callback,
	// so we postpone it until we have added this widget to the panel

	textField.autocorrectionType = UITextAutocorrectionTypeNo;
	textField.autocapitalizationType = UITextAutocapitalizationTypeNone;
	textField.keyboardType = UIKeyboardTypeDefault;
	textField.returnKeyType = UIReturnKeyDone;
	textField.clearButtonMode = UITextFieldViewModeWhileEditing;
	textField.clearsOnBeginEditing = YES;
	textField.contentVerticalAlignment = UIControlContentVerticalAlignmentCenter;
	textField.delegate = globalAppDelegate;	// what methods are delegated???

	if( SOB_TYPE(sop) == SOT_PASSWORD )
		textField.secureTextEntry = YES;

	SET_SOB_CONTROL(sop,textField);
}

void install_initial_text(Screen_Obj *sop)
{
	UITextField *textField;

	// BUG make sure sop is right kind of widget
	textField = (UITextField *)SOB_CONTROL(sop);
	textField.placeholder = STRINGOBJ( SOB_CONTENT(sop) );
}

void make_text_box(QSP_ARG_DECL  Screen_Obj *sop, BOOL is_editable )
{
	CGRect r;
	quipView *qv;
	UITextView *msg_view;

	qv = PO_QV(curr_panel);

	// BUG should set width based on width of device!
	// We should be able to get that from the View...
	// We should set the font size based on which type of device, and what
	// screen resolution we have...

	r=CGRectMake(PO_CURR_X(curr_panel), PO_CURR_Y(curr_panel),
		SOB_WIDTH(sop), SOB_HEIGHT(sop) );

	msg_view = [[UITextView alloc] initWithFrame:r];
	msg_view.font = [UIFont systemFontOfSize:SOB_FONT_SIZE(sop)];
	//msg_view.borderStyle = UITextBorderStyleRoundedRect;

	//msg_view.text = @"system output:\n";	// appropriate for console, but not general BUG
	msg_view.text = STRINGOBJ(SOB_CONTENT(sop));

	msg_view.editable=is_editable;

//	textField.contentVerticalAlignment = UIControlContentVerticalAlignmentCenter;

	msg_view.delegate = globalAppDelegate;

	// Don't we need to set SOB_CONTROL???

	[qv addSubview:msg_view];

	SET_SOB_CONTROL(sop,msg_view);
}

void make_act_ind(QSP_ARG_DECL  Screen_Obj *sop)
{
	quipView *qv;
	UIActivityIndicatorView *ai;

	qv = PO_QV(curr_panel);

	// BUG should set width based on width of device!
	// We should be able to get that from the View...
	// We should set the font size based on which type of device, and what
	// screen resolution we have...

	ai = [[UIActivityIndicatorView alloc]
		initWithActivityIndicatorStyle:UIActivityIndicatorViewStyleWhite];

	//[ai setCenter:CGPointMake(PO_WIDTH(curr_panel)/2.0f,
	//			PO_HEIGHT(curr_panel)/2.0f)];
	[ai setCenter:CGPointMake(PO_CURR_X(curr_panel)+10.0f,
				PO_CURR_Y(curr_panel)+10.0f)];

	// Don't we need to set SOB_CONTROL???

	[qv addSubview:ai];

	SET_SOB_CONTROL(sop,ai);
}

void set_activity_indicator(Screen_Obj *sop, int on_off )
{
	UIActivityIndicatorView *ai;

	ai = (UIActivityIndicatorView *)SOB_CONTROL(sop);

//int state = [ai isAnimating];
//fprintf(stderr,"set_activitiy_indicator:  previous state = %d\n",state);

	if( on_off ){
		Panel_Obj *po;
		po = SOB_PANEL(sop);
		[PO_QV(po) bringSubviewToFront:ai];
		ai.hidden = NO;	// should not be necessary?
		[ai startAnimating];
	}
	else
		[ai stopAnimating];
}

// BUG this is a hack tuned to the EZJet app on iPod 5
#define EDIT_BOX_FRACTION	0.7

void make_edit_box(QSP_ARG_DECL  Screen_Obj *sop)
{
	/* put this after addition to panel list, because setting initial value causes a callback */
	get_device_dims(sop);

	// BUG - we reduce the height from the requested height
	// to make sure there is room for the keyboard -
	// but this is the wrong computation!?

	SET_SOB_HEIGHT(sop,(int)(EDIT_BOX_FRACTION*(float)SOB_HEIGHT(sop)));

	make_text_box(DEFAULT_QSP_ARG  sop, YES );

	//msg_view = (UITextView *)SOB_CONTROL(sop);	// remember this for later
}

/* For a text widget, this sets the text!? */

void update_prompt(Screen_Obj *sop)
{
	NWARN("update_prompt not implemented!?");

}

// It is not clear that we need to have this separate!?

void update_edit_text(Screen_Obj *sop, const char *string)
{
	update_text_field(sop, string);
}

const char *get_text(Screen_Obj *sop)
{
	return NULL;
}

void make_gauge(QSP_ARG_DECL  Screen_Obj *sop)
{
	WARN("make_gauge not implemented!?");

}

/* Updates the label of a gauge or slider */

void set_gauge_label(Screen_Obj *gp, const char *str )
{
	UILabel *l;

	// BUG make sure gp is really a gauge.
	// but this should be done before calling?

	l=(UILabel *) SOB_LABEL(gp);
	l.text = STRINGOBJ( str );
}

void update_gauge_label(Screen_Obj *gp )
{
	UILabel *l;
	char valstr[64];

	// BUG make sure gp is really a gauge.
	// but this should be done before calling?

	l=(UILabel *) SOB_LABEL(gp);
	sprintf(valstr,":  %d",SOB_VAL(gp));
	l.text = [STRINGOBJ( SOB_NAME(gp) ) stringByAppendingString: STRINGOBJ(valstr) ];
}

/* This should leave enough room to display the anchor values.
 * But where can we display the current value?
 */

#define SLIDER_SIDE_GAP	55.0
#define SLIDER_DELTA_Y	40.0	/* just the slider itself */
#define SLIDER_TEXT_DY	20.0	// should depend on font size!

void make_slider(QSP_ARG_DECL  Screen_Obj *sop)
{
	int slider_width;
	int slider_height=SLIDER_DELTA_Y;

	/* 320 is the width of the iPod display? */

	/* BUG make sure that display
	 * is at least that wide...
	 */
	slider_width = SOB_WIDTH(sop)-2*SLIDER_SIDE_GAP;

	UISlider *slider = [[UISlider alloc] initWithFrame:
		CGRectMake(
			PO_CURR_X(curr_panel) + SLIDER_SIDE_GAP,
			PO_CURR_Y(curr_panel),
			slider_width,
			SLIDER_DELTA_Y
		)
	];

	slider.minimumValue = (float) SOB_MIN(sop);
	slider.maximumValue = (float) SOB_MAX(sop);
	slider.value = (float) SOB_VAL(sop);

	if( SOB_TYPE(sop) == SOT_SLIDER ){
		[slider
			addTarget:globalAppDelegate
			action:@selector(genericSliderAction:)
			forControlEvents:
				UIControlEventTouchUpInside
		];
	} else if( SOB_TYPE(sop) == SOT_ADJUSTER ){
		[slider
			addTarget:globalAppDelegate
			action:@selector(genericSliderAction:)
			forControlEvents: (
				UIControlEventTouchUpInside |
				UIControlEventTouchDragInside
			)
		];
	}
#ifdef CAUTIOUS
	  else {
		WARN("CAUTIOUS:  make_slider:  unrecognized object type!?");
		return;
	}
#endif /* CAUTIOUS */

	SET_SOB_CONTROL(sop,slider);
	// where do we add the subview???

	/* Now we want to add views for the limit values and the legend... */
	UILabel *l;
	CGRect f;
	char numstr[32];

	f=CGRectMake(
		PO_CURR_X(curr_panel)+SLIDER_SIDE_GAP,
		PO_CURR_Y(curr_panel)+SLIDER_DELTA_Y /* +SLIDER_TEXT_DY */,
		slider_width,
		LABEL_HEIGHT
	);
	slider_height += LABEL_HEIGHT;
	l=[[UILabel alloc] initWithFrame:f];
	SET_SOB_LABEL(sop,l);
	update_gauge_label(sop);
	l.backgroundColor = [UIColor clearColor];
	l.textColor = [UIColor DEFAULT_UI_COLOR];
	l.textAlignment = NSTextAlignmentCenter;

	// We don't need to use add_to_panel, because these
	// labels never generate a callback...

	quipView *qv=PO_QV(curr_panel);
	[qv addSubview:l];

	// Now do the side legends...

	f=CGRectMake(
		PO_CURR_X(curr_panel),
		PO_CURR_Y(curr_panel),
		SLIDER_SIDE_GAP,
		SLIDER_DELTA_Y
	);
	l=[[UILabel alloc] initWithFrame:f];
	l.backgroundColor = [UIColor clearColor];
	l.textColor = [UIColor DEFAULT_UI_COLOR];
	l.textAlignment = NSTextAlignmentCenter;
	sprintf(numstr,"%d",SOB_MIN(sop));
	l.text = STRINGOBJ( numstr );
	[qv addSubview:l];

	f=CGRectMake(
		PO_CURR_X(curr_panel)+SLIDER_SIDE_GAP+slider_width,
		PO_CURR_Y(curr_panel),
		SLIDER_SIDE_GAP,
		SLIDER_DELTA_Y
	);
	l=[[UILabel alloc] initWithFrame:f];
	l.backgroundColor = [UIColor clearColor];
	l.textColor = [UIColor DEFAULT_UI_COLOR];
	l.textAlignment = NSTextAlignmentCenter;
	sprintf(numstr,"%d",SOB_MAX(sop));
	l.text = STRINGOBJ( numstr );
	[qv addSubview:l];

	SET_SOB_HEIGHT(sop,slider_height);
}

void make_adjuster(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef FOOBAR
	UISlider *slider = [[UISlider alloc] initWithFrame:
		CGRectMake(
			PO_CURR_X(curr_panel) + 55.0,
			PO_CURR_Y(curr_panel),
			300-2*55.0,
			40.0
		)
	];

	[slider addTarget:globalAppDelegate
		action:@selector(genericSliderAction:)
		forControlEvents:
			(UIControlEventTouchUpInside|
			 UIControlEventTouchDragInside)];
	SET_SOB_CONTROL(sop,slider);
#endif // FOOBAR
	make_slider( QSP_ARG   sop );
}

void new_slider_range(Screen_Obj *sop, int xmin, int xmax)
{
	NWARN("new_slider_range not implemented!?");
}


void new_slider_pos(Screen_Obj *sop, int val)
{
	NWARN("new_slider_pos not implemented!?");
}

void set_toggle_state(Screen_Obj *sop, int val)
{
	UISwitch *sw;

	// BUG check for correct widget type here
	sw = (UISwitch *) SOB_CONTROL(sop);

	// BUG maybe nice to animate sometimes - how to determine when?
	// Only if panel is visible?
	[ sw setOn:(val?YES:NO) animated:NO ];
}

void set_gauge_value(Screen_Obj *gp, int n)
{
	UISlider *s;

	// BUG make sure gp is really a gauge

	s=(UISlider *) SOB_CONTROL(gp);
	s.value = (float) n;

	SET_SOB_VAL(gp,n);
	update_gauge_label(gp);
}

void update_message(Screen_Obj *sop)
{
	UILabel *l;
	char s[LLEN];

	sprintf(s,"%s:  %s",SOB_NAME(sop),SOB_CONTENT(sop));
	// BUG check widget type
	l = (UILabel *) SOB_CONTROL(sop);
	l.text = STRINGOBJ( s );
	//[l sizeToFit];
	[l setNeedsDisplay];
}

void update_label(Screen_Obj *sop)
{
	UILabel *l;

	// BUG check widget type
	l = (UILabel *) SOB_CONTROL(sop);
	l.text = STRINGOBJ( SOB_CONTENT(sop) );
	//[l sizeToFit];
//fprintf(stderr,"update_label #2:  label is %f x %f\n",
//l.frame.size.width,l.frame.size.height);
	[l setNeedsDisplay];
}

// text to append is passed in SOB_CONTENT

void update_text_box(Screen_Obj *sop)
{
	UITextView *tb;

	tb = (UITextView *) SOB_CONTROL(sop);
	tb.text = [tb.text stringByAppendingString: STRINGOBJ( SOB_CONTENT(sop) ) ];
}

// what kind of object can this be?
// UILabel, UITextField...

void update_text_field(Screen_Obj *sop, const char *string)
{
	switch( SOB_TYPE(sop) ){

		case SOT_MESSAGE:
		{
			UILabel *l;

			l = (UILabel *) SOB_CONTROL(sop);
			l.text = STRINGOBJ( string );
		}
			break;

		case SOT_TEXT:
		case SOT_PASSWORD:
		case SOT_TEXT_BOX:
		case SOT_EDIT_BOX:
		{
			UITextView *tb;

			tb = (UITextView *) SOB_CONTROL(sop);
			tb.text = STRINGOBJ( string );
		}
			break;

		default:
			sprintf(DEFAULT_ERROR_STRING,
	"update_text_field:  unhandled object type, screen object %s",SOB_NAME(sop));
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
}

#ifdef NOT_USED
static int panel_mapped(Panel_Obj *po)
{

	return(1);
}
#endif // NOT_USED

// Orinally we had a big stack of views, and rotated them.
// But now we have separate controllers and use navigation...

void show_panel(QSP_ARG_DECL  Panel_Obj *po)
{
	push_nav(QSP_ARG  PO_GW(po));
} /* end show_panel */

void unshow_panel(QSP_ARG_DECL  Panel_Obj *po)
{
	pop_nav(QSP_ARG 1);
}

void posn_panel(Panel_Obj *po)
{
	NWARN("posn_panel not implemented!?");
}

void free_wsys_stuff(Panel_Obj *po)
{
	/* BUG - should check what we need to delete here */
}

void give_notice(const char **msg_array)
{
	NWARN("give_notice:  notices not implemented in this version\n");
}

void set_std_cursor(void)
{
	NWARN("set_std_cursor not implemented!?");
}

void set_busy_cursor(void)
{
	NWARN("set_busy_cursor not implemented!?");
}

void make_scroller(QSP_ARG_DECL  Screen_Obj *sop)
{
	WARN("make_scroller not implemented!?");

}

void set_scroller_list(Screen_Obj *sop, const char *string_list[],
	int nlist)
{
	NWARN("set_scroller_list not implemented!?");
}

void window_cm(Panel_Obj *po,Data_Obj *cm_dp)
{
	if( CANNOT_SHOW_PANEL(po) ) return;

//	install_colors(&po->po_top);
}

#ifdef NOT_USED

static void dump_scrnobj_list(IOS_List *lp)
{
	IOS_Node *np;

	np = IOS_LIST_HEAD(lp);
	while(np!=NULL){
		Screen_Obj *sop;
		sop = (Screen_Obj *)IOS_NODE_DATA(np);
		fprintf(stderr,"\t%s\n",SOB_NAME(sop));
		np=IOS_NODE_NEXT(np);
	}
}

#endif // NOT_USED

/* The screen objects live in per-panel contexts.  When we "show" a panel,
 * it's context gets pushed...  However, if a panel appears because the
 * views in front of it have been sent to the back, then we don't
 * necessarily know this (as we do not explicitly keep track of the
 * ordering ourselves).  Therefore, if we fail to locate a control,
 * we should try to identify the view that the control came from,
 * and find the associated panel...
 */

/* In principle, we only need to search the top context?
 * But in iOS 8, we are getting callbacks to screens that
 * have already been dismissed!?  Hence find_any_scrnobj...
 */

Screen_Obj *find_scrnobj(UIView *cp)
{
	IOS_Item_Context *icp;

	icp = top_scrnobj_context();
	assert( icp != NULL );

//fprintf(stderr,"find_scrnobj:  searching context %s for control 0x%lx\n",
//IOS_CTX_NAME(icp),(u_long)cp);

	IOS_List *lp = [icp getListOfItems];
	assert( lp != NULL );

	IOS_Node *np;
	np=IOS_LIST_HEAD(lp);
	while(np!=NULL){
		Screen_Obj *sop = (Screen_Obj *)IOS_NODE_DATA(np);

//fprintf(stderr,"find_scrnobj:  checking %s, control = 0x%lx, target = 0x%lx\n",
//SOB_NAME(sop),(u_long)SOB_CONTROL(sop),(u_long)cp);
		if( SOB_CONTROL(sop) == cp ){
//fprintf(stderr,"find_scrnobj:  FOUND %s\n",SOB_NAME(sop));
			return sop;
		}
		np=IOS_NODE_NEXT(np);
	}

	/* If we haven't found it, then it must mean that we have a context error... */
//fprintf(stderr,"find_scrnobj:  FOUND NOTHING!?\n");
	return NULL;
}

// find_any_scrnobj should search all contexts...

Screen_Obj *find_any_scrnobj(UIView *cp)
{
	IOS_List *lp;
	IOS_Node *np;
	Screen_Obj *sop;

	// Does this return all objects, or just current context stack?
	lp = all_scrnobjs(SGL_DEFAULT_QSP_ARG);
	if( lp == NULL ) return NULL;

	np=IOS_LIST_HEAD(lp);
	while(np!=NULL){
		sop = (Screen_Obj *)IOS_NODE_DATA(np);
//fprintf(stderr,"find_any_scrnobj 0x%lx:  %s, ctrl = 0x%lx\n",
//(u_long)cp,SOB_NAME(sop),(u_long) SOB_CONTROL(sop));
		if( SOB_CONTROL(sop) == cp ) return sop;
		np=IOS_NODE_NEXT(np);
	}

	return NULL;
}

// We have been experimenting with using other threads...
// UI stuff has to run on main thread...

// This comment doesn't seem to be true now:
//
// push_nav works fine for pushing viewers and panels,
// but it doesn't push another navigation panel...
// Is that because the view controller is wrong???
//
//

int n_pushed_panels(void)
{
	NSArray *a = [ root_view_controller viewControllers];
	return (int) a.count;
}

// If we call push_nav after pop_nav, the pop_nav may not have taken effect yet!?

void push_nav(QSP_ARG_DECL  Gen_Win *gwp)
{
	// Make sure that this has not been pushed already by scanning the stack

	NSArray *a = [ root_view_controller viewControllers];
//fprintf(stderr,"push_nav:  root view controller has %lu view controllers\n",(unsigned long)a.count);
	int i;
//fprintf(stderr,"push_nav %s BEGIN\n",GW_NAME(gwp));
	for(i=0;i<a.count;i++){
		UIViewController *vc;
		vc = [a objectAtIndex:i];
		if( vc == GW_VC(gwp) ){
			sprintf(DEFAULT_ERROR_STRING,
	"push_nav %s:  panel already pushed!?",GW_NAME(gwp));
			advise(DEFAULT_ERROR_STRING);
			return;
		}
	}

	// The current view controller should be a table controller...

	// Is this next comment still valid?  I don't think so...
	//
	// The first time we call this, the nav bar changes, but the main
	// panel does not - why not?  (New bug w/ ios 6)
	// Maybe viewDidLoad has not been called?

	// The next section is a total hack, to insure that the back
	// button doesn't display when we don't want it to.
	// It is possible that there is a correct, clean way
	// to make this work properly.  Alternatively, there may
	// be a bug in iOS which makes this necessary?

	// The view controller is a property of the gen_win -
	// how many different kinds can we have?

	switch( GW_VC_TYPE(gwp) ){
		case GW_VC_QVC:
		{
			quipViewController *qvc;
			qvc = (quipViewController *)GW_VC(gwp);
			if( qvc.qvc_flags & QVC_HIDE_BACK_BUTTON ){
				// We push and pop - why does this work?
				[ root_view_controller
					pushViewController:GW_VC(gwp)
					animated:NO];
[ root_view_controller.navigationBar.topItem setHidesBackButton:YES animated:NO];
				[ root_view_controller
					popViewControllerAnimated:NO ];
			}
			if( qvc.qvc_done_action != NULL ){

			// BUG - we shouldn't have to do allocate
			// this item every time???
	UIBarButtonItem *item = [[UIBarButtonItem alloc]
			initWithBarButtonSystemItem:UIBarButtonSystemItemDone
			target:qvc
			action:@selector(qvcDoneButtonPressed)];

	// BUG?  do we have to remove this also
	root_view_controller.navigationItem.rightBarButtonItem = item;

			}
		}
			break;

		case GW_VC_QTVC:
		{
			//quipTableViewController *qtvc;
			//qtvc = (quipTableViewController *)GW_VC(gwp);
			// do nothing
		}
			break;
		default:
fprintf(stderr,"Genwin %s has an unknown view controller type!?\n",GW_NAME(gwp));
			break;
	}

	// We may be pushing a quipViewController...
	// but maybe not?
	[ (quipViewController *)GW_VC(gwp) clearBlockedAutorotation];

	// What kind of controller is the root view controller?
	// Ans:  quipNavController

//fprintf(stderr,"push_nav %s:  vc = 0x%lx\n",GW_NAME(gwp),(long) GW_VC(gwp));

	// If we want to hide the back button in the nav bar, we
	// should try to find the viewDidLoad routine...
	// BUT viewDidLoad has already been called, and at that
	// time there is no associated nav controller???
	//
	// perhaps the solution to this is the UINavigationControllerDelegate
	// methods ???

	[ root_view_controller
		pushViewController:GW_VC(gwp) animated:YES];

	// push the screen object context too, so that we will
	// be able to look up widgets in other places,
	// we push a PO_CONTEXT, not a GW_CONTEXT!?

	// The context is associated with a genwin,
	// when our delegate is called, it is passed a view
	// controller - how do we look up the view controller
	// to get back the context?  The common link seems
	// to be the genwin...

	// We used to call push_scrnobj_context at this point, but
	// we handle this instead in the nav controller delegate,
	// which works correctly when we use the nav bar back button
	// to pop...
//	push_scrnobj_context(DEFAULT_QSP_ARG  GW_CONTEXT(gwp));

	SET_GW_FLAG_BITS(gwp,GENWIN_SHOWN);
} // end push_nav

// pop_nav pops the panel widget context, but this won't happen
// if the controller is popped using the nav bar!?
// We haven't seen problems from this yet, it would probably
// take a name overloading to reveal it...  BUG
// we should be able to fix this using UINavigationControllerDelegate...
//
// The name overloading comment above may have been true
// when find_scrnobj searched the entire context stack for
// widgets, but now we only search the top context, which makes
// it important that it be correct.

void pop_nav(QSP_ARG_DECL int n_levels)
{
	//quipViewController *qvc;
	quipViewController *old_vc;	// could also be quipTableViewController!?

	// If the current view controller has locked the orientation,
	// the device orientation may have changed without generating
	// an autorotation; in that case, the new view controller
	// revealed by the pop might need to rotate, but unless
	// we do something this won't happen until a new orientation
	// change event is generated by the accelerometer...

//fprintf(stderr,"pop_nav %d:  before popping, root view controller has %lu view controllers\n",n_levels,(unsigned long)a.count);

//fprintf(stderr,"pop_nav %d BEGIN\n",n_levels);
	old_vc = (quipViewController *) root_view_controller.topViewController;

	if( n_levels > 1 ){
	NSArray *a = [ root_view_controller viewControllers];
	if( n_levels >= a.count ){
			WARN("pop_nav:  too many levels requested!?");
			n_levels = (int)(a.count - 1);
		}
		UIViewController *target_vc;
		target_vc = a[a.count-n_levels-1];
		[ root_view_controller popToViewController:target_vc animated:YES ];
	} else {
	/* qvc = (quipViewController *) */
		[ root_view_controller popViewControllerAnimated:YES
			checkOrientation:[old_vc didBlockAutorotation]
			];
	}

	// The new view controller may not be a quipViewController!?
	// It could also be a quipTableViewController???

	// pop_scrnobj_context now done in quipNavController delegate
}

void make_console_panel(QSP_ARG_DECL  const char *name)
{
	Panel_Obj *po = new_panel(QSP_ARG  name,	/* make_console_panel */
			globalAppDelegate.dev_size.width,
			globalAppDelegate.dev_size.height);

	// should be done from main thread???

	[((quipViewController *)PO_VC(po))	addConsoleWithSize:globalAppDelegate.dev_size];
}

#define FATAL_ERROR_TYPE_STR	"FATAL ERROR"

struct alert_data {
	const char *type;
	const char *msg;
};

static struct alert_data deferred_alert = { NULL, NULL };

static void clear_deferred_alert(void)
{
	assert(deferred_alert.type != NULL);
	assert(deferred_alert.msg != NULL);

	rls_str(deferred_alert.type);
	rls_str(deferred_alert.msg);

	deferred_alert.type=NULL;
	deferred_alert.msg=NULL;
}

// We call suspend__busy when the busy alert is up, and we want to
// display a different alert.  we are probably calling this from
// finish_digestion...

static void suspend__busy(void)
{
	// we need to display an alert while we are busy...
	suspended_busy_p = busy_alert_p;
	end_busy(0);	// takes the alert down
}

static void defer_alert(const char *type, const char *msg)
{
	if( deferred_alert.type != NULL ){
		fprintf(stderr,"MORE THAN ONE DEFERRED ALERT!?\n");
		fprintf(stderr,"Discarding deferred alert \"%s %s\"\n",
			deferred_alert.type,deferred_alert.msg);
		clear_deferred_alert();
		defer_alert("TOO MANY DEFERRED ALERTS!?",msg);
	} else {
		assert(deferred_alert.msg==NULL);
		deferred_alert.type = savestr(type);
		deferred_alert.msg = savestr(msg);
	}
}

static void busy_dismissal_checks(QUIP_ALERT_OBJ_TYPE *a)
{
	if( a == final_ending_busy_p ){	// dismissed busy alert
		// nothing more to do.
		final_ending_busy_p=NULL;
	}
	resume_quip(SGL_DEFAULT_QSP_ARG);
}

// Call this when dismissing a non-busy alert

static void alert_dismissal_busy_checks(Alert_Info *aip)
{
	assert(aip.the_alert_p != final_ending_busy_p);
	assert( busy_alert_p == NULL );

	if( suspended_busy_p != NULL ){
		assert( aip.the_alert_p != suspended_busy_p );
		resume_busy();
	} else {
		resume_quip(DEFAULT_QSP_ARG);
	}
} // alert_dismissal_busy_check

static void quip_alert_dismissal_actions(QUIP_ALERT_OBJ_TYPE *alertView, NSInteger buttonIndex)
{
	if( alertView == fatal_alert_view ){
		[first_quip_controller qtvcExitProgram];
	}

	Alert_Info *aip;

	aip = [Alert_Info alertInfoFor:alertView];

	if( IS_VALID_ALERT(aip) ){
		[aip forget];	// removes from list

		alert_dismissal_busy_checks(aip);
	}
#ifdef CAUTIOUS
	 else {
		sprintf(DEFAULT_ERROR_STRING,
"CAUTIOUS:  quip_alert_dismissal_actions:  Unrecognized alert type %d!?",aip.type);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
#endif // CAUTIOUS
}

static void confirmation_alert_dismissal_actions(QUIP_ALERT_OBJ_TYPE *alertView, NSInteger buttonIndex)
{
	if( buttonIndex == 0 ){
		assign_reserved_var(DEFAULT_QSP_ARG  "confirmed","0");
	} else {
		assign_reserved_var(DEFAULT_QSP_ARG  "confirmed","1");
	}
	quip_alert_dismissal_actions(alertView,buttonIndex);
}

static QUIP_ALERT_OBJ_TYPE *create_alert_with_no_buttons(const char *type, const char *msg)
{
	QUIP_ALERT_OBJ_TYPE *alert;
#ifdef OLD
	// open an alert with a single dismiss button
	// How can we changed the behavior of the delegate???
	alert = [[QUIP_ALERT_OBJ_TYPE alloc]
		initWithTitle:STRINGOBJ(type)
		message:STRINGOBJ(msg)
		delegate:vc
		cancelButtonTitle: @"Please be patient..."
		otherButtonTitles: nil];
#else // !OLD
	alert = [UIAlertController
				alertControllerWithTitle: STRINGOBJ(type)
				message: STRINGOBJ(msg)
				preferredStyle:UIAlertControllerStyleAlert];

#endif // !OLD

	return alert;
}

static QUIP_ALERT_OBJ_TYPE *create_alert_with_one_button(const char *type, const char *msg)
{
#ifdef OLD
	// open an alert with a single dismiss button
	QUIP_ALERT_OBJ_TYPE *alert = [[QUIP_ALERT_OBJ_TYPE alloc]
		initWithTitle:STRINGOBJ(type)
		message:STRINGOBJ(msg)
		delegate:vc
		cancelButtonTitle: is_fatal ?  @"Exit program" : @"OK"
		otherButtonTitles: nil];
	fatal_alert_view= is_fatal ? alert : NULL;
	//[alert show];
#else // ! OLD
	UIAlertController * alert=   [UIAlertController
		 alertControllerWithTitle: STRINGOBJ(type)
		 message: STRINGOBJ(msg)
		 preferredStyle:UIAlertControllerStyleAlert];

	UIAlertAction* ok = [UIAlertAction
		actionWithTitle:@"OK"
		style:UIAlertActionStyleDefault
		handler:^(UIAlertAction * action)	/* block */
		{
			// is the handler called before or after the alert
			// is dismissed???
				quip_alert_dismissal_actions(alert,0);
			}];

	[alert addAction:ok];
#endif // ! OLD

	return alert;
}

static QUIP_ALERT_OBJ_TYPE *create_alert_with_two_buttons(const char *type, const char *msg)
{
	QUIP_ALERT_OBJ_TYPE *alert;

#ifdef OLD
	// open an alert with a single dismiss button
	alert = [[QUIP_ALERT_OBJ_TYPE alloc]
		initWithTitle:STRINGOBJ(title)
		message:STRINGOBJ(question)
		delegate:vc
		cancelButtonTitle: @"Cancel"
		otherButtonTitles: @"Proceed", nil ];
#else // ! OLD
	alert = [UIAlertController
		 alertControllerWithTitle: STRINGOBJ(type)
		 message: STRINGOBJ(msg)
		 preferredStyle:UIAlertControllerStyleAlert];

// BUG need to copy code from the delegate into the actions!
	UIAlertAction* ok = [UIAlertAction
		actionWithTitle:@"Proceed"
		style:UIAlertActionStyleDefault
		handler:^(UIAlertAction * action) {
			[alert dismissViewControllerAnimated:YES completion:nil];
				confirmation_alert_dismissal_actions(alert,1);
		}
		];
	UIAlertAction* cancel = [UIAlertAction
		actionWithTitle:@"Cancel"
		style:UIAlertActionStyleDefault
		handler:^(UIAlertAction * action) {
				confirmation_alert_dismissal_actions(alert,0);
		}
		];

	[alert addAction:cancel];
	[alert addAction:ok];
#endif // ! OLD

	return alert;
}

static void present_generic_alert(QSP_ARG_DECL  const char *type, const char *msg)
{
	UIViewController *vc;
	int is_fatal;

	vc = root_view_controller.topViewController;

	// vc can be null if we call this before we have set
	// root_view_controller (during startup)

	if( vc == NULL ){
		defer_alert(type,msg);
		return;
	}

	is_fatal = !strcmp(type,FATAL_ERROR_TYPE_STR) ? 1 : 0 ;

	QUIP_ALERT_OBJ_TYPE *alert;
	alert = create_alert_with_one_button(type,msg);
	remember_normal_alert(alert);
	fatal_alert_view= is_fatal ? alert : NULL;
	show_alert(QSP_ARG  alert);
} // generic_alert

static void generic_alert(QSP_ARG_DECL  const char *type, const char *msg)
{
	if( busy_alert_p != NULL ) {
		suspend__busy();
		// relinquish control and come back later
		defer_alert(type,msg);
		suspend_quip_interpreter();
		return;
	}
	present_generic_alert(QSP_ARG  type, msg);
}

void get_confirmation(QSP_ARG_DECL  const char *title, const char *question)
{
	UIViewController *vc;

	if( busy_alert_p != NULL ) {
		suspend__busy();
	}

	vc = root_view_controller.topViewController;

	// vc can be null if we call this before we have set
	// root_view_controller (during startup)

	if( vc == NULL ){
		assign_reserved_var(DEFAULT_QSP_ARG  "confirmed","1");
		return;
	}
	QUIP_ALERT_OBJ_TYPE *alert;
	alert = create_alert_with_two_buttons(title,question);
	remember_confirmation_alert(alert);
	show_alert(QSP_ARG  alert);
} // get_confirmation

/* Like an alert, but we don't stop execution - but wait, we have to,
 * or else the alert won't appear???
 */

void notify_busy(QSP_ARG_DECL  const char *type, const char *msg)
{
	UIViewController *vc;

	if( busy_alert_p != NULL ){
		// we have to dismiss the busy indicator
		// to print a warning pop-up!?
		return;
	}

	vc = root_view_controller.topViewController;

	// vc can be null if we call this before we have set
	// root_view_controller (during startup)

	if( vc == NULL ) return;

	QUIP_ALERT_OBJ_TYPE *alert;

	alert = create_alert_with_no_buttons(type,msg);
	remember_busy_alert(alert);
	show_alert(QSP_ARG  alert);
	busy_alert_p=alert;	// remember for later
} // notify_busy

int check_deferred_alert(SINGLE_QSP_ARG_DECL)
{
	if( deferred_alert.type == NULL ) return 0;

	generic_alert(QSP_ARG  deferred_alert.type,deferred_alert.msg);

	// We release these strings, but are we sure that the system is
	// done with them?  Did generic_alert copy them or make NSStrings?
	clear_deferred_alert();
	return 1;
}

static void dismiss_busy_alert(QUIP_ALERT_OBJ_TYPE *a)
{
#ifdef OLD

	[a dismissWithClickedButtonIndex:0 animated:YES];
#else // ! OLD
	[root_view_controller dismissViewControllerAnimated:YES completion:^(void)
		{
			dispatch_after(0, dispatch_get_main_queue(), ^{
				if( ! check_deferred_alert() ){
					busy_dismissal_checks(a);
				}
			});
		}
	];
	suspend_quip_interpreter(SGL_DEFAULT_QSP_ARG);
#endif // ! OLD
}

// When we call end_busy, we are not really suspended, we already
// did the things as if we were dismissing the alert when we faked
// a dismissal when the alert was shown.  Therefore, when we dismiss
// the gui element now, we have already popped the menu...

void end_busy(int final)
{
	QUIP_ALERT_OBJ_TYPE *a;

	if( busy_alert_p == NULL ){
		NWARN("end_busy:  no busy indicator!?");
		return;
	}

	a=busy_alert_p;
	busy_alert_p=NULL;	// in case the delegate is called...
// This should dismiss the busy alert but not generate a callback?

	// it seems that we are getting a callback...

	if( final ){
		assert( final_ending_busy_p == NULL );
		final_ending_busy_p = a;
	}

	dismiss_busy_alert(a);
} // end_busy

void simple_alert(QSP_ARG_DECL  const char *type, const char *msg)
{
	generic_alert(QSP_ARG  type,msg);
}

void fatal_alert(QSP_ARG_DECL  const char *msg)
{
	generic_alert(QSP_ARG  FATAL_ERROR_TYPE_STR,msg);
}

static IOS_List *alert_lp=NULL;

// need to call this after the alert has been displayed...
// This used to be a callback, but now with UIAlertController we don't have
// a callback?
//
// Note that this is only needed for busy alerts...

void quip_alert_shown(QUIP_ALERT_OBJ_TYPE *alertView)
{
	Alert_Info *aip;
	aip = [Alert_Info alertInfoFor:alertView];
	if( aip.type == QUIP_ALERT_BUSY ){
		// we used to pass aip.qlevel here, is that needed??
		resume_quip(DEFAULT_QSP_ARG);
	}
}

// We call this after halting the interpreter to allow the system to do its thing.
//
void sync_with_ios(void)
{
	[globalAppDelegate insure_wakeup];
}

void dismiss_keyboard(Screen_Obj *sop)
{
	UITextView * tvp;

	// make sure that this is SOT_EDIT_BOX?

	tvp = (UITextView *)SOB_CONTROL(sop);
	SET_SOB_FLAG_BITS(sop,SOF_KEYBOARD_DISMISSING);
	[ tvp resignFirstResponder ];
	CLEAR_SOB_FLAG_BITS(sop,SOF_KEYBOARD_DISMISSING);
}

void enable_widget(QSP_ARG_DECL  Screen_Obj *sop, int yesno)
{
	WARN("Oops, enable_widget not implemented yet!?");
}

void hide_widget(QSP_ARG_DECL  Screen_Obj *sop, int yesno)
{
	WARN("Oops, hide_widget not implemented yet!?");
}

@implementation Alert_Info

static IOS_Node *node_for_alert(QUIP_ALERT_OBJ_TYPE *a)
{
	IOS_Node *np;

	assert( alert_lp != NULL );
	np=IOS_LIST_HEAD(alert_lp);
	while(np!=NULL){
		Alert_Info *ai;
		ai = (Alert_Info *) IOS_NODE_DATA(np);
		if( ai.the_alert_p == a )
			return np;
		np = IOS_NODE_NEXT(np);
	}
	NWARN("CAUTIOUS:  node_for_alert:  alert not found!?");
	// assert!
	return NULL;
}

+(Alert_Info *) alertInfoFor: (QUIP_ALERT_OBJ_TYPE *)a
{
	IOS_Node *np;
	np = node_for_alert(a);
	if(np==NULL) return NULL;
	return (Alert_Info *) IOS_NODE_DATA(np);
}

+(void) rememberAlert:(QUIP_ALERT_OBJ_TYPE *)a withType:(Quip_Alert_Type)t
{
	if( alert_lp == NULL )
		alert_lp = new_ios_list();

	Alert_Info *ai=[[Alert_Info alloc] init];

	ai.type = t;
	ai.the_alert_p = a;
	ai.qlevel = QS_LEVEL(DEFAULT_QSP);

	IOS_Node *np;

	np = mk_ios_node(ai);
	ios_addHead(alert_lp,np);
}

-(void) forget
{
	IOS_Node *np;
	np = node_for_alert(self.the_alert_p);
	assert( np != NULL );

	np = ios_remNode(alert_lp,np);

	assert( np != NULL );
	np.data = NULL;		// ARC garbage collection
				// will clean Alert_Info
				// is this necessary?
	rls_ios_node(np);
}

@end

#endif /* BUILD_FOR_IOS */

