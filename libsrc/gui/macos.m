#include "quip_config.h"

#ifdef BUILD_FOR_MACOS

#define DEFAULT_NS_COLOR	blackColor

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
#include "viewer.h"
#include "cmaps.h"

//#include <Foundation/NSString.h>
#include <AppKit/NSButton.h>
#include <AppKit/NSStringDrawing.h>
#include <AppKit/NSColor.h>
#include <AppKit/NSTableView.h>
#include <AppKit/NSTextContainer.h>

#include "quipWindowController.h"

static NSAlert *fatal_alert_view=NULL;
static NSAlert *busy_alert_p=NULL;
static NSAlert *hidden_busy_p=NULL;
static NSAlert *ending_busy_p=NULL;


#define PIXELS_PER_CHAR 9	// BUG should depend on font, may not be fixed-width!?
#define EXTRA_WIDTH	20	// pad for string length...

#define LABEL_HEIGHT	BUTTON_HEIGHT

// The original X11 code assumed 0,0 at the upper left,
// but the Mac puts 0,0 at the lower left...
// So we have to calculate

//#define MACOS_PO_CURR_Y(p,s)	(PO_HEIGHT(p) - PO_CURR_Y(p) - SOB_HEIGHT(s))
#define MACOS_PO_CURR_Y(p,s)	(PO_HEIGHT(p) - PO_CURR_Y(p))

// find_any_scrnobj should search all contexts...
// taken from ios.m...

Screen_Obj *find_any_scrnobj(NSView *cp)
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

void window_sys_init(SINGLE_QSP_ARG_DECL)
{
	// nop
}

void reposition(Screen_Obj *sop)
{
	/* Reposition frame if it has one */
	NWARN("need to implement reposition in ios.c");
}

#ifdef FOOBAR
Panel_Obj *find_panel(QSP_ARG_DECL  quipView *qv)
{
	IOS_List *lp;
	IOS_Node *np;
	Panel_Obj *po;

	lp=panel_obj_list(SINGLE_QSP_ARG);
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
#endif // FOOBAR

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

#ifdef FOOBAR
static NSView *find_first_responder(NSView *uivp)
{
sprintf(DEFAULT_ERROR_STRING,"find_first_responder:  testing 0x%lx",(long)uivp);
NADVISE(DEFAULT_ERROR_STRING);
	if( [uivp isFirstResponder] ) return(uivp);
	for (NSView *subView in uivp.subviews) {
		NSView *svp;
		svp = find_first_responder(subView);
		if( svp != NULL ) return(svp);
	}
	return NULL;
}
#endif // FOOBAR

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
#ifdef BUILD_FOR_IOS
		if( yesno ){
			[PO_QV(po) bringSubviewToFront:SOB_CONTROL(sop)];
		} else {
			[PO_QV(po) sendSubviewToBack:SOB_CONTROL(sop)];
		}
#endif // BUILD_FOR_IOS
		np = IOS_NODE_NEXT(np);
	}
}

#ifdef FOOBAR
void check_first(Panel_Obj *po)
{
	NSView *uiv;

	uiv = find_first_responder( PO_QV(po).superview.superview );
	sprintf(DEFAULT_ERROR_STRING,"check_first:  found 0x%lx",(long)uiv);
	NADVISE(DEFAULT_ERROR_STRING);
}
#endif // FOOBAR


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

void make_button(QSP_ARG_DECL  Screen_Obj *sop)
{
	/* We need the app delegate (to set event handling),
	 * and the view (to render the button)
	 */

	int extra_width=20;	// BUG should depend on font size, button height...

	NSButton *newButton = [[NSButton alloc] init];

	// BUG?  should the button size depend on the text label?
	// BUG this should depend on the font size...
	int button_width = (int)strlen(SOB_NAME(sop))*PIXELS_PER_CHAR + extra_width;

	SET_SOB_WIDTH(sop,button_width);
	SET_SOB_HEIGHT(sop,BUTTON_HEIGHT);

	newButton.frame = CGRectMake(PO_CURR_X(curr_panel), 
		MACOS_PO_CURR_Y(curr_panel,sop)-SOB_HEIGHT(sop),
		SOB_WIDTH(sop), SOB_HEIGHT(sop) );

	[newButton setTitle:STRINGOBJ(SOB_NAME(sop)) ];

	[newButton setTarget:GW_WC(PO_GW(curr_panel))];
	// xcode complains, but GW_WC is a quipWindowController,
	// and windowButtonAction is one of its methods???
	[newButton setAction: @selector(windowButtonAction:)];

	SET_SOB_CONTROL(sop,newButton);
}

// the NSSwitch is just the switch - no descriptive text...
// Maybe a NSText view for the label?

//#define TOGGLE_LABEL_WIDTH	130
#define SWITCH_HEIGHT		22

void make_toggle(QSP_ARG_DECL  Screen_Obj *sop)
{
	NSTextView *l;
	CGRect f;
	int w;

	// How do we know how wide to make the label?
	w=PIXELS_PER_CHAR*(int)strlen(SOB_NAME(sop))+EXTRA_WIDTH;

	f=CGRectMake(PO_CURR_X(curr_panel),PO_CURR_Y(curr_panel),w,SWITCH_HEIGHT);
	l=[[NSTextView alloc] initWithFrame:f];
	//l.text = STRINGOBJ( SOB_NAME(sop) );
	l.backgroundColor = [NSColor clearColor];
	l.textColor = [NSColor DEFAULT_NS_COLOR];
	//l.textAlignment = NSTextAlignmentCenter;

	// width and height are don't care's for switches...
	f=CGRectMake(PO_CURR_X(curr_panel)+w,PO_CURR_Y(curr_panel),0,0);
	NSButton *newSwitch = [[NSButton alloc ] initWithFrame: f];
	[newSwitch setButtonType:NSToggleButton];

//	[newSwitch setTitle:STRINGOBJ(SOB_NAME(sop)) forState:NSControlStateNormal];

#ifdef FOOBAR
	[newSwitch addTarget:globalAppDelegate
		action:@selector(genericSwitchAction:)
		/*forControlEvents:NSControlEventTouchUpInside*/];
#endif // FOOBAR
	
	//[ PO_QV(curr_panel) addSubview:l ];
	[PO_WINDOW(curr_panel) setContentView:l];
	// also makeKeyAndOrderFront:nil
	// also makeFirstResponder:l - ???  only for input...

	// The switch appears, but where is the addSubview???
	SET_SOB_CONTROL(sop,newSwitch);
}

#ifdef FOOBAR
// This code may be helpful in determining the size of a label...
#if defined(TARGET_IPHONE_SIMULATOR) || defined(TARGET_OS_IPHONE)
    UIFont *theFont = [UIFont fontWithName:fontName size:fontSize];
    CGSize textSize = [text sizeWithFont:theFont];
#else
    NSFont *theFont = [NSFont fontWithName:fontName size:fontSize];
    CGSize textSize = NSSizeToCGSize([text sizeWithAttributes:[NSDictionary dictionaryWithObject:theFont forKey: NSFontAttributeName]]);
#endif
#endif // FOOBAR

//static char *label_font_name="Times Roman";
static char *label_font_name="Helvetica";	// This is cocoa's default
static float label_font_size=12.0;

/*	justification
 *
 *	4 bits, with 9 possibilities:
 *	H:  L,C,R
 *	V:  B,C,T
 */

#define HJUST_LEFT	1
#define HJUST_RIGHT	2
#define HJUST_CENTER	3
#define HJUST_MASK	3
#define VJUST_BOTTOM	4
#define VJUST_TOP	8
#define VJUST_CENTER	12
#define VJUST_MASK	12

static NSTextView * new_label(const char *s, int x, int y, int justification_code)
{
	CGRect f;

	NSString *label_string = STRINGOBJ( s );
	// BUG should cache these things...
	NSFont *labelFont = [NSFont fontWithName:STRINGOBJ(label_font_name)
							size:label_font_size];
    // sizeWithAttributes is documented for UIKit only?
	NSSize ns_size=[label_string sizeWithAttributes:
		[NSDictionary dictionaryWithObject:labelFont
				forKey:NSFontAttributeName]  ];
	CGSize labelSize = NSSizeToCGSize(ns_size);

	int dy=(int)ceil(labelSize.height);
	int dx=(int)ceil(labelSize.width);
	int x0,y0;
	switch( justification_code & HJUST_MASK ){
		case HJUST_LEFT: x0=x; break;
		case HJUST_RIGHT: x0=x-dx; break;
		case HJUST_CENTER: x0=x-dx/2; break;
#ifdef CAUTIOUS
		default:
			NWARN("CAUTIOUS:  new_label:  bad H justification code!?");
			x0=x;
			break;
#endif // CAUTIOUS
	}
	switch( justification_code & VJUST_MASK ){
		case VJUST_BOTTOM: y0=y; break;
		case VJUST_TOP: y0=y+dy; break;
		case VJUST_CENTER: y0=y+dy/2; break;
#ifdef CAUTIOUS
		default:
			NWARN("CAUTIOUS:  new_label:  bad V justification code!?");
			y0=y;
			break;
#endif // CAUTIOUS
	}
	f=CGRectMake( x0, PO_HEIGHT(curr_panel)-y0,dx,dy );
	NSTextView *l=[[NSTextView alloc] initWithFrame:f];
	l.backgroundColor = [NSColor clearColor];
	l.textColor = [NSColor DEFAULT_NS_COLOR];
	l.textContainer.lineFragmentPadding = 0;
	//l.textAlignment = NSTextAlignmentCenter;
	l.string = STRINGOBJ( s );

	return l;
}

#define RIGHT_LABEL_MARGIN	10

void make_label(QSP_ARG_DECL  Screen_Obj *sop)
{
	NSTextView *l;

	l = new_label( SOB_CONTENT(sop), 
		PO_CURR_X(curr_panel),PO_CURR_Y(curr_panel),
		HJUST_LEFT|VJUST_TOP);

	SET_SOB_HEIGHT(sop,(int)l.frame.size.height);
	SET_SOB_WIDTH(sop,(int)l.frame.size.width);

	SET_SOB_CONTROL(sop,(NSControl *)l);
}

void make_message(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef FOOBAR
	NSTextView *l;
	CGRect f;
	int w;
	char s[LLEN];

	sprintf(s,"%s:  %s",SOB_NAME(sop),SOB_CONTENT(sop));
	//w=PIXELS_PER_CHAR*strlen(s)+EXTRA_WIDTH;
	w=globalAppDelegate.dev_size.width,
	f=CGRectMake(PO_CURR_X(curr_panel),PO_CURR_Y(curr_panel),w,BUTTON_HEIGHT);
	l=[[NSLabel alloc] initWithFrame:f];
	l.text = STRINGOBJ( s );
	l.backgroundColor = [NSColor clearColor];
	l.textAlignment = NSTextAlignmentCenter;

	// font?
//	l.font = [NSFont systemFontOfSize:SOB_FONT_SIZE(sop)];

	[ PO_QV(curr_panel) addSubview:l ];
	SET_SOB_CONTROL(sop,l);

#else
	WARN("make_message not implemented yet!?");
#endif // ! FOOBAR
}

void reload_chooser(Screen_Obj *sop)
{
	NSTableView *t=(NSTableView *)SOB_CONTROL(sop);

	[t reloadData];
}

void set_choice(Screen_Obj *sop, int which)
{
	if( ! IS_CHOOSER(sop) ){
		sprintf(DEFAULT_ERROR_STRING,
			"set_choice:  screen object %s is not a chooser!?",
			SOB_NAME(sop));
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	NSMatrix *m = (NSMatrix *)SOB_CONTROL(sop);
	[m selectCellAtRow:which column:0];
}

#ifdef FOOBAR
	if( IS_CHOOSER(sop) ){
		NSTableView *t;
		NSIndexPath *path;
		NSUInteger idxs[2];

		t=(NSTableView *)SOB_CONTROL(sop);

		idxs[0]=0;	// section
		idxs[1]=which;
		path = [[ NSIndexPath alloc ] initWithIndexes:idxs length:2 ];

		[ t	selectRowAtIndexPath:path
			animated:YES
			scrollPosition:NSTableViewScrollPositionMiddle ];
	} else if( SOB_TYPE(sop) == SOT_PICKER ){
		NSPickerView *p;
		p=(NSPickerView *)SOB_CONTROL(sop);
		[p selectRow:which inComponent:0 animated:NO];
	} else {
		sprintf(DEFAULT_ERROR_STRING,
"set_choice:  object %s is neither a picker nor a chooser!?",SOB_NAME(sop));
		NWARN(DEFAULT_ERROR_STRING);
	}
#endif // FOOBAR


void set_pick(Screen_Obj *sop, int cyl, int which)
{
#ifdef FOOBAR
	if( SOB_TYPE(sop) == SOT_PICKER ){
		NSPickerView *p;
		p=(NSPickerView *)SOB_CONTROL(sop);

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
"set_pick:  object %s is not a picker!?",SOB_NAME(sop));
		NWARN(DEFAULT_ERROR_STRING);
	}
#endif // FOOBAR
}

#define DY_PER_CHOICE	19	// BUG - just a guess

void make_chooser(QSP_ARG_DECL  Screen_Obj *sop, int n, const char **stringlist)
{
	NSMatrix *m;
	NSRect f;
	int dy;
	NSButtonCell *myCell;

	// put a label above the radio buttons
	NSTextView *l=new_label(SOB_NAME(sop),PO_CURR_X(curr_panel),
			PO_CURR_Y(curr_panel),
			HJUST_LEFT|VJUST_TOP );
	[[GW_WINDOW(PO_GW(curr_panel)) contentView] addSubview : l ];
	INC_PO_CURR_Y(curr_panel, (int)l.frame.size.height );

	// BUG need to get dy with knowledge of font etc.
	dy = n * DY_PER_CHOICE;
	int y0=PO_CURR_Y(curr_panel)+dy;
fprintf(stderr,"make_chooser:  OBJECT_GAP = %d\n",OBJECT_GAP);
	f=NSMakeRect(PO_CURR_X(curr_panel),PO_HEIGHT(curr_panel)-y0,
		PO_WIDTH(curr_panel)-2*OBJECT_GAP, dy);

	// A problem with this prototype method, is that the size of all the cells
	// is restricted by the length of the first string...

	myCell = [[NSButtonCell alloc] init];
	// If we initialize with the first string, then this limits
	// the width of ALL the cells...
	//[myCell setTitle:STRINGOBJ(stringlist[0])];
	// This is a total hack
	//[myCell setTitle:STRINGOBJ("This is the longest conceivable choice string")];
#define MAX_CHOICE_CHARS	40
	int max_choice_chars=0;
	int i;

	//                          01234567890123456789012345678901234567890
	char *dummy_string=(char *)savestr("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww");
	for(i=0;i<n;i++){
		if( strlen(stringlist[i]) > max_choice_chars )
			max_choice_chars = (int) strlen(stringlist[i]);
	}
	if( max_choice_chars > MAX_CHOICE_CHARS ){
		NWARN("make_chooser:  need to increase MAX_CHOICE_CHARS!?");
		max_choice_chars=MAX_CHOICE_CHARS;
	}
	dummy_string[max_choice_chars]=0;

	[myCell setButtonType:NSRadioButton];
	[myCell setTitle:STRINGOBJ(dummy_string)];
	rls_str(dummy_string);
	
	m=[[NSMatrix alloc] initWithFrame:f mode:NSRadioModeMatrix
		prototype:(NSCell *)myCell
		numberOfRows:n numberOfColumns:1 ];

	// The matrix delegate is just for text?
	//m.delegate = globalAppDelegate;

	NSArray *cellArray = m.cells;

	for(i=0;i<n;i++){
		NSButtonCell *c;
		c = [cellArray objectAtIndex:i];
		[c setTitle:STRINGOBJ(stringlist[i])];
		[c setTarget:GW_WC(PO_GW(curr_panel))];
		[c setAction: @selector(windowChooserAction:)];
	}

	SET_SOB_WIDTH(sop,(int) f.size.width );
	SET_SOB_HEIGHT(sop,(int) f.size.height );

	SET_SOB_CONTROL(sop,m);
}

void make_picker(QSP_ARG_DECL  Screen_Obj *sop )
{
#ifdef FOOBAR
	NSPickerView *p;
	CGRect f;

	// most widgets have a small gap from the left hand edge (10 pixels?)
	// but ios NSPickerView's take up the full screen width on the iPhone...
	// BUG make left hand gap device-type-dependent
	f=CGRectMake(PO_CURR_X(curr_panel),PO_CURR_Y(curr_panel),PO_WIDTH(curr_panel)-2*OBJECT_GAP,0);
	p=[[NSPickerView alloc] initWithFrame:f];

	p.delegate = globalAppDelegate;
	p.dataSource = globalAppDelegate;
	//p.backgroundColor = [NSColor clearColor];

	p.showsSelectionIndicator = YES;

	SET_SOB_CONTROL(sop,p);
#endif // FOOBAR
}


void make_text_field(QSP_ARG_DECL  Screen_Obj *sop)
{
	NSTextView *l=new_label(SOB_NAME(sop),
		PO_CURR_X(curr_panel),
		PO_CURR_Y(curr_panel)+MESSAGE_HEIGHT/2,
		HJUST_LEFT|VJUST_CENTER );
	[[GW_WINDOW(PO_GW(curr_panel)) contentView] addSubview : l ];

	// BUG - make sure so_type is text field or password

	// BUG should set width based on width of device!
	// We should be able to get that from the View...

#define SMALL_GAP 3
	int legend_w=SMALL_GAP+(int)l.frame.size.width;
	SET_SOB_WIDTH(sop,SOB_WIDTH(sop)-legend_w);
	CGRect r = CGRectMake(PO_CURR_X(curr_panel)+legend_w,
		PO_HEIGHT(curr_panel)-PO_CURR_Y(curr_panel)-SOB_HEIGHT(sop),
		SOB_WIDTH(sop), SOB_HEIGHT(sop) ) ;
	NSTextField *textField = [[NSTextField alloc] initWithFrame:r];

	//textField.borderStyle = NSTextBorderStyleRoundedRect;
	textField.font = [NSFont systemFontOfSize:SOB_FONT_SIZE(sop)];

	// Setting the default text causes a callback,
	// so we postpone it until we have added this widget to the panel

	//textField.autocorrectionType = NSTextAutocorrectionTypeNo;
	//textField.autocapitalizationType = NSTextAutocapitalizationTypeNone;
	//textField.keyboardType = NSKeyboardTypeDefault;
	//textField.returnKeyType = NSReturnKeyDone;
	//textField.clearButtonMode = NSTextFieldViewModeWhileEditing;
	//textField.clearsOnBeginEditing = YES;
	//textField.contentVerticalAlignment = NSControlContentVerticalAlignmentCenter;
	textField.delegate = globalAppDelegate;

	/*if( SOB_TYPE(sop) == SOT_PASSWORD )
		textField.secureTextEntry = YES;
*/
	
	SET_SOB_CONTROL(sop,textField);
}

void install_initial_text(Screen_Obj *sop)
{
	NSTextField *textField;

	// BUG make sure sop is right kind of widget
	textField = (NSTextField *)SOB_CONTROL(sop);
	//textField.placeholder = STRINGOBJ( SOB_CONTENT(sop) );
}

void make_text_box(QSP_ARG_DECL  Screen_Obj *sop, BOOL is_editable )
{
	CGRect r;
	//quipView *qv;
	NSWindow *w;
	NSTextView *msg_view;

	w = PO_WINDOW(curr_panel);

	// BUG should set width based on width of device!
	// We should be able to get that from the View...
	// We should set the font size based on which type of device, and what
	// screen resolution we have...

	r=CGRectMake(PO_CURR_X(curr_panel), PO_CURR_Y(curr_panel),
		SOB_WIDTH(sop), SOB_HEIGHT(sop) );

	msg_view = [[NSTextView alloc] initWithFrame:r];
	msg_view.font = [NSFont systemFontOfSize:SOB_FONT_SIZE(sop)];
	//msg_view.borderStyle = NSTextBorderStyleRoundedRect;

	//msg_view.text = @"system output:\n";	// appropriate for console, but not general BUG
	//msg_view.text = STRINGOBJ(SOB_CONTENT(sop));

	msg_view.editable=is_editable;

//	textField.contentVerticalAlignment = NSControlContentVerticalAlignmentCenter;

	msg_view.delegate = globalAppDelegate;

	// Don't we need to set SOB_CONTROL???

	//[qv addSubview:msg_view];

	//SET_SOB_CONTROL(sop,msg_view);
}

void make_act_ind(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef FOOBAR
	quipView *qv;
	NSActivityIndicatorView *ai;

	qv = PO_QV(curr_panel);

	// BUG should set width based on width of device!
	// We should be able to get that from the View...
	// We should set the font size based on which type of device, and what
	// screen resolution we have...

	ai = [[NSActivityIndicatorView alloc]
		initWithActivityIndicatorStyle:NSActivityIndicatorViewStyleWhite];

	//[ai setCenter:CGPointMake(PO_WIDTH(curr_panel)/2.0f,
	//			PO_HEIGHT(curr_panel)/2.0f)];
	[ai setCenter:CGPointMake(PO_CURR_X(curr_panel)+10.0f,
				PO_CURR_Y(curr_panel)+10.0f)];

	// Don't we need to set SOB_CONTROL???

	[qv addSubview:ai];

	SET_SOB_CONTROL(sop,ai);
#endif // FOOBAR
}

void set_activity_indicator(Screen_Obj *sop, int on_off )
{
#ifdef FOOBAR
	NSActivityIndicatorView *ai;

	ai = (NSActivityIndicatorView *)SOB_CONTROL(sop);

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
#endif // FOOBAR
}

// BUG this is a hack tuned to the EZJet app on iPod 5
#define EDIT_BOX_FRACTION	0.7

void make_edit_box(QSP_ARG_DECL  Screen_Obj *sop)
{
	/* put this after addition to panel list, because setting initial value causes a callback */
	//get_device_dims(sop);

	// BUG - we reduce the height from the requested height
	// to make sure there is room for the keyboard -
	// but this is the wrong computation!?

	SET_SOB_HEIGHT(sop,(int)(EDIT_BOX_FRACTION*(float)SOB_HEIGHT(sop)));

	make_text_box(DEFAULT_QSP_ARG  sop, YES );

	//msg_view = (NSTextView *)SOB_CONTROL(sop);	// remember this for later
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
	//NSTextView *t=(NSTextView *)SOB_CONTROL(sop);
	NSControl *t=SOB_CONTROL(sop);
	return t.stringValue.UTF8String;
}

void make_gauge(QSP_ARG_DECL  Screen_Obj *sop)
{
	WARN("make_gauge not implemented!?");

}

/* Updates the label of a gauge or slider */

void set_gauge_label(Screen_Obj *gp, const char *str )
{
#ifdef FOOBAR
	NSLabel *l;

	// BUG make sure gp is really a gauge.
	// but this should be done before calling?

	l=(NSLabel *) SOB_LABEL(gp);
	l.text = STRINGOBJ( str );
#endif // FOOBAR
}

void update_gauge_label(Screen_Obj *gp )
{
	NSTextView *l;
	char valstr[64];

	// BUG make sure gp is really a gauge.
	// but this should be done before calling?

	l=(NSTextView *) SOB_LABEL(gp);
	sprintf(valstr,":  %d",SOB_VAL(gp));
	l.string = [STRINGOBJ( SOB_NAME(gp) )
			stringByAppendingString: STRINGOBJ(valstr) ];
}


/* This should leave enough room to display the anchor values.
 * But where can we display the current value?
 */

#define SLIDER_SIDE_GAP	55
#define SLIDER_DELTA_Y	40	/* just the slider itself */
#define SLIDER_TEXT_DY	20	// should depend on font size!
#define MACOS_SLIDER_HEIGHT	38

void make_slider(QSP_ARG_DECL  Screen_Obj *sop)
{
	int slider_width;

	SET_SOB_HEIGHT(sop,MACOS_SLIDER_HEIGHT);

	/* BUG make sure that display
	 * is at least that wide...
	 */
	slider_width = SOB_WIDTH(sop)-2*SLIDER_SIDE_GAP;

	NSSlider *slider = [[NSSlider alloc] initWithFrame:
		CGRectMake(
			PO_CURR_X(curr_panel) + SLIDER_SIDE_GAP,
			MACOS_PO_CURR_Y(curr_panel,sop)-SLIDER_DELTA_Y,
			slider_width,
			SLIDER_DELTA_Y
		)
	];

	slider.minValue = (double) SOB_MIN(sop);
	slider.maxValue = (double) SOB_MAX(sop);
	slider.floatValue = (float) SOB_VAL(sop);

	[slider setTarget:GW_WC(PO_GW(curr_panel))];
	// this selector comes from quipWindowController???
	[slider setAction: @selector(windowSliderAction:)];

	// The default NSSliders seem to behave like adjusters,
	// events are generated for every movement...

	if( SOB_TYPE(sop) == SOT_SLIDER ){
fprintf(stderr,"clearing continuous flag for slider\n");
		[slider setContinuous:NO];
	} else if( SOB_TYPE(sop) == SOT_ADJUSTER ){
		[slider setContinuous:YES];	// not needed, this is the default
	}
#ifdef CAUTIOUS
	  else {
		WARN("CAUTIOUS:  make_slider:  unrecognized object type!?");
		return;
	}
#endif /* CAUTIOUS */

	SET_SOB_CONTROL(sop,slider);

	NSTextView *l;
	char min_str[16];
	char max_str[16];

	sprintf(max_str,"%d",SOB_MAX(sop));
	sprintf(min_str,"%d",SOB_MIN(sop));

	char *legend_str;
	unsigned long nneed=strlen(SOB_NAME(sop)) + 4 +
		(strlen(max_str)>strlen(min_str)?strlen(max_str):strlen(min_str));
	legend_str = getbuf(nneed);
	sprintf(legend_str,"%s:  %s",SOB_NAME(sop),
		(strlen(max_str)>strlen(min_str)?max_str:min_str) );

	l=new_label(legend_str,PO_CURR_X(curr_panel)+slider_width/2+SLIDER_SIDE_GAP,
			PO_CURR_Y(curr_panel)+SLIDER_DELTA_Y,
			HJUST_CENTER|VJUST_TOP );
	givbuf(legend_str);
	SET_SOB_LABEL(sop,l);
	update_gauge_label(sop);
	[[GW_WINDOW(PO_GW(curr_panel)) contentView] addSubview : l ];
	SET_SOB_HEIGHT(sop,MACOS_SLIDER_HEIGHT+(int)l.frame.size.height);


	l=new_label(max_str,PO_CURR_X(curr_panel)+slider_width+SLIDER_SIDE_GAP,
			PO_CURR_Y(curr_panel)+SLIDER_DELTA_Y/2,
			HJUST_LEFT|VJUST_CENTER );
	// This is normally done in add_to_panel
	[[GW_WINDOW(PO_GW(curr_panel)) contentView] addSubview : l ];
	
	l=new_label(min_str,PO_CURR_X(curr_panel)+SLIDER_SIDE_GAP,
			PO_CURR_Y(curr_panel)+SLIDER_DELTA_Y/2,
			HJUST_RIGHT|VJUST_CENTER);
	//add_view_to_panel(curr_panel,l);
	[[GW_WINDOW(PO_GW(curr_panel)) contentView] addSubview : l ];
}

// Does this still work in iOS?

void make_adjuster(QSP_ARG_DECL  Screen_Obj *sop)
{
	make_slider(QSP_ARG  sop);
#ifdef FOOBAR
	NSSlider *slider = [[NSSlider alloc] initWithFrame:
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
			(NSControlEventTouchUpInside|
			 NSControlEventTouchDragInside)];
	SET_SOB_CONTROL(sop,slider);
#endif // FOOBAR
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
#ifdef FOOBAR
	NSSwitch *sw;

	// BUG check for correct widget type here
	sw = (NSSwitch *) SOB_CONTROL(sop);

	// BUG maybe nice to animate sometimes - how to determine when?
	// Only if panel is visible?
	[ sw setOn:(val?YES:NO) animated:NO ];
#endif // FOOBAR
}

void set_gauge_value(Screen_Obj *gp, int n)
{
	NSSlider *s;

	// BUG make sure gp is really a gauge

	s=(NSSlider *) SOB_CONTROL(gp);
	//s.value = (float) n;

	SET_SOB_VAL(gp,n);
	update_gauge_label(gp);
}

void update_message(Screen_Obj *sop)
{
#ifdef FOOBAR
	NSLabel *l;
	char s[LLEN];

	sprintf(s,"%s:  %s",SOB_NAME(sop),SOB_CONTENT(sop));
	// BUG check widget type
	l = (NSLabel *) SOB_CONTROL(sop);
	l.text = STRINGOBJ( s );
	//[l sizeToFit];
	[l setNeedsDisplay];
#endif // FOOBAR
}

void update_label(Screen_Obj *sop)
{
#ifdef FOOBAR
	NSLabel *l;

	// BUG check widget type
	l = (NSLabel *) SOB_CONTROL(sop);
	l.text = STRINGOBJ( SOB_CONTENT(sop) );
	//[l sizeToFit];
//fprintf(stderr,"update_label #2:  label is %f x %f\n",
//l.frame.size.width,l.frame.size.height);
	[l setNeedsDisplay];
#endif // FOOBAR
}

// text to append is passed in SOB_CONTENT

void update_text_box(Screen_Obj *sop)
{
	NSTextView *tb;

	tb = (NSTextView *) SOB_CONTROL(sop);
	//tb.text = [tb.text stringByAppendingString: STRINGOBJ( SOB_CONTENT(sop) ) ];
}

// what kind of object can this be?
// NSLabel, NSTextField...

void update_text_field(Screen_Obj *sop, const char *string)
{
#ifdef FOOBAR
	switch( SOB_TYPE(sop) ){

		case SOT_MESSAGE:
		{
			NSLabel *l;

			l = (NSLabel *) SOB_CONTROL(sop);
			l.text = STRINGOBJ( string );
		}
			break;

		case SOT_TEXT:
		case SOT_PASSWORD:
		case SOT_TEXT_BOX:
		case SOT_EDIT_BOX:
		{
			NSTextView *tb;

			tb = (NSTextView *) SOB_CONTROL(sop);
			tb.text = STRINGOBJ( string );
		}
			break;

		default:
			sprintf(DEFAULT_ERROR_STRING,
	"update_text_field:  unhandled object type, screen object %s",SOB_NAME(sop));
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
#endif // FOOBAR
}

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

/* IOS:  In principle, we only need to search the top context?
 * MacOS, unix:	we can have many windows open at once...
 */

Screen_Obj *find_scrnobj(SOB_CTL_TYPE *cp)
{
	IOS_List *lp = [Screen_Obj getListOfAllItems];

	assert( lp != NULL );

	IOS_Node *np;
	np=IOS_LIST_HEAD(lp);
	while(np!=NULL){
		Screen_Obj *sop = (Screen_Obj *)IOS_NODE_DATA(np);

		if( SOB_CONTROL(sop) == cp )
			return sop;
		np=IOS_NODE_NEXT(np);
	}

	/* If we haven't found it, then it must mean that we have a context error... */
	return NULL;
}

// We have been experimenting with using other threads...
// NS stuff has to run on main thread...

// This comment doesn't seem to be true now:
//
// push_nav works fine for pushing viewers and panels,
// but it doesn't push another navigation panel...
// Is that because the view controller is wrong???
//
//

int n_pushed_panels(void)
{
	//NSArray *a = [ root_view_controller viewControllers];
	//return a.count;
	return 0;
}

void push_nav(QSP_ARG_DECL  Gen_Win *gwp)
{
	// Make sure that this has not been pushed already by scanning the stack
#ifdef FOOBAR
	NSArray *a = [ root_view_controller viewControllers];
	int i;
	for(i=0;i<a.count;i++){
		NSViewController *vc;
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
	NSBarButtonItem *item = [[NSBarButtonItem alloc]
			initWithBarButtonSystemItem:NSBarButtonSystemItemDone
			target:qvc
			action:@selector(qvcDoneButtonPressed)];

	// BUG?  do we have to remove this also
	root_view_controller.navigationItem.rightBarButtonItem = item;

			}
		}
			break;

		case GW_VC_QTVC:
		{
			quipTableViewController *qtvc;
			qtvc = (quipTableViewController *)GW_VC(gwp);
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
	// perhaps the solution to this is the NSNavigationControllerDelegate
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
#endif // FOOBAR
} // end push_nav

// pop_nav pops the panel widget context, but this won't happen
// if the controller is popped using the nav bar!?
// We haven't seen problems from this yet, it would probably
// take a name overloading to reveal it...  BUG
// we should be able to fix this using NSNavigationControllerDelegate...
//
// The name overloading comment above may have been true
// when find_scrnobj searched the entire context stack for
// widgets, but now we only search the top context, which makes
// it important that it be correct.

void pop_nav(QSP_ARG_DECL int n_levels)
{
#ifdef FOOBAR
	quipViewController *qvc;
	quipViewController *old_vc;	// could also be quipTableViewController!?

	// If the current view controller has locked the orientation,
	// the device orientation may have changed without generating
	// an autorotation; in that case, the new view controller
	// revealed by the pop might need to rotate, but unless
	// we do something this won't happen until a new orientation
	// change event is generated by the accelerometer...

	old_vc = (quipViewController *) root_view_controller.topViewController;

	while( (n_levels--) > 1 ){
		qvc = (quipViewController *)
			[ root_view_controller popViewControllerAnimated:NO
				checkOrientation:[old_vc didBlockAutorotation]
				];
		// don't do this here any more.
		// pop_scrnobj_context(SINGLE_QSP_ARG);
	}

	qvc = (quipViewController *)
		[ root_view_controller popViewControllerAnimated:YES
			checkOrientation:[old_vc didBlockAutorotation]
			];

	// The new view controller may not be a quipViewController!?
	// It could also be a quipTableViewController???

	// pop_scrnobj_context now done in quipNavController delegate
//	pop_scrnobj_context(SINGLE_QSP_ARG);
#endif // FOOBAR
}

void make_console_panel(QSP_ARG_DECL  const char *name)
{
#ifdef NOT_USED
	Panel_Obj *po = new_panel(QSP_ARG  name,	/* make_console_panel */
			(int)globalAppDelegate.dev_size.width,
			(int)globalAppDelegate.dev_size.height);
#endif // NOT_USED
	WARN("make_console_panel:  NOT making console!?");
    // should be done from main thread???

	//[((quipViewController *)PO_VC(po))	addConsoleWithSize:globalAppDelegate.dev_size];
}

#define FATAL_ERROR_TYPE_STR	"FATAL ERROR"

struct alert_data {
	const char *type;
	const char *msg;
};

static struct alert_data deferred_alert = { NULL, NULL };

// We call suspend_busy when the busy alert is up, and we want to
// display a different alert.  we are probably calling this from 
// finish_digestion...

static void suspend_busy(void)
{
	// we need to display an alert while we are busy...
	//hidden_busy_p = busy_alert_p;
	end_busy(0);	// takes the alert down
}

static void generic_alert(QSP_ARG_DECL  const char *type, const char *msg)
{
	NSViewController *vc;
	int is_fatal;

	if( busy_alert_p != NULL ) {
		suspend_busy();
	}

	//vc = root_view_controller.topViewController;

	// vc can be null if we call this before we have set
	// root_view_controller (during startup)

	if( vc != NULL ){

		is_fatal = !strcmp(type,FATAL_ERROR_TYPE_STR) ? 1 : 0 ;

		// open an alert with a single dismiss button
        /*
		NSAlert *alert = [[NSAlert alloc]
			initWithTitle:STRINGOBJ(type)
			message:STRINGOBJ(msg)
			delegate:vc
			cancelButtonTitle: is_fatal ?  @"Exit program" : @"OK"
			otherButtonTitles: nil];
		*/
        NSAlert *alert=[[NSAlert alloc] init];
        
        fatal_alert_view= is_fatal ? alert : NULL;
		[Alert_Info rememberAlert:alert withType:QUIP_ALERT_NORMAL];
		//[alert show];
	} else {
		deferred_alert.type = savestr(type);
		deferred_alert.msg = savestr(msg);
	}

	// The alert won't be shown until we relinquish control
	// back to the system...
	// So we need to inform the system not to interpret any more commands!

	// When multiple alerts are issued, it seems to put them on a stack,
	// so they are shown in reverse order.  We would like to inhibit
	// script execution until the alert is dismissed...
	//
	// BUT if vc is null, we don't trap the callback...

	SET_QS_FLAG_BITS(THIS_QSP,QS_HALTING);
} // generic_alert

void get_confirmation(QSP_ARG_DECL  const char *title, const char *question)
{
	NSViewController *vc;

	if( busy_alert_p != NULL ) {
		suspend_busy();
	}

	//vc = root_view_controller.topViewController;

	// vc can be null if we call this before we have set
	// root_view_controller (during startup)

	if( vc != NULL ){
		// open an alert with a single dismiss button
        /*
		NSAlertView *alert = [[NSAlertView alloc]
			initWithTitle:STRINGOBJ(title)
			message:STRINGOBJ(question)
			delegate:vc
			cancelButtonTitle: @"Cancel"
			otherButtonTitles: @"Proceed", nil ];
         */
        NSAlert *alert = [[NSAlert alloc] init];

		[Alert_Info rememberAlert:alert withType:QUIP_ALERT_CONFIRMATION];
		//[alert show];
	} else {
		/*
		deferred_alert.type = savestr(type);
		deferred_alert.msg = savestr(msg);
		*/
		assign_reserved_var(DEFAULT_QSP_ARG  "confirmed","1");
		return;
	}

	// The alert won't be shown until we relinquish control
	// back to the system...
	// So we need to inform the system not to interpret any more commands!

	SET_QS_FLAG_BITS(THIS_QSP,QS_HALTING);
}

/* Like an alert, but we don't stop execution - but wait, we have to,
 * or else the alert won't appear???
 */

void notify_busy(QSP_ARG_DECL  const char *type, const char *msg)
{
	NSViewController *vc;

	if( busy_alert_p != NULL ){
		// we have to dismiss the busy indicator
		// to print a warning pop-up!?
fprintf(stderr,"OOPS - notify_busy called twice!?\n");
		return;
	}

	//vc = root_view_controller.topViewController;

	// vc can be null if we call this before we have set
	// root_view_controller (during startup)

	if( vc != NULL ){

		// open an alert with a single dismiss button
		// How can we changed the behavior of the delegate???
		/*
         NSAlertView *alert = [[NSAlertView alloc]
			initWithTitle:STRINGOBJ(type)
			message:STRINGOBJ(msg)
			delegate:vc
			cancelButtonTitle: @"Please be patient..."
			otherButtonTitles: nil];
         */
        NSAlert *alert=[[NSAlert alloc] init];
        
		[Alert_Info rememberAlert:alert withType:QUIP_ALERT_BUSY];

		//[alert show];
		busy_alert_p=alert;	// remember for later
	}

	// The alert won't be shown until we relinquish control
	// back to the system...
	// So we need to inform the system not to interpret any more commands!

	// BUT in the case of a busy notification,
	// we want to resume execution when the alert
	// displays, not when it is dismissed...

	SET_QS_FLAG_BITS(THIS_QSP,QS_HALTING);
}

// When we call end_busy, we are not really suspended, we already
// did the things as if we were dismissing the alert when we faked
// a dismissal when the alert was shown.  Therefore, when we dismiss
// the gui element now, we have already popped the menu...

void end_busy(int final)
{
	NSAlert *a;

	if( busy_alert_p == NULL ){
		NWARN("end_busy:  no busy indicator!?");
		return;
	}

	a=busy_alert_p;
	busy_alert_p=NULL;	// in case the delegate is called...
// This should dismiss the busy alert but not generate a callback?

	// it seems that we are getting a callback...

	if( final ){
		assert( ending_busy_p == NULL );
		ending_busy_p = a;
	}

	//[a dismissWithClickedButtonIndex:0 animated:NO];
} // end_busy

static void resume_busy(void)
{
	if( hidden_busy_p == NULL ){
		NWARN("resume_busy:  no hidden busy alert!?");
		return;
	}
	busy_alert_p=hidden_busy_p;
	hidden_busy_p=NULL;
	[Alert_Info rememberAlert:busy_alert_p withType:QUIP_ALERT_BUSY];
	//[busy_alert_p show];
}


void check_deferred_alert(SINGLE_QSP_ARG_DECL)
{
	if( deferred_alert.type == NULL ) return;

	generic_alert(QSP_ARG  deferred_alert.type,deferred_alert.msg);

	// We release these strings, but are we sure that the system is
	// done with them?  Did generic_alert copy them or make NSStrings?
	rls_str(deferred_alert.type);
	rls_str(deferred_alert.msg);
	deferred_alert.type=NULL;
	deferred_alert.msg=NULL;
}

void simple_alert(QSP_ARG_DECL  const char *type, const char *msg)
{
	generic_alert(QSP_ARG  type,msg);
}

void fatal_alert(QSP_ARG_DECL  const char *msg)
{
	generic_alert(QSP_ARG  FATAL_ERROR_TYPE_STR,msg);
}

static void dismiss_normal_alert(Alert_Info *aip)
{
	int level;

	level = aip.qlevel;
	[aip forget];	// removes from list

	if( ALERT_INFO_OBJ(aip) == ending_busy_p ){	// dismissed busy alert
		// nothing more to do.
		ending_busy_p=NULL;
		return;
	}

	if( ALERT_INFO_OBJ(aip) == hidden_busy_p ){
		return;
	}

	if( busy_alert_p == NULL ){	// no hidden busy alert?
		if( hidden_busy_p != NULL ){
			if( ALERT_INFO_OBJ(aip) != hidden_busy_p ){
				resume_busy();
			} else {
			// otherwise this is generated by the
			// programmatic dismissal of the busy alert?
				resume_quip(SGL_DEFAULT_QSP_ARG);
			}
		} else {
			resume_quip(DEFAULT_QSP_ARG);
		}
	} else {
		// don't allow the busy panel to be dismissed by the user
		//[busy_alert_p show];
	}

	// If we had an alarm while the alert was up, 
	// we now have a stored chunk...
}

static IOS_List *alert_lp=NULL;

void dismiss_quip_alert(NSAlert *alert, NSInteger buttonIndex)
{
	if( alert == fatal_alert_view )
		[first_quip_controller qtvcExitProgram];

	Alert_Info *aip;

	aip = [Alert_Info alertInfoFor:alert];

	if( IS_VALID_ALERT(aip) ){
		if( aip.type == QUIP_ALERT_CONFIRMATION ){
			if( buttonIndex == 0 ){
				assign_reserved_var(DEFAULT_QSP_ARG  "confirmed","0");
			} else {
				assign_reserved_var(DEFAULT_QSP_ARG  "confirmed","1");
			}
		}
		dismiss_normal_alert(aip);
	} 
#ifdef CAUTIOUS
	  else {
		sprintf(DEFAULT_ERROR_STRING,
"CAUTIOUS:  dismiss_quip_alert:  Unrecognized alert type %d!?",aip.type);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
#endif // CAUTIOUS
}

void quip_alert_shown(QUIP_ALERT_OBJ_TYPE *alertView)
{
	Alert_Info *aip;
	aip = [Alert_Info alertInfoFor:alertView];
	if( aip.type == QUIP_ALERT_BUSY ){
		// we used to pass aip.qlevel here, is that needed??
		resume_quip(DEFAULT_QSP_ARG);
	}
}


void dismiss_keyboard(Screen_Obj *sop)
{
	NSTextView * tvp;

	// make sure that this is SOT_EDIT_BOX?

	tvp = (NSTextView *)SOB_CONTROL(sop);
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
		if( ALERT_INFO_OBJ(ai) == a )
			return np;
		np = IOS_NODE_NEXT(np);
	}
	NWARN("CAUTIOUS:  node_for_alert:  alert not found!?");
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
	ALERT_INFO_OBJ(ai) = a;
	ai.qlevel = QS_LEVEL(DEFAULT_QSP);

	IOS_Node *np;

	np = mk_ios_node(ai);
	ios_addHead(alert_lp,np);
}

-(void) forget
{
	IOS_Node *np;
	np = node_for_alert(ALERT_INFO_OBJ(self));
	assert( np != NULL );
	
	np = ios_remNode(alert_lp,np);

	assert( np != NULL );
	np.data = NULL;		// ARC garbage collection
				// will clean Alert_Info
				// is this necessary?
	rls_ios_node(np);
}

@end

#endif /* BUILD_FOR_MACOS */

