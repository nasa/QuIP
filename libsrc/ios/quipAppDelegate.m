//
//  quipAppDelegate.m
//  oq4t
//
//  Created by Jeff Mulligan on 7/29/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//

#import "quipAppDelegate.h"
#import "quipTableViewController.h"
#import "quipNavController.h"

#include "quip_prot.h"
#include "ios_gui.h"

#define EMPTY_SELECTOR(name)					\
								\
-(void) name : (id) sender					\
{								\
	fprintf(stderr,"%s CALLED\n",#name);			\
}

#ifdef CAUTIOUS

#define CHECK_SCRNOBJ_VOID(sop,sender,whence,warnflag)			\
	if( sop == NULL ){						\
		if( warnflag ){						\
			sprintf(DEFAULT_ERROR_STRING,			\
	"CAUTIOUS:  %s:  Couldn't find screen object for 0x%lx!?\n",	\
				#whence,(u_long)sender);		\
			NWARN(DEFAULT_ERROR_STRING);			\
		}							\
		return;							\
	}

#define CHECK_SCRNOBJ_BOOL(sop,sender,whence,warnflag)				\
	if( sop == NULL ){						\
		if( warnflag ){						\
			sprintf(DEFAULT_ERROR_STRING,			\
	"CAUTIOUS:  %s:  Couldn't find screen object for 0x%lx!?\n",	\
				#whence,(u_long)sender);		\
			NWARN(DEFAULT_ERROR_STRING);			\
		}							\
		return NO;						\
	}

#define CHECK_SCRNOBJ_INT(sop,sender,whence,warnflag)			\
	if( sop == NULL ){						\
		if( warnflag ){						\
			sprintf(DEFAULT_ERROR_STRING,			\
	"CAUTIOUS:  %s:  Couldn't find screen object for 0x%lx!?\n",	\
				#whence,(u_long)sender);		\
			NWARN(DEFAULT_ERROR_STRING);			\
		}							\
		return 0;						\
	}

#define CHECK_SCRNOBJ_STR(sop,sender,whence,warnflag)			\
	if( sop == NULL ){						\
		if( warnflag ){						\
			sprintf(DEFAULT_ERROR_STRING,			\
	"CAUTIOUS:  %s:  Couldn't find screen object for 0x%lx!?\n",	\
				#whence,(u_long)sender);		\
			NWARN(DEFAULT_ERROR_STRING);			\
		}							\
		return @"???";						\
	}

#endif /* CAUTIOUS */

// globals
int version_major, version_minor, version_release;

#ifdef BUILD_FOR_IOS

// We can put this define in the build settings to force this for a project
#ifdef XCODE_DEBUG
int xcode_debug=1;	//
#else // ! XCODE_DEBUG
int xcode_debug=0;	// production build, disable in project settings
#endif // ! XCODE_DEBUG

#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
int xcode_debug=1;	// BUG we want this disabled for
			// production builds!?
#endif // BUILD_FOR_MACOS

quipAppDelegate *globalAppDelegate=NULL;
quipTableViewController *first_quip_controller=NULL;
quipNavController *root_view_controller=NULL;

#ifdef BUILD_FOR_IOS
static NSString *kCellIdentifier = @"MyIdentifier2";
#endif // BUILD_FOR_IOS

@implementation quipAppDelegate

//@synthesize console_panel;
#ifdef BUILD_FOR_IOS
@synthesize window;
@synthesize qvc;
@synthesize nvc;
@synthesize dev_type;
@synthesize dev_size;
@synthesize wakeup_timer;
@synthesize wakeup_lp;


-(void) applicationDidEnterBackground:(UIApplication *)app
{
	/* Use this method to release shared resources,
	 * save user data, invalidate timers, and store
	 * enough application state information to restore
	 * your application to its current state in case
	 * it is terminated later.
	 * If your application supports background execution,
	 * this method is called instead of
	 * applicationWillTerminate: when the user quits.
	 */

	if( done_button_pushed ){
		log_message("App exiting.");
		exit(0);
	}
}

-(void) applicationDidReceiveMemoryWarning:(UIApplication *)app
{
	NWARN("Low memory!?  Consider rebooting app and/or device");
	// What should we do???
}

// Where does this get called from?  how do we know which qsp to use???
// This gets called when the user types 'DONE' on the pop-up keyboard...
// This routine should handle the result.  In the case of the console,
// we interpret the text, but in general we want to store it in a variable
// and push the action text.

- (BOOL) textFieldShouldReturn : (UITextField *) textField
{
	Screen_Obj *sop = find_any_scrnobj((UIControl *)textField);
	CHECK_SCRNOBJ_BOOL(sop,textField,textFieldShouldReturn,1);

	NSString *s= textField.text;
	assign_var(DEFAULT_QSP_ARG  "input_string",s.UTF8String);

	/* now interpret the action */
	chew_text(DEFAULT_QSP_ARG  SOB_ACTION(sop), "(text field event)");

	[textField resignFirstResponder];

	return YES;
}

-(void) selectionDidChange:(id<UITextInput>)textInput
{
	// nop
	NADVISE("selectionDidChange");
}

-(void) selectionWillChange:(id<UITextInput>)textInput
{
	// nop
	NADVISE("selectionWillChange");
}

- (void)textWillChange:(id <UITextInput>)textInput
{
	// nop
	NADVISE("textWillChange");
}

- (void)textDidChange:(id <UITextInput>)textInput
{
	// nop
	NADVISE("textDidChange");
}

/******* tableView (chooser) support *********/


// returns the number of 'columns' to display.
- (NSInteger)numberOfSectionsInTableView:(UITableView *)tableView
{
	Screen_Obj *sop=find_any_scrnobj(tableView);
	CHECK_SCRNOBJ_INT(sop,tableView,numberOfSectionsInTableView,1);

	if( SOB_TYPE(sop) == SOT_CHOOSER || SOB_TYPE(sop) == SOT_MLT_CHOOSER ){
		return 1;
	} else {
		NWARN("CAUTIOUS:  numberOfSectionsIntableView:  bad screen object!?");
		return 0;
	}
}

- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)component
{
	Screen_Obj *sop=find_any_scrnobj(tableView);
	CHECK_SCRNOBJ_INT(sop,tableView,numberOfRowsInSection,1);

	if( SOB_TYPE(sop) == SOT_CHOOSER || SOB_TYPE(sop) == SOT_MLT_CHOOSER){
		// now access stored information in the object
#ifdef CAUTIOUS
		if( component != 0 ) {
			sprintf(DEFAULT_ERROR_STRING,
"CAUTIOUS:  numberOfRowsInSection (TableView):  component (%ld) should be 0 for a chooser!?",
				(long)component);
			NWARN(DEFAULT_ERROR_STRING);
			return 0;
		}
#endif // CAUTIOUS
		return SOB_N_SELECTORS(sop);
	} else {
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  numberOfRowsInSection:  Bad screen object type!?");
		NWARN(DEFAULT_ERROR_STRING);
		return 0;
	}
//#else // ! CAUTIOUS
	return 0;	// shouldn't happen
//#endif // ! CAUTIOUS
}

- (UITableViewCell *) tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath
{
	UITableViewCell *c;

	Screen_Obj *sop=find_any_scrnobj(tableView);
	CHECK_SCRNOBJ_INT(sop,tableView,cellForRowAtIndexPath,1);

	int r = (int) indexPath.row;
#ifdef CAUTIOUS
	if( r < 0 || r >= SOB_N_SELECTORS(sop) ){
		NWARN("CAUTIOUS:  cellForRowAtIndexPath (TableView):  index out of range!?");
		return 0;
	}
#endif /* CAUTIOUS */
	const char **strings = SOB_SELECTORS(sop);

	c = [tableView dequeueReusableCellWithIdentifier: kCellIdentifier ];
	if (c == nil) {
		c = [[UITableViewCell alloc]
			initWithStyle:UITableViewCellStyleDefault
			reuseIdentifier:kCellIdentifier];
		//c.textLabel.lineBreakMode = UILineBreakModeWordWrap;
		c.textLabel.lineBreakMode = NSLineBreakByWordWrapping;
		c.textLabel.numberOfLines = 0;
	}

	//[c setTextLabel:  STRINGOBJ(strings[r]) ];
	UILabel *l;

	l = c.textLabel;
	l.text = STRINGOBJ(strings[r]);

	return c;
}

#endif // BUILD_FOR_IOS

static void clear_selection(Screen_Obj *sop, NSIndexPath *path )
{
#ifdef BUILD_FOR_IOS
	UITableView *tableView = (UITableView *) SOB_CONTROL(sop);
	[tableView deselectRowAtIndexPath:path animated:NO];
#endif // BUILD_FOR_IOS
}

// Call a function on all of the selections (in a multi-chooser)

static int path_iterate(Screen_Obj *sop, void (*func)() )
{
#ifdef BUILD_FOR_IOS
	UITableView *tableView = (UITableView *) SOB_CONTROL(sop);
	NSArray *multChoicePaths = [tableView indexPathsForSelectedRows];
	NSUInteger numMultChoices = [multChoicePaths count];

	for (int i=0; i<numMultChoices; i++) {
		NSIndexPath *path = [multChoicePaths objectAtIndex:i];
		(*func)(sop,path);
	}

	return (int) numMultChoices;
#else
	return 0;
#endif // ! BUILD_FOR_IOS
}


void clear_all_selections(Screen_Obj *sop)
{
	//int n;

	if( IS_CHOOSER(sop) ){	// chooser or mlt_chooser
		/*n =*/ path_iterate(sop,clear_selection);
	}
#ifdef CAUTIOUS
	  else {
		NWARN("CAUTIOUS:  clear_all_selections:  bad widget type!?");
	}
#endif // CAUTIOUS
}

String_Buf *choice_sbp=NULL;

#ifdef BUILD_FOR_IOS

static int n_choices;

static void add_selection_to_choice( Screen_Obj *sop, NSIndexPath *path )
{
	const char **strings;
	char varname[16];
	int r;

	strings = SOB_SELECTORS(sop);

	// We assume only one section
#ifdef BUILD_FOR_IOS
	r = (int) path.row;
#else
	r=0;	// BUG
#endif // BUILD_FOR_IOS

	if( sb_buffer(choice_sbp) == NULL ){
		copy_string(choice_sbp,strings[r]);
		n_choices = 1;
	} else {
		cat_string(choice_sbp,", ");
		cat_string(choice_sbp,strings[r]);
		n_choices ++;
	}
	sprintf(varname,"choice_%d",n_choices);	// BUG? use snprint?
	assign_var(DEFAULT_QSP_ARG varname, strings[r] );
}

// update the script variable $choice to be a comma-separated list
// of the selected choice strings...

static void updateMultipleChoices(Screen_Obj *sop)
{
	char val_string[16];
	int n;

	choice_sbp = new_stringbuf();

	n = path_iterate(sop,add_selection_to_choice);

	sprintf(val_string,"%d",n);	// BUG? use snprint?
	assign_var(DEFAULT_QSP_ARG "n_selections", val_string );

	if( sb_buffer(choice_sbp) == NULL )	// no selections
		assign_var(DEFAULT_QSP_ARG "choice", "(nothing_selected)" );
	else
		assign_var(DEFAULT_QSP_ARG "choice", sb_buffer(choice_sbp) );

	rls_stringbuf(choice_sbp);
	choice_sbp = NULL;
}

// multiple choice choosers also need to update when rows are deselected

- (void)tableView:(UITableView *)tableView didDeselectRowAtIndexPath:(NSIndexPath *)indexPath
{
	Screen_Obj *sop = find_any_scrnobj(tableView);
	CHECK_SCRNOBJ_VOID(sop,tableView,didDeselectRowAtIndexPath,1);

	if(SOB_TYPE(sop) == SOT_MLT_CHOOSER) {
		updateMultipleChoices(sop);
	}
}

- (void)tableView:(UITableView *)tableView
			didSelectRowAtIndexPath:(NSIndexPath *)indexPath
{
	Screen_Obj *sop=find_any_scrnobj(tableView);
	CHECK_SCRNOBJ_VOID(sop,tableView,didSelectRowAtIndexPath,1);

	int section = (int) indexPath.section;
	if( section != 0 ){
		NWARN("CAUTIOUS:  didSelectRowAtIndexPath:  unexpected section!?");
		return;
	}
	int row = (int) indexPath.row;	// assume just one section

#ifdef CAUTIOUS
	if( sop == NULL ){
		NWARN("CAUTIOUS:  didSelectRow (TableView):  couldn't find screen object!?");
		return;
	}
#endif // CAUTIOUS

	if( SOB_TYPE(sop) == SOT_CHOOSER || SOB_TYPE(sop) == SOT_MLT_CHOOSER){
#ifdef CAUTIOUS
		if( row < 0 || row >= SOB_N_SELECTORS(sop) ){
			NWARN("CAUTIOUS:  didSelectRow (TableView):  unexpected row index!?");
			return;
		}
#endif /* CAUTIOUS */
		if (SOB_TYPE(sop) == SOT_CHOOSER) {
			assign_var(DEFAULT_QSP_ARG "choice", SOB_SELECTORS(sop)[row] );
		} else { //SOB_TYPE(sop) == SOT_MLT_CHOOSER
			updateMultipleChoices(sop);
		}
	}
#ifdef CAUTIOUS
  else {
	NWARN("CAUTIOUS:  tableView didSelectRow: unexpected widget type!?");
	return;
	}
#endif /* CAUTIOUS */

	chew_text(DEFAULT_QSP_ARG  SOB_ACTION(sop), "(table selection event)");
} // end didSelectRowAtIndexPath

/******* picker (chooser) delegate support *********/

// returns the number of 'columns' to display.
- (NSInteger)numberOfComponentsInPickerView:(UIPickerView *)pickerView
{
	Screen_Obj *sop=find_any_scrnobj(pickerView);
	CHECK_SCRNOBJ_INT(sop,pickerView,numberOfComponentsInPickerView,1);

	if( SOB_TYPE(sop) == SOT_CHOOSER || SOB_TYPE(sop) == SOT_MLT_CHOOSER)
		return 1;
	else if( SOB_TYPE(sop) == SOT_PICKER )
		return SOB_N_CYLINDERS(sop);
	else {
		NWARN("CAUTIOUS:  numberOfComponentsInPickerView:  bad screen object!?");
		return 0;
	}
}

- (NSInteger)pickerView:(UIPickerView *)pickerView numberOfRowsInComponent:(NSInteger)component
{
	Screen_Obj *sop=find_any_scrnobj(pickerView);
	CHECK_SCRNOBJ_INT(sop,pickerView,numberOfRowsInComponent,1);

	if( SOB_TYPE(sop) == SOT_CHOOSER || SOB_TYPE(sop) == SOT_MLT_CHOOSER){
		// now access stored information in the object
#ifdef CAUTIOUS
		if( component != 0 ) {
			sprintf(DEFAULT_ERROR_STRING,
"CAUTIOUS:  numberOfRowsInComponent (PickerView):  component (%ld) should be 0 for a chooser!?",
				(long)component);
			NWARN(DEFAULT_ERROR_STRING);
			return 0;
		}
#endif // CAUTIOUS
		return SOB_N_SELECTORS(sop);
	} else if( SOB_TYPE(sop) == SOT_PICKER ){
#ifdef CAUTIOUS
		if( component < 0 || component > (SOB_N_CYLINDERS(sop)-1) ) {
			sprintf(DEFAULT_ERROR_STRING,
"CAUTIOUS:  numberOfRowsInComponent (PickerView):  component (%ld) out of range for picker %s!?",
				(long)component,SOB_NAME(sop));
			NWARN(DEFAULT_ERROR_STRING);
			return 0;
		}

#endif // CAUTIOUS
		int n= SOB_N_SELECTORS_AT_IDX(sop,component);
		return n;
	}
#ifdef CAUTIOUS
	else {
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  numberOfRowsInComponent:  Bad screen object type!?");
		NWARN(DEFAULT_ERROR_STRING);
		return 0;
	}
#else // ! CAUTIOUS
	return 0;	// shouldn't happen
#endif // ! CAUTIOUS
}

// A "picker" is iOS for what we have called a "chooser"

- (NSString *)pickerView:(UIPickerView *)pickerView titleForRow:(NSInteger)row forComponent:(NSInteger)component
{
	Screen_Obj *sop=find_any_scrnobj(pickerView);
	CHECK_SCRNOBJ_STR(sop,pickerView,titleForRow,1);

	if( SOB_TYPE(sop) == SOT_CHOOSER ){
#ifdef CAUTIOUS
		if( component != 0 ){
			sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  titleForRow (PickerView):  unexpected chooser component index %ld!?",
				(long)component);
			NWARN(DEFAULT_ERROR_STRING);
			return @"???";
		}
		if( row < 0 || row >= SOB_N_SELECTORS(sop) ){
			sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  titleForRow (PickerView):  unexpected chooser row index %ld!?",(long)row);
			NWARN(DEFAULT_ERROR_STRING);
			return @"???";
		}
#endif /* CAUTIOUS */
		return STRINGOBJ( SOB_SELECTORS(sop)[row] );
	} else if( SOB_TYPE(sop) == SOT_PICKER ){
		// Make sure component and row are OK
#ifdef CAUTIOUS
		if( component < 0 || component >= SOB_N_CYLINDERS(sop) ){
			sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  titleForRow (PickerView):  unexpected picker component index %ld!?",(long)row);
			NWARN(DEFAULT_ERROR_STRING);
			return @"???";
		}
		if( row < 0 || row >= SOB_N_SELECTORS_AT_IDX(sop, component) ){
			sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  titleForRow (PickerView):  unexpected picker row index %ld!?",(long)row);
			NWARN(DEFAULT_ERROR_STRING);
			return @"???";
		}
#endif /* CAUTIOUS */
		//const char *s=SOB_SELECTOR_AT_IDX(sop, component,row);
		const char ***tbl;
		tbl = SOB_SELECTOR_TBL(sop);
		const char **list;
		list = tbl[component];
		const char *s;
		s=list[row];
		return STRINGOBJ( s );

	}
	return @"???";
}

- (void)pickerView:(UIPickerView *)pickerView didSelectRow:(NSInteger)row inComponent:(NSInteger)component
{
	Screen_Obj *sop=find_any_scrnobj(pickerView);
	CHECK_SCRNOBJ_VOID(sop,pickerView,didSelectRow,1);

	if( SOB_TYPE(sop) == SOT_CHOOSER ){
//#ifdef CAUTIOUS
//		if( component != 0 ){
//			NWARN("CAUTIOUS:  titleForRow (PickerView):  unexpected component index!?");
//			return;
//		}
		assert( component == 0 );

//		if( row < 0 || row >= SOB_N_SELECTORS(sop) ){
//			NWARN("CAUTIOUS:  titleForRow (PickerView):  unexpected row index!?");
//			return;
//		}
//#endif /* CAUTIOUS */

		assert( row>=0 && row < SOB_N_SELECTORS(sop) );

		assign_var(DEFAULT_QSP_ARG "choice", SOB_SELECTORS(sop)[row] );
	} else if( SOB_TYPE(sop) == SOT_PICKER ){
		assert( component >= 0 && component < SOB_N_CYLINDERS(sop));
		assert(row>=0&&row<SOB_N_SELECTORS_AT_IDX(sop,component));

		char choice_idx[16];
		// We cast this because NSInteger can be int or long,
		// depending on platform.
		// (See Platform Dependencies in String Programming Guide)
		sprintf(choice_idx,"%ld",(long)component+1);
		assign_var(DEFAULT_QSP_ARG "choice_index", choice_idx );
		assign_var(DEFAULT_QSP_ARG "choice", SOB_SELECTOR_AT_IDX(sop,(int)component,(int)row) );
	}
#ifdef CAUTIOUS
	  else {
		  assert( ! "Unexpected widget type!?" );
	}
#endif /* CAUTIOUS */
	chew_text(DEFAULT_QSP_ARG  SOB_ACTION(sop), "(row selection event)");
} // end didSelectRow

/************** end of UIPicker delgate support */


/************** UITextView delgate support */

-(void) textViewDidChange:(UITextView *) tvp
{
	Screen_Obj *sop=find_any_scrnobj(tvp);
	//const char *old_content;
	char *new_content=NULL;

	CHECK_SCRNOBJ_VOID(sop,tvp,textViewDidChange,1);

	if( SOB_TYPE(sop) == SOT_EDIT_BOX ){
		if( SOB_FLAGS(sop) & SOF_KEYBOARD_DISMISSING )
			goto event_done;

		NSString *s=tvp.text;
		new_content = (char *)savestr(s.UTF8String);

		if( SOB_CONTENT(sop) != NULL ){
			//old_content=SOB_CONTENT(sop);
			rls_str(SOB_CONTENT(sop));
		} /* else {
			old_content="";
		}*/

		// BUG need to set variable here...
		assign_var(DEFAULT_QSP_ARG  "input_text",new_content);
	} else if( SOB_TYPE(sop) == SOT_TEXT || SOB_TYPE(sop) == SOT_TEXT_BOX ){
fprintf(stderr,"non-editable text box changed!?\n");
	}
#ifdef CAUTIOUS
	  else {
		sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  Unexpected screen object %s generated textViewDidChange callback!?",
			SOB_NAME(sop));
		NWARN(DEFAULT_ERROR_STRING);
	}
#endif /* CAUTIOUS */

	if( new_content != NULL )
		SET_SOB_CONTENT(sop, new_content );

event_done:

	return;
}

/************** end of UITextView delgate support */

#endif // BUILD_FOR_IOS


-(void) genericButtonAction : (id) caller
{
	// the caller arg should tell us which button we have -
	// we need to use it to find the Screen_Obj...
	//
	// In the case of a button, it is probably a UIButton???

	Screen_Obj *sop = find_any_scrnobj((/*UIControl*/ SOB_CTL_TYPE *)caller);
	CHECK_SCRNOBJ_VOID(sop,caller,genericButtonAction,1);

	/* now interpret the action */
	chew_text(DEFAULT_QSP_ARG  SOB_ACTION(sop), "(button event)");
}

-(void) genericSwitchAction : (id) caller
{
#ifdef BUILD_FOR_IOS
	if( ((UISwitch *)caller).on )
		assign_var(DEFAULT_QSP_ARG  "toggle_state","1");
	else
		assign_var(DEFAULT_QSP_ARG  "toggle_state","0");
#endif // BUILD_FOR_IOS

	[self genericButtonAction:caller];
}

#ifdef BUILD_FOR_MACOS
- (void) genericChooserAction:(id)sender
{
	Screen_Obj *sop = find_any_scrnobj((SOB_CTL_TYPE *)sender);

	// sop can be null because the NSButtonCells aren't sop's...
	CHECK_SCRNOBJ_VOID(sop,sender,genericChooserAction,0);

	NSMatrix *m = (NSMatrix *) sender;
	// Now find the selected cell
	NSButtonCell *c=m.selectedCell;
	int i;
	int choice_idx=(-1);
	NSArray *a=m.cells;
	for(i=0;i<SOB_N_SELECTORS(sop);i++){
		NSButtonCell *c2=[a objectAtIndex:i];
		if( c2 == c ){	// found it!
			choice_idx = i;
			i=SOB_N_SELECTORS(sop);	// break out
		}
	}
#ifdef CAUTIOUS
	if( choice_idx < 0 ){
		NWARN("CAUTIOUS:  genericChooserAction:  bad choice!?");
		return;
	}
#endif // CAUTIOUS

	const char *s=SOB_SELECTORS(sop)[choice_idx];
	assign_var(DEFAULT_QSP_ARG "choice", s );

	chew_text(DEFAULT_QSP_ARG  SOB_ACTION(sop), "(chooser event)");
}
#endif // BUILD_FOR_MACOS

- (void)genericSliderAction:(id)sender
{
	float value;
	Screen_Obj *sop = find_any_scrnobj((SOB_CTL_TYPE *)sender);
	CHECK_SCRNOBJ_VOID(sop,sender,genericSliderAction,1);

#ifdef BUILD_FOR_IOS
	UISlider *slider = (UISlider *)sender;

	value = slider.value;
#else
	NSSlider *slider = (NSSlider *)sender;
	value = slider.floatValue;
#endif // BUILD_FOR_IOS

	// The default range is between 0 and 1,
	// but we don't scale to the range of this slider
	// because we reset the range when we make the slider...

	sprintf(DEFAULT_ERROR_STRING,"%d",
		(int) round( value ) );
	assign_var(DEFAULT_QSP_ARG  "slider_val",DEFAULT_ERROR_STRING);

	SET_SOB_VAL(sop,(int)round(value));
	update_gauge_label(sop);

	/* now interpret the action */
	chew_text(DEFAULT_QSP_ARG  SOB_ACTION(sop), "(slider event)");
} // end genericSliderAction

#ifdef BUILD_FOR_IOS

// The variable DISPLAY is set (in the environment?)
// to /tmp/launch-NKKWcw/org.x:0
// which looks an awfully lot like X-windows,
// but we aren't using x11 here, so we discard
// it and reset it with a device name string...
// Then in order to be able to use this as an
// argument to the size functions, we create a dummy
// panel.  Kind of a kludge, because it will break
// it we use this name for other panel ops!?

-(void) getDevTypeForSize
{

	// Here we look at the size and decide what type of device we have
	//
	// We create the dummy panel so that we can use expressions
	// like ncols(iPad2)...

	// dev_size fields are float, can't switch
	int w = dev_size.width, h = dev_size.height;

	//Gen_Win *po;
	switch( w ){
		case 320:
			// iPod w/ retina display - 320 x 480???
			dev_type = DEV_TYPE_IPOD_RETINA;
			force_reserved_var(DEFAULT_QSP_ARG  "DISPLAY","iPod");
			// This is wrong...
			/*po=*/ dummy_panel(DEFAULT_QSP_ARG  "iPod",
				dev_size.width, dev_size.height);
			break;
		// What should we do if the app is launched with the device in landscape mode?
		case 768:
		case 1024:
			// ipad Simulator - iPad 2, non-retina?
			dev_type = DEV_TYPE_IPAD2;
			force_reserved_var(DEFAULT_QSP_ARG  "DISPLAY","iPad2");
			/*po=*/ dummy_panel(DEFAULT_QSP_ARG  "iPad2",
				dev_size.width, dev_size.height);
			break;
		case 2732:
ipad_pro_12_9:
			dev_type = DEV_TYPE_IPAD_PRO_12_9;
			force_reserved_var(DEFAULT_QSP_ARG  "DISPLAY","iPad_Pro_12_9");
			dummy_panel(DEFAULT_QSP_ARG  "iPad_Pro_12_9",
				dev_size.width, dev_size.height);
			break;
		case 1536:
ipad_pro_9_7:
			dev_type = DEV_TYPE_IPAD_PRO_9_7;
			force_reserved_var(DEFAULT_QSP_ARG  "DISPLAY","iPad_Pro_9_7");
			dummy_panel(DEFAULT_QSP_ARG  "iPad_Pro_9_7",
				dev_size.width, dev_size.height);
			break;
		case 2048:	// width
			switch( h ){
				case 2732:
					goto ipad_pro_12_9;
				case 1536:
					goto ipad_pro_9_7;
				default:
					dev_type = DEV_TYPE_DEFAULT;
					sprintf(DEFAULT_ERROR_STRING,
						"Unexpected display size %d x %d!?",h,w);
					NWARN(DEFAULT_ERROR_STRING);
					break;
			}
			break;
		default:
			dev_type = DEV_TYPE_DEFAULT;
			sprintf(DEFAULT_ERROR_STRING,
				"Unexpected view width %d!?\n",w);
			NWARN(DEFAULT_ERROR_STRING);
			break;
	}
fprintf(stderr,"getDevTypeForSize: dev size is %g (w) x %g (h)\n",dev_size.width,dev_size.height);  
}

int is_portrait(void)
{
	UIDevice *dev;
	static int warned=0;
	UIDeviceOrientation dev_ori;
	UIInterfaceOrientation ui_ori;
	int retval;

	dev = [UIDevice currentDevice];
	// Is this really necessary?
	//[dev beginGeneratingDeviceOrientationNotifications];
	dev_ori = [dev orientation];
	//[dev endGeneratingDeviceOrientationNotifications];

	switch( dev_ori ){
		case UIDeviceOrientationPortrait:
		case UIDeviceOrientationPortraitUpsideDown:
//advise("portrait!");
			retval = 1;
			break;
		case UIDeviceOrientationLandscapeLeft:
		case UIDeviceOrientationLandscapeRight:
//advise("landscape!");
			retval = 0;
			break;
		case UIDeviceOrientationUnknown:
			// This is returned when the startup script
			// is being read...
//fprintf(stderr,"is_portrait:  unknown orientation!?\n");
			retval = -1;
			break;
		case UIDeviceOrientationFaceUp:
		case UIDeviceOrientationFaceDown:
//fprintf(stderr,"is_portrait:  face up/down, using UI orientation...\n");
			ui_ori = [[UIApplication sharedApplication]
				statusBarOrientation];
			switch( ui_ori ){
				case UIInterfaceOrientationPortrait:
				case UIInterfaceOrientationPortraitUpsideDown:
					retval = 1;
					break;
				case UIInterfaceOrientationLandscapeLeft:
				case UIInterfaceOrientationLandscapeRight:
					retval = 0;
					break;
				// no other cases - include
				// a CAUTIOUS default?


				//This case is present in XCode 6, but not 5...
				case UIInterfaceOrientationUnknown:
				retval = 1;
				break;
			}
		default:
/*
fprintf(stderr,"is_portrait:  [UIDevice currentDevice] = 0x%lx\n",(long)dev);
if( dev != NULL )
fprintf(stderr,"is_portrait:  [dev orientation] = 0x%lx\n",(long)[dev orientation]);
*/
			// In the simulator, we get a value of 0???
if( dev != NULL && [dev orientation] == 0 && !warned ){
fprintf(stderr,"is_portrait:  orientation is 0!?\n");
warned=1;
}

			// does this ever happen???
			// It appears that it can happen
			// when the device is flat...
			// The previous orientation is remembered by the system.
			// How can we get it here???
			retval = -1;
			break;
	}
//fprintf(stderr,"is_portrait:  retval = %d\n",retval);
	return retval;
}
#else // ! BUILD_FOR_IOS

static int is_portrait(void) { return 0; }

#endif // ! BUILD_FOR_IOS

static const char *get_display_height(SINGLE_QSP_ARG_DECL)
{
	int h;
	static char hstr[16];

//#ifdef BUILD_FOR_IOS
//	if( is_portrait() )
//		h=(int)globalAppDelegate.dev_size.height;
//	else
//		h=(int)globalAppDelegate.dev_size.width;
//#else // ! BUILD_FOR_IOS
//	h=(int)globalAppDelegate.dev_size.height;
//#endif // ! BUILD_FOR_IOS

	h=(int)globalAppDelegate.dev_size.height;

	sprintf(hstr,"%d",h);
	return hstr;
}

// dev_size property doesn't seem to be set in cocoa!?

static const char *get_display_width(SINGLE_QSP_ARG_DECL)
{
	int w;
	static char wstr[16];

fprintf(stderr,"get_display_width:  globalAppDelegate.dev_size = %f (w) x %f (h)\n",
globalAppDelegate.dev_size.width,globalAppDelegate.dev_size.height);
//#ifdef BUILD_FOR_IOS
//	if( is_portrait() )
//		w=(int)globalAppDelegate.dev_size.width;
//	else
//		w=(int)globalAppDelegate.dev_size.height;
//#else // ! BUILD_FOR_IOS
//	w=(int)globalAppDelegate.dev_size.width;
//#endif // ! BUILD_FOR_IOS

//fprintf(stderr,"get_display_width:  w = %d, is_portrait = %d\n",w,is_portrait());

	w=(int)globalAppDelegate.dev_size.width;

	sprintf(wstr,"%d",w);
	return wstr;
}

static const char *prev_aspect=NULL;

static const char * get_device_aspect(SINGLE_QSP_ARG_DECL)
{
	const char *s;
	int status;

	if( prev_aspect == NULL ) prev_aspect="portrait";	// default

	if( (status=is_portrait()) < 0 )
		s=prev_aspect;	// not sure if this ever happens?
	else if( status )
		s="portrait";
	else
		s="landscape";

	prev_aspect=s;
	return s;
}

#ifdef BUILD_FOR_IOS

static const char * get_device_orientation(SINGLE_QSP_ARG_DECL)
{
	UIDevice *dev;
	const char *s;

	dev = [UIDevice currentDevice];

	switch( [dev orientation] ){
	case UIDeviceOrientationUnknown:
		s="unknown"; break;
	case UIDeviceOrientationPortrait:
		s="portrait"; break;
	case UIDeviceOrientationPortraitUpsideDown:
		s="upside down portrait"; break;
	case UIDeviceOrientationLandscapeLeft:
		s="landscape left"; break;
	case UIDeviceOrientationLandscapeRight:
		s="landscape right"; break;
	case UIDeviceOrientationFaceUp:
		s="face up"; break;
	case UIDeviceOrientationFaceDown:
		s="face down"; break;
	default:
		s="shouldn't happen!?!?"; break;
	}
	return s;
}

static void init_ios_device(void)
{
	UIDevice *dev;
	NSUUID *id;
	NSArray *parts;
	NSString *s;

	dev = [UIDevice currentDevice];
	// identifierForVendor throws an exception on iPad 5.0...
	// So first we check the sw version
	assign_var(DEFAULT_QSP_ARG  "ios_version",
		[dev systemVersion].UTF8String );

	parts=[[dev systemVersion]
		componentsSeparatedByCharactersInSet:
		[NSCharacterSet characterSetWithCharactersInString:@"."]];
	s = [parts objectAtIndex:0];

	version_major = s.intValue;
	assign_var(DEFAULT_QSP_ARG  "ios_version_major", s.UTF8String );

	if( parts.count > 1 ){
		s = [parts objectAtIndex:1];
		version_minor = s.intValue;
		assign_var(DEFAULT_QSP_ARG  "ios_version_minor", s.UTF8String );
	} else {
		version_minor=(-1);
		assign_var(DEFAULT_QSP_ARG  "ios_version_minor", "-1");
	}

	if( parts.count > 2 ){
		s = [parts objectAtIndex:2];
		version_release = s.intValue;
		assign_var(DEFAULT_QSP_ARG  "ios_version_release", s.UTF8String );
	} else {
		version_release=(-1);
		assign_var(DEFAULT_QSP_ARG  "ios_version_release", "-1");
	}

	// can get minor version and release with split('.')[0] etc...
	if( version_major >= 5 && version_minor > 0 ){
		id = [dev identifierForVendor];
		assign_var(DEFAULT_QSP_ARG  "device_uuid",id.UUIDString.UTF8String);
	} else {
		assign_var(DEFAULT_QSP_ARG  "device_uuid","(unknown)");
	}
}

- (BOOL)application:(QUIP_APPLICATION_TYPE *)application
	didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
	// Override point for customization after application launch.

	// This assignment used to be down lower, but we need it for console
	// initialization...
	globalAppDelegate = self;

	// Apparently the bounds always reflect portrait orientation?
	window = [[UIWindow alloc]
		initWithFrame:[[UIScreen mainScreen] bounds]];

	// The status bar shows "Carrier", the time and battery status at the
	// top of the screen...

	[[UIApplication sharedApplication]
		setStatusBarHidden:YES
		//setStatusBarHidden:NO
		withAnimation:UIStatusBarAnimationFade];

	dev_size = [[UIScreen mainScreen] bounds].size;
//fprintf(stderr,"didFinishLaunching:  dev_size = %f x %f\n",dev_size.width,dev_size.height);

	// This returns the same thing regardless of the device
	// orientation.  The dimensions correspond to portrait mode.
	// BUT WHY SHOULD THAT BE???
	init_dynamic_var(DEFAULT_QSP_ARG  "DISPLAY_WIDTH",get_display_width);
	init_dynamic_var(DEFAULT_QSP_ARG  "DISPLAY_HEIGHT",get_display_height);
	init_dynamic_var(DEFAULT_QSP_ARG  "DEVICE_ASPECT",get_device_aspect);


	init_dynamic_var(DEFAULT_QSP_ARG  "DEVICE_ORIENTATION",get_device_orientation);
	[self getDevTypeForSize];

	/* We need a root view controller before we can create
	 * the nav bar controller.
	 * But we would like to create the root view controller in the script...
	 */

	init_ios_device();	// set the uuid to a script var

	// now interpreter the startup file...
	// We'd like the startup file to define the navigation interface...

	// we can't call this thread synchronously, and then
	// have it call us back synchronously, or we will hang...

fprintf(stderr,"didFinishLaunchingWithOptions calling exec_quip...\n");
	exec_quip(SGL_DEFAULT_QSP_ARG);		// didFinishLaunching
fprintf(stderr,"back from exec_quip, application:  didFinishLaunchingWithOptions:\n");

	// this might return before doing all the commands if
	// there is an alert...

	/* We used to finish the initialization here, but
	 * that caused problems when we went to multiple dispatch
	 * queues...  If we have reciprocal synchronous calls
	 * between two queues, the whole system hangs.  So at
	 * least one has to be asynchronous.  If this call was
	 * asynchronous, then the checks below would fail...
	 */

// This needs to run after we have read the startup file
// and created the main nav panel.  Or else we need to add
// the commands to do these things to the startup script...


	/* Now make sure that we created a root view controller */

	if( first_quip_controller == NULL )
		NERROR1("No navigation controller declared in startup file!?");

	// Should we insist upon a console also???

	root_view_controller = [[quipNavController alloc]
			initWithRootViewController:first_quip_controller];


	// This button is labelled 'Done'
	UIBarButtonItem *item = [[UIBarButtonItem alloc]
			initWithBarButtonSystemItem:UIBarButtonSystemItemDone
			target:first_quip_controller
			action:@selector(qtvcExitProgram)];

	first_quip_controller.navigationItem.rightBarButtonItem = item;

	self.window.backgroundColor = [UIColor blueColor];
	[self.window makeKeyAndVisible];

	[window setRootViewController:root_view_controller];

	init_ios_text_output();	// set function vectors for warnings etc.

	// If an alert occurred in the startup script,
	// the dismiss function won't have been caught by the proper view
	// controllers...  Is there any harm in calling this even
	// if no alert has occurred?

	// This causes the app to exit...
	//dismiss_quip_alert(NULL);

	check_deferred_alert(SGL_DEFAULT_QSP_ARG);

	if( xcode_debug )
		fprintf(stderr,
		"ADVISORY:  xcode_debug=1, set to 0 for production build...\n");
	else
		fprintf(stderr,
		"ADVISORY:  xcode_debug=0\nScript messages will not display to debug console after QuIP console is initialized...\n");

	return YES;

} // end didFinishLaunchingWithOptions


- (void)applicationWillResignActive:(UIApplication *)application
{
	/* Sent when the application is about to move
	 * from active to inactive state. This can occur
	 * for certain types of temporary interruptions
	 * (such as an incoming phone call or SMS message)
	 * or when the user quits the application and it
	 * begins the transition to the background state.
	 * Use this method to pause ongoing tasks, disable
	 * timers, and throttle down OpenGL ES frame rates.
	 * Games should use this method to pause the game.
	 */

	/* We should set some sort of flag or var here,
	 * so that scripts that use alarms (e.g. pvt)
	 * can do the right thing...
	 */
	//SET_QS_FLAG_BITS(DEFAULT_QSP,QS_SUSPENDED);
	if( QLEVEL >= 0 ){
		SET_QS_FLAG_BITS(THIS_QSP,QS_HALTING);
		log_message("App will resign active.");
	}
}

- (void)applicationWillEnterForeground:(UIApplication *)application
{
	/* Called as part of the transition from the background
	 * to the inactive state; here you can undo many of the
	 * changes made on entering the background.
	 */
}

- (void)applicationDidBecomeActive:(UIApplication *)application
{
	/*
	 * Restart any tasks that were paused (or not yet started)
	 * while the application was inactive. If the application
	 * was previously in the background, optionally refresh the
	 * user interface.
	 */
	if( QLEVEL >= 0 ){
		if( IS_HALTING(DEFAULT_QSP) ){
			log_message("App became active.");
			resume_quip(SGL_DEFAULT_QSP_ARG);
		}
	}
}

- (void)applicationWillTerminate:(UIApplication *)application
{
	/* Called when the application is about to terminate.
	 * Save data if appropriate.
	 * See also applicationDidEnterBackground:.
	 */
	log_message("App will terminate.");
}

/*- (CMMotionManager *)motionManager
{
  if (!motionManager) motionManager = [[CMMotionManager alloc] init];
  return motionManager;
}*/

- (CMMotionManager *)motionManager
{
	CMMotionManager *mgr = nil;
	id appDelegate = [UIApplication sharedApplication].delegate;
	if ([appDelegate respondsToSelector:@selector(motionManager)]) {
		mgr = [appDelegate motionManager];
	}
	return mgr;
}

#ifdef FOOBAR		// part of old UIAccelerometerDelegate...

// This is the callback for the accelerometer

#define kFilteringFactor 0.1
static double accel[3]={0,0,0};

- (void)accelerometer:(CMAccelerometerData*)accelerometer
	 didAccelerate:(CMAcceleration*)acceleration
{
	// Use a basic low-pass filter to only keep the gravity
	// in the accelerometer values
	accel[0] = acceleration->x * kFilteringFactor + accel[0] * (1.0 - kFilteringFactor);
	accel[1] = acceleration->y * kFilteringFactor + accel[1] * (1.0 - kFilteringFactor);
	accel[2] = acceleration->z * kFilteringFactor + accel[2] * (1.0 - kFilteringFactor);

	sprintf(DEFAULT_ERROR_STRING,"accel:  %g %g %g	  %g %g %g\n",
	acceleration->x,acceleration->y,acceleration->z,
	accel[0],accel[1],accel[2]);
	NADVISE(DEFAULT_ERROR_STRING);

	//Update the accelerometer values for the view
	//[glView setAccel:accel];
}

#endif // FOOBAR

-(void) quip_wakeup
{

	[wakeup_timer removeFromRunLoop:[NSRunLoop currentRunLoop]
			forMode:NSDefaultRunLoopMode];

#ifdef THREAD_SAFE_QUERY
	// How do we know which qsp to wake?
	IOS_Node *np;

#ifdef CAUTIOUS
	if( wakeup_lp == NULL ){
		NWARN("CAUTIOUS:  quip_wakeup:  null wakeup list!?");
		return;
	}
#endif // CAUTIOUS

	np = IOS_LIST_HEAD(wakeup_lp);
	while( np != NULL ){
		Query_Stack *qsp;
		qsp = (__bridge Query_Stack *) IOS_NODE_DATA(np);
		resume_quip(qsp);
		np = IOS_NODE_NEXT(np);
	}
#else // ! THREAD_SAFE_QUERY
	resume_quip();
#endif // ! THREAD_SAFE_QUERY

}

#ifdef THREAD_SAFE_QUERY
-(void) request_wakeup:(Query_Stack *)qsp
{
	IOS_Node *np;

	if( wakeup_lp == NULL ){
		wakeup_lp == new_ios_list();
	}
	np = mk_ios_node(qsp);
	ios_addTail(np,wakeup_lp);
	// BUG?  should we check?
}
#endif // THREAD_SAFE_QUERY

-(void) insure_wakeup
{
	if( wakeup_timer == NULL ){
		wakeup_timer = [CADisplayLink
			displayLinkWithTarget:self
			selector:@selector(quip_wakeup)];
	}

	// We use the display refresh, but we could use a different
	// timer, like a one-shot???

	[wakeup_timer addToRunLoop:[NSRunLoop currentRunLoop]
			forMode:NSDefaultRunLoopMode];
}

#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS

@synthesize dev_size;
static NSString *applicationName=NULL;

- (BOOL)applicationWillFinishLaunching:(NSNotification *) notif
{
	globalAppDelegate = self;

//fprintf(stderr,"applicationWillFinishLaunching BEGIN\n");
	[self init_main_menu];
//fprintf(stderr,"applicationWillFinishLaunching DONE\n");
    return TRUE;
}

- (BOOL)applicationDidFinishLaunching:(NSNotification *) notif
{
//	globalAppDelegate = self;
//
//	[self init_main_menu];

	dev_size = [[NSScreen mainScreen] visibleFrame].size;
//fprintf(stderr,"dev_size initialized, %f x %f\n",dev_size.width,dev_size.height);

	// these things use the dev_size property
	init_dynamic_var(DEFAULT_QSP_ARG  "DISPLAY_WIDTH",get_display_width);
	init_dynamic_var(DEFAULT_QSP_ARG  "DISPLAY_HEIGHT",get_display_height);
	init_dynamic_var(DEFAULT_QSP_ARG  "DEVICE_ASPECT",get_device_aspect);


	// now interpreter the startup file...

	exec_quip(SGL_DEFAULT_QSP_ARG);	// applicationDidFinishLaunching
fprintf(stderr,"back from exec_quip, applicationDidFinishLaunching:\n");

	root_view_controller = [[quipNavController alloc]
			initWithRootViewController:first_quip_controller];

	check_deferred_alert(SGL_DEFAULT_QSP_ARG);

	if( xcode_debug )
		fprintf(stderr,
		"ADVISORY:  xcode_debug=1, set to 0 for production build...\n");
	else
		fprintf(stderr,
		"ADVISORY:  xcode_debug=0\nScript messages will not display to debug console after QuIP console is initialized...\n");


	return YES;
}

// These all have NSApp as the target - how can we suppress the unknown
// selector warnings???

EMPTY_SELECTOR(hide)
EMPTY_SELECTOR(terminate)
EMPTY_SELECTOR(unhideAllApplications)
EMPTY_SELECTOR(hideOtherApplications)
EMPTY_SELECTOR(orderFrontStandardAboutPanel)

-(void) populateApplicationMenu:(NSMenu *)aMenu
{
	NSMenuItem * menuItem;

	menuItem = [aMenu
			addItemWithTitle: [NSString stringWithFormat:@"%@ %@",
				NSLocalizedString(@"About", nil),
				applicationName]
			action:@selector(orderFrontStandardAboutPanel:)
			keyEquivalent:@""];

	[menuItem setTarget:NSApp];

	[aMenu addItem:[NSMenuItem separatorItem]];

	menuItem = [aMenu
			addItemWithTitle:
				NSLocalizedString(@"Preferences...", nil)
			action:NULL
			keyEquivalent:@","];

	[aMenu addItem:[NSMenuItem separatorItem]];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Services", nil)
			action:NULL
			keyEquivalent:@""];

	NSMenu * servicesMenu = [[NSMenu alloc] initWithTitle:@"Services"];
	[aMenu setSubmenu:servicesMenu forItem:menuItem];
	[NSApp setServicesMenu:servicesMenu];

	[aMenu addItem:[NSMenuItem separatorItem]];

	menuItem = [aMenu
			addItemWithTitle:[NSString stringWithFormat:@"%@ %@", NSLocalizedString(@"Hide", nil), applicationName]
			action:@selector(hide:)
			keyEquivalent:@"h"];

	[menuItem setTarget:NSApp];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Hide Others", nil)
			action:@selector(hideOtherApplications:)
			keyEquivalent:@"h"];
	[menuItem setKeyEquivalentModifierMask:NSCommandKeyMask | NSAlternateKeyMask];
	[menuItem setTarget:NSApp];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Show All", nil)
			action:@selector(unhideAllApplications:)
			keyEquivalent:@""];
	[menuItem setTarget:NSApp];

	[aMenu addItem:[NSMenuItem separatorItem]];

	menuItem = [aMenu
			addItemWithTitle:[NSString stringWithFormat:@"%@ %@", NSLocalizedString(@"Quit", nil), applicationName]
			action:@selector(terminate:)
			keyEquivalent:@"q"];
	[menuItem setTarget:NSApp];
}

-(void) populateDebugMenu:(NSMenu *)aMenu
{
	// TODO
}

EMPTY_SELECTOR(undo)
EMPTY_SELECTOR(redo)
EMPTY_SELECTOR(cut)
EMPTY_SELECTOR(copy)
EMPTY_SELECTOR(paste)
EMPTY_SELECTOR(pasteAsPlainText)
EMPTY_SELECTOR(delete)
EMPTY_SELECTOR(selectAll)

-(void) populateEditMenu:(NSMenu *)aMenu
{
	NSMenuItem * menuItem;

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Undo", nil)
			action:@selector(undo:)
			keyEquivalent:@"z"];
	menuItem.target = self;

//fprintf(stderr,"menuItem target = 0x%lx\n",(long)menuItem.target);

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Redo", nil)
			action:@selector(redo:)
			keyEquivalent:@"Z"];

	[aMenu addItem:[NSMenuItem separatorItem]];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Cut", nil)
			action:@selector(cut:)
			keyEquivalent:@"x"];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Copy", nil)
			action:@selector(copy:)
			keyEquivalent:@"c"];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Paste", nil)
			action:@selector(paste:)
			keyEquivalent:@"v"];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Paste and Match Style", nil)
			action:@selector(pasteAsPlainText:)
			keyEquivalent:@"V"];
	[menuItem setKeyEquivalentModifierMask:NSCommandKeyMask | NSAlternateKeyMask];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Delete", nil)
			action:@selector(delete:)
			keyEquivalent:@""];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Select All", nil)
			action:@selector(selectAll:)
			keyEquivalent:@"a"];

	[aMenu addItem:[NSMenuItem separatorItem]];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Find", nil)
			action:NULL
			keyEquivalent:@""];
	NSMenu * findMenu = [[NSMenu alloc] initWithTitle:@"Find"];
	[self populateFindMenu:findMenu];
	[aMenu setSubmenu:findMenu forItem:menuItem];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Spelling", nil)
			action:NULL
			keyEquivalent:@""];
	NSMenu * spellingMenu = [[NSMenu alloc] initWithTitle:@"Spelling"];
	[self populateSpellingMenu:spellingMenu];
	[aMenu setSubmenu:spellingMenu forItem:menuItem];
}

-(void) clearRecentDocuments : (id) sender
{
	[[NSDocumentController sharedDocumentController] clearRecentDocuments:sender];
}

EMPTY_SELECTOR(performClose)
EMPTY_SELECTOR(runPageLayout)
EMPTY_SELECTOR(print)

-(void) sheetDidEnd:(NSWindow *)sheet returnCode:(NSInteger) code
	contextInfo:(void *)info
{
	NSWindow *win;

	win = (__bridge NSWindow *) info;
//fprintf(stderr,"sheetDidEnd BEGIN\n");
	[sheet orderOut:self];
	// dismiss the window too?
//fprintf(stderr,"sheetDidEnd DONE\n");
}

#ifdef FOOBAR
	static NSWindow *win=NULL;

	//if( win == NULL ){
		NSRect r;
		r.origin.x = 100;
		r.origin.y = 100;
		r.size.width = 400;
		r.size.height = 400;	// BUG should remember user
					// drags and resizes...
		win = [[NSWindow alloc] initWithContentRect:r
			styleMask:NSTitledWindowMask
			backing:NSBackingStoreBuffered
			defer:NO ];
	//}

	[op beginSheetModalForWindow:win
		completionHander:
		^(NSInteger result){
			if( result == NSFileHandlingPanelOKButton ){
//fprintf(stderr,"Will open file!\n");
			} else if( result == NSFileHandlingPanelCancelButton ){
//fprintf(stderr,"Will NOT open any file.\n");
			}
#ifdef CAUTIOUS
			  else {
//fprintf(stderr,"CAUTIOUS:  unexpected file dialog result %ld!?\n",(long)result);
			}
#endif // CAUTIOUS

		}
		];
	// If we call this with a null window arg, then nothing happens.
	// If we create a single window and reuse it, nothing
	// happens the second time...
	[NSApp beginSheet: op
		modalForWindow: win
		modalDelegate: self
		didEndSelector: @selector(sheetDidEnd:returnCode:contextInfo:)
		contextInfo: (__bridge void *)(win)];
#endif // FOOBAR

static void chdir_to_file(const char *filename)
{
    int n;
    char *s, *t;
    
    n=(int)strlen(filename);
    s=getbuf(n+1);
    strcpy(s,filename);
    t=s+n;
    while( t!=s && *t != '/' ) t--;
    if( t != s ){
        *t=0;
        //fprintf(stderr,"will cd to %s\n",s);
        if( chdir(s) < 0 ){
            tell_sys_error("chdir");
            sprintf(ERROR_STRING,"Failed to chdir to %s",s);
            WARN(ERROR_STRING);
        }
    }
    givbuf(s);
}

// BUG it would be nice for this to work with arbitrary URLs...

static bool read_quip_file(const char *pathname)
{
	FILE *fp;
	fp = fopen(pathname,"r");
	if( ! fp ) {
		// Should we send up an alert here?
//fprintf(stderr,"Error opening file %s\n", pathname);
		return FALSE;
	} else {
		// Because scripts often redirect
		// to other files in the same directory,
		// it might make sense to set the directory here?
        	chdir_to_file(pathname);

		redir(DEFAULT_QSP_ARG  fp, pathname );
		exec_quip(SGL_DEFAULT_QSP_ARG);	// read_quip_file
fprintf(stderr,"back from exec_quip, read_quip_file\n");
		return TRUE;
	}
}

-(void) quipOpen : (id) sender
{
	NSOpenPanel *op = [NSOpenPanel openPanel];
	op.canChooseFiles = YES;
	op.canChooseDirectories = YES;

	[op beginWithCompletionHandler:
		^(NSInteger result){
			if( result == NSFileHandlingPanelOKButton ){
				NSURL *url;
				url = op.URL;
				if( read_quip_file(url.path.UTF8String) ){
					// add the file to the recent files menu
					[[NSDocumentController sharedDocumentController]
						noteNewRecentDocumentURL: url ];
							/*[NSURL fileURLWithPath:url.path]*/

					// BUG?  We might like to close
					// the open file dialog window
					// BEFORE we execute the script
					// We might like to execute the
					// script on a different thread,
					// but on iOS UI stuff has to happen
					// on the main thread, and the same
					// may be true for Cocoa.
				}
			} else if( result == NSFileHandlingPanelCancelButton ){
fprintf(stderr,"User cancel, will NOT open any file.\n");
			}
#ifdef CAUTIOUS
			  else {
fprintf(stderr,"CAUTIOUS:  unexpected file dialog result %ld!?\n",(long)result);
			}
#endif // CAUTIOUS

		}
		];
}

-(void) populateFileMenu:(NSMenu *)aMenu
{
	NSMenuItem * menuItem;

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"New", nil)
			action:NULL
			keyEquivalent:@"n"];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Open...", nil)
			action:@selector(quipOpen:)
			keyEquivalent:@"o"];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Open Recent", nil)
			action:NULL
			//action:@selector(quipOpen:)
			keyEquivalent:@""];

	NSMenu * openRecentMenu = [[NSMenu alloc] initWithTitle:@"Open Recent"];
	//NSMenu * openRecentMenu = [[NSMenu alloc] initWithTitle:@"NSRecentDocumentsMenu"];

	// This is a private method, so the compiler complains...
	// But, without this line, the thingies do not appear in the recent
	// files menu...
	[openRecentMenu performSelector:@selector(_setMenuName:) withObject:@"NSRecentDocumentsMenu"];

	[aMenu setSubmenu:openRecentMenu forItem:menuItem];

	menuItem = [openRecentMenu
			addItemWithTitle:NSLocalizedString(@"Clear Menu", nil)
			action:@selector(clearRecentDocuments:)
			keyEquivalent:@""];

	[aMenu addItem:[NSMenuItem separatorItem]];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Close", nil)
			action:@selector(performClose:)
			keyEquivalent:@"w"];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Save", nil)
			action:NULL
			keyEquivalent:@"s"];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Save As...", nil)
			action:NULL
			keyEquivalent:@"S"];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Revert", nil)
			action:NULL
			keyEquivalent:@""];

	[aMenu addItem:[NSMenuItem separatorItem]];

	menuItem = [aMenu
			addItemWithTitle:
				NSLocalizedString(@"Page Setup...", nil)
			action:@selector(runPageLayout:)
			keyEquivalent:@"P"];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Print...", nil)
			action:@selector(print:)
			keyEquivalent:@"p"];
}

EMPTY_SELECTOR(performFindPanelAction)
EMPTY_SELECTOR(centerSelectionInVisibleArea)

-(void) populateFindMenu:(NSMenu *)aMenu
{
	NSMenuItem * menuItem;

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Find...", nil)
			action:@selector(performFindPanelAction:)
			keyEquivalent:@"f"];

	[menuItem setTag:NSFindPanelActionShowFindPanel];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Find Next", nil)
			action:@selector(performFindPanelAction:)
			keyEquivalent:@"g"];
	[menuItem setTag:NSFindPanelActionNext];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Find Previous", nil)
			action:@selector(performFindPanelAction:)
			keyEquivalent:@"G"];
	[menuItem setTag:NSFindPanelActionPrevious];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Use Selection for Find", nil)
			action:@selector(performFindPanelAction:)
			keyEquivalent:@"e"];
	[menuItem setTag:NSFindPanelActionSetFindString];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Jump to Selection", nil)
			action:@selector(centerSelectionInVisibleArea:)
			keyEquivalent:@"j"];
}

EMPTY_SELECTOR(showHelp)	// BUG not needed, target is NSApp

-(void) populateHelpMenu:(NSMenu *)aMenu
{
	NSMenuItem * menuItem;

	menuItem = [aMenu addItemWithTitle:[NSString stringWithFormat:@"%@ %@", applicationName, NSLocalizedString(@"Help", nil)]
								action:@selector(showHelp:)
						keyEquivalent:@"?"];
	[menuItem setTarget:NSApp];
}

- (BOOL)application:(NSApplication *)theApplication
		   openFile:(NSString *)filename
{
fprintf(stderr,"openFile called, filename = %s\n",filename.UTF8String);
	// borrow code from quipOpen...
	read_quip_file(filename.UTF8String);
	return TRUE;
}


EMPTY_SELECTOR(showGuessPanel)
EMPTY_SELECTOR(checkSpelling)
EMPTY_SELECTOR(toggleContinuousSpellChecking)

-(void) populateSpellingMenu:(NSMenu *)aMenu
{
	NSMenuItem * menuItem;

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Spelling...", nil)
			action:@selector(showGuessPanel:)
			keyEquivalent:@":"];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Check Spelling", nil)
			action:@selector(checkSpelling:)
			keyEquivalent:@";"];

	menuItem = [aMenu
			addItemWithTitle:NSLocalizedString(@"Check Spelling as You Type", nil)
			action:@selector(toggleContinuousSpellChecking:)
			keyEquivalent:@""];
}

-(void) populateViewMenu:(NSMenu *)aMenu
{
	// TODO
}

EMPTY_SELECTOR(performMinimize)
EMPTY_SELECTOR(arrangeInFront)
EMPTY_SELECTOR(performZoom)


-(void) populateWindowMenu:(NSMenu *)aMenu
{
	NSMenuItem * menuItem;

	menuItem = [aMenu addItemWithTitle:NSLocalizedString(@"Minimize", nil)
								action:@selector(performMinimize:)
						keyEquivalent:@"m"];

	menuItem = [aMenu addItemWithTitle:NSLocalizedString(@"Zoom", nil)
								action:@selector(performZoom:)
						keyEquivalent:@""];

	[aMenu addItem:[NSMenuItem separatorItem]];

	menuItem = [aMenu addItemWithTitle:NSLocalizedString(@"Bring All to Front", nil)
								action:@selector(arrangeInFront:)
						keyEquivalent:@""];
}

EMPTY_SELECTOR(setAppleMenu)

- (void) addMenuToBar:(NSMenu *)mainMenu withName:(const char *)name
					withDesc:(const char *)desc
{
	NSMenuItem * menuItem;
	NSMenu * submenu;

	menuItem = [mainMenu
		addItemWithTitle:STRINGOBJ(name)
		action:NULL
		keyEquivalent:@""];
	submenu = [[NSMenu alloc]
		initWithTitle:NSLocalizedString(STRINGOBJ(name),
							STRINGOBJ(desc))];
	if( !strcmp(name,"Apple") ){
		[NSApp performSelector:@selector(setAppleMenu:)
			withObject:submenu];
		[self populateApplicationMenu:submenu];
	} else if( !strcmp(name,"File") )
		[self populateFileMenu:submenu];
	else if( !strcmp(name,"Edit") )
		[self populateEditMenu:submenu];
	else if( !strcmp(name,"View") )
		[self populateViewMenu:submenu];
	else if( !strcmp(name,"Window") ){
		[self populateWindowMenu:submenu];
		[NSApp setWindowsMenu:submenu];
	}
	else if( !strcmp(name,"Help") )
		[self populateHelpMenu:submenu];
	else if( !strcmp(name,"Debug") )
		[self populateDebugMenu:submenu];
	else {
		fprintf(stderr,"addMenuToBar:  unrecognized menu name '%s'!?\n",
			name);
	}

	[mainMenu setSubmenu:submenu forItem:menuItem];
}

-(void) populateMainMenu
{
	NSMenu * mainMenu = [[NSMenu alloc] initWithTitle:@"MainMenu"];

	applicationName=@"MacQuIP";

	// The titles of the menu items are for identification purposes
	// only and shouldn't be localized.
	//
	// The strings in the menu bar come from the submenu titles,
	// except for the application menu, whose title is ignored at runtime.

	[self addMenuToBar:mainMenu withName:"Apple" withDesc:"The Application menu" ];
	[self addMenuToBar:mainMenu withName:"File" withDesc:"The File menu" ];
	[self addMenuToBar:mainMenu withName:"Edit" withDesc:"The Edit menu" ];
	//[self addMenuToBar:mainMenu withName:"View" withDesc:"The View menu" ];
	[self addMenuToBar:mainMenu withName:"Window" withDesc:"The Window menu" ];
	[self addMenuToBar:mainMenu withName:"Help" withDesc:"The Help menu" ];
	//[self addMenuToBar:mainMenu withName:"Debug" withDesc:"The Debug menu" ];

	[NSApp setMainMenu:mainMenu];
}

-(void) init_main_menu
{
	[self populateMainMenu];

	NSMenu *mm = [NSApp mainMenu];
	long n = [mm numberOfItems];
//fprintf(stderr,"main menu has %ld items\n",n);
	int i;

	for(i=0;i<n;i++){
		NSMenuItem *_mi;
		_mi = [mm itemAtIndex:i];
		fprintf(stderr,"item %d at 0x%lx:  %s\n",i,(long)_mi,
			_mi.title.UTF8String);
	}

}

#endif // BUILD_FOR_MACOS

@end

