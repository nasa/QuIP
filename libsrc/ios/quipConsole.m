//
//  quipConsole.m
//
//  Created by Jeff Mulligan on 7/29/12.
//  Copyright 2012 __MyCompanyName__. All rights reserved.
//
#include "quip_config.h"

#import "quipAppDelegate.h"
#import "quipViewController.h"
#import "quipView.h"

#include "quip_prot.h"
#include "ios_gui.h"

// For now, this only affects console output
static int console_output_enabled=0;

// Set the dims for the console text box.
// Can override height later for input line...

void get_device_dims(Screen_Obj *sop)
{
	// Where do we set the size of the text box???
	switch( globalAppDelegate.dev_type ){
		case DEV_TYPE_IPAD2:
			SET_SOB_FONT_SIZE(sop, 20);
			SET_SOB_HEIGHT(sop,600);
			SET_SOB_WIDTH(sop,748);
			break;
		case DEV_TYPE_IPOD_RETINA:
			// this device is 320x480, we need to deduct
			// space for the input line, the nav bar
			// and buttons below...
			SET_SOB_FONT_SIZE(sop, 18);
			SET_SOB_HEIGHT(sop,300);
			SET_SOB_WIDTH(sop,300);
			break;
		default:
			SET_SOB_FONT_SIZE(sop, 15);
			SET_SOB_HEIGHT(sop,600);
			SET_SOB_WIDTH(sop,300);
sprintf(DEFAULT_ERROR_STRING,
	"get_device_dims:  unrecognized dev type %d!?",
	globalAppDelegate.dev_type);
NWARN(DEFAULT_ERROR_STRING);
			break;
	}
}

static int add_text_input( /* quipAppDelegate *adp , */ Panel_Obj *po, const char *name )
{
	Screen_Obj *sop;

	sop = simple_object(DEFAULT_QSP_ARG  name);
	if( sop == NO_SCREEN_OBJ ) NERROR1("Failed to create console input line");
	SET_SOB_ACTION(sop, "InterpretString" );
	SET_SOB_CONTENT(sop, "Enter command" );
	SET_SOB_TYPE(sop, SOT_TEXT);
	get_device_dims(sop);
	SET_SOB_HEIGHT(sop, 40);	// BUG use symbolic constant
	SET_SOB_WIDTH(sop, PO_WIDTH(po)-2*SIDE_GAP);

	make_text_field(DEFAULT_QSP_ARG  sop);
	add_to_panel(po,sop);

	/* setting initial value causes a callback,
	 * so we have to do that after add_to_panel...
	 */
	install_initial_text(sop);

	return WIDGET_VERTICAL_GAP+SOB_HEIGHT(sop);
}


static UITextView *msg_view=NULL;

static void display_text_fragment(QSP_ARG_DECL  const char *s)
{
	NSMutableString *str;
	NSString *t;


	// We also print to the debugger window
	// by using printf here...
	fputs(s,stderr);
	fflush(stderr);

	if( msg_view == NULL ) return;
	
	t = msg_view.text;

	NSString *s2;
	s2=STRINGOBJ(s);
	if( t == NULL )
		str=[[NSMutableString alloc] initWithString: s2 ];
	else {
		str=[[NSMutableString alloc] initWithString: msg_view.text];
//fprintf(stderr,"display_text_fragment calling appendString\n");

		// There have been some strange crashes with appendString
		// implicated with a null arg...
#ifdef CAUTIOUS
if( str == NULL || s2 == NULL ){
fprintf(stderr,
"CAUTIOUS:  display_text_fragment:  Null NSSTring!?\n\tstr = 0x%lx, s2 = 0x%lx\n",
(long)str,(long)s2);
abort();
} else {
#endif // CAUTIOUS
		[str appendString:s2];
#ifdef CAUTIOUS
}
#endif // CAUTIOUS
	}
	msg_view.text = str;
	int n= (int) str.length;
	[msg_view scrollRangeToVisible:NSMakeRange(n-1,0) ];
}

// formerly display_prt_msg_frag

static void ios_prt_msg_frag(QSP_ARG_DECL  const char *s)
{
	FILE *fp;
	if( (fp=tell_msgfile(SINGLE_QSP_ARG)) != stdout ){
		fputs(s,fp);
		fflush(fp);
		return;
	} else {	// output not redirected
		if( console_output_enabled )
			display_text_fragment(QSP_ARG  s);
	}
	if( xcode_debug )
		printf("%s",s);
}

#define PRINT_LINE(s,fp)	{fputs(s,fp);fputc('\n',fp);fflush(fp);}

// formerly display_advise

static void ios_advise(QSP_ARG_DECL  const char *s)
{
	FILE *fp;
	if( (fp=tell_errfile(SINGLE_QSP_ARG)) != stderr ){
		PRINT_LINE(s,fp);
	} else {	// send stderr to console
		if( console_output_enabled ){
			display_text_fragment(QSP_ARG  s);
			display_text_fragment(QSP_ARG  "\n");
		}
	}
	if( xcode_debug )
		PRINT_LINE(s,stderr);	// to xcode console
}

static const char *add_prefix_to_msg(const char *prefix,const char *msg)
{
	static char str[LLEN];	// BUG not thread safe,
				// also danger of buffer overrun...

	if( strlen(msg)+strlen(prefix) >= LLEN ){
		fprintf(stderr,"add_prefix_to_msg:  message too long \"%s\"\n",msg);
		// use snprintf?
		sprintf(str,"%sWarning message too long for buffer!?",
							prefix);
	} else {
		sprintf(str,"%s%s",prefix,msg);
	}
	return str;
}

static void ios_warn(QSP_ARG_DECL  const char *msg)
{
	const char *wstr;
fprintf(stderr,"ios_warn:  %s\n",msg);
	wstr = add_prefix_to_msg(WARNING_PREFIX,msg);
	ios_advise(wstr); // log to console or file, and possibly xcode console
	simple_alert(QSP_ARG  "WARNING",msg);
}

static void ios_error(QSP_ARG_DECL  const char *msg)
{
	const char *wstr;
	wstr = add_prefix_to_msg(ERROR_PREFIX,msg);
	ios_advise(wstr); // log to console or file, and possibly xcode console
	fatal_alert(QSP_ARG  msg);
}

// Send interpreter warnings to an alert pop-up
// and material destined for stdout and stderr goes to console
// (when enabled)

void init_ios_text_output()
{
	set_warn_func(ios_warn);
	set_error_func(ios_error);
	set_advise_func(ios_advise);
	set_prt_msg_frag_func(ios_prt_msg_frag);
}

static int add_text_output( /* quipAppDelegate *adp , */ Panel_Obj *po )
{
	Screen_Obj *sop;

	sop = simple_object(DEFAULT_QSP_ARG  CONSOLE_DISPLAY_NAME );
	if( sop == NO_SCREEN_OBJ ) NERROR1("Failed to create console display");
	SET_SOB_ACTION(sop, "" );	// action string is initial text to display
	//SET_SOB_CONTENT(sop, "" );
	SET_SOB_TYPE(sop, SOT_TEXT_BOX);
	add_to_panel(po,sop);
	/* put this after addition to panel list, because setting initial value causes a callback */
	get_device_dims(sop);
	SET_SOB_CONTENT(sop, savestr("system output:\n") );
	make_text_box(DEFAULT_QSP_ARG  sop, NO );

	msg_view = (UITextView *)SOB_CONTROL(sop);	// remember this for later

	// Now make the interpreter send its output here

	//set_advise_func(display_advise);
	//set_prt_msg_frag_func(display_prt_msg_frag);
	// warnings don't go to the console, the go to alert pop-ups...
	//	set_warn_func(display_warn);

	// what about PO_CURR_Y ??
	return WIDGET_VERTICAL_GAP + SOB_HEIGHT(sop);
}


static int add_clear_button( /* quipAppDelegate *adp , */ Panel_Obj *po )
{
	//quipView *qv;

	//qv = PO_QV(po);

	// make the button using the gui library...
	// Ultimately we will create the console in script...
	Screen_Obj *sop;

	sop = simple_object(DEFAULT_QSP_ARG  "Clear");
	SET_SOB_ACTION(sop,savestr("ClearConsole"));
	SET_SOB_FLAG_BITS(sop, SOT_BUTTON);

	make_button(DEFAULT_QSP_ARG  sop);
	add_to_panel(curr_panel,sop);

	return WIDGET_VERTICAL_GAP+SOB_HEIGHT(sop);
}

#ifdef NOT_USED
static int add_dismiss_button( /* quipAppDelegate *adp , */ Panel_Obj *po )
{
	quipView *qv;

	qv = PO_QV(po);

	// make the button using the gui library...
	// Ultimately we will create the console in script...
	Screen_Obj *sop;

	sop = simple_object(DEFAULT_QSP_ARG  "Dismiss");
	SET_SOB_ACTION(sop,savestr("DismissConsole"));
	SET_SOB_FLAG_BITS(sop, SOT_BUTTON);

	make_button(DEFAULT_QSP_ARG  sop);
	add_to_panel(curr_panel,sop);

	return WIDGET_VERTICAL_GAP+SOB_HEIGHT(sop);
}
#endif // NOT_USED

void addConsoleToQVC( quipViewController *qvc_p )
{
	Panel_Obj *po;

//NADVISE("adding console to quipViewController...");
	po = QVC_PO(qvc_p);
	curr_panel=po;

	push_scrnobj_context(DEFAULT_QSP_ARG  PO_CONTEXT(po));

	//[self attach_view:PO_QV(po)];

	[PO_QV(po) addDefaultBG];


	// causes the wait_fences warning!?
	SET_PO_CURR_X(po, 10);
	int y0=10;
	SET_PO_CURR_Y(po, y0);
	y0 += add_text_input(po,"Command input");
	SET_PO_CURR_Y(po, y0);
	y0 += add_text_output(/*qvc_p.qadp,*/po);
	SET_PO_CURR_Y(po, y0);
	y0 += add_clear_button(/*qvc_p.qadp,*/po);
	SET_PO_CURR_Y(po, y0);

//	y0 += add_dismiss_button(/*qvc_p.qadp,*/po);

	pop_scrnobj_context(SGL_DEFAULT_QSP_ARG);

	// We'd like to remember this...
	//globalAppDelegate.console_panel = po;
}

void enable_console_output(int yesno)
{
	console_output_enabled=yesno;
}

