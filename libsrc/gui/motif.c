#include "quip_config.h"

char VersionId_gui_motif[] = QUIP_VERSION_STRING;

#ifdef HAVE_MOTIF

/*
 * motif.c
 *
 * Drew Hess, NASA Ames Research Center
 * 4/27/94
 *
 * 11/96	P. Stassart - Added Motif code for:
 *		popup menu, chooser, scroller, text,
 *		slider (scale)
 *
 *	Motif-specific UI routines for proto
 */

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>		/* strdup */
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* free */
#endif

#include "data_obj.h"
#include "dispobj.h"
#include "gui.h"
#include "my_motif.h"	/* prototypes */
#include "items.h"
#include "debug.h"
#include "savestr.h"
#include "chewtext.h"
#include "xsupp.h"
#include "../interpreter/callback.h"

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_XM_XMALL_H
#include <Xm/XmAll.h>
#endif

/* this file is needed on sun... */
/* it also has the prototype for XtMoveWidget !? */

#include <X11/Intrinsic.h>
#include <X11/IntrinsicP.h>
#include <X11/StringDefs.h>

static XtAppContext globalAppContext;
static Display *display;
static const char *the_dname;
static void post_func(Widget buttonID, XtPointer app_data,
	XtPointer widget_data);
static int dialog_y;


#include "cmaps.h"

//static XmStringCharSet char_set=XmSTRING_DEFAULT_CHARSET;

static Panel_Obj *last_panel=NO_PANEL_OBJ;
static Screen_Obj *find_object(QSP_ARG_DECL  Widget obj);


static Query_Stream *motif_qsp=NULL;
static void motif_dispatch(SINGLE_QSP_ARG_DECL);

#ifdef THREAD_SAFE_QUERY
#define INIT_MOTIF_QSP		qsp = motif_qsp==NULL?motif_qsp=default_qsp:motif_qsp;
#else /* ! THREAD_SAFE_QUERY */
#define INIT_MOTIF_QSP
#endif /* ! THREAD_SAFE_QUERY */

void reposition(Screen_Obj *sop)
{
	/* Reposition frame if it has one */
	if(sop->so_frame){
		XtMoveWidget(sop->so_frame, sop->so_x, sop->so_y);
	} else {
		XtMoveWidget(sop->so_obj, sop->so_x, sop->so_y);
	}
}

Panel_Obj *find_panel(QSP_ARG_DECL  Widget obj)
{
	List *lp;
	Node *np;
	Panel_Obj *po;

	lp=panel_list(SINGLE_QSP_ARG);
	if( lp == NO_LIST ) return(NO_PANEL_OBJ);
	np=lp->l_head;
	while( np!=NO_NODE ){
		po = (Panel_Obj *)np->n_data;
		if( ((Widget)po->po_panel_obj) == obj ){
			return(po);
		}
		if( (Widget)po->po_pw == obj ){
			return(po);
		}
		np=np->n_next;
	}
	return(NO_PANEL_OBJ);
}

void panel_repaint(Widget panel,Widget pw)
{
	fprintf(stderr, "panel repaint not implemented in this version\n");
}

/* make the panel use the given lutbuffer */

void panel_cmap(Panel_Obj *po,Data_Obj *cm_dp)
{
	Arg al[5];
	int ac = 0;

	po->po_cm_dp = cm_dp;
/*	XtSetArg(
//			al[ac],
//			XmNcolormap,
//			(XtArgVal)( ((XlibData * )(lbp->lb_data)) ->xld_cmap ) );
//#define XtSetArg(arg, n, d)
//   ((void)( (arg).name = (n), (arg).value = (XtArgVal)(d) ))
*/

	al[ac].name = XmNcolormap;

	al[ac].value = (XtArgVal)( po->po_cmap );

	/* it is the first of the next two lines in the macro expansion of XtSetArg
	 * that generates compiler warnings...
	 */
	/* al[ac].name = XmNcolormap; */
	/* al[ac].value = (((XlibData *)(lbp->lb_data))->xld_cmap) ; */

	ac++;
	XtSetValues(po->po_frame_obj, al, ac);
}

void label_panel(Panel_Obj *po, const char *s)
{
	XStoreName(po->po_dpy, po->po_frame_obj->core.window, s);
}

void make_panel(QSP_ARG_DECL  Panel_Obj *po)
{
	Arg al[64];
	int ac = 0;

	po->po_dop = curr_dop();
#ifdef CAUTIOUS
	if( po->po_dop == NO_DISP_OBJ )
		ERROR1("CAUTIOUS:  no display object");
#endif


	set_curr_win(po->po_dop->do_rootw);
	/* let the panel share a colormap? */

	/* set parameters for the "panel" (shell) to be created */
	XtSetArg(al[ac], XmNallowShellResize, FALSE); ac++;	/* used to be true!? */
	XtSetArg(al[ac], XmNx, po->po_x); ac++;
	XtSetArg(al[ac], XmNy, po->po_y); ac++;
	XtSetArg(al[ac], XmNwidth, po->po_width); ac++;
	XtSetArg(al[ac], XmNheight, po->po_height); ac++;

	po->po_frame_obj = (Widget) XtAppCreateShell(po->po_name, "guimenu",
				applicationShellWidgetClass, display,
				al, ac);

	if( po->po_frame_obj == (Widget) NULL )
		ERROR1("error creating frame");

	ac = 0;
	XtSetArg(al[ac], XmNautoUnmanage, FALSE); ac++;
	po->po_panel_obj = XmCreateForm(po->po_frame_obj, (String) NULL,
				al, ac);

	if( (Widget) po->po_panel_obj == (Widget) NULL )
		ERROR1("error creating panel");

	po->po_dpy = XtDisplay(po->po_frame_obj);
	po->po_screen_no = DefaultScreen(po->po_dpy);
	po->po_gc = DefaultGC(po->po_dpy,DefaultScreen(po->po_dpy));
	po->po_visual = DefaultVisual(po->po_dpy,po->po_screen_no);

	/* indicate that the panel has not yet been realized */
	po->po_realized = 0;
	/* po->po_flags = 0; */		/* the caller did this already... */

	XtManageChild(po->po_panel_obj);

	/* XXX unsupported until I figure this out */
	/*po->po_xwin = xv_get(po->po_frame_obj,XV_XID);*/

	/* this gives "incomplete type" compiler error */
	/* po->po_xwin = po->po_panel_obj->core.window; */
} /* end make_panel */

void post_menu_handler (Widget w, XtPointer client_data,
	XButtonPressedEvent *event)
{
	Widget popup = (Widget) client_data;

	XmMenuPosition(popup, (XButtonPressedEvent *)event);
	XtManageChild (popup);
}

Widget generic_frame(Widget parent, Screen_Obj *sop, int shadow_type)
{
	Arg al[10];
	int ac = 0;
	Widget frame, label;

	/* set orientation */
	XtSetArg(al[ac], XmNorientation, XmVERTICAL); ac++;
	/* set shadow type */
	XtSetArg(al[ac], XmNshadowType, shadow_type); ac++;
	XtSetArg(al[ac], XmNshadowThickness, 4); ac++;
	/* set geometry and placement */
	XtSetArg(al[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
	XtSetArg(al[ac], XmNtopOffset, curr_panel->po_curry); ac++;
	XtSetArg(al[ac], XmNbottomAttachment, XmATTACH_NONE); ac++;
	XtSetArg(al[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
	XtSetArg(al[ac], XmNleftOffset, curr_panel->po_currx); ac++;
	XtSetArg(al[ac], XmNrightAttachment, XmATTACH_NONE); ac++;

	frame = XmCreateFrame(curr_panel->po_panel_obj, (char *)sop->so_name, al, ac);
	XtManageChild(frame);

	/* Save object position */
	sop->so_x = curr_panel->po_currx;
	sop->so_y = curr_panel->po_curry;

	ac = 0;
	XtSetArg(al[ac], XmNchildType, XmFRAME_TITLE_CHILD); ac++;
	//label = XmCreateLabelGadget(frame,sop->so_name, al, ac);
	label = XmCreateLabel(frame, (char *)sop->so_name, al, ac);
	XtManageChild(label);

	return(frame);
}


Widget simple_frame(Widget parent, Screen_Obj *sop, int shadow_type)
{
	Arg al[10];
	int ac = 0;
	Widget frame;

	/* set orientation */
	XtSetArg(al[ac], XmNorientation, XmVERTICAL); ac++;
	/* set shadow type */
	XtSetArg(al[ac], XmNshadowType, shadow_type); ac++;
	XtSetArg(al[ac], XmNshadowThickness, 0); ac++;
	/* set geometry and placement */
	XtSetArg(al[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
	XtSetArg(al[ac], XmNtopOffset, curr_panel->po_curry); ac++;
	XtSetArg(al[ac], XmNbottomAttachment, XmATTACH_NONE); ac++;
	XtSetArg(al[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
	XtSetArg(al[ac], XmNleftOffset, curr_panel->po_currx); ac++;
	XtSetArg(al[ac], XmNrightAttachment, XmATTACH_NONE); ac++;

	frame = XmCreateFrame(curr_panel->po_panel_obj,
	(char *)sop->so_name, al, ac);
	XtManageChild(frame);

	/* Save object position */
	sop->so_x = curr_panel->po_currx;
	sop->so_y = curr_panel->po_curry;

	return(frame);
}



/* Creates Popup Menu, popup happens in post_menu_handler */
/* In the old code, pop-ups were only supported on SGI machines... */

void make_menu(QSP_ARG_DECL  Screen_Obj *sop, Screen_Obj *mip)
{
	Arg al[10];
	int ac = 0;
	Widget post_button;
	char buf[8];

	sop->so_frame = generic_frame(curr_panel->po_panel_obj,
		sop, XmSHADOW_OUT);

	strcpy(buf,"Popup");	/* all this to elim a compiler warning */

#ifdef FOOBAR
	if(popups_supported)
	{
		/* XmCreatePopupMenu works on SGI and X, but not with
		OPENWIN */
		sop->so_obj = XmCreatePopupMenu(sop->so_frame,
			buf, al, ac);
		XtAddEventHandler(sop->so_frame,
			ButtonPressMask, FALSE,
			(XtEventHandler) post_menu_handler,
			(XtPointer)  sop->so_obj);
	}
	else
#endif /* FOOBAR */
	{
		dialog_y = 1;
		sop->so_obj = XmCreateBulletinBoardDialog(sop->so_frame,
			buf, al, ac);
		XtSetArg(al[ac], XmNuserData, sop); ac++;
		/* BUG?  should this be so_action_text??? */
		post_button = XmCreatePushButton(sop->so_frame,
			(char *)sop->so_action_text, al, ac);
		XtManageChild(post_button);
		XtAddCallback(post_button, XmNactivateCallback, post_func, NULL);
	}
}

void make_menu_choice(QSP_ARG_DECL  Screen_Obj *mip, Screen_Obj *parent)
{
	Arg al[5];
	int ac = 0;

	XtSetArg(al[ac], XmNy, dialog_y); ac++;
#ifdef FOOBAR
	if(popups_supported)
	{
		/* XmCreatePopupMenu works on SGI and X, but not with
		   OPENWIN */
		mip->so_obj = XtVaCreateManagedWidget(mip->so_name,
			xmPushButtonWidgetClass, parent->so_obj, NULL);
		XtAddCallback(mip->so_obj, XmNactivateCallback, button_func, NULL);
	}
	else
#endif /* FOOBAR */
	{
		mip->so_obj = XmCreatePushButton(parent->so_obj,
			(char *)mip->so_name, al, ac);
		XtManageChild(mip->so_obj);
		XtAddCallback(mip->so_obj, XmNactivateCallback, button_func, NULL);
		dialog_y += BUTTON_HEIGHT + GAP_HEIGHT + 15;
	}
}

void make_pullright(QSP_ARG_DECL  Screen_Obj *mip, Screen_Obj *pr,
	Screen_Obj *parent)
{
	fprintf(stderr, "menus not implemented in this version\n");
}

static void post_func(Widget buttonID, XtPointer app_data,
	XtPointer widget_data)
{
	Arg al[5];
	int ac = 0;
	Screen_Obj *sop;

	/* BUG?  do we really want to pass the address of sop here?
	 * XtSetArg doesn't know squat aobut ScreenObj's!? */

	XtSetArg(al[ac], XmNuserData, &sop);
	XtGetValues(buttonID, al, 1);

	XtManageChild(sop->so_obj);
}

/*
 * Event callback function for a chooser (radio buttons).
 *
 * It seems as if when we punch a new button, we get a callback for
 * a previously punched button that is now un-punched...
 *
 * This is called each time a widget value changes - so selecting
 * a new button generates TWO calls, first for the de-selected button,
 * then for the newly-selected button.
 */

void chooser_func(Widget buttonID, XtPointer app_data,	/* app_data should be NULL, see call to XtAddCallback */
	XtPointer widget_data)
{
	Screen_Obj *sop;
	XmToggleButtonCallbackStruct *cp;
	QSP_DECL

	INIT_MOTIF_QSP

	sop = find_object(QSP_ARG  buttonID);

	if( sop == NO_SCREEN_OBJ ) {
		WARN("couldn't locate chooser button");
		return;
	}

	cp = (XmToggleButtonCallbackStruct *)widget_data;

	/* For now, we don't support de-select functions, we handle that
	 * in the script handler.
	 */

	if( cp->set == 0 ) return;

	/*chew_text(DEFAULT_QSP_ARG sop->so_action_text); */
	ASSIGN_VAR("choice",sop->so_action_text);	/* BUG? action text? */

	/* Now we've found the button, how do we find the parent chooser? */

	sop = (Screen_Obj *)sop->so_parent;

#ifdef CAUTIOUS
	if( sop == NO_SCREEN_OBJ )
		ERROR1("CAUTIOUS:  chooser button with no parent!?");
#endif /* CAUTIOUS */

	chew_text(DEFAULT_QSP_ARG sop->so_action_text);
}

/*
 * Event callback function for `button1'.
 */
void button_func(Widget buttonID, XtPointer app_data,
	XtPointer widget_data)
{
	Screen_Obj *sop;
	QSP_DECL

	INIT_MOTIF_QSP

	sop = find_object(QSP_ARG  buttonID);
	if( sop != NO_SCREEN_OBJ ){
		chew_text(DEFAULT_QSP_ARG sop->so_action_text);
	}
	else WARN("couldn't locate button");
}

void toggle_func(Widget toggleID, XtPointer app_data,
	XtPointer widget_data)
{
	Screen_Obj *sop;
	unsigned  char value;
	Arg al[10];
	int ac = 0;
	char val_str[4];
	QSP_DECL

	INIT_MOTIF_QSP

	sop = find_object(QSP_ARG  toggleID);
	if( sop != NO_SCREEN_OBJ ){
		XtSetArg(al[ac], XmNset, &value);
		XtGetValues(toggleID, al, 1);

		if( value > 1 ){
			sprintf(error_string,"toggle has a value of %d, expected 0 or 1!?",value);
			WARN(error_string);
			value &= 1;
		}
		sprintf(val_str,"%d",value);
		ASSIGN_VAR("toggle_state",val_str);
		chew_text(DEFAULT_QSP_ARG sop->so_action_text);
	}
	else WARN("couldn't locate toggle button");
}

/* callback for text widgets - when is this called?  We know it is called when the window loses the focus, but
 * can we get a callback when a return is typed?
 *
 * This is supposed to be the value changed callback, but it doesn't seem to be called...
 *
 * this function gets handed to Xlib, so we can't add a qsp arg...
 * we solve it by having our screen objects have a qsp...
 */

void text_func(Widget textID, XtPointer app_data, XtPointer widget_data )
{
	Screen_Obj *sop;
	char *s;

	sop=find_object(DEFAULT_QSP_ARG  textID);
#ifdef CAUTIOUS
	if( sop == NO_SCREEN_OBJ ){
		NWARN("CAUTIOUS:  text_func:  couldn't locate text widget");
		return;
	}
#endif /* CAUTIOUS */

	s = get_text(sop);

	if( s == NULL )
		assign_var( SO_QSP_ARG  "input_string","(null)");
	else {
		assign_var( SO_QSP_ARG  "input_string",s);
		free(s);
	}

	/* We should chew the text when a return is typed, or something? */
	chew_text(DEFAULT_QSP_ARG sop->so_action_text);
}

/* this is supposed to be the losing focus callback */

void text_func2(Widget textID, XtPointer app_data, XtPointer widget_data )
{
	Screen_Obj *sop;
	char *s;
	QSP_DECL

	INIT_MOTIF_QSP

	sop=find_object(DEFAULT_QSP_ARG  textID);
#ifdef CAUTIOUS
	if( sop == NO_SCREEN_OBJ ){
		WARN("CAUTIOUS:  text_func:  couldn't locate text widget");
		return;
	}
#endif /* CAUTIOUS */

	s = get_text(sop);

	if( s == NULL )
		ASSIGN_VAR("input_string","(null)");
	else {
		ASSIGN_VAR("input_string",s);
		free(s);
	}

	chew_text(DEFAULT_QSP_ARG sop->so_action_text);
}

void make_separator(QSP_ARG_DECL  Screen_Obj *so)
{
#ifdef FOO
	Arg al[10];
	int ac = 0;

	/* set slider parameters */
	XtSetArg(al[ac], XmNx, curr_panel->po_currx); ac++;
	XtSetArg(al[ac], XmNy, curr_panel->po_curry); ac++;

	XtManageChild( XmCreateSeparator(so->so_obj,
		"separator",al,ac) );

	/* Save object position */
	sop->so_x = curr_panel->po_currx;
	sop->so_y = curr_panel->po_curry;
#endif /* FOO */
}

void make_button(QSP_ARG_DECL  Screen_Obj *sop)
{
	Arg al[10];
	int ac = 0;

	/* set geometry and placement */
	XtSetArg(al[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
	XtSetArg(al[ac], XmNtopOffset, curr_panel->po_curry); ac++;
	XtSetArg(al[ac], XmNbottomAttachment, XmATTACH_NONE); ac++;
	XtSetArg(al[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
	XtSetArg(al[ac], XmNleftOffset, curr_panel->po_currx); ac++;
	XtSetArg(al[ac], XmNrightAttachment, XmATTACH_NONE); ac++;

	sop->so_obj = XmCreatePushButton(curr_panel->po_panel_obj,
		(char *)sop->so_name, al, ac);

	/* Save object position */
	sop->so_x = curr_panel->po_currx;
	sop->so_y = curr_panel->po_curry;

	/* add callback function */
	XtAddCallback(sop->so_obj, XmNactivateCallback, button_func, NULL);

	/* manage the child */
	XtManageChild(sop->so_obj);
}

void make_toggle(QSP_ARG_DECL  Screen_Obj *sop)
{
	Arg al[10];
	int ac = 0;

	/* set geometry and placement */
	XtSetArg(al[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
	XtSetArg(al[ac], XmNtopOffset, curr_panel->po_curry); ac++;
	XtSetArg(al[ac], XmNbottomAttachment, XmATTACH_NONE); ac++;
	XtSetArg(al[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
	XtSetArg(al[ac], XmNleftOffset, curr_panel->po_currx); ac++;
	XtSetArg(al[ac], XmNrightAttachment, XmATTACH_NONE); ac++;

	sop->so_obj = XmCreateToggleButton(curr_panel->po_panel_obj,
		(char *)sop->so_name, al, ac);

	/* Save object position */
	sop->so_x = curr_panel->po_currx;
	sop->so_y = curr_panel->po_curry;

	/* add callback function */
	/*XtAddCallback(sop->so_obj, XmNactivateCallback, toggle_func, NULL); */
	XtAddCallback(sop->so_obj, XmNvalueChangedCallback, toggle_func, NULL);

	/* manage the child */
	XtManageChild(sop->so_obj);
}

void make_text_field(QSP_ARG_DECL  Screen_Obj *sop)
{
	Arg al[5];
	int ac = 0;
	int w;

	sop->so_frame = generic_frame(curr_panel->po_panel_obj,
		sop, XmSHADOW_IN);

	XtSetArg(al[ac], XmNeditable, TRUE); ac++;
	XtSetArg(al[ac], XmNrecomputeSize, FALSE); ac++;
	/* used to use full width of panel */
	//XtSetArg(al[ac], XmNwidth, curr_panel->po_top.c_width-30 ); ac++;
	/* Pick a width based on the length of the initial text - assume font width of 8 pixels? */
	w = 10+strlen(sop->so_content_text)*8;
	XtSetArg(al[ac], XmNwidth, w ); ac++;

	sop->so_obj = XmCreateTextField(sop->so_frame, (char *)sop->so_name, al, ac);

	XtAddCallback(sop->so_obj, XmNvalueChangedCallback, text_func, NULL);
	XtAddCallback(sop->so_obj, XmNlosingFocusCallback, text_func2, NULL);

	XtManageChild(sop->so_obj);

	XmTextFieldSetString(sop->so_obj, (char *)sop->so_content_text );
}

void make_edit_box(QSP_ARG_DECL  Screen_Obj *sop)
{
	Arg al[8];
	int ac = 0;

	sop->so_frame = generic_frame(curr_panel->po_panel_obj,
		sop, XmSHADOW_IN);

	XtSetArg(al[ac], XmNeditable, TRUE); ac++;
	XtSetArg(al[ac], XmNrecomputeSize, FALSE); ac++;
	XtSetArg(al[ac], XmNwidth, curr_panel->po_top.c_width-30 ); ac++;
	/* BUG do something sensible about the height... */
	XtSetArg(al[ac], XmNheight, curr_panel->po_top.c_height-30 ); ac++;

	XtSetArg(al[ac], XmNscrollHorizontal, FALSE); ac++;
	XtSetArg(al[ac], XmNwordWrap, TRUE); ac++;

	//sop->so_obj = XmCreateTextField(sop->so_frame, (char *)sop->so_name, al, ac);
	sop->so_obj = XmCreateScrolledText(sop->so_frame, (char *)sop->so_name, al, ac);

	XtAddCallback(sop->so_obj, XmNvalueChangedCallback, text_func, NULL);
	XtAddCallback(sop->so_obj, XmNlosingFocusCallback, text_func2, NULL);

	XtManageChild(sop->so_obj);

	/* this works for a TextField, but not ScrolledText!? */
	//XmTextFieldSetString(sop->so_obj, (char *)sop->so_content_text );

	/* this works! */
	XmTextSetString(sop->so_obj, (char *)sop->so_content_text );
}

/* For a text widget, this sets the text!? */

void update_prompt(Screen_Obj *sop)
{
	XmTextFieldSetString(sop->so_obj, (char *)sop->so_content_text);
}

void update_text_field(Screen_Obj *sop, const char *string)
{
	XmTextFieldSetString(sop->so_obj, (char *)string );
}

void update_edit_text(Screen_Obj *sop, const char *string)
{
	XmTextSetString(sop->so_obj, (char *)string );
}

char *get_text(Screen_Obj *sop)
{
	char *s, *p;
	int len;

	p = XmTextGetString(sop->so_obj);
	len = strlen(p);
	/* BUG?  memory leak?  when is the strdup string freed? */
	if (len ==  0)
		s = (char *) NULL;
	else
		s = strdup(p);

	XtFree(p);
	return s;
}

static void make_generic_slider(QSP_ARG_DECL  Screen_Obj *sop)
{
	Arg al[16];
	int ac = 0;
	Widget bb;
	char buf[4];

	sop->so_frame = generic_frame(curr_panel->po_panel_obj,
		sop, XmSHADOW_IN);

	/* create a bulletin board to hold the slider */
	strcpy(buf,"bb");	// elim compiler warning
	bb=XmCreateBulletinBoard(sop->so_frame,buf,NULL,0);
	XtManageChild(bb);

	/* set slider parameters */
	XtSetArg(al[ac], XmNorientation, XmHORIZONTAL); ac++;
	XtSetArg(al[ac], XmNshowValue, TRUE); ac++;
	XtSetArg(al[ac], XmNminimum, sop->so_min); ac++;
	XtSetArg(al[ac], XmNmaximum, sop->so_max); ac++;
	XtSetArg(al[ac], XmNvalue, sop->so_val); ac++;
	XtSetArg(al[ac], XmNprocessingDirection, XmMAX_ON_RIGHT); ac++;
	XtSetArg(al[ac], XmNdecimalPoints, 0); ac++;
	XtSetArg(al[ac], XmNscaleMultiple, 1); ac++;
	XtSetArg(al[ac], XmNscaleHeight, 15); ac++;
	XtSetArg(al[ac], XmNscaleWidth, 300); ac++;

	/* create slider */
	sop->so_obj = XmCreateScale(bb, (char *)sop->so_name, al, ac);

	/* arrange for slider to become visible */
	XtManageChild(sop->so_obj);
}

void make_gauge(QSP_ARG_DECL  Screen_Obj *sop)
{
	make_generic_slider(QSP_ARG  sop);

	/* ReadOnly */
	XtSetSensitive(sop->so_obj, FALSE);
}

void slider_func(Widget sliderID, XtPointer app_data,
	XtPointer widget_data)
{
	Screen_Obj *sop;
	char str[LLEN];
	int value;
	QSP_DECL

	INIT_MOTIF_QSP

	sop = find_object(DEFAULT_QSP_ARG  sliderID);
	if( sop != NO_SCREEN_OBJ ){

		/* get the value from the slider */
		XmScaleGetValue(sliderID, &value);
		sprintf(str,"%d",value);
		ASSIGN_VAR("slider_val",str);
		chew_text(DEFAULT_QSP_ARG sop->so_action_text);
	} else ERROR1("can't locate slider");
}

void make_slider(QSP_ARG_DECL  Screen_Obj *sop)
{
	make_generic_slider(QSP_ARG  sop);

	XtAddCallback(sop->so_obj, XmNvalueChangedCallback, slider_func, NULL);
}

void make_width_slider(QSP_ARG_DECL  Screen_Obj *sop)
{
	Arg al[16];
	int ac = 0;
	Widget bb;
	char buf[4];

	sop->so_frame = generic_frame(curr_panel->po_panel_obj,
		sop, XmSHADOW_IN);


	/* create a bulletin board to hold the slider */
	strcpy(buf,"bb");
	bb=XmCreateBulletinBoard(sop->so_frame,buf,NULL,0);
	XtManageChild(bb);

	/* set slider parameters */
	XtSetArg(al[ac], XmNorientation, XmHORIZONTAL); ac++;
	XtSetArg(al[ac], XmNshowValue, TRUE); ac++;
	XtSetArg(al[ac], XmNminimum, sop->so_min); ac++;
	XtSetArg(al[ac], XmNmaximum, sop->so_max); ac++;
	XtSetArg(al[ac], XmNvalue, sop->so_val); ac++;
	XtSetArg(al[ac], XmNprocessingDirection, XmMAX_ON_RIGHT); ac++;
	XtSetArg(al[ac], XmNdecimalPoints, 0); ac++;
	XtSetArg(al[ac], XmNscaleMultiple, 1); ac++;
	XtSetArg(al[ac], XmNscaleHeight, 15); ac++;
	XtSetArg(al[ac], XmNscaleWidth, sop->so_width); ac++;

	/* create slider */
	sop->so_obj = XmCreateScale(bb, (char *)sop->so_name, al, ac);

	/* arrange for slider to become visible */
	XtManageChild(sop->so_obj);
}


void new_slider_range(Screen_Obj *sop, int xmin, int xmax)
{
	Arg al[10];
	int ac = 0;

	XtSetArg(al[ac], XmNminimum, xmin); ac++;
	XtSetArg(al[ac], XmNmaximum, xmax); ac++;
	XtSetArg(al[ac], XmNvalue, sop->so_val); ac++;
	XtSetValues(sop->so_obj, al, ac);
}


void new_slider_pos(Screen_Obj *sop, int val)
{
	Arg al[10];
	int ac = 0;

	XtSetArg(al[ac], XmNvalue, val); ac++;
	XtSetValues(sop->so_obj, al, ac);
}

void new_toggle_state(Screen_Obj *sop, int val)
{
	Arg al[10];
	int ac = 0;

	XtSetArg(al[ac], XmNset, val); ac++;
	XtSetValues(sop->so_obj, al, ac);
}

void make_adjuster(QSP_ARG_DECL  Screen_Obj *sop)
{
	make_generic_slider(QSP_ARG  sop);

	XtAddCallback(sop->so_obj, XmNvalueChangedCallback, slider_func, NULL);
	XtAddCallback(sop->so_obj, XmNdragCallback, slider_func, NULL);
}

void make_slider_w(QSP_ARG_DECL  Screen_Obj *sop)
{
	make_width_slider(QSP_ARG  sop);

	XtAddCallback(sop->so_obj, XmNvalueChangedCallback, slider_func, NULL);
}


void make_adjuster_w(QSP_ARG_DECL  Screen_Obj *sop)
{
	make_width_slider(QSP_ARG  sop);

	XtAddCallback(sop->so_obj, XmNvalueChangedCallback, slider_func, NULL);
	XtAddCallback(sop->so_obj, XmNdragCallback, slider_func, NULL);
}

void make_message(QSP_ARG_DECL  Screen_Obj *sop)
{
	Arg al[10];
	int ac = 0;
	int w;

	sop->so_frame = generic_frame(curr_panel->po_panel_obj,
		sop, XmSHADOW_IN);

	/* set alignment of label to be left-justified */
	XtSetArg(al[ac], XmNalignment, XmALIGNMENT_BEGINNING); ac++;

	/* message boxes resize to fit text stuck in there... */
	/* XtSetArg(al[ac], XmNallowShellResize, FALSE); ac++; */	/* does this work? no, does nothing */
	/* XtSetArg(al[ac], XmCAllowShellResize, FALSE); ac++; */ /* does this work? nothing */
	XtSetArg(al[ac], XmCNoResize, TRUE); ac++; /* does this work? nothing */

	XtSetArg(al[ac], XmNrecomputeSize, FALSE); ac++;

	/* Set widget width */

	/* Default behavior was to span the full width of the panel... */
	// w = curr_panel->po_top.c_width-30;
	/* Pick a width based on the length of the initial text - assume font width of 8 pixels? */
	w = strlen(sop->so_content_text)*8;
	XtSetArg(al[ac], XmNwidth, w ); ac++;

	/* should this be the action text? */
	sop->so_obj = XmCreateLabel(sop->so_frame, (char *)sop->so_content_text, al, ac);
	XtManageChild(sop->so_obj);
	return;
}

COMMAND_FUNC( do_dispatch )
{
	motif_qsp = THIS_QSP;
	motif_dispatch(SINGLE_QSP_ARG);
}

static void motif_dispatch(SINGLE_QSP_ARG_DECL)
{
	XtInputMask mask;

	while( (mask = XtAppPending(globalAppContext)) != 0)
		XtAppProcessEvent(globalAppContext, mask);
}

void motif_init(const char *progname)
{
	const char *argv[1];
	int argc=1;

	argv[0]=progname;

	the_dname = check_display();

	/*
	 * initialize the Xt toolkit and create an application context
	 * and shell
	 */

	XtSetLanguageProc ( (XtAppContext) NULL, (XtLanguageProc) NULL, (XtPointer) NULL );
	XtToolkitInitialize ();
	globalAppContext = XtCreateApplicationContext ();
	display = XtOpenDisplay (globalAppContext, the_dname, argv[0],"guimenu",
			NULL, 0, &argc, (char **)argv);
	if (!display) {
		fprintf(stderr, "can't open display\n");
		return;
	}

	add_event_func(motif_dispatch);
}

/* Updates the value of a gauge or slider */
void set_gauge(Screen_Obj *gp, int n)
{
	Arg al[5];
	int ac = 0;

	XtSetArg(al[ac], XmNvalue, n); ac++;
	XtSetValues(gp->so_obj, al, ac);
}

void update_message(Screen_Obj *sop)
{
	Arg al[5];
	int ac = 0;
	XmString str;

	/* This version causes the window to be resized!? why??? */
	/* perhaps because the frame (shell) was created with AllowShellResize TRUE? */
	/* This has been fixed! */

	str=XmStringCreate((char *)sop->so_content_text,(char *)XmSTRING_DEFAULT_CHARSET);
	XtSetArg(al[ac], XmNlabelString,str);
	ac++;

	XtSetValues(sop->so_obj, al, ac);
	XmStringFree(str);
}

static int panel_mapped(Panel_Obj *po)
{

	XWindowAttributes attr;

	XGetWindowAttributes(po->po_dpy,
		po->po_frame_obj->core.window,&attr);

	if( attr.map_state != IsViewable ) return(0);
	return(1);
}

void show_panel(Panel_Obj *po)
{
	if( PANEL_MAPPED(po) ){
		sprintf(DEFAULT_ERROR_STRING,"show_panel:  panel %s is already mapped!?",po->po_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	/* if widget has already been realized, then just map it; otherwise
	   realize it and set the flag */
	if (!(po->po_realized))  {
		XtRealizeWidget(po->po_frame_obj);
		po->po_realized = 1;

#ifdef FOOBAR
		/* This must be the first time we see this thing - lets
		 * reset the positions of all the screen objects...
		 */
		lp=po->po_children;
		np=lp->l_head;
		while(np!=NO_NODE){
			sop=np->n_data;
			if( sop != NULL ){
				reposition(sop);
			}
			np=np->n_next;
		}
#endif /* FOOBAR */

	} else
		XtMapWidget(po->po_frame_obj);

	/* Now wait until it really is mapped */
#ifdef CAUTIOUS
	if( ! XtIsRealized(po->po_frame_obj) )
		NERROR1("CAUTIOUS:  show_panel:  object not realized!?");
#endif /* CAUTIOUS */
	/* get the window id */
	while( ! panel_mapped(po) )
		;
	po->po_flags |= PANEL_SHOWN;
} /* end show_panel */

void unshow_panel(Panel_Obj *po)
{
	if( PANEL_UNMAPPED(po) ){
		sprintf(DEFAULT_ERROR_STRING,"unshow_panel:  panel %s is not currently mapped!?",po->po_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}

	XtUnmapWidget(po->po_frame_obj);
	po->po_flags &= ~PANEL_SHOWN;
}

void posn_panel(Panel_Obj *po)
{
	Arg al[10];
	int ac = 0;

	XtSetArg(al[ac], XmNx, po->po_x); ac++;
	XtSetArg(al[ac], XmNy, po->po_y); ac++;
	XtSetValues(po->po_frame_obj, al, ac);
}

void free_wsys_stuff(Panel_Obj *po)
{
	/* BUG - should check what we need to delete here */
}

void give_notice(const char **msg_array)
{
	NWARN("give_notice:  notices not implemented in this version\n");
}

void init_cursors( void )
{
}

void set_win_cursors(Widget which_cursor)
{
}

void set_std_cursor(void)
{
}

void set_busy_cursor(void)
{
}

void scroller_func(Widget scrollerID, XtPointer app_data,
	XmListCallbackStruct *list_cbs)
{
	String selection;
	Screen_Obj *sop;
	QSP_DECL

	INIT_MOTIF_QSP

	/* list->cbs holds XmString of selected list item */
	/* map this to "ordinary" string */

	XmStringGetLtoR(list_cbs->item, (char *)XmSTRING_DEFAULT_CHARSET,
		&selection);

	ASSIGN_VAR("selection",selection);
	sop = find_object(DEFAULT_QSP_ARG  scrollerID);
	if( sop == NO_SCREEN_OBJ ) return;
	chew_text(DEFAULT_QSP_ARG sop->so_action_text);
}

void make_scroller(QSP_ARG_DECL  Screen_Obj *sop)
{
	Arg al[10];
	int ac = 0;
	char buf[10];

	sop->so_frame = generic_frame(curr_panel->po_panel_obj,
		sop, XmSHADOW_IN);

	XtSetArg(al[ac], XmNselectionPolicy, XmSINGLE_SELECT); ac++;
	XtSetArg(al[ac], XmNvisibleItemCount, N_SCROLLER_LINES); ac++;
	XtSetArg(al[ac], XmNscrollBarDisplayPolicy, XmSTATIC); ac++;
	XtSetArg(al[ac], XmNlistSizePolicy, XmCONSTANT); ac++;
	XtSetArg(al[ac], XmNwidth, SCROLLER_WIDTH); ac++;
	strcpy(buf,"Scroller");
	sop->so_obj = XmCreateScrolledList(sop->so_frame,
		buf, al, ac);
	XtAddCallback(sop->so_obj, XmNsingleSelectionCallback,
		/*(XtPointer)*/
		(void (*)( Widget, void *, void *))
		scroller_func, NULL);

}

void set_scroller_list(Screen_Obj *sop, const char *string_list[],
	int nlist)
{
	int	j;
	XmString item;

	/* first delete any old items */
	XmListDeselectAllItems(sop->so_obj);
	XmListDeleteAllItems(sop->so_obj);

	for(j=0; j<nlist; j++)
	{
		item = XmStringCreateSimple((char *)string_list[j]);
		XmListAddItem(sop->so_obj, item, 0);
		XmStringFree(item);
	}
	XtManageChild(sop->so_obj);
}

void make_chooser(QSP_ARG_DECL  Screen_Obj *sop, int n, const char **stringlist)
{
	int	j;
	Arg	al[20];
	int	ac = 0;
	Screen_Obj *b_sop;	/* button ptr */
	char buf[6];

	sop->so_frame = generic_frame(curr_panel->po_panel_obj,
		sop, XmSHADOW_IN);

	XtSetArg(al[ac], XmNentryClass, xmToggleButtonWidgetClass); ac++;
	strcpy(buf,"name");
	sop->so_obj = XmCreateRadioBox(sop->so_frame,
		buf, al, ac);

	XtManageChild(sop->so_obj);

	for(j=0; j<n; j++)
	{
		b_sop = simple_object(QSP_ARG  stringlist[j]);
		if( b_sop==NO_SCREEN_OBJ ) return;
		b_sop->so_action_text = savestr(stringlist[j]);
		b_sop->so_parent = sop;
		b_sop->so_flags |= SO_MENU_ITEM;

		/* The choices need to be part of the panel list (so we can find them
		 * with find_object), but also on the parent list, so we can find them
		 * through it...
		 */

		addHead(curr_panel->po_children,mk_node(b_sop));
		addTail(sop->so_children,mk_node(b_sop));

		b_sop->so_obj = XtCreateManagedWidget(
			b_sop->so_name,			/* widget name */
			xmToggleButtonWidgetClass,	/* widget class */
			sop->so_obj,			/* parent widget */
			NULL, 0);
		fix_names(QSP_ARG  b_sop,sop);
										/* client data */
		XtAddCallback(b_sop->so_obj, XmNvalueChangedCallback, chooser_func, NULL);
		XtManageChild(b_sop->so_obj);
	}
}

void window_cm(Panel_Obj *po,Data_Obj *cm_dp)
{
	if( CANNOT_SHOW_PANEL(po) ) return;

	install_colors(&po->po_top);
}

static Screen_Obj *find_object(QSP_ARG_DECL  Widget obj)
{
	List *lp,*lp2;
	Node *np,*np2;
	Panel_Obj *po;

	/* otherwise check all panels */

	lp=panel_list(SINGLE_QSP_ARG);
	if( lp==NO_LIST )
{
WARN("no panel list");
 return(NO_SCREEN_OBJ);
}
	np=lp->l_head;
	while( np != NO_NODE ){
		po=(Panel_Obj *)np->n_data;
#ifdef DEBUG
//if( debug ) fprintf(stderr,"Searching panel %s\n",po->po_name);
#endif
		lp2=po->po_children;
		if( lp2 == NO_LIST )
			WARN("null child list for panel!?");
		np2=lp2->l_head;
		while(np2!=NO_NODE ){
			Screen_Obj *sop;
			sop = (Screen_Obj *)np2->n_data;
			if( sop->so_obj == obj ){
				last_panel = po;
				return(sop);
			}
			np2=np2->n_next;
		}
		np=np->n_next;
	}
	return(NO_SCREEN_OBJ);
}

#endif /* HAVE_MOTIF */
