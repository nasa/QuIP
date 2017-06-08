#include "quip_config.h"

/*
 * motif.c
 *
 * Motif-specific implemetation to support generic widgets
 */

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>		/* strdup */
#endif

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* free */
#endif

#include "quip_prot.h"
#include "data_obj.h"
#include "dispobj.h"
#include "my_motif.h"
#include "gui_prot.h"
#include "gen_win.h"
#include "screen_obj.h"
#include "panel_obj.h"
#include "nav_panel.h"
#include "xsupp.h"
#include "stack.h"

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_MOTIF
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
//static void post_func(Widget buttonID, XtPointer app_data, XtPointer widget_data);
static const char *the_dname;
static int dialog_y;
static Panel_Obj *last_panel=NO_PANEL_OBJ;
static Query_Stack *motif_qsp=NULL;
static void motif_dispatch(SINGLE_QSP_ARG_DECL);
#endif /* HAVE_MOTIF */



#include "cmaps.h"

//static XmStringCharSet char_set=XmSTRING_DEFAULT_CHARSET;




#ifdef THREAD_SAFE_QUERY
#define INIT_MOTIF_QSP		qsp = motif_qsp==NULL?motif_qsp=default_qsp:motif_qsp;
#else /* ! THREAD_SAFE_QUERY */
#define INIT_MOTIF_QSP
#endif /* ! THREAD_SAFE_QUERY */

Stack *nav_stack=NULL;

static Item_Type *nav_item_itp=NO_ITEM_TYPE;
ITEM_INIT_FUNC(Nav_Item,nav_item,0)
ITEM_NEW_FUNC(Nav_Item,nav_item)
ITEM_PICK_FUNC(Nav_Item,nav_item)
ITEM_DEL_FUNC(Nav_Item,nav_item)

static Item_Type *nav_panel_itp=NO_ITEM_TYPE;
ITEM_INIT_FUNC(Nav_Panel,nav_panel,0)
ITEM_NEW_FUNC(Nav_Panel,nav_panel)
ITEM_CHECK_FUNC(Nav_Panel,nav_panel)
ITEM_GET_FUNC(Nav_Panel,nav_panel)
ITEM_PICK_FUNC(Nav_Panel,nav_panel)

static Item_Type *nav_group_itp=NO_ITEM_TYPE;
ITEM_INIT_FUNC(Nav_Group,nav_group,0)
ITEM_NEW_FUNC(Nav_Group,nav_group)
ITEM_PICK_FUNC(Nav_Group,nav_group)
ITEM_DEL_FUNC(Nav_Group,nav_group)

#ifdef HAVE_MOTIF
static Screen_Obj *find_object(QSP_ARG_DECL  Widget obj)
{
	List *lp,*lp2;
	Node *np,*np2;
	Panel_Obj *po;

	/* otherwise check all panels */

	lp=panel_obj_list(SINGLE_QSP_ARG);
	if( lp==NULL )
{
WARN("no panel list");
 return(NO_SCREEN_OBJ);
}
	np=QLIST_HEAD(lp);
	while( np != NO_NODE ){
		po=(Panel_Obj *)np->n_data;
#ifdef QUIP_DEBUG
//if( debug ) fprintf(stderr,"Searching panel %s\n",PO_NAME(po));
#endif
		lp2=po->po_children;
		if( lp2 == NULL )
			WARN("null child list for panel!?");
		np2=QLIST_HEAD(lp2);
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

static void push_widget_context(QSP_ARG_DECL  Screen_Obj *sop)
{

	int n;
	Item_Context *icp;
	char *ctx_name;

	icp = current_scrnobj_context(SINGLE_QSP_ARG);
	assert( icp != NULL );
	n = 2 + strlen( CTX_NAME(icp) ) + strlen( SOB_NAME(sop) );
	ctx_name = getbuf(n);
	sprintf(ctx_name,"%s.%s",CTX_NAME(icp),SOB_NAME(sop) );
	icp = create_scrnobj_context(QSP_ARG  ctx_name );
	givbuf(ctx_name);
	push_scrnobj_context(QSP_ARG  icp);
}

#endif /* HAVE_MOTIF */

void reposition(Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
	/* Reposition frame if it has one */
	if(sop->so_frame){
		XtMoveWidget(sop->so_frame, sop->so_x, sop->so_y);
	} else {
		XtMoveWidget(sop->so_obj, sop->so_x, sop->so_y);
	}
#endif /* HAVE_MOTIF */
}

#ifdef HAVE_MOTIF
Panel_Obj *find_panel(QSP_ARG_DECL  Widget obj)
{
	List *lp;
	Node *np;
	Panel_Obj *po;

	lp=panel_obj_list(SINGLE_QSP_ARG);
	if( lp == NULL ) return(NO_PANEL_OBJ);
	np=QLIST_HEAD(lp);
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
	NWARN_ONCE("panel repaint not implemented in this version");
}
#endif /* HAVE_MOTIF */

/* make the panel use the given lutbuffer */

void panel_cmap(Panel_Obj *po,Data_Obj *cm_dp)
{
#ifdef HAVE_MOTIF
	Arg al[5];
	int ac = 0;

	SET_PO_CMAP_OBJ(po,cm_dp);
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
#endif /* HAVE_MOTIF */
}

void label_panel(Panel_Obj *po, const char *s)
{
#ifdef HAVE_MOTIF
	XStoreName(po->po_dpy, po->po_frame_obj->core.window, s);
#endif /* HAVE_MOTIF */
}

void make_panel(QSP_ARG_DECL  Panel_Obj *po,int width,int height)
{
#ifdef HAVE_MOTIF
	Arg al[64];
	int ac = 0;

//fprintf(stderr,"(motif.c) make_panel BEGIN\n");
	SET_PO_DOP(po, curr_dop());
#ifdef CAUTIOUS
	if( PO_DOP(po) == NO_DISP_OBJ )
		ERROR1("CAUTIOUS:  no display object");
#endif

	set_curr_win(PO_ROOTW(po));
	/* let the panel share a colormap? */

	/* set parameters for the "panel" (shell) to be created */
	XtSetArg(al[ac], XmNallowShellResize, FALSE); ac++;	/* used to be true!? */
	XtSetArg(al[ac], XmNx, po->po_x); ac++;
	XtSetArg(al[ac], XmNy, po->po_y); ac++;
	XtSetArg(al[ac], XmNwidth, width); ac++;
	XtSetArg(al[ac], XmNheight, height); ac++;

//fprintf(stderr,"make_panel calling XtAppCreateShell\n");
	po->po_frame_obj = (Widget) XtAppCreateShell(PO_NAME(po), "guimenu",
				applicationShellWidgetClass, display,
				al, ac);

	if( po->po_frame_obj == (Widget) NULL )
		ERROR1("error creating frame");

	ac = 0;
//fprintf(stderr,"make_panel calling XmNautoUnmanage\n");
	XtSetArg(al[ac], XmNautoUnmanage, FALSE); ac++;
//fprintf(stderr,"make_panel calling XmCreateForm\n");
	po->po_panel_obj = XmCreateForm(po->po_frame_obj, (String) NULL,
				al, ac);

	if( (Widget) po->po_panel_obj == (Widget) NULL )
		ERROR1("error creating panel");

//fprintf(stderr,"make_panel calling XtDisplay\n");
	po->po_dpy = XtDisplay(po->po_frame_obj);
	po->po_screen_no = DefaultScreen(po->po_dpy);
	po->po_gc = DefaultGC(po->po_dpy,DefaultScreen(po->po_dpy));
	po->po_visual = DefaultVisual(po->po_dpy,po->po_screen_no);

	/* indicate that the panel has not yet been realized */
	po->po_realized = 0;
	/* po->po_flags = 0; */		/* the caller did this already... */

//fprintf(stderr,"make_panel calling XtManageChild\n");
	XtManageChild(po->po_panel_obj);

	/* XXX unsupported until I figure this out */
	/*po->po_xwin = xv_get(po->po_frame_obj,XV_XID);*/

	/* this gives "incomplete type" compiler error */
	/* po->po_xwin = po->po_panel_obj->core.window; */

	po->po_xwin = XtWindow( po->po_panel_obj );
//fprintf(stderr,"XtWindow returned 0x%lx\n",(long)po->po_xwin);

#endif /* HAVE_MOTIF */

	/* For compatibility w/ iOS, we make a viewer with the same
	 * name that points to the same window...
	 */
} /* end make_panel */

#ifdef HAVE_MOTIF
void post_menu_handler (Widget w, XtPointer client_data,
	XButtonPressedEvent *event)
{
	Widget popup = (Widget) client_data;

	XmMenuPosition(popup, (XButtonPressedEvent *)event);
	XtManageChild (popup);
}

static Widget generic_frame(Widget parent, Screen_Obj *sop, int shadow_type)
{
	Arg al[10];
	int ac = 0;
	Widget frame, label;

	/* set orientation */
	XtSetArg(al[ac], XmNorientation, XmVERTICAL); ac++;
	/* set shadow type */
	// How can we specify NO shadow???
	XtSetArg(al[ac], XmNshadowType, shadow_type); ac++;
	XtSetArg(al[ac], XmNshadowThickness, 2); ac++;
	/* set geometry and placement */
//fprintf(stderr,"generic_frame:  top offset = %d\n",curr_panel->po_curry);
	XtSetArg(al[ac], XmNtopAttachment, XmATTACH_FORM); ac++;
	XtSetArg(al[ac], XmNtopOffset, curr_panel->po_curry); ac++;
	XtSetArg(al[ac], XmNbottomAttachment, XmATTACH_NONE); ac++;
	XtSetArg(al[ac], XmNleftAttachment, XmATTACH_FORM); ac++;
	XtSetArg(al[ac], XmNleftOffset, curr_panel->po_currx); ac++;
	XtSetArg(al[ac], XmNrightAttachment, XmATTACH_NONE); ac++;

	frame = XmCreateFrame(parent, (char *)SOB_NAME(sop), al, ac);
	XtManageChild(frame);

	/* Save object position */
	// BUG with nav_panels, this is messed up...
	if( curr_panel != NULL ){
		sop->so_x = curr_panel->po_currx;
		sop->so_y = curr_panel->po_curry;
	} else {
		NWARN("CAUTIOUS:  generic_frame:  current_panel is NULL!?");
	}

	ac = 0;
	XtSetArg(al[ac], XmNchildType, XmFRAME_TITLE_CHILD); ac++;

	//label = XmCreateLabelGadget(frame,sop->so_name, al, ac);
	label = XmCreateLabel(frame, (char *)sop->so_name, al, ac);
	XtManageChild(label);

	return(frame);
}
#endif /* HAVE_MOTIF */


#ifdef NOT_USED
static Widget simple_frame(Widget parent, Screen_Obj *sop, int shadow_type)
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
#endif /* NOT_USED */

#ifdef HAVE_MOTIF
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
 * Event callback function for `button1'.
 */
static void button_func(Widget buttonID, XtPointer app_data,
	XtPointer widget_data)
{
	Screen_Obj *sop;
	QSP_DECL

	INIT_MOTIF_QSP

	sop = find_object(QSP_ARG  buttonID);
	if( sop != NO_SCREEN_OBJ ){
		chew_text(DEFAULT_QSP_ARG sop->so_action_text,
						"(button event)");
	}
	else WARN("couldn't locate button");
}

#endif /* HAVE_MOTIF */



/* Creates Popup Menu, popup happens in post_menu_handler */
/* In the old code, pop-ups were only supported on SGI machines... */

void make_menu(QSP_ARG_DECL  Screen_Obj *sop, Screen_Obj *mip)
{
#ifdef HAVE_MOTIF
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
#endif /* HAVE_MOTIF */
}

void make_menu_choice(QSP_ARG_DECL  Screen_Obj *mip, Screen_Obj *parent)
{
#ifdef HAVE_MOTIF
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
#endif /* HAVE_MOTIF */
}

void make_pullright(QSP_ARG_DECL  Screen_Obj *mip, Screen_Obj *pr,
	Screen_Obj *parent)
{
	WARN_ONCE("menus not implemented in this version");
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

#ifdef HAVE_MOTIF
static void chooser_func(Widget buttonID, XtPointer app_data,	/* app_data should be NULL, see call to XtAddCallback */
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

	/* There is a special case where we re-select an already
	 * selected choice (which necessarily happens when there is
	 * only one choice).  Then we get two callbacks, which we may
	 * not want...
	 * We may be able to deal with that in the script...
	 */

	/*chew_text(DEFAULT_QSP_ARG sop->so_action_text); */
	ASSIGN_RESERVED_VAR("choice",sop->so_action_text);	/* BUG? action text? */

	/* Now we've found the button, how do we find the parent chooser? */

	sop = (Screen_Obj *)sop->so_parent;

#ifdef CAUTIOUS
	if( sop == NO_SCREEN_OBJ )
		ERROR1("CAUTIOUS:  chooser button with no parent!?");
#endif /* CAUTIOUS */

	chew_text(DEFAULT_QSP_ARG sop->so_action_text, "(chooser event)");
}
#endif /* HAVE_MOTIF */

#ifdef HAVE_MOTIF
static void toggle_func(Widget toggleID, XtPointer app_data,
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
			sprintf(ERROR_STRING,"toggle has a value of %d, expected 0 or 1!?",value);
			WARN(ERROR_STRING);
			value &= 1;
		}
		sprintf(val_str,"%d",value);
		ASSIGN_RESERVED_VAR("toggle_state",val_str);
		chew_text(DEFAULT_QSP_ARG sop->so_action_text,"(toggle event)");
	}
	else WARN("couldn't locate toggle button");
} // toggle_func

/* callback for text widgets - when is this called?  We know it is called when the window loses the focus, but
 * can we get a callback when a return is typed?
 *
 * This is supposed to be the value changed callback, but it doesn't seem to be called...
 *
 * this function gets handed to Xlib, so we can't add a qsp arg...
 * we solve it by having our screen objects have a qsp...
 */

static void text_func(Widget textID, XtPointer app_data, XtPointer widget_data )
{
	Screen_Obj *sop;
	const char *s;

	sop=find_object(DEFAULT_QSP_ARG  textID);
#ifdef CAUTIOUS
	if( sop == NO_SCREEN_OBJ ){
		NWARN("CAUTIOUS:  text_func:  couldn't locate text widget");
		return;
	}
#endif /* CAUTIOUS */

	s = get_text(sop);

	if( s == NULL )
		assign_reserved_var( SOB_QSP_ARG  "input_string","(null)");
	else {
		assign_reserved_var( SOB_QSP_ARG  "input_string",s);
		free((void *)s);
	}

	/* We should chew the text when a return is typed, or something? */
//NADVISE("text_func calling chew_text...");
	chew_text(DEFAULT_QSP_ARG sop->so_action_text,"(text event)");
} // text_func

/* this is supposed to be the losing focus callback */

static void text_func2(Widget textID, XtPointer app_data, XtPointer widget_data )
{
	Screen_Obj *sop;
	const char *s;
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
		ASSIGN_RESERVED_VAR("input_string","(null)");
	else {
		ASSIGN_RESERVED_VAR("input_string",s);
		free((void *)s);
	}

//NADVISE("text_func2 calling chew_text...");
	chew_text(DEFAULT_QSP_ARG sop->so_action_text,"(text2 event)");
} // text_func2
#endif /* HAVE_MOTIF */

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
#ifdef HAVE_MOTIF
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
#endif /* HAVE_MOTIF */
}

void make_toggle(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
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
#endif /* HAVE_MOTIF */
}

void make_text_field(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
	Arg al[5];
	int ac = 0;
	int w;

//fprintf(stderr,"make_text_field BEGIN\n");
	sop->so_frame = generic_frame(curr_panel->po_panel_obj,
		sop, XmSHADOW_IN);

	XtSetArg(al[ac], XmNeditable, TRUE); ac++;
	XtSetArg(al[ac], XmNrecomputeSize, FALSE); ac++;
	/* used to use full width of panel */
	//XtSetArg(al[ac], XmNwidth, PO_WIDTH(curr_panel)-30 ); ac++;
	/* Pick a width based on the length of the initial text - assume font width of 8 pixels? */

	w = 10+strlen(sop->so_content_text)*8;
//fprintf(stderr,"Using width of %d for string \"%s\"\n",w,sop->so_content_text);
	XtSetArg(al[ac], XmNwidth, w ); ac++;
	// Do we need to set the height explicitly???
	XtSetArg(al[ac], XmNheight, 32 ); ac++;

	XtSetArg(al[ac], XmNmaxLength, 40 ); ac++;

	// This generates a lot of warnings, not correct usage?
	//XtSetArg(al[ac], XmNvalue, (char *)sop->so_content_text ); ac++;

	sop->so_obj = XmCreateTextField(sop->so_frame, (char *)sop->so_name, al, ac);

	// The callback is called when we set the default value
	// if this is called after the callback is assigned.
	// So we set the value BEFORE assigning the callback...

	XmTextFieldSetString(sop->so_obj, (char *)sop->so_content_text );

//fprintf(stderr,"make_text_field:  obj = 0x%lx\n",(long)sop->so_obj);
//fprintf(stderr,"make_text_field:  adding text_func callback for value changed\n");
	XtAddCallback(sop->so_obj, XmNvalueChangedCallback, text_func, NULL);
	XtAddCallback(sop->so_obj, XmNlosingFocusCallback, text_func2, NULL);

	// the callback seems to be called right away?
	// Why?
	// This was commented out, but in that case the widget does not appear!?
	XtManageChild(sop->so_obj);

	// The callback is called when we set the default value
	// Let's try setting the value BEFORE assigning the callback...

	//XmTextFieldSetString(sop->so_obj, (char *)sop->so_content_text );

#endif /* HAVE_MOTIF */
}

void make_edit_box(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
	Arg al[8];
	int ac = 0;

	sop->so_frame = generic_frame(curr_panel->po_panel_obj,
		sop, XmSHADOW_IN);

	XtSetArg(al[ac], XmNeditable, TRUE); ac++;
	XtSetArg(al[ac], XmNrecomputeSize, FALSE); ac++;
	XtSetArg(al[ac], XmNwidth, PO_WIDTH(curr_panel)-30 ); ac++;
	/* BUG do something sensible about the height... */
	//XtSetArg(al[ac], XmNheight, PO_HEIGHT(curr_panel)-30 ); ac++;
#define EDIT_BOX_HEIGHT 100	// height of editable area?
#define EDIT_BOX_EXTRA 30	// add for total height - this value is guessed!?
	XtSetArg(al[ac], XmNheight, EDIT_BOX_HEIGHT); ac++;

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

	SET_SOB_HEIGHT(sop,EDIT_BOX_HEIGHT+EDIT_BOX_EXTRA);
#endif /* HAVE_MOTIF */
}

/* For a text widget, this sets the text!? */

void update_prompt(Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
	XmTextFieldSetString(sop->so_obj, (char *)sop->so_content_text);
#endif /* HAVE_MOTIF */
}

void update_text_field(Screen_Obj *sop, const char *string)
{
#ifdef HAVE_MOTIF
	XmTextFieldSetString(sop->so_obj, (char *)string );
#endif /* HAVE_MOTIF */
}

void update_edit_text(Screen_Obj *sop, const char *string)
{
#ifdef HAVE_MOTIF
	XmTextSetString(sop->so_obj, (char *)string );
#endif /* HAVE_MOTIF */
}

const char *get_text(Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
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
#endif /* HAVE_MOTIF */
	return "NO MOTIF!?";
}

#ifdef HAVE_MOTIF
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

	SET_SOB_HEIGHT(sop,SLIDER_HEIGHT);	// BUG where does this value come from?
}

#endif /* HAVE_MOTIF */

void make_gauge(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
	make_generic_slider(QSP_ARG  sop);

	/* ReadOnly */
	XtSetSensitive(sop->so_obj, FALSE);
#endif /* HAVE_MOTIF */
}

#ifdef HAVE_MOTIF

#define MAX_NUMBER_STRING_LEN	64

static void slider_func(Widget sliderID, XtPointer app_data,
	XtPointer widget_data)
{
	Screen_Obj *sop;
	char str[MAX_NUMBER_STRING_LEN];
	int value;
	QSP_DECL

	INIT_MOTIF_QSP

	sop = find_object(DEFAULT_QSP_ARG  sliderID);
	if( sop != NO_SCREEN_OBJ ){

		/* get the value from the slider */
		XmScaleGetValue(sliderID, &value);
		sprintf(str,"%d",value);			// BUG overrun?
		ASSIGN_RESERVED_VAR("slider_val",str);
		chew_text(DEFAULT_QSP_ARG sop->so_action_text,"(slider event)");
	} else ERROR1("can't locate slider");
} // slider_func
#endif /* HAVE_MOTIF */

void make_slider(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
	make_generic_slider(QSP_ARG  sop);

	XtAddCallback(sop->so_obj, XmNvalueChangedCallback, slider_func, NULL);
#endif /* HAVE_MOTIF */
}

#ifdef HAVE_MOTIF
static void make_width_slider(QSP_ARG_DECL  Screen_Obj *sop)
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
#endif /* HAVE_MOTIF */


void new_slider_range(Screen_Obj *sop, int xmin, int xmax)
{
#ifdef HAVE_MOTIF
	Arg al[10];
	int ac = 0;

	// BUG should we make sure it's really a slider or gauge?
	// BUG?  should we make sure the value is within the new range?
	sop->so_min = xmin;
	sop->so_max = xmax;
	if( sop->so_val < sop->so_min )
		sop->so_val = sop->so_min;
	if( sop->so_val > sop->so_max )
		sop->so_val = sop->so_max;

	XtSetArg(al[ac], XmNminimum, xmin); ac++;
	XtSetArg(al[ac], XmNmaximum, xmax); ac++;
	XtSetArg(al[ac], XmNvalue, sop->so_val); ac++;
	XtSetValues(sop->so_obj, al, ac);
#endif /* HAVE_MOTIF */
}


void new_slider_pos(Screen_Obj *sop, int val)
{
#ifdef HAVE_MOTIF
	Arg al[10];
	int ac = 0;

	XtSetArg(al[ac], XmNvalue, val); ac++;
	XtSetValues(sop->so_obj, al, ac);
#endif /* HAVE_MOTIF */
}

void set_toggle_state(Screen_Obj *sop, int val)
{
#ifdef HAVE_MOTIF
	Arg al[10];
	int ac = 0;

	XtSetArg(al[ac], XmNset, val); ac++;
	XtSetValues(sop->so_obj, al, ac);
#endif /* HAVE_MOTIF */
}

void make_adjuster(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
	make_generic_slider(QSP_ARG  sop);

	/* This doesn't seem to work? adjuster works like slider??? */
	XtAddCallback(sop->so_obj, XmNvalueChangedCallback, slider_func, NULL);
	XtAddCallback(sop->so_obj, XmNdragCallback, slider_func, NULL);
#endif /* HAVE_MOTIF */
}

void make_slider_w(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
	make_width_slider(QSP_ARG  sop);

	XtAddCallback(sop->so_obj, XmNvalueChangedCallback, slider_func, NULL);
#endif /* HAVE_MOTIF */
}


void make_adjuster_w(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
	make_width_slider(QSP_ARG  sop);

	XtAddCallback(sop->so_obj, XmNvalueChangedCallback, slider_func, NULL);
	XtAddCallback(sop->so_obj, XmNdragCallback, slider_func, NULL);
#endif /* HAVE_MOTIF */
}

void make_message(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
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
	// w = PO_WIDTH(curr_panel)-30;
	/* Pick a width based on the length of the initial text - assume font width of 8 pixels? */
	w = strlen(sop->so_content_text)*8;
	XtSetArg(al[ac], XmNwidth, w ); ac++;

	/* should this be the action text? */
	sop->so_obj = XmCreateLabel(sop->so_frame, (char *)sop->so_content_text, al, ac);
	XtManageChild(sop->so_obj);
	return;
#endif /* HAVE_MOTIF */
}


#ifdef HAVE_MOTIF
void delete_motif_widget(Screen_Obj *sop)
{
	// unmanageChild removes from panel but does not deallocate...
	XtUnmanageChild(sop->so_obj);

	if( SOB_FRAME(sop) != NULL )
		XtDestroyWidget(SOB_FRAME(sop));

	// destroy might handle unmanaging?
	//XtDestroyWidget(sop->so_obj);
}

COMMAND_FUNC( do_dispatch )
{
	motif_qsp = THIS_QSP;
	motif_dispatch(SINGLE_QSP_ARG);
}
#endif // HAVE_MOTIF

static void motif_dispatch(SINGLE_QSP_ARG_DECL)
{
#ifdef HAVE_MOTIF
	XtInputMask mask;

	while( (mask = XtAppPending(globalAppContext)) != 0)
		XtAppProcessEvent(globalAppContext, mask);
#endif // HAVE_MOTIF
}


#ifdef HAVE_MOTIF
static void navp_genwin_posn(QSP_ARG_DECL  const char *s, int x, int y)
{
	Nav_Panel *np_p;
	np_p=GET_NAV_PANEL(s);
	if( np_p != NO_NAV_PANEL ) {
		//SET_NAVP_X(np_p, x);
		//SET_NAVP_Y(np_p, y);
		SET_PO_X(NAVP_PANEL(np_p), x);
		SET_PO_Y(NAVP_PANEL(np_p), y);
		posn_panel(NAVP_PANEL(np_p));
	}
	return;
}

static void navp_genwin_show(QSP_ARG_DECL  const char *s)
{
	Nav_Panel *np_p;

	np_p=GET_NAV_PANEL(s);
	if( np_p != NO_NAV_PANEL ) show_panel(QSP_ARG  NAVP_PANEL(np_p));
	return;
}

static void navp_genwin_unshow(QSP_ARG_DECL  const char *s)
{
	Nav_Panel *np_p;

	np_p=GET_NAV_PANEL(s);
	if( np_p != NO_NAV_PANEL ) unshow_panel(QSP_ARG  NAVP_PANEL(np_p));
	return;
}

static void navp_genwin_delete(QSP_ARG_DECL  const char *s)
{
	Nav_Panel *np_p;
	np_p=GET_NAV_PANEL(s);
	if( np_p != NO_NAV_PANEL ) {
		WARN("sorry, don't know how to delete a nav_panel yet");
	}
	return;

}

static Genwin_Functions navp_genwin_funcs={
	navp_genwin_posn,
	navp_genwin_show,
	navp_genwin_unshow,
	navp_genwin_delete
};

void motif_init(QSP_ARG_DECL  const char *progname)
{
	const char *argv[1];
	int argc=1;

	argv[0]=progname;

	the_dname = check_display(SINGLE_QSP_ARG);

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

	add_event_func(QSP_ARG  motif_dispatch);

	// This is not really a motif-specific thing,
	// but we do this here because nav_panel_itp is
	// static to this file...
	if( nav_panel_itp == NO_ITEM_TYPE ){
		init_nav_panels(SINGLE_QSP_ARG);
		add_genwin(QSP_ARG  nav_panel_itp, &navp_genwin_funcs, NULL);
	}
}
#endif /* HAVE_MOTIF */

/* Updates the value of a gauge or slider */
void set_gauge_value(Screen_Obj *gp, int n)
{
#ifdef HAVE_MOTIF
	Arg al[5];
	int ac = 0;

	XtSetArg(al[ac], XmNvalue, n); ac++;
	XtSetValues(gp->so_obj, al, ac);
#endif /* HAVE_MOTIF */
}

/* Updates the value of a gauge or slider */
void set_gauge_label(Screen_Obj *gp, const char *s )
{
	NWARN("set_gauge_label:  Sorry, don't know how to do this for X11.");
}


void update_message(Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
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
#endif /* HAVE_MOTIF */
}

#ifdef HAVE_MOTIF
static int panel_mapped(Panel_Obj *po)
{

	XWindowAttributes attr;

	XGetWindowAttributes(po->po_dpy,
		po->po_frame_obj->core.window,&attr);

	if( attr.map_state != IsViewable ) return(0);
	return(1);
}
#endif /* HAVE_MOTIF */

void show_panel(QSP_ARG_DECL  Panel_Obj *po)
{
#ifdef HAVE_MOTIF
	if( PANEL_MAPPED(po) ){
		sprintf(ERROR_STRING,"show_panel:  panel %s is already mapped!?",PO_NAME(po));
		WARN(ERROR_STRING);
		return;
	}

	// On the mac (and linux???), when we unshow, and re-show a panel,
	// it moves down by the thickness of the top of the window???
	posn_panel(po);

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
		np=QLIST_HEAD(lp);
		while(np!=NO_NODE){
			sop=np->n_data;
			if( sop != NULL ){
				reposition(sop);
			}
			np=np->n_next;
		}
#endif /* FOOBAR */

	} else {
		XtMapWidget(po->po_frame_obj);
	}

	/* Now wait until it really is mapped */
#ifdef CAUTIOUS
	if( ! XtIsRealized(po->po_frame_obj) )
		NERROR1("CAUTIOUS:  show_panel:  object not realized!?");
#endif /* CAUTIOUS */
	/* get the window id */
	while( ! panel_mapped(po) )
		;
	po->po_flags |= PANEL_SHOWN;

#endif /* HAVE_MOTIF */
} /* end show_panel */

void unshow_panel(QSP_ARG_DECL  Panel_Obj *po)
{
#ifdef HAVE_MOTIF
	if( PANEL_UNMAPPED(po) ){
		sprintf(ERROR_STRING,"unshow_panel:  panel %s is not currently mapped!?",PO_NAME(po));
		WARN(ERROR_STRING);
		return;
	}

	XtUnmapWidget(po->po_frame_obj);
	po->po_flags &= ~PANEL_SHOWN;
#endif /* HAVE_MOTIF */
}

void posn_panel(Panel_Obj *po)
{
#ifdef HAVE_MOTIF
	Arg al[10];
	int ac = 0;

//fprintf(stderr,"posn_panel:  setting position of %s to %d %d\n",
//PO_NAME(po),po->po_x,po->po_y);
	XtSetArg(al[ac], XmNx, po->po_x); ac++;
	XtSetArg(al[ac], XmNy, po->po_y); ac++;
	XtSetValues(po->po_frame_obj, al, ac);
#endif /* HAVE_MOTIF */
}

void free_wsys_stuff(Panel_Obj *po)
{
	/* BUG - should check what we need to delete here */
}

void give_notice(const char **msg_array)
{
	NWARN_ONCE("give_notice:  notices not implemented in this version\n");
}

void set_std_cursor(void)
{
}

void set_busy_cursor(void)
{
}

#ifdef HAVE_MOTIF

#ifdef NOT_USED
void set_win_cursors(Widget which_cursor)
{
}
#endif /* NOT_USED */

static void scroller_func(Widget scrollerID, XtPointer app_data,
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

	ASSIGN_RESERVED_VAR("selection",selection);
	sop = find_object(DEFAULT_QSP_ARG  scrollerID);
	if( sop == NO_SCREEN_OBJ ) return;
	chew_text(DEFAULT_QSP_ARG sop->so_action_text,"(scroller event)");
} // scroller_func
#endif /* HAVE_MOTIF */

void make_scroller(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
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
#endif /* HAVE_MOTIF */

}

void set_scroller_list(Screen_Obj *sop, const char *string_list[],
	int nlist)
{
#ifdef HAVE_MOTIF
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
#endif /* HAVE_MOTIF */
}

void make_chooser(QSP_ARG_DECL  Screen_Obj *sop, int n, const char **stringlist)
{
#ifdef HAVE_MOTIF
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

#ifdef CAUTIOUS
	if( sop->so_children != NULL ){
		sprintf(ERROR_STRING,"CAUTIOUS:  Chooser %s already has a child list!?",SOB_NAME(sop));
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	SET_SOB_CHILDREN(sop,new_list());

	push_widget_context(QSP_ARG  sop);

	for(j=0; j<n; j++)
	{
		b_sop = simple_object(QSP_ARG  stringlist[j]);
		if( b_sop==NO_SCREEN_OBJ ) return;
		b_sop->so_action_text = savestr(stringlist[j]);
		b_sop->so_parent = sop;
		b_sop->so_flags |= SOT_MENU_ITEM;

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

	pop_scrnobj_context(SINGLE_QSP_ARG);

#endif /* HAVE_MOTIF */

	SET_SOB_HEIGHT(sop, CHOOSER_HEIGHT + CHOOSER_ITEM_HEIGHT*n );
}

// This is copied from make_chooser, and doesn't implement multiple components...
// For this to ever work, we need to change the callback.

void make_picker(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
	int	j;
	Arg	al[20];
	int	ac = 0;
	Screen_Obj *b_sop;	/* button ptr */
	char buf[6];
	int n;
	const char **stringlist;

	if( SOB_N_CYLINDERS(sop) != 1 ){
		sprintf(ERROR_STRING,"picker %s needs %d components, but we're only implementing 1!?",
			SOB_NAME(sop),SOB_N_CYLINDERS(sop));
		WARN(ERROR_STRING);
	}
	n= SOB_N_SELECTORS_AT_IDX(sop,/*component*/ 0 );

	sop->so_frame = generic_frame(curr_panel->po_panel_obj,
		sop, XmSHADOW_IN);

	XtSetArg(al[ac], XmNentryClass, xmToggleButtonWidgetClass); ac++;
	strcpy(buf,"name");
	sop->so_obj = XmCreateRadioBox(sop->so_frame,
		buf, al, ac);

	XtManageChild(sop->so_obj);

#ifdef CAUTIOUS
	if( sop->so_children != NULL ){
		sprintf(ERROR_STRING,"CAUTIOUS:  Picker %s already has a child list!?",SOB_NAME(sop));
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	SET_SOB_CHILDREN(sop,new_list());

	stringlist = SOB_SELECTORS_AT_IDX(sop,0);

	// The choices are created as screen_objs, we
	// need to create a special context for them so that we
	// can have the same choices in multiple pickers and choosers
	// The context name should be the concatenation of the current
	// scrnobj context, and the name of this widget...

	push_widget_context(QSP_ARG  sop);

	for(j=0; j<n; j++) {
		b_sop = simple_object(QSP_ARG  stringlist[j]);
		if( b_sop==NO_SCREEN_OBJ ) return;
		b_sop->so_action_text = savestr(stringlist[j]);
		b_sop->so_parent = sop;
		b_sop->so_flags |= SOT_MENU_ITEM;

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

	pop_scrnobj_context(SINGLE_QSP_ARG);

	SET_SOB_HEIGHT(sop, CHOOSER_HEIGHT + CHOOSER_ITEM_HEIGHT*n );
#endif /* HAVE_MOTIF */
}

void set_choice(Screen_Obj *sop,int i)
{
	Node *np;
	int n;
	Screen_Obj *bsop;

	n=0;

	if( SOB_CHILDREN(sop) == NULL ) return;	// if no motif

	np=QLIST_HEAD(SOB_CHILDREN(sop));
	while( np != NO_NODE ){
		bsop = (Screen_Obj *)NODE_DATA(np);
		if( n == i )
			set_toggle_state(bsop,1);
		else
			set_toggle_state(bsop,0);

		n++;
		np=NODE_NEXT(np);
	}
}

void window_cm(Panel_Obj *po,Data_Obj *cm_dp)
{
#ifdef HAVE_MOTIF
	if( CANNOT_SHOW_PANEL(po) ) return;

	install_colors(PO_DPYABLE(po));
#endif /* HAVE_MOTIF */
}

// This was introduced to help with iOS, where there are lots of different
// device sizes, we provide this now just to make sure we have some sort
// of semi-reasonable values.  We can always override it later.

void get_device_dims(Screen_Obj *sop)
{
	SET_SOB_WIDTH(sop,300);
	SET_SOB_HEIGHT(sop,50);
	SET_SOB_FONT_SIZE(sop, 15);
}

void clear_all_selections(Screen_Obj *sop)
{
	NWARN_ONCE("clear_all_selections:  not implemented!?");
}

void set_pick(Screen_Obj *sop, int cyl, int which )
{
	NWARN_ONCE("set_pick:  not implemented for motif!?");
}

void make_label(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
	Arg al[10];
	int ac = 0;
	int w;

	sop->so_frame = generic_frame( curr_panel->po_panel_obj, sop, XmSHADOW_IN);

	/* set alignment of label to be left-justified */
	XtSetArg(al[ac], XmNalignment, XmALIGNMENT_BEGINNING); ac++;

	/* message boxes resize to fit text stuck in there... */
	/* XtSetArg(al[ac], XmNallowShellResize, FALSE); ac++; */	/* does this work? no, does nothing */
	/* XtSetArg(al[ac], XmCAllowShellResize, FALSE); ac++; */ /* does this work? nothing */
	XtSetArg(al[ac], XmCNoResize, TRUE); ac++; /* does this work? nothing */

	XtSetArg(al[ac], XmNrecomputeSize, FALSE); ac++;

	/* Set widget width */

	/* Default behavior was to span the full width of the panel... */
	// w = PO_WIDTH(curr_panel)-30;
	/* Pick a width based on the length of the initial text - assume font width of 8 pixels? */
	w = strlen(sop->so_content_text)*8;
	XtSetArg(al[ac], XmNwidth, w ); ac++;

	/* should this be the action text? */
	sop->so_obj = XmCreateLabel(sop->so_frame, (char *)sop->so_content_text, al, ac);
	XtManageChild(sop->so_obj);
	return;
#endif /* HAVE_MOTIF */
}

void enable_widget(QSP_ARG_DECL  Screen_Obj *sop, int yesno)
{
#ifdef HAVE_MOTIF
	// ?? BUG? should we warn for redundant calls?
	if( yesno )
		XtSetSensitive(sop->so_obj,TRUE);
	else
		XtSetSensitive(sop->so_obj,FALSE);
#endif // HAVE_MOTIF
}

void hide_widget(QSP_ARG_DECL  Screen_Obj *sop, int yesno)
{
#ifdef HAVE_MOTIF
	// ?? BUG? should we warn for redundant calls?
	if( yesno )
		XtUnmapWidget(sop->so_obj);
	else
		XtMapWidget(sop->so_obj);
#endif // HAVE_MOTIF
}

// Stuff added to emulate iOS gui elements

void add_navitm_to_group(Nav_Group *ng_p,Nav_Item *ni_p)
{
}

void hide_nav_bar(QSP_ARG_DECL  int hide)
{
}

Item_Context *pop_navitm_context(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp;

	icp = pop_item_context(QSP_ARG  nav_item_itp);
	return icp;
}

void push_navitm_context(QSP_ARG_DECL  Item_Context *icp)
{
	push_item_context(QSP_ARG  nav_item_itp, icp );
}

Item_Context *pop_navgrp_context(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp;

	icp = pop_item_context(QSP_ARG  nav_group_itp);
	return icp;
}

void push_navgrp_context(QSP_ARG_DECL  Item_Context *icp)
{
	push_item_context(QSP_ARG  nav_group_itp, icp );
}

void delete_widget(QSP_ARG_DECL  Screen_Obj *sop)
{
#ifdef HAVE_MOTIF
	// get rid of motif stuff...
	delete_motif_widget(sop);
#endif // HAVE_MOTIF

	remove_from_panel(curr_panel,sop);
	del_so(QSP_ARG  sop);
}

// undoes the things done by create_nav_group

void remove_nav_group(QSP_ARG_DECL  Nav_Group *ng_p)
{
	// first remove the items
	Item_Context *icp;

fprintf(stderr,"remove_nav_group %s BEGIN\n",NAVGRP_NAME(ng_p));
	icp = NAVGRP_ITEM_CONTEXT(ng_p);
	assert(icp!=NULL);

fprintf(stderr,"removing item context for nav group %s\n",NAVGRP_NAME(ng_p));
	delete_item_context(QSP_ARG  icp);
fprintf(stderr,"DONE removing item context for nav group %s\n",NAVGRP_NAME(ng_p));

	// Now update the panel itself...
	delete_widget( QSP_ARG  NAVGRP_SCRNOBJ(ng_p) );
	//delete_widget( QSP_ARG  NAVGRP_SCRNOBJ(ng_p) );

//fprintf(stderr,"OOPS, need to delete screen_obj %s\n",SOB_NAME(NAVGRP_SCRNOBJ(ng_p)));
	remove_from_panel( NAVGRP_PANEL(ng_p), NAVGRP_SCRNOBJ(ng_p) );

	// remove from database
	del_nav_group(QSP_ARG  ng_p);
}

void remove_nav_item(QSP_ARG_DECL  Nav_Item *ni_p)
{
fprintf(stderr,"remove_nav_item %s BEGIN\n",NAVITM_NAME(ni_p));

	// need to remove from panel
	delete_widget(QSP_ARG  NAVITM_SCRNOBJ(ni_p) );

	del_nav_item(QSP_ARG  ni_p);
}

Item_Context *create_navitm_context(QSP_ARG_DECL  const char *name)
{
	if( nav_item_itp == NO_IOS_ITEM_TYPE ){
		init_nav_items(SINGLE_QSP_ARG);
		set_del_method(QSP_ARG  nav_item_itp, (void (*)(QSP_ARG_DECL  Item *))&remove_nav_item);
	}

	return create_item_context(QSP_ARG  nav_item_itp, name );
}

Item_Context *create_navgrp_context(QSP_ARG_DECL  const char *name)
{
	if( nav_group_itp == NO_IOS_ITEM_TYPE ){
		init_nav_groups(SINGLE_QSP_ARG);
	}

	return create_item_context(QSP_ARG  nav_group_itp, name );
}

Nav_Panel *create_nav_panel(QSP_ARG_DECL  const char *name)
{
	Nav_Panel *np_p;
#ifdef HAVE_MOTIF
	Panel_Obj *po;
#endif

//fprintf(stderr,"create_nav_panel %s BEGIN\n",name);

	np_p = new_nav_panel(QSP_ARG  name);
	if( np_p == NO_NAV_PANEL ){
		sprintf(ERROR_STRING,
"create_nav_panel:  error creating nav_panel \"%s\"!?",name);
		WARN(ERROR_STRING);
		return NO_NAV_PANEL;
	}
	SET_GW_TYPE( NAVP_GW(np_p), GW_NAV_PANEL );

#ifdef HAVE_MOTIF

// BUG on iOS the panel size defaults to the whole screen...

#define DEFAULT_NAV_PANEL_WIDTH		480
#define DEFAULT_NAV_PANEL_HEIGHT	720

	// Now make a regular panel...
	// new_panel is supposed to push a scrnobj context...
	po = new_panel(QSP_ARG  name, DEFAULT_NAV_PANEL_WIDTH, DEFAULT_NAV_PANEL_HEIGHT );
	if( po == NO_PANEL_OBJ ){
		WARN("Error creating panel for nav_panel!?");
		// BUG clean up (delete np_p)
		return NULL;
	}
	//np_p->np_po = po;
	SET_NAVP_PANEL(np_p,po);

	SET_GW_TYPE(PO_GW(po),GW_NAV_PANEL_OBJ);

#endif /* HAVE_MOTIF */

	IOS_Item_Context *icp;
	
	icp = create_navgrp_context(QSP_ARG  name );
	// We need to push the context, and pop when we finish?
	// We don't push until we enter the navigation submenu...

//fprintf(stderr,"create_nav_panel, pushing group context %s\n",
//CTX_NAME(icp));
//	PUSH_ITEM_CONTEXT(nav_group_itp, icp);
	SET_NAVP_GRP_CONTEXT(np_p, icp);

	icp = create_navitm_context(QSP_ARG  name );
//	PUSH_ITEM_CONTEXT(nav_item_itp, icp);
	SET_NAVP_ITM_CONTEXT(np_p, icp);

	// In motif, we don't have real nav panels (an iOS-ism)
	// So to emulate the functionality, we have to add
	// a "back" button...
	{
	Screen_Obj *bo;
	prepare_for_decoration(QSP_ARG  NAVP_PANEL(np_p) );

	// next 7 lines from get_parts, screen_objs.c
//fprintf(stderr,"create_nav_panel adding back button...\n");
	bo = simple_object(QSP_ARG  "Back");
	if( bo == NO_SCREEN_OBJ ){
		WARN("Error creating back button for nav_panel!?");
		goto no_back_button;
	}

	// Is pop_nav enough here???

	SET_SOB_ACTION(bo, savestr("Pop_Nav"));

	// next 6 lines from mk_button, screen_objs.c
	SET_SOB_TYPE(bo, SOT_BUTTON);

	make_button(QSP_ARG  bo);
	add_to_panel(curr_panel,bo);

	INC_PO_CURR_Y(curr_panel, BUTTON_HEIGHT + GAP_HEIGHT );



	unprepare_for_decoration(SINGLE_QSP_ARG);
	}
no_back_button:

	return np_p;
} // create_nav_panel

Nav_Group *create_nav_group(QSP_ARG_DECL  Nav_Panel *np_p, const char *name)
{
	Nav_Group *ng_p;
	int n;
	char *s;

	ng_p = new_nav_group(QSP_ARG  name);
	if( ng_p == NO_NAV_GROUP ){
		sprintf(ERROR_STRING,
"create_nav_group:  error creating nav_group \"%s\"!?",name);
		WARN(ERROR_STRING);
		return NO_NAV_GROUP;
	}

	// BUG initialize other stuff here...
	

	// need to init item context...
	Item_Context *icp;
	//icp = create_navgrp_context(QSP_ARG  name );
	
	// The context name needs to include the panel name,
	// so that group names can be repeated on different panels
	n = strlen(NAVP_NAME(np_p))+strlen(name)+2;
	s = getbuf(n);
	sprintf(s,"%s.%s",NAVP_NAME(np_p),name);

	icp = create_navitm_context(QSP_ARG  s );

	givbuf(s);

	SET_NAVGRP_ITEM_CONTEXT(ng_p,icp);

	return ng_p;
} // create_nav_group

#ifdef ONLY_IF_NEEDED
static void show_panel_stack(const char *s)
{
	Node *np;
	//Nav_Panel *np_p;
	Gen_Win *gwp;

	if( nav_stack == NULL ){
		fprintf(stderr,"show_panel_stack:  stack is uninitialized!?\n");
		return;
	}

	fprintf(stderr,"Panel stack:\n");
	np = STACK_TOP_NODE(nav_stack);
	while( np != NO_NODE ){
		gwp = NODE_DATA(np);
		//assert( GW_TYPE(NAVP_GW(np_p)) == GW_NAV_PANEL );
		switch( GW_TYPE(gwp) ){
			case GW_NAV_PANEL:
		fprintf(stderr,"\tnav_panel at 0x%lx, name at 0x%lx\n",
				(long)gwp,(long)GW_NAME(gwp));
				fflush(stderr);
				fprintf(stderr,"\t%s\n",GW_NAME(gwp));
				fflush(stderr);
				break;
			default:
				fprintf(stderr,
			"show_panel_stack, unhandled genwin type %d, at 0x%lx\n",
					GW_TYPE(gwp),(long)gwp);
				fprintf(stderr,"\t%s\n",GW_NAME(gwp));
				break;
		}
		np = NODE_NEXT(np);
	}
	fprintf(stderr,"\n");
}
#endif // ONLY_IF_NEEDED

void push_nav(QSP_ARG_DECL  Gen_Win *gwp)
{
	Gen_Win *current_gwp;

	// We can push a viewer or anything!?
	//assert( GW_TYPE(NAVP_GW(np_p)) == GW_NAV_PANEL );
//fprintf(stderr,"push_nav %s BEGIN\n",GW_NAME(gwp));
	if( nav_stack == NULL )
		nav_stack = new_stack();

	// We need to keep a stack of panels...
	if( (current_gwp=TOP_OF_STACK(nav_stack)) != NULL ){
//fprintf(stderr,"push_nav %s:  un-showing %s\n",GW_NAME(gwp),GW_NAME(current_gwp));
		unshow_genwin(QSP_ARG  current_gwp);
	}

	PUSH_TO_STACK(nav_stack,gwp);
//fprintf(stderr,"showing associated panel %s\n",PO_NAME(NAVP_PANEL(gwp)));
	show_genwin(QSP_ARG  gwp);
}

void pop_nav(QSP_ARG_DECL  int count)
{
	Gen_Win *gwp;

//fprintf(stderr,"pop_nav %d BEGIN\n",count);
	if( nav_stack == NULL )
		nav_stack = new_stack();

	gwp = POP_FROM_STACK(nav_stack);
	assert( gwp != NULL );
//fprintf(stderr,"pop_nav un-showing current top-of-stack %s\n",GW_NAME(gwp));
	unshow_genwin(QSP_ARG  gwp);

	count --;
	while( count -- ){
//fprintf(stderr,"pop_nav popping again, count = %d\n",count);
		gwp = POP_FROM_STACK(nav_stack);
		assert( gwp != NULL );
	}

//fprintf(stderr,"pop_nav done popping\n");
	assert( (gwp=TOP_OF_STACK(nav_stack)) != NULL );
	show_genwin(QSP_ARG  gwp);
}

void end_busy(int final)
{
	NWARN("end_busy:  not implemented!?");
}

void get_confirmation(QSP_ARG_DECL  const char *title, const char *question)
{
	//WARN("get_confirmation:  not implemented!?");
	ASSIGN_VAR("confirmed","1");
	fprintf(stderr,"ALERT:  get_confirmation:  confirming without use input...\n");
}

void simple_alert(QSP_ARG_DECL  const char *title, const char *msg)
{
	// ideally this should be a popup window that keeps the focus until
	// dismissed, but for now we just print the message to the console
	//WARN("simple_alert:  not implemented!?");
	fprintf(stderr,"%s:  %s\n",title,msg);
}

void notify_busy(QSP_ARG_DECL  const char *title, const char *msg)
{
	WARN("notify_busy:  not implemented!?");
}

int n_pushed_panels(void)
{
	return 0;
}

