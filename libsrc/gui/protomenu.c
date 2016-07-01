#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif /* HAVE_STRING_H */

#include "quip_prot.h"
#include "view_cmds.h"
#include "gui_prot.h"		// push_scrnobj_context for X11...
#include "gen_win.h"		// push_nav
#include "gui_cmds.h"
#include "ios_gui.h"
#include "screen_obj.h"
#include "cmaps.h"		/* set_colormap() */

#ifdef HAVE_MOTIF
#include "my_motif.h"
#endif /* HAVE_MOTIF */

#ifndef BUILD_FOR_IOS
#include "xsupp.h"		/* which_display(), set_curr_win */
#endif /* BUILD_FOR_IOS */

static IOS_Item_Context *pushed_navitm_context=NULL;

static COMMAND_FUNC( do_panel_cmap )
{
	Panel_Obj *po;
	Data_Obj *cm_dp;

	po = PICK_PANEL("");
	cm_dp = PICK_OBJ("colormap object");

	if( po == NO_PANEL_OBJ || cm_dp == NO_OBJ )
		return;
	panel_cmap(po,cm_dp);
}

void prepare_for_decoration( QSP_ARG_DECL  Panel_Obj *pnl_p )
{
	curr_panel=pnl_p;
	if( pnl_p != NULL ){
		push_scrnobj_context(QSP_ARG  PO_CONTEXT(pnl_p));

#ifdef PO_XWIN
		/* what is PO_XWIN??? */
		set_curr_win( pnl_p->po_xwin );
		colormap( PO_CMAP_OBJ(pnl_p) );
#endif /* PO_XWIN */
	}
}

void unprepare_for_decoration(SINGLE_QSP_ARG_DECL)
{
	if( curr_panel != NULL ){
		pop_scrnobj_context(SINGLE_QSP_ARG);
	}
}

static COMMAND_FUNC( end_decorate )
{
	unprepare_for_decoration(SINGLE_QSP_ARG);
	pop_menu(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_hide_back )
{
	int hide;
	hide = ASKIF("hide back button");
#ifdef BUILD_FOR_IOS
	quipViewController *qvc;
	qvc= (quipViewController *)PO_VC(curr_panel);
	[ qvc hideBackButton:hide ];
#endif // BUILD_FOR_IOS
}


static COMMAND_FUNC( do_show_done )
{
	const char *s;

	s=NAMEOF("action for done button (or 'no' to hide)");

#ifdef BUILD_FOR_IOS
	quipViewController *qvc;
	qvc= (quipViewController *)PO_VC(curr_panel);
	[ qvc setDoneAction:s ];
#endif // BUILD_FOR_IOS
}

static COMMAND_FUNC(do_accept_edits)
{
	Screen_Obj *sop;

	sop=PICK_SCRNOBJ("edit box");
	if( sop == NO_SCREEN_OBJ ) return;

#ifdef BUILD_FOR_IOS
	dismiss_keyboard(sop);
#endif // BUILD_FOR_IOS

fprintf(stderr,"do_accept_edits calling chew_text...\n");
	chew_text(DEFAULT_QSP_ARG  SOB_ACTION(sop), SOB_FILENAME );
}

static COMMAND_FUNC( do_scrnobj_info )
{
	Screen_Obj *sop;

	sop=PICK_SCRNOBJ("widget name");
	if( sop == NO_SCREEN_OBJ ) return;

	switch(SOB_TYPE(sop)){
		case SOT_CHOOSER:
			sprintf(MSG_STR, "Chooser '%s':",SOB_NAME(sop));
			prt_msg(MSG_STR);
			sprintf(MSG_STR, "\t%d x %d",SOB_WIDTH(sop),SOB_HEIGHT(sop));
			prt_msg(MSG_STR);
			break;
		default:
			sprintf(ERROR_STRING,
	"Sorry, info for '%s' (type %d) not implemented yet.",
				SOB_NAME(sop),SOB_TYPE(sop));
			advise(ERROR_STRING);
			break;
	}
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(decorate_menu,s,f,h)
MENU_BEGIN(decorate)
ADD_CMD( info,			do_scrnobj_info,	report info about a widgt )
ADD_CMD( position,		mk_position,		set create position )
ADD_CMD( button,		mk_button,		create new button )
ADD_CMD( toggle,		mk_toggle,		create new toggle button )
ADD_CMD( label,			mk_label,		create new text label )
#ifdef FOOBAR
ADD_CMD( menu,			mk_menu_button,		create new menu )
#endif /* FOOBAR */
ADD_CMD( scroller,		mk_scroller,		create new scrolling list )
ADD_CMD( gauge,			mk_gauge,		create new gauge )
ADD_CMD( slider,		mk_slider,		create new slider )
ADD_CMD( slider_w,		mk_slider_w,		create new slider with width spec )
ADD_CMD( adjuster,		mk_adjuster,		create new adjuster )
ADD_CMD( adjuster_w,		mk_adjuster_w,		create new adjuster with width spec. )
ADD_CMD( message,		mk_message,		create new message )
// maybe this command should be edit_line instead of text...
ADD_CMD( text,			mk_text,		create new text input )
ADD_CMD( password,		mk_password,		create new password input )

ADD_CMD( text_box,		mk_text_box,		create new text display box )
ADD_CMD( append_text,		do_append_text,		append text to a text box )
ADD_CMD( activity_indicator,	mk_act_ind,		create new activity indicator )
ADD_CMD( set_active,		do_set_active,		set state of activity indicator )

ADD_CMD( edit_box,		mk_edit_box,		create new edit box )
ADD_CMD( accept_edits,		do_accept_edits,	accept edits and perform action )
ADD_CMD( cmap,			do_panel_cmap,		set custion LUT for panel )
ADD_CMD( chooser,		do_chooser,		create new chooser )
ADD_CMD( mlt_chooser,		do_mlt_chooser,		create new multiple-choice chooser )
ADD_CMD( picker,		do_picker,		create new picker )
ADD_CMD( items,			do_set_scroller,	set scroller items )
ADD_CMD( file_list,		do_file_scroller,	make a list of items from a file )
ADD_CMD( get_text,		assign_text,		assign text to script variable )
ADD_CMD( get_position,		do_get_posn_object,	assign obj posn to script variable )
ADD_CMD( set_message,		do_set_message,		update message )
ADD_CMD( set_label,		do_set_label,		update label )
ADD_CMD( set_edit_text,		do_set_edit_text,	update text )
ADD_CMD( set_text_field,	do_set_text_field,	update text field )
ADD_CMD( set_text_prompt,	do_set_prompt,		update text prompt )
// should call this set_scale_value, but we leave it alone for the sake
// of backwards-compatibility with legacy scripts...
ADD_CMD( set_scale,		do_set_gauge_value,	update gauge/slider/adjuster )
ADD_CMD( set_scale_label,	do_set_gauge_label,	update gauge/slider/adjuster label )
ADD_CMD( set_position,		do_set_posn_object,	move object )
ADD_CMD( set_toggle,		do_set_toggle,		set state of toggle )
/* these two are unimplemented for X11 */
ADD_CMD( add_choice,		do_add_choice,		add a selection to a chooser )
ADD_CMD( del_choice,		do_del_choice,		delete a selection from a chooser )
ADD_CMD( set_picks,		do_set_picks,		reset the choices in a picker )

ADD_CMD( set_choice,		do_set_choice,		set state of chooser )
ADD_CMD( clear_choices,		do_clear_choices,	clear all choices )
ADD_CMD( range,			set_new_range,		reset slider range )
ADD_CMD( slide_pos,		set_new_pos,		set slider position )
ADD_CMD( enable,		do_enable_widget,	enable/disable widget )
ADD_CMD( label_window,		set_panel_label,	set panel window label )
ADD_CMD( hide_back_button,	do_hide_back,		hide/reveal the nav back button on new panels )
ADD_CMD( show_done_button,	do_show_done,		specify action for the done button )
ADD_CMD( quit,			end_decorate,		exit submenu )
MENU_SIMPLE_END(decorate)

static COMMAND_FUNC( do_decorate_panel )
{
	Panel_Obj *po;

	po=PICK_PANEL( "" );
	prepare_for_decoration(QSP_ARG  po);
	PUSH_MENU(decorate);
}

/*
COMMAND_FUNC( do_xsync )
{
	if( askif("synchronize Xlib execution") )
		x_sync_on();
	else x_sync_off();
}
*/


static COMMAND_FUNC(do_scroll)
{
	Panel_Obj *po;
	int yn;

	po=PICK_PANEL( "" );
	yn = ASKIF("allow scrolling for panel");

	if( po == NO_PANEL_OBJ ) return;

#ifdef BUILD_FOR_IOS
	if( yn )
		[po enableScrolling];
	else
		[po disableScrolling];
#endif /* BUILD_FOR_IOS */
}

// What does check_first do???

static COMMAND_FUNC( do_check_first )
{
	Panel_Obj *po;

	po=PICK_PANEL( "" );

	if( po == NO_PANEL_OBJ ) return;

#ifdef BUILD_FOR_IOS
	check_first(po);
#endif /* BUILD_FOR_IOS */
}

static COMMAND_FUNC( do_activate )
{
	Panel_Obj *po;

	po=PICK_PANEL( "" );

	if( po == NO_PANEL_OBJ ) return;

#ifdef BUILD_FOR_IOS
	activate_panel(QSP_ARG  po,1);
#endif /* BUILD_FOR_IOS */
}

static COMMAND_FUNC( do_deactivate )
{
	Panel_Obj *po;

	po=PICK_PANEL( "" );

	if( po == NO_PANEL_OBJ ) return;

#ifdef BUILD_FOR_IOS
	activate_panel(QSP_ARG  po,0);
#endif /* BUILD_FOR_IOS */
}




#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(control_menu,s,f,h)

MENU_BEGIN(control)
#ifdef FOOBAR
ADD_CMD( clear,		clear_screen,	delete a panel )
#endif /* FOOBAR */

ADD_CMD( show,		do_show,	display a panel or viewer )
ADD_CMD( unshow,	do_unshow,	stop displaying panel or viewer )
ADD_CMD( check_first,	do_check_first,	find first responder )
ADD_CMD( activate,	do_activate,	activate a panel by bringing to front )
ADD_CMD( deactivate,	do_deactivate,	deactivate a panel by sending to back )
ADD_CMD( scroll,	do_scroll,	enable/disable scrolling for a panel )

#ifdef HAVE_MOTIF
ADD_CMD( dispatch,	do_dispatch,	start implicit dispatching )
ADD_CMD( position,	do_pposn,	position a panel or viewer )
ADD_CMD( xsync,		do_xsync,	enable/disable Xlib synchronization )
#endif /* HAVE_MOTIF */
MENU_END(control)

static COMMAND_FUNC( do_control_menu )
{
	PUSH_MENU(control);
}

static COMMAND_FUNC( do_list_panels ){ list_panel_objs(SINGLE_QSP_ARG); }

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(objects_menu,s,f,h)

MENU_BEGIN(objects)
ADD_CMD( panels,	do_list_panels,		list panels )
ADD_CMD( objects,	do_list_panel_objs,	list objects belonging to a panel )
ADD_CMD( info,		do_so_info,		give object info )
MENU_END(objects)

static COMMAND_FUNC( do_so_menu )
{
	PUSH_MENU(objects);
}

#ifdef MALLOC_DEBUG
COMMAND_FUNC( do_mallocverify )
{
	malloc_verify();
}

COMMAND_FUNC( do_mallocdebug )
{
	int n;

	n=how_many("malloc debugging level");
	if( n>= 0 && n <= 2 ) malloc_debug(n);
	else warn("bad level");
}
#endif /* MALLOC_DEBUG */

#ifdef HAVE_MOTIF
static COMMAND_FUNC( do_sel_panel )
{
	Panel_Obj *po;

	po=PICK_PANEL( "" );
	set_curr_win( po->po_xwin );
	set_colormap( PO_CMAP_OBJ(po) );
}
#endif /* HAVE_MOTIF */

static Nav_Panel *curr_nav_p=NO_NAV_PANEL;
static Nav_Group *curr_nav_g=NO_NAV_GROUP;

static COMMAND_FUNC( do_new_nav_group )
{
	const char *s;
	Nav_Group *nav_g;

	s=NAMEOF("name for group");

	nav_g = create_nav_group(QSP_ARG  curr_nav_p, s);

	if( nav_g == NO_NAV_GROUP ) return;

	// push the group context here...
	// and pop the old one if there is one.

	if( curr_nav_g != NO_NAV_GROUP ){
		IOS_Item_Context *icp;
		icp=pop_navitm_context(SINGLE_QSP_ARG);
	}
	push_navitm_context(QSP_ARG  NAVGRP_ITEM_CONTEXT(nav_g) );
	pushed_navitm_context = NAVGRP_ITEM_CONTEXT(nav_g);

	curr_nav_g = nav_g;

#ifndef BUILD_FOR_OBJC
	// make a label for the group
	// We need to push the nav_panel context on screen_objs too...
	Screen_Obj *sop;

	// BUG?  why get action text for a message???
	// Here the action text is not the action, it is the message.
	// The message can be different from the name.
	sop = simple_object(QSP_ARG  s);
	if( sop == NO_SCREEN_OBJ ) return;
	SET_SOB_ACTION(sop, savestr(s));

	SET_SOB_TYPE(sop, SOT_LABEL);
	SET_SOB_CONTENT(sop, SOB_ACTION(sop));
	SET_SOB_ACTION(sop, NULL);
	SET_SOB_HEIGHT(sop, MESSAGE_HEIGHT);

	make_label(QSP_ARG  sop);

	add_to_panel(curr_panel,sop);
//fprintf(stderr,"before increment, panel %s y = %d\n",PO_NAME(curr_panel),PO_CURR_Y(curr_panel));

	// BUG - when we can wrap lines, the height will be
	// variable.  How do we determine the height from
	// the content?
	INC_PO_CURR_Y(curr_panel, SOB_HEIGHT(sop) + GAP_HEIGHT );

//fprintf(stderr,"after increment, panel %s y = %d\n",PO_NAME(curr_panel),PO_CURR_Y(curr_panel));

#endif // ! BUILD_FOR_OBJC
}

static COMMAND_FUNC( do_set_nav_group )
{
	Nav_Group *nav_g;

	nav_g = PICK_NAV_GROUP("existing navigation group");

	if( nav_g == NO_NAV_GROUP ) return;

	if( curr_nav_g != NO_NAV_GROUP ){
		IOS_Item_Context *icp;
		icp=pop_navitm_context(SINGLE_QSP_ARG);
	}
	push_navitm_context(QSP_ARG  NAVGRP_ITEM_CONTEXT(nav_g) );
	pushed_navitm_context = NAVGRP_ITEM_CONTEXT(nav_g);

	curr_nav_g = nav_g;
}


static void do_table_item(QSP_ARG_DECL  Table_Item_Type t)
{
	const char *s;
	const char *e;
	const char *a;
	Nav_Item *nav_i;

	s=NAMEOF("name for item");
	e=NAMEOF("short explanatory text for item");
	a=NAMEOF("navigation item action text");

	if( curr_nav_g == NO_NAV_GROUP ){
		WARN("do_table_item:  no group declared!?");
		return;
	}

	nav_i = new_nav_item(QSP_ARG  s);
	if( nav_i == NO_NAV_ITEM ) return;

#ifdef BUILD_FOR_OBJC
	nav_i.explanation = [[NSString alloc] initWithUTF8String:e];
#else
	SET_NAVITM_EXPLANATION(nav_i,savestr(e));
#endif
	SET_NAVITM_ACTION(nav_i, savestr(a) );
	SET_NAVITM_TYPE(nav_i, t);

#ifdef BUILD_FOR_OBJC
	// add this item to the current group
	[ curr_nav_g add_nav_item: nav_i ];
#else	// ! BUILD_FOR_OBJC
	add_navitm_to_group(curr_nav_g,nav_i);

	Screen_Obj *bo;
	char *str;

	str=getbuf( strlen(s)+strlen(e)+6 );
	sprintf(str,"%s  -  %s",s,e);

	bo = simple_object(QSP_ARG  str);
	if( bo == NO_SCREEN_OBJ ) return;

	SET_SOB_ACTION(bo, savestr(a));

	SET_SOB_TYPE(bo, SOT_BUTTON);

	make_button(QSP_ARG  bo);
	add_to_panel(curr_panel,bo);

//fprintf(stderr,"do_table_item:  before increment, panel %s y = %d\n",PO_NAME(curr_panel),PO_CURR_Y(curr_panel));
	INC_PO_CURR_Y(curr_panel, BUTTON_HEIGHT + GAP_HEIGHT );
//fprintf(stderr,"do_table_item:  after increment, panel %s y = %d\n",PO_NAME(curr_panel),PO_CURR_Y(curr_panel));
#endif /* BUILD_FOR_OBJC */
}

static COMMAND_FUNC( do_nav_item )
{
	do_table_item(QSP_ARG  TABLE_ITEM_TYPE_NAV);
}

static COMMAND_FUNC( do_plain_item )
{
	do_table_item(QSP_ARG  TABLE_ITEM_TYPE_PLAIN);
}

static COMMAND_FUNC( do_set_desc )
{
	Nav_Item *nav_i;
	const char *s;

	nav_i=pick_nav_item( QSP_ARG  "item name" );
	s = NAMEOF("description string");

	if( curr_nav_g == NO_NAV_GROUP ){
		WARN("do_set_desc:  no group selected");
		return;
	}

	if( nav_i == NULL ) return;

#ifdef BUILD_FOR_OBJC
	nav_i.explanation = [[NSString alloc] initWithUTF8String:s];

	// now reload...
	[nav_i.group reload_group];
#else
	SET_NAVITM_EXPLANATION(nav_i, savestr(s));
#endif // BUILD_FOR_OBJC
}

static COMMAND_FUNC( do_del_nav_item )
{
	Nav_Item *nav_i;

	nav_i=pick_nav_item( QSP_ARG  "item name" );
	if( nav_i == NULL ) return;

	remove_nav_item(QSP_ARG  nav_i);
}

static COMMAND_FUNC( do_del_nav_group )
{
	Nav_Group *nav_g;

	nav_g=pick_nav_group( QSP_ARG  "group name" );
	if( nav_g == NULL ) return;

	remove_nav_group(QSP_ARG  nav_g);
}

static COMMAND_FUNC( do_hide_nav_bar )
{
	int hide;

	hide = ASKIF("hide navigation bar");
	hide_nav_bar( QSP_ARG  hide );
}

#ifdef FOOBAR
static COMMAND_FUNC( do_set_done )
{
	const char *s;

	s=NAMEOF("action for done button");

#ifdef BUILD_FOR_IOS
	if( curr_nav_p != NO_NAV_PANEL ){
		[curr_nav_p setDoneAction:s];
	} else {
		WARN("done_button:  current nav panel!?");
	}
#endif /* BUILD_FOR_IOS */
}
#endif // FOOBAR

static COMMAND_FUNC( do_end_navigation )
{
	IOS_Item_Context *icp;

#ifndef BUILD_FOR_OBJC
	assert( curr_panel != NULL );
	icp=pop_scrnobj_context(SINGLE_QSP_ARG);
#endif // BUILD_FOR_OBJC

	assert( curr_nav_p != NULL );

	icp=pop_navgrp_context( SINGLE_QSP_ARG  );

	// We can't be sure that we have pushed a navitm context
	// without checking the flag...
	if( pushed_navitm_context != NULL ){
		icp=pop_navitm_context(SINGLE_QSP_ARG);
		assert( icp == pushed_navitm_context );
		pushed_navitm_context = NULL;

	}

	pop_menu(SINGLE_QSP_ARG);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(navigation_menu,s,f,h)

MENU_BEGIN(navigation)
ADD_CMD( new_group,	do_new_nav_group,	create new navigation group )
ADD_CMD( nav_item,	do_nav_item,		create new navigation item )
ADD_CMD( plain_item,	do_plain_item,		create non-navigation table entry )
ADD_CMD( delete_item,	do_del_nav_item,	delete a table entry )
ADD_CMD( delete_group,	do_del_nav_group,	delete a table group )
ADD_CMD( set_group,	do_set_nav_group,		set current navigation group )
ADD_CMD( set_description,	do_set_desc,	reset navigation item description )
/*ADD_CMD( done_button,	do_set_done,		specify action for done button ) */
ADD_CMD( quit,		do_end_navigation,	exit submenu )
MENU_SIMPLE_END(navigation)

static COMMAND_FUNC( mk_nav_panel )
{
	Nav_Panel *nav_p;
	const char *s;

	s=NAMEOF("name for navigation panel");

	// make sure doesn't already exist.
	// Is this check done by new_nav_panel???
	nav_p = create_nav_panel(QSP_ARG  s);

	// BUG - shouldn't we put this in when
	// we enter the navigation submenu???
#ifndef BUILD_FOR_OBJC
	curr_panel = NAVP_PANEL(nav_p);
#endif // ! BUILD_FOR_OBJC
}

/* When we enter the navigation menu, we have a new context for groups.
 * We used to pop the old item context, but we don't do that now
 * that the navitm contexts pertain to groups, not panels.
 *
 * It might be cleaner to do the popping when we exit the menu,
 * but our new menu convenience macros add the default pop_menu
 * by default, so it is difficult to customize...  So instead
 * we do the popping when we enter the second (and subsequent) time.
 *
 * We need to do this when "decorating" regular panels too!?
 */

static COMMAND_FUNC( do_nav_menu )
{
#ifndef BUILD_FOR_OBJC
	Panel_Obj *pnl_p;
#endif // ! BUILD_FOR_OBJC

	curr_nav_p = PICK_NAV_PANEL("navigation panel");

	// Need to push the item context for the panel!
	if( curr_nav_p == NO_NAV_PANEL ) return;

#ifndef BUILD_FOR_OBJC
	pnl_p = NAVP_PANEL(curr_nav_p);
	assert( pnl_p != NULL );

	// do what we do in decorate...
	prepare_for_decoration(QSP_ARG  pnl_p);
#endif // ! BUILD_FOR_OBJC

	push_navgrp_context(QSP_ARG  NAVP_GRP_CONTEXT(curr_nav_p));
	// we push the item context when we have the group...
	// But how do we know whether or not we pushed a navitm context???
	// We use a (non thread-safe) global, pushed_navitm_context
	// Here we set it to null to indicate that we haven't pushed anything...
	pushed_navitm_context = NULL;

	// Setting the group to nothing makes sense for creation,
	// but what about deletion?
	// What does this comment mean?
	curr_nav_g = NO_NAV_GROUP;

	PUSH_MENU(navigation);
} // do_nav_menu

static COMMAND_FUNC( do_push_nav )
{
	//Panel_Obj *po;
	Gen_Win *gwp;
	const char *s;

	// Are nav panels really regular panels???
	//po = PICK_PANEL("");
	//if( po == NO_PANEL_OBJ ) return;

	s = NAMEOF("name of panel or viewer");
	gwp = find_genwin(QSP_ARG  s);
	if( gwp == NULL ) return;

#ifdef BUILD_FOR_IOS
	if( GW_PO(gwp) == console_po ){
		// enable output to the console
		enable_console_output(1);
	}
#endif /* BUILD_FOR_IOS */

	push_nav(QSP_ARG  gwp);
}

static COMMAND_FUNC(do_pop_nav)
{
	int n;

	n=(int)HOW_MANY("number of levels to pop");
	if( n < 1 ){
		WARN("pop_nav:  number of levels must be positive!?");
		return;
	}
	pop_nav(QSP_ARG n);
}

static COMMAND_FUNC(do_top_nav)
{
	int n;

	n = n_pushed_panels();
	if( n > 1 ) pop_nav(QSP_ARG n-1);
}

Panel_Obj *console_po=NULL;

static COMMAND_FUNC( mk_console )
{
	// We only want to have one console...
	static int console_created=0;
	const char *s;

	s=NAMEOF("name for console panel");

	if( console_created ){
		WARN("Console already created!?");
		return;
	}

#ifdef BUILD_FOR_IOS
	make_console_panel(QSP_ARG  s);

	console_po = panel_obj_of(QSP_ARG  s);
#endif /* BUILD_FOR_IOS */
}

static COMMAND_FUNC( do_alert )
{
	const char *type, *msg;

	type=NAMEOF("type of alert");
	msg=NAMEOF("alert message");

	simple_alert(QSP_ARG  type,msg);

	// another event can occur while the alert is getting
	// ready to go up, pushing text onto the command stack.
	// We might want to pop the menu stack here...
}

static COMMAND_FUNC( do_confirm )
{
	const char *title, *question;

	title=NAMEOF("title for alert");
	question=NAMEOF("confirmation question");

	get_confirmation(QSP_ARG  title,question);

	// another event can occur while the alert is getting
	// ready to go up, pushing text onto the command stack.
	// We might want to pop the menu stack here...
}

static COMMAND_FUNC( do_end_busy )
{
	end_busy(1);	// arg=1 means final
}

static COMMAND_FUNC( do_notify_busy )
{
	const char *title, *msg;

	title=NAMEOF("title for alert");
	msg=NAMEOF("alert message");

	notify_busy(QSP_ARG  title,msg);

	// another event can occur while the alert is getting
	// ready to go up, pushing text onto the command stack.
	// We might want to pop the menu stack here...
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(interface_menu,s,f,h)

MENU_BEGIN(interface)
ADD_CMD( panel,		mk_panel,		create new control panel )
ADD_CMD( decorate,	do_decorate_panel,	add objects to a panel )
ADD_CMD( control,	do_control_menu,	window system control )

ADD_CMD( console,	mk_console,		create the QuIP console )
ADD_CMD( nav_panel,	mk_nav_panel,		create new navigator pane )
ADD_CMD( navigation,	do_nav_menu,		navigation submenu )
ADD_CMD( push_nav,	do_push_nav,		push a navigation panel )
ADD_CMD( pop_nav,	do_pop_nav,		pop current (one or more) navigation panel )
ADD_CMD( top_nav,	do_top_nav,		pop to first navigation panel )
ADD_CMD( alert,		do_alert,		deliver a pop-up alert)
ADD_CMD( confirm,	do_confirm,		get confirmation in a pop-up alert)
ADD_CMD( notify_busy,	do_notify_busy,		deliver a non-blocking pop-up alert)
ADD_CMD( end_busy,	do_end_busy,		dismiss busy pop-up alert)
ADD_CMD( hide_nav_bar,	do_hide_nav_bar,	hide/reveal the navigation bar )

ADD_CMD( resize_panel, do_resize_panel, resize an existing panel )

ADD_CMD( objects,	do_so_menu,		object database submenu )
#ifdef BUILD_FOR_MACOS
ADD_CMD( menu_bar,	do_menu_bar,		menu bar submenu )
#endif // BUILD_FOR_MACOS
#ifdef HAVE_MOTIF
ADD_CMD( select,	do_sel_panel,		select current panel for colormap ops )
ADD_CMD( delete,	do_delete,		delete a canvas or panel )
// why is do_notice only motif??
ADD_CMD( notice,	do_notice,		give a notice )
#endif /* HAVE_MOTIF */
/* support for genwin */
ADD_CMD( genwin,	do_genwin_menu,		general window operations submenu )
#ifdef NOT_YET
ADD_CMD( luts,		do_lut_menu,		color map submenu )
#endif /* NOT_YET */
#ifdef MALLOC_DEBUG
ADD_CMD( malloc_debug,	do_mallocdebug,		set malloc debugging level )
ADD_CMD( malloc_verify,	do_mallocverify,	verify memory heap )
#endif
MENU_END(interface)

/* If SGI window manager, then popups are supported */

COMMAND_FUNC( do_protomenu )
{
	static int inited=0;

	if( ! inited ){
		const char *prog_name;
#ifdef HAVE_X11
		const char *display_name;
		display_name = which_display(SINGLE_QSP_ARG);
fprintf(stderr,"do_protomenu:  display is %s\n",display_name);
#endif /* HAVE_X11 */

		prog_name = tell_progname();
		so_init(QSP_ARG  1,&prog_name);
fprintf(stderr,"do_protomenu:  back from so_init\n");

		DECLARE_STR1_FUNCTION(	panel_exists,	panel_exists )

		inited=1;
	}

#ifndef BUILD_FOR_OBJC
#ifndef HAVE_MOTIF
	WARN("Program not built with Motif support!?");
	advise("Commands will parse but have no effect.");
#endif /* ! HAVE_MOTIF */
#endif /* ! BUILD_FOR_OBJC */
	PUSH_MENU(interface);
}

