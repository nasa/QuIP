#include "quip_config.h"

/*
 * General screen object stuff, pretty independent of window system
 */

#include <stdio.h>

#include "quip_prot.h"
#include "gui_cmds.h"
#include "gui_prot.h"
#include "sizable.h"
#include "debug.h"
#include "stack.h"		// BUG
#include "query_stack.h"		// BUG

#define BAD_STRING(s)	(s==NULL || *s==0)

#ifdef BUILD_FOR_OBJC

#include "ios_gui.h"
#include "sizable.h"

// an ios "picker" has a fixed size
#define PICKER_HEIGHT	220

#else /* ! BUILD_FOR_OBJC */

#ifdef HAVE_X11
#include "xsupp.h"
#ifdef HAVE_MOTIF
#include "my_motif.h"
#endif /* HAVE_MOTIF */
#endif /* HAVE_X11 */

#define new_ios_list	new_list
#define create_ios_item_context	create_item_context

#endif /* ! BUILD_FOR_OBJC */

#include "gen_win.h"	/* add_genwin() */

Panel_Obj *curr_panel=NULL;

#define BUF_LEN 128
#define COL_STR	":"

/* local prototypes */

static void _get_menu_items(QSP_ARG_DECL  Screen_Obj *mp);
#define get_menu_items(mp)		_get_menu_items(QSP_ARG  mp)
#ifdef NOT_YET
static void del_po(QSP_ARG_DECL  Panel_Obj *po);
#endif /* NOT_YET */


/* local prototypes */


#define BASENAME	"Base"

#define MAX_STACK 8
static Screen_Obj *curr_parent, *parent_stack[MAX_STACK];
static int parent_index=(-1);

// BUG the order of this list has to match what is in
// the .h file!?

static const char *widget_type_name[N_WIDGET_TYPES]={
	"text",
	"button",
	"slider",
	"menu",
	"menu_item",
	"gauge",
	"message",
	"pullright",
	"menu_button",
	"scroller/chooser",
	"toggle"
};

#ifndef BUILD_FOR_OBJC
ITEM_INTERFACE_DECLARATIONS(Panel_Obj,panel_obj,0)
ITEM_INTERFACE_DECLARATIONS(Screen_Obj,scrnobj,0)
#endif /* ! BUILD_FOR_OBJC */

#ifdef NOT_USED
void show_panel_children(Panel_Obj *po)
{
	Node *np;
	Screen_Obj *sop;

	np=QLIST_HEAD(PO_CHILDREN(po));
	while(np!=NULL){
		sop=(Screen_Obj *)np->n_data;
		if( sop != NULL ){
			sprintf(ERROR_STRING,"\t%s",SOB_NAME(sop));
			advise(ERROR_STRING);
		} else {
			advise("\tnull screen_obj!?");
		}
		np=np->n_next;
	}
}
#endif /* NOT_USED */

#ifndef BUILD_FOR_OBJC

void add_to_panel(Panel_Obj *po, Screen_Obj *sop)	// for X11
{
	addHead(PO_CHILDREN(po),mk_node(sop));
	SET_SOB_PARENT(sop, po);
}

void remove_from_panel( Panel_Obj *po, Screen_Obj *sop)	// for X11
{
	remData(PO_CHILDREN(po),sop);
}


void add_to_navp(Nav_Panel *np_p, Screen_Obj *sop)	// for X11
{
	addHead(PO_CHILDREN(np_p->np_po),mk_node(sop));
	SET_SOB_PARENT(sop, np_p->np_po);
}

/* Why don't these do anything!?!? */

Item_Context *_pop_scrnobj_context(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp;

	icp = pop_item_context(scrnobj_itp);
	return icp;
}

void _push_scrnobj_context(QSP_ARG_DECL  Item_Context *icp)
{
	push_item_context(scrnobj_itp, icp );
}

Item_Context *_current_scrnobj_context(SINGLE_QSP_ARG_DECL)
{
	return current_context(scrnobj_itp);
}

Item_Context *_create_scrnobj_context(QSP_ARG_DECL  const char *name)
{
	if( scrnobj_itp == NULL ){
		init_scrnobjs();
	}

	return create_item_context(scrnobj_itp, name );
}

#endif /* ! BUILD_FOR_OBJC */


#define WIDGET_TYPE_NAME(sop)		widget_type_name[WIDGET_INDEX(sop)]

Screen_Obj *_simple_object(QSP_ARG_DECL  const char *name)
{
	Screen_Obj *sop;
	sop = new_scrnobj(name);
	if( sop == NULL ) return(sop);

	SET_SOB_ACTION(sop,NULL);
	SET_SOB_SELECTOR(sop,NULL);
	SET_SOB_PANEL(sop, curr_panel);
	SET_SOB_PARENT(sop, NULL);
	SET_SOB_CHILDREN(sop, NULL);
#ifdef HAVE_MOTIF
	SET_SOB_FRAME(sop, NULL);
#endif /* HAVE_MOTIF */
	SET_SOB_VAL(sop,0);		/* so_nlist too! */
	SET_SOB_MIN(sop,0);
	SET_SOB_MAX(sop,0);
	SET_SOB_FLAGS(sop, 0);
#ifdef THREAD_SAFE_QUERY
	SET_SOB_QSP(sop, qsp);
#endif /* THREAD_SAFE_QUERY */
	return(sop);
}

Screen_Obj *dup_so(QSP_ARG_DECL  Screen_Obj *sop)
{
	Screen_Obj *dup;
	char name[BUF_LEN];

	sprintf(name,"%s.dup",SOB_NAME(sop));
	dup = simple_object(name);
	if( sop == NULL ) return(sop);
	SET_SOB_PARENT(dup, SOB_PARENT(sop) );
	SET_SOB_PANEL(dup, SOB_PANEL(sop) );
#ifdef HAVE_MOTIF
	SET_SOB_FRAME(dup, SOB_FRAME(sop) );
#endif /* HAVE_MOTIF */
	SET_SOB_ACTION(dup,savestr(SOB_ACTION(sop)));
	SET_SOB_SELECTOR(dup,NULL);
	return(dup);
}

void so_info(Screen_Obj *sop)
{
	if( sop==NULL ) return;

	printf("%s\t\t%s\n",SOB_NAME(sop),WIDGET_TYPE_NAME(sop));
	if( SOB_SELECTOR(sop) != NULL )
		printf("\tselector:\t%s\n",SOB_SELECTOR(sop));
	if( SOB_ACTION(sop) != NULL )
		printf("\taction text:\t%s\n",SOB_ACTION(sop));
	if( WIDGET_PANEL(sop) != NULL )
		printf("\tpanel:\t%s\n",PO_NAME(WIDGET_PANEL(sop)));
	/* else printf("no frame object; this must be a frame itself\n"); */
}

COMMAND_FUNC( do_so_info )
{
	Screen_Obj *sop;

	sop=pick_scrnobj("");
	so_info(sop);
}

/* Sets position used to create next object */
COMMAND_FUNC( mk_position )
{
	int x, y;

	x = (int)how_many("x position");
	y = (int)how_many("y position");
	if( !curr_panel ) return;
	SET_PO_CURR_X(curr_panel, x);
	SET_PO_CURR_Y(curr_panel, y);
}

/* Gets object x,y position and inserts value in strs x_val & y_val */
COMMAND_FUNC( do_get_posn_object )
{
	Screen_Obj *sop;
	char str[BUF_LEN];

	sop = pick_scrnobj("");
	if( sop == NULL ) return;
	sprintf(str, "%d", SOB_X(sop));
	assign_reserved_var("x_val",str);
	sprintf(str, "%d", SOB_Y(sop));
	assign_reserved_var("y_val",str);
}

/* Moves an object */
COMMAND_FUNC( do_set_posn_object )
{
	Screen_Obj *sop;
	int x,y;

	sop = pick_scrnobj("");
	x=(int)how_many("x position");
	y=(int)how_many("y position");
	if( sop == NULL ) return;
	SET_SOB_X(sop,x);
	SET_SOB_Y(sop,y);
	reposition(sop);
}

#ifdef BUILD_FOR_IOS

/* dummy_panel is used to create a sizable for $DISPLAY in IOS */

Gen_Win *dummy_panel(QSP_ARG_DECL  const char *name,int dx,int dy)
{
	Gen_Win *gwp;

	gwp = new_genwin(QSP_ARG  name);
#ifdef CAUTIOUS
	if( gwp == NULL )
		error1("CAUTIOUS:  dummy_panel:  error creating panel object!?");
#endif /* CAUTIOUS */

	SET_GW_WIDTH(gwp, dx);
	SET_GW_HEIGHT(gwp, dy);

	/* SHould we set the depth??? */

	return gwp;
}

#endif /* BUILD_FOR_IOS */

static void make_scrnobj_ctx_for_panel(QSP_ARG_DECL  Panel_Obj *po)
{
	IOS_Item_Context *icp = create_scrnobj_context(PO_NAME(po));
//sprintf(ERROR_STRING,"new_panel %s (0x%lx, qvc = 0x%lx), size is %d x %d,  setting context to 0x%lx",
//PO_NAME(po),(long)po,(long)PO_QVC(po),PO_WIDTH(po),PO_HEIGHT(po),(long)icp);
//advise(ERROR_STRING);
	SET_PO_CONTEXT(po, icp);
}

Panel_Obj *new_panel(QSP_ARG_DECL  const char *name,int dx,int dy)
{
	Panel_Obj *po;

	po = new_panel_obj(name);
	if( po == NULL ){
		printf("ERROR creating new panel object!?\n");
		return(po);
	}
//fprintf(stderr,"panel obj created...\n");


#ifndef BUILD_FOR_OBJC
	SET_PO_WIDTH(po, dx);
	SET_PO_HEIGHT(po, dy);
	// This represents where we put the panel on the screen...
	// for X11, place near the upper left...
	SET_PO_X(po, 100);
	SET_PO_Y(po, 100);
	SET_PO_FLAGS(po, 0);	// why not initialize the flags for IOS too?

	SET_GW_TYPE( PO_GW(po), GW_PANEL);
#endif /* ! BUILD_FOR_OBJC */


	SET_PO_CHILDREN(po, new_ios_list() );

	// In IOS, we don't need to have a distinction
	// between viewers and panels, we can add controls
	// to viewers and images to panels.  So we need
	// to have a stripped down version of new_panel
	// to set up an existing viewer for getting widgets.

//sprintf(ERROR_STRING,"new_panel %s calling make_panel",PO_NAME(po));
//advise(ERROR_STRING);
	make_panel(QSP_ARG  po,dx,dy);			/* Xlib calls */

#ifdef BUILD_FOR_OBJC
	/* these fields are part of the associated Gen_Win,
	 * and must be set after the call to make_panel
	 * That means we can't use the flags field to
	 * pass the hide_back_button flag...
	 */
	// for IOS, typically we use the whole screen
	// BUG - for nav_panels, we need to push down to clear
	// the nav bar!
	SET_PO_X(po, 0);
	SET_PO_Y(po, 0);
	SET_PO_FLAGS(po, 0);
#endif /* BUILD_FOR_OBJC */

	make_scrnobj_ctx_for_panel(QSP_ARG  po);

	// This is where we place the first object
	SET_PO_CURR_X(po,OBJECT_GAP);
	SET_PO_CURR_Y(po,OBJECT_GAP);

	curr_panel=po;
//fprintf(stderr,"new_panel DONE\n");
	return(po);
}

COMMAND_FUNC( mk_panel )
{
	const char *s;
	//Panel_Obj *po;
	int dx,dy;

	s=nameof("name for panel");
	dx=(int)how_many("panel width");
	dy=(int)how_many("panel height");
	/*po=*/ new_panel(QSP_ARG  s,dx,dy);
}

COMMAND_FUNC( do_resize_panel )
{
    Panel_Obj *po = pick_panel( "" );
    if (po == NULL) {
        advise("do_resize_panel: no panel found");
        return;
    }

    int dx = (int)how_many("panel width");
    int dy = (int)how_many("panel height");

    SET_PO_WIDTH(po, dx);
    SET_PO_HEIGHT(po, dy);
printf("%d %d\n",PO_WIDTH(po), PO_HEIGHT(po));

}


#define list_widgets(po) _list_widgets(QSP_ARG  po)

static void _list_widgets(QSP_ARG_DECL  Panel_Obj *po)
{
	IOS_List *lp;
	IOS_Node *np;
	/* lp=item_list(QSP_ARG  scrnobj_itp); */
	lp = PO_CHILDREN(po);

	np=IOS_LIST_HEAD(lp);
	while(np!=NULL ){
		Screen_Obj *sop;

		sop=(Screen_Obj *)IOS_NODE_DATA(np);
		if( WIDGET_PANEL(sop) == po ){
			/* list_scrnobj(sop); */
		//	printf("\t%s\n",SOB_NAME(sop));
			so_info(sop);
		} else {
			sprintf(ERROR_STRING,"widget %s does not belong to panel %s!?",
				SOB_NAME(sop),PO_NAME(po));
			warn(ERROR_STRING);
		}
		np=IOS_NODE_NEXT(np);
	}
}

COMMAND_FUNC( do_list_panel_objs )
{
	Panel_Obj *po;
	po=pick_panel("");
 	if( po == NULL ) return;

	list_widgets(po);
}

/*
 *  Prompt for object name, panel, and action text
 *  If ptr to return value for panel is NULL, don't ask for panel
 *
 *  class_str is imbedded in the prompts
 */

Screen_Obj *_get_parts(QSP_ARG_DECL  const char *class_str)
{
	char pmpt[BUF_LEN];
	//char text[BUF_LEN];
	//char label[BUF_LEN];
	Screen_Obj *sop;
	const char *text, *label;

	// Why are the label and text strings saved
	// in these local buffers???

	sprintf(pmpt,"%s label",class_str);
	//strcpy( label, nameof(pmpt) );
	label = nameof(pmpt) ;

	sprintf(pmpt,"%s action text",class_str);
	text = nameof(pmpt) ;

	if( curr_panel == NULL ) return NULL;
	if( label == NULL ) return NULL;	// a non-existent variable, for example...
	if( text == NULL ) return NULL;	// a non-existent variable, for example...

	sop = simple_object(label);
	if( sop == NULL ) return(sop);

	SET_SOB_ACTION(sop, savestr(text));
	return(sop);
}

Screen_Obj *_mk_menu(QSP_ARG_DECL  Screen_Obj *mip)
{
	Screen_Obj *mp;

	char buf[BUF_LEN];

	mp=dup_so(QSP_ARG  mip);
	givbuf((void *)SOB_NAME(mp));

	/* append colon to name */
	strcpy(buf,SOB_NAME(mip));
	strcat(buf,COL_STR);
	SET_SOB_NAME(mp,buf);

#ifdef QUIP_DEBUG
if( debug ) fprintf(stderr,"Making menu \"%s\"\n",SOB_NAME(mp));
#endif /* QUIP_DEBUG */

	make_menu(QSP_ARG  mp,mip);

	get_menu_items(mp);

#ifdef QUIP_DEBUG
if( debug ) fprintf(stderr,"make_menu:  back from get_menu_items (menu %s)\n",
SOB_NAME(mp));
#endif /* QUIP_DEBUG */
	SET_SOB_TYPE(mp, SOT_MENU);
	return(mp);
}

#define SEPARATOR_HEIGHT 3
#ifdef HAVE_XM_SEPARATOR_H
#include <Xm/Separator.h>
#endif

/* What is a menu button?  not in motif? */

#ifdef FOOBAR
COMMAND_FUNC( mk_menu_button )
{
	Screen_Obj *mp, *bp;

	bp=get_parts("menu");
	if( bp == NULL ) return;
	SET_SOB_TYPE(bp, SOT_MENU_BUTTON);

	mp=MK_MENU(bp);

	/* The popup menu is created in mk_menu, don't
	 * need to make this call from MOTIF */
#ifndef MOTIF
	make_menu_button(QSP_ARG  bp,mp);
#endif /* !MOTIF */

	add_to_panel(curr_panel,bp);

	/* What is the 30??? */
	INC_PO_CURR_Y(curr_panel, BUTTON_HEIGHT + GAP_HEIGHT + 30 );
}
#endif /* FOOBAR */

/* copy name to selector, and replace name with parent's name + selector.
 * This seems to be done for the items in a menu?
 * I guess the reason is so that all the objects can have unique names,
 * even if we have duplicate names in different choosers...
 */

void fix_names(QSP_ARG_DECL  Screen_Obj *mip,Screen_Obj *parent)
{
	char buf[BUF_LEN];

	SET_SOB_SELECTOR(mip,savestr(SOB_NAME(mip)) );
	strcpy(buf,SOB_NAME(parent));
	strcat(buf,".");
	strcat(buf,SOB_SELECTOR(mip));

	//rename_item(QSP_ARG  scrnobj_itp,mip,buf);
}

COMMAND_FUNC( pop_parent )
{
	if( parent_index < 0 ){
		WARN("no parent to pop");
		parent_index=(-1);
		return;
	}
	parent_index--;
	if( parent_index >= 0 )
		curr_parent = parent_stack[parent_index];
	else curr_parent = NULL;
}

COMMAND_FUNC( end_menu )
{
#ifdef QUIP_DEBUG
if( debug ){
fprintf(stderr,"Popping menu \"%s\"\n",SOB_NAME(curr_parent));
}
#endif /* QUIP_DEBUG */
	// pop_parent pops the parent screen object...
	pop_parent(SINGLE_QSP_ARG);
#ifdef QUIP_DEBUG
if( debug ){
if( curr_parent != NULL )
fprintf(stderr,"New parent menu \"%s\"\n",SOB_NAME(curr_parent));
else
WARN("all parent menus popped!!");
}
#endif /* QUIP_DEBUG */
	pop_menu();
}

void push_parent(Screen_Obj *mp)
{
	parent_index++;
	if( parent_index >= MAX_STACK ) NERROR1("parent stack full");
	curr_parent=parent_stack[parent_index]=mp;
}

COMMAND_FUNC( do_normal )
{
	Screen_Obj *mip;

	mip=get_parts("menu");
	if( mip==NULL ) return;
	SET_SOB_PANEL(mip, SOB_PANEL(curr_parent) );
	SET_SOB_TYPE(mip, SOT_MENU_ITEM);

	add_to_panel(curr_panel,mip);

	make_menu_choice(QSP_ARG  mip,curr_parent);
	fix_names(QSP_ARG  mip,curr_parent);
}

#ifdef TEST_DROP_OUT

COMMAND_FUNC( do_pullright )
{
	Screen_Obj *pr, *mip;

	mip=get_parts("menu");
	if( mip==NULL ) return;
	SET_SOB_PANEL(mip, SOB_PANEL(curr_parent) );
	SET_SOB_PARENT(mip, NULL);
	SET_SOB_TYPE(mip, SOT_PULLRIGHT);


	/* now dup the names for the menu */
	/* need to get the submenu */
	pr=MK_MENU(mip);

	make_pullright(QSP_ARG  mip,pr,curr_parent);
	fix_names(QSP_ARG  mip,curr_parent);
}
#endif /* TEST_DROP_OUT */

#define ADD_CMD(s,f,h)	ADD_COMMAND(menu_item_menu,s,f,h)

MENU_BEGIN(menu_item)
ADD_CMD( button,	do_normal,	specify normal menu item )
//ADD_CMD( pullright,	do_pullright,	specify pullright submenu )
ADD_CMD( end_menu,	end_menu,	quit adding menu items )
MENU_SIMPLE_END(menu_item)	// use SIMPLE to avoid auto quit/pop_menu

static void _get_menu_items(QSP_ARG_DECL  Screen_Obj *mp)
{
	int depth;

	push_parent(mp);
	CHECK_AND_PUSH_MENU(menu_item);

	/* now need to do_cmd() until menu is exited */
	/* But don't we normally just execute commands??? */

	//depth = cmd_depth(SINGLE_QSP_ARG);
	depth = STACK_DEPTH(QS_MENU_STACK(THIS_QSP));
	while( depth==STACK_DEPTH(QS_MENU_STACK(THIS_QSP)) ) {
		qs_do_cmd(THIS_QSP);
	}
}

void _get_min_max_val(QSP_ARG_DECL  int *minp,int *maxp,int *valp)
{
	*minp=(int)how_many("min value");
	*maxp=(int)how_many("max value");
	*valp=(int)how_many("initial value");
}

void _get_so_width(QSP_ARG_DECL int *widthp)
{
	*widthp=(int)how_many("width");
}

COMMAND_FUNC( mk_button )
{
	Screen_Obj *bo;

	bo = get_parts("button");
	if( bo == NULL ) return;
	SET_SOB_TYPE(bo, SOT_BUTTON);

	make_button(bo);
	add_to_panel(curr_panel,bo);

	INC_PO_CURR_Y(curr_panel, BUTTON_HEIGHT + GAP_HEIGHT );
}

COMMAND_FUNC( mk_toggle )
{
	Screen_Obj *to;

	to = get_parts("toggle");
	if( to == NULL ) return;
	SET_SOB_TYPE(to, SOT_TOGGLE);

	make_toggle(QSP_ARG  to);
	add_to_panel(curr_panel, to );

	INC_PO_CURR_Y(curr_panel, TOGGLE_HEIGHT + GAP_HEIGHT );
}

static void mk_text_input_line(QSP_ARG_DECL  int so_code)
{
	Screen_Obj *to;
	const char *s;

	to = get_parts("text");		// sets name and action

	s = nameof("default value");

	if( to == NULL ) return;

	SET_SOB_CONTENT(to, save_possibly_empty_str(s));
	SET_SOB_TYPE(to, so_code);

#ifdef BUILD_FOR_IOS
	get_device_dims(to);	// sets width, height, and font size
#else
	SET_SOB_WIDTH(to, PO_WIDTH(curr_panel)-2*SIDE_GAP );
#endif // BUILD_FOR_IOS

	// height is set for console output...
	SET_SOB_HEIGHT(to, MESSAGE_HEIGHT);

	/*
	 * OLD COMMENT:
	 * we used to put this after addition to panel list,
	 * because setting initial value causes a callback
	 * (true for motif!!)
	 *
	 * Why did we stop doing that???
	 * We put it back for now, to prevent the warning...
	 *
	 * Note that most of the other functions add it to the
	 * panel list after the widget proper has been created...
	 *
	 * for iOS, add_to_panel must follow the widget creation,
	 * because that is where addSubview is called...
	 */
#ifdef BUILD_FOR_OBJC
	make_text_field(QSP_ARG  to);
//fprintf(stderr,"adding text field to panel...\n");
	add_to_panel(curr_panel,to);
#else // ! BUILD_FOR_OBJC
	// the callback that occurs is a "value changed" callback...
	// Is that because the widget creation routine
	// sets an initial value?
	add_to_panel(curr_panel,to);
	make_text_field(QSP_ARG  to);
#endif // ! BUILD_FOR_OBJC

#ifdef BUILD_FOR_IOS
	install_initial_text(to);	// set the placeholder value here...
#endif /* BUILD_FOR_IOS */

#ifdef BUILD_FOR_MACOS
	INC_PO_CURR_Y(curr_panel, SOB_HEIGHT(to) + GAP_HEIGHT );
#else // ! BUILD_FOR_MACOS
	// Why add an extra 10???  BUG?
	INC_PO_CURR_Y(curr_panel, MESSAGE_HEIGHT + GAP_HEIGHT + 10 );
#endif // ! BUILD_FOR_MACOS

} // mk_text_input_line

COMMAND_FUNC( mk_text )
{
	mk_text_input_line(QSP_ARG  SOT_TEXT);
}

COMMAND_FUNC( mk_password )
{
	mk_text_input_line(QSP_ARG  SOT_PASSWORD);
}

COMMAND_FUNC( mk_edit_box )
{
	Screen_Obj *sop;
	const char *s;

	sop = get_parts("edit box");
	s = nameof("default value");
	if( sop == NULL ) return;
	SET_SOB_CONTENT(sop, save_possibly_empty_str(s));	// overloads action?
	SET_SOB_TYPE(sop, SOT_EDIT_BOX);
	add_to_panel(curr_panel,sop);
	/* put this after addition to panel list, because setting initial value causes a callback */
	make_edit_box(QSP_ARG  sop);

	/* BUG how tall is the edit box? */
	INC_PO_CURR_Y(curr_panel, SOB_HEIGHT(sop) + GAP_HEIGHT );
	//INC_PO_CURR_Y(curr_panel, MESSAGE_HEIGHT + GAP_HEIGHT );	// copied from mk_text_box???
}

COMMAND_FUNC( assign_text )
{
	const char *s;
	const char *v;
	Screen_Obj *sop;
	/* char msg[80]; */

	sop = pick_scrnobj("");
	s = nameof("variable name");
	if( sop == NULL ) return;

	/* Before we try to get the text, we should check which type of widget we have */
	if( ! IS_TEXT(sop) ){
		sprintf(ERROR_STRING,"assign_text:  widget %s is a %s, not a text object",
			SOB_NAME(sop),WIDGET_TYPE_NAME(sop));
		WARN(ERROR_STRING);
		return;
	}

	v=get_text(sop);

	/* PAS - Check that v is not nil */
	if(v)
		assign_var(s,v);
	else
		assign_var(s,"");
}

COMMAND_FUNC( do_set_prompt )
{
	const char *s;
	Screen_Obj *sop;

	sop = pick_scrnobj("");
	s = nameof("new prompt");
	if( sop == NULL ) return;
	if( BAD_STRING(s) ) return;	// a non-existent variable, for example...
	givbuf((void *)SOB_ACTION(sop));
	SET_SOB_ACTION(sop, savestr(s));
	update_prompt(sop);
}

COMMAND_FUNC( do_set_edit_text )
{
	const char *s;
	Screen_Obj *sop;

	sop = pick_scrnobj("");
	s = nameof("text to display");
	if( sop == NULL ) return;
	if( BAD_STRING(s) ) return;	// a non-existent variable, for example...

	// BUG make sure SOT_EDIT_BOX
	if( SOB_CONTENT(sop) != NULL )
		rls_str(SOB_CONTENT(sop));
	SET_SOB_CONTENT(sop,savestr(s));

	update_edit_text(sop,s);
}

COMMAND_FUNC( do_set_text_field )
{
	const char *s;
	Screen_Obj *sop;

	sop = pick_scrnobj("");
	s = nameof("text to display");
	if( sop == NULL ) return;

	// BUG - make sure this is an appropriate object

	// BUG?  should we save this in SOB_CONTENT?
	update_text_field(sop,s);
}

COMMAND_FUNC( mk_gauge )
{
	Screen_Obj *gop;

	int minval;
	int maxval;
	int val;

	gop=get_parts("gauge");
	get_min_max_val(&minval,&maxval,&val);
	if( gop == NULL ) return;
	SET_SOB_TYPE(gop, SOT_GAUGE);
	SET_SOB_MIN(gop,minval);
	SET_SOB_MAX(gop,maxval);
	SET_SOB_VAL(gop,val);
	make_gauge(QSP_ARG  gop);

	add_to_panel(curr_panel,gop);

	INC_PO_CURR_Y(curr_panel, GAUGE_HEIGHT + GAP_HEIGHT );
}

COMMAND_FUNC( set_panel_label )
{
	const char *s;

	s=nameof("new panel label");
	label_panel(curr_panel,s);
}

COMMAND_FUNC( set_new_range )
{
	Screen_Obj *sop;
	int min;
	int max;

	sop=pick_scrnobj("slider");
	min=(int)how_many("min value");
	max=(int)how_many("max value");
	if( sop == NULL ) return;
	new_slider_range(sop,min,max);
}


COMMAND_FUNC( set_new_pos )
{
	Screen_Obj *sop;
	int val;

	sop=pick_scrnobj("slider");
	val=(int)how_many("value");
	if( sop == NULL ) return;
	new_slider_pos(sop,val);
}

#define DEFAULT_SLIDER_WIDTH	320	// for iOS, includes side gaps


#define FINISH_SLIDER(type)				\
							\
	SET_SOB_TYPE(sop, type);			\
	SET_SOB_MIN(sop,min);				\
	SET_SOB_MAX(sop,max);				\
	SET_SOB_VAL(sop,val);				\
	if( type == SOT_SLIDER )			\
		make_slider(sop);			\
	else if( type == SOT_ADJUSTER )			\
		make_adjuster(QSP_ARG  sop);		\
	else {						\
sprintf(ERROR_STRING,"FINISH_SLIDER:  bad slider type!?");	\
		WARN(ERROR_STRING);			\
		make_slider(sop);			\
	}						\
	add_to_panel(curr_panel,sop);			\
	INC_PO_CURR_Y(curr_panel, SOB_HEIGHT(sop) + GAP_HEIGHT );

COMMAND_FUNC( mk_slider )
{
	Screen_Obj *sop;
	int min, max, val;

	sop = get_parts("slider");
	get_min_max_val(&min,&max,&val);
	if( sop == NULL ) return;
	SET_SOB_WIDTH(sop,DEFAULT_SLIDER_WIDTH);
	FINISH_SLIDER(SOT_SLIDER)
}

COMMAND_FUNC( mk_adjuster )
{
	Screen_Obj *sop;
	int min, max, val;

	sop = get_parts("adjuster");
	get_min_max_val(&min,&max,&val);
	if( sop == NULL ) return;
	SET_SOB_WIDTH(sop,DEFAULT_SLIDER_WIDTH);
	FINISH_SLIDER(SOT_ADJUSTER)
}

COMMAND_FUNC( mk_slider_w )
{
	Screen_Obj *sop;
	int min;
	int max;
	int val;
	int width;

	sop = get_parts("slider");
	get_min_max_val(&min,&max,&val);
	get_so_width(&width);
	if( sop == NULL ) return;
	SET_SOB_WIDTH(sop,width);
	FINISH_SLIDER(SOT_SLIDER)
}

COMMAND_FUNC( mk_adjuster_w )
{
	Screen_Obj *sop;
	int min;
	int max;
	int val;
	int width;

	sop = get_parts("slider");
	get_min_max_val(&min,&max,&val);
	get_so_width(&width);
	if( sop == NULL ) return;
	SET_SOB_WIDTH(sop,width);
	FINISH_SLIDER(SOT_ADJUSTER)
}

// almost the same as mk_message...
COMMAND_FUNC( mk_label )
{
	Screen_Obj *mp;

	// BUG?  why get action text for a message???
	// Here the action text is not the action, it is the message.
	// The message can be different from the name.
	mp=get_parts("label");
	if( mp == NULL ) return;

	SET_SOB_TYPE(mp, SOT_LABEL);
	SET_SOB_CONTENT(mp, SOB_ACTION(mp));
	SET_SOB_ACTION(mp, NULL);

	// BUG can we do something better than hard-coding this???
	SET_SOB_HEIGHT(mp, MESSAGE_HEIGHT);
	// what sets the width?
	make_label(mp);
	add_to_panel(curr_panel,mp);

	// BUG - when we can wrap lines, the height will be
	// variable.  How do we determine the height from
	// the content?
	INC_PO_CURR_Y(curr_panel, SOB_HEIGHT(mp) + GAP_HEIGHT );
}

COMMAND_FUNC( mk_message )
{
	Screen_Obj *mp;

	// BUG?  why get action text for a message???
	// Here the action text is not the action, it is the message.
	// The message can be different from the name.
	mp=get_parts("message");
	if( mp == NULL ) return;
	SET_SOB_TYPE(mp, SOT_MESSAGE);
	/* get_parts gets the name and the action text, but for a message
	 * we have no action, just content_text
	 */
	SET_SOB_CONTENT(mp, SOB_ACTION(mp));
	SET_SOB_ACTION(mp, NULL);

	//get_device_dims(mp);	// sets width, height, and font size
	// height is set for console output...
	SET_SOB_HEIGHT(mp, MESSAGE_HEIGHT);

	make_message(QSP_ARG  mp);
	add_to_panel(curr_panel,mp);
	INC_PO_CURR_Y(curr_panel, MESSAGE_HEIGHT + GAP_HEIGHT );
}


COMMAND_FUNC( mk_text_box )
{
	Screen_Obj *tb;

	// BUG?  why get action text for a text_box???
	// Here the action text is not the action, it is the initial text.

	tb=get_parts("initial text");
#ifdef BUILD_FOR_IOS
	if( tb == NULL ) return;
	SET_SOB_TYPE(tb, SOT_TEXT_BOX);
	/* get_parts gets the name and the action text, but for a message
	 * we have no action, just content_text
	 */
	SET_SOB_CONTENT(tb, SOB_ACTION(tb));
	SET_SOB_ACTION(tb, NULL);

	// I don't get it - is this a message or a text box???

	make_message(QSP_ARG  tb);
	add_to_panel(curr_panel,tb);

	make_text_box(QSP_ARG  tb, NO /* not editable */ );
#else // ! BUILD_FOR_IOS
	fprintf(stderr,"tb = 0x%lx\n",(long) tb);	// suppress compiler unused value warning
#endif /* BUILD_FOR_IOS */

	INC_PO_CURR_Y(curr_panel, MESSAGE_HEIGHT + GAP_HEIGHT );
}

COMMAND_FUNC( mk_act_ind )
{
#ifdef BUILD_FOR_IOS
	Screen_Obj *sop;
#endif /* BUILD_FOR_IOS */
	const char *name;

	name = nameof("name for activity indicator");
#ifdef BUILD_FOR_IOS
	sop = simple_object(QSP_ARG  name);
	if( sop == NULL ) return;
	SET_SOB_TYPE(sop, SOT_ACTIVITY_INDICATOR);
	make_act_ind(QSP_ARG  sop);
	add_to_panel(curr_panel,sop);

	// The activity indicator can come out in front of
	// other controls, we want to position it in the center.
	// We normally wouldn't have more than one per panel,
	// but that is not strictly enforced.
#else /* ! BUILD_FOR_IOS */
	// This is here simply to suppress a compiler warning...
	fprintf(stderr,"Not making activity indicator %s (iOS only)\n",name);
#endif /* ! BUILD_FOR_IOS */

	// just for testing - BUG find the real height of this widget
	INC_PO_CURR_Y(curr_panel, MESSAGE_HEIGHT + GAP_HEIGHT );
}

COMMAND_FUNC( do_enable_widget )
{
	Screen_Obj *sop;
	int yesno;

	sop=pick_scrnobj("widget");
	yesno=ASKIF("Enable widget");

	if( sop == NULL ) return;

	enable_widget(QSP_ARG  sop, yesno);
}


COMMAND_FUNC( do_hide_widget )
{
	Screen_Obj *sop;
	int yesno;

	sop=pick_scrnobj("widget");
	yesno=ASKIF("Hide widget");

	if( sop == NULL ) return;

	hide_widget(QSP_ARG  sop, yesno);
}


COMMAND_FUNC( do_show )
{
	Panel_Obj *po;

	po=pick_panel("");
	if( po != NULL ) show_panel(po);
}

#ifndef BUILD_FOR_OBJC

static void do_genwin_panel_posn(QSP_ARG_DECL  const char *s, int x, int y)
{
	Panel_Obj *po;
	po=get_panel_obj(s);
	if( po != NULL ) {
		SET_PO_X(po, x);
		SET_PO_Y(po, y);
		posn_panel(po);
	}
	return;
}

static void do_genwin_panel_show(QSP_ARG_DECL  const char *s)
{
	Panel_Obj *po;

	po=get_panel_obj(s);
	if( po != NULL ) show_panel(po);
	return;
}

static void do_genwin_panel_unshow(QSP_ARG_DECL  const char *s)
{
	Panel_Obj *po;

	po=get_panel_obj(s);
	if( po != NULL ) unshow_panel(po);
	return;
}

static void do_genwin_panel_delete(QSP_ARG_DECL  const char *s)
{
	Panel_Obj *po;
	po=get_panel_obj(s);
	if( po != NULL ) {
		WARN("sorry, don't know how to delete panel yet");
	}
	return;

}
#endif // BUILD_FOR_OBJC


COMMAND_FUNC( do_unshow )
{
	Panel_Obj *po;

	po=pick_panel("");
	if( po != NULL ) unshow_panel(po);
}

#define INSIST_GAUGE(gp,func)						\
									\
	if( ! IS_A_TYPE_OF_GAUGE(gp) ){					\
		sprintf(ERROR_STRING,					\
			"%s:  Widget %s is not a gauge/slider!?",	\
			#func,SOB_NAME(gp));				\
		WARN(ERROR_STRING);					\
		return;							\
	}

COMMAND_FUNC( do_set_gauge_label )
{
	Screen_Obj *gp;
	const char * s;

	gp=pick_scrnobj("guage");
	s=nameof("gauge lable");

	if( gp == NULL ) return;
	INSIST_GAUGE(gp,set_gauge_label)

	set_gauge_label(gp,s);
}

COMMAND_FUNC( do_set_gauge_value )
{
	Screen_Obj *gp;
	int n;

	gp=pick_scrnobj("guage");
	n=(int)how_many("setting");

	if( gp == NULL ) return;
	INSIST_GAUGE(gp,set_gauge_value)

	set_gauge_value(gp,n);
}

COMMAND_FUNC( do_set_toggle )
{
	Screen_Obj *sop;
	int state;

	sop=pick_scrnobj("toggle");
	state = ASKIF("toggle set");

	if( sop == NULL ) return;

	set_toggle_state(sop,state);
}


/* BUG - for radio buttons, we might have a max number of choices,
 * but in iOS there is really no limit for a picker...
 */

#define MAX_CHOICES 16

COMMAND_FUNC( do_set_choice )
{
	Screen_Obj *sop;
	int i;
	//const char *s;

	sop=pick_scrnobj("chooser");
	if( sop == NULL ){
		/*s=*/ nameof("dummy word");
		return;
	}
	/* BUG make sure sop points to a chooser... */
	if( IS_CHOOSER(sop) ){
//fprintf(stderr,"do_set_choice:  %s has %d choices...\n",
//SOB_NAME(sop),SOB_N_SELECTORS(sop));

		i = WHICH_ONE("choice",SOB_N_SELECTORS(sop),SOB_SELECTORS(sop));
		if( i < 0 ) return;
	} else if( SOB_TYPE(sop) == SOT_PICKER ){
		i = WHICH_ONE("choice", SOB_N_SELECTORS_AT_IDX(sop,0),
			SOB_SELECTORS_AT_IDX(sop,0) );
		if( i < 0 ) return;
	} else {
		sprintf(ERROR_STRING,"%s is not a chooser or a picker!?",
			SOB_NAME(sop));
		WARN(ERROR_STRING);
		return;
	}

	set_choice(sop,i);
}

// Set script variable $choice to the current selection.
// Simulates the action of making a choice without the user making it.
// This is useful to make the system reflect the current state after
// an item has been selected for deletion.

COMMAND_FUNC(do_get_choice)
{
	Screen_Obj *sop;

	sop=pick_scrnobj("chooser");
	if( sop == NULL ) return;

#ifdef BUILD_FOR_IOS
	get_choice(sop);
#else // ! BUILD_FOR_IOS
	WARN("do_get_choice:  not implemented for this platform!?");
#endif // ! BUILD_FOR_IOS
}

// We search the child list of the panel to find choosers, pickers
// This makes sure that nothing is selected, or for a picker
// that the first line is selected.

COMMAND_FUNC(do_clear_choices)
{
	Screen_Obj *sop;
	IOS_List *lp;
	IOS_Node *np;

	// In the EasyJet app, we call this as part
	// of the NextQ (next questionnaire) function,
	// but not all questionnaires have choices...
	// Exit silently if the widget does not exist...

	lp = PO_CHILDREN(curr_panel);
	if( lp == NULL ) return;
	np=IOS_LIST_HEAD(lp);
	while( np != NULL ){
		sop = (Screen_Obj *) IOS_NODE_DATA(np);
		if( IS_CHOOSER(sop) ){
			clear_all_selections(sop);
		} else if( SOB_TYPE(sop) == SOT_PICKER ) {
			int i;

			for(i=0;i<SOB_N_CYLINDERS(sop);i++)
				// set the the first entry...
				set_pick(sop,i,0);
		}
		np = IOS_NODE_NEXT(np);
	}
}

COMMAND_FUNC( do_set_message )
{
	Screen_Obj *mp;
	const char *s;

	mp=pick_scrnobj("message");
	s=nameof("message text");
	if( mp == NULL ) return;
	if( BAD_STRING(s) ) return;	// a non-existent variable, for example...

	// BUG make sure the screen_obj is the right type!

	givbuf((void *)SOB_CONTENT(mp));
	SET_SOB_CONTENT(mp, savestr(s));
	update_message(mp);
}

COMMAND_FUNC( do_set_label )
{
	Screen_Obj *mp;
	const char *s;

	mp=pick_scrnobj("message");
	s=nameof("message text");
#ifdef BUILD_FOR_IOS
	if( mp == NULL ) return;
	if( BAD_STRING(s) ) return;	// a non-existent variable, for example...

	// BUG make sure the screen_obj is the right type!

	givbuf((void *)SOB_CONTENT(mp));
	SET_SOB_CONTENT(mp, savestr(s));
	update_label(mp);
#else /* ! BUILD_FOR_IOS */
	WARN_ONCE("Sorry, resetting labels not yet supported for unix/motif!?");
	// Suppress compiler warnings
	sprintf(ERROR_STRING,"Can't install \"%s\" to message 0x%lx",s,(long)mp);
	advise(ERROR_STRING);
#endif /* ! BUILD_FOR_IOS */
}


COMMAND_FUNC( do_append_text )
{
	Screen_Obj *tb;
	const char *s;

	tb=pick_scrnobj("text_box");
	s=nameof("text to append");
#ifdef BUILD_FOR_IOS
	if( tb == NULL ) return;
	if( BAD_STRING(s) ) return;	// a non-existent variable, for example...

	// BUG make sure the screen_obj is the right type!

	givbuf((void *)SOB_CONTENT(tb));
	SET_SOB_CONTENT(tb, savestr(s));
	update_text_box(tb);
#else /* ! BUILD_FOR_IOS */
	sprintf(ERROR_STRING,"append_text:  not appending text \"%s\" to text box 0x%lx",
		s,(long)tb);
	advise(ERROR_STRING);
#endif /* ! BUILD_FOR_IOS */
}

COMMAND_FUNC( do_set_active )
{
	Screen_Obj *ai;
	int yesno;

	ai=pick_scrnobj("activity indicator");
	yesno = ASKIF("animate indicator");

#ifdef BUILD_FOR_IOS
	if( ai == NULL ) return;

	// BUG make sure this is the right type

	set_activity_indicator(ai,yesno);
#else /* ! BUILD_FOR_IOS */
	advise("set_active:  Sorry, not implemented");
	// suppress warnings
	sprintf(ERROR_STRING,"yesno = %d, indicator = 0x%lx",yesno,(long)ai);
	advise(ERROR_STRING);
#endif /* ! BUILD_FOR_IOS */
}

#ifndef BUILD_FOR_OBJC		// why do we not need this in iOS???
void del_so(QSP_ARG_DECL  Screen_Obj *sop)
{
	del_scrnobj(sop);	// delete from database...

	if( SOB_ACTION(sop) != NULL ) rls_str(SOB_ACTION(sop));
	if( SOB_SELECTOR(sop) != NULL ) rls_str(SOB_SELECTOR(sop));
}
#endif // ! BUILD_FOR_OBJC

#ifdef NOT_YET
COMMAND_FUNC( clear_screen )
{
	List *lp;
	Node *np;
	Screen_Obj *sop;
	Panel_Obj *po;

	po=pick_panel("");
	if( po==NULL ) return;

	lp=item_list(scrnobj_itp);
	assert( lp != NULL );

	np=QLIST_HEAD(lp);
	while( np != NULL ){
		sop = (Screen_Obj *)np->n_data;
		if( WIDGET_PANEL(sop) == po )
			del_so(sop);
		np=np->n_next;
	}
	del_po(po);
}

static void del_po(QSP_ARG_DECL  Panel_Obj *po)
{
	/* should deallocate any window system stuff first */
	free_wsys_stuff(po);

	del_panel_obj(po);
	givbuf((void *)PO_NAME(po));
}
#endif /* NOT_YET */

COMMAND_FUNC( do_pposn )
{
	Panel_Obj *po;
	int x;
	int y;

	po=pick_panel("");
	x=(int)how_many("x position");
	y=(int)how_many("y position");

	if( po != NULL ){
		SET_PO_X(po,x);
		SET_PO_Y(po,y);
		posn_panel(po);
	}
}

COMMAND_FUNC( do_delete )
{
	Panel_Obj *po;

	po=pick_panel("");
	if( po != NULL ){
		WARN("Sorry, don't know how to delete panels yet");
	}
}

// What is a notice?  an alert with two choices?

COMMAND_FUNC( do_notice )
{
	const char *msg_tbl[5];

	msg_tbl[0]=nameof("notice message");
	msg_tbl[1]=nameof("Yes prompt");
	msg_tbl[2]=nameof("Yes action");
	msg_tbl[3]=nameof("No prompt");
	msg_tbl[4]=nameof("No action");

	give_notice(msg_tbl);
}

COMMAND_FUNC( mk_scroller )
{
	Screen_Obj *sop;

	sop=get_parts("scroller");
	if( sop == NULL ) return;
	SET_SOB_TYPE(sop, SOT_SCROLLER);

	make_scroller(sop);
	add_to_panel(curr_panel,sop);
	INC_PO_CURR_Y(curr_panel, SCROLLER_HEIGHT + GAP_HEIGHT + 50 );
}

#define MAX_STRINGS	128

// get_strings prompts for a count, and then prompts for that
// number of strings.  It allocates a string array which must
// be freed later.  It used to set SOB_SELECTORS to this value,
// but that was incorrect for the motif implementation of pickers,
// so for choosers it is important to do this after calling get_strings...

int _get_strings(QSP_ARG_DECL Screen_Obj *sop,const char ***sss)
{
	const char **string_arr;
	int i;
	int n;

	n=(int)how_many("number of items");
	if( n < 0 ){
		SET_SOB_N_SELECTORS(sop,0);
		sprintf(ERROR_STRING,"get_strings:  number of selectors must be non-negative!?");
		WARN(ERROR_STRING);
		return -1;
	}
	// do this after returning
	//SET_SOB_N_SELECTORS(sop,n);

	/* so_action_text is set to some garbage */
	if( n > 0 ){
		string_arr = (const char **)getbuf( n * sizeof(char *) );
		// do this after returning
		// SET_SOB_SELECTORS(sop,string_arr);
		for(i=0;i<n;i++){
			const char *s;
			s = nameof("selector text");
			assert(s!=NULL);
			if( *s == 0 ){
				warn("get_strings:  null string!?");
				string_arr[i] = "BAD_STRING"; 
			} else {
				string_arr[i] = savestr(s);
			}
		}
		*sss = string_arr;
	} else {
		*sss = NULL;
	}

	return(n);
}

/* Called by "items" */
COMMAND_FUNC( do_set_scroller )
{
	const char **string_arr;
	Screen_Obj *sop;
	int n;

	sop = pick_scrnobj("scroller");
	if( sop==NULL ) return;

	n=get_strings(sop,&string_arr);
	if( n < 0 ) return;

	SET_SOB_N_SELECTORS(sop,n);
	SET_SOB_SELECTORS(sop,string_arr);

	set_scroller_list(sop,string_arr,n);
}

void _mk_it_scroller(QSP_ARG_DECL  Screen_Obj *sop,Item_Type *itp)
{
	List *lp;
	Node *np;
	int i;
	int n=0;
	const char **sp;
	const char *string_arr[MAX_STRINGS];
	Item *ip;

	lp=item_list(itp);
	if( lp == NULL ) return;
	np=QLIST_HEAD(lp);
	while(np!=NULL){
		ip=(Item *)np->n_data;
		if( n < MAX_STRINGS )
			string_arr[n]=ip->item_name;
		n++;
		np = np->n_next;
	}
	set_scroller_list(sop,string_arr,n);

	SET_SOB_SELECTORS(sop, (const char **)getbuf( n * sizeof(char *) ));
	sp = SOB_SELECTORS(sop);
	for(i=0;i<n;i++) sp[i]=string_arr[i];
}

#define NCHOICES	4

COMMAND_FUNC( do_item_scroller )
{
	Screen_Obj *sop;
	Item_Type *itp;

	sop = pick_scrnobj("scroller");
	itp = pick_ittyp("");
	if( sop==NULL || itp == NULL ) return;

	mk_it_scroller(sop,itp);
}

COMMAND_FUNC( do_file_scroller )
{
	int i;
	int n=0;
	Screen_Obj *sop;
	const char **sp;
	const char *string_arr[MAX_STRINGS];
	char word[BUF_LEN];
	FILE *fp;

	sop = pick_scrnobj("scroller");
	fp = try_open( nameof("item file"), "r" );
	if( !fp ) return;

	while( fscanf(fp,"%s",word) == 1 ){
		if( n < MAX_STRINGS )
			string_arr[n]=savestr(word);
		n++;
	}
	fclose(fp);
	set_scroller_list(sop,string_arr,n);

	SET_SOB_SELECTORS(sop, (const char **)getbuf( n * sizeof(char *) ));
	sp = SOB_SELECTORS(sop);
	for(i=0;i<n;i++) sp[i]=string_arr[i];
}

// what is a "mlt_chooser" ???

COMMAND_FUNC( do_mlt_chooser )
{
	Screen_Obj *sop;
	int n;
	const char **string_arr;

	sop=get_parts("mlt_chooser");
	if( sop == NULL ) return;
	SET_SOB_TYPE(sop, SOT_MLT_CHOOSER);

	// the string_arr is allocated for exactly n strings,
	// if we later want to add something we have to reallocate!?
	n=get_strings(sop,&string_arr);
	if( n < 0 ) return;

	SET_SOB_N_SELECTORS(sop,n);
	SET_SOB_SELECTORS(sop,string_arr);

	make_chooser(sop,n,string_arr);
	add_to_panel(curr_panel,sop);

//fprintf(stderr,"before increment, curr_y = %d\n",PO_CURR_Y(curr_panel));
	INC_PO_CURR_Y( curr_panel, SOB_HEIGHT(sop) + GAP_HEIGHT );
//fprintf(stderr,"after increment, curr_y = %d\n",PO_CURR_Y(curr_panel));
} // do_mlt_chooser

COMMAND_FUNC( do_chooser )
{
	Screen_Obj *sop;
	int n;
	const char **string_arr;

	sop=get_parts("chooser");
	if( sop == NULL ) return;
	SET_SOB_TYPE(sop, SOT_CHOOSER);

	// the string_arr is allocated for exactly n strings,
	// if we later want to add something we have to reallocate!?
	n=get_strings(sop,&string_arr);
	if( n < 0 ) return;

	SET_SOB_N_SELECTORS(sop,n);
	SET_SOB_SELECTORS(sop,string_arr);

	make_chooser(sop,n,string_arr);
	add_to_panel(curr_panel,sop);

//fprintf(stderr,"before increment, curr_y = %d\n",PO_CURR_Y(curr_panel));
	INC_PO_CURR_Y( curr_panel, SOB_HEIGHT(sop) + GAP_HEIGHT );
//fprintf(stderr,"after increment, curr_y = %d\n",PO_CURR_Y(curr_panel));
}

#define get_picker_strings(sop,n_cyl) _get_picker_strings(QSP_ARG  sop, n_cyl )

static void _get_picker_strings(QSP_ARG_DECL  Screen_Obj *sop, int n_cyl )
{
	int *count_tbl;
	int i, j, n_max;
	const char **string_arr;
	int n;	// set by get_strings?

	SET_SOB_SELECTOR_TBL(sop, (const char ***)getbuf( n_cyl * sizeof(char **) ));
	SET_SOB_N_CYLINDERS(sop, n_cyl );
	count_tbl = getbuf( n_cyl * sizeof(int) );
	SET_SOB_COUNT_TBL(sop,count_tbl);

	n_max=0;
	for(i=0;i<n_cyl;i++){
		const char **selectors;
		n=get_strings(sop,&string_arr);
		if( n < 0 ) return;	// BUG clean up!
		else if( n > 0 ){
			SET_SOB_N_SELECTORS_AT_IDX(sop, i, n);
			selectors = (const char **)getbuf( n * sizeof(char **) );
			SET_SOB_SELECTORS_AT_IDX(sop, i, selectors );

			for(j=0;j<n;j++){
				SET_SOB_SELECTOR_AT_IDX(sop,i,j,string_arr[j]);
			}
			givbuf(string_arr);
			if( n > n_max ) n_max = n;
		} else {
			assert(string_arr == NULL);
		}
	}
}

#define del_picks(sop) _del_picks(QSP_ARG  sop)

static void _del_picks( QSP_ARG_DECL  Screen_Obj *sop )
{
	int i, n_cyl;
	void *p;

	n_cyl = SOB_N_CYLINDERS(sop);
	assert( n_cyl >= 1 );

	for(i=0;i<n_cyl;i++){
		p = SOB_SELECTORS_AT_IDX(sop, i );
		givbuf(p);
		// these counts and pointers will be release later anyway
		//SET_SOB_N_SELECTORS_AT_IDX(sop, i, 0);
		//SET_SOB_SELECTORS_AT_IDX(sop, i, NULL );
	}

	p=SOB_SELECTOR_TBL(sop);
	givbuf(p);
	p=SOB_COUNT_TBL(sop);
	givbuf(p);

	SET_SOB_N_CYLINDERS(sop, 0 );
	SET_SOB_COUNT_TBL(sop,NULL);
	SET_SOB_SELECTOR_TBL(sop,NULL);
}

COMMAND_FUNC( do_picker )
{
	Screen_Obj *sop;
	int n_cyl;

	sop=get_parts("picker");
	if( sop == NULL ) return;
	SET_SOB_TYPE(sop, SOT_PICKER);

	n_cyl = (int)how_many("number of cylinders");
	if( n_cyl < 1 || n_cyl > MAX_CYLINDERS ){
		sprintf(ERROR_STRING,"Number of cylinders (%d) must be between 1 and %d!?",
			n_cyl,MAX_CYLINDERS);
		WARN(ERROR_STRING);
		return;
	}

	get_picker_strings(sop, n_cyl );

	make_picker(sop);
	add_to_panel(curr_panel,sop);

	INC_PO_CURR_Y( curr_panel, SOB_HEIGHT(sop) + GAP_HEIGHT );
}

#ifdef BUILD_FOR_IOS

static inline const char **new_selector_table_less_one(const char *choice_to_delete, const char **tbl, int old_n)
{
	const char **new_string_arr;
	int i,j, new_n;

	new_n = old_n - 1;

	if( new_n > 0 )
		new_string_arr = getbuf( new_n * sizeof(char *) );
	else
		return NULL;

	// First copy the addresses of the old non-matching strings
	j=0;
	for(i=0;i<old_n;i++){
		if( strcmp(tbl[i],choice_to_delete) ){
			if( j < new_n )
				new_string_arr[j] = tbl[i];
			j++;
		}
	}

	// Complain if match not found
	if( j == old_n ){
		sprintf(ERROR_STRING,
	"new_selector_table_less_one:  string \"%s\" not found among %d choices",
			choice_to_delete,old_n);
		WARN(ERROR_STRING);
		givbuf(new_string_arr);
		return NULL;
	}

	assert( j == new_n );	// deleted item should only appear once

	return new_string_arr;
}

static const char *find_string_in_table( const char **tbl, int n, const char *s )
{
	int i;

	for(i=0;i<n;i++){
		if( !strcmp(tbl[i],s) ) return tbl[i];
	}
	return NULL;
}

static inline int insist_choice_not_present(const char **tbl, int n, const char *s, const char *name, const char *whence)
{
	if( find_string_in_table(tbl,n,s) != NULL ){
		sprintf(ERROR_STRING,"%s:  choice \"%s\" is already present in chooser/picker \"%s\"!?\n",
			whence,s,name);
		WARN(ERROR_STRING);
		return -1;
	}
	return 0;
}

static inline int insist_choice_present(const char **tbl, int n, const char *s, const char *name, const char *whence)
{
	if( find_string_in_table(tbl,n,s) == NULL ){
		sprintf(ERROR_STRING,"%s:  choice \"%s\" is not present in chooser/picker \"%s\"!?\n",
			whence,s,name);
		WARN(ERROR_STRING);
		return -1;
	}
	return 0;
}

static inline void add_choice_to_chooser(QSP_ARG_DECL  Screen_Obj *sop, const char *s)
{
	int i, n;
	const char **new_string_arr;
	
	assert( SOB_N_SELECTORS(sop) >= 0 );

	// first make sure that this choice is not already present...
	if( insist_choice_not_present(SOB_SELECTORS(sop),SOB_N_SELECTORS(sop),s,SOB_NAME(sop),"add_choice_to_chooser") < 0 )
		return;

	n = SOB_N_SELECTORS(sop) + 1;
	new_string_arr = getbuf( n * sizeof(char *) );

	i=0;
	if( SOB_N_SELECTORS(sop) > 0 ){
		assert( SOB_SELECTORS(sop) != NULL );
		for(;i<SOB_N_SELECTORS(sop);i++)
			new_string_arr[i] = SOB_SELECTORS(sop)[i];
		givbuf(SOB_SELECTORS(sop));	// release old string table
	} else {
		assert( SOB_SELECTORS(sop) == NULL );
	}
		
	new_string_arr[i] = savestr(s);
	SET_SOB_N_SELECTORS(sop,n);
	SET_SOB_SELECTORS(sop,new_string_arr);
}

static inline int insist_one_component(QSP_ARG_DECL  Screen_Obj *sop, const char *whence)
{
	if( SOB_N_CYLINDERS(sop) != 1 ){
		sprintf(ERROR_STRING,
			"%s:  picker %s has more than one component (%d)",
			whence,SOB_NAME(sop),SOB_N_CYLINDERS(sop));
		WARN(ERROR_STRING);
		return -1;
	}
	return 0;
}

// If there is more than one component, we will need to have
// another command that takes the argument...

static void add_choice_to_picker(QSP_ARG_DECL  Screen_Obj *sop, const char *s)
{
	const char **new_selectors;
	int i, n;

	assert( SOB_TYPE(sop) == SOT_PICKER );
	if( insist_one_component(sop, "add_choice_to_picker") < 0 ) return;

	if( insist_choice_not_present(SOB_SELECTORS_AT_IDX(sop,0),SOB_N_SELECTORS_AT_IDX(sop,0),s,
			SOB_NAME(sop),"add_choice_to_picker") < 0 )
		return;

	n = 1 + SOB_N_SELECTORS_AT_IDX( sop, 0 );
	new_selectors = (const char **)getbuf( n * sizeof(char **) );

	i=0;
	if( SOB_N_SELECTORS_AT_IDX(sop,0) > 0 ){
		assert( SOB_SELECTORS_AT_IDX(sop,0) != NULL );
		for(;i<SOB_N_SELECTORS_AT_IDX(sop,0);i++)
			new_selectors[i] = SOB_SELECTOR_AT_IDX(sop,0,i);
		givbuf(SOB_SELECTORS_AT_IDX(sop,0));
	} else {
		assert( SOB_SELECTORS_AT_IDX(sop,0) == NULL );
	}

	new_selectors[i] = savestr(s);
	SET_SOB_N_SELECTORS_AT_IDX( sop, 0, n );
	SET_SOB_SELECTORS_AT_IDX(sop, 0, new_selectors );
}

static inline void delete_choice_from_chooser(QSP_ARG_DECL  Screen_Obj *sop, const char *s)
{
	const char **new_string_arr;

	assert( SOB_TYPE(sop) == SOT_CHOOSER );

	// first make sure that this choice is already present...
	if( insist_choice_present(SOB_SELECTORS(sop),SOB_N_SELECTORS(sop),s,SOB_NAME(sop),"delete_choice_from_chooser") < 0 )
		return;

	new_string_arr = new_selector_table_less_one(s,SOB_SELECTORS(sop),SOB_N_SELECTORS(sop));
	// can return NULL if table empty or item not found

	givbuf(SOB_SELECTORS(sop));	// release old string table
	SET_SOB_N_SELECTORS(sop,SOB_N_SELECTORS(sop)-1);
	SET_SOB_SELECTORS(sop,new_string_arr);
	reload_chooser(sop);
}

static inline void delete_choice_from_picker(QSP_ARG_DECL  Screen_Obj *sop, const char *s)
{
	const char **new_string_arr;

	assert( SOB_TYPE(sop) == SOT_PICKER );
	if( insist_one_component(sop, "add_choice_to_picker") < 0 ) return;

	if( insist_choice_present(SOB_SELECTORS_AT_IDX(sop,0),SOB_N_SELECTORS_AT_IDX(sop,0),s,SOB_NAME(sop),"delete_choice_from_picker") < 0 )
		return;

	new_string_arr = new_selector_table_less_one(s,SOB_SELECTORS_AT_IDX(sop,0),SOB_N_SELECTORS_AT_IDX(sop,0));
	// can return NULL if table empty or item not found

	givbuf(SOB_SELECTORS_AT_IDX(sop,0));	// release old string table
	SET_SOB_N_SELECTORS_AT_IDX(sop,0,SOB_N_SELECTORS_AT_IDX(sop,0)-1);
	SET_SOB_SELECTORS_AT_IDX(sop,0,new_string_arr);
	reload_chooser(sop);
}

#endif // BUILD_FOR_IOS

static inline Screen_Obj *pick_chooser_or_picker(SINGLE_QSP_ARG_DECL)
{
	Screen_Obj *sop;

	sop = pick_scrnobj("name of chooser or picker");
	if( sop == NULL ) return NULL;

	if( SOB_TYPE(sop) == SOT_CHOOSER ) return sop;
	if( SOB_TYPE(sop) == SOT_PICKER ) return sop;

	sprintf(ERROR_STRING,"Object %s is not a chooser or picker!?",SOB_NAME(sop));
	WARN(ERROR_STRING);

	return NULL;
}

/* In motif, a chooser is implemented with radio buttons, but in iOS it is a "cylinder"
 * that can accommodate a large number of entries without taking up any extra
 * screen real estate...
 */

COMMAND_FUNC( do_add_choice )
{
	Screen_Obj *sop;
	const char *s;

	sop = pick_chooser_or_picker(SINGLE_QSP_ARG);
	s = nameof("choice string");

	if( sop == NULL ) return;

#ifdef BUILD_FOR_IOS

	// increase the number of choices,
	// allocate a new string array,
	// copy the old choices and then
	// get the new one.

	// Make sure it's the right kind
	// BUG this should work for "scrollers" too!

	if( SOB_TYPE(sop) == SOT_PICKER ){
		add_choice_to_picker(sop, s);
	} else {
		assert( SOB_TYPE(sop) == SOT_CHOOSER );
		add_choice_to_chooser(sop, s);
	}
	reload_chooser(sop);

#else /* ! BUILD_FOR_IOS */

	WARN("Sorry, can't add choices in the X11 implementation.");
	// suppress warnings
	sprintf(ERROR_STRING,"Not adding \"%s\" to 0x%lx",s,(long)sop);
	advise(ERROR_STRING);

#endif /* ! BUILD_FOR_IOS */

}

// BUG for pickers, we might need to allow the user to specify the cylinder index.
// Here we implicitly assume there is only one (with index = 0).

COMMAND_FUNC( do_del_choice )
{
	Screen_Obj *sop;
	const char *s;

	sop = pick_chooser_or_picker(SINGLE_QSP_ARG);
	s = nameof("choice string");

	if( sop == NULL ) return;

#ifdef BUILD_FOR_IOS

	if( SOB_TYPE(sop) == SOT_CHOOSER ){
		delete_choice_from_chooser(sop,s);
	} else {
		assert( SOB_TYPE(sop) == SOT_PICKER );
		delete_choice_from_picker(sop,s);
	}

#else /* ! BUILD_FOR_IOS */

	WARN("Sorry, can't delete choices in the X11 implementation");
	// suppress warnings
	sprintf(ERROR_STRING,"Not deleting \"%s\" from 0x%lx",s,(long)sop);
	advise(ERROR_STRING);


#endif /* ! BUILD_FOR_IOS */
}


#ifdef BUILD_FOR_IOS
#define RELOAD_PICKER reload_picker(sop);
#else // ! BUILD_FOR_IOS
#define RELOAD_PICKER
#endif // ! BUILD_FOR_IOS

COMMAND_FUNC( do_set_picks )
{
	Screen_Obj *sop;
	int n_cyl;

	sop = pick_scrnobj("picker");

	// decrease the number of choices,
	// allocate a new string array,
	// copy the old choices (less one);
	// if we don't find a match during the
	// copy process, it is an error.

	n_cyl = (int) how_many("number of cylinders");

	if( n_cyl < 1 || n_cyl > MAX_CYLINDERS ){
		sprintf(ERROR_STRING,"Number of cylinders (%d) should be between %d and %d!?",n_cyl,1,MAX_CYLINDERS);
		WARN(ERROR_STRING);
		return;
	}

	if( sop == NULL ) return;

	// Make sure it's the right kind
	// BUG this should work for "scrollers" too!
	// BUG need to implement for pickers too...
	if( SOB_TYPE(sop) != SOT_PICKER ){
		sprintf(ERROR_STRING,"set_picks:  object %s is not a picker!?",SOB_NAME(sop));
		WARN(ERROR_STRING);
		return;
	}

	del_picks( sop );

	get_picker_strings(sop, n_cyl );

	RELOAD_PICKER

//#ifdef BUILD_FOR_IOS
//#else /* ! BUILD_FOR_IOS */
//
//	WARN("Sorry, can't delete picks in the X11 implementation");
//
//#endif /* ! BUILD_FOR_IOS */

}

#ifdef NOT_YET
Node *first_panel_node(SINGLE_QSP_ARG_DECL)
{
	List *lp;

	lp=item_list(panel_obj_itp);
	if( lp==NULL ) return NULL;
	else return(QLIST_HEAD(lp));
}
#endif /* NOT_YET */

#ifdef NOT_USED
Screen_Obj *find_object_at(Panel_Obj *po,int x,int y)
{
	Node *np;
	Screen_Obj *sop;

	np=QLIST_HEAD(PO_CHILDREN(po));
	while(np!=NULL){
		sop=(Screen_Obj *)np->n_data;
		if(	   x >= SOB_X(sop)
			&& x <  (int) (SOB_X(sop)+SOB_DX(sop))
			&& y >= SOB_Y(sop)
			&& y < (int) (SOB_Y(sop)+SOB_DY(sop)) )
			return(sop);
		np=np->n_next;
	}
	return NULL;
}
#endif /* NOT_USED */

#ifndef BUILD_FOR_OBJC

static Genwin_Functions gwfp={
	do_genwin_panel_posn,
	do_genwin_panel_show,
	do_genwin_panel_unshow,
	do_genwin_panel_delete
};

#endif // ! BUILD_FOR_OBJC

static double get_panel_size(QSP_ARG_DECL  IOS_Item *ip,int index)
{
	double d;
	Panel_Obj *po;

	po = (Panel_Obj *)ip;

	switch(index){
		case 1:	d = PO_WIDTH(po); break;
		case 2:	d = PO_HEIGHT(po); break;
		default: d=1.0; break;
	}
	return(d);
}

static double get_scrnobj_size(QSP_ARG_DECL  IOS_Item *ip,int index)
{
	double d;
	Screen_Obj *sop;

	sop = (Screen_Obj *)ip;

	switch(index){
		case 1:	d = SOB_WIDTH(sop); break;
		case 2:	d = SOB_HEIGHT(sop); break;
		default: d=1.0; break;
	}
	return(d);
}

// can be static, but not static here to find multiple symbols...

IOS_Size_Functions panel_sf={
		get_panel_size,
		(const char * (*)(QSP_ARG_DECL  IOS_Item *))default_prec_name
};

IOS_Size_Functions scrnobj_sf={
		get_scrnobj_size,
		(const char * (*)(QSP_ARG_DECL  IOS_Item *))default_prec_name
};

void _so_init(QSP_ARG_DECL  int argc,const char **argv)
{
	static int so_inited=0;

	if( so_inited ) return;
	window_sys_init(SINGLE_QSP_ARG);

#ifdef HAVE_MOTIF
	motif_init(argv[0]);
#endif /* HAVE_MOTIF */

#ifndef BUILD_FOR_OBJC

	if(panel_obj_itp == NULL)
		init_panel_objs();

	/* support for genwin */
	add_genwin(panel_obj_itp, &gwfp, NULL);
	
	add_sizable(panel_obj_itp,&panel_sf, NULL );

	// scrnobj_itp is null at this point!?
	if( scrnobj_itp == NULL )
		init_scrnobjs();
	add_sizable(scrnobj_itp,&scrnobj_sf, NULL );
	
#else

	//if(panel_obj_itp == NULL)
	//	init_panel_objs(SINGLE_QSP_ARG);

#endif /* ! BUILD_FOR_OBJC */

// We need to have genwin's be sizable for ios too!?

	so_inited=1;
}

double panel_exists(QSP_ARG_DECL  const char *name)
{
	Panel_Obj *po;
	po = panel_obj_of(name);

	if( po == NULL ) return(0);
	return(1.0);
}

