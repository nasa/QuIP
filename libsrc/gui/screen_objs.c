#include "quip_config.h"

char VersionId_gui_screen_objs[] = QUIP_VERSION_STRING;

#ifdef HAVE_MOTIF

/*
 * General screen object stuff, pretty independent of window system
 */

#include <stdio.h>

#include "gui.h"
#include "gui_cmds.h"

#include "items.h"
#include "getbuf.h"
#include "debug.h"
#include "savestr.h"
#include "version.h"
#include "xsupp.h"
#include "function.h"
#include "gen_win.h"	/* add_genwin() */
#include "my_motif.h"

Panel_Obj *curr_panel=NO_PANEL_OBJ;
#define BUF_LEN 128
#define COL_STR	":"

/* local prototypes */
static void list_widgets(Panel_Obj *po);

//static void panel_obj_init(void);
static void get_menu_items(QSP_ARG_DECL  Screen_Obj *mp);
#define GET_MENU_ITEMS(mp)		get_menu_items(QSP_ARG  mp)
static void del_so(QSP_ARG_DECL  Screen_Obj *sop);
static void del_po(QSP_ARG_DECL  Panel_Obj *po);


ITEM_INTERFACE_DECLARATIONS(Panel_Obj,panel_obj)
#define GET_PANEL_OBJ(s)	get_panel_obj(QSP_ARG  s)

/* local prototypes */

ITEM_INTERFACE_DECLARATIONS(Screen_Obj,scrnobj)

#define PICK_SCRNOBJ(pmpt)		pick_scrnobj(QSP_ARG  pmpt)

#define BASENAME	"Base"

#define MAX_STACK 8
static Screen_Obj *curr_parent, *parent_stack[MAX_STACK];
static int parent_index=(-1);

/* from gui.h:
#define SO_TEXT		1
#define SO_BUTTON	2
#define SO_SLIDER	3
#define SO_MENU		4
#define SO_MENU_ITEM	5
#define SO_GAUGE	6
#define SO_MESSAGE	7
#define SO_PULLRIGHT	8
#define SO_MENU_BUTTON	9
#define SO_SCROLLER	10
#define SO_CHOOSER	10
#define SO_TOGGLE	11
*/

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

void show_panel_children(Panel_Obj *po)
{
	Node *np;
	Screen_Obj *sop;

	np=po->po_children->l_head;
	while(np!=NO_NODE){
		sop=(Screen_Obj *)np->n_data;
		if( sop != NULL ){
			sprintf(DEFAULT_ERROR_STRING,"\t%s",sop->so_name);
			advise(DEFAULT_ERROR_STRING);
		} else {
			advise("\tnull screen_obj!?");
		}
		np=np->n_next;
	}
}

#define WIDGET_TYPE_NAME(sop)		widget_type_name[WIDGET_INDEX(sop)]

Screen_Obj *simple_object(QSP_ARG_DECL  const char *name)
{
	Screen_Obj *sop;

	sop = new_scrnobj(QSP_ARG  name);
	if( sop == NO_SCREEN_OBJ ) return(sop);

	sop->so_action_text = sop->so_selector = NULL;
	sop->so_panel = curr_panel;
	sop->so_parent = NO_SCREEN_OBJ;
	sop->so_frame = NULL;
	sop->so_val=0;		/* so_nlist too! */
	sop->so_min=sop->so_max=0;
	sop->so_flags = 0;
#ifdef THREAD_SAFE_QUERY
	sop->so_qsp = qsp;
#endif /* THREAD_SAFE_QUERY */
	return(sop);
}

Screen_Obj *dup_so(QSP_ARG_DECL  Screen_Obj *sop)
{
	Screen_Obj *dup;
	char name[BUF_LEN];

	sprintf(name,"%s.dup",sop->so_name);
	dup = simple_object(QSP_ARG  name);
	if( sop == NO_SCREEN_OBJ ) return(sop);
	dup->so_parent = sop->so_parent;
	dup->so_panel = sop->so_panel;
	dup->so_frame = sop->so_frame;
	dup->so_action_text=savestr(sop->so_action_text);
	dup->so_selector=NULL;
	return(dup);
}

void so_info(Screen_Obj *sop)
{
	if( sop==NO_SCREEN_OBJ ) return;

	printf("%s\t\t%s\n",sop->so_name,WIDGET_TYPE_NAME(sop));
	if( sop->so_selector != NULL )
		printf("\tselector:\t%s\n",sop->so_selector);
	if( sop->so_action_text != NULL )
		printf("\taction text:\t%s\n",sop->so_action_text);
	if( WIDGET_PANEL(sop) != NO_PANEL_OBJ )
		printf("\tpanel:\t%s\n",WIDGET_PANEL(sop)->po_name);
	/* else printf("no frame object; this must be a frame itself\n"); */
}

COMMAND_FUNC( do_so_info )
{
	Screen_Obj *sop;

	sop=PICK_SCRNOBJ("");
	so_info(sop);
}

/* Sets position used to create next object */
COMMAND_FUNC( mk_position )
{
	int x, y;

	x = HOW_MANY("x position");
	y = HOW_MANY("y position");
	if( !curr_panel ) return;
	curr_panel->po_currx = x;
	curr_panel->po_curry = y;
}

/* Gets object x,y position and inserts value in strs x_val & y_val */
COMMAND_FUNC( do_get_posn_object )
{
	Screen_Obj *sop;
	char str[BUF_LEN];

	sop = PICK_SCRNOBJ("");
	if( sop == NO_SCREEN_OBJ ) return;
	sprintf(str, "%d", sop->so_x);
	ASSIGN_VAR("x_val",str);
	sprintf(str, "%d", sop->so_y);
	ASSIGN_VAR("y_val",str);
}

/* Moves an object */
COMMAND_FUNC( do_set_posn_object )
{
	Screen_Obj *sop;
	int x,y;

	sop = PICK_SCRNOBJ("");
	x=HOW_MANY("x position");
	y=HOW_MANY("y position");
	if( sop == NO_SCREEN_OBJ ) return;
	sop->so_x=x;
	sop->so_y=y;
	reposition(sop);
}

Panel_Obj *new_panel(QSP_ARG_DECL  const char *name,int dx,int dy)
{
	Panel_Obj *po;
char str[BUF_LEN];

	po = new_panel_obj(QSP_ARG  name);
	if( po == NO_PANEL_OBJ )
		return(po);

	po->po_width = dx;
	po->po_height = dy;
	po->po_x = 100;
	po->po_y = 100;
	po->po_flags = 0;
	po->po_children = new_list();

	make_panel(QSP_ARG  po);			/* Xlib calls */

sprintf(str,"%s panel objects",po->po_name);

	if( scrnobj_itp == NULL ) scrnobj_init(SINGLE_QSP_ARG);

	po->po_icp = create_item_context(QSP_ARG  scrnobj_itp,po->po_name);

	po->po_currx=
	po->po_curry=OBJECT_GAP;

	curr_panel=po;
	return(po);
}

COMMAND_FUNC( mk_panel )
{
	const char *s;
	Panel_Obj *po;
	int dx,dy;
	
	s=NAMEOF("name for panel");
	dx=HOW_MANY("panel width");
	dy=HOW_MANY("panel height");
	po=new_panel(QSP_ARG  s,dx,dy);
}

COMMAND_FUNC( do_list_panel_objs )
{
	Panel_Obj *po;

	po=PICK_PANEL("");
	if( po == NO_PANEL_OBJ ) return;

	list_widgets(po);
}

static void list_widgets(Panel_Obj *po)
{
	List *lp;
	Node *np;

	/* lp=item_list(QSP_ARG  scrnobj_itp); */
	lp = po->po_children;
	np=lp->l_head;
	while(np!=NO_NODE ){
		Screen_Obj *sop;

		sop=(Screen_Obj *)np->n_data;
		if( WIDGET_PANEL(sop) == po ){
			/* list_scrnobj(sop); */
		//	printf("\t%s\n",sop->so_name);
			so_info(sop);
		} else {
			sprintf(DEFAULT_ERROR_STRING,"widget %s does not belong to panel %s!?",
				sop->so_name,po->po_name);
			NWARN(DEFAULT_ERROR_STRING);
		}
		np=np->n_next;
	}
}


/*
 *  Prompt for object name, panel, and action text
 *  If ptr to return value for panel is NULL, don't ask for panel
 *
 *  class_str is imbedded in the prompts
 */

Screen_Obj *get_parts(QSP_ARG_DECL  const char *class_str)
{
	char pmpt[BUF_LEN];
	char text[BUF_LEN];
	char label[BUF_LEN];
	Screen_Obj *sop;

	sprintf(pmpt,"%s label",class_str);
	strcpy( label, NAMEOF(pmpt) );

	sprintf(pmpt,"%s action text",class_str);
	strcpy( text, NAMEOF(pmpt) );

	if( curr_panel == NO_PANEL_OBJ ) return(NO_SCREEN_OBJ);

	sop = simple_object(QSP_ARG  label);
	if( sop == NO_SCREEN_OBJ ) return(sop);

	sop->so_action_text = savestr(text);
	return(sop);
}

Screen_Obj *mk_menu(QSP_ARG_DECL  Screen_Obj *mip)
{
	Screen_Obj *mp;

	char buf[BUF_LEN];

	mp=dup_so(QSP_ARG  mip);
	givbuf((void *)mp->so_name);

	/* append colon to name */
	strcpy(buf,mip->so_name);
	strcat(buf,COL_STR);
	mp->so_name=savestr(buf);

#ifdef DEBUG
if( debug ) fprintf(stderr,"Making menu \"%s\"\n",mp->so_name);
#endif /* DEBUG */

	make_menu(QSP_ARG  mp,mip);

	GET_MENU_ITEMS(mp);

#ifdef DEBUG
if( debug ) fprintf(stderr,"make_menu:  back from get_menu_items (menu %s)\n",
mp->so_name);
#endif /* DEBUG */
	mp->so_flags |= SO_MENU;
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

	bp=GET_PARTS("menu");
	if( bp == NO_SCREEN_OBJ ) return;
	bp->so_flags |= SO_MENU_BUTTON;

	mp=MK_MENU(bp);

	/* The popup menu is created in mk_menu, don't
	   need to make this call from MOTIF */
#ifndef MOTIF
	make_menu_button(QSP_ARG  bp,mp);
#endif /* !MOTIF */

	addHead(curr_panel->po_children,mk_node(bp));
	bp->so_parent = curr_panel;

	/* What is the 30??? */
	curr_panel->po_curry += BUTTON_HEIGHT + GAP_HEIGHT + 30;
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

	mip->so_selector=savestr(mip->so_name);
	strcpy(buf,parent->so_name);
	strcat(buf,".");
	strcat(buf,mip->so_selector);
	rename_item(QSP_ARG  scrnobj_itp,mip,buf);
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
	else curr_parent = NO_SCREEN_OBJ;
}

COMMAND_FUNC( end_menu )
{
#ifdef DEBUG
if( debug ){
fprintf(stderr,"Popping menu \"%s\"\n",curr_parent->so_name);
}
#endif /* DEBUG */
	pop_parent(SINGLE_QSP_ARG);
#ifdef DEBUG
if( debug ){
if( curr_parent != NO_SCREEN_OBJ )
fprintf(stderr,"New parent menu \"%s\"\n",curr_parent->so_name);
else
WARN("all parent menus popped!!");
}
#endif /* DEBUG */
	popcmd(SINGLE_QSP_ARG);
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

	mip=GET_PARTS("menu");
	if( mip==NO_SCREEN_OBJ ) return;
	mip->so_panel = curr_parent->so_panel;
	mip->so_flags |= SO_MENU_ITEM;
	addHead(curr_panel->po_children,mk_node(mip));
	mip->so_parent = curr_panel;
	make_menu_choice(QSP_ARG  mip,curr_parent);
	fix_names(QSP_ARG  mip,curr_parent);
}

COMMAND_FUNC( do_pullright )
{
	Screen_Obj *pr, *mip;

	mip=GET_PARTS("menu");
	if( mip==NO_SCREEN_OBJ ) return;
	mip->so_panel = curr_parent->so_panel;
	mip->so_parent = NO_SCREEN_OBJ;
	mip->so_flags |= SO_PULLRIGHT;


	/* now dup the names for the menu */
	/* need to get the submenu */
	pr=MK_MENU(mip);

	make_pullright(QSP_ARG  mip,pr,curr_parent);
	fix_names(QSP_ARG  mip,curr_parent);
}

Command mi_ctbl[]={
{ "button",	do_normal,	"specify normal menu item"	},
{ "pullright",	do_pullright,	"specify pullright submenu"	},
{ "end_menu",	end_menu,	"quit adding menu items"	},
{ NULL,		NULL,		NULL				}
};

static void get_menu_items(QSP_ARG_DECL  Screen_Obj *mp)
{
	int depth;

	push_parent(mp);
	PUSHCMD(mi_ctbl,"menu_items");

	/* now need to do_cmd() until menu is exited */

	depth = cmd_depth(SINGLE_QSP_ARG);
	while( depth==cmd_depth(SINGLE_QSP_ARG) ) {
		do_cmd(SINGLE_QSP_ARG);
	}
}

void get_min_max_val(QSP_ARG_DECL  int *minp,int *maxp,int *valp)
{
	*minp=HOW_MANY("min value");
	*maxp=HOW_MANY("max value");
	*valp=HOW_MANY("initial value");
}

void get_so_width(QSP_ARG_DECL int *widthp)
{
	*widthp=HOW_MANY("width");
}

COMMAND_FUNC( mk_button )
{
	Screen_Obj *bo;

	bo = GET_PARTS("button");
	if( bo == NO_SCREEN_OBJ ) return;
	bo->so_flags |= SO_BUTTON;

	make_button(QSP_ARG  bo);

	addHead(curr_panel->po_children,mk_node(bo));
	bo->so_parent = curr_panel;

	curr_panel->po_curry += BUTTON_HEIGHT + GAP_HEIGHT;
}

COMMAND_FUNC( mk_toggle )
{
	Screen_Obj *to;

	to = GET_PARTS("toggle");
	if( to == NO_SCREEN_OBJ ) return;
	to->so_flags |= SO_TOGGLE;

	make_toggle(QSP_ARG  to);

	addHead(curr_panel->po_children,mk_node(to));
	to->so_parent = curr_panel;

	curr_panel->po_curry += TOGGLE_HEIGHT + GAP_HEIGHT;
}

COMMAND_FUNC( mk_text )
{
	Screen_Obj *to;
	const char *s;

	to = GET_PARTS("text");

	s = NAMEOF("default value");

	if( to == NO_SCREEN_OBJ ) return;

	to->so_content_text = savestr(s);

	to->so_flags |= SO_TEXT;

	addHead(curr_panel->po_children,mk_node(to));
	to->so_parent = curr_panel;

	/* put this after addition to panel list, because setting initial value causes a callback */
	make_text_field(QSP_ARG  to);

	curr_panel->po_curry += MESSAGE_HEIGHT + GAP_HEIGHT + 10;
}


COMMAND_FUNC( mk_edit_box )
{
	Screen_Obj *to;
	const char *s;

	to = GET_PARTS("edit box");

	s = NAMEOF("default value");

	if( to == NO_SCREEN_OBJ ) return;

	to->so_content_text = savestr(s);

	to->so_flags |= SO_TEXT;

	addHead(curr_panel->po_children,mk_node(to));
	to->so_parent = curr_panel;

	/* put this after addition to panel list, because setting initial value causes a callback */
	make_edit_box(QSP_ARG  to);

	/* BUG how tall is the edit box? */
	curr_panel->po_curry += MESSAGE_HEIGHT + GAP_HEIGHT + 10;
}

COMMAND_FUNC( assign_text )
{
	const char *s,*v;
	Screen_Obj *sop;
	/* char msg[80]; */

	sop = PICK_SCRNOBJ("");
	s = NAMEOF("variable name");
	if( sop == NO_SCREEN_OBJ ) return;

	/* Before we try to get the text, we should check which type of widget we have */
	if( ! IS_TEXT(sop) ){
		sprintf(ERROR_STRING,"assign_text:  widget %s is a %s, not a text object",
			sop->so_name,WIDGET_TYPE_NAME(sop));
		WARN(ERROR_STRING);
		return;
	}

	v=get_text(sop);

	/* PAS - Check that v is not nil */
        if(v)
	   ASSIGN_VAR(s,v);
	else
		ASSIGN_VAR(s,"");
	/*
	{
		sprintf(msg, "variable name %s is nill", s);
		WARN(msg);
	}
	*/
}

COMMAND_FUNC( do_set_prompt )
{
	const char *s;
	Screen_Obj *sop;

	sop = PICK_SCRNOBJ("");
	s = NAMEOF("new prompt");
	if( sop == NO_SCREEN_OBJ ) return;
	givbuf((void *)sop->so_action_text);
	sop->so_action_text = savestr(s);
	update_prompt(sop);
}

COMMAND_FUNC( do_set_edit_text )
{
	const char *s;
	Screen_Obj *sop;

	sop = PICK_SCRNOBJ("");
	s = NAMEOF("text to display");
	if( sop == NO_SCREEN_OBJ ) return;
	update_edit_text(sop,s);
}

COMMAND_FUNC( do_set_text_field )
{
	const char *s;
	Screen_Obj *sop;

	sop = PICK_SCRNOBJ("");
	s = NAMEOF("text to display");
	if( sop == NO_SCREEN_OBJ ) return;
	update_text_field(sop,s);
}

COMMAND_FUNC( mk_gauge )
{
	Screen_Obj *go;
	int min,max,val;

	go=GET_PARTS("gauge");
	GET_MIN_MAX_VAL(&min,&max,&val);
	if( go == NO_SCREEN_OBJ ) return;
	go->so_flags |= SO_GAUGE;
	go->so_min=min;
	go->so_max=max;
	go->so_val=val; 
	make_gauge(QSP_ARG  go);

	addHead(curr_panel->po_children,mk_node(go));
	go->so_parent = curr_panel;

	curr_panel->po_curry += GAUGE_HEIGHT + GAP_HEIGHT;
}

COMMAND_FUNC( set_panel_label )
{
	const char *s;

	s=NAMEOF("new panel label");
	label_panel(curr_panel,s);
}

COMMAND_FUNC( set_new_range )
{
	Screen_Obj *sop;
	int min,max;

	sop=PICK_SCRNOBJ("slider");
	min=HOW_MANY("min value");
	max=HOW_MANY("max value");
	if( sop == NO_SCREEN_OBJ ) return;
	new_slider_range(sop,min,max);
}


COMMAND_FUNC( set_new_pos )
{
	Screen_Obj *sop;
	int val;

	sop=PICK_SCRNOBJ("slider");
	val=HOW_MANY("value");
	if( sop == NO_SCREEN_OBJ ) return;
	new_slider_pos(sop,val);
}

COMMAND_FUNC( mk_slider )
{
	Screen_Obj *sop;
	int min,max,val;

	sop = GET_PARTS("slider");
	GET_MIN_MAX_VAL(&min,&max,&val);
	if( sop == NO_SCREEN_OBJ ) return;
	sop->so_flags |= SO_SLIDER;
	sop->so_min=min;
	sop->so_max=max;
	sop->so_val=val; 
	
	make_slider(QSP_ARG  sop);

	addHead(curr_panel->po_children,mk_node(sop));
	sop->so_parent = curr_panel;

	curr_panel->po_curry += SLIDER_HEIGHT + GAP_HEIGHT;
}

COMMAND_FUNC( mk_slider_w )
{
	Screen_Obj *sop;
	int min,max,val,width;

	sop = GET_PARTS("slider");
	GET_MIN_MAX_VAL(&min,&max,&val);
	GET_SO_WIDTH(&width);
	if( sop == NO_SCREEN_OBJ ) return;
	sop->so_flags |= SO_SLIDER;
	sop->so_min=min;
	sop->so_max=max;
	sop->so_val=val;
	sop->so_width=width;
	
	make_slider_w(QSP_ARG  sop);

	addHead(curr_panel->po_children,mk_node(sop));
	sop->so_parent = curr_panel;

	curr_panel->po_curry += SLIDER_HEIGHT + GAP_HEIGHT;
}

COMMAND_FUNC( mk_adjuster )
{
	Screen_Obj *sop;
	int min,max,val;

	sop = GET_PARTS("slider");
	GET_MIN_MAX_VAL(&min,&max,&val);
	if( sop == NO_SCREEN_OBJ ) return;
	sop->so_flags |= SO_SLIDER;
	sop->so_min=min;
	sop->so_max=max;
	sop->so_val=val; 
	
	make_adjuster(QSP_ARG  sop);


	addHead(curr_panel->po_children,mk_node(sop));
	sop->so_parent = curr_panel;

	curr_panel->po_curry += SLIDER_HEIGHT + GAP_HEIGHT;
}

COMMAND_FUNC( mk_adjuster_w )
{
	Screen_Obj *sop;
	int min,max,val,width;

	sop = GET_PARTS("slider");
	GET_MIN_MAX_VAL(&min,&max,&val);
	GET_SO_WIDTH(&width);
	if( sop == NO_SCREEN_OBJ ) return;
	sop->so_flags |= SO_SLIDER;
	sop->so_min=min;
	sop->so_max=max;
	sop->so_val=val;
	sop->so_width=width;
	
	make_adjuster_w(QSP_ARG  sop);


	addHead(curr_panel->po_children,mk_node(sop));
	sop->so_parent = curr_panel;

	curr_panel->po_curry += SLIDER_HEIGHT + GAP_HEIGHT;
}

COMMAND_FUNC( mk_message )
{
	Screen_Obj *mp;

	mp=GET_PARTS("message");
	if( mp == NO_SCREEN_OBJ ) return;
	mp->so_flags |= SO_MESSAGE;
	/* get_parts gets the name and the action text, but for a message
	 * we have no action, just content_text
	 */
	mp->so_content_text = mp->so_action_text;
	mp->so_action_text = NULL;
	make_message(QSP_ARG  mp);

	addHead(curr_panel->po_children,mk_node(mp));
	mp->so_parent = curr_panel;

	curr_panel->po_curry += MESSAGE_HEIGHT + GAP_HEIGHT;
}

	
COMMAND_FUNC( do_show )
{
	Panel_Obj *po;

	po=PICK_PANEL("");
	if( po != NO_PANEL_OBJ ) show_panel(po);
}

void do_genwin_panel_show(QSP_ARG_DECL  const char *s)
{
	Panel_Obj *po;

	po=GET_PANEL_OBJ(s);
	if( po != NO_PANEL_OBJ ) show_panel(po);
	return;
}


COMMAND_FUNC( do_unshow )
{
	Panel_Obj *po;

	po=PICK_PANEL("");
	if( po != NO_PANEL_OBJ ) unshow_panel(po);
}

void do_genwin_panel_unshow(QSP_ARG_DECL  const char *s)
{
	Panel_Obj *po;

	po=GET_PANEL_OBJ(s);
	if( po != NO_PANEL_OBJ ) unshow_panel(po);
	return;
}

COMMAND_FUNC( do_set_gauge )
{
	Screen_Obj *gp;
	int n;

	gp=PICK_SCRNOBJ("guage");
	n=HOW_MANY("setting");

	if( gp == NO_SCREEN_OBJ ) return;

	set_gauge(gp,n);
}

COMMAND_FUNC( do_set_toggle )
{
	Screen_Obj *sop;
	int state;

	sop=PICK_SCRNOBJ("toggle");
	state = ASKIF("toggle set");

	if( sop == NO_SCREEN_OBJ ) return;

	new_toggle_state(sop,state);
}


#define MAX_CHOICES 16

COMMAND_FUNC( do_set_choice )
{
	Screen_Obj *sop, *bsop;
	int i,n;
	const char *s;
	const char *choices[MAX_CHOICES];
	Node *np;

	sop=PICK_SCRNOBJ("chooser");
	if( sop == NO_SCREEN_OBJ ){
		s=NAMEOF("dummy word");
		return;
	}
	/* BUG make sure sop points to a chooser... */
	/* traverse the list of children to get the names... */

	n = eltcount(sop->so_children);
	if( n <= 0 ){
		sprintf(ERROR_STRING,"Chooser %s has no children!?",sop->so_name);
		WARN(ERROR_STRING);
		s=NAMEOF("dummy word");
		return;
	}

	if( n > MAX_CHOICES ){
		sprintf(ERROR_STRING,"Chooser %s has %d choices, max is %d",
			sop->so_name,n,MAX_CHOICES);
		WARN(ERROR_STRING);
		advise("Need to recompile screen_objs.c w/ increased value of MAX_CHOICES");
		s=NAMEOF("dummy word");
		return;
	}

	n=0;
	np=sop->so_children->l_head;
	while( np != NO_NODE ){
		const char *s;
		bsop = (Screen_Obj *)np->n_data;
		s = bsop->so_name;
		/* The name is CHOOSER_NAME.CHOICE_NAME, we want to strip
		 * CHOOSER_NAME from the front, so we skip forward until we pass
		 * the first period '.'
		 */
		while( *s && *s != '.' )
			s++;
#ifdef CAUTIOUS
		if( *s != '.' ){
			sprintf(ERROR_STRING,
		"CAUTIOUS:  do_set_choice:  no period in choice object name (%s)!?",bsop->so_name);
			WARN(ERROR_STRING);
			return;
		}
#endif /* CAUTIOUS */
		
		s ++;

#ifdef CAUTIOUS
		if( *s == 0 ){
			sprintf(ERROR_STRING,
		"CAUTIOUS:  do_set_choice:  no choice name after period (choice object %s)!?",bsop->so_name);
			WARN(ERROR_STRING);
			return;
		}
#endif /* CAUTIOUS */
		
		choices[n] = s;
		n++;
		np=np->n_next;
	}

	i = WHICH_ONE("choice",n,choices);
	if( i < 0 ) return;

	/* Now scan the list again, clearing all choices except the i_th */

	n=0;
	np=sop->so_children->l_head;
	while( np != NO_NODE ){
		bsop = (Screen_Obj *)np->n_data;
		if( n == i )
			new_toggle_state(bsop,1);
		else
			new_toggle_state(bsop,0);

		n++;
		np=np->n_next;
	}
}

COMMAND_FUNC( do_set_message )
{
	Screen_Obj *mp;
	const char *s;


	mp=PICK_SCRNOBJ("message");
	s=NAMEOF("message text");
	if( mp == NO_SCREEN_OBJ ) return;

	givbuf((void *)mp->so_content_text);
	mp->so_content_text = savestr(s);
	update_message(mp);
}

COMMAND_FUNC( clear_screen )
{
	List *lp;
	Node *np;
	Screen_Obj *sop;
	Panel_Obj *po;

	po=PICK_PANEL("");
	if( po==NO_PANEL_OBJ ) return;

	lp=item_list(QSP_ARG  scrnobj_itp);
#ifdef CAUTIOUS
	if( lp==NO_LIST ){
		WARN("CAUTIOUS:  no list!?");
		return;
	}
#endif /* CAUTIOUS */
	np=lp->l_head;
	while( np != NO_NODE ){
		sop = (Screen_Obj *)np->n_data;
		if( WIDGET_PANEL(sop) == po )
			del_so(QSP_ARG  sop);
		np=np->n_next;
	}
	del_po(QSP_ARG  po);
}

static void del_so(QSP_ARG_DECL  Screen_Obj *sop)
{
	del_scrnobj(QSP_ARG  sop->so_name);
	/* BUG? are there nameless objects? */
	if( sop->so_name != NULL ) givbuf((void *)sop->so_name);
	if( sop->so_action_text != NULL ) givbuf((void *)sop->so_action_text);
	if( sop->so_selector != NULL ) givbuf((void *)sop->so_selector);
}

static void del_po(QSP_ARG_DECL  Panel_Obj *po)
{
	/* should deallocate any window system stuff first */
	free_wsys_stuff(po);

	del_panel_obj(QSP_ARG  po->po_name);
	givbuf((void *)po->po_name);
}

COMMAND_FUNC( do_pposn )
{
	Panel_Obj *po;
	int x,y;

	po=PICK_PANEL("");
	x=HOW_MANY("x position");
	y=HOW_MANY("y position");

	if( po != NO_PANEL_OBJ ){
		po->po_x=x;
		po->po_y=y;
		posn_panel(po);
	}
}

void do_genwin_panel_posn(QSP_ARG_DECL  const char *s, int x, int y)
{
	Panel_Obj *po;
	po=GET_PANEL_OBJ(s);
	if( po != NO_PANEL_OBJ ) {
		po->po_x = x;
		po->po_y = y;
		posn_panel(po);
	}
	return;
}

void do_genwin_panel_delete(QSP_ARG_DECL  const char *s)
{
	Panel_Obj *po;
	po=GET_PANEL_OBJ(s);
	if( po != NO_PANEL_OBJ ) {
		WARN("sorry, don't know how to delete panel yet");
	}
	return;

}
COMMAND_FUNC( do_delete )
{
	Panel_Obj *po;

	po=PICK_PANEL("");
	if( po != NO_PANEL_OBJ ){
		WARN("Sorry, don't know how to delete panels yet");
	}
}

COMMAND_FUNC( do_notice )
{
	const char *msg_tbl[5];

	msg_tbl[0]=NAMEOF("notice message");
	msg_tbl[1]=NAMEOF("Yes prompt");
	msg_tbl[2]=NAMEOF("Yes action");
	msg_tbl[3]=NAMEOF("No prompt");
	msg_tbl[4]=NAMEOF("No action");

	give_notice(msg_tbl);
}

List *panel_list(SINGLE_QSP_ARG_DECL)
{
	return( item_list(QSP_ARG  panel_obj_itp) );
}

COMMAND_FUNC( mk_scroller )
{
	Screen_Obj *sop;

	sop=GET_PARTS("scroller");
	if( sop == NO_SCREEN_OBJ ) return;
	sop->so_flags |= SO_SCROLLER;

	make_scroller(QSP_ARG  sop);

	addHead(curr_panel->po_children,mk_node(sop));
	sop->so_parent = curr_panel;

	curr_panel->po_curry += SCROLLER_HEIGHT + GAP_HEIGHT + 50;
}

#define MAX_STRINGS	128

int get_strings(QSP_ARG_DECL Screen_Obj *sop,const char ***sss)
{
	const char **string_arr;
	int i,n;

	n=HOW_MANY("number of items");
	/* so_action_text is set to some garbage */
	/* we'll steal so_selector to point to an array of strings */
	sop->so_selectors = (const char **)getbuf( n * sizeof(char *) );
	string_arr = sop->so_selectors;
	for(i=0;i<n;i++){
		string_arr[i]=savestr(NAMEOF("selector text") );
	}
	*sss = string_arr;
	return(n);
}

/* Called by "items" */
COMMAND_FUNC( do_set_scroller )
{
	const char **string_arr;
	Screen_Obj *sop;
	int n;

	sop = PICK_SCRNOBJ("scroller");
	if( sop==NO_SCREEN_OBJ ) return;

	n=GET_STRINGS(sop,&string_arr);
	set_scroller_list(sop,string_arr,n);
}

void mk_it_scroller(QSP_ARG_DECL  Screen_Obj *sop,Item_Type *itp)
{
	List *lp;
	Node *np;
	int i,n=0;
	const char **sp;
	const char *string_arr[MAX_STRINGS];
	Item *ip;

	lp=item_list(QSP_ARG  itp);
	if( lp == NO_LIST ) return;
	np=lp->l_head;
	while(np!=NO_NODE){
		ip=(Item *)np->n_data;
		if( n < MAX_STRINGS )
			string_arr[n]=ip->item_name;
		n++;
		np = np->n_next;
	}
	set_scroller_list(sop,string_arr,n);

	sop->so_selectors = (const char **)getbuf( n * sizeof(char *) );
	sp = sop->so_selectors;
	for(i=0;i<n;i++) sp[i]=string_arr[i];
}

#define NCHOICES	4

COMMAND_FUNC( do_item_scroller )
{
	Screen_Obj *sop;
	Item_Type *itp;

	sop = PICK_SCRNOBJ("scroller");
	itp = PICK_ITTYP("");
	if( sop==NO_SCREEN_OBJ || itp == NO_ITEM_TYPE ) return;

	mk_it_scroller(QSP_ARG  sop,itp);
}

COMMAND_FUNC( do_file_scroller )
{
	int i,n=0;
	Screen_Obj *sop;
	const char **sp, *string_arr[MAX_STRINGS];
	char word[BUF_LEN];
	FILE *fp;

	sop = PICK_SCRNOBJ("scroller");
	fp = TRY_OPEN( NAMEOF("item file"), "r" );
	if( !fp ) return;

	while( fscanf(fp,"%s",word) == 1 ){
		if( n < MAX_STRINGS )
			string_arr[n]=savestr(word);
		n++;
#ifdef DEBUG
if( debug ){
sprintf(ERROR_STRING,"choice %d %s\n",n,string_arr[n-1]);
WARN(ERROR_STRING);
}
#endif
	}
	fclose(fp);
	set_scroller_list(sop,string_arr,n);

	sop->so_selectors = (const char **)getbuf( n * sizeof(char *) );
	sp = sop->so_selectors;
	for(i=0;i<n;i++) sp[i]=string_arr[i];
}

COMMAND_FUNC( do_chooser )
{
	Screen_Obj *sop;
	int n;
	const char **stringlist;

	sop=GET_PARTS("chooser");
	if( sop == NO_SCREEN_OBJ ) return;
	sop->so_flags |= SO_CHOOSER;
	sop->so_children = new_list();

	n=GET_STRINGS(sop,&stringlist);

	make_chooser(QSP_ARG  sop,n,stringlist);

	addHead(curr_panel->po_children,mk_node(sop));
	sop->so_parent = curr_panel;

	curr_panel->po_curry += CHOOSER_HEIGHT + GAP_HEIGHT +
		CHOOSER_ITEM_HEIGHT*n;
}

Node *first_panel_node(SINGLE_QSP_ARG_DECL)
{
	List *lp;

	lp=item_list(QSP_ARG  panel_obj_itp);
	if( lp==NO_LIST ) return(NO_NODE);
	else return(lp->l_head);
}

Screen_Obj *find_object_at(Panel_Obj *po,int x,int y)
{
	Node *np;
	Screen_Obj *sop;

	np=po->po_children->l_head;
	while(np!=NO_NODE){
		sop=(Screen_Obj *)np->n_data;
		if(	   x >= sop->so_x
			&& x <  (int) (sop->so_x+sop->so_dx)
			&& y >= sop->so_y
			&& y < (int) (sop->so_y+sop->so_dy) )
			return(sop);
		np=np->n_next;
	}
	return(NO_SCREEN_OBJ);
}

static Genwin_Functions gwfp={
	do_genwin_panel_posn,
	do_genwin_panel_show,
	do_genwin_panel_unshow,
	do_genwin_panel_delete
};		

static double get_panel_size(Item *ip,int index)
{
	double d;
	Panel_Obj *po;

	po = (Panel_Obj *)ip;

	switch(index){
		case 1:	d = po->po_width; break;
		case 2:	d = po->po_height; break;
		default: d=1.0; break;
	}
	return(d);
}

static Size_Functions panel_sf={
	/*(double (*)(Item *,int))*/		get_panel_size,
	/*(Item * (*)(Item *,index_t))*/	NULL,
	/*(Item * (*)(Item *,index_t))*/	NULL,
	/*(double (*)(Item *))*/		NULL
};

void so_init(QSP_ARG_DECL  int argc,char *argv[])
{
	static int so_inited=0;

	if( so_inited ) return;
	window_sys_init(SINGLE_QSP_ARG);

	motif_init(argv[0]);

	if(panel_obj_itp == NO_ITEM_TYPE) panel_obj_init(SINGLE_QSP_ARG);	
	
	/* support for genwin */
	add_genwin(QSP_ARG  panel_obj_itp, &gwfp, NULL);
	add_sizable(QSP_ARG  panel_obj_itp,&panel_sf, NULL );

	auto_version(QSP_ARG  "GUI","VersionId_gui");

	so_inited=1;
}

double panel_exists(QSP_ARG_DECL  const char *name)
{
	Panel_Obj *po;
	po = panel_obj_of(QSP_ARG  name);

	if( po == NO_PANEL_OBJ ) return(0);
	return(1.0);
}

#endif /* HAVE_MOTIF */
