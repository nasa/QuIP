/* This file is used to control the Cocoa menu bar from scripts */

#include "quip_config.h"
#include "quip_prot.h"
#include "menu_bar.h"
#include "gui_cmds.h"

//IOS_ITEM_INTERFACE_DECLARATIONS(Menu_Bar_Menu,menu_bar_menu)
//IOS_ITEM_INTERFACE_DECLARATIONS(Menu_Bar_Item,menu_bar_item)

static IOS_Item_Type *menu_bar_menu_itp=NULL;
IOS_ITEM_INIT_FUNC(Menu_Bar_Menu,menu_bar_menu,0)
IOS_ITEM_NEW_FUNC(Menu_Bar_Menu,menu_bar_menu)
IOS_ITEM_CHECK_FUNC(Menu_Bar_Menu,menu_bar_menu)
IOS_ITEM_PICK_FUNC(Menu_Bar_Menu,menu_bar_menu)

static IOS_Item_Type *menu_bar_item_itp=NULL;
IOS_ITEM_INIT_FUNC(Menu_Bar_Item,menu_bar_item,0)
IOS_ITEM_NEW_FUNC(Menu_Bar_Item,menu_bar_item)
IOS_ITEM_CHECK_FUNC(Menu_Bar_Item,menu_bar_item)
IOS_ITEM_PICK_FUNC(Menu_Bar_Item,menu_bar_item)
IOS_ITEM_LIST_FUNC(Menu_Bar_Item,menu_bar_item)

@implementation Menu_Bar_Menu
@synthesize menu;
@synthesize bar_item;

+(void) initClass
{
	menu_bar_menu_itp = [[IOS_Item_Type alloc] initWithName:@"Menu_Bar_Menu"];
}

@end

static Menu_Bar_Item *find_menu_bar_item(NSObject *id)
{
	IOS_List *lp;

	//lp = ios_item_list(DEFAULT_QSP_ARG  menu_bar_item_itp);
	lp = all_ios_items(DEFAULT_QSP_ARG  menu_bar_item_itp);
	IOS_Node *np;
	np = IOS_LIST_HEAD(lp);
	while( np != NULL ){
		Menu_Bar_Item *mbi_p;
		mbi_p = IOS_NODE_DATA(np);
		if( mbi_p.menu_item == id )
			return(mbi_p);
		np = IOS_NODE_NEXT(np);
	}
	return NULL;
}

@implementation Menu_Bar_Item
@synthesize menu_item;

+(void) initClass
{
	menu_bar_item_itp = [[IOS_Item_Type alloc] initWithName:@"Menu_Bar_Item"];
}

-(void) genericMenuAction:(id) sender
{
	Menu_Bar_Item *mbi_p;

	mbi_p = find_menu_bar_item(sender);
	if( mbi_p != NULL ){
		chew_text(DEFAULT_QSP_ARG  mbi_p.action, "(menu action)");
	}
}


@end

static Menu_Bar_Menu *curr_menu=NULL;
#define CHECK_CURR_MENU				\
						\
	if( curr_menu == NULL ){		\
		WARN("No currrent menu!?");	\
		return;				\
	}

#define ADD_CMD(s,f,h)	ADD_COMMAND(populate_menu_menu,s,f,h)

static COMMAND_FUNC( do_new_menu_item )
{
	Menu_Bar_Item *mbi_p;
	const char *name;
	const char *action;

	name=NAMEOF("name for menu item");
	action=NAMEOF("action string");
	mbi_p = new_menu_bar_item(QSP_ARG  name);
	if( mbi_p == NULL ) return;

	mbi_p.action = savestr(action);

	CHECK_CURR_MENU

	NSString *s=STRINGOBJ(name);

	// BUG we might not want to add here?
	mbi_p.menu_item = [curr_menu.menu
		addItemWithTitle:NSLocalizedString(s, nil)
		action:@selector(genericMenuAction:)
		keyEquivalent:@""];

	mbi_p.menu_item.target = mbi_p;
}

static COMMAND_FUNC( do_add_separator )
{
	CHECK_CURR_MENU

	[curr_menu.menu addItem:[NSMenuItem separatorItem]];
}

MENU_BEGIN(populate_menu)
ADD_CMD( item,		do_new_menu_item,	create a menu item )
ADD_CMD( separator,	do_add_separator,	add a separator to the menu )
//ADD_CMD( add,		do_add_menu_item,	add item to current menu )
//ADD_CMD( remove,	do_rem_menu_item,	remove item from current menu )
//ADD_CMD( enable,	do_enable_menu_item,	enable menu item )
//ADD_CMD( disable,	do_disable_menu_item,	disable menu item )
MENU_END(populate_menu)

static COMMAND_FUNC(do_populate_menu)
{
	curr_menu = PICK_MENU_BAR_MENU("menu name");

	// First, pop any old context
	int n;
	n = menu_bar_item_itp.contextStack.depth;
	if( n > 1 ){
		IOS_Item_Context *icp;
		icp = pop_ios_item_context(QSP_ARG  menu_bar_item_itp);
	}
	if( curr_menu != NULL ){
		push_ios_item_context(QSP_ARG  menu_bar_item_itp, 
						curr_menu.item_icp );
	}
	PUSH_MENU(populate_menu);
}

#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(menu_bar_menu,s,f,h)

static COMMAND_FUNC(do_new_menu)
{
	Menu_Bar_Menu *mbm_p;
	const char *name;

	name=NAMEOF("name for menu");

	mbm_p = new_menu_bar_menu(QSP_ARG  name);
	if( mbm_p == NULL ) return;

	NSString *s=STRINGOBJ(name);
	mbm_p.menu = [[NSMenu alloc] initWithTitle:NSLocalizedString(s,s) ];

	mbm_p.bar_item = [ [NSApp mainMenu]
		addItemWithTitle:s
		action:NULL
		keyEquivalent:@""];

	// make sure this is good...
	if( menu_bar_item_itp == NULL )
		init_menu_bar_items(SINGLE_QSP_ARG);

	mbm_p.item_icp =
		create_ios_item_context(QSP_ARG  menu_bar_item_itp, name);
}

static COMMAND_FUNC(do_add_menu)
{
	Menu_Bar_Menu *mbm_p;

	mbm_p = PICK_MENU_BAR_MENU("menu name");
	if( mbm_p == NULL ) return;

	[[NSApp mainMenu] setSubmenu:mbm_p.menu forItem:mbm_p.bar_item];
}

MENU_BEGIN(menu_bar)
ADD_CMD( new_menu,	do_new_menu,		create a new menu )
ADD_CMD( populate,	do_populate_menu,	add items to a menu )
ADD_CMD( add_to_bar,	do_add_menu,		add menu to the menu bar )
//ADD_CMD( remove_from_bar,	do_rem_menu,	remove menu from the menu bar )
//ADD_CMD( list,		do_list_menus,		list all menus )
//ADD_CMD( info,		do_menu_info,		print info about a menu )
MENU_END(menu_bar)

COMMAND_FUNC(do_menu_bar)
{
	PUSH_MENU(menu_bar)
}

