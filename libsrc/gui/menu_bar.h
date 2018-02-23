#include "ios_item.h"

#include <AppKit/NSMenu.h>

@interface Menu_Bar_Menu : IOS_Item
@property (retain) NSMenu *		menu;
@property (retain) NSMenuItem *		bar_item;
@property (retain) IOS_Item_Context *	item_icp;
+(void) initClass;
@end

@interface Menu_Bar_Item : IOS_Item
@property (retain) NSMenuItem *	menu_item;
@property const char *		action;
+(void) initClass;
@end

//IOS_ITEM_INTERFACE_PROTOTYPES(Menu_Bar_Menu,menu_bar_menu)
//IOS_ITEM_INTERFACE_PROTOTYPES(Menu_Bar_Item,menu_bar_item)

IOS_ITEM_INIT_PROT(Menu_Bar_Menu,menu_bar_menu)
IOS_ITEM_NEW_PROT(Menu_Bar_Menu,menu_bar_menu)
IOS_ITEM_CHECK_PROT(Menu_Bar_Menu,menu_bar_menu)
IOS_ITEM_PICK_PROT(Menu_Bar_Menu,menu_bar_menu)
#define pick_menu_bar_menu(s)	_pick_menu_bar_menu(QSP_ARG  s)
#define menu_bar_menu_of(s)	_menu_bar_menu_of(QSP_ARG  s)
#define new_menu_bar_menu(s)	_new_menu_bar_menu(QSP_ARG  s)
#define init_menu_bar_menus()	_init_menu_bar_menus(SINGLE_QSP_ARG)

IOS_ITEM_INIT_PROT(Menu_Bar_Item,menu_bar_item)
IOS_ITEM_NEW_PROT(Menu_Bar_Item,menu_bar_item)
IOS_ITEM_CHECK_PROT(Menu_Bar_Item,menu_bar_item)
IOS_ITEM_PICK_PROT(Menu_Bar_Item,menu_bar_item)
IOS_ITEM_LIST_PROT(Menu_Bar_Item,menu_bar_item)
#define pick_menu_bar_item(s)	_pick_menu_bar_item(QSP_ARG  s)
#define list_menu_bar_items()	_list_menu_bar_items(SINGLE_QSP_ARG)
#define menu_bar_item_of(s)	_menu_bar_item_of(QSP_ARG  s)
#define new_menu_bar_item(s)	_new_menu_bar_item(QSP_ARG  s)
#define init_menu_bar_items()	_init_menu_bar_items(SINGLE_QSP_ARG)

