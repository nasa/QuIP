
#ifndef _NAV_PANEL_H_
#define _NAV_PANEL_H_

typedef enum {
	TABLE_ITEM_TYPE_NAV,
	TABLE_ITEM_TYPE_PLAIN
} Table_Item_Type;

#ifdef BUILD_FOR_OBJC
/* This file is only used in ios - for the present */

//#include "quipAppDelegate.h"
//#include "quipViewController.h"
//#include "quipView.h"
#include "quipTableViewController.h"

#include "ios_item.h"
#include "panel_obj.h"

//#include "gen_win.h"

@interface Nav_Panel : IOS_Item
// the group list might be better as NSMutableArray?
@property (retain) IOS_List *		groups;		// list of groups
@property (retain) IOS_Item_Context *	grp_icp;	// context for group names
//@property (retain) IOS_Item_Context *	itm_icp;	// context for item names
@property (retain) quipTableViewController *	qnc;		// do we need a per-panel controller?
+(void) initClass;
-(void) setDoneAction:(const char *)action;
@end



IOS_ITEM_INIT_PROT(Nav_Panel,nav_panel)
IOS_ITEM_NEW_PROT(Nav_Panel,nav_panel)
IOS_ITEM_PICK_PROT(Nav_Panel,nav_panel)
IOS_ITEM_GET_PROT(Nav_Panel,nav_panel)
IOS_ITEM_DEL_PROT(Nav_Panel,nav_panel)
IOS_ITEM_CHECK_PROT(Nav_Panel,nav_panel)
IOS_ITEM_ENUM_PROT(Nav_Panel,nav_panel)
IOS_ITEM_LIST_PROT(Nav_Panel,nav_panel)

#define GET_NAV_PANEL(s)	get_nav_panel(QSP_ARG  s)
#define NAV_PANEL_OF(s)		nav_panel_of(QSP_ARG  s)
#define PICK_NAV_PANEL(pmpt)	pick_nav_panel(QSP_ARG  pmpt)

#define NAVP_GRP_CONTEXT(nav_p)		(nav_p).grp_icp
#define NAVP_ITM_CONTEXT(nav_p)		(nav_p).itm_icp


extern void init_nav_panel(Nav_Panel *nav_p);


@interface Nav_Group : IOS_Item
// the group list might be better as NSMutableArray?
@property (nonatomic, retain) NSMutableArray * items;
@property (retain) IOS_Item_Context *	itm_icp;	// context for item names
@property (retain) Nav_Panel *		ng_panel;	// BUG?  circular ref?
+(void) initClass;
-(void) add_nav_item:(Nav_Item *)ip;
-(void) del_nav_item:(Nav_Item *)ip;
-(void) reload_group;

@end


IOS_ITEM_INIT_PROT(Nav_Group,nav_group)
IOS_ITEM_NEW_PROT(Nav_Group,nav_group)
IOS_ITEM_PICK_PROT(Nav_Group,nav_group)
IOS_ITEM_GET_PROT(Nav_Group,nav_group)
IOS_ITEM_DEL_PROT(Nav_Group,nav_group)
IOS_ITEM_CHECK_PROT(Nav_Group,nav_group)
IOS_ITEM_ENUM_PROT(Nav_Group,nav_group)
IOS_ITEM_LIST_PROT(Nav_Group,nav_group)

// ios macros
#define NAVGRP_ITEM_CONTEXT(nav_g)	(nav_g).itm_icp
#define NAVGRP_NAME(nav_g)		(nav_g).name.UTF8String

@interface Nav_Item : IOS_Item
// Do we need a weak reference up?
@property (retain) NSString *		explanation;
@property const char *			action;
@property Table_Item_Type		type;
#ifdef BUILD_FOR_IOS
@property (retain) UITableViewCell *	cell;
#endif // BUILD_FOR_IOS
@property (retain) Nav_Group *		group;	// retention cycle?  BUG?  weak ref?
+(void) initClass;
@end

#define SET_NAVITM_ACTION(ni_p,s)	(ni_p).action = s
#define SET_NAVITM_TYPE(ni_p,t)		(ni_p).type = t

IOS_ITEM_INIT_PROT(Nav_Item,nav_item)
IOS_ITEM_NEW_PROT(Nav_Item,nav_item)
IOS_ITEM_PICK_PROT(Nav_Item,nav_item)
IOS_ITEM_GET_PROT(Nav_Item,nav_item)
IOS_ITEM_DEL_PROT(Nav_Item,nav_item)
IOS_ITEM_CHECK_PROT(Nav_Item,nav_item)
IOS_ITEM_ENUM_PROT(Nav_Item,nav_item)
IOS_ITEM_LIST_PROT(Nav_Item,nav_item)

#define GET_NAV_ITEM(s)	get_nav_item(QSP_ARG  s)
#define PICK_NAV_ITEM(pmpt)	pick_nav_item(QSP_ARG  pmpt)

extern void set_supported_orientations(UIInterfaceOrientationMask m);
extern void set_autorotation_allowed(BOOL yesno);


#else // ! BUILD_FOR_OBJC





typedef struct nav_panel {
	Gen_Win			np_genwin;
	Panel_Obj *		np_po;
	IOS_Item_Context *	np_grp_icp;	// context for group names
	IOS_Item_Context *	np_itm_icp;	// context for item names
} Nav_Panel;

#define np_name		np_genwin.gw_name


ITEM_INTERFACE_PROTOTYPES(Nav_Panel,nav_panel)


#define NAVP_GW(nav_p)			(&((nav_p)->np_genwin))

#define NAVP_NAME(nav_p)		(nav_p)->np_genwin.gw_name
#define NAVP_GRP_CONTEXT(nav_p)		(nav_p)->np_grp_icp
#define NAVP_ITM_CONTEXT(nav_p)		(nav_p)->np_itm_icp
#define SET_NAVP_GRP_CONTEXT(nav_p,icp)	(nav_p)->np_grp_icp = icp
#define SET_NAVP_ITM_CONTEXT(nav_p,icp)	(nav_p)->np_itm_icp = icp


typedef struct nav_group {
	Item		ng_item;
	Item_Context *	ng_itm_icp;
	Screen_Obj *	ng_sop;
	Panel_Obj *	ng_panel_p;
} Nav_Group;

// UNIX macros

#define NAVGRP_NAME(nav_g)		(nav_g)->ng_item.item_name

#define NAVGRP_ITEM_CONTEXT(ng_p)		(ng_p)->ng_itm_icp
#define SET_NAVGRP_ITEM_CONTEXT(ng_p,icp)	(ng_p)->ng_itm_icp = icp

#define NAVGRP_SCRNOBJ(ng_p)			(ng_p)->ng_sop
#define SET_NAVGRP_SCRNOBJ(ng_p,sop)		(ng_p)->ng_sop = sop
#define NAVGRP_PANEL(ng_p)			(ng_p)->ng_panel_p
#define SET_NAVGRP_PANEL(ng_p,pnl_p)		(ng_p)->ng_panel_p = pnl_p

ITEM_INTERFACE_PROTOTYPES(Nav_Group,nav_group)

typedef struct nav_item {
	Item		ni_item;
	const char *	ni_explanation;
	const char *	ni_action;
	Table_Item_Type	ni_type;

	Screen_Obj *	ni_sop;
	Panel_Obj *	ni_panel_p;
} Nav_Item;

ITEM_INTERFACE_PROTOTYPES(Nav_Item,nav_item)

#define NAVITM_NAME(ni_p)		(ni_p)->ni_item.item_name

#define NAVITM_EXPLANATION(ni_p)	(ni_p)->ni_explanation
#define SET_NAVITM_EXPLANATION(ni_p,s)	(ni_p)->ni_explanation = s
#define NAVITM_ACTION(ni_p)		(ni_p)->ni_action
#define SET_NAVITM_ACTION(ni_p,s)	(ni_p)->ni_action = s
#define NAVITM_TYPE(ni_p)		(ni_p)->ni_type
#define SET_NAVITM_TYPE(ni_p,t)		(ni_p)->ni_type = t

#define NAVITM_SCRNOBJ(ni_p)		(ni_p)->ni_sop
#define NAVITM_PANEL(ni_p)		(ni_p)->ni_panel_p
#define SET_NAVITM_SCRNOBJ(ni_p,sop)	(ni_p)->ni_sop = sop
#define SET_NAVITM_PANEL(ni_p,pnl_p)	(ni_p)->ni_panel_p = pnl_p

#define NAVP_PANEL(np_p)		(np_p)->np_po
#define SET_NAVP_PANEL(np_p,v)		(np_p)->np_po = v



#endif // ! BUILD_FOR_OBJC

#define pick_nav_item(s)	_pick_nav_item(QSP_ARG  s)
#define del_nav_item(s)		_del_nav_item(QSP_ARG  s)

#define init_nav_panels()	_init_nav_panels(SINGLE_QSP_ARG)
#define pick_nav_panel(s)	_pick_nav_panel(QSP_ARG  s)
#define get_nav_panel(s)	_get_nav_panel(QSP_ARG  s)
#define new_nav_panel(s)	_new_nav_panel(QSP_ARG  s)

#define init_nav_groups()	_init_nav_groups(SINGLE_QSP_ARG)
#define pick_nav_group(pmpt)	_pick_nav_group(QSP_ARG  pmpt)
#define get_nav_group(s)	_get_nav_group(QSP_ARG  s)
#define del_nav_group(s)	_del_nav_group(QSP_ARG  s)
#define new_nav_group(s)	_new_nav_group(QSP_ARG  s)

#define init_nav_items()	_init_nav_items(SINGLE_QSP_ARG)
#define new_nav_item(s)		_new_nav_item(QSP_ARG  s)

// These prototypes are the same on either system...

extern void push_navgrp_context(QSP_ARG_DECL  IOS_Item_Context *icp);
extern void push_navitm_context(QSP_ARG_DECL  IOS_Item_Context *icp);
extern IOS_Item_Context * pop_navgrp_context(SINGLE_QSP_ARG_DECL);
extern IOS_Item_Context * pop_navitm_context(SINGLE_QSP_ARG_DECL);
extern IOS_Item_Context * create_navgrp_context(QSP_ARG_DECL const char *name);
extern IOS_Item_Context * create_navitm_context(QSP_ARG_DECL const char *name);
extern Nav_Group *create_nav_group(QSP_ARG_DECL  Nav_Panel *nav_p, const char *name);
extern Nav_Panel *create_nav_panel(QSP_ARG_DECL  const char *name);
extern void remove_nav_item(QSP_ARG_DECL  Nav_Item *nav_i);
extern void remove_nav_group(QSP_ARG_DECL  Nav_Group *nav_g);
extern void hide_nav_bar(QSP_ARG_DECL  int hide);
extern void hide_back_button(QSP_ARG_DECL  Panel_Obj *po, int hide);

extern void posn_navp(Nav_Panel *np_p);
extern void show_nav(QSP_ARG_DECL  Nav_Panel *np_p);
extern void unshow_nav(QSP_ARG_DECL  Nav_Panel *np_p);
extern void add_to_navp(Nav_Panel *np_p, Screen_Obj *sop);

#endif /* ! _NAV_PANEL_H_ */

