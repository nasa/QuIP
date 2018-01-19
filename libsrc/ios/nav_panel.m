
#include "quip_config.h"

#import "quip_prot.h"
#import "quipView.h"
#include "nav_panel.h"
#include "ios_gui.h"

static IOS_Item_Type *nav_panel_itp=NULL;

@implementation Nav_Panel

@synthesize groups;
@synthesize grp_icp;
//@synthesize itm_icp;
@synthesize qnc;

+(void) initClass
{
	nav_panel_itp=[[IOS_Item_Type alloc] initWithName: STRINGOBJ("Nav_Panel") ];
}

-(void) setDoneAction:(const char *)action
{
	[self.qnc qtvcDoneButtonPressed];
}

@end

static IOS_Item_Type *nav_group_itp=NULL;

@implementation Nav_Group

@synthesize items;
//@synthesize icp;
@synthesize ng_panel;

+(void) initClass
{
	nav_group_itp=[[IOS_Item_Type alloc] initWithName: STRINGOBJ("Nav_Group") ];
}

-(void) add_nav_item: (Nav_Item *) nip;
{
	if( items == NULL )
		//items = [[NSMutableArray alloc] init];
		items = [NSMutableArray array];

	[items addObject:nip];
	
	nip.group = self;	// BUG? retention cycle?

	// We call reloadData here so that new entries
	// will display if they are added after the panel
	// has already been shown the first time.
	// That property is used to allow an admin mode
	// to show more commands.  At present, we don't
	// have a way to back those out...
#ifdef BUILD_FOR_IOS
	[ng_panel.qnc.tableView reloadData];
#endif // BUILD_FOR_IOS
}

-(void) del_nav_item:  (Nav_Item *) nip;
{
#ifdef CAUTIOUS
	if( items == NULL ){
		NWARN("CAUTIOUS:  del_nav_item:  nav_group item list is null!?");
		return;
	}
#endif /* CAUTIOUS */

	[items removeObject:nip];
#ifdef BUILD_FOR_IOS
	[ng_panel.qnc.tableView reloadData];
#endif // BUILD_FOR_IOS
}

-(void) reload_group
{
#ifdef BUILD_FOR_IOS
    [ng_panel.qnc.tableView reloadData];
#endif // BUILD_FOR_IOS
}

@end		// Nav_Group

static IOS_Item_Type *nav_item_itp=NULL;

@implementation Nav_Item

@synthesize action;
@synthesize explanation;
@synthesize type;
#ifdef BUILD_FOR_IOS
@synthesize cell;
#endif // BUILD_FOR_IOS
@synthesize group;

+(void) initClass
{
	nav_item_itp=[[IOS_Item_Type alloc] initWithName: STRINGOBJ("Nav_Item") ];
}

@end



IOS_ITEM_INIT_FUNC(Nav_Panel,nav_panel,0)
IOS_ITEM_NEW_FUNC(Nav_Panel,nav_panel)
IOS_ITEM_CHECK_FUNC(Nav_Panel,nav_panel)
IOS_ITEM_PICK_FUNC(Nav_Panel,nav_panel)
IOS_ITEM_ENUM_FUNC(Nav_Panel,nav_panel)

IOS_ITEM_INIT_FUNC(Nav_Group,nav_group,0)
IOS_ITEM_NEW_FUNC(Nav_Group,nav_group)
IOS_ITEM_CHECK_FUNC(Nav_Group,nav_group)
IOS_ITEM_PICK_FUNC(Nav_Group,nav_group)
IOS_ITEM_DEL_FUNC(Nav_Group,nav_group)

IOS_ITEM_INIT_FUNC(Nav_Item,nav_item,0)
IOS_ITEM_NEW_FUNC(Nav_Item,nav_item)
IOS_ITEM_PICK_FUNC(Nav_Item,nav_item)
IOS_ITEM_CHECK_FUNC(Nav_Item,nav_item)
IOS_ITEM_DEL_FUNC(Nav_Item,nav_item)

IOS_Item_Context *_pop_navgrp_context(SINGLE_QSP_ARG_DECL)
{
	IOS_Item_Context *icp;
	icp = pop_ios_item_context(QSP_ARG  nav_group_itp );
	return icp;
}

// every panel has its own namespace of groups

void push_navgrp_context(QSP_ARG_DECL  IOS_Item_Context *icp)
{
	push_ios_item_context(QSP_ARG  nav_group_itp, icp );
}

IOS_Item_Context *create_navgrp_context(QSP_ARG_DECL  const char *name)
{
	if( nav_group_itp == NULL )
		init_nav_groups();
	
	return create_ios_item_context(QSP_ARG  nav_group_itp, name );
}

IOS_Item_Context *_pop_navitm_context(SINGLE_QSP_ARG_DECL)
{
	IOS_Item_Context *icp;

	icp = pop_ios_item_context(QSP_ARG  nav_item_itp );
	return icp;
}

// every group has its own namespace of items

void push_navitm_context(QSP_ARG_DECL  IOS_Item_Context *icp)
{
	push_ios_item_context(QSP_ARG  nav_item_itp, icp );
}


IOS_Item_Context *create_navitm_context(QSP_ARG_DECL  const char *name)
{
	if( nav_item_itp == NULL )
		init_nav_items();
	
	// the context might already exist, if it's not destroyed
	// when we delete a group and then recreate!?
	return create_ios_item_context(QSP_ARG  nav_item_itp, name );
}

// Do we want the globalAppDelegate to the the delegate here???

void init_nav_panel(Nav_Panel *nav_p)
{
	nav_p.groups = NULL;

	quipTableViewController *c= [[quipTableViewController alloc]
		initWithSize:globalAppDelegate.dev_size
			withDelegate:globalAppDelegate
			withPanel:nav_p ];

	nav_p.qnc=c;
	// the first panel to be declared is the root

	if( first_quip_controller == NULL ){
		first_quip_controller = nav_p.qnc;
		// in this case we need to call finish_launching
		// when we are done...
	}

	// we make this be a panel also so that we can use it generally...

	Panel_Obj *po = panel_obj_of(DEFAULT_QSP_ARG  nav_p.name.UTF8String );
	// When else can this have been created???
	if( po == NULL ){
fprintf(stderr,"init_nav_panel:  calling new_panel %s...\n",nav_p.name.UTF8String);
		po = new_panel(DEFAULT_QSP_ARG  nav_p.name.UTF8String, 
			(int)globalAppDelegate.dev_size.width,
			(int)globalAppDelegate.dev_size.height );
	}
	SET_PO_CURR_Y(po,56);	// BUG - we don't know how many pixels are in the nav bar,
				// it probably depends on screen resolution!?
	// We have to reset the view controller
	// to nav_p.qnc...
#ifdef BUILD_FOR_IOS
	SET_GW_VC(PO_GW(po),nav_p.qnc);
	SET_GW_VC_TYPE(PO_GW(po),GW_VC_QTVC);
#endif // BUILD_FOR_IOS
	
}

Nav_Group *create_nav_group(QSP_ARG_DECL  Nav_Panel *nav_p, const char *name)
{
	Nav_Group *nav_g;
	char tmp_name[LLEN];

	nav_g = new_nav_group(QSP_ARG  name);

	if( nav_g == NULL ) return nav_g;

	if( nav_item_itp == NULL )
		[Nav_Item initClass];
	
	sprintf(tmp_name,"%s.%s",nav_p.name.UTF8String,name);
	nav_g.itm_icp = create_navitm_context(QSP_ARG  tmp_name);
	nav_g.ng_panel = nav_p;

	if( nav_p.groups == NULL )
		nav_p.groups = new_ios_list();

	IOS_Node *np;
	np = mk_ios_node(nav_g);

	ios_addTail(nav_p.groups,np);

	return nav_g;
}

Nav_Panel *create_nav_panel( QSP_ARG_DECL  const char *name)
{
	Nav_Panel *nav_p;
	nav_p = new_nav_panel(QSP_ARG  name);
	if( nav_p == NULL ) return nav_p;

	// Initialize the structure
	init_nav_panel(nav_p);

	if( nav_item_itp == NULL )
		[Nav_Item initClass];
	// this context is now part of the group...
    //nav_p.itm_icp = create_ios_item_context(QSP_ARG  nav_item_itp, name);

	if( nav_group_itp == NULL )
		[Nav_Group initClass];
	nav_p.grp_icp = create_ios_item_context(QSP_ARG  nav_group_itp, name);
	
	return nav_p;
}

void remove_nav_item(QSP_ARG_DECL  Nav_Item *nav_i)
{
	// remove from associated group
	[ nav_i.group del_nav_item:nav_i ];

	del_nav_item(QSP_ARG  nav_i);
}

void remove_nav_group( QSP_ARG_DECL  Nav_Group *nav_g )
{
	Nav_Item *nav_i;
	Nav_Panel *nav_p;
	IOS_Node *np;
//    IOS_Item_Context *icp;

	// first remove all the items
	// Do we know that the context has been pushed???

//	// this line prevents the error crash, but doesn't seem to be needed???
//	push_navitm_context(QSP_ARG NAVGRP_ITEM_CONTEXT(nav_g) );

	while( nav_g.items.count > 0 ){
		nav_i = [nav_g.items objectAtIndex:0];
		remove_nav_item(QSP_ARG  nav_i);
	}
	nav_g.items = NULL;

	// remove it from the panel too!
	nav_p = nav_g.ng_panel;
	np = ios_remData( nav_p.groups, nav_g );
#ifdef CAUTIOUS
	if( np == NULL ){
		WARN("CAUTIOUS:  remove_nav_group:  group not found in panel group list!?");
		return;
	}
#endif /* CAUTIOUS */
	rls_ios_node(np);

	delete_ios_item_context(QSP_ARG  nav_g.itm_icp);

	// remove from dictionary
	del_nav_group(QSP_ARG  nav_g);
#ifdef BUILD_FOR_IOS
	[nav_g.ng_panel.qnc.tableView reloadData];
#endif // BUILD_FOR_IOS
}

void hide_nav_bar(QSP_ARG_DECL  int hide)
{
#ifdef BUILD_FOR_IOS
	if( hide )
		root_view_controller.navigationBarHidden = YES;
	else
		root_view_controller.navigationBarHidden = NO;
#endif // BUILD_FOR_IOS
}

#ifdef FUBAR

// BUG the button isn't hidden the first time this is used,
// although it is deactivated!?
//
// However, the button does disappear from the previous nav bar,
// suggesting that the push_nav hasn't really taken place yet...

void hide_back_button(QSP_ARG_DECL  Panel_Obj *po, int hide)
{
#ifdef FOOBAR
	// Need to get the NavigationItem...
	UINavigationBar *navbar;
	navbar=root_view_controller.navigationBar;
	UINavigationItem *nip;
	nip=navbar.topItem;

	if( hide )
		[nip setHidesBackButton: YES  animated:YES ];
	else
		[nip setHidesBackButton: NO  animated:YES ];
#else /* ! FOOBAR */
	
	quipViewController *qvc;
	Gen_Win *gwp;
	gwp = PO_GW(po);
	
	qvc= (quipViewController *)GW_VC(gwp);

	if( hide )
		[ qvc  setQvc_flags: qvc.qvc_flags | QVC_HIDE_BACK_BUTTON ];
	else
		[ qvc  setQvc_flags: qvc.qvc_flags & ~QVC_HIDE_BACK_BUTTON ];

	// set needs display?
	//d[qvc.view setNeedsDisplay];
#endif /* ! FOOBAR */
}

#endif // FUBAR
