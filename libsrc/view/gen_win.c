#include "quip_config.h"

#include "quip_prot.h"
#include "view_prot.h"
#include "view_cmds.h"
#include "gen_win.h"
IOS_Item_Type *itp_4;
#include "panel_obj.h"
#include "viewer.h"
#include "nav_panel.h"
IOS_Item_Type *itp_5;

static IOS_Item_Class *gw_iclp=NO_IOS_ITEM_CLASS;

#ifdef BUILD_FOR_OBJC

static IOS_Item_Type *genwin_itp=NO_IOS_ITEM_TYPE;

static double get_genwin_size(QSP_ARG_DECL  IOS_Item *ip, int index)
{
	Gen_Win *gwp;

	gwp = (Gen_Win *)ip;

//#define DEFAULT_GENWIN_DEPTH 32.0		// bits, but we need bytes
#define DEFAULT_GENWIN_DEPTH 4.0

	switch(index){
		case 0:	return DEFAULT_GENWIN_DEPTH; break;
		case 1:	return GW_WIDTH(gwp); break;
		case 2:	return GW_HEIGHT(gwp); break;
		case 3: return 1.0; break;
		case 4: return 1.0; break;
#ifdef CAUTIOUS
		default:
			WARN("CAUTIOUS:  bad index passed to get_genwin_size!?");
			return 0.0;
			break;
#endif /* CAUTIOUS */
	}
}

static const char * get_genwin_prec_string(QSP_ARG_DECL  IOS_Item *ip )
{
	return "u_byte";
}

static double get_genwin_posn(QSP_ARG_DECL  IOS_Item *ip, int index)
{
	Gen_Win *gwp;

	gwp = (Gen_Win *)ip;

	WARN("get_genwin_posn:  NOT IMPLEMENTED YET!?");
    return 0.0;
}


static IOS_Size_Functions genwin_sf={
	get_genwin_size,
	get_genwin_prec_string
};

// BUG should have position functions too!

static IOS_Position_Functions genwin_pf={
	get_genwin_posn,
};

@implementation Gen_Win

@synthesize vp;
@synthesize po;
@synthesize event_tbl;
@synthesize cmap;

#ifdef BUILD_FOR_IOS
@synthesize vc;
@synthesize vc_type;
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
@synthesize wc;
#endif // BUILD_FOR_MACOS

@synthesize qad;
@synthesize flags;
@synthesize x;
@synthesize y;
@synthesize width;
@synthesize height;
@synthesize icp;

+(void)    initClass
{
	genwin_itp = new_ios_item_type(DEFAULT_QSP_ARG  "Gen_Win");

	// final NULL arg means use default lookup function
	add_ios_sizable(DEFAULT_QSP_ARG  genwin_itp, &genwin_sf, NULL );
	add_ios_positionable(DEFAULT_QSP_ARG  genwin_itp, &genwin_pf, NULL );
}

@end

IOS_ITEM_INIT_FUNC(Gen_Win,genwin)
IOS_ITEM_NEW_FUNC(Gen_Win,genwin)
IOS_ITEM_CHECK_FUNC(Gen_Win,genwin)
IOS_ITEM_GET_FUNC(Gen_Win,genwin)
IOS_ITEM_PICK_FUNC(Gen_Win,genwin)
IOS_ITEM_LIST_FUNC(Gen_Win,genwin)
IOS_ITEM_ENUM_FUNC(Gen_Win,genwin)

Gen_Win *find_genwin(QSP_ARG_DECL  const char *name)
{
	return (Gen_Win *) get_genwin(QSP_ARG  name);
}

/* Make a new window...
 *
 * Where does the window actually get created?
 */

Gen_Win *make_genwin(QSP_ARG_DECL  const char *name,int width,int height)
{
	Gen_Win *gwp;
	CGSize s;

fprintf(stderr,"make_genwin %s BEGIN\n",name);
	gwp = new_genwin(QSP_ARG  name);

	s.width = width;
	s.height = height;
//#ifdef BUILD_FOR_IOS
	//initWithSize:[[UIScreen mainScreen] bounds].size
	quipViewController *qvc=[[quipViewController alloc]
		initWithSize:s
		withDelegate:globalAppDelegate];
	SET_QVC_GW(qvc,gwp);
#ifdef BUILD_FOR_IOS
	qvc.view.backgroundColor = [QUIP_COLOR_TYPE blackColor];
    [QVC_QV(qvc) addDefaultBG];
    SET_GW_VC_TYPE(gwp,GW_VC_QVC);
#endif // BUILD_FOR_IOS

	/* store the pointer in our named struct */
	SET_GW_VC(gwp,qvc);
//#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
	NSWindow * win;
	NSRect r;
	// BUG should allow user to set position ahead of time.
	r.origin.x = 100;
	r.origin.y = 100;
	r.size.width = width;
	r.size.height = height;	// BUG should remember user

	win = [[NSWindow alloc] initWithContentRect:r
		styleMask: NSTitledWindowMask |
				NSClosableWindowMask |
				NSMiniaturizableWindowMask |
				NSResizableWindowMask
		backing: NSBackingStoreBuffered
		defer: NO
		];
fprintf(stderr,"make_genwin:  win = 0x%lx\n",(long)win);
	//gwp.window = win;
	SET_GW_WINDOW(gwp,win);

	quipWindowController *qwc_p;
	qwc_p = [[quipWindowController alloc] initWithWindow:win];
	SET_GW_WC(gwp,qwc_p);
	win.delegate = qwc_p;

	[win makeKeyAndOrderFront:NSApp];

	// for testing...
	if( win.contentView == NULL ){
		fprintf(stderr,"NSWindow has null content view!?\n");
	}

#endif // BUILD_FOR_MACOS

	SET_GW_WIDTH(gwp,width);
	SET_GW_HEIGHT(gwp,height);

	// and reciprocate...
	// BUG - shouldn't one of these be a weak reference???
	// As long as we don't delete, we probably don't
	// have to worry about memory leaks...

	// This might have been called either for viewer creation, or
	// panel creation...  make sure the other one is created also.
	Panel_Obj *po = panel_obj_of(QSP_ARG  GW_NAME(gwp));
	if( po == NO_PANEL_OBJ ){
		/* po = */ new_panel(QSP_ARG  GW_NAME(gwp), GW_WIDTH(gwp), GW_HEIGHT(gwp) );
	}
	Viewer *vp = vwr_of(QSP_ARG  GW_NAME(gwp));
	if( vp == NO_VIEWER ){
		/* vp = */ viewer_init(QSP_ARG  GW_NAME(gwp), GW_WIDTH(gwp), GW_HEIGHT(gwp), VIEW_BUTTON_ARENA);
	}

	return gwp;
}

Gen_Win *find_genwin_for_vc(QUIP_VIEW_CONTROLLER_TYPE *vc)
{
	IOS_List *lp;
	IOS_Node *np;

	lp = genwin_list();
	if( lp == NO_IOS_LIST ) return NULL;
	np = IOS_LIST_HEAD(lp);
	while( np != NO_IOS_NODE ){
		Gen_Win *gwp;
		gwp = (Gen_Win *) IOS_NODE_DATA(np);
#ifdef BUILD_FOR_IOS
		if( GW_VC(gwp) == vc ) return gwp;
#endif // BUILD_FOR_IOS
#ifdef BUILD_FOR_MACOS
		if( GW_WC(gwp) == vc ) return gwp;
#endif // BUILD_FOR_MACOS
		np = IOS_NODE_NEXT(np);
	}
	return NULL;
}

#else /* ! BUILD_FOR_OBJC */


// Originally, we used an item "class" framework to handle gen_win objects:
// The class is a collection of item types that are searched sequentially.
// Now that we have put the Gen_Win object as the top of the instance structs,
// and the Gen_Win now contains a type field, we might simply have it be its own
// item_type - in parallel with the more specific item types - and give a warning
// when we have a name conflict...

#define INSURE_GENWIN					\
							\
	if( gw_iclp == NO_IOS_ITEM_CLASS )		\
		init_genwin_class(SINGLE_QSP_ARG);


static void init_genwin_class(SINGLE_QSP_ARG_DECL)
{
#ifdef CAUTIOUS
	if( gw_iclp != NO_IOS_ITEM_CLASS ){
		WARN("CAUTIOUS:  redundant call to init_genwin_class");
		return;
	}
#endif /* CAUTIOUS */
	gw_iclp = new_ios_item_class(QSP_ARG  "genwin");
}

IOS_Item_Type *itp_2;

void add_genwin(QSP_ARG_DECL  IOS_Item_Type *itp,Genwin_Functions *gwfp,IOS_Item *(*lookup)(QSP_ARG_DECL  const char *))
{
	INSURE_GENWIN

	add_items_to_ios_class(gw_iclp,itp,gwfp,lookup);
}

/* BUG - get_member() returns first occurence
 * of 'name' which restricts us to having
 * unique names for items of each item_type
 * of the class.
 */

Gen_Win *find_genwin(QSP_ARG_DECL  const char *name)
{
	INSURE_GENWIN
	return( (Gen_Win *) get_ios_member(QSP_ARG  gw_iclp,name) );
}

#endif /* ! BUILD_FOR_OBJC */

#ifdef HAVE_X11

static Viewer *genwin_viewer(QSP_ARG_DECL  Gen_Win *gwp)
{
	Viewer *vp;
	IOS_Member_Info *mip;

	if( gwp == NO_GENWIN ) return(NO_VIEWER);

	INSURE_GENWIN

	mip = get_ios_member_info(QSP_ARG  gw_iclp,GW_NAME(gwp));
	if( mip->mi_lookup != NULL ){
		vp = (Viewer *)mip->mi_lookup(QSP_ARG  GW_NAME(gwp));
	} else {
		vp = (Viewer *)item_of(QSP_ARG  mip->mi_itp,GW_NAME(gwp));
	}
	return(vp);
}
#endif /* HAVE_X11 */


#ifdef HAVE_MOTIF

static Panel_Obj *genwin_panel(QSP_ARG_DECL  Gen_Win *gwp)
{
	Panel_Obj *po;
	IOS_Member_Info *mip;

	if( gwp == NULL ) return(NO_PANEL_OBJ);

	INSURE_GENWIN

	mip = get_ios_member_info(QSP_ARG  gw_iclp,GW_NAME(gwp));
	if( mip->mi_lookup != NULL )
		po = (Panel_Obj *)mip->mi_lookup(QSP_ARG  GW_NAME(gwp));
	else
		po = (Panel_Obj *)item_of(QSP_ARG  mip->mi_itp,GW_NAME(gwp));
	return(po);
}

#endif /* HAVE_MOTIF */

Dpyable *genwin_display(QSP_ARG_DECL  Gen_Win *gwp)
{
#ifdef HAVE_X11
	Viewer *vp;

	vp = genwin_viewer(QSP_ARG  gwp);
	if( vp != NO_VIEWER ){
		return( VW_DPYABLE(vp) );
	} else {
#ifdef HAVE_MOTIF
		Panel_Obj *po;

		po = genwin_panel(QSP_ARG  gwp);
		if( po != NO_PANEL_OBJ ){
			return( PO_DPYABLE(po) );
		}
#endif /* HAVE_MOTIF */
	}
#endif /* HAVE_X11 */
	return NO_DISPLAY;
}


/* position a window */
#ifndef BUILD_FOR_MACOS
static
#endif // BUILD_FOR_MACOS
	void posn_genwin(QSP_ARG_DECL  Gen_Win *gwp,int x,int y)
{
#ifndef BUILD_FOR_OBJC
	Genwin_Functions *gwfp;
	IOS_Member_Info *mip;

	if( gwp == NULL ) return;

	INSURE_GENWIN

	mip = get_ios_member_info(QSP_ARG  gw_iclp,GW_NAME(gwp));
#ifdef CAUTIOUS
	if( mip == NO_IOS_MEMBER_INFO ){
		sprintf(ERROR_STRING,
			"CAUTIOUS:  position_genwin %s %d %d, missing member info #2",
			GW_NAME(gwp),x,y);
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	gwfp = (Genwin_Functions *) IOS_MBR_DATA(mip);


	(*gwfp->posn_func)(QSP_ARG  GW_NAME(gwp),x,y) ;
#else /* BUILD_FOR_OBJC */

#ifdef BUILD_FOR_MACOS
	NSPoint pt;
	NSRect r;

	r = [GW_WINDOW(gwp) frame];
fprintf(stderr,"old position:  %f %f\n",r.origin.x,r.origin.y);
	pt.x = x;
	pt.y = y;
	[GW_WINDOW(gwp) setFrameTopLeftPoint:pt];

#else // ! BUILD_FOR_MACOS
	advise("posn_genwin not implemented for iOS");
#endif // ! BUILD_FOR_MACOS

#endif /* ! BUILD_FOR_OBJC */
}

#ifdef FOOBAR
/* show a window */
static void show_genwin(QSP_ARG_DECL  IOS_Item *ip)
{
#ifndef BUILD_FOR_OBJC
	Genwin_Functions *gwfp;
	IOS_Member_Info *mip;

	if( ip == NO_IOS_ITEM ) return;

	INSURE_GENWIN

	mip = get_ios_member_info(QSP_ARG  gw_iclp,IOS_ITEM_NAME(ip));

#ifdef CAUTIOUS
	if( mip == NO_IOS_MEMBER_INFO ){
		sprintf(ERROR_STRING,
			"CAUTIOUS:  show_genwin %s, missing member info #2",
			IOS_ITEM_NAME(ip));
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	gwfp = (Genwin_Functions *) IOS_MBR_DATA(mip);

	(*gwfp->show_func)(QSP_ARG  IOS_ITEM_NAME(ip));
#endif /* ! BUILD_FOR_OBJC */
}

/* unshow a window */
static void unshow_genwin(QSP_ARG_DECL  IOS_Item *ip)
{
#ifndef BUILD_FOR_OBJC
	Genwin_Functions *gwfp;
	IOS_Member_Info *mip;

	if( ip == NO_IOS_ITEM ) return;

	INSURE_GENWIN

	mip = get_ios_member_info(QSP_ARG  gw_iclp,IOS_ITEM_NAME(ip));

#ifdef CAUTIOUS
	if( mip == NO_IOS_MEMBER_INFO ){
		sprintf(ERROR_STRING,
			"CAUTIOUS:  position_genwin %s, missing member info #2",
			IOS_ITEM_NAME(ip));
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	gwfp = (Genwin_Functions *) IOS_MBR_DATA(mip);

	(*gwfp->unshow_func)(QSP_ARG  IOS_ITEM_NAME(ip));
#endif /* ! BUILD_FOR_OBJC */
}
#endif // FOOBAR

void show_genwin(QSP_ARG_DECL  Gen_Win *gwp)
{
	switch( GW_TYPE(gwp) ){
		case GW_PANEL:
			show_panel(QSP_ARG  (Panel_Obj *)gwp);
			break;
		case GW_VIEWER:
			show_viewer(QSP_ARG  (Viewer *) gwp);
			break;
		case GW_NAV_PANEL:
#ifdef BUILD_FOR_OBJC
            push_nav(QSP_ARG  gwp);
#else // ! BUILD_FOR_OBJC
			show_panel(QSP_ARG  NAVP_PANEL( ((Nav_Panel *)gwp) ) );
#endif // ! BUILD_FOR_OBJC
			break;
		default:
			sprintf(ERROR_STRING,
	"show_genwin:  unexpected type code %d!?",GW_TYPE(gwp));
			WARN(ERROR_STRING);
			break;
	}
}

void unshow_genwin(QSP_ARG_DECL  Gen_Win *gwp)
{
	switch( GW_TYPE(gwp) ){
		case GW_PANEL:
			unshow_panel(QSP_ARG  (Panel_Obj *)gwp);
			break;
		case GW_VIEWER:
			unshow_viewer(QSP_ARG  (Viewer *) gwp);
			break;
		case GW_NAV_PANEL:
#ifdef BUILD_FOR_OBJC
            pop_nav(SINGLE_QSP_ARG  1);
#else // ! BUILD_FOR_OBJC
			unshow_panel(QSP_ARG  NAVP_PANEL(((Nav_Panel *)gwp)) );
#endif // ! BUILD_FOR_OBJC
			break;
		default:
			sprintf(ERROR_STRING,
	"show_genwin:  unexpected type code %d!?",GW_TYPE(gwp));
			WARN(ERROR_STRING);
			break;
	}
}


/* delete a window */
static void delete_genwin(QSP_ARG_DECL  Gen_Win *gwp)
{
#ifndef BUILD_FOR_OBJC
	Genwin_Functions *gwfp;
	IOS_Member_Info *mip;

	if( gwp == NULL ) return;

	INSURE_GENWIN

	mip = get_ios_member_info(QSP_ARG  gw_iclp,GW_NAME(gwp));

#ifdef CAUTIOUS
	if( mip == NO_IOS_MEMBER_INFO ){
		sprintf(ERROR_STRING,
			"CAUTIOUS:  position_genwin %s, missing member info #2",
			GW_NAME(gwp));
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	gwfp = (Genwin_Functions *) IOS_MBR_DATA(mip);

	(*gwfp->delete_func)(QSP_ARG  GW_NAME(gwp));
#endif /* ! BUILD_FOR_OBJC */
}

static COMMAND_FUNC( do_posn_func )
{
	Gen_Win *gwp;
	const char *s;
	int x, y;

	s = NAMEOF("genwin");
	x = (int)HOW_MANY("x position");
	y = (int)HOW_MANY("y position");

	gwp = find_genwin(QSP_ARG  s);
	if( gwp != NULL ) posn_genwin(QSP_ARG  gwp, x, y);
}

static COMMAND_FUNC( do_show_func )
{
	Gen_Win *gwp;
	const char *s;

	s = NAMEOF("genwin");
	gwp = find_genwin(QSP_ARG  s);
	if( gwp != NULL ) show_genwin(QSP_ARG  gwp);
}

static COMMAND_FUNC( do_unshow_func )
{
	Gen_Win *gwp;
	const char *s;

	s = NAMEOF("genwin");
	gwp = find_genwin(QSP_ARG  s);
	if( gwp != NULL ) unshow_genwin(QSP_ARG  gwp);
}

static COMMAND_FUNC( do_delete_func )
{
	Gen_Win *gwp;
	const char *s;

	s = NAMEOF("genwin");
	gwp = find_genwin(QSP_ARG  s);
	if( gwp != NULL ) delete_genwin(QSP_ARG  gwp);
}

#ifdef BUILD_FOR_OBJC

/* These functions are easier to implement for iOS, because genwin's
 * are a superclass, whereas for unix we construct our own classes...
 * But we should still be able to support list and info!?
 */

static COMMAND_FUNC( do_list_genwins )
{
	list_genwins(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_genwin_info )
{
	Gen_Win *gwp;
	const char *s;

	s = NAMEOF("genwin");
	gwp = find_genwin(QSP_ARG  s);
	if( gwp == NULL ) return;

	advise("Sorry, genwin_info not implemented!?");
}
#endif /* BUILD_FOR_OBJC */

#define ADD_CMD(s,f,h)	ADD_COMMAND(genwin_menu,s,f,h)

MENU_BEGIN(genwin)
ADD_CMD( position,	do_posn_func,		position a window	)
ADD_CMD( show,		do_show_func,		show a window		)
ADD_CMD( unshow,	do_unshow_func,		unshow a window		)
ADD_CMD( delete,	do_delete_func,		delete a window		)
#ifdef BUILD_FOR_OBJC
ADD_CMD( list,		do_list_genwins,	list all genwins	)
ADD_CMD( info,		do_genwin_info,		give info about a genwin )
#endif /* BUILD_FOR_OBJC */
MENU_END(genwin)

COMMAND_FUNC( do_genwin_menu )
{
	// used to call auto_version here, but this was in its own lib...
	PUSH_MENU(genwin);
}

