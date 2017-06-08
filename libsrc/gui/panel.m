
#include "quip_config.h"

#import "quip_prot.h"
#include "panel_obj.h"
#include "ios_gui.h"
#include "sizable.h"

#ifdef BUILD_FOR_OBJC

@implementation Panel_Obj

@synthesize currx;
@synthesize curry;
@synthesize children;
@synthesize gwp;

+(void) initClass
{
	/* nop */
}

// We haven't provided an init function - why not?

/* These don't compile if quipView is not a UIScrollView...
 * We will eventually need to deal with this.
 */

-(void) disableScrolling
{
#ifdef BUILD_FOR_IOS

#ifdef SCROLLABLE_QUIP_VIEW
	[PO_QV(self) setScrollEnabled:NO];
#else // ! SCROLLABLE_QUIP_VIEW
    ADVISE("disableScrolling:  quipView objects are not scrollable in this build!?");
#endif // ! SCROLLABLE_QUIP_VIEW
    
#endif // BUILD_FOR_IOS
}

-(void) enableScrolling
{
#ifdef BUILD_FOR_IOS

#ifdef SCROLLABLE_QUIP_VIEW
	[PO_QV(self) setScrollEnabled:YES];
#else // ! SCROLLABLE_QUIP_VIEW
    ADVISE("disableScrolling:  quipView objects are not scrollable in this build!?");
#endif // ! SCROLLABLE_QUIP_VIEW
    
#endif // BUILD_FOR_IOS
}

@end


static IOS_Item_Type *panel_obj_itp=NULL;

// Don't use the macro for the init func, so that we can
// also call the size function thingy...
//IOS_ITEM_INIT_FUNC(Panel_Obj,panel_obj)

void init_panel_objs(SINGLE_QSP_ARG_DECL)
{
	[Panel_Obj initClass];
	panel_obj_itp = [[IOS_Item_Type alloc]
		initWithName : @"Panel_Obj" ];
	//add_ios_sizable(QSP_ARG  panel_obj_itp,&panel_sf, NULL );
}

IOS_ITEM_NEW_FUNC(Panel_Obj,panel_obj)
IOS_ITEM_PICK_FUNC(Panel_Obj,panel_obj)
IOS_ITEM_CHECK_FUNC(Panel_Obj,panel_obj)
IOS_ITEM_ENUM_FUNC(Panel_Obj,panel_obj)
IOS_ITEM_LIST_FUNC(Panel_Obj,panel_obj)

void add_to_panel(Panel_Obj *po, Screen_Obj *sop)
{
#ifdef BUILD_FOR_IOS
	quipView *qv=PO_QV(po);
	
	UIView *cp = SOB_CONTROL(sop);

	[qv addSubview:cp];
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
	NSView *cp = SOB_CONTROL(sop);

#ifdef CAUTIOUS
	if( [GW_WINDOW(PO_GW(po)) contentView] == nil ){
		fprintf(stderr,"CAUTIOUS:  add_to_panel:  window contentView is null!?\n");
	}
#endif // CAUTIOUS

	[[GW_WINDOW(PO_GW(po)) contentView] addSubview : cp ];
#endif // BUILD_FOR_MACOS
    
	// The last two things are the things we do in the unix code...
	ios_addHead(PO_CHILDREN(po),mk_ios_node(sop));
	SET_SOB_PARENT(sop, po);
}

#ifdef FOOBAR
double get_panel_size(QSP_ARG_DECL  IOS_Item *ip, int index)
{
	Panel_Obj *po;

	po = (Panel_Obj *)ip;

#define DEFAULT_PANEL_DEPTH 32.0

	switch(index){
		case 0:	return DEFAULT_PANEL_DEPTH; break;
		case 1:	return PO_WIDTH(po); break;
		case 2:	return PO_HEIGHT(po); break;
		case 3: return 1.0; break;
		case 4: return 1.0; break;
#ifdef CAUTIOUS
		default:
			WARN("CAUTIOUS:  bad index passed to get_panel_size!?");
			return 0.0;
			break;
#endif /* CAUTIOUS */
	}
}
#endif // FOOBAR

#endif // BUILD_FOR_OBJC

