
#ifndef _PANEL_H_
#define _PANEL_H_

/* panel_obj.h */

#include "quip_prot.h"
#include "data_obj.h"
#include "display.h"
#include "genwin_flags.h"

#ifdef BUILD_FOR_OBJC
#include "quipAppDelegate.h"
#include "quipController.h"
#include "quipView.h"
#include "ios_item.h"
#include "gen_win.h"
#include "sizable.h"

@class Screen_Obj;

@interface Panel_Obj : IOS_Item

@property (retain) Gen_Win *	gwp;
@property int                   currx;		// position for placing next widget
@property int			curry;
@property (retain) IOS_List *	children;	// list of widgets

+(void) initClass;
//#ifdef NOT_USED
-(void) enableScrolling;
-(void) disableScrolling;
//#endif /* NOT_USED */

@end


#define PO_FLAGS(po)			GW_FLAGS(PO_GW(po))
#define PO_NAME(po)			po.name.UTF8String
#define PO_CURR_X(po)			po.currx
#define PO_CURR_Y(po)			po.curry
#define PO_WIDTH(po)			GW_WIDTH(PO_GW(po))
#define PO_HEIGHT(po)			GW_HEIGHT(PO_GW(po))
#define PO_X(po)			GW_X(PO_GW(po))
#define PO_Y(po)			GW_Y(PO_GW(po))
#define PO_CHILDREN(po)			po.children
#define PO_CONTEXT(po)			GW_CONTEXT(PO_GW(po))
#define PO_VC(po)			GW_VC(PO_GW(po))
#define PO_WINDOW(po)			GW_WINDOW(PO_GW(po))
#define PO_GW(po)			(po).gwp
#define PO_QV(po)			((quipView *)PO_VC(po).view)

#define SET_PO_GW(po,v)			(po).gwp = v
#define SET_PO_FLAGS(po,f)		SET_GW_FLAGS(PO_GW(po),f)
#define SET_PO_CURR_X(po,v)		[po setCurrx:v]
#define SET_PO_CURR_Y(po,v)		[po setCurry:v]
#define INC_PO_CURR_Y(po,v)		[po setCurry:(PO_CURR_Y(po)+v)]
#define SET_PO_WIDTH(po,v)		SET_GW_WIDTH(PO_GW(po), v)
#define SET_PO_HEIGHT(po,v)		SET_GW_HEIGHT(PO_GW(po), v)
#define SET_PO_X(po,v)			SET_GW_X(PO_GW(po),v)
#define SET_PO_Y(po,v)			SET_GW_Y(PO_GW(po),v)
#define SET_PO_CHILDREN(po,v)		[po setChildren:v]
#define SET_PO_CONTEXT(po,v)		SET_GW_CONTEXT(PO_GW(po),v)

IOS_ITEM_INIT_PROT(Panel_Obj,panel_obj)
IOS_ITEM_NEW_PROT(Panel_Obj,panel_obj)
IOS_ITEM_PICK_PROT(Panel_Obj,panel_obj)
IOS_ITEM_GET_PROT(Panel_Obj,panel_obj)
IOS_ITEM_DEL_PROT(Panel_Obj,panel_obj)
IOS_ITEM_CHECK_PROT(Panel_Obj,panel_obj)
IOS_ITEM_ENUM_PROT(Panel_Obj,panel_obj)
IOS_ITEM_LIST_PROT(Panel_Obj,panel_obj)

//#define GET_PANEL_OBJ(s)	get_panel_obj(QSP_ARG  s)

/* panel.m */
extern void add_to_panel(Panel_Obj *po, Screen_Obj *sop);

#else /* ! BUILD_FOR_OBJC */

#include "xsupp.h"

typedef struct panel_object {
	Gen_Win		po_genwin;
	Dpyable		po_dpyable;
	int		po_currx, po_curry;	/* for added items */
	int		object_item_type;	// what is this???
	Item_Context *	po_icp;	// for screen objects?
	int		po_flags;
#ifdef HAVE_MOTIF
	Display *	po_dpy;
	Visual *	po_visual;
	int		po_screen_no;
	GC		po_gc;
#endif /* HAVE_MOTIF */
    
} Panel_Obj;

#define PO_NAV_PANEL	1
#define IS_NAV_PANEL(pnl_p)	(PO_FLAGS(pnl_p)&PO_NAV_PANEL)

#define po_cmap		po_dpyable.dpa_cmap
#define po_cm_dp	po_dpyable.dpa_cm_dp
#define po_lt_dp	po_dpyable.dpa_lt_dp

//#define po_dop		po_dpyable.dpa_dop
#define PO_DPA(pop)	(&((pop)->po_dpyable))
#define PO_GWP(po)	(&((po)->po_genwin))

#define PO_NAME(pop)	GW_NAME(PO_GWP(pop))

#define PO_WIDTH(pop)		DPA_WIDTH(PO_DPA(pop))
#define SET_PO_WIDTH(po,v)	SET_DPA_WIDTH(PO_DPA(po),v)

#define PO_DOP(po)			DPA_DOP(PO_DPA(po))
#define SET_PO_DOP(po,v)		SET_DPA_DOP(PO_DPA(po),v)
#define PO_ROOTW(po)			DO_ROOTW(PO_DOP(po))

#define po_height	po_dpyable.dpa_height
#define po_dx		po_dpyable.dpa_width
#define po_dy		po_dpyable.dpa_height
#define po_x		po_dpyable.dpa_x
#define po_y		po_dpyable.dpa_y
#define po_xwin		po_dpyable.dpa_xwin
#define po_flags	po_dpyable.dpa_flags
#define po_children	po_dpyable.dpa_children
/* #define po_dpy		po_dpyable.dpa_dpy		*/
/* #define po_gc		po_dpyable.dpa_gc		*/
/* #define po_screen_no	po_dpyable.dpa_screen_no		*/
/* #define po_visual	po_dpyable.dpa_visual		*/
#define po_frame_obj	po_dpyable.dpa_frame_obj
#define po_panel_obj	po_dpyable.dpa_thing_obj
#define po_pw		po_dpyable.dpa_pw
#define po_realized     po_dpyable.dpa_realized


#define PANEL_HAS_SYSCOLS( po )		HAS_SYSCOLS(po->po_xldp)
#define PANEL_KNOWS_SYSCOLS( po )	(po->po_flags & PANEL_KNOWS_SYSTEM_COLORS)


#define PO_DPYABLE(po)			(&(po->po_dpyable))
#define PO_CMAP_OBJ(po)			DPA_CMAP_OBJ( PO_DPYABLE(po) )
#define PO_LINTBL_OBJ(po)		DPA_LINTBL_OBJ( PO_DPYABLE(po) )
#define SET_PO_CMAP_OBJ(po,v)		DPA_CMAP_OBJ( PO_DPYABLE(po) ) = v
#define SET_PO_LINTBL_OBJ(po,v)		DPA_LINTBL_OBJ( PO_DPYABLE(po) ) = v

#define PO_FLAGS(po)			po->po_flags
#define PO_CURR_X(po)			po->po_currx
#define PO_CURR_Y(po)			po->po_curry
#define PO_HEIGHT(po)			po->po_height
#define PO_X(po)			po->po_x
#define PO_Y(po)			po->po_y
#define PO_CHILDREN(po)			po->po_children
#define PO_CONTEXT(po)			po->po_icp

#define SET_PO_FLAGS(po,f)		PO_FLAGS(po) = (f)
#define SET_PO_CURR_X(po,v)		PO_CURR_X(po) = v
#define SET_PO_CURR_Y(po,v)		PO_CURR_Y(po) = v
#define INC_PO_CURR_Y(po,v)		PO_CURR_Y(po) += v
#define SET_PO_HEIGHT(po,v)		PO_HEIGHT(po) = v
#define SET_PO_X(po,v)			PO_X(po) = v
#define SET_PO_Y(po,v)			PO_Y(po) = v
#define SET_PO_CHILDREN(po,v)		PO_CHILDREN(po) = v
#define SET_PO_CONTEXT(po,v)		PO_CONTEXT(po) = v

// For unix, the "genwin" is simply the panel itself...
// Does that make sense???
#define PO_GW(po)			(Gen_Win *)po

ITEM_INTERFACE_PROTOTYPES(Panel_Obj,panel_obj)

struct screen_obj;
typedef struct screen_obj Screen_Obj;


#endif /* ! BUILD_FOR_OBJC */

#define pick_panel(pmpt)	_pick_panel_obj(QSP_ARG  pmpt)
#define init_panel_objs()	_init_panel_objs(SINGLE_QSP_ARG)
#define new_panel_obj(s)	_new_panel_obj(QSP_ARG  s)
#define get_panel_obj(s)	_get_panel_obj(QSP_ARG  s)
#define panel_obj_of(s)		_panel_obj_of(QSP_ARG  s)
#define list_panel_objs(fp)	_list_panel_objs(QSP_ARG  fp)
#define panel_obj_list()	_panel_obj_list(SINGLE_QSP_ARG)

#define CLEAR_PANEL_FLAG_BITS(po,bits)	SET_GW_FLAGS(PO_GW(po),GW_FLAGS(PO_GW(po)) & ~(bits) )
#define SET_PANEL_FLAG_BITS(po,bits)	SET_GW_FLAGS(PO_GW(po),GW_FLAGS(PO_GW(po)) | (bits) )

#define PANEL_MAPPED( po )		( PO_FLAGS(po) & PANEL_SHOWN )
#define PANEL_UNMAPPED( po )		(( PO_FLAGS(po) & PANEL_SHOWN )==0)
#define CANNOT_SHOW_PANEL( po )	(( PO_FLAGS( po )&PANEL_IS_SHOWABLE)==0)
#define CAN_SHOW_PANEL( po )	( PO_FLAGS( po )&PANEL_IS_SHOWABLE )

/* globals */

extern Panel_Obj *curr_panel;
#ifdef HAVE_MOTIF
extern Widget curr_thing_obj;
#endif // HAVE_MOTIF

/* prototypes */



/* screen_objs.c */

extern void list_panels(void);
extern Panel_Obj *new_panel(QSP_ARG_DECL  const char *name,int dx,int dy);
extern List *panel_list(SINGLE_QSP_ARG_DECL);
extern Node *first_panel_node(SINGLE_QSP_ARG_DECL);
extern double panel_exists(QSP_ARG_DECL  const char *);

extern void make_panel(QSP_ARG_DECL  Panel_Obj *po,int,int);
extern void activate_panel(QSP_ARG_DECL  Panel_Obj *po, int yesno);
extern void panel_cmap(Panel_Obj *po, Data_Obj *cm_dp);
extern void label_panel(Panel_Obj *po, const char *s);
extern void posn_panel(Panel_Obj *po);
extern void free_wsys_stuff(Panel_Obj *po);
extern void window_cm(Panel_Obj *po,Data_Obj *cm_dp);

#ifdef BUILD_FOR_OBJC
extern Panel_Obj *find_panel(QSP_ARG_DECL  quipView *qv);
extern Gen_Win *dummy_panel(QSP_ARG_DECL  const char *name,int dx,int dy);
//extern double get_panel_size(QSP_ARG_DECL  IOS_Item *ip, int index);
extern IOS_Size_Functions panel_sf;
#endif /* BUILD_FOR_OBJC */

// These functions typically go in an implementation support file...
extern void _show_panel(QSP_ARG_DECL  Panel_Obj *po);
extern void _unshow_panel(QSP_ARG_DECL  Panel_Obj *po);

#define show_panel(po) _show_panel(QSP_ARG  po)
#define unshow_panel(po) _unshow_panel(QSP_ARG  po)

/* panel.m */
extern void add_to_panel(Panel_Obj *po, Screen_Obj *sop);
extern void remove_from_panel(Panel_Obj *po, Screen_Obj *sop);

#endif /* ! _PANEL_H_ */

