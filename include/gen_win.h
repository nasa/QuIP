
#ifndef _GEN_WIN_H_
#define _GEN_WIN_H_

#include "quip_config.h"
#include "quip_prot.h"
#include "item_type.h"
#include "function.h"
#include "genwin_flags.h"

typedef enum {
	GW_NONE,
	GW_PANEL,
	GW_VIEWER,
	GW_NAV_PANEL,
	GW_NAV_PANEL_OBJ,
	N_GW_TYPES
} GenWin_Type;

#ifndef BUILD_FOR_OBJC

// In standard C, a gen_win is a union of viewers and panels...
// But in Objective C gen_win is a superclass of both and must be defined
// first.  Do we really need this here for std C?

#include "map_ios_item.h"

//#define Gen_Win	IOS_Item	// Why IOS_Item, when not building for iOS???


typedef struct gen_win {
	Item		gw_item;
#define gw_name	gw_item.item_name
	GenWin_Type	gw_type;
} Gen_Win;

#define GW_NAME(gwp)		(gwp)->gw_item.item_name
#define GW_TYPE(gwp)		(gwp)->gw_type
#define SET_GW_TYPE(gwp,v)	(gwp)->gw_type = v

#else /* BUILD_FOR_OBJC */

#include "ios_item.h"
#include "quipAppDelegate.h"
//#ifdef BUILD_FOR_IOS
#include "quipViewController.h"
//#endif // BUILD_FOR_IOS
#ifdef BUILD_FOR_MACOS
#include "quipWindowController.h"
#endif // BUILD_FOR_MACOS

typedef enum {
	GW_VC_QVC,
	GW_VC_QTVC,
	N_VW_VC_TYPES
} Genwin_VC_Code;

// Logically, Viewer and Panel_Obj should be subclasses
// of Gen_Win.  But in order for them to each have their own
// namespace, they have to be separate subclasses of IOS_Item...
// Is that really true???

// The only reason they need to have their own namespace
// is for legacy compatibility with unix quip.  But it is
// probably bad practice to duplicate names?

// The view controller has a primary view...

@class Viewer;

@interface Gen_Win : IOS_Item
// retention cycle involving quipView, Viewer(Gen_Win) ??

@property (retain) quipAppDelegate *	qad;

// why?  other controllers?
//@property (retain) quipViewController *	qvc;
#ifdef BUILD_FOR_IOS
@property (retain) UIViewController *	vc;
@property Genwin_VC_Code		vc_type;
#endif // BUILD_FOR_IOS
#ifdef BUILD_FOR_MACOS
@property (retain) quipViewController *	vc;
@property (retain) quipWindowController *	wc;
@property (retain) NSWindow *		window;
@property (nonatomic, retain) quipCanvas *	canvas;		// for drawing
#endif // BUILD_FOR_MACOS

@property int32_t 			flags;
@property GenWin_Type 			type;
@property int				width;
@property int				height;
@property int				x;
@property int				y;
@property (retain) NSMutableArray *	cmap;
@property (retain) IOS_Item_Context *	icp;		// context for widget names
// BUG circular refs, one should be weak...
@property (retain) Panel_Obj *		po;
@property (retain) Viewer *		vp;

// We want the list of events to be flexible...
// But what is this an array of???
// It is an array of strings, indexed by event code...
@property (retain) NSMutableArray *	event_tbl;


+(void) initClass;

#define GW_CONTEXT(gwp)		(gwp).icp

//#ifdef BUILD_FOR_IOS
#define GW_VC(gwp)		(gwp).vc
#define GW_VC_TYPE(gwp)		(gwp).vc_type
#define SET_GW_VC(gwp,v)		(gwp).vc = v
#define SET_GW_VC_TYPE(gwp,t)		(gwp).vc_type = t
//#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS
#define GW_WINDOW(gwp)		(gwp).window
#define GW_WC(gwp)		(gwp).wc
#define GW_CANVAS(gwp)		(gwp).canvas
#define SET_GW_WINDOW(gwp,w)	(gwp).window = w
#define SET_GW_WC(gwp,wcp)	(gwp).wc = wcp
#define SET_GW_CANVAS(gwp,c)	(gwp).canvas = c
#endif // BUILD_FOR_MACOS

#define GW_TYPE(gwp)		(gwp).type
#define GW_CMAP(gwp)		(gwp).cmap
#define GW_NAME(gwp)		(gwp).name.UTF8String
#define GW_FLAGS(gwp)		(gwp).flags
#define GW_EVENT_TBL(gwp)	(gwp).event_tbl
#define GW_PO(gwp)		(gwp).po
#define GW_VW(gwp)		(gwp).vp

#define GW_WIDTH(gwp)		(gwp).width
#define GW_HEIGHT(gwp)		(gwp).height
#define GW_X(gwp)		(gwp).x
#define GW_Y(gwp)		(gwp).y

#define SET_GW_CONTEXT(gwp,icp)	(gwp).icp = icp
#define SET_GW_TYPE(gwp,v)	(gwp).type = v
#define SET_GW_CMAP(gwp,v)	(gwp).cmap = v
#define SET_GW_FLAGS(gwp,v)		(gwp).flags = v
#define SET_GW_FLAG_BITS(gwp,bits)	SET_GW_FLAGS(gwp,GW_FLAGS(gwp)|bits)
#define CLEAR_GW_FLAG_BITS(gwp,bits)	SET_GW_FLAGS(gwp,GW_FLAGS(gwp)&(~(bits)))
#define SET_GW_EVENT_TBL(gwp,v)	(gwp).event_tbl = v

#define SET_GW_WIDTH(gwp,v)	(gwp).width = v
#define SET_GW_HEIGHT(gwp,v)	(gwp).height = v
#define SET_GW_X(gwp,v)		(gwp).x = v
#define SET_GW_Y(gwp,v)		(gwp).y = v

#define SET_GW_PO(gwp,v)	(gwp).po = v
#define SET_GW_VW(gwp,v)	(gwp).vp = v

IOS_ITEM_INIT_PROT(Gen_Win,genwin)
IOS_ITEM_NEW_PROT(Gen_Win,genwin)
IOS_ITEM_CHECK_PROT(Gen_Win,genwin)
IOS_ITEM_GET_PROT(Gen_Win,genwin)
IOS_ITEM_PICK_PROT(Gen_Win,genwin)
IOS_ITEM_LIST_PROT(Gen_Win,genwin)
IOS_ITEM_ENUM_PROT(Gen_Win,genwin)

extern Gen_Win *find_genwin_for_vc(QUIP_VIEW_CONTROLLER_TYPE *vc);

@end

extern Gen_Win *curr_genwin;
extern int default_hide_back_button;


#define INSURE_GW_CMAP(gwp)				\
							\
	if( GW_CMAP(gwp) == NULL )			\
		init_gw_lut(gwp);

extern void init_gw_lut(Gen_Win *gwp);
extern Gen_Win *make_genwin(QSP_ARG_DECL  const char *name,int width,int height);

#ifdef BUILD_FOR_IOS
extern UIViewController *current_view_controller;
#endif // BUILD_FOR_IOS

#endif /* BUILD_FOR_OBJC */

#define init_genwins()		_init_genwins(SINGLE_QSP_ARG)
#define new_genwin(s)		_new_genwin(QSP_ARG  s)
#define genwin_of(s)		_genwin_of(QSP_ARG  s)
#define get_genwin(s)		_get_genwin(QSP_ARG  s)
#define pick_genwin(s)		_pick_genwin(QSP_ARG  s)
//#define del_genwin(s)		_del_genwin(QSP_ARG  s)
#define list_genwins(fp)	_list_genwins(QSP_ARG  fp)
#define genwin_list()		_genwin_list(SINGLE_QSP_ARG)

extern void _add_genwin(QSP_ARG_DECL  IOS_Item_Type *itp, Genwin_Functions *gwfp,
	IOS_Item *(*lookup)(QSP_ARG_DECL  const char *));
extern Gen_Win *_find_genwin(QSP_ARG_DECL  const char *);

#define add_genwin(itp,gwfp,lookup) _add_genwin(QSP_ARG  itp,gwfp,lookup)
#define find_genwin(s) _find_genwin(QSP_ARG  s)

extern COMMAND_FUNC( do_genwin_menu );

//#ifdef HAVE_X11
//extern Viewer *genwin_viewer(QSP_ARG_DECL  IOS_Item *);
//#endif
//
//#ifdef HAVE_MOTIF
//extern Panel_Obj *genwin_panel(QSP_ARG_DECL  IOS_Item *);
//#endif /* HAVE_MOTIF */

#include "display.h"
extern Dpyable *_genwin_display(QSP_ARG_DECL  Gen_Win *gwp);

extern void _show_genwin(QSP_ARG_DECL  Gen_Win *gwp);
extern void _unshow_genwin(QSP_ARG_DECL  Gen_Win *gwp);

extern void _push_nav(QSP_ARG_DECL  Gen_Win *gwp);
extern void _pop_nav(QSP_ARG_DECL int n_levels);
extern int n_pushed_panels(void);

#define genwin_display(gwp) _genwin_display(QSP_ARG  gwp)
#define show_genwin(gwp) _show_genwin(QSP_ARG  gwp)
#define unshow_genwin(gwp) _unshow_genwin(QSP_ARG  gwp)
#define push_nav(gwp) _push_nav(QSP_ARG  gwp)
#define pop_nav(n_levels) _pop_nav(QSP_ARG n_levels)


#endif /* _GEN_WIN_H_ */

