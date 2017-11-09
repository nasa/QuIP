
#ifndef _SCREEN_OBJ_H_
#define _SCREEN_OBJ_H_

/* screen_obj.h */

#include "panel_obj.h"
#include "display.h"
#include "sizable.h"

// only needed for iOS?
typedef enum {
	QUIP_ORI_ALL,
	QUIP_ORI_PORTRAIT_BOTH,
	QUIP_ORI_PORTRAIT_UP,
	QUIP_LANDSCAPE_BOTH,
	QUIP_LANDSCAPE_RIGHT,
	QUIP_LANDSCAPE_LEFT,
	N_TABLET_ORIENTATIONS
} Quip_Allowed_Orientations;

#define SOB_FILENAME	"(screen object pushed text)"

typedef enum {
	SOT_UNUSED,		// don't use 0 as a code
	SOT_TEXT,
	SOT_PASSWORD,		// like text, but don't display chars
	SOT_BUTTON,
	SOT_SLIDER,
	SOT_ADJUSTER,		// just a slider with continuous updates
	SOT_MENU,
	SOT_MENU_ITEM,
	SOT_GAUGE,
	SOT_MESSAGE,
	SOT_PULLRIGHT,
	SOT_MENU_BUTTON,
	SOT_SCROLLER,
	SOT_CHOOSER,
	SOT_MLT_CHOOSER,
	SOT_PICKER,
	SOT_TOGGLE,
	SOT_LABEL,
	// These used to be ifdef'd BUILD_FOR_IOS...
	SOT_TEXT_BOX,			// not implemented yet for X11
	SOT_EDIT_BOX,			// not implemented yet for X11
	SOT_ACTIVITY_INDICATOR,		// not implemented yet for X11
	N_WIDGET_TYPES		// one too big because of UNUSED...
} Screen_Obj_Type;

#define IS_A_TYPE_OF_GAUGE(sop)					\
								\
	(	SOB_TYPE(sop) == SOT_SLIDER	||		\
		SOB_TYPE(sop) == SOT_ADJUSTER	||		\
		SOB_TYPE(sop) == SOT_GAUGE		)

#define IS_CHOOSER(sop)						\
								\
	(	SOB_TYPE(sop) == SOT_CHOOSER	||		\
		SOB_TYPE(sop) == SOT_MLT_CHOOSER		)

#define MAX_CYLINDERS		3


#ifdef BUILD_FOR_OBJC

#include "ios_item.h"

@interface Screen_Obj : IOS_Item

@property (retain) IOS_List *	children;
@property uint32_t flags;
// in C version these three are a union...
@property const char *selector;
@property const char **selectors;
@property const char ***selectorTbl;
@property int nCylinders;
@property int *countTbl;	// selections per cylinder

@property const char *action;
@property Screen_Obj_Type widget_type;
@property const char *content;
@property int min;
@property int max;
@property int val;	// overloaded as n_selectors, and font_size
@property int width;
@property int height;
@property (retain) Panel_Obj *panel;
@property int	x;
@property int	y;

#ifdef BUILD_FOR_IOS
@property (retain) UIView *control;
@property (retain) UILabel *label;
#endif // BUILD_FOR_IOS
#ifdef BUILD_FOR_MACOS
@property (retain) NSControl *control;
@property (retain) NSTextView *label;
#endif // BUILD_FOR_MACOS

/* there is only one app delegate, so
 * why should every one need a personal copy?
 * We make is a class variable...
 */

#ifdef THREAD_SAFE_QUERY
@property Query_Stack *	qsp;
#endif /* THREAD_SAFE_QUERY */

+(void) initClass;
+(quipAppDelegate *) getAppDelegate;
+(int) contextStackDepth;

#ifdef BUILD_FOR_MACOS
+(IOS_List *) getListOfAllItems;
#endif // BUILD_FOR_MACOS

-(void) setSelectorsAtIdx: (int) idx withValue: (const char **) string_arr;
-(const char **) getSelectorsAtIdx: (int) idx;
-(void) setSelectorInTbl: (int) tbl atIdx: (int) idx withValue: (const char *) string;
-(void) setCount:(int) n atIdx: (int) idx ;
-(const char *) getSelectorAtIdx:(int) idx fromTbl: (int) tbl ;

@end

#define SOB_TYPE(sop)		sop.widget_type
#define SOB_ACTION(sop)		sop.action
#define SOB_CONTENT(sop)		sop.content
#define SOB_SELECTOR(sop)	sop.selector
#define SOB_SELECTOR_TBL(sop)	sop.selectorTbl
#define SOB_COUNT_TBL(sop)	sop.countTbl
#define SOB_SELECTOR_AT_IDX(sop,tbl,idx)	[sop getSelectorAtIdx:idx fromTbl:tbl]
#define SOB_SELECTORS_AT_IDX(sop,idx)	[sop getSelectorsAtIdx:idx]
#define SOB_SELECTORS(sop)	sop.selectors
#define SOB_N_SELECTORS(sop)	sop.val
#define SOB_N_SELECTORS_AT_IDX(sop,idx)	(((sop).countTbl)[idx])
#define SOB_FLAGS(sop)		sop.flags
#define SOB_NAME(sop)		sop.name.UTF8String
#define SOB_MIN(sop)		sop.min
#define SOB_MAX(sop)		sop.max
#define SOB_VAL(sop)		sop.val
#define SOB_FONT_SIZE(sop)	sop.val
#define SOB_WIDTH(sop)		sop.width
#define SOB_HEIGHT(sop)		sop.height
#define SOB_PANEL(sop)		sop.panel
#define SOB_QSP(sop)		sop.qsp
#define SOB_PARENT(sop)		NULL
#define SOB_X(sop)		sop.x
#define SOB_Y(sop)		sop.y
#define SOB_CHILDREN(sop)	sop.children
#define SOB_CONTROL(sop)		sop.control
#define SOB_LABEL(sop)		sop.label
#define SOB_N_CYLINDERS(sop)	sop.nCylinders

#define SET_SOB_TYPE(sop,v)	[sop setWidget_type:v]
#define SET_SOB_ACTION(sop,v)	[sop setAction:v]
#define SET_SOB_CONTENT(sop,v)	[sop setContent:v]
// We can have one selector,
// an array of selectors,
// or a table of multiple selector arrays
#define SET_SOB_SELECTOR(sop,v)	[sop setSelector:v]

#define SET_SOB_SELECTORS(sop,v)	[sop setSelectors:v]
#define SET_SOB_N_SELECTORS(sop,v)	[sop setVal:v]

#define SET_SOB_SELECTOR_TBL(sop,v)	[sop setSelectorTbl:v]
#define SET_SOB_COUNT_TBL(sop,v)	[sop setCountTbl:v]
#define SET_SOB_N_CYLINDERS(sop,v)	[sop setNCylinders:v]
#define SET_SOB_SELECTORS_AT_IDX(sop,tbl,v)	[sop setSelectorsAtIdx:tbl withValue: v]
#define SET_SOB_N_SELECTORS_AT_IDX(sop,idx,v)	[sop setCount:v atIdx:idx]
#define SET_SOB_SELECTOR_AT_IDX(sop,tbl,idx,v)	[sop setSelectorInTbl:tbl atIdx:idx withValue: v]

// BUG - this looks wrong!
#define SET_SOB_COUNT_TBL(sop,v)	[sop setCountTbl:v]

#define SET_SOB_FLAGS(sop,v)	[sop setFlags:v]
#define SET_SOB_FLAG_BITS(sop,v)	[sop setFlags:(sop.flags | (v) )]
#define CLEAR_SOB_FLAG_BITS(sop,v)	[sop setFlags:(sop.flags & (~(v)) )]
#define SET_SOB_NAME(sop,v)	[sop setName:STRINGOBJ(v)]
#define SET_SOB_MIN(sop,v)	[sop setMin:v]
#define SET_SOB_MAX(sop,v)	[sop setMax:v]
#define SET_SOB_VAL(sop,v)	[sop setVal:v]
#define SET_SOB_FONT_SIZE(sop,v)	[sop setVal:v]
#define SET_SOB_WIDTH(sop,v)	[sop setWidth:v]
#define SET_SOB_HEIGHT(sop,v)	[sop setHeight:v]
#define SET_SOB_PANEL(sop,v)	[sop setPanel:v]
#define SET_SOB_QSP(sop,v)	[sop setQsp:v]
#define SET_SOB_PARENT(sop,v)	/* nop */ [sop setFlags : sop.flags]
#define SET_SOB_X(sop,v)		[sop setX: v]
#define SET_SOB_Y(sop,v)		[sop setY: v]
#define SET_SOB_CHILDREN(sop,l)	[sop setChildren:l]
#define SET_SOB_CONTROL(sop,v)	[sop setControl:v]
#define SET_SOB_LABEL(sop,l)	(sop).label = l

#else /* ! BUILD_FOR_OBJC */

struct screen_obj {
	IOS_Item		so_item;
	Dpyable			so_dpyable;
	union {
		const char ***	u_selector_tbl;
		const char **	u_selectors;
		const char *	u_selector;
	} so_sel_u;
	int *			so_count_tbl;	// used for iOS pickers?
#define so_selector	so_sel_u.u_selector
#define so_selectors	so_sel_u.u_selectors
#define so_selector_tbl	so_sel_u.u_selector_tbl
	const char *		so_action_text;		/* text to interpret */
	const char *		so_content_text;	/* text to display */
	int			so_intarr[1+MAX_CYLINDERS];
	int *			so_n_selectors_tbl;
#define so_min			so_intarr[0]
#define so_max			so_intarr[1]
#define so_val			so_intarr[2]
#define so_n_selectors		so_intarr[2]
#define so_n_cylinders		so_intarr[0]

//	int		so_min,so_max;		/* for sliders & guages */
//	int		so_val;			/* slider or guage value, overloaded as n_selectors */

	int		so_width;		/* adjuster width */
	int		so_height;
	Panel_Obj *	so_menu;
	Panel_Obj *	so_panel;
	Screen_Obj_Type	so_widget_type;
#ifdef HAVE_MOTIF
	Widget		so_obj;
#endif /* HAVE_MOTIF */

#ifdef THREAD_SAFE_QUERY
	Query_Stack *	so_qsp;
#endif /* THREAD_SAFE_QUERY */

};

#ifdef THREAD_SAFE_QUERY
#define SOB_QSP_ARG	(sop)->so_qsp,
#else
#define SOB_QSP_ARG
#endif /* ! THREAD_SAFE_QUERY */

/* lists don't need a value, so we do a union with this macro: */
#define so_nlist	so_val

#define so_xwin		so_dpyable.dpa_xwin
#define so_name		so_item.item_name
#define so_x		so_dpyable.dpa_x
#define so_y		so_dpyable.dpa_y
#define so_dx		so_dpyable.dpa_width
#define so_dy		so_dpyable.dpa_height
// Is it a good idea to share these flags with so_dpyable?
#define so_flags	so_dpyable.dpa_flags
#define so_parent	so_dpyable.dpa_parent
#define so_children	so_dpyable.dpa_children
#define so_frame	so_dpyable.dpa_frame_obj

#define SOB_TYPE(sop)		(sop)->so_widget_type
#define SOB_ACTION(sop)		(sop)->so_action_text
#define SOB_CONTENT(sop)		(sop)->so_content_text
#define SOB_SELECTOR(sop)	(sop)->so_selector
#define SOB_SELECTORS(sop)	(sop)->so_selectors
#define SOB_SELECTOR_TBL(sop)	(sop)->so_selector_tbl
#define SOB_COUNT_TBL(sop)	(sop)->so_count_tbl
#define SOB_N_SELECTORS(sop)	(sop)->so_val
#define SOB_FLAGS(sop)		(sop)->so_flags
#define SOB_NAME(sop)		(sop)->so_name
#define SOB_MIN(sop)		(sop)->so_min
#define SOB_MAX(sop)		(sop)->so_max
#define SOB_VAL(sop)		(sop)->so_val
#define SOB_FONT_SIZE(sop)	(sop)->so_val
#define SOB_WIDTH(sop)		(sop)->so_width
#define SOB_HEIGHT(sop)		(sop)->so_height
#define SOB_PANEL(sop)		(sop)->so_panel
#define SOB_QSP(sop)		(sop)->so_qsp
#define SOB_PARENT(sop)		(sop)->so_parent
#define SOB_X(sop)		(sop)->so_x
#define SOB_Y(sop)		(sop)->so_y
#define SOB_CHILDREN(sop)	(sop)->so_children
#define SOB_FRAME(sop)		(sop)->so_frame
#define SOB_N_CYLINDERS(sop)	(sop)->so_n_cylinders
// now COUNT_TBL
//#define SOB_N_SELECTORS_TBL(sop)	(sop)->so_n_selectors_tbl
#define SOB_N_SELECTORS_AT_IDX(sop,idx)	((SOB_COUNT_TBL(sop))[idx])
#define SOB_SELECTORS_AT_IDX(sop,idx)	(SOB_SELECTOR_TBL(sop))[idx]
#define SOB_SELECTOR_AT_IDX(sop,tbl,idx)	((SOB_SELECTOR_TBL(sop))[tbl])[idx]

#define SET_SOB_TYPE(sop,v)	(sop)->so_widget_type = v
#define SET_SOB_ACTION(sop,v)	SOB_ACTION(sop) = v
#define SET_SOB_CONTENT(sop,v)	SOB_CONTENT(sop) = v
#define SET_SOB_SELECTOR(sop,v)	SOB_SELECTOR(sop) = v
#define SET_SOB_SELECTORS(sop,v)	SOB_SELECTORS(sop) = v
#define SET_SOB_SELECTOR_TBL(sop,v)	SOB_SELECTOR_TBL(sop) =  v
#define SET_SOB_COUNT_TBL(sop,v)	SOB_COUNT_TBL(sop) =  v
#define SET_SOB_N_SELECTORS(sop,v)	SOB_N_SELECTORS(sop) = v
#define SET_SOB_N_SELECTORS_AT_IDX(sop,idx,v)	SOB_N_SELECTORS_AT_IDX(sop,idx)=v
#define SET_SOB_FLAGS(sop,v)	SOB_FLAGS(sop) = v
#define SET_SOB_FLAG_BITS(sop,v)	SOB_FLAGS(sop) |= v
#define CLEAR_SOB_FLAG_BITS(sop,v)	SOB_FLAGS(sop) &= ~(v)
#define SET_SOB_NAME(sop,v)	SOB_NAME(sop) = v
#define SET_SOB_MIN(sop,v)	SOB_MIN(sop) = v
#define SET_SOB_MAX(sop,v)	SOB_MAX(sop) = v
#define SET_SOB_VAL(sop,v)	SOB_VAL(sop) = v
#define SET_SOB_WIDTH(sop,v)	SOB_WIDTH(sop) = v
#define SET_SOB_HEIGHT(sop,v)	SOB_HEIGHT(sop) = v
#define SET_SOB_FONT_SIZE(sop,v)	SOB_FONT_SIZE(sop) = v
#define SET_SOB_PANEL(sop,v)	SOB_PANEL(sop) = v
#define SET_SOB_QSP(sop,v)	SOB_QSP(sop) = v
#define SET_SOB_PARENT(sop,v)	SOB_PARENT(sop) = v
#define SET_SOB_X(sop,v)		SOB_X(sop) = v
#define SET_SOB_Y(sop,v)		SOB_Y(sop) = v
#define SET_SOB_CHILDREN(sop,lp)	(sop)->so_children = lp
#define SET_SOB_FRAME(sop,v)	SOB_FRAME(sop) = v
#define SET_SOB_N_CYLINDERS(sop,n)	SOB_N_CYLINDERS(sop) = n
#define SET_SOB_SELECTORS_AT_IDX(sop,tbl,v)	SOB_SELECTORS_AT_IDX(sop,tbl)=v
#define SET_SOB_SELECTOR_AT_IDX(sop,tbl,idx,v)	SOB_SELECTOR_AT_IDX(sop,tbl,idx)=v

#include "map_ios_item.h"

#endif /* ! BUILD_FOR_OBJC */

#define WIDGET_PANEL(sop) ((Panel_Obj *)SOB_PARENT(sop))

#define OBJECT_TYPE_MASK	15

//#define WIDGET_INDEX(sop)	(WIDGET_CODE(sop)-1)
//#define WIDGET_CODE(sop)	(SOB_FLAGS(sop) & OBJECT_TYPE_MASK)
#define WIDGET_INDEX(sop)	(WIDGET_TYPE(sop)-1)
#define WIDGET_TYPE(sop)	(SOB_TYPE(sop))

#define IS_BUTTON(sop)		(WIDGET_TYPE(sop)==SOT_BUTTON)
#define IS_TEXT(sop)		(WIDGET_TYPE(sop)==SOT_TEXT)
#define IS_TOGGLE(sop)		(WIDGET_TYPE(sop)==SOT_TOGGLE)
#define IS_MENU_BUTTON(sop)	(WIDGET_TYPE(sop)==SOT_MENU_BUTTON)
#define IS_LIT(sop)		((sop)->so_flags & SOF_LIT)
#define IS_ACTIVE(sop)		((sop)->so_flags & SOF_ACTIVE)
#define LIGHT(bo)		( bo )->so_flags |= SOF_LIT
#define UNLIGHT(bo)		( bo )->so_flags &= ~SOF_LIT
#define ACTIVATE(bo)		( bo )->so_flags |= SOF_ACTIVE
#define DEACTIVATE(bo)		( bo )->so_flags &= ~SOF_ACTIVE

/* object flags */
// BUG the flag word is shared with so_dpyable (Dpyable), so these
// flag bits can't conflict with those...
// That seems especially bad now that we are merged with iOS,
// which doesn't even use so_dpyable!?
#define SOF_ACTIVE		16
#define SOF_LIT			32
#define SOF_KEYBOARD_DISMISSING	64


#define BUTTON_UP	1	/* event_flags(event) == 1 */
#define BUTTON_WIDTH	50
//#define BUTTON_HEIGHT	19	// where did this value come from?
#define BUTTON_HEIGHT	30	// BUG should be a variable!

#ifdef BUILD_FOR_IOS
#define TOGGLE_HEIGHT	BUTTON_HEIGHT
#else
#define TOGGLE_HEIGHT	19
#endif /* ! BUILDE_FOR_IOS */

#ifdef BUILD_FOR_MACOS
#define MESSAGE_HEIGHT	24	// BUG - should vary with font
#else // ! BUILD_FOR_MACOS
#define MESSAGE_HEIGHT	40
#endif // ! BUILD_FOR_MACOS

// Are these motif constants?
#define OBJECT_GAP	5
#define GAP_HEIGHT	10
#define SIDE_GAP	10
#define SLIDER_HEIGHT	75
#define SLIDER_WIDTH	100
#define GAUGE_HEIGHT	SLIDER_HEIGHT
#define GAUGE_WIDTH	78
#define MESSAGE_WIDTH	78

#define SHRINK_WHEN_LIT	2

#define MENU_WIDTH	50
#define MENU_LENGTH	100

#define SCROLLER_GAP		10
#define N_SCROLLER_LINES	8
#define GAP_PER_LINE		19
#define SCROLLER_HEIGHT		(SCROLLER_GAP+N_SCROLLER_LINES*GAP_PER_LINE)
#define SCROLLER_WIDTH 		300

#define CHOOSER_HEIGHT		22
#define CHOOSER_ITEM_HEIGHT	30

/* globals */


/* prototypes */

#ifdef BUILD_FOR_OBJC

IOS_ITEM_INIT_PROT(Screen_Obj,scrnobj)
IOS_ITEM_NEW_PROT(Screen_Obj,scrnobj)
IOS_ITEM_CHECK_PROT(Screen_Obj,scrnobj)
IOS_ITEM_PICK_PROT(Screen_Obj,scrnobj)
IOS_ITEM_ENUM_PROT(Screen_Obj,scrnobj)

extern IOS_Item_Context *_pop_scrnobj_context(SINGLE_QSP_ARG_DECL);
extern void _push_scrnobj_context(QSP_ARG_DECL  IOS_Item_Context *icp);

#define pop_scrnobj_context() _pop_scrnobj_context(SINGLE_QSP_ARG)
#define push_scrnobj_context(icp) _push_scrnobj_context(QSP_ARG  icp)

extern IOS_Size_Functions scrnobj_sf;

extern IOS_List *all_scrnobjs(SINGLE_QSP_ARG_DECL);

#else /* ! BUILD_FOR_OBJC */

ITEM_INTERFACE_PROTOTYPES(Screen_Obj,scrnobj)


#endif /* ! BUILD_FOR_OBJC */

#define init_scrnobjs()		_init_scrnobjs(SINGLE_QSP_ARG)
#define new_scrnobj(s)		_new_scrnobj(QSP_ARG  s)
#define scrnobj_of(s)		_scrnobj_of(QSP_ARG  s)
#define pick_scrnobj(pmpt)	_pick_scrnobj(QSP_ARG  pmpt)
#define del_scrnobj(s)		_del_scrnobj(QSP_ARG  s)


/* screen_objs.c */

extern Screen_Obj *dup_so(QSP_ARG_DECL  Screen_Obj *sop);
extern void so_info(Screen_Obj *sop);
extern Screen_Obj *_get_parts(QSP_ARG_DECL const char *class_str);
#define get_parts(class_str)		_get_parts(QSP_ARG  class_str)
extern Screen_Obj *_mk_menu(QSP_ARG_DECL  Screen_Obj *mip);
#define mk_menu(mip)			_mk_menu(QSP_ARG  mip)
extern void fix_names(QSP_ARG_DECL  Screen_Obj *mip,Screen_Obj *parent);
extern void push_parent(Screen_Obj *mp);
extern void _get_min_max_val(QSP_ARG_DECL  int *minp,int *maxp,int *valp);
extern void _get_so_width(QSP_ARG_DECL int *widthp);
extern int _get_strings(QSP_ARG_DECL  Screen_Obj *sop,const char ***sss);
extern void _mk_it_scroller(QSP_ARG_DECL  Screen_Obj *sop,Item_Type *itp);
extern Screen_Obj *_simple_object(QSP_ARG_DECL  const char *name);

#define get_min_max_val(minp,maxp,valp)		_get_min_max_val(QSP_ARG  minp,maxp,valp)
#define get_so_width(widthp)			_get_so_width(QSP_ARG  widthp)
#define get_strings(sop,sss)			_get_strings(QSP_ARG  sop,sss)
#define mk_it_scroller(sop,itp)			_mk_it_scroller(QSP_ARG  sop,itp)
#define simple_object(name) _simple_object(QSP_ARG  name)

extern COMMAND_FUNC(mk_panel);

// BUG most of these prototypes can go in the lib-local include file...
extern void make_menu(QSP_ARG_DECL  Screen_Obj *mp,Screen_Obj *mip);
extern void make_menu_choice(QSP_ARG_DECL  Screen_Obj *mip,Screen_Obj *parent);

extern void _reposition(QSP_ARG_DECL  Screen_Obj *sop);
#define reposition(sop) _reposition(QSP_ARG  sop)

#ifdef HAVE_MOTIF
extern void delete_motif_widget(Screen_Obj *sop);
#endif // HAVE_MOTIF
extern void _delete_widget(QSP_ARG_DECL  Screen_Obj *sop);
#define delete_widget(sop) _delete_widget(QSP_ARG  sop)

extern void make_pullright(QSP_ARG_DECL  Screen_Obj *mip,Screen_Obj *pr,Screen_Obj *parent);
extern void make_toggle(QSP_ARG_DECL  Screen_Obj *bo);
extern void make_text_field(QSP_ARG_DECL  Screen_Obj *to);
#ifdef BUILD_FOR_OBJC
extern void make_text_box(QSP_ARG_DECL  Screen_Obj *tb, BOOL is_editable);
extern void update_text_box(Screen_Obj *mp);
extern void make_act_ind(QSP_ARG_DECL  Screen_Obj *tb);
extern void set_activity_indicator(Screen_Obj *mp,int on_off);
#endif /* BUILD_FOR_OBJC */

extern void make_separator(QSP_ARG_DECL  Screen_Obj *so);
extern void make_edit_box(QSP_ARG_DECL  Screen_Obj *to);
extern void update_edit_text(Screen_Obj *sop, const char *string);
extern const char *get_text(Screen_Obj *to);
extern void make_gauge(QSP_ARG_DECL  Screen_Obj *go);
extern void make_slider(QSP_ARG_DECL  Screen_Obj *sop);
extern void make_slider_w(QSP_ARG_DECL  Screen_Obj *sop);
extern void set_toggle_state(Screen_Obj *sop,int pos);
//extern void set_choice(Screen_Obj *sop,int which);
extern void make_adjuster(QSP_ARG_DECL  Screen_Obj *sop);
extern void make_adjuster_w(QSP_ARG_DECL  Screen_Obj *sop);
extern void make_message(QSP_ARG_DECL  Screen_Obj *mp);
extern void set_gauge_value(Screen_Obj *gp,int n);
extern void set_gauge_label(Screen_Obj *gp,const char *s);
extern void update_gauge_label(Screen_Obj *gp);
extern void update_message(Screen_Obj *mp);
extern void update_label(Screen_Obj *mp);


extern void _set_scroller_list(QSP_ARG_DECL  Screen_Obj *sop,const char *string_list[],int nlist);
extern void _update_text_field(QSP_ARG_DECL  Screen_Obj *sop, const char *string);
extern void _new_slider_range(QSP_ARG_DECL  Screen_Obj *sop,int xmin,int xmax);
extern void _new_slider_pos(QSP_ARG_DECL  Screen_Obj *sop,int pos);
extern void _update_prompt(QSP_ARG_DECL  Screen_Obj *to);
extern IOS_Item_Context *_create_scrnobj_context(QSP_ARG_DECL  const char *name);
#define set_scroller_list(sop,string_list,nlist) _set_scroller_list(QSP_ARG  sop,string_list,nlist)
#define update_text_field(sop,string) _update_text_field(QSP_ARG  sop,string)
#define new_slider_range(sop,xmin,xmax) _new_slider_range(QSP_ARG  sop,xmin,xmax)
#define new_slider_pos(sop,pos) _new_slider_pos(QSP_ARG  sop,pos)
#define update_prompt(to) _update_prompt(QSP_ARG  to)
#define create_scrnobj_context(name) _create_scrnobj_context(QSP_ARG  name)

extern void get_device_dims(Screen_Obj *sop);
extern void del_so(QSP_ARG_DECL  Screen_Obj *sop);

extern void _make_label(QSP_ARG_DECL  Screen_Obj *sop);
extern void _make_scroller(QSP_ARG_DECL  Screen_Obj *sop);
extern void _make_chooser(QSP_ARG_DECL  Screen_Obj *sop,int n,const char **stringlist);
extern void _make_picker(QSP_ARG_DECL  Screen_Obj *sop);
extern void _make_button(QSP_ARG_DECL  Screen_Obj *bo);

#define make_label(sop) _make_label(QSP_ARG  sop)
#define make_scroller(sop) _make_scroller(QSP_ARG  sop)
#define make_chooser(sop,n,stringlist) _make_chooser(QSP_ARG  sop,n,stringlist)
#define make_picker(sop) _make_picker(QSP_ARG  sop)
#define make_button(bo) _make_button(QSP_ARG  bo)

extern void clear_all_selections(Screen_Obj *sop);

extern IOS_Item_Context *top_scrnobj_context(void);

//#define MAX_DEBUG

#ifdef MAX_DEBUG
extern void show_ctx_stack(void);
#endif // MAX_DEBUG

#endif /* ! _SCREEN_OBJ_H_ */

