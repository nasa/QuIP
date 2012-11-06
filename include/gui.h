
#ifndef NO_SCREEN_OBJ

#ifdef HAVE_MOTIF

/* gui.h */

/* #include "config.h" */

#include "node.h"
#include "data_obj.h"
#include "display.h"
#include "xsupp.h"
#include "query.h"

typedef struct panel_object {
	Dpyable		po_top;
	int		po_currx, po_curry;	/* for added items */
	int		object_item_type;
	Display *	po_dpy;
	Visual *	po_visual;
	int		po_screen_no;
	GC		po_gc;
	Item_Context *	po_icp;
} Panel_Obj;

#define po_cmap		po_top.c_cmap
#define po_cm_dp	po_top.c_cm_dp
#define po_lt_dp	po_top.c_lt_dp

#define po_dop		po_top.c_dop

#define po_name		po_top.c_name
#define po_width	po_top.c_width
#define po_height	po_top.c_height
#define po_dx		po_top.c_width
#define po_dy		po_top.c_height
#define po_x		po_top.c_x
#define po_y		po_top.c_y
#define po_xwin		po_top.c_xwin
#define po_flags	po_top.c_flags
#define po_children	po_top.c_children
/* #define po_dpy		po_top.c_dpy		*/
/* #define po_gc		po_top.c_gc		*/
/* #define po_screen_no	po_top.c_screen_no		*/
/* #define po_visual	po_top.c_visual		*/
#define po_frame_obj	po_top.c_frame_obj
#define po_panel_obj	po_top.c_thing_obj
#define po_pw		po_top.c_pw
#define po_realized     po_top.c_realized

/* panel flags */
#define PANEL_IS_SHOWABLE		1
#define PANEL_KNOWS_SYSTEM_COLORS	2
#define PANEL_SHOWN			4		/* "mapped" */

#define CANNOT_SHOW_PANEL( po )	(( ( po )->po_flags&PANEL_IS_SHOWABLE)==0)
#define CAN_SHOW_PANEL( po )	( ( po )->po_flags&PANEL_IS_SHOWABLE )
#define PANEL_MAPPED( po )		( (po)->po_flags & PANEL_SHOWN )
#define PANEL_UNMAPPED( po )		(( (po)->po_flags & PANEL_SHOWN )==0)

#define NO_PANEL_OBJ	((Panel_Obj *) NULL)

#define PANEL_HAS_SYSCOLS( po )		HAS_SYSCOLS(po->po_xldp)
#define PANEL_KNOWS_SYSCOLS( po )	(po->po_flags & PANEL_KNOWS_SYSTEM_COLORS)


typedef struct screen_object {
	Dpyable	so_top;
	union {
		const char **	u_selectors;
		const char *	u_selector;
	} so_sel_u;
#define so_selector	so_sel_u.u_selector
#define so_selectors	so_sel_u.u_selectors
	const char *		so_action_text;		/* text to interpret */
	const char *		so_content_text;	/* text to display */
	int		so_min,so_max;		/* for sliders & guages */
	int		so_val;			/* slider or guage value */
        int		so_width;		/* adjuster width */
	Panel_Obj *	so_menu;
	Panel_Obj *	so_panel;
	Widget		so_obj;
#ifdef THREAD_SAFE_QUERY
	Query_Stream *	so_qsp;
#endif
} Screen_Obj;

#ifdef THREAD_SAFE_QUERY
#define SO_QSP_ARG	sop->so_qsp,
#else
#define SO_QSP_ARG
#endif
/* lists don't need a value, so we do a union with this macro: */
#define so_nlist	so_val

#define WIDGET_PANEL(sop) ((Panel_Obj *)( sop )->so_parent)

#define so_xwin		so_top.c_xwin
#define so_name		so_top.c_name
#define so_x		so_top.c_x
#define so_y		so_top.c_y
#define so_dx		so_top.c_width
#define so_dy		so_top.c_height
#define so_flags	so_top.c_flags
#define so_parent	so_top.c_parent
#define so_children	so_top.c_children
#define so_frame	so_top.c_frame_obj

#define NO_SCREEN_OBJ	((Screen_Obj *)NULL)
#define NO_SO_PTR	((Screen_Obj **)NULL)

/* panel object types */
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
#define SO_CHOOSER	10		/* note that this duplicates scroller!? */
#define SO_TOGGLE	11

#define N_WIDGET_TYPES	11

#define OBJECT_TYPE_MASK	15

#define WIDGET_CODE(sop)	((sop)->so_flags & OBJECT_TYPE_MASK)
#define WIDGET_INDEX(sop)	(WIDGET_CODE(sop)-1)

#define IS_BUTTON(sop)		(WIDGET_CODE(sop)==SO_BUTTON)
#define IS_TEXT(sop)		(WIDGET_CODE(sop)==SO_TEXT)
#define IS_TOGGLE(sop)		(WIDGET_CODE(sop)==SO_TOGGLE)
#define IS_MENU_BUTTON(sop)	(WIDGET_CODE(sop)==SO_MENU_BUTTON)
#define IS_LIT(sop)		(sop->so_flags & SO_LIT)
#define IS_ACTIVE(sop)		(sop->so_flags & SO_ACTIVE)
#define LIGHT(bo)		( bo )->so_flags |= SO_LIT
#define UNLIGHT(bo)		( bo )->so_flags &= ~SO_LIT
#define ACTIVATE(bo)		( bo )->so_flags |= SO_ACTIVE
#define DEACTIVATE(bo)		( bo )->so_flags &= ~SO_ACTIVE

/* object flags */
#define SO_ACTIVE	16
#define SO_LIT		32


#define BUTTON_UP	1	/* event_flags(event) == 1 */
#define BUTTON_WIDTH	50
#define BUTTON_HEIGHT	19
#define TOGGLE_HEIGHT	19
#define OBJECT_GAP	5
#define GAP_HEIGHT	10
#define SLIDER_HEIGHT	75
#define SLIDER_WIDTH	100
#define GAUGE_HEIGHT	SLIDER_HEIGHT
#define GAUGE_WIDTH	78
#define MESSAGE_WIDTH	78
#define MESSAGE_HEIGHT	40

#define SHRINK_WHEN_LIT	2

#define MENU_WIDTH	50
#define MENU_LENGTH	100

#define SCROLLER_GAP		10
#define N_SCROLLER_LINES	8
#define GAP_PER_LINE		19
#define SCROLLER_HEIGHT		(SCROLLER_GAP+N_SCROLLER_LINES*GAP_PER_LINE)
#define SCROLLER_WIDTH 		300

#define CHOOSER_HEIGHT		22
#define CHOOSER_ITEM_HEIGHT     30

/* globals */

extern Panel_Obj *curr_panel;

/* prototypes */

ITEM_INTERFACE_PROTOTYPES(Panel_Obj,panel_obj)
ITEM_INTERFACE_PROTOTYPES(Screen_Obj,scrnobj)

#define PICK_PANEL(pmpt)	pick_panel_obj(QSP_ARG  pmpt)



/* screen_objs.c */

extern Screen_Obj *simple_object(QSP_ARG_DECL  const char *name);
extern void list_panels(void);
extern Screen_Obj *dup_so(QSP_ARG_DECL  Screen_Obj *sop);
extern void so_info(Screen_Obj *sop);
extern Panel_Obj *new_panel(QSP_ARG_DECL  const char *name,int dx,int dy);
extern Screen_Obj *get_parts(QSP_ARG_DECL const char *class_str);
#define GET_PARTS(class_str)		get_parts(QSP_ARG  class_str)
extern Screen_Obj *mk_menu(QSP_ARG_DECL  Screen_Obj *mip);
#define MK_MENU(mip)			mk_menu(QSP_ARG  mip)
extern void fix_names(QSP_ARG_DECL  Screen_Obj *mip,Screen_Obj *parent);
extern void push_parent(Screen_Obj *mp);
extern void get_min_max_val(QSP_ARG_DECL  int *minp,int *maxp,int *valp);
#define GET_MIN_MAX_VAL(minp,maxp,valp)		get_min_max_val(QSP_ARG  minp,maxp,valp)
extern void get_so_width(QSP_ARG_DECL int *widthp);
#define GET_SO_WIDTH(widthp)			get_so_width(QSP_ARG  widthp)
extern void so_init(QSP_ARG_DECL  int argc,char **argv);
extern List *panel_list(SINGLE_QSP_ARG_DECL);
extern int get_strings(QSP_ARG_DECL  Screen_Obj *sop,const char ***sss);
#define GET_STRINGS(sop,sss)		get_strings(QSP_ARG  sop,sss)
extern void mk_it_scroller(QSP_ARG_DECL  Screen_Obj *sop,Item_Type *itp);
extern Node *first_panel_node(SINGLE_QSP_ARG_DECL);
extern Screen_Obj *find_object_at(Panel_Obj *po,int x,int y);
extern double panel_exists(QSP_ARG_DECL  const char *);

/* protomenu.c */

#endif /* HAVE_MOTIF */

#endif /* ! NO_SCREEN_OBJ */


