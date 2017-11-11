// prototypes for functions used just within the module

#ifndef _GUI_PROT_H_
#define _GUI_PROT_H_

#include "screen_obj.h"
#include "gui_cmds.h"

extern void _set_pick(QSP_ARG_DECL  Screen_Obj *sop, int cyl, int which );
extern void _set_choice(QSP_ARG_DECL  Screen_Obj *sop,int which);
extern void _get_choice(QSP_ARG_DECL  Screen_Obj *sop);
#define set_pick(sop, cyl, which ) _set_pick(QSP_ARG  sop, cyl, which )
#define set_choice(sop,which) _set_choice(QSP_ARG  sop,which)
#define get_choice(sop) _get_choice(QSP_ARG  sop)

extern void enable_widget(QSP_ARG_DECL  Screen_Obj *sop, int yesno );
extern void hide_widget(QSP_ARG_DECL  Screen_Obj *sop, int yesno );

extern void set_allowed_orientations(Quip_Allowed_Orientations o);

// Not all of these should be ios only!?
#ifdef BUILD_FOR_OBJC

extern void dismiss_keyboard(Screen_Obj *sop);

extern Panel_Obj *console_po;

// ios.m
extern void make_console_panel(QSP_ARG_DECL  const char *name);
//extern void fatal_alert(QSP_ARG_DECL  const char *msg);
extern void get_confirmation(QSP_ARG_DECL  const char *title, const char *question);
extern void notify_busy(QSP_ARG_DECL  const char *title, const char *msg);

extern void check_first(Panel_Obj *po);

extern void reload_chooser(Screen_Obj *sop);
extern void reload_picker(Screen_Obj *sop);
#else /* ! BUILD_FOR_OBJC */

extern void give_notice(const char **msg_array);

extern Item_Context *_pop_scrnobj_context(SINGLE_QSP_ARG_DECL);
extern Item_Context *_current_scrnobj_context(SINGLE_QSP_ARG_DECL);
extern void _push_scrnobj_context(QSP_ARG_DECL  Item_Context *icp);

#define pop_scrnobj_context() _pop_scrnobj_context(SINGLE_QSP_ARG)
#define current_scrnobj_context() _current_scrnobj_context(SINGLE_QSP_ARG)
#define push_scrnobj_context(icp) _push_scrnobj_context(QSP_ARG  icp)

#endif /* ! BUILD_FOR_OBJC */

extern void _so_init(QSP_ARG_DECL  int argc,const char **argv);
#define so_init(argc,argv) _so_init(QSP_ARG  argc,argv)

//extern void push_navitm_context(QSP_ARG_DECL  IOS_Item_Context *icp);
//extern Item_Context * pop_navitm_context(SINGLE_QSP_ARG_DECL);
//extern void push_navgrp_context(QSP_ARG_DECL  IOS_Item_Context *icp);
//extern Item_Context * pop_navgrp_context(SINGLE_QSP_ARG_DECL);

#include "nav_panel.h"

extern void add_navitm_to_group(Nav_Group *ng_p, Nav_Item *ni_p);
// in nav_panel.h??
//extern void remove_nav_item(QSP_ARG_DECL  Nav_Item *ni_p);
//extern void remove_nav_group(QSP_ARG_DECL  Nav_Group *ng_p);
//extern void hide_nav_bar(QSP_ARG_DECL  int hide);
//extern Nav_Panel *create_nav_panel(QSP_ARG_DECL  const char *s);
//extern Nav_Group *_create_nav_group(QSP_ARG_DECL  Nav_Panel *np_p, const char *s);
extern void _simple_alert(QSP_ARG_DECL  const char *type, const char *msg);
extern void _get_confirmation(QSP_ARG_DECL  const char *title, const char *question);
extern void _notify_busy(QSP_ARG_DECL  const char *title, const char *msg);
extern void _end_busy(QSP_ARG_DECL  int final);

#define simple_alert(type,msg) _simple_alert(QSP_ARG  type,msg)
#define get_confirmation(title, question) _get_confirmation(QSP_ARG  title, question)
#define notify_busy(title,msg) _notify_busy(QSP_ARG  title,msg)
#define end_busy(final) _end_busy(QSP_ARG  final)

/* protomenu.c */
extern void _prepare_for_decoration( QSP_ARG_DECL  Panel_Obj *pnl_p );
extern void _unprepare_for_decoration( SINGLE_QSP_ARG_DECL );

#define prepare_for_decoration(pnl_p) _prepare_for_decoration( QSP_ARG  pnl_p )
#define unprepare_for_decoration() _unprepare_for_decoration( SINGLE_QSP_ARG )

#endif /* _GUI_PROT_H_ */

