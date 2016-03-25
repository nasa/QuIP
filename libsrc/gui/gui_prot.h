// prototypes for functions used just within the module

#ifndef _GUI_PROT_H_
#define _GUI_PROT_H_

#include "screen_obj.h"
#include "gui_cmds.h"

extern void set_choice(Screen_Obj *sop,int which);
extern void set_pick(Screen_Obj *sop, int cyl, int which );

extern void enable_widget(QSP_ARG_DECL  Screen_Obj *sop, int yesno );
extern void hide_widget(QSP_ARG_DECL  Screen_Obj *sop, int yesno );

// Not all of these should be ios only!?
#ifdef BUILD_FOR_OBJC

extern void dismiss_keyboard(Screen_Obj *sop);

extern Panel_Obj *console_po;

// ios.m
extern void make_console_panel(QSP_ARG_DECL  const char *name);
//extern void fatal_alert(QSP_ARG_DECL  const char *msg);
extern void get_confirmation(QSP_ARG_DECL  const char *title, const char *question);
extern void notify_busy(QSP_ARG_DECL  const char *title, const char *msg);
extern void end_busy(int final);

extern void check_first(Panel_Obj *po);

extern void reload_chooser(Screen_Obj *sop);
extern void reload_picker(Screen_Obj *sop);
#else /* ! BUILD_FOR_OBJC */

extern Item_Context *pop_scrnobj_context(SINGLE_QSP_ARG_DECL);
extern Item_Context *current_scrnobj_context(SINGLE_QSP_ARG_DECL);
extern void push_scrnobj_context(QSP_ARG_DECL  Item_Context *icp);
extern void give_notice(const char **msg_array);

#endif /* ! BUILD_FOR_OBJC */

extern void so_init(QSP_ARG_DECL  int argc,const char **argv);

//extern void push_navitm_context(QSP_ARG_DECL  IOS_Item_Context *icp);
//extern Item_Context * pop_navitm_context(SINGLE_QSP_ARG_DECL);
//extern void push_navgrp_context(QSP_ARG_DECL  IOS_Item_Context *icp);
//extern Item_Context * pop_navgrp_context(SINGLE_QSP_ARG_DECL);

#include "nav_panel.h"

extern void add_navitm_to_group(Nav_Group *ng_p, Nav_Item *ni_p);
extern void remove_nav_item(QSP_ARG_DECL  Nav_Item *ni_p);
extern void remove_nav_group(QSP_ARG_DECL  Nav_Group *ng_p);
extern void hide_nav_bar(QSP_ARG_DECL  int hide);
extern Nav_Panel *create_nav_panel(QSP_ARG_DECL  const char *s);
extern Nav_Group *create_nav_group(QSP_ARG_DECL  Nav_Panel *np_p, const char *s);
extern void simple_alert(QSP_ARG_DECL  const char *type, const char *msg);
extern void get_confirmation(QSP_ARG_DECL  const char *title, const char *question);
extern void notify_busy(QSP_ARG_DECL  const char *title, const char *msg);
extern void end_busy(int final);

/* protomenu.c */
extern void prepare_for_decoration( QSP_ARG_DECL  Panel_Obj *pnl_p );
extern void unprepare_for_decoration( SINGLE_QSP_ARG_DECL );


#endif /* _GUI_PROT_H_ */

