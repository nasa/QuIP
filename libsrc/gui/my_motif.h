#include "quip_config.h"

#ifdef HAVE_MOTIF

extern Item_Type *scrnobj_itp;
ITEM_INTERFACE_PROTOTYPES( Screen_Obj, scrnobj )

/* prototypes */

/* motif.c */

extern void update_edit_text(Screen_Obj *sop, const char *string);
extern void update_text_field(Screen_Obj *sop, const char *string);
extern void label_panel(Panel_Obj *po, const char *s);
extern void reposition(Screen_Obj *sop);
extern Panel_Obj *find_panel(QSP_ARG_DECL  Widget obj);
extern void panel_repaint(Widget panel,Widget pw);
extern void panel_cmap(Panel_Obj *po, Data_Obj *cm_dp);
extern void make_panel(QSP_ARG_DECL  Panel_Obj *po);
extern void post_menu_handler (Widget w, XtPointer client_data,
			XButtonPressedEvent *event);
extern void make_menu(QSP_ARG_DECL  Screen_Obj *mp,Screen_Obj *mip);
extern void make_menu_button(QSP_ARG_DECL  Screen_Obj *bp,Screen_Obj *mp);
extern void make_menu_choice(QSP_ARG_DECL  Screen_Obj *mip,Screen_Obj *parent);
extern void make_pullright(QSP_ARG_DECL  Screen_Obj *mip,Screen_Obj *pr,Screen_Obj *parent);
extern void button_func(Widget buttonID, XtPointer app_data, XtPointer widget_data);
extern void toggle_func(Widget buttonID, XtPointer app_data, XtPointer widget_data);
extern void make_separator(QSP_ARG_DECL  Screen_Obj *so);
extern void make_button(QSP_ARG_DECL  Screen_Obj *bo);
extern void make_toggle(QSP_ARG_DECL  Screen_Obj *bo);
extern void make_edit_box(QSP_ARG_DECL  Screen_Obj *to);
extern void make_text_field(QSP_ARG_DECL  Screen_Obj *to);
extern void update_prompt(Screen_Obj *to);
extern char *get_text(Screen_Obj *to);
extern void gauge_func(Widget item, XEvent *event);
extern void make_gauge(QSP_ARG_DECL  Screen_Obj *go);
extern void slider_func(Widget sliderID, XtPointer app_data, XtPointer widget_data);
extern void make_slider(QSP_ARG_DECL  Screen_Obj *sop);
extern void make_slider_w(QSP_ARG_DECL  Screen_Obj *sop);
extern void new_slider_range(Screen_Obj *sop,int xmin,int xmax);
extern void new_slider_pos(Screen_Obj *sop,int pos);
extern void new_toggle_state(Screen_Obj *sop,int pos);
extern void adjuster_func(Widget adjID, XtPointer app_data, XtPointer widget_data);
extern void make_adjuster(QSP_ARG_DECL  Screen_Obj *sop);
extern void make_adjuster_w(QSP_ARG_DECL  Screen_Obj *sop);
extern void make_message(QSP_ARG_DECL  Screen_Obj *mp);
extern COMMAND_FUNC( do_dispatch );
extern void motif_init(const char *progname);
extern void setting_func(Widget item, XEvent *event);
extern void set_gauge(Screen_Obj *gp,int n);
extern void update_message(Screen_Obj *mp);
extern void show_panel(Panel_Obj *po);
extern void unshow_panel(Panel_Obj *po);
extern void posn_panel(Panel_Obj *po);
extern void free_wsys_stuff(Panel_Obj *po);
extern void give_notice(const char **msg_array);
extern void init_cursors(void);
extern void set_win_cursors(Widget which_cursor);
extern void set_std_cursor(void);
extern void set_busy_cursor(void);
extern void make_scroller(QSP_ARG_DECL  Screen_Obj *sop);
extern void set_scroller_list(Screen_Obj *sop,const char *string_list[],int nlist);
extern void make_chooser(QSP_ARG_DECL  Screen_Obj *sop,int n,const char **stringlist);
extern void window_cm(Panel_Obj *po,Data_Obj *cm_dp);

#endif /* HAVE_MOTIF */

