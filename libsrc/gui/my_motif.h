#include "quip_config.h"

#ifdef HAVE_MOTIF

#include "screen_obj.h"
#include "nav_panel.h"
#include "gen_win.h"

extern COMMAND_FUNC( do_dispatch );
extern void _motif_init(QSP_ARG_DECL  const char *progname);
extern Panel_Obj *find_panel(QSP_ARG_DECL  Widget obj);
extern void panel_repaint(Widget panel,Widget pw);
extern void post_menu_handler (Widget w, XtPointer client_data,
			XButtonPressedEvent *event);

#define motif_init(progname) _motif_init(QSP_ARG  progname)

#endif /* HAVE_MOTIF */

extern void set_std_cursor(void);
extern void set_busy_cursor(void);

