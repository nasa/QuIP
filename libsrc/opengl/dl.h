
#ifndef _DL_H_
#define _DL_H_

#include "quip_prot.h"

typedef struct display_list {
	char *		dl_name;
	int		dl_serial;
} Display_List;

ITEM_INTERFACE_PROTOTYPES(Display_List,dl)

extern COMMAND_FUNC( do_new_dl );
extern COMMAND_FUNC( do_del_dl );
extern COMMAND_FUNC( do_info_dl );
extern COMMAND_FUNC( do_dump_dl );
extern COMMAND_FUNC( do_call_dl );
extern COMMAND_FUNC( do_end_dl );

extern void info_dl(QSP_ARG_DECL  Display_List *);
extern void dump_dl(Display_List *);
extern void call_dl(Display_List *);

extern void new_display_list(QSP_ARG_DECL  const char *name);
extern void end_dl(void);
extern void delete_dl(QSP_ARG_DECL  Display_List *dlp);

extern double display_list_exists(QSP_ARG_DECL  const char *name);

#endif // ! _DL_H_

