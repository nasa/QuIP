
#ifndef _DL_H_
#define _DL_H_

#include "quip_prot.h"

typedef struct display_list {
	char *		dl_name;
	int		dl_serial;
} Display_List;

ITEM_INTERFACE_PROTOTYPES(Display_List,dl)

#define list_dls(fp)	_list_dls(QSP_ARG  fp)
#define pick_dl(p)	_pick_dl(QSP_ARG  p)
#define del_dl(p)	_del_dl(QSP_ARG  p)
#define dl_of(s)	_dl_of(QSP_ARG  s)
#define new_dl(s)	_new_dl(QSP_ARG  s)

extern COMMAND_FUNC( do_new_dl );
extern COMMAND_FUNC( do_del_dl );
extern COMMAND_FUNC( do_info_dl );
extern COMMAND_FUNC( do_dump_dl );
extern COMMAND_FUNC( do_call_dl );
extern COMMAND_FUNC( do_end_dl );

extern void info_dl(QSP_ARG_DECL  Display_List *);
extern void dump_dl(Display_List *);
extern void _call_dl(QSP_ARG_DECL  Display_List *);
#define call_dl(dlp) _call_dl(QSP_ARG dlp)

extern void new_display_list(QSP_ARG_DECL  const char *name);
extern void _end_dl(SINGLE_QSP_ARG_DECL);
#define end_dl() _end_dl(SINGLE_QSP_ARG)

extern void delete_dl(QSP_ARG_DECL  Display_List *dlp);

extern double display_list_exists(QSP_ARG_DECL  const char *name);

#endif // ! _DL_H_

