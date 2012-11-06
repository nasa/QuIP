
/* prototypes for datamenu high level funcs */

#ifndef _DATAPROT_H_
#define _DATAPROT_H_

#include "data_obj.h"
#include "query.h"

#ifdef INC_VERSION
char VersionId_inc_dataprot[] = QUIP_VERSION_STRING;
#endif /* INC_VERSION */

#ifdef	MAC
#define DATA_CMD_WORD		"Data"
#else	/* !MAC */
#define DATA_CMD_WORD		"data"
#endif


/* ascmenu.c */
extern COMMAND_FUNC( asciimenu );


/* datamenu.c */
extern COMMAND_FUNC( do_area );

/* BUG these should probably be static ... */
extern int get_prec(void);

extern Data_Obj *req_obj(char *s);
extern void dataport_init(void);
extern void dm_init(SINGLE_QSP_ARG_DECL);


/* ops_menu.c */
extern COMMAND_FUNC( buf_ops );

/* verdatam.c */
extern void verdatam(SINGLE_QSP_ARG_DECL);



#endif /* _DATAPROT_H_ */

