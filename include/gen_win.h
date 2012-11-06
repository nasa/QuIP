
#ifndef _GEN_WIN_H_
#define _GEN_WIN_H_

#include "quip_config.h"

#include "items.h"
#include "function.h"
#include "viewer.h"
#include "gui.h"

extern void add_genwin(QSP_ARG_DECL  Item_Type *itp, Genwin_Functions *gwfp,
	Item *(*lookup)(QSP_ARG_DECL  const char *));
extern Item *find_genwin(QSP_ARG_DECL  const char *);
#define NO_GENWIN	((Item *) NULL)

#ifdef HAVE_X11
extern Viewer *genwin_viewer(QSP_ARG_DECL  Item *);
#endif

#ifdef HAVE_MOTIF
extern Panel_Obj *genwin_panel(QSP_ARG_DECL  Item *);
#endif /* HAVE_MOTIF */

#endif /* _GEN_WIN_H_ */

