char VersionId_datamenu_datamain[] = "$RCSfile$ $Revision$ $Date$";

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include "debug.h"
#include "nports.h"
#include "query.h"
#include "version.h"
#include "dataprot.h"
#include "menuname.h"
#include "img_file.h"	/* fiomenu */

#ifndef MAC
Command dmctbl[]={
{ DATA_CMD_WORD,	datamenu,	"data object submenu"		},
{ "ports",		portmenu,	"communication port submenu"	},
#ifndef PC
{ "fileio",		fiomenu,	"file I/O submenu"		},
#endif /* !PC */
{ "quit",		popcmd,		"quit program"			},
{ NULL_COMMAND								}
};
#else
#include "mac_support.h"
#endif /* !MAC */

int main(int ac, char **av)
{
	QSP_DECL

	INIT_QSP

#ifdef MAC

	jbm_init(OPEN_CONSOLE);
	rcfile(QSP_ARG  "data");
	datamenu();

#else	/* ! MAC */

	rcfile(QSP_ARG  av[0]);
	set_args(ac,av);
	PUSHCMD(dmctbl,DATATEST_MENU_NAME);

#endif	/* ! MAC */

	while(1) do_cmd(SINGLE_QSP_ARG);
}
