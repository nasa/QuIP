
char VersionId_vectree_vt_main[] = "$RCSfile$ $Revision$ $Date$";

#include "quip_config.h"

#include "dataprot.h"
#include "nports.h"
#include "nvf.h"
#include "vectree.h"
#include "query.h"	/* prototype for rcfile() */

#ifdef VIEWERS
#include "vwmprot.h"
#endif /* VIEWERS */

#ifdef INTERFACE
#include "gui.h"		/* protomenu() */
#include "genwin_menu.h"
#endif /* INTERFACE */

#include "mseq.h"
#include "img_file.h"	/* fiomenu() */
#include "rawvol.h"
#include "gmovie.h"	/* moviemenu */
#include "sound.h"	/* soundmenu */
#ifdef HAVE_W3C
#include "server.h"	/* svr_menu */
#endif /* HAVE_W3C */
#ifdef HAVE_LIBCURL
#include "server.h"	/* svr_menu */
#endif /* HAVE_LIBCURL */
#ifdef HAVE_NUMREC
#include "nrmprot.h"
#endif /* HAVE_NUMREC */

#include "stc.h"	/* salac_menu (staircases a la carte) */
#ifdef MAC
#include "mac_support.h"
#endif /* MAC */

#ifdef FONTMENU
#include "fonted.h"
#endif /* FONTMENU */

#ifdef PIXRECT
extern void pr_menu(VOID);
#endif /* PIXRECT */


#ifdef STEPIT
#include "optmenu.h"
#endif /* STEPIT */


#ifndef MAC

static COMMAND_FUNC( finish_vt )
{
	nice_exit(0);
}

Command wm_ctbl[]={
{ "veclib",	vl_menu,	"vector function submenu"		},
{ "war",	warmenu,	"old-style vector function submenu"	},
{ "compute",	warmenu,	"old-style vector function submenu"	},
{ "data",	datamenu,	"create and examine data objects"	},
{ "fileio",	fiomenu,	"file I/O submenu"			},
{ "ports",	portmenu,	"communication port submenu"		},
{ "rawvol",	rv_menu,	"raw disk volume submenu"		},
#ifdef FONTMENU
{ "fonts",	fontmenu,	"font operation submenu"		},
#endif /* FONTMENU */
#ifdef PIXRECT
{ "pixrect",	pr_menu,	"pixrect operation submenu"		},
#endif /* PIXRECT */
#ifdef INTERFACE
{ "interface",	protomenu,	"user interface submenu"		},
{ "genwin",	genwin_menu,	"viewer/panel submenu"			},
#endif /* INTERFACE */
#ifdef STEPIT
{ "stepit",	stepmenu,	"stepit submenu"			},
#endif /* STEPIT */
{ "expressions",do_exprs,	"vector expression submenu"		},
#ifdef VIEWERS
{ "view",	viewmenu,	"image viewer submenu"			},
#endif /* VIEWERS */

#ifdef HAVE_NUMREC
{ "numrec",	nrmenu,		"numerical recipes submenu"		},
#endif /* HAVE_NUMREC */

#ifdef HAVE_W3C
{ "http",	svr_menu,	"http server submenu"			},
#endif /* HAVE_W3C */

#ifdef HAVE_LIBCURL
{ "http",	svr_menu,	"http server submenu"			},
#endif /* HAVE_LIBCURL */

{ "movie",	moviemenu,	"movie submenu"				},
#ifdef HAVE_SOUND
{ "sound",	soundmenu,	"sound submenu"				},
#endif

{ "mseq",	mseq_menu,	"M-sequence submenu"			},
{ "vectest",	vl_menu,	"vector library test menu"		},

{ "staircases",	salac_menu,	"staircase submenu"			},

{ "quit",	finish_vt,	"exit program"				},
{ NULL_COMMAND								}
};

#endif /* ! MAC */


int main(ac,av)
char **av;
{
	QSP_DECL

	INIT_QSP

#ifdef MAC
	jbm_init(OPEN_CONSOLE);
	rcfile(QSP_ARG  "wartest");
#else /* ! MAC */
	rcfile(QSP_ARG  av[0]);
	set_args(ac,av);
#endif /* ! MAC */

	/* dm_init(); */

#ifdef MAC
	warmenu();
	datamenu();
#else
	PUSHCMD(wm_ctbl,"top_level");
#endif

	check_suid_root();

	while(1) do_cmd(SINGLE_QSP_ARG);
	return(0);
}

