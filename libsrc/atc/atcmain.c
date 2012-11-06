

char VersionId_atc_atcmain[] = "%Z% $RCSfile: atcmain.c,v $ ver: $Revision: 1.24 $ $Date: 2007/12/27 16:16:02 $";

#include "rn.h"			/* scramble */
#include "myerror.h"		/* error1() */
#include "debug.h"		/* verbose */
#include "version.h"		/* verbose */
#include "query.h"
#include "vwmprot.h"		/* viewmenu() */
#include "nports.h"		/* portmenu() */
#include "dataprot.h"		/* datamenu() */
#include "genwin_menu.h"	/* genwin_menu() */
#ifdef HAVE_NUMREC
#include "nrmprot.h"		/* nrmenu() */
#endif /* HAVE_NUMREC */
#include "optmenu.h"		/* stepmenu() */
#include "knox.h"		/* knoxmenu() */
#include "sound.h"		/* soundmenu() */
#ifdef HAVE_POLHEMUS
#include "phmenu.h"	/* ph_menu() */
#endif /* HAVE_POLHEMUS */
#include "stc.h"
#include "nvf.h"		/* warmenu */

#ifdef INTERFACE
//extern void protomenu(VOID);
#include "gui.h"
#endif /* INTERFACE */


/* img_file.h creates a lot of conflicts w/ conflict.h */
/* BUG there should be another include file for external linkage... */
/* #include "img_file.h" */		/* fiomenu() */
extern COMMAND_FUNC( fiomenu );

#include "conflict.h"


#ifdef SGI_MOVIE
#include "cosmo.h"
#include "gmovie.h"
#endif /* SGI_MOVIE */


static Command main_ctbl[]={
{ "atc",	atc_menu,	"ATC submenu"				},
{ "view",	viewmenu,	"viewer submenu"			},
{ "genwin",	genwin_menu,	"generalized viewer/panel submenu"	},
{ "step",	stair_menu,	"trial sequencing submenu"		},
{ "lookit",	lookmenu,	"staircase data analysis submenu"	},
{ "data",	datamenu,	"data object submenu"			},
{ "fileio",	fiomenu,	"data file I/O submenu"			},
{ "ports",	portmenu,	"socket port submenu"			},
#ifdef SGI_MOVIE
{ "movie",	moviemenu,	"movie control submenu"			},
#endif /* SGI_MOVIE */
{ "war",	warmenu,	"vector processing submenu"		},
#ifdef HAVE_NUMREC
{ "numrec",	nrmenu,		"numerical recipes submenu"		},
#endif /* HAVE_NUMREC */
{ "stepit",	stepmenu,	"optimization submenu"			},
{ "knox",	knoxmenu,	"knox routing switcher submenu"		},
{ "sound",	soundmenu,	"sound submenu"				},
#ifdef HAVE_POLHEMUS
{ "polhemus",	ph_menu,	"polhemus submenu"			},
#endif /* HAVE_POLHEMUS */
#ifdef INTERFACE
{ "interface",	protomenu,	"user interface submenu"		},
#endif /* INTERFACE */
{ "quit",	popcmd,		"exit program"				},
{ NULL_COMMAND								}
};

int main(int argc, char *argv[])
{
	QSP_DECL

	INIT_QSP
	set_args(argc,argv);
	rcfile(QSP_ARG argv[0]);

	rninit();
	/* vl_init(); */		/* not sure why this has to be here... */

#ifdef SGI_MOVIE
	load_movie_module(&cosmo_movie_module);
#endif /* SGI_MOVIE */

	auto_version("ATC","VersionId_atc");

	PUSHCMD(main_ctbl,"atc_main");
	while(1) do_cmd(SINGLE_QSP_ARG);
}

