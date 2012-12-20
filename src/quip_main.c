
#include "quip_config.h"

char VersionId_src_quip_main[] = QUIP_VERSION_STRING;

#include "query.h"	/* prototype for rcfile() */
#include "submenus.h"
#include "smarteye_api.h"

static COMMAND_FUNC( finish_vt )
{
	nice_exit(0);
}

static void missing_menu(QSP_ARG_DECL const char *msg)
{
	if( intractive(SINGLE_QSP_ARG) ){
		WARN(msg);
	} else {
		WARN(msg);
		advise("Exiting.");
		nice_exit(1);
	}
}

#ifndef HAVE_SOUND

#define NO_SOUND_MSG	"Unable to enter sound submenu.\n\nThis means that either an appropriate sound subsystem (e.g. ALSA on Linux)\nwas not found when the program was built, or that sound support was\nexplicitly disabled by passing the --disable-sound argument to the configure script.\n"

COMMAND_FUNC(soundmenu)
{
	missing_menu(QSP_ARG NO_SOUND_MSG);
}

#endif /* ! HAVE_SOUND */

#ifndef STEPIT

#define NO_STEPIT_MSG	"Unable to enter stepit submenu.\n\nThis means that stepit support was explicitly disabled by passing the --disable-stepit argument to the configure script."

static COMMAND_FUNC(soundmenu)
{
	missing_menu(QSP_ARG NO_STEPIT_MSG);
}

#endif /* ! STEPIT */

Command quip_ctbl[]={
{ "data",	datamenu,	"create and examine data objects"	},
{ "expressions",do_exprs,	"vector expression submenu"		},
{ "fileio",	fiomenu,	"file I/O submenu"			},
{ "ports",	portmenu,	"communication port submenu"		},
/* BUG - why synonyms here??? */
{ "veclib",	vl_menu,	"vector function submenu"		},
{ "vectest",	vl_menu,	"vector library test menu"		},
{ "compute",	warmenu,	"old-style vector function submenu"	},
{ "rawvol",	rv_menu,	"raw disk volume submenu"		},
{ "movie",	moviemenu,	"movie submenu"				},
{ "mseq",	mseq_menu,	"M-sequence submenu"			},
{ "staircases",	salac_menu,	"staircase submenu"			},
#ifdef HAVE_PIC
{ "pic",	picmenu,	"Microchip PIC submenu"			},
#endif /* HAVE_PIC */

{ "genwin",	genwin_menu,	"viewer/panel submenu"			},

#ifdef INTERFACE
{ "interface",	protomenu,	"user interface submenu"		},
#endif /* INTERFACE */

#ifdef STEPIT
{ "stepit",	stepmenu,	"stepit submenu"			},
#endif /* STEPIT */

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

{ "sound",	soundmenu,	"sound submenu"				},

/* This stuff is super-old and obsolete... */
#ifdef FONTMENU
{ "fonts",	fontmenu,	"font operation submenu"		},
#endif /* FONTMENU */
#ifdef PIXRECT
{ "pixrect",	pr_menu,	"pixrect operation submenu"		},
#endif /* PIXRECT */

#ifdef HAVE_V4L2
{ "v4l2",	v4l2_menu,	"V4L2 submenu"				},
#endif /* HAVE_V4L2 */

{ "requantize",	do_requant,	"Dithering submenu"			},
{ "knox",	knoxmenu,	"Knox video RS8x8HB routing switcher"	},
{ "gps",	gps_menu,	"Communicate w/ GPS receiver"		},

#ifdef HAVE_VISCA
{ "visca",	visca_menu,	"Sony VISCA camera control protocol"	},
#endif

#ifdef HAVE_METEOR
{ "meteor",	meteor_menu,	"Matrox Meteor I frame grabber"		},
#endif

#ifdef HAVE_OPENCV
{ "opencv",	ocv_menu,	"OpenCV submenu"			},
#endif

#ifdef HAVE_CUDA
{ "cuda",	cuda_menu,	"nVidia CUDA submenu"			},
#endif

#ifdef HAVE_LIBDV
{ "dv",		dv_menu,	"ieee1394 camera submenu"		},
#endif

#ifdef HAVE_OPENGL
{ "gl",		gl_menu,	"OpenGL submenu"			},
#endif /* HAVE_OPENGL */

#ifdef HAVE_GSL
{ "gsl",	gsl_menu,	"GNU scientific library submenu"	},
#endif /* HAVE_GSL */

#ifdef HAVE_DAS1602
{ "aio",	aio_menu,	"Analog I/O submenu"			},
#endif /* HAVE_DAS1602 */

/* BUG - these should be ifdef'd */

{ "pgr",	pgr_menu,	"PGR camera submenu"			},

#ifdef HAVE_X11
{ "atc",	atc_menu,	"ATC submenu"				},
#endif /* HAVE_X11 */

{ "smarteye",	smarteye_menu,	"SmartEye submenu"			},

{ "quit",	finish_vt,	"exit program"				},
{ NULL_COMMAND								}
};


int main(int ac,char **av)
{
	QSP_DECL

	INIT_QSP

	rcfile(QSP_ARG  av[0]);
	set_args(QSP_ARG  ac,av);
	PUSHCMD(quip_ctbl,"quip");

	check_suid_root();

	while(1) do_cmd(SINGLE_QSP_ARG);
	return(0);
}

