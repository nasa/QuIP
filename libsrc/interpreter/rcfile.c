#include "quip_config.h"

/*
 * rcfile(progname) tries to redirect to an appropriate startup file.
 *
 * We used to look for a file .progname - but too many hidden files!?
 * Now we look for progname.scr.
 *
 * (BUG?  is this comment still valid?)
 * We look first for $HOME/.quiprc - if that does not exist, we then
 * look for a global quip startup file (where?)
 * But for testing and comparing to the old quip,
 * we call this file $HOME/.coqrc instead...
 * UPDATE:  use {progname}rc, no hard-coded references to coq or quip
 *
 * Directories are searched in the following order:
 *	1) current directory
 *	2) $QUIPSTARTUPDIR
 *	3) $HOME/.quip/startup
 *	4) /usr/local/share/quip/macros/startup
 *
 * In cases 1,3 and 4 QUIPSTARTUPDIR is set within the program to the directory.
 *
 * What about for IOS?
 */

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* for getenv(), getwd() */
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "query_prot.h"
#include "warn.h"
//#include "pathnm.h"	/* strip_fullpath() */

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>		/* defines MAXPATHLEN */
#endif

#ifndef MAXPATHLEN
#define MAXPATHLEN	1024
#endif /* MAXPATHLEN */

/* BUG?  Should we let this directory be set in configure? */
#define QUIP_DEFAULT_DIR	"/usr/local/share/quip/macros/startup"

#define STARTUP_DIRNAME	"QUIPSTARTUPDIR"

#include "debug.h"		/* verbose */

#ifdef BUILD_FOR_CMD_LINE

static int read_traditional_startup(QSP_ARG_DECL  const char *progname)
{
	char *home;
	char filename[MAXPATHLEN];
	FILE *fp;

	home=getenv("HOME");

	if( home == NULL ){
		warn("read_traditional_startup:  no HOME in environment");
		return -1;
	}

	// In the IOS simulator, HOME expands to a very long and strange path...
	// There is an install script to copy the file, but this fails when
	// we copy the file to the device...

	// BUG possible buffer overrun
	sprintf(filename,"%s/.%src",home,progname);	// e.g. .quiprc
	fp=fopen(filename,"r");

	if( fp!=NULL ) {
		if( verbose ){
			sprintf(ERROR_STRING,
	"Interpreting global startup file %s",filename);
			advise(ERROR_STRING);
		}

		/* We might like to interpret commands here,
		 * but no menus have been pushed, so we can't!?
		 * Could perhaps use builtin menu only?
		 */
		redir(fp, filename );

		// If the startup file contains widget creation commands,
		// they can't run before the appDelegate has started (ios).
		// Therefore, we defer execution of the startup commands
		// until later.

		return 0;
	}
	return -1;
}
#endif // BUILD_FOR_CMD_LINE

static int read_global_startup(QSP_ARG_DECL const char *progname)
{
#ifdef BUILD_FOR_CMD_LINE

fprintf(stderr,"building for command line...\n");
	return read_traditional_startup(QSP_ARG  progname);

#else // ! BUILD_FOR_CMD_LINE

#ifdef BUILD_FOR_IOS
	// this is called from the main thread at startup...
	return ios_read_global_startup(SINGLE_QSP_ARG);	// in .m file
#endif // BUILD_FOR_IOS

#ifdef BUILD_FOR_MACOS_APP
fprintf(stderr,"building Mac OSX app...\n");
	// We do this only when building a cocoa app, but not
	// for the native Mac command line version...
	//
	// this is called from the main thread at startup...
	return macos_read_global_startup(SINGLE_QSP_ARG);
#endif // BUILD_FOR_MACOS_APP

#endif /* ! BUILD_FOR_CMD_LINE */

}

#ifndef BUILD_FOR_OBJC

/*
 * Contruct the name of the startup file based on the program name.
 *
 * Look for this file in the named directory.  Return the name
 * of the file if found, or NULL if not.
 */

/* I used #elif here, but the mac choked on it  - jbm */
#define DIR_DELIM	"/"

#define try_directory(dir,progname) _try_directory(QSP_ARG  dir,progname)

static char *_try_directory(QSP_ARG_DECL  const char *dir,const char* progname)
{
	FILE *fp;
	static char filename[MAXPATHLEN];

	if( *dir ){
		strcpy(filename,dir);
		strcat(filename,DIR_DELIM);
	} else strcpy(filename,"");

	strcat(filename,progname);
	strcat(filename,".scr");

	fp=fopen(filename,"r");

	if( fp!=NULL ) {
		redir(fp, filename );
		if( *dir ){
			// We should only set the variable here if
			// it doesn't already exist - vars defined
			// in the environment are reserved!
			Variable *vp;
			vp = var_of(STARTUP_DIRNAME);
			if( vp == NULL ){
				assign_var(STARTUP_DIRNAME,dir);
			}
		}
		return(filename);
	} else {
		return(NULL);
	}
}

static char *try_cwd(QSP_ARG_DECL  char *progname)
{
	return( try_directory(".",progname) );
}

static char *try_home(QSP_ARG_DECL  char *progname)	/* look for dotfile in user's home directory */
{
	char *home;

	home=getenv("HOME");

	if( home == NULL ){
		warn("try_home:  no HOME in environment");
		return(NULL);
	}
	return( try_directory(home,progname) );
}

static char *try_user_spec(QSP_ARG_DECL  char *progname) /* look for dotfile in user-specified directory */
{
	char *dir;

	dir=getenv(STARTUP_DIRNAME);
	if( dir == NULL ) return(NULL);
	return( try_directory(dir,progname) );
}

static char *try_default(QSP_ARG_DECL  char *progname) /* look for dotfile in default system directory */
{
	return( try_directory(QUIP_DEFAULT_DIR,progname) );
}

#endif // BUILD_FOR_OBJC

void rcfile( Query_Stack *qsp, char* progname )
{
	char *s=NULL;
	int status;

	// Should we make sure that the qsp has already been initialized?
	set_progname(progname); 	/* this is for get_progfile */

#ifndef BUILD_FOR_OBJC
	// For unix, the user can put their own startup in:
	// current directory, $STARTUP_DIRNAME, $HOME, and QUIP_DEFAULT_DIR (/usr/local/share/quip/macros/startup/)
	strip_fullpath(&progname);	/* strip leading components */

	s=try_cwd(QSP_ARG  progname);
	if( s == NULL ) s=try_user_spec(QSP_ARG  progname);
	if( s == NULL ) s=try_home(QSP_ARG  progname);
	if( s == NULL ) s=try_default(QSP_ARG  progname);
#endif /* ! BUILD_FOR_OBJC */

	/* Because these functions push the input but do not execute,
	 * this one is interpreted first, because it is pushed last.
	 * It would be better to execute right away, so that settings
	 * such as verbose and QUIPSTARTUPDIR could be put there and
	 * used here, but, when this is executed, no menus have been pushed...
	 * We could push the builtin menu?
	 */
	status = read_global_startup(QSP_ARG  progname);

	if( status < 0 && s == NULL ){
		advise("No startup file found");
	} else if( verbose ){
		/* How would verbose ever be set here? Only by changing compilation default? */
		// We may not want to print this message if we are using the global startup?
		if( s != NULL ){
			sprintf(ERROR_STRING,"Interpreting startup file %s",s);
			advise(ERROR_STRING);
		}
	}
}



