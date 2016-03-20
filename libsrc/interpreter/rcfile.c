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
#define QUIP_DEFAULT_DIR	"/usr/local/share/coq/macros/startup"

#define STARTUP_DIRNAME	"QUIPSTARTUPDIR"

#include "debug.h"		/* verbose */
#include "query.h"

static int read_global_startup(SINGLE_QSP_ARG_DECL)
{
#ifdef BUILD_FOR_OBJC
#ifdef BUILD_FOR_IOS
	// this is called from the main thread at startup...
	return ios_read_global_startup(SINGLE_QSP_ARG);	// in .m file
#endif // BUILD_FOR_IOS
#ifdef BUILD_FOR_MACOS
	// this is called from the main thread at startup...
	return macos_read_global_startup(SINGLE_QSP_ARG);
#endif // BUILD_FOR_MACOS
#else /* ! BUILD_FOR_OBJC */

	char *home;
	char filename[MAXPATHLEN];
	FILE *fp;

	home=getenv("HOME");

	if( home == NULL ){
		WARN("read_global_startup:  no HOME in environment");
		return -1;
	}

	// In the IOS simulator, HOME expands to a very long and strange path...
	// There is an install script to copy the file, but this fails when
	// we copy the file to the device...

	sprintf(filename,"%s/.coqrc",home);	// BUG possible buffer overrun
	fp=fopen(filename,"r");

	if( fp!=NULL ) {
//		int lvl;

		/* We might like to interpret commands here,
		 * but no menus have been pushed, so we can't!?
		 * Could perhaps use builtin menu only?
		 */
		redir(QSP_ARG  fp, filename );

		// If the startup file contains widget creation commands,
		// they can't run before the appDelegate has started (ios).
		// Therefore, we defer execution of the startup commands
		// until later.

		return 0;
	}
	return -1;
#endif /* ! BUILD_FOR_OBJC */

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

static char *try_directory(QSP_ARG_DECL  const char *dir,const char* progname)
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
		redir(QSP_ARG  fp, filename );
		if( *dir ){
//sprintf(ERROR_STRING,"Setting %s to %s",STARTUP_DIRNAME,dir);
//advise(ERROR_STRING);
			// We should only set the variable here if
			// it doesn't already exist - vars defined
			// in the environment are reserved!
			Variable *vp;
			vp = VAR_OF(STARTUP_DIRNAME);
			if( vp == NO_VARIABLE ){
				assign_var(QSP_ARG  STARTUP_DIRNAME,dir);
			}
		}
		return(filename);
	} else {
		return(NULL);
	}
}

static char *try_cwd(QSP_ARG_DECL  char *progname)
{
	return( try_directory(QSP_ARG  ".",progname) );
}

static char *try_home(QSP_ARG_DECL  char *progname)	/* look for dotfile in user's home directory */
{
	char *home;

	home=getenv("HOME");

	if( home == NULL ){
		WARN("try_home:  no HOME in environment");
		return(NULL);
	}
	return( try_directory(QSP_ARG  home,progname) );
}

static char *try_user_spec(QSP_ARG_DECL  char *progname) /* look for dotfile in user-specified directory */
{
	char *dir;

	dir=getenv(STARTUP_DIRNAME);
	if( dir == NULL ) return(NULL);
	return( try_directory(QSP_ARG  dir,progname) );
}

static char *try_default(QSP_ARG_DECL  char *progname) /* look for dotfile in default system directory */
{
	return( try_directory(QSP_ARG  QUIP_DEFAULT_DIR,progname) );
}

#endif // BUILD_FOR_OBJC

void rcfile( Query_Stack *qsp, char* progname )
{
	char *s=NULL;
	int status;

	// Should we make sure that the qsp has already been initialized?
	set_progname(progname); 	/* this is for get_progfile */

#ifndef BUILD_FOR_OBJC
	strip_fullpath(&progname);	/* strip leading components */

	s=try_cwd(QSP_ARG  progname);
	if( s == NULL ) s=try_user_spec(QSP_ARG  progname);
	if( s == NULL ) s=try_home(QSP_ARG  progname);
	if( s == NULL ) s=try_default(QSP_ARG  progname);
#endif /* ! BUILD_FOR_OBJC */

	// We probably don't want to print this message if we are using the global startup...


	/* Because these functions push the input but do not execute,
	 * this one is interpreted first, because it is pushed last.
	 * It would be better to execute right away, so that settings
	 * such as verbose and QUIPSTARTUPDIR could be put there and
	 * used here, but when this is executed no menus have been pushed...
	 * We could push the builtin menu?
	 */
	status = read_global_startup(SINGLE_QSP_ARG);

	if( status < 0 && s == NULL ){
		advise("No startup file found");
	} else if( verbose ){
		/* How would verbose ever be set here? Only by changing compilation default? */
		if( status == 0 ) advise("Interpreting global startup file $HOME/.coqrc");
		if( s != NULL ){
			sprintf(ERROR_STRING,"Interpreting startup file %s",s);
			advise(ERROR_STRING);
		}
	}
}



