#include "quip_config.h"

char VersionId_interpreter_rcfile[] = QUIP_VERSION_STRING;

/*
 * rcfile(progname) tries to redirect to an appropriate startup file.
 *
 * We used to look for a file .progname - but too many hidden files!?
 * Now we look for progname.scr.
 *
 * We look first for $HOME/.quiprc - if that does not exist, we then
 * look for a global quip startup file (where?)
 *
 * Directories are searched in the following order:
 *	1) current directory
 *	2) $QUIPSTARTUPDIR
 *	3) $HOME/.quip/startup
 *	4) /usr/local/share/quip/macros/startup
 *
 * In cases 1,3 and 4 QUIPSTARTUPDIR is set within the program to the directory.
 */

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* for getenv(), getwd() */
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "query.h"
#include "pathnm.h"	/* strip_fullpath() */

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>		/* defines MAXPATHLEN */
#endif

#ifndef MAXPATHLEN
#define MAXPATHLEN	1024
#endif /* MAXPATHLEN */

/* BUG?  Should we let this directory be set in configure? */
#define QUIP_DEFAULT_DIR	"/usr/local/share/quip/macros/startup"

/* local prototypes */
static char *try_directory(QSP_ARG_DECL  const char *,const char *);
static char *try_cwd(QSP_ARG_DECL  char *);
static char *try_user_spec(QSP_ARG_DECL  char *progname);
static char *try_home(QSP_ARG_DECL  char *progname);

#define STARTUP_DIRNAME	"QUIPSTARTUPDIR"

#include "debug.h"		/* verbose */
#include "query.h"

static void read_global_startup(SINGLE_QSP_ARG_DECL)
{
	char *home;
	char filename[MAXPATHLEN];
	FILE *fp;

	home=getenv("HOME");

	if( home == NULL ){
		WARN("read_global_startup:  no HOME in environment");
		return;
	}

	sprintf(filename,"%s/.quiprc",home);	// BUG possible buffer overrun

	fp=fopen(filename,"r");

	if( fp!=NULL ) {
		push_input_file(QSP_ARG filename);
		redir(QSP_ARG fp);
		/* We might like to interpret commands here,
		 * but no menus have been pushed, so we can't!?
		 * Could perhaps use builtin menu only?
		 */
	}
}

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
		push_input_file(QSP_ARG filename);
		redir(QSP_ARG fp);
		if( *dir ) assign_var(QSP_ARG  STARTUP_DIRNAME,dir);
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

void rcfile( QSP_ARG_DECL char* progname )
{
	char *s=NULL;

	init_query_stream(THIS_QSP);		/* make sure query file stack initialized */
	set_progname(progname); 	/* this is for get_progfile */

	strip_fullpath(&progname);	/* strip leading components */

	s=try_cwd(QSP_ARG  progname);
	if( s == NULL ) s=try_user_spec(QSP_ARG  progname);
	if( s == NULL ) s=try_home(QSP_ARG  progname);
	if( s == NULL ) s=try_default(QSP_ARG  progname);

	if( s == NULL ){
		advise("No startup file found");
	} else if( verbose ){
		/* How would verbose ever be set here? Only by changing compilation default? */
		sprintf(ERROR_STRING,"Interpreting startup file %s",s);
		advise(ERROR_STRING);
	}

	/* Because these functions push the input but do not execute,
	 * this one is interpreted first, because it is pushed last.
	 * It would be better to execute right away, so that settings
	 * such as verbose and QUIPSTARTUPDIR could be put there and
	 * used here, but when this is executed no menus have been pushed...
	 * We could push the builtin menu?
	 */
	read_global_startup(SINGLE_QSP_ARG);

}



