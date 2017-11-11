#include "quip_config.h"

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_PWD_H
#include <pwd.h>
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_PARAM_H
#include <sys/param.h>	/* MAXPATHLEN */
#endif

#include "quip_prot.h"	
#include "query_stack.h"	
//#include "warn.h"	// ERROR_STRING

#ifdef HAVE_GETPWENT

#define MAX_PW_ENTS	200
static struct passwd *pw_tab[MAX_PW_ENTS];	/* BUG - how to get the proper size? */

static struct passwd **get_pw_tbl()
{
	int i=0;

	while( i < MAX_PW_ENTS ){
		pw_tab[i] = getpwent();
		if( pw_tab[i] == NULL ) i=MAX_PW_ENTS;
		i++;
	}
	if( i != MAX_PW_ENTS+1 ){
		sprintf(DEFAULT_ERROR_STRING,"Oops, need to increase MAX_PW_ENTS in tryhard.c");
		warn(NULL_QSP_ARG  DEFAULT_ERROR_STRING);
		pw_tab[MAX_PW_ENTS-1] = NULL;
	}
	return(pw_tab);
}

#endif /* HAVE_GETPWENT */		

/*
 * Open the named file with the given mode (see fopen(3)).
 * Prints a warning message if unable to open the file.
 * Returns a valid file pointer or NULL on error.
 */

FILE *_try_open(QSP_ARG_DECL  const char *filename, const char *mode)
			/* filename = name of file to open */
			/* mode = file mode (see fopen(3)) */
{
        FILE *fp;
#ifdef HAVE_STAT

#ifdef HAVE_GETPWENT
	static struct passwd **pw_tbl=NULL;
	struct passwd **pwp;
	const char *owner_name, *my_name;
#endif /* HAVE_GETPWENT */

	struct stat statb;
#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif /* ! MAXPATHLEN */
	char dirname[MAXPATHLEN];
	uid_t file_owner, my_uid;
	int i;
#endif /* HAVE_STAT */

	assert( filename[0] != 0 );

	/* fopen() will fail if the file has another owner,
	 * even if the directory is writable.  But in this case
	 * the old file can be removed.
	 */
	if( strcmp(mode,"w") )		/* mode not "w" ?? */
		goto proceed;

#ifdef HAVE_STAT
	/* stat the file and check the owner */
	if( stat(filename,&statb) < 0 ){
		/* We expect ENOENT if file doesn't exist */
		if( errno != ENOENT ){
			sprintf(ERROR_STRING,"stat %s (try_open)",filename);
			perror(ERROR_STRING);
		}
		goto proceed;
	}
	if( (file_owner=statb.st_uid) == (my_uid=geteuid()) ) /* uid matches? */
		goto proceed;

	/* Now we know that the uid does not match... */

	if( statb.st_mode & S_IWOTH )	/* other-write permission? */
		goto proceed;
    
	/* uid does not match, and no write permission...
	 * check and see if the directory is writable.
	 */
	if( strlen(filename) >= MAXPATHLEN ){
		// what is the type of strlen?  on debian seems to be size_t...
		// is size_t always long???
		sprintf(ERROR_STRING,
"try_open:  filename length (%ld) is greater than MAXPATHLEN (%d) !?",
			(long)strlen(filename),MAXPATHLEN);
		warn(ERROR_STRING);
		advise("Not checking directory permissions...");
		goto proceed;
	}
	strcpy(dirname,filename);
	i=(int)strlen(dirname)-1;
	while( i>=0 && dirname[i] != '/' )
		i--;
	if( i < 0 )
		strcpy(dirname,".");
	else
		dirname[i]=0;

	/* now stat the directory and see if we have write permission... */
	if( stat(dirname,&statb) < 0 ){
		sprintf(ERROR_STRING,"stat %s (try_open)",dirname);
		perror(ERROR_STRING);
	}

	if( statb.st_mode & S_IWOTH ){ /* directory writable? */
		/* We don't own the file, but the directory is writable.
		 * So we try to remove the file.
		 */
		if( unlink(filename) < 0 ){
			sprintf(ERROR_STRING,"unlink %s (try_open)",filename);
			perror(ERROR_STRING);
		}
		goto proceed;
	}
#ifdef HAVE_GETPWENT

	/* If we get to here, we know that the file already exists, but
	 * we don't own it, we don't have write permission, and we
	 * don't have directory write permissions, so we can't
	 * do anything!?
	 * We can just let the fopen fail and give us the normal error
	 * message, but here we print a more informative msg.
	 */
	owner_name=NULL;
	my_name=NULL;

	if( pw_tbl == NULL ) pw_tbl=get_pw_tbl();

	pwp=pw_tbl;
	while( *pwp != NULL ){
		if( (*pwp)->pw_uid == file_owner )
			owner_name=(*pwp)->pw_name;
		else if( (*pwp)->pw_uid == my_uid )
			my_name=(*pwp)->pw_name;
		pwp++;
	}

	if( owner_name != NULL && my_name != NULL ){
		sprintf(ERROR_STRING,
	"File %s is owned by %s, not writable by %s,",
			filename,owner_name,my_name);
		advise(ERROR_STRING);
		sprintf(ERROR_STRING,
	"and directory %s does not have other-write permission.",dirname);
		advise(ERROR_STRING);
	}
#endif /* HAVE_GETPWENT */

#endif /* HAVE_STAT */

proceed:


        fp=fopen(filename, mode);
        if( !fp ){
		if( strlen(filename) > (LLEN-32) ){
			// what is type of strlen?  size_t?  long?
			sprintf(ERROR_STRING,
		"Can't open file (filename too long - %ld chars), mode %s",
				(long)strlen(filename),mode);
			warn(ERROR_STRING);
		} else {
			sprintf(ERROR_STRING,"can't open file %s, mode %s",
				filename, mode);
			warn(ERROR_STRING);
		}
	}
        return(fp);
}

/*
 * Open the named file.
 * Treat any errors as fatal, exit program.
 */

FILE *_try_hard( QSP_ARG_DECL  const char *filename, const char *mode )
		/* filename = name of file to open */
		/* mode = file mode (see fopen(3)) */
{
        FILE *fp;
        fp=try_open(filename,mode);
        if( !fp ) error1("Missing file caused fatal error");
        return(fp);
}

static int cautious=1;

COMMAND_FUNC( togclobber )
{
	if( cautious ){
		cautious=0;
		warn("allowing file overwrites without explicit confirmation");
	} else {
		cautious=1;
		warn("requiring confirmation for file overwrites");
	}
}

FILE *_try_nice(QSP_ARG_DECL  const char *fnam, const char *mode)
{
        FILE *fp;
#ifdef HAVE_STAT
	char pstr[LLEN];
	struct stat statb;
#endif /* HAVE_STAT */
    
	if( fnam[0]==0 ){
		sprintf(ERROR_STRING,"null file name");
		warn(ERROR_STRING);
		return(NULL);
	}
#ifdef HAVE_STAT
	if( !strcmp(mode,"w") ){
		if( cautious && stat(fnam,&statb) != (-1) ){	/* file found */
			if( (strlen(fnam)+23+1) > LLEN ){
				// The filename is too long to print the prompt
                       		sprintf(pstr, "Overwrite existing file");
			} else {
                       		sprintf(pstr, "file %s exists, overwrite",fnam);
			}
                       	if( !confirm(pstr) ) return(NULL);
                }
        }
#endif /* HAVE_STAT */
    
        fp=try_open(fnam, mode);
	return(fp);
}

