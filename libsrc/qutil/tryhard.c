#include "quip_config.h"

char VersionId_qutil_tryhard[] = QUIP_VERSION_STRING;

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

#include "query.h"	// ERROR_STRING

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

		

/*
 * Open the named file with the given mode (see fopen(3)).
 * Prints a warning message if unable to open the file.
 * Returns a valid file pointer or NULL on error.
 */

FILE *try_open(QSP_ARG_DECL  const char *filename, const char *mode)
			/* filename = name of file to open */
			/* mode = file mode (see fopen(3)) */
{
        FILE *fp;
	static struct passwd **pw_tbl=NULL;
	struct passwd **pwp;
	struct stat statb;
	char dirname[MAXPATHLEN];
	uid_t file_owner, my_uid;
	char *owner_name, *my_name;
	int i;

#ifdef CAUTIOUS
	if( filename[0]==0 ){
		WARN("try_open:  null file name");
		return(NULL);
	}
#endif /* CAUTIOUS */

	/* fopen() will fail if the file has another owner,
	 * even if the directory is writable.  But in this case
	 * the old file can be removed.
	 */
	if( strcmp(mode,"w") )		/* mode not "w" ?? */
		goto proceed;

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
		sprintf(ERROR_STRING,"try_open:  filename length (%ld) is greater than MAXPATHLEN (%d) !?",strlen(filename),MAXPATHLEN);
		WARN(ERROR_STRING);
		ADVISE("Not checking directory permissions...");
		goto proceed;
	}
	strcpy(dirname,filename);
	i=strlen(dirname)-1;
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
		ADVISE(ERROR_STRING);
		sprintf(ERROR_STRING,
	"and directory %s does not have other-write permission.",dirname);
		ADVISE(ERROR_STRING);
	}

proceed:


        fp=fopen(filename, mode);
        if( !fp ){
		if( strlen(filename) > (LLEN-32) ){
			// what is type of strlen?  size_t?  long?
			sprintf(ERROR_STRING,"Can't open file (filename too long - %ld chars), mode %s",
				strlen(filename),mode);
			WARN(ERROR_STRING);
		} else {
			sprintf(ERROR_STRING,"can't open file %s, mode %s",
				filename, mode);
			WARN(ERROR_STRING);
		}
	}
        return(fp);
}

/*
 * Open the named file.
 * Treat any errors as fatal, exit program.
 */

FILE *try_hard( QSP_ARG_DECL  const char *filename, const char *mode )
		/* filename = name of file to open */
		/* mode = file mode (see fopen(3)) */
{
        FILE *fp;
        fp=try_open(QSP_ARG  filename,mode);
        if( !fp ) ERROR1("Missing file caused fatal error");
        return(fp);
}

