
#include "quip_config.h"

char VersionId_qutil_fileck[] = QUIP_VERSION_STRING;

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>		/* for stat(2) */
#endif

#include <errno.h>


#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* unlink() */
#endif

#include "query.h"
#include "fileck.h"

int path_exists(const char *pathname)
{
	struct stat statb;

	if( stat(pathname,&statb) < 0 ){
		if( verbose || errno != ENOENT ) tell_sys_error(pathname);
		return(0);
	}
	if( S_ISDIR(statb.st_mode) || S_ISREG(statb.st_mode) || S_ISCHR(statb.st_mode) ){
/*
sprintf(ERROR_STRING,"path_exists(%s) = 1",pathname);
advise(ERROR_STRING);
*/
		return(1);
	}

/*
sprintf(ERROR_STRING,"path_exists(%s) = 0",pathname);
advise(ERROR_STRING);
*/
	return(0);
}

int directory_exists(const char *dirname)
{
	struct stat statb;

	if( stat(dirname,&statb) < 0 ){
		if( verbose ) tell_sys_error(dirname);
		return(0);
	}

	if( ! S_ISDIR(statb.st_mode) ){
		sprintf(DEFAULT_ERROR_STRING,"%s is not a directory!?",dirname);
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	return(1);
}

int regfile_exists(const char *pathname)
{
	struct stat statb;

	if( stat(pathname,&statb) < 0 ){
		if( verbose ) tell_sys_error(pathname);
		return(0);
	}

	if( ! S_ISREG(statb.st_mode) ){
		sprintf(DEFAULT_ERROR_STRING,"%s is not a regular file!?",pathname);
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}
	return(1);
}

int can_write_to(const char *name)
{
	struct stat statb;

	if( stat(name,&statb) < 0 ){
		if( verbose ) tell_sys_error(name);
		return(0);
	}

	/* make sure we have permission to write here */

	/* are we the owner of this file/directory? */

	if( statb.st_uid == getuid() ||
		statb.st_uid == geteuid() ){

		/* we are owner or effective owner */

		if( (statb.st_mode & S_IWUSR) == 0 ){
			sprintf(DEFAULT_ERROR_STRING,
				"No owner write-permission on %s", name);
			NWARN(DEFAULT_ERROR_STRING);
			return(0);
		}
	} else if( (statb.st_mode & S_IWOTH) == 0 ){
		sprintf(DEFAULT_ERROR_STRING,
			"No other write-permission on %s",name);
		NWARN(DEFAULT_ERROR_STRING);
		return(0);
	}

	return(1);
}

int file_exists(const char *pathname)
{
	struct stat statb;

	if( stat(pathname,&statb) < 0 ){
		if( verbose ){
			sprintf(msg_str,
				"file %s does not already exist",pathname);
			prt_msg(msg_str);
		}
		return(0);
	}
	return(1);
}


