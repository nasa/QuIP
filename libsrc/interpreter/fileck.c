#include <stdio.h>

#include "quip_config.h"
#include "fileck.h"

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

#include "quip_prot.h"

// this implementation requires unix stat!?
int _path_exists(QSP_ARG_DECL  const char *pathname)
{
#ifdef HAVE_SYS_STAT_H
	struct stat statb;

	if( stat(pathname,&statb) < 0 ){
		if( verbose || errno != ENOENT ) tell_sys_error(pathname);
		return(0);
	}
#ifndef S_ISDIR
// windows
#define S_ISDIR(mode)	(mode&S_IFDIR)
#endif // ! S_ISDIR
#ifndef S_ISREG
// windows
#define S_ISREG(mode)	(mode&S_IFREG)
#endif // ! S_ISREG
#ifndef S_ISCHR
// windows
#define S_ISCHR(mode)	(mode&S_IFCHR)
#endif // ! S_ISCHR

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
#else /* !HAVE_SYS_STAT_H */
	warn("Need to implement path_exists without stat!?");
	return(0);
#endif /* HAVE_SYS_STAT_H */
}

int _directory_exists(QSP_ARG_DECL  const char *dirname)
{
#ifdef HAVE_SYS_STAT_H
	struct stat statb;

	if( stat(dirname,&statb) < 0 ){
		if( verbose ) tell_sys_error(dirname);
		return(0);
	}

	if( ! S_ISDIR(statb.st_mode) ){
		sprintf(DEFAULT_ERROR_STRING,"%s is not a directory!?",dirname);
		warn(DEFAULT_ERROR_STRING);
		return(0);
	}
	return(1);
#else /* !HAVE_SYS_STAT_H */
	warn("Need to implement directory_exists without stat!?");
	return(0);
#endif /* HAVE_SYS_STAT_H */
}

int _regfile_exists(QSP_ARG_DECL  const char *pathname)
{
#ifdef HAVE_SYS_STAT_H
	struct stat statb;

	if( stat(pathname,&statb) < 0 ){
		if( verbose ) tell_sys_error(pathname);
		return(0);
	}

	if( ! S_ISREG(statb.st_mode) ){
		sprintf(DEFAULT_ERROR_STRING,"%s is not a regular file!?",pathname);
		warn(DEFAULT_ERROR_STRING);
		return(0);
	}
	return(1);
#else /* !HAVE_SYS_STAT_H */
	warn("Need to implement regfile_exists without stat!?");
	return(0);
#endif /* HAVE_SYS_STAT_H */
}

int _can_write_to(QSP_ARG_DECL  const char *name)
{
#ifdef HAVE_SYS_STAT_H
	struct stat statb;

	if( stat(name,&statb) < 0 ){
		if( verbose ) tell_sys_error(name);
		return(0);
	}

	/* make sure we have permission to write here */

	/* are we the owner of this file/directory? */

#ifndef BUILD_FOR_WINDOWS
	if( statb.st_uid == getuid() ||
		statb.st_uid == geteuid() ){
#endif // ! BUILD_FOR_WINDOWS

		/* we are owner or effective owner */

#ifdef S_IWUSR
#define OWNER_CAN_WRITE	S_IWUSR
#else
// windows?
#ifdef S_IWRITE
#define OWNER_CAN_WRITE	S_IWRITE
#else
#error Oops - missing mode mask definition
#endif // ! S_IWRITE
#endif // ! S_IWUSR

		if( (statb.st_mode & OWNER_CAN_WRITE ) == 0 ){
			sprintf(DEFAULT_ERROR_STRING,
				"No owner write-permission on %s", name);
			warn(DEFAULT_ERROR_STRING);
			return(0);
		}
#ifndef BUILD_FOR_WINDOWS
	}
#endif // ! BUILD_FOR_WINDOWS

// No other permissions on windows???
#ifdef S_IWOTH
	  else if( (statb.st_mode & S_IWOTH) == 0 ){
		sprintf(DEFAULT_ERROR_STRING,
			"No other write-permission on %s",name);
		warn(DEFAULT_ERROR_STRING);
		return(0);
	}
#endif /* S_IWOTH */

	return(1);
#else /* !HAVE_SYS_STAT_H */
	warn("Need to implement can_write_to without stat!?");
	return(0);
#endif /* HAVE_SYS_STAT_H */
}

int _file_exists(QSP_ARG_DECL  const char *pathname)
{
#ifdef HAVE_SYS_STAT_H
	struct stat statb;

	if( stat(pathname,&statb) < 0 ){
		if( verbose ){
			sprintf(MSG_STR,
				"file %s does not already exist",pathname);
			prt_msg(MSG_STR);
		}
		return(0);
	}
	return(1);
#else /* !HAVE_SYS_STAT_H */
	warn("Need to implement file_exists without stat!?");
	return(0);
#endif /* HAVE_SYS_STAT_H */
}

/* long */ off_t _file_content_size(QSP_ARG_DECL  const char *pathname)
{
#ifdef HAVE_SYS_STAT_H
	struct stat statb;

	if( stat(pathname,&statb) < 0 ){
		sprintf(ERROR_STRING,
			"file_content_size:  file %s does not exist!?",
			pathname);
		warn(ERROR_STRING);
		return(-1);	// off_t is unsigned!?
	}
	return statb.st_size;

#else /* !HAVE_SYS_STAT_H */
	warn("Need to implement file_content_size without stat!?");
	return(-1);
#endif /* HAVE_SYS_STAT_H */
}


/* long */ off_t _fp_content_size(QSP_ARG_DECL  FILE *fp)
{
#ifdef HAVE_SYS_STAT_H
	struct stat statb;

	if( fstat(fileno(fp),&statb) < 0 ){
		warn("fp_content_size:  couldn't fstat!?");
		return(-1);
	}
	return statb.st_size;

#else /* !HAVE_SYS_STAT_H */
	warn("Need to implement fp_content_size without fstat!?");
	return(-1);
#endif /* HAVE_SYS_STAT_H */
}


// Checks for existence, and write permission.
// Originally introduced in CUDA support library,
// to check for device files...

int _check_file_access(QSP_ARG_DECL  const char *filename)
{
	if( ! file_exists(filename) ){
		sprintf(ERROR_STRING,"File %s does not exist.",filename);
		//error1(ERROR_STRING);
		warn(ERROR_STRING);
		return -1;
	}
	/*if( ! can_read_from(filename) ){
		sprintf(ERROR_STRING,"File %s exists, but no read permission.",filename);
		error1(ERROR_STRING);
	}*/
	if( ! can_write_to(filename) ){
		sprintf(ERROR_STRING,"File %s exists, but no write permission.",filename);
		//error1(ERROR_STRING);
		warn(ERROR_STRING);
		return -1;
	}
	return 0;
}

