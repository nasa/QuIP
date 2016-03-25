
#include "quip_config.h"

int force_avi_info_load;		/* see comment in matio.c */

#ifdef HAVE_AVI_SUPPORT

#include <stdio.h>		/* fileno() */

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>		/* stat() */
#endif

#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>		/* stat() */
#endif

#ifdef SOLARIS
#define _POSIX_C_SOURCE	1		/* force fileno() in stdio.h */
#endif /* SOLARIS */

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* unlink() */
#endif

//#include <ctype.h>
//#include <time.h>
//#include <math.h>		/* floor() */

// added for debian:
#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#include "quip_prot.h"
#include "fio_prot.h"

// BUG this defn is duplicated from avi.c
#define HDR_P(ifp)		((AVCodec_Hdr *)ifp->if_hdr_p)

/* local prototypes */

#define AVI_INFO_MAGIC_STRING	"AVIinfo"

static void make_avs_name(char buf[LLEN], Image_File *ifp )
{
	int l,i;

	strcpy(buf,ifp->if_pathname);
	/* strip the suffix */
	l=strlen(ifp->if_pathname);
	i=l-1;
	while( i > 0 && buf[i] != '.' )
		i--;
	if( buf[i]=='.' ){
		if( (!strcmp(&buf[i+1],"avi")) || (!strcmp(&buf[i+1],"AVI")) ){
			buf[i]=0;
		}
	}
	strcat(buf,".avs");
}

static FILE * remove_avi_info_if_stale(const char *info_name,FILE *info_fp,const char *src_name)
{
	struct stat info_statb, file_statb;

	/* if the info file is older than the file itself, then it should be unlinked and recomputed */
	/* need to stat both files... */

	if( fstat(fileno(info_fp),&info_statb) < 0 ){
		_tell_sys_error(DEFAULT_QSP_ARG  "remove_avi_info_if_stale:  fstat:");
		NERROR1("unable to stat avi info file");
	}

	if( stat(src_name,&file_statb) < 0 ){
#ifdef LONG_64_BIT
//#ifdef IA64
		_tell_sys_error(DEFAULT_QSP_ARG  "remove_avi_info_if_stale:  stat:");
		sprintf(DEFAULT_ERROR_STRING,"unable to stat avi data file %s",src_name);
		NERROR1(DEFAULT_ERROR_STRING);
#elif defined(LONG_32_BIT)
		if( errno == EOVERFLOW ){
			struct stat64 stat64b;
			if( stat64(src_name,&stat64b) < 0 ){
				_tell_sys_error(DEFAULT_QSP_ARG  "remove_avi_info_if_stale:  stat64:");
				sprintf(DEFAULT_ERROR_STRING,"unable to stat64 avi data file %s",src_name);
				NERROR1(DEFAULT_ERROR_STRING);
			}
			file_statb.st_mtime = stat64b.st_mtime;
		} else {
			_tell_sys_error(DEFAULT_QSP_ARG  "remove_avi_info_if_stale:  stat:");
			sprintf(DEFAULT_ERROR_STRING,"unable to stat avi data file %s",src_name);
			NERROR1(DEFAULT_ERROR_STRING);
		}
#else
#error "sizeof(long) not properly set by configure!?"
//		NERROR1("CAUTIOUS:  sizeof(long) not properly set by configure!?");
		assert( ! "sizeof(long) not properly set by configure!?");
#endif
	}

	/* now compare mod times */
	if( file_statb.st_mtime > info_statb.st_mtime ){
		sprintf(DEFAULT_ERROR_STRING,"Existing jpeg info file %s is older than file %s, will unlink and recompute",
				info_name,src_name);
		NADVISE(DEFAULT_ERROR_STRING);
		fclose(info_fp);

		if( unlink(info_name) < 0 ){
			_tell_sys_error(DEFAULT_QSP_ARG  "remove_avi_info_if_stale:  unlink:");
			NWARN("unable to remove stale jpeg info file");
			/* may not have permission */
		}
		return(NULL);
	}

	return(info_fp);
}

static int read_avi_info(Image_File *ifp, FILE *info_fp)
{
	char buf[16];
	unsigned long nf, n_skew, n_seek;
	int i;

	/* first check the magic number */
	if( fread(buf,1,strlen(AVI_INFO_MAGIC_STRING)+1,info_fp) != strlen(AVI_INFO_MAGIC_STRING)+1 ){
		sprintf(DEFAULT_ERROR_STRING,"read_avi_info_top:  missing magic number data");
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	if( buf[strlen(AVI_INFO_MAGIC_STRING)] != '\n' ){
		sprintf(DEFAULT_ERROR_STRING,"read_avi_info_top:  bad magic number terminator");
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	buf[strlen(AVI_INFO_MAGIC_STRING)]=0;
	if( strcmp(buf,AVI_INFO_MAGIC_STRING) ){
		sprintf(DEFAULT_ERROR_STRING,"read_avi_info_top:  bad magic number data");
		NWARN(DEFAULT_ERROR_STRING);
		sprintf(DEFAULT_ERROR_STRING,"read_jpeg_info_top:  expected \"%s\" but encountered \"%s\"",
			AVI_INFO_MAGIC_STRING,buf);
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	/* now we know that the file starts out in the right format */

	if( fscanf(info_fp,"%ld",&nf) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"read_avi_info_top:  bad nframes");
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	if( fscanf(info_fp,"%ld",&n_skew) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"read_avi_info_top:  bad n_skew");
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}
	HDR_P(ifp)->avch_n_skew_tbl_entries = n_skew;

	/* allocate the table */
	HDR_P(ifp)->avch_skew_tbl = (Frame_Skew *)
		getbuf( HDR_P(ifp)->avch_n_skew_tbl_entries * sizeof(Frame_Skew) );

	for(i=0;i<HDR_P(ifp)->avch_n_skew_tbl_entries;i++){
		if( fscanf(info_fp,"%d %d",
			&HDR_P(ifp)->avch_skew_tbl[i].frame_index,
			&HDR_P(ifp)->avch_skew_tbl[i].pts_offset ) != 2 ){
			NWARN("error reading frame skew data");
			return(-1);
		}
	}

	if( fscanf(info_fp,"%ld",&n_seek) != 1 ){
		sprintf(DEFAULT_ERROR_STRING,"read_avi_info_top:  bad n_seek");
		NWARN(DEFAULT_ERROR_STRING);
		return(-1);
	}

	HDR_P(ifp)->avch_n_seek_tbl_entries = n_seek;

	/* allocate the table */
	HDR_P(ifp)->avch_seek_tbl = (Seek_Info *)
		getbuf( HDR_P(ifp)->avch_n_seek_tbl_entries * sizeof(Seek_Info) );

	for(i=0;i<HDR_P(ifp)->avch_n_seek_tbl_entries;i++){
		if( fscanf(info_fp,"%d %d",
			&HDR_P(ifp)->avch_seek_tbl[i].seek_target,
			&HDR_P(ifp)->avch_seek_tbl[i].seek_result ) != 2 ){
			NWARN("error reading seek data");
			return(-1);
		}
	}

	SET_OBJ_FRAMES(ifp->if_dp, nf);

	return(0);
}


/* Check for a .avs file, and if it exists, read it!
 *
 */

int check_avi_info(Image_File *ifp)
{
	char avs_name[LLEN];
	FILE *fp;

	make_avs_name(avs_name,ifp);	/* look for (fast) binary version first */
//sprintf(error_string,"Checking for avi info file %s",avs_name);
//advise(error_string);
	fp=fopen(avs_name,"r");
	if( !fp ) {
		return(-1);
	} else {
		fp = remove_avi_info_if_stale(avs_name,fp,ifp->if_pathname);
		if( !fp ) return(-1);
		return( read_avi_info(ifp,fp) );
	}
	/* NOT_REACHED */
	return(0);
}

#ifdef FOOBAR
/* We do this for portability */

static void put_long(u_long l, FILE *fp)
{
	if( fputc( l & 0xff, fp ) == EOF )
		NERROR1("put_long:  error writing first byte");
	if( fputc( (l & 0xff00) >> 8, fp ) == EOF )
		NERROR1("put_long:  error writing second byte");
	if( fputc( (l & 0xff0000) >> 16, fp ) == EOF )
		NERROR1("put_long:  error writing third byte");
	if( fputc( (l & 0xff000000) >> 24, fp ) == EOF )
		NERROR1("put_long:  error writing fourth byte");
}

static long get_long(FILE *fp)
{
	u_long l,r;
	r  = getc(fp);
	if( r == EOF ) NERROR1("get_long:  error getting first byte");

	l  = getc(fp);
	if( l == EOF ) NERROR1("get_long:  error getting second byte");
	r += l << 8;

	l  = getc(fp);
	if( l == EOF ) NERROR1("get_long:  error getting third byte");
	r += l << 16;

	l  = getc(fp);
	if( l == EOF ) NERROR1("get_long:  error getting fourth byte");
	r += l << 24;

	return(r);
}
#endif /* FOOBAR */

void save_avi_info(Image_File *ifp)
{
	char avs_name[LLEN];
	FILE *fp;
	int i;

	make_avs_name(avs_name,ifp);
	fp = fopen(avs_name,"w");
	if( !fp ){
		sprintf(DEFAULT_ERROR_STRING,"save_avi_info:  couldn't open avi info file %s for writing",avs_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
	fprintf(fp,"AVIinfo\n");
	fprintf(fp,"%d\n",OBJ_FRAMES(ifp->if_dp));
	fprintf(fp,"%d\n",HDR_P(ifp)->avch_n_skew_tbl_entries);
	for(i=0;i<HDR_P(ifp)->avch_n_skew_tbl_entries;i++){
		fprintf(fp,"%d\t%d\n",HDR_P(ifp)->avch_skew_tbl[i].frame_index,
					HDR_P(ifp)->avch_skew_tbl[i].pts_offset);
		/*
		put_long(HDR_P(ifp)->avch_skew_tbl[i].frame_index,fp);
		put_long(HDR_P(ifp)->avch_skew_tbl[i].pts_offset,fp);
		*/
	}

	fprintf(fp,"%d\n",HDR_P(ifp)->avch_n_seek_tbl_entries);
	for(i=0;i<HDR_P(ifp)->avch_n_seek_tbl_entries;i++){
		/*
		put_long(HDR_P(ifp)->avch_seek_tbl[i].seek_target,fp);
		put_long(HDR_P(ifp)->avch_seek_tbl[i].seek_result,fp);
		*/
		fprintf(fp,"%d\t%d\n",HDR_P(ifp)->avch_seek_tbl[i].seek_target,
					HDR_P(ifp)->avch_seek_tbl[i].seek_result);
	}
	fclose(fp);
}


#endif /* HAVE_AVI_SUPPORT */

