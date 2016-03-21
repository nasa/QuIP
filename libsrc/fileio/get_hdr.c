/*
 *	filched from libchips.a to replace halloc() with getbuf()
 *	tidied up in the process, jbm 11-19-88
 */

#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "fio_api.h"
#include "fio_prot.h"	// for some reason this has to come before fio_api.h???
//#include "fio_api.h"
#include "debug.h"
#include "getbuf.h"
#include "img_file/img_file_hdr.h"
#include "hips/hipl_fmt.h"
//#include "get_hdr.h"

#define LINES 100

static int fh_err(Header *hd,const char *s)
{
	NWARN("error reading HIPS file header:");
	NWARN(s);
	return(0);
}

static int read_chr(int fd)
{
	char c;

	if( read(fd,&c,1) != 1 ) return(-1);
	return(c);
}

static char *gtline(int fd)	/** get a non-empty line */
{
	int n_avail, n_gotten;
	int c;
	register char *s,*s2;
	char *tbuf;

	tbuf=(char *)getbuf(LINELENGTH);
	tbuf[0]=0;

	s=tbuf;
	n_avail = LINELENGTH-1;
	n_gotten=0;
	while( (c=read_chr(fd))!= (-1) && c !='\n' && c !='\r' ){
		if(n_avail <= 2) {
			tbuf[n_gotten]=0;
			s2 = (char *)getbuf(LINELENGTH+n_gotten);
			strcpy(s2,tbuf);
			givbuf(tbuf);
			tbuf=s2;
			n_avail += LINELENGTH;
			s = tbuf + strlen(tbuf);
		}
		*s++ = (char)c;
		n_gotten++;
		n_avail--;
	}
 	if( c==(-1) ) return(NULL);
 	else { 					/* c == '\n' or '\r' */
		*s ++ = (char) c;
		*s = '\0';
		return(tbuf);
	}
}

static int dfscanf(QSP_ARG_DECL  int fd,Header *hd)
{
	int n;
	register char *s;

	s=gtline(fd);
	if( s==NULL ) return( fh_err(hd,"no string for dfscanf") );
	if( sscanf(s,"%d",&n) != 1 ){

		sprintf(ERROR_STRING,
		"dfscanf:  string \"%s\" is not an integer",s);
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,"Supposed filetype is %s",
			FT_NAME(current_filetype()) );
		advise(ERROR_STRING);

		givbuf(s);
		return(0);
	}
	givbuf(s);
	return( n );
}

static char *catlines(const char *s1,const char *s2)
{
	char *s;

	s=(char *)getbuf(strlen(s1)+strlen(s2)+1);
	strcpy(s,s1);
	givbuf((void *)s1);
	strcat(s,s2);
	givbuf((void *)s2);
	return(s);
}

int ftch_header(QSP_ARG_DECL  int fd,Header *hd)
{
	const char *s2;
	register char *s;

 	if( (s=gtline(fd)) == NULL) return( fh_err(hd,"no orig name") );
#ifdef QUIP_DEBUG
if( debug ) advise("saving original name");
#endif
	hd->orig_name = s;
 	if( (s=gtline(fd)) == NULL) return( fh_err(hd,"no seq name") );
	hd->seq_name = s;
#ifdef QUIP_DEBUG
if( debug ) advise("saving sequence name");
#endif
	hd->num_frame = dfscanf(QSP_ARG  fd,hd);
 	if( (s=gtline(fd)) == NULL) return( fh_err(hd,"no orig date") );
	hd->orig_date = s;
#ifdef QUIP_DEBUG
if( debug ) advise("saving original date");
#endif
	hd->rows = dfscanf(QSP_ARG  fd,hd);
	hd->cols = dfscanf(QSP_ARG  fd,hd);
	hd->bits_per_pixel = dfscanf(QSP_ARG  fd,hd);
	hd->bit_packing = dfscanf(QSP_ARG  fd,hd);
	hd->pixel_format = dfscanf(QSP_ARG  fd,hd);

	/* get sequence history */
 	if( (s=gtline(fd)) == NULL) return( fh_err(hd,"no seq hist") );
	while( s[ strlen(s)-3 ] == '|') {	/* end of string: "...|\\\n" */
		s2=gtline(fd);
		if( s2 == NULL ) return( fh_err(hd,"no seq hist cont.") );
		s=catlines(s,s2);
	}

#ifdef QUIP_DEBUG
if( debug ) advise("saving sequence history");
#endif
	hd->seq_history = s;

	/* get sequence description */
	s2=savestr("");
	while(1) {
		if( (s=gtline(fd)) == NULL ) return( fh_err(hd,"no seq desc") );
		if( (!strcmp(s,".\n")) || (!strcmp(s,".\r")) ){
			givbuf(s);
			goto gotit;
		}
		s2=catlines(s2,s);
	}
gotit:
#ifdef QUIP_DEBUG
if( debug ) advise("saving sequence description");
#endif
	hd->seq_desc = s2;

 	return(1);
}

