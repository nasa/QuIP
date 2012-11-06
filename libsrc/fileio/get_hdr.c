/*
 *	filched from libchips.a to replace halloc() with getbuf()
 *	tidied up in the process, jbm 11-19-88
 */

#include "quip_config.h"

char VersionId_fio_get_hdr[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "img_file_hdr.h"
#include "fio_api.h"
#include "uio.h"
#include "hipl_fmt.h"
#include "debug.h"
#include "getbuf.h"
#include "savestr.h"
#include "get_hdr.h"
#include "filetype.h"

#define LINES 100

int fh_err(Header *hd,const char *s)
{
	NWARN("error reading HIPS file header:");
	NWARN(s);
	return(0);
}

int read_chr(int fd)
{
	char c;

	if( read(fd,&c,1) != 1 ) return(-1);
	return(c);
}

char *gtline(int fd)	/** get a non-empty line */
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
		*s ++ = c;
		*s = '\0';
		return(tbuf);
	}
}

int dfscanf(int fd,Header *hd)
{
	int n;
	register char *s;

	s=gtline(fd);
	if( s==NULL ) return( fh_err(hd,"no string for dscanf") );
	if( sscanf(s,"%d",&n) != 1 ){

		sprintf(DEFAULT_ERROR_STRING,
		"dscanf:  string \"%s\" is not an integer",s);
		NWARN(DEFAULT_ERROR_STRING);
		sprintf(DEFAULT_ERROR_STRING,"Supposed filetype is %s",
			ft_tbl[get_filetype()].ft_name);
		advise(DEFAULT_ERROR_STRING);

		givbuf(s);
		return(0);
	}
	givbuf(s);
	return( n );
}

char *catlines(const char *s1,const char *s2)
{
	char *s;

	s=(char *)getbuf(strlen(s1)+strlen(s2)+1);
	strcpy(s,s1);
	givbuf(s1);
	strcat(s,s2);
	givbuf(s2);
	return(s);
}

int ftch_header(int fd,Header *hd)
{
	const char *s2;
	register char *s;

 	if( (s=gtline(fd)) == NULL) return( fh_err(hd,"no orig name") );
#ifdef DEBUG
if( debug ) advise("saving original name");
#endif
	hd->orig_name = s;
 	if( (s=gtline(fd)) == NULL) return( fh_err(hd,"no seq name") );
	hd->seq_name = s;
#ifdef DEBUG
if( debug ) advise("saving sequence name");
#endif
	hd->num_frame = dfscanf(fd,hd);
 	if( (s=gtline(fd)) == NULL) return( fh_err(hd,"no orig date") );
	hd->orig_date = s;
#ifdef DEBUG
if( debug ) advise("saving original date");
#endif
	hd->rows = dfscanf(fd,hd);
	hd->cols = dfscanf(fd,hd);
	hd->bits_per_pixel = dfscanf(fd,hd);
	hd->bit_packing = dfscanf(fd,hd);
	hd->pixel_format = dfscanf(fd,hd);

	/* get sequence history */
 	if( (s=gtline(fd)) == NULL) return( fh_err(hd,"no seq hist") );
	while( s[ strlen(s)-3 ] == '|') {	/* end of string: "...|\\\n" */
		s2=gtline(fd);
		if( s2 == NULL ) return( fh_err(hd,"no seq hist cont.") );
		s=catlines(s,s2);
	}

#ifdef DEBUG
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
#ifdef DEBUG
if( debug ) advise("saving sequence description");
#endif
	hd->seq_desc = s2;

 	return(1);
}

