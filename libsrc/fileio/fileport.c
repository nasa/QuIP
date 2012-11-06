
#include "quip_config.h"

char VersionId_fio_fileport[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "fio_prot.h"
#include "nports_api.h"
#include "debug.h"
#include "img_file.h"
#include "filetype.h"

/* flag ignored, for arg compatibility w/ xmit_data */
void xmit_file(QSP_ARG_DECL  Port *mpp,Image_File *ifp,int flag)	/** send a file header */
{
	u_long len;
	long code;

#ifdef SIGPIPE
	signal(SIGPIPE,if_pipe);
#endif /* SIGPIPE */

	code=P_FILE;
	if( put_port_int32(mpp,code) == (-1) )
		WARN("xmit_file:  error sending code");

	len=strlen(ifp->if_name)+1;


	/* we don't send the file data, just the header data */
	/* We need to send:
	 *
	 *	name
	 *	if_nfrms	(frms written/read)
	 *	if_type		(file format)
	 *	(file type specific header?)
	 *	if_flags
	 *	(dp ptr to obj with dimensions?)
	 *	(if_pathname)	(don't really care since file is on remot sys)
	 */

	if( put_port_int32(mpp,len) == -1 )
		WARN("xmit_file:  error writing name length word");

	if( put_port_int32(mpp,(long) ifp->if_nfrms) == -1 ||
	    put_port_int32(mpp,(long) ifp->if_type) == -1 ||
	    put_port_int32(mpp,(long) ifp->if_flags ) == -1 )
		WARN("error sending image file header data");
	    
	if( write_port(mpp,ifp->if_name,len) == (-1) )
		WARN("xmit_file:  error writing image file name");

	/* now send the associated data_obj header... */

#ifdef CAUTIOUS
	if( ifp->if_dp == NO_OBJ ){
		sprintf(error_string,
	"CAUTIOUS:  xmit_file:  file %s has no associated data object!?",
			ifp->if_name);
		WARN(error_string);
	}
#endif /* CAUTIOUS */

	xmit_obj(QSP_ARG  mpp,ifp->if_dp,0);
}

Image_File *
recv_file(QSP_ARG_DECL  Port *mpp)			/** recieve a new data object */
{
	long len;
	Image_File *old_ifp, *new_ifp;
	char namebuf[LLEN];
	Image_File imgf, *ifp;
	long code;

	ifp=(&imgf);
	len=get_port_int32(QSP_ARG  mpp);
	if( len <= 0 ) return(NO_IMAGE_FILE);

#ifdef DEBUG
if( debug & debug_data ){
sprintf(error_string,
"recv_file:  want %ld name bytes",len);
advise(error_string);
}
#endif /* DEBUG */

	if( (ifp->if_nfrms = get_port_int32(QSP_ARG  mpp)) == BAD_PORT_LONG ||
	    (ifp->if_type = (filetype_code) get_port_int32(QSP_ARG  mpp)) == (filetype_code)BAD_PORT_LONG ||
	    (ifp->if_flags = /*(short)*/ get_port_int32(QSP_ARG  mpp)) == (short)BAD_PORT_LONG ){
		WARN("error getting image file data");
		return(NO_IMAGE_FILE);
	}

	if( len > LLEN ){
		WARN("more than LLEN name chars!?");
		return(NO_IMAGE_FILE);
	}
	if( read_port(QSP_ARG  mpp,namebuf,len) != len ){
		WARN("recv_file:  error reading data object name");
		return(NO_IMAGE_FILE);
	}

	/* where does the string get null-terminated? */

	if( (long)strlen( namebuf ) != len-1 ){
		u_int i;

		sprintf(error_string,"name length %ld, expected %ld",
			(long)strlen(namebuf), len-1);
		advise(error_string);
		sprintf(error_string,"name:  \"%s\"",namebuf);
		advise(error_string);
		for(i=0;i<strlen(namebuf);i++){
			sprintf(error_string,"name[%d] = '%c' (0%o)",
				i,namebuf[i],namebuf[i]);
			advise(error_string);
		}
		ERROR1("choked");
	}

	old_ifp=img_file_of(QSP_ARG  namebuf);

	if( old_ifp != NO_IMAGE_FILE ){
		DEL_IMG_FILE(old_ifp->if_name);
		rls_str((char *)old_ifp->if_name);
		old_ifp = NO_IMAGE_FILE;
	}

	new_ifp=new_img_file(QSP_ARG  namebuf);
	if( new_ifp==NO_IMAGE_FILE ){
		sprintf(error_string,
			"recv_file:  couldn't create file struct \"%s\"",
			namebuf);
		WARN(error_string);
		return(NO_IMAGE_FILE);
	}

	new_ifp->if_nfrms = imgf.if_nfrms;
	new_ifp->if_type = IFT_NETWORK;
	new_ifp->if_flags = imgf.if_flags;
	new_ifp->if_dp = NO_OBJ;	/* BUG should receive? */
	new_ifp->if_pathname = new_ifp->if_name;

	code = get_port_int32(QSP_ARG  mpp);
	if( code == -1 )
		ERROR1("error port code received!?");
	if( code != P_DATA ){
		sprintf(error_string,
	"recv_file:  expected data object packet to complete transmission of file %s!?",
			new_ifp->if_name);
		ERROR1(error_string);
	}
		
	new_ifp->if_dp = recv_obj(QSP_ARG  mpp);

	return(new_ifp);
}

