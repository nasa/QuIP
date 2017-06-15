
#include "quip_config.h"

#include <stdio.h>

#ifdef HAVE_SIGNAL_H
#include <signal.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "query_bits.h"	// LLEN - BUG
#include "fio_prot.h"
#include "nports_api.h"
#include "quip_prot.h"
#include "img_file.h"
//#include "filetype.h"

/* flag ignored, for arg compatibility w/ xmit_data */
void xmit_img_file(QSP_ARG_DECL  Port *mpp,Image_File *ifp,int flag)	/** send a file header */
{
	int32_t len;
	int32_t code;

#ifdef SIGPIPE
	// We used to call a handler (if_pipe) that performed
	// a fatal error exit.  Now we would like to recover
	// a little more gracefully.  That means that put_port_int32 etc
	// can return after the port is closed.
	signal(SIGPIPE,SIG_IGN);
#endif /* SIGPIPE */

	code=P_IMG_FILE;
	if( put_port_int32(QSP_ARG  mpp,code) == (-1) ){
		WARN("xmit_file:  error sending code");
		return;
	}

	len=(int32_t)strlen(ifp->if_name)+1;


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

	if( put_port_int32(QSP_ARG  mpp,len) == -1 ){
		WARN("xmit_file:  error writing name length word");
		return;
	}

	if( put_port_int32(QSP_ARG  mpp, ifp->if_nfrms) == -1 ||
	    put_port_int32(QSP_ARG  mpp, FT_CODE(IF_TYPE(ifp)) ) == -1 ||
	    put_port_int32(QSP_ARG  mpp, ifp->if_flags ) == -1 ){
		WARN("error sending image file header data");
		return;
	}
	    
	if( write_port(QSP_ARG  mpp,ifp->if_name,len) == (-1) ){
		WARN("xmit_file:  error writing image file name");
		return;
	}

	/* now send the associated data_obj header... */

	assert( ifp->if_dp != NULL );

	xmit_obj(QSP_ARG  mpp,ifp->if_dp,0);
}

/** recv_img_file - recieve a new data object */

long recv_img_file(QSP_ARG_DECL  Port *mpp, /* char **bufp */ Packet *pkp )
{
	long len;
	Image_File *old_ifp, *new_ifp;
	char namebuf[LLEN];
	Image_File imgf, *ifp;
	long code;

	ifp=(&imgf);
	len=get_port_int32(QSP_ARG  mpp);
	if( len <= 0 ) goto error_return;

#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(ERROR_STRING,
"recv_file:  want %ld name bytes",len);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( (ifp->if_nfrms = get_port_int32(QSP_ARG  mpp)) == BAD_PORT_LONG ||
	    (ifp->if_ftp = filetype_for_code( QSP_ARG  (filetype_code) get_port_int32(QSP_ARG  mpp))) == NULL ||
	    (ifp->if_flags = (short) get_port_int32(QSP_ARG  mpp)) == (short)BAD_PORT_LONG ){
		WARN("error getting image file data");
		goto error_return;
	}

	if( len > LLEN ){
		WARN("more than LLEN name chars!?");
		goto error_return;
	}
	if( read_port(QSP_ARG  mpp,namebuf,len) != len ){
		WARN("recv_file:  error reading data object name");
		goto error_return;
	}

	/* where does the string get null-terminated? */

	if( (long)strlen( namebuf ) != len-1 ){
		u_int i;

		sprintf(ERROR_STRING,"name length %ld, expected %ld",
			(long)strlen(namebuf), len-1);
		advise(ERROR_STRING);
		sprintf(ERROR_STRING,"name:  \"%s\"",namebuf);
		advise(ERROR_STRING);
		for(i=0;i<strlen(namebuf);i++){
			sprintf(ERROR_STRING,"name[%d] = '%c' (0%o)",
				i,namebuf[i],namebuf[i]);
			advise(ERROR_STRING);
		}
		ERROR1("choked");
	}

	old_ifp=img_file_of(QSP_ARG  namebuf);

	if( old_ifp != NULL ){
		DEL_IMG_FILE(old_ifp);
		rls_str((char *)old_ifp->if_name);	// BUG?  release name here or not?
		old_ifp = NULL;
	}

	new_ifp=new_img_file(QSP_ARG  namebuf);
	if( new_ifp==NULL ){
		sprintf(ERROR_STRING,
			"recv_file:  couldn't create file struct \"%s\"",
			namebuf);
		WARN(ERROR_STRING);
		goto error_return;
	}

	new_ifp->if_nfrms = imgf.if_nfrms;
	new_ifp->if_ftp = filetype_for_code(QSP_ARG  IFT_NETWORK);
	new_ifp->if_flags = imgf.if_flags;
	new_ifp->if_dp = NULL;	/* BUG should receive? */
	new_ifp->if_pathname = new_ifp->if_name;

	code = get_port_int32(QSP_ARG  mpp);
	if( code == -1 )
		ERROR1("error port code received!?");
	if( code != P_DATA ){
		sprintf(ERROR_STRING,
	"recv_file:  expected data object packet to complete transmission of file %s!?",
			new_ifp->if_name);
		ERROR1(ERROR_STRING);
	}
		
	// the cast generates a compiler warning???
	if( recv_obj(QSP_ARG  mpp, pkp ) != sizeof(Data_Obj) ){
		WARN("Error receiving data object!?");
		goto error_return;
	}

	// The packet returns the dp in pk_extra...
	return sizeof(*new_ifp);	// BUG - what size should we return???

error_return:
	return -1;
}


