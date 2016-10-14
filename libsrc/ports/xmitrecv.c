#include "quip_config.h"

#include "quip_config.h"

#include <stdio.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* write() */
#endif
#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
/* these are needed for select(2): */
#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif
#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

#include "ports.h"
#include "getbuf.h"
#include "debug.h"
#include "rn.h"
//#include "submenus.h"	/* call_event_funcs */

//static char * remote_filename=NULL;

#ifdef BUILD_FOR_WINDOWS

void report_windows_error(void)
{
	int e;
	switch( (e=WSAGetLastError()) ){
		// Lots of possible error codes to go here and
		// print a more informative message...
		default:
			fprintf(stderr,"Unhandled Windows error %d!?\n",e);
			break;
	}
	fflush(stderr);
}

#endif // BUILD_FOR_WINDOWS




/* Use select() to check for data.  Return 1 if there is data available,
 * 0 if no data, and < 0 for error.
 *
 */

int check_port_data(QSP_ARG_DECL  Port *mpp, uint32_t microseconds)
{
#ifdef HAVE_SELECT
	fd_set rd_fds, null_fds;
	struct timeval time_out;
	int nfds;
	int v;

	FD_ZERO(&rd_fds);
	FD_ZERO(&null_fds);
	FD_SET( mpp->mp_sock, &rd_fds );

	// per select(2) man page, nfds should be 1 more than the
	// largest file descriptor.  We only have one.
	nfds = 1 + mpp->mp_sock;

	time_out.tv_sec=0;
	time_out.tv_usec=microseconds;	/* 0 for immediate poll */

	// select returns the number of ready descriptors
	// < 0 for error, 0 if nothing readable
	// returns 1 if EOF!

//fprintf(stderr,"check_port_data calling select, usecs = %d...\n",microseconds);
	v=select( nfds, &rd_fds, &null_fds, &null_fds, &time_out );
//fprintf(stderr,"check_port_data: select returned %d.\n",v);
	if( v < 0 ){
		if( errno != EINTR ){
			tell_sys_error("(check_for_data) select");
			return -1;
		}
		/* else EINTR, which is normal */

		/* we used to call a "recovery" function here!? */
		/* but it had been commented out for a long time... */
		return 0;
	}
	if( v == 0 ) return 0;

#ifdef CAUTIOUS
	if( !FD_ISSET(mpp->mp_sock,&rd_fds) ){
		WARN("CAUTIOUS:  check_for_data:  shouldn't happen");
		return(-1);
	}
#endif // CAUTIOUS
	return 1;

#else // ! HAVE_SELECT

	// If no select, return 1 even though we may block...
	return 1;

#endif /* ! HAVE_SELECT */
}

/*
 * Use select() to wait for readable data on this port.
 * Returns 0 when there is readable data, -1 for some other error.
 */

static int wait_for_data(QSP_ARG_DECL  Port *mpp)
{
	uint32_t usecs;
	int v;

	v=check_port_data(QSP_ARG  mpp, 0);
	if( v > 0 ) return 0;
	if( v < 0 ) return v;

	usecs=100000;	/* 10 Hz poll */

	while( (v=check_port_data(QSP_ARG  mpp, usecs)) != 1 ){
		// We return -1 from here on a select error
		if( v < 0 ) return(v);

#ifndef BUILD_FOR_IOS
		call_event_funcs(SINGLE_QSP_ARG);	// wait_for_data

		// on iOS, call_event_funcs puts the interpreter
		// in the halting state...
		// This is fine when we are waiting for a new command, but
		// not when we are in the middle of reading a packet...
		//
		// OR when we are just trying to read a packet that is there!?
		// 
		// When we enable this code with iOS, we get command synchronization
		// erros, e.g. when we send 'advise foo' the device displays
		// an alert 'Command "foo" not found!?' - as if the halting action
		// after interpreting the advise command causes the query stream
		// to be reset?
#endif // BUILD_FOR_IOS

		/* We want to sleep here, but maybe not for too long.
		 * If we don't sleep, then the daemon burns up cpu when it's
		 * inactive, but if we always sleep here, then response is slow
		 * when we have a lot of transactions.
		 *
		 * Let's try this approach:  when we have data,
		 * we reset sleeptime to something small.  Then,
		 * every time we have nothing, we double it, up until
		 * an upper limit of half a second.
		 */

		if( mpp->mp_sleeptime > 0 ){
#ifdef BUILD_FOR_WINDOWS
			Sleep( 1 + mpp->mp_sleeptime/1000 );
#else
			usleep(mpp->mp_sleeptime);
#endif
			mpp->mp_sleeptime *= 1.5;
			if( mpp->mp_sleeptime > MAX_MP_SLEEPTIME )
				mpp->mp_sleeptime = MAX_MP_SLEEPTIME;
		}
	}
	/* When we have data, reset the sleeptime to something small */
	if( mpp->mp_sleeptime > 0 )
		mpp->mp_sleeptime=MIN_MP_SLEEPTIME;
	return(0);
}


/*
 * Read port *mpp repeatedly as needed until n_want_bytes
 * chars have been placed into buf.
 * BUG!!! read_expected_bytes is supposed to return the number actually
 * read (or -1 for error), but it doesn't really return the number
 * read; in fact, if the requested number all come in at the first
 * attempt, 0 is returned!?  This bug is probably harmless since
 * the return value is only sign checked...
 */

static long read_expected_bytes(QSP_ARG_DECL  Port *mpp,void *buf,ssize_t n_want_bytes)
{
	ssize_t n;
	int have=0;

	/* BUG questionable pc cast */
//fprintf(stderr,"read_expected_bytes calling read_port\n");
	while( (n=read_port(QSP_ARG  mpp,&(((char *)buf)[have]),n_want_bytes)) != n_want_bytes ){
		if( n==(-1) ){		/* why would this happen? */
			/* BUG - Is it possible that an error inside
			 * of read_port would cause the port to be
			 * deallocated?  In that case, we might be
			 * in trouble referencing the former port's name.
			 */
			sprintf(ERROR_STRING,
				"read_expected_bytes: problem reading port \"%s\"",
				mpp->mp_name);
			WARN(ERROR_STRING);
			return(-1);
		} else if( n==0 ){
//fprintf(stderr,"read_expected_bytes saw EOF?\n");
			return(-1); /* EOF */
		}
		have+=n;
		n_want_bytes-=n;
//fprintf(stderr,"n_want_bytes = %ld\n",(long)n_want_bytes);
	}
	have += n;	// be sure to count the final read
//fprintf(stderr,"read_expected_bytes succeeded\n");
	return(have);
}

/*
 * Get an long integer from the port.
 * Returns the value of the integer, or -1.
 *
 * This seems like a BUG, what if -1 is the value to be transmitted!?
 */

int32_t get_port_int32(QSP_ARG_DECL  Port *mpp)
{
	int32_t word;
	int32_t net_data;
	long n;

	n=read_expected_bytes(QSP_ARG  mpp,&net_data,sizeof(int32_t));

	if( n < 0 ){
//advise("get_port_int32:  returning BAD_PORT_LONG");
		return(BAD_PORT_LONG);
	}

	/* now decode the length */
	/* what shall the byte order be?? */
	/* how about msb first? */

//#ifdef HAVE_NTOHL
	word = ntohl(net_data);
	return(word);
//#else // ! HAVE_NTOHL
	//WARN("get_port_int32:  Sorry, no ntohl implementation!?");
	//return(0);
//#endif // ! HAVE_NTOHL

}

/*
 * Receive a text packet.  This function is called from recv_data(),
 * after that function has seen a P_TEXT code.
 * The text packets consist of a word giving the length of the string
 * in bytes, followed by the text.
 *
 * A new buffer is allocated for the text.  recv_text() returns
 * the address of this buffer, or a null pointer if some error occurs,
 * usually EOF.
 */

long recv_text(QSP_ARG_DECL  Port *mpp, Packet *pkp)
{
	long len;		/* number requested */
	long nread;		/* number read on last gulp */

	len=get_port_int32(QSP_ARG  mpp);
	if( len<=0 ) return(-1);

	// BUG? - should we check the len against a limit?
	pkp->pk_data = (char *) getbuf(len);

	// getbuf should die within itself if there's an error...
	if( pkp->pk_data == NULL ) mem_err("recv_text");

	nread = read_expected_bytes(QSP_ARG  mpp,(char *)pkp->pk_data,len);
	if( nread < 0 ){
		givbuf(pkp->pk_data);
		pkp->pk_data = NULL;
		return(-1);
	}
	// We rely upon a null-terminated string...
	pkp->pk_user_data = pkp->pk_data;
	return(nread);
} // recv_text

// If this is just recv_text, why have a separate function?

long recv_plain_file(QSP_ARG_DECL  Port *mpp, Packet *pkp)
{
	return recv_text(QSP_ARG  mpp, pkp);
}

long recv_filename(QSP_ARG_DECL  Port *mpp, Packet *pkp)
{
	return recv_text(QSP_ARG  mpp, pkp);
}

#ifdef HAVE_ENCRYPTION

#define ENCRYPTION_PREFIX_STRING	"Text:"
//#define ENCRYPTION_PREFIX_LEN		strlen(ENCRYPTION_PREFIX_STRING)
#define ENCRYPTION_PREFIX_LEN		5	// BUG? make sure this matches!

#define N_SALT_CHARS			5
#define SALT_CHAR_MIN			040	// space
#define SALT_CHAR_MAX			0176	// tilde
#define N_SALT_CHAR_CHOICES		(1+SALT_CHAR_MAX-SALT_CHAR_MIN)

static char *add_encryption_prefix(const char *t)
{
	long nneed;
	char *buf,*s;
	int i,c;

	nneed=strlen(t)+ENCRYPTION_PREFIX_LEN+N_SALT_CHARS+1;
	buf = getbuf(nneed);
	strcpy(buf,ENCRYPTION_PREFIX_STRING);

	s= &buf[ENCRYPTION_PREFIX_LEN];
	for(i=0;i<N_SALT_CHARS;i++){
		c = SALT_CHAR_MIN + (int)rn(N_SALT_CHAR_CHOICES-1);
		*s++ = (char)c;
	}
	*s=0;

	strcat(buf,t);
	return(buf);
}

// If this is just recv_enc_text, why have a separate function?

long recv_enc_file(QSP_ARG_DECL  Port *mpp, Packet *pkp)
{
	return recv_enc_text(QSP_ARG  mpp, pkp);
}

long recv_enc_text(QSP_ARG_DECL  Port *mpp, Packet *pkp)
{
	char *dbuf=NULL;
	long len;		/* number requested */
	long nread;		/* number read on last gulp */
	long n_decrypted;

	len=get_port_int32(QSP_ARG  mpp);
	if( len<=0 ){
		WARN("recv_enc_text:  bad length");
		return(-1);
	}
	// BUG?  should we limit the max length?

	pkp->pk_data =(char *)getbuf(len);
	if( pkp->pk_data == NULL ) mem_err("recv_enc_text");

	nread = read_expected_bytes(QSP_ARG  mpp,pkp->pk_data,len);
	if( nread <= 0 ){
		WARN("recv_enc_text:  error reading encrypted data");
		n_decrypted=(-1);
		goto cleanup;
	}
	dbuf = (char *)getbuf(nread);
	n_decrypted=decrypt_char_buf((uint8_t *)(pkp->pk_data),nread,dbuf,nread);
	if( n_decrypted <= 0 ){
		WARN("recv_enc_text:  error decrypting buffer");
		givbuf(dbuf);
		dbuf=NULL;
		goto cleanup;
	}

	// Verify that the packet has the proper prefix
	if( strncmp(dbuf,ENCRYPTION_PREFIX_STRING,ENCRYPTION_PREFIX_LEN) ){
		advise("recv_enc_text:  bad prefix!?");
		givbuf(dbuf);
		dbuf=NULL;
		n_decrypted = -1;
		goto cleanup;
	}

	pkp->pk_user_data = dbuf + ENCRYPTION_PREFIX_LEN + N_SALT_CHARS;
	n_decrypted -= (ENCRYPTION_PREFIX_LEN + N_SALT_CHARS);
#ifdef CAUTIOUS
	if( n_decrypted <= 0 ){
		WARN("CAUTIOUS:  recv_enc_text:  n_decrypted <= 0 after subtracting prefix length!?");
		n_decrypted= -1;
	}
#endif // CAUTIOUS

cleanup:
	givbuf(pkp->pk_data);
	pkp->pk_data=dbuf;
	return(n_decrypted);
} // end recv_enc_text
#endif /* HAVE_ENCRYPTION */

/*
 * Transmit a word.  Companion routine to get_port_word().
 */

int put_port_int32(QSP_ARG_DECL  Port *mpp,int32_t  wrd)
{
	int32_t net_data;

	if( NEEDS_RESET(mpp) ) return -1;

//#ifdef HAVE_HTONL
	net_data = htonl(wrd);
//#else // ! HAVE_HTONL
	//WARN("put_port_int32:  Sorry, no implementation of htonl!?");
	//return -1;
//#endif // ! HAVE_HTONL

	if( write_port(QSP_ARG  mpp,&net_data,sizeof(net_data)) != sizeof(net_data) ){
		/* BUG
		 * One reason that this fails is that the listening
		 * program has been killed (^C).  (on pc anyway...)
		 * We should probably do something more active here
		 * to straighten things out.
		 */
		return(-1);
	}
	else return(0);
}

/*
 * Transmit a string.  Send the code word P_TEXT, followed by a word
 * giving the length of the string, followed by the string itself.
 */
/* flag ignored, but there to be compatible w/ xmit_data() */

static void xmit_buffer(QSP_ARG_DECL  Port *mpp,const void *buf, int32_t len, int pkt_code)
{
	if( put_port_int32(QSP_ARG  mpp,PORT_MAGIC_NUMBER) == (-1) ){
		WARN("xmit_buffer:  error sending magic number");
		return;
	}

	if( put_port_int32(QSP_ARG  mpp,pkt_code) == (-1) ){
		WARN("xmit_buffer:  error sending packet code");
		return;
	}

	if( put_port_int32(QSP_ARG  mpp,len) == (-1) ){
		WARN("xmit_buffer:  error sending buffer length");
		return;
	}

	if( write_port(QSP_ARG  mpp,buf,len) != len )
		WARN("xmit_buffer:  error sending byte buffer");
}

#ifdef HAVE_STAT
static char *get_file_contents(QSP_ARG_DECL  const char *filename, struct stat *stb_p)
{
	FILE *fp;
	char *buf;

	fp=try_open(QSP_ARG  filename,"r");
	if( !fp ) return NULL;

	// get the file size
	if( fstat(fileno(fp),stb_p) < 0 ){
		tell_sys_error("fstat");
		fclose(fp);
		return NULL;
	}
	// Should we make sure it is really a plain file?
	// Allocate an extra char so we can add a null
	// terminator, if we will xmit as text.

	buf = getbuf( (long)stb_p->st_size + 1 );

	if( fread(buf,1,(long)stb_p->st_size,fp) != stb_p->st_size ){
		sprintf(ERROR_STRING,"xmit_plain_file:  error reading file %s",
			filename);
		WARN(ERROR_STRING);
		fclose(fp);
		return NULL;
	}
	// Now we have the buffer
	fclose(fp);
	buf[stb_p->st_size] = 0;	// add null terminator
	return buf;
}
#endif // HAVE_STAT

void xmit_file_as_text(QSP_ARG_DECL  Port *mpp,const void *_filename,
							int flag)
{
#ifdef HAVE_STAT
	const char *filename;
	struct stat statb;
	char *buf;

	filename=(const char *)_filename;
	buf = get_file_contents(QSP_ARG  filename,&statb);
	if( buf == NULL ) return;

	// add null termination
	buf[ statb.st_size ] = 0;

	xmit_buffer(QSP_ARG  mpp,buf,(int32_t)(1+statb.st_size),P_TEXT);

#else // ! HAVE_STAT
	WARN("Sorry, don't know how to transmit a file without fstat to determine file size!?");
#endif // ! HAVE_STAT
}

void xmit_plain_file(QSP_ARG_DECL  Port *mpp,const void *_filename,int flag)
{
#ifdef HAVE_STAT
	const char *filename;
	struct stat statb;
	char *buf;
#ifdef XMIT_FILENAME		// not now...
	const char *remote_name;
#endif // XMIT_FILENAME

	filename=(const char *)_filename;

#ifdef XMIT_FILENAME		// not now...

	// The set_port_output_filename command has been executd
	// on the receiver, with a command sent first...
	// We'd like to execute it on the sender, and then
	// transmit the filename ahead of the file data.
	//
	// This is problematic, however, because we don't know the local
	// directory structure on the receiver.

	if( mpp->mp_output_filename == NULL ){
		sprintf(ERROR_STRING,
	"Remote filename not specified, not transmitting file %s.",
			filename);
		WARN(ERROR_STRING);
		advise("Use port_output_file command to specify the remote filename.");
		return;
	}

	remote_name = mpp->mp_output_filename;
	xmit_buffer(QSP_ARG  mpp,remote_name,1+(int32_t)strlen(remote_name),
		P_FILENAME);
	rls_str(remote_name);
	mpp->mp_output_filename = NULL;
#endif // XMIT_FILENAME

	buf = get_file_contents(QSP_ARG  filename,&statb);
	if( buf == NULL ) return;

	xmit_buffer(QSP_ARG  mpp,buf,(int32_t)statb.st_size,P_PLAIN_FILE);


#else // ! HAVE_STAT
	WARN("Sorry, don't know how to transmit a file without fstat to determine file size!?");
#endif // ! HAVE_STAT
}

void xmit_filename(QSP_ARG_DECL  Port *mpp,const void *_filename,int flag)
{
	xmit_buffer(QSP_ARG  mpp,_filename,(int32_t)(1+strlen(_filename)),
			P_FILENAME);
}

void xmit_text(QSP_ARG_DECL  Port *mpp,const void *text,int flag)
{
	xmit_buffer(QSP_ARG  mpp,text,(int32_t)(1+strlen(text)),P_TEXT);
}

#ifdef HAVE_ENCRYPTION

static uint8_t *encrypt_text_buffer(QSP_ARG_DECL  const void *text,size_t *siz_p)
{
	// first encrypt the text
	size_t max_enc_size;
	uint8_t *encbuf;

	SET_OUTPUT_SIZE( max_enc_size , 1+strlen(text) );
	encbuf = (uint8_t *)getbuf(max_enc_size);

	*siz_p = encrypt_char_buf(text,1+strlen(text),
				encbuf, max_enc_size);

	if( *siz_p <= 0 ){
		WARN("encrypt_text_buffer:  encryption error");
		givbuf(encbuf);
		encbuf=NULL;
	}
	return encbuf;
}

static void xmit_enc_text_packet(QSP_ARG_DECL  Port *mpp, const void *text,
					int flag, Packet_Type code )
{
	uint8_t *encbuf;
	size_t enc_size;
	char *s;

	// To make sure that we don't try to interpret
	// garbage, we prefix the caller's message
	// with a fixed string (to validate), and a random salt.

	s = add_encryption_prefix((const char *)text);
	encbuf = encrypt_text_buffer(QSP_ARG  s,&enc_size);
	givbuf(s);

	if( encbuf == NULL ) return;
	// BUG xmit_buffer should take arg of type size_t!?
	xmit_buffer(QSP_ARG  mpp,encbuf,(int32_t)enc_size,code);
	givbuf(encbuf);
}

void xmit_enc_text(QSP_ARG_DECL  Port *mpp,const void *text,int flag)
{
	xmit_enc_text_packet(QSP_ARG  mpp, text, flag, P_ENCRYPTED_TEXT);
}

void xmit_auth(QSP_ARG_DECL  Port *mpp,const void *text,int flag)
{
	xmit_enc_text_packet(QSP_ARG  mpp, text, flag, P_AUTHENTICATION);
}

static void xmit_enc_file_packet(QSP_ARG_DECL  Port *mpp, const void *_filename,
					int flag, Packet_Type code )
{
#ifdef HAVE_STAT
	char *buf;
	struct stat statb;
	const char *filename;
	uint8_t *encbuf;
	size_t enc_size;
	char *s;

	filename=(const char *)_filename;
	buf = get_file_contents(QSP_ARG  filename,&statb);
	if( buf == NULL ) return;

	// add null termination
	buf[ statb.st_size ] = 0;

	// Now encrypt the contents

	s = add_encryption_prefix((const char *)buf);
	encbuf = encrypt_text_buffer(QSP_ARG  s,&enc_size);
	givbuf(s);

	xmit_buffer(QSP_ARG  mpp,encbuf,(int32_t)enc_size,P_ENCRYPTED_FILE);
	givbuf(encbuf);

#else // ! HAVE_STAT
	WARN("Sorry, don't know how to transmit (encrypted) a file without fstat to determine file size!?");
#endif // ! HAVE_STAT
}

void xmit_enc_file(QSP_ARG_DECL  Port *mpp,const void *_filename,int flag)
{
	xmit_enc_file_packet(QSP_ARG  mpp, _filename, flag, P_ENCRYPTED_FILE );
}

void xmit_enc_file_as_text(QSP_ARG_DECL  Port *mpp,const void *_filename,
							int flag)
{
	xmit_enc_file_packet(QSP_ARG  mpp, _filename, flag, P_ENC_FILE_AS_TEXT );
}

#endif /* HAVE_ENCRYPTION */

ssize_t write_port(QSP_ARG_DECL  Port *mpp,const void *buf,u_long  n)
{
	ssize_t n2;

	if( NEEDS_RESET(mpp) ) return -1;

#ifdef BUILD_FOR_WINDOWS

	n2=send(mpp->mp_sock,buf,(int)n,0);
	if( n2 == SOCKET_ERROR ){
		report_windows_error();
		return -1;
	}

#else // ! BUILD_FOR_WINDOWS

	n2=write(mpp->mp_sock,buf,(int)n);
	if( n2 < 0 ){
		int last_err=errno;
		tell_sys_error("(write_port) write:");
		// We get SIGPIPE when the other end is shut down...
		if( last_err == EPIPE ){
			sprintf(ERROR_STRING,"Write error on port %s",
				mpp->mp_name);
			WARN(ERROR_STRING);
			// Close the port
			// We don't necessarily want to reconnect here?
			// Maybe we need to clean up from the error, and
			// then reconnect?

			//reset_port(QSP_ARG  mpp);
			SET_PORT_FLAG_BITS(mpp,PORT_NEEDS_RESET);
		}
	}

#endif // ! BUILD_FOR_WINDOWS

	return(n2);
}

/*
 * Try to read n characters from a port.
 * Returns number of characters read, 0 for EOF, -1 for error.
 */


ssize_t read_port(QSP_ARG_DECL  Port *mpp,void *buf,u_long  n)
{
	ssize_t n2;

//fprintf(stderr,"read_port calling wait_for_data...\n");
	if( wait_for_data(QSP_ARG  mpp) < 0 ){
		// Should we return something different if HALTING?
		return(-1);
	}

#ifdef BUILD_FOR_WINDOWS

	n2=recv(mpp->mp_sock,buf,(int)n,0);

	if( n2 == SOCKET_ERROR ){
		report_windows_error();
		return -1;
	}

#else	// ! BUILD_FOR_WINDOWS

//fprintf(stderr,"read_port calling read...\n");
	n2=read(mpp->mp_sock,buf,(int)n);

	if( n2 < 0 ){
		if( errno != EINTR ){
			tell_sys_error("(read_port) read");
			WARN("error reading stream packet");
			return(-1);
		} else if( errno == 0 ){		/* EOF */
advise("read_port:  errno is zero after read failed - should this happen?????");
			goto eof_encountered;
		} else {
			WARN("read_port:  unexpected read error");
		}
	}

#endif	// ! BUILD_FOR_WINDOWS

	if( n2 == 0 ){		/* EOF */
eof_encountered:
		if( verbose ){
			sprintf(ERROR_STRING,"EOF encountered on port \"%s\"",
				mpp->mp_name);
			advise(ERROR_STRING);
		}
		/* we used to close the linked port here,
		 * but now we take care of that in reset_port()
		 */
		return(0);
	}
#ifdef CAUTIOUS
if( n2 == 0 ) WARN("CAUTIOUS:  read_port() returning 0 normally!?");
#endif /* CAUTIOUS */
	return(n2);
}

/* Shutdown a port and then release it's resources */

void close_port(QSP_ARG_DECL  Port *mpp)
{
#ifdef BUILD_FOR_WINDOWS
	closesocket(mpp->mp_sock);

	if( mpp->mp_o_sock != (-1) )		// BUG is that the right code?
		closesocket(mpp->mp_o_sock);
#else
	close(mpp->mp_sock);

	if( mpp->mp_o_sock != (-1) ){
		close(mpp->mp_o_sock);
	}
#endif

	mpp->mp_sock = (-1);
	mpp->mp_o_sock = (-1);

	givbuf((char *)mpp->mp_addrp);
	mpp->mp_addrp = NULL;
	// delport frees the name.
	if( mpp->mp_flags & PORT_REDIR ){
		// defer cleanup until the last read attempt
		// That is sensible if the close command
		// itself came over the port (e.g., remote closure
		// of a server port), but doesn't make sense
		// for a client closing its own port...
		mpp->mp_flags |= PORT_ZOMBIE;
	} else {
		delport(QSP_ARG  mpp);
	}
}

