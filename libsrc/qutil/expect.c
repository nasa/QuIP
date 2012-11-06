
#include "quip_config.h"

char VersionId_qutil_expect[] = QUIP_VERSION_STRING;

#include <stdio.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>	/* atoi() */
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* usleep */
#endif

#include "query.h"
#include "serbuf.h"
#include "serial.h"

#ifdef DEBUG
static u_long sb_debug=0;

#define CHECK_DEBUG_INITED						\
				if( sb_debug == 0 ){			\
					sb_debug = add_debug_module(QSP_ARG  "expect_string");	\
				}
#endif /* DEBUG */

static int i_string=0;
#define N_STRINGS	10

char *printable_version(int c)
{
	static char str[N_STRINGS][4];
	char *s;

	s=str[i_string];
	i_string++;
	i_string %= N_STRINGS;

	if( c == '\n' )
		strcpy(s,"\\n");
	else if( c == '\r' )
		strcpy(s,"\\r");
	else if( c == '\b' )
		strcpy(s,"\\b");
	else if( c == '\t' )
		strcpy(s,"\\t");
	else if( c == '\f' )
		strcpy(s,"\\f");
	else if( c == '\v' )
		strcpy(s,"\\v");
	else if( c == '\0' )
		strcpy(s,"\\0");
	else if( iscntrl(c) )
		sprintf(s,"^%c",'A'+c-1);
	else if( isprint(c) )
		sprintf(s,"%c",c);
	else
		sprintf(s,"\%3x",c);
			
	return(s);
}

char *printable_string(const char *s)
{
	static char pbuf[LLEN];
	char *p;

	pbuf[0] = 0;
	while(*s){
		p = printable_version(*s);
		strcat(pbuf,p);
		s++;
	}
	return(pbuf);
}

void show_buffer(Serial_Buffer *sbp)
{
	u_char *s;
	int i;

/*sprintf(ERROR_STRING,"show_buffer 0x%lx, %d received, %d scanned",(u_long)sbp,sbp->sb_n_recvd,sbp->sb_n_scanned);*/
/* ADVISE(ERROR_STRING);*/
	s=sbp->sb_buf+sbp->sb_n_scanned;
	while( *s ){
		fputs(printable_version(*s),stderr);
		if( *s == '\n' ) putc('\n',stderr);
		s++;
	}
	fputs("#end\n",stderr);

	for(i=0;i<sbp->sb_n_scanned;i++)
		putc(' ',stderr);
	putc('^',stderr);
	putc('\n',stderr);
	fflush(stderr);
}

void reset_buffer(Serial_Buffer *sbp)
{
/*ADVISE("RESET_BUFFER"); */
	sbp->sb_buf[0]=0;
	sbp->sb_n_recvd = 0;
	sbp->sb_n_scanned = 0;
}

/* replenish_buffer will read up to max_expected chars, but is not guaranteed to read any.
 * the characters are placed in the response buffer.
 */
		
int replenish_buffer(QSP_ARG_DECL  Serial_Buffer *sbp,int max_expected)
{
	int n;

#ifdef DEBUG
	CHECK_DEBUG_INITED
#endif

	/* BUG? are we guaranteed that this won't overflow? */
	if( sbp->sb_n_recvd + max_expected >= BUFSIZE ){
		sprintf(ERROR_STRING,"trouble?  n_recvd = %d, max_expected = %d. BUFSIZE = %d",
			sbp->sb_n_recvd,max_expected,BUFSIZE);
		ADVISE(ERROR_STRING);
	}
	n = recv_somex(QSP_ARG  sbp->sb_fd, &sbp->sb_buf[sbp->sb_n_recvd], BUFSIZE-sbp->sb_n_recvd, max_expected);
//#ifdef DEBUG
//if( (debug & sb_debug) ){
//sprintf(msg_str,"replenish_buffer:  expected up to %d chars, received %d",
//max_expected,n);
//prt_msg(msg_str);
//}
//#endif /* DEBUG */

	sbp->sb_n_recvd += n;		/* Can n_recvd be >= BUFSIZE?? shouldnt... BUG? */
	sbp->sb_buf[sbp->sb_n_recvd]=0;	/* make sure string is NULL-terminated */
	return(n);
}

#define MAX_TRIES	5000		/* 5 seconds? */

/*
 * 5000 sleeps of 200 usec, should be about 1 second?
 */

int buffered_char(QSP_ARG_DECL  Serial_Buffer *sbp)
{
	u_char *buf;
	int n_tries=0;

#ifdef DEBUG
	CHECK_DEBUG_INITED
#endif

	buf = sbp->sb_buf + sbp->sb_n_scanned;

#ifdef DEBUG
/*
if( debug & sb_debug ){
if( *buf != 0 ) prt_msg_frag(".");
else prt_msg_frag(",");
}
*/
#endif /* DEBUG */

/*sprintf(ERROR_STRING,"buffered_char 0x%lx, %d received, %d scanned",(u_long)sbp,sbp->sb_n_recvd,sbp->sb_n_scanned);*/
/*ADVISE(ERROR_STRING); */

	if( *buf == 0 ){
		int n;
		n=n_serial_chars(QSP_ARG  sbp->sb_fd);
		while(n==0 && n_tries < MAX_TRIES ){
			usleep(200);
			n=n_serial_chars(QSP_ARG  sbp->sb_fd);
			n_tries++;
		}
		if( n == 0 ){
			if( n_tries >= MAX_TRIES ){
				NWARN("buffered_char:  No readable chars from serial device, check power");
				return(-1);
			}
		}
//#ifdef DEBUG
//if( debug & sb_debug ){
//sprintf(ERROR_STRING,"buffered char 0x%lx:  there are %d new chars available, calling replenish_buffer",(u_long) sbp,n);
//ADVISE(ERROR_STRING);
//}
//#endif /* DEBUG */
		replenish_buffer(QSP_ARG  sbp,n);
/*show_buffer(sbp); */
	}

//#ifdef DEBUG
//if( debug & sb_debug ){
//sprintf(ERROR_STRING,"buffered_char returning '%s' (0x%x)",printable_version(*buf),*buf);
//ADVISE(ERROR_STRING);
//}
//#endif /* DEBUG */
	return(*buf);
}


#ifdef FOOBAR
/* We used to get dropped chars (from the knox switcher) before we started
 * reading the echo char-by-char as the command is sent...
 *
 * expect_char() needs a timeout arg...
 */

static int expect_char(Serial_Buffer *sbp, int expected)
{
	int c;

//#ifdef DEBUG
//if( debug & sb_debug ){
//if( expected > 0 ){
//sprintf(ERROR_STRING,"expect_char:  '%s' (0x%x)",printable_version(expected),expected);
//ADVISE(ERROR_STRING);
//} else {
//ADVISE("expect_char:  looking for any char after mismatch");
//}
//}
//#endif /* DEBUG */
	c = buffered_char(QSP_ARG  sbp);
	if( c < 0 ) return(c);

	sbp->sb_n_scanned ++;

	if( c != expected ){
		if( verbose ){
			sprintf(ERROR_STRING,"expect_char:  expected 0x%x ('%s'), but saw 0x%x ('%s')",
					expected,printable_version(expected),c,printable_version(c));
			NWARN(ERROR_STRING);
		}
		/* pretend the char hasn't been read... */
		sbp->sb_n_scanned --;
		return(-1);
	}
	
	return(c);
} /* end expect_char() */
#endif /* FOOBAR */

/* This used to be called read_until, and only read until the first char
 * was found.  That should be called read_until_char !?
 */

void read_until_string(QSP_ARG_DECL  char *dst,Serial_Buffer *sbp, const char *marker,
							Consume_Flag consume)
{
	int c;
	const char *w;

	/* BUG?  should we be careful about not overrunning dst? */

	w=marker;
	while( 1 ){

		c = buffered_char(QSP_ARG  sbp);
		sbp->sb_n_scanned ++;
		*dst++ = c;

		if( c == *w ){	/* what we expect? */
			w++;
			if( *w == 0 ){	/* done? */
				/* We have put the expected string in the buffer,
				 * now get rid of it.
				 */
				dst -= strlen(marker);
				/* If the consume flag is not set,
				 * then we leave the marker in the
				 * input buffer to be read later.
				 */
				if( consume == PRESERVE_MARKER )
					sbp->sb_n_scanned -= strlen(marker);
				*dst=0;
				return;
			}
		} else {
			w=marker;	/* back to the start */
		}
	}
	/* NOTREACHED */
}

/* 
 * If the remote device drops one char from the expected string, we catch that here,
 * but we don't handle well the case of unexpected chars...
 *
 * expect_string() now returns the received string, or NULL for a perfect match.
 */

char *expect_string(QSP_ARG_DECL  Serial_Buffer *sbp, const char *expected_str)
{
	int i_e;		/* index of expected char */
	int i_r;		/* index of received char */
	int expected, received;
	static char rcvd_str[LLEN];	/* BUG - is this big enough? */
	int mismatch=0;
	int reading=1;

#ifdef DEBUG
	CHECK_DEBUG_INITED
#endif

	expected=0;		// pointless initialization to quiet compiler

#ifdef DEBUG
if( debug & sb_debug ){
//int n;
sprintf(ERROR_STRING,"expect_string(\"%s\");",printable_string(expected_str));
ADVISE(ERROR_STRING);
//n = n_serial_chars(QSP_ARG  sbp->sb_fd);
//sprintf(ERROR_STRING,"%d chars available on serial port",n);
//ADVISE(ERROR_STRING);
}
#endif /* DEBUG */
	i_r=i_e=0;
	while( reading ){
		if( !mismatch ) {
			// expected always gets set the first time through.
			expected = expected_str[i_e];
			i_e++;
		}

		received = buffered_char(QSP_ARG  sbp);
		if( received < 0 ){	/* no chars? */
#ifdef DEBUG
if( debug & sb_debug ){
sprintf(msg_str,"expect_string:  no buffered char!?");
ADVISE(msg_str);
}
#endif /* DEBUG */
			goto finish_expect_string;
		} else if( received == '\n' || received == '\r' ){
			if( mismatch )
				goto finish_expect_string;
		}


		sbp->sb_n_scanned ++;

		/* is this the first mismatch? */
		if( (!mismatch) ){
			if( received != expected ){
#ifdef DEBUG
if( debug & sb_debug ){
sprintf(msg_str,"expect_string:  received 0x%x, expected 0x%x",received,expected);
ADVISE(msg_str);
}
#endif /* DEBUG */

				/* If we have a partial match already, copy into the
				 * received buffer...
				 */
				if( i_e > 1 ){
					strncpy(rcvd_str,expected_str,i_e-1);
					i_r = i_e-1;
				}
				mismatch=1;
				sbp->sb_n_scanned --;
			}
		} else {
//if( mismatch ){
//sprintf(msg_str,"expect_string:  received '%s' after mismatch",printable_version(received));
//ADVISE(msg_str);
//}
			rcvd_str[i_r] = received;
			i_r++;
			if( i_r >= LLEN ){
				sprintf(ERROR_STRING,"expect_string:  too many received characters (%d max)",LLEN-1);
				NWARN(ERROR_STRING);
			}
		}

		/* Stop if we reach the end of the expected string with no error */
		/* Otherwise read a complete line */
		if( mismatch == 0 && expected_str[i_e] == 0 )
			reading=0;
	}

finish_expect_string:

	rcvd_str[i_r] = 0;	/* terminate string */

#ifdef DEBUG
//if( debug & sb_debug ){ prt_msg("#"); }
#endif /* DEBUG */

	if( mismatch ) {
#ifdef DEBUG
if( debug & sb_debug ){
sprintf(ERROR_STRING,"expect_string:  expected \"%s\", but received \"%s\"",
	printable_string(expected_str),printable_string(rcvd_str));
ADVISE(ERROR_STRING);
}
#endif /* DEBUG */
		return(rcvd_str);
	} else {
#ifdef DEBUG
if( debug & sb_debug ){
sprintf(ERROR_STRING,"expect_string:  received expected string \"%s\";",
	printable_string(expected_str));
ADVISE(ERROR_STRING);
}
#endif /* DEBUG */
		return(NULL);
	}

} /* end expect_string() */

/* What does get_token do???
 * The original version filled the token buffer with alphanumeric characters,
 * up until a white space character.  But for the knox switcher condensed map
 * format, the numbers are packed with letters.  So we only want to pack digit
 * characters.  We may have to revisit this if we have a device that transmists hex...
 */

#define MAX_TOKEN_CHARS	3

static int get_token(QSP_ARG_DECL  Serial_Buffer *sbp, char *token)
{
	int scanning;
	int n_scanned=0;
	int i=0;
	u_char *s;

	scanning=1;
	token[0]=0;
	s = &sbp->sb_buf[sbp->sb_n_scanned];

	while(scanning){

#ifdef DEBUG
/*
if( debug & sb_debug ){
sprintf(ERROR_STRING,"get_token:  input string is \"%s\"",s);
ADVISE(ERROR_STRING);
}
*/
#endif /* DEBUG */
		if( *s == 0 ){
#ifdef DEBUG
/*
if( debug & sb_debug ){
sprintf(ERROR_STRING,"get_token fetching more input, position = %d",s-sbp->sb_buf);
ADVISE(ERROR_STRING);
}
*/
#endif /* DEBUG */
			replenish_buffer(QSP_ARG  sbp,MAX_TOKEN_CHARS);
		} else if( isdigit(*s) ){
			/* Store this char */
			if( i < MAX_TOKEN_CHARS ){
				token[i] = *s;
				s++;
				i++;
				n_scanned++;
#ifdef DEBUG
/*
if( debug & sb_debug ){
sprintf(ERROR_STRING,"get_token:  token = \"%s\", \"%s\" left to scan",token,s);
ADVISE(ERROR_STRING);
}
*/
#endif /* DEBUG */
			} else {
				sprintf(ERROR_STRING,"get_token:  too many token chars!?");
				NWARN(ERROR_STRING);
				s++;
				n_scanned++;
			}
		} else {
			if( strlen(token) == 0 ){
				NWARN("get_token:  non-digit seen before any digits!?");
				s++;
				n_scanned++;
			} else {
				/* non-digit delimits end of token */
				token[i]=0;
				scanning=0;
			}
		}
	}
#ifdef DEBUG
if( debug & sb_debug ){
sprintf(ERROR_STRING,"get_token returning \"%s\" after scanning %d chars",token,n_scanned);
ADVISE(ERROR_STRING);
}
#endif /* DEBUG */
	sbp->sb_n_scanned += n_scanned;

	return(n_scanned);
}

int get_number( QSP_ARG_DECL  Serial_Buffer *sbp )
{
	int n_tok;
	char token[MAX_TOKEN_CHARS+1];

	n_tok = get_token(QSP_ARG  sbp,token);
	if( n_tok > 0 ){	/* success */
		return( atoi(token) );
	}
	NWARN("get_number:  no token!?");
/* show_buffer(sbp); */
	return(-1);	/* BUG?  can the jukebox ever return a negative number? */
}

/* this has the calling sequence like the old expect_string(), but prints a better error msg */

void expected_response( QSP_ARG_DECL  Serial_Buffer *sbp, const char *expected_str )
{
	const char *s;

	s=expect_string(QSP_ARG  sbp,expected_str);
	if( s != NULL ){
		NWARN("response mismatch");
		sprintf(ERROR_STRING,"Expected string:  \"%s\"",printable_string(expected_str));
		ADVISE(ERROR_STRING);
		sprintf(ERROR_STRING,"Received string:  \"%s\"",printable_string(s));
		ADVISE(ERROR_STRING);
	}
}

