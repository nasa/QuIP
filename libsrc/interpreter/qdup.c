#include "quip_config.h"

char VersionId_interpreter_qdup[] = QUIP_VERSION_STRING;

/**/
/**		input and output stuff		**/
/**/

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_SYS_FILIO_H
#include <sys/filio.h>
#endif

#include "query.h"
#include "debug.h"
#include "items.h"
#include "savestr.h"
#include "node.h"
#include "getbuf.h"

#ifdef HAVE_HISTORY
#include "history.h"
#endif /* HAVE_HISTORY */


static int need_to_chunk(const char *);


/* dup stuff */

void end_dupline(SINGLE_QSP_ARG_DECL)
{
	FILE *fp;

	fp = THIS_QSP->qs_query[QLEVEL].q_dupfile;
	
#ifdef CAUTIOUS
	if( ! IS_DUPING ){
		WARN("CAUTIOUS:  NOT duping!?");
		return;
	}
#endif

	fputs("\r",fp);
	fflush(fp);
	SET_Q_FLAG( THIS_QSP, Q_FIRST_WORD );
}

// For reasons that are no longer clear, this code quotes strings that contain
// a space...  This would only make sense if enclosing quotes had already
// been stripped.

static int need_to_chunk(const char *s)		/* does s contain non-quoted spaces? */
{
#ifdef FOOBAR
	int insgl=0, indbl=0;
	
	while( *s ){
		if( !(insgl|indbl) ){
			if( *s == ' ' || *s == '\t' )
				return(1);
			else if( *s == '\'' ) insgl=1;
			else if( *s == '"' ) indbl=1;
		} else if( insgl ){
			if( *s == '\'' ) insgl=0;
		} else if( indbl ){
			if( *s == '"' ) indbl=0;
		}
		s++;
	}
	return(0);
#else
	return(0);	// never chunk
#endif
}

void dup_word(QSP_ARG_DECL  const char *s)
{
	int chunkme;
	FILE *fp;

#ifdef THREAD_SAFE_QUERY
	// This null test wasn't needed until we tried to exit a thread...
	if( s == NULL ) return;
#endif /* THREAD_SAFE_QUERY */

	fp = THIS_QSP->qs_query[QLEVEL].q_dupfile;
	
#ifdef CAUTIOUS
	if( ! IS_DUPING ){
		WARN("CAUTIOUS:  NOT duping!?");
		return;
	}
#endif

/*
sprintf(ERROR_STRING,"duping string \"%s\"",s);
WARN(ERROR_STRING);
*/
	if( ! FIRST_WORD_ON_LINE )
		fputs(" ",fp);
	// Why do we ever need to chunk???
	chunkme = need_to_chunk(s);
	
	if( chunkme ) fputs("'",fp);
	fputs(s,fp);
	if( chunkme ) fputs("'",fp);
	CLR_Q_FLAG( THIS_QSP, Q_FIRST_WORD );
}


#ifndef MAC
#ifdef HAVE_HISTORY
#ifdef TTY_CTL

Query *hist_select(QSP_ARG_DECL char *buf,const char* pline)
{
	const char *s;
	Query *qp;

	qp = &THIS_QSP->qs_query[QLEVEL];
	s=get_sel(QSP_ARG  pline,qp->q_file,stderr);
	if( s==NULL ){			/* ^D */
		if( QLEVEL > 0 ){
			popfile(SINGLE_QSP_ARG);
			qp--;
			return(qp);
		} else {
			advise("EOF");
			nice_exit(0);
		}
	}
	strcpy(buf,s);
	qp->q_havtext=1;
	qp->q_lbptr=buf;

	return(qp);
}

#endif /* TTY_CTL */
#endif /* HAVE_HISTORY */
#endif /* ! MAC */

void ql_debug(SINGLE_QSP_ARG_DECL)
{
	sprintf(ERROR_STRING,"qlevel = %d",QLEVEL);
	advise(ERROR_STRING);
	sprintf(ERROR_STRING,"q_file = 0x%lx",(u_long)THIS_QSP->qs_query[QLEVEL].q_file);
	advise(ERROR_STRING);
}

int dupout(QSP_ARG_DECL  FILE *fp)			/** save input text to file fp */
{
	if( IS_DUPING ){
		WARN("already dup'ing");
		return(-1);
	} else {
		THIS_QSP->qs_query[QLEVEL].q_dupfile=fp;
		return(0);
	}
}


void savechar(QSP_ARG_DECL  Query *qp,int c)
{
	char buf[2];

	buf[0]=c;
	buf[1]=0;
	savetext(QSP_ARG  qp,buf);
}

void unsavechar(QSP_ARG_DECL  Query *qp, int c)
{
	int n;

	while( qp != (&THIS_QSP->qs_query[0]) && (qp-1)->q_saving ){
		qp--;
#ifdef CAUTIOUS
		if( qp->q_text == NULL ){
			ERROR1("CAUTIOUS:  unsavechar:  no saved text!?");
		}
#endif /* CAUTIOUS */

		n=strlen(qp->q_text);

#ifdef CAUTIOUS
		if( n <= 0 ) ERROR1("CAUTIOUS:  unsavechar:  no saved text!?");

#endif /* CAUTIOUS */
		n--;

#ifdef CAUTIOUS
		if( qp->q_text[n] != c ) 
			ERROR1("CAUTIOUS:  unsavechar:  last char not what expected!?");
#endif /* CAUTIOUS */

		qp->q_text[n] = 0; 
		qp->q_txtfree ++;
	}
}

/*
 *	Save as string for later interpretation.
 *
 *	We save text for reinterpreting in loops;
 *	we save parsed words, and quotes (which are
 *	ignored when parsing words).
 *
 *	We save at levels down the stack...
 *	(When would we ever save at more than one level?  nested loops?)
 */

void savetext(QSP_ARG_DECL  Query *qp,const char* buf)
	/* query structure pointer */
	/* text to save */
{
	int n_more;
	char *str;
Query *first_qp;

first_qp=qp;

	n_more=strlen(buf)+1;
	while( qp != (&THIS_QSP->qs_query[0]) && (qp-1)->q_saving ){
		qp--;
#ifdef CAUTIOUS
		if( qp->q_text == NULL ){
			int index;
index = (first_qp-&THIS_QSP->qs_query[0]);
sprintf(ERROR_STRING, "savetext started at level %d", index);
advise(ERROR_STRING);
index = (qp-&THIS_QSP->qs_query[0]);
sprintf(ERROR_STRING, "savetext trouble at level %d", index);
advise(ERROR_STRING);

			ERROR1("CAUTIOUS:  whoa, null text buffer!!!");
		}
#endif /* CAUTIOUS */

		/* See if we have enough room for this text,
		 * and grow the buffer if necessary
		 */
		while( n_more > qp->q_txtfree ){
			qp->q_txtsiz += LOOPSIZE;
			qp->q_txtfree += LOOPSIZE;
			str=(char*) getbuf(qp->q_txtsiz);
			if( str==NULL ) mem_err("save_text");
			strcpy(str,qp->q_text);
			givbuf(qp->q_text);
			qp->q_text=str;
		}

		strcat(qp->q_text,buf);
		qp->q_txtfree -= n_more;

#ifdef FOOBAR
		/* these next two lines are here since we're
		 * already executing at the higher level
		 *
		 * For some reason, when we store the last "end",
		 * that gets stored here too???
		 * Why did that not happen before?
		 */

		// This gets called over and over again,
		// for every character in a saved word...
		// very inefficient!?
		// Maybe we could do this from rd_word???

//sprintf(ERROR_STRING,"savetext:  setting lbptr to 0x%lx (\"%s\")",
//(long)(qp+1)->q_lbptr,(qp+1)->q_lbptr);
//advise(ERROR_STRING);
		qp->q_lbptr = (qp+1)->q_lbptr;
		qp->q_havtext = (qp+1)->q_havtext;
#endif /* FOOBAR */
	}
} // end savetext

void sync_lbptrs(QSP_ARG_DECL  Query *qp)
{
	while( qp != (&THIS_QSP->qs_query[0]) && (qp-1)->q_saving ){
		qp--;
		qp->q_lbptr = (qp+1)->q_lbptr;
		qp->q_havtext = (qp+1)->q_havtext;
	}
}

