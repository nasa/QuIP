%{

#include "quip_config.h"

/* generalized item sequencer */

#include <stdio.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "item_type.h"
#include "seq.h"
#include "gmovie.h"

typedef union {
	Seq *yysp;
	void *yyvp;
	int yyi;
} YYSTYPE;

#define YYSTYPE_IS_DECLARED		/* needed on 2.6 machine? */

// This attempts to do what is accomplished with the bison
// directive name-prefix, but doesn't suffer from the deprecated
// syntax warnings resulting from the change in bison 3.0
// (see nexpr.y, vectree.y)
#define YACC_HACK_PREFIX	seq
#include "yacc_hack.h"	// change yy prefix to seq_

//int yylex(SINGLE_QSP_ARG_DECL);
//#define YYLEX_PARAM SINGLE_QSP_ARG

// For the time being a single signature, regardless of THREAD_SAFE_QUERY
static int yylex(YYSTYPE *yylval_p, Query_Stack *qsp);

//#ifdef THREAD_SAFE_QUERY
//#define YYPARSE_PARAM qsp	/* gets declared void * instead of Query_Stream * */
///* For yyerror */
//#define YY_(msg)	QSP_ARG msg
//#endif /* THREAD_SAFE_QUERY */


int yyerror(Query_Stack *qsp,  const char *);


// BUG global not thread-sage
static int refno=1;
static const char *ipptr;
static int ended;
static Seq *final_mviseq;

static void _null_show_func(QSP_ARG_DECL  void *vp)
{
	warn("no sequence show function defined");
}

static void * _null_get_func(QSP_ARG_DECL  const char *name)
{
	warn("no sequence item get function defined");
	return(NULL);
}

static void _null_rev_func(QSP_ARG_DECL  void *vp)
{
	warn("no reverse function defined");
}

static int _null_init_func(QSP_ARG_DECL  void *vp)
{
	warn("no init function defined");
	return(0);
}

static void _null_wait_func(SINGLE_QSP_ARG_DECL)
{
	warn("no wait function defined");
}

static void _null_ref_func(QSP_ARG_DECL  void *vp)
{
	warn("no reference function defined");
}

static Seq_Module null_seq_module = {
	_null_get_func,
	_null_init_func,
	_null_show_func,
	_null_rev_func,
	_null_wait_func,
	_null_ref_func
};

static Seq_Module *the_smp=(&null_seq_module);

static void init_seq_struct(Seq *sp)
{
	sp->seq_flags=SEQFREE;
	sp->seq_refcnt=0;
	sp->seq_first=NULL;
	sp->seq_next=NULL;
	sp->seq_count=0;
}

#define unnamed_seq() _unnamed_seq(SINGLE_QSP_ARG)

static Seq *_unnamed_seq(SINGLE_QSP_ARG_DECL)	/* return an unused seq. struct */
{
	register Seq *sp;

	sp = (Seq *)getbuf(sizeof(*sp));

	if( sp == NULL )
		error1("no more memory for sequences");

	sp->seq_name=NULL;

	init_seq_struct(sp);
	return(sp);
}

#define joinseq(s1,s2) _joinseq(QSP_ARG  s1,s2)

static Seq *_joinseq(QSP_ARG_DECL  Seq *s1,Seq *s2)		/* pointer to concatenation */
{
	register Seq *sp;

	sp=unnamed_seq();
	if( sp!= NULL ){
		sp->seq_first=s1;
		sp->seq_next=s2;
		sp->seq_count=1;
		sp->seq_flags=SUPSEQ;

		s1->seq_refcnt++;
		s2->seq_refcnt++;
		sp->seq_refcnt++;

	}
	return(sp);
}

#define reptseq(cnt,seqptr) _reptseq(QSP_ARG  cnt,seqptr)

static Seq *_reptseq(QSP_ARG_DECL  int cnt,Seq *seqptr)	/* pointer to repetition */
{
	register Seq *sp;

	sp=unnamed_seq();
	if( sp!=NULL ){
		sp->seq_first=seqptr;
		sp->seq_count=(short)cnt;
		sp->seq_flags=SUPSEQ;
		seqptr->seq_refcnt++;
		sp->seq_refcnt++;
	}
	return(sp);
}

#define revseq(seqptr) _revseq(QSP_ARG  seqptr)

static Seq *_revseq(QSP_ARG_DECL  Seq *seqptr)
{
	register Seq *sp;

	sp=unnamed_seq();
	if( sp!=NULL ){
		sp->seq_first=seqptr;
		sp->seq_count = -1;
		sp->seq_flags=SUPSEQ;
		seqptr->seq_refcnt++;
		sp->seq_refcnt++;
	}
	return(sp);
}

#define makfrm(cnt,vp) _makfrm(QSP_ARG  cnt,vp)

static inline Seq *_makfrm(QSP_ARG_DECL  int cnt,void *vp)	/* get a new link for this frame */
{
	register Seq *sp;

	sp=unnamed_seq();
	if( sp!=NULL ){
		sp->seq_count=(short)cnt;
		sp->seq_data=vp;
		sp->seq_flags = SEQ_MOVIE;
		sp->seq_refcnt++;

		/* increment reference count on this leaf */
		(*the_smp->ref_func)(QSP_ARG  vp);
	}
	return(sp);
}

%}
//%union
//{
//	Seq *yysp;
//	void *yyvp;
//	int yyi;
//}

%pure-parser	// make the parser rentrant (thread-safe)

// parse-param also affects yyerror!

%parse-param{ Query_Stack *qsp }
%lex-param{ Query_Stack *qsp }


%type < yysp > seqst sequence movie
%token <yyi> REVERSE NUMBER END
%token < yysp > SEQNAME
%token < yyvp > MY_MOVIE_NAME
%start seqst
%left '+'
%nonassoc '*'

%%

seqst		: sequence END
			{ final_mviseq = $$ ; }
		;

sequence	: sequence '+' sequence
			{ $$=joinseq( $1, $3 ); }
		| movie
		| NUMBER '*' sequence
			{ $$=reptseq( $1, $3 ); }
		| NUMBER '(' sequence ')'
			{ $$=reptseq( $1, $3 ); }
		| NUMBER '*' '(' sequence ')'
			{ $$=reptseq( $1, $4 ); }
		| SEQNAME
		| REVERSE '(' sequence ')'
			{ $$=revseq( $3 ); }
		;


movie		: NUMBER '*' MY_MOVIE_NAME
			{
				$$=makfrm($1,$3);
			}
		;

%%

ITEM_INTERFACE_DECLARATIONS( Seq, mviseq, 0 )

// This function is called from mvimenu module
void init_movie_sequences(SINGLE_QSP_ARG_DECL)
{
	if( mviseq_itp == NULL )
		init_mviseqs();
	add_playable(mviseq_itp,NULL);
}

void load_seq_module(Seq_Module *smp)
{
	the_smp = smp;
}

int _init_show_seq(QSP_ARG_DECL  Seq *sp)
{
	/* travel down this sequence until we find a leaf to pass
	 * to the application specific routine
	 */

	while( sp->seq_flags != SEQ_MOVIE )
		sp = sp->seq_first;

	return( (*the_smp->init_func)(QSP_ARG  sp->seq_data) );
}

void _wait_show_seq(SINGLE_QSP_ARG_DECL)
{
	(*the_smp->wait_func)(SINGLE_QSP_ARG);
}

void show_sequence(QSP_ARG_DECL  const char *s)
{
	Seq *sp;

	sp = get_mviseq(s);
	if( sp==NULL ) return;

	if( init_show_seq(sp) < 0 ) return;
	evalseq(sp);
	wait_show_seq();
}



#define IS_LEGAL_NAME_START( c )	( isalpha( c ) ||	\
					( c ) == '.'   ||	\
					( c ) == '/'   ||	\
					( c ) == '-'   ||	\
					( c ) == '_' )

#define IS_LEGAL_NAME_CHAR( c )		( isalnum( c ) ||	\
					( c ) == '.'   ||	\
					( c ) == '/'   ||	\
					( c ) == '-'   ||	\
					( c ) == '_' )

int yylex( YYSTYPE *yylval_p, Query_Stack *qsp )
{
	char *numptr, *wrdptr;
	int n;
	char str[128];

	if( ended ) return(0);
	while( isspace(*ipptr) ) ipptr++;
	if( (*ipptr) == 0 ) {
		ended=1;
		return(END);
	}
	if( isdigit( *ipptr ) ){
		char numbuf[LLEN];

		numptr=numbuf;
		while( isdigit(*ipptr) )
			*numptr++ = (*ipptr++);
		*numptr=0;
		sscanf(numbuf,"%d",&n);
		yylval_p->yyi=n;
		return(NUMBER);
	} else if( IS_LEGAL_NAME_START(*ipptr) ){
		char wrdbuf[LLEN];

		wrdptr=wrdbuf;
		while( IS_LEGAL_NAME_CHAR(*ipptr) )
			*wrdptr++ = (*ipptr++);
		*wrdptr=0;

		if( !strcmp(wrdbuf,"reverse") ) return(REVERSE);

		yylval_p->yysp = mviseq_of( wrdbuf );
		if( yylval_p->yysp != NULL ) return( SEQNAME );

		/* not a sequence, try a pattern name */

		yylval_p->yyvp = (*the_smp->get_func)( QSP_ARG  wrdbuf );
		if( yylval_p->yyvp != NULL ) return( MY_MOVIE_NAME );

		/* error */
		sprintf(str,"%s is not a sequence or a movie name", wrdbuf);
		yyerror(qsp,str);
		return(0);
	} else {
		yylval_p->yyi=0;
		return(*ipptr++);
	}
}

static Seq *seqparse(QSP_ARG_DECL  const char *strbuf)		/* compile sequence in strbuf */
{
	ipptr=strbuf;
	ended=0;
	if( yyparse(THIS_QSP)==0 ) return(final_mviseq);
	else {
		sprintf(ERROR_STRING,
			"Error parsing sequence definition \"%s\"", strbuf);
		WARN(ERROR_STRING);
		return(NULL);
	}
}

int yyerror(Query_Stack *qsp,  const char *s)
{
	sprintf(ERROR_STRING,"seqparse (yyerror): %s",s);
	WARN(ERROR_STRING);
	return(0);
}

static Seq *new_seq(QSP_ARG_DECL  const char *name)
{
	Seq *sp;

	sp=new_mviseq(name);	/* get a new item */
	if( sp == NULL ) return(sp);

	init_seq_struct(sp);
	return(sp);
}

Seq *defseq(QSP_ARG_DECL  const char *name,const char *seqstr)	/** define new seq by seqstr */
{
	Seq *sp, *tmp_sp;

	sp=new_seq(QSP_ARG  name);
	if( sp==NULL ) return(sp);

	tmp_sp=seqparse(QSP_ARG  seqstr);
	if( tmp_sp == NULL ){
		delseq(QSP_ARG  sp);
		return(NULL);
	}

	sp->seq_first  = tmp_sp->seq_first;
	sp->seq_next   = tmp_sp->seq_next;
	sp->seq_data   = tmp_sp->seq_data;
	sp->seq_count  = tmp_sp->seq_count;
	sp->seq_flags  = tmp_sp->seq_flags;
	sp->seq_refcnt = tmp_sp->seq_refcnt;

	givbuf(tmp_sp);

	return(sp);
}

void delseq(QSP_ARG_DECL  Seq *sp)
{
	sp->seq_refcnt--;
	if( sp->seq_first != NULL ) delseq(QSP_ARG  sp->seq_first);
	if( sp->seq_next != NULL ) delseq(QSP_ARG  sp->seq_next);
	if( sp->seq_refcnt <= 0 ){
		if( sp->seq_name != NULL ){
			del_mviseq(sp);
			// return to item free list
		} else {
			givbuf(sp);
		}
	}
}

void _evalseq(QSP_ARG_DECL  Seq *seqptr)		/* recursive procedure to compile a subsequence */
{
	int cnt;

	if( seqptr == NULL ) return;

	cnt=seqptr->seq_count;
	if( cnt < 0 ){
		reverse_eval(seqptr->seq_first);
		return;
	}
	if( seqptr->seq_flags==SEQ_MOVIE ){	 		/* a frame */
		while( cnt-- )
			(*the_smp->show_func)(QSP_ARG  seqptr->seq_data);
	} else {
		while( cnt-- ){
			evalseq(seqptr->seq_first);
			evalseq(seqptr->seq_next);
		}
	}
}

void _reverse_eval(QSP_ARG_DECL  Seq *seqptr)	/* recursive procedure to reverse a sequence */
{
	int cnt;

	if( seqptr == NULL ) return;

	cnt=seqptr->seq_count;
	if( cnt < 0 ){
		evalseq(seqptr->seq_first);
		return;
	}
	if( seqptr->seq_flags==SEQ_MOVIE ){	 		/* a frame */
		while( cnt-- )
			(*the_smp->rev_func)(QSP_ARG  seqptr->seq_data);
	} else {
		while( cnt-- ){
			reverse_eval(seqptr->seq_next);
			reverse_eval(seqptr->seq_first);
		}
	}
}

void setrefn(int n)			/** set refresh count per frame */
{
	refno=n;
	if( refno <1 ){
		NWARN("ridiculous refno");
		refno=1;
	}
}

static int contains(Seq *seqp,void *data)
{
	if( seqp->seq_flags==SEQ_MOVIE ){	 		/* a frame */
		if( seqp->seq_data == data )
			return(1);
		else
			return(0);
	}

	if( seqp->seq_first != NULL && contains(seqp->seq_first,data) )
		return(1);

	if( seqp->seq_next != NULL && contains(seqp->seq_next,data) )
		return(1);

	return(0);
}

#include "gmovie.h"

List *seqs_referring(QSP_ARG_DECL  void *data)
{
	List *lp;
	Node *np;

	lp=item_list(mviseq_itp);
	np=QLIST_HEAD(lp);

	lp=new_list();
	while( np != NULL ){
		if( contains((Seq *)np->n_data,data) )
			addTail(lp,mk_node(np->n_data));
		np=np->n_next;
	}
	return(lp);
}


