%{

#include "quip_config.h"
#include <errno.h>
#include <math.h>

#include "quip_prot.h"
#include "warn.h"
#include "shape_bits.h"

//#define YACC_HACK_PREFIX	quip
//#include "yacc_hack.h"

//static char err_str[LLEN];
static const char *original_string;
#define YY_ORIGINAL	original_string
//#define ERROR_STRING err_str

//#include <stdio.h>
//
//#include <stdlib.h>
//#include <string.h>
//#include <ctype.h>

#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif // HAVE_LIMITS_H

double rn_number();

#include "query_prot.h"
#include "nexpr.h"

//#include "nexpr_func.h"

//typedef void Function;  // so (Function *) is (void *)

#include "func_helper.h"

#ifdef MOVED
// Need this local prototype because of call structure...
static Item * eval_szbl_expr( QSP_ARG_DECL  Scalar_Expr_Node *);
#endif // MOVED

#ifdef MOVED
static Data_Obj *obj_for_string(const char *string);
#endif // MOVED

static Item * default_eval_szbl( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{ return NULL; }

//#define EVAL_EXPR( s )		eval_expr( QSP_ARG  s )
//#define EVAL_SZBL_EXPR( s )	eval_szbl_expr( QSP_ARG  s )
static Item * ( * eval_szbl_func ) ( QSP_ARG_DECL  Scalar_Expr_Node *enp ) = default_eval_szbl;
#define EVAL_SZBL_EXPR_FUNC( s )	(*eval_szbl_func)( QSP_ARG  s )
#define EVAL_ILBL_EXPR( s )	eval_interlaceable_expr( QSP_ARG  s )
#define EVAL_PSNBL_EXPR( s )	eval_positionable_expr( QSP_ARG  s )

#ifdef QUIP_DEBUG
static debug_flag_t expr_debug=0;
#endif /* QUIP_DEBUG */

/*
 * at one time minus signs were allowed embedded in names;
 * Do we really need this?
 *
 * Similar problems with .'s in names...   We currently have
 * a problem with things like nrows('a.1[0][0]') because it
 * wants to treat it like a string...
 */

/* We want to allow the colon in names because it is used in X
 * display names (e.g.  depth(:0));
 * But maybe we better insist that that be quoted!
 */

#define IS_LEGAL_NAME_PUNCT(c)	( (c)=='.' || (c)=='_' /* || (c)==':' */ )
#define IS_LEGAL_FIRST_CHAR(c)	( isalpha( c ) || IS_LEGAL_NAME_PUNCT(c) )
#define IS_LEGAL_NAME_CHAR(c)	( isalnum( c ) || IS_LEGAL_NAME_PUNCT(c) )

// moved to query.h
#define MAXEDEPTH	20

// BUG all these static globals are not thread-safe!?

static List *free_enp_lp=NO_LIST;

// BUG - probably 4 is not enough now that these are being used
// to store strings for the whole tree...
#define MAX_E_STRINGS	64
static String_Buf *expr_string[MAX_E_STRINGS];
static Typed_Scalar string_scalar[MAX_E_STRINGS];
static int estrings_inited=0;

static Scalar_Expr_Node *alloc_expr_node(void);

/* These have to be put into query stream... */
/*
static Scalar_Expr_Node * final;
*/

static Scalar_Expr_Node * final_expr_node_p;
// BUG not thread-sage
static const char *yystrptr[MAXEDEPTH];
static int edepth=(-1);
static int which_str=0;
static int in_pexpr=0;
#define FINAL_EXPR_NODE_P	final_expr_node_p
#define YYSTRPTR yystrptr
#define EDEPTH edepth
#define WHICH_EXPR_STR	which_str
#define IN_PEXPR	in_pexpr

#define ADVANCE_EXPR_STR				\
							\
	WHICH_EXPR_STR++;				\
	WHICH_EXPR_STR %= MAX_E_STRINGS;


static Typed_Scalar ts_dbl_zero={
	{ 0.0 }, PREC_DP, TS_STATIC };

static Typed_Scalar ts_dbl_one={
	{ 1.0 }, PREC_DP, TS_STATIC };

static Typed_Scalar ts_dbl_minus_one={
	{ -1.0 }, PREC_DP, TS_STATIC };

//static int n_function_classes=0;
//#define MAX_FUNCTION_CLASSES	10
//static Function_Class func_class[MAX_FUNCTION_CLASSES];

#define LLERROR(s)	llerror(s)

/* what yylval can be */

typedef union {
//	double			dval;		/* actual value */
	int			fundex;		/* function index */
	Function *		func_p;
//	char *			e_string;
	Scalar_Expr_Node *	enp;
	Typed_Scalar *		tsp;
} YYSTYPE;

#define YYSTYPE_IS_DECLARED		/* needed on 2.6 machine? */



#ifdef THREAD_SAFE_QUERY

//#error THREAD_SAFE_QUERY is defined
//#define YYPARSE_PARAM qsp	/* gets declared void * instead of Query_Stack * */

/* For yyerror */
#define YY_(msg)	msg

//#define YYLEX_PARAM SINGLE_QSP_ARG
//static int yylex(YYSTYPE *yylvp, Query_Stack *qsp);


#ifdef HAVE_PTHREADS
// We have one mutex that a thread has to hold to manipulate the list,
// and a shared flag to show whether or not they are locked...

static pthread_mutex_t	enode_mutex=PTHREAD_MUTEX_INITIALIZER;
static int enode_flags=0;

/* We don't bother with the mutex if the number of threads is less
 * than 1, but this could create a problem if we create a thread?
 * probably not...
 */

#define LOCK_ENODES						\
								\
	if( n_active_threads > 1 )				\
	{							\
		int status;					\
								\
		status = pthread_mutex_lock(&enode_mutex);	\
		if( status != 0 )				\
			report_mutex_error(QSP_ARG  status,"LOCK_ENODES");\
		enode_flags |= LIST_LOCKED;			\
	}

#define UNLOCK_ENODES						\
								\
	if( enode_flags & LIST_LOCKED )				\
	{							\
		int status;					\
								\
		enode_flags &= ~LIST_LOCKED;			\
		status = pthread_mutex_unlock(&enode_mutex);	\
		if( status != 0 )				\
			report_mutex_error(QSP_ARG  status,"UNLOCK_ENODES");\
	}
#else /* ! HAVE_PTHREADS */

#define LOCK_ENODES
#define UNLOCK_ENODES

#endif /* ! HAVE_PTHREADS */

#else /* ! THREAD_SAFE_QUERY */

#define LOCK_ENODES
#define UNLOCK_ENODES
//static int yylex(YYSTYPE *yylvp);		// undef THREAD_SAFE_QUERY

#endif /* ! THREAD_SAFE_QUERY */

// For the time being a single signature, regardless of THREAD_SAFE_QUERY
static int yylex(YYSTYPE *yylvp, Query_Stack *qsp);

int yyerror(Query_Stack *, char *);
/* int yyparse(void); */

#ifdef MOVED
// We try to avoid local prototypes, but these have to be here...
static Data_Obj * _def_obj(QSP_ARG_DECL  const char *);
static Data_Obj * _def_sub(QSP_ARG_DECL  Data_Obj * , index_t);
#endif // MOVED


#ifdef FOOBAR
// we avoid local prototypes, but this has to be here
// because of circular calls between eval_expr and eval_dobj_expr...
//static double eval_expr(QSP_ARG_DECL  Scalar_Expr_Node *);
//static Typed_Scalar * eval_expr(QSP_ARG_DECL  Scalar_Expr_Node *);
#endif // FOOBAR


/* globals */

#ifdef MOVED
// These pointers are here so that this module can be build (and loaded)
// independently of the data_obj module.
Data_Obj * (*obj_get_func)(QSP_ARG_DECL  const char *)=_def_obj;
//Data_Obj * (*exist_func)(QSP_ARG_DECL  const char *)=_def_obj;
Data_Obj * (*sub_func)(QSP_ARG_DECL  Data_Obj *,index_t)=_def_sub;
Data_Obj * (*csub_func)(QSP_ARG_DECL  Data_Obj *,index_t)=_def_sub;
#endif // MOVED

#define NODE0(code)	node0(code)
#define NODE1(code,enp)	node1(code,enp)
#define NODE2(code,enp1,enp2)	node2(code,enp1,enp2)
#define NODE3(code,enp1,enp2,enp3)	node3(code,enp1,enp2,enp3)

static Scalar_Expr_Node *node0( Scalar_Expr_Node_Code code )
{
	Scalar_Expr_Node *enp;

	enp = alloc_expr_node();
	enp->sen_code = code;
	return(enp);
}

static Scalar_Expr_Node *node1( Scalar_Expr_Node_Code code, Scalar_Expr_Node *child )
{
	Scalar_Expr_Node *enp;

	enp = alloc_expr_node();
	enp->sen_code = code;
	enp->sen_child[0]=child;
	return(enp);
}

static Scalar_Expr_Node *node2( Scalar_Expr_Node_Code code, Scalar_Expr_Node *child1, Scalar_Expr_Node *child2 )
{
	Scalar_Expr_Node *enp;

	enp = alloc_expr_node();
	enp->sen_code = code;
	enp->sen_child[0]=child1;
	enp->sen_child[1]=child2;
	return(enp);
}

static Scalar_Expr_Node *node3( Scalar_Expr_Node_Code code, Scalar_Expr_Node *child1, Scalar_Expr_Node *child2, Scalar_Expr_Node *child3 )
{
	Scalar_Expr_Node *enp;

	enp = alloc_expr_node();
	enp->sen_code = code;
	enp->sen_child[0]=child1;
	enp->sen_child[1]=child2;
	enp->sen_child[2]=child3;
	return(enp);
}

// New version to avoid fixed length strings

static char *get_expr_stringbuf( int index, long min_len )
{
	String_Buf *sbp;

	if( expr_string[index] == NULL ){
		sbp = new_stringbuf();
		expr_string[index]=sbp;
	} else {
		sbp = expr_string[index];
	}
	if( expr_string[index]->sb_size < min_len )
		enlarge_buffer(sbp,min_len);
	return(sbp->sb_buf);
}

%}

// %pure_parser	// make the parser rentrant (thread-safe)
%pure-parser	// make the parser rentrant (thread-safe)
// not on brewster...
%name-prefix="quip_"
// this does not work on pavlov:
//%name-prefix "quip_"

/* The YYPARSE_PARAM macro has been deprecated in favor of %parse-param
 * BUT parse-param is a bison statment that comes outside of the the C code
 * block, BEFORE being run through the C preprocessor.  So it is not easy
 * to have it conditionally defined.  For the time being, we specify
 * the THREAD_SAFE_QUERY argument (qsp), accepting that we will be incurring
 * a small penalty when we are building a single-thread version for speed.
 */

// parse-param also affects yyerror!

%parse-param{ Query_Stack *qsp }
%lex-param{ Query_Stack *qsp }

/* expressison operators */

/* 
 * The lower down in this list an operator appears,
 * the higher the precedence.  Operators on the same line have
 * equal precedence.
 *
 * These precedences are copied from C Ref. Manual (K&R, p. 215).
 */

%left <fundex> '?' ':'
%left <fundex> LOGOR
%left <fundex> LOGAND
%left <fundex> LOGXOR
%left <fundex> '|'
%left <fundex> '^'
%left <fundex> '&'
%left <fundex> EQUIV NE
%left <fundex> '<' '>' GE LE
%left <fundex> SHL SHR
%left <fundex> '+' '-'
%left <fundex> '*' '/' '%'
%nonassoc <fundex> '!'
/* just a tag for assigning high precedence to unary minus, plus */
%left <fundex> UNARY

%token <tsp> NUMBER
%token <func_p> MATH0_FUNC
%token <func_p> MATH1_FUNC
%token <func_p> MATH2_FUNC
%token <func_p> INT1_FUNC
%token <func_p> DATA_FUNC
%token <func_p> SIZE_FUNC
%token <func_p> TS_FUNC
%token <func_p> IL_FUNC
%token <func_p> POSN_FUNC
%token <func_p> STR_FUNC
%token <func_p> STR2_FUNC
%token <func_p> STR3_FUNC
%token <func_p> STRV_FUNC
%token <func_p> STRV2_FUNC
%token <func_p> CHAR_FUNC
// What is the difference between an E_STRING and an E_QSTRING??
// QSTRING is quoted, and can be used to initialize a data object.
//
// Problem:  ncols("Front Camera") returns 13,
// because iOS cameras are not yet sizable objects, and instead of
// saying "no sizable object found" it creates a string-valued
// data object and gives its dimensions !?  (12 chars plus terminating null)
%token <tsp> E_STRING
%token <tsp> E_QSTRING

%start topexp
%type <enp> expression
%type <enp> data_object
//%type <enp> sizable_object
%type <enp> timestampable_object
%type <enp> e_string
%type <enp> strv_func

/* Why did we allow topexp to be an e_string?  That broke the ability
 * to have expressions like isdigit("abc"[0])...
 *
 * Now it is disabled, but what is now broken?
 * Probably string functions like toupper() - ?
 */

%%

topexp		: expression { 
			// qsp is passed to yyparse through YYPARSE_PARAM, but it is void *
			final_expr_node_p = $1 ;
			}
		/*
		| e_string {
			final_expr_node_p = $1;
			}
			*/
		| strv_func {
			final_expr_node_p = $1;
			}
		;

			
strv_func	: STRV_FUNC '(' /* e_string */ data_object ')' {
			// string-valued functions (e.g. precision(obj))
			//
			// This doesn't have to be a data_object,
			// it can be the name of a sizable object...

			// STRV_FUNC's were introduced to handle string
			// conversions like toupper...  But it seems as
			// if that was never fully implemented.  Now we
			// are using it to return the precision of a sizable object...
			// We will worry what to do about viewers later...

			// Does it break the grammar to put a string here?
			$$=NODE1(N_STRVFUNC,$3);
			$$->sen_func_p = $1;
			}
		| STRV2_FUNC '(' data_object ',' data_object ')' {
			$$=NODE2(N_STRV2FUNC,$3,$5);
			$$->sen_func_p = $1;
			}

		;

e_string	: E_STRING {
			$$ = NODE0(N_LITSTR);
			$$->sen_tsp = $1;
			}
		| E_QSTRING {
			$$ = NODE0(N_LITSTR);
			$$->sen_tsp = $1;
			}
		| strv_func
		;

data_object	: /* E_STRING {
			Scalar_Expr_Node *enp;
			enp=NODE0(N_LITSTR);
			enp->sen_tsp = $1;
			$$=NODE1(N_OBJNAME,enp);
			}
		| E_QSTRING {
			/* the string is the data, not the name of an object... */
			/* We added this node type to support things like
			 * isdigit("abc1"[2]), but treating quoted strings
			 * as row vectors broke a lot of things where we quote
			 * the name of an image file...
			 *
			 * A compromise is to strip the quotes as normally,
			 * but if the data_obj does not exist, THEN treat it
			 * as a string... messy!
			 */
			 /*
			Scalar_Expr_Node *enp;
			enp=NODE0(N_LITSTR);
			enp->sen_tsp = $1;
			$$=NODE1(N_QUOT_STR,enp);
			}
		| */ e_string {
			Scalar_Expr_Node *enp;
			enp=NODE1(N_STRING,$1);
			//enp->sen_tsp = $1;
			$$=NODE1(N_QUOT_STR,enp);
			}

		| data_object '[' expression ']' {
			$$=NODE2(N_SUBSCRIPT,$1,$3); }
		| data_object '{' expression '}' {
			$$=NODE2(N_CSUBSCRIPT,$1,$3); }
		;

/* Because the size functions can take things other than data objects
 * as arguments (e.g., a viewer name), it seemed more readable to
 * have a separate nonterminal token for these...  but because
 * we can want to have a subscripted data object as a term,
 * we need to have data_object, and because they are matched by
 * a string it still works...
 */

/*
sizable_object	: e_string {
			$$ = NODE0(N_SIZABLE);
			$$->sen_string=savestr($1);
			}
*/

timestampable_object		: e_string {
			$$ = NODE1(N_TSABLE,$1);
			}
		;


expression	: NUMBER {
			$$ = NODE0(N_LITNUM);
			//$$->sen_dblval = $1;
			$$->sen_tsp = $1;
			}
		| MATH0_FUNC '(' ')' {
			$$=NODE0(N_MATH0FUNC);
			$$->sen_func_p = $1;
			}
		| MATH1_FUNC '(' expression ')' {
			$$=NODE1(N_MATH1FUNC,$3);
			$$->sen_func_p = $1;
			}
		| MATH2_FUNC '(' expression ',' expression ')' {
			$$=NODE2(N_MATH2FUNC,$3,$5);
			$$->sen_func_p = $1;
			}
		| INT1_FUNC '(' expression ')' {
			$$=NODE1(N_INT1FUNC,$3);
			$$->sen_func_p = $1;
			}
		| CHAR_FUNC '(' expression ')' {
			$$=NODE1(N_CHARFUNC,$3);
			$$->sen_func_p = $1;
			}
		| DATA_FUNC '(' data_object ')' {
			$$=NODE1(N_DATAFUNC,$3);
			$$->sen_func_p=$1;
			}
		| IL_FUNC '(' data_object ')' {
			// This doesn't have to be a data_object,
			// it can be the name of a sizable object...
			$$=NODE1(N_ILACEFUNC,$3);
			$$->sen_func_p=$1;
			}
		| POSN_FUNC '(' data_object ')' {
			// This doesn't have to be a data_object,
			// it can be the name of a sizable object...
			$$=NODE1(N_POSNFUNC,$3);
			$$->sen_func_p=$1;
			}
		| SIZE_FUNC '(' data_object ')' {
			// This doesn't have to be a data_object,
			// it can be the name of a sizable object...
			$$=NODE1(N_SIZFUNC,$3);
			$$->sen_func_p=$1;
			}
		| TS_FUNC '(' timestampable_object ',' expression ')' {
			$$=NODE2(N_TSFUNC,$3,$5);
			$$->sen_func_p=$1;
			}
		| STR_FUNC '(' e_string ')' {
			$$=NODE1(N_STRFUNC,$3);
			$$->sen_func_p = $1;
			}
		| STR2_FUNC '(' e_string ',' e_string ')' {
			$$=NODE2(N_STR2FUNC,$3,$5);
			$$->sen_func_p = $1;
			}
		| STR3_FUNC '(' e_string ',' e_string ',' expression ')' {
			$$=NODE3(N_STR3FUNC,$3,$5,$7);
			$$->sen_func_p = $1;
			}
		| '(' expression ')' {
			$$ = $2 ; }
		| expression '+' expression {
			$$=NODE2(N_PLUS,$1,$3);
			}
		| expression '-' expression { $$ = NODE2(N_MINUS,$1,$3); }
		| expression '/' expression { $$ = NODE2(N_DIVIDE,$1,$3); }
		| expression '*' expression { $$ = NODE2(N_TIMES,$1,$3); }
		| expression '%' expression { $$ = NODE2(N_MODULO,$1,$3); }
		| expression '&' expression { $$ = NODE2(N_BITAND,$1,$3); }
		| expression '|' expression { $$ = NODE2(N_BITOR,$1,$3); }
		| expression '^' expression { $$ = NODE2(N_BITXOR,$1,$3); }
		| expression SHL expression { $$ = NODE2(N_SHL,$1,$3); }
		| expression SHR expression { $$ = NODE2(N_SHR,$1,$3); }
		| '[' expression ']' { $$ = $2; }
		| '~' expression %prec UNARY { $$ = NODE1(N_BITCOMP,$2); }
		| '+' expression %prec UNARY { $$ = $2; }
		| '-' expression %prec UNARY { $$ = NODE1(N_UMINUS,$2); }
		| expression '?' expression ':' expression
			{ $$ = NODE3(N_CONDITIONAL,$1,$3,$5); }
		| expression EQUIV expression { $$=NODE2(N_EQUIV,$1,$3); }
		| '!' expression %prec UNARY { $$ = NODE1(N_NOT,$2); }
		| expression LOGOR expression { $$=NODE2(N_LOGOR,$1,$3); }
		| expression LOGAND expression { $$=NODE2(N_LOGAND,$1,$3); }
		| expression LOGXOR expression { $$=NODE2(N_LOGXOR,$1,$3); }
		| expression '<' expression { $$=NODE2(N_LT,$1,$3); }
		| expression '>' expression { $$=NODE2(N_GT,$1,$3); }
		| expression GE expression { $$=NODE2(N_GE,$1,$3); }
		| expression LE expression { $$=NODE2(N_LE,$1,$3); }
		| expression NE expression { $$=NODE2(N_NE,$1,$3); }
		/* subscripting a string to get a char constant -
		 * why commented out??
		 */

		/*
		| e_string '[' expression ']'
			{ $$=NODE1(N_SLCT_CHAR,$3);
			$$->sen_string = savestr($1);
			}
			*/
		| data_object
			{ $$=NODE1(N_SCALAR_OBJ,$1); }

		// We used to allow a named scalar object to be given -
		// equivalent to value(objname), but without having to explicitly
		// say "value" - but after the addition of string expressions,
		// this created an ambiguity.
		/*
		| data_object {
			// must be a scalar object
			$$=NODE1(N_SCALAR_OBJ,$1);
			}
		*/

		;


%%

#ifdef MOVED
static Data_Obj *obj_for_string(const char *string)
{
	Dimension_Set *dsp;
	Data_Obj *dp;

	INIT_DIMSET_PTR(dsp)

	/* this is just a string that we treat as a row vector
	 * of character data...
	 * We haven't actually created the data yet.
	 */
	SET_DIMENSION(dsp,0,1);
	SET_DIMENSION(dsp,1,(dimension_t)strlen(string)+1);
	SET_DIMENSION(dsp,2,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,4,1);
	dp=make_dobj(DEFAULT_QSP_ARG  localname(),dsp,prec_for_code(PREC_STR));
	if( dp != NULL ) strcpy((char *)OBJ_DATA_PTR(dp),string);
	return(dp);
}
#endif // MOVED

static Scalar_Expr_Node *alloc_expr_node(void)
{
	Scalar_Expr_Node *enp;
	int i;

	if( free_enp_lp != NO_LIST && QLIST_HEAD(free_enp_lp) != NO_NODE ){
		Node *np;
		np=remHead(free_enp_lp);
		enp = (Scalar_Expr_Node *)NODE_DATA(np);
		rls_node(np);
	} else {
		enp=malloc(sizeof(*enp));
	}

	// All of these initializations should be unnecessary...
	enp->sen_code=(-1);
	for(i=0;i<MAX_SEN_CHILDREN;i++)
		enp->sen_child[i]=NULL;
	enp->sen_index=(-1);
	enp->sen_func_p=NULL;
	//enp->sen_dblval=0.0;
	enp->sen_tsp=NULL;
	/*
	enp->sen_string=NULL;
	enp->sen_string2=NULL;
	*/
	return(enp);
}

const char *eval_scalexp_string(QSP_ARG_DECL  Scalar_Expr_Node *enp)
{
	Typed_Scalar *tsp;
	const char *s, *s2;

	switch(enp->sen_code){
		case N_OBJNAME:
		case N_QUOT_STR:
			return EVAL_SCALEXP_STRING(enp->sen_child[0]);
			break;

		case N_STRVFUNC:
#ifdef BUILD_FOR_OBJC
			// BUG BUG BUG
			if( check_ios_strv_func(&s,enp->sen_func_p,
					enp->sen_child[0]) ){
				// BUG?  does the function get called in check_ios_sizable_func???
				return s;
			}
#endif /* BUILD_FOR_OBJC */
			// why sizable?  this is supposed to be a string arg...
			// This makes sense only for the "precision" function -
			// but what about touuper etc?
			//szp = EVAL_SZBL_EXPR_FUNC(enp->sen_child[0]);
			s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
			s = (*enp->sen_func_p->fn_u.strv_func)( QSP_ARG  s );
			return s;
			break;

		case N_STRV2FUNC:
#ifdef BUILD_FOR_OBJC
			// BUG BUG BUG ? (why?)
			if( check_ios_strv2_func(&s,enp->sen_func_p,
					enp->sen_child[0],enp->sen_child[1]) ){
				// BUG?  does the function get called in check_ios_sizable_func???
				return s;
			}
#endif /* BUILD_FOR_OBJC */
			// why sizable???
			/*
			szp = EVAL_SZBL_EXPR_FUNC(enp->sen_child[0]);
			s = (*enp->sen_func_p->fn_u.strv_func)( QSP_ARG  szp );
			return s;
			*/
			s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
			s2 = EVAL_SCALEXP_STRING(enp->sen_child[1]);
			s = (*enp->sen_func_p->fn_u.strv2_func)( QSP_ARG  s, s2 );
			return s;
			break;

		case N_LITSTR:
			tsp = enp->sen_tsp;
//#ifdef CAUTIOUS
//			if( tsp->ts_prec_code != PREC_STR ){
//				NWARN("CAUTIOUS:  eval_scalexp_string:  typed scalar does not have string precision!?");
//				return "foobar";
//			}
//#endif // CAUTIOUS
			assert( tsp->ts_prec_code == PREC_STR );

			return (char *) tsp->ts_value.u_vp;
			break;
//#ifdef CAUTIOUS
		case N_STRING:
			return EVAL_SCALEXP_STRING(enp->sen_child[0]);
			break;
		default:
//			sprintf(DEFAULT_ERROR_STRING,
//		"CAUTIOUS:  eval_scalexp_string:  unhandled node code %d!?",enp->sen_code);
//			NWARN(DEFAULT_ERROR_STRING);
//			dump_etree(DEFAULT_QSP_ARG  enp);
			assert( 0 );
			break;
//#endif // CAUTIOUS
	}
	return "foobar";
}


static Item* eval_tsbl_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Item *ip=NULL;
	const char *s;

	switch(enp->sen_code){
		case N_TSABLE:
			s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
			ip = find_tsable( DEFAULT_QSP_ARG  s );
			if( ip == NULL ){
				sprintf(ERROR_STRING,
					"No time-stampable object \"%s\"!?",s);
				NWARN(ERROR_STRING);
				return(NULL);
			}
			break;
//#ifdef CAUTIOUS
		default:
//			NWARN("unexpected case in eval_tsbl_expr");
			assert( AERROR("unexpected case in eval_tsbl_expr") );
			break;
//#endif /* CAUTIOUS */
	}
	return(ip);
}

static void dump_enode(QSP_ARG_DECL  Scalar_Expr_Node *enp)
{
	const char *s;

/* Need to do these unary ops:
N_BITCOMP
N_UMINUS
N_EQUIV
N_CONDITIONAL
*/
	switch(enp->sen_code){
		case N_QUOT_STR:
			s = EVAL_SCALEXP_STRING(enp);
			sprintf(ERROR_STRING,"0x%lx\tstring\t%s",
				(long/*int_for_addr*/)enp, s);
			ADVISE(ERROR_STRING);
			break;

#ifdef FOOBAR
		case N_SIZABLE:
			sprintf(ERROR_STRING,"0x%lx\tsizable\t%s",
				(long/*int_for_addr*/)enp, enp->sen_string);
			ADVISE(ERROR_STRING);
			break;
#endif /* FOOBAR */

		case N_TSABLE:
			s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
			sprintf(ERROR_STRING,"0x%lx\ttsable\t%s",
				(long/*int_for_addr*/)enp, s);
			ADVISE(ERROR_STRING);
			break;

		case N_STRVFUNC:
			sprintf(ERROR_STRING,"0x%lx\tstrvfunc\t%s",
				(long/*int_for_addr*/)enp, FUNC_NAME( enp->sen_func_p ) );
			break;

		case N_STRV2FUNC:
			sprintf(ERROR_STRING,"0x%lx\tstrv2func\t%s",
				(long/*int_for_addr*/)enp, FUNC_NAME( enp->sen_func_p ) );
			break;

		case N_SIZFUNC:
			sprintf(ERROR_STRING,"0x%lx\tsizefunc\t%s",
				(long/*int_for_addr*/)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_TSFUNC:
			sprintf(ERROR_STRING,"0x%lx\tts_func\t%s",
				(long/*int_for_addr*/)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_ILACEFUNC:
			sprintf(ERROR_STRING,"0x%lx\tinterlace_func\t%s",
				(long/*int_for_addr*/)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_POSNFUNC:
			sprintf(ERROR_STRING,"0x%lx\tposn_func\t%s",
				(long/*int_for_addr*/)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_MATH0FUNC:
			sprintf(ERROR_STRING,"0x%lx\tmath0_func\t%s",
				(long/*int_for_addr*/)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_MATH2FUNC:
			sprintf(ERROR_STRING,"0x%lx\tmath2_func\t%s",
				(long/*int_for_addr*/)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_MISCFUNC:
			sprintf(ERROR_STRING,"0x%lx\tmisc_func\t%s",
				(long/*int_for_addr*/)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_STR2FUNC:
			sprintf(ERROR_STRING,"0x%lx\tstr2_func\t%s",
				(long/*int_for_addr*/)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_STR3FUNC:
			sprintf(ERROR_STRING,"0x%lx\tstr3_func\t%s",
				(long/*int_for_addr*/)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_DATAFUNC:
			sprintf(ERROR_STRING,"0x%lx\tdatafunc\t%s",
				(long/*int_for_addr*/)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_OBJNAME:
			s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
			sprintf(ERROR_STRING,"0x%lx\tobjname\t%s",
				(long/*int_for_addr*/)enp, s);
			ADVISE(ERROR_STRING);
			break;

		case N_SCALAR_OBJ:
			sprintf(ERROR_STRING,"0x%lx\tscalar_obj\t0x%lx",
				(long/*int_for_addr*/)enp, (long/*int_for_addr*/)enp->sen_child[0]);
			ADVISE(ERROR_STRING);
			break;

		case N_SUBSCRIPT:
			sprintf(ERROR_STRING,"0x%lx\tsubscript\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp, (long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
		case N_CSUBSCRIPT:
			sprintf(ERROR_STRING,"0x%lx\tcsubscript\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp, (long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_MATH1FUNC:
			sprintf(ERROR_STRING,"0x%lx\tmath1func\t%s",
				(long/*int_for_addr*/)enp, FUNC_NAME(enp->sen_func_p) );
			ADVISE(ERROR_STRING);
			break;

		case N_PLUS:
			sprintf(ERROR_STRING,"0x%lx\tplus\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp,(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_MINUS:
			sprintf(ERROR_STRING,"0x%lx\tminus\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp,(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_TIMES:
			sprintf(ERROR_STRING,"0x%lx\ttimes\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp,(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_DIVIDE:
			sprintf(ERROR_STRING,"0x%lx\tdivide\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp,(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_MODULO:
			sprintf(ERROR_STRING,"0x%lx\tmodulo\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp,(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_BITAND:
			sprintf(ERROR_STRING,"0x%lx\tbitand\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp,(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_BITOR:
			sprintf(ERROR_STRING,"0x%lx\tbitor\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp,(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_BITXOR:
			sprintf(ERROR_STRING,"0x%lx\tbitxor\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp,(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_SHL:
			sprintf(ERROR_STRING,"0x%lx\tshl\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp,(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_SHR:
			sprintf(ERROR_STRING,"0x%lx\tshr\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp,(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_LOGOR:
			sprintf(ERROR_STRING,"0x%lx\tlog_or\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp,(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_LOGAND:
			sprintf(ERROR_STRING,"0x%lx\tlog_and\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp,(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_LOGXOR:
			sprintf(ERROR_STRING,"0x%lx\tlog_xor\t0x%lx\t0x%lx",
				(long/*int_for_addr*/)enp,(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_LITNUM:
			string_for_typed_scalar(MSG_STR,enp->sen_tsp);
			sprintf(ERROR_STRING,"0x%lx\tlit_num\t%s",
				(long/*int_for_addr*/)enp,MSG_STR);
			ADVISE(ERROR_STRING);
			break;
		case N_LE:
			sprintf(ERROR_STRING,"0x%lx\t<= (LE)\t0x%lx, 0x%lx",(long/*int_for_addr*/)enp,
				(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
		case N_GE:
			sprintf(ERROR_STRING,"0x%lx\t>= (GE)\t0x%lx, 0x%lx",(long/*int_for_addr*/)enp,
				(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
		case N_NE:
			sprintf(ERROR_STRING,"0x%lx\t!= (NE)\t0x%lx, 0x%lx",(long/*int_for_addr*/)enp,
				(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
		case N_LT:
			sprintf(ERROR_STRING,"0x%lx\t< (LT)\t0x%lx, 0x%lx",(long/*int_for_addr*/)enp,
				(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
		case N_GT:
			sprintf(ERROR_STRING,"0x%lx\t> (GT)\t0x%lx, 0x%lx",(long/*int_for_addr*/)enp,
				(long/*int_for_addr*/)enp->sen_child[0],(long/*int_for_addr*/)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
		case N_NOT:
			sprintf(ERROR_STRING,"0x%lx\t! (NOT)\t0x%lx",
				(long/*int_for_addr*/)enp,
				(long/*int_for_addr*/)enp->sen_child[0]);
			ADVISE(ERROR_STRING);
			break;
		case N_STRFUNC:
			s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
			sprintf(ERROR_STRING,"0x%lx\tSTRFUNC %s\t\"%s\"",
				(long/*int_for_addr*/)enp,
				FUNC_NAME(enp->sen_func_p),
				s);
			ADVISE(ERROR_STRING);
			break;

// comment out the default case for the compiler to show unhandled cases...
		default:
			sprintf(ERROR_STRING,
		"%s - %s:  unhandled node code %d",
				WHENCE2(dump_enode),enp->sen_code);
			ADVISE(ERROR_STRING);
			break;

	}
}

// We provide this default to remove the dependency on the dobj library
static Data_Obj *default_eval_dobj( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{ return NULL; }

static Data_Obj * (*eval_dobj_func)(QSP_ARG_DECL  Scalar_Expr_Node *enp) = default_eval_dobj;

void set_eval_szbl_func(QSP_ARG_DECL  Item * (*func)(QSP_ARG_DECL  Scalar_Expr_Node *) )
{
	eval_szbl_func = func;
}

void set_eval_dobj_func(QSP_ARG_DECL  Data_Obj * (*func)(QSP_ARG_DECL  Scalar_Expr_Node *) )
{
	eval_dobj_func = func;
}

#ifdef MOVED
/* Evaluate a parsed expression */

static Data_Obj *eval_dobj_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Data_Obj *dp=NULL,*dp2;
	Typed_Scalar *tsp;
	index_t index;
	const char *s;

	switch(enp->sen_code){
		case N_QUOT_STR:
			s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
			/* first try object lookup... */
			/* we don't want a warning if does not exist... */
			dp = (*exist_func)( QSP_ARG  s );
			/* We have a problem here with indexed objects,
			 * since the indexed names aren't in the database...
			 */
			if( dp == NULL ){
				/* treat the string like a rowvec of chars */
				dp = obj_for_string(s);
				return(dp);
			}
			break;

		case N_SCALAR_OBJ:
			dp = eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
			if( IS_SCALAR(dp) ) return(dp);
			return(NULL);
			break;
		case N_OBJNAME:
			s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
			dp = (*obj_get_func)( QSP_ARG  s );
			break;
		case N_SUBSCRIPT:
			dp2=eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
			tsp = EVAL_EXPR(enp->sen_child[1]);
			index=index_for_scalar( tsp );
			RELEASE_SCALAR(tsp)
			dp=(*sub_func)( QSP_ARG  dp2, index );
			break;
		case N_CSUBSCRIPT:
			dp2=eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
			tsp=EVAL_EXPR(enp->sen_child[1]);
			index=index_for_scalar(tsp);
			RELEASE_SCALAR(tsp)
			dp=(*csub_func)( QSP_ARG  dp2, index );
			break;

//#ifdef CAUTIOUS
		default:
//			sprintf(ERROR_STRING,
//		"unexpected case (%d) in eval_dobj_expr",enp->sen_code);
//			NWARN(ERROR_STRING);
			assert(0);
			break;
//#endif /* CAUTIOUS */
	}
	return(dp);
} // end eval_dobj_expr

static Item * eval_szbl_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Item *szp=NULL,*szp2;
	index_t index;
	const char *s;

	switch(enp->sen_code){
		case N_QUOT_STR:
			s = EVAL_SCALEXP_STRING(enp);
			szp = check_sizable( DEFAULT_QSP_ARG  s );
			if( szp == NULL ){
				Data_Obj *dp;
				dp = obj_for_string(s);
				szp = (Item *)dp;
			}
			break;

		//case N_SIZABLE:
		case N_OBJNAME:
			// Not necessarily a data object!?
			s = EVAL_SCALEXP_STRING(enp);
			szp = find_sizable( DEFAULT_QSP_ARG  s );
			if( szp == NULL ){
				sprintf(ERROR_STRING,
					"No sizable object \"%s\"!?",s);
				NWARN(ERROR_STRING);
				return(NULL);
			}
			break;
		//case N_SUBSIZ:
		case N_SUBSCRIPT:
			szp2=EVAL_SZBL_EXPR(enp->sen_child[0]);
			if( szp2 == NULL )
				return(NULL);
			index = index_for_scalar( EVAL_EXPR(enp->sen_child[1]) );
			szp = sub_sizable(DEFAULT_QSP_ARG  szp2,index);
			break;
		//case N_CSUBSIZ:
		case N_CSUBSCRIPT:
			szp2=EVAL_SZBL_EXPR(enp->sen_child[0]);
			if( szp2 == NULL )
				return(NULL);
			index = index_for_scalar( EVAL_EXPR(enp->sen_child[1]) );
			szp = csub_sizable(DEFAULT_QSP_ARG  szp2,index);
			break;
//#ifdef CAUTIOUS
		default:
//			sprintf(ERROR_STRING,
//		"unexpected case in eval_szbl_expr %d",enp->sen_code);
//			NWARN(ERROR_STRING);
			assert(0);
			break;
//#endif /* CAUTIOUS */
	}
	return(szp);
}
#endif // MOVED

static Item * eval_positionable_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Item *szp=NULL;
	const char *s;

	switch(enp->sen_code){
		case N_QUOT_STR:
		case N_OBJNAME:
			// Not necessarily a data object!?
			s = EVAL_SCALEXP_STRING(enp);
			szp = find_positionable( DEFAULT_QSP_ARG  s );
			break;
//#ifdef CAUTIOUS
		default:
//			sprintf(ERROR_STRING,
//		"unexpected case in eval_szbl_expr %d",enp->sen_code);
//			NWARN(ERROR_STRING);
			assert(0);
			break;
//#endif /* CAUTIOUS */
	}
	if( szp == NULL ){
		sprintf(ERROR_STRING,
			"No positionable object \"%s\"!?",s);
		NWARN(ERROR_STRING);
		return(NULL);
	}

	return(szp);
}

static Item * eval_interlaceable_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Item *szp=NULL;
	const char *s;

	switch(enp->sen_code){
		case N_QUOT_STR:
		case N_OBJNAME:
			// Not necessarily a data object!?
			s = EVAL_SCALEXP_STRING(enp);
			szp = find_interlaceable( DEFAULT_QSP_ARG  s );
			if( szp == NULL ){
				sprintf(ERROR_STRING,
					"No interlaceable object \"%s\"!?",s);
				NWARN(ERROR_STRING);
				return(NULL);
			}
			break;
		// are data objects interlaceable???
#ifdef FOOBAR
		//case N_SUBSIZ:
		case N_SUBSCRIPT:
			szp2=EVAL_SZBL_EXPR(enp->sen_child[0]);
			if( szp2 == NULL )
				return(NULL);
			index = index_for_scalar( EVAL_EXPR(enp->sen_child[1]) );
			szp = sub_sizable(DEFAULT_QSP_ARG  szp2,index);
			break;
		//case N_CSUBSIZ:
		case N_CSUBSCRIPT:
			szp2=EVAL_SZBL_EXPR(enp->sen_child[0]);
			if( szp2 == NULL )
				return(NULL);
			index = index_for_scalar( EVAL_EXPR(enp->sen_child[1]) );
			szp = csub_sizable(DEFAULT_QSP_ARG  szp2,index);
			break;
#endif // FOOBAR

//#ifdef CAUTIOUS
		default:
//			sprintf(ERROR_STRING,
//		"unexpected case in eval_szbl_expr %d",enp->sen_code);
//			NWARN(ERROR_STRING);
			assert(0);
			break;
//#endif /* CAUTIOUS */
	}
	return(szp);
}

static void divzer_error(SINGLE_QSP_ARG_DECL)
{
	sprintf(DEFAULT_ERROR_STRING,"Error parsing \"%s\"",YY_ORIGINAL);
	ADVISE(DEFAULT_ERROR_STRING);
	NWARN("eval_expr:  divide by 0!?");
}

// These could be defined to something different on different hardware?

#define INT_TYPE	int64_t
#define UINT_TYPE	uint64_t
#define SCALAR_FOR_INT_TYPE	scalar_for_llong

#define SET_RESULT_ONE	tsp = &ts_dbl_one;
#define SET_RESULT_ZERO	tsp = &ts_dbl_zero;

#define SET_RESULT( test )			\
		if( test )			\
			SET_RESULT_ONE		\
		else				\
			SET_RESULT_ZERO

#define GET_TWO_DOUBLES					\
		tsp2=EVAL_EXPR(enp->sen_child[0]);	\
		tsp3=EVAL_EXPR(enp->sen_child[1]);	\
		dval2=double_for_scalar(tsp2);		\
		dval3=double_for_scalar(tsp3);		\
		RELEASE_BOTH

#define GET_ONE_DOUBLE					\
		tsp2=EVAL_EXPR(enp->sen_child[0]);	\
		dval2=double_for_scalar(tsp2);		\
		RELEASE_FIRST

#define GET_ONE_LONG					\
		tsp2=EVAL_EXPR(enp->sen_child[0]);	\
		ival2=long_for_scalar(tsp2);		\
		RELEASE_FIRST

#define GET_TWO_ULONGS					\
		tsp2=EVAL_EXPR(enp->sen_child[0]);	\
		tsp3=EVAL_EXPR(enp->sen_child[1]);	\
		uval2=llong_for_scalar(tsp2);		\
		uval3=llong_for_scalar(tsp3);		\
		RELEASE_BOTH

#define GET_TWO_LONGS					\
		tsp2=EVAL_EXPR(enp->sen_child[0]);	\
		tsp3=EVAL_EXPR(enp->sen_child[1]);	\
		ival2=llong_for_scalar(tsp2);		\
		ival3=llong_for_scalar(tsp3);		\
		RELEASE_BOTH

#define RELEASE_BOTH					\
		RELEASE_SCALAR(tsp2);			\
		RELEASE_SCALAR(tsp3);

#define RELEASE_FIRST RELEASE_SCALAR(tsp2);
#define RELEASE_SECOND RELEASE_SCALAR(tsp3);

// We have to release the typed scalars on unevaluated tree branches...
// Where are the evaluated ones released???

static void release_branch(QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	switch(enp->sen_code){
		case N_LITNUM:
			RELEASE_SCALAR( enp->sen_tsp );
			break;
		case N_MATH0FUNC:
		case N_STRFUNC:
		case N_STR2FUNC:
		case N_STR3FUNC:
		case N_QUOT_STR:
		// why are these three do-nothings?
		case N_OBJNAME:
		case N_SUBSCRIPT:
		case N_CSUBSCRIPT:

			break;
		case N_MATH1FUNC:
		case N_INT1FUNC:
		case N_DATAFUNC:
		case N_SIZFUNC:
		case N_TSFUNC:
		case N_ILACEFUNC:
		case N_POSNFUNC:
		case N_SCALAR_OBJ:
		case N_NOT:
		case N_BITCOMP:
		case N_UMINUS:
			release_branch(QSP_ARG  enp->sen_child[0]);
			break;
		case N_MATH2FUNC:
		case N_PLUS:
		case N_MINUS:
		case N_DIVIDE:
		case N_TIMES:
		case N_MODULO:
		case N_BITAND:
		case N_BITOR:
		case N_BITXOR:
		case N_SHL:
		case N_SHR:
		case N_EQUIV:
		case N_LOGOR:
		case N_LOGAND:
		case N_LOGXOR:
		case N_LT: case N_GT: case N_LE: case N_GE:
		case N_NE:
			release_branch(QSP_ARG  enp->sen_child[0]);
			release_branch(QSP_ARG  enp->sen_child[1]);
			break;
		case N_CONDITIONAL:
			release_branch(QSP_ARG  enp->sen_child[0]);
			release_branch(QSP_ARG  enp->sen_child[1]);
			release_branch(QSP_ARG  enp->sen_child[2]);
			break;
		default:
			fprintf(stderr,"release_branch:  unhandled expression node!? (code = %d, 0x%x)\n",enp->sen_code,enp->sen_code);
			dump_etree(QSP_ARG  enp);
			break;
	}
}

Typed_Scalar * eval_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	double dval,dval2,dval3;
	const char *s, *s2;
	//char *dst;
	Data_Obj *dp;
	Item *szp;
	INT_TYPE ival,ival2,ival3;
	UINT_TYPE uval,uval2,uval3;
	Typed_Scalar *tsp, *tsp2, *tsp3;

	dimension_t frm;
	static Function *val_func_p=NO_FUNCTION;

#ifdef QUIP_DEBUG
if( debug & expr_debug ){
sprintf(ERROR_STRING,"eval_expr:  code = %d",enp->sen_code);
ADVISE(ERROR_STRING);
dump_enode(QSP_ARG  enp);
}
#endif /* QUIP_DEBUG */

	switch(enp->sen_code){

	case N_MATH0FUNC:		// eval_expr
		dval = evalD0Function(enp->sen_func_p);
		tsp = scalar_for_double(dval);
		break;
	case N_MATH1FUNC:		// eval_expr
		GET_ONE_DOUBLE
		dval = evalD1Function(enp->sen_func_p,dval2);
		tsp = scalar_for_double(dval);
		break;
	case N_INT1FUNC:		// eval_expr
		// BUG?  should this be a double arg???
		GET_ONE_LONG
		ival = evalI1Function(enp->sen_func_p,ival2);
		tsp = scalar_for_long((long)ival);
		break;
	case N_CHARFUNC:		// eval_expr
		GET_ONE_DOUBLE
		dval = evalCharFunction(enp->sen_func_p,(char)dval2);
		tsp = scalar_for_double(dval);
		break;
	case N_MATH2FUNC:		// eval_expr
		GET_TWO_DOUBLES
		dval = evalD2Function(enp->sen_func_p,dval2,dval3);
		tsp = scalar_for_double(dval);
		break;
	case N_DATAFUNC:		// eval_expr
		/*
		dp = eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
		*/
		dp = (*eval_dobj_func)(QSP_ARG enp->sen_child[0]);
		if( dp == NULL ){ dval = 0.0; } else {
		dval = (*enp->sen_func_p->fn_u.dobj_func)( QSP_ARG  dp ); }
		tsp = scalar_for_double(dval);
		break;
	case N_POSNFUNC:		// eval_expr
		/* We have problems mixing IOS objects and C structs... */
#ifdef BUILD_FOR_OBJC
		if( check_ios_positionable_func(&dval,enp->sen_func_p,
				enp->sen_child[0]) ){
			tsp = scalar_for_double(dval);
			return tsp;
		}
#endif /* BUILD_FOR_OBJC */
		szp = EVAL_PSNBL_EXPR(enp->sen_child[0]);
		dval = (*enp->sen_func_p->fn_u.posn_func)( QSP_ARG  szp );
		tsp = scalar_for_double(dval);
		break;
	case N_ILACEFUNC:		// eval_expr
		/* We have problems mixing IOS objects and C structs... */
#ifdef BUILD_FOR_OBJC
		if( check_ios_interlaceable_func(&dval,enp->sen_func_p,
				enp->sen_child[0]) ){
			tsp = scalar_for_double(dval);
			return tsp;
		}
#endif /* BUILD_FOR_OBJC */
		szp = EVAL_ILBL_EXPR(enp->sen_child[0]);
		dval = (*enp->sen_func_p->fn_u.il_func)( QSP_ARG  szp );
		tsp = scalar_for_double(dval);
		break;
	case N_SIZFUNC:		// eval_expr
		/* We have problems mixing IOS objects and C structs... */
#ifdef BUILD_FOR_OBJC
		if( check_ios_sizable_func(&dval,enp->sen_func_p,
				enp->sen_child[0]) ){
			// BUG?  does the function get called in check_ios_sizable_func???
			tsp = scalar_for_double(dval);
			return tsp;
		}
#endif /* BUILD_FOR_OBJC */
		szp = EVAL_SZBL_EXPR_FUNC(enp->sen_child[0]);
		dval = (*enp->sen_func_p->fn_u.sz_func)( QSP_ARG  szp );
		tsp = scalar_for_double(dval);
		break;
	case N_STRVFUNC:		// eval_expr
		/* We have problems mixing IOS objects and C structs... */
#ifdef BUILD_FOR_OBJC
		if( check_ios_strv_func(&s,enp->sen_func_p,
				enp->sen_child[0]) ){
			tsp = scalar_for_string(s);
			return tsp;
		}
#endif /* BUILD_FOR_OBJC */
		// BUG - if this is to support the precision function, then
		// the sizable lookup should be done within the function, not here!
		/*
		szp = EVAL_SZBL_EXPR_FUNC(enp->sen_child[0]);
		s = (*enp->sen_func_p->fn_u.strv_func)( QSP_ARG  szp );
		*/
		s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
		s = (*enp->sen_func_p->fn_u.strv_func)( QSP_ARG s );
		tsp = scalar_for_string(s);
		break;
	case N_STRV2FUNC:		// eval_expr
#ifdef BUILD_FOR_OBJC
		if( check_ios_strv2_func(&s,enp->sen_func_p,
				enp->sen_child[0],enp->sen_child[0]) ){
			tsp = scalar_for_string(s);
			return tsp;
		}
#endif /* BUILD_FOR_OBJC */
		s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
		s2 = EVAL_SCALEXP_STRING(enp->sen_child[1]);
		s = (*enp->sen_func_p->fn_u.strv2_func)( QSP_ARG s, s2 );
		tsp = scalar_for_string(s);
		break;

	case N_TSFUNC:		// eval_expr
		szp = eval_tsbl_expr(QSP_ARG  enp->sen_child[0]);
		tsp2=EVAL_EXPR(enp->sen_child[1]);
		frm = (dimension_t)double_for_scalar(tsp2);
		RELEASE_FIRST
		dval = (*enp->sen_func_p->fn_u.ts_func)( QSP_ARG  szp, frm );
		tsp = scalar_for_double(dval);
		break;
	case N_STRFUNC:		// eval_expr
		s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
		dval = evalStr1Function(QSP_ARG  enp->sen_func_p,s);
		tsp = scalar_for_double(dval);
		break;

	case N_STR2FUNC:		// eval_expr
		s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
		s2 = EVAL_SCALEXP_STRING(enp->sen_child[1]);
		dval = evalStr2Function(enp->sen_func_p,s,s2);
		tsp = scalar_for_double(dval);
		break;

	case N_STR3FUNC:		// eval_expr
		s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
		s2 = EVAL_SCALEXP_STRING(enp->sen_child[1]);
		dval = evalStr3Function( enp->sen_func_p, s, s2,
			(int) double_for_scalar( EVAL_EXPR( enp->sen_child[2]) ) );
		tsp = scalar_for_double(dval);
		break;

//#ifdef FOOBAR
//	case N_STRVFUNC:		// eval_expr
//		// string valued functions, tolower toupper etc
//		s = EVAL_SCALEXP_STRING(enp->sen_child[0]);
//		// We take advantage of knowing that the input
//		// and output strings should have the same lengths...
//
//		// scalar_for_string doesn't allocate new string storage!?
//		dst=get_expr_stringbuf(WHICH_EXPR_STR,strlen(s));
//		ADVANCE_EXPR_STR
//		evalStrVFunction( enp->sen_func_p, dst, s );
//		tsp = scalar_for_string(dst);
//		break;
//#endif // FOOBAR

	case N_PLUS:		// eval_expr
		GET_TWO_DOUBLES
		dval=dval2+dval3;
		tsp = scalar_for_double(dval);
		break;

	case N_SCALAR_OBJ:		// eval_expr
		/* instead of explicitly extracting the scalar value,
		 * we just call the function from the datafunctbl...
		 * We look up the function using the name "value"...
		 */

		/*
		dp = eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
		*/
		dp = (*eval_dobj_func)(QSP_ARG enp->sen_child[0]);
		if( dp == NULL ){ dval = 0.0; } else {
		/* BUG the 0 index in the line below is only correct
		 * because the "value" function is the first one
		 * in the data_functbl - this should be a symbolic
		 * constant, or better yet, the table should be scanned
		 * for the string "value", and the index stored in a var.
		 * But this is good enough for now...
		 */

		if( val_func_p == NO_FUNCTION )
			val_func_p = function_of(QSP_ARG  "value");
//#ifdef CAUTIOUS
//		if( val_func_p == NO_FUNCTION ){
//			ERROR1("CAUTIOUS:  couldn't find object value function!?");
//			IOS_RETURN_VAL(NULL);
//		}
//#endif /* CAUTIOUS */
		assert( val_func_p != NO_FUNCTION );

		/* This seems buggy - there should be a value function
		 * that returns a typed scalar!?
		 */
		dval = (*val_func_p->fn_u.dobj_func)( QSP_ARG  dp ); }
		tsp = scalar_for_double(dval);

		break;

	/* do-nothing */
	case N_OBJNAME:			// eval_expr
	case N_SUBSCRIPT:		// eval_expr
	case N_CSUBSCRIPT:		// eval_expr
//#ifdef FOOBAR
//	case N_SIZABLE:
//	case N_SUBSIZ:
//	case N_CSUBSIZ:
//	case N_TSABLE:
//#endif /* FOOBAR */
		sprintf(ERROR_STRING,
			"unexpected case (%d) in eval_expr",
			enp->sen_code);
		NWARN(ERROR_STRING);
		SET_RESULT_ZERO
		break;

	case N_LITNUM:		// eval_expr
		tsp = enp->sen_tsp;
		// WHEN SHOULD WE RELEASE THIS???
		RELEASE_SCALAR(enp->sen_tsp);	// Is this OK?
		break;

	case N_LITSTR:
//advise("eval_expr:  literal string seen");
		tsp = enp->sen_tsp;
//show_typed_scalar(tsp);
		// WHEN SHOULD WE RELEASE THIS???
		RELEASE_SCALAR(enp->sen_tsp);	// Is this OK?
						// We are still pointing to it!?
		break;

	case N_MINUS:		// eval_expr
		GET_TWO_DOUBLES
		dval=dval2-dval3;
		tsp = scalar_for_double(dval);
		break;
	case N_DIVIDE:		// eval_expr
		GET_TWO_DOUBLES
		if( dval3==0.0 ){
			divzer_error(SINGLE_QSP_ARG);
			dval=0;
		} else dval=dval2/dval3;
		tsp = scalar_for_double(dval);
		break;
	case N_TIMES:		// eval_expr
		GET_TWO_DOUBLES
		dval=dval2*dval3;
		tsp = scalar_for_double(dval);
		break;


	case N_MODULO:		// eval_expr
		GET_TWO_LONGS
		if( ival3==0.0 ){
			divzer_error(SINGLE_QSP_ARG);
			ival=0;
		} else ival=(ival2%ival3);
		tsp = SCALAR_FOR_INT_TYPE(ival);
		break;
	case N_BITAND:		// eval_expr
		// BUG?  when we converted these to signed integers,
		// bad things happened...
		GET_TWO_ULONGS
		uval = (uval2&uval3);
		tsp = SCALAR_FOR_INT_TYPE(uval);
		break;
	case N_BITOR:		// eval_expr
		GET_TWO_ULONGS
		uval = (uval2|uval3);
		tsp = SCALAR_FOR_INT_TYPE(uval);
		break;
	case N_BITXOR:		// eval_expr
		GET_TWO_ULONGS
		uval = (uval2^uval3);
		tsp = SCALAR_FOR_INT_TYPE(uval);
		break;
	case N_SHL:		// eval_expr
		GET_TWO_ULONGS
		uval =  uval2 << uval3;
		tsp = SCALAR_FOR_INT_TYPE(uval);
		break;
	case N_NOT:		// eval_expr
		tsp2=EVAL_EXPR(enp->sen_child[0]);
		SET_RESULT( has_zero_value(tsp2) )
		RELEASE_FIRST
		break;
	case N_BITCOMP:		// eval_expr
		tsp2=EVAL_EXPR(enp->sen_child[0]);
		uval2=llong_for_scalar(tsp2);
		RELEASE_FIRST
		uval = (~uval2);
		tsp = SCALAR_FOR_INT_TYPE(uval);
		break;
	case N_UMINUS:		// eval_expr
		tsp2=EVAL_EXPR(enp->sen_child[0]);
		dval2=double_for_scalar(tsp2);
		RELEASE_FIRST
		dval=(-dval2);
		tsp = scalar_for_double(dval);
		break;

	case N_EQUIV:		// eval_expr
		tsp2=EVAL_EXPR(enp->sen_child[0]);
		tsp3=EVAL_EXPR(enp->sen_child[1]);
		if( scalars_are_equal(tsp2,tsp3) )
			tsp = &ts_dbl_one;
		else
			tsp = &ts_dbl_zero;
		//if( dval2 == dval3 ) dval=1.0;
		//else dval=0.0;
		RELEASE_BOTH
		break;
	case N_LOGOR:		// eval_expr
		tsp2=EVAL_EXPR(enp->sen_child[0]);
		if( ! has_zero_value(tsp2) ){
			tsp=&ts_dbl_one;
			release_branch(QSP_ARG  enp->sen_child[1]);
		} else {
			tsp3=EVAL_EXPR(enp->sen_child[1]);
			if( ! has_zero_value(tsp3) ) tsp=&ts_dbl_one;
			else tsp=&ts_dbl_zero;
			RELEASE_SECOND
		}
		RELEASE_FIRST
		break;
	case N_LOGAND:		// eval_expr
		tsp2=EVAL_EXPR(enp->sen_child[0]);
		if( has_zero_value(tsp2) ){
			tsp=&ts_dbl_zero;
			release_branch(QSP_ARG  enp->sen_child[1]);
		} else {
			tsp3=EVAL_EXPR(enp->sen_child[1]);
			if( has_zero_value(tsp3) ) tsp=&ts_dbl_zero;
			else tsp=&ts_dbl_one;
			RELEASE_SECOND
		}
		RELEASE_FIRST
		break;
	case N_LOGXOR:		// eval_expr
		tsp2=EVAL_EXPR(enp->sen_child[0]);
		tsp3=EVAL_EXPR(enp->sen_child[1]);
		{ int z2,z3;
		z2=has_zero_value(tsp2);
		z3=has_zero_value(tsp3);
		if( ( z2 && ! z3) || ( (!z2) && z3 ) ){
			tsp = &ts_dbl_one;
		} else {
			tsp = &ts_dbl_zero;
		}
		}
		RELEASE_BOTH
		break;
	// BUG?  these numeric comparisons should be done for each
	// type instead of casting to double.
	case N_LT:		// eval_expr
		GET_TWO_DOUBLES
		SET_RESULT( dval2 < dval3 )
		break;
	case N_GT:		// eval_expr
		GET_TWO_DOUBLES
		SET_RESULT( dval2 > dval3 )
		break;
	case N_GE:		// eval_expr
		GET_TWO_DOUBLES
		if( dval2 >= dval3 )
			tsp = &ts_dbl_one;
		else
			tsp = &ts_dbl_zero;
		break;
	case N_LE:		// eval_expr
		GET_TWO_DOUBLES
		if( dval2 <= dval3 )
			tsp = &ts_dbl_one;
		else 
			tsp = &ts_dbl_zero;
		break;
	case N_NE:		// eval_expr
		GET_TWO_DOUBLES
		if( scalars_are_equal(tsp2,tsp3) )
			tsp = &ts_dbl_zero;
		else
			tsp = &ts_dbl_one;
		break;

	case N_SHR:		// eval_expr
		tsp2=EVAL_EXPR(enp->sen_child[0]);
		tsp3=EVAL_EXPR(enp->sen_child[1]);
		uval2 = llong_for_scalar(tsp2);
		ival = llong_for_scalar(tsp3);
		RELEASE_BOTH
		/* this version clears the sign bit */
		/* dval = ((((int)dval2) >> 1) & 077777) >> ((ival) - 1); */
		uval = (uval2)>>ival;
		tsp=SCALAR_FOR_INT_TYPE(uval);
		break;
	case N_CONDITIONAL:		// eval_expr
		tsp2 = EVAL_EXPR(enp->sen_child[0]);
		if( ! has_zero_value(tsp2) ) {
			tsp = EVAL_EXPR(enp->sen_child[1]);
			release_branch( QSP_ARG   enp->sen_child[2]);
		} else {
			tsp = EVAL_EXPR(enp->sen_child[2]);
			release_branch(QSP_ARG  enp->sen_child[1]);
		}
		RELEASE_FIRST
		break;

//#ifdef FOOBAR
//	case N_SLCT_CHAR:		// eval_expr
//ADVISE("case N_SLCT_CHAR");
//		ival = EVAL_EXPR(enp->sen_child[0]);
//		if( ival < 0 || ival >= strlen(enp->sen_string) )
//			dval = -1;
//		else
//			dval = enp->sen_string[ival];
//		break;
//#endif /* FOOBAR */

//#ifdef CAUTIOUS
	default:		// eval_expr
//		sprintf(ERROR_STRING,
//			"CAUTIOUS:  %s - %s:  unhandled node code case %d!?",
//			WHENCE2(eval_expr),
//			enp->sen_code);
//		NWARN(ERROR_STRING);
//		SET_RESULT_ZERO
		assert(0);
		break;
//#endif /* CAUTIOUS */

	}

	return(tsp);
} // eval_expr

/*static*/ void dump_etree(QSP_ARG_DECL  Scalar_Expr_Node *enp)
{
	if( enp == NULL ) return;
	dump_enode(QSP_ARG  enp);
	dump_etree(QSP_ARG  enp->sen_child[0]);
	dump_etree(QSP_ARG  enp->sen_child[1]);
}

/* We used to call yyerror for lexical scan errors, but that often
 * caused yyerror to be called twice, the first time for the lexer
 * error, the second time with a parsing error...
 */

static void llerror(const char *msg)
{
	char tmp_str[LLEN];	/* don't use error_string! */

	sprintf(tmp_str,"pexpr lexical scan error:  %s",msg);
	NWARN(tmp_str);
}

#define BUF_CHAR(c)					\
							\
	if( i > buflen ){				\
		NWARN("extract_number_string:  string buffer too small!?");	\
		return(-1);				\
	}						\
	buf[i++] = (c);


static int extract_number_string(char *buf, int buflen, const char **srcp)
{
	int i=0;
	const char *s;
	//int sign=1;
	int retval=0;

	s = *srcp;

	while( isspace(*s) && *s!=0 ) s++;

	if( *s == 0 ) return(-1);

	if( (!isdigit(*s)) && *s!='+' && *s!='-' && *s!='.' ) return(-1);

	if( *s == '+' ) s++;
	else if( *s == '-' ){
		//sign=(-1);
		s++;
	}

	if( *s == '.' )
		goto decimal;

	if( !isdigit(*s) ) return(-1);

	if( *s == '0' ){
		retval |= 1;		/* indicate leading zero */
		if( *(s+1) == 'x' ){
			BUF_CHAR(*s)	/* the 0 */
			s++;
			BUF_CHAR(*s)	/* the x */
			s++;
			if( ! isxdigit(*s) ){
				NWARN("extract_number_string:  mal-formed hex string");
				return(-1);
			}
			while( isxdigit(*s) ){
				BUF_CHAR(*s)
				s++;
			}
			*srcp = s;
			BUF_CHAR(0)		/* null termination */
			return(4);		/* indicate hex */
		}
	}

	while( isdigit(*s) ){
		BUF_CHAR(*s)
		s++;
	}
	if( *s == '.' ){
decimal:
		BUF_CHAR(*s)
		s++;
		while( isdigit(*s) ){
			BUF_CHAR(*s)
			s++;
		}
		retval |= 2;		/* indicate decimal pt seen */
	}
	if( *s == 'e' || *s == 'E' ){
		BUF_CHAR(*s)
		s++;
		if( *s == '-' || *s == '+' ){
			BUF_CHAR(*s)
			s++;
		}
		if( !isdigit(*s) ){
			NWARN("extract_number_string:  malformed exponent!?");
			return(-1);
		}
		while( isdigit(*s) ){
			BUF_CHAR(*s)
			s++;
		}
	}

	*srcp = s;
	BUF_CHAR(0)

	return(retval);
}


/*
 * Parse a number string.
 * Strings can be as in C: decimal, hex following a leading 0x,
 * octal following a leading 0.  Floating point notation is parsed.
 *
 * The argument strptr is the address of a pointer into a string buffer
 * containing the text.  This pointer will be advanced so that on
 * return it points to the next character in the buffer following
 * the parsed number.
 */

Typed_Scalar *parse_number(QSP_ARG_DECL  const char **strptr)
{
	/* the pointed-to text is not necessarily null-terminated... */
	const char *ptr;
	char *endptr;
	double d;
	long l;
	char buf[128];
	int status;

	ptr = *strptr;

	status = extract_number_string(buf,128,&ptr);
	*strptr = ptr;
	if( status < 0 ){
		sprintf(DEFAULT_ERROR_STRING,"parse_number:  bad number string \"%s\"",
			ptr);
		NWARN(DEFAULT_ERROR_STRING);
		//return(-1);
		return(&ts_dbl_minus_one);
	}
	if( status == 4					/* hex */
		|| status == 1				/* leading zero, no decimal */
		){

		// This includes the string "0" !?
		errno = 0;
		l=strtol(buf,&endptr,0);
		if( errno == ERANGE ){
#ifdef HAVE_STRTOLL
			long long ll1;

			errno=0;
//advise("trying strtoll...");
			ll1=strtoll(buf,&endptr,0);
			if( errno == ERANGE ){
				// try unsigned conversion
//advise("trying strtoull...");
				// errno is not reset to zero
				// when there is no error...
				errno=0;
				ll1=strtoull(buf,&endptr,0);
				if( errno == ERANGE ){
					sprintf(DEFAULT_ERROR_STRING,"parse_number %s:  long long conversion error!?  (errno=%d)",buf,errno);
					NWARN(DEFAULT_ERROR_STRING);
					tell_sys_error("strtoull");
					sprintf(DEFAULT_ERROR_STRING,"value returned:  0x%llx",ll1);
					NADVISE(DEFAULT_ERROR_STRING);
					sprintf(DEFAULT_ERROR_STRING,"unsigned long long max:  0x%llx",ULLONG_MAX);
					NADVISE(DEFAULT_ERROR_STRING);
				}
			}
			if( errno != 0 ){
				sprintf(DEFAULT_ERROR_STRING,"parse_number %s:  long long conversion error!?  (errno=%d)",buf,errno);
				NWARN(DEFAULT_ERROR_STRING);
				tell_sys_error("strtoll");
				sprintf(DEFAULT_ERROR_STRING,"value returned:  0x%llx",ll1);
				NADVISE(DEFAULT_ERROR_STRING);
				sprintf(DEFAULT_ERROR_STRING,"long long range:  0x%llx - 0x%llx",LLONG_MIN,LLONG_MAX);
				NADVISE(DEFAULT_ERROR_STRING);
			}
			return( SCALAR_FOR_INT_TYPE(ll1) );
			//return ll1;
#else // ! HAVE_STRTOLL
			sprintf(DEFAULT_ERROR_STRING,"long conversion error!?  (errno=%d)",errno);
			NWARN(DEFAULT_ERROR_STRING);
			tell_sys_error("strtol");
#endif // ! HAVE_STRTOLL

		} else if( errno != 0 ){
			sprintf(DEFAULT_ERROR_STRING,"long conversion error!?  (errno=%d)",errno);
			NWARN(DEFAULT_ERROR_STRING);
			tell_sys_error("strtol");
		}
		return( scalar_for_long(l) );
	}
	/*   else if( status & 2 ){ */	/* decimal pt seen */

		errno = 0;
//sprintf(ERROR_STRING,"converting string \"%s\"",buf);
//ADVISE(ERROR_STRING);
		d = strtod(buf,&endptr);
		if( errno == ERANGE ){
			// This message is printing, but the returned
			// value is not +-HUGE_VAL, in contradiction to
			// the documentation.
			if( d == 0.0 ){
sprintf(DEFAULT_ERROR_STRING,"strtod:  possible underflow buf=\"%s\", d = %g",buf,d);
ADVISE(DEFAULT_ERROR_STRING);
			} else if( d == HUGE_VAL || d == -HUGE_VAL ){
sprintf(DEFAULT_ERROR_STRING,"strtod:  possible overflow buf=\"%s\", d = %g  HUGE_VAL = %g",buf,d,HUGE_VAL);
ADVISE(DEFAULT_ERROR_STRING);
			} else {
				if( verbose ){
sprintf(DEFAULT_ERROR_STRING,"strtod:  possible overflow (inconsistent) buf=\"%s\", d = %g  HUGE_VAL = %g",buf,d,HUGE_VAL);
ADVISE(DEFAULT_ERROR_STRING);
				}
			}
		} else if( errno != 0 ){
			sprintf(DEFAULT_ERROR_STRING,"double conversion error!?  (errno=%d)",errno);
			NWARN(DEFAULT_ERROR_STRING);
			tell_sys_error("strtod");
		}

//sprintf(DEFAULT_ERROR_STRING,"flt conversion returning %lg",d);
//ADVISE(DEFAULT_ERROR_STRING);
//		return( scalar_for_double(d) );

	{
		Typed_Scalar *tsp;
		tsp = scalar_for_double(d) ;
		return( tsp );
	}
	/* } */

} /* end parse_number() */

//static double yynumber(SINGLE_QSP_ARG_DECL)
static Typed_Scalar * yynumber(SINGLE_QSP_ARG_DECL)
{
	Typed_Scalar *tsp;

//sprintf(ERROR_STRING,"yynumber calling parse_number %s",YYSTRPTR[EDEPTH]);
//ADVISE(ERROR_STRING);
	tsp = parse_number(DEFAULT_QSP_ARG  (const char **)&YYSTRPTR[EDEPTH]);
	return tsp;
}

#define TMPBUF_LEN	128

static const char * varval(void)
{
	char tmpbuf[TMPBUF_LEN];
	const char *s;
	const char *var_valstr;
	int c;

	/* indirect variable reference? */

	if( *YYSTRPTR[EDEPTH] == '$' ){
		YYSTRPTR[EDEPTH]++;
		s = varval() ;
	} else {
		/* read in the variable name */
		char *sp;
		int n_stored=0;
		sp=tmpbuf;
		c=(*YYSTRPTR[EDEPTH]);
		while( isalpha(c) || c == '_' || isdigit(c) ){
			*sp++ = (char) c;
			if( ++n_stored >= TMPBUF_LEN ){
				NWARN("varval:  buffer overrun!?");
				sp--;
			}

			YYSTRPTR[EDEPTH]++;
			c=(*YYSTRPTR[EDEPTH]);
		}
		*sp=0;
		s=tmpbuf;
	}

	var_valstr = var_value(DEFAULT_QSP_ARG  s);

	if( var_valstr == NULL )
		return("0");
	else
		return( var_valstr );
}


static int token_for_func_type(int type)
{
	switch( type ){
		case D0_FUNCTYP:	return(MATH0_FUNC);	break;
		case D1_FUNCTYP:	return(MATH1_FUNC);	break;
		case D2_FUNCTYP:	return(MATH2_FUNC);	break;
		case I1_FUNCTYP:	return(INT1_FUNC);	break;
		case STR1_FUNCTYP:	return(STR_FUNC);	break;
		case STR2_FUNCTYP:	return(STR2_FUNC);	break;
		case STR3_FUNCTYP:	return(STR3_FUNC);	break;
		case STRV_FUNCTYP:	return(STRV_FUNC);	break;
		case STRV2_FUNCTYP:	return(STRV2_FUNC);	break;
		case CHAR_FUNCTYP:	return(CHAR_FUNC);	break;
		case SIZE_FUNCTYP:	return(SIZE_FUNC);	break;
		case DOBJ_FUNCTYP:	return(DATA_FUNC);	break;
		case TS_FUNCTYP:	return(TS_FUNC);	break;
		case ILACE_FUNCTYP:	return(IL_FUNC);	break;
		case POSN_FUNCTYP:	return(POSN_FUNC);	break;
//#ifdef CAUTIOUS
		default:
//			NERROR1("CAUTIOUS:  token_for_func_type:  bad type!?");
			assert( AERROR("token_for_func_type:  bad type!?") );
			break;
//#endif /* CAUTIOUS */
	}
	return(-1);
}


#ifdef FOOBAR
#ifdef THREAD_SAFE_QUERY
static int yylex(YYSTYPE *yylvp, Query_Stack *qsp)	/* return the next token */
#else /* ! THREAD_SAFE_QUERY */
static int yylex(YYSTYPE *yylvp)			/* return the next token */
#endif /* ! THREAD_SAFE_QUERY */
#endif // FOOBAR

// With new bison, we can't specify these args conditionally
// without some effort...
static int yylex(YYSTYPE *yylvp, Query_Stack *qsp)	/* return the next token */
{
	int c;
	char *s;

	if( IS_HALTING(THIS_QSP) ) return(0);

	while( EDEPTH >= 0 ){
		/* skip spaces */

		while( *YYSTRPTR[EDEPTH]
			&& isspace(*YYSTRPTR[EDEPTH]) )
			YYSTRPTR[EDEPTH]++;
		/* pop if line empty */
		if( *YYSTRPTR[EDEPTH] == 0 ) {
			EDEPTH--;
			continue;
		}
		c=(*YYSTRPTR[EDEPTH]);
		if( isdigit(c) || c=='.' ) {
			yylvp->tsp=yynumber(SINGLE_QSP_ARG);
			return(NUMBER);
		} else if( c == '$' ) {
			YYSTRPTR[EDEPTH]++;
			if( (EDEPTH+1) >= MAXEDEPTH ){
				LLERROR("expression depth too large");
				return(0);
			}
			YYSTRPTR[EDEPTH+1]=varval();
			/* varval should advance YYSTRPTR[edpth] */
			EDEPTH++;
			/* keep looping */
		} else if( IS_LEGAL_FIRST_CHAR(c) ){	/* get a name */
			int n=1;
			s=get_expr_stringbuf(WHICH_EXPR_STR,strlen(YYSTRPTR[EDEPTH]));
			*s++ = (*YYSTRPTR[EDEPTH]++);
			while( IS_LEGAL_NAME_CHAR(*YYSTRPTR[EDEPTH]) ){
				*s++ = (*YYSTRPTR[EDEPTH]++);
				n++;
//#ifdef CAUTIOUS
//				if( n >= expr_string[WHICH_EXPR_STR]->sb_size ){
//					LLERROR("string buffer overflow #1");
//					s--;
//					n--;
//				}
//#endif // CAUTIOUS
				assert(n< SB_SIZE(expr_string[WHICH_EXPR_STR]));
			}
			*s=0;

			yylvp->func_p = function_of(QSP_ARG
				expr_string[WHICH_EXPR_STR]->sb_buf);
			if( yylvp->func_p != NULL ){
				int t;
				t = token_for_func_type(FUNC_TYPE(yylvp->func_p));
				return t;
			}

			// use an array of tsp's here instead???
			//yylvp->e_string=expr_string[WHICH_EXPR_STR]->sb_buf;
			string_scalar[WHICH_EXPR_STR].ts_value.u_vp=
				expr_string[WHICH_EXPR_STR]->sb_buf;
			yylvp->tsp=(&string_scalar[WHICH_EXPR_STR]);
			ADVANCE_EXPR_STR
			return(E_STRING);	/* unquoted string */

		} else if( ispunct(c) ){
			YYSTRPTR[EDEPTH]++;
			yylvp->fundex=c;

			if( c=='>' ){
				if( *YYSTRPTR[EDEPTH] == '>' ){
					YYSTRPTR[EDEPTH]++;
					return(SHR);
				} else if( *YYSTRPTR[EDEPTH] == '=' ){
					YYSTRPTR[EDEPTH]++;
					return(GE);
				}
			} else if( c=='<' ){
				if( *YYSTRPTR[EDEPTH]=='<' ){
					YYSTRPTR[EDEPTH]++;
					return(SHL);
				} else if( *YYSTRPTR[EDEPTH] == '=' ){
					YYSTRPTR[EDEPTH]++;
					return(LE);
				}
			} else if( c == '=' ){
				if( *YYSTRPTR[EDEPTH] == '=' ){
					YYSTRPTR[EDEPTH]++;
					return(EQUIV);
				}
			} else if( c == '|' ){
				if( *YYSTRPTR[EDEPTH] == '|' ){
					YYSTRPTR[EDEPTH]++;
					return(LOGOR);
				}
			} else if( c == '&' ){
				if( *YYSTRPTR[EDEPTH] == '&' ){
					YYSTRPTR[EDEPTH]++;
					return(LOGAND);
				}
			} else if( c == '^' ){
				if( *YYSTRPTR[EDEPTH] == '^' ){
					YYSTRPTR[EDEPTH]++;
					return(LOGXOR);
				}
			} else if( c == '!' ){
				if( *YYSTRPTR[EDEPTH] == '=' ){
					YYSTRPTR[EDEPTH]++;
					return(NE);
				}
			} else if ( c == '"' || c == '\'' ){
				/* We don't currently handle C-style
				 * character constants, e.g. 'a' etc.
				 * But it would be convenient to do so?
				 * Maybe it would break fewer things if
				 * instead we introduced a function that
				 * returns the ascii value of a length 1 string,
				 * called "char" or "ascii"??
				 * e.g.  ascii('a') or ascii("a")
				 */

				int qchar;
				int n=0;

				/* This is the first character of a
				 * quoted string.  Read characters in
				 * until we reach a match.  We would
				 * like to skip escaped quotes...
				 */
				qchar=c;
				s=get_expr_stringbuf(WHICH_EXPR_STR,strlen(YYSTRPTR[EDEPTH]));
				/* copy string into a buffer */
				c = *YYSTRPTR[EDEPTH];
				while( c && c != qchar ){
					if( c == '\\' ){
						YYSTRPTR[EDEPTH]++;
						c = *YYSTRPTR[EDEPTH];
						if( c == 0 ){
							// syntax error
				LLERROR("unmatched quote");
						} else {
							*s++=(char)c;
							YYSTRPTR[EDEPTH]++;
							n++;
						}
					} else {
						*s++=(char)c;
						YYSTRPTR[EDEPTH]++;
						n++;
					}
					c = *YYSTRPTR[EDEPTH];
//#ifdef CAUTIOUS
//					if( n >= expr_string[WHICH_EXPR_STR]->sb_size ){
//						LLERROR("CAUTIOUS:  string buffer overflow #2");
//						s--;
//						*s = 0;
//						//LLERROR(_strbuf[WHICH_EXPR_STR]);
//						n--;
//					}
//#endif // CAUTIOUS
					assert(n<SB_SIZE(expr_string[WHICH_EXPR_STR]));
				}
				*s=0;
				if( *YYSTRPTR[EDEPTH] == qchar ){
					YYSTRPTR[EDEPTH]++;
					/* used to call var_expand here,
					 * but now this is done automatically.
					 */
				} else LLERROR("unmatched quote");

				yylvp->tsp=(&string_scalar[WHICH_EXPR_STR]);
				string_scalar[WHICH_EXPR_STR].ts_value.u_vp
					= expr_string[WHICH_EXPR_STR]->sb_buf;

				ADVANCE_EXPR_STR
#ifdef QUIP_DEBUG
//if( debug & expr_debug ){
//ADVISE("yylex:  quoted E_STRING");
//}
#endif /* QUIP_DEBUG */
				return(E_QSTRING);	/* quoted string */
			}
#ifdef QUIP_DEBUG
//if( debug & expr_debug ){
//ADVISE("yylex:  punct char");
//}
#endif /* QUIP_DEBUG */

			return(c);
		} else {
			LLERROR("yylex error");
			return(0);
		}
	}
#ifdef QUIP_DEBUG
//if( debug & expr_debug ){
//ADVISE("yylex:  0 (EDEPTH<0)");
//}
#endif /* QUIP_DEBUG */

	return(0);
}

/* rls_tree should only be called when locked */

static void rls_tree( Scalar_Expr_Node *enp )
{
	Node *np;

	if( enp->sen_child[0] != NO_EXPR_NODE )
		rls_tree(enp->sen_child[0]);
	if( enp->sen_child[1] != NO_EXPR_NODE )
		rls_tree(enp->sen_child[1]);
	/*
	if( enp->sen_string != NULL )
		rls_str(enp->sen_string);
	if( enp->sen_string2 != NULL )
		rls_str(enp->sen_string2);
		*/

	if( free_enp_lp == NO_LIST ){
		free_enp_lp = new_list();
//#ifdef CAUTIOUS
//		if( free_enp_lp == NO_LIST ) NERROR1("CAUTIOUS:  rls_tree:  error creating free enp list");
//#endif /* CAUTIOUS */
		assert( free_enp_lp != NO_LIST );
	}
	np = mk_node(enp);
	addHead(free_enp_lp,np);
}

static void initialize_estrings(void)
{
	// BUG using these static structs is not thread-safe!

	int i;

	for(i=0;i<MAX_E_STRINGS;i++){
		expr_string[i]=NULL;
		string_scalar[i].ts_value.u_vp = NULL;
		string_scalar[i].ts_prec_code = PREC_STR;
	}
	estrings_inited=1;
}

/*
 * Yacc doesn't allow recursive calls to the parser,
 * so we check for a recursive call here.  This arose when
 * we allowed indexed data objects to be valid size function
 * arguments, since normally pars_obj calls pexpr to evaluate
 * the index.  When this happens we assume it's a number
 * and hope for the best.
 * (This is a BUG with no obvious solution.)  wait - fixable!
 *
 * Wait, there is a solution:  bison allows %pure_parser which
 * generates a reentrant parser, so we could call it recursively...
 * Should look into this fixable BUG.
 *
 * We'd like for error messages to be printed with an input file
 * and line number.  We know how to do that, but in the multi-thread
 * environment, it requires a qsp...  We have the qsp here, but 
 * we don't have an easy way for yyparse to pass it to yyerror!?
 * YES WE DO:  YY_PARSE_PARAM!
 *
 * in_pexpr was a global flag, not thread-safe...
 */

/* double */
Typed_Scalar *
pexpr(QSP_ARG_DECL  const char *buf)	/** parse expression */
{
	int stat;
	Typed_Scalar *tsp;

	if( ! estrings_inited ) initialize_estrings();

#ifdef QUIP_DEBUG
	if( expr_debug <= 0 )
		expr_debug = add_debug_module(QSP_ARG  "expressions");
#endif /* QUIP_DEBUG */

#ifdef QUIP_DEBUG
if( debug & expr_debug ){
sprintf(ERROR_STRING,"%s - %s:  BEGIN %s, in_pexpr = %d",
WHENCE2(pexpr),buf,IN_PEXPR);
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	if( IN_PEXPR ) {
#ifdef QUIP_DEBUG
if( debug & expr_debug ){
ADVISE("pexpr:  nested call to pexpr, calling parse_number");
}
#endif /* QUIP_DEBUG */
		return( parse_number(QSP_ARG  &buf) );
	}

	IN_PEXPR=1;
	EDEPTH=0;
	YY_ORIGINAL=YYSTRPTR[EDEPTH]=buf;

	stat=yyparse(/*SINGLE_QSP_ARG*/ THIS_QSP );

	if( stat != 0 ){
		/* Need to somehow free allocated nodes... */
		if( verbose ){
			sprintf(ERROR_STRING,"yyparse returned status %d",stat);
			ADVISE(ERROR_STRING);
		}
		IN_PEXPR=0;
		//return(0.0);
		return(&ts_dbl_zero);
	}

#ifdef QUIP_DEBUG
if( debug & expr_debug ){
dump_etree(QSP_ARG  FINAL_EXPR_NODE_P);
}
#endif /* QUIP_DEBUG */

	tsp = EVAL_EXPR(FINAL_EXPR_NODE_P);

#ifdef QUIP_DEBUG
if( debug & expr_debug ){
if( tsp->ts_prec_code == PREC_DP ){
sprintf(ERROR_STRING,"pexpr:  s=\"%s\", dval = %g",buf,tsp->ts_value.u_d);
ADVISE(ERROR_STRING);
}
}
#endif /* QUIP_DEBUG */

#ifdef SUN
	/*
	 * this is a strange thing on the SUN, that
	 * zero can have the sign bit set, and that printf
	 * recognizes this and prints -0 !?
	 */

	if( tsp->ts_prec_code == PREC_DP ){
		double dval;
		dval = tsp->ts_value.u_d;
		if( iszero(dval) && signbit(dval) ){
			Scalar_Expr_Node *enp;

			enp=NODE0(N_LITNUM);
			//enp->sen_dblval=(-1.0);
			enp->sen_tsp = &ts_dbl_minus_one;
			enp=NODE2(N_TIMES,FINAL_EXPR_NODE_P,enp);
			FINAL_EXPR_NODE_P=enp;
			tsp2 = EVAL_EXPR(FINAL_EXPR_NODE_P);
			RELEASE_SCALAR(tsp)
			tsp = tsp2;
		}
	}
#endif /* SUN */

	LOCK_ENODES

	rls_tree(FINAL_EXPR_NODE_P);

	UNLOCK_ENODES

	IN_PEXPR=0;

	return( tsp );
}

// OLD:  We can't add a qsp arg here, because this one is defined by yacc...
// With bison v3, parse-param goes here too!

int yyerror(Query_Stack *qsp, char *s)
{
	// BUG - this is wrong if we have multiple
	// interpreter threads
	if( IS_HALTING(DEFAULT_QSP) )
		goto cleanup;
	
	sprintf(DEFAULT_ERROR_STRING,"parsing \"%s\"",YY_ORIGINAL);
	NADVISE(DEFAULT_ERROR_STRING);

	if( *YYSTRPTR[0] ){
		sprintf(DEFAULT_ERROR_STRING,"\"%s\" left to parse",YYSTRPTR[0]);
		NADVISE(DEFAULT_ERROR_STRING);
	} else {
		NADVISE("No buffered text left to parse");
	}

	// Print the warning after the informational messages, in case
	// this warning causes the program to exit.
	sprintf(DEFAULT_ERROR_STRING,"YYERROR:  %s",s);
	NWARN(DEFAULT_ERROR_STRING);

	/* final=(-1); */
	/* -1 is a bad value, because when the target is an
	 * unsigned in (dimension_t), the cast makes it a very
	 * large number...  causes all sorts of problems!
	 */
cleanup:
	//FINAL_EXPR_NODE_P=NO_EXPR_NODE;
	final_expr_node_p = NO_EXPR_NODE;
	/* BUG need to release nodes here... */
	return(0);
}

#ifdef NOT_USED
/*
 * This is the default data object locater.
 * If the data module were assumed to be always included
 * with the support library, then we wouldn't need this
 * here, but doing this allows us to run the parser
 * without the data module, but has the same grammer...
 */

static Data_Obj * _def_obj(QSP_ARG_DECL  const char *name)
{
	sprintf(DEFAULT_ERROR_STRING,"can't search for object \"%s\"; ",name);
	NWARN(DEFAULT_ERROR_STRING);

	NWARN("data module not linked");
	return(NULL);
}

static Data_Obj *_def_sub(QSP_ARG_DECL  Data_Obj *object,index_t index)
{
	NWARN("can't get subobject; data module not linked");
	return(NULL);
}


void set_obj_funcs(
			Data_Obj *(*ofunc)(QSP_ARG_DECL  const char *),
			Data_Obj *(*efunc)(QSP_ARG_DECL  const char *),
			Data_Obj *(*sfunc)(QSP_ARG_DECL  Data_Obj *,index_t),
			Data_Obj *(*cfunc)(QSP_ARG_DECL  Data_Obj *,index_t))
{
	obj_get_func=ofunc;
	exist_func=efunc;
	sub_func=sfunc;
	csub_func=cfunc;
}
#endif // NOT_USED

