%{

#include "quip_config.h"
#include <errno.h>
#include <math.h>
#include <string.h>

#include "quip_prot.h"
#include "warn.h"
#include "shape_bits.h"
#include "function.h"
//#include "strbuf.h"
#include "query_stack.h"

#ifdef HAVE_LIMITS_H
#include <limits.h>
#endif // HAVE_LIMITS_H

double rn_number(double);

#include "nexpr.h"
#include "func_helper.h"


#define THIS_SPD	QS_SCALAR_PARSER_DATA_AT_IDX(THIS_QSP,QS_SCALAR_PARSER_CALL_DEPTH(THIS_QSP))

#define YYSTRSTK 	SPD_YYSTRSTK(THIS_SPD)
#define EDEPTH 		SPD_EDEPTH(THIS_SPD)
#define YY_ORIGINAL	SPD_ORIGINAL_STRING(THIS_SPD)
#define WHICH_EXPR_STR	SPD_WHICH_STR(THIS_SPD)
// Modern versions of bison allow the creation of a reentrant parser,
// so this hack (and the resulting bugs!) are no longer necessary!
#define EXPR_STRING	SPD_EXPR_STRING(THIS_SPD)
#define FINAL_EXPR_NODE_P	SPD_FINAL_EXPR_NODE_P(THIS_SPD)
#define STRING_SCALAR	SPD_STRING_SCALAR(THIS_SPD)
#define ESTRINGS_INITED	SPD_ESTRINGS_INITED(THIS_SPD)

#define ADVANCE_EXPR_STR				\
							\
	WHICH_EXPR_STR++;				\
	WHICH_EXPR_STR %= MAX_E_STRINGS;

/* We used to call yyerror for lexical scan errors, but that often
 * caused yyerror to be called twice, the first time for the lexer
 * error, the second time with a parsing error...
 */

#define llerror(msg) _llerror(QSP_ARG  msg)

static void _llerror(QSP_ARG_DECL  const char *msg)
{
	char tmp_str[LLEN];	/* don't use error_string! */

	sprintf(tmp_str,"pexpr lexical scan error:  %s",msg);
	warn(tmp_str);
}


#define peek_parser_input() _peek_parser_input(SINGLE_QSP_ARG)

static inline int _peek_parser_input(SINGLE_QSP_ARG_DECL)
{
	while( *YYSTRSTK[EDEPTH] == 0 ){	// end of this line
		if( EDEPTH == 0 ) return -1;
		EDEPTH--;
	}
//fprintf(stderr,"peek_parser_input:  will return first char of '%s' (depth = %d)\n",YYSTRSTK[EDEPTH],EDEPTH);
	return *YYSTRSTK[EDEPTH];
}

#define has_parser_input() _has_parser_input(SINGLE_QSP_ARG)

static inline int _has_parser_input(SINGLE_QSP_ARG_DECL)
{
	if( EDEPTH >= 0 ) return 1;
	return 0;
}

#define advance_parser_input() _advance_parser_input(SINGLE_QSP_ARG)

static inline void _advance_parser_input(SINGLE_QSP_ARG_DECL)
{
	YYSTRSTK[EDEPTH]++;
}

#define push_parser_input(s) _push_parser_input(QSP_ARG  s)

static inline void _push_parser_input(QSP_ARG_DECL  const char *s)
{
	if( (EDEPTH+1) >= MAXEDEPTH ){
		llerror("expression depth too large");
		return;
	}
	EDEPTH++;
	YYSTRSTK[EDEPTH]=s;
}

static Item * default_eval_szbl( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{ return NULL; }

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


// This has to be a global, because the nodes are allocated in a context
// without a qsp...
static List *free_enp_lp=NULL;

// BUG - probably 4 is not enough now that these are being used
// to store strings for the whole tree...

static Scalar_Expr_Node *alloc_expr_node(void);

static Typed_Scalar ts_dbl_zero={
	{ 0.0 }, PREC_DP, TS_STATIC };

static Typed_Scalar ts_dbl_one={
	{ 1.0 }, PREC_DP, TS_STATIC };

static Typed_Scalar ts_dbl_minus_one={
	{ -1.0 }, PREC_DP, TS_STATIC };

/* what yylval can be */

typedef union {
	int			fundex;		/* function index */
	Quip_Function *		func_p;
	Scalar_Expr_Node *	enp;
	Typed_Scalar *		tsp;
} YYSTYPE;

#define YYSTYPE_IS_DECLARED		/* needed on 2.6 machine? */



#define YY_(msg)	msg

// With parser data stored in the query stack, we are thread-safe!

#ifdef FOOBAR
#ifdef THREAD_SAFE_QUERY

/* For yyerror */


#ifdef HAVE_PTHREADS
// We have one mutex that a thread has to hold to manipulate the list,
// and a shared flag to show whether or not they are locked...

static pthread_mutex_t	enode_mutex=PTHREAD_MUTEX_INITIALIZER;
static int enode_flags=0;

/* We don't bother with the mutex if the number of threads is less
 * than 1, but this could create a problem if we create a thread?
 * probably not...
 *
 * This code doesn't seem to be used now (commented out around rls_tree
 * is the only appearance), but it has not been tested with multiple threads,
 * and could have problems???
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
		enode_flags |= LIST_LOCKED_FLAG_BITS;			\
	}

#define UNLOCK_ENODES						\
								\
	if( enode_flags & LIST_LOCKED_FLAG_BITS )		\
	{							\
		int status;					\
								\
		enode_flags &= ~LIST_LOCKED_FLAG_BITS;		\
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

#endif /* ! THREAD_SAFE_QUERY */
#endif // FOOBAR

// For the time being a single signature, regardless of THREAD_SAFE_QUERY
static int yylex(YYSTYPE *yylvp, Query_Stack *qsp);

int yyerror(Query_Stack *, char *);

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

#define get_expr_stringbuf(index,min_len) _get_expr_stringbuf(QSP_ARG   index,min_len)

static char *_get_expr_stringbuf( QSP_ARG_DECL   int index, long min_len )
{
	String_Buf *sbp;

	if( EXPR_STRING[index] == NULL ){
		sbp = new_stringbuf();
		EXPR_STRING[index]=sbp;
	} else {
		sbp = EXPR_STRING[index];
	}
	if( sb_size(EXPR_STRING[index]) < min_len )
		enlarge_buffer(sbp,min_len);
	return(sb_buffer(sbp));
}

%}

%pure-parser	// make the parser rentrant (thread-safe)

// I've installed bison 3.0 on all mac systems using fink,
// but xcode still wants to use its own old version!?
// I wish there was a conditional processing (ifde

// this form works with bison 2.X (xcode)
// issues a deprecated warning in 3.0, but hopefully still works?
%name-prefix="quip_"

// this form requires bison 3.0 or later
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
%token <func_p> DOBJV_STR_ARG_FUNC
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
%type <enp> scalar_obj
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
			FINAL_EXPR_NODE_P = $1 ;
			}
		| e_string {
			FINAL_EXPR_NODE_P = $1;
			}
		/*
		| strv_func {
			final_expr_node_p = $1;
			}
			*/
		;

			
strv_func	: STRV_FUNC '(' e_string /* data_object*/  ')' {
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

// We would really like a data object to be a quoted string only...
// Because when we say ncols(xxx) we want the number of columns of an object
// named xxx, not the length of the string "xxx"...

data_object	:
		  e_string {
			$$=NODE1(N_OBJNAME,$1);
			}
 
		 /* The data_object-valued function with a string arg:
		  * string_obj("a string")
		  * This construct allows us to index strings and test char values.
		  */

		 | DOBJV_STR_ARG_FUNC '(' e_string ')' {
			$$=NODE1(N_DOBJV_STR_ARG_FUNC,$3);
			$$->sen_func_p=$1;
		  	}
		| data_object '[' expression ']' {
			$$=NODE2(N_SUBSCRIPT,$1,$3); }
		| data_object '{' expression '}' {
			$$=NODE2(N_CSUBSCRIPT,$1,$3); }
		;

scalar_obj	: data_object { $$=NODE1(N_SCALAR_OBJ,$1); }

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
		| CHAR_FUNC '(' scalar_obj ')' {
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
		| POSN_FUNC '(' e_string ')' {
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
		/*
		| data_object
			{ $$=NODE1(N_SCALAR_OBJ,$1); }
			*/

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

static Scalar_Expr_Node *alloc_expr_node(void)
{
	Scalar_Expr_Node *enp;
	int i;

	if( free_enp_lp != NULL && QLIST_HEAD(free_enp_lp) != NULL ){
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

// This gets called from outside?
// Where does the node come from???

const char *_eval_scalexp_string(QSP_ARG_DECL  Scalar_Expr_Node *enp)
{
	Typed_Scalar *tsp;
	const char *s, *s2;

	switch(enp->sen_code){
		case N_OBJNAME:
		case N_QUOT_STR:
			return eval_scalexp_string(enp->sen_child[0]);
			break;

		case N_STRVFUNC:
#ifdef FOOBAR
#ifdef BUILD_FOR_OBJC
			// BUG BUG BUG
			if( check_ios_strv_func(&s,enp->sen_func_p,
					enp->sen_child[0]) ){
				// BUG?  does the function get called in check_ios_sizable_func???
				return s;
			}
#endif /* BUILD_FOR_OBJC */
#endif // FOOBAR
			// why sizable?  this is supposed to be a string arg...
			// This makes sense only for the "precision" function -
			// but what about touuper etc?
			//szp = EVAL_SZBL_EXPR_FUNC(enp->sen_child[0]);
			s = eval_scalexp_string(enp->sen_child[0]);
			s = (*enp->sen_func_p->fn_u.strv_func)( QSP_ARG  s );
			return s;
			break;

		case N_STRV2FUNC:
#ifdef FOOBAR
#ifdef BUILD_FOR_OBJC
			// BUG BUG BUG ? (why?)
			if( check_ios_strv2_func(&s,enp->sen_func_p,
					enp->sen_child[0],enp->sen_child[1]) ){
				// BUG?  does the function get called in check_ios_sizable_func???
				return s;
			}
#endif /* BUILD_FOR_OBJC */
#endif // FOOBAR
			// why sizable???
			/*
			szp = EVAL_SZBL_EXPR_FUNC(enp->sen_child[0]);
			s = (*enp->sen_func_p->fn_u.strv_func)( QSP_ARG  szp );
			return s;
			*/
			s = eval_scalexp_string(enp->sen_child[0]);
			s2 = eval_scalexp_string(enp->sen_child[1]);
			s = (*enp->sen_func_p->fn_u.strv2_func)( QSP_ARG  s, s2 );
			return s;
			break;

		case N_LITSTR:
			tsp = enp->sen_tsp;
			assert( tsp->ts_prec_code == PREC_STR );

			return (char *) tsp->ts_value.u_vp;
			break;
		case N_STRING:
			return eval_scalexp_string(enp->sen_child[0]);
			break;
		default:
			assert( 0 );
			break;
	}
	return "foobar";
}


static Item* eval_tsbl_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Item *ip=NULL;
	const char *s;

	switch(enp->sen_code){
		case N_TSABLE:
			s = eval_scalexp_string(enp->sen_child[0]);
			ip = find_tsable( QSP_ARG  s );
			if( ip == NULL ){
				sprintf(ERROR_STRING,
					"No time-stampable object \"%s\"!?",s);
				warn(ERROR_STRING);
				return NULL;
			}
			break;
		default:
			assert( AERROR("unexpected case in eval_tsbl_expr") );
			break;
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
			s = eval_scalexp_string(enp);
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tstring\t%s",
				(uintptr_t)enp, s);
			advise(ERROR_STRING);
			break;

#ifdef FOOBAR
		case N_SIZABLE:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tsizable\t%s",
				(uintptr_t)enp, enp->sen_string);
			advise(ERROR_STRING);
			break;
#endif /* FOOBAR */

		case N_TSABLE:
			s = eval_scalexp_string(enp->sen_child[0]);
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\ttsable\t%s",
				(uintptr_t)enp, s);
			advise(ERROR_STRING);
			break;

		case N_STRVFUNC:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tstrvfunc\t%s",
				(uintptr_t)enp, FUNC_NAME( enp->sen_func_p ) );
			break;

		case N_STRV2FUNC:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tstrv2func\t%s",
				(uintptr_t)enp, FUNC_NAME( enp->sen_func_p ) );
			break;

		case N_SIZFUNC:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tsizefunc\t%s",
				(uintptr_t)enp, FUNC_NAME( enp->sen_func_p ) );
			advise(ERROR_STRING);
			break;

		case N_TSFUNC:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tts_func\t%s",
				(uintptr_t)enp, FUNC_NAME( enp->sen_func_p ) );
			advise(ERROR_STRING);
			break;

		case N_ILACEFUNC:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tinterlace_func\t%s",
				(uintptr_t)enp, FUNC_NAME( enp->sen_func_p ) );
			advise(ERROR_STRING);
			break;

		case N_POSNFUNC:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tposn_func\t%s",
				(uintptr_t)enp, FUNC_NAME( enp->sen_func_p ) );
			advise(ERROR_STRING);
			break;

		case N_MATH0FUNC:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tmath0_func\t%s",
				(uintptr_t)enp, FUNC_NAME( enp->sen_func_p ) );
			advise(ERROR_STRING);
			break;

		case N_MATH2FUNC:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tmath2_func\t%s",
				(uintptr_t)enp, FUNC_NAME( enp->sen_func_p ) );
			advise(ERROR_STRING);
			break;

		case N_MISCFUNC:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tmisc_func\t%s",
				(uintptr_t)enp, FUNC_NAME( enp->sen_func_p ) );
			advise(ERROR_STRING);
			break;

		case N_STR2FUNC:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tstr2_func\t%s",
				(uintptr_t)enp, FUNC_NAME( enp->sen_func_p ) );
			advise(ERROR_STRING);
			break;

		case N_STR3FUNC:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tstr3_func\t%s",
				(uintptr_t)enp, FUNC_NAME( enp->sen_func_p ) );
			advise(ERROR_STRING);
			break;

		case N_DATAFUNC:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tdatafunc\t%s",
				(uintptr_t)enp, FUNC_NAME( enp->sen_func_p ) );
			advise(ERROR_STRING);
			break;

		case N_OBJNAME:
			s = eval_scalexp_string(enp->sen_child[0]);
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tobjname\t%s",
				(uintptr_t)enp, s);
			advise(ERROR_STRING);
			break;

		case N_SCALAR_OBJ:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tscalar_obj\t0x%"PRIxPTR,
				(uintptr_t)enp, (uintptr_t)enp->sen_child[0]);
			advise(ERROR_STRING);
			break;

		case N_SUBSCRIPT:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tsubscript\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp, (uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;
		case N_CSUBSCRIPT:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tcsubscript\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp, (uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_MATH1FUNC:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tmath1func\t%s",
				(uintptr_t)enp, FUNC_NAME(enp->sen_func_p) );
			advise(ERROR_STRING);
			break;

		case N_PLUS:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tplus\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp,(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_MINUS:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tminus\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp,(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_TIMES:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\ttimes\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp,(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_DIVIDE:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tdivide\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp,(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_MODULO:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tmodulo\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp,(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_BITAND:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tbitand\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp,(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_BITOR:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tbitor\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp,(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_BITXOR:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tbitxor\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp,(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_SHL:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tshl\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp,(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_SHR:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tshr\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp,(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_LOGOR:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tlog_or\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp,(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_LOGAND:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tlog_and\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp,(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_LOGXOR:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tlog_xor\t0x%"PRIxPTR"\t0x%"PRIxPTR,
				(uintptr_t)enp,(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;

		case N_LITNUM:
			string_for_typed_scalar(MSG_STR,LLEN,enp->sen_tsp);
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tlit_num\t%s",
				(uintptr_t)enp,MSG_STR);
			advise(ERROR_STRING);
			break;
		case N_LE:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\t<= (LE)\t0x%"PRIxPTR", 0x%"PRIxPTR,(uintptr_t)enp,
				(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;
		case N_GE:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\t>= (GE)\t0x%"PRIxPTR", 0x%"PRIxPTR,(uintptr_t)enp,
				(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;
		case N_NE:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\t!= (NE)\t0x%"PRIxPTR", 0x%"PRIxPTR,(uintptr_t)enp,
				(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;
		case N_LT:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\t< (LT)\t0x%"PRIxPTR", 0x%"PRIxPTR,(uintptr_t)enp,
				(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;
		case N_GT:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\t> (GT)\t0x%"PRIxPTR", 0x%"PRIxPTR,(uintptr_t)enp,
				(uintptr_t)enp->sen_child[0],(uintptr_t)enp->sen_child[1]);
			advise(ERROR_STRING);
			break;
		case N_NOT:
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\t! (NOT)\t0x%"PRIxPTR,
				(uintptr_t)enp,
				(uintptr_t)enp->sen_child[0]);
			advise(ERROR_STRING);
			break;
		case N_STRFUNC:
			s = eval_scalexp_string(enp->sen_child[0]);
			sprintf(ERROR_STRING,"0x%"PRIxPTR"\tSTRFUNC %s\t\"%s\"",
				(uintptr_t)enp,
				FUNC_NAME(enp->sen_func_p),
				s);
			advise(ERROR_STRING);
			break;

// comment out the default case for the compiler to show unhandled cases...
		default:
			sprintf(ERROR_STRING,
		"%s - %s:  unhandled node code %d",
				WHENCE2(dump_enode),enp->sen_code);
			advise(ERROR_STRING);
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
			s = eval_scalexp_string(enp->sen_child[0]);
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
			return NULL;
			break;
		case N_OBJNAME:	// eval_dobj_expr
			s = eval_scalexp_string(enp->sen_child[0]);
			dp = (*exist_func)( QSP_ARG  s );
			if( dp == NULL ){	// could be an identifier?
				Identifier *idp;
				idp = id_of(s);
				if( idp == NULL ){
					sprintf(ERROR_STRING,
	"No object or identifier %s!?",s);
					warn(ERROR_STRING);
					return NULL;
				}
				if( ID_TYPE(idp) == ID_SCALAR ){
					// create a temp scalar object
					dp = mk_scalar("tmp_scal",ID_PREC_PTR(idp));

					// BUG when to release this object???
				} else {
					sprintf(ERROR_STRING,
	"Identifier %s is not a scalar!?",s);
					warn(ERROR_STRING);
					return NULL;
				}
			}
			break;
		case N_SUBSCRIPT:
			dp2=eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
			tsp = eval_expr(enp->sen_child[1]);
			index=index_for_scalar( tsp );
			RELEASE_SCALAR(tsp)
			dp=(*sub_func)( QSP_ARG  dp2, index );
			break;
		case N_CSUBSCRIPT:
			dp2=eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
			tsp=eval_expr(enp->sen_child[1]);
			index=index_for_scalar(tsp);
			RELEASE_SCALAR(tsp)
			dp=(*csub_func)( QSP_ARG  dp2, index );
			break;

		default:
			assert(0);
			break;
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
			s = eval_scalexp_string(enp);
			szp = check_sizable( QSP_ARG  s );
			if( szp == NULL ){
				Data_Obj *dp;
				dp = obj_for_string(s);
				szp = (Item *)dp;
			}
			break;

		//case N_SIZABLE:
		case N_OBJNAME:
			// Not necessarily a data object!?
			s = eval_scalexp_string(enp);
			szp = find_sizable( QSP_ARG  s );
			if( szp == NULL ){
				sprintf(ERROR_STRING,
					"No sizable object \"%s\"!?",s);
				warn(ERROR_STRING);
				return NULL;
			}
			break;
		//case N_SUBSIZ:
		case N_SUBSCRIPT:
			szp2=EVAL_SZBL_EXPR(enp->sen_child[0]);
			if( szp2 == NULL )
				return NULL;
			index = index_for_scalar( eval_expr(enp->sen_child[1]) );
			szp = sub_sizable(QSP_ARG  szp2,index);
			break;
		//case N_CSUBSIZ:
		case N_CSUBSCRIPT:
			szp2=EVAL_SZBL_EXPR(enp->sen_child[0]);
			if( szp2 == NULL )
				return NULL;
			index = index_for_scalar( eval_expr(enp->sen_child[1]) );
			szp = csub_sizable(QSP_ARG  szp2,index);
			break;
		default:
			assert(0);
			break;
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
		case N_LITSTR:
		case N_OBJNAME:
			// Not necessarily a data object!?
			s = eval_scalexp_string(enp);
			szp = find_positionable( QSP_ARG  s );
			break;
#ifdef CAUTIOUS
		default:
			sprintf(ERROR_STRING,
		"unexpected case in eval_szbl_expr %d",enp->sen_code);
			warn(ERROR_STRING);
			//assert(0);
			break;
#endif /* CAUTIOUS */
	}
	if( szp == NULL ){
		sprintf(ERROR_STRING,
			"No positionable object \"%s\"!?",s);
		warn(ERROR_STRING);
		return NULL;
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
			s = eval_scalexp_string(enp);
			szp = find_interlaceable( QSP_ARG  s );
			if( szp == NULL ){
				sprintf(ERROR_STRING,
					"No interlaceable object \"%s\"!?",s);
				warn(ERROR_STRING);
				return NULL;
			}
			break;
		// are data objects interlaceable???
#ifdef FOOBAR
		//case N_SUBSIZ:
		case N_SUBSCRIPT:
			szp2=EVAL_SZBL_EXPR(enp->sen_child[0]);
			if( szp2 == NULL )
				return NULL;
			index = index_for_scalar( eval_expr(enp->sen_child[1]) );
			szp = sub_sizable(QSP_ARG  szp2,index);
			break;
		//case N_CSUBSIZ:
		case N_CSUBSCRIPT:
			szp2=EVAL_SZBL_EXPR(enp->sen_child[0]);
			if( szp2 == NULL )
				return NULL;
			index = index_for_scalar( eval_expr(enp->sen_child[1]) );
			szp = csub_sizable(QSP_ARG  szp2,index);
			break;
#endif // FOOBAR

		default:
			assert(0);
			break;
	}
	return(szp);
}

static void divzer_error(SINGLE_QSP_ARG_DECL)
{
	sprintf(ERROR_STRING,"Error parsing \"%s\"",YY_ORIGINAL);
	advise(ERROR_STRING);
	warn("eval_expr:  divide by 0!?");
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
		tsp2=eval_expr(enp->sen_child[0]);	\
		tsp3=eval_expr(enp->sen_child[1]);	\
		dval2=double_for_scalar(tsp2);		\
		dval3=double_for_scalar(tsp3);		\
		RELEASE_BOTH

#define GET_ONE_DOUBLE					\
		tsp2=eval_expr(enp->sen_child[0]);	\
		dval2=double_for_scalar(tsp2);		\
		RELEASE_FIRST

#define GET_ONE_LONG					\
		tsp2=eval_expr(enp->sen_child[0]);	\
		ival2=long_for_scalar(tsp2);		\
		RELEASE_FIRST

#define GET_TWO_ULONGS					\
		tsp2=eval_expr(enp->sen_child[0]);	\
		tsp3=eval_expr(enp->sen_child[1]);	\
		uval2=llong_for_scalar(tsp2);		\
		uval3=llong_for_scalar(tsp3);		\
		RELEASE_BOTH

#define GET_TWO_LONGS					\
		tsp2=eval_expr(enp->sen_child[0]);	\
		tsp3=eval_expr(enp->sen_child[1]);	\
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

Typed_Scalar * _eval_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
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
	static Quip_Function *val_func_p=NULL;

#ifdef QUIP_DEBUG
if( debug & expr_debug ){
sprintf(ERROR_STRING,"eval_expr:  code = %d",enp->sen_code);
advise(ERROR_STRING);
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
#ifdef FOOBAR
		/* We have problems mixing IOS objects and C structs... */
#ifdef BUILD_FOR_OBJC
		if( check_ios_strv_func(&s,enp->sen_func_p,
				enp->sen_child[0]) ){
			tsp = scalar_for_string(s);
			return tsp;
		}
#endif /* BUILD_FOR_OBJC */
#endif // FOOBAR
		// BUG - if this is to support the precision function, then
		// the sizable lookup should be done within the function, not here!
		/*
		szp = EVAL_SZBL_EXPR_FUNC(enp->sen_child[0]);
		s = (*enp->sen_func_p->fn_u.strv_func)( QSP_ARG  szp );
		*/
		s = eval_scalexp_string(enp->sen_child[0]);
		s = (*enp->sen_func_p->fn_u.strv_func)( QSP_ARG s );
//fprintf(stderr,"eval_expr:  strv_func returned string at 0x%"PRIxPTR"\n",(uintptr_t)s);
		tsp = scalar_for_string(s);
		break;
	case N_STRV2FUNC:		// eval_expr
#ifdef FOOBAR
#ifdef BUILD_FOR_OBJC
		if( check_ios_strv2_func(&s,enp->sen_func_p,
				enp->sen_child[0],enp->sen_child[0]) ){
			tsp = scalar_for_string(s);
			return tsp;
		}
#endif /* BUILD_FOR_OBJC */
#endif // FOOBAR
		s = eval_scalexp_string(enp->sen_child[0]);
		s2 = eval_scalexp_string(enp->sen_child[1]);
		s = (*enp->sen_func_p->fn_u.strv2_func)( QSP_ARG s, s2 );
		tsp = scalar_for_string(s);
		break;

	case N_TSFUNC:		// eval_expr
		szp = eval_tsbl_expr(QSP_ARG  enp->sen_child[0]);
		tsp2=eval_expr(enp->sen_child[1]);
		frm = (dimension_t)double_for_scalar(tsp2);
		RELEASE_FIRST
		dval = (*enp->sen_func_p->fn_u.ts_func)( QSP_ARG  szp, frm );
		tsp = scalar_for_double(dval);
		break;
	case N_STRFUNC:		// eval_expr
		s = eval_scalexp_string(enp->sen_child[0]);
		dval = evalStr1Function(QSP_ARG  enp->sen_func_p,s);
		tsp = scalar_for_double(dval);
		break;

	case N_STR2FUNC:		// eval_expr
		s = eval_scalexp_string(enp->sen_child[0]);
		s2 = eval_scalexp_string(enp->sen_child[1]);
		dval = evalStr2Function(enp->sen_func_p,s,s2);
		tsp = scalar_for_double(dval);
		break;

	case N_STR3FUNC:		// eval_expr
		s = eval_scalexp_string(enp->sen_child[0]);
		s2 = eval_scalexp_string(enp->sen_child[1]);
		dval = evalStr3Function( enp->sen_func_p, s, s2,
			(int) double_for_scalar( eval_expr( enp->sen_child[2]) ) );
		tsp = scalar_for_double(dval);
		break;

//#ifdef FOOBAR
//	case N_STRVFUNC:		// eval_expr
//		// string valued functions, tolower toupper etc
//		s = eval_scalexp_string(enp->sen_child[0]);
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

		if( val_func_p == NULL )
			val_func_p = function_of("value");
		assert( val_func_p != NULL );

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
		warn(ERROR_STRING);
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
		tsp2=eval_expr(enp->sen_child[0]);
		SET_RESULT( has_zero_value(tsp2) )
		RELEASE_FIRST
		break;
	case N_BITCOMP:		// eval_expr
		tsp2=eval_expr(enp->sen_child[0]);
		uval2=llong_for_scalar(tsp2);
		RELEASE_FIRST
		uval = (~uval2);
		tsp = SCALAR_FOR_INT_TYPE(uval);
		break;
	case N_UMINUS:		// eval_expr
		tsp2=eval_expr(enp->sen_child[0]);
		dval2=double_for_scalar(tsp2);
		RELEASE_FIRST
		dval=(-dval2);
		tsp = scalar_for_double(dval);
		break;

	case N_EQUIV:		// eval_expr
		tsp2=eval_expr(enp->sen_child[0]);
		tsp3=eval_expr(enp->sen_child[1]);
		if( scalars_are_equal(tsp2,tsp3) )
			tsp = &ts_dbl_one;
		else
			tsp = &ts_dbl_zero;
		//if( dval2 == dval3 ) dval=1.0;
		//else dval=0.0;
		RELEASE_BOTH
		break;
	case N_LOGOR:		// eval_expr
		tsp2=eval_expr(enp->sen_child[0]);
		if( ! has_zero_value(tsp2) ){
			tsp=&ts_dbl_one;
			release_branch(QSP_ARG  enp->sen_child[1]);
		} else {
			tsp3=eval_expr(enp->sen_child[1]);
			if( ! has_zero_value(tsp3) ) tsp=&ts_dbl_one;
			else tsp=&ts_dbl_zero;
			RELEASE_SECOND
		}
		RELEASE_FIRST
		break;
	case N_LOGAND:		// eval_expr
		tsp2=eval_expr(enp->sen_child[0]);
		if( has_zero_value(tsp2) ){
			tsp=&ts_dbl_zero;
			release_branch(QSP_ARG  enp->sen_child[1]);
		} else {
			tsp3=eval_expr(enp->sen_child[1]);
			if( has_zero_value(tsp3) ) tsp=&ts_dbl_zero;
			else tsp=&ts_dbl_one;
			RELEASE_SECOND
		}
		RELEASE_FIRST
		break;
	case N_LOGXOR:		// eval_expr
		tsp2=eval_expr(enp->sen_child[0]);
		tsp3=eval_expr(enp->sen_child[1]);
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
		tsp2=eval_expr(enp->sen_child[0]);
		tsp3=eval_expr(enp->sen_child[1]);
		uval2 = llong_for_scalar(tsp2);
		ival = llong_for_scalar(tsp3);
		RELEASE_BOTH
		/* this version clears the sign bit */
		/* dval = ((((int)dval2) >> 1) & 077777) >> ((ival) - 1); */
		uval = (uval2)>>ival;
		tsp=SCALAR_FOR_INT_TYPE(uval);
		break;
	case N_CONDITIONAL:		// eval_expr
		tsp2 = eval_expr(enp->sen_child[0]);
		if( ! has_zero_value(tsp2) ) {
			tsp = eval_expr(enp->sen_child[1]);
			release_branch( QSP_ARG   enp->sen_child[2]);
		} else {
			tsp = eval_expr(enp->sen_child[2]);
			release_branch(QSP_ARG  enp->sen_child[1]);
		}
		RELEASE_FIRST
		break;

//#ifdef FOOBAR
//	case N_SLCT_CHAR:		// eval_expr
//advise("case N_SLCT_CHAR");
//		ival = eval_expr(enp->sen_child[0]);
//		if( ival < 0 || ival >= strlen(enp->sen_string) )
//			dval = -1;
//		else
//			dval = enp->sen_string[ival];
//		break;
//#endif /* FOOBAR */

	default:		// eval_expr
		assert(0);
		break;

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

#define BUF_CHAR(c)					\
							\
	if( i > buflen ){				\
		warn("extract_number_string:  string buffer too small!?");	\
		return(-1);				\
	}						\
	buf[i++] = (c);


#define extract_number_string(buf,buflen,srcp) _extract_number_string(QSP_ARG  buf,buflen,srcp)

static int _extract_number_string(QSP_ARG_DECL  char *buf, int buflen, const char **srcp)
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
				warn("extract_number_string:  mal-formed hex string");
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
			warn("extract_number_string:  malformed exponent!?");
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

Typed_Scalar *_parse_number(QSP_ARG_DECL  const char **strptr)
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
		sprintf(ERROR_STRING,"parse_number:  bad number string \"%s\"",
			ptr);
		warn(ERROR_STRING);
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
					sprintf(ERROR_STRING,"parse_number %s:  long long conversion error!?  (errno=%d)",buf,errno);
					warn(ERROR_STRING);
					tell_sys_error("strtoull");
					sprintf(ERROR_STRING,"value returned:  0x%llx",ll1);
					advise(ERROR_STRING);
					sprintf(ERROR_STRING,"unsigned long long max:  0x%llx",ULLONG_MAX);
					advise(ERROR_STRING);
				}
			}
			if( errno != 0 ){
				sprintf(ERROR_STRING,"parse_number %s:  long long conversion error!?  (errno=%d)",buf,errno);
				warn(ERROR_STRING);
				tell_sys_error("strtoll");
				sprintf(ERROR_STRING,"value returned:  0x%llx",ll1);
				advise(ERROR_STRING);
				sprintf(ERROR_STRING,"long long range:  0x%llx - 0x%llx",LLONG_MIN,LLONG_MAX);
				advise(ERROR_STRING);
			}
			return( SCALAR_FOR_INT_TYPE(ll1) );
			//return ll1;
#else // ! HAVE_STRTOLL
			sprintf(ERROR_STRING,"long conversion error!?  (errno=%d)",errno);
			warn(ERROR_STRING);
			tell_sys_error("strtol");
#endif // ! HAVE_STRTOLL

		} else if( errno != 0 ){
			sprintf(ERROR_STRING,"long conversion error!?  (errno=%d)",errno);
			warn(ERROR_STRING);
			tell_sys_error("strtol");
		}
		return( scalar_for_long(l) );
	}
	/*   else if( status & 2 ){ */	/* decimal pt seen */

		errno = 0;
//sprintf(ERROR_STRING,"converting string \"%s\"",buf);
//advise(ERROR_STRING);
		d = strtod(buf,&endptr);
		if( errno == ERANGE ){
			// This message is printing, but the returned
			// value is not +-HUGE_VAL, in contradiction to
			// the documentation.
			if( d == 0.0 ){
sprintf(ERROR_STRING,"strtod:  possible underflow buf=\"%s\", d = %g",buf,d);
advise(ERROR_STRING);
			} else if( d == HUGE_VAL || d == -HUGE_VAL ){
sprintf(ERROR_STRING,"strtod:  possible overflow buf=\"%s\", d = %g  HUGE_VAL = %g",buf,d,HUGE_VAL);
advise(ERROR_STRING);
			} else {
				if( verbose ){
sprintf(ERROR_STRING,"strtod:  possible overflow (inconsistent) buf=\"%s\", d = %g  HUGE_VAL = %g",buf,d,HUGE_VAL);
advise(ERROR_STRING);
				}
			}
		} else if( errno != 0 ){
			sprintf(ERROR_STRING,"double conversion error!?  (errno=%d)",errno);
			warn(ERROR_STRING);
			tell_sys_error("strtod");
		}

//sprintf(ERROR_STRING,"flt conversion returning %lg",d);
//advise(ERROR_STRING);
//		return( scalar_for_double(d) );

	{
		Typed_Scalar *tsp;
		tsp = scalar_for_double(d) ;
		return( tsp );
	}
	/* } */

} /* end parse_number() */

//static double yynumber(SINGLE_QSP_ARG_DECL)
#define yynumber() _yynumber(SINGLE_QSP_ARG)

static Typed_Scalar * _yynumber(SINGLE_QSP_ARG_DECL)
{
	Typed_Scalar *tsp;

	tsp = parse_number((const char **)&YYSTRSTK[EDEPTH]);
	return tsp;
}

#define TMPBUF_LEN	128

#define varval() _varval(SINGLE_QSP_ARG)

static const char * _varval(SINGLE_QSP_ARG_DECL)
{
	char tmpbuf[TMPBUF_LEN];
	const char *s;
	const char *var_valstr;
	int c;

	/* indirect variable reference? */

	if( peek_parser_input() == '$' ){
		advance_parser_input(); // skip dollar sign
		s = varval() ;
	} else {
		/* read in the variable name */
		char *sp;
		int n_stored=0;
		sp=tmpbuf;
		c=peek_parser_input();
		while( isalpha(c) || c == '_' || isdigit(c) ){
			*sp++ = (char) c;
			if( ++n_stored >= TMPBUF_LEN ){
				warn("varval:  buffer overrun!?");
				sp--;
			}

			advance_parser_input();
			c=peek_parser_input();
		}
		*sp=0;
		s=tmpbuf;
	}

	var_valstr = var_value(s);

	if( var_valstr == NULL )
		return("0");
	else {
		return( var_valstr );
	}
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
		case DOBJV_STR_ARG_FUNCTYP:	return(DOBJV_STR_ARG_FUNC);	break;
		case TS_FUNCTYP:	return(TS_FUNC);	break;
		case ILACE_FUNCTYP:	return(IL_FUNC);	break;
		case POSN_FUNCTYP:	return(POSN_FUNC);	break;
		default:
			assert( AERROR("token_for_func_type:  bad type!?") );
			break;
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

	while( has_parser_input() ){
		/* skip spaces */

		while( (c=peek_parser_input())>=0 && isspace(c) )
			advance_parser_input();

		if( peek_parser_input() < 0 )
			return 0;

		c=peek_parser_input();
		if( isdigit(c) || c=='.' ) {
			yylvp->tsp=yynumber();
			return(NUMBER);
		} else if( c == '$' ) {
			advance_parser_input();
			// we push the value of the variable onto the input stack
			push_parser_input(varval());
		} else if( IS_LEGAL_FIRST_CHAR(c) ){	/* get a name */
			int n=1;
			s=get_expr_stringbuf(WHICH_EXPR_STR,strlen(YYSTRSTK[EDEPTH]));
			*s++ = peek_parser_input();
			advance_parser_input();
			while( IS_LEGAL_NAME_CHAR((c=peek_parser_input())) ){
				*s++ = c;
				advance_parser_input();
				n++;
				assert( n < sb_size(EXPR_STRING[WHICH_EXPR_STR]) );
			}
			*s=0;

			yylvp->func_p = function_of(sb_buffer(EXPR_STRING[WHICH_EXPR_STR]));
			if( yylvp->func_p != NULL ){
				int t;
				t = token_for_func_type(FUNC_TYPE(yylvp->func_p));
//fprintf(stderr,"Found function at 0x%"PRIxPTR", token = %d\n",(long)yylvp->func_p,t);
				return t;
			}

			// use an array of tsp's here instead???
			//yylvp->e_string=EXPR_STRING[WHICH_EXPR_STR]->sb_buf;
			STRING_SCALAR[WHICH_EXPR_STR].ts_value.u_vp=
				sb_buffer(EXPR_STRING[WHICH_EXPR_STR]);
			yylvp->tsp=(&STRING_SCALAR[WHICH_EXPR_STR]);
			ADVANCE_EXPR_STR
			return(E_STRING);	/* unquoted string */

		} else if( ispunct(c) ){
			advance_parser_input();
			yylvp->fundex=c;

			if( c=='>' ){
				if( peek_parser_input() == '>' ){
					advance_parser_input();
					return(SHR);
				} else if( peek_parser_input() == '=' ){
					advance_parser_input();
					return(GE);
				}
			} else if( c=='<' ){
				if( peek_parser_input() == '<' ){
					advance_parser_input();
					return(SHL);
				} else if( peek_parser_input() == '=' ){
					advance_parser_input();
					return(LE);
				}
			} else if( c == '=' ){
				if( peek_parser_input() == '=' ){
					advance_parser_input();
					return(EQUIV);
				}
			} else if( c == '|' ){
				if( peek_parser_input() == '|' ){
					advance_parser_input();
					return(LOGOR);
				}
			} else if( c == '&' ){
				if( peek_parser_input() == '&' ){
					advance_parser_input();
					return(LOGAND);
				}
			} else if( c == '^' ){
				if( peek_parser_input() == '^' ){
					advance_parser_input();
					return(LOGXOR);
				}
			} else if( c == '!' ){
				if( peek_parser_input() == '=' ){
					advance_parser_input();
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
				s=get_expr_stringbuf(WHICH_EXPR_STR,strlen(YYSTRSTK[EDEPTH]));
				/* copy string into a buffer */
				c = peek_parser_input();
				while( c && c != qchar ){
					if( c == '\\' ){
						advance_parser_input();
						c = peek_parser_input();
						if( c == 0 ){
							// syntax error
				llerror("unmatched quote");
						} else {
							*s++=(char)c;
							advance_parser_input();
							n++;
						}
					} else {
						*s++=(char)c;
						advance_parser_input();
						n++;
					}
					c = peek_parser_input();
					assert( n < sb_size(EXPR_STRING[WHICH_EXPR_STR]) );
				}
				*s=0;
				if( peek_parser_input() == qchar ){
					advance_parser_input();
					/* used to call var_expand here,
					 * but now this is done automatically.
					 */
				} else llerror("unmatched quote");

				yylvp->tsp=(&STRING_SCALAR[WHICH_EXPR_STR]);
				STRING_SCALAR[WHICH_EXPR_STR].ts_value.u_vp
					= sb_buffer(EXPR_STRING[WHICH_EXPR_STR]);

				ADVANCE_EXPR_STR
				return(E_QSTRING);	/* quoted string */
			}
			return(c);	// punctuation char
		} else {
			llerror("yylex error");
			return(0);
		}
	}
	return 0;
}

/* rls_tree should only be called when locked */

#define rls_tree(enp) _rls_tree(QSP_ARG  enp)

static void _rls_tree(QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Node *np;

	if( enp->sen_child[0] != NULL )
		rls_tree(enp->sen_child[0]);
	if( enp->sen_child[1] != NULL )
		rls_tree(enp->sen_child[1]);
	/*
	if( enp->sen_string != NULL )
		rls_str(enp->sen_string);
	if( enp->sen_string2 != NULL )
		rls_str(enp->sen_string2);
		*/

	if( free_enp_lp == NULL ){
		free_enp_lp = new_list();
		assert( free_enp_lp != NULL );
	}
	np = mk_node(enp);
	addHead(free_enp_lp,np);
}

static void initialize_estrings(SINGLE_QSP_ARG_DECL)
{
	// BUG using these static structs is not thread-safe!

	int i;

	for(i=0;i<MAX_E_STRINGS;i++){
		EXPR_STRING[i]=NULL;
		STRING_SCALAR[i].ts_value.u_vp = NULL;
		STRING_SCALAR[i].ts_prec_code = PREC_STR;
	}
	ESTRINGS_INITED=1;
}

static void scalar_parser_cleanup(SINGLE_QSP_ARG_DECL)
{
	if( FINAL_EXPR_NODE_P != NULL )
		rls_tree(FINAL_EXPR_NODE_P);
	//UNLOCK_ENODES
	SET_QS_SCALAR_PARSER_CALL_DEPTH(THIS_QSP,QS_SCALAR_PARSER_CALL_DEPTH(THIS_QSP)-1);
	assert( QS_SCALAR_PARSER_CALL_DEPTH(THIS_QSP) >= (-1) );
}

/*
 * OLD:
 * Yacc doesn't allow recursive calls to the parser,
 * so we check for a recursive call here.  This arose when
 * we allowed indexed data objects to be valid size function
 * arguments, since normally pars_obj calls pexpr to evaluate
 * the index.  When this happens we assume it's a number
 * and hope for the best.
 * (This is a BUG with no obvious solution.)  wait - fixable!
 *
 * NEW:
 * Wait, there is a solution:  bison allows %pure_parser which
 * generates a reentrant parser, so we can call it recursively...
 *
 * We'd like for error messages to be printed with an input file
 * and line number.  In the multi-thread environment, that requires a qsp...
 * We have the qsp here,  for yyparse to pass it to yyerror we have
 * to use YY_PARSE_PARAM!
 *
 */

/* double */
Typed_Scalar *
_pexpr(QSP_ARG_DECL  const char *buf)	/** parse expression */
{
	int stat;
	Typed_Scalar *tsp;

//fprintf(stderr,"pexpr('%s') BEGIN, call depth = %d\n",buf,QS_SCALAR_PARSER_CALL_DEPTH(THIS_QSP));

#ifdef QUIP_DEBUG
	if( expr_debug <= 0 )
		expr_debug = add_debug_module("expressions");
#endif /* QUIP_DEBUG */

	// The parser won't parse things inside of quote strings.
	// To handle this, we have to make pexpr rentrant (like yyparse)
	// Therefore the per-qsp parser data has to be kept on a stack
	
	SET_QS_SCALAR_PARSER_CALL_DEPTH(THIS_QSP,1+QS_SCALAR_PARSER_CALL_DEPTH(THIS_QSP));
	assert(QS_SCALAR_PARSER_CALL_DEPTH(THIS_QSP)<MAX_SCALAR_PARSER_CALL_DEPTH);
	if( QS_CURR_SCALAR_PARSER_DATA(THIS_QSP) == NULL ){
		init_scalar_parser_data_at_idx(QS_SCALAR_PARSER_CALL_DEPTH(THIS_QSP));
	}

	if( ! ESTRINGS_INITED ) initialize_estrings(SINGLE_QSP_ARG);

	EDEPTH=0;
	YY_ORIGINAL=YYSTRSTK[EDEPTH]=buf;
	stat=yyparse(/*SINGLE_QSP_ARG*/ THIS_QSP );

	if( stat != 0 ){
		/* Need to somehow free allocated nodes... */
		if( verbose ){
			sprintf(ERROR_STRING,"yyparse returned status %d",stat);
			advise(ERROR_STRING);
		}
		//return(0.0);
		scalar_parser_cleanup(SINGLE_QSP_ARG);
		return(&ts_dbl_zero);
	}

#ifdef QUIP_DEBUG
if( debug & expr_debug ){
dump_etree(QSP_ARG  FINAL_EXPR_NODE_P);
}
#endif /* QUIP_DEBUG */

//fprintf(stderr,"pexpr:  evaluating expression tree, thread %d\n",QS_SERIAL);
	tsp = eval_expr(FINAL_EXPR_NODE_P);
//fprintf(stderr,"pexpr:  done with expression tree, thread %d\n",QS_SERIAL);

#ifdef QUIP_DEBUG
if( debug & expr_debug ){
if( tsp->ts_prec_code == PREC_DP ){
sprintf(ERROR_STRING,"pexpr:  s=\"%s\", dval = %g",buf,tsp->ts_value.u_d);
advise(ERROR_STRING);
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
			tsp2 = eval_expr(FINAL_EXPR_NODE_P);
			RELEASE_SCALAR(tsp)
			tsp = tsp2;
		}
	}
#endif /* SUN */

	//LOCK_ENODES

	scalar_parser_cleanup(SINGLE_QSP_ARG);

	return( tsp );
}

// OLD:  We can't add a qsp arg here, because this one is defined by yacc...
// With bison v3, parse-param goes here too!

int yyerror(Query_Stack *qsp, char *s)
{
	// BUG - this is wrong if we have multiple
	// interpreter threads
	if( IS_HALTING(THIS_QSP) )
		goto cleanup;
	
	sprintf(ERROR_STRING,"parsing \"%s\"",YY_ORIGINAL);
	advise(ERROR_STRING);

	if( has_parser_input() ){
		sprintf(ERROR_STRING,"\"%s\" left to parse",YYSTRSTK[0]);
		advise(ERROR_STRING);
	} else {
		advise("No buffered text left to parse");
	}

	// Print the warning after the informational messages, in case
	// this warning causes the program to exit.
	sprintf(ERROR_STRING,"YYERROR:  %s",s);
	warn(ERROR_STRING);

	/* final=(-1); */
	/* -1 is a bad value, because when the target is an
	 * unsigned in (dimension_t), the cast makes it a very
	 * large number...  causes all sorts of problems!
	 */
cleanup:
	FINAL_EXPR_NODE_P=NULL;
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
	sprintf(ERROR_STRING,"can't search for object \"%s\"; ",name);
	warn(ERROR_STRING);

	warn("data module not linked");
	return NULL;
}

static Data_Obj *_def_sub(QSP_ARG_DECL  Data_Obj *object,index_t index)
{
	warn("can't get subobject; data module not linked");
	return NULL;
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

