%{
#include "quip_config.h"

char VersionId_qutil_nexpr[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

double rn_number();

#include "query.h"
#include "nexpr.h"
#include "debug.h"
#include "function.h"

#define EVAL_EXPR( s )		eval_expr( QSP_ARG  s )
#define EVAL_SZBL_EXPR( s )	eval_szbl_expr( QSP_ARG  s )

#ifdef DEBUG
static debug_flag_t expr_debug=0;
#endif /* DEBUG */

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
//#define MAXEDEPTH	20

static List *free_enp_lp=NO_LIST;

#define MAX_E_STRINGS	4

char _strbuf[MAX_E_STRINGS][128];	/* temporary storage for names */

/* These have to be put into query stream... */
/*
static int which_str=0;
static int edepth=(-1);
static const char *yystrptr[MAXEDEPTH];
static Scalar_Expr_Node * final;
*/

static int n_function_classes=0;
#define MAX_FUNCTION_CLASSES	10
static Function_Class func_class[MAX_FUNCTION_CLASSES];

/* local prototypes */

static Item * eval_tsbl_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp );
static void divzer_error(SINGLE_QSP_ARG_DECL);
static void rls_tree(Scalar_Expr_Node *enp);
static void llerror(QSP_ARG_DECL  const char *msg);
#define LLERROR(s)	llerror(QSP_ARG  s)
static Data_Obj *eval_dobj_expr(QSP_ARG_DECL  Scalar_Expr_Node *);
static Item * eval_szbl_expr(QSP_ARG_DECL  Scalar_Expr_Node *enp);

static double yynumber(SINGLE_QSP_ARG_DECL);
static const char *varval(SINGLE_QSP_ARG_DECL);


/* what yylval can be */

typedef union {
	double fval;		/* actual value */
	int   fundex;		/* function index */
	char *e_string;
	Scalar_Expr_Node *enp;
} YYSTYPE;

#define YYSTYPE_IS_DECLARED		/* needed on 2.6 machine? */



#ifdef THREAD_SAFE_QUERY

#define YYPARSE_PARAM qsp	/* gets declared void * instead of Query_Stream * */
/* For yyerror */
#define YY_(msg)	QSP_ARG msg

static pthread_mutex_t	enode_mutex=PTHREAD_MUTEX_INITIALIZER;
static int enode_flags=0;

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

int yylex(YYSTYPE *yylvp, Query_Stream *qsp);
#define YYLEX_PARAM SINGLE_QSP_ARG

#define YY_QSP	((Query_Stream *)qsp)

#else /* ! THREAD_SAFE_QUERY */

#define YY_QSP	THIS_QSP

#define LOCK_ENODES
#define UNLOCK_ENODES
int yylex(YYSTYPE *yylvp);

#endif /* ! THREAD_SAFE_QUERY */

int yyerror(QSP_ARG_DECL  char *);
/* int yyparse(void); */

static Data_Obj *_def_obj(QSP_ARG_DECL  const char *);
static Data_Obj *_def_sub(QSP_ARG_DECL  Data_Obj *,index_t);

/*
void set_obj_funcs(
	Data_Obj *(*obj_func)(char *),
	Data_Obj *(*sub_func)(Data_Obj *,index_t),
	Data_Obj *(*csub_func)(Data_Obj *,index_t) );
*/

static void init_function_classes(void);
static void add_function_class(int class_index,Function *tbl);
static int find_function(char *name,int *type_p);

static Scalar_Expr_Node *node0(QSP_ARG_DECL  Scalar_Expr_Node_Code);
static Scalar_Expr_Node *node1(QSP_ARG_DECL  Scalar_Expr_Node_Code,Scalar_Expr_Node *);
static Scalar_Expr_Node *node2(QSP_ARG_DECL  Scalar_Expr_Node_Code,Scalar_Expr_Node *,Scalar_Expr_Node *);
static Scalar_Expr_Node *node3(QSP_ARG_DECL  Scalar_Expr_Node_Code,Scalar_Expr_Node *,Scalar_Expr_Node *,Scalar_Expr_Node *);

static double eval_expr(QSP_ARG_DECL  Scalar_Expr_Node *);

#define NODE0(code)	node0(QSP_ARG  code)
#define NODE1(code,enp)	node1(QSP_ARG  code,enp)
#define NODE2(code,enp1,enp2)	node2(QSP_ARG  code,enp1,enp2)
#define NODE3(code,enp1,enp2,enp3)	node3(QSP_ARG  code,enp1,enp2,enp3)

/* globals */
Data_Obj *(*obj_func)(QSP_ARG_DECL  const char *)=_def_obj;
Data_Obj *(*exist_func)(QSP_ARG_DECL  const char *)=_def_obj;
Data_Obj *(*sub_func)(QSP_ARG_DECL Data_Obj *,index_t)=_def_sub;
Data_Obj *(*csub_func)(QSP_ARG_DECL Data_Obj *,index_t)=_def_sub;

%}

%pure_parser	// make the parser rentrant (thread-safe)

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

%token <fval> NUMBER
%token <fundex> MATH0_FUNC
%token <fundex> MATH1_FUNC
%token <fundex> MATH2_FUNC
%token <fundex> DATA_FUNC
%token <fundex> SIZE_FUNC
%token <fundex> TS_FUNC
%token <fundex> MISC_FUNC
%token <fundex> STR_FUNC
%token <fundex> STR2_FUNC
%token <fundex> STR3_FUNC
%token <e_string> E_STRING
%token <e_string> E_QSTRING

%start topexp
%type <enp> expression
%type <enp> data_object
%type <enp> tsable
%type <e_string> e_string

%%

topexp		: expression { 
			// qsp is passed to yyparse through YYPARSE_PARAM, but it is void *
			YY_QSP->qs_final_expr_node_p = $1 ;
			}
		;

e_string	: E_STRING
		| E_QSTRING
		;

data_object	: E_STRING {
			$$=NODE0(N_OBJNAME);
			$$->sen_string = savestr($1);
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
			$$=NODE0(N_STRING);
			//$$=NODE0(N_OBJNAME);
			$$->sen_string = savestr($1);
			}
		| data_object '[' expression ']' {
			$$=NODE2(N_SUBSCRIPT,$1,$3); }
		| data_object '{' expression '}' {
			$$=NODE2(N_CSUBSCRIPT,$1,$3); }
		;

tsable		: E_STRING {
			$$ = NODE0(N_TSABLE);
			$$->sen_string=savestr($1);
			}
		;


expression	: NUMBER {
			$$ = NODE0(N_LITDBL);
			$$->sen_dblval = $1;
//sprintf(ERROR_STRING,"LITDBL node set to %28.28lg",$$->sen_dblval);
//ADVISE(ERROR_STRING);
			}
		| MATH0_FUNC '(' ')' {
			$$=NODE0(N_MATH0FUNC);
			$$->sen_index = $1;
			}
		| MATH1_FUNC '(' expression ')' {
			$$=NODE1(N_MATH1FUNC,$3);
			$$->sen_index = $1;
			}
		| MATH2_FUNC '(' expression ',' expression ')' {
			$$=NODE2(N_MATH2FUNC,$3,$5);
			$$->sen_index = $1;
			}
		| DATA_FUNC '(' data_object ')' {
			$$=NODE1(N_DATAFUNC,$3);
			$$->sen_index=$1;
			}
		| SIZE_FUNC '(' data_object ')' {
			$$=NODE1(N_SIZFUNC,$3);
			$$->sen_index=$1;
			}
		| TS_FUNC '(' tsable ',' expression ')' {
			$$=NODE2(N_TSFUNC,$3,$5);
			$$->sen_index=$1;
			}
		| MISC_FUNC '(' ')' {
			$$=NODE0(N_MISCFUNC);
			$$->sen_index=$1;
			}
		| STR_FUNC '(' e_string ')' {
			$$=NODE0(N_STRFUNC);
			$$->sen_string = savestr($3);
			$$->sen_index = $1;
			}
		| STR2_FUNC '(' e_string ',' e_string ')' {
			$$=NODE0(N_STR2FUNC);
			$$->sen_string=savestr($3);
			$$->sen_string2=savestr($5);
			$$->sen_index = $1;
			}
		| STR3_FUNC '(' e_string ',' e_string ',' expression ')' {
			$$=NODE1(N_STR3FUNC,$7);
			$$->sen_string=savestr($3);
			$$->sen_string2=savestr($5);
			$$->sen_index = $1;
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

		| data_object {
			/* must be a scalar object */
			$$=NODE1(N_SCALAR_OBJ,$1);
			}
		;


%%

static Data_Obj *obj_for_string(QSP_ARG_DECL  const char *string)
{
	Dimension_Set ds;
	Data_Obj *dp;

	/* this is just a string that we treat as a row vector
	 * of character data...
	 * We haven't actually created the data yet.
	 */
	ds.ds_dimension[0]=1;
	ds.ds_dimension[1]=strlen(string)+1;
	ds.ds_dimension[2]=1;
	ds.ds_dimension[3]=1;
	ds.ds_dimension[4]=1;
	dp=make_dobj(QSP_ARG  localname(),&ds,PREC_STR);
	if( dp != NO_OBJ ) strcpy((char *)dp->dt_data,string);
	return(dp);
}

static Scalar_Expr_Node *alloc_expr_node(SINGLE_QSP_ARG_DECL)
{
	Scalar_Expr_Node *enp;

	LOCK_ENODES

	if( free_enp_lp != NO_LIST ){
		Node *np;
		np = remHead(free_enp_lp);
		if( np != NO_NODE ){
			enp = (Scalar_Expr_Node *) np->n_data;
			rls_node(np);
		} else {
			enp = (Scalar_Expr_Node *) getbuf( sizeof(*enp) );
		}
	} else {
		enp = (Scalar_Expr_Node *) getbuf( sizeof(*enp) );
	}
	if( enp==NULL ) mem_err("alloc_expr_node");

	UNLOCK_ENODES


	enp->sen_child[0] = NO_EXPR_NODE;
	enp->sen_child[1] = NO_EXPR_NODE;
	enp->sen_string = NULL;
	enp->sen_string2 = NULL;
	return(enp);
}

static Scalar_Expr_Node *node0( QSP_ARG_DECL  Scalar_Expr_Node_Code code )
{
	Scalar_Expr_Node *enp;

//sprintf(ERROR_STRING,"node0( %d )",code);
//ADVISE(ERROR_STRING);
	enp = alloc_expr_node(SINGLE_QSP_ARG);
	enp->sen_code = code;
	return(enp);
}

static Scalar_Expr_Node *node1( QSP_ARG_DECL  Scalar_Expr_Node_Code code, Scalar_Expr_Node *child )
{
	Scalar_Expr_Node *enp;

//sprintf(ERROR_STRING,"node1( %d )",code);
//ADVISE(ERROR_STRING);
	enp = alloc_expr_node(SINGLE_QSP_ARG);
	enp->sen_code = code;
	enp->sen_child[0]=child;
	return(enp);
}

static Scalar_Expr_Node *node2( QSP_ARG_DECL  Scalar_Expr_Node_Code code, Scalar_Expr_Node *child1, Scalar_Expr_Node *child2 )
{
	Scalar_Expr_Node *enp;

//sprintf(ERROR_STRING,"node2( %d )",code);
//ADVISE(ERROR_STRING);
	enp = alloc_expr_node(SINGLE_QSP_ARG);
	enp->sen_code = code;
	enp->sen_child[0]=child1;
	enp->sen_child[1]=child2;
	return(enp);
}

static Scalar_Expr_Node *node3( QSP_ARG_DECL  Scalar_Expr_Node_Code code, Scalar_Expr_Node *child1, Scalar_Expr_Node *child2, Scalar_Expr_Node *child3 )
{
	Scalar_Expr_Node *enp;

//sprintf(ERROR_STRING,"node3( %d )",code);
//ADVISE(ERROR_STRING);
	enp = alloc_expr_node(SINGLE_QSP_ARG);
	enp->sen_code = code;
	enp->sen_child[0]=child1;
	enp->sen_child[1]=child2;
	enp->sen_child[2]=child3;
	return(enp);
}

/* Evaluate a parsed expression */

static Data_Obj *eval_dobj_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Data_Obj *dp=NO_OBJ,*dp2;
	index_t index;

	switch(enp->sen_code){
		case N_STRING:
			/* first try object lookup... */
			/* we don't want a warning if does not exist... */
			dp = (*exist_func)( QSP_ARG  enp->sen_string );
			/* We have a problem here with indexed objects,
			 * since the indexed names aren't in the database...
			 */
			if( dp == NO_OBJ ){
				/* treat the string like a rowvec of chars */
				dp = obj_for_string(QSP_ARG  enp->sen_string);
				return(dp);
			}
			break;

		case N_SCALAR_OBJ:
			dp = eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
			if( IS_SCALAR(dp) ) return(dp);
			return(NO_OBJ);
			break;
		case N_OBJNAME:
			dp = (*obj_func)( QSP_ARG  enp->sen_string );
			break;
		case N_SUBSCRIPT:
			dp2=eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
			index=(index_t)EVAL_EXPR(enp->sen_child[1]);
			dp=(*sub_func)( QSP_ARG  dp2, index );
			break;
		case N_CSUBSCRIPT:
			dp2=eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
			index=(index_t)EVAL_EXPR(enp->sen_child[1]);
			dp=(*csub_func)( QSP_ARG  dp2, index );
			break;

#ifdef CAUTIOUS
		default:
			sprintf(ERROR_STRING,
		"unexpected case (%d) in eval_dobj_expr",enp->sen_code);
			WARN(ERROR_STRING);
			break;
#endif /* CAUTIOUS */
	}
	return(dp);
}

static Item * eval_tsbl_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Item *ip=NULL;

	switch(enp->sen_code){
		case N_TSABLE:
			ip = find_tsable( QSP_ARG  enp->sen_string );
			if( ip == NO_ITEM ){
				sprintf(ERROR_STRING,
					"No time-stampable object \"%s\"!?",enp->sen_string);
				WARN(ERROR_STRING);
				return(NO_ITEM);
			}
			break;
#ifdef CAUTIOUS
		default:
			WARN("unexpected case in eval_tsbl_expr");
			break;
#endif /* CAUTIOUS */
	}
	return(ip);
}

static Item * eval_szbl_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Item *szp=NULL,*szp2;
	index_t index;
	switch(enp->sen_code){
		case N_STRING:
			szp = find_sizable( QSP_ARG  enp->sen_string );
			if( szp == NO_ITEM ){
				Data_Obj *dp;
				dp = obj_for_string(QSP_ARG  enp->sen_string);
				szp = (Item *)dp;
			}
			break;

		case N_OBJNAME:
		case N_SIZABLE:
			szp = find_sizable( QSP_ARG  enp->sen_string );
			if( szp == NO_ITEM ){
				sprintf(ERROR_STRING,
					"No sizable object \"%s\"!?",enp->sen_string);
				WARN(ERROR_STRING);
				return(NO_ITEM);
			}
			break;
		//case N_SUBSIZ:
		case N_SUBSCRIPT:
			szp2=EVAL_SZBL_EXPR(enp->sen_child[0]);
			if( szp2 == NO_ITEM )
				return(NO_ITEM);
			index = (index_t)EVAL_EXPR(enp->sen_child[1]);
			szp = sub_sizable(QSP_ARG  szp2,index);
			break;
		//case N_CSUBSIZ:
		case N_CSUBSCRIPT:
			szp2=EVAL_SZBL_EXPR(enp->sen_child[0]);
			if( szp2 == NO_ITEM )
				return(NO_ITEM);
			index = (index_t)EVAL_EXPR(enp->sen_child[1]);
			szp = csub_sizable(QSP_ARG  szp2,index);
			break;
#ifdef CAUTIOUS
		default:
			sprintf(ERROR_STRING,
		"unexpected case in eval_szbl_expr %d",enp->sen_code);
			WARN(ERROR_STRING);
			break;
#endif /* CAUTIOUS */
	}
	return(szp);
}

static void divzer_error(SINGLE_QSP_ARG_DECL)
{
	sprintf(ERROR_STRING,"Error parsing \"%s\"",YY_ORIGINAL);
	ADVISE(ERROR_STRING);
	WARN("eval_expr:  divide by 0!?");
}

static void dump_enode(QSP_ARG_DECL  Scalar_Expr_Node *enp)
{
	switch(enp->sen_code){
		case N_STRING:
			sprintf(ERROR_STRING,"0x%lx\t%d string\t%s",
				(int_for_addr)enp, enp->sen_code, enp->sen_string);
			ADVISE(ERROR_STRING);
			break;

		case N_SIZABLE:
			sprintf(ERROR_STRING,"0x%lx\t%d sizable\t%s",
				(int_for_addr)enp, enp->sen_code, enp->sen_string);
			ADVISE(ERROR_STRING);
			break;

		case N_SIZFUNC:
			sprintf(ERROR_STRING,"0x%lx\t%d sizefunc\t%s",
				(int_for_addr)enp, enp->sen_code, size_functbl[ enp->sen_index ].fn_name);
			ADVISE(ERROR_STRING);
			break;

		case N_DATAFUNC:
			sprintf(ERROR_STRING,"0x%lx\tdatafunc\t%s",
				(int_for_addr)enp, data_functbl[ enp->sen_index ].fn_name);
			ADVISE(ERROR_STRING);
			break;

		case N_OBJNAME:
			sprintf(ERROR_STRING,"0x%lx\tobjname\t%s",
				(int_for_addr)enp, enp->sen_string);
			ADVISE(ERROR_STRING);
			break;
			
		case N_SCALAR_OBJ:
			sprintf(ERROR_STRING,"0x%lx\tscalar_obj\t0x%lx",
				(int_for_addr)enp, (int_for_addr)enp->sen_child[0]);
			ADVISE(ERROR_STRING);
			break;
			
		case N_SUBSCRIPT:
			sprintf(ERROR_STRING,"0x%lx\tsubscript\t0x%lx\t0x%lx",
				(int_for_addr)enp, (int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_CSUBSCRIPT:
			sprintf(ERROR_STRING,"0x%lx\t%d Csubscript\t0x%lx\t0x%lx",
				(int_for_addr)enp, enp->sen_code,
				(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
			
		case N_MATH1FUNC:
			sprintf(ERROR_STRING,"0x%lx\tmath1func\t%s",
				(int_for_addr)enp, math1_functbl[ enp->sen_index ].fn_name);
			ADVISE(ERROR_STRING);
			break;
			
		case N_TIMES:
			sprintf(ERROR_STRING,"0x%lx\ttimes\t0x%lx\t0x%lx",
				(int_for_addr)enp,(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
		case N_DIVIDE:
			sprintf(ERROR_STRING,"0x%lx\tdivide\t0x%lx\t0x%lx",
				(int_for_addr)enp,(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
		case N_LITDBL:
			sprintf(ERROR_STRING,"0x%lx\tlit_dbl\t%g",
				(int_for_addr)enp,enp->sen_dblval);
			ADVISE(ERROR_STRING);
			break;
		case N_GT:
			sprintf(ERROR_STRING,"> (GT)");
			ADVISE(ERROR_STRING);
			break;
		/* Comment out the default case for the compiler to
		 * report unhandled cases from the enumerated type...
		 */
		default:
			sprintf(ERROR_STRING,
		"%s - %s:  unhandled node code %d",
				WHENCE(dump_enode),enp->sen_code);
			ADVISE(ERROR_STRING);
			break;
	}
	
}

static void dump_etree(QSP_ARG_DECL  Scalar_Expr_Node *enp)
{
	if( enp == NULL ) return;
	dump_enode(QSP_ARG  enp);
	dump_etree(QSP_ARG  enp->sen_child[0]);
	dump_etree(QSP_ARG  enp->sen_child[1]);
}

static double eval_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	double dval,dval2,dval3;
	Data_Obj *dp;
	Item *szp;
	long ival,ival2,ival3;
	u_long uval2,uval3;
	u_long frm;

#ifdef DEBUG
if( debug & expr_debug ){
sprintf(ERROR_STRING,"eval_expr:  code = %d",enp->sen_code);
ADVISE(ERROR_STRING);
}
#endif /* DEBUG */

	switch(enp->sen_code){

	case N_MATH0FUNC:
		dval = (*math0_functbl[ enp->sen_index ].fn_func.v_func)( );
		break;
	case N_MATH1FUNC:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval = (*math1_functbl[ enp->sen_index ].fn_func.d1_func)( dval2 );
		break;
	case N_MATH2FUNC:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		dval = (*math2_functbl[ enp->sen_index ].fn_func.d2_func)( dval2 , dval3 );
		break;

	case N_DATAFUNC:
		dp = eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
		if( dp == NO_OBJ ){ dval = 0.0; } else {
		dval = (*data_functbl[ enp->sen_index ].dof_func)( dp ); }
		break;
	case N_SIZFUNC:
		szp = EVAL_SZBL_EXPR(enp->sen_child[0]);
		dval = (*size_functbl[ enp->sen_index ].szf_func)( QSP_ARG  szp );
		break;
	case N_TSFUNC:
		szp = eval_tsbl_expr(QSP_ARG  enp->sen_child[0]);
		frm = EVAL_EXPR(enp->sen_child[1]);
		dval = (*timestamp_functbl[ enp->sen_index ].tsf_func)( QSP_ARG  szp, frm );
		break;
	case N_MISCFUNC:
		dval = (*misc_functbl[ enp->sen_index ].vd_func)();
		break;
	case N_STRFUNC:
		dval = (*str1_functbl[ enp->sen_index ].str1f_func)( QSP_ARG  enp->sen_string );
		break;

	case N_STR2FUNC:
		dval = (*str2_functbl[ enp->sen_index ].str2f_func)( enp->sen_string, enp->sen_string2 );
		break;

	case N_STR3FUNC:
		dval = (*str3_functbl[ enp->sen_index ].str3f_func)( enp->sen_string, enp->sen_string2, (int) EVAL_EXPR( enp->sen_child[0]) );
		break;

	case N_PLUS:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		dval=dval2+dval3;
		break;

	case N_SCALAR_OBJ:
		/* to get the value of a scalar object, we need the data object library, but we'd like
		 * this to be standalone for cases where we aren't linking with libdata...
		 */
#ifdef FOOBAR
		dp=eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
		if( dp == NO_OBJ ) {
			/* This looks like the wrong error message?  Should be obj does not exist??? */
			WARN("eval_expr:  object is not a scalar");
			dval = 0.0;
		} else {
			Scalar_Value sv;
			extract_scalar_value(&sv,dp);
			dval = cast_from_scalar_value(&sv,dp->dt_prec);
		}
#endif /* FOOBAR */
		/* instead of explicitly doing this stuff, we just call the function from the datafunctbl...
		 * We have looked and know that the value function is the first entry with index 0, but
		 * it is a BUG to have that value hard-coded here...
		 */

		dp = eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
		if( dp == NO_OBJ ){ dval = 0.0; } else {
		/* BUG the 0 index in the line below is only correct
		 * because the "value" function is the first one
		 * in the data_functbl - this should be a symbolic
		 * constant, or better yet, the table should be scanned
		 * for the string "value", and the index stored in a var.
		 * But this is good enough for now...
		 */
		dval = (*data_functbl[ 0 ].dof_func)( dp ); }

		break;

	/* do-nothing */
	case N_OBJNAME:
	case N_SUBSCRIPT:
	case N_CSUBSCRIPT:
	case N_SIZABLE:
#ifdef FOOBAR
	case N_SUBSIZ:
	case N_CSUBSIZ:
#endif
	case N_TSABLE:
		sprintf(ERROR_STRING,
			"unexpected case (%d) in eval_expr",
			enp->sen_code);
		WARN(ERROR_STRING);
		dval = 0.0;	// quiet compiler
		break;

	case N_LITDBL:
		dval=enp->sen_dblval;
		break;

	case N_MINUS:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		dval=dval2-dval3;
		break;
	case N_DIVIDE:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		if( dval3==0.0 ){
			divzer_error(SINGLE_QSP_ARG);
			dval=0;
		} else dval=dval2/dval3;
		break;
	case N_TIMES:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		dval=dval2*dval3;
		break;
	case N_MODULO:
		ival2=(long)EVAL_EXPR(enp->sen_child[0]);
		ival3=(long)EVAL_EXPR(enp->sen_child[1]);
		if( ival3==0.0 ){
			divzer_error(SINGLE_QSP_ARG);
			dval=0;
		} else dval=(ival2%ival3);
		break;
	case N_BITAND:
		ival2=(long)EVAL_EXPR(enp->sen_child[0]);
		ival3=(long)EVAL_EXPR(enp->sen_child[1]);
		dval = (ival2&ival3);
		break;
	case N_BITOR:
		ival2=(long)EVAL_EXPR(enp->sen_child[0]);
		ival3=(long)EVAL_EXPR(enp->sen_child[1]);
		dval = (ival2|ival3);
		break;
	case N_BITXOR:
		ival2=(long)EVAL_EXPR(enp->sen_child[0]);
		ival3=(long)EVAL_EXPR(enp->sen_child[1]);
		dval = (ival2^ival3);
		break;
	case N_SHL:
		uval2=(u_long)EVAL_EXPR(enp->sen_child[0]);
		uval3=(u_long)EVAL_EXPR(enp->sen_child[1]);
		dval =  uval2 << uval3;
		break;
	case N_NOT:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		if( dval2 == 0.0 ) dval=1.0;
		else dval=0.0;
		break;
	case N_BITCOMP:
		uval2=(u_long)EVAL_EXPR(enp->sen_child[0]);
		dval = (double)(~uval2);
		break;
	case N_UMINUS:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval=(-dval2);
		break;
	case N_EQUIV:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		if( dval2 == dval3 ) dval=1.0;
		else dval=0.0;
		break;
	case N_LOGOR:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		if( dval2!=0.0 ) dval=1.0;
		else {
			dval3=EVAL_EXPR(enp->sen_child[1]);
			if( dval3!=0.0 ) dval=1.0;
			else dval=0.0;
		}
		break;
	case N_LOGAND:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		if( dval2==0.0 ) dval=0.0;
		else {
			dval3=EVAL_EXPR(enp->sen_child[1]);
			if( dval3==0.0 ) dval=0.0;
			else dval=1.0;
		}
		break;
	case N_LOGXOR:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		if( (dval2 == 0.0 && dval3 != 0.0) || (dval2 != 0.0 && dval3 == 0.0) ){
			dval = 1;
		} else {
			dval = 0;
		}
		break;
	case N_LT:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		if( dval2 < dval3 ) dval=1.0;
		else dval=0.0;
		break;
	case N_GT:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		if( dval2 > dval3 ) dval=1.0;
		else dval=0.0;
		break;
	case N_GE:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		if( dval2 >= dval3 ) dval=1.0;
		else dval=0.0;
		break;
	case N_LE:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		if( dval2 <= dval3 ) dval=1.0;
		else dval=0.0;
		break;
	case N_NE:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		if( dval2 != dval3 ) dval=1.0;
		else dval=0.0;
		break;

	case N_SHR:
		dval2 = EVAL_EXPR(enp->sen_child[0]);
		ival = (long)EVAL_EXPR(enp->sen_child[1]);
		/* this version clears the sign bit */
		/* dval = ((((int)dval2) >> 1) & 077777) >> ((ival) - 1); */
		dval = ((u_long)dval2)>>ival;
		break;
	case N_CONDITIONAL:
		dval2 = EVAL_EXPR(enp->sen_child[0]);
		if( dval2 != 0 )
			dval = EVAL_EXPR(enp->sen_child[1]);
		else
			dval = EVAL_EXPR(enp->sen_child[2]);
		break;
#ifdef FOOBAR
	case N_SLCT_CHAR:
ADVISE("case N_SLCT_CHAR");
		ival = EVAL_EXPR(enp->sen_child[0]);
		if( ival < 0 || ival >= strlen(enp->sen_string) )
			dval = -1;
		else
			dval = enp->sen_string[ival];
		break;
#endif /* FOOBAR */

#ifdef CAUTIOUS
	default:
		sprintf(ERROR_STRING,
			"%s - %s:  CAUTIOUS:  unhandled node code case %d!?",
			WHENCE(eval_expr),
			enp->sen_code);
		WARN(ERROR_STRING);
		dval=0.0;	// quiet compiler
		break;
#endif /* CAUTIOUS */

	}

	return(dval);
}

/* We used to call yyerror for lexical scan errors, but that often
 * caused yyerror to be called twice, the first time for the lexer
 * error, the second time with a parsing error...
 */

static void llerror(QSP_ARG_DECL  const char *msg)
{
	char tmp_str[LLEN];	/* don't use error_string! */

	sprintf(tmp_str,"pexpr lexical scan error:  %s",msg);
	WARN(tmp_str);
}

#define BUF_CHAR(c)					\
							\
	if( i > buflen ){				\
		WARN("extract_number_string:  string buffer too small!?");	\
		return(-1);				\
	}						\
	buf[i++] = (c);


static int extract_number_string(QSP_ARG_DECL  char *buf, int buflen, const char **srcp)
{
	int i=0;
	const char *s;
	int sign=1;
	int retval=0;

	s = *srcp;

	while( isspace(*s) && *s!=0 ) s++;

	if( *s == 0 ) return(-1);

	if( (!isdigit(*s)) && *s!='+' && *s!='-' && *s!='.' ) return(-1);

	if( *s == '+' ) s++;
	else if( *s == '-' ){
		sign=(-1);
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
				WARN("extract_number_string:  mal-formed hex string");
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
			WARN("extract_number_string:  malformed exponent!?");
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

double parse_number(QSP_ARG_DECL  const char **strptr)
{
	/* the pointed-to text is not necessarily null-terminated... */
	const char *ptr;
	char *endptr;
	double d;
	long l;
	char buf[128];
	int status;

	ptr = *strptr;

	status = extract_number_string(QSP_ARG  buf,128,&ptr);
	*strptr = ptr;
	if( status < 0 ){
		sprintf(ERROR_STRING,"parse_number:  bad number string \"%s\"",
			ptr);
		WARN(ERROR_STRING);
		return(-1);
	}
	if( status == 4					/* hex */
		|| status == 1				/* leading zero, no decimal */
		){

		errno = 0;
		l=strtol(buf,&endptr,0);
		if( errno == ERANGE ){
			long long ll1;

			errno=0;
			ll1=strtoll(buf,&endptr,0);
			if( errno != 0 ){
				sprintf(ERROR_STRING,"long long conversion error!?  (errno=%d)",errno);
				WARN(ERROR_STRING);
				tell_sys_error("strtoll");
			}
			if( ll1 > 0 && ll1 <=0xffffffff ){	/* fits in an unsigned long */
				unsigned long ul;
				ul = ll1;
				return(ul);
			} else {
				sprintf(ERROR_STRING,"long conversion error!?  (errno=%d)",errno);
				WARN(ERROR_STRING);
				tell_sys_error("strtol");
				errno=0;
			}
		}
		if( errno != 0 ){
			sprintf(ERROR_STRING,"long conversion error!?  (errno=%d)",errno);
			WARN(ERROR_STRING);
			tell_sys_error("strtol");
		}
		return(l);
	}
	/*   else if( status & 2 ){ */	/* decimal pt seen */

		errno = 0;
//sprintf(ERROR_STRING,"converting string \"%s\"",buf);
//ADVISE(ERROR_STRING);
		d = strtod(buf,&endptr);
		if( errno == ERANGE ){
sprintf(ERROR_STRING,"strtod:  range error buf=\"%s\", d = %g",buf,d);
ADVISE(ERROR_STRING);
		} else if( errno != 0 ){
			sprintf(ERROR_STRING,"double conversion error!?  (errno=%d)",errno);
			WARN(ERROR_STRING);
			tell_sys_error("strtod");
		}

//sprintf(ERROR_STRING,"flt conversion returning %lg",d);
//ADVISE(ERROR_STRING);
		return(d);
	/* } */

} /* end parse_number() */

static double yynumber(SINGLE_QSP_ARG_DECL)
{
//sprintf(ERROR_STRING,"yynumber calling parse_number %s",YYSTRPTR[EDEPTH]);
//ADVISE(ERROR_STRING);
	return( parse_number( QSP_ARG  (const char **)&YYSTRPTR[EDEPTH]) );
}

static const char * varval(SINGLE_QSP_ARG_DECL)
{
	char tmpbuf[128];
	const char *s;
	const char *var_valstr;
	int c;

	/* indirect variable reference? */

	if( *YYSTRPTR[EDEPTH] == '$' ){
		YYSTRPTR[EDEPTH]++;
		s = varval(SINGLE_QSP_ARG) ;
	} else {
		/* read in the variable name */
		char *sp;
		sp=tmpbuf;
		c=(*YYSTRPTR[EDEPTH]);
		while( isalpha(c) || c == '_' || isdigit(c) ){
			*sp++ = c;
			YYSTRPTR[EDEPTH]++;
			c=(*YYSTRPTR[EDEPTH]);
		}
		*sp=0;
		s=tmpbuf;
	}

	var_valstr = var_value(QSP_ARG  s);

	if( var_valstr == NULL )
		return("0");
	else
		return( var_valstr );
}

int whfunc(Function *table,const char *str)		/* shouldn't this be in function.c ? */
{
	int i;
	Function *func;

	i=0;
	func=table;;
	while( (*func->fn_name) != 0 ){
		if( !strcmp(str,func->fn_name) ) return(i);
		func++;
		i++;
	}
	return(-1);
}

static void add_function_class(int class_index,Function *tbl)
{
	if( n_function_classes >= MAX_FUNCTION_CLASSES ){
		NERROR1("Too many function classes; recompile with larger value for MAX_FUNCTION_CLASSES in nexpr.y");
		return;
	}
	func_class[n_function_classes].fc_class = class_index;
	func_class[n_function_classes].fc_tbl = tbl;
	n_function_classes++;
}

static void init_function_classes()
{
	add_function_class(MATH0_FUNC,(Function *)(void *)math0_functbl);
	add_function_class(MATH1_FUNC,(Function *)(void *)math1_functbl);
	add_function_class(MATH2_FUNC,(Function *)(void *)math2_functbl);
	add_function_class(DATA_FUNC,(Function *)(void *)data_functbl);
	add_function_class(SIZE_FUNC,(Function *)(void *)size_functbl);
	add_function_class(TS_FUNC,(Function *)(void *)timestamp_functbl);
	add_function_class(MISC_FUNC,(Function *)(void *)misc_functbl);
	add_function_class(STR_FUNC,(Function *)(void *)str1_functbl);
	add_function_class(STR2_FUNC,(Function *)(void *)str2_functbl);
	add_function_class(STR3_FUNC,(Function *)(void *)str3_functbl);
}

static int find_function(char *name,int *type_p)
{
	int i,j;

	if( n_function_classes <= 0 )
		init_function_classes();

	for(j=0;j<n_function_classes;j++){
		i=whfunc(func_class[j].fc_tbl,name);
		if( i!= (-1) ){
			*type_p = func_class[j].fc_class;
			return(i);
		}
	}
	return(-1);
}

#ifdef THREAD_SAFE_QUERY
int yylex(YYSTYPE *yylvp, Query_Stream *qsp)	/* return the next token */
#else /* ! THREAD_SAFE_QUERY */
int yylex(YYSTYPE *yylvp)			/* return the next token */
#endif /* ! THREAD_SAFE_QUERY */
{
	int c;
	char *s;

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
			yylvp->fval=yynumber(SINGLE_QSP_ARG);
#ifdef DEBUG
//if( debug & expr_debug ){
////sprintf(ERROR_STRING,"yylex:  NUMBER = %g",yylvp->fval);
//sprintf(ERROR_STRING,"yylex:  NUMBER = XXX");
////ADVISE(ERROR_STRING);
//}
#endif /* DEBUG */
			return(NUMBER);
		} else if( c == '$' ) {
			YYSTRPTR[EDEPTH]++;
			if( (EDEPTH+1) >= MAXEDEPTH ){
				LLERROR("expression depth too large");
				return(0);
			}
			YYSTRPTR[EDEPTH+1]=varval(SINGLE_QSP_ARG);
			/* varval should advance YYSTRPTR[edpth] */
			EDEPTH++;
			/* keep looping */
		} else if( IS_LEGAL_FIRST_CHAR(c) ){	/* get a name */
			int i;

			s=_strbuf[WHICH_EXPR_STR];
			*s++ = (*YYSTRPTR[EDEPTH]++);
			while( IS_LEGAL_NAME_CHAR(*YYSTRPTR[EDEPTH]) ){
				*s++ = (*YYSTRPTR[EDEPTH]++);
			}
			*s=0;

			yylvp->fundex = find_function(_strbuf[WHICH_EXPR_STR],&i);
			if( yylvp->fundex >= 0 ){
#ifdef DEBUG
//if( debug & expr_debug ){
//ADVISE("yylex:  function");
//}
#endif /* DEBUG */
				return(i);
			}

			yylvp->e_string=_strbuf[WHICH_EXPR_STR];
			WHICH_EXPR_STR++;
			WHICH_EXPR_STR %= MAX_E_STRINGS;
#ifdef DEBUG
//if( debug & expr_debug ){
//ADVISE("yylex:  E_STRING");
//}
#endif /* DEBUG */
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
				int qchar;

				qchar=c;
				s=_strbuf[WHICH_EXPR_STR];

				/* copy string into a buffer */
				while( *YYSTRPTR[EDEPTH] &&
					*YYSTRPTR[EDEPTH] != qchar )
					*s++ = (*YYSTRPTR[EDEPTH]++);
				*s=0;
				if( *YYSTRPTR[EDEPTH] == qchar ){
					YYSTRPTR[EDEPTH]++;
					/* used to call var_expand here,
					 * but now this is done automatically.
					 */
				} else LLERROR("unmatched quote");

				yylvp->e_string=_strbuf[WHICH_EXPR_STR];
				WHICH_EXPR_STR++;
				WHICH_EXPR_STR %= MAX_E_STRINGS;
#ifdef DEBUG
//if( debug & expr_debug ){
//ADVISE("yylex:  quoted E_STRING");
//}
#endif /* DEBUG */
				return(E_QSTRING);	/* quoted string */
			}
#ifdef DEBUG
//if( debug & expr_debug ){
//ADVISE("yylex:  punct char");
//}
#endif /* DEBUG */
			return(c);
		} else {
			LLERROR("yylex error");
			return(0);
		}
	}
#ifdef DEBUG
//if( debug & expr_debug ){
//ADVISE("yylex:  0 (EDEPTH<0)");
//}
#endif /* DEBUG */
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
	if( enp->sen_string != NULL )
		rls_str(enp->sen_string);
	if( enp->sen_string2 != NULL )
		rls_str(enp->sen_string2);

	if( free_enp_lp == NO_LIST ){
		free_enp_lp = new_list();
#ifdef CAUTIOUS
		if( free_enp_lp == NO_LIST ) NERROR1("CAUTIOUS:  rls_tree:  error creating free enp list");
#endif /* CAUTIOUS */
	}
	np = mk_node(enp);
	addHead(free_enp_lp,np);
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

double pexpr(QSP_ARG_DECL  const char *buf)	/** parse expression */
{
	int stat;
	double dval;

#ifdef DEBUG
	if( expr_debug == 0 )
		expr_debug = add_debug_module(QSP_ARG  "expressions");
#endif /* DEBUG */

#ifdef DEBUG
if( debug & expr_debug ){
sprintf(ERROR_STRING,"%s - %s:  BEGIN %s, in_pexpr = %d",
WHENCE(pexpr),buf,IN_PEXPR);
ADVISE(ERROR_STRING);
}
#endif /* DEBUG */

	if( IN_PEXPR ) {
#ifdef DEBUG
if( debug & expr_debug ){
ADVISE("pexpr:  nested call to pexpr, calling parse_number");
}
#endif /* DEBUG */
		return( parse_number(QSP_ARG  &buf) );
	}

	IN_PEXPR=1;
	EDEPTH=0;
	YY_ORIGINAL=YYSTRPTR[EDEPTH]=buf;

	stat=yyparse(SINGLE_QSP_ARG);

	if( stat != 0 ){
		/* Need to somehow free allocated nodes... */
		if( verbose ){
			sprintf(ERROR_STRING,"yyparse returned status %d",stat);
			ADVISE(ERROR_STRING);
		}
		IN_PEXPR=0;
		return(0.0);
	}

#ifdef DEBUG
if( debug & expr_debug ){
dump_etree(QSP_ARG  FINAL_EXPR_NODE_P);
}
#endif /* DEBUG */

	dval = EVAL_EXPR(FINAL_EXPR_NODE_P);
#ifdef DEBUG
if( debug & expr_debug ){
sprintf(ERROR_STRING,"pexpr:  s=\"%s\", dval = %g",buf,dval);
ADVISE(ERROR_STRING);
}
#endif /* DEBUG */

#ifdef SUN
	/*
	 * this is a strange thing on the SUN, that
	 * zero can have the sign bit set, and that printf
	 * recognizes this and prints -0 !?
	 */

	if( iszero(dval) && signbit(dval) ){
		Scalar_Expr_Node *enp;

		enp=NODE0(N_LITDBL);
		enp->sen_dblval=(-1.0);
		enp=NODE2(N_TIMES,FINAL_EXPR_NODE_P,enp);
		FINAL_EXPR_NODE_P=enp;
		dval = EVAL_EXPR(FINAL_EXPR_NODE_P);
	}
#endif /* SUN */

	LOCK_ENODES

	rls_tree(FINAL_EXPR_NODE_P);

	UNLOCK_ENODES

	IN_PEXPR=0;

	return( dval );
}

int yyerror(QSP_ARG_DECL  char *s)
{
	sprintf(ERROR_STRING,"YYERROR:  %s",s);
	WARN(ERROR_STRING);

	sprintf(ERROR_STRING,"parsing \"%s\"",YY_ORIGINAL);
	ADVISE(ERROR_STRING);

	if( *YYSTRPTR[0] ){
		sprintf(ERROR_STRING,"\"%s\" left to parse",YYSTRPTR[0]);
		ADVISE(ERROR_STRING);
	} else {
		ADVISE("No buffered text left to parse");
	}

	/* final=(-1); */
	/* -1 is a bad value, because when the target is an
	 * unsigned in (dimension_t), the cast makes it a very
	 * large number...  causes all sorts of problems!
	 */
	//FINAL_EXPR_NODE_P=NO_EXPR_NODE;
	THIS_QSP->qs_final_expr_node_p = NO_EXPR_NODE;
	/* BUG need to release nodes here... */
	return(0);
}

/*
 * This is the default data object locater.
 * If the data module were assumed to be always included
 * with the support library, then we wouldn't need this
 * here, but doing this allows us to run the parser
 * without the data module, but has the same grammer...
 */

static Data_Obj *_def_obj(QSP_ARG_DECL  const char *name)
{
	sprintf(ERROR_STRING,"can't search for object \"%s\"; ",name);
	WARN(ERROR_STRING);

	WARN("data module not linked");
	return(NO_OBJ);
}

static Data_Obj *_def_sub(QSP_ARG_DECL  Data_Obj *object,index_t index)
{
	WARN("can't get subobject; data module not linked");
	return(NO_OBJ);
}

void set_obj_funcs(Data_Obj *(*ofunc)(QSP_ARG_DECL  const char *),
			Data_Obj *(*efunc)(QSP_ARG_DECL  const char *),
			Data_Obj *(*sfunc)(QSP_ARG_DECL  Data_Obj *,index_t),
			Data_Obj *(*cfunc)(QSP_ARG_DECL  Data_Obj *,index_t))
{
	obj_func=ofunc;
	exist_func=efunc;
	sub_func=sfunc;
	csub_func=cfunc;
}

