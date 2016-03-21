%{
#include "quip_config.h"

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

// The file yacc_hack redefines the symbols used by yacc so that two parsers
// can play together...
#include "yacc_hack.h"		// could be obviated by bison cmd line arg

//#include "savestr.h"		/* not needed? BUG */
#include "data_obj.h"
#include "undef_sym.h"
#include "debug.h"
#include "getbuf.h"
#include "node.h"
#include "function.h"
/* #include "warproto.h" */
#include "query.h"
#include "quip_prot.h"
#include "veclib/vec_func.h"
#include "warn.h"

#include "vectree.h"

/* for definition of function codes */
#include "veclib/vecgen.h"

#ifdef SGI
#include <alloca.h>
#endif

#ifdef QUIP_DEBUG
debug_flag_t parser_debug=0;
#endif /* QUIP_DEBUG */

#define YY_LLEN 1024

static char yy_word_buf[YY_LLEN]; // BUG global is not thread-safe!


//#ifdef THREAD_SAFE_QUERY
// old-style, now use %parse-param {}
//#define YYPARSE_PARAM qsp	/* gets declared void * instead of Query_Stack * */
/* For yyerror */
//#define YY_(msg)	QSP_ARG msg
//#endif /* THREAD_SAFE_QUERY */

void yyerror(Query_Stack *qsp,  char *);

static int whkeyword(Keyword *table,const char *str);

double parse_stuff(SINGLE_QSP_ARG_DECL);

double nullfunc();
double dummyfunc();
double rn_number();
double dstrcmp();

/* We use a fixed number of static string buffers to hold string names.
 * This works ok if we have a short program, or if we use the strings
 * right away, but craps out for subroutine declarations, where we have
 * to remember the subroutine name until the end of the body!
 *
 * Alternatively, we could save all strings, and then free when we're
 * done parsing... or free when we're done with each one.
 */

struct position { int x,y; };

typedef struct decl_info {
	char *di_name;
	int di_cols;
	int di_rows;
	int di_frms;
} decl_info;


/* what yylval can be */

typedef union {
	Vec_Expr_Node *enp;
	//Vec_Func_Code fcode;	/* index to our tables here... */
	int   fundex;		/* function index */
	Function *func_p;
	double dval;		/* actual value */
	int intval;
	Data_Obj *dp;
	Identifier *idp;
	Subrt *srp;
	const char *e_string;
	List *list;
	Node *node;
	void *v;
	struct position posn;
	Vec_Func_Code vfc;
	Precision *prec_p;
} YYSTYPE;

static int name_token(QSP_ARG_DECL  YYSTYPE *yylvp);

#define YYSTYPE_IS_DECLARED			/* necessary on a 2.6 machine?? */

//#ifdef THREAD_SAFE_QUERY
//
//int yylex(YYSTYPE *yylvp, Query_Stack *qsp);
//#define YYLEX_PARAM SINGLE_QSP_ARG
//
//#else
//
//int yylex(YYSTYPE *yylvp);
//
//#endif
int yylex(YYSTYPE *yylvp, Query_Stack *qsp);

#define YY_ERR_STR	QS_ERROR_STRING(((Query_Stack *)qsp))	// this macro casts qsp, unlike ERROR_STRING

%}

/* This line stops yacc invoked on linux... */
//%pure_parser	/* make the parser rentrant (thread-safe) */
%pure-parser	/* updated syntax - make the parser rentrant (thread-safe) */

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
 * The first entries have lower precedence.
 */

/* We have 22 shift/reduce conflicts, this line suppresses the
 * message reporting that when we run bison, but will generate
 * an error if the number changes.
 */

/* %expect 22 */

/* a phony precedence for int 2 float conversion
 * to suppress(?) 8 reduce reduce conflicts...
 */
%right '='
%right TIMES_EQ
%right PLUS_EQ
%right PLUS_PLUS
%right MINUS_MINUS
%right MINUS_EQ
%right DIV_EQ
%right OR_EQ
%right AND_EQ
%right XOR_EQ
%right SHL_EQ
%right SHR_EQ
%left '?'		/* ot sure where these should go...   need to check K&R */
%left ':'
%left <vfc> LOGOR
%left <vfc> LOGXOR
%left <vfc> LOGAND
/* why do the bitwise operators have such low precedence? */
%left <vfc> '|'
%left <vfc> '^'
%left <vfc> '&'
%left <vfc> LOG_EQ NE	/* logical operators ==,!=   is equal to, is not equal to */
%left <vfc> '<' '>' GE LE
%left <vfc> SHL SHR
/* why should subtraction be right associative?? */
%left <vfc> '+' '-'
%left <vfc> '*' '/' '%'
%left <vfc> DOT
%left <vfc> '[' ']'
%right <vfc> '!' '~'
/* just a tag for assigning high precedence to unary minus, plus */
%left <vfc> UNARY
%left <vfc> ','

%token <dval> NUMBER
//%token <dval> VARIABLE
%token <dval> INT_NUM
%token <dval> CHAR_CONST

%token <func_p> MATH0_FUNC
%token <func_p> MATH1_FUNC
%token <func_p> MATH2_FUNC
%token <func_p> INT1_FUNC
%token <func_p> STR1_FUNC
%token <func_p> STR2_FUNC
%token <func_p> STR3_FUNC
%token <func_p> STRV_FUNC
%token <func_p> CHAR_FUNC
%token <func_p> DATA_FUNC
%token <func_p> SIZE_FUNC
%token <func_p> TS_FUNC
/* %token <func_p> MISC_FUNC */

%token <fundex> BEGIN_COMMENT
%token <fundex> END_COMMENT

/* keyword tokens */
%token <fundex> WHILE
%token <fundex> UNTIL
%token <fundex> CONTINUE
%token <fundex> SWITCH
%token <fundex> CASE
%token <fundex> DEFAULT
%token <fundex> BREAK
%token <fundex> GOTO
%token <fundex> DO
%token <fundex> FOR

%token <fundex> STATIC
%token <fundex> BYTE
%token <fundex> CHAR
%token <fundex> STRING
%token <fundex> FLOAT
%token <fundex> DOUBLE
%token <fundex> SHORT
%token <fundex> INT32
%token <fundex> INT64
%token <fundex> BIT
%token <fundex> UBYTE
%token <fundex> USHORT
%token <fundex> UINT32
%token <fundex> UINT64
%token <fundex> COLOR
%token <fundex> COMPLEX
%token <fundex> DBLCPX
%token <fundex> QUATERNION
%token <fundex> DBLQUAT

%token <fundex> STRCPY
%token <fundex> NAME_FUNC
%token <fundex> FILE_EXISTS
%token <fundex> STRCAT
%token <fundex> ECHO
%token <fundex> ADVISE_FUNC
%token <fundex> DISPLAY
%token <fundex> F_WARN
%token <fundex> PRINT
%token <fundex> INFO
%token <fundex> IF
%token <fundex> ELSE
%token <fundex> RETURN
%token <fundex> EXIT

%token <fundex> MINVAL
%token <fundex> MAXVAL
%token <fundex> WRAP
%token <fundex> SCROLL
%token <fundex> DILATE
%token <fundex> FIX_SIZE
%token <fundex> FILL
%token <fundex> CLR_OPT_PARAMS
%token <fundex> ADD_OPT_PARAM
%token <fundex> OPTIMIZE
%token <fundex> ERODE
/*
%token <fundex> RENDER
%token <fundex> SVD
%token <fundex> SVBK
*/
%token <fundex> ENLARGE
%token <fundex> REDUCE
%token <fundex> WARP
%token <fundex> LOOKUP
%token <fundex> EQUIVALENCE
%token <fundex> TRANSPOSE
%token <fundex> CONJ
%token <fundex> MAX_TIMES
%token <fundex> MAX_INDEX
%token <fundex> MIN_INDEX
%token <fundex> DFT
%token <fundex> IDFT
%token <fundex> RDFT
%token <fundex> RIDFT
%token <fundex> REAL_PART
%token <fundex> IMAG_PART
%token <fundex> RAMP
%token <fundex> SUM
%token <fundex> END
%token <fundex> NEXT_TOKEN	/* used by yylex, not parser */
%token <fundex> NEWLINE
%token <fundex> SET_OUTPUT_FILE
%token <fundex> LOAD
%token <fundex> SAVE
%token <fundex> FILETYPE
%token <fundex> OBJ_OF


%token <srp> FUNCNAME
%token <srp> REFFUNC
%token <srp> SCRIPTFUNC
%token <dp> OBJNAME
%token <idp> PTRNAME
%token <idp> STRNAME
%token <idp> LABELNAME
%token <idp> FUNCPTRNAME

%token <e_string> LEX_STRING
%token <e_string> NEWNAME
/*
%type  <e_string> oldname
%type  <e_string> badname
*/
%type  <e_string> decl_identifier

%start program

%type <enp> program
%type <enp> prog_elt
%type <enp> subroutine
%type <enp> new_func_decl
%type <enp> old_func_decl
/* %type <enp> function_prototype */
%type <enp> func_args
%type <enp> func_arg
%type <enp> arg_decl
%type <enp> arg_decl_list
%type <enp> blk_stat
%type <enp> statline
%type <enp> loop_stuff
%type <enp> simple_stat
%type <enp> stat_list
%type <enp> stat_block
%type <enp> if_statement
%type <enp> case_statement
%type <enp> case_list
%type <enp> single_case
%type <enp> switch_cases
%type <enp> switch_statement
%type <enp> loop_statement
%type <enp> assignment
%type <enp> str_assgn
%type <enp> ptr_assgn
%type <enp> funcptr_assgn
%type <enp> ref_arg
%type <enp> func_ref_arg
%type <enp> void_call
%type <enp> decl_statement
%type <enp> decl_stat_list
%type <enp> print_stat
%type <enp> misc_stat
%type <enp> fileio_stat
%type <enp> script_stat
%type <enp> return_stat
%type <enp> exit_stat
%type <enp> info_stat

%type <enp> objref
%type <enp> pointer
%type <enp> func_ptr
%type <enp> str_ptr
%type <enp> str_ptr_arg
%type <enp> subsamp_spec

%type <enp> string
%type <enp> string_list
%type <enp> string_arg
%type <enp> printable
%type <enp> print_list
%type <enp> mixed_list
%type <enp> mixed_item

%type <enp> expr_list
%type <enp> row_list
%type <enp> comp_list
%type <enp> list_obj
%type <enp> comp_stack

%type <enp> decl_item_list

%type <enp> decl_item

%type <enp> expression

%type <prec_p> data_type
%type <prec_p> precision
%token <intval> VOID_TYPE
%token <intval> EXTERN
//%token <intval> CONST_TYPE
%token <intval> NATIVE_FUNC_NAME

%%

/*
 * References
 *
 * We would like to save the dp's in the node, to speed execution,
 * But for subroutines (local variables) we have no idea where the
 * objects will be when they are created...  We might like to store
 * their index in their context, or something like that, but for
 * now we just save their name.
 */

pointer		: PTRNAME
			{
			$$=NODE0(T_POINTER);
			SET_VN_STRING($$, savestr(ID_NAME($1)));
			}
		;

func_ptr	: FUNCPTRNAME
			{
			$$ = NODE0(T_FUNCPTR);
			SET_VN_STRING($$, savestr(ID_NAME($1)));
			}
		;

str_ptr		: STRNAME	/* name of a string object */
			{
			$$=NODE0(T_STR_PTR);
			SET_VN_STRING($$, savestr(ID_NAME($1)));
			}
		;

subsamp_spec	:	expression ':' expression ':' expression
			{
			$$=NODE3(T_RANGE,$1,$3,$5);
			}
		;

objref		: OBJNAME
			{
			if( OBJ_FLAGS($1) & DT_STATIC ){
				$$=NODE0(T_STATIC_OBJ);
				SET_VN_OBJ($$, $1);
				// To be safe, we need to mark
				// the object so that it can't be
				// deleted while this reference
				// exists...  We don't want to 
				// have a dangling pointer!?
			} else {
				const char *s;
				$$=NODE0(T_DYN_OBJ);
				s=savestr(OBJ_NAME($1));
				SET_VN_STRING($$,s);
			}
			}
		| '*' pointer %prec UNARY
			{
			$$ = NODE1(T_DEREFERENCE,$2);
			}
		| OBJ_OF '(' string_list ')'
			{
			$$=NODE1(T_OBJ_LOOKUP,$3);
			}
		| NEWNAME
			{
			Undef_Sym *usp;

			usp=undef_of(QSP_ARG  $1);
			if( usp == NO_UNDEF ){
				/* BUG?  are contexts handled correctly??? */
				sprintf(YY_ERR_STR,"Undefined symbol %s",$1);
				yyerror(qsp,  YY_ERR_STR);
				/*usp=*/new_undef(QSP_ARG  $1);
			}
			$$=NODE0(T_UNDEF);
			SET_VN_STRING($$, savestr($1));
			CURDLE($$)
			}
		| REAL_PART '(' objref ')' {
			$$=NODE1(T_REAL_PART,$3);
			}
		| IMAG_PART '(' objref ')' {
			$$=NODE1(T_IMAG_PART,$3);
			}
		| objref '[' expression ']' {
			$$=NODE2(T_SQUARE_SUBSCR,$1,$3);
			}
		| objref '{' expression '}' {
			$$=NODE2(T_CURLY_SUBSCR,$1,$3);
			}
		| objref '[' expression ':' expression ']'
			{
			$$=NODE3(T_SUBVEC,$1,$3,$5);
			}
		| objref '{' expression ':' expression '}'
			{
			/* Why not use T_RANGE2 here?  The current version
			 * is fine as-is, but don't get rid of T_RANGE2 because
			 * mlab.y uses it...
			 */
			$$=NODE3(T_CSUBVEC,$1,$3,$5);
			}
		| objref '[' subsamp_spec ']'
			{
			$$=NODE2(T_SUBSAMP,$1,$3);
			}
		| objref '{' subsamp_spec '}'
			{
			$$=NODE2(T_CSUBSAMP,$1,$3);
			}
		;


expression	: FIX_SIZE '(' expression ')'
			{
			$$=NODE1(T_FIX_SIZE,$3);
			}
		/*
		| pointer {
			$$=NO_VEXPR_NODE;
			sprintf(YY_ERR_STR,"Need to dereference pointer \"%s\"",VN_STRING($1));
			yyerror(THIS_QSP,  YY_ERR_STR);
			}
			*/
		| string_arg
		| '(' data_type ')' expression %prec UNARY
			{
			$$ = NODE1(T_TYPECAST,$4);
			SET_VN_CAST_PREC_PTR($$,$2);
			}
		| '(' expression ')' {
			$$ = $2; }
		| expression '+' expression {
			$$=NODE2(T_PLUS,$1,$3); }
		| expression '-' expression {
			$$=NODE2(T_MINUS,$1,$3); }
		| expression '*' expression {
			$$=NODE2(T_TIMES,$1,$3); }
		| expression '/' expression {
			$$=NODE2(T_DIVIDE,$1,$3); }
		| expression '%' expression {
			$$=NODE2(T_MODULO,$1,$3); }
		| expression '&' expression {
			$$=NODE2(T_BITAND,$1,$3); }
		| expression '|' expression {
			$$=NODE2(T_BITOR,$1,$3); }
		| expression '^' expression {
			$$=NODE2(T_BITXOR,$1,$3); }
		| expression SHL expression {
			$$=NODE2(T_BITLSHIFT,$1,$3); }
		| expression SHR expression {
			$$=NODE2(T_BITRSHIFT,$1,$3); }
		| '~' expression %prec UNARY {
			$$=NODE1(T_BITCOMP,$2); }
		| INT_NUM {
			$$ = NODE0(T_LIT_INT);
			SET_VN_INTVAL($$, (int) $1);
			}
		| expression LOG_EQ expression {
			$$=NODE2(T_BOOL_EQ,$1,$3);
			}
		| expression '<' expression {
			$$ = NODE2(T_BOOL_LT,$1,$3);
			}
		| expression '>' expression {
			$$=NODE2(T_BOOL_GT,$1,$3);
			}
		| expression GE expression {
			$$=NODE2(T_BOOL_GE,$1,$3);
			}
		| expression LE expression {
			$$=NODE2(T_BOOL_LE,$1,$3);
			}
		| expression NE expression {
			$$=NODE2(T_BOOL_NE,$1,$3);
			}
		| expression LOGAND expression {
			$$=NODE2(T_BOOL_AND,$1,$3);
			}
		| expression LOGOR expression {
			$$=NODE2(T_BOOL_OR,$1,$3);
			}
		| expression LOGXOR expression {
			$$=NODE2(T_BOOL_XOR,$1,$3);
			}
		| '!' expression {
			$$=NODE1(T_BOOL_NOT,$2);
			}
		| pointer NE ref_arg {
			Vec_Expr_Node *enp;
			enp=NODE2(T_BOOL_PTREQ,$1,$3);
			$$=NODE1(T_BOOL_NOT,enp);
			}
		/* We'd like to have ref_arg == ref_arg, but we can't figure out
		 * how to get rid of the parsing ambiguity when we see:
		 *	& objref ==
		 *
		 * arising from the rules:
		 *	ref_arg -> & objref
		 *	expression -> expression == expression
		 *	expression -> objref
		 */

		| pointer LOG_EQ ref_arg {
			$$=NODE2(T_BOOL_PTREQ,$1,$3);
			}

		| MATH0_FUNC '(' ')' 
			{
			$$=NODE0(T_MATH0_FN);
			SET_VN_FUNC_PTR($$,$1);
			}
		| MATH1_FUNC '(' expression ')' 
			{
			$$=NODE1(T_MATH1_FN,$3);
			SET_VN_FUNC_PTR($$,$1);
			}
		| MATH2_FUNC '(' expression ',' expression ')' 
			{
			$$=NODE2(T_MATH2_FN,$3,$5);
			SET_VN_FUNC_PTR($$,$1);
			}
		| INT1_FUNC '(' expression ')' 
			{
			$$=NODE1(T_INT1_FN,$3);
			SET_VN_FUNC_PTR($$,$1);
			}
		/* can we have a general object here?? */
		| expression DOT expression {
			$$ = NODE2(T_INNER,$1,$3);
			}
		| NUMBER {
			$$=NODE0(T_LIT_DBL);
			SET_VN_DBLVAL($$,$1);
			}
		| expression '?' expression ':' expression
			{
			/* We determine exactly which type later */
			$$ = NODE3(T_SS_S_CONDASS,$1,$3,$5);
			}
		| CHAR_CONST {
			$$ = NODE0(T_LIT_INT);
			SET_VN_INTVAL($$, (int) $1);
			}
		| DATA_FUNC
			{
			$$=NODE0(T_BADNAME);
			NODE_ERROR($$);
			CURDLE($$)
			WARN("illegal use of data function");
			}
		| DATA_FUNC '(' objref ')' {
			$$=NODE1(T_DATA_FN,$3);
			SET_VN_FUNC_PTR($$,$1);
			}

		| SIZE_FUNC '(' string_arg ')' {
			$$=NODE1(T_SIZE_FN,$3);
			SET_VN_FUNC_PTR($$,$1);
			}
		| SIZE_FUNC '(' objref ')' {
			$$=NODE1(T_SIZE_FN,$3);
			SET_VN_FUNC_PTR($$,$1);
			}
		| SIZE_FUNC '(' pointer ')' {
			$$=NODE1(T_SIZE_FN,$3);
			SET_VN_FUNC_PTR($$,$1);
			NODE_ERROR($$);
			advise("dereference pointer before passing to size function");
			CURDLE($$)
			}
		| SUM '(' pointer ')' {
			sprintf(YY_ERR_STR,"need to dereference pointer %s",VN_STRING($3));
			yyerror(THIS_QSP,  YY_ERR_STR);
			$$=NO_VEXPR_NODE;
			}

		| SUM '(' expr_list ')' {
			$$=NODE1(T_SUM,$3);
			}

		| FILE_EXISTS '(' string_arg ')'
			{
			$$=NODE1(T_FILE_EXISTS,$3);
			}
		| STR1_FUNC '(' string_arg ')'
			{
			$$=NODE1(T_STR1_FN,$3);
			SET_VN_FUNC_PTR($$,$1);
			}
		| STR2_FUNC '(' string_arg ',' string_arg ')'
			{
			$$=NODE2(T_STR2_FN,$3,$5);
			SET_VN_FUNC_PTR($$,$1);
			}
		// What are the 3-arg string functions???
		| STR3_FUNC '(' string_arg ',' string_arg ')'
			{
			$$=NODE2(T_STR2_FN,$3,$5);
			SET_VN_FUNC_PTR($$,$1);
			}
		// string-valued functions, toupper, tolower
		| STRV_FUNC '(' string_arg ')'
			{
			$$=NODE1(T_STRV_FN,$3);
			SET_VN_FUNC_PTR($$,$1);
			}
		// char arg functions, isupper, islower, etc
		// output is a bitmap...
		| CHAR_FUNC '(' string_arg ')'
			{
			$$=NODE1(T_CHAR_FN,$3);
			SET_VN_FUNC_PTR($$,$1);
			}
		/* miscellaneous functions are currframe() and recordable() for omdr */
/*
		| MISC_FUNC '(' ')' {
				$$=NODE0(T_MISC_FN);
				SET_VN_FUNC_PTR($$,$1);
				}
*/

		| CONJ '(' expression ')'
			{
			$$=NODE1(T_CONJ,$3);
			}

		/* unary minus */
		| '-' expression %prec UNARY {
				$$=NODE1(T_UMINUS,$2);
				}
		| MINVAL '(' expr_list ')'
			{
			$$=NODE1(T_MINVAL,$3);
			}
		| MAXVAL '(' expr_list ')'
			{
			$$=NODE1(T_MAXVAL,$3);
			}

		| MAX_INDEX '(' expression ')'
			{ $$=NODE1(T_MAX_INDEX,$3); }
		| MIN_INDEX '(' expression ')'
			{ $$=NODE1(T_MIN_INDEX,$3); }

		| '(' '*' func_ptr ')' '(' func_args ')'
			{
			$$ = NODE2(T_INDIR_CALL,$3,$6);
			}
		| FUNCNAME '(' func_args ')'
			{
			$$=NODE1(T_CALLFUNC,$3);
			SET_VN_CALL_SUBRT($$, $1);
			/* make sure this is not a void subroutine! */
			if( SR_PREC_CODE($1) == PREC_VOID ){
				NODE_ERROR($$);
				sprintf(YY_ERR_STR,"void subroutine %s used in expression!?",SR_NAME($1));
				advise(YY_ERR_STR);
				CURDLE($$)
			}
			}
		| comp_stack {
			$$=$1;
			}
		| list_obj {
			$$=$1;
			}
		| WARP '(' expression ',' expression ')' {
			WARN("warp not implemented");
			$$=$3;
			}
		| LOOKUP '(' expression ',' expression ')' {
			$$=NODE2(T_LOOKUP,$3,$5);
			}
		| TRANSPOSE '(' expression ')'			/* transpose */
			{
			$$ = NODE1(T_TRANSPOSE,$3);
			SET_VN_SIZCH_SHAPE($$, ALLOC_SHAPE );
			}
		| DFT '(' expression ')' { $$ = NODE1(T_DFT,$3); }
		| IDFT '(' expression ')' { $$ = NODE1(T_IDFT,$3); }
		| RDFT '(' expression ')' {
			$$ = NODE1(T_RDFT,$3);
			SET_VN_SIZCH_SHAPE($$, ALLOC_SHAPE );
			}
		| RIDFT '(' expression ')' {
			$$ = NODE1(T_RIDFT,$3);
			SET_VN_SIZCH_SHAPE($$, ALLOC_SHAPE );
			}
		| assignment
		| WRAP '(' expression ')' {
			$$=NODE1(T_WRAP,$3);
			}
		| SCROLL '(' expression ',' expression ',' expression ')' {
			$$=NODE3(T_SCROLL,$3,$5,$7);
			}

		| ERODE '(' expression ')'
			{ $$ = NODE1(T_ERODE,$3); }

		| DILATE '(' expression ')'
			{ $$ = NODE1(T_DILATE,$3); }

		| ENLARGE '(' expression ')' {
			$$=NODE1(T_ENLARGE,$3);
			SET_VN_SIZCH_SHAPE($$, ALLOC_SHAPE );
			}
		| REDUCE '(' expression ')' {
			$$=NODE1(T_REDUCE,$3);
			SET_VN_SIZCH_SHAPE($$, ALLOC_SHAPE );
			}
		| LOAD '(' string_arg ')'
			{ $$=NODE1(T_LOAD,$3); }
		| RAMP '(' expression ',' expression ',' expression')' {
				$$=NODE3(T_RAMP,$3,$5,$7);
				}
		| MAX_TIMES '(' ref_arg ',' ref_arg ',' expression ')'
			{
			$$ = NODE3(T_MAX_TIMES,$3,$5,$7);
			}

		;

func_arg	: expression
		| '&' expression
			{ $$=NODE1(T_REFERENCE,$2); }
		/*
		| '&' FUNCNAME
			{
			$$=NODE0(T_FUNCREF);
			SET_VN_SUBRT($$, $2);
			}
		*/
		| func_ref_arg
		| pointer
		| ptr_assgn
		| '&' pointer
			{
			sprintf(YY_ERR_STR,"shouldn't try to reference pointer variable %s",VN_STRING($2));
			yyerror(THIS_QSP,  YY_ERR_STR);
			$$=$2;
			}
		|
			{
			$$=NO_VEXPR_NODE;
			}
		;

func_args	: func_arg
		| func_args ',' func_arg
			{
			$$=NODE2(T_ARGLIST,$1,$3);
			}
		;




void_call	: FUNCNAME '(' func_args ')'
			{
			/* BUG check to see that this subrt is void! */
			$$=NODE1(T_CALLFUNC,$3);
			SET_VN_CALL_SUBRT($$, $1);
			if( SR_PREC_CODE($1) != PREC_VOID ){
				NODE_ERROR($$);
				sprintf(YY_ERR_STR,"return value of function %s is ignored",SR_NAME($1));
				advise(YY_ERR_STR);
			}
			}
		| '(' '*' func_ptr ')' '(' func_args ')'
			{
			/* BUG check to see that the pointed to subrt is void -
			 * OR should we check that on pointer assignment?
			 */
			$$ = NODE2(T_INDIR_CALL,$3,$6);
			}
		;


ref_arg		: '&' objref %prec UNARY
			{ $$ = NODE1(T_REFERENCE,$2); }
		| ptr_assgn
		| pointer
		| EQUIVALENCE '(' objref ',' expr_list ',' precision ')'
			{
				$$=NODE2(T_EQUIVALENCE,$3,$5);
				SET_VN_DECL_PREC($$, $7);
			}
		| REFFUNC '(' func_args ')'
			{
			$$=NODE1(T_CALLFUNC,$3);
			SET_VN_CALL_SUBRT($$, $1);
			/* make sure this is not a void subroutine! */
			if( SR_PREC_CODE($1) == PREC_VOID ){
				NODE_ERROR($$);
				sprintf(YY_ERR_STR,"void subroutine %s used in pointer expression!?",SR_NAME($1));
				advise(YY_ERR_STR);
				CURDLE($$)
			}
			}
		;

func_ref_arg	: '&' FUNCNAME %prec UNARY
			{
			$$=NODE0(T_FUNCREF);
			SET_VN_SUBRT($$, $2);
			}
		| funcptr_assgn
		| func_ptr
		;

ptr_assgn	: pointer '=' ref_arg {
			$$=NODE2(T_SET_PTR,$1,$3);
			}
		;

funcptr_assgn	: func_ptr '=' func_ref_arg {
			$$ = NODE2(T_SET_FUNCPTR,$1,$3);
			}
		;

str_assgn	: str_ptr '=' print_list {
			$$=NODE2(T_SET_STR,$1,$3);
			}
		;

/*
 * Assignments
 *
 * Perhaps we should define some rules for bad assignments,
 * so that we can give a more helpful error msg than YYERROR!?
 */

assignment	: objref '=' expression {
			$$=NODE2(T_ASSIGN,$1,$3);
			}
		| objref PLUS_PLUS { $$=NODE1(T_POSTINC,$1); }
		| PLUS_PLUS objref { $$=NODE1(T_PREINC,$2); }
		| MINUS_MINUS objref { $$=NODE1(T_PREDEC,$2); }
		| objref MINUS_MINUS { $$=NODE1(T_POSTDEC,$1); }
		| objref PLUS_EQ expression {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_PLUS,$1,$3);
			dup_enp=DUP_TREE($1);
			$$=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
		| objref TIMES_EQ expression {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_TIMES,$1,$3);
			dup_enp=DUP_TREE($1);
			$$=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
		| objref MINUS_EQ expression {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_MINUS,$1,$3);
			dup_enp=DUP_TREE($1);
			$$=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
		| objref DIV_EQ expression {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_DIVIDE,$1,$3);
			dup_enp=DUP_TREE($1);
			$$=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
		| objref AND_EQ expression {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITAND,$1,$3);
			dup_enp=DUP_TREE($1);
			$$=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
		| objref OR_EQ expression {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITOR,$1,$3);
			dup_enp=DUP_TREE($1);
			$$=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
		| objref XOR_EQ expression {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITXOR,$1,$3);
			dup_enp=DUP_TREE($1);
			$$=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
		| objref SHL_EQ expression {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITLSHIFT,$1,$3);
			dup_enp=DUP_TREE($1);
			$$=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
		| objref SHR_EQ expression {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITRSHIFT,$1,$3);
			dup_enp=DUP_TREE($1);
			$$=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
		;


/*
 * Statements
 */

statline	: simple_stat ';'
			{ $$ = $1; }
		| stat_block
		| switch_statement
		| NEWNAME ':'
			{
			Identifier *idp;
			$$ = NODE0(T_LABEL);
			idp = new_id(QSP_ARG  $1);
			SET_ID_TYPE(idp, ID_LABEL);
			SET_VN_STRING($$, savestr(ID_NAME(idp)));
			}
		| LABELNAME ':'
			{
			$$ = NODE0(T_LABEL);
			SET_VN_STRING($$, savestr(ID_NAME($1)));
			}
		| error ';'
			{ $$ = NO_VEXPR_NODE; }
		;

/* a stat_list is something that can appear within curly braces */

stat_list	: statline
		| blk_stat
		| stat_list statline
			{
			if( $2 != NULL ){
				if( $1 != NULL )
					$$=NODE2(T_STAT_LIST,$1,$2);
				else
					$$ = $2;
			} else {
				$$=$1;
			}
			}
		| stat_list blk_stat
			{
			$$=NODE2(T_STAT_LIST,$1,$2);
			}
		;

/* A stat_block is a group of statements inside curly braces.
 * Although we allow declarations at the beginning of such a block,
 * they are not properly scoped (BUG).  To fix this, we would probably
 * have to create and destroy contexts on block entrance/exit
 * (like we do now for subroutines).
 */

stat_block	: '{' stat_list '}'
			{
			$$=$2;
			}
		/* This doesn't do anything, but should not be a syntax error */
		| '{' decl_stat_list '}'
			{
			$$=$2;
			}
		| '{' decl_stat_list stat_list '}'
			{
			$$=NODE2(T_STAT_LIST,$2,$3);
			}
		| '{' '}'		/* empty block */
			{
			$$=NO_VEXPR_NODE;
			}
		| '{' error '}'
			{
			$$=NO_VEXPR_NODE;
			}
		| '{' stat_list END
			{
			yyerror(THIS_QSP,  (char *)"missing '}'");
			$$=NO_VEXPR_NODE;
			}
		;

new_func_decl	: NEWNAME '(' arg_decl_list ')'
			{
			set_subrt_ctx(QSP_ARG  $1);		/* when do we unset??? */
			/* We evaluate the declarations here so we can parse the body, but
			 * the declarations get interpreted a second time when we compile the nodes -
			 * at least, for prototype declarations!?  Not a problem for regular declarations?
			 */
			if( $3 != NO_VEXPR_NODE )
				EVAL_DECL_TREE($3);
			$$ = NODE1(T_PROTO,$3);
			SET_VN_STRING($$, savestr($1));
			}
		;

old_func_decl	: FUNCNAME '(' arg_decl_list ')'
			{
			if( SR_FLAGS($1) != SR_PROTOTYPE ){
				sprintf(YY_ERR_STR,"Subroutine %s multiply defined!?",SR_NAME($1));
				yyerror(THIS_QSP,  YY_ERR_STR);
				/* now what??? */
			}
			set_subrt_ctx(QSP_ARG  SR_NAME($1));		/* when do we unset??? */

			/* compare the two arg decl trees
			 * and issue a warning if they do not match.
			 */
			compare_arg_trees(QSP_ARG  $3,SR_ARG_DECLS($1));

			/* use the new ones */
			SET_SR_ARG_DECLS($1, $3);
			/* BUG?? we might want to release the old tree... */

			/* We also need to make sure that the type of the function matches
			 * the original prototype...
			 * But we do this later.
			 */

			/* We have to evaluate the new declarations to be able to parse
			 * the body...
			 */

			if( $3 != NO_VEXPR_NODE )
				EVAL_DECL_TREE($3);

			$$=NODE1(T_PROTO,$3);
			/* BUG why are we storing the name again?? */
			SET_VN_STRING($$, savestr(SR_NAME($1)));
			}
		;

subroutine	: data_type new_func_decl stat_block
			{
			Subrt *srp;
			srp=remember_subrt(QSP_ARG  $1,VN_STRING($2),VN_CHILD($2,0),$3);
			SET_SR_PREC_PTR(srp, $1);
			$$=NODE0(T_SUBRT);
			SET_VN_SUBRT($$,srp);
			delete_subrt_ctx(QSP_ARG  VN_STRING($2));	/* this deletes the objects... */
			// But why is the context in existence here?
			COMPILE_SUBRT(srp);
			}
		| data_type '*' new_func_decl stat_block
			{
			Subrt *srp;
			srp=remember_subrt(QSP_ARG  $1,VN_STRING($3),VN_CHILD($3,0),$4);
			SET_SR_PREC_PTR(srp, $1);
			SET_SR_FLAG_BITS(srp, SR_REFFUNC);
			/* set a flag to show returns ptr */
			$$=NODE0(T_SUBRT);
			SET_VN_SUBRT($$,srp);
			delete_subrt_ctx(QSP_ARG  VN_STRING($3));	/* this deletes the objects... */
			COMPILE_SUBRT(srp);
			}
		| data_type old_func_decl stat_block
			{
			/* BUG make sure that precision matches prototype decl */
			Subrt *srp;
			srp=subrt_of(QSP_ARG  VN_STRING($2));
//#ifdef CAUTIOUS
//			if( srp == NO_SUBRT ) {
//				NODE_ERROR($2);
//				ERROR1("CAUTIOUS:  missing subrt!?");
//			}
//#endif /* CAUTIOUS */
			assert( srp != NO_SUBRT );

			update_subrt(QSP_ARG  srp,$3);
			$$=NODE0(T_SUBRT);
			SET_VN_SUBRT($$,srp);
			delete_subrt_ctx(QSP_ARG  VN_STRING($2));
			COMPILE_SUBRT(srp);
			}
		;


arg_decl_list	:		/* nuthin */
			{
			$$=NO_VEXPR_NODE;
			}
		| arg_decl
			{
			$$=$1;
			}
		| arg_decl_list ',' arg_decl
			{
			$$=NODE2(T_DECL_STAT_LIST,$1,$3);
			}
		;

prog_elt	: subroutine
		| decl_statement
			{
			if( $$ != NO_VEXPR_NODE ) {
				// decl_stats are always evaluated,
				// to create the objects for compilation...
				SET_VN_FLAG_BITS($$,NODE_FINISHED);
			}
			}
		| statline
			{
			if( $$ != NO_VEXPR_NODE ) {
				EVAL_IMMEDIATE($$);
				// We don't release here,
				// because these nodes get passed up
				// to program nonterminal...
				SET_VN_FLAG_BITS($$,NODE_FINISHED);
			}
			}
		| blk_stat
			{
			if( $$ != NO_VEXPR_NODE ) {
				EVAL_IMMEDIATE($$);
				SET_VN_FLAG_BITS($$,NODE_FINISHED);
			}
			}
		;

program		: prog_elt END
			{ TOP_NODE=$1;  }
		| prog_elt
			{ TOP_NODE=$1; }
		| program prog_elt END {
			$$=NODE2(T_STAT_LIST,$1,$2);
			if( $1 != NULL && NODE_IS_FINISHED($1) &&
					$2 != NULL && NODE_IS_FINISHED($2) )
				SET_VN_FLAG_BITS($$,NODE_FINISHED);
			TOP_NODE=$$;
			}
		| program prog_elt {
			// We don't need to make lists of statements
			// already executed!?
			$$=NODE2(T_STAT_LIST,$1,$2);
			if( $1 != NULL && NODE_IS_FINISHED($1) &&
					$2 != NULL && NODE_IS_FINISHED($2) )
				SET_VN_FLAG_BITS($$,NODE_FINISHED);
			TOP_NODE=$$;
			}
		| error END
			{
			$$ = NO_VEXPR_NODE;
			TOP_NODE=$$;
			}
		;

data_type	: precision
		/* | CONST_TYPE precision { $$ = const_precision($2) ; } */
		;

precision	: BYTE { $$		= PREC_FOR_CODE(PREC_BY);	}
		| CHAR { $$		= PREC_FOR_CODE(PREC_CHAR);	}
		| STRING { $$		= PREC_FOR_CODE(PREC_STR);	}
		| FLOAT { $$		= PREC_FOR_CODE(PREC_SP);	}
		| DOUBLE { $$		= PREC_FOR_CODE(PREC_DP);	}
		| COMPLEX { $$		= PREC_FOR_CODE(PREC_CPX);	}
		| DBLCPX { $$		= PREC_FOR_CODE(PREC_DBLCPX);	}
		| QUATERNION { $$	= PREC_FOR_CODE(PREC_QUAT);	}
		| DBLQUAT { $$		= PREC_FOR_CODE(PREC_DBLQUAT);	}
		| SHORT { $$		= PREC_FOR_CODE(PREC_IN);	}
		| INT32 { $$		= PREC_FOR_CODE(PREC_DI);	}
		| INT64 { $$		= PREC_FOR_CODE(PREC_LI);	}
		| UBYTE { $$		= PREC_FOR_CODE(PREC_UBY);	}
		| USHORT { $$		= PREC_FOR_CODE(PREC_UIN);	}
		| UINT32 { $$		= PREC_FOR_CODE(PREC_UDI);	}
		| UINT64 { $$		= PREC_FOR_CODE(PREC_ULI);	}
		| BIT { $$		= PREC_FOR_CODE(PREC_BIT);	}
		| COLOR { $$		= PREC_FOR_CODE(PREC_COLOR);	}
		| VOID_TYPE { $$	= PREC_FOR_CODE(PREC_VOID);	}
		;


info_stat	: INFO '(' expr_list ')'
			{ $$=NODE1(T_INFO,$3); }
		| DISPLAY '(' expr_list ')'
			{ $$=NODE1(T_DISPLAY,$3); }
		;

exit_stat	: EXIT { $$=NODE0(T_EXIT); }
		| EXIT '(' ')' { $$=NODE0(T_EXIT); }
		| EXIT '(' expression ')' { $$=NODE1(T_EXIT,$3); }
		;

return_stat	: RETURN
			{
			$$=NODE1(T_RETURN,NO_VEXPR_NODE);
			}
		| RETURN '(' ')'
			{
			$$=NODE1(T_RETURN,NO_VEXPR_NODE);
			}
		/*
		| RETURN '(' expression ')'
			{
			$$=NODE1(T_RETURN,$3);
			}
			*/
		| RETURN expression
			{
			$$=NODE1(T_RETURN,$2);
			}
		| RETURN '(' ref_arg ')'
			{
			$$=NODE1(T_RETURN,$3);
			}
		| RETURN ref_arg
			{
			$$=NODE1(T_RETURN,$2);
			}
		;

fileio_stat	:	SAVE '(' string_arg ',' expression ')'
			{ $$=NODE2(T_SAVE,$3,$5); }
		| FILETYPE '(' string_arg ')'
			{ $$=NODE1(T_FILETYPE,$3); }
		;

script_stat	:	SCRIPTFUNC '(' print_list ')'
			{
			$$=NODE1(T_SCRIPT,$3);
			SET_VN_SUBRT($$, $1);
			}
		| SCRIPTFUNC '(' ')'
			{
			$$=NODE1(T_SCRIPT,NO_VEXPR_NODE);
			SET_VN_SUBRT($$, $1);
			}
		;

str_ptr_arg	: str_ptr
		| NEWNAME
			{
			sprintf(YY_ERR_STR,"undefined string pointer \"%s\"",$1);
			yyerror(THIS_QSP,  YY_ERR_STR);
			$$=NO_VEXPR_NODE;
			}
		;

misc_stat	: STRCPY '(' str_ptr_arg ',' printable ')'
			{
			$$ = NODE2(T_STRCPY,$3,$5);
			}
		| STRCAT '(' str_ptr_arg ',' printable ')'
			{
			$$ = NODE2(T_STRCAT,$3,$5);
			}
		/*
		| SVD '(' ref_arg ',' ref_arg ',' ref_arg ')'
			{ $$ = NODE3(T_SVD,$3,$5,$7); }
	*/

		| NATIVE_FUNC_NAME '(' func_args ')'
			{
			$$ = NODE1(T_CALL_NATIVE,$3);
			SET_VN_INTVAL($$, $1);
			}
			/*
		| SVBK '(' ref_arg ',' expression ',' expression ',' expression ',' expression ')'
			{
				Vec_Expr_Node *enp,*enp2;
				enp=NODE2(T_EXPR_LIST,$5,$7);
				enp2=NODE2(T_EXPR_LIST,$9,$11);
				$$ = NODE3(T_SVBK,$3,enp,enp2);
			}
			*/
		| FILL '(' ref_arg ',' expression ',' expression ',' expression ',' expression ')'
			{
			Vec_Expr_Node *enp,*enp2;
			enp=NODE2(T_EXPR_LIST,$5,$7);
			enp2=NODE2(T_EXPR_LIST,$9,$11);
			$$ = NODE3(T_FILL,$3,enp,enp2);
			}
		| CLR_OPT_PARAMS '(' ')'
			{
			$$ = NODE0(T_CLR_OPT_PARAMS);
			}
		/*                                initial        min            max            incr           mininc */
		| ADD_OPT_PARAM '(' ref_arg ',' expression ',' expression ',' expression ',' expression ',' expression ')'
			{
			Vec_Expr_Node *enp1,*enp2,*enp3;
			enp1=NODE2(T_EXPR_LIST,$3,$5);
			enp2=NODE2(T_EXPR_LIST,$7,$9);
			enp3=NODE2(T_EXPR_LIST,$11,$13);
			$$ = NODE3(T_ADD_OPT_PARAM,enp1,enp2,enp3);
			}
		| OPTIMIZE '(' FUNCNAME ')'
			{
			$$ = NODE0(T_OPTIMIZE);
			SET_VN_SUBRT($$, $3);
			}


		| SET_OUTPUT_FILE '(' string_arg ')'
			{ $$=NODE1(T_OUTPUT_FILE,$3); }

		;

print_stat	: PRINT '(' mixed_list ')' { $$=NODE1(T_EXP_PRINT,$3); }
		| ECHO '(' print_list ')' { $$=NODE1(T_EXP_PRINT,$3); }
		| ADVISE_FUNC '(' print_list ')' { $$=NODE1(T_ADVISE,$3); }
		| F_WARN '(' print_list ')' { $$=NODE1(T_WARN,$3); }
		;

decl_identifier	: NEWNAME
		| OBJNAME
			{ $$ = OBJ_NAME($1); }
		| PTRNAME
			{ $$ = ID_NAME($1); }
		| STRNAME
			{ $$ = ID_NAME($1); }
		|	precision
			{
			yyerror(THIS_QSP,  (char *)"illegal attempt to use a keyword as an identifier");
			$$="<illegal_keyword_use>";
			}
		;

decl_item	: decl_identifier {
			$$ = NODE0(T_SCAL_DECL);
			// WHY VN_STRING and not VN_DECL_NAME???
			SET_VN_STRING($$,savestr($1));	/* bug need to save??? */
			}
		/*
		| decl_identifier '(' arg_decl_list ')' {
			$$ = NODE1(T_PROTO,$3);
			SET_VN_STRING($$, savestr($1));
			}
		*/
		| new_func_decl
			{
			delete_subrt_ctx(QSP_ARG  VN_STRING($1));
			}
		| old_func_decl				/* repeated prototype */
			{
			delete_subrt_ctx(QSP_ARG  VN_STRING($1));
			}
		| '(' '*' decl_identifier ')' '(' arg_decl_list ')'
			{
			/* function pointer */
			$$ = NODE1(T_FUNCPTR_DECL,$6);
			SET_VN_DECL_NAME($$,savestr($3));
			}
		| decl_identifier '{' expression '}' {
			$$ = NODE1(T_CSCAL_DECL,$3);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| decl_identifier '[' expression ']' {
			$$ = NODE1(T_VEC_DECL,$3);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| decl_identifier '[' expression ']' '{' expression '}' {
			$$ = NODE2(T_CVEC_DECL,$3,$6);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| decl_identifier '[' expression ']' '[' expression ']' {
			// The type is stored at the parent node...
			// Since we "compile" the nodes depth first,
			// how does it get here?
			$$=NODE2(T_IMG_DECL,$3,$6);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| decl_identifier '[' expression ']' '[' expression ']' '{' expression '}' {
			$$=NODE3(T_CIMG_DECL,$3,$6,$9);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| decl_identifier '[' expression ']' '[' expression ']' '[' expression ']' {
			$$=NODE3(T_SEQ_DECL,$3,$6,$9);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| decl_identifier '[' expression ']' '[' expression ']' '[' expression ']' '{' expression '}' {
			Vec_Expr_Node *enp;
			enp = NODE2(T_EXPR_LIST,$9,$12);
			$$=NODE3(T_CSEQ_DECL,$3,$6,enp);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| decl_identifier '{' '}' {
			$$ = NODE1(T_CSCAL_DECL,NO_VEXPR_NODE);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| decl_identifier '[' ']'
			{
			$$ = NODE1(T_VEC_DECL,NO_VEXPR_NODE);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| decl_identifier '[' ']' '{' '}'
			{
			$$ = NODE2(T_CVEC_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| decl_identifier '[' ']' '[' ']'
			{
			$$ = NODE2(T_IMG_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| decl_identifier '[' ']' '[' ']' '{' '}'
			{
			$$ = NODE3(T_CIMG_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE,NO_VEXPR_NODE);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| decl_identifier '[' ']' '[' ']' '[' ']'
			{
			$$ = NODE3(T_SEQ_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE,NO_VEXPR_NODE);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| decl_identifier '[' ']' '[' ']' '[' ']' '{' '}'
			{
			$$ = NODE3(T_CSEQ_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE,NO_VEXPR_NODE);
			SET_VN_DECL_NAME($$,savestr($1));
			}
		| '*' decl_identifier
			{
			$$=NODE0(T_PTR_DECL);
			SET_VN_DECL_NAME($$, savestr($2));
			}
		| DATA_FUNC
			{
			$$=NODE0(T_BADNAME);
			SET_VN_STRING($$, savestr( FUNC_NAME( $1 )) );
			CURDLE($$)
			NODE_ERROR($$);
			WARN("illegal data function name use");
			}
		| SIZE_FUNC
			{
			$$=NODE0(T_BADNAME);
			SET_VN_STRING($$, savestr( FUNC_NAME($1) ) );
			CURDLE($$)
			NODE_ERROR($$);
			WARN("illegal size function name use");
			}
		/*
		| badname
			{
			$$=NODE0(T_BADNAME);
			SET_VN_STRING($$, $1);
			}
		| badname '[' expression ']'
			{
			$$=NODE0(T_BADNAME);
			SET_VN_STRING($$, $1);
			}
		| badname '[' expression ']' '[' expression ']'
			{
			$$=NODE0(T_BADNAME);
			SET_VN_STRING($$, $1);
			}
		| badname '[' expression ']' '[' expression ']' '[' expression ']'
			{
			$$=NODE0(T_BADNAME);
			SET_VN_STRING($$, $1);
			}
		| badname '[' ']'
			{
			$$=NODE0(T_BADNAME);
			SET_VN_STRING($$, $1);
			}
		| badname '[' ']' '[' ']'
			{
			$$=NODE0(T_BADNAME);
			SET_VN_STRING($$, $1);
			}
		| badname '[' ']' '[' ']' '[' ']'
			{
			$$=NODE0(T_BADNAME);
			SET_VN_STRING($$, $1);
			}
		*/
		;

decl_item_list	: decl_item
		| decl_item '=' expression {
			$$=NODE2(T_DECL_INIT,$1,$3);
			}
		| decl_item_list ',' decl_item {
			$$=NODE2(T_DECL_ITEM_LIST,$1,$3); }
		| decl_item_list ',' decl_item '=' expression {
			Vec_Expr_Node *enp;
			enp=NODE2(T_DECL_INIT,$3,$5);
			$$=NODE2(T_DECL_ITEM_LIST,$1,enp); }
		/* | function_prototype */
		/*
		| decl_item_list ',' function_prototype {
			$$ = NODE2(T_DECL_ITEM_LIST,$1,$3); }
			*/
		;

arg_decl	: data_type decl_item {
			$$=NODE1(T_DECL_STAT,$2);
/*
			if( PREC_RDONLY($1) )
				SET_VN_DECL_FLAGS($$, DECL_IS_CONST);
*/
			SET_VN_DECL_PREC($$,$1);
			}
		;

decl_stat_list	: decl_statement
		| decl_stat_list decl_statement
			{
			$$=NODE2(T_DECL_STAT_LIST,$1,$2);
			}
		;


decl_statement	: data_type decl_item_list ';' {
			$$ = NODE1(T_DECL_STAT,$2);
/*
			if( $1 & DT_RDONLY )
				SET_VN_DECL_FLAGS($$, DECL_IS_CONST);
*/
			SET_VN_DECL_PREC($$,$1);
			EVAL_IMMEDIATE($$);
			// don't release here because may be in subrt decl...
			// But we need to release otherwise!?
			}
		| EXTERN data_type decl_item_list ';' {
			$$ = NODE1(T_EXTERN_DECL,$3);
/*
			if( $2 & DT_RDONLY )
				SET_VN_DECL_FLAGS($$, DECL_IS_CONST);
*/
			SET_VN_DECL_PREC($$,$2);
			EVAL_IMMEDIATE($$);
			// don't release here because may be in subrt decl...
			}
		| STATIC data_type decl_item_list ';' {
			$$ = NODE1(T_DECL_STAT,$3);
/*
			if( $2 & DT_RDONLY )
				SET_VN_DECL_FLAGS($$, DECL_IS_CONST);
*/
			SET_VN_DECL_FLAG_BITS($$,DECL_IS_STATIC);
			SET_VN_DECL_PREC($$,$2);
			EVAL_IMMEDIATE($$);
			// don't release here because may be in subrt decl...
			}
		;

loop_stuff	:	statline
		|	blk_stat
		;

loop_statement	: WHILE '(' expression ')' loop_stuff
			{
				if( $5 != NULL )
					$$ = NODE2(T_WHILE,$3,$5);
				else
					$$ = NULL;
			}
		| UNTIL '(' expression ')' loop_stuff
			{
				if( $5 != NULL )
					$$ = NODE2(T_UNTIL,$3,$5);
				else
					$$ = NULL;
			}
		| FOR '(' simple_stat ';' expression ';' simple_stat ')' loop_stuff
			{
			Vec_Expr_Node *loop_enp;

			loop_enp=NODE3(T_FOR,$5,$9,$7);
			if( $3 != NULL ){
				$$ = NODE2(T_STAT_LIST,$3,loop_enp);
			} else {
				$$ = loop_enp;
			}
			}
		| DO loop_stuff WHILE '(' expression ')' ';'
			{
			/* we want to preserve a strict tree structure */
			$$ = NODE2(T_DO_WHILE,$2,$5);
			}
		| DO loop_stuff UNTIL '(' expression ')' ';'
			{
			/* we want to preserve a strict tree structure */
			$$ = NODE2(T_DO_UNTIL,$2,$5);
			}
		;

case_statement	:	case_list stat_list
			{ $$ = NODE2(T_CASE_STAT,$1,$2); }
		;

case_list	:	single_case
		|	case_list single_case
			{ $$ = NODE2(T_CASE_LIST,$1,$2); }
		;

single_case	:	CASE expression ':'
			{ $$ = NODE1(T_CASE,$2); }
		|	DEFAULT ':'
			{ $$ = NODE0(T_DEFAULT); }
		;

switch_cases	:	case_statement
		|	switch_cases case_statement
			{ $$ = NODE2(T_SWITCH_LIST,$1,$2); }
		;

switch_statement	: SWITCH '(' expression ')' '{' switch_cases '}'
			{ $$=NODE2(T_SWITCH,$3,$6); }
		;


if_statement	: IF '(' expression ')' loop_stuff
			{ $$ = NODE3(T_IFTHEN,$3,$5,NO_VEXPR_NODE); }
		| IF '(' expression ')' loop_stuff ELSE loop_stuff
			{ $$ = NODE3(T_IFTHEN,$3,$5,$7); }
		;

/* Simple statements are terminated with a semicolon always */

simple_stat	:	/* null empty statement */

			/* We used to create a null node for this.
			 * But they were never pruned...
			 * Maybe better to return NULL here and not add later?
			 * Perhaps we should print a warning about
			 * a null statement?
			 * Or it could be useful to allow a possibly empty
			 * script var to hold a statement?
			 */
			{ $$ = NULL; }
		| info_stat
		| print_stat
		| misc_stat
		| fileio_stat
		| script_stat
		| return_stat
		| exit_stat
		| assignment
		| ptr_assgn
		| funcptr_assgn
		| str_assgn
		| void_call
		| BREAK { $$=NODE0(T_BREAK); }
		| CONTINUE { $$=NODE0(T_CONTINUE); }
		| GOTO LABELNAME
			{
			$$ = NODE0(T_GO_BACK);
			SET_VN_STRING($$, savestr(ID_NAME($2)));
			}
		| GOTO NEWNAME
			{
			$$ = NODE0(T_GO_FWD);
			SET_VN_STRING($$, savestr($2));
			}
		;

/*
 * block statements can be terminated with a semicolon, or
 * end with a group of statements in braces.
 */

blk_stat	: if_statement
			{ $$ = $1; }
		| loop_statement
			{ $$ = $1; }
		;


/*
 * Data Declarations
 */

comp_stack	: '{' comp_list '}' {
			$$=NODE1(T_COMP_OBJ,$2);
			}
		;

list_obj	: '[' row_list ']' {
			$$=NODE1(T_LIST_OBJ,$2);
			}
		;

comp_list	: expression
		| comp_list ',' expression
			{
			$$=NODE2(T_COMP_LIST,$1,$3);
			}
		;

/* what is the difference between a row_list and an expr_list?
 * An expr_list can appear as the argument to a function...
 */

row_list	: expression
		| row_list ',' expression
			{
			$$=NODE2(T_ROW_LIST,$1,$3);
			}
		;

expr_list	: expression
		| expr_list ',' expression
			{
			$$=NODE2(T_EXPR_LIST,$1,$3);
			}
		;


print_list	: expression
		| print_list ',' expression
			{ $$=NODE2(T_PRINT_LIST,$1,$3); }
		;

mixed_item	: expression
		| pointer
		;

mixed_list	: mixed_item
		| mixed_list ',' mixed_item
			{ $$=NODE2(T_MIXED_LIST,$1,$3); }
		;

string_list	: string_arg
		| string_list ',' string_arg
			{ $$=NODE2(T_STRING_LIST,$1,$3); }
		;

string		: LEX_STRING
			{
			const char *s;
			s=savestr($1);
			$$=NODE0(T_STRING);
			SET_VN_STRING($$, s);
				/* BUG?  make sure to free if tree deleted */
			}
		| NAME_FUNC '(' objref ')'
			{ $$ = NODE1(T_NAME_FUNC,$3); }
		/*
		| pointer
		*/
		;

printable	: expression
		;

/* What is a string_arg? */

string_arg	: string
		| str_ptr
		| str_assgn
		| objref
		| '(' str_assgn ')' { $$ = $2; }
		| '(' print_list ')' { $$ = $2; }
		;

/*
 * Error Handling
 */

/* A badname could be an oldname or a keyword...
 * But we can't put other stuff here easily because keywords have type fundex...
 *
 * Also, oldnames are not necessarily bad since we now allow prototype declarations...
 */

/*

badname		:	oldname
		|	data_type
			{
			yyerror(THIS_QSP,  (char *)"illegal attempt to use a keyword as an identifier");
			$$="<illegal_keyword_use>";
			}
		;

oldname		:	OBJNAME
			{
			sprintf(YY_ERR_STR,"Object %s already declared",
				OBJ_NAME($1));
			yyerror(THIS_QSP,  YY_ERR_STR);
			$$ = OBJ_NAME($1);
			}
		|	STRNAME
			{
			sprintf(YY_ERR_STR,"string %s already declared",
				ID_NAME($1));
			yyerror(THIS_QSP,  YY_ERR_STR);
			$$ = ID_NAME($1);
			}
		|	PTRNAME
			{
			sprintf(YY_ERR_STR,"Pointer %s already declared",
				ID_NAME($1));
			yyerror(THIS_QSP,  YY_ERR_STR);
			$$ = ID_NAME($1);
			}
		;

*/

%%

/* table of keywords */

Keyword kw_tbl[]={
	{	"extern",	EXTERN			},
	{	"static",	STATIC			},
// why do we need to support const?
//	{	"const",	CONST_TYPE		},
	{	"void",		VOID_TYPE		},
	{	"byte",		BYTE			},
	{	"char",		CHAR			},
	{	"string",	STRING			},
	{	"float",	FLOAT			},
	{	"complex",	COMPLEX			},
	{	"dblcpx",	DBLCPX			},
	{	"quaternion",	QUATERNION		},
	{	"dblquat",	DBLQUAT			},
	{	"double",	DOUBLE			},
	{	"short",	SHORT			},
	{	"int",		INT32			},
	{	"int32",	INT32			},
	{	"long",		INT32			},
	{	"llong",	INT64			},
	{	"int64",	INT64			},
	{	"u_byte",	UBYTE			},
	{	"u_short",	USHORT			},
	{	"u_long",	UINT32			},
	{	"u_int",	UINT32			},
	{	"u_int32",	UINT32			},
	{	"u_llong",	UINT64			},
	{	"u_int64",	UINT64			},
	{	"bit",		BIT			},
	{	"color",	COLOR			},

	{	"return",	RETURN			},
	{	"if",		IF			},
	{	"else",		ELSE			},
	{	"while",	WHILE			},
	{	"until",	UNTIL			},
	{	"do",		DO			},
	{	"repeat",	DO			},
	{	"for",		FOR			},
	{	"exit",		EXIT			},
	{	"switch",	SWITCH			},
	{	"case",		CASE			},
	{	"default",	DEFAULT			},
	{	"break",	BREAK			},
	{	"goto",		GOTO			},
	{	"continue",	CONTINUE		},

	{	"equivalence",	EQUIVALENCE		},
	{	"transpose",	TRANSPOSE		},
	{	"conj",		CONJ			},
	{	"max_times",	MAX_TIMES		},
	{	"max_index",	MAX_INDEX		},
	{	"min_index",	MIN_INDEX		},
	{	"dft",		DFT			},
	{	"fft",		DFT			},
	{	"rdft",		RDFT			},
	{	"rfft",		RDFT			},
	{	"idft",		IDFT			},
	{	"ift",		IDFT			},
	{	"ifft",		IDFT			},
	{	"ridft",	RIDFT			},
	{	"rift",		RIDFT			},
	{	"rifft",	RIDFT			},
	{	"wrap",		WRAP			},
	{	"scroll",	SCROLL			},
	{	"dilate",	DILATE			},
	{	"fill",		FILL			},
	{	"clear_opt_params",CLR_OPT_PARAMS	},
	{	"add_opt_param",ADD_OPT_PARAM		},
	{	"optimize",	OPTIMIZE		},
	{	"erode",	ERODE			},
	/*
	{	"render",	RENDER			},
	{	"svd",		SVD			},
	{	"svbk",		SVBK			},
	*/
	{	"enlarge",	ENLARGE			},
	{	"reduce",	REDUCE			},
	{	"warp",		WARP			},
	{	"lookup",	LOOKUP			},
	{	"ramp",		RAMP			},
	{	"sum",		SUM			},
	{	"min",		MINVAL			},
	{	"max",		MAXVAL			},
	{	"Re",		REAL_PART		},
	{	"Im",		IMAG_PART		},

	{	"file_exists",	FILE_EXISTS		},
	{	"obj_name",	NAME_FUNC		},
	{	"strcpy",	STRCPY			},
	{	"strcat",	STRCAT			},

	{	"echo",		ECHO			},
	{	"advise",	ADVISE_FUNC		},
	{	"display",	DISPLAY			},
	{	"warn",		F_WARN			},
	{	"print",	PRINT			},
	{	"info",		INFO			},

	{	"set_output_file",	SET_OUTPUT_FILE	},
	{	"load",		LOAD			},
	{	"read",		LOAD			},
	{	"save",		SAVE			},
	{	"filetype",	FILETYPE		},
	{	"obj_of",	OBJ_OF			},
	{	"fix_size",	FIX_SIZE		},

	{	"end",		END			},
	{	"",		-1			}
};


/*
 *	Lexical analyser.
 */


/* Parse a number.
 * We call this when we encounter a digit...
 * If we see a decimal point, or scientific notation,
 * we set a global var decimal_seen, to indicate that this should
 * be stored as a float...
 * A leading 0 indicates octal, leading 0x indicates hex.
 */

static int decimal_seen;

static double parse_num(QSP_ARG_DECL  const char **strptr)
{
	const char *ptr;
	double place, value=0.0;
	unsigned long intval=0;
	int c;
	int radix;

	decimal_seen=0;

	ptr=(*strptr);

	radix=10;
	if( *ptr=='0' ){
		ptr++;
		if( *ptr == 'x' ){
			radix=16;
			ptr++;
		} else if( isdigit( *ptr ) ){
			radix=8;
		}
	}
	switch(radix){
		case 10:
			while( isdigit(*ptr) ){
				value*=10;
				value+=(*ptr) - '0';
				ptr++;
			}
			break;
		case 8:
			while( isdigit(*ptr)
				&& *ptr!='8'
				&& *ptr!='9' ){

				intval<<=3;
				intval+=(*ptr) - '0';
				ptr++;
			}
			value=intval;
			break;
		case 16:
			while( isdigit(*ptr)
				|| (*ptr>='a'
					&& *ptr<='f')
				|| (*ptr>='A'
					&& *ptr<='F' ) ){

				/* value*=16; */
				intval <<= 4;
				if( isdigit(*ptr) )
					intval+=(*ptr)-'0';
				else if( isupper(*ptr) )
					intval+= 10 + (*ptr)-'A';
				else intval+= 10 + (*ptr)-'a';
				ptr++;
			}
			value=intval;
			break;
		default:
			yyerror(THIS_QSP,  (char *)"bizarre radix in intscan");
	}
	if( radix==10 && *ptr == '.' ){
		decimal_seen=1;
		ptr++;
		place =1.0;
		while( isdigit(*ptr) ){
			c=(*ptr++);
			place /= 10.0;
			value += (c-'0') * place;
		}
	}
	if( radix == 10 && *ptr == 'e' ){	/* read an exponent */
		int ponent=0;
		int sign;

		decimal_seen=1;
		ptr++;

		if( *ptr == '-' ){
			sign= -1;
			ptr++;
		} else if( *ptr == '+' ){
			sign = 1;
			ptr++;
		} else sign=1;

		if( !isdigit(*ptr) )
			yyerror(THIS_QSP,  (char *)"no digits for exponent!?");
		while( *ptr && 
			isdigit( *ptr ) ){
			ponent *= 10;
			ponent += *ptr - '0';
			ptr++;
		}
		if( sign > 0 ){
			while( ponent-- )
				value *= 10;
		} else {
			while( ponent-- )
				value /= 10;
		}
	}
	*strptr = ptr;
	return(value);
}

static int whkeyword(Keyword *table,const char *str)
{
	register int i;
	register Keyword *kwp;

	i=0;
	kwp=table;
	while( kwp->kw_code != -1 ){
		if( !strcmp(str,kwp->kw_token) ) return(i);
		kwp++;
		i++;
	}
	return(-1);
}

// Ideally we would grow this table dynamically as needed!  BUG
#define MAX_SAVED_PARSE_STRINGS	2048
static int n_saved_parse_strings=0;
const char *saved_parse_string[MAX_SAVED_PARSE_STRINGS];

static const char *save_parse_string(char *s)
{
	const char *ss;

	ss = savestr(s);

	// remember s so we can free it later...
	if( n_saved_parse_strings >= MAX_SAVED_PARSE_STRINGS ){
		NERROR1("Need to increase MAX_SAVED_PARSE_STRINGS!?");
	} else {
		saved_parse_string[n_saved_parse_strings++] = ss;
	}

	return ss;
}


/* Read text up to a matching quote, and update the pointer */

static const char *match_quote(QSP_ARG_DECL  const char **spp)
{
	char *s;
	int c;

	s=VEXP_STR;

	while( (c=(**spp)) && c!='"' ){
		*s++ = (char) c;
		(*spp)++; /* YY_CP++; */
	}
	*s=0;
	if( c != '"' ) {
		NWARN("missing quote");
		sprintf(DEFAULT_ERROR_STRING,"string \"%s\" stored",CURR_STRING);
		advise(DEFAULT_ERROR_STRING);
	} else (*spp)++;			/* skip over closing quote */

	s=(char *)save_parse_string(VEXP_STR);
	SET_CURR_STRING(s);
	return(CURR_STRING);
} // match_quote

//#ifdef THREAD_SAFE_QUERY
//int yylex(YYSTYPE *yylvp, Query_Stack *qsp)	/* return the next token */
//#else /* ! THREAD_SAFE_QUERY */
//int yylex(YYSTYPE *yylvp)			/* return the next token */
//#endif /* ! THREAD_SAFE_QUERY */
int yylex(YYSTYPE *yylvp, Query_Stack *qsp)	/* return the next token */
{
	register int c;
	int in_comment=0;
nexttok:
	if( END_SEEN ){
		return(0);
	}

	if( *YY_CP == 0 ){	/* nothing in the buffer? */
		int ql;

		/* BUG since qword doesn't tell us about line breaks,
		 * it is hard to know when to zero the line buffer.
		 * We can try to handle this by watcing q_lineno.
		 */


		lookahead_til(QSP_ARG  EXPR_LEVEL-1);
		if( EXPR_LEVEL > (ql=QLEVEL) ){
			return(0);	/* EOF */
		}

		/* remember the name of the current input file if we don't already have one */
		// CURR_INFILE is part of the query stack struct...
		// while CURRENT_FILENAME is part of the query struct.
		// CURR_INFILE doesn't seem to be used by the interpreter - why
		// was it introduced???  Used here, but needs to be thread-safe,
		// so part of query_stack...
		//
		// Better would be to have a saved string with reference count.
		// 

		if( CURR_INFILE == NULL ){
			CURR_INFILE = save_stringref(CURRENT_FILENAME);
		}

		/* if the name now is different from the remembered name, change it! */
		if( strcmp(CURRENT_FILENAME,SR_STRING(CURR_INFILE)) ){
			/* This is a problem - we don't want to release the old string, because
			 * existing nodes point to it.  We don't want them to have private copies,
			 * either, because there would be too many.  We compromise here by not
			 * releasing the old string, but not remembering it either.  Thus we may
			 * end up with a few more copies later, if we have nested file inclusion...
			 */
			/* rls_str(CURR_INFILE); */
			if( SR_COUNT(CURR_INFILE) == 0 )
				rls_stringref(CURR_INFILE);
			CURR_INFILE = save_stringref(CURRENT_FILENAME);
		}

		/* why disable stripping quotes? */
		disable_stripping_quotes(SINGLE_QSP_ARG);
		/* we copy this now, because script functions may damage
		 * the contents of nameof's buffer.
		 */
		strcpy(yy_word_buf,NAMEOF("statement"));
		YY_CP=yy_word_buf;
/*
sprintf(ERROR_STRING,"read word \"%s\", lineno is now %d, qlevel = %d",
YY_CP,query[ql].q_lineno,qlevel);
advise(ERROR_STRING);
*/

		/* BUG?  lookahead advances lineno?? */
		/* BUG no line numbers in macros? */
		// Should we compare lineno or rdlineno???
		if( QRY_LINENO(QRY_AT_LEVEL(THIS_QSP,ql)) != LASTLINENO ){
/*
sprintf(ERROR_STRING,"line number changed from %d to %d, saving line \"%s\"",
LASTLINENO,query[ql].q_lineno,YY_INPUT_LINE);
advise(ERROR_STRING);
*/
			strcpy(YY_LAST_LINE,YY_INPUT_LINE);
			YY_INPUT_LINE[0]=0;
			SET_PARSER_LINENO( QRY_LINENO( QRY_AT_LEVEL(THIS_QSP,ql) ) );
			SET_LASTLINENO( QRY_LINENO( QRY_AT_LEVEL(THIS_QSP,ql) ) );
		}

		if( (strlen(YY_INPUT_LINE) + strlen(YY_CP) + 2) >= YY_LLEN ){
			WARN("yy_input line buffer overflow");
			sprintf(ERROR_STRING,"%ld chars in input line:",(long)strlen(YY_INPUT_LINE));
			advise(ERROR_STRING);
			advise(YY_INPUT_LINE);
			sprintf(ERROR_STRING,"%ld previously buffered chars:",(long)strlen(YY_CP));
			advise(ERROR_STRING);
			advise(YY_CP);
			return(0);
		}
		strcat(YY_INPUT_LINE,YY_CP);
		strcat(YY_INPUT_LINE," ");

	}

#ifdef QUIP_DEBUG
if( debug & parser_debug ){
sprintf(ERROR_STRING,"yylex scanning \"%s\"",YY_CP);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	/* skip spaces */
	while( *YY_CP
		&& isspace(*YY_CP) ){
		if( *YY_CP == '\n' ){
			YY_CP++;
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning NEWLINE"); }
#endif /* QUIP_DEBUG */
			return(NEWLINE);
		}
		YY_CP++;
	}

	c=(*YY_CP);

	/* check for C comment */

	if( in_comment ){
		if( c=='*' && *(YY_CP+1)=='/' ){
			in_comment=0;
			YY_CP+=2;
			goto nexttok;
		} else if( c=='/' && *(YY_CP+1)=='*' ){
			/* BUG print the line number here */
			WARN("comment within a comment!?");
			YY_CP+=2;
			goto nexttok;
		}
		YY_CP++;
		goto nexttok;
	}

	if( (! in_comment) && c=='/' && *(YY_CP+1)=='*' ){
		in_comment=1;
		YY_CP+=2;
		goto nexttok;
	}

	if( c == 0 || c == '#' ) goto nexttok;

	if( c=='.' && !isdigit(*(YY_CP+1)) ){
		YY_CP++;
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning DOT"); }
#endif /* QUIP_DEBUG */
		return(DOT);
	}
		
	if( isdigit(c) || c=='.' ) {
		const char *s;

		// In objC, we can't take the address of a property...
		s=YY_CP;
		yylvp->dval= parse_num(QSP_ARG  &s) ;
		SET_YY_CP(s);

		if( decimal_seen ){
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning NUMBER"); }
#endif /* QUIP_DEBUG */
			return(NUMBER);
		} else {
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning INT_NUM"); }
#endif /* QUIP_DEBUG */
			return(INT_NUM);
		}
	} else if( c == '"' ){		/* read to matching quote */
		const char *s;
		//SET_YY_CP( YY_CP + 1 );
		// objC can't take the address of a property...
		s=YY_CP+1;
		yylvp->e_string=match_quote(QSP_ARG  &s);
		SET_YY_CP(s);
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning LEX_STRING"); }
#endif /* QUIP_DEBUG */
		return(LEX_STRING);
	} else if( c == '\'' ){		/* char const? */
		if( *(YY_CP+1) != '\\' ){		/* not an escape */
			if( *(YY_CP+2) == '\'' ){
				yylvp->dval = *(YY_CP+1);
				SET_YY_CP( YY_CP + 3);
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning CHAR_CONST"); }
#endif /* QUIP_DEBUG */
				return(CHAR_CONST);
			}
		} else {				/* first char is a backslash */
			if( *(YY_CP+3) == '\'' ){	/* single char escape */
				switch( *(YY_CP+2) ){
					case 'n':  yylvp->dval = '\n'; break;
					case 'r':  yylvp->dval = '\r'; break;
					case 'b':  yylvp->dval = '\b'; break;
					case 'f':  yylvp->dval = '\f'; break;
					default:   yylvp->dval = *(YY_CP+2); break;
				}
				SET_YY_CP( YY_CP + 4 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning CHAR_CONST"); }
#endif /* QUIP_DEBUG */
				return(CHAR_CONST);
			}
		}
	}

/* although the minus sign is allowed in names,
 * allowing it to be the first char creates
 * problems since it is also an operator!
 *
 * Should we make colon ':' a legal character?
 * '.' appears a lot in filenames, but if we want it to be the dotprod
 * operator, then this is dicey...
 */

/* #define islegal(c)	(isalpha(c) || (c)=='.' || (c) == '_' )  */
#define islegal(c)	(isalpha(c) || (c) == '_' )

	else if( islegal(c) ){
		int tok;

		tok=name_token(QSP_ARG yylvp);
		if( tok == END ){
			END_SEEN=1;
		}

		/* when a macro name is encountered, we
		 * read the args, and push the string onto the
		 * input stack
		 */
		if( tok == NEXT_TOKEN ) goto nexttok;
		else {
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ sprintf(ERROR_STRING,"yylex returning token %d",tok); advise(ERROR_STRING); }
#endif /* QUIP_DEBUG */
			return(tok);
		}

	} else if( ispunct(c) ){
		SET_YY_CP( YY_CP + 1 );
		yylvp->fundex=c;
		if( c==';' ){
			SEMI_SEEN=1;
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning semicolon"); }
#endif /* QUIP_DEBUG */
			return(c);
		} else if( c=='>' ){
			if( *YY_CP == '>' ){
				SET_YY_CP( YY_CP + 1 );
				if( *YY_CP == '=' ){
					SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning SHR_EQ"); }
#endif /* QUIP_DEBUG */
					return(SHR_EQ);
				}
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning SHR"); }
#endif /* QUIP_DEBUG */
				return(SHR);
			} else if( *YY_CP == '=' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning GE"); }
#endif /* QUIP_DEBUG */
				return(GE);
			}
		} else if( c=='<' ){
			if( *YY_CP=='<' ){
				SET_YY_CP( YY_CP + 1 );
				if( *YY_CP == '=' ){
					SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning SHL_EQ"); }
#endif /* QUIP_DEBUG */
					return(SHL_EQ);
				}
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning SHL"); }
#endif /* QUIP_DEBUG */
				return(SHL);
			} else if( *YY_CP == '=' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning LE"); }
#endif /* QUIP_DEBUG */
				return(LE);
			}
		} else if( c == '=' ){
			if( *YY_CP == '=' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning LOG_EQ"); }
#endif /* QUIP_DEBUG */
				return(LOG_EQ);
			}
		} else if( c == '|' ){
			if( *YY_CP == '|' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning LOGOR"); }
#endif /* QUIP_DEBUG */
				return(LOGOR);
			} else if( *YY_CP == '=' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning OR_EQ"); }
#endif /* QUIP_DEBUG */
				return(OR_EQ);
			}
		} else if( c == '^' ){
			if( *YY_CP == '^' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning LOGXOR"); }
#endif /* QUIP_DEBUG */
				return(LOGXOR);
			} else if( *YY_CP == '=' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning XOR_EQ"); }
#endif /* QUIP_DEBUG */
				return(XOR_EQ);
			}
		} else if( c == '&' ){
			if( *YY_CP == '&' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning LOGAND"); }
#endif /* QUIP_DEBUG */
				return(LOGAND);
			} else if( *YY_CP == '=' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning AND_EQ"); }
#endif /* QUIP_DEBUG */
				return(AND_EQ);
			}
		} else if( c == '!' ){
			if( *YY_CP == '=' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning NE"); }
#endif /* QUIP_DEBUG */
				return(NE);
			}
		} else if( c == '*' ){
			if( *YY_CP == '=' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning TIMES_EQ"); }
#endif /* QUIP_DEBUG */
				return(TIMES_EQ);
			}
		} else if( c == '+' ){
			if( *YY_CP == '=' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning PLUS_EQ"); }
#endif /* QUIP_DEBUG */
				return(PLUS_EQ);
			} else if( *YY_CP == '+' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning PLUS_PLUS"); }
#endif /* QUIP_DEBUG */
				return(PLUS_PLUS);
			}
		} else if( c == '-' ){
			if( *YY_CP == '=' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning MINUS_EQ"); }
#endif /* QUIP_DEBUG */
				return(MINUS_EQ);
			} else if( *YY_CP == '-' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning MINUS_MINUS"); }
#endif /* QUIP_DEBUG */
				return(MINUS_MINUS);
			}
		} else if( c == '/' ){
			if( *YY_CP == '=' ){
				SET_YY_CP( YY_CP + 1 );
#ifdef QUIP_DEBUG
if( debug & parser_debug ){ advise("yylex returning DIV_EQ"); }
#endif /* QUIP_DEBUG */
				return(DIV_EQ);
			}
		}

#ifdef QUIP_DEBUG
if( debug & parser_debug ){ sprintf(ERROR_STRING,"yylex returning char '%c' (0x%x)",c,c); advise(ERROR_STRING); }
#endif /* QUIP_DEBUG */
		return(c);
	} else {
		WARN("yylex error");
		return(0);
	}
	/* NOTREACHED */
	return(0);
}

static const char *read_word(QSP_ARG_DECL  const char **spp)
{
	char *s;
	int c;

	s=VEXP_STR;

	while( islegal(c=(**spp)) || isdigit(c) ){
		*s++ = (char) c;
		(*spp)++; /* YY_CP++; */
	}
	*s=0;

	return(save_parse_string(VEXP_STR));
}

/* this function should go in the lexical analyser... */

static int name_token(QSP_ARG_DECL  YYSTYPE *yylvp)
{
	int i;
	/* Data_Obj *dp; */
	Identifier *idp;
	Subrt *srp;
	const char *s;
	const char *sptr;
	Function *func_p;

	/*
	 * Currently, function names don't have
	 * any non-alphbetic characters...
	 * BUT data object names can begin with underscore...
	 * since other strings (filenames) might contain
	 * other characters, we may append more stuff
	 * after testing against the function names
	 */

	/* read in a word, possibly expanding macro args */

	// a little kludgy for objC...
	sptr=YY_CP;
	// read_word is saved, must be freed somewhere!?
	s=read_word(QSP_ARG  &sptr);
	SET_YY_CP(sptr);

	SET_CURR_STRING(s);	// call savestr here?
	/* if word was a macro arg, macro arg is now on input stack */
	if( s == NULL ) return(NEXT_TOKEN);

	/* see if it's a keyword */
	i=whkeyword(kw_tbl,CURR_STRING);
	if( i!= (-1) ){
		int code;
		yylvp->fundex=i;
		code=kw_tbl[i].kw_code;
		return(code);
	}
	/* see if it's a system- or user-defined function */

	i=whkeyword(vt_native_func_tbl,CURR_STRING);
	if( i!=(-1) ){
//#ifdef CAUTIOUS
//if( i != vt_native_func_tbl[i].kw_code ){
//sprintf(ERROR_STRING,"CAUTIOUS:  OOPS vt_native_func_tbl[%d].kw_code = %d (expected %d)",i,vt_native_func_tbl[i].kw_code,i);
//ERROR1(ERROR_STRING);
//}
//#endif /* CAUTIOUS */

		// this assertion says that the table is in the correct order,
		// either by initialization or sorting...
		assert( i == vt_native_func_tbl[i].kw_code );

		yylvp->fundex = i;
		return(NATIVE_FUNC_NAME);
	}

	func_p=function_of(QSP_ARG  CURR_STRING);

	if( func_p != NULL ){
		yylvp->func_p = func_p;
		switch(FUNC_TYPE(func_p)){
			case D0_FUNCTYP:	return(MATH0_FUNC);	break;
			case D1_FUNCTYP:	return(MATH1_FUNC);	break;
			case D2_FUNCTYP:	return(MATH2_FUNC);	break;
			case I1_FUNCTYP:	return(INT1_FUNC);	break;
			case STR1_FUNCTYP:	return(STR1_FUNC);	break;
			case STR2_FUNCTYP:	return(STR2_FUNC);	break;
			case STR3_FUNCTYP:	return(STR3_FUNC);	break;
			case STRV_FUNCTYP:	return(STRV_FUNC);	break;
			case CHAR_FUNCTYP:	return(CHAR_FUNC);	break;
			case DOBJ_FUNCTYP:	return(DATA_FUNC);	break;
			case SIZE_FUNCTYP:	return(SIZE_FUNC);	break;
			case TS_FUNCTYP:	return(TS_FUNC);	break;
//#ifdef CAUTIOUS
			default:
//				NERROR1("CAUTIOUS:  name_token:  bad function type!?");
				assert( ! "name_token:  bad function type!?");
				break;
//#endif /* CAUTIOUS */
		}
	}

	/*
	 * doesn't match a reserved word (function name)
	 * See if it is a function or an object name
	 */
	
	srp = subrt_of(QSP_ARG  CURR_STRING);
	if( srp != NO_SUBRT ){
		yylvp->srp = srp;
		if( IS_SCRIPT(srp) )
			return(SCRIPTFUNC);
		else if( IS_REFFUNC(srp) )
			return(REFFUNC);
		else {
			return(FUNCNAME);
		}
	}

	idp = ID_OF( CURR_STRING );
	if( idp != NO_IDENTIFIER ){
		if( IS_STRING_ID(idp) ){
			yylvp->idp = idp;
			return(STRNAME);
		} else if( IS_POINTER(idp) ){
			yylvp->idp = idp;
			return(PTRNAME);
		} else if( IS_FUNCPTR(idp) ){
			yylvp->idp=idp;
			return(FUNCPTRNAME);
		} else if( IS_REFERENCE(idp) ){
			/* the identifier refers to a real object? */
			//yylvp->idp = idp;
			/* can we dereference it now?
			 * Only safe it it is static...
			 */
			if( REF_TYPE(ID_REF(idp)) == OBJ_REFERENCE ){
				yylvp->dp = REF_OBJ(ID_REF(idp));
			} else if( REF_TYPE(ID_REF(idp)) == STR_REFERENCE ){
sprintf(ERROR_STRING,"name_token:  identifier %s refers to a string!?",ID_NAME(idp));
WARN(ERROR_STRING);
				yylvp->dp = (Data_Obj *)REF_SBUF(ID_REF(idp));
			}
			return(OBJNAME);
		} else if( IS_LABEL(idp) ){
			yylvp->idp = idp;
			return(LABELNAME);
		}
//#ifdef CAUTIOUS
		else {
//			WARN("CAUTIOUS:  unhandled identifier type!?");
			assert( ! "unhandled identifier type!?" );
		}
//#endif /* CAUTIOUS */

	} else {
		yylvp->e_string=CURR_STRING;
		return(NEWNAME);
	}
	/* NOTREACHED */
	return(-1);
}

static void release_parse_strings(SINGLE_QSP_ARG_DECL)
{
	int i;

	for(i=0;i<n_saved_parse_strings;i++){
		givbuf((void *)(saved_parse_string[i]));
	}
	n_saved_parse_strings=0;
}

double parse_stuff(SINGLE_QSP_ARG_DECL)		/** parse expression */
{
	int stat;

	FINAL=0.0;
	YY_INPUT_LINE[0]=0;		/* clear record of input string */
	LASTLINENO=(-1);
	YY_CP="";
	SEMI_SEEN=0;
	END_SEEN=0;

	/* disable_lookahead(); */	/* to keep line numbers straight -
				 * but may screw up EOF detection!?
				 */

	TOP_NODE=NO_VEXPR_NODE;

	// we only use the last node for a commented out error dump?
	LAST_NODE=NO_VEXPR_NODE;

	/* The best way to do this would be to pass qsp to yyparse, but since this
	 * routine is generated automatically by bison, we would have to hand-edit
	 * vectree.c each time we run bison...
	 */
	stat=yyparse(THIS_QSP);
	if( TOP_NODE != NO_VEXPR_NODE )	/* successful parsing */
		{
		if( dumpit ) {
			print_shape_key(SINGLE_QSP_ARG);
			DUMP_TREE(TOP_NODE);
		}
		// What are we releasing?  the immediately
		// executed statements?
		check_release(TOP_NODE);
	} else {
		// Do we get here on a syntax error???
		WARN("Unsuccessfully parsed statement (top_node=NULL");
		sprintf(ERROR_STRING,"status = %d\n",stat);	// suppress compiler warning
		advise(ERROR_STRING);
	}

	/* yylex call qline - */

	/* enable_lookahead(); */

	// Here we need to free all of the strings we allocated during the
	// parsing!

	release_parse_strings(SINGLE_QSP_ARG);

	return(FINAL);
} // end parse_stuff

void yyerror(Query_Stack *qsp,  char *s)
{
	const char *filename;
	int ql,n;
	char yyerror_str[YY_LLEN];

	/* get the filename and line number */

fprintf(stderr,"yyerror BEGIN\n");
	filename=CURRENT_FILENAME;
	ql = QLEVEL;
	//n = THIS_QSP->qs_query[ql].q_lineno;
	n = QRY_LINENO( QRY_AT_LEVEL(THIS_QSP,ql) );

	sprintf(yyerror_str,"%s, line %d:  %s",filename,n,s);
	NWARN(yyerror_str);

	sprintf(yyerror_str,"\t%s",YY_INPUT_LINE);
	advise(yyerror_str);
	/* print an arrow at the problem point... */
	n=(int)(strlen(YY_INPUT_LINE)-strlen(YY_CP));
	n-=2;
	if( n < 0 ) n=0;
	strcpy(yyerror_str,"\t");
	while(n--) strcat(yyerror_str," ");
	strcat(yyerror_str,"^");
	NADVISE(yyerror_str);

	/* we might use this to print an arrow at the problem point... */
	/*
	if( *YY_CP ){
		sprintf(yyerror_str,"\"%s\" left in the buffer",YY_CP);
		NADVISE(yyerror_str);
	} else NADVISE("no buffered text");
	*/

	/*
	if( LAST_NODE != NO_VEXPR_NODE ){
		DUMP_TREE(LAST_NODE);
	}
	*/

	FINAL=(-1);
}

#ifdef STANDALONE
double rn_number(double dlimit)
{
	double dret;
	int ilimit, iret;

	ilimit=dlimit;
	iret=rn(ilimit);
	dret=iret;
	return(dret);
}

double dstrcmp(char *s1,char *s2)
{
	double d;
	d=strcmp(s1,s2);
	return(d);
}
#endif /* STANDALONE */

int vecexp_ing=1;

void expr_file(SINGLE_QSP_ARG_DECL)
{
	SET_EXPR_LEVEL(QLEVEL);	/* yylex checks this... */

	parse_stuff(SINGLE_QSP_ARG);

	/* We can break out of this loop
	 * for two reasons; either the file has ended, or we have
	 * encountered and "end" statement
	 *
	 * In the latter case, we may need to pop a dup file
	 * & do some housekeeping
	 */
}

