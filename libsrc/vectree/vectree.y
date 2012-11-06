%{
#include "quip_config.h"

char VersionId_vectree_vectree[] = QUIP_VERSION_STRING;


#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

#include "yacc_hack.h"
#include "savestr.h"		/* not needed? BUG */
#include "data_obj.h"
#include "debug.h"
#include "getbuf.h"
#include "node.h"
#include "function.h"
/* #include "warproto.h" */
#include "query.h"

#include "vectree.h"

/* for definition of function codes */
#include "vecgen.h"

#ifdef SGI
#include <alloca.h>
#endif

#ifdef DEBUG
debug_flag_t parser_debug=0;
#endif /* DEBUG */



#define YY_LLEN 1024

/* local prototypes */

/* extern int yyparse(void); */

static double parse_num(QSP_ARG_DECL  const char **strptr);


#ifdef THREAD_SAFE_QUERY
#define YYPARSE_PARAM qsp	/* gets declared void * instead of Query_Stream * */
/* For yyerror */
#define YY_(msg)	QSP_ARG msg
#endif /* THREAD_SAFE_QUERY */

void yyerror(QSP_ARG_DECL  char *);

static const char *read_word(QSP_ARG_DECL  const char **spp);
static const char *match_quote(QSP_ARG_DECL  const char **spp);

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
	Vec_Func_Code fcode;	/* index to our tables here... */
	int   fundex;		/* function index */
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
} YYSTYPE;

static int name_token(QSP_ARG_DECL  YYSTYPE *yylvp);

#define YYSTYPE_IS_DECLARED			/* necessary on a 2.6 machine?? */

#ifdef THREAD_SAFE_QUERY

int yylex(YYSTYPE *yylvp, Query_Stream *qsp);
#define YYLEX_PARAM SINGLE_QSP_ARG

#else

int yylex(YYSTYPE *yylvp);

#endif


%}

%pure_parser	// make the parser rentrant (thread-safe)

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
%token <dval> VARIABLE
%token <dval> INT_NUM
%token <dval> CHAR_CONST

%token <vfc> MATH0_FUNC
%token <vfc> MATH1_FUNC
%token <vfc> MATH2_FUNC
%token <fundex> DATA_FUNC
%token <fundex> SIZE_FUNC
%token <fundex> MISC_FUNC
%token <fundex> STR1_FUNC
%token <fundex> STR2_FUNC

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
%token <fundex> LONG
%token <fundex> BIT
%token <fundex> UBYTE
%token <fundex> USHORT
%token <fundex> ULONG
%token <fundex> COLOR
%token <fundex> COMPLEX
%token <fundex> DBLCPX

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

%token <fundex> MIN
%token <fundex> MAX
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
%type <enp> expr_stack
%type <enp> comp_stack

%type <enp> decl_item_list

%type <enp> decl_item

%type <enp> expression

%type <intval> data_type
%type <intval> precision
%token <intval> VOID_TYPE
%token <intval> EXTERN
%token <intval> CONST_TYPE
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
			$$->en_string = savestr($1->id_name);
			}
		;

func_ptr	: FUNCPTRNAME
			{
			$$ = NODE0(T_FUNCPTR);
			$$->en_string = savestr($1->id_name);
			}
		;

str_ptr		: STRNAME	/* name of a string object */
			{
			$$=NODE0(T_STR_PTR);
			$$->en_string = savestr($1->id_name);
			}
		;

subsamp_spec	:	expression ':' expression ':' expression
			{
			$$=NODE3(T_RANGE,$1,$3,$5);
			}
		;

objref		: OBJNAME
			{
			if( $1->dt_flags & DT_STATIC ){
				$$=NODE0(T_STATIC_OBJ);
				$$->en_dp = $1;
			} else {
				$$=NODE0(T_DYN_OBJ);
				$$->en_string = savestr($1->dt_name);
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
				sprintf(error_string,"Undefined symbol %s",$1);
				yyerror(QSP_ARG  error_string);
				usp=new_undef(QSP_ARG  $1);
			}
			$$=NODE0(T_UNDEF);
			$$->en_string = savestr($1);
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
			sprintf(error_string,"Need to dereference pointer \"%s\"",$1->en_string);
			yyerror(QSP_ARG  error_string);
			}
			*/
		| string_arg
		| '(' data_type ')' expression %prec UNARY
			{
			$$ = NODE1(T_TYPECAST,$4);
			$$->en_cast_prec=$2;
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
			$$->en_intval = $1;
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
			$$->en_func_index=$1;
			}
		| MATH1_FUNC '(' expression ')' 
			{
			$$=NODE1(T_MATH1_FN,$3);
			$$->en_func_index=$1;
			}
		| MATH2_FUNC '(' expression ',' expression ')' 
			{
			$$=NODE2(T_MATH2_FN,$3,$5);
			$$->en_func_index=$1;
			}
		/* can we have a general object here?? */
		| expression DOT expression {
			$$ = NODE2(T_INNER,$1,$3);
			}
		| NUMBER {
			$$=NODE0(T_LIT_DBL);
			$$->en_dblval=$1;
			}
		| expression '?' expression ':' expression
			{
			/* We determine exactly which type later */
			$$ = NODE3(T_SS_S_CONDASS,$1,$3,$5);
			}
		| CHAR_CONST {
			$$ = NODE0(T_LIT_INT);
			$$->en_intval = $1;
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
			$$->en_func_index=$1;
			}

		| SIZE_FUNC '(' string_arg ')' {
			$$=NODE1(T_SIZE_FN,$3);
			$$->en_func_index=$1;
			}
		| SIZE_FUNC '(' objref ')' {
			$$=NODE1(T_SIZE_FN,$3);
			$$->en_func_index=$1;
			}
		| SIZE_FUNC '(' pointer ')' {
			$$=NODE1(T_SIZE_FN,$3);
			$$->en_func_index=$1;
			NODE_ERROR($$);
			advise("dereference pointer before passing to size function");
			CURDLE($$)
			}
		| SUM '(' pointer ')' {
			sprintf(error_string,"need to dereference pointer %s",$3->en_string);
			yyerror(QSP_ARG  error_string);
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
			$$->en_func_index=$1;
			}
		| STR2_FUNC '(' string_arg ',' string_arg ')'
			{
			$$=NODE2(T_STR2_FN,$3,$5);
			$$->en_func_index=$1;
			}
		/* miscellaneous functions are currframe() and recordable() for omdr */
		| MISC_FUNC '(' ')' {
				$$=NODE0(T_MISC_FN);
				$$->en_func_index=$1;
				}

		| CONJ '(' expression ')'
			{
			$$=NODE1(T_CONJ,$3);
			}

		/* unary minus */
		| '-' expression %prec UNARY {
				$$=NODE1(T_UMINUS,$2);
				}
		| MIN '(' expr_list ')'
			{
			$$=NODE1(T_MINVAL,$3);
			}
		| MAX '(' expr_list ')'
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
			$$->en_call_srp = $1;
			/* make sure this is not a void subroutine! */
			if( $1->sr_prec == PREC_VOID ){
				NODE_ERROR($$);
				sprintf(error_string,"void subroutine %s used in expression!?",$1->sr_name);
				advise(error_string);
				CURDLE($$)
			}
			}
		| comp_stack {
			$$=$1;
			}
		| expr_stack {
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
			$$->en_child_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
			}
		| DFT '(' expression ')' { $$ = NODE1(T_DFT,$3); }
		| IDFT '(' expression ')' { $$ = NODE1(T_IDFT,$3); }
		| RDFT '(' expression ')' {
			$$ = NODE1(T_RDFT,$3);
			$$->en_child_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
			}
		| RIDFT '(' expression ')' {
			$$ = NODE1(T_RIDFT,$3);
			$$->en_child_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
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
			$$->en_child_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
			}
		| REDUCE '(' expression ')' {
			$$=NODE1(T_REDUCE,$3);
			$$->en_child_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
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
			$$->en_srp = $2;
			}
		*/
		| func_ref_arg
		| pointer
		| ptr_assgn
		| '&' pointer
			{
			sprintf(error_string,"shouldn't try to reference pointer variable %s",$2->en_string);
			yyerror(QSP_ARG  error_string);
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
			$$->en_call_srp = $1;
			if( $1->sr_prec != PREC_VOID ){
				NODE_ERROR($$);
				sprintf(error_string,"return value of function %s is ignored",$1->sr_name);
				advise(error_string);
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
				$$->en_decl_prec = $7;
			}
		| REFFUNC '(' func_args ')'
			{
			$$=NODE1(T_CALLFUNC,$3);
			$$->en_call_srp = $1;
			/* make sure this is not a void subroutine! */
			if( $1->sr_prec == PREC_VOID ){
				NODE_ERROR($$);
				sprintf(error_string,"void subroutine %s used in pointer expression!?",$1->sr_name);
				advise(error_string);
				CURDLE($$)
			}
			}
		;

func_ref_arg	: '&' FUNCNAME %prec UNARY
			{
			$$=NODE0(T_FUNCREF);
			$$->en_srp = $2;
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
			idp->id_type = ID_LABEL;
			$$->en_string = savestr(idp->id_name);
			}
		| LABELNAME ':'
			{
			$$ = NODE0(T_LABEL);
			$$->en_string = savestr($1->id_name);
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
			yyerror(QSP_ARG  (char *)"missing '}'");
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
			$$->en_string = savestr($1);
			}
		;

old_func_decl	: FUNCNAME '(' arg_decl_list ')'
			{
			if( $1->sr_flags != SR_PROTOTYPE ){
				sprintf(error_string,"Subroutine %s multiply defined!?",$1->sr_name);
				yyerror(QSP_ARG  error_string);
				/* now what??? */
			}
			set_subrt_ctx(QSP_ARG  $1->sr_name);		/* when do we unset??? */

			/* compare the two arg decl trees
			 * and issue a warning if they do not match.
			 */
			compare_arg_trees(QSP_ARG  $3,$1->sr_arg_decls);

			/* use the new ones */
			$1->sr_arg_decls = $3;
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
			$$->en_string = savestr($1->sr_name);
			}
		;

subroutine	: data_type new_func_decl stat_block
			{
			Subrt *srp;
			srp=remember_subrt(QSP_ARG  $1,$2->en_string,$2->en_child[0],$3);
			srp->sr_prec = $1;
			$$=NODE0(T_SUBRT);
			$$->en_srp=srp;
			delete_subrt_ctx(QSP_ARG  $2->en_string);	/* this deletes the objects... */
			COMPILE_SUBRT(srp);
			}
		| data_type '*' new_func_decl stat_block
			{
			Subrt *srp;
			srp=remember_subrt(QSP_ARG  $1,$3->en_string,$3->en_child[0],$4);
			srp->sr_prec = $1;
			srp->sr_flags |= SR_REFFUNC;
			/* set a flag to show returns ptr */
			$$=NODE0(T_SUBRT);
			$$->en_srp=srp;
			delete_subrt_ctx(QSP_ARG  $3->en_string);	/* this deletes the objects... */
			COMPILE_SUBRT(srp);
			}
		| data_type old_func_decl stat_block
			{
			/* BUG make sure that precision matches prototype decl */
			Subrt *srp;
			srp=subrt_of(QSP_ARG  $2->en_string);
#ifdef CAUTIOUS
			if( srp == NO_SUBRT ) {
				NODE_ERROR($2);
				ERROR1("CAUTIOUS:  missing subrt!?");
			}
#endif /* CAUTIOUS */
			update_subrt(QSP_ARG  srp,$3);
			$$=NODE0(T_SUBRT);
			$$->en_srp=srp;
			delete_subrt_ctx(QSP_ARG  $2->en_string);
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
		| statline
			{
			if( $$ != NO_VEXPR_NODE ) EVAL_IMMEDIATE($$);
			}
		| blk_stat
			{
			if( $$ != NO_VEXPR_NODE ) EVAL_IMMEDIATE($$);
			}
		;

program		: prog_elt END
			{ TOP_NODE=$1;  }
		| prog_elt
			{ TOP_NODE=$1; }
		| program prog_elt END {
			$$=NODE2(T_STAT_LIST,$1,$2);
			TOP_NODE=$$;
			}
		| program prog_elt {
			$$=NODE2(T_STAT_LIST,$1,$2);
			TOP_NODE=$$;
			}
		| error END
			{
			$$ = NO_VEXPR_NODE;
			TOP_NODE=$$;
			}
		;

data_type	: precision
		| CONST_TYPE precision { $$ = $2 | DT_RDONLY ; }
		;

precision	: BYTE { $$		= PREC_BY;	}
		| CHAR { $$		= PREC_CHAR;	}
		| STRING { $$		= PREC_STR;	}
		| FLOAT { $$		= PREC_SP;	}
		| DOUBLE { $$		= PREC_DP;	}
		| COMPLEX { $$		= PREC_CPX;	}
		| DBLCPX { $$		= PREC_DBLCPX;	}
		| SHORT { $$		= PREC_IN;	}
		| LONG { $$		= PREC_DI;	}
		| UBYTE { $$		= PREC_UBY;	}
		| USHORT { $$		= PREC_UIN;	}
		| ULONG { $$		= PREC_UDI;	}
		| BIT { $$		= PREC_BIT;	}
		| COLOR { $$		= PREC_COLOR;	}
		| VOID_TYPE { $$	= PREC_VOID;	}
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
		| RETURN '(' expression ')'
			{
			$$=NODE1(T_RETURN,$3);
			}
		| RETURN '(' ref_arg ')'
			{
			$$=NODE1(T_RETURN,$3);
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
			$$->en_srp = $1;
			}
		| SCRIPTFUNC '(' ')'
			{
			$$=NODE1(T_SCRIPT,NO_VEXPR_NODE);
			$$->en_srp = $1;
			}
		;

str_ptr_arg	: str_ptr
		| NEWNAME
			{
			sprintf(error_string,"undefined string pointer \"%s\"",$1);
			yyerror(QSP_ARG  error_string);
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
			$$->en_intval = $1;
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
			$$->en_srp = $3;
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
			{ $$ = $1->dt_name; }
		| PTRNAME
			{ $$ = $1->id_name; }
		| STRNAME
			{ $$ = $1->id_name; }
		|	precision
			{
			yyerror(QSP_ARG  (char *)"illegal attempt to use a keyword as an identifier");
			$$="<illegal_keyword_use>";
			}
		;

decl_item	: decl_identifier {
			$$ = NODE0(T_SCAL_DECL);
			$$->en_string=savestr($1);	/* bug need to save??? */
			}
		/*
		| decl_identifier '(' arg_decl_list ')' {
			$$ = NODE1(T_PROTO,$3);
			$$->en_string = savestr($1);
			}
		*/
		| new_func_decl
			{
			delete_subrt_ctx(QSP_ARG  $1->en_string);
			}
		| old_func_decl				/* repeated prototype */
			{
			delete_subrt_ctx(QSP_ARG  $1->en_string);
			}
		| '(' '*' decl_identifier ')' '(' arg_decl_list ')'
			{
			/* function pointer */
			$$ = NODE1(T_FUNCPTR_DECL,$6);
			$$->en_decl_name=savestr($3);
			}
		| decl_identifier '{' expression '}' {
			$$ = NODE1(T_CSCAL_DECL,$3);
			$$->en_decl_name=savestr($1);
			}
		| decl_identifier '[' expression ']' {
			$$ = NODE1(T_VEC_DECL,$3);
			$$->en_decl_name=savestr($1);
			}
		| decl_identifier '[' expression ']' '{' expression '}' {
			$$ = NODE2(T_CVEC_DECL,$3,$6);
			$$->en_decl_name=savestr($1);
			}
		| decl_identifier '[' expression ']' '[' expression ']' {
			$$=NODE2(T_IMG_DECL,$3,$6);
			$$->en_decl_name=savestr($1);
			}
		| decl_identifier '[' expression ']' '[' expression ']' '{' expression '}' {
			$$=NODE3(T_CIMG_DECL,$3,$6,$9);
			$$->en_decl_name=savestr($1);
			}
		| decl_identifier '[' expression ']' '[' expression ']' '[' expression ']' {
			$$=NODE3(T_SEQ_DECL,$3,$6,$9);
			$$->en_decl_name=savestr($1);
			}
		| decl_identifier '[' expression ']' '[' expression ']' '[' expression ']' '{' expression '}' {
			Vec_Expr_Node *enp;
			enp = NODE2(T_EXPR_LIST,$9,$12);
			$$=NODE3(T_CSEQ_DECL,$3,$6,enp);
			$$->en_decl_name=savestr($1);
			}
		| decl_identifier '{' '}' {
			$$ = NODE1(T_CSCAL_DECL,NO_VEXPR_NODE);
			$$->en_decl_name=savestr($1);
			}
		| decl_identifier '[' ']'
			{
			$$ = NODE1(T_VEC_DECL,NO_VEXPR_NODE);
			$$->en_decl_name=savestr($1);
			}
		| decl_identifier '[' ']' '{' '}'
			{
			$$ = NODE2(T_CVEC_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE);
			$$->en_decl_name=savestr($1);
			}
		| decl_identifier '[' ']' '[' ']'
			{
			$$ = NODE2(T_IMG_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE);
			$$->en_decl_name=savestr($1);
			}
		| decl_identifier '[' ']' '[' ']' '{' '}'
			{
			$$ = NODE3(T_CIMG_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE,NO_VEXPR_NODE);
			$$->en_decl_name=savestr($1);
			}
		| decl_identifier '[' ']' '[' ']' '[' ']'
			{
			$$ = NODE3(T_SEQ_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE,NO_VEXPR_NODE);
			$$->en_decl_name=savestr($1);
			}
		| decl_identifier '[' ']' '[' ']' '[' ']' '{' '}'
			{
			$$ = NODE3(T_CSEQ_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE,NO_VEXPR_NODE);
			$$->en_decl_name=savestr($1);
			}
		| '*' decl_identifier
			{
			$$=NODE0(T_PTR_DECL);
			$$->en_decl_name  = savestr($2);
			}
		| DATA_FUNC
			{
			$$=NODE0(T_BADNAME);
			$$->en_string = savestr( ((Function *)data_functbl) [$1].fn_name);
			CURDLE($$)
			NODE_ERROR($$);
			WARN("illegal data function name use");
			}
		| SIZE_FUNC
			{
			$$=NODE0(T_BADNAME);
			$$->en_string = savestr( ((Function *)size_functbl) [$1].fn_name);
			CURDLE($$)
			NODE_ERROR($$);
			WARN("illegal size function name use");
			}
		/*
		| badname
			{
			$$=NODE0(T_BADNAME);
			$$->en_string = $1;
			}
		| badname '[' expression ']'
			{
			$$=NODE0(T_BADNAME);
			$$->en_string = $1;
			}
		| badname '[' expression ']' '[' expression ']'
			{
			$$=NODE0(T_BADNAME);
			$$->en_string = $1;
			}
		| badname '[' expression ']' '[' expression ']' '[' expression ']'
			{
			$$=NODE0(T_BADNAME);
			$$->en_string = $1;
			}
		| badname '[' ']'
			{
			$$=NODE0(T_BADNAME);
			$$->en_string = $1;
			}
		| badname '[' ']' '[' ']'
			{
			$$=NODE0(T_BADNAME);
			$$->en_string = $1;
			}
		| badname '[' ']' '[' ']' '[' ']'
			{
			$$=NODE0(T_BADNAME);
			$$->en_string = $1;
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
			if( $1 & DT_RDONLY )
				$$->en_decl_flags = DECL_IS_CONST;
			$$->en_decl_prec=$1 & ~DT_RDONLY;
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
			if( $1 & DT_RDONLY )
				$$->en_decl_flags = DECL_IS_CONST;
			$$->en_decl_prec=$1 & ~DT_RDONLY;
			EVAL_IMMEDIATE($$);
			}
		| EXTERN data_type decl_item_list ';' {
			$$ = NODE1(T_EXTERN_DECL,$3);
			if( $2 & DT_RDONLY )
				$$->en_decl_flags = DECL_IS_CONST;
			$$->en_decl_prec=$2 & ~DT_RDONLY;
			EVAL_IMMEDIATE($$);
			}
		| STATIC data_type decl_item_list ';' {
			$$ = NODE1(T_DECL_STAT,$3);
			if( $2 & DT_RDONLY )
				$$->en_decl_flags = DECL_IS_CONST;
			$$->en_decl_flags |= DECL_IS_STATIC;
			$$->en_decl_prec=$2 & ~DT_RDONLY;
			EVAL_IMMEDIATE($$);
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
			$$->en_string = savestr($2->id_name);
			}
		| GOTO NEWNAME
			{
			$$ = NODE0(T_GO_FWD);
			$$->en_string = savestr($2);
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

expr_stack	: '[' row_list ']' {
			$$=NODE1(T_LIST_OBJ,$2);
			}
		;

comp_list	: expression
		| comp_list ',' expression
			{
			$$=NODE2(T_COMP_LIST,$1,$3);
			}
		;

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
			$$=NODE0(T_STRING);
			$$->en_string = savestr($1);
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
			yyerror(QSP_ARG  (char *)"illegal attempt to use a keyword as an identifier");
			$$="<illegal_keyword_use>";
			}
		;

oldname		:	OBJNAME
			{
			sprintf(error_string,"Object %s already declared",
				$1->dt_name);
			yyerror(QSP_ARG  error_string);
			$$ = $1->dt_name;
			}
		|	STRNAME
			{
			sprintf(error_string,"string %s already declared",
				$1->id_name);
			yyerror(QSP_ARG  error_string);
			$$ = $1->id_name;
			}
		|	PTRNAME
			{
			sprintf(error_string,"Pointer %s already declared",
				$1->id_name);
			yyerror(QSP_ARG  error_string);
			$$ = $1->id_name;
			}
		;

*/

%%

/* table of keywords */

Keyword kw_tbl[]={
	{	"extern",	EXTERN			},
	{	"static",	STATIC			},
	{	"const",	CONST_TYPE		},
	{	"void",		VOID_TYPE		},
	{	"byte",		BYTE			},
	{	"char",		CHAR			},
	{	"string",	STRING			},
	{	"float",	FLOAT			},
	{	"complex",	COMPLEX			},
	{	"dblcpx",	DBLCPX			},
	{	"double",	DOUBLE			},
	{	"short",	SHORT			},
	{	"int",		LONG			},
	{	"long",		LONG			},
	{	"u_byte",	UBYTE			},
	{	"u_short",	USHORT			},
	{	"u_long",	ULONG			},
	{	"u_int",	ULONG			},
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
	{	"min",		MIN			},
	{	"max",		MAX			},
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
			yyerror(QSP_ARG  (char *)"bizarre radix in intscan");
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
			yyerror(QSP_ARG  (char *)"no digits for exponent!?");
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

static char yy_word_buf[YY_LLEN];

#ifdef THREAD_SAFE_QUERY
int yylex(YYSTYPE *yylvp, Query_Stream *qsp)	/* return the next token */
#else /* ! THREAD_SAFE_QUERY */
int yylex(YYSTYPE *yylvp)			/* return the next token */
#endif /* ! THREAD_SAFE_QUERY */
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


		lookahead_til(QSP_ARG  THIS_QSP->qs_expr_level-1);
		if( THIS_QSP->qs_expr_level > (ql=tell_qlevel(SINGLE_QSP_ARG)) ){
			return(0);	/* EOF */
		}

		/* remember the name of the current input file if we don't already have one */
		if( curr_infile == NULL )
			curr_infile = savestr(current_input_file(SINGLE_QSP_ARG));

		/* if the name now is different from the remembered name, change it! */
		if( strcmp(current_input_file(SINGLE_QSP_ARG),curr_infile) ){
			/* This is a problem - we don't want to release the old string, because
			 * existing nodes point to it.  We don't want them to have private copies,
			 * either, because there would be too many.  We compromise here by not
			 * releasing the old string, but not remembering it either.  Thus we may
			 * end up with a few more copies later, if we have nested file inclusion...
			 */
			/* rls_str(curr_infile); */
			curr_infile = savestr(current_input_file(SINGLE_QSP_ARG));
		}

		/* why disable stripping quotes? */
		disable_stripping_quotes(SINGLE_QSP_ARG);
		/* we copy this now, because script functions may damage
		 * the contents of nameof's buffer.
		 */
		strcpy(yy_word_buf,NAMEOF("statement"));
		YY_CP=yy_word_buf;
/*
sprintf(error_string,"read word \"%s\", lineno is now %d, qlevel = %d",
YY_CP,query[ql].q_lineno,qlevel);
advise(error_string);
*/

		/* BUG?  lookahead advances lineno?? */
		/* BUG no line numbers in macros? */
		if( THIS_QSP->qs_query[ql].q_lineno != LASTLINENO ){
/*
sprintf(error_string,"line number changed from %d to %d, saving line \"%s\"",
LASTLINENO,query[ql].q_lineno,YY_INPUT_LINE);
advise(error_string);
*/
			strcpy(YY_LAST_LINE,YY_INPUT_LINE);
			YY_INPUT_LINE[0]=0;
			PARSER_LINENO =
			LASTLINENO = THIS_QSP->qs_query[ql].q_lineno;
		}

		if( (strlen(YY_INPUT_LINE) + strlen(YY_CP) + 2) >= YY_LLEN ){
			WARN("yy_input line buffer overflow");
			sprintf(error_string,"%ld chars in input line:",(long)strlen(YY_INPUT_LINE));
			advise(error_string);
			advise(YY_INPUT_LINE);
			sprintf(error_string,"%ld previously buffered chars:",(long)strlen(YY_CP));
			advise(error_string);
			advise(YY_CP);
			return(0);
		}
		strcat(YY_INPUT_LINE,YY_CP);
		strcat(YY_INPUT_LINE," ");

	}

#ifdef DEBUG
if( debug & parser_debug ){
sprintf(error_string,"yylex scanning \"%s\"",YY_CP);
advise(error_string);
}
#endif /* DEBUG */
	/* skip spaces */
	while( *YY_CP
		&& isspace(*YY_CP) ){
		if( *YY_CP == '\n' ){
			YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning NEWLINE"); }
#endif /* DEBUG */
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
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning DOT"); }
#endif /* DEBUG */
		return(DOT);
	}
		
	if( isdigit(c) || c=='.' ) {
		yylvp->dval= parse_num(QSP_ARG  &YY_CP) ;
		if( decimal_seen ){
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning NUMBER"); }
#endif /* DEBUG */
			return(NUMBER);
		} else {
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning INT_NUM"); }
#endif /* DEBUG */
			return(INT_NUM);
		}
	} else if( c == '"' ){		/* read to matching quote */
		YY_CP++;
		yylvp->e_string=match_quote(QSP_ARG  &YY_CP);
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning LEX_STRING"); }
#endif /* DEBUG */
		return(LEX_STRING);
	} else if( c == '\'' ){		/* char const? */
		if( *(YY_CP+1) != '\\' ){		/* not an escape */
			if( *(YY_CP+2) == '\'' ){
				yylvp->dval = *(YY_CP+1);
				YY_CP += 3;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning CHAR_CONST"); }
#endif /* DEBUG */
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
				YY_CP += 4;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning CHAR_CONST"); }
#endif /* DEBUG */
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
#ifdef DEBUG
if( debug & parser_debug ){ sprintf(error_string,"yylex returning token %d",tok); advise(error_string); }
#endif /* DEBUG */
			return(tok);
		}

	} else if( ispunct(c) ){
		YY_CP++;
		yylvp->fundex=c;
		if( c==';' ){
			SEMI_SEEN=1;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning semicolon"); }
#endif /* DEBUG */
			return(c);
		} else if( c=='>' ){
			if( *YY_CP == '>' ){
				YY_CP++;
				if( *YY_CP == '=' ){
					YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning SHR_EQ"); }
#endif /* DEBUG */
					return(SHR_EQ);
				}
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning SHR"); }
#endif /* DEBUG */
				return(SHR);
			} else if( *YY_CP == '=' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning GE"); }
#endif /* DEBUG */
				return(GE);
			}
		} else if( c=='<' ){
			if( *YY_CP=='<' ){
				YY_CP++;
				if( *YY_CP == '=' ){
					YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning SHL_EQ"); }
#endif /* DEBUG */
					return(SHL_EQ);
				}
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning SHL"); }
#endif /* DEBUG */
				return(SHL);
			} else if( *YY_CP == '=' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning LE"); }
#endif /* DEBUG */
				return(LE);
			}
		} else if( c == '=' ){
			if( *YY_CP == '=' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning LOG_EQ"); }
#endif /* DEBUG */
				return(LOG_EQ);
			}
		} else if( c == '|' ){
			if( *YY_CP == '|' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning LOGOR"); }
#endif /* DEBUG */
				return(LOGOR);
			} else if( *YY_CP == '=' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning OR_EQ"); }
#endif /* DEBUG */
				return(OR_EQ);
			}
		} else if( c == '^' ){
			if( *YY_CP == '^' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning LOGXOR"); }
#endif /* DEBUG */
				return(LOGXOR);
			} else if( *YY_CP == '=' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning XOR_EQ"); }
#endif /* DEBUG */
				return(XOR_EQ);
			}
		} else if( c == '&' ){
			if( *YY_CP == '&' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning LOGAND"); }
#endif /* DEBUG */
				return(LOGAND);
			} else if( *YY_CP == '=' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning AND_EQ"); }
#endif /* DEBUG */
				return(AND_EQ);
			}
		} else if( c == '!' ){
			if( *YY_CP == '=' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning NE"); }
#endif /* DEBUG */
				return(NE);
			}
		} else if( c == '*' ){
			if( *YY_CP == '=' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning TIMES_EQ"); }
#endif /* DEBUG */
				return(TIMES_EQ);
			}
		} else if( c == '+' ){
			if( *YY_CP == '=' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning PLUS_EQ"); }
#endif /* DEBUG */
				return(PLUS_EQ);
			} else if( *YY_CP == '+' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning PLUS_PLUS"); }
#endif /* DEBUG */
				return(PLUS_PLUS);
			}
		} else if( c == '-' ){
			if( *YY_CP == '=' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning MINUS_EQ"); }
#endif /* DEBUG */
				return(MINUS_EQ);
			} else if( *YY_CP == '-' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning MINUS_MINUS"); }
#endif /* DEBUG */
				return(MINUS_MINUS);
			}
		} else if( c == '/' ){
			if( *YY_CP == '=' ){
				YY_CP++;
#ifdef DEBUG
if( debug & parser_debug ){ advise("yylex returning DIV_EQ"); }
#endif /* DEBUG */
				return(DIV_EQ);
			}
		}

#ifdef DEBUG
if( debug & parser_debug ){ sprintf(error_string,"yylex returning char '%c' (0x%x)",c,c); advise(error_string); }
#endif /* DEBUG */
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
		*s++ = c;
		(*spp)++; /* YY_CP++; */
	}
	*s=0;

	return(savestr(VEXP_STR));
}

static const char *match_quote(QSP_ARG_DECL  const char **spp)
{
	char *s;
	int c;

	s=VEXP_STR;

	while( (c=(**spp)) && c!='"' ){
		*s++ = c;
		(*spp)++; /* YY_CP++; */
	}
	*s=0;
	if( c != '"' ) {
		NWARN("missing quote");
		sprintf(DEFAULT_ERROR_STRING,"string \"%s\" stored",CURR_STRING);
		advise(DEFAULT_ERROR_STRING);
	} else (*spp)++;			/* skip over closing quote */

	CURR_STRING=savestr(VEXP_STR);
	return(CURR_STRING);
}

/* this function should go in the lexical analyser... */

static int name_token(QSP_ARG_DECL  YYSTYPE *yylvp)
{
	int i;
	/* Data_Obj *dp; */
	Identifier *idp;
	Subrt *srp;
	const char *s;

	/*
	 * Currently, function names don't have
	 * any non-alphbetic characters...
	 * BUT data object names can begin with underscore...
	 * since other strings (filenames) might contain
	 * other characters, we may append more stuff
	 * after testing against the function names
	 */

	/* read in a word, possibly expanding macro args */

	CURR_STRING=s=read_word(QSP_ARG  &YY_CP);
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
#ifdef CAUTIOUS
if( i != vt_native_func_tbl[i].kw_code ){
sprintf(error_string,"CAUTIOUS:  OOPS vt_native_func_tbl[%d].kw_code = %d (expected %d)",i,vt_native_func_tbl[i].kw_code,i);
ERROR1(error_string);
}
#endif /* CAUTIOUS */
		yylvp->fundex = i;
		return(NATIVE_FUNC_NAME);
	}

	i=whfunc((Function *)math0_functbl,CURR_STRING);
	if( i!= (-1) ){
		yylvp->fundex =i;
		return(MATH0_FUNC);
	}
	i=whfunc((Function *)math1_functbl,CURR_STRING);
	if( i!= (-1) ){
		yylvp->fundex =i;
		return(MATH1_FUNC);
	}
	i=whfunc((Function *)math2_functbl,CURR_STRING);
	if( i!= (-1) ){
		yylvp->fundex =i;
		return(MATH2_FUNC);
	}
	i=whfunc((Function *)data_functbl,CURR_STRING);
	if( i!=(-1) ){
		yylvp->fundex=i;
		return(DATA_FUNC);
	}
	i=whfunc((Function *)size_functbl,CURR_STRING);
	if( i!=(-1) ){
		yylvp->fundex=i;
		return(SIZE_FUNC);
	}
	i=whfunc((Function *)misc_functbl,CURR_STRING);
	if( i!=(-1) ){
		yylvp->fundex=i;
		return(MISC_FUNC);
	}
	i=whfunc((Function *)str1_functbl,CURR_STRING);
	if( i!=(-1) ){
		yylvp->fundex=i;
		return(STR1_FUNC);
	}
	i=whfunc((Function *)str2_functbl,CURR_STRING);
	if( i!=(-1) ){
		yylvp->fundex=i;
		return(STR2_FUNC);
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
			if( idp->id_refp->ref_typ == OBJ_REFERENCE ){
				yylvp->dp = idp->id_refp->ref_u.u_dp;
			} else if( idp->id_refp->ref_typ == STR_REFERENCE ){
sprintf(error_string,"name_token:  identifier %s refers to a string!?",idp->id_name);
WARN(error_string);
				yylvp->dp = (Data_Obj *)idp->id_refp->ref_u.u_sbp;
			}
			return(OBJNAME);
		} else if( IS_LABEL(idp) ){
			yylvp->idp = idp;
			return(LABELNAME);
		}
#ifdef CAUTIOUS
		else {
			WARN("CAUTIOUS:  unhandled identifier type!?");
		}
#endif /* CAUTIOUS */

	} else {
		yylvp->e_string=CURR_STRING;
		return(NEWNAME);
	}
	/* NOTREACHED */
	return(-1);
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
	last_node=NO_VEXPR_NODE;

	/* The best way to do this would be to pass qsp to yyparse, but since this
	 * routine is generated automatically by bison, we would have to hand-edit
	 * vectree.c each time we run bison...
	 */
	stat=yyparse(SINGLE_QSP_ARG);
	if( TOP_NODE != NO_VEXPR_NODE )	/* successful parsing */
		{
		if( dumpit ) {
			print_shape_key();
			DUMP_TREE(TOP_NODE);
		}
	}
	else
		WARN("Unsuccessfully parsed statement (top_node=NULL");

	/* yylex call qline - */

	/* enable_lookahead(); */

	return(FINAL);
}

void yyerror(QSP_ARG_DECL  char *s)
{
	const char *filename;
	int ql,n;
	char yyerror_str[YY_LLEN];

	/* get the filename and line number */

	filename=current_input_file(SINGLE_QSP_ARG);
	ql = tell_qlevel(SINGLE_QSP_ARG);
	n = THIS_QSP->qs_query[ql].q_lineno;

	sprintf(yyerror_str,"%s, line %d:  %s",filename,THIS_QSP->qs_query[ql].q_lineno,s);
	WARN(yyerror_str);

	sprintf(yyerror_str,"\t%s",YY_INPUT_LINE);
	advise(yyerror_str);
	/* print an arrow at the problem point... */
	n=strlen(YY_INPUT_LINE)-strlen(YY_CP);
	n-=2;
	if( n < 0 ) n=0;
	strcpy(yyerror_str,"\t");
	while(n--) strcat(yyerror_str," ");
	strcat(yyerror_str,"^");
	advise(yyerror_str);

	/* we might use this to print an arrow at the problem point... */
	/*
	if( *YY_CP ){
		sprintf(yyerror_str,"\"%s\" left in the buffer",YY_CP);
		advise(yyerror_str);
	} else advise("no buffered text");
	*/

	/*
	if( last_node != NO_VEXPR_NODE ){
		DUMP_TREE(last_node);
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
	THIS_QSP->qs_expr_level=tell_qlevel(SINGLE_QSP_ARG);	/* yylex checks this... */

	parse_stuff(SINGLE_QSP_ARG);

	/* We can break out of this loop
	 * for two reasons; either the file has ended, or we have
	 * encountered and "end" statement
	 *
	 * In the latter case, we may need to pop a dup file
	 * & do some housekeeping
	 */
}

