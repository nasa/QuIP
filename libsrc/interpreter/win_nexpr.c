/* A Bison parser, made by GNU Bison 2.3.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C

   Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin Street, Fifth Floor,
   Boston, MA 02110-1301, USA.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.3"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 1

/* Using locations.  */
#define YYLSP_NEEDED 0



/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     LOGOR = 258,
     LOGAND = 259,
     LOGXOR = 260,
     NE = 261,
     EQUIV = 262,
     LE = 263,
     GE = 264,
     SHR = 265,
     SHL = 266,
     UNARY = 267,
     NUMBER = 268,
     MATH0_FUNC = 269,
     MATH1_FUNC = 270,
     MATH2_FUNC = 271,
     DATA_FUNC = 272,
     SIZE_FUNC = 273,
     TS_FUNC = 274,
     STR_FUNC = 275,
     STR2_FUNC = 276,
     STR3_FUNC = 277,
     E_STRING = 278,
     E_QSTRING = 279
   };
#endif
/* Tokens.  */
#define LOGOR 258
#define LOGAND 259
#define LOGXOR 260
#define NE 261
#define EQUIV 262
#define LE 263
#define GE 264
#define SHR 265
#define SHL 266
#define UNARY 267
#define NUMBER 268
#define MATH0_FUNC 269
#define MATH1_FUNC 270
#define MATH2_FUNC 271
#define DATA_FUNC 272
#define SIZE_FUNC 273
#define TS_FUNC 274
#define STR_FUNC 275
#define STR2_FUNC 276
#define STR3_FUNC 277
#define E_STRING 278
#define E_QSTRING 279




/* Copy the first part of user declarations.  */


#include "quip_config.h"

#include "warn.h"
#include "shape_bits.h"

//static char err_str[LLEN];
static const char *original_string;
#define YY_ORIGINAL	original_string
//#define ERROR_STRING err_str

#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <errno.h>

double rn_number();

#include "quip_prot.h"
#include "query_prot.h"
#include "nexpr.h"

//#include "nexpr_func.h"

//typedef void Function;  // so (Function *) is (void *)

#include "func_helper.h"

static Data_Obj *obj_for_string(const char *string);

#define EVAL_EXPR( s )		eval_expr( QSP_ARG  s )
#define EVAL_SZBL_EXPR( s )	eval_szbl_expr( QSP_ARG  s )

#ifdef QUIP_DEBUG
static debug_flag_t expr_debug=1;
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

static List *free_enp_lp=NO_LIST;

#define MAX_E_STRINGS	4

char _strbuf[MAX_E_STRINGS][128];	/* temporary storage for names */

static Scalar_Expr_Node *alloc_expr_node(void);

/* These have to be put into query stream... */
/*
static Scalar_Expr_Node * final;
*/

static Scalar_Expr_Node * final_expr_node_p;
static const char *yystrptr[MAXEDEPTH];
static int edepth=(-1);
static int which_str=0;
static int in_pexpr=0;
#define FINAL_EXPR_NODE_P	final_expr_node_p
#define YYSTRPTR yystrptr
#define EDEPTH edepth
#define WHICH_EXPR_STR	which_str
#define IN_PEXPR	in_pexpr

//static int n_function_classes=0;
//#define MAX_FUNCTION_CLASSES	10
//static Function_Class func_class[MAX_FUNCTION_CLASSES];

/* local prototypes */

static Item * eval_tsbl_expr(QSP_ARG_DECL   Scalar_Expr_Node *enp );

static void rls_tree(Scalar_Expr_Node *enp);
static void llerror(const char *msg);
#define LLERROR(s)	llerror(s)
static Data_Obj *eval_dobj_expr(QSP_ARG_DECL  Scalar_Expr_Node *);
static Item * eval_szbl_expr(QSP_ARG_DECL  Scalar_Expr_Node *enp);

static double yynumber(SINGLE_QSP_ARG_DECL);
static const char *varval(void);


/* what yylval can be */

typedef union {
	double			fval;		/* actual value */
	int			fundex;		/* function index */
	Function *		func_p;
	char *			e_string;
	Scalar_Expr_Node *	enp;
} YYSTYPE;

#define YYSTYPE_IS_DECLARED		/* needed on 2.6 machine? */



#ifdef THREAD_SAFE_QUERY

#define YYPARSE_PARAM qsp	/* gets declared void * instead of Query_Stack * */
/* For yyerror */
#define YY_(msg)	msg

#define YYLEX_PARAM SINGLE_QSP_ARG
static int yylex(YYSTYPE *yylvp, Query_Stack *qsp);


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
static int yylex(YYSTYPE *yylvp);

#endif /* ! THREAD_SAFE_QUERY */

int yyerror(char *);
/* int yyparse(void); */

static Data_Obj * _def_obj(QSP_ARG_DECL  const char *);
static Data_Obj * _def_sub(QSP_ARG_DECL  Data_Obj * , index_t);

// BUG this is not the real prototype, but yacc files aren't compiled in
// a way which is compatible with objc .h files!?

static Scalar_Expr_Node *node0(Scalar_Expr_Node_Code);
static Scalar_Expr_Node *node1(Scalar_Expr_Node_Code,Scalar_Expr_Node *);
static Scalar_Expr_Node *node2(Scalar_Expr_Node_Code,Scalar_Expr_Node *,Scalar_Expr_Node *);
static Scalar_Expr_Node *node3(Scalar_Expr_Node_Code,Scalar_Expr_Node *,Scalar_Expr_Node *,Scalar_Expr_Node *);

static double eval_expr(QSP_ARG_DECL  Scalar_Expr_Node *);

#define NODE0(code)	node0(code)
#define NODE1(code,enp)	node1(code,enp)
#define NODE2(code,enp1,enp2)	node2(code,enp1,enp2)
#define NODE3(code,enp1,enp2,enp3)	node3(code,enp1,enp2,enp3)

/* globals */
Data_Obj * (*obj_get_func)(QSP_ARG_DECL  const char *)=_def_obj;
Data_Obj * (*exist_func)(QSP_ARG_DECL  const char *)=_def_obj;
Data_Obj * (*sub_func)(QSP_ARG_DECL  Data_Obj *,index_t)=_def_sub;
Data_Obj * (*csub_func)(QSP_ARG_DECL  Data_Obj *,index_t)=_def_sub;



/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif

#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
# define YYSTYPE_IS_TRIVIAL 1
#endif



/* Copy the second part of user declarations.  */


/* Line 216 of yacc.c.  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int i)
#else
static int
YYID (i)
    int i;
#endif
{
  return i;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss;
  YYSTYPE yyvs;
  };

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack)					\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack, Stack, yysize);				\
	Stack = &yyptr->Stack;						\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  37
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   498

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  46
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  6
/* YYNRULES -- Number of rules.  */
#define YYNRULES  46
/* YYNRULES -- Number of states.  */
#define YYNSTATES  118

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   279

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    24,     2,     2,     2,    23,    10,     2,
      42,    43,    21,    19,    44,    20,     2,    22,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     4,     2,
      13,     2,    14,     3,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    38,     2,    39,     9,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    40,     8,    41,    45,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     5,     6,
       7,    11,    12,    15,    16,    17,    18,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint8 yyprhs[] =
{
       0,     0,     3,     5,     7,     9,    11,    13,    18,    23,
      25,    27,    31,    36,    43,    48,    53,    60,    65,    72,
      81,    85,    89,    93,    97,   101,   105,   109,   113,   117,
     121,   125,   129,   132,   135,   138,   144,   148,   151,   155,
     159,   163,   167,   171,   175,   179,   183
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      47,     0,    -1,    51,    -1,    36,    -1,    37,    -1,    36,
      -1,    37,    -1,    49,    38,    51,    39,    -1,    49,    40,
      51,    41,    -1,    48,    -1,    26,    -1,    27,    42,    43,
      -1,    28,    42,    51,    43,    -1,    29,    42,    51,    44,
      51,    43,    -1,    30,    42,    49,    43,    -1,    31,    42,
      49,    43,    -1,    32,    42,    50,    44,    51,    43,    -1,
      33,    42,    48,    43,    -1,    34,    42,    48,    44,    48,
      43,    -1,    35,    42,    48,    44,    48,    44,    51,    43,
      -1,    42,    51,    43,    -1,    51,    19,    51,    -1,    51,
      20,    51,    -1,    51,    22,    51,    -1,    51,    21,    51,
      -1,    51,    23,    51,    -1,    51,    10,    51,    -1,    51,
       8,    51,    -1,    51,     9,    51,    -1,    51,    18,    51,
      -1,    51,    17,    51,    -1,    38,    51,    39,    -1,    45,
      51,    -1,    19,    51,    -1,    20,    51,    -1,    51,     3,
      51,     4,    51,    -1,    51,    12,    51,    -1,    24,    51,
      -1,    51,     5,    51,    -1,    51,     6,    51,    -1,    51,
       7,    51,    -1,    51,    13,    51,    -1,    51,    14,    51,
      -1,    51,    16,    51,    -1,    51,    15,    51,    -1,    51,
      11,    51,    -1,    49,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   260,   260,   266,   267,   270,   274,   289,   291,   310,
     317,   323,   327,   331,   335,   339,   343,   347,   352,   358,
     364,   366,   369,   370,   371,   372,   373,   374,   375,   376,
     377,   378,   379,   380,   381,   382,   384,   385,   386,   387,
     388,   389,   390,   391,   392,   393,   405
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "'?'", "':'", "LOGOR", "LOGAND",
  "LOGXOR", "'|'", "'^'", "'&'", "NE", "EQUIV", "'<'", "'>'", "LE", "GE",
  "SHR", "SHL", "'+'", "'-'", "'*'", "'/'", "'%'", "'!'", "UNARY",
  "NUMBER", "MATH0_FUNC", "MATH1_FUNC", "MATH2_FUNC", "DATA_FUNC",
  "SIZE_FUNC", "TS_FUNC", "STR_FUNC", "STR2_FUNC", "STR3_FUNC", "E_STRING",
  "E_QSTRING", "'['", "']'", "'{'", "'}'", "'('", "')'", "','", "'~'",
  "$accept", "topexp", "e_string", "data_object", "timestampable_object",
  "expression", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,    63,    58,   258,   259,   260,   124,    94,
      38,   261,   262,    60,    62,   263,   264,   265,   266,    43,
      45,    42,    47,    37,    33,   267,   268,   269,   270,   271,
     272,   273,   274,   275,   276,   277,   278,   279,    91,    93,
     123,   125,    40,    41,    44,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    46,    47,    48,    48,    49,    49,    49,    49,    50,
      51,    51,    51,    51,    51,    51,    51,    51,    51,    51,
      51,    51,    51,    51,    51,    51,    51,    51,    51,    51,
      51,    51,    51,    51,    51,    51,    51,    51,    51,    51,
      51,    51,    51,    51,    51,    51,    51
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     4,     4,     1,
       1,     3,     4,     6,     4,     4,     6,     4,     6,     8,
       3,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     2,     2,     2,     5,     3,     2,     3,     3,
       3,     3,     3,     3,     3,     3,     1
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     0,     0,     0,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     5,     6,     0,     0,     0,     0,
      46,     2,    33,    34,    37,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    32,     1,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      11,     0,     0,     0,     0,     3,     4,     9,     0,     0,
       0,     0,    31,    20,     0,     0,     0,    38,    39,    40,
      27,    28,    26,    45,    36,    41,    42,    44,    43,    30,
      29,    21,    22,    24,    23,    25,    12,     0,    14,    15,
       0,    17,     0,     0,     7,     8,     0,     0,     0,     0,
       0,    35,    13,    16,    18,     0,     0,    19
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,    19,    67,    20,    68,    21
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -29
static const yytype_int16 yypact[] =
{
      52,    52,    52,    52,   -29,    28,    31,    35,    50,    56,
      60,    61,    62,    65,   -29,   -29,    52,    52,    52,    22,
      53,   475,   -29,   -29,   -29,    63,    52,    52,   -23,   -23,
      64,    64,    64,    64,   382,   150,   -29,   -29,    52,    52,
      52,    52,    52,    52,    52,    52,    52,    52,    52,    52,
      52,    52,    52,    52,    52,    52,    52,    52,    52,    52,
     -29,   189,   110,   -20,   -19,   -29,   -29,   -29,    51,    66,
      67,    68,   -29,   -29,   417,   345,   454,   168,   206,   244,
     282,   320,   358,    46,    46,   -11,   -11,   -11,   -11,     8,
       8,    11,    11,   -29,   -29,   -29,   -29,    52,   -29,   -29,
      52,   -29,    64,    64,   -29,   -29,    52,   228,   267,   187,
     224,   129,   -29,   -29,   -29,    52,   306,   -29
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -29,   -29,   -28,     7,   -29,    -1
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint8 yytable[] =
{
      22,    23,    24,    69,    70,    71,    53,    54,    55,    56,
      57,    58,    59,    14,    15,    34,    35,    36,    38,    38,
      39,    39,    37,    98,    99,    61,    62,    55,    56,    57,
      58,    59,    57,    58,    59,    63,    64,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    90,    91,    92,    93,    94,    95,    49,
      50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
      25,     1,     2,    26,   109,   110,     3,    27,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    38,    28,    39,    17,   100,   107,    18,    29,   108,
      65,    66,    30,    31,    32,   111,    60,    33,     0,   101,
       0,   102,   103,    40,   116,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    40,    97,    41,    42,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    40,    73,    41,    42,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
     114,    40,    96,    41,    42,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,   115,     0,
      40,   112,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,     0,     0,     0,    40,
     113,    41,    42,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    57,    58,    59,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,     0,     0,     0,     0,    40,   117,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,     0,     0,     0,    40,   105,    41,    42,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      40,    72,    41,    42,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   104,    40,   106,    41,
      42,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    40,     0,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59
};

static const yytype_int8 yycheck[] =
{
       1,     2,     3,    31,    32,    33,    17,    18,    19,    20,
      21,    22,    23,    36,    37,    16,    17,    18,    38,    38,
      40,    40,     0,    43,    43,    26,    27,    19,    20,    21,
      22,    23,    21,    22,    23,    28,    29,    38,    39,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      42,    19,    20,    42,   102,   103,    24,    42,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      38,    38,    42,    40,    42,    44,    97,    45,    42,   100,
      36,    37,    42,    42,    42,   106,    43,    42,    -1,    43,
      -1,    44,    44,     3,   115,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,     3,    44,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,     3,    43,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      43,     3,    43,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    44,    -1,
       3,    43,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    -1,    -1,    -1,     3,
      43,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    -1,    -1,    -1,    -1,     3,    43,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    -1,    -1,    -1,     3,    41,     5,     6,     7,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
       3,    39,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    39,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,     3,    -1,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    19,    20,    24,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    42,    45,    47,
      49,    51,    51,    51,    51,    42,    42,    42,    42,    42,
      42,    42,    42,    42,    51,    51,    51,     0,    38,    40,
       3,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      43,    51,    51,    49,    49,    36,    37,    48,    50,    48,
      48,    48,    39,    43,    51,    51,    51,    51,    51,    51,
      51,    51,    51,    51,    51,    51,    51,    51,    51,    51,
      51,    51,    51,    51,    51,    51,    43,    44,    43,    43,
      44,    43,    44,    44,    39,    41,     4,    51,    51,    48,
      48,    51,    43,    43,    43,    44,    51,    43
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (&yylval, YYLEX_PARAM)
#else
# define YYLEX yylex (&yylval)
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *bottom, yytype_int16 *top)
#else
static void
yy_stack_print (bottom, top)
    yytype_int16 *bottom;
    yytype_int16 *top;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; bottom <= top; ++bottom)
    YYFPRINTF (stderr, " %d", *bottom);
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yyrule)
    YYSTYPE *yyvsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      fprintf (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       		       );
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
#else
static void
yydestruct (yymsg, yytype, yyvaluep)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
#endif
{
  YYUSE (yyvaluep);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}


/* Prevent warnings from -Wmissing-prototypes.  */

#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */






/*----------.
| yyparse.  |
`----------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{
  /* The look-ahead symbol.  */
int yychar;

/* The semantic value of the look-ahead symbol.  */
YYSTYPE yylval;

/* Number of syntax errors so far.  */
int yynerrs;

  int yystate;
  int yyn;
  int yyresult;
  /* Number of tokens to shift before error messages enabled.  */
  int yyerrstatus;
  /* Look-ahead token as an internal (translated) token number.  */
  int yytoken = 0;
#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

  /* Three stacks and their tools:
     `yyss': related to states,
     `yyvs': related to semantic values,
     `yyls': related to locations.

     Refer to the stacks thru separate pointers, to allow yyoverflow
     to reallocate them elsewhere.  */

  /* The state stack.  */
  yytype_int16 yyssa[YYINITDEPTH];
  yytype_int16 *yyss = yyssa;
  yytype_int16 *yyssp;

  /* The semantic value stack.  */
  YYSTYPE yyvsa[YYINITDEPTH];
  YYSTYPE *yyvs = yyvsa;
  YYSTYPE *yyvsp;



#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  YYSIZE_T yystacksize = YYINITDEPTH;

  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;


  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY;		/* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */

  yyssp = yyss;
  yyvsp = yyvs;

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;


	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),

		    &yystacksize);

	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss);
	YYSTACK_RELOCATE (yyvs);

#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;


      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     look-ahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to look-ahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a look-ahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid look-ahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the look-ahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token unless it is eof.  */
  if (yychar != YYEOF)
    yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 2:
    { 
			// qsp is passed to yyparse through YYPARSE_PARAM, but it is void *
			final_expr_node_p = (yyvsp[(1) - (1)].enp) ;
			}
    break;

  case 5:
    {
			(yyval.enp)=NODE0(N_OBJNAME);
			(yyval.enp)->sen_string = savestr((yyvsp[(1) - (1)].e_string));
			}
    break;

  case 6:
    {
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
			(yyval.enp)=NODE0(N_STRING);
			//$$=NODE0(N_OBJNAME);
			(yyval.enp)->sen_string = savestr((yyvsp[(1) - (1)].e_string));
			}
    break;

  case 7:
    {
			(yyval.enp)=NODE2(N_SUBSCRIPT,(yyvsp[(1) - (4)].enp),(yyvsp[(3) - (4)].enp)); }
    break;

  case 8:
    {
			(yyval.enp)=NODE2(N_CSUBSCRIPT,(yyvsp[(1) - (4)].enp),(yyvsp[(3) - (4)].enp)); }
    break;

  case 9:
    {
			(yyval.enp) = NODE0(N_TSABLE);
			(yyval.enp)->sen_string=savestr((yyvsp[(1) - (1)].e_string));
			}
    break;

  case 10:
    {
			(yyval.enp) = NODE0(N_LITDBL);
			(yyval.enp)->sen_dblval = (yyvsp[(1) - (1)].fval);
//sprintf(ERROR_STRING,"LITDBL node set to %28.28lg",$$->sen_dblval);
//ADVISE(ERROR_STRING);
			}
    break;

  case 11:
    {
			(yyval.enp)=NODE0(N_MATH0FUNC);
			(yyval.enp)->sen_func_p = (yyvsp[(1) - (3)].func_p);
			}
    break;

  case 12:
    {
			(yyval.enp)=NODE1(N_MATH1FUNC,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->sen_func_p = (yyvsp[(1) - (4)].func_p);
			}
    break;

  case 13:
    {
			(yyval.enp)=NODE2(N_MATH2FUNC,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			(yyval.enp)->sen_func_p = (yyvsp[(1) - (6)].func_p);
			}
    break;

  case 14:
    {
			(yyval.enp)=NODE1(N_DATAFUNC,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->sen_func_p=(yyvsp[(1) - (4)].func_p);
			}
    break;

  case 15:
    {
			(yyval.enp)=NODE1(N_SIZFUNC,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->sen_func_p=(yyvsp[(1) - (4)].func_p);
			}
    break;

  case 16:
    {
			(yyval.enp)=NODE2(N_TSFUNC,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			(yyval.enp)->sen_func_p=(yyvsp[(1) - (6)].func_p);
			}
    break;

  case 17:
    {
			(yyval.enp)=NODE0(N_STRFUNC);
			(yyval.enp)->sen_string = savestr((yyvsp[(3) - (4)].e_string));
			(yyval.enp)->sen_func_p = (yyvsp[(1) - (4)].func_p);
			}
    break;

  case 18:
    {
			(yyval.enp)=NODE0(N_STR2FUNC);
			(yyval.enp)->sen_string=savestr((yyvsp[(3) - (6)].e_string));
			(yyval.enp)->sen_string2=savestr((yyvsp[(5) - (6)].e_string));
			(yyval.enp)->sen_func_p = (yyvsp[(1) - (6)].func_p);
			}
    break;

  case 19:
    {
			(yyval.enp)=NODE1(N_STR3FUNC,(yyvsp[(7) - (8)].enp));
			(yyval.enp)->sen_string=savestr((yyvsp[(3) - (8)].e_string));
			(yyval.enp)->sen_string2=savestr((yyvsp[(5) - (8)].e_string));
			(yyval.enp)->sen_func_p = (yyvsp[(1) - (8)].func_p);
			}
    break;

  case 20:
    {
			(yyval.enp) = (yyvsp[(2) - (3)].enp) ; }
    break;

  case 21:
    {
			(yyval.enp)=NODE2(N_PLUS,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 22:
    { (yyval.enp) = NODE2(N_MINUS,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 23:
    { (yyval.enp) = NODE2(N_DIVIDE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 24:
    { (yyval.enp) = NODE2(N_TIMES,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 25:
    { (yyval.enp) = NODE2(N_MODULO,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 26:
    { (yyval.enp) = NODE2(N_BITAND,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 27:
    { (yyval.enp) = NODE2(N_BITOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 28:
    { (yyval.enp) = NODE2(N_BITXOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 29:
    { (yyval.enp) = NODE2(N_SHL,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 30:
    { (yyval.enp) = NODE2(N_SHR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 31:
    { (yyval.enp) = (yyvsp[(2) - (3)].enp); }
    break;

  case 32:
    { (yyval.enp) = NODE1(N_BITCOMP,(yyvsp[(2) - (2)].enp)); }
    break;

  case 33:
    { (yyval.enp) = (yyvsp[(2) - (2)].enp); }
    break;

  case 34:
    { (yyval.enp) = NODE1(N_UMINUS,(yyvsp[(2) - (2)].enp)); }
    break;

  case 35:
    { (yyval.enp) = NODE3(N_CONDITIONAL,(yyvsp[(1) - (5)].enp),(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp)); }
    break;

  case 36:
    { (yyval.enp)=NODE2(N_EQUIV,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 37:
    { (yyval.enp) = NODE1(N_NOT,(yyvsp[(2) - (2)].enp)); }
    break;

  case 38:
    { (yyval.enp)=NODE2(N_LOGOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 39:
    { (yyval.enp)=NODE2(N_LOGAND,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 40:
    { (yyval.enp)=NODE2(N_LOGXOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 41:
    { (yyval.enp)=NODE2(N_LT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 42:
    { (yyval.enp)=NODE2(N_GT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 43:
    { (yyval.enp)=NODE2(N_GE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 44:
    { (yyval.enp)=NODE2(N_LE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 45:
    { (yyval.enp)=NODE2(N_NE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 46:
    {
			/* must be a scalar object */
			(yyval.enp)=NODE1(N_SCALAR_OBJ,(yyvsp[(1) - (1)].enp));
			}
    break;


/* Line 1267 of yacc.c.  */
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;


  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse look-ahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse look-ahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;


      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  if (yyn == YYFINAL)
    YYACCEPT;

  *++yyvsp = yylval;


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#ifndef yyoverflow
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}




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
	SET_DIMENSION(dsp,1,strlen(string)+1);
	SET_DIMENSION(dsp,2,1);
	SET_DIMENSION(dsp,3,1);
	SET_DIMENSION(dsp,4,1);
	dp=make_dobj(DEFAULT_QSP_ARG  localname(),dsp,prec_for_code(PREC_STR));
	if( dp != NULL ) strcpy((char *)OBJ_DATA_PTR(dp),string);
	return(dp);
}

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
	enp->sen_dblval=0.0;
	enp->sen_string=NULL;
	enp->sen_string2=NULL;
	return(enp);
}
static Scalar_Expr_Node *node0( Scalar_Expr_Node_Code code )
{
	Scalar_Expr_Node *enp;

//sprintf(ERROR_STRING,"node0( %d )",code);
//ADVISE(ERROR_STRING);
	enp = alloc_expr_node();
	enp->sen_code = code;
	return(enp);
}

static Scalar_Expr_Node *node1( Scalar_Expr_Node_Code code, Scalar_Expr_Node *child )
{
	Scalar_Expr_Node *enp;

//sprintf(ERROR_STRING,"node1( %d )",code);
//ADVISE(ERROR_STRING);
	enp = alloc_expr_node();
	enp->sen_code = code;
	enp->sen_child[0]=child;
	return(enp);
}

static Scalar_Expr_Node *node2( Scalar_Expr_Node_Code code, Scalar_Expr_Node *child1, Scalar_Expr_Node *child2 )
{
	Scalar_Expr_Node *enp;

//sprintf(ERROR_STRING,"node2( %d )",code);
//ADVISE(ERROR_STRING);
	enp = alloc_expr_node();
	enp->sen_code = code;
	enp->sen_child[0]=child1;
	enp->sen_child[1]=child2;
	return(enp);
}

static Scalar_Expr_Node *node3( Scalar_Expr_Node_Code code, Scalar_Expr_Node *child1, Scalar_Expr_Node *child2, Scalar_Expr_Node *child3 )
{
	Scalar_Expr_Node *enp;

//sprintf(ERROR_STRING,"node3( %d )",code);
//ADVISE(ERROR_STRING);
	enp = alloc_expr_node();
	enp->sen_code = code;
	enp->sen_child[0]=child1;
	enp->sen_child[1]=child2;
	enp->sen_child[2]=child3;
	return(enp);
}

/* Evaluate a parsed expression */

static Data_Obj *eval_dobj_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Data_Obj *dp=NULL,*dp2;
	index_t index;

	switch(enp->sen_code){
		case N_STRING:
			/* first try object lookup... */
			/* we don't want a warning if does not exist... */
			dp = (*exist_func)( QSP_ARG  enp->sen_string );
			/* We have a problem here with indexed objects,
			 * since the indexed names aren't in the database...
			 */
			if( dp == NULL ){
				/* treat the string like a rowvec of chars */
				dp = obj_for_string(enp->sen_string);
				return(dp);
			}
			break;

		case N_SCALAR_OBJ:
			dp = eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
			if( IS_SCALAR(dp) ) return(dp);
			return(NULL);
			break;
		case N_OBJNAME:
			dp = (*obj_get_func)( QSP_ARG  enp->sen_string );
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

//#ifdef CAUTIOUS
		default:
//			sprintf(ERROR_STRING,
//		"unexpected case (%d) in eval_dobj_expr",enp->sen_code);
//			NWARN(ERROR_STRING);
			assert( ! "unexpected case in eval_dobj_expr" );
			break;
//#endif /* CAUTIOUS */
	}
	return(dp);
}

static Item* eval_tsbl_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Item *ip=NULL;

	switch(enp->sen_code){
		case N_TSABLE:
			ip = find_tsable( DEFAULT_QSP_ARG  enp->sen_string );
			if( ip == NULL ){
				sprintf(ERROR_STRING,
					"No time-stampable object \"%s\"!?",enp->sen_string);
				NWARN(ERROR_STRING);
				return(NULL);
			}
			break;
//#ifdef CAUTIOUS
		default:
			//NWARN("unexpected case in eval_tsbl_expr");
			assert( ! "unexpected case in eval_tsbl_expr");
			break;
//#endif /* CAUTIOUS */
	}
	return(ip);
}

static Item * eval_szbl_expr( QSP_ARG_DECL  Scalar_Expr_Node *enp )
{
	Item *szp=NULL,*szp2;
	index_t index;

	switch(enp->sen_code){
		case N_STRING:
			szp = find_sizable( DEFAULT_QSP_ARG  enp->sen_string );
			if( szp == NULL ){
				Data_Obj *dp;
				dp = obj_for_string(enp->sen_string);
				szp = (Item *)dp;
			}
			break;

		case N_OBJNAME:
		//case N_SIZABLE:
			szp = find_sizable( DEFAULT_QSP_ARG  enp->sen_string );
			if( szp == NULL ){
				sprintf(ERROR_STRING,
					"No sizable object \"%s\"!?",enp->sen_string);
				NWARN(ERROR_STRING);
				return(NULL);
			}
			break;
		//case N_SUBSIZ:
		case N_SUBSCRIPT:
			szp2=EVAL_SZBL_EXPR(enp->sen_child[0]);
			if( szp2 == NULL )
				return(NULL);
			index = (index_t)EVAL_EXPR(enp->sen_child[1]);
			szp = sub_sizable(DEFAULT_QSP_ARG  szp2,index);
			break;
		//case N_CSUBSIZ:
		case N_CSUBSCRIPT:
			szp2=EVAL_SZBL_EXPR(enp->sen_child[0]);
			if( szp2 == NULL )
				return(NULL);
			index = (index_t)EVAL_EXPR(enp->sen_child[1]);
			szp = csub_sizable(DEFAULT_QSP_ARG  szp2,index);
			break;
//#ifdef CAUTIOUS
		default:
//			sprintf(ERROR_STRING,
//		"unexpected case in eval_szbl_expr %d",enp->sen_code);
//			NWARN(ERROR_STRING);
			assert( ! "unexpected case in eval_szbl_expr" );
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

static void dump_enode(QSP_ARG_DECL  Scalar_Expr_Node *enp)
{
/* Need to do these unary ops:
N_BITCOMP
N_UMINUS
N_EQUIV
N_CONDITIONAL
*/
	switch(enp->sen_code){
		case N_STRING:
			sprintf(ERROR_STRING,"0x%lx\tstring\t%s",
				(int_for_addr)enp, enp->sen_string);
			ADVISE(ERROR_STRING);
			break;

#ifdef FOOBAR
		case N_SIZABLE:
			sprintf(ERROR_STRING,"0x%lx\tsizable\t%s",
				(int_for_addr)enp, enp->sen_string);
			ADVISE(ERROR_STRING);
			break;
#endif /* FOOBAR */

		case N_TSABLE:
			sprintf(ERROR_STRING,"0x%lx\ttsable\t%s",
				(int_for_addr)enp, enp->sen_string);
			ADVISE(ERROR_STRING);
			break;

		case N_SIZFUNC:
			sprintf(ERROR_STRING,"0x%lx\tsizefunc\t%s",
				(int_for_addr)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_TSFUNC:
			sprintf(ERROR_STRING,"0x%lx\tts_func\t%s",
				(int_for_addr)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_MATH0FUNC:
			sprintf(ERROR_STRING,"0x%lx\tmath0_func\t%s",
				(int_for_addr)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_MATH2FUNC:
			sprintf(ERROR_STRING,"0x%lx\tmath2_func\t%s",
				(int_for_addr)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_MISCFUNC:
			sprintf(ERROR_STRING,"0x%lx\tmisc_func\t%s",
				(int_for_addr)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_STR2FUNC:
			sprintf(ERROR_STRING,"0x%lx\tstr2_func\t%s",
				(int_for_addr)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_STR3FUNC:
			sprintf(ERROR_STRING,"0x%lx\tstr3_func\t%s",
				(int_for_addr)enp, FUNC_NAME( enp->sen_func_p ) );
			ADVISE(ERROR_STRING);
			break;

		case N_DATAFUNC:
			sprintf(ERROR_STRING,"0x%lx\tdatafunc\t%s",
				(int_for_addr)enp, FUNC_NAME( enp->sen_func_p ) );
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
			sprintf(ERROR_STRING,"0x%lx\tcsubscript\t0x%lx\t0x%lx",
				(int_for_addr)enp, (int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_MATH1FUNC:
			sprintf(ERROR_STRING,"0x%lx\tmath1func\t%s",
				(int_for_addr)enp, FUNC_NAME(enp->sen_func_p) );
			ADVISE(ERROR_STRING);
			break;

		case N_PLUS:
			sprintf(ERROR_STRING,"0x%lx\tplus\t0x%lx\t0x%lx",
				(int_for_addr)enp,(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_MINUS:
			sprintf(ERROR_STRING,"0x%lx\tminus\t0x%lx\t0x%lx",
				(int_for_addr)enp,(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
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

		case N_MODULO:
			sprintf(ERROR_STRING,"0x%lx\tmodulo\t0x%lx\t0x%lx",
				(int_for_addr)enp,(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_BITAND:
			sprintf(ERROR_STRING,"0x%lx\tbitand\t0x%lx\t0x%lx",
				(int_for_addr)enp,(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_BITOR:
			sprintf(ERROR_STRING,"0x%lx\tbitor\t0x%lx\t0x%lx",
				(int_for_addr)enp,(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_BITXOR:
			sprintf(ERROR_STRING,"0x%lx\tbitxor\t0x%lx\t0x%lx",
				(int_for_addr)enp,(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_SHL:
			sprintf(ERROR_STRING,"0x%lx\tshl\t0x%lx\t0x%lx",
				(int_for_addr)enp,(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_SHR:
			sprintf(ERROR_STRING,"0x%lx\tshr\t0x%lx\t0x%lx",
				(int_for_addr)enp,(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_LOGOR:
			sprintf(ERROR_STRING,"0x%lx\tlog_or\t0x%lx\t0x%lx",
				(int_for_addr)enp,(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_LOGAND:
			sprintf(ERROR_STRING,"0x%lx\tlog_and\t0x%lx\t0x%lx",
				(int_for_addr)enp,(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_LOGXOR:
			sprintf(ERROR_STRING,"0x%lx\tlog_xor\t0x%lx\t0x%lx",
				(int_for_addr)enp,(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;

		case N_LITDBL:
			sprintf(ERROR_STRING,"0x%lx\tlit_dbl\t%g",
				(int_for_addr)enp,enp->sen_dblval);
			ADVISE(ERROR_STRING);
			break;
		case N_LE:
			sprintf(ERROR_STRING,"0x%lx\t<= (LE)\t0x%lx, 0x%lx",(int_for_addr)enp,
				(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
		case N_GE:
			sprintf(ERROR_STRING,"0x%lx\t>= (GE)\t0x%lx, 0x%lx",(int_for_addr)enp,
				(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
		case N_NE:
			sprintf(ERROR_STRING,"0x%lx\t!= (NE)\t0x%lx, 0x%lx",(int_for_addr)enp,
				(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
		case N_LT:
			sprintf(ERROR_STRING,"0x%lx\t< (LT)\t0x%lx, 0x%lx",(int_for_addr)enp,
				(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
		case N_GT:
			sprintf(ERROR_STRING,"0x%lx\t> (GT)\t0x%lx, 0x%lx",(int_for_addr)enp,
				(int_for_addr)enp->sen_child[0],(int_for_addr)enp->sen_child[1]);
			ADVISE(ERROR_STRING);
			break;
		case N_NOT:
			sprintf(ERROR_STRING,"0x%lx\t! (NOT)\t0x%lx",
				(int_for_addr)enp,
				(int_for_addr)enp->sen_child[0]);
			ADVISE(ERROR_STRING);
			break;
		case N_STRFUNC:
			sprintf(ERROR_STRING,"0x%lx\tSTRFUNC %s\t\"%s\"",
				(int_for_addr)enp,
				FUNC_NAME(enp->sen_func_p),
				enp->sen_string);
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

/*static*/ void dump_etree(QSP_ARG_DECL  Scalar_Expr_Node *enp)
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
	static Function *val_func_p=NO_FUNCTION;

#ifdef QUIP_DEBUG
if( debug & expr_debug ){
sprintf(ERROR_STRING,"eval_expr:  code = %d",enp->sen_code);
ADVISE(ERROR_STRING);
dump_enode(QSP_ARG  enp);
}
#endif /* QUIP_DEBUG */

	switch(enp->sen_code){

	case N_MATH0FUNC:
		dval = evalD0Function(enp->sen_func_p);
		break;
	case N_MATH1FUNC:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval = evalD1Function(enp->sen_func_p,dval2);
		break;
	case N_MATH2FUNC:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		dval = evalD2Function(enp->sen_func_p,dval2,dval3);
		break;
	case N_DATAFUNC:
		dp = eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
		if( dp == NULL ){ dval = 0.0; } else {
		dval = (*enp->sen_func_p->fn_u.dobj_func)( QSP_ARG  dp ); }
		break;
	case N_SIZFUNC:
		/* We have problems mixing IOS objects and C structs... */
#ifdef BUILD_FOR_IOS
		if( check_ios_sizfunc(&dval,enp->sen_func_p,enp->sen_child[0]) ){
			return dval;
		}
#endif /* BUILD_FOR_IOS */
		szp = EVAL_SZBL_EXPR(enp->sen_child[0]);
		dval = (*enp->sen_func_p->fn_u.sz_func)( QSP_ARG  szp );
		break;
	case N_TSFUNC:
		szp = eval_tsbl_expr(QSP_ARG  enp->sen_child[0]);
		frm = EVAL_EXPR(enp->sen_child[1]);
		dval = (*enp->sen_func_p->fn_u.ts_func)( QSP_ARG  szp, frm );
		break;
	case N_STRFUNC:
		dval = evalStr1Function(QSP_ARG  enp->sen_func_p,enp->sen_string);
		break;

	case N_STR2FUNC:
		dval = evalStr2Function(enp->sen_func_p,enp->sen_string, enp->sen_string2 );
		break;

	case N_STR3FUNC:
		dval = evalStr3Function( enp->sen_func_p, enp->sen_string, enp->sen_string2, (int) EVAL_EXPR( enp->sen_child[0]) );
		break;

	case N_PLUS:
		dval2=EVAL_EXPR(enp->sen_child[0]);
		dval3=EVAL_EXPR(enp->sen_child[1]);
		dval=dval2+dval3;
		break;

	case N_SCALAR_OBJ:
#ifdef FOOBAR
		/* to get the value of a scalar object, we need the data object library, but we'd like
		 * this to be standalone for cases where we aren't linking with libdata...
		 */
		dp=eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
		if( dp == NULL ) {
			/* This looks like the wrong error message?  Should be obj does not exist??? */
			//WARN("eval_expr:  object is not a scalar");
			WARN("eval_expr:  object not found");
			dval = 0.0;
			break;
		} else {
			Scalar_Value sv;
			extract_scalar_value(&sv,dp);
			dval = cast_from_scalar_value(QSP_ARG  &sv,OBJ_PREC_PTR(dp));
		}
#endif /* FOOBAR */
		/* instead of explicitly extracting the scalar value,
		 * we just call the function from the datafunctbl...
		 * We have looked and know that the value function is the first entry with index 0, but
		 * it is a BUG to have that value hard-coded here...
		 */

		dp = eval_dobj_expr(QSP_ARG  enp->sen_child[0]);
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
//		if( val_func_p == NO_FUNCTION )
//			ERROR1("CAUTIOUS:  couldn't find object value function!?");
//#endif /* CAUTIOUS */
		assert( val_func_p != NO_FUNCTION );

		dval = (*val_func_p->fn_u.dobj_func)( QSP_ARG  dp ); }

		break;

	/* do-nothing */
	case N_OBJNAME:
	case N_SUBSCRIPT:
	case N_CSUBSCRIPT:
#ifdef FOOBAR
	case N_SIZABLE:
	case N_SUBSIZ:
	case N_CSUBSIZ:
	case N_TSABLE:
#endif /* FOOBAR */
		sprintf(ERROR_STRING,
			"unexpected case (%d) in eval_expr",
			enp->sen_code);
		NWARN(ERROR_STRING);
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

//#ifdef CAUTIOUS
	default:
//		sprintf(ERROR_STRING,
//			"CAUTIOUS:  %s - %s:  unhandled node code case %d!?",
//			WHENCE2(eval_expr),
//			enp->sen_code);
//		NWARN(ERROR_STRING);
//		dval=0.0;	// quiet compiler
		assert( ! "unhandled node code" );
		break;
//#endif /* CAUTIOUS */

	}

	return(dval);
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

	status = extract_number_string(buf,128,&ptr);
	*strptr = ptr;
	if( status < 0 ){
		sprintf(DEFAULT_ERROR_STRING,"parse_number:  bad number string \"%s\"",
			ptr);
		NWARN(DEFAULT_ERROR_STRING);
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
#ifdef HAVE_STRTOLL
			ll1=strtoll(buf,&endptr,0);
#else // ! HAVE_STRTOLL
			ll1=strtol(buf,&endptr,0);
			// BUG?  should we print a warning??
#endif // ! HAVE_STRTOLL
			if( errno != 0 ){
				sprintf(DEFAULT_ERROR_STRING,"long long conversion error!?  (errno=%d)",errno);
				NWARN(DEFAULT_ERROR_STRING);
				tell_sys_error("strtoll");
			}
			if( ll1 > 0 && ll1 <=0xffffffff ){	/* fits in an unsigned long */
				unsigned long ul;
				ul = (unsigned long)ll1;
				return(ul);
			} else {
				sprintf(DEFAULT_ERROR_STRING,"long conversion error!?  (errno=%d)",errno);
				NWARN(DEFAULT_ERROR_STRING);
				tell_sys_error("strtol");
				errno=0;
			}
		}
		if( errno != 0 ){
			sprintf(DEFAULT_ERROR_STRING,"long conversion error!?  (errno=%d)",errno);
			NWARN(DEFAULT_ERROR_STRING);
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
sprintf(DEFAULT_ERROR_STRING,"strtod:  range error buf=\"%s\", d = %g",buf,d);
ADVISE(DEFAULT_ERROR_STRING);
		} else if( errno != 0 ){
			sprintf(DEFAULT_ERROR_STRING,"double conversion error!?  (errno=%d)",errno);
			NWARN(DEFAULT_ERROR_STRING);
			tell_sys_error("strtod");
		}

//sprintf(DEFAULT_ERROR_STRING,"flt conversion returning %lg",d);
//ADVISE(DEFAULT_ERROR_STRING);
		return(d);
	/* } */

} /* end parse_number() */

static double yynumber(SINGLE_QSP_ARG_DECL)
{
//sprintf(ERROR_STRING,"yynumber calling parse_number %s",YYSTRPTR[EDEPTH]);
//ADVISE(ERROR_STRING);
	return( parse_number(DEFAULT_QSP_ARG  (const char **)&YYSTRPTR[EDEPTH]) );
}

static const char * varval()
{
	char tmpbuf[128];
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
		sp=tmpbuf;
		c=(*YYSTRPTR[EDEPTH]);
		while( isalpha(c) || c == '_' || isdigit(c) ){
			*sp++ = (char) c;
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
		case STR1_FUNCTYP:	return(STR_FUNC);	break;
		case STR2_FUNCTYP:	return(STR2_FUNC);	break;
		case STR3_FUNCTYP:	return(STR3_FUNC);	break;
		case SIZE_FUNCTYP:	return(SIZE_FUNC);	break;
		case DOBJ_FUNCTYP:	return(DATA_FUNC);	break;
		case TS_FUNCTYP:	return(TS_FUNC);	break;
//#ifdef CAUTIOUS
		default:
//			NERROR1("CAUTIOUS:  token_for_func_type:  bad type!?");
			assert( ! "token_for_func_type:  bad type!?");

			break;
//#endif /* CAUTIOUS */
	}
	return(-1);
}


#ifdef THREAD_SAFE_QUERY
static int yylex(YYSTYPE *yylvp, Query_Stack *qsp)	/* return the next token */
#else /* ! THREAD_SAFE_QUERY */
static int yylex(YYSTYPE *yylvp)			/* return the next token */
#endif /* ! THREAD_SAFE_QUERY */
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
			yylvp->fval=yynumber(SINGLE_QSP_ARG);
#ifdef QUIP_DEBUG
//if( debug & expr_debug ){
////sprintf(ERROR_STRING,"yylex:  NUMBER = %g",yylvp->fval);
//sprintf(ERROR_STRING,"yylex:  NUMBER = XXX");
////ADVISE(ERROR_STRING);
//}
#endif /* QUIP_DEBUG */

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
			s=_strbuf[WHICH_EXPR_STR];
			*s++ = (*YYSTRPTR[EDEPTH]++);
			while( IS_LEGAL_NAME_CHAR(*YYSTRPTR[EDEPTH]) ){
				*s++ = (*YYSTRPTR[EDEPTH]++);
			}
			*s=0;

			yylvp->func_p = function_of(QSP_ARG  _strbuf[WHICH_EXPR_STR]);
			if( yylvp->func_p != NULL ){
				return( token_for_func_type(FUNC_TYPE(yylvp->func_p)) );
			}

			yylvp->e_string=_strbuf[WHICH_EXPR_STR];
			WHICH_EXPR_STR++;
			WHICH_EXPR_STR %= MAX_E_STRINGS;
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
	if( enp->sen_string != NULL )
		rls_str(enp->sen_string);
	if( enp->sen_string2 != NULL )
		rls_str(enp->sen_string2);

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

#ifdef FOOBAR
#ifdef QUIP_DEBUG
	if( expr_debug == 0 )
		expr_debug = add_debug_module("expressions");
#endif /* QUIP_DEBUG */
#endif /* FOOBAR */

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

#ifdef QUIP_DEBUG
if( debug & expr_debug ){
dump_etree(QSP_ARG  FINAL_EXPR_NODE_P);
}
#endif /* QUIP_DEBUG */

	dval = EVAL_EXPR(FINAL_EXPR_NODE_P);
#ifdef QUIP_DEBUG
if( debug & expr_debug ){
sprintf(ERROR_STRING,"pexpr:  s=\"%s\", dval = %g",buf,dval);
ADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

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

// We can't add a qsp arg here, because this one is defined by yacc...

int yyerror(char *s)
{
	// BUG - this is wrong if we have multiple
	// interpreter threads
	if( IS_HALTING(DEFAULT_QSP) )
		goto cleanup;
	
	sprintf(DEFAULT_ERROR_STRING,"YYERROR:  %s",s);
	NWARN(DEFAULT_ERROR_STRING);

	sprintf(DEFAULT_ERROR_STRING,"parsing \"%s\"",YY_ORIGINAL);
	NADVISE(DEFAULT_ERROR_STRING);

	if( *YYSTRPTR[0] ){
		sprintf(DEFAULT_ERROR_STRING,"\"%s\" left to parse",YYSTRPTR[0]);
		NADVISE(DEFAULT_ERROR_STRING);
	} else {
		NADVISE("No buffered text left to parse");
	}

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


