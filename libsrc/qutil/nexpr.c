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
     MISC_FUNC = 275,
     STR_FUNC = 276,
     STR2_FUNC = 277,
     STR3_FUNC = 278,
     E_STRING = 279,
     E_QSTRING = 280
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
#define MISC_FUNC 275
#define STR_FUNC 276
#define STR2_FUNC 277
#define STR3_FUNC 278
#define E_STRING 279
#define E_QSTRING 280




/* Copy the first part of user declarations.  */
#line 1 "nexpr.y"

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
#line 348 "nexpr.c"

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
# if YYENABLE_NLS
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
#define YYFINAL  39
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   493

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  47
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  6
/* YYNRULES -- Number of rules.  */
#define YYNRULES  47
/* YYNRULES -- Number of states.  */
#define YYNSTATES  121

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   280

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    24,     2,     2,     2,    23,    10,     2,
      43,    44,    21,    19,    45,    20,     2,    22,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     4,     2,
      13,     2,    14,     3,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    39,     2,    40,     9,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    41,     8,    42,    46,     2,     2,     2,
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
      28,    29,    30,    31,    32,    33,    34,    35,    36,    37,
      38
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint8 yyprhs[] =
{
       0,     0,     3,     5,     7,     9,    11,    13,    18,    23,
      25,    27,    31,    36,    43,    48,    53,    60,    64,    69,
      76,    85,    89,    93,    97,   101,   105,   109,   113,   117,
     121,   125,   129,   133,   136,   139,   142,   148,   152,   155,
     159,   163,   167,   171,   175,   179,   183,   187
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      48,     0,    -1,    52,    -1,    37,    -1,    38,    -1,    37,
      -1,    38,    -1,    50,    39,    52,    40,    -1,    50,    41,
      52,    42,    -1,    37,    -1,    26,    -1,    27,    43,    44,
      -1,    28,    43,    52,    44,    -1,    29,    43,    52,    45,
      52,    44,    -1,    30,    43,    50,    44,    -1,    31,    43,
      50,    44,    -1,    32,    43,    51,    45,    52,    44,    -1,
      33,    43,    44,    -1,    34,    43,    49,    44,    -1,    35,
      43,    49,    45,    49,    44,    -1,    36,    43,    49,    45,
      49,    45,    52,    44,    -1,    43,    52,    44,    -1,    52,
      19,    52,    -1,    52,    20,    52,    -1,    52,    22,    52,
      -1,    52,    21,    52,    -1,    52,    23,    52,    -1,    52,
      10,    52,    -1,    52,     8,    52,    -1,    52,     9,    52,
      -1,    52,    18,    52,    -1,    52,    17,    52,    -1,    39,
      52,    40,    -1,    46,    52,    -1,    19,    52,    -1,    20,
      52,    -1,    52,     3,    52,     4,    52,    -1,    52,    12,
      52,    -1,    24,    52,    -1,    52,     5,    52,    -1,    52,
       6,    52,    -1,    52,     7,    52,    -1,    52,    13,    52,
      -1,    52,    14,    52,    -1,    52,    16,    52,    -1,    52,
      15,    52,    -1,    52,    11,    52,    -1,    50,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   243,   243,   249,   250,   253,   257,   272,   274,   278,
     285,   291,   295,   299,   303,   307,   311,   315,   319,   324,
     330,   336,   338,   341,   342,   343,   344,   345,   346,   347,
     348,   349,   350,   351,   352,   353,   354,   356,   357,   358,
     359,   360,   361,   362,   363,   364,   365,   377
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
  "SIZE_FUNC", "TS_FUNC", "MISC_FUNC", "STR_FUNC", "STR2_FUNC",
  "STR3_FUNC", "E_STRING", "E_QSTRING", "'['", "']'", "'{'", "'}'", "'('",
  "')'", "','", "'~'", "$accept", "topexp", "e_string", "data_object",
  "tsable", "expression", 0
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
     272,   273,   274,   275,   276,   277,   278,   279,   280,    91,
      93,   123,   125,    40,    41,    44,   126
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    47,    48,    49,    49,    50,    50,    50,    50,    51,
      52,    52,    52,    52,    52,    52,    52,    52,    52,    52,
      52,    52,    52,    52,    52,    52,    52,    52,    52,    52,
      52,    52,    52,    52,    52,    52,    52,    52,    52,    52,
      52,    52,    52,    52,    52,    52,    52,    52
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     1,     1,     4,     4,     1,
       1,     3,     4,     6,     4,     4,     6,     3,     4,     6,
       8,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     2,     2,     2,     5,     3,     2,     3,
       3,     3,     3,     3,     3,     3,     3,     1
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     0,     0,     0,    10,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     5,     6,     0,     0,     0,
       0,    47,     2,    34,    35,    38,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    33,     1,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    11,     0,     0,     0,     0,     9,     0,    17,
       3,     4,     0,     0,     0,    32,    21,     0,     0,     0,
      39,    40,    41,    28,    29,    27,    46,    37,    42,    43,
      45,    44,    31,    30,    22,    23,    25,    24,    26,    12,
       0,    14,    15,     0,    18,     0,     0,     7,     8,     0,
       0,     0,     0,     0,    36,    13,    16,    19,     0,     0,
      20
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,    20,    72,    21,    68,    22
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -32
static const yytype_int16 yypact[] =
{
     405,   405,   405,   405,   -32,    -5,    34,    54,    58,    70,
      77,   155,   194,   195,   233,   -32,   -32,   405,   405,   405,
      32,    59,   470,   -32,   -32,   -32,   234,   405,   405,    72,
      72,    79,   271,    74,    74,    74,   354,   116,   -32,   -32,
     405,   405,   405,   405,   405,   405,   405,   405,   405,   405,
     405,   405,   405,   405,   405,   405,   405,   405,   405,   405,
     405,   405,   -32,   156,    73,   -11,   -10,   -32,   232,   -32,
     -32,   -32,   272,   273,   309,   -32,   -32,   390,   316,   449,
     174,   213,   252,   291,   330,    50,    -8,    -8,     2,     2,
       2,     2,    84,    84,    14,    14,   -32,   -32,   -32,   -32,
     405,   -32,   -32,   405,   -32,    74,    74,   -32,   -32,   405,
     196,   236,   311,   333,   135,   -32,   -32,   -32,   405,   276,
     -32
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -32,   -32,   -31,    85,   -32,    -1
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint8 yytable[] =
{
      23,    24,    25,    73,    74,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    36,    37,    38,    55,
      56,    57,    58,    59,    60,    61,    63,    64,    40,    40,
      41,    41,    39,   101,   102,    59,    60,    61,    26,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    89,    90,    91,    92,    93,    94,    95,    96,    97,
      98,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,   112,   113,    42,    27,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    28,    40,   110,
      41,    29,   111,    57,    58,    59,    60,    61,   114,    15,
      16,    70,    71,    30,    65,    66,    67,   119,   100,    42,
      31,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    42,
      76,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,    61,    32,    42,
      99,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    33,    34,    42,
     115,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    57,    58,    59,    60,    61,    35,   103,    62,    42,
     116,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      57,    58,    59,    60,    61,    69,   104,     0,   105,    42,
     120,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    57,    58,    59,    60,    61,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,   106,   117,     0,    42,   108,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    57,    58,    59,    60,    61,   118,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    42,    75,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     1,     2,     0,     0,     0,     3,
     107,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      13,    14,    15,    16,    17,     0,     0,     0,    18,     0,
       0,    19,    42,   109,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    61,    42,     0,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    57,
      58,    59,    60,    61
};

static const yytype_int8 yycheck[] =
{
       1,     2,     3,    34,    35,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    17,    18,    19,    17,
      18,    19,    20,    21,    22,    23,    27,    28,    39,    39,
      41,    41,     0,    44,    44,    21,    22,    23,    43,    40,
      41,    42,    43,    44,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    55,    56,    57,    58,    59,    60,
      61,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,   105,   106,     3,    43,     5,     6,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    43,    39,   100,
      41,    43,   103,    19,    20,    21,    22,    23,   109,    37,
      38,    37,    38,    43,    29,    30,    37,   118,    45,     3,
      43,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,     3,
      44,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    43,     3,
      44,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
       7,     8,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    43,    43,     3,
      44,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
       8,     9,    10,    11,    12,    13,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    43,    45,    44,     3,
      44,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    44,    44,    -1,    45,     3,
      44,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    45,    44,    -1,     3,    42,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    45,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,     3,    40,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    19,    20,    -1,    -1,    -1,    24,
      40,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    -1,    -1,    -1,    43,    -1,
      -1,    46,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,     3,    -1,     5,     6,     7,     8,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    19,    20,    24,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    37,    38,    39,    43,    46,
      48,    50,    52,    52,    52,    52,    43,    43,    43,    43,
      43,    43,    43,    43,    43,    43,    52,    52,    52,     0,
      39,    41,     3,     5,     6,     7,     8,     9,    10,    11,
      12,    13,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    44,    52,    52,    50,    50,    37,    51,    44,
      37,    38,    49,    49,    49,    40,    44,    52,    52,    52,
      52,    52,    52,    52,    52,    52,    52,    52,    52,    52,
      52,    52,    52,    52,    52,    52,    52,    52,    52,    44,
      45,    44,    44,    45,    44,    45,    45,    40,    42,     4,
      52,    52,    49,    49,    52,    44,    44,    44,    45,    52,
      44
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
# if YYLTYPE_IS_TRIVIAL
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
#line 243 "nexpr.y"
    { 
			// qsp is passed to yyparse through YYPARSE_PARAM, but it is void *
			YY_QSP->qs_final_expr_node_p = (yyvsp[(1) - (1)].enp) ;
			}
    break;

  case 5:
#line 253 "nexpr.y"
    {
			(yyval.enp)=NODE0(N_OBJNAME);
			(yyval.enp)->sen_string = savestr((yyvsp[(1) - (1)].e_string));
			}
    break;

  case 6:
#line 257 "nexpr.y"
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
#line 272 "nexpr.y"
    {
			(yyval.enp)=NODE2(N_SUBSCRIPT,(yyvsp[(1) - (4)].enp),(yyvsp[(3) - (4)].enp)); }
    break;

  case 8:
#line 274 "nexpr.y"
    {
			(yyval.enp)=NODE2(N_CSUBSCRIPT,(yyvsp[(1) - (4)].enp),(yyvsp[(3) - (4)].enp)); }
    break;

  case 9:
#line 278 "nexpr.y"
    {
			(yyval.enp) = NODE0(N_TSABLE);
			(yyval.enp)->sen_string=savestr((yyvsp[(1) - (1)].e_string));
			}
    break;

  case 10:
#line 285 "nexpr.y"
    {
			(yyval.enp) = NODE0(N_LITDBL);
			(yyval.enp)->sen_dblval = (yyvsp[(1) - (1)].fval);
//sprintf(ERROR_STRING,"LITDBL node set to %28.28lg",$$->sen_dblval);
//ADVISE(ERROR_STRING);
			}
    break;

  case 11:
#line 291 "nexpr.y"
    {
			(yyval.enp)=NODE0(N_MATH0FUNC);
			(yyval.enp)->sen_index = (yyvsp[(1) - (3)].fundex);
			}
    break;

  case 12:
#line 295 "nexpr.y"
    {
			(yyval.enp)=NODE1(N_MATH1FUNC,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->sen_index = (yyvsp[(1) - (4)].fundex);
			}
    break;

  case 13:
#line 299 "nexpr.y"
    {
			(yyval.enp)=NODE2(N_MATH2FUNC,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			(yyval.enp)->sen_index = (yyvsp[(1) - (6)].fundex);
			}
    break;

  case 14:
#line 303 "nexpr.y"
    {
			(yyval.enp)=NODE1(N_DATAFUNC,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->sen_index=(yyvsp[(1) - (4)].fundex);
			}
    break;

  case 15:
#line 307 "nexpr.y"
    {
			(yyval.enp)=NODE1(N_SIZFUNC,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->sen_index=(yyvsp[(1) - (4)].fundex);
			}
    break;

  case 16:
#line 311 "nexpr.y"
    {
			(yyval.enp)=NODE2(N_TSFUNC,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			(yyval.enp)->sen_index=(yyvsp[(1) - (6)].fundex);
			}
    break;

  case 17:
#line 315 "nexpr.y"
    {
			(yyval.enp)=NODE0(N_MISCFUNC);
			(yyval.enp)->sen_index=(yyvsp[(1) - (3)].fundex);
			}
    break;

  case 18:
#line 319 "nexpr.y"
    {
			(yyval.enp)=NODE0(N_STRFUNC);
			(yyval.enp)->sen_string = savestr((yyvsp[(3) - (4)].e_string));
			(yyval.enp)->sen_index = (yyvsp[(1) - (4)].fundex);
			}
    break;

  case 19:
#line 324 "nexpr.y"
    {
			(yyval.enp)=NODE0(N_STR2FUNC);
			(yyval.enp)->sen_string=savestr((yyvsp[(3) - (6)].e_string));
			(yyval.enp)->sen_string2=savestr((yyvsp[(5) - (6)].e_string));
			(yyval.enp)->sen_index = (yyvsp[(1) - (6)].fundex);
			}
    break;

  case 20:
#line 330 "nexpr.y"
    {
			(yyval.enp)=NODE1(N_STR3FUNC,(yyvsp[(7) - (8)].enp));
			(yyval.enp)->sen_string=savestr((yyvsp[(3) - (8)].e_string));
			(yyval.enp)->sen_string2=savestr((yyvsp[(5) - (8)].e_string));
			(yyval.enp)->sen_index = (yyvsp[(1) - (8)].fundex);
			}
    break;

  case 21:
#line 336 "nexpr.y"
    {
			(yyval.enp) = (yyvsp[(2) - (3)].enp) ; }
    break;

  case 22:
#line 338 "nexpr.y"
    {
			(yyval.enp)=NODE2(N_PLUS,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 23:
#line 341 "nexpr.y"
    { (yyval.enp) = NODE2(N_MINUS,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 24:
#line 342 "nexpr.y"
    { (yyval.enp) = NODE2(N_DIVIDE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 25:
#line 343 "nexpr.y"
    { (yyval.enp) = NODE2(N_TIMES,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 26:
#line 344 "nexpr.y"
    { (yyval.enp) = NODE2(N_MODULO,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 27:
#line 345 "nexpr.y"
    { (yyval.enp) = NODE2(N_BITAND,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 28:
#line 346 "nexpr.y"
    { (yyval.enp) = NODE2(N_BITOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 29:
#line 347 "nexpr.y"
    { (yyval.enp) = NODE2(N_BITXOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 30:
#line 348 "nexpr.y"
    { (yyval.enp) = NODE2(N_SHL,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 31:
#line 349 "nexpr.y"
    { (yyval.enp) = NODE2(N_SHR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 32:
#line 350 "nexpr.y"
    { (yyval.enp) = (yyvsp[(2) - (3)].enp); }
    break;

  case 33:
#line 351 "nexpr.y"
    { (yyval.enp) = NODE1(N_BITCOMP,(yyvsp[(2) - (2)].enp)); }
    break;

  case 34:
#line 352 "nexpr.y"
    { (yyval.enp) = (yyvsp[(2) - (2)].enp); }
    break;

  case 35:
#line 353 "nexpr.y"
    { (yyval.enp) = NODE1(N_UMINUS,(yyvsp[(2) - (2)].enp)); }
    break;

  case 36:
#line 355 "nexpr.y"
    { (yyval.enp) = NODE3(N_CONDITIONAL,(yyvsp[(1) - (5)].enp),(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp)); }
    break;

  case 37:
#line 356 "nexpr.y"
    { (yyval.enp)=NODE2(N_EQUIV,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 38:
#line 357 "nexpr.y"
    { (yyval.enp) = NODE1(N_NOT,(yyvsp[(2) - (2)].enp)); }
    break;

  case 39:
#line 358 "nexpr.y"
    { (yyval.enp)=NODE2(N_LOGOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 40:
#line 359 "nexpr.y"
    { (yyval.enp)=NODE2(N_LOGAND,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 41:
#line 360 "nexpr.y"
    { (yyval.enp)=NODE2(N_LOGXOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 42:
#line 361 "nexpr.y"
    { (yyval.enp)=NODE2(N_LT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 43:
#line 362 "nexpr.y"
    { (yyval.enp)=NODE2(N_GT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 44:
#line 363 "nexpr.y"
    { (yyval.enp)=NODE2(N_GE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 45:
#line 364 "nexpr.y"
    { (yyval.enp)=NODE2(N_LE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 46:
#line 365 "nexpr.y"
    { (yyval.enp)=NODE2(N_NE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 47:
#line 377 "nexpr.y"
    {
			/* must be a scalar object */
			(yyval.enp)=NODE1(N_SCALAR_OBJ,(yyvsp[(1) - (1)].enp));
			}
    break;


/* Line 1267 of yacc.c.  */
#line 1999 "nexpr.c"
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


#line 384 "nexpr.y"


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
			sprintf(ERROR_STRING,"0x%lx\tstring\t%s",
				(int_for_addr)enp, enp->sen_string);
			ADVISE(ERROR_STRING);
			break;

		case N_SIZABLE:
			sprintf(ERROR_STRING,"0x%lx\tsizable\t%s",
				(int_for_addr)enp, enp->sen_string);
			ADVISE(ERROR_STRING);
			break;

		case N_SIZFUNC:
			sprintf(ERROR_STRING,"0x%lx\tsizefunc\t%s",
				(int_for_addr)enp, size_functbl[ enp->sen_index ].fn_name);
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


