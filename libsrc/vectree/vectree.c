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
     TIMES_EQ = 258,
     PLUS_EQ = 259,
     PLUS_PLUS = 260,
     MINUS_MINUS = 261,
     MINUS_EQ = 262,
     DIV_EQ = 263,
     OR_EQ = 264,
     AND_EQ = 265,
     XOR_EQ = 266,
     SHL_EQ = 267,
     SHR_EQ = 268,
     LOGOR = 269,
     LOGXOR = 270,
     LOGAND = 271,
     NE = 272,
     LOG_EQ = 273,
     LE = 274,
     GE = 275,
     SHR = 276,
     SHL = 277,
     DOT = 278,
     UNARY = 279,
     NUMBER = 280,
     VARIABLE = 281,
     INT_NUM = 282,
     CHAR_CONST = 283,
     MATH0_FUNC = 284,
     MATH1_FUNC = 285,
     MATH2_FUNC = 286,
     DATA_FUNC = 287,
     SIZE_FUNC = 288,
     MISC_FUNC = 289,
     STR1_FUNC = 290,
     STR2_FUNC = 291,
     BEGIN_COMMENT = 292,
     END_COMMENT = 293,
     WHILE = 294,
     UNTIL = 295,
     CONTINUE = 296,
     SWITCH = 297,
     CASE = 298,
     DEFAULT = 299,
     BREAK = 300,
     GOTO = 301,
     DO = 302,
     FOR = 303,
     STATIC = 304,
     BYTE = 305,
     CHAR = 306,
     STRING = 307,
     FLOAT = 308,
     DOUBLE = 309,
     SHORT = 310,
     LONG = 311,
     BIT = 312,
     UBYTE = 313,
     USHORT = 314,
     ULONG = 315,
     COLOR = 316,
     COMPLEX = 317,
     DBLCPX = 318,
     STRCPY = 319,
     NAME_FUNC = 320,
     FILE_EXISTS = 321,
     STRCAT = 322,
     ECHO = 323,
     ADVISE_FUNC = 324,
     DISPLAY = 325,
     F_WARN = 326,
     PRINT = 327,
     INFO = 328,
     IF = 329,
     ELSE = 330,
     RETURN = 331,
     EXIT = 332,
     MIN = 333,
     MAX = 334,
     WRAP = 335,
     SCROLL = 336,
     DILATE = 337,
     FIX_SIZE = 338,
     FILL = 339,
     CLR_OPT_PARAMS = 340,
     ADD_OPT_PARAM = 341,
     OPTIMIZE = 342,
     ERODE = 343,
     ENLARGE = 344,
     REDUCE = 345,
     WARP = 346,
     LOOKUP = 347,
     EQUIVALENCE = 348,
     TRANSPOSE = 349,
     CONJ = 350,
     MAX_TIMES = 351,
     MAX_INDEX = 352,
     MIN_INDEX = 353,
     DFT = 354,
     IDFT = 355,
     RDFT = 356,
     RIDFT = 357,
     REAL_PART = 358,
     IMAG_PART = 359,
     RAMP = 360,
     SUM = 361,
     END = 362,
     NEXT_TOKEN = 363,
     NEWLINE = 364,
     SET_OUTPUT_FILE = 365,
     LOAD = 366,
     SAVE = 367,
     FILETYPE = 368,
     OBJ_OF = 369,
     FUNCNAME = 370,
     REFFUNC = 371,
     SCRIPTFUNC = 372,
     OBJNAME = 373,
     PTRNAME = 374,
     STRNAME = 375,
     LABELNAME = 376,
     FUNCPTRNAME = 377,
     LEX_STRING = 378,
     NEWNAME = 379,
     VOID_TYPE = 380,
     EXTERN = 381,
     CONST_TYPE = 382,
     NATIVE_FUNC_NAME = 383
   };
#endif
/* Tokens.  */
#define TIMES_EQ 258
#define PLUS_EQ 259
#define PLUS_PLUS 260
#define MINUS_MINUS 261
#define MINUS_EQ 262
#define DIV_EQ 263
#define OR_EQ 264
#define AND_EQ 265
#define XOR_EQ 266
#define SHL_EQ 267
#define SHR_EQ 268
#define LOGOR 269
#define LOGXOR 270
#define LOGAND 271
#define NE 272
#define LOG_EQ 273
#define LE 274
#define GE 275
#define SHR 276
#define SHL 277
#define DOT 278
#define UNARY 279
#define NUMBER 280
#define VARIABLE 281
#define INT_NUM 282
#define CHAR_CONST 283
#define MATH0_FUNC 284
#define MATH1_FUNC 285
#define MATH2_FUNC 286
#define DATA_FUNC 287
#define SIZE_FUNC 288
#define MISC_FUNC 289
#define STR1_FUNC 290
#define STR2_FUNC 291
#define BEGIN_COMMENT 292
#define END_COMMENT 293
#define WHILE 294
#define UNTIL 295
#define CONTINUE 296
#define SWITCH 297
#define CASE 298
#define DEFAULT 299
#define BREAK 300
#define GOTO 301
#define DO 302
#define FOR 303
#define STATIC 304
#define BYTE 305
#define CHAR 306
#define STRING 307
#define FLOAT 308
#define DOUBLE 309
#define SHORT 310
#define LONG 311
#define BIT 312
#define UBYTE 313
#define USHORT 314
#define ULONG 315
#define COLOR 316
#define COMPLEX 317
#define DBLCPX 318
#define STRCPY 319
#define NAME_FUNC 320
#define FILE_EXISTS 321
#define STRCAT 322
#define ECHO 323
#define ADVISE_FUNC 324
#define DISPLAY 325
#define F_WARN 326
#define PRINT 327
#define INFO 328
#define IF 329
#define ELSE 330
#define RETURN 331
#define EXIT 332
#define MIN 333
#define MAX 334
#define WRAP 335
#define SCROLL 336
#define DILATE 337
#define FIX_SIZE 338
#define FILL 339
#define CLR_OPT_PARAMS 340
#define ADD_OPT_PARAM 341
#define OPTIMIZE 342
#define ERODE 343
#define ENLARGE 344
#define REDUCE 345
#define WARP 346
#define LOOKUP 347
#define EQUIVALENCE 348
#define TRANSPOSE 349
#define CONJ 350
#define MAX_TIMES 351
#define MAX_INDEX 352
#define MIN_INDEX 353
#define DFT 354
#define IDFT 355
#define RDFT 356
#define RIDFT 357
#define REAL_PART 358
#define IMAG_PART 359
#define RAMP 360
#define SUM 361
#define END 362
#define NEXT_TOKEN 363
#define NEWLINE 364
#define SET_OUTPUT_FILE 365
#define LOAD 366
#define SAVE 367
#define FILETYPE 368
#define OBJ_OF 369
#define FUNCNAME 370
#define REFFUNC 371
#define SCRIPTFUNC 372
#define OBJNAME 373
#define PTRNAME 374
#define STRNAME 375
#define LABELNAME 376
#define FUNCPTRNAME 377
#define LEX_STRING 378
#define NEWNAME 379
#define VOID_TYPE 380
#define EXTERN 381
#define CONST_TYPE 382
#define NATIVE_FUNC_NAME 383




/* Copy the first part of user declarations.  */
#line 1 "vectree.y"

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
#line 483 "vectree.c"

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
#define YYFINAL  164
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   4770

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  152
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  63
/* YYNRULES -- Number of rules.  */
#define YYNRULES  299
/* YYNRULES -- Number of states.  */
#define YYNSTATES  719

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   383

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    39,     2,     2,     2,    35,    22,     2,
     147,   148,    33,    31,    42,    32,     2,    34,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    16,   151,
      25,     3,    26,    15,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    37,     2,    38,    21,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,   149,    20,   150,    40,     2,     2,     2,
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
       2,     2,     2,     2,     2,     2,     1,     2,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,    17,
      18,    19,    23,    24,    27,    28,    29,    30,    36,    41,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    57,    58,    59,    60,    61,    62,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    90,    91,    92,
      93,    94,    95,    96,    97,    98,    99,   100,   101,   102,
     103,   104,   105,   106,   107,   108,   109,   110,   111,   112,
     113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,   125,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   139,   140,   141,   142,
     143,   144,   145,   146
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint16 yyprhs[] =
{
       0,     0,     3,     5,     7,     9,    15,    17,    20,    25,
      27,    32,    37,    42,    47,    54,    61,    66,    71,    76,
      78,    83,    87,    91,    95,    99,   103,   107,   111,   115,
     119,   123,   127,   130,   132,   136,   140,   144,   148,   152,
     156,   160,   164,   168,   171,   175,   179,   183,   188,   195,
     199,   201,   207,   209,   211,   216,   221,   226,   231,   236,
     241,   246,   251,   258,   262,   267,   270,   275,   280,   285,
     290,   298,   303,   305,   307,   314,   321,   326,   331,   336,
     341,   346,   348,   353,   362,   367,   372,   377,   382,   387,
     396,   405,   407,   410,   412,   414,   416,   419,   420,   422,
     426,   431,   439,   442,   444,   446,   455,   460,   463,   465,
     467,   471,   475,   479,   483,   486,   489,   492,   495,   499,
     503,   507,   511,   515,   519,   523,   527,   531,   534,   536,
     538,   541,   544,   547,   549,   551,   554,   557,   561,   566,
     569,   573,   577,   582,   587,   591,   596,   600,   601,   603,
     607,   609,   611,   613,   615,   618,   620,   624,   627,   630,
     632,   635,   637,   639,   641,   643,   645,   647,   649,   651,
     653,   655,   657,   659,   661,   663,   665,   670,   675,   677,
     681,   686,   688,   692,   697,   702,   709,   714,   719,   723,
     725,   727,   734,   741,   746,   759,   763,   778,   783,   788,
     793,   798,   803,   808,   810,   812,   814,   816,   818,   820,
     822,   824,   832,   837,   842,   850,   858,   869,   880,   894,
     898,   902,   908,   914,   922,   930,   940,   943,   945,   947,
     949,   953,   957,   963,   966,   968,   971,   975,   980,   985,
     987,   989,   995,  1001,  1011,  1019,  1027,  1030,  1032,  1035,
    1039,  1042,  1044,  1047,  1055,  1061,  1069,  1070,  1072,  1074,
    1076,  1078,  1080,  1082,  1084,  1086,  1088,  1090,  1092,  1094,
    1096,  1098,  1101,  1104,  1106,  1108,  1112,  1116,  1118,  1122,
    1124,  1128,  1130,  1134,  1136,  1140,  1142,  1144,  1146,  1150,
    1152,  1156,  1158,  1163,  1165,  1167,  1169,  1171,  1173,  1177
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     176,     0,    -1,   137,    -1,   140,    -1,   138,    -1,   158,
      16,   158,    16,   158,    -1,   136,    -1,    33,   153,    -1,
     132,   147,   211,   148,    -1,   142,    -1,   121,   147,   157,
     148,    -1,   122,   147,   157,   148,    -1,   157,    37,   158,
      38,    -1,   157,   149,   158,   150,    -1,   157,    37,   158,
      16,   158,    38,    -1,   157,   149,   158,    16,   158,   150,
      -1,   157,    37,   156,    38,    -1,   157,   149,   156,   150,
      -1,   101,   147,   158,   148,    -1,   214,    -1,   147,   177,
     148,   158,    -1,   147,   158,   148,    -1,   158,    31,   158,
      -1,   158,    32,   158,    -1,   158,    33,   158,    -1,   158,
      34,   158,    -1,   158,    35,   158,    -1,   158,    22,   158,
      -1,   158,    20,   158,    -1,   158,    21,   158,    -1,   158,
      30,   158,    -1,   158,    29,   158,    -1,    40,   158,    -1,
      45,    -1,   158,    24,   158,    -1,   158,    25,   158,    -1,
     158,    26,   158,    -1,   158,    28,   158,    -1,   158,    27,
     158,    -1,   158,    23,   158,    -1,   158,    19,   158,    -1,
     158,    17,   158,    -1,   158,    18,   158,    -1,    39,   158,
      -1,   153,    23,   162,    -1,   153,    24,   162,    -1,    47,
     147,   148,    -1,    48,   147,   158,   148,    -1,    49,   147,
     158,    42,   158,   148,    -1,   158,    36,   158,    -1,    43,
      -1,   158,    15,   158,    16,   158,    -1,    46,    -1,    50,
      -1,    50,   147,   157,   148,    -1,    51,   147,   214,   148,
      -1,    51,   147,   157,   148,    -1,    51,   147,   153,   148,
      -1,   124,   147,   153,   148,    -1,   124,   147,   207,   148,
      -1,    84,   147,   214,   148,    -1,    53,   147,   214,   148,
      -1,    54,   147,   214,    42,   214,   148,    -1,    52,   147,
     148,    -1,   113,   147,   158,   148,    -1,    32,   158,    -1,
      96,   147,   207,   148,    -1,    97,   147,   207,   148,    -1,
     115,   147,   158,   148,    -1,   116,   147,   158,   148,    -1,
     147,    33,   154,   148,   147,   160,   148,    -1,   133,   147,
     160,   148,    -1,   203,    -1,   204,    -1,   109,   147,   158,
      42,   158,   148,    -1,   110,   147,   158,    42,   158,   148,
      -1,   112,   147,   158,   148,    -1,   117,   147,   158,   148,
      -1,   118,   147,   158,   148,    -1,   119,   147,   158,   148,
      -1,   120,   147,   158,   148,    -1,   167,    -1,    98,   147,
     158,   148,    -1,    99,   147,   158,    42,   158,    42,   158,
     148,    -1,   106,   147,   158,   148,    -1,   100,   147,   158,
     148,    -1,   107,   147,   158,   148,    -1,   108,   147,   158,
     148,    -1,   129,   147,   214,   148,    -1,   123,   147,   158,
      42,   158,    42,   158,   148,    -1,   114,   147,   162,    42,
     162,    42,   158,   148,    -1,   158,    -1,    22,   158,    -1,
     163,    -1,   153,    -1,   164,    -1,    22,   153,    -1,    -1,
     159,    -1,   160,    42,   159,    -1,   133,   147,   160,   148,
      -1,   147,    33,   154,   148,   147,   160,   148,    -1,    22,
     157,    -1,   164,    -1,   153,    -1,   111,   147,   157,    42,
     207,    42,   178,   148,    -1,   134,   147,   160,   148,    -1,
      22,   133,    -1,   165,    -1,   154,    -1,   153,     3,   162,
      -1,   154,     3,   163,    -1,   155,     3,   208,    -1,   157,
       3,   158,    -1,   157,     6,    -1,     6,   157,    -1,     7,
     157,    -1,   157,     7,    -1,   157,     5,   158,    -1,   157,
       4,   158,    -1,   157,     8,   158,    -1,   157,     9,   158,
      -1,   157,    11,   158,    -1,   157,    10,   158,    -1,   157,
      12,   158,    -1,   157,    13,   158,    -1,   157,    14,   158,
      -1,   201,   151,    -1,   170,    -1,   199,    -1,   142,    16,
      -1,   139,    16,    -1,     1,   151,    -1,   168,    -1,   202,
      -1,   169,   168,    -1,   169,   202,    -1,   149,   169,   150,
      -1,   149,   191,   169,   150,    -1,   149,   150,    -1,   149,
       1,   150,    -1,   149,   169,   125,    -1,   142,   147,   174,
     148,    -1,   133,   147,   174,   148,    -1,   177,   171,   170,
      -1,   177,    33,   171,   170,    -1,   177,   172,   170,    -1,
      -1,   190,    -1,   174,    42,   190,    -1,   173,    -1,   192,
      -1,   168,    -1,   202,    -1,   175,   125,    -1,   175,    -1,
     176,   175,   125,    -1,   176,   175,    -1,     1,   125,    -1,
     178,    -1,   145,   178,    -1,    68,    -1,    69,    -1,    70,
      -1,    71,    -1,    72,    -1,    80,    -1,    81,    -1,    73,
      -1,    74,    -1,    76,    -1,    77,    -1,    78,    -1,    75,
      -1,    79,    -1,   143,    -1,    91,   147,   207,   148,    -1,
      88,   147,   207,   148,    -1,    95,    -1,    95,   147,   148,
      -1,    95,   147,   158,   148,    -1,    94,    -1,    94,   147,
     148,    -1,    94,   147,   158,   148,    -1,    94,   147,   162,
     148,    -1,   130,   147,   214,    42,   158,   148,    -1,   131,
     147,   214,   148,    -1,   135,   147,   208,   148,    -1,   135,
     147,   148,    -1,   155,    -1,   142,    -1,    82,   147,   184,
      42,   213,   148,    -1,    85,   147,   184,    42,   213,   148,
      -1,   146,   147,   160,   148,    -1,   102,   147,   162,    42,
     158,    42,   158,    42,   158,    42,   158,   148,    -1,   103,
     147,   148,    -1,   104,   147,   162,    42,   158,    42,   158,
      42,   158,    42,   158,    42,   158,   148,    -1,   105,   147,
     133,   148,    -1,   128,   147,   214,   148,    -1,    90,   147,
     210,   148,    -1,    86,   147,   208,   148,    -1,    87,   147,
     208,   148,    -1,    89,   147,   208,   148,    -1,   142,    -1,
     136,    -1,   137,    -1,   138,    -1,   178,    -1,   187,    -1,
     171,    -1,   172,    -1,   147,    33,   187,   148,   147,   174,
     148,    -1,   187,   149,   158,   150,    -1,   187,    37,   158,
      38,    -1,   187,    37,   158,    38,   149,   158,   150,    -1,
     187,    37,   158,    38,    37,   158,    38,    -1,   187,    37,
     158,    38,    37,   158,    38,   149,   158,   150,    -1,   187,
      37,   158,    38,    37,   158,    38,    37,   158,    38,    -1,
     187,    37,   158,    38,    37,   158,    38,    37,   158,    38,
     149,   158,   150,    -1,   187,   149,   150,    -1,   187,    37,
      38,    -1,   187,    37,    38,   149,   150,    -1,   187,    37,
      38,    37,    38,    -1,   187,    37,    38,    37,    38,   149,
     150,    -1,   187,    37,    38,    37,    38,    37,    38,    -1,
     187,    37,    38,    37,    38,    37,    38,   149,   150,    -1,
      33,   187,    -1,    50,    -1,    51,    -1,   188,    -1,   188,
       3,   158,    -1,   189,    42,   188,    -1,   189,    42,   188,
       3,   158,    -1,   177,   188,    -1,   192,    -1,   191,   192,
      -1,   177,   189,   151,    -1,   144,   177,   189,   151,    -1,
      67,   177,   189,   151,    -1,   168,    -1,   202,    -1,    57,
     147,   158,   148,   193,    -1,    58,   147,   158,   148,   193,
      -1,    66,   147,   201,   151,   158,   151,   201,   148,   193,
      -1,    65,   193,    57,   147,   158,   148,   151,    -1,    65,
     193,    58,   147,   158,   148,   151,    -1,   196,   169,    -1,
     197,    -1,   196,   197,    -1,    61,   158,    16,    -1,    62,
      16,    -1,   195,    -1,   198,   195,    -1,    60,   147,   158,
     148,   149,   198,   150,    -1,    92,   147,   158,   148,   193,
      -1,    92,   147,   158,   148,   193,    93,   193,    -1,    -1,
     179,    -1,   186,    -1,   185,    -1,   182,    -1,   183,    -1,
     181,    -1,   180,    -1,   167,    -1,   164,    -1,   165,    -1,
     166,    -1,   161,    -1,    63,    -1,    59,    -1,    64,   139,
      -1,    64,   142,    -1,   200,    -1,   194,    -1,   149,   205,
     150,    -1,    37,   206,    38,    -1,   158,    -1,   205,    42,
     158,    -1,   158,    -1,   206,    42,   158,    -1,   158,    -1,
     207,    42,   158,    -1,   158,    -1,   208,    42,   158,    -1,
     158,    -1,   153,    -1,   209,    -1,   210,    42,   209,    -1,
     214,    -1,   211,    42,   214,    -1,   141,    -1,    83,   147,
     157,   148,    -1,   158,    -1,   212,    -1,   155,    -1,   166,
      -1,   157,    -1,   147,   166,   148,    -1,   147,   208,   148,
      -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   388,   388,   395,   402,   409,   415,   425,   429,   433,
     448,   451,   454,   457,   460,   464,   472,   476,   483,   494,
     495,   500,   502,   504,   506,   508,   510,   512,   514,   516,
     518,   520,   522,   524,   528,   531,   534,   537,   540,   543,
     546,   549,   552,   555,   558,   573,   577,   582,   587,   593,
     596,   600,   605,   609,   616,   621,   625,   629,   636,   642,
     646,   650,   655,   661,   666,   672,   675,   679,   684,   686,
     689,   693,   705,   708,   711,   715,   718,   723,   724,   725,
     729,   733,   734,   737,   741,   744,   747,   751,   755,   757,
     760,   767,   768,   777,   778,   779,   780,   787,   792,   793,
     802,   813,   823,   825,   826,   827,   832,   846,   851,   852,
     855,   860,   865,   877,   880,   881,   882,   883,   884,   890,
     896,   902,   908,   914,   920,   926,   932,   945,   947,   948,
     949,   957,   962,   968,   969,   970,   981,   994,   998,  1002,
    1006,  1010,  1017,  1031,  1067,  1077,  1089,  1110,  1113,  1117,
    1123,  1124,  1125,  1129,  1135,  1137,  1139,  1143,  1147,  1154,
    1155,  1158,  1159,  1160,  1161,  1162,  1163,  1164,  1165,  1166,
    1167,  1168,  1169,  1170,  1171,  1172,  1176,  1178,  1182,  1183,
    1184,  1187,  1191,  1195,  1199,  1205,  1207,  1211,  1216,  1223,
    1224,  1232,  1236,  1245,  1259,  1266,  1271,  1279,  1286,  1291,
    1292,  1293,  1294,  1297,  1298,  1300,  1302,  1304,  1311,  1321,
    1325,  1329,  1335,  1339,  1343,  1347,  1351,  1355,  1359,  1365,
    1369,  1374,  1379,  1384,  1389,  1394,  1399,  1404,  1412,  1459,
    1460,  1463,  1465,  1476,  1484,  1485,  1492,  1499,  1506,  1516,
    1517,  1520,  1527,  1534,  1545,  1550,  1557,  1561,  1562,  1566,
    1568,  1572,  1573,  1577,  1582,  1584,  1600,  1601,  1602,  1603,
    1604,  1605,  1606,  1607,  1608,  1609,  1610,  1611,  1612,  1613,
    1614,  1615,  1620,  1632,  1634,  1643,  1648,  1653,  1654,  1660,
    1661,  1667,  1668,  1675,  1676,  1680,  1681,  1684,  1685,  1689,
    1690,  1694,  1700,  1707,  1712,  1713,  1714,  1715,  1716,  1717
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "'='", "TIMES_EQ", "PLUS_EQ",
  "PLUS_PLUS", "MINUS_MINUS", "MINUS_EQ", "DIV_EQ", "OR_EQ", "AND_EQ",
  "XOR_EQ", "SHL_EQ", "SHR_EQ", "'?'", "':'", "LOGOR", "LOGXOR", "LOGAND",
  "'|'", "'^'", "'&'", "NE", "LOG_EQ", "'<'", "'>'", "LE", "GE", "SHR",
  "SHL", "'+'", "'-'", "'*'", "'/'", "'%'", "DOT", "'['", "']'", "'!'",
  "'~'", "UNARY", "','", "NUMBER", "VARIABLE", "INT_NUM", "CHAR_CONST",
  "MATH0_FUNC", "MATH1_FUNC", "MATH2_FUNC", "DATA_FUNC", "SIZE_FUNC",
  "MISC_FUNC", "STR1_FUNC", "STR2_FUNC", "BEGIN_COMMENT", "END_COMMENT",
  "WHILE", "UNTIL", "CONTINUE", "SWITCH", "CASE", "DEFAULT", "BREAK",
  "GOTO", "DO", "FOR", "STATIC", "BYTE", "CHAR", "STRING", "FLOAT",
  "DOUBLE", "SHORT", "LONG", "BIT", "UBYTE", "USHORT", "ULONG", "COLOR",
  "COMPLEX", "DBLCPX", "STRCPY", "NAME_FUNC", "FILE_EXISTS", "STRCAT",
  "ECHO", "ADVISE_FUNC", "DISPLAY", "F_WARN", "PRINT", "INFO", "IF",
  "ELSE", "RETURN", "EXIT", "MIN", "MAX", "WRAP", "SCROLL", "DILATE",
  "FIX_SIZE", "FILL", "CLR_OPT_PARAMS", "ADD_OPT_PARAM", "OPTIMIZE",
  "ERODE", "ENLARGE", "REDUCE", "WARP", "LOOKUP", "EQUIVALENCE",
  "TRANSPOSE", "CONJ", "MAX_TIMES", "MAX_INDEX", "MIN_INDEX", "DFT",
  "IDFT", "RDFT", "RIDFT", "REAL_PART", "IMAG_PART", "RAMP", "SUM", "END",
  "NEXT_TOKEN", "NEWLINE", "SET_OUTPUT_FILE", "LOAD", "SAVE", "FILETYPE",
  "OBJ_OF", "FUNCNAME", "REFFUNC", "SCRIPTFUNC", "OBJNAME", "PTRNAME",
  "STRNAME", "LABELNAME", "FUNCPTRNAME", "LEX_STRING", "NEWNAME",
  "VOID_TYPE", "EXTERN", "CONST_TYPE", "NATIVE_FUNC_NAME", "'('", "')'",
  "'{'", "'}'", "';'", "$accept", "pointer", "func_ptr", "str_ptr",
  "subsamp_spec", "objref", "expression", "func_arg", "func_args",
  "void_call", "ref_arg", "func_ref_arg", "ptr_assgn", "funcptr_assgn",
  "str_assgn", "assignment", "statline", "stat_list", "stat_block",
  "new_func_decl", "old_func_decl", "subroutine", "arg_decl_list",
  "prog_elt", "program", "data_type", "precision", "info_stat",
  "exit_stat", "return_stat", "fileio_stat", "script_stat", "str_ptr_arg",
  "misc_stat", "print_stat", "decl_identifier", "decl_item",
  "decl_item_list", "arg_decl", "decl_stat_list", "decl_statement",
  "loop_stuff", "loop_statement", "case_statement", "case_list",
  "single_case", "switch_cases", "switch_statement", "if_statement",
  "simple_stat", "blk_stat", "comp_stack", "expr_stack", "comp_list",
  "row_list", "expr_list", "print_list", "mixed_item", "mixed_list",
  "string_list", "string", "printable", "string_arg", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,    61,   258,   259,   260,   261,   262,   263,
     264,   265,   266,   267,   268,    63,    58,   269,   270,   271,
     124,    94,    38,   272,   273,    60,    62,   274,   275,   276,
     277,    43,    45,    42,    47,    37,   278,    91,    93,    33,
     126,   279,    44,   280,   281,   282,   283,   284,   285,   286,
     287,   288,   289,   290,   291,   292,   293,   294,   295,   296,
     297,   298,   299,   300,   301,   302,   303,   304,   305,   306,
     307,   308,   309,   310,   311,   312,   313,   314,   315,   316,
     317,   318,   319,   320,   321,   322,   323,   324,   325,   326,
     327,   328,   329,   330,   331,   332,   333,   334,   335,   336,
     337,   338,   339,   340,   341,   342,   343,   344,   345,   346,
     347,   348,   349,   350,   351,   352,   353,   354,   355,   356,
     357,   358,   359,   360,   361,   362,   363,   364,   365,   366,
     367,   368,   369,   370,   371,   372,   373,   374,   375,   376,
     377,   378,   379,   380,   381,   382,   383,    40,    41,   123,
     125,    59
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,   152,   153,   154,   155,   156,   157,   157,   157,   157,
     157,   157,   157,   157,   157,   157,   157,   157,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   159,   159,   159,   159,   159,   159,   159,   160,   160,
     161,   161,   162,   162,   162,   162,   162,   163,   163,   163,
     164,   165,   166,   167,   167,   167,   167,   167,   167,   167,
     167,   167,   167,   167,   167,   167,   167,   168,   168,   168,
     168,   168,   168,   169,   169,   169,   169,   170,   170,   170,
     170,   170,   171,   172,   173,   173,   173,   174,   174,   174,
     175,   175,   175,   175,   176,   176,   176,   176,   176,   177,
     177,   178,   178,   178,   178,   178,   178,   178,   178,   178,
     178,   178,   178,   178,   178,   178,   179,   179,   180,   180,
     180,   181,   181,   181,   181,   182,   182,   183,   183,   184,
     184,   185,   185,   185,   185,   185,   185,   185,   185,   186,
     186,   186,   186,   187,   187,   187,   187,   187,   188,   188,
     188,   188,   188,   188,   188,   188,   188,   188,   188,   188,
     188,   188,   188,   188,   188,   188,   188,   188,   188,   189,
     189,   189,   189,   190,   191,   191,   192,   192,   192,   193,
     193,   194,   194,   194,   194,   194,   195,   196,   196,   197,
     197,   198,   198,   199,   200,   200,   201,   201,   201,   201,
     201,   201,   201,   201,   201,   201,   201,   201,   201,   201,
     201,   201,   201,   202,   202,   203,   204,   205,   205,   206,
     206,   207,   207,   208,   208,   209,   209,   210,   210,   211,
     211,   212,   212,   213,   214,   214,   214,   214,   214,   214
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     5,     1,     2,     4,     1,
       4,     4,     4,     4,     6,     6,     4,     4,     4,     1,
       4,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     2,     1,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     2,     3,     3,     3,     4,     6,     3,
       1,     5,     1,     1,     4,     4,     4,     4,     4,     4,
       4,     4,     6,     3,     4,     2,     4,     4,     4,     4,
       7,     4,     1,     1,     6,     6,     4,     4,     4,     4,
       4,     1,     4,     8,     4,     4,     4,     4,     4,     8,
       8,     1,     2,     1,     1,     1,     2,     0,     1,     3,
       4,     7,     2,     1,     1,     8,     4,     2,     1,     1,
       3,     3,     3,     3,     2,     2,     2,     2,     3,     3,
       3,     3,     3,     3,     3,     3,     3,     2,     1,     1,
       2,     2,     2,     1,     1,     2,     2,     3,     4,     2,
       3,     3,     4,     4,     3,     4,     3,     0,     1,     3,
       1,     1,     1,     1,     2,     1,     3,     2,     2,     1,
       2,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     4,     4,     1,     3,
       4,     1,     3,     4,     4,     6,     4,     4,     3,     1,
       1,     6,     6,     4,    12,     3,    14,     4,     4,     4,
       4,     4,     4,     1,     1,     1,     1,     1,     1,     1,
       1,     7,     4,     4,     7,     7,    10,    10,    13,     3,
       3,     5,     5,     7,     7,     9,     2,     1,     1,     1,
       3,     3,     5,     2,     1,     2,     3,     4,     4,     1,
       1,     5,     5,     9,     7,     7,     2,     1,     2,     3,
       2,     1,     2,     7,     5,     7,     0,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     2,     2,     1,     1,     3,     3,     1,     3,     1,
       3,     1,     3,     1,     3,     1,     1,     1,     3,     1,
       3,     1,     4,     1,     1,     1,     1,     1,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       0,     0,     0,     0,     0,     0,     0,   270,     0,   269,
       0,     0,     0,     0,   161,   162,   163,   164,   165,   168,
     169,   173,   170,   171,   172,   174,   166,   167,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   181,   178,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     6,     2,     4,     0,     3,     9,   175,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   268,   265,   266,
     267,   264,   152,   128,   150,   155,     0,     0,   159,   257,
     263,   262,   260,   261,   259,   258,   151,   274,   129,   273,
       0,   153,   158,   132,     9,   115,   116,     7,     0,     0,
       0,   271,   272,     0,   239,     0,   240,   256,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
      97,     0,   131,   130,     0,   160,    97,     0,     0,   139,
     133,     0,     0,     0,   234,   134,     0,     0,     0,     0,
       0,     0,   114,   117,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   154,     1,   157,     0,   227,   228,     0,
     204,   205,   206,   203,     0,   209,   210,   207,   208,   229,
       0,   127,     0,     0,     0,     0,    50,    33,    52,     0,
       0,     0,    53,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   291,     0,     0,     0,   295,   297,     0,
     296,    81,    72,    73,   294,    19,     0,     0,     0,     0,
       0,     0,   209,   210,     0,   190,   189,     0,     0,   283,
       0,     0,   281,     0,     0,   286,   285,   287,     0,     0,
       0,     0,     0,     0,   182,   104,     0,     0,   103,   179,
       0,   104,     0,   195,     0,     0,     0,     0,     0,   297,
       0,     0,     0,     0,   289,     0,    94,   109,    91,    98,
       0,    93,    95,   108,   188,     0,     0,     0,     0,   140,
     141,   137,   135,   136,     0,   235,   110,     0,   111,   112,
     113,   119,   118,   120,   121,   123,   122,   124,   125,   126,
       0,     0,     0,     0,   156,     0,   226,   147,   147,     0,
     144,   146,     0,     0,     0,     0,   236,    65,   279,     0,
      43,    32,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    97,     0,   283,   296,     0,
       0,   277,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   203,   238,     0,     0,     0,   200,   201,     0,
     177,   202,     0,   199,   176,     0,   102,     0,    97,   183,
     184,   180,     0,     0,   197,    10,    11,   198,     0,   186,
       0,     8,   107,    96,    92,    97,   100,   187,   237,   193,
       0,   138,   107,    16,     0,    12,    17,     0,    13,   145,
       0,     0,   148,     0,     0,   220,     0,   219,     0,   230,
     231,   276,     0,    46,     0,     0,     0,     0,     0,     0,
      63,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    21,   298,     0,   299,     0,   275,    44,    45,     0,
      41,    42,    40,    28,    29,    27,    39,    34,    35,    36,
      38,    37,    31,    30,    22,    23,    24,    25,    26,    49,
     241,   242,     0,     0,     0,     0,   293,     0,     0,   284,
     282,   288,   254,     0,     0,     0,     0,     0,   290,    99,
      97,     0,     0,     0,   143,   233,   142,     0,     0,     0,
     213,   212,     0,   280,    47,     0,    54,    57,    56,    55,
      61,     0,   292,    60,    66,    67,    82,     0,    85,    18,
      84,    86,    87,     0,     0,    76,    64,     0,    68,    69,
      77,    78,    79,    80,     0,    58,    59,    88,    71,     0,
      20,   278,     0,     0,     0,   251,     0,   247,     0,     0,
       0,   256,   191,   192,     0,     0,   106,     0,     0,   185,
       0,     0,    14,    15,   149,   147,   222,   221,     0,     0,
     232,     0,     0,     0,     0,     0,     0,     0,    97,    51,
       0,   250,     0,   248,   253,   252,   244,   245,     0,   255,
       0,     0,     0,   101,     5,     0,     0,     0,     0,     0,
      48,    62,     0,    74,    75,     0,     0,     0,   249,     0,
       0,     0,     0,   211,   224,   223,   215,   214,     0,     0,
       0,    70,   243,     0,     0,     0,     0,     0,     0,    83,
      90,    89,   105,     0,     0,   225,     0,     0,     0,     0,
     217,   216,   194,     0,     0,     0,     0,   196,   218
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,   226,    64,   227,   320,   228,   288,   289,   290,    67,
     267,   291,    68,    69,   230,   231,   104,   141,    73,   242,
     243,    74,   460,    75,    76,   461,    78,    79,    80,    81,
      82,    83,   247,    84,    85,   178,   179,   180,   462,   143,
      86,   105,    87,   615,   616,   617,   618,    88,    89,    90,
     106,   232,   233,   382,   339,   253,   380,   257,   258,   283,
     234,   547,   235
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -388
static const yytype_int16 yypact[] =
{
    1149,   -83,   232,   232,   -98,   -57,   -52,  -388,   -47,  -388,
     156,  1797,   -43,   530,  -388,  -388,  -388,  -388,  -388,  -388,
    -388,  -388,  -388,  -388,  -388,  -388,  -388,  -388,   -41,   -39,
     -37,   -28,   -22,   -14,    29,    39,    64,    83,    89,   120,
     127,   131,   136,   146,   152,   154,   173,   176,   186,   193,
     195,  -388,  -388,  -388,    66,  -388,   241,  -388,   530,  1913,
     197,    69,  1041,   285,   310,   331,   194,  -388,  -388,  -388,
    -388,  -388,  -388,  -388,  -388,   225,   806,   144,  -388,  -388,
    -388,  -388,  -388,  -388,  -388,  -388,  -388,  -388,  -388,  -388,
     205,  -388,  -388,  -388,  -388,    24,    24,  -388,  2876,  2876,
    2876,  -388,  -388,   206,  -388,    15,  -388,  3066,  4105,   -55,
     -55,  2876,  2876,  2876,  2876,  2876,  2876,  2876,  2279,  2519,
      38,   210,    38,   227,   232,   232,   314,   314,   314,   314,
    2400,  2638,  -388,  -388,  4105,  -388,  2400,   221,   158,  -388,
    -388,  1473,  4105,  1257,  -388,  -388,    38,     5,  2876,  2876,
    2876,  2876,  -388,  -388,  2876,  2876,  2876,  2876,  2876,  2876,
    2876,  2876,  2876,  -388,  -388,   237,   779,  -388,  -388,   219,
    -388,  -388,  -388,   220,   336,   239,   239,  -388,    30,   368,
     -13,  -388,  2876,  2876,  2876,  2876,  -388,  -388,  -388,   248,
     249,   255,   256,   258,   264,   265,   270,   271,   273,   274,
     280,   282,   283,   284,   286,   287,   290,   291,   293,   302,
     304,   311,   312,   313,   316,   318,   320,   322,   323,   324,
     326,   327,   328,  -388,  2036,  2876,   139,   331,   194,  3265,
    -388,  -388,  -388,  -388,  -388,  -388,  3287,  3309,   329,   333,
     281,  1836,  -388,  -388,    -9,  -388,  -388,   399,   415,  4734,
       4,    43,  4734,    44,    86,   139,  4734,  -388,    87,    90,
    3333,   232,   337,   338,  -388,    88,  3355,   330,  -388,  -388,
    3397,   285,   441,  -388,   444,   342,    61,    79,  2876,    24,
     343,   450,   345,   108,  -388,  2995,    88,   310,  4734,  -388,
     114,  -388,  -388,  -388,  -388,   115,    36,   141,   346,  -388,
    -388,  -388,  -388,  -388,  1689,  -388,  -388,   363,  -388,   455,
    4734,  4734,  4734,  4734,  4734,  4734,  4734,  4734,  4734,  4734,
     460,  4576,   349,  2559,  -388,   239,  -388,   530,   530,  1836,
    -388,  -388,  2757,  1917,  2876,  4105,  -388,  -388,  4734,    54,
    -388,  -388,   361,  2876,  2876,   232,   272,   362,   314,   314,
     232,   314,  2876,  2876,  2876,  2876,  2876,  2876,  2876,  2876,
    2876,  2876,  2876,  2876,  2876,    38,  2876,  2876,  2876,  2876,
    2876,  2876,  2876,  2876,   314,  2400,   166,  3419,   369,   370,
     187,  4734,    14,    38,    38,  2876,  2876,  2876,  2876,  2876,
    2876,  2876,  2876,  2876,  2876,  2876,  2876,  2876,  2876,  2876,
    2876,  2876,  2876,  2876,  2876,  2876,  1797,  1797,   364,  2876,
    2876,  2876,  -388,  -388,  2876,  2876,  2876,  -388,  -388,  2876,
    -388,  -388,  2876,  -388,  -388,  1797,    24,   232,  2400,  -388,
    -388,  -388,  2876,  2876,  -388,  -388,  -388,  -388,  2876,  -388,
     314,  -388,   328,   139,  4734,  2400,  -388,  -388,  -388,  -388,
     372,  -388,  -388,  -388,  2876,  -388,  -388,  2876,  -388,  -388,
     191,  4105,  -388,   204,   373,    32,  4622,  -388,  3035,  4734,
     508,  -388,  2876,  -388,  3441,  4189,   100,   374,   103,   375,
    -388,   376,   474,   110,   377,   211,   222,  3465,  4236,  3487,
    3529,  3551,  3573,  3597,  4262,  4288,  3619,  3661,   484,  3683,
    3705,  3729,  3751,  3793,  3815,  4314,    26,   224,   380,   229,
     381,  -388,  -388,  2876,  -388,  2876,  -388,  -388,  -388,  4690,
    3638,  3375,  3768,   679,  1410,  1517,   559,   559,   532,   532,
     532,   532,   294,   294,   365,   365,   494,   494,   494,  -388,
    -388,  -388,   276,  3837,  3861,  2319,  4734,   383,   385,  -388,
    4734,  -388,   442,    22,   230,  4340,  4366,  3883,  -388,  -388,
    2400,  4600,  2797,   530,  -388,  -388,  -388,   387,   498,   391,
      40,  -388,  2876,  4734,  -388,  2876,  -388,  -388,  -388,  -388,
    -388,   314,  -388,  -388,  -388,  -388,  -388,  2876,  -388,  -388,
    -388,  -388,  -388,  2876,  2876,  -388,  -388,    38,  -388,  -388,
    -388,  -388,  -388,  -388,  2876,  -388,  -388,  -388,  -388,   395,
    -388,  4734,  2876,  2876,   537,  -388,  1581,  -388,   235,   403,
     404,  3066,  -388,  -388,  1797,  2876,  -388,  2876,  2876,  -388,
     231,  2876,  -388,  -388,  -388,   530,    77,  -388,  2876,  2876,
    4734,  3925,   408,  4392,  3947,  3969,   515,  4418,  2400,  3507,
    4712,  -388,  1365,  -388,  -388,  -388,  -388,  -388,   410,  -388,
     517,  4444,  4470,  -388,  4734,   233,   531,   420,  4644,  3199,
    -388,  -388,  2876,  -388,  -388,  2876,  2876,   234,  -388,  1797,
    2155,  2876,  2876,  -388,   426,  -388,    94,  -388,  3993,  4015,
    4057,  -388,  -388,   428,  4496,  4522,   427,  2876,  2876,  -388,
    -388,  -388,  -388,  2876,  2876,  -388,  4666,  3221,  4079,  4548,
     429,  -388,  -388,  2876,  2876,  4101,  3243,  -388,  -388
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -388,     0,  -121,    17,   417,    41,   389,   135,  -134,  -388,
    -115,   434,   -65,   -96,    37,    47,    12,  -137,  -140,   -32,
     505,  -388,  -318,   520,  -388,     8,   -45,  -388,  -388,  -388,
    -388,  -388,   502,  -388,  -388,  -136,  -322,   -33,    52,  -388,
      -8,  -387,  -388,    -1,  -388,     2,  -388,  -388,  -388,  -106,
      18,  -388,  -388,  -388,  -388,  -113,   200,   198,  -388,  -388,
    -388,   208,  -104
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -257
static const yytype_int16 yytable[] =
{
      63,   240,   297,   259,    97,   272,   304,   274,    77,   287,
     463,    63,    72,   470,   135,   287,   298,    65,    91,   540,
     541,   108,   280,   281,   282,   284,   287,   307,    65,   335,
     326,   306,   177,   335,   293,   330,   331,    70,   552,    52,
     293,    66,    92,    95,    96,   175,   416,    71,    70,   383,
     384,   293,    66,   268,   144,   268,   515,   268,    71,   161,
     261,   161,    63,   177,   625,   292,   134,   332,    93,   568,
     142,   292,   238,   239,   140,   244,    63,   638,   335,    65,
     145,   268,   132,    53,    77,   416,   419,   245,    72,   177,
      98,   146,   471,    65,    91,    99,   472,   177,   161,    70,
     100,   296,   137,    66,   107,   326,   109,    63,   110,    71,
     111,   383,   384,    70,   666,   255,   161,    66,   265,   112,
     271,   177,   271,    71,    65,   113,   246,   246,   416,   422,
     286,   697,   419,   114,   325,   305,   286,   161,   336,   565,
     161,    63,   413,    63,    70,    55,   271,   161,    66,   262,
     440,   142,   417,   302,    71,   140,   445,   416,    65,   303,
      65,   145,   383,   384,   516,   276,   277,   279,   279,   279,
     279,   162,   263,   162,   605,    52,   115,   166,    70,   333,
      70,   569,    66,   445,    66,   459,   116,   448,    71,   639,
      71,   418,   420,   464,   167,   168,   177,   149,   150,   151,
     152,   153,   154,   155,   156,   157,   158,   159,   160,   435,
     162,   117,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,   667,   436,   162,   416,
     118,   161,   379,   563,   421,   423,   119,   659,   424,   485,
     486,   509,   479,   698,   481,   482,   563,   484,   576,   162,
     498,   578,   162,   419,   287,   510,   441,   133,   582,   162,
     507,   378,   446,   447,   419,     4,   419,   120,   517,   518,
     508,   445,   445,   445,   121,   563,   445,   169,   122,   293,
     170,   171,   172,   123,   177,   443,   173,    57,   146,   449,
     177,   174,   692,   124,   554,   101,   613,   614,   102,   125,
     268,   126,   426,    52,    63,     4,    55,   287,   299,    93,
     292,   250,   251,   147,   254,   378,   302,   665,   268,   268,
     127,    65,   303,   128,   287,   400,   401,   402,   403,   404,
     405,   295,   293,   129,   148,   514,   558,   613,   614,   564,
     130,    70,   131,   162,   136,    66,   477,     4,   309,   293,
     163,    71,   566,    43,    44,   197,   181,    93,   273,   584,
     275,    55,   324,   292,    48,   271,   327,   328,    51,   329,
     585,   334,   606,   506,    94,   286,    97,   608,   626,   663,
     292,   683,   691,   271,   271,   654,   476,   478,    62,   279,
     279,   483,   279,    43,    44,   342,   343,   197,   402,   403,
     404,   405,   344,   345,    48,   346,    63,    63,    51,    52,
      53,   347,   348,   223,    94,   279,   177,   349,   350,   278,
     351,   352,   255,    65,    65,    63,   630,   353,   286,   354,
     355,   356,   411,   357,   358,    43,    44,   359,   360,   287,
     361,   414,    65,    70,    70,   286,    48,    66,    66,   362,
      51,   363,    53,    71,    71,   223,    94,   415,   364,   365,
     366,   278,    70,   367,   293,   368,    66,   369,   553,   370,
     371,   372,    71,   373,   374,   375,   409,   642,   430,   652,
     410,   279,   646,   432,   427,   428,   433,   229,   236,   237,
     434,   437,   438,   439,   450,   292,   452,   416,   453,   456,
     249,   249,   252,   249,   256,   252,   260,   266,   270,   473,
     480,   572,   660,   542,   677,   658,   581,   512,   513,   560,
     249,   567,   577,   579,   580,   583,   597,   287,   607,   609,
     405,   622,   268,   623,   635,   624,   636,   249,   310,   311,
     312,   637,   648,   313,   314,   315,   316,   317,   318,   319,
     321,   323,   293,   651,   656,   657,   671,   675,   679,   680,
     286,   398,   399,   400,   401,   402,   403,   404,   405,   684,
     685,   337,   338,   340,   341,   696,   702,   705,   714,   322,
     559,   308,   176,   292,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,   165,   271,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,   248,   377,   381,   634,    63,   655,   653,     0,
     551,    63,   279,   548,    63,     0,     0,     0,   140,     0,
       0,     0,     0,    65,   145,   693,     0,     0,    65,     0,
       0,    65,     0,     0,     0,     0,     0,     0,   286,     0,
       0,     0,    63,    70,     0,     0,     0,    66,    70,     0,
       0,    70,    66,    71,   302,    66,     0,   249,    71,    65,
     303,    71,     0,    57,   444,    59,     0,     0,     0,    63,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    70,
       0,     0,     0,    66,     0,     0,    65,     0,     0,    71,
     390,   391,   392,   393,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,    70,     0,     0,     0,
      66,   466,   468,   469,     0,     0,    71,     0,     0,     0,
       0,     0,   474,   475,     0,     0,     0,     0,     0,     0,
       0,   252,   252,   487,   488,   489,   490,   491,   492,   493,
     494,   495,   496,   497,     0,   499,   500,   501,   502,   503,
     504,   505,   252,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   519,   520,   521,   522,   523,   524,
     525,   526,   527,   528,   529,   530,   531,   532,   533,   534,
     535,   536,   537,   538,   539,     0,     0,     0,   543,   544,
     545,     0,     0,   546,   546,   549,   164,   103,   550,     0,
       0,   256,     2,     3,     0,     0,     0,     0,     0,     0,
       0,   555,   556,     0,     0,     0,     0,   557,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     4,
       0,     0,     0,   561,     0,     0,   562,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,   573,     0,     5,     6,     7,     8,     0,     0,     9,
      10,    11,    12,    13,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,     0,
       0,    29,    30,    31,    32,    33,    34,    35,    36,     0,
      37,    38,   610,     0,   611,     0,     0,     0,    39,    40,
      41,    42,     0,     0,     0,   170,   171,   172,     0,     0,
       0,   173,    57,     0,     0,     0,     0,    43,    44,     0,
       0,     0,     0,     0,    45,     0,    46,    47,    48,    49,
       0,    50,    51,    52,    53,    54,    55,     0,    56,    57,
      58,    59,    60,    61,     0,    62,     0,  -256,     0,     0,
       0,   640,     0,     0,   641,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   643,     0,     0,     0,
       0,     0,   644,   645,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   647,     0,     0,     0,     0,     0,     0,
       0,   649,   650,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   252,     0,   661,   662,     0,     0,
     664,     0,     0,     0,     0,     0,     0,   668,   669,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   138,     0,     0,     0,     0,     2,     3,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   688,     0,     0,   689,   690,     0,     0,     0,   550,
     694,   695,     0,     0,     4,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   706,   707,     0,     0,
       0,     0,   708,   709,     0,     0,     0,     0,     5,     6,
       7,     8,   715,   716,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,     0,     0,    29,    30,    31,    32,
      33,    34,    35,    36,     0,    37,    38,     0,     0,     0,
       0,     0,     0,    39,    40,    41,    42,     0,     0,     0,
       1,     0,     0,     0,     0,     2,     3,     0,     0,     0,
       0,     0,    43,    44,     0,     0,     0,     0,     0,    45,
       0,    46,    47,    48,    49,     0,    50,    51,    52,    53,
      54,    55,     4,    56,    57,    58,    59,    60,    61,     0,
      62,   139,  -256,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     5,     6,     7,     8,
       0,     0,     9,    10,    11,    12,    13,    14,    15,    16,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,     0,     0,    29,    30,    31,    32,    33,    34,
      35,    36,     0,    37,    38,     0,     0,     0,     0,     0,
       0,    39,    40,    41,    42,     0,     0,     0,   103,     0,
       0,     0,     0,     2,     3,     0,     0,     0,     0,     0,
      43,    44,     0,     0,     0,     0,     0,    45,     0,    46,
      47,    48,    49,     0,    50,    51,    52,    53,    54,    55,
       4,    56,    57,    58,    59,    60,    61,     0,    62,     0,
    -256,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     5,     6,     7,     8,     0,     0,
       9,    10,    11,    12,    13,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
       0,     0,    29,    30,    31,    32,    33,    34,    35,    36,
       0,    37,    38,     0,     0,     0,     0,     0,     0,    39,
      40,    41,    42,     0,     0,     0,   103,     0,     0,     0,
       0,     2,     3,     0,     0,     0,     0,     0,    43,    44,
       0,     0,     0,     0,     0,    45,     0,    46,    47,    48,
      49,     0,    50,    51,    52,    53,    54,    55,     4,    56,
      57,    58,    59,    60,    61,     0,    62,     0,  -256,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     5,     6,     7,     8,  -246,  -246,     9,    10,
      11,    12,   391,   392,   393,   394,   395,   396,   397,   398,
     399,   400,   401,   402,   403,   404,   405,    28,     0,     0,
      29,    30,    31,    32,    33,    34,    35,    36,     0,    37,
      38,     0,     0,     0,     0,     0,     0,    39,    40,    41,
      42,     0,     0,     0,   103,     0,     0,     0,     0,     2,
       3,     0,     0,     0,     0,     0,    43,    44,     0,     0,
       0,     0,     0,    45,     0,    46,    47,    48,    49,     0,
      50,    51,    52,    53,    54,    55,     4,    56,     0,     0,
       0,    60,    61,     0,    62,  -246,  -256,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       5,     6,     7,     8,     0,     0,     9,    10,    11,    12,
     392,   393,   394,   395,   396,   397,   398,   399,   400,   401,
     402,   403,   404,   405,     0,    28,     0,     0,    29,    30,
      31,    32,    33,    34,    35,    36,     0,    37,    38,     0,
       0,     0,     0,     0,     0,    39,    40,    41,    42,     0,
       0,     0,   103,     0,     0,     0,     0,     2,     3,     0,
       0,     0,     0,     0,    43,    44,     0,     0,   300,     0,
       0,    45,     0,    46,    47,    48,    49,     0,    50,    51,
      52,    53,    54,    55,     4,    56,     0,     0,     0,    60,
      61,     0,    62,   301,  -256,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     5,     6,
       7,     8,   613,   614,     9,    10,    11,    12,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    28,     0,     0,    29,    30,    31,    32,
      33,    34,    35,    36,     0,    37,    38,     0,     0,     0,
       0,     0,     0,    39,    40,    41,    42,     0,     0,     0,
     103,     0,     0,     0,     0,     2,     3,     0,     0,     0,
       0,     0,    43,    44,     0,     0,     0,     0,     0,    45,
       0,    46,    47,    48,    49,     0,    50,    51,    52,    53,
      54,    55,     4,    56,     0,     0,     0,    60,    61,     0,
      62,     0,  -256,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     5,     6,     7,     8,
       0,     0,     9,    10,    11,    12,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    28,     0,     0,    29,    30,    31,    32,    33,    34,
      35,    36,     0,    37,    38,     0,     0,     0,     0,     0,
       0,    39,    40,    41,    42,     0,     0,     0,   103,     0,
       0,     0,     0,     2,     3,     0,     0,     0,     0,     0,
      43,    44,     0,     0,     0,     0,     0,    45,     0,    46,
      47,    48,    49,     0,    50,    51,    52,    53,    54,    55,
       4,    56,     0,     0,     0,    60,    61,     0,    62,   451,
    -256,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     5,     6,     7,     8,     0,     0,
       9,    10,    11,    12,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    28,
       0,     0,    29,    30,    31,    32,    33,    34,    35,    36,
       0,    37,    38,     0,     0,     0,     0,     0,     0,    39,
      40,    41,    42,     0,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    43,    44,
       0,     0,     0,     2,     3,    45,     0,    46,    47,    48,
      49,     0,    50,    51,    52,    53,    54,    55,     0,    56,
       0,     0,     0,    60,    61,     0,    62,     0,  -256,   182,
       4,     0,     0,     0,   183,     0,   184,   185,     0,     0,
     186,     0,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   170,   171,   172,     0,     0,     0,   412,    57,
       0,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,     0,     0,     0,     0,     0,
     197,   198,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   199,   200,   201,   202,   203,   204,     0,
       0,     0,     0,   205,   206,   207,   208,   209,     0,   210,
     211,   212,   213,   214,   215,   216,   217,   218,    43,    44,
     219,   220,     2,     3,     0,     0,   221,     0,     0,    48,
     222,     0,     0,    51,    52,    53,    57,     0,   223,    94,
       0,     0,     0,     0,   224,     0,   225,   467,   182,   376,
       0,     0,     0,   183,     0,   184,   185,     0,     0,   186,
       0,   187,   188,   189,   190,   191,   192,   193,   194,   195,
     196,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    14,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,     0,   197,
     198,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   199,   200,   201,   202,   203,   204,     0,     0,
       0,     0,   205,   206,   207,   208,   209,     0,   210,   211,
     212,   213,   214,   215,   216,   217,   218,    43,    44,   219,
     220,     2,     3,     0,     0,   221,     0,     0,    48,   222,
       0,     0,    51,    52,    53,     0,     0,   223,    94,    57,
       0,    59,     0,   224,     0,   225,     0,   182,     4,     0,
       0,     0,   183,     0,   184,   185,     0,     0,   186,     0,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,     0,   197,   198,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   199,   200,   201,   202,   203,   204,     0,     0,     0,
       0,   205,   206,   207,   208,   209,     0,   210,   211,   212,
     213,   214,   215,   216,   217,   218,    43,    44,   219,   220,
       0,     0,     0,     0,   221,     2,     3,    48,   222,     0,
       0,    51,    52,    53,     0,     0,   223,    94,    57,     0,
       0,   261,   224,     0,   225,     0,     0,     0,     0,     0,
       0,   182,     4,     0,     0,     0,   183,     0,   184,   185,
       0,     0,   186,     0,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   385,     0,   386,   387,   388,   389,
     390,   391,   392,   393,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,     0,     0,     0,     0,
       0,     0,   197,   198,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   199,   200,   201,   202,   203,
     204,     0,     0,     0,     0,   205,   206,   207,   208,   209,
     262,   210,   211,   212,   213,   214,   215,   216,   217,   218,
      43,    44,   219,   220,     0,     0,     2,     3,   221,     0,
       0,    48,   222,   263,     0,    51,    52,    53,     0,     0,
     223,    94,   285,     0,     0,     0,   224,   264,   225,     0,
       0,     0,   182,     4,     0,     0,     0,   183,     0,   184,
     185,     0,     0,   186,     0,   187,   188,   189,   190,   191,
     192,   193,   194,   195,   196,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     621,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   197,   198,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   199,   200,   201,   202,
     203,   204,     0,     0,     0,     0,   205,   206,   207,   208,
     209,     0,   210,   211,   212,   213,   214,   215,   216,   217,
     218,    43,    44,   219,   220,     2,     3,     0,     0,   221,
       0,     0,    48,   222,     0,     0,    51,    52,    53,     0,
      55,   223,    94,     0,     0,     0,     0,   224,     0,   225,
       0,   182,     4,     0,     0,     0,   183,     0,   184,   185,
       0,     0,   186,     0,   187,   188,   189,   190,   191,   192,
     193,   194,   195,   196,   385,   457,   386,   387,   388,   389,
     390,   391,   392,   393,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,     0,     0,     0,     0,
       0,     0,   197,   198,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   199,   200,   201,   202,   203,
     204,     0,     0,     0,     0,   205,   206,   207,   208,   209,
       0,   210,   211,   212,   213,   214,   215,   216,   217,   218,
      43,    44,   219,   220,     2,     3,     0,     0,   221,     0,
       0,    48,   222,     0,     0,    51,    52,    53,     0,     0,
     223,    94,     0,     0,     0,     0,   224,   269,   225,     0,
     182,     4,     0,     0,     0,   183,     0,   184,   185,     0,
       0,   186,     0,   187,   188,   189,   190,   191,   192,   193,
     194,   195,   196,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   458,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   197,   198,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   199,   200,   201,   202,   203,   204,
       0,     0,     0,     0,   205,   206,   207,   208,   209,     0,
     210,   211,   212,   213,   214,   215,   216,   217,   218,    43,
      44,   219,   220,     2,     3,     0,     0,   221,     0,     0,
      48,   222,     0,     0,    51,    52,    53,     0,     0,   223,
      94,     0,     0,     0,     0,   224,   294,   225,     0,   182,
       4,     0,     0,     0,   183,   465,   184,   185,     0,     0,
     186,     0,   187,   188,   189,   190,   191,   192,   193,   194,
     195,   196,   385,   631,   386,   387,   388,   389,   390,   391,
     392,   393,   394,   395,   396,   397,   398,   399,   400,   401,
     402,   403,   404,   405,     0,     0,     0,     0,     0,     0,
     197,   198,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   199,   200,   201,   202,   203,   204,     0,
       0,     0,     0,   205,   206,   207,   208,   209,     0,   210,
     211,   212,   213,   214,   215,   216,   217,   218,    43,    44,
     219,   220,     2,     3,     0,     0,   221,     0,     0,    48,
     222,     0,     0,    51,    52,    53,     0,     0,   223,    94,
       0,     0,     0,     0,   224,     0,   225,     0,   182,     4,
       0,     0,     0,   183,     0,   184,   185,     0,     0,   186,
       0,   187,   188,   189,   190,   191,   192,   193,   194,   195,
     196,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   633,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   197,
     198,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   199,   200,   201,   202,   203,   204,     0,     0,
       0,     0,   205,   206,   207,   208,   209,     0,   210,   211,
     212,   213,   214,   215,   216,   217,   218,    43,    44,   219,
     220,     2,     3,     0,     0,   221,     0,     0,    48,   222,
       0,     0,    51,    52,    53,     0,     0,   223,    94,     0,
       0,     0,     0,   224,     0,   225,     0,   182,     4,     0,
       0,     0,   183,     0,   184,   185,     0,     0,   186,     0,
     187,   188,   189,   190,   191,   192,   193,   194,   195,   196,
     385,     0,   386,   387,   388,   389,   390,   391,   392,   393,
     394,   395,   396,   397,   398,   399,   400,   401,   402,   403,
     404,   405,     2,     3,     0,     0,     0,     0,   197,   198,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   199,   200,   201,   202,   203,   204,     0,     0,     4,
       0,   205,   206,   207,   208,   209,     0,   210,   211,   212,
     213,   214,   215,   216,   217,   218,    43,    44,   219,   220,
       0,     0,     0,     0,   221,     7,     0,    48,   442,     9,
      10,    51,    52,    53,     0,     0,   223,    94,     0,     0,
       0,     0,   224,     0,   225,     0,     0,     0,    28,     0,
       0,    29,    30,    31,    32,    33,    34,    35,     0,     0,
      37,    38,     0,     0,     0,     0,     0,     0,    39,    40,
      41,    42,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   571,     0,    43,    44,     0,
       0,     0,     0,     0,    45,     0,    46,    47,    48,    49,
       0,    50,    51,    52,    53,     0,    55,     0,    94,     0,
       0,     0,    60,    61,   385,     0,   386,   387,   388,   389,
     390,   391,   392,   393,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,   385,     0,   386,   387,
     388,   389,   390,   391,   392,   393,   394,   395,   396,   397,
     398,   399,   400,   401,   402,   403,   404,   405,   385,     0,
     386,   387,   388,   389,   390,   391,   392,   393,   394,   395,
     396,   397,   398,   399,   400,   401,   402,   403,   404,   405,
     385,     0,   386,   387,   388,   389,   390,   391,   392,   393,
     394,   395,   396,   397,   398,   399,   400,   401,   402,   403,
     404,   405,   385,     0,   386,   387,   388,   389,   390,   391,
     392,   393,   394,   395,   396,   397,   398,   399,   400,   401,
     402,   403,   404,   405,   385,     0,   386,   387,   388,   389,
     390,   391,   392,   393,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,     0,     0,   385,   687,
     386,   387,   388,   389,   390,   391,   392,   393,   394,   395,
     396,   397,   398,   399,   400,   401,   402,   403,   404,   405,
     385,   711,   386,   387,   388,   389,   390,   391,   392,   393,
     394,   395,   396,   397,   398,   399,   400,   401,   402,   403,
     404,   405,     0,   718,   388,   389,   390,   391,   392,   393,
     394,   395,   396,   397,   398,   399,   400,   401,   402,   403,
     404,   405,   385,   406,   386,   387,   388,   389,   390,   391,
     392,   393,   394,   395,   396,   397,   398,   399,   400,   401,
     402,   403,   404,   405,   385,   407,   386,   387,   388,   389,
     390,   391,   392,   393,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,   385,   408,   386,   387,
     388,   389,   390,   391,   392,   393,   394,   395,   396,   397,
     398,   399,   400,   401,   402,   403,   404,   405,     0,     0,
     385,   425,   386,   387,   388,   389,   390,   391,   392,   393,
     394,   395,   396,   397,   398,   399,   400,   401,   402,   403,
     404,   405,   385,   429,   386,   387,   388,   389,   390,   391,
     392,   393,   394,   395,   396,   397,   398,   399,   400,   401,
     402,   403,   404,   405,   386,   387,   388,   389,   390,   391,
     392,   393,   394,   395,   396,   397,   398,   399,   400,   401,
     402,   403,   404,   405,   385,   431,   386,   387,   388,   389,
     390,   391,   392,   393,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,   385,   511,   386,   387,
     388,   389,   390,   391,   392,   393,   394,   395,   396,   397,
     398,   399,   400,   401,   402,   403,   404,   405,   385,   574,
     386,   387,   388,   389,   390,   391,   392,   393,   394,   395,
     396,   397,   398,   399,   400,   401,   402,   403,   404,   405,
       0,     0,   385,   586,   386,   387,   388,   389,   390,   391,
     392,   393,   394,   395,   396,   397,   398,   399,   400,   401,
     402,   403,   404,   405,   385,   588,   386,   387,   388,   389,
     390,   391,   392,   393,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,   387,   388,   389,   390,
     391,   392,   393,   394,   395,   396,   397,   398,   399,   400,
     401,   402,   403,   404,   405,     0,   385,   589,   386,   387,
     388,   389,   390,   391,   392,   393,   394,   395,   396,   397,
     398,   399,   400,   401,   402,   403,   404,   405,   385,   590,
     386,   387,   388,   389,   390,   391,   392,   393,   394,   395,
     396,   397,   398,   399,   400,   401,   402,   403,   404,   405,
     385,   591,   386,   387,   388,   389,   390,   391,   392,   393,
     394,   395,   396,   397,   398,   399,   400,   401,   402,   403,
     404,   405,     0,     0,   385,   592,   386,   387,   388,   389,
     390,   391,   392,   393,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,   385,   595,   386,   387,
     388,   389,   390,   391,   392,   393,   394,   395,   396,   397,
     398,   399,   400,   401,   402,   403,   404,   405,   389,   390,
     391,   392,   393,   394,   395,   396,   397,   398,   399,   400,
     401,   402,   403,   404,   405,     0,     0,     0,   385,   596,
     386,   387,   388,   389,   390,   391,   392,   393,   394,   395,
     396,   397,   398,   399,   400,   401,   402,   403,   404,   405,
     385,   598,   386,   387,   388,   389,   390,   391,   392,   393,
     394,   395,   396,   397,   398,   399,   400,   401,   402,   403,
     404,   405,   385,   599,   386,   387,   388,   389,   390,   391,
     392,   393,   394,   395,   396,   397,   398,   399,   400,   401,
     402,   403,   404,   405,     0,     0,   385,   600,   386,   387,
     388,   389,   390,   391,   392,   393,   394,   395,   396,   397,
     398,   399,   400,   401,   402,   403,   404,   405,   385,   601,
     386,   387,   388,   389,   390,   391,   392,   393,   394,   395,
     396,   397,   398,   399,   400,   401,   402,   403,   404,   405,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     385,   602,   386,   387,   388,   389,   390,   391,   392,   393,
     394,   395,   396,   397,   398,   399,   400,   401,   402,   403,
     404,   405,   385,   603,   386,   387,   388,   389,   390,   391,
     392,   393,   394,   395,   396,   397,   398,   399,   400,   401,
     402,   403,   404,   405,   385,   619,   386,   387,   388,   389,
     390,   391,   392,   393,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,     0,     0,   385,   620,
     386,   387,   388,   389,   390,   391,   392,   393,   394,   395,
     396,   397,   398,   399,   400,   401,   402,   403,   404,   405,
     385,   629,   386,   387,   388,   389,   390,   391,   392,   393,
     394,   395,   396,   397,   398,   399,   400,   401,   402,   403,
     404,   405,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   385,   670,   386,   387,   388,   389,   390,   391,
     392,   393,   394,   395,   396,   397,   398,   399,   400,   401,
     402,   403,   404,   405,   385,   673,   386,   387,   388,   389,
     390,   391,   392,   393,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,   385,   674,   386,   387,
     388,   389,   390,   391,   392,   393,   394,   395,   396,   397,
     398,   399,   400,   401,   402,   403,   404,   405,   241,     0,
       0,   699,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   167,   168,     0,     0,     0,
       0,     0,     0,   700,     0,     0,     0,     0,     0,     0,
       0,     0,     0,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   385,   701,   386,   387,   388,   389,
     390,   391,   392,   393,   394,   395,   396,   397,   398,   399,
     400,   401,   402,   403,   404,   405,     0,   712,     0,     0,
       0,   575,     0,     0,     0,     0,     0,     0,   169,     0,
       0,   170,   171,   172,     0,     0,     0,   173,    57,   717,
       0,   385,   174,   386,   387,   388,   389,   390,   391,   392,
     393,   394,   395,   396,   397,   398,   399,   400,   401,   402,
     403,   404,   405,     0,     0,     0,     0,   385,   587,   386,
     387,   388,   389,   390,   391,   392,   393,   394,   395,   396,
     397,   398,   399,   400,   401,   402,   403,   404,   405,     0,
       0,     0,     0,   385,   593,   386,   387,   388,   389,   390,
     391,   392,   393,   394,   395,   396,   397,   398,   399,   400,
     401,   402,   403,   404,   405,     0,     0,     0,     0,   385,
     594,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,     0,     0,     0,     0,   385,   604,   386,   387,   388,
     389,   390,   391,   392,   393,   394,   395,   396,   397,   398,
     399,   400,   401,   402,   403,   404,   405,     0,     0,     0,
       0,   385,   627,   386,   387,   388,   389,   390,   391,   392,
     393,   394,   395,   396,   397,   398,   399,   400,   401,   402,
     403,   404,   405,     0,     0,     0,     0,   385,   628,   386,
     387,   388,   389,   390,   391,   392,   393,   394,   395,   396,
     397,   398,   399,   400,   401,   402,   403,   404,   405,     0,
       0,     0,     0,   385,   672,   386,   387,   388,   389,   390,
     391,   392,   393,   394,   395,   396,   397,   398,   399,   400,
     401,   402,   403,   404,   405,     0,     0,     0,     0,   385,
     676,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,     0,     0,     0,     0,   385,   681,   386,   387,   388,
     389,   390,   391,   392,   393,   394,   395,   396,   397,   398,
     399,   400,   401,   402,   403,   404,   405,     0,     0,     0,
       0,   385,   682,   386,   387,   388,   389,   390,   391,   392,
     393,   394,   395,   396,   397,   398,   399,   400,   401,   402,
     403,   404,   405,     0,     0,     0,     0,   385,   703,   386,
     387,   388,   389,   390,   391,   392,   393,   394,   395,   396,
     397,   398,   399,   400,   401,   402,   403,   404,   405,     0,
       0,     0,     0,   385,   704,   386,   387,   388,   389,   390,
     391,   392,   393,   394,   395,   396,   397,   398,   399,   400,
     401,   402,   403,   404,   405,     0,     0,     0,     0,     0,
     713,   385,   454,   386,   387,   388,   389,   390,   391,   392,
     393,   394,   395,   396,   397,   398,   399,   400,   401,   402,
     403,   404,   405,     0,   455,   385,   631,   386,   387,   388,
     389,   390,   391,   392,   393,   394,   395,   396,   397,   398,
     399,   400,   401,   402,   403,   404,   405,   385,   632,   386,
     387,   388,   389,   390,   391,   392,   393,   394,   395,   396,
     397,   398,   399,   400,   401,   402,   403,   404,   405,   385,
     570,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405,   385,   686,   386,   387,   388,   389,   390,   391,   392,
     393,   394,   395,   396,   397,   398,   399,   400,   401,   402,
     403,   404,   405,     0,   710,   385,   612,   386,   387,   388,
     389,   390,   391,   392,   393,   394,   395,   396,   397,   398,
     399,   400,   401,   402,   403,   404,   405,   385,   678,   386,
     387,   388,   389,   390,   391,   392,   393,   394,   395,   396,
     397,   398,   399,   400,   401,   402,   403,   404,   405,   385,
       0,   386,   387,   388,   389,   390,   391,   392,   393,   394,
     395,   396,   397,   398,   399,   400,   401,   402,   403,   404,
     405
};

static const yytype_int16 yycheck[] =
{
       0,   107,   136,   116,     4,   120,   143,   122,     0,   130,
     328,    11,     0,   335,    59,   136,   137,     0,     0,   406,
     407,    13,   126,   127,   128,   129,   147,    22,    11,    42,
     166,   146,    77,    42,   130,   175,   176,     0,   425,   137,
     136,     0,   125,     2,     3,    77,    42,     0,    11,    23,
      24,   147,    11,   118,    62,   120,    42,   122,    11,    37,
      22,    37,    62,   108,    42,   130,    58,    37,   151,    37,
      62,   136,    57,    58,    62,   108,    76,    37,    42,    62,
      62,   146,    16,   138,    76,    42,    42,   142,    76,   134,
     147,     3,    38,    76,    76,   147,    42,   142,    37,    62,
     147,   134,    33,    62,   147,   241,   147,   107,   147,    62,
     147,    23,    24,    76,    37,   115,    37,    76,   118,   147,
     120,   166,   122,    76,   107,   147,   109,   110,    42,    42,
     130,    37,    42,   147,   166,   143,   136,    37,   151,   461,
      37,   141,   151,   143,   107,   140,   146,    37,   107,   111,
      42,   143,   148,   141,   107,   143,    42,    42,   141,   141,
     143,   143,    23,    24,   150,   124,   125,   126,   127,   128,
     129,   149,   134,   149,   148,   137,   147,    33,   141,   149,
     143,   149,   141,    42,   143,   325,   147,   151,   141,   149,
     143,   148,   148,   329,    50,    51,   241,     3,     4,     5,
       6,     7,     8,     9,    10,    11,    12,    13,    14,   148,
     149,   147,    68,    69,    70,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,   149,   148,   149,    42,
     147,    37,   224,    42,   148,   148,   147,   624,   148,   352,
     353,   375,   346,   149,   348,   349,    42,   351,   148,   149,
     365,   148,   149,    42,   375,   376,   148,    16,   148,   149,
     373,   224,   148,   148,    42,    33,    42,   147,   383,   384,
     374,    42,    42,    42,   147,    42,    42,   133,   147,   375,
     136,   137,   138,   147,   329,   285,   142,   143,     3,   148,
     335,   147,   679,   147,   428,   139,    61,    62,   142,   147,
     365,   147,   261,   137,   304,    33,   140,   428,   150,   151,
     375,   111,   112,     3,   114,   278,   304,   635,   383,   384,
     147,   304,   304,   147,   445,    31,    32,    33,    34,    35,
      36,   131,   428,   147,     3,   148,   440,    61,    62,   148,
     147,   304,   147,   149,   147,   304,   346,    33,   148,   445,
     125,   304,   148,   121,   122,    83,   151,   151,   148,   148,
     133,   140,   125,   428,   132,   365,   147,   147,   136,    33,
     148,     3,   148,   373,   142,   375,   376,   148,   148,   148,
     445,   148,   148,   383,   384,   150,   345,   346,   149,   348,
     349,   350,   351,   121,   122,   147,   147,    83,    33,    34,
      35,    36,   147,   147,   132,   147,   406,   407,   136,   137,
     138,   147,   147,   141,   142,   374,   461,   147,   147,   147,
     147,   147,   422,   406,   407,   425,   560,   147,   428,   147,
     147,   147,   151,   147,   147,   121,   122,   147,   147,   560,
     147,    42,   425,   406,   407,   445,   132,   406,   407,   147,
     136,   147,   138,   406,   407,   141,   142,    42,   147,   147,
     147,   147,   425,   147,   560,   147,   425,   147,   427,   147,
     147,   147,   425,   147,   147,   147,   147,   581,   148,   616,
     147,   440,   597,    42,   147,   147,    42,    98,    99,   100,
     148,   148,    42,   148,   148,   560,   133,    42,    38,   150,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   148,
     148,     3,   625,   149,   648,   621,    42,   148,   148,   147,
     131,   148,   148,   148,   148,   148,    42,   648,   148,   148,
      36,   148,   597,   148,   147,    93,    38,   148,   149,   150,
     151,   150,   147,   154,   155,   156,   157,   158,   159,   160,
     161,   162,   648,    16,   151,   151,   148,    42,   148,    42,
     560,    29,    30,    31,    32,    33,    34,    35,    36,    38,
     150,   182,   183,   184,   185,   149,   148,   150,   149,   162,
     445,   147,    77,   648,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    76,   597,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    81,   110,   224,   225,   563,   616,   618,   616,    -1,
     422,   621,   581,   415,   624,    -1,    -1,    -1,   616,    -1,
      -1,    -1,    -1,   616,   616,   680,    -1,    -1,   621,    -1,
      -1,   624,    -1,    -1,    -1,    -1,    -1,    -1,   648,    -1,
      -1,    -1,   652,   616,    -1,    -1,    -1,   616,   621,    -1,
      -1,   624,   621,   616,   652,   624,    -1,   278,   621,   652,
     652,   624,    -1,   143,   285,   145,    -1,    -1,    -1,   679,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   652,
      -1,    -1,    -1,   652,    -1,    -1,   679,    -1,    -1,   652,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,   679,    -1,    -1,    -1,
     679,   332,   333,   334,    -1,    -1,   679,    -1,    -1,    -1,
      -1,    -1,   343,   344,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   352,   353,   354,   355,   356,   357,   358,   359,   360,
     361,   362,   363,   364,    -1,   366,   367,   368,   369,   370,
     371,   372,   373,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   385,   386,   387,   388,   389,   390,
     391,   392,   393,   394,   395,   396,   397,   398,   399,   400,
     401,   402,   403,   404,   405,    -1,    -1,    -1,   409,   410,
     411,    -1,    -1,   414,   415,   416,     0,     1,   419,    -1,
      -1,   422,     6,     7,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   432,   433,    -1,    -1,    -1,    -1,   438,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    33,
      -1,    -1,    -1,   454,    -1,    -1,   457,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,   472,    -1,    57,    58,    59,    60,    -1,    -1,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    -1,
      -1,    85,    86,    87,    88,    89,    90,    91,    92,    -1,
      94,    95,   513,    -1,   515,    -1,    -1,    -1,   102,   103,
     104,   105,    -1,    -1,    -1,   136,   137,   138,    -1,    -1,
      -1,   142,   143,    -1,    -1,    -1,    -1,   121,   122,    -1,
      -1,    -1,    -1,    -1,   128,    -1,   130,   131,   132,   133,
      -1,   135,   136,   137,   138,   139,   140,    -1,   142,   143,
     144,   145,   146,   147,    -1,   149,    -1,   151,    -1,    -1,
      -1,   572,    -1,    -1,   575,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   587,    -1,    -1,    -1,
      -1,    -1,   593,   594,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   604,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   612,   613,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   625,    -1,   627,   628,    -1,    -1,
     631,    -1,    -1,    -1,    -1,    -1,    -1,   638,   639,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,     1,    -1,    -1,    -1,    -1,     6,     7,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   672,    -1,    -1,   675,   676,    -1,    -1,    -1,   680,
     681,   682,    -1,    -1,    33,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   697,   698,    -1,    -1,
      -1,    -1,   703,   704,    -1,    -1,    -1,    -1,    57,    58,
      59,    60,   713,   714,    63,    64,    65,    66,    67,    68,
      69,    70,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    -1,    -1,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    94,    95,    -1,    -1,    -1,
      -1,    -1,    -1,   102,   103,   104,   105,    -1,    -1,    -1,
       1,    -1,    -1,    -1,    -1,     6,     7,    -1,    -1,    -1,
      -1,    -1,   121,   122,    -1,    -1,    -1,    -1,    -1,   128,
      -1,   130,   131,   132,   133,    -1,   135,   136,   137,   138,
     139,   140,    33,   142,   143,   144,   145,   146,   147,    -1,
     149,   150,   151,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    57,    58,    59,    60,
      -1,    -1,    63,    64,    65,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    -1,    -1,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    94,    95,    -1,    -1,    -1,    -1,    -1,
      -1,   102,   103,   104,   105,    -1,    -1,    -1,     1,    -1,
      -1,    -1,    -1,     6,     7,    -1,    -1,    -1,    -1,    -1,
     121,   122,    -1,    -1,    -1,    -1,    -1,   128,    -1,   130,
     131,   132,   133,    -1,   135,   136,   137,   138,   139,   140,
      33,   142,   143,   144,   145,   146,   147,    -1,   149,    -1,
     151,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    57,    58,    59,    60,    -1,    -1,
      63,    64,    65,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      -1,    -1,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    94,    95,    -1,    -1,    -1,    -1,    -1,    -1,   102,
     103,   104,   105,    -1,    -1,    -1,     1,    -1,    -1,    -1,
      -1,     6,     7,    -1,    -1,    -1,    -1,    -1,   121,   122,
      -1,    -1,    -1,    -1,    -1,   128,    -1,   130,   131,   132,
     133,    -1,   135,   136,   137,   138,   139,   140,    33,   142,
     143,   144,   145,   146,   147,    -1,   149,    -1,   151,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    82,    -1,    -1,
      85,    86,    87,    88,    89,    90,    91,    92,    -1,    94,
      95,    -1,    -1,    -1,    -1,    -1,    -1,   102,   103,   104,
     105,    -1,    -1,    -1,     1,    -1,    -1,    -1,    -1,     6,
       7,    -1,    -1,    -1,    -1,    -1,   121,   122,    -1,    -1,
      -1,    -1,    -1,   128,    -1,   130,   131,   132,   133,    -1,
     135,   136,   137,   138,   139,   140,    33,   142,    -1,    -1,
      -1,   146,   147,    -1,   149,   150,   151,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      57,    58,    59,    60,    -1,    -1,    63,    64,    65,    66,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    -1,    82,    -1,    -1,    85,    86,
      87,    88,    89,    90,    91,    92,    -1,    94,    95,    -1,
      -1,    -1,    -1,    -1,    -1,   102,   103,   104,   105,    -1,
      -1,    -1,     1,    -1,    -1,    -1,    -1,     6,     7,    -1,
      -1,    -1,    -1,    -1,   121,   122,    -1,    -1,   125,    -1,
      -1,   128,    -1,   130,   131,   132,   133,    -1,   135,   136,
     137,   138,   139,   140,    33,   142,    -1,    -1,    -1,   146,
     147,    -1,   149,   150,   151,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    57,    58,
      59,    60,    61,    62,    63,    64,    65,    66,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    82,    -1,    -1,    85,    86,    87,    88,
      89,    90,    91,    92,    -1,    94,    95,    -1,    -1,    -1,
      -1,    -1,    -1,   102,   103,   104,   105,    -1,    -1,    -1,
       1,    -1,    -1,    -1,    -1,     6,     7,    -1,    -1,    -1,
      -1,    -1,   121,   122,    -1,    -1,    -1,    -1,    -1,   128,
      -1,   130,   131,   132,   133,    -1,   135,   136,   137,   138,
     139,   140,    33,   142,    -1,    -1,    -1,   146,   147,    -1,
     149,    -1,   151,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    57,    58,    59,    60,
      -1,    -1,    63,    64,    65,    66,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    82,    -1,    -1,    85,    86,    87,    88,    89,    90,
      91,    92,    -1,    94,    95,    -1,    -1,    -1,    -1,    -1,
      -1,   102,   103,   104,   105,    -1,    -1,    -1,     1,    -1,
      -1,    -1,    -1,     6,     7,    -1,    -1,    -1,    -1,    -1,
     121,   122,    -1,    -1,    -1,    -1,    -1,   128,    -1,   130,
     131,   132,   133,    -1,   135,   136,   137,   138,   139,   140,
      33,   142,    -1,    -1,    -1,   146,   147,    -1,   149,   150,
     151,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    57,    58,    59,    60,    -1,    -1,
      63,    64,    65,    66,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    82,
      -1,    -1,    85,    86,    87,    88,    89,    90,    91,    92,
      -1,    94,    95,    -1,    -1,    -1,    -1,    -1,    -1,   102,
     103,   104,   105,    -1,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,   121,   122,
      -1,    -1,    -1,     6,     7,   128,    -1,   130,   131,   132,
     133,    -1,   135,   136,   137,   138,   139,   140,    -1,   142,
      -1,    -1,    -1,   146,   147,    -1,   149,    -1,   151,    32,
      33,    -1,    -1,    -1,    37,    -1,    39,    40,    -1,    -1,
      43,    -1,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,   136,   137,   138,    -1,    -1,    -1,   142,   143,
      -1,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    -1,    -1,    -1,    -1,    -1,
      83,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    96,    97,    98,    99,   100,   101,    -1,
      -1,    -1,    -1,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,     6,     7,    -1,    -1,   129,    -1,    -1,   132,
     133,    -1,    -1,   136,   137,   138,   143,    -1,   141,   142,
      -1,    -1,    -1,    -1,   147,    -1,   149,   150,    32,    33,
      -1,    -1,    -1,    37,    -1,    39,    40,    -1,    -1,    43,
      -1,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    -1,    83,
      84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    96,    97,    98,    99,   100,   101,    -1,    -1,
      -1,    -1,   106,   107,   108,   109,   110,    -1,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,     6,     7,    -1,    -1,   129,    -1,    -1,   132,   133,
      -1,    -1,   136,   137,   138,    -1,    -1,   141,   142,   143,
      -1,   145,    -1,   147,    -1,   149,    -1,    32,    33,    -1,
      -1,    -1,    37,    -1,    39,    40,    -1,    -1,    43,    -1,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    -1,    83,    84,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    96,    97,    98,    99,   100,   101,    -1,    -1,    -1,
      -1,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
      -1,    -1,    -1,    -1,   129,     6,     7,   132,   133,    -1,
      -1,   136,   137,   138,    -1,    -1,   141,   142,   143,    -1,
      -1,    22,   147,    -1,   149,    -1,    -1,    -1,    -1,    -1,
      -1,    32,    33,    -1,    -1,    -1,    37,    -1,    39,    40,
      -1,    -1,    43,    -1,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    15,    -1,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,
      -1,    -1,    83,    84,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    96,    97,    98,    99,   100,
     101,    -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,
     111,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,    -1,    -1,     6,     7,   129,    -1,
      -1,   132,   133,   134,    -1,   136,   137,   138,    -1,    -1,
     141,   142,    22,    -1,    -1,    -1,   147,   148,   149,    -1,
      -1,    -1,    32,    33,    -1,    -1,    -1,    37,    -1,    39,
      40,    -1,    -1,    43,    -1,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     151,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    83,    84,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    96,    97,    98,    99,
     100,   101,    -1,    -1,    -1,    -1,   106,   107,   108,   109,
     110,    -1,   112,   113,   114,   115,   116,   117,   118,   119,
     120,   121,   122,   123,   124,     6,     7,    -1,    -1,   129,
      -1,    -1,   132,   133,    -1,    -1,   136,   137,   138,    -1,
     140,   141,   142,    -1,    -1,    -1,    -1,   147,    -1,   149,
      -1,    32,    33,    -1,    -1,    -1,    37,    -1,    39,    40,
      -1,    -1,    43,    -1,    45,    46,    47,    48,    49,    50,
      51,    52,    53,    54,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,
      -1,    -1,    83,    84,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    96,    97,    98,    99,   100,
     101,    -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,
      -1,   112,   113,   114,   115,   116,   117,   118,   119,   120,
     121,   122,   123,   124,     6,     7,    -1,    -1,   129,    -1,
      -1,   132,   133,    -1,    -1,   136,   137,   138,    -1,    -1,
     141,   142,    -1,    -1,    -1,    -1,   147,   148,   149,    -1,
      32,    33,    -1,    -1,    -1,    37,    -1,    39,    40,    -1,
      -1,    43,    -1,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   150,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    83,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    96,    97,    98,    99,   100,   101,
      -1,    -1,    -1,    -1,   106,   107,   108,   109,   110,    -1,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   121,
     122,   123,   124,     6,     7,    -1,    -1,   129,    -1,    -1,
     132,   133,    -1,    -1,   136,   137,   138,    -1,    -1,   141,
     142,    -1,    -1,    -1,    -1,   147,   148,   149,    -1,    32,
      33,    -1,    -1,    -1,    37,    38,    39,    40,    -1,    -1,
      43,    -1,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    -1,    -1,    -1,    -1,    -1,    -1,
      83,    84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    96,    97,    98,    99,   100,   101,    -1,
      -1,    -1,    -1,   106,   107,   108,   109,   110,    -1,   112,
     113,   114,   115,   116,   117,   118,   119,   120,   121,   122,
     123,   124,     6,     7,    -1,    -1,   129,    -1,    -1,   132,
     133,    -1,    -1,   136,   137,   138,    -1,    -1,   141,   142,
      -1,    -1,    -1,    -1,   147,    -1,   149,    -1,    32,    33,
      -1,    -1,    -1,    37,    -1,    39,    40,    -1,    -1,    43,
      -1,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   150,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    83,
      84,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    96,    97,    98,    99,   100,   101,    -1,    -1,
      -1,    -1,   106,   107,   108,   109,   110,    -1,   112,   113,
     114,   115,   116,   117,   118,   119,   120,   121,   122,   123,
     124,     6,     7,    -1,    -1,   129,    -1,    -1,   132,   133,
      -1,    -1,   136,   137,   138,    -1,    -1,   141,   142,    -1,
      -1,    -1,    -1,   147,    -1,   149,    -1,    32,    33,    -1,
      -1,    -1,    37,    -1,    39,    40,    -1,    -1,    43,    -1,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      15,    -1,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,     6,     7,    -1,    -1,    -1,    -1,    83,    84,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    96,    97,    98,    99,   100,   101,    -1,    -1,    33,
      -1,   106,   107,   108,   109,   110,    -1,   112,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
      -1,    -1,    -1,    -1,   129,    59,    -1,   132,   133,    63,
      64,   136,   137,   138,    -1,    -1,   141,   142,    -1,    -1,
      -1,    -1,   147,    -1,   149,    -1,    -1,    -1,    82,    -1,
      -1,    85,    86,    87,    88,    89,    90,    91,    -1,    -1,
      94,    95,    -1,    -1,    -1,    -1,    -1,    -1,   102,   103,
     104,   105,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   150,    -1,   121,   122,    -1,
      -1,    -1,    -1,    -1,   128,    -1,   130,   131,   132,   133,
      -1,   135,   136,   137,   138,    -1,   140,    -1,   142,    -1,
      -1,    -1,   146,   147,    15,    -1,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    15,    -1,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    15,    -1,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      15,    -1,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    15,    -1,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    15,    -1,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    -1,    -1,    15,   150,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      15,   150,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    -1,   150,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    15,   148,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    15,   148,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    15,   148,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    -1,    -1,
      15,   148,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    15,   148,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    15,   148,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    15,   148,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    15,   148,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      -1,    -1,    15,   148,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    15,   148,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    -1,    15,   148,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    15,   148,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      15,   148,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    -1,    -1,    15,   148,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    15,   148,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    -1,    -1,    -1,    15,   148,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      15,   148,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    15,   148,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    -1,    -1,    15,   148,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    15,   148,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      15,   148,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    15,   148,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    15,   148,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    -1,    -1,    15,   148,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      15,   148,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    15,   148,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    15,   148,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    15,   148,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    33,    -1,
      -1,   148,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    50,    51,    -1,    -1,    -1,
      -1,    -1,    -1,   148,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    68,    69,    70,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    15,   148,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    -1,   148,    -1,    -1,
      -1,    42,    -1,    -1,    -1,    -1,    -1,    -1,   133,    -1,
      -1,   136,   137,   138,    -1,    -1,    -1,   142,   143,   148,
      -1,    15,   147,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    -1,    -1,    -1,    -1,    15,    42,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    -1,
      -1,    -1,    -1,    15,    42,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,    15,
      42,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    -1,    -1,    -1,    -1,    15,    42,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    -1,    -1,    -1,
      -1,    15,    42,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    -1,    -1,    -1,    -1,    15,    42,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    -1,
      -1,    -1,    -1,    15,    42,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,    15,
      42,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    -1,    -1,    -1,    -1,    15,    42,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    -1,    -1,    -1,
      -1,    15,    42,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    -1,    -1,    -1,    -1,    15,    42,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    -1,
      -1,    -1,    -1,    15,    42,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,    -1,
      42,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    -1,    38,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    15,    38,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    15,
      38,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    15,    38,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    -1,    38,    15,    16,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    15,
      -1,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     1,     6,     7,    33,    57,    58,    59,    60,    63,
      64,    65,    66,    67,    68,    69,    70,    71,    72,    73,
      74,    75,    76,    77,    78,    79,    80,    81,    82,    85,
      86,    87,    88,    89,    90,    91,    92,    94,    95,   102,
     103,   104,   105,   121,   122,   128,   130,   131,   132,   133,
     135,   136,   137,   138,   139,   140,   142,   143,   144,   145,
     146,   147,   149,   153,   154,   155,   157,   161,   164,   165,
     166,   167,   168,   170,   173,   175,   176,   177,   178,   179,
     180,   181,   182,   183,   185,   186,   192,   194,   199,   200,
     201,   202,   125,   151,   142,   157,   157,   153,   147,   147,
     147,   139,   142,     1,   168,   193,   202,   147,   177,   147,
     147,   147,   147,   147,   147,   147,   147,   147,   147,   147,
     147,   147,   147,   147,   147,   147,   147,   147,   147,   147,
     147,   147,    16,    16,   177,   178,   147,    33,     1,   150,
     168,   169,   177,   191,   192,   202,     3,     3,     3,     3,
       4,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      14,    37,   149,   125,     0,   175,    33,    50,    51,   133,
     136,   137,   138,   142,   147,   171,   172,   178,   187,   188,
     189,   151,    32,    37,    39,    40,    43,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    83,    84,    96,
      97,    98,    99,   100,   101,   106,   107,   108,   109,   110,
     112,   113,   114,   115,   116,   117,   118,   119,   120,   123,
     124,   129,   133,   141,   147,   149,   153,   155,   157,   158,
     166,   167,   203,   204,   212,   214,   158,   158,    57,    58,
     201,    33,   171,   172,   189,   142,   155,   184,   184,   158,
     208,   208,   158,   207,   208,   153,   158,   209,   210,   207,
     158,    22,   111,   134,   148,   153,   158,   162,   164,   148,
     158,   153,   162,   148,   162,   133,   157,   157,   147,   157,
     214,   214,   214,   211,   214,    22,   153,   154,   158,   159,
     160,   163,   164,   165,   148,   208,   189,   160,   154,   150,
     125,   150,   168,   202,   169,   192,   162,    22,   163,   208,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     156,   158,   156,   158,   125,   171,   187,   147,   147,    33,
     170,   170,    37,   149,     3,    42,   151,   158,   158,   206,
     158,   158,   147,   147,   147,   147,   147,   147,   147,   147,
     147,   147,   147,   147,   147,   147,   147,   147,   147,   147,
     147,   147,   147,   147,   147,   147,   147,   147,   147,   147,
     147,   147,   147,   147,   147,   147,    33,   158,   166,   177,
     208,   158,   205,    23,    24,    15,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,   148,   148,   148,   147,
     147,   151,   142,   151,    42,    42,    42,   148,   148,    42,
     148,   148,    42,   148,   148,   148,   157,   147,   147,   148,
     148,   148,    42,    42,   148,   148,   148,   148,    42,   148,
      42,   148,   133,   153,   158,    42,   148,   148,   151,   148,
     148,   150,   133,    38,    16,    38,   150,    16,   150,   170,
     174,   177,   190,   174,   187,    38,   158,   150,   158,   158,
     188,    38,    42,   148,   158,   158,   157,   153,   157,   214,
     148,   214,   214,   157,   214,   207,   207,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   162,   158,
     158,   158,   158,   158,   158,   158,   153,   207,   214,   160,
     154,   148,   148,   148,   148,    42,   150,   162,   162,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     158,   158,   158,   158,   158,   158,   158,   158,   158,   158,
     193,   193,   149,   158,   158,   158,   158,   213,   213,   158,
     158,   209,   193,   157,   160,   158,   158,   158,   214,   159,
     147,   158,   158,    42,   148,   188,   148,   148,    37,   149,
      38,   150,     3,   158,   148,    42,   148,   148,   148,   148,
     148,    42,   148,   148,   148,   148,   148,    42,   148,   148,
     148,   148,   148,    42,    42,   148,   148,    42,   148,   148,
     148,   148,   148,   148,    42,   148,   148,   148,   148,   148,
     158,   158,    16,    61,    62,   195,   196,   197,   198,   148,
     148,   151,   148,   148,    93,    42,   148,    42,    42,   148,
     160,    16,    38,   150,   190,   147,    38,   150,    37,   149,
     158,   158,   214,   158,   158,   158,   162,   158,   147,   158,
     158,    16,   169,   197,   150,   195,   151,   151,   201,   193,
     207,   158,   158,   148,   158,   174,    37,   149,   158,   158,
     148,   148,    42,   148,   148,    42,    42,   160,    16,   148,
      42,    42,    42,   148,    38,   150,    38,   150,   158,   158,
     158,   148,   193,   178,   158,   158,   149,    37,   149,   148,
     148,   148,   148,    42,    42,   150,   158,   158,   158,   158,
      38,   150,   148,    42,   149,   158,   158,   148,   150
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
#line 389 "vectree.y"
    {
			(yyval.enp)=NODE0(T_POINTER);
			(yyval.enp)->en_string = savestr((yyvsp[(1) - (1)].idp)->id_name);
			}
    break;

  case 3:
#line 396 "vectree.y"
    {
			(yyval.enp) = NODE0(T_FUNCPTR);
			(yyval.enp)->en_string = savestr((yyvsp[(1) - (1)].idp)->id_name);
			}
    break;

  case 4:
#line 403 "vectree.y"
    {
			(yyval.enp)=NODE0(T_STR_PTR);
			(yyval.enp)->en_string = savestr((yyvsp[(1) - (1)].idp)->id_name);
			}
    break;

  case 5:
#line 410 "vectree.y"
    {
			(yyval.enp)=NODE3(T_RANGE,(yyvsp[(1) - (5)].enp),(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp));
			}
    break;

  case 6:
#line 416 "vectree.y"
    {
			if( (yyvsp[(1) - (1)].dp)->dt_flags & DT_STATIC ){
				(yyval.enp)=NODE0(T_STATIC_OBJ);
				(yyval.enp)->en_dp = (yyvsp[(1) - (1)].dp);
			} else {
				(yyval.enp)=NODE0(T_DYN_OBJ);
				(yyval.enp)->en_string = savestr((yyvsp[(1) - (1)].dp)->dt_name);
			}
			}
    break;

  case 7:
#line 426 "vectree.y"
    {
			(yyval.enp) = NODE1(T_DEREFERENCE,(yyvsp[(2) - (2)].enp));
			}
    break;

  case 8:
#line 430 "vectree.y"
    {
			(yyval.enp)=NODE1(T_OBJ_LOOKUP,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 9:
#line 434 "vectree.y"
    {
			Undef_Sym *usp;

			usp=undef_of(QSP_ARG  (yyvsp[(1) - (1)].e_string));
			if( usp == NO_UNDEF ){
				/* BUG?  are contexts handled correctly??? */
				sprintf(error_string,"Undefined symbol %s",(yyvsp[(1) - (1)].e_string));
				yyerror(QSP_ARG  error_string);
				usp=new_undef(QSP_ARG  (yyvsp[(1) - (1)].e_string));
			}
			(yyval.enp)=NODE0(T_UNDEF);
			(yyval.enp)->en_string = savestr((yyvsp[(1) - (1)].e_string));
			CURDLE((yyval.enp))
			}
    break;

  case 10:
#line 448 "vectree.y"
    {
			(yyval.enp)=NODE1(T_REAL_PART,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 11:
#line 451 "vectree.y"
    {
			(yyval.enp)=NODE1(T_IMAG_PART,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 12:
#line 454 "vectree.y"
    {
			(yyval.enp)=NODE2(T_SQUARE_SUBSCR,(yyvsp[(1) - (4)].enp),(yyvsp[(3) - (4)].enp));
			}
    break;

  case 13:
#line 457 "vectree.y"
    {
			(yyval.enp)=NODE2(T_CURLY_SUBSCR,(yyvsp[(1) - (4)].enp),(yyvsp[(3) - (4)].enp));
			}
    break;

  case 14:
#line 461 "vectree.y"
    {
			(yyval.enp)=NODE3(T_SUBVEC,(yyvsp[(1) - (6)].enp),(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			}
    break;

  case 15:
#line 465 "vectree.y"
    {
			/* Why not use T_RANGE2 here?  The current version
			 * is fine as-is, but don't get rid of T_RANGE2 because
			 * mlab.y uses it...
			 */
			(yyval.enp)=NODE3(T_CSUBVEC,(yyvsp[(1) - (6)].enp),(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			}
    break;

  case 16:
#line 473 "vectree.y"
    {
			(yyval.enp)=NODE2(T_SUBSAMP,(yyvsp[(1) - (4)].enp),(yyvsp[(3) - (4)].enp));
			}
    break;

  case 17:
#line 477 "vectree.y"
    {
			(yyval.enp)=NODE2(T_CSUBSAMP,(yyvsp[(1) - (4)].enp),(yyvsp[(3) - (4)].enp));
			}
    break;

  case 18:
#line 484 "vectree.y"
    {
			(yyval.enp)=NODE1(T_FIX_SIZE,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 20:
#line 496 "vectree.y"
    {
			(yyval.enp) = NODE1(T_TYPECAST,(yyvsp[(4) - (4)].enp));
			(yyval.enp)->en_cast_prec=(yyvsp[(2) - (4)].intval);
			}
    break;

  case 21:
#line 500 "vectree.y"
    {
			(yyval.enp) = (yyvsp[(2) - (3)].enp); }
    break;

  case 22:
#line 502 "vectree.y"
    {
			(yyval.enp)=NODE2(T_PLUS,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 23:
#line 504 "vectree.y"
    {
			(yyval.enp)=NODE2(T_MINUS,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 24:
#line 506 "vectree.y"
    {
			(yyval.enp)=NODE2(T_TIMES,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 25:
#line 508 "vectree.y"
    {
			(yyval.enp)=NODE2(T_DIVIDE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 26:
#line 510 "vectree.y"
    {
			(yyval.enp)=NODE2(T_MODULO,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 27:
#line 512 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BITAND,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 28:
#line 514 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BITOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 29:
#line 516 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BITXOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 30:
#line 518 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BITLSHIFT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 31:
#line 520 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BITRSHIFT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 32:
#line 522 "vectree.y"
    {
			(yyval.enp)=NODE1(T_BITCOMP,(yyvsp[(2) - (2)].enp)); }
    break;

  case 33:
#line 524 "vectree.y"
    {
			(yyval.enp) = NODE0(T_LIT_INT);
			(yyval.enp)->en_intval = (yyvsp[(1) - (1)].dval);
			}
    break;

  case 34:
#line 528 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_EQ,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 35:
#line 531 "vectree.y"
    {
			(yyval.enp) = NODE2(T_BOOL_LT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 36:
#line 534 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_GT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 37:
#line 537 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_GE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 38:
#line 540 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_LE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 39:
#line 543 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_NE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 40:
#line 546 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_AND,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 41:
#line 549 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_OR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 42:
#line 552 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_XOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 43:
#line 555 "vectree.y"
    {
			(yyval.enp)=NODE1(T_BOOL_NOT,(yyvsp[(2) - (2)].enp));
			}
    break;

  case 44:
#line 558 "vectree.y"
    {
			Vec_Expr_Node *enp;
			enp=NODE2(T_BOOL_PTREQ,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			(yyval.enp)=NODE1(T_BOOL_NOT,enp);
			}
    break;

  case 45:
#line 573 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_PTREQ,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 46:
#line 578 "vectree.y"
    {
			(yyval.enp)=NODE0(T_MATH0_FN);
			(yyval.enp)->en_func_index=(yyvsp[(1) - (3)].vfc);
			}
    break;

  case 47:
#line 583 "vectree.y"
    {
			(yyval.enp)=NODE1(T_MATH1_FN,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_func_index=(yyvsp[(1) - (4)].vfc);
			}
    break;

  case 48:
#line 588 "vectree.y"
    {
			(yyval.enp)=NODE2(T_MATH2_FN,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			(yyval.enp)->en_func_index=(yyvsp[(1) - (6)].vfc);
			}
    break;

  case 49:
#line 593 "vectree.y"
    {
			(yyval.enp) = NODE2(T_INNER,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 50:
#line 596 "vectree.y"
    {
			(yyval.enp)=NODE0(T_LIT_DBL);
			(yyval.enp)->en_dblval=(yyvsp[(1) - (1)].dval);
			}
    break;

  case 51:
#line 601 "vectree.y"
    {
			/* We determine exactly which type later */
			(yyval.enp) = NODE3(T_SS_S_CONDASS,(yyvsp[(1) - (5)].enp),(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp));
			}
    break;

  case 52:
#line 605 "vectree.y"
    {
			(yyval.enp) = NODE0(T_LIT_INT);
			(yyval.enp)->en_intval = (yyvsp[(1) - (1)].dval);
			}
    break;

  case 53:
#line 610 "vectree.y"
    {
			(yyval.enp)=NODE0(T_BADNAME);
			NODE_ERROR((yyval.enp));
			CURDLE((yyval.enp))
			WARN("illegal use of data function");
			}
    break;

  case 54:
#line 616 "vectree.y"
    {
			(yyval.enp)=NODE1(T_DATA_FN,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_func_index=(yyvsp[(1) - (4)].fundex);
			}
    break;

  case 55:
#line 621 "vectree.y"
    {
			(yyval.enp)=NODE1(T_SIZE_FN,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_func_index=(yyvsp[(1) - (4)].fundex);
			}
    break;

  case 56:
#line 625 "vectree.y"
    {
			(yyval.enp)=NODE1(T_SIZE_FN,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_func_index=(yyvsp[(1) - (4)].fundex);
			}
    break;

  case 57:
#line 629 "vectree.y"
    {
			(yyval.enp)=NODE1(T_SIZE_FN,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_func_index=(yyvsp[(1) - (4)].fundex);
			NODE_ERROR((yyval.enp));
			advise("dereference pointer before passing to size function");
			CURDLE((yyval.enp))
			}
    break;

  case 58:
#line 636 "vectree.y"
    {
			sprintf(error_string,"need to dereference pointer %s",(yyvsp[(3) - (4)].enp)->en_string);
			yyerror(QSP_ARG  error_string);
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 59:
#line 642 "vectree.y"
    {
			(yyval.enp)=NODE1(T_SUM,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 60:
#line 647 "vectree.y"
    {
			(yyval.enp)=NODE1(T_FILE_EXISTS,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 61:
#line 651 "vectree.y"
    {
			(yyval.enp)=NODE1(T_STR1_FN,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_func_index=(yyvsp[(1) - (4)].fundex);
			}
    break;

  case 62:
#line 656 "vectree.y"
    {
			(yyval.enp)=NODE2(T_STR2_FN,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			(yyval.enp)->en_func_index=(yyvsp[(1) - (6)].fundex);
			}
    break;

  case 63:
#line 661 "vectree.y"
    {
				(yyval.enp)=NODE0(T_MISC_FN);
				(yyval.enp)->en_func_index=(yyvsp[(1) - (3)].fundex);
				}
    break;

  case 64:
#line 667 "vectree.y"
    {
			(yyval.enp)=NODE1(T_CONJ,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 65:
#line 672 "vectree.y"
    {
				(yyval.enp)=NODE1(T_UMINUS,(yyvsp[(2) - (2)].enp));
				}
    break;

  case 66:
#line 676 "vectree.y"
    {
			(yyval.enp)=NODE1(T_MINVAL,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 67:
#line 680 "vectree.y"
    {
			(yyval.enp)=NODE1(T_MAXVAL,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 68:
#line 685 "vectree.y"
    { (yyval.enp)=NODE1(T_MAX_INDEX,(yyvsp[(3) - (4)].enp)); }
    break;

  case 69:
#line 687 "vectree.y"
    { (yyval.enp)=NODE1(T_MIN_INDEX,(yyvsp[(3) - (4)].enp)); }
    break;

  case 70:
#line 690 "vectree.y"
    {
			(yyval.enp) = NODE2(T_INDIR_CALL,(yyvsp[(3) - (7)].enp),(yyvsp[(6) - (7)].enp));
			}
    break;

  case 71:
#line 694 "vectree.y"
    {
			(yyval.enp)=NODE1(T_CALLFUNC,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_call_srp = (yyvsp[(1) - (4)].srp);
			/* make sure this is not a void subroutine! */
			if( (yyvsp[(1) - (4)].srp)->sr_prec == PREC_VOID ){
				NODE_ERROR((yyval.enp));
				sprintf(error_string,"void subroutine %s used in expression!?",(yyvsp[(1) - (4)].srp)->sr_name);
				advise(error_string);
				CURDLE((yyval.enp))
			}
			}
    break;

  case 72:
#line 705 "vectree.y"
    {
			(yyval.enp)=(yyvsp[(1) - (1)].enp);
			}
    break;

  case 73:
#line 708 "vectree.y"
    {
			(yyval.enp)=(yyvsp[(1) - (1)].enp);
			}
    break;

  case 74:
#line 711 "vectree.y"
    {
			WARN("warp not implemented");
			(yyval.enp)=(yyvsp[(3) - (6)].enp);
			}
    break;

  case 75:
#line 715 "vectree.y"
    {
			(yyval.enp)=NODE2(T_LOOKUP,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			}
    break;

  case 76:
#line 719 "vectree.y"
    {
			(yyval.enp) = NODE1(T_TRANSPOSE,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_child_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
			}
    break;

  case 77:
#line 723 "vectree.y"
    { (yyval.enp) = NODE1(T_DFT,(yyvsp[(3) - (4)].enp)); }
    break;

  case 78:
#line 724 "vectree.y"
    { (yyval.enp) = NODE1(T_IDFT,(yyvsp[(3) - (4)].enp)); }
    break;

  case 79:
#line 725 "vectree.y"
    {
			(yyval.enp) = NODE1(T_RDFT,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_child_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
			}
    break;

  case 80:
#line 729 "vectree.y"
    {
			(yyval.enp) = NODE1(T_RIDFT,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_child_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
			}
    break;

  case 82:
#line 734 "vectree.y"
    {
			(yyval.enp)=NODE1(T_WRAP,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 83:
#line 737 "vectree.y"
    {
			(yyval.enp)=NODE3(T_SCROLL,(yyvsp[(3) - (8)].enp),(yyvsp[(5) - (8)].enp),(yyvsp[(7) - (8)].enp));
			}
    break;

  case 84:
#line 742 "vectree.y"
    { (yyval.enp) = NODE1(T_ERODE,(yyvsp[(3) - (4)].enp)); }
    break;

  case 85:
#line 745 "vectree.y"
    { (yyval.enp) = NODE1(T_DILATE,(yyvsp[(3) - (4)].enp)); }
    break;

  case 86:
#line 747 "vectree.y"
    {
			(yyval.enp)=NODE1(T_ENLARGE,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_child_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
			}
    break;

  case 87:
#line 751 "vectree.y"
    {
			(yyval.enp)=NODE1(T_REDUCE,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_child_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
			}
    break;

  case 88:
#line 756 "vectree.y"
    { (yyval.enp)=NODE1(T_LOAD,(yyvsp[(3) - (4)].enp)); }
    break;

  case 89:
#line 757 "vectree.y"
    {
				(yyval.enp)=NODE3(T_RAMP,(yyvsp[(3) - (8)].enp),(yyvsp[(5) - (8)].enp),(yyvsp[(7) - (8)].enp));
				}
    break;

  case 90:
#line 761 "vectree.y"
    {
			(yyval.enp) = NODE3(T_MAX_TIMES,(yyvsp[(3) - (8)].enp),(yyvsp[(5) - (8)].enp),(yyvsp[(7) - (8)].enp));
			}
    break;

  case 92:
#line 769 "vectree.y"
    { (yyval.enp)=NODE1(T_REFERENCE,(yyvsp[(2) - (2)].enp)); }
    break;

  case 96:
#line 781 "vectree.y"
    {
			sprintf(error_string,"shouldn't try to reference pointer variable %s",(yyvsp[(2) - (2)].enp)->en_string);
			yyerror(QSP_ARG  error_string);
			(yyval.enp)=(yyvsp[(2) - (2)].enp);
			}
    break;

  case 97:
#line 787 "vectree.y"
    {
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 99:
#line 794 "vectree.y"
    {
			(yyval.enp)=NODE2(T_ARGLIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 100:
#line 803 "vectree.y"
    {
			/* BUG check to see that this subrt is void! */
			(yyval.enp)=NODE1(T_CALLFUNC,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_call_srp = (yyvsp[(1) - (4)].srp);
			if( (yyvsp[(1) - (4)].srp)->sr_prec != PREC_VOID ){
				NODE_ERROR((yyval.enp));
				sprintf(error_string,"return value of function %s is ignored",(yyvsp[(1) - (4)].srp)->sr_name);
				advise(error_string);
			}
			}
    break;

  case 101:
#line 814 "vectree.y"
    {
			/* BUG check to see that the pointed to subrt is void -
			 * OR should we check that on pointer assignment?
			 */
			(yyval.enp) = NODE2(T_INDIR_CALL,(yyvsp[(3) - (7)].enp),(yyvsp[(6) - (7)].enp));
			}
    break;

  case 102:
#line 824 "vectree.y"
    { (yyval.enp) = NODE1(T_REFERENCE,(yyvsp[(2) - (2)].enp)); }
    break;

  case 105:
#line 828 "vectree.y"
    {
				(yyval.enp)=NODE2(T_EQUIVALENCE,(yyvsp[(3) - (8)].enp),(yyvsp[(5) - (8)].enp));
				(yyval.enp)->en_decl_prec = (yyvsp[(7) - (8)].intval);
			}
    break;

  case 106:
#line 833 "vectree.y"
    {
			(yyval.enp)=NODE1(T_CALLFUNC,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_call_srp = (yyvsp[(1) - (4)].srp);
			/* make sure this is not a void subroutine! */
			if( (yyvsp[(1) - (4)].srp)->sr_prec == PREC_VOID ){
				NODE_ERROR((yyval.enp));
				sprintf(error_string,"void subroutine %s used in pointer expression!?",(yyvsp[(1) - (4)].srp)->sr_name);
				advise(error_string);
				CURDLE((yyval.enp))
			}
			}
    break;

  case 107:
#line 847 "vectree.y"
    {
			(yyval.enp)=NODE0(T_FUNCREF);
			(yyval.enp)->en_srp = (yyvsp[(2) - (2)].srp);
			}
    break;

  case 110:
#line 855 "vectree.y"
    {
			(yyval.enp)=NODE2(T_SET_PTR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 111:
#line 860 "vectree.y"
    {
			(yyval.enp) = NODE2(T_SET_FUNCPTR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 112:
#line 865 "vectree.y"
    {
			(yyval.enp)=NODE2(T_SET_STR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 113:
#line 877 "vectree.y"
    {
			(yyval.enp)=NODE2(T_ASSIGN,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 114:
#line 880 "vectree.y"
    { (yyval.enp)=NODE1(T_POSTINC,(yyvsp[(1) - (2)].enp)); }
    break;

  case 115:
#line 881 "vectree.y"
    { (yyval.enp)=NODE1(T_PREINC,(yyvsp[(2) - (2)].enp)); }
    break;

  case 116:
#line 882 "vectree.y"
    { (yyval.enp)=NODE1(T_PREDEC,(yyvsp[(2) - (2)].enp)); }
    break;

  case 117:
#line 883 "vectree.y"
    { (yyval.enp)=NODE1(T_POSTDEC,(yyvsp[(1) - (2)].enp)); }
    break;

  case 118:
#line 884 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_PLUS,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 119:
#line 890 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_TIMES,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 120:
#line 896 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_MINUS,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 121:
#line 902 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_DIVIDE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 122:
#line 908 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITAND,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 123:
#line 914 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 124:
#line 920 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITXOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 125:
#line 926 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITLSHIFT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 126:
#line 932 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITRSHIFT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 127:
#line 946 "vectree.y"
    { (yyval.enp) = (yyvsp[(1) - (2)].enp); }
    break;

  case 130:
#line 950 "vectree.y"
    {
			Identifier *idp;
			(yyval.enp) = NODE0(T_LABEL);
			idp = new_id(QSP_ARG  (yyvsp[(1) - (2)].e_string));
			idp->id_type = ID_LABEL;
			(yyval.enp)->en_string = savestr(idp->id_name);
			}
    break;

  case 131:
#line 958 "vectree.y"
    {
			(yyval.enp) = NODE0(T_LABEL);
			(yyval.enp)->en_string = savestr((yyvsp[(1) - (2)].idp)->id_name);
			}
    break;

  case 132:
#line 963 "vectree.y"
    { (yyval.enp) = NO_VEXPR_NODE; }
    break;

  case 135:
#line 971 "vectree.y"
    {
			if( (yyvsp[(2) - (2)].enp) != NULL ){
				if( (yyvsp[(1) - (2)].enp) != NULL )
					(yyval.enp)=NODE2(T_STAT_LIST,(yyvsp[(1) - (2)].enp),(yyvsp[(2) - (2)].enp));
				else
					(yyval.enp) = (yyvsp[(2) - (2)].enp);
			} else {
				(yyval.enp)=(yyvsp[(1) - (2)].enp);
			}
			}
    break;

  case 136:
#line 982 "vectree.y"
    {
			(yyval.enp)=NODE2(T_STAT_LIST,(yyvsp[(1) - (2)].enp),(yyvsp[(2) - (2)].enp));
			}
    break;

  case 137:
#line 995 "vectree.y"
    {
			(yyval.enp)=(yyvsp[(2) - (3)].enp);
			}
    break;

  case 138:
#line 999 "vectree.y"
    {
			(yyval.enp)=NODE2(T_STAT_LIST,(yyvsp[(2) - (4)].enp),(yyvsp[(3) - (4)].enp));
			}
    break;

  case 139:
#line 1003 "vectree.y"
    {
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 140:
#line 1007 "vectree.y"
    {
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 141:
#line 1011 "vectree.y"
    {
			yyerror(QSP_ARG  (char *)"missing '}'");
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 142:
#line 1018 "vectree.y"
    {
			set_subrt_ctx(QSP_ARG  (yyvsp[(1) - (4)].e_string));		/* when do we unset??? */
			/* We evaluate the declarations here so we can parse the body, but
			 * the declarations get interpreted a second time when we compile the nodes -
			 * at least, for prototype declarations!?  Not a problem for regular declarations?
			 */
			if( (yyvsp[(3) - (4)].enp) != NO_VEXPR_NODE )
				EVAL_DECL_TREE((yyvsp[(3) - (4)].enp));
			(yyval.enp) = NODE1(T_PROTO,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_string = savestr((yyvsp[(1) - (4)].e_string));
			}
    break;

  case 143:
#line 1032 "vectree.y"
    {
			if( (yyvsp[(1) - (4)].srp)->sr_flags != SR_PROTOTYPE ){
				sprintf(error_string,"Subroutine %s multiply defined!?",(yyvsp[(1) - (4)].srp)->sr_name);
				yyerror(QSP_ARG  error_string);
				/* now what??? */
			}
			set_subrt_ctx(QSP_ARG  (yyvsp[(1) - (4)].srp)->sr_name);		/* when do we unset??? */

			/* compare the two arg decl trees
			 * and issue a warning if they do not match.
			 */
			compare_arg_trees(QSP_ARG  (yyvsp[(3) - (4)].enp),(yyvsp[(1) - (4)].srp)->sr_arg_decls);

			/* use the new ones */
			(yyvsp[(1) - (4)].srp)->sr_arg_decls = (yyvsp[(3) - (4)].enp);
			/* BUG?? we might want to release the old tree... */

			/* We also need to make sure that the type of the function matches
			 * the original prototype...
			 * But we do this later.
			 */

			/* We have to evaluate the new declarations to be able to parse
			 * the body...
			 */

			if( (yyvsp[(3) - (4)].enp) != NO_VEXPR_NODE )
				EVAL_DECL_TREE((yyvsp[(3) - (4)].enp));

			(yyval.enp)=NODE1(T_PROTO,(yyvsp[(3) - (4)].enp));
			/* BUG why are we storing the name again?? */
			(yyval.enp)->en_string = savestr((yyvsp[(1) - (4)].srp)->sr_name);
			}
    break;

  case 144:
#line 1068 "vectree.y"
    {
			Subrt *srp;
			srp=remember_subrt(QSP_ARG  (yyvsp[(1) - (3)].intval),(yyvsp[(2) - (3)].enp)->en_string,(yyvsp[(2) - (3)].enp)->en_child[0],(yyvsp[(3) - (3)].enp));
			srp->sr_prec = (yyvsp[(1) - (3)].intval);
			(yyval.enp)=NODE0(T_SUBRT);
			(yyval.enp)->en_srp=srp;
			delete_subrt_ctx(QSP_ARG  (yyvsp[(2) - (3)].enp)->en_string);	/* this deletes the objects... */
			COMPILE_SUBRT(srp);
			}
    break;

  case 145:
#line 1078 "vectree.y"
    {
			Subrt *srp;
			srp=remember_subrt(QSP_ARG  (yyvsp[(1) - (4)].intval),(yyvsp[(3) - (4)].enp)->en_string,(yyvsp[(3) - (4)].enp)->en_child[0],(yyvsp[(4) - (4)].enp));
			srp->sr_prec = (yyvsp[(1) - (4)].intval);
			srp->sr_flags |= SR_REFFUNC;
			/* set a flag to show returns ptr */
			(yyval.enp)=NODE0(T_SUBRT);
			(yyval.enp)->en_srp=srp;
			delete_subrt_ctx(QSP_ARG  (yyvsp[(3) - (4)].enp)->en_string);	/* this deletes the objects... */
			COMPILE_SUBRT(srp);
			}
    break;

  case 146:
#line 1090 "vectree.y"
    {
			/* BUG make sure that precision matches prototype decl */
			Subrt *srp;
			srp=subrt_of(QSP_ARG  (yyvsp[(2) - (3)].enp)->en_string);
#ifdef CAUTIOUS
			if( srp == NO_SUBRT ) {
				NODE_ERROR((yyvsp[(2) - (3)].enp));
				ERROR1("CAUTIOUS:  missing subrt!?");
			}
#endif /* CAUTIOUS */
			update_subrt(QSP_ARG  srp,(yyvsp[(3) - (3)].enp));
			(yyval.enp)=NODE0(T_SUBRT);
			(yyval.enp)->en_srp=srp;
			delete_subrt_ctx(QSP_ARG  (yyvsp[(2) - (3)].enp)->en_string);
			COMPILE_SUBRT(srp);
			}
    break;

  case 147:
#line 1110 "vectree.y"
    {
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 148:
#line 1114 "vectree.y"
    {
			(yyval.enp)=(yyvsp[(1) - (1)].enp);
			}
    break;

  case 149:
#line 1118 "vectree.y"
    {
			(yyval.enp)=NODE2(T_DECL_STAT_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 152:
#line 1126 "vectree.y"
    {
			if( (yyval.enp) != NO_VEXPR_NODE ) EVAL_IMMEDIATE((yyval.enp));
			}
    break;

  case 153:
#line 1130 "vectree.y"
    {
			if( (yyval.enp) != NO_VEXPR_NODE ) EVAL_IMMEDIATE((yyval.enp));
			}
    break;

  case 154:
#line 1136 "vectree.y"
    { TOP_NODE=(yyvsp[(1) - (2)].enp);  }
    break;

  case 155:
#line 1138 "vectree.y"
    { TOP_NODE=(yyvsp[(1) - (1)].enp); }
    break;

  case 156:
#line 1139 "vectree.y"
    {
			(yyval.enp)=NODE2(T_STAT_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(2) - (3)].enp));
			TOP_NODE=(yyval.enp);
			}
    break;

  case 157:
#line 1143 "vectree.y"
    {
			(yyval.enp)=NODE2(T_STAT_LIST,(yyvsp[(1) - (2)].enp),(yyvsp[(2) - (2)].enp));
			TOP_NODE=(yyval.enp);
			}
    break;

  case 158:
#line 1148 "vectree.y"
    {
			(yyval.enp) = NO_VEXPR_NODE;
			TOP_NODE=(yyval.enp);
			}
    break;

  case 160:
#line 1155 "vectree.y"
    { (yyval.intval) = (yyvsp[(2) - (2)].intval) | DT_RDONLY ; }
    break;

  case 161:
#line 1158 "vectree.y"
    { (yyval.intval)		= PREC_BY;	}
    break;

  case 162:
#line 1159 "vectree.y"
    { (yyval.intval)		= PREC_CHAR;	}
    break;

  case 163:
#line 1160 "vectree.y"
    { (yyval.intval)		= PREC_STR;	}
    break;

  case 164:
#line 1161 "vectree.y"
    { (yyval.intval)		= PREC_SP;	}
    break;

  case 165:
#line 1162 "vectree.y"
    { (yyval.intval)		= PREC_DP;	}
    break;

  case 166:
#line 1163 "vectree.y"
    { (yyval.intval)		= PREC_CPX;	}
    break;

  case 167:
#line 1164 "vectree.y"
    { (yyval.intval)		= PREC_DBLCPX;	}
    break;

  case 168:
#line 1165 "vectree.y"
    { (yyval.intval)		= PREC_IN;	}
    break;

  case 169:
#line 1166 "vectree.y"
    { (yyval.intval)		= PREC_DI;	}
    break;

  case 170:
#line 1167 "vectree.y"
    { (yyval.intval)		= PREC_UBY;	}
    break;

  case 171:
#line 1168 "vectree.y"
    { (yyval.intval)		= PREC_UIN;	}
    break;

  case 172:
#line 1169 "vectree.y"
    { (yyval.intval)		= PREC_UDI;	}
    break;

  case 173:
#line 1170 "vectree.y"
    { (yyval.intval)		= PREC_BIT;	}
    break;

  case 174:
#line 1171 "vectree.y"
    { (yyval.intval)		= PREC_COLOR;	}
    break;

  case 175:
#line 1172 "vectree.y"
    { (yyval.intval)	= PREC_VOID;	}
    break;

  case 176:
#line 1177 "vectree.y"
    { (yyval.enp)=NODE1(T_INFO,(yyvsp[(3) - (4)].enp)); }
    break;

  case 177:
#line 1179 "vectree.y"
    { (yyval.enp)=NODE1(T_DISPLAY,(yyvsp[(3) - (4)].enp)); }
    break;

  case 178:
#line 1182 "vectree.y"
    { (yyval.enp)=NODE0(T_EXIT); }
    break;

  case 179:
#line 1183 "vectree.y"
    { (yyval.enp)=NODE0(T_EXIT); }
    break;

  case 180:
#line 1184 "vectree.y"
    { (yyval.enp)=NODE1(T_EXIT,(yyvsp[(3) - (4)].enp)); }
    break;

  case 181:
#line 1188 "vectree.y"
    {
			(yyval.enp)=NODE1(T_RETURN,NO_VEXPR_NODE);
			}
    break;

  case 182:
#line 1192 "vectree.y"
    {
			(yyval.enp)=NODE1(T_RETURN,NO_VEXPR_NODE);
			}
    break;

  case 183:
#line 1196 "vectree.y"
    {
			(yyval.enp)=NODE1(T_RETURN,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 184:
#line 1200 "vectree.y"
    {
			(yyval.enp)=NODE1(T_RETURN,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 185:
#line 1206 "vectree.y"
    { (yyval.enp)=NODE2(T_SAVE,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp)); }
    break;

  case 186:
#line 1208 "vectree.y"
    { (yyval.enp)=NODE1(T_FILETYPE,(yyvsp[(3) - (4)].enp)); }
    break;

  case 187:
#line 1212 "vectree.y"
    {
			(yyval.enp)=NODE1(T_SCRIPT,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_srp = (yyvsp[(1) - (4)].srp);
			}
    break;

  case 188:
#line 1217 "vectree.y"
    {
			(yyval.enp)=NODE1(T_SCRIPT,NO_VEXPR_NODE);
			(yyval.enp)->en_srp = (yyvsp[(1) - (3)].srp);
			}
    break;

  case 190:
#line 1225 "vectree.y"
    {
			sprintf(error_string,"undefined string pointer \"%s\"",(yyvsp[(1) - (1)].e_string));
			yyerror(QSP_ARG  error_string);
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 191:
#line 1233 "vectree.y"
    {
			(yyval.enp) = NODE2(T_STRCPY,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			}
    break;

  case 192:
#line 1237 "vectree.y"
    {
			(yyval.enp) = NODE2(T_STRCAT,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			}
    break;

  case 193:
#line 1246 "vectree.y"
    {
			(yyval.enp) = NODE1(T_CALL_NATIVE,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_intval = (yyvsp[(1) - (4)].intval);
			}
    break;

  case 194:
#line 1260 "vectree.y"
    {
			Vec_Expr_Node *enp,*enp2;
			enp=NODE2(T_EXPR_LIST,(yyvsp[(5) - (12)].enp),(yyvsp[(7) - (12)].enp));
			enp2=NODE2(T_EXPR_LIST,(yyvsp[(9) - (12)].enp),(yyvsp[(11) - (12)].enp));
			(yyval.enp) = NODE3(T_FILL,(yyvsp[(3) - (12)].enp),enp,enp2);
			}
    break;

  case 195:
#line 1267 "vectree.y"
    {
			(yyval.enp) = NODE0(T_CLR_OPT_PARAMS);
			}
    break;

  case 196:
#line 1272 "vectree.y"
    {
			Vec_Expr_Node *enp1,*enp2,*enp3;
			enp1=NODE2(T_EXPR_LIST,(yyvsp[(3) - (14)].enp),(yyvsp[(5) - (14)].enp));
			enp2=NODE2(T_EXPR_LIST,(yyvsp[(7) - (14)].enp),(yyvsp[(9) - (14)].enp));
			enp3=NODE2(T_EXPR_LIST,(yyvsp[(11) - (14)].enp),(yyvsp[(13) - (14)].enp));
			(yyval.enp) = NODE3(T_ADD_OPT_PARAM,enp1,enp2,enp3);
			}
    break;

  case 197:
#line 1280 "vectree.y"
    {
			(yyval.enp) = NODE0(T_OPTIMIZE);
			(yyval.enp)->en_srp = (yyvsp[(3) - (4)].srp);
			}
    break;

  case 198:
#line 1287 "vectree.y"
    { (yyval.enp)=NODE1(T_OUTPUT_FILE,(yyvsp[(3) - (4)].enp)); }
    break;

  case 199:
#line 1291 "vectree.y"
    { (yyval.enp)=NODE1(T_EXP_PRINT,(yyvsp[(3) - (4)].enp)); }
    break;

  case 200:
#line 1292 "vectree.y"
    { (yyval.enp)=NODE1(T_EXP_PRINT,(yyvsp[(3) - (4)].enp)); }
    break;

  case 201:
#line 1293 "vectree.y"
    { (yyval.enp)=NODE1(T_ADVISE,(yyvsp[(3) - (4)].enp)); }
    break;

  case 202:
#line 1294 "vectree.y"
    { (yyval.enp)=NODE1(T_WARN,(yyvsp[(3) - (4)].enp)); }
    break;

  case 204:
#line 1299 "vectree.y"
    { (yyval.e_string) = (yyvsp[(1) - (1)].dp)->dt_name; }
    break;

  case 205:
#line 1301 "vectree.y"
    { (yyval.e_string) = (yyvsp[(1) - (1)].idp)->id_name; }
    break;

  case 206:
#line 1303 "vectree.y"
    { (yyval.e_string) = (yyvsp[(1) - (1)].idp)->id_name; }
    break;

  case 207:
#line 1305 "vectree.y"
    {
			yyerror(QSP_ARG  (char *)"illegal attempt to use a keyword as an identifier");
			(yyval.e_string)="<illegal_keyword_use>";
			}
    break;

  case 208:
#line 1311 "vectree.y"
    {
			(yyval.enp) = NODE0(T_SCAL_DECL);
			(yyval.enp)->en_string=savestr((yyvsp[(1) - (1)].e_string));	/* bug need to save??? */
			}
    break;

  case 209:
#line 1322 "vectree.y"
    {
			delete_subrt_ctx(QSP_ARG  (yyvsp[(1) - (1)].enp)->en_string);
			}
    break;

  case 210:
#line 1326 "vectree.y"
    {
			delete_subrt_ctx(QSP_ARG  (yyvsp[(1) - (1)].enp)->en_string);
			}
    break;

  case 211:
#line 1330 "vectree.y"
    {
			/* function pointer */
			(yyval.enp) = NODE1(T_FUNCPTR_DECL,(yyvsp[(6) - (7)].enp));
			(yyval.enp)->en_decl_name=savestr((yyvsp[(3) - (7)].e_string));
			}
    break;

  case 212:
#line 1335 "vectree.y"
    {
			(yyval.enp) = NODE1(T_CSCAL_DECL,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (4)].e_string));
			}
    break;

  case 213:
#line 1339 "vectree.y"
    {
			(yyval.enp) = NODE1(T_VEC_DECL,(yyvsp[(3) - (4)].enp));
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (4)].e_string));
			}
    break;

  case 214:
#line 1343 "vectree.y"
    {
			(yyval.enp) = NODE2(T_CVEC_DECL,(yyvsp[(3) - (7)].enp),(yyvsp[(6) - (7)].enp));
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (7)].e_string));
			}
    break;

  case 215:
#line 1347 "vectree.y"
    {
			(yyval.enp)=NODE2(T_IMG_DECL,(yyvsp[(3) - (7)].enp),(yyvsp[(6) - (7)].enp));
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (7)].e_string));
			}
    break;

  case 216:
#line 1351 "vectree.y"
    {
			(yyval.enp)=NODE3(T_CIMG_DECL,(yyvsp[(3) - (10)].enp),(yyvsp[(6) - (10)].enp),(yyvsp[(9) - (10)].enp));
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (10)].e_string));
			}
    break;

  case 217:
#line 1355 "vectree.y"
    {
			(yyval.enp)=NODE3(T_SEQ_DECL,(yyvsp[(3) - (10)].enp),(yyvsp[(6) - (10)].enp),(yyvsp[(9) - (10)].enp));
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (10)].e_string));
			}
    break;

  case 218:
#line 1359 "vectree.y"
    {
			Vec_Expr_Node *enp;
			enp = NODE2(T_EXPR_LIST,(yyvsp[(9) - (13)].enp),(yyvsp[(12) - (13)].enp));
			(yyval.enp)=NODE3(T_CSEQ_DECL,(yyvsp[(3) - (13)].enp),(yyvsp[(6) - (13)].enp),enp);
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (13)].e_string));
			}
    break;

  case 219:
#line 1365 "vectree.y"
    {
			(yyval.enp) = NODE1(T_CSCAL_DECL,NO_VEXPR_NODE);
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (3)].e_string));
			}
    break;

  case 220:
#line 1370 "vectree.y"
    {
			(yyval.enp) = NODE1(T_VEC_DECL,NO_VEXPR_NODE);
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (3)].e_string));
			}
    break;

  case 221:
#line 1375 "vectree.y"
    {
			(yyval.enp) = NODE2(T_CVEC_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE);
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (5)].e_string));
			}
    break;

  case 222:
#line 1380 "vectree.y"
    {
			(yyval.enp) = NODE2(T_IMG_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE);
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (5)].e_string));
			}
    break;

  case 223:
#line 1385 "vectree.y"
    {
			(yyval.enp) = NODE3(T_CIMG_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE,NO_VEXPR_NODE);
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (7)].e_string));
			}
    break;

  case 224:
#line 1390 "vectree.y"
    {
			(yyval.enp) = NODE3(T_SEQ_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE,NO_VEXPR_NODE);
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (7)].e_string));
			}
    break;

  case 225:
#line 1395 "vectree.y"
    {
			(yyval.enp) = NODE3(T_CSEQ_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE,NO_VEXPR_NODE);
			(yyval.enp)->en_decl_name=savestr((yyvsp[(1) - (9)].e_string));
			}
    break;

  case 226:
#line 1400 "vectree.y"
    {
			(yyval.enp)=NODE0(T_PTR_DECL);
			(yyval.enp)->en_decl_name  = savestr((yyvsp[(2) - (2)].e_string));
			}
    break;

  case 227:
#line 1405 "vectree.y"
    {
			(yyval.enp)=NODE0(T_BADNAME);
			(yyval.enp)->en_string = savestr( ((Function *)data_functbl) [(yyvsp[(1) - (1)].fundex)].fn_name);
			CURDLE((yyval.enp))
			NODE_ERROR((yyval.enp));
			WARN("illegal data function name use");
			}
    break;

  case 228:
#line 1413 "vectree.y"
    {
			(yyval.enp)=NODE0(T_BADNAME);
			(yyval.enp)->en_string = savestr( ((Function *)size_functbl) [(yyvsp[(1) - (1)].fundex)].fn_name);
			CURDLE((yyval.enp))
			NODE_ERROR((yyval.enp));
			WARN("illegal size function name use");
			}
    break;

  case 230:
#line 1460 "vectree.y"
    {
			(yyval.enp)=NODE2(T_DECL_INIT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 231:
#line 1463 "vectree.y"
    {
			(yyval.enp)=NODE2(T_DECL_ITEM_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 232:
#line 1465 "vectree.y"
    {
			Vec_Expr_Node *enp;
			enp=NODE2(T_DECL_INIT,(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp));
			(yyval.enp)=NODE2(T_DECL_ITEM_LIST,(yyvsp[(1) - (5)].enp),enp); }
    break;

  case 233:
#line 1476 "vectree.y"
    {
			(yyval.enp)=NODE1(T_DECL_STAT,(yyvsp[(2) - (2)].enp));
			if( (yyvsp[(1) - (2)].intval) & DT_RDONLY )
				(yyval.enp)->en_decl_flags = DECL_IS_CONST;
			(yyval.enp)->en_decl_prec=(yyvsp[(1) - (2)].intval) & ~DT_RDONLY;
			}
    break;

  case 235:
#line 1486 "vectree.y"
    {
			(yyval.enp)=NODE2(T_DECL_STAT_LIST,(yyvsp[(1) - (2)].enp),(yyvsp[(2) - (2)].enp));
			}
    break;

  case 236:
#line 1492 "vectree.y"
    {
			(yyval.enp) = NODE1(T_DECL_STAT,(yyvsp[(2) - (3)].enp));
			if( (yyvsp[(1) - (3)].intval) & DT_RDONLY )
				(yyval.enp)->en_decl_flags = DECL_IS_CONST;
			(yyval.enp)->en_decl_prec=(yyvsp[(1) - (3)].intval) & ~DT_RDONLY;
			EVAL_IMMEDIATE((yyval.enp));
			}
    break;

  case 237:
#line 1499 "vectree.y"
    {
			(yyval.enp) = NODE1(T_EXTERN_DECL,(yyvsp[(3) - (4)].enp));
			if( (yyvsp[(2) - (4)].intval) & DT_RDONLY )
				(yyval.enp)->en_decl_flags = DECL_IS_CONST;
			(yyval.enp)->en_decl_prec=(yyvsp[(2) - (4)].intval) & ~DT_RDONLY;
			EVAL_IMMEDIATE((yyval.enp));
			}
    break;

  case 238:
#line 1506 "vectree.y"
    {
			(yyval.enp) = NODE1(T_DECL_STAT,(yyvsp[(3) - (4)].enp));
			if( (yyvsp[(2) - (4)].intval) & DT_RDONLY )
				(yyval.enp)->en_decl_flags = DECL_IS_CONST;
			(yyval.enp)->en_decl_flags |= DECL_IS_STATIC;
			(yyval.enp)->en_decl_prec=(yyvsp[(2) - (4)].intval) & ~DT_RDONLY;
			EVAL_IMMEDIATE((yyval.enp));
			}
    break;

  case 241:
#line 1521 "vectree.y"
    {
				if( (yyvsp[(5) - (5)].enp) != NULL )
					(yyval.enp) = NODE2(T_WHILE,(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp));
				else
					(yyval.enp) = NULL;
			}
    break;

  case 242:
#line 1528 "vectree.y"
    {
				if( (yyvsp[(5) - (5)].enp) != NULL )
					(yyval.enp) = NODE2(T_UNTIL,(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp));
				else
					(yyval.enp) = NULL;
			}
    break;

  case 243:
#line 1535 "vectree.y"
    {
			Vec_Expr_Node *loop_enp;

			loop_enp=NODE3(T_FOR,(yyvsp[(5) - (9)].enp),(yyvsp[(9) - (9)].enp),(yyvsp[(7) - (9)].enp));
			if( (yyvsp[(3) - (9)].enp) != NULL ){
				(yyval.enp) = NODE2(T_STAT_LIST,(yyvsp[(3) - (9)].enp),loop_enp);
			} else {
				(yyval.enp) = loop_enp;
			}
			}
    break;

  case 244:
#line 1546 "vectree.y"
    {
			/* we want to preserve a strict tree structure */
			(yyval.enp) = NODE2(T_DO_WHILE,(yyvsp[(2) - (7)].enp),(yyvsp[(5) - (7)].enp));
			}
    break;

  case 245:
#line 1551 "vectree.y"
    {
			/* we want to preserve a strict tree structure */
			(yyval.enp) = NODE2(T_DO_UNTIL,(yyvsp[(2) - (7)].enp),(yyvsp[(5) - (7)].enp));
			}
    break;

  case 246:
#line 1558 "vectree.y"
    { (yyval.enp) = NODE2(T_CASE_STAT,(yyvsp[(1) - (2)].enp),(yyvsp[(2) - (2)].enp)); }
    break;

  case 248:
#line 1563 "vectree.y"
    { (yyval.enp) = NODE2(T_CASE_LIST,(yyvsp[(1) - (2)].enp),(yyvsp[(2) - (2)].enp)); }
    break;

  case 249:
#line 1567 "vectree.y"
    { (yyval.enp) = NODE1(T_CASE,(yyvsp[(2) - (3)].enp)); }
    break;

  case 250:
#line 1569 "vectree.y"
    { (yyval.enp) = NODE0(T_DEFAULT); }
    break;

  case 252:
#line 1574 "vectree.y"
    { (yyval.enp) = NODE2(T_SWITCH_LIST,(yyvsp[(1) - (2)].enp),(yyvsp[(2) - (2)].enp)); }
    break;

  case 253:
#line 1578 "vectree.y"
    { (yyval.enp)=NODE2(T_SWITCH,(yyvsp[(3) - (7)].enp),(yyvsp[(6) - (7)].enp)); }
    break;

  case 254:
#line 1583 "vectree.y"
    { (yyval.enp) = NODE3(T_IFTHEN,(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp),NO_VEXPR_NODE); }
    break;

  case 255:
#line 1585 "vectree.y"
    { (yyval.enp) = NODE3(T_IFTHEN,(yyvsp[(3) - (7)].enp),(yyvsp[(5) - (7)].enp),(yyvsp[(7) - (7)].enp)); }
    break;

  case 256:
#line 1600 "vectree.y"
    { (yyval.enp) = NULL; }
    break;

  case 269:
#line 1613 "vectree.y"
    { (yyval.enp)=NODE0(T_BREAK); }
    break;

  case 270:
#line 1614 "vectree.y"
    { (yyval.enp)=NODE0(T_CONTINUE); }
    break;

  case 271:
#line 1616 "vectree.y"
    {
			(yyval.enp) = NODE0(T_GO_BACK);
			(yyval.enp)->en_string = savestr((yyvsp[(2) - (2)].idp)->id_name);
			}
    break;

  case 272:
#line 1621 "vectree.y"
    {
			(yyval.enp) = NODE0(T_GO_FWD);
			(yyval.enp)->en_string = savestr((yyvsp[(2) - (2)].e_string));
			}
    break;

  case 273:
#line 1633 "vectree.y"
    { (yyval.enp) = (yyvsp[(1) - (1)].enp); }
    break;

  case 274:
#line 1635 "vectree.y"
    { (yyval.enp) = (yyvsp[(1) - (1)].enp); }
    break;

  case 275:
#line 1643 "vectree.y"
    {
			(yyval.enp)=NODE1(T_COMP_OBJ,(yyvsp[(2) - (3)].enp));
			}
    break;

  case 276:
#line 1648 "vectree.y"
    {
			(yyval.enp)=NODE1(T_LIST_OBJ,(yyvsp[(2) - (3)].enp));
			}
    break;

  case 278:
#line 1655 "vectree.y"
    {
			(yyval.enp)=NODE2(T_COMP_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 280:
#line 1662 "vectree.y"
    {
			(yyval.enp)=NODE2(T_ROW_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 282:
#line 1669 "vectree.y"
    {
			(yyval.enp)=NODE2(T_EXPR_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 284:
#line 1677 "vectree.y"
    { (yyval.enp)=NODE2(T_PRINT_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 288:
#line 1686 "vectree.y"
    { (yyval.enp)=NODE2(T_MIXED_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 290:
#line 1691 "vectree.y"
    { (yyval.enp)=NODE2(T_STRING_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 291:
#line 1695 "vectree.y"
    {
			(yyval.enp)=NODE0(T_STRING);
			(yyval.enp)->en_string = savestr((yyvsp[(1) - (1)].e_string));
				/* BUG?  make sure to free if tree deleted */
			}
    break;

  case 292:
#line 1701 "vectree.y"
    { (yyval.enp) = NODE1(T_NAME_FUNC,(yyvsp[(3) - (4)].enp)); }
    break;

  case 298:
#line 1716 "vectree.y"
    { (yyval.enp) = (yyvsp[(2) - (3)].enp); }
    break;

  case 299:
#line 1717 "vectree.y"
    { (yyval.enp) = (yyvsp[(2) - (3)].enp); }
    break;


/* Line 1267 of yacc.c.  */
#line 4990 "vectree.c"
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


#line 1765 "vectree.y"


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


