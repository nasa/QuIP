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
     INT_NUM = 281,
     CHAR_CONST = 282,
     MATH0_FUNC = 283,
     MATH1_FUNC = 284,
     MATH2_FUNC = 285,
     INT1_FUNC = 286,
     STR1_FUNC = 287,
     STR2_FUNC = 288,
     STR3_FUNC = 289,
     STRV_FUNC = 290,
     CHAR_FUNC = 291,
     DATA_FUNC = 292,
     SIZE_FUNC = 293,
     TS_FUNC = 294,
     BEGIN_COMMENT = 295,
     END_COMMENT = 296,
     WHILE = 297,
     UNTIL = 298,
     CONTINUE = 299,
     SWITCH = 300,
     CASE = 301,
     DEFAULT = 302,
     BREAK = 303,
     GOTO = 304,
     DO = 305,
     FOR = 306,
     STATIC = 307,
     BYTE = 308,
     CHAR = 309,
     STRING = 310,
     FLOAT = 311,
     DOUBLE = 312,
     SHORT = 313,
     INT32 = 314,
     INT64 = 315,
     BIT = 316,
     UBYTE = 317,
     USHORT = 318,
     UINT32 = 319,
     UINT64 = 320,
     COLOR = 321,
     COMPLEX = 322,
     DBLCPX = 323,
     QUATERNION = 324,
     DBLQUAT = 325,
     STRCPY = 326,
     NAME_FUNC = 327,
     FILE_EXISTS = 328,
     STRCAT = 329,
     ECHO = 330,
     ADVISE_FUNC = 331,
     DISPLAY = 332,
     F_WARN = 333,
     PRINT = 334,
     INFO = 335,
     IF = 336,
     ELSE = 337,
     RETURN = 338,
     EXIT = 339,
     MINVAL = 340,
     MAXVAL = 341,
     WRAP = 342,
     SCROLL = 343,
     DILATE = 344,
     FIX_SIZE = 345,
     FILL = 346,
     CLR_OPT_PARAMS = 347,
     ADD_OPT_PARAM = 348,
     OPTIMIZE = 349,
     ERODE = 350,
     ENLARGE = 351,
     REDUCE = 352,
     WARP = 353,
     LOOKUP = 354,
     EQUIVALENCE = 355,
     TRANSPOSE = 356,
     CONJ = 357,
     MAX_TIMES = 358,
     MAX_INDEX = 359,
     MIN_INDEX = 360,
     DFT = 361,
     IDFT = 362,
     RDFT = 363,
     RIDFT = 364,
     REAL_PART = 365,
     IMAG_PART = 366,
     RAMP = 367,
     SUM = 368,
     END = 369,
     NEXT_TOKEN = 370,
     NEWLINE = 371,
     SET_OUTPUT_FILE = 372,
     LOAD = 373,
     SAVE = 374,
     FILETYPE = 375,
     OBJ_OF = 376,
     FUNCNAME = 377,
     REFFUNC = 378,
     SCRIPTFUNC = 379,
     OBJNAME = 380,
     PTRNAME = 381,
     STRNAME = 382,
     LABELNAME = 383,
     FUNCPTRNAME = 384,
     LEX_STRING = 385,
     NEWNAME = 386,
     VOID_TYPE = 387,
     EXTERN = 388,
     NATIVE_FUNC_NAME = 389
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
#define INT_NUM 281
#define CHAR_CONST 282
#define MATH0_FUNC 283
#define MATH1_FUNC 284
#define MATH2_FUNC 285
#define INT1_FUNC 286
#define STR1_FUNC 287
#define STR2_FUNC 288
#define STR3_FUNC 289
#define STRV_FUNC 290
#define CHAR_FUNC 291
#define DATA_FUNC 292
#define SIZE_FUNC 293
#define TS_FUNC 294
#define BEGIN_COMMENT 295
#define END_COMMENT 296
#define WHILE 297
#define UNTIL 298
#define CONTINUE 299
#define SWITCH 300
#define CASE 301
#define DEFAULT 302
#define BREAK 303
#define GOTO 304
#define DO 305
#define FOR 306
#define STATIC 307
#define BYTE 308
#define CHAR 309
#define STRING 310
#define FLOAT 311
#define DOUBLE 312
#define SHORT 313
#define INT32 314
#define INT64 315
#define BIT 316
#define UBYTE 317
#define USHORT 318
#define UINT32 319
#define UINT64 320
#define COLOR 321
#define COMPLEX 322
#define DBLCPX 323
#define QUATERNION 324
#define DBLQUAT 325
#define STRCPY 326
#define NAME_FUNC 327
#define FILE_EXISTS 328
#define STRCAT 329
#define ECHO 330
#define ADVISE_FUNC 331
#define DISPLAY 332
#define F_WARN 333
#define PRINT 334
#define INFO 335
#define IF 336
#define ELSE 337
#define RETURN 338
#define EXIT 339
#define MINVAL 340
#define MAXVAL 341
#define WRAP 342
#define SCROLL 343
#define DILATE 344
#define FIX_SIZE 345
#define FILL 346
#define CLR_OPT_PARAMS 347
#define ADD_OPT_PARAM 348
#define OPTIMIZE 349
#define ERODE 350
#define ENLARGE 351
#define REDUCE 352
#define WARP 353
#define LOOKUP 354
#define EQUIVALENCE 355
#define TRANSPOSE 356
#define CONJ 357
#define MAX_TIMES 358
#define MAX_INDEX 359
#define MIN_INDEX 360
#define DFT 361
#define IDFT 362
#define RDFT 363
#define RIDFT 364
#define REAL_PART 365
#define IMAG_PART 366
#define RAMP 367
#define SUM 368
#define END 369
#define NEXT_TOKEN 370
#define NEWLINE 371
#define SET_OUTPUT_FILE 372
#define LOAD 373
#define SAVE 374
#define FILETYPE 375
#define OBJ_OF 376
#define FUNCNAME 377
#define REFFUNC 378
#define SCRIPTFUNC 379
#define OBJNAME 380
#define PTRNAME 381
#define STRNAME 382
#define LABELNAME 383
#define FUNCPTRNAME 384
#define LEX_STRING 385
#define NEWNAME 386
#define VOID_TYPE 387
#define EXTERN 388
#define NATIVE_FUNC_NAME 389




/* Copy the first part of user declarations.  */
#line 1 "vectree.y"

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
#line 495 "vectree.c"

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
#define YYFINAL  227
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   4986

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  158
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  63
/* YYNRULES -- Number of rules.  */
#define YYNRULES  307
/* YYNRULES -- Number of states.  */
#define YYNSTATES  737

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   389

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    39,     2,     2,     2,    35,    22,     2,
     153,   154,    33,    31,    42,    32,     2,    34,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    16,   157,
      25,     3,    26,    15,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    37,     2,    38,    21,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,   155,    20,   156,    40,     2,     2,     2,
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
     143,   144,   145,   146,   147,   148,   149,   150,   151,   152
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
     200,   204,   206,   212,   214,   216,   221,   226,   231,   236,
     241,   246,   251,   256,   263,   270,   275,   280,   285,   288,
     293,   298,   303,   308,   316,   321,   323,   325,   332,   339,
     344,   349,   354,   359,   364,   366,   371,   380,   385,   390,
     395,   400,   405,   414,   423,   425,   428,   430,   432,   434,
     437,   438,   440,   444,   449,   457,   460,   462,   464,   473,
     478,   481,   483,   485,   489,   493,   497,   501,   504,   507,
     510,   513,   517,   521,   525,   529,   533,   537,   541,   545,
     549,   552,   554,   556,   559,   562,   565,   567,   569,   572,
     575,   579,   583,   588,   591,   595,   599,   604,   609,   613,
     618,   622,   623,   625,   629,   631,   633,   635,   637,   640,
     642,   646,   649,   652,   654,   656,   658,   660,   662,   664,
     666,   668,   670,   672,   674,   676,   678,   680,   682,   684,
     686,   688,   690,   692,   697,   702,   704,   708,   713,   715,
     719,   722,   727,   730,   737,   742,   747,   751,   753,   755,
     762,   769,   774,   787,   791,   806,   811,   816,   821,   826,
     831,   836,   838,   840,   842,   844,   846,   848,   850,   852,
     860,   865,   870,   878,   886,   897,   908,   922,   926,   930,
     936,   942,   950,   958,   968,   971,   973,   975,   977,   981,
     985,   991,   994,   996,   999,  1003,  1008,  1013,  1015,  1017,
    1023,  1029,  1039,  1047,  1055,  1058,  1060,  1063,  1067,  1070,
    1072,  1075,  1083,  1089,  1097,  1098,  1100,  1102,  1104,  1106,
    1108,  1110,  1112,  1114,  1116,  1118,  1120,  1122,  1124,  1126,
    1129,  1132,  1134,  1136,  1140,  1144,  1146,  1150,  1152,  1156,
    1158,  1162,  1164,  1168,  1170,  1172,  1174,  1178,  1180,  1184,
    1186,  1191,  1193,  1195,  1197,  1199,  1201,  1205
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int16 yyrhs[] =
{
     182,     0,    -1,   144,    -1,   147,    -1,   145,    -1,   164,
      16,   164,    16,   164,    -1,   143,    -1,    33,   159,    -1,
     139,   153,   217,   154,    -1,   149,    -1,   128,   153,   163,
     154,    -1,   129,   153,   163,   154,    -1,   163,    37,   164,
      38,    -1,   163,   155,   164,   156,    -1,   163,    37,   164,
      16,   164,    38,    -1,   163,   155,   164,    16,   164,   156,
      -1,   163,    37,   162,    38,    -1,   163,   155,   162,   156,
      -1,   108,   153,   164,   154,    -1,   220,    -1,   153,   183,
     154,   164,    -1,   153,   164,   154,    -1,   164,    31,   164,
      -1,   164,    32,   164,    -1,   164,    33,   164,    -1,   164,
      34,   164,    -1,   164,    35,   164,    -1,   164,    22,   164,
      -1,   164,    20,   164,    -1,   164,    21,   164,    -1,   164,
      30,   164,    -1,   164,    29,   164,    -1,    40,   164,    -1,
      44,    -1,   164,    24,   164,    -1,   164,    25,   164,    -1,
     164,    26,   164,    -1,   164,    28,   164,    -1,   164,    27,
     164,    -1,   164,    23,   164,    -1,   164,    19,   164,    -1,
     164,    17,   164,    -1,   164,    18,   164,    -1,    39,   164,
      -1,   159,    23,   168,    -1,   159,    24,   168,    -1,    46,
     153,   154,    -1,    47,   153,   164,   154,    -1,    48,   153,
     164,    42,   164,   154,    -1,    49,   153,   164,   154,    -1,
     164,    36,   164,    -1,    43,    -1,   164,    15,   164,    16,
     164,    -1,    45,    -1,    55,    -1,    55,   153,   163,   154,
      -1,    56,   153,   220,   154,    -1,    56,   153,   163,   154,
      -1,    56,   153,   159,   154,    -1,   131,   153,   159,   154,
      -1,   131,   153,   213,   154,    -1,    91,   153,   220,   154,
      -1,    50,   153,   220,   154,    -1,    51,   153,   220,    42,
     220,   154,    -1,    52,   153,   220,    42,   220,   154,    -1,
      53,   153,   220,   154,    -1,    54,   153,   220,   154,    -1,
     120,   153,   164,   154,    -1,    32,   164,    -1,   103,   153,
     213,   154,    -1,   104,   153,   213,   154,    -1,   122,   153,
     164,   154,    -1,   123,   153,   164,   154,    -1,   153,    33,
     160,   154,   153,   166,   154,    -1,   140,   153,   166,   154,
      -1,   209,    -1,   210,    -1,   116,   153,   164,    42,   164,
     154,    -1,   117,   153,   164,    42,   164,   154,    -1,   119,
     153,   164,   154,    -1,   124,   153,   164,   154,    -1,   125,
     153,   164,   154,    -1,   126,   153,   164,   154,    -1,   127,
     153,   164,   154,    -1,   173,    -1,   105,   153,   164,   154,
      -1,   106,   153,   164,    42,   164,    42,   164,   154,    -1,
     113,   153,   164,   154,    -1,   107,   153,   164,   154,    -1,
     114,   153,   164,   154,    -1,   115,   153,   164,   154,    -1,
     136,   153,   220,   154,    -1,   130,   153,   164,    42,   164,
      42,   164,   154,    -1,   121,   153,   168,    42,   168,    42,
     164,   154,    -1,   164,    -1,    22,   164,    -1,   169,    -1,
     159,    -1,   170,    -1,    22,   159,    -1,    -1,   165,    -1,
     166,    42,   165,    -1,   140,   153,   166,   154,    -1,   153,
      33,   160,   154,   153,   166,   154,    -1,    22,   163,    -1,
     170,    -1,   159,    -1,   118,   153,   163,    42,   213,    42,
     184,   154,    -1,   141,   153,   166,   154,    -1,    22,   140,
      -1,   171,    -1,   160,    -1,   159,     3,   168,    -1,   160,
       3,   169,    -1,   161,     3,   214,    -1,   163,     3,   164,
      -1,   163,     6,    -1,     6,   163,    -1,     7,   163,    -1,
     163,     7,    -1,   163,     5,   164,    -1,   163,     4,   164,
      -1,   163,     8,   164,    -1,   163,     9,   164,    -1,   163,
      11,   164,    -1,   163,    10,   164,    -1,   163,    12,   164,
      -1,   163,    13,   164,    -1,   163,    14,   164,    -1,   207,
     157,    -1,   176,    -1,   205,    -1,   149,    16,    -1,   146,
      16,    -1,     1,   157,    -1,   174,    -1,   208,    -1,   175,
     174,    -1,   175,   208,    -1,   155,   175,   156,    -1,   155,
     197,   156,    -1,   155,   197,   175,   156,    -1,   155,   156,
      -1,   155,     1,   156,    -1,   155,   175,   132,    -1,   149,
     153,   180,   154,    -1,   140,   153,   180,   154,    -1,   183,
     177,   176,    -1,   183,    33,   177,   176,    -1,   183,   178,
     176,    -1,    -1,   196,    -1,   180,    42,   196,    -1,   179,
      -1,   198,    -1,   174,    -1,   208,    -1,   181,   132,    -1,
     181,    -1,   182,   181,   132,    -1,   182,   181,    -1,     1,
     132,    -1,   184,    -1,    71,    -1,    72,    -1,    73,    -1,
      74,    -1,    75,    -1,    85,    -1,    86,    -1,    87,    -1,
      88,    -1,    76,    -1,    77,    -1,    78,    -1,    80,    -1,
      81,    -1,    82,    -1,    83,    -1,    79,    -1,    84,    -1,
     150,    -1,    98,   153,   213,   154,    -1,    95,   153,   213,
     154,    -1,   102,    -1,   102,   153,   154,    -1,   102,   153,
     164,   154,    -1,   101,    -1,   101,   153,   154,    -1,   101,
     164,    -1,   101,   153,   168,   154,    -1,   101,   168,    -1,
     137,   153,   220,    42,   164,   154,    -1,   138,   153,   220,
     154,    -1,   142,   153,   214,   154,    -1,   142,   153,   154,
      -1,   161,    -1,   149,    -1,    89,   153,   190,    42,   219,
     154,    -1,    92,   153,   190,    42,   219,   154,    -1,   152,
     153,   166,   154,    -1,   109,   153,   168,    42,   164,    42,
     164,    42,   164,    42,   164,   154,    -1,   110,   153,   154,
      -1,   111,   153,   168,    42,   164,    42,   164,    42,   164,
      42,   164,    42,   164,   154,    -1,   112,   153,   140,   154,
      -1,   135,   153,   220,   154,    -1,    97,   153,   216,   154,
      -1,    93,   153,   214,   154,    -1,    94,   153,   214,   154,
      -1,    96,   153,   214,   154,    -1,   149,    -1,   143,    -1,
     144,    -1,   145,    -1,   184,    -1,   193,    -1,   177,    -1,
     178,    -1,   153,    33,   193,   154,   153,   180,   154,    -1,
     193,   155,   164,   156,    -1,   193,    37,   164,    38,    -1,
     193,    37,   164,    38,   155,   164,   156,    -1,   193,    37,
     164,    38,    37,   164,    38,    -1,   193,    37,   164,    38,
      37,   164,    38,   155,   164,   156,    -1,   193,    37,   164,
      38,    37,   164,    38,    37,   164,    38,    -1,   193,    37,
     164,    38,    37,   164,    38,    37,   164,    38,   155,   164,
     156,    -1,   193,   155,   156,    -1,   193,    37,    38,    -1,
     193,    37,    38,   155,   156,    -1,   193,    37,    38,    37,
      38,    -1,   193,    37,    38,    37,    38,   155,   156,    -1,
     193,    37,    38,    37,    38,    37,    38,    -1,   193,    37,
      38,    37,    38,    37,    38,   155,   156,    -1,    33,   193,
      -1,    55,    -1,    56,    -1,   194,    -1,   194,     3,   164,
      -1,   195,    42,   194,    -1,   195,    42,   194,     3,   164,
      -1,   183,   194,    -1,   198,    -1,   197,   198,    -1,   183,
     195,   157,    -1,   151,   183,   195,   157,    -1,    70,   183,
     195,   157,    -1,   174,    -1,   208,    -1,    60,   153,   164,
     154,   199,    -1,    61,   153,   164,   154,   199,    -1,    69,
     153,   207,   157,   164,   157,   207,   154,   199,    -1,    68,
     199,    60,   153,   164,   154,   157,    -1,    68,   199,    61,
     153,   164,   154,   157,    -1,   202,   175,    -1,   203,    -1,
     202,   203,    -1,    64,   164,    16,    -1,    65,    16,    -1,
     201,    -1,   204,   201,    -1,    63,   153,   164,   154,   155,
     204,   156,    -1,    99,   153,   164,   154,   199,    -1,    99,
     153,   164,   154,   199,   100,   199,    -1,    -1,   185,    -1,
     192,    -1,   191,    -1,   188,    -1,   189,    -1,   187,    -1,
     186,    -1,   173,    -1,   170,    -1,   171,    -1,   172,    -1,
     167,    -1,    66,    -1,    62,    -1,    67,   146,    -1,    67,
     149,    -1,   206,    -1,   200,    -1,   155,   211,   156,    -1,
      37,   212,    38,    -1,   164,    -1,   211,    42,   164,    -1,
     164,    -1,   212,    42,   164,    -1,   164,    -1,   213,    42,
     164,    -1,   164,    -1,   214,    42,   164,    -1,   164,    -1,
     159,    -1,   215,    -1,   216,    42,   215,    -1,   220,    -1,
     217,    42,   220,    -1,   148,    -1,    90,   153,   163,   154,
      -1,   164,    -1,   218,    -1,   161,    -1,   172,    -1,   163,
      -1,   153,   172,   154,    -1,   153,   214,   154,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   405,   405,   412,   419,   426,   432,   449,   453,   457,
     472,   475,   478,   481,   484,   488,   496,   500,   507,   518,
     519,   524,   526,   528,   530,   532,   534,   536,   538,   540,
     542,   544,   546,   548,   552,   555,   558,   561,   564,   567,
     570,   573,   576,   579,   582,   597,   601,   606,   611,   616,
     622,   625,   629,   634,   638,   645,   650,   654,   658,   665,
     671,   675,   679,   684,   690,   696,   703,   716,   722,   725,
     729,   734,   736,   739,   743,   755,   758,   761,   765,   768,
     773,   774,   775,   779,   783,   784,   787,   791,   794,   797,
     801,   805,   807,   810,   817,   818,   827,   828,   829,   830,
     837,   842,   843,   852,   863,   873,   875,   876,   877,   882,
     896,   901,   902,   905,   910,   915,   927,   930,   931,   932,
     933,   934,   940,   946,   952,   958,   964,   970,   976,   982,
     995,   997,   998,   999,  1007,  1012,  1018,  1019,  1020,  1031,
    1044,  1049,  1053,  1057,  1061,  1065,  1072,  1086,  1122,  1133,
    1145,  1168,  1171,  1175,  1181,  1182,  1190,  1200,  1209,  1211,
    1213,  1220,  1229,  1236,  1240,  1241,  1242,  1243,  1244,  1245,
    1246,  1247,  1248,  1249,  1250,  1251,  1252,  1253,  1254,  1255,
    1256,  1257,  1258,  1262,  1264,  1268,  1269,  1270,  1273,  1277,
    1287,  1291,  1295,  1301,  1303,  1307,  1312,  1319,  1320,  1328,
    1332,  1341,  1355,  1362,  1367,  1375,  1382,  1387,  1388,  1389,
    1390,  1393,  1394,  1396,  1398,  1400,  1407,  1418,  1422,  1426,
    1432,  1436,  1440,  1444,  1451,  1455,  1459,  1465,  1469,  1474,
    1479,  1484,  1489,  1494,  1499,  1504,  1512,  1559,  1560,  1563,
    1565,  1576,  1586,  1587,  1594,  1605,  1615,  1628,  1629,  1632,
    1639,  1646,  1657,  1662,  1669,  1673,  1674,  1678,  1680,  1684,
    1685,  1689,  1694,  1696,  1712,  1713,  1714,  1715,  1716,  1717,
    1718,  1719,  1720,  1721,  1722,  1723,  1724,  1725,  1726,  1727,
    1732,  1744,  1746,  1755,  1760,  1765,  1766,  1776,  1777,  1783,
    1784,  1791,  1792,  1796,  1797,  1800,  1801,  1805,  1806,  1810,
    1818,  1825,  1830,  1831,  1832,  1833,  1834,  1835
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
  "'~'", "UNARY", "','", "NUMBER", "INT_NUM", "CHAR_CONST", "MATH0_FUNC",
  "MATH1_FUNC", "MATH2_FUNC", "INT1_FUNC", "STR1_FUNC", "STR2_FUNC",
  "STR3_FUNC", "STRV_FUNC", "CHAR_FUNC", "DATA_FUNC", "SIZE_FUNC",
  "TS_FUNC", "BEGIN_COMMENT", "END_COMMENT", "WHILE", "UNTIL", "CONTINUE",
  "SWITCH", "CASE", "DEFAULT", "BREAK", "GOTO", "DO", "FOR", "STATIC",
  "BYTE", "CHAR", "STRING", "FLOAT", "DOUBLE", "SHORT", "INT32", "INT64",
  "BIT", "UBYTE", "USHORT", "UINT32", "UINT64", "COLOR", "COMPLEX",
  "DBLCPX", "QUATERNION", "DBLQUAT", "STRCPY", "NAME_FUNC", "FILE_EXISTS",
  "STRCAT", "ECHO", "ADVISE_FUNC", "DISPLAY", "F_WARN", "PRINT", "INFO",
  "IF", "ELSE", "RETURN", "EXIT", "MINVAL", "MAXVAL", "WRAP", "SCROLL",
  "DILATE", "FIX_SIZE", "FILL", "CLR_OPT_PARAMS", "ADD_OPT_PARAM",
  "OPTIMIZE", "ERODE", "ENLARGE", "REDUCE", "WARP", "LOOKUP",
  "EQUIVALENCE", "TRANSPOSE", "CONJ", "MAX_TIMES", "MAX_INDEX",
  "MIN_INDEX", "DFT", "IDFT", "RDFT", "RIDFT", "REAL_PART", "IMAG_PART",
  "RAMP", "SUM", "END", "NEXT_TOKEN", "NEWLINE", "SET_OUTPUT_FILE", "LOAD",
  "SAVE", "FILETYPE", "OBJ_OF", "FUNCNAME", "REFFUNC", "SCRIPTFUNC",
  "OBJNAME", "PTRNAME", "STRNAME", "LABELNAME", "FUNCPTRNAME",
  "LEX_STRING", "NEWNAME", "VOID_TYPE", "EXTERN", "NATIVE_FUNC_NAME",
  "'('", "')'", "'{'", "'}'", "';'", "$accept", "pointer", "func_ptr",
  "str_ptr", "subsamp_spec", "objref", "expression", "func_arg",
  "func_args", "void_call", "ref_arg", "func_ref_arg", "ptr_assgn",
  "funcptr_assgn", "str_assgn", "assignment", "statline", "stat_list",
  "stat_block", "new_func_decl", "old_func_decl", "subroutine",
  "arg_decl_list", "prog_elt", "program", "data_type", "precision",
  "info_stat", "exit_stat", "return_stat", "fileio_stat", "script_stat",
  "str_ptr_arg", "misc_stat", "print_stat", "decl_identifier", "decl_item",
  "decl_item_list", "arg_decl", "decl_stat_list", "decl_statement",
  "loop_stuff", "loop_statement", "case_statement", "case_list",
  "single_case", "switch_cases", "switch_statement", "if_statement",
  "simple_stat", "blk_stat", "comp_stack", "list_obj", "comp_list",
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
     377,   378,   379,   380,   381,   382,   383,   384,   385,   386,
     387,   388,   389,    40,    41,   123,   125,    59
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,   158,   159,   160,   161,   162,   163,   163,   163,   163,
     163,   163,   163,   163,   163,   163,   163,   163,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   164,   164,   165,   165,   165,   165,   165,   165,
     165,   166,   166,   167,   167,   168,   168,   168,   168,   168,
     169,   169,   169,   170,   171,   172,   173,   173,   173,   173,
     173,   173,   173,   173,   173,   173,   173,   173,   173,   173,
     174,   174,   174,   174,   174,   174,   175,   175,   175,   175,
     176,   176,   176,   176,   176,   176,   177,   178,   179,   179,
     179,   180,   180,   180,   181,   181,   181,   181,   182,   182,
     182,   182,   182,   183,   184,   184,   184,   184,   184,   184,
     184,   184,   184,   184,   184,   184,   184,   184,   184,   184,
     184,   184,   184,   185,   185,   186,   186,   186,   187,   187,
     187,   187,   187,   188,   188,   189,   189,   190,   190,   191,
     191,   191,   191,   191,   191,   191,   191,   192,   192,   192,
     192,   193,   193,   193,   193,   193,   194,   194,   194,   194,
     194,   194,   194,   194,   194,   194,   194,   194,   194,   194,
     194,   194,   194,   194,   194,   194,   194,   195,   195,   195,
     195,   196,   197,   197,   198,   198,   198,   199,   199,   200,
     200,   200,   200,   200,   201,   202,   202,   203,   203,   204,
     204,   205,   206,   206,   207,   207,   207,   207,   207,   207,
     207,   207,   207,   207,   207,   207,   207,   207,   207,   207,
     207,   208,   208,   209,   210,   211,   211,   212,   212,   213,
     213,   214,   214,   215,   215,   216,   216,   217,   217,   218,
     218,   219,   220,   220,   220,   220,   220,   220
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     1,     1,     1,     5,     1,     2,     4,     1,
       4,     4,     4,     4,     6,     6,     4,     4,     4,     1,
       4,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     2,     1,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     2,     3,     3,     3,     4,     6,     4,
       3,     1,     5,     1,     1,     4,     4,     4,     4,     4,
       4,     4,     4,     6,     6,     4,     4,     4,     2,     4,
       4,     4,     4,     7,     4,     1,     1,     6,     6,     4,
       4,     4,     4,     4,     1,     4,     8,     4,     4,     4,
       4,     4,     8,     8,     1,     2,     1,     1,     1,     2,
       0,     1,     3,     4,     7,     2,     1,     1,     8,     4,
       2,     1,     1,     3,     3,     3,     3,     2,     2,     2,
       2,     3,     3,     3,     3,     3,     3,     3,     3,     3,
       2,     1,     1,     2,     2,     2,     1,     1,     2,     2,
       3,     3,     4,     2,     3,     3,     4,     4,     3,     4,
       3,     0,     1,     3,     1,     1,     1,     1,     2,     1,
       3,     2,     2,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     4,     4,     1,     3,     4,     1,     3,
       2,     4,     2,     6,     4,     4,     3,     1,     1,     6,
       6,     4,    12,     3,    14,     4,     4,     4,     4,     4,
       4,     1,     1,     1,     1,     1,     1,     1,     1,     7,
       4,     4,     7,     7,    10,    10,    13,     3,     3,     5,
       5,     7,     7,     9,     2,     1,     1,     1,     3,     3,
       5,     2,     1,     2,     3,     4,     4,     1,     1,     5,
       5,     9,     7,     7,     2,     1,     2,     3,     2,     1,
       2,     7,     5,     7,     0,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     1,     2,
       2,     1,     1,     3,     3,     1,     3,     1,     3,     1,
       3,     1,     3,     1,     1,     1,     3,     1,     3,     1,
       4,     1,     1,     1,     1,     1,     3,     3
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint16 yydefact[] =
{
       0,     0,     0,     0,     0,     0,     0,   278,     0,   277,
       0,     0,     0,     0,   164,   165,   166,   167,   168,   173,
     174,   175,   180,   176,   177,   178,   179,   181,   169,   170,
     171,   172,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   188,   185,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     6,     2,     4,     0,     3,
       9,   182,     0,     0,     0,     0,     0,     0,     0,     0,
     276,   273,   274,   275,   272,   156,   131,   154,   159,     0,
       0,   163,   265,   271,   270,   268,   269,   267,   266,   155,
     282,   132,   281,     0,   157,   162,   135,     9,   118,   119,
       7,     0,     0,     0,   279,   280,     0,   247,     0,   248,
     264,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    51,    33,    53,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    54,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   299,     0,
       0,   107,   303,   305,   190,   192,   106,   304,    84,    75,
      76,   302,    19,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   100,     0,   134,   133,     0,   100,
       0,     0,   143,   136,     0,     0,     0,   242,   137,     0,
       0,     0,     0,     0,     0,   117,   120,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   158,     1,   161,     0,
     235,   236,     0,   212,   213,   214,   211,     0,   217,   218,
     215,   216,   237,     0,   130,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   217,   218,     0,   198,   197,     0,
       0,   291,     0,     0,   289,     0,     0,   294,   293,   295,
       0,     0,     0,   105,    68,   287,     0,    43,    32,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   100,   100,     0,   189,
     291,     0,   304,     0,     0,   285,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     186,     0,   107,     0,   203,     0,     0,     0,     0,     0,
     305,     0,     0,     0,     0,   297,     0,    97,   112,    94,
     101,     0,    96,    98,   111,   196,     0,     0,     0,     0,
     144,   145,   140,   138,   139,   141,     0,   243,   113,     0,
     114,   115,   116,   122,   121,   123,   124,   126,   125,   127,
     128,   129,     0,     0,     0,     0,   160,     0,   234,   151,
     151,     0,   148,   150,     0,     0,     0,     0,   244,     0,
       0,     0,     0,     0,     0,   211,   246,     0,     0,     0,
     208,   209,     0,   184,   210,     0,   207,   183,     0,   284,
       0,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    21,   191,   306,     0,   307,     0,
     283,    44,    45,     0,    41,    42,    40,    28,    29,    27,
      39,    34,    35,    36,    38,    37,    31,    30,    22,    23,
      24,    25,    26,    50,   187,     0,     0,   205,    10,    11,
     206,     0,   194,     0,     8,   110,    99,    95,   100,   103,
     195,   245,   201,     0,   142,   110,    16,     0,    12,    17,
       0,    13,   149,     0,     0,   152,     0,     0,   228,     0,
     227,     0,   238,   239,   249,   250,     0,     0,     0,     0,
     301,     0,     0,   292,   290,   296,   262,   288,    47,     0,
      49,    62,     0,     0,    65,    66,    55,    58,    57,    56,
     300,    61,    69,    70,    85,     0,    88,    18,    87,    89,
      90,     0,     0,     0,    79,    67,     0,    71,    72,    80,
      81,    82,    83,     0,    59,    60,    91,    74,   109,     0,
      20,   286,     0,     0,     0,     0,   298,   102,   100,     0,
       0,     0,   147,   241,   146,     0,     0,     0,   221,   220,
       0,     0,     0,   259,     0,   255,     0,     0,     0,   264,
     199,   200,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   100,    52,     0,     0,   193,     0,     0,    14,
      15,   153,   151,   230,   229,     0,     0,   240,     0,   258,
       0,   256,   261,   260,   252,   253,     0,   263,    48,    63,
      64,     0,    77,    78,     0,     0,     0,     0,     0,     0,
     104,     5,     0,     0,     0,     0,     0,   257,     0,     0,
       0,     0,     0,    73,     0,     0,   219,   232,   231,   223,
     222,   251,    86,   108,    93,    92,     0,     0,     0,     0,
       0,     0,     0,   233,     0,     0,     0,     0,   225,   224,
     202,     0,     0,     0,     0,   204,   226
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,   246,    67,   172,   402,   173,   369,   370,   371,    70,
     175,   372,    71,    72,   177,   178,   107,   204,    76,   254,
     255,    77,   543,    78,    79,   544,    81,    82,    83,    84,
      85,    86,   259,    87,    88,   241,   242,   243,   545,   206,
      89,   108,    90,   633,   634,   635,   636,    91,    92,    93,
     109,   179,   180,   326,   276,   265,   324,   269,   270,   364,
     181,   561,   182
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -406
static const yytype_int16 yypact[] =
{
    1477,   -89,   242,   242,  -104,   -96,   -90,  -406,   -84,  -406,
     -38,  1889,   -69,  2531,  -406,  -406,  -406,  -406,  -406,  -406,
    -406,  -406,  -406,  -406,  -406,  -406,  -406,  -406,  -406,  -406,
    -406,  -406,   -61,   -56,   -51,   -46,   -44,   -40,   -36,   -30,
     -28,  2402,   -25,   -21,   -17,   -12,    52,    65,    94,   128,
     132,   140,   143,   145,   154,  -406,  -406,  -406,    61,  -406,
     147,  -406,  2531,   155,   261,  1279,   177,   308,   309,   255,
    -406,  -406,  -406,  -406,  -406,  -406,  -406,  -406,   183,  1180,
    4171,  -406,  -406,  -406,  -406,  -406,  -406,  -406,  -406,  -406,
    -406,  -406,  -406,   162,  -406,  -406,  -406,  -406,     5,     5,
    -406,  3034,  3034,  3034,  -406,  -406,   167,  -406,   223,  -406,
    3264,  4254,   -83,   -83,  3034,  3034,  3034,  3034,  3034,  3034,
    3034,   242,  3034,  3034,  3034,  3034,  -406,  -406,  -406,   173,
     176,   186,   187,   190,   191,   194,   195,   196,   198,   199,
     205,   208,   212,   214,   216,   219,   220,   221,   222,   225,
     227,   240,   241,   244,   245,   246,   247,   249,   250,   251,
     252,   253,   256,   259,   260,   262,   263,   265,  -406,  2015,
    3034,    36,   309,   255,  4950,  -406,  -406,  -406,  -406,  -406,
    -406,  -406,  -406,  2656,    26,   168,    26,   239,   242,   242,
     344,   344,   344,   344,  2530,  2782,  -406,  -406,  4254,  2530,
     267,  -105,  -406,  -406,  1576,  4254,  1378,  -406,  -406,    26,
       8,  3034,  3034,  3034,  3034,  -406,  -406,  3034,  3034,  3034,
    3034,  3034,  3034,  3034,  3034,  3034,  -406,  -406,   213,  4272,
    -406,  -406,   268,  -406,  -406,  -406,   270,   330,   232,   232,
    -406,    19,   386,   -18,  -406,  2141,   264,  1130,  3425,  3447,
     271,   272,   273,  4290,  -406,  -406,   -11,  -406,  -406,   366,
     375,  4950,     2,    12,  4950,    39,    49,   264,  4950,  -406,
      53,    54,  3469,     5,  -406,  4950,    56,  -406,  -406,   278,
    3034,  3034,  3034,   344,   344,   344,   344,   344,   242,   749,
     242,   344,  3034,  3034,  3034,  3034,  3034,  3034,  3034,  3034,
    3034,  3034,  3034,   242,  3034,  3034,    26,  3034,  3034,  3034,
    3034,  3034,  3034,  3034,  3034,   344,  2530,  2530,   -20,  -406,
    3491,   283,   285,   288,    72,  4950,   -13,    26,    26,  3034,
    3034,  3034,  3034,  3034,  3034,  3034,  3034,  3034,  3034,  3034,
    3034,  3034,  3034,  3034,  3034,  3034,  3034,  3034,  3034,  3034,
    -406,  3513,   177,   384,  -406,   385,   289,    18,    27,  3034,
       5,   292,   406,   296,    74,  -406,  3160,    36,   308,  4950,
    -406,   119,  -406,  -406,  -406,  -406,   123,    -9,   141,   297,
    -406,  -406,  -406,  -406,  -406,  -406,  1790,  -406,  -406,   312,
    -406,   411,  4950,  4950,  4950,  4950,  4950,  4950,  4950,  4950,
    4950,  4950,   419,  4792,   302,  1039,  -406,   232,  -406,  2531,
    2531,  4290,  -406,  -406,  2908,   867,  3034,  4254,  -406,  1889,
    1889,   305,  3034,  3034,  3034,  -406,  -406,  3034,  3034,  3034,
    -406,  -406,  3034,  -406,  -406,  3034,  -406,  -406,  1889,  -406,
    3034,  -406,  3563,  4426,  3585,   307,   420,   422,   314,   315,
      68,   316,    78,   317,   101,   320,   159,   169,  3607,  4452,
    3629,  3651,  3673,  3701,  3723,  4478,  4504,    45,  3745,  3767,
     433,  3789,  3811,  3839,  3861,  3883,  3905,  4530,     4,   188,
     323,   192,   210,   324,  -406,  -406,  -406,  3034,  -406,  3034,
    -406,  -406,  -406,  4906,  1628,  1742,  1106,  1713,  1013,  1060,
     721,   721,   469,   469,   469,   469,   269,   269,   321,   321,
     443,   443,   443,  -406,  -406,  3034,  3034,  -406,  -406,  -406,
    -406,  3034,  -406,   344,  -406,   263,   264,  4950,  2530,  -406,
    -406,  -406,  -406,   327,  -406,  -406,  -406,  3034,  -406,  -406,
    3034,  -406,  -406,   228,  4254,  -406,   229,   336,    33,  4838,
    -406,  2698,  4950,   481,  -406,  -406,   226,  3927,  3949,   744,
    4950,   340,   341,  -406,  4950,  -406,   396,  4950,  -406,  3034,
    -406,  -406,   344,   344,  -406,  -406,  -406,  -406,  -406,  -406,
    -406,  -406,  -406,  -406,  -406,  3034,  -406,  -406,  -406,  -406,
    -406,  3034,  3034,  3034,  -406,  -406,    26,  -406,  -406,  -406,
    -406,  -406,  -406,  3034,  -406,  -406,  -406,  -406,  -406,   353,
    -406,  4950,  3034,  4556,  4582,  3977,  -406,  -406,  2530,  4816,
    2444,  2531,  -406,  -406,  -406,   354,   480,   363,    62,  -406,
    3034,  3034,   504,  -406,  1691,  -406,   -19,   365,   367,  3264,
    -406,  -406,  1889,  3999,   369,   372,  4608,  4021,  4043,   485,
     487,  4634,  2530,  3315,  3034,  3034,  -406,   230,  3034,  -406,
    -406,  -406,  2531,    66,  -406,  3034,  3034,  4950,  4928,  -406,
     448,  -406,  -406,  -406,  -406,  -406,   379,  -406,  -406,  -406,
    -406,  3034,  -406,  -406,  2267,  3034,  3034,   234,  4660,  4686,
    -406,  4950,   236,   496,   380,  4860,  2950,  -406,  1889,  4065,
     381,  4087,  4115,  -406,  3034,  3034,  -406,   383,  -406,    69,
    -406,  -406,  -406,  -406,  -406,  -406,  4712,  4738,   392,  3034,
    3034,  3034,  3034,  -406,  4882,  3207,  4137,  4764,   397,  -406,
    -406,  3034,  3034,  4159,  3403,  -406,  -406
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -406,     0,  -127,    21,   326,    47,   513,    11,  -196,  -406,
    -108,   343,    -7,  -187,     9,    25,     6,  -198,  -222,   -65,
     475,  -406,  -405,   477,  -406,    80,   -58,  -406,  -406,  -406,
    -406,  -406,   449,  -406,  -406,  -215,  -391,   -76,   -60,  -406,
     -52,  -401,  -406,   -73,  -406,   -70,  -406,  -406,  -406,  -109,
      10,  -406,  -406,  -406,  -406,  -117,   165,   130,  -406,  -406,
    -406,   138,   -41
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -265
static const yytype_int16 yytable[] =
{
      66,   252,   271,   378,   100,   546,    75,   374,   386,    73,
      94,    66,   374,   207,   408,   238,   412,   413,   554,   555,
      73,    68,   240,   374,   417,    74,   553,   327,   328,   489,
     389,   417,    68,   417,   176,   256,    74,   566,   408,   209,
      56,   171,   224,    95,   429,   631,   632,    69,   121,    98,
      99,   380,    96,   240,   429,   224,   414,   101,    69,   327,
     328,   321,    57,   102,   224,    66,   257,   368,    96,   103,
     626,   203,   368,   379,    73,   208,   353,   196,   355,    66,
      80,   432,   224,   368,   110,    75,    68,   593,    73,    94,
      74,   429,   112,   111,   439,   435,   432,   113,   440,   665,
      68,   388,   114,   693,    74,   224,   719,   115,   104,   116,
      66,   105,    69,   117,   429,   224,   523,   118,   267,    73,
     481,   482,   377,   119,    56,   120,    69,    59,   183,   374,
     374,    68,   184,   258,   258,    74,   185,   672,   224,   418,
     240,   186,   198,   490,   153,   205,   426,   240,   531,   361,
     362,   363,   365,   623,   387,    59,   430,    69,   604,    80,
     225,   528,   176,   197,   407,   429,   431,   167,   273,   171,
      56,   240,   518,   225,   415,   456,   457,   176,   322,   176,
     209,   519,   225,   528,   352,   542,   352,   373,   627,   368,
     368,   483,   373,   433,   367,   240,   547,   479,   470,   367,
     225,   432,   176,   434,    66,   187,    66,   436,   437,   352,
     383,   432,   203,    73,   384,    73,   208,   666,   188,   491,
     492,   694,   576,   225,   720,    68,   488,    68,   524,    74,
     432,    74,   578,   225,   528,   357,   358,   360,   360,   360,
     360,   677,   445,   446,   447,   448,   449,   189,   453,   323,
     455,    69,   528,    69,   322,   580,   225,   692,   212,   213,
     214,   215,   216,   217,   218,   219,   220,   221,   222,   223,
     621,   621,   528,   529,   480,     4,   528,   530,   621,   262,
     263,   190,   266,   250,   251,   191,   205,   327,   328,   451,
     631,   632,   224,   192,   200,   532,   193,   711,   194,   176,
     344,   345,   346,   347,   348,   349,   352,   195,   199,   373,
     373,   210,   211,   582,   478,   226,   367,   367,   100,   244,
     176,   176,   354,   583,    96,   323,   279,   352,   352,   280,
     360,   360,   360,   360,   360,   450,   452,   454,   360,   281,
     282,   374,   605,   283,   284,   406,   607,   285,   286,   287,
     467,   288,   289,   240,   346,   347,   348,   349,   290,   240,
     376,   291,   360,   411,   608,   292,   526,   293,   322,   294,
      47,    48,   295,   296,   297,   298,   391,     4,   299,   356,
     300,    52,   622,   624,   690,    55,    66,    65,   703,   416,
     706,    97,   383,   301,   302,    73,   384,   303,   304,   305,
     306,   368,   307,   308,   309,   310,   311,    68,   427,   312,
     225,    74,   313,   314,    59,   315,   316,   428,   317,    66,
      66,   409,   657,   410,   422,   423,   515,   516,    73,    73,
     424,   374,   441,    69,   140,   267,   670,   485,    66,   486,
      68,    68,   487,   517,    74,    74,   520,    73,   521,   106,
     522,   533,   535,   429,     2,     3,   687,   536,   539,    68,
     556,   571,   572,    74,   573,   374,    69,    69,   574,   575,
     577,   579,    47,    48,   581,   596,   649,   606,   609,   349,
     618,     4,   616,    52,   630,    69,   240,    55,   650,    57,
     625,   368,   168,    97,   640,   641,   642,   359,   342,   343,
     344,   345,   346,   347,   348,   349,   652,   662,     5,     6,
       7,     8,  -254,  -254,     9,    10,    11,    12,   663,   664,
     669,   373,   674,   679,   675,   368,   680,   684,   367,   685,
     676,   644,   645,   698,   707,   713,   708,    32,   718,   617,
      33,    34,    35,    36,    37,    38,    39,    40,   723,    41,
      42,   404,   732,   390,   174,   239,   228,    43,    44,    45,
      46,   661,   260,   673,   671,   565,   562,     0,     0,     0,
     360,     0,     0,     0,     0,     0,    47,    48,     0,     0,
       0,     0,     0,    49,     0,    50,    51,    52,    53,   176,
      54,    55,    56,    57,    58,    59,   352,    60,     0,     0,
      63,    64,     0,    65,  -254,  -264,     0,     0,     0,     0,
       0,   373,     0,     0,   247,   248,   249,     0,   367,   360,
     360,     0,     0,     0,     0,     0,   700,   261,   261,   264,
     261,   268,   264,   272,    66,   274,   275,   277,   278,    66,
     203,     0,    66,    73,   208,   373,     0,     0,    73,     0,
       0,    73,   367,     0,     0,    68,     0,     0,     0,    74,
      68,     0,     0,    68,    74,     0,     0,    74,     0,     0,
      66,     0,     0,     0,     0,     0,   383,     0,     0,    73,
     384,    69,   320,   325,     0,     0,    69,     0,     0,    69,
       0,    68,     0,     0,     0,    74,   351,     0,    66,     0,
       0,     0,     0,     0,     0,     0,     0,    73,   261,     0,
       0,     0,     0,     0,     0,     0,     0,    69,     0,    68,
       0,     0,     0,    74,   261,   392,   393,   394,     0,     0,
     395,   396,   397,   398,   399,   400,   401,   403,   405,     0,
       0,     0,     0,     0,     0,    69,   338,   339,   340,   341,
     342,   343,   344,   345,   346,   347,   348,   349,   320,   329,
       0,   330,   331,   332,   333,   334,   335,   336,   337,   338,
     339,   340,   341,   342,   343,   344,   345,   346,   347,   348,
     349,     0,     4,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   442,   443,   444,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   264,   264,   458,   459,   460,
     461,   462,   463,   464,   465,   466,     0,   468,   469,     0,
     471,   472,   473,   474,   475,   476,   477,   264,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   140,
       0,     0,   493,   494,   495,   496,   497,   498,   499,   500,
     501,   502,   503,   504,   505,   506,   507,   508,   509,   510,
     511,   512,   513,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   261,     2,     3,     0,     0,    47,    48,   527,
       0,     0,     0,     0,     0,     0,     0,     0,    52,     0,
       0,     0,    55,    56,    57,     0,     0,   168,    97,   122,
       4,   639,   359,     0,   123,     0,   124,   125,     0,     0,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,   139,     0,     0,     0,   549,   551,   552,
       0,     0,     0,     0,     0,   557,   558,   559,     0,     0,
     560,   560,   563,     0,     0,   564,     0,     0,   268,     0,
       0,     0,     0,   567,     0,     0,     0,   140,   141,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     142,   143,   144,   145,   146,   147,     0,     0,     0,     0,
     148,   149,   150,   151,   152,     0,   154,   155,   156,   157,
     158,   159,   160,   161,   162,    47,    48,   163,   164,     0,
     610,     0,   611,   165,     0,     0,    52,   166,     0,     0,
      55,    56,    57,     0,     0,   168,    97,     0,     0,     0,
     245,     0,   170,   550,     0,     0,     0,     0,   613,   614,
       0,     0,     0,     0,   615,   335,   336,   337,   338,   339,
     340,   341,   342,   343,   344,   345,   346,   347,   348,   349,
     619,     0,     0,   620,   329,   540,   330,   331,   332,   333,
     334,   335,   336,   337,   338,   339,   340,   341,   342,   343,
     344,   345,   346,   347,   348,   349,     0,     0,     0,     0,
       0,     0,   643,   336,   337,   338,   339,   340,   341,   342,
     343,   344,   345,   346,   347,   348,   349,     0,   646,     0,
       0,     0,     0,     0,   647,   648,   264,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   651,     0,     0,     0,
       0,     0,     0,     0,     0,   653,   333,   334,   335,   336,
     337,   338,   339,   340,   341,   342,   343,   344,   345,   346,
     347,   348,   349,   667,   668,   329,     0,   330,   331,   332,
     333,   334,   335,   336,   337,   338,   339,   340,   341,   342,
     343,   344,   345,   346,   347,   348,   349,   688,   689,     0,
       0,   691,     0,     0,     0,     0,     0,     0,   695,   696,
     227,   106,     0,     0,     0,     0,     2,     3,     0,     0,
       0,     0,     0,     0,   699,   541,     0,   564,   701,   702,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     4,     0,     0,     0,   716,   717,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   724,   725,   726,   727,     0,     0,     0,     0,
       5,     6,     7,     8,   733,   734,     9,    10,    11,    12,
      13,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
       0,     0,    33,    34,    35,    36,    37,    38,    39,    40,
     201,    41,    42,     0,   419,     2,     3,     0,     0,    43,
      44,    45,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    47,    48,
       0,     0,     4,     0,     0,    49,     0,    50,    51,    52,
      53,     0,    54,    55,    56,    57,    58,    59,     0,    60,
      61,    62,    63,    64,     0,    65,     0,  -264,     0,     5,
       6,     7,     8,     0,     0,     9,    10,    11,    12,    13,
      14,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,     0,
       0,    33,    34,    35,    36,    37,    38,    39,    40,   106,
      41,    42,     0,     0,     2,     3,     0,     0,    43,    44,
      45,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     4,     0,     0,    49,     0,    50,    51,    52,    53,
       0,    54,    55,    56,    57,    58,    59,     0,    60,    61,
      62,    63,    64,     0,    65,   202,  -264,     0,     5,     6,
       7,     8,     0,     0,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,     0,     0,
      33,    34,    35,    36,    37,    38,    39,    40,     1,    41,
      42,     0,     0,     2,     3,     0,     0,    43,    44,    45,
      46,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    47,    48,     0,     0,
       4,     0,     0,    49,     0,    50,    51,    52,    53,     0,
      54,    55,    56,    57,    58,    59,     0,    60,    61,    62,
      63,    64,     0,    65,   385,  -264,     0,     5,     6,     7,
       8,     0,     0,     9,    10,    11,    12,    13,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,     0,     0,    33,
      34,    35,    36,    37,    38,    39,    40,   106,    41,    42,
       0,     0,     2,     3,     0,     0,    43,    44,    45,    46,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    47,    48,     0,     0,     4,
       0,     0,    49,     0,    50,    51,    52,    53,     0,    54,
      55,    56,    57,    58,    59,     0,    60,    61,    62,    63,
      64,     0,    65,     0,  -264,     0,     5,     6,     7,     8,
       0,     0,     9,    10,    11,    12,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,    32,     0,     0,    33,    34,
      35,    36,    37,    38,    39,    40,     0,    41,    42,     0,
       0,     0,     0,     0,     0,    43,    44,    45,    46,     0,
       0,     0,   106,     0,     0,     0,     0,     2,     3,     0,
       0,     0,     0,     0,    47,    48,     0,     0,   381,     0,
       0,    49,     0,    50,    51,    52,    53,     0,    54,    55,
      56,    57,    58,    59,     4,    60,     0,     0,    63,    64,
       0,    65,   382,  -264,   334,   335,   336,   337,   338,   339,
     340,   341,   342,   343,   344,   345,   346,   347,   348,   349,
       0,     5,     6,     7,     8,   631,   632,     9,    10,    11,
      12,   332,   333,   334,   335,   336,   337,   338,   339,   340,
     341,   342,   343,   344,   345,   346,   347,   348,   349,     0,
      32,     0,     0,    33,    34,    35,    36,    37,    38,    39,
      40,   106,    41,    42,     0,     0,     2,     3,     0,     0,
      43,    44,    45,    46,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    47,
      48,     0,     0,     4,     0,     0,    49,     0,    50,    51,
      52,    53,     0,    54,    55,    56,    57,    58,    59,     0,
      60,     0,     0,    63,    64,     0,    65,     0,  -264,     0,
       5,     6,     7,     8,     0,     0,     9,    10,    11,    12,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,    32,
       0,     0,    33,    34,    35,    36,    37,    38,    39,    40,
     106,    41,    42,     0,     0,     2,     3,     0,     0,    43,
      44,    45,    46,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    47,    48,
       0,     0,     4,     0,     0,    49,     0,    50,    51,    52,
      53,     0,    54,    55,    56,    57,    58,    59,     0,    60,
       0,     0,    63,    64,     0,    65,   534,  -264,     0,     5,
       6,     7,     8,     0,     0,     9,    10,    11,    12,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    32,     0,
       0,    33,    34,    35,    36,    37,    38,    39,    40,     0,
      41,    42,     0,     0,     0,     0,     0,     0,    43,    44,
      45,    46,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    47,    48,     0,
       0,     2,     3,     0,    49,     0,    50,    51,    52,    53,
       0,    54,    55,    56,    57,    58,    59,   121,    60,     0,
       0,    63,    64,     0,    65,     0,  -264,   122,   318,     0,
       0,     0,   123,     0,   124,   125,     0,     0,   126,   127,
     128,   129,   130,   131,   132,   133,   134,   135,   136,   137,
     138,   139,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    14,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,     0,   140,   141,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   142,   143,
     144,   145,   146,   147,     0,     0,     0,     0,   148,   149,
     150,   151,   152,   153,   154,   155,   156,   157,   158,   159,
     160,   161,   162,    47,    48,   163,   164,     2,     3,     0,
       0,   165,     0,     0,    52,   166,   167,     0,    55,    56,
      57,     0,     0,   168,    97,    61,     0,     0,   245,   319,
     170,     0,     0,   122,   318,     0,     0,     0,   123,     0,
     124,   125,     0,     0,   126,   127,   128,   129,   130,   131,
     132,   133,   134,   135,   136,   137,   138,   139,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
       0,   140,   141,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   142,   143,   144,   145,   146,   147,
       0,     0,     0,     0,   148,   149,   150,   151,   152,     0,
     154,   155,   156,   157,   158,   159,   160,   161,   162,    47,
      48,   163,   164,     2,     3,     0,     0,   165,     0,     0,
      52,   166,     0,     0,    55,    56,    57,     0,     0,   168,
      97,    61,     0,     0,   245,     0,   170,     0,     0,   122,
       4,     0,     0,     0,   123,     0,   124,   125,     0,     0,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,   139,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,    14,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,     0,   140,   141,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     142,   143,   144,   145,   146,   147,     0,     0,     0,     0,
     148,   149,   150,   151,   152,     0,   154,   155,   156,   157,
     158,   159,   160,   161,   162,    47,    48,   163,   164,     0,
       0,     0,     0,   165,     0,     0,    52,   166,     2,     3,
      55,    56,    57,     0,     0,   168,    97,    61,     0,     0,
     245,     0,   170,     0,   121,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   122,     4,     0,     0,     0,   123,
       0,   124,   125,     0,     0,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,   139,   329,
     658,   330,   331,   332,   333,   334,   335,   336,   337,   338,
     339,   340,   341,   342,   343,   344,   345,   346,   347,   348,
     349,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   140,   141,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   142,   143,   144,   145,   146,
     147,     0,     0,     0,     0,   148,   149,   150,   151,   152,
     153,   154,   155,   156,   157,   158,   159,   160,   161,   162,
      47,    48,   163,   164,     0,     0,     2,     3,   165,     0,
       0,    52,   166,   167,     0,    55,    56,    57,     0,     0,
     168,    97,   366,     0,     0,   169,     0,   170,     0,     0,
       0,     0,   122,     4,     0,     0,     0,   123,     0,   124,
     125,     0,     0,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   139,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     660,     0,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
     140,   141,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   142,   143,   144,   145,   146,   147,     0,
       0,     0,     0,   148,   149,   150,   151,   152,     0,   154,
     155,   156,   157,   158,   159,   160,   161,   162,    47,    48,
     163,   164,     2,     3,     0,     0,   165,     0,     0,    52,
     166,     0,     0,    55,    56,    57,     0,    59,   168,    97,
       0,    61,     0,   245,     0,   170,     0,     0,   122,     4,
       0,     0,     0,   123,     0,   124,   125,     0,     0,   126,
     127,   128,   129,   130,   131,   132,   133,   134,   135,   136,
     137,   138,   139,   329,     0,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   140,   141,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   142,
     143,   144,   145,   146,   147,     0,     0,     0,     0,   148,
     149,   150,   151,   152,     0,   154,   155,   156,   157,   158,
     159,   160,   161,   162,    47,    48,   163,   164,     2,     3,
       0,     0,   165,     0,     0,    52,   166,     0,     0,    55,
      56,    57,     0,     0,   168,    97,     0,     0,     0,   245,
     350,   170,     0,     0,   122,     4,     0,     0,     0,   123,
       0,   124,   125,     0,     0,   126,   127,   128,   129,   130,
     131,   132,   133,   134,   135,   136,   137,   138,   139,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   629,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   140,   141,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   142,   143,   144,   145,   146,
     147,     0,     0,     0,     0,   148,   149,   150,   151,   152,
       0,   154,   155,   156,   157,   158,   159,   160,   161,   162,
      47,    48,   163,   164,     2,     3,     0,     0,   165,     0,
       0,    52,   166,     0,     0,    55,    56,    57,     0,     0,
     168,    97,     0,     0,     0,   245,   375,   170,     0,     0,
     122,     4,     0,     0,     0,   123,   548,   124,   125,     0,
       0,   126,   127,   128,   129,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   329,     0,   330,   331,   332,
     333,   334,   335,   336,   337,   338,   339,   340,   341,   342,
     343,   344,   345,   346,   347,   348,   349,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   140,   141,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   142,   143,   144,   145,   146,   147,     0,     0,     0,
       0,   148,   149,   150,   151,   152,     0,   154,   155,   156,
     157,   158,   159,   160,   161,   162,    47,    48,   163,   164,
       2,     3,     0,     0,   165,     0,     0,    52,   166,     0,
       0,    55,    56,    57,     0,     0,   168,    97,     0,     0,
       0,   245,     0,   170,     0,     0,   122,     4,     0,     0,
       0,   123,     0,   124,   125,     0,     0,   126,   127,   128,
     129,   130,   131,   132,   133,   134,   135,   136,   137,   138,
     139,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   710,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   140,   141,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   142,   143,   144,
     145,   146,   147,     0,     0,     0,     0,   148,   149,   150,
     151,   152,     0,   154,   155,   156,   157,   158,   159,   160,
     161,   162,    47,    48,   163,   164,     2,     3,     0,     0,
     165,     0,     0,    52,   166,     0,     0,    55,    56,    57,
       0,     0,   168,    97,     0,     0,     0,   245,     0,   170,
       0,     0,   122,     4,     0,     0,     0,   123,     0,   124,
     125,     0,     0,   126,   127,   128,   129,   130,   131,   132,
     133,   134,   135,   136,   137,   138,   139,     0,     0,     0,
       0,     0,   329,     0,   330,   331,   332,   333,   334,   335,
     336,   337,   338,   339,   340,   341,   342,   343,   344,   345,
     346,   347,   348,   349,     0,     0,     0,     0,     0,     0,
     140,   141,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   142,   143,   144,   145,   146,   147,     0,
       2,     3,     0,   148,   149,   150,   151,   152,     0,   154,
     155,   156,   157,   158,   159,   160,   161,   162,    47,    48,
     163,   164,     0,     0,     0,     0,   165,     4,     0,    52,
     525,     0,     0,    55,    56,    57,     0,     0,   168,    97,
       0,     0,     0,   245,     0,   170,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     7,     0,     0,     0,
       9,    10,   330,   331,   332,   333,   334,   335,   336,   337,
     338,   339,   340,   341,   342,   343,   344,   345,   346,   347,
     348,   349,     0,    32,     0,     0,    33,    34,    35,    36,
      37,    38,    39,   729,     0,    41,    42,     0,     0,     0,
       0,     0,     0,    43,    44,    45,    46,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    47,    48,     0,     0,     0,     0,     0,    49,
       0,    50,    51,    52,    53,     0,    54,    55,    56,    57,
       0,    59,     0,    97,     0,     0,    63,    64,   329,     0,
     330,   331,   332,   333,   334,   335,   336,   337,   338,   339,
     340,   341,   342,   343,   344,   345,   346,   347,   348,   349,
     329,     0,   330,   331,   332,   333,   334,   335,   336,   337,
     338,   339,   340,   341,   342,   343,   344,   345,   346,   347,
     348,   349,   329,     0,   330,   331,   332,   333,   334,   335,
     336,   337,   338,   339,   340,   341,   342,   343,   344,   345,
     346,   347,   348,   349,   329,     0,   330,   331,   332,   333,
     334,   335,   336,   337,   338,   339,   340,   341,   342,   343,
     344,   345,   346,   347,   348,   349,   329,     0,   330,   331,
     332,   333,   334,   335,   336,   337,   338,   339,   340,   341,
     342,   343,   344,   345,   346,   347,   348,   349,   329,     0,
     330,   331,   332,   333,   334,   335,   336,   337,   338,   339,
     340,   341,   342,   343,   344,   345,   346,   347,   348,   349,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   736,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   329,   420,
     330,   331,   332,   333,   334,   335,   336,   337,   338,   339,
     340,   341,   342,   343,   344,   345,   346,   347,   348,   349,
     329,   421,   330,   331,   332,   333,   334,   335,   336,   337,
     338,   339,   340,   341,   342,   343,   344,   345,   346,   347,
     348,   349,   329,   438,   330,   331,   332,   333,   334,   335,
     336,   337,   338,   339,   340,   341,   342,   343,   344,   345,
     346,   347,   348,   349,   329,   484,   330,   331,   332,   333,
     334,   335,   336,   337,   338,   339,   340,   341,   342,   343,
     344,   345,   346,   347,   348,   349,   329,   514,   330,   331,
     332,   333,   334,   335,   336,   337,   338,   339,   340,   341,
     342,   343,   344,   345,   346,   347,   348,   349,   329,     0,
     330,   331,   332,   333,   334,   335,   336,   337,   338,   339,
     340,   341,   342,   343,   344,   345,   346,   347,   348,   349,
       0,     0,     0,     0,     0,     0,   329,   568,   330,   331,
     332,   333,   334,   335,   336,   337,   338,   339,   340,   341,
     342,   343,   344,   345,   346,   347,   348,   349,   329,   570,
     330,   331,   332,   333,   334,   335,   336,   337,   338,   339,
     340,   341,   342,   343,   344,   345,   346,   347,   348,   349,
     329,   584,   330,   331,   332,   333,   334,   335,   336,   337,
     338,   339,   340,   341,   342,   343,   344,   345,   346,   347,
     348,   349,   329,   586,   330,   331,   332,   333,   334,   335,
     336,   337,   338,   339,   340,   341,   342,   343,   344,   345,
     346,   347,   348,   349,   329,   587,   330,   331,   332,   333,
     334,   335,   336,   337,   338,   339,   340,   341,   342,   343,
     344,   345,   346,   347,   348,   349,   329,   588,   330,   331,
     332,   333,   334,   335,   336,   337,   338,   339,   340,   341,
     342,   343,   344,   345,   346,   347,   348,   349,     0,     0,
       0,     0,     0,     0,   329,   589,   330,   331,   332,   333,
     334,   335,   336,   337,   338,   339,   340,   341,   342,   343,
     344,   345,   346,   347,   348,   349,   329,   590,   330,   331,
     332,   333,   334,   335,   336,   337,   338,   339,   340,   341,
     342,   343,   344,   345,   346,   347,   348,   349,   329,   594,
     330,   331,   332,   333,   334,   335,   336,   337,   338,   339,
     340,   341,   342,   343,   344,   345,   346,   347,   348,   349,
     329,   595,   330,   331,   332,   333,   334,   335,   336,   337,
     338,   339,   340,   341,   342,   343,   344,   345,   346,   347,
     348,   349,   329,   597,   330,   331,   332,   333,   334,   335,
     336,   337,   338,   339,   340,   341,   342,   343,   344,   345,
     346,   347,   348,   349,   329,   598,   330,   331,   332,   333,
     334,   335,   336,   337,   338,   339,   340,   341,   342,   343,
     344,   345,   346,   347,   348,   349,     0,     0,     0,     0,
       0,     0,   329,   599,   330,   331,   332,   333,   334,   335,
     336,   337,   338,   339,   340,   341,   342,   343,   344,   345,
     346,   347,   348,   349,   329,   600,   330,   331,   332,   333,
     334,   335,   336,   337,   338,   339,   340,   341,   342,   343,
     344,   345,   346,   347,   348,   349,   329,   601,   330,   331,
     332,   333,   334,   335,   336,   337,   338,   339,   340,   341,
     342,   343,   344,   345,   346,   347,   348,   349,   329,   602,
     330,   331,   332,   333,   334,   335,   336,   337,   338,   339,
     340,   341,   342,   343,   344,   345,   346,   347,   348,   349,
     329,   637,   330,   331,   332,   333,   334,   335,   336,   337,
     338,   339,   340,   341,   342,   343,   344,   345,   346,   347,
     348,   349,   329,   638,   330,   331,   332,   333,   334,   335,
     336,   337,   338,   339,   340,   341,   342,   343,   344,   345,
     346,   347,   348,   349,     0,     0,     0,     0,     0,     0,
     329,   656,   330,   331,   332,   333,   334,   335,   336,   337,
     338,   339,   340,   341,   342,   343,   344,   345,   346,   347,
     348,   349,   329,   678,   330,   331,   332,   333,   334,   335,
     336,   337,   338,   339,   340,   341,   342,   343,   344,   345,
     346,   347,   348,   349,   329,   682,   330,   331,   332,   333,
     334,   335,   336,   337,   338,   339,   340,   341,   342,   343,
     344,   345,   346,   347,   348,   349,     0,   683,     0,     0,
       0,     0,     0,     0,   229,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   712,
       0,     0,     0,     0,     0,     0,   230,   231,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   714,    14,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   715,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,   253,     0,     0,
       0,   730,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   230,
     231,   232,     0,   735,   233,   234,   235,     0,     0,     0,
     236,    61,     0,     0,   237,    14,    15,    16,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    14,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   232,     0,     0,   233,   234,   235,
       0,     0,     0,   236,    61,     0,     0,   237,     0,     0,
       0,     0,     0,     0,     0,   233,   234,   235,     0,     0,
       0,   236,    61,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,   233,   234,   235,     0,     0,     0,   425,
      61,   329,     0,   330,   331,   332,   333,   334,   335,   336,
     337,   338,   339,   340,   341,   342,   343,   344,   345,   346,
     347,   348,   349,     0,     0,     0,     0,   329,   569,   330,
     331,   332,   333,   334,   335,   336,   337,   338,   339,   340,
     341,   342,   343,   344,   345,   346,   347,   348,   349,     0,
       0,     0,     0,   329,   585,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,     0,     0,     0,     0,   329,
     591,   330,   331,   332,   333,   334,   335,   336,   337,   338,
     339,   340,   341,   342,   343,   344,   345,   346,   347,   348,
     349,     0,     0,     0,     0,   329,   592,   330,   331,   332,
     333,   334,   335,   336,   337,   338,   339,   340,   341,   342,
     343,   344,   345,   346,   347,   348,   349,     0,     0,     0,
       0,   329,   603,   330,   331,   332,   333,   334,   335,   336,
     337,   338,   339,   340,   341,   342,   343,   344,   345,   346,
     347,   348,   349,     0,     0,     0,     0,   329,   654,   330,
     331,   332,   333,   334,   335,   336,   337,   338,   339,   340,
     341,   342,   343,   344,   345,   346,   347,   348,   349,     0,
       0,     0,     0,   329,   655,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,     0,     0,     0,     0,   329,
     681,   330,   331,   332,   333,   334,   335,   336,   337,   338,
     339,   340,   341,   342,   343,   344,   345,   346,   347,   348,
     349,     0,     0,     0,     0,   329,   686,   330,   331,   332,
     333,   334,   335,   336,   337,   338,   339,   340,   341,   342,
     343,   344,   345,   346,   347,   348,   349,     0,     0,     0,
       0,   329,   704,   330,   331,   332,   333,   334,   335,   336,
     337,   338,   339,   340,   341,   342,   343,   344,   345,   346,
     347,   348,   349,     0,     0,     0,     0,   329,   705,   330,
     331,   332,   333,   334,   335,   336,   337,   338,   339,   340,
     341,   342,   343,   344,   345,   346,   347,   348,   349,     0,
       0,     0,     0,   329,   721,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,     0,     0,     0,     0,   329,
     722,   330,   331,   332,   333,   334,   335,   336,   337,   338,
     339,   340,   341,   342,   343,   344,   345,   346,   347,   348,
     349,     0,     0,     0,     0,     0,   731,   329,   537,   330,
     331,   332,   333,   334,   335,   336,   337,   338,   339,   340,
     341,   342,   343,   344,   345,   346,   347,   348,   349,     0,
     538,   329,   658,   330,   331,   332,   333,   334,   335,   336,
     337,   338,   339,   340,   341,   342,   343,   344,   345,   346,
     347,   348,   349,   329,   659,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   329,   628,   330,   331,   332,
     333,   334,   335,   336,   337,   338,   339,   340,   341,   342,
     343,   344,   345,   346,   347,   348,   349,   329,   709,   330,
     331,   332,   333,   334,   335,   336,   337,   338,   339,   340,
     341,   342,   343,   344,   345,   346,   347,   348,   349,     0,
     728,   329,   612,   330,   331,   332,   333,   334,   335,   336,
     337,   338,   339,   340,   341,   342,   343,   344,   345,   346,
     347,   348,   349,   329,   697,   330,   331,   332,   333,   334,
     335,   336,   337,   338,   339,   340,   341,   342,   343,   344,
     345,   346,   347,   348,   349,   329,     0,   330,   331,   332,
     333,   334,   335,   336,   337,   338,   339,   340,   341,   342,
     343,   344,   345,   346,   347,   348,   349
};

static const yytype_int16 yycheck[] =
{
       0,   110,   119,   199,     4,   410,     0,   194,   206,     0,
       0,    11,   199,    65,   229,    80,   238,   239,   419,   420,
      11,     0,    80,   210,    42,     0,   417,    23,    24,    42,
      22,    42,    11,    42,    41,   111,    11,   438,   253,     3,
     144,    41,    37,   132,    42,    64,    65,     0,    22,     2,
       3,   156,   157,   111,    42,    37,    37,   153,    11,    23,
      24,   169,   145,   153,    37,    65,   149,   194,   157,   153,
      37,    65,   199,   200,    65,    65,   184,    16,   186,    79,
       0,    42,    37,   210,   153,    79,    65,    42,    79,    79,
      65,    42,   153,    13,    38,    42,    42,   153,    42,    37,
      79,   209,   153,    37,    79,    37,    37,   153,   146,   153,
     110,   149,    65,   153,    42,    37,    42,   153,   118,   110,
     316,   317,   198,   153,   144,   153,    79,   147,   153,   316,
     317,   110,   153,   112,   113,   110,   153,   156,    37,   157,
     198,   153,    62,   156,   118,    65,   157,   205,   157,   190,
     191,   192,   193,   544,   206,   147,   154,   110,   154,    79,
     155,    42,   169,    16,   229,    42,   154,   141,   121,   169,
     144,   229,   154,   155,   155,   292,   293,   184,   169,   186,
       3,   154,   155,    42,   184,   407,   186,   194,   155,   316,
     317,   318,   199,   154,   194,   253,   411,   314,   306,   199,
     155,    42,   209,   154,   204,   153,   206,   154,   154,   209,
     204,    42,   206,   204,   204,   206,   206,   155,   153,   327,
     328,   155,   154,   155,   155,   204,   154,   206,   154,   204,
      42,   206,   154,   155,    42,   188,   189,   190,   191,   192,
     193,   642,   283,   284,   285,   286,   287,   153,   289,   169,
     291,   204,    42,   206,   245,   154,   155,   662,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      42,    42,    42,   154,   315,    33,    42,   154,    42,   114,
     115,   153,   117,    60,    61,   153,   206,    23,    24,   289,
      64,    65,    37,   153,    33,   154,   153,   698,   153,   306,
      31,    32,    33,    34,    35,    36,   306,   153,   153,   316,
     317,     3,     3,   154,   314,   132,   316,   317,   318,   157,
     327,   328,   154,   154,   157,   245,   153,   327,   328,   153,
     283,   284,   285,   286,   287,   288,   289,   290,   291,   153,
     153,   528,   154,   153,   153,   132,   154,   153,   153,   153,
     303,   153,   153,   411,    33,    34,    35,    36,   153,   417,
     195,   153,   315,    33,   154,   153,   366,   153,   359,   153,
     128,   129,   153,   153,   153,   153,   211,    33,   153,   140,
     153,   139,   154,   154,   154,   143,   386,   155,   154,     3,
     154,   149,   386,   153,   153,   386,   386,   153,   153,   153,
     153,   528,   153,   153,   153,   153,   153,   386,    42,   153,
     155,   386,   153,   153,   147,   153,   153,    42,   153,   419,
     420,   153,   618,   153,   153,   153,    42,    42,   419,   420,
     157,   618,   154,   386,    90,   435,   634,   154,   438,   154,
     419,   420,   154,   154,   419,   420,   154,   438,    42,     1,
     154,   154,   140,    42,     6,     7,   652,    38,   156,   438,
     155,   154,    42,   438,    42,   652,   419,   420,   154,   154,
     154,   154,   128,   129,   154,    42,   593,   154,   154,    36,
     153,    33,   523,   139,     3,   438,   544,   143,   596,   145,
     154,   618,   148,   149,   154,   154,   100,   153,    29,    30,
      31,    32,    33,    34,    35,    36,   153,   153,    60,    61,
      62,    63,    64,    65,    66,    67,    68,    69,    38,   156,
      16,   528,   157,   154,   157,   652,   154,    42,   528,    42,
     639,   572,   573,   154,    38,   154,   156,    89,   155,   528,
      92,    93,    94,    95,    96,    97,    98,    99,   156,   101,
     102,   225,   155,   210,    41,    80,    79,   109,   110,   111,
     112,   621,   113,   636,   634,   435,   428,    -1,    -1,    -1,
     523,    -1,    -1,    -1,    -1,    -1,   128,   129,    -1,    -1,
      -1,    -1,    -1,   135,    -1,   137,   138,   139,   140,   596,
     142,   143,   144,   145,   146,   147,   596,   149,    -1,    -1,
     152,   153,    -1,   155,   156,   157,    -1,    -1,    -1,    -1,
      -1,   618,    -1,    -1,   101,   102,   103,    -1,   618,   572,
     573,    -1,    -1,    -1,    -1,    -1,   684,   114,   115,   116,
     117,   118,   119,   120,   634,   122,   123,   124,   125,   639,
     634,    -1,   642,   634,   634,   652,    -1,    -1,   639,    -1,
      -1,   642,   652,    -1,    -1,   634,    -1,    -1,    -1,   634,
     639,    -1,    -1,   642,   639,    -1,    -1,   642,    -1,    -1,
     670,    -1,    -1,    -1,    -1,    -1,   670,    -1,    -1,   670,
     670,   634,   169,   170,    -1,    -1,   639,    -1,    -1,   642,
      -1,   670,    -1,    -1,    -1,   670,   183,    -1,   698,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   698,   195,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   670,    -1,   698,
      -1,    -1,    -1,   698,   211,   212,   213,   214,    -1,    -1,
     217,   218,   219,   220,   221,   222,   223,   224,   225,    -1,
      -1,    -1,    -1,    -1,    -1,   698,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,   245,    15,
      -1,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    -1,    33,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   280,   281,   282,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   292,   293,   294,   295,   296,
     297,   298,   299,   300,   301,   302,    -1,   304,   305,    -1,
     307,   308,   309,   310,   311,   312,   313,   314,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    90,
      -1,    -1,   329,   330,   331,   332,   333,   334,   335,   336,
     337,   338,   339,   340,   341,   342,   343,   344,   345,   346,
     347,   348,   349,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   359,     6,     7,    -1,    -1,   128,   129,   366,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   139,    -1,
      -1,    -1,   143,   144,   145,    -1,    -1,   148,   149,    32,
      33,   157,   153,    -1,    37,    -1,    39,    40,    -1,    -1,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    -1,    -1,    -1,   414,   415,   416,
      -1,    -1,    -1,    -1,    -1,   422,   423,   424,    -1,    -1,
     427,   428,   429,    -1,    -1,   432,    -1,    -1,   435,    -1,
      -1,    -1,    -1,   440,    -1,    -1,    -1,    90,    91,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     103,   104,   105,   106,   107,   108,    -1,    -1,    -1,    -1,
     113,   114,   115,   116,   117,    -1,   119,   120,   121,   122,
     123,   124,   125,   126,   127,   128,   129,   130,   131,    -1,
     487,    -1,   489,   136,    -1,    -1,   139,   140,    -1,    -1,
     143,   144,   145,    -1,    -1,   148,   149,    -1,    -1,    -1,
     153,    -1,   155,   156,    -1,    -1,    -1,    -1,   515,   516,
      -1,    -1,    -1,    -1,   521,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
     537,    -1,    -1,   540,    15,    16,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,
      -1,    -1,   569,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    -1,   585,    -1,
      -1,    -1,    -1,    -1,   591,   592,   593,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   603,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   612,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,   630,   631,    15,    -1,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,   654,   655,    -1,
      -1,   658,    -1,    -1,    -1,    -1,    -1,    -1,   665,   666,
       0,     1,    -1,    -1,    -1,    -1,     6,     7,    -1,    -1,
      -1,    -1,    -1,    -1,   681,   156,    -1,   684,   685,   686,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    33,    -1,    -1,    -1,   704,   705,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   719,   720,   721,   722,    -1,    -1,    -1,    -1,
      60,    61,    62,    63,   731,   732,    66,    67,    68,    69,
      70,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    89,
      -1,    -1,    92,    93,    94,    95,    96,    97,    98,    99,
       1,   101,   102,    -1,   154,     6,     7,    -1,    -1,   109,
     110,   111,   112,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   128,   129,
      -1,    -1,    33,    -1,    -1,   135,    -1,   137,   138,   139,
     140,    -1,   142,   143,   144,   145,   146,   147,    -1,   149,
     150,   151,   152,   153,    -1,   155,    -1,   157,    -1,    60,
      61,    62,    63,    -1,    -1,    66,    67,    68,    69,    70,
      71,    72,    73,    74,    75,    76,    77,    78,    79,    80,
      81,    82,    83,    84,    85,    86,    87,    88,    89,    -1,
      -1,    92,    93,    94,    95,    96,    97,    98,    99,     1,
     101,   102,    -1,    -1,     6,     7,    -1,    -1,   109,   110,
     111,   112,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   128,   129,    -1,
      -1,    33,    -1,    -1,   135,    -1,   137,   138,   139,   140,
      -1,   142,   143,   144,   145,   146,   147,    -1,   149,   150,
     151,   152,   153,    -1,   155,   156,   157,    -1,    60,    61,
      62,    63,    -1,    -1,    66,    67,    68,    69,    70,    71,
      72,    73,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,    88,    89,    -1,    -1,
      92,    93,    94,    95,    96,    97,    98,    99,     1,   101,
     102,    -1,    -1,     6,     7,    -1,    -1,   109,   110,   111,
     112,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   128,   129,    -1,    -1,
      33,    -1,    -1,   135,    -1,   137,   138,   139,   140,    -1,
     142,   143,   144,   145,   146,   147,    -1,   149,   150,   151,
     152,   153,    -1,   155,   156,   157,    -1,    60,    61,    62,
      63,    -1,    -1,    66,    67,    68,    69,    70,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    89,    -1,    -1,    92,
      93,    94,    95,    96,    97,    98,    99,     1,   101,   102,
      -1,    -1,     6,     7,    -1,    -1,   109,   110,   111,   112,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   128,   129,    -1,    -1,    33,
      -1,    -1,   135,    -1,   137,   138,   139,   140,    -1,   142,
     143,   144,   145,   146,   147,    -1,   149,   150,   151,   152,
     153,    -1,   155,    -1,   157,    -1,    60,    61,    62,    63,
      -1,    -1,    66,    67,    68,    69,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    89,    -1,    -1,    92,    93,
      94,    95,    96,    97,    98,    99,    -1,   101,   102,    -1,
      -1,    -1,    -1,    -1,    -1,   109,   110,   111,   112,    -1,
      -1,    -1,     1,    -1,    -1,    -1,    -1,     6,     7,    -1,
      -1,    -1,    -1,    -1,   128,   129,    -1,    -1,   132,    -1,
      -1,   135,    -1,   137,   138,   139,   140,    -1,   142,   143,
     144,   145,   146,   147,    33,   149,    -1,    -1,   152,   153,
      -1,   155,   156,   157,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      -1,    60,    61,    62,    63,    64,    65,    66,    67,    68,
      69,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    -1,
      89,    -1,    -1,    92,    93,    94,    95,    96,    97,    98,
      99,     1,   101,   102,    -1,    -1,     6,     7,    -1,    -1,
     109,   110,   111,   112,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   128,
     129,    -1,    -1,    33,    -1,    -1,   135,    -1,   137,   138,
     139,   140,    -1,   142,   143,   144,   145,   146,   147,    -1,
     149,    -1,    -1,   152,   153,    -1,   155,    -1,   157,    -1,
      60,    61,    62,    63,    -1,    -1,    66,    67,    68,    69,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,
      -1,    -1,    92,    93,    94,    95,    96,    97,    98,    99,
       1,   101,   102,    -1,    -1,     6,     7,    -1,    -1,   109,
     110,   111,   112,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   128,   129,
      -1,    -1,    33,    -1,    -1,   135,    -1,   137,   138,   139,
     140,    -1,   142,   143,   144,   145,   146,   147,    -1,   149,
      -1,    -1,   152,   153,    -1,   155,   156,   157,    -1,    60,
      61,    62,    63,    -1,    -1,    66,    67,    68,    69,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    89,    -1,
      -1,    92,    93,    94,    95,    96,    97,    98,    99,    -1,
     101,   102,    -1,    -1,    -1,    -1,    -1,    -1,   109,   110,
     111,   112,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   128,   129,    -1,
      -1,     6,     7,    -1,   135,    -1,   137,   138,   139,   140,
      -1,   142,   143,   144,   145,   146,   147,    22,   149,    -1,
      -1,   152,   153,    -1,   155,    -1,   157,    32,    33,    -1,
      -1,    -1,    37,    -1,    39,    40,    -1,    -1,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    71,    72,    73,    74,
      75,    76,    77,    78,    79,    80,    81,    82,    83,    84,
      85,    86,    87,    88,    -1,    90,    91,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   103,   104,
     105,   106,   107,   108,    -1,    -1,    -1,    -1,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   128,   129,   130,   131,     6,     7,    -1,
      -1,   136,    -1,    -1,   139,   140,   141,    -1,   143,   144,
     145,    -1,    -1,   148,   149,   150,    -1,    -1,   153,   154,
     155,    -1,    -1,    32,    33,    -1,    -1,    -1,    37,    -1,
      39,    40,    -1,    -1,    43,    44,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      -1,    90,    91,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   103,   104,   105,   106,   107,   108,
      -1,    -1,    -1,    -1,   113,   114,   115,   116,   117,    -1,
     119,   120,   121,   122,   123,   124,   125,   126,   127,   128,
     129,   130,   131,     6,     7,    -1,    -1,   136,    -1,    -1,
     139,   140,    -1,    -1,   143,   144,   145,    -1,    -1,   148,
     149,   150,    -1,    -1,   153,    -1,   155,    -1,    -1,    32,
      33,    -1,    -1,    -1,    37,    -1,    39,    40,    -1,    -1,
      43,    44,    45,    46,    47,    48,    49,    50,    51,    52,
      53,    54,    55,    56,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    71,    72,
      73,    74,    75,    76,    77,    78,    79,    80,    81,    82,
      83,    84,    85,    86,    87,    88,    -1,    90,    91,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     103,   104,   105,   106,   107,   108,    -1,    -1,    -1,    -1,
     113,   114,   115,   116,   117,    -1,   119,   120,   121,   122,
     123,   124,   125,   126,   127,   128,   129,   130,   131,    -1,
      -1,    -1,    -1,   136,    -1,    -1,   139,   140,     6,     7,
     143,   144,   145,    -1,    -1,   148,   149,   150,    -1,    -1,
     153,    -1,   155,    -1,    22,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    32,    33,    -1,    -1,    -1,    37,
      -1,    39,    40,    -1,    -1,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    15,
      16,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    90,    91,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   103,   104,   105,   106,   107,
     108,    -1,    -1,    -1,    -1,   113,   114,   115,   116,   117,
     118,   119,   120,   121,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,    -1,    -1,     6,     7,   136,    -1,
      -1,   139,   140,   141,    -1,   143,   144,   145,    -1,    -1,
     148,   149,    22,    -1,    -1,   153,    -1,   155,    -1,    -1,
      -1,    -1,    32,    33,    -1,    -1,    -1,    37,    -1,    39,
      40,    -1,    -1,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
     156,    -1,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      90,    91,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   103,   104,   105,   106,   107,   108,    -1,
      -1,    -1,    -1,   113,   114,   115,   116,   117,    -1,   119,
     120,   121,   122,   123,   124,   125,   126,   127,   128,   129,
     130,   131,     6,     7,    -1,    -1,   136,    -1,    -1,   139,
     140,    -1,    -1,   143,   144,   145,    -1,   147,   148,   149,
      -1,   150,    -1,   153,    -1,   155,    -1,    -1,    32,    33,
      -1,    -1,    -1,    37,    -1,    39,    40,    -1,    -1,    43,
      44,    45,    46,    47,    48,    49,    50,    51,    52,    53,
      54,    55,    56,    15,    -1,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    90,    91,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   103,
     104,   105,   106,   107,   108,    -1,    -1,    -1,    -1,   113,
     114,   115,   116,   117,    -1,   119,   120,   121,   122,   123,
     124,   125,   126,   127,   128,   129,   130,   131,     6,     7,
      -1,    -1,   136,    -1,    -1,   139,   140,    -1,    -1,   143,
     144,   145,    -1,    -1,   148,   149,    -1,    -1,    -1,   153,
     154,   155,    -1,    -1,    32,    33,    -1,    -1,    -1,    37,
      -1,    39,    40,    -1,    -1,    43,    44,    45,    46,    47,
      48,    49,    50,    51,    52,    53,    54,    55,    56,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   156,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    90,    91,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   103,   104,   105,   106,   107,
     108,    -1,    -1,    -1,    -1,   113,   114,   115,   116,   117,
      -1,   119,   120,   121,   122,   123,   124,   125,   126,   127,
     128,   129,   130,   131,     6,     7,    -1,    -1,   136,    -1,
      -1,   139,   140,    -1,    -1,   143,   144,   145,    -1,    -1,
     148,   149,    -1,    -1,    -1,   153,   154,   155,    -1,    -1,
      32,    33,    -1,    -1,    -1,    37,    38,    39,    40,    -1,
      -1,    43,    44,    45,    46,    47,    48,    49,    50,    51,
      52,    53,    54,    55,    56,    15,    -1,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    90,    91,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   103,   104,   105,   106,   107,   108,    -1,    -1,    -1,
      -1,   113,   114,   115,   116,   117,    -1,   119,   120,   121,
     122,   123,   124,   125,   126,   127,   128,   129,   130,   131,
       6,     7,    -1,    -1,   136,    -1,    -1,   139,   140,    -1,
      -1,   143,   144,   145,    -1,    -1,   148,   149,    -1,    -1,
      -1,   153,    -1,   155,    -1,    -1,    32,    33,    -1,    -1,
      -1,    37,    -1,    39,    40,    -1,    -1,    43,    44,    45,
      46,    47,    48,    49,    50,    51,    52,    53,    54,    55,
      56,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,   156,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    90,    91,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,   103,   104,   105,
     106,   107,   108,    -1,    -1,    -1,    -1,   113,   114,   115,
     116,   117,    -1,   119,   120,   121,   122,   123,   124,   125,
     126,   127,   128,   129,   130,   131,     6,     7,    -1,    -1,
     136,    -1,    -1,   139,   140,    -1,    -1,   143,   144,   145,
      -1,    -1,   148,   149,    -1,    -1,    -1,   153,    -1,   155,
      -1,    -1,    32,    33,    -1,    -1,    -1,    37,    -1,    39,
      40,    -1,    -1,    43,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    -1,    -1,    -1,
      -1,    -1,    15,    -1,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    -1,    -1,    -1,    -1,    -1,    -1,
      90,    91,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   103,   104,   105,   106,   107,   108,    -1,
       6,     7,    -1,   113,   114,   115,   116,   117,    -1,   119,
     120,   121,   122,   123,   124,   125,   126,   127,   128,   129,
     130,   131,    -1,    -1,    -1,    -1,   136,    33,    -1,   139,
     140,    -1,    -1,   143,   144,   145,    -1,    -1,   148,   149,
      -1,    -1,    -1,   153,    -1,   155,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    62,    -1,    -1,    -1,
      66,    67,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    -1,    89,    -1,    -1,    92,    93,    94,    95,
      96,    97,    98,   156,    -1,   101,   102,    -1,    -1,    -1,
      -1,    -1,    -1,   109,   110,   111,   112,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,   128,   129,    -1,    -1,    -1,    -1,    -1,   135,
      -1,   137,   138,   139,   140,    -1,   142,   143,   144,   145,
      -1,   147,    -1,   149,    -1,    -1,   152,   153,    15,    -1,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      15,    -1,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    15,    -1,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    15,    -1,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    15,    -1,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    15,    -1,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   156,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    15,   154,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      15,   154,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    15,   154,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    15,   154,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    15,   154,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    15,    -1,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      -1,    -1,    -1,    -1,    -1,    -1,    15,   154,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    15,   154,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      15,   154,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    15,   154,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    15,   154,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    15,   154,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    -1,    -1,
      -1,    -1,    -1,    -1,    15,   154,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    15,   154,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    15,   154,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      15,   154,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    15,   154,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    15,   154,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,
      -1,    -1,    15,   154,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    15,   154,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    15,   154,    17,    18,
      19,    20,    21,    22,    23,    24,    25,    26,    27,    28,
      29,    30,    31,    32,    33,    34,    35,    36,    15,   154,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
      15,   154,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    15,   154,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    -1,    -1,    -1,    -1,    -1,    -1,
      15,   154,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    15,   154,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      33,    34,    35,    36,    15,   154,    17,    18,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    33,    34,    35,    36,    -1,   154,    -1,    -1,
      -1,    -1,    -1,    -1,    33,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   154,
      -1,    -1,    -1,    -1,    -1,    -1,    55,    56,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,   154,    71,    72,    73,    74,    75,    76,    77,    78,
      79,    80,    81,    82,    83,    84,    85,    86,    87,    88,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,   154,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    33,    -1,    -1,
      -1,   154,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    55,
      56,   140,    -1,   154,   143,   144,   145,    -1,    -1,    -1,
     149,   150,    -1,    -1,   153,    71,    72,    73,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,    88,    71,    72,    73,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
      88,    71,    72,    73,    74,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,    88,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,   140,    -1,    -1,   143,   144,   145,
      -1,    -1,    -1,   149,   150,    -1,    -1,   153,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,   143,   144,   145,    -1,    -1,
      -1,   149,   150,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,   143,   144,   145,    -1,    -1,    -1,   149,
     150,    15,    -1,    17,    18,    19,    20,    21,    22,    23,
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
      32,    33,    34,    35,    36,    -1,    -1,    -1,    -1,    15,
      42,    17,    18,    19,    20,    21,    22,    23,    24,    25,
      26,    27,    28,    29,    30,    31,    32,    33,    34,    35,
      36,    -1,    -1,    -1,    -1,    -1,    42,    15,    16,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    -1,
      38,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    15,    38,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    15,    38,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    15,    38,    17,
      18,    19,    20,    21,    22,    23,    24,    25,    26,    27,
      28,    29,    30,    31,    32,    33,    34,    35,    36,    -1,
      38,    15,    16,    17,    18,    19,    20,    21,    22,    23,
      24,    25,    26,    27,    28,    29,    30,    31,    32,    33,
      34,    35,    36,    15,    16,    17,    18,    19,    20,    21,
      22,    23,    24,    25,    26,    27,    28,    29,    30,    31,
      32,    33,    34,    35,    36,    15,    -1,    17,    18,    19,
      20,    21,    22,    23,    24,    25,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     1,     6,     7,    33,    60,    61,    62,    63,    66,
      67,    68,    69,    70,    71,    72,    73,    74,    75,    76,
      77,    78,    79,    80,    81,    82,    83,    84,    85,    86,
      87,    88,    89,    92,    93,    94,    95,    96,    97,    98,
      99,   101,   102,   109,   110,   111,   112,   128,   129,   135,
     137,   138,   139,   140,   142,   143,   144,   145,   146,   147,
     149,   150,   151,   152,   153,   155,   159,   160,   161,   163,
     167,   170,   171,   172,   173,   174,   176,   179,   181,   182,
     183,   184,   185,   186,   187,   188,   189,   191,   192,   198,
     200,   205,   206,   207,   208,   132,   157,   149,   163,   163,
     159,   153,   153,   153,   146,   149,     1,   174,   199,   208,
     153,   183,   153,   153,   153,   153,   153,   153,   153,   153,
     153,    22,    32,    37,    39,    40,    43,    44,    45,    46,
      47,    48,    49,    50,    51,    52,    53,    54,    55,    56,
      90,    91,   103,   104,   105,   106,   107,   108,   113,   114,
     115,   116,   117,   118,   119,   120,   121,   122,   123,   124,
     125,   126,   127,   130,   131,   136,   140,   141,   148,   153,
     155,   159,   161,   163,   164,   168,   170,   172,   173,   209,
     210,   218,   220,   153,   153,   153,   153,   153,   153,   153,
     153,   153,   153,   153,   153,   153,    16,    16,   183,   153,
      33,     1,   156,   174,   175,   183,   197,   198,   208,     3,
       3,     3,     3,     4,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    14,    37,   155,   132,     0,   181,    33,
      55,    56,   140,   143,   144,   145,   149,   153,   177,   178,
     184,   193,   194,   195,   157,   153,   159,   164,   164,   164,
      60,    61,   207,    33,   177,   178,   195,   149,   161,   190,
     190,   164,   214,   214,   164,   213,   214,   159,   164,   215,
     216,   213,   164,   163,   164,   164,   212,   164,   164,   153,
     153,   153,   153,   153,   153,   153,   153,   153,   153,   153,
     153,   153,   153,   153,   153,   153,   153,   153,   153,   153,
     153,   153,   153,   153,   153,   153,   153,   153,   153,   153,
     153,   153,   153,   153,   153,   153,   153,   153,    33,   154,
     164,   168,   172,   183,   214,   164,   211,    23,    24,    15,
      17,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    33,    34,    35,    36,
     154,   164,   159,   168,   154,   168,   140,   163,   163,   153,
     163,   220,   220,   220,   217,   220,    22,   159,   160,   164,
     165,   166,   169,   170,   171,   154,   214,   195,   166,   160,
     156,   132,   156,   174,   208,   156,   175,   198,   168,    22,
     169,   214,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   162,   164,   162,   164,   132,   177,   193,   153,
     153,    33,   176,   176,    37,   155,     3,    42,   157,   154,
     154,   154,   153,   153,   157,   149,   157,    42,    42,    42,
     154,   154,    42,   154,   154,    42,   154,   154,   154,    38,
      42,   154,   164,   164,   164,   220,   220,   220,   220,   220,
     163,   159,   163,   220,   163,   220,   213,   213,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   163,   164,   164,
     168,   164,   164,   164,   164,   164,   164,   164,   159,   213,
     220,   166,   166,   160,   154,   154,   154,   154,   154,    42,
     156,   168,   168,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   164,   164,   164,   164,   164,   164,   164,   164,
     164,   164,   164,   164,   154,    42,    42,   154,   154,   154,
     154,    42,   154,    42,   154,   140,   159,   164,    42,   154,
     154,   157,   154,   154,   156,   140,    38,    16,    38,   156,
      16,   156,   176,   180,   183,   196,   180,   193,    38,   164,
     156,   164,   164,   194,   199,   199,   155,   164,   164,   164,
     164,   219,   219,   164,   164,   215,   199,   164,   154,    42,
     154,   154,    42,    42,   154,   154,   154,   154,   154,   154,
     154,   154,   154,   154,   154,    42,   154,   154,   154,   154,
     154,    42,    42,    42,   154,   154,    42,   154,   154,   154,
     154,   154,   154,    42,   154,   154,   154,   154,   154,   154,
     164,   164,    16,   164,   164,   164,   220,   165,   153,   164,
     164,    42,   154,   194,   154,   154,    37,   155,    38,   156,
       3,    64,    65,   201,   202,   203,   204,   154,   154,   157,
     154,   154,   100,   164,   220,   220,   164,   164,   164,   213,
     168,   164,   153,   164,    42,    42,   154,   166,    16,    38,
     156,   196,   153,    38,   156,    37,   155,   164,   164,    16,
     175,   203,   156,   201,   157,   157,   207,   199,   154,   154,
     154,    42,   154,   154,    42,    42,    42,   166,   164,   164,
     154,   164,   180,    37,   155,   164,   164,    16,   154,   164,
     184,   164,   164,   154,    42,    42,   154,    38,   156,    38,
     156,   199,   154,   154,   154,   154,   164,   164,   155,    37,
     155,    42,    42,   156,   164,   164,   164,   164,    38,   156,
     154,    42,   155,   164,   164,   154,   156
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
      yyerror (qsp, YY_("syntax error: cannot back up")); \
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
# define YYLEX yylex (&yylval, qsp)
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
		  Type, Value, qsp); \
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
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, Query_Stack *qsp)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep, qsp)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
    Query_Stack *qsp;
#endif
{
  if (!yyvaluep)
    return;
  YYUSE (qsp);
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
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, Query_Stack *qsp)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep, qsp)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
    Query_Stack *qsp;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep, qsp);
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
yy_reduce_print (YYSTYPE *yyvsp, int yyrule, Query_Stack *qsp)
#else
static void
yy_reduce_print (yyvsp, yyrule, qsp)
    YYSTYPE *yyvsp;
    int yyrule;
    Query_Stack *qsp;
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
		       		       , qsp);
      fprintf (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, Rule, qsp); \
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
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, Query_Stack *qsp)
#else
static void
yydestruct (yymsg, yytype, yyvaluep, qsp)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
    Query_Stack *qsp;
#endif
{
  YYUSE (yyvaluep);
  YYUSE (qsp);

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
int yyparse (Query_Stack *qsp);
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
yyparse (Query_Stack *qsp)
#else
int
yyparse (qsp)
    Query_Stack *qsp;
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
#line 406 "vectree.y"
    {
			(yyval.enp)=NODE0(T_POINTER);
			SET_VN_STRING((yyval.enp), savestr(ID_NAME((yyvsp[(1) - (1)].idp))));
			}
    break;

  case 3:
#line 413 "vectree.y"
    {
			(yyval.enp) = NODE0(T_FUNCPTR);
			SET_VN_STRING((yyval.enp), savestr(ID_NAME((yyvsp[(1) - (1)].idp))));
			}
    break;

  case 4:
#line 420 "vectree.y"
    {
			(yyval.enp)=NODE0(T_STR_PTR);
			SET_VN_STRING((yyval.enp), savestr(ID_NAME((yyvsp[(1) - (1)].idp))));
			}
    break;

  case 5:
#line 427 "vectree.y"
    {
			(yyval.enp)=NODE3(T_RANGE,(yyvsp[(1) - (5)].enp),(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp));
			}
    break;

  case 6:
#line 433 "vectree.y"
    {
			if( OBJ_FLAGS((yyvsp[(1) - (1)].dp)) & DT_STATIC ){
				(yyval.enp)=NODE0(T_STATIC_OBJ);
				SET_VN_OBJ((yyval.enp), (yyvsp[(1) - (1)].dp));
				// To be safe, we need to mark
				// the object so that it can't be
				// deleted while this reference
				// exists...  We don't want to 
				// have a dangling pointer!?
			} else {
				const char *s;
				(yyval.enp)=NODE0(T_DYN_OBJ);
				s=savestr(OBJ_NAME((yyvsp[(1) - (1)].dp)));
				SET_VN_STRING((yyval.enp),s);
			}
			}
    break;

  case 7:
#line 450 "vectree.y"
    {
			(yyval.enp) = NODE1(T_DEREFERENCE,(yyvsp[(2) - (2)].enp));
			}
    break;

  case 8:
#line 454 "vectree.y"
    {
			(yyval.enp)=NODE1(T_OBJ_LOOKUP,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 9:
#line 458 "vectree.y"
    {
			Undef_Sym *usp;

			usp=undef_of(QSP_ARG  (yyvsp[(1) - (1)].e_string));
			if( usp == NO_UNDEF ){
				/* BUG?  are contexts handled correctly??? */
				sprintf(YY_ERR_STR,"Undefined symbol %s",(yyvsp[(1) - (1)].e_string));
				yyerror(qsp,  YY_ERR_STR);
				/*usp=*/new_undef(QSP_ARG  (yyvsp[(1) - (1)].e_string));
			}
			(yyval.enp)=NODE0(T_UNDEF);
			SET_VN_STRING((yyval.enp), savestr((yyvsp[(1) - (1)].e_string)));
			CURDLE((yyval.enp))
			}
    break;

  case 10:
#line 472 "vectree.y"
    {
			(yyval.enp)=NODE1(T_REAL_PART,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 11:
#line 475 "vectree.y"
    {
			(yyval.enp)=NODE1(T_IMAG_PART,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 12:
#line 478 "vectree.y"
    {
			(yyval.enp)=NODE2(T_SQUARE_SUBSCR,(yyvsp[(1) - (4)].enp),(yyvsp[(3) - (4)].enp));
			}
    break;

  case 13:
#line 481 "vectree.y"
    {
			(yyval.enp)=NODE2(T_CURLY_SUBSCR,(yyvsp[(1) - (4)].enp),(yyvsp[(3) - (4)].enp));
			}
    break;

  case 14:
#line 485 "vectree.y"
    {
			(yyval.enp)=NODE3(T_SUBVEC,(yyvsp[(1) - (6)].enp),(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			}
    break;

  case 15:
#line 489 "vectree.y"
    {
			/* Why not use T_RANGE2 here?  The current version
			 * is fine as-is, but don't get rid of T_RANGE2 because
			 * mlab.y uses it...
			 */
			(yyval.enp)=NODE3(T_CSUBVEC,(yyvsp[(1) - (6)].enp),(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			}
    break;

  case 16:
#line 497 "vectree.y"
    {
			(yyval.enp)=NODE2(T_SUBSAMP,(yyvsp[(1) - (4)].enp),(yyvsp[(3) - (4)].enp));
			}
    break;

  case 17:
#line 501 "vectree.y"
    {
			(yyval.enp)=NODE2(T_CSUBSAMP,(yyvsp[(1) - (4)].enp),(yyvsp[(3) - (4)].enp));
			}
    break;

  case 18:
#line 508 "vectree.y"
    {
			(yyval.enp)=NODE1(T_FIX_SIZE,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 20:
#line 520 "vectree.y"
    {
			(yyval.enp) = NODE1(T_TYPECAST,(yyvsp[(4) - (4)].enp));
			SET_VN_CAST_PREC_PTR((yyval.enp),(yyvsp[(2) - (4)].prec_p));
			}
    break;

  case 21:
#line 524 "vectree.y"
    {
			(yyval.enp) = (yyvsp[(2) - (3)].enp); }
    break;

  case 22:
#line 526 "vectree.y"
    {
			(yyval.enp)=NODE2(T_PLUS,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 23:
#line 528 "vectree.y"
    {
			(yyval.enp)=NODE2(T_MINUS,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 24:
#line 530 "vectree.y"
    {
			(yyval.enp)=NODE2(T_TIMES,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 25:
#line 532 "vectree.y"
    {
			(yyval.enp)=NODE2(T_DIVIDE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 26:
#line 534 "vectree.y"
    {
			(yyval.enp)=NODE2(T_MODULO,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 27:
#line 536 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BITAND,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 28:
#line 538 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BITOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 29:
#line 540 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BITXOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 30:
#line 542 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BITLSHIFT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 31:
#line 544 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BITRSHIFT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 32:
#line 546 "vectree.y"
    {
			(yyval.enp)=NODE1(T_BITCOMP,(yyvsp[(2) - (2)].enp)); }
    break;

  case 33:
#line 548 "vectree.y"
    {
			(yyval.enp) = NODE0(T_LIT_INT);
			SET_VN_INTVAL((yyval.enp), (int) (yyvsp[(1) - (1)].dval));
			}
    break;

  case 34:
#line 552 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_EQ,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 35:
#line 555 "vectree.y"
    {
			(yyval.enp) = NODE2(T_BOOL_LT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 36:
#line 558 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_GT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 37:
#line 561 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_GE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 38:
#line 564 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_LE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 39:
#line 567 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_NE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 40:
#line 570 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_AND,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 41:
#line 573 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_OR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 42:
#line 576 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_XOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 43:
#line 579 "vectree.y"
    {
			(yyval.enp)=NODE1(T_BOOL_NOT,(yyvsp[(2) - (2)].enp));
			}
    break;

  case 44:
#line 582 "vectree.y"
    {
			Vec_Expr_Node *enp;
			enp=NODE2(T_BOOL_PTREQ,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			(yyval.enp)=NODE1(T_BOOL_NOT,enp);
			}
    break;

  case 45:
#line 597 "vectree.y"
    {
			(yyval.enp)=NODE2(T_BOOL_PTREQ,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 46:
#line 602 "vectree.y"
    {
			(yyval.enp)=NODE0(T_MATH0_FN);
			SET_VN_FUNC_PTR((yyval.enp),(yyvsp[(1) - (3)].func_p));
			}
    break;

  case 47:
#line 607 "vectree.y"
    {
			(yyval.enp)=NODE1(T_MATH1_FN,(yyvsp[(3) - (4)].enp));
			SET_VN_FUNC_PTR((yyval.enp),(yyvsp[(1) - (4)].func_p));
			}
    break;

  case 48:
#line 612 "vectree.y"
    {
			(yyval.enp)=NODE2(T_MATH2_FN,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			SET_VN_FUNC_PTR((yyval.enp),(yyvsp[(1) - (6)].func_p));
			}
    break;

  case 49:
#line 617 "vectree.y"
    {
			(yyval.enp)=NODE1(T_INT1_FN,(yyvsp[(3) - (4)].enp));
			SET_VN_FUNC_PTR((yyval.enp),(yyvsp[(1) - (4)].func_p));
			}
    break;

  case 50:
#line 622 "vectree.y"
    {
			(yyval.enp) = NODE2(T_INNER,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 51:
#line 625 "vectree.y"
    {
			(yyval.enp)=NODE0(T_LIT_DBL);
			SET_VN_DBLVAL((yyval.enp),(yyvsp[(1) - (1)].dval));
			}
    break;

  case 52:
#line 630 "vectree.y"
    {
			/* We determine exactly which type later */
			(yyval.enp) = NODE3(T_SS_S_CONDASS,(yyvsp[(1) - (5)].enp),(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp));
			}
    break;

  case 53:
#line 634 "vectree.y"
    {
			(yyval.enp) = NODE0(T_LIT_INT);
			SET_VN_INTVAL((yyval.enp), (int) (yyvsp[(1) - (1)].dval));
			}
    break;

  case 54:
#line 639 "vectree.y"
    {
			(yyval.enp)=NODE0(T_BADNAME);
			NODE_ERROR((yyval.enp));
			CURDLE((yyval.enp))
			WARN("illegal use of data function");
			}
    break;

  case 55:
#line 645 "vectree.y"
    {
			(yyval.enp)=NODE1(T_DATA_FN,(yyvsp[(3) - (4)].enp));
			SET_VN_FUNC_PTR((yyval.enp),(yyvsp[(1) - (4)].func_p));
			}
    break;

  case 56:
#line 650 "vectree.y"
    {
			(yyval.enp)=NODE1(T_SIZE_FN,(yyvsp[(3) - (4)].enp));
			SET_VN_FUNC_PTR((yyval.enp),(yyvsp[(1) - (4)].func_p));
			}
    break;

  case 57:
#line 654 "vectree.y"
    {
			(yyval.enp)=NODE1(T_SIZE_FN,(yyvsp[(3) - (4)].enp));
			SET_VN_FUNC_PTR((yyval.enp),(yyvsp[(1) - (4)].func_p));
			}
    break;

  case 58:
#line 658 "vectree.y"
    {
			(yyval.enp)=NODE1(T_SIZE_FN,(yyvsp[(3) - (4)].enp));
			SET_VN_FUNC_PTR((yyval.enp),(yyvsp[(1) - (4)].func_p));
			NODE_ERROR((yyval.enp));
			advise("dereference pointer before passing to size function");
			CURDLE((yyval.enp))
			}
    break;

  case 59:
#line 665 "vectree.y"
    {
			sprintf(YY_ERR_STR,"need to dereference pointer %s",VN_STRING((yyvsp[(3) - (4)].enp)));
			yyerror(THIS_QSP,  YY_ERR_STR);
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 60:
#line 671 "vectree.y"
    {
			(yyval.enp)=NODE1(T_SUM,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 61:
#line 676 "vectree.y"
    {
			(yyval.enp)=NODE1(T_FILE_EXISTS,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 62:
#line 680 "vectree.y"
    {
			(yyval.enp)=NODE1(T_STR1_FN,(yyvsp[(3) - (4)].enp));
			SET_VN_FUNC_PTR((yyval.enp),(yyvsp[(1) - (4)].func_p));
			}
    break;

  case 63:
#line 685 "vectree.y"
    {
			(yyval.enp)=NODE2(T_STR2_FN,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			SET_VN_FUNC_PTR((yyval.enp),(yyvsp[(1) - (6)].func_p));
			}
    break;

  case 64:
#line 691 "vectree.y"
    {
			(yyval.enp)=NODE2(T_STR2_FN,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			SET_VN_FUNC_PTR((yyval.enp),(yyvsp[(1) - (6)].func_p));
			}
    break;

  case 65:
#line 697 "vectree.y"
    {
			(yyval.enp)=NODE1(T_STRV_FN,(yyvsp[(3) - (4)].enp));
			SET_VN_FUNC_PTR((yyval.enp),(yyvsp[(1) - (4)].func_p));
			}
    break;

  case 66:
#line 704 "vectree.y"
    {
			(yyval.enp)=NODE1(T_CHAR_FN,(yyvsp[(3) - (4)].enp));
			SET_VN_FUNC_PTR((yyval.enp),(yyvsp[(1) - (4)].func_p));
			}
    break;

  case 67:
#line 717 "vectree.y"
    {
			(yyval.enp)=NODE1(T_CONJ,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 68:
#line 722 "vectree.y"
    {
				(yyval.enp)=NODE1(T_UMINUS,(yyvsp[(2) - (2)].enp));
				}
    break;

  case 69:
#line 726 "vectree.y"
    {
			(yyval.enp)=NODE1(T_MINVAL,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 70:
#line 730 "vectree.y"
    {
			(yyval.enp)=NODE1(T_MAXVAL,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 71:
#line 735 "vectree.y"
    { (yyval.enp)=NODE1(T_MAX_INDEX,(yyvsp[(3) - (4)].enp)); }
    break;

  case 72:
#line 737 "vectree.y"
    { (yyval.enp)=NODE1(T_MIN_INDEX,(yyvsp[(3) - (4)].enp)); }
    break;

  case 73:
#line 740 "vectree.y"
    {
			(yyval.enp) = NODE2(T_INDIR_CALL,(yyvsp[(3) - (7)].enp),(yyvsp[(6) - (7)].enp));
			}
    break;

  case 74:
#line 744 "vectree.y"
    {
			(yyval.enp)=NODE1(T_CALLFUNC,(yyvsp[(3) - (4)].enp));
			SET_VN_CALL_SUBRT((yyval.enp), (yyvsp[(1) - (4)].srp));
			/* make sure this is not a void subroutine! */
			if( SR_PREC_CODE((yyvsp[(1) - (4)].srp)) == PREC_VOID ){
				NODE_ERROR((yyval.enp));
				sprintf(YY_ERR_STR,"void subroutine %s used in expression!?",SR_NAME((yyvsp[(1) - (4)].srp)));
				advise(YY_ERR_STR);
				CURDLE((yyval.enp))
			}
			}
    break;

  case 75:
#line 755 "vectree.y"
    {
			(yyval.enp)=(yyvsp[(1) - (1)].enp);
			}
    break;

  case 76:
#line 758 "vectree.y"
    {
			(yyval.enp)=(yyvsp[(1) - (1)].enp);
			}
    break;

  case 77:
#line 761 "vectree.y"
    {
			WARN("warp not implemented");
			(yyval.enp)=(yyvsp[(3) - (6)].enp);
			}
    break;

  case 78:
#line 765 "vectree.y"
    {
			(yyval.enp)=NODE2(T_LOOKUP,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			}
    break;

  case 79:
#line 769 "vectree.y"
    {
			(yyval.enp) = NODE1(T_TRANSPOSE,(yyvsp[(3) - (4)].enp));
			SET_VN_SIZCH_SHAPE((yyval.enp), ALLOC_SHAPE );
			}
    break;

  case 80:
#line 773 "vectree.y"
    { (yyval.enp) = NODE1(T_DFT,(yyvsp[(3) - (4)].enp)); }
    break;

  case 81:
#line 774 "vectree.y"
    { (yyval.enp) = NODE1(T_IDFT,(yyvsp[(3) - (4)].enp)); }
    break;

  case 82:
#line 775 "vectree.y"
    {
			(yyval.enp) = NODE1(T_RDFT,(yyvsp[(3) - (4)].enp));
			SET_VN_SIZCH_SHAPE((yyval.enp), ALLOC_SHAPE );
			}
    break;

  case 83:
#line 779 "vectree.y"
    {
			(yyval.enp) = NODE1(T_RIDFT,(yyvsp[(3) - (4)].enp));
			SET_VN_SIZCH_SHAPE((yyval.enp), ALLOC_SHAPE );
			}
    break;

  case 85:
#line 784 "vectree.y"
    {
			(yyval.enp)=NODE1(T_WRAP,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 86:
#line 787 "vectree.y"
    {
			(yyval.enp)=NODE3(T_SCROLL,(yyvsp[(3) - (8)].enp),(yyvsp[(5) - (8)].enp),(yyvsp[(7) - (8)].enp));
			}
    break;

  case 87:
#line 792 "vectree.y"
    { (yyval.enp) = NODE1(T_ERODE,(yyvsp[(3) - (4)].enp)); }
    break;

  case 88:
#line 795 "vectree.y"
    { (yyval.enp) = NODE1(T_DILATE,(yyvsp[(3) - (4)].enp)); }
    break;

  case 89:
#line 797 "vectree.y"
    {
			(yyval.enp)=NODE1(T_ENLARGE,(yyvsp[(3) - (4)].enp));
			SET_VN_SIZCH_SHAPE((yyval.enp), ALLOC_SHAPE );
			}
    break;

  case 90:
#line 801 "vectree.y"
    {
			(yyval.enp)=NODE1(T_REDUCE,(yyvsp[(3) - (4)].enp));
			SET_VN_SIZCH_SHAPE((yyval.enp), ALLOC_SHAPE );
			}
    break;

  case 91:
#line 806 "vectree.y"
    { (yyval.enp)=NODE1(T_LOAD,(yyvsp[(3) - (4)].enp)); }
    break;

  case 92:
#line 807 "vectree.y"
    {
				(yyval.enp)=NODE3(T_RAMP,(yyvsp[(3) - (8)].enp),(yyvsp[(5) - (8)].enp),(yyvsp[(7) - (8)].enp));
				}
    break;

  case 93:
#line 811 "vectree.y"
    {
			(yyval.enp) = NODE3(T_MAX_TIMES,(yyvsp[(3) - (8)].enp),(yyvsp[(5) - (8)].enp),(yyvsp[(7) - (8)].enp));
			}
    break;

  case 95:
#line 819 "vectree.y"
    { (yyval.enp)=NODE1(T_REFERENCE,(yyvsp[(2) - (2)].enp)); }
    break;

  case 99:
#line 831 "vectree.y"
    {
			sprintf(YY_ERR_STR,"shouldn't try to reference pointer variable %s",VN_STRING((yyvsp[(2) - (2)].enp)));
			yyerror(THIS_QSP,  YY_ERR_STR);
			(yyval.enp)=(yyvsp[(2) - (2)].enp);
			}
    break;

  case 100:
#line 837 "vectree.y"
    {
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 102:
#line 844 "vectree.y"
    {
			(yyval.enp)=NODE2(T_ARGLIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 103:
#line 853 "vectree.y"
    {
			/* BUG check to see that this subrt is void! */
			(yyval.enp)=NODE1(T_CALLFUNC,(yyvsp[(3) - (4)].enp));
			SET_VN_CALL_SUBRT((yyval.enp), (yyvsp[(1) - (4)].srp));
			if( SR_PREC_CODE((yyvsp[(1) - (4)].srp)) != PREC_VOID ){
				NODE_ERROR((yyval.enp));
				sprintf(YY_ERR_STR,"return value of function %s is ignored",SR_NAME((yyvsp[(1) - (4)].srp)));
				advise(YY_ERR_STR);
			}
			}
    break;

  case 104:
#line 864 "vectree.y"
    {
			/* BUG check to see that the pointed to subrt is void -
			 * OR should we check that on pointer assignment?
			 */
			(yyval.enp) = NODE2(T_INDIR_CALL,(yyvsp[(3) - (7)].enp),(yyvsp[(6) - (7)].enp));
			}
    break;

  case 105:
#line 874 "vectree.y"
    { (yyval.enp) = NODE1(T_REFERENCE,(yyvsp[(2) - (2)].enp)); }
    break;

  case 108:
#line 878 "vectree.y"
    {
				(yyval.enp)=NODE2(T_EQUIVALENCE,(yyvsp[(3) - (8)].enp),(yyvsp[(5) - (8)].enp));
				SET_VN_DECL_PREC((yyval.enp), (yyvsp[(7) - (8)].prec_p));
			}
    break;

  case 109:
#line 883 "vectree.y"
    {
			(yyval.enp)=NODE1(T_CALLFUNC,(yyvsp[(3) - (4)].enp));
			SET_VN_CALL_SUBRT((yyval.enp), (yyvsp[(1) - (4)].srp));
			/* make sure this is not a void subroutine! */
			if( SR_PREC_CODE((yyvsp[(1) - (4)].srp)) == PREC_VOID ){
				NODE_ERROR((yyval.enp));
				sprintf(YY_ERR_STR,"void subroutine %s used in pointer expression!?",SR_NAME((yyvsp[(1) - (4)].srp)));
				advise(YY_ERR_STR);
				CURDLE((yyval.enp))
			}
			}
    break;

  case 110:
#line 897 "vectree.y"
    {
			(yyval.enp)=NODE0(T_FUNCREF);
			SET_VN_SUBRT((yyval.enp), (yyvsp[(2) - (2)].srp));
			}
    break;

  case 113:
#line 905 "vectree.y"
    {
			(yyval.enp)=NODE2(T_SET_PTR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 114:
#line 910 "vectree.y"
    {
			(yyval.enp) = NODE2(T_SET_FUNCPTR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 115:
#line 915 "vectree.y"
    {
			(yyval.enp)=NODE2(T_SET_STR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 116:
#line 927 "vectree.y"
    {
			(yyval.enp)=NODE2(T_ASSIGN,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 117:
#line 930 "vectree.y"
    { (yyval.enp)=NODE1(T_POSTINC,(yyvsp[(1) - (2)].enp)); }
    break;

  case 118:
#line 931 "vectree.y"
    { (yyval.enp)=NODE1(T_PREINC,(yyvsp[(2) - (2)].enp)); }
    break;

  case 119:
#line 932 "vectree.y"
    { (yyval.enp)=NODE1(T_PREDEC,(yyvsp[(2) - (2)].enp)); }
    break;

  case 120:
#line 933 "vectree.y"
    { (yyval.enp)=NODE1(T_POSTDEC,(yyvsp[(1) - (2)].enp)); }
    break;

  case 121:
#line 934 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_PLUS,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 122:
#line 940 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_TIMES,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 123:
#line 946 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_MINUS,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 124:
#line 952 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_DIVIDE,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 125:
#line 958 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITAND,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 126:
#line 964 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 127:
#line 970 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITXOR,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 128:
#line 976 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITLSHIFT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 129:
#line 982 "vectree.y"
    {
			Vec_Expr_Node *new_enp,*dup_enp;
			new_enp=NODE2(T_BITRSHIFT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			dup_enp=DUP_TREE((yyvsp[(1) - (3)].enp));
			(yyval.enp)=NODE2(T_ASSIGN,dup_enp,new_enp);
			}
    break;

  case 130:
#line 996 "vectree.y"
    { (yyval.enp) = (yyvsp[(1) - (2)].enp); }
    break;

  case 133:
#line 1000 "vectree.y"
    {
			Identifier *idp;
			(yyval.enp) = NODE0(T_LABEL);
			idp = new_id(QSP_ARG  (yyvsp[(1) - (2)].e_string));
			SET_ID_TYPE(idp, ID_LABEL);
			SET_VN_STRING((yyval.enp), savestr(ID_NAME(idp)));
			}
    break;

  case 134:
#line 1008 "vectree.y"
    {
			(yyval.enp) = NODE0(T_LABEL);
			SET_VN_STRING((yyval.enp), savestr(ID_NAME((yyvsp[(1) - (2)].idp))));
			}
    break;

  case 135:
#line 1013 "vectree.y"
    { (yyval.enp) = NO_VEXPR_NODE; }
    break;

  case 138:
#line 1021 "vectree.y"
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

  case 139:
#line 1032 "vectree.y"
    {
			(yyval.enp)=NODE2(T_STAT_LIST,(yyvsp[(1) - (2)].enp),(yyvsp[(2) - (2)].enp));
			}
    break;

  case 140:
#line 1045 "vectree.y"
    {
			(yyval.enp)=(yyvsp[(2) - (3)].enp);
			}
    break;

  case 141:
#line 1050 "vectree.y"
    {
			(yyval.enp)=(yyvsp[(2) - (3)].enp);
			}
    break;

  case 142:
#line 1054 "vectree.y"
    {
			(yyval.enp)=NODE2(T_STAT_LIST,(yyvsp[(2) - (4)].enp),(yyvsp[(3) - (4)].enp));
			}
    break;

  case 143:
#line 1058 "vectree.y"
    {
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 144:
#line 1062 "vectree.y"
    {
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 145:
#line 1066 "vectree.y"
    {
			yyerror(THIS_QSP,  (char *)"missing '}'");
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 146:
#line 1073 "vectree.y"
    {
			set_subrt_ctx(QSP_ARG  (yyvsp[(1) - (4)].e_string));		/* when do we unset??? */
			/* We evaluate the declarations here so we can parse the body, but
			 * the declarations get interpreted a second time when we compile the nodes -
			 * at least, for prototype declarations!?  Not a problem for regular declarations?
			 */
			if( (yyvsp[(3) - (4)].enp) != NO_VEXPR_NODE )
				EVAL_DECL_TREE((yyvsp[(3) - (4)].enp));
			(yyval.enp) = NODE1(T_PROTO,(yyvsp[(3) - (4)].enp));
			SET_VN_STRING((yyval.enp), savestr((yyvsp[(1) - (4)].e_string)));
			}
    break;

  case 147:
#line 1087 "vectree.y"
    {
			if( SR_FLAGS((yyvsp[(1) - (4)].srp)) != SR_PROTOTYPE ){
				sprintf(YY_ERR_STR,"Subroutine %s multiply defined!?",SR_NAME((yyvsp[(1) - (4)].srp)));
				yyerror(THIS_QSP,  YY_ERR_STR);
				/* now what??? */
			}
			set_subrt_ctx(QSP_ARG  SR_NAME((yyvsp[(1) - (4)].srp)));		/* when do we unset??? */

			/* compare the two arg decl trees
			 * and issue a warning if they do not match.
			 */
			compare_arg_trees(QSP_ARG  (yyvsp[(3) - (4)].enp),SR_ARG_DECLS((yyvsp[(1) - (4)].srp)));

			/* use the new ones */
			SET_SR_ARG_DECLS((yyvsp[(1) - (4)].srp), (yyvsp[(3) - (4)].enp));
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
			SET_VN_STRING((yyval.enp), savestr(SR_NAME((yyvsp[(1) - (4)].srp))));
			}
    break;

  case 148:
#line 1123 "vectree.y"
    {
			Subrt *srp;
			srp=remember_subrt(QSP_ARG  (yyvsp[(1) - (3)].prec_p),VN_STRING((yyvsp[(2) - (3)].enp)),VN_CHILD((yyvsp[(2) - (3)].enp),0),(yyvsp[(3) - (3)].enp));
			SET_SR_PREC_PTR(srp, (yyvsp[(1) - (3)].prec_p));
			(yyval.enp)=NODE0(T_SUBRT);
			SET_VN_SUBRT((yyval.enp),srp);
			delete_subrt_ctx(QSP_ARG  VN_STRING((yyvsp[(2) - (3)].enp)));	/* this deletes the objects... */
			// But why is the context in existence here?
			COMPILE_SUBRT(srp);
			}
    break;

  case 149:
#line 1134 "vectree.y"
    {
			Subrt *srp;
			srp=remember_subrt(QSP_ARG  (yyvsp[(1) - (4)].prec_p),VN_STRING((yyvsp[(3) - (4)].enp)),VN_CHILD((yyvsp[(3) - (4)].enp),0),(yyvsp[(4) - (4)].enp));
			SET_SR_PREC_PTR(srp, (yyvsp[(1) - (4)].prec_p));
			SET_SR_FLAG_BITS(srp, SR_REFFUNC);
			/* set a flag to show returns ptr */
			(yyval.enp)=NODE0(T_SUBRT);
			SET_VN_SUBRT((yyval.enp),srp);
			delete_subrt_ctx(QSP_ARG  VN_STRING((yyvsp[(3) - (4)].enp)));	/* this deletes the objects... */
			COMPILE_SUBRT(srp);
			}
    break;

  case 150:
#line 1146 "vectree.y"
    {
			/* BUG make sure that precision matches prototype decl */
			Subrt *srp;
			srp=subrt_of(QSP_ARG  VN_STRING((yyvsp[(2) - (3)].enp)));
//#ifdef CAUTIOUS
//			if( srp == NO_SUBRT ) {
//				NODE_ERROR($2);
//				ERROR1("CAUTIOUS:  missing subrt!?");
//			}
//#endif /* CAUTIOUS */
			assert( srp != NO_SUBRT );

			update_subrt(QSP_ARG  srp,(yyvsp[(3) - (3)].enp));
			(yyval.enp)=NODE0(T_SUBRT);
			SET_VN_SUBRT((yyval.enp),srp);
			delete_subrt_ctx(QSP_ARG  VN_STRING((yyvsp[(2) - (3)].enp)));
			COMPILE_SUBRT(srp);
			}
    break;

  case 151:
#line 1168 "vectree.y"
    {
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 152:
#line 1172 "vectree.y"
    {
			(yyval.enp)=(yyvsp[(1) - (1)].enp);
			}
    break;

  case 153:
#line 1176 "vectree.y"
    {
			(yyval.enp)=NODE2(T_DECL_STAT_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 155:
#line 1183 "vectree.y"
    {
			if( (yyval.enp) != NO_VEXPR_NODE ) {
				// decl_stats are always evaluated,
				// to create the objects for compilation...
				SET_VN_FLAG_BITS((yyval.enp),NODE_FINISHED);
			}
			}
    break;

  case 156:
#line 1191 "vectree.y"
    {
			if( (yyval.enp) != NO_VEXPR_NODE ) {
				EVAL_IMMEDIATE((yyval.enp));
				// We don't release here,
				// because these nodes get passed up
				// to program nonterminal...
				SET_VN_FLAG_BITS((yyval.enp),NODE_FINISHED);
			}
			}
    break;

  case 157:
#line 1201 "vectree.y"
    {
			if( (yyval.enp) != NO_VEXPR_NODE ) {
				EVAL_IMMEDIATE((yyval.enp));
				SET_VN_FLAG_BITS((yyval.enp),NODE_FINISHED);
			}
			}
    break;

  case 158:
#line 1210 "vectree.y"
    { TOP_NODE=(yyvsp[(1) - (2)].enp);  }
    break;

  case 159:
#line 1212 "vectree.y"
    { TOP_NODE=(yyvsp[(1) - (1)].enp); }
    break;

  case 160:
#line 1213 "vectree.y"
    {
			(yyval.enp)=NODE2(T_STAT_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(2) - (3)].enp));
			if( (yyvsp[(1) - (3)].enp) != NULL && NODE_IS_FINISHED((yyvsp[(1) - (3)].enp)) &&
					(yyvsp[(2) - (3)].enp) != NULL && NODE_IS_FINISHED((yyvsp[(2) - (3)].enp)) )
				SET_VN_FLAG_BITS((yyval.enp),NODE_FINISHED);
			TOP_NODE=(yyval.enp);
			}
    break;

  case 161:
#line 1220 "vectree.y"
    {
			// We don't need to make lists of statements
			// already executed!?
			(yyval.enp)=NODE2(T_STAT_LIST,(yyvsp[(1) - (2)].enp),(yyvsp[(2) - (2)].enp));
			if( (yyvsp[(1) - (2)].enp) != NULL && NODE_IS_FINISHED((yyvsp[(1) - (2)].enp)) &&
					(yyvsp[(2) - (2)].enp) != NULL && NODE_IS_FINISHED((yyvsp[(2) - (2)].enp)) )
				SET_VN_FLAG_BITS((yyval.enp),NODE_FINISHED);
			TOP_NODE=(yyval.enp);
			}
    break;

  case 162:
#line 1230 "vectree.y"
    {
			(yyval.enp) = NO_VEXPR_NODE;
			TOP_NODE=(yyval.enp);
			}
    break;

  case 164:
#line 1240 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_BY);	}
    break;

  case 165:
#line 1241 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_CHAR);	}
    break;

  case 166:
#line 1242 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_STR);	}
    break;

  case 167:
#line 1243 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_SP);	}
    break;

  case 168:
#line 1244 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_DP);	}
    break;

  case 169:
#line 1245 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_CPX);	}
    break;

  case 170:
#line 1246 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_DBLCPX);	}
    break;

  case 171:
#line 1247 "vectree.y"
    { (yyval.prec_p)	= PREC_FOR_CODE(PREC_QUAT);	}
    break;

  case 172:
#line 1248 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_DBLQUAT);	}
    break;

  case 173:
#line 1249 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_IN);	}
    break;

  case 174:
#line 1250 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_DI);	}
    break;

  case 175:
#line 1251 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_LI);	}
    break;

  case 176:
#line 1252 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_UBY);	}
    break;

  case 177:
#line 1253 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_UIN);	}
    break;

  case 178:
#line 1254 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_UDI);	}
    break;

  case 179:
#line 1255 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_ULI);	}
    break;

  case 180:
#line 1256 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_BIT);	}
    break;

  case 181:
#line 1257 "vectree.y"
    { (yyval.prec_p)		= PREC_FOR_CODE(PREC_COLOR);	}
    break;

  case 182:
#line 1258 "vectree.y"
    { (yyval.prec_p)	= PREC_FOR_CODE(PREC_VOID);	}
    break;

  case 183:
#line 1263 "vectree.y"
    { (yyval.enp)=NODE1(T_INFO,(yyvsp[(3) - (4)].enp)); }
    break;

  case 184:
#line 1265 "vectree.y"
    { (yyval.enp)=NODE1(T_DISPLAY,(yyvsp[(3) - (4)].enp)); }
    break;

  case 185:
#line 1268 "vectree.y"
    { (yyval.enp)=NODE0(T_EXIT); }
    break;

  case 186:
#line 1269 "vectree.y"
    { (yyval.enp)=NODE0(T_EXIT); }
    break;

  case 187:
#line 1270 "vectree.y"
    { (yyval.enp)=NODE1(T_EXIT,(yyvsp[(3) - (4)].enp)); }
    break;

  case 188:
#line 1274 "vectree.y"
    {
			(yyval.enp)=NODE1(T_RETURN,NO_VEXPR_NODE);
			}
    break;

  case 189:
#line 1278 "vectree.y"
    {
			(yyval.enp)=NODE1(T_RETURN,NO_VEXPR_NODE);
			}
    break;

  case 190:
#line 1288 "vectree.y"
    {
			(yyval.enp)=NODE1(T_RETURN,(yyvsp[(2) - (2)].enp));
			}
    break;

  case 191:
#line 1292 "vectree.y"
    {
			(yyval.enp)=NODE1(T_RETURN,(yyvsp[(3) - (4)].enp));
			}
    break;

  case 192:
#line 1296 "vectree.y"
    {
			(yyval.enp)=NODE1(T_RETURN,(yyvsp[(2) - (2)].enp));
			}
    break;

  case 193:
#line 1302 "vectree.y"
    { (yyval.enp)=NODE2(T_SAVE,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp)); }
    break;

  case 194:
#line 1304 "vectree.y"
    { (yyval.enp)=NODE1(T_FILETYPE,(yyvsp[(3) - (4)].enp)); }
    break;

  case 195:
#line 1308 "vectree.y"
    {
			(yyval.enp)=NODE1(T_SCRIPT,(yyvsp[(3) - (4)].enp));
			SET_VN_SUBRT((yyval.enp), (yyvsp[(1) - (4)].srp));
			}
    break;

  case 196:
#line 1313 "vectree.y"
    {
			(yyval.enp)=NODE1(T_SCRIPT,NO_VEXPR_NODE);
			SET_VN_SUBRT((yyval.enp), (yyvsp[(1) - (3)].srp));
			}
    break;

  case 198:
#line 1321 "vectree.y"
    {
			sprintf(YY_ERR_STR,"undefined string pointer \"%s\"",(yyvsp[(1) - (1)].e_string));
			yyerror(THIS_QSP,  YY_ERR_STR);
			(yyval.enp)=NO_VEXPR_NODE;
			}
    break;

  case 199:
#line 1329 "vectree.y"
    {
			(yyval.enp) = NODE2(T_STRCPY,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			}
    break;

  case 200:
#line 1333 "vectree.y"
    {
			(yyval.enp) = NODE2(T_STRCAT,(yyvsp[(3) - (6)].enp),(yyvsp[(5) - (6)].enp));
			}
    break;

  case 201:
#line 1342 "vectree.y"
    {
			(yyval.enp) = NODE1(T_CALL_NATIVE,(yyvsp[(3) - (4)].enp));
			SET_VN_INTVAL((yyval.enp), (yyvsp[(1) - (4)].intval));
			}
    break;

  case 202:
#line 1356 "vectree.y"
    {
			Vec_Expr_Node *enp,*enp2;
			enp=NODE2(T_EXPR_LIST,(yyvsp[(5) - (12)].enp),(yyvsp[(7) - (12)].enp));
			enp2=NODE2(T_EXPR_LIST,(yyvsp[(9) - (12)].enp),(yyvsp[(11) - (12)].enp));
			(yyval.enp) = NODE3(T_FILL,(yyvsp[(3) - (12)].enp),enp,enp2);
			}
    break;

  case 203:
#line 1363 "vectree.y"
    {
			(yyval.enp) = NODE0(T_CLR_OPT_PARAMS);
			}
    break;

  case 204:
#line 1368 "vectree.y"
    {
			Vec_Expr_Node *enp1,*enp2,*enp3;
			enp1=NODE2(T_EXPR_LIST,(yyvsp[(3) - (14)].enp),(yyvsp[(5) - (14)].enp));
			enp2=NODE2(T_EXPR_LIST,(yyvsp[(7) - (14)].enp),(yyvsp[(9) - (14)].enp));
			enp3=NODE2(T_EXPR_LIST,(yyvsp[(11) - (14)].enp),(yyvsp[(13) - (14)].enp));
			(yyval.enp) = NODE3(T_ADD_OPT_PARAM,enp1,enp2,enp3);
			}
    break;

  case 205:
#line 1376 "vectree.y"
    {
			(yyval.enp) = NODE0(T_OPTIMIZE);
			SET_VN_SUBRT((yyval.enp), (yyvsp[(3) - (4)].srp));
			}
    break;

  case 206:
#line 1383 "vectree.y"
    { (yyval.enp)=NODE1(T_OUTPUT_FILE,(yyvsp[(3) - (4)].enp)); }
    break;

  case 207:
#line 1387 "vectree.y"
    { (yyval.enp)=NODE1(T_EXP_PRINT,(yyvsp[(3) - (4)].enp)); }
    break;

  case 208:
#line 1388 "vectree.y"
    { (yyval.enp)=NODE1(T_EXP_PRINT,(yyvsp[(3) - (4)].enp)); }
    break;

  case 209:
#line 1389 "vectree.y"
    { (yyval.enp)=NODE1(T_ADVISE,(yyvsp[(3) - (4)].enp)); }
    break;

  case 210:
#line 1390 "vectree.y"
    { (yyval.enp)=NODE1(T_WARN,(yyvsp[(3) - (4)].enp)); }
    break;

  case 212:
#line 1395 "vectree.y"
    { (yyval.e_string) = OBJ_NAME((yyvsp[(1) - (1)].dp)); }
    break;

  case 213:
#line 1397 "vectree.y"
    { (yyval.e_string) = ID_NAME((yyvsp[(1) - (1)].idp)); }
    break;

  case 214:
#line 1399 "vectree.y"
    { (yyval.e_string) = ID_NAME((yyvsp[(1) - (1)].idp)); }
    break;

  case 215:
#line 1401 "vectree.y"
    {
			yyerror(THIS_QSP,  (char *)"illegal attempt to use a keyword as an identifier");
			(yyval.e_string)="<illegal_keyword_use>";
			}
    break;

  case 216:
#line 1407 "vectree.y"
    {
			(yyval.enp) = NODE0(T_SCAL_DECL);
			// WHY VN_STRING and not VN_DECL_NAME???
			SET_VN_STRING((yyval.enp),savestr((yyvsp[(1) - (1)].e_string)));	/* bug need to save??? */
			}
    break;

  case 217:
#line 1419 "vectree.y"
    {
			delete_subrt_ctx(QSP_ARG  VN_STRING((yyvsp[(1) - (1)].enp)));
			}
    break;

  case 218:
#line 1423 "vectree.y"
    {
			delete_subrt_ctx(QSP_ARG  VN_STRING((yyvsp[(1) - (1)].enp)));
			}
    break;

  case 219:
#line 1427 "vectree.y"
    {
			/* function pointer */
			(yyval.enp) = NODE1(T_FUNCPTR_DECL,(yyvsp[(6) - (7)].enp));
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(3) - (7)].e_string)));
			}
    break;

  case 220:
#line 1432 "vectree.y"
    {
			(yyval.enp) = NODE1(T_CSCAL_DECL,(yyvsp[(3) - (4)].enp));
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (4)].e_string)));
			}
    break;

  case 221:
#line 1436 "vectree.y"
    {
			(yyval.enp) = NODE1(T_VEC_DECL,(yyvsp[(3) - (4)].enp));
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (4)].e_string)));
			}
    break;

  case 222:
#line 1440 "vectree.y"
    {
			(yyval.enp) = NODE2(T_CVEC_DECL,(yyvsp[(3) - (7)].enp),(yyvsp[(6) - (7)].enp));
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (7)].e_string)));
			}
    break;

  case 223:
#line 1444 "vectree.y"
    {
			// The type is stored at the parent node...
			// Since we "compile" the nodes depth first,
			// how does it get here?
			(yyval.enp)=NODE2(T_IMG_DECL,(yyvsp[(3) - (7)].enp),(yyvsp[(6) - (7)].enp));
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (7)].e_string)));
			}
    break;

  case 224:
#line 1451 "vectree.y"
    {
			(yyval.enp)=NODE3(T_CIMG_DECL,(yyvsp[(3) - (10)].enp),(yyvsp[(6) - (10)].enp),(yyvsp[(9) - (10)].enp));
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (10)].e_string)));
			}
    break;

  case 225:
#line 1455 "vectree.y"
    {
			(yyval.enp)=NODE3(T_SEQ_DECL,(yyvsp[(3) - (10)].enp),(yyvsp[(6) - (10)].enp),(yyvsp[(9) - (10)].enp));
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (10)].e_string)));
			}
    break;

  case 226:
#line 1459 "vectree.y"
    {
			Vec_Expr_Node *enp;
			enp = NODE2(T_EXPR_LIST,(yyvsp[(9) - (13)].enp),(yyvsp[(12) - (13)].enp));
			(yyval.enp)=NODE3(T_CSEQ_DECL,(yyvsp[(3) - (13)].enp),(yyvsp[(6) - (13)].enp),enp);
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (13)].e_string)));
			}
    break;

  case 227:
#line 1465 "vectree.y"
    {
			(yyval.enp) = NODE1(T_CSCAL_DECL,NO_VEXPR_NODE);
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (3)].e_string)));
			}
    break;

  case 228:
#line 1470 "vectree.y"
    {
			(yyval.enp) = NODE1(T_VEC_DECL,NO_VEXPR_NODE);
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (3)].e_string)));
			}
    break;

  case 229:
#line 1475 "vectree.y"
    {
			(yyval.enp) = NODE2(T_CVEC_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE);
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (5)].e_string)));
			}
    break;

  case 230:
#line 1480 "vectree.y"
    {
			(yyval.enp) = NODE2(T_IMG_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE);
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (5)].e_string)));
			}
    break;

  case 231:
#line 1485 "vectree.y"
    {
			(yyval.enp) = NODE3(T_CIMG_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE,NO_VEXPR_NODE);
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (7)].e_string)));
			}
    break;

  case 232:
#line 1490 "vectree.y"
    {
			(yyval.enp) = NODE3(T_SEQ_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE,NO_VEXPR_NODE);
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (7)].e_string)));
			}
    break;

  case 233:
#line 1495 "vectree.y"
    {
			(yyval.enp) = NODE3(T_CSEQ_DECL,NO_VEXPR_NODE,NO_VEXPR_NODE,NO_VEXPR_NODE);
			SET_VN_DECL_NAME((yyval.enp),savestr((yyvsp[(1) - (9)].e_string)));
			}
    break;

  case 234:
#line 1500 "vectree.y"
    {
			(yyval.enp)=NODE0(T_PTR_DECL);
			SET_VN_DECL_NAME((yyval.enp), savestr((yyvsp[(2) - (2)].e_string)));
			}
    break;

  case 235:
#line 1505 "vectree.y"
    {
			(yyval.enp)=NODE0(T_BADNAME);
			SET_VN_STRING((yyval.enp), savestr( FUNC_NAME( (yyvsp[(1) - (1)].func_p) )) );
			CURDLE((yyval.enp))
			NODE_ERROR((yyval.enp));
			WARN("illegal data function name use");
			}
    break;

  case 236:
#line 1513 "vectree.y"
    {
			(yyval.enp)=NODE0(T_BADNAME);
			SET_VN_STRING((yyval.enp), savestr( FUNC_NAME((yyvsp[(1) - (1)].func_p)) ) );
			CURDLE((yyval.enp))
			NODE_ERROR((yyval.enp));
			WARN("illegal size function name use");
			}
    break;

  case 238:
#line 1560 "vectree.y"
    {
			(yyval.enp)=NODE2(T_DECL_INIT,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 239:
#line 1563 "vectree.y"
    {
			(yyval.enp)=NODE2(T_DECL_ITEM_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 240:
#line 1565 "vectree.y"
    {
			Vec_Expr_Node *enp;
			enp=NODE2(T_DECL_INIT,(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp));
			(yyval.enp)=NODE2(T_DECL_ITEM_LIST,(yyvsp[(1) - (5)].enp),enp); }
    break;

  case 241:
#line 1576 "vectree.y"
    {
			(yyval.enp)=NODE1(T_DECL_STAT,(yyvsp[(2) - (2)].enp));
/*
			if( PREC_RDONLY($1) )
				SET_VN_DECL_FLAGS($$, DECL_IS_CONST);
*/
			SET_VN_DECL_PREC((yyval.enp),(yyvsp[(1) - (2)].prec_p));
			}
    break;

  case 243:
#line 1588 "vectree.y"
    {
			(yyval.enp)=NODE2(T_DECL_STAT_LIST,(yyvsp[(1) - (2)].enp),(yyvsp[(2) - (2)].enp));
			}
    break;

  case 244:
#line 1594 "vectree.y"
    {
			(yyval.enp) = NODE1(T_DECL_STAT,(yyvsp[(2) - (3)].enp));
/*
			if( $1 & DT_RDONLY )
				SET_VN_DECL_FLAGS($$, DECL_IS_CONST);
*/
			SET_VN_DECL_PREC((yyval.enp),(yyvsp[(1) - (3)].prec_p));
			EVAL_IMMEDIATE((yyval.enp));
			// don't release here because may be in subrt decl...
			// But we need to release otherwise!?
			}
    break;

  case 245:
#line 1605 "vectree.y"
    {
			(yyval.enp) = NODE1(T_EXTERN_DECL,(yyvsp[(3) - (4)].enp));
/*
			if( $2 & DT_RDONLY )
				SET_VN_DECL_FLAGS($$, DECL_IS_CONST);
*/
			SET_VN_DECL_PREC((yyval.enp),(yyvsp[(2) - (4)].prec_p));
			EVAL_IMMEDIATE((yyval.enp));
			// don't release here because may be in subrt decl...
			}
    break;

  case 246:
#line 1615 "vectree.y"
    {
			(yyval.enp) = NODE1(T_DECL_STAT,(yyvsp[(3) - (4)].enp));
/*
			if( $2 & DT_RDONLY )
				SET_VN_DECL_FLAGS($$, DECL_IS_CONST);
*/
			SET_VN_DECL_FLAG_BITS((yyval.enp),DECL_IS_STATIC);
			SET_VN_DECL_PREC((yyval.enp),(yyvsp[(2) - (4)].prec_p));
			EVAL_IMMEDIATE((yyval.enp));
			// don't release here because may be in subrt decl...
			}
    break;

  case 249:
#line 1633 "vectree.y"
    {
				if( (yyvsp[(5) - (5)].enp) != NULL )
					(yyval.enp) = NODE2(T_WHILE,(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp));
				else
					(yyval.enp) = NULL;
			}
    break;

  case 250:
#line 1640 "vectree.y"
    {
				if( (yyvsp[(5) - (5)].enp) != NULL )
					(yyval.enp) = NODE2(T_UNTIL,(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp));
				else
					(yyval.enp) = NULL;
			}
    break;

  case 251:
#line 1647 "vectree.y"
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

  case 252:
#line 1658 "vectree.y"
    {
			/* we want to preserve a strict tree structure */
			(yyval.enp) = NODE2(T_DO_WHILE,(yyvsp[(2) - (7)].enp),(yyvsp[(5) - (7)].enp));
			}
    break;

  case 253:
#line 1663 "vectree.y"
    {
			/* we want to preserve a strict tree structure */
			(yyval.enp) = NODE2(T_DO_UNTIL,(yyvsp[(2) - (7)].enp),(yyvsp[(5) - (7)].enp));
			}
    break;

  case 254:
#line 1670 "vectree.y"
    { (yyval.enp) = NODE2(T_CASE_STAT,(yyvsp[(1) - (2)].enp),(yyvsp[(2) - (2)].enp)); }
    break;

  case 256:
#line 1675 "vectree.y"
    { (yyval.enp) = NODE2(T_CASE_LIST,(yyvsp[(1) - (2)].enp),(yyvsp[(2) - (2)].enp)); }
    break;

  case 257:
#line 1679 "vectree.y"
    { (yyval.enp) = NODE1(T_CASE,(yyvsp[(2) - (3)].enp)); }
    break;

  case 258:
#line 1681 "vectree.y"
    { (yyval.enp) = NODE0(T_DEFAULT); }
    break;

  case 260:
#line 1686 "vectree.y"
    { (yyval.enp) = NODE2(T_SWITCH_LIST,(yyvsp[(1) - (2)].enp),(yyvsp[(2) - (2)].enp)); }
    break;

  case 261:
#line 1690 "vectree.y"
    { (yyval.enp)=NODE2(T_SWITCH,(yyvsp[(3) - (7)].enp),(yyvsp[(6) - (7)].enp)); }
    break;

  case 262:
#line 1695 "vectree.y"
    { (yyval.enp) = NODE3(T_IFTHEN,(yyvsp[(3) - (5)].enp),(yyvsp[(5) - (5)].enp),NO_VEXPR_NODE); }
    break;

  case 263:
#line 1697 "vectree.y"
    { (yyval.enp) = NODE3(T_IFTHEN,(yyvsp[(3) - (7)].enp),(yyvsp[(5) - (7)].enp),(yyvsp[(7) - (7)].enp)); }
    break;

  case 264:
#line 1712 "vectree.y"
    { (yyval.enp) = NULL; }
    break;

  case 277:
#line 1725 "vectree.y"
    { (yyval.enp)=NODE0(T_BREAK); }
    break;

  case 278:
#line 1726 "vectree.y"
    { (yyval.enp)=NODE0(T_CONTINUE); }
    break;

  case 279:
#line 1728 "vectree.y"
    {
			(yyval.enp) = NODE0(T_GO_BACK);
			SET_VN_STRING((yyval.enp), savestr(ID_NAME((yyvsp[(2) - (2)].idp))));
			}
    break;

  case 280:
#line 1733 "vectree.y"
    {
			(yyval.enp) = NODE0(T_GO_FWD);
			SET_VN_STRING((yyval.enp), savestr((yyvsp[(2) - (2)].e_string)));
			}
    break;

  case 281:
#line 1745 "vectree.y"
    { (yyval.enp) = (yyvsp[(1) - (1)].enp); }
    break;

  case 282:
#line 1747 "vectree.y"
    { (yyval.enp) = (yyvsp[(1) - (1)].enp); }
    break;

  case 283:
#line 1755 "vectree.y"
    {
			(yyval.enp)=NODE1(T_COMP_OBJ,(yyvsp[(2) - (3)].enp));
			}
    break;

  case 284:
#line 1760 "vectree.y"
    {
			(yyval.enp)=NODE1(T_LIST_OBJ,(yyvsp[(2) - (3)].enp));
			}
    break;

  case 286:
#line 1767 "vectree.y"
    {
			(yyval.enp)=NODE2(T_COMP_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 288:
#line 1778 "vectree.y"
    {
			(yyval.enp)=NODE2(T_ROW_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 290:
#line 1785 "vectree.y"
    {
			(yyval.enp)=NODE2(T_EXPR_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp));
			}
    break;

  case 292:
#line 1793 "vectree.y"
    { (yyval.enp)=NODE2(T_PRINT_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 296:
#line 1802 "vectree.y"
    { (yyval.enp)=NODE2(T_MIXED_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 298:
#line 1807 "vectree.y"
    { (yyval.enp)=NODE2(T_STRING_LIST,(yyvsp[(1) - (3)].enp),(yyvsp[(3) - (3)].enp)); }
    break;

  case 299:
#line 1811 "vectree.y"
    {
			const char *s;
			s=savestr((yyvsp[(1) - (1)].e_string));
			(yyval.enp)=NODE0(T_STRING);
			SET_VN_STRING((yyval.enp), s);
				/* BUG?  make sure to free if tree deleted */
			}
    break;

  case 300:
#line 1819 "vectree.y"
    { (yyval.enp) = NODE1(T_NAME_FUNC,(yyvsp[(3) - (4)].enp)); }
    break;

  case 306:
#line 1834 "vectree.y"
    { (yyval.enp) = (yyvsp[(2) - (3)].enp); }
    break;

  case 307:
#line 1835 "vectree.y"
    { (yyval.enp) = (yyvsp[(2) - (3)].enp); }
    break;


/* Line 1267 of yacc.c.  */
#line 5172 "vectree.c"
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
      yyerror (qsp, YY_("syntax error"));
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
	    yyerror (qsp, yymsg);
	  }
	else
	  {
	    yyerror (qsp, YY_("syntax error"));
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
		      yytoken, &yylval, qsp);
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
		  yystos[yystate], yyvsp, qsp);
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
  yyerror (qsp, YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEOF && yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval, qsp);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp, qsp);
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


#line 1883 "vectree.y"


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


