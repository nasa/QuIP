
#ifndef NEXPR_H
#define NEXPR_H

#include "data_obj.h"

typedef enum {
	N_OBJNAME,			/*  0 */
	N_SUBSCRIPT,			/*  1 */
	N_CSUBSCRIPT,			/*  2 */
	N_SIZABLE,			/*  3 */
//	N_SUBSIZ,			/*  4 */
//	N_CSUBSIZ,			/*  5 */
	N_TSABLE,			/*  6 */
	N_TSFUNC,			/*  7 */
	N_LITDBL,			/*  8 */
	N_MATH0FUNC,			/*  9 */
	N_MATH1FUNC,			/* 10 */
	N_MATH2FUNC,			/* 11 */
	N_DATAFUNC,			/* 12 */
	N_SIZFUNC,			/* 13 */
	N_MISCFUNC,			/* 14 */
	N_STRFUNC,			/* 15 */
	N_STR2FUNC,			/* 16 */
	N_STR3FUNC,			/* 17 */
	N_PLUS,				/* 18 */
	N_MINUS,			/* 19 */
	N_DIVIDE,			/* 20 */
	N_TIMES,			/* 21 */
	N_MODULO,			/* 22 */
	N_BITAND,			/* 23 */
	N_BITOR,			/* 24 */
	N_BITXOR,			/* 25 */
	N_SHL,				/* 26 */
	N_SHR,				/* 27 */
	N_NOT,				/* 28 */
	N_BITCOMP,			/* 29 */
	N_UMINUS,			/* 30 */
	N_EQUIV,			/* 31 */
	N_LOGOR,			/* 32 */
	N_LOGAND,			/* 33 */
	N_LT,				/* 34 */
	N_GT,				/* 35 */
	N_GE,				/* 36 */
	N_LE,				/* 37 */
	N_NE,				/* 38 */
	N_CONDITIONAL,			/* 39 */
	N_SCALAR_OBJ,			/* 40 */
	N_STRING,			/* 41 */
	N_LOGXOR			/* 42 */
	/* N_SLCT_CHAR */		/* 40 */	/* obsolete? */
} Scalar_Expr_Node_Code;

typedef struct scalar_expression_node {
	Scalar_Expr_Node_Code	sen_code;
	const char *		sen_string;
	const char *		sen_string2;
	index_t			sen_index;
	struct scalar_expression_node	*sen_child[3];
	double			sen_dblval;
} Scalar_Expr_Node;

#define NO_EXPR_NODE ((Scalar_Expr_Node *)NULL)


/* globals */

extern Data_Obj *(*obj_func)(QSP_ARG_DECL  const char *);
extern Data_Obj *(*sub_func)(QSP_ARG_DECL  Data_Obj *,index_t);
extern Data_Obj *(*csub_func)(QSP_ARG_DECL  Data_Obj *,index_t);

extern double parse_number(QSP_ARG_DECL  const char **);
extern double pexpr(QSP_ARG_DECL  const char *);


#endif /* !NEXPR_H */

