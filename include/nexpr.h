
#ifndef NEXPR_H
#define NEXPR_H

//#include "shape_bits.h"
//#include "dobj_basic.h"
#include "quip_fwd.h"
#include "data_obj.h"
#include "typed_scalar.h"

typedef enum {
	N_OBJNAME,			/*  0 */
	N_SUBSCRIPT,			/*  1 */
	N_CSUBSCRIPT,			/*  2 */
	N_TSABLE,			/*  3 */
	N_STRVFUNC,			/*  4 */
	N_LITSTR,			/*  5 */
	N_INT1FUNC,			/*  6 */
	N_TSFUNC,			/*  7 */
	N_LITNUM,			/*  8 */
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
	N_QUOT_STR,			/* 41 */
	N_LOGXOR,			/* 42 */
	N_CHARFUNC,			/* 43 */
	N_ILACEFUNC,			/* 44 */
	N_POSNFUNC,			/* 45 */
	N_STRING,			/* 46 */
	N_STRV2FUNC,			/* 47 */
	N_DOBJVFUNC,			/* 48 */
	/* N_SLCT_CHAR */		/* 49 */	/* obsolete? */
} Scalar_Expr_Node_Code;

#define MAX_SEN_CHILDREN  3

struct scalar_expr_node {
	Scalar_Expr_Node_Code	sen_code;
//	const char *		sen_string;
//	const char *		sen_string2;
	index_t			sen_index;
	Function *		sen_func_p;
	Scalar_Expr_Node	*sen_child[MAX_SEN_CHILDREN];
	//double		sen_dblval;
	Typed_Scalar *		sen_tsp;
};

#define NO_EXPR_NODE ((Scalar_Expr_Node *)NULL)


/* globals */

//extern double parse_number(QSP_ARG_DECL  const char **);
//extern double pexpr(QSP_ARG_DECL  const char *);
extern Typed_Scalar * parse_number(QSP_ARG_DECL  const char **);
extern Typed_Scalar * pexpr(QSP_ARG_DECL  const char *);
extern const char *function_name(void *objp);
extern const char *eval_scalexp_string(QSP_ARG_DECL  Scalar_Expr_Node *enp);

extern void set_obj_funcs(
	Data_Obj *(*obj_get_func)(QSP_ARG_DECL  const char *),
	Data_Obj *(*obj_check_func)(QSP_ARG_DECL  const char *),
	Data_Obj *(*sub_func)(QSP_ARG_DECL  Data_Obj*,index_t),
	Data_Obj *(*csub_func)(QSP_ARG_DECL  Data_Obj*,index_t) );

#ifdef BUILD_FOR_OBJC
extern int check_ios_sizable_func( double *retval, Function *funcp, Scalar_Expr_Node *argp );
extern int check_ios_strv_func( const char **strptr, Function *funcp, Scalar_Expr_Node *argp );
extern int check_ios_positionable_func( double *retval, Function *funcp, Scalar_Expr_Node *argp );
extern int check_ios_interlaceable_func( double *retval, Function *funcp, Scalar_Expr_Node *argp );
#endif /* BUILD_FOR_OBJC */

extern void set_eval_szbl_func(QSP_ARG_DECL  Item * (*func)(QSP_ARG_DECL  Scalar_Expr_Node *) );
extern void set_eval_dobj_func(QSP_ARG_DECL  Data_Obj * (*func)(QSP_ARG_DECL  Scalar_Expr_Node *) );

// temporary!  BUG
/*static*/ extern void dump_etree(QSP_ARG_DECL  Scalar_Expr_Node *enp);

extern Typed_Scalar * eval_expr(QSP_ARG_DECL  Scalar_Expr_Node *);
#define EVAL_SCALEXP_STRING( s ) eval_scalexp_string( QSP_ARG  s )
#define EVAL_EXPR( s )		eval_expr( QSP_ARG  s )
#define EVAL_SZBL_EXPR( s )	eval_szbl_expr( QSP_ARG  s )

#endif /* !NEXPR_H */

