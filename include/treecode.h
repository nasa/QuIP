
#ifndef BINARY_BOOLOP_CASES

typedef enum {
	T_PROTO,				/* 0 */
	/* flow control */
	T_IFTHEN,				/* 1 */
	T_WHILE,				/* 2 */
	T_UNTIL,				/* 3 */
	T_FOR,					/* 4 */
	T_DO_WHILE,				/* 5 */
	T_DO_UNTIL,				/* 6 */
	T_CONTINUE,				/* 7 */
	T_SWITCH,				/* 8 */
	T_CASE,					/* 9 */
	T_CASE_STAT,				/* 10 */
	T_SWITCH_LIST,				/* 11 */
	T_BREAK,				/* 12 */
	T_DEFAULT,				/* 13 */
	T_GO_BACK,				/* 14 */
	T_GO_FWD,				/* 15 */
	T_LABEL,				/* 16 */
	T_CASE_LIST,				/* 17 */
	/* lists */
	T_STRING_LIST,				/* 18 */
	T_MIXED_LIST,				/* 19 */
	T_PRINT_LIST,				/* 20 */
	T_STAT_LIST,				/* 21 */
	T_DECL_STAT_LIST,			/* 22 */
	T_DECL_ITEM_LIST,			/* 23 */
	T_DECL_STAT,				/* 24 */
	T_EXTERN_DECL,				/* 25 */
	/* autoinitialization */
	T_DECL_INIT,				/* 26 */
	/* individual declarations */
	T_PTR_DECL,				/* 27 */
	T_SCAL_DECL,				/* 28 */
	T_VEC_DECL,				/* 29 */
	T_IMG_DECL,				/* 30 */
	T_SEQ_DECL,				/* 31 */
	T_CSCAL_DECL,				/* 32 */
	T_CVEC_DECL,				/* 33 */
	T_CIMG_DECL,				/* 34 */
	T_CSEQ_DECL,				/* 35 */
	T_FUNCPTR_DECL,				/* 36 */
	T_LIST_OBJ,				/* 37 */
	T_COMP_OBJ,				/* 38 */
	T_EXPR_LIST,				/* 39 */
	T_EXP_PRINT,				/* 40 */
	/* subroutine stuff */
	T_SUBRT,				/* 41 */
	T_RETURN,				/* 42 */
	T_EXIT,					/* 43 */
	T_ARGLIST,				/* 44 */
	T_CALLFUNC,				/* 45 */
	T_CALL_NATIVE,				/* 46 */
	/* indirect function call stuff */
	T_FUNCPTR,				/* 47 */
	T_INDIR_CALL,				/* 48 */
	T_FUNCREF,				/* 49 */
	/* builtin functions */
	T_INFO,					/* 50 */
	T_DISPLAY,				/* 51 */
	T_OBJ_LOOKUP,				/* 52 */
	/* assignments */
	T_ASSIGN,				/* 53 */
	T_DIM_ASSIGN,				/* 54 */ /* img=row */
	T_SET_STR,				/* 55 */
	T_SET_PTR,				/* 56 */
	T_SET_FUNCPTR,				/* 57 */
	/* syntax checking nodes */
	T_BADNAME,				/* 58 */
	T_UNDEF,				/* 59 */
	/* object related nodes */
	T_REFERENCE,				/* 60 */
	T_DEREFERENCE,				/* 61 */
	T_DYN_OBJ,				/* 62 */
	T_POINTER,				/* 63 */
	T_NAME_FUNC,				/* 64 */
	T_STR_PTR,				/* 65 */
	/* leaf types */
	T_STRING,				/* 66 */
	T_SCRIPT,				/* 67 */
	/* scalar constants */
	T_LIT_DBL,				/* 68 */
	T_LIT_INT,				/* 69 */
	/* binary boolean operators */
	T_BOOL_AND,				/* 70 */
	T_BOOL_OR,				/* 71 */
	T_BOOL_XOR,				/* 72 */
	T_BOOL_NOT,				/* 73 */
	/* scalar numeric comparison */
	T_BOOL_GT,				/* 74 */
	T_BOOL_LT,				/* 75 */
	T_BOOL_GE,				/* 76 */
	T_BOOL_LE,				/* 77 */
	T_BOOL_NE,				/* 78 */
	T_BOOL_EQ,				/* 79 */
	/* binary ptr op */
	T_BOOL_PTREQ,				/* 80 */
	/* unary ops (scalar) */
	T_UMINUS,				/* 81 */
	T_RECIP,				/* 82 */
	T_POSTINC,				/* 83 */
	T_PREINC,				/* 84 */
	T_POSTDEC,				/* 85 */
	T_PREDEC,				/* 86 */
	T_CONJ,					/* 87 */ /* complex conjugate */
	/* binary ops (scalar) */
	T_PLUS,					/* 88 */ /* binary scalar ops */
	T_MINUS,				/* 89 */
	T_TIMES,				/* 90 */
	T_DIVIDE,				/* 91 */
	/* multiple operands, scalar args... */
	T_SCALMAX,				/* 92 */
	T_SCALMIN,				/* 93 */
	T_MODULO,				/* 94 */
	T_BITAND,				/* 95 */
	T_BITOR,				/* 96 */
	T_BITXOR,				/* 97 */
	T_BITLSHIFT,				/* 98 */
	T_BITRSHIFT,				/* 99 */
	T_BITCOMP,				/* 100 */
	/* scalar functions */
	T_MATH0_FN,				/* 101 */
	T_MATH1_FN,				/* 102 */
	T_MATH2_FN,				/* 103 */
	T_MAXVAL,				/* 104 */
	T_MINVAL,				/* 105 */
	T_SUM,					/* 106 */
	T_SIZE_FN,				/* 107 */
	T_SS_S_CONDASS,				/* 108 */
	T_MATH0_VFN,				/* 109 */
	T_MATH1_VFN,				/* 110 */
	T_MATH2_VFN,				/* 111 */
	/* vector-scalar functions */
	T_MATH2_VSFN,				/* 112 */
	T_DATA_FN,				/* 113 */ /* value?  and what else?  scalar? */
	T_STR1_FN,				/* 114 */ /* strlen, iof_exists, etc */
	T_STR2_FN,				/* 115 */ /* strcmp */
	T_MISC_FN,				/* 116 */
	T_EQUIVALENCE,				/* 117 */
	T_TRANSPOSE,				/* 118 */ /* unary vector function w/ vector result */
	T_VCOMP,				/* 119 */
	T_SS_B_CONDASS,				/* 120 */
	T_VS_S_CONDASS,				/* 121 */
	T_DILATE,				/* 122 */
	T_ERODE,				/* 123 */
	T_FILL,					/* 124 */
	T_WRAP,					/* 125 */ /* pre-defined numerical functions */
	T_SCROLL,				/* 126 */

	T_VS_B_CONDASS,				/* 127 */
	T_VV_S_CONDASS,				/* 128 */
	T_VV_B_CONDASS,				/* 129 */ /* trinary vector-vector-vector */

	T_MAX_TIMES,				/* 130 */ /* single vector arg, multiple results */
	T_MIN_TIMES,				/* 131 */ /* single vector arg, multiple results */
	T_MAX_INDEX,				/* 132 */
	T_MIN_INDEX,				/* 133 */
	T_INNER,				/* 134 */
	T_SQUARE_SUBSCR,			/* 135 */ /* unary object operators that change size */
	T_CURLY_SUBSCR,				/* 136 */
	T_REAL_PART,				/* 137 */
	T_IMAG_PART,				/* 138 */
	T_SUBVEC,				/* 139 */
	T_CSUBVEC,				/* 140 */
	T_SUBSAMP,				/* 141 */
	T_CSUBSAMP,
	T_ENLARGE,
	T_REDUCE,
	T_DFT,					/* fft stuff */
	T_IDFT,
	T_RDFT,
	T_RIDFT,
	T_RAMP,
	T_LOOKUP,
	T_CLR_OPT_PARAMS,			/* cstepit optimization */
	T_ADD_OPT_PARAM,
	T_OPTIMIZE,
	T_ADVISE,				/* print stuff */
	T_WARN,
	T_STRCAT,				/* string functions */
	T_STRCPY,
	T_FILE_EXISTS,				/* fileio stuff */
	T_OUTPUT_FILE,
	T_LOAD,
	T_SAVE,
	T_FILETYPE,
	T_TYPECAST,				/* unary, scalar or vector */
	T_PERFORM,				/* ??? */
	T_FIX_SIZE,				/* a hack for shape resolution */
	T_I,					/* sqrt(-1), for matlab */
	T_MFILE,			/* matlab */
	T_MCMD,				/* matlab */
	T_CLEAR,			/* matlab */
	T_CLF,				/* matlab */
	T_DRAWNOW,			/* matlab */
	T_ROW,					/* from matlab, maybe mergable... */
	/* T_ROWLIST, */				/* from matlab */
	T_SUBSCRIPT1,				/* matlab, merge? */
	T_SUBMTRX,				/* matlab */
	T_INDEX_SPEC,				/* matlab, ':' operator, e.g. a[2:4] */
	T_INDEX_LIST,
	T_SSCANF,
	T_NOP,
	T_MLFUNC,
	T_RET_LIST,
	T_RANGE,
	T_RANGE2,
	T_NULLVEC,
	T_STRUCT,
	T_FIRST_INDEX,
	T_LAST_INDEX,
	T_ENTIRE_RANGE,
	T_OBJ_LIST,
	T_GLOBAL,
	T_POWER,
	T_FVSPOW,
	T_FVSPOW2,
	T_FVPOW,
	T_FIND,
	/* end matlab */
	T_STATIC_OBJ,
	T_VV_FUNC,
	T_VS_FUNC,
	T_ROW_LIST,				/* like EXPR_LIST but elements have to have same shape */
	T_COMP_LIST,				/* like ROW_LIST but for expressions in curly braces */
	T_END,					/* special token for interactive input */

	T_VV_VV_CONDASS,			/* new condass functions */
	T_VS_VV_CONDASS,			/* new condass functions */
	T_SS_VV_CONDASS,			/* new condass functions */
	T_VV_VS_CONDASS,			/* new condass functions */
	T_VS_VS_CONDASS,			/* new condass functions */
	T_SS_VS_CONDASS,			/* new condass functions */
	T_VV_SS_CONDASS,			/* new condass functions */
	T_VS_SS_CONDASS,			/* new condass functions */
	T_SS_SS_CONDASS,			/* new condass functions */
	T_INT1_FN,				/* new for isnan etc. */
	T_INT1_VFN,				/* new for isnan etc. */
	T_STR3_FN,
	T_STRV_FN,
	T_CHAR_FN,
	T_CHAR_VFN,

	N_TREE_CODES				/* this has to be last */
} Tree_Code;

#define ALL_MATLAB_CASES					\
								\
		case T_SUBSCRIPT1:				\
		SOME_MATLAB_CASES


#define SOME_MATLAB_CASES					\
								\
		case T_I:					\
		case T_MFILE:					\
		case T_CLEAR:					\
		case T_DRAWNOW:					\
		case T_CLF:					\
		case T_MCMD:					\
		case T_ROW:					\
		/* case T_ROWLIST: */					\
		case T_SUBMTRX:					\
		case T_SSCANF:					\
		case T_NOP:					\
		case T_MLFUNC:					\
		case T_RET_LIST:				\
		case T_NULLVEC:					\
		case T_INDEX_LIST:				\
		case T_RANGE:				\
		case T_RANGE2:				\
		case T_STRUCT:				\
		case T_FIRST_INDEX:				\
		case T_LAST_INDEX:				\
		case T_OBJ_LIST:				\
		case T_GLOBAL:				\
		case T_FIND:				\
		case T_INDEX_SPEC:

#define ALL_PRINT_CASES						\
								\
		case T_WARN:					\
		case T_ADVISE:					\
		case T_EXP_PRINT:

#define ALL_OBJFUNC_CASES					\
								\
		case T_DISPLAY:					\
		case T_INFO:

#define ALL_FILEIO_CASES					\
								\
		case T_FILE_EXISTS:				\
		case T_OUTPUT_FILE:				\
		case T_LOAD:					\
		case T_SAVE:					\
		case T_FILETYPE:

#define ALL_STRFUNC_CASES					\
								\
		case T_STR1_FN:					\
		case T_STR2_FN:					\
		case T_STR3_FN:					\
		case T_STRV_FN:					\
		case T_CHAR_FN:					\
		case T_STRCAT:					\
		case T_STRCPY:

#define ALL_DFT_CASES						\
								\
		case T_DFT:					\
		case T_IDFT:					\
		case T_RDFT:					\
		case T_RIDFT:

#define ALL_INCDEC_CASES					\
								\
		case T_PREINC:					\
		case T_PREDEC:					\
		case T_POSTINC:					\
		case T_POSTDEC:

/* are MINVAL/MAXVAL fucntions or binary operators??? */
/* they take a list arg... */

#define ALL_PROJECTION_CASES					\
								\
		case T_PROJECT_OP:				\
		case T_MATH0_FN:				\
		case T_MATH1_FN:				\
		case T_MATH2_FN:				\
		case T_INT1_FN:					\
		case T_SIZE_FN:

#define ALL_SCALAR_FUNCTION_CASES				\
								\
		case T_MATH0_FN:				\
		case T_MATH1_FN:				\
		case T_MATH2_FN:				\
		case T_INT1_FN:					\
		case T_SIZE_FN:

#define ALL_SCALAR_CONSTANT_CASES				\
								\
		case T_LIT_DBL:					\
		case T_LIT_INT:

/* For the name ALL_SCALAR_CASES make sense, this should also include SCALAR_BINOP... */

#define ALL_SCALAR_CASES					\
								\
		ALL_SCALAR_FUNCTION_CASES			\
		ALL_SCALAR_CONSTANT_CASES

#define ALL_LIST_NODE_CASES					\
								\
		case T_ARGLIST:					\
		case T_EXPR_LIST:				\
		case T_DECL_ITEM_LIST:				\
		case T_DECL_STAT_LIST:				\
		case T_STAT_LIST:				\
		case T_CASE_LIST:				\
		case T_STRING_LIST:				\
		case T_MIXED_LIST:				\
		case T_SWITCH_LIST:				\
		case T_PRINT_LIST:

#define ALL_CTL_FLOW_CASES					\
								\
		case T_IFTHEN:					\
		case T_WHILE:					\
		case T_UNTIL:					\
		case T_FOR:					\
		case T_DO_WHILE:				\
		case T_DO_UNTIL:				\
		case T_SWITCH:					\
		case T_CASE:					\
		case T_CASE_STAT:				\
		case T_DEFAULT:					\
		case T_BREAK:					\
		case T_GO_FWD:					\
		case T_GO_BACK:					\
		case T_LABEL:					\
		case T_CONTINUE:

#define ALL_UNARY_CASES						\
								\
		case T_DILATE:					\
		case T_ERODE:					\
		case T_FILL:					\
		case T_WRAP:					\
		case T_SCROLL:


#define ALL_MATHFN_CASES					\
								\
		ALL_MATH_VFN_CASES				\
		ALL_MATH_SFN_CASES


#define ALL_MATH_SFN_CASES					\
								\
		case T_INT1_FN:					\
		case T_MATH0_FN:				\
		case T_MATH1_FN:				\
		case T_MATH2_FN:

#define ALL_MATH_VFN_CASES					\
								\
		case T_INT1_VFN:				\
		case T_MATH0_VFN:				\
		case T_MATH1_VFN:				\
		case T_MATH2_VFN:				\
		case T_MATH2_VSFN:

#define ALL_MATH2FN_CASES					\
								\
		case T_MATH2_FN:				\
		case T_MATH2_VFN:				\
		case T_MATH2_VSFN:

/* Originally we had T_EQUIVALENCE here too... */

#define MOST_OBJREF_CASES					\
		MOST_NON_SUBVEC_OBJREF_CASES			\
		case T_SUBVEC:					\
		case T_CSUBVEC:	

#define ALL_OBJREF_CASES					\
								\
		NONSUBVEC_OBJREF_CASES				\
		case T_SUBVEC:					\
		case T_CSUBVEC:	

#define NONSUBVEC_OBJREF_CASES					\
								\
		case T_COMP_OBJ:				\
		MOST_NON_SUBVEC_OBJREF_CASES


#define MOST_NON_SUBVEC_OBJREF_CASES				\
								\
		case T_LIST_OBJ:				\
		case T_OBJ_LOOKUP:				\
		case T_DEREFERENCE:				\
		case T_DYN_OBJ:					\
		case T_STATIC_OBJ:					\
		case T_SUBSAMP:					\
		case T_CSUBSAMP:				\
		case T_SQUARE_SUBSCR:				\
		case T_CURLY_SUBSCR:				\
		case T_SUBSCRIPT1:				\
		case T_REAL_PART:				\
		case T_IMAG_PART:

#define ALL_VARIABLE_SIZECHNG_CASES				\
								\
		case T_SUBSAMP:					\
		case T_CSUBSAMP:				\
		case T_SUBVEC:					\
		case T_CSUBVEC:					\
		case T_SQUARE_SUBSCR:				\
		case T_CURLY_SUBSCR:

// Why "scal" ???

#define ALL_SCALINT_BINOP_CASES					\
								\
		case T_MODULO:					\
		case T_BITAND:					\
		case T_BITOR:					\
		case T_BITXOR:					\
		case T_BITLSHIFT:				\
		case T_BITRSHIFT:


#define OTHER_UNMIXED_SCALAR_MATHOP_CASES			\
								\
		case T_PLUS:					\
		case T_MINUS:					\
		case T_POWER:					\
		case T_DIVIDE:

#define OTHER_SCALAR_MATHOP_CASES				\
								\
		case T_TIMES:					\
		OTHER_UNMIXED_SCALAR_MATHOP_CASES


#define OTHER_UNMIXED_SCALAR_BINOP_CASES			\
								\
		OTHER_UNMIXED_SCALAR_MATHOP_CASES

#define OTHER_SCALAR_BINOP_CASES				\
								\
		OTHER_SCALAR_MATHOP_CASES

#define OTHER_BINOP_CASES					\
								\
		OTHER_SCALAR_BINOP_CASES			\
		OTHER_VECTOR_BINOP_CASES

#define ALL_UNMIXED_SCALAR_BINOP_CASES				\
								\
		ALL_SCALINT_BINOP_CASES				\
		OTHER_UNMIXED_SCALAR_BINOP_CASES

#define ALL_SCALAR_BINOP_CASES					\
								\
		ALL_SCALINT_BINOP_CASES				\
		OTHER_SCALAR_BINOP_CASES


#define ALL_DECL_STAT_CASES					\
								\
		case T_DECL_STAT:				\
		case T_EXTERN_DECL:

#define ALL_DECL_CASES						\
								\
		ALL_DECL_STAT_CASES				\
		ALL_DECL_ITEM_CASES

#define ALL_DECL_ITEM_CASES					\
								\
		case T_DECL_INIT:				\
		NON_INIT_DECL_ITEM_CASES


#define NON_INIT_DECL_ITEM_CASES					\
								\
		case T_SCAL_DECL:				\
		case T_IMG_DECL:				\
		case T_VEC_DECL:				\
		case T_SEQ_DECL:				\
		case T_PTR_DECL:				\
		case T_CIMG_DECL:				\
		case T_CVEC_DECL:				\
		case T_CSCAL_DECL:				\
		case T_CSEQ_DECL:				\
		case T_FUNCPTR_DECL:


#define ALL_INTEGER_VV_CASES					\
								\
		case T_FVMOD:					\
		case T_VAND:					\
		case T_VOR:					\
		case T_VXOR:					\
		case T_VSHR:					\
		case T_VSHL:

#define OTHER_VV_BINOP_CASES					\
								\
		case T_FVMUL:					\
		case T_FVPOW:					\
		OTHER_UNMIXED_VV_BINOP_CASES

#define OTHER_UNMIXED_VV_BINOP_CASES				\
								\
		case T_FVADD:					\
		case T_FVSUB:					\
		case T_FVDIV:					\
		case T_FVMIN:					\
		case T_FVMAX:

/* What are the old CONDASS cases?  In the new scheme, the first two
 * key chars indicate the sources (VV, VS, SS) and the second two
 * indicate the test (VV,VS).  It is obvious that VVV -> VV_VV
 * and SSS -> SS_VS.  That leaves VVS, VSS, SVV and SVS.
 */

#define TRINARY_CONDASS_CASES					\
		case T_VV_B_CONDASS:				\
		case T_VS_B_CONDASS:				\
		case T_SS_B_CONDASS:				\
		case T_VV_S_CONDASS:				\
		case T_VS_S_CONDASS:				\
		case T_SS_S_CONDASS:

#define ALL_NEW_CONDASS_CASES					\
		case T_VV_VV_CONDASS:				\
		case T_VS_VV_CONDASS:				\
		case T_SS_VV_CONDASS:				\
		case T_VV_VS_CONDASS:				\
		case T_VS_VS_CONDASS:				\
		case T_SS_VS_CONDASS:				\
		case T_VV_SS_CONDASS:				\
		case T_VS_SS_CONDASS:				\
		case T_SS_SS_CONDASS:

#define ALL_CONDASS_CASES					\
								\
		ALL_NEW_CONDASS_CASES				\
		TRINARY_CONDASS_CASES

#define NON_BITMAP_CONDASS_CASES				\
								\
		ALL_NEW_CONDASS_CASES				\
		case T_VV_S_CONDASS:				\
		case T_VS_S_CONDASS:				\
		case T_SS_S_CONDASS:

#define ALL_NUMERIC_COMPARISON_CASES				\
								\
		case T_BOOL_GT:					\
		case T_BOOL_LT:					\
		case T_BOOL_GE:					\
		case T_BOOL_LE:					\
		case T_BOOL_NE:					\
		case T_BOOL_EQ:

#define ALL_BOOLOP_CASES					\
								\
		BINARY_BOOLOP_CASES				\
		case T_BOOL_NOT:

#define BINARY_BOOLOP_CASES					\
								\
		case T_BOOL_AND:					\
		case T_BOOL_OR:					\
		case T_BOOL_XOR:					\


#endif /* undef BINARY_BOOLOP_CASES */

