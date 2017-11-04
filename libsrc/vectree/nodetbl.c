#include "quip_config.h"

#include <stdlib.h>	/* qsort() */
#include "quip_prot.h"
#include "vectree.h"

/* the entries that are now ND_NULL used to be 0 - w/ g++, this caused
 * a type mismatch error, these nodes have no data, but are we really
 * sure that no code depends on the zero value?
 */

#define ND_NULL	ND_NONE

Tree_Node_Type tnt_tbl[N_TREE_CODES]={
{	T_SUBRT_DECL,	"subroutine_decl",	0,	NO_SHP,	ND_NONE	},
{	T_STAT_LIST,	"stat_list",		2,	NO_SHP,	ND_LIST	},

{	T_DECL_STAT_LIST,"decl_stat_list",	2,	NO_SHP,	ND_LIST	},
{	T_DECL_STAT,	"decl_stat",		1,	NO_SHP,	ND_DECL	},
{	T_EXTERN_DECL,	"extern_decl",		1,	NO_SHP,	ND_DECL	},
{	T_DECL_ITEM_LIST,"decl_item_list",	2,	NO_SHP,	ND_LIST	},
{	T_DECL_INIT,	"decl_init",		2,	CP_SHP,	ND_NONE	},
{	T_SCAL_DECL,	"scal_decl",		0,	CP_SHP,	ND_DECL	},
{	T_VEC_DECL,	"vec_decl",		1,	CP_SHP,	ND_DECL	},
{	T_IMG_DECL,	"img_decl",		2,	CP_SHP,	ND_DECL	},
{	T_SEQ_DECL,	"seq_decl",		3,	CP_SHP,	ND_DECL	},
{	T_CSCAL_DECL,	"cscal_decl",		1,	CP_SHP,	ND_DECL	},
{	T_CVEC_DECL,	"cvec_decl",		2,	CP_SHP,	ND_DECL	},
{	T_CIMG_DECL,	"cimg_decl",		3,	CP_SHP,	ND_DECL	},
{	T_CSEQ_DECL,	"cseq_decl",		3,	CP_SHP,	ND_DECL	},
{	T_PTR_DECL,	"ptr_decl",		0,	CP_SHP,	ND_DECL	},

{	T_ARGLIST,	"arglist",		2,	NO_SHP,	ND_LIST	},
{	T_CALLFUNC,	"callfunc",		1,	CP_SHP,	ND_CALLF	},
{	T_CALL_NATIVE,	"call_native",		1,	CP_SHP,	ND_CALLF	},
{	T_RETURN,	"return",		1,	PT_SHP,	ND_NONE	},
{	T_EXIT,		"exit",			1,	NO_SHP,	ND_NONE	},
{	T_STRING,	"string",		0,	CP_SHP,	ND_STRING	},
{	T_STRING_LIST,	"stringlist",		2,	CP_SHP,	ND_LIST	},
{	T_MIXED_LIST,	"mixed_list",		2,	NO_SHP,	ND_LIST	},
{	T_PRINT_LIST,	"print_list",		2,	NO_SHP,	ND_LIST	},
{	T_SCRIPT,	"script",		1,	NO_SHP,	ND_CALLF },
{	T_NAME_FUNC,	"namefunc",		1,	NO_SHP,	ND_NONE	},

{	T_LIT_DBL,	"lit_dbl",		0,	PT_SHP,	ND_DBL	},
{	T_LIT_INT,	"lit_int",		0,	PT_SHP,	ND_INT	},
{	T_LIST_OBJ,	"listobj",		1,	CP_SHP,	ND_NONE	},
{	T_COMP_OBJ,	"compobj",		1,	CP_SHP,	ND_NONE	},
{	T_DYN_OBJ,	"dynamic_object",	0,	PT_SHP,	ND_STRING	},
{	T_STATIC_OBJ,	"static_object",	0,	PT_SHP,	ND_STRING	},
{	T_SCALAR_VAR,	"scalar_var",		0,	PT_SHP,	ND_STRING	},
{	T_POINTER,	"pointer",		0,	PT_SHP,	ND_STRING	},
{	T_STR_PTR,	"str_ptr",		0,	PT_SHP,	ND_STRING	},
{	T_REFERENCE,	"reference",		1,	PT_SHP,	ND_NONE	},
{	T_DEREFERENCE,	"dereference",		1,	PT_SHP,	ND_NONE	},
{	T_OBJ_LOOKUP,	"obj_of",		1,	PT_SHP,	ND_NONE	},
{	T_SQUARE_SUBSCR,"square_subscript",	2,	CP_SHP,	ND_NONE	},
{	T_CURLY_SUBSCR,	"curly_subscript",	2,	CP_SHP,	ND_NONE	},
{	T_SUBVEC,	"subvec",		3,	CP_SHP,	ND_NONE	},
{	T_CSUBVEC,	"csubvec",		3,	CP_SHP,	ND_NONE	},
{	T_SUBSAMP,	"subsample",		2,	CP_SHP,	ND_NONE },
{	T_CSUBSAMP,	"csubsample",		2,	CP_SHP,	ND_NONE },
{	T_REAL_PART,	"real_part",		1,	CP_SHP,	ND_NONE	},
{	T_IMAG_PART,	"imag_part",		1,	CP_SHP,	ND_NONE	},
{	T_BADNAME,	"bad_name",		0,	NO_SHP,	ND_NONE	},
{	T_LOAD,		"load",			1,	PT_SHP,	ND_NONE	},
{	T_SAVE,		"save",			1,	NO_SHP,	ND_NONE	},
{	T_FILETYPE,	"filetype",		1,	NO_SHP,	ND_NONE	},

{	T_POSTDEC,	"postdec",		1,	PT_SHP,	ND_NONE	},
{	T_PREDEC,	"predec",		1,	PT_SHP,	ND_NONE	},
{	T_POSTINC,	"postinc",		1,	PT_SHP,	ND_NONE	},
{	T_PREINC,	"preinc",		1,	PT_SHP,	ND_NONE	},
{	T_ASSIGN,	"assign",		2,	PT_SHP,	ND_NONE	},
{	T_DIM_ASSIGN,	"dim_assign",		2,	CP_SHP,	ND_NONE	},
{	T_SET_STR,	"set_str",		2,	PT_SHP,	ND_NONE	},
{	T_SET_PTR,	"set_ptr",		2,	PT_SHP,	ND_NONE	},
{	T_ADVISE,	"advise",		1,	NO_SHP,	ND_NONE	},
{	T_WARN,		"warn",			1,	NO_SHP,	ND_NONE	},
{	T_EXP_PRINT,	"exp_print",		1,	NO_SHP,	ND_NONE	},
{	T_EXPR_LIST,	"expr_list",		2,	NO_SHP,	ND_LIST	},
{	T_INFO,		"info",			1,	NO_SHP,	ND_NONE	},
{	T_DISPLAY,	"display",		1,	NO_SHP,	ND_NONE	},
{	T_IFTHEN,	"if-then",		3,	NO_SHP,	ND_NONE	},

{	T_PLUS,		"plus",			2,	PT_SHP,	ND_FUNC	},
{	T_MINUS,	"minus",		2,	PT_SHP,	ND_FUNC	},
{	T_TIMES,	"times",		2,	PT_SHP,	ND_FUNC	},
{	T_DIVIDE,	"div",			2,	PT_SHP,	ND_FUNC	},
{	T_UMINUS,	"uminus",		1,	PT_SHP,	ND_FUNC	},
{	T_RECIP,	"recip",		1,	PT_SHP,	ND_FUNC	},

{	T_MODULO,	"modulo",		2,	PT_SHP,	ND_FUNC	},
{	T_BITAND,	"bitand",		2,	PT_SHP,	ND_FUNC	},
{	T_BITOR,	"bitor",		2,	PT_SHP,	ND_FUNC	},

{	T_BITXOR,	"bitxor",		2,	PT_SHP,	ND_FUNC	},
{	T_BITLSHIFT,	"bitlshift",		2,	PT_SHP,	ND_FUNC	},
{	T_BITRSHIFT,	"bitrshift",		2,	PT_SHP,	ND_FUNC	},
{	T_BITCOMP,	"bitcomp",		1,	PT_SHP,	ND_FUNC	},
{	T_VCOMP,	"vcomp",		1,	PT_SHP,	ND_FUNC	},

{	T_BOOL_AND,	"bool_and",		2,	CP_SHP,	ND_FUNC	},
{	T_BOOL_OR,	"bool_or",		2,	CP_SHP,	ND_FUNC	},
{	T_BOOL_XOR,	"bool_xor",		2,	CP_SHP,	ND_FUNC	},
{	T_BOOL_EQ,	"bool_eq",		2,	CP_SHP,	ND_FUNC	},
{	T_BOOL_PTREQ,	"bool_ptreq",		2,	PT_SHP,	ND_NONE	},
{	T_BOOL_NE,	"bool_ne",		2,	CP_SHP,	ND_FUNC	},
{	T_BOOL_GT,	"bool_gt",		2,	CP_SHP,	ND_FUNC	},
{	T_BOOL_LT,	"bool_lt",		2,	CP_SHP,	ND_FUNC	},
{	T_BOOL_GE,	"bool_ge",		2,	CP_SHP,	ND_FUNC	},
{	T_BOOL_LE,	"bool_le",		2,	CP_SHP,	ND_FUNC	},
{	T_BOOL_NOT,	"not",			1,	CP_SHP,	ND_FUNC	},

{	T_MATH0_FN,	"math0_fn",		0,	PT_SHP,	ND_FUNC	},
{	T_MATH1_FN,	"math1_fn",		1,	PT_SHP,	ND_FUNC	},
{	T_MATH2_FN,	"math2_fn",		2,	PT_SHP,	ND_FUNC	},
{	T_MATH0_VFN,	"math0_vfn",		0,	PT_SHP,	ND_FUNC	},
{	T_MATH1_VFN,	"math1_vfn",		1,	PT_SHP,	ND_FUNC	},
{	T_MATH2_VFN,	"math2_vfn",		2,	PT_SHP,	ND_FUNC	},
{	T_MATH2_VSFN,	"math2_vsfn",		2,	PT_SHP,	ND_FUNC	},

{	T_INT1_FN,	"int1_fn",		1,	PT_SHP,	ND_FUNC	},
{	T_INT1_VFN,	"int1_vfn",		1,	PT_SHP,	ND_FUNC	},

{	T_SIZE_FN,	"size_fn",		1,	PT_SHP,	ND_FUNC	},
{	T_DATA_FN,	"data_fn",		1,	PT_SHP,	ND_FUNC	},
{	T_STR1_FN,	"str1_fn",		1,	PT_SHP,	ND_FUNC	},
{	T_STR2_FN,	"str2_fn",		2,	PT_SHP,	ND_FUNC	},
{	T_STR3_FN,	"str3_fn",		3,	PT_SHP,	ND_FUNC	},
{	T_STRV_FN,	"strv_fn",		1,	PT_SHP,	ND_FUNC	},
{	T_CHAR_FN,	"char_fn",		1,	PT_SHP,	ND_FUNC	},
{	T_CHAR_VFN,	"char_vfn",		1,	PT_SHP,	ND_FUNC	},
{	T_MISC_FN,	"misc_fn",		1,	PT_SHP,	ND_FUNC	},

{	T_MINVAL,	"minval",		1,	CP_SHP,	ND_FUNC	},
{	T_MAXVAL,	"maxval",		1,	CP_SHP,	ND_FUNC	},
{	T_SUM,		"sum",			1,	CP_SHP,	ND_FUNC	},
{	T_SCALMAX,	"scalmax",		2,	PT_SHP,	ND_FUNC	},
{	T_SCALMIN,	"scalmin",		2,	PT_SHP,	ND_FUNC	},

{	T_EQUIVALENCE,	"equivalence",		2,	CP_SHP,	ND_NONE	},
{	T_TRANSPOSE,	"transpose",		1,	CP_SHP,	ND_NONE	},
{	T_INNER,	"inner",		2,	CP_SHP,	ND_NONE	},

{	T_RAMP,		"ramp",			3,	PT_SHP,	ND_NONE	},
{	T_LOOKUP,	"lookup",		0,	PT_SHP,	ND_NULL	},

{	T_WRAP,		"wrap",			1,	PT_SHP,	ND_NONE	},
{	T_SCROLL,	"scroll",		3,	PT_SHP,	ND_NONE	},
{	T_DFT,		"dft",			1,	PT_SHP,	ND_NONE	},
{	T_IDFT,		"idft",			1,	PT_SHP,	ND_NONE	},
{	T_RDFT,		"rdft",			1,	CP_SHP,	ND_SIZE_CHANGE	},
{	T_RIDFT,	"ridft",		1,	CP_SHP,	ND_SIZE_CHANGE	},

{	T_FUNCPTR,	"funcptr",		0,	PT_SHP,	ND_STRING	},
{	T_FUNCPTR_DECL,	"funcptr_decl",		1,	CP_SHP,	ND_DECL	},
{	T_FUNCREF,	"funcref",		0,	NO_SHP,	ND_CALLF },
{	T_SET_FUNCPTR,	"set_funcptr",		2,	PT_SHP,	ND_NULL	},
{	T_INDIR_CALL,	"indir_call",		2,	PT_SHP,	ND_NONE /* should be ND_CALLF??? BUG? */	},
{	T_UNDEF,	"undef",		0,	PT_SHP,	ND_STRING	},
{	T_TYPECAST,	"typecast",		1,	CP_SHP,	ND_CAST	},
/* Should these own their own shape or not??? */
{	T_ENLARGE,	"enlarge",		1,	CP_SHP,	ND_SIZE_CHANGE	},
{	T_REDUCE,	"reduce",		1,	CP_SHP,	ND_SIZE_CHANGE	},

{	T_MAX_TIMES,	"max_times",		3,	PT_SHP,	ND_FUNC	},
{	T_MIN_TIMES,	"min_times",		3,	PT_SHP,	ND_FUNC	},
{	T_MAX_INDEX,	"max_index",		1,	PT_SHP,	ND_FUNC	},
{	T_MIN_INDEX,	"min_index",		1,	PT_SHP,	ND_FUNC	},

{	T_CONJ,		"conj",			1,	PT_SHP,	ND_NONE	},

{	T_DILATE,	"dilate",		1,	PT_SHP,	ND_NONE	},
{	T_ERODE,	"erode",		1,	PT_SHP,	ND_NONE	},
{	T_FILL,		"fill",			3,	PT_SHP,	ND_NONE	},

{	T_CLR_OPT_PARAMS,"clr_opt_prms",	0,	PT_SHP,	ND_NONE	},
{	T_ADD_OPT_PARAM,"add_opt_prm",		3,	PT_SHP,	ND_NONE	},
{	T_OPTIMIZE,	"optimize",		0,	CP_SHP,	ND_CALLF	},

{	T_PERFORM,	"perform",		1,	PT_SHP,	ND_NULL	},
{	T_WHILE,	"while",		2,	NO_SHP,	ND_NONE	},
{	T_UNTIL,	"until",		2,	NO_SHP,	ND_NONE	},
{	T_FOR,		"for",			3,	NO_SHP,	ND_NONE	},
{	T_DO_WHILE,	"do_while",		2,	NO_SHP,	ND_NONE	},
{	T_DO_UNTIL,	"do_until",		2,	NO_SHP,	ND_NONE	},
{	T_CONTINUE,	"continue",		0,	NO_SHP,	ND_NONE	},
{	T_SWITCH,	"switch",		2,	NO_SHP,	ND_NONE	},
{	T_CASE,		"case",			1,	NO_SHP,	ND_NONE	},
{	T_CASE_STAT,	"case_stat",		2,	NO_SHP,	ND_NONE	},
{	T_SWITCH_LIST,	"switch_list",		2,	NO_SHP,	ND_LIST	},
{	T_CASE_LIST,	"case_list",		2,	NO_SHP,	ND_LIST	},
{	T_DEFAULT,	"default",		0,	NO_SHP,	ND_NONE	},
{	T_BREAK,	"break",		0,	NO_SHP,	ND_NONE	},
{	T_GO_FWD,	"goto_fwd",		0,	NO_SHP,	ND_STRING	},
{	T_GO_BACK,	"goto_back",		0,	NO_SHP,	ND_STRING	},
{	T_LABEL,	"label",		0,	NO_SHP,	ND_STRING	},

{	T_FILE_EXISTS,	"file_exists",		1,	PT_SHP,	ND_NONE	},
{	T_STRCPY,	"strcpy",		2,	PT_SHP,	ND_NONE	},
{	T_STRCAT,	"strcat",		2,	PT_SHP,	ND_NONE	},
{	T_OUTPUT_FILE,	"set_output_file",	1,	PT_SHP,	ND_NONE	},
{	T_PROTO,	"prototype",		1,	PT_SHP,	ND_STRING	},
{	T_FIX_SIZE,	"fix_size",		1,	PT_SHP,	ND_NONE	},

{	T_VV_S_CONDASS,	"vv_s_condass",		3,	CP_SHP,	ND_BMAP	},
{	T_VS_S_CONDASS,	"vs_s_condass",		3,	CP_SHP,	ND_BMAP	},
{	T_SS_S_CONDASS,	"ss_s_condass",		3,	CP_SHP,	ND_NONE	},
{	T_VV_B_CONDASS,	"vv_b_condass",		3,	CP_SHP,	ND_BMAP	},
{	T_VS_B_CONDASS,	"vs_b_condass",		3,	CP_SHP,	ND_BMAP	},
{	T_SS_B_CONDASS,	"ss_b_condass",		3,	CP_SHP,	ND_BMAP	},

{	T_VV_VV_CONDASS,"vv_vv_condass",	4,	CP_SHP,	ND_NONE	},
{	T_VS_VV_CONDASS,"vs_vv_condass",	4,	CP_SHP,	ND_NONE	},
{	T_SS_VV_CONDASS,"ss_vv_condass",	4,	CP_SHP,	ND_NONE	},
{	T_VV_VS_CONDASS,"vv_vs_condass",	4,	CP_SHP,	ND_NONE	},
{	T_VS_VS_CONDASS,"vs_vs_condass",	4,	CP_SHP,	ND_NONE	},
{	T_SS_VS_CONDASS,"ss_vs_condass",	4,	CP_SHP,	ND_NONE	},
{	T_VV_SS_CONDASS,"vv_ss_condass",	4,	CP_SHP,	ND_NONE	},
{	T_VS_SS_CONDASS,"vs_ss_condass",	4,	CP_SHP,	ND_NONE	},
{	T_SS_SS_CONDASS,"ss_ss_condass",	4,	CP_SHP,	ND_NONE	},

/* matlab codes */
{	T_I,		"i",			0,	PT_SHP, ND_NONE	},
{	T_MFILE,	"mfile",		0,	NO_SHP,	ND_NONE	},
{	T_CLEAR,	"clear",		0,	NO_SHP,	ND_NONE	},
{	T_DRAWNOW,	"drawnow",		0,	NO_SHP,	ND_NONE	},
{	T_CLF,		"clf",			0,	NO_SHP,	ND_NONE	},
{	T_MCMD,		"mcmd",			0,	NO_SHP, ND_NONE	},
{	T_ROW,		"row",			2,	CP_SHP, ND_NONE	},
/*
{	T_ROWLIST,	"rowlist",		2,	CP_SHP,	ND_NONE	},
*/
{	T_SUBSCRIPT1,	"subscr1",		2,	CP_SHP,	ND_NONE	},
{	T_SUBMTRX,	"submtrx",		3,	CP_SHP,	ND_NONE	},
{	T_INDEX_SPEC,	"index_spec",		2,	NO_SHP,	ND_NONE	},
{	T_INDEX_LIST,	"index_list",		2,	NO_SHP,	ND_NONE	},
{	T_SSCANF,	"sscanf",		0,	PT_SHP,	ND_NONE	},
{	T_NOP,		"nop",			0,	NO_SHP,	ND_NONE	},
{	T_MLFUNC,	"mlfunc",		3,	NO_SHP,	ND_NONE	},
{	T_RET_LIST,	"retlist",		2,	CP_SHP,	ND_NONE	},
{	T_RANGE,	"range",		3,	CP_SHP,	ND_NONE	},
{	T_RANGE2,	"range2",		2,	CP_SHP,	ND_NONE	},
{	T_NULLVEC,	"nullvec",		0,	NO_SHP,	ND_NONE	},
{	T_STRUCT,	"struct",		1,	PT_SHP,	ND_NONE	},
{	T_FIRST_INDEX,	"first_index",		1,	NO_SHP,	ND_NONE	},
{	T_LAST_INDEX,	"last_index",		1,	NO_SHP,	ND_NONE	},
{	T_ENTIRE_RANGE,	"entire_range",		0,	NO_SHP,	ND_NONE	},
{	T_OBJ_LIST,	"obj_list",		2,	NO_SHP,	ND_NONE	},
{	T_GLOBAL,	"global",		1,	NO_SHP,	ND_NONE	},
{	T_FIND,		"find",			1,	CP_SHP,	ND_NONE	},
{	T_POWER,	"power",		2,	PT_SHP,	ND_FUNC	},
{	T_FVSPOW,	"vspow",		2,	PT_SHP,	ND_FUNC	},
{	T_FVSPOW2,	"vspow2",		2,	PT_SHP,	ND_FUNC	},
{	T_FVPOW,	"vpow",			2,	PT_SHP,	ND_FUNC	},
{	T_VV_FUNC,	"vv_func",		2,	CP_SHP,	ND_FUNC },
{	T_VS_FUNC,	"vs_func",		2,	PT_SHP,	ND_FUNC },
{	T_ROW_LIST,	"row_list",		2,	CP_SHP,	ND_LIST	},
{	T_COMP_LIST,	"comp_list",		2,	CP_SHP,	ND_LIST	},

{	T_END,		"end",			0,	PT_SHP,	ND_NONE	}
};

static int tnt_cmp(const void *vp1, const void *vp2)
{
	const Tree_Node_Type *tntp1, *tntp2;

	tntp1=(const Tree_Node_Type *)vp1;
	tntp2=(const Tree_Node_Type *)vp2;
	if( tntp1->tnt_code < tntp2->tnt_code ) return(-1);
	else if( tntp1->tnt_code > tntp2->tnt_code ) return(1);
	else return(0);
}

void sort_tree_tbl()
{
	qsort(tnt_tbl,N_TREE_CODES,sizeof(Tree_Node_Type),tnt_cmp);

#ifdef CAUTIOUS
	{
		int i;

		for(i=0;i<N_TREE_CODES;i++){
			assert( tnt_tbl[i].tnt_code == i );
		}
	}
#endif /* CAUTIOUS */
}

