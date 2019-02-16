#include "quip_prot.h"
#include "string_ref.h"

struct vector_parser_data {
	void *		vpd_top_enp;	// really Vec_Expr_Node
	void *		vpd_last_enp;	// really Vec_Expr_Node
	const char *	vpd_yy_cp;
	int		vpd_expr_level;
	int		vpd_last_line_num;
	int		vpd_parser_line_num;
	String_Buf *	vpd_yy_last_line;
	String_Buf *	vpd_yy_input_line;
	String_Buf *	vpd_expr_string;	// for reading the next word?
	String_Buf *	vpd_yy_word_buf;
	int		vpd_semi_seen;		// boolean flag...
	int		vpd_end_seen;		// boolean flag?
						// BUG - combine flags into single flag word?
	int		vpd_edepth;		// same as expr_level???
	const char *	vpd_curr_string;
	double		vpd_final;
	String_Ref *	vpd_curr_infile;
	List *		vpd_subroutine_context_stack;
};

#define	VPD_TOP_ENP(vpd_p)		(vpd_p)->vpd_top_enp
#define	VPD_LAST_ENP(vpd_p)		(vpd_p)->vpd_last_enp
#define	VPD_END_SEEN(vpd_p)		(vpd_p)->vpd_end_seen
#define	VPD_YY_CP(vpd_p)		(vpd_p)->vpd_yy_cp
#define	VPD_EXPR_LEVEL(vpd_p)		(vpd_p)->vpd_expr_level
#define	VPD_LAST_LINE_NUM(vpd_p)	(vpd_p)->vpd_last_line_num
#define	VPD_PARSER_LINE_NUM(vpd_p)	(vpd_p)->vpd_parser_line_num
#define	VPD_YY_LAST_LINE(vpd_p)		(vpd_p)->vpd_yy_last_line
#define	VPD_YY_INPUT_LINE(vpd_p)	(vpd_p)->vpd_yy_input_line
#define	VPD_YY_WORD_BUF(vpd_p)		(vpd_p)->vpd_yy_word_buf
#define	VPD_SEMI_SEEN(vpd_p)		(vpd_p)->vpd_semi_seen
#define	VPD_EXPR_STRING(vpd_p)		(vpd_p)->vpd_expr_string
#define	VPD_EDEPTH(vpd_p)		(vpd_p)->vpd_edepth
#define	VPD_CURR_STRING(vpd_p)		(vpd_p)->vpd_curr_string
#define	VPD_FINAL(vpd_p)		(vpd_p)->vpd_final
#define	VPD_CURR_INFILE(vpd_p)		(vpd_p)->vpd_curr_infile
#define VPD_SUBRT_CTX_STACK(vpd_p)	(vpd_p)->vpd_subroutine_context_stack


#define	SET_VPD_TOP_ENP(vpd_p,v)		(vpd_p)->vpd_top_enp = v
#define	SET_VPD_LAST_ENP(vpd_p,v)		(vpd_p)->vpd_last_enp = v
#define	SET_VPD_END_SEEN(vpd_p,v)		(vpd_p)->vpd_end_seen = v
#define	SET_VPD_YY_CP(vpd_p,v)			(vpd_p)->vpd_yy_cp = v
#define	SET_VPD_EXPR_LEVEL(vpd_p,v)		(vpd_p)->vpd_expr_level = v
#define	SET_VPD_LAST_LINE_NUM(vpd_p,v)		(vpd_p)->vpd_last_line_num = v
#define	SET_VPD_PARSER_LINE_NUM(vpd_p,v)	(vpd_p)->vpd_parser_line_num = v
#define	SET_VPD_YY_LAST_LINE(vpd_p,v)		(vpd_p)->vpd_yy_last_line = v
#define	SET_VPD_YY_WORD_BUF(vpd_p,v)		(vpd_p)->vpd_yy_word_buf = v
#define	SET_VPD_YY_INPUT_LINE(vpd_p,v)		(vpd_p)->vpd_yy_input_line = v
#define	SET_VPD_SEMI_SEEN(vpd_p,v)		(vpd_p)->vpd_semi_seen = v
#define	SET_VPD_EXPR_STRING(vpd_p,v)		(vpd_p)->vpd_expr_string = v
#define	SET_VPD_EDEPTH(vpd_p,v)			(vpd_p)->vpd_edepth = v
#define	SET_VPD_CURR_STRING(vpd_p,v)		(vpd_p)->vpd_curr_string = v
#define	SET_VPD_FINAL(vpd_p,v)			(vpd_p)->vpd_final = v
#define	SET_VPD_CURR_INFILE(vpd_p,v)		(vpd_p)->vpd_curr_infile = v
#define SET_VPD_SUBRT_CTX_STACK(vpd_p,v)	(vpd_p)->vpd_subroutine_context_stack = v

#define CURR_INFILE			VPD_CURR_INFILE( THIS_VPD )
#define SET_CURR_INFILE(v)		SET_VPD_CURR_INFILE( THIS_VPD, v )


//#define QS_VECTOR_PARSER_DATA_STACK(qsp)	(qsp)->qs_vector_parser_data_stack
//#define QS_VECTOR_PARSER_DATA_FREELIST(qsp)	(qsp)->qs_vector_parser_data_freelist
//#define SET_QS_VECTOR_PARSER_DATA(qsp,d)	(qsp)->qs_vector_parser_data = d
//#define SET_QS_VECTOR_PARSER_DATA_STACK(qsp,v)	(qsp)->qs_vector_parser_data_stack = v
//#define SET_QS_VECTOR_PARSER_DATA_FREELIST(qsp,v)	(qsp)->qs_vector_parser_data_freelist = v

#define LAST_NODE		VPD_LAST_ENP( THIS_VPD )
#define END_SEEN		VPD_END_SEEN( THIS_VPD )
#define YY_CP			VPD_YY_CP( THIS_VPD )
#define EXPR_LEVEL		VPD_EXPR_LEVEL( THIS_VPD )
#define LAST_LINE_NUM		VPD_LAST_LINE_NUM( THIS_VPD )
#define PARSER_LINE_NUM		VPD_PARSER_LINE_NUM( THIS_VPD )
#define YY_LAST_LINE		VPD_YY_LAST_LINE( THIS_VPD )
#define YY_INPUT_LINE		VPD_YY_INPUT_LINE( THIS_VPD )
#define YY_WORD_BUF		VPD_YY_WORD_BUF( THIS_VPD )
#define SEMI_SEEN		VPD_SEMI_SEEN( THIS_VPD )
#define FINAL			VPD_FINAL( THIS_VPD )
#define SUBRT_CTX_STACK		VPD_SUBRT_CTX_STACK( THIS_VPD )

#define SET_LAST_NODE(v)		SET_VPD_LAST_ENP( THIS_VPD, v )
#define SET_END_SEEN(v)			SET_VPD_END_SEEN( THIS_VPD, v )
#define SET_YY_CP(v)			SET_VPD_YY_CP( THIS_VPD, v )
#define SET_EXPR_LEVEL(v)		SET_VPD_EXPR_LEVEL( THIS_VPD, v )
#define SET_LAST_LINE_NUM(v)		SET_VPD_LAST_LINE_NUM( THIS_VPD, v )
#define SET_PARSER_LINE_NUM(v)		SET_VPD_PARSER_LINE_NUM( THIS_VPD, v )
#define SET_YY_LAST_LINE(v)		SET_VPD_YY_LAST_LINE( THIS_VPD, v )
#define SET_YY_INPUT_LINE(v)		SET_VPD_YY_INPUT_LINE( THIS_VPD, v )
#define SET_SEMI_SEEN(v)		SET_VPD_SEMI_SEEN( THIS_VPD, v )
#define SET_FINAL(v)			SET_VPD_FINAL( THIS_VPD, v )
#define SET_SUBRT_CTX_STACK(v)		SET_VPD_SUBRT_CTX_STACK( THIS_VPD, v )

#define QS_VECTOR_PARSER_DATA_FREELIST	qs_vector_parser_data_freelist(SINGLE_QSP_ARG)
#define QS_VECTOR_PARSER_DATA_STACK	qs_vector_parser_data_stack(SINGLE_QSP_ARG)
#define THIS_VPD		(qs_vector_parser_data())
//#define QS_VECTOR_PARSER_DATA		qs_vector_parser_data()

