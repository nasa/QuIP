#ifndef _ASCII_FMTS_H
#define _ASCII_FMTS_H

#include "shape_bits.h"

/* some codes used for printing.
 * These used to be in data_obj.h, and there was a global var that held the current value.
 * To have a per-thread value, it has to be defined here...
 */

typedef enum {
	IN_FMT_INT,
	IN_FMT_FLT,
	IN_FMT_STR,
	IN_FMT_LIT,
	N_INPUT_FORMAT_TYPES
} Input_Format_Type_Code;

struct input_format_spec;

struct input_format_type {
	const char *		name;
	Input_Format_Type_Code	type_code;
	void (*display_format)(QSP_ARG_DECL  struct input_format_spec *);
	void (*release)(struct input_format_spec *);
	void (*consume)(QSP_ARG_DECL  Precision *prec_p);
	int (*read_long)(QSP_ARG_DECL  int64_t *result, const char *pmpt, Input_Format_Spec *fmt_p);
	int (*read_double)(QSP_ARG_DECL  double *result, const char *pmpt, Input_Format_Spec *fmt_p);
	int (*read_string)(QSP_ARG_DECL  const char **result, const char *pmpt, Input_Format_Spec *fmt_p);
};

struct input_format_spec {
	//Input_Format_Type	fmt_type;
	struct input_format_type *	fmt_type;
	const char *			fmt_litstr;	// could be part of a union
};

//#define MAX_FORMAT_FIELDS	64

typedef enum {
	// lots of different ways to print integers...
	FMT_DECIMAL,
	FMT_HEX,
	FMT_OCTAL,
	FMT_UDECIMAL,
	FMT_POSTSCRIPT,
	N_INT_PRINT_FORMATS		/* must be last */
} Integer_Output_Fmt_Code;

typedef struct output_number {
	union {	
		double d;
		int64_t l;
	} on_u;
} Output_Number;

struct integer_output_fmt {
	Item			iof_item;
	Integer_Output_Fmt_Code	iof_code;
	const char *		iof_padded_fmt_str;	/* printf format */
	const char *		iof_plain_fmt_str;	/* printf format */
	void			(*iof_fmt_string_func)(QSP_ARG_DECL  char *,Scalar_Value *);
	void			(*iof_fmt_char_func)(QSP_ARG_DECL  char *,Scalar_Value *);
	void			(*iof_fmt_byte_func)(QSP_ARG_DECL  char *,Scalar_Value *);
	void			(*iof_fmt_u_byte_func)(QSP_ARG_DECL  char *,Scalar_Value *);
	void			(*iof_fmt_short_func)(QSP_ARG_DECL  char *,Scalar_Value *);
	void			(*iof_fmt_u_short_func)(QSP_ARG_DECL  char *,Scalar_Value *);
	void			(*iof_fmt_int_func)(QSP_ARG_DECL  char *,Scalar_Value *);
	void			(*iof_fmt_u_int_func)(QSP_ARG_DECL  char *,Scalar_Value *);
	void			(*iof_fmt_long_func)(QSP_ARG_DECL  char *,Scalar_Value *);
	void			(*iof_fmt_u_long_func)(QSP_ARG_DECL  char *,Scalar_Value *);
};

extern void set_format_string(Integer_Output_Fmt *, const char *);
extern void _set_integer_print_fmt(QSP_ARG_DECL  Integer_Output_Fmt *);
#define set_integer_print_fmt(iof_p) _set_integer_print_fmt(QSP_ARG  iof_p)

ITEM_INTERFACE_PROTOTYPES(Integer_Output_Fmt,int_out_fmt)
#define init_int_out_fmts()	_init_int_out_fmts(SINGLE_QSP_ARG)
#define pick_int_out_fmt(s)	_pick_int_out_fmt(QSP_ARG  s)
#define new_int_out_fmt(s)	_new_int_out_fmt(QSP_ARG  s)
#define int_out_fmt_of(s)	_int_out_fmt_of(QSP_ARG  s)

// Is this all for ascii input?

typedef struct dobj_ascii_info {
	// output control variables
	int			dai_pad_flag;		// there is no way to set this - BUG!
							// defaults to one for evenly spaced columns on printout
	int			dai_ascii_warned;
	Integer_Output_Fmt *	dai_output_int_fmt_p;
	dimension_t		dai_dobj_max_per_line;
	int			dai_ret_dim;
	int			dai_min_field_width;
	int			dai_display_precision;

	// input control variables
	int			dai_ascii_level;
	dimension_t		dai_dobj_n_gotten;

	// not sure
	struct data_obj *	dai_ascii_data_dp;

	char 			dai_padded_flt_fmt_str[16];

	List *			dai_fmt_lp;
	Node *			dai_curr_fmt_np;
} Dobj_Ascii_Info;

#define	curr_output_int_fmt_p	THIS_QSP->qs_dai_p->dai_output_int_fmt_p
#define padded_flt_fmt_str	THIS_QSP->qs_dai_p->dai_padded_flt_fmt_str
#define plain_flt_fmt_str	THIS_QSP->qs_dai_p->dai_plain_flt_fmt_str
#define	ascii_warned		THIS_QSP->qs_dai_p->dai_ascii_warned
#define	ascii_data_dp		THIS_QSP->qs_dai_p->dai_ascii_data_dp
#define	dobj_n_gotten		THIS_QSP->qs_dai_p->dai_dobj_n_gotten
#define	dobj_max_per_line	THIS_QSP->qs_dai_p->dai_dobj_max_per_line
#define	ret_dim			THIS_QSP->qs_dai_p->dai_ret_dim
#define	pad_ffmtstr		THIS_QSP->qs_dai_p->dai_pad_ffmtstr
#define	min_field_width		THIS_QSP->qs_dai_p->dai_min_field_width
#define	display_precision	THIS_QSP->qs_dai_p->dai_display_precision
#define	pad_flag		THIS_QSP->qs_dai_p->dai_pad_flag

#define IS_FIRST_FORMAT		( CURRENT_FORMAT_NODE != QLIST_HEAD(INPUT_FORMAT_LIST) )
#define INPUT_FORMAT_LIST	(THIS_QSP->qs_dai_p->dai_fmt_lp)
#define CURRENT_FORMAT_NODE	(THIS_QSP->qs_dai_p->dai_curr_fmt_np)
#define CURRENT_FORMAT		((Input_Format_Spec *)NODE_DATA(CURRENT_FORMAT_NODE))
#define HAS_FORMAT_LIST		(INPUT_FORMAT_LIST != NULL )
#define FIRST_INPUT_FORMAT_NODE	(QLIST_HEAD(INPUT_FORMAT_LIST))


//extern void set_integer_print_fmt(QSP_ARG_DECL  Integer_Output_Fmt_Code fmt_code );

#endif /* ! _ASCII_FMTS_H */

