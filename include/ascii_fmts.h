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
	void (*consume)(QSP_ARG_DECL  prec_t c);
	int (*read_long)(QSP_ARG_DECL  long *result, const char *pmpt, Input_Format_Spec *fmt_p);
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
	FMT_DECIMAL,
	FMT_HEX,
	FMT_OCTAL,
	FMT_UDECIMAL,
	FMT_FLOAT,
	FMT_POSTSCRIPT,
	N_PRINT_FORMATS		/* must be last */
} Number_Fmt;

// Is this all for ascii input?

typedef struct dobj_ascii_info {
	int			dai_padflag;
	struct data_obj *	dai_ascii_data_dp;
	int			dai_ascii_warned;
	int			dai_ascii_level;
	Number_Fmt		dai_the_fmt_code;
	int			dai_ret_dim;
	dimension_t		dai_dobj_n_gotten;
	dimension_t		dai_dobj_max_per_line;
	int			dai_min_field_width;
	int			dai_display_precision;
	const char *		dai_ascii_separator;
	const char *		dai_ffmtstr;	/* float format string */
	const char *		dai_ifmtstr;	/* integer format string */
	char 			dai_pad_ffmtstr[16];
//	Input_Format_Spec	dai_input_fmt[MAX_FORMAT_FIELDS];
	List *			dai_fmt_lp;
	Node *			dai_curr_fmt_np;
} Dobj_Ascii_Info;

//#define	ascii_input_fmt		THIS_QSP->qs_dai_p->dai_input_fmt
#define	ascii_warned		THIS_QSP->qs_dai_p->dai_ascii_warned
#define	ascii_data_dp		THIS_QSP->qs_dai_p->dai_ascii_data_dp
#define	dobj_n_gotten		THIS_QSP->qs_dai_p->dai_dobj_n_gotten
#define	ffmtstr			THIS_QSP->qs_dai_p->dai_ffmtstr
#define	ifmtstr			THIS_QSP->qs_dai_p->dai_ifmtstr
#define	dobj_max_per_line	THIS_QSP->qs_dai_p->dai_dobj_max_per_line
#define	ascii_separator		THIS_QSP->qs_dai_p->dai_ascii_separator
#define	ret_dim			THIS_QSP->qs_dai_p->dai_ret_dim
#define	pad_ffmtstr		THIS_QSP->qs_dai_p->dai_pad_ffmtstr
#define	min_field_width		THIS_QSP->qs_dai_p->dai_min_field_width
#define	display_precision	THIS_QSP->qs_dai_p->dai_display_precision
#define	padflag			THIS_QSP->qs_dai_p->dai_padflag

#define IS_FIRST_FORMAT		( CURRENT_FORMAT_NODE != QLIST_HEAD(INPUT_FORMAT_LIST) )
#define INPUT_FORMAT_LIST	(THIS_QSP->qs_dai_p->dai_fmt_lp)
#define CURRENT_FORMAT_NODE	(THIS_QSP->qs_dai_p->dai_curr_fmt_np)
#define CURRENT_FORMAT		((Input_Format_Spec *)NODE_DATA(CURRENT_FORMAT_NODE))
#define HAS_FORMAT_LIST		(INPUT_FORMAT_LIST != NULL )
#define FIRST_INPUT_FORMAT_NODE	(QLIST_HEAD(INPUT_FORMAT_LIST))


//extern void set_integer_print_fmt(QSP_ARG_DECL  Number_Fmt fmt_code );

#endif /* ! _ASCII_FMTS_H */

