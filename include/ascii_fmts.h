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
	IN_FMT_LIT
} Input_Format_Type;

typedef struct input_fmt_spec {
	Input_Format_Type	fmt_type;
	const char *		fmt_litstr;
} Input_Format_Spec;

#define MAX_FORMAT_FIELDS	64

typedef enum {
	FMT_DECIMAL,
	FMT_HEX,
	FMT_OCTAL,
	FMT_UDECIMAL,
	FMT_FLOAT,
	FMT_POSTSCRIPT,
	N_PRINT_FORMATS		/* must be last */
} Number_Fmt;


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
	Input_Format_Spec	dai_input_fmt[MAX_FORMAT_FIELDS];
} Dobj_Ascii_Info;

#define	ascii_input_fmt		THIS_QSP->qs_dai_p->dai_input_fmt
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

//extern void set_integer_print_fmt(QSP_ARG_DECL  Number_Fmt fmt_code );

#endif /* ! _ASCII_FMTS_H */

