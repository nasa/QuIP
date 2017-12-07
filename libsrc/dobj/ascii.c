#include "quip_config.h"

/* more generalized dimensions & subscripts added 8-92 jbm */

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#include "quip_prot.h"
#include "data_obj.h"
#include "ascii_fmts.h"
#include "dobj_private.h"
#include "query_stack.h"	// like to eliminate this dependency...
#include "veclib/obj_args.h"	// argset_prec

static void init_format_type_tbl(void);


/*
 * For printing out data with more than one number per line,
 * ret_dim is the index of the dimension where a return is forced
 */

#define DEFAULT_MAX_PER_LINE	12
#define ENFORCED_MAX_PER_LINE	64

/* We have different needs for formats;
 * When we are print a table of dobj values, we'd like to have fixed length fields,
 * but when we are constructing filenames or strings from object values we don't
 * want any space padding...
 */
#define DEFAULT_MIN_FIELD_WIDTH		10
#define DEFAULT_DISPLAY_PRECISION	6

#define PAD_FLT_FMT_STR		"%10.24g"
#define PAD_INT_FMT_STR 	"%10ld"
#define NOPAD_FLT_FMT_STR	"%g"
#define NOPAD_INT_FMT_STR	"%ld"
#define PS_INT_FMT_STR		"%x"

#define NORMAL_SEPARATOR	" "
#define POSTSCRIPT_SEPARATOR	""

/* get a pixel from the input stream, store data at *cp */

/* We often encounter situations where we have data stored with text, e.g.:
 *
 * FRAME 10 pup_x 10 pup_y 9
 * FRAME 11 pup_x 11 pup_y 8
 * ...
 *
 * The approach we have taken in the past is to use dm to pick out
 * the numeric fields...  We would like to add that capability here...
 * We allow the user to specify a format string:
 *
 * FRAME %d %s %f %s %d
 *
 * The rules are:
 * 	1) anything not starting with '%' is a literal string, read w/ nameof().
 *	2) literal strings must be matched, or a warning is printed
 *	3) %s strings are read w/ nameof() and discarded
 *	4) %d, %o, %x read w/ how_many()
 *	5) %f, %g read w/ how_much()
 */


// This global defines the format types used by all threads
// and is not modified once initialized.

static struct input_format_type input_format_type_tbl[N_INPUT_FORMAT_TYPES];

void init_dobj_ascii_info(QSP_ARG_DECL  Dobj_Ascii_Info *dai_p)
{
	dai_p->dai_padflag = 0;
	dai_p->dai_ascii_data_dp = NULL;
	dai_p->dai_ascii_warned = 0;
	dai_p->dai_dobj_max_per_line = DEFAULT_MAX_PER_LINE;
	dai_p->dai_min_field_width = DEFAULT_MIN_FIELD_WIDTH;
	dai_p->dai_display_precision = DEFAULT_DISPLAY_PRECISION;
	dai_p->dai_ascii_separator = NORMAL_SEPARATOR;
	dai_p->dai_ffmtstr = NOPAD_FLT_FMT_STR;
	dai_p->dai_ifmtstr = NOPAD_INT_FMT_STR;
	dai_p->dai_fmt_lp = NULL;
}

static void show_input_format(SINGLE_QSP_ARG_DECL)
{
	if( ! HAS_FORMAT_LIST ){
		advise("no input format specified");
		return;
	}

	CURRENT_FORMAT_NODE = FIRST_INPUT_FORMAT_NODE;
	while( CURRENT_FORMAT_NODE != NULL ){
		if( ! IS_FIRST_FORMAT )
			prt_msg_frag(" ");
		CURRENT_FORMAT->fmt_type->display_format(QSP_ARG  CURRENT_FORMAT);
		CURRENT_FORMAT_NODE = NODE_NEXT(CURRENT_FORMAT_NODE);
	}
	prt_msg("");
}

static void display_int_format(QSP_ARG_DECL  Input_Format_Spec *fmt_p)
{
	prt_msg_frag("%d");
}

static void display_literal_format(QSP_ARG_DECL  Input_Format_Spec *fmt_p)
{
	prt_msg_frag(fmt_p->fmt_litstr);
}

static void display_string_format(QSP_ARG_DECL  Input_Format_Spec *fmt_p)
{
	prt_msg_frag("%s");
}

static void display_float_format(QSP_ARG_DECL  Input_Format_Spec *fmt_p)
{
	prt_msg_frag("%f");
}

static /*Input_Format_Type*/ struct input_format_type *format_type_for_code(Input_Format_Type_Code c)
{
	static int inited=0;

	if( ! inited ){
		init_format_type_tbl();
		inited=1;
	}
	assert( c >= 0 && c < N_INPUT_FORMAT_TYPES );
	return( & input_format_type_tbl[c] );
}

static Input_Format_Spec *new_format_spec(Input_Format_Type_Code c)
{
	Input_Format_Spec *fmt_p;

	fmt_p = getbuf(sizeof(*fmt_p));
	fmt_p->fmt_type = format_type_for_code(c);
	fmt_p->fmt_litstr = NULL;
	return fmt_p;
}

static void add_format_for_code(QSP_ARG_DECL  Input_Format_Type_Code c)
{
	Input_Format_Spec *fmt_p;
	Node *np;

	fmt_p = new_format_spec(c);
	np = mk_node(fmt_p);
	if( ! HAS_FORMAT_LIST )
		INPUT_FORMAT_LIST = new_list();
	addTail( INPUT_FORMAT_LIST, np );
}

static void release_format_list(SINGLE_QSP_ARG_DECL)
{
	Node *np;
	Input_Format_Spec *fmt_p;

	np = QLIST_HEAD(INPUT_FORMAT_LIST);
	while(np!=NULL){
		fmt_p = (Input_Format_Spec *) NODE_DATA(np);
		(*(fmt_p->fmt_type->release))(fmt_p);
		np = NODE_NEXT(np);
	}
	dellist(INPUT_FORMAT_LIST);
	INPUT_FORMAT_LIST = NULL;
}

static void set_literal_format_string(QSP_ARG_DECL  char *s)
{
	Input_Format_Spec *fmt_p;
	Node *np;

	np = QLIST_TAIL(INPUT_FORMAT_LIST);
	assert(np!=NULL);
	fmt_p = NODE_DATA(np);
	assert( fmt_p != NULL );
	assert( fmt_p->fmt_type->type_code == IN_FMT_LIT );
	fmt_p->fmt_litstr = savestr(s);
}

#define MAX_LIT_STR_LEN	255

static void read_literal_format_string(QSP_ARG_DECL  const char **sptr)
{
	const char *s;
	int i=0;
	char lit_str[MAX_LIT_STR_LEN+1];

	s = *sptr;
	while( *s && !isspace(*s) && i < MAX_LIT_STR_LEN )
		lit_str[i++]=(*s++);
	lit_str[i] = 0;
	if( *s && !isspace(*s) )
		warn("literal string overflow in input format spec");

	*sptr = s;

	set_literal_format_string(QSP_ARG  lit_str);
}

static int process_format_char(QSP_ARG_DECL  const char **sptr )
{
	const char *s = *sptr;
	int c = *s;

	switch(c){
		case 0:
			return -1;

		case 'd':
		case 'x':
		case 'o':
		case 'i':
			add_format_for_code(QSP_ARG  IN_FMT_INT);
			break;
		case 'f':
		case 'g':
			add_format_for_code(QSP_ARG  IN_FMT_FLT);
			break;
		case 's':
			add_format_for_code(QSP_ARG  IN_FMT_STR);
			break;
	}
	s++;

	if( *s && !isspace(*s) ){
		sprintf(ERROR_STRING,
			"white space should follow format descriptor!?");
		warn(ERROR_STRING);
	}

	*sptr = s;
	return 0;
}

static int process_format_string_char(QSP_ARG_DECL  const char **sptr)
{
	const char *s = *sptr;
	int c;

	c = *s++;
	if( isspace(c) ){
		while( *s && isspace(*s) )
			s++;
	} else if( c == '%' ){
		if( process_format_char(QSP_ARG  &s) < 0 ){
			*sptr = s;
			return -1;
		}
	} else {	/* literal string */
		s--;
		add_format_for_code(QSP_ARG  IN_FMT_LIT);
		read_literal_format_string(QSP_ARG  &s);
	}
	*sptr = s;
	return 0;
}

void set_input_format_string( QSP_ARG_DECL  const char *s )
{
	const char *orig_str=s;
	//Input_Format_Spec *fmt_p;

	if( HAS_FORMAT_LIST ) release_format_list(SINGLE_QSP_ARG);
	/* parse the string */

	while( *s ){
		if( process_format_string_char(QSP_ARG  &s) < 0 ){
			sprintf(ERROR_STRING,
		"Poorly formed input format string \"%s\"" , orig_str);
			warn(ERROR_STRING);
			// BUG?  clean up by releasing format?
			return;
		}
	}
}

static void literal_format_release(Input_Format_Spec *fmt_p)
{
	rls_str( (char *) fmt_p->fmt_litstr );
}

static void default_format_release(Input_Format_Spec *fmt_p)
{ /* nop */ }

static int float_format_read_long(QSP_ARG_DECL  long *result, const char *pmpt, Input_Format_Spec *fmt_p)
{
	if( !ascii_warned ){
		sprintf(ERROR_STRING,
			"Float format data assigned to integer object %s!?",
			OBJ_NAME( ascii_data_dp) );
		warn(ERROR_STRING);
		ascii_warned=1;
	}

	*result = (long) HOW_MUCH(pmpt);
	return 1;
}

static int int_format_read_long(QSP_ARG_DECL  long *result, const char *pmpt, Input_Format_Spec *fmt_p)
{
	*result = HOW_MANY(pmpt);
	return 1;
}

static void consume_format_line(QSP_ARG_DECL  Precision *prec_p)
{
	Input_Format_Spec *fmt_p;

	do {
		if( QLEVEL != ASCII_LEVEL ){
			warn("Incomplete formatted input line!?");
			return;
		}
		fmt_p = CURRENT_FORMAT;
		fmt_p->fmt_type->consume(QSP_ARG  prec_p);
		lookahead_til(ASCII_LEVEL-1);
	} while( CURRENT_FORMAT_NODE != FIRST_INPUT_FORMAT_NODE );
}

static int is_numeric_prec( Precision *prec_p )
{
	return (*(prec_p->is_numeric_func))();
}

#define consume_literal_string(fmt_p) _consume_literal_string(QSP_ARG  fmt_p)

static inline void _consume_literal_string(QSP_ARG_DECL  Input_Format_Spec *fmt_p)
{
	const char *s;
	s=nameof(fmt_p->fmt_litstr);
	if( strcmp(s,fmt_p->fmt_litstr) ){
		sprintf(ERROR_STRING,
	"expected literal string \"%s\", saw string \"%s\"",
			fmt_p->fmt_litstr,s);
		warn(ERROR_STRING);
	}
}

#define RESET_INPUT_FORMAT_FIELD	CURRENT_FORMAT_NODE = QLIST_HEAD(INPUT_FORMAT_LIST);

#define advance_format() _advance_format(SINGLE_QSP_ARG)

static inline void _advance_format(SINGLE_QSP_ARG_DECL)
{
	assert(HAS_FORMAT_LIST);
	if( CURRENT_FORMAT_NODE == NULL )
		CURRENT_FORMAT_NODE = FIRST_INPUT_FORMAT_NODE;
	else {
		CURRENT_FORMAT_NODE = NODE_NEXT(CURRENT_FORMAT_NODE);
		if( CURRENT_FORMAT_NODE == NULL )
			CURRENT_FORMAT_NODE = FIRST_INPUT_FORMAT_NODE;
	}
	assert(CURRENT_FORMAT_NODE!=NULL);
}

/*
 * get the next number (int or float format),
 * skipping strings and literals
 *
 * 
 */

long next_input_int_with_format(QSP_ARG_DECL   const char *pmpt)
{
	long l=0;
	int done=0;

	do {
		assert(INPUT_FORMAT_LIST != NULL);
		assert(CURRENT_FORMAT_NODE != NULL);
		assert(CURRENT_FORMAT != NULL);
		done = CURRENT_FORMAT->fmt_type->read_long(QSP_ARG  &l, pmpt, CURRENT_FORMAT);
		advance_format();
	} while(!done);

	return(l);
}

/*
 * Read input fields until a number is encountered
 */

double next_input_flt_with_format(QSP_ARG_DECL  const char *pmpt)
{
	int done=0;
	double d=0.0;

	do {
		assert(CURRENT_FORMAT_NODE != NULL);
		assert(CURRENT_FORMAT != NULL);
		done = CURRENT_FORMAT->fmt_type->read_double(QSP_ARG  &d, pmpt, CURRENT_FORMAT);
		advance_format();
	} while(!done);

	return(d);
}

// The "consume" methods read and discard a field, UNLESS the precision code
// is appropriate for that format.

static void int_format_consume(QSP_ARG_DECL  Precision *prec_p)
{
	if( is_numeric_prec(prec_p) ){
		RESET_INPUT_FORMAT_FIELD
		return;
	} else {
		long l;
		l=HOW_MANY("dummy integer");
		advance_format();
	}
}

static void float_format_consume(QSP_ARG_DECL  Precision *prec_p)
{
	if( is_numeric_prec(prec_p) ){
		RESET_INPUT_FORMAT_FIELD
		return;
	} else {
		double d;
		d=HOW_MUCH("dummy float");
		advance_format();
	}
}

static void string_format_consume(QSP_ARG_DECL  Precision *prec_p)
{
	if( PREC_CODE(prec_p) == PREC_STR ){
		RESET_INPUT_FORMAT_FIELD
		return;
	} else {
		const char *s;
		s=nameof("dummy string");
		advance_format();
	}
}

static void literal_format_consume(QSP_ARG_DECL  Precision *prec_p)
{
	consume_literal_string(CURRENT_FORMAT);
	advance_format();
}


static void consume_variable_string(SINGLE_QSP_ARG_DECL)
{
	/*s=*/nameof("don't-care string");
}

static int string_format_read_long(QSP_ARG_DECL  long *result, const char *pmpt, Input_Format_Spec *fmt_p)
{
	consume_variable_string(SINGLE_QSP_ARG);
	return 0;
}

static int literal_format_read_long(QSP_ARG_DECL  long *result, const char *pmpt, Input_Format_Spec *fmt_p)
{
	consume_literal_string(fmt_p);
	return 0;
}

static int int_format_read_double(QSP_ARG_DECL  double *result, const char *pmpt, Input_Format_Spec *fmt_p)
{
	*result = HOW_MANY(pmpt);
	return 1;
}

static int float_format_read_double(QSP_ARG_DECL  double *result, const char *pmpt, Input_Format_Spec *fmt_p)
{
	*result = HOW_MUCH(pmpt);
	return 1;
}

static int string_format_read_double(QSP_ARG_DECL  double *result, const char *pmpt, Input_Format_Spec *fmt_p)
{
	consume_variable_string(SINGLE_QSP_ARG);
	return 0;
}

static int literal_format_read_double(QSP_ARG_DECL  double *result, const char *pmpt, Input_Format_Spec *fmt_p)
{
	consume_literal_string(fmt_p);
	return 0;
}

#define next_input_str(pmpt) _next_input_str(QSP_ARG  pmpt)

static const char * _next_input_str(QSP_ARG_DECL  const char *pmpt)
{
	const char *s = NULL;	/* quiet compiler to elim possibly used w/o init warning */
	int done=0;

	do {
		assert(CURRENT_FORMAT != NULL);
		done = CURRENT_FORMAT->fmt_type->read_string(QSP_ARG  &s, pmpt, CURRENT_FORMAT);
		advance_format();
	} while(!done);

	return(s);
}

static int int_format_read_string(QSP_ARG_DECL  const char **sptr, const char *pmpt, Input_Format_Spec *fmt_p)
{
	long l;
	l=HOW_MANY("dummy integer value");
	return 0;
}

static int float_format_read_string(QSP_ARG_DECL  const char **sptr, const char *pmpt, Input_Format_Spec *fmt_p)
{
	double d;
	d=HOW_MUCH("dummy float value");
	return 0;
}

static int string_format_read_string(QSP_ARG_DECL  const char **sptr, const char *pmpt, Input_Format_Spec *fmt_p)
{
	*sptr = nameof(pmpt);
	return 1;
}

static int literal_format_read_string(QSP_ARG_DECL  const char **sptr, const char *pmpt, Input_Format_Spec *fmt_p)
{
	consume_literal_string(fmt_p);
	return 0;
}

#define check_input_level() _check_input_level(SINGLE_QSP_ARG)

static inline int _check_input_level(SINGLE_QSP_ARG_DECL)
{
	if( QLEVEL != ASCII_LEVEL ){
		sprintf(ERROR_STRING,"check_input_level (ascii):  input depth is %d, expected %d!?",
			QLEVEL,ASCII_LEVEL);
		warn(ERROR_STRING);
		advise("premature end of data");
		sprintf(ERROR_STRING,"%d elements read so far",dobj_n_gotten);
		advise(ERROR_STRING);
		if( HAS_FORMAT_LIST ){
			prt_msg_frag("input_format:  ");
			show_input_format(SINGLE_QSP_ARG);
		}
		return(-1);
	}
	return 0;
}

static int get_a_string(QSP_ARG_DECL  Data_Obj *dp,char *datap,int dim)
{
	const char *s, *orig;
	char *t;
	dimension_t i;

	assert( dim >= 0 );

	if( check_input_level() < 0 ) return(-1);

	/* see if we need to look at the input format string */
	if( ! HAS_FORMAT_LIST )
		s = nameof("string data");
	else
		s = next_input_str("string data");

	t=datap;

	// Old (deleted) code assumed strings were in image rows...
	// This new code takes the dimension passed in the arg.
	i=0;
	orig=s;
	while( *s && i < DIMENSION(OBJ_TYPE_DIMS(dp),dim) ){
		*t = *s;
		t += INCREMENT(OBJ_TYPE_INCS(dp),dim);
		s++;
		i++;
	}
	if( i >= DIMENSION(OBJ_TYPE_DIMS(dp),dim) ){
		t -= INCREMENT(OBJ_TYPE_INCS(dp),dim);
		sprintf(ERROR_STRING,
"get_a_string:  input string (%ld chars) longer than data buffer (%ld chars)",
			(long)(strlen(orig)+1),
			(long)DIMENSION(OBJ_TYPE_DIMS(dp),dim));
		warn(ERROR_STRING);
	}
	*t = 0;	// add null terminator

	/* now lookahead to pop the file if it is empty */
	lookahead_til(ASCII_LEVEL-1);

	return(0);
} // get_a_string

#ifdef HAVE_ANY_GPU
int object_is_in_ram(QSP_ARG_DECL  Data_Obj *dp, const char *op_str)
{
	static Data_Obj *warned_dp=NULL;

	if( ! OBJ_IS_RAM(dp) ){
		if( dp != warned_dp ){
			sprintf(ERROR_STRING,
		"Object %s is not in host ram, cannot %s!?",
				OBJ_NAME(dp), op_str);
			warn(ERROR_STRING);
			warned_dp=dp;
		}
		return 0;
	}
	return 1;
}
#endif // HAVE_ANY_GPU

static int get_next_element(QSP_ARG_DECL   Data_Obj *dp,void *datap)
{
	Precision *prec_p;
	char *prompt;

	if( check_input_level() < 0 ) return(-1);

	// should the old test have been >= instead of > ???
	assert( OBJ_MACH_PREC(dp) < N_MACHINE_PRECS );
	

#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(ERROR_STRING,"get_next_element:  getting a %s value for address 0x%lx",
OBJ_MACH_PREC_NAME(dp),(u_long)datap);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	prec_p = OBJ_MACH_PREC_PTR(dp);
	prompt = msg_str;	// use this buffer...
	sprintf(prompt,"%s data",PREC_NAME(prec_p));
	(*(prec_p->set_value_from_input_func))(QSP_ARG  datap, prompt);

	dobj_n_gotten++;

	/* now lookahead to pop the file if it is empty */
	lookahead_til(ASCII_LEVEL-1);

	return(0);
} /* end get_next_element() */

static void bit_set_value_from_input(QSP_ARG_DECL  bitmap_word *wp, bitnum_t i_bit )
{
	bitmap_word val;
	bitmap_word bit;

	if( ! HAS_FORMAT_LIST )
		val = how_many("bit value");
	else
		val = next_input_int_with_format(QSP_ARG  "bit value");

	if( /* val < 0 || */ val > 1 ){     // bitmap_word is an unsigned type
		warn("Truncation error converting bit");
	}
	bit = 1 << (i_bit % BITS_PER_BITMAP_WORD);

	if( val == 0 )
		*( wp + i_bit/BITS_PER_BITMAP_WORD ) &= ~bit;
	else
		*( wp + i_bit/BITS_PER_BITMAP_WORD ) |=  bit;
}

static int get_next_bit(QSP_ARG_DECL  void *ptr, bitnum_t bit0)
{
	if( ptr != NULL )
		bit_set_value_from_input(QSP_ARG  (bitmap_word *) ptr, bit0 );
	return 0;
}

#ifdef BITMAP_FOOBAR
static int64_t get_bit_from_bitmap(Data_Obj *dp, void *data)
{
	int which_bit;
	int64_t l;

	/* if it's a bitmap, we don't really have the address of
	 * the data, we need to get the bit offset...
	 */

	/* We encode the bit in the data address - it's not the real address. */
	which_bit = ((u_char *)data) - ((u_char *)OBJ_DATA_PTR(dp));
sprintf(ERROR_STRING,"get_bit_from_bitmap:  which_bit = %d",which_bit);
advise(ERROR_STRING);
	which_bit += OBJ_BIT0(dp);

	/* now we know which bit it is, but we still need to figure
	 * out which word...  If this is a subobject of a bit image,
	 * we are ok, because the ultimate ancestor will be contiguous,
	 * but we can get in trouble if it is an equivalence of
	 * a non-contiguous object...  not sure how to handle this?
	 */

	data = ((BITMAP_DATA_TYPE *)OBJ_DATA_PTR(dp)) +
		(which_bit>>LOG2_BITS_PER_BITMAP_WORD);
	l= (int64_t)(* (BITMAP_DATA_TYPE *) data );
	which_bit &= BIT_NUMBER_MASK;

	if( l & (1L<<which_bit) )
		l=1;
	else
		l=0;
	return(l);
}
#endif /* BITMAP_FOOBAR */


/*
 * Format one (possibly multidimensional) pixel value
 *
 * It seems we are confused about what to do about bitmaps - BUG?
 */

void format_scalar_obj(QSP_ARG_DECL  char *buf,int buflen,Data_Obj *dp,void *data)
{
	//int64_t l;
	int c;

	if( OBJ_PREC(dp) == PREC_CHAR || OBJ_PREC(dp) == PREC_STR ){
		c=(*(char *)data);
		// BUG we don't check against buflen here, but should be OK
		if( isalnum(c) || ispunct(c) )
			sprintf(buf,"'%c'",c);
		else
			sprintf(buf,"0%o",c);
		return;
	}

	if( ! IS_BITMAP(dp) ){
		format_scalar_value(QSP_ARG  buf,buflen,data,OBJ_PREC_PTR(dp));
	}
	/*
	else {
		l = get_bit_from_bitmap(dp,data);
		sprintf(buf,ifmtstr,l);
	}
	*/
}

void format_scalar_value(QSP_ARG_DECL  char *buf,int buflen,void *data,Precision *prec_p)
{
	mach_prec mp;
	// long double is kind of inefficient unless we really need it?  BUG
#ifdef USE_LONG_DOUBLE
	long
#endif // USE_LONG_DOUBLE
	double ddata;
	int64_t l;

	mp = (mach_prec)(MP_BITS(PREC_CODE(prec_p)));
	switch( mp ){

		case PREC_NONE:
			assert( AERROR("format_scalar_value:  null precision!?") );
			break;

#ifdef USE_LONG_DOUBLE
		case PREC_LP: ddata=(* ((long double *)data) ); goto pntflt;
#endif // USE_LONG_DOUBLE
		case PREC_DP: ddata=(* ((double *)data) ); goto pntflt;
		case PREC_SP: ddata=(* ((float *)data) ); goto pntflt;
		case PREC_BY: l= (*(char *)data); goto pntlng;
		case PREC_IN: l= (* (short *) data ); goto pntlng;
		case PREC_DI: l= (* (int32_t *) data ); goto pntlng;
		case PREC_LI: l= (* (int64_t *) data ); goto pntlng;
		case PREC_UBY: l= (*(u_char *)data); goto pntlng;
		case PREC_UIN: l= (* (u_short *) data ); goto pntlng;
		case PREC_UDI: l= (* (uint32_t *) data ); goto pntlng;
		case PREC_ULI: l= (* (uint64_t *) data ); goto pntlng;

pntflt:
			/* BUG - sprintf suppresses the fractional part if there are only zeroes
			 * after the decimal point...
			 * But for a number like 4.000000001, we'd like to
			 * suppress the fraction if "digits" is set to something small...
			 */
			// BUG - need to use safe snprintf or something!?
			sprintf(buf,ffmtstr,ddata);
			break;

pntlng:
			sprintf(buf,ifmtstr,l);
			break;

		case PREC_INVALID:
		case N_MACHINE_PRECS:	/* silence compiler */
			assert( AERROR("format_scalar_value:  bad machine precision") );
			break;
	}
}

char * string_for_scalar(QSP_ARG_DECL  void *data,Precision *prec_p )
{
	static char buf[64];

	format_scalar_value(QSP_ARG  buf,64,data,prec_p);
	return buf;
}

Precision *src_prec_for_argset_prec(Argset_Prec * ap_p, argset_type at)
{
	int code=PREC_NONE;

	switch(ARGSPREC_CODE(ap_p)){
		case BY_ARGS:	code=PREC_BY; break;
		case IN_ARGS:	code=PREC_IN; break;
		case DI_ARGS:	code=PREC_DI; break;
		case LI_ARGS:	code=PREC_LI; break;
		case SP_ARGS:
			switch(at){
				case REAL_ARGS: 	code=PREC_SP; break;
				case MIXED_ARGS:
				case COMPLEX_ARGS:	code=PREC_CPX; break;
				case QMIXED_ARGS:
				case QUATERNION_ARGS:	code=PREC_QUAT; break;
				default:
					assert( AERROR("bad argset type in src_prec_for_argset_prec()") );
					break;
			}
			break;
		case DP_ARGS:
			switch(at){
				case REAL_ARGS: 	code=PREC_DP; break;
				case MIXED_ARGS:
				case COMPLEX_ARGS:	code=PREC_DBLCPX; break;
				case QMIXED_ARGS:
				case QUATERNION_ARGS:	code=PREC_DBLQUAT; break;
				default:
					assert( AERROR("bad argset type in src_prec_for_argset_prec()") );
					break;
			}
			break;
		case UBY_ARGS:	code=PREC_UBY; break;
		case UIN_ARGS:	code=PREC_UIN; break;
		case UDI_ARGS:	code=PREC_UDI; break;
		case ULI_ARGS:	code=PREC_ULI; break;
		case BYIN_ARGS:	code=PREC_BY; break;
		case INBY_ARGS:	code=PREC_IN; break;
		case INDI_ARGS:	code=PREC_IN; break;
		case SPDP_ARGS:	code=PREC_SP; break;
		default:
			assert( AERROR("bad argset prec in src_prec_for_argset_prec()") );
			break;
	}

	return( PREC_FOR_CODE(code) );
}

/*
 * Print one (possibly multidimensional) pixel value
 *
 * BUG - if we want it to go to a redirected output (prt_msg_vec),
 * it won't happen, because this doesn't use prt_msg!?
 *
 * We don't necessarily want to use prt_msg, because this can be
 * called both from a display function and a write function...
 */

static void pnt_one(QSP_ARG_DECL  FILE *fp, Data_Obj *dp,  u_char *data )
{
	char buf[128];

	if( OBJ_MACH_DIM(dp,0) > 1 ){	/* not real, complex or higher */
		if( IS_STRING(dp) ){
			fprintf(fp,"%s",data);
		} else {
			incr_t inc;
			dimension_t j;

			inc = (ELEMENT_SIZE(dp) * OBJ_MACH_INC(dp,0));
			for(j=0;j<OBJ_MACH_DIM(dp,0);j++){
				/* WHen this code was written, we didn't anticipate
				 * having too many components, so ret_dim=0 means
				 * print a return after each pixel.  But if
				 * we have many many components, then the resulting
				 * lines can be too long!?  Fixing this here is
				 * kind of a hack...
				 */
				if( j>0 && OBJ_MACH_DIM(dp,0) > dobj_max_per_line ){
					fprintf(fp,"\n");
				}
				format_scalar_obj(QSP_ARG  buf,128,dp,data);
				fprintf(fp," %s",buf);
				data += inc;
			}
		}
	} else {
		format_scalar_obj(QSP_ARG  buf,128,dp,data);
		fprintf(fp,"%s%s",ascii_separator,buf);
	}
} /* end pnt_one */

static void pnt_dim( QSP_ARG_DECL  FILE *fp, Data_Obj *dp, unsigned char *data, int dim )
{
	dimension_t i;
	incr_t inc;

	assert( ! IS_BITMAP(dp) );

	// Old code assumed that strings were the rows
	// of an image, but other code seemed to insist
	// that strings are multi-dimensional pixels...
	// By using OBJ_MINDIM, we handle both cases.

	if( dim==OBJ_MINDIM(dp) && OBJ_PREC(dp) == PREC_STR ){
		/* here we use .* to put the max number of chars
		 * to print in the next arg (is this a gcc
		 * extension or standard?).
		 * We do this because we are not guaranteed
		 * a null terminator char.
		 */
		fprintf(fp,"%.*s\n",(int)OBJ_DIMENSION(dp,OBJ_MINDIM(dp)),data);
		return;
	}

	if( dim > 0 ){
		inc=(ELEMENT_SIZE(dp)*OBJ_MACH_INC(dp,dim));
#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(ERROR_STRING,"pntdim: dim=%d, n=%d, inc=%d",dim,OBJ_MACH_DIM(dp,dim),
inc);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

		for(i=0;i<OBJ_MACH_DIM(dp,dim);i++)
			pnt_dim(QSP_ARG  fp,dp,data+i*inc,dim-1);
	} else {
		pnt_one(QSP_ARG  fp,dp,data);
	}
	if( dim == ret_dim ) fprintf(fp,"\n");
}

/* This is a speeded-up version for floats - do we need it? */

static void sp_pntvec( QSP_ARG_DECL  Data_Obj *dp, FILE *fp )
{
	float *base, *fbase, *rbase, *pbase;
	dimension_t i3,i2,i1,i0;

	base = (float *) OBJ_DATA_PTR(dp);

	for(i3=0;i3<OBJ_FRAMES(dp);i3++){
	    fbase=base+i3*OBJ_MACH_INC(dp,3);
	    for(i2=0;i2<OBJ_ROWS(dp);i2++){
		rbase=fbase+i2*OBJ_MACH_INC(dp,2);
		for(i1=0;i1<OBJ_COLS(dp);i1++){
		    pbase=rbase+i1*OBJ_MACH_INC(dp,1);
		    for(i0=0;i0<OBJ_MACH_DIM(dp,0);i0++){
			fprintf(fp," ");
			fprintf(fp,ffmtstr,*(pbase+i0*OBJ_MACH_INC(dp,0)));
		        if( ret_dim == 0 && OBJ_MACH_DIM(dp,0) > dobj_max_per_line )
				fprintf(fp,"\n");
		    }
		    if( ret_dim == 0 && OBJ_MACH_DIM(dp,0) <= dobj_max_per_line )
		    	fprintf(fp,"\n");
		}
		if( ret_dim > 0 ) fprintf(fp,"\n");
	    }
	}
}

static void set_pad_ffmt_str(SINGLE_QSP_ARG_DECL)
{
	sprintf(pad_ffmtstr,"%%%d.%dg",min_field_width,display_precision);
	ffmtstr = pad_ffmtstr;
}

#ifdef NOT_USED
void set_min_field_width(int fw)
{
	min_field_width=fw;
}
#endif /* NOT_USED */

void set_display_precision(QSP_ARG_DECL  int digits)
{
	display_precision=digits;
}

#define MAX_BITS_PER_LINE 32

static void display_bitmap(QSP_ARG_DECL  Data_Obj *dp, FILE *fp)
{
	int i,j,k,l,m;
	bitmap_word *bwp,val;
	bitnum_t which_bit;
	bitnum_t bit_index, word_offset;
	int bits_this_line;

	bwp = (bitmap_word *)OBJ_DATA_PTR(dp);

	bits_this_line=0;
	for(i=0;i<OBJ_SEQS(dp);i++){
		for(j=0;j<OBJ_FRAMES(dp);j++){
			for(k=0;k<OBJ_ROWS(dp);k++){
				for(l=0;l<OBJ_COLS(dp);l++){
					for(m=0;m<OBJ_COMPS(dp);m++){
						which_bit = OBJ_BIT0(dp)
							+ m * OBJ_TYPE_INC(dp,0)
							+ l * OBJ_TYPE_INC(dp,1)
							+ k * OBJ_TYPE_INC(dp,2)
							+ j * OBJ_TYPE_INC(dp,3)
							+ i * OBJ_TYPE_INC(dp,4);
						bit_index = which_bit % BITS_PER_BITMAP_WORD;
						word_offset = which_bit/BITS_PER_BITMAP_WORD;
						val = *(bwp + word_offset) & NUMBERED_BIT(bit_index); 
						if( bits_this_line >= MAX_BITS_PER_LINE ){
							bits_this_line=0;
							fputc('\n',fp);
							//prt_msg("");
						} else if( bits_this_line > 0 )
							fputc(' ',fp);
							//prt_msg_frag(" ");
						//prt_msg_frag(val?"1":"0");
						fputc(val?'1':'0',fp);
						bits_this_line++;
					}
				}
				if( bits_this_line > 0 ){
					bits_this_line=0;
					fputc('\n',fp);
					//prt_msg("");
				}
			}
		}
	}
}

// Not needed, because bitmaps don't use ret_dim
#ifdef FOOBAR
#define MULTIPLE_COLUMNS_OK(dp)						\
									\
	( ( IS_BITMAP(dp) && OBJ_COLS(dp) <= BITS_PER_BITMAP_WORD ) ||	\
	( ( ! IS_BITMAP(dp) ) && OBJ_COLS(dp) <= dobj_max_per_line ) )
#endif // FOOBAR

void pntvec(QSP_ARG_DECL  Data_Obj *dp,FILE *fp)			/**/
{
	const char *save_ifmt;
	const char *save_ffmt;
	/* first let's figure out when to print returns */
	/* ret_dim == 0 means a return is printed after every pixel */
	/* ret_dim == 1 means a return is printed after every row */

#ifdef THREAD_SAFE_QUERY
	// This is done for the main thread in dataobj_init()
	INSURE_QS_DOBJ_ASCII_INFO(THIS_QSP)
#endif // THREAD_SAFE_QUERY

// where is ret_dim declared?  part of qsp?
	if( OBJ_MACH_DIM(dp,0) == 1 && OBJ_COLS(dp) <= dobj_max_per_line)
		ret_dim=1;
	else ret_dim=0;

	/* We pad with spaces only if we are printing more than one element */

	save_ffmt=ffmtstr;
	save_ifmt=ifmtstr;
	/* BUG should set format based on desired radix !!! */
	padflag = 1;
	set_integer_print_fmt(QSP_ARG   THE_FMT_CODE);	/* handles integer formats */
	set_pad_ffmt_str(SINGLE_QSP_ARG);

	if( OBJ_MACH_PREC(dp) == PREC_SP ){
		sp_pntvec(QSP_ARG  dp,fp);
	} else if( OBJ_PREC(dp) == PREC_BIT )
		display_bitmap(QSP_ARG  dp,fp);
	else {
		/* the call to pnt_dim() was commented out,
		 * and the following lines calling dobj_iterate
		 * were there instead - why?
		 * Perhaps pnt_dim does not handle non-contig data
		 * correctly???  Or perhaps it was deemed better
		 * to use a general mechanism and dump the
		 * special purpose code???
		 *
		 * For now we reinstate pnt_dim to regain column
		 * format printing!
		 */

		pnt_dim(QSP_ARG  fp,dp,(u_char *)OBJ_DATA_PTR(dp),N_DIMENSIONS-1);
	}
	fflush(fp);

	ffmtstr = save_ffmt ;
	ifmtstr = save_ifmt ;
}

static void shp_trace(QSP_ARG_DECL  const char *name,Shape_Info *shpp)
{
	sprintf(ERROR_STRING,
		"%s: mindim = %d,  maxdim = %d",
		name, SHP_MINDIM(shpp), SHP_MAXDIM(shpp));
	advise(ERROR_STRING);

	sprintf(ERROR_STRING,
		"%s dim:  %u %u %u %u %u",
		name,
		SHP_TYPE_DIM(shpp,0),
		SHP_TYPE_DIM(shpp,1),
		SHP_TYPE_DIM(shpp,2),
		SHP_TYPE_DIM(shpp,3),
		SHP_TYPE_DIM(shpp,4));
	advise(ERROR_STRING);
}

void dptrace( QSP_ARG_DECL  Data_Obj *dp )
{
	shp_trace(QSP_ARG  OBJ_NAME( dp) ,OBJ_SHAPE(dp) );

	sprintf(ERROR_STRING,
		// why %u format when increment can be negative???
		"%s inc:  %u %u %u %u %u  (%u %u %u %u %u)",
		OBJ_NAME( dp) ,
		OBJ_TYPE_INC(dp,0),
		OBJ_TYPE_INC(dp,1),
		OBJ_TYPE_INC(dp,2),
		OBJ_TYPE_INC(dp,3),
		OBJ_TYPE_INC(dp,4),
		OBJ_MACH_INC(dp,0),
		OBJ_MACH_INC(dp,1),
		OBJ_MACH_INC(dp,2),
		OBJ_MACH_INC(dp,3),
		OBJ_MACH_INC(dp,4));
	advise(ERROR_STRING);
}

/* read an array of strings... */

static int get_strings(QSP_ARG_DECL  Data_Obj *dp,char *data,int dim)
{
	int status=0;

	if( dim == OBJ_MINDIM(dp) ){
		return( get_a_string(QSP_ARG  dp,data,dim) );
	} else {
		dimension_t i;
		long offset;

		offset = ELEMENT_SIZE( dp);
		offset *= OBJ_MACH_INC(dp,dim);
		for(i=0;i<OBJ_MACH_DIM(dp,dim);i++){
			status = get_strings(QSP_ARG  dp,data+i*offset,dim-1);
			if( status < 0 ) return status;
		}
	}
	return status;
}

static int get_bits(QSP_ARG_DECL  Data_Obj *dp, void *ptr, int dim, int bit0 )
{
	dimension_t i;
	long offset;

	if( dim < 0 ){
		return( get_next_bit(QSP_ARG  ptr,bit0) );
	}

	offset = OBJ_TYPE_INC(dp,dim);
	for(i=0;i<OBJ_TYPE_DIM(dp,dim);i++){
		int status;
		status = get_bits(QSP_ARG  dp,ptr,dim-1,bit0+(int)(i*offset));
		if( status < 0 ) return status;
	}
	return 0;
}

static int get_sheets(QSP_ARG_DECL  Data_Obj *dp,unsigned char *data,int dim)
{
	dimension_t i;
	long offset;
	int status=0;

	assert( ! IS_BITMAP(dp) );

	if( dim < 0 ){	/* get a component */
		return( get_next_element(QSP_ARG  dp,data) );
	}
	
	offset = ELEMENT_SIZE(dp);
	offset *= OBJ_MACH_INC(dp,dim);
	for(i=0;i<OBJ_MACH_DIM(dp,dim);i++){
		// if data is NULL, that means we aren't really writing...
		status = get_sheets(QSP_ARG  dp,
			data == NULL ? data : data+i*offset,
			dim-1);
		if( status < 0 ) return status;
	}
	return status;
}

void _read_ascii_data(QSP_ARG_DECL  Data_Obj *dp, FILE *fp, const char *s, int expect_exact_count)
{
	const char *orig_filename;
	int level;

	orig_filename = savestr(s);	/* with input formats, we might lose it */

	/*
	 * We check qlevel, so that if the file is bigger than
	 * the data object, the file will be closed so that
	 * the succeeding data will not be read as a command.
	 * This logic cannot deal with too *little* data in the
	 * file, that has to be taken care of in read_obj()
	 */

	//push_input_file(QSP_ARG  s);
	redir(fp, orig_filename);

	/* BUG we'd like to have the string be 'Pipe: "command args"' or something... */
	if( !strncmp(s,"Pipe",4) ){
		// THIS_QSP->qs_query[QLEVEL].q_flags |= Q_PIPE;
		SET_QS_FLAG_BITS( THIS_QSP, Q_PIPE );
	}

	level = QLEVEL;

	read_obj(dp);

	if( level == QLEVEL ){
		if( expect_exact_count ){
			sprintf(ERROR_STRING,
				"Needed %d values for object %s, file %s has more!?",
				OBJ_N_MACH_ELTS(dp),OBJ_NAME( dp) ,orig_filename);
			warn(ERROR_STRING);
		}
		pop_file();
	}

	rls_str( orig_filename);
} // read_ascii_data

void _read_obj(QSP_ARG_DECL   Data_Obj *dp)
{
	void *data_ptr;

	ASCII_LEVEL = QLEVEL;
	dobj_n_gotten = 0;

	if( ! OBJ_IS_RAM(dp) ){
		sprintf(ERROR_STRING,
	"read_obj:  object %s must be in RAM for assignment!?",
			OBJ_NAME(dp));
		warn(ERROR_STRING);
		data_ptr = NULL;
	} else {
		data_ptr = OBJ_DATA_PTR(dp);
	}

	if( HAS_FORMAT_LIST ){
		CURRENT_FORMAT_NODE = FIRST_INPUT_FORMAT_NODE;
	}

	if( dp != ascii_data_dp ){
		/* We store the target object so we can print its name if we need to... */
		ascii_data_dp = dp;		/* if this is global, then this is not thread-safe!? */
		ascii_warned=0;
	}

	if( OBJ_PREC(dp) == PREC_CHAR || OBJ_PREC(dp) == PREC_STR ){
		if( get_strings(QSP_ARG  dp,(char *)data_ptr,N_DIMENSIONS-1) < 0 ){
			sprintf(ERROR_STRING,"error reading strings for object %s",OBJ_NAME( dp) );
			warn(ERROR_STRING);
		}
	} else if( IS_BITMAP(dp) ){
		if( get_bits(QSP_ARG  dp,data_ptr,N_DIMENSIONS-1,OBJ_BIT0(dp)) < 0){
			sprintf(ERROR_STRING,"expected %d bits for bitmap object %s",
				OBJ_N_TYPE_ELTS(dp),OBJ_NAME( dp) );
			warn(ERROR_STRING);
		}
	} else {	// normal object
		if( get_sheets(QSP_ARG  dp,(u_char *)data_ptr,N_DIMENSIONS-1) < 0 ){
			sprintf(ERROR_STRING,"expected %d elements for object %s",
				OBJ_N_MACH_ELTS(dp),OBJ_NAME( dp) );
			warn(ERROR_STRING);
		}
	}

	// If we are reading formatted input, there may be some irrelavant fields
	// that could trigger a "too many values" warning...
	if( HAS_FORMAT_LIST ){
		if( CURRENT_FORMAT_NODE != FIRST_INPUT_FORMAT_NODE ){
			consume_format_line(QSP_ARG  OBJ_PREC_PTR(dp));
		}
	}

} // read_obj

/* has to be external, called from datamenu/ascmenu.c */

void set_integer_print_fmt(QSP_ARG_DECL  Number_Fmt fmt_code )
{
	THE_FMT_CODE = fmt_code;	/* per qsp variable */
	switch(fmt_code){
		case FMT_POSTSCRIPT:
			ifmtstr= "%02x"; break;
		case FMT_UDECIMAL:
			ifmtstr= (padflag ? "%10lu"   : "%lu")  ; break;
		case FMT_DECIMAL:
			ifmtstr= (padflag ? "%10ld"   : "%ld")  ; break;
		case FMT_HEX:
			ifmtstr= (padflag ? "0x%-10lx" : "0x%lx"); break;
		case FMT_OCTAL:
			ifmtstr= (padflag ? "0%-10lo"  : "0%lo") ; break;
		default:
			assert( AERROR("unrecognized format code") );
	}
}

void set_max_per_line(QSP_ARG_DECL  int n )
{
	if( n < 1 )
		warn("max_per_line must be positive");
	else if( n > ENFORCED_MAX_PER_LINE ){
		sprintf(ERROR_STRING,"Requested max_per_line (%d) exceeds hard-coded maximum (%d)",
				n,ENFORCED_MAX_PER_LINE);
		warn(ERROR_STRING);
	} else
		dobj_max_per_line = n;
}

static void int_format_release(Input_Format_Spec *fmt_p) { default_format_release(fmt_p); }
static void float_format_release(Input_Format_Spec *fmt_p) { default_format_release(fmt_p); }
static void string_format_release(Input_Format_Spec *fmt_p) { default_format_release(fmt_p); }

#define INIT_FORMAT_TYPE(code,fmt_name,stem)			\
								\
	{							\
	struct input_format_type *ft_p;				\
	assert( code >= 0 && code < N_INPUT_FORMAT_TYPES );	\
	ft_p = &input_format_type_tbl[code];			\
	ft_p->name = #fmt_name;					\
	ft_p->type_code = code;					\
	ft_p->display_format = display_##stem;			\
	ft_p->release = stem##_release;				\
	ft_p->consume = stem##_consume;				\
	ft_p->read_long = stem##_read_long;			\
	ft_p->read_double = stem##_read_double;			\
	ft_p->read_string = stem##_read_string;			\
	}


static void init_format_type_tbl(void)
{
	INIT_FORMAT_TYPE(IN_FMT_INT,integer,int_format)
	INIT_FORMAT_TYPE(IN_FMT_FLT,float,float_format)
	INIT_FORMAT_TYPE(IN_FMT_STR,string,string_format)
	INIT_FORMAT_TYPE(IN_FMT_LIT,literal,literal_format)
}

