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
#include "query_stack.h"	// like to eliminate this dependency...



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

/* all these variables should be per-thread...
 * Instead of adding them all to the Query_Stream struct, better to have them in a separate
 * data-module struct, that we allocate dynamically and point to from the query_stream...
 * FIXME
 */

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

static int n_format_fields, curr_fmt_i;

void init_dobj_ascii_info(QSP_ARG_DECL  Dobj_Ascii_Info *dai_p)
{
	dai_p->dai_padflag = 0;
	dai_p->dai_ascii_data_dp = NO_OBJ;
	dai_p->dai_ascii_warned = 0;
	dai_p->dai_dobj_max_per_line = DEFAULT_MAX_PER_LINE;
	dai_p->dai_min_field_width = DEFAULT_MIN_FIELD_WIDTH;
	dai_p->dai_display_precision = DEFAULT_DISPLAY_PRECISION;
	dai_p->dai_ascii_separator = NORMAL_SEPARATOR;
	dai_p->dai_ffmtstr = NOPAD_FLT_FMT_STR;
	dai_p->dai_ifmtstr = NOPAD_INT_FMT_STR;
}

static void show_input_format(SINGLE_QSP_ARG_DECL)
{
	int i=0;

	if( n_format_fields <= 0 ){
		advise("no input format specified");
		return;
	}

	while(i<n_format_fields){
		if( i > 0 ) prt_msg_frag(" ");
		switch( ascii_input_fmt_tbl[i].fmt_type ){
			case IN_FMT_STR:  prt_msg_frag("%s"); break;
			case IN_FMT_LIT:  prt_msg_frag(ascii_input_fmt_tbl[i].fmt_litstr); break;
			case IN_FMT_FLT:  prt_msg_frag("%f"); break;
			case IN_FMT_INT:  prt_msg_frag("%d"); break;
		}
		i++;
	}
	prt_msg("");
}

#define MAX_LIT_STR_LEN	255

void set_input_format_string( QSP_ARG_DECL  const char *s )
{
	const char *orig_str=s;
	char lit_str[MAX_LIT_STR_LEN+1];

	while( n_format_fields > 0 ){
		/* release any old literal strings */
		n_format_fields--;
		if( ascii_input_fmt_tbl[n_format_fields].fmt_type == IN_FMT_LIT )
			rls_str( (char *) ascii_input_fmt_tbl[n_format_fields].fmt_litstr );
	}

	/* parse the string */

	while( *s && n_format_fields < MAX_FORMAT_FIELDS ){
		while( isspace(*s) ) s++;
		if( *s == 0 ) return;

		if( *s == '%' ){
			s++;
			switch(*s){
				case 0:
					sprintf(ERROR_STRING,
		"Poorly formed input format string \"%s\"" , orig_str);
					WARN(ERROR_STRING);
					return;

				case 'd':
				case 'x':
				case 'o':
				case 'i':
					ascii_input_fmt_tbl[n_format_fields].fmt_type = IN_FMT_INT;
					break;
				case 'f':
				case 'g':
					ascii_input_fmt_tbl[n_format_fields].fmt_type = IN_FMT_FLT;
					break;
				case 's':
					ascii_input_fmt_tbl[n_format_fields].fmt_type = IN_FMT_STR;
					break;
			}
			s++;
			if( *s && !isspace(*s) ){
				sprintf(ERROR_STRING,
					"white space should follow format, \"%s\"",
					orig_str);
				WARN(ERROR_STRING);
			}
		} else {	/* literal string */
			int i=0;

			ascii_input_fmt_tbl[n_format_fields].fmt_type=IN_FMT_LIT;
			while( *s && !isspace(*s) && i < MAX_LIT_STR_LEN )
				lit_str[i++]=(*s++);
			lit_str[i] = 0;
			if( *s && !isspace(*s) )
				WARN("literal string overflow in input format spec");
			ascii_input_fmt_tbl[n_format_fields].fmt_litstr = savestr(lit_str);
		}
		n_format_fields++;
	}
	if( n_format_fields >= MAX_FORMAT_FIELDS && *s != 0 ){
		sprintf(ERROR_STRING,
	"Max number of format fields (%d) used up before done processing format string!?",
			MAX_FORMAT_FIELDS);
		WARN(ERROR_STRING);
	}
}

#define NEXT_FORMAT		{	curr_fmt_i++;				\
					if( curr_fmt_i >= n_format_fields ){	\
						curr_fmt_i=0;			\
						done = havit;			\
					}					\
				}

#define READ_LITERAL		{					\
									\
	s=NAMEOF(ascii_input_fmt_tbl[curr_fmt_i].fmt_litstr);			\
	if( strcmp(s,ascii_input_fmt_tbl[curr_fmt_i].fmt_litstr) ){		\
		sprintf(ERROR_STRING,					\
	"expected literal string \"%s\", saw string \"%s\"",		\
			ascii_input_fmt_tbl[curr_fmt_i].fmt_litstr,s);		\
		WARN(ERROR_STRING);					\
	}								\
}

#define SET_DONE	{ done=1; curr_fmt_i--; }

static long next_input_int(QSP_ARG_DECL   const char *pmpt)
{
	const char *s;
	long l=0;
	int havit=0;
	int done=0;

	do {
		switch( ascii_input_fmt_tbl[curr_fmt_i].fmt_type ){
			case IN_FMT_LIT:
				READ_LITERAL;
				break;
			case IN_FMT_STR: /*s=*/NAMEOF("don't-care string"); break;
			case IN_FMT_INT:
				if( havit ) SET_DONE
				else {
					l = HOW_MANY(pmpt);
					havit=1;
				}
				break;
			case IN_FMT_FLT:
				if( havit ) SET_DONE
				else {
					if( !ascii_warned ){
						sprintf(ERROR_STRING,
							"Float format data assigned to integer object %s!?",
							OBJ_NAME( ascii_data_dp) );
						WARN(ERROR_STRING);
						ascii_warned=1;
					}

					l = (long) HOW_MUCH(pmpt);
					havit=1;
				}
				break;
		}
		NEXT_FORMAT;
	} while(!done);

	return(l);
}

static double next_input_flt(QSP_ARG_DECL   const char *pmpt)
{
	const char *s;
	double d=0.0;
	int havit=0;
	int done=0;

	do {
		switch( ascii_input_fmt_tbl[curr_fmt_i].fmt_type ){
			case IN_FMT_LIT: READ_LITERAL; break;
			case IN_FMT_STR: /*s=*/NAMEOF("don't-care string"); break;
			case IN_FMT_FLT:
				if( havit ) SET_DONE
				else {
					d = HOW_MUCH(pmpt);
					havit=1;
				}
				break;
			case IN_FMT_INT:
				if( havit ) SET_DONE
				else {
					d = HOW_MANY(pmpt);
					havit=1;
				}
				break;
		}
		NEXT_FORMAT;
	} while(!done);

	return(d);
}

static const char * next_input_str(QSP_ARG_DECL  const char *pmpt)
{
	const char *s
		= NULL		/* to elim possibly used w/o init warning */
	;
	//double d;
	int havit=0;
	int done=0;

	do {
		switch( ascii_input_fmt_tbl[curr_fmt_i].fmt_type ){
			case IN_FMT_LIT: READ_LITERAL; break;
			case IN_FMT_STR:
				if( havit ) SET_DONE
				else {
					s=NAMEOF(pmpt);
					havit=1;
				}
				break;
			case IN_FMT_FLT:
				/*d =*/ HOW_MUCH("don't care float number");
				break;
			case IN_FMT_INT:
				/*d =*/ HOW_MANY("don't care integer number");
				break;
		}
		NEXT_FORMAT;
	} while(!done);

	return(s);
}

static int check_input_level(SINGLE_QSP_ARG_DECL)
{
	if( QLEVEL != ASCII_LEVEL ){
		sprintf(ERROR_STRING,"check_input_level (ascii):  input depth is %d, expected %d!?",
			QLEVEL,ASCII_LEVEL);
		WARN(ERROR_STRING);
		advise("premature end of data");
		sprintf(ERROR_STRING,"%d elements read so far",dobj_n_gotten);
		advise(ERROR_STRING);
		if( n_format_fields > 0 ){
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

//#ifdef CAUTIOUS
//	if( dim < 0 ){
//		WARN("CAUTIOUS:  get_a_string:  negative dim!?");
//		return -1;
//	}
//#endif // CAUTIOUS
	assert( dim >= 0 );

	if( check_input_level(SINGLE_QSP_ARG) < 0 ) return(-1);

	/* see if we need to look at the input format string */
	if( n_format_fields == 0 )
		s = NAMEOF("string data");
	else
		s = next_input_str(QSP_ARG  "string data");

	/* FIXME should use strncpy() here */
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
		WARN(ERROR_STRING);
	}
	*t = 0;	// add null terminator

	/* now lookahead to pop the file if it is empty */
	lookahead_til(QSP_ARG  ASCII_LEVEL-1);

	return(0);
}

#ifdef HAVE_ANY_GPU
int object_is_in_ram(QSP_ARG_DECL  Data_Obj *dp, const char *op_str)
{
	static Data_Obj *warned_dp=NO_OBJ;

	if( ! OBJ_IS_RAM(dp) ){
		if( dp != warned_dp ){
			sprintf(ERROR_STRING,
		"Object %s is not in host ram, cannot %s!?",
				OBJ_NAME(dp), op_str);
			WARN(ERROR_STRING);
			warned_dp=dp;
		}
		return 0;
	}
	return 1;
}
#endif // HAVE_ANY_GPU

#define DEREF(ptr,type)	 (*((type *)ptr))

static void set_one_value(QSP_ARG_DECL  Data_Obj *dp, void *datap, void * num_ptr)
{
	mach_prec mp;
	long l;
	static Data_Obj *warned_dp=NO_OBJ;

#ifdef FOOBAR
#ifdef HAVE_CUDA
	if( ! object_is_in_ram(QSP_ARG  dp,"set a value") ) return;
#endif //HAVE_CUDA
#endif // FOOBAR

	mp = OBJ_MACH_PREC(dp);
	switch( mp ){
#ifdef USE_LONG_DOUBLE
		case PREC_LP:
			* ((long double *)datap) = DEREF(num_ptr,long double);
			break;
#endif // USE_LONG_DOUBLE
		case PREC_DP:
			* ((double *)datap) = DEREF(num_ptr,double);
			break;
		case PREC_SP:
			* ((float *)datap) =(float) DEREF(num_ptr,double);
			break;
		case PREC_BY:
			l = DEREF(num_ptr,long);
			if( (l < -128 || l > 127) && warned_dp!=dp ){
				sprintf(ERROR_STRING,
			"data (0x%lx) out of range for byte conversion, object %s",
			l,OBJ_NAME( dp) );
				WARN(ERROR_STRING);
				warned_dp=dp;
			}
			*(char *)datap = (char)(l);
			break;
		case PREC_UBY:
			l = DEREF(num_ptr,long);
			if( (l < 0 || l > 255) && warned_dp!=dp ){
				sprintf(ERROR_STRING,
			"data (0x%lx) out of range for unsigned byte conversion, object %s",
			l,OBJ_NAME( dp) );
				WARN(ERROR_STRING);
				warned_dp=dp;
			}
			*(u_char *)datap = (u_char)(l);
			break;

/* these values are for two's complement!? */
#define MIN_SIGNED_SHORT	-32768		/* 0x8000 */
#define MAX_SIGNED_SHORT	0x7fff		/*  32767 */
#define MIN_UNSIGNED_SHORT	0x0000
#define MAX_UNSIGNED_SHORT	0xffff

		case PREC_IN:
			l = DEREF(num_ptr,long);
			if( (l < MIN_SIGNED_SHORT || l > MAX_SIGNED_SHORT )
					&& warned_dp!=dp ){

				sprintf(ERROR_STRING,
		"number %ld (0x%lx) won't fit in a signed short, object %s",
					l,l,OBJ_NAME( dp) );
				WARN(ERROR_STRING);
				warned_dp=dp;
			}
			* ((short *)datap)=(short)l;
			break;
		case PREC_UIN:
			l = DEREF(num_ptr,long);
			if( (l < MIN_UNSIGNED_SHORT || l > MAX_UNSIGNED_SHORT )
					&& warned_dp!=dp ){

				sprintf(ERROR_STRING,
		"number %ld (0x%lx) won't fit in an unsigned short, object %s",
					l,l,OBJ_NAME( dp) );
				WARN(ERROR_STRING);
				warned_dp=dp;
			}
			* ((u_short *)datap)=(u_short)l;
			break;
		case PREC_DI:
			l = DEREF(num_ptr,long);
			* ((int32_t *)datap)=(int32_t)l;
			break;
		case PREC_UDI:
			l = DEREF(num_ptr,long);
			// BUG do range checks on 64 bit arch!
			if( IS_BITMAP(dp) ){
				long offset;
				int bit;
		/* BUG - here we are assuming that the bitmap word type is PREC_UDI,
		 * so we are right shifting by 5 instead of LOG2_BITS_PER_BITMAP_WORD...
		 * BUT we are using BITMAP_DATA_TYPE!?
		 * CAUTIOUS check here???
		 */
		/* We have faked datap to get the bit offset */
				offset = ((BITMAP_DATA_TYPE *)datap) -
					((BITMAP_DATA_TYPE *)OBJ_DATA_PTR(dp));
				/* offset is in bits... */
				datap = ((BITMAP_DATA_TYPE *)OBJ_DATA_PTR(dp)) +
					((OBJ_BIT0(dp) + offset)>>5);
				bit = (OBJ_BIT0(dp)+offset)&BIT_NUMBER_MASK;
				/* We used to have 1<<bit here, but that gave 0
				 * for more than 32 bits.
				 * Is that because the compiler treats the 1
				 * as a 32 bit number?
				 */
				if( l==1 )
					*((BITMAP_DATA_TYPE *)datap) |= NUMBERED_BIT(bit);

				else if( l == 0 )
					*((BITMAP_DATA_TYPE *)datap) &= ~NUMBERED_BIT(bit);
				else {
					sprintf(ERROR_STRING,
				"Non-boolean value %ld specified for bitmap %s!?",
						l,OBJ_NAME( dp) );
					WARN(ERROR_STRING);
				}
			} else {
				* ((uint32_t *)datap)=(uint32_t)l;
			}
			break;

		case PREC_LI:
			l = DEREF(num_ptr,long);
			* ((int64_t *)datap)=(int64_t)l;
			break;
		case PREC_ULI:
			l = DEREF(num_ptr,long);
			if( IS_BITMAP(dp) ){
				long offset;
				int bit;
				/* hard coded to 6 instead of LOG2_BITS_PER_BITMAP_WORD?
				 * BUG?
				 * See comment above...
				 */
				offset = ((BITMAP_DATA_TYPE *)datap) -
						((BITMAP_DATA_TYPE *)OBJ_DATA_PTR(dp));
				datap = ((BITMAP_DATA_TYPE *)OBJ_DATA_PTR(dp)) +
						((OBJ_BIT0(dp) + offset)>>6);
				bit = (OBJ_BIT0(dp)+offset)&BIT_NUMBER_MASK;
				if( l==1 )
					*((BITMAP_DATA_TYPE *)datap) |= (1<<bit);
				else if( l == 0 )
					*((BITMAP_DATA_TYPE *)datap) &= ~(1<<bit);
				else {
					sprintf(ERROR_STRING,
					"Non-boolean value %ld specified for bitmap %s!?",
						l,OBJ_NAME( dp) );
					WARN(ERROR_STRING);
				}
			} else {
				* ((uint64_t *)datap)=(uint64_t)l;
			}
			break;

//#ifdef CAUTIOUS
		case PREC_INVALID:
		case N_MACHINE_PRECS:	/* just to silence compiler */
		case PREC_NONE:		/* should have been handled above */
			assert( AERROR("Unexpected case in switch!?") );
			break;
//#endif /* CAUTIOUS */

	}
}

static int get_next(QSP_ARG_DECL   Data_Obj *dp,void *datap)
{
	/* init these to eliminate optimizer warnings */
#ifdef USE_LONG_DOUBLE
	long
#endif // USE_LONG_DOUBLE
	double d_number=0.0;
	long l=0L;
	mach_prec mp;
	void *num_ptr;

	if( check_input_level(SINGLE_QSP_ARG) < 0 ) return(-1);

//#ifdef CAUTIOUS
//	if( OBJ_MACH_PREC(dp) > N_MACHINE_PRECS ){
//		sprintf(ERROR_STRING,
//	"CAUTIOUS:  get_next:  Object %s precision %s is not a machine precision!?",
//			OBJ_NAME( dp) ,OBJ_MACH_PREC_NAME(dp) );
//			WARN(ERROR_STRING);
//		return(-1);
//	}
//#endif /* CAUTIOUS */
	// should the old test have been >= instead of > ???
	assert( OBJ_MACH_PREC(dp) < N_MACHINE_PRECS );
	

#ifdef QUIP_DEBUG
if( debug & debug_data ){
sprintf(ERROR_STRING,"get_next:  getting a %s value for address 0x%lx",
OBJ_MACH_PREC_NAME(dp),(u_long)datap);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	mp = OBJ_MACH_PREC(dp);
	num_ptr = NULL;
	switch( mp ){
#ifdef USE_LONG_DOUBLE
		case PREC_LP:
#endif // USE_LONG_DOUBLE
		case PREC_DP:  case PREC_SP:
			if( n_format_fields == 0 )
				d_number = HOW_MUCH("real data");
			else
				d_number = next_input_flt(QSP_ARG  "real data");
			num_ptr = &d_number;
			break;
		case PREC_BY:
		case PREC_IN:
		case PREC_DI:
		case PREC_LI:
		case PREC_UBY:
		case PREC_UIN:
		case PREC_UDI:
		case PREC_ULI:
			if( n_format_fields == 0 )
				l=HOW_MANY("integer data");
			else
				l=next_input_int(QSP_ARG  "integer data");
			num_ptr = &l;
			break;
		case PREC_NONE:
			sprintf(ERROR_STRING,"get_next:  object %s has no data!?",OBJ_NAME( dp) );
			WARN(ERROR_STRING);
			return(-1);
			break;
//#ifdef CAUTIOUS
		case PREC_INVALID:
		case N_MACHINE_PRECS:	/* have this case here to silence compiler */
//			ERROR1("bad case in get_next");
			assert( AERROR("bad case in get_next"));
			break;
		/* default: ERROR1("CAUTIOUS:  get_next, bad machine precision"); break; */
//#endif /* CAUTIOUS */
	}

	set_one_value(QSP_ARG  dp, datap, num_ptr);
	dobj_n_gotten++;

	/* now lookahead to pop the file if it is empty */
	lookahead_til(QSP_ARG  ASCII_LEVEL-1);

	return(0);
} /* end get_next() */

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

//#ifdef CAUTIOUS
		case PREC_NONE:
//			NERROR1("CAUTIOUS:  format_scalar_value:  null precision!?");
			assert( AERROR("format_scalar_value:  null precision!?") );
			break;
//#endif /* CAUTIOUS */

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

//#ifdef CAUTIOUS
		case PREC_INVALID:
		case N_MACHINE_PRECS:	/* silence compiler */
//			NERROR1("CAUTIOUS:  format_scalar_value:  bad machine precision");
			assert( AERROR("format_scalar_value:  bad machine precision") );
			break;
		/* default: ERROR1("CAUTIOUS:  format_scalar_value:  unknown prec"); break; */
//#endif /* CAUTIOUS */
	}
}

char * string_for_scalar(QSP_ARG_DECL  void *data,Precision *prec_p )
{
	static char buf[64];

fprintf(stderr,"string_for_scalar using precision %s\n",PREC_NAME(prec_p));
	format_scalar_value(QSP_ARG  buf,64,data,prec_p);
	return buf;
}

Precision *src_prec_for_argset_prec(argset_prec ap,argset_type at)
{
	int code=PREC_NONE;

	switch(ap){
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
				// BUG mixed args depend on WHICH source arg!
				default:
					//NERROR1("CAUTIOUS:  bad argset type in src_prec_for_argset_prec()");
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
				// BUG mixed args depend on WHICH source arg!
				default:
					//NERROR1("CAUTIOUS:  bad argset type in src_prec_for_argset_prec()");
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
//			NERROR1("CAUTIOUS:  bad argset prec in src_prec_for_argset_prec()");
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
sprintf(DEFAULT_ERROR_STRING,"pntdim: dim=%d, n=%d, inc=%d",dim,OBJ_MACH_DIM(dp,dim),
inc);
advise(DEFAULT_ERROR_STRING);
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
	int bit_index, word_offset;
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

#ifdef FOOBAR
#ifdef HAVE_CUDA
	if( ! object_is_in_ram(QSP_ARG  dp,"display") ) return;
#endif //HAVE_CUDA
#endif // FOOBAR

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
	sprintf(DEFAULT_ERROR_STRING,
		"%s: mindim = %d,  maxdim = %d",
		name, SHP_MINDIM(shpp), SHP_MAXDIM(shpp));
	advise(DEFAULT_ERROR_STRING);

	sprintf(DEFAULT_ERROR_STRING,
		"%s dim:  %u %u %u %u %u",
		name,
		SHP_TYPE_DIM(shpp,0),
		SHP_TYPE_DIM(shpp,1),
		SHP_TYPE_DIM(shpp,2),
		SHP_TYPE_DIM(shpp,3),
		SHP_TYPE_DIM(shpp,4));
	advise(DEFAULT_ERROR_STRING);
}

void dptrace( QSP_ARG_DECL  Data_Obj *dp )
{
	shp_trace(QSP_ARG  OBJ_NAME( dp) ,OBJ_SHAPE(dp) );

	sprintf(DEFAULT_ERROR_STRING,
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
	advise(DEFAULT_ERROR_STRING);
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
			if( status < 0 ) return(status);
		}
	}
	return(status);
}

static int get_sheets(QSP_ARG_DECL  Data_Obj *dp,unsigned char *data,int dim)
{
	dimension_t i;
	long offset;

	if( dim < 0 ){	/* get a component */
		return( get_next(QSP_ARG  dp,data) );
	} else {
		int status=0;

		offset = ELEMENT_SIZE( dp);
		if( IS_BITMAP(dp) ){
			offset *= OBJ_TYPE_INC(dp,dim);
			for(i=0;i<OBJ_TYPE_DIM(dp,dim);i++){
				status = get_sheets(QSP_ARG  dp,data+i*offset,dim-1);
				if( status < 0 ) return(status);
			}
		} else {
			offset *= OBJ_MACH_INC(dp,dim);
			for(i=0;i<OBJ_MACH_DIM(dp,dim);i++){
				status = get_sheets(QSP_ARG  dp,data+i*offset,dim-1);
				if( status < 0 ) return(status);
			}
		}
		return(status);
	}
}

void read_ascii_data(QSP_ARG_DECL  Data_Obj *dp, FILE *fp, const char *s, int expect_exact_count)
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
	redir(QSP_ARG  fp, orig_filename);

	/* BUG we'd like to have the string be 'Pipe: "command args"' or something... */
	if( !strncmp(s,"Pipe",4) ){
		// THIS_QSP->qs_query[QLEVEL].q_flags |= Q_PIPE;
		SET_QS_FLAG_BITS( THIS_QSP, Q_PIPE );
	}

	level = QLEVEL;

	read_obj(QSP_ARG  dp);

	if( level == QLEVEL ){
		if( expect_exact_count ){
			sprintf(ERROR_STRING,
				"Needed %d values for object %s, file %s has more!?",
				OBJ_N_MACH_ELTS(dp),OBJ_NAME( dp) ,orig_filename);
			WARN(ERROR_STRING);
		}
		pop_file(SINGLE_QSP_ARG);
	}

	rls_str( orig_filename);
}

void read_obj(QSP_ARG_DECL   Data_Obj *dp)
{
	ASCII_LEVEL = QLEVEL;
	dobj_n_gotten = 0;

	if( dp != ascii_data_dp ){
		/* We store the target object so we can print its name if we need to... */
		ascii_data_dp = dp;		/* if this is global, then this is not thread-safe!? */
		ascii_warned=0;
	}

	if( OBJ_PREC(dp) == PREC_CHAR || OBJ_PREC(dp) == PREC_STR ){
		if( get_strings(QSP_ARG  dp,(char *)OBJ_DATA_PTR(dp),N_DIMENSIONS-1) < 0 ){
			sprintf(ERROR_STRING,"error reading strings for object %s",OBJ_NAME( dp) );
			WARN(ERROR_STRING);
		}
	} else if( get_sheets(QSP_ARG  dp,(u_char *)OBJ_DATA_PTR(dp),N_DIMENSIONS-1) < 0 ){
		/*
		sprintf(ERROR_STRING,"error reading ascii data for object %s",OBJ_NAME( dp) );
		WARN(ERROR_STRING);
		*/
		sprintf(ERROR_STRING,"expected %d elements for object %s",
			OBJ_N_MACH_ELTS(dp),OBJ_NAME( dp) );
		WARN(ERROR_STRING);
	}
}

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
//#ifdef CAUTIOUS
		default:
//			ERROR1("CAUTIOUS:  unrecognized format code");
			assert( AERROR("unrecognized format code") );
//#endif /* CAUTIOUS */
	}
}

void set_max_per_line(QSP_ARG_DECL  int n )
{
	if( n < 1 )
		WARN("max_per_line must be positive");
	else if( n > ENFORCED_MAX_PER_LINE ){
		sprintf(ERROR_STRING,"Requested max_per_line (%d) exceeds hard-coded maximum (%d)",
				n,ENFORCED_MAX_PER_LINE);
		WARN(ERROR_STRING);
	} else
		dobj_max_per_line = n;
}

