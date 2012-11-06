#include "quip_config.h"

char VersionId_dataf_ascii[] = QUIP_VERSION_STRING;

/* more generalized dimensions & subscripts added 8-92 jbm */

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#include "data_obj.h"
#include "debug.h"
#include "query.h"		/* tell_qlevel() */

static int padflag=0;
static Data_Obj *ascii_data_dp=NO_OBJ;
static int ascii_warned=0;


/* local prototypes */
static void show_input_format(void);
static int get_next(QSP_ARG_DECL  Data_Obj *,void *);
static void pnt_one(FILE *,Data_Obj *,unsigned char *);
static void pnt_dim(FILE *,Data_Obj *,unsigned char *,int);
static void sp_pntvec(Data_Obj *,FILE *);
static int get_sheets(QSP_ARG_DECL  Data_Obj *,unsigned char *,int);
static long next_input_int(QSP_ARG_DECL  const char *pmpt);
static double next_input_flt(QSP_ARG_DECL  const char *pmpt);
static const char * next_input_str(QSP_ARG_DECL  const char *pmpt);

/*
 * For printing out data with more than one number per line,
 * ret_dim is the index of the dimension where a return is forced
 */

static int ret_dim;
#define DEFAULT_MAX_PER_LINE	12
#define ENFORCED_MAX_PER_LINE	50

static dimension_t	n_gotten;

static dimension_t max_per_line=DEFAULT_MAX_PER_LINE;
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
static int min_field_width=DEFAULT_MIN_FIELD_WIDTH;
static int display_precision=DEFAULT_DISPLAY_PRECISION;

#define NORMAL_SEPARATOR	" "
#define POSTSCRIPT_SEPARATOR	""
static const char *separator=NORMAL_SEPARATOR;

static const char *ffmtstr=NOPAD_FLT_FMT_STR;		/* float format string */
static char pad_ffmtstr[16];
static const char *ifmtstr=NOPAD_INT_FMT_STR;		/* integer format string */

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
static Input_Format_Spec input_fmt[MAX_FORMAT_FIELDS];

static void show_input_format()
{
	int i=0;

	if( n_format_fields <= 0 ){
		advise("no input format specified");
		return;
	}

	while(i<n_format_fields){
		if( i > 0 ) prt_msg_frag(" ");
		switch( input_fmt[i].fmt_type ){
			case IN_FMT_STR:  prt_msg_frag("%s"); break;
			case IN_FMT_LIT:  prt_msg_frag(input_fmt[i].fmt_litstr); break;
			case IN_FMT_FLT:  prt_msg_frag("%f"); break;
			case IN_FMT_INT:  prt_msg_frag("%d"); break;
		}
		i++;
	}
	prt_msg("");
}

void set_input_format_string( QSP_ARG_DECL  const char *s )
{
	const char *orig_str=s;
	char lit_str[256];

	while( n_format_fields > 0 ){
		/* release any old literal strings */
		n_format_fields--;
		if( input_fmt[n_format_fields].fmt_type == IN_FMT_LIT )
			rls_str( input_fmt[n_format_fields].fmt_litstr );
	}

	/* parse the string */

	while( *s && n_format_fields < MAX_FORMAT_FIELDS ){
		while( isspace(*s) ) s++;
		if( *s == 0 ) return;

		if( *s == '%' ){
			s++;
			switch(*s){
				case 0:
					sprintf(error_string,
		"Poorly formed input format string \"%s\"" , orig_str);
					WARN(error_string);
					return;

				case 'd':
				case 'x':
				case 'o':
				case 'i':
					input_fmt[n_format_fields].fmt_type = IN_FMT_INT;
					break;
				case 'f':
				case 'g':
					input_fmt[n_format_fields].fmt_type = IN_FMT_FLT;
					break;
				case 's':
					input_fmt[n_format_fields].fmt_type = IN_FMT_STR;
					break;
			}
			s++;
			if( *s && !isspace(*s) ){
				sprintf(error_string,
					"white space should follow format, \"%s\"",
					orig_str);
				WARN(error_string);
			}
		} else {	/* literal string */
			int i=0;

			input_fmt[n_format_fields].fmt_type=IN_FMT_LIT;
			while( *s && !isspace(*s) && i < (LLEN-1) )
				lit_str[i++]=(*s++);
			lit_str[i] = 0;
			if( *s && !isspace(*s) )
				WARN("literal string overflow in input format spec");
			input_fmt[n_format_fields].fmt_litstr = savestr(lit_str);
		}
		n_format_fields++;
	}
	if( n_format_fields >= MAX_FORMAT_FIELDS && *s != 0 ){
		sprintf(error_string,
	"Max number of format fields (%d) used up before done processing format string!?",
			MAX_FORMAT_FIELDS);
		WARN(error_string);
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
	s=NAMEOF(input_fmt[curr_fmt_i].fmt_litstr);			\
	if( strcmp(s,input_fmt[curr_fmt_i].fmt_litstr) ){		\
		sprintf(error_string,					\
	"expected literal string \"%s\", saw string \"%s\"",		\
			input_fmt[curr_fmt_i].fmt_litstr,s);		\
		WARN(error_string);					\
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
		switch( input_fmt[curr_fmt_i].fmt_type ){
			case IN_FMT_LIT:
				READ_LITERAL;
				break;
			case IN_FMT_STR: s=NAMEOF("don't-care string"); break;
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
						sprintf(error_string,
							"Float format data assigned to integer object %s!?",
							ascii_data_dp->dt_name);
						WARN(error_string);
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
		switch( input_fmt[curr_fmt_i].fmt_type ){
			case IN_FMT_LIT: READ_LITERAL; break;
			case IN_FMT_STR: s=NAMEOF("don't-care string"); break;
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
	double d;
	int havit=0;
	int done=0;

	do {
		switch( input_fmt[curr_fmt_i].fmt_type ){
			case IN_FMT_LIT: READ_LITERAL; break;
			case IN_FMT_STR:
				if( havit ) SET_DONE
				else {
					s=NAMEOF(pmpt);
					havit=1;
				}
				break;
			case IN_FMT_FLT:
				d = HOW_MUCH("don't care float number");
				break;
			case IN_FMT_INT:
				d = HOW_MANY("don't care integer number");
				break;
		}
		NEXT_FORMAT;
	} while(!done);

	return(s);
}

static int check_input_level(SINGLE_QSP_ARG_DECL)
{
	if( tell_qlevel(SINGLE_QSP_ARG) != ASCII_LEVEL ){
		sprintf(error_string,"check_input_level (ascii):  input depth is %d, expected %d!?",
			tell_qlevel(SINGLE_QSP_ARG),ASCII_LEVEL);
		WARN(error_string);
		advise("premature end of data");
		sprintf(error_string,"%d elements read so far",n_gotten);
		advise(error_string);
		if( n_format_fields > 0 ){
			prt_msg_frag("input_format:  ");
			show_input_format();
		}
		return(-1);
	}
	return 0;
}

static int get_a_string(QSP_ARG_DECL  Data_Obj *dp,char *datap)
{
	const char *s;
	char *t;
	dimension_t i;

	if( check_input_level(SINGLE_QSP_ARG) < 0 ) return(-1);

	/* see if we need to look at the input format string */
	if( n_format_fields == 0 )
		s = NAMEOF("string data");
	else
		s = next_input_str(QSP_ARG  "string data");

	/* FIXME should use strncpy() here */
	t=datap;
	i=0;
	while( *s && i < dp->dt_cols ){
		*t = *s;
		t += dp->dt_pinc;
		s++;
		i++;
	}
	if( i < dp->dt_cols )
		*t = 0;

	/* now lookahead to pop the file if it is empty */
	lookahead_til(QSP_ARG  ASCII_LEVEL-1);

	return(0);
}

static int get_next(QSP_ARG_DECL   Data_Obj *dp,void *datap)
{
	/* init these to eliminate optimizer warnings */
	double d_number=0.0;
	long l=0L;
	mach_prec mp;

	static Data_Obj *warned_dp=NO_OBJ;

	if( check_input_level(SINGLE_QSP_ARG) < 0 ) return(-1);

#ifdef CAUTIOUS
	if( MACHINE_PREC(dp) > N_MACHINE_PRECS ){
		sprintf(error_string,
	"CAUTIOUS:  get_next:  Object %s precision %s is not a machine precision!?",
			dp->dt_name,prec_name[MACHINE_PREC(dp)]);
			WARN(error_string);
		return(-1);
	}
#endif /* CAUTIOUS */

#ifdef DEBUG
if( debug & debug_data ){
sprintf(error_string,"get_next:  getting a %s value for address 0x%lx",
prec_name[MACHINE_PREC(dp)],(u_long)datap);
advise(error_string);
}
#endif /* DEBUG */

	mp = MACHINE_PREC(dp);
	switch( mp ){
		case PREC_DP:  case PREC_SP:
			if( n_format_fields == 0 )
				d_number = HOW_MUCH("real data");
			else
				d_number = next_input_flt(QSP_ARG  "real data");
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
			break;
		case PREC_NONE:
			sprintf(error_string,"get_next:  object %s has no data!?",dp->dt_name);
			WARN(error_string);
			return(-1);
			break;
#ifdef CAUTIOUS
		case N_MACHINE_PRECS:	/* have this case here to silence compiler */
			ERROR1("bad case in get_next");
			break;
		/* default: ERROR1("CAUTIOUS:  get_next, bad machine precision"); break; */
#endif /* CAUTIOUS */
	}

	switch( mp ){
		case PREC_DP:
			* ((double *)datap) =d_number;
			break;
		case PREC_SP:
			* ((float *)datap) =(float)d_number;
			break;
		case PREC_BY:
			if( (l < -128 || l > 127) && warned_dp!=dp ){
				sprintf(error_string,
			"data out of range for byte conversion, object %s",
			dp->dt_name);
				WARN(error_string);
				warned_dp=dp;
			}
			*(char *)datap = (char)(l);
			break;
		case PREC_UBY:
			if( (l < 0 || l > 255) && warned_dp!=dp ){
				sprintf(error_string,
			"data out of range for unsigned byte conversion, object %s",
			dp->dt_name);
				WARN(error_string);
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
			if( (l < MIN_SIGNED_SHORT || l > MAX_SIGNED_SHORT )
					&& warned_dp!=dp ){

				sprintf(error_string,
		"number %ld (0x%lx) won't fit in a signed short, object %s",
					l,l,dp->dt_name);
				WARN(error_string);
				warned_dp=dp;
			}
			* ((short *)datap)=(short)l;
			break;
		case PREC_UIN:
			if( (l < MIN_UNSIGNED_SHORT || l > MAX_UNSIGNED_SHORT )
					&& warned_dp!=dp ){

				sprintf(error_string,
		"number %ld (0x%lx) won't fit in an unsigned short, object %s",
					l,l,dp->dt_name);
				WARN(error_string);
				warned_dp=dp;
			}
			* ((u_short *)datap)=(u_short)l;
			break;
		case PREC_DI:  * ((int32_t *)datap)=(int32_t)l; break;
		case PREC_UDI:
			if( IS_BITMAP(dp) ){
				long offset;
				int bit;
				/* BUG - here we are assuming that the bitmap word type is PREC_UDI,
				 * so we are right shifting by 5 instead of LOG2_BITS_PER_BITMAP_WORD...
				 * BUT we are using BITMAP_DATA_TYPE!?
				 */
				/* We have faked datap to get the bit offset */
				offset = ((BITMAP_DATA_TYPE *)datap) - ((BITMAP_DATA_TYPE *)dp->dt_data);
				/* offset is in bits... */
				datap = ((BITMAP_DATA_TYPE *)dp->dt_data) + ((dp->dt_bit0 + offset)>>5);
				bit = (dp->dt_bit0+offset)&BIT_NUMBER_MASK;
				/* We used to have 1<<bit here, but that gave 0 for more than 32 bits.
				 * Is that because the compiler treats the 1 as a 32 bit number?
				 */
				if( l==1 )
					*((BITMAP_DATA_TYPE *)datap) |= NUMBERED_BIT(bit);

				else if( l == 0 )
					*((BITMAP_DATA_TYPE *)datap) &= ~NUMBERED_BIT(bit);
				else {
					sprintf(error_string,"Non-boolean value %ld specified for bitmap %s!?",
						l,dp->dt_name);
					WARN(error_string);
				}
			} else {
				* ((uint32_t *)datap)=(uint32_t)l;
			}
			break;

		case PREC_LI:  * ((int64_t *)datap)=(int64_t)l; break;
		case PREC_ULI:
			if( IS_BITMAP(dp) ){
				long offset;
				int bit;
				/* hard coded to 6 instead of LOG2_BITS_PER_BITMAP_WORD? BUG? */
				offset = ((BITMAP_DATA_TYPE *)datap) - ((BITMAP_DATA_TYPE *)dp->dt_data);
				datap = ((BITMAP_DATA_TYPE *)dp->dt_data) + ((dp->dt_bit0 + offset)>>6);
				bit = (dp->dt_bit0+offset)&BIT_NUMBER_MASK;
				if( l==1 )
					*((BITMAP_DATA_TYPE *)datap) |= (1<<bit);
				else if( l == 0 )
					*((BITMAP_DATA_TYPE *)datap) &= ~(1<<bit);
				else {
					sprintf(error_string,"Non-boolean value %ld specified for bitmap %s!?",
						l,dp->dt_name);
					WARN(error_string);
				}
			} else {
				* ((uint64_t *)datap)=(uint64_t)l;
			}
			break;

#ifdef CAUTIOUS
		case PREC_NONE:		/* should have been handled above */
			break;
#endif /* CAUTIOUS */

#ifdef CAUTIOUS
		case N_MACHINE_PRECS:	/* just to silence compiler */
			ERROR1("bad case in get_next");
			break;
		/*
		default:
			WARN("CAUTIOUS:  get_next:  unexpected pseudo-precision");
			break;
			*/
#endif /* CAUTIOUS */
	}
	n_gotten++;

	/* now lookahead to pop the file if it is empty */
	lookahead_til(QSP_ARG  ASCII_LEVEL-1);

	return(0);
} /* end get_next() */

#ifdef FOOBAR
static int64_t get_bit_from_bitmap(Data_Obj *dp, void *data)
{
	int which_bit;
	int64_t l;

	/* if it's a bitmap, we don't really have the address of
	 * the data, we need to get the bit offset...
	 */

	/* We encode the bit in the data address - it's not the real address. */
	which_bit = ((u_char *)data) - ((u_char *)dp->dt_data);
sprintf(error_string,"get_bit_from_bitmap:  which_bit = %d",which_bit);
advise(error_string);
	which_bit += dp->dt_bit0;

	/* now we know which bit it is, but we still need to figure
	 * out which word...  If this is a subobject of a bit image,
	 * we are ok, because the ultimate ancestor will be contiguous,
	 * but we can get in trouble if it is an equivalence of
	 * a non-contiguous object...  not sure how to handle this?
	 */

	data = ((BITMAP_DATA_TYPE *)dp->dt_data) +
		(which_bit>>LOG2_BITS_PER_BITMAP_WORD);
	l= (int64_t)(* (BITMAP_DATA_TYPE *) data );
	which_bit &= BIT_NUMBER_MASK;

	if( l & (1L<<which_bit) )
		l=1;
	else
		l=0;
	return(l);
}
#endif /* FOOBAR */


/*
 * Format one (possibly multidimensional) pixel value
 *
 * It seems we are confused about what to do about bitmaps - BUG?
 */

void format_scalar_obj(char *buf,Data_Obj *dp,void *data)
{
	//int64_t l;
	int c;

	if( dp->dt_prec == PREC_CHAR || dp->dt_prec == PREC_STR ){
		c=(*(char *)data);
		if( isalnum(c) || ispunct(c) )
			sprintf(buf,"'%c'",c);
		else
			sprintf(buf,"0%o",c);
		return;
	}

	if( ! IS_BITMAP(dp) ){
		format_scalar_value(buf,data,dp->dt_prec);
	}
	/*
	else {
		l = get_bit_from_bitmap(dp,data);
		sprintf(buf,ifmtstr,l);
	}
	*/
}

void format_scalar_value(char *buf,void *data,prec_t prec)
{
	mach_prec mp;
	double ddata;
	int64_t l;

	mp = (mach_prec)(prec & MACH_PREC_MASK);
	switch( mp ){

#ifdef CAUTIOUS
		case PREC_NONE:
			NERROR1("CAUTIOUS:  format_scalar_value:  null precision!?");
			break;
#endif /* CAUTIOUS */

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
			sprintf(buf,ffmtstr,ddata);
			break;

pntlng:
			sprintf(buf,ifmtstr,l);
			break;

#ifdef CAUTIOUS
		case N_MACHINE_PRECS:	/* silence compiler */
			NERROR1("CAUTIOUS:  format_scalar_value:  bad machine precision");
			break;
		/* default: ERROR1("CAUTIOUS:  format_scalar_value:  unknown prec"); break; */
#endif /* CAUTIOUS */
	}
}

/*
 * Print one (possibly multidimensional) pixel value
 */

static void pnt_one( FILE *fp, Data_Obj *dp,  u_char *data )
{
	char buf[128];

	if( dp->dt_mach_dim[0] > 1 ){	/* not real, complex or higher */
		if( IS_STRING(dp) ){
			fprintf(fp,"%s",data);
		} else {
			incr_t inc;
			dimension_t j;

			inc = (ELEMENT_SIZE(dp) * dp->dt_mach_inc[0]);
			for(j=0;j<dp->dt_mach_dim[0];j++){
				/* WHen this code was written, we didn't anticipate
				 * having too many components, so ret_dim=0 means
				 * print a return after each pixel.  But if
				 * we have many many components, then the resulting
				 * lines can be too long!?  Fixing this here is
				 * kind of a hack...
				 */
				if( j>0 && dp->dt_mach_dim[0] > max_per_line ){
					fprintf(fp,"\n");
				}
				format_scalar_obj(buf,dp,data);
				fprintf(fp," %s",buf);
				data += inc;
			}
		}
	} else {
		format_scalar_obj(buf,dp,data);
		fprintf(fp,"%s%s",separator,buf);
	}
} /* end pnt_one */

static void pnt_dim( FILE *fp, Data_Obj *dp, unsigned char *data, int dim )
{
	dimension_t i;
	incr_t inc;

#ifdef CAUITOUS
	if( IS_BITMAP(dp) ) ERROR1("CAUTIOUS:  pnt_dim called with bitmap argument.");
#endif /* CAUTIOUS */

	if( dim > 0 ){
		if( dim==1 && dp->dt_prec == PREC_STR ){
			/* here we use .* to put the max number of chars to print
			 * in the next arg (is this a gcc extension or standard?).
			 * We do this because we are not guaranteed a null terminator char.
			 */
			fprintf(fp,"%.*s\n",(int)dp->dt_cols,data);
			return;
		}
		inc=(ELEMENT_SIZE(dp)*dp->dt_mach_inc[dim]);
#ifdef DEBUG
if( debug & debug_data ){
sprintf(DEFAULT_ERROR_STRING,"pntdim: dim=%d, n=%d, inc=%d",dim,dp->dt_mach_dim[dim],
inc);
advise(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */

		for(i=0;i<dp->dt_mach_dim[dim];i++)
			pnt_dim(fp,dp,data+i*inc,dim-1);
	} else {
		pnt_one(fp,dp,data);
	}
	if( dim == ret_dim ) fprintf(fp,"\n");
}

/* This is a speeded-up version for floats - do we need it? */

static void sp_pntvec( Data_Obj *dp, FILE *fp )
{
	float *base, *fbase, *rbase, *pbase;
	dimension_t i3,i2,i1,i0;

	base = (float *) dp->dt_data;

	for(i3=0;i3<dp->dt_frames;i3++){
	    fbase=base+i3*dp->dt_mach_inc[3];
	    for(i2=0;i2<dp->dt_rows;i2++){
		rbase=fbase+i2*dp->dt_mach_inc[2];
		for(i1=0;i1<dp->dt_cols;i1++){
		    pbase=rbase+i1*dp->dt_mach_inc[1];
		    for(i0=0;i0<dp->dt_mach_dim[0];i0++){
			fprintf(fp," ");
			fprintf(fp,ffmtstr,*(pbase+i0*dp->dt_mach_inc[0]));
		        if( ret_dim == 0 && dp->dt_mach_dim[0] > max_per_line )
				fprintf(fp,"\n");
		    }
		    if( ret_dim == 0 && dp->dt_mach_dim[0] <= max_per_line )
		    	fprintf(fp,"\n");
		}
		if( ret_dim > 0 ) fprintf(fp,"\n");
	    }
	}
}

static void set_pad_ffmt_str(void)
{
	sprintf(pad_ffmtstr,"%%%d.%dg",min_field_width,display_precision);
	ffmtstr = pad_ffmtstr;
}

void set_min_field_width(int fw)
{
	min_field_width=fw;
}

void set_display_precision(int digits)
{
	display_precision=digits;
}

#define MAX_BITS_PER_LINE 32

static void display_bitmap(Data_Obj *dp, FILE *fp)
{
	int i,j,k,l,m;
	bitmap_word *bwp,val;
	int which_bit, bit_index, word_offset;
	int bits_this_line;

	bwp = (bitmap_word *)dp->dt_data;

	bits_this_line=0;
	for(i=0;i<dp->dt_seqs;i++){
		for(j=0;j<dp->dt_frames;j++){
			for(k=0;k<dp->dt_rows;k++){
				for(l=0;l<dp->dt_cols;l++){
					for(m=0;m<dp->dt_comps;m++){
						which_bit = dp->dt_bit0
							+ m * dp->dt_type_inc[0]
							+ l * dp->dt_type_inc[1]
							+ k * dp->dt_type_inc[2]
							+ j * dp->dt_type_inc[3]
							+ i * dp->dt_type_inc[4];
						bit_index = which_bit % BITS_PER_BITMAP_WORD;
						word_offset = which_bit/BITS_PER_BITMAP_WORD;
						val = *(bwp + word_offset) & NUMBERED_BIT(bit_index); 
						if( bits_this_line >= MAX_BITS_PER_LINE ){
							bits_this_line=0;
							prt_msg("");
						} else if( bits_this_line > 0 )
							prt_msg_frag(" ");
						prt_msg_frag(val?"1":"0");
						bits_this_line++;
					}
				}
				if( bits_this_line > 0 ){
					bits_this_line=0;
					prt_msg("");
				}
			}
		}
	}
}

void pntvec(QSP_ARG_DECL  Data_Obj *dp,FILE *fp)			/**/
{
	const char *save_ifmt;
	const char *save_ffmt;

	/* first let's figure out when to print returns */
	/* ret_dim == 0 means a return is printed after every pixel */
	/* ret_dim == 1 means a return is printed after every row */

	if( dp->dt_mach_dim[0] == 1 && dp->dt_cols <= max_per_line )
		ret_dim=1;
	else ret_dim=0;

	/* We pad with spaces only if we are printing more than one element */

	save_ffmt=ffmtstr;
	save_ifmt=ifmtstr;
	/* BUG should set format based on desired radix !!! */
	padflag = 1;
	set_integer_print_fmt(QSP_ARG   THE_FMT_CODE);	/* handles integer formats */
	set_pad_ffmt_str();

	if( MACHINE_PREC(dp) == PREC_SP )
		sp_pntvec(dp,fp);
	else if( dp->dt_prec == PREC_BIT )
		display_bitmap(dp,fp);
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

		pnt_dim(fp,dp,(u_char *)dp->dt_data,N_DIMENSIONS-1);
	}
	fflush(fp);

	ffmtstr = save_ffmt ;
	ifmtstr = save_ifmt ;
}

void shp_trace(const char *name,Shape_Info *shpp)
{
	sprintf(DEFAULT_ERROR_STRING,
		"%s: mindim = %d,  maxdim = %d",
		name, shpp->si_mindim, shpp->si_maxdim);
	advise(DEFAULT_ERROR_STRING);

	sprintf(DEFAULT_ERROR_STRING,
		"%s dim:  %u %u %u %u %u",
		name,
		shpp->si_type_dim[0],
		shpp->si_type_dim[1],
		shpp->si_type_dim[2],
		shpp->si_type_dim[3],
		shpp->si_type_dim[4]);
	advise(DEFAULT_ERROR_STRING);
}

void dptrace( Data_Obj *dp )
{
	shp_trace(dp->dt_name,&dp->dt_shape);

	sprintf(DEFAULT_ERROR_STRING,
		// why %u format when increment can be negative???
		"%s inc:  %u %u %u %u %u  (%u %u %u %u %u)",
		dp->dt_name,
		dp->dt_type_inc[0],
		dp->dt_type_inc[1],
		dp->dt_type_inc[2],
		dp->dt_type_inc[3],
		dp->dt_type_inc[4],
		dp->dt_mach_inc[0],
		dp->dt_mach_inc[1],
		dp->dt_mach_inc[2],
		dp->dt_mach_inc[3],
		dp->dt_mach_inc[4]);
	advise(DEFAULT_ERROR_STRING);
}

static int get_strings(QSP_ARG_DECL  Data_Obj *dp,char *data,int dim)
{
	int status=0;

	if( dim < 2 ){	/* get a row */
		return( get_a_string(QSP_ARG  dp,data) );
	} else {
		dimension_t i;
		long offset;

		offset = ELEMENT_SIZE( dp);
		offset *= dp->dt_mach_inc[dim];
		for(i=0;i<dp->dt_mach_dim[dim];i++){
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
			offset *= dp->dt_type_inc[dim];
			for(i=0;i<dp->dt_type_dim[dim];i++){
				status = get_sheets(QSP_ARG  dp,data+i*offset,dim-1);
				if( status < 0 ) return(status);
			}
		} else {
			offset *= dp->dt_mach_inc[dim];
			for(i=0;i<dp->dt_mach_dim[dim];i++){
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

	push_input_file(QSP_ARG  s);
	redir(QSP_ARG  fp);

	/* BUG we'd like to have the string be 'Pipe: "command args"' or something... */
	if( !strncmp(s,"Pipe",4) ){
		THIS_QSP->qs_query[QLEVEL].q_flags |= Q_PIPE;
	}

	level = tell_qlevel(SINGLE_QSP_ARG);

	read_obj(QSP_ARG  dp);

	if( level == tell_qlevel(SINGLE_QSP_ARG) ){
		if( expect_exact_count ){
			sprintf(error_string,
				"Needed %d values for object %s, file %s has more!?",
				dp->dt_n_mach_elts,dp->dt_name,orig_filename);
			WARN(error_string);
		}
		popfile(SINGLE_QSP_ARG);
	}

	rls_str(orig_filename);
}

void read_obj(QSP_ARG_DECL   Data_Obj *dp)
{
	ASCII_LEVEL = tell_qlevel(SINGLE_QSP_ARG);
	n_gotten = 0;

	if( dp != ascii_data_dp ){
		/* We store the target object so we can print its name if we need to... */
		ascii_data_dp = dp;		/* if this is global, then this is not thread-safe!? */
		ascii_warned=0;
	}

	if( dp->dt_prec == PREC_CHAR || dp->dt_prec == PREC_STR ){
		if( get_strings(QSP_ARG  dp,(char *)dp->dt_data,N_DIMENSIONS-1) < 0 ){
			sprintf(error_string,"error reading strings for object %s",dp->dt_name);
			WARN(error_string);
		}
	} else if( get_sheets(QSP_ARG  dp,(u_char *)dp->dt_data,N_DIMENSIONS-1) < 0 ){
		/*
		sprintf(error_string,"error reading ascii data for object %s",dp->dt_name);
		WARN(error_string);
		*/
		sprintf(error_string,"expected %d elements for object %s",
			dp->dt_n_mach_elts,dp->dt_name);
		WARN(error_string);
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
#ifdef CAUTIOUS
		default:
			ERROR1("CAUTIOUS:  unrecognized format code");
#endif /* CAUTIOUS */
	}
}

void set_max_per_line(QSP_ARG_DECL  int n )
{
	if( n < 1 )
		WARN("max_per_line must be positive");
	else if( n > ENFORCED_MAX_PER_LINE ){
		sprintf(error_string,"Requested max_per_line (%d) exceeds hard-coded maximum (%d)",
				n,ENFORCED_MAX_PER_LINE);
		WARN(error_string);
	} else
		max_per_line = n;
}

