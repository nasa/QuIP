#include "quip_config.h"

#include <stdio.h>
#include <string.h>
#include "data_obj.h"
//#include "dataprot.h"
#include "debug.h"
#include "query_stack.h"
#include "variable.h"
#include "ascii_fmts.h"
#include "warn.h"
#include "quip_prot.h"
#include "getbuf.h"
#include "veclib_api.h"	// BUG should integrate into data_obj.h...

//#include "menuname.h"

// BUG should be per-thread variable...
static int expect_exact_count=1;

#define DELETE_IF_COPY(dp)						\
									\
if( (OBJ_FLAGS(dp) & DT_VOLATILE) && (OBJ_FLAGS(dp) & DT_TEMP) == 0 )	\
	delvec(QSP_ARG  dp);

#define DNAME_PREFIX "downloaded_"
#define CNAME_PREFIX "continguous_"

/*static*/ Data_Obj *insure_ram_obj(QSP_ARG_DECL  Data_Obj *dp)
{
	Data_Obj *tmp_dp;
	char *tname;
	Data_Area *save_ap;
	Data_Obj *c_dp=NULL;

	if( OBJ_IS_RAM(dp) ) return dp;

	// This object lives on a different platform.
	// We create a copy in RAM, and download the data
	// using the platform download function.

	save_ap = curr_ap;
	curr_ap = ram_area_p;

	tname = getbuf( strlen(OBJ_NAME(dp)) + strlen(DNAME_PREFIX) + 1 );
	sprintf(tname,"%s%s",DNAME_PREFIX,OBJ_NAME(dp));
	tmp_dp = dup_obj(QSP_ARG  dp, tname);
	givbuf(tname);
	if( tmp_dp == NULL ){
		// This can happen if the object is subscripted,
		// as the bracket characters are illegal in names
		return NULL;
	}

	curr_ap = save_ap;

	// We can't download if the source data is not contiguous...
	//
	// OLD OBSOLETE COMMENT:
	// We have a problem with bit precision, because the bits can
	// be non-contiguous when the long words are - any time the number of columns
	// is not evenly divided by the bits-per-word
	//
	// Now, bitmaps wrap bits around, not trying to align word and row boundaries

	if( (! IS_CONTIGUOUS(dp)) && ! HAS_CONTIGUOUS_DATA(dp) ){
		Vec_Obj_Args oa1, *oap=&oa1;

advise("object is not contiguous, and does not have contiguous data, creating temp object for copy...");
longlist(QSP_ARG  dp);
		save_ap = curr_ap;
		curr_ap = OBJ_AREA( dp );

		tname = getbuf( strlen(OBJ_NAME(dp)) + strlen(CNAME_PREFIX) + 1 );
		sprintf(tname,"%s%s",CNAME_PREFIX,OBJ_NAME(dp));
		c_dp = dup_obj(QSP_ARG  dp, tname );
		givbuf(tname);

		curr_ap = save_ap;

		// Now do the move...

		setvarg2(oap,c_dp,dp);
		if( IS_BITMAP(dp) ){
			SET_OA_SBM(oap,dp);
			SET_OA_SRC1(oap,NULL);
		}

		if( IS_REAL(dp) ) /* BUG case for QUAT too? */
			OA_ARGSTYPE(oap) = REAL_ARGS;
		else if( IS_COMPLEX(dp) ) /* BUG case for QUAT too? */
			OA_ARGSTYPE(oap) = COMPLEX_ARGS;
		else if( IS_QUAT(dp) ) /* BUG case for QUAT too? */
			OA_ARGSTYPE(oap) = QUATERNION_ARGS;
		else
			assert( AERROR("insure_ram_obj:  bad argset type!?") );

//fprintf(stderr,"insure_ram_obj:  moving remote data to a contiguous object\n");  
		call_vfunc( QSP_ARG  FIND_VEC_FUNC(FVMOV), oap );
//fprintf(stderr,"insure_ram_obj:  DONE moving remote data to a contiguous object\n");  

		dp = c_dp;
	}

	gen_obj_dnload(QSP_ARG  tmp_dp, dp);

	if( c_dp != NULL )
		delvec(QSP_ARG  c_dp);

	// BUG - when to delete?
	// We try using the VOLATILE flag.  This will work as long as
	// the input object is not VOLATILE!?

	SET_OBJ_FLAG_BITS(tmp_dp, DT_VOLATILE ) ;

	return tmp_dp;
}


/*
 * BUG do_read_obj will not work correctly for subimages
 * (or will it with the new revision?
 */

static COMMAND_FUNC( do_read_obj )
{
	Data_Obj *dp;
	FILE *fp;
	const char *s;

	dp=PICK_OBJ("");
	s=NAMEOF("input file");

	if( dp == NULL ) return;

#ifdef QUIP_DEBUG
//if( debug ) dptrace(dp);
#endif /* QUIP_DEBUG */

	// reading is tricker for non-ram, because
	// we must create the copy, then read into
	// the copy, then xfer to the device...

	INSIST_RAM_OBJ(dp,"do_read_obj")

	if( strcmp(s,"-") && strcmp(s,"stdin") ){
		fp=TRY_OPEN( s, "r" );
		if( !fp ) return;

		read_ascii_data(QSP_ARG  dp,fp,s,expect_exact_count);
	} else {
		/* read from stdin, no problem... */

		read_obj(QSP_ARG  dp);
	}
}

static COMMAND_FUNC( do_pipe_obj )
{
	Data_Obj *dp;
	Pipe *pp;
	char cmdbuf[LLEN];

	dp=PICK_OBJ("");
	pp=PICK_PIPE("readable pipe");

	if( dp == NULL ) return;
	if( pp == NULL ) return;

	// reading is tricker for non-ram, because
	// we must create the copy, then read into
	// the copy, then xfer to the device...

	INSIST_RAM_OBJ(dp,"pipe_read_obj")

	sprintf(cmdbuf,"Pipe:  %s",pp->p_cmd);
	read_ascii_data(QSP_ARG  dp,pp->p_fp,cmdbuf,expect_exact_count);
	/* If there was just enough data, then the pipe
	 * will have been closed already... */

	/* BUG we should check qlevel to make sure that the pipe was popped... */
	pp->p_fp = NULL;
}

static COMMAND_FUNC( do_set_var_from_obj )
{
	Data_Obj *dp;
	const char *s;

	s=NAMEOF("variable");
	dp=PICK_OBJ("");

	if( dp == NULL ) return;

	dp = insure_ram_obj(QSP_ARG  dp);
	if( dp == NULL ) return;

	if( ! IS_STRING(dp) ){
		sprintf(ERROR_STRING,"do_set_var_from_obj:  object %s (%s) does not have string precision",
			OBJ_NAME(dp),OBJ_PREC_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}

	ASSIGN_VAR(s,(char *)OBJ_DATA_PTR(dp));

	DELETE_IF_COPY(dp)
}

static COMMAND_FUNC( do_set_obj_from_var )
{
	Data_Obj *dp;
	const char *s;

	dp=PICK_OBJ("");
	s=NAMEOF("string");

	if( dp == NULL ) return;

	INSIST_RAM_OBJ(dp,"set_string")

#ifdef QUIP_DEBUG
//if( debug ) dptrace(dp);
#endif /* QUIP_DEBUG */

	if( ! IS_STRING(dp) ){
		sprintf(ERROR_STRING,"do_set_obj_from_var:  object %s (%s) does not have string precision",
			OBJ_NAME(dp),OBJ_PREC_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}

	if( (strlen(s)+1) > OBJ_COMPS(dp) ){
		sprintf(ERROR_STRING,
	"Type dimension (%d) of string object %s is too small for string of length %d",
			OBJ_COMPS(dp),OBJ_NAME(dp),(int)strlen(s));
		WARN(ERROR_STRING);
		return;
	}
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(ERROR_STRING,"Sorry, object %s must be contiguous for string reading",
			OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}

	strcpy((char *)OBJ_DATA_PTR(dp),s);
}

static COMMAND_FUNC( do_disp_obj )
{
	Data_Obj *dp;
	FILE *fp;

	dp=PICK_OBJ("");
	if( dp==NULL ) return;

	// We used to insist that the object be in RAM,
	// but we make life easier by automatically creating
	// a temporary object...

	dp = insure_ram_obj(QSP_ARG  dp);
	if( dp == NULL ) return;

	fp = tell_msgfile(SINGLE_QSP_ARG);
	if( fp == stdout ){
		if( IS_IMAGE(dp) || IS_SEQUENCE(dp) )
			if( !CONFIRM(
		"are you sure you want to display an image/sequence in ascii") )
				return;
		list_dobj(QSP_ARG  dp);
	}
	pntvec(QSP_ARG  dp,fp);
	fflush(fp);

	DELETE_IF_COPY(dp)
}

/* BUG wrvecd will not work correctly for subimages */

/* Now that we can redirect the output file, and do_disp_obj prints
 * to msgfile, we really don't need this... maybe delete it later.
 */

static COMMAND_FUNC( do_wrt_obj )
{
	Data_Obj *dp;
	FILE *fp;
	/* BUG what if pathname is longer than 256??? */
	const char *filename;

	dp=PICK_OBJ("");
	filename = NAMEOF("output file");

	if( dp==NULL ) return;

	if( strcmp(filename,"-") && strcmp(filename,"stdout") ){
		// BUG? we don't check append flag here,
		// but there is a separate append command...

		fp=TRYNICE( filename, "w" );
		if( !fp ) return;
	} else {
		// If the invoking script has redirected stdout,
		// then use that
		if( QS_MSG_FILE(THIS_QSP)!=NULL )
			fp = QS_MSG_FILE(THIS_QSP);
		else
			fp = stdout;
	}

	if( IS_IMAGE(dp) || IS_SEQUENCE(dp) )
		if( !CONFIRM(
		"are you sure you want to write an image/sequence in ascii") ){
			fclose(fp);
			return;
		}

	dp = insure_ram_obj(QSP_ARG  dp);
	if( dp == NULL ) return;

	pntvec(QSP_ARG  dp,fp);
	if( fp != stdout && QS_MSG_FILE(THIS_QSP)!=NULL && fp != QS_MSG_FILE(THIS_QSP) ) {
		if( verbose ){
			sprintf(MSG_STR,"closing file %s",filename);
			prt_msg(MSG_STR);
		}
		fclose(fp);
	}

	DELETE_IF_COPY(dp)
}

static COMMAND_FUNC( do_append )
{
	Data_Obj *dp;
	FILE *fp;

	dp=PICK_OBJ("");
	if( dp==NULL ) return;

	if( IS_IMAGE(dp) || IS_SEQUENCE(dp) )
		if( !CONFIRM(
		"are you sure you want to write an image/sequence in ascii") )
			return;
	fp=TRYNICE( NAMEOF("output file"), "a" );
	if( !fp ) return;

	dp = insure_ram_obj(QSP_ARG  dp);
	if( dp == NULL ) return;

	pntvec(QSP_ARG  dp,fp);
	fclose(fp);

	DELETE_IF_COPY(dp)
}

static const char *print_fmt_name[N_PRINT_FORMATS];
static int fmt_names_inited=0;

static void init_print_fmt_names(void)
{
	int i;

	assert( ! fmt_names_inited );

	/* BUG should insure that all formats are inited */
	for(i=0;i<N_PRINT_FORMATS;i++){
		switch(i){
			case FMT_POSTSCRIPT:
				print_fmt_name[i] = "postscript";
				break;
			case FMT_DECIMAL:
				print_fmt_name[i] = "decimal";
				break;
			case FMT_FLOAT:
				print_fmt_name[i] = "float";
				break;
			case FMT_HEX:
				print_fmt_name[i] = "hexadecimal";
				break;
			case FMT_OCTAL:
				print_fmt_name[i] = "octal";
				break;
			case FMT_UDECIMAL:
				print_fmt_name[i] = "unsigned_decimal";
				break;
			default:
				assert( AERROR("Missing format initialization!?") );
				break;
		}
	}
	fmt_names_inited=1;
}

static COMMAND_FUNC( do_set_fmt )
{
	int i;

	if( ! fmt_names_inited ) init_print_fmt_names();

	i = WHICH_ONE("format",N_PRINT_FORMATS,print_fmt_name);
	if( i < 0 ) return;
	set_integer_print_fmt(QSP_ARG  (Number_Fmt)i);
}

static COMMAND_FUNC( do_set_max )
{
	set_max_per_line( QSP_ARG  (int) HOW_MANY("max number of items per line") );
}

static COMMAND_FUNC( do_set_in_fmt )
{
	const char *s;

	s=NAMEOF("input line format string");
	set_input_format_string(QSP_ARG  s);
}

static COMMAND_FUNC( do_exact )
{
	expect_exact_count = ASKIF("expect ascii files to contain exactly the right number of elements");
}

static COMMAND_FUNC( do_set_digits )
{
	int d;
	d = (int)HOW_MANY("number of significant digits to print");
	set_display_precision(QSP_ARG  d);
}

#define ADD_CMD(s,f,h)	ADD_COMMAND(ascii_menu,s,f,h)

MENU_BEGIN(ascii)

ADD_CMD( display,	do_disp_obj,	display data	)
ADD_CMD( read,		do_read_obj,	read vector data from ascii file	)
ADD_CMD( pipe_read,	do_pipe_obj,	read vector data from named pipe	)
ADD_CMD( set_string,	do_set_obj_from_var,	read vector data from a string	)
ADD_CMD( get_string,	do_set_var_from_obj,	set a script variable to the value of a stored string	)
ADD_CMD( write,		do_wrt_obj,	write vector data to ascii file	)
ADD_CMD( append,	do_append,	append vector data to ascii file	)
ADD_CMD( exact,		do_exact,	enable/disable warnings for file/vector length mismatch	)
ADD_CMD( output_fmt,	do_set_fmt,	set data print format	)
ADD_CMD( input_fmt,	do_set_in_fmt,	specify format of input line	)
ADD_CMD( max_per_line,	do_set_max,	set maximum number of items per line	)
ADD_CMD( digits,	do_set_digits,	set number of significant digits to print	)

MENU_END(ascii)

COMMAND_FUNC( asciimenu )			/** ascii object submenu */
{
	PUSH_MENU(ascii);
}

