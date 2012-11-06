#include "quip_config.h"

char VersionId_datamenu_ascmenu[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <string.h>
#include "data_obj.h"
#include "dataprot.h"
#include "debug.h"
#include "query.h"
#include "menuname.h"

static COMMAND_FUNC( do_read_obj );
static COMMAND_FUNC( do_pipe_obj );
static COMMAND_FUNC( do_disp_obj );
static COMMAND_FUNC( do_wrt_obj );
static COMMAND_FUNC( do_append );

static int expect_exact_count=1;

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

	if( dp == NO_OBJ ) return;

#ifdef DEBUG
//if( debug ) dptrace(dp);
#endif /* DEBUG */

	INSIST_RAM(dp,"do_read_obj")

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

	if( dp == NO_OBJ ) return;
	if( pp == NO_PIPE ) return;

	INSIST_RAM(dp,"pipe_read_obj")

	sprintf(cmdbuf,"Pipe:  %s",pp->p_cmd);
	read_ascii_data(QSP_ARG  dp,pp->p_fp,cmdbuf,expect_exact_count);
	/* If there was just enough data, then the pipe
	 * will have been closed already... */

	/* BUG we should check qlevel to make sure that the pipe was popped... */
	pp->p_fp = NULL;
}

static COMMAND_FUNC( do_wrt_str )
{
	Data_Obj *dp;
	const char *s;

	dp=PICK_OBJ("");
	s=NAMEOF("variable");

	if( dp == NO_OBJ ) return;

	INSIST_RAM(dp,"write_string")

	if( ! IS_STRING(dp) ){
		sprintf(error_string,"do_read_str:  object %s (%s) does not have string precision",
			dp->dt_name,name_for_prec(dp->dt_prec));
		WARN(error_string);
		return;
	}

	ASSIGN_VAR(s,(char *)dp->dt_data);
}

static COMMAND_FUNC( do_read_str )
{
	Data_Obj *dp;
	const char *s;

	dp=PICK_OBJ("");
	s=NAMEOF("string");

	if( dp == NO_OBJ ) return;

	INSIST_RAM(dp,"read_string")

#ifdef DEBUG
//if( debug ) dptrace(dp);
#endif /* DEBUG */

	if( ! IS_STRING(dp) ){
		sprintf(error_string,"do_read_str:  object %s (%s) does not have string precision",
			dp->dt_name,name_for_prec(dp->dt_prec));
		WARN(error_string);
		return;
	}

	if( (strlen(s)+1) > dp->dt_comps ){
		sprintf(error_string,
	"Type dimension (%d) of string object %s is too small for string of length %d",
			dp->dt_comps,dp->dt_name,(int)strlen(s));
		WARN(error_string);
		return;
	}
	if( ! IS_CONTIGUOUS(dp) ){
		sprintf(error_string,"Sorry, object %s must be contiguous for string reading",
			dp->dt_name);
		WARN(error_string);
		return;
	}

	strcpy((char *)dp->dt_data,s);
}

static COMMAND_FUNC( do_disp_obj )
{
	Data_Obj *dp;
	FILE *fp;

	dp=PICK_OBJ("");
	if( dp==NO_OBJ ) return;

	INSIST_RAM(dp,"display_obj")

	fp = tell_msgfile();
	if( fp == stdout ){
		if( IS_IMAGE(dp) || IS_SEQUENCE(dp) )
			if( !CONFIRM(
		"are you sure you want to display an image/sequence in ascii") )
				return;
		listone(dp);
	}
	pntvec(QSP_ARG  dp,fp);
	fflush(fp);
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

	if( dp==NO_OBJ ) return;

	INSIST_RAM(dp,"display_obj")

	if( strcmp(filename,"-") && strcmp(filename,"stdout") ){
		fp=TRYNICE( filename, "w" );
		if( !fp ) return;
	} else {
		fp = stdout;
	}

	if( IS_IMAGE(dp) || IS_SEQUENCE(dp) )
		if( !CONFIRM(
		"are you sure you want to write an image/sequence in ascii") ){
			fclose(fp);
			return;
		}

	pntvec(QSP_ARG  dp,fp);
	if( fp != stdout ) {
		if( verbose ){
			sprintf(msg_str,"closing file %s",filename);
			prt_msg(msg_str);
		}
		fclose(fp);
	}
}

static COMMAND_FUNC( do_append )
{
	Data_Obj *dp;
	FILE *fp;

	dp=PICK_OBJ("");
	if( dp==NO_OBJ ) return;

	INSIST_RAM(dp,"append_obj")

	if( IS_IMAGE(dp) || IS_SEQUENCE(dp) )
		if( !CONFIRM(
		"are you sure you want to write an image/sequence in ascii") )
			return;
	fp=TRYNICE( NAMEOF("output file"), "a" );
	if( !fp ) return;
	pntvec(QSP_ARG  dp,fp);
	fclose(fp);
}

static const char *print_fmt_name[N_PRINT_FORMATS];
static int fmt_names_inited=0;

static void init_print_fmt_names(void)
{
	int i;

#ifdef CAUTIOUS
	if( fmt_names_inited ){
NWARN("CAUTIOUS:  unnecessary call to init_print_fmt_names!?");
		return;
	}
#endif /* CAUTIOUS */

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
#ifdef CAUTIOUS
			default:
				sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  Oops, no initialization for print format %d!?",i);
				NERROR1(DEFAULT_ERROR_STRING);
				break;
#endif /* CAUTIOUS */
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
	d = HOW_MANY("number of significant digits to print");
	set_display_precision(d);
}

Command asc_ctbl[]={
{ "display",	do_disp_obj,	"display data"				},
{ "read",	do_read_obj,	"read vector data from ascii file"	},
{ "pipe_read",	do_pipe_obj,	"read vector data from named pipe"	},
{ "set_string",	do_read_str,	"read vector data from a string"	},
{ "get_string",	do_wrt_str,	"set a script variable to the value of a stored string"	},
{ "write",	do_wrt_obj,	"write vector data to ascii file"	},
{ "append",	do_append,	"append vector data to ascii file"	},
{ "exact",	do_exact,	"enable/disable warnings for file/vector length mismatch"	},
{ "output_fmt",	do_set_fmt,	"set data print format"			},
{ "input_fmt",	do_set_in_fmt,	"specify format of input line"		},
{ "max_per_line",do_set_max,	"set maximum number of items per line"	},
{ "digits",	do_set_digits,	"set number of significant digits to print"	},
#ifndef	MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif	/* ! MAC */
{ NULL_COMMAND								}
};

COMMAND_FUNC( asciimenu )			/** ascii object submenu */
{
	PUSHCMD(asc_ctbl,ASCII_MENU_NAME);
}

