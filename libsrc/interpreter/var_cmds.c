#include "quip_config.h"

char VersionId_interpreter_var_cmds[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_MATH_H
#include <math.h>
#endif

#include "savestr.h"
#include "debug.h"
#include "items.h"
#include "nexpr.h"
#include "submenus.h"
#include "menuname.h"

/* local prototypes */
static COMMAND_FUNC( setvar );
static COMMAND_FUNC( do_assign );
static COMMAND_FUNC( unsetvar );
static COMMAND_FUNC( showvar );
static COMMAND_FUNC( do_set_nsig );

static const char *def_gfmt_str="%.7g";
static const char *def_xfmt_str="0x%lx";
static const char *def_ofmt_str="0%lo";
static const char *def_dfmt_str="%ld";
static char gfmt_str[8];
static char xfmt_str[8];
static char ofmt_str[8];
static char dfmt_str[8];
static char *number_fmt_string=NULL;

static const char **var_fmt_list=NULL;

#define MAX_FMT_STACK_DEPTH	16
static char *var_fmt_stack[MAX_FMT_STACK_DEPTH];
static int fmt_stack_depth=(-1);

static void init_fmt_choices(SINGLE_QSP_ARG_DECL)
{
#ifdef CAUTIOUS
	int i;
#endif /* CAUTIOUS */

	var_fmt_list = (const char **) getbuf( N_PRINT_FORMATS * sizeof(char *) );

#ifdef CAUTIOUS
	/* Set to known value */
	for(i=0;i<N_PRINT_FORMATS;i++) var_fmt_list[i]=NULL;
#endif /* CAUTIOUS */

	var_fmt_list[ FMT_DECIMAL ] = "decimal";
	var_fmt_list[ FMT_HEX ] = "hex";
	var_fmt_list[ FMT_OCTAL ] = "octal";
	var_fmt_list[ FMT_UDECIMAL ] = "unsigned_decimal";
	var_fmt_list[ FMT_FLOAT ] = "float";
	var_fmt_list[ FMT_POSTSCRIPT ] = "postscript";

#ifdef CAUTIOUS
	/* Now make sure we have initialized all */
	for(i=0;i<N_PRINT_FORMATS;i++)
		if( var_fmt_list[i] == NULL ){
			sprintf(ERROR_STRING,"CAUTIOUS:  init_fmt_choices:  no initialization for format %d!?",i);
			ERROR1(ERROR_STRING);
		}
#endif /* CAUTIOUS */

}

static void init_default_formats(void)
{
#ifdef CAUTIOUS
	if( number_fmt_string != NULL )
NWARN("CAUTIOUS:  var_cmds.c:  init_default_formats:  format strings already initialized!?");
#endif
	strcpy(gfmt_str,def_gfmt_str);
	strcpy(dfmt_str,def_dfmt_str);
	strcpy(xfmt_str,def_xfmt_str);
	strcpy(ofmt_str,def_ofmt_str);
	number_fmt_string = dfmt_str;
}

#define CHECK_FMT_STRINGS							\
	if( number_fmt_string == NULL ) init_default_formats();

static void set_fmt(QSP_ARG_DECL  Number_Fmt i)
{
	switch(i){
		case FMT_FLOAT:  number_fmt_string=gfmt_str; break;

		case FMT_UDECIMAL:	/* do something special for unsigned? */
		case FMT_DECIMAL:  number_fmt_string=dfmt_str; break;

		case FMT_HEX:  number_fmt_string=xfmt_str; break;

		case FMT_OCTAL:  number_fmt_string=ofmt_str; break;
		case FMT_POSTSCRIPT:
			/* does this make sense? */
WARN("set_fmt:  not sure what to do with FMT_POSTSCRIPT - using decimal.");
			number_fmt_string=dfmt_str;
			break;
#ifdef CAUTIOUS
		default:
			sprintf(ERROR_STRING,"CAUTIOUS:  set_fmt:  unexpected format code %d!?",i);
			WARN(ERROR_STRING);
			break;
#endif /* CAUTIOUS */
	}
}

static COMMAND_FUNC( do_set_fmt )
{
	Number_Fmt i;

	if( var_fmt_list == NULL ) init_fmt_choices(SINGLE_QSP_ARG);

	i=(Number_Fmt)WHICH_ONE("print format for variable evaluation",
		N_PRINT_FORMATS,var_fmt_list);
	if( i < 0 ) return;

	set_fmt(QSP_ARG  i);
}

static COMMAND_FUNC( do_push_fmt )
{
	Number_Fmt i;

	if( var_fmt_list == NULL ) init_fmt_choices(SINGLE_QSP_ARG);

	i=(Number_Fmt)WHICH_ONE("print format for variable evaluation",
		N_PRINT_FORMATS,var_fmt_list);
	if( i < 0 ) return;

	fmt_stack_depth++;
	if( fmt_stack_depth>= MAX_FMT_STACK_DEPTH ){
		fmt_stack_depth--;
		WARN("variable format stack overflow");
		return;
	}

	CHECK_FMT_STRINGS

	var_fmt_stack[fmt_stack_depth] = number_fmt_string;
	set_fmt(QSP_ARG  i);
}

static COMMAND_FUNC( do_pop_fmt )
{
	if( fmt_stack_depth < 0 ){
		WARN("do_pop_fmt:  stack underflow!?");
		return;
	}
	number_fmt_string = var_fmt_stack[fmt_stack_depth];
	fmt_stack_depth--;
}

static COMMAND_FUNC( do_assign )
{
	char varname[LLEN];
	char str[LLEN];
	const char *s;
	double d;

	strcpy( varname,NAMEOF(VARNAME_PROMPT) );
	s = NAMEOF("expression");
	d=pexpr(QSP_ARG  s);

	CHECK_FMT_STRINGS

	if( number_fmt_string == xfmt_str || number_fmt_string == dfmt_str ){
		/* We used to cast the value to integer if
		 * the format string is an integer format -
		 * But now we don't do this if the number has
		 * a fractional part...
		 */
#ifdef HAVE_ROUND
		if( d != round(d) )
			sprintf(str,gfmt_str,d);
		else
			sprintf(str,number_fmt_string,(unsigned long)d);
#else /* ! HAVE_ROUND */
#ifdef HAVE_FLOOR
		if( d != floor(d) )
			sprintf(str,gfmt_str,d);
		else
			sprintf(str,number_fmt_string,(unsigned long)d);
#else /* ! HAVE_FLOOR */
		sprintf(str,number_fmt_string,(unsigned long)d);
#endif /* ! HAVE_FLOOR */
#endif /* ! HAVE_ROUND */

	} else
		sprintf(str,number_fmt_string,d);

	ASSIGN_VAR( varname, str );
}

static COMMAND_FUNC( setvar )
{
	char varname[LLEN];
	const char *varvalue;

	strcpy( varname,NAMEOF(VARNAME_PROMPT) );
	varvalue=NAMEOF("value");
	ASSIGN_VAR( varname, varvalue );
}

static COMMAND_FUNC( unsetvar )
{
	Var *vp;

	vp=PICK_VAR("");
	if( vp!=NO_VAR ) freevar(QSP_ARG  vp);
}

static COMMAND_FUNC( showvar )
{
	Var *vp;

	vp=PICK_VAR("");
	if( vp == NO_VAR ) return;

	if( vp->v_func != NULL )	/* update value of reserved var */
		(*vp->v_func)(vp);

	sprintf(msg_str,"$%s = %s",vp->v_name,vp->v_value); 
	prt_msg(msg_str);
}

static COMMAND_FUNC( findvars )
{
	const char *s;

	s=NAMEOF("name fragment");

	find_vars(QSP_ARG  s);
}

static COMMAND_FUNC( searchvars )
{
	const char *s;

	s=NAMEOF("value fragment");

	search_vars(QSP_ARG  s);
}

#define MIN_SIG_DIGITS	4
#define MAX_SIG_DIGITS	24

static COMMAND_FUNC( do_set_nsig )
{
	int n;

	n = (int) HOW_MANY("number of digits to print in numeric variables");
	if( n<MIN_SIG_DIGITS || n>MAX_SIG_DIGITS ){
		sprintf(ERROR_STRING,
	"Requested number of digits (%d) should be between %d and %d, using %d",
			n,MIN_SIG_DIGITS,MAX_SIG_DIGITS,MAX_SIG_DIGITS);
		WARN(ERROR_STRING);
		n=MAX_SIG_DIGITS;
	}
	CHECK_FMT_STRINGS

	sprintf(number_fmt_string,"%%.%dg",n);
}

static COMMAND_FUNC( do_list_vars ) { list_var_s(SINGLE_QSP_ARG); }
static COMMAND_FUNC( do_var_stats ) { var_stats(SINGLE_QSP_ARG); }

static COMMAND_FUNC( do_list_var_ctxs )
{
	list_var_contexts(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_new_var_ctx )
{
	const char *s;
	Item_Context *icp;

	s=NAMEOF("name for context");

	icp = new_var_context(QSP_ARG  s);

	if( icp == NO_ITEM_CONTEXT ) return;
	// make this context current?
}

static COMMAND_FUNC( do_del_var_ctx )
{
	WARN("do_del_var_ctx not implemented");
}

static COMMAND_FUNC( do_push_var_ctx )
{
	const char *s;

	s=NAMEOF("name of variable context");

	push_var_ctx(QSP_ARG  s);
}

static COMMAND_FUNC( do_pop_var_ctx )
{
	pop_var_ctx(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_show_var_ctx_stk )
{
	show_var_ctx_stk(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_restrict )
{
	int yn;

	yn = ASKIF("restrict variable search to top context");
	restrict_var_context(QSP_ARG  yn);
}


static Command var_ctx_ctbl[]={
{ "list",	do_list_var_ctxs,	"list all variable contexts"	},
{ "show_stack",	do_show_var_ctx_stk,	"show current context stack"	},
{ "new_ctx",	do_new_var_ctx,		"create new context"		},
{ "delete",	do_del_var_ctx,		"delete a variable context"	},
{ "push",	do_push_var_ctx,	"push a variable context onto the stack"	},
{ "pop",	do_pop_var_ctx,		"pop current variable context from the stack"	},
{ "restrict",	do_restrict,		"enable/disable restriction of variable search to top context"	},
{ "quit",	popcmd,			"exit submenu"			},
{ NULL_COMMAND								}
};

static COMMAND_FUNC( do_var_ctxs ){ PUSHCMD(var_ctx_ctbl,"contexts"); }

Command varctbl[]={
{ "set",	setvar,		"set a variable"			},
{ "assign",	do_assign,	"set a variable to an expression value" },
{ "unset",	unsetvar,	"unset a variable"			},
{ "show",	showvar,	"show value of a variable"		},
{ "list",	do_list_vars,	"list all variables"			},
{ "find",	findvars,	"list variables sharing name fragment " },
{ "search",	searchvars,	"list variables containing value fragment " },
{ "stats",	do_var_stats,	"show variable hash table statistics"	},
{ "digits",	do_set_nsig,	"specify number of significant digits"	},
{ "format",	do_set_fmt,	"format for numeric vars"		},
{ "push_fmt",	do_push_fmt,	"push numeric format to stack"		},
{ "pop_fmt",	do_pop_fmt,	"pop numeric format from stack"		},
{ "contexts",	do_var_ctxs,	"variable context submenu"		},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif /* !MAC */
{ NULL_COMMAND								}
};

COMMAND_FUNC( varmenu )
{
	PUSHCMD(varctbl,VAR_MENU_NAME);
}

