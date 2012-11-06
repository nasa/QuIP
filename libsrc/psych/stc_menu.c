#include "quip_config.h"

char VersionId_psych_stairmenu[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <string.h>

#include "stc.h"
#include "getbuf.h"
#include "savestr.h"
#include "rn.h"
#include "query.h"
#include "debug.h"		/* verbose */

int is_fc=0;

void general_mod(QSP_ARG_DECL int t_class)
{
	const char *s;
	Trial_Class *clp;

	s=NAMEOF("stimulus command string");
	clp = index_class(QSP_ARG  t_class);
#ifdef CAUTIOUS
	if( clp == NO_CLASS ) ERROR1("CAUTIOUS:  general_mod, missing class");
#endif /* CAUTIOUS */

	if( clp->cl_data != NULL )
		givbuf(clp->cl_data);
	clp->cl_data = savestr(s);
}

/* BUG?  this routine duplicates chew_text??? */

void interpret_text_fragment(QSP_ARG_DECL const char *s)
{
	int ql;

	PUSHTEXT(s);

	ql=TELL_QLEVEL;

	/* now need to interpret input intil text is eaten */

	while( TELL_QLEVEL >= ql ) {
		do_cmd(SINGLE_QSP_ARG);
		lookahead_til(QSP_ARG ql-1);
	}
}

int general_stim(QSP_ARG_DECL  int t_class,int val,Staircase *stcp)
{
	char stim_str[256], *s;
	int coin;
	int rsp;
	struct var *vp;
	Trial_Class *clp;

	if( is_fc ){
		coin=(int)rn(1);
		sprintf(stim_str,"%d",coin);
		ASSIGN_VAR("coin",stim_str);
	}

	sprintf(stim_str,"%f",xval_array[val]);

	/* clip trailing zeros if there is a decimal point */
	s=stim_str;
	while( *s ){
		if( *s == '.' ){
			s=stim_str+strlen(stim_str)-1;
			while( *s == '0' ) {
				*s=0;
				s--;
			}
			/*
			 * if ONLY 0's after the decimal pt.,
			 * remove the pt too!
			 */
			if( *s == '.' ){
				*s=0;
				s--;
			}
		}
		s++;
	}

	ASSIGN_VAR("xval",stim_str);
	sprintf(stim_str,"%d",val);
	ASSIGN_VAR("val",stim_str);
	sprintf(stim_str,"%d",t_class);
	ASSIGN_VAR("class",stim_str);

	clp = index_class(QSP_ARG  t_class);
#ifdef CAUTIOUS
	if( clp == NO_CLASS ) ERROR1("CAUTIOUS:  missing class");
#endif
	sprintf(msg_str,"Text \"%s\"",(char *)(clp->cl_data));
	PUSH_INPUT_FILE(msg_str);

	interpret_text_fragment(QSP_ARG clp->cl_data);		/* use chew_text??? */
	vp=VAR_OF("response_string");
	if( vp != NO_VAR )
		rsp=response(QSP_ARG  vp->v_value);
	else {
		static int warned=0;

		if( !warned ){
			WARN("script variable $response_string not defined");
			warned=1;
		}
		rsp=response(QSP_ARG  "Enter response: ");
	}

	if( is_fc ){
		/* stimulus routine may have changed value of coin */
		vp=VAR_OF("coin");
		if( vp == NO_VAR )
			WARN("variable \"coin\" not set!!!");
		else {
			if( sscanf(vp->v_value,"%d",&coin) != 1 )
			WARN("error scanning integer from variable \"coin\"\n");
		}

		/*
		if( coin ){
			if( rsp == YES ) rsp = NO;
			else if( rsp == NO ) rsp = YES;
		}
		*/
#ifdef CAUTIOUS
		if( stcp == NULL )
			ERROR1("CAUTIOUS:  stcp is null, but expt is forced choice!?");
#endif /* CAUTIOUS */
		if( coin ){
			stcp->stc_crctrsp = NO;
		} else {
			stcp->stc_crctrsp = YES;
		}
		if( verbose ){
			if( rsp == stcp->stc_crctrsp )
				advise("correct");
			else
				advise("incorrect");
		}
	}
	return(rsp);
}

COMMAND_FUNC( stair_menu )
{
	static int inited=0;

	if( !inited ){
		stmrt=general_stim;
		modrt=general_mod;
		/* we don't want a warning msg if default file doesn't exist */
		/* rdxvals("xvals"); */
		inited++;
	}

	exp_menu(SINGLE_QSP_ARG);
}

