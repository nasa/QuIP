#include "quip_config.h"

#include <stdio.h>
#include <string.h>

#include "stc.h"
#include "quip_prot.h"
#include "rn.h"
#include "getbuf.h"
#include "variable.h"

int is_fc=0;

void general_mod(QSP_ARG_DECL int t_class)
{
	const char *s;
	Trial_Class *tcp;

	s=NAMEOF("stimulus command string");
	tcp = index_class(QSP_ARG  t_class);
	assert( tcp != NULL );

	if( CLASS_CMD(tcp) != NULL )
		givbuf((void *) CLASS_CMD(tcp) );
	SET_CLASS_CMD( tcp, savestr(s) );
}

#ifdef FOOBAR
/* BUG?  this routine duplicates chew_text??? */

void interpret_text_fragment(QSP_ARG_DECL const char *s)
{
	int ql;

	PUSH_TEXT(s);

	ql=QLEVEL;

	/* now need to interpret input intil text is eaten */

	while( QLEVEL >= ql ) {
		qs_do_cmd(THIS_QSP);
		lookahead_til(QSP_ARG ql-1);
	}
}
#endif // FOOBAR

int default_stim(QSP_ARG_DECL  Trial_Class *tcp,int val,Staircase *stcp)
{
	char stim_str[256], *s;
	int coin=0;	// initialize to quiet compiler, but not necessary!?
	int rsp;
	//struct var *vp;
	Variable *vp;

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
	sprintf(stim_str,"%d",CLASS_INDEX(tcp));
	ASSIGN_VAR("class",stim_str);

	assert( tcp != NULL );

	//sprintf(msg_str,"Text \"%s\"",(char *)(tcp->cl_data));
	//PUSH_INPUT_FILE(msg_str);

	//interpret_text_fragment(QSP_ARG tcp->cl_data);		/* use chew_text??? */
	chew_text(QSP_ARG CLASS_CMD(tcp), "(stimulus text)");
	vp=VAR_OF("response_string");
	if( vp != NULL )
		rsp=response(QSP_ARG  VAR_VALUE(vp));
	else {
		static int warned=0;

		if( !warned ){
			WARN("default_stim:  script variable $response_string not defined");
			warned=1;
		}
		rsp=response(QSP_ARG  "Enter response: ");
	}

	if( is_fc ){
		/* stimulus routine may have changed value of coin */
		vp=VAR_OF("coin");
		if( vp == NULL )
			WARN("variable \"coin\" not set!!!");
		else {
			if( sscanf(VAR_VALUE(vp),"%d",&coin) != 1 )
			WARN("error scanning integer from variable \"coin\"\n");
		}

		/*
		if( coin ){
			if( rsp == YES ) rsp = NO;
			else if( rsp == NO ) rsp = YES;
		}
		*/
		assert( stcp != NULL );
        
        // analyzer complains coin is a garbage value??? BUG?
		if( coin ){
			SET_STAIR_CRCT_RSP(stcp,NO);
		} else {
			SET_STAIR_CRCT_RSP(stcp,YES);
		}
		if( verbose ){
			if( rsp == STAIR_CRCT_RSP(stcp) )
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
		stmrt=default_stim;
		modrt=general_mod;
		/* we don't want a warning msg if default file doesn't exist */
		/* rdxvals("xvals"); */
		inited++;
	}

	do_exp_menu(SINGLE_QSP_ARG);
}

