#include "quip_config.h"

#include <stdio.h>
#include <string.h>

#include "stc.h"
#include "quip_prot.h"
#include "rn.h"
#include "getbuf.h"
#include "variable.h"

int is_fc=0;

void general_mod(QSP_ARG_DECL Trial_Class * tc_p)
{
	const char *s;

	s=NAMEOF("stimulus command string");
	assert( tc_p != NULL );

	if( CLASS_CMD(tc_p) != NULL )
		givbuf((void *) CLASS_CMD(tc_p) );
	SET_CLASS_CMD( tc_p, savestr(s) );
}

int default_stim(QSP_ARG_DECL  Trial_Class *tc_p,int val,Staircase *stc_p)
{
	char stim_str[256], *s;
	int coin=0;	// initialize to quiet compiler, but not necessary!?
	int rsp;
	//struct var *vp;
	Variable *vp;
	float *xv_p;

	if( is_fc ){
		coin=(int)rn(1);
		sprintf(stim_str,"%d",coin);
		assign_var("coin",stim_str);
	}

	assert( CLASS_XVAL_OBJ(tc_p) != NULL );
	xv_p = indexed_data( CLASS_XVAL_OBJ(tc_p), val );
	sprintf(stim_str,"%f",*xv_p);

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

	assign_var("xval",stim_str);
	sprintf(stim_str,"%d",val);
	assign_var("val",stim_str);
	sprintf(stim_str,"%d",CLASS_INDEX(tc_p));
	assign_var("class",stim_str);

	assert( tc_p != NULL );

	//sprintf(msg_str,"Text \"%s\"",(char *)(tc_p->cl_data));
	//PUSH_INPUT_FILE(msg_str);

	//interpret_text_fragment(QSP_ARG tc_p->cl_data);		/* use chew_text??? */
	chew_text(CLASS_CMD(tc_p), "(stimulus text)");
	vp=var_of("response_string");
	if( vp != NULL )
		rsp = collect_response(VAR_VALUE(vp));
	else {
		static int warned=0;

		if( !warned ){
			warn("default_stim:  script variable $response_string not defined");
			warned=1;
		}
		rsp = collect_response("Enter response: ");
	}

	if( is_fc ){
		/* stimulus routine may have changed value of coin */
		vp=var_of("coin");
		if( vp == NULL )
			warn("variable \"coin\" not set!!!");
		else {
			if( sscanf(VAR_VALUE(vp),"%d",&coin) != 1 )
			warn("error scanning integer from variable \"coin\"\n");
		}

		/*
		if( coin ){
			if( rsp == YES ) rsp = NO;
			else if( rsp == NO ) rsp = YES;
		}
		*/
		assert( stc_p != NULL );
        
        // analyzer complains coin is a garbage value??? BUG?
		if( coin ){
			SET_STAIR_CRCT_RSP(stc_p,NO);
		} else {
			SET_STAIR_CRCT_RSP(stc_p,YES);
		}
		if( verbose ){
			if( rsp == STAIR_CRCT_RSP(stc_p) )
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

