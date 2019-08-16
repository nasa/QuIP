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

	if( CLASS_STIM_CMD(tc_p) != NULL )
		givbuf((void *) CLASS_STIM_CMD(tc_p) );
	SET_CLASS_STIM_CMD( tc_p, savestr(s) );
}

#define flip_coin() _flip_coin(SINGLE_QSP_ARG)

static void _flip_coin(SINGLE_QSP_ARG_DECL)
{
	char buf[128];
	int coin=0;	// initialize to quiet compiler, but not necessary!?

	coin=(int)rn(1);
	sprintf(buf,"%d",coin);
	assign_var("coin",buf);
}

/* clip trailing zeros if there is a decimal point */

static void strip_trailing_zeroes(char *buf)
{
	char *s;

	s=buf;
	while( *s ){
		if( *s == '.' ){
			// go to the end of the string and work back
			s=buf+strlen(buf)-1;
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
}

#define get_this_xval(tc_p, val_idx) _get_this_xval(QSP_ARG  tc_p, val_idx)

static void _get_this_xval(QSP_ARG_DECL  Trial_Class *tc_p, int val_idx)
{
	char buf[256];
	float *xv_p;

	assert( CLASS_XVAL_OBJ(tc_p) != NULL );
	xv_p = indexed_data( CLASS_XVAL_OBJ(tc_p), val_idx );
	sprintf(buf,"%f",*xv_p);

	strip_trailing_zeroes(buf);
	assign_var("xval",buf);

	sprintf(buf,"%d",val_idx);
	assign_var("val",buf);
}

int _default_response(QSP_ARG_DECL  Staircase *stc_p, Experiment *exp_p)
{
	int rsp;

	rsp = get_response(stc_p,exp_p);

	// BUG better to have a general feedback scheme?
	if( verbose ){
		if( rsp == STAIR_CRCT_RSP(stc_p) )
			advise("correct");
		else
			advise("incorrect");
	}

	return(rsp);
}

#define set_class_index(stc_p) _set_class_index(QSP_ARG  stc_p)

static void _set_class_index(QSP_ARG_DECL  Staircase *stc_p)
{
	char buf[256];
	Trial_Class *tc_p = STAIR_CLASS(stc_p);

	sprintf(buf,"%d",CLASS_INDEX(tc_p));
	assign_var("class",buf);
}

// default_stim presents the stimulus and collects a response...
//
// When we use a gui, we need a different response routine!?

void _default_stim(QSP_ARG_DECL  Staircase *stc_p)
{
	Trial_Class *tc_p;
	
	assert( stc_p != NULL );

	tc_p = STAIR_CLASS(stc_p);
	assert( tc_p != NULL );

	if( is_fc ) flip_coin();
	get_this_xval(tc_p,STAIR_VAL(stc_p));

	set_class_index(stc_p);
	chew_text(CLASS_STIM_CMD(STAIR_CLASS(stc_p)), "(stimulus text)");

//	return default_response(stc_p);
}

