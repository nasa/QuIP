#include "quip_config.h"

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* prototype for sync() */
#endif

#include "quip_prot.h"
#include "param_api.h"
#include "stc.h"
#include "rn.h"
#include "stack.h"	// BUG
#include "query_stack.h"	// BUG

/* local prototypes */

struct param n_trials_ptbl[]={
{ "n_prelim",	"# of preliminary trials per stair (<0 for variable to criterion)",
							INT_PARAM(&EXPT_N_PRELIM_TRIALS(&expt1)) },
{ "n_data",	"# of recorded trials per stair",	INT_PARAM(&EXPT_N_RECORDED_TRIALS(&expt1)) },
{ NULL_UPARAM }
};

struct param n_stairs_ptbl[]={
{ "n_updn",	"up-down stairs per cond.",		INT_PARAM(&EXPT_N_UPDN(&expt1)) },
{ "n_dnup",	"down-up stairs per cond.",		INT_PARAM(&EXPT_N_DNUP(&expt1)) },
{ "n_2up",	"two-to-one stairs per cond.",		INT_PARAM(&EXPT_N_2UP(&expt1))  },
{ "n_2dn",	"one-to-two stairs per cond.",		INT_PARAM(&EXPT_N_2DN(&expt1))  },
{ "n_2iup",	"inverted two-to-one stairs per cond.",	INT_PARAM(&EXPT_N_2IUP(&expt1)) },
{ "n_2idn",	"inverted one-to-two stairs per cond.",	INT_PARAM(&EXPT_N_2IDN(&expt1)) },
{ "n_3up",	"three-to-one stairs per condition",	INT_PARAM(&EXPT_N_3UP(&expt1))  },
{ "n_3dn",	"one-to-three stairs per condition",	INT_PARAM(&EXPT_N_3DN(&expt1))  },
{ NULL_UPARAM }
};


/* make the staircases specified by the parameter table */

#define make_staircases(exp_p) _make_staircases(QSP_ARG  exp_p)

static void _make_staircases(QSP_ARG_DECL  Experiment *exp_p)
{
	int j;
	List *lp;
	Node *np;
	Trial_Class *tc_p;

	lp=trial_class_list();
	assert( lp != NULL );

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		tc_p=(Trial_Class *)np->n_data;
		np=np->n_next;

		/* make_staircase( type, class, mininc, correct rsp, inc rsp ); */
		for( j=0;j<EXPT_N_UPDN(exp_p);j++)
			make_staircase( UP_DOWN, tc_p, 1, YES_INDEX, YES_INDEX );
		for( j=0;j<EXPT_N_DNUP(exp_p);j++)
			make_staircase( UP_DOWN, tc_p, -1, YES_INDEX, YES_INDEX );
		for(j=0;j<EXPT_N_2UP(exp_p);j++)
			/*
			 * 2-up increases val after 2 YES's,
			 * decreases val after 1 NO
			 * seeks 71% YES, YES decreasing with val
			 */
			make_staircase( TWO_TO_ONE, tc_p, 1, YES_INDEX, YES_INDEX );
		for(j=0;j<EXPT_N_2DN(exp_p);j++)
			/*
			 * 2-down decreases val after 2 NO's,
			 * increases val after 1 YES
			 * Seeks 71% NO, NO increasing with val
			 */
			make_staircase( TWO_TO_ONE, tc_p, -1, YES_INDEX, NO_INDEX );
		for(j=0;j<EXPT_N_2IUP(exp_p);j++)
			/*
			 * 2-inverted-up decreases val after 2 YES's,
			 * increases val after 1 NO
			 * Seeks 71% YES, YES increasing with val
			 */
			make_staircase( TWO_TO_ONE, tc_p, -1, YES_INDEX, YES_INDEX );
		for(j=0;j<EXPT_N_2IDN(exp_p);j++)
			/*
			 * 2-inverted-down increases val after 2 NO's,
			 * decreases val after 1 YES
			 * Seeks 71% NO, NO decreasing with val
			 */
			make_staircase( TWO_TO_ONE, tc_p, 1, YES_INDEX, NO_INDEX );
		for(j=0;j<EXPT_N_3UP(exp_p);j++)
			make_staircase( THREE_TO_ONE, tc_p, 1, YES_INDEX, YES_INDEX );
		for(j=0;j<EXPT_N_3DN(exp_p);j++)
			make_staircase( THREE_TO_ONE, tc_p, -1, YES_INDEX, NO_INDEX );
	}
}

static COMMAND_FUNC( do_chng_trials )
{
	chngp(QSP_ARG n_trials_ptbl);
}

static COMMAND_FUNC( do_chng_stairs )
{
	int d;

	/* BUG?
	 * by clearing out the actual staircases, any
	 * hand edited changes will be lost.
	 * Since the old system didn't include hand editing,
	 * this is OK (at least unlikely to break any old stuff).
	 */

	delete_all_stairs();	/* clear out old ones */

	chngp(QSP_ARG n_stairs_ptbl);

	/* chngp just pushes the parameter menu...
	 * But we need to executed some routines when we are done.
	 * So we have to duplicate the loop here...
	 *
	 * This is very ugly!  Better to have chngp call a callback func
	 * when it is done???  Or get rid of chngp altogether???
	 */

	d = STACK_DEPTH(QS_MENU_STACK(THIS_QSP));
	while( STACK_DEPTH(QS_MENU_STACK(THIS_QSP)) == d )
		qs_do_cmd(THIS_QSP);

	new_exp(SINGLE_QSP_ARG);
	make_staircases(&expt1);
}



#undef ADD_CMD
#define ADD_CMD(s,f,h)	ADD_COMMAND(exp_params_menu,s,f,h)

MENU_BEGIN(exp_params)
ADD_CMD( trials,	do_chng_trials,	specify number of trials)
ADD_CMD( staircases,	do_chng_stairs,	specify numbers and types of staircases)
MENU_END(exp_params)

COMMAND_FUNC( do_exp_param_menu )
{
	CHECK_AND_PUSH_MENU(exp_params);
}

