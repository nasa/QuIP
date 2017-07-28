
#include <string.h>
#include "quip_config.h"

/* used to be query.c ... */

/**/
/**		input and output stuff		**/
/**/

#include "quip_prot.h"
#include "query_prot.h"
#include "nexpr.h"
#include "history.h"

ITEM_PICK_FUNC(Item_Type,ittyp)
ITEM_PICK_FUNC(Macro,macro)

long how_many(QSP_ARG_DECL  const char *prompt)
{
	const char *s;
	const char *pline;
	long n;
	//double dn;
	Typed_Scalar *tsp;

	// Why does how_many all next_query_word, while how_much uses nameof???

	pline = format_prompt(QSP_ARG  PROMPT_FORMAT, prompt);
	s=next_query_word(QSP_ARG  pline);

	tsp=pexpr(QSP_ARG  s);

	if( SCALAR_PREC_CODE(tsp) == PREC_STR ){
		sprintf(ERROR_STRING,
	"how_many:  can't convert string \"%s\" to an integer!?",s);
		WARN(ERROR_STRING);
		n = 0;
	} else {
		// What should we do on 32 bit machines?
		n= (long) llong_for_scalar(tsp);
	}

	RELEASE_SCALAR(tsp)

	/* SOME OLD COMMENTS THAT WERE FOR SOME RANGE CHECKING CODE,
	 * WHICH HAS BEEN OBVIATED BY THE USE OF TYPED SCALARS...
	 */

	/* we used to check for rounding error here... */
	/* We still ought to check for gross error
	 * caused by going to a signed int...
	 */

	/* The idea is that we want to warn the user if
	 * a number like 4.5 is entered when an
	 * integer is needed...  But this test was giving
	 * false positives because some integers
	 * were not represented as pure integers in
	 * the double representation  -  how can that be?
	 *
	 * We need to know something about the machine precision
	 * and the rounding error...
	 */

	return(n);
}

double how_much(QSP_ARG_DECL  const char* s)		/**/
{
	const char *estr;
	Typed_Scalar *tsp;
	double d;

	estr=nameof(QSP_ARG  s);
	tsp=pexpr(QSP_ARG  estr);
	d=double_for_scalar(tsp);
	RELEASE_SCALAR(tsp)
	return( d );
}

#define N_BOOL_CHOICES	4
static const char *bool_choices[N_BOOL_CHOICES]={"no","yes","false","true"};

#define YES	1
#define NO	0

#define ASKIF_FORMAT	"%s? (y/n) "

int askif(QSP_ARG_DECL  const char *prompt)
{
	const char *pline;
	int n;

	pline = format_prompt(QSP_ARG  ASKIF_FORMAT, prompt);

	do {
		inhibit_next_prompt_format(SINGLE_QSP_ARG);	// prompt already formatted!
		n = which_one(QSP_ARG  pline,N_BOOL_CHOICES,bool_choices);
		enable_prompt_format(SINGLE_QSP_ARG);
	} while( n < 0 && intractive( SINGLE_QSP_ARG ) );


	switch(n){
		case 0:				/* no */
		case 2:				/* false */
			return(0);
		case 1:				/* yes */
		case 3:				/* true */
			return(1);
	}
	return( -1 );
}

int confirm(QSP_ARG_DECL  const char *s)
{
	if( !intractive( SINGLE_QSP_ARG ) ) return(1);
	return(askif(QSP_ARG  s));
}

/*
 * Get a string from the query file.
 *
 * Get a string from the query file by calling next_query_word().
 * Macro expansion is disabled during this call.
 * The prompt string is prefixed by "Enter " and postfixed by a colon.
 * Used to get user command arguments.
 */

const char * nameof(QSP_ARG_DECL  const char *prompt)
{
	const char *pline;
	int v;
	const char *buf;

	pline = format_prompt(QSP_ARG  PROMPT_FORMAT, prompt);

	/* turn macros off so we can enter macro names!? */

	v = QS_FLAGS(THIS_QSP) & QS_EXPAND_MACS;		/* save current value */
	CLEAR_QS_FLAG_BITS(THIS_QSP,QS_EXPAND_MACS);
	buf=next_query_word(QSP_ARG  pline);
	SET_QS_FLAG_BITS(THIS_QSP,v);		/* restore macro state */
	return(buf);
}

static const char *insure_item_prompt(Item_Type *itp, const char *prompt)
{
	if( prompt == NULL || *prompt==0 )
		return IT_NAME(itp);
	return prompt;
}

static void remove_from_history_list(QSP_ARG_DECL  const char *prompt, const char *s)
{
	const char *pline;
	pline = format_prompt(QSP_ARG  PROMPT_FORMAT, prompt);
	rem_def(QSP_ARG  pline,s);
}


/*
 * Use this function instead of get_xxx(nameof("item name"))
 *
 * When the number of items is large, the current algorithm is unacceptably slow.
 * (Here large means 100,000 items or more - but how few items can cause the problem?
 */

Item *pick_item(QSP_ARG_DECL  Item_Type *itp,const char *prompt)
{
	Item *ip;
	const char *s;

	/* use item type name as the prompt */

	assert( itp != NULL );

	if( ! IS_COMPLETING(THIS_QSP) ){
		s = NAMEOF(prompt);
		return get_item(QSP_ARG  itp, s);
	}

	prompt = insure_item_prompt(itp,prompt);

	assert( QS_PICKING_ITEM_ITP(THIS_QSP) == NULL );

	SET_QS_PICKING_ITEM_ITP(THIS_QSP,itp);
	s=NAMEOF(prompt);
	SET_QS_PICKING_ITEM_ITP(THIS_QSP,NULL);

	ip=item_of(QSP_ARG  itp,s);	// report_invalid_pick will complain, so don't need to here

	if( ip == NULL ){
		remove_from_history_list(QSP_ARG  prompt, s);
		// list the valid items
		report_invalid_pick(QSP_ARG  itp, s);
	}

	return(ip);
}

#ifdef HAVE_HISTORY

/* This function is used to initialize completion history for
 * nameof() for the cases where we cannot use pick_item(),
 * such as subscripted data objects, or when we want to allow
 * the user to enter "all" instead of an object name
 */

void init_item_hist( QSP_ARG_DECL  Item_Type *itp, const char* prompt )
{
	List *lp;

	assert( itp != NULL );

	// Don't do this if the number of choices is too large...
	// We should set a flag in the itp...

	lp=item_list(QSP_ARG  itp);
	if( lp == NULL ) return;
	init_hist_from_item_list(QSP_ARG  prompt,lp);
}
#endif /* HAVE_HISTORY */

