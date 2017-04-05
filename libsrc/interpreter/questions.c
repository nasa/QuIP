
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
	char pline[LLEN];
	long n;
	//double dn;
	Typed_Scalar *tsp;

	// Why does how_many all qword, while how_much uses nameof???

	// BUG? can prompt get too long?
	assert( strlen(prompt) < LLEN );

	if( prompt[0] != 0 ) sprintf(pline,PROMPT_FORMAT,prompt);
	else pline[0]=0;

	s=qword(QSP_ARG  pline);

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

int askif(QSP_ARG_DECL  const char *prompt)
{
	char pline[LLEN];
	int n;

	// BUG? can prompt get too long?
	assert( strlen(prompt) < (LLEN-10) );

	if( prompt[0] != 0 ) sprintf(pline,"%s? (y/n) ",prompt);
	else pline[0]=0;

	do {
		n = which_one2(QSP_ARG  pline,N_BOOL_CHOICES,bool_choices);
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
 * Get a string from the query file by calling qword().
 * Macro expansion is disabled during this call.
 * The prompt string is prefixed by "Enter " and postfixed by a colon.
 * Used to get user command arguments.
 */

const char * nameof(QSP_ARG_DECL  const char *prompt)
		/* user prompt */
{
	char pline[LLEN];
	int v;
	const char *buf;

//	assert( strlen(prompt) < (LLEN-10) );
//	make_prompt checks string length

	make_prompt(QSP_ARG  pline,prompt);

	/* turn macros off so we can enter macro names!? */

	v = QS_FLAGS(THIS_QSP) & QS_EXPAND_MACS;		/* save current value */
	CLEAR_QS_FLAG_BITS(THIS_QSP,QS_EXPAND_MACS);
	buf=qword(QSP_ARG  pline);
	SET_QS_FLAG_BITS(THIS_QSP,v);		/* restore macro state */
	return(buf);
}

/*
 * Get a string from the query file with macro expansion.
 *
 * Like nameof(), but macro expansion is enabled and the prompts
 * are not modified.  Used to get command words.
 *
 * The command prompt has the potential to grow too much!?
 */

const char * nameof2(QSP_ARG_DECL  const char *prompt)
{
	//char pline[LLEN];
	const char *buf;

	//strcpy(pline,prompt);
	//buf=qword(QSP_ARG  pline);

	// Why were we copying the prompt?
	// Now the prompt is allowed to grow using
	// string buffers, so the strcpy is a potential
	// buffer overflow!
	buf=qword(QSP_ARG  prompt);

	return(buf);
}

/* Make prompt takes a query string (like "number of elements") and
 * prepends "Enter " and appends ":  ".
 * We can inhibit this by clearing the flag,
 * but in that case we reset the flag after use,
 * so that we can always assume the default behavior.
 */

void make_prompt(QSP_ARG_DECL char buffer[LLEN],const char* s)
{
	if( QS_FLAGS(THIS_QSP) & QS_FORMAT_PROMPT ){
		// BUG possible buffer overrun
		if( strlen(s) + strlen(PROMPT_FORMAT) -2 >= LLEN ){
			sprintf(ERROR_STRING,"make_prompt:  formatted prompt too long for buffer!?");
			WARN(ERROR_STRING);
			buffer[0]=0;
		} else {
			if(  s[0]  != 0 ) sprintf(buffer,PROMPT_FORMAT,s);
			else  buffer[0]=0;
		}
	} else {
		if( strlen(s) >= LLEN ){
			sprintf(ERROR_STRING,"make_prompt:  prompt too long for buffer!?");
			WARN(ERROR_STRING);
			buffer[0]=0;
		} else {
			strcpy(buffer,s);	/* BUG possible overrun error */
			SET_QS_FLAG_BITS(THIS_QSP, QS_FORMAT_PROMPT); /* this is a one-shot deal. */
		}
	}
}

#ifdef USE_CHOICE_LIST
static const char **make_choices( QSP_ARG_DECL  int* countp, List* lp )
{
	const char **choices;
	int i;
	Item *ip;
	Node *np;

	*countp = eltcount(lp);
	if( *countp == 0 ){
		if( verbose ) WARN("make_choices:  passed empty list!?");
		return(NULL);
	}

	choices = (const char**) getbuf(*countp*sizeof(char *));

	if( choices == NULL ) {
		ERROR1("make_choices:  out of memory");
		IOS_RETURN_VAL(NULL)
	}
	
	np=QLIST_HEAD(lp);
	i=0;
	while(np!=NO_NODE){
		ip = (Item*) NODE_DATA(np);
		choices[i++] = ITEM_NAME(ip);
		np = NODE_NEXT(np);
	}
	return(choices);
}
#endif // USE_CHOICE_LIST

#ifdef USE_CHOICE_LIST
static void setup_item_choices( QSP_ARG_DECL  Item_Type *itp )
{
	int count;

	if( IT_CHOICES(itp) != NO_STR_ARRAY ){
		givbuf( (char *) IT_CHOICES(itp));
	}
	SET_IT_CHOICES(itp, make_choices(QSP_ARG  &count,item_list(QSP_ARG  itp)) );
	SET_IT_N_CHOICES(itp, count);
	CLEAR_IT_FLAG_BITS(itp, NEED_CHOICES);
}
#endif // USE_CHOICE_LIST


/*
 * Use this function instead of get_xxx(nameof("item name"))
 *
 * When the number of items is large, the current algorithm is unacceptably slow.
 * (Here large means 100,000 items or more - but how few items can cause the problem?
 */

Item *pick_item(QSP_ARG_DECL  Item_Type *itp,const char *prompt)
{
#ifdef USE_CHOICE_LIST
	int i;
#endif // USE_CHOICE_LIST
	Item *ip;
	const char *s;

	/* use item type name as the prompt */

//#ifdef CAUTIOUS
//	if( itp == NO_ITEM_TYPE ){
//		WARN("CAUTIOUS:  Uninitialized item type given to pick_item");
//		/*s=*/NAMEOF("dummy");
//		return(NO_ITEM);
//	}
//#endif /* CAUTIOUS */
	assert( itp != NO_ITEM_TYPE );

	if( ! IS_COMPLETING(THIS_QSP) ){
		s = NAMEOF(prompt);
		return get_item(QSP_ARG  itp, s);
	}

	/* use the item type name as the prompt if unspecified */
	if( prompt == NULL || *prompt==0 )
		prompt=IT_NAME(itp);

	// The old way with an array of choices is no good when there are 100k items...

	// picking_item_itp
	assert( QS_PICKING_ITEM_ITP(THIS_QSP) == NULL );

	SET_QS_PICKING_ITEM_ITP(THIS_QSP,itp);
//fprintf(stderr,"pick_item:  picking_item_itp set to %s\n",ITEM_TYPE_NAME(itp));
	s=NAMEOF(prompt);
	SET_QS_PICKING_ITEM_ITP(THIS_QSP,NULL);

	//ip=get_item(QSP_ARG  itp,s);
	ip=item_of(QSP_ARG  itp,s);	// report_invalid_pick will complain, so don't need to here

	if( ip == NULL ){
		// remove from history list
		char pline[LLEN];
		make_prompt(QSP_ARG  pline,prompt);
		rem_def(QSP_ARG  pline,s);

		// list the valid items
		// BUG? should only do this for interactive?
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

//#ifdef CAUTIOUS
//	if( itp == NO_ITEM_TYPE ){
//		WARN("CAUTIOUS:  init_item_hist passed negative index");
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( itp != NO_ITEM_TYPE );

	// Don't do this if the number of choices is too large...
	// We should set a flag in the itp...

	lp=item_list(QSP_ARG  itp);
	if( lp == NO_LIST ) return;
	init_hist_from_item_list(QSP_ARG  prompt,lp);
}
#endif /* HAVE_HISTORY */

