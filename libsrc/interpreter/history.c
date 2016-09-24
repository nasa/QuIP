
#include "quip_config.h"

#ifdef HAVE_HISTORY

#include <stdio.h>
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>
#endif
#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#include "quip_prot.h"	// should be just for external API
#include "query_prot.h"	// should be for things used in interpreter module
#include "history.h"
#include "container.h"
//#include "debug.h"
//#include "getbuf.h"
//#include "savestr.h"
//#include "substr.h"
//#include "query.h"

int history_flag=1;

#ifdef QUIP_DEBUG
debug_flag_t hist_debug=0;
#endif /* QUIP_DEBUG */

/* local variables */

// These maintain info about a list of possible command completions...
static Node *cur_node;	// might make more sense to put in qsp, but we will only ever have one
			// interactive shell...
static List *cur_list;


static char *get_hist_ctx_name(const char *prompt);

/* local prototypes */
static void rem_hcp(QSP_ARG_DECL  Item_Context *icp,Hist_Choice *hcp);
static void add_word_to_history_list(QSP_ARG_DECL  Item_Context *icp,const char *string);
static void clr_defs_if(QSP_ARG_DECL  Item_Context *icp,int n,const char **choices);

// need macro to make these all static
ITEM_INTERFACE_PROTOTYPES(Hist_Choice,choice)
ITEM_INTERFACE_DECLARATIONS(Hist_Choice,choice)

/* This has to match what is up above!!! */
#define HIST_CHOICE_STRING	"Hist_Choice"

static char *get_hist_ctx_name(const char* prompt)
{
	static char str[LLEN];
	/* int n; */

	sprintf(str,"%s.%s",HIST_CHOICE_STRING,prompt);
	return(str);
}

/*
 * Return the history context for this prompt, creating if necessary.
 */

Item_Context *find_hist(QSP_ARG_DECL  const char *prompt)
{
	Item_Context *icp;
	char *ctxname;

	if( choice_itp == NO_ITEM_TYPE ) init_choices(SINGLE_QSP_ARG);

	ctxname = get_hist_ctx_name(prompt);
	icp = ctx_of(QSP_ARG  ctxname);
	if( icp == NO_ITEM_CONTEXT ){
		icp = create_item_context(QSP_ARG  choice_itp,prompt);
//#ifdef CAUTIOUS
//		if( icp == NO_ITEM_CONTEXT ){
//			ERROR1("CAUTIOUS:  error creating history context");
//			IOS_RETURN_VAL(NULL)
//		}
//#endif /* CAUTIOUS */
		assert( icp != NO_ITEM_CONTEXT );
	}

	return(icp);
}

/* Scan a history list, removing any choices which are not in the new list */

static void clr_defs_if(QSP_ARG_DECL  Item_Context *icp,int n,const char** choices)
{
	//Node *np;
	//List *lp;
	Enumerator *ep;

	//lp = dictionary_list(CTX_DICT(icp));
	//np=QLIST_HEAD(lp);
	ep = new_enumerator(CTX_CONTAINER(icp), 0);	// 0 -> default type
//	while(np!=NO_NODE){
	while(ep!=NULL){
		Hist_Choice *hcp;
//		Node *next;
		int i, found;

		/* Because the item nodes get moved to the item free
		 * list when they are deleted, we have to get the next
		 * node BEFORE deletion!!!
		 */
//		next=NODE_NEXT(np);
//		hcp = (Hist_Choice *) NODE_DATA(np);
		hcp = (Hist_Choice *) enumerator_item(ep);
		ep = advance_enumerator(ep);
		found=0;

		/*
		 * This could be done more efficiently,
		 * but who cares...
		 */

		for(i=0;i<n;i++)
			if( !strcmp(hcp->hc_text,choices[i]) )
				found++;
		if( !found )
			rem_hcp(QSP_ARG  icp,hcp);
//		np=next;
	}
}

void preload_history_list(QSP_ARG_DECL  const char* prompt,unsigned int n,const char** choices)
{
	Item_Context *icp;
	unsigned int i;

	icp = find_hist(QSP_ARG  prompt);

	/* remove any history list choices not on the current list */

	clr_defs_if(QSP_ARG  icp,n,choices);

	/*
	 * This gets called from which_one, redundantly in
	 * the case of calls after the first if the choice
	 * list is fixed.  Therefore, to save work, we return
	 * if it looks like the list is the same, i.e. the counts
	 * are equal.
	 */

//	if( eltcount(dictionary_list(CTX_DICT(icp))) == n )
// don't insist that we listify!
	if( container_eltcount(CTX_CONTAINER(icp)) == n )
		return;

	/*
	 * Because of the preceding call to clr_defs_if(), now
	 * ec <= n, and we only have to add choices
	 */

#ifdef QUIP_DEBUG
if( debug & hist_debug ){
sprintf(ERROR_STRING,"preload_history_list for prompt \"%s\"",prompt);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	for(i=0;i<n;i++){
		add_word_to_history_list(QSP_ARG  icp,choices[i]);
	}
}

static void rem_hcp(QSP_ARG_DECL  Item_Context *icp,Hist_Choice *hcp)
{
	PUSH_ITEM_CONTEXT(choice_itp,icp);
	/* BUG? this will search all contexts...
	 * BUT - we expect to find it in the first one!?
	 */
	del_item(QSP_ARG  choice_itp,hcp);
	rls_str((char *)hcp->hc_text);	/* BUG? saved w/ savestr??? */
	pop_item_context(QSP_ARG  choice_itp);
}

void rem_def(QSP_ARG_DECL  const char *prompt,const char* choice)	/** remove selection from list, return next */
{
	Item_Context *icp;
	Hist_Choice *hcp;

	icp = find_hist(QSP_ARG  prompt);

	/* We don't appear to use icp ??? */

	/* but it does create the context if it doesn't exist...
	 * but who cares???
	 */

	PUSH_ITEM_CONTEXT(choice_itp,icp);
	hcp = (Hist_Choice *) choice_of(QSP_ARG  choice);
	pop_item_context(QSP_ARG  choice_itp);

	if( hcp == NO_CHOICE ){
		return;
	}

	rem_hcp(QSP_ARG  icp,hcp);
}

void new_defs(QSP_ARG_DECL  const char* prompt)
{
	Item_Context *icp;

	icp = find_hist(QSP_ARG  prompt);

	clr_defs_if(QSP_ARG  icp,0,(const char **)NULL);
}

static void add_word_to_history_list(QSP_ARG_DECL  Item_Context *icp,const char* string)
{
	Node *np;
	Hist_Choice *hcp;
	List *lp;

//fprintf(stderr,"add_word_to_history_list BEGIN\n");

//#ifdef CAUTIOUS
//	if( string[0]==0 ) {		/* don't add empty string */
//		sprintf(ERROR_STRING,
//			"CAUTIOUS: add_word_to_history_list:  not adding empty string to context %s",
//			CTX_NAME(icp));
//		WARN(ERROR_STRING);
//		return;
//	}
//#endif /* CAUTIOUS */
	assert( string[0] != 0 );		/* don't add empty string */

//fprintf(stderr,"add_word_to_history_list, adding \"%s\" to context %s\n",string,CTX_NAME(icp));
	/* first see if this string is already on the list */

	PUSH_ITEM_CONTEXT(choice_itp,icp);
	hcp = choice_of(QSP_ARG  string);

	/* this is outside of the conditional (even though it may
	 * not be needed) in order to force item node creation,
	 * so that the initial list order matches that of the
	 * passed list.
	 */
//	lp = dictionary_list(CTX_DICT(icp));
	lp = container_list(CTX_CONTAINER(icp));
//fprintf(stderr,"add_word_to_history_list, container list has %d elements\n",eltcount(lp));

	if( hcp != NO_CHOICE ){

//fprintf(stderr,"found choice %s\n",ITEM_NAME((Item *)hcp));

#ifdef QUIP_DEBUG
if( debug & hist_debug ){
sprintf(ERROR_STRING,"add_word_to_history_list:  increasing priority for choice \"%s\"",string);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		/* we should boost the priority of an existing choice! */
		np = nodeOf(lp,hcp);
//#ifdef CAUTIOUS
//		if( np==NO_NODE ){
//	WARN("CAUTIOUS:  add_word_to_history_list can't find node of existing choice");
//			return;
//		}
//#endif	/* CAUTIOUS */
		assert( np != NO_NODE );

		np->n_pri++;
		p_sort(lp);
		pop_item_context(QSP_ARG  choice_itp);
//fprintf(stderr,"add_word_to_history_list, returning after increasing node priority\n");
		return;
	}

	/* make a new choice */

	hcp=new_choice(QSP_ARG  string);	// do we save this somewhere???
	pop_item_context(QSP_ARG  choice_itp);

//#ifdef CAUTIOUS
//	if( hcp==NO_CHOICE ){
//		ERROR1("CAUTIOUS: error creating menu choice");
//		IOS_RETURN
//	}
//#endif /* CAUTIOUS */
	assert( hcp != NO_CHOICE );
//fprintf(stderr,"add_word_to_history_list, returning after creating new choice\n");
}

void add_def( QSP_ARG_DECL  const char *prompt, const char *string )
{
	Item_Context *icp;

#ifdef QUIP_DEBUG
if( debug & hist_debug ){
sprintf(ERROR_STRING,"add_def \"%s\" for prompt \"%s\"",string,prompt);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	icp=find_hist(QSP_ARG  prompt);
	add_word_to_history_list(QSP_ARG  icp,string);
}

void rem_phist(QSP_ARG_DECL  const char *prompt,const char* word)
{
	char s[LLEN];

	sprintf(s,PROMPT_FORMAT,prompt);
	rem_def(QSP_ARG  s,word);
}

void add_phist(QSP_ARG_DECL  const char *prompt,const char* word)
{
	char s[LLEN];

	sprintf(s,PROMPT_FORMAT,prompt);
	add_def(QSP_ARG  s,word);
}

/* find a match to a partial response */

const char *get_match( QSP_ARG_DECL  const char *prompt, const char* so_far )
{
	Item_Context *icp;
	List *lp;
	Node *np;

	/* why not show the default when nothing is typed? */
	/* if( *so_far == 0 ) return(""); */

	if( *prompt == 0 ) return("");	/* e.g. hand entry of macros */

	icp=find_hist(QSP_ARG  prompt);

//	lp = dictionary_list(CTX_DICT(icp));
	lp = container_list(CTX_CONTAINER(icp));

	np=QLIST_HEAD(lp);

	while(np!=NO_NODE) {
		Hist_Choice *hcp;

		/* priority sorted!? */

		hcp=(Hist_Choice *) NODE_DATA(np);
		if( is_a_substring( so_far, hcp->hc_text ) ){
			cur_node=np;
			cur_list=lp;
			return(hcp->hc_text);
		}
		np=NODE_NEXT(np);
	}

	return("");
}

/* return the next element of the history list that matches a fragment */

#define NEXT_NODE								\
										\
	if( direction == CYC_FORWARD ){						\
		np=NODE_NEXT(np);							\
		if( np == NO_NODE )	/* at end of list */			\
			np=QLIST_HEAD(cur_list);					\
	} else {								\
		np=NODE_PREV(np);							\
		if( np == NO_NODE )						\
			np=QLIST_TAIL(cur_list);					\
	}

static const char *cyc_list_match(QSP_ARG_DECL  const char *so_far, int direction)
{
	Node *np, *first;
	Hist_Choice *hcp;


	first=np=cur_node;
	if( np == NO_NODE ) return("");

	NEXT_NODE

	while(np!=first){
		hcp=(Hist_Choice *) NODE_DATA(np);
		if( is_a_substring( so_far, hcp->hc_text ) ){
			cur_node=np;
			return(hcp->hc_text);
		}
		NEXT_NODE
	}

	/* nothing was accomplished - could ring bell or something? */

	hcp=(Hist_Choice *) NODE_DATA(first);
	return(hcp->hc_text);
}

const char *cyc_match(QSP_ARG_DECL  const char *sofar, int direction )
{
	if( QS_PICKING_ITEM_ITP(THIS_QSP) != NULL )
		cyc_tree_match(QSP_ARG  so_far, direction );
	else
		cyc_list_match(QSP_ARG  so_far, direction );
}

/* this was introduced to simplify the initialization of cmd menus */

void init_hist_from_list(QSP_ARG_DECL  const char *prompt,List* lp)

{
	Node *np;
	Item_Context *icp;

//#ifdef CAUTIOUS
//	if( lp == NO_LIST ){
//		ERROR1("CAUTIOUS:  init_hist_from_list passed null list");
//		IOS_RETURN
//	}
//#endif /* CAUTIOUS */
	assert( lp != NO_LIST );

#ifdef QUIP_DEBUG
	if( hist_debug == 0 )
		hist_debug = add_debug_module(QSP_ARG  "history");
#endif /* QUIP_DEBUG */

#ifdef QUIP_DEBUG
if( debug & hist_debug ){
sprintf(ERROR_STRING,"init_hist_from_list for prompt \"%s\"",prompt);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	icp=find_hist(QSP_ARG  prompt);

	np=QLIST_HEAD(lp);
	while(np!=NO_NODE){
		Item *ip;
		ip=(Item *) NODE_DATA(np);
		add_word_to_history_list(QSP_ARG  icp,ITEM_NAME(ip));
		np=NODE_NEXT(np);
	}
}

/* Add a list of words to a history list */

void init_hist_from_item_list(QSP_ARG_DECL  const char *prompt,List *lp)
{
	char s[LLEN];

//#ifdef CAUTIOUS
//	if( lp == NO_LIST ){
//		ERROR1("CAUTIOUS:  init_hist_from_list passed null list");
//		IOS_RETURN
//	}
//#endif /* CAUTIOUS */
	assert( lp != NO_LIST );

	sprintf(s,PROMPT_FORMAT,prompt);
	init_hist_from_list(QSP_ARG  s,lp);
}

void init_hist_from_class(QSP_ARG_DECL  const char* prompt,Item_Class *iclp)
{
	Node *np;
	List *lp;
	Member_Info *mip;
	Item_Context *icp;
	char s[LLEN];

	sprintf(s,PROMPT_FORMAT,prompt);

	icp = find_hist(QSP_ARG  s);

	if( icp != NO_ITEM_CONTEXT ){
		if( (iclp->icl_flags&NEED_CLASS_CHOICES)==0 ){

#ifdef QUIP_DEBUG
if( debug & hist_debug )
advise("init_hist_from_class:  don't need new choices");
#endif /* QUIP_DEBUG */

			return;
		}

#ifdef QUIP_DEBUG
		else {
if( debug & hist_debug )
advise("init_hist_from_class:  redoing class choices");
		}
#endif /* QUIP_DEBUG */

	}

#ifdef QUIP_DEBUG
	else {
if( debug & hist_debug )
advise("making new hist list for class");
	}
#endif /* QUIP_DEBUG */

	np=QLIST_HEAD(iclp->icl_lp);
	while(np!=NO_NODE){
		mip=(Member_Info *) NODE_DATA(np);
		lp = item_list(QSP_ARG  mip->mi_itp);
		if( lp != NO_LIST )
			init_hist_from_list(QSP_ARG  s,lp);
		np=NODE_NEXT(np);
	}
	iclp->icl_flags &= ~NEED_CLASS_CHOICES;
}

#endif /* HAVE_HISTORY */
