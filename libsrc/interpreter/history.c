
#include "quip_config.h"

char VersionId_interpreter_history[] = QUIP_VERSION_STRING;

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

#include "items.h"
#include "history.h"
#include "debug.h"
#include "getbuf.h"
#include "savestr.h"
#include "substr.h"
#include "query.h"

int history=1;

#ifdef DEBUG
debug_flag_t hist_debug=0;
#endif /* DEBUG */

/* local variables */
static Node *cur_node;
static List *cur_list;

static char *get_hist_ctx_name(const char *prompt);

/* local prototypes */
static void rem_hcp(QSP_ARG_DECL  Item_Context *icp,Hist_Choice *hcp);
static void add_hl_def(QSP_ARG_DECL  Item_Context *icp,const char *string);
static void clr_defs_if(QSP_ARG_DECL  Item_Context *icp,int n,const char **choices);

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

	if( choice_itp == NO_ITEM_TYPE ) choice_init(SINGLE_QSP_ARG);

	ctxname = get_hist_ctx_name(prompt);
	icp = ctx_of(QSP_ARG  ctxname);
	if( icp == NO_ITEM_CONTEXT ){
		icp = create_item_context(QSP_ARG  choice_itp,prompt);
#ifdef CAUTIOUS
		if( icp == NO_ITEM_CONTEXT )
			ERROR1("CAUTIOUS:  error creating history context");
#endif /* CAUTIOUS */
	}

	return(icp);
}

/* Scan a history list, removing any choices which are not in the new list */

static void clr_defs_if(QSP_ARG_DECL  Item_Context *icp,int n,const char** choices)
{
	Node *np;
	List *lp;

	lp = namespace_list(icp->ic_nsp);
	np=lp->l_head;
	while(np!=NO_NODE){
		Hist_Choice *hcp;
		Node *next;
		int i, found;

		/* Because the item nodes get moved to the item free
		 * list when they are deleted, we have to get the next
		 * node BEFORE deletion!!!
		 */
		next=np->n_next;
		hcp = (Hist_Choice *) np->n_data;
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
		np=next;
	}
}

void set_defs(QSP_ARG_DECL  const char* prompt,unsigned int n,const char** choices)
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

	if( eltcount(namespace_list(icp->ic_nsp)) == n )
		return;

	/*
	 * Because of the preceding call to clr_defs_if(), now
	 * ec <= n, and we only have to add choices
	 */

#ifdef DEBUG
if( debug & hist_debug ){
sprintf(ERROR_STRING,"set_defs for prompt \"%s\"",prompt);
advise(ERROR_STRING);
}
#endif /* DEBUG */

	for(i=0;i<n;i++){
		add_hl_def(QSP_ARG  icp,choices[i]);
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

static void add_hl_def(QSP_ARG_DECL  Item_Context *icp,const char* string)
{
	Node *np;
	Hist_Choice *hcp;
	List *lp;

#ifdef CAUTIOUS
	if( string[0]==0 ) {		/* don't add empty string */
		sprintf(ERROR_STRING,
			"CAUTIOUS: add_hl_def:  not adding empty string to context %s",
			icp->ic_name);
		WARN(ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

	/* first see if this string is already on the list */

	PUSH_ITEM_CONTEXT(choice_itp,icp);
	hcp = choice_of(QSP_ARG  string);

	/* this is outside of the conditional (even though it may
	 * not be needed) in order to force item node creation,
	 * so that the initial list order matches that of the
	 * passed list.
	 */

	lp = namespace_list(icp->ic_nsp);

	if( hcp != NO_CHOICE ){
#ifdef DEBUG
if( debug & hist_debug ){
sprintf(ERROR_STRING,"add_hl_def:  increasing priority for choice \"%s\"",string);
advise(ERROR_STRING);
}
#endif /* DEBUG */
		/* we should boost the priority of an existing choice! */
		np = nodeOf(lp,hcp);
#ifdef CAUTIOUS
		if( np==NO_NODE ){
	WARN("CAUTIOUS:  add_hl_def can't find node of existing choice");
			return;
		}
#endif	/* CAUTIOUS */
		np->n_pri++;
		p_sort(lp);
		pop_item_context(QSP_ARG  choice_itp);
		return;
	}

	/* make a new choice */

	hcp=new_choice(QSP_ARG  string);
	pop_item_context(QSP_ARG  choice_itp);

#ifdef CAUTIOUS
	if( hcp==NO_CHOICE )
		ERROR1("CAUTIOUS: error creating menu choice");
#endif /* CAUTIOUS */
}

void add_def( QSP_ARG_DECL  const char *prompt, const char *string )
{
	Item_Context *icp;

#ifdef DEBUG
if( debug & hist_debug ){
sprintf(ERROR_STRING,"add_def \"%s\" for prompt \"%s\"",string,prompt);
advise(ERROR_STRING);
}
#endif /* DEBUG */

	icp=find_hist(QSP_ARG  prompt);
	add_hl_def(QSP_ARG  icp,string);
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

	lp = namespace_list(icp->ic_nsp);

	np=lp->l_head;

	while(np!=NO_NODE) {
		Hist_Choice *hcp;

		/* priority sorted!? */

		hcp=(Hist_Choice *) np->n_data;
		if( is_a_substring( so_far, hcp->hc_text ) ){
			cur_node=np;
			cur_list=lp;
			return(hcp->hc_text);
		}
		np=np->n_next;
	}

	return("");
}

/* return the next element of the history list that matches a fragment */

#define NEXT_NODE								\
										\
	if( direction == CYC_FORWARD ){						\
		np=np->n_next;							\
		if( np == NO_NODE )	/* at end of list */			\
			np=cur_list->l_head;					\
	} else {								\
		np=np->n_last;							\
		if( np == NO_NODE )						\
			np=cur_list->l_tail;					\
	}

const char *cyc_match(QSP_ARG_DECL  const char *so_far, int direction)
{
	Node *np, *first;
	Hist_Choice *hcp;


	first=np=cur_node;
	if( np == NO_NODE ) return("");

	NEXT_NODE

	while(np!=first){
		hcp=(Hist_Choice *) np->n_data;
		if( is_a_substring( so_far, hcp->hc_text ) ){
			cur_node=np;
			return(hcp->hc_text);
		}
		NEXT_NODE
	}

	/* nothing was accomplished - could ring bell or something? */

	hcp=(Hist_Choice *) first->n_data;
	return(hcp->hc_text);
}

/* this was introduced to simplify the initialization of cmd menus */

void init_hist_from_list(QSP_ARG_DECL  const char *prompt,List* lp)

{
	Node *np;
	Item_Context *icp;

#ifdef CAUTIOUS
	if( lp == NO_LIST )
		ERROR1("CAUTIOUS:  init_hist_from_list passed null list");
#endif /* CAUTIOUS */

#ifdef DEBUG
	if( hist_debug == 0 )
		hist_debug = add_debug_module(QSP_ARG  "history");
#endif /* DEBUG */

#ifdef DEBUG
if( debug & hist_debug ){
sprintf(ERROR_STRING,"init_hist_from_list for prompt \"%s\"",prompt);
advise(ERROR_STRING);
}
#endif /* DEBUG */

	icp=find_hist(QSP_ARG  prompt);

	np=lp->l_head;
	while(np!=NO_NODE){
		Item *ip;
		ip=(Item *) np->n_data;
		add_hl_def(QSP_ARG  icp,ip->item_name);
		np=np->n_next;
	}
}

/* Add a list of words to a history list */

void init_hist_from_item_list(QSP_ARG_DECL  const char *prompt,List *lp)
{
	char s[LLEN];

#ifdef CAUTIOUS
	if( lp == NO_LIST )
		ERROR1("CAUTIOUS:  init_hist_from_list passed null list");
#endif /* CAUTIOUS */

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

#ifdef DEBUG
if( debug & hist_debug )
advise("init_hist_from_class:  don't need new choices");
#endif /* DEBUG */

			return;
		}

#ifdef DEBUG
		else {
if( debug & hist_debug )
advise("init_hist_from_class:  redoing class choices");
		}
#endif /* DEBUG */

	}

#ifdef DEBUG
	else {
if( debug & hist_debug )
advise("making new hist list for class");
	}
#endif /* DEBUG */

	np=iclp->icl_lp->l_head;
	while(np!=NO_NODE){
		mip=(Member_Info *) np->n_data;
		lp = item_list(QSP_ARG  mip->mi_itp);
		if( lp != NO_LIST )
			init_hist_from_list(QSP_ARG  s,lp);
		np=np->n_next;
	}
	iclp->icl_flags &= ~NEED_CLASS_CHOICES;
}

#endif /* HAVE_HISTORY */
