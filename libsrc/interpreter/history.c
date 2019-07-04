
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

int history_flag=1;

#ifdef QUIP_DEBUG
debug_flag_t hist_debug=0;
#endif /* QUIP_DEBUG */

/* local variables */

// These maintain info about a list of possible command completions...
// These two vars allow us to cycle through multiple possibilities...
static Node *cur_node;	// might make more sense to put in qsp, but we will only ever have one
			// interactive shell...
static Node *cur_node;
static List *cur_list;

// BUT they are not so helpful when we also have item names that are not
// copied to a list!?


static char *get_hist_ctx_name(const char *prompt);

/* local prototypes */
static void add_word_to_history_list(QSP_ARG_DECL  Item_Context *icp,const char *string);

// need macro to make these all static
ITEM_INTERFACE_CONTAINER(hist_choice,LIST_CONTAINER)
ITEM_INTERFACE_PROTOTYPES(Hist_Choice,hist_choice)

#define init_hist_choices()	_init_hist_choices(SINGLE_QSP_ARG)
#define new_hist_choice(name)	_new_hist_choice(QSP_ARG  name)
#define hist_choice_of(name)	_hist_choice_of(QSP_ARG  name)

ITEM_INTERFACE_DECLARATIONS(Hist_Choice,hist_choice,LIST_CONTAINER)

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

Item_Context *_find_hist(QSP_ARG_DECL  const char *prompt)
{
	Item_Context *icp;
	char *ctxname;

	if( hist_choice_itp == NULL ) init_hist_choices();

	ctxname = get_hist_ctx_name(prompt);
	icp = ctx_of(ctxname);
	if( icp == NULL ){
		icp = create_item_context(hist_choice_itp,prompt);
		assert( icp != NULL );
	}

	return(icp);
}

static void rem_hcp(QSP_ARG_DECL  Item_Context *icp,Hist_Choice *hcp)
{
	push_item_context(hist_choice_itp,icp);
	/* BUG? this will search all contexts...
	 * BUT - we expect to find it in the first one!?
	 */
	del_item(hist_choice_itp,hcp);
	pop_item_context(hist_choice_itp);
}

/* Scan a history list, removing any choices which are not in the new list */

static void clr_defs_if(QSP_ARG_DECL  Item_Context *icp,int n,const char** choices)
{
	//Node *np;
	//List *lp;
	Enumerator *ep;

	//lp = dictionary_list(CTX_DICT(icp));
	//np=QLIST_HEAD(lp);
	ep = (CTX_CONTAINER(icp)->cnt_typ_p->new_enumerator)(QSP_ARG  CTX_CONTAINER(icp));
	if( ep == NULL ) return;

	while(ep!=NULL){
		Hist_Choice *hcp;
		int i, found;

		/* Because the item nodes get moved to the item free
		 * list when they are deleted, we have to get the next
		 * node BEFORE deletion!!!
		 */
		//hcp = (Hist_Choice *) enumerator_item(ep);
		hcp = (Hist_Choice *) ep->e_typ_p->current_enum_item(ep);
		ep = ep->e_typ_p->advance_enum(ep);

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
	}
}

void _preload_history_list(QSP_ARG_DECL  const char* prompt,unsigned int n,const char** choices)
{
	Item_Context *icp;
	unsigned int i;

	icp = find_hist(prompt);

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

void _rem_def(QSP_ARG_DECL  const char *prompt,const char* choice)	/** remove selection from list, return next */
{
	Item_Context *icp;
	Hist_Choice *hcp;

	icp = find_hist(prompt);

	/* We don't appear to use icp ??? */

	/* but it does create the context if it doesn't exist...
	 * but who cares???
	 */

	push_item_context(hist_choice_itp,icp);
	hcp = (Hist_Choice *) hist_choice_of(choice);
	pop_item_context(hist_choice_itp);

	if( hcp == NULL ){
		return;
	}

	rem_hcp(QSP_ARG  icp,hcp);
}

void _new_defs(QSP_ARG_DECL  const char* prompt)
{
	Item_Context *icp;

	icp = find_hist(prompt);

	clr_defs_if(QSP_ARG  icp,0,(const char **)NULL);
}

static inline void boost_choice(QSP_ARG_DECL  Hist_Choice *hcp, List *lp)
{
	Node *np;

#ifdef QUIP_DEBUG
if( debug & hist_debug ){
sprintf(ERROR_STRING,"boost_choice:  increasing priority for choice \"%s\"",ITEM_NAME((Item *)hcp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
		/* we should boost the priority of an existing choice! */
		np = nodeOf(lp,hcp);
		assert( np != NULL );

		np->n_pri++;
		p_sort(lp);
}

static void add_word_to_history_list(QSP_ARG_DECL  Item_Context *icp,const char* string)
{
	Hist_Choice *hcp;
	List *lp;

	assert( string[0] != 0 );		/* don't add empty string */

//fprintf(stderr,"add_word_to_history_list, adding \"%s\" to context %s\n",string,CTX_NAME(icp));
	/* first see if this string is already on the list */

	push_item_context(hist_choice_itp,icp);
	hcp = hist_choice_of(string);

	/* this is outside of the conditional (even though it may
	 * not be needed) in order to force item node creation,
	 * so that the initial list order matches that of the
	 * passed list.
	 */
//	lp = dictionary_list(CTX_DICT(icp));
	lp = container_list(CTX_CONTAINER(icp));
//fprintf(stderr,"add_word_to_history_list, container list has %d elements\n",eltcount(lp));

	if( hcp != NULL ){
		boost_choice(QSP_ARG  hcp,lp);
		pop_item_context(hist_choice_itp);
//fprintf(stderr,"add_word_to_history_list, returning after increasing node priority\n");
		return;
	}

	/* make a new choice */

	hcp=new_hist_choice(string);	// do we save this somewhere???
	pop_item_context(hist_choice_itp);

	assert( hcp != NULL );
//fprintf(stderr,"add_word_to_history_list, returning after creating new choice\n");
}

void _add_def( QSP_ARG_DECL  const char *prompt, const char *string )
{
	Item_Context *icp;

#ifdef QUIP_DEBUG
if( debug & hist_debug ){
sprintf(ERROR_STRING,"add_def \"%s\" for prompt \"%s\"",string,prompt);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	icp=find_hist(prompt);
	add_word_to_history_list(QSP_ARG  icp,string);
}

#ifdef NOT_USED
void _rem_phist(QSP_ARG_DECL  const char *prompt,const char* word)
{
	const char *formatted_prompt;

	formatted_prompt = format_prompt(prompt);
	char s[LLEN];

	sprintf(s,PROMPT_FORMAT,prompt);
	rem_def(QSP_ARG  s,word);
}
#endif // NOT_USED

void _add_phist(QSP_ARG_DECL  const char *prompt,const char* word)
{
	char s[LLEN];

	sprintf(s,PROMPT_FORMAT,prompt);
	add_def(s,word);
}

/* find a match to a partial response */

const char *_get_match( QSP_ARG_DECL  const char *prompt, const char* so_far )
{
	Item_Context *icp;
	List *lp;
	Node *np;

	/* why not show the default when nothing is typed? */
	/* if( *so_far == 0 ) return(""); */

	if( *prompt == 0 ) return("");	/* e.g. hand entry of macros */

	icp=find_hist(prompt);

	lp = container_list(CTX_CONTAINER(icp));

	np=QLIST_HEAD(lp);

	while(np!=NULL) {
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
		if( np == NULL )	/* at end of list */			\
			np=QLIST_HEAD(cur_list);					\
	} else {								\
		np=NODE_PREV(np);							\
		if( np == NULL )						\
			np=QLIST_TAIL(cur_list);					\
	}

static const char *cyc_list_match(QSP_ARG_DECL  const char *so_far, int direction)
{
	Node *np, *first;
	Hist_Choice *hcp;

	first=np=cur_node;
	if( np == NULL ) return("");

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

#define advance_frag_match(fmi_p,direction) _advance_frag_match(QSP_ARG  fmi_p,direction)

static const char *_advance_frag_match(QSP_ARG_DECL  Frag_Match_Info * fmi_p, int direction )
{
	Container *cnt_p;

	cnt_p = CTX_CONTAINER(FMI_CTX(fmi_p));
	return (*(cnt_p->cnt_typ_p->advance_frag_match))(QSP_ARG  fmi_p,direction);
}

static const char * current_frag_match( Frag_Match_Info * fmi_p )
{
	const Item *ip;
	Container *cnt_p;

	cnt_p = CTX_CONTAINER(FMI_CTX(fmi_p));
	ip = (*(cnt_p->cnt_typ_p->current_frag_match_item))(fmi_p);
	// is this check needed?
	if( ip == NULL ) return NULL;
	else return ip->item_name;
}

#define reset_frag_match(fmi_p,direction) _reset_frag_match(QSP_ARG  fmi_p,direction)

static void _reset_frag_match(QSP_ARG_DECL  Frag_Match_Info *fmi_p, int direction )
{
	Container *cnt_p;

	cnt_p = FMI_CONTAINER(fmi_p);
	(*(cnt_p->cnt_typ_p->reset_frag_match))(QSP_ARG  fmi_p,direction);
}

static inline Node *cycle_match_info( Node *np, int direction, Match_Cycle *mc_p )
{
	if( direction == CYC_FORWARD ){
		np = NODE_NEXT(np);
		if( np == NULL )
			np = QLIST_HEAD( MATCH_CYCLE_LIST(mc_p) );
	} else {
		np = NODE_PREV(np);
		if( np == NULL )
			np = QLIST_TAIL( MATCH_CYCLE_LIST(mc_p) );
	}
	return np;
}

// We can have matches in different contexts on the context stack.  For each context,
// we keep a list that has matches; we cycle the current one, and advance if we are
// at the end of the cycle for the current frag_match_info

static const char * cyc_item_match(QSP_ARG_DECL  const char *so_far, int direction )
{
	Frag_Match_Info *fmi_p;
	Match_Cycle *mc_p;
	const char *s;
	Node *np;

	assert( QS_PICKING_ITEM_ITP(THIS_QSP) != NULL );

	//fmi_p = IT_FRAG_MATCH_INFO( QS_PICKING_ITEM_ITP(THIS_QSP) );
	//if( fmi_p == NULL ) return so_far;

	mc_p = IT_MATCH_CYCLE( QS_PICKING_ITEM_ITP(THIS_QSP) );
	//assert(mc_p!=NULL);
	if( mc_p == NULL ) return so_far;

	np = mc_p->mc_curr_np;
	assert(np!=NULL);

	fmi_p = NODE_DATA(np);

//sprintf(ERROR_STRING,"cyc_item_match \"%s\", will call advance_frag_match...",so_far);
//advise(ERROR_STRING);
	s = advance_frag_match(fmi_p,direction);
	if( s != NULL ){
		return s;
	}

	do {
		np = cycle_match_info( np, direction, mc_p );

		fmi_p = NODE_DATA(np);
		reset_frag_match(fmi_p,direction);

		s = current_frag_match(fmi_p);
		// s can be null if there are no matches in this context...
		if( s != NULL ){
			return s;
		}
		// Can this become an infinite loop?
	} while(1);
}

//fprintf(stderr,"cyc_item_match: frag_match_info '%s' has type %d\n",
//fmi_p->it.item_name,fmi_p->type);


	// find out what kind of container...
const char *_cyc_match(QSP_ARG_DECL  const char *so_far, int direction )
{
	if( QS_PICKING_ITEM_ITP(THIS_QSP) != NULL ){
//advise("cyc_match will return cyc_item_match...");
		return cyc_item_match(QSP_ARG  so_far, direction );
	}
	return cyc_list_match(QSP_ARG  so_far, direction );
}

/* this was introduced to simplify the initialization of cmd menus */

void _init_hist_from_list(QSP_ARG_DECL  const char *prompt,List* lp)

{
	Node *np;
	Item_Context *icp;

	assert( lp != NULL );

#ifdef QUIP_DEBUG
	if( hist_debug == 0 )
		hist_debug = add_debug_module("history");
#endif /* QUIP_DEBUG */

#ifdef QUIP_DEBUG
if( debug & hist_debug ){
sprintf(ERROR_STRING,"init_hist_from_list for prompt \"%s\"",prompt);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	icp=find_hist(prompt);

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		Item *ip;
		ip=(Item *) NODE_DATA(np);
		add_word_to_history_list(QSP_ARG  icp,ITEM_NAME(ip));
		np=NODE_NEXT(np);
	}
}

/* Add a list of words to a history list */

void _init_hist_from_item_list(QSP_ARG_DECL  const char *prompt,List *lp)
{
	char s[LLEN];

	assert( lp != NULL );
	sprintf(s,PROMPT_FORMAT,prompt);
	init_hist_from_list(s,lp);
}

void _init_hist_from_class(QSP_ARG_DECL  const char* prompt,Item_Class *iclp)
{
	Node *np;
	List *lp;
	Member_Info *mip;
	Item_Context *icp;
	char s[LLEN];

	sprintf(s,PROMPT_FORMAT,prompt);

	icp = find_hist(s);

	if( icp != NULL ){
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
	while(np!=NULL){
		mip=(Member_Info *) NODE_DATA(np);
		lp = item_list(mip->mi_itp);
		if( lp != NULL )
			init_hist_from_list(s,lp);
		np=NODE_NEXT(np);
	}
	iclp->icl_flags &= ~NEED_CLASS_CHOICES;
}

#endif /* HAVE_HISTORY */

