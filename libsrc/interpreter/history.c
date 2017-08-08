
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
ITEM_INTERFACE_CONTAINER(choice,LIST_CONTAINER)
ITEM_INTERFACE_PROTOTYPES(Hist_Choice,choice)
ITEM_INTERFACE_DECLARATIONS(Hist_Choice,choice,LIST_CONTAINER)

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

	if( choice_itp == NULL ) init_choices(SINGLE_QSP_ARG);

	ctxname = get_hist_ctx_name(prompt);
	icp = ctx_of(QSP_ARG  ctxname);
	if( icp == NULL ){
		icp = create_item_context(QSP_ARG  choice_itp,prompt);
		assert( icp != NULL );
	}

	return(icp);
}

static void rem_hcp(QSP_ARG_DECL  Item_Context *icp,Hist_Choice *hcp)
{
	PUSH_ITEM_CONTEXT(choice_itp,icp);
	/* BUG? this will search all contexts...
	 * BUT - we expect to find it in the first one!?
	 */
	del_item(QSP_ARG  choice_itp,hcp);
	pop_item_context(QSP_ARG  choice_itp);
}

/* Scan a history list, removing any choices which are not in the new list */

static void clr_defs_if(QSP_ARG_DECL  Item_Context *icp,int n,const char** choices)
{
	//Node *np;
	//List *lp;
	Enumerator *ep;

	//lp = dictionary_list(CTX_DICT(icp));
	//np=QLIST_HEAD(lp);
	ep = (CTX_CONTAINER(icp)->cnt_typ_p->new_enumerator)(CTX_CONTAINER(icp));
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

	if( hcp == NULL ){
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

	if( hcp != NULL ){
		boost_choice(QSP_ARG  hcp,lp);
		pop_item_context(QSP_ARG  choice_itp);
//fprintf(stderr,"add_word_to_history_list, returning after increasing node priority\n");
		return;
	}

	/* make a new choice */

	hcp=new_choice(QSP_ARG  string);	// do we save this somewhere???
	pop_item_context(QSP_ARG  choice_itp);

	assert( hcp != NULL );
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

static inline void insure_prompt_buf(QSP_ARG_DECL  const char *fmt, const char *pmpt)
{
	int n_need;

	n_need = (int) (strlen(fmt) + strlen(pmpt));
	if( n_need > sb_size(QS_QRY_PROMPT_SB(THIS_QSP)) )
		enlarge_buffer(QS_QRY_PROMPT_SB(THIS_QSP),n_need+32);
}

/* Make prompt takes a query string (like "number of elements") and
 * prepends "Enter " and appends ":  ".
 * We can inhibit this by clearing the flag.
 *
 * OLD COMMENT:
 * but in that case we reset the flag after use,
 * so that we can always assume the default behavior.
 * - what does that mean?  should we reset the flag here???
 */

const char *format_prompt(QSP_ARG_DECL  const char *fmt, const char *prompt)
{
	char *pline;

	if( prompt == QS_QRY_PROMPT_STR(THIS_QSP) ){
		return prompt;
	}

	insure_prompt_buf(QSP_ARG  fmt,prompt);
	pline = sb_buffer(QS_QRY_PROMPT_SB(THIS_QSP));

	if( QS_FLAGS(THIS_QSP) & QS_FORMAT_PROMPT ){
		sprintf(pline,fmt,prompt);
	} else {
		strcpy(pline,prompt);
	}

	return pline;
}

#ifdef NOT_USED
void rem_phist(QSP_ARG_DECL  const char *prompt,const char* word)
{
	const char *formatted_prompt;

	formatted_prompt = format_prompt(QSP_ARG  prompt);
	char s[LLEN];

	sprintf(s,PROMPT_FORMAT,prompt);
	rem_def(QSP_ARG  s,word);
}
#endif // NOT_USED

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

#ifdef NOT_USED_YET

static const char * cyc_tree_match(Frag_Match_Info *fmi_p, int direction )
{
	Item *ip;

	// there may be no items!?
	assert( fmi_p != NULL );

	if( direction == CYC_FORWARD ){
		if( fmi_p->fmi_u.rbti.curr_n_p == fmi_p->fmi_u.rbti.last_n_p )
			fmi_p->fmi_u.rbti.curr_n_p = fmi_p->fmi_u.rbti.first_n_p;
		else {
			fmi_p->fmi_u.rbti.curr_n_p = rb_successor_node( fmi_p->fmi_u.rbti.curr_n_p );
			assert( fmi_p->fmi_u.rbti.curr_n_p != NULL );
		}
	} else {
		if( fmi_p->fmi_u.rbti.curr_n_p == fmi_p->fmi_u.rbti.first_n_p )
			fmi_p->fmi_u.rbti.curr_n_p = fmi_p->fmi_u.rbti.last_n_p;
		else {
			fmi_p->fmi_u.rbti.curr_n_p = rb_predecessor_node( fmi_p->fmi_u.rbti.curr_n_p );
			assert( fmi_p->fmi_u.rbti.curr_n_p != NULL );
		}
	}
	ip = fmi_p->fmi_u.rbti.curr_n_p->data;
	return ip->item_name;
}
#endif // NOT_USED_YET

#ifdef NOT_USED_YET

static const char *cyc_item_list( Frag_Match_Info *fmi_p, int direction )
{
	Item *ip;

	assert( fmi_p != NULL );

	if( direction == CYC_FORWARD ){
		if( fmi_p->fmi_u.li.curr_np == fmi_p->fmi_u.li.last_np )
			fmi_p->fmi_u.li.curr_np = fmi_p->fmi_u.li.first_np;
		else {
			fmi_p->fmi_u.li.curr_np = NODE_NEXT(fmi_p->fmi_u.li.curr_np);
			assert( fmi_p->fmi_u.li.curr_np != NULL );
		}
	} else {
		if( fmi_p->fmi_u.li.curr_np == fmi_p->fmi_u.li.first_np )
			fmi_p->fmi_u.li.curr_np = fmi_p->fmi_u.li.last_np;
		else {
			fmi_p->fmi_u.li.curr_np = NODE_PREV( fmi_p->fmi_u.li.curr_np );
			assert( fmi_p->fmi_u.li.curr_np != NULL );
		}
	}
	ip = fmi_p->fmi_u.li.curr_np->n_data;
	return ip->item_name;
}
#endif // NOT_USED_YET

static const char * advance_frag_match( Frag_Match_Info * fmi_p, int direction )
{
	Container *cnt_p;

	cnt_p = CTX_CONTAINER(FMI_CTX(fmi_p));
	return (*(cnt_p->cnt_typ_p->advance_frag_match))(fmi_p,direction);
}

static const char * current_frag_match( Frag_Match_Info * fmi_p )
{
	Item *ip;
	Container *cnt_p;

	cnt_p = CTX_CONTAINER(FMI_CTX(fmi_p));
	ip = (*(cnt_p->cnt_typ_p->current_frag_match_item))(fmi_p);
	return ip->item_name;
}

static void reset_frag_match( Frag_Match_Info *fmi_p, int direction )
{
	Container *cnt_p;

	cnt_p = FMI_CONTAINER(fmi_p);
	(*(cnt_p->cnt_typ_p->reset_frag_match))(fmi_p,direction);
}

// We can have matches in different contexts on the context stack.  We keep a list that has matches
// we cycle the current one, and advance if we are at the end of the cycle for the current frag_match_info

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
	s = advance_frag_match(fmi_p,direction);
	if( s != NULL ) return s;

	if( direction == CYC_FORWARD ){
		np = NODE_NEXT(np);
		if( np == NULL )
			np = QLIST_HEAD( MATCH_CYCLE_LIST(mc_p) );
	} else {
		np = NODE_PREV(np);
		if( np == NULL )
			np = QLIST_TAIL( MATCH_CYCLE_LIST(mc_p) );
	}

	fmi_p = NODE_DATA(np);
	reset_frag_match(fmi_p,direction);

	return current_frag_match(fmi_p);
}

//fprintf(stderr,"cyc_item_match: frag_match_info '%s' has type %d\n",
//fmi_p->it.item_name,fmi_p->type);


	// find out what kind of container...
const char *cyc_match(QSP_ARG_DECL  const char *so_far, int direction )
{
	if( QS_PICKING_ITEM_ITP(THIS_QSP) != NULL ){
		return cyc_item_match(QSP_ARG  so_far, direction );
	}
	return cyc_list_match(QSP_ARG  so_far, direction );
}

/* this was introduced to simplify the initialization of cmd menus */

void init_hist_from_list(QSP_ARG_DECL  const char *prompt,List* lp)

{
	Node *np;
	Item_Context *icp;

	assert( lp != NULL );

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
	while(np!=NULL){
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

	assert( lp != NULL );
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
		lp = item_list(QSP_ARG  mip->mi_itp);
		if( lp != NULL )
			init_hist_from_list(QSP_ARG  s,lp);
		np=NODE_NEXT(np);
	}
	iclp->icl_flags &= ~NEED_CLASS_CHOICES;
}

#endif /* HAVE_HISTORY */
