
OBSOLET_FILE!!!

#include "quip_config.h"

/*
 * Name spaces
 *
 * We use name spaces to store and retrieve names, a function
 * which was previously handled by hash tables.  We still use
 * hash tables when advantageous, but the name space abstraction
 * allows us to do things in the most efficient way possible.
 * For small numbers of items, we just use a linked list and
 * forget about hashing.  For larger numbers of items we grow
 * a hash table.  When a list is requested, we either already
 * have it, or we build it from the hash table.
 *
 * For oq we rename name spaces dictionaries to make an analogy
 * with NSDictionary...
 */

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "debug.h"
#include "debug.h"

#ifdef QUIP_DEBUG
static u_long debug_dictionary=DICT_DEBUG_MASK;
#endif /* QUIP_DEBUG */


/* This is a tuneable parameter which determines when we abandon linear
 * search in favor of a hash table.
 */

#define MAX_CMPS_PER_SEARCH	3

Dictionary *create_dictionary(const char *name)
{
	Dictionary *dict_p;

	dict_p=(Dictionary *) getbuf(sizeof(*dict_p));
	if( dict_p==NULL ) {
		NERROR1("create_dictionary:  out of memory");
		IOS_RETURN_VAL(NULL)
	}
	
	SET_DICT_NAME(dict_p, savestr(name) );
	SET_DICT_LIST(dict_p, new_list() );
	SET_DICT_HT(dict_p, NULL);
	SET_DICT_FLAGS(dict_p, NS_LIST_IS_CURRENT);
	SET_DICT_N_COMPS(dict_p, 0);
	SET_DICT_N_FETCHES(dict_p, 0);
	return(dict_p);
}

List *_dictionary_list(QSP_ARG_DECL  Dictionary *dict_p)
{
	Node *np;

	if( DICT_FLAGS(dict_p) & NS_LIST_IS_CURRENT ){
		return(DICT_LIST(dict_p));
	}

	/* list is not current, so we must be hashing, and have
	 * inserted or removed item(s) since the creation of the hash tbl.
	 */

//#ifdef CAUTIOUS
//	if( ! IS_HASHING(dict_p) ){
//		sprintf(DEFAULT_ERROR_STRING,
//	"CAUTIOUS:  dictionary_list:  dictionary %s needs a new list but is not hashing!?",
//			DICT_NAME(dict_p));
//		NERROR1(DEFAULT_ERROR_STRING);
//		IOS_RETURN_VAL(NULL)
//	}
//#endif /* CAUTIOUS */
	assert( IS_HASHING(dict_p) );

	/* trash the old list */
	while( (np=remHead(DICT_LIST(dict_p))) != NULL )
		rls_node(np);
	rls_list(DICT_LIST(dict_p));

	SET_DICT_LIST(dict_p, ht_list(QSP_ARG  DICT_HT(dict_p)) );
	SET_DICT_FLAG_BITS(dict_p, NS_LIST_IS_CURRENT);
	return(DICT_LIST(dict_p));
}

void delete_dictionary(Dictionary *dict_p)
{
	Node *np;

	/* delete the list */
	while( (np=remHead(DICT_LIST(dict_p))) != NULL )
		rls_node(np);
	rls_list(DICT_LIST(dict_p));

	if( IS_HASHING(dict_p) )
		zap_hash_tbl(DICT_HT(dict_p));

	rls_str((char *)DICT_NAME(dict_p));
	givbuf(dict_p);	/* these are not items, so we return the mem */
}

// We create a hash table if the number of fetches is > 8.  This is completely
// arbitrary, and has never been tested to see when we actually break even.
//
// It is not clear that we should be using the fetch count for this trigger -
// unless we are sure that we attempt a fetch every time we create a new item.
// Otherwise, we might create many, many items, adding to a long list, and then
// We are considering switching from hash tables to red-black trees...

Item *fetch_name(const char *name,Dictionary *dict_p)
{
	Node *np;

	/* If we're not hashing, make sure we're not thrashing! */

	if( ( ! IS_HASHING(dict_p) ) && DICT_N_FETCHES(dict_p) > 8 ){
		float cmps_per_search;

		cmps_per_search = DICT_N_COMPS(dict_p) / (float) DICT_N_FETCHES(dict_p);

		if( cmps_per_search > MAX_CMPS_PER_SEARCH ){	/* thrashing? */

			/* Create a hash table and load w/ existing items */
#ifdef QUIP_DEBUG
if( debug & debug_dictionary ){
sprintf(DEFAULT_ERROR_STRING,
"Creating hash table %s during search for word \"%s\" (0x%lx)",
DICT_NAME(dict_p), name,(u_long)name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			SET_DICT_HT(dict_p, ht_init(DICT_NAME(dict_p)));
#ifdef QUIP_DEBUG
if( debug & debug_dictionary ){
sprintf(DEFAULT_ERROR_STRING,
"hash table %s created, word \"%s\" (0x%lx)",
DICT_NAME(dict_p), name,(u_long)name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
			/* initialize the hash table from the list */
			np=QLIST_HEAD(DICT_LIST(dict_p));
			while(np!=NULL){
				Item *ip;
				ip=(Item*) NODE_DATA(np);

#ifdef QUIP_DEBUG
if( debug & debug_dictionary ){
sprintf(DEFAULT_ERROR_STRING,
"inserting item %s into new hash table (word \"%s\" 0x%lx)",
ITEM_NAME(ip),name,(u_long)name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
				if( insert_hash(ip,DICT_HT(dict_p)) < 0 ){
sprintf(DEFAULT_ERROR_STRING,
"error inserting item name \"%s\" into new hash table",ITEM_NAME(ip));
NWARN(DEFAULT_ERROR_STRING);
				}
				np=NODE_NEXT(np);
			}
		}
	}

	/* if this dictionary has a hash table , use it */

	if( IS_HASHING(dict_p) ){
		Item *ip;

		ip = (Item*) fetch_hash(name,DICT_HT(dict_p));
#ifdef QUIP_DEBUG
if( (debug & debug_dictionary) ){
if( ip!=NULL ){
sprintf(DEFAULT_ERROR_STRING,"fetch_name:  fetch hash found %s when looking for %s",
ITEM_NAME(ip),name);
NADVISE(DEFAULT_ERROR_STRING);
} else {
sprintf(DEFAULT_ERROR_STRING,"fetch_name:  %s (0x%lx) not found in hash table",
name,(u_long)name);
NADVISE(DEFAULT_ERROR_STRING);
}
}
#endif /* QUIP_DEBUG */
		return(ip);
	}

	/* fetch from the list */

	SET_DICT_N_FETCHES(dict_p,
		DICT_N_FETCHES(dict_p) + 1 );

	np=QLIST_HEAD(DICT_LIST(dict_p));
	while(np!=NULL){
		Item *ip;

		ip = (Item*) NODE_DATA(np);
		if( !strcmp(name,ITEM_NAME(ip)) )
			return(ip);

		SET_DICT_N_COMPS(dict_p,
			DICT_N_COMPS(dict_p) + 1 );
		np = NODE_NEXT(np);
	}
	return(NULL);
} // end fetch_name

int insert_name(Item* ip, Node* np, Dictionary* dict_p)
{
	if( DICT_LIST(dict_p) == NULL )
		SET_DICT_LIST(dict_p,new_list());

	if( IS_HASHING(dict_p) ){
		int stat;

		stat=insert_hash(ip,DICT_HT(dict_p));
		if( np!=NULL )
			addTail(DICT_LIST(dict_p),np);
		else {
			CLEAR_DICT_FLAG_BITS(dict_p, NS_LIST_IS_CURRENT);
		}
		return(stat);
	} else {
		if( np == NULL )
			np=mk_node(ip);

		/* BUG?  for commands, we want the first items to be at the head
		 * of the list, so that they are displayed in order...
		 * But for other items, we might want the most newly
		 * created objects to be at the head of the list?
		 */

		addTail(DICT_LIST(dict_p),np);

		return(0);	/* success */
	}
}

/* Append the items from this dictionary to a given list */

void _cat_dict_items(QSP_ARG_DECL  List *lp, Dictionary* dict_p)
{
	Node *np, *new_np;
	List *nslp;

	nslp = dictionary_list(dict_p);
	if( nslp == NULL ) return;

	np=QLIST_HEAD(nslp);
	while(np!=NULL){
		new_np=mk_node(NODE_DATA(np));
		addTail(lp,new_np);
		np=NODE_NEXT(np);
	}
}

int remove_name(Item *ip,Dictionary *dict_p)
{
	Node *np;

	if( IS_HASHING(dict_p) ){
		int stat;

		stat=remove_hash(ip,DICT_HT(dict_p));
		/* Although we may not be using lists, a node ptr
		 * to this item was probably added to the list when
		 * created (see insert_name() above).
		 * This is always the case when an item is set
		 * up that came from the item free list...
		 * If we don't release the node here, AND
		 * we never have other occasion to update the list,
		 * then we can end up allocating another node when
		 * we add this item to the free list.
		 * Therefore, we need to make sure the node is
		 * released, if it exists.  It might be more
		 * efficient for the item to carry around it's own np...
		 */
		np=remData(DICT_LIST(dict_p),ip);
		if( np!=NULL ) rls_node(np);
		/* CLEAR_DICT_FLAG_BITS(dict_p, NS_LIST_IS_CURRENT); */
		return(stat);
	}

	np=remData(DICT_LIST(dict_p),ip);
	if( np==NULL ) return(-1);
	rls_node(np);

	return(0);
}

void tell_name_stats(QSP_ARG_DECL  Dictionary *dict_p)
{
	if( IS_HASHING(dict_p) )
		tell_hash_stats(QSP_ARG  DICT_HT(dict_p));
	else {
		prt_msg("\tLinked list stats:");
		sprintf(MSG_STR,"\t%d name searches",DICT_N_FETCHES(dict_p));
		prt_msg(MSG_STR);
		sprintf(MSG_STR,"\t%d name comparisons",DICT_N_COMPS(dict_p));
		prt_msg(MSG_STR);
	}
}

void _dump_dict_info(QSP_ARG_DECL  Dictionary *dict_p)
{
	tell_name_stats(QSP_ARG  dict_p);
}

Dictionary *new_dictionary(void)
{
	Dictionary *dict_p;

	dict_p = getbuf(sizeof(Dictionary));

	dict_p->dict_name = NULL;
	dict_p->dict_htp = NULL;
	dict_p->dict_lp = NULL;
	dict_p->dict_flags = 0;
	dict_p->dict_fetches = 0;
	dict_p->dict_ncmps = 0;
	return dict_p;
}

