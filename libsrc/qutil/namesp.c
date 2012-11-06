
#include "quip_config.h"

char VersionId_qutil_namesp[] = QUIP_VERSION_STRING;

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
 */

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "query.h"	// error_string
#include "items.h"

#ifdef DEBUG
static u_long debug_namespace=NAMESP_DEBUG_MASK;
#endif /* DEBUG */


/* This is a tuneable parameter which determines when we abandon linear
 * search in favor of a hash table.
 */

#define MAX_CMPS_PER_SEARCH	3

Name_Space *create_namespace(const char *name)
{
	Name_Space *nsp;

	nsp=(Name_Space *) getbuf(sizeof(*nsp));
	if( nsp==NULL ) NERROR1("create_namespace:  out of memory");

	nsp->ns_name = savestr(name);
	nsp->ns_lp = new_list();
	nsp->ns_htp = NO_HASH_TBL;
	nsp->ns_flags = NS_LIST_IS_CURRENT;
	nsp->ns_ncmps = 0;
	nsp->ns_fetches = 0;
	return(nsp);
}

List *namespace_list(Name_Space *nsp)
{
	Node *np;

	if( nsp->ns_flags & NS_LIST_IS_CURRENT ){
		return(nsp->ns_lp);
	}

	/* list is not current, so we must be hashing, and have
	 * inserted or removed item(s) since the creation of the hash tbl.
	 */

#ifdef CAUTIOUS
	if( ! IS_HASHING(nsp) ){
		sprintf(DEFAULT_ERROR_STRING,
	"CAUTIOUS:  namespace_list:  namespace %s needs a new list but is not hashing!?",
			nsp->ns_name);
		NERROR1(DEFAULT_ERROR_STRING);
	}
#endif /* CAUTIOUS */

	/* trash the old list */
	while( (np=remHead(nsp->ns_lp)) != NO_NODE )
		rls_node(np);
	rls_list(nsp->ns_lp);

	nsp->ns_lp = ht_list(nsp->ns_htp);
	nsp->ns_flags |= NS_LIST_IS_CURRENT;
	return(nsp->ns_lp);
}

void delete_namespace(Name_Space *nsp)
{
	Node *np;

	/* delete the list */
	while( (np=remHead(nsp->ns_lp)) != NO_NODE )
		rls_node(np);
	rls_list(nsp->ns_lp);

	if( IS_HASHING(nsp) )
		zap_hash_tbl(nsp->ns_htp);

	rls_str((char *)nsp->ns_name);
	givbuf(nsp);	/* these are not items, so we return the mem */
}

Item *fetch_name(const char *name,Name_Space *nsp)
{
	Node *np;

	/* If we're not hashing, make sure we're not thrashing! */

	if( ( ! IS_HASHING(nsp) ) && nsp->ns_fetches > 8 ){
		float cmps_per_search;

		cmps_per_search = nsp->ns_ncmps / (float) nsp->ns_fetches;

		if( cmps_per_search > MAX_CMPS_PER_SEARCH ){	/* thrashing? */

			/* Create a hash table and load w/ existing items */
#ifdef DEBUG
if( debug & debug_namespace ){
sprintf(DEFAULT_ERROR_STRING,
"Creating hash table %s during search for word \"%s\" (0x%lx)",
nsp->ns_name, name,(u_long)name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */
			nsp->ns_htp = ht_init(nsp->ns_name);
#ifdef DEBUG
if( debug & debug_namespace ){
sprintf(DEFAULT_ERROR_STRING,
"hash table %s created, word \"%s\" (0x%lx)",
nsp->ns_name, name,(u_long)name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */
			/* initialize the hash table from the list */
			np=nsp->ns_lp->l_head;
			while(np!=NO_NODE){
				Item *ip;
				ip=(Item*) np->n_data;

#ifdef DEBUG
if( debug & debug_namespace ){
sprintf(DEFAULT_ERROR_STRING,
"inserting item %s into new hash table (word \"%s\" 0x%lx)",
ip->item_name,name,(u_long)name);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */
				if( insert_hash(ip,nsp->ns_htp) < 0 ){
sprintf(DEFAULT_ERROR_STRING,
"error inserting item name \"%s\" into new hash table",ip->item_name);
NWARN(DEFAULT_ERROR_STRING);
				}
				np=np->n_next;
			}
		}
	}

	/* if this namespace has a hash table , use it */

	if( IS_HASHING(nsp) ){
		Item *ip;

		ip = (Item*) fetch_hash(name,nsp->ns_htp);
#ifdef DEBUG
if( (debug & debug_namespace) ){
if( ip!=NO_ITEM ){
sprintf(DEFAULT_ERROR_STRING,"fetch_name:  fetch hash found %s when looking for %s",
ip->item_name,name);
NADVISE(DEFAULT_ERROR_STRING);
} else {
sprintf(DEFAULT_ERROR_STRING,"fetch_name:  %s (0x%lx) not found in hash table",
name,(u_long)name);
NADVISE(DEFAULT_ERROR_STRING);
}
}
#endif /* DEBUG */
		return(ip);
	}

	/* fetch from the list */

	nsp->ns_fetches ++;

	np=nsp->ns_lp->l_head;
	while(np!=NO_NODE){
		Item *ip;

		ip = (Item*) np->n_data;
		if( !strcmp(name,ip->item_name) )
			return(ip);

		nsp->ns_ncmps ++;
		np = np->n_next;
	}
	return(NO_ITEM);
}

int insert_name(Item* ip, Node* np, Name_Space* nsp)
{
	if( IS_HASHING(nsp) ){
		int stat;

		stat=insert_hash(ip,nsp->ns_htp);
		if( np!=NO_NODE )
			addTail(nsp->ns_lp,np);
		else
			nsp->ns_flags &= ~NS_LIST_IS_CURRENT;
		return(stat);
	}

	if( np == NO_NODE )
		np=mk_node(ip);

	/* BUG?  for commands, we want the first items to be at the head
	 * of the list, so that they are displayed in order...
	 * But for other items, we might want the most newly
	 * created objects to be at the head of the list?
	 */

	addTail(nsp->ns_lp,np);

	return(0);	/* success */
}

/* Append the items from this namespace to a given list */

void cat_ns_items(List *lp, Name_Space* nsp)
{
	Node *np, *new_np;
	List *nslp;

	nslp = namespace_list(nsp);
	if( nslp == NO_LIST ) return;

	np=nslp->l_head;
	while(np!=NO_NODE){
		new_np=mk_node(np->n_data);
		addTail(lp,new_np);
		np=np->n_next;
	}
}

int remove_name(Item *ip,Name_Space *nsp)
{
	Node *np;

	if( IS_HASHING(nsp) ){
		int stat;

		stat=remove_hash(ip,nsp->ns_htp);
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
		np=remData(nsp->ns_lp,ip);
		if( np!=NO_NODE ) rls_node(np);
		/* nsp->ns_flags &= ~NS_LIST_IS_CURRENT; */
		return(stat);
	}

	np=remData(nsp->ns_lp,ip);
	if( np==NO_NODE ) return(-1);
	rls_node(np);

	return(0);
}

void tell_name_stats(Name_Space *nsp)
{
	if( IS_HASHING(nsp) )
		tell_hash_stats(nsp->ns_htp);
	else {
		prt_msg("\tLinked list stats:");
		sprintf(msg_str,"\t%d name searches",nsp->ns_fetches);
		prt_msg(msg_str);
		sprintf(msg_str,"\t%d name comparisons",nsp->ns_ncmps);
		prt_msg(msg_str);
	}
}

void dump_ns_info(Name_Space *nsp)
{
	tell_name_stats(nsp);
}

