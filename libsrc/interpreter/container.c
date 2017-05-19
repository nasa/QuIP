#include "quip_config.h"
#include "quip_prot.h"
#include "container.h"
#include "getbuf.h"
#include "history.h"
#include <assert.h>
#include <string.h>

#define HAS_VALID_CONTAINER_TYPE(cnt_p)	VALID_CONTAINER_TYPE( CONTAINER_TYPE(cnt_p) )

#define VALID_CONTAINER_TYPE(t)	( t == LIST_CONTAINER || t == HASH_TBL_CONTAINER || t == RB_TREE_CONTAINER )

// forward declaration
static void init_container(Container *cnt_p,int type);

Container * create_container(const char *name,int type)
{
	Container *cnt_p;

	if( type == 0 ) type=HASH_TBL_CONTAINER;	// default
	cnt_p = new_container(type);
	cnt_p->name = savestr(name);
	return cnt_p;
}

#ifdef FOOBAR
static void add_type_to_container(Container *cnt_p, int type)
{
	switch(type){
		case LIST_CONTAINER:
			cnt_p->cnt_lp = new_list();
			break;
		case HASH_TBL_CONTAINER:
			cnt_p->cnt_htp = ht_init(NULL);
			break;
		case RB_TREE_CONTAINER:
			cnt_p->cnt_tree_p = create_rb_tree();
			break;
		default:
			// could be assertion?
			sprintf(DEFAULT_ERROR_STRING,"add_type_to_container:  Invalid container type code %d",type);
			NERROR1(DEFAULT_ERROR_STRING);
			break;
	}
}
#endif // FOOBAR

#ifdef FOOBAR
static void make_container_current( Container *cnt_p, int type )
{
	switch(type){
		case LIST_CONTAINER:
			if( cnt_p->cnt_lp == NULL ){
				cnt_p->cnt_lp = new_list();
				cnt_p->types |= LIST_CONTAINER;
			} else {
				// BUG release the old nodes from the list!?
			}
			cat_container_items(cnt_p->cnt_lp,cnt_p);
			break;
		case HASH_TBL_CONTAINER:
			NERROR1("make_container_current:  Sorry, can't transfer to hash table!?");
			break;
		case RB_TREE_CONTAINER:
			NERROR1("make_container_current:  Sorry, can't transfer to red-black tree!?");
			break;
		default:
			NERROR1("make_container_current:  bad container type!?");
			break;
	}
	cnt_p->is_current |= type;
}
#endif // FOOBAR

void set_container_type(Container *cnt_p, int type)
{
	assert(type!=0);

#ifdef FOOBAR
	if( type == 0 ) return;	// keep default

	if( ! (cnt_p->types & type) ){
		add_type_to_container(cnt_p,type);
		cnt_p->types |= type;
	}
	make_container_current(cnt_p,type);
	cnt_p->primary_type = type;
#endif // FOOBAR

	// This code used to be add_type_to_container
	switch(type){
		case LIST_CONTAINER:
			cnt_p->cnt_lp = new_list();
			break;
		case HASH_TBL_CONTAINER:
			cnt_p->cnt_htp = ht_init(NULL);
			break;
		case RB_TREE_CONTAINER:
			cnt_p->cnt_tree_p = create_rb_tree();
			break;
		default:
			// could be assertion?
			sprintf(DEFAULT_ERROR_STRING,"add_type_to_container:  Invalid container type code %d",type);
			NERROR1(DEFAULT_ERROR_STRING);
			break;
	}
}

static void *list_insert_item(Container *cnt_p, Item *ip)
{
	Node *np;
	np = mk_node(ip);
	addTail( cnt_p->cnt_lp, np );
	return np;
}

static void *hash_tbl_insert_item(Container *cnt_p,Item *ip)
{
	int stat;
	stat=insert_hash(ip,cnt_p->cnt_htp);
	if( stat == 0 )
		return ip;
	return NULL;
}

static void *rb_tree_insert_item(Container *cnt_p,Item *ip)
{
	return rb_insert_item(cnt_p->cnt_tree_p, ip );
}

Container * new_container(int type)
{
	Container *cnt_p=NULL;

	assert( VALID_CONTAINER_TYPE(type) );

	cnt_p = getbuf( sizeof(Container) );
	init_container(cnt_p,type);
	set_container_type(cnt_p,type);
	return cnt_p;
}

int add_to_container(Container *cnt_p, Item *ip)
{
//	qrb_node *tnp;
	void *ptr;
//	int stat=0;

#ifdef FOOBAR
	switch(cnt_p->primary_type){
		case LIST_CONTAINER:
			np = mk_node(ip);
			// add to head so that temp items can be found
			// easily for later deletion?
			//addHead( cnt_p->cnt_lp, np );
			addTail( cnt_p->cnt_lp, np );
			break;
		case HASH_TBL_CONTAINER:
			stat=insert_hash(ip,cnt_p->cnt_htp);
			break;
		case RB_TREE_CONTAINER:
			tnp = rb_insert_item(cnt_p->cnt_tree_p, ip );
			if( tnp == NULL ) stat = (-1);
			break;
		default:
			NERROR1("add_to_container:  invalid container type!?");
			break;
	}
#endif // FOOBAR
	ptr = (*(cnt_p->insert_item_func))(cnt_p,ip);

	if( ptr == NULL )
		return -1;
	return 0;
}

int remove_name_from_container(QSP_ARG_DECL  Container *cnt_p, const char *name)
{
	Node *np;
	int stat=0;

	switch(cnt_p->primary_type){
		case LIST_CONTAINER:
			np = list_find_named_item(cnt_p->cnt_lp,name);
			np=remNode(cnt_p->cnt_lp,np);
			if( np!=NO_NODE ) rls_node(np);
			break;
		case HASH_TBL_CONTAINER:
			stat=remove_name_from_hash(name,cnt_p->cnt_htp);
			break;
		case RB_TREE_CONTAINER:
			stat = rb_delete_key(cnt_p->cnt_tree_p, name );
			break;
		default:
			NERROR1("remove_from_container:  invalid container type!?");
			break;
	}
//	cnt_p->is_current = cnt_p->primary_type;
	return stat;
}

Item *container_find_match(Container *cnt_p, const char *name)
{
	Item *ip;
	Node *np;
	qrb_node *tnp;

	switch(cnt_p->primary_type){
		case LIST_CONTAINER:
			np = list_find_named_item(cnt_p->cnt_lp,name);
			if( np != NULL )	ip = NODE_DATA(np);
			else			ip = NULL;
			break;
		case HASH_TBL_CONTAINER:
			ip = (Item*) fetch_hash(name,cnt_p->cnt_htp);
			break;
		case RB_TREE_CONTAINER:
			tnp = rb_find(cnt_p->cnt_tree_p,name);
			if( tnp != NULL )
				ip = tnp->data;
			else 	ip = NULL;
			break;
		default:
			ip = NULL; // QUIET COMPILER
			NERROR1("container_find_match:  invalid container type!?");
			return NULL;
			// never returns, but compiler may not know...
			break;
	}
	return ip;
}

static void search_list_for_fragment(List *lp, Frag_Match_Info *fmi_p, const char *frag)
{
	int n;
	Node* np;

	lp = alpha_sort(DEFAULT_QSP_ARG  lp);	// BUG should sort in-place???

	np = QLIST_HEAD(lp);

	n = (int) strlen(frag);
	fmi_p->fmi_u.li.curr_np = NULL;	// default
	fmi_p->fmi_u.li.first_np = NULL;
	fmi_p->fmi_u.li.last_np = NULL;

	while( np != NULL ){
		Item *ip;
		int compVal;

		ip = NODE_DATA(np);
		compVal = strncmp( frag, ITEM_NAME(ip), n );
		if( compVal == 0 ){
			// We have found the first node that is a match,
			// but we want also determine the last...
			fmi_p->fmi_u.li.curr_np = np;
			fmi_p->fmi_u.li.first_np = np;
			fmi_p->fmi_u.li.last_np = np;
			np = NODE_NEXT(np);
			while( np != NULL ){
				ip = NODE_DATA(np);
				compVal = strncmp( frag, ITEM_NAME(ip), n );
				if( compVal != 0 )
					return;
				fmi_p->fmi_u.li.last_np = np;
				np = NODE_NEXT(np);
			}
			return;
		}
		np = NODE_NEXT(np);
	}
}

static void list_substring_find( Frag_Match_Info *fmi_p, const char *frag )
{
	Container *cnt_p;

	cnt_p = FMI_CONTAINER(fmi_p);

	search_list_for_fragment(cnt_p->cnt_lp,fmi_p,frag);
}

static Container *ht_list_container(Hash_Tbl *htp)
{
	Container *cnt_p;

	cnt_p = new_container(LIST_CONTAINER);
	cnt_p->cnt_lp = ht_list(htp);
	return cnt_p;
}

static void zap_list_container(Container *cnt_p)
{
	assert(cnt_p->primary_type == LIST_CONTAINER);
	zap_list(cnt_p->cnt_lp);	// what does "zap" mean?  BUG?
}

// Because the hash table items are not sorted, we need to keep a list of the items...

static void hash_tbl_substring_find(Frag_Match_Info *fmi_p,const char *frag)
{
	Container *cnt_p;
	Container *list_cnt_p;

	cnt_p = FMI_CONTAINER(fmi_p);
	list_cnt_p = ht_list_container(cnt_p->cnt_htp);
	// fmi_p still points to the hash_tbl container...
	search_list_for_fragment(list_cnt_p->cnt_lp,fmi_p,frag);
	zap_list_container(list_cnt_p);
	// BUG memory leak?
}

static void rb_tree_substring_find(Frag_Match_Info *fmi_p, const char *frag )
{
	Container *cnt_p;

	cnt_p = FMI_CONTAINER(fmi_p);
	rb_substring_find( fmi_p, cnt_p->cnt_tree_p, frag );
}


// container_find_substring_matches
//
// It the container is a list, it's pretty easy, we just traverse the list...
//
// Similarly, for an rbtree, we should be even quicker, because we can find the start (like
// an insertion), and then traverse until we hit a mismatch.
//
// The hash table is a little more complicated, because we have to scan the entire table.
// We used to cache the corresponding list...
//
// We used to call make_container_current to map things to a list...

void container_find_substring_matches(Frag_Match_Info *fmi_p, const char *frag)
{
	Container *cnt_p;
	cnt_p = FMI_CONTAINER(fmi_p);
	(*(cnt_p->substring_find_func))(fmi_p,frag);
}

Enumerator *advance_enumerator(Enumerator *ep)
{
	switch( ENUMERATOR_TYPE(ep) ){
		case LIST_CONTAINER:
			advance_list_enumerator(ep->e_p.lep);
			if( list_enumerator_item(ep->e_p.lep) == NULL )
				return NULL;
			else
				return ep;
			break;
		case HASH_TBL_CONTAINER:
			advance_ht_enumerator(ep->e_p.htep);
			if( ht_enumerator_item(ep->e_p.htep) == NULL )
				return NULL;
			else
				return ep;
			break;
		case RB_TREE_CONTAINER:
			advance_rbtree_enumerator(ep->e_p.rbtep);
			if( rbtree_enumerator_item(ep->e_p.rbtep) == NULL )
				return NULL;
			else
				return ep;
			break;
		default:
			NERROR1("advance_enumerator:  bad container type!?");
			return NULL;
			break;
	}
}

static void rls_enumerator( Enumerator *ep )
{
	// Do type-specific releasing first
	switch( ENUMERATOR_TYPE(ep) ){
		case LIST_CONTAINER:
			rls_list_enumerator(ep->e_p.lep);
			break;
		case HASH_TBL_CONTAINER:
			rls_hash_tbl_enumerator(ep->e_p.htep);
			break;
		case RB_TREE_CONTAINER:
			rls_rbtree_enumerator(ep->e_p.rbtep);
			break;
		default:
			NERROR1("rls_enumerator:  bad container type!?");
			break;
	}
	givbuf(ep);
}
// add the items from this container to the given list

void cat_container_items(List *lp, Container *cnt_p)
{
	Node *np;
	Item *ip;
	Enumerator *ep, *orig;

	//ep = new_enumerator(cnt_p);
	ep = (*(cnt_p->new_enumerator_func))(cnt_p);

	if( ep == NULL ) return;	// enumerator is null if the container is empty

	orig = ep;

	while( ep != NULL ){
		ip = enumerator_item(ep);
		if( ip != NULL ){
			np = mk_node(ip);
			addTail(lp,np);
		}
		ep = advance_enumerator(ep);
	}
	rls_enumerator(orig);
}

long container_eltcount(Container *cnt_p)
{
	switch(cnt_p->primary_type){
		case LIST_CONTAINER:
			return eltcount(cnt_p->cnt_lp);
			break;
		case HASH_TBL_CONTAINER:
			return cnt_p->cnt_htp->ht_n_entries;
			break;
		case RB_TREE_CONTAINER:
			return rb_node_count(cnt_p->cnt_tree_p);
			break;
		default:
			NERROR1("container_eltcount:  bad container type!?");
			return 0;
			break;
	}
}

List *container_list(Container *cnt_p)
{
	//if( cnt_p->is_current & LIST_CONTAINER )
	//	return cnt_p->cnt_lp;
	switch( CONTAINER_TYPE(cnt_p) ){
		case LIST_CONTAINER:
			return cnt_p->cnt_lp;
			break;
		case HASH_TBL_CONTAINER:
			return hash_tbl_list(cnt_p->cnt_htp);
			break;
		case RB_TREE_CONTAINER:
			return rbtree_list(cnt_p->cnt_tree_p);
			break;
		default:
			NERROR1("container_list:  bad container type!?");
			break;
	}
	return NULL;

	/*
	cat_container_items(cnt_p->cnt_lp,cnt_p);
	cnt_p->is_current |= LIST_CONTAINER;
	return cnt_p->cnt_lp;
	*/
}

void delete_container(Container *cnt_p)
{
	switch(CONTAINER_TYPE(cnt_p)){
		case LIST_CONTAINER:
			zap_list(cnt_p->cnt_lp);
			break;
		case HASH_TBL_CONTAINER:
			zap_hash_tbl(cnt_p->cnt_htp);
			break;
		case RB_TREE_CONTAINER:
			release_rb_tree(cnt_p->cnt_tree_p);
			break;
		default:
			NERROR1("delete_container:  bad container type!?");
			break;
	}
	rls_str((char *)cnt_p->name);
	givbuf(cnt_p);
}

void dump_container_info(QSP_ARG_DECL  Container *cnt_p)
{
	sprintf(MSG_STR,"Container %s:\n",cnt_p->name==NULL?"<null>":cnt_p->name);
	prt_msg(MSG_STR);

	switch(CONTAINER_TYPE(cnt_p)){
		case LIST_CONTAINER:
			sprintf(MSG_STR,"\tlist with %d elements\n",eltcount(cnt_p->cnt_lp));
			prt_msg(MSG_STR);
			break;
		case HASH_TBL_CONTAINER:
			tell_hash_stats(QSP_ARG  cnt_p->cnt_htp );
			break;
		case RB_TREE_CONTAINER:
			prt_msg("\tRed-black tree, sorry no stats...");
			break;
		default:
			NERROR1("dump_container_info:  bad container type!?");
			break;
	}
}

void * enumerator_item(Enumerator *ep)
{
	switch( ENUMERATOR_TYPE(ep) ){
		case LIST_CONTAINER:
			return NODE_DATA(ep->e_p.lep->np);
			break;
		case HASH_TBL_CONTAINER:
			if( ep->e_p.htep->current_entry == NULL )
				return NULL;
			else {
				return *(ep->e_p.htep->current_entry);
			}
			break;
		case RB_TREE_CONTAINER:
			return ep->e_p.rbtep->node_p->data;
			break;
		default:
			NERROR1("enumerated_data:  bad container type!?");
			return NULL;
			break;
	}
}

static Enumerator *create_enumerator(Container *cnt_p, void *ptr)
{
	Enumerator *ep;

	if( ptr == NULL ) return NULL;

	ep = getbuf( sizeof(Enumerator) );

	ep->e_cnt_p = cnt_p;	// is this used?
	ep->e_p.vp = ptr;
	return ep;
}

static Enumerator *list_create_enumerator(Container *cnt_p)
{
	List_Enumerator *lep;

	lep = new_list_enumerator(cnt_p->cnt_lp);
	return create_enumerator(cnt_p, lep);
}

static Enumerator *rb_tree_create_enumerator(Container *cnt_p)
{
	void *vp;

	vp = new_rbtree_enumerator(cnt_p->cnt_tree_p);
	return create_enumerator(cnt_p, vp);
}

static Enumerator *hash_tbl_create_enumerator(Container *cnt_p)
{
	void *vp;

	vp = new_hash_tbl_enumerator(cnt_p->cnt_htp);
	return create_enumerator(cnt_p, vp);
}

Enumerator *new_enumerator(Container *cnt_p )
{
	return (*(cnt_p->new_enumerator_func))(cnt_p);
}

#ifdef FOOBAR
Enumerator *new_enumerator(Container *cnt_p )
{
	void *vp;
	Enumerator *ep;

	assert( HAS_VALID_CONTAINER_TYPE(cnt_p) );

	switch( CONTAINER_TYPE(cnt_p) ){
		case LIST_CONTAINER:
			vp = new_list_enumerator(cnt_p->cnt_lp);
			break;
		case HASH_TBL_CONTAINER:
			vp = new_hash_tbl_enumerator(cnt_p->cnt_htp);
			break;
		case RB_TREE_CONTAINER:
			vp = new_rbtree_enumerator(cnt_p->cnt_tree_p);
			break;
		default:
			NERROR1("new_enumerator:  bad container type!?");
			vp = NULL;
			break;
	}
	if( vp == NULL ) return NULL;

	ep = getbuf( sizeof(Enumerator) );

	ep->e_cnt_p = cnt_p;	// is this used?
	ep->e_p.vp = vp;

	return ep;
}
#endif // FOOBAR

static Item *rb_tree_frag_item(Frag_Match_Info *fmi_p)
{
	if( CURR_RBT_FRAG(fmi_p) == NULL ) return NULL;

	return RB_NODE_DATA( CURR_RBT_FRAG(fmi_p) );
}

static Item *list_frag_item(Frag_Match_Info *fmi_p)
{
	if( CURR_LIST_FRAG(fmi_p) == NULL ) return NULL;
	return NODE_DATA( CURR_LIST_FRAG(fmi_p) );
}

static Item *hash_tbl_frag_item(Frag_Match_Info *fmi_p)
{
	return list_frag_item(fmi_p);
}


Item *current_frag_item( Frag_Match_Info *fmi_p )
{
	Container *cnt_p;

	cnt_p = CTX_CONTAINER(FMI_CTX(fmi_p));
	return (*(cnt_p->frag_item_func))(fmi_p);
}

static Item *list_current_item(Frag_Match_Info *fmi_p)
{
	return fmi_p->fmi_u.li.curr_np->n_data;
}

static Item *rb_tree_current_item(Frag_Match_Info *fmi_p)
{
	return fmi_p->fmi_u.rbti.curr_n_p->data;
}

static Item *hash_tbl_current_item(Frag_Match_Info *fmi_p)
{
	assert( "hash_tbl_current_item should never be called" == NULL );
	return NULL;
}

static void list_reset_frag_match( Frag_Match_Info *fmi_p, int direction )
{
	if( direction == CYC_FORWARD )
		fmi_p->fmi_u.li.curr_np = fmi_p->fmi_u.li.first_np;
	else
		fmi_p->fmi_u.li.curr_np = fmi_p->fmi_u.li.last_np;
}

static void rb_tree_reset_frag_match( Frag_Match_Info *fmi_p, int direction )
{
	if( direction == CYC_FORWARD )
		fmi_p->fmi_u.rbti.curr_n_p = fmi_p->fmi_u.rbti.first_n_p;
	else
		fmi_p->fmi_u.rbti.curr_n_p = fmi_p->fmi_u.rbti.last_n_p;
}

static void hash_tbl_reset_frag_match( Frag_Match_Info *fmi_p, int direction )
{
	assert("hash_tbl_reset_frag_match should never be called!?"==NULL);
}

static const char *list_advance_frag_match( Frag_Match_Info * fmi_p, int direction )
{
	Item *ip;

	assert( fmi_p != NULL );

	if( direction == CYC_FORWARD ){
		if( fmi_p->fmi_u.li.curr_np == fmi_p->fmi_u.li.last_np )
			return NULL;
		else {
			fmi_p->fmi_u.li.curr_np = NODE_NEXT(fmi_p->fmi_u.li.curr_np);
			assert( fmi_p->fmi_u.li.curr_np != NULL );
		}
	} else {
		if( fmi_p->fmi_u.li.curr_np == fmi_p->fmi_u.li.first_np )
			return NULL;
		else {
			fmi_p->fmi_u.li.curr_np = NODE_PREV( fmi_p->fmi_u.li.curr_np );
			assert( fmi_p->fmi_u.li.curr_np != NULL );
		}
	}
	ip = fmi_p->fmi_u.li.curr_np->n_data;
	return ip->item_name;
}

static const char *rb_tree_advance_frag_match( Frag_Match_Info * fmi_p, int direction )
{
	Item *ip;

	// there may be no items!?
	assert( fmi_p != NULL );

	if( direction == CYC_FORWARD ){
		if( fmi_p->fmi_u.rbti.curr_n_p == fmi_p->fmi_u.rbti.last_n_p )
			return NULL;
		else {
			fmi_p->fmi_u.rbti.curr_n_p = rb_successor_node( fmi_p->fmi_u.rbti.curr_n_p );
			assert( fmi_p->fmi_u.rbti.curr_n_p != NULL );
		}
	} else {
		if( fmi_p->fmi_u.rbti.curr_n_p == fmi_p->fmi_u.rbti.first_n_p )
			return NULL;
		else {
			fmi_p->fmi_u.rbti.curr_n_p = rb_predecessor_node( fmi_p->fmi_u.rbti.curr_n_p );
			assert( fmi_p->fmi_u.rbti.curr_n_p != NULL );
		}
	}
	ip = fmi_p->fmi_u.rbti.curr_n_p->data;
	return ip->item_name;
}

static const char *hash_tbl_advance_frag_match( Frag_Match_Info * fmi_p, int direction )
{
	assert( "hash_tbl_advance_frag_match should never be called" == NULL );
	return NULL;
}

#define INIT_CONTAINER_FUNCTIONS(stem)					\
	cnt_p->insert_item_func = stem##_insert_item;			\
	cnt_p->substring_find_func = stem##_substring_find;		\
	cnt_p->frag_item_func = stem##_frag_item;			\
	cnt_p->current_item_func = stem##_current_item;		\
	cnt_p->advance_func = stem##_advance_frag_match;		\
	cnt_p->reset_frag_match_func = stem##_reset_frag_match;		\
	cnt_p->new_enumerator_func = stem##_create_enumerator;		\


static void init_container(Container *cnt_p, int type)
{
	cnt_p->primary_type = type;
	cnt_p->name = NULL;

	switch(type){
		case LIST_CONTAINER:
			cnt_p->cnt_lp = NULL;
			INIT_CONTAINER_FUNCTIONS(list)
			break;
		case HASH_TBL_CONTAINER:
			cnt_p->cnt_htp = NULL;
			INIT_CONTAINER_FUNCTIONS(hash_tbl)
			break;
		case RB_TREE_CONTAINER:
			cnt_p->cnt_tree_p = NULL;
			INIT_CONTAINER_FUNCTIONS(rb_tree)
			break;
		default:
			assert("bad container type in init_container!?" == NULL);
			break;
	}
}

