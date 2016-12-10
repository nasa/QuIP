#include "quip_config.h"
#include "quip_prot.h"
#include "container.h"
#include "getbuf.h"
#include <assert.h>
#include <string.h>

#define HAS_VALID_CONTAINER_TYPE(cnt_p)	VALID_CONTAINER_TYPE( CONTAINER_TYPE(cnt_p) )

#define VALID_CONTAINER_TYPE(t)	( t == LIST_CONTAINER || t == HASH_TBL_CONTAINER || t == RB_TREE_CONTAINER )

Container * create_container(const char *name,int type)
{
	Container *cnt_p;

//fprintf(stderr,"create_container %s %d BEGIN\n",name,type);
	if( type == 0 ) type=HASH_TBL_CONTAINER;	// default
	cnt_p = new_container(type);
	cnt_p->name = savestr(name);
	return cnt_p;
}

#ifdef FOOBAR
static void add_type_to_container(Container *cnt_p, int type)
{
//fprintf(stderr,"add_type_to_container %s %d (primary type %d)\n",cnt_p->name,type,cnt_p->primary_type);
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
//fprintf(stderr,"make_container_current:  NOT releasing old nodes from list!?\n");
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
//fprintf(stderr,"set_container_type %s %d\n",cnt_p->name,type);
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

Container * new_container(int type)
{
	Container *cnt_p=NULL;

//fprintf(stderr,"new_container:  type = %d\n",type);
	assert( VALID_CONTAINER_TYPE(type) );

	cnt_p = getbuf( sizeof(Container) );

//	cnt_p->types = type;
	cnt_p->primary_type = type;
//	cnt_p->is_current = type;
	cnt_p->name = NULL;

	// null all pointers by default
	cnt_p->cnt_lp = NULL;
	cnt_p->cnt_htp = NULL;
	cnt_p->cnt_tree_p = NULL;

//fprintf(stderr,"new_container %s at 0x%lx calling add_type_to_container %d\n",cnt_p->name,(long)cnt_p,type);
	//add_type_to_container(cnt_p,type);
	set_container_type(cnt_p,type);
	return cnt_p;
}

int add_to_container(Container *cnt_p, Item *ip)
{
	Node *np;
	qrb_node *tnp;
	int stat=0;

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
//	cnt_p->is_current = cnt_p->primary_type;
//fprintf(stderr,"add_to_container:  types = %d, primary_type = %d, is_current = %d\n", cnt_p->types,cnt_p->primary_type,cnt_p->is_current);
	return stat;
}

int remove_name_from_container(QSP_ARG_DECL  Container *cnt_p, const char *name)
{
	Node *np;
	int stat=0;

	switch(cnt_p->primary_type){
		case LIST_CONTAINER:
			np = list_find_named_item(cnt_p->cnt_lp,name);
//fprintf(stderr,"remove_name_from_container calling remNode, lp = 0x%lx\n",(long)cnt_p->cnt_lp);
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
			NERROR1("container_find_match:  invalid container type!?");
			break;
	}
	return ip;
}

// assume the items in the list are sorted...
static void list_substring_find(Frag_Match_Info *fmi_p, List *lp, const char *frag )
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

// Because the hash table items are not sorted, we need to keep a list of the items...

static void hash_tbl_substring_find(Frag_Match_Info *fmi_p,Hash_Tbl *htp,const char *frag)
{
	List *lp;

	//lp = hash_tbl_list(htp);
	lp = ht_list(htp);
	list_substring_find(fmi_p,lp,frag);
	zap_list(lp);
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

void container_find_substring_matches(Frag_Match_Info *fmi_p, Container *cnt_p, const char *frag)
{
	switch(cnt_p->primary_type){
		case LIST_CONTAINER:
			list_substring_find(fmi_p,cnt_p->cnt_lp,frag);
			break;
		case HASH_TBL_CONTAINER:
			//make_container_current(cnt_p,LIST_CONTAINER);
			//list_substring_find(fmi_p,cnt_p->cnt_lp,frag);
			// NEED TO ALPHA-SORT !!!
			hash_tbl_substring_find(fmi_p,cnt_p->cnt_htp,frag);
			break;
		case RB_TREE_CONTAINER:
			rb_substring_find( fmi_p, cnt_p->cnt_tree_p, frag );
			break;
		default:
			NERROR1("container_find_substring_matches:  bad container type!?");
			break;
	}

#ifdef FOOBAR
	if( cnt_p->types & RB_TREE_CONTAINER ){
		if( ! (cnt_p->is_current&RB_TREE_CONTAINER) ){
			make_container_current(cnt_p,RB_TREE_CONTAINER);
		}
//fprintf(stderr,"container_find_substring_matches:  calling rb_substring_find, frag = %s\n",frag);
		rb_substring_find(fmi_p,cnt_p->cnt_tree_p,frag);
//fprintf(stderr,"container_find_substring_matches:  after rb_substring_find, first = 0x%lx, last = 0x%lx\n", (long)fmi_p->first_n_p,(long)fmi_p->last_n_p);
		return;
	}
//fprintf(stderr,"container_find_substring_matches:  types = %d, primary = %d, is_current = %d\n",
//cnt_p->types,cnt_p->primary_type,cnt_p->is_current);
	if( cnt_p->primary_type == HASH_TBL_CONTAINER ){
//fprintf(stderr,"calling make_container_current\n");
		make_container_current(cnt_p,LIST_CONTAINER);
		// NEED TO ALPHA-SORT !!!
	}
//fprintf(stderr,"container_find_substring_matches:  types = %d, primary = %d, is_current = %d\n",
//cnt_p->types,cnt_p->primary_type,cnt_p->is_current);
	if( cnt_p->types & LIST_CONTAINER ){
		if( ! (cnt_p->is_current&LIST_CONTAINER) ){
			make_container_current(cnt_p,LIST_CONTAINER);
		}
		list_substring_find(fmi_p,cnt_p->cnt_lp,frag);
		return;
	}
	NERROR1("container_find_substring_matches:  Unhandled container type!?");
#endif // FOOBAR

}


//extern Item *container_find_substring_matches(QSP_ARG_DECL  Container *cnt_p, const char *frag);


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

	ep = new_enumerator(cnt_p);
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
//fprintf(stderr,"enumerator_item BEGIN\n");
	switch( ENUMERATOR_TYPE(ep) ){
		case LIST_CONTAINER:
			return NODE_DATA(ep->e_p.lep->np);
			break;
		case HASH_TBL_CONTAINER:
			if( ep->e_p.htep->current_entry == NULL )
				return NULL;
			else {
//fprintf(stderr,"enumerator_item returning current hash table entry, 0x%lx at 0x%lx\n",(long)*(ep->e_p.htep->current_entry), (long)ep->e_p.htep->current_entry);
//fprintf(stderr,"entries at 0x%lx, n = %ld\n",(long)ep->e_p.htep->htp->ht_entries,ep->e_p.htep->htp->ht_n_entries);
//fprintf(stderr,"first invalid entry at 0x%lx\n",(long)(ep->e_p.htep->htp->ht_entries+ep->e_p.htep->htp->ht_n_entries));
				return *(ep->e_p.htep->current_entry);
			}
			break;
		case RB_TREE_CONTAINER:
//fprintf(stderr,"enumerator_item:  returning item from tree node at 0x%lx\n",(long)ep->e_p.rbtep->node_p);
			return ep->e_p.rbtep->node_p->data;
			break;
		default:
			NERROR1("enumerated_data:  bad container type!?");
			return NULL;
			break;
	}
}

Enumerator *new_enumerator (Container *cnt_p )
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
//fprintf(stderr,"back from new_hash_tbl_enumerator\n");
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
//fprintf(stderr,"new_enumerator %s at 0x%lx BEGIN\n",cnt_p->name,(long)cnt_p);

	ep = getbuf( sizeof(Enumerator) );

	ep->e_cnt_p = cnt_p;	// is this used?
	ep->e_p.vp = vp;

	return ep;
}



Item *current_frag_item( Frag_Match_Info *fmi_p )
{
	// what type of container is used?
	switch( IT_CONTAINER_TYPE(CTX_IT(FMI_CTX(fmi_p))) ){
		case RB_TREE_CONTAINER:
			return fmi_p->fmi_u.rbti.curr_n_p->data;
			break;
		case LIST_CONTAINER:
		case HASH_TBL_CONTAINER:
			return fmi_p->fmi_u.li.curr_np->n_data;
			break;
		default:
fprintf(stderr,"current_frag_item:  context %s at 0x%lx\n", CTX_NAME(FMI_CTX(fmi_p)), (long) FMI_CTX(fmi_p) );
fprintf(stderr,"current_frag_item:  item_type %s at 0x%lx\n", ITEM_TYPE_NAME(CTX_IT(FMI_CTX(fmi_p))), (long) CTX_IT(FMI_CTX(fmi_p)) );
fprintf(stderr,"current_frag_item:  container_type = %d\n", IT_CONTAINER_TYPE(CTX_IT(FMI_CTX(fmi_p))) );
			NERROR1("current_frag_item:  Bad container type!?");
			break;
	}
	return NULL;
}

