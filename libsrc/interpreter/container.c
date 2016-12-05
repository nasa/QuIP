#include "quip_config.h"
#include "quip_prot.h"
#include "container.h"
#include "getbuf.h"
#include <assert.h>
#include <string.h>

Container * create_container(const char *name,int type)
{
	Container *cnt_p;

//fprintf(stderr,"create_container %s %d BEGIN\n",name,type);
	if( type == 0 ) type=HASH_TBL_CONTAINER;	// default
	cnt_p = new_container(type);
	cnt_p->name = savestr(name);
	return cnt_p;
}

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

void set_container_type(Container *cnt_p, int type)
{
	if( type == 0 ) return;	// keep default

	if( ! (cnt_p->types & type) ){
		add_type_to_container(cnt_p,type);
		cnt_p->types |= type;
	}
	make_container_current(cnt_p,type);
	cnt_p->primary_type = type;
}

Container * new_container(int type)
{
	Container *cnt_p=NULL;

//fprintf(stderr,"new_container:  type = %d\n",type);
	assert( type == LIST_CONTAINER ||
		type == HASH_TBL_CONTAINER ||
		type == RB_TREE_CONTAINER );

	cnt_p = getbuf( sizeof(Container) );

	cnt_p->types = type;
	cnt_p->primary_type = type;
	cnt_p->is_current = type;
	cnt_p->name = NULL;

	// null all pointers by default
	cnt_p->cnt_lp = NULL;
	cnt_p->cnt_htp = NULL;
	cnt_p->cnt_tree_p = NULL;

//fprintf(stderr,"new_container %s at 0x%lx calling add_type_to_container %d\n",cnt_p->name,(long)cnt_p,type);
	add_type_to_container(cnt_p,type);
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
	cnt_p->is_current = cnt_p->primary_type;
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
	cnt_p->is_current = cnt_p->primary_type;
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
	fmi_p->u.li.curr_np = NULL;	// default
	fmi_p->u.li.first_np = NULL;
	fmi_p->u.li.last_np = NULL;

	while( np != NULL ){
		Item *ip;
		int compVal;

		ip = NODE_DATA(np);
		compVal = strncmp( frag, ITEM_NAME(ip), n );
		if( compVal == 0 ){
			// We have found the first node that is a match,
			// but we want also determine the last...
			fmi_p->u.li.curr_np = np;
			fmi_p->u.li.first_np = np;
			fmi_p->u.li.last_np = np;
			np = NODE_NEXT(np);
			while( np != NULL ){
				ip = NODE_DATA(np);
				compVal = strncmp( frag, ITEM_NAME(ip), n );
				if( compVal != 0 )
					return;
				fmi_p->u.li.last_np = np;
				np = NODE_NEXT(np);
			}
			return;
		}
		np = NODE_NEXT(np);
	}
}


//Item *container_find_substring_match(Container *cnt_p, const char *frag)
void container_find_substring_matches(Frag_Match_Info *fmi_p, Container *cnt_p, const char *frag)
{
//	Item *ip=NULL;
//	Enumerator *ep;
//	int n;

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
#ifdef FOOBAR
	switch(cnt_p->primary_type){
		case HASH_TBL_CONTAINER:
			// We have no way to find anything based on substrings,
			// so we simply enumerate...
			ep = new_enumerator(cnt_p,cnt_p->primary_type);
			n = strlen(frag);
			while( ep != NULL ){
				ip = enumerator_item(ep);
				assert(ip!=NULL);
				if( ! strncmp(ITEM_NAME(ip),frag,n) ){
					
					// found a match!
					// BUG - can we remember where we are?
					//return ip;
fprintf(stderr,"container_find_substring_matches, found something but wrong container type!?\n");
					return;
				}
				ep = advance_enumerator(ep);
			}
			break;
		default:
			NERROR1("container_find_substring_matches:  unexpected container type!?");
			break;
	}
#endif // FOOBAR
	NERROR1("container_find_substring_matches:  Unhandled container type!?");
}


//extern Item *container_find_substring_matches(QSP_ARG_DECL  Container *cnt_p, const char *frag);


Enumerator *advance_enumerator(Enumerator *ep)
{
	switch(ep->type){
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
	switch(ep->type){
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

	ep = new_enumerator(cnt_p,0);
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
	if( cnt_p->is_current & LIST_CONTAINER )
		return cnt_p->cnt_lp;

	if( cnt_p->cnt_lp == NO_LIST ){
//fprintf(stderr,"container_list:  creating new list for container %s...\n",cnt_p->name);
		cnt_p->cnt_lp = new_list();
		cnt_p->types |= LIST_CONTAINER;
	} else {
//fprintf(stderr,"container_list:  releasing old nodes, container %s...\n",cnt_p->name);
		rls_nodes_from_list( cnt_p->cnt_lp );
	}

//fprintf(stderr,"container_list:  building list, container %s...\n",cnt_p->name);
	cat_container_items(cnt_p->cnt_lp,cnt_p);
	cnt_p->is_current |= LIST_CONTAINER;
	return cnt_p->cnt_lp;
}

void delete_container(Container *cnt_p)
{
	if( cnt_p->types & LIST_CONTAINER ){
		Node *np;

		while( (np=remHead(cnt_p->cnt_lp)) != NO_NODE )
			rls_node(np);
		rls_list(cnt_p->cnt_lp);
	}
	if( cnt_p->types & HASH_TBL_CONTAINER ){
		zap_hash_tbl(cnt_p->cnt_htp);
	}
	if( cnt_p->types & RB_TREE_CONTAINER ){
		release_rb_tree(cnt_p->cnt_tree_p);
	}


	rls_str((char *)cnt_p->name);
	givbuf(cnt_p);

}

void dump_container_info(QSP_ARG_DECL  Container *cnt_p)
{
	sprintf(MSG_STR,"Container %s:\n",cnt_p->name==NULL?"<null>":cnt_p->name);
	prt_msg(MSG_STR);

	if( cnt_p->types & LIST_CONTAINER ){
		prt_msg("\tSorry, no linked list stats...");
	}
	if( cnt_p->types & HASH_TBL_CONTAINER ){
		tell_hash_stats(QSP_ARG  cnt_p->cnt_htp );
	}
	if( cnt_p->types & RB_TREE_CONTAINER ){
		prt_msg("\tSorry, no linked red-black tree stats...");
	}
}

void * enumerator_item(Enumerator *ep)
{
//fprintf(stderr,"enumerator_item BEGIN\n");
//fprintf(stderr,"enumerator_item ep = 0x%lx, type = %d\n",(long)ep,ep->type);
	switch(ep->type){
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

Enumerator *new_enumerator (Container *cnt_p, int type)
{
	void *vp;
	Enumerator *ep;

	if( type == 0 ) type = cnt_p->primary_type;

	switch( type ){
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

	if( type == 0 ) type = cnt_p->primary_type;

	if( ! (cnt_p->is_current & type) ){
		// now we have to transfer the items to a different container type!?
		//NERROR1("new_enumerator:  Sorry, container transfer not implemented yet...");
		if( ! (cnt_p->types & type) ){
//fprintf(stderr,"new_enumerator calling add_type_to_container %s, type code = %d\n",cnt_p->name,type);
			add_type_to_container(cnt_p,type);
		}
		make_container_current(cnt_p,type);
	}
//else
//fprintf(stderr,"new_enumerator: containter %s is current for type = %d\n",cnt_p->name,type);

	ep->type = type;
	ep->e_cnt_p = cnt_p;	// is this used?
	ep->e_p.vp = vp;

	return ep;
}



Item *current_frag_item( Frag_Match_Info *fmi_p )
{
	// what type of container is used?
	switch( fmi_p->icp->ic_itp->it_default_container_type ){
		case RB_TREE_CONTAINER:
			return fmi_p->u.rbti.curr_n_p->data;
			break;
		case LIST_CONTAINER:
		case HASH_TBL_CONTAINER:
			return fmi_p->u.li.curr_np->n_data;
			break;
		default:
			NERROR1("current_frag_item:  Bad container type!?");
			break;
	}
	return NULL;
}

