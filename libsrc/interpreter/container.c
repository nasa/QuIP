#include "quip_config.h"
#include "quip_prot.h"
#include "container.h"
#include <assert.h>

Container * create_container(const char *name)
{
	Container *cnt_p;

	cnt_p = new_container(HASH_TBL_CONTAINER);	// the default
	cnt_p->name = savestr(name);
	return cnt_p;
}

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
				// BUG first release the nodes from the list!?
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
}

void set_container_type(Container *cnt_p, int type)
{
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

//	assert( type >= 0 && type < N_CONTAINER_TYPES );
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
	rb_node *tnp;
	int stat=0;

//fprintf(stderr,"add_to_container:  adding %s to %s\n",ITEM_NAME(ip),cnt_p->name);
	switch(cnt_p->primary_type){
		case LIST_CONTAINER:
			np = mk_node(ip);
			// add to head so that temp items can be found
			// easily for later deletion?
			//addHead( cnt_p->cnt_lp, np );
			addTail( cnt_p->cnt_lp, np );
			break;
		case HASH_TBL_CONTAINER:
//fprintf(stderr,"add_to_container:  calling insert_hash\n");
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
	rb_node *tnp;

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

//extern Item *container_find_substring_match(QSP_ARG_DECL  Container *cnt_p, const char *frag);


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

// add the items from this container to the given list

void cat_container_items(List *lp, Container *cnt_p)
{
	Node *np;
	Item *ip;
	Enumerator *ep;

	ep = new_enumerator(cnt_p,0);
	assert(ep!=NULL);

	while( ep != NULL ){
		ip = enumerator_item(ep);
//fprintf(stderr,"cat_container_items:  ip = 0x%lx\n",(long)ip);
		if( ip != NULL ){
			np = mk_node(ip);
			addTail(lp,np);
		}
		ep = advance_enumerator(ep);
	}
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
	switch(ep->type){
		case LIST_CONTAINER:
			return NODE_DATA(ep->e_p.lep->np);
			break;
		case HASH_TBL_CONTAINER:
			if( ep->e_p.htep->current_entry == NULL )
				return NULL;
			else {
/*
fprintf(stderr,"enumerator_item returning current hash table entry, 0x%lx at 0x%lx\n",(long)*(ep->e_p.htep->current_entry),
(long)ep->e_p.htep->current_entry);
fprintf(stderr,"entries at 0x%lx, n = %ld\n",(long)ep->e_p.htep->htp->ht_entries,ep->e_p.htep->htp->ht_n_entries);
fprintf(stderr,"first invalid entry at 0x%lx\n",(long)(ep->e_p.htep->htp->ht_entries+ep->e_p.htep->htp->ht_n_entries));
*/
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

Enumerator *new_enumerator (Container *cnt_p, int type)
{
	Enumerator *ep;

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
	switch( type ){
		case LIST_CONTAINER:
			ep->e_p.lep = new_list_enumerator(cnt_p->cnt_lp);
			break;
		case HASH_TBL_CONTAINER:
			ep->e_p.htep = new_hash_tbl_enumerator(cnt_p->cnt_htp);
//fprintf(stderr,"back from new_hash_tbl_enumerator\n");
			break;
		case RB_TREE_CONTAINER:
			ep->e_p.rbtep = new_rbtree_enumerator(cnt_p->cnt_tree_p);
			break;
		default:
			// give memory back here, if not erroring out...
			NERROR1("new_enumerator:  bad container type!?");
			return NULL;
			break;
	}
	return ep;
}

