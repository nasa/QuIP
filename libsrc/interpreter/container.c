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
static void init_container(Container *cnt_p,container_type_code type);

Container * create_container(const char *name, container_type_code type)
{
	Container *cnt_p;

	if( type == 0 ) type=HASH_TBL_CONTAINER;	// default
	cnt_p = new_container(type);
	cnt_p->name = savestr(name);
	return cnt_p;
}

void set_container_type(Container *cnt_p, int type)
{
	assert(type!=0);

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
	void *ptr;

	ptr = (*(cnt_p->cnt_typ_p->insert_item))(cnt_p,ip);

	if( ptr == NULL )
		return -1;
	return 0;
}

static int list_remove_name(Container *cnt_p, const char *name)
{
	Node *np;

	np = list_find_named_item(cnt_p->cnt_lp,name);
	if( np == NULL ) return -1;
	np = remNode(cnt_p->cnt_lp,np);
	assert( np != NULL );
	rls_node(np);
	return 0;
}

static int hash_tbl_remove_name(Container *cnt_p, const char *name)
{
	return remove_name_from_hash(name,cnt_p->cnt_htp);
}

static int rb_tree_remove_name(Container *cnt_p, const char *name)
{
	return rb_delete_key(cnt_p->cnt_tree_p, name );
}

int remove_name_from_container(QSP_ARG_DECL  Container *cnt_p, const char *name)
{
	return cnt_p->cnt_typ_p->remove_name(cnt_p,name);
}

static Item *list_find_match(Container *cnt_p, const char *name)
{
	Node *np;

	np = list_find_named_item(cnt_p->cnt_lp,name);
	if( np == NULL ) return NULL;
	return (Item *) NODE_DATA(np);
}

static Item *hash_tbl_find_match(Container *cnt_p, const char *name)
{
	return (Item*) fetch_hash(name,cnt_p->cnt_htp);
}

static Item *rb_tree_find_match(Container *cnt_p, const char *name)
{
	qrb_node *tnp;

	tnp = rb_find(cnt_p->cnt_tree_p,name);
	if( tnp == NULL )
		return NULL;
	return (Item *) tnp->data;
}

Item *container_find_match(Container *cnt_p, const char *name)
{
	return cnt_p->cnt_typ_p->find_match(cnt_p,name);
}

static void search_list_for_fragment(List *lp, Frag_Match_Info *fmi_p, const char *frag)
{
	int n;
	Node* np;

	lp = _alpha_sort(DEFAULT_QSP_ARG  lp);	// BUG should sort in-place???

	np = QLIST_HEAD(lp);

	n = (int) strlen(frag);
	fmi_p->fmi_u.li.curr_np = NULL;	// default
	fmi_p->fmi_u.li.first_np = NULL;
	fmi_p->fmi_u.li.last_np = NULL;

	while( np != NULL ){
		const Item *ip;
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

// Because the hash table items are not sorted, we need to keep a list of the items...

static void hash_tbl_substring_find(Frag_Match_Info *fmi_p,const char *frag)
{
	Container *cnt_p;
	Container *list_cnt_p;

	cnt_p = FMI_CONTAINER(fmi_p);
	list_cnt_p = ht_list_container(cnt_p->cnt_htp);
	// fmi_p still points to the hash_tbl container...
	search_list_for_fragment(list_cnt_p->cnt_lp,fmi_p,frag);
	zap_list(list_cnt_p->cnt_lp);	// what does "zap" mean?  BUG?
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
// Similarly, for an rb_tree, we should be even quicker, because we can find the start (like
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
	(*(cnt_p->cnt_typ_p->substring_find))(fmi_p,frag);
}

static Enumerator *list_advance_enum(Enumerator *ep)
{
	advance_list_enumerator(ep->e_p.lep);
	if( list_enumerator_item(ep->e_p.lep) == NULL )
		return NULL;
	else
		return ep;
}

static Enumerator *rb_tree_advance_enum(Enumerator *ep)
{
	advance_rb_tree_enumerator(ep->e_p.rbtep);
	if( rb_tree_enumerator_item(ep->e_p.rbtep) == NULL )
		return NULL;
	else
		return ep;
}

static Enumerator *hash_tbl_advance_enum(Enumerator *ep)
{
	advance_ht_enumerator(ep->e_p.htep);
	if( ht_enumerator_item(ep->e_p.htep) == NULL )
		return NULL;
	else
		return ep;
}

// add the items from this container to the given list

void cat_container_items(List *lp, Container *cnt_p)
{
	Node *np;
	Item *ip;
	Enumerator *ep, *orig;

	ep = (*(cnt_p->cnt_typ_p->new_enumerator))(cnt_p);

	if( ep == NULL ) return;	// enumerator is null if the container is empty

	orig = ep;

	while( ep != NULL ){
		ip = (*(ep->e_typ_p->current_enum_item))(ep);
		if( ip != NULL ){
			np = mk_node(ip);
			addTail(lp,np);
		}
		ep = ep->e_typ_p->advance_enum(ep);
	}
	orig->e_typ_p->release_enum(orig);
}

inline static long list_eltcount(Container *cnt_p)
{
	return eltcount(cnt_p->cnt_lp);
}

inline static long hash_tbl_eltcount(Container *cnt_p)
{
	return cnt_p->cnt_htp->ht_n_entries;
}

inline static long rb_tree_eltcount(Container *cnt_p)
{
	return rb_node_count(cnt_p->cnt_tree_p);
}

long container_eltcount(Container *cnt_p)
{
	return cnt_p->cnt_typ_p->eltcount(cnt_p);
}

List *container_list(Container *cnt_p)
{
	return cnt_p->cnt_typ_p->list_of_items(cnt_p);
}

static List *list_list_of_items(Container *cnt_p)
{
	return cnt_p->cnt_lp;
}

static List *hash_tbl_list_of_items(Container *cnt_p)
{
	return hash_tbl_list(cnt_p->cnt_htp);
}

static List *rb_tree_list_of_items(Container *cnt_p)
{
	return rb_tree_list(cnt_p->cnt_tree_p);
}

static void release_container(Container *cnt_p)
{
	rls_str((char *)cnt_p->name);
	givbuf(cnt_p);
}

static void list_delete(Container *cnt_p)
{
	zap_list(cnt_p->cnt_lp);
	release_container(cnt_p);
}

static void hash_tbl_delete(Container *cnt_p)
{
	zap_hash_tbl(cnt_p->cnt_htp);
	release_container(cnt_p);
}

static void rb_tree_delete(Container *cnt_p)
{
	release_rb_tree(cnt_p->cnt_tree_p);
	release_container(cnt_p);
}


static void print_container_info_header(QSP_ARG_DECL  Container *cnt_p)
{
	sprintf(MSG_STR,"Container %s:\n",
		cnt_p->name==NULL?"<null>":cnt_p->name);
	prt_msg(MSG_STR);
}

static void list_dump_info(QSP_ARG_DECL Container *cnt_p)
{
	print_container_info_header(QSP_ARG  cnt_p);
	sprintf(MSG_STR,"\tlist with %d elements\n",eltcount(cnt_p->cnt_lp));
	prt_msg(MSG_STR);
}

static void hash_tbl_dump_info(QSP_ARG_DECL Container *cnt_p)
{
	print_container_info_header(QSP_ARG  cnt_p);
	tell_hash_stats(QSP_ARG  cnt_p->cnt_htp );
}

static void rb_tree_dump_info(QSP_ARG_DECL Container *cnt_p)
{
	print_container_info_header(QSP_ARG  cnt_p);
	prt_msg("\tRed-black tree, sorry no stats...");
}

static void *list_current_enum_item(Enumerator *ep)
{
	//assert( ep->e_p.lep != NULL );
	// enumerator can be null if the list is empty!
	return list_enumerator_item(ep->e_p.lep);
}

static void *rb_tree_current_enum_item(Enumerator *ep)
{
	return ep->e_p.rbtep->node_p->data;
}

static void *hash_tbl_current_enum_item(Enumerator *ep)
{
	if( ep->e_p.htep->current_entry == NULL )
		return NULL;
	else
		return *(ep->e_p.htep->current_entry);
}

static void list_release_enum(Enumerator *ep)
{
	rls_list_enumerator(ep->e_p.lep);
	givbuf(ep);
}

static void rb_tree_release_enum(Enumerator *ep)
{
	rls_rb_tree_enumerator(ep->e_p.rbtep);
	givbuf(ep);
}

static void hash_tbl_release_enum(Enumerator *ep)
{
	rls_hash_tbl_enumerator(ep->e_p.htep);
	givbuf(ep);
}

#define INIT_ONE_ENUM_FUNCTION(func,stem)				\
	etp->func = stem##_##func;

#define INIT_ENUMERATOR_TYPE_FUNCTIONS(stem)				\
	INIT_ONE_ENUM_FUNCTION(advance_enum,stem)			\
	INIT_ONE_ENUM_FUNCTION(release_enum,stem)			\
	INIT_ONE_ENUM_FUNCTION(current_enum_item,stem)			\

static Enumerator_Type *list_enumerator_type(void)
{
	static Enumerator_Type *etp=NULL;

	if( etp != NULL ) return etp;

	etp = getbuf( sizeof(Enumerator_Type) );
	INIT_ENUMERATOR_TYPE_FUNCTIONS(list)
	return etp;
}

static Enumerator_Type *rb_tree_enumerator_type(void)
{
	static Enumerator_Type *etp=NULL;

	if( etp != NULL ) return etp;

	etp = getbuf( sizeof(Enumerator_Type) );
	INIT_ENUMERATOR_TYPE_FUNCTIONS(rb_tree)
	return etp;
}

static Enumerator_Type *hash_tbl_enumerator_type(void)
{
	static Enumerator_Type *etp=NULL;

	if( etp != NULL ) return etp;

	etp = getbuf( sizeof(Enumerator_Type) );
	INIT_ENUMERATOR_TYPE_FUNCTIONS(hash_tbl)
	return etp;
}

static Enumerator *new_enumerator(Container *cnt_p)
{
	Enumerator *ep;

	ep = getbuf( sizeof(Enumerator) );
	ep->e_cnt_p = cnt_p;	// is this used?
	return ep;
}

#define DECLARE_NEW_ENUM_FUNC(container_type,enumerator_type,member)		\
										\
static Enumerator * container_type##_new_enumerator(Container *cnt_p)		\
{										\
	Enumerator *ep;								\
	enumerator_type *vp;							\
										\
	vp = new_##container_type##_enumerator(cnt_p->member);			\
	if( vp == NULL ) return NULL;						\
										\
	ep = new_enumerator(cnt_p);						\
	ep->e_typ_p = container_type##_enumerator_type();			\
	ep->e_p.vp = vp;							\
	return ep;								\
}

DECLARE_NEW_ENUM_FUNC(list,List_Enumerator,cnt_lp)
DECLARE_NEW_ENUM_FUNC(hash_tbl,Hash_Tbl_Enumerator,cnt_htp)
DECLARE_NEW_ENUM_FUNC(rb_tree,RB_Tree_Enumerator,cnt_tree_p)

#ifdef FOOBAR
static Enumerator *list_new_enumerator(Container *cnt_p)
{
	Enumerator *ep;
	List_Enumerator *lep;

	lep = new_list_enumerator(cnt_p->cnt_lp);
	if( lep == NULL ) return NULL;

	ep = new_enumerator(cnt_p);
	ep->e_typ_p = list_enumerator_type();
	ep->e_p.lep = lep;
	return ep;
}

static Enumerator *hash_tbl_new_enumerator(Container *cnt_p)
{
	Enumerator *ep;
	Hash_Tbl_Enumerator *htep;

	htep = new_hash_tbl_enumerator(cnt_p->cnt_htp);
	if( htep == NULL ) return NULL;

	ep = new_enumerator(cnt_p);
	ep->e_typ_p = hash_tbl_enumerator_type();
	ep->e_p.vp = htep;
	return ep;
}

static Enumerator *rb_tree_new_enumerator(Container *cnt_p)
{
	Enumerator *ep;
	RB_Tree_Enumerator *rbtep;

	rbtep = new_rb_tree_enumerator(cnt_p->cnt_tree_p);
	if( rbtep == NULL ) return NULL;

	ep = new_enumerator(cnt_p);
	ep->e_typ_p = rb_tree_enumerator_type();
	ep->e_p.vp = rbtep;
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
	return (*(cnt_p->cnt_typ_p->frag_item))(fmi_p);
}

static const Item *list_current_frag_match_item(Frag_Match_Info *fmi_p)
{
	return fmi_p->fmi_u.li.curr_np->n_data;
}

static const Item *rb_tree_current_frag_match_item(Frag_Match_Info *fmi_p)
{
	return fmi_p->fmi_u.rbti.curr_n_p->data;
}

static const Item *hash_tbl_current_frag_match_item(Frag_Match_Info *fmi_p)
{
	assert( "hash_tbl_current_frag_match_item should never be called" == NULL );
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
	const Item *ip;

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

static void list_init_data(Container *cnt_p)
{
	cnt_p->cnt_lp = NULL;
}

static void hash_tbl_init_data(Container *cnt_p)
{
	cnt_p->cnt_htp = NULL;
}

static void rb_tree_init_data(Container *cnt_p)
{
	cnt_p->cnt_tree_p = NULL;
}

#define INIT_ONE_FUNCTION(func,stem)					\
	ctp->func = stem##_##func;

#define INIT_CONTAINER_TYPE_FUNCTIONS(stem)				\
	INIT_ONE_FUNCTION(insert_item,stem)				\
	INIT_ONE_FUNCTION(remove_name,stem)				\
	INIT_ONE_FUNCTION(find_match,stem)				\
	INIT_ONE_FUNCTION(eltcount,stem)				\
	INIT_ONE_FUNCTION(delete,stem)					\
	INIT_ONE_FUNCTION(init_data,stem)				\
	INIT_ONE_FUNCTION(list_of_items,stem)				\
	INIT_ONE_FUNCTION(dump_info,stem)				\
	INIT_ONE_FUNCTION(new_enumerator,stem)				\
									\
	/* frag_match methods */					\
	INIT_ONE_FUNCTION(substring_find,stem)				\
	INIT_ONE_FUNCTION(frag_item,stem)				\
	INIT_ONE_FUNCTION(current_frag_match_item,stem)			\
	INIT_ONE_FUNCTION(advance_frag_match,stem)			\
	INIT_ONE_FUNCTION(reset_frag_match,stem)			\


static Container_Type *list_container_type(void)
{
	static Container_Type *ctp=NULL;

	if( ctp != NULL ) return ctp;

	ctp = getbuf(sizeof(Container_Type));
	INIT_CONTAINER_TYPE_FUNCTIONS(list);
	return ctp;
}

static Container_Type *hash_tbl_container_type(void)
{
	static Container_Type *ctp=NULL;

	if( ctp != NULL ) return ctp;

	ctp = getbuf(sizeof(Container_Type));
	INIT_CONTAINER_TYPE_FUNCTIONS(hash_tbl);
	return ctp;
}

static Container_Type *rb_tree_container_type(void)
{
	static Container_Type *ctp=NULL;

	if( ctp != NULL ) return ctp;

	ctp = getbuf(sizeof(Container_Type));
	INIT_CONTAINER_TYPE_FUNCTIONS(rb_tree);
	return ctp;
}

static void init_container(Container *cnt_p, container_type_code type)
{
	cnt_p->name = NULL;

	switch(type){
		case LIST_CONTAINER:
			cnt_p->cnt_typ_p = list_container_type();
			break;
		case HASH_TBL_CONTAINER:
			cnt_p->cnt_typ_p = hash_tbl_container_type();
			break;
		case RB_TREE_CONTAINER:
			cnt_p->cnt_typ_p = rb_tree_container_type();
			break;
		default:
			assert("bad container type in init_container!?" == NULL);
			break;
	}
	(*(cnt_p->cnt_typ_p->init_data))(cnt_p);
}

