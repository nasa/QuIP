#include <stdio.h>

#include "quip_config.h"


#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* isatty() */
#endif

#ifdef THREAD_SAFE_QUERY
#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif /* HAVE_ERRNO_H */
#endif /* THREAD_SAFE_QUERY */

/* This was not defined on old SunOS */
#ifndef tolower
#define tolower(c)		(( c )+'a'-'A')
#endif


#include "quip_prot.h"
#include "query_prot.h"
#include "item_prot.h"
#include "container.h"
#include "stack.h"
#include "debug.h"

#ifdef QUIP_DEBUG
static u_long item_debug=ITEM_DEBUG_MASK;
static u_long debug_contexts=CTX_DEBUG_MASK;
#endif /* QUIP_DEBUG */

#ifdef HAVE_HISTORY
#include "history.h"	/* add_phist() */
#else /* ! HAVE_HISTORY */
#include "query_bits.h"	/* qldebug */
#endif /* HAVE_HISTORY */

/* local prototypes */

// Make needy also did things to classes - need to do something about that???
// Classes need a serial number field too?
//static void make_needy(QSP_ARG_DECL  Item_Context *icp);
static void _init_itp(QSP_ARG_DECL  Item_Type *itp, int container_type);
static int item_cmp(const void *,const void *);
static Item_Type * init_item_type(QSP_ARG_DECL  const char *name, int container_type);
static void _store_item(QSP_ARG_DECL  Item_Context *icp, Item *ip);

static ITEM_INIT_PROT(Item_Context,ctx)
//static ITEM_GET_PROT(Item_Context,ctx)
static ITEM_NEW_PROT(Item_Context,ctx)
static ITEM_DEL_PROT(Item_Context,ctx)

#define init_ctxs()	_init_ctxs(SINGLE_QSP_ARG)
#define new_ctx(name)	_new_ctx(QSP_ARG  name)
#define del_ctx(name)	_del_ctx(QSP_ARG  name)

static Item_Type *	_new_ittyp(QSP_ARG_DECL  const char *);

//#define IT_LIST(itp)	_item_list(QSP_ARG  itp)

#define ITEM_TYPE_STRING	"Item_Type"
Item_Type * ittyp_itp=NULL;

/*static*/ ITEM_INIT_FUNC(Item_Type,ittyp,0)
static ITEM_NEW_FUNC(Item_Type,ittyp)

ITEM_LIST_FUNC(Item_Type,ittyp)
// move to questions.c
//ITEM_PICK_FUNC(Item_Type,ittyp)

static Item_Type *ctx_itp=NULL;

#define CTX_IT_NAME	"Context"
#define DEF_CTX_NAME	"default"

static ITEM_INIT_FUNC(Item_Context,ctx,0)
ITEM_CHECK_FUNC(Item_Context,ctx)
static ITEM_NEW_FUNC(Item_Context,ctx)
static ITEM_DEL_FUNC(Item_Context,ctx)

#define CHECK_ITEM_INDEX( itp )	if( ( itp ) == NULL ){		\
					warn("Null item type");		\
					return;				\
				}


/* a global */
u_long total_from_malloc = 0;


/* If we don't use an ansii style declaration,
 * we get warnings on the pc (microsoft compiler)
 */

static void no_del_method(QSP_ARG_DECL  Item *ip)
{
	warn("Undefined item deletion method");
	sprintf(ERROR_STRING,"Can't delete item '%s'",ITEM_NAME(ip));
	advise(ERROR_STRING);
}

void _set_del_method(QSP_ARG_DECL  Item_Type *itp,void (*func)(QSP_ARG_DECL  Item *))
{
	assert( IT_DEL_METHOD(itp) == no_del_method );

	SET_IT_DEL_METHOD(itp, func);
}

#define init_itci(itp,idx) _init_itci(QSP_ARG  itp,idx)

static void _init_itci(QSP_ARG_DECL  Item_Type *itp,int idx)
{
	Item_Type_Context_Info *itci_p;

	itci_p = &(itp->it_itci[idx]);

	SET_ITCI_ITEMS_LIST(itci_p, new_list() );
	SET_ITCI_CSTK(itci_p,new_stack());
	SET_ITCI_CTX(itci_p,NULL);
	// This appear to clear the restriction bit...
	SET_ITCI_FLAGS(itci_p,0);
	SET_ITCI_MATCH_CYCLE(itci_p,NULL);

	SET_ITCI_ITEMS_SERIAL(itci_p,0);
	SET_ITCI_STACK_SERIAL(itci_p,0);
	SET_ITCI_LIST_ITEMS_SERIAL(itci_p,0);
	SET_ITCI_LIST_STACK_SERIAL(itci_p,0);
}

static void _init_itp(QSP_ARG_DECL  Item_Type *itp, int container_type)
{
	int i;

	// first zero everything except the name (which is first)
	memset(((char *)itp)+sizeof(char *),0,sizeof(Item_Type)-sizeof(char *));

	/* should we really do this? */
	for(i=0;i<MAX_QUERY_STACKS;i++){
		init_itci(itp,i);
	}


	// We used to only initialize the first one.
	// What we do here is wasteful, but at least we don't
	// have to constantly be checking!
	/*
	init_itci(itp,0);
	memset(&(itp->it_itci[1]),0,(MAX_QUERY_STACKS-1)*sizeof(Item_Type_Context_Info));
	*/

	SET_IT_FREE_LIST(itp, new_list() );

	SET_IT_CLASS_LIST(itp, NULL);	// this was commented out - why?
	SET_IT_DEL_METHOD(itp, no_del_method);
	SET_IT_CONTAINER_TYPE(itp,container_type);	// _init_itp

	//SET_IT_FRAG_MATCH_INFO(itp,NULL);

	SET_IT_MATCH_CYCLE(itp,NULL);

#ifdef THREAD_SAFE_QUERY
#ifdef HAVE_PTHREADS
	{
		//itp->it_mutex = PTHREAD_MUTEX_INITIALIZER;
		int status;

		status = pthread_mutex_init(&itp->it_mutex,NULL);
		if( status != 0 )
			NERROR1("error initializing mutex");
	}
#endif /* HAVE_PTHREADS */
#endif /* ! THREAD_SAFE_QUERY */
} // _init_itp

#ifdef THREAD_SAFE_QUERY

#ifdef NOT_USED

/* In the multi-thread environment, each item type has a separate context stack
 * for each query stack.  This is done so that each thread can have different
 * variables, for example...
 * When we create a new query stack, we have to initialize
 * the corresponding context stack.
 * We call this when we have added a new query stream; But how do we know???
 * There are many instances where we WANT the threads to share contexts...
 * particularly data objects.  Maybe the best strategy is to have new threads
 * inherit the context of their invoking thread (usually the main thread)?
 */

static void setup_item_type_context(QSP_ARG_DECL  Item_Type *itp, Query_Stack *new_qsp)
{
	Item_Context *icp;

	icp = (Item_Context *) BOTTOM_OF_STACK(IT_CSTK_AT_IDX(itp,0));
	push_item_context(new_qsp,itp,icp);
}

// Why not???  BUG?
void setup_all_item_type_contexts(QSP_ARG_DECL  void *new_qsp)
{
	/* Push the default context onto ALL item types */
	List *lp;
	Item_Type *itp;
	Node *np;

	lp = item_list(QSP_ARG  ittyp_itp);
	np=QLIST_HEAD(lp);
	while( np != NULL ){
		itp = (Item_Type *)NODE_DATA(np);
		setup_item_type_context(QSP_ARG  itp,(Query_Stack *)new_qsp);
		np=NODE_NEXT(np);
	}
}
#endif /* NOT_USED */


#endif /* THREAD_SAFE_QUERY */

static Item_Type * init_item_type(QSP_ARG_DECL  const char *name, int container_type)
{
	static int is_first=1;	// how many times is this auto-initialized?
	Item_Type *itp;
	Item_Context *icp;

//sprintf(ERROR_STRING,"init_item_type %s, is_first = %d",name,is_first);
//NADVISE(ERROR_STRING);
	if( is_first ){		// is this the first item type?  (the item type items...)
		static Item_Context first_item_context;
		/* Item_Context * */ icp=(&first_item_context);
		//Node *np;	// unused?
#ifdef THREAD_SAFE_QUERY
		/* a real hack... */
//		Query_Stack dummy_qs;
#endif /* THREAD_SAFE_QUERY */

		// The first call is for Query_Stack.
		// Item_Type is initialized below to prevent recursion. 
		static Item_Type first_item_type;

//sprintf(ERROR_STRING,"init_item_type:  first_item_type is %s",name);
//advise(ERROR_STRING);

		/* Item_Type's are themselves items...
		 * but we can't call new_item_type to create
		 * the item_type for item_types!
		 */

		/* we don't call _new_ittyp() to avoid a recursion problem */
		ittyp_itp = &first_item_type;
		SET_IT_NAME(ittyp_itp, ITEM_TYPE_STRING );
		is_first=0;

		_init_itp(QSP_ARG  ittyp_itp,/*DEFAULT_CONTAINER_TYPE*/ RB_TREE_CONTAINER );

		/* We need to create the first context, but we don't want
		 * infinite recursion...
		 */

		/* BUG make sure DEF_CTX_NAME matches what is here... */
		SET_CTX_NAME( icp, savestr("Item_Type.default") );
		// change Dictionary to Container here?
		//SET_CTX_DICT( icp,  create_dictionary("Item_Type.default") );
		SET_CTX_CONTAINER( icp,
			create_container("Item_Type.default",IT_CONTAINER_TYPE(ittyp_itp)) );

		SET_CTX_IT( icp, ittyp_itp );
		/*np =*/ mk_node(icp);

		push_item_context(ittyp_itp, icp);

		//addHead(CTX_LIST(FIRST_CONTEXT(ittyp_itp)),np);

#ifdef THREAD_SAFE_QUERY
		/* a real hack... */
		/* What is the purpose of this??? */
//		qsp=&dummy_qs;
//		SET_QS_SERIAL(qsp,0);
#endif /* THREAD_SAFE_QUERY */

		/* why do this? Do we really need to? */
		add_item(ittyp_itp,ittyp_itp);
	}

	assert( strcmp(name,ITEM_TYPE_STRING) );

	itp = _new_ittyp(QSP_ARG  name);
	_init_itp(QSP_ARG  itp,container_type);

	icp = create_item_context(itp,DEF_CTX_NAME);
	push_item_context(itp,icp);

	return(itp);
} // init_item_type

/*
 * Create a new item type.  Allocate a hash table with hashsize
 * entries.  Return the item type index or -1 on failure.
 */

Item_Type * _new_item_type(QSP_ARG_DECL  const char *atypename, int container_type)
{
	Item_Type * itp;

	if( ittyp_itp != NULL ){
		Item *ip;
		ip = item_of(ittyp_itp,atypename);
		if( ip != NULL ){
			sprintf(ERROR_STRING,
			"Item type name \"%s\" is already in use\n",atypename);
			warn(ERROR_STRING);
			return(NULL);
		}
	}
	/* else we are initializing the item type Item_Type */

	if( container_type == 0 )
		container_type = DEFAULT_CONTAINER_TYPE;

	itp=init_item_type(QSP_ARG  atypename, container_type);
	assert( itp != NULL );

	if( ittyp_itp==NULL ){
		ittyp_itp = itp;
	}

	return(itp);
}

/*
 * Put an item into the corresponding name space
 */

static void _store_item( QSP_ARG_DECL  Item_Context *icp, Item *ip )
{
	if( add_to_container(CTX_CONTAINER(icp),ip) < 0){
		sprintf(ERROR_STRING,
			"Error storing name %s",ITEM_NAME(ip));
		NERROR1(ERROR_STRING);
	}
	INC_CTX_ITEM_SERIAL(icp);
}

/* This routine was eliminated, but we have reinstated it, so that
 * command tables can be preallocated items.
 */

int _add_item( QSP_ARG_DECL  Item_Type *itp, void *ip )
{
	assert( itp != NULL );
	assert( ip != NULL );

	_store_item(QSP_ARG  current_context(itp), (Item*) ip);
	INC_ITEMS_CHANGE_COUNT(itp);

	return 0;
}

static int insure_item_name_available(QSP_ARG_DECL  Item_Type *itp, const char *name)
{
	const Item *ip;

	/* We will allow name conflicts if they are not in the same context */

	/* Only check for conflicts in the current context */
	//ip = fetch_name(name,CTX_DICT(current_context(itp)));

	// When we start a new thread, the current context may be null!?

	ip = container_find_match(CTX_CONTAINER(current_context(itp)), name );

	if( ip != NULL ){
		sprintf(ERROR_STRING,
	"%s name \"%s\" is already in use in context %s",
			IT_NAME(itp),name,CTX_NAME(current_context(itp)));
		warn(ERROR_STRING);
		return -1;
	}
	return 0;
}

static inline int get_n_per_page(size_t size)
{
	int n_per_page;

#define FOUR_K	4096

	n_per_page = FOUR_K / size;	/* BUG use PAGESIZE */

	if( n_per_page <= 0 ){
		/* cast size to u_long because size_t is u_long on IA64 and
		 * u_int on IA32!?
		 */
		// If the item is bigger than a page, just get one
		n_per_page=1;
	}
	return n_per_page;
}

#define alloc_more_items(itp,size) _alloc_more_items(QSP_ARG  itp,size)

static void _alloc_more_items(QSP_ARG_DECL  Item_Type *itp, size_t size)
{
	int n_per_page;
	char *nip;

	n_per_page = get_n_per_page(size);	// make a field of itp?
#ifdef QUIP_DEBUG
if( debug & item_debug ){
sprintf(DEFAULT_ERROR_STRING,"malloc'ing %d more %s items",n_per_page,IT_NAME(itp));
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	/* get a pages worth of items */
	nip = (char*)  malloc( n_per_page * size );
	total_from_malloc += n_per_page*size;

	if( nip == NULL ){
		sprintf(DEFAULT_ERROR_STRING,
	"alloc_more_items:  out of memory while getting a new page of %s's",
			IT_NAME(itp));
		NERROR1(DEFAULT_ERROR_STRING);
	}

	while(n_per_page--){
		Node *np;

		np = mk_node(nip);
		addTail(IT_FREE_LIST(itp),np);
		nip += size;
	}
}
/*
 * Return a ptr to a new item.  Reuse a previously freed item
 * if one exists, otherwise allocate some new memory.
 * NOTE:  this memory is not obtained with getbuf, so it can't be
 * freed with givbuf!!
 *
 * OK, I'll bite - why is it allocated with malloc instead of getbuf?
 * Remember that getbuf was originally introduced because of distrust of
 * malloc on a very old system, but now it is still useful occasionally
 * for tracking down memory leaks...
 */

Item *_new_item( QSP_ARG_DECL  Item_Type *itp, const char* name, size_t size )
{
	Item *ip;
	Node *np;

	//assert( *name != 0 );
	assert( name != NULL );
	if( *name == 0 ) return NULL;

	LOCK_ITEM_TYPE(itp)

	if( insure_item_name_available(QSP_ARG  itp, name) < 0 ){
		UNLOCK_ITEM_TYPE(itp);
		return NULL;
	}

	// Try to get a structure from the free list
	// If the free list is empty, then allocate a page's worth

	if( QLIST_HEAD(IT_FREE_LIST(itp)) == NULL ){
		alloc_more_items(itp,size);
	}

	np = remHead(IT_FREE_LIST(itp));
	assert(np!=NULL);
	ip = (Item *) NODE_DATA(np);
	assert(ip!=NULL);
	rls_node(np);

	SET_ITEM_NAME( ip, savestr(name) );
	SET_ITEM_CTX( ip, current_context(itp) );

#ifdef BUILD_FOR_OBJC
	SET_ITEM_MAGIC(ip,QUIP_ITEM_MAGIC);
#endif /* BUILD_FOR_OBJC */

	/* BUG? should we worry about nodes here? */

	add_item(itp,ip);

	UNLOCK_ITEM_TYPE(itp)

	return(ip);
} // end new_item

int _remove_from_item_free_list(QSP_ARG_DECL  Item_Type *itp, void *ip)
{
	Node *np;

	LOCK_ITEM_TYPE(itp)
	np = remData(IT_FREE_LIST(itp),ip);
	UNLOCK_ITEM_TYPE(itp)

	if( np == NULL ) return -1;
	return 0;
}

/* Create a new context with the given name.
 * It needs to be push'ed in order to make it be
 * the current context for new item creation.
 */

Item_Context * _create_item_context( QSP_ARG_DECL  Item_Type *itp, const char* name )
{
	Item_Context *icp;
	char cname[LLEN];

	/* maybe we should have contexts for contexts!? */

	sprintf(cname,"%s.%s",IT_NAME(itp),name);

	// special case for the default context of the context item type
	// 
	if( (!strcmp(IT_NAME(itp),CTX_IT_NAME)) && !strcmp(name,DEF_CTX_NAME) ){
		static Item_Context first_context;

		/* can't use new_ctx()
		 * because ctx_itp isn't up and running yet.
		 */
		icp = &first_context;
		SET_CTX_NAME( icp, savestr(cname) );
		SET_CTX_IT( icp, itp );
		// See the comment below regarding create_dictionary...
		// Because the number of item_types and contexts is
		// not too large (e.g. dozens), we probably don't need
		// to be TOO concerned with efficiency here...
//		SET_CTX_DICT(icp , create_dictionary(CTX_NAME(icp)) );
		SET_CTX_CONTAINER(icp , create_container(CTX_NAME(icp), IT_CONTAINER_TYPE(itp)) );
		//SET_CTX_FLAGS(icp,0);
		SET_CTX_ITEM_SERIAL(icp,0);		// create_item_context
		SET_CTX_LIST_SERIAL(icp,0);		// create_item_context
		SET_CTX_ITEM_LIST(icp,NULL);
		SET_CTX_FRAG_ICP(icp,NULL);
		/* BUG?  not in the context database?? */
		return(icp);
	}

	/* Create an item type for contexts.
	 *
	 * Because new_item_type() calls create_item_text for the default
	 * context, we have the special case above...
	 */

	if( ctx_itp == NULL )
		ctx_itp = new_item_type(CTX_IT_NAME, DEFAULT_CONTAINER_TYPE);

#ifdef QUIP_DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  creating context %s",
WHENCE(create_item_context),cname);
advise(ERROR_STRING);
}
#endif	/* QUIP_DEBUG */

	icp = new_ctx(cname);

	/* If it_context_itp was null, make sure it has the right value now */
	// This is the item type for ALL the contexts of all item types, not just
	// for this item type...  The context names typically have the item name
	// and a single dot as a prefix...
//	SET_IT_CTX_IT(itp,ctx_itp);

	if( icp == NULL ){
		return(icp);
	}

	// Now initialize the fields of the new context
	SET_CTX_IT( icp, itp );
	// Here we might like to have a flag control
	// which type of dictionary we use?
	// We have been using hash tables all these
	// years, but a tree would be better for item types
	// with many, many items, to speed partial name
	// matching!
	SET_CTX_CONTAINER(icp , create_container(CTX_NAME(icp),IT_CONTAINER_TYPE(itp)) );
	SET_CTX_FRAG_ICP(icp,NULL);
	SET_CTX_ITEM_SERIAL(icp,0);		// create_item_context
	SET_CTX_LIST_SERIAL(icp,0);		// create_item_context
	SET_CTX_ITEM_LIST(icp,NULL);

	assert( CTX_CONTAINER(icp) != NULL );

	return(icp);
} // create_item_context


/* push an existing context onto the top of stack for this item class */

void _push_item_context( QSP_ARG_DECL   Item_Type *itp, Item_Context *icp )
{
	Node *np;

	/* might be a good idea to check here that the context item type
	 * matches itp.
	 */

	assert( icp != NULL );

#ifdef QUIP_DEBUG
if( debug & debug_contexts ){
sprintf(ERROR_STRING,"push_item_context:  pushing %s context %s",IT_NAME(itp),CTX_NAME(icp));
NADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	np=mk_node(icp);

	assert(LIST_OF_CONTEXTS(itp)!=NULL);

	addHead(LIST_OF_CONTEXTS(itp),np);

	SET_CURRENT_CONTEXT(itp, icp);
	INC_STACK_CHANGE_COUNT(itp);

	// mark the list as dirty?  item_list

} // end push_item_context

/*
 * Remove the top-of-stack context, but do not destroy.
 */

Item_Context * _pop_item_context( QSP_ARG_DECL  Item_Type *itp )
{
	Node *np;
	Item_Context *new_icp, *popped_icp;

	/* don't remove the context from the list yet, it needs
	 * to be there to find the objects in the context...
	 */
	np=remHead(LIST_OF_CONTEXTS(itp));
	if( np==NULL ){
		sprintf(ERROR_STRING,
			"Item type %s has no context to pop",IT_NAME(itp));
		warn(ERROR_STRING);
		return(NULL);
	}
	rls_node(np);

	np=QLIST_HEAD(LIST_OF_CONTEXTS(itp));
	// Normally this is not null, because we always have
	// the default context, but currently secondary threads
	// may not inherit the default!?
//	assert(np!=NULL);
	if( np != NULL )
		new_icp = (Item_Context *) NODE_DATA(np);
	else
		new_icp = NULL;

	popped_icp = current_context(itp);
	SET_CURRENT_CONTEXT(itp,new_icp);

#ifdef QUIP_DEBUG
if( debug & debug_contexts ){
sprintf(ERROR_STRING,"pop_item_context:  %s context %s popped",IT_NAME(itp),CTX_NAME(popped_icp));
NADVISE(ERROR_STRING);
}
#endif /* QUIP_DEBUG */

	// BUG - flags should be qsp-specific???
	//SET_IT_FLAG_BITS(itp, NEED_CHOICES|NEED_LIST );
	//SET_ITCI_FLAG_BITS( THIS_ITCI(itp), ITCI_NEEDS_LIST );	// mark list as not current
	INC_STACK_CHANGE_COUNT(itp);

	return(popped_icp);
}

void _delete_item_context( QSP_ARG_DECL  Item_Context *icp )
{
	delete_item_context_with_callback( icp, NULL );
}

void _delete_item_context_with_callback( QSP_ARG_DECL  Item_Context *icp, void (*func)(Item *) )
{
	Item_Type *itp;
	Node *np;

	/* delete all the items in this context.
	 * Because we use the user-supplied deletion method for the items
	 * in the context, the context must be active (preferably top
	 * of stack) when this is called.
	 * We search the entire context list for references, remove them,
	 * and then push the context back onto the top...
	 *
	 * This requires a user-supplied deletion method to avoid memory leaks.
	 *
	 * The above comment suggests that this routine is responsible for
	 * deleting ALL aspects of the object, not just the dictionary entries?
	 */

	itp = (Item_Type *) CTX_IT(icp);
	assert(itp!=NULL);

	while( (np=remData(LIST_OF_CONTEXTS(itp),icp)) != NULL ){
		rls_node(np);
	}

	/* make the context be on the top of the stack */
	push_item_context(itp,icp);

	if( IT_DEL_METHOD(itp) == no_del_method ){
		sprintf(ERROR_STRING,
	"No object deletion method provided for item type %s",IT_NAME(itp));
		warn(ERROR_STRING);
	} else {
		List *lp;

		//lp=dictionary_list(CTX_DICT(icp));
		lp=container_list(CTX_CONTAINER(icp));

		/* Don't use remHead to get the node, del_item()
		 * will remove it for us, and put it on the free list.
		 */
		while( lp!=NULL && (np=QLIST_HEAD(lp))!=NULL ){
			Item *ip;
			ip = (Item*) NODE_DATA(np);
			if( func != NULL ) (*func)(ip);
			(*IT_DEL_METHOD(itp))(QSP_ARG  ip);
			/* force list update in case hashing */
			//lp=dictionary_list(CTX_DICT(icp));
			lp=container_list(CTX_CONTAINER(icp));
		}
	}

	//delete_container(CTX_CONTAINER(icp));
	(*(CTX_CONTAINER(icp)->cnt_typ_p->delete))(QSP_ARG  CTX_CONTAINER(icp));

	del_ctx(icp);

	/* BUG? - perhaps we should make sure
	 * the context is not also pushed deeper down,
	 * to avoid a dangling ptr?
	 */

	/* Now pop the deleted context */

	pop_item_context(itp);
}

#ifdef NOT_NEEDED
Item *check_context(Item_Context *icp, const char *name)
{
	assert( icp != NULL );
	//return fetch_name(name,CTX_DICT(icp));
	return container_find_match(CTX_CONTAINER(icp),name);
}
#endif // NOT_NEEDED


/*
 * Return a pointer to the item of the given type with the given name,
 * or a null pointer if not found.  Use of this routine does not
 * imply a belief that the named item actually exists.
 *
 * Item types have different "contexts" or name spaces.
 * Originally, when contexts were introduced, there was a single
 * stack that was part of the item type.  But with multi-threading,
 * we would like to have different contexts for different
 * threads.  (That would mean, among other things, that we
 * wouldn't necessarily have to perform mutex locks when accessing
 * different contexts, although it would still be an issue
 * with the global context.)
 *
 * So now there is a tricky question - the context stack has to part
 * of the item type, but we want to have different ones based on the qsp;
 * We don't want to put a context stack for every item type into
 * the query stream...  We don't necessarily need different contexts
 * for every item type.  But for some we do:  one example is script
 * variables, where macros like Assign use scratch script vars...
 * One approach we can try is to use qsp->qs_serial to index an
 * array of context stacks?
 */

Item *_item_of( QSP_ARG_DECL  Item_Type *itp, const char *name )
		/* itp = type of item to search for */
		/* name = name of item */
{
	Node *np;

	assert( itp != NULL );

	if( *name == 0 ) return(NULL);

	assert( LIST_OF_CONTEXTS(itp) != NULL );

	np=QLIST_HEAD(LIST_OF_CONTEXTS(itp));

#ifdef THREAD_SAFE_QUERY
	if( np == NULL ){
		Item_Context *icp;
		// This occurs when we have a brand new thread...
		assert(QS_SERIAL!=0);
		// get the bottom of the stack from the root qsp, and push it...
		np = QLIST_TAIL( FIRST_LIST_OF_CONTEXTS(itp) );
		assert( np != NULL );
		icp = NODE_DATA(np);
		assert(icp!=NULL);
		// We tried pushing
		// BUG?  if a context can be pushed on two stacks, what happens when
		// it is popped from one?  Will it be released?
		push_item_context(itp, icp );
		np=QLIST_HEAD(LIST_OF_CONTEXTS(itp));
	}
#endif // THREAD_SAFE_QUERY

	assert( np != NULL );

	/* check the top context first */

	while(np!=NULL){
		Item_Context *icp;
		Item *ip;

		icp= (Item_Context*) NODE_DATA(np);
		assert(icp!=NULL);
		ip = container_find_match(CTX_CONTAINER(icp),name);
		if( ip!=NULL ){
			return(ip);
		}
		if( IS_RESTRICTED(itp) ){
			// We used to clear the flag here,
			// But now the place that depended on it,
			// (vectree/evaltree.c) has been changed to
			// explicitly clear it.

			//CTX_RSTRCT_FLAG(itp)=0;

			return(NULL);
		}
		np=NODE_NEXT(np);
	}

	/* not found in any context, including default */

	return(NULL);
} // item_of

/*
 * Return a pointer to the item of the given type with the given name,
 * or a null pointer if not found.  Use of this routine
 * implies a belief that the named item does exist.
 */

Item *_get_item( QSP_ARG_DECL  Item_Type *itp, const char* name )
		/* itp = type of item to search for */
		/* name = name of item */
{
	Item *ip;

	ip=item_of(itp,name);
	if( ip==NULL ){
		assert( itp != NULL );

		sprintf(ERROR_STRING,"no %s \"%s\"",
			IT_NAME(itp),name);
		warn(ERROR_STRING);
	}
	return(ip);
}

static int item_cmp(const void* ipp1,const void* ipp2)
{
#ifdef __cplusplus
  return strcmp(  ITEM_NAME((*(reinterpret_cast<const Item* const *>(ipp1)))),
  		ITEM_NAME((*(reinterpret_cast<const Item* const *>(ipp2)))) );
#else
  return( strcmp( ITEM_NAME( (*(const Item * const *)ipp1) ),
  		ITEM_NAME( (*(const Item * const *)ipp2) )) );
#endif
}


/*
 * Return a pointer to a list containing the items of the given type,
 * sorted in lexographic order by name.
 *
 * What do we do with contexts?  Do we include everything in the list?
 */

List *_item_list(QSP_ARG_DECL  Item_Type *itp)
	/* type of items to list */
{
	Node *np;

	assert( itp != NULL );

	/* First check and see if any of the contexts have been updated */

	assert(LIST_OF_CONTEXTS(itp)!=NULL);

	/*if( LIST_OF_CONTEXTS(itp) != NULL )*/
	{
		Node *context_np;
		context_np=QLIST_HEAD(LIST_OF_CONTEXTS(itp));
		while(context_np!=NULL){
			Item_Context *icp;
			icp=(Item_Context *) NODE_DATA(context_np);
			if( CTX_LIST_SERIAL(icp) != CTX_ITEM_SERIAL(icp) )
				INC_ITCI_ITEMS_CHANGE_COUNT(THIS_ITCI(itp));
			context_np=NODE_NEXT(context_np);
		}
	}

	// NEEDS_NEW_LIST tests the list serial against the item serial in ITCI...
	if( ! NEEDS_NEW_LIST(itp) ){
		/* Nothing changed, just return the existing list */
		return(current_item_list(itp));
	}

	/* Something has changed, so we have to rebuild the list.
	 * Begin by trashing the old list.
	 */
	
	while( (np=remHead(current_item_list(itp))) != NULL )
		rls_node(np);

	/* now make up the new list, by concatenating the context lists */
	if( LIST_OF_CONTEXTS(itp) != NULL ){
		Node *context_np;
		context_np=QLIST_HEAD(LIST_OF_CONTEXTS(itp));
		while(context_np!=NULL){
			Item_Context *icp;
			icp=(Item_Context *) NODE_DATA(context_np);
			cat_container_items(current_item_list(itp),CTX_CONTAINER(icp));
			context_np=NODE_NEXT(context_np);
			SET_CTX_LIST_SERIAL(icp, CTX_ITEM_SERIAL(icp) );
		}
	}

	//CLEAR_ITCI_FLAG_BITS(THIS_ITCI(itp), ITCI_NEEDS_LIST );
	SET_ITCI_LIST_ITEMS_SERIAL( THIS_ITCI(itp), ITCI_ITEMS_SERIAL(THIS_ITCI(itp)) );
	SET_ITCI_LIST_STACK_SERIAL( THIS_ITCI(itp), ITCI_STACK_SERIAL(THIS_ITCI(itp)) );

	return(current_item_list(itp));
}

/* reorder a list of items into alphanumeric order of item names */
/* the caller must dispose of the list! */

List *_alpha_sort(QSP_ARG_DECL  List *lp)
{
	count_t n2sort;
	void **ptr_array;
	Node *np;
	int i;


	assert( lp != NULL );

	n2sort=eltcount(lp);

	if( n2sort == 0 ) return(lp);

	ptr_array =(void**)  getbuf( n2sort * sizeof(void *) );

	if( ptr_array == NULL ) {
		NERROR1("make_choices:  out of memory");
		IOS_RETURN_VAL(NULL)
	}
	
	np=QLIST_HEAD(lp);
	i=0;
	while(np!=NULL){
		ptr_array[i++]=NODE_DATA(np);
		np=NODE_NEXT(np);
	}

	/* now sort the pointers */
	qsort(ptr_array,(size_t)n2sort,(size_t)sizeof(char *),item_cmp);

	lp = new_list();	// who is going to release this list!?
	for(i=0;i<n2sort;i++){
		np=mk_node(ptr_array[i]);
		addTail(lp,np);
	}
	givbuf(ptr_array);
	return(lp);
}

void _report_invalid_pick(QSP_ARG_DECL  Item_Type *itp, const char *s)
{
	sprintf(ERROR_STRING,"No %s \"%s\"",ITEM_TYPE_NAME(itp),s);
	warn(ERROR_STRING);

	sprintf(ERROR_STRING,"Possible %s choices:",ITEM_TYPE_NAME(itp));
	advise(ERROR_STRING);

	list_items(itp, tell_errfile());
}

/* BUG this should be gotten from enviroment, termcap, something... */
#define CHARS_PER_LINE	78

/*
 * Print the names of all of the items of the given type to stdout
 */

void _list_items(QSP_ARG_DECL  Item_Type *itp, FILE *fp)
	/* type of items to list */
{
	List *lp;

	CHECK_ITEM_INDEX(itp)

	lp=item_list(itp);
	print_list_of_items(lp, fp);
}


/* decap - convert a string from mixed case to all lower case.
 * We do this allow case-insensitive matching.
 */

void decap(char* sto,const char* sfr)
{
	while(*sfr){

		/* use braces in case macro is multiple statements... */
		/* don't increment inside macro ... */
		/* superstitious pc behavior */


		/* BUG should do this properly with autoconf... */

#ifdef SGI	/* or other SYSV os... */
		if( isupper(*sfr) ) { *sto++ = _tolower(*sfr); }

#else		/* sun 4.1.2 */
		if( isupper(*sfr) ) { *sto++ = tolower(*sfr); }
#endif

		else *sto++ = *sfr;

		sfr++;
	}
	*sto = 0;	/* terminate string */
}


/*
 * Find all items of a given type whose names
 * contain a geven substring.
 * Return a pointer to a list of the items,
 * caller must dispose of the list.
 *
 * BUG - should this be redone now that items are in rb trees?
 */

List *_find_items(QSP_ARG_DECL  Item_Type *itp,const char* frag)
{
	List *lp, *newlp=NULL;
	Node *np, *newnp;
	Item *ip;
	char lc_frag[LLEN];

	lp=item_list(itp);
	if( lp == NULL ) return(lp);

	np=QLIST_HEAD(lp);
	decap(lc_frag,frag);
	while(np!=NULL){
		char str1[LLEN];
		ip = (Item*) np->n_data;
		/* make the match case insensitive */
		decap(str1,ip->item_name);
		// strstr will match anywhere in the string!
		if( strstr(str1,lc_frag) != NULL ){
			if( newlp == NULL )
				newlp=new_list();
			newnp=mk_node(ip);
			addTail(newlp,newnp);
		}
		np=np->n_next;
	}
	return(newlp);
}


#ifdef NOT_USED
/* Sort the item list based on node priorities (set elsewhere) */

void _sort_item_list(QSP_ARG_DECL  Item_Type *itp)
{
	List *lp;

	lp=item_list(QSP_ARG  itp);
	if( lp == NULL ) return;

	p_sort(lp);
}
#endif /* NOT_USED */

void _print_list_of_items(QSP_ARG_DECL  List *lp, FILE *fp)
{
	Node *np;
	int n_per_line;
	char fmtstr[16];
	int i, n_lines, n_total;
#ifdef HAVE_ISATTY
	//FILE *out_fp;
	int maxlen;
#endif /* HAVE_ISATTY */

	/* allocate an array of pointers for sorting */

	if(lp==NULL) return;
	if( (n_total=eltcount(lp)) == 0 ) return;

	lp=alpha_sort(lp);
	assert( lp != NULL );

	/* If we are printing to the terminal, then
	 * we want as few lines as possible, but if we're
	 * printing to a file let's have 1 item per line
	 */

#ifdef HAVE_ISATTY
	//out_fp = tell_msgfile(SINGLE_QSP_ARG);
	if( isatty( fileno(fp /*out_fp*/) ) ){
		/* find the maximum length */

		np=QLIST_HEAD(lp);
		maxlen=0;
		while(np!=NULL){
			int l;
			if( (l=strlen( ITEM_NAME( ((Item *)NODE_DATA(np)) ) )) > maxlen )
				maxlen=l;
			np=NODE_NEXT(np);
		}
		n_per_line = CHARS_PER_LINE / (maxlen+3);
		if( n_per_line < 1 ) n_per_line=1;
		sprintf(fmtstr,"%%-%ds",maxlen+3);
		n_lines = 1+(n_total-1)/n_per_line;
	} else {
		/* one item per line, nothing fancy */
		/* we still have to count the items */
		n_lines = eltcount(lp);
		n_per_line = 1;
		sprintf(fmtstr,"%%s");
	}

#else /* ! HAVE_ISATTY */


	/* one item per line, nothing fancy */
	/* we still have to count the items */
	n_lines = eltcount(lp);
	n_per_line = 1;
	sprintf(fmtstr,"%%s");

#endif /* ! HAVE_ISATTY */

	/* print the names */

	for(i=0;i<n_lines;i++){
		int j;

		MSG_STR[0]=0;
		for(j=0; j< n_per_line; j++){
			int k;

			k=j*n_lines+i;
			if( k < n_total ){
				char tmp_str[LLEN];

				np = nth_elt(lp,k);

				assert( np != NULL );

				assert( strlen( ITEM_NAME(((Item *)NODE_DATA(np)) )) < LLEN );

				sprintf(tmp_str,fmtstr,
					ITEM_NAME(((Item *)NODE_DATA(np))));
				strcat(MSG_STR,tmp_str);
			}
		}
		//prt_msg(MSG_STR);
		fprintf(fp,"%s\n",MSG_STR);
	}

	/* now free the list */
	dellist(lp);
}


#ifdef NOT_USED

/*
 * Print the hash table statistics for a given class of item
 */

void item_stats(QSP_ARG_DECL  Item_Type * itp)
		/* type of item */
{
	Node *np;

	assert( itp != NULL );

	np = QLIST_HEAD(LIST_OF_CONTEXTS(itp));
	while(np!=NULL){
		Item_Context *icp;
		icp=(Item_Context *) NODE_DATA(np);
		sprintf(MSG_STR,"Context %s:",CTX_NAME(icp));
		prt_msg(MSG_STR);
//		tell_name_stats(CTX_DICT(icp));
		tell_container_stats(QSP_ARG  CTX_CONTAINER(icp));
		np = NODE_NEXT(np);
	}
}
#endif /* NOT_USED */

// used to be called add_to_item_freelist

void _recycle_item(QSP_ARG_DECL  Item_Type *itp, void *ip)
{
	Node *np;

	assert( itp != NULL );

	np=mk_node(ip);
	addTail(IT_FREE_LIST(itp),np);
}

/*
 * Delete the item pointed to by ip from the item database.
 * The actual storage for the item is not freed, but is
 * added to the item type's free list.  This routine
 * releases the name, so The caller must not
 * free it also...
 */

void _del_item(QSP_ARG_DECL  Item_Type *itp,void* ip)
		/* itp = type of item to be deleted */
		/* ip = pointer to item to be deleted */
{
	assert( itp != NULL );

	LOCK_ITEM_TYPE(itp)

	zombie_item(itp,(Item*) ip);
	rls_str( (char *) ITEM_NAME(((Item *)ip)) );
	SET_ITEM_NAME( ((Item *)ip), NULL );

	recycle_item(itp,ip);

	INC_ITEMS_CHANGE_COUNT(itp);

	UNLOCK_ITEM_TYPE(itp)
}

static int remove_item_from_context(QSP_ARG_DECL  Item_Context *icp, Item *ip)
{
	Item *tmp_ip;

	tmp_ip = container_find_match( CTX_CONTAINER(icp), ITEM_NAME((Item *)ip) );
	if( tmp_ip == ip ){	/* found it */
		if( remove_name_from_container(QSP_ARG  CTX_CONTAINER(icp),
					ITEM_NAME((Item *)ip) ) < 0 ){
			sprintf(ERROR_STRING,
	"zombie_item:  unable to remove item %s from context %s",
				ITEM_NAME(ip),CTX_NAME(icp));
			warn(ERROR_STRING);
		}
		/* BUG make the context needy... */
		/* does this do it? */
		INC_CTX_ITEM_SERIAL(icp);
		return 0;
	}
	return -1;
}

/* Remove an item from the item database, but do not return it to the
 * item free list.  This function was introduced to allow image objects
 * to be deleted after they have been displayed in viewers.  The
 * viewers can retain a ptr to the data which they need to access in
 * order to refresh the window.
 *
 * This function could have a more descriptive name, like zombieize_item,
 * or hide_item.
 *
 * This is kind of a hack and has a big potential for memory leaks!?
 * It is used in the initial implementation of static objects in vectree,
 * but this is probably not a good idea...  BUG.
 */

void _zombie_item(QSP_ARG_DECL  Item_Type *itp,Item* ip)
		/* itp = type of item to be deleted */
		/* ip = pointer to item to be deleted */
{
	Node *np;

	assert( itp != NULL );

	/* Find the context that contains the item, then remove it.
	 */

	np=QLIST_HEAD(LIST_OF_CONTEXTS(itp));
	while(np!=NULL){
		Item_Context *icp;

		icp = (Item_Context*) NODE_DATA(np);
		if( remove_item_from_context(QSP_ARG  icp,ip) == 0 ) return;
		np=NODE_NEXT(np);
	}
	// should never get here?
	error1("zombie_item:  unable to find item!?");
	// Is this point ever reached?
} // zombie_item

/* Rename an item.
 *
 * This function frees the old name and allocates permanent
 * storage for the new name.
 *
 * BUG?  It looks as if this could change the context, because it stores using
 * the current context, not the one  associated with the original object...
 */

void _rename_item(QSP_ARG_DECL  Item_Type *itp,void *ip,char* newname)
		/* itp = type of item to be deleted */
{
	LOCK_ITEM_TYPE(itp)

	assert( itp != NULL );

	zombie_item(itp,(Item*) ip);
	rls_str( (char *) ITEM_NAME(((Item *)ip)) );
	SET_ITEM_NAME( ((Item *)ip), savestr(newname) );
	_store_item(QSP_ARG  ITEM_CTX( (Item *)ip ),(Item*) ip);

	UNLOCK_ITEM_TYPE(itp)
}

static void _dump_item_context(QSP_ARG_DECL  Item_Context *icp)
{
	sprintf(MSG_STR,"\tContext \"%s\"",CTX_NAME(icp));
	prt_msg(MSG_STR);

	//dump_container_info(QSP_ARG  CTX_CONTAINER(icp));
	(*(CTX_CONTAINER(icp)->cnt_typ_p->dump_info))(QSP_ARG  CTX_CONTAINER(icp));

	list_item_context(icp);
}

void _list_item_context(QSP_ARG_DECL  Item_Context *icp)
{
	List *lp;
	lp=container_list(CTX_CONTAINER(icp));
	print_list_of_items(lp, tell_msgfile());
}

/*
 * Print a list of the names of the items of the given type,
 * preceded by the item type name and hash table size.
 */

static void _dump_item_type(QSP_ARG_DECL  Item_Type *itp)
{
	Node *np;

	sprintf(MSG_STR,
		"%s items",IT_NAME(itp));
	prt_msg(MSG_STR);

	np = QLIST_HEAD(LIST_OF_CONTEXTS(itp));
	while( np != NULL ){
		Item_Context *icp;

		icp = (Item_Context*) NODE_DATA(np);
		_dump_item_context(QSP_ARG  icp);

		np = NODE_NEXT(np);
	}
}

/*
 * Dump all existing item types.
 */

void _dump_items(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;
	Item_Type *itp;

	lp = item_list(ittyp_itp);
	if( lp == NULL ) return;
	np=QLIST_HEAD(lp);
	while(np!=NULL){
		itp=(Item_Type *) NODE_DATA(np);
		_dump_item_type(QSP_ARG  itp);
		np=NODE_NEXT(np);
	}
}

Item_Type *_get_item_type(QSP_ARG_DECL  const char* name)
{
	return( (Item_Type *) get_item(ittyp_itp,name) );
}

#ifdef THREAD_SAFE_QUERY

#ifdef HAVE_PTHREADS
void report_mutex_error(QSP_ARG_DECL  int status,const char *whence)
{
	const char *msg;

	sprintf(ERROR_STRING,"%s:  report_mutex_error:  status = %d",
		whence,status);
	advise(ERROR_STRING);

	switch(status){
		case EINVAL: msg = "invalid argument"; break;
		case EBUSY: msg = "mutex already locked"; break;
		case EAGAIN: msg = "too many recursive locks"; break;
		case EDEADLK: msg = "lock already owned"; break;
		case EPERM: msg = "not mutex owner"; break;
		default: msg = "unexpected mutex error"; break;
	}
	warn(msg);
}
#endif /* HAVE_PTHREADS */

#endif /* THREAD_SAFE_QUERY */

static Item_Type *frag_itp=NULL;
static ITEM_INIT_FUNC(Frag_Match_Info,frag,HASH_TBL_CONTAINER)
static ITEM_NEW_FUNC(Frag_Match_Info,frag)
//static ITEM_CHECK_FUNC(Frag_Match_Info,frag);

static Item_Context *setup_frag_context(QSP_ARG_DECL  Item_Context *icp)
{
	char cname[LLEN];
	Item_Context *frag_icp;

	if( frag_itp == NULL ) _init_frags(SINGLE_QSP_ARG);
	sprintf(cname,"fragments.%s",CTX_NAME(icp));
	frag_icp = new_ctx(cname);
	assert(frag_icp!=NULL);

	// should we have a function to encapsulate the setup?
	SET_CTX_IT( frag_icp, frag_itp );
	SET_CTX_CONTAINER(frag_icp , create_container(CTX_NAME(frag_icp),HASH_TBL_CONTAINER) );
	//SET_CTX_FLAGS(icp,0);

	return frag_icp;
}

static void rebuild_frag_match_info(QSP_ARG_DECL  Frag_Match_Info *fmi_p )
{
	// nothing to free, just clear old ptrs
	bzero(&(fmi_p->fmi_u),sizeof(fmi_p->fmi_u));
	// BUG  no need to pass all these args - sloppy
	container_find_substring_matches(fmi_p,FMI_STRING(fmi_p));
	SET_FMI_ITEM_SERIAL(fmi_p, CTX_ITEM_SERIAL( FMI_CTX(fmi_p) ) );
}

// return a ptr to a struct containing <a list> of all of the partial matches from a context.
// The "list" may be ptrs into a rbtree...
// We have a complication when we have a stack of contexts...
// because searching a context clears the old content of the frag match info.
// We could associate the frag_match_info with the context, but how would we cycle?

static Frag_Match_Info *context_partial_match(QSP_ARG_DECL  Item_Context *icp, const char *s )
{
	Frag_Match_Info *fmi_p;

	// first see if we have fragment context for this contex
	if( CTX_FRAG_ICP(icp) == NULL ){
		SET_CTX_FRAG_ICP( icp, setup_frag_context(QSP_ARG  icp) );
	}
	// now search the context for the string
	fmi_p = (Frag_Match_Info *) container_find_match(CTX_CONTAINER(CTX_FRAG_ICP(icp)),s);
	if( fmi_p==NULL ){
		// create the struct
		push_item_context(frag_itp, CTX_FRAG_ICP(icp) );
		fmi_p = _new_frag(QSP_ARG  s );
		pop_item_context(frag_itp );
		assert(fmi_p!=NULL);
		SET_FMI_CTX(fmi_p,icp);
		// Now we need to fill in the entries!
		rebuild_frag_match_info(QSP_ARG  fmi_p);
	}
	  else {
		// If the context has changed, then we need to make sure that this
		// fragment doesn't match something which has been deleted
		if( FMI_ITEM_SERIAL(fmi_p) != CTX_ITEM_SERIAL(icp) ){
			rebuild_frag_match_info(QSP_ARG  fmi_p);
		}
		// BUG?  do we need to re-search after rebuilding???
		// Or does rebuild alter the target of fmi_p???
	}
	return fmi_p;
} // context_partial_match

static Item *current_cycle_item(Match_Cycle *mc_p)
{
	Frag_Match_Info *fmi_p;

	assert( MATCH_CYCLE_CURR_NODE(mc_p) != NULL );
	fmi_p = NODE_DATA(MATCH_CYCLE_CURR_NODE(mc_p));
	assert(fmi_p!=NULL);

	return current_frag_item(fmi_p);
}

static Item_Type *match_cycle_itp=NULL;
static ITEM_INIT_FUNC(Match_Cycle,match_cycle,0)
static ITEM_CHECK_FUNC(Match_Cycle,match_cycle)
static ITEM_NEW_FUNC(Match_Cycle,match_cycle)

#define init_match_cycles()	_init_match_cycles(SINGLE_QSP_ARG)
#define match_cycle_of(name)	_match_cycle_of(QSP_ARG  name)
#define new_match_cycle(name)	_new_match_cycle(QSP_ARG  name)

#define find_match_cycle(itp,s)	_find_match_cycle(QSP_ARG  itp,s)

static Match_Cycle * _find_match_cycle(QSP_ARG_DECL  Item_Type *itp, const char *s )
{
	Match_Cycle *mc_p;

	if( IT_MC_CONTEXT(itp) == NULL ){
		// Make a context
		char cname[LLEN];
		sprintf(cname,"Matches.%s.%d",ITEM_TYPE_NAME(itp),QS_SERIAL);	// BUG possible buffer overrun
		if( match_cycle_itp == NULL ){
			//match_cycle_itp = new_item_type("Match_Cycle", 0 );
			init_match_cycles();
			assert(match_cycle_itp!=NULL);
		}
		SET_IT_MC_CONTEXT(itp, create_item_context(match_cycle_itp,cname));
	}

	push_item_context(match_cycle_itp, IT_MC_CONTEXT(itp) );

	mc_p = match_cycle_of(s);
	if( mc_p != NULL ){	// already exists?
		pop_item_context(match_cycle_itp);
		// BUG here we need to make sure that match cycle is still current...
		return mc_p;
	}

	mc_p = new_match_cycle(s);
	pop_item_context(match_cycle_itp);
	SET_MATCH_CYCLE_LIST(mc_p,new_list());
	SET_MATCH_CYCLE_CURR_NODE(mc_p,NULL);
	SET_MATCH_CYCLE_IT(mc_p,itp);
	SET_MC_ITEM_SERIAL(mc_p,0);
	SET_MC_STACK_SERIAL(mc_p,0);

	return mc_p;
} // find_match_cycle

static void add_matches_to_cycle(QSP_ARG_DECL  /*Match_Cycle*/ void *mc_p, Item_Context *icp )
{
	Frag_Match_Info *fmi_p;
	Node *mc_np;

	fmi_p=context_partial_match(QSP_ARG  icp,MATCH_CYCLE_STRING((Match_Cycle *)mc_p));

	assert( fmi_p != NULL );
	// Should this return non-null if there are no matches?

	mc_np = mk_node(fmi_p);
	addTail(MATCH_CYCLE_LIST((Match_Cycle *)mc_p),mc_np);
	SET_MATCH_CYCLE_CURR_NODE((Match_Cycle *)mc_p,mc_np);
}

static void apply_to_context_stack(QSP_ARG_DECL  Item_Type *itp,
	void (*func)(QSP_ARG_DECL  void *, Item_Context *), void * ptr )
{
	Node *np;

	np=QLIST_HEAD(LIST_OF_CONTEXTS(itp));
	assert( np != NULL );

	while(np!=NULL){
		Item_Context *icp;
		icp= (Item_Context*) NODE_DATA(np);
		(*func)(QSP_ARG  ptr, icp );
		np = NODE_NEXT(np);
	}
}

static void insure_cycle_ready(QSP_ARG_DECL  Match_Cycle *mc_p)
{
	Node *first_np;
	Node *np;
	Frag_Match_Info *fmi_p;

	first_np = MATCH_CYCLE_CURR_NODE(mc_p);
	np = first_np;
	do {
		Item *ip;
		fmi_p = NODE_DATA(np);
		ip = current_frag_item(fmi_p);
		if( ip != NULL ){
			SET_MATCH_CYCLE_CURR_NODE(mc_p,np);
			return;
		}

		np = NODE_NEXT(np);
		if( np == NULL )
			np = QLIST_HEAD( MATCH_CYCLE_LIST(mc_p) );
	} while( np != first_np );
}

static Match_Cycle * get_partial_match_cycle(QSP_ARG_DECL  Item_Type *itp, const char *s )
{
	//Frag_Match_Info *fmi_p;
	Match_Cycle *mc_p;

	mc_p = find_match_cycle(itp, s);

	apply_to_context_stack(QSP_ARG  itp, add_matches_to_cycle, mc_p );

	SET_MC_STACK_SERIAL(mc_p, ITCI_STACK_SERIAL( THIS_ITCI(MATCH_CYCLE_IT(mc_p)) ) );

	// Make sure that the cycle points to a node with something...
	insure_cycle_ready(QSP_ARG  mc_p);

	return mc_p;
} // get_partial_match_cycle

static int cycle_is_current(QSP_ARG_DECL  Match_Cycle *mc_p)
{
	if( mc_p == NULL ) return 1;

	// The Match_Cycle list should match the context stack...
	// But the context stack can have contexts pushed and popped.
	// So we should rebuild the cycle list to match the stack -
	// Or can we use the stack itself?
	if( MC_STACK_SERIAL(mc_p) != ITCI_STACK_SERIAL(THIS_ITCI(MATCH_CYCLE_IT(mc_p))) )
		return 0;
	if( MC_ITEM_SERIAL(mc_p) != ITCI_ITEMS_SERIAL(THIS_ITCI(MATCH_CYCLE_IT(mc_p))) )
		return 0;
	return 1;
}

// The stack may have changed, so we have to start over...

static void rebuild_match_cycle(QSP_ARG_DECL  Match_Cycle *mc_p)
{
	// First releast everything
	Node *np;

	while( (np=remHead(MATCH_CYCLE_LIST(mc_p))) != NULL ){
		//Frag_Match_Info *fmi_p;
		//fmi_p = NODE_DATA(np);

		// We don't have to deallocate the frag_match_info structs...
		// they should retain references through their own context.
		rls_node(np);
	}
	SET_MATCH_CYCLE_CURR_NODE(mc_p,NULL);

	// Now rebuild, scan the whole context stack
	np = QLIST_HEAD( ITCI_CSTK( THIS_ITCI( MATCH_CYCLE_IT(mc_p)) ));
	while( np != NULL ){
		Frag_Match_Info *fmi_p;
		Item_Context *icp;
		Node *new_np;

		icp = NODE_DATA(np);
		fmi_p = context_partial_match( QSP_ARG  icp, MATCH_CYCLE_STRING(mc_p) );
		new_np = mk_node(fmi_p);
		addTail(MATCH_CYCLE_LIST(mc_p),new_np);
		if( MATCH_CYCLE_CURR_NODE(mc_p) == NULL )
			SET_MATCH_CYCLE_CURR_NODE(mc_p,new_np);

		np = NODE_NEXT(np);
	}
	SET_MC_STACK_SERIAL(mc_p, ITCI_STACK_SERIAL( THIS_ITCI(MATCH_CYCLE_IT(mc_p)) ) );
} // rebuild_match_cycle

// update_frag_matches is used when the stack has not changed, but some items have

static void update_frag_matches(QSP_ARG_DECL  Match_Cycle *mc_p)
{
	Node *np;
	Frag_Match_Info *fmi_p;

	assert( MATCH_CYCLE_LIST(mc_p) != NULL );
	np = QLIST_HEAD( MATCH_CYCLE_LIST(mc_p) );
	while( np != NULL ){
		fmi_p = NODE_DATA(np);
		if( FMI_ITEM_SERIAL(fmi_p) != CTX_ITEM_SERIAL( FMI_CTX(fmi_p) ) ){
			rebuild_frag_match_info(QSP_ARG  fmi_p);
		}
		np = NODE_NEXT(np);
	}
}

static void update_match_cycle(QSP_ARG_DECL  Match_Cycle *mc_p)
{
	// The update could be required either because the stack has changed, or an item has changed...

	if( MC_STACK_SERIAL(mc_p) == ITCI_STACK_SERIAL( THIS_ITCI(MATCH_CYCLE_IT(mc_p)) ) ){
		// update items only
		update_frag_matches(QSP_ARG  mc_p);
	} else {
		// The stack has changed.  We don't know how it has changed, so the most fool-proof
		// way to go is just to release and rebuild.  BUT we would like to remember our place in
		// the cycle...  ignore for now - BUG.
		rebuild_match_cycle(QSP_ARG  mc_p);
	}
}

// We can't cycle through a hash table as we can a list or tree,
// so we have to construct match "cycles" of partial matches...

const char *_find_partial_match( QSP_ARG_DECL  Item_Type *itp, const char *s )
{
	Match_Cycle *mc_p;
	Item *ip;

	if( (mc_p=IT_MATCH_CYCLE(itp)) == NULL  ||		// old match cycle cached?
			strcmp( s, IT_MATCH_CYCLE(itp)->it.item_name) ){	// same fragment?
		// If there is no cached cycle, or the cached cycle is for a different fragment,
		// then we have to get a different one...
		mc_p = get_partial_match_cycle(QSP_ARG  itp, s);
		SET_IT_MATCH_CYCLE(itp,mc_p);
	} else if( ! cycle_is_current(QSP_ARG  mc_p) ){
		update_match_cycle(QSP_ARG  mc_p);
	}

	if( mc_p == NULL ) return "";	// there may be no matches
	ip = current_cycle_item(mc_p);	// BUG should be agnostic with regard to container type!
	if( ip == NULL ) return "";
	return ITEM_NAME(ip);
}

// These used to be accessed directly from the struct, but now we provide accessor functions
// to insure proper initialization (necessary when multi-threading)

List *_context_stack(QSP_ARG_DECL  Item_Type *itp)
{
	Item_Type_Context_Info *itci_p;

	itci_p = THIS_ITCI(itp);

	if( ITCI_CSTK( itci_p ) == NULL ){
		SET_ITCI_CSTK(itci_p,new_list());
	}
	return ITCI_CSTK(itci_p);
}

List *_current_item_list(QSP_ARG_DECL  Item_Type *itp)
{
	return ITCI_ITEMS_LIST(THIS_ITCI(itp));
}

Item_Context *_current_context(QSP_ARG_DECL  Item_Type *itp)
{
	List *lp;
	Item_Context *icp;
	Item_Type_Context_Info *itci_p;

	itci_p = THIS_ITCI(itp);
	if( ITCI_CTX( itci_p ) != NULL )
		return ITCI_CTX(itci_p);

	// If it is null, then this is probably the first access
	// from a new thread

	lp = context_stack(itp);
	if( eltcount(lp) > 0 ){
		Node *np;
		np = QLIST_HEAD(lp);
		icp = NODE_DATA(np);
		SET_ITCI_CTX(itci_p,icp);
		return icp;
	}

	// The context stack is empty...
	// Get the default context from the parent thread,
	// and push it.  It is conceivable that the parent
	// thread doesn't have a context - then we would like
	// it to go to its own parent.  But for now we
	// won't worry about that...

#ifdef THREAD_SAFE_QUERY
	icp = ITCI_CTX( ITCI_AT_INDEX(itp,QS_PARENT_SERIAL(THIS_QSP)) );

#else // ! THREAD_SAFE_QUERY
	icp = ITCI_CTX( ITCI_AT_INDEX(itp,0) );
#endif // ! THREAD_SAFE_QUERY

	assert(icp!=NULL);

	push_item_context(itp, icp );

	assert(ITCI_CTX(itci_p)!=NULL);
	return ITCI_CTX(itci_p);
}


