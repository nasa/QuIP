#include "quip_config.h"

char VersionId_qutil_items[] = QUIP_VERSION_STRING;

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

#ifdef HAVE_UNISTD_H
#include <unistd.h>		/* isatty() */
#endif

/* This was not defined on old SunOS */
#ifndef tolower
#define tolower(c)		(( c )+'a'-'A')
#endif


#include "items.h"
#include "query.h"

#ifdef DEBUG
static u_long item_debug=ITEM_DEBUG_MASK;
static u_long debug_contexts=CTX_DEBUG_MASK;
#endif /* DEBUG */

#ifdef HAVE_HISTORY
#include "../interpreter/history.h"	/* add_phist() */
#endif /* HAVE_HISTORY */

/* local prototypes */
#ifdef CAUTIOUS
static void check_item_type(Item_Type *itp);
#endif /* CAUTIOUS */
static void init_itp(Item_Type *itp);
static int item_cmp(const void *,const void *);
static const char **make_choices(QSP_ARG_DECL  int *countp,List *lp);
static void setup_item_choices(QSP_ARG_DECL  Item_Type * itp);
static Item_Type * init_item_type(QSP_ARG_DECL  const char *name);
static void store_item(QSP_ARG_DECL  Item_Type *itp,Item *ip,Node *np);

static void ctx_init(SINGLE_QSP_ARG_DECL);
static Item_Context *get_ctx(QSP_ARG_DECL  const char *name);
static Item_Context *new_ctx(QSP_ARG_DECL  const char *name);
static Item_Context *del_ctx(QSP_ARG_DECL  const char *name);

/*
static Item_Context *pick_ctx(const char *prompt);
static void list_ctxs(void);
*/

static void 		type_init(SINGLE_QSP_ARG_DECL);
static Item_Type *	new_ittyp(QSP_ARG_DECL  const char *);

static void		dump_item_context(QSP_ARG_DECL  Item_Context *);


#define NO_STR_ARRAY	((const char **)NULL)


static Item_Type * type_itp=NO_ITEM_TYPE;

#define ITEM_TYPE_STRING	"Item_Type"

static DECL_INIT_FUNC(type_init,type_itp,ITEM_TYPE_STRING)
static DECL_NEW_FUNC(new_ittyp,Item_Type *,type_itp,type_init)

DECL_LIST_FUNC(list_ittyps,type_itp,type_init)
DECL_PICK_FUNC(pick_ittyp,Item_Type *,type_itp,type_init)

static Item_Type *ctx_itp=NO_ITEM_TYPE;

#define CTX_IT_NAME	"Contexts"
#define DEF_CTX_NAME	"default"

static DECL_INIT_FUNC(ctx_init,ctx_itp,CTX_IT_NAME)
DECL_OF_FUNC(ctx_of,Item_Context *,ctx_itp,ctx_init)
static DECL_GET_FUNC(get_ctx,Item_Context *,ctx_itp,ctx_init)
DECL_LIST_FUNC(list_ctxs,ctx_itp,ctx_init)
static DECL_NEW_FUNC(new_ctx,Item_Context *,ctx_itp,ctx_init)
static DECL_DEL_FUNC(del_ctx,Item_Context *,ctx_itp,get_ctx)
/* static DECL_PICK_FUNC(pick_ctx,Item_Context *,ctx_itp,ctx_init) */

#define CHECK_ITEM_INDEX( itp )	if( ( itp ) == NO_ITEM_TYPE ){		\
					WARN("Null item type");		\
					return;				\
				}


/* a global */
u_long total_from_malloc = 0;


#ifdef CAUTIOUS
static void check_item_type(Item_Type *itp)
{
	if( itp == NO_ITEM_TYPE )
		NERROR1("CAUTIOUS:  Null item type");
}
#endif /* CAUTIOUS */

/* If we don't use an ansii style declaration,
 * we get warnings on the pc (microsoft compiler)
 */

static void no_del_method(TMP_QSP_ARG_DECL  Item *ip)
{
	WARN("Undefined item deletion method");
}

void set_del_method(QSP_ARG_DECL  Item_Type *itp,void (*func)(TMP_QSP_ARG_DECL  Item *))
{
#ifdef CAUTIOUS
	if( itp->it_del_method != no_del_method ){
		sprintf(ERROR_STRING,
	"Item type %s already has a deletion method defined!?",itp->it_name);
		WARN(ERROR_STRING);
	}
#endif /* CAUTIOUS */
	itp->it_del_method = func;
}

static void init_itp(Item_Type *itp)
{
	/* should we really do this? */
	itp->it_lp = new_list();
	itp->it_free = new_list();
	itp->it_flags = NEED_CHOICES;
	itp->it_choices = NO_STR_ARRAY;

	itp->it_context_itp=NO_ITEM_TYPE;

	itp->it_classlist = NO_LIST;
	itp->it_del_method = no_del_method;

#ifdef THREAD_SAFE_QUERY
	{
		int i;
//if( n_active_threads==0 )NERROR1("init_itp:  no active threads!?");
		/*
		itp->it_contexts[0] = new_list();
		for(i=1;i<n_active_threads;i++)
			itp->it_contexts[i] = new_list();
		for(;i<MAX_QUERY_STREAMS;i++)
			itp->it_contexts[i] = NO_LIST;
			*/
		/* What the heck - just allocate all the lists */
		for(i=0;i<MAX_QUERY_STREAMS;i++)
			itp->it_contexts[i] = new_list();
	}
	{
		//itp->it_mutex = PTHREAD_MUTEX_INITIALIZER;
		int status;

		status = pthread_mutex_init(&itp->it_mutex,NULL);
		if( status != 0 )
			NERROR1("error initializing mutex");
	}
#else /* ! THREAD_SAFE_QUERY */
	itp->it_contexts = new_list();
#endif /* ! THREAD_SAFE_QUERY */
}

#ifdef THREAD_SAFE_QUERY
/* We call this when we have added a new query stream */

static void setup_item_type_context(QSP_ARG_DECL  Item_Type *itp, Query_Stream *new_qsp)
{
	Item_Context *icp;
	Node *np;

	np = itp->it_contexts[0]->l_tail;
	icp = (Item_Context *)np->n_data;
	push_item_context(new_qsp,itp,icp);
}

void setup_all_item_type_contexts(TMP_QSP_ARG_DECL  void *new_qsp)
{
	/* Push the default context onto ALL item types */
	List *lp;
	Item_Type *itp;
	Node *np;

	lp = item_list(QSP_ARG  type_itp);
	np=lp->l_head;
	while( np != NO_NODE ){
		itp = (Item_Type *)np->n_data;
		setup_item_type_context(QSP_ARG  itp,(Query_Stream *)new_qsp);
		np=np->n_next;
	}
}
#endif /* THREAD_SAFE_QUERY */

static Item_Type * init_item_type(QSP_ARG_DECL  const char *name)
{
	static int is_first=1;	// how many times is this auto-initialized?
	Item_Type *itp;
	Item_Context *icp;

//sprintf(ERROR_STRING,"init_item_type %s, is_first = %d",name,is_first);
//NADVISE(ERROR_STRING);
	if( is_first ){
		static Item_Context first_item_context;
		Node *np;
#ifdef THREAD_SAFE_QUERY
		/* a real hack... */
		Query_Stream dummy_qs;
#endif /* THREAD_SAFE_QUERY */

		// The first call is for Query_Stream.
		// Item_Type is initialized below to prevent recursion. 
		static Item_Type first_item_type;

//sprintf(ERROR_STRING,"init_item_type:  first_item_type is %s",name);
//advise(ERROR_STRING);

		/* Item_Type's are themselves items...
		 * but we can't call new_item_type to create
		 * the item_type for item_types!
		 */

		/* we don't call new_ittyp() to avoid a recursion problem */
		type_itp = &first_item_type;
		type_itp->it_name = savestr(ITEM_TYPE_STRING);
		is_first=0;
		init_itp(type_itp);

		/* We need to create the first context, but we don't want
		 * infinite recursion...
		 */

		FIRST_CONTEXT(type_itp) = &first_item_context;
		/* BUG make sure DEF_CTX_NAME matches what is here... */
		FIRST_CONTEXT(type_itp)->ic_name = savestr("Item_Type.default");
		FIRST_CONTEXT(type_itp)->ic_nsp = create_namespace("Item_Type.default");
		FIRST_CONTEXT(type_itp)->ic_itp = type_itp;
		np = mk_node(FIRST_CONTEXT(type_itp));
		addHead(FIRST_CONTEXT_LIST(type_itp),np);

#ifdef THREAD_SAFE_QUERY
		/* a real hack... */
		qsp=&dummy_qs;
		qsp->qs_serial=0;
#endif /* THREAD_SAFE_QUERY */

		/* why do this? Do we really need to? */
		add_item(QSP_ARG  type_itp,type_itp,NO_NODE);
	}
#ifdef CAUTIOUS
	if( !strcmp(name,ITEM_TYPE_STRING) ){
		WARN("CAUTIOUS:  don't call init_item_type for item_type");
		return(NO_ITEM_TYPE);
	}
#endif /* CAUTIOUS */

	itp = new_ittyp(QSP_ARG  name);
	init_itp(itp);
	icp = create_item_context(QSP_ARG  itp,DEF_CTX_NAME);
	PUSH_ITEM_CONTEXT(itp,icp);

	return(itp);
}

/*
 * Create a new item type.  Allocate a hash table with hashsize
 * entries.  Return the item type index or -1 on failure.
 */

Item_Type * new_item_type(QSP_ARG_DECL  const char *atypename)
{
	Item_Type * itp;

	if( type_itp != NO_ITEM_TYPE ){
		Item *ip;

		ip = item_of(QSP_ARG  type_itp,atypename);
		if( ip != NO_ITEM ){
			sprintf(ERROR_STRING,
			"Item type name \"%s\" is already in use\n",atypename);
			WARN(ERROR_STRING);
			return(NO_ITEM_TYPE);
		}
	}
	/* else we are initializing the item type Item_Type */

	itp=init_item_type(QSP_ARG  atypename);
#ifdef CAUTIOUS
	if( itp == NO_ITEM_TYPE )
		WARN("CAUTIOUS:  new_item_type failed!?");
#endif /* CAUTIOUS */

	if( type_itp==NO_ITEM_TYPE ){
		type_itp = itp;
	}

	return(itp);
}

/*
 * Put an item into the corresponding name space
 */

static void store_item( QSP_ARG_DECL  Item_Type *itp, Item *ip, Node *np )
{
	if( insert_name(ip,np,CURRENT_CONTEXT(itp)->ic_nsp) < 0 ){

		/* We used to enlarge the hash table here, but now we automatically enlarge
		 * the hash table before it becomes completely full...
		 */

		sprintf(ERROR_STRING,
			"Error storing name %s",ip->item_name);
		NERROR1(ERROR_STRING);
	}
}

/* This routine was eliminated, but we have reinstated it, so that
 * command tables can be preallocated items.
 */

void add_item( TMP_QSP_ARG_DECL  Item_Type *itp, void *ip, Node *np )
{
#ifdef CAUTIOUS
	check_item_type( itp );
#endif /* CAUTIOUS */

	store_item(QSP_ARG  itp,(Item*) ip,np);

	/*
	if( np==NO_NODE )
	*/
	/*
		itp->it_flags |= CONTEXT_CHANGED;

	itp->it_flags |= NEED_CHOICES;
	*/
	make_needy(itp);
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

Item *new_item( QSP_ARG_DECL  Item_Type *itp, const char* name, size_t size )
{
	Item *ip;
	Node *np;

#ifdef CAUTIOUS
	if( *name == 0 ){
		sprintf(ERROR_STRING,
	"CAUTIOUS Can't create item of type \"%s\" with null name",
			itp->it_name);
		WARN(ERROR_STRING);
		return(NO_ITEM);
	}
#endif /* CAUTIOUS */

	LOCK_ITEM_TYPE(itp)

	/* We will allow name conflicts if they are not in the same context */

	/* Only check for conflicts in the current context */
	ip = fetch_name(name,CURRENT_CONTEXT(itp)->ic_nsp);

	if( ip != NO_ITEM ){
		UNLOCK_ITEM_TYPE(itp);
		sprintf(ERROR_STRING,
	"%s name \"%s\" is already in use in context %s",
			itp->it_name,name,CURRENT_CONTEXT(itp)->ic_name);
		WARN(ERROR_STRING);
		return(NO_ITEM);
	}

	// Try to get a structure from the free list
	// If the free list is empty, then allocate a page's worth

	if( itp->it_free->l_head == NO_NODE ){
		int n_per_page;
		char *nip;

#define FOUR_K	4096

		n_per_page = FOUR_K / size;	/* BUG use PAGESIZE */

		if( n_per_page <= 0 ){
			/* cast size to u_long because size_t is u_long on IA64 and
			 * u_int on IA32!?
			 */
			// If the item is bigger than a page, just get one
			n_per_page=1;
		}

#ifdef DEBUG
if( debug & item_debug ){
sprintf(ERROR_STRING,"malloc'ing %d more %s items",n_per_page,itp->it_name);
NADVISE(ERROR_STRING);
}
#endif /* DEBUG */
		/* get a pages worth of items */
		nip = (char*)  malloc( n_per_page * size );
		total_from_malloc += n_per_page*size;

		if( nip == NULL ){
			sprintf(ERROR_STRING,
		"new_item:  out of memory while getting a new page of %s's",
				itp->it_name);
			NERROR1(ERROR_STRING);
		}

		while(n_per_page--){
			Node *np;

			np = mk_node(nip);
			addTail(itp->it_free,np);
			nip += size;
		}
	}

	np = remHead(itp->it_free);

#ifdef CAUTIOUS
	if( np == NO_NODE )
		NERROR1("CAUTIOUS:  new_item:  couldn't remove node from item free list!?");
#endif /* CAUTIOUS */

	ip = (Item *) np->n_data;
	ip->item_name = savestr(name);

	/* BUG? should we worry about nodes here? */

	add_item(QSP_ARG  itp,ip,np);

	UNLOCK_ITEM_TYPE(itp)

	return(ip);
} // end new_item

void list_item_contexts( QSP_ARG_DECL   Item_Type *itp )
{
	Node *np;

	if( CONTEXT_LIST(itp)==NO_LIST ||
		(np=CONTEXT_LIST(itp)->l_head)==NO_NODE){

		sprintf(ERROR_STRING,"Item type \"%s\" has no contexts",
			itp->it_name);
		NADVISE(ERROR_STRING);
		return;
	}

	while(np!=NO_NODE ){
		Item_Context *icp;

		icp=(Item_Context *) np->n_data;
		sprintf(msg_str,"%s",icp->ic_name);
		prt_msg(msg_str);

		np=np->n_next;
	}
}

/* Create a new context with the given name.
 * It needs to be push'ed in order to make it be
 * the current context for new item creation.
 */

Item_Context * create_item_context( QSP_ARG_DECL  Item_Type *itp, const char* name )
{
	Item_Context *icp;
	char cname[LLEN];

	/* maybe we should have contexts for contexts!? */

	sprintf(cname,"%s.%s",itp->it_name,name);

	if( (!strcmp(itp->it_name,CTX_IT_NAME)) && !strcmp(name,DEF_CTX_NAME) ){
		static Item_Context first_context;

		/* can't use new_ctx()
		 *
		 * Why not???
		 */
		icp = &first_context;
		icp->ic_name = savestr(cname);
		icp->ic_itp=itp;
		icp->ic_nsp = create_namespace(icp->ic_name);
		icp->ic_flags = 0;
		/* BUG?  not in the context database?? */
		return(icp);
	}

	/* Create an item type for contexts.
	 *
	 * Because new_item_type() calls create_item_text for the default
	 * context, we have the special case above...
	 */

	if( ctx_itp == NO_ITEM_TYPE )
		ctx_itp = new_item_type(QSP_ARG  CTX_IT_NAME);

#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  creating context %s",
WHENCE(create_item_context),cname);
advise(ERROR_STRING);
}
#endif	/* DEBUG */

	icp = new_ctx(QSP_ARG  cname);

	/* If it_context_itp was null, make sure it has the right value now */
	// This is the item type for ALL the contexts of all item types, not just
	// for this item type...  The context names typically have the item name
	// and a single dot as a prefix...
	itp->it_context_itp = ctx_itp;

	if( icp == NO_ITEM_CONTEXT ){
		return(icp);
	}

	icp->ic_itp=itp;
	icp->ic_nsp = create_namespace(icp->ic_name);
	icp->ic_flags = 0;

#ifdef CAUTIOUS
	if( icp->ic_nsp == NO_NAMESPACE ){
		sprintf(ERROR_STRING,
	"CAUTIOUS:  error creating namespace %s",icp->ic_name);
		NERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	return(icp);
}


#ifdef FOO
void show_context_stack(Item_Type *itp)
{
	Item_Context *icp;
	Node *np;

	if( CONTEXT_LIST(itp)==NO_LIST ){
none:
		sprintf(ERROR_STRING,"No contexts in existence for %s items",itp->it_name);
		NADVISE(ERROR_STRING);
		return;
	}
	np=CONTEXT_LIST(itp)->l_head;
	if( np == NO_NODE ) goto none;

	sprintf(ERROR_STRING,"%s contexts:",itp->it_name);
	NADVISE(ERROR_STRING);
	while( np != NO_NODE ){
		icp = np->n_data;
		sprintf(ERROR_STRING,"\t%s",icp->ic_name);
		NADVISE(ERROR_STRING);
		np=np->n_next;
	}
}
#endif /* FOO */

/* push an existing context onto the top of stack for this item class */

void push_item_context( QSP_ARG_DECL   Item_Type *itp, Item_Context *icp )
{
	Node *np;

	/* might be a good idea to check here that the context item type
	 * matches itp.
	 */

#ifdef CAUTIOUS
	if( icp == NO_ITEM_CONTEXT ){
		sprintf(ERROR_STRING,"CAUTIOUS:  attempted to push null context, %s item type!?",
			itp->it_name);
		NERROR1(ERROR_STRING);
	}
#endif

#ifdef DEBUG
if( debug & debug_contexts ){
sprintf(ERROR_STRING,"push_item_context:  pushing %s context %s",itp->it_name,icp->ic_name);
NADVISE(ERROR_STRING);
}
#endif /* DEBUG */
	np=mk_node(icp);
	if( CONTEXT_LIST(itp) == NO_LIST )
		CONTEXT_LIST(itp) = new_list();
	addHead(CONTEXT_LIST(itp),np);
	CURRENT_CONTEXT(itp) = icp;

	// BUG - flags should be qsp-specific???
	itp->it_flags |= NEED_CHOICES|NEED_LIST;
}

/*
 * Remove the top-of-stack context, but do not destroy.
 */

Item_Context * pop_item_context( QSP_ARG_DECL  Item_Type *itp )
{
	Node *np;
	Item_Context *icp;

	/* don't remove the context from the list yet, it needs
	 * to be there to find the objects in the context...
	 */
	np=remHead(CONTEXT_LIST(itp));
	if( np==NO_NODE ){
		sprintf(ERROR_STRING,
			"Item type %s has no context to pop",itp->it_name);
		WARN(ERROR_STRING);
		return(NO_ITEM_CONTEXT);
	}
	icp = (Item_Context *)  np->n_data;
	rls_node(np);

#ifdef DEBUG
if( debug & debug_contexts ){
sprintf(ERROR_STRING,"pop_item_context:  %s context %s popped",itp->it_name,icp->ic_name);
NADVISE(ERROR_STRING);
}
#endif /* DEBUG */
	/* Set the current context to the new top of stack */
	np=CONTEXT_LIST(itp)->l_head;
	if( np!=NO_NODE )
		CURRENT_CONTEXT(itp) = (Item_Context *) np->n_data;
	else
		CURRENT_CONTEXT(itp) = NO_ITEM_CONTEXT;

	itp->it_flags |= NEED_CHOICES|NEED_LIST;

	return(icp);
}

void delete_item_context( QSP_ARG_DECL  Item_Context *icp )
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
	 */

	itp = (Item_Type *) icp->ic_itp;

	while( (np=remData(CONTEXT_LIST(itp),icp)) != NO_NODE ){
		rls_node(np);
	}

	/* make the context be on the top of the stack */
	PUSH_ITEM_CONTEXT(itp,icp);

	if( itp->it_del_method == no_del_method ){
		sprintf(ERROR_STRING,
	"No object deletion method provided for item type %s",itp->it_name);
		WARN(ERROR_STRING);
	} else {
		List *lp;
		Node *np;

		lp=namespace_list(icp->ic_nsp);

		/* Don't use remHead to get the node, del_item()
		 * will remove it for us, and put it on the free list.
		 */
		while( lp!=NO_LIST && (np=lp->l_head)!=NO_NODE ){
			(*itp->it_del_method)(QSP_ARG  (Item*) np->n_data);
			/* force list update in case hashing */
			lp=namespace_list(icp->ic_nsp);
		}
	}

	delete_namespace(icp->ic_nsp);
	/* zap_hash_tbl(icp->ic_htp); */

	del_ctx(QSP_ARG  icp->ic_name);
	rls_str((char *)icp->ic_name);

	/* BUG? - perhaps we should make sure
	 * the context is not also pushed deeper down,
	 * to avoid a dangling ptr?
	 */

	/* Now pop the deleted context */

	pop_item_context(QSP_ARG  itp);
}


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

Item *item_of( QSP_ARG_DECL  Item_Type *itp, const char *name )
		/* itp = type of item to search for */
		/* name = name of item */
{
	Node *np;

#ifdef CAUTIOUS
	check_item_type( itp );
#endif /* CAUTIOUS */

	if( *name == 0 ) return(NO_ITEM);

	np=CONTEXT_LIST(itp)->l_head;

#ifdef CAUTIOUS
if(np==NO_NODE ){
sprintf(ERROR_STRING,"CAUTIOUS:  item type %s has no contexts pushed!?",itp->it_name);
NERROR1(ERROR_STRING);
}
#endif /* CAUTIOUS */

	/* check the top context first */

	while(np!=NO_NODE){
		Item_Context *icp;
		Item *ip;

		icp= (Item_Context*) np->n_data;
		ip=fetch_name(name,icp->ic_nsp);
		if( ip!=NO_ITEM ){
			return(ip);
		}
		if( CTX_RSTRCT_FLAG(itp) ){
			// We used to clear the flag here,
			// But now the place that depended on it,
			// (vectree/evaltree.c) has been changed to
			// explicitly clear it.

			//CTX_RSTRCT_FLAG(itp)=0;

			return(NO_ITEM);
		}
		np=np->n_next;
	}

	/* not found in any context, including default */

	return(NO_ITEM);
}

/*
 * Return a pointer to the item of the given type with the given name,
 * or a null pointer if not found.  Use of this routine
 * implies a belief that the named item does exist.
 */

Item *get_item( QSP_ARG_DECL  Item_Type *itp, const char* name )
		/* itp = type of item to search for */
		/* name = name of item */
{
	Item *ip;

	ip=item_of(QSP_ARG  itp,name);
	if( ip==NO_ITEM ){
#ifdef CAUTIOUS
		check_item_type( itp );
#endif /* CAUTIOUS */
		sprintf(ERROR_STRING,"no %s \"%s\"",
			itp->it_name,name);
		WARN(ERROR_STRING);
	}
	return(ip);
}

static const char **make_choices( QSP_ARG_DECL  int* countp, List* lp )
{
	const char **choices;
	int i;
	Item *ip;
	Node *np;

	*countp = eltcount(lp);
	if( *countp == 0 ){
		if( verbose ) WARN("empty list!?");
		return(NULL);
	}

	choices = (const char**) getbuf(*countp*sizeof(char *));

	if( choices == NULL ) NERROR1("make_choices:  out of memory");

	np=lp->l_head;
	i=0;
	while(np!=NO_NODE){
		ip = (Item*) np->n_data;
		choices[i++] = ip->item_name;
		np = np->n_next;
	}
	return(choices);
}

#ifdef HAVE_HISTORY

/* This function is used to initialize completion history for
 * nameof() for the cases where we cannot use pick_item(),
 * such as subscripted data objects, or when we want to allow
 * the user to enter "all" instead of an object name
 */

void init_item_hist( QSP_ARG_DECL  Item_Type *itp, const char* prompt )
{
	List *lp;

#ifdef CAUTIOUS
	if( itp == NO_ITEM_TYPE ){
		WARN("CAUTIOUS:  init_item_hist passed negative index");
		return;
	}
#endif /* CAUTIOUS */

	lp=item_list(QSP_ARG  itp);
	if( lp == NO_LIST ) return;
	init_hist_from_item_list(QSP_ARG  prompt,lp);
}
#endif /* HAVE_HISTORY */

static void setup_item_choices( QSP_ARG_DECL  Item_Type *itp )
{
	int count;

	if( itp->it_choices != NO_STR_ARRAY ){
		givbuf(itp->it_choices);
	}
	itp->it_choices = make_choices(QSP_ARG  &count,item_list(QSP_ARG  itp));
	itp->it_nchoices = count;
	itp->it_flags &= ~NEED_CHOICES;
}

/*
 * Use this function instead of get_xxx(nameof("item name"))
 */

Item *pick_item(QSP_ARG_DECL  Item_Type *itp,const char *prompt)
{
	int i;
	Item *ip;
	const char *s;

	/* use item type name as the prompt */

#ifdef CAUTIOUS
	if( itp == NO_ITEM_TYPE ){
		WARN("CAUTIOUS:  Uninitialized item type given to pick_item");
		s=NAMEOF("dummy");
		return(NO_ITEM);
	}
#endif /* CAUTIOUS */


	/* use the item type name as the prompt if unspecified */
	if( prompt == NULL || *prompt==0 )
		prompt=itp->it_name;

	if( NEEDS_NEW_CHOICES(itp) ){
		setup_item_choices(QSP_ARG  itp);
#ifdef HAVE_HISTORY

#ifdef DEBUG
if( debug & hist_debug ){
sprintf(ERROR_STRING,"resetting history list for %s items",itp->it_name);
NADVISE(ERROR_STRING);
}
#endif /* DEBUG */

		/*
		 * clear the old history list in case an item was deleted.
		 *
		 * this is a kind of BUG, since we lose the old item priorities...
		 * It might be better to have separate flags for additions
		 * and deletions...
		 */

		if( intractive(SINGLE_QSP_ARG) ){
			char pline[LLEN];
			sprintf(pline,PROMPT_FORMAT,prompt);
			new_defs(QSP_ARG  pline);
		}
#endif /* HAVE_HISTORY */
	}

	/* eat a meaningless word if there are no items */

	if( itp->it_nchoices <= 0 ){
		s=NAMEOF(prompt);	/* eat a word */
		sprintf(ERROR_STRING,"No %s %s (No items in existence)",
			itp->it_name,s);
		WARN(ERROR_STRING);
		return(NO_ITEM);
	}

	i = WHICH_ONE(prompt,itp->it_nchoices,itp->it_choices);

	if( i < 0 )
		return(NO_ITEM);

	s=itp->it_choices[i];
	ip=get_item(QSP_ARG  itp,s);

	return(ip);
}


static int item_cmp(CONST void* ipp1,CONST void* ipp2)
{
#ifdef __cplusplus
  return strcmp(  (*(reinterpret_cast<const Item* const *>(ipp1)))->item_name,
  		(*(reinterpret_cast<const Item* const *>(ipp2)))->item_name);
#else
  return( strcmp( (*(CONST Item * CONST *)ipp1)->item_name,
  		(*(CONST Item * CONST *)ipp2)->item_name) );
#endif
}


/*
 * Return a pointer to a list containing the items of the given type,
 * sorted in lexographic order by name.
 *
 * What do we do with contexts?  Do we include everything in the list?
 */

List *item_list(TMP_QSP_ARG_DECL  Item_Type *itp)
	/* type of items to list */
{
	Node *np;

#ifdef CAUTIOUS
	if( itp == NO_ITEM_TYPE ){
		WARN("CAUTIOUS:  item_list passed null item type pointer");
		return(NO_LIST);
	}
#endif /* CAUTIOUS */

	/* First check and see if any of the contexts have been updated */

	if( CONTEXT_LIST(itp) != NO_LIST ){
		Node *context_np;
		context_np=CONTEXT_LIST(itp)->l_head;
		while(context_np!=NO_NODE){
			Item_Context *icp;
			icp=(Item_Context *) context_np->n_data;
			/* We can't look just at the namespace LIST_IS_CURRENT
			 * flag, because if the name space is not using hashing,
			 * then its list is always current, although the context
			 * may have had items added or deleted.
			 */
			if( icp->ic_flags & CONTEXT_CHANGED )
				itp->it_flags |= CONTEXT_CHANGED;
			context_np=context_np->n_next;
		}
	}

	if( ! NEEDS_NEW_LIST(itp) ){
		/* Nothing changed, just return the existing list */
		return(itp->it_lp);
	}

	/* Something has changed, so we have to rebuild the list.
	 * Begin by trashing the old list.
	 */
	
	while( (np=remHead(itp->it_lp)) != NO_NODE )
		rls_node(np);

	/* now make up the new list, by concatenating the context lists */
	if( CONTEXT_LIST(itp) != NO_LIST ){
		Node *context_np;
		context_np=CONTEXT_LIST(itp)->l_head;
		while(context_np!=NO_NODE){
			Item_Context *icp;
			icp=(Item_Context *) context_np->n_data;
			cat_ns_items(itp->it_lp,icp->ic_nsp);
			context_np=context_np->n_next;
			icp->ic_flags &= ~CONTEXT_CHANGED;
		}
	}

	itp->it_flags &= ~(CONTEXT_CHANGED|NEED_LIST);

	return(itp->it_lp);
}

/* reorder a list of items into alphanumeric order of item names */
/* the caller must dispose of the list! */

List *alpha_sort(TMP_QSP_ARG_DECL  List *lp)
{
	short n2sort;
	void **ptr_array;
	Node *np;
	int i;


#ifdef CAUTIOUS
	if( lp == NO_LIST ){
		WARN("CAUTIOUS:  null list passed to alpha_sort");
		return(NO_LIST);
	}
#endif /* CAUTIOUS */

	n2sort=eltcount(lp);

	if( n2sort == 0 ) return(lp);

	ptr_array =(void**)  getbuf( n2sort * sizeof(void *) );

	if( ptr_array == NULL ) NERROR1("make_choices:  out of memory");

	np=lp->l_head;
	i=0;
	while(np!=NO_NODE){
		ptr_array[i++]=np->n_data;
		np=np->n_next;
	}

	/* now sort the pointers */
	qsort(ptr_array,(size_t)n2sort,(size_t)sizeof(char *),item_cmp);

	lp = new_list();
	for(i=0;i<n2sort;i++){
		np=mk_node(ptr_array[i]);
		addTail(lp,np);
	}
	givbuf(ptr_array);
	return(lp);
}

/* BUG this should be gotten from enviroment, termcap, something... */
#define CHARS_PER_LINE	78

/*
 * Print the names of all of the items of the given type to stdout
 */

void list_items(TMP_QSP_ARG_DECL  Item_Type *itp)
	/* type of items to list */
{
	List *lp;

	CHECK_ITEM_INDEX(itp)

	lp=item_list(QSP_ARG  itp);
	print_list_of_items(QSP_ARG  lp);
}

/* Sort the item list based on node priorities (set elsewhere) */

void sort_item_list(TMP_QSP_ARG_DECL  Item_Type *itp)
{
	List *lp;

	lp=item_list(QSP_ARG  itp);
	if( lp == NO_LIST ) return;

	p_sort(lp);
}

void print_list_of_items(TMP_QSP_ARG_DECL  List *lp)
{
	Node *np;
	int maxlen;
	int n_per_line;
	char fmtstr[16];
	int i, n_lines, n_total;
	FILE *out_fp;

	/* allocate an array of pointers for sorting */

	if(lp==NO_LIST) return;
	if( (n_total=eltcount(lp)) == 0 ) return;

	lp=alpha_sort(QSP_ARG  lp);
#ifdef CAUTIOUS
	if( lp == NO_LIST ){
		WARN("CAUTIOUS:  no items to print");
		return;
	}
#endif /* CAUTIOUS */

	/* If we are printing to the terminal, then
	 * we want as few lines as possible, but if we're
	 * printing to a file let's have 1 item per line
	 */

	out_fp = tell_msgfile();
	if( isatty( fileno(out_fp) ) ){
		/* find the maximum length */

		np=lp->l_head;
		maxlen=0;
		while(np!=NO_NODE){
			int l;
			if( (l=strlen( ((Item *)np->n_data)->item_name ) ) > maxlen )
				maxlen=l;
			np=np->n_next;
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

	/* print the names */

	for(i=0;i<n_lines;i++){
		int j;

		msg_str[0]=0;
		for(j=0; j< n_per_line; j++){
			int k;

			k=j*n_lines+i;
			if( k < n_total ){
				char tmp_str[LLEN];

				np = nth_elt(lp,k);
#ifdef CAUTIOUS
if( np==NO_NODE ) NERROR1("CAUTIOUS:  missing element!?");
if( strlen( ((Item *)np->n_data)->item_name ) > LLEN-1 )
NERROR1("item name too long");
#endif /* CAUTIOUS */
				sprintf(tmp_str,fmtstr,
					((Item *)np->n_data)->item_name);
				strcat(msg_str,tmp_str);
			}
		}
		prt_msg(msg_str);
	}

	/* now free the list */
	dellist(lp);
}


/*
 * Print the hash table statistics for a given class of item
 */

void item_stats(TMP_QSP_ARG_DECL  Item_Type * itp)
		/* type of item */
{
	Node *np;

#ifdef CAUTIOUS
	check_item_type( itp );
#endif /* CAUTIOUS */

	np = CONTEXT_LIST(itp)->l_head;
	while(np!=NO_NODE){
		Item_Context *icp;
		icp=(Item_Context *) np->n_data;
		sprintf(msg_str,"Context %s:",icp->ic_name);
		prt_msg(msg_str);
		tell_name_stats(icp->ic_nsp);
		np = np->n_next;
	}
}

/*
 * Delete the item pointed to by ip from the item database.
 * The actual storage for the item is not freed, but is
 * added to the item type's free list.  The caller must
 * free the stored name to prevent a memory leak.
 */

void del_item(TMP_QSP_ARG_DECL  Item_Type *itp,void* ip)
		/* itp = type of item to be deleted */
		/* ip = pointer to item to be deleted */
{
	LOCK_ITEM_TYPE(itp)

#ifdef CAUTIOUS
	check_item_type( itp );
#endif /* CAUTIOUS */

	zombie_item(QSP_ARG  itp,(Item*) ip);
	add_to_item_freelist(itp,ip);

	UNLOCK_ITEM_TYPE(itp)
}

void add_to_item_freelist(Item_Type *itp, void *ip)
{
	Node *np;

#ifdef CAUTIOUS
	check_item_type( itp );
#endif /* CAUTIOUS */

	np=mk_node(ip);
	addTail(itp->it_free,np);
}

/* Set the needy flag for this item type and any classes of which
 * it is a member.
 */

void make_needy(Item_Type *itp)
{
	itp->it_flags |= NEED_CHOICES | NEED_LIST;

	if( itp->it_classlist != NO_LIST ){
		Node *np;

		np = itp->it_classlist->l_head;
		while( np != NO_NODE ){
			Item_Class *iclp;

			iclp = (Item_Class*) np->n_data;
			iclp->icl_flags |= NEED_CLASS_CHOICES;
			np = np->n_next;
		}
	}
}

/* Remove an item from the item database, but do not return it to the
 * item free list.  This function was introduced to allow image objects
 * to be deleted after they have been displayed in viewers.  The
 * viewers retain a ptr to the data which they need to access in
 * order to refresh the window.
 *
 * This function could have a more descriptive name, like zombieize_item,
 * or hide_item.
 *
 * This is kind of a hack and has a big potential for memory leaks!?
 * It is used in the initial implementation of static objects in vectree,
 * but this is probably not a good idea...  BUG.
 */

void zombie_item(QSP_ARG_DECL  Item_Type *itp,Item* ip)
		/* itp = type of item to be deleted */
		/* ip = pointer to item to be deleted */
{
	Node *np;

#ifdef CAUTIOUS
	check_item_type( itp );
#endif /* CAUTIOUS */

	/* We used to remove the item from the item list here...
	 * but now with the namespace abstraction, we just remove
	 * it from the the namespace, and count on item_list()
	 * to detect when the list needs to be updated...
	 */

	np=CONTEXT_LIST(itp)->l_head;
	while(np!=NO_NODE){
		Item *tmp_ip;
		Item_Context *icp;

		icp = (Item_Context*) np->n_data;
		tmp_ip = fetch_name(((Item *)ip)->item_name,icp->ic_nsp);
		if( tmp_ip == ip ){	/* found it */
			if( remove_name(ip,icp->ic_nsp ) < 0 ){
				sprintf(ERROR_STRING,
		"zombie_item:  unable to remove %s item %s from context %s",
					itp->it_name,ip->item_name,icp->ic_name);
				WARN(ERROR_STRING);
				return;
			}
			/* BUG make the context needy... */
			/* does this do it? */
			icp->ic_flags |= CONTEXT_CHANGED;

			make_needy(itp);
			np=NO_NODE; /* or return? */
		} else
			np=np->n_next;
	}
}

/* Rename an item.
 *
 * This function frees the old name and allocates permanent
 * storage for the new name.
 */

void rename_item(TMP_QSP_ARG_DECL  Item_Type *itp,void *ip,char* newname)
		/* itp = type of item to be deleted */
{
	Node *np;

	LOCK_ITEM_TYPE(itp)

#ifdef CAUTIOUS
	check_item_type( itp );
#endif /* CAUTIOUS */

	zombie_item(QSP_ARG  itp,(Item*) ip);
	rls_str( (char *) ((Item *)ip)->item_name );
	((Item *)ip)->item_name = savestr(newname);
	np=mk_node(ip);
	store_item(QSP_ARG  itp,(Item*) ip,np);

	make_needy(itp);

	UNLOCK_ITEM_TYPE(itp)
}

static void dump_item_context(QSP_ARG_DECL  Item_Context *icp)
{
	List *lp;

	sprintf(msg_str,"\tContext \"%s\"",icp->ic_name);
	prt_msg(msg_str);

	dump_ns_info(icp->ic_nsp);

	lp=namespace_list(icp->ic_nsp);
	print_list_of_items(QSP_ARG  lp);
}

/*
 * Print a list of the names of the items of the given type,
 * preceded by the item type name and hash table size.
 */

void dump_item_type(TMP_QSP_ARG_DECL  Item_Type *itp)
{
	Node *np;

	sprintf(msg_str,
		"%s items",itp->it_name);
	prt_msg(msg_str);

	np = CONTEXT_LIST(itp)->l_head;
	while( np != NO_NODE ){
		Item_Context *icp;

		icp = (Item_Context*) np->n_data;
		dump_item_context(QSP_ARG  icp);

		np = np->n_next;
	}
}

/*
 * Dump all existing item types.
 */

void dump_items(TMP_SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;
	Item_Type *itp;

	lp = item_list(QSP_ARG  type_itp);
	if( lp == NO_LIST ) return;
	np=lp->l_head;
	while(np!=NO_NODE){
		itp=(Item_Type *) np->n_data;
		dump_item_type(QSP_ARG  itp);
		np=np->n_next;
	}
}

/* convert a string from mixed case to all lower case.
 * we do this allow case-insensitive matching.
 */

void decap(char* sto,const char* sfr)
{
	while(*sfr){

		/* use braces in case macro is multiple statements... */
		/* don't increment inside macro ... */
		/* superstitious pc behavior */


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
 */

List *find_items(TMP_QSP_ARG_DECL  Item_Type *itp,const char* frag)
{
	List *lp, *newlp=NO_LIST;
	Node *np, *newnp;
	Item *ip;
	char lc_frag[LLEN];

	lp=item_list(QSP_ARG  itp);
	if( lp == NO_LIST ) return(lp);

	np=lp->l_head;
	decap(lc_frag,frag);
	while(np!=NO_NODE){
		char str1[LLEN];
		ip = (Item*) np->n_data;
		/* make the match case insensitive */
		decap(str1,ip->item_name);
		if( strstr(str1,lc_frag) != NULL ){
			if( newlp == NO_LIST )
				newlp=new_list();
			newnp=mk_node(ip);
			addTail(newlp,newnp);
		}
		np=np->n_next;
	}
	return(newlp);
}

Item_Type *get_item_type(QSP_ARG_DECL  const char* name)
{
	return( (Item_Type *) get_item(QSP_ARG  type_itp,name) );
}

/*
 * Search all item types for items with matching names
 * BUG? this needs to be tested, may not work...
 */

List *find_all_items(TMP_QSP_ARG_DECL  const char* frag)
{
	List *lp, *newlp=NO_LIST;
	List *itlp;
	Node *itnp;

	itlp=item_list(QSP_ARG  type_itp);
	if( itlp == NO_LIST ) return(itlp);
	itnp=itlp->l_head;
	while(itnp!=NO_NODE){
		lp=find_items(QSP_ARG  (Item_Type *)itnp->n_data,frag);
		if( lp != NO_LIST ){
			if( newlp == NO_LIST )
				newlp=lp;
			else {
				Node *np;

				while( (np=remHead(lp)) != NO_NODE )
					addTail(newlp,np);
				rls_list(lp);
			}
		}
		itnp=itnp->n_next;
	}
	return(newlp);
}

#ifdef THREAD_SAFE_QUERY

void report_mutex_error(QSP_ARG_DECL  int status,const char *whence)
{
	const char *msg;

	sprintf(ERROR_STRING,"%s:  report_mutex_error:  status = %d",
		whence,status);
	ADVISE(ERROR_STRING);

	switch(status){
		case EINVAL: msg = "invalid argument"; break;
		case EBUSY: msg = "mutex already locked"; break;
		case EAGAIN: msg = "too many recursive locks"; break;
		case EDEADLK: msg = "lock already owned"; break;
		case EPERM: msg = "not mutex owner"; break;
		default: msg = "unexpected mutex error"; break;
	}
	WARN(msg);
}

#endif /* THREAD_SAFE_QUERY */
