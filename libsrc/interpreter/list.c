
#include <string.h>
#include "quip_config.h"
#include "quip_prot.h"
#include "query_stack.h"
#include "list_private.h"


/*	the term "ring" refers to a circular list
	rings have head and tail pointers, which should
	point to adjacent elements, but the tail pts
	forward to the head and the head pts backward to
	the tail
 */

static List *free_node_list=NULL;
static List *free_list_list=NULL;
//static u_long node_debug=NODE_DEBUG_MASK;
#ifdef QUIP_DEBUG
//static u_long node_debug=0;
#endif // QUIP_DEBUG
static u_long total_nodes=0;

static Node * safe_remHead(List *lp);
static void safe_addTail(List *lp,Node *np);

int n_active_threads=0;

void report_node_data(SINGLE_QSP_ARG_DECL)
{
        sprintf(ERROR_STRING,"%ld nodes allocated",total_nodes);
        advise(ERROR_STRING);

        if( free_node_list == NULL )
                advise("no nodes freed");
        else {
                sprintf(ERROR_STRING,
                        "%d free nodes available",eltcount(free_node_list));
                advise(ERROR_STRING);
        }

        if( free_list_list == NULL )
                advise("no lists freed");
        else {
                sprintf(ERROR_STRING,
                        "%d free lists available",eltcount(free_list_list));
                advise(ERROR_STRING);
        }
}


/*
 * remove a node from a list
 * returns removed node or NULL if not in list
 */

Node *remNode( List * lp, Node* node )
		/* lp = nlist to be searched */
		/* node = node to be searched for */
{
	Node *np;
	int is_ring=0;

	np=_LIST_HEAD(lp);
	if( np == NULL ) return(NULL);

	LOCK_LIST(lp,remNode)

	if( NODE_PREV(np) != NULL ) is_ring=1;
	while( np != NULL ){
		if( np==node ){
			if( NODE_PREV(np) != NULL )
				SET_NODE_NEXT(NODE_PREV(np),NODE_NEXT(np));
			else SET_LIST_HEAD(lp, NODE_NEXT(np));
			if( NODE_NEXT(np) != NULL )
				SET_NODE_PREV(NODE_NEXT(np),NODE_PREV(np));
			else SET_LIST_TAIL(lp, NODE_PREV(np));

			/* the above doesn't work for rings!! */

			if( is_ring ){
				if( _LIST_HEAD(lp) == np ){
					if( _LIST_TAIL(lp) == np ){
						SET_LIST_HEAD(lp,NULL);
						SET_LIST_TAIL(lp,NULL);
					} else SET_LIST_HEAD(lp,NODE_NEXT(np));
				} else if( _LIST_TAIL(lp) == np ){
					SET_LIST_TAIL(lp, NODE_PREV(np));
				}
				SET_NODE_NEXT(np,np);
				SET_NODE_PREV(np,np);
			} else {
				SET_NODE_NEXT(np,NULL);
				SET_NODE_PREV(np,NULL);
			}

			UNLOCK_LIST(lp,remNode)
			return(np);
		}
		np = NODE_NEXT(np);
		if( np == _LIST_HEAD(lp) ) np=NULL;
	}

	NWARN("remNode:  node not found!?");	// can this ever happen?
	UNLOCK_LIST(lp,remNode)

	return(np);
} // remNode

/*
 * search list for node pointing to data
 * returns pointer to node or NULL
 */

Node *nodeOf( List *lp, void* data )
		/* list to be searched */
		/* data target node should point to */
{
	Node *np;

	np=_LIST_HEAD(lp);
	while( np != NULL ){
		if( np->n_data == data ) return(np);
		np=NODE_NEXT(np);
		if( np==_LIST_HEAD(lp) ) np=NULL;
	}
	return(NULL);
}

Node *remData( List *lp, void* data )	/** remove node pointing to data */
{
	Node *np, *stat;

	np=nodeOf( lp, data );
	if( np==NULL )
		return(NULL);
	stat=remNode( lp, np );
	assert( stat != NULL );

	return(np);
}

Node *nth_elt( List *lp , count_t n )	/** get ptr to nth node from head */
{
	Node *np, *np0;

    // count_t is unsigned?
	//if( n < 0 ) return(NULL);
	np0=np=_LIST_HEAD(lp);
	while( n-- ) {
		if( np==NULL )
			return( NULL );
		np=NODE_NEXT(np);
		if( np==np0 )
			return( NULL );
	}
	return( np );
}

Node *nth_elt_from_tail( List *lp , count_t n )
{
	Node *np, *np0;

	np0=np=_LIST_TAIL(lp);
	while( n-- ) {
		if( np==NULL )
			return( NULL );
		np=NODE_PREV(np);
		if( np==np0 )
			return( NULL );
	}
	return( np );
}

count_t eltcount( List * lp )	/** returns number of elements (for rings too) */
{
	int i=0;
	Node *np;
	Node *np0;

	if( lp==NULL ) return(0);

	np0=np=_LIST_HEAD(lp);

	while( np!=NULL ){
		i++;
		np=NODE_NEXT(np);
		if( np==np0 ) np=NULL;
	}
	return(i);
}

/*
 * Put node np on the list of free and available nodes
 * we used to test free_node_list here, but since it gets initialized
 * in newnodw(), we now no longer need to...
 */

void rls_node(Node *np)
{
#ifdef QUIP_DEBUG
//if( debug & node_debug ){
//sprintf(DEFAULT_ERROR_STRING,"releasing node 0x%lx",(u_long)np);
//NADVISE(DEFAULT_ERROR_STRING);
//}
#endif /* QUIP_DEBUG */

	addHead(free_node_list,np);
}

/*
 * Return a pointer to a node.
 * If there is one available on the free list, return it, otherwise
 * allocate memory for a new one.
 */

static Node *newnode()			/**/
{
	Node *np;

	// what if free_node_list doesn't exist???
	// BUG?  we might want to lock free_node_list,
	// but in practice this will be executed early,
	// before a second thread has been created...
	if( free_node_list == NULL )
		free_node_list = new_list();

	LOCK_LIST(free_node_list,newnode)

	if( IS_EMPTY(free_node_list) ){		/* grab another page's worth of nodes */
		int n_per_page;

		n_per_page = 4096 / sizeof(Node);	/* BUG use symbolic const */

#ifdef QUIP_DEBUG
//if( debug & node_debug ){
//sprintf(DEFAULT_ERROR_STRING,"allocating memory for %d more nodes (old total=%ld)",
//n_per_page,total_nodes);
//NADVISE(DEFAULT_ERROR_STRING);
//}
#endif /* QUIP_DEBUG */
		np = (Node *) malloc( n_per_page * sizeof(Node) );
		if( np == NULL ){
			sprintf(DEFAULT_ERROR_STRING,
				"no more memory for nodes (%ld allocated)",
				total_nodes);
			NERROR1(DEFAULT_ERROR_STRING);
		}
		total_nodes += n_per_page;
		while( n_per_page-- ) {
			/* Originally, we didn't initialize these nodes...
			 * That worked OK until we ran on the SUN
			 * under a peculiar set of circumstances...
			 */
			init_node(np,NULL);
			safe_addTail(free_node_list,np++);
		}
	}
	np=safe_remHead(free_node_list);

	UNLOCK_LIST(free_node_list,newnode)

	SET_NODE_NEXT(np,NULL);
	SET_NODE_PREV(np,NULL);
	np->n_pri = 0;
	np->n_data = NULL;
#ifdef QUIP_DEBUG
//if( debug & node_debug ){
//sprintf(DEFAULT_ERROR_STRING,"newnode:  np = 0x%lx",(u_long)np);
//NADVISE(DEFAULT_ERROR_STRING);
//}
#endif /* QUIP_DEBUG */
	return(np);
} // newnode

/*
 * Release all of the nodes belonging to a list,
 * and then release the list pointer.
 */

void dellist( List *lp )
{
	rls_nodes_from_list(lp);
	rls_list(lp);
}

void rls_nodes_from_list(List *lp)
{
	Node *np, *np2;

	LOCK_LIST(lp,rls_nodes_from_list)
	np=_LIST_HEAD(lp);
	while( np != NULL ){
		np2=np;
		np=NODE_NEXT(np2);
		rls_node(np2);
		if( np == _LIST_HEAD(lp) ) np=NULL;
	}
	SET_LIST_HEAD(lp,NULL);
	SET_LIST_TAIL(lp,NULL);
	UNLOCK_LIST(lp,rls_nodes_from_list)
}

static void init_list(List *lp)
{
//#ifdef CAUTIOUS
//	if( lp == NULL ){
//		NERROR1("CAUTIOUS:  init_list:  null list pointer!?");
//		IOS_RETURN
//	}
//#endif // CAUTIOUS

	assert( lp != NULL );
	
#ifdef THREAD_SAFE_QUERY
#ifdef HAVE_PTHREADS
	int status;
	pthread_mutexattr_t attr;

	if( pthread_mutexattr_init(&attr) != 0 )
		NWARN("error initializing mutex attributes!?");
	if( pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK) != 0 )
		NWARN("error setting mutex type!?");
	status = pthread_mutex_init(&lp->l_mutex,/*NULL*/&attr);
	if( status != 0 ){
		NERROR1("error initializing mutex!?");
		IOS_RETURN
	}
//fprintf(stderr,"mutex initialized for list at 0x%lx\n",(long)lp);
#endif /* HAVE_PTHREADS */

	lp->l_flags=0;
#endif /* THREAD_SAFE_QUERY */

	SET_LIST_HEAD(lp,NULL);
	SET_LIST_TAIL(lp,NULL);
}

/*
 * Return a pointer to a new list structure
 */

List *new_list()
{
	List *lp;

	if( free_list_list == NULL || IS_EMPTY(free_list_list) ){
		lp=(List*) getbuf(sizeof(*lp));
		//if( lp == NULL ) mem_err("new_list");
		if( lp == NULL ) NERROR1("new_list");
	} else {
		Node *np;

		np=remHead(free_list_list);
		lp=(List *)np->n_data;
		rls_node(np);
	}
	init_list(lp);
//fprintf(stderr,"new_list returning initialized list at 0x%lx\n",(long)lp);
	return(lp);
}

/*
 * Return a list pointer to the list of free list pointers
 */

void rls_list(List *lp)
{
	Node *np;

	if( free_list_list == NULL )
		free_list_list = new_list();
	np=mk_node(lp);
	addHead(free_list_list,np);
}

void addHead( List *lp, Node* np )		/**/
{
	assert(lp!=NULL);
	LOCK_LIST(lp,addHead)
	if( _LIST_HEAD(lp) != NULL ){
		if( NODE_PREV(_LIST_HEAD(lp)) != NULL ){	/* ring */
			SET_NODE_PREV(np, _LIST_TAIL(lp));
			SET_NODE_NEXT(_LIST_TAIL(lp), np);

		}
		SET_NODE_PREV(_LIST_HEAD(lp), np);
		SET_NODE_NEXT(np, _LIST_HEAD(lp));
	} else {
		/* don't initialize this (for rings)
		SET_NODE_NEXT(np, NULL);
		*/

		SET_LIST_TAIL(lp, np);
	}
	SET_LIST_HEAD(lp, np);
	UNLOCK_LIST(lp,addHead)
}

#define ADD_TAIL(lp,np)						\
								\
	if( _LIST_TAIL(lp) != NULL ){				\
		if( NODE_NEXT(_LIST_TAIL(lp)) != NULL ){		\
			SET_NODE_PREV(_LIST_HEAD(lp), np);		\
			SET_NODE_NEXT(np, _LIST_HEAD(lp));		\
		}						\
		SET_NODE_NEXT(_LIST_TAIL(lp), np);			\
		SET_NODE_PREV(np, _LIST_TAIL(lp));			\
	} else {						\
		SET_LIST_HEAD(lp, np);				\
	}							\
	SET_LIST_TAIL(lp, np);

void addTail( List *lp, Node* np )		/**/
{
	LOCK_LIST(lp,addTail)
	ADD_TAIL(lp,np)
	UNLOCK_LIST(lp,addTail)
}

void safe_addTail( List *lp, Node* np )		/**/
{
	ADD_TAIL(lp,np)
}

#define REM_HEAD(np,lp)							\
									\
	SET_LIST_HEAD(lp, NODE_NEXT(np));					\
	if( NODE_PREV(np) != NULL ){		/* ring */		\
		if( _LIST_HEAD(lp) == np ){	/* last node of ring list ? */	\
			SET_LIST_TAIL(lp,NULL);				\
			SET_LIST_HEAD(lp, NULL);		\
		} else {						\
			SET_NODE_NEXT(NODE_PREV(np), NODE_NEXT(np));		\
			SET_NODE_PREV(NODE_NEXT(np), NODE_PREV(np));		\
		}							\
									\
		/* keep it a ring link */				\
		SET_NODE_NEXT(np,np);						\
		SET_NODE_PREV(np,np);						\
	} else {							\
		if( NODE_NEXT(np) != NULL ){				\
			SET_NODE_PREV(NODE_NEXT(np), NULL);			\
		} else SET_LIST_TAIL(lp, NULL);				\
									\
		SET_NODE_NEXT(np, NULL);					\
		SET_NODE_PREV(np, NULL);					\
	}

Node *remHead( List *lp )		/**/
{
	Node *np;

	assert(lp!=NULL);
	if( (np=_LIST_HEAD(lp)) == NULL ){
		return( NULL );
	}
	LOCK_LIST(lp,remHead)
	REM_HEAD(np,lp)
	UNLOCK_LIST(lp,remHead)

	return(np);
}

Node *safe_remHead( List *lp )		/**/
{
	Node *np;

	if( (np=_LIST_HEAD(lp)) == NULL ){
		return( NULL );
	}

	REM_HEAD(np,lp)

	return(np);
}

Node *remTail( List *lp )		/**/
{
	Node *np;

	if( (np=_LIST_TAIL(lp)) == NULL ) return(NULL);
	LOCK_LIST(lp,remTail)
	SET_LIST_TAIL(lp, NODE_PREV(np));
	if( NODE_NEXT(np) != NULL ){		/* ring */
		if( _LIST_TAIL(lp) == np ){	/* last link of ring */
			SET_LIST_TAIL(lp,NULL);
			SET_LIST_HEAD(lp,NULL);
		} else {
			SET_NODE_PREV(NODE_NEXT(np), NODE_PREV(np));
			SET_NODE_NEXT(NODE_PREV(np), NODE_NEXT(np));
		}

		/* keep it a ring link */
		SET_NODE_NEXT(np,np);
		SET_NODE_PREV(np,np);
	} else {
		if( NODE_PREV(np) != NULL )		/* last node */
			SET_NODE_NEXT(NODE_PREV(np), NULL);
		else SET_LIST_HEAD(lp,NULL);

		SET_NODE_NEXT(np, NULL);
		SET_NODE_PREV(np, NULL);
	}
	UNLOCK_LIST(lp,remTail)
	return(np);
}

static void l_exch( List *lp, Node* np1, Node* np2 )		/** exchange two list elements */
{
	Node tmp;
	Node *tmp_np=(&tmp);

	LOCK_LIST(lp,l_exch)

	/* this procedure has to be different for adjacent nodes! */

	if( np1 == NODE_NEXT(np2) ){	/* np1 follows np2 */
		if( np2 != NODE_NEXT(np1) ){
			if( NODE_NEXT(np1) != NULL )
				SET_NODE_PREV(NODE_NEXT(np1),np2);
			if( NODE_PREV(np2) != NULL )
				SET_NODE_NEXT(NODE_PREV(np2),np1);
			SET_NODE_NEXT(np2,NODE_NEXT(np1));
			SET_NODE_PREV(np1,NODE_PREV(np2));
			SET_NODE_NEXT(np1,np2);
			SET_NODE_PREV(np2,np1);
		}
		/* else two element ring; do nothing 'cept fix head & tail */
	} else if( np2 == NODE_NEXT(np1) ){
		if( NODE_PREV(np1) != NULL )
			SET_NODE_NEXT(NODE_PREV(np1),np2);
		if( NODE_NEXT(np2) != NULL )
			SET_NODE_PREV(NODE_NEXT(np2),np1);
		SET_NODE_NEXT(np1,NODE_NEXT(np2));
		SET_NODE_PREV(np2,NODE_PREV(np1));
		SET_NODE_NEXT(np2,np1);
		SET_NODE_PREV(np1,np2);
	} else {
		if( NODE_NEXT(np1) != NULL )
			SET_NODE_PREV(NODE_NEXT(np1),np2);
		if( NODE_PREV(np1) != NULL )
			SET_NODE_NEXT(NODE_PREV(np1),np2);
		if( NODE_NEXT(np2) != NULL )
			SET_NODE_PREV(NODE_NEXT(np2),np1);
		if( NODE_PREV(np2) != NULL )
			SET_NODE_NEXT(NODE_PREV(np2),np1);

		memcpy(tmp_np,np1,sizeof(Node));

		SET_NODE_NEXT(np1, NODE_NEXT(np2));
		SET_NODE_PREV(np1, NODE_PREV(np2));

		SET_NODE_NEXT(np2, NODE_NEXT(tmp_np));
		SET_NODE_PREV(np2, NODE_PREV(tmp_np));
	}
	if( _LIST_HEAD(lp) == np1 ) SET_LIST_HEAD(lp,np2);
	else if( _LIST_HEAD(lp) == np2 ) SET_LIST_HEAD(lp,np1);
	
	if( _LIST_TAIL(lp) == np1 ) SET_LIST_TAIL(lp,np2);
	else if( _LIST_TAIL(lp) == np2 ) SET_LIST_TAIL(lp,np1);
	UNLOCK_LIST(lp,l_exch)
}
	
void p_sort( List* lp )		/** sort list with highest priority at head */
{
	int done=0;
	Node *np;

	/* bubble sort */

	if( eltcount(lp) < 2 ) return;

	while( !done ){
		done=1;
		np=_LIST_HEAD(lp);
		while( NODE_NEXT(np) != NULL && np!=_LIST_TAIL(lp) ){
			if( NODE_NEXT(np)->n_pri > np->n_pri ){
/*
sprintf(ERROR_STRING,"exchanging nodes w/ priorities %d, %d",
NODE_NEXT(np)->n_pri,np->n_pri);
NADVISE(ERROR_STRING);
*/
				l_exch( lp, np, NODE_NEXT(np) );
				done=0;
			} else {
/*
sprintf(ERROR_STRING,"leaving node w/ pri %d",np->n_pri);
NADVISE(ERROR_STRING);
*/
				np=NODE_NEXT(np);
			}
		}
	}
}

Node *mk_node(void* dp)	/** returns a node for a two-ended list */
{
	Node *np;

	np=newnode();
	if( np == NULL ) return(np);
	init_node(np,dp);
	return(np);
}

void init_node(Node *np,void* dp)
{
//#ifdef CAUTIOUS
//	if( np == NULL ){
//		NERROR1("CAUTIOUS:  init_node:  null node pointer!?");
//		IOS_RETURN
//	}
//#endif // CAUTIOUS

	assert( np != NULL );

	np->n_data=dp;
	SET_NODE_NEXT(np,NULL);
	SET_NODE_PREV(np,NULL);
	np->n_pri = 0;
}

void advance_list_enumerator(List_Enumerator *lep)
{
	if( lep->np == NULL ) return;
	if( lep->np == lep->lp->l_tail ){	// already at end?
		lep->np = NULL;
	} else {
		lep->np = NODE_NEXT(lep->np);
	}
}

Item *list_enumerator_item(List_Enumerator *lep)
{
	assert(lep!=NULL);
	if( lep->np == NULL ) return NULL;
	return (Item *) NODE_DATA(lep->np);
}

Node *list_find_named_item(List *lp, const char *name)
{
	Node *np;
	Item *ip;

	assert(lp!=NULL);

	np = _LIST_HEAD(lp);
	while( np != NULL ){	// BUG?  won't work for circular list!
		ip = NODE_DATA(np);
		if( !strcmp(name,ITEM_NAME(ip)) )
			return np;
		np = NODE_NEXT(np);
	}
	return NULL;
}

List_Enumerator *new_list_enumerator(List *lp)
{
	List_Enumerator *lep;

	if( _LIST_HEAD(lp) == NULL ) return NULL;

	lep = getbuf(sizeof(List_Enumerator));
	lep->lp = lp;	// needed?
	lep->np = _LIST_HEAD(lp);
	return lep;
}

void rls_list_enumerator(List_Enumerator *lep)
{
	givbuf(lep);	// keep a pool?
}

void rls_list_nodes(List *lp)
{
	Node *np;
	while( (np=remHead(lp)) != NULL )
		rls_node(np);
}


// release all the nodes in a list and the list too

void zap_list(List *lp)
{
	rls_list_nodes(lp);
	rls_list(lp);
}

Node *list_head(List *lp)
{
	return lp->l_head;
}

Node *list_tail(List *lp)
{
	return lp->l_tail;
}

