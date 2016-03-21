
#include "quip_config.h"
#include "quip_prot.h"

/*	the term "ring" refers to a circular list
	rings have head and tail pointers, which should
	point to adjacent elements, but the tail pts
	forward to the head and the head pts backward to
	the tail
 */

static List *free_node_list=NO_LIST;
static List *free_list_list=NO_LIST;
//static u_long node_debug=NODE_DEBUG_MASK;
#ifdef QUIP_DEBUG
//static u_long node_debug=0;
#endif // QUIP_DEBUG
static u_long total_nodes=0;

static Node * safe_remHead(List *lp);
static void safe_addTail(List *lp,Node *np);


void report_node_data(SINGLE_QSP_ARG_DECL)
{
        sprintf(ERROR_STRING,"%ld nodes allocated",total_nodes);
        ADVISE(ERROR_STRING);

        if( free_node_list == NO_LIST )
                ADVISE("no nodes freed");
        else {
                sprintf(ERROR_STRING,
                        "%d free nodes available",eltcount(free_node_list));
                ADVISE(ERROR_STRING);
        }

        if( free_list_list == NO_LIST )
                ADVISE("no lists freed");
        else {
                sprintf(ERROR_STRING,
                        "%d free lists available",eltcount(free_list_list));
                ADVISE(ERROR_STRING);
        }
}


/*
 * remove a node from a list
 * returns removed node or NO_NODE if not in list
 */

Node *remNode( List * lp, Node* node )
		/* lp = nlist to be searched */
		/* node = node to be searched for */
{
	Node *np;
	int is_ring=0;

	np=QLIST_HEAD(lp);
	if( np == NO_NODE ) return(NO_NODE);

	LOCK_LIST(lp)

	if( NODE_PREV(np) != NO_NODE ) is_ring=1;
	while( np != NO_NODE ){
		if( np==node ){
			if( NODE_PREV(np) != NO_NODE )
				SET_NODE_NEXT(NODE_PREV(np),NODE_NEXT(np));
			else SET_QLIST_HEAD(lp, NODE_NEXT(np));
			if( NODE_NEXT(np) != NO_NODE )
				SET_NODE_PREV(NODE_NEXT(np),NODE_PREV(np));
			else SET_QLIST_TAIL(lp, NODE_PREV(np));

			/* the above doesn't work for rings!! */

			if( is_ring ){
				if( QLIST_HEAD(lp) == np ){
					if( QLIST_TAIL(lp) == np ){
						SET_QLIST_HEAD(lp,NO_NODE);
						SET_QLIST_TAIL(lp,NO_NODE);
					} else SET_QLIST_HEAD(lp,NODE_NEXT(np));
				} else if( QLIST_TAIL(lp) == np ){
					SET_QLIST_TAIL(lp, NODE_PREV(np));
				}
				SET_NODE_NEXT(np,np);
				SET_NODE_PREV(np,np);
			} else {
				SET_NODE_NEXT(np,NO_NODE);
				SET_NODE_PREV(np,NO_NODE);
			}

			return(np);
		}
		np = NODE_NEXT(np);
		if( np == QLIST_HEAD(lp) ) np=NO_NODE;
	}

	UNLOCK_LIST(lp)

	return(np);
}

/*
 * search list for node pointing to data
 * returns pointer to node or NO_NODE
 */

Node *nodeOf( List *lp, void* data )
		/* list to be searched */
		/* data target node should point to */
{
	Node *np;

	np=QLIST_HEAD(lp);
	while( np != NO_NODE ){
		if( np->n_data == data ) return(np);
		np=NODE_NEXT(np);
		if( np==QLIST_HEAD(lp) ) np=NO_NODE;
	}
	return(NO_NODE);
}

Node *remData( List *lp, void* data )	/** remove node pointing to data */
{
	Node *np, *stat;

	np=nodeOf( lp, data );
	if( np==NO_NODE )
		return(NO_NODE);
	stat=remNode( lp, np );

//#ifdef CAUTIOUS
//	if( stat == NO_NODE ){
//		NWARN("CAUTIOUS:  remData:  remNode failed");
//		return(NO_NODE);
//	}
//#endif /* CAUTIOUS */

	assert( stat != NO_NODE );

	return(np);
}

Node *nth_elt( List *lp , count_t n )	/** get ptr to nth node from head */
{
	Node *np, *np0;

    // count_t is unsigned?
	//if( n < 0 ) return(NO_NODE);
	np0=np=QLIST_HEAD(lp);
	while( n-- ) {
		if( np==NO_NODE )
			return( NO_NODE );
		np=NODE_NEXT(np);
		if( np==np0 )
			return( NO_NODE );
	}
	return( np );
}

Node *nth_elt_from_tail( List *lp , count_t n )
{
	Node *np, *np0;

	np0=np=QLIST_TAIL(lp);
	while( n-- ) {
		if( np==NO_NODE )
			return( NO_NODE );
		np=NODE_PREV(np);
		if( np==np0 )
			return( NO_NODE );
	}
	return( np );
}

count_t eltcount( List * lp )	/** returns number of elements (for rings too) */
{
	int i=0;
	Node *np;
	Node *np0;

	if( lp==NO_LIST ) return(0);

	np0=np=QLIST_HEAD(lp);

	while( np!=NO_NODE ){
		i++;
		np=NODE_NEXT(np);
		if( np==np0 ) np=NO_NODE;
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
	if( free_node_list == NO_LIST )
		free_node_list = new_list();

	LOCK_LIST(free_node_list)

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
		if( np == NO_NODE ){
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

	UNLOCK_LIST(free_node_list)

	SET_NODE_NEXT(np,NO_NODE);
	SET_NODE_PREV(np,NO_NODE);
	np->n_pri = 0;
	np->n_data = NULL;
#ifdef QUIP_DEBUG
//if( debug & node_debug ){
//sprintf(DEFAULT_ERROR_STRING,"newnode:  np = 0x%lx",(u_long)np);
//NADVISE(DEFAULT_ERROR_STRING);
//}
#endif /* QUIP_DEBUG */
	return(np);
}

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

	LOCK_LIST(lp)
	np=QLIST_HEAD(lp);
	while( np != NO_NODE ){
		np2=np;
		np=NODE_NEXT(np2);
		rls_node(np2);
		if( np == QLIST_HEAD(lp) ) np=NO_NODE;
	}
	SET_QLIST_HEAD(lp,NO_NODE);
	SET_QLIST_TAIL(lp,NO_NODE);
	UNLOCK_LIST(lp)
}

static void init_list(List *lp)
{
//#ifdef CAUTIOUS
//	if( lp == NO_LIST ){
//		NERROR1("CAUTIOUS:  init_list:  null list pointer!?");
//		IOS_RETURN
//	}
//#endif // CAUTIOUS

	assert( lp != NO_LIST );
	
#ifdef THREAD_SAFE_QUERY
#ifdef HAVE_PTHREADS
	int status;

	status = pthread_mutex_init(&lp->l_mutex,NULL);
	if( status != 0 ){
		NERROR1("error initializing mutex");
		IOS_RETURN
	}
#endif /* HAVE_PTHREADS */

	//lp->l_flags=0;
#endif /* THREAD_SAFE_QUERY */

	SET_QLIST_HEAD(lp,NO_NODE);
	SET_QLIST_TAIL(lp,NO_NODE);
}

/*
 * Return a pointer to a new list structure
 */

List *new_list()
{
	List *lp;

	if( free_list_list == NO_LIST || IS_EMPTY(free_list_list) ){
		lp=(List*) getbuf(sizeof(*lp));
		//if( lp == NO_LIST ) mem_err("new_list");
		if( lp == NO_LIST ) NERROR1("new_list");
	} else {
		Node *np;

		np=remHead(free_list_list);
		lp=(List *)np->n_data;
		rls_node(np);
	}
	init_list(lp);
	return(lp);
}

/*
 * Return a list pointer to the list of free list pointers
 */

void rls_list(List *lp)
{
	Node *np;

	if( free_list_list == NO_LIST )
		free_list_list = new_list();
	np=mk_node(lp);
	addHead(free_list_list,np);
}

void addHead( List *lp, Node* node )		/**/
{
	LOCK_LIST(lp)
	if( QLIST_HEAD(lp) != NO_NODE ){
		if( NODE_PREV(QLIST_HEAD(lp)) != NO_NODE ){	/* ring */
			SET_NODE_PREV(node, QLIST_TAIL(lp));
			SET_NODE_NEXT(QLIST_TAIL(lp), node);

		}
		SET_NODE_PREV(QLIST_HEAD(lp), node);
		SET_NODE_NEXT(node, QLIST_HEAD(lp));
	} else {
		/* don't initialize this (for rings)
		SET_NODE_NEXT(node, NO_NODE);
		*/

		SET_QLIST_TAIL(lp, node);
	}
	SET_QLIST_HEAD(lp, node);
	UNLOCK_LIST(lp)
}

#define ADD_TAIL(lp,np)						\
								\
	if( QLIST_TAIL(lp) != NO_NODE ){				\
		if( NODE_NEXT(QLIST_TAIL(lp)) != NO_NODE ){		\
			SET_NODE_PREV(QLIST_HEAD(lp), np);		\
			SET_NODE_NEXT(np, QLIST_HEAD(lp));		\
		}						\
		SET_NODE_NEXT(QLIST_TAIL(lp), np);			\
		SET_NODE_PREV(np, QLIST_TAIL(lp));			\
	} else {						\
		SET_QLIST_HEAD(lp, np);				\
	}							\
	SET_QLIST_TAIL(lp, np);

void addTail( List *lp, Node* np )		/**/
{
	LOCK_LIST(lp)
	ADD_TAIL(lp,np)
	UNLOCK_LIST(lp)
}

void safe_addTail( List *lp, Node* np )		/**/
{
	ADD_TAIL(lp,np)
}

#define REM_HEAD(np,lp)							\
									\
	SET_QLIST_HEAD(lp, NODE_NEXT(np));					\
	if( NODE_PREV(np) != NO_NODE ){		/* ring */		\
		if( QLIST_HEAD(lp) == np ){	/* last node of ring list ? */	\
			SET_QLIST_TAIL(lp,NO_NODE);				\
			SET_QLIST_HEAD(lp, NO_NODE);		\
		} else {						\
			SET_NODE_NEXT(NODE_PREV(np), NODE_NEXT(np));		\
			SET_NODE_PREV(NODE_NEXT(np), NODE_PREV(np));		\
		}							\
									\
		/* keep it a ring link */				\
		SET_NODE_NEXT(np,np);						\
		SET_NODE_PREV(np,np);						\
	} else {							\
		if( NODE_NEXT(np) != NO_NODE )				\
			SET_NODE_PREV(NODE_NEXT(np), NO_NODE);			\
		else SET_QLIST_TAIL(lp, NO_NODE);				\
									\
		SET_NODE_NEXT(np, NO_NODE);					\
		SET_NODE_PREV(np, NO_NODE);					\
	}

Node *remHead( List *lp )		/**/
{
	Node *np;

	if( (np=QLIST_HEAD(lp)) == NO_NODE ){
		return( NO_NODE );
	}

	LOCK_LIST(lp)
	REM_HEAD(np,lp)
	UNLOCK_LIST(lp)

	return(np);
}

Node *safe_remHead( List *lp )		/**/
{
	Node *np;

	if( (np=QLIST_HEAD(lp)) == NO_NODE ){
		return( NO_NODE );
	}

	REM_HEAD(np,lp)

	return(np);
}

Node *remTail( List *lp )		/**/
{
	Node *np;

	if( (np=QLIST_TAIL(lp)) == NO_NODE ) return(NO_NODE);
	LOCK_LIST(lp)
	SET_QLIST_TAIL(lp, NODE_PREV(np));
	if( NODE_NEXT(np) != NO_NODE ){		/* ring */
		if( QLIST_TAIL(lp) == np ){	/* last link of ring */
			SET_QLIST_TAIL(lp,NO_NODE);
			SET_QLIST_HEAD(lp,NO_NODE);
		} else {
			SET_NODE_PREV(NODE_NEXT(np), NODE_PREV(np));
			SET_NODE_NEXT(NODE_PREV(np), NODE_NEXT(np));
		}

		/* keep it a ring link */
		SET_NODE_NEXT(np,np);
		SET_NODE_PREV(np,np);
	} else {
		if( NODE_PREV(np) != NO_NODE )		/* last node */
			SET_NODE_NEXT(NODE_PREV(np), NO_NODE);
		else SET_QLIST_HEAD(lp,NO_NODE);

		SET_NODE_NEXT(np, NO_NODE);
		SET_NODE_PREV(np, NO_NODE);
	}
	UNLOCK_LIST(lp)
	return(np);
}

static void l_exch( List *lp, Node* np1, Node* np2 )		/** exchange two list elements */
{
	Node tmp;
	Node *tmp_np=(&tmp);

	LOCK_LIST(lp)

	/* this procedure has to be different for adjacent nodes! */

	if( np1 == NODE_NEXT(np2) ){	/* np1 follows np2 */
		if( np2 != NODE_NEXT(np1) ){
			if( NODE_NEXT(np1) != NO_NODE )
				SET_NODE_PREV(NODE_NEXT(np1),np2);
			if( NODE_PREV(np2) != NO_NODE )
				SET_NODE_NEXT(NODE_PREV(np2),np1);
			SET_NODE_NEXT(np2,NODE_NEXT(np1));
			SET_NODE_PREV(np1,NODE_PREV(np2));
			SET_NODE_NEXT(np1,np2);
			SET_NODE_PREV(np2,np1);
		}
		/* else two element ring; do nothing 'cept fix head & tail */
	} else if( np2 == NODE_NEXT(np1) ){
		if( NODE_PREV(np1) != NO_NODE )
			SET_NODE_NEXT(NODE_PREV(np1),np2);
		if( NODE_NEXT(np2) != NO_NODE )
			SET_NODE_PREV(NODE_NEXT(np2),np1);
		SET_NODE_NEXT(np1,NODE_NEXT(np2));
		SET_NODE_PREV(np2,NODE_PREV(np1));
		SET_NODE_NEXT(np2,np1);
		SET_NODE_PREV(np1,np2);
	} else {
		if( NODE_NEXT(np1) != NO_NODE )
			SET_NODE_PREV(NODE_NEXT(np1),np2);
		if( NODE_PREV(np1) != NO_NODE )
			SET_NODE_NEXT(NODE_PREV(np1),np2);
		if( NODE_NEXT(np2) != NO_NODE )
			SET_NODE_PREV(NODE_NEXT(np2),np1);
		if( NODE_PREV(np2) != NO_NODE )
			SET_NODE_NEXT(NODE_PREV(np2),np1);

		memcpy(tmp_np,np1,sizeof(Node));

		SET_NODE_NEXT(np1, NODE_NEXT(np2));
		SET_NODE_PREV(np1, NODE_PREV(np2));

		SET_NODE_NEXT(np2, NODE_NEXT(tmp_np));
		SET_NODE_PREV(np2, NODE_PREV(tmp_np));
	}
	if( QLIST_HEAD(lp) == np1 ) SET_QLIST_HEAD(lp,np2);
	else if( QLIST_HEAD(lp) == np2 ) SET_QLIST_HEAD(lp,np1);
	
	if( QLIST_TAIL(lp) == np1 ) SET_QLIST_TAIL(lp,np2);
	else if( QLIST_TAIL(lp) == np2 ) SET_QLIST_TAIL(lp,np1);
	UNLOCK_LIST(lp)
}
	
void p_sort( List* lp )		/** sort list with highest priority at head */
{
	int done=0;
	Node *np;

	/* bubble sort */

	if( eltcount(lp) < 2 ) return;

	while( !done ){
		done=1;
		np=QLIST_HEAD(lp);
		while( NODE_NEXT(np) != NO_NODE && np!=QLIST_TAIL(lp) ){
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
	if( np == NO_NODE ) return(np);
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
	SET_NODE_NEXT(np,NO_NODE);
	SET_NODE_PREV(np,NO_NODE);
	np->n_pri = 0;
}

