#include "quip_config.h"

char VersionId_qutil_node[] = QUIP_VERSION_STRING;

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* malloc() */
#endif

#ifdef HAVE_STRING_H
#include <string.h>		/* memcpy() */
#endif

#include "query.h"		// ERROR_STRING
#include "node.h"
#include "getbuf.h"

/*	the term "ring" refers to a circular list
	rings have head and tail pointers, which should
	point to adjacent elements, but the tail pts
	forward to the head and the head pts backward to
	the tail
 */

static List *free_node_list=NO_LIST;
static List *free_list_list=NO_LIST;
static u_long node_debug=NODE_DEBUG_MASK;
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

void give_list_data(List *lp)
{
	Node *np;

	if( lp == NO_LIST ) return;

	np=lp->l_head;
	while( np != NO_NODE ){
		givbuf(np->n_data);
		np = np->n_next;
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

	np=lp->l_head;
	if( np == NO_NODE ) return(NO_NODE);

	LOCK_LIST(lp)

	if( np->n_last != NO_NODE ) is_ring=1;
	while( np != NO_NODE ){
		if( np==node ){
			if( np->n_last != NO_NODE )
				np->n_last->n_next=np->n_next;
			else lp->l_head = np->n_next;
			if( np->n_next != NO_NODE )
				np->n_next->n_last=np->n_last;
			else lp->l_tail = np->n_last;

			/* the above doesn't work for rings!! */

			if( is_ring ){
				if( lp->l_head == np ){
					if( lp->l_tail == np ){
						lp->l_head=
						lp->l_tail=NO_NODE;
					} else lp->l_head=np->n_next;
				} else if( lp->l_tail == np ){
					lp->l_tail = np->n_last;
				}
				np->n_next=np->n_last=np;
			} else np->n_next=np->n_last=NO_NODE;

			return(np);
		}
		np = np->n_next;
		if( np == lp->l_head ) np=NO_NODE;
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

	np=lp->l_head;
	while( np != NO_NODE ){
		if( np->n_data == data ) return(np);
		np=np->n_next;
		if( np==lp->l_head ) np=NO_NODE;
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
#ifdef CAUTIOUS
	if( stat == NO_NODE ){
		NWARN("CAUTIOUS:  remData:  remNode failed");
		return(NO_NODE);
	}
#endif /* CAUTIOUS */
	return(np);
}

count_t indexOf( List *lp, void* data )
{
	Node *np;
	int n=0;

	np=lp->l_head;
	while( np!=NO_NODE ){
		if( np->n_data == data ) return(n);
		n++;
		np=np->n_next;
		if( np == lp->l_head ) np=NO_NODE;
	}
	return(-1);
}

Node *nth_elt( List *lp , count_t n )	/** get ptr to nth node from head */
{
	Node *np, *np0;

	if( n < 0 ) return(NO_NODE);
	np0=np=lp->l_head;
	while( n-- ) {
		if( np==NO_NODE )
			return( NO_NODE );
		np=np->n_next;
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

	np0=np=lp->l_head;

	while( np!=NO_NODE ){
		i++;
		np=np->n_next;
		if( np==np0 ) np=NO_NODE;
	}
	return(i);
}

void *comdat( List *lp1, List* lp2 )	/** return ptr to data common to both lists */
{
	Node *np1, *np2;
	Node *l2h;

	np1=lp1->l_head;
	l2h=lp2->l_head;
	while( np1!=NO_NODE ){
		np2=l2h;
		while( np2 != NO_NODE ){
			if( np1->n_data == np2->n_data )
				return( np1->n_data );
			np2 = np2->n_next;
			if( np2==l2h ) np2=NO_NODE;
		}
		np1 = np1->n_next;
		if( np1 == lp1->l_head ) np1=NO_NODE;
	}
	return( NULL );
}

/*
 * Put node np on the list of free and available nodes
 * we used to test free_node_list here, but since it gets initialized
 * in newnodw(), we now no longer need to...
 */

void rls_node(Node *np)
{
#ifdef DEBUG
if( debug & node_debug ){
sprintf(DEFAULT_ERROR_STRING,"releasing node 0x%lx",(u_long)np);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */

	addHead(free_node_list,np);
}

/*
 * Return a pointer to a node.
 * If there is one available on the free list, return it, otherwise
 * allocate memory for a new one.
 */

Node *newnode()			/**/
{
	Node *np;

	// what if free_node_list doesn't exist???

	LOCK_LIST(free_node_list)

	if( IS_EMPTY(free_node_list) ){		/* grab another page's worth of nodes */
		int n_per_page;

		n_per_page = 4096 / sizeof(Node);	/* BUG use symbolic const */

#ifdef DEBUG
if( debug & node_debug ){
sprintf(DEFAULT_ERROR_STRING,"allocating memory for %d more nodes (old total=%ld)",
n_per_page,total_nodes);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */
		np = (Node *) malloc( n_per_page * sizeof(Node) );
		if( np == NO_NODE ){
			sprintf(DEFAULT_ERROR_STRING,
				"no more memory for nodes (%ld allocated)",
				total_nodes);
			NERROR1(DEFAULT_ERROR_STRING);
		}
		total_nodes += n_per_page;
		// BUG?  we might want to lock free_node_list,
		// but in practice this will be executed early,
		// before a second thread has been created...
		if( free_node_list == NO_LIST )
			free_node_list = new_list();
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

	np->n_next=np->n_last=NO_NODE;
	np->n_pri = 0;
	np->n_data = NULL;
#ifdef DEBUG
if( debug & node_debug ){
sprintf(DEFAULT_ERROR_STRING,"newnode:  np = 0x%lx",(u_long)np);
NADVISE(DEFAULT_ERROR_STRING);
}
#endif /* DEBUG */
	return(np);
}

/*
 * Release all of the nodes belonging to a list,
 * and then release the list pointer.
 */

void dellist( List *lp )
{
	Node *np, *np2;

	LOCK_LIST(lp)

	np=lp->l_head;
	while( np != NO_NODE ){
		np2=np;
		np=np2->n_next;
		rls_node(np2);
		if( np == lp->l_head ) np=NO_NODE;
	}
	rls_list(lp);

	UNLOCK_LIST(lp)
}

void init_list(List *lp)
{
#ifdef THREAD_SAFE_QUERY
	int status;

	status = pthread_mutex_init(&lp->l_mutex,NULL);
	if( status != 0 )
		NERROR1("error initializing mutex");
	lp->l_flags=0;
#endif /* THREAD_SAFE_QUERY */

	lp->l_head=lp->l_tail=NO_NODE;
}

/*
 * Return a pointer to a new list structure
 */

List *new_list()
{
	List *lp;

	if( IS_EMPTY(free_list_list) ){
		lp=(List*) getbuf(sizeof(*lp));
		if( lp == NO_LIST ) mem_err("new_list");
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

int insert( List* lp, Node* new_node, Node* listNode )	/** insert new_node after ListNode */
					/** ListNode==NULL -> addHead */
{
	Node *np;

	/* the Amiga manuals say to put at head if listNode == head!? */
	/* we don't do that here */

	if( listNode == NO_NODE )
		addHead( lp, new_node );
	else {
		LOCK_LIST(lp)
		np=lp->l_head;
		while( np!=NO_NODE ){
			if( np == listNode ){
				new_node->n_next = np->n_next;
				new_node->n_last = np;
				if( np->n_next != NO_NODE )
					np->n_next->n_last = new_node;
				np->n_next = new_node;
				if( np == lp->l_tail )
					lp->l_tail = new_node;
				UNLOCK_LIST(lp)
				return(0);
			}
			np=np->n_next;
			if( np==lp->l_head ) np=NO_NODE;
		}
		// I don't think we have ever seen this error!?
		// But maybe we don't use this function much?
		NWARN("insert:  predicate not on list");
		return(-1);
	}
	return(0);
}

void addHead( List *lp, Node* node )		/**/
{
	LOCK_LIST(lp)
	if( lp->l_head != NO_NODE ){
		if( lp->l_head->n_last != NO_NODE ){	/* ring */
			node->n_last = lp->l_tail;
			lp->l_tail->n_next = node;

		}
		lp->l_head->n_last = node;
		node->n_next = lp->l_head;
	} else {
		/* don't initialize this (for rings)
		node->n_next = NO_NODE;
		*/

		lp->l_tail = node;
	}
	lp->l_head = node;
	UNLOCK_LIST(lp)
}

#define ADD_TAIL(lp,np)						\
								\
	if( lp->l_tail != NO_NODE ){				\
		if( lp->l_tail->n_next != NO_NODE ){		\
			lp->l_head->n_last = np;		\
			np->n_next = lp->l_head;		\
		}						\
		lp->l_tail->n_next = np;			\
		np->n_last = lp->l_tail;			\
	} else {						\
		lp->l_head = np;				\
	}							\
	lp->l_tail = np;

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
	lp->l_head = np->n_next;					\
	if( np->n_last != NO_NODE ){		/* ring */		\
		if( lp->l_head == np ){	/* last node of ring list ? */	\
			lp->l_tail=lp->l_head = NO_NODE;		\
		} else {						\
			np->n_last->n_next = np->n_next;		\
			np->n_next->n_last = np->n_last;		\
		}							\
									\
		/* keep it a ring link */				\
		np->n_next=np;						\
		np->n_last=np;						\
	} else {							\
		if( np->n_next != NO_NODE )				\
			np->n_next->n_last = NO_NODE;			\
		else lp->l_tail = NO_NODE;				\
									\
		np->n_next = NO_NODE;					\
		np->n_last = NO_NODE;					\
	}

Node *remHead( List *lp )		/**/
{
	Node *np;

	if( (np=lp->l_head) == NO_NODE ){
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

	if( (np=lp->l_head) == NO_NODE ){
		return( NO_NODE );
	}

	REM_HEAD(np,lp)

	return(np);
}

Node *remTail( List *lp )		/**/
{
	Node *np;

	if( (np=lp->l_tail) == NO_NODE ) return(NO_NODE);
	LOCK_LIST(lp)
	lp->l_tail = np->n_last;
	if( np->n_next != NO_NODE ){		/* ring */
		if( lp->l_tail == np ){	/* last link of ring */
			lp->l_tail=lp->l_head=NO_NODE;
		} else {
			np->n_next->n_last = np->n_last;
			np->n_last->n_next = np->n_next;
		}

		/* keep it a ring link */
		np->n_next=np;
		np->n_last=np;
	} else {
		if( np->n_last != NO_NODE )		/* last node */
			np->n_last->n_next = NO_NODE;
		else lp->l_head=NO_NODE;

		np->n_next = NO_NODE;
		np->n_last = NO_NODE;
	}
	UNLOCK_LIST(lp)
	return(np);
}

void enqueue( List *lp, Node *node )		/* priority insertion */
{
	Node *np;

	np=lp->l_head;
	if( np==NO_NODE ){
		addHead( lp, node );
		return;
	}
	while( np != NO_NODE ){
		if( np->n_pri < node->n_pri ){
			insert( lp, node, np->n_last );
			return;
		}
		np = np->n_next;
		if( np==lp->l_head ) np=NO_NODE;
	}
	addTail( lp, node );
}

void l_exch( List *lp, Node* np1, Node* np2 )		/** exchange two list elements */
{
	Node tmp;

	LOCK_LIST(lp)

	/* this procedure has to be different for adjacent nodes! */

	if( np1 == np2->n_next ){	/* np1 follows np2 */
		if( np2 != np1->n_next ){
			if( np1->n_next != NO_NODE )
				np1->n_next->n_last=np2;
			if( np2->n_last != NO_NODE )
				np2->n_last->n_next=np1;
			np2->n_next=np1->n_next;
			np1->n_last=np2->n_last;
			np1->n_next=np2;
			np2->n_last=np1;
		}
		/* else two element ring; do nothing 'cept fix head & tail */
	} else if( np2 == np1->n_next ){
		if( np1->n_last != NO_NODE )
			np1->n_last->n_next=np2;
		if( np2->n_next != NO_NODE )
			np2->n_next->n_last=np1;
		np1->n_next=np2->n_next;
		np2->n_last=np1->n_last;
		np2->n_next=np1;
		np1->n_last=np2;
	} else {
		if( np1->n_next != NO_NODE )
			np1->n_next->n_last=np2;
		if( np1->n_last != NO_NODE )
			np1->n_last->n_next=np2;
		if( np2->n_next != NO_NODE )
			np2->n_next->n_last=np1;
		if( np2->n_last != NO_NODE )
			np2->n_last->n_next=np1;

		memcpy(&tmp,np1,sizeof(tmp));

		np1->n_next = np2->n_next;
		np1->n_last = np2->n_last;

		np2->n_next = tmp.n_next;
		np2->n_last = tmp.n_last;
	}
	if( lp->l_head == np1 ) lp->l_head=np2;
	else if( lp->l_head == np2 ) lp->l_head=np1;
	
	if( lp->l_tail == np1 ) lp->l_tail=np2;
	else if( lp->l_tail == np2 ) lp->l_tail=np1;
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
		np=lp->l_head;
		while( np->n_next != NO_NODE && np!=lp->l_tail ){
			if( np->n_next->n_pri > np->n_pri ){
/*
sprintf(ERROR_STRING,"exchanging nodes w/ priorities %d, %d",
np->n_next->n_pri,np->n_pri);
NADVISE(ERROR_STRING);
*/
				l_exch( lp, np, np->n_next );
				done=0;
			} else {
/*
sprintf(ERROR_STRING,"leaving node w/ pri %d",np->n_pri);
NADVISE(ERROR_STRING);
*/
				np=np->n_next;
			}
		}
	}
}

int in_list( List *lp, void* data )	/** is node pointing to data in list? */
{
	Node *np;

	np=nodeOf( lp, data );
	if( np==NO_NODE ) return(0);
	return(1);
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
	np->n_data=dp;
	np->n_next=NO_NODE;
	np->n_last=NO_NODE;
	np->n_pri = 0;
}

Node *mk_link(void*dp)	/** returns a link for a circular list */
{
	Node *np;

	np=newnode();
	if( np == NO_NODE ) return(np);
	np->n_data=dp;
	np->n_next=np;
	np->n_last=np;
	return(np);
}


