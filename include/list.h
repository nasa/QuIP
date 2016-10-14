#ifndef _LIST_H_
#define _LIST_H_

#include "node.h"
#include "typedefs.h"
#include "item_obj.h"

#ifdef THREAD_SAFE_QUERY
extern int n_active_threads;	// Number of qsp's
#endif /* THREAD_SAFE_QUERY */

struct list {
	Node *		l_head;
	Node *		l_tail;
#ifdef THREAD_SAFE_QUERY
#ifdef HAVE_PTHREADS
	pthread_mutex_t	l_mutex;
	int		l_flags;	// Flags
#endif /* HAVE_PTHREADS */
#endif /* THREAD_SAFE_QUERY */
} ;

#define NEW_LIST		new_list()
#define INIT_LIST(lp)		{ lp->l_head=NO_NODE; lp->l_tail=NO_NODE; }
#define ALLOC_LIST		((List *)getbuf(sizeof(List)))

#ifdef THREAD_SAFE_QUERY

#define LIST_LOCKED	1

#ifdef HAVE_PTHREADS

#define LIST_IS_LOCKED(lp)	(lp->l_flags&LIST_LOCKED)

#define LOCK_LIST(lp)						\
								\
	if( n_active_threads > 1 )				\
	{							\
		int status;					\
								\
		status = pthread_mutex_lock(&lp->l_mutex);	\
		if( status != 0 )				\
			report_mutex_error(DEFAULT_QSP_ARG  status,"LOCK_LIST");	\
		lp->l_flags |= LIST_LOCKED;			\
	}

#define UNLOCK_LIST(lp)						\
								\
	if( LIST_IS_LOCKED(lp) )				\
	{							\
		int status;					\
								\
		lp->l_flags &= ~LIST_LOCKED;			\
		status = pthread_mutex_unlock(&lp->l_mutex);	\
		if( status != 0 )				\
			report_mutex_error(DEFAULT_QSP_ARG  status,"UNLOCK_LIST");\
	}

#else /* ! HAVE_PTHREADS */

#define LOCK_LIST(lp)
#define UNLOCK_LIST(lp)
#define LIST_IS_LOCKED(lp)	0

#endif /* ! HAVE_PTHREADS */

#else /* ! THREAD_SAFE_QUERY */

#define LOCK_LIST(lp)
#define UNLOCK_LIST(lp)

#endif /* ! THREAD_SAFE_QUERY */

#define NO_LIST		((List *)NULL)

/* sys/queue.h defines LIST_HEAD also!? */
#define QLIST_HEAD(lp)	lp->l_head
//#define LIST_HEAD(lp)	lp->l_head
#define QLIST_TAIL(lp)	lp->l_tail
#define SET_QLIST_HEAD(lp,np)	lp->l_head = np
#define SET_QLIST_TAIL(lp,np)	lp->l_tail = np

extern List *new_list(void);
extern count_t eltcount( List * lp );
extern Node *remHead(List *lp);
extern Node *remTail(List *lp);
extern void addHead(List *lp, Node *np);
extern Node * remNode(List *lp, Node *np);
extern Node *remData(List *lp, void * data);
extern void rls_list(List *lp);
extern void rls_nodes_from_list(List *lp);
extern void addTail(List *lp, Node *np);
extern void dellist(List *lp);
extern Node *nodeOf( List *lp, void * ip );
extern Node * list_find_named_item(List *lp, const char *name);

extern void p_sort(List *lp);
extern Node *nth_elt(List *lp, count_t k);
extern Node *nth_elt_from_tail(List *lp, count_t k);

#define IS_EMPTY(lp)	(QLIST_HEAD(lp)==NO_NODE)

typedef struct {
	List *lp;
	Node *np;
} List_Enumerator;

extern void advance_list_enumerator(List_Enumerator *lep);
extern Item *list_enumerator_item(List_Enumerator *lep);
extern List_Enumerator *new_list_enumerator(List *lp);

#endif /* ! _LIST_H_ */

