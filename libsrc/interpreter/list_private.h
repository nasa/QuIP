
#ifndef _LIST_PRIVATE_H_
#define _LIST_PRIVATE_H_

//#include "quip_fwd.h"
//#include "node.h"
//#include "typedefs.h"
//#include "item_obj.h"

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

#define INIT_LIST(lp)		{ lp->l_head=NULL; lp->l_tail=NULL; }
#define ALLOC_LIST		((List *)getbuf(sizeof(List)))

#ifdef THREAD_SAFE_QUERY

#define LIST_LOCKED_FLAG_BIT	1

#ifdef HAVE_PTHREADS

#define LIST_IS_LOCKED(lp)	(lp->l_flags&LIST_LOCKED_FLAG_BIT)

#define LOCK_LIST(lp,whence)						\
								\
	if( n_active_threads > 1 )				\
	{							\
		int status;					\
								\
		status = pthread_mutex_lock(&lp->l_mutex);	\
		if( status != 0 )				\
			report_mutex_error(DEFAULT_QSP_ARG  status,"LOCK_LIST");	\
		lp->l_flags |= LIST_LOCKED_FLAG_BIT;			\
	}

#define UNLOCK_LIST(lp,whence)						\
								\
	if( LIST_IS_LOCKED(lp) )				\
	{							\
		int status;					\
								\
		lp->l_flags &= ~LIST_LOCKED_FLAG_BIT;		\
		status = pthread_mutex_unlock(&lp->l_mutex);	\
		if( status != 0 )				\
			report_mutex_error(DEFAULT_QSP_ARG  status,"UNLOCK_LIST");\
	}

#else /* ! HAVE_PTHREADS */

#define LOCK_LIST(lp,whence)
#define UNLOCK_LIST(lp,whence)
#define LIST_IS_LOCKED(lp)	0

#endif /* ! HAVE_PTHREADS */

#else /* ! THREAD_SAFE_QUERY */

#define LOCK_LIST(lp,whence)
#define UNLOCK_LIST(lp,whence)

#endif /* ! THREAD_SAFE_QUERY */

/* sys/queue.h defines LIST_HEAD also!? */
#define _LIST_HEAD(lp)	lp->l_head
//#define LIST_HEAD(lp)	lp->l_head
#define _LIST_TAIL(lp)	lp->l_tail
#define SET_LIST_HEAD(lp,np)	lp->l_head = np
#define SET_LIST_TAIL(lp,np)	lp->l_tail = np

#define IS_EMPTY(lp)	(_LIST_HEAD(lp)==NULL)

struct list_enumerator {
	List *lp;
	Node *np;
};

#endif /* ! _LIST_PRIVATE_H_ */

