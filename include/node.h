
#ifndef NO_NODE

#ifdef THREAD_SAFE_QUERY
#include <pthread.h>
#endif /* THREAD_SAFE_QUERY */

typedef struct node {
	struct node *	n_next;
	struct node *	n_last;
	short		n_pri;
	void *		n_data;
} Node;

#define NO_NODE	((Node *) NULL)

typedef struct {
	Node *		l_head;
	Node *		l_tail;
#ifdef THREAD_SAFE_QUERY
	pthread_mutex_t	l_mutex;
	int		l_flags;
#endif /* THREAD_SAFE_QUERY */
} List;

#ifdef THREAD_SAFE_QUERY

#define LIST_LOCKED	1
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

#else /* ! THREAD_SAFE_QUERY */

#define LOCK_LIST(lp)
#define UNLOCK_LIST(lp)

#endif /* ! THREAD_SAFE_QUERY */


#define NO_LIST	((List *) NULL)

#define IS_EMPTY( lp )	(( lp ) == NO_LIST || ( lp )->l_head == NO_NODE)

#include "typedefs.h"


extern void	give_list_data(List *);
extern Node *	remNode(List *list,Node *node);
extern Node *	nodeOf(List *list,void *data);
extern Node *	remData(List *list,void *data);
extern count_t	indexOf(List *list,void *data);
extern Node *	nth_elt(List *list,count_t n);
extern count_t	eltcount(List *list);
extern void *	comdat(List *list1,List *list2);
extern Node *	newnode(void);
extern void	dellist(List *list);
extern void	rls_node(Node *);
extern void	rls_list(List *);
extern List *	new_list(void);
extern void	init_list(List *lp);
extern int	insert(List *list,Node *new_node,Node *listNode);
extern void	addHead(List *list,Node *node);
extern void	addTail(List *list,Node *node);
extern Node *	remHead(List *list);
extern Node *	remTail(List *list);
extern void	enqueue(List *list,Node *node);
extern void	l_exch(List *list,Node *np1,Node *np2);
extern void	p_sort(List *list);
extern int	in_list(List *list,void *data);
extern Node *	mk_node(void *dp);
extern void	init_node(Node *np,void *dp);
extern Node *	mk_link(void *dp);


#endif /* NO_NODE */

