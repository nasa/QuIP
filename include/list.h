#ifndef _LIST_H_
#define _LIST_H_

#include "quip_fwd.h"
#include "node.h"
#include "typedefs.h"
#include "item_obj.h"

#ifdef FOOBAR
#define INIT_LIST(lp)		{ lp->l_head=NULL; lp->l_tail=NULL; }
#define ALLOC_LIST		((List *)getbuf(sizeof(List)))

#ifdef THREAD_SAFE_QUERY

#define LIST_LOCKED	1

#ifdef HAVE_PTHREADS

#define LIST_IS_LOCKED(lp)	(lp->l_flags&LIST_LOCKED)

#define LOCK_LIST(lp,whence)						\
								\
	if( n_active_threads > 1 )				\
	{							\
		int status;					\
fprintf(stderr,"LOCK_LIST  n_active_threads = %d\n",n_active_threads);\
								\
/*fprintf(stderr,"%s:  locking list 0x%lx\n",#whence,(long)lp);*/\
		status = pthread_mutex_lock(&lp->l_mutex);	\
		if( status != 0 )				\
			report_mutex_error(DEFAULT_QSP_ARG  status,"LOCK_LIST");	\
		lp->l_flags |= LIST_LOCKED;			\
/*fprintf(stderr,"%s:  list 0x%lx is locked\n",#whence,(long)lp);*/\
	}

#define UNLOCK_LIST(lp,whence)						\
								\
	if( LIST_IS_LOCKED(lp) )				\
	{							\
		int status;					\
								\
fprintf(stderr,"list at 0x%lx is locked, flags = 0x%x\n",(long)lp,lp->l_flags);\
		lp->l_flags &= ~LIST_LOCKED;			\
/*fprintf(stderr,"%s:  unlocking list 0x%lx\n",#whence,(long)lp);*/\
		status = pthread_mutex_unlock(&lp->l_mutex);	\
		if( status != 0 )				\
			report_mutex_error(DEFAULT_QSP_ARG  status,"UNLOCK_LIST");\
/*fprintf(stderr,"%s:  list 0x%lx is unlocked\n\n",#whence,(long)lp);*/\
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
#define QLIST_HEAD(lp)	lp->l_head
//#define LIST_HEAD(lp)	lp->l_head
#define QLIST_TAIL(lp)	lp->l_tail
#define SET_QLIST_HEAD(lp,np)	lp->l_head = np
#define SET_QLIST_TAIL(lp,np)	lp->l_tail = np

#define IS_EMPTY(lp)	(QLIST_HEAD(lp)==NULL)

#endif // FOOBAR

// list.c
extern void report_node_data(SINGLE_QSP_ARG_DECL);
extern count_t eltcount( List * lp );
#define NEW_LIST		new_list()
extern List *new_list(void);

extern void rls_list_nodes(List *lp);
extern void _zap_list(QSP_ARG_DECL List *lp);
extern Node *mk_node(void * ip );
#define zap_list(lp) QSP_ARG _zap_list(QSP_ARG lp)
extern Node *remHead(List *lp);
extern Node *remTail(List *lp);
extern Node * remNode(List *lp, Node *np);
extern Node *remData(List *lp, void * data);
extern void rls_list(List *lp);
//extern void _rls_list(QSP_ARG_DECL  List *lp);
//#define rls_list(lp) _rls_list(QSP_ARG  lp)

extern void rls_nodes_from_list(List *lp);
extern void addTail(List *lp, Node *np);
extern void addHead( List *lp, Node* np );
extern void _dellist(QSP_ARG_DECL  List *lp);
#define dellist(lp) _dellist(QSP_ARG  lp)

extern Node *nodeOf( List *lp, void * ip );
extern Node * list_find_named_item(List *lp, const char *name);

extern void p_sort(List *lp);
extern Node *nth_elt(List *lp, count_t k);
extern Node *nth_elt_from_tail(List *lp, count_t k);



extern void advance_list_enumerator(List_Enumerator *lep);
extern Item *list_enumerator_item(List_Enumerator *lep);
extern List_Enumerator *_new_list_enumerator(QSP_ARG_DECL  List *lp);
#define new_list_enumerator(lp) _new_list_enumerator(QSP_ARG  lp)

extern void rls_list_enumerator(List_Enumerator *lp);

extern Node *list_head(List *lp);
extern Node *list_tail(List *lp);
#define QLIST_HEAD(lp)	list_head(lp)
#define QLIST_TAIL(lp)	list_tail(lp)

#endif /* ! _LIST_H_ */

