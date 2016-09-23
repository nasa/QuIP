#ifndef _ITEM_TYPE_H_
#define _ITEM_TYPE_H_

//#include "query_stack.h"
#include "stack.h"
#include "list.h"
#include "getbuf.h"
#include "qs_basic.h"
//#include "container.h"
//#include "dictionary.h"

#include "item_obj.h"

//struct dictionary;
// forward declarations...
struct container;
typedef struct container Container;

#include "dict.h"

#define NO_ITEM	((Item *) NULL)


typedef struct item_context {
	Item			ic_item;
	List *			ic_lp;
	Item_Type *		ic_itp;
	int			ic_flags;
	// We use a "dictionary" to store the items; traditionally,
	// this has been a hash table, but to support partial name
	// lookup (for more efficient completion), we might prefer
	// to use a red-black tree...
//	struct dictionary *	ic_dict_p;
	Container *		ic_cnt_p;
} Item_Context;


//#define NO_NAMESPACE		NULL

#define NO_ITEM_CONTEXT		((Item_Context *)NULL)

/* Item_Context */
#define CTX_NAME(icp)			(icp)->ic_item.item_name
//#define CTX_DICT(icp)			(icp)->ic_dict_p
#define CTX_CONTAINER(icp)		(icp)->ic_cnt_p
#define CTX_FLAGS(icp)			(icp)->ic_flags
#define CTX_IT(icp)			(icp)->ic_itp

#define SET_CTX_NAME(icp,s)		(icp)->ic_item.item_name = s
#define SET_CTX_IT(icp,itp)		(icp)->ic_itp = itp
#define SET_CTX_FLAGS(icp,f)		(icp)->ic_flags = f
//#define SET_CTX_DICT(icp,dict_p)	(icp)->ic_dict_p = dict_p
#define SET_CTX_CONTAINER(icp,cnt_p)	(icp)->ic_cnt_p = cnt_p
#define SET_CTX_FLAG_BITS(icp,f)	(icp)->ic_flags |= f
#define CLEAR_CTX_FLAG_BITS(icp,f)	(icp)->ic_flags &= ~(f)


#define MAX_QUERY_STACKS	5	// why do we need to have a limit?


struct item_type {
	Item		it_item;
	int		it_flags;
	List *		it_lp;
	List *		it_free_lp;
	int		it_default_container_type;
	void		(*it_del_method)(QSP_ARG_DECL  Item *);
	/*
	const char **	it_choices;
	int		it_n_choices;
	*/
	List *		it_class_lp;

	// If we can have multiple interpreter threads, then each thread
	// needs its own context stack...

#ifdef THREAD_SAFE_QUERY

#ifdef HAVE_PTHREADS
	pthread_mutex_t	it_mutex;
#endif /* HAVE_PTHREADS */
	Stack *		it_context_stack[MAX_QUERY_STACKS];	// need to have one per thread
	Item_Context *	it_icp[MAX_QUERY_STACKS];		// current context
	int 		it_ctx_restricted[MAX_QUERY_STACKS];	// flags

//#define FIRST_CONTEXT(itp)	itp->it_icp[0]

#else /* ! THREAD_SAFE_QUERY */
	Stack *		it_context_stack;
	Item_Context *	it_icp;			// current context
	int 		it_ctx_restricted;	// flags
#endif /* ! THREAD_SAFE_QUERY */

};

#define it_name	it_item.item_name
#define ITEM_TYPE_NAME(itp)	((itp)->it_name)

// flag bits
#define LIST_IS_CURRENT	1
#define NEED_CHOICES	2
#define NEED_LIST	4
#define RESTRICTED	8
#define CONTEXT_CHANGED	16


#define NO_ITEM_TYPE		((Item_Type *)NULL)


// BUG for thread-safe operation, this flag needs to be per-context stack!

#define RESTRICT_ITEM_CONTEXT(itp,yesno)		\
							\
{							\
	if( yesno )					\
		SET_IT_FLAG_BITS(itp,RESTRICTED);	\
	else						\
		CLEAR_IT_FLAG_BITS(itp,RESTRICTED);	\
}

#define IS_RESTRICTED(itp)	(IT_FLAGS(itp) & RESTRICTED)
#define NEEDS_NEW_CHOICES(itp)	(IT_FLAGS(itp) & NEED_CHOICES)
#define NEEDS_NEW_LIST(itp)	(IT_FLAGS(itp) & NEED_LIST)


/* Item_Type */
#define IT_NAME(itp)			(itp)->it_item.item_name
#define IT_FREE_LIST(itp)		(itp)->it_free_lp
#define IT_FLAGS(itp)			(itp)->it_flags
#define IT_CHOICES(itp)			(itp)->it_choices
#define IT_N_CHOICES(itp)		(itp)->it_n_choices
#define IT_LIST(itp)			(itp)->it_lp
#define IT_CLASS_LIST(itp)		(itp)->it_class_lp
#define IT_CONTAINER_TYPE(itp)		(itp)->it_default_container_type
#define SET_IT_FLAGS(itp,f)		(itp)->it_flags=f
#define SET_IT_FLAG_BITS(itp,f)		(itp)->it_flags |= f
#define CLEAR_IT_FLAG_BITS(itp,f)	(itp)->it_flags &= ~(f)
#define IT_DEL_METHOD(itp)		(itp)->it_del_method
#define SET_IT_DEL_METHOD(itp,f)	(itp)->it_del_method = f
#define SET_IT_LIST(itp,lp)		(itp)->it_lp = lp
#define SET_IT_FREE_LIST(itp,lp)	(itp)->it_free_lp = lp
#define SET_IT_CHOICES(itp,choices)	(itp)->it_choices = choices
#define SET_IT_N_CHOICES(itp,n)		(itp)->it_n_choices = n
#define SET_IT_CTX_IT(itp,citp)		(itp)->it_ctx_itp = citp
#define SET_IT_CLASS_LIST(itp,lp)	(itp)->it_class_lp = lp
#define SET_IT_CONTAINER_TYPE(itp,t)	(itp)->it_default_container_type = t


#ifdef THREAD_SAFE_QUERY

#define IT_CSTK_AT_IDX(itp,i)		(itp)->it_context_stack[i]
#define SET_IT_CSTK_AT_IDX(itp,i,sp)	(itp)->it_context_stack[i] = sp
#define SET_IT_CTX_RESTRICTED_AT_IDX(itp,i,flag)	\
					(itp)->it_ctx_restricted[i] = flag
#define THIS_CTX_STACK(itp)		((itp)->it_context_stack[((Query_Stack *)THIS_QSP)->qs_serial])

#define CURRENT_CONTEXT(itp)	(itp)->it_icp[((Query_Stack *)qsp)->qs_serial]
#define SET_CURRENT_CONTEXT(itp,icp)	(itp)->it_icp[((Query_Stack *)qsp)->qs_serial] = icp
#define CONTEXT_STACK(itp)	(itp)->it_context_stack[((Query_Stack *)qsp)->qs_serial]
#define CTX_RSTRCT_FLAG(itp)	(itp)->it_ctx_restricted[((Query_Stack *)qsp)->qs_serial]
#define FIRST_CONTEXT_STACK(itp)	(itp)->it_context_stack[0]


#else /* ! THREAD_SAFE_QUERY */

#define IT_CSTK(itp)			(itp)->it_context_stack
#define SET_IT_CSTK(itp,sp)		(itp)->it_context_stack = sp
#define SET_IT_CTX_RESTRICTED(itp,flag)	(itp)->it_ctx_restricted = flag
#define IT_CSTK_AT_IDX(itp,i)		IT_CSTK(itp)
#define SET_IT_CSTK_AT_IDX(itp,i,sp)	SET_IT_CSTK(itp,sp)
#define THIS_CTX_STACK(itp)		((itp)->it_context_stack)

#define CURRENT_CONTEXT(itp)	(itp)->it_icp
#define SET_CURRENT_CONTEXT(itp,icp)	(itp)->it_icp = icp
#define CONTEXT_STACK(itp)	(itp)->it_context_stack
#define FIRST_CONTEXT_STACK(itp)	(itp)->it_context_stack
#define CTX_RSTRCT_FLAG(itp)	(itp)->it_ctx_restricted
/*#define FIRST_CONTEXT(itp)	(itp)->it_icp */

#endif /* ! THREAD_SAFE_QUERY */


#define SET_IT_NAME(itp,s)		(itp)->it_item.item_name=s





typedef struct item_class {
	Item		icl_item;
	List *		icl_lp;		// list of itp's
	int		icl_flags;
} Item_Class;

#define NO_ITEM_CLASS	((Item_Class *)NULL)

// flag bits
#define NEED_CLASS_CHOICES	1

typedef struct member_info {
	Item_Type *	mi_itp;
	void *		mi_data;
	Item *		(*mi_lookup)(QSP_ARG_DECL  const char *);
} Member_Info;

#define MBR_DATA(mp)	(mp)->mi_data

#define NO_MEMBER_INFO	((Member_Info *)NULL)

#define ITEM_NEW_PROT(type,stem)	type * new_##stem(QSP_ARG_DECL  const char *name);
#define ITEM_INIT_PROT(type,stem)	void init_##stem##s(SINGLE_QSP_ARG_DECL );
#define ITEM_CHECK_PROT(type,stem)	type * stem##_of(QSP_ARG_DECL  const char *name);
#define ITEM_GET_PROT(type,stem)	type * get_##stem(QSP_ARG_DECL  const char *name);
#define ITEM_LIST_PROT(type,stem)	void  list_##stem##s(SINGLE_QSP_ARG_DECL );
#define ITEM_PICK_PROT(type,stem)	type *pick_##stem(QSP_ARG_DECL  const char *pmpt);
#define ITEM_ENUM_PROT(type,stem)	List *stem##_list(SINGLE_QSP_ARG_DECL);
#define ITEM_DEL_PROT(type,stem)	void del_##stem(QSP_ARG_DECL  type *ip);

#define ITEM_INTERFACE_PROTOTYPES(type,stem)	IIF_PROTS(type,stem,extern)
#define ITEM_INTERFACE_PROTOTYPES_STATIC(type,stem)		\
						IIF_PROTS(type,stem,static)

#define IIF_PROTS(type,stem,storage)				\
								\
storage ITEM_INIT_PROT(type,stem)				\
storage ITEM_NEW_PROT(type,stem)				\
storage ITEM_CHECK_PROT(type,stem)				\
storage ITEM_GET_PROT(type,stem)				\
storage ITEM_LIST_PROT(type,stem)				\
storage ITEM_ENUM_PROT(type,stem)				\
storage ITEM_DEL_PROT(type,stem)				\
storage ITEM_PICK_PROT(type,stem)

#define ITEM_INTERFACE_CONTAINER(stem,type)

#define ITEM_INTERFACE_DECLARATIONS(type,stem,container_type)	IIF_DECLS(type,stem,,container_type)

#define ITEM_INTERFACE_DECLARATIONS_STATIC(type,stem,container_type)		\
						IIF_DECLS(type,stem,static,container_type)

#define IIF_DECLS(type,stem,storage,container_type)		\
								\
static Item_Type *stem##_itp=NO_ITEM_TYPE;			\
storage ITEM_INIT_FUNC(type,stem,container_type)		\
storage ITEM_NEW_FUNC(type,stem)				\
storage ITEM_CHECK_FUNC(type,stem)				\
storage ITEM_GET_FUNC(type,stem)				\
storage ITEM_LIST_FUNC(type,stem)				\
storage ITEM_ENUM_FUNC(type,stem)				\
storage ITEM_DEL_FUNC(type,stem)				\
storage ITEM_PICK_FUNC(type,stem)


// BUG we should use new_item to take advantage of item free lists...

#define ITEM_NEW_FUNC(type,stem)				\
								\
type *new_##stem(QSP_ARG_DECL  const char *name)		\
{								\
	type * stem##_p;					\
								\
	if( stem##_itp == NO_ITEM_TYPE )			\
		init_##stem##s(SINGLE_QSP_ARG);			\
								\
	stem##_p = (type *) new_item(QSP_ARG  stem##_itp, name, \
					sizeof(type));		\
	if( stem##_p == NULL ){					\
		sprintf(ERROR_STRING,				\
	"Error creating item %s!?",name);			\
		WARN(ERROR_STRING);				\
		/* BUG release name here */			\
	}							\
	return stem##_p;					\
}

#define ITEM_INIT_FUNC(type,stem,container_type)		\
								\
void init_##stem##s(SINGLE_QSP_ARG_DECL)			\
{								\
	stem##_itp = new_item_type(QSP_ARG  #type);		\
	SET_IT_CONTAINER_TYPE(stem##_itp,container_type);	\
}

#define ITEM_CHECK_FUNC(type,stem)				\
								\
type *stem##_of(QSP_ARG_DECL  const char *name)			\
{								\
	if( stem##_itp == NO_ITEM_TYPE )			\
		init_##stem##s(SINGLE_QSP_ARG);			\
	return (type *)item_of(QSP_ARG  stem##_itp, name );	\
}

#define ITEM_GET_FUNC(type,stem)				\
								\
type *get_##stem(QSP_ARG_DECL  const char *name)		\
{								\
	if( stem##_itp == NO_ITEM_TYPE )			\
		init_##stem##s(SINGLE_QSP_ARG);			\
	return (type *)get_item(QSP_ARG  stem##_itp, name );	\
}

#define ITEM_PICK_FUNC(type,stem)				\
								\
type *pick_##stem(QSP_ARG_DECL  const char *pmpt)		\
{								\
	if( stem##_itp == NO_ITEM_TYPE )			\
		init_##stem##s(SINGLE_QSP_ARG);			\
	return (type *)pick_item(QSP_ARG  stem##_itp, pmpt );	\
}

#define ITEM_LIST_FUNC(type,stem)				\
								\
void list_##stem##s(SINGLE_QSP_ARG_DECL)			\
{								\
	if( stem##_itp == NO_ITEM_TYPE )			\
		init_##stem##s(SINGLE_QSP_ARG);			\
	list_items(QSP_ARG  stem##_itp );			\
}

#define ITEM_ENUM_FUNC(type,stem)				\
								\
List * stem##_list(SINGLE_QSP_ARG_DECL)				\
{								\
	if( stem##_itp == NO_ITEM_TYPE )			\
		init_##stem##s(SINGLE_QSP_ARG);			\
	return item_list(QSP_ARG  stem##_itp);			\
}

#define ITEM_DEL_FUNC(type,stem)				\
								\
void del_##stem(QSP_ARG_DECL  type *ip)				\
{								\
	del_item(QSP_ARG  stem##_itp, (Item *)ip);		\
}

extern ITEM_INIT_PROT(Item_Type,ittyp)
extern int add_item( QSP_ARG_DECL  Item_Type *itp, void *ip, Node *np );
extern Item *check_context(Item_Context *icp, const char *name);

#endif /* ! _ITEM_TYPE_H_ */

