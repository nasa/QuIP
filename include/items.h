
#ifndef ITEM_H
#define ITEM_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef THREAD_SAFE_QUERY
#include <pthread.h>
#include <errno.h>
#endif /* THREAD_SAFE_QUERY */

#include "savestr.h"
#include "node.h"
#include "namesp.h"
#include "typedefs.h"
#include "void.h"

/*
 *	what are items?
 *
 *	menus
 *	variables
 *	macros
 *	commands (one item list per menu)
 *	history lists (one item list per prompt)
 */

#define MAX_ITEM_TYPES	256

/* items themselves are defined in namesp.h */

typedef struct item_context {
	Item		ic_item;	/* Item is defined in namesp.h */
	Name_Space *	ic_nsp;
	int		ic_flags;	// what are the flag bits???
	void *		ic_itp;		/* really Item_Type *  */
} Item_Context;

/* #define ic_itp	ic_item.item_icp	 */

#define ic_name		ic_item.item_name

#define NO_ITEM_CONTEXT		((Item_Context *)NULL)

/* flag bits - see item type flags */

// We use TMP here because we haven't ready query.h yet... */
#ifdef THREAD_SAFE_QUERY

#define TMP_QSP_ARG_DECL	void *qsp,
#define TMP_SINGLE_QSP_ARG_DECL	void *qsp
#define MAX_QUERY_STREAMS	8



#define LOCK_ITEM_TYPE(itp)					\
								\
	if( n_active_threads > 1 )				\
	{							\
		int status;					\
								\
		status = pthread_mutex_lock(&itp->it_mutex);	\
		if( status != 0 ){				\
			char w[LLEN];				\
			sprintf(w,				\
		"LOCK_ITEM_TYPE(%s)",itp->it_name);		\
			report_mutex_error(qsp,status,w);	\
		}						\
		itp->it_flags |= CONTEXT_LOCKED;		\
	}

#define UNLOCK_ITEM_TYPE(itp)					\
								\
	if( ITEM_TYPE_IS_LOCKED(itp) )				\
	{							\
		int status;					\
								\
		itp->it_flags &= ~CONTEXT_LOCKED;		\
		status = pthread_mutex_unlock(&itp->it_mutex);	\
		if( status != 0 ){				\
			char w[LLEN];				\
			sprintf(w,				\
		"UNLOCK_ITEM_TYPE(%s)",itp->it_name);		\
			report_mutex_error(qsp,status,w);	\
		}						\
	}


#else /* ! THREAD_SAFE_QUERY */
#define TMP_QSP_ARG_DECL
#define TMP_SINGLE_QSP_ARG_DECL
#define LOCK_ITEM_TYPE(itp)
#define UNLOCK_ITEM_TYPE(itp)
#endif /* ! THREAD_SAFE_QUERY */

typedef struct it_type {
	Item		it_item;
	/* Name_Space *	it_nsp; */
	List *		it_lp;		/* list of all items */
	List *		it_free;
	const char **	it_choices;
	short		it_flags;
	short		it_nchoices;
	List *		it_classlist;	/* list of classes in which is member */
	struct it_type *it_context_itp;	/* the contexts are items... */
	void		(*it_del_method)(TMP_QSP_ARG_DECL  Item *);

#ifdef THREAD_SAFE_QUERY

	pthread_mutex_t	it_mutex;
	List *		it_contexts[MAX_QUERY_STREAMS];	/* list of contexts */
	Item_Context *	it_icp[MAX_QUERY_STREAMS];	// current context
	int 		it_ctx_restricted[MAX_QUERY_STREAMS];	// flags

#define CURRENT_CONTEXT(itp)	itp->it_icp[((Query_Stream *)qsp)->qs_serial]
#define CONTEXT_LIST(itp)	itp->it_contexts[((Query_Stream *)qsp)->qs_serial]
#define CTX_RSTRCT_FLAG(itp)	itp->it_ctx_restricted[((Query_Stream *)qsp)->qs_serial]
#define FIRST_CONTEXT(itp)	itp->it_icp[0]
#define FIRST_CONTEXT_LIST(itp)	itp->it_contexts[0]

#else /* ! THREAD_SAFE_QUERY */
	List *		it_contexts;	/* list of contexts */
	Item_Context *	it_icp;		// current context
	int 		it_ctx_restricted;	// flags

#define CURRENT_CONTEXT(itp)	itp->it_icp
#define CONTEXT_LIST(itp)	itp->it_contexts
#define FIRST_CONTEXT(itp)	itp->it_icp
#define FIRST_CONTEXT_LIST(itp)	itp->it_contexts
#define CTX_RSTRCT_FLAG(itp)	itp->it_ctx_restricted

#endif /* ! THREAD_SAFE_QUERY */

} Item_Type;

#define		it_name	it_item.item_name

#define NO_ITEM_TYPE	((Item_Type *)NULL)

/* flag bits - shared w/ contexts */
#define NEED_LIST		1
#define NEED_CHOICES		2
#define CONTEXT_CHANGED		8
#define HAVE_DEFAULTS		16
#define CONTEXT_LOCKED		32

#define RESTRICT_ITEM_CONTEXT(itp,flag)			\
							\
	CTX_RSTRCT_FLAG(itp)=flag;

/* #define LIST_IS_CURRENT(itp)						\
	(( ( itp ) ->it_flags & (NEED_LIST))==0) */

#define NEEDS_NEW_LIST(itp)						\
	( (itp) ->it_flags & (NEED_LIST|CONTEXT_CHANGED))

#define NEEDS_NEW_CHOICES(itp)	( ( itp )->it_flags & NEED_CHOICES)

#define CONTEXT_HAS_DEFAULTS(icp)	( ( icp )->ic_flags & HAVE_DEFAULTS)

#define ITEM_TYPE_IS_LOCKED(itp)	(itp->it_flags&CONTEXT_LOCKED)

#include "getbuf.h"

/*
 * Function declaration macros
 *
 * These macros provide a conveniant way to declare the basic
 * functions needed to declare a new item class.  E.g.
 *

DECL_INIT_FUNC(my_obj_init,my_itp,"MY_ITEM")
DECL_OF_FUNC(my_obj_of,My_Obj *,my_itp,my_obj_init)
DECL_GET_FUNC(get_my_obj,My_Obj *,my_itp,my_obj_init)
DECL_LIST_FUNC(list_my_objs,my_itp,my_obj_init)
DECL_NEW_FUNC(new_my_obj,My_Obj *,my_itp,my_obj_init)
DECL_DEL_FUNC(del_my_obj,My_Obj *,my_itp,get_my_obj)
DECL_PICK_FUNC(pick_my_obj,My_Obj *,my_itp,my_obj_init)

 *
 * Here are the corresponding prototypes:
 *

extern void my_obj_init(void);
extern My_Obj *my_obj_of(const char *name);
extern My_Obj *get_my_obj(const char *name);
extern void list_my_objs(void);
extern My_Obj *new_my_obj(const char *name);
extern My_Obj *del_my_obj(const char *name);
extern My_Obj *pick_my_obj(QSP_ARG_DECL const char *prompt);

 *
 *
 */

#define ITEM_INTERFACE_PROTOTYPES( type, string )			\
									\
extern void string##_init(SINGLE_QSP_ARG_DECL);				\
extern type * string##_of(QSP_ARG_DECL  const char *name);		\
extern type *get_##string(QSP_ARG_DECL  const char *name);		\
extern void list_##string##s(SINGLE_QSP_ARG_DECL);			\
extern type *new_##string(QSP_ARG_DECL  const char *name);		\
extern type *del_##string(QSP_ARG_DECL  const char *name);		\
extern type *pick_##string(QSP_ARG_DECL const char *prompt);



#define ITEM_INTERFACE_DECLARATIONS( type, string )			\
Item_Type * string##_itp = NO_ITEM_TYPE;				\
DECL_INIT_FUNC( string##_init, string##_itp, #type )			\
DECL_OF_FUNC(string##_of,type *,string##_itp,string##_init)		\
DECL_GET_FUNC(get_##string,type *,string##_itp,string##_init)		\
DECL_LIST_FUNC(list_##string##s,string##_itp,string##_init)		\
DECL_NEW_FUNC(new_##string,type *,string##_itp,string##_init)		\
DECL_DEL_FUNC(del_##string,type *,string##_itp,get_##string)		\
DECL_PICK_FUNC(pick_##string,type *,string##_itp,string##_init)


#define DECL_INIT_FUNC(name,itp,classname)				\
void name(SINGLE_QSP_ARG_DECL)						\
{									\
	if( itp != NO_ITEM_TYPE ){					\
sprintf(ERROR_STRING,"%s object class already initialized\n",classname);\
		WARN(ERROR_STRING);					\
		return;							\
	}								\
	itp = new_item_type(QSP_ARG  classname);			\
}

#define DECL_LIST_FUNC(name,itp,initfunc)				\
									\
void name(SINGLE_QSP_ARG_DECL)						\
{									\
	if( itp == NO_ITEM_TYPE ) initfunc(SINGLE_QSP_ARG);		\
	list_items(QSP_ARG  itp);					\
}


#define DECL_NEW_FUNC(func_name,type,itp,initfunc)			\
									\
type func_name(QSP_ARG_DECL  const char* item_name)			\
{									\
	type ptr;							\
									\
	if( itp == NO_ITEM_TYPE ) initfunc(SINGLE_QSP_ARG);		\
	ptr = (type) new_item( QSP_ARG  itp, item_name, sizeof(*ptr) );		\
	return(ptr);							\
}

#define DECL_PICK_FUNC(name,type,itp,initfunc)				\
									\
type name(QSP_ARG_DECL  const char* pmpt)				\
{									\
	type ptr;							\
									\
	if( itp == NO_ITEM_TYPE )	initfunc(SINGLE_QSP_ARG);	\
	ptr=(type)pick_item(QSP_ARG itp,pmpt);				\
	return(ptr);							\
}

#define DECL_OF_FUNC(name,type,itp,initfunc)				\
									\
type name(QSP_ARG_DECL  const char* s)					\
{									\
	if( itp == NO_ITEM_TYPE )	initfunc(SINGLE_QSP_ARG);	\
	return( ( type )item_of(QSP_ARG  itp,s) );			\
}

#define DECL_GET_FUNC(name,type,itp,initfunc)					\
type name(QSP_ARG_DECL  const char* s)						\
{										\
	if( itp == NO_ITEM_TYPE ) initfunc(SINGLE_QSP_ARG);			\
	return( ( type )get_item(QSP_ARG  itp,s) );				\
}

/* this del_func removes the item but doesn't free name or structure! */

#define DECL_DEL_FUNC(name,type,itp,get_name)				\
									\
type name(QSP_ARG_DECL  const char* s)					\
{									\
	type ptr;							\
									\
	ptr=get_name(QSP_ARG  s);					\
	if( ptr == ((type)NULL) ){					\
		sprintf(ERROR_STRING,					\
			"Can't delete \"%s\" (doesn't exist)\n",s);	\
		WARN(ERROR_STRING);					\
		return((type)NULL);					\
	}								\
	del_item(QSP_ARG  itp,ptr);					\
	return(ptr);							\
}





/* extern void		show_context_stack(Item_Type *); */
// moved to query.h
//extern void		delete_item_context(QSP_ARG_DECL  Item_Context *);
//extern Item_Context *	ctx_of(QSP_ARG_DECL  const char *);
//extern Item_Type * new_item_type(QSP_ARG_DECL  const char * atypename);
//extern Item *item_of(QSP_ARG_DECL  Item_Type * item_type,const char *name);
//extern Item *get_item(QSP_ARG_DECL  Item_Type * item_type,const char *name);
//extern void list_ittyps(SINGLE_QSP_ARG_DECL);
//extern void		list_ctxs(SINGLE_QSP_ARG_DECL);
//extern Item_Context *	create_item_context(QSP_ARG_DECL  Item_Type *,const char *);
//extern void init_item_hist(QSP_ARG_DECL  Item_Type * itp, const char *prompt);

extern void setup_all_item_type_contexts(TMP_QSP_ARG_DECL  void *new_qsp);

extern void add_item(TMP_QSP_ARG_DECL  Item_Type *,void *,Node *);
extern List *item_list(TMP_QSP_ARG_DECL  Item_Type * item_type);
extern List *alpha_sort(TMP_QSP_ARG_DECL  List *);
extern void sort_item_list(TMP_QSP_ARG_DECL  Item_Type *);
extern void list_items(TMP_QSP_ARG_DECL  Item_Type * item_type);
extern void print_list_of_items(TMP_QSP_ARG_DECL  List *lp);
extern void item_stats(TMP_QSP_ARG_DECL  Item_Type * item_type);
extern void del_item(TMP_QSP_ARG_DECL  Item_Type * item_type,void *ip);
extern void add_to_item_freelist(Item_Type * item_type,void *ip);
extern void make_needy(Item_Type *);
extern void rename_item(TMP_QSP_ARG_DECL  Item_Type * item_type, void *ip, char *newname);
extern void dump_item_type(TMP_QSP_ARG_DECL  Item_Type * type);
extern void dump_items(TMP_SINGLE_QSP_ARG_DECL);
extern List *		find_items(TMP_QSP_ARG_DECL  Item_Type *,const char *);
extern List *		find_all_items(TMP_QSP_ARG_DECL  const char *);

/*
extern void ittyp_init(void);
extern Item_Type *ittyp_of(char *name);
extern Item_Type *get_ittyp(char *name);
extern Item_Type *new_ittyp(char *name);
extern Item_Type *del_ittyp(char *name);
*/

extern void		decap(char *,const char *);

#ifdef __cplusplus
}
#endif

#endif /* ! ITEM_H */

