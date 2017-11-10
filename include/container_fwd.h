#ifndef _CONTAINER_FWD_H_
#define _CONTAINER_FWD_H_

#include "list.h"

typedef enum {
	LIST_CONTAINER_BIT,
	HASH_TBL_CONTAINER_BIT,
	RB_TREE_CONTAINER_BIT,
	N_CONTAINER_TYPES
} container_type_index;

#define LIST_CONTAINER		(1<<LIST_CONTAINER_BIT)
#define HASH_TBL_CONTAINER	(1<<HASH_TBL_CONTAINER_BIT)
#define RB_TREE_CONTAINER	(1<<RB_TREE_CONTAINER_BIT)

#define DEFAULT_CONTAINER_TYPE	HASH_TBL_CONTAINER

//// fwd declarations...
// now in quip_fwd.h
//struct container;
//typedef struct container Container;

extern long container_eltcount(Container *cnt_p);
extern void _cat_container_items(QSP_ARG_DECL  List *lp,Container *cnt_p);
#define cat_container_items(lp,cnt_p) _cat_container_items(QSP_ARG  lp,cnt_p)

extern void delete_container(Container *cnt_p);
extern void dump_container_info(QSP_ARG_DECL  Container *cnt_p);
extern Container * _create_container(QSP_ARG_DECL  const char *name,int type);
#define create_container(name,type) _create_container(QSP_ARG  name,type)
extern List *_container_list(QSP_ARG_DECL  Container *cnt_p);
#define container_list(cnt_p) _container_list(QSP_ARG  cnt_p)

extern Container * _new_container(QSP_ARG_DECL  int type);
#define new_container(type) _new_container(QSP_ARG  type)
extern int _add_to_container(QSP_ARG_DECL  Container *cnt_p, Item *ip);
#define add_to_container(cnt_p,ip) _add_to_container(QSP_ARG  cnt_p,ip)
extern int remove_name_from_container(QSP_ARG_DECL  Container *cnt_p, const char *name);
extern Item *_container_find_match(QSP_ARG_DECL  Container *cnt_p, const char *name);
#define container_find_match(cnt_p, name) _container_find_match(QSP_ARG  cnt_p, name)

//extern void set_container_primary(QSP_ARG_DECL  Container *cnt_p, int type);
extern void _set_container_type(QSP_ARG_DECL  Container *cnt_p,int type);
#define set_container_type(cnt_p,type) _set_container_type(QSP_ARG  cnt_p,type)

#endif // ! _CONTAINER_FWD_H_

