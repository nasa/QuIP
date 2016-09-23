#ifndef _CONTAINER_FWD_H_
#define _CONTAINER_FWD_H_

#include "list.h"

typedef enum {
	LIST_CONTAINER_BIT,
	HASH_TBL_CONTAINER_BIT,
	RB_TREE_CONTAINER_BIT,
	N_CONTAINER_TYPES
} container_type_index;

#define LIST_CONTAINER	(1<<LIST_CONTAINER_BIT)
#define HASH_TBL_CONTAINER	(1<<HASH_TBL_CONTAINER_BIT)
#define RB_TREE_CONTAINER	(1<<RB_TREE_CONTAINER_BIT)

#define DEFAULT_CONTAINER_TYPE	HASH_TBL_CONTAINER

// fwd declarations...
struct container;
typedef struct container Container;

extern long container_eltcount(Container *cnt_p);
extern void cat_container_items(List *lp,Container *cnt_p);
extern void delete_container(Container *cnt_p);
extern void dump_container_info(QSP_ARG_DECL  Container *cnt_p);
extern Container * create_container(const char *name,int type);
extern List *container_list(Container *cnt_p);
extern Container * new_container(int type);
extern int add_to_container(Container *cnt_p, Item *ip);
extern int remove_name_from_container(QSP_ARG_DECL  Container *cnt_p, const char *name);
extern Item *container_find_match(Container *cnt_p, const char *name);
extern Item *container_find_substring_match(QSP_ARG_DECL  Container *cnt_p, const char *frag);
//extern void set_container_primary(QSP_ARG_DECL  Container *cnt_p, int type);
extern void set_container_type(Container *cnt_p,int type);

#endif // ! _CONTAINER_FWD_H_

