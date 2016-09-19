#ifndef _CONTAINER_H_
#define _CONTAINER_H_

// implementation-agnostic interface

#include "list.h"
#include "hash.h"
#include "rbtree.h"
#include "item_type.h"

typedef enum {
	LIST_CONTAINER,
	HASH_TBL_CONTAINER,
	RB_TREE_CONTAINER,
	N_CONTAINER_TYPES
} container_type_code;

typedef struct container {
	container_type_code	type;
	union {
		List *		cnt_lp;
		Hash_Tbl *	cnt_htp;
		rb_tree *	cnt_tree_p;
	} ptr;
} Container;

extern Container * new_container(QSP_ARG_DECL  container_type_code type);
extern void add_to_container(QSP_ARG_DECL  Container *cnt_p, Item *ip);
extern void remove_from_container(QSP_ARG_DECL  Container *cnt_p, const char *name);
extern Item *container_find_match(QSP_ARG_DECL  Container *cnt_p, const char *name);
extern Item *container_find_substring_match(QSP_ARG_DECL  Container *cnt_p, const char *frag);

#endif // ! _CONTAINER_H_

