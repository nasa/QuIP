#ifndef _RBTREE_H_
#define _RBTREE_H_

//#define RB_TREE_DEBUG

//#include"misc.h"
//#include"stack.h"
#include "item_obj.h"

typedef enum {
	RBT_RED,
	RBT_BLACK
} rbnode_color;

typedef struct qrb_node {
//	const void * key;
	Item *			data;	// Item *
	rbnode_color		color;
	struct qrb_node *	left;
	struct qrb_node *	right;
	struct qrb_node *	parent;

#ifdef RB_TREE_DEBUG
	// these fields are for debugging consistency checks
	int			depth;
	int			black_depth;
	int			max_black_leaf;
	int			min_black_leaf;
#endif // RB_TREE_DEBUG

} qrb_node;

#define IS_BLACK(np)	( (np) == NULL || (np)->color == RBT_BLACK )
#define IS_RED(np)	( (np) != NULL && (np)->color == RBT_RED )
#define MAKE_BLACK(np)	(np)->color = RBT_BLACK
#define MAKE_RED(np)	(np)->color = RBT_RED
#define COPY_COLOR(dest_np,src_np)	(dest_np)->color = (src_np)->color
#define IS_ROOT_NODE(np)	((np)->parent == NULL)

// This assumes that all nodes point to Items
#define RB_NODE_KEY(np)	(((Item *)((np)->data))->item_name)
#define RB_NODE_ITEM(np)	((Item *)((np)->data))

/* (*comp_func)(a,b) should return 1 if *a > *b, -1 if *a < *b, and 0 otherwise */
/* Destroy(a) takes a pointer to whatever key might be and frees it accordingly */

// This used to just be called rb_tree, but it conflicts with sys/rbtree.h on Mac OS X
// There might be some advantage to adopting their interface and using their routines
// when available...

typedef struct qrb_tree {
	int (*comp_func)(const void* a, const void* b); 
	void (*key_destroy_func)(void* a);
	void (*data_destroy_func)(void* a);

	long		node_count;
	qrb_node* 	root;

	int		flags;
	List *		item_lp;
} qrb_tree;

// flags bits
#define RBT_LIST_IS_CURRENT	1


#define RB_TREE_ROOT(tp)	((tp)->root)
#define RB_TREE_ITEM_LIST(tp)	(tp)->item_lp
#define SET_RB_TREE_ITEM_LIST(tp,lp)	(tp)->item_lp = lp

#define RB_TREE_FLAGS(tp)	(tp)->flags
#define SET_RBT_FLAG_BITS(tp,bits)	(tp)->flags |= (bits)
#define CLEAR_RBT_FLAG_BITS(tp,bits)	(tp)->flags &= ~(bits)

#define RB_TREE_LIST_IS_CURRENT(tp)	(RB_TREE_FLAGS(tp) & RBT_LIST_IS_CURRENT)
#define MARK_RB_TREE_CURRENT(tp)	SET_RBT_FLAG_BITS(tp, RBT_LIST_IS_CURRENT)
#define MARK_RB_TREE_DIRTY(tp)		CLEAR_RBT_FLAG_BITS(tp, RBT_LIST_IS_CURRENT)

//extern qrb_tree* create_rb_tree(int  (*comp_func)(const void*, const void*),
//			     void (*key_destroy)(void*), 
//			     void (*data_destroy)(void*)
//			     );

extern qrb_tree* create_rb_tree(void);

extern qrb_node * rb_insert_item(qrb_tree*, Item * ip );
extern int rb_delete_key(qrb_tree*, const char *);
extern int rb_delete_named_item(qrb_tree*, const char *name);
extern int rb_delete_item(qrb_tree*, Item *ip);
extern qrb_node* rb_find(qrb_tree*, const char * key );
extern void rb_substring_find(Frag_Match_Info * fmi_p, qrb_tree*, const char * frag );
extern void rb_traverse( qrb_node *np, void (*func)(qrb_node *,qrb_tree *), qrb_tree *tree_p );
#ifdef RB_TREE_DEBUG
extern void rb_check(qrb_tree *);
#endif //  RB_TREE_DEBUG
extern qrb_node * rb_successor_node( qrb_node *n_p );
extern qrb_node * rb_predecessor_node( qrb_node *n_p );

//void RBTreePrint(qrb_tree*);

//void RBDelete(qrb_tree* , qrb_node* );
//void RBTreeDestroy(qrb_tree*);
//qrb_node* TreePredecessor(qrb_tree*,qrb_node*);
//qrb_node* TreeSuccessor(qrb_tree*,qrb_node*);

//stk_stack * RBEnumerate(qrb_tree* tree,void* low, void* high);
//void NullFunction(void*);

typedef struct {
	qrb_tree *	tree_p;
	qrb_node *	node_p;
} RB_Tree_Enumerator;

extern RB_Tree_Enumerator *new_rbtree_enumerator(qrb_tree *tp);
extern void advance_rbtree_enumerator(RB_Tree_Enumerator *rbtep);
extern void rls_rbtree_enumerator(RB_Tree_Enumerator *rbtep);
extern Item * rbtree_enumerator_item(RB_Tree_Enumerator *rbtep);
extern long rb_node_count(qrb_tree *tree_p);
extern void release_rb_tree(qrb_tree *tree_p);

extern List *rbtree_list(qrb_tree *tree_p);

#endif // ! _RBTREE_H_

