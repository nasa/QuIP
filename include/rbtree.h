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

typedef struct rb_node {
//	const void * key;
	Item *			data;	// Item *
	rbnode_color		color;
	struct rb_node *	left;
	struct rb_node *	right;
	struct rb_node *	parent;

#ifdef RB_TREE_DEBUG
	// these fields are for debugging consistency checks
	int			depth;
	int			black_depth;
	int			max_black_leaf;
	int			min_black_leaf;
#endif // RB_TREE_DEBUG

} rb_node;

#define IS_BLACK(np)	( (np) == NULL || (np)->color == RBT_BLACK )
#define IS_RED(np)	( (np) != NULL && (np)->color == RBT_RED )
#define MAKE_BLACK(np)	(np)->color = RBT_BLACK
#define MAKE_RED(np)	(np)->color = RBT_RED
#define COPY_COLOR(dest_np,src_np)	(dest_np)->color = (src_np)->color
#define IS_ROOT_NODE(np)	((np)->parent == NULL)

// This assumes that all nodes point to Items
#define RB_NODE_KEY(np)	(((Item *)((np)->data))->item_name)

/* (*comp_func)(a,b) should return 1 if *a > *b, -1 if *a < *b, and 0 otherwise */
/* Destroy(a) takes a pointer to whatever key might be and frees it accordingly */
typedef struct rb_tree {
	int (*comp_func)(const void* a, const void* b); 
	void (*key_destroy_func)(void* a);
	void (*data_destroy_func)(void* a);

	long	node_count;
	rb_node* root;             
} rb_tree;

#define RB_TREE_ROOT(tp)	((tp)->root)

//extern rb_tree* create_rb_tree(int  (*comp_func)(const void*, const void*),
//			     void (*key_destroy)(void*), 
//			     void (*data_destroy)(void*)
//			     );

extern rb_tree* create_rb_tree(void);

extern rb_node * rb_insert_item(rb_tree*, Item * ip );
extern int rb_delete_key(rb_tree*, const char *);
extern int rb_delete_named_item(rb_tree*, const char *name);
extern int rb_delete_item(rb_tree*, Item *ip);
extern rb_node* rb_find(rb_tree*, const char * key );
extern void rb_substring_find(Frag_Match_Info * fmi_p, rb_tree*, const char * frag );
extern void rb_traverse( rb_node *np, void (*func)(rb_node *) );
#ifdef RB_TREE_DEBUG
extern void rb_check(rb_tree *);
#endif //  RB_TREE_DEBUG
extern rb_node * rb_successor_node( rb_node *n_p );
extern rb_node * rb_predecessor_node( rb_node *n_p );

//void RBTreePrint(rb_tree*);

//void RBDelete(rb_tree* , rb_node* );
//void RBTreeDestroy(rb_tree*);
//rb_node* TreePredecessor(rb_tree*,rb_node*);
//rb_node* TreeSuccessor(rb_tree*,rb_node*);

//stk_stack * RBEnumerate(rb_tree* tree,void* low, void* high);
//void NullFunction(void*);

typedef struct {
	rb_tree *	tree_p;
	rb_node *	node_p;
} RB_Tree_Enumerator;

extern RB_Tree_Enumerator *new_rbtree_enumerator(rb_tree *tp);
extern void advance_rbtree_enumerator(RB_Tree_Enumerator *rbtep);
extern Item * rbtree_enumerator_item(RB_Tree_Enumerator *rbtep);
extern long rb_node_count(rb_tree *tree_p);
extern void release_rb_tree(rb_tree *tree_p);

#endif // ! _RBTREE_H_

