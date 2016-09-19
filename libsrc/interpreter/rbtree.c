#include "quip_config.h"
#include "quip_prot.h"
#include "rbtree.h"
#include "getbuf.h"
#include <assert.h>
#include <stdio.h>

// This assumes that the keys are strings!
#define NODE_NAME(np)	(np==NULL?"<null>":RB_NODE_KEY(np))

static void set_root_node(rb_tree *tree_p, rb_node *np)
{
	tree_p->root = np;
	np->parent = NULL;
	MAKE_BLACK(np);
}

rb_tree* create_rb_tree( void )
/* int (*compare) (const void*,const void*),
						void (*destroy_key) (void*),
						void (*destroy_data) (void*) */
{
	rb_tree* new_tree_p;

	new_tree_p=(rb_tree*) getbuf(sizeof(rb_tree));
//	new_tree_p->comp_func =  compare;
//	new_tree_p->key_destroy_func = destroy_key;
//	new_tree_p->data_destroy_func= destroy_data;
	new_tree_p->root = NULL;

	return(new_tree_p);
}

// binary_tree_insert - stick the new node where it goes, regardless of red-black stuff

static void binary_tree_insert(rb_tree* tree, rb_node* new_node_p)
{
	rb_node* curr_node_p;

	if( RB_TREE_ROOT(tree) == NULL ){	// first time
//fprintf(stderr,"making new node %s root\n",(char *)(new_node_p->key));
		set_root_node(tree,new_node_p);
		return;
	}
	curr_node_p = RB_TREE_ROOT(tree);

	// descend the tree to find where the new key goes

	while( curr_node_p != NULL ){
		if( /*tree->comp_func*/( RB_NODE_KEY(new_node_p),RB_NODE_KEY(curr_node_p)) < 0 ){
			if( curr_node_p->left == NULL ){
				new_node_p->parent = curr_node_p;
				curr_node_p->left = new_node_p;
				return;
			}
			curr_node_p = curr_node_p->left;
		} else {
			if( curr_node_p->right == NULL ){
				new_node_p->parent = curr_node_p;
				curr_node_p->right = new_node_p;
				return;
			}
			curr_node_p = curr_node_p->right;
		}
	}
	// should never get here!
	assert(1==0);
}

static rb_node *grandparent( rb_node *np )
{
	if( IS_ROOT_NODE(np) ) return NULL;
	if( IS_ROOT_NODE(np->parent) ) return NULL;
	return np->parent->parent;
}

static rb_node *uncle( rb_node *np )
{
	rb_node *g_p;

	g_p = grandparent(np);
	if( g_p == NULL ) return NULL;

	if( np->parent == g_p->left )
		return g_p->right;
	else
		return g_p->left;
}


// if we call rotate_right, the node should be the left child of its parent

#define GENERAL_ROTATION(func_name,current_side,direction)	\
								\
static void func_name(rb_tree *tree_p, rb_node *np)		\
{								\
	rb_node *gp_p;						\
	rb_node *parent;					\
	rb_node *tmp;						\
								\
	assert( np->parent != NULL );				\
	assert( np->parent->current_side == np );		\
								\
	gp_p = grandparent(np);					\
								\
	parent = np->parent;					\
								\
	if( gp_p != NULL ){					\
		if( parent == gp_p->direction )			\
			gp_p->direction = np;			\
		else						\
			gp_p->current_side = np;		\
	}							\
								\
	/* child node becomes grandchild */			\
	tmp = np->direction;					\
	np->direction = parent;					\
	parent->current_side = tmp;				\
	if( tmp != NULL ) tmp->parent = parent;			\
								\
	parent->parent = np;					\
	np->parent = gp_p;					\
								\
	if( IS_ROOT_NODE(np) )					\
		tree_p->root = np;				\
}

GENERAL_ROTATION(rotate_right,left,right)
GENERAL_ROTATION(rotate_left,right,left)

rb_node * rb_insert_item(rb_tree* tree_p, const Item *ip )
{
	rb_node * x_p;
	rb_node * new_node_p;
	rb_node * gp_p;	// grandparent
	rb_node * u_p;	// uncle

	new_node_p = (rb_node*) getbuf(sizeof(rb_node));
//	new_node_p->key = key;
	new_node_p->data  = ip;
	new_node_p->left = NULL;
	new_node_p->right = NULL;
	MAKE_RED(new_node_p);

	binary_tree_insert(tree_p,new_node_p);

	x_p = new_node_p;

	while( 1 ){
		if( IS_ROOT_NODE(x_p) ){	// wikipedia case 1
			MAKE_BLACK(x_p);
			return new_node_p;
		}

		if( IS_BLACK(x_p->parent) ){	// wikipedia case 2
			return new_node_p;
		}

		gp_p = grandparent(x_p);
		assert( gp_p != NULL );

		u_p = uncle(x_p);

		// We know the parent is red

		if( IS_RED(u_p) ){	// wikipedia case 3
			MAKE_BLACK(x_p->parent);
			MAKE_BLACK(u_p);
			MAKE_RED(gp_p);
			x_p = gp_p;	// loop on grandparent
		} else {
			// uncle is black
			if( x_p == x_p->parent->left && x_p->parent == gp_p->left ){
				// wikipedia case 5, left child of a left child
				rotate_right(tree_p,x_p->parent);
				MAKE_BLACK(x_p->parent);
				// new uncle is old grandparent
				MAKE_RED(gp_p);
				return new_node_p;
			} else if( x_p == x_p->parent->right && x_p->parent == gp_p->right ){
				// wikipedia case 5 mirror image
				rotate_left(tree_p,x_p->parent);
				MAKE_BLACK(x_p->parent);
				// new uncle is old grandparent
				MAKE_RED(gp_p);
				return new_node_p;
			} else {
				// wikipedia case 4
				if( x_p == x_p->parent->right ){
					// right child of left child
					rotate_left(tree_p,x_p);
					x_p = x_p->left;
				} else {
					// left child of right child
					rotate_right(tree_p,x_p);
					x_p = x_p->right;
				}
			}
		}
	} // end tail recursion loop
	//MAKE_BLACK( RB_TREE_ROOT(tree) );
	return new_node_p;
} // rb_insert

rb_node* rb_find( rb_tree * tree, const char * key )
{
	int compVal;
	rb_node* n_p = RB_TREE_ROOT(tree);

	while(1){
		if( n_p == NULL ) return(NULL);

		compVal = /*tree->comp_func*/ strcmp( key, RB_NODE_KEY(n_p) );
		if( compVal == 0 ) return n_p;

		else if( compVal < 0 ){
			n_p = n_p->left;
		} else {
			n_p = n_p->right;
		}
	}
}

static rb_node * binary_tree_delete(rb_node *np)
{
	rb_node *p;
//const void *tmp_p1;
//void *tmp_p2;

	assert(np->left!=NULL && np->right!=NULL );

	if( IS_ROOT_NODE(np) ) return np;

	// find the predecessor node
	p = np->left;
	if( p == NULL ){	// no predecessor
		// find the successor
		p = np->right;
		if( p == NULL ) return np;
		while( p->left != NULL )
			p = p->left;
	} else {
		while( p->right != NULL )
			p = p->right;
	}

//// as we will delete p anyway, we don't really need to exchange the data fields,
//// but for debugging it is easier to see what is going on if we do so
//tmp_p1 = np->key;
//tmp_p2 = np->data;

	// p is the predecessor/successor;
	// copy the values, then return p
	//np->key = p->key;		// for now the key is in the data (ip)
	np->data = p->data;

//p->key = tmp_p1;
//p->data = tmp_p2;

	return p;
}

static rb_node *sibling( rb_node *np, rb_node *parent )
{
	if( parent == NULL ) return NULL;

	if( np == parent->left )
		return parent->right;
	else
		return parent->left;
}

static void rebalance(rb_tree *tree_p, rb_node *n_p, rb_node *parent)
{
	rb_node *s_p;		// sibling
	int c;

	if( n_p != NULL && IS_ROOT_NODE(n_p) ) return;	// wikipedia case 1

	assert(parent!=NULL);
	assert( IS_BLACK(n_p) );

	s_p = sibling(n_p,parent);	// we have to pass the parent because n_p may be NULL

	if( IS_RED(s_p) ){		// wikipedia case 2 : sibling is red
		MAKE_RED(/*n_p->*/parent);
		MAKE_BLACK(s_p);
		if( n_p == /*n_p->*/parent->left ){
			rotate_left(tree_p,s_p);
		} else {
			rotate_right(tree_p,s_p);
		}
		//assert(n_p!=NULL);
		s_p = sibling(n_p,parent);		// has changed
	}
	
	if( IS_BLACK(s_p) ){
		if( IS_BLACK(s_p->left) && IS_BLACK(s_p->right) ){
			// sibling and its children are black
			if( IS_BLACK(/*n_p->*/parent) ){
				// wikipedia case 3 - make the sibling red
				MAKE_RED(s_p);
				rebalance(tree_p,/*n_p->*/parent,parent->parent);
				return;
			} else {
				// wikipedia case 4
				MAKE_BLACK(/*n_p->*/parent);
				MAKE_RED(s_p);
				return;
			}
		}
		// remaining cases depend on which side S is on
		if( s_p == /*n_p->*/ parent->right ){
			// S on right - the case shown in wikipedia examples
			if( IS_RED(s_p->left) && IS_BLACK(s_p->right) ){
				// wikipedia case 5
				assert( IS_BLACK(s_p->right) );
				rotate_right(tree_p,s_p->left);
				MAKE_RED(s_p);
				MAKE_BLACK(s_p->parent);	// used to be s_p->left

				// N's sibling has changed
				s_p = sibling(n_p,parent);

				// goto case 6
			}
		} else {
			// sibling is left child - mirror case
			if( IS_RED(s_p->right) && IS_BLACK(s_p->left) ){
				// wikipedia case 5
				assert( IS_BLACK(s_p->left) );
				rotate_left(tree_p,s_p->right);
				MAKE_RED(s_p);
				MAKE_BLACK(s_p->parent);	// used to be s_p->left

				// N's sibling has changed
				s_p = sibling(n_p,parent);

				// goto case 6
			}
		}

		// fall-through to case 6
			
		if( s_p == /*n_p->*/parent->right ){
			// case 6 illustrated on wikipedia
			assert( IS_RED(s_p->right) );
			rotate_left(tree_p,s_p);
			// exchange colors - s_p is now the parent and the old parent is now its left child
			c = s_p->color;
			s_p->color = s_p->left->color;
			s_p->left->color = c;
			MAKE_BLACK(s_p->right);
		} else {
			// case 6, mirror condition
			assert( IS_RED(s_p->left) );
			rotate_right(tree_p,s_p);
			// exchange colors - s_p is now the parent and the old parent is now its left child
			c = s_p->color;
			s_p->color = s_p->right->color;
			s_p->right->color = c;
			MAKE_BLACK(s_p->left);
		}
	}
}

// replace a node by one of its children (the other child is known to by NULL)
//

static void replace_node( rb_tree *tree_p, rb_node *n_p, rb_node *c_p)
{
	// point the parent to the new node

	if( n_p == n_p->parent->right ){
		n_p->parent->right = c_p;
	} else {
		n_p->parent->left = c_p;
	}

	// point the new node to its new parent
	if( c_p != NULL )
		c_p->parent = n_p->parent;

	//if( tree_p->root == n_p )
	//	tree_p->root = c_p;
	assert( tree_p->root != n_p );
}

static void rb_delete(rb_tree *tree_p, rb_node *n_p )
{
	rb_node *c_p;		// the single non-leaf child
	rb_node *parent;

	if( IS_ROOT_NODE(n_p) ){
		givbuf(n_p);
		tree_p->root = NULL;
		return;
	}

	if( n_p->left != NULL && n_p->right != NULL ){
		n_p = binary_tree_delete(n_p);
	}
	assert( ! IS_ROOT_NODE(n_p) );

	assert( n_p->left == NULL || n_p->right == NULL );

	if( n_p->left != NULL )
		c_p = n_p->left;
	else	c_p = n_p->right;	// may be NULL


	// if c_p is null, then we need to keep a reference to the parent...
	parent=n_p->parent;
	replace_node(tree_p,n_p,c_p);

	if( IS_RED(n_p) ){
		givbuf(n_p);
		return;
	}
	
	// Now we can free n_p
	givbuf(n_p);

	if( IS_RED(c_p) ){
		MAKE_BLACK(c_p);
		return;
	}

	// Now we know the node in question is black and has no red child
	//assert( n_p->left != NULL && n_p->right != NULL );

	// Now we know that both n_p and c_p are black

	rebalance(tree_p,c_p,parent);
}

void rb_delete_key(rb_tree *tree_p, const void *key)
{
	rb_node *n_p;

	n_p = rb_find(tree_p,key);
	if( n_p == NULL ){
		fprintf(stderr,"rb_delete_key:  failed to find key %s!?\n",(char *)key);
		return;
	}
	rb_delete(tree_p,n_p);
}


void rb_traverse( rb_node *np, void (*func)(rb_node *) )
{
	if( np == NULL ) return;

	rb_traverse( np->left, func );
	(*func)(np);
	rb_traverse( np->right, func );

}

#ifdef DEBUG

#define MIN(a,b)	(a<b?a:b)
#define MAX(a,b)	(a>b?a:b)

static void rb_node_scan( rb_node *n_p )
{
	int min,max;

	if( ! IS_ROOT_NODE(n_p) ){
		n_p->depth = n_p->parent->depth + 1;
		if( IS_BLACK(n_p) )
			n_p->black_depth = n_p->parent->black_depth + 1;
		else
			n_p->black_depth = n_p->parent->black_depth;
	}
	n_p->max_black_leaf = 0;
	n_p->min_black_leaf = 0;

	if( n_p->left != NULL ){
		rb_node_scan(n_p->left);
		min = n_p->left->min_black_leaf;
		max = n_p->left->max_black_leaf;
		if( IS_BLACK(n_p->left) ){
			min ++;
			max ++;
		}
	} else {
		min = 1;
		max = 1;
	}

	n_p->min_black_leaf = min;
	n_p->max_black_leaf = max;

	if( n_p->right != NULL ){
		rb_node_scan(n_p->right);
		min = n_p->right->min_black_leaf;
		max = n_p->right->max_black_leaf;
		if( IS_BLACK(n_p->right) ){
			min ++;
			max ++;
		}
	} else {
		min = 1;
		max = 1;
	}

	n_p->min_black_leaf = MIN(n_p->min_black_leaf,min);
	n_p->max_black_leaf = MAX(n_p->max_black_leaf,max);

	if( n_p->max_black_leaf != n_p->min_black_leaf )
		fprintf(stderr,"OOPS - unbalanced black node counts!?\n");

	if( IS_RED(n_p) && ! ( IS_BLACK(n_p->left) && IS_BLACK(n_p->right)) )
		fprintf(stderr,"OOPS Red node %s does not have two black children!?\n",NODE_NAME(n_p));

	// make sure linkage with parent is OK
	if( n_p->parent != NULL ){
		if( n_p != n_p->parent->left && n_p != n_p->parent->right )
			fprintf(stderr,"OOPS node %s is not a child of parent %s!?\n",NODE_NAME(n_p),NODE_NAME(n_p->parent));
	}

fprintf(stderr,"rb_node_scan 0x%lx (%s)\n"
"\tcolor = %s, left = 0x%lx (%s), right = 0x%lx (%s), parent = 0x%lx (%s)\n"
"\tdepth = %d, black_depth = %d, min_black_leaf = %d, max_black_leaf = %d\n",
(long)n_p,RB_NODE_KEY(n_p),
IS_BLACK(n_p)?"black":"red",(long)n_p->left,NODE_NAME(n_p->left),(long)n_p->right,NODE_NAME(n_p->right),(long)n_p->parent,
	NODE_NAME(n_p->parent),
n_p->depth,n_p->black_depth,n_p->min_black_leaf,n_p->max_black_leaf);
}

void rb_check( rb_tree *tree_p )
{
	if( tree_p->root == NULL ) return;

	if( ! IS_BLACK(tree_p->root) ){
		fprintf(stderr,"OOPS rb_check:  root node should be black!?\n");
		return;
	}
	tree_p->root->depth=0;
	tree_p->root->black_depth=1;
	rb_node_scan(tree_p->root);
}
#endif // DEBUG

// Comments are written for dir = right, and opp_dir = left

#define GENERIC_NEXT_NODE_FUNC( func_name, dir, opp_dir )	\
								\
rb_node* func_name(rb_node* n_p)				\
{								\
	rb_node* s_p;						\
								\
	if( n_p->dir != NULL ){	/* has right subtree */		\
		/* find minimum (left-most) node */		\
		s_p = n_p->dir;					\
		while( s_p->opp_dir != NULL )			\
			s_p = s_p->opp_dir;			\
		return s_p;					\
	}							\
								\
	s_p = n_p->parent;					\
	while( s_p != NULL ){					\
		if( n_p == s_p->opp_dir )			\
			return s_p;				\
		n_p = s_p;					\
		s_p = s_p->parent;				\
	}							\
	return s_p;	/* NULL */				\
}

GENERIC_NEXT_NODE_FUNC( rb_successor_node, right, left )
GENERIC_NEXT_NODE_FUNC( rb_predecessor_node, left, right )

