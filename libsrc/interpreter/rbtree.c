#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "quip_config.h"
#include "quip_prot.h"
#include "rbtree.h"
#include "getbuf.h"
#include "list.h"
#include "item_type.h"

// just for debugging
//static void rb_tree_dump( qrb_tree *tree_p );
//static void dump_rb_node( qrb_node *np );

// This assumes that the keys are strings!
#define NODE_NAME(np)	(np==NULL?"<null>":RB_NODE_KEY(np))

static void set_root_node(qrb_tree *tree_p, qrb_node *np)
{
	tree_p->root = np;
	if( np != NULL ){
		np->parent = NULL;
		MAKE_BLACK(np);
	}
}

qrb_tree* _create_rb_tree( SINGLE_QSP_ARG_DECL )
/* int (*compare) (const void*,const void*),
						void (*destroy_key) (void*),
						void (*destroy_data) (void*) */
{
	qrb_tree* new_tree_p;

	new_tree_p=(qrb_tree*) getbuf(sizeof(qrb_tree));
//	new_tree_p->comp_func =  compare;
//	new_tree_p->key_destroy_func = destroy_key;
//	new_tree_p->data_destroy_func= destroy_data;
	new_tree_p->root = NULL;
	new_tree_p->node_count = 0;

	// added for list support
	new_tree_p->flags = 0;
	new_tree_p->item_lp = NULL;

	return(new_tree_p);
}

// binary_tree_insert - stick the new node where it goes, regardless of red-black stuff

static void binary_tree_insert(qrb_tree* tree, qrb_node* new_node_p)
{
	qrb_node* curr_node_p;

	if( RB_TREE_ROOT(tree) == NULL ){	// first time
//fprintf(stderr,"making new node %s root\n",(char *)(new_node_p->key));
		set_root_node(tree,new_node_p);
		return;
	}
	curr_node_p = RB_TREE_ROOT(tree);

	// descend the tree to find where the new key goes

	while( curr_node_p != NULL ){
		if( /*tree->comp_func*/ strcmp( RB_NODE_KEY(new_node_p),RB_NODE_KEY(curr_node_p)) < 0 ){
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

static qrb_node *grandparent( qrb_node *np )
{
	if( IS_ROOT_NODE(np) ) return NULL;
	if( IS_ROOT_NODE(np->parent) ) return NULL;
	return np->parent->parent;
}

static qrb_node *uncle( qrb_node *np )
{
	qrb_node *g_p;

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
static void func_name(qrb_tree *tree_p, qrb_node *np)		\
{								\
	qrb_node *gp_p;						\
	qrb_node *parent;					\
	qrb_node *tmp;						\
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

qrb_node * _rb_insert_item(QSP_ARG_DECL  qrb_tree* tree_p, Item *ip )
{
	qrb_node * x_p;
	qrb_node * new_node_p;
	qrb_node * gp_p;	// grandparent
	qrb_node * u_p;	// uncle

	new_node_p = (qrb_node*) getbuf(sizeof(qrb_node));
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

	// NOTREACHED ???
	/*
	tree_p->node_count ++;

	return new_node_p;
	 */
} // rb_insert

qrb_node* rb_find( qrb_tree * tree, const char * key )
{
	int compVal;
	qrb_node* n_p = RB_TREE_ROOT(tree);

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

void rb_substring_find( Frag_Match_Info *fmi_p, qrb_tree * tree, const char * frag )
{
	int compVal;
	int n;
	qrb_node* n_p = RB_TREE_ROOT(tree);

	n = (int) strlen(frag);
	fmi_p->fmi_u.rbti.curr_n_p = NULL;	// default
	fmi_p->fmi_u.rbti.first_n_p = NULL;
	fmi_p->fmi_u.rbti.last_n_p = NULL;

//rb_tree_dump( tree );
	while(1){
		if( n_p == NULL ) return;

//fprintf(stderr,"rb_substring_find:  comparing '%s' to '%s', n = %d\n",frag,RB_NODE_KEY(n_p),n);
		compVal = strncmp( frag, RB_NODE_KEY(n_p), n );
		if( compVal == 0 ){
			// We have found a node that may be a match,
			// but we want to return the first match
			qrb_node *p_p;

//fprintf(stderr,"rb_substring_find:  match found at 0x%lx\n",(long)n_p);
			fmi_p->fmi_u.rbti.curr_n_p = n_p;

			// find the first one
			p_p = rb_predecessor_node(n_p);
			while( p_p != NULL &&  ! strncmp(frag,RB_NODE_KEY(p_p),n) ){
				n_p = p_p;
				p_p = rb_predecessor_node(n_p);
			}
			fmi_p->fmi_u.rbti.first_n_p = n_p;

			// find the last one
			p_p = rb_successor_node(n_p);
			while( p_p != NULL &&  ! strncmp(frag,RB_NODE_KEY(p_p),n) ){
				n_p = p_p;
				p_p = rb_successor_node(n_p);
			}
			fmi_p->fmi_u.rbti.last_n_p = n_p;

			// now set current to the first
			fmi_p->fmi_u.rbti.curr_n_p = fmi_p->fmi_u.rbti.first_n_p;
			return;
		} else if( compVal < 0 ){
//fprintf(stderr,"descending left\n");
			n_p = n_p->left;
		} else {
//fprintf(stderr,"descending right\n");
			n_p = n_p->right;
		}
	}
}

// a misnomer - this exchanges the data of the node-to-be-deleted with its predecessor,
// in preparation for the real deletion.

static qrb_node * binary_tree_delete(qrb_node *np)
{
	qrb_node *p;
//const void *tmp_p1;
//void *tmp_p2;

	assert(np->left!=NULL && np->right!=NULL );

// WHY WAS THIS HERE???
//	if( IS_ROOT_NODE(np) ) return np;

	// find the predecessor node
	p = np->left;
	if( p == NULL ){	// no predecessor
		// find the successor
		p = np->right;
		assert(p!=NULL);	// per assertion above
//		if( p == NULL ) return np;
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

static qrb_node *sibling( qrb_node *np, qrb_node *parent )
{
	if( parent == NULL ) return NULL;

	if( np == parent->left )
		return parent->right;
	else
		return parent->left;
}

static void rebalance(qrb_tree *tree_p, qrb_node *n_p, qrb_node *parent)
{
	qrb_node *s_p;		// sibling
	int c;

	if( n_p != NULL && IS_ROOT_NODE(n_p) ) return;	// wikipedia case 1

	assert(parent!=NULL);
	assert( IS_BLACK(n_p) );

	s_p = sibling(n_p,parent);	// we have to pass the parent because n_p may be NULL
    assert(s_p!=NULL);

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
        assert(s_p!=NULL);
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

static void replace_node( qrb_tree *tree_p, qrb_node *n_p, qrb_node *c_p)
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

// delete a single node from the tree
// Don't worry about the pointed-to data, that's someone else's responsibility.

static void rb_delete(qrb_tree *tree_p, qrb_node *n_p )
{
	qrb_node *c_p;		// the single non-leaf child
	qrb_node *parent;

	tree_p->node_count --;

	if( tree_p->node_count == 0 ){
		assert( IS_ROOT_NODE(n_p) );
		givbuf(n_p);
		tree_p->root = NULL;
		return;
	}

	if( n_p->left != NULL && n_p->right != NULL ){
//fprintf(stderr,"before calling binary_tree_delete:\n");
//dump_rb_node(n_p);
		n_p = binary_tree_delete(n_p);
//fprintf(stderr,"after calling binary_tree_delete:\n");
//dump_rb_node(n_p);
	} else if( IS_ROOT_NODE(n_p) ){
		// we are deleting a root node with only one child
		if( n_p->left != NULL ){
			set_root_node(tree_p,n_p->left);
		} else {
			set_root_node(tree_p,n_p->right);
		}
		givbuf(n_p);
		return;
	}

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

// delete the node containing an item from the tree

int rb_delete_key(qrb_tree *tree_p, const char *key)
{
	qrb_node *n_p;

	n_p = rb_find(tree_p,key);
	if( n_p == NULL ){
		fprintf(stderr,"rb_delete_key:  failed to find key \"%s\"!?\n",key);
		return -1;
	}
	rb_delete(tree_p,n_p);	// rb_delete frees the memory...
	return 0;
}

int rb_delete_item(qrb_tree *tree_p, Item *ip)
{
	return rb_delete_key(tree_p,ITEM_NAME(ip));
}

void _rb_traverse(QSP_ARG_DECL  qrb_node *np, void (*func)(QSP_ARG_DECL  qrb_node *, qrb_tree *), qrb_tree* tree_p )
{
	if( np == NULL ) return;

	rb_traverse( np->left, func, tree_p );
	(*func)(QSP_ARG  np, tree_p);
	rb_traverse( np->right, func, tree_p );

}

#ifdef RB_TREE_DEBUG

#define MIN(a,b)	(a<b?a:b)
#define MAX(a,b)	(a>b?a:b)

static void rb_node_scan( qrb_node *n_p )
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

	// make sure that the order is correct!
	if( n_p->left != NULL ){
		if( strcmp(NODE_NAME(n_p->left),NODE_NAME(n_p)) > 0 )
			fprintf(stderr,"OOPS left child %s out of order with node %s!?\n",
				NODE_NAME(n_p->left),NODE_NAME(n_p));
	}
	if( n_p->right != NULL ){
		if( strcmp(NODE_NAME(n_p),NODE_NAME(n_p->right)) > 0 )
			fprintf(stderr,"OOPS right child %s out of order with node %s!?\n",
				NODE_NAME(n_p->right),NODE_NAME(n_p));
	}

fprintf(stderr,"rb_node_scan 0x%lx (%s)\n"
"\tcolor = %s, left = 0x%lx (%s), right = 0x%lx (%s), parent = 0x%lx (%s)\n"
"\tdepth = %d, black_depth = %d, min_black_leaf = %d, max_black_leaf = %d\n",
(long)n_p,RB_NODE_KEY(n_p),
IS_BLACK(n_p)?"black":"red",(long)n_p->left,NODE_NAME(n_p->left),(long)n_p->right,NODE_NAME(n_p->right),(long)n_p->parent,
	NODE_NAME(n_p->parent),
n_p->depth,n_p->black_depth,n_p->min_black_leaf,n_p->max_black_leaf);
}

void rb_check( qrb_tree *tree_p )
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
#endif // RB_TREE_DEBUG

// Comments are written for dir = right, and opp_dir = left

#define GENERIC_NEXT_NODE_FUNC( func_name, dir, opp_dir )	\
								\
qrb_node* func_name(qrb_node* n_p)				\
{								\
	qrb_node* s_p;						\
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

static long rb_branch_count(qrb_node *np)
{
	long l,r;
	if( np->left != NULL )
		l = rb_branch_count(np->left);
	else
		l = 0;
	if( np->right != NULL )
		r = rb_branch_count(np->right);
	else
		r=0;
	return 1 + l + r;
}

// BUG - should we keep a count while inserting and deleting???
long rb_node_count(qrb_tree *tree_p)
{
	long count = 0;

	if( tree_p->node_count >= 0 ) return(tree_p->node_count);

	if( tree_p->root != NULL )
		count = rb_branch_count(tree_p->root);

	return count;
}

void advance_rb_tree_enumerator(RB_Tree_Enumerator *rbtep)
{
	if( rbtep->node_p == NULL ) return;
	rbtep->node_p = rb_successor_node(rbtep->node_p);
}

static void release_rb_branch(qrb_node *np)
{
	assert( np != NULL );

	if( np->left != NULL ) release_rb_branch(np->left);
	if( np->right != NULL ) release_rb_branch(np->right);
	// eventually we might want to put this on a free list instead of returning to heap...
	givbuf(np);
}

void release_rb_tree(qrb_tree *tree_p)
{
	if( tree_p->root != NULL ) release_rb_branch(tree_p->root);
	givbuf(tree_p);
}

void rls_rb_tree_enumerator(RB_Tree_Enumerator *ep)
{
	givbuf(ep);	// keep a pool for efficiency?  Maybe the tree should have an enumerator as part of it?
}

RB_Tree_Enumerator *_new_rb_tree_enumerator(QSP_ARG_DECL  qrb_tree *tree_p)
{
	RB_Tree_Enumerator *rbtep;

	if( tree_p->root == NULL ) return NULL;

	rbtep = getbuf( sizeof(*rbtep) );
	rbtep->tree_p = tree_p;
	rbtep->node_p = tree_p->root;
	while( rbtep->node_p->left != NULL )
		rbtep->node_p = rbtep->node_p->left;
	return rbtep;
}

Item *rb_tree_enumerator_item(RB_Tree_Enumerator *rbtep)
{
	if( rbtep->node_p == NULL ) return NULL;
	return rbtep->node_p->data;
}

/*
// for debugging

static void dump_rb_node( qrb_node *np, qrb_tree *tree_p )
{
	fprintf(stderr,"node 0x%lx (%s)\n",(long)np,ITEM_NAME( (Item *)(np->data) ) );
	fprintf(stderr,"\tleft 0x%lx (%s)\t\tright 0x%lx (%s)\n",
		(long)np->left,np->left==NULL?"<null>":ITEM_NAME( (Item *)(np->left->data) ),
		(long)np->right,np->right==NULL?"<null>":ITEM_NAME( (Item *)(np->right->data) )
		);
}

static void rb_tree_dump( qrb_tree *tree_p )
{
	if( tree_p->root == NULL ) return;
	rb_traverse(tree_p->root,dump_rb_node,tree_p);
}
*/

#define add_rb_node_to_list(rbn_p,tree_p) _add_rb_node_to_list(QSP_ARG  rbn_p,tree_p)

static void _add_rb_node_to_list(QSP_ARG_DECL  qrb_node *rbn_p, qrb_tree *tree_p)
{
	Node *np;
	Item *ip;

	ip = RB_NODE_ITEM(rbn_p);
	np = mk_node(ip);
	addTail( RB_TREE_ITEM_LIST(tree_p), np );
}

#define make_rb_tree_list(tree_p) _make_rb_tree_list(QSP_ARG  tree_p)

static void _make_rb_tree_list(QSP_ARG_DECL  qrb_tree *tree_p)
{
	assert( RB_TREE_ITEM_LIST(tree_p) == NULL );

	SET_RB_TREE_ITEM_LIST(tree_p,new_list());
	rb_traverse(tree_p->root,_add_rb_node_to_list,tree_p);
	MARK_RB_TREE_CURRENT(tree_p);
}

List *_rb_tree_list(QSP_ARG_DECL  qrb_tree *tree_p)
{
	if( RB_TREE_ITEM_LIST(tree_p) == NULL ){
		make_rb_tree_list(tree_p);	// allocates and populates a new list
	} else {
		if( ! RB_TREE_LIST_IS_CURRENT(tree_p) ){
			zap_list( RB_TREE_ITEM_LIST(tree_p) );
			make_rb_tree_list(tree_p);	// allocates and populates a new list
		}
	}
	return RB_TREE_ITEM_LIST(tree_p);
}


