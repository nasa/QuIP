#ifndef _NODE_H_
#define _NODE_H_

typedef struct node {
	void *		n_data;
	struct node *	n_next;
	struct node *	n_prev;
	int		n_pri;
} Node;


#define NO_NODE 	((Node *)NULL)

extern void rls_node(Node *np);
extern Node *mk_node( void * ip );
extern void init_node(Node *np,void* dp);
//extern Node *nodeOf( struct list *lp, void * ip );

#define NODE_DATA(np)		(np)->n_data
#define SET_NODE_DATA(np,d)	(np)->n_data=d
#define NODE_NEXT(np)		(np)->n_next
#define NODE_PREV(np)		(np)->n_prev
#define SET_NODE_NEXT(np,_np)	(np)->n_next = _np
#define SET_NODE_PREV(np,_np)	(np)->n_prev = _np

#endif /* !  _NODE_H_ */

