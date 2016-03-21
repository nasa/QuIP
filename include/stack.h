
#ifndef _STACK_H_
#define _STACK_H_

#include "list.h"

typedef List Stack;

#define NO_STACK ((Stack *)NULL)
#define new_stack new_list

extern void push_item( Stack *stkp, void *ip );
extern void * pop_item( Stack *stkp );

#define TOP_OF_STACK(stkp)	\
	(QLIST_HEAD(stkp)==NULL?NULL:NODE_DATA(QLIST_HEAD(stkp)))

#define STACK_TOP_NODE(stkp)	QLIST_HEAD(stkp)

#define BOTTOM_OF_STACK(stkp)	\
	(QLIST_TAIL(stkp)==NULL?NULL:NODE_DATA(QLIST_TAIL(stkp)))

#define STACK_IS_EMPTY(stkp)	(QLIST_HEAD(stkp) == NO_NODE)
#define STACK_DEPTH(stkp)	eltcount(stkp)
#define PUSH_TO_STACK(stkp,p)	push_item(stkp,p)
#define POP_FROM_STACK(stkp)	pop_item(stkp)

#endif /* ! _STACK_H_ */

