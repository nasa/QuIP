
#ifndef _IOS_LIST_H_
#define _IOS_LIST_H_

#include "ios_node.h"

@interface IOS_List: NSObject

// Why "copy" for this property?
//@property (copy) IOS_Node *head;
//@property (copy) IOS_Node *tail;
@property (retain) IOS_Node *head;
@property (retain) IOS_Node *tail;

-(void) addHead: (IOS_Node *)np;
-(void) addTail: (IOS_Node *)np;
-(IOS_Node *) remHead;
-(IOS_Node *) elementAtIndex : (int) index;
-(int) length;
-(void) addListOfItems:(IOS_List *)lp;
//-(IOS_Node *) remTail;

#define NO_IOS_LIST		((IOS_List *)NULL)
@end

#define IOS_LIST_HEAD(lp)		(lp).head
#define IOS_LIST_TAIL(lp)		(lp).tail
#define SET_IOS_LIST_HEAD(lp,np)	(lp).head = np
#define SET_IOS_LIST_TAIL(lp,np)	(lp).tail = np

extern IOS_List *new_ios_list(void);
extern int ios_eltcount(IOS_List *lp);
extern IOS_Node *ios_remHead(IOS_List *lp);
extern IOS_Node *ios_remTail(IOS_List *lp);
extern void ios_addHead(IOS_List *lp, IOS_Node *np);
extern IOS_Node * ios_remNode(IOS_List *lp, IOS_Node *np);
extern IOS_Node *ios_remData(IOS_List *lp, id data);
extern void rls_ios_list(IOS_List *lp);
extern void ios_addTail(IOS_List *lp, IOS_Node *np);
extern void ios_dellist(IOS_List *lp);
extern void rls_nodes_from_ios_list(IOS_List *lp);
extern IOS_Node *ios_nth_elt(IOS_List *lp, int k);

/* BUG these are not implemented yet */
#define LOCK_IOS_LIST(lp)
#define UNLOCK_IOS_LIST(lp)

#endif // ! _IOS_LIST_H_

