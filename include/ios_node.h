#ifndef _IOS_NODE_H_
#define _IOS_NODE_H_

#import <Foundation/Foundation.h>

@class IOS_Item;
@class IOS_List;

@interface IOS_Node: NSObject

@property (retain) id data;
@property (retain) IOS_Node *next;
@property (retain) IOS_Node *prev;

//-(void) execute;

+(IOS_Node *) createNode : (id) data;

@end

#define NO_IOS_NODE 	((IOS_Node *)NULL)

#define IOS_NODE_DATA(np)	(np).data
#define IOS_NODE_NEXT(np)	(np).next
#define IOS_NODE_PREV(np)	(np).prev
#define SET_IOS_NODE_DATA(np,v)	(np).data = v
#define SET_IOS_NODE_NEXT(np,v)	(np).next = v
#define SET_IOS_NODE_PREV(np,v)	(np).prev = v

extern void rls_ios_node(IOS_Node *np);
extern IOS_Node *mk_ios_node( id ip );
extern IOS_Node *ios_nodeOf( IOS_List *lp, id ip );

#endif // ! _IOS_NODE_H_

