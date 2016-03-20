#include "quip_config.h"

#import "ios_list.h"
#import "quip_prot.h"

@class IOS_Item;
@class IOS_List;

static IOS_List *ios_node_free_list=NO_IOS_LIST;

@implementation IOS_Node

@synthesize data;
@synthesize next;
@synthesize prev;

+(IOS_Node *) createNode : (id) data
{
	IOS_Node *np;

	if( ios_node_free_list != NO_IOS_LIST &&
			ios_eltcount(ios_node_free_list) > 0 ){
		np = ios_remHead(ios_node_free_list);
	} else {
		np = [[IOS_Node alloc] init ];
	}

	[np setData : data];
	[np setNext : NO_IOS_NODE ];
	[np setPrev : NO_IOS_NODE ];

	return np;
}

@end

IOS_Node *mk_ios_node( id d )
{
	IOS_Node *np=[[IOS_Node alloc] init];
	[np setData : d ];
	return np;
}

/* return a node with this data */
IOS_Node *ios_nodeOf( IOS_List *lp, id d )
{
	IOS_Node *np;

	np = IOS_LIST_HEAD(lp);
	while(np!=NO_IOS_NODE){
		if( IOS_NODE_DATA(np) == d ) return(np);
		np = IOS_NODE_NEXT(np);
	}
	return NO_IOS_NODE;
}

void rls_ios_node(IOS_Node *np)
{
#ifdef CAUTIOUS
	if( np == NULL ){
		fprintf(stderr,"CAUTIOUS:  rls_ios_node passed NULL!?\n");
		abort();
	}
#endif /* CAUTIOUS */
	if( ios_node_free_list == NO_IOS_LIST )
		ios_node_free_list = new_ios_list();
	ios_addHead(ios_node_free_list,np);
}

