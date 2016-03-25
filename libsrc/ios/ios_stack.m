
#import "ios_stack.h"

@implementation IOS_Stack

@synthesize list;

-(void) push : (id) obj
{
	IOS_Node *np;

	np = [IOS_Node createNode : obj ];

#ifdef CAUTIOUS
	if( np == NULL ){
		fprintf(stderr,"IOS_Node createNode returned NULL !?!?!?\n");
		abort();
	}
#endif /* CAUTIOUS */

	[ list addHead : np ];
}

-(id) pop
{
	IOS_Node *np;

	np = [list remHead];
	if( np == NO_IOS_NODE ) return NULL;

	return np.data;
}

-(id) init
{
	self=[super init];

	list = [[IOS_List alloc] init ];
	return self;
}

-(int) depth
{
	return list.length;
}

-(id) top
{
	IOS_Node *np;

	np = list.head;
	return np.data;
}

-(id) bottom
{
    IOS_Node *np;
    
    np = list.tail;
    return np.data;
}

@end

