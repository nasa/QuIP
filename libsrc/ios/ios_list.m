
#import "ios_list.h"
#import "quip_prot.h"

@implementation IOS_List

@synthesize head;
@synthesize tail;

-(IOS_Node *) elementAtIndex : (int) index;
{
	IOS_Node *np=head;
	int i=index;

	while(i--){
		if( np == NO_IOS_NODE ) return np;
		np = np.next;
	}
	return np;
}
			
-(void) addHead : (IOS_Node *) np
{
	[ np setNext : head ];
	[ np setPrev : NO_IOS_NODE ];

	if( head != NO_IOS_NODE ){
		[ head setPrev : np ];
	} else {
		// list was empty, so set the tail too
		tail = np;
	}

	head = np;
}

-(void) addTail : (IOS_Node *) np
{
	[ np setPrev : tail ];
	[ np setNext : NO_IOS_NODE ];

	if( tail != NO_IOS_NODE ){
		[ tail setNext : np ];
	} else {
		// list was empty, so set the head too
		head = np;
	}

	tail = np;
}

-(id) init
{
	self=[super init];

	head = NO_IOS_NODE;
	tail = NO_IOS_NODE;
	return self;
}

-(int) length
{
	int l=0;
	IOS_Node *np;

	np = head;
	while( np != NO_IOS_NODE ){
		l++;
		np = np.next;
	}
	return l;
}

-(IOS_Node *) remHead
{
	IOS_Node *np;
	IOS_Node *next;

	if( head == NO_IOS_NODE ) return(head);

	next = head.next;

	if( next != NO_IOS_NODE ){
		[next setPrev : NO_IOS_NODE ];
	} else {
        tail = NO_IOS_NODE;
    }

	np = head;
	head = next;
    // BUG?  should we clear the next pointer in np?
	return np;
}

-(IOS_Node *) remTail
{
	IOS_Node *np;
	IOS_Node *prev;

	if( tail == NO_IOS_NODE ) return(tail);

	prev = head.prev;


	if( prev != NO_IOS_NODE ){
		[prev setNext : NO_IOS_NODE ];
	} else {
        head=NO_IOS_NODE;
    }

	np = tail;
	tail = prev;
	return np;
}

-(void) addListOfItems:(IOS_List *)lp
{
    IOS_Node *np;
    
    np = IOS_LIST_HEAD(lp);
    while(np!=NO_IOS_NODE){
        IOS_Node *new_np = mk_ios_node( IOS_NODE_DATA(np) );
        [self addTail : new_np];
        np=IOS_NODE_NEXT(np);
    }
    
}

@end

IOS_List *new_ios_list(void)
{
	return [[IOS_List alloc]init];
}

int ios_eltcount(IOS_List *lp)
{
	int n;
	IOS_Node *np;

	n=0;
	np=lp.head;
	while(np!=NO_IOS_NODE){
		n++;
		np=np.next;
	}
	return(n);
}

IOS_Node *ios_remHead(IOS_List *lp)
{
	return [lp remHead];
}

IOS_Node *ios_remTail(IOS_List *lp)
{
	return [lp remTail];
}

void ios_addHead(IOS_List *lp, IOS_Node *np)
{
	[lp addHead : np];
}

IOS_Node * ios_remNode(IOS_List *lp, IOS_Node *np)
{
	IOS_Node *scan_np;

	scan_np=lp.head;
	while( scan_np != NO_IOS_NODE ){
		if( scan_np == np ){
			if( scan_np == lp.head )
				[lp setHead : scan_np.next];
			if( scan_np == lp.tail )
				[lp setTail : scan_np.prev];
			if( scan_np.prev != NO_IOS_NODE )
				[scan_np.prev setNext : scan_np.next];
			if( scan_np.next != NO_IOS_NODE )
				[scan_np.next setPrev : scan_np.prev];
			return scan_np;
		}
		scan_np = scan_np.next;
	}
	return scan_np;
}

IOS_Node *ios_remData(IOS_List *lp, id data)
{
	IOS_Node *np;

	np=lp.head;
	while(np!=NO_IOS_NODE){
		if( np.data == data ){
			if( np == lp.head )
				[lp setHead : np.next];
			if( np == lp.tail )
				[lp setTail : np.prev];
			if( np.prev != NO_IOS_NODE )
				[np.prev setNext : np.next];
			if( np.next != NO_IOS_NODE )
				[np.next setPrev : np.prev];
			return np;
		}
		np = np.next;
	}
	return np;
}

void rls_nodes_from_ios_list(IOS_List *lp)
{
	IOS_Node *np;
    
	LOCK_IOS_LIST(lp)
	while( IOS_LIST_HEAD(lp) != NO_IOS_NODE ) {
		np=ios_remHead(lp);
		// We can let the system handle the releasing...
		// But we null out the data pointer just in case...
		SET_IOS_NODE_DATA(np,NULL);
		SET_IOS_NODE_NEXT(np,NULL);
		SET_IOS_NODE_PREV(np,NULL);
	}
	UNLOCK_IOS_LIST(lp)
}


void rls_ios_list(IOS_List *lp)
{
	NWARN("rls_ios_list not implemented!?");
}

void ios_addTail(IOS_List *lp, IOS_Node *np)
{
	[lp addTail : np];
}

void ios_dellist(IOS_List *lp)
{
	NWARN("ios_dellist not implemented");
}

IOS_Node *ios_nth_elt(IOS_List *lp, int index)
{
	// inefficient?
	IOS_Node *np;

	np = IOS_LIST_HEAD(lp);
	while(np!=NO_IOS_NODE && index-- )
		np=IOS_NODE_NEXT(np);

	if( index != (-1) ){
		sprintf(DEFAULT_ERROR_STRING,"ios_nth_elt 0x%lx (%d elts)  %d",(long)lp,
			ios_eltcount(lp),index);
		NADVISE(DEFAULT_ERROR_STRING);
		NWARN("ios_nth_elt:  index out of range!?");
	}

	return np;
}

