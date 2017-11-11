
#import "quip_config.h"

#import "ios_item.h"
#import "quip_prot.h"

static IOS_Item_Type *ios_ctx_itp=NULL;

#define IOS_CTX_IT_NAME		"IOS_Context"
#define DEF_IOS_CTX_NAME	"default"

IOS_ITEM_NEW_PROT(IOS_Item_Context,ios_ctx)

@implementation IOS_Item_Context
@synthesize dict;
@synthesize flags;
@synthesize ic_lp;
@synthesize ic_itp;
//@synthesize ic_nsp;		// dict is the namespace???


-(id) initWithName:(NSString *)new_name
{
	self=[super init];
	[self setName: new_name];
	dict = [NSMutableDictionary dictionary];
	flags=0;
	ic_lp=NULL;
	return self;
}

-(IOS_List *) getListOfItems	// IOS_Item_Context
{
	if( flags & LIST_IS_CURRENT ) {
		return ic_lp;
	}
	// free old list if exists?
	ic_lp = [[IOS_List alloc] init];
	NSString *key;
	for(key in dict){
		IOS_Item *ip;
		IOS_Node *np;
		ip = [dict objectForKey : key];
		np = [IOS_Node createNode : ip];
		[ic_lp addTail : np];
	}
	flags |= LIST_IS_CURRENT;
	return ic_lp;
}

-(int) list_items : (FILE *) fp			// IOS_Item_Context
{
	IOS_List *lp;
	int n_listed=0;

	lp = [self getListOfItems];
	IOS_Node *np;

	np = IOS_LIST_HEAD(lp);
	while(np!=NULL){
		IOS_Item *ip;
		ip = IOS_NODE_DATA(np);
		fprintf(fp,"\t%s\n",ip.name.UTF8String);
		np = IOS_NODE_NEXT(np);
		n_listed++;
	}
#ifdef JUST_FOR_TESTING
	// Now dump the dictionary for a check...
	NSEnumerator *enumerator = [dict keyEnumerator];
	if( enumerator == NULL ){
		advise("dict has a null enumerator!?");
		return;
	}
	IOS_Item *ip;

	while ((ip = [enumerator nextObject])) {
		/* code that uses the returned key */
		fprintf(fp,"from dict:\t\t%s\n",IOS_ITEM_NAME(ip));
	}
#endif /* JUST_FOR_TESTING */

	fflush(fp);

	return n_listed;
}

-(IOS_Item *) check: (NSString *) name		/* IOS_Item_Context */
{
	IOS_Item *ip;
	ip = [dict objectForKey : name ];

	return ip;
}

-(int) addItem : (IOS_Item *) ip		/* IOS_Item_Context */
{
	// BUG? should we make sure this item is not already in the dictionary?
	[dict setObject : ip forKey : ip.name];
	flags &= ~LIST_IS_CURRENT;
	return 0;
	
}

-(int) delItem : (IOS_Item *) ip		/* IOS_Item_Context */
{
	[dict removeObjectForKey: ip.name];
	flags &= ~LIST_IS_CURRENT;
	return 0;
}

-(void) show
{
	sprintf(DEFAULT_MSG_STR,"\t%s",[self name].UTF8String);
	prt_msg(DEFAULT_QSP_ARG  DEFAULT_MSG_STR);
}

+(void) initClass	/* IOS_Item_Context */
{
	ios_ctx_itp=[[IOS_Item_Type alloc] initWithName: STRINGOBJ(IOS_CTX_IT_NAME) ];
}

@end	// IOS_Item_Context


@implementation IOS_Item_Type

@synthesize contextStack;
@synthesize it_contexts;
@synthesize flags;
@synthesize it_lp;
@synthesize it_all_lp;
@synthesize it_class_lp;

static IOS_Item_Type *ios_item_type_itp=NULL;

-(void) push : (IOS_Item_Context *) icp	// IOS_Item_Type
{
//fprintf(stderr,"IOS_Item_Type %s push:  pushing context %s\n",
//IOS_ITEM_NAME(self),
//IOS_CTX_NAME(icp));
//fflush(stderr);
	[contextStack push : icp];
	flags &= ~LIST_IS_CURRENT;
}

-(void) showStack
{
	IOS_List *lp;

	lp=contextStack.list;
	if( lp == NULL ) return;
	IOS_Node *np;
	np = IOS_LIST_HEAD(lp);
	while( np != NULL ){
		IOS_Item_Context *icp;
		icp = (IOS_Item_Context *) IOS_NODE_DATA(np);
		// BUG - need to show the context here???
		[icp show];
		np = IOS_NODE_NEXT(np);
	}
}

-(IOS_Item_Context *) pop	// IOS_Item_Type
{
	IOS_Item_Context *icp;
	icp = [contextStack pop];
	flags &= ~LIST_IS_CURRENT;
//fprintf(stderr,"IOS_Item_Type %s pop:  popped %s\n",
//IOS_ITEM_NAME(self),
//IOS_CTX_NAME(icp));
//fflush(stderr);
	return icp;
}

-(int) addItem : (IOS_Item *) ip		// IOS_Item_Type
{
	// BUG? should we make sure this item is not already in the dictionary?
	//[ [[contextStack top] dict] setObject : ip forKey : ip.name];
	int status;
	
	IOS_Item_Context *icp = contextStack.top;
	assert(icp!=NULL);
	status = [icp addItem: ip];
	flags &= ~(LIST_IS_CURRENT|ALL_LIST_IS_CURRENT);
	return status;
}

// what is the return value supposed to indicate?

-(IOS_Item *) delItem : (IOS_Item *) ip		// IOS_Item_Type
{
	IOS_Item_Context *icp;

	if( contextStack == NULL ) NERROR1("delItem:  context stack is NULL!?\n");
	IOS_List *lp=contextStack.list;
	if( lp == NULL ) NERROR1("delItem:  context stack list is NULL!?");
	IOS_Node *np=lp.head;
	if( np == NULL ) {
		NERROR1("delItem:  first context node is NULL!?");
	}

	while(np!=NULL){
		icp=IOS_NODE_DATA(np);
		if( [icp check : ip.name] != NULL ){
			// delete item from this context...
			[icp delItem:ip];
			return ip;
		}
		np=np.next;
	}
	return NULL;
}

// This function only lists the items in the current top context!?

-(void) list : (FILE *) fp			// IOS_Item_Type
{
	IOS_Item_Context *icp = [contextStack top];
	NSMutableDictionary *d=[icp dict];

	/* unsorted listing */
	/*
	for( key in d )
		printf("%s\n",key.UTF8String);
	*/

	/* sorted listing */
	NSArray *keys=[d keysSortedByValueUsingComparator : 
		^(id ip1, id ip2){
	return [((IOS_Item *)ip1).name compare : ((IOS_Item *)ip2).name];
		}
		];

	int i;
	for(i=0;i<keys.count;i++){
		NSString *s=(NSString *)[keys objectAtIndex : i];
		//sprintf(DEFAULT_ERROR_STRING,"\t%s",s.UTF8String);
		//_prt_msg(DEFAULT_QSP_ARG  DEFAULT_ERROR_STRING);
		fprintf(fp,"\t%s\n",s.UTF8String);
	}

	if( keys.count == 0 ){
		sprintf(DEFAULT_ERROR_STRING,"No %ss in existence.",IOS_ITEM_NAME(self));
		//_prt_msg(DEFAULT_QSP_ARG  DEFAULT_ERROR_STRING);
		NWARN(DEFAULT_ERROR_STRING);
	}

	fflush(fp);
}

-(IOS_Item *) check: (NSString *) s		/* IOS_Item_Type */
{
	IOS_Item_Context *icp;
	IOS_Item *ip=NULL;

	assert( contextStack != NULL );

	IOS_List *lp=contextStack.list;
	assert( lp != NULL );

	IOS_Node *np=lp.head;
	assert( np != NULL );

	while(np!=NULL){
		icp=IOS_NODE_DATA(np);
		ip=[icp check : s];
		if( ip != NULL ){
			return ip;
		}
		np=np.next;
	}
	return NULL;
}

-(IOS_Item_Context *) topContext
{
	IOS_Node *np = IOS_LIST_HEAD(contextStack.list);
	if( np == NULL ) return NULL;
	return (IOS_Item_Context *)IOS_NODE_DATA(np);
}

-(IOS_List *) getListOfAllItems		// IOS_Item_Type
{
	if( flags & ALL_LIST_IS_CURRENT ) {
		return it_all_lp;
	}
	
	// Should we release the old list, or will
	// garbage collection take care of it for us?

	it_all_lp = new_ios_list();

	IOS_Node *np = IOS_LIST_HEAD(it_contexts);
	while(np!=NULL){
		IOS_Item_Context *icp =
			(IOS_Item_Context *)IOS_NODE_DATA(np);
		[it_all_lp addListOfItems : [icp getListOfItems] ];
		np = IOS_NODE_NEXT(np);
	}
	flags |= ALL_LIST_IS_CURRENT;
	return it_all_lp;
}

-(IOS_List *) getListOfItems		// IOS_Item_Type
{
	if( flags & LIST_IS_CURRENT ) {
		return it_lp;
	}
	
	// Should we release the old list, or will
	// garbage collection take care of it for us?

	it_lp = new_ios_list();

	IOS_Node *np = IOS_LIST_HEAD(contextStack.list);
	while(np!=NULL){
		IOS_Item_Context *icp =
			(IOS_Item_Context *)IOS_NODE_DATA(np);
		[it_lp addListOfItems : [icp getListOfItems] ];
		np = IOS_NODE_NEXT(np);
	}
	flags |= LIST_IS_CURRENT;
	return it_lp;
}

-(IOS_Item *) pick : (Query_Stack *) qsp
{
	return [self pick : qsp withPrompt: self.name];
}

-(IOS_Item *) pick: (Query_Stack *) qsp withPrompt : (NSString *) prompt
{
	NSString *str=STRINGOBJ( nameof(QSP_ARG  prompt.UTF8String) );
	IOS_Item *ip=[self check : str];
	if( ip != NULL ) return ip;

	// Now print a useful message
	NSString *msg=[[NSString alloc] initWithFormat : @"No %@ named \"%@\" in existence.",self.name,str];
	//warn(QSP_ARG  msg.UTF8String );
	WARN(msg.UTF8String );

	advise("Legal values are:");
    [self list : tell_errfile()];
	return NULL;
}

// like check, but report an error if item does not exist
-(IOS_Item *) get: (NSString *) name
{
	IOS_Item *ip;
	ip = [self check : name];
	if( ip == NULL ){
		NSString *msg=[[NSString alloc] initWithFormat :
	@"No %@ \"%@\"",self.name,name ];
		NWARN(msg.UTF8String);
	}
	return ip;
}

-(IOS_Stack *) getContextStack
{
	return contextStack;
}

-(void) addToContextList: (IOS_Item_Context *) icp
{
	IOS_Node *np;
	np = mk_ios_node(icp);
	ios_addTail(it_contexts,np); 
}

-(id) initWithName : (NSString *) s	// IOS_Item_Type
{
	self=[super init];

#ifdef CAUTIOUS
	IOS_Item_Type *itp;
	itp = (IOS_Item_Type *)[ios_item_type_itp check : s ];
	assert( itp == NULL );
#endif /* CAUTIOUS */

	IOS_Item_Context *default_context=[[IOS_Item_Context alloc] initWithName:[s stringByAppendingString:@".default"] ];
	contextStack = [[IOS_Stack alloc] init];
	[ contextStack push : default_context ];
	it_contexts = new_ios_list();
	[ self addToContextList : default_context ];
	flags=0;
	[self setName : s];

	// could this ever be true here??
	if( ios_item_type_itp == NULL )
		[IOS_Item_Type initClass];
	[ios_item_type_itp addItem : self ];

	return self;
}

+(IOS_Item_Type *) get : (NSString *) name
{
	return (IOS_Item_Type *)[ios_item_type_itp get : name];
}

+(void) list : (FILE *) fp
{
	[ ios_item_type_itp list : fp ];
}

+(void) initClass	/* IOS_Item_Type */
{
	// Can't use initWithName, must avoid infinite recursion
	ios_item_type_itp = [[IOS_Item_Type alloc] init];
//	[ios_item_type_itp setName : @"item_type" ];
	[ios_item_type_itp setName : @"IOS_Item_Type" ];
	[ios_item_type_itp setContextStack: [[IOS_Stack alloc]init] ];
	[ios_item_type_itp setFlags:0];
	
	// Can't use initWithName ?
	IOS_Item_Context *icp=[[IOS_Item_Context alloc] init];
	[icp setName: @"IOS_Item_Type.default"];
	[icp setDict:[NSMutableDictionary dictionary]];
	[icp setFlags:0];
	[icp setIc_lp:NULL];
	
	[ios_item_type_itp.contextStack push: icp];
	
	[ios_item_type_itp addItem : ios_item_type_itp ];
}

@end	// end of IOS_Item_Type methods

IOS_Item *pick_ios_item(QSP_ARG_DECL  IOS_Item_Type *itp, const char *prompt)
{
	return [itp pick: THIS_QSP  withPrompt: STRINGOBJ(prompt) ];
}


IOS_Item_Context * pop_ios_item_context(QSP_ARG_DECL IOS_Item_Type *itp)
{
	IOS_Item_Context *icp;

	icp = [itp pop];
	return icp;
}

void ios_set_del_method(QSP_ARG_DECL  IOS_Item_Type *itp,void (*func)(QSP_ARG_DECL  IOS_Item *) )
{
	WARN("Oops, set_del_method is not implemented!?");
}

IOS_ITEM_CHECK_FUNC(IOS_Item_Context,ios_ctx)

IOS_Item_Context * create_ios_item_context( QSP_ARG_DECL  IOS_Item_Type *itp, const char* name )
{
	IOS_Item_Context *icp;
	char cname[LLEN];

	/* maybe we should have contexts for contexts!? */

	sprintf(cname,"%s.%s",IOS_ITEM_NAME(itp),name);

	if( (!strcmp(IOS_ITEM_NAME(itp),IOS_CTX_IT_NAME)) && !strcmp(name,DEF_IOS_CTX_NAME) ){
		//static IOS_Item_Context first_context;

		/* can't use new_ctx()
		 *
		 * Why not???
		 */
		//icp = &first_context;
		icp = NEW_IOS_ITEM_CONTEXT;
		SET_IOS_CTX_NAME(icp,savestr(cname));
		SET_IOS_CTX_IT(icp,itp);
		SET_IOS_CTX_FLAGS(icp, 0);
		/* BUG?  not in the context database?? */
		return(icp);
	}

	/* Create an item type for contexts.
	 *
	 * Because new_item_type() calls create_item_text for the default
	 * context, we have the special case above...
	 */

	if( ios_ctx_itp == NULL )
		[IOS_Item_Context initClass];

#ifdef QUIP_DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  creating context %s",
WHENCE(create_ios_item_context),cname);
advise(ERROR_STRING);
}
#endif	/* QUIP_DEBUG */

	icp = new_ios_ctx(QSP_ARG  cname);

	if( icp == NULL ){
		printf("OOPS couldn't create new item context!?\n");
		return(icp);
	}

	SET_IOS_CTX_IT(icp,itp);
	SET_IOS_CTX_FLAGS(icp,0);

	[itp addToContextList:icp];
	return(icp);
}

IOS_List *ios_item_list(QSP_ARG_DECL  IOS_Item_Type *itp)
{
	return [itp getListOfItems];
}

IOS_List *all_ios_items(QSP_ARG_DECL  IOS_Item_Type *itp)
{
	return [itp getListOfAllItems];
}

void delete_ios_item_context(QSP_ARG_DECL  IOS_Item_Context *icp)
{
	// deallocate memory here, and remove from the dictionary

	// remove from name database
	/*icp =*/ del_ios_ctx(QSP_ARG  icp);

	// release other resources?
	// there don't appear to be any, perhaps we don't need this?

	// BUG?  there is a special case in new_ios_ctx
	// for the first context - we assume that it will
	// never be deleted, but we ought to make sure that
	// nothing bad will happen if we do!
}

IOS_ITEM_INIT_FUNC(IOS_Item_Type,ios_item_type,0)
IOS_ITEM_NEW_FUNC(IOS_Item_Type,ios_item_type)
IOS_ITEM_CHECK_FUNC(IOS_Item_Type,ios_item_type)
IOS_ITEM_INIT_FUNC(IOS_Item_Context,ios_ctx,0)
IOS_ITEM_NEW_FUNC(IOS_Item_Context,ios_ctx)
IOS_ITEM_DEL_FUNC(IOS_Item_Context,ios_ctx)



@implementation IOS_Item

@synthesize name;

-(IOS_Item *) initWithName : (NSString *) s
{
	self=[super init];
	name=s;
	return self;
}


@end	// IOS_Item

void push_ios_item_context(QSP_ARG_DECL  IOS_Item_Type *itp, IOS_Item_Context *icp)
{
	[itp push:icp];
}

void del_ios_item(QSP_ARG_DECL  IOS_Item_Type *itp, IOS_Item *ip)
{
	if( [itp delItem:ip] == NULL ){
		sprintf(ERROR_STRING,"del_ios_item:  error deleting %s named %s",
			IOS_IT_NAME(itp),IOS_ITEM_NAME(ip));
		WARN(ERROR_STRING);
	}
}

IOS_Item *ios_item_of(QSP_ARG_DECL  IOS_Item_Type *itp, const char *s)
{
	return [itp check:STRINGOBJ(s)];
}

