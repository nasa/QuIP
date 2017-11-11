
#include "quip_config.h"

#import "quip_prot.h"
#include "screen_obj.h"

#include "ios_gui.h"

static quipAppDelegate *scrnobjAppDelegate=NULL;
static IOS_Item_Type *scrnobj_itp=NULL;

@implementation Screen_Obj

@synthesize widget_type;
@synthesize flags;
@synthesize selector;
@synthesize selectors;
@synthesize selectorTbl;
@synthesize countTbl;
@synthesize action;
@synthesize content;
@synthesize min;
@synthesize max;
@synthesize val;
@synthesize width;
@synthesize panel;
#ifdef THREAD_SAFE_QUERY
@synthesize qsp;
#endif // THREAD_SAFE_QUERY
@synthesize x;
@synthesize y;
@synthesize children;
@synthesize control;

+(void) initClass
{
	scrnobj_itp=[[IOS_Item_Type alloc] initWithName: STRINGOBJ("Screen_Obj") ];
	/* nop */
}

#ifdef BUILD_FOR_MACOS

+(IOS_List *) getListOfAllItems
{
	if( scrnobj_itp == NULL ) [Screen_Obj initClass];
	return [scrnobj_itp getListOfAllItems];
}

#endif // BUILD_FOR_MACOS

+(quipAppDelegate *) getAppDelegate
{
	return scrnobjAppDelegate;
}

+(int) contextStackDepth
{
	return scrnobj_itp.contextStack.depth;
}

-(void) setSelectorsAtIdx:(int)idx withValue:(const char **)string_arr
{
	selectorTbl[idx] = string_arr;
}

-(const char **) getSelectorsAtIdx:(int)idx
{
	return selectorTbl[idx];
}

-(void) setSelectorInTbl:(int)tbl atIdx:(int)idx withValue:(const char *)string
{
	selectorTbl[tbl][idx] = string;
}

-(void) setCount:(int)n atIdx:(int)idx
{
	countTbl[idx] = n;
}

-(const char *) getSelectorAtIdx:(int)idx fromTbl:(int)tbl
{
	return selectorTbl[tbl][idx];
}
@end

IOS_ITEM_INIT_FUNC(Screen_Obj,scrnobj,0)
IOS_ITEM_CHECK_FUNC(Screen_Obj,scrnobj)
IOS_ITEM_NEW_FUNC(Screen_Obj,scrnobj)
IOS_ITEM_PICK_FUNC(Screen_Obj,scrnobj)
IOS_ITEM_ENUM_FUNC(Screen_Obj,scrnobj)

IOS_List *all_scrnobjs(SINGLE_QSP_ARG_DECL)
{
	return all_ios_items(QSP_ARG  scrnobj_itp);
}

#ifdef MAX_DEBUG
void show_ctx_stack(void)
{
	IOS_Item_Context *icp;
	IOS_List *lp;
	lp = scrnobj_itp.contextStack.list;
	if( lp == NULL ){
		NWARN("show_ctx_stack:  null list!?");
		return;
	}
	IOS_Node *np;
	np = IOS_LIST_HEAD(lp);
	while(np!=NULL){
		icp = (IOS_Item_Context *)IOS_NODE_DATA(np);
		fprintf(stderr,"\t%s\n",icp.name.UTF8String);
		np = IOS_NODE_NEXT(np);
	}
}
#endif // MAX_DEBUG

IOS_Item_Context *_pop_scrnobj_context(SINGLE_QSP_ARG_DECL)
{
	IOS_Item_Context *icp;
	icp = pop_ios_item_context(QSP_ARG  scrnobj_itp );
#ifdef MAX_DEBUG
fprintf(stderr,"pop_scrnobj_context: popped %s\n",IOS_CTX_NAME(icp));
show_ctx_stack();
#endif // MAX_DEBUG
	return icp;
}

void push_scrnobj_context(QSP_ARG_DECL  IOS_Item_Context *icp)
{
	push_ios_item_context(QSP_ARG  scrnobj_itp, icp );
#ifdef MAX_DEBUG
fprintf(stderr,"push_scrnobj_context: pushing %s\n",IOS_CTX_NAME(icp));
show_ctx_stack();
#endif // MAX_DEBUG
}

IOS_Item_Context *create_scrnobj_context(QSP_ARG_DECL  const char *name)
{
	static int sizable_added=0;

	if( scrnobj_itp == NULL ){
		init_scrnobjs();
	}

	if( ! sizable_added ){
		add_ios_sizable(QSP_ARG  scrnobj_itp,&scrnobj_sf, NULL );
		sizable_added = 1;
	}

	return create_ios_item_context(QSP_ARG  scrnobj_itp, name );
}

IOS_Item_Context *top_scrnobj_context(void)
{
	return [scrnobj_itp topContext];
}

