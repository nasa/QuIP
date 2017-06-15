
#include "quip_config.h"
#include "quip_prot.h"
#include "ios_item.h"

#include "data_obj.h"	/* hunt_obj() */

@implementation IOS_Member_Info
@synthesize member_data;
@synthesize member_itp;
@synthesize member_lookup;
@end

static IOS_Item_Type * ios_icl_itp=NULL;

@implementation IOS_Item_Class
@synthesize member_list;
@synthesize flags;

+(void) initClass
{
#ifdef CAUTIOUS
if( ios_icl_itp != NULL ){
NADVISE("ios item classes already inited!?");
abort();
}
#endif /* CAUTIOUS */
	ios_icl_itp = [[IOS_Item_Type alloc] initWithName:@"IOS_Item_Class"];
}
@end



/*static*/ IOS_ITEM_INIT_PROT(IOS_Item_Class,ios_icl)
/*static*/ IOS_ITEM_NEW_PROT(IOS_Item_Class,ios_icl)

IOS_ITEM_CHECK_PROT(IOS_Item_Class,ios_icl)

/*static*/ IOS_ITEM_INIT_FUNC(IOS_Item_Class,ios_icl,0)
/*static*/ IOS_ITEM_NEW_FUNC(IOS_Item_Class,ios_icl)

IOS_ITEM_CHECK_FUNC(IOS_Item_Class,ios_icl)


IOS_Item_Class * new_ios_item_class(QSP_ARG_DECL  const char *name)
{
	IOS_Item_Class *icp;

	icp = new_ios_icl(QSP_ARG  name);
	if( icp == NULL )
		return(icp);

	SET_IOS_CL_LIST(icp, new_ios_list());
	SET_IOS_CL_FLAGS(icp, NEED_CLASS_CHOICES);
	return(icp);
}

#ifdef FOOBAR
static void dump_ios_node(IOS_Node *np)
{
	fprintf(stderr,"\tnp = 0x%lx  next = 0x%lx  prev = 0x%lx\n",
		(long)np,(long)IOS_NODE_NEXT(np),(long)IOS_NODE_PREV(np));
}

static void dump_ios_list(IOS_List *lp)
{
	if( lp == NULL ){
		fprintf(stderr,"dump_ios_list:  list is NULL!\n");
		return;
	}
	fprintf(stderr,"dump_ios_list:  lp = 0x%lx,  head = 0x%lx,  tail = 0x%lx\n",
		(long)lp,(long)IOS_LIST_HEAD(lp),(long)IOS_LIST_TAIL(lp));
	IOS_Node *np;
	np=IOS_LIST_HEAD(lp);
	while(np!=NULL){
		dump_ios_node(np);
		np=IOS_NODE_NEXT(np);
	}
}

#endif /* FOOBAR */

/* expand the membership of this class */

void add_items_to_ios_class(IOS_Item_Class *icp,IOS_Item_Type * itp,void* data,
		IOS_Item * (*lookup)(QSP_ARG_DECL  const char *))
{
	IOS_Node *np;
	IOS_Member_Info *mip;

	mip = [[IOS_Member_Info alloc] init];
	SET_IOS_MBR_ITP(mip, itp);
	SET_IOS_MBR_DATA(mip, data);
	SET_IOS_MBR_LOOKUP(mip, lookup);
	np = mk_ios_node(mip);
	ios_addTail(IOS_CL_LIST(icp),np);

	/* now make the item type point to this class too */
	if( IOS_IT_CLASS_LIST(itp) == NULL )
		SET_IOS_IT_CLASS_LIST(itp, new_ios_list());
	np = mk_ios_node(icp);
	ios_addTail(IOS_IT_CLASS_LIST(itp),np);

	SET_IOS_CL_FLAG_BITS(icp, NEED_CLASS_CHOICES);
}

/* return the member info struct for this item */

IOS_Member_Info *get_ios_member_info(QSP_ARG_DECL  IOS_Item_Class *icp,const char *name)
{
	IOS_Node *np;
	IOS_Member_Info *mip;

	np = IOS_LIST_HEAD(IOS_CL_LIST(icp));
	while(np!=NULL){
		IOS_Item *ip;

		mip = (IOS_Member_Info*) IOS_NODE_DATA(np);

		if( IOS_MBR_LOOKUP(mip) != NULL ){
			ip = (*IOS_MBR_LOOKUP(mip))(QSP_ARG  name);
		} else {
			ip = ios_item_of(QSP_ARG  IOS_MBR_ITP(mip),name);
		}

		if( ip != NULL ){
			return(mip);
		}

		np = IOS_NODE_NEXT(np);
	}

	sprintf(ERROR_STRING,"No member %s in item class %s",name,IOS_CL_NAME(icp));
	WARN(ERROR_STRING);

	return(NULL);
}

/* return a ptr to the named member of this class */

IOS_Item * check_ios_member(QSP_ARG_DECL  IOS_Item_Class *icp,const char *name)
{
	IOS_Node *np;
	IOS_Item *ip;
	IOS_Member_Info *mip;

	np = IOS_LIST_HEAD(IOS_CL_LIST(icp));
	while(np!=NULL){
		mip = (IOS_Member_Info*) IOS_NODE_DATA(np);

		if( IOS_MBR_LOOKUP(mip) != NULL ){
			ip = (*IOS_MBR_LOOKUP(mip))(QSP_ARG  name);
		} else {
			IOS_Item_Type *itp= IOS_MBR_ITP(mip);

			ip = ios_item_of(QSP_ARG  itp,name);
		}

		if( ip != NULL ){
			return(ip);
		}
		np=IOS_NODE_NEXT(np);
	}
	return NULL;
}

IOS_Item * get_ios_member(QSP_ARG_DECL  IOS_Item_Class *icp,const char *name)
{
	IOS_Item *ip;

	ip = check_ios_member(QSP_ARG  icp, name);
	if( ip == NULL ){
		sprintf(ERROR_STRING,"No member %s found in %s class (iOS)",
				name,IOS_CL_NAME(icp));
		WARN(ERROR_STRING);
	}

	return(ip);
}


