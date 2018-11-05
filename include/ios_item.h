#ifndef _IOS_ITEM_H_
#define _IOS_ITEM_H_

#include "ios_stack.h"
#include "ios_list.h"

#include "query_stack.h"

@class IOS_Item_Type;


@interface IOS_Item: NSObject

@property (retain) NSString *name;

-(IOS_Item *) initWithName : (NSString *) s;

@end

#define IOS_ITEM_NAME(ip)	ip.name.UTF8String
#define IOS_ITEM_TYPE_NAME(itp)		IOS_ITEM_NAME(itp)


@interface IOS_Item_Context : IOS_Item
@property (retain) NSMutableDictionary *	dict;
@property (retain) IOS_List *			ic_lp;
@property (retain) IOS_Item_Type *		ic_itp;
@property int 					flags;

-(IOS_List *) getListOfItems;
-(IOS_Item *) check : (NSString *) name;
-(int) list_items : (FILE *) fp;
@end

@interface IOS_Item_Type : IOS_Item

@property (retain) IOS_Stack *	contextStack;
@property int			flags;
@property (retain) IOS_List *	it_lp;	// list of items
@property (retain) IOS_List *	it_all_lp;
@property (retain) IOS_List *	it_class_lp;
// When we want to search all the contexts (even those not on the stack)
// It might be nice to have a list of all the contexts...
@property (retain) IOS_List *	it_contexts;

// flag bits
#define LIST_IS_CURRENT		1
#define ALL_LIST_IS_CURRENT	2

-(void) push : (IOS_Item_Context *) icp;
-(void) addToContextList : (IOS_Item_Context *) icp;
-(void) showStack;
-(IOS_Item_Context *) pop;
-(IOS_Item_Context *) topContext;
-(int) addItem : (IOS_Item *) ip;
-(IOS_Item *) delItem : (IOS_Item *) ip;
-(void) list : (FILE *) fp;

-(IOS_Item *) get : (NSString *) name;
-(IOS_Item *) check : (NSString *) name;
-(IOS_Item *) pick : (Query_Stack *) qsp;
-(IOS_Item *) pick : (Query_Stack *) qsp withPrompt : (NSString *) pmpt;
-(IOS_Stack *) getContextStack;
-(IOS_List *) getListOfItems;
-(IOS_List *) getListOfAllItems;

-(id) initWithName : (NSString *) name;

+(IOS_Item_Type *) get : (NSString *) name;
+(void) list : (FILE *) fp;
+(void) initClass;

@end

#define IOS_IT_NAME(itp)	(itp).name.UTF8String
#define IOS_IT_CLASS_LIST(itp)	(itp).it_class_lp
#define SET_IOS_IT_CLASS_LIST(itp,v)	(itp).it_class_lp = v

/*
#define RESTRICT_IOS_ITEM_CONTEXT(itp,flag)			\
							\
	IOS_CTX_RSTRCT_FLAG(itp)=flag;
*/

#define RESTRICT_IOS_ITEM_CONTEXT(itp,flag)			\
							\
	SET_IOS_IT_FLAGS(itp,flag);

extern IOS_Item_Context *	create_ios_item_context(QSP_ARG_DECL  IOS_Item_Type *,const char *name);
extern IOS_List *ios_item_list(QSP_ARG_DECL  IOS_Item_Type *itp);
extern IOS_List *all_ios_items(QSP_ARG_DECL  IOS_Item_Type *itp);
extern IOS_Item_Context * pop_ios_item_context(QSP_ARG_DECL  IOS_Item_Type *itp);
extern void push_ios_item_context(QSP_ARG_DECL  IOS_Item_Type *itp, IOS_Item_Context *icp);
extern IOS_Item * pick_ios_item(QSP_ARG_DECL  IOS_Item_Type *itp, const char *prompt);
extern IOS_Item * ios_item_of(QSP_ARG_DECL  IOS_Item_Type *itp, const char *name);

extern void	ios_set_del_method(QSP_ARG_DECL  IOS_Item_Type *,void (*func)(QSP_ARG_DECL  IOS_Item *) );
extern void del_ios_item(QSP_ARG_DECL IOS_Item_Type *itp, IOS_Item *ip );
extern void delete_ios_item_context(QSP_ARG_DECL IOS_Item_Context *icp );

//extern IOS_Item_Type *_new_ios_item_type(QSP_ARG_DECL  const char *name);
extern IOS_Item_Context *_new_ctx(QSP_ARG_DECL  const char *name);
extern IOS_Item *_new_ios_item(QSP_ARG_DECL  IOS_Item_Type *itp, const char *name, int size);

//#define new_ios_item_type(name) _new_ios_item_type(QSP_ARG  name)
#define new_ctx(name) _new_ctx(QSP_ARG  name)
#define new_ios_item(itp, name, size) _new_ios_item(QSP_ARG  itp, name, size)


#define IOS_ITEM_INIT_PROT(type,stem)		\
								\
extern void _init_##stem##s(SINGLE_QSP_ARG_DECL);


#define IOS_ITEM_NEW_PROT(type,stem)				\
								\
extern type *_new_##stem(QSP_ARG_DECL  const char *name);


#define IOS_ITEM_PICK_PROT(type,stem)				\
								\
extern type *_pick_##stem(QSP_ARG_DECL  const char *pmpt);


#define IOS_ITEM_DEL_PROT(type,stem)				\
								\
extern type *_del_##stem(QSP_ARG_DECL  type *ip);


#define IOS_ITEM_GET_PROT(type,stem)				\
								\
extern type *_get_##stem(QSP_ARG_DECL  const char *name);


#define IOS_ITEM_CHECK_PROT(type,stem)				\
								\
extern type *_##stem##_of(QSP_ARG_DECL  const char *name);


#define IOS_ITEM_ENUM_PROT(type,stem)				\
								\
extern IOS_List *_##stem##_list(SINGLE_QSP_ARG_DECL );


#define IOS_ITEM_LIST_PROT(type,stem)				\
								\
extern void _list_##stem##s(QSP_ARG_DECL  FILE *);



#define IOS_ITEM_NEW_FUNC(type,stem)				\
								\
type *_new_##stem(QSP_ARG_DECL  const char *name)		\
{								\
	IOS_Item *ip;						\
								\
	if( stem##_itp == NULL ){				\
		_##init_##stem##s(SINGLE_QSP_ARG);		\
	}							\
    ip = _##stem##_of(QSP_ARG  name);				\
	if( ip != NULL ){					\
		sprintf(ERROR_STRING,"new_%s:  \"%s\" already exists!?", \
			#stem,name);				\
		WARN(ERROR_STRING);				\
		return((type *)NULL);				\
	}							\
	type *stem##_p=[[type alloc] initWithName:		\
			STRINGOBJ(name) ];			\
	[stem##_itp addItem: stem##_p];				\
	return stem##_p;					\
}

#define IOS_ITEM_DEL_FUNC(type,stem)				\
								\
type *_del_##stem(QSP_ARG_DECL  type *ip)			\
{								\
	type *stem##_p;						\
								\
	stem##_p = stem##_of(ip.name.UTF8String);	\
	if( stem##_p == NULL ){					\
		/* BUG print warning */				\
		return stem##_p;				\
	}							\
	del_ios_item(stem##_itp, stem##_p );		\
	return stem##_p;					\
}


#define IOS_ITEM_CHECK_FUNC(type,stem)				\
								\
type *_##stem##_of(QSP_ARG_DECL  const char *name)			\
{								\
	return (type *)[stem##_itp check: STRINGOBJ(name) ];	\
}

// This init_func definition used to set up the itp,
// but this is already done in the initClass method!?
#define IOS_ITEM_INIT_FUNC(type,stem,container_type)		\
								\
void _init_##stem##s(SINGLE_QSP_ARG_DECL)			\
{								\
	[type initClass];					\
	/*stem##_itp = [[IOS_Item_Type alloc]			\
		initWithName : STRINGOBJ(#type) ];*/		\
}

#define IOS_ITEM_GET_FUNC(type,stem)				\
								\
type *_get_##stem(QSP_ARG_DECL  const char *name)		\
{								\
	return (type *)[stem##_itp get: STRINGOBJ(name) ];	\
}

#define IOS_ITEM_PICK_FUNC(type,stem)				\
								\
type *_pick_##stem(QSP_ARG_DECL  const char *pmpt)		\
{								\
	if( stem##_itp == NULL )				\
		_##init_##stem##s(SINGLE_QSP_ARG);		\
	return (type *)pick_ios_item(stem##_itp, pmpt);		\
}

#define IOS_ITEM_LIST_FUNC(type,stem)				\
								\
void _list_##stem##s(QSP_ARG_DECL  FILE *fp)			\
{								\
	if( stem##_itp == NULL )				\
		init_##stem##s();				\
	[stem##_itp list:fp];					\
}

#define IOS_ITEM_ENUM_FUNC(type,stem)				\
								\
IOS_List * _##stem##_list(SINGLE_QSP_ARG_DECL)			\
{								\
	return [stem##_itp getListOfItems];			\
}

#define STRINGOBJ(s)	[[NSString alloc] initWithUTF8String:(s)]

#define NEW_IOS_ITEM_CONTEXT	[[IOS_Item_Context alloc] init]

#define IOS_CTX_NAME(icp)	icp.name.UTF8String

#define SET_IOS_CTX_NAME(icp,s)	[icp setName: STRINGOBJ(s) ]
#define SET_IOS_CTX_IT(icp,itp)	[icp setIc_itp: itp ]
#define SET_IOS_CTX_FLAGS(icp,f)	[icp setFlags: f ]


IOS_ITEM_INIT_PROT(IOS_Item_Type,ios_item_type)
IOS_ITEM_NEW_PROT(IOS_Item_Type,ios_item_type)
IOS_ITEM_CHECK_PROT(IOS_Item_Type,ios_item_type)

#define init_ios_item_types()		_init_ios_item_types(SINGLE_QSP_ARG)
#define new_ios_item_type(s)		_new_ios_item_type(QSP_ARG  s)
#define ios_item_type_of(s)		_ios_item_type_of(QSP_ARG  s)


IOS_ITEM_INIT_PROT(IOS_Item_Context,ios_ctx)
IOS_ITEM_NEW_PROT(IOS_Item_Context,ios_ctx)
IOS_ITEM_CHECK_PROT(IOS_Item_Context,ios_ctx)
IOS_ITEM_DEL_PROT(IOS_Item_Context,ios_ctx)

#define init_ios_ctxs()		_init_ios_ctxs(SINGLE_QSP_ARG)
#define new_ios_ctx(s)		_new_ios_ctx(QSP_ARG  s)
#define ios_ctx_of(s)		_ios_ctx_of(QSP_ARG  s)
#define del_ios_ctx(s)		_del_ios_ctx(QSP_ARG  s)

@interface IOS_Item_Class : IOS_Item

@property (retain) IOS_List *	member_list;	
@property int			flags;

+(void) initClass;

@end

// flag bits
// these are the same as regular classes...
#define NEED_CLASS_CHOICES	1

#define IOS_CL_NAME(clp)	(clp).name.UTF8String
#define IOS_CL_LIST(clp)	(clp).member_list
#define IOS_CL_FLAGS(clp)	(clp).flags

#define SET_IOS_CL_LIST(clp,v)	(clp).member_list = v
#define SET_IOS_CL_FLAGS(clp,v)	(clp).flags = v
#define SET_IOS_CL_FLAG_BITS(clp,v)	SET_IOS_CL_FLAGS( clp, IOS_CL_FLAGS(clp) | (v) )
#define CLEAR_IOS_CL_FLAG_BITS(clp,v)	SET_IOS_CL_FLAGS( clp, IOS_CL_FLAGS(clp) & ~(v) )

@interface IOS_Member_Info : NSObject
@property (retain) IOS_Item_Type *	member_itp;
@property void *			member_data;	// this is used a table of size functions, for example
@property IOS_Item *		(*member_lookup)(QSP_ARG_DECL  const char *);
@end

#define IOS_MBR_ITP(mip)	(mip).member_itp
#define IOS_MBR_DATA(mip)	(mip).member_data
#define IOS_MBR_LOOKUP(mip)	(mip).member_lookup

#define SET_IOS_MBR_ITP(mip,v)		(mip).member_itp = v
#define SET_IOS_MBR_DATA(mip,v)		(mip).member_data = v
#define SET_IOS_MBR_LOOKUP(mip,v)	(mip).member_lookup = v

extern IOS_Item_Class * new_ios_item_class(QSP_ARG_DECL  const char *name);
extern void add_items_to_ios_class(IOS_Item_Class *icp,IOS_Item_Type * itp,void* data,
		IOS_Item * (*lookup)(QSP_ARG_DECL  const char *));
extern IOS_Item * get_ios_member(QSP_ARG_DECL  IOS_Item_Class *icp,const char *name);
extern IOS_Item * check_ios_member(QSP_ARG_DECL  IOS_Item_Class *icp,const char *name);
extern IOS_Member_Info *get_ios_member_info(QSP_ARG_DECL  IOS_Item_Class *icp,const char *name);
extern IOS_Member_Info *get_ios_member_info(QSP_ARG_DECL  IOS_Item_Class *icp,const char *name);

#endif /* ! _IOS_ITEM_H_ */
