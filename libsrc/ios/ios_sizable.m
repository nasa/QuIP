#include "quip_config.h"
#include "quip_prot.h"
#include "ios_item.h"
#include "viewer.h"
#include "nexpr.h"

static IOS_Item_Class *sizable_ios_icp=NO_IOS_ITEM_CLASS;
static IOS_Item_Class *positionable_ios_icp=NO_IOS_ITEM_CLASS;
static IOS_Item_Class *interlaceable_ios_icp=NO_IOS_ITEM_CLASS;

#define DECLARE_CLASS_INIT_FUNC(type_stem)					\
static void init_ios_##type_stem##_class(SINGLE_QSP_ARG_DECL)			\
{										\
	if( type_stem##_ios_icp != NO_IOS_ITEM_CLASS ){				\
		WARN("CAUTIOUS:  redundant call to ios class initializer!?");	\
		return;								\
	}									\
	type_stem##_ios_icp = new_ios_item_class(QSP_ARG  "ios_" #type_stem);	\
}

DECLARE_CLASS_INIT_FUNC(sizable)
DECLARE_CLASS_INIT_FUNC(positionable)
DECLARE_CLASS_INIT_FUNC(interlaceable)

#define DECLARE_CLASS_GET_FUNC(type_stem,func_str_type)				\
										\
static func_str_type *get_ios_##type_stem##_functions(QSP_ARG_DECL  IOS_Item *ip)\
{										\
	IOS_Member_Info *mip;							\
										\
	if( type_stem##_ios_icp == NO_IOS_ITEM_CLASS )				\
		init_ios_##type_stem##_class(SINGLE_QSP_ARG);			\
										\
	mip = get_ios_member_info(QSP_ARG  type_stem##_ios_icp,			\
							IOS_ITEM_NAME(ip));	\
										\
	if( mip == NO_IOS_MEMBER_INFO ){					\
		sprintf(ERROR_STRING,						\
	"CAUTIOUS:  class get function %s, missing member info #2",		\
			IOS_ITEM_NAME(ip));					\
		NERROR1(ERROR_STRING);						\
		return NULL;							\
	}									\
										\
	return( (func_str_type *) IOS_MBR_DATA(mip) );			\
}

DECLARE_CLASS_GET_FUNC(sizable,IOS_Size_Functions)
DECLARE_CLASS_GET_FUNC(positionable,IOS_Position_Functions)
DECLARE_CLASS_GET_FUNC(interlaceable,IOS_Interlace_Functions)

double get_object_size(QSP_ARG_DECL  void *ip,int d_index)
{
	if( ip == NULL ) return(0.0);

	// How do we know what type of item it is???
	// In fact, we do not have any idea, and if we treat it like a C Item
	// we run into trouble.
	// A kludge would be to add a magic number to the C Item struct,
	// but that is REALLY ugly.  But we have to do something

	// First try classic...
	if( ITEM_MAGIC( ((Item *)ip) ) == QUIP_ITEM_MAGIC ){
		Size_Functions *sfp;
		sfp = get_sizable_functions(QSP_ARG  (Item *)ip);
		if( sfp != NULL ){
			return( (*sfp->sz_func)(QSP_ARG  (Item *)ip,d_index) );
		}
	} else {
		IOS_Size_Functions *isfp;
		isfp=get_ios_sizable_functions(QSP_ARG  (__bridge IOS_Item *)ip);
		if( isfp != NULL ){
			return( (*isfp->sz_func)(QSP_ARG  (__bridge IOS_Item *)ip,d_index) );
		}
	}
#ifdef CAUTIOUS
	ERROR1("CAUTIOUS:  get_object_size:  shouldn't happen");
	return -1.0;
#endif /* CAUTIOUS */
}

const char * get_object_prec_string(QSP_ARG_DECL  void *ip)
{
	if( ip == NULL ) return(NULL);

	// How do we know what type of item it is???
	// In fact, we do not have any idea, and if we treat it like a C Item
	// we run into trouble.
	// A kludge would be to add a magic number to the C Item struct,
	// but that is REALLY ugly.  But we have to do something

	// First try classic...
	if( ITEM_MAGIC( ((Item *)ip) ) == QUIP_ITEM_MAGIC ){
		Size_Functions *sfp;
		sfp = get_sizable_functions(QSP_ARG  (Item *)ip);
		if( sfp != NULL ){
			return( (*sfp->prec_func)(QSP_ARG  (Item *)ip) );
		}
	} else {
		IOS_Size_Functions *isfp;
		isfp=get_ios_sizable_functions(QSP_ARG  (__bridge IOS_Item *)ip);
		if( isfp != NULL ){
			return( (*isfp->prec_func)(QSP_ARG  (__bridge IOS_Item *)ip) );
		}
	}
#ifdef CAUTIOUS
	ERROR1("CAUTIOUS:  get_object_prec_string:  shouldn't happen");
	return NULL;
#endif /* CAUTIOUS */
}

/*
const char * precision_string(QSP_ARG_DECL  IOS_Item *ip)
{
	IOS_Size_Functions *isfp;
	isfp=get_ios_sizable_functions(QSP_ARG  ip);
	if( isfp != NULL )
		return( (*isfp->prec_func)(QSP_ARG  ip) );
	return NULL;
}
*/

#define DECLARE_CLASS_ADD_FUNC(type_stem,func_str_type)				\
										\
void add_ios_##type_stem(QSP_ARG_DECL  IOS_Item_Type *itp,func_str_type *sfp,	\
			IOS_Item *(*lookup)(QSP_ARG_DECL  const char *))	\
{										\
	if( type_stem##_ios_icp == NO_IOS_ITEM_CLASS )				\
		init_ios_##type_stem##_class(SINGLE_QSP_ARG);			\
	add_items_to_ios_class(type_stem##_ios_icp,itp,sfp,lookup);		\
}

DECLARE_CLASS_ADD_FUNC(sizable,IOS_Size_Functions)
DECLARE_CLASS_ADD_FUNC(positionable,IOS_Position_Functions)

IOS_Item *find_ios_sizable(QSP_ARG_DECL  const char *name)
{
	if( sizable_ios_icp == NO_IOS_ITEM_CLASS )
		init_ios_sizable_class(SINGLE_QSP_ARG);
	
	return( get_ios_member(QSP_ARG  sizable_ios_icp,name) );
}

#define DECLARE_CLASS_CHECK_FUNC(type_stem)					\
										\
IOS_Item *check_ios_##type_stem(QSP_ARG_DECL  const char *name)			\
{										\
	if( type_stem##_ios_icp == NO_IOS_ITEM_CLASS )				\
		init_ios_##type_stem##_class(SINGLE_QSP_ARG);			\
	return( check_ios_member(QSP_ARG  type_stem##_ios_icp,name) );		\
}

DECLARE_CLASS_CHECK_FUNC(sizable)
DECLARE_CLASS_CHECK_FUNC(positionable)
DECLARE_CLASS_CHECK_FUNC(interlaceable)

// BUG?  This should really be in nexpr.y, but that file doesn't know about ios objects...

#define DECLARE_CLASS_EVAL_FUNC(type_stem)					\
										\
static IOS_Item *eval_ios_##type_stem(Scalar_Expr_Node *enp)			\
{										\
	const char *s;								\
										\
	switch(enp->sen_code){							\
		case N_OBJNAME:							\
		case N_QUOT_STR:						\
			s = eval_scalexp_string(enp->sen_child[0]);		\
			return check_ios_##type_stem( DEFAULT_QSP_ARG  s );	\
			break;							\
		/* BUG?  for objects, allow subscripting? */			\
		/* But maybe for now data_obj's are not IOS objects? */		\
		default:							\
			sprintf(DEFAULT_ERROR_STRING,				\
	"Unhandled case in ios eval func!?  (code = %d)",enp->sen_code);	\
			NWARN(DEFAULT_ERROR_STRING);				\
			dump_etree(DEFAULT_QSP_ARG  enp);			\
			break;							\
	}									\
	return NO_IOS_ITEM;							\
}

DECLARE_CLASS_EVAL_FUNC(sizable)
DECLARE_CLASS_EVAL_FUNC(positionable)
DECLARE_CLASS_EVAL_FUNC(interlaceable)


/* These comments came after the first return:

	// Now we know we have a valid item - but we don't know what type...	\
	// We should know this because we found it already!?			\
	// We do have the global class pointer...				\
	// But the class just has a list of item types...			\
	// Where are the functions stored???					\
	// How do we figure out which member function we need???		\

*/

#define DECLARE_CLASS_FCHECK_FUNC(type_stem,func_str_type,member)		\
										\
int check_ios_##type_stem##_func( double *retval, Function *funcp,		\
						Scalar_Expr_Node *argp )	\
{										\
	IOS_Item *ip;								\
										\
	ip = eval_ios_##type_stem(argp);					\
	if( ip == NO_IOS_ITEM ){						\
		/* *retval = -1; */	/* don't really need to set anything */	\
		return 0;							\
	}									\
										\
	func_str_type *isfp =							\
			get_ios_##type_stem##_functions(DEFAULT_QSP_ARG  ip);	\
										\
	if( isfp == NULL ){							\
		NERROR1("Error getting ios functions!?");			\
		return 0;							\
	}									\
	*retval = (*(funcp->fn_u.member) )					\
				(DEFAULT_QSP_ARG  (__bridge Item *) ip );	\
	return 1;								\
}

DECLARE_CLASS_FCHECK_FUNC(sizable,IOS_Size_Functions,sz_func)
DECLARE_CLASS_FCHECK_FUNC(positionable,IOS_Position_Functions,posn_func)
DECLARE_CLASS_FCHECK_FUNC(interlaceable,IOS_Interlace_Functions,il_func)

#ifdef FOOBAR
int check_ios_strv_func( const char **strptr, Function *funcp,
						Scalar_Expr_Node *argp )
{
	IOS_Item *ip;

	ip = eval_ios_sizable(argp);
	if( ip == NO_IOS_ITEM ){
		/* *retval = -1; */	/* don't really need to set anything */
		return 0;
	}

	IOS_Size_Functions *isfp =
			get_ios_sizable_functions(DEFAULT_QSP_ARG  ip);

	if( isfp == NULL ){
		NERROR1("Error getting ios functions!?");
		return 0;
	}
    /*
	*strptr = (*(funcp->fn_u.strv_func) )
				(DEFAULT_QSP_ARG  (__bridge Item *) ip );
     */
    *strptr = (*(funcp->fn_u.strv_func) )
				(DEFAULT_QSP_ARG  IOS_ITEM_NAME(ip) );
    
	return 1;
}

int check_ios_strv2_func( const char **strptr, Function *funcp,
				Scalar_Expr_Node *argp, Scalar_Expr_Node *arg2p )
{
	IOS_Item *ip;

	ip = eval_ios_sizable(argp);
	if( ip == NO_IOS_ITEM ){
		/* *retval = -1; */	/* don't really need to set anything */
		return 0;
	}

	IOS_Size_Functions *isfp =
			get_ios_sizable_functions(DEFAULT_QSP_ARG  ip);

	if( isfp == NULL ){
		NERROR1("Error getting ios functions!?");
		return 0;
	}
    /*
	*strptr = (*(funcp->fn_u.strv_func) )
				(DEFAULT_QSP_ARG  (__bridge Item *) ip );
     */
    *strptr = (*(funcp->fn_u.strv_func) )
				(DEFAULT_QSP_ARG  IOS_ITEM_NAME(ip) );
    
	return 1;
}
#endif // FOOBAR

const char *default_prec_name(QSP_ARG_DECL  void *ip)
{
	return "u_byte";
}

