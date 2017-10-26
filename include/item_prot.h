#ifndef _ITEM_PROT_H_
#define _ITEM_PROT_H_

#ifdef __cplusplus
extern "C" {
#endif


#include <stdio.h>

#include "quip_fwd.h"

extern void _dump_items(SINGLE_QSP_ARG_DECL);
extern void _rename_item(QSP_ARG_DECL  Item_Type *itp,void *ip,char* newname);
extern Item_Context *	_create_item_context(QSP_ARG_DECL  Item_Type *,const char *name);
extern Item_Context *	_ctx_of(QSP_ARG_DECL  const char *name);
extern List *_item_list(QSP_ARG_DECL  Item_Type *itp);
extern List *_current_item_list(QSP_ARG_DECL  Item_Type *itp);
extern Item_Context * _pop_item_context(QSP_ARG_DECL  Item_Type *itp);
extern void _push_item_context(QSP_ARG_DECL  Item_Type *itp, Item_Context *icp);
extern Item * _pick_item(QSP_ARG_DECL  Item_Type *itp, const char *prompt);

#define dump_items()			_dump_items(SINGLE_QSP_ARG)
#define rename_item(itp,ip,name)	_rename_item(QSP_ARG  itp,ip,name)
#define create_item_context(itp,name)	_create_item_context(QSP_ARG  itp,name)
#define ctx_of(name)	_ctx_of(QSP_ARG  name)
#define item_list(type)			_item_list(QSP_ARG  type)
#define current_item_list(type)		_current_item_list(QSP_ARG  type)
#define pop_item_context(itp)		_pop_item_context(QSP_ARG  itp)
#define push_item_context(itp,icp)	_push_item_context(QSP_ARG  itp,icp)
#define pick_item(type,prompt)		_pick_item(QSP_ARG  type,prompt)

extern void	_set_del_method(QSP_ARG_DECL  Item_Type *,void (*func)(QSP_ARG_DECL  Item *) );
extern void _del_item(QSP_ARG_DECL Item_Type *itp, void *ip );
extern void recycle_item(Item_Type *itp, void *ip );
extern void _delete_item_context(QSP_ARG_DECL Item_Context *icp );
extern void _delete_item_context_with_callback(QSP_ARG_DECL Item_Context *icp, void (*func)(Item *) );

#define set_del_method(itp,func)	_set_del_method(QSP_ARG  itp,func)
#define del_item(itp,ip)		_del_item(QSP_ARG itp, ip )
#define delete_item_context(icp)	_delete_item_context(QSP_ARG icp )
#define delete_item_context_with_callback(icp,func)	_delete_item_context_with_callback(QSP_ARG icp, func)

extern Item_Type *_new_item_type(QSP_ARG_DECL  const char *name, int container_type);
extern Item *	_new_item(QSP_ARG_DECL  Item_Type * item_type,const char *name,size_t size);
extern void _report_invalid_pick(QSP_ARG_DECL  Item_Type *itp, const char *s);
extern void _list_items(QSP_ARG_DECL  Item_Type *itp, FILE *fp);
extern void _list_item_context(QSP_ARG_DECL  Item_Context *icp);
extern List * _find_items(QSP_ARG_DECL  Item_Type *itp, const char *frag);
extern Item *_item_of(QSP_ARG_DECL  Item_Type *itp, const char *name);
extern Item *	_get_item(QSP_ARG_DECL  Item_Type * item_type,const char *name);
extern void _print_list_of_items(QSP_ARG_DECL  List *lp, FILE *fp);
extern void _zombie_item(QSP_ARG_DECL  Item_Type *itp,Item* ip);

#define new_item_type(name,type_code)	_new_item_type(QSP_ARG  name,type_code)
#define new_item(itp,name,size)	_new_item(QSP_ARG  itp,name,size)
#define report_invalid_pick(itp,s)	 _report_invalid_pick(QSP_ARG  itp, s)
#define list_items(itp,fp)		_list_items(QSP_ARG  itp,fp)
#define list_item_context(icp)		_list_item_context(QSP_ARG  icp)
#define find_items(itp,frag)		_find_items(QSP_ARG  itp,frag)
#define item_of(itp,name)		_item_of(QSP_ARG  itp,name)
#define get_item(itp,name)		_get_item(QSP_ARG  itp,name)
#define print_list_of_items(lp,fp)	_print_list_of_items(QSP_ARG  lp,fp)
#define zombie_item(itp,ip)		_zombie_item(QSP_ARG  itp,ip)


extern Item_Type * _get_item_type(QSP_ARG_DECL  const char *name);
extern Item_Class * _new_item_class(QSP_ARG_DECL  const char *name);
extern Item * _check_member(QSP_ARG_DECL  Item_Class *icp,const char *name);
extern Item * _get_member(QSP_ARG_DECL  Item_Class *icp,const char *name);
extern Member_Info *_get_member_info(QSP_ARG_DECL  Item_Class *icp,const char *name);
extern Member_Info *_check_member_info(QSP_ARG_DECL  Item_Class *icp,const char *name);

extern void add_items_to_class(Item_Class *icp,Item_Type * itp,void* data,
		Item * (*lookup)(QSP_ARG_DECL  const char *));

#define get_item_type(name)	_get_item_type(QSP_ARG  name)
#define new_item_class(name)	_new_item_class(QSP_ARG  name)
#define check_member(icp,name)	_check_member(QSP_ARG  icp,name)
#define get_member(icp,name)	_get_member(QSP_ARG  icp,name)
#define get_member_info(icp,name)	_get_member_info(QSP_ARG  icp,name)
#define check_member_info(icp,name)	_check_member_info(QSP_ARG  icp,name)

extern void decap(char *,const char *);


#ifdef HAVE_HISTORY
void init_item_hist( QSP_ARG_DECL  Item_Type *itp, const char* prompt );
#endif /* HAVE_HISTORY */

#ifdef __cplusplus
}
#endif

#endif /* ! _ITEM_PROT_H_ */

