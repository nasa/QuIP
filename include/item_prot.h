#ifndef _ITEM_PROT_H_
#define _ITEM_PROT_H_

#include "query_stack.h"
#include "item_type.h"


extern void dump_items(SINGLE_QSP_ARG_DECL);
extern void rename_item(QSP_ARG_DECL  Item_Type *itp,void *ip,char* newname);
extern Item_Context *	create_item_context(QSP_ARG_DECL  Item_Type *,const char *name);
extern Item_Context *	ctx_of(QSP_ARG_DECL  const char *name);
extern List *item_list(QSP_ARG_DECL  Item_Type *itp);
extern Item_Context * pop_item_context(QSP_ARG_DECL  Item_Type *itp);
extern void push_item_context(QSP_ARG_DECL  Item_Type *itp, Item_Context *icp);
extern Item * pick_item(QSP_ARG_DECL  Item_Type *itp, const char *prompt);

#define PUSH_ITEM_CONTEXT(itp,icp)	push_item_context(QSP_ARG  itp,icp)
#define POP_ITEM_CONTEXT(itp)		pop_item_context(QSP_ARG  itp)

extern void	set_del_method(QSP_ARG_DECL  Item_Type *,void (*func)(QSP_ARG_DECL  Item *) );
extern void del_item(QSP_ARG_DECL Item_Type *itp, void *ip );
extern void recycle_item(Item_Type *itp, void *ip );
extern void delete_item_context(QSP_ARG_DECL Item_Context *icp );
extern void delete_item_context_with_callback(QSP_ARG_DECL Item_Context *icp, void (*func)(Item *) );

extern Item_Type *new_item_type(QSP_ARG_DECL  const char *name, int container_type);

extern Item *	new_item(QSP_ARG_DECL  Item_Type * item_type,const char *name,size_t size);
extern void list_items(QSP_ARG_DECL  Item_Type *itp);
extern void list_item_context(QSP_ARG_DECL  Item_Context *icp);
extern List * find_items(QSP_ARG_DECL  Item_Type *itp, const char *frag);
extern Item *item_of(QSP_ARG_DECL  Item_Type *itp, const char *name);
extern Item *	get_item(QSP_ARG_DECL  Item_Type * item_type,const char *name);

extern void print_list_of_items(QSP_ARG_DECL  List *lp);
extern void zombie_item(QSP_ARG_DECL  Item_Type *itp,Item* ip);

extern Item_Type * get_item_type(QSP_ARG_DECL  const char *name);
extern Item_Class * new_item_class(QSP_ARG_DECL  const char *name);
extern void add_items_to_class(Item_Class *icp,Item_Type * itp,void* data,
		Item * (*lookup)(QSP_ARG_DECL  const char *));
extern Item * check_member(QSP_ARG_DECL  Item_Class *icp,const char *name);
extern Item * get_member(QSP_ARG_DECL  Item_Class *icp,const char *name);
extern Member_Info *get_member_info(QSP_ARG_DECL  Item_Class *icp,const char *name);
extern Member_Info *check_member_info(QSP_ARG_DECL  Item_Class *icp,const char *name);

extern void decap(char *,const char *);


#ifdef HAVE_HISTORY
void init_item_hist( QSP_ARG_DECL  Item_Type *itp, const char* prompt );
#endif /* HAVE_HISTORY */

#endif /* ! _ITEM_PROT_H_ */

