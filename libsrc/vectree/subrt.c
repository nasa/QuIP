
#include "quip_config.h"
#include "quip_prot.h"
#include "subrt.h"

Item_Type *subrt_itp=NULL;

ITEM_INIT_FUNC(Subrt,subrt,0)
ITEM_CHECK_FUNC(Subrt,subrt)
ITEM_PICK_FUNC(Subrt,subrt)
ITEM_NEW_FUNC(Subrt,subrt)
ITEM_LIST_FUNC(Subrt,subrt)
ITEM_ENUM_FUNC(Subrt,subrt)

//Subrt *subrt_of(QSP_ARG_DECL  const char *name)
//{
	//return (Subrt *)item_of(QSP_ARG  subrt_itp ,name);
//}

/*
Subrt *new_subrt(QSP_ARG_DECL  const char *name)
{
	Subrt *srp;
	srp=getbuf(sizeof(Subrt));
	srp->sr_item.item_name = savestr(name);
	add_item(QSP_ARG  subrt_itp, srp, NO_NODE);
	return srp;
}
*/

/*
Subrt *pick_subrt(QSP_ARG_DECL const char *prompt)
{
	return (Subrt *)pick_item(QSP_ARG  subrt_itp, prompt);
}

void list_subrts(SINGLE_QSP_ARG_DECL)
{
	list_items(QSP_ARG  subrt_itp);
}

List *list_of_subrts(SINGLE_QSP_ARG_DECL)
{
	return item_list(QSP_ARG  subrt_itp);
}
*/

