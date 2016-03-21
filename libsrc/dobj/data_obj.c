
#include "quip_config.h"
#include "quip_prot.h"
#include "data_obj.h"

Item_Type *dobj_itp=NULL;

const char * dimension_name[N_DIMENSIONS]={
	"component",
	"column",
	"row",
	"frame",
	"sequence"
};

debug_flag_t debug_data=0;

ITEM_INIT_FUNC(Data_Obj,dobj)
ITEM_LIST_FUNC(Data_Obj,dobj)
ITEM_CHECK_FUNC(Data_Obj,dobj)
ITEM_NEW_FUNC(Data_Obj,dobj)		// what does this do?
ITEM_DEL_FUNC(Data_Obj,dobj)		// what does this do?

Item_Context *create_dobj_context(QSP_ARG_DECL  const char *name)
{
	return create_item_context(QSP_ARG  dobj_itp, name);
}

List *dobj_list(SINGLE_QSP_ARG_DECL)
{
	return item_list(QSP_ARG  dobj_itp);
}

void push_dobj_context(QSP_ARG_DECL  Item_Context *icp)
{
	PUSH_ITEM_CONTEXT(dobj_itp,icp);
}

Item_Context *pop_dobj_context(SINGLE_QSP_ARG_DECL)
{
	return POP_ITEM_CONTEXT(dobj_itp);
}

Item_Context *current_dobj_context(SINGLE_QSP_ARG_DECL)
{
	return CURRENT_CONTEXT(dobj_itp);
}

Data_Obj *pick_dobj(QSP_ARG_DECL  const char *pmpt)
{
	return (Data_Obj *) pick_item(QSP_ARG  dobj_itp, pmpt);
}


