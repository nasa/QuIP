
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

ITEM_INIT_FUNC(Data_Obj,dobj,0)
ITEM_LIST_FUNC(Data_Obj,dobj)
ITEM_CHECK_FUNC(Data_Obj,dobj)
ITEM_NEW_FUNC(Data_Obj,dobj)		// what does this do?
ITEM_DEL_FUNC(Data_Obj,dobj)		// what does this do?

Item_Context *create_dobj_context(QSP_ARG_DECL  const char *name)
{
	return create_item_context(dobj_itp, name);
}

List *_dobj_list(SINGLE_QSP_ARG_DECL)
{
	return item_list(dobj_itp);
}

void _push_dobj_context(QSP_ARG_DECL  Item_Context *icp)
{
	push_item_context(dobj_itp,icp);
}

Item_Context *_pop_dobj_context(SINGLE_QSP_ARG_DECL)
{
	return pop_item_context(dobj_itp);
}

Item_Context *_current_dobj_context(SINGLE_QSP_ARG_DECL)
{
	return current_context(dobj_itp);
}

Data_Obj *_pick_dobj(QSP_ARG_DECL  const char *pmpt)
{
	return (Data_Obj *) pick_item(dobj_itp, pmpt);
}


