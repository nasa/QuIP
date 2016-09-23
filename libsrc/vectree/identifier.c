#include "quip_config.h"

#include "quip_prot.h"
#include "identifier.h"

Item_Type *id_itp=NULL;

ITEM_INIT_FUNC(Identifier,id,0)
ITEM_GET_FUNC(Identifier,id)
ITEM_CHECK_FUNC(Identifier,id)
ITEM_NEW_FUNC(Identifier,id)
ITEM_DEL_FUNC(Identifier,id)

void restrict_id_context(QSP_ARG_DECL  int flag)
{
	RESTRICT_ITEM_CONTEXT(id_itp,flag);
}

Item_Context *create_id_context(QSP_ARG_DECL  const char *name)
{
	return create_item_context(QSP_ARG  id_itp, name);
}

