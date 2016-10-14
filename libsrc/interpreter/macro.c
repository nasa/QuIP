
#include "quip_config.h"
#include "quip_prot.h"
#include "item_type.h"
#include "macro.h"

Item_Type * macro_itp=NULL;

ITEM_INIT_FUNC(Macro,macro,0)
ITEM_NEW_FUNC(Macro,macro)
ITEM_CHECK_FUNC(Macro,macro)
//ITEM_PICK_FUNC(Macro,macro)
ITEM_DEL_FUNC(Macro,macro)

const char *macro_text(Macro *mp)
{
	return mp->m_text;
	
}


