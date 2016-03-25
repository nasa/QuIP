
#include "item_type.h"
#include "query_stack.h"

typedef struct undef_sym {
	Item		us_item;
} Undef_Sym;


ITEM_NEW_PROT(Undef_Sym,undef)
ITEM_INIT_PROT(Undef_Sym,undef)
ITEM_CHECK_PROT(Undef_Sym,undef)

