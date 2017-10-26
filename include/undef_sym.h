
#include "item_type.h"

typedef struct undef_sym {
	Item		us_item;
} Undef_Sym;


ITEM_NEW_PROT(Undef_Sym,undef)
ITEM_INIT_PROT(Undef_Sym,undef)
ITEM_CHECK_PROT(Undef_Sym,undef)

#define new_undef(s)	_new_undef(QSP_ARG  s)
#define undef_of(s)	_undef_of(QSP_ARG  s)

