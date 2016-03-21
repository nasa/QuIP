#include "item_type.h"

typedef struct context_pair {
	Item_Context *	id_icp;
	Item_Context *	dobj_icp;
} Context_Pair;


#define NO_CONTEXT_PAIR	((Context_Pair *)NULL)

