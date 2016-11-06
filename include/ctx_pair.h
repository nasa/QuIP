#include "item_type.h"

typedef struct context_pair {
	Item_Context *	id_icp;
	Item_Context *	dobj_icp;
} Context_Pair;


/* ContextPair */
#define CP_ID_CTX(cpp)			(cpp)->id_icp
#define SET_CP_ID_CTX(cpp,icp)		(cpp)->id_icp = icp
#define CP_OBJ_CTX(cpp)			(cpp)->dobj_icp
#define SET_CP_OBJ_CTX(cpp,icp)		(cpp)->dobj_icp = icp

#define INIT_CPAIR_PTR(cpp)		cpp=((Context_Pair *)getbuf(sizeof(Context_Pair)));

#define NO_CONTEXT_PAIR	((Context_Pair *)NULL)

