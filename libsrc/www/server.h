
#include "query.h"
#include "items.h"

typedef struct server {
	Item	svr_item;
} Server;

#define svr_name	svr_item.item_name

ITEM_INTERFACE_PROTOTYPES(Server,svr)

extern COMMAND_FUNC( svr_menu );

