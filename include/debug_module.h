#ifndef _DEBUG_MODULE_H_
#define _DEBUG_MODULE_H_

typedef struct debug_module {
	Item		db_item;
	debug_flag_t	db_mask;
	debug_flag_t	db_flags;
} Debug_Module;

// flag bits

#define DEBUG_SET	1

#define db_name	db_item.item_name

#define NO_DEBUG_MODULE	((Debug_Module *)NULL)

#endif /* ! _DEBUG_MODULE_H_ */

