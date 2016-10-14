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

/* Debug_Module */
#define DEBUG_NAME(dbmp)		(dbmp)->db_name
#define DEBUG_MASK(dbmp)		(dbmp)->db_mask
#define SET_DEBUG_MASK(dbmp,m)		(dbmp)->db_mask = m
#define DEBUG_FLAGS(dbmp)		(dbmp)->db_flags
#define SET_DEBUG_FLAGS(dbmp,f)		(dbmp)->db_flags = f
#define CLEAR_DEBUG_FLAG_BITS(dbmp,b)	(dbmp)->db_flags &= ~(b)


#endif /* ! _DEBUG_MODULE_H_ */

