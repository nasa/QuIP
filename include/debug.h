
#ifndef NO_DEBUG
#define NO_DEBUG

#include "quip_config.h"

#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif

#include "items.h"

/* change to uint64_t if we run out of modules... */
typedef uint32_t debug_flag_t;

typedef struct db_module {
	Item		db_item;
	debug_flag_t	db_mask;
} Debug_Module;

#define db_name	db_item.item_name


#define MAX_DEBUG_MODULES	(sizeof(debug_flag_t)*8)

/* We can't use add_debug_module() to initialize the system
 * debugging modules, because add_debug_module *uses* the system modules...
 * and sometimes we need to debug before these modules are up and running.
 * We therefore predefine these...
 */

#define DEBUG_SYSTEM

#ifdef DEBUG_SYSTEM

#define AUTO_MODULE_GETBUF	"getbuf"
#define AUTO_MODULE_FREEL	"freel"
#define AUTO_MODULE_HASH	"hash"
#define AUTO_MODULE_NAMESPACE	"namespace"
#define AUTO_MODULE_NODES	"nodes"
#define AUTO_MODULE_ITEMS	"items"
#define AUTO_MODULE_CONTEXTS	"contexts"

#define GETBUF_DEBUG_MASK	1
#define FREEL_DEBUG_MASK	2
#define HASH_DEBUG_MASK		4
#define NAMESP_DEBUG_MASK	8
#define NODE_DEBUG_MASK		16
#define ITEM_DEBUG_MASK		32
#define CTX_DEBUG_MASK		64

#define N_AUTO_DEBUG_MODULES	7

#else /* ! DEBUG_SYSTEM */

#define GETBUF_DEBUG_MASK	0
#define FREEL_DEBUG_MASK	0
#define HASH_DEBUG_MASK		0
#define NAMESP_DEBUG_MASK	0
#define NODE_DEBUG_MASK		0
#define ITEM_DEBUG_MASK		0
#define CTX_DEBUG_MASK		0

#define N_AUTO_DEBUG_MODULES	0

#endif /* ! DEBUG_SYSTEM */


#include "node.h"

extern debug_flag_t debug;

extern int sup_verbose;
#define verbose sup_verbose

extern List *dbm_list(void);
extern void clr_verbose(void);
extern void set_verbose(void);
extern void verbtog(void);

#endif /* NO_DEBUG */

