
#ifndef _DEBUG_H_
#define _DEBUG_H_

#include "quip_config.h"
#include "item_type.h"

#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif

#ifdef THREAD_SAFE_QUERY
#define WHENCE_L(func)		QS_NAME(qsp),#func,QLEVEL
#define WHENCE(func)		QS_NAME(qsp),#func
#else
#define WHENCE_L(func)		"",#func,QLEVEL
#define WHENCE(func)		"",#func
#endif

#define WHENCE2(func)		"",#func

#include "item_type.h"

#ifdef QUIP_DEBUG

/* change to uint64_t if we run out of modules... */
#define ALL_DEBUG_MODULES	0xffffffff

#include "debug_module.h"

#define MAX_DEBUG_MODULES	(sizeof(debug_flag_t)*8)

/* We can't use add_debug_module() to initialize the system
 * debugging modules, because add_debug_module *uses* the system modules...
 * and sometimes we need to debug before these modules are up and running.
 * We therefore predefine these...
 */

#define DEBUG_SYSTEM

#ifdef DEBUG_SYSTEM

#define AUTO_MODULE_GETBUF	"getbuf"
#define AUTO_MODULE_FREEL	"free_lists"
#define AUTO_MODULE_HASH	"hash_tables"
#define AUTO_MODULE_DICT	"dictionaries"
#define AUTO_MODULE_NODES	"nodes"
#define AUTO_MODULE_ITEMS	"items"
#define AUTO_MODULE_CONTEXTS	"contexts"

#define GETBUF_DEBUG_MASK	1
#define FREEL_DEBUG_MASK	2
#define HASH_DEBUG_MASK		4
#define DICT_DEBUG_MASK		8
#define NODE_DEBUG_MASK		16
#define ITEM_DEBUG_MASK		32
#define CTX_DEBUG_MASK		64

#define N_AUTO_DEBUG_MODULES	7

#else /* ! DEBUG_SYSTEM */

#define GETBUF_DEBUG_MASK	0
#define FREEL_DEBUG_MASK	0
#define HASH_DEBUG_MASK		0
#define DICT_DEBUG_MASK		0
#define NODE_DEBUG_MASK		0
#define ITEM_DEBUG_MASK		0
#define CTX_DEBUG_MASK		0

#define N_AUTO_DEBUG_MODULES	0

#endif /* ! DEBUG_SYSTEM */

#define DEBUG_MSG(flag,msg)	if( debug & flag ) advise("msg");

#else /* ! QUIP_DEBUG */

#define DEBUG_MSG(flag,msg)

#endif /* ! QUIP_DEBUG */

#include <string.h>

// helper macro for assertions with nothing but a message...
// We used to write assert( ! "message" )
// but this produces warnings in xcode...
#define AERROR(s)	strlen(s)==0


#include "node.h"

extern debug_flag_t debug;

extern List *dbm_list(void);
extern void clr_verbose(SINGLE_QSP_ARG_DECL);
extern void set_verbose(SINGLE_QSP_ARG_DECL);
extern void verbtog(SINGLE_QSP_ARG_DECL);

#endif /* _DEBUG_H_ */

