
#ifndef NO_MACROS
#define NO_MACROS

#include "node.h"
#include "items.h"

typedef struct macro {
	Item			m_item;
	const char *		m_text;
	int			m_bytes;	/* length of text */
	int			m_nargs;
	const char **		m_prompt;
	Item_Type **		m_itps;
	int			m_flags;
	const char *		m_filename;	/* where defined */
	int			m_lineno;	/* line number of defn */
} Macro ;

#define	m_name	m_item.item_name

/* flag values */

#define ALLOW_RECURS	1	/* 
				 * done for mac: input redirect is a macro
				 * instead of a command, and this is needed
				 * to allow nested redirect files
				 *
				 * Later allowed under unix too...
				 */

#define RECURSION_FORBIDDEN( mp )			\
							\
	( (( mp )->m_flags & ALLOW_RECURS) == 0 )


#define MACRONAME_PROMPT	"macro name"

#define NO_MACRO	((Macro *)NULL)

#include "query.h"
ITEM_INTERFACE_PROTOTYPES( Macro , macro )
#define PICK_MACRO( s )		_pick_macro(QSP_ARG s )
extern Macro *_pick_macro(QSP_ARG_DECL const char *s );

extern void list_macs(void);
//extern Macro *macro_of(const char *);
extern void _del_macro(QSP_ARG_DECL  Macro *);
extern void mac_stats(SINGLE_QSP_ARG_DECL);
extern void showmac(Macro *mp);
extern Macro *_def_macro(QSP_ARG_DECL  const char *name, int nargs, const char **pmptlist,
	Item_Type **itps, const char *text);
extern void find_macros(QSP_ARG_DECL  const char *);
extern void macro_info(Macro *);
extern List *search_macros(QSP_ARG_DECL  const char *);
extern double dmacroexists(QSP_ARG_DECL  const char *);

#endif	/* NO_MACROS */
