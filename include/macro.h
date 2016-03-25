#ifndef _MACRO_H_
#define _MACRO_H_

//#include "fragment.h"
//#include "query_stack.h"
#include "item_type.h"


typedef struct macro_arg {
	const char *		ma_prompt;
	Item_Type *		ma_itp;
} Macro_Arg;


typedef struct macro {
	Item			m_item;
	Macro_Arg **		m_arg_tbl;
	int			m_n_args;
	const char *		m_text;
	int			m_flags;
	const char *		m_filename;
	int			m_lineno;
} Macro;

#define NO_MACRO ((Macro *)NULL)

// flag bits
#define ALLOW_RECURSION		1
#define MACRO_INVOKED		2	// keep track of what has been used

#define RECURSION_FORBIDDEN( mp )			\
							\
	( (MACRO_FLAGS( mp ) & ALLOW_RECURSION) == 0 )

// macro.c
ITEM_INIT_PROT(Macro,macro)
ITEM_NEW_PROT(Macro,macro)
ITEM_CHECK_PROT(Macro,macro)
ITEM_PICK_PROT(Macro,macro)
ITEM_DEL_PROT(Macro,macro)
#define PICK_MACRO(pmpt)	pick_macro(QSP_ARG  pmpt)

#endif /* ! _MACRO_H_ */

