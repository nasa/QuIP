#ifndef _MACRO_H_
#define _MACRO_H_

//#include "fragment.h"
//#include "query_stack.h"
#include "item_type.h"


struct macro_arg {
	const char *		ma_prompt;
	Item_Type *		ma_itp;
} ;

/* Macro argument */
#define MA_PROMPT(map)			map->ma_prompt
#define MA_ITP(map)			map->ma_itp


struct macro {
	Item			m_item;
	Macro_Arg **		m_arg_tbl;
	int			m_n_args;
	const char *		m_text;
	int			m_flags;
	const char *		m_filename;
	int			m_lineno;
} ;

#define INIT_MACRO_PTR(mp)		mp=((Macro *)getbuf(sizeof(Macro)));

/* Macro macros */
#define SET_MACRO_NAME(mp,s)		(mp)->m_item.item_name=s
#define SET_MACRO_N_ARGS(mp,n)		(mp)->m_n_args=n
#define SET_MACRO_TEXT(mp,t)		(mp)->m_text=t
#define SET_MACRO_FLAGS(mp,f)		(mp)->m_flags=f
#define SET_MACRO_ARG_TBL(mp,tbl)	(mp)->m_arg_tbl=tbl
#define SET_MACRO_FILENAME(mp,s)	(mp)->m_filename = s
#define SET_MACRO_LINENO(mp,n)		(mp)->m_lineno = n

#define MACRO_NAME(mp)			(mp)->m_item.item_name
#define MACRO_N_ARGS(mp)		(mp)->m_n_args
#define MACRO_TEXT(mp)			(mp)->m_text
#define MACRO_FLAGS(mp)			(mp)->m_flags
#define MACRO_ARG_TBL(mp)		(mp)->m_arg_tbl
#define MACRO_ARG(mp,idx)		MACRO_ARG_TBL(mp)[idx]
#define MACRO_PROMPT(mp,idx)		MA_PROMPT(MACRO_ARG(mp,idx))
#define MACRO_ITPS(mp)			(mp)->m_itps
#define MACRO_FILENAME(mp)		(mp)->m_filename
#define MACRO_LINENO(mp)		(mp)->m_lineno
#define SET_MACRO_FLAG_BITS(mp,f)	(mp)->m_flags |= f
#define CLEAR_MACRO_FLAG_BITS(mp,f)	(mp)->m_flags &= ~(f)

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

