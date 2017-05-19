
#include "quip_config.h"
#include "quip_prot.h"
#include "item_type.h"
#include "macro.h"
#include "query_prot.h"

Item_Type * macro_itp=NULL;

ITEM_INIT_FUNC(Macro,macro,0)
ITEM_NEW_FUNC(Macro,macro)
ITEM_CHECK_FUNC(Macro,macro)
//ITEM_PICK_FUNC(Macro,macro)
ITEM_DEL_FUNC(Macro,macro)

inline const char *macro_text(Macro *mp)
{
	return mp->m_text;
	
}

inline void allow_recursion_for_macro(Macro *mp)
{
	mp->m_flags |= ALLOW_RECURSION;
}

inline const char *macro_filename(Macro *mp)
{
	return MACRO_FILENAME(mp);
}

inline int macro_lineno(Macro *mp)
{
	return MACRO_LINENO(mp);
}

void rls_macro(QSP_ARG_DECL  Macro *mp)
{
	Macro_Arg **ma_tbl;
	int i;

	// first release the resources
	rls_str( MACRO_FILENAME(mp) );

	ma_tbl = MACRO_ARG_TBL(mp);
	for(i=0;i<MACRO_N_ARGS(mp);i++){
		rls_macro_arg(ma_tbl[i]);
	}
	givbuf(ma_tbl);

	// free the stored text (body)
	rls_str(MACRO_TEXT(mp));

	rls_str(MACRO_NAME(mp));
	del_macro(QSP_ARG  mp);
}

void create_macro(QSP_ARG_DECL  const char *name, int n, Macro_Arg **ma_tbl, String_Buf *sbp, int lineno)
{
	Macro *mp;

	mp = new_macro(QSP_ARG  name);
	SET_MACRO_N_ARGS(mp,n);
	SET_MACRO_FLAGS(mp,0);
	SET_MACRO_ARG_TBL(mp,ma_tbl);
	SET_MACRO_TEXT(mp,savestr(sb_buffer(sbp)));
	// Can we access the filename here, or do we
	// need to do it earlier because of lookahead?
	// We have to save it, because qry_filename may
	// be released later...
	SET_MACRO_FILENAME( mp, savestr(current_filename(SINGLE_QSP_ARG)) );
	SET_MACRO_LINENO( mp, lineno );
}

inline int macro_is_invoked(Macro *mp)
{
	return IS_INVOKED(mp);
}
