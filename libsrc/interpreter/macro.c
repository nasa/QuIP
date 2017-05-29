
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

Macro * create_macro(QSP_ARG_DECL  const char *name, int n, Macro_Arg **ma_tbl, String_Buf *sbp, int lineno)
{
	Macro *mp;

	mp = new_macro(QSP_ARG  name);
	SET_MACRO_N_ARGS(mp,n);
	SET_MACRO_FLAGS(mp,0);
	SET_MACRO_ARG_TBL(mp,ma_tbl);
	assert(sbp!=NULL);
	if( *sb_buffer(sbp) != 0 )
		SET_MACRO_TEXT(mp,savestr(sb_buffer(sbp)));
	else
		SET_MACRO_TEXT(mp,NULL);

	// Can we access the filename here, or do we
	// need to do it earlier because of lookahead?
	// We have to save it, because qry_filename may
	// be released later...
	SET_MACRO_FILENAME( mp, savestr(current_filename(SINGLE_QSP_ARG)) );
	SET_MACRO_LINENO( mp, lineno );
	return mp;
}

inline int macro_is_invoked(Macro *mp)
{
	return IS_INVOKED(mp);
}

static void setup_generic_macro_arg(Macro_Arg *map, int idx)
{
	char str[32];
	map->ma_itp=NULL;		// default
	sprintf(str,"argument %d",idx+1);
	map->ma_prompt = savestr(str);	// memory leak?  BUG?  where freed?
}

Macro_Arg **create_generic_macro_args(int n)
{
	Macro_Arg **ma_tbl;
	int i;

	assert(n>0&&n<32);	// 32 is somewhat arbitrary...
	ma_tbl = getbuf( n * sizeof(Macro_Arg *));
	for(i=0;i<n;i++){
		ma_tbl[i] = getbuf( sizeof(Macro_Arg) );
		setup_generic_macro_arg(&ma_tbl[i],i);
	}
	return ma_tbl;
}

