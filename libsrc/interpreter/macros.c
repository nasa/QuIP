#include "quip_config.h"

char VersionId_interpreter_macros[] = QUIP_VERSION_STRING;

#include <stdio.h>

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "query.h"
#include "savestr.h"
#include "macros.h"

/* local prototypes */

ITEM_INTERFACE_DECLARATIONS(Macro,macro)

static void show_margs(Macro *mp);
static void show_mflags(Macro *mp);


Macro *_pick_macro(QSP_ARG_DECL  const char *pmpt)
{
	Macro *mp;

	QUERY_FLAGS &= ~QS_EXPAND_MACS;
	mp = pick_macro(QSP_ARG pmpt);
	QUERY_FLAGS |= QS_EXPAND_MACS;
	return(mp);
}

void mac_stats(SINGLE_QSP_ARG_DECL)
{
	item_stats(QSP_ARG  macro_itp);
}

static void show_margs(Macro *mp)
{
	int i;

	for(i=0;i<mp->m_nargs;i++) {
		sprintf(msg_str,"\t$%d:   %s",i+1,mp->m_prompt[i]);
		prt_msg(msg_str);
	}
}

static void show_mflags(Macro *mp)
{
	if( RECURSION_FORBIDDEN(mp) )
		sprintf(msg_str,"\trecursion forbidden");
	else
		sprintf(msg_str,"\trecursion allowed");
	prt_msg(msg_str);
}

void macro_info(Macro *mp)
{
	sprintf(msg_str,"Macro \"%s\" (file \"%s\", line %d)",
		mp->m_name,mp->m_filename,mp->m_lineno);
	prt_msg(msg_str);

	show_margs(mp);
	show_mflags(mp);
	prt_msg("\n");
}

void showmac(Macro *mp)		/** show macro text */
{
	macro_info(mp);
	prt_msg(mp->m_text);
}

void _del_macro(QSP_ARG_DECL  Macro *mp)
{
	int i;

	del_macro(QSP_ARG  mp->m_name);		/* remove from item table */

	/* free prompts */
	for(i=0;i<mp->m_nargs;i++)
		rls_str((char *)mp->m_prompt[i]);	/* this is allocated in mac_cmds.c... */
	givbuf(mp->m_prompt);			/* this is allocated in mac_cmds.c... */
	givbuf((void *)mp->m_text);
	rls_str((char *)mp->m_name);
	rls_str((char *)mp->m_filename);
}

Macro *_def_macro( QSP_ARG_DECL  const char *name, int nargs, const char **pmptlist,
	Item_Type **itps, const char *text )
{
	Macro *mp;

	mp = new_macro(QSP_ARG  name);
	if( mp == NO_MACRO ) return(mp);

	mp->m_nargs=nargs;
	mp->m_prompt = pmptlist;
	mp->m_itps = itps;
	mp->m_text=text;
	mp->m_flags = 0;	/* disallow recursion as default */

	return(mp);
}

/* search macros for a name containing the fragment */

void find_macros(QSP_ARG_DECL  const char *s)
{
	List *lp;

	lp=find_items(QSP_ARG  macro_itp,s);
	if( lp==NO_LIST ) return;
	print_list_of_items(QSP_ARG  lp);
}

/* search macros for those whose text contains the fragment */

List *search_macros(QSP_ARG_DECL  const char *frag)
{
	List *lp, *newlp=NO_LIST;
	Node *np, *newnp;
	Macro *mp;
	const char *mbuf;
	char lc_frag[LLEN];

	if( macro_itp == NO_ITEM_TYPE ) return(NO_LIST);
	lp=item_list(QSP_ARG  macro_itp);
	if( lp == NO_LIST ) return(lp);

	np=lp->l_head;
	decap(lc_frag,frag);
	while(np!=NO_NODE){
		mp = (Macro*) np->n_data;
		if( mp->m_text != NULL ){	/* NULL if no macro text... */
			mbuf = savestr( mp->m_text );
			/* make the match case insensitive */
			decap((char *)mbuf,mbuf);
			if( strstr(mbuf,lc_frag) != NULL ){
				if( newlp == NO_LIST )
					newlp=new_list();
				newnp=mk_node(mp);
				addTail(newlp,newnp);
			}
			rls_str(mbuf);
		}
		np=np->n_next;
	}
	return(newlp);
}

double dmacroexists(QSP_ARG_DECL  const char *name)
{
	Macro *mp;

	mp = macro_of(QSP_ARG  name);
	if( mp == NO_MACRO ) return(0.0);
	else return(1.0);
}


