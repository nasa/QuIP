#include "quip_config.h"

char VersionId_interpreter_mac_cmds[] = QUIP_VERSION_STRING;

/** BUG?  comments are not discarded from macro text when defined */

#include <stdio.h>
#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "debug.h"
#include "macros.h"
#include "submenus.h"
#include "query.h"
#include "menuname.h"
#include "items.h"

#define MAXLINES	128

static COMMAND_FUNC( show_mac );
static COMMAND_FUNC( do_del_mac );
static COMMAND_FUNC( def_macro );
static COMMAND_FUNC( do_find_mac );
static COMMAND_FUNC( do_search_macs );
static COMMAND_FUNC( let_nest );

static COMMAND_FUNC( show_mac )
{
	Macro *mp;

	mp=PICK_MACRO("");

	if( mp != NO_MACRO ) showmac(mp);
}

static COMMAND_FUNC( do_del_mac )
{
	Macro *mp;

	mp=PICK_MACRO("");

	if( mp==NO_MACRO ) return;

	_del_macro(QSP_ARG  mp);
}

static COMMAND_FUNC( def_macro )		/** prompt for macro name and def. */
{
	int i;
	char mname[LLEN];
	int nargs;
	const char **mpmpts=NULL;
	Item_Type **itps=NULL;
	const char *mtext;
	const char *fn;
	Macro *mp;
	int lineno;

	static const char *nsuff[]={"st","nd","rd","th"};

	THIS_QSP->qs_flags &= ~QS_EXPAND_MACS;

	strcpy(mname,NAMEOF(MACRONAME_PROMPT));

	THIS_QSP->qs_flags |= QS_EXPAND_MACS;

	fn = savestr( current_input_file(SINGLE_QSP_ARG) );

	nargs=(int)HOW_MANY("number of arguments");

	if( nargs > 0 ){
		/* this won't be freed until the macro is released... */
		mpmpts = (const char **)getbuf(nargs*sizeof(char *));
#ifdef CAUTIOUS
		if( mpmpts == NULL ){
			sprintf(ERROR_STRING,"nargs = %d",nargs);
			advise(ERROR_STRING);
			ERROR1("CAUTIOUS:  error allocating space for macro args");
		}
#endif /* CAUTIOUS */

		for(i=0;i<nargs;i++){
			char pstr[LLEN];
			char pstr2[LLEN];
			const char *s;

			if( i<3 )
				sprintf(pstr,"prompt for %d%s argument",
					i+1,nsuff[i]);
			else
				sprintf(pstr,"prompt for %d%s argument",
					i+1,nsuff[3]);
			sprintf(pstr2,"%s (or optional item type spec)",pstr);
			/* this won't be freed until the macro is released... */
			s = NAMEOF(pstr2);
			/* We can specify the item type of the prompted-for object
			 * by preceding the prompt with an item type name in brackets,
			 * e.g. <Data_Obj>
			 */
			if( *s == '<' ){
				Item_Type *itp;
				int n;
				n=strlen(s);
				if( s[n-1] == '>' ){
					strcpy(pstr2,s+1);
					pstr2[n-2]=0;	/* kill closing bracket */
					itp = get_item_type(QSP_ARG  pstr2);
					if( itp == NO_ITEM_TYPE ){
		WARN("Unable to process macro argument item type specification.");
					} else {
						if( itps == NULL ){
							int j;
							itps=getbuf(nargs*sizeof(Item_Type *));
							for(j=0;j<nargs;j++)
								itps[i]=NO_ITEM_TYPE;
						}
						itps[i]=itp;
					}
				} else {
					WARN("Unterminated macro argument item type specification.");
				}
				s=NAMEOF(pstr);
			}
			mpmpts[i] = savestr(s);
		}
	}

	if( verbose ){
		sprintf(ERROR_STRING,"reading text for macro %s",mname);
		advise(ERROR_STRING);
	}

	// We want to store the line number of the file where the macro
	// is declared...  We can read it now from the query stream...
	lineno = (&THIS_QSP->qs_query[QLEVEL])->q_rdlineno;

	mtext=rdmtext(SINGLE_QSP_ARG);		/* read the macro text */

	if( (mp=_def_macro( QSP_ARG   (const char *)mname, nargs, mpmpts, itps, mtext )) == NO_MACRO )
		goto failure;

	mp->m_filename = savestr(fn);
	mp->m_lineno = lineno;

#ifndef MAC
	popcmd(SINGLE_QSP_ARG);	/* for backwards compatibility with old scripts */
#endif /* ! MAC */
	return;

failure:
	/* this usually occurs when the macro has already been defined */
	mp = macro_of(QSP_ARG  mname);
	if( mp != NO_MACRO ){
		sprintf(ERROR_STRING,"Macro \"%s\" defined in file %s",
			mname,mp->m_filename);
		advise(ERROR_STRING);
		sprintf(ERROR_STRING,"Second definition from file %s ignored",
			fn);
		advise(ERROR_STRING);
	} else {
		sprintf(ERROR_STRING,"Unknown problem creating macro \"%s\"",
			mname);
		WARN(ERROR_STRING);
	}
	/* We used to release the prompts here, but now they aren't re-saved in _def_macro() */

#ifndef MAC
	popcmd(SINGLE_QSP_ARG);	/* for backwards compatibility with old scripts */
#endif /* ! MAC */
}

static COMMAND_FUNC( let_nest )
{
	Macro *mp;
	
	mp=PICK_MACRO("");
	if( mp == NO_MACRO ) return;
	mp->m_flags |= ALLOW_RECURS;
}

static COMMAND_FUNC( do_find_mac )
{
	const char *s;

	s=NAMEOF("name fragment");

	find_macros(QSP_ARG  s);
}

static COMMAND_FUNC( do_search_macs )
{
	const char *s;
	List *lp;

	s=NAMEOF("macro fragment");
	lp=search_macros(QSP_ARG  s);
	if( lp == NO_LIST ) return;

	sprintf(msg_str,"Fragment \"%s\" occurs in the following macros:",s);
	prt_msg(msg_str);

	print_list_of_items(QSP_ARG  lp);
}

static COMMAND_FUNC( do_mac_info )
{
	Macro *mp;

	mp=PICK_MACRO("");
	if( mp == NO_MACRO ) return;
	macro_info(mp);
}

static COMMAND_FUNC( do_exp_mac )
{
	Macro *mp;
	int i,level;
	char macro_call[LLEN];

	mp=PICK_MACRO("");
	if( mp == NO_MACRO ) return;

	sprintf(macro_call,"%s",mp->m_name);
	for(i=0;i<mp->m_nargs;i++){
		const char *arg;
		arg=NAMEOF(mp->m_prompt[i]);
		strcat(macro_call," ");
		strcat(macro_call,arg);
	}
sprintf(ERROR_STRING,"macro_call =\"%s\"",macro_call);
advise(ERROR_STRING);
	
	level = tell_qlevel(SINGLE_QSP_ARG);
	sprintf(ERROR_STRING,"Macro %s",mp->m_name);
	push_input_file(QSP_ARG  ERROR_STRING);
	PUSHTEXT(macro_call);
	while( level < tell_qlevel(SINGLE_QSP_ARG) ){
		const char *s;
		s=qword(QSP_ARG  "");
		prt_msg(s);
	}
}

static COMMAND_FUNC(do_list_macs)
{ list_macros(SINGLE_QSP_ARG); }

static COMMAND_FUNC(do_mac_stats){mac_stats(SINGLE_QSP_ARG);}

Command macctbl[]={
{ "define",	def_macro,	"define macro"				},
{ "info",	do_mac_info,	"display information about a macro"	},
{ "show",	show_mac,	"display macro text"			},
{ "list",	do_list_macs,	"list all macros"			},
{ "delete",	do_del_mac,	"delete macro"				},
{ "stats",	do_mac_stats,	"show macro hash table statistics"	},
{ "find",	do_find_mac,	"find macros"				},
{ "search",	do_search_macs,	"search macro contents"			},
{ "nest",	let_nest,	"allow nested macro calls"		},
{ "expand",	do_exp_mac,	"expand macro without executing"	},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif /* !MAC */
{ NULL_COMMAND								}
};

COMMAND_FUNC( macmenu )
{
	PUSHCMD(macctbl,MACRO_MENU_NAME);
}

/* perhaps this stuff should be in its own file ? */

static Item_Type *curr_itp=NO_ITEM_TYPE;

static COMMAND_FUNC( sel_ittyp )
{
	Item_Type *itp;

	itp = pick_ittyp(QSP_ARG  "");
	if( itp == NO_ITEM_TYPE ) return;
	curr_itp=itp;
}

static COMMAND_FUNC( list_them )
{
	if( curr_itp==NO_ITEM_TYPE ){
		WARN("no item type selected");
		return;
	}
	list_items(QSP_ARG  curr_itp);
}

static COMMAND_FUNC( it_stats )
{
	if( curr_itp==NO_ITEM_TYPE ){
		WARN("no item type selected");
		return;
	}
	item_stats(QSP_ARG  curr_itp);
}

static COMMAND_FUNC( do_find )
{
	List *lp;
	const char *s;

	s=NAMEOF("name fragment");
	/* find_all_items(s); */

	if( curr_itp==NO_ITEM_TYPE ){
		WARN("no item type selected");
		return;
	}
	lp=find_items(QSP_ARG  curr_itp,s);
	if( lp==NO_LIST ) return;
	print_list_of_items(QSP_ARG  lp);
}

static COMMAND_FUNC( do_item_info )
{
	if( curr_itp==NO_ITEM_TYPE ){
		WARN("no item type selected");
		return;
	}
	dump_item_type(QSP_ARG  curr_itp);
}

static COMMAND_FUNC(do_list_ittyps)
{ list_ittyps(SINGLE_QSP_ARG); }

Command ittyp_ctbl[]={
{ "list",	do_list_ittyps,	"list all item types"			},
{ "select",	sel_ittyp,	"select item type"			},
{ "members",	list_them,	"list members of selected item type"	},
{ "info",	do_item_info,	"print info about selected item type"	},
{ "stats",	it_stats,	"show hash table statistics"		},
{ "find",	do_find,	"find items"				},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif /* !MAC */
{ NULL_COMMAND								}
};

COMMAND_FUNC( ittyp_menu )
{
	PUSHCMD(ittyp_ctbl,ITEM_MENU_NAME);
}

