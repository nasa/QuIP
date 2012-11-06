#include "quip_config.h"

char VersionId_vectree_vt_menu[] = QUIP_VERSION_STRING;

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include <stdio.h>

#include "debug.h"
#include "menuname.h"
#include "version.h"
#include "nvf_api.h"
#include "vectree.h"
#include "query.h"		/* redir() */

/* local prototypes */

int dumpit=0;

static COMMAND_FUNC( do_dumpit )
{
	dumpit=ASKIF("dump trees before immediate execution");
}

static COMMAND_FUNC( do_fileparse )
{
	FILE *fp;
	const char *s;

	/* disable_lookahead(); */
	s=NAMEOF("expression file");
	if( strcmp(s,"-") ){
		fp=TRY_OPEN( s, "r" );
		if( !fp ) return;
		push_input_file(QSP_ARG  s);
		redir(QSP_ARG  fp);
	}
	expr_file(SINGLE_QSP_ARG);
	/* enable_lookahead(); */

	enable_stripping_quotes(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_parse )
{
	expr_file(SINGLE_QSP_ARG);
}

static COMMAND_FUNC( do_show_shp )
{
	int flg;

	flg=ASKIF("Show node shapes");
	set_show_shape(flg);
}

static COMMAND_FUNC( do_show_res )
{
	if( ASKIF("show resolvers") )
		dump_flags |= SHOW_RES;
	else	dump_flags &= ~SHOW_RES;
}

static COMMAND_FUNC( do_show_key )
{
	if( ASKIF("show shape key") )
		dump_flags |= SHOW_KEY;
	else	dump_flags &= ~SHOW_KEY;
}

static COMMAND_FUNC( do_show_lhs_refs )
{
	int flg;

	flg=ASKIF("Show lhs ref counts");
	set_show_lhs_refs(flg);
}

static COMMAND_FUNC( do_mk_scr )
{
	Subrt *srp;
	const char *txt;
	const char *name;
	int n;

	name = NAMEOF("subroutine name");
	n = HOW_MANY("number of arguments");
	txt = NAMEOF("subroutine text");

	srp = create_script_subrt(QSP_ARG  name,n,txt);
}

static COMMAND_FUNC( do_dump_tree )
{
	int n;
	Vec_Expr_Node *enp;

	n=HOW_MANY("node serial number");
	enp = find_node_by_number(QSP_ARG  n);
	if( enp == NO_VEXPR_NODE ) return;

	DUMP_TREE(enp);
}

static COMMAND_FUNC( do_unexport )
{
	Data_Obj *dp;
	Identifier *idp;

	dp = PICK_OBJ("");
	if( dp == NO_OBJ ) return;

	idp = ID_OF(dp->dt_name);
	if( idp == NO_IDENTIFIER ){
		sprintf(ERROR_STRING,"do_unexport:  object %s has not been exported",dp->dt_name);
		WARN(ERROR_STRING);
		return;
	}
	/* now remove it */
	delete_id(QSP_ARG  (Item *)idp);
	dp->dt_flags &= ~DT_EXPORTED;
}

static COMMAND_FUNC( do_export )
{
	Data_Obj *dp;
	Identifier *idp;

	dp = PICK_OBJ("");
	if( dp == NO_OBJ ) return;

	idp = ID_OF(dp->dt_name);
	if( idp != NO_IDENTIFIER ){
		sprintf(ERROR_STRING,"do_export:  identifier %s already exists!?",idp->id_name);
		WARN(ERROR_STRING);
		return;
	}
	idp = make_named_reference(QSP_ARG  dp->dt_name);
#ifdef CAUTIOUS
	if( idp == NO_IDENTIFIER ) ERROR1("CAUTIOUS:  do_export:  unable to create named reference");
#endif
	idp->id_refp->ref_dp = dp;
	dp->dt_flags |= DT_EXPORTED;
}

static void node_info(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int save_flags;

	save_flags = dump_flags;
	dump_flags = SHOW_ALL;
	DUMP_NODE(enp);
	dump_flags = save_flags;
}

static COMMAND_FUNC( do_node_info )
{
	Vec_Expr_Node *enp;
	int n;

	n=HOW_MANY("node serial number");
	enp = find_node_by_number(QSP_ARG  n);
	if( enp == NO_VEXPR_NODE ) return;

	node_info(QSP_ARG  enp);
}

static COMMAND_FUNC( do_list_subrts )
{ list_subrts(SINGLE_QSP_ARG); }

Command expr_ctbl[]={
{ "export",	do_export,	"export data object"			},
{ "unexport",	do_unexport,	"un-export data object"			},
{ "parse",	do_parse,	"parse single expression"		},
{ "read",	do_fileparse,	"parse contents of a file"		},
{ "list",	do_list_subrts,	"list defined subroutines"		},
{ "run",	do_run_subrt,	"run a subroutine"			},
{ "dumpflag",	do_dumpit,	"set flag to dump before immediate execution" },
{ "show_lhs",	do_show_lhs_refs,"show node lhs ref counts"		},
{ "optimize",	do_opt_subrt,	"optimize a subroutine tree"		},
{ "info",	do_subrt_info,	"print subroutine info"			},
{ "node",	do_node_info,	"print node info"			},
{ "dump",	do_dump_subrt,	"dump a subroutine tree"		},
{ "tree",	do_dump_tree,	"dump a tree from a given node"		},
{ "show_shp",	do_show_shp,	"show node shapes during dump"		},
{ "show_res",	do_show_res,	"show unknown node resolvers"		},
{ "show_key",	do_show_key,	"show shape key"			},
{ "script",	do_mk_scr,	"create a script subroutine"		},
{ "cost",	do_tell_cost,	"report subroutine cost"		},
#ifndef MAC
{ "quit",	popcmd,		"exit submenu"				},
#endif
{ NULL_COMMAND								}
};

void vt_init(SINGLE_QSP_ARG_DECL)
{
	sort_tree_tbl();
	init_fixed_nodes(SINGLE_QSP_ARG);

	/* we put this here for the heck of it */
	if( id_itp == NO_ITEM_TYPE ) id_init(SINGLE_QSP_ARG);
	set_del_method(QSP_ARG  id_itp,delete_id);

#ifdef DEBUG
	resolve_debug=add_debug_module(QSP_ARG  "resolver");
	eval_debug=add_debug_module(QSP_ARG  "evaluator");
	scope_debug=add_debug_module(QSP_ARG  "scope");
	cast_debug=add_debug_module(QSP_ARG  "typecast");
	parser_debug=add_debug_module(QSP_ARG  "quip_parser");
#endif /* DEBUG */

	auto_version(QSP_ARG  "VECTREE","VersionId_vectree");
}

COMMAND_FUNC( do_exprs )
{
	static int inited=0;

	if( ! inited ){
		vl_init(SINGLE_QSP_ARG);
		vt_init(SINGLE_QSP_ARG);
		inited=1;
	}
	PUSHCMD(expr_ctbl,EXPR_MENU_NAME);
}

