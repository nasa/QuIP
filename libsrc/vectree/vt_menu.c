#include "quip_config.h"

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include <stdio.h>

#include "debug.h"
//#include "menuname.h"
//#include "version.h"
#include "veclib_api.h"
#include "vectree.h"
//#include "query.h"		/* redir() */
#include "vt_api.h"
#include "quip_prot.h"
#include "warn.h"
#include "getbuf.h"

// BUG global is not thread-safe
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
fprintf(stderr,"Parsing contents of file %s\n",s);
		//push_input_file(QSP_ARG  s);
		redir(QSP_ARG  fp, s );
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
	//Subrt *srp;
	const char *txt;
	const char *name;
	int n;

	name = NAMEOF("subroutine name");
	n = (int) HOW_MANY("number of arguments");
	txt = NAMEOF("subroutine text");

	/*srp =*/ create_script_subrt(QSP_ARG  name,n,txt);
}

static COMMAND_FUNC( do_dump_tree )
{
	int n;
	Vec_Expr_Node *enp;

	n= (int) HOW_MANY("node serial number");
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

	idp = ID_OF(OBJ_NAME(dp));
	if( idp == NO_IDENTIFIER ){
		sprintf(ERROR_STRING,"do_unexport:  object %s has not been exported",OBJ_NAME(dp));
		WARN(ERROR_STRING);
		return;
	}
	/* now remove it */
	delete_id(QSP_ARG  (Item *)idp);
	CLEAR_OBJ_FLAG_BITS(dp,DT_EXPORTED);
}

// Make this a function because it might be useful elsewhere...

static Data_Obj *base_object( Data_Obj *dp )
{
	while( OBJ_PARENT(dp) != NO_OBJ && !strncmp(OBJ_NAME(OBJ_PARENT(dp)),OBJ_NAME(dp),
							strlen(OBJ_NAME(OBJ_PARENT(dp))) ) )
		dp = OBJ_PARENT(dp);
	return dp;
}
			
static COMMAND_FUNC( do_export )
{
	Data_Obj *dp, *parent;
	Identifier *idp;

	dp = PICK_OBJ("");
	if( dp == NO_OBJ ) return;

	// If this is a subscripted object, export the parent
	//
	// If the object is subscripted from the parent,
	// then the parent's name will be a substring of the
	// object's name
	parent = base_object(dp);

	// We want to export the parent if this is a subscripted object, but not if it is a subimage.
	// We compare the names:  if the object name begins with the parent name, which is then
	// followed by an index delimiter, then we export the parent.
	if( !strncmp(OBJ_NAME(dp),OBJ_NAME(parent),strlen(OBJ_NAME(parent))) ){
		int c;
		const char *s=OBJ_NAME(dp);
		c=s[ strlen(OBJ_NAME(parent)) ];
		if( c == '{' || c == '[' )
			dp = parent;
	}

	// The OBJ_EXTRA field holds decl_enp for objects declared in the
	// expression language...  exported objects are not, so we have to
	// explicitly clear the field here.  WHAT ELSE MIGHT THIS FIELD BE
	// USED FOR???
	SET_OBJ_EXTRA(dp,NULL);

	idp = ID_OF(OBJ_NAME(dp));
	if( idp != NO_IDENTIFIER ){
		sprintf(ERROR_STRING,"do_export:  identifier %s already exists!?",ID_NAME(idp));
		WARN(ERROR_STRING);
		return;
	}
	idp = make_named_reference(QSP_ARG  OBJ_NAME(dp));
//#ifdef CAUTIOUS
//	if( idp == NO_IDENTIFIER ){
//		ERROR1("CAUTIOUS:  do_export:  unable to create named reference");
//		IOS_RETURN
//	}
//#endif
	assert( idp != NO_IDENTIFIER );

	SET_REF_OBJ(ID_REF(idp), dp);
	SET_OBJ_FLAG_BITS(dp, DT_EXPORTED);
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

	n= (int) HOW_MANY("node serial number");
	enp = find_node_by_number(QSP_ARG  n);
	if( enp == NO_VEXPR_NODE ) return;

	node_info(QSP_ARG  enp);
}

static COMMAND_FUNC( do_list_subrts )
{ list_subrts(SINGLE_QSP_ARG); }

#define ADD_CMD(s,f,h)	ADD_COMMAND(expressions_menu,s,f,h)

MENU_BEGIN(expressions)

ADD_CMD( export,	do_export,	export data object	)
ADD_CMD( unexport,	do_unexport,	un-export data object	)
ADD_CMD( parse,		do_parse,	parse single expression	)
ADD_CMD( read,		do_fileparse,	parse contents of a file	)
ADD_CMD( list,		do_list_subrts,	list defined subroutines	)
ADD_CMD( run,		do_run_subrt,	run a subroutine	)
ADD_CMD( dumpflag,	do_dumpit,	set flag to dump before immediate execution	)
ADD_CMD( show_lhs,	do_show_lhs_refs,show node lhs ref counts	)
ADD_CMD( optimize,	do_opt_subrt,	optimize a subroutine tree	)
ADD_CMD( info,		do_subrt_info,	print subroutine info	)
ADD_CMD( node,		do_node_info,	print node info	)
ADD_CMD( dump,		do_dump_subrt,	dump a subroutine tree	)
ADD_CMD( tree,		do_dump_tree,	dump a tree from a given node	)
ADD_CMD( show_shp,	do_show_shp,	show node shapes during dump	)
ADD_CMD( show_res,	do_show_res,	show unknown node resolvers	)
ADD_CMD( show_key,	do_show_key,	show shape key	)
ADD_CMD( script,	do_mk_scr,	create a script subroutine	)
ADD_CMD( cost,		do_tell_cost,	report subroutine cost	)

MENU_END(expressions)

static double id_exists(QSP_ARG_DECL  const char *name)
{
	Identifier *ip;

	ip=ID_OF(name);
	if( ip==NO_IDENTIFIER ){
		// We didn't find it, but it might be a subscripted object
		Data_Obj *dp;

		dp=hunt_obj( QSP_ARG  name);
		if( dp != NO_OBJ ){
			dp = base_object(dp);
			ip=ID_OF(OBJ_NAME(dp));
			if( ip==NO_IDENTIFIER )
				return 0.0;
			else return 1.0;
		}
		return 0.0;
	}
	else return 1.0;
}

void vt_init(SINGLE_QSP_ARG_DECL)
{
	sort_tree_tbl();
	init_fixed_nodes(SINGLE_QSP_ARG);

	init_ids(SINGLE_QSP_ARG);		// init other item types too???
	// BUG need to figure out how to do this without referencing itp
	// why?
	set_del_method(QSP_ARG  id_itp,delete_id);
	DECLARE_STR1_FUNCTION(	id_exists,	id_exists )

#ifdef QUIP_DEBUG
	resolve_debug=add_debug_module(QSP_ARG  "resolver");
	eval_debug=add_debug_module(QSP_ARG  "evaluator");
	scope_debug=add_debug_module(QSP_ARG  "scope");
	cast_debug=add_debug_module(QSP_ARG  "typecast");
	parser_debug=add_debug_module(QSP_ARG  "quip_parser");
#endif /* QUIP_DEBUG */

}

COMMAND_FUNC( do_exprs )
{
	static int inited=0;

	if( ! inited ){
		vl_init(SINGLE_QSP_ARG);
		vt_init(SINGLE_QSP_ARG);
		inited=1;
	}
	PUSH_MENU(expressions);
}

