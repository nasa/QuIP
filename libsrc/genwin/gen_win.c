#include "quip_config.h"

char VersionId_gen_win_gen_win[] = QUIP_VERSION_STRING;

#include "items.h"
#include "query.h"
#include "function.h"
#include "gen_win.h"
#include "version.h"

/* local prototypes */
static void init_genwin_class(SINGLE_QSP_ARG_DECL);
static Item_Class *gw_icp=NO_ITEM_CLASS;

#define INSURE_GENWIN					\
							\
	if( gw_icp == NO_ITEM_CLASS )			\
		init_genwin_class(SINGLE_QSP_ARG);


static void init_genwin_class(SINGLE_QSP_ARG_DECL)
{
#ifdef CAUTIOUS
	if( gw_icp != NO_ITEM_CLASS ){
		WARN("CAUTIOUS:  redundant call to init_genwin_class");
		return;
	}
#endif /* CAUTIOUS */
	gw_icp = new_item_class(QSP_ARG  "genwin");
}

void add_genwin(QSP_ARG_DECL  Item_Type *itp,Genwin_Functions *gwfp,Item *(*lookup)(QSP_ARG_DECL  const char *))
{
	INSURE_GENWIN

	add_items_to_class(gw_icp,itp,gwfp,lookup);
}

/* BUG - get_member() returns first occurence
 * of 'name' which restricts us to having
 * unique names for items of each item_type
 * of the class.
 */

Item *find_genwin(QSP_ARG_DECL  const char *name)
{
	INSURE_GENWIN
	return( get_member(QSP_ARG  gw_icp,name) );
}

#ifdef HAVE_X11

Viewer *genwin_viewer(QSP_ARG_DECL  Item *ip)
{
	Viewer *vp;
	Member_Info *mip;

	if( ip == NO_ITEM ) return(NO_VIEWER);

	INSURE_GENWIN

	mip = get_member_info(QSP_ARG  gw_icp,ip->item_name);
	if( mip->mi_lookup != NULL ){
		vp = (Viewer *)mip->mi_lookup(QSP_ARG  ip->item_name);
	} else {
		vp = (Viewer *)item_of(QSP_ARG  mip->mi_itp,ip->item_name);
	}
	return(vp);
}
#endif /* HAVE_X11 */


#ifdef HAVE_MOTIF

Panel_Obj *genwin_panel(QSP_ARG_DECL  Item *ip)
{
	Panel_Obj *po;
	Member_Info *mip;

	if( ip == NO_ITEM ) return(NO_PANEL_OBJ);

	INSURE_GENWIN

	mip = get_member_info(QSP_ARG  gw_icp,ip->item_name);
	if( mip->mi_lookup != NULL )
		po = (Panel_Obj *)mip->mi_lookup(QSP_ARG  ip->item_name);
	else
		po = (Panel_Obj *)item_of(QSP_ARG  mip->mi_itp,ip->item_name);
	return(po);
}

#endif /* HAVE_MOTIF */


/* position a window */
void posn_genwin(QSP_ARG_DECL  Item *ip,int x,int y)
{
	Genwin_Functions *gwfp;
	Member_Info *mip;

	if( ip == NO_ITEM ) return;

	INSURE_GENWIN

	mip = get_member_info(QSP_ARG  gw_icp,ip->item_name);
#ifdef CAUTIOUS
	if( mip == NO_MEMBER_INFO ){
		sprintf(ERROR_STRING,
			"CAUTIOUS:  position_genwin %s %d %d, missing member info #2",
			ip->item_name,x,y);
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	gwfp = (Genwin_Functions *) mip->mi_data;


	(*gwfp->posn_func)(QSP_ARG  ip->item_name,x,y) ;
}

/* show a window */
void show_genwin(QSP_ARG_DECL  Item *ip)
{
	Genwin_Functions *gwfp;
	Member_Info *mip;

	if( ip == NO_ITEM ) return;

	INSURE_GENWIN

	mip = get_member_info(QSP_ARG  gw_icp,ip->item_name);

#ifdef CAUTIOUS
	if( mip == NO_MEMBER_INFO ){
		sprintf(ERROR_STRING,
			"CAUTIOUS:  show_genwin %s, missing member info #2",
			ip->item_name);
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	gwfp = (Genwin_Functions *) mip->mi_data;

	(*gwfp->show_func)(QSP_ARG  ip->item_name);
}

/* unshow a window */
void unshow_genwin(QSP_ARG_DECL  Item *ip)
{
	Genwin_Functions *gwfp;
	Member_Info *mip;

	if( ip == NO_ITEM ) return;

	INSURE_GENWIN

	mip = get_member_info(QSP_ARG  gw_icp,ip->item_name);

#ifdef CAUTIOUS
	if( mip == NO_MEMBER_INFO ){
		sprintf(ERROR_STRING,
			"CAUTIOUS:  position_genwin %s, missing member info #2",
			ip->item_name);
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	gwfp = (Genwin_Functions *) mip->mi_data;

	(*gwfp->unshow_func)(QSP_ARG  ip->item_name);
}

/* delete a window */
void delete_genwin(QSP_ARG_DECL  Item *ip)
{
	Genwin_Functions *gwfp;
	Member_Info *mip;

	if( ip == NO_ITEM ) return;

	INSURE_GENWIN

	mip = get_member_info(QSP_ARG  gw_icp,ip->item_name);

#ifdef CAUTIOUS
	if( mip == NO_MEMBER_INFO ){
		sprintf(ERROR_STRING,
			"CAUTIOUS:  position_genwin %s, missing member info #2",
			ip->item_name);
		ERROR1(ERROR_STRING);
	}
#endif /* CAUTIOUS */

	gwfp = (Genwin_Functions *) mip->mi_data;

	(*gwfp->delete_func)(QSP_ARG  ip->item_name);
}

COMMAND_FUNC( do_posn_func )
{
	Item *ip;
	const char *s;
	int x, y;

	s = NAMEOF("genwin");
	x = HOW_MANY("x position");
	y = HOW_MANY("y position");

	ip = find_genwin(QSP_ARG  s);
	if( ip != NO_ITEM ) posn_genwin(QSP_ARG  ip, x, y);
}

COMMAND_FUNC( do_show_func )
{
	Item *ip;
	const char *s;

	s = NAMEOF("genwin");
	ip = find_genwin(QSP_ARG  s);
	if( ip != NO_ITEM ) show_genwin(QSP_ARG  ip);
}

COMMAND_FUNC( do_unshow_func )
{
	Item *ip;
	const char *s;

	s = NAMEOF("genwin");
	ip = find_genwin(QSP_ARG  s);
	if( ip != NO_ITEM ) unshow_genwin(QSP_ARG  ip);
}

COMMAND_FUNC( do_delete_func )
{
	Item *ip;
	const char *s;

	s = NAMEOF("genwin");
	ip = find_genwin(QSP_ARG  s);
	if( ip != NO_ITEM ) delete_genwin(QSP_ARG  ip);
}

static Command genwin_ctbl[]={
{ "position",	do_posn_func,		"position a window"	},
{ "show",	do_show_func,		"show a window"		},
{ "unshow",	do_unshow_func,		"unshow a window"	},
{ "delete",	do_delete_func,		"delete a window"	},
{ "quit",	popcmd,			"exit submenu"		},
{ NULL_COMMAND							}
};

COMMAND_FUNC( genwin_menu )
{
	static int inited=0;

	if( ! inited ){
		auto_version(QSP_ARG  "GENWIN","VersionId_gen_win");
		inited=1;
	}

	PUSHCMD(genwin_ctbl, "genwin");
}

