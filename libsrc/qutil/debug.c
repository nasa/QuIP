#include "quip_config.h"

char VersionId_qutil_debug[] = QUIP_VERSION_STRING;

#include <stdio.h>	// NULL

#include "query.h"	// error1
#include "node.h"
#include "getbuf.h"

debug_flag_t debug=0;
int sup_verbose=0;

/*
 * Because the debug modules use the item package, subsystems
 * which support this (getbuf, hash) need to have their bit masks
 * pre-defined...
 */


static unsigned int  n_debug_modules=0;

#ifdef DEBUG_SYSTEM
static Debug_Module auto_dbm_tbl[N_AUTO_DEBUG_MODULES];
#endif /* DEBUG_SYSTEM */

static List *db_list = NO_LIST;

/* local prototypes */
static void	init_dbm(void);
static void	add_auto_module(const char *);

#ifdef DEBUG_SYSTEM
static void add_auto_module(const char *name)
{
	Node *np;

	/* user of savestr() eliminates a warning, but wastes some memory... */
	//auto_dbm_tbl[n_debug_modules].db_name = (char *) name;
	auto_dbm_tbl[n_debug_modules].db_name = savestr( name );
	auto_dbm_tbl[n_debug_modules].db_mask = 1<<n_debug_modules;
	np = mk_node(&auto_dbm_tbl[n_debug_modules]);
	addHead(db_list,np);
	n_debug_modules++;
}
#endif /* DEBUG_SYSTEM */

static void init_dbm()
{
	db_list=new_list();


#ifdef DEBUG_SYSTEM
	/* These have to be called in the correct order...
	 * To make sure everything is consistent, we define
	 * these in debug.h
	 */

	/* We enclosed this in ifdef USE_GETBUG, but that messed up the CAUTIOUS check below,
	 * so we just leave it here even when we're using malloc.
	 */
	add_auto_module(AUTO_MODULE_GETBUF);

	add_auto_module(AUTO_MODULE_FREEL);
	add_auto_module(AUTO_MODULE_HASH);
	add_auto_module(AUTO_MODULE_NAMESPACE);
	add_auto_module(AUTO_MODULE_NODES);
	add_auto_module(AUTO_MODULE_ITEMS);
	add_auto_module(AUTO_MODULE_CONTEXTS);

#ifdef CAUTIOUS
	if( n_debug_modules != N_AUTO_DEBUG_MODULES )
		NERROR1("CAUTIOUS:  bad number of automatic debug modules");
#endif /* CAUTIOUS */
#endif /* DEBUG_SYSTEM */

}

void set_debug(QSP_ARG_DECL  Debug_Module *dbmp)
{
	sprintf(ERROR_STRING,"enabling debugging messages for %s module",
		dbmp->db_name);
	ADVISE(ERROR_STRING);
//sprintf(ERROR_STRING,"mask = 0x%x",dbmp->db_mask);
//ADVISE(ERROR_STRING);
	debug |= dbmp->db_mask;
//sprintf(ERROR_STRING,"debug = 0x%x",debug);
//ADVISE(ERROR_STRING);
}

void clr_debug(QSP_ARG_DECL  Debug_Module *dbmp)
{
	sprintf(ERROR_STRING,"suppressing debugging messages for %s module",
		dbmp->db_name);
	ADVISE(ERROR_STRING);
	debug &= ~ dbmp->db_mask;
}

debug_flag_t add_debug_module(QSP_ARG_DECL  const char *name)
{
	Debug_Module *dbmp;
	Node *np;
	List *lp;

	if( n_debug_modules == 0 ) init_dbm();

	if( n_debug_modules >= MAX_DEBUG_MODULES ){
		sprintf(ERROR_STRING,"Can't add debug module %s",name);
		WARN(ERROR_STRING);
		sprintf(ERROR_STRING,"n is %d",n_debug_modules);
		ADVISE(ERROR_STRING);
		sprintf(ERROR_STRING,"Max is %ld",(long)MAX_DEBUG_MODULES);
		ADVISE(ERROR_STRING);
		ADVISE("Modules:");
		lp=dbm_list();
		np=lp->l_head;
		while(np!=NO_NODE){
			dbmp= (Debug_Module *) np->n_data;
			ADVISE(dbmp->db_name);
			np=np->n_next;
		}
		return(0);
	}

	/* do this now to insure n_debug_modules is initialized */
	lp = dbm_list();

	dbmp = (Debug_Module *) getbuf(sizeof(*dbmp));
	if( dbmp==NULL ) mem_err("add_debug_module");
	dbmp->db_name = name;
	dbmp->db_mask = 1 << n_debug_modules++;

	/* add to list */
	np=mk_node(dbmp);
	addHead(lp,np);

	return(dbmp->db_mask);
}

List *dbm_list()
{
	if( n_debug_modules == 0 ) init_dbm();
	return(db_list);
}

void clr_verbose()
{
	if( debug )  NADVISE("suppressing verbose messages");
	sup_verbose=0;
}

void set_verbose()
{
	if( debug ) NADVISE("printing verbose messages");
	sup_verbose=1;
}

void verbtog()
{
	if( sup_verbose ){
		clr_verbose();
	} else {
		set_verbose();
	}
}

