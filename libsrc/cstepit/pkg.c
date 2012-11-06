#include "quip_config.h"

char VersionId_cstepit_pkg[] = QUIP_VERSION_STRING;

#include "items.h"
#include "optimize.h"
#include "cstepit.h"
//#include "new_cstepit.h"	/* just for testing! */
#include "debug.h"	/* verbose */
#include "query.h"

/* local prototypes */

static void init_opt_pkgs(SINGLE_QSP_ARG_DECL);

static void init_one_pkg( QSP_ARG_DECL
	const char *,
	void (*scr_func)(SINGLE_QSP_ARG_DECL),
	void (*c_func)(float (*f)(void)),
	void (*h_func)(void)
	);

ITEM_INTERFACE_DECLARATIONS(Opt_Pkg,opt_pkg)

void insure_opt_pkg(SINGLE_QSP_ARG_DECL)
{
	if( curr_opt_pkg == NO_OPT_PKG ){
		init_opt_pkgs(SINGLE_QSP_ARG);
		curr_opt_pkg=get_opt_pkg(QSP_ARG  /* DEFAULT_OPT_PKG */ "cstepit" );
#ifdef CAUTIOUS
		if( curr_opt_pkg==NO_OPT_PKG )
			ERROR1("CAUTIOUS:  no default optimization package");
#endif /* CAUTIOUS */
		if( verbose ){
			sprintf(error_string,
				"Using default optimization package \"%s\"",
				curr_opt_pkg->pkg_name);
			advise(error_string);
		}
	}
}

static void init_one_pkg( QSP_ARG_DECL
	const char *name,
	void (*scr_func)(SINGLE_QSP_ARG_DECL),
	void (*c_func)(float (*f)(void)),
	void (*h_func)(void) )
{
	Opt_Pkg *pkp;

	pkp = new_opt_pkg(QSP_ARG  name);

#ifdef CAUTIOUS
	if( pkp==NO_OPT_PKG ){
		sprintf(error_string,
	"CAUTIOUS:  error creating optimization package %s",name);
		NWARN(error_string);
	}
#endif /* CAUTIOUS */

	pkp->pkg_scr_func	= scr_func;
	pkp->pkg_c_func		= c_func;
	pkp->pkg_halt_func	= h_func;
}

static void init_opt_pkgs(SINGLE_QSP_ARG_DECL)
{
	init_one_pkg(QSP_ARG  "cstepit",	run_cstepit_scr,run_cstepit_c,halt_cstepit);
	/* init_one_pkg(QSP_ARG  "stepit",run_stepit_scr,run_stepit_c); */
#ifdef HAVE_NUMREC
	init_one_pkg(QSP_ARG  AMOEBA_PKG_NAME,run_amoeba_scr,run_amoeba_c,halt_amoeba);
	init_one_pkg(QSP_ARG  FRPRMN_PKG_NAME,run_frprmn_scr,run_frprmn_c,halt_frprmn);
#endif /* HAVE_NUMREC */
}

