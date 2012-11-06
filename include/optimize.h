
#ifndef NO_OPT_PARAM

#include "items.h"
#include "query.h"

/* Modular optimization package with semi-uniform API */

typedef struct opt_param {
	Item	op_item;
#define op_name	op_item.item_name	/* name of parameter */
	float	ans;		/* current, starting, final value */
	float	maxv;		/* maximum value */
	float	minv;		/* minimum value */
	float	delta;		/* initial delta */
	float	mindel;		/* minimum delta */
} Opt_Param;

#define NO_OPT_PARAM	((Opt_Param *)NULL)

typedef struct opt_pkg {
	Item	pkg_item;
#define pkg_name	pkg_item.item_name

	void (*pkg_scr_func)(SINGLE_QSP_ARG_DECL);
	void (*pkg_c_func)(float (*f)());
	void (*pkg_halt_func)(void);
} Opt_Pkg;

#define NO_OPT_PKG	((Opt_Pkg *)NULL)

extern Opt_Pkg *curr_opt_pkg;

#define FRPRMN_PKG_NAME		"frprmn"
#define AMOEBA_PKG_NAME		"amoeba"
#define DEFAULT_OPT_PKG		AMOEBA_PKG_NAME


/* #define MAX_OPT_PARAMS	27 */	/* historical, from stepit */
#define MAX_OPT_PARAMS	512

#ifdef HAVE_NUMREC
#include "numrec.h"
#endif /* HAVE_NUMREC */

/* global vars */

extern const char *opt_func_string;

/* Optimization modules need to provide the following functions:
 *
 * Called from the interpreter menu, run_xxx(void) calls the optimization routine,
 * which calls a script function to evaluate the error.
 *
 * Called from a C routine:
 *
 *
 * delete_opt_params();
 * add_opt_param(opp);
 * optimize(func);
 */


/* stepsupp.c */
ITEM_INTERFACE_PROTOTYPES(Opt_Param,opt_param)

extern void init_opt_params(void);
extern void delete_opt_params(SINGLE_QSP_ARG_DECL);
extern Opt_Param * add_opt_param(QSP_ARG_DECL  Opt_Param *);
extern void optimize( QSP_ARG_DECL  float (*func)(SINGLE_QSP_ARG_DECL) );
extern List *opt_param_list(SINGLE_QSP_ARG_DECL);
extern void opt_param_info(Opt_Param *);
extern float get_opt_param_value(QSP_ARG_DECL  const char *);


/* pkg.c */

extern void insure_opt_pkg(SINGLE_QSP_ARG_DECL);
ITEM_INTERFACE_PROTOTYPES(Opt_Pkg,opt_pkg)


/* am_supp.c */
extern void run_amoeba_scr(SINGLE_QSP_ARG_DECL);
extern void run_amoeba_c( float (*func)(void) );
extern void halt_amoeba(void);

/* pr_supp.c */
extern void run_frprmn_scr(SINGLE_QSP_ARG_DECL);
extern void run_frprmn_c( float (*func)(void) );
extern void halt_frprmn(void);

#ifdef STEPIT
/* st_supp.c */
extern void run_stepit_scr(void);
extern void run_stepit_c( float (*func)(void) );
#endif /* STEPIT */



#endif /* undef NO_OPT_PARAM */
