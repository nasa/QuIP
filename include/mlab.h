
#ifndef MLAB_DEFINED

#include "vectree.h"

typedef struct ml_func {
	char *	mlf_name;
	double	(*mlf_func)(double);
}ML_Func;

extern ML_Func math1_func_tbl[];
extern int show_tokens;

typedef struct complex {
	double real,imag;
} Complex;

#include "treecode.h"

typedef enum {
	MC_WHO,
	MC_WHOS,
	MC_HELP,
	MC_LOOKFOR,
	MC_SAVE,
	N_MATLAB_CMDS
} Matlab_Command;

#define N_TREE_CODES (((int)T_OBJECT)+1)

// this is defined in vectree.h...
//#define MAX_NODE_CHILDREN	3

/* globals */
extern char *code_name[];
//extern int vecexp_ing;
//extern int scanning_args;
//extern int dumpit;


#include "vectree.h"

/* prototypes */

/* mlabmisc.c */
extern void mc_who(void);
extern void mc_whos(void);
extern void mc_help(void);
extern void mc_lookfor(void);
extern void mc_save(void);

/* mlab.y */

extern void dump_ml_top(void);
extern void mlab_expr_file(void);
extern void mlab_print_tree(Vec_Expr_Node *);
extern void read_matlab_file(const char *);

/* ml_menu.c */

extern void do_mlab(void);


#define MLAB_DEFINED

#endif /* ! MLAB_DEFINED */

