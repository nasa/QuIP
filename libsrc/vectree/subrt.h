
#include "item_type.h"
#include "vec_expr_node.h"

typedef struct subrt {
	Item		sr_item;
	Vec_Expr_Node *	sr_arg_decls;
	Vec_Expr_Node *	sr_arg_vals;
	Vec_Expr_Node *	sr_body;
	int		sr_n_args;
	Shape_Info *	sr_shpp;
	Shape_Info *	sr_dest_shpp;
	List *		sr_ret_lp;
	List *		sr_call_lp;
	Precision *	sr_prec_p;
	int		sr_flags;
	Vec_Expr_Node *	sr_call_enp;
} Subrt;

	

/* flag bits */
#define SR_SCANNING	1
#define SR_SCRIPT	2
#define SR_PROTOTYPE	4
#define SR_REFFUNC	8
#define SR_COMPILED	16

#define IS_SCANNING(srp)	( SR_FLAGS(srp) & SR_SCANNING )
#define IS_SCRIPT(srp)		( SR_FLAGS(srp) & SR_SCRIPT )
#define IS_REFFUNC(srp)		( SR_FLAGS(srp) & SR_REFFUNC )
#define IS_COMPILED(srp)	( SR_FLAGS(srp) & SR_COMPILED )

#define NO_SUBRT	((Subrt *)NULL)

ITEM_INIT_PROT(Subrt,subrt)
ITEM_CHECK_PROT(Subrt,subrt)
ITEM_NEW_PROT(Subrt,subrt)
ITEM_PICK_PROT(Subrt,subrt)
ITEM_ENUM_PROT(Subrt,subrt)
ITEM_LIST_PROT(Subrt,subrt)

//extern Subrt *subrt_of(QSP_ARG_DECL  const char *name);
//extern Subrt *new_subrt(QSP_ARG_DECL  const char *name);
//extern Subrt *pick_subrt(QSP_ARG_DECL const char *prompt);

#define PICK_SUBRT(p)	pick_subrt(QSP_ARG  p)

extern Item_Type *subrt_itp;

/*
extern void list_subrts(SINGLE_QSP_ARG_DECL);
extern List * list_of_subrts(SINGLE_QSP_ARG_DECL);
*/

