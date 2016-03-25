#ifndef _VARIABLE_H_
#define _VARIABLE_H_

#include "item_type.h"

typedef struct variable {
	Item			var_item;
	int			var_flags;
	union {
		const char *	u_value;
		const char *	(*u_func)(SINGLE_QSP_ARG_DECL);
	}			var_u;
} Variable;

// flag bits
#define	VAR_SIMPLE	1
#define VAR_DYNAMIC	2
#define VAR_RESERVED	4

#define VAR_FLAGS(vp)			(vp)->var_flags
#define SET_VAR_FLAGS(vp,f)		(vp)->var_flags = f
#define SET_VAR_FLAG_BITS(vp,f)		(vp)->var_flags |= f

#define IS_DYNAMIC_VAR(vp)	((vp)->var_flags & VAR_DYNAMIC)
#define IS_SIMPLE_VAR(vp)	((vp)->var_flags & VAR_SIMPLE)
#define IS_RESERVED_VAR(vp)	((vp)->var_flags & VAR_RESERVED)

#define NO_VARIABLE 	((Variable *)NULL)

#define ASSIGN_VAR( s1 , s2 )	assign_var(QSP_ARG  s1 , s2 )
#define ASSIGN_RESERVED_VAR( s1 , s2 )	assign_reserved_var(QSP_ARG  s1 , s2 )

extern void find_vars(QSP_ARG_DECL  const char *s);
extern void search_vars(QSP_ARG_DECL  const char *s);
extern const char *var_value(QSP_ARG_DECL  const char *vname);
extern const char *var_p_value(QSP_ARG_DECL  Variable *vp);

ITEM_INIT_PROT(Variable,var_)
ITEM_NEW_PROT(Variable,var_)
ITEM_CHECK_PROT(Variable,var_)
ITEM_PICK_PROT(Variable,var_)

extern Variable *assign_var(QSP_ARG_DECL  const char *name, const char *value);
extern Variable *create_reserved_var(QSP_ARG_DECL  const char *name, const char *value);
extern Variable *force_reserved_var(QSP_ARG_DECL  const char *name, const char *value);
extern Variable *assign_reserved_var(QSP_ARG_DECL  const char *name, const char *value);
extern Variable *var_of(QSP_ARG_DECL const char *name);
extern void init_dynamic_var(QSP_ARG_DECL  const char *name,
			const char *(*func)(SINGLE_QSP_ARG_DECL) );
extern void init_variables(SINGLE_QSP_ARG_DECL);
extern void reserve_variable(QSP_ARG_DECL  const char *name);
extern void replace_var_string(QSP_ARG_DECL  Variable *vp, const char *find,
							const char *replace);
extern void show_var(QSP_ARG_DECL  Variable *vp );
extern void set_script_var_from_int(QSP_ARG_DECL  const char *varname, long val );



#endif /* ! _VARIABLE_H_ */

