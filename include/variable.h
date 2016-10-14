#ifndef _VARIABLE_H_
#define _VARIABLE_H_

#include "item_type.h"

struct variable {
	Item			var_item;
	int			var_flags;
	union {
		const char *	u_value;
		const char *	(*u_func)(SINGLE_QSP_ARG_DECL);
	}			var_u;
} ;

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

/* Variable */
#define VAR_NAME(vp)		vp->var_item.item_name
#define VAR_VALUE(vp)		vp->var_u.u_value
#define VAR_TYPE(vp)		vp->var_type
#define VAR_FUNC(vp)		vp->var_u.u_func
#define SET_VAR_NAME(vp,s)	vp->var_item.item_name=s
/* this crashes if the value is NULL... */
/*#define SET_VAR_VALUE(vp,s)	{if (vp->var_u.u_value != NULL ) rls_str(vp->var_u.u_value); vp->var_u.u_value=savestr(s); } */
#define SET_VAR_VALUE(vp,s)	vp->var_u.u_value=s
#define SET_VAR_TYPE(vp,t)		vp->var_type=t
#define SET_VAR_FUNC(vp,f)		vp->var_u.u_func=f

#define NEW_VARIABLE(vp)	{vp=(Variable *)getbuf(sizeof(Variable));	\
				SET_VAR_NAME(vp,NULL); SET_VAR_VALUE(vp,NULL); }


extern void find_vars(QSP_ARG_DECL  const char *s);
extern void search_vars(QSP_ARG_DECL  const char *s);
extern const char *var_value(QSP_ARG_DECL  const char *vname);
extern const char *var_p_value(QSP_ARG_DECL  Variable *vp);

ITEM_INIT_PROT(Variable,var_)
ITEM_NEW_PROT(Variable,var_)
ITEM_CHECK_PROT(Variable,var_)
ITEM_PICK_PROT(Variable,var_)

#define VAR_OF(s)	var_of(QSP_ARG  s)
#define PICK_VAR(s)	pick_var_(QSP_ARG  s)

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

