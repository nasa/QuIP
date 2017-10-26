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
ITEM_DEL_PROT(Variable,var_)

#define init_var_s()	_init_var_s(SINGLE_QSP_ARG)
#define new_var_(name)	_new_var_(QSP_ARG  name)
#define var__of(name)	_var__of(QSP_ARG  name)
#define del_var_(name)	_del_var_(QSP_ARG  name)


extern Variable *create_reserved_var(QSP_ARG_DECL  const char *name, const char *value);
extern Variable *force_reserved_var(QSP_ARG_DECL  const char *name, const char *value);
extern void init_dynamic_var(QSP_ARG_DECL  const char *name,
			const char *(*func)(SINGLE_QSP_ARG_DECL) );
extern void _reserve_variable(QSP_ARG_DECL  const char *name);
extern void _replace_var_string(QSP_ARG_DECL  Variable *vp, const char *find,
							const char *replace);
extern void _show_var(QSP_ARG_DECL  Variable *vp );

#define replace_var_string(vp,find,replace)	_replace_var_string(QSP_ARG  vp,find,replace)
#define reserve_variable(name)			_reserve_variable(QSP_ARG  name)
#define show_var(vp)				_show_var(QSP_ARG  vp)


#endif /* ! _VARIABLE_H_ */

