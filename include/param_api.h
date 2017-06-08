
#ifndef NO_PARAM

#include <stdio.h>

typedef struct param {
	const char *	p_name; /* usually = variable name */
	const char *	p_comment; /* brief description, units */
	int		p_type;
/*
 *
 *	the union is used when compiling param.c, but not for declaring
 *	param structs in application programs
 *
 */

#ifdef UPARAM
	union {
		int *ip;
		short *sp;
		float *fp;
		char *strp;
	} u;
#else
	void *	p_data;
#endif
} Param;

#define NULL_PARAM	(const char *)NULL,			\
			(const char *)NULL, 0L

#define NULL_UPARAM	(const char *)NULL,			\
			(const char *)NULL,0L,(void *)NULL

#define NO_PTYPE	NULL

/* codes for the type */
/*
 * use a long for the type
 *
 * forget the sign bit, with 5 flag bits, call it 24 bits for array len
 */

#define ARRAY_LEN_BITS	24
/* THIS HAS TO BE CHANGED IF ARRAY_LEN_BITS IS CHANGED!!! */
#define NELTMASK	((long)0x00ffffff)


#define INTP	(1L<<ARRAY_LEN_BITS)
#define FLOATP	(2L<<ARRAY_LEN_BITS)
#define STRINGP	(4L<<ARRAY_LEN_BITS)
#define SHORTP	(8L<<ARRAY_LEN_BITS)

#define ARRAY		(16L<<ARRAY_LEN_BITS)
#define TYPEMASK	(ARRAY|INTP|FLOATP|STRINGP|SHORTP)

#define IARRP	(ARRAY|INTP)
#define FARRP	(ARRAY|FLOATP)
#define SARRP	(ARRAY|SHORTP)
#define PARAM_TYPE( f )		( f & (~NELTMASK) )

#define NOTARR	(-1)

#define IS_INT_PARAM(p)			((p)->p_type & INTP)
#define IS_FLOAT_PARAM(p)		((p)->p_type & FLOATP)
#define IS_SHORT_PARAM(p)		((p)->p_type & SHORTP)
#define IS_STRING_PARAM(p)		((p)->p_type & STRINGP)
#define IS_ARRAY_PARAM(p)		((p)->p_type & ARRAY)
#define IS_INT_ARRAY_PARAM(p)		(((p)->p_type & TYPEMASK)==IARRP)
#define IS_FLOAT_ARRAY_PARAM(p)		(((p)->p_type & TYPEMASK)==FARRP)
#define IS_SHORT_ARRAY_PARAM(p)		(((p)->p_type & TYPEMASK)==SARRP)

/* this define was added for making string lists for which_one */
#define MAXPARAMS	64

#define INT_PARAM( ptr )	INTP,(char *)(ptr)
#define FLT_PARAM( ptr )	FLOATP,(char *)(ptr)
#define NULL_P_TYPE		0L

extern int chngp_ing;

extern void	chngp(QSP_ARG_DECL  Param *p);
extern COMMAND_FUNC( prm_menu );

#endif /* NO_PARAM */

