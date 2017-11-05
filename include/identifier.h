#ifndef _IDENTIFIER_H_
#define _IDENTIFIER_H_

#include "item_type.h"	// Item_Context
#include "shape_info.h"
#include "pointer.h"

/* identifier flags */
typedef enum {
	ID_POINTER,	// 0
	ID_OBJ_REF,	// 1
	ID_SUBRT,	// 2
	ID_STRING,	// 3
	ID_FUNCPTR,	// 4
	ID_LABEL,	// 5
	ID_SCALAR,	// 6
	ID_INVALID
} Id_Type;

typedef struct id_data {
	Pointer *	id_ptrp;
	Reference *	id_refp;
	Scalar_Value *	id_svp;
} Id_Data;

struct identifier {
	Item		id_item;
	int		id_type;
	void *		id_data;
	Shape_Info *	id_shpp;
	Item_Context *	id_icp;
} ;

/* Identifier */
#define ID_NAME(idp)			(idp)->id_item.item_name

#define ID_TYPE(idp)			(idp)->id_type
#define ID_REF(idp)			((Reference *)(idp)->id_data)
#define SET_ID_REF(idp,refp)		(idp)->id_data = refp
#define ID_FUNC(idp)			((Function_Ptr *)(idp)->id_data)
#define SET_ID_FUNC(idp,funcp)		(idp)->id_data = funcp
#define ID_DOBJ_CTX(idp)		(idp)->id_icp
#define ID_PTR(idp)			((Pointer *)(idp)->id_data)
#define SET_ID_PTR(idp,p)		(idp)->id_data = p
#define ID_SVAL_PTR(idp)		((Scalar_Value *)((idp)->id_data))
#define SET_ID_SVAL_PTR(idp,v)		(idp)->id_data = v

#define ID_SHAPE(idp)			(idp)->id_shpp
#define PUSH_ID_CONTEXT(icp)		push_item_context(id_itp,icp)
#define POP_ID_CONTEXT			pop_item_context(id_itp)

#define SET_ID_TYPE(idp,t)		(idp)->id_type = t
#define SET_ID_SHAPE(idp,shpp)		(idp)->id_shpp = shpp
#define SET_ID_DATA(idp,v)		(idp)->id_data = v
#define SET_ID_DOBJ_CTX(idp,icp)	(idp)->id_icp = icp

#define ID_PREC_PTR(idp)		SHP_PREC_PTR(ID_SHAPE(idp))
#define ID_PREC_CODE(idp)		PREC_CODE(ID_PREC_PTR(idp))

#define IS_STRING_ID(idp)	(ID_TYPE(idp) == ID_STRING)
#define IS_POINTER(idp)		(ID_TYPE(idp) == ID_POINTER)
#define IS_OBJ_REF(idp)		(ID_TYPE(idp) == ID_OBJ_REF)
#define IS_SUBRT(idp)		(ID_TYPE(idp) == ID_SUBRT)
/* #define IS_OBJECT(idp)	(ID_TYPE(idp) == ID_OBJECT) */
#define IS_FUNCPTR(idp)		(ID_TYPE(idp) == ID_FUNCPTR)
#define IS_LABEL(idp)		(ID_TYPE(idp) == ID_LABEL)
#define IS_SCALAR_ID(idp)	(ID_TYPE(idp) == ID_SCALAR)

#define STRING_IS_SET(idp)	(sb_buffer(REF_SBUF(ID_REF(idp))) != NULL)
#define POINTER_IS_SET(idp)	(PTR_FLAGS(ID_PTR(idp)) & POINTER_SET)


ITEM_NEW_PROT(Identifier,id)
ITEM_CHECK_PROT(Identifier,id)
ITEM_GET_PROT(Identifier,id)
ITEM_INIT_PROT(Identifier,id)
extern void _del_id(QSP_ARG_DECL  Identifier *idp);

#define new_id(name)	_new_id(QSP_ARG  name)
#define id_of(name)	_id_of(QSP_ARG  name)
#define get_id(name)	_get_id(QSP_ARG  name)
#define init_ids()	_init_ids(SINGLE_QSP_ARG)
#define del_id(idp)	_del_id(QSP_ARG  idp)

extern Identifier *_new_identifier(QSP_ARG_DECL  const char *s);
#define new_identifier(s) _new_identifier(QSP_ARG  s)

extern Item_Context *_create_id_context(QSP_ARG_DECL  const char *);
extern void _restrict_id_context(QSP_ARG_DECL  int flag);

#define create_id_context(s)	_create_id_context(QSP_ARG  s)
#define restrict_id_context(f)	_restrict_id_context(QSP_ARG  f)

extern void set_id_shape(Identifier *idp, Shape_Info *shpp);

#endif /* ! _IDENTIFIER_H_ */

