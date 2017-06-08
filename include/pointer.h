#ifndef _POINTER_H_
#define _POINTER_H_

#include "vec_expr_node.h"
#include "reference.h"

typedef struct pointer {
	uint32_t	ptr_flags;
	Vec_Expr_Node *	ptr_decl_enp;
	Reference *	ptr_refp;
} Pointer;

/* Pointer */
#define PTR_DECL_VN(ptrp)		(ptrp)->ptr_decl_enp
#define SET_PTR_DECL_VN(ptrp,enp)	(ptrp)->ptr_decl_enp = enp
#define PTR_REF(ptrp)			(ptrp)->ptr_refp
#define SET_PTR_REF(ptrp,refp)		(ptrp)->ptr_refp = refp
#define PTR_FLAGS(ptrp)			(ptrp)->ptr_flags
#define SET_PTR_FLAGS(ptrp,f)		(ptrp)->ptr_flags = f
#define SET_PTR_FLAG_BITS(ptrp,f)	(ptrp)->ptr_flags |= f
#define CLEAR_PTR_FLAG_BITS(ptrp,f)	(ptrp)->ptr_flags &= ~(f)

#define NEW_POINTER		((Pointer *)getbuf(sizeof(Pointer)))

/* pointer flags */
#define POINTER_SET	1


#endif /* ! _POINTER_H_ */

