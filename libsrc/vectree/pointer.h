#ifndef _POINTER_H_
#define _POINTER_H_

#include "vec_expr_node.h"
#include "reference.h"

typedef struct pointer {
	uint32_t	ptr_flags;
	Vec_Expr_Node *	ptr_decl_enp;
	Reference *	ptr_refp;
} Pointer;


#define NO_POINTER	((Pointer *)NULL)

/* pointer flags */
#define POINTER_SET	1


#endif /* ! _POINTER_H_ */

