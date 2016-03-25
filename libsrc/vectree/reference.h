#ifndef _REFERENCE_H_
#define _REFERENCE_H_

#include "vec_expr_node.h"
#include "strbuf.h"

typedef enum {
	OBJ_REFERENCE,
	STR_REFERENCE
} Reference_Type;

#ifdef FOOBAR
typedef union {
	Data_Obj *		u_dp;
	String_Buf *		u_sbp;
} Ref_Data;
#endif /* FOOBAR */

struct identifier;

typedef struct reference {
	Vec_Expr_Node *		ref_decl_enp;
	struct identifier *	ref_idp;
	Reference_Type		ref_type;
	Data_Obj *		ref_dp;
	String_Buf *		ref_sbp;
} Reference;


#define NO_REFERENCE	((Reference *)NULL)

#define IS_OBJECT_REF(refp)			(REF_TYPE(refp) == OBJ_REFERENCE)
#define IS_STRING_REF(refp)			(REF_TYPE(refp) == STR_REFERENCE)

#endif /* ! _REFERENCE_H_ */

