#ifndef _REFERENCE_H_
#define _REFERENCE_H_

#include "vec_expr_node.h"

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

/* Reference */
#define REF_OBJ(refp)		(refp)->ref_dp
#define SET_REF_OBJ(refp,dp)	(refp)->ref_dp = dp
#define REF_ID(refp)		(refp)->ref_idp
#define SET_REF_ID(refp,idp)	(refp)->ref_idp = idp
#define REF_TYPE(refp)		(refp)->ref_type
#define SET_REF_TYPE(refp,t)	(refp)->ref_type = t
#define REF_SBUF(refp)		(refp)->ref_sbp
#define SET_REF_SBUF(refp,sbp)	(refp)->ref_sbp = sbp
#define REF_DECL_VN(refp)	(refp)->ref_decl_enp
#define SET_REF_DECL_VN(refp,enp)	(refp)->ref_decl_enp = enp

#define NEW_REFERENCE		((Reference *)getbuf(sizeof(Reference)))

#define IS_OBJECT_REF(refp)			(REF_TYPE(refp) == OBJ_REFERENCE)
#define IS_STRING_REF(refp)			(REF_TYPE(refp) == STR_REFERENCE)

#endif /* ! _REFERENCE_H_ */

