
#include "quip_config.h"
//#include "expr_node.h"
#include "nexpr.h"
#include "getbuf.h"

Scalar_Expr_Node *alloc_expr_node(void)
{
	return getbuf(sizeof(Scalar_Expr_Node));
}



