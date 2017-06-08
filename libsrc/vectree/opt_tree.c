#include "quip_config.h"

#include <stdio.h>
#include <ctype.h>
#include <math.h>

#include "quip_prot.h"
#include "data_obj.h"
#include "vectree.h"
#include "subrt.h"
#include "debug.h"	// AERROR

/* for definition of function codes */
#include "veclib_api.h"

/* local prototypes - needed because of recursive calls */
static void optimize_tree(QSP_ARG_DECL Vec_Expr_Node *enp);

#define COLLAPSE_LITERAL(enp)		collapse_literal(QSP_ARG  enp)
#define OPTIMIZE_TREE(enp)		optimize_tree(QSP_ARG  enp)
#define TRY_COLLAPSE(enp)		try_collapse(QSP_ARG  enp)


/* collapse an arithmetic expression involving literals */

static void collapse_literal(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	/* scan_tree should set the precision... */

	if( VN_PREC(enp) == PREC_SP ){
		double dval;

		dval = EVAL_FLT_EXP(enp);

		SET_VN_DBLVAL(enp, dval);
		SET_VN_CODE(enp, T_LIT_DBL);

		/* BUG need to free the children here */
	} else if( VN_PREC(enp) == PREC_DI ){
		long lval;

		lval = EVAL_INT_EXP(enp);

		SET_VN_INTVAL(enp, lval);
		SET_VN_CODE(enp, T_LIT_INT);

		/* BUG need to free the children here */
	} else {
		sprintf(ERROR_STRING,
			"collapse_literal:  unexpected node precision %s",
			NAME_FOR_PREC_CODE(VN_PREC(enp)));
		WARN(ERROR_STRING);
		return;
	}
	/* BUG this is where we should free these nodes;
	 * after this point, we have no way to find them!?
	 */
	SET_VN_CHILD(enp,0,NULL );
	SET_VN_CHILD(enp,1,NULL );
}

/* This function handles collapses of vsmul and vsadd:
 * (x+2)+3 = x+(2+3)
 * (x*3)*5 = x*(3*5)
 * 
 * We should be able to handle mixed ops, like:
 *
 * 2-(x+3) = (2-3)-x
 * 2/(x*3) = (2/3)/x
 *
 * but, for now, we don't !?
 *
 * We call it when we see a node with code T_VS_FUNC, whose child
 * has the same code, AND the ops match...
 */

static void try_collapse(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	/* (x+2)+3 = x+5;
	 * 2-(3-x) = x+(2-3);
	 * (x*2)*3 = x*(2*3);
	 * 2/(3/x) = x*(2/3);
	 */

	Vec_Expr_Node *tmp_enp;
	Tree_Code scalar_code;
	Vec_Func_Code vs_code;

	tmp_enp = VN_CHILD(enp,0);
	SET_VN_CHILD(enp,0, VN_CHILD(tmp_enp,0) ); /* x */

	/* the old object child node is now dangling,
	 * we change it to a scalar op node and make it
	 * the new scalar operand.
	 */
	switch(VN_VFUNC_CODE(tmp_enp)){
		case FVSADD:  scalar_code = T_PLUS;  vs_code = FVSADD; break;
		case FVSSUB:  scalar_code = T_MINUS; vs_code = FVSADD; break;
		case FVSMUL:  scalar_code = T_TIMES; vs_code = FVSMUL; break;
		case FVSDIV:  scalar_code = T_DIVIDE;vs_code = FVSMUL; break;
		default:
			assert( AERROR("try_collapse:  unhandled function code") );
			break;
	}

	SET_VN_CHILD(tmp_enp,0, VN_CHILD(enp,1) );
	SET_VN_CODE(tmp_enp, scalar_code);

	SET_VN_VFUNC_CODE(enp, vs_code);
	SET_VN_CHILD(enp,1, tmp_enp );

	/* now call optimize_tree() to collapse literals */
	OPTIMIZE_TREE(VN_CHILD(enp,1));
}

/*
 *
 * Do some optimization, to minimize the number of flops:
 *
 * Normally addition is left associative:
 * a + b + c == (a + b) + c
 *
 * But, if a is an image and b and c are scalars,
 * we want to evaluate it
 * a + (b + c)
 * AND if b is an image, and a and c are scalars, we should rearrange:
 * b + (a + c)
 *
 * This could conceivably be quite complicated; if i1 is an image
 * and s1,s2,s3 are scalars, then
 * s1*(s2+i1)+s3
 * would be most efficiently evaluated:
 * s1*i1+(s1*s2+s3)
 * The top node is still addition of an image and a scalar,
 * but the depth of the image tree has been reduced.
 * This should be the general goal, to rearrange the tree to
 * reduce the depth of branches w/ image dimensionality...
 *
 * Here we list the optimizations that are performed:
 *
 * 	1.  expressions involving only literals are evaluated,
 *	    and replaced with a new literal value.
 */

static void optimize_tree(QSP_ARG_DECL Vec_Expr_Node *enp)
{
	int i;

	if( enp == NULL ) return;

	/* BUG need to set & destroy the context! */

	for(i=0;i<3;i++){
		if( VN_CHILD(enp,i) != NULL )
			OPTIMIZE_TREE(VN_CHILD(enp,i));
	}

	/* now all the child nodes have been scanned, process this one */

	switch(VN_CODE(enp)){
		/* These are a bunch of codes we don't need to scan,
		 * they should probably be a default case once we
		 * know all the cases we have to process!
		 */
		case T_STAT_LIST:
		case T_ADVISE:
		case T_WARN:
		case T_DECL_ITEM_LIST: case T_DECL_STAT:
		case T_SCAL_DECL: case T_VEC_DECL: case T_IMG_DECL: case T_SEQ_DECL:
		case T_CSCAL_DECL: case T_CVEC_DECL: case T_CIMG_DECL: case T_CSEQ_DECL:

		case T_EXP_PRINT:
		case T_DYN_OBJ:
		case T_LIT_INT:
		case T_LIT_DBL:
		case T_RAMP:
		case T_MATH0_FN:
		case T_MATH0_VFN:
		case T_MATH1_FN:
		case T_MATH1_VFN:
		case T_MATH2_FN:
		case T_MATH2_VFN:
		case T_MATH2_VSFN:

			break;


		case T_PLUS:	case T_MINUS:	case T_TIMES:	case T_DIVIDE:

			/* Since we have already scanned,
		 	 * can be assumed to be scalar expressions.
			 *
			 * check and see it the operands are both
			 * literals, if so, evaluate and replace.
			 */
			assert( VN_CHILD(enp,0) != NULL && VN_CHILD(enp,1) != NULL );

			 if( IS_LITERAL(VN_CHILD(enp,0)) &&
			 	IS_LITERAL(VN_CHILD(enp,1)) )
				COLLAPSE_LITERAL(enp);

			/* at this point, we know that at least
			 * one of the branches contains an object;
			 * If that branch also has literals, we
			 * may be able to combine with literals in
			 * the other branch:
			 *
			 * (x+2)+3 = x+(2+3) = x+5
			 *
			 * Because the code immediately above performs
			 * the replacement of (2+3) with 5, any rearranging
			 * we do here will necessitate re-calling
			 * optimize_tree on the new branches...
			 *
			 * A more complicated case is:
			 * (x+2)+(y+3) = x+y+5
			 * Perhaps the way to deal with this is to have
			 * a node flag which indicates the branch contains
			 * literals.
			 */

			/* We might be able to fall-thru here to VSADD code,
			 * etc
			 */

			break;


		case T_VS_FUNC:
			assert( VN_CHILD(enp,0) != NULL );

			if( VN_CODE(VN_CHILD(enp,0)) == T_VS_FUNC &&
				VN_VFUNC_CODE(enp) == VN_VFUNC_CODE(VN_CHILD(enp,0)) )

				TRY_COLLAPSE(enp);

			break;

		/* If we have a repeated object, we can collapse:
		 * (x*y)+(x*z)=x*(y+z)
		 */
		case T_VV_FUNC:
		case T_ASSIGN:

			break;
			
		default:
			MISSING_CASE(enp,"optimize_tree");
			break;
	}
}
void optimize_subrt(QSP_ARG_DECL Subrt *srp)
{
	u_long nf_before, nf_after;

	cost_tree(QSP_ARG  SR_BODY(srp));
	nf_before = VN_FLOPS(SR_BODY(srp));

	OPTIMIZE_TREE(SR_BODY(srp));

	cost_tree(QSP_ARG  SR_BODY(srp));
	nf_after = VN_FLOPS(SR_BODY(srp));

	if( nf_before == nf_after ){
		sprintf(ERROR_STRING,
		"No optimization found for subroutine %s",SR_NAME(srp));
		advise(ERROR_STRING);
	} else {
		sprintf(ERROR_STRING,
	"Optimization of subroutine %s reduced flop count from %ld to %ld",
			SR_NAME(srp),nf_before,nf_after);
		advise(ERROR_STRING);
	}
}
