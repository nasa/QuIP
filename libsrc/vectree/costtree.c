/* costtree.c
 *
 * Figure out the cost of a subroutine, number of flops
 * (arithmetic ops or copies), and math lib calls.
 *
 */

#include "quip_config.h"

#include <stdio.h>
#include <ctype.h>
#include <math.h>

#include "quip_prot.h"
#include "data_obj.h"
//#include "fio_api.h"

#include "vectree.h"
#include "subrt.h"
#include "veclib_api.h"


#define max( n1 , n2 )		(n1>n2?n1:n2)

void _tell_cost(QSP_ARG_DECL  Subrt *srp)
{
	cost_tree(SR_BODY(srp) );
	warn("Sorry, cost reporting not implemented");
}

// count the number of flops and math library calls
// for this node, and all of its children

#define cost_node(enp) _cost_node(QSP_ARG  enp)

static void _cost_node(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	uint32_t nf_here,nf1,nf2;

	/* now all the child nodes have been scanned, process this one */

	switch(VN_CODE(enp)){
		case T_ENLARGE:
		case T_REDUCE:
			break;

		case T_LIST_OBJ:
			break;

		case T_TYPECAST:
			break;

		case T_SQUARE_SUBSCR:
			break;

		case T_REAL_PART:
		case T_IMAG_PART:
			break;
			
		case T_TRANSPOSE:
			break;

		case T_SUBVEC:
		case T_CSUBVEC:
			break;

		case T_RDFT:		/* scan_node */
			break;

		case T_RIDFT:		/* scan_node */
			break;

		case T_WRAP:
		case T_SCROLL:
			SET_VN_N_MATH( enp, VN_N_MATH(VN_CHILD(enp,0)) );
			SET_VN_FLOPS( enp, VN_FLOPS(VN_CHILD(enp,0))
				+ SHP_N_MACH_ELTS(VN_SHAPE(enp)) );

			break;

		case T_SIZE_FN:
			 break;

		case T_FILETYPE:	/* could check for valid type here... */
		case T_SAVE:
		case T_INFO:
			/* no-op */
			break;

		case T_LOAD:
			break;

		case T_CURLY_SUBSCR:
			break;


		case T_RETURN:
			break;

		case T_CALLFUNC:			/* scan_node() */
			break;

		case T_ARGLIST:
			break;

			
		case T_EXPR_LIST:
			SET_VN_FLOPS( enp, VN_FLOPS(VN_CHILD(enp,0))
				+ VN_FLOPS(VN_CHILD(enp,1)) );
			SET_VN_N_MATH( enp, VN_N_MATH(VN_CHILD(enp,0))
				+ VN_N_MATH(VN_CHILD(enp,1)) );

			break;

		case T_SCALMAX:
		case T_SCALMIN:
		/*
		case T_FVSMAX: case T_FVMAX:
		case T_FVSMIN: case T_FVMIN:
		*/
			break;

		case T_MAXVAL:
		case T_MINVAL:
			break;

		case T_RAMP:
			break;

		case T_UMINUS:

			SET_VN_N_MATH( enp, VN_N_MATH(VN_CHILD(enp,0)) );
			SET_VN_FLOPS( enp, VN_FLOPS(VN_CHILD(enp,0))
				+ SHP_N_MACH_ELTS(VN_SHAPE(enp)) );

			break;

		case T_RECIP:
			SET_VN_N_MATH( enp, VN_N_MATH(VN_CHILD(enp,0)) );
			SET_VN_FLOPS( enp, VN_FLOPS(VN_CHILD(enp,0))
				+ SHP_N_MACH_ELTS(VN_SHAPE(enp)) );

			break;

		case T_MATH1_VFN:
		case T_MATH1_FN:
			SET_VN_N_MATH( enp, VN_N_MATH(VN_CHILD(enp,0))
				+ SHP_N_MACH_ELTS( VN_SHAPE(enp) ) );
			SET_VN_FLOPS( enp, VN_FLOPS(VN_CHILD(enp,0)) );

			break;

		case T_MATH2_FN:
		case T_MATH2_VFN:
		case T_MATH2_VSFN:
			if( VN_SHAPE(enp) != NULL ){
				SET_VN_N_MATH( enp, VN_N_MATH(VN_CHILD(enp,0))
					+ VN_N_MATH(VN_CHILD(enp,1))
					+ SHP_N_MACH_ELTS( VN_SHAPE(enp) ) );
				SET_VN_FLOPS( enp, VN_FLOPS(VN_CHILD(enp,0))
					+ VN_FLOPS(VN_CHILD(enp,1))  );
			}

			break;

			

		/* These are a bunch of codes we don't need to scan,
		 * they should probably be a default case once we
		 * know all the cases we have to process!
		 */
		case T_STAT_LIST:
			/* a stat_list can have null children if
			 * the statements had parse errors
			 */
			if( VN_CHILD(enp,0) == NULL ){
				if( VN_CHILD(enp,1) == NULL ){
					SET_VN_FLOPS( enp, 0 );
					SET_VN_N_MATH( enp, 0 );
				} else {
					SET_VN_FLOPS( enp,
						VN_FLOPS(VN_CHILD(enp,1)) );
					SET_VN_N_MATH( enp,
						VN_N_MATH(VN_CHILD(enp,1)) );
				}
			} else if( VN_CHILD(enp,1) == NULL ){
					SET_VN_FLOPS( enp,
						VN_FLOPS(VN_CHILD(enp,0)) );
					SET_VN_N_MATH( enp,
						VN_N_MATH(VN_CHILD(enp,0)) );
			} else {
				SET_VN_FLOPS( enp, VN_FLOPS(VN_CHILD(enp,0))
					+ VN_FLOPS(VN_CHILD(enp,1)) );

				SET_VN_N_MATH( enp, VN_N_MATH(VN_CHILD(enp,0))
					+ VN_N_MATH(VN_CHILD(enp,1)) );

			}

			break;

		case T_ASSIGN:
			/*
			 * For T_ASSIGN nodes whose RHS is
			 * an expression and the LHS is an object,
			 * the moves are done as part of the operation,
			 * so they don't need to be counted at the assign node.
			 * 
			 * To properly count how many operations will be
			 * required, we should be determining ahead of time
			 * (e.g. NOW) whether we will need temp objects
			 * for intermediate storage...
			 * Now this is done in eval_obj_assignment...
			 */

			if( VN_CODE( VN_CHILD(enp,1) ) == T_DYN_OBJ )
				nf_here = SHP_N_MACH_ELTS(VN_SHAPE(enp));
			else {
				if( SCALAR_SHAPE(VN_SHAPE(VN_CHILD(enp,1))) )
					nf_here=SHP_N_MACH_ELTS(VN_SHAPE(enp));
				else
					nf_here = 0;
			}

			SET_VN_FLOPS( enp, nf_here
				+ VN_FLOPS(VN_CHILD(enp,1)) );
			SET_VN_N_MATH( enp, VN_N_MATH(VN_CHILD(enp,1)) );

			break;



		case T_VV_FUNC:
		case T_VS_FUNC:
			break;

		case T_PLUS:	case T_MINUS:	case T_TIMES:	case T_DIVIDE:


			nf1=VN_FLOPS(VN_CHILD(enp,0));
			nf2=VN_FLOPS(VN_CHILD(enp,1));

			/* The number of flops for this node and its
			 * children is the sum of the children's
			 * flops, plus the number of operations
			 * at this node.  A move counts as a flop...
			 *
			 * We used to take the max of the child element
			 * counts, but that is wrong if it is an
			 * outer product or something like that...
			 */

			//nf_here = max(ne1,ne2);
			nf_here = SHP_N_MACH_ELTS( VN_SHAPE(enp) );

			SET_VN_FLOPS( enp, nf_here + nf1 + nf2 );

			break;
			
		case T_DYN_OBJ:
			break;

		case T_LIT_INT:
		case T_LIT_DBL:
			break;

		/* these are do-nothing cases, but we put them here
		 * for completeness...
		 */
		case T_DECL_STAT_LIST:
		case T_STRING_LIST:
		case T_STRING:
		case T_ADVISE:
		case T_WARN:
		case T_DECL_ITEM_LIST:
		case T_EXP_PRINT:
		case T_SCRIPT:
		ALL_DECL_CASES
			break;


		default:
			missing_case(enp,"cost_node");
			break;
	}
}

void _cost_tree(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int i;

	if( enp == NULL ) return;

	for(i=0;i<MAX_CHILDREN(enp);i++){
		if( VN_CHILD(enp,i) != NULL )
			cost_tree(VN_CHILD(enp,i));
	}

	/* now all the child nodes have been scanned, process this one */

	cost_node(enp);	/* code shared w/ rescan_tree() */
}

