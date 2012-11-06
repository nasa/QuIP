/* costtree.c
 *
 * Figure out the cost of a subroutine, number of flops
 * (arithmetic ops or copies), and math lib calls.
 *
 */

#include "quip_config.h"

char VersionId_vectree_costtree[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <ctype.h>
#include <math.h>

#include "data_obj.h"
#include "debug.h"
#include "getbuf.h"
#include "function.h"
#include "fio_api.h"

#include "vectree.h"
#include "nvf_api.h"

/* local prototypes */

static void cost_node(QSP_ARG_DECL  Vec_Expr_Node *);

#define max( n1 , n2 )		(n1>n2?n1:n2)

void tell_cost(QSP_ARG_DECL  Subrt *srp)
{
	cost_tree(QSP_ARG  srp->sr_body);
	WARN("Sorry, cost reporting not implemented");
}

void cost_tree(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int i;

	if( enp == NO_VEXPR_NODE ) return;

	for(i=0;i<MAX_CHILDREN(enp);i++){
		if( enp->en_child[i] != NO_VEXPR_NODE )
			cost_tree(QSP_ARG  enp->en_child[i]);
	}

	/* now all the child nodes have been scanned, process this one */

	cost_node(QSP_ARG  enp);	/* code shared w/ rescan_tree() */
}

static void cost_node(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Shape_Info *shpp1=NULL,*shpp2=NULL;	/* auto-init just to silence compiler warnings? */
	u_long nf_here,nf1,nf2;
	dimension_t ne1,ne2;

	/* now all the child nodes have been scanned, process this one */

	switch(enp->en_code){
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
			enp->en_nmath = enp->en_child[0]->en_nmath;
			enp->en_flops = enp->en_child[0]->en_flops
				+ enp->en_shpp->si_n_mach_elts;

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
			enp->en_flops = enp->en_child[0]->en_flops
				+ enp->en_child[1]->en_flops;
			enp->en_nmath = enp->en_child[0]->en_nmath
				+ enp->en_child[1]->en_nmath;

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

			enp->en_nmath = enp->en_child[0]->en_nmath;
			enp->en_flops = enp->en_child[0]->en_flops
				+ enp->en_shpp->si_n_mach_elts;

			break;

		case T_RECIP:
			enp->en_nmath = enp->en_child[0]->en_nmath;
			enp->en_flops = enp->en_child[0]->en_flops
				+ enp->en_shpp->si_n_mach_elts;

			break;

		case T_MATH1_VFN:
		case T_MATH1_FN:
			enp->en_nmath = enp->en_child[0]->en_nmath
				+ shpp1->si_n_mach_elts;
			enp->en_flops = enp->en_child[0]->en_flops;

			break;

		case T_MATH2_FN:
		case T_MATH2_VFN:
		case T_MATH2_VSFN:
			if( enp->en_shpp != NO_SHAPE ){
				enp->en_nmath = enp->en_child[0]->en_nmath
					+ enp->en_child[1]->en_nmath
					+ shpp1->si_n_mach_elts;
				enp->en_flops = enp->en_child[0]->en_flops
					+ enp->en_child[1]->en_flops ;
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
			if( enp->en_child[0] == NO_VEXPR_NODE ){
				if( enp->en_child[1] == NO_VEXPR_NODE ){
					enp->en_flops = 0;
					enp->en_nmath = 0;
				} else {
					enp->en_flops =
						enp->en_child[1]->en_flops;
					enp->en_nmath =
						enp->en_child[1]->en_nmath;
				}
			} else if( enp->en_child[1] == NO_VEXPR_NODE ){
					enp->en_flops =
						enp->en_child[0]->en_flops;
					enp->en_nmath =
						enp->en_child[0]->en_nmath;
			} else {
				enp->en_flops = enp->en_child[0]->en_flops
					+ enp->en_child[1]->en_flops;

				enp->en_nmath = enp->en_child[0]->en_nmath
					+ enp->en_child[1]->en_nmath;

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

			if( enp->en_child[1]->en_code == T_DYN_OBJ )
				nf_here = enp->en_shpp->si_n_mach_elts;
			else {
				if( SCALAR_SHAPE(shpp2) )
					nf_here=enp->en_shpp->si_n_mach_elts;
				else
					nf_here = 0;
			}

			enp->en_flops = nf_here
				+ enp->en_child[1]->en_flops;
			enp->en_nmath = enp->en_child[1]->en_nmath;

			break;



		case T_VV_FUNC:
		case T_VS_FUNC:
			break;

		case T_PLUS:	case T_MINUS:	case T_TIMES:	case T_DIVIDE:


			nf1=enp->en_child[0]->en_flops;
			nf2=enp->en_child[1]->en_flops;
			ne1=shpp1->si_n_mach_elts;
			ne2=shpp2->si_n_mach_elts;

			/* The number of flops for this node and its
			 * children is the sum of the children's
			 * flops, plus the number of operations
			 * at this node.  A move counts as a flop...
			 */

			nf_here = max(ne1,ne2);

			enp->en_flops = nf_here + nf1 + nf2;

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
			MISSING_CASE(enp,"cost_node");
			break;
	}
}

