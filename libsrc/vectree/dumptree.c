#include "quip_config.h"

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <inttypes.h>	// PRId64 etc

#include "quip_prot.h"
//#include "savestr.h"
#include "data_obj.h"
#include "debug.h"
//#include "getbuf.h"
#include "node.h"
#include "function.h"
#include "nexpr.h"
#include "veclib_api.h"
#include "warn.h"

#include "vectree.h"
#include "subrt.h"

/* for definition of function codes */
/* #include "wartbl.h" */

#ifdef NOT_YET
static Keyword *curr_native_func_tbl=vt_native_func_tbl;
#endif /* NOT_YET */



/* Dump flags */

int dump_flags=0;

int dumping=0;

#ifdef NOT_YET
void set_native_func_tbl(Keyword *tbl)
{
	curr_native_func_tbl = tbl;
}
#endif /* NOT_YET */

void set_show_shape(int flg)
{
	if( flg )
		dump_flags |= SHOW_SHAPES;
	else
		dump_flags &= ~SHOW_SHAPES;
}

#ifdef NOT_USED

void set_show_key(int flg)
{
	if( flg )	dump_flags |= SHOW_KEY;
	else		dump_flags &= ~SHOW_KEY;
}

void set_show_cost(int flg)
{
	if( flg )
		dump_flags |= SHOW_COST;
	else
		dump_flags &= ~SHOW_COST;
}

#endif /* NOT_USED */

void set_show_lhs_refs(int flg)
{
	if( flg )
		dump_flags |= SHOW_LHS_REFS;
	else
		dump_flags &= ~SHOW_LHS_REFS;
}

void print_dump_legend(SINGLE_QSP_ARG_DECL)
{
	/* blank line */
	prt_msg("");

	/* first line */
	prt_msg_frag("node\t");

	if( SHOWING_LHS_REFS )
		prt_msg_frag("n_lhs\t");
	if( SHOWING_COST )
		prt_msg_frag("depth,\tflops\tnmath");

	prt_msg("\top\t\tchildren");

	/* second line */
	if( SHOWING_COST )
		prt_msg("\tnelts");

	/* blank line */
	prt_msg("");
}

void print_shape_key(SINGLE_QSP_ARG_DECL)
{
	prt_msg("_\tno shape\n?\tunknown shape\n*\towns shape\n@\tshape ref\n#\tunknown leaf");
}

static void prt_node(Vec_Expr_Node *enp,char *buf)
{
	int key;

	if( VN_SHAPE(enp) == NULL ) key='_';
	else if( UNKNOWN_SHAPE(VN_SHAPE(enp)) ) key='?';
	else if ( OWNS_SHAPE(enp) ) key='*';
	else key='@';

	sprintf(buf,"n%-4d %c%c%c",VN_SERIAL(enp), key,
		RESOLVED_AT_CALLTIME(enp) ? '!' : ' ',
		HAS_CONSTANT_VALUE(enp) ? 'C' : ' '
		
		);
}

#define dump_node_basic(enp) _dump_node_basic(QSP_ARG  enp)

static void _dump_node_basic(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Tree_Code code;
	int i;
	const char *s;

	if( enp == NULL ) return;

	/* print the node "name", and a code that tells about shape knowledge */

// Temporarily print to stderr instead of stdout for debugging...
	prt_node(enp,msg_str);
	prt_msg_frag(msg_str);

	if( SHOWING_LHS_REFS ){
		sprintf(msg_str,"\t%d",VN_LHS_REFS(enp));
		prt_msg_frag(msg_str);
	}

	if( SHOWING_COST ){
		if( VN_SHAPE(enp) != NULL ){
			sprintf(msg_str,"\t%d", SHP_N_MACH_ELTS(VN_SHAPE(enp)));
		}

		prt_msg_frag(msg_str);

		sprintf(msg_str,"\t%d\t%d", VN_FLOPS(enp),VN_N_MATH(enp));
		prt_msg_frag(msg_str);
	}

	if( IS_CURDLED(enp) ){
		sprintf(msg_str,"\t%s (curdled!?)", NNAME(enp));
		prt_msg(msg_str);
		return;
	}

	sprintf(msg_str,"\t%s", NNAME(enp));
	prt_msg_frag(msg_str);

	/* print the special op-dependent args in human-readable form */

	code = VN_CODE(enp);

	if( code==T_DYN_OBJ || code == T_UNDEF || code == T_PROTO || code==T_POINTER || code==T_FUNCPTR || code==T_STR_PTR ){
		sprintf(msg_str,"\t%s",VN_STRING(enp));
		prt_msg_frag(msg_str);
		if( code == T_POINTER ){
			Identifier *idp;
			/* We don't use get_set_ptr() here because we don't want an error msg... */
			idp = id_of(VN_STRING(enp));
			if( idp != NULL && IS_POINTER(idp) && POINTER_IS_SET(idp) ){
				if( PTR_REF(ID_PTR(idp)) == NULL ){
					/* how could this ever happen??? */
					prt_msg_frag("->???");
				} else {
					Data_Obj *dp;
					dp = REF_OBJ(PTR_REF(ID_PTR(idp)));
					sprintf(msg_str,"->%s",OBJ_NAME(dp));
					prt_msg_frag(msg_str);
				}
			}
		}
	} else if( code == T_STATIC_OBJ ){
		sprintf(msg_str,"\t%s",OBJ_NAME(VN_OBJ(enp)));
		prt_msg_frag(msg_str);
#ifdef SCALARS_NOT_OBJECTS
	} else if( code == T_SCALAR_VAR ){
		sprintf(msg_str,"\t%s",VN_STRING(enp));
		prt_msg_frag(msg_str);
#endif // SCALARS_NOT_OBJECTS
	} else if ( code == T_FUNCREF ){
		Subrt *srp;
		srp=VN_SUBRT(enp);
		sprintf(msg_str,"\t%s",SR_NAME(srp));
		prt_msg_frag(msg_str);
	} else if( code == T_SIZE_FN ){
		sprintf(msg_str,"\t%s",FUNC_NAME(VN_FUNC_PTR(enp)));
		prt_msg_frag(msg_str);
	}
#ifdef NOT_YET
	else if(code == T_CALL_NATIVE ){
		// was kw_token???
		// curr_native_func_tbl...
		sprintf(msg_str,"\t%s",FUNC_NAME(VN_FUNC_PTR(enp)));
		prt_msg_frag(msg_str);
	}
#endif /* NOT_YET */
	else if(code == T_TYPECAST ){
		// BUG not how we do precision any more!!!
		//sprintf(msg_str,"  %s",NAME_FOR_PREC_CODE(VN_INTVAL(enp)));
        if( VN_SHAPE(enp) == NULL ) error1("CAUTIOUS:  null node shape for typecast node!?");
        else {
            sprintf(msg_str,"  %s",PREC_NAME(VN_PREC_PTR(enp)));
            prt_msg_frag(msg_str);
        }
    } else if( code == T_SUBRT_DECL || code == T_SCRIPT ){
		Subrt *srp;
		srp=VN_SUBRT(enp);
		sprintf(msg_str,"\t%s",SR_NAME(srp));
		prt_msg_frag(msg_str);
	} else if( code==T_DECL_STAT ){
		//sprintf(msg_str," %s",NAME_FOR_PREC_CODE(VN_INTVAL(enp)));
		sprintf(msg_str," %s",PREC_NAME(VN_DECL_PREC(enp)));
		prt_msg_frag(msg_str);
	} else if( IS_DECL(code) ){
		sprintf(msg_str," %s",VN_STRING(enp));
		prt_msg_frag(msg_str);
	} else if( code==T_ADVISE ){
		/* BUG need to elim yylex_qsp */
		s=eval_string(VN_CHILD(enp,0));
		sprintf(msg_str,"\t\"%s\"",s);
		prt_msg_frag(msg_str);
	} else if( code==T_WARN ){
		/* BUG need to elim yylex_qsp */
		s=eval_string(VN_CHILD(enp,0));
		sprintf(msg_str,"\t\"%s\"",s);
		prt_msg_frag(msg_str);
	} else if( code==T_STRING ){
		sprintf(msg_str,"\t\"%s\"",VN_STRING(enp));
		prt_msg_frag(msg_str);
	} else if( code == T_LABEL || code ==T_GO_BACK || code == T_GO_FWD ){
		sprintf(msg_str," %s",VN_STRING(enp));
		prt_msg_frag(msg_str);
	} else if( code==T_LIT_DBL ){
		sprintf(msg_str," %g",VN_DBLVAL(enp));
		prt_msg_frag(msg_str);
	} else if( code == T_MATH0_FN ){
		sprintf(msg_str," %s",FUNC_NAME(VN_FUNC_PTR(enp)));
		prt_msg_frag(msg_str);
	} else if( code == T_MATH1_FN ){
		sprintf(msg_str," %s",FUNC_NAME(VN_FUNC_PTR(enp)));
		prt_msg_frag(msg_str);
	} else if( code == T_MATH2_FN ){
		sprintf(msg_str," %s",FUNC_NAME(VN_FUNC_PTR(enp)));
		prt_msg_frag(msg_str);
	} else if (
		   code == T_MATH0_VFN
		|| code == T_MATH1_VFN
		|| code == T_MATH2_VFN
		|| code == T_MATH2_VSFN
		|| code == T_CHAR_VFN
			/* BUG? shouldn't there bre a VSFN2 ??? */
		|| code == T_VS_FUNC
		|| code == T_VV_FUNC
		){
		sprintf(msg_str," %s",VF_NAME(FIND_VEC_FUNC(VN_VFUNC_CODE(enp))));
		prt_msg_frag(msg_str);
	} else if( code==T_CALLFUNC ){
assert(VN_SUBRT(enp)!=NULL);
		sprintf(msg_str," %s", SR_NAME(VN_SUBRT(enp)));
		prt_msg_frag(msg_str);
	} else if( code==T_LIT_INT ){
		sprintf(msg_str," %"PRId64, VN_INTVAL(enp) );
		prt_msg_frag(msg_str);
	} else if( code==T_ASSIGN ){
		prt_msg_frag("\t");
	} else if( code==T_MAXVAL ){
		prt_msg_frag("\t");
	} else if( code==T_MINVAL ){
		prt_msg_frag("\t");
	} else if( code==T_RAMP ){
		prt_msg_frag("\t");
	} else if( code == T_VS_VS_CONDASS ){
		sprintf(msg_str," %s",VF_NAME(FIND_VEC_FUNC(VN_BM_CODE(enp))));
		prt_msg_frag(msg_str);
	}

	/* Now print the addresses of the child nodes */

#define N_EXPECTED_CHILDREN(enp)	(tnt_tbl[code].tnt_nchildren)

	if( VN_CHILD(enp,0)!=NULL){
fprintf(stderr,"child 0 at 0x%lx\n",(long)VN_CHILD(enp,0));
		assert( N_EXPECTED_CHILDREN(enp) >= 1 );
		sprintf(msg_str,"\t\tn%d",VN_SERIAL(VN_CHILD(enp,0)));
		prt_msg_frag(msg_str);
	}
	for(i=1;i<MAX_CHILDREN(enp);i++){
		if( VN_CHILD(enp,i)!=NULL){
			assert( N_EXPECTED_CHILDREN(enp) > i );
			sprintf(msg_str,", n%d",VN_SERIAL(VN_CHILD(enp,i)));
			prt_msg_frag(msg_str);
		}
	}
	prt_msg("");

	if( SHOWING_SHAPES && VN_SHAPE(enp) != NULL ){
		prt_msg_frag("\t");
		if( OWNS_SHAPE(enp) ){
			sprintf(msg_str,"* 0x%lx  ",(u_long)VN_SHAPE(enp));
			prt_msg_frag(msg_str);
		}
		else {
			sprintf(msg_str,"@ 0x%lx  ",(u_long)VN_SHAPE(enp));
			prt_msg_frag(msg_str);
		}
		prt_msg_frag("\t");
		describe_shape(VN_SHAPE(enp));
	}

	if( SHOWING_RESOLVERS && VN_RESOLVERS(enp)!=NULL ){
		Node *np; Vec_Expr_Node *enp2;
		prt_msg("\tResolvers:");
		np=QLIST_HEAD(VN_RESOLVERS(enp));
		while(np!=NULL){
			enp2=(Vec_Expr_Node *)NODE_DATA(np);
			sprintf(msg_str,"\t\t%s",node_desc(enp2));
			prt_msg(msg_str);
			np=NODE_NEXT(np);
		}
	}
}

#define dump_subtree(enp) _dump_subtree(QSP_ARG  enp)

static void _dump_subtree(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int i;

	dump_node_with_shape(enp);

	for(i=0;i<MAX_CHILDREN(enp);i++){
		if( VN_CHILD(enp,i)!=NULL){
			dump_subtree(VN_CHILD(enp,i));
		}
	}
}


void _dump_tree_with_key(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	if( dump_flags & SHOW_KEY ){
		print_shape_key(SINGLE_QSP_ARG);
		dump_flags &= ~SHOW_KEY;	/* clear flag bit */
	}
	dumping=1;
	dump_subtree(enp);
	dumping=0;
}

void _dump_node_with_shape(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	if( dump_flags & SHOW_KEY ){
		print_shape_key(SINGLE_QSP_ARG);
		dump_flags &= ~SHOW_KEY;	/* clear flag bit */
	}
	dumping=1;
	dump_node_basic(enp);
	dumping=0;
}

/* We have a few buffers which we cycle, so we can have multiple descriptions in
 * a single printf...
 */

#define N_DESC	4
static int which_desc=0;

char *node_desc(Vec_Expr_Node *enp)
{
	static char desc_str[N_DESC][64];

	which_desc++;
	which_desc %= N_DESC;
	sprintf(desc_str[which_desc],"%s node ",NNAME(enp));
	prt_node(enp,&desc_str[which_desc][strlen(desc_str[which_desc])]);
	return(desc_str[which_desc]);
}

