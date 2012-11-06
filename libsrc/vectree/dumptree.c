#include "quip_config.h"

char VersionId_vectree_dumptree[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#include "savestr.h"
#include "data_obj.h"
#include "debug.h"
#include "getbuf.h"
#include "node.h"
#include "function.h"
#include "nexpr.h"
#include "nvf_api.h"
#include "query.h"

#include "vectree.h"

/* for definition of function codes */
/* #include "wartbl.h" */

static Keyword *curr_native_func_tbl=vt_native_func_tbl;

/* local prototypes */

static void _dump_tree(QSP_ARG_DECL  Vec_Expr_Node *);
#define _DUMP_TREE(enp)		_dump_tree(QSP_ARG  enp)
static void _dump_node(QSP_ARG_DECL  Vec_Expr_Node *enp);
#define _DUMP_NODE(enp)		_dump_node(QSP_ARG  enp)


/* Dump flags */

int dump_flags=0;

int dumping=0;

void set_native_func_tbl(Keyword *tbl)
{
	curr_native_func_tbl = tbl;
}

void set_show_shape(int flg)
{
	if( flg )
		dump_flags |= SHOW_SHAPES;
	else
		dump_flags &= ~SHOW_SHAPES;
}

void set_show_key(int flg)
{
	if( flg )	dump_flags |= SHOW_KEY;
	else		dump_flags &= ~SHOW_KEY;
}

void set_show_lhs_refs(int flg)
{
	if( flg )
		dump_flags |= SHOW_LHS_REFS;
	else
		dump_flags &= ~SHOW_LHS_REFS;
}

void set_show_cost(int flg)
{
	if( flg )
		dump_flags |= SHOW_COST;
	else
		dump_flags &= ~SHOW_COST;
}

void print_dump_legend(void)
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

void print_shape_key(void)
{
	prt_msg("_\tno shape\n?\tunknown shape\n*\towns shape\n@\tshape ref\n#\tunknown leaf");
}

static void prt_node(Vec_Expr_Node *enp,char *buf)
{
	int key;

	if( enp->en_shpp == NO_SHAPE ) key='_';
	else if( UNKNOWN_SHAPE(enp->en_shpp) ) key='?';
	else if ( OWNS_SHAPE(enp) ) key='*';
	else key='@';

	sprintf(buf,"n%-4d %c%c%c",enp->en_serial, key,
		RESOLVED_AT_CALLTIME(enp) ? '!' : ' ',
		HAS_CONSTANT_VALUE(enp) ? 'C' : ' '
		
		);
}

static void _dump_node(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Tree_Code code;
	int i;
	const char *s;

	if( enp==NO_VEXPR_NODE ) return;


	/* print the node "name", and a code that tells about shape knowledge */

	
	prt_node(enp,msg_str);
	prt_msg_frag(msg_str);

	if( SHOWING_LHS_REFS ){
		sprintf(msg_str,"\t%d",enp->en_lhs_refs);
		prt_msg_frag(msg_str);
	}

	if( SHOWING_COST ){
		if( enp->en_shpp != NO_SHAPE ){
			sprintf(msg_str,"\t%d", enp->en_shpp->si_n_mach_elts);
		}

		prt_msg_frag(msg_str);

		sprintf(msg_str,"\t%d\t%d", enp->en_flops,enp->en_nmath);
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

	code = enp->en_code;

	if( code==T_DYN_OBJ || code == T_UNDEF || code == T_PROTO || code==T_POINTER || code==T_FUNCPTR || code==T_STR_PTR ){
		sprintf(msg_str,"\t%s",enp->en_string);
		prt_msg_frag(msg_str);
		if( code == T_POINTER ){
			Identifier *idp;
			/* We don't use get_set_ptr() here because we don't want an error msg... */
			idp = ID_OF(enp->en_string);
			if( idp != NO_IDENTIFIER && IS_POINTER(idp) && POINTER_IS_SET(idp) ){
				if( idp->id_ptrp->ptr_refp == NO_REFERENCE ){
					/* how could this ever happen??? */
					prt_msg_frag("->???");
				} else {
					Data_Obj *dp;
					dp = idp->id_ptrp->ptr_refp->ref_dp;
					sprintf(msg_str,"->%s",dp->dt_name);
					prt_msg_frag(msg_str);
				}
			}
		}
	} else if( code == T_STATIC_OBJ ){
		sprintf(msg_str,"\t%s",enp->en_dp->dt_name);
		prt_msg_frag(msg_str);
	} else if ( code == T_FUNCREF ){
		Subrt *srp;
		srp=enp->en_srp;
		sprintf(msg_str,"\t%s",srp->sr_name);
		prt_msg_frag(msg_str);
	} else if( code == T_SIZE_FN ){
		sprintf(msg_str,"\t%s",size_functbl[enp->en_intval].fn_name);
		prt_msg_frag(msg_str);
	} else if(code == T_CALL_NATIVE ){
		sprintf(msg_str,"\t%s",curr_native_func_tbl[enp->en_intval].kw_token);
		prt_msg_frag(msg_str);
	} else if(code == T_TYPECAST ){
		sprintf(msg_str,"  %s",name_for_prec(enp->en_intval));
		prt_msg_frag(msg_str);
	} else if( code == T_SUBRT || code == T_SCRIPT ){
		Subrt *srp;
		srp=enp->en_srp;
		sprintf(msg_str,"\t%s",srp->sr_name);
		prt_msg_frag(msg_str);
	} else if( code==T_DECL_STAT ){
		sprintf(msg_str," %s",prec_name[enp->en_intval]);
		prt_msg_frag(msg_str);
	} else if( IS_DECL(code) ){
		sprintf(msg_str," %s",enp->en_string);
		prt_msg_frag(msg_str);
	} else if( code==T_ADVISE ){
		/* BUG need to elim yylex_qsp */
		s=eval_string(QSP_ARG  enp->en_child[0]);
		sprintf(msg_str,"\t\"%s\"",s);
		prt_msg_frag(msg_str);
	} else if( code==T_WARN ){
		/* BUG need to elim yylex_qsp */
		s=eval_string(QSP_ARG  enp->en_child[0]);
		sprintf(msg_str,"\t\"%s\"",s);
		prt_msg_frag(msg_str);
	} else if( code==T_STRING ){
		sprintf(msg_str,"\t\"%s\"",enp->en_string);
		prt_msg_frag(msg_str);
	} else if( code == T_LABEL || code ==T_GO_BACK || code == T_GO_FWD ){
		sprintf(msg_str," %s",enp->en_string);
		prt_msg_frag(msg_str);
	} else if( code==T_LIT_DBL ){
		sprintf(msg_str," %g",enp->en_dblval);
		prt_msg_frag(msg_str);
	} else if( code==T_MATH1_FN ){
		sprintf(msg_str," %s ",
			math1_functbl[enp->en_intval].fn_name
			);
		prt_msg_frag(msg_str);
	} else if( code == T_MATH0_FN ){
		sprintf(msg_str," %s",math0_functbl[enp->en_func_index].fn_name);
		prt_msg_frag(msg_str);
	} else if( code == T_MATH1_FN ){
		sprintf(msg_str," %s",math1_functbl[enp->en_func_index].fn_name);
		prt_msg_frag(msg_str);
	} else if( code == T_MATH2_FN ){
		sprintf(msg_str," %s",math2_functbl[enp->en_func_index].fn_name);
		prt_msg_frag(msg_str);
	} else if (
		   code == T_MATH0_VFN
		|| code == T_MATH1_VFN
		|| code == T_MATH2_VFN
		|| code == T_MATH2_VSFN
		|| code == T_VS_FUNC
		|| code == T_VV_FUNC
		){
		sprintf(msg_str," %s",vec_func_tbl[enp->en_vfunc_code].vf_name);
		prt_msg_frag(msg_str);
	} else if( code==T_CALLFUNC ){
		sprintf(msg_str," %s", ((Subrt *)enp->en_srp)->sr_name);
		prt_msg_frag(msg_str);
	} else if( code==T_LIT_INT ){
		sprintf(msg_str," %ld",enp->en_intval);
		prt_msg_frag(msg_str);
	} else if( code==T_ASSIGN ){
		prt_msg_frag("\t");
	} else if( code==T_MAXVAL ){
		prt_msg_frag("\t");
	} else if( code==T_MINVAL ){
		prt_msg_frag("\t");
	} else if( code==T_RAMP ){
		prt_msg_frag("\t");
	}

	/* Now print the addresses of the child nodes */

	if( enp->en_child[0]!=NO_VEXPR_NODE){
		sprintf(msg_str,"\t\tn%d",enp->en_child[0]->en_serial);
		prt_msg_frag(msg_str);
	}
	for(i=1;i<MAX_CHILDREN(enp);i++){
		if( enp->en_child[i]!=NO_VEXPR_NODE){
			sprintf(msg_str,", n%d",enp->en_child[i]->en_serial);
			prt_msg_frag(msg_str);
		}
	}
	prt_msg("");

	if( SHOWING_SHAPES && enp->en_shpp != NO_SHAPE ){
		prt_msg_frag("\t");
		if( OWNS_SHAPE(enp) ){
			sprintf(msg_str,"* 0x%lx  ",(u_long)enp->en_shpp);
			prt_msg_frag(msg_str);
		}
		else {
			sprintf(msg_str,"@ 0x%lx  ",(u_long)enp->en_shpp);
			prt_msg_frag(msg_str);
		}
		prt_msg_frag("\t");
		describe_shape(enp->en_shpp);
	}

	if( SHOWING_RESOLVERS && enp->en_resolvers!=NO_LIST ){
		Node *np; Vec_Expr_Node *enp2;
		prt_msg("\tResolvers:");
		np=enp->en_resolvers->l_head;
		while(np!=NO_NODE){
			enp2=(Vec_Expr_Node *)np->n_data;
			sprintf(msg_str,"\t\t%s",node_desc(enp2));
			prt_msg(msg_str);
			np=np->n_next;
		}
	}
}

static void _dump_tree(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int i;

	_dump_node(QSP_ARG  enp);

	for(i=0;i<MAX_CHILDREN(enp);i++){
		if( enp->en_child[i]!=NO_VEXPR_NODE){
			_DUMP_TREE(enp->en_child[i]);
		}
	}
}


void dump_tree(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	if( dump_flags & SHOW_KEY ){
		print_shape_key();
		dump_flags &= ~SHOW_KEY;	/* clear flag bit */
	}
	dumping=1;
	_DUMP_TREE(enp);
	dumping=0;
}

void dump_node(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	if( dump_flags & SHOW_KEY ){
		print_shape_key();
		dump_flags &= ~SHOW_KEY;	/* clear flag bit */
	}
	dumping=1;
	_DUMP_NODE(enp);
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

