#include "quip_config.h"

char VersionId_vectree_vecnodes[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

#include "savestr.h"		/* not needed? BUG */
#include "data_obj.h"
#include "debug.h"
#include "getbuf.h"
#include "node.h"
#include "function.h"
/* #include "warproto.h" */
#include "query.h"
#include "vectree.h"

/* for definition of function codes */
#include "vecgen.h"

#ifdef SGI
#include <alloca.h>
#endif

const char *curr_infile=NULL;
Vec_Expr_Node *last_node=NO_VEXPR_NODE;

static Vec_Expr_Node *dup_node(QSP_ARG_DECL  Vec_Expr_Node *enp);
#define DUP_NODE(enp)			dup_node(QSP_ARG  enp)

int whkeyword(Keyword *table,char *str)
{
	register int i;
	register Keyword *kwp;

	i=0;
	kwp=table;
	while( kwp->kw_code != -1 ){
		if( !strcmp(str,kwp->kw_token) ) return(i);
		kwp++;
		i++;
	}
	return(-1);
}

static int node_serial=1;

/* alloc_node doesn't do anything except grab some memory... */

static Vec_Expr_Node *alloc_node(void)
{
	Vec_Expr_Node *enp;

	enp = (Vec_Expr_Node *)getbuf(sizeof(*enp));

	/* initialize node structure fields */

	/* Do this after the code is set */
	/* init_expr_node(QSP_ARG  enp); */

	return(enp);
}

/* init_expr_node receives a virgin node with only the tree code filled in.
 * Here we initialize all the other fields.
 */

void init_expr_node(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	int i;

	for(i=0;i<MAX_NODE_CHILDREN;i++)
		enp->en_child[i]=NO_VEXPR_NODE;
	enp->en_parent=NO_VEXPR_NODE;
	enp->en_shpp = NO_SHAPE;
	enp->en_flags = 0;
	enp->en_serial = node_serial++;
	enp->en_lineno = PARSER_LINENO;	/* BUG need to point to filename also */
	enp->en_infile = curr_infile;
	enp->en_resolvers = NO_LIST;

	switch( tnt_tbl[enp->en_code].tnt_data_type ){
		case ND_DECL:
			enp->en_decl_icp = NO_ITEM_CONTEXT;
			enp->en_decl_ref_list = NO_LIST;
			enp->en_decl_name = NULL;
			enp->en_decl_prec = (-1);
			enp->en_decl_flags = 0;
			break;
		case ND_LIST:
			enp->en_n_elts = 0;
			break;
		case ND_DBL:
			enp->en_dblval = 0.0;
			break;
		case ND_INT:
			enp->en_intval = 0;
			break;
		case ND_SUBRT:
			enp->en_srp = NO_SUBRT;
			break;
		case ND_CALLF:
			enp->en_uk_args = NO_LIST;
			enp->en_call_srp = NO_SUBRT;
			break;
		case ND_STRING:
			enp->en_string = NULL;
			break;
		case ND_CAST:
			enp->en_cast_prec = (-1);
			break;
		case ND_VFUNC:
			enp->en_vfunc_code = N_VEC_FUNCS;	/* illegal value? */
			break;
		case ND_FUNC:
			enp->en_func_index = (-1);
			break;
		case ND_SIZE_CHANGE:
			enp->en_child_shpp = NO_SHAPE;
			break;
		case ND_BMAP:
			enp->en_bm_code = N_VEC_FUNCS;		/* illegal value */
			enp->en_bm_child_shpp = NO_SHAPE;
			break;

		case ND_NONE:
			/* this is legal and may happen */
			break;

		case ND_UNUSED:
		case N_NODE_DATA_TYPES:
			/* just here to suppress a compiler warning */
#ifdef CAUTIOUS
			sprintf(DEFAULT_ERROR_STRING,
		"CAUTIOUS:  init_expr_node:  %s has bad data type code %d",
				node_desc(enp),tnt_tbl[enp->en_code].tnt_data_type);
			NERROR1(DEFAULT_ERROR_STRING);
#endif /* CAUTIOUS */
			break;
	}

	/*
	enp->en_ref_list = NO_LIST;
	enp->en_uk_args = NO_LIST;
	*/
}

static const char *data_type_names[N_NODE_DATA_TYPES];

const char *node_data_type_desc(Node_Data_Type t)
{
	static int inited=0;

	if( ! inited ){
		data_type_names[ ND_UNUSED ] = "unused";
		data_type_names[ ND_NONE ] = "none";
		data_type_names[ ND_LIST ] = "list";
		data_type_names[ ND_DBL ] = "dbl";
		data_type_names[ ND_INT ] = "int";
		data_type_names[ ND_SUBRT ] = "subrt";
		data_type_names[ ND_CALLF ] = "callf";
		data_type_names[ ND_STRING ] = "string";
		data_type_names[ ND_CAST ] = "cast";
		data_type_names[ ND_FUNC ] = "func";
		data_type_names[ ND_DECL ] = "decl";
		data_type_names[ ND_SIZE_CHANGE ] = "size_change";
		data_type_names[ ND_BMAP ] = "bmap";
		if( N_NODE_DATA_TYPES != 13 ){
			NERROR1("CAUTIOUS:  check initialization of data_type_names");
		}
	}

	return( data_type_names[t] );
}

/* set up a new node with the given code */

#define NOTHER_NODE(code)	nother_node(QSP_ARG  code)

static Vec_Expr_Node *nother_node(QSP_ARG_DECL  Tree_Code code)
{
	Vec_Expr_Node *enp;

	enp=alloc_node();
	enp->en_code=code;
	init_expr_node(QSP_ARG  enp);
	last_node=enp;
	return(enp);
}

static void nother_child(Vec_Expr_Node * enp,Vec_Expr_Node * child,int index)
{
	enp->en_child[index] = child;
	if( child != NO_VEXPR_NODE ){
#ifdef CAUTIOUS
		/* The strict tree structure is violated by plus-eq, do-while, etc */
		/*
		if( child->en_parent != NO_VEXPR_NODE ){
			sprintf(error_string,
				"CAUTIOUS:  nother_child:  node n%d (%s) has parent n%d (%s), rival n%d (%s)!?",
				child->en_serial,
				NNAME(child),
				child->en_parent->en_serial,
				NNAME(child->en_parent),
				enp->en_serial,
				NNAME(enp)
				);
			WARN(error_string);
		}
		*/
#endif /* CAUTIOUS */
		child->en_parent = enp;
	}
}

#define VERIFY_N_CHILDREN(code,n)							\
	if( tnt_tbl[code].tnt_nchildren != n ){						\
		sprintf(DEFAULT_ERROR_STRING,"%s node expects %d children, assigned %d!?",	\
		tnt_tbl[code].tnt_name,tnt_tbl[code].tnt_nchildren,n);			\
		NERROR1(DEFAULT_ERROR_STRING);							\
	}


Vec_Expr_Node *node3(QSP_ARG_DECL  Tree_Code code,Vec_Expr_Node *lchld,Vec_Expr_Node *rchld,Vec_Expr_Node *chld3)
{
	Vec_Expr_Node *enp;

#ifdef CAUTIOUS
	VERIFY_N_CHILDREN(code,3);
#endif /* CAUTIOUS */

	enp=NOTHER_NODE(code);
	nother_child(enp,lchld,0);
	nother_child(enp,rchld,1);
	nother_child(enp,chld3,2);

	return(enp);
}



Vec_Expr_Node *node2(QSP_ARG_DECL  Tree_Code code,Vec_Expr_Node *lchld,Vec_Expr_Node *rchld)
{
	Vec_Expr_Node *enp;

#ifdef CAUTIOUS
	VERIFY_N_CHILDREN(code,2);
#endif /* CAUTIOUS */

	enp=NOTHER_NODE(code);
	nother_child(enp,lchld,0);
	nother_child(enp,rchld,1);

	return(enp);
}


Vec_Expr_Node *node1(QSP_ARG_DECL  Tree_Code code,Vec_Expr_Node *lchld)
{
	Vec_Expr_Node *enp;

#ifdef CAUTIOUS
	VERIFY_N_CHILDREN(code,1);
#endif /* CAUTIOUS */

	enp=NOTHER_NODE(code);
	nother_child(enp,lchld,0);

	return(enp);
}


Vec_Expr_Node *node0(QSP_ARG_DECL  Tree_Code code)
{
#ifdef CAUTIOUS
	VERIFY_N_CHILDREN(code,0);
#endif /* CAUTIOUS */

	return( NOTHER_NODE(code) );
}

static Vec_Expr_Node *dup_node(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Vec_Expr_Node *new_enp;

	new_enp = NOTHER_NODE(enp->en_code);
	/* now copy over the relevant data */
	switch(enp->en_code){
		case T_STRING:
		case T_DYN_OBJ:
		case T_POINTER:
			new_enp->en_string = savestr(enp->en_string);
			break;

		case T_STATIC_OBJ:
			new_enp->en_dp = enp->en_dp;
			break;

		case T_LIT_DBL:
			new_enp->en_dblval = enp->en_dblval;
			break;

		case T_LIT_INT:
			new_enp->en_intval = enp->en_intval;
			break;

		case T_SIZE_FN:
			new_enp->en_func_index = enp->en_func_index;
			break;

		case T_CALL_NATIVE:
		case T_MATH0_FN:
		case T_MATH0_VFN:
		case T_MATH1_FN:
		case T_MATH1_VFN:
			new_enp->en_intval = enp->en_intval;
			break;

		case T_CALLFUNC:
			new_enp->en_srp = enp->en_srp;
			break;

		/* safe do-nothing cases */
		case T_ENTIRE_RANGE:
		case T_DEREFERENCE:
		case T_MINUS:
		case T_PLUS:
		case T_CURLY_SUBSCR:
		case T_SQUARE_SUBSCR:
		case T_IMAG_PART:
		case T_SUBVEC:
		case T_CSUBVEC:
		case T_DIVIDE:
		case T_INNER:

		case T_ASSIGN:
		/* matlab junk */
		case T_ROW:
		/* case T_ROWLIST: */
		case T_UMINUS:
		case T_TIMES:
		/* matlab */
		case T_NOP:
		case T_I:
		case T_SUBSCRIPT1:
		case T_RANGE:
		case T_RANGE2:
		case T_FIRST_INDEX:
		case T_LAST_INDEX:
		case T_SUBMTRX:
		case T_INDEX_SPEC:
		case T_INDEX_LIST:
		case T_EXPR_LIST:
		case T_ARGLIST:
			break;

		default:
			MISSING_CASE(enp,"dup_node");
			break;
	}
	return(new_enp);
}

Vec_Expr_Node *dup_tree(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Vec_Expr_Node *new_enp;
	int i;

	new_enp = DUP_NODE(enp);

	for(i=0;i<MAX_CHILDREN(enp);i++)
		if( enp->en_child[i] != NO_VEXPR_NODE ){
			Vec_Expr_Node *new_child;
			new_child = DUP_TREE(enp->en_child[i]);
			new_enp->en_child[i] = new_child;
			new_child->en_parent = new_enp;
		}

	return(new_enp);
}

/* We created this routine to get rid of function prototype arg decls...
 * But, in general this strategy is broken for fragments that are
 * multiply connected (e.g., child nodes with multiple parents)
 * because a child node could be visited twice!?
 * Perhaps we could only release a child if the parent ptr
 * points to this node.
 */

void rls_tree(Vec_Expr_Node *enp)
{
	int i;

	for(i=0;i<MAX_CHILDREN(enp);i++)
		if( enp->en_child[i] != NO_VEXPR_NODE && 
			enp->en_child[i]->en_parent == enp )

			rls_tree(enp->en_child[i]);


	/* now release this node */

	/* BUG we should check whether or not the node owns any lists,
	 * shape info, etc.
	 */
	givbuf(enp);

}

void set_global_ctx(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp;

	icp = (Item_Context *)CONTEXT_LIST(dobj_itp)->l_tail->n_data;
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"set_global_ctx:  pushing global context %s",icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */
	PUSH_ITEM_CONTEXT(dobj_itp,icp);
	icp = (Item_Context *)CONTEXT_LIST(id_itp)->l_tail->n_data;
	PUSH_ITEM_CONTEXT(id_itp,icp);
}

void unset_global_ctx(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp;
	icp=pop_item_context(QSP_ARG  dobj_itp);
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"unset_global_ctx:  global context %s popped",icp->ic_name);
advise(error_string);
}
#endif /* DEBUG */
	icp=pop_item_context(QSP_ARG  id_itp);
}


void show_context_stack(QSP_ARG_DECL  Item_Type *itp)
{
	Node *np;
	Item_Context *icp;

	sprintf(error_string,"Context stack for item type %s",itp->it_name);
	advise(error_string);

	np=CONTEXT_LIST(itp)->l_head;

	if( np==NO_NODE ) {
		WARN("context list is empty");
		return;
	}
	icp=(Item_Context *)np->n_data;
	sprintf(error_string,"%s (%s)",icp->ic_name,CURRENT_CONTEXT(itp)->ic_name);
	advise(error_string);
	np=np->n_next;
	while(np!=NO_NODE){
		icp=(Item_Context *)np->n_data;
		sprintf(error_string,"%s",icp->ic_name);
		advise(error_string);
		np=np->n_next;
	}
}

#ifdef CAUTIOUS
int not_printable(const char *s)
{
	while( *s ){
		if( ! isprint(*s) ) return(1);
		s++;
	}
	return(0);
}
#endif /* CAUTIOUS */

void node_error(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	/* infile may be null if we are reading stdin??? */
	if( enp->en_infile == NULL ){
sprintf(DEFAULT_ERROR_STRING,"node_error:  en_infile is NULL??");
advise(DEFAULT_ERROR_STRING);
		return;
	}

#ifdef CAUTIOUS
	if( not_printable(enp->en_infile) ){
		dump_node(QSP_ARG  enp);
sprintf(DEFAULT_ERROR_STRING,"trying:  \"%s\"",enp->en_infile);
advise(DEFAULT_ERROR_STRING);
		NERROR1("node infile not printable!?");
	}
#endif /* CAUTIOUS */

	sprintf(DEFAULT_ERROR_STRING,"File %s, line %d:",enp->en_infile,enp->en_lineno);
	advise(DEFAULT_ERROR_STRING);
}

