#include "quip_config.h"

#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

#include "quip_prot.h"
#include "data_obj.h"
#include "vectree.h"
#include "query_stack.h"	// BUG?

/* for definition of function codes */
#include "veclib/vecgen.h"

#ifdef SGI
#include <alloca.h>
#endif

Vec_Expr_Node *special_enp=NULL;	// JUST FOR DEBUGGING!!!
Vec_Expr_Node *last_debugged_enp=NULL;	// JUST FOR DEBUGGING!!!

static Vec_Expr_Node *dup_node(QSP_ARG_DECL  Vec_Expr_Node *enp);
#define DUP_NODE(enp)			dup_node(QSP_ARG  enp)

#ifdef FOOBAR
// This appears in vectree.y, where it is used!?
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
#endif /* FOOBAR */

static int node_serial=1;

/* alloc_node doesn't do anything except grab some memory...
 * So it could just as well be a macro!
 */

static Vec_Expr_Node *alloc_node(void)
{
	Vec_Expr_Node *enp;

	enp = (Vec_Expr_Node *)getbuf(sizeof(*enp));
//fprintf(stderr,"alloc_node returning new node at 0x%lx\n",(long)enp);

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
		SET_VN_CHILD(enp,i,NO_VEXPR_NODE);
	SET_VN_PARENT(enp,NO_VEXPR_NODE);
	SET_VN_SHAPE(enp, NO_SHAPE);
	SET_VN_FLAGS(enp, 0);
	SET_VN_SERIAL(enp, node_serial++);
	SET_VN_LINENO(enp, PARSER_LINENO);	/* BUG need to point to filename also */
	SET_VN_INFILE(enp, CURR_INFILE);
	if( CURR_INFILE != NULL ){
		INC_SR_COUNT(CURR_INFILE);
	}
	SET_VN_RESOLVERS(enp, NULL);

	switch( VN_DATA_TYPE(enp) ){
		case ND_DECL:
//fprintf(stderr,"initializing ND_DECL data for node at 0x%lx, code = %d\n",(long)enp,VN_CODE(enp));
			SET_VN_DECL_NAME(enp, NULL);
			SET_VN_DECL_CTX(enp, NO_ITEM_CONTEXT);
			SET_VN_DECL_PREC(enp, NO_PRECISION );
			SET_VN_DECL_REFS(enp, NULL);
			SET_VN_DECL_FLAGS(enp, 0);
			SET_VN_DECL_OBJ(enp, NO_OBJ);
			break;
		case ND_LIST:
			SET_VN_N_ELTS(enp, 0);
			break;
		case ND_DBL:
			SET_VN_DBLVAL(enp, 0.0);
			break;
		case ND_INT:
			SET_VN_INTVAL(enp, 0);
			break;
		case ND_SUBRT:
			SET_VN_SUBRT(enp, NULL);
			break;
		case ND_CALLF:
			SET_VN_UK_ARGS(enp, NULL);
			SET_VN_CALL_SUBRT(enp, NULL);
			break;
		case ND_STRING:
			SET_VN_STRING(enp, NULL);
			break;
		case ND_CAST:
			SET_VN_CAST_PREC_PTR(enp, NO_PRECISION );
			break;
		case ND_VFUNC:
			SET_VN_VFUNC_CODE(enp, N_VEC_FUNCS);	/* illegal value? */
			break;
		case ND_FUNC:
			SET_VN_FUNC_PTR(enp, NULL);
			break;
		case ND_SIZE_CHANGE:
			SET_VN_SIZCH_SHAPE(enp, NO_SHAPE);
			break;
		case ND_BMAP:
			SET_VN_BM_CODE(enp, N_VEC_FUNCS);		/* illegal value */
			SET_VN_BM_SHAPE(enp, NO_SHAPE);
			break;

		case ND_NONE:
			/* this is legal and may happen */
			break;

		case ND_UNUSED:
		case N_NODE_DATA_TYPES:
			/* just here to suppress a compiler warning */
			assert( AERROR("init_expr_node:  bad data type code!?") );
			break;
	}
}

static const char *data_type_names[N_NODE_DATA_TYPES];

const char *node_data_type_desc(Node_Data_Type t)
{
	static int inited=0;

	if( ! inited ){
		assert( N_NODE_DATA_TYPES == 13 );

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
//		if( N_NODE_DATA_TYPES != 13 ){
//			NERROR1("CAUTIOUS:  check initialization of data_type_names");
//		}
	}

	return( data_type_names[t] );
}

/* set up a new node with the given code */

#define NOTHER_NODE(code)	nother_node(QSP_ARG  code)

static Vec_Expr_Node *nother_node(QSP_ARG_DECL  Tree_Code code)
{
	Vec_Expr_Node *enp;

	enp=alloc_node();
	SET_VN_CODE(enp,code);
	init_expr_node(QSP_ARG  enp);
	LAST_NODE=enp;
	return(enp);
}

// nother child sets the child field int the parent, and if the child is not NULL
// then sets the parent field in the child.

static void nother_child(Vec_Expr_Node * enp,Vec_Expr_Node * child,int index)
{
	SET_VN_CHILD(enp,index, child);
	if( child != NO_VEXPR_NODE ){
//#ifdef CAUTIOUS
		/* The strict tree structure is violated by plus-eq, do-while, etc */
		/* what does that comment mean??? */
		/* Perhaps something about how the nodes are created? */

		/*
//		if( VN_PARENT(child) != NO_VEXPR_NODE ){
//			sprintf(ERROR_STRING,
//				"CAUTIOUS:  nother_child:  node n%d (%s) has parent n%d (%s), rival n%d (%s)!?",
//				child->en_serial,
//				NNAME(child),
//				VN_PARENT(child)->en_serial,
//				NNAME(VN_PARENT(child)),
//				enp->en_serial,
//				NNAME(enp)
//				);
//			WARN(ERROR_STRING);
//		}
		assert( VN_PARENT(child) == NO_VEXPR_NODE );
		*/
//#endif /* CAUTIOUS */
		SET_VN_PARENT(child, enp);
	}
}

#define VERIFY_N_CHILDREN(code,n)							\
	/*if( tnt_tbl[code].tnt_nchildren != n ){						\
		sprintf(DEFAULT_ERROR_STRING,"%s node expects %d children, assigned %d!?",	\
		tnt_tbl[code].tnt_name,tnt_tbl[code].tnt_nchildren,n);			\
		NERROR1(DEFAULT_ERROR_STRING);							\
	}*/											\
	assert( tnt_tbl[code].tnt_nchildren == n );


Vec_Expr_Node *node3(QSP_ARG_DECL  Tree_Code code,Vec_Expr_Node *lchld,Vec_Expr_Node *rchld,Vec_Expr_Node *chld3)
{
	Vec_Expr_Node *enp;

//#ifdef CAUTIOUS
	VERIFY_N_CHILDREN(code,3);
//#endif /* CAUTIOUS */

	enp=NOTHER_NODE(code);
	nother_child(enp,lchld,0);
	nother_child(enp,rchld,1);
	nother_child(enp,chld3,2);

	return(enp);
}



Vec_Expr_Node *node2(QSP_ARG_DECL  Tree_Code code,Vec_Expr_Node *lchld,Vec_Expr_Node *rchld)
{
	Vec_Expr_Node *enp;

//#ifdef CAUTIOUS
	VERIFY_N_CHILDREN(code,2);
//#endif /* CAUTIOUS */

	enp=NOTHER_NODE(code);
	nother_child(enp,lchld,0);
	nother_child(enp,rchld,1);

	return(enp);
}


Vec_Expr_Node *node1(QSP_ARG_DECL  Tree_Code code,Vec_Expr_Node *lchld)
{
	Vec_Expr_Node *enp;

//#ifdef CAUTIOUS
	VERIFY_N_CHILDREN(code,1);
//#endif /* CAUTIOUS */

	enp=NOTHER_NODE(code);
	nother_child(enp,lchld,0);

	return(enp);
}


Vec_Expr_Node *node0(QSP_ARG_DECL  Tree_Code code)
{
//#ifdef CAUTIOUS
	VERIFY_N_CHILDREN(code,0);
//#endif /* CAUTIOUS */

	return( NOTHER_NODE(code) );
}

static Vec_Expr_Node *dup_node(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Vec_Expr_Node *new_enp;

	new_enp = NOTHER_NODE(VN_CODE(enp));
	/* now copy over the relevant data */
	switch(VN_CODE(enp)){
		case T_STRING:
		case T_DYN_OBJ:
		case T_POINTER:
			SET_VN_STRING(new_enp, savestr(VN_STRING(enp)));
			break;

		case T_STATIC_OBJ:
			SET_VN_OBJ(new_enp, VN_OBJ(enp));
			break;

		case T_LIT_DBL:
			SET_VN_DBLVAL(new_enp, VN_DBLVAL(enp));
			break;

		case T_LIT_INT:
			SET_VN_INTVAL(new_enp,VN_INTVAL(enp));
			break;

		case T_SIZE_FN:
			SET_VN_FUNC_PTR(new_enp, VN_FUNC_PTR(enp));
			break;

		case T_CALL_NATIVE:
		case T_MATH0_FN:
		case T_MATH0_VFN:
		case T_MATH1_FN:
		case T_MATH1_VFN:
			SET_VN_INTVAL(new_enp, VN_INTVAL(enp));
			break;

		case T_CALLFUNC:
			SET_VN_SUBRT(new_enp, VN_SUBRT(enp) );
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
		if( VN_CHILD(enp,i) != NO_VEXPR_NODE ){
			Vec_Expr_Node *new_child;
			new_child = DUP_TREE(VN_CHILD(enp,i));
			SET_VN_CHILD(new_enp,i, new_child);
			SET_VN_PARENT(new_child, new_enp);
		}

	return(new_enp);
}

void check_release(Vec_Expr_Node *enp)
{
	if( enp == NULL ) return;
	if( NODE_IS_FINISHED(enp) ){
		rls_vectree(enp);
	} else if( VN_CODE(enp) == T_STAT_LIST ){
		// Check children
		check_release(VN_CHILD(enp,0));
		check_release(VN_CHILD(enp,1));
	}
}

//#ifdef NOT_USED

/* We created this routine to get rid of function prototype arg decls...
 * But, in general this strategy is broken for fragments that are
 * multiply connected (e.g., child nodes with multiple parents)
 * because a child node could be visited twice!?
 * (how does that come about?)
 * Perhaps we could only release a child if the parent ptr
 * points to this node.
 * (what does that mean??)
 *
 * Another way to deal with that problem might be to have a reference count...
 */

void rls_vectree(Vec_Expr_Node *enp)
{
	int i;

//DEBUG_IT_3(enp,rls_vectree BEGIN)
//dump_tree(DEFAULT_QSP_ARG  enp);

// BUG?  we could use the expected number of children based on node code...
	for(i=0;i<MAX_CHILDREN(enp);i++)
		if( VN_CHILD(enp,i) != NO_VEXPR_NODE ){
//#ifdef CAUTIOUS
//			if( VN_PARENT(VN_CHILD(enp,i)) != enp ){
//				sprintf(DEFAULT_ERROR_STRING,
//	"CAUTIOUS:  rls_vectree:  %s child %d does not point to parent!?",
//					node_desc(enp),i);
//				NWARN(DEFAULT_ERROR_STRING);
//			}
//#endif // CAUTIOUS
			assert( VN_PARENT(VN_CHILD(enp,i)) == enp );
			rls_vectree(VN_CHILD(enp,i));
		}


	/* now release this node */
	if( VN_INFILE(enp) != NULL ){
		DEC_SR_COUNT(VN_INFILE(enp));
		// this gets done in vectree.y...
		/* if( SR_COUNT(VN_INFILE(enp)) == 0 ){ rls_stringref(VN_INFILE(enp)); } */
	}

	switch( VN_DATA_TYPE(enp) ){
		case ND_DECL:
// we used to dump here, but children have already been freed!
 //           fprintf(stderr,"VN_DECL_OBJ(0x%lx) = 0x%lx\n",(long)enp,(long)VN_DECL_OBJ(enp));
//if( VN_DECL_OBJ(enp) != NO_OBJ )
//longlist(DEFAULT_QSP_ARG  VN_DECL_OBJ(enp) );
//else
//fprintf(stderr,"VN_DECL_OBJ is NULL...\n");
//DEBUG_IT_3(enp,checking)
			// Does this get initialized?
			if( VN_DECL_OBJ(enp) != NO_OBJ )
				SET_OBJ_EXTRA(VN_DECL_OBJ(enp),NULL);
			break;
		case ND_LIST:
		case ND_DBL:
		case ND_INT:
		case ND_SUBRT:
		case ND_CALLF:
		case ND_STRING:
		case ND_CAST:
		case ND_VFUNC:
		case ND_FUNC:
		case ND_SIZE_CHANGE:
			break;

		case ND_BMAP:
			if( VN_BM_SHAPE(enp) != NO_SHAPE )
				RELEASE_SHAPE_PTR( VN_BM_SHAPE(enp) );
			break;
		case ND_NONE:
		case ND_UNUSED:
			break;

		case N_NODE_DATA_TYPES:
			/* just here to suppress a compiler warning */
//#ifdef CAUTIOUS
//			sprintf(DEFAULT_ERROR_STRING,
//		"CAUTIOUS:  init_expr_node:  %s has bad data type code %d",
//				node_desc(enp),VN_DATA_TYPE(enp) );
//			NERROR1(DEFAULT_ERROR_STRING);
//#endif /* CAUTIOUS */
			assert( AERROR("init_expr_node:  bad data type code!?") );
			break;
		/*
		default:
			sprintf(DEFAULT_ERROR_STRING,
"rls_vectree:  missing case for tree node data type %s",
tnt_tbl[VN_CODE(enp)].tnt_name);
			NWARN(DEFAULT_ERROR_STRING);
			break;
			*/
	}

	/* BUG we should check whether or not the node owns any lists,
	 * shape info, etc.
	 */
	switch(VN_CODE(enp)){
		case T_STRING:
		case T_LABEL:
		case T_PROTO:
		case T_BADNAME:
		case T_GO_BACK:
		case T_GO_FWD:
		case T_UNDEF:
		case T_DYN_OBJ:
		case T_STR_PTR:
		case T_POINTER:
		case T_SCAL_DECL:
			// free the string data
//DEBUG_IT_3(enp,releasing string)
			rls_str(VN_STRING(enp));
//DEBUG_IT_3(enp,done releasing string)
			break;

		case T_FUNCPTR_DECL:
		case T_CSCAL_DECL:
		case T_VEC_DECL:
		case T_CVEC_DECL:
		case T_IMG_DECL:
		case T_CIMG_DECL:
		case T_SEQ_DECL:
		case T_CSEQ_DECL:
		case T_PTR_DECL:
//DEBUG_IT_3(enp,releasing decl name)
//fprintf(stderr,"node at 0x%lx has decl name at 0x%lx\n",(long)enp,(long)VN_DECL_NAME(enp));
			rls_str(VN_DECL_NAME(enp));
//DEBUG_IT_3(enp,done releasing decl name)
			break;

		default:
			// do nothing?
			break;
	}
	if( OWNS_SHAPE(enp) ){
//DEBUG_IT_3(enp,releasing shape)
//fprintf(stderr,"rls_vectree releasing shape for %s node 0x%lx\n",node_desc(enp),(long)enp);
		rls_shape(VN_SHAPE(enp));
//DEBUG_IT_3(enp,done releasing shape)
	}
//DEBUG_IT_3(enp,releasing node itself)
	givbuf(enp);
//DEBUG_IT_3(enp,done releasing node itself)
}

void set_global_ctx(SINGLE_QSP_ARG_DECL)
{
	Item_Context *icp;

	icp = (Item_Context *)NODE_DATA(QLIST_TAIL(DOBJ_CONTEXT_LIST));
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"set_global_ctx:  pushing global context %s",CTX_NAME(icp));
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	//PUSH_ITEM_CONTEXT(DOBJ_ITEM_TYPE,icp);
	PUSH_DOBJ_CONTEXT(icp);
	icp = (Item_Context *)NODE_DATA(QLIST_TAIL(ID_CONTEXT_LIST));
	PUSH_ID_CONTEXT(icp);
}

void unset_global_ctx(SINGLE_QSP_ARG_DECL)
{
#ifdef QUIP_DEBUG
	Item_Context *icp;
	icp= pop_item_context(QSP_ARG  dobj_itp);
if( debug & scope_debug ){
sprintf(ERROR_STRING,"unset_global_ctx:  global context %s popped",CTX_NAME(icp));
advise(ERROR_STRING);
}
#else // ! QUIP_DEBUG
	pop_item_context(QSP_ARG  dobj_itp);
#endif /* QUIP_DEBUG */
	
	/*icp=*/ pop_item_context(QSP_ARG  id_itp);
}


void show_context_stack(QSP_ARG_DECL  Item_Type *itp)
{
	Node *np;
	Item_Context *icp;

	sprintf(ERROR_STRING,"Context stack for item type %s",IT_NAME(itp));
	advise(ERROR_STRING);

	np=QLIST_HEAD(CONTEXT_LIST(itp));

	if( np==NO_NODE ) {
		WARN("context list is empty");
		return;
	}
	icp=(Item_Context *)NODE_DATA(np);
	sprintf(ERROR_STRING,"%s (%s)",CTX_NAME(icp),CTX_NAME(CURRENT_CONTEXT(itp)));
	advise(ERROR_STRING);
	np=NODE_NEXT(np);
	while(np!=NO_NODE){
		icp=(Item_Context *)NODE_DATA(np);
		sprintf(ERROR_STRING,"%s",CTX_NAME(icp));
		advise(ERROR_STRING);
		np=NODE_NEXT(np);
	}
}

#ifdef CAUTIOUS
//static int not_printable(const char *s)
//{
//	while( *s ){
//		if( ! isprint(*s) ) return(1);
//		s++;
//	}
//	return(0);
//}

static int string_is_printable(const char *s)
{
	while( *s ){
		if( ! isprint(*s) ) return(0);
		s++;
	}
	return(1);
}

#endif /* CAUTIOUS */

void node_error(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	/* infile may be null if we are reading stdin??? */
	if( VN_INFILE(enp) == NULL ){
		return;
	}

//#ifdef CAUTIOUS
//	if( not_printable(SR_STRING(VN_INFILE(enp))) ){
//		dump_node(QSP_ARG  enp);
//		NERROR1("node infile not printable!?");
//	}
//#endif /* CAUTIOUS */
	assert( string_is_printable(SR_STRING(VN_INFILE(enp))) );

	sprintf(DEFAULT_ERROR_STRING,"File %s, line %d:",SR_STRING(VN_INFILE(enp)),VN_LINENO(enp));
	advise(DEFAULT_ERROR_STRING);
}

