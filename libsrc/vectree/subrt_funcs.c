#include "quip_config.h"

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "quip_prot.h"
#include "vectree.h"
#include "subrt.h"
#include "query_stack.h"	// BUG


/* We call this after we scan the body of a subroutine for which
 * we already had a prototype.
 */

void _update_subrt(QSP_ARG_DECL  Subrt *srp, Vec_Expr_Node *body )
{
	if( SR_BODY(srp) != NULL ){
		node_error(body);
		warn("subroutine body is not null!?");
	}

	SET_SR_BODY(srp, body);
}

Subrt * _remember_subrt(QSP_ARG_DECL  Precision * prec_p,const char *name,Vec_Expr_Node *args,Vec_Expr_Node *body)
{
	Subrt *srp;
	int i;

	srp=new_subrt(name);
	if( srp==NULL ) return(NULL);

	SET_SR_ARG_DECLS(srp, args);
	SET_SR_BODY(srp, body);
	SET_SR_RET_LIST(srp, NULL);
	SET_SR_CALL_LIST(srp, NULL);
	SET_SR_PREC_PTR(srp, prec_p);
	// Now only subroutine calls have a shape...
//	/* We used to give void subrt's a null shape ptr... */
//	SET_SR_SHAPE(srp, ALLOC_SHAPE );
//	COPY_SHAPE(SR_SHAPE(srp), uk_shape(PREC_CODE(prec_p)));
//	if( PREC_CODE(prec_p) == PREC_VOID ){
//		CLEAR_SHP_FLAG_BITS(SR_SHAPE(srp), DT_UNKNOWN_SHAPE);
//	}
	SET_SR_FLAGS(srp, 0);
//	SET_SR_CALL_VN(srp, NULL);

	for(i=0;i<N_PLATFORM_TYPES;i++)
		SET_SR_KERNEL_INFO_PTR(srp,i,NULL);

	return(srp);
}

#define get_scalar_arg(prec_p, prompt) _get_scalar_arg(QSP_ARG  prec_p, prompt)

static Vec_Expr_Node *_get_scalar_arg(QSP_ARG_DECL  Precision *prec_p, const char *prompt)
{
	Scalar_Value sv;
	Vec_Expr_Node *enp;

	(*(prec_p->set_value_from_input_func))(QSP_ARG  &sv, prompt);

	switch( PREC_CODE(prec_p) ){
		case PREC_SP:
		case PREC_DP:
			enp = node0(T_LIT_DBL);
			SET_VN_DBLVAL(enp,(*(prec_p->cast_to_double_func))(QSP_ARG  &sv));
			break;
		case PREC_BY: case PREC_IN: case PREC_DI: case PREC_LI:
		case PREC_UBY: case PREC_UIN: case PREC_UDI: case PREC_ULI:
			enp = node0(T_LIT_INT);
			// BUG should cast to long not int???
			SET_VN_INTVAL(enp,(int) (*(prec_p->cast_to_double_func))(QSP_ARG  &sv));
			break;
		default:
			fprintf(stderr,"get_scalar_arg:  unhandled precision %s!?",PREC_NAME(prec_p));
			enp = NULL;
			break;
	}
	return enp;
}

#define get_one_arg(enp, prec_p) _get_one_arg(QSP_ARG  enp, prec_p)

static Vec_Expr_Node *_get_one_arg(QSP_ARG_DECL  Vec_Expr_Node *enp, Precision *prec_p)
{
	const char *s;
	Data_Obj *dp;

	Vec_Expr_Node *ret_enp=NULL;
	switch(VN_CODE(enp)){
		case T_PTR_DECL:
			sprintf(msg_str,"object for %s * %s",
				PREC_NAME(prec_p),
				VN_DECL_NAME(enp)
				);
			s = nameof(msg_str);
			dp = get_obj(s);
			if( dp != NULL ){
				Vec_Expr_Node *obj_enp;
				obj_enp=node0(T_STATIC_OBJ);
				SET_VN_OBJ(obj_enp, dp);
				point_node_shape(obj_enp,OBJ_SHAPE(dp));
				SET_VN_PFDEV(obj_enp,OBJ_PFDEV(dp));
				ret_enp = node1(T_REFERENCE,obj_enp);
				point_node_shape(ret_enp,OBJ_SHAPE(dp));
				SET_VN_PFDEV(ret_enp,OBJ_PFDEV(dp));
			}
			break;
		case T_SCAL_DECL:
			sprintf(msg_str,"%s scalar for %s",
				PREC_NAME(prec_p),
				VN_DECL_NAME(enp)
				);
			ret_enp = get_scalar_arg(prec_p, msg_str);
			point_node_shape(ret_enp,scalar_shape(PREC_CODE(prec_p)));
			break;
		default:
			fprintf(stderr,"get_one_arg:  unhandled case %s\n",node_desc(enp));
			break;
	}
	return ret_enp;
}

#ifdef HAVE_ANY_GPU

#define pfdev_for_node(enp) _pfdev_for_node(QSP_ARG  enp)

static Platform_Device *_pfdev_for_node(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Platform_Device *pdp=NULL;

	switch( VN_CODE(enp) ){
		case T_STATIC_OBJ:
			assert(VN_OBJ(enp)!=NULL);
			pdp = OBJ_PFDEV( VN_OBJ(enp) );
			break;
		// no-ops
		case T_LIT_DBL:
		case T_LIT_INT:
		case T_ARGLIST:
		case T_REFERENCE:
			break;
		default:
			sprintf(ERROR_STRING,"Missing case for %s in pfdev_for_node!?",
				node_desc(enp));
			warn(ERROR_STRING);
			break;
	}
	return pdp;
}

void _update_pfdev_from_children(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Platform_Device *pdp=NULL;
	Vec_Expr_Node *defining_enp=NULL;
	int curdled=0;
	int i;

	for(i=0;i<MAX_NODE_CHILDREN;i++){	// BUG - node should record number of children...
		if( VN_CHILD(enp,i) == NULL ){
			i = MAX_NODE_CHILDREN;	// terminate loop
			continue;
		} else {
			if( VN_PFDEV( VN_CHILD(enp,i) ) == NULL ){
				// recursive call
				update_pfdev_from_children(VN_CHILD(enp,i) );
			}
			if( VN_PFDEV( VN_CHILD(enp,i) ) != NULL && pdp == NULL ){
				pdp = VN_PFDEV( VN_CHILD(enp,i) );
				defining_enp = VN_CHILD(enp,i);
			}

			// If the device is still null then it is probably a constant
			// that works on any device...
			if( VN_PFDEV( VN_CHILD(enp,i) ) != NULL &&
					pdp != VN_PFDEV( VN_CHILD(enp,i) ) ){
				sprintf(ERROR_STRING,"Platform mismatch:  %s (%s) and %s (%s)!?",
					node_desc(defining_enp),PFDEV_NAME(pdp),
					node_desc(VN_CHILD(enp,i)), PFDEV_NAME(VN_PFDEV(VN_CHILD(enp,i))) );
				warn(ERROR_STRING);
				curdled=1;
			}
		}
	}
	// Now handle special cases
	if( pdp == NULL ){
		pdp = pfdev_for_node(enp);
	}

	// Some nodes have no children, and constant nodes have no associated device???
	if( curdled /* || pdp==NULL */ ){
		return;
	}
	SET_VN_PFDEV(enp,pdp);
} // update_pfdev_from_children
#endif // HAVE_ANY_GPU

#define get_subrt_arg_tree(enp) _get_subrt_arg_tree(QSP_ARG  enp)

static Vec_Expr_Node * _get_subrt_arg_tree(QSP_ARG_DECL  Vec_Expr_Node *enp)
{
	Vec_Expr_Node *return_enp=NULL;
	Vec_Expr_Node *enp1, *enp2;

	switch(VN_CODE(enp)){
		case T_DECL_STAT:
			assert( VN_CHILD(enp,0) != NULL );
			return_enp = get_one_arg(VN_CHILD(enp,0), VN_DECL_PREC(enp));
			break;
		case T_DECL_STAT_LIST:
			enp1 = get_subrt_arg_tree(VN_CHILD(enp,0));
			enp2 = get_subrt_arg_tree(VN_CHILD(enp,1));
			// BUG release good node if only one bad
			if( enp1 != NULL && enp2 != NULL ){
				return_enp = node2(T_ARGLIST,enp1,enp2);
#ifdef HAVE_ANY_GPU
				update_pfdev_from_children(return_enp);
#endif // HAVE_ANY_GPU
			}
			break;
			
		default:
			sprintf(ERROR_STRING,"get_subrt_arg_tree:  unhandled case %s",node_desc(enp));
			warn(ERROR_STRING);
			break;
	}
	return return_enp;
}

#define get_subrt_args(srp) _get_subrt_args(QSP_ARG  srp)

static Vec_Expr_Node * _get_subrt_args(QSP_ARG_DECL  Subrt *srp)
{
	Vec_Expr_Node *enp;

	enp = get_subrt_arg_tree(SR_ARG_DECLS(srp));
	return enp;
}

COMMAND_FUNC( do_run_subrt )
{
	Subrt *srp;
	Vec_Expr_Node *args_enp;
	Vec_Expr_Node *call_enp;

	srp=pick_subrt("");

	if( srp==NULL ) return;

	// What do we do if there is a fused kernel for this subrt???

	push_vector_parser_data(SINGLE_QSP_ARG);
	args_enp = get_subrt_args(srp);
	call_enp = node1(T_CALLFUNC,args_enp);

	run_subrt_immed(srp,NULL,call_enp);
	pop_vector_parser_data(SINGLE_QSP_ARG);
}

COMMAND_FUNC( do_dump_subrt )
{
	Subrt *srp;

	srp=pick_subrt("");

	if( srp==NULL ) return;
	dump_subrt(srp);
}

void _dump_subrt(QSP_ARG_DECL Subrt *srp)
{
	if( IS_SCRIPT(srp) ){
		sprintf(msg_str,"Script subrt %s:",SR_NAME(srp));
		prt_msg(msg_str);
		prt_msg((char *)SR_BODY(srp));
		return;
	}

	if( SR_ARG_DECLS(srp) != NULL ){
		sprintf(msg_str,"Subrt %s arg declarations:\n",SR_NAME(srp));
		prt_msg(msg_str);
		print_dump_legend(SINGLE_QSP_ARG);
		dump_tree(SR_ARG_DECLS(srp));
	}

	if( SR_BODY(srp) != NULL ){
		sprintf(msg_str,"Subrt %s body:\n",SR_NAME(srp));
		prt_msg(msg_str);
		print_dump_legend(SINGLE_QSP_ARG);
		dump_tree(SR_BODY(srp));
	}
}

COMMAND_FUNC( do_opt_subrt )
{
	Subrt *srp;

	srp=pick_subrt("");
	if( srp==NULL ) return;

	optimize_subrt(srp);
}

COMMAND_FUNC( do_fuse_kernel )
{
	Subrt *srp;

	srp=pick_subrt("");
	if( srp==NULL ) return;

	fuse_subrt(srp);
}

COMMAND_FUNC( do_tell_cost )
{
	Subrt *srp;

	srp=pick_subrt("");

	if( srp==NULL ) return;

	tell_cost(srp);
}

COMMAND_FUNC( do_subrt_info )
{
	Subrt *srp;
	Vec_Expr_Node *enp;
	Node *np;

	srp=pick_subrt("");

	if( srp==NULL ) return;

	if( IS_SCRIPT(srp) ){
		sprintf(msg_str,"\nScript subroutine %s, %d arguments:\n",SR_NAME(srp),SR_N_ARGS(srp));
		prt_msg(msg_str);

		prt_msg((char *)SR_BODY(srp));
		return;
	}

	sprintf(msg_str,"\nSubroutine %s:",SR_NAME(srp));
	prt_msg(msg_str);

	enp = SR_BODY(srp);

	if( VN_SHAPE(enp) != NULL ){
		describe_shape(VN_SHAPE(enp));
	} else prt_msg("shape not determinable");

	sprintf(msg_str,"\t%d arguments",SR_N_ARGS(srp));
	prt_msg(msg_str);

	if( SR_RET_LIST(srp) != NULL ){
		sprintf(msg_str,"%d unknown return shape nodes:",eltcount(SR_RET_LIST(srp)));
		prt_msg(msg_str);
		np=QLIST_HEAD(SR_RET_LIST(srp));
		while(np!=NULL){
			Vec_Expr_Node *ret_enp;
			ret_enp = (Vec_Expr_Node *)NODE_DATA(np);
			sprintf(msg_str,"\tn%d:",VN_SERIAL(ret_enp));
			prt_msg(msg_str);
			dump_tree(ret_enp);
			np=NODE_NEXT(np);
		}
	}
	if( SR_CALL_LIST(srp) != NULL ){
		sprintf(msg_str,"%d unknown callfunc shape nodes:",eltcount(SR_CALL_LIST(srp)));
		prt_msg(msg_str);
		np=QLIST_HEAD(SR_CALL_LIST(srp));
		while(np!=NULL){
			Vec_Expr_Node *c_enp;
			c_enp = (Vec_Expr_Node *)NODE_DATA(np);
			sprintf(msg_str,"\tn%d:",VN_SERIAL(c_enp));
			prt_msg(msg_str);
			dump_tree(c_enp);
			np=NODE_NEXT(np);
		}
	}

	// Display the file and line number where declared...
	if( VN_INFILE(enp) == NULL ){
		warn("subroutine has no associated input file!?");
	} else {
		assert( string_is_printable(SR_STRING(VN_INFILE(enp))) );
		sprintf(msg_str,"Subroutine %s declared at line %d, file %s",
			SR_NAME(srp),
			VN_LINENO(enp),
			SR_STRING(VN_INFILE(enp))
			);
		prt_msg(msg_str);
	}


	/*
	sprintf(msg_str,"\t%ld flops, %ld math calls",
		enp->en_flops,enp->en_nmath);
	prt_msg(msg_str);

	*/
}

Subrt *_create_script_subrt(QSP_ARG_DECL  const char *name,int nargs,const char *text)
{
	Subrt *srp;

	srp = remember_subrt(PREC_FOR_CODE(PREC_VOID),name,NULL,NULL);
	if( srp == NULL ) return srp;

	SET_SR_FLAG_BITS(srp, SR_SCRIPT);
	SET_SR_N_ARGS(srp, nargs);

	SET_SR_TEXT(srp, savestr(text));
	return srp;
}

static void insure_subrt_ctx_stack(SINGLE_QSP_ARG_DECL)
{
	assert(THIS_VPD!=NULL);
	if( SUBRT_CTX_STACK == NULL ){
		SET_SUBRT_CTX_STACK(new_list());
		// BUG?  freed?
	}
}

// Make up a unique string by concatenating all the strings in the context stack
// BUG should use String_Bufs to avoid buffer overflow!?

static const char *name_for_ctx_stack(SINGLE_QSP_ARG_DECL)
{
	static char ctxname[LLEN];
	Node *np;

	ctxname[0]=0;
	np=QLIST_HEAD(SUBRT_CTX_STACK);
	while( np != NULL ){
		/* BUG check for string overflow */
		if( strlen(ctxname) > 0 ) strcat(ctxname,".");
		strcat(ctxname,(char *)NODE_DATA(np));
		np=NODE_NEXT(np);
	}
	return(ctxname);
}

/* get_surt_id - get a unique name for the subroutine context
 *
 * The original scheme of naming the context Subr.subrtname
 * is no good, because it can't handle multiple instances,
 * as occur with recursion or multi-threading.
 * We might handle recursion by having a current subroutine
 * context (for each thread!)
 */

#define get_subrt_id(name) _get_subrt_id(QSP_ARG  name)

static const char *_get_subrt_id(QSP_ARG_DECL  const char *name)
{
	Node *np;
	const char *s;

	assert(THIS_VPD != NULL);

	insure_subrt_ctx_stack(SINGLE_QSP_ARG);
	s=savestr(name);
	np=mk_node((void *)s);
	addTail(SUBRT_CTX_STACK,np);

	return(name_for_ctx_stack(SINGLE_QSP_ARG));
}

/* set_subrt_ctx - set subroutine context
 * Creates a data object context for this subroutine, so that names
 * can have sensible scoping, and so that all the objects created
 * inside a subrt can be easily deleted on exit...
 *
 * This gets called 3 times:
 * 1st, from an embedded action in the parser, when the subroutine
 * is first read in; 2nd, at the beginning of scan_subrt;
 * 3rd, at the beginning of run_subrt().
 *
 * This pushing and popping of dobj contexts seems likely to fail
 * in a multi-threaded environment unless the context stacks
 * are per-qsp!?  But they are!  (see ITCI - item type context info)
 */

void _set_subrt_ctx(QSP_ARG_DECL  const char *name)
{
	const char *ctxname;
	Item_Context *icp;	/* data_obj, identifier context */

	ctxname = get_subrt_id(name);
	icp=create_id_context(ctxname);
	PUSH_ID_CONTEXT(icp);

	icp=create_dobj_context(ctxname);
	push_dobj_context(icp);
}

static void rls_reference(Reference *refp)
{
	rls_stringbuf( REF_SBUF(refp) );
	givbuf(refp);
}

// We may not need to do any of this on objC ???
//
// really a memory release function...

void _delete_id(QSP_ARG_DECL  Item *ip)
{
	Identifier *idp;

	idp = (Identifier *)ip;

	switch(ID_TYPE(idp)){
		case ID_OBJ_REF:
			/* We used to call delvec(ID_REF(idp)->ref_dp)... */
			/* We don't do this, because this case occurs in an export/unexport cycle */
			givbuf(ID_REF(idp));
			break;
		case ID_STRING:
			rls_reference(ID_REF(idp));
			break;
#ifdef SCALARS_NOT_OBJECTS
//		case ID_SCALAR:
//			givbuf(ID_SVAL_PTR(idp));
//			break;
#endif // SCALARS_NOT_OBJECTS
		case ID_POINTER:
			givbuf(ID_PTR(idp));
			break;
		case ID_FUNCPTR:
			givbuf(ID_FUNC(idp));
			break;
		case ID_LABEL:
			break;

		default:
			sprintf(ERROR_STRING,"delete_id:  unhandled id type %d",ID_TYPE(idp));
			warn(ERROR_STRING);
			break;
	}
	del_id(idp);	// releases name for us
}

// This function is called when we delete an object declared in a subroutine...

static void clear_decl_obj( Item *ip )
{
	Vec_Expr_Node *enp;
	Data_Obj *dp;

	dp = (Data_Obj *) ip;
	enp = (Vec_Expr_Node *) OBJ_EXTRA(dp);

//fprintf(stderr,"clear_decl_obj:  obj is %s\n",OBJ_NAME(dp));
	// This assertion fails when a local temp obj is passed???
//	assert( enp != NULL );
	if( enp == NULL ){
		// This should be a temp object that we created,
		// with name of the form L.[0-9]*
		assert( strncmp(OBJ_NAME(dp),"L.",2) == 0 );
		return;
	}

	assert( VN_DECL_OBJ(enp) == dp );

	SET_VN_DECL_OBJ(enp,NULL);
}

/* When we exit a subroutine, we call this to automatically destroy
 * all of the objects it created.
 */

void _delete_subrt_ctx(QSP_ARG_DECL  const char *name)
{
	Item_Context *icp;
	Node *np;

#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"delete_subrt_ctx %s:  calling pop_subrt_ctx",name);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	icp = POP_SUBRT_ID_CTX(name);
	delete_item_context(icp);

	/* We don't want to delete static objects!? BUG */

	/* We have another problem:  the objects are pointed to by
	 * their declaration nodes, and when the objects are deleted the nodes
	 * have dangling pointers.  In order to set these pointers to NULL,
	 * we have to delete the context ourselves - or provide a callback?
	 */
	icp = POP_SUBRT_DOBJ_CTX(name);
	delete_item_context_with_callback(icp, clear_decl_obj);

	np = remTail(SUBRT_CTX_STACK);
	assert( np != NULL );

	givbuf(NODE_DATA(np));	/* string stored w/ savestr */
	rls_node(np);
}

void _pop_subrt_cpair(QSP_ARG_DECL  Context_Pair *cpp,const char *name)
{
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
sprintf(ERROR_STRING,"pop_subrt_cpair %s:  calling pop_subrt_ctx",name);
advise(ERROR_STRING);
}
#endif /* QUIP_DEBUG */
	SET_CP_ID_CTX(cpp, POP_SUBRT_ID_CTX(name));
	SET_CP_OBJ_CTX(cpp,POP_SUBRT_DOBJ_CTX(name));
}

// Get the name of the current context for a given item_type

#define get_subrt_ctx_name(name,itp) _get_subrt_ctx_name(QSP_ARG  name,itp)

static const char *_get_subrt_ctx_name(QSP_ARG_DECL  const char *name,Item_Type *itp)
{
	/* having this static makes it not thread-safe!? BUG */
	static char ctxname[LLEN];
	Node *np;

	assert( SUBRT_CTX_STACK != NULL );

	assert( QLIST_TAIL(SUBRT_CTX_STACK) != NULL );

	np = QLIST_TAIL(SUBRT_CTX_STACK);

	assert( ! strcmp(name,(char *)NODE_DATA(np)) );

	/* BUG possible string overflow */
	sprintf(ctxname,"%s.%s",IT_NAME(itp), name_for_ctx_stack(SINGLE_QSP_ARG) );

	return( ctxname );
}

/* Pop the context, but don't delete the items */

Item_Context * pop_subrt_ctx(QSP_ARG_DECL  const char *name,Item_Type *itp)
{
	const char *ctxname;
	Item_Context *icp;

	ctxname = get_subrt_ctx_name(name,itp);

//sprintf(ERROR_STRING,"Searching for context %s",ctxname);
//advise(ERROR_STRING);
	icp = ctx_of(ctxname);
	assert( icp != NULL );
	assert( icp == current_context(itp) );

//fprintf(stderr,"popping %s from %s\n",CTX_NAME(icp),ITEM_TYPE_NAME(itp));
	pop_item_context(itp);

	return(icp);
}


static Vec_Expr_Node *find_numbered_node_in_tree(Vec_Expr_Node *root_enp,int n)
{
	Vec_Expr_Node *enp;
	int i;

	if( root_enp==NULL ) return(NULL);

	if( VN_SERIAL(root_enp) == n ) return(root_enp);

	if( VN_CODE(root_enp) == T_SCRIPT ) return(NULL);

	for(i=0;i<MAX_CHILDREN(root_enp);i++){
		if( VN_CHILD(root_enp,i) != NULL ){
			enp = find_numbered_node_in_tree(VN_CHILD(root_enp,i),n);
			if( enp!= NULL ) return(enp);
		}
	}
	return(NULL);
}

static Vec_Expr_Node *find_numbered_node_in_subrt(Subrt *srp,int n)
{
	Vec_Expr_Node *enp;

	if( ! IS_SCRIPT(srp) ){
		enp=find_numbered_node_in_tree(SR_BODY(srp),n);
		if( enp != NULL ) return enp;
	}

	enp=find_numbered_node_in_tree(SR_ARG_DECLS(srp),n);
	return enp;
}

Vec_Expr_Node *_find_node_by_number(QSP_ARG_DECL  int n)
{
	List *lp;
	Subrt *srp;
	Node *np;

	if( subrt_itp == NULL ) return NULL;

	lp=subrt_list();
	if( lp == NULL )
		return NULL;

	np=QLIST_HEAD(lp);
	while(np!=NULL){
		Vec_Expr_Node *enp;

		srp=(Subrt *)NODE_DATA(np);
		enp=find_numbered_node_in_subrt(srp,n);
		if( enp!=NULL ) return(enp);
		np=NODE_NEXT(np);
	}
	return(NULL);
}

