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

void update_subrt(QSP_ARG_DECL  Subrt *srp, Vec_Expr_Node *body )
{
	if( SR_BODY(srp) != NULL ){
		NODE_ERROR(body);
		NWARN("subroutine body is not null!?");
	}

	SET_SR_BODY(srp, body);
}

Subrt * remember_subrt(QSP_ARG_DECL  Precision * prec_p,const char *name,Vec_Expr_Node *args,Vec_Expr_Node *body)
{
	Subrt *srp;

	srp=new_subrt(QSP_ARG  name);
	if( srp==NULL ) return(NULL);

	SET_SR_ARG_DECLS(srp, args);
	SET_SR_BODY(srp, body);
	SET_SR_RET_LIST(srp, NULL);
	SET_SR_CALL_LIST(srp, NULL);
	SET_SR_PREC_PTR(srp, prec_p);
	/* We used to give void subrt's a null shape ptr... */
	SET_SR_SHAPE(srp, ALLOC_SHAPE );
	COPY_SHAPE(SR_SHAPE(srp), uk_shape(PREC_CODE(prec_p)));
	if( PREC_CODE(prec_p) == PREC_VOID ){
		CLEAR_SHP_FLAG_BITS(SR_SHAPE(srp), DT_UNKNOWN_SHAPE);
	}
	SET_SR_FLAGS(srp, 0);
	SET_SR_CALL_VN(srp, NULL);
	return(srp);
}

COMMAND_FUNC( do_run_subrt )
{
	Subrt *srp;

	srp=PICK_SUBRT("");

	if( srp==NULL ) return;

	SET_SR_ARG_VALS(srp,NULL);

	RUN_SUBRT_IMMED(srp,NULL,NULL);
}

COMMAND_FUNC( do_dump_subrt )
{
	Subrt *srp;

	srp=PICK_SUBRT("");

	if( srp==NULL ) return;
	DUMP_SUBRT(srp);
}

void dump_subrt(QSP_ARG_DECL Subrt *srp)
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
		DUMP_TREE(SR_ARG_DECLS(srp));
	}

	if( SR_BODY(srp) != NULL ){
		sprintf(msg_str,"Subrt %s body:\n",SR_NAME(srp));
		prt_msg(msg_str);
		print_dump_legend(SINGLE_QSP_ARG);
		DUMP_TREE(SR_BODY(srp));
	}
}

COMMAND_FUNC( do_opt_subrt )
{
	Subrt *srp;

	srp=PICK_SUBRT("");

	if( srp==NULL ) return;

	OPTIMIZE_SUBRT(srp);
}

COMMAND_FUNC( do_tell_cost )
{
	Subrt *srp;

	srp=PICK_SUBRT("");

	if( srp==NULL ) return;

	tell_cost(QSP_ARG  srp);
}

COMMAND_FUNC( do_subrt_info )
{
	Subrt *srp;
	Vec_Expr_Node *enp;
	Node *np;

	srp=PICK_SUBRT("");

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
		DESCRIBE_SHAPE(VN_SHAPE(enp));
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
			DUMP_TREE(ret_enp);
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
			DUMP_TREE(c_enp);
			np=NODE_NEXT(np);
		}
	}

	/*
	sprintf(msg_str,"\t%ld flops, %ld math calls",
		enp->en_flops,enp->en_nmath);
	prt_msg(msg_str);

	*/
}

Subrt *create_script_subrt(QSP_ARG_DECL  const char *name,int nargs,const char *text)
{
	Subrt *srp;

	srp = remember_subrt(QSP_ARG  PREC_FOR_CODE(PREC_VOID),name,NULL,NULL);
	if( srp == NULL ) return srp;

	SET_SR_FLAG_BITS(srp, SR_SCRIPT);
	SET_SR_N_ARGS(srp, nargs);

	SET_SR_TEXT(srp, savestr(text));
	return srp;
}

static const char *name_from_stack(SINGLE_QSP_ARG_DECL)
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
//sprintf(ERROR_STRING,"name_from_stack:  ctxname = \"%s\"",ctxname);
//advise(ERROR_STRING);
	return(ctxname);
}

/* The original scheme of naming the context Subr.subrtname
 * is no good, because it can't handle multiple instances,
 * as occur with recursion or multi-threading.
 * We might handle recursion by having a current subroutine
 * context (for each thread!)
 */

static const char *get_subrt_id(QSP_ARG_DECL  const char *name)
{
	Node *np;
	const char *s;

//sprintf(ERROR_STRING,"get_subrt_id %s BEGIN",name);
//advise(ERROR_STRING);
	if( SUBRT_CTX_STACK == NULL ){
		SUBRT_CTX_STACK = new_list();
	}
	s=savestr(name);
	np=mk_node((void *)s);
//fprintf(stderr,"Adding context '%s' at 0x%lx to stack\n",s,(long)s);
	addTail(SUBRT_CTX_STACK,np);

	return(name_from_stack(SINGLE_QSP_ARG));
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
 */

void set_subrt_ctx(QSP_ARG_DECL  const char *name)
{
	const char *ctxname;
	Item_Context *icp;	/* data_obj, identifier context */

	ctxname = get_subrt_id(QSP_ARG  name);
//sprintf(ERROR_STRING,"set_subrt_ctx, context name is %s",ctxname);
//advise(ERROR_STRING);
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
/*
sprintf(ERROR_STRING,"set_subrt_ctx:  pushing context %s for ojects and identifiers",
ctxname);
advise(ERROR_STRING);
*/
}
#endif /* QUIP_DEBUG */

//fprintf(stderr,"set_subrt_ctx, name = %s at 0x%lx\n",ctxname,(long)ctxname);
	icp=create_id_context(QSP_ARG  ctxname);
	PUSH_ID_CONTEXT(icp);

	icp=create_dobj_context(QSP_ARG  ctxname);
	PUSH_DOBJ_CONTEXT(icp);
}

static void rls_reference(Reference *refp)
{
	rls_stringbuf( REF_SBUF(refp) );
	givbuf(refp);
}

// We may not need to do any of this on objC ???
//
// really a memory release function...

void delete_id(QSP_ARG_DECL  Item *ip)
{
	Identifier *idp;

	idp = (Identifier *)ip;

	switch(ID_TYPE(idp)){
		case ID_REFERENCE:
			/* We used to call delvec(ID_REF(idp)->ref_dp)... */
			/* We don't do this, because this case occurs in an export/unexport cycle */
			givbuf(ID_REF(idp));
			break;
		case ID_STRING:
			rls_reference(ID_REF(idp));
			break;
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
			NWARN(ERROR_STRING);
			break;
	}
	del_id(QSP_ARG  idp );	// releases name for us
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

void delete_subrt_ctx(QSP_ARG_DECL  const char *name)
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
	delete_item_context(QSP_ARG  icp);

	/* We don't want to delete static objects!? BUG */

	/* We have another problem:  the objects are pointed to by
	 * their declaration nodes, and when the objects are deleted the nodes
	 * have dangling pointers.  In order to set these pointers to NULL,
	 * we have to delete the context ourselves - or provide a callback?
	 */
	icp = POP_SUBRT_DOBJ_CTX(name);
	delete_item_context_with_callback(QSP_ARG  icp, clear_decl_obj);

	np = remTail(SUBRT_CTX_STACK);
	assert( np != NULL );

	givbuf(NODE_DATA(np));	/* string stored w/ savestr */
	rls_node(np);
}

void pop_subrt_cpair(QSP_ARG_DECL  Context_Pair *cpp,const char *name)
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

static const char *get_subrt_ctx_name(QSP_ARG_DECL  const char *name,Item_Type *itp)
{
	/* having this static makes it not thread-safe!? BUG */
	static char ctxname[LLEN];
	Node *np;

	assert( SUBRT_CTX_STACK != NULL );
	assert( QLIST_TAIL(SUBRT_CTX_STACK) != NULL );

	np = QLIST_TAIL(SUBRT_CTX_STACK);

	assert( ! strcmp(name,(char *)NODE_DATA(np)) );

	/* BUG possible string overflow */
	sprintf(ctxname,"%s.%s",IT_NAME(itp), name_from_stack(SINGLE_QSP_ARG) );

	return( ctxname );
}

/* Pop the context, but don't delete the items */

Item_Context * pop_subrt_ctx(QSP_ARG_DECL  const char *name,Item_Type *itp)
{
	const char *ctxname;
	Item_Context *icp;

	ctxname = get_subrt_ctx_name(QSP_ARG  name,itp);
#ifdef QUIP_DEBUG
if( debug & scope_debug ){
/*
sprintf(ERROR_STRING,"pop_subrt_ctx:  popping subroutine context %s for %s items",ctxname,IT_NAME(itp));
advise(ERROR_STRING);
*/
}
#endif /* QUIP_DEBUG */

//sprintf(ERROR_STRING,"Searching for context %s",ctxname);
//advise(ERROR_STRING);
	icp = ctx_of(QSP_ARG  ctxname);
	assert( icp != NULL );
	assert( icp == CURRENT_CONTEXT(itp) );

	pop_item_context(QSP_ARG  itp);

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

Vec_Expr_Node *find_node_by_number(QSP_ARG_DECL  int n)
{
	List *lp;
	Subrt *srp;
	Node *np;

	if( subrt_itp == NULL ) return NULL;

	lp=subrt_list(SINGLE_QSP_ARG);
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


