#include "quip_config.h"

char VersionId_vectree_subrt[] = QUIP_VERSION_STRING;

#ifdef HAVE_STRING_H
#include <string.h>
#endif

#include "query.h"
#include "img_file.h"
#include "debug.h"

#include "vectree.h"

List *global_uk_lp=NO_LIST;
char *subroutine_context_name=NULL;
List *subroutine_context_stack=NO_LIST;

/* local prototypes */

static const char *get_subrt_id(const char *name);
static Vec_Expr_Node *find_numbered_node_in_subrt(Subrt *srp,int n);
static Vec_Expr_Node *find_numbered_node_in_tree(Vec_Expr_Node *root_enp,int n);
static const char *get_subrt_ctx_name(const char *name,Item_Type *itp);
static const char *name_from_stack(void);

ITEM_INTERFACE_DECLARATIONS(Identifier,id)
ITEM_INTERFACE_DECLARATIONS(Subrt,subrt)
ITEM_INTERFACE_DECLARATIONS(Undef_Sym,undef)

/* We call this after we scan the body of a subroutine for which
 * we already had a prototype.
 */

void update_subrt(QSP_ARG_DECL  Subrt *srp, Vec_Expr_Node *body )
{
	if( srp->sr_body != NO_VEXPR_NODE ){
		NODE_ERROR(body);
		NWARN("subroutine body is not null!?");
	}

	srp->sr_body = body;
}

Subrt * remember_subrt(QSP_ARG_DECL  prec_t prec,const char *name,Vec_Expr_Node *args,Vec_Expr_Node *body)
{
	Subrt *srp;

	srp=new_subrt(QSP_ARG  name);
	if( srp==NO_SUBRT ) return(NO_SUBRT);

	srp->sr_arg_decls = args;
	srp->sr_body = body;
	srp->sr_ret_lp = NO_LIST;
	srp->sr_call_lp = NO_LIST;
	srp->sr_prec = prec;
	/* We used to give void subrt's a null shape ptr... */
	srp->sr_shpp = (Shape_Info *)getbuf(sizeof(Shape_Info));
	*srp->sr_shpp = * uk_shape(prec);
	if( prec == PREC_VOID ){
		srp->sr_shpp->si_flags &= ~DT_UNKNOWN_SHAPE;
	}
	srp->sr_flags = 0;
	srp->sr_call_enp = NO_VEXPR_NODE;
	return(srp);
}

COMMAND_FUNC( do_run_subrt )
{
	Subrt *srp;

	srp=PICK_SUBRT("");

	if( srp==NO_SUBRT ) return;

	srp->sr_arg_vals=NO_VEXPR_NODE;

	RUN_SUBRT(srp,NO_VEXPR_NODE,NO_OBJ);
}

COMMAND_FUNC( do_dump_subrt )
{
	Subrt *srp;

	srp=PICK_SUBRT("");

	if( srp==NO_SUBRT ) return;
	DUMP_SUBRT(srp);
}

void dump_subrt(QSP_ARG_DECL Subrt *srp)
{
	if( IS_SCRIPT(srp) ){
		sprintf(msg_str,"Script subrt %s:",srp->sr_name);
		prt_msg(msg_str);
		prt_msg((char *)srp->sr_body);
		return;
	}

	if( srp->sr_arg_decls != NO_VEXPR_NODE ){
		sprintf(msg_str,"Subrt %s arg declarations:\n",srp->sr_name);
		prt_msg(msg_str);
		print_dump_legend();
		DUMP_TREE(srp->sr_arg_decls);
	}

	if( srp->sr_body != NO_VEXPR_NODE ){
		sprintf(msg_str,"Subrt %s body:\n",srp->sr_name);
		prt_msg(msg_str);
		print_dump_legend();
		DUMP_TREE(srp->sr_body);
	}
}

COMMAND_FUNC( do_opt_subrt )
{
	Subrt *srp;

	srp=PICK_SUBRT("");

	if( srp==NO_SUBRT ) return;

	OPTIMIZE_SUBRT(srp);
}

COMMAND_FUNC( do_tell_cost )
{
	Subrt *srp;

	srp=PICK_SUBRT("");

	if( srp==NO_SUBRT ) return;

	tell_cost(QSP_ARG  srp);
}

COMMAND_FUNC( do_subrt_info )
{
	Subrt *srp;
	Vec_Expr_Node *enp;
	Node *np;

	srp=PICK_SUBRT("");

	if( srp==NO_SUBRT ) return;

	if( IS_SCRIPT(srp) ){
		sprintf(msg_str,"\nScript subroutine %s, %d arguments:\n",srp->sr_name,srp->sr_nargs);
		prt_msg(msg_str);

		prt_msg((char *)srp->sr_body);
		return;
	}

	sprintf(msg_str,"\nSubroutine %s:",srp->sr_name);
	prt_msg(msg_str);

	enp = srp->sr_body;

	if( enp->en_shpp != NO_SHAPE ){
		describe_shape(enp->en_shpp);
	} else prt_msg("shape not determinable");

	sprintf(msg_str,"\t%d arguments",srp->sr_nargs);
	prt_msg(msg_str);

	if( srp->sr_ret_lp != NO_LIST ){
		sprintf(msg_str,"%d unknown return shape nodes:",eltcount(srp->sr_ret_lp));
		prt_msg(msg_str);
		np=srp->sr_ret_lp->l_head;
		while(np!=NO_NODE){
			Vec_Expr_Node *enp;
			enp = (Vec_Expr_Node *)np->n_data;
			sprintf(msg_str,"\tn%d:",enp->en_serial);
			prt_msg(msg_str);
			DUMP_TREE(enp);
			np=np->n_next;
		}
	}
	if( srp->sr_call_lp != NO_LIST ){
		sprintf(msg_str,"%d unknown callfunc shape nodes:",eltcount(srp->sr_call_lp));
		prt_msg(msg_str);
		np=srp->sr_call_lp->l_head;
		while(np!=NO_NODE){
			Vec_Expr_Node *enp;
			enp = (Vec_Expr_Node *)np->n_data;
			sprintf(msg_str,"\tn%d:",enp->en_serial);
			prt_msg(msg_str);
			DUMP_TREE(enp);
			np=np->n_next;
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

	srp = remember_subrt(QSP_ARG  PREC_VOID,name,NO_VEXPR_NODE,NO_VEXPR_NODE);
	if( srp == NO_SUBRT ) return(srp);

	srp->sr_flags |= SR_SCRIPT;
	srp->sr_nargs = nargs;

	srp->sr_body = (Vec_Expr_Node *) savestr(text);
	return(srp);
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

	ctxname = get_subrt_id(name);
//sprintf(error_string,"set_subrt_ctx, context name is %s",ctxname);
//advise(error_string);
#ifdef DEBUG
if( debug & scope_debug ){
/*
sprintf(error_string,"set_subrt_ctx:  pushing context %s for ojects and identifiers",
ctxname);
advise(error_string);
*/
}
#endif /* DEBUG */

	icp=create_item_context(QSP_ARG  id_itp,ctxname);
	PUSH_ITEM_CONTEXT(id_itp,icp);

	icp=create_item_context(QSP_ARG  dobj_itp,ctxname);
	PUSH_ITEM_CONTEXT(dobj_itp,icp);
}

void delete_id(TMP_QSP_ARG_DECL  Item *ip)
{
	Identifier *idp;

	idp = (Identifier *)ip;

	switch(idp->id_type){
		case ID_REFERENCE:
			/* We used to call delvec(idp->id_refp->ref_dp)... */
			/* We don't do this, because this case occurs in an export/unexport cycle */
			givbuf(idp->id_refp);
			break;
		case ID_STRING:
			if( idp->id_refp->ref_sbp->sb_buf != NULL )
				givbuf(idp->id_refp->ref_sbp->sb_buf);
			givbuf(idp->id_refp->ref_sbp);
			givbuf(idp->id_refp);
			break;
		case ID_POINTER:
			givbuf(idp->id_ptrp);
			break;
		case ID_FUNCPTR:
			givbuf(idp->id_fpp);
			break;
		case ID_LABEL:
			break;

		default:
			sprintf(error_string,"delete_id:  unhandled id type %d",idp->id_type);
			NWARN(error_string);
			break;
	}
	del_item(QSP_ARG  id_itp, idp );
	rls_str((char *)idp->id_name);
}


/* When we exit a subroutine, we call this to automatically destroy
 * all of the objects it created.
 */

void delete_subrt_ctx(QSP_ARG_DECL  const char *name)
{
	Item_Context *icp;
	Node *np;

#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"delete_subrt_ctx %s:  calling pop_subrt_ctx",name);
advise(error_string);
}
#endif /* DEBUG */
	icp = POP_SUBRT_CTX(name,id_itp);
	delete_item_context(QSP_ARG  icp);

	/* We don't want to delete static objects!? BUG */

	icp = POP_SUBRT_CTX(name,dobj_itp);
	delete_item_context(QSP_ARG  icp);

	np = remTail(subroutine_context_stack);
#ifdef CAUTIOUS
	if( np == NO_NODE ){
		sprintf(error_string,"CAUTIOUS:  delete_subrt_ctx:  can't pop name from context stack!?");
		WARN(error_string);
		return;
	}
#endif /* CAUTIOUS */
	givbuf(np->n_data);	/* string stored w/ savestr */
	rls_node(np);
}

void pop_subrt_cpair(QSP_ARG_DECL  Context_Pair *cpp,const char *name)
{
#ifdef DEBUG
if( debug & scope_debug ){
sprintf(error_string,"pop_subrt_cpair %s:  calling pop_subrt_ctx",name);
advise(error_string);
}
#endif /* DEBUG */
	cpp->cp_id_icp = POP_SUBRT_CTX(name,id_itp);
	cpp->cp_dobj_icp = POP_SUBRT_CTX(name,dobj_itp);
}

/* Pop the context, but don't delete the items */

Item_Context * pop_subrt_ctx(QSP_ARG_DECL  const char *name,Item_Type *itp)
{
	const char *ctxname;
	Item_Context *icp;

	ctxname = get_subrt_ctx_name(name,itp);
#ifdef DEBUG
if( debug & scope_debug ){
/*
sprintf(error_string,"pop_subrt_ctx:  popping subroutine context %s for %s items",ctxname,itp->it_name);
advise(error_string);
*/
}
#endif /* DEBUG */
//sprintf(error_string,"Searching for context %s",ctxname);
//advise(error_string);
	icp = ctx_of(QSP_ARG  ctxname);
#ifdef CAUTIOUS
	if( icp == NO_ITEM_CONTEXT ){
		sprintf(error_string,
	"CAUTIOUS:  pop_subrt_ctx couldn't find %s context %s",itp->it_name,ctxname);
		WARN(error_string);
		return(icp);
	}

	if( icp != CURRENT_CONTEXT(itp) ){
		sprintf(error_string,
	"CAUTIOUS:  pop_subrt_ctx:  Subroutine context %s does not match current %s context %s!?",
			icp->ic_name,itp->it_name,CURRENT_CONTEXT(itp)->ic_name);
		WARN(error_string);
		show_context_stack(QSP_ARG  itp);
		return(NO_ITEM_CONTEXT);
	}
#endif /* CAUTIOUS */
	pop_item_context(QSP_ARG  itp);

	return(icp);
}

/* The original scheme of naming the context Subr.subrtname is no good, because
 * it can't handle multiple instances, as occur with recursion or multi-threading.
 * We might handle recursion by having a current subroutine context (for each thread!)
 */

static const char *get_subrt_id(const char *name)
{
	Node *np;
	const char *s;

//sprintf(error_string,"get_subrt_id %s BEGIN",name);
//advise(error_string);
	if( subroutine_context_stack == NO_LIST ){
		subroutine_context_stack = new_list();
	}
	s=savestr(name);
	np=mk_node((void *)s);
	addTail(subroutine_context_stack,np);

	return(name_from_stack());
}

static const char *name_from_stack(void)
{
	static char ctxname[LLEN];
	Node *np;

	ctxname[0]=0;
	np=subroutine_context_stack->l_head;
	while( np != NO_NODE ){
		/* BUG check for string overflow */
		strcat(ctxname,(char *)np->n_data);
		np=np->n_next;
	}
//sprintf(error_string,"name_from_stack:  ctxname = \"%s\"",ctxname);
//advise(error_string);
	return(ctxname);
}

static const char *get_subrt_ctx_name(const char *name,Item_Type *itp)
{
	/* having this static makes it not thread-safe!? BUG */
	static char ctxname[LLEN];
	Node *np;

#ifdef CAUTIOUS
	if( subroutine_context_stack==NO_LIST ){
		NWARN("CAUTIOUS:  get_subrt_ctx_name:  stack is uninitialized!?");
		return("");
	}
	if( subroutine_context_stack->l_tail == NO_NODE ){
		NWARN("CAUTIOUS:  get_subrt_ctx_name:  stack empty!?");
		return("");
	}
#endif /* CAUTIOUS */

	np = subroutine_context_stack->l_tail;

#ifdef CAUTIOUS
	if( strcmp(name,(char *)np->n_data) ){
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  get_subrt_ctx_name:  name \"%s\" does not match top of stack \"%s\"",
			name,(char *)np->n_data);
		NWARN(DEFAULT_ERROR_STRING);
		return("");
	}
#endif /* CAUTIOUS */

	/* BUG possible string overflow */
	sprintf(ctxname,"%s.%s",itp->it_name, name_from_stack() );

	return( ctxname );
}


static Vec_Expr_Node *find_numbered_node_in_tree(Vec_Expr_Node *root_enp,int n)
{
	Vec_Expr_Node *enp;
	int i;

	if( root_enp==NO_VEXPR_NODE ) return(NO_VEXPR_NODE);

	if( root_enp->en_serial == n ) return(root_enp);

	if( root_enp->en_code == T_SCRIPT ) return(NO_VEXPR_NODE);

	for(i=0;i<MAX_CHILDREN(root_enp);i++){
		if( root_enp->en_child[i] != NO_VEXPR_NODE ){
			enp = find_numbered_node_in_tree(root_enp->en_child[i],n);
			if( enp!= NO_VEXPR_NODE ) return(enp);
		}
	}
	return(NO_VEXPR_NODE);
}

static Vec_Expr_Node *find_numbered_node_in_subrt(Subrt *srp,int n)
{
	Vec_Expr_Node *enp;

	if( ! IS_SCRIPT(srp) ){
		enp=find_numbered_node_in_tree(srp->sr_body,n);
		if( enp != NO_VEXPR_NODE ) return(enp);
	}

	enp=find_numbered_node_in_tree(srp->sr_arg_decls,n);
	return(enp);
}

Vec_Expr_Node *find_node_by_number(QSP_ARG_DECL  int n)
{
	List *lp;
	Subrt *srp;
	Node *np;

	if( subrt_itp == NO_ITEM_TYPE ) return(NO_VEXPR_NODE);

	lp=item_list(QSP_ARG  subrt_itp);
	if( lp == NO_LIST )
		return(NO_VEXPR_NODE);

	np=lp->l_head;
	while(np!=NO_NODE){
		Vec_Expr_Node *enp;

		srp=(Subrt *)np->n_data;
		enp=find_numbered_node_in_subrt(srp,n);
		if( enp!=NO_VEXPR_NODE ) return(enp);
		np=np->n_next;
	}
	return(NO_VEXPR_NODE);
}


