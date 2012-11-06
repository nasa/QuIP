#include "quip_config.h"

char VersionId_interpreter_vars[] = QUIP_VERSION_STRING;

#include <stdio.h>
#ifdef HAVE_CTYPE_H
#include <ctype.h>
#endif
#ifdef HAVE_STRING_H
#include <string.h>	/* strstr() */
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>	/* getcwd() */
#endif

#include "items.h"
#include "void.h"
#include "savestr.h"
#include "query.h"
#include "macros.h"

/* local prototypes */

static void update_working_dir(Var *vp);
static void update_uid(Var *vp);

ITEM_INTERFACE_DECLARATIONS(Var,var_)

#ifdef THREAD_SAFE_QUERY

#define LOCK_VAR(vp)						\
								\
	if( n_active_threads > 1 )				\
	{							\
		int status;					\
								\
		status = pthread_mutex_lock(&vp->v_mutex);	\
		if( status != 0 )				\
			report_mutex_error(DEFAULT_QSP_ARG  status,"LOCK_VAR");	\
		vp->v_flags |= VAR_LOCKED;			\
	}



#define UNLOCK_VAR(vp)						\
								\
	if( VAR_IS_LOCKED(vp) )					\
	{							\
		int status;					\
								\
		vp->v_flags &= ~VAR_LOCKED;			\
		status = pthread_mutex_unlock(&vp->v_mutex);	\
		if( status != 0 )				\
			report_mutex_error(DEFAULT_QSP_ARG  status,"UNLOCK_VAR");\
	}

#else /* ! THREAD_SAFE_QUERY */

#define LOCK_VAR(vp)
#define UNLOCK_VAR(vp)

#endif /* ! THREAD_SAFE_QUERY */


void var_stats(SINGLE_QSP_ARG_DECL)
{
	if( var__itp == NO_ITEM_TYPE ) var__init(SINGLE_QSP_ARG);
	item_stats(QSP_ARG  var__itp);
}

void restrict_var_context(QSP_ARG_DECL int flag)
{
	RESTRICT_ITEM_CONTEXT(var__itp,flag)
}

Var *assign_var(QSP_ARG_DECL  const char *name,const char *value)
{
	Var *vp;

#ifdef CAUTIOUS
	if( name == NULL ){
		sprintf(ERROR_STRING,"CAUTIOUS:  assign_var passed NULL name");
		return(NO_VAR);
	}
	if( *name == 0 ){
		sprintf(ERROR_STRING,"CAUTIOUS:  assign_var passed empty string");
		return(NO_VAR);
	}
#endif /* CAUTIOUS */

	// BUG? - var__of returns a variable from any level of the context stack...
	// But for setting, we usually want to set in the context on the top of the stack.
	// So we should only check the top context here...
	// This change should not affect any old scripts, as we have not used
	// variable contexts before multi-threading...
	// BUT sometimes we might want to set a global var from a thread?
	// We probably need another function
	//
	// Inspection of items.c revealed that there was
	// already a context restriction flag (global).
	// That was moved into the Item_Type struct,
	// we left all else alone so as not to break
	// vectree, but it might be better to have control
	// of it rather than doing this automatically...

	vp=var__of(QSP_ARG  name);

	if( vp==NO_VAR ){
		vp=new_var_(QSP_ARG  name);
		vp->v_func = NULL;
		vp->v_value = NULL;
#ifdef THREAD_SAFE_QUERY
		vp->v_flags = 0;
		pthread_mutex_init(&vp->v_mutex,NULL);

#endif /* THREAD_SAFE_QUERY */
	}

	LOCK_VAR(vp)

	if( vp->v_value != NULL ){
		if( vp->v_func != NULL ){
			sprintf(ERROR_STRING,
			"Variable name \"%s\" is reserved, can't reassign value",name);
			WARN(ERROR_STRING);
			UNLOCK_VAR(vp)
			return(NO_VAR);
		}

#ifdef DEBUG
if( debug & qldebug ){
sprintf(ERROR_STRING,"%s - %s:  freeing old value of $%s at 0x%lx",
WHENCE(assign_var),vp->v_name,(int_for_addr)vp->v_value);
advise(ERROR_STRING);
}
#endif	/* DEBUG */

		rls_str(vp->v_value);
	}

	vp->v_value = savestr(value);

	UNLOCK_VAR(vp)

	return(vp);
}

void freevar(QSP_ARG_DECL  Var *vp)
{
	del_var_(QSP_ARG  vp->v_name);
	rls_str((char *)vp->v_name);
	rls_str(vp->v_value);
}

const char *var_value(QSP_ARG_DECL  const char *name)
{
	Var *vp;

	vp=var_of(QSP_ARG  name);
	if( vp==NO_VAR ){
		sprintf(ERROR_STRING,"undefined variable \"%s\"",name);
		WARN(ERROR_STRING);
		return(NULL);
	} else {
		if( vp->v_func != NULL )	/* update value of reserved var */
			(*vp->v_func)(vp);
		return(vp->v_value);
	}
}

void list_var_contexts(SINGLE_QSP_ARG_DECL)
{
	//list_items(QSP_ARG  var__itp->it_context_itp);
	List *lp;
	Node *np;

	lp = item_list(QSP_ARG  var__itp->it_context_itp);
	if( lp == NO_LIST ) return;

	np=lp->l_head;
	while(np!=NO_NODE){
		Item_Context *icp;
		icp = np->n_data;
		if( icp->ic_itp == var__itp ){
			sprintf(msg_str,"%s",icp->ic_name);
			prt_msg(msg_str);
		} else {
			if( verbose ){
				sprintf(msg_str,"Context %s is not for variables.",icp->ic_name);
				prt_msg(msg_str);
			}
		}
		np = np->n_next;
	}
}

#define MAKE_VAR_CTX_NAME(dest,suffix)			\
	sprintf(dest,"%s.%s",var__itp->it_name,suffix);

Item_Context *new_var_context(QSP_ARG_DECL  const char *name)
{
	Item_Context *icp;

	icp = create_item_context(QSP_ARG  var__itp, name);

	return(icp);
#ifdef FOOBAR
	Item *ip;
	Item_Context *icp;
	char fullname[LLEN];

	MAKE_VAR_CTX_NAME(fullname,name)

	ip = item_of(QSP_ARG  var__itp->it_context_itp, fullname);

	if( ip != NO_ITEM ){
		sprintf(ERROR_STRING,"new_var_context:  Variable context \"%s\" already exists!?",fullname);
		WARN(ERROR_STRING);
		return(NULL);
	}

	ip = new_item(QSP_ARG  var__itp->it_context_itp,fullname,sizeof(Item_Context));
	if( ip == NO_ITEM ){
		sprintf(ERROR_STRING,"new_var_context:  error creating \"%s\"!?",fullname);
		WARN(ERROR_STRING);
		return(NULL);
	}
	icp = (Item_Context *)ip;
	// init context here...
	icp->ic_itp = var__itp;
	icp->ic_flags = 0;
	icp->ic_nsp = NULL;	// BUG? - who creates the namespace???
	return(icp);
#endif /* FOOBAR */
}

void push_var_ctx(QSP_ARG_DECL const char *s)
{
	char f[LLEN];
	Item_Context *icp;

	MAKE_VAR_CTX_NAME(f,s)

	icp = (Item_Context *)get_item(QSP_ARG  var__itp->it_context_itp,f);
	if( icp == NO_ITEM_CONTEXT ) return;

	PUSH_ITEM_CONTEXT(var__itp,icp);
}

void pop_var_ctx(SINGLE_QSP_ARG_DECL)
{
	// Does popping the context release the objects?  I think not...
	POP_ITEM_CONTEXT(var__itp);
}

void show_var_ctx_stk(SINGLE_QSP_ARG_DECL)
{
	List *lp;
	Node *np;

	lp = CONTEXT_LIST(var__itp);

#ifdef CAUTIOUS
	if( lp == NO_LIST )
		ERROR1("CAUTIOUS:  do_show_var_ctx_stk:  Context list is missing!?");
#endif /* CAUTIOUS */

	np=lp->l_head;
	while(np!=NO_NODE){
		Item_Context *icp;
		icp=np->n_data;
		sprintf(ERROR_STRING,"%s",icp->ic_name);
		prt_msg(ERROR_STRING);
		np=np->n_next;
	}
}

void find_vars(QSP_ARG_DECL  const char *s)
{
	List *lp;

	lp=find_items(QSP_ARG  var__itp,s);
	if( lp==NO_LIST ) return;
	print_list_of_items(QSP_ARG  lp);
}

void search_vars(QSP_ARG_DECL  const char *frag)
{
	List *lp;
	Node *np;
	Var *vp;
	char lc_frag[LLEN];

	lp=item_list(QSP_ARG  var__itp);
	if( lp == NO_LIST ) return;

	np=lp->l_head;
	decap(lc_frag,frag);
	while(np!=NO_NODE){
		char str1[LLEN];
		vp = (Var*) np->n_data;
		/* make the match case insensitive */
		decap(str1,vp->v_value);
		if( strstr(str1,lc_frag) != NULL ){
			sprintf(msg_str,"%s:\t%s",vp->v_name,vp->v_value);
			prt_msg(msg_str);
		}
		np=np->n_next;
	}
}

double dvarexists(QSP_ARG_DECL  const char *varname)
{
	Var *vp;

	vp = var_of( /* CURR_QSP_ARG */  QSP_ARG   varname);
	if( vp == NO_VAR ) return(0.0);
	else return(1.0);
}


/* Reserved variables
 *
 * Inspired by the c-shell's $cwd variable...
 *
 * Reserved variables are implemented by adding a function field to the variable
 * structure (defaults to NULL for non-reserved [normal] variables).
 * When a value is fetched, using var_value(), if the function is non-NULL,
 * then it is called prior to returning the value.
 */

static void update_working_dir(Var *vp)
{
	char dirname[LLEN];

	if( getcwd(dirname,LLEN) == NULL ){
		tell_sys_error("getcwd");
		NWARN("error getting current working directory");
		return;
	}

	rls_str(vp->v_value);
	vp->v_value = savestr(dirname);
}

static void update_uid(Var *vp)
{
	int uid;
	char str[16];

	uid=getuid();
	sprintf(str,"%d",uid);

	/* BUG?  This is inefficient, because the user id
	 * will hardly ever change during a program...
	 * But probably we'll hardly ever use this,
	 * so who cares if it's not maximally efficient?
	 */

	rls_str(vp->v_value);
	vp->v_value = savestr(str);
}

static void update_pid(Var *vp)
{
	int pid;
	char str[16];

	pid=getpid();
	sprintf(str,"%d",pid);

	/* This makes more sense for pid's than for uid's,
	 * since we may have the option of forking a child process...
	 */

	rls_str(vp->v_value);
	vp->v_value = savestr(str);
}

void init_reserved_vars(SINGLE_QSP_ARG_DECL)
{
	Var *vp;

	vp = ASSIGN_VAR("cwd",".");
	vp->v_func = update_working_dir;

	vp = ASSIGN_VAR("uid","0");
	vp->v_func = update_uid;

	vp = ASSIGN_VAR("pid","0");
	vp->v_func = update_pid;
}

