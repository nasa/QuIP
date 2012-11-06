
#include "quip_config.h"


char VersionId_newvec_vec_chn[] = QUIP_VERSION_STRING;

#include <stdio.h>
#include <string.h>	/* memcpy */
#include "nvf.h"
#include "items.h"
#include "new_chains.h"



/* local prototypes */
static Chain *new_chain(QSP_ARG_DECL  const char *name);
static void _del_chain(QSP_ARG_DECL  Chain *cp);

/* BUG these variables should be per-qsp */
int is_chaining=0;
static Chain *curr_cp=NULL;

ITEM_INTERFACE_DECLARATIONS(Chain,vec_chain)

static Chain *new_chain(QSP_ARG_DECL  const char *name)
{
	Chain *cp;

	cp=new_vec_chain(QSP_ARG  name);

	if( cp==NO_CHAIN ) return(NO_CHAIN);

	cp->ch_lp = new_list();
	return(cp);
}

static void _del_chain(QSP_ARG_DECL  Chain *cp)
{
	Node *np;

	/* CAUTIOUS lp should never be null */
#ifdef CAUTIOUS
	if( cp->ch_lp == NO_LIST ) ERROR1("CAUTIOUS:  _del_chain:  null list!?");
#endif /* CAUTIOUS */
	while( (np=remTail(cp->ch_lp)) != NO_NODE ){
		Vec_Chn_Blk *vcb_p;
		/* BUG could release to pool */
		vcb_p=np->n_data;
		if( vcb_p->vcb_args.va_spi_p != NULL )
			givbuf(vcb_p->vcb_args.va_spi_p);
		if( vcb_p->vcb_args.va_szi_p != NULL )
			givbuf(vcb_p->vcb_args.va_szi_p);
		givbuf(vcb_p);
		rls_node(np);
	}
	rls_list(cp->ch_lp);

	del_vec_chain(QSP_ARG  cp->ch_name);
	rls_str((char *)cp->ch_name);
}

void exec_chain(Chain *cp)
{
	Vec_Chn_Blk *vcb_p;
	Node *np;

#ifdef CAUTIOUS
	if( cp->ch_lp == NO_LIST ){
		sprintf(DEFAULT_ERROR_STRING,"CAUTIOUS:  exec_chain:  chain %s is empty!?",cp->ch_name);
		NWARN(DEFAULT_ERROR_STRING);
		return;
	}
#endif /* CAUTIOUS */

	np = cp->ch_lp->l_head;
	while( np != NO_NODE ){
		vcb_p = (Vec_Chn_Blk *)np->n_data;
#ifdef CAUTIOUS
		if( vcb_p->vcb_func == NULL ) NERROR1("CAUTIOUS:  exec_chain:  chain block has null function ptr!?");
#endif /* CAUTIOUS */
		(*vcb_p->vcb_func)(&vcb_p->vcb_args);
		np=np->n_next;
	}
}

void chain_info(Chain *cp)
{
	sprintf(msg_str,"Chain %s:  %d blocks",cp->ch_name,eltcount(cp->ch_lp));
	prt_msg(msg_str);
}

void start_chain(QSP_ARG_DECL  const char *name)
{
	Chain *cp;

	if( is_chaining ){
		NWARN("a chain buffer is already open");
		return;
	}
	cp=new_chain(QSP_ARG  name);
	curr_cp = cp;
	is_chaining=1;
}

void add_link(void (*func)(Vector_Args *), Vector_Args *vap)
{
	Vec_Chn_Blk *vcb_p;
	Node *np;

#ifdef CAUTIOUS
	if( ! is_chaining ){
		NWARN("CAUTIOUS:  add_link:  need to start a chain before adding links");
		return;
	}
	if( curr_cp == NULL ){
		NERROR1("CAUTIOUS:  is_chaining is true, but curr_cp is NULL!?");
	}
#endif /* CAUTIOUS */

	vcb_p=getbuf(sizeof(Vec_Chn_Blk));
	vcb_p->vcb_func = func;
	vcb_p->vcb_args = (*vap);
	if( vap->va_spi_p != NULL ){
		vcb_p->vcb_args.va_spi_p = getbuf(sizeof(Spacing_Info));
		*vcb_p->vcb_args.va_spi_p = *vap->va_spi_p;
	}
	if( vap->va_szi_p != NULL ){
		vcb_p->vcb_args.va_szi_p = getbuf(sizeof(Size_Info));
		*vcb_p->vcb_args.va_szi_p = *vap->va_szi_p;
	}

	np = mk_node(vcb_p);
	addTail(curr_cp->ch_lp,np);
}

void end_chain(void)
{
	if( !is_chaining ){
		NWARN("no chain buffer currently open");
		return;
	}
	is_chaining=0;
}


int chain_breaks(const char *routine_name)
{
	if( is_chaining ){
		sprintf(DEFAULT_ERROR_STRING,
	"Routine \"%s\" is not chainable!?",routine_name);
		NWARN(DEFAULT_ERROR_STRING);
		return(1);
	}
	return(0);
}

/* If we remove an object whose data is used in a chain block, there will be
 * a dangling pointer if we remove the object and then try to use the chain
 * block.  To prevent this, we require that all objects used in chain blocks
 * be static.
 */

#define SCHECK(dp)				\
						\
	if( dp != NO_OBJ ){			\
		if( ! IS_STATIC(dp) ){		\
			sprintf(DEFAULT_ERROR_STRING,	\
"Object %s must be static for use in chain %s.",\
		dp->dt_name,curr_cp->ch_name);	\
			NWARN(DEFAULT_ERROR_STRING);	\
			return(-1);		\
		}				\
	}

int insure_static(Vec_Obj_Args *oap)
{
	int i;

	SCHECK( oap->oa_dest )
	for(i=0;i<MAX_N_ARGS;i++){
		SCHECK( oap->oa_dp[i] )
	}
	for(i=0;i<MAX_RETSCAL_ARGS;i++){
		SCHECK( oap->oa_sdp[i] )
	}
	return(0);
}


