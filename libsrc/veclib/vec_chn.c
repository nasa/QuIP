
#include "quip_config.h"

#include <stdio.h>
#include <string.h>	/* memcpy */
#include "quip_prot.h"
#include "veclib/obj_args.h"
#include "nvf.h"
#include "vec_chain.h"
#include "veclib_prot.h"


/* BUG these global variables should be per-qsp - not thread-safe */
int is_chaining=0;
static Chain *curr_cp=NULL;

static Item_Type *vec_chain_itp=NULL;
static ITEM_INIT_FUNC(Chain,vec_chain,0)
ITEM_CHECK_FUNC(Chain,vec_chain)
ITEM_LIST_FUNC(Chain,vec_chain)
ITEM_PICK_FUNC(Chain,vec_chain)

#ifdef NOT_USED

static void _del_chain(QSP_ARG_DECL  Chain *cp)
{
	Node *np;

	assert( CHAIN_LIST(cp) != NULL );
	while( (np=remTail(CHAIN_LIST(cp) )) != NULL ){
		Vec_Chn_Blk *vcb_p;
		/* BUG could release to pool */
		vcb_p=NODE_DATA(np);
		RELEASE_BLOCK(vcb_p);
		//if( vcb_p->vcb_args.va_spi_p != NULL )
			//givbuf(vcb_p->vcb_args.va_spi_p);
		//if( vcb_p->vcb_args.va_szi_p != NULL )
			//givbuf(BLOCK_ARGS(vcb_p->vcb_args.va_szi_p);
		//givbuf(vcb_p);
		rls_node(np);
	}
	rls_list(CHAIN_LIST(cp) );

	del_vec_chain(QSP_ARG  CHAIN_NAME(cp) );
}
#endif /* NOT_USED */

void exec_chain(Chain *cp)
{
	Vec_Chn_Blk *vcb_p;
	Node *np;

	assert( CHAIN_LIST(cp) != NULL );

	np = QLIST_HEAD( CHAIN_LIST(cp) );
	while( np != NULL ){
		vcb_p = (Vec_Chn_Blk *)NODE_DATA(np);
		assert( CHAIN_FUNC(vcb_p) != NULL );

		(* CHAIN_FUNC(vcb_p))( CHAIN_ARGS(vcb_p) );
		np=NODE_NEXT(np);
	}
}

void chain_info(QSP_ARG_DECL  Chain *cp)
{
	sprintf(msg_str,"Chain %s:  %d blocks",CHAIN_NAME(cp),eltcount(CHAIN_LIST(cp) ));
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

void add_link(void (*func)(LINK_FUNC_ARG_DECLS), LINK_FUNC_ARG_DECLS)
{
	Vec_Chn_Blk *vcb_p;
	Node *np;

	assert( is_chaining );
	assert( curr_cp != NULL );

	NEW_CHAIN_BLOCK(vcb_p,vf_code,func,vap);

	//vcb_p=getbuf(sizeof(Vec_Chn_Blk));
	//vcb_p->vcb_func = func;
	//vcb_p->vcb_args = (*vap);
	//if( vap->va_spi_p != NULL ){
		//vcb_p->vcb_args.va_spi_p = getbuf(sizeof(Spacing_Info));
		//*vcb_p->vcb_args.va_spi_p = *vap->va_spi_p;
	//}
	//if( vap->va_szi_p != NULL ){
		//vcb_p->vcb_args.va_szi_p = getbuf(sizeof(Size_Info));
		//*vcb_p->vcb_args.va_szi_p = *vap->va_szi_p;
	//}

	np = mk_node(vcb_p);
	addTail(CHAIN_LIST(curr_cp) ,np);
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
	if( dp != NULL ){			\
		if( ! IS_STATIC(dp) ){		\
			sprintf(DEFAULT_ERROR_STRING,	\
"Object %s must be static for use in chain %s.",\
		OBJ_NAME(dp),CHAIN_NAME(curr_cp) );	\
			NWARN(DEFAULT_ERROR_STRING);	\
			return(-1);		\
		}				\
	}

int insure_static(const Vec_Obj_Args *oap)
{
	int i;

	SCHECK( OA_DEST(oap) )
	for(i=0;i<MAX_N_ARGS;i++){
		SCHECK( OA_SRC_OBJ(oap,i) )
	}
	for(i=0;i<MAX_RETSCAL_ARGS;i++){
		SCHECK( OA_SCLR_OBJ(oap,i) )
	}
	return(0);
}


