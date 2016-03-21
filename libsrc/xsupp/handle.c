
#include "quip_config.h"

/* support for relocatable memory objects
 *
 * Who uses this stuff???
 */

#ifdef HAVE_STDLIB_H
#include <stdlib.h>		/* malloc() */
#endif

#include "quip_prot.h"	// warn
#include "handle.h"

#define NO_HANDLE ((Handle)NULL)


#define MAX_POINTERS	4096
static u_long n_handles=0;


// BUG static globals not thread-safe!?
// However, the node/list routines are thread-safe,
// so it may be OK...  but n_handles may be problematic
// without locking!

static pointer ptr_tbl[MAX_POINTERS];
static List *free_hdl_list=NO_LIST;
static List *used_hdl_list=NO_LIST;

static void hdl_init()
{
	int i;
	Node *np;

	free_hdl_list=new_list();
	used_hdl_list=new_list();
	for(i=0;i<MAX_POINTERS;i++){
		ptr_tbl[i]=NULL;
		np = mk_node(&ptr_tbl[i]);
		addTail(free_hdl_list,np);
	}
	n_handles=MAX_POINTERS;
}

static void more_handles(u_long n)
{
	pointer *pp;
	u_long i;

	pp=(pointer *)malloc((size_t) (n*sizeof(pointer)) );	/* size_t for pc */
	if( pp == NULL ){
		NWARN("couldn't malloc more pointers");
		return;
	}
	for(i=0;i<n;i++){
		Node *np;

		pp[i]=NULL;
		np = mk_node(&pp[i]);
		addTail(free_hdl_list,np);
	}
	n_handles += n;
}

Handle new_hdl(u_long size)
{
	Node *np;
	pointer *pp;

	if( free_hdl_list==NO_LIST )
		hdl_init();

	np=remHead(free_hdl_list);
	if( np == NO_NODE ){
		if( verbose ){
			sprintf(DEFAULT_MSG_STR,"%ld more handles",n_handles);
			NADVISE(DEFAULT_MSG_STR);
		}
		more_handles(n_handles);	/* double the size */
		np=remHead(free_hdl_list);
		if( np == NO_NODE ){
			NWARN("no more handles");
			return(NO_HANDLE);
		}

	}
	pp = (pointer*) np->n_data;
	*pp = (void *)malloc((size_t)size);	/* size_t for pc */

	addTail(used_hdl_list,np);
	return(pp);
}

void rls_hdl(Handle hdl)
{
	Node *np;

	np = remData(used_hdl_list,hdl);
	if( np == NO_NODE ){
		NWARN("node for handle not found");
		return;
	}
	free(*hdl);
	*hdl = NULL;
	addTail(free_hdl_list,np);
}

